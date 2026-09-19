#!/usr/bin/env python
"""
train_foul_accrual_v1.py -- Block F of possession-outcome round 6: the
team-foul accrual law that sets the bonus state.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 13,
as amended by section 15 (AMENDMENT A, commit BEFORE any fitting).

    .venv/Scripts/python.exe scripts/train_foul_accrual_v1.py --seed 0
    .venv/Scripts/python.exe scripts/train_foul_accrual_v1.py --seed 7   # noise floor

Reads `data/processed/models/possession_outcome/round6/foul_accrual_poss_v1.parquet`
(`scripts/build_foul_accrual_design_v1.py`). Writes, per fold and seed, ONE
parquet holding the test-slice keys plus one probability column per arm, so
`scripts/grade_foul_accrual_v1.py` scores every arm through one code path.

TARGET (15.1): `y = def_silent >= 1`, the non-shooting personal foul charged to
the DEFENCE during the possession that awards no free-throw trip -- exactly the
event `loop.py` draws once per possession from the constant
`silent_foul_per_possession`. The offence-side channel (`off_silent`) is fitted
under the same arms and written alongside; it is the channel the engine has no
mechanism for at all.

FIT / PRIMARY MASK (13.5 breakdown 8, 15.6 v): `period <= 2 and
seconds_remaining > 120`. The final 2:00 of regulation and overtime are written
to the prediction file but are NEVER fitted on and are graded as held-out
segments only.

NO ARM READS A POST-OUTCOME COLUMN (L27): `duration_s`, `is_transition` and the
possession's own terminal event are absent from every feature list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

ROUND_DIR = Path("data/processed/models/possession_outcome/round6")
DESIGN = ROUND_DIR / "foul_accrual_poss_v1.parquet"
FOLDS = {"F1": {"train": [2022, 2023], "test": [2024]},
         "F2": {"train": [2022, 2023, 2024], "test": [2025]}}
SERVED_CONSTANT = 0.123346

#: clock buckets for the cell arms, in seconds remaining in the period
CLOCK_EDGES = [0, 120, 300, 600, 900, 1201]
MARGIN_EDGES = [-999, -15, -7, -3, 0, 3, 7, 15, 999]


# ---------------------------------------------------------------------------
# as-of team foul rates, league-centred, strictly before the game
# ---------------------------------------------------------------------------
def team_game_fouls(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, game_id, team): fouls committed, fouls drawn, and
    the possession counts they are rates over."""
    d = df[["season", "game_id", "game_date", "offense_team_id", "defense_team_id",
            "def_silent", "def_trip", "off_silent", "off_trip"]]
    dfn = d.groupby(["season", "game_id", "game_date", "defense_team_id"], sort=False).agg(
        def_fouls=("def_silent", lambda s: 0), n_def_poss=("def_silent", "size"),
        def_silent=("def_silent", "sum"), def_trip=("def_trip", "sum")).reset_index()
    dfn["def_fouls"] = dfn["def_silent"] + dfn["def_trip"]
    dfn = dfn.rename(columns={"defense_team_id": "team_id"})
    off = d.groupby(["season", "game_id", "game_date", "offense_team_id"], sort=False).agg(
        n_off_poss=("off_silent", "size"), off_silent=("off_silent", "sum"),
        off_trip=("off_trip", "sum"), drawn=("def_silent", "sum"),
        drawn_trip=("def_trip", "sum")).reset_index()
    off["off_fouls"] = off["off_silent"] + off["off_trip"]
    off["fouls_drawn"] = off["drawn"] + off["drawn_trip"]
    off = off.rename(columns={"offense_team_id": "team_id"})
    tg = dfn.merge(off, on=["season", "game_id", "game_date", "team_id"], how="outer")
    for c in ("def_fouls", "n_def_poss", "off_fouls", "n_off_poss", "fouls_drawn"):
        tg[c] = tg[c].fillna(0.0)
    tg["fouls_committed"] = tg["def_fouls"] + tg["off_fouls"]
    tg["n_poss"] = tg["n_def_poss"] + tg["n_off_poss"]
    return tg.sort_values(["season", "team_id", "game_date", "game_id"]).reset_index(drop=True)


def _asof_rate(tg: pd.DataFrame, num: str, den: str, name: str,
               carry_prior: bool = False, prior_games: float = 5.0) -> pd.DataFrame:
    """Expanding, strictly-before team rate, minus the strictly-before league
    mean of the same quantity. A team with no prior games gets exactly 0.0 (the
    league mean) unless `carry_prior`, in which case the prior season's final
    rate is carried in with weight `prior_games` team-games."""
    g = tg.groupby(["season", "team_id"], sort=False)
    cn = g[num].cumsum() - tg[num]
    cd = g[den].cumsum() - tg[den]
    n_prior = g.cumcount()
    if carry_prior:
        fin = tg.groupby(["season", "team_id"], sort=False)[[num, den]].sum().reset_index()
        fin["prior_rate"] = fin[num] / fin[den].replace(0, np.nan)
        fin["season"] = fin["season"] + 1
        pr = tg[["season", "team_id"]].merge(
            fin[["season", "team_id", "prior_rate"]], on=["season", "team_id"], how="left")
        prior = pr["prior_rate"].to_numpy()
        lg_prior = np.nanmean(prior) if np.isfinite(prior).any() else 0.0
        prior = np.where(np.isfinite(prior), prior, lg_prior)
        # pseudo-count: `prior_games` team-games' worth of the prior rate
        pseudo_d = prior_games * tg[den].mean()
        cn = cn + prior * pseudo_d
        cd = cd + pseudo_d
    rate = (cn / cd.replace(0, np.nan)).to_numpy()
    # strictly-before LEAGUE mean of the same rate, by date
    day = tg.groupby(["season", "game_date"], sort=False)[[num, den]].sum().reset_index()
    day = day.sort_values(["season", "game_date"])
    day["cn"] = day.groupby("season")[num].cumsum() - day[num]
    day["cd"] = day.groupby("season")[den].cumsum() - day[den]
    day["lg"] = day["cn"] / day["cd"].replace(0, np.nan)
    lg = tg[["season", "game_date"]].merge(day[["season", "game_date", "lg"]],
                                           on=["season", "game_date"], how="left")["lg"].to_numpy()
    out = tg[["season", "game_id", "team_id"]].copy()
    v = rate - lg
    out[name] = np.where(np.isfinite(v), v, 0.0) * 100.0
    out[f"{name}_nprior"] = n_prior.to_numpy()
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    tg = team_game_fouls(df)
    feats = [
        _asof_rate(tg, "def_fouls", "n_def_poss", "def_foul_c"),
        _asof_rate(tg, "fouls_drawn", "n_off_poss", "off_drawn_c"),
        _asof_rate(tg, "def_fouls", "n_def_poss", "def_foul_pc", carry_prior=True),
        _asof_rate(tg, "fouls_drawn", "n_off_poss", "off_drawn_pc", carry_prior=True),
    ]
    f = feats[0]
    for k in feats[1:]:
        f = f.merge(k, on=["season", "game_id", "team_id"], how="outer")
    dcols = ["def_foul_c", "def_foul_pc", "def_foul_c_nprior"]
    ocols = ["off_drawn_c", "off_drawn_pc"]
    df = df.merge(f[["season", "game_id", "team_id"] + dcols].rename(
        columns={"team_id": "defense_team_id"}), on=["season", "game_id", "defense_team_id"],
        how="left")
    df = df.merge(f[["season", "game_id", "team_id"] + ocols].rename(
        columns={"team_id": "offense_team_id"}), on=["season", "game_id", "offense_team_id"],
        how="left")
    # Decision 9a: opponent adjustment -- the defence's raw centred foul rate
    # minus the average centred DRAWN rate of the offences it has faced so far.
    opp = f[["season", "game_id", "team_id", "off_drawn_c"]].rename(
        columns={"team_id": "defense_team_id", "off_drawn_c": "_dummy"})
    df = df.merge(opp, on=["season", "game_id", "defense_team_id"], how="left")
    df["def_foul_oadj"] = df["def_foul_c"] - 0.5 * df["off_drawn_c"].fillna(0.0)
    df["off_drawn_oadj"] = df["off_drawn_c"] - 0.5 * df["def_foul_c"].fillna(0.0)
    for c in dcols + ocols + ["def_foul_oadj", "off_drawn_oadj"]:
        df[c] = df[c].astype("float32").fillna(0.0)
    # Decision 9b: conference-game flag
    try:
        from cbb_sim.features.conference import build_conference_flags
        cf = build_conference_flags(sorted(df["season"].unique().tolist()))
        df = df.merge(cf[["game_id", "is_conf_game"]], on="game_id", how="left")
        df["is_conf_game"] = df["is_conf_game"].fillna(False).astype("float32")
    except Exception as exc:                                         # noqa: BLE001
        print(f"[warn] conference flags unavailable ({exc}); D9b/D9c will be SKIPPED")
        df["is_conf_game"] = np.nan
    df["sec_rem"] = df["start_clock"].astype("float32")
    df["margin"] = df["start_score_diff"].astype("float32")
    df["abs_margin"] = df["margin"].abs()
    df["clock_cell"] = pd.cut(df["sec_rem"], CLOCK_EDGES, right=False, labels=False)
    df["margin_cell"] = pd.cut(df["margin"], MARGIN_EDGES, right=False, labels=False)
    df["site_cell"] = (df["site_home"] * 2 + df["site_away"]).astype("int8")
    df["half"] = np.where(df["period"] <= 1, 0, 1)
    return df


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
STATE = ["period", "sec_rem", "margin", "abs_margin", "game_seconds",
         "def_team_fouls", "off_team_fouls", "site_home", "site_away", "is_ot"]
F3 = STATE
F3B = STATE + ["off_in_bonus", "def_in_bonus", "off_in_double_bonus"]
F3C = F3B + ["def_foul_c", "off_drawn_c"]
F3D = F3B + ["def_foul_pc", "off_drawn_pc"]
D9A = F3B + ["def_foul_oadj", "off_drawn_oadj"]
D9B = F3C + ["is_conf_game"]
H1F = F3C + ["x_home_sec", "x_away_sec", "x_home_fouls", "x_away_fouls"]

ARMS = ["F0", "F0_served", "F1", "F2", "F2m", "F3", "F3b", "F3c", "F3d",
        "F5", "H1", "D9a", "D9b", "D9c"]


def _cells(tr: pd.DataFrame, te: pd.DataFrame, keys: list[str], y: str,
           prior: float) -> np.ndarray:
    """Laplace-smoothed cell means (smoothing toward the pooled train rate with
    a 200-row pseudo-count, fixed before the run, not tuned)."""
    k = 200.0
    g = tr.groupby(keys, observed=True)[y].agg(["sum", "size"])
    g["p"] = (g["sum"] + k * prior) / (g["size"] + k)
    idx = pd.MultiIndex.from_frame(te[keys]) if len(keys) > 1 else pd.Index(te[keys[0]])
    p = g["p"].reindex(idx).to_numpy()
    return np.where(np.isfinite(p), p, prior)


def _glm(tr, te, feats, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(tr[feats].to_numpy(dtype="float64"))
    m = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
    m.fit(X, tr[y].to_numpy())
    return m.predict_proba(sc.transform(te[feats].to_numpy(dtype="float64")))[:, 1]


def _gbm(tr, te, feats, y, seed):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05,
                           num_leaves=31, min_child_samples=500, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                           n_jobs=1, verbose=-1)
    m.fit(tr[feats].to_numpy(dtype="float32"), tr[y].to_numpy())
    return m.predict_proba(te[feats].to_numpy(dtype="float32"))[:, 1]


def run_fold(df: pd.DataFrame, fold: str, seed: int, y: str) -> pd.DataFrame:
    spec = FOLDS[fold]
    assert_not_sealed(spec["train"], context=f"{fold} train seasons")
    assert_not_sealed(spec["test"], context=f"{fold} test seasons")
    tr_all = df[df["season"].isin(spec["train"])]
    te_all = df[df["season"].isin(spec["test"])].copy()
    tr = tr_all[tr_all["in_fit_window"] == 1].copy()     # FIT MASK
    prior = float(tr[y].mean())
    out = te_all[["game_id", "season", "period", "poss_index", "offense_team_id",
                  "defense_team_id", "in_fit_window", "game_minute", "sec_rem",
                  "site_home", "site_away", "is_conf_game", "days_since_start",
                  "def_team_fouls", "off_in_bonus", y]].copy()
    te = te_all
    t0 = time.time()
    out["F0"] = prior
    out["F0_served"] = SERVED_CONSTANT
    out["F1"] = _cells(tr, te, ["half"], y, prior)
    out["F2"] = _cells(tr, te, ["period", "clock_cell", "site_cell"], y, prior)
    out["F2m"] = _cells(tr, te, ["period", "clock_cell", "site_cell", "margin_cell"], y, prior)
    for name, feats in (("F3", F3), ("F3b", F3B), ("F3c", F3C), ("F3d", F3D), ("D9a", D9A)):
        out[name] = _glm(tr, te, feats, y, seed)
    for c, base in (("x_home_sec", "site_home"), ("x_away_sec", "site_away")):
        tr[c] = tr[base] * tr["game_seconds"] / 100.0
        te[c] = te[base] * te["game_seconds"] / 100.0
    for c, base in (("x_home_fouls", "site_home"), ("x_away_fouls", "site_away")):
        tr[c] = tr[base] * tr["def_team_fouls"]
        te[c] = te[base] * te["def_team_fouls"]
    out["H1"] = _glm(tr, te, H1F, y, seed)
    out["F5"] = _gbm(tr, te, F3C, y, seed)
    if tr["is_conf_game"].notna().all():
        out["D9b"] = _glm(tr, te, D9B, y, seed)
        # D9c: conference-ALIGNED fit -- separate models for conference and
        # non-conference rows (the cheap proxy for a conference-aligned refit
        # cadence; declared as such in the results section).
        p = np.full(len(te), np.nan)
        for flag in (0.0, 1.0):
            m_tr = tr[tr["is_conf_game"] == flag]
            m_te = te["is_conf_game"].to_numpy() == flag
            if len(m_tr) > 5000 and m_te.any():
                p[m_te] = _glm(m_tr, te[m_te], F3C, y, seed)
        out["D9c"] = np.where(np.isfinite(p), p, out["F3c"].to_numpy())
    print(f"  {fold} seed {seed} target {y}: n_train {len(tr):,} n_test {len(te):,} "
          f"base {prior:.6f} in {time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="both", choices=["def", "off", "both"])
    a = ap.parse_args()
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DESIGN)
    df = build_features(df)
    df["y_def"] = (df["def_silent"] >= 1).astype("int8")
    df["y_off"] = (df["off_silent"] >= 1).astype("int8")
    targets = ["y_def"] if a.target == "def" else ["y_off"] if a.target == "off" else ["y_def", "y_off"]
    meta = {"seed": a.seed, "n_rows": int(len(df)), "arms": ARMS,
            "served_constant": SERVED_CONSTANT,
            "mean_def_silent_count": float(df["def_silent"].mean()),
            "mean_y_def": float(df["y_def"].mean()),
            "mean_off_silent_count": float(df["off_silent"].mean()),
            "mean_y_off": float(df["y_off"].mean())}
    for y in targets:
        for fold in ("F1", "F2"):
            out = run_fold(df, fold, a.seed, y)
            out.to_parquet(ROUND_DIR / f"preds_{y}_{fold}_seed{a.seed}.parquet", index=False)
    (ROUND_DIR / f"train_meta_seed{a.seed}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
