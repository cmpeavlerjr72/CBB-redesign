#!/usr/bin/env python
"""
train_shot_block_v1.py -- the SHOT BLOCK sub-model bake-off.

Pre-registration: `docs/models/shot_block/experiments.md` section 1, written and
COMMITTED before this file was written to run anything. This script executes it
and APPENDS its results; it never edits what is already there.

TARGET: `P(blocked | the attempt MISSED, context)`. The population is missed
field-goal attempts (free throws excluded: n = 1 blocked missed FT in
2022-2025). The cascade placement -- after the make/miss draw, before the
rebound draw -- and the reason it is the placement that does not double count
`fg_make` (which BANS `blocked` as post-outcome and so already prices blocked
attempts inside its miss population) are section 1.2 of the pre-registration.

ROW SET. The rows are the missed-FGA rows of the rebound round-3 design
(`data/processed/models/rebound/round3/design_round3.parquet`), so this model's
prediction is a drop-in feed for that round's `B3`/`B3e` arms with no join at
serve time. `shooter_id` is joined from
`data/processed/models/fg_make/events_v2_shotshooter.parquet` (the audited
shooter key) on (game_id, period, seconds_remaining, off_team_id, score_diff),
dropping ambiguous keys on both sides; the match rate is MEASURED and reported,
and unmatched rows carry `shooter_known = 0` with a shooter feature of exactly
0.0 -- the league mean on a centred scale, never a fabricated level.

    .venv/Scripts/python.exe scripts/train_shot_block_v1.py
    .venv/Scripts/python.exe scripts/train_shot_block_v1.py --no-append
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_THREADS = os.environ.get("CBB_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _THREADS)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from cbb_sim.features import conference as CF  # noqa: E402
from cbb_sim.features import opponent_adjust as OA  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402

DESIGN = _ROOT / "data/processed/models/rebound/round3/design_round3.parquet"
SHOOTER_SRC = _ROOT / "data/processed/models/fg_make/events_v2_shotshooter.parquet"
OUT_DIR = _ROOT / "data/processed/models/shot_block"
DOC = _ROOT / "docs/models/shot_block/experiments.md"
SEASONS = [2022, 2023, 2024, 2025]
FOLDS = {"F1": {"train": [2022, 2023], "test": [2024]},
         "F2": {"train": [2022, 2023, 2024], "test": [2025]}}
SELECTION_FOLD = "F2"
SHOT_TYPES = ("rim", "jump2", "three")
SHOOTER_PRIOR_GRID = (0, 50, 100, 200, 400)
MIN_CELL_N = 300
N_JOBS = int(_THREADS)

LGBM_PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.06,
                   num_leaves=63, min_child_samples=400, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
                   verbose=-1, n_jobs=N_JOBS)

KA = ["miss_rim", "miss_jump2", "miss_three"]
KB = KA + ["def_block_c", "off_blocked_c", "shooter_blocked_c", "shooter_known",
           "site_home", "site_away"]
KC = KB + ["off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c",
           "period", "seconds_remaining", "score_diff", "in_bonus"]


def _rate(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


# ---------------------------------------------------------------------------
# 1. Design
# ---------------------------------------------------------------------------
def join_shooter(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    key = ["game_id", "period", "seconds_remaining", "off_team_id", "score_diff"]
    f = pd.read_parquet(SHOOTER_SRC, columns=key + ["made", "blocked", "shooter_id"])
    f = f[~f["made"].astype(bool)]
    for c in ("period", "seconds_remaining", "score_diff"):
        f[c] = f[c].astype("int64")
    f = f.drop_duplicates(key, keep=False)
    dd = d.copy()
    for c in ("period", "seconds_remaining", "score_diff"):
        dd[c] = dd[c].astype("int64")
    amb = float(dd.duplicated(key, keep=False).mean())
    dd = dd.merge(f[key + ["shooter_id", "blocked"]].rename(columns={"blocked": "_blk_chk"}),
                  on=key, how="left")
    matched = dd["shooter_id"].notna()
    agree = float((dd.loc[matched, "blocked"].astype(bool)
                   == dd.loc[matched, "_blk_chk"].astype(bool)).mean())
    meta = {"match_rate": round(float(matched.mean()), 5),
            "ambiguous_key_share_design": round(amb, 5),
            "blocked_agreement_on_matched": round(agree, 6),
            "n_rows": int(len(dd)), "source": str(SHOOTER_SRC)}
    assert agree > 0.9999, f"joined rows disagree on `blocked` ({agree}); the key is wrong"
    dd["shooter_known"] = matched.astype("float32")
    dd["shooter_id"] = dd["shooter_id"].fillna(-1).astype("int64")
    return dd.drop(columns=["_blk_chk"]), meta


def team_block_form(d: pd.DataFrame) -> pd.DataFrame:
    """As-of blocked share, league-centred, expanding within season and strictly
    before the game, on BOTH sides: what a defence blocks, and what an offence
    gets blocked on. Pooled over shot type -- the type interaction is carried by
    the miss dummies, and the (type, defence) cell rate is a separate arm."""
    b = d[["season", "game_id", "game_date", "off_team_id", "def_team_id", "blocked"]].copy()
    b["_blk"] = b["blocked"].astype("int32")
    b["_n"] = 1
    tg = b.groupby(["season", "game_id", "game_date", "off_team_id", "def_team_id"],
                   as_index=False)[["_blk", "_n"]].sum()
    tg = tg.sort_values(["season", "game_date", "game_id"], kind="stable").reset_index(drop=True)

    off = tg.sort_values(["season", "off_team_id", "game_date", "game_id"], kind="stable")
    oa = PM.expanding_asof(off, ["season", "off_team_id"], ["_blk", "_n"])
    oa.columns = [f"o_{c}" for c in oa.columns]
    off = pd.concat([off[["season", "game_id", "off_team_id", "def_team_id", "game_date"]], oa],
                    axis=1)
    dfd = tg.sort_values(["season", "def_team_id", "game_date", "game_id"], kind="stable")
    da = PM.expanding_asof(dfd, ["season", "def_team_id"], ["_blk", "_n"])
    da.columns = [f"d_{c}" for c in da.columns]
    dfd = pd.concat([dfd[["season", "game_id", "def_team_id"]], da], axis=1)

    day = tg.groupby(["season", "game_date"], as_index=False)[["_blk", "_n"]].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PM.expanding_asof(day, ["season"], ["_blk", "_n"])
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)

    form = off.merge(dfd, on=["season", "game_id", "def_team_id"], how="left")
    form = form.merge(day, on=["season", "game_date"], how="left")
    lgr = _rate(form["lg__blk"].to_numpy(), form["lg__n"].to_numpy())
    ov = _rate(form["o__blk"].to_numpy(), form["o__n"].to_numpy())
    dv = _rate(form["d__blk"].to_numpy(), form["d__n"].to_numpy())
    form["off_blocked_c"] = np.where(np.isnan(ov) | np.isnan(lgr), 0.0, ov - lgr).astype("float32")
    form["def_block_c"] = np.where(np.isnan(dv) | np.isnan(lgr), 0.0, dv - lgr).astype("float32")
    form["off_blk_D"] = form["o__n"].fillna(0.0).astype("float64")
    form["def_blk_D"] = form["d__n"].fillna(0.0).astype("float64")
    return form[["season", "game_id", "off_team_id", "def_team_id",
                 "off_blocked_c", "def_block_c", "off_blk_D", "def_blk_D"]], tg


def shooter_block_rates(d: pd.DataFrame, k: float) -> pd.Series:
    """Shooter's as-of blocked share, league-centred, EB-shrunk toward the
    league's own as-of rate with `k` pseudo-attempts. Unknown shooters get 0.0."""
    b = d[["season", "game_id", "game_date", "shooter_id", "blocked"]].copy()
    b = b[b["shooter_id"] >= 0]
    b["_blk"] = b["blocked"].astype("int32")
    b["_n"] = 1
    pg = b.groupby(["season", "shooter_id", "game_id", "game_date"],
                   as_index=False)[["_blk", "_n"]].sum()
    pg = pg.sort_values(["season", "shooter_id", "game_date", "game_id"],
                        kind="stable").reset_index(drop=True)
    a = PM.expanding_asof(pg, ["season", "shooter_id"], ["_blk", "_n"])
    pg = pd.concat([pg[["season", "shooter_id", "game_id", "game_date"]], a], axis=1)
    day = b.groupby(["season", "game_date"], as_index=False)[["_blk", "_n"]].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PM.expanding_asof(day, ["season"], ["_blk", "_n"])
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)
    pg = pg.merge(day, on=["season", "game_date"], how="left")
    lgr = _rate(pg["lg__blk"].to_numpy(), pg["lg__n"].to_numpy())
    lgr = np.where(np.isnan(lgr), 0.0, lgr)
    num = pg["_blk"].to_numpy() + k * lgr
    den = pg["_n"].to_numpy() + k
    val = np.where(den > 0, num / np.maximum(den, 1e-9), lgr) - lgr
    pg["shooter_blocked_c"] = val.astype("float32")
    out = d.merge(pg[["season", "shooter_id", "game_id", "shooter_blocked_c"]],
                  on=["season", "shooter_id", "game_id"], how="left")
    return out["shooter_blocked_c"].fillna(0.0).astype("float32")


def _mom_k(box: pd.DataFrame, team_col: str) -> float:
    """Method-of-moments empirical-Bayes `k = s2 / tau2` for the block rate, in
    units of missed attempts, from the COMPLETED seasons in `box` -- the
    estimator `scripts/train_possession_outcome_v4.fit_k` uses, transcribed."""
    g = box[[team_col, "season", "blk", "nmiss"]].rename(columns={team_col: "tid"})
    g = g[g["nmiss"] > 0]
    ts = g.groupby(["season", "tid"], as_index=False)[["blk", "nmiss"]].sum()
    ts["n_games"] = g.groupby(["season", "tid"]).size().to_numpy()
    ts["rate"] = _rate(ts["blk"].to_numpy(), ts["nmiss"].to_numpy())
    lgs = g.groupby("season", as_index=False)[["blk", "nmiss"]].sum()
    lgs["lg_rate"] = _rate(lgs["blk"].to_numpy(), lgs["nmiss"].to_numpy())
    ts = ts.merge(lgs[["season", "lg_rate"]], on="season", how="left")
    gg = g.merge(ts[["season", "tid", "rate", "n_games"]], on=["season", "tid"], how="left")
    gg["rate_g"] = _rate(gg["blk"].to_numpy(), gg["nmiss"].to_numpy())
    gg["sq"] = gg["nmiss"].to_numpy() * (gg["rate_g"].to_numpy() - gg["rate"].to_numpy()) ** 2
    per = gg[gg["n_games"] >= 2].groupby(["season", "tid"]).agg(ss=("sq", "sum"),
                                                                n=("sq", "size")).reset_index()
    s2 = float(per["ss"].sum() / max((per["n"] - 1).sum(), 1))
    w = ts["nmiss"].to_numpy().astype("float64")
    dev = ts["rate"].to_numpy() - ts["lg_rate"].to_numpy()
    mu = float(np.average(dev, weights=w))
    var_obs = float(np.average((dev - mu) ** 2, weights=w))
    tau2 = var_obs - float(np.average(s2 / np.maximum(w, 1e-9), weights=w))
    return float(min(max(s2 / max(tau2, 1e-9), 0.0), 1e6))


def build_design() -> tuple[pd.DataFrame, dict]:
    d = pd.read_parquet(DESIGN)
    d = d[d["miss_type"].isin(SHOT_TYPES)].reset_index(drop=True)
    d, jmeta = join_shooter(d)
    form, tg = team_block_form(d)
    d = d.merge(form, on=["season", "game_id", "off_team_id", "def_team_id"], how="left")
    for c in ("off_blocked_c", "def_block_c"):
        d[c] = d[c].fillna(0.0).astype("float32")
    for c in ("off_blk_D", "def_blk_D"):
        d[c] = d[c].fillna(0.0).astype("float64")

    # Decision 9a: opponent-adjusted defence block rate (arm K1_oa)
    box = tg.rename(columns={"off_team_id": "team_id", "def_team_id": "opp_id",
                             "_blk": "blk", "_n": "nmiss"})
    res = OA.adjust_team_form(box, {"blk": ("blk", "nmiss")}, {"blk": 1.0}, "one_pass")
    bb = box[["season", "game_id", "opp_id"]].copy()
    bb["corr_def"] = res.def_adj["blk"].to_numpy()
    bb = bb.groupby(["season", "game_id", "opp_id"], as_index=False)["corr_def"].mean()
    bb = bb.rename(columns={"opp_id": "def_team_id"})
    d = d.merge(bb, on=["season", "game_id", "def_team_id"], how="left")
    d["corr_def"] = d["corr_def"].fillna(0.0)
    d["def_block_oa_c"] = (d["def_block_c"].to_numpy() - d["corr_def"].to_numpy()).astype("float32")

    # prior-season carry of the two team block rates (arm K3_prior)
    ts = box.groupby(["season", "team_id"], as_index=False)[["blk", "nmiss"]].sum()
    lgs = box.groupby("season", as_index=False)[["blk", "nmiss"]].sum()
    lgs["lg"] = _rate(lgs["blk"].to_numpy(), lgs["nmiss"].to_numpy())
    ts = ts.merge(lgs[["season", "lg"]], on="season", how="left")
    ts["priorc"] = (_rate(ts["blk"].to_numpy(), ts["nmiss"].to_numpy()) - ts["lg"]).astype("float32")
    ts["Dprev"] = ts["nmiss"].astype("float64")
    ts["season"] = ts["season"] + 1
    off_p = ts.rename(columns={"team_id": "off_team_id", "priorc": "off_priorc_blk",
                               "Dprev": "off_Dprev_blk"})[["season", "off_team_id",
                                                           "off_priorc_blk", "off_Dprev_blk"]]
    tsd = box.groupby(["season", "opp_id"], as_index=False)[["blk", "nmiss"]].sum()
    tsd = tsd.merge(lgs[["season", "lg"]], on="season", how="left")
    tsd["priorc"] = (_rate(tsd["blk"].to_numpy(), tsd["nmiss"].to_numpy())
                     - tsd["lg"]).astype("float32")
    tsd["Dprev"] = tsd["nmiss"].astype("float64")
    tsd["season"] = tsd["season"] + 1
    def_p = tsd.rename(columns={"opp_id": "def_team_id", "priorc": "def_priorc_blk",
                                "Dprev": "def_Dprev_blk"})[["season", "def_team_id",
                                                            "def_priorc_blk", "def_Dprev_blk"]]
    d = d.merge(off_p, on=["season", "off_team_id"], how="left")
    d = d.merge(def_p, on=["season", "def_team_id"], how="left")
    for c in ("off_priorc_blk", "def_priorc_blk", "off_Dprev_blk", "def_Dprev_blk"):
        d[c] = d[c].fillna(0.0)
    # EB weight: method-of-moments k = s2/tau2 per side, from COMPLETED seasons
    # only -- the same estimator `train_possession_outcome_v4.fit_k` uses, never
    # a chosen constant.
    kk = {}
    for side, D, pri, team_col in (("off", "off_blk_D", "off_priorc_blk", "team_id"),
                                   ("def", "def_blk_D", "def_priorc_blk", "opp_id")):
        kk[side] = _mom_k(box, team_col)
        w = d[D].to_numpy() / np.maximum(d[D].to_numpy() + kk[side], 1e-9)
        base = d["off_blocked_c"] if side == "off" else d["def_block_c"]
        d[f"{side}_block_carry_c"] = (w * base.to_numpy()
                                      + (1 - w) * d[pri].to_numpy()).astype("float32")

    d["y"] = d["blocked"].astype("int8")
    d["month"] = pd.to_datetime(d["game_date"]).dt.month.astype("int16")
    jmeta["k_block_carry"] = kk
    return d, jmeta


# ---------------------------------------------------------------------------
# 2. Arms
# ---------------------------------------------------------------------------
def fit_predict(arm: str, tr: pd.DataFrame, te: pd.DataFrame, seed: int = 0
                ) -> tuple[np.ndarray, object, list[str]]:
    if arm == "K0":
        last = int(tr["season"].max())
        t = tr[tr["season"] == last]
        tab = t.groupby("miss_type")["y"].mean().to_dict()
        p = te["miss_type"].map(tab).to_numpy(dtype="float64")
        return np.clip(p, 1e-6, 1 - 1e-6), tab, ["miss_type"]
    if arm == "K0_oracle":
        tab = te.groupby("miss_type")["y"].mean().to_dict()
        p = te["miss_type"].map(tab).to_numpy(dtype="float64")
        return np.clip(p, 1e-6, 1 - 1e-6), tab, ["miss_type"]
    if arm in ("K1", "K1_oa"):
        col = "def_block_c" if arm == "K1" else "def_block_oa_c"
        last_lg = float(tr[tr["season"] == int(tr["season"].max())]["y"].mean())
        by_t = tr[tr["season"] == int(tr["season"].max())].groupby("miss_type")["y"].mean().to_dict()
        base = te["miss_type"].map(by_t).to_numpy(dtype="float64")
        # the defence's centred pooled deviation, scaled into the shot type's own
        # rate by its share of the league rate; a cell rate, not a fit
        scale = base / max(last_lg, 1e-9)
        p = base + scale * te[col].to_numpy(dtype="float64")
        return np.clip(p, 1e-6, 1 - 1e-6), {"col": col, "by_type": by_t}, [col]
    feats = {"K2": KC, "K3": KC, "K3_conf": KC + ["is_conf_game"],
             "K3_prior": [c if c not in ("def_block_c", "off_blocked_c")
                          else {"def_block_c": "def_block_carry_c",
                                "off_blocked_c": "off_block_carry_c"}[c] for c in KC]}[arm]
    X = np.ascontiguousarray(tr[feats].to_numpy(dtype="float32"))
    Xt = np.ascontiguousarray(te[feats].to_numpy(dtype="float32"))
    y = tr["y"].to_numpy()
    if arm == "K2":
        from sklearn.linear_model import LogisticRegression
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-8] = 1.0
        clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs",
                                 random_state=seed).fit((X - mu) / sd, y)
        p = clf.predict_proba((Xt - mu) / sd)[:, 1]
        return p, {"mu": mu, "sd": sd, "clf": clf}, feats
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(random_state=seed, **LGBM_PARAMS).fit(X, y)
    return clf.predict_proba(Xt)[:, 1], clf, feats


# ---------------------------------------------------------------------------
# 3. One blind grader
# ---------------------------------------------------------------------------
def _ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _decile_gap(y, p, n_bins: int = 10) -> float:
    q = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(a=("y", "mean"), e=("p", "mean"))
    return float((g["e"] - g["a"]).abs().max() * 100.0)


def _quintile_slope(prior: np.ndarray, y: np.ndarray, p: np.ndarray, n_q: int = 5) -> dict:
    ok = np.isfinite(prior)
    if ok.sum() < n_q * MIN_CELL_N:
        return {"underpowered": True, "n": int(ok.sum())}
    q = pd.qcut(pd.Series(prior[ok]).rank(method="first"), n_q, labels=False)
    g = pd.DataFrame({"y": y[ok], "p": p[ok], "q": q}).groupby("q").agg(
        a=("y", "mean"), e=("p", "mean"), n=("y", "size"))
    da, de = np.diff(g["a"].to_numpy()), np.diff(g["e"].to_numpy())
    sign = 1.0 if g["a"].to_numpy()[-1] >= g["a"].to_numpy()[0] else -1.0
    span_a = float(g["a"].to_numpy()[-1] - g["a"].to_numpy()[0])
    span_e = float(g["e"].to_numpy()[-1] - g["e"].to_numpy()[0])
    return {"underpowered": False, "n": int(ok.sum()),
            "slope_ratio": round(span_e / span_a, 4) if abs(span_a) > 1e-9 else None,
            "monotone_steps": int(((de * sign) > 0).sum()), "steps": 4,
            "actual_by_q": [round(v, 5) for v in g["a"].tolist()],
            "pred_by_q": [round(v, 5) for v in g["e"].tolist()],
            "n_by_q": [int(v) for v in g["n"].tolist()]}


def _seg_level(te: pd.DataFrame, y: np.ndarray, p: np.ndarray, col: str) -> dict:
    out = {}
    for k, idx in te.groupby(col).indices.items():
        n = len(idx)
        cell = {"n": int(n)}
        if n < MIN_CELL_N:
            cell["UNDERPOWERED"] = True
        else:
            cell["level_pp"] = round(float(p[idx].mean() - y[idx].mean()) * 100, 4)
            cell["log_loss"] = round(_ll(y[idx], p[idx]), 6)
        out[str(k)] = cell
    return out


def grade(te: pd.DataFrame, p: np.ndarray) -> dict:
    y = te["y"].to_numpy().astype("float64")
    lvl = float(p.mean() - y.mean()) * 100
    out = {"n": int(len(te)), "log_loss": round(_ll(y, p), 6),
           "brier": round(float(((p - y) ** 2).mean()), 6),
           "actual_rate": round(float(y.mean()), 6), "pred_rate": round(float(p.mean()), 6),
           "level_pp": round(lvl, 4),
           "calib_worst_decile_gap_pp": round(_decile_gap(y, p), 4)}
    out["calib_pass"] = bool(out["calib_worst_decile_gap_pp"] <= 2.0)
    out["by_shot_type"] = _seg_level(te, y, p, "miss_type")
    out["level_pass"] = bool(abs(lvl) <= 0.25 and all(
        abs(v.get("level_pp", 0.0)) <= 0.50 for v in out["by_shot_type"].values()
        if not v.get("UNDERPOWERED")))
    out["slope_def_prior"] = _quintile_slope(te["def_priorc_blk"].to_numpy(), y, p)
    out["slope_off_prior"] = _quintile_slope(te["off_priorc_blk"].to_numpy(), y, p)
    sd = out["slope_def_prior"]
    out["resp_pass"] = bool((not sd.get("underpowered"))
                            and sd.get("monotone_steps", 0) >= 3
                            and (sd.get("slope_ratio") or 0) > 0)
    out["by_month"] = _seg_level(te, y, p, "month")
    site = np.where(te["site_home"].to_numpy() > 0, "home",
                    np.where(te["site_away"].to_numpy() > 0, "away", "neutral"))
    out["by_site"] = _seg_level(te.assign(_s=site), y, p, "_s")
    out["by_conf"] = _seg_level(te.assign(_c=np.where(te["is_conf_game"].to_numpy() > 0,
                                                      "conf", "nonconf")), y, p, "_c")
    out["by_period"] = _seg_level(te.assign(_p=te["period"].astype(int).clip(upper=5)), y, p, "_p")
    gm = ((te["period"].to_numpy().clip(max=2) - 1) * 1200
          + (1200 - te["seconds_remaining"].to_numpy())) / 300.0
    out["by_game_minute_bucket"] = _seg_level(te.assign(_g=np.clip(gm, 0, 7).astype(int)),
                                              y, p, "_g")
    early = np.where(te["month"].isin([11, 12]), "Nov-Dec", "Jan-Apr")
    out["by_season_half"] = _seg_level(te.assign(_e=early), y, p, "_e")
    return out


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
ARMS = ("K0", "K1", "K1_oa", "K2", "K3", "K3_conf", "K3_prior")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-append", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="shot_block bake-off")

    d, jmeta = build_design()
    print(f"[{time.time()-t0:.0f}s] design {d.shape}; shooter match "
          f"{jmeta['match_rate']}", flush=True)

    # shooter shrinkage strength: FITTED on the selection fold's TRAIN slice
    tr2 = d[d["season"].isin(FOLDS["F2"]["train"])]
    grid_ll = {}
    from sklearn.linear_model import LogisticRegression
    for k in SHOOTER_PRIOR_GRID:
        col = shooter_block_rates(d, float(k))
        x = col[d["season"].isin(FOLDS["F2"]["train"])].to_numpy().reshape(-1, 1).astype("float64")
        mu, sd = x.mean(), max(x.std(), 1e-8)
        clf = LogisticRegression(C=1.0, max_iter=200).fit((x - mu) / sd, tr2["y"].to_numpy())
        grid_ll[k] = round(_ll(tr2["y"].to_numpy().astype(float),
                               clf.predict_proba((x - mu) / sd)[:, 1]), 7)
    k_star = min(grid_ll, key=grid_ll.get)
    d["shooter_blocked_c"] = shooter_block_rates(d, float(k_star))
    print(f"[{time.time()-t0:.0f}s] shooter shrinkage fitted k={k_star} {grid_ll}", flush=True)

    results: dict[str, dict] = {}
    for fold, spec in FOLDS.items():
        tr = d[d["season"].isin(spec["train"])]
        te = d[d["season"].isin(spec["test"])]
        assert_not_sealed(spec["train"], context=f"{fold} train")
        assert_not_sealed(spec["test"], context=f"{fold} test")
        for arm in ARMS + (("K0_oracle",) if fold == SELECTION_FOLD else ()):
            t1 = time.time()
            p, model, feats = fit_predict(arm, tr, te)
            g = grade(te, p)
            g["fit_s"] = round(time.time() - t1, 1)
            g["n_features"] = len(feats)
            results[f"{fold}|{arm}"] = g
            if fold == SELECTION_FOLD and arm in ("K3", "K3_conf", "K3_prior", "K2"):
                joblib.dump({"arm": arm, "features": feats, "model": model, "fold": fold,
                             "note": "shot_block bake-off, experiments.md section 1"},
                            OUT_DIR / f"{arm}_{fold}.joblib")
            print(f"[{time.time()-t0:.0f}s] {fold}|{arm}: ll={g['log_loss']} "
                  f"lvl={g['level_pp']}pp calib={g['calib_worst_decile_gap_pp']} "
                  f"slope={g['slope_def_prior'].get('slope_ratio')} "
                  f"mono={g['slope_def_prior'].get('monotone_steps')} {g['fit_s']}s", flush=True)

    # noise floor: spec-identical second-seed retrain of the leading tree arm and K0
    tr2 = d[d["season"].isin(FOLDS["F2"]["train"])]
    te2 = d[d["season"].isin(FOLDS["F2"]["test"])]
    tree_arms = [a for a in ARMS if a.startswith("K3")]
    lead = min(tree_arms, key=lambda a: results[f"F2|{a}"]["log_loss"])
    p1, _, _ = fit_predict(lead, tr2, te2, seed=1)
    g1 = grade(te2, p1)
    floor = abs(g1["log_loss"] - results[f"F2|{lead}"]["log_loss"])
    lvl_floor = abs(g1["level_pp"] - results[f"F2|{lead}"]["level_pp"])
    print(f"[{time.time()-t0:.0f}s] floor from {lead} seed1: {floor:.6f} "
          f"(level {lvl_floor:.4f} pp)", flush=True)

    # the winner, by the pre-registered rule
    ref = results[f"F2|K0"]["log_loss"]
    elig = [a for a in ARMS if a != "K0"
            and (ref - results[f"F2|{a}"]["log_loss"]) > floor
            and results[f"F2|{a}"]["calib_pass"] and results[f"F2|{a}"]["resp_pass"]
            and results[f"F2|{a}"]["level_pass"]]
    order = {a: i for i, a in enumerate(("K0", "K1", "K1_oa", "K2", "K3", "K3_conf", "K3_prior"))}
    winner = None
    if elig:
        best = min(elig, key=lambda a: results[f"F2|{a}"]["log_loss"])
        tied = [a for a in elig
                if results[f"F2|{a}"]["log_loss"] - results[f"F2|{best}"]["log_loss"] <= floor]
        winner = min(tied, key=lambda a: order[a])
        lin = [a for a in elig if a == "K2"]
        if winner.startswith("K3") and lin:
            if (results[f"F2|{lin[0]}"]["log_loss"]
                    - results[f"F2|{winner}"]["log_loss"]) <= floor:
                winner = lin[0]

    report = {"created_at": pd.Timestamp.now("UTC").isoformat(), "join": jmeta,
              "shooter_k_grid_train_ll": grid_ll, "shooter_k_fitted": int(k_star),
              "noise_floor_log_loss": round(float(floor), 6),
              "noise_floor_level_pp": round(float(lvl_floor), 4),
              "floor_source_arm": lead, "cells": results,
              "eligible": elig, "winner": winner,
              "runtime_min": round((time.time() - t0) / 60, 1)}
    (OUT_DIR / "run_report_v1.json").write_text(json.dumps(report, indent=1, default=str),
                                                encoding="utf-8")
    sec = render(report)
    if args.no_append:
        print(sec)
    else:
        with DOC.open("a", encoding="utf-8") as fh:
            fh.write("\n" + sec)
        print(f"appended to {DOC}")
    print(f"done in {report['runtime_min']} min; winner={winner}")
    return 0


def _tbl(rows, cols):
    return "\n".join(["| " + " | ".join(cols) + " |",
                      "|" + "|".join("---" for _ in cols) + "|",
                      *["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]])


def render(rep: dict) -> str:
    L = [f"## 2. Results (run {rep['created_at']}, `scripts/train_shot_block_v1.py`)\n"]
    L.append(f"Row set: missed FGA of the rebound round-3 design. Shooter join match rate "
             f"**{rep['join']['match_rate']}**, `blocked` agreement on matched rows "
             f"{rep['join']['blocked_agreement_on_matched']} (the key is verified, not assumed); "
             f"unmatched rows carry `shooter_known = 0` and a shooter feature of exactly 0.0.\n")
    L.append(f"Shooter EB pseudo-count FITTED on the F2 train slice over "
             f"{list(SHOOTER_PRIOR_GRID)}: **{rep['shooter_k_fitted']}** "
             f"(train log loss {rep['shooter_k_grid_train_ll']}).\n")
    for fold in ("F1", "F2"):
        rows = []
        for key, g in rep["cells"].items():
            f, a = key.split("|")
            if f != fold:
                continue
            sd = g["slope_def_prior"]
            rows.append({"arm": a, "n": g["n"], "log_loss": g["log_loss"], "brier": g["brier"],
                         "pred_rate": g["pred_rate"], "actual_rate": g["actual_rate"],
                         "level_pp": g["level_pp"],
                         "calib_gap_pp": g["calib_worst_decile_gap_pp"],
                         "calib": "PASS" if g["calib_pass"] else "FAIL",
                         "level_gate": "PASS" if g["level_pass"] else "FAIL",
                         "slope_def_q": sd.get("slope_ratio"),
                         "mono": sd.get("monotone_steps"),
                         "resp": "PASS" if g["resp_pass"] else "FAIL",
                         "fit_s": g["fit_s"]})
        rows.sort(key=lambda r: r["log_loss"])
        L.append(f"**{fold}** -- train {FOLDS[fold]['train']} test {FOLDS[fold]['test']}"
                 + (" (SELECTION)\n" if fold == SELECTION_FOLD else "\n"))
        L.append(_tbl(rows, ["arm", "n", "log_loss", "brier", "pred_rate", "actual_rate",
                             "level_pp", "calib_gap_pp", "calib", "level_gate",
                             "slope_def_q", "mono", "resp", "fit_s"]))
        L.append("")
    L.append(f"Noise floor: spec-identical second-seed retrain of `{rep['floor_source_arm']}` on "
             f"F2 -- log-loss spread **{rep['noise_floor_log_loss']}**, level spread "
             f"{rep['noise_floor_level_pp']} pp.\n")
    L.append(f"Eligible by rule 1.9.1 (beats `K0` beyond the floor AND passes all three gates): "
             f"{rep['eligible'] or 'NONE'}.\n")
    none_txt = ("NONE -- no arm clears the rule; the engine keeps feeding "
                "blocked_f = 0.0 and G4 channel 3 stays open")
    L.append(f"**WINNER: {rep['winner'] or none_txt}**\n")
    L.append("### 2.1 Segments (every arm, both folds)\n")
    L.append("Full per-arm segment tables (shot type, month, site, conference, period, "
             "game-minute bucket, season half, and both prior-quintile slopes with their "
             "per-quintile actual/predicted vectors) are in "
             "`data/processed/models/shot_block/run_report_v1.json`; cells below "
             f"n = {MIN_CELL_N} are marked `UNDERPOWERED` there and are excluded from every "
             "pass/fail above.\n")
    for key in ("F2|K0", "F2|K1", "F2|K3"):
        g = rep["cells"].get(key)
        if not g:
            continue
        rows = [{"shot type": k, "n": v["n"],
                 "level_pp": v.get("level_pp", "UNDERPOWERED"),
                 "log_loss": v.get("log_loss", "")} for k, v in g["by_shot_type"].items()]
        L.append(f"`{key}` by shot type:\n")
        L.append(_tbl(rows, ["shot type", "n", "level_pp", "log_loss"]))
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
