"""build_late_game_design_v1.py -- the L7 late-game EVENT-half design.

Lane: late-game regime, round 1 (2026-09-18).  Pre-registration:
`docs/models/late_game/experiments.md` sections 1 and 2 (the amendment).
Feature provenance: `docs/models/late_game/features.md`.

WHAT IT BUILDS.  The SERVED possession-outcome round-2 design
(`data/processed/models/possession_outcome/round2/design.parquet` -- possessions
v2, `style_source=first_chance`, `require_pbp_complete=True`, the exact frame
the served `round2_s1` arms were fitted on) plus the late-game state columns of
`features.md` section 1.  Nothing in the round-2 design is modified; the new
columns are appended, so bundle `L0_reference` is bit-identical to the served
`C_plus_state`.

PRE-OUTCOME DISCIPLINE (the L27 trap).  Every added column is a function of
`start_score_diff`, `start_clock`, `period`, `off_in_bonus`,
`off_in_double_bonus` and the site only.

  * `start_score_diff` is PRE-OUTCOME and this build re-asserts it: the own-row
    delta test (next chance's start margin == this chance's start margin + this
    chance's own signed points) holds on 98.45% of SCORING chances against
    0.90% for the post-outcome alternative, and the residual is the and-one /
    multi-chance bookkeeping, not a leak.  L27 already recorded that
    `possession_outcome` and `clock` read the clean column; the assertion is
    re-run here so the claim is this build's, not an inherited one.
  * `duration_s` is POST-OUTCOME and is NEVER added as a feature.  It is the
    duration half's TARGET and lives in the clock trainer.
  * `is_transition` in the chances table is `possession duration <= 8 AND
    start_reason in {DREB, TOV}`, i.e. post-OUTCOME but pre-EVENT: the engine
    draws the duration BEFORE the event model runs (`loop.py` step (a), and the
    module docstring says so explicitly), so it is legitimately available to
    the event half and is carried unchanged from the served bundle.  It is
    forbidden to the duration half, where it would be the target.

The two as-of team columns pass a leak test before they may enter `L4_team`
(CLAUDE.md backtest rules): change-form correlation with own-week margin,
`|corr| <= 0.15`.  The test is written to `leak_test_v1.csv` and the builder
REFUSES to mark the bundle usable if it fails.

Usage:  build_late_game_design_v1.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from cbb_sim.features import conference as CONF                      # noqa: E402
from cbb_sim.models import possession_outcome as PO                  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]
R2_DESIGN = ROOT / "data/processed/models/possession_outcome/round2/design.parquet"
OUT_DIR = ROOT / "data/processed/models/late_game"

# ---------------------------------------------------------------------------
# The regime gate, FROZEN before any fit (experiments.md 1.1)
# ---------------------------------------------------------------------------
GATE_SEC, GATE_MARGIN = 120.0, 6.0
#: Declared now, used for REPORTING only, never for selection.
SENSITIVITY_GATES: tuple[tuple[float, float], ...] = ((150.0, 8.0), (90.0, 5.0))

#: The two behavioural-gate cut points of `features.md`.  Both are SPEC
#: constants fixed before any fit and never searched: `K_FOUL` is the midpoint
#: of the 120 s window, `K_HOLD` is `clock_adapter_v3.EOH_WINDOW_S`, the
#: end-of-half constant the clock lane already uses.  Neither is tuned on any
#: outcome.
K_FOUL, K_HOLD = 60.0, 35.0

CLASSES = list(PO.CLASSES)
I_BONUS = PO.CLASS_INDEX["FT_trip_bonus"]
I_THREE = PO.CLASS_INDEX["FGA_3"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ===========================================================================
# 1. The pre-outcome assertion on `start_score_diff`
# ===========================================================================
def assert_score_diff_pre_outcome(season: int = 2025) -> dict:
    """The L27 own-row delta test, re-run on this lane's own read of the data.

    PRE-outcome means the NEXT chance's start margin is this chance's start
    margin PLUS this chance's own signed points.  POST-outcome would mean the
    next chance's start margin equals this one's.  Both are measured; the test
    is run on SCORING chances, where the two hypotheses separate."""
    ch = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{season}.parquet")
    ch = ch.sort_values(["game_id", "period", "poss_index", "chance_number"])
    sg = np.where(ch["offense_is_home"].to_numpy(), 1.0, -1.0)
    home_sd = ch["start_score_diff"].to_numpy() * sg
    pts = ch["points"].to_numpy() * sg
    g = ch["game_id"].to_numpy()
    same = np.concatenate([g[1:] == g[:-1], [False]])
    nxt = np.concatenate([home_sd[1:], [np.nan]])
    scoring = (ch["points"].to_numpy() > 0) & same
    pre = float(np.isclose(nxt[scoring], (home_sd + pts)[scoring]).mean())
    post = float(np.isclose(nxt[scoring], home_sd[scoring]).mean())
    out = {"season": season, "n_scoring_chances": int(scoring.sum()),
           "pre_outcome_match": round(pre, 6), "post_outcome_match": round(post, 6)}
    if pre < 0.95 or post > 0.05:
        raise AssertionError(
            "start_score_diff failed the L27 own-row delta test: "
            f"pre {pre:.4f} post {post:.4f}. Refusing to build a design on it.")
    log(f"  L27 own-row delta test PASS: pre-outcome {pre:.4%}, post-outcome {post:.4%} "
        f"on {int(scoring.sum()):,} scoring chances of {season}")
    return out


# ===========================================================================
# 2. As-of, strictly-before, league-centred late-window team rates
# ===========================================================================
def window_team_rates(d: pd.DataFrame) -> pd.DataFrame:
    """Four as-of team columns, built from WINDOW rows of strictly prior games.

    Per (game, team) and per side of the ball:
        late_foul  = share of the team's window chances ending `FT_trip_bonus`
        late_3pa   = share of the team's window chances ending `FGA_3`
    accumulated over that team's own EARLIER games only (a `cumsum` minus the
    row's own value, the same shift(1) expanding builder
    `possession_outcome._expanding_asof` uses), then expressed as a deviation
    from the league's OWN as-of mean on the same date (CLAUDE.md: raw levels
    are banned).  A team with no prior window chances sits at exactly 0.0,
    which IS the league mean and not a fabricated level.

    Offence side and defence side are both built: the offence's own late foul
    rate and the DEFENCE's late foul-conceded rate are different quantities and
    `FT_trip_bonus` is produced by the pair."""
    w = d[(d["period"] == 2) & (d["seconds_remaining"] <= GATE_SEC)
          & (d["score_diff"].abs() <= GATE_MARGIN)]
    y = w["y"].to_numpy()
    base = pd.DataFrame({
        "season": w["season"].to_numpy(), "game_id": w["game_id"].to_numpy(),
        "game_date": pd.to_datetime(w["game_date"]).to_numpy(),
        "off": w["offense_team_id"].to_numpy(), "def": w["defense_team_id"].to_numpy(),
        "n": 1.0, "foul": (y == I_BONUS).astype("float64"),
        "three": (y == I_THREE).astype("float64"),
    })
    per_game = base.groupby(["season", "game_id", "game_date", "off", "def"],
                            as_index=False)[["n", "foul", "three"]].sum()

    cols = ["n", "foul", "three"]
    out = []
    for side, key in (("off", "off"), ("def", "def")):
        g = per_game.rename(columns={key: "team_id"})
        g = g.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
        asof = PO._expanding_asof(g, ["season", "team_id"], cols)
        frame = pd.concat([g[["season", "game_id", "team_id", "game_date"]], asof], axis=1)
        out.append((side, frame))

    # league as-of: cumulative over every team-game strictly earlier
    day = per_game.groupby(["season", "game_date"], as_index=False)[cols].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PO._expanding_asof(day, ["season"], cols)
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)

    frames = {}
    for side, frame in out:
        f = frame.merge(day, on=["season", "game_date"], how="left")
        n, lgn = f["n"].to_numpy(), f["lg_n"].to_numpy()
        for what, col in (("foul", f"{side}_late_foul_c"), ("three", f"{side}_late_3pa_c")):
            own = np.where(n > 0, f[what].to_numpy() / np.maximum(n, 1e-9), np.nan)
            lgr = np.where(lgn > 0, f[f"lg_{what}"].to_numpy() / np.maximum(lgn, 1e-9), np.nan)
            f[col] = np.where(np.isnan(own) | np.isnan(lgr), 0.0, own - lgr).astype("float32")
        f["n_prior_window"] = f["n"].astype("float32")
        frames[side] = f[["season", "game_id", "team_id",
                          f"{side}_late_foul_c", f"{side}_late_3pa_c", "n_prior_window"]]
    return frames


def leak_test(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """CLAUDE.md's change-form leak test: the WEEK-ON-WEEK CHANGE in the feature
    must not correlate with the team's OWN margin in that same week.

    An honest as-of feature moves a little with the week's result (it is built
    from prior games, and form is persistent), which the rule prices at
    0.04-0.08; a feature that has seen the week's own games moves a lot.  The
    bar is `|corr| <= 0.15`."""
    u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet")
    u = u[["game_id", "game_date", "home_team_id", "away_team_id",
           "home_score", "away_score"]].copy()
    u["game_date"] = pd.to_datetime(u["game_date"])
    tg = d.groupby(["season", "game_id", "offense_team_id"], as_index=False)[cols].first()
    tg = tg.rename(columns={"offense_team_id": "team_id"})
    tg = tg.merge(u, on="game_id", how="inner")
    home = tg["team_id"].to_numpy() == tg["home_team_id"].to_numpy()
    tg["own_margin"] = np.where(home, tg["home_score"] - tg["away_score"],
                                tg["away_score"] - tg["home_score"])
    tg["week"] = tg["game_date"].dt.to_period("W").dt.start_time
    wk = tg.groupby(["season", "team_id", "week"], as_index=False).agg(
        **{c: (c, "mean") for c in cols}, own_margin=("own_margin", "mean"))
    wk = wk.sort_values(["season", "team_id", "week"], kind="stable")
    rows = []
    for c in cols:
        chg = wk.groupby(["season", "team_id"], sort=False)[c].diff()
        m = chg.notna()
        r = float(np.corrcoef(chg[m], wk["own_margin"][m])[0, 1])
        rows.append({"feature": c, "n_week_changes": int(m.sum()),
                     "corr_change_vs_own_week_margin": round(r, 5),
                     "bar": 0.15, "pass": bool(abs(r) <= 0.15)})
    return pd.DataFrame(rows)


# ===========================================================================
# 3. The late-game state columns
# ===========================================================================
LATE_COLS: tuple[str, ...] = (
    "site_neutral", "role", "poss_deficit", "gt_margin", "in_double_bonus",
    "trail_must_foul", "lead_can_hold", "x_role__sec", "x_gt_margin__sec",
    "is_conf_game",
    "off_late_foul_c", "off_late_3pa_c", "def_late_foul_c", "def_late_3pa_c",
)


def add_late_state(d: pd.DataFrame) -> pd.DataFrame:
    sd = d["score_diff"].to_numpy(dtype="float64")
    sec = d["seconds_remaining"].to_numpy(dtype="float64")
    a = np.abs(sd)
    d["site_neutral"] = (1.0 - d["site_home"].to_numpy()
                         - d["site_away"].to_numpy()).astype("float32")
    d["role"] = np.sign(sd).astype("float32")
    d["poss_deficit"] = np.ceil(a / 3.0).astype("float32")
    d["gt_margin"] = np.clip(a, 0.0, 6.0).astype("float32")
    d["in_double_bonus"] = d["off_in_double_bonus"].astype("float32")
    d["trail_must_foul"] = ((sd < 0) & (sec <= K_FOUL)).astype("float32")
    d["lead_can_hold"] = ((sd > 0) & (sec <= K_HOLD)).astype("float32")
    d["x_role__sec"] = (d["role"].to_numpy() * sec / 120.0).astype("float32")
    d["x_gt_margin__sec"] = (d["gt_margin"].to_numpy() * sec / 120.0).astype("float32")
    return d


# ===========================================================================
# 4. Bundles (features.md section 2)
# ===========================================================================
def bundle(name: str, population: str) -> list[str]:
    """`L0_reference` is EXACTLY the served `C_plus_state`; every richer bundle
    is a strict superset of it plus `site_neutral`, so home/away/neutral is a
    first-class three-level feature in every arm (CLAUDE.md)."""
    l0 = PO.feature_set("C_plus_state", population)
    if name == "L0_reference":
        return l0
    l1 = l0 + ["site_neutral", "role", "gt_margin", "x_role__sec"]
    if name == "L1_role":
        return l1
    l2 = l1 + ["poss_deficit", "in_double_bonus", "x_gt_margin__sec"]
    if name == "L2_role_poss":
        return l2
    l3 = l2 + ["trail_must_foul", "lead_can_hold"]
    if name == "L3_gates":
        return l3
    if name == "L4_team":
        return l3 + ["off_late_foul_c", "off_late_3pa_c",
                     "def_late_foul_c", "def_late_3pa_c", "is_conf_game"]
    raise KeyError(f"unknown bundle {name!r}")


BUNDLES: tuple[str, ...] = ("L0_reference", "L1_role", "L2_role_poss",
                            "L3_gates", "L4_team")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    diag = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source_design": str(R2_DESIGN.relative_to(ROOT)).replace("\\", "/"),
            "gate": {"period": 2, "seconds_remaining_max": GATE_SEC,
                     "abs_score_diff_max": GATE_MARGIN},
            "sensitivity_gates": [list(g) for g in SENSITIVITY_GATES],
            "k_foul_s": K_FOUL, "k_hold_s": K_HOLD}

    diag["l27_own_row_delta_test"] = assert_score_diff_pre_outcome(2025)

    log(f"loading {R2_DESIGN.name}")
    d = pd.read_parquet(R2_DESIGN)
    d["game_date"] = pd.to_datetime(d["game_date"])
    log(f"  {len(d):,} rows, seasons {sorted(d['season'].unique().tolist())}")

    log("conference flags")
    conf = CONF.build_conference_flags(SEASONS)
    diag["n_missing_conference_id"] = int(conf.attrs.get("n_missing_conference_id", -1))
    d = d.merge(conf[["game_id", "is_conf_game"]], on="game_id", how="left")
    diag["n_conf_flag_unmatched"] = int(d["is_conf_game"].isna().sum())
    d["is_conf_game"] = d["is_conf_game"].fillna(False).astype("float32")

    log("late-game state columns")
    d = add_late_state(d)

    log("as-of window team rates (strictly prior games, league-centred)")
    frames = window_team_rates(d)
    d = d.merge(frames["off"].rename(columns={"team_id": "offense_team_id",
                                              "n_prior_window": "n_prior_w_off"}),
                on=["season", "game_id", "offense_team_id"], how="left")
    d = d.merge(frames["def"].rename(columns={"team_id": "defense_team_id",
                                              "n_prior_window": "n_prior_w_def"}),
                on=["season", "game_id", "defense_team_id"], how="left")
    for c in ("off_late_foul_c", "off_late_3pa_c", "def_late_foul_c", "def_late_3pa_c"):
        d[c] = d[c].astype("float32").fillna(0.0)
    for c in ("n_prior_w_off", "n_prior_w_def"):
        d[c] = d[c].astype("float32").fillna(0.0)

    log("leak test on the two as-of window rate families")
    lk = leak_test(d, ["off_late_foul_c", "off_late_3pa_c",
                       "def_late_foul_c", "def_late_3pa_c"])
    lk.to_csv(out / "leak_test_v1.csv", index=False)
    diag["leak_test"] = lk.to_dict("records")
    print(lk.to_string(index=False))
    if not lk["pass"].all():
        raise AssertionError("an as-of window rate failed the leak test; L4_team is unusable")

    # --- window flags, main gate and both sensitivity members ---------------
    per2 = d["period"].to_numpy() == 2
    sec = d["seconds_remaining"].to_numpy()
    asd = np.abs(d["score_diff"].to_numpy())
    d["in_window"] = (per2 & (sec <= GATE_SEC) & (asd <= GATE_MARGIN))
    for j, (s, m) in enumerate(SENSITIVITY_GATES):
        d[f"in_window_sens{j + 1}"] = (per2 & (sec <= s) & (asd <= m))
    d["is_regulation"] = d["period"].to_numpy() <= 2

    counts = {
        "n_rows": int(len(d)),
        "n_regulation": int(d["is_regulation"].sum()),
        "n_window": int(d["in_window"].sum()),
        "window_share_of_regulation": round(float(d["in_window"].sum()
                                                  / d["is_regulation"].sum()), 5),
        "by_season": {int(s): int(v) for s, v in
                      d[d["in_window"]].groupby("season").size().items()},
        "by_population": {str(k): int(v) for k, v in
                          d[d["in_window"]].groupby("population").size().items()},
    }
    for j in (1, 2):
        counts[f"n_window_sens{j}"] = int(d[f"in_window_sens{j}"].sum())
    diag["counts"] = counts
    log("  " + json.dumps(counts))

    keep = [c for c in d.columns if c not in ("start_reason",)]
    p = out / "design_v1.parquet"
    d[keep].to_parquet(p, index=False)
    diag["runtime_min"] = round((time.time() - t0) / 60, 2)
    diag["size_mb"] = round(p.stat().st_size / 1e6, 1)
    diag["bundles"] = {b: bundle(b, "first") for b in BUNDLES}
    (out / "design_v1.meta.json").write_text(json.dumps(diag, indent=1, default=str),
                                             encoding="utf-8")
    log(f"wrote {p} ({diag['size_mb']} MB) in {diag['runtime_min']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
