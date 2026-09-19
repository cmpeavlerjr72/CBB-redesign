"""train_late_game_clock_v1.py -- L7 late-game regime, round 1: the DURATION half.

Pre-registration: `docs/models/late_game/experiments.md` section 1 and the
section-2 amendment.  This trainer computes NO metric: it writes, per cell, the
predicted duration PMF over the fold's WINDOW test rows, and
`scripts/grade_late_game_v1.py` scores every cell by one code path.

WHAT THE AMENDMENT CHANGED, AND WHY THIS FILE EXISTS IN THIS SHAPE.  The
2026-09-11 evidence named the duration defect as "the served clock carries no
`score_diff` at all", citing `clock_adapter_v3.CELL_DIMS`.  That constant is the
`ENGINE_CLOCK_DIAG` accumulator's grid, NOT the serving cell grid.  The served
arm is `empirical_km3_srfloor | P3`, whose cells are
`clock_v3.EMPIRICAL_DIMS["P3_dummy"] = (prev_end, r2_bucket, r2_period_type,
eg_regime, tempo_tercile)` with
`eg_regime = {0 outside, 1 trailing by >= 4, 2 leading by >= 4, 3 close}` inside
the last 120 s of H2/OT -- a role-conditioned, clock-gated end-game dimension.
The real structural limit is a DIFFERENT one, and it is exact:
`sr_floor_bucket = 5` floors the fine clock bucket at "45-59", so every row with
under 45 seconds left is served the 45-59 s law.  That is why the sim duration
flatlines at ~10.4 s where the data falls to 6.56 and 2.89, and it is what the
arms below vary.

THE ARMS (all the same FAMILY -- the served Kaplan-Meier cell law -- so the grid
tests the CONDITIONING, exactly as section 1.2 requires):

  A_clk   reference : served spec.  P3 dims, `sr_floor_bucket = 5`, all rows.
  B1_clk  enrichment: P3 dims, floor REMOVED, all rows.  The single-change arm.
  B2_clk  enrichment: floor removed AND `eg_regime` refined from 4 levels to 6
                      (trailing>=4 / trailing 1-3 / tied / leading 1-3 /
                      leading>=4, plus "outside"), all rows.
  C_clk   regime refit : B1's spec fitted on WINDOW ROWS ONLY.
  C2_clk  regime refit : B2's spec fitted on WINDOW ROWS ONLY.
  D_clk   dedicated : its own grid, window rows only --
                      (role3 x fine clock bucket x bonus x prev_end).

NOISE FLOOR.  The cell law is deterministic given its training rows, so a
"second seed" does nothing.  The floor is therefore the game-level block
bootstrap SE of the test metric, computed by the grader -- the same instrument
`possession_outcome`'s pre-registration uses for its deterministic arms.

PRE-OUTCOME.  `duration_s` is the TARGET.  `is_transition` is a function of the
duration and is FORBIDDEN here; it appears in no cell dimension.  Every cell
dimension is a function of `seconds_remaining`, `period`, `start_score_diff`,
`in_bonus`, `prev_end` and the pregame tempo, all read at possession start.

Usage: train_late_game_clock_v1.py [--folds F1,F2]
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
    os.environ.setdefault(_v, "3")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from cbb_sim.models import clock as ck                               # noqa: E402
from cbb_sim.models import clock_v3 as c3                            # noqa: E402

DESIGN_V2 = ROOT / "data/processed/models/clock/design_v2.parquet"
OUT = ROOT / "data/processed/models/late_game/round1_clock"
ALL_SEASONS = [2022, 2023, 2024, 2025]
GATE_SEC, GATE_MARGIN = 120.0, 6.0

# --- new cell dimensions, registered in THIS PROCESS only -------------------
# `clock_v3` sets the precedent: it mutates `ck.EMPIRICAL_DIMS` at import so
# every fitter sees P2/P3 without `clock.py` on disk changing.  Same pattern.
ck.EMPIRICAL_DIM_SIZES["eg_role6"] = 6
ck.EMPIRICAL_DIM_SIZES["role3"] = 3
ck.EMPIRICAL_DIMS["P3R_dummy"] = (
    "prev_end_code", "r2_bucket_code", "r2_period_type", "eg_role6", "tempo_tercile")
ck.EMPIRICAL_DIMS["LGD_dummy"] = (
    "role3", "r2_bucket_code", "bonus_code", "prev_end_code")
ck.FEATURE_SETS["P3R_dummy"] = ck.FEATURE_SETS["P3_dummy"]
ck.FEATURE_SETS["LGD_dummy"] = ck.FEATURE_SETS["P3_dummy"]


def eg_role6_code(sr, per, sd) -> np.ndarray:
    """0 outside the end-game window; inside it, five ordered role bands.

    The refinement of `clock_v3.eg_regime_code`: that function collapses
    `|margin| < 4` into one "close" level, which is precisely the band the
    trailing team is most likely to tie from, and it therefore cannot separate
    a one-possession game from a tie."""
    sr = np.asarray(sr, dtype="float64")
    per = np.asarray(per, dtype="float64")
    sd = np.asarray(sd, dtype="float64")
    inside = (sr <= c3.ENDGAME_WINDOW_S) & (per >= 2.0)
    out = np.zeros(sr.shape, dtype="int64")
    out = np.where(inside & (sd <= -4), 1, out)
    out = np.where(inside & (sd < 0) & (sd > -4), 2, out)
    out = np.where(inside & (sd == 0), 3, out)
    out = np.where(inside & (sd > 0) & (sd < 4), 4, out)
    out = np.where(inside & (sd >= 4), 5, out)
    return out


class LateCellArm(c3.EmpiricalArmV3P):
    """`EmpiricalArmV3P` plus the two new cell codes.  Every other behaviour --
    the Kaplan-Meier estimator, the hierarchical fallback, the tail rule, the
    `min_cell`/`min_events` thresholds -- is inherited unchanged, so the arms
    differ from the reference ONLY in their conditioning."""

    def _codes(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        codes = super()._codes(df)
        sr = df["seconds_remaining"].to_numpy()
        per = df["period"].to_numpy()
        sd = df["score_diff"].to_numpy()
        codes["eg_role6"] = eg_role6_code(sr, per, sd)
        codes["role3"] = (np.sign(sd).astype("int64") + 1)
        codes["bonus_code"] = df["in_bonus"].to_numpy().astype("int64")
        return codes


def fit_cell_arm(tr: pd.DataFrame, fs: str, floor: int, tag: str) -> LateCellArm:
    """`clock_v3._fit_empirical_p`'s body with `LateCellArm` in place of
    `EmpiricalArmV3P`.  Nothing else differs; the estimator is the fitted
    object the served arm uses."""
    dims = ck.EMPIRICAL_DIMS[fs]
    sizes = tuple(ck.EMPIRICAL_DIM_SIZES[d] for d in dims)
    tempo = tr["tempo_prior_game"].to_numpy(dtype="float64")
    arm = LateCellArm(
        feature_set_name=fs, dims=dims, sizes=sizes,
        level_pmfs=[], level_counts=[], level_events=[],
        tempo_edges=(float(np.quantile(tempo, 1 / 3)), float(np.quantile(tempo, 2 / 3))),
        season_map={s: i for i, s in enumerate(sorted(int(x) for x in tr["season"].unique()))},
        sr_floor_bucket=floor, features=ck.feature_set(fs), name=tag)
    codes = arm._codes(tr)
    y = tr["duration_s"].to_numpy(dtype="int64")
    cen = tr["censored"].to_numpy(dtype=bool)
    pooled = np.zeros((1, c3.N_GRID), dtype="float64")
    np.add.at(pooled, (np.zeros(int((~cen).sum()), dtype="int64"), y[~cen]), 1.0)
    pooled = ck._normalise(pooled)
    for lv in range(len(dims) + 1):
        n_cells = 1 if lv == 0 else int(np.prod(sizes[:lv]))
        keys = arm._keys(codes, lv)
        if lv == 0:
            parent_pmf, parent_of = pooled, np.zeros(1, dtype="int64")
        else:
            parent_pmf = arm.level_pmfs[lv - 1]
            parent_of = ((np.arange(n_cells, dtype="int64") // sizes[lv - 1])
                         if lv > 1 else np.zeros(n_cells, dtype="int64"))
        pmf, counts, events = c3.kaplan_meier_pmf_v3(keys, y, cen, n_cells,
                                                     parent_pmf, parent_of)
        arm.level_pmfs.append(ck._normalise(pmf))
        arm.level_counts.append(counts)
        arm.level_events.append(events)
    arm.level_counts[0] = np.maximum(arm.level_counts[0], arm.min_cell)
    arm.level_events[0] = np.maximum(arm.level_events[0], arm.min_events)
    return arm


#: (cell id, scope, feature set, sr floor).  `scope` is `full` (fit on every
#: modelled possession, as the served arm is) or `window` (fit on window rows).
ARMS: tuple[tuple[str, str, str, int], ...] = (
    ("A_clk",  "full",   "P3_dummy",  c3.SR_FLOOR_BUCKET),
    ("B1_clk", "full",   "P3_dummy",  0),
    ("B2_clk", "full",   "P3R_dummy", 0),
    ("C_clk",  "window", "P3_dummy",  0),
    ("C2_clk", "window", "P3R_dummy", 0),
    ("D_clk",  "window", "LGD_dummy", 0),
)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="F1,F2")
    a = ap.parse_args()
    folds = tuple(x.strip() for x in a.folds.split(",") if x.strip())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pmf").mkdir(exist_ok=True)

    log(f"loading {DESIGN_V2.name}")
    d = pd.read_parquet(DESIGN_V2)
    d, cdiag = c3.attach_horn_censoring(d, ALL_SEASONS)
    c3.set_flag_inplace(d, "horn")
    d = c3.add_p_state(d)
    log(f"  {len(d):,} rows; horn-censored {cdiag['censored_horn_pct']}%")

    per2 = d["period"].to_numpy() == 2
    sec = d["seconds_remaining"].to_numpy()
    asd = np.abs(d["score_diff"].to_numpy())
    d["in_window"] = per2 & (sec <= GATE_SEC) & (asd <= GATE_MARGIN)
    log(f"  window rows {int(d['in_window'].sum()):,} "
        f"({d['in_window'].mean() * 100:.2f}% of all modelled possessions)")

    manifest = []
    for fold in folds:
        tr_all, te_all = ck.fold_slices(d, fold)
        order = ["game_id", "period", "poss_index"]
        te = te_all[te_all["in_window"].to_numpy()].sort_values(order, kind="stable")
        p = OUT / f"window_test_{fold}.parquet"
        keep = ["game_id", "season", "period", "poss_index", "game_date",
                "offense_team_id", "defense_team_id", "offense_is_home",
                "duration_s", "censored", "seconds_remaining", "score_diff",
                "in_bonus", "prev_end", "tempo_prior_game", "terminal_event",
                "site_home", "site_away"]
        te[[c for c in keep if c in te.columns]].to_parquet(p, index=False)
        log(f"{fold}: train {len(tr_all):,}  window test {len(te):,} "
            f"(censored {int(te['censored'].sum()):,})")

        for cell, scope, fs, floor in ARMS:
            t0 = time.time()
            tr = tr_all if scope == "full" else tr_all[tr_all["in_window"].to_numpy()]
            arm = fit_cell_arm(tr, fs, floor, f"{cell}|{fs}|floor{floor}")
            pmf = arm.pmf(te.copy())
            np.save(OUT / "pmf" / f"{cell}_{fold}.npy", pmf.astype("float32"))
            lv = arm.level_of(te.copy())
            manifest.append({
                "cell_id": f"{cell}_{fold}", "arm": cell, "scope": scope,
                "feature_set": fs, "sr_floor_bucket": floor, "fold": fold,
                "dims": list(arm.dims), "sizes": list(arm.sizes),
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "max_train_date": str(pd.Timestamp(tr["game_date"].max()).date()),
                "level_used_mean": round(float(lv.mean()), 3),
                "level_used_full_pct": round(float((lv == len(arm.dims)).mean() * 100), 2),
                "runtime_s": round(time.time() - t0, 1),
            })
            log(f"  {cell}_{fold}: {len(tr):,} train rows, dims {arm.dims}, "
                f"mean fallback level {lv.mean():.2f}/{len(arm.dims)}, "
                f"{manifest[-1]['runtime_s']}s")

    (OUT / "cells.json").write_text(json.dumps(manifest, indent=1, default=str),
                                    encoding="utf-8")
    log(f"done, {len(manifest)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
