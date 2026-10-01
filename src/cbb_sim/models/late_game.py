"""late_game.py -- L7 late-game regime: the fitted window objects round 2 serves.

Pre-registration: `docs/models/late_game/experiments.md` section 4 (round 2).
Round 1 (`scripts/train_late_game_clock_v1.py`) defined the duration cells in a
SCRIPT, which is fine for an offline grid and fatal for serving: a pickle of a
class defined in `__main__` cannot be loaded by an engine worker. This module
holds the same code, verbatim in behaviour, so the round-2 artifacts pickle
against an importable class. `scripts/train_late_game_r2_v1.py` refits the two
selected duration cells and asserts their test-row PMFs equal round 1's saved
ones EXACTLY before writing anything.

Nothing here is served unless `ENGINE_LATE_GAME` names it
(`cbb_sim.engine.late_game_adapter`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cbb_sim.models import clock as ck
from cbb_sim.models import clock_v3 as c3

#: The regime gate, section 1.1 (frozen before round 1 fitted anything).
GATE_PERIOD, GATE_SEC, GATE_MARGIN = 2.0, 120.0, 6.0

#: Behavioural-gate cut points of the L3 bundle (round-1 amendment 2.5.6).
K_FOUL, K_HOLD = 60.0, 35.0

# --- new cell dimensions, registered on import exactly as round 1 did --------
ck.EMPIRICAL_DIM_SIZES["eg_role6"] = 6
ck.EMPIRICAL_DIM_SIZES["role3"] = 3
ck.EMPIRICAL_DIMS["P3R_dummy"] = (
    "prev_end_code", "r2_bucket_code", "r2_period_type", "eg_role6", "tempo_tercile")
ck.EMPIRICAL_DIMS["LGD_dummy"] = (
    "role3", "r2_bucket_code", "bonus_code", "prev_end_code")
ck.FEATURE_SETS["P3R_dummy"] = ck.FEATURE_SETS["P3_dummy"]
ck.FEATURE_SETS["LGD_dummy"] = ck.FEATURE_SETS["P3_dummy"]
# round 4 (experiments.md section 9): D's cell with role3 refined to round 1's five bands.
ck.EMPIRICAL_DIMS["LGL_dummy"] = ("eg_role6", "r2_bucket_code", "bonus_code", "prev_end_code")
ck.FEATURE_SETS["LGL_dummy"] = ck.FEATURE_SETS["P3_dummy"]


def in_window(period, seconds_remaining, score_diff) -> np.ndarray:
    """`period == 2 and seconds_remaining <= 120 and |score_diff| <= 6`.

    Overtime is OUTSIDE (section 4.1): round 1 fitted and graded on period 2."""
    per = np.asarray(period, dtype=np.float64)
    sr = np.asarray(seconds_remaining, dtype=np.float64)
    sd = np.asarray(score_diff, dtype=np.float64)
    return (per == GATE_PERIOD) & (sr <= GATE_SEC) & (np.abs(sd) <= GATE_MARGIN)


def eg_role6_code(sr, per, sd) -> np.ndarray:
    """0 outside the end-game window; inside it, five ordered role bands
    (round 1's `train_late_game_clock_v1.eg_role6_code`, unchanged)."""
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
    """`EmpiricalArmV3P` plus the two new cell codes (round 1, unchanged)."""

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
    """Round 1's `fit_cell_arm`, unchanged in behaviour."""
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


#: The L3_gates extras, in bundle order (`build_late_game_design_v1.bundle`):
#: appended after the served `C_plus_state` first-chance list.
L3_EXTRAS: tuple[str, ...] = (
    "site_neutral", "role", "gt_margin", "x_role__sec",
    "poss_deficit", "in_double_bonus", "x_gt_margin__sec",
    "trail_must_foul", "lead_can_hold",
)


def l3_extras(site_home, site_away, score_diff, seconds_remaining,
              in_double_bonus) -> np.ndarray:
    """The nine L3 columns from live state, the same formulas as
    `build_late_game_design_v1.add_late_state` (float32 there, float64 here;
    LightGBM casts to float32 at predict)."""
    sd = np.asarray(score_diff, dtype=np.float64)
    sec = np.asarray(seconds_remaining, dtype=np.float64)
    a = np.abs(sd)
    role = np.sign(sd)
    gt = np.clip(a, 0.0, 6.0)
    cols = {
        "site_neutral": 1.0 - np.asarray(site_home, float) - np.asarray(site_away, float),
        "role": role,
        "gt_margin": gt,
        "x_role__sec": role * sec / 120.0,
        "poss_deficit": np.ceil(a / 3.0),
        "in_double_bonus": np.asarray(in_double_bonus, dtype=np.float64),
        "x_gt_margin__sec": gt * sec / 120.0,
        "trail_must_foul": ((sd < 0) & (sec <= K_FOUL)).astype(np.float64),
        "lead_can_hold": ((sd > 0) & (sec <= K_HOLD)).astype(np.float64),
    }
    return np.column_stack([cols[c] for c in L3_EXTRAS])
