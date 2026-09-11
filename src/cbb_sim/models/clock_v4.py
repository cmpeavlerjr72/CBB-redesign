"""
clock_v4.py -- round 4: carrying the CURRENT season's duration level.

Pre-registration: `docs/models/clock/experiments.md` section 14, committed
before this file existed. Diagnosis:
`docs/tests/clock_duration_shortfall_2026-09-11.md`.

WHAT ROUND 4 IS FOR
-------------------
Round 3c's residual was recorded (L31) as a uniform -1.9% mean-duration
shortfall. The round-4 diagnosis shows it is not uniform: two thirds of it is
the ENGINE's `prev_end` composition (an upstream defect this module does not
touch) and the clock's own share is a LEVEL problem. A cell law pooled over four
seasons under-tracks the season being simulated: 2025's clock-complete mean
regulation possession duration is 17.653 s against 17.510 / 17.584 / 17.547 for
2024 / 2023 / 2022, and inside 2025 it runs 17.374 s in November against
17.836 s in February. S1 already refits monthly on all prior seasons plus the
season to date, but by January the current season is only 13% of the training
mass, so the pooled cell mean cannot move far enough.

The three answers this module implements, all inside the SAME family (cell
resampling, corrected horn censoring, the round-3 Kaplan-Meier tail rule, P3
state, S1 schedule) so that the round is a comparison of one thing:

  A1 `recency`    exponentially recency-weighted training rows through a
                  WEIGHTED discrete-time Kaplan-Meier. One parameter, the
                  half-life, chosen on F1 ONLY.
  A2 `curseason`  a two-level calendar CELL DIMENSION (the season being
                  simulated vs prior seasons) appended LAST, so the existing
                  hierarchical fallback serves a cell from the current season's
                  own rows when it has enough of them and from the pooled cell
                  otherwise. No new knob.
  A3 `calpart`    a three-level SEASON-PART dimension appended last (calendar
                  month {11,12} -> 0, {1} -> 1, {2,3,4} -> 2), pooling across
                  seasons within a part. Fixes the within-season trend and not
                  the cross-season level, which is what separates it from A2.

WHAT IS NOT HERE, DELIBERATELY
------------------------------
No multiplier, scale, cap, clip, offset or calibration curve on a predicted or
simulated duration. A recency WEIGHT enters the fit and nothing else; the
predictive law is still the cell's own censored empirical law. A "league mean
duration" rescaling would be exactly the multiplicative duration scaling the
standing no-hand-tuning rule and `docs/SIM_GUARDRAILS.md` forbid, and it is not
implemented in any form.

The eligibility thresholds stay on RAW counts (`EMPIRICAL_MIN_CELL` rows,
`EMPIRICAL_MIN_EVENTS` uncensored exits) even under weighting, so A1 and the
reference fall back at the same places and a weight cannot silently starve a
cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cbb_sim.models import clock as ck
from cbb_sim.models import clock_v3 as c3

N_GRID = c3.N_GRID

#: Calendar treatments. "none" reproduces round 3's arm exactly.
CAL_MODES: dict[str, int] = {"none": 1, "curseason": 2, "calpart": 3}

#: Pre-registered half-life grid for A1, in days. Chosen on F1 ONLY.
HALF_LIFE_GRID_DAYS: tuple[int, ...] = (120, 365, 730)

#: Season-part coding for A3, from the calendar month. Fixed here, not fitted.
CALPART_OF_MONTH: dict[int, int] = {11: 0, 12: 0, 1: 1, 2: 2, 3: 2, 4: 2,
                                    5: 2, 6: 2, 7: 2, 8: 0, 9: 0, 10: 0}


def cal_code(df: pd.DataFrame, mode: str, cur_season: int) -> np.ndarray:
    """The calendar cell code per row.

    `curseason`: 0 is the season being simulated, 1 is every prior season. At
    serve time every row is the current season, so a served row always asks the
    level-0 slice of this dimension -- which is exactly the intent, and why the
    dimension is appended LAST where the fallback drops it first.

    `calpart`: the part of the season, pooled across seasons."""
    n = len(df)
    if mode == "none":
        return np.zeros(n, dtype="int64")
    if mode == "curseason":
        if "season" not in df.columns:
            # The engine frame always describes the season being simulated.
            return np.zeros(n, dtype="int64")
        return (df["season"].to_numpy().astype("int64") != int(cur_season)).astype("int64")
    if mode == "calpart":
        if "month" not in df.columns:
            raise KeyError("calpart needs a `month` column; the engine adapter supplies it")
        m = df["month"].to_numpy().astype("int64")
        out = np.full(n, 2, dtype="int64")
        for k, v in CALPART_OF_MONTH.items():
            out[m == k] = v
        return out
    raise ValueError(f"unknown calendar mode {mode!r}")


# ===========================================================================
# 1. weighted Kaplan-Meier
# ===========================================================================
def kaplan_meier_pmf_v4(cell: np.ndarray, y: np.ndarray, censored: np.ndarray,
                        n_cells: int, parent_pmf: np.ndarray, parent_of: np.ndarray,
                        w: np.ndarray | None = None,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`clock_v3.kaplan_meier_pmf_v3` with per-row weights.

    Identical in every other respect, including the tail rule: the survival that
    never resolves inside a cell is distributed over t > t* in proportion to the
    PARENT cell's pmf restricted to t > t*. With `w = None` (or all ones) this
    returns bit-identical output to the v3 function, which is the property the
    round-4 reference arm depends on.

    The returned counts are RAW, not weighted: cell eligibility must not depend
    on the weighting, or an arm with a short half-life would fall back in
    different places from the reference and the comparison would be of two
    things at once."""
    if w is None:
        return c3.kaplan_meier_pmf_v3(cell, y, censored, n_cells, parent_pmf, parent_of)
    w = np.asarray(w, dtype="float64")
    d = np.zeros((n_cells, N_GRID), dtype="float64")
    c = np.zeros((n_cells, N_GRID), dtype="float64")
    unc = ~censored
    np.add.at(d, (cell[unc], y[unc]), w[unc])
    np.add.at(c, (cell[censored], y[censored]), w[censored])
    n_total = (d + c).sum(axis=1, keepdims=True)
    left_before = np.concatenate([np.zeros((n_cells, 1)),
                                  np.cumsum(d + c, axis=1)[:, :-1]], axis=1)
    at_risk = n_total - left_before
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(at_risk > 0, d / np.maximum(at_risk, 1e-12), 0.0)
    surv = np.cumprod(1.0 - h, axis=1)
    prev = np.concatenate([np.ones((n_cells, 1)), surv[:, :-1]], axis=1)
    pmf = prev - surv

    has_event = d > 0
    t_star = np.where(has_event.any(axis=1),
                      N_GRID - 1 - has_event[:, ::-1].argmax(axis=1), -1)
    residual = surv[:, -1]
    for k in np.flatnonzero(residual > 1e-12):
        ts = int(t_star[k])
        if ts >= N_GRID - 1:
            continue
        tail = parent_pmf[int(parent_of[k]), ts + 1:]
        s = tail.sum()
        if s <= 0:
            continue
        pmf[k, ts + 1:] += residual[k] * (tail / s)

    # RAW counts for eligibility
    raw_n = np.zeros(n_cells, dtype="float64")
    raw_e = np.zeros(n_cells, dtype="float64")
    np.add.at(raw_n, cell, 1.0)
    np.add.at(raw_e, cell[unc], 1.0)
    return pmf, raw_n, raw_e


# ===========================================================================
# 2. the arm
# ===========================================================================
class CalendarEmpiricalArm(c3.EmpiricalArmV3P):
    """`EmpiricalArmV3P` plus an optional calendar cell dimension.

    `cal_mode == "none"` makes `_codes` identical to the parent's and the arm is
    the round-3 object under another name, which is how the reference stays
    reproducible inside the round-4 harness."""

    cal_mode: str = "none"
    cur_season: int = 0
    half_life_days: float | None = None

    def _codes(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        codes = super()._codes(df)
        codes["cal_code"] = cal_code(df, self.cal_mode, self.cur_season)
        return codes


def _dims_for(fs: str, cal_mode: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """The cell grid, with the calendar dimension appended LAST.

    Appended last on purpose: `EmpiricalArmV3._keys` drops dimensions from the
    RIGHT, so a cell without enough current-season (or current-part) rows falls
    back to the pooled cell BEFORE it loses any of the state dimensions round 2
    and round 3 established. Nothing about the existing grid moves."""
    dims = tuple(ck.EMPIRICAL_DIMS[fs])
    sizes = tuple(ck.EMPIRICAL_DIM_SIZES[d] for d in dims)
    if cal_mode == "none":
        return dims, sizes
    return dims + ("cal_code",), sizes + (CAL_MODES[cal_mode],)


def recency_weights(game_date: pd.Series, refit_date, half_life_days: float) -> np.ndarray:
    """`0.5 ** (age_days / half_life)`, age measured back from the refit date.

    A row dated after the refit date cannot exist (the S1 fit is built from rows
    strictly before it) and would get a weight above 1; it is clipped to age 0
    and counted, not silently dropped, because a silent drop would hide a
    schedule defect."""
    d = pd.to_datetime(game_date).to_numpy(dtype="datetime64[D]")
    cut = np.datetime64(pd.Timestamp(refit_date).date(), "D")
    age = np.maximum((cut - d).astype("int64"), 0).astype("float64")
    return np.power(0.5, age / float(half_life_days))


def fit_empirical_v4(tr: pd.DataFrame, fs: str, parametrisation: str,
                     sr_floor_bucket: int = 0, cal_mode: str = "none",
                     cur_season: int = 0, half_life_days: float | None = None,
                     refit_date=None) -> CalendarEmpiricalArm:
    """One round-4 cell arm. `cal_mode="none"` and `half_life_days=None`
    reproduce `clock_v3._fit_empirical_p` exactly."""
    dims, sizes = _dims_for(fs, cal_mode)
    tempo = tr["tempo_prior_game"].to_numpy(dtype="float64")
    arm = CalendarEmpiricalArm(
        feature_set_name=fs, dims=dims, sizes=sizes,
        level_pmfs=[], level_counts=[], level_events=[],
        tempo_edges=(float(np.quantile(tempo, 1 / 3)), float(np.quantile(tempo, 2 / 3))),
        season_map={s: i for i, s in enumerate(sorted(int(x) for x in tr["season"].unique()))},
        sr_floor_bucket=sr_floor_bucket, features=ck.feature_set(fs),
        name=("empirical_km3_srfloor" if sr_floor_bucket else "empirical_km3")
             + f"_{parametrisation}_cal-{cal_mode}"
             + ("" if half_life_days is None else f"_hl{int(half_life_days)}"))
    arm.cal_mode = cal_mode
    arm.cur_season = int(cur_season)
    arm.half_life_days = half_life_days

    w = None
    if half_life_days is not None:
        if refit_date is None:
            raise ValueError("a recency weight needs the refit date it is measured from")
        w = recency_weights(tr["game_date"], refit_date, half_life_days)

    codes = arm._codes(tr)
    y = tr["duration_s"].to_numpy(dtype="int64")
    cen = tr["censored"].to_numpy(dtype=bool)

    pooled = np.zeros((1, N_GRID), dtype="float64")
    if w is None:
        np.add.at(pooled, (np.zeros(int((~cen).sum()), dtype="int64"), y[~cen]), 1.0)
    else:
        np.add.at(pooled, (np.zeros(int((~cen).sum()), dtype="int64"), y[~cen]), w[~cen])
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
        pmf, counts, events = kaplan_meier_pmf_v4(keys, y, cen, n_cells,
                                                  parent_pmf, parent_of, w)
        arm.level_pmfs.append(ck._normalise(pmf))
        arm.level_counts.append(counts)
        arm.level_events.append(events)
    arm.level_counts[0] = np.maximum(arm.level_counts[0], arm.min_cell)
    arm.level_events[0] = np.maximum(arm.level_events[0], arm.min_events)
    return arm


#: Round-4 arm tag -> (base arm, calendar mode, uses a recency half-life).
V4_ARMS: dict[str, dict] = {
    "recency": {"base": "empirical_km3_srfloor", "cal": "none", "recency": True},
    "curseason": {"base": "empirical_km3_srfloor", "cal": "curseason", "recency": False},
    "calpart": {"base": "empirical_km3_srfloor", "cal": "calpart", "recency": False},
    "nofloor": {"base": "empirical_km3", "cal": "none", "recency": False},
    "ref": {"base": "empirical_km3_srfloor", "cal": "none", "recency": False},
}


def fit_arm_v4(tag: str, train: pd.DataFrame, parametrisation: str = "P3",
               cur_season: int = 2025, half_life_days: float | None = None,
               refit_date=None):
    """Fit one round-4 arm, returned chain-safe (wrapped for P2/P3)."""
    spec = V4_ARMS[tag]
    fs = c3.P_FEATURES[parametrisation]
    tr = train if parametrisation == "P1" else c3.add_p_state(train.copy())
    floor_b = c3.SR_FLOOR_BUCKET if spec["base"] == "empirical_km3_srfloor" else 0
    hl = half_life_days if spec["recency"] else None
    inner = fit_empirical_v4(tr, fs, parametrisation, sr_floor_bucket=floor_b,
                             cal_mode=spec["cal"], cur_season=cur_season,
                             half_life_days=hl, refit_date=refit_date)
    return inner if parametrisation == "P1" else c3.StateWrapArm(inner, parametrisation)


#: Columns any round-4 fit reads, on top of `clock_v3.s1_fit_columns`.
V4_EXTRA_COLS: tuple[str, ...] = ("month",)


def s1_fit_columns_v4(base_arm: str) -> list[str]:
    cols = c3.s1_fit_columns("empirical_km3")
    for c in V4_EXTRA_COLS:
        if c not in cols:
            cols.append(c)
    return cols
