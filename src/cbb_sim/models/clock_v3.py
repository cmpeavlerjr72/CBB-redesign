"""clock_v3.py -- ROUND 3 of the L5 clock bake-off: censored (intended) duration.

Pre-registration: `docs/models/clock/experiments.md` section 8, PM-appended and
COMMITTED (0c7cc21) before any line of this module was written. Evidence for the
direction: `docs/LEARNINGS.md` L20 and
`docs/tests/clock_censoring_audit_2026-09-10.md`.

WHY A SEPARATE MODULE. Rounds 1 and 2 live in `clock.py` and their fitted
objects are pickled on disk and loaded by the engine
(`reference_not_adopted_*.pkl`, `v2_reference_not_adopted_*.pkl`). Changing a
dataclass or a fitted-object contract in `clock.py` would break those pickles
for a worker that is reading them right now. Everything shared -- the design
builder, the feature sets, the cell coding, the pmf/CRPS/PIT machinery, the
emergent chain -- is IMPORTED from `clock.py` and not reimplemented, so there is
still exactly one feature builder and one chain.

WHAT ROUND 3 CHANGES, and nothing else:

  1. THE CENSORING FLAG. `clock.build_design` sets
     `censored = (terminal_event == "end_period")`, 0.2148% of rows. The correct
     statement is "the possession consumed every second that was left", i.e.
     `end_clock <= 0`, 0.6751% of rows -- 3.1x as many, and 61.1% of possessions
     that start with under 5 seconds left. `attach_horn_censoring` joins that
     flag from the SIDE TABLE
     `data/processed/clock_censoring/censoring_v1_{season}.parquet` and never
     rewrites a possession table.

  2. THE KAPLAN-MEIER TAIL RULE (pre-registration 8.3). Round 2's
     `kaplan_meier_pmf` drops the unresolved survival and renormalises, which
     with a 61%-censored cell puts the dropped mass straight back onto the SHORT
     durations and silently reproduces the truncated law. `kaplan_meier_pmf_v3`
     distributes it over t > t* in proportion to the PARENT cell's pmf.

  3. THE METRIC. A test row is uncensored iff T < R, so it is a draw from
     T | T < R and the predictive law must be renormalised onto {0..R-1} before
     it is scored. `crps_trunc` / `pit_trunc` do that; `censored_loglik` is the
     proper likelihood that the censored rows enter.

  4. ONE NEW ARM CLASS: `XGBAftArm`, XGBoost `survival:aft` -- a censoring-aware
     TREE loss, which LightGBM's quantile objective has no form of.

SEAL: nothing here loads season 2026; `clock.fold_slices` guards every slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

from cbb_sim.models import clock as ck

DEFAULT_CENSOR_DIR = Path("data/processed/clock_censoring")
DEFAULT_UNIVERSE_V2 = Path("data/processed/games_universe_v2.parquet")
HALF_COMPLETENESS = DEFAULT_CENSOR_DIR / "half_clock_completeness_v1.parquet"

#: Minimum UNCENSORED exits a cell needs before the hierarchical fallback fires,
#: on top of round 2's `EMPIRICAL_MIN_CELL = 300` rows (pre-registration 8.3).
EMPIRICAL_MIN_EVENTS = 100

#: Round-2 fine-bucket index of "45-59" -- the floor arm A2's single parameter,
#: fixed by the pre-registration, not searched.
SR_FLOOR_BUCKET = 5

#: Continuity offset, the same 0.5 the parametric arms use.
CONTINUITY = ck.CONTINUITY

GRID = ck.GRID
N_GRID = ck.N_GRID
PMF_FLOOR = ck.PMF_FLOOR

ARMS_V3: tuple[str, ...] = (
    "empirical_km3", "empirical_km3_srfloor", "gamma_aft", "lognormal_aft",
    "hazard3", "xgb_aft", "lgbm_quantile_r2", "empirical_r2", "gamma_r2",
)
#: Which censoring flag each arm trains on (pre-registration 8.2).
ARM_FLAG: dict[str, str] = {
    "empirical_km3": "horn", "empirical_km3_srfloor": "horn", "gamma_aft": "horn",
    "lognormal_aft": "horn", "hazard3": "horn", "xgb_aft": "horn",
    "lgbm_quantile_r2": "old", "empirical_r2": "old", "gamma_r2": "old",
}
ARM_FEATURES_V3: dict[str, str] = {
    "empirical_km3": "R2_dummy", "empirical_km3_srfloor": "R2_dummy",
    "gamma_aft": "R2_dummy", "lognormal_aft": "R2_dummy", "hazard3": "R2_dummy",
    "xgb_aft": "R2_tree", "lgbm_quantile_r2": "R2_tree",
    "empirical_r2": "R2_dummy", "gamma_r2": "R2_dummy",
}
TREE_ARMS_V3: tuple[str, ...] = ("xgb_aft", "lgbm_quantile_r2")
SIMPLICITY_RANK_V3: dict[str, int] = {
    "empirical_km3_srfloor": 0, "empirical_km3": 1, "empirical_r2": 1,
    "gamma_aft": 2, "lognormal_aft": 2, "gamma_r2": 2, "hazard3": 3,
    "xgb_aft": 4, "lgbm_quantile_r2": 4,
}
STOCHASTIC_ARMS: tuple[str, ...] = ("xgb_aft", "lgbm_quantile_r2")


# ===========================================================================
# 1. The censoring flag
# ===========================================================================
def attach_horn_censoring(design: pd.DataFrame, seasons: list[int],
                          censor_dir: Path | str = DEFAULT_CENSOR_DIR) -> tuple[pd.DataFrame, dict]:
    """Add `censored_horn` (the correct flag) and keep `censored_old` (rounds
    1-2's) on a `clock.build_design` frame.

    The flag is joined on (game_id, period, poss_index) from the audit's side
    table. Nothing in `data/processed/possessions*/` is read for writing and
    nothing is rewritten."""
    frames = []
    for s in sorted({int(x) for x in seasons}):
        p = Path(censor_dir) / f"censoring_v1_{s}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"missing censoring side table {p} -- run scripts/diag_clock_censoring_v1.py")
        frames.append(pd.read_parquet(
            p, columns=["game_id", "period", "poss_index", "censored_horn", "flag_end_period"]))
    side = pd.concat(frames, ignore_index=True)
    side["period"] = side["period"].astype("float32")

    out = design.copy()
    out["censored_old"] = out["censored"].to_numpy(dtype=bool)
    out["period"] = out["period"].astype("float32")
    n_before = len(out)
    out = out.merge(side, on=["game_id", "period", "poss_index"], how="left", validate="1:1")
    if len(out) != n_before:
        raise ValueError("censoring side-table join changed the row count")
    miss = int(out["censored_horn"].isna().sum())
    if miss:
        raise ValueError(f"{miss} design rows have no censoring side-table row")
    out["censored_horn"] = out["censored_horn"].to_numpy(dtype=bool)

    diag = {
        "rows": int(len(out)),
        "censored_old_pct": round(100.0 * float(out["censored_old"].mean()), 4),
        "censored_horn_pct": round(100.0 * float(out["censored_horn"].mean()), 4),
        "old_and_not_horn": int((out["censored_old"] & ~out["censored_horn"]).sum()),
        "horn_and_not_old": int((out["censored_horn"] & ~out["censored_old"]).sum()),
    }
    # A censored row must have consumed the clock it started with; assert it
    # rather than trust it.
    bad = int((out["censored_horn"] &
               (out["duration_s"].to_numpy() < out["seconds_remaining"].to_numpy() - 2)).sum())
    diag["censored_rows_not_consuming_clock"] = bad
    return out, diag


def set_flag(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Point the shared `censored` column at the requested flag. Every arm in
    `clock.py` reads `censored`, so this is the ONE place round 3 switches
    between the corrected flag and rounds 1-2's."""
    out = df.copy()
    return set_flag_inplace(out, flag)


def set_flag_inplace(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    """`set_flag` without the 1 GB copy, for the trainer's own frame. Four other
    workers share this machine, so the design is switched in place rather than
    duplicated nine times."""
    if flag not in ("horn", "old"):
        raise ValueError(flag)
    src = "censored_horn" if flag == "horn" else "censored_old"
    df["censored"] = df[src].to_numpy(dtype=bool)
    return df


def pred_mean_by_band(arm, te: pd.DataFrame, chunk: int = 200_000) -> pd.DataFrame:
    """Round 1's D1 diagnostic, re-read for round 3: the PREDICTED mean duration
    by the seconds-remaining band the possession started in, beside the actual
    observed mean and the horn rate.

    This is where the censoring fix has to show up. Under the corrected flag the
    predicted INTENDED duration inside the last 10 seconds must be far larger
    than the observed truncated mean; if it is not, the arm has not changed."""
    bands = [0, 5, 10, 20, 30, 45, 60, 90, 1201]
    m = np.empty(len(te), dtype="float64")
    for a in range(0, len(te), chunk):
        p = arm.pmf(te.iloc[a:a + chunk])
        m[a:a + len(p)] = ck.pmf_mean_sd(p)[0]
    band = pd.cut(te["seconds_remaining"], bands, right=False)
    out = pd.DataFrame({
        "band": band.to_numpy(),
        "duration_s": te["duration_s"].to_numpy(),
        "censored": te["censored"].to_numpy(dtype=bool),
        "pred_mean": m,
    })
    g = out.groupby("band", observed=True, as_index=False).agg(
        n=("pred_mean", "size"), horn_rate=("censored", "mean"),
        actual_mean_observed=("duration_s", "mean"), pred_mean_intended=("pred_mean", "mean"))
    g["band"] = g["band"].astype(str)
    return g


# ===========================================================================
# 2. Kaplan-Meier with the pre-registered tail rule
# ===========================================================================
def kaplan_meier_pmf_v3(cell: np.ndarray, y: np.ndarray, censored: np.ndarray,
                        n_cells: int, parent_pmf: np.ndarray,
                        parent_of: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrete-time Kaplan-Meier per cell, with the round-3 tail rule.

    Identical to `clock.kaplan_meier_pmf` up to the point where the survival
    that never resolves inside the cell has to go somewhere. Round 2 dropped it
    and renormalised -- which, in a cell where 61% of the rows are censored,
    hands that mass back to the SHORT durations and rebuilds the truncated law
    this round exists to remove. Round 3 instead distributes it over t > t*
    (t* = the last uncensored exit in the cell) in proportion to the PARENT
    cell's pmf restricted to t > t*, which is the next-coarser estimate of the
    same conditional law.

    Returns (pmf, n_rows, n_events) per cell."""
    d = np.zeros((n_cells, N_GRID), dtype="float64")
    c = np.zeros((n_cells, N_GRID), dtype="float64")
    np.add.at(d, (cell[~censored], y[~censored]), 1.0)
    np.add.at(c, (cell[censored], y[censored]), 1.0)
    n_total = (d + c).sum(axis=1, keepdims=True)
    left_before = np.concatenate([np.zeros((n_cells, 1)), np.cumsum(d + c, axis=1)[:, :-1]], axis=1)
    at_risk = n_total - left_before
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(at_risk > 0, d / np.maximum(at_risk, 1e-12), 0.0)
    surv = np.cumprod(1.0 - h, axis=1)
    prev = np.concatenate([np.ones((n_cells, 1)), surv[:, :-1]], axis=1)
    pmf = prev - surv

    has_event = d > 0
    t_star = np.where(has_event.any(axis=1), N_GRID - 1 - has_event[:, ::-1].argmax(axis=1), -1)
    residual = surv[:, -1]

    for k in np.flatnonzero(residual > 1e-12):
        ts = int(t_star[k])
        if ts >= N_GRID - 1:
            continue  # nowhere above t* to put it; _normalise drops it
        tail = parent_pmf[int(parent_of[k]), ts + 1:]
        s = tail.sum()
        if s <= 0:
            continue
        pmf[k, ts + 1:] += residual[k] * (tail / s)

    return pmf, n_total.ravel(), d.sum(axis=1)


@dataclass
class EmpiricalArmV3:
    """Round-3 cell-resampling arm: round 2's grid, the corrected censoring
    flag, `kaplan_meier_pmf_v3`'s tail rule, and an events-per-cell minimum.

    `sr_floor_bucket` > 0 is arm A2: the fine clock bucket is FLOORED, so every
    row with fewer than 45 seconds left is served the 45-59 s cell's
    intended-duration law and the whole end-of-period effect is left to the
    engine's truncation. It is a falsification arm, and it is also strictly
    simpler than A1 (fewer distinct cells), which is why the pre-registered
    tie-break ranks it first."""

    feature_set_name: str
    dims: tuple[str, ...]
    sizes: tuple[int, ...]
    level_pmfs: list[np.ndarray]
    level_counts: list[np.ndarray]
    level_events: list[np.ndarray]
    tempo_edges: tuple[float, float]
    season_map: dict[int, int]
    min_cell: int = ck.EMPIRICAL_MIN_CELL
    min_events: int = EMPIRICAL_MIN_EVENTS
    sr_floor_bucket: int = 0
    features: list[str] = field(default_factory=list)
    dropped_zero_variance: list[str] = field(default_factory=list)
    name: str = "empirical_km3"

    def _codes(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        sr = df["seconds_remaining"].to_numpy()
        bucket = ck.r2_bucket_id(sr)
        if self.sr_floor_bucket:
            bucket = np.maximum(bucket, self.sr_floor_bucket)
        prev = df["prev_end"].map(ck.PREV_END_INDEX).to_numpy()
        tempo = np.searchsorted(np.asarray(self.tempo_edges),
                                df["tempo_prior_game"].to_numpy(), side="right")
        return {
            "prev_end_code": prev.astype("int64"),
            "r2_bucket_code": bucket.astype("int64"),
            "r2_period_type": ck.r2_period_type(df["period"].to_numpy()),
            "r2_score_state": ck.r2_score_state(df["score_diff"].to_numpy()),
            "tempo_tercile": tempo.astype("int64"),
        }

    def _keys(self, codes: dict[str, np.ndarray], level: int) -> np.ndarray:
        if level == 0:
            return np.zeros(len(next(iter(codes.values()))), dtype="int64")
        return np.ravel_multi_index([codes[d] for d in self.dims[:level]], self.sizes[:level])

    def _eligible(self, level: int, keys: np.ndarray) -> np.ndarray:
        return ((self.level_counts[level][keys] >= self.min_cell) &
                (self.level_events[level][keys] >= self.min_events))

    def level_of(self, df: pd.DataFrame) -> np.ndarray:
        codes = self._codes(df)
        out = np.full(len(df), -1, dtype="int64")
        for lv in range(len(self.dims), -1, -1):
            todo = out < 0
            if not todo.any():
                break
            keys = self._keys({k: v[todo] for k, v in codes.items()}, lv)
            idx = np.flatnonzero(todo)[self._eligible(lv, keys)]
            out[idx] = lv
        return out

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        codes = self._codes(df)
        n = len(df)
        out = np.zeros((n, N_GRID), dtype="float64")
        done = np.zeros(n, dtype=bool)
        for lv in range(len(self.dims), -1, -1):
            todo = ~done
            if not todo.any():
                break
            keys = self._keys({k: v[todo] for k, v in codes.items()}, lv)
            ok = self._eligible(lv, keys)
            idx = np.flatnonzero(todo)[ok]
            if len(idx):
                out[idx] = self.level_pmfs[lv][keys[ok]]
                done[idx] = True
        if not done.all():
            out[~done] = self.level_pmfs[0][0]
        return ck._normalise(out)


def fit_empirical_v3(train: pd.DataFrame, feature_set_name: str = "R2_dummy",
                     min_cell: int = ck.EMPIRICAL_MIN_CELL,
                     min_events: int = EMPIRICAL_MIN_EVENTS,
                     sr_floor_bucket: int = 0, seed: int = 0) -> EmpiricalArmV3:
    dims = ck.EMPIRICAL_DIMS[feature_set_name]
    sizes = tuple(ck.EMPIRICAL_DIM_SIZES[d] for d in dims)
    tempo = train["tempo_prior_game"].to_numpy(dtype="float64")
    tempo_edges = (float(np.quantile(tempo, 1 / 3)), float(np.quantile(tempo, 2 / 3)))
    seasons = sorted(int(s) for s in train["season"].unique())

    arm = EmpiricalArmV3(
        feature_set_name=feature_set_name, dims=dims, sizes=sizes,
        level_pmfs=[], level_counts=[], level_events=[], tempo_edges=tempo_edges,
        season_map={s: i for i, s in enumerate(seasons)}, min_cell=min_cell,
        min_events=min_events, sr_floor_bucket=sr_floor_bucket,
        features=ck.feature_set(feature_set_name),
        name="empirical_km3_srfloor" if sr_floor_bucket else "empirical_km3",
    )
    codes = arm._codes(train)
    y = train["duration_s"].to_numpy(dtype="int64")
    cen = train["censored"].to_numpy(dtype=bool)

    # Level 0's "parent" is the pooled UNCENSORED empirical law -- the only
    # estimate of the tail that exists once every dimension has been dropped.
    pooled = np.zeros((1, N_GRID), dtype="float64")
    np.add.at(pooled, (np.zeros((~cen).sum(), dtype="int64"), y[~cen]), 1.0)
    pooled = ck._normalise(pooled)

    for lv in range(len(dims) + 1):
        n_cells = 1 if lv == 0 else int(np.prod(sizes[:lv]))
        keys = arm._keys(codes, lv)
        if lv == 0:
            parent_pmf, parent_of = pooled, np.zeros(1, dtype="int64")
        else:
            parent_pmf = arm.level_pmfs[lv - 1]
            parent_of = (np.arange(n_cells, dtype="int64") // sizes[lv - 1]) if lv > 1 \
                else np.zeros(n_cells, dtype="int64")
        pmf, counts, events = kaplan_meier_pmf_v3(keys, y, cen, n_cells, parent_pmf, parent_of)
        arm.level_pmfs.append(ck._normalise(pmf))
        arm.level_counts.append(counts)
        arm.level_events.append(events)
    arm.level_counts[0] = np.maximum(arm.level_counts[0], min_cell)
    arm.level_events[0] = np.maximum(arm.level_events[0], min_events)
    return arm


# ===========================================================================
# 3. XGBoost AFT -- a censoring-aware tree loss
# ===========================================================================
#: Tree complexity is round 2's F1-chosen pair, carried over verbatim so round 3
#: adds no search (pre-registration 8.2).
XGB_PARAMS = dict(
    objective="survival:aft",
    eval_metric="aft-nloglik",
    tree_method="hist",
    max_depth=0,
    grow_policy="lossguide",
    max_leaves=63,
    min_child_weight=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=4,
)
XGB_ROUNDS = 400
AFT_DISTS: tuple[str, ...] = ("normal", "logistic", "extreme")


def _aft_logcdf(dist: str, z: np.ndarray) -> np.ndarray:
    if dist == "normal":
        return special.log_ndtr(z)
    if dist == "logistic":
        return -np.logaddexp(0.0, -z)
    if dist == "extreme":  # smallest-extreme-value: F = 1 - exp(-exp(z))
        return np.log1p(-np.exp(-np.exp(np.clip(z, -50, 30))) + 1e-300)
    raise ValueError(dist)


def _aft_cdf(dist: str, z: np.ndarray) -> np.ndarray:
    if dist == "normal":
        return special.ndtr(z)
    if dist == "logistic":
        return special.expit(z)
    if dist == "extreme":
        return 1.0 - np.exp(-np.exp(np.clip(z, -50, 30)))
    raise ValueError(dist)


def _aft_scale_mle(dist: str, mu: np.ndarray, y: np.ndarray, cen: np.ndarray) -> float:
    """1-D MLE of the AFT scale given the fitted locations.

    XGBoost takes `aft_loss_distribution_scale` as a HYPERPARAMETER and does not
    fit it, so it has to come from somewhere. It is fitted here by profile
    likelihood on the TRAINING rows -- a fitted parameter, not a searched one,
    and not a post-hoc adjustment to any output."""
    logt = np.log(np.maximum(y, 0.0) + CONTINUITY)
    grid = np.exp(np.linspace(np.log(0.05), np.log(3.0), 60))
    best, best_ll = grid[0], -np.inf
    for s in grid:
        z = (logt - mu) / s
        ll = np.empty_like(z)
        if dist == "normal":
            ll[~cen] = -0.5 * z[~cen] ** 2 - 0.5 * np.log(2 * np.pi) - np.log(s)
        elif dist == "logistic":
            ll[~cen] = -z[~cen] - 2 * np.logaddexp(0.0, -z[~cen]) - np.log(s)
        else:
            zz = np.clip(z[~cen], -50, 30)
            ll[~cen] = zz - np.exp(zz) - np.log(s)
        sf = np.clip(1.0 - _aft_cdf(dist, z[cen]), 1e-300, 1.0)
        ll[cen] = np.log(sf)
        tot = float(ll.mean())
        if tot > best_ll:
            best_ll, best = tot, float(s)
    return best


@dataclass
class XGBAftArm:
    """XGBoost `survival:aft`: log T = f(x) + sigma * eps, eps ~ `dist`.

    CENSORING is native to the objective -- a right-censored row enters as the
    interval [c, inf), which is exactly "the possession had not ended when the
    horn went". This is the tree arm the pre-registration asks for: LightGBM's
    quantile objective has no censored form, so round 2's tree arm had to
    EXCLUDE censored rows, and that exclusion is one of the things round 3
    exists to remove."""

    feature_set_name: str
    features: list[str]
    dropped_zero_variance: list[str]
    booster: object
    dist: str
    scale: float
    n_obs: int
    n_censored: int
    name: str = "xgb_aft"

    def _mu(self, df: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb
        x = ck._matrix(df, self.features)
        d = xgb.DMatrix(x, feature_names=list(self.features), nthread=4)
        return np.log(np.maximum(self.booster.predict(d), 1e-9))

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        mu = self._mu(df)[:, None]
        edges = np.concatenate([[CONTINUITY], GRID[1:] + CONTINUITY]).astype("float64")
        z = (np.log(edges)[None, :] - mu) / self.scale
        cdf = _aft_cdf(self.dist, z)
        out = np.empty((len(mu), N_GRID), dtype="float64")
        out[:, 0] = cdf[:, 0]
        out[:, 1:] = np.diff(cdf, axis=1)
        return ck._normalise(out)


def fit_xgb_aft(train: pd.DataFrame, feature_set_name: str = "R2_tree",
                dist: str = "normal", seed: int = 0,
                n_rounds: int = XGB_ROUNDS, nthread: int = 4) -> XGBAftArm:
    import xgboost as xgb

    feats = ck.feature_set(feature_set_name)
    keep, dropped = ck._drop_zero_variance(train, feats)
    x = ck._matrix(train, keep)
    y = train["duration_s"].to_numpy(dtype="float64")
    cen = train["censored"].to_numpy(dtype=bool)

    lower = y + CONTINUITY
    upper = np.where(cen, np.inf, y + CONTINUITY)
    d = xgb.DMatrix(x, feature_names=keep, nthread=nthread)
    d.set_float_info("label_lower_bound", lower)
    d.set_float_info("label_upper_bound", upper)

    params = dict(XGB_PARAMS)
    params.update(aft_loss_distribution=dist, aft_loss_distribution_scale=1.0,
                  seed=int(seed), nthread=int(nthread))
    bst = xgb.train(params, d, num_boost_round=n_rounds)
    mu = np.log(np.maximum(bst.predict(d), 1e-9))
    scale = _aft_scale_mle(dist, mu, y, cen)

    # Second pass with the fitted scale: the gradient of the AFT loss depends on
    # it, so a scale fitted against a nominal-scale fit is not the same model.
    params["aft_loss_distribution_scale"] = scale
    bst = xgb.train(params, d, num_boost_round=n_rounds)
    mu = np.log(np.maximum(bst.predict(d), 1e-9))
    scale = _aft_scale_mle(dist, mu, y, cen)

    return XGBAftArm(feature_set_name=feature_set_name, features=keep,
                     dropped_zero_variance=dropped, booster=bst, dist=dist,
                     scale=float(scale), n_obs=int(len(y)), n_censored=int(cen.sum()))


def fit_arm_v3(arm: str, train: pd.DataFrame, seed: int = 0, **kw):
    """One dispatch for all nine arms. Every arm reads `censored`, which
    `set_flag` has already pointed at that arm's pre-registered flag."""
    fs = ARM_FEATURES_V3[arm]
    if arm == "empirical_km3":
        return fit_empirical_v3(train, fs, seed=seed)
    if arm == "empirical_km3_srfloor":
        return fit_empirical_v3(train, fs, sr_floor_bucket=SR_FLOOR_BUCKET, seed=seed)
    if arm == "gamma_aft":
        return ck.fit_parametric(train, fs, family="gamma", seed=seed)
    if arm == "lognormal_aft":
        return ck.fit_parametric(train, fs, family="lognormal", seed=seed)
    if arm == "hazard3":
        return ck.fit_hazard(train, fs, seed=seed)
    if arm == "xgb_aft":
        return fit_xgb_aft(train, fs, dist=kw.get("dist", "normal"), seed=seed)
    if arm == "lgbm_quantile_r2":
        return ck.fit_quantile_tree(train, fs, seed=seed, n_jobs=kw.get("n_jobs", 4),
                                    params=kw.get("tree_params"))
    if arm == "empirical_r2":
        return ck.fit_empirical(train, fs, seed=seed)
    if arm == "gamma_r2":
        return ck.fit_parametric(train, fs, family="gamma", seed=seed)
    raise ValueError(f"unknown round-3 arm: {arm}")


# ===========================================================================
# 4. Censored metrics
# ===========================================================================
def truncate_pmf(pmf: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renormalise each row's predictive law onto {0, ..., R-1}.

    An UNCENSORED observation satisfies T < R by definition of the observation
    mechanism, so it is a draw from T | T < R. Scoring it against the
    unconditional law of T is improper and would reward exactly the arms whose
    unconditional law is already truncated -- the defect round 3 removes. For
    R > DURATION_CAP (91.8% of rows) the mask is all-True and this is a no-op.

    Returns (truncated pmf, mask of rows whose predictive mass below R was
    numerically zero -- reported, never silently replaced)."""
    mask = GRID[None, :] < np.asarray(r, dtype="float64")[:, None]
    p = np.where(mask, np.asarray(pmf, dtype="float64"), 0.0)
    s = p.sum(axis=1, keepdims=True)
    dead = (s <= PMF_FLOOR).ravel()
    if dead.any():
        # No invented distribution: floor the allowed cells and renormalise, so
        # the row scores badly rather than being quietly excused.
        p[dead] = np.where(mask[dead], PMF_FLOOR, 0.0)
        s = p.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return p / s, dead


def censored_loglik_rows(pmf: np.ndarray, y: np.ndarray, censored: np.ndarray,
                         r: np.ndarray) -> np.ndarray:
    """log P(T = d) on uncensored rows, log P(T >= R) on censored rows. The
    proper likelihood under right censoring, and the only metric the censored
    rows enter."""
    p = np.clip(np.asarray(pmf, dtype="float64"), PMF_FLOOR, None)
    out = np.empty(len(p), dtype="float64")
    unc = ~censored
    out[unc] = np.log(p[unc, np.asarray(y, dtype="int64")[unc]])
    if censored.any():
        tail = (GRID[None, :] >= np.asarray(r, dtype="float64")[censored][:, None])
        out[censored] = np.log(np.clip((p[censored] * tail).sum(axis=1), PMF_FLOOR, None))
    return out


def score_arm_v3(arm, te: pd.DataFrame, chunk: int = 100_000,
                 pit_seed: int = 20260910) -> dict:
    """One blind scoring path for every arm (pre-registration 8.4).

    Produces, per test row: CRPS of the TRUNCATED law (primary, uncensored rows
    only), CRPS of the untruncated law over all rows scored as complete (the
    labelled round-2 bridge), the censored log-likelihood, and the randomised
    PIT of the truncated law."""
    y = te["duration_s"].to_numpy(dtype="int64")
    cen = te["censored"].to_numpy(dtype=bool)
    r = te["seconds_remaining"].to_numpy(dtype="float64")
    n = len(te)

    crps_t = np.full(n, np.nan)
    crps_r2 = np.empty(n)
    ll = np.empty(n)
    pit_row = np.full(n, np.nan)
    pred_mean = np.empty(n)
    cens_surv = np.full(n, np.nan)
    n_dead = 0

    rng = np.random.default_rng(pit_seed)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        blk = te.iloc[a:b]
        pmf = arm.pmf(blk)
        yy, cc, rr = y[a:b], cen[a:b], r[a:b]

        crps_r2[a:b] = ck.crps(pmf, yy)
        ll[a:b] = censored_loglik_rows(pmf, yy, cc, rr)
        m, _ = ck.pmf_mean_sd(pmf)
        pred_mean[a:b] = m
        if cc.any():
            tail = (GRID[None, :] >= rr[cc][:, None])
            cens_surv[np.arange(a, b)[cc]] = (pmf[cc] * tail).sum(axis=1)

        tp, dead = truncate_pmf(pmf, rr)
        n_dead += int((dead & ~cc).sum())
        unc = ~cc
        if unc.any():
            idx = np.arange(a, b)[unc]
            crps_t[idx] = ck.crps(tp[unc], yy[unc])
            v = rng.random(int(unc.sum()))
            pit_row[idx] = ck.pit(tp[unc], yy[unc], v)

    unc = ~cen
    ks_D, ks_p = ck.ks_uniform(pit_row[unc])
    return {
        "crps_trunc": float(np.nanmean(crps_t[unc])),
        "crps_r2def": float(crps_r2.mean()),
        "censored_loglik": float(ll.mean()),
        "censored_loglik_uncensored_only": float(ll[unc].mean()),
        "censored_loglik_censored_only": float(ll[cen].mean()) if cen.any() else float("nan"),
        "pooled_pit_ks_D": float(ks_D), "pooled_pit_ks_p": float(ks_p),
        "pred_mean_duration": float(pred_mean.mean()),
        "actual_mean_duration_uncensored": float(y[unc].mean()),
        "censored_mean_pred_survival": float(np.nanmean(cens_surv)) if cen.any() else float("nan"),
        "n_test": int(n), "n_censored": int(cen.sum()),
        "n_zero_mass_below_R": int(n_dead),
        "_crps_trunc_rows": crps_t,
        "_pit_rows": pit_row,
        "_loglik_rows": ll,
    }


def pit_by_cell_v3(te: pd.DataFrame, pit_row: np.ndarray,
                   min_cell: int = ck.PIT_MIN_CELL) -> pd.DataFrame:
    """PIT K-S by the pre-registered reporting cell (previous end type x
    round-2 fine bucket), uncensored rows only."""
    cen = te["censored"].to_numpy(dtype=bool)
    cell = (te["prev_end"].astype(str) + " | " +
            pd.Series(np.asarray(ck.R2_SR_LABELS)[ck.r2_bucket_id(
                te["seconds_remaining"].to_numpy())], index=te.index))
    df = pd.DataFrame({"cell": cell.to_numpy(), "u": pit_row, "cen": cen})
    df = df[~df["cen"] & np.isfinite(df["u"])]
    rows = []
    for c, g in df.groupby("cell", sort=True):
        D, p = ck.ks_uniform(g["u"].to_numpy())
        rows.append({"cell": c, "n": int(len(g)), "ks_D": float(D), "ks_p": float(p),
                     "powered": bool(len(g) >= min_cell),
                     "leak_sized": bool(len(g) >= min_cell and D > ck.PIT_KS_D_GATE)})
    return pd.DataFrame(rows).sort_values("ks_D", ascending=False).reset_index(drop=True)


def segment_table(te: pd.DataFrame, crps_rows: np.ndarray, ll_rows: np.ndarray,
                  pred_mean: np.ndarray | None = None) -> pd.DataFrame:
    """Pre-registration 8.6: by half (H1/H2/OT), by score-margin bucket, by
    season. The shot-clock era is 30 s in every season 2022-2025, so that
    segment is degenerate and season stands in its place."""
    cen = te["censored"].to_numpy(dtype=bool)
    per = te["period"].to_numpy(dtype="float64")
    half = np.where(per == 1, "H1", np.where(per == 2, "H2", "OT"))
    base = pd.DataFrame({
        "half": half,
        "score_bucket": te["score_bucket"].to_numpy(),
        "season": te["season"].to_numpy(),
        "crps_trunc": crps_rows,
        "loglik": ll_rows,
        "censored": cen,
    })
    out = []
    for name in ("half", "score_bucket", "season"):
        g = base.groupby(name, dropna=False)
        t = pd.DataFrame({
            "segment": name,
            "level": g.size().index.astype(str),
            "n": g.size().to_numpy(),
            "n_censored": g["censored"].sum().to_numpy(),
            "crps_trunc": g["crps_trunc"].mean().to_numpy(),
            "censored_loglik": g["loglik"].mean().to_numpy(),
        })
        out.append(t)
    return pd.concat(out, ignore_index=True)


# ===========================================================================
# 5. Clock-complete gate universe and the emergent read
# ===========================================================================
def load_clock_complete(path: Path | str = HALF_COMPLETENESS) -> pd.DataFrame:
    """(game_id, period, clock_complete) from the audit's side table -- the
    three-part definition, not round 2's tail-only one."""
    hc = pd.read_parquet(path, columns=["game_id", "period", "clock_complete"])
    hc["period"] = hc["period"].astype("int64")
    return hc


def actual_end_of_half_cc(design: pd.DataFrame, hc: pd.DataFrame) -> pd.DataFrame:
    """`clock.actual_end_of_half` with `clock_complete` REPLACED by the audit's
    three-part flag. The column name is kept so `clock.clock_complete_read` and
    `clock.emergent_report` consume it unchanged."""
    ah = ck.actual_end_of_half(design)
    ah = ah.drop(columns=["clock_complete"]).merge(
        hc.rename(columns={"period": "half"}), on=["game_id", "half"], how="left")
    ah["clock_complete"] = ah["clock_complete"].fillna(False).astype(bool)
    return ah


def emergent_cc(res: ck.ChainResult, actual_half: pd.DataFrame,
                month_min_games: int = ck.G1_MONTH_MIN_GAMES) -> dict:
    """G1 on CLOCK-COMPLETE GAMES -- the round-3 selection gate -- plus the
    end-of-half read on clock-complete halves. The all-games read stays
    available through `clock.emergent_report` and is reported beside this one,
    labelled, never as the gate."""
    per_game_cc = actual_half.groupby("game_id")["clock_complete"].all()
    good = set(per_game_cc[per_game_cc].index)
    pg = res.per_game[res.per_game["game_id"].isin(good)].dropna(subset=["sim_poss", "actual_poss"])
    sim, act = pg["sim_poss"].to_numpy(), pg["actual_poss"].to_numpy()

    months = []
    for m, g in pg.groupby("month", sort=True):
        powered = len(g) >= month_min_games
        dm = float(g["sim_poss"].mean() - g["actual_poss"].mean())
        ds = float(g["sim_poss"].std(ddof=1) - g["actual_poss"].std(ddof=1))
        months.append({"month": int(m), "n_games": int(len(g)), "powered": bool(powered),
                       "sim_mean": round(float(g["sim_poss"].mean()), 3),
                       "actual_mean": round(float(g["actual_poss"].mean()), 3),
                       "mean_delta": round(dm, 3), "sd_delta": round(ds, 3),
                       "pass": bool(powered and abs(dm) <= ck.G1_MEAN_TOL and abs(ds) <= ck.G1_SD_TOL)})

    ah = actual_half[actual_half["clock_complete"]]
    keys = set(zip(ah["game_id"].to_numpy(), ah["half"].to_numpy(), strict=False))
    sh = res.per_half
    sel = np.array([(g, h) in keys for g, h in
                    zip(sh["game_id"].to_numpy(), sh["half"].to_numpy(), strict=False)])
    a_share, a_dur = ck.eoh_stats(ah)
    s_share, s_dur = ck.eoh_stats(sh[sel])

    mean_delta = float(sim.mean() - act.mean())
    sd_delta = float(sim.std(ddof=1) - act.std(ddof=1))
    powered = [m for m in months if m["powered"]]
    ks_D, ks_p = stats.ks_2samp(sim, act)[:2]
    return {
        "n_games": int(len(pg)), "n_halves": int(len(ah)),
        "sim_mean": round(float(sim.mean()), 3), "actual_mean": round(float(act.mean()), 3),
        "mean_delta": round(mean_delta, 3),
        "sim_sd": round(float(sim.std(ddof=1)), 3), "actual_sd": round(float(act.std(ddof=1)), 3),
        "sd_delta": round(sd_delta, 3),
        "count_ks_D": round(float(ks_D), 4), "count_ks_p": float(ks_p),
        "months": months, "n_powered_months": len(powered),
        "months_pass": int(sum(m["pass"] for m in powered)),
        "g1_pass": bool(abs(mean_delta) <= ck.G1_MEAN_TOL and abs(sd_delta) <= ck.G1_SD_TOL
                        and bool(powered) and all(m["pass"] for m in powered)),
        "eoh_actual_share": round(a_share, 4), "eoh_sim_share": round(s_share, 4),
        "eoh_share_gap": round(s_share - a_share, 5),
        "eoh_actual_dur": round(a_dur, 3), "eoh_sim_dur": round(s_dur, 3),
        "eoh_duration_gap": round(s_dur - a_dur, 4),
    }


def eoh_cc_stats(res: ck.ChainResult, actual_half: pd.DataFrame) -> tuple[float, float]:
    ah = actual_half[actual_half["clock_complete"]]
    keys = set(zip(ah["game_id"].to_numpy(), ah["half"].to_numpy(), strict=False))
    sh = res.per_half
    sel = np.array([(g, h) in keys for g, h in
                    zip(sh["game_id"].to_numpy(), sh["half"].to_numpy(), strict=False)])
    return ck.eoh_stats(sh[sel])


def responsiveness(res: ck.ChainResult, design: pd.DataFrame, n_q: int = 5) -> pd.DataFrame:
    """CLAUDE.md standing rule: the emergent count must SLOPE with the pregame
    tempo prior, not sit flat at the league mean."""
    tp = design.groupby("game_id")["tempo_prior_game"].first().rename("tempo_prior")
    pg = res.per_game.merge(tp, on="game_id", how="left").dropna(
        subset=["sim_poss", "actual_poss", "tempo_prior"])
    pg["q"] = pd.qcut(pg["tempo_prior"], n_q, labels=False, duplicates="drop") + 1
    g = pg.groupby("q", as_index=False).agg(
        n=("sim_poss", "size"), sim_mean=("sim_poss", "mean"), actual_mean=("actual_poss", "mean"))
    g["delta"] = g["sim_mean"] - g["actual_mean"]
    span_sim = float(g["sim_mean"].iloc[-1] - g["sim_mean"].iloc[0])
    span_act = float(g["actual_mean"].iloc[-1] - g["actual_mean"].iloc[0])
    g["span_sim"] = span_sim
    g["span_actual"] = span_act
    g["slope_ratio"] = span_sim / span_act if span_act else np.nan
    g["steps_agreeing"] = int(sum(
        (np.sign(np.diff(g["sim_mean"].to_numpy())) == np.sign(np.diff(g["actual_mean"].to_numpy())))))
    return g


# ===========================================================================
# 6. Binned lookup-table export (deliverable, pre-registration 8.9)
# ===========================================================================
LOOKUP_DIMS: tuple[str, ...] = (
    "prev_end_code", "r2_bucket_code", "r2_period_type", "r2_score_state", "tempo_tercile")
LOOKUP_SIZES: tuple[int, ...] = tuple(ck.EMPIRICAL_DIM_SIZES[d] for d in LOOKUP_DIMS)


def lookup_codes(df: pd.DataFrame, tempo_edges: tuple[float, float]) -> np.ndarray:
    codes = [
        df["prev_end"].map(ck.PREV_END_INDEX).to_numpy().astype("int64"),
        ck.r2_bucket_id(df["seconds_remaining"].to_numpy()),
        ck.r2_period_type(df["period"].to_numpy()),
        ck.r2_score_state(df["score_diff"].to_numpy()),
        np.searchsorted(np.asarray(tempo_edges), df["tempo_prior_game"].to_numpy(),
                        side="right").astype("int64"),
    ]
    return np.ravel_multi_index(codes, LOOKUP_SIZES)


@dataclass
class LookupArm:
    """A (n_cells, 91) table of pmfs over the binned state grid -- what the
    engine needs instead of a batched predict. Exposes the SAME `pmf(df)`
    interface as every fitted arm, so the same scorer and the same chain grade
    it with no second code path."""

    table: np.ndarray
    tempo_edges: tuple[float, float]
    source_arm: str
    name: str = "lookup"

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        return self.table[lookup_codes(df, self.tempo_edges)]


def build_lookup(arm, train: pd.DataFrame, tempo_edges: tuple[float, float],
                 chunk: int = 200_000, seed: int = 20260910) -> tuple[LookupArm, dict]:
    """Average the live arm's own pmf inside each binned cell, weighted by the
    training rows that fall in it.

    This is a BINNING of the fitted model, not a refit and not a smoothing: an
    empty cell falls back to the pooled average, and the fallback count is
    reported. Nothing is tuned to make the binned table match anything."""
    n_cells = int(np.prod(LOOKUP_SIZES))
    acc = np.zeros((n_cells, N_GRID), dtype="float64")
    cnt = np.zeros(n_cells, dtype="float64")
    for a in range(0, len(train), chunk):
        blk = train.iloc[a:a + chunk]
        k = lookup_codes(blk, tempo_edges)
        p = arm.pmf(blk)
        np.add.at(acc, k, p)
        np.add.at(cnt, k, 1.0)
    empty = cnt == 0
    pooled = acc.sum(axis=0) / max(cnt.sum(), 1.0)
    table = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1.0)[:, None], pooled[None, :])
    table = ck._normalise(table)
    diag = {"n_cells": n_cells, "n_empty_cells": int(empty.sum()),
            "cell_dims": list(LOOKUP_DIMS), "cell_sizes": list(LOOKUP_SIZES),
            "train_rows_binned": int(len(train)), "source_arm": getattr(arm, "name", "?")}
    return LookupArm(table=table, tempo_edges=tempo_edges,
                     source_arm=getattr(arm, "name", "?")), diag


def binning_error(live, lut: LookupArm, te: pd.DataFrame, chunk: int = 100_000) -> dict:
    """Total-variation distance between the live and binned pmfs, row by row."""
    tv_max, tv_sum, n = 0.0, 0.0, 0
    for a in range(0, len(te), chunk):
        blk = te.iloc[a:a + chunk]
        p, q = live.pmf(blk), lut.pmf(blk)
        tv = 0.5 * np.abs(p - q).sum(axis=1)
        tv_max = max(tv_max, float(tv.max()))
        tv_sum += float(tv.sum())
        n += len(blk)
    return {"tv_mean": tv_sum / max(n, 1), "tv_max": tv_max, "n_rows": n}


# ===========================================================================
# 7. ROUND 3b -- the S1 training scheme (in-season monthly walk-forward refit)
# ===========================================================================
# L21 / `docs/models/README.md` standing result: every sub-model refits monthly
# in-season on all prior seasons plus the season to date, strictly before the
# refit date. Round 3's pre-registration (experiments.md section 8) is STATIC
# ONLY and is not edited; round 3b is a separate, separately pre-registered
# confirmation that takes round 3's winner and asks whether S1 moves any gate.
#
# The scheme definition is `possession_outcome.month_boundaries` /
# `fit_predict_scheme`'s S1 branch, REUSED rather than re-implemented, so the
# two models cannot drift apart on what "monthly walk-forward" means.

#: Columns any round-3 arm reads while fitting. A monthly refit concatenates
#: the training slice with the test season to date; on a 2.6M-row design the
#: full-width copy is ~1 GB per refit, and there are five or six of them.
#: Subsetting to these columns cannot change a fitted model -- every fitter
#: reads exactly its feature set plus these -- it only stops paying for the
#: rest.
S1_FIT_EXTRA_COLS: tuple[str, ...] = (
    "duration_s", "censored", "censored_horn", "censored_old", "season",
    "game_date", "prev_end", "tempo_prior_game", "seconds_remaining", "period",
    "score_diff", "is_ot", "in_bonus", "site_home", "site_away",
)


def s1_fit_columns(arm: str) -> list[str]:
    feats = ck.feature_set(ARM_FEATURES_V3[arm])
    out: list[str] = []
    for c in [*feats, *S1_FIT_EXTRA_COLS]:
        if c not in out:
            out.append(c)
    return out


@dataclass
class MonthlyArm:
    """An S1 arm is a SCHEDULE of fits, not one fit.

    `pmf` routes each row to the most recent refit at or before that row's own
    game date, which is the same rule the scorer and `chain_halves` then see --
    so the sim draws from exactly the object that was scored, and no game is
    ever served by a model that has seen it."""

    cuts: list[pd.Timestamp]
    arms: list
    base_arm: str
    name: str = "s1"

    @property
    def cut_index(self) -> np.ndarray:
        return np.array([np.datetime64(c) for c in self.cuts], dtype="datetime64[ns]")

    def route(self, df: pd.DataFrame) -> np.ndarray:
        d = pd.to_datetime(df["game_date"]).to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(self.cut_index, d, side="right") - 1
        return np.clip(idx, 0, len(self.cuts) - 1)

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        idx = self.route(df)
        out = np.zeros((len(df), N_GRID), dtype="float64")
        for k in np.unique(idx):
            rows = np.flatnonzero(idx == k)
            out[rows] = self.arms[int(k)].pmf(df.iloc[rows])
        return out


def fit_s1(arm_name: str, tr: pd.DataFrame, te: pd.DataFrame, seed: int = 0,
           **kw) -> tuple[MonthlyArm, dict]:
    """Refit `arm_name` at every month boundary of the TEST season on all prior
    seasons plus the test season strictly before that boundary."""
    from cbb_sim.models import possession_outcome as PO

    te_dates = pd.to_datetime(te["game_date"])
    tr_dates = pd.to_datetime(tr["game_date"])
    cuts = PO.month_boundaries(te_dates)
    cols = s1_fit_columns(arm_name)
    tr_small = tr[cols]

    arms, segments, kept_cuts = [], [], []
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not seg.any():
            continue
        before = (te_dates < cut).to_numpy()
        prior = te.loc[before, cols]
        fit_rows = tr_small if not len(prior) else pd.concat([tr_small, prior], ignore_index=True)
        arms.append(fit_arm_v3(arm_name, fit_rows, seed=seed, **kw))
        kept_cuts.append(pd.Timestamp(cut))
        max_train = tr_dates.max() if not before.any() else max(tr_dates.max(), te_dates[before].max())
        segments.append({
            "refit_date": str(pd.Timestamp(cut).date()),
            "valid_from": str(pd.Timestamp(cut).date()),
            "valid_to": None if nxt is None else str((pd.Timestamp(nxt) - pd.Timedelta(days=1)).date()),
            "n_train": int(len(fit_rows)),
            "n_train_from_test_season": int(len(prior)),
            "n_scored": int(seg.sum()),
            "max_train_date": str(pd.Timestamp(max_train).date()),
        })
        del fit_rows, prior
    if not arms:
        raise AssertionError("S1 produced no refits; the month partition is empty")
    ma = MonthlyArm(cuts=kept_cuts, arms=arms, base_arm=arm_name)
    # every test row must be routed to a fit whose training data precedes it
    idx = ma.route(te)
    bad = int((te_dates.to_numpy() < ma.cut_index[idx]).sum())
    meta = {"scheme": "S1", "base_arm": arm_name, "n_fits": len(arms),
            "segments": segments, "rows_before_their_own_cut": bad,
            "earliest_train_date": str(tr_dates.min().date()) if len(tr_dates) else None}
    return ma, meta


# ===========================================================================
# 8. ROUND 3b -- state parametrisation crossed with censoring
# ===========================================================================
# Pre-registration: `docs/models/clock/experiments.md` section 10, appended
# after round 3's verdict. L23 / Decision 10: `score_diff` is PRODUCED BY the
# simulation, so conditioning the clock on it closes a feedback loop that
# offline scoring cannot see. Round 3b crosses the (confirmed) censoring fix
# with three state parametrisations and adds the closed-loop gate.
#
#   P1  as designed    round-2 state verbatim (score_diff, its clock
#                      interaction, and the fine-bucket x period x SCORE-STATE
#                      cross)
#   P2  score removed  all three dropped; the cross collapses to fine bucket x
#                      period type. No simulation-produced score information
#                      reaches the model at all
#   P3  engine-safe    score_diff replaced by three mutually exclusive
#                      indicators that are non-zero ONLY in the last
#                      ENDGAME_WINDOW_S seconds of H2/OT: trailing big (the
#                      intentional-foul regime), leading big (run out the
#                      clock), close. Outside the window the model sees no
#                      score at all.

#: The only window in which round 3b lets the margin reach the clock model.
ENDGAME_WINDOW_S = 120
#: Margin separating the intentional-foul / run-out-the-clock regimes from a
#: close game. Fixed by the pre-registration, not searched.
ENDGAME_MARGIN = 4

P3_FEATURES: tuple[str, ...] = ("eg_trailing_big", "eg_leading_big", "eg_close")

#: Cross of fine clock bucket x period type ONLY -- P2/P3 replacement for round
#: 2 three-way cross. Level 0 is the dropped reference level.
P2_CROSS_COLUMNS: tuple[str, ...] = tuple(
    f"sr2x_{ck.R2_SR_LABELS[b]}__{ck.R2_PERIOD_TYPES[p]}"
    for b in range(len(ck.R2_SR_LABELS)) for p in range(len(ck.R2_PERIOD_TYPES))
)
P2_CROSS_DUMMIES: tuple[str, ...] = P2_CROSS_COLUMNS[1:]

#: R2_CARRIED without the two score columns.
R2_CARRIED_NOSCORE: tuple[str, ...] = tuple(
    c for c in ck.R2_CARRIED
    if c not in ("score_diff", "x_score_diff__seconds_remaining")
)

FEATURES_P2_DUMMY: tuple[str, ...] = (
    R2_CARRIED_NOSCORE + P2_CROSS_DUMMIES + ck.R2_WINDOW_FEATURES)
FEATURES_P3_DUMMY: tuple[str, ...] = FEATURES_P2_DUMMY + P3_FEATURES

# Registered with the shared lookups so `clock.feature_set` and every fitter see
# them. This mutates THIS PROCESS dict only; `clock.py` on disk is untouched.
ck.FEATURE_SETS["P2_dummy"] = FEATURES_P2_DUMMY
ck.FEATURE_SETS["P3_dummy"] = FEATURES_P3_DUMMY
ck.EMPIRICAL_DIM_SIZES["eg_regime"] = 4
ck.EMPIRICAL_DIMS["P2_dummy"] = (
    "prev_end_code", "r2_bucket_code", "r2_period_type", "tempo_tercile")
ck.EMPIRICAL_DIMS["P3_dummy"] = (
    "prev_end_code", "r2_bucket_code", "r2_period_type", "eg_regime", "tempo_tercile")

PARAMETRISATIONS: tuple[str, ...] = ("P1", "P2", "P3")
P_FEATURES: dict[str, str] = {"P1": "R2_dummy", "P2": "P2_dummy", "P3": "P3_dummy"}
V3B_BASE_ARMS: tuple[str, ...] = ("empirical_km3", "gamma_aft")
#: Fewer simulation-produced inputs is simpler AND safer (pre-registration 10.4).
P_SIMPLICITY: dict[str, int] = {"P2": 0, "P3": 1, "P1": 2}


def eg_regime_code(seconds_remaining, period, score_diff) -> np.ndarray:
    """0 outside the end-game window, 1 trailing big, 2 leading big, 3 close."""
    sr = np.asarray(seconds_remaining, dtype="float64")
    per = np.asarray(period, dtype="float64")
    sd = np.asarray(score_diff, dtype="float64")
    inside = (sr <= ENDGAME_WINDOW_S) & (per >= 2.0)
    out = np.zeros(sr.shape, dtype="int64")
    out = np.where(inside & (sd <= -ENDGAME_MARGIN), 1, out)
    out = np.where(inside & (sd >= ENDGAME_MARGIN), 2, out)
    out = np.where(inside & (np.abs(sd) < ENDGAME_MARGIN), 3, out)
    return out


def add_p_state(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the P2 collapsed cross and the P3 end-game indicators IN PLACE.

    Recomputed from the frame's own clock / period / score every time, exactly
    as `clock.add_r2_state` is, so the emergent chain's OVERRIDDEN clock is what
    selects the state and no real possession's end-game state can leak into the
    simulation."""
    sr = df["seconds_remaining"].to_numpy(dtype="float64")
    per = df["period"].to_numpy(dtype="float64")
    sd = df["score_diff"].to_numpy(dtype="float64")
    key2 = np.ravel_multi_index(
        (ck.r2_bucket_id(sr), ck.r2_period_type(per)),
        (len(ck.R2_SR_LABELS), len(ck.R2_PERIOD_TYPES)))
    for j, name in enumerate(P2_CROSS_COLUMNS):
        if j == 0:
            continue
        df[name] = (key2 == j).astype("int8")
    reg = eg_regime_code(sr, per, sd)
    df["eg_regime"] = reg.astype("float32")
    df["eg_trailing_big"] = (reg == 1).astype("float32")
    df["eg_leading_big"] = (reg == 2).astype("float32")
    df["eg_close"] = (reg == 3).astype("float32")
    return df


@dataclass
class StateWrapArm:
    """Recompute the P2/P3 state columns from the frame's CURRENT clock before
    delegating to the fitted arm.

    `clock.apply_clock_override` knows only about round 2 columns, so without
    this wrapper a chained draw would carry the REAL possession's end-game
    indicators while the simulated clock had moved -- a leak of the answer into
    the emergent test. Wrapping keeps one chain and one scorer."""

    inner: object
    parametrisation: str

    @property
    def name(self) -> str:
        return f"{getattr(self.inner, 'name', '?')}|{self.parametrisation}"

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        return self.inner.pmf(add_p_state(df.copy()))


class EmpiricalArmV3P(EmpiricalArmV3):
    """`EmpiricalArmV3` that also offers the `eg_regime` cell dimension and
    recomputes it from the frame's own clock."""

    def _codes(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        codes = super()._codes(df)
        codes["eg_regime"] = eg_regime_code(
            df["seconds_remaining"].to_numpy(), df["period"].to_numpy(),
            df["score_diff"].to_numpy())
        return codes


def _fit_empirical_p(tr: pd.DataFrame, fs: str, parametrisation: str,
                     sr_floor_bucket: int = 0) -> EmpiricalArmV3P:
    dims = ck.EMPIRICAL_DIMS[fs]
    sizes = tuple(ck.EMPIRICAL_DIM_SIZES[d] for d in dims)
    tempo = tr["tempo_prior_game"].to_numpy(dtype="float64")
    arm = EmpiricalArmV3P(
        feature_set_name=fs, dims=dims, sizes=sizes,
        level_pmfs=[], level_counts=[], level_events=[],
        tempo_edges=(float(np.quantile(tempo, 1 / 3)), float(np.quantile(tempo, 2 / 3))),
        season_map={s: i for i, s in enumerate(sorted(int(x) for x in tr["season"].unique()))},
        sr_floor_bucket=sr_floor_bucket, features=ck.feature_set(fs),
        name=("empirical_km3_srfloor" if sr_floor_bucket else "empirical_km3")
             + f"_{parametrisation}")
    codes = arm._codes(tr)
    y = tr["duration_s"].to_numpy(dtype="int64")
    cen = tr["censored"].to_numpy(dtype=bool)
    pooled = np.zeros((1, N_GRID), dtype="float64")
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
        pmf, counts, events = kaplan_meier_pmf_v3(keys, y, cen, n_cells, parent_pmf, parent_of)
        arm.level_pmfs.append(ck._normalise(pmf))
        arm.level_counts.append(counts)
        arm.level_events.append(events)
    arm.level_counts[0] = np.maximum(arm.level_counts[0], arm.min_cell)
    arm.level_events[0] = np.maximum(arm.level_events[0], arm.min_events)
    return arm


def fit_arm_v3b(base_arm: str, parametrisation: str, train: pd.DataFrame,
                seed: int = 0, **kw):
    """Fit `base_arm` under `parametrisation`, returning a chain-safe object."""
    fs = P_FEATURES[parametrisation]
    tr = train if parametrisation == "P1" else add_p_state(train.copy())
    if base_arm in ("empirical_km3", "empirical_km3_srfloor"):
        # ROUND 3c (experiments.md section 12.1): `empirical_km3_srfloor` is
        # arm A2 -- the same cell grid with the fine clock bucket FLOORED at
        # 45-59 s -- crossed with a parametrisation. The floor is a property of
        # the CELL CODING, so it has to be set before the levels are built;
        # this argument is additive and every round-3b call, which passes none,
        # is byte-identical to what it was.
        floor_b = SR_FLOOR_BUCKET if base_arm == "empirical_km3_srfloor" else 0
        inner = (fit_empirical_v3(tr, fs, sr_floor_bucket=floor_b)
                 if parametrisation == "P1"
                 else _fit_empirical_p(tr, fs, parametrisation, sr_floor_bucket=floor_b))
    elif base_arm == "gamma_aft":
        inner = ck.fit_parametric(tr, fs, family="gamma", seed=seed)
    else:
        raise ValueError(base_arm)
    return inner if parametrisation == "P1" else StateWrapArm(inner, parametrisation)
