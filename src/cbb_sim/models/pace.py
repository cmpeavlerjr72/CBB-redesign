"""
pace.py -- L2 pace bake-off: feature builder, model trainers, predictive
sampler.

Spec: `docs/models/pace/experiments.md` (pre-registered 2026-09-10, before any
of this code existed). This module builds the feature table (feature sets
A/B/C/D, targets T_box and T_pbp) and every model class in the grid
(multiplicative, ridge, Gaussian GLM -- the Control's own incumbent pace
mechanism, reused verbatim -- and LightGBM), the three distribution families
(Gaussian fixed SD, heteroscedastic Gaussian, NegBin/Poisson on the integer
count), and a deterministic sampler keyed on (seed, game_id, family) through
`cbb_sim.control.rng`, the engine-wide RNG contract.

Reused, not reimplemented (see `docs/models/pace/model.md` section 6):
    cbb_sim.ratings.own_ratings   as-of ridge tempo, and the team-game box rows
    cbb_sim.control.features      KenPom as-of join (build_kenpom_key_map,
                                   join_kenpom_as_of) -- identical
                                   strictly-before semantics as the Control's
                                   own KenPom arm
    cbb_sim.control.models        FittedModel / fit_pace (the incumbent
                                   Gaussian GLM) and the NegBin dispersion
                                   machinery (_medians/_design/_nb2_alpha)
    cbb_sim.control.rng           counter-based (seed, game_id, family)
                                   streams and inverse-CDF variate draws
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from cbb_sim.control import features as cf
from cbb_sim.control import models as cm
from cbb_sim.control import rng as crng
from cbb_sim.ratings import own_ratings as orat

DEFAULT_HOOPR_DIR = Path("data/raw/hoopr")
DEFAULT_KENPOM = Path("data/processed/kenpom_snapshots.parquet")
DEFAULT_POSS_PBP = Path("data/processed/possessions_pbp.parquet")
DEFAULT_MODEL_DIR = Path("data/processed/models/pace")

# season_index = season - ANCHOR, fixed across every call so F1 and F2 share
# one scale (F1 train {2022,2023} -> {0,1}; F2 train {2022,2023,2024} -> {0,1,2}).
SEASON_INDEX_ANCHOR = 2022

FOLDS: dict[str, dict[str, list[int]]] = {
    "F1": {"train": [2022, 2023], "test": [2024]},
    "F2": {"train": [2022, 2023, 2024], "test": [2025]},
}


def fold_seasons(fold: str) -> tuple[list[int], list[int]]:
    f = FOLDS[fold]
    return list(f["train"]), list(f["test"])


TARGETS: dict[str, str] = {"T_box": "poss_box", "T_pbp": "poss_pbp"}

STYLE_RAW: tuple[str, ...] = ("tpa_per100", "fta_per100", "tov_per100", "oreb_pct")

FEATURES_A: tuple[str, ...] = (
    "home_tempo_rel", "away_tempo_rel",
    "home_kp_adj_t_rel", "away_kp_adj_t_rel",
    "neutral",
)
FEATURES_B: tuple[str, ...] = FEATURES_A + ("days_since_start", "month", "season_index")
_STYLE_COLS: tuple[str, ...] = tuple(
    f"{side}_{kind}_{c}"
    for side in ("home", "away")
    for kind in ("own_style", "opp_allowed_style")
    for c in STYLE_RAW
)
FEATURES_C: tuple[str, ...] = FEATURES_B + _STYLE_COLS
FEATURES_D: tuple[str, ...] = FEATURES_C + (
    "home_rest_days", "away_rest_days", "home_b2b", "away_b2b", "conf_game",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "A_tempo": FEATURES_A,
    "B_plus_season": FEATURES_B,
    "C_plus_style": FEATURES_C,
    "D_plus_state": FEATURES_D,
}

MODEL_CLASSES: tuple[str, ...] = ("multiplicative", "ridge", "glm_gaussian", "lightgbm")


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------
def load_conference_flags(seasons: list[int], hoopr_dir: Path | str) -> pd.DataFrame:
    """game_id -> conf_game (1.0 if both sides share a non-null conference_id,
    else 0.0). Source: hoopR schedules, the same columns
    `cbb_sim.data.universe.compute_d1_team_seasons` reads for the D-I flag."""
    frames = []
    for season in seasons:
        p = Path(hoopr_dir) / "schedules" / f"mbb_schedule_{season}.parquet"
        sch = pd.read_parquet(p, columns=["game_id", "home_conference_id", "away_conference_id"])
        frames.append(sch)
    sch = pd.concat(frames, ignore_index=True)
    sch["game_id"] = pd.to_numeric(sch["game_id"], errors="coerce").astype("int64")
    both_known = sch["home_conference_id"].notna() & sch["away_conference_id"].notna()
    sch["conf_game"] = (both_known & (sch["home_conference_id"] == sch["away_conference_id"])).astype("float64")
    return sch[["game_id", "conf_game"]].drop_duplicates("game_id")


def build_pace_table(
    seasons: list[int],
    universe: pd.DataFrame | None = None,
    hoopr_dir: Path | str = DEFAULT_HOOPR_DIR,
    ratings_dir: Path | str = orat.DEFAULT_OUT_DIR,
    kenpom_path: Path | str = DEFAULT_KENPOM,
    possessions_pbp_path: Path | str = DEFAULT_POSS_PBP,
) -> pd.DataFrame:
    """One row per D-I, non-truncated game in `seasons`: every FEATURES_D
    column plus both targets. `poss_box` is populated for every row; `poss_pbp`
    is NaN for the small share of games `scripts/build_possessions_pbp.py`
    excludes (no pbp file coverage) -- callers drop those rows before fitting
    a T_pbp arm, never fill them in.

    Every feature is pregame by construction: as-of ridge/KenPom joins
    (strictly-before semantics, `cbb_sim.ratings.own_ratings` /
    `cbb_sim.control.features`), `shift(1)+expanding().mean()` style rates
    (uses only that team's STRICTLY EARLIER games this season), and rest days
    from the previous calendar date in that team's own schedule.
    """
    if universe is None:
        universe = orat.load_universe()
    tg = orat.load_team_games(universe, seasons, hoopr_dir)

    # ---- day-of-season / month / season index ------------------------------
    start = tg.groupby("season")["game_date"].transform("min")
    tg["days_since_start"] = (tg["game_date"] - start).dt.days.astype("float64")
    tg["month"] = pd.to_datetime(tg["game_date"]).dt.month.astype("float64")
    tg["season_index"] = (tg["season"] - SEASON_INDEX_ANCHOR).astype("float64")

    # ---- own ridge tempo, as-of (+ the as-of league mean, for `multiplicative`)
    ratings = orat.load_ratings(sorted(set(seasons)), ratings_dir)
    tg = orat.join_as_of(tg, ratings, team_col="team_id", suffix="",
                          cols=("tempo_rel", "league_tempo_mean"))

    # ---- centred KenPom tempo, as-of (reuses the Control's own helpers) ----
    kp = pd.read_parquet(kenpom_path)
    key_map = cf.build_kenpom_key_map(sorted(set(seasons)), hoopr_dir, kp)
    tg = cf.join_kenpom_as_of(tg, kp, key_map, team_col="team_id", prefix="own_kp_")

    # ---- style rates: own as-of expanding mean + opponents-allowed ---------
    opp_dreb = tg[["game_id", "team_id", "dreb"]].rename(
        columns={"team_id": "opp_team_id", "dreb": "opp_dreb"}
    )
    tg = tg.merge(opp_dreb, on=["game_id", "opp_team_id"], how="left")
    tg["tpa_per100"] = 100.0 * tg["tpa"] / tg["game_poss"]
    tg["fta_per100"] = 100.0 * tg["fta"] / tg["game_poss"]
    tg["tov_per100"] = 100.0 * tg["tov"] / tg["game_poss"]
    tg["oreb_pct"] = tg["oreb"] / (tg["oreb"] + tg["opp_dreb"])

    tg = tg.sort_values(["season", "team_id", "game_date", "game_id"], kind="mergesort")
    for c in STYLE_RAW:
        tg[f"own_style_{c}"] = tg.groupby(["season", "team_id"])[c].transform(
            lambda s: s.shift(1).expanding().mean()
        )
    opp_rates = tg[["game_id", "team_id", *STYLE_RAW]].rename(
        columns={"team_id": "opp_team_id", **{c: f"_opp_raw_{c}" for c in STYLE_RAW}}
    )
    tg = tg.merge(opp_rates, on=["game_id", "opp_team_id"], how="left")
    for c in STYLE_RAW:
        tg[f"opp_allowed_style_{c}"] = tg.groupby(["season", "team_id"])[f"_opp_raw_{c}"].transform(
            lambda s: s.shift(1).expanding().mean()
        )

    # ---- rest days / back-to-back, from each team's own schedule -----------
    sched = (
        tg[["season", "team_id", "game_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["season", "team_id", "game_date", "game_id"], kind="mergesort")
    )
    sched["prev_date"] = sched.groupby(["season", "team_id"])["game_date"].shift(1)
    sched["rest_days"] = (sched["game_date"] - sched["prev_date"]).dt.days.astype("float64")
    sched["b2b"] = np.where(sched["rest_days"].isna(), np.nan, (sched["rest_days"] <= 1.0).astype("float64"))
    tg = tg.merge(sched[["season", "team_id", "game_id", "rest_days", "b2b"]],
                  on=["season", "team_id", "game_id"], how="left")

    # ---- collapse to one row per game (home_/away_ prefixed columns) -------
    rename_map = {
        "tempo_rel": "tempo_rel",
        "own_kp_adj_t_rel": "kp_adj_t_rel",
        **{f"own_style_{c}": f"own_style_{c}" for c in STYLE_RAW},
        **{f"opp_allowed_style_{c}": f"opp_allowed_style_{c}" for c in STYLE_RAW},
        "rest_days": "rest_days", "b2b": "b2b",
    }
    per_team_cols = ["game_id", *rename_map]
    home = tg[tg["team_id"] == tg["home_team_id"]][per_team_cols].rename(
        columns={k: f"home_{v}" for k, v in rename_map.items()}
    )
    away = tg[tg["team_id"] == tg["away_team_id"]][per_team_cols].rename(
        columns={k: f"away_{v}" for k, v in rename_map.items()}
    )

    meta = tg.drop_duplicates("game_id")[[
        "game_id", "season", "game_date", "tipoff_utc", "home_team_id", "away_team_id",
        "neutral", "days_since_start", "month", "season_index", "game_poss", "n_periods",
        "cbbd_game_id", "league_tempo_mean",
    ]].rename(columns={"game_poss": "poss_box", "league_tempo_mean": "league_tempo_mean_asof"})

    conf = load_conference_flags(sorted(set(seasons)), hoopr_dir)
    g = meta.merge(home, on="game_id").merge(away, on="game_id").merge(conf, on="game_id", how="left")

    poss_pbp = pd.read_parquet(possessions_pbp_path)[["game_id", "poss_pbp"]]
    g = g.merge(poss_pbp, on="game_id", how="left")

    return g.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model classes -- point-estimate mean, all sharing a `.predict(df)` contract
# ---------------------------------------------------------------------------
def _fit_medians(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    return {f: float(df[f].astype("float64").median()) for f in features}


def _fillna(df: pd.DataFrame, features: list[str], medians: dict[str, float]) -> pd.DataFrame:
    return df[features].astype("float64").fillna(pd.Series(medians))


@dataclass
class MultiplicativeModel:
    """Zero-fitted-parameter KenPom-style formula:
    pred = home_tempo_rel * away_tempo_rel * league_tempo_mean_asof.
    Uses the project's self-contained own-ratings tempo (L9: ties centred
    KenPom within seed noise) rather than a third-party-dependent arm.
    Ignores every feature beyond team tempo by construction, so its row is
    identical across feature sets B/C/D."""

    medians: dict[str, float]
    n_params: int = 0

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = _fillna(df, ["home_tempo_rel", "away_tempo_rel", "league_tempo_mean_asof"], self.medians)
        return (x["home_tempo_rel"] * x["away_tempo_rel"] * x["league_tempo_mean_asof"]).to_numpy()


def fit_multiplicative(train: pd.DataFrame) -> MultiplicativeModel:
    cols = ["home_tempo_rel", "away_tempo_rel", "league_tempo_mean_asof"]
    return MultiplicativeModel(medians=_fit_medians(train, cols))


RIDGE_ALPHA_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


@dataclass
class RidgeModel:
    features: list[str]
    medians: dict[str, float]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float
    cv_rmse: float
    n_params: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_params = len(self.features) + 1

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        x = _fillna(df, self.features, self.medians).to_numpy()
        return (x - self.scaler_mean) / self.scaler_scale

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._design(df) @ self.coef + self.intercept


def fit_ridge(train: pd.DataFrame, features: list[str], target: str,
              alphas: tuple[float, ...] = RIDGE_ALPHA_GRID, seed: int = 20260910) -> RidgeModel:
    d = train.dropna(subset=[target])
    medians = _fit_medians(d, features)
    x = _fillna(d, features, medians).to_numpy()
    y = d[target].to_numpy(dtype="float64")
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_alpha, best_rmse = alphas[0], np.inf
    for a in alphas:
        rmses = []
        for tr_idx, va_idx in kf.split(xs):
            m = Ridge(alpha=a).fit(xs[tr_idx], y[tr_idx])
            pred = m.predict(xs[va_idx])
            rmses.append(np.sqrt(np.mean((pred - y[va_idx]) ** 2)))
        mrmse = float(np.mean(rmses))
        if mrmse < best_rmse:
            best_rmse, best_alpha = mrmse, a

    final = Ridge(alpha=best_alpha).fit(xs, y)
    return RidgeModel(
        features=list(features), medians=medians, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
        coef=np.asarray(final.coef_, dtype="float64"), intercept=float(final.intercept_),
        alpha=float(best_alpha), cv_rmse=best_rmse,
    )


def fit_glm_gaussian(train: pd.DataFrame, features: list[str], target: str) -> cm.FittedModel:
    """The Control's own incumbent pace mechanism, reused verbatim (Gaussian
    GLM, identity link, residual SD fitted from training residuals)."""
    d = train.dropna(subset=[target])
    return cm.fit_pace(d, list(features), target=target)


LGBM_PARAM_GRID: tuple[dict, ...] = tuple(
    {"num_leaves": nl, "learning_rate": lr, "n_estimators": 400, "min_child_samples": 30,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0}
    for nl in (7, 15, 31)
    for lr in (0.02, 0.05, 0.1)
)


@dataclass
class LGBMModel:
    features: list[str]
    medians: dict[str, float]
    booster: object
    params: dict
    cv_rmse: float
    n_params: int = 6  # nominal effective-parameter count for df_resid; documented, not exact for a tree

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = _fillna(df, self.features, self.medians)
        return np.asarray(self.booster.predict(x), dtype="float64")


def fit_lightgbm(train: pd.DataFrame, features: list[str], target: str,
                  param_grid: tuple[dict, ...] = LGBM_PARAM_GRID, seed: int = 20260910) -> LGBMModel:
    import lightgbm as lgb

    d = train.dropna(subset=[target])
    medians = _fit_medians(d, features)
    x = _fillna(d, features, medians)
    y = d[target].to_numpy(dtype="float64")

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_params, best_rmse = param_grid[0], np.inf
    for params in param_grid:
        rmses = []
        for tr_idx, va_idx in kf.split(x):
            m = lgb.LGBMRegressor(random_state=seed, verbosity=-1, **params)
            m.fit(x.iloc[tr_idx], y[tr_idx])
            pred = m.predict(x.iloc[va_idx])
            rmses.append(np.sqrt(np.mean((pred - y[va_idx]) ** 2)))
        mrmse = float(np.mean(rmses))
        if mrmse < best_rmse:
            best_rmse, best_params = mrmse, params

    final = lgb.LGBMRegressor(random_state=seed, verbosity=-1, **best_params)
    final.fit(x, y)
    return LGBMModel(features=list(features), medians=medians, booster=final, params=dict(best_params), cv_rmse=best_rmse)


def fit_point_model(model_class: str, train: pd.DataFrame, features: list[str], target: str, seed: int = 20260910):
    if model_class == "multiplicative":
        return fit_multiplicative(train)
    if model_class == "ridge":
        return fit_ridge(train, features, target, seed=seed)
    if model_class == "glm_gaussian":
        return fit_glm_gaussian(train, features, target)
    if model_class == "lightgbm":
        return fit_lightgbm(train, features, target, seed=seed)
    raise ValueError(f"unknown model class: {model_class!r}")


def predict_point(model, df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(df), dtype="float64")
    if hasattr(model, "linpred"):  # cm.FittedModel (glm_gaussian, count GLM)
        return np.asarray(model.linpred(df), dtype="float64")
    raise TypeError(f"model {model!r} has neither .predict nor .linpred")


def model_n_params(model) -> int:
    if isinstance(model, cm.FittedModel):
        return len(model.features) + 1
    return int(getattr(model, "n_params", 1))


# ---------------------------------------------------------------------------
# Distribution families (wrap a point-estimate model's residuals)
# ---------------------------------------------------------------------------
def fit_gaussian_fixed(train: pd.DataFrame, target: str, mu_train: np.ndarray, n_params: int) -> dict:
    """`mu_train` must be the point-estimate model's predictions on
    `train.dropna(subset=[target])`, in that row order (same contract as
    `fit_gaussian_hetero`)."""
    d = train.dropna(subset=[target])
    y = d[target].to_numpy(dtype="float64")
    resid = y - mu_train
    df_resid = max(len(y) - n_params, 1)
    sd = float(np.sqrt(np.sum(resid ** 2) / df_resid))
    return {"family": "gaussian", "sd": sd}


#  For resid ~ N(0, sigma^2(x)), log(resid^2) = log(sigma^2(x)) + log(Z^2) with
#  Z ~ N(0,1); E[log(Z^2)] = -(ln 2 + Euler-Mascheroni gamma) ~= -1.2704, NOT 0.
#  A plain OLS of log(resid^2) on x therefore estimates log(sigma^2(x)) minus
#  this constant, and exponentiating the raw fit understates sigma by a
#  factor of ~1.89 (verified against the fitted-constant SD on this data:
#  correcting it is what makes the heteroscedastic PIT comparable to the
#  fixed-SD one instead of spuriously tighter). Added back here so `hetero_sd`
#  returns an unbiased sigma(x), not a systematically-too-narrow one.
_LOG_CHI2_1_MEAN_CORRECTION: float = np.log(2.0) + np.euler_gamma


def fit_gaussian_hetero(train: pd.DataFrame, features: list[str], target: str, mu_train: np.ndarray) -> dict:
    """Normal with heteroscedastic SD, log-linear in the features: OLS of
    log(residual^2) on the same feature set; sd(x) = sqrt(exp(fitted + bias
    correction))."""
    d = train.dropna(subset=[target]).reset_index(drop=True)
    medians = _fit_medians(d, features)
    x = _fillna(d, features, medians)
    resid = d[target].to_numpy(dtype="float64") - mu_train
    z = np.log(np.maximum(resid ** 2, 1e-6))
    xc = sm.add_constant(x.to_numpy(), has_constant="add")
    res = sm.OLS(z, xc).fit()
    return {"family": "gaussian_hetero", "features": list(features), "medians": medians,
            "coef": np.asarray(res.params, dtype="float64")}


def hetero_sd(dist: dict, df: pd.DataFrame) -> np.ndarray:
    x = _fillna(df, dist["features"], dist["medians"]).to_numpy()
    xc = sm.add_constant(x, has_constant="add")
    log_var = xc @ dist["coef"] + _LOG_CHI2_1_MEAN_CORRECTION
    log_var = np.clip(log_var, -20, 20)
    return np.sqrt(np.exp(log_var))


def dist_sd(dist: dict, df: pd.DataFrame | None = None) -> np.ndarray | float:
    """SD to use for a Gaussian-family dist on `df`'s rows.

    If `dist` already carries a materialized `sd` (a fitted constant for
    `gaussian`, or a precomputed per-row array for `gaussian_hetero`), that is
    returned directly -- no `df` needed. Otherwise, for `gaussian_hetero`
    without one, it is computed from `df` via `hetero_sd`.
    """
    if dist.get("sd") is not None:
        return dist["sd"]
    if dist["family"] == "gaussian_hetero":
        if df is None:
            raise ValueError(
                "gaussian_hetero requires df to compute per-row SD when dist has no precomputed 'sd'"
            )
        return hetero_sd(dist, df)
    raise ValueError(f"dist_sd: cannot compute SD for family {dist['family']!r} without 'sd' or df")


def fit_count_glm(train: pd.DataFrame, features: list[str], target: str,
                   threshold: float = cm.NEGBIN_DEVIANCE_DF_THRESHOLD) -> cm.FittedModel:
    """NegBin/Poisson on the integer count: `game_poss` (or `poss_pbp`)
    rounded, straight GLM (no offset -- unlike the Control's per-100-poss
    rates, this target is already a whole-game count). Reuses the Control's
    exact NegBin-selection rule and dispersion estimator."""
    d = train.dropna(subset=[target]).copy()
    d[target] = d[target].round().clip(lower=1)
    med = cm._medians(d, features)
    x = cm._design(d, features, med)
    y = d[target].to_numpy(dtype="float64")

    pois = sm.GLM(y, x, family=sm.families.Poisson()).fit()
    dev_df = float(pois.deviance / pois.df_resid)
    if dev_df > threshold:
        alpha = max(cm._nb2_alpha(y, np.asarray(pois.fittedvalues, dtype="float64")), 1e-8)
        res = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=alpha), offset=None).fit()
        coef, family, alpha_out = np.asarray(res.params, dtype="float64"), "negbin", alpha
    else:
        coef, family, alpha_out = np.asarray(pois.params, dtype="float64"), "poisson", float("nan")

    return cm.FittedModel(
        kind="pace_count", target=target, features=list(features), coef=coef, medians=med,
        n_obs=int(len(y)), df_resid=int(pois.df_resid), family=family, alpha=alpha_out,
        extra={"poisson_deviance_df": dev_df},
    )


# ---------------------------------------------------------------------------
# PIT / coverage
# ---------------------------------------------------------------------------
def pit_gaussian(y: np.ndarray, mu: np.ndarray, sd) -> np.ndarray:
    return stats.norm.cdf(y, loc=mu, scale=sd)


def coverage_gaussian(y: np.ndarray, mu: np.ndarray, sd, level: float) -> float:
    z = stats.norm.ppf(0.5 + level / 2)
    lo, hi = mu - z * sd, mu + z * sd
    return float(np.mean((y >= lo) & (y <= hi)))


def pit_discrete(y: np.ndarray, mu: np.ndarray, family: str, alpha: float, seed: int = 20260910) -> np.ndarray:
    """Randomized PIT for a discrete count distribution (Poisson/NegBin):
    F(y-1) + U * (F(y) - F(y-1)), same technique as
    `scripts/grade_control.py::gate_g5`'s margin PIT."""
    if family == "negbin":
        n = 1.0 / alpha
        p = n / (n + mu)
        f_lo = stats.nbinom.cdf(y - 1, n, p)
        f_hi = stats.nbinom.cdf(y, n, p)
    else:
        f_lo = stats.poisson.cdf(y - 1, mu)
        f_hi = stats.poisson.cdf(y, mu)
    rng = np.random.default_rng(seed)
    u = rng.random(len(y))
    return f_lo + u * (f_hi - f_lo)


def coverage_discrete(y: np.ndarray, mu: np.ndarray, family: str, alpha: float, level: float) -> float:
    lo_q, hi_q = 0.5 - level / 2, 0.5 + level / 2
    if family == "negbin":
        n = 1.0 / alpha
        p = n / (n + mu)
        lo, hi = stats.nbinom.ppf(lo_q, n, p), stats.nbinom.ppf(hi_q, n, p)
    else:
        lo, hi = stats.poisson.ppf(lo_q, mu), stats.poisson.ppf(hi_q, mu)
    return float(np.mean((y >= lo) & (y <= hi)))


# ---------------------------------------------------------------------------
# Deterministic predictive-distribution sampler
# ---------------------------------------------------------------------------
def sample_pace(
    mu: np.ndarray, game_ids: np.ndarray, seed: int, dist: dict, family: str = "pace",
    df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Draw one realisation per game from the chosen predictive distribution,
    keyed on (seed, game_id, family) -- the same counter-based RNG contract as
    the Control (`cbb_sim.control.rng`): a game's draw depends on nothing but
    that triple, so paired arms/seeds are bit-identically comparable and
    dropping a game from the run never moves another game's draw.

    `df` is required only for `gaussian_hetero` (its per-row SD is a function
    of the features, not a fitted constant).
    """
    keys = crng.stream_keys(seed, np.asarray(game_ids), family)
    fam = dist["family"]
    if fam in ("gaussian", "gaussian_hetero"):
        return crng.normal(keys, 0, mu, dist_sd(dist, df))
    if fam == "negbin":
        return crng.negbin(keys, 0, mu, dist["alpha"])
    if fam == "poisson":
        return crng.poisson(keys, 0, mu)
    raise ValueError(f"unknown distribution family: {fam!r}")
