"""
models.py -- the Control engine's component models.

Fixed by `docs/models/control_engine/model.md` section 3; these are NOT
bake-off candidates. Three families:

  pace    Gaussian GLM (identity link) on the symmetric game-level features,
          with the residual SD FITTED from the training residuals
          (sqrt(SSE / df_resid)) and reported. No hand-set dispersion anywhere.

  rates   per-100-possession counts (3PA, 2PA, FTA, TOV), Poisson GLM with a
          log link and offset log(game_poss / 100). The pre-registered rule:
          a target uses NegBin ONLY IF the Poisson training deviance/df
          exceeds 1.2. The test statistic and the resulting choice are stored
          on the model and reported. The NegBin dispersion alpha is estimated
          by the standard NB2 auxiliary regression of
          ((y - mu)^2 - y) / mu on mu through the origin, then the GLM is
          refitted at that alpha -- a fitted parameter, not a tuned one.

  pcts    make rates (3P%, 2P%, FT%), Binomial GLM with a logit link, weighted
          by trials (the two-column makes/misses endog IS trials weighting),
          with a Beta-Binomial overdispersion rho fitted by moment-matching
          the Pearson chi-square of the fitted binomial:

              sum (y - n p)^2 = sum n p (1-p) [1 + (n-1) rho]

          which is Williams' estimator. Reported, never assumed.

Every fitted object stores its own feature list AND the train-time median of
each feature, so the sim can (a) preflight that every feature exists in the sim
rows and (b) fill a missing pregame value with the training median rather than
a fabricated constant. Coefficients are stored as plain numpy so the sim loop
is pure vectorised arithmetic and never calls a model object (CLAUDE.md
modelling rule: "Sim loop uses lookup tables and vectorized NumPy, never live
model calls").
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_MODEL_DIR = Path("data/processed/models/control_engine")

# Pre-registered rule (experiments.md): NegBin only if Poisson deviance/df > 1.2
NEGBIN_DEVIANCE_DF_THRESHOLD = 1.2

RATE_TARGETS: tuple[str, ...] = ("tpa", "fg2a", "fta", "tov")
PCT_TARGETS: dict[str, tuple[str, str]] = {
    "tp_pct": ("tpm", "tpa"),
    "fg2_pct": ("fg2m", "fg2a"),
    "ft_pct": ("ftm", "fta"),
}


class FeaturePreflightError(RuntimeError):
    """Raised when a persisted model's features are not all present in the
    rows the simulator is about to score."""


@dataclass
class FittedModel:
    kind: str                       # 'pace' | 'rate' | 'pct'
    target: str
    features: list[str]
    coef: np.ndarray                # includes the intercept as element 0
    medians: dict[str, float]
    n_obs: int
    df_resid: int
    # family-specific fitted dispersion parameters
    family: str = "gaussian"
    resid_sd: float = float("nan")          # pace
    deviance_df: float = float("nan")       # rate: Poisson deviance / df
    alpha: float = float("nan")             # rate: NegBin dispersion (NaN if Poisson)
    rho: float = float("nan")               # pct: Beta-Binomial overdispersion
    extra: dict = field(default_factory=dict)

    # -- scoring ---------------------------------------------------------
    def design(self, df: pd.DataFrame) -> np.ndarray:
        preflight(df, self)
        x = df[self.features].astype("float64")
        x = x.fillna(pd.Series(self.medians))
        return np.column_stack([np.ones(len(x)), x.to_numpy()])

    def linpred(self, df: pd.DataFrame) -> np.ndarray:
        return self.design(df) @ self.coef


def preflight(df: pd.DataFrame, model: FittedModel) -> None:
    missing = [f for f in model.features if f not in df.columns]
    if missing:
        raise FeaturePreflightError(
            f"{model.kind}/{model.target}: features missing from the sim rows: {missing}"
        )


def preflight_bundle(df: pd.DataFrame, bundle: ControlModels) -> list[str]:
    """Check every model in the bundle against `df`. Returns the sorted union
    of the features checked; raises on the first model with a missing one."""
    feats: set[str] = set()
    for m in bundle.all_models():
        preflight(df, m)
        feats.update(m.features)
    return sorted(feats)


@dataclass
class ControlModels:
    fold: str
    anchor: str
    pace: FittedModel
    rates: dict[str, FittedModel]
    pcts: dict[str, FittedModel]
    train_seasons: list[int]
    created_at: str = ""
    n_games_train: int = 0

    def all_models(self) -> list[FittedModel]:
        return [self.pace, *self.rates.values(), *self.pcts.values()]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _medians(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    return {f: float(df[f].astype("float64").median()) for f in features}


def _design(df: pd.DataFrame, features: list[str], medians: dict[str, float]) -> np.ndarray:
    x = df[features].astype("float64").fillna(pd.Series(medians))
    return np.column_stack([np.ones(len(x)), x.to_numpy()])


def fit_pace(games: pd.DataFrame, features: list[str], target: str = "game_poss") -> FittedModel:
    """Gaussian GLM (identity link) for game possessions, with the residual SD
    fitted from the training residuals."""
    d = games.dropna(subset=[target]).copy()
    med = _medians(d, features)
    x = _design(d, features, med)
    y = d[target].to_numpy(dtype=float)
    res = sm.GLM(y, x, family=sm.families.Gaussian()).fit()
    resid = y - res.fittedvalues
    df_resid = int(len(y) - x.shape[1])
    sd = float(np.sqrt((resid ** 2).sum() / df_resid))
    return FittedModel(
        kind="pace", target=target, features=list(features),
        coef=np.asarray(res.params, dtype=float), medians=med,
        n_obs=int(len(y)), df_resid=df_resid, family="gaussian", resid_sd=sd,
        extra={"r2": float(1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())},
    )


def _nb2_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """Cameron & Trivedi NB2 auxiliary regression: regress
    ((y - mu)^2 - y) / mu on mu through the origin. The slope is alpha."""
    z = ((y - mu) ** 2 - y) / np.maximum(mu, 1e-9)
    denom = float((mu ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((z * mu).sum() / denom)


def fit_rate(
    team_games: pd.DataFrame,
    target: str,
    features: list[str],
    exposure_col: str = "game_poss",
    threshold: float = NEGBIN_DEVIANCE_DF_THRESHOLD,
) -> FittedModel:
    """Per-100-possession count GLM. Poisson unless the Poisson training
    deviance/df exceeds `threshold`, in which case NegBin at a fitted alpha."""
    d = team_games.dropna(subset=[target, exposure_col]).copy()
    d = d[d[exposure_col] > 0]
    med = _medians(d, features)
    x = _design(d, features, med)
    y = d[target].to_numpy(dtype=float)
    offset = np.log(d[exposure_col].to_numpy(dtype=float) / 100.0)

    pois = sm.GLM(y, x, family=sm.families.Poisson(), offset=offset).fit()
    dev_df = float(pois.deviance / pois.df_resid)
    use_nb = dev_df > threshold

    if use_nb:
        alpha = _nb2_alpha(y, np.asarray(pois.fittedvalues, dtype=float))
        alpha = float(max(alpha, 1e-8))
        res = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
        coef = np.asarray(res.params, dtype=float)
        family = "negbin"
    else:
        alpha = float("nan")
        coef = np.asarray(pois.params, dtype=float)
        family = "poisson"

    return FittedModel(
        kind="rate", target=target, features=list(features), coef=coef, medians=med,
        n_obs=int(len(y)), df_resid=int(pois.df_resid), family=family,
        deviance_df=dev_df, alpha=alpha,
        extra={"mean_per_100": float(100.0 * y.sum() / d[exposure_col].sum())},
    )


def _betabinom_rho(y: np.ndarray, n: np.ndarray, p: np.ndarray) -> float:
    """Williams' moment estimator for the Beta-Binomial intra-cluster
    correlation rho, from Var(y) = n p (1-p) [1 + (n-1) rho]."""
    v = n * p * (1.0 - p)
    num = float(((y - n * p) ** 2 - v).sum())
    den = float((v * (n - 1.0)).sum())
    if den <= 0:
        return 0.0
    return float(max(num / den, 0.0))


def fit_pct(
    team_games: pd.DataFrame,
    name: str,
    makes_col: str,
    trials_col: str,
    features: list[str],
) -> FittedModel:
    """Trials-weighted Binomial GLM (logit link) with a fitted Beta-Binomial
    overdispersion."""
    d = team_games.dropna(subset=[makes_col, trials_col]).copy()
    d = d[d[trials_col] > 0]
    med = _medians(d, features)
    x = _design(d, features, med)
    makes = d[makes_col].to_numpy(dtype=float)
    trials = d[trials_col].to_numpy(dtype=float)
    endog = np.column_stack([makes, trials - makes])
    res = sm.GLM(endog, x, family=sm.families.Binomial()).fit()
    p_hat = np.asarray(res.fittedvalues, dtype=float)
    rho = _betabinom_rho(makes, trials, p_hat)
    return FittedModel(
        kind="pct", target=name, features=list(features),
        coef=np.asarray(res.params, dtype=float), medians=med,
        n_obs=int(len(makes)), df_resid=int(res.df_resid), family="betabinomial",
        rho=rho,
        extra={
            "pooled_rate": float(makes.sum() / trials.sum()),
            "mean_trials": float(trials.mean()),
            "pearson_dispersion": float(res.pearson_chi2 / res.df_resid),
        },
    )


def fit_all(
    team_games: pd.DataFrame,
    games: pd.DataFrame,
    team_features: list[str],
    pace_features: list[str],
    fold: str,
    anchor: str,
    train_seasons: list[int],
    created_at: str = "",
) -> ControlModels:
    pace = fit_pace(games, pace_features)
    rates = {t: fit_rate(team_games, t, team_features) for t in RATE_TARGETS}
    pcts = {
        name: fit_pct(team_games, name, mk, tr, team_features)
        for name, (mk, tr) in PCT_TARGETS.items()
    }
    return ControlModels(
        fold=fold, anchor=anchor, pace=pace, rates=rates, pcts=pcts,
        train_seasons=list(train_seasons), created_at=created_at,
        n_games_train=int(len(games)),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save(bundle: ControlModels, model_dir: Path | str = DEFAULT_MODEL_DIR) -> dict[str, Path]:
    """Persist as the three artifacts model.md section 8 names, suffixed with
    the anchor: {pace,rates,pcts}_{fold}_{anchor}.pkl."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = {"fold": bundle.fold, "anchor": bundle.anchor, "train_seasons": bundle.train_seasons,
            "created_at": bundle.created_at, "n_games_train": bundle.n_games_train}
    paths = {}
    for kind, obj in (("pace", bundle.pace), ("rates", bundle.rates), ("pcts", bundle.pcts)):
        p = model_dir / f"{kind}_{bundle.fold}_{bundle.anchor}.pkl"
        with open(p, "wb") as fh:
            pickle.dump({"meta": meta, kind: obj}, fh)
        paths[kind] = p
    return paths


def load(fold: str, anchor: str, model_dir: Path | str = DEFAULT_MODEL_DIR) -> ControlModels:
    model_dir = Path(model_dir)
    out = {}
    meta = {}
    for kind in ("pace", "rates", "pcts"):
        p = model_dir / f"{kind}_{fold}_{anchor}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"missing control model artifact: {p}")
        with open(p, "rb") as fh:
            blob = pickle.load(fh)
        meta = blob["meta"]
        out[kind] = blob[kind]
    return ControlModels(
        fold=meta["fold"], anchor=meta["anchor"], pace=out["pace"],
        rates=out["rates"], pcts=out["pcts"], train_seasons=meta["train_seasons"],
        created_at=meta.get("created_at", ""), n_games_train=meta.get("n_games_train", 0),
    )


def summary_table(bundle: ControlModels) -> pd.DataFrame:
    rows = []
    p = bundle.pace
    rows.append({"kind": "pace", "target": p.target, "family": p.family, "n": p.n_obs,
                 "fitted_dispersion": f"resid_sd={p.resid_sd:.4f}", "deviance_df": np.nan,
                 "value": p.resid_sd})
    for t, m in bundle.rates.items():
        disp = f"alpha={m.alpha:.5f}" if m.family == "negbin" else "poisson (no alpha)"
        rows.append({"kind": "rate", "target": t, "family": m.family, "n": m.n_obs,
                     "fitted_dispersion": disp, "deviance_df": m.deviance_df, "value": m.alpha})
    for t, m in bundle.pcts.items():
        rows.append({"kind": "pct", "target": t, "family": m.family, "n": m.n_obs,
                     "fitted_dispersion": f"rho={m.rho:.5f}", "deviance_df": np.nan, "value": m.rho})
    return pd.DataFrame(rows)
