"""clock_r8.py -- clock round 8 (experiments.md section 36): league-relative, asymmetric
team tempo WITHOUT game-level over-spread, plus an in-season level term.

Lane H, 2026-10-01. NOT ADOPTED, NOT A DEFAULT. New module; no served object changes.

Same law family as round 7 (`clock_r7.AFTArmR7`): the P3n KM cell grid (no tempo tercile)
times a continuous accelerated-failure-time scale served through a continuous CDF,

    log k = sum_s [start == s] (b_o,s x_o + b_d,s x_d)  [+ b_t x_t]

with x_o / x_d the logs of the offence / defence team's tempo rating RELATIVE to the as-of
league mean, and x_t = days_since_start / 100 (the in-season level term). What changes is
the ESTIMATOR: round 7 used OLS on log(duration + 0.5), a geometric-mean elasticity, and then
applied it to the whole law, i.e. to the mean; round 8 fits the coefficients on the MEAN
scale (Poisson pseudo-likelihood, baseline-cell fixed effects profiled out), which is the
scale the possession count reads.

Arms: M1 (mean scale, round-7 rows: uncensored, sr >= 60), M2 (mean scale, every
uncensored regulation row), M2D (M2 + x_t), G2D (M2D with the team coefficients times a
training-fit game-level elasticity factor lambda).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cbb_sim.models import clock as ck
from cbb_sim.models import clock_v3 as c3
from cbb_sim.models import clock_r7 as r7

N_START = r7.N_START

#: arm -> (fit rows, uses the in-season term)
ARM_SPEC: dict[str, dict] = {
    "M1": {"rows": "sr60", "t": False},
    "M2": {"rows": "all", "t": False},
    "M2D": {"rows": "all", "t": True},
    "G2D": {"rows": "all", "t": True},
}
TEAM_FEATS = ("x_o", "x_d")
T_SCALE = 100.0


def add_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = r7.add_x(df)
    if "days_since_start" in df.columns:
        df["x_t"] = df["days_since_start"].to_numpy(dtype="float64") / T_SCALE
    return df


def season_day(design: pd.DataFrame) -> np.ndarray:
    """days_since_start as the engine inputs define it: game date minus the season's first
    game date (`pace.build_pace_table`). Recomputed here so a design row the builder filled
    with a median cannot leak a value the engine never serves."""
    gd = pd.to_datetime(design["game_date"])
    start = gd.groupby(design["season"]).transform("min")
    return (gd - start).dt.days.to_numpy(dtype="float64")


def design_matrix(df: pd.DataFrame, use_t: bool) -> np.ndarray:
    X = r7.design_matrix(df, TEAM_FEATS)
    if use_t:
        X = np.column_stack([X, df["x_t"].to_numpy(dtype="float64")])
    return X


def fit_rows_mask(tr: pd.DataFrame, rows: str) -> np.ndarray:
    m = (~tr["censored"].to_numpy(dtype=bool)) & (tr["period"].to_numpy() <= 2)
    if rows == "sr60":
        m &= tr["seconds_remaining"].to_numpy() >= 60
    return m


def fit_poisson(y: np.ndarray, X: np.ndarray, key: np.ndarray, max_iter: int = 100) -> tuple[np.ndarray, dict]:
    """Poisson pseudo-likelihood E[y] = mu_cell * exp(X b) with the cell effects profiled out
    (Newton on b with the Schur-complement Hessian). Deterministic."""
    _, inv = np.unique(key, return_inverse=True)
    keep = np.abs(X).sum(axis=0) > 0
    b = np.zeros(X.shape[1])
    ysum = np.bincount(inv, weights=y)
    it = 0
    for it in range(max_iter):
        eta = X @ b
        ee = np.exp(eta)
        mu_c = ysum / np.bincount(inv, weights=ee)
        mu = mu_c[inv] * ee
        g = X.T @ (y - mu)
        H = (X * mu[:, None]).T @ X
        wsum = np.bincount(inv, weights=mu)
        Xm = np.stack([np.bincount(inv, weights=mu * X[:, j]) for j in range(X.shape[1])], axis=1)
        # a cell whose every fitted duration is 0 has mu = 0 and carries no information on b
        inv_w = np.divide(1.0, wsum, out=np.zeros_like(wsum), where=wsum > 0)
        H = H - (Xm * inv_w[:, None]).T @ Xm
        step = np.zeros_like(b)
        step[keep] = np.linalg.solve(H[np.ix_(keep, keep)], g[keep])
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    return b, {"newton_iter": int(it + 1), "n_fit_rows": int(len(y))}


@dataclass
class AFTArmR8:
    """Baseline P3n cell law x continuous AFT scale fitted on the mean scale."""

    base: object
    arm: str
    use_t: bool
    coef: np.ndarray
    parametrisation: str = "P3"
    info: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"clock_r8_{self.arm}|P3"

    @property
    def needs_days_since_start(self) -> bool:
        return bool(self.use_t)

    def log_k(self, df: pd.DataFrame) -> np.ndarray:
        df = add_feats(df.copy())
        return design_matrix(df, self.use_t) @ self.coef

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        p0 = np.asarray(self.base.pmf(df), dtype="float64")
        return r7.scale_pmf(p0, np.exp(self.log_k(df)))


def fit_coef(arm: str, tr: pd.DataFrame) -> tuple[np.ndarray, dict]:
    spec = ARM_SPEC[arm]
    m = fit_rows_mask(tr, spec["rows"])
    t = add_feats(tr.loc[m].copy())
    X = design_matrix(t, spec["t"])
    y = t["duration_s"].to_numpy(dtype="float64")
    return fit_poisson(y, X, r7.cell_key(t))


def fit_base(arm: str, tr: pd.DataFrame, coef: np.ndarray, use_t: bool, info: dict) -> AFTArmR8:
    t = add_feats(tr.copy())
    k = np.exp(design_matrix(t, use_t) @ coef)
    scaled = tr.copy()
    scaled["duration_s"] = np.clip(np.rint(tr["duration_s"].to_numpy(dtype="float64") / k),
                                   0, ck.DURATION_CAP).astype("int64")
    sp = c3.add_p_state(scaled)
    inner = c3._fit_empirical_p(sp, "P3n_dummy", "P3", sr_floor_bucket=c3.SR_FLOOR_BUCKET)
    base = c3.StateWrapArm(inner, "P3")
    return AFTArmR8(base=base, arm=arm, use_t=use_t, coef=np.asarray(coef, dtype="float64"),
                    info={**info, "n_train": int(len(tr)), "k_sd": float(np.std(np.log(k)))})


def fit_arm(arm: str, tr: pd.DataFrame, lam: float | None = None) -> AFTArmR8:
    """One refit. `lam` (G2D only) multiplies the team coefficients, not b_t."""
    spec = ARM_SPEC[arm]
    coef, info = fit_coef(arm, tr)
    if lam is not None:
        coef = coef.copy()
        coef[: N_START * len(TEAM_FEATS)] *= lam
        info["lambda"] = float(lam)
    return fit_base(arm, tr, coef, spec["t"], info)


def game_elasticity(d: pd.DataFrame, e: np.ndarray) -> dict:
    """OLS elasticity of log team-game possessions (1200 / mean duration) on
    X = x_o + x_d (= log rel_home + log rel_away, identical on every row of a game) with
    month fixed effects; model (e = E[min(T,R)] per row) vs actual on the same games."""
    g = pd.DataFrame({"game_id": d["game_id"].to_numpy(), "e": e,
                      "a": d["duration_s"].to_numpy(dtype="float64"),
                      "X": (np.log(d["off_tempo_rel"].clip(0.5, 2.0)) +
                            np.log(d["def_tempo_rel"].clip(0.5, 2.0))).to_numpy(),
                      "mo": pd.to_datetime(d["game_date"]).dt.month.to_numpy()})
    gg = g.groupby("game_id").agg(e=("e", "mean"), a=("a", "mean"), X=("X", "first"), mo=("mo", "first"))
    M = pd.get_dummies(gg["mo"].astype(int), drop_first=True).to_numpy(dtype=float)
    Z = np.column_stack([np.ones(len(gg)), gg["X"].to_numpy(), M])
    out = {}
    for k in ("e", "a"):
        b, *_ = np.linalg.lstsq(Z, np.log(1200.0 / gg[k].to_numpy()), rcond=None)
        out[k] = float(b[1])
    return {"e_law": out["e"], "e_act": out["a"], "n_games": int(len(gg))}
