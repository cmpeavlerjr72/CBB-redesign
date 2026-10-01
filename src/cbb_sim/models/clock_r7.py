"""clock_r7.py -- clock round 7 (experiments.md section 32): team pace responsiveness.

Lane H, 2026-09-30. NOT ADOPTED, NOT A DEFAULT. New module; no served object
changes.

The served cell law (`clock_v3` `empirical_km3_srfloor|P3`) carries the team
only through a TERCILE of the symmetric game prior `tempo_prior_game`. Round 7
replaces that dimension with a continuous accelerated-failure-time (AFT) time
scale on top of the same KM cell grid without the tempo dimension:

    T = k * T0,   T0 ~ cell law,   log k = sum_j b_{j,s} x_j

with `s` the possession's start type (`prev_end`) and `x` the arm's features
(`x_o = log off_tempo_rel`, `x_d = log def_tempo_rel`, and for A3 the as-of
team random effects `re_o`, `re_d`). Serving maps the baseline pmf through a
CONTINUOUS CDF (mass at integer j spread uniformly over [j-0.5, j+0.5)), so a
small k is never the identity (section 31's degeneracy):

    p_k(t) = F0((t+0.5)/k) - F0((t-0.5)/k),  mass past the cap kept at the cap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cbb_sim.models import clock as ck
from cbb_sim.models import clock_v3 as c3

#: The L2 / P3 cell grid with the tempo tercile REMOVED.
ck.EMPIRICAL_DIMS["P3n_dummy"] = ("prev_end_code", "r2_bucket_code", "r2_period_type", "eg_regime")
ck.FEATURE_SETS.setdefault("P3n_dummy", ck.FEATURE_SETS["P3_dummy"])

N_START = len(ck.PREV_END_LEVELS)

#: arm -> feature columns of log k (each interacted with the start type)
ARM_FEATURES: dict[str, tuple[str, ...]] = {
    "A1": ("x_sym",),
    "A2": ("x_o", "x_d"),
    "A3": ("x_o", "x_d", "re_o", "re_d"),
}


def add_x(df: pd.DataFrame) -> pd.DataFrame:
    """The relative tempo features. `off_tempo_rel`/`def_tempo_rel` are already
    relative to the as-of league mean (own ratings), so their log is centred."""
    o = np.log(np.clip(df["off_tempo_rel"].to_numpy(dtype="float64"), 0.5, 2.0))
    d = np.log(np.clip(df["def_tempo_rel"].to_numpy(dtype="float64"), 0.5, 2.0))
    df["x_o"] = o
    df["x_d"] = d
    df["x_sym"] = o + d
    return df


def start_code(df: pd.DataFrame) -> np.ndarray:
    return df["prev_end"].map(ck.PREV_END_INDEX).fillna(0).to_numpy().astype("int64")


def design_matrix(df: pd.DataFrame, feats: tuple[str, ...]) -> np.ndarray:
    """Start-type-interacted features: column (s, j) = x_j * [start == s]."""
    s = start_code(df)
    X = np.zeros((len(df), N_START * len(feats)), dtype="float64")
    for j, f in enumerate(feats):
        x = df[f].to_numpy(dtype="float64")
        X[np.arange(len(df)), s * len(feats) + j] = x
    return X


def cell_key(df: pd.DataFrame) -> np.ndarray:
    sr = df["seconds_remaining"].to_numpy()
    b = np.maximum(ck.r2_bucket_id(sr), c3.SR_FLOOR_BUCKET)
    pt = ck.r2_period_type(df["period"].to_numpy())
    eg = c3.eg_regime_code(sr, df["period"].to_numpy(), df["score_diff"].to_numpy())
    return ((start_code(df) * 16 + b) * 4 + pt) * 4 + eg


def fit_coefs(tr: pd.DataFrame, feats: tuple[str, ...]) -> np.ndarray:
    """OLS of log(duration + 0.5) on the interacted features with baseline-cell
    fixed effects, on uncensored rows with >= 60 s left (pre-registered)."""
    m = (~tr["censored"].to_numpy(dtype=bool)) & (tr["seconds_remaining"].to_numpy() >= 60)
    t = tr.loc[m]
    y = np.log(t["duration_s"].to_numpy(dtype="float64") + 0.5)
    X = design_matrix(t, feats)
    key = cell_key(t)
    _, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv).astype("float64")
    y = y - (np.bincount(inv, weights=y) / cnt)[inv]
    for j in range(X.shape[1]):
        X[:, j] -= (np.bincount(inv, weights=X[:, j]) / cnt)[inv]
    keep = np.abs(X).sum(axis=0) > 0
    b = np.zeros(X.shape[1])
    b[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    return b


def scale_pmf(p0: np.ndarray, k: np.ndarray) -> np.ndarray:
    """p_k(t) = F0((t+0.5)/k) - F0((t-0.5)/k) on the integer grid, F0 the
    piecewise-linear CDF with knots at half-integers; mass past the cap at the cap."""
    n, g = p0.shape
    knots = np.concatenate([np.zeros((n, 1)), np.cumsum(p0, axis=1)], axis=1)  # knot i at v = i - 0.5

    def F(v):
        pos = np.clip(v + 0.5, 0.0, g)
        i0 = np.minimum(np.floor(pos).astype(np.int64), g - 1)
        fr = pos - i0
        lo = np.take_along_axis(knots, i0, axis=1)
        hi = np.take_along_axis(knots, i0 + 1, axis=1)
        return lo + fr * (hi - lo)

    grid = np.arange(g, dtype="float64")[None, :]
    kk = np.asarray(k, dtype="float64")[:, None]
    up = F((grid + 0.5) / kk)
    dn = F((grid - 0.5) / kk)
    out = up - dn
    out[:, -1] += 1.0 - up[:, -1]
    return np.clip(out, 0.0, None)


@dataclass
class AFTArmR7:
    """Baseline cell law (no tempo dimension) x continuous AFT team scale."""

    base: object                     # StateWrapArm around EmpiricalArmV3P (P3n cells)
    arm: str
    feats: tuple[str, ...]
    coef: np.ndarray
    parametrisation: str = "P3"
    info: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"clock_r7_{self.arm}|P3"

    def log_k(self, df: pd.DataFrame) -> np.ndarray:
        df = add_x(df.copy()) if "x_o" not in df.columns else df
        return design_matrix(df, self.feats) @ self.coef

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        p0 = np.asarray(self.base.pmf(df), dtype="float64")
        return scale_pmf(p0, np.exp(self.log_k(df)))


def fit_arm(arm: str, tr: pd.DataFrame) -> AFTArmR7:
    """Fit one round-7 arm on one refit's training rows (design already carries
    `censored`, and for A3 `re_o`/`re_d`)."""
    feats = ARM_FEATURES[arm]
    tr = add_x(tr.copy()) if "x_o" not in tr.columns else tr
    b = fit_coefs(tr, feats)
    k = np.exp(design_matrix(tr, feats) @ b)
    scaled = tr.copy()
    scaled["duration_s"] = np.clip(np.rint(tr["duration_s"].to_numpy(dtype="float64") / k),
                                   0, ck.DURATION_CAP).astype("int64")
    sp = c3.add_p_state(scaled)
    inner = c3._fit_empirical_p(sp, "P3n_dummy", "P3", sr_floor_bucket=c3.SR_FLOOR_BUCKET)
    base = c3.StateWrapArm(inner, "P3")
    return AFTArmR7(base=base, arm=arm, feats=feats, coef=b,
                    info={"n_train": int(len(tr)), "k_sd": float(np.std(np.log(k)))})


# ---------------------------------------------------------------------------
# A3: walk-forward partial-pooled team effects with prior-season carry
# ---------------------------------------------------------------------------
def team_re(design: pd.DataFrame, train_seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    """As-of (strictly earlier dates) shrunk team offence/defence duration effects.

    u = log(duration+0.5) minus its baseline-cell mean (cell means from the
    fold's TRAINING seasons), on uncensored rows with >= 60 s left. Per team
    and season, the as-of sum S and count n of u over earlier dates; the as-of
    league mean lg on the same dates; prev = the team's previous-season final
    mean minus that season's league mean. re = (S - n*lg + kappa*c*prev)/(n+kappa)
    with c and kappa estimated on the training seasons only.
    Returns per-row re_o, re_d aligned to design.index, and the parameters."""
    d = design
    elig = (~d["censored"].to_numpy(dtype=bool)) & (d["seconds_remaining"].to_numpy() >= 60)
    key = cell_key(d)
    ly = np.log(d["duration_s"].to_numpy(dtype="float64") + 0.5)
    trm = elig & d["season"].isin(train_seasons).to_numpy()
    cm = pd.Series(ly[trm]).groupby(key[trm]).mean()
    mu = pd.Series(key).map(cm).to_numpy()
    mu = np.where(np.isfinite(mu), mu, np.nanmean(ly[trm]))
    u = np.where(elig, ly - mu, np.nan)
    date = pd.to_datetime(d["game_date"]).to_numpy()
    out = {}
    params = {}
    for side, col in (("o", "offense_team_id"), ("d", "defense_team_id")):
        f = pd.DataFrame({"season": d["season"].to_numpy(), "team": d[col].to_numpy(),
                          "date": date, "u": u, "e": elig.astype(float)})
        f["u0"] = f["u"].fillna(0.0)
        g = f.groupby(["season", "team", "date"]).agg(S=("u0", "sum"), n=("e", "sum")).reset_index()
        g = g.sort_values(["season", "team", "date"])
        g["S_prev"] = g.groupby(["season", "team"])["S"].cumsum() - g["S"]
        g["n_prev"] = g.groupby(["season", "team"])["n"].cumsum() - g["n"]
        lgd = f.groupby(["season", "date"]).agg(S=("u0", "sum"), n=("e", "sum")).reset_index()
        lgd = lgd.sort_values(["season", "date"])
        lgd["lg"] = ((lgd.groupby("season")["S"].cumsum() - lgd["S"]) /
                     (lgd.groupby("season")["n"].cumsum() - lgd["n"]).replace(0, np.nan))
        g = g.merge(lgd[["season", "date", "lg"]], on=["season", "date"], how="left")
        g["lg"] = g["lg"].fillna(0.0)
        # season finals (team mean minus league mean of that season)
        fin = f[f["e"] > 0].groupby(["season", "team"])["u"].agg(["mean", "count", "var"]).reset_index()
        lgs = f[f["e"] > 0].groupby("season")["u"].mean()
        fin["m"] = fin["mean"] - fin["season"].map(lgs)
        prev = fin[["season", "team", "m"]].copy()
        prev["season"] = prev["season"] + 1
        g = g.merge(prev.rename(columns={"m": "prev"}), on=["season", "team"], how="left")
        # carry c and kappa from TRAINING seasons only
        ft = fin[fin["season"].isin(train_seasons)]
        pairs = ft.merge(prev.rename(columns={"m": "prev"}), on=["season", "team"])
        c = float(np.polyfit(pairs["prev"], pairs["m"], 1)[0]) if len(pairs) > 30 else 0.0
        sigma2 = float(np.nanmean(ft["var"]))
        tau2 = float(np.var(ft["m"]) - np.mean(sigma2 / ft["count"]))
        kappa = sigma2 / max(tau2, 1e-6)
        g["prev"] = g["prev"].fillna(0.0)
        g["re"] = (g["S_prev"] - g["n_prev"] * g["lg"] + kappa * c * g["prev"]) / (g["n_prev"] + kappa)
        params[side] = {"carry_c": c, "kappa": kappa, "sigma2": sigma2, "tau2": tau2,
                        "n_pairs": int(len(pairs))}
        m = pd.DataFrame({"season": f["season"], "team": f["team"], "date": f["date"]})
        m = m.merge(g[["season", "team", "date", "re"]], on=["season", "team", "date"], how="left")
        out[f"re_{side}"] = m["re"].fillna(0.0).to_numpy()
    return pd.DataFrame(out, index=design.index), params
