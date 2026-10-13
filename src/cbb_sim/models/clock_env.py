"""clock_env.py -- clock round 9 (experiments.md section 41): scoring-environment and mismatch
response of possession duration. NOT ADOPTED, NOT A DEFAULT. New module; no served object changes.

Every arm keeps the served K2 law family (`clock_r8.KChanceArm`, cont_mode "K2": the P3 KM
first-chance cell law incl. the tempo tercile, srfloor, horn censoring, S1 monthly refits, the
end-class-conditional re-draw law, the continuation pmf; the game latent is the adapter's) and
multiplies BOTH first-chance laws by one continuous accelerated-failure-time scale served through
the round-7/8 continuous CDF (`clock_r7.scale_pmf`):

    log k = c * z_off [+ d * z_gap]             (E1, E2; E3 interacts c, d with the 6 start types)
    z_off = (off_c(offence) + def_c(defence)) / sd_off          (def_c > 0 = worse defence)
    z_gap = (|net(offence) - net(defence)| - mu_gap) / sd_gap,  net = off_c - def_c

`off_c` / `def_c` are our own as-of ratings, each relative to its own snapshot's league mean
(`as_of_date == game date` = fitted on games strictly before it); the engine frame already carries
them as `off_rating_off_c, off_rating_def_c, def_rating_off_c, def_rating_def_c` (Q1 pre-check:
offline design, live path and F2 engine inputs identical). `sd_off`, `mu_gap`, `sd_gap` are fixed
per fold from the fold's TRAINING seasons and stored in the artifact (a constant re-scaling; it
changes no fit).

Estimator (section 37's M2): Poisson pseudo-likelihood E[d1] = mu_cell exp(X b) on EVERY uncensored
regulation first-chance row, cell fixed effects profiled out (`clock_r8.fit_poisson`). The cell is
the K2 CONDITIONAL law's finest cell (P3 cell x tempo tercile x chance-1 end class), so the
coefficient is the within-outcome time response: the sim's outcome mix already responds to the
ratings through the possession-outcome model, and the clock consumes the conditional law.
The base K2 laws are then refitted on de-scaled durations round(d1 / k), as every AFT arm of
rounds 7-8. Continuation chances are not scaled (the draft scales the first-chance law).
No multiplier, cap, clip or blend on sim output: the scale is an input of the law.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cbb_sim.models import clock as ck
from cbb_sim.models import clock_v3 as c3
from cbb_sim.models import clock_r7 as r7
from cbb_sim.models import clock_r8 as r8

RATING_COLS = ("off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c")

#: arm -> (features, interacted with the start type)
ARM_SPEC: dict[str, dict] = {
    "E1": {"feats": ("z_off",), "by_start": False},
    "E2": {"feats": ("z_off", "z_gap"), "by_start": False},
    "E3": {"feats": ("z_off", "z_gap"), "by_start": True},
}


def raw_env(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per row: (off_c(offence) + def_c(defence), |net(offence) - net(defence)|)."""
    oo, od, do, dd = (df[c].to_numpy(dtype="float64") for c in RATING_COLS)
    return oo + dd, np.abs((oo - od) - (do - dd))


def fold_norms(tr: pd.DataFrame) -> dict:
    """sd_off, mu_gap, sd_gap over the regulation rows of the fold's training seasons."""
    t = tr[tr["period"].to_numpy() <= 2]
    e, g = raw_env(t)
    return {"sd_off": float(np.std(e)), "mu_gap": float(np.mean(g)), "sd_gap": float(np.std(g)),
            "n_rows": int(len(t))}


def add_z(df: pd.DataFrame, norms: dict) -> pd.DataFrame:
    e, g = raw_env(df)
    df["z_off"] = e / norms["sd_off"]
    df["z_gap"] = (g - norms["mu_gap"]) / norms["sd_gap"]
    return df


def design_matrix(df: pd.DataFrame, arm: str) -> np.ndarray:
    spec = ARM_SPEC[arm]
    if spec["by_start"]:
        return r7.design_matrix(df, spec["feats"])
    return np.column_stack([df[f].to_numpy(dtype="float64") for f in spec["feats"]])


def tempo_edges(tr: pd.DataFrame) -> tuple[float, float]:
    """The K2 cell law's own tercile edges (`clock_r8.fit_empirical_end` / `_fit_empirical_p`)."""
    t = tr["tempo_prior_game"].to_numpy(dtype="float64")
    return float(np.quantile(t, 1 / 3)), float(np.quantile(t, 2 / 3))


def cond_cell_key(df: pd.DataFrame, edges: tuple[float, float]) -> np.ndarray:
    """K2 conditional finest cell: P3 cell (start, floored clock bucket, period type, end-game regime)
    x tempo tercile x chance-1 end class."""
    terc = np.searchsorted(np.asarray(edges), df["tempo_prior_game"].to_numpy(dtype="float64"), side="right")
    return (r7.cell_key(df) * 3 + terc) * len(r8.END1_LEVELS) + df["end1_code"].to_numpy().astype("int64")


@dataclass
class EnvAFT:
    """A K2 first-chance law (marginal or conditional) x the round-9 AFT scale."""

    base: object
    arm: str
    coef: np.ndarray
    norms: dict
    which: str                      # "marginal" | "conditional"
    parametrisation: str = "P3"

    def log_k(self, df: pd.DataFrame) -> np.ndarray:
        t = add_z(pd.DataFrame({c: df[c].to_numpy() for c in RATING_COLS}), self.norms)
        if ARM_SPEC[self.arm]["by_start"]:
            t["prev_end"] = df["prev_end"].to_numpy()
        return design_matrix(t, self.arm) @ self.coef

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        p0 = np.asarray(self.base.pmf(df), dtype="float64")
        return r7.scale_pmf(p0, np.exp(self.log_k(df)))


def fit_coef(arm: str, tr_d1: pd.DataFrame, norms: dict) -> tuple[np.ndarray, dict]:
    m = (~tr_d1["censored"].to_numpy(dtype=bool)) & (tr_d1["period"].to_numpy() <= 2)
    t = add_z(tr_d1.loc[m].copy(), norms)
    edges = tempo_edges(tr_d1)
    X = design_matrix(t, arm)
    y = t["duration_s"].to_numpy(dtype="float64")
    b, info = r8.fit_poisson(y, X, cond_cell_key(t, edges))
    return b, {**info, "tempo_edges": list(edges)}


def fit_env_arm(arm: str, tr_d1: pd.DataFrame, cont_pmf: np.ndarray, norms: dict, seed: int = 0) -> r8.KChanceArm:
    """One refit: coefficients (M2 estimator), then both K2 first-chance laws on de-scaled d1."""
    coef, info = fit_coef(arm, tr_d1, norms)
    t = add_z(tr_d1.copy(), norms)
    k = np.exp(design_matrix(t, arm) @ coef)
    scaled = tr_d1.copy()
    scaled["duration_s"] = np.clip(np.rint(tr_d1["duration_s"].to_numpy(dtype="float64") / k),
                                   0, ck.DURATION_CAP).astype("int64")
    marg = c3.fit_arm_v3b("empirical_km3_srfloor", "P3", scaled, seed=seed)
    cond = r8.fit_empirical_end(scaled, "P3e_dummy")
    info = {"coef": np.asarray(coef).round(6).tolist(), "norms": norms, "k_sd": float(np.std(np.log(k))),
            "k_mean": float(np.mean(k)), **info}
    return r8.KChanceArm(marginal=EnvAFT(marg, arm, coef, norms, "marginal"),
                         conditional=EnvAFT(cond, arm, coef, norms, "conditional"),
                         cont_pmf=cont_pmf, arm=f"env{arm}", cont_mode="K2", info=info)
