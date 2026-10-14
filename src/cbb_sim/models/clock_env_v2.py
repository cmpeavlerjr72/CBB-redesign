"""clock_env_v2.py -- clock experiments.md section 46 block C: bounded scoring-environment arms on the served K2 law.
NOT ADOPTED, NOT A DEFAULT, NOT WIRED (no engine flag; offline arms behind the C-pre gate only).

Same law family and estimator as round 9 (`clock_env`, section 41.B): each arm is a `clock_r8.KChanceArm` (cont_mode
K2) whose marginal AND end-class-conditional first-chance laws are the K2 cell laws refitted on de-scaled d1
(round(d1 / k)) and served through `clock_r7.scale_pmf` at the same k; the coefficients come from the round-8 M2
Poisson pseudo-likelihood with the K2 CONDITIONAL finest cell as fixed effects, on every uncensored regulation
first-chance row. Only the log-k feature construction differs (s46 2.C; the bound is a hypothesis C-pre tests):

    ED   c_o * q(off_c(offence)) + c_d * q(def_c(defence))
         q = within-snapshot quantile score in [-0.5, 0.5]: (rank - 0.5) / N - 0.5 of the team's value among every
         team of its own as-of snapshot (season, as_of_date == game date; `own_ratings` batch files). def_c > 0 =
         worse defence, so a high q(def_c) is a weak defence.
    EG   c * z_env_game, z_env_game = env / (2 sd_off), env = off_c(o) + def_c(o) + off_c(d) + def_c(d): the mean of
         the two offence/defence pair values (off_c(home) + def_c(away), off_c(away) + def_c(home)) / sd_off;
         identical for both offences of a game.
    EM   d * z_gap, z_gap = (|net(o) - net(d)| - mu_gap) / sd_gap (round 9's centred mismatch), no env term.
    EH   c_1 * z_off on first-half rows, c_2 * z_off on second-half (and overtime) rows; z_off as round 9 E1. Fitted
         as two separate fits by giving each half its own cell fixed effects (cell key x half), which is identical
         to fitting each half alone.

Features are league-centred at the as-of snapshot (`off_c`, `def_c` relative to their snapshot's league mean; q is a
rank within the snapshot); sd_off, mu_gap, sd_gap are fixed per fold on the fold's training seasons (`clock_env`).
Home/away/neutral stays in the K2 base law (no arm removes a K2 cell dimension). No multiplier, cap, clip or blend
on sim output: the scale is an input of the first-chance law.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.models import clock as ck
from cbb_sim.models import clock_env as CE
from cbb_sim.models import clock_r8 as r8
from cbb_sim.models import clock_v3 as c3

RATINGS_DIR = Path(__file__).resolve().parents[3] / "data/processed/ratings"
ARMS = ("ED", "EG", "EM", "EH")
FEATS = {"ED": ("q_off_o", "q_def_d"), "EG": ("z_env_game",), "EM": ("z_gap",), "EH": ("z_off_h1", "z_off_h2")}
EXTRA_COLS = ("season", "game_date", "offense_team_id", "defense_team_id", "period")


@lru_cache(maxsize=1)
def q_table() -> pd.DataFrame:
    """(season, as_of_date, team_id) -> within-snapshot quantile scores of off_c and def_c in [-0.5, 0.5]."""
    parts = []
    for p in sorted(RATINGS_DIR.glob("own_ratings_20*.parquet")):
        r = pd.read_parquet(p, columns=["season", "as_of_date", "team_id", "off_c", "def_c"])
        if int(r["season"].max()) > 2025:
            continue                                   # 2025-26 sealed; never read
        g = r.groupby(["season", "as_of_date"])
        n = g["team_id"].transform("size").to_numpy(np.float64)
        r["q_off"] = (g["off_c"].rank(method="average").to_numpy() - 0.5) / n - 0.5
        r["q_def"] = (g["def_c"].rank(method="average").to_numpy() - 0.5) / n - 0.5
        parts.append(r[["season", "as_of_date", "team_id", "q_off", "q_def"]])
    t = pd.concat(parts, ignore_index=True)
    t["as_of_date"] = pd.to_datetime(t["as_of_date"]).astype("datetime64[ns]")
    t["season"] = t["season"].astype("int64")
    t["team_id"] = t["team_id"].astype("int64")
    return t


def add_q(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """q(off_c(offence)), q(def_c(defence)) for each row (as-of snapshot of the game date)."""
    t = q_table()
    k = pd.DataFrame({"season": df["season"].to_numpy().astype("int64"),
                      "as_of_date": pd.to_datetime(df["game_date"]).to_numpy().astype("datetime64[ns]"),
                      "o": df["offense_team_id"].to_numpy().astype("int64"),
                      "d": df["defense_team_id"].to_numpy().astype("int64")})
    a = k.merge(t.rename(columns={"team_id": "o"})[["season", "as_of_date", "o", "q_off"]],
                on=["season", "as_of_date", "o"], how="left")
    b = k.merge(t.rename(columns={"team_id": "d"})[["season", "as_of_date", "d", "q_def"]],
                on=["season", "as_of_date", "d"], how="left")
    qo, qd = a["q_off"].to_numpy(np.float64), b["q_def"].to_numpy(np.float64)
    bad = ~(np.isfinite(qo) & np.isfinite(qd))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows without an as-of snapshot quantile")
    return qo, qd


def features(df: pd.DataFrame, arm: str, norms: dict) -> np.ndarray:
    oo, od, do, dd = (df[c].to_numpy(dtype="float64") for c in CE.RATING_COLS)
    if arm == "ED":
        return np.column_stack(add_q(df))
    if arm == "EG":
        return ((oo + od + do + dd) / (2.0 * norms["sd_off"]))[:, None]
    if arm == "EM":
        g = np.abs((oo - od) - (do - dd))
        return ((g - norms["mu_gap"]) / norms["sd_gap"])[:, None]
    if arm == "EH":
        z = (oo + dd) / norms["sd_off"]
        h1 = df["period"].to_numpy() <= 1
        return np.column_stack([np.where(h1, z, 0.0), np.where(h1, 0.0, z)])
    raise KeyError(arm)


@dataclass
class EnvAFT2:
    """A K2 first-chance law (marginal or conditional) x a block-C AFT scale."""

    base: object
    arm: str
    coef: np.ndarray
    norms: dict
    which: str
    parametrisation: str = "P3"

    def log_k(self, df: pd.DataFrame) -> np.ndarray:
        return features(df, self.arm, self.norms) @ self.coef

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        from cbb_sim.models import clock_r7 as r7
        p0 = np.asarray(self.base.pmf(df), dtype="float64")
        return r7.scale_pmf(p0, np.exp(self.log_k(df)))


def fe_key(df: pd.DataFrame, arm: str, edges: tuple[float, float]) -> np.ndarray:
    key = CE.cond_cell_key(df, edges)
    if arm == "EH":
        key = key * 2 + (df["period"].to_numpy() > 1).astype("int64")
    return key


def fit_coef(arm: str, tr_d1: pd.DataFrame, norms: dict) -> tuple[np.ndarray, dict]:
    m = (~tr_d1["censored"].to_numpy(dtype=bool)) & (tr_d1["period"].to_numpy() <= 2)
    t = tr_d1.loc[m]
    edges = CE.tempo_edges(tr_d1)
    X = features(t, arm, norms)
    y = t["duration_s"].to_numpy(dtype="float64")
    b, info = r8.fit_poisson(y, X, fe_key(t, arm, edges))
    return b, {**info, "tempo_edges": list(edges), "feats": list(FEATS[arm])}


def fit_env_arm(arm: str, tr_d1: pd.DataFrame, cont_pmf: np.ndarray, norms: dict, seed: int = 0) -> r8.KChanceArm:
    """One refit: coefficients (M2 estimator), then both K2 first-chance laws on de-scaled d1 (as round 9)."""
    coef, info = fit_coef(arm, tr_d1, norms)
    k = np.exp(features(tr_d1, arm, norms) @ coef)
    scaled = tr_d1.copy()
    scaled["duration_s"] = np.clip(np.rint(tr_d1["duration_s"].to_numpy(dtype="float64") / k),
                                   0, ck.DURATION_CAP).astype("int64")
    marg = c3.fit_arm_v3b("empirical_km3_srfloor", "P3", scaled, seed=seed)
    cond = r8.fit_empirical_end(scaled, "P3e_dummy")
    info = {"coef": np.asarray(coef).round(6).tolist(), "norms": norms, "k_sd": float(np.std(np.log(k))),
            "k_mean": float(np.mean(k)), **info}
    return r8.KChanceArm(marginal=EnvAFT2(marg, arm, coef, norms, "marginal"),
                         conditional=EnvAFT2(cond, arm, coef, norms, "conditional"),
                         cont_pmf=cont_pmf, arm=f"s46C{arm}", cont_mode="K2", info=info)
