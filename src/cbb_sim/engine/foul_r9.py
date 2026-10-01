"""Foul round 9 (possession-outcome experiments.md section 26, lane C, 2026-09-30).

DEFAULT OFF. Reached only through `ENGINE_FOUL_JOINT=<R9 arm>` (`foul_joint.load` delegates
any arm named `R9*` here). Every R9 arm is the round-8 `R8b` arm (A2 accrual + T2c trip
offsets, `foul_joint.FoulJoint`, unchanged) PLUS a fitted and-one model in place of the served
per-shot-class constant `rules.and_one_rate_given_made`:

* ``ao``  P(and-one | made FG) lookup exported from FOLD-2 TRAIN fits
  (`scripts/build_foul_r9_lut_v1.py`), indexed (shot class rim/jump2/three, period index
  H1/H2/OT, clock bucket on `seconds_remaining` with cuts 120/300/600/900). Arm AO3 adds a
  per-game logit term from PRIOR-season team rates (offence drawn, defence conceded, each
  centred on its own season's league mean): ``logit p = logit lut + b_off*c_off + b_def*c_def``.

The and-one draw keeps its own `and_one` uniform, so no RNG family is added; with the flag
unset or naming a non-R9 arm, `loop.py` never calls this module and the scalar path is taken.
Nothing here is fitted to sim output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cbb_sim.engine import foul_joint as FJ

LUT_DIR = Path("data/processed/models/possession_outcome/round9")
CLASS_ORDER = {"FGA_rim": 0, "FGA_jump2": 1, "FGA_3": 2}

#: arm -> (round-8 base arm, and-one lut stem)
ARMS = {
    "R9ao1": ("R8b", "lut_ao_AO1_F2"),
    "R9ao3": ("R8b", "lut_ao_AO3_F2"),
}


def _logit(p):
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


class FoulR9(FJ.FoulJoint):
    def __init__(self, arm: str):
        base, stem = ARMS[arm]
        super().__init__(base)
        self.arm = arm
        z = np.load(LUT_DIR / f"{stem}.npz")
        self.ao = {k: z[k] for k in z.files}
        self._team = None            # (n_games, 2) logit term with the OFFENCE on side 0/1

    def _team_term(self, inp) -> np.ndarray:
        if self._team is None:
            g = inp.games
            n = len(g)
            t = np.zeros((n, 2))
            if "team_b" in self.ao:
                import pandas as pd
                tab = pd.read_parquet(LUT_DIR / str(self.ao["team_table"]))
                key = dict(zip(zip(tab["season"].astype(int), tab["team_id"].astype(int)),
                               zip(tab["ao_off_prior_c"].astype(float), tab["ao_def_prior_c"].astype(float))))
                b_off, b_def = (float(x) for x in self.ao["team_b"])
                seas = g["season"].to_numpy().astype(int)
                home = g["home_team_id"].to_numpy().astype(int)
                away = g["away_team_id"].to_numpy().astype(int)
                for i in range(n):
                    h = key.get((seas[i], home[i]), (0.0, 0.0))
                    a = key.get((seas[i], away[i]), (0.0, 0.0))
                    t[i, 0] = b_off * h[0] + b_def * a[1]      # home on offence
                    t[i, 1] = b_off * a[0] + b_def * h[1]      # away on offence
            self._team = t
        return self._team

    def and_one_p(self, shot_class: str, inp, gidx, off_side, period, sec_rem) -> np.ndarray:
        c = CLASS_ORDER[shot_class]
        pi = np.where(period <= 1, 0, np.where(period == 2, 1, 2))
        ci = np.searchsorted(self.ao["clock_cuts"], sec_rem, side="right")
        p = self.ao["lut"][c, pi, ci]
        if "team_b" in self.ao:
            p = 1.0 / (1.0 + np.exp(-(_logit(p) + self._team_term(inp)[gidx, off_side])))
        return p


def load(arm: str) -> FoulR9:
    if arm not in ARMS:
        raise KeyError(f"unknown round-9 arm {arm!r}; known: {sorted(ARMS)}")
    return FoulR9(arm)
