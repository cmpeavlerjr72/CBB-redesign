"""Foul-accrual calendar term (possession_outcome experiments.md section 32, 2026-10-07).

DEFAULT OFF. `ENGINE_FOUL_CAL` unset / "off" makes `load()` return None; `loop.py` then calls
`FoulJoint.accrual` exactly as before (no extra argument, no extra computation, no RNG change).

Arm `A2dbk`: the round-7 A2 accrual GBM refitted with a days-since-start bucket (0-7 / 8-14 / 15-30 /
31-45 / 46+) as one more feature, exported on round 7's grid with a leading 5-level days axis
(`scripts/build_foul_cal_lut_v1.py`, fold TRAIN fits only). It REPLACES the foul joint's accrual
probability for the defence; the possession's single `foul_accrual` uniform is unchanged, so no stream
moves. `days_since_start` is the engine inputs' pregame team column (identical to the training design's
on every matched game, s32.1). Nothing here is fitted to sim output.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

LUT_DIR = Path("data/processed/models/possession_outcome/round10cal")
FOLD_STEM = {"A2dbk": "lut_acc_A2dbk_F2"}       # fold-1 reads re-point LUT_DIR / this map
ENV = "ENGINE_FOUL_CAL"


class FoulCal:
    def __init__(self, arm: str, inp):
        if arm not in FOLD_STEM:
            raise KeyError(f"unknown {ENV}={arm!r}; known: {sorted(FOLD_STEM)} or 'off'")
        z = np.load(LUT_DIR / f"{FOLD_STEM[arm]}.npz")
        self.arm = arm
        self.lut = z["lut"]
        self.edges = z["dss_edges"]
        j = int(inp.team_names["days_since_start"])
        dss = inp.team_static[:, 0, j].astype(np.float64)
        if not np.isfinite(dss).all():
            raise ValueError("days_since_start has non-finite values in the engine inputs")
        self.bkt_game = np.searchsorted(self.edges, dss, side="left").astype(np.int64)

    def p_def(self, ix: tuple, gidx: np.ndarray) -> np.ndarray:
        return self.lut[(self.bkt_game[gidx],) + tuple(ix)]


def load(inp, fj) -> FoulCal | None:
    arm = os.environ.get(ENV, "off") or "off"
    if arm == "off":
        return None
    if fj is None:
        raise ValueError(f"{ENV} needs the foul joint (ENGINE_FOUL_JOINT must not be 'reference')")
    return FoulCal(arm, inp)
