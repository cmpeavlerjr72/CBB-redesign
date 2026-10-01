"""late_foul.py -- late foul accrual (possession_outcome experiments.md section 29), DEFAULT OFF.

`ENGINE_LATE_FOUL` unset / `off` -> `load()` returns None and `loop.py` takes exactly the served path.

Inside the window (period 2, possession start <= 120 s; regulation only) the served accrual draw (a Bernoulli on
`lut_acc_A2_F2`, a table that was never fitted on these rows and caps the defence at one non-trip foul per
possession) is replaced by a fitted COUNT law of the defence's non-trip fouls, K in {0..3}, by defence role x
start clock x defence team fouls (`scripts/train_late_foul_v1.py`, fold-2 train seasons). The draw uses THE SAME
`foul_accrual` uniform (inverse CDF), so no RNG stream is added and every other draw is unchanged. Fitted on data,
never on sim output.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

DIR = Path("data/processed/models/possession_outcome/late_foul")
ARMS = ("LF1", "LF2")


class LateFoul:
    def __init__(self, arm: str):
        d = json.loads((DIR / f"late_foul_{arm}_F2.json").read_text(encoding="utf-8"))
        self.arm = arm
        self.cdf = np.cumsum(np.asarray(d["probs"], dtype=np.float64), axis=-1)
        self.cdf[..., -1] = 1.0
        self.clock_edges = np.asarray(d["clock_edges"])
        self.foul_edges = np.asarray(d["foul_edges"])
        self.n_window = 0
        self.n_fouls = 0

    def count(self, u, period, sec0, sd0, def_f0) -> np.ndarray:
        """K per row; -1 outside the window (the served draw stands there)."""
        k = np.full(len(u), -1, dtype=np.int64)
        w = (np.asarray(period) == 2) & (np.asarray(sec0) <= 120)
        if not w.any():
            return k
        r = np.flatnonzero(w)
        om = np.asarray(sd0)[r]
        a = np.abs(om)
        role = np.where(om > 0, np.where(a <= 3, 0, np.where(a <= 6, 1, 2)), np.where(om == 0, 3, 4))
        ci = np.searchsorted(self.clock_edges, np.asarray(sec0)[r], side="left").clip(0, 2)
        fi = np.searchsorted(self.foul_edges, np.asarray(def_f0)[r], side="left").clip(0, 3)
        c = self.cdf[role, ci, fi]
        k[r] = (np.asarray(u)[r][:, None] > c).sum(axis=1)
        self.n_window += len(r)
        self.n_fouls += int(k[r].sum())
        return k


def load():
    mode = os.environ.get("ENGINE_LATE_FOUL", "off") or "off"
    if mode == "off":
        return None
    if mode not in ARMS:
        raise ValueError(f"ENGINE_LATE_FOUL={mode!r}: expected one of {ARMS}")
    return LateFoul(mode)
