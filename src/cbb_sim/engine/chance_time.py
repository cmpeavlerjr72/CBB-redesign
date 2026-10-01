"""Chance-time feed to fg_make / possession_outcome (lane I, 2026-09-30).

Pre-registration: `docs/models/chance_time/experiments.md` section 1.

DEFAULT OFF. `ENGINE_CHANCE_TIME` unset (or "reference") makes `load()` return
None and `loop.py` takes exactly the served path: no draw, no stream family
touched, bit-identical.

The served engine feeds fg_make, at chance 1, `chance_elapsed_s` = the whole
possession's drawn duration and `is_transition` = (duration <= 8 s and the
previous possession ended DREB/TOV); at chance >= 2 it feeds the training
median elapsed time by chance number. The training tables measure chance 1
only (fg_make: chance start to the shot; possession_outcome chance rows:
chance-1 duration) and the real chance >= 2 elapsed time is widely spread.

* ``C2``   chance >= 2: `chance_elapsed_s` drawn by inverse CDF from the
           TRAINING distribution of the fg_make design's own column in the cell
           (chance bucket {2, 3+} x shot class). Chance 1 served.
* ``C12``  C2 plus chance 1: `e1 = round(r * used)`, `r` drawn from the TRAINING
           distribution of chance-1 duration / possession duration in the cell
           (possession-duration bin x start group {DREB/TOV, other}); fg_make's
           chance-1 elapsed = min(e1, 60) and transition = (e1 <= 8 and the
           previous end DREB/TOV), fed to possession_outcome `first` and fg_make.

Tables: `data/processed/models/chance_time/<fold>/lut_v1.npz`
(`scripts/build_chance_time_lut_v1.py`, training seasons only). Draws come from
a stream family `chance_time` on its own StreamBook keyed (seed, game_id), so no
other family's stream moves. Nothing here is fitted to sim output and nothing
adjusts a model output: it is the state the models are fed.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

FAMILY = "chance_time"
ENV = "ENGINE_CHANCE_TIME"
ARMS = ("C2", "C12", "K", "KD")
LUT3 = Path("data/processed/models/chance_time/{fold}/lut_v3.npz")   # addendum 1c (arm KD)
LUT = Path("data/processed/models/chance_time/{fold}/lut_v1.npz")
LUT2 = Path("data/processed/models/chance_time/{fold}/lut_v2.npz")   # addendum 1b (arm K)
KEY_OF_SHOT = {"FGA_rim": 0, "FGA_jump2": 1, "FGA_3": 2}
TRANS_MAX_S = 8.0
MAX_ELAPSED_S = 60.0


class ChanceTime:
    def __init__(self, arm: str, seeds, game_ids, fold: str = "F2"):
        if arm not in ARMS:
            raise KeyError(f"unknown {ENV}={arm!r}; known: {ARMS} or 'reference'")
        from cbb_sim.engine.rng import StreamBook
        self.arm = arm
        z = np.load(str({"K": LUT2, "KD": LUT3}.get(arm, LUT)).format(fold=fold))
        self.c1_q = z["c1_q"] if arm in ("K", "KD") else None    # (2, 3, Q): start group x class
        self.c1d_q = z["c1d_q"] if arm == "KD" else None         # (2, 3, B, Q): ... x duration bin
        self.cont_q = z["cont_q"]              # (2, 3, Q)
        self.r_q = z["r_q"]                    # (2, B, Q)
        self.bin_edges = z["bin_edges"]
        self.nq = self.cont_q.shape[2]
        self.book = StreamBook(seeds, game_ids, families=(FAMILY,))

    def _qi(self, u: np.ndarray) -> np.ndarray:
        return np.minimum(np.round(u * (self.nq - 1)).astype(np.int64), self.nq - 1)

    def first(self, act: np.ndarray, used: np.ndarray, prev_tr: np.ndarray,
              trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Chance-1 elapsed and transition for the possessions `act` (sim rows).
        C2 returns the served pair unchanged and draws nothing."""
        if self.arm != "C12":
            return used, trans
        u = self.book.draw(FAMILY, act)
        g = np.where(prev_tr, 0, 1)
        b = np.clip(np.searchsorted(self.bin_edges, used.astype(np.int64), side="right") - 1,
                    0, self.r_q.shape[1] - 1)
        e1r = np.round(self.r_q[g, b, self._qi(u)] * used)
        tr = ((e1r <= TRANS_MAX_S) & prev_tr).astype(np.float64)
        return np.minimum(e1r, MAX_ELAPSED_S), tr

    def cont(self, xs: np.ndarray, col: int, chance: np.ndarray, shot_class: str,
             sim_rows: np.ndarray, prev_tr: np.ndarray | None = None, idx: dict | None = None,
             dur: np.ndarray | None = None) -> None:
        """In place: chance >= 2 rows of the fg_make state block get a drawn elapsed.
        Arm K also draws chance-1 rows' elapsed from the class x start-group table (addendum 1b)
        and sets fg_make's transition flag from it; one draw per row, every row of the call.
        Arm KD conditions that table also on the possession's drawn duration `dur` (addendum 1c)."""
        if self.arm in ("K", "KD"):
            u = self.book.draw(FAMILY, sim_rows)
            k = KEY_OF_SHOT[shot_class]
            first = chance < 2
            g = np.where(prev_tr, 0, 1)
            qi = self._qi(u)
            bk = np.where(chance >= 3, 1, 0)
            if self.arm == "KD":
                db = np.clip(np.searchsorted(self.bin_edges, dur.astype(np.int64), side="right") - 1,
                             0, self.c1d_q.shape[2] - 1)
                c1 = self.c1d_q[g, k, db, qi]
            else:
                c1 = self.c1_q[g, k, qi]
            el = np.where(first, c1, self.cont_q[bk, k, qi])
            lim = np.minimum(xs[:, idx["seconds_remaining"]], MAX_ELAPSED_S)
            if self.arm == "KD":
                lim = np.minimum(lim, dur)
            el = np.where(first, np.minimum(el, lim), el)
            xs[:, col] = el
            xs[:, idx["is_transition_f"]] = np.where(first, ((el <= TRANS_MAX_S) & prev_tr).astype(np.float64), 0.0)
            return
        c2 = np.flatnonzero(chance >= 2)
        if not len(c2):
            return
        u = self.book.draw(FAMILY, sim_rows[c2])
        bk = np.where(chance[c2] >= 3, 1, 0)
        xs[c2, col] = self.cont_q[bk, KEY_OF_SHOT[shot_class], self._qi(u)]


#: SERVED DEFAULT, ADOPTED 2026-10-01 (Decision 11 set, PM under user delegation;
#: docs/tests/adoption_served_v2_2026-10-01.md). `reference` reproduces served-v1.
DEFAULT = "KD"


def load(seeds, game_ids) -> ChanceTime | None:
    arm = os.environ.get(ENV, DEFAULT)
    if arm in ("", "reference"):
        return None
    return ChanceTime(arm, seeds, game_ids)
