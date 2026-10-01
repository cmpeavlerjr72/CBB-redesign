"""Shared (game-level) shooting latent on the fg_make logit (lane B, 2026-09-30).

Pre-registration: `docs/models/shared_shooting/experiments.md` section 1.

DEFAULT OFF. `ENGINE_SHARED_SHOOTING` unset (or "reference") makes `load()`
return None and `loop.py` takes exactly the served path: no draw, no stream
family touched, bit-identical.

One realisation per simulated game, drawn from its own stream family
`shared_shooting` keyed on (seed, game_id) (the round-8 whistle pattern), and
added to the make logit of BOTH teams' field-goal attempts:

* ``G1``  one effect u ~ N(0, s^2) on every shot type.
* ``GP``  G1 coupled to the clock's own per-game pace latent:
          u = s (rho z_pace + sqrt(1 - rho^2) z), z_pace read (not advanced) from
          the clock stream key at `clock_adapter_v3.LATENT_ORDINAL`.
* ``G3``  per-type effects u ~ N(0, Sigma), Sigma 3x3 (rim, jump2, three).
* ``U1``  CONTROL: an independent effect per TEAM with G1's s^2 (same marginal
          variance, no sharing).

Every variance parameter is READ from `params_v1.json`, the fold-2 TRAIN fit of
`scripts/exp_shared_shooting_v1.py bakeoff`; nothing here is fitted, and
nothing is fitted to sim output. The effect is a random effect of the fg_make
sub-model, not a multiplier on engine output.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

PARAMS = Path("data/processed/models/shared_shooting/params_v1.json")
FAMILY = "shared_shooting"
ARMS = ("G1", "GP", "G3", "U1")
ENV = "ENGINE_SHARED_SHOOTING"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


class SharedShooting:
    def __init__(self, arm: str):
        if arm not in ARMS:
            raise KeyError(f"unknown {ENV}={arm!r}; known: {ARMS} or 'reference'")
        self.arm = arm
        self.par = json.loads(PARAMS.read_text(encoding="utf-8"))
        self.u = None            # (n, 2, 3): side x type logit shift

    def init_game(self, seeds, game_ids, clock_keys=None) -> None:
        """Draw every simulation's latent once, before the first possession."""
        from scipy.special import ndtri

        from cbb_sim.engine.rng import StreamBook, uniforms_at
        n = len(seeds)
        uu = StreamBook(seeds, game_ids, families=(FAMILY,)).draw_block(FAMILY, np.arange(n), 3)
        z = ndtri(np.clip(uu, 1e-12, 1 - 1e-12))            # (n, 3)
        u = np.zeros((n, 2, 3))
        if self.arm == "G1":
            s = np.sqrt(self.par["G1_s2"])
            u[:] = (s * z[:, 0])[:, None, None]
        elif self.arm == "U1":
            s = np.sqrt(self.par["U1_s2"])
            u[:, 0, :] = (s * z[:, 0])[:, None]
            u[:, 1, :] = (s * z[:, 1])[:, None]
        elif self.arm == "GP":
            from cbb_sim.engine.clock_adapter_v3 import LATENT_ORDINAL
            if clock_keys is None:
                raise ValueError("GP needs the clock stream keys")
            up = uniforms_at(np.asarray(clock_keys, dtype=np.uint64),
                             np.full(n, LATENT_ORDINAL, dtype=np.int64))
            zp = ndtri(up)
            s = np.sqrt(self.par["GP_s2"])
            rho = float(self.par["GP_rho"])
            u[:] = (s * (rho * zp + np.sqrt(1.0 - rho * rho) * z[:, 0]))[:, None, None]
        elif self.arm == "G3":
            S = np.asarray(self.par["G3_Sigma"], dtype=np.float64)
            v, Q = np.linalg.eigh(0.5 * (S + S.T))
            L = Q * np.sqrt(np.clip(v, 0.0, None))              # S = L L^T
            g = z @ L.T                                         # (n, 3)
            u[:] = g[:, None, :]
        self.u = u

    def shift(self, p: np.ndarray, rows: np.ndarray, side: np.ndarray, type_idx: int) -> np.ndarray:
        """`p` with the game's logit effect for (side, type) added."""
        x = _logit(np.asarray(p, dtype=np.float64)) + self.u[rows, side, type_idx]
        return 1.0 / (1.0 + np.exp(-x))


#: SERVED DEFAULT, ADOPTED 2026-10-01 (Decision 11 set, PM under user delegation;
#: docs/tests/adoption_served_v2_2026-10-01.md). `reference` reproduces served-v1.
DEFAULT = "G3"


def load():
    """None for 'reference' (served-v1), else the arm; unset serves DEFAULT."""
    arm = os.environ.get(ENV, DEFAULT)
    if not arm or arm == "reference":
        return None
    return SharedShooting(arm)
