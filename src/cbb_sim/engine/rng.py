"""
rng.py -- the engine's (seed, game_id, family, ordinal) stream discipline.

CLAUDE.md modelling rule: "RNG seeded on (seed, game_id, family). Paired
bake-off arms share aligned streams."

This module adds NOTHING to `cbb_sim.control.rng`'s construction; it reuses the
same splitmix64 mixing and the same `stream_keys`, and only generalises
`uniforms(keys, index)` -- whose `index` is a SCALAR shared by every row -- to
`uniforms_at(keys, index_per_row)`, because the possession engine's rows are
at different points of their own game's stream.

WHY THAT GENERALISATION IS THE WHOLE POINT (L19). The usage bake-off's
shared-uniform bug was exactly this: a scalar index means every event of a
game-sim draws the SAME uniform, and the whole game's allocation collapses onto
one player (top-1 usage share 60.9% vs 32.3% real). `usage.event_stream_keys`
fixed it on the offline path by folding the event ordinal into the KEY;
`UsageState.counter` fixes it on the sampler path by advancing an index. The
engine does the second thing, vectorised: every family carries a per-simulation
ordinal counter that advances once per draw, so two consecutive draws in one
game-sim can never be the same uniform. `tests/test_engine.py` pins it.

Equivalence to control.rng is a test, not a claim: `uniforms_at(keys, j)` with a
constant `j` reproduces `control.rng.uniforms(keys, j)` bit for bit.
"""

from __future__ import annotations

import numpy as np

from cbb_sim.control.rng import _mix64, family_hash, stream_keys  # noqa: F401  (re-export)

_PHI = np.uint64(0x9E3779B97F4A7C15)
_S11 = np.uint64(11)
_TWO53 = np.float64(1.0 / (1 << 53))

#: Every random draw the engine makes belongs to one of these families. A
#: family is a separate counter-based stream off the same (seed, game_id) key,
#: so adding a draw to one family can never shift another family's draws --
#: which is what lets a paired arm be differenced game by game.
FAMILIES: tuple[str, ...] = (
    "clock",          # possession duration
    "event",          # terminal-event class (possession_outcome)
    "usage",          # which of the five on the floor
    "fg_make",        # shot make/miss
    "free_throw",     # free-throw make/miss
    "rebound",        # OREB / DREB / dead ball
    "and_one",        # and-one on a made field goal
    "foul_accrual",   # non-shooting foul that produces no attempt
    "rotation",       # substitution targets and availability
    "rotation_foul",  # personal-foul hazard on the five on the floor
    "tipoff",         # opening possession, and each overtime's
)


def uniforms_at(keys: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Draw element `index[i]` of stream `keys[i]`, as a float64 on (0, 1).

    Identical construction to `control.rng.uniforms`, with `index` per row
    instead of shared. `index` may also be a python int, in which case this is
    exactly `control.rng.uniforms`."""
    idx = np.asarray(index, dtype=np.uint64)
    with np.errstate(over="ignore"):
        off = idx * _PHI
        z = _mix64(np.asarray(keys, dtype=np.uint64) + off)
    u = (z >> _S11).astype(np.float64) * _TWO53
    return np.clip(u, 1e-12, 1.0 - 1e-12)


class StreamBook:
    """One counter-based stream per (family, simulation).

    `keys[family]` is the (seed, game_id, family) key of every simulation in
    the batch; `counter[family]` is how far along that stream each simulation
    is. `draw(family, mask)` advances ONLY the simulations it draws for, so a
    simulation that skipped a possession (its game already over) does not
    consume a draw and the remaining simulations are unaffected -- the stream
    is a property of the game-sim, never of the batch it happened to be in.
    """

    __slots__ = ("keys", "counter", "n")

    def __init__(self, seeds: np.ndarray, game_ids: np.ndarray,
                 families: tuple[str, ...] = FAMILIES):
        seeds = np.asarray(seeds, dtype=np.int64)
        game_ids = np.asarray(game_ids, dtype=np.int64)
        if len(seeds) != len(game_ids):
            raise ValueError("seeds and game_ids must be the same length")
        self.n = len(seeds)
        self.keys: dict[str, np.ndarray] = {}
        self.counter: dict[str, np.ndarray] = {}
        uniq = np.unique(seeds)
        for fam in families:
            k = np.empty(self.n, dtype=np.uint64)
            for s in uniq:
                m = seeds == s
                k[m] = stream_keys(int(s), game_ids[m], fam)
            self.keys[fam] = k
            self.counter[fam] = np.zeros(self.n, dtype=np.int64)

    def draw(self, family: str, rows: np.ndarray | None = None) -> np.ndarray:
        """One fresh uniform per selected row; advances those rows' counters."""
        if rows is None:
            u = uniforms_at(self.keys[family], self.counter[family])
            self.counter[family] += 1
            return u
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        if len(rows) == 0:
            return np.empty(0, dtype=np.float64)
        u = uniforms_at(self.keys[family][rows], self.counter[family][rows])
        self.counter[family][rows] += 1
        return u

    def draw_block(self, family: str, rows: np.ndarray, width: int) -> np.ndarray:
        """`width` fresh uniforms per selected row, shape (len(rows), width)."""
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        if len(rows) == 0:
            return np.empty((0, width), dtype=np.float64)
        base = self.counter[family][rows]
        out = np.empty((len(rows), width), dtype=np.float64)
        k = self.keys[family][rows]
        for j in range(width):
            out[:, j] = uniforms_at(k, base + j)
        self.counter[family][rows] += width
        return out


def categorical(u: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Inverse-CDF pick from a row-stochastic matrix. Exactly the rule
    `usage.draw_player` uses: `(u > cumsum(p)).sum()`, clipped to the last
    column so a row whose probabilities sum marginally below 1 still resolves."""
    if len(u) == 0:
        return np.zeros(0, dtype=np.int64)
    c = np.cumsum(probs, axis=1)
    idx = (u[:, None] > c).sum(axis=1)
    return np.minimum(idx, probs.shape[1] - 1).astype(np.int64)
