"""team_rate_draw.py -- Stage C team-rate draw arms (S1 / S2 / S3), DEFAULT OFF.

Pre-registration: docs/models/team_rate_estimator/experiments.md sections 7, 7a and 7c.

ENGINE_TEAM_RATE_DRAW names a draw file written by `scripts/build_engine_inputs_trdraw_v1.py`. If it is
unset or "off", `active()` returns None and the engine path is byte-for-byte the served one: the hooks in
loop.py and adapters.py take their `is None` branch, which is the original expression.

A draw file holds K perturbed input sets for the SAME games, in the same order, as the engine inputs it
was built from:
    team_static_k  (K, G, 2, Ft)  float32  replaces inp.team_static at the team-rate gather sites
    team_block_k   (K, G, 2, 16)  float32  replaces EventAdapter.team_block (round-2 event block)
    game_ids       (G,)                    asserted equal to inp.games.game_id

The draw index is picked ONCE per simulated game:
    k = floor(K * u), u = element 0 of the (seed, game_id, "team_rate") stream (rng.uniforms_at).
It is a separate family, so no other family's stream moves, and paired arms (same seeds, same games)
pick the same k. With K = 1 there is NO RNG call and k = 0, so a K = 1 file whose single set equals the
inputs reproduces the no-draw path bit for bit.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cbb_sim.engine.rng import stream_keys, uniforms_at

ENV = "ENGINE_TEAM_RATE_DRAW"
FAMILY = "team_rate"
_CACHE: dict[str, "TeamRateDraw"] = {}


@dataclass
class TeamRateDraw:
    K: int
    team_static_k: np.ndarray
    team_block_k: np.ndarray
    game_ids: np.ndarray
    meta: dict = field(default_factory=dict)
    path: str = ""

    def k_index(self, seeds: np.ndarray, game_ids: np.ndarray) -> np.ndarray:
        """Draw index per simulation. K == 1: zeros, and no RNG call at all."""
        seeds = np.asarray(seeds, dtype=np.int64)
        if self.K == 1:
            return np.zeros(len(seeds), dtype=np.int64)
        keys = np.empty(len(seeds), dtype=np.uint64)
        gids = np.asarray(game_ids, dtype=np.int64)
        for s in np.unique(seeds):
            m = seeds == s
            keys[m] = stream_keys(int(s), gids[m], FAMILY)
        u = uniforms_at(keys, 0)
        return np.minimum((u * self.K).astype(np.int64), self.K - 1)


    def k_from_book(self, book) -> np.ndarray:
        """Same value as `k_index`, drawn through the chunk's StreamBook (first draw of the family)."""
        if self.K == 1:
            return np.zeros(book.n, dtype=np.int64)
        u = book.draw(FAMILY)
        return np.minimum((u * self.K).astype(np.int64), self.K - 1)


def load(path: str | os.PathLike) -> TeamRateDraw:
    p = str(Path(path))
    if p not in _CACHE:
        z = np.load(p)
        meta_p = Path(p).with_suffix(".json")
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        ts, tb = z["team_static_k"], z["team_block_k"]
        assert ts.shape[0] == tb.shape[0], "K mismatch between team_static_k and team_block_k"
        _CACHE[p] = TeamRateDraw(K=int(ts.shape[0]), team_static_k=ts, team_block_k=tb,
                                 game_ids=z["game_ids"], meta=meta, path=p)
    return _CACHE[p]


def active(inp) -> TeamRateDraw | None:
    """The draw for this run, or None (the served path). Asserts alignment with `inp`."""
    val = os.environ.get(ENV, "off")
    if val in ("", "off"):
        return None
    d = load(val)
    gids = inp.games["game_id"].to_numpy()
    if len(d.game_ids) != len(gids) or not np.array_equal(d.game_ids.astype(np.int64), gids.astype(np.int64)):
        raise ValueError(f"{ENV}={val}: draw file game order does not match the engine inputs")
    if d.team_static_k.shape[1:] != inp.team_static.shape:
        raise ValueError(f"{ENV}: team_static_k {d.team_static_k.shape[1:]} != inputs {inp.team_static.shape}")
    return d
