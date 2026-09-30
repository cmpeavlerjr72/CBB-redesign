"""Unit tests for cbb_sim.engine.team_rate_draw (Stage C draw; default off)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cbb_sim.engine import team_rate_draw as TRD
from cbb_sim.engine.rng import FAMILIES, StreamBook


class _Inp:
    def __init__(self, gids, F=5):
        self.games = pd.DataFrame({"game_id": gids})
        self.team_static = np.zeros((len(gids), 2, F), dtype=np.float32)


def _draw(K, G=3, F=5):
    return TRD.TeamRateDraw(K=K, team_static_k=np.zeros((K, G, 2, F), np.float32),
                            team_block_k=np.zeros((K, G, 2, 16), np.float32), game_ids=np.arange(G) + 100)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(TRD.ENV, raising=False)
    assert TRD.active(_Inp(np.arange(3) + 100)) is None
    monkeypatch.setenv(TRD.ENV, "off")
    assert TRD.active(_Inp(np.arange(3) + 100)) is None


def test_K1_makes_no_rng_call():
    d = _draw(1)
    book = StreamBook(np.arange(6), np.full(6, 101))
    k = d.k_from_book(book)
    assert (k == 0).all() and int(book.counter["team_rate"].max()) == 0


def test_K_gt_1_uses_only_team_rate_family_and_matches_k_index():
    d = _draw(64)
    seeds = np.tile(np.arange(20), 3); gids = np.repeat(np.array([100, 101, 102]), 20)
    b0, b1 = StreamBook(seeds, gids), StreamBook(seeds, gids)
    k = d.k_from_book(b1)
    assert np.array_equal(k, d.k_index(seeds, gids))
    assert ((k >= 0) & (k < 64)).all() and len(np.unique(k)) > 10
    for f in FAMILIES:
        if f != "team_rate":
            assert np.array_equal(b0.counter[f], b1.counter[f])
    assert int(b1.counter["team_rate"].max()) == 1


def test_alignment_is_asserted(tmp_path, monkeypatch):
    p = tmp_path / "d.npz"
    np.savez(p, team_static_k=np.zeros((2, 3, 2, 5), np.float32), team_block_k=np.zeros((2, 3, 2, 16), np.float32),
             game_ids=np.array([100, 101, 102]))
    monkeypatch.setenv(TRD.ENV, str(p))
    assert TRD.active(_Inp(np.array([100, 101, 102]))).K == 2
    with pytest.raises(ValueError):
        TRD.active(_Inp(np.array([100, 102, 101])))
