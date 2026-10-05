"""apply_availability in scripts/build_engine_inputs_live.py: default-off player-out exclusion before share normalisation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_engine_inputs_live as BL  # noqa: E402
from cbb_sim.live import guards as G  # noqa: E402
from cbb_sim.models.rotation import TeamPrior  # noqa: E402

T0 = pd.Timestamp("2026-11-03T12:00:00Z")


def _prior(pids=(101, 102, 103, 104, 105, 106, 107, 108, -1, -2)):
    n = len(pids)
    sh = np.linspace(2.0, 0.5, n); sh = sh / sh.sum()
    return TeamPrior(game_id=1, team_id=10, pids=np.array(pids, dtype="int64"), share=sh, rank=np.arange(1, n + 1),
                     srank=np.arange(1, n + 1), start_order=np.arange(n), fpm=np.linspace(0.1, 0.05, n),
                     p_avail=np.linspace(1.0, 0.5, n), n_prior_games=5)


def _run(out_ids, extra=None, priors=None):
    priors = priors or {(1, 10): _prior(), (1, 11): _prior([201, 202, 203, 204, 205, 206, 207, 208])}
    df = pd.DataFrame({"cbbd_player_id": out_ids, **(extra or {})})
    diag = {}
    return priors, BL.apply_availability(priors, df, T0, diag), diag


def test_one_player_out():
    priors, new, diag = _run([103])
    a, b = priors[(1, 10)], new[(1, 10)]
    assert 103 not in b.pids and len(b.pids) == len(a.pids) - 1
    assert new[(1, 11)] is priors[(1, 11)]                      # untouched team-game is the same object
    assert (a.pids == _prior().pids).all()                       # input not mutated
    assert sorted(b.srank) == list(range(1, len(b.pids) + 1))
    assert list(b.pids[b.start_order[:5]]) == [101, 102, 104, 105, 106]   # next player moves into the starting five
    assert diag["availability"]["players_dropped"] == 1 and diag["availability"]["unknown_pids"] == []
    # surviving players keep their own as-of values (no hand redistribution)
    k = list(a.pids).index(104)
    assert b.share[list(b.pids).index(104)] == a.share[k] and b.fpm[list(b.pids).index(104)] == a.fpm[k]


def test_all_starters_out():
    _, new, diag = _run([101, 102, 103, 104, 105])
    b = new[(1, 10)]
    assert list(b.pids[b.start_order[:5]]) == [106, 107, 108, -1, -2]
    assert diag["availability"]["players_dropped"] == 5


def test_everyone_real_out_falls_back():
    priors, new, diag = _run([101, 102, 103, 104, 105, 106, 107, 108])
    assert (1, 10) not in new and (1, 11) in new                 # omitted -> league-mean fallback path in build_live
    assert diag["availability"]["team_games_fallback_after_drop"] == 1


def test_unknown_id_reported():
    _, new, diag = _run([103, 999999], extra={"athlete_id": [1, 2]})
    assert diag["availability"]["unknown_pids"] == [999999]
    _, _, d2 = _run([np.nan, 103])
    assert d2["availability"]["rows_without_cbbd_id"] == 1


def test_empty_and_leak_guard():
    priors, new, diag = _run([])
    assert all(new[k] is priors[k] for k in priors)
    with pytest.raises(G.LeakGuardError):
        _run([103], extra={"created_at": ["2026-11-03T13:00:00Z"]})
