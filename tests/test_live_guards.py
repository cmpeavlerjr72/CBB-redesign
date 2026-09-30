"""Guards of the live-slate path (docs/ops/live_slate_path_2026-09-30.md)."""
import pandas as pd
import pytest

from cbb_sim.live import features as LF
from cbb_sim.live import guards as G


def _slate(**over):
    d = pd.DataFrame({
        "game_id": [1, 2], "cbbd_game_id": [11, 12], "season": 2025,
        "game_date": pd.to_datetime(["2024-11-12", "2024-11-12"]),
        "tipoff_utc": pd.to_datetime(["2024-11-12T18:00:00Z", "2024-11-12T23:00:00Z"]),
        "home_team_id": [10, 12], "away_team_id": [11, 13], "neutral": [False, False]})
    for k, v in over.items():
        d[k] = v
    return d


def _universe():
    return pd.DataFrame({"game_id": [100], "season": [2025], "game_date": ["2024-11-11"],
                         "tipoff_utc": pd.to_datetime(["2024-11-11T20:00:00Z"]), "is_d1_game": [True],
                         "pbp_truncated": [False], "pbp_complete": [True], "cbbd_game_id": [5]})


def test_created_at_before_tipoff_passes_and_fails():
    s = _slate()
    G.assert_created_before_tipoff(s.assign(created_at=pd.Timestamp("2024-11-12T12:00:00Z")))
    with pytest.raises(G.LeakGuardError):
        G.assert_created_before_tipoff(s.assign(created_at=pd.Timestamp("2024-11-12T19:00:00Z")))
    with pytest.raises(G.LeakGuardError):
        G.assert_created_before_tipoff(s)                       # column missing


def test_cutoff_must_precede_first_tip():
    s = _slate()
    G.assert_cutoff_before_slate("2024-11-12T17:59:00Z", s)
    with pytest.raises(G.LeakGuardError):
        G.assert_cutoff_before_slate("2024-11-12T18:00:00Z", s)


def test_slate_with_results_is_rejected():
    with pytest.raises(G.LeakGuardError):
        LF.build_ctx(_slate(home_score=[70, 60]), "2024-11-12T12:00:00Z", 2025, _universe())


def test_slate_must_be_one_date():
    s = _slate()
    s.loc[1, "game_date"] = pd.Timestamp("2024-11-13")
    with pytest.raises(ValueError):
        LF.build_ctx(s, "2024-11-12T12:00:00Z", 2025, _universe())


def test_universe_prior_is_strictly_before_the_slate_date():
    u = _universe()
    u2 = pd.concat([u, u.assign(game_id=101, game_date="2024-11-12",
                                tipoff_utc=pd.Timestamp("2024-11-12T01:00:00Z"))], ignore_index=True)
    ctx = LF.build_ctx(_slate(), "2024-11-12T12:00:00Z", 2025, u2)
    assert set(ctx.universe_prior["game_id"]) == {100}


def test_unfinished_source_game_is_rejected():
    u = _universe()
    u["tipoff_utc"] = pd.to_datetime(["2024-11-12T10:00:00Z"])
    u["game_date"] = "2024-11-11"                                # dated before, tipped inside the window
    with pytest.raises(G.LeakGuardError):
        LF.build_ctx(_slate(), "2024-11-12T12:00:00Z", 2025, u)
