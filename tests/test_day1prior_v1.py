"""Day-1 player priors (scripts/build_engine_inputs_day1prior_v1.py): hard stops and the 'only all-anonymous team-games' rule."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import build_engine_inputs_day1prior_v1 as D  # noqa: E402


def test_2027_hard_stops_on_sealed_prior_season(monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.delenv("CBB_UNSEAL", raising=False)
    with pytest.raises(Exception):
        D.tables(2027, True)


def test_missing_roster_is_source_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO)
    D._C.clear()
    monkeypatch.setattr(D, "prev_minutes", lambda s, src: pd.DataFrame({"team_id": [1], "pid": [10], "minutes": [30.0]}))
    with pytest.raises(D.SourceMissing):
        D.tables(2025, True, str(tmp_path / "roster_2025.parquet"))
    empty = tmp_path / "empty.parquet"
    pd.DataFrame({"team_source_id": pd.Series(dtype="int64"), "cbbd_player_id": pd.Series(dtype="int64")}).to_parquet(empty)
    with pytest.raises(D.SourceMissing, match="empty"):
        D.tables(2025, True, str(empty))


def test_seed_only_touches_all_anonymous_rows(monkeypatch):
    D._C.clear()
    T = {"by_team": {1: [(10, 300.0), (11, 200.0)], 2: [(20, 300.0)]}, "team_min": {1: 1000.0, 2: 1000.0},
         "any_min": {10: 300.0, 11: 200.0, 20: 300.0}, "roster": {1: {10}, 2: {20}}}
    monkeypatch.setattr(D, "tables", lambda *a, **k: T)
    S = 15
    games = pd.DataFrame({"game_id": [100]})
    tg = pd.DataFrame({"game_id": [100, 100], "team_id": [1, 2], "is_home": [True, False]})
    rc = np.zeros((1, 2, S), dtype=np.int64)
    rc[0, 0] = -np.arange(1, S + 1)                     # home: anonymous (day 1)
    rc[0, 1, :3] = [55, 56, 57]; rc[0, 1, 3:] = -np.arange(4, S + 1)   # away: has an in-season prior
    fn = D.make_seed_fn("A1")
    cand = fn(SimpleNamespace(season=2025), games, tg, rc, np.ones_like(rc, bool), S, {})
    assert cand[(100, 1)] == [10]                       # returner on the roster only (11 departed)
    assert cand[(100, 2)] == [55, 56, 57]               # untouched
    fn2 = D.make_seed_fn("A1n")
    rc[0, 0] = -np.arange(1, S + 1)
    assert fn2(SimpleNamespace(season=2025), games, tg, rc, np.ones_like(rc, bool), S, {})[(100, 1)] == [10, 11]
