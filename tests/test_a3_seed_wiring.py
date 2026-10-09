"""A3+R1 day-1 player prior wired into the served daily sim / inputs stages (ops 2026-10-09, docs/ops/a3_seed_wiring_2026-10-09.md)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import run_daily_sim_v1 as SIM  # noqa: E402
import chain_daily_v2 as V2  # noqa: E402
import build_engine_inputs_live as BL  # noqa: E402


class _Stop(Exception):
    pass


def _slate():
    return pd.DataFrame({"game_id": [1], "cbbd_game_id": [11], "season": [2027], "game_date": ["2026-11-02"],
                         "tipoff_utc": [pd.Timestamp("2026-11-02T23:00:00Z")], "home_team_id": [2], "away_team_id": [3],
                         "neutral": [False], "tip_source": ["cbbd"], "tip_time_is_placeholder": [False]})


def test_applies_only_to_live_serving_seasons():
    assert SIM.day1_prior_applies(2027, replay=False)
    assert not SIM.day1_prior_applies(2027, replay=True)
    assert not SIM.day1_prior_applies(2025, replay=False)
    assert not SIM.day1_prior_applies(2027, replay=False, day1_prior=None)


@pytest.fixture(autouse=True)
def _no_live_injury_files(monkeypatch):
    """These tests are about the seed wiring; the injury feed has its own tests (test_injury_feed.py) and must not read today's real files."""
    monkeypatch.setattr(SIM, "injuries_for", lambda now, season, replay: (None, frozenset(), {}, "none"))


def _capture(monkeypatch, tmp_path, season, replay, **kw):
    seen = {}
    monkeypatch.setattr(SIM, "load_slate", lambda *a, **k: _slate().assign(season=season))
    monkeypatch.setattr(SIM, "season_start_of", lambda *a, **k: "2026-11-02")
    sentinel = object()
    monkeypatch.setattr(SIM, "day1_prior_seed", lambda s: (SimpleNamespace(post=lambda i, f: None), sentinel))

    def fake_build_live(*a, **k):
        seen.update(k)
        raise _Stop()
    monkeypatch.setattr(BL, "build_live", fake_build_live)
    with pytest.raises(_Stop):
        SIM.run_sim_stage("2026-11-02", season, seeds=1, now="2026-10-09T16:00:00Z", root=tmp_path, schedule_source="cbbd",
                          replay=replay, **kw)
    return seen, sentinel


def test_daily_sim_passes_the_a3_seed_fn_for_2027(monkeypatch, tmp_path):
    seen, sentinel = _capture(monkeypatch, tmp_path, 2027, False)
    assert seen["seed_fn"] is sentinel


def test_daily_sim_replay_and_opt_out_pass_no_seed_fn(monkeypatch, tmp_path):
    seen, _ = _capture(monkeypatch, tmp_path, 2025, True)
    assert seen["seed_fn"] is None
    seen, _ = _capture(monkeypatch, tmp_path / "b", 2027, False, day1_prior=None)
    assert seen["seed_fn"] is None


def test_day1_prior_seed_hard_stops_on_missing_sources(monkeypatch):
    monkeypatch.setattr(V2, "day1_player_prior_missing", lambda season, repo=V2.REPO: ["A3 roster table missing"])
    with pytest.raises(RuntimeError, match="A3 roster table missing"):
        SIM.day1_prior_seed(2027)


def test_day1_prior_seed_uses_the_chain_definition(monkeypatch):
    monkeypatch.setattr(V2, "day1_player_prior_missing", lambda season, repo=V2.REPO: [])
    monkeypatch.setattr(V2, "day1_player_prior_seed", lambda season, repo=V2.REPO: ("mod", ("fn", season)))
    assert SIM.day1_prior_seed(2027) == ("mod", ("fn", 2027))


def test_inputs_census_stage_passes_the_seed(monkeypatch):
    import chain_day1_2027_v1 as DAY1
    seen = {}
    sentinel = object()
    monkeypatch.setattr(V2, "day1_player_prior_missing", lambda season, repo=V2.REPO: [])
    monkeypatch.setattr(SIM, "day1_prior_seed", lambda s: (SimpleNamespace(post=lambda i, f: None), sentinel))
    monkeypatch.setattr(SIM, "load_slate", lambda *a, **k: _slate())
    monkeypatch.setattr(SIM, "season_start_of", lambda *a, **k: "2026-11-02")

    def fake_build_live(*a, **k):
        seen.update(k)
        raise _Stop()
    monkeypatch.setattr(BL, "build_live", fake_build_live)
    ctx = SimpleNamespace(slate_date="2026-11-02", now=pd.Timestamp("2026-10-09T16:00:00Z"), state={"ratings_dir": "x"})
    with pytest.raises(_Stop):
        DAY1.stage_inputs(ctx, None, SimpleNamespace(fold="F2"), SIM, season=2027)
    assert seen["seed_fn"] is sentinel


def test_espn_map_fallback_only_for_uncrosswalked_season(tmp_path, monkeypatch):
    cw = pd.DataFrame({"season": [2026, 2026], "cbbd_player_id": [10, 11], "espn_athlete_id": [100.0, 110.0]})
    ros = tmp_path / "data/raw/cbbd/rosters"
    ros.mkdir(parents=True)
    pd.DataFrame({"cbbd_player_id": [11, 12], "espn_player_id": [111, 120]}).to_parquet(ros / "roster_2027.parquet")
    monkeypatch.setattr(BL, "ROOT", tmp_path)
    mp, src = BL.espn_map_uncrosswalked(cw, 2027)
    assert mp == {10: 100, 11: 111, 12: 120}          # season-S roster wins over the prior crosswalk season
    assert "crosswalk season 2026" in src and "roster_2027" in src
