"""Daily sim cache key includes the actual inputs (ops 2026-10-09 PM ruling): identical inputs may reuse, any changed input re-runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import run_daily_sim_v1 as SIM  # noqa: E402
import build_engine_inputs_live as BL  # noqa: E402
from cbb_sim.live import daily as D  # noqa: E402
from cbb_sim.live import inputs_key as IK  # noqa: E402


class _Stop(Exception):
    pass


def _slate(tip="2026-11-02T23:00:00Z"):
    return pd.DataFrame({"game_id": [1], "cbbd_game_id": [11], "season": [2027], "game_date": ["2026-11-02"],
                         "tipoff_utc": [pd.Timestamp(tip)], "home_team_id": [2], "away_team_id": [3],
                         "neutral": [False], "tip_source": ["cbbd"], "tip_time_is_placeholder": [False]})


def _fake_repo(tmp_path, tip_row_shift=0, roster_extra=0):
    r = tmp_path / "repo"
    (r / "data/processed/ingest").mkdir(parents=True, exist_ok=True)
    (r / "data/raw/cbbd/rosters").mkdir(parents=True, exist_ok=True)
    (r / "data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": [1, 2], "tipoff_utc": pd.to_datetime(["2026-11-02T23:00:00Z", "2026-11-02T20:00:00Z"]) + pd.Timedelta(minutes=tip_row_shift),
                  "tip_source": ["cbbd", "cbbd"], "tip_fetched_at": [pd.Timestamp.now("UTC")] * 2}).to_parquet(r / "data/processed/ingest/tip_times_2027.parquet")
    pd.DataFrame({"cbbd_player_id": list(range(5 + roster_extra))}).to_parquet(r / "data/raw/cbbd/rosters/roster_2027.parquet")
    pd.DataFrame({"cbbd_player_id": [1]}).to_parquet(r / "data/raw/cbbd/rosters/roster_2026.parquet")
    pd.DataFrame({"game_id": [1]}).to_parquet(r / "data/processed/games_universe.parquet")
    return r


def _h(repo, slate=None, seeds=200, **kw):
    return IK.inputs_hash(2027, _slate() if slate is None else slate, seeds, 0, None, repo=repo, **kw)[0]


def test_identical_inputs_same_hash_even_when_pull_stamps_change(tmp_path):
    a = _h(_fake_repo(tmp_path / "a"))
    b = _h(_fake_repo(tmp_path / "b"))          # tip_fetched_at differs between the two builds
    assert a == b


def test_one_tip_row_change_changes_the_hash(tmp_path):
    assert _h(_fake_repo(tmp_path / "a")) != _h(_fake_repo(tmp_path / "b", tip_row_shift=30))


def test_roster_seed_count_slate_and_extra_file_change_the_hash(tmp_path):
    base = _h(_fake_repo(tmp_path / "a"))
    assert base != _h(_fake_repo(tmp_path / "b", roster_extra=1))
    assert base != _h(_fake_repo(tmp_path / "c"), seeds=4)
    assert base != _h(_fake_repo(tmp_path / "d"), slate=_slate("2026-11-02T22:00:00Z"))
    f = tmp_path / "inj.csv"
    f.write_text("a\n1\n")
    h1 = _h(_fake_repo(tmp_path / "e"), extra_files={"injuries": f})
    f.write_text("a\n2\n")
    assert h1 != _h(_fake_repo(tmp_path / "f"), extra_files={"injuries": f})


def test_stack_version_changes_with_engine_env(monkeypatch):
    a = IK.stack_version()
    monkeypatch.setenv("ENGINE_SHOT_BLOCK", "reference")
    assert IK.stack_version() != a


def _run(monkeypatch, tmp_path, ihash, **kw):
    monkeypatch.setattr(SIM, "load_slate", lambda *a, **k: _slate())
    monkeypatch.setattr(SIM, "season_start_of", lambda *a, **k: "2026-11-02")
    monkeypatch.setattr(SIM, "day1_prior_seed", lambda s: (None, None))
    monkeypatch.setattr(IK, "inputs_hash", lambda *a, **k: (ihash, {"x": ihash}))

    def fake_build_live(*a, **k):
        raise _Stop()
    monkeypatch.setattr(BL, "build_live", fake_build_live)
    return SIM.run_sim_stage("2026-11-02", 2027, seeds=1, now="2026-10-09T16:00:00Z", root=tmp_path, schedule_source="cbbd",
                             pass_name="evening", **kw)


def _mark_done(tmp_path, ihash):
    out = D.sim_dir(tmp_path, "2026-11-02", D.default_run_id(1, 0))
    out.mkdir(parents=True)
    h = SIM.config_hash({"slate_date": "2026-11-02", "season": 2027, "fold": "F2", "seeds": 1, "seed_offset": 0, "schedule_source": "cbbd",
                         "schedule_path": None, "tips": None, "players": False, "max_games": 0, "replay": False, "pass": "evening",
                         "day1_prior": "A3+R1"})
    (out / "_DONE.json").write_text(json.dumps({"config_hash": h, "inputs_hash": ihash, "inputs_components": {"x": ihash}, "n_rows": 5, "n_games": 1}))
    (out / "games.parquet").write_bytes(b"x")
    return out


def test_stage_reuses_only_on_identical_inputs(monkeypatch, tmp_path):
    _mark_done(tmp_path, "AAA")
    r = _run(monkeypatch, tmp_path, "AAA")
    assert r["cached"] is True and r["inputs_hash"] == "AAA"


def test_stage_reruns_when_inputs_changed_and_clears_the_stale_done(monkeypatch, tmp_path):
    out = _mark_done(tmp_path, "AAA")
    with pytest.raises(_Stop):                      # reached build_live = re-ran
        _run(monkeypatch, tmp_path, "BBB")
    assert not (out / "_DONE.json").exists() and not (out / "games.parquet").exists()


def test_stage_reruns_when_old_done_has_no_inputs_key(monkeypatch, tmp_path):
    out = _mark_done(tmp_path, None)
    with pytest.raises(_Stop):
        _run(monkeypatch, tmp_path, "AAA")
