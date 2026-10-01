"""2027 hard stops (lane F, 2026-10-01): a season missing from the R9ao3 team-prior table or the shot-block design is an error that names the seal."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from cbb_sim.engine import foul_r9 as FR9  # noqa: E402


class _Inp:
    def __init__(self, season):
        self.games = pd.DataFrame({"season": [season], "home_team_id": [1], "away_team_id": [2]})


def test_r9ao3_missing_season_raises_and_names_seal(monkeypatch):
    monkeypatch.chdir(REPO)
    fr = FR9.load("R9ao3")
    with pytest.raises(RuntimeError, match="SEAL"):
        fr._team_term(_Inp(2027))


def test_r9ao3_present_season_does_not_raise(monkeypatch):
    monkeypatch.chdir(REPO)
    fr = FR9.load("R9ao3")
    assert fr._team_term(_Inp(2025)).shape == (1, 2)


def test_table_seasons_has_2025_not_2027(monkeypatch):
    monkeypatch.chdir(REPO)
    have = FR9.table_seasons("R9ao3")
    assert 2025 in have and 2027 not in have


def test_shot_block_live_season_guard_names_seal():
    import build_shot_block_lut_live_v1 as SBL
    with pytest.raises(RuntimeError, match="SEAL"):
        SBL.season_events(2027)


def test_chain_prereqs_block_2027_on_team_prior():
    import chain_daily_v3 as C
    m = C.sim_prereqs(2027, "F2", "2026-11-02", None)
    assert any("R9ao3 team-prior table" in x and "SEAL" in x for x in m)
    assert not any("R9ao3 team-prior" in x for x in C.sim_prereqs(2025, "F2", "2025-02-11", "x"))


def test_replay_requires_root():
    import chain_daily_v3 as C
    with pytest.raises(SystemExit, match="--root"):
        C.main(["--replay-season", "2025", "--slate-date", "2025-02-11"])
