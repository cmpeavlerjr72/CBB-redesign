"""Roster (ESPN) pull + injuries player-out parser: synthetic payloads, no network."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_player_crosswalk as BPC  # noqa: E402
import pull_injuries_player_out_v1 as PI  # noqa: E402
import pull_rosters_espn_v1 as PR  # noqa: E402


class _Resp:
    headers = {}
    def __init__(self, data): self._d = data
    def raise_for_status(self): pass
    def json(self): return self._d


def test_cbbd_empty_season_is_hard_error(tmp_path, monkeypatch):
    class S:
        def __init__(self): self.headers = {}
        def get(self, *a, **k): return _Resp([{"teamId": 1, "team": "X", "players": []}, {"teamId": 2, "team": "Y"}])
    monkeypatch.setattr(BPC.requests, "Session", S)
    monkeypatch.setattr(BPC, "load_api_key", lambda: "k")
    monkeypatch.setattr(BPC, "ROSTER_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="0 players"):
        BPC.pull_rosters([2027])
    assert not list(tmp_path.glob("*.parquet"))          # never an empty file


def test_espn_build_ids_and_transfers():
    xw = pd.DataFrame({"espn_team_id": [10, 20], "cbbd_team_id": [1, 2], "cbbd_name": ["A", "B"]})
    prior = pd.DataFrame({"season": [2026, 2026], "source_id": ["100", "200"], "cbbd_player_id": [7, 8], "team_source_id": ["10", "10"]})
    raw = {10: [{"id": "100", "displayName": "P1", "status": {"type": "active"}}, {"id": "300", "displayName": "Fr"}],
           20: [{"id": "200", "displayName": "T1"}, {"id": "400", "displayName": "Out", "status": {"type": "inactive"}}]}
    df = PR.build(2027, raw, xw, prior, {1: "C1", 2: "C2"}).set_index("espn_player_id")
    assert list(df.columns[:16]) == PR.SCHEMA
    assert df.loc[100, "cbbd_player_id"] == 7 and not df.loc[100, "is_transfer_in"]        # returner
    assert df.loc[200, "is_transfer_in"] and df.loc[200, "prev_team_source_id"] == 10      # moved 10 -> 20
    assert df.loc[300, "new_to_d1"] and pd.isna(df.loc[300, "cbbd_player_id"])
    assert 400 not in df.index and df["team_source_id"].map(type).eq(str).all()


def test_fetch_team_rejects_stale_season():
    class S:
        def get(self, *a, **k):
            r = _Resp({"season": {"year": 2026}, "athletes": [{"id": "1"}]}); r.status_code = 200; return r
    tid, ath, err = PR.fetch_team(5, S(), 2027)
    assert ath == [] and "season.year" in err


def test_roster_cadence(tmp_path):
    out = tmp_path / "roster_2027.parquet"
    assert PR.roster_due(date(2026, 10, 5), 2027, out)[0]
    out.write_text("x"); out.with_suffix(".report.json").write_text('{"pulled_at": "2026-10-05T10:00:00+00:00"}')
    assert not PR.roster_due(date(2026, 10, 10), 2027, out)[0]      # weekly before opening
    assert PR.roster_due(date(2026, 10, 12), 2027, out)[0]
    assert PR.roster_due(date(2026, 11, 3), 2027, out)[0]           # daily from 2026-11-02


PAYLOAD = {"injuries": [{"id": "153", "injuries": [
    {"status": "Out", "athlete": {"id": "5196877"}, "details": {"type": "Knee"}},
    {"status": "Questionable", "athlete": {"id": "999"}}]}]}


def test_injury_parse_out_rule_and_empty():
    df = PI.parse_league(PAYLOAD, "2026-11-02", "t")
    assert df["out"].tolist() == [True, False] and df.loc[0, "status"] == "out"
    assert PI.pull(date(2026, 11, 2), "t", payload={"injuries": []}).empty                     # empty payload = no reports
    with pytest.raises(RuntimeError, match="schema drift"):
        PI.pull(date(2026, 11, 2), "t", payload={"injuries": [{"id": "1", "injuries": [{"status": "Out"}]}]})


def test_player_out_merges_manual(tmp_path):
    ov = tmp_path / "av.csv"
    ov.write_text("athlete_id,date,status,note\n5,2026-11-02,out,manual\n6,2026-11-02,questionable,x\n7,2026-11-03,out,other day\n")
    auto = pd.DataFrame(PI.parse_league(PAYLOAD, "2026-11-02", "t")).reindex(columns=PI.COLS)
    auto.to_csv(tmp_path / "player_out_2026-11-02.csv", index=False)
    po = PI.player_out_for(date(2026, 11, 2), tmp_path, ov)
    assert sorted(po["athlete_id"]) == [5, 5196877]
