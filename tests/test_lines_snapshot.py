import sys
from datetime import date
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pull_espn_odds_snapshot_v1 as S  # noqa: E402
import build_lines_open_close_v1 as B  # noqa: E402

SB = {"events": [{"id": "100", "competitions": [{"odds": [
    {"provider": {"name": "Draft Kings"}, "spread": -4.5, "overUnder": 140.5,
     "homeTeamOdds": {"favorite": False, "moneyLine": 150}, "awayTeamOdds": {"favorite": True, "moneyLine": -180}}]}]},
    {"id": "101", "competitions": [{"odds": []}]}]}
CB = [{"gameId": 7, "lines": [{"provider": "Bovada", "spread": -3, "overUnder": 150, "spreadOpen": -2.5, "overUnderOpen": 149}]}]


def fx():
    return {"espn": lambda d: SB, "cbbd": lambda d: (CB, {7: 100})}


def test_parse_home_spread_and_ids():
    t = pd.Timestamp("2026-11-02T12:00Z")
    r = S.parse_espn_scoreboard(SB, t)
    assert len(r) == 1 and r[0]["spread_home"] == 4.5 and r[0]["away_ml"] == -180
    c = S.parse_cbbd_lines(CB, {7: 100}, t)
    assert c[0]["game_id"] == 100 and c[0]["spread_open"] == -2.5


def test_idempotent_and_append_only(tmp_path):
    d = date(2026, 11, 2)
    assert S.run(d, root=tmp_path, now="2026-11-02T12:00:00Z", fetchers=fx())["rows_added"] == 2
    assert S.run(d, root=tmp_path, now="2026-11-02T12:00:00Z", fetchers=fx())["rows_added"] == 0
    assert S.run(d, root=tmp_path, now="2026-11-02T13:00:00Z", fetchers=fx())["rows_added"] == 2
    df = pd.read_parquet(S.out_path(d, tmp_path))
    assert len(df) == 4 and df.captured_at.nunique() == 2


def test_no_odds_writes_nothing(tmp_path):
    d = date(2026, 11, 2)
    r = S.run(d, root=tmp_path, now="2026-11-02T12:00:00Z",
              fetchers={"espn": lambda x: {"events": []}, "cbbd": lambda x: ([], {})})
    assert r["rows_added"] == 0 and not S.out_path(d, tmp_path).exists()


def test_open_close_excludes_at_or_after_tip():
    tip = pd.Timestamp("2026-11-02T20:00Z")
    tips = pd.DataFrame({"game_id": [100], "tipoff_utc": [tip], "tip_time_is_placeholder": [False]})
    caps = ["2026-11-02T12:00Z", "2026-11-02T18:00Z", "2026-11-02T20:00Z", "2026-11-02T21:00Z"]
    sn = pd.DataFrame({"game_id": [100] * 4, "provider": "DK", "source": "espn", "spread_home": [-1.0, -2.0, -9.0, -9.5],
                       "total": 140.0, "home_ml": None, "away_ml": None, "captured_at": pd.to_datetime(caps, utc=True)})
    r = B.build(sn, tips).iloc[0]
    assert r.n_captures == 2 and r.open_spread_home == -1.0 and r.close_spread_home == -2.0
    assert r.open_lead_min == 480 and r.close_lead_min == 120
