"""Tests for scripts/pull_daily_ingest_v1.py: finals verification classes and idempotent upsert (no network)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pull_daily_ingest_v1 as PI  # noqa: E402


def _hs(rows):
    base = dict(status_type_completed=True, status_type_name="STATUS_FINAL", home_conference_id=1.0, away_conference_id=2.0,
                game_date="2025-01-15", game_date_time=pd.Timestamp("2025-01-15 19:00", tz="America/New_York"),
                home_id=10, away_id=11)
    return pd.DataFrame([{**base, **r} for r in rows])


def _cg(rows):
    base = dict(status="final", startDate="2025-01-16T00:00:00.000Z")
    return pd.DataFrame([{**base, **r} for r in rows])


def test_verify_finals_classes():
    hs = _hs([
        dict(game_id=1, home_score=70, away_score=60),                                   # agree
        dict(game_id=2, home_score=70, away_score=60),                                   # flipped in cbbd
        dict(game_id=3, home_score=70, away_score=60),                                   # score disagree
        dict(game_id=4, home_score=0, away_score=0, status_type_completed=False, status_type_name="STATUS_POSTPONED"),
        dict(game_id=5, home_score=70, away_score=60),                                   # no cbbd row
        dict(game_id=6, home_score=70, away_score=60),                                   # cbbd not final
        dict(game_id=7, home_score=70, away_score=60, away_conference_id=np.nan),        # non D-I
        dict(game_id=8, home_score=0, away_score=0, status_type_completed=False, status_type_name="STATUS_SCHEDULED"),
    ])
    cg = _cg([
        dict(id=101, sourceId="1", homePoints=70, awayPoints=60),
        dict(id=102, sourceId="2", homePoints=60, awayPoints=70),
        dict(id=103, sourceId="3", homePoints=72, awayPoints=60),
        dict(id=104, sourceId="4", status="cancelled", homePoints=np.nan, awayPoints=np.nan),
        dict(id=106, sourceId="6", status="in_progress", homePoints=40, awayPoints=30),
        dict(id=108, sourceId="8", status="scheduled", homePoints=np.nan, awayPoints=np.nan),
    ])
    v = PI.verify_finals(hs, cg).set_index("game_id")
    assert v.loc[1, "state"] == "final_verified"
    assert (v.loc[2, "state"], v.loc[2, "reason"]) == ("pending", "sides_flipped")
    assert (v.loc[3, "state"], v.loc[3, "reason"]) == ("pending", "score_disagree")
    assert (v.loc[4, "state"], v.loc[4, "reason"]) == ("pending", "not_final_either_source")
    assert (v.loc[5, "state"], v.loc[5, "reason"]) == ("pending", "awaiting_cbbd_game_row")
    assert (v.loc[6, "state"], v.loc[6, "reason"]) == ("pending", "awaiting_cbbd_final")
    assert v.loc[7, "state"] == "out_of_scope_non_d1"
    assert (v.loc[8, "state"], v.loc[8, "reason"]) == ("pending", "not_final_either_source")


def test_hoopr_final_cbbd_not_yet():
    hs = _hs([dict(game_id=9, home_score=50, away_score=40, status_type_completed=False, status_type_name="STATUS_IN_PROGRESS")])
    cg = _cg([dict(id=109, sourceId="9", homePoints=50, awayPoints=40)])
    v = PI.verify_finals(hs, cg).iloc[0]
    assert v["state"] == "pending" and v["reason"] == "not_final_either_source" or v["reason"] == "awaiting_hoopr_final"


def test_upsert_is_idempotent_and_stamps_ingested_at(tmp_path):
    p = tmp_path / "t.parquet"
    ts1, ts2 = pd.Timestamp("2026-11-03T10:00:00Z"), pd.Timestamp("2026-11-03T11:00:00Z")
    new = pd.DataFrame({"game_id": [1, 1, 2], "v": [1.0, np.nan, 3.0], "arr": [np.array([1, 2]), np.array([3]), None]})
    led: dict = {}
    PI.upsert(p, new, "game_id", ts1, led, "t")
    assert led["t"]["rows_added"] == 3
    a = pd.read_parquet(p)
    assert a[PI.INGEST_COL].notna().all()
    led2: dict = {}
    PI.upsert(p, new.copy(), "game_id", ts2, led2, "t")          # same content again: nothing written
    assert led2["t"]["rows_added"] == 0
    b = pd.read_parquet(p)
    assert b.equals(a)
    changed = new.copy(); changed.loc[2, "v"] = 9.0            # changed game 2: replaced, game 1 untouched
    led3: dict = {}
    PI.upsert(p, changed, "game_id", ts2, led3, "t")
    c = pd.read_parquet(p)
    assert led3["t"]["rows_added"] == 1 and led3["t"]["rows_replaced"] == 1 and len(c) == 3
    assert c.loc[c["game_id"] == 1, PI.INGEST_COL].iloc[0] == ts1


def test_season_of():
    from datetime import date
    assert PI.season_of(date(2026, 11, 2)) == 2027 and PI.season_of(date(2027, 3, 20)) == 2027
