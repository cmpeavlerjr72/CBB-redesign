"""created_at < tipoff guard vs MIDNIGHT PLACEHOLDER tips (ops 2026-10-09, docs/ops/tip_guard_2026-10-09.md).

A 00:00 ET tip is a placeholder: it proves only that the game is not before 00:00 ET of its date. A build before that bound may be served
(flagged unverified); a same-day build after it is NOT proven pre-tip and must be refused as "tip time unknown", not passed and not mislabelled
as tipped; a real same-day tip still in the future must pass. Grading re-checks placeholder rows against the real tip once known.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from cbb_sim.live import daily as D  # noqa: E402
from cbb_sim.live import guards as G  # noqa: E402
from cbb_sim.live import tips as TP  # noqa: E402

DAY = "2026-11-02"                                   # EST: 00:00 ET = 05:00Z
MIDNIGHT = pd.Timestamp("2026-11-02T05:00:00Z")
REAL = pd.Timestamp("2026-11-03T00:00:00Z")          # 19:00 ET on the slate date
EVENING_BEFORE = pd.Timestamp("2026-11-02T01:00:00Z")  # 20:00 ET on 11-01
SAME_DAY = pd.Timestamp("2026-11-02T15:00:00Z")      # 10:00 ET on the slate date


def slate(flag_col: bool = True) -> pd.DataFrame:
    s = pd.DataFrame({"game_id": [1, 2], "cbbd_game_id": [11, 12], "season": [2027, 2027], "game_date": [DAY, DAY],
                      "tipoff_utc": [MIDNIGHT, REAL], "home_team_id": [10, 20], "away_team_id": [30, 40], "neutral": [False, False],
                      "tip_source": ["cbbd_placeholder_midnight_et", "cbbd"]})
    if flag_col:
        s["tip_time_is_placeholder"] = [True, False]
    return s


def games(ids=(1, 2), created=EVENING_BEFORE) -> pd.DataFrame:
    rows = []
    for g in ids:
        for sd in range(3):
            rows.append(dict(game_id=g, seed=sd, home_pts=70 + sd, away_pts=65, possessions=68, n_periods=2, created_at=created))
    return pd.DataFrame(rows)


def test_raw_guard_alone_passes_a_placeholder_so_it_is_not_the_proof():
    # the row guard compares against the placeholder and passes: this is exactly why placeholder rows must carry the flag below
    G.assert_created_before_tipoff(pd.DataFrame({"created_at": [EVENING_BEFORE], "tipoff_utc": [MIDNIGHT]}))
    with pytest.raises(G.LeakGuardError):
        G.assert_created_before_tipoff(pd.DataFrame({"created_at": [SAME_DAY], "tipoff_utc": [MIDNIGHT]}))


@pytest.mark.parametrize("flag_col", [True, False])
def test_same_day_build_refuses_placeholder_as_tip_unknown_and_keeps_real_tip(flag_col):
    ok, ref = TP.select_for_pass(slate(flag_col), SAME_DAY, "evening")
    assert ok["game_id"].tolist() == [2]                       # real 19:00 ET tip still ahead: NOT blocked
    assert ref["game_id"].tolist() == [1]
    assert ref["refuse_reason"].iloc[0] == TP.TIP_UNKNOWN       # not "tipped", not passed


def test_evening_before_build_serves_placeholder_flagged_unverified():
    ok, ref = TP.select_for_pass(slate(), EVENING_BEFORE, "evening")
    assert sorted(ok["game_id"]) == [1, 2] and not len(ref)
    out = TP.stamp_pre_tip_basis(games(created=EVENING_BEFORE).assign(tipoff_utc=lambda d: d["game_id"].map({1: MIDNIGHT, 2: REAL})),
                                 ok, DAY)
    v = out.drop_duplicates("game_id").set_index("game_id")
    assert not v.at[1, "pre_tip_verified"] and v.at[1, "pre_tip_basis"] == "placeholder_lower_bound"
    assert v.at[2, "pre_tip_verified"] and v.at[2, "pre_tip_basis"] == "real_tip"


def test_feed_flagged_non_midnight_placeholder_is_bounded_by_midnight_of_game_date():
    # hoopR placeholder 05:00Z in EDT = 01:00 ET; a 00:30 ET clock is past the 00:00 ET bound, so the tip is unknown
    s = pd.DataFrame({"game_id": [5], "game_date": ["2026-10-31"], "tipoff_utc": [pd.Timestamp("2026-10-31T05:00:00Z")],
                      "tip_time_is_placeholder": [True]})
    ok, ref = TP.select_for_pass(s, pd.Timestamp("2026-10-31T04:30:00Z"), "evening")
    assert not len(ok) and ref["refuse_reason"].iloc[0] == TP.TIP_UNKNOWN
    ok, _ = TP.select_for_pass(s, pd.Timestamp("2026-10-31T03:30:00Z"), "evening")     # 23:30 ET the day before: servable
    assert len(ok) == 1


def test_split_tipped_without_flag_column_treats_midnight_as_placeholder():
    ok, late = D.split_tipped(slate(flag_col=False), SAME_DAY)
    assert ok["game_id"].tolist() == [2] and late["game_id"].tolist() == [1]
    assert TP.late_reasons(late).tolist() == [TP.TIP_UNKNOWN]
    ok, late = D.split_tipped(slate(flag_col=False), EVENING_BEFORE)
    assert len(ok) == 2 and not len(late)


def test_publish_flags_placeholder_rows_and_labels_same_day_refusals():
    pub, refused = D.build_publish(games(), slate(), None, EVENING_BEFORE, "r")
    v = pub.set_index("game_id")
    assert not v.at[1, "pre_tip_verified"] and v.at[2, "pre_tip_verified"]
    pub, refused = D.build_publish(games(created=EVENING_BEFORE), slate(), None, SAME_DAY, "r")
    assert pub["game_id"].tolist() == [2]
    assert refused.set_index("game_id").at[1, "refused_reason"] == "tip_time_unknown_placeholder"


def test_stamp_rows_raises_for_placeholder_row_built_after_the_bound():
    rows = slate().assign(published_at=SAME_DAY)
    with pytest.raises(G.LeakGuardError):
        TP.stamp_pre_tip_rows(rows.iloc[[0]], "published_at")
    out = TP.stamp_pre_tip_rows(rows.iloc[[1]], "published_at")          # real tip after the clock: fine
    assert bool(out["pre_tip_verified"].iloc[0])


def test_grade_reverify_against_real_tip():
    pub = slate().assign(published_at=EVENING_BEFORE)
    pub = pd.concat([pub, slate().iloc[[0]].assign(game_id=3, published_at=pd.Timestamp("2026-11-02T20:00:00Z"))], ignore_index=True)
    tt = pd.DataFrame({"game_id": [1, 3], "tipoff_utc": [pd.Timestamp("2026-11-02T23:00:00Z"), pd.Timestamp("2026-11-02T17:00:00Z")],
                       "tip_time_is_placeholder": [False, False]})
    out = TP.reverify_pre_tip(pub, tt, "published_at").set_index("game_id")
    assert out.at[1, "pre_tip_status"] == "placeholder_reverified"
    assert out.at[2, "pre_tip_status"] == "real_tip"
    assert out.at[3, "pre_tip_status"] == "violated"              # built 15:00 ET, real tip 12:00 ET: never graded
    out = TP.reverify_pre_tip(pub.iloc[[0]], None, "published_at")
    assert out["pre_tip_status"].iloc[0] == "placeholder_unverified"   # no real tip yet: stays flagged, not assumed


def test_grade_stage_never_grades_a_violated_row(tmp_path, monkeypatch):
    import grade_daily_v1 as GR
    pub, _ = D.build_publish(games(), slate(), None, EVENING_BEFORE, "r")
    pub = pub[pub["game_id"] == 1].assign(published_at=pd.Timestamp("2026-11-02T20:00:00Z"))   # placeholder row, stamped 15:00 ET
    monkeypatch.setattr(GR, "load_publications", lambda root, d, run_id=None: [pub])
    monkeypatch.setattr(D, "first_publication", lambda pubs, pid=None: pubs[0])
    real = pd.DataFrame({"game_id": [1], "tipoff_utc": [pd.Timestamp("2026-11-02T17:00:00Z")], "tip_time_is_placeholder": [False]})
    tdir = tmp_path / "data/processed/ingest"
    tdir.mkdir(parents=True)
    real.to_parquet(tdir / "tip_times_2027.parquet")
    monkeypatch.setattr(GR, "REPO", tmp_path)
    fin = pd.DataFrame({"game_id": [1], "home_score": [70], "away_score": [60], "finals_source": ["test"]})
    r = GR.run_grade_stage(DAY, 2027, now="2026-11-03T12:00:00Z", root=tmp_path / "daily", finals_frame=fin, truth_box=pd.DataFrame(),
                           tov_monitor=False, n_boot=10)
    assert r["graded"] == 0 and r["pre_tip_status"].get("violated") == 1
    pend = pd.read_parquet(tmp_path / "daily/grade" / DAY / "pending.parquet")
    assert pend["reason"].tolist() == ["created_after_real_tip"]
