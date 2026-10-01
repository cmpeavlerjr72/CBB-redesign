"""Two-pass chain: placeholder tips never certify a row as pre-tip (lane F, 2026-10-01)."""
import pandas as pd
import pytest

from cbb_sim.live import guards as G
from cbb_sim.live import tips as TP

D = "2026-11-02"


def _slate():
    mid = TP.earliest_possible_tip(D)
    return pd.DataFrame({"game_id": [1, 2, 3], "tipoff_utc": [mid, mid + pd.Timedelta(hours=19), mid],
                         "tip_time_is_placeholder": [True, False, True], "tip_source": ["cbbd_placeholder_midnight_et", "cbbd", "x"]})


def test_flag_placeholders_catches_midnight_without_feed_flag():
    s = _slate().drop(columns="tip_time_is_placeholder")
    assert TP.flag_placeholders(s)["tip_time_is_placeholder"].tolist() == [True, False, True]


def test_evening_includes_placeholders_morning_refuses_them():
    s = _slate()
    ok, late = TP.select_for_pass(s, TP.default_clock(D, "evening"), "evening")
    assert len(ok) == 3 and len(late) == 0
    ok, late = TP.select_for_pass(s, TP.default_clock(D, "morning"), "morning")
    assert ok["game_id"].tolist() == [2]
    assert set(late["refuse_reason"]) == {"placeholder tip time (not real); morning pass needs a real tip"}


def test_stamp_marks_placeholder_rows_unverified():
    s = _slate()
    now = TP.default_clock(D, "evening")
    g = s[["game_id", "tipoff_utc"]].assign(created_at=now)
    out = TP.stamp_pre_tip_basis(g, s, D)
    assert out.set_index("game_id")["pre_tip_verified"].to_dict() == {1: False, 2: True, 3: False}
    assert not out.loc[out["tip_time_is_placeholder"], "pre_tip_verified"].any()


def test_placeholder_row_after_earliest_possible_tip_raises():
    s = _slate()
    g = s[["game_id", "tipoff_utc"]].assign(created_at=TP.earliest_possible_tip(D) + pd.Timedelta(hours=1))
    with pytest.raises(G.LeakGuardError):
        TP.stamp_pre_tip_basis(g, s, D)
