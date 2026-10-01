"""OT team-foul carry switch (lane F, 2026-10-01): default off is unchanged; on carries the 2nd-half count into OT only."""
import numpy as np
import pandas as pd

from cbb_sim.models import event_stream as ES
from cbb_sim.pbp import possessions as PZ


def _df():
    # one game, home (side 0) fouls: 2 in period 1, 3 in period 2, then a foul in OT
    rows = [(1, 1, "foul", 0), (1, 1, "foul", 0), (1, 2, "foul", 0), (1, 2, "foul", 0), (1, 2, "foul", 0),
            (1, 3, "foul", 0), (1, 3, "shot", 1)]
    return pd.DataFrame(rows, columns=["cbbd_game_id", "period", "cls", "side"])


def test_event_stream_default_resets_per_period(monkeypatch):
    monkeypatch.delenv(ES.OT_FOUL_CARRY_ENV, raising=False)
    out = ES._attach_team_fouls(_df())
    assert out["fouls_home"].tolist() == [0, 1, 0, 1, 2, 0, 1]


def test_event_stream_carry_continues_second_half_into_ot(monkeypatch):
    monkeypatch.setenv(ES.OT_FOUL_CARRY_ENV, "1")
    out = ES._attach_team_fouls(_df())
    assert out["fouls_home"].tolist() == [0, 1, 0, 1, 2, 3, 4]   # period 1 reset at the half; OT continues from 3


def test_version_registry_v4_unchanged_and_v4otc_adds_one_switch():
    assert "ot_foul_carry" not in PZ.VERSION_EVENT_FIXES["v4"]
    assert not any(PZ.VERSION_EVENT_FIXES["v2"].values())
    assert PZ.VERSION_EVENT_FIXES["v4otc"] == {**PZ.VERSION_EVENT_FIXES["v4"], "ot_foul_carry": True}
    assert PZ.VERSION_TECH_LOOKAHEAD["v4otc"] is True
