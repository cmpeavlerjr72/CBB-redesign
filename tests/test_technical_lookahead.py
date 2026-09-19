"""Bounded same-clock technical free-throw lookahead
(`cbb_sim.pbp.possessions`, "TECHNICAL FREE THROWS"; added 2026-09-18).

Every sequence below is a REAL CBBD row sequence, transcribed from the
hand-read games listed in
`docs/tests/event_layer_technical_lookahead_2026-09-18.md` section 1 (game id
in each test's docstring), reduced to the columns the state machine reads.

The first block is the one that matters most: with the flag OFF, every handler
must behave exactly as it did before the flag existed, so the tables on disk
stay bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbb_sim.models import event_stream as ES
from cbb_sim.pbp.possessions import (
    DEFAULT_TECH_LOOKAHEAD,
    TECH_LOOKAHEAD_MAX_ROWS,
    VERSION_TECH_LOOKAHEAD,
    _GameMachine,
    technical_ft_index,
)

HOME, AWAY = 0, 1

META = {"game_id": 1, "cbbd_game_id": 1, "season": 2025,
        "home_team_id": 100, "away_team_id": 200}


def ev(rows: list[tuple]) -> dict[str, np.ndarray]:
    """rows = (cls, team_side, period, seconds_remaining, made)."""
    cls = np.array([r[0] for r in rows], dtype=object)
    return {
        "cls": cls,
        "team": np.array([r[1] for r in rows], dtype="int64"),
        "period": np.array([r[2] for r in rows], dtype="int64"),
        "sec": np.array([r[3] for r in rows], dtype="int64"),
        "made": np.array([bool(r[4]) for r in rows], dtype=bool),
        "hs": np.zeros(len(rows), dtype="int64"),
        "as_": np.zeros(len(rows), dtype="int64"),
        "stolen": np.zeros(len(rows), dtype=bool),
        "on_floor": None,
    }


def run(rows: list[tuple], tech_lookahead: bool) -> _GameMachine:
    m = _GameMachine(META, ev(rows), tech_lookahead=tech_lookahead)
    m.run()
    return m


def summary(m: _GameMachine) -> dict:
    return {
        "n_poss": len(m.possessions),
        "terminals": [p.chances[-1].terminal_event for p in m.possessions],
        "offense": [p.offense_team_id for p in m.possessions],
        "fta": sum(c.fta for p in m.possessions for c in p.chances),
        "ftm": sum(c.ftm for p in m.possessions for c in p.chances),
        "points": sum(c.points for p in m.possessions for c in p.chances),
        "tech_off": sum(p.tech_points_off for p in m.possessions),
        "tech_def": sum(p.tech_points_def for p in m.possessions),
        "n_tech_trips": m.n_tech_trips,
        "team_fouls": dict(m.team_fouls),
    }


# ---------------------------------------------------------------------------
# game 401591429 (2024), p2 563s: Technical -> Lost Ball Turnover -> 2 FTs.
# The 89% case. Alabama (home side here) is technical'd; Southern Miss shoots.
# ---------------------------------------------------------------------------
TOV_INSERT = [
    ("FGA_jump2", HOME, 2, 600, False),   # a live possession for HOME first
    ("OREB", HOME, 2, 565, False),
    ("technical", HOME, 2, 563, False),
    ("TOV", HOME, 2, 563, False),         # the inserted administrative row
    ("FT_made", AWAY, 2, 563, True),
    ("FT_made", AWAY, 2, 563, True),
    ("FGA_3", AWAY, 2, 537, False),
    ("DREB", HOME, 2, 535, False),
]

# ---------------------------------------------------------------------------
# game 401583794 (2024), p1 910s: Technical -> duplicate PersonalFoul on the
# SAME player -> 2 FTs. The 10% case.
# ---------------------------------------------------------------------------
PF_INSERT = [
    ("FGA_jump2", HOME, 1, 930, False),
    ("OREB", HOME, 1, 928, False),
    ("technical", HOME, 1, 910, False),
    ("foul", HOME, 1, 910, False),        # the duplicate row for the technical
    ("FT_made", AWAY, 1, 910, True),
    ("FT_made", AWAY, 1, 910, True),
    ("FGA_3", AWAY, 1, 906, True),
]

# ---------------------------------------------------------------------------
# The case the one-row rule ALREADY catches (game 401597885, 2024, p1 749s
# second technical): Technical -> FT directly. Must not change at all.
# ---------------------------------------------------------------------------
DIRECT = [
    ("FGA_jump2", HOME, 1, 760, False),
    ("OREB", HOME, 1, 758, False),
    ("technical", HOME, 1, 749, False),
    ("FT_made", AWAY, 1, 749, True),
    ("FGA_3", HOME, 1, 710, False),
    ("DREB", AWAY, 1, 708, False),
]

# ---------------------------------------------------------------------------
# A REAL foul trip at the same frozen clock, with no technical anywhere: the
# lookahead must not touch it (game 401706151, 2025, p2 169s).
# ---------------------------------------------------------------------------
REAL_BONUS_TRIP = [("foul", HOME, 2, 400, False)] * 6 + [
    ("FGA_jump2", AWAY, 2, 200, False),
    ("DREB", AWAY, 2, 198, False),
    ("foul", HOME, 2, 169, False),
    ("FT_made", AWAY, 2, 169, True),
    ("FT_made", AWAY, 2, 169, True),
    ("FGA_3", HOME, 2, 150, False),
    ("DREB", AWAY, 2, 148, False),
]

# ---------------------------------------------------------------------------
# Offsetting technicals, one per team, at one clock: no free throws by rule.
# ---------------------------------------------------------------------------
OFFSETTING = [
    ("FGA_jump2", HOME, 2, 500, False),
    ("OREB", HOME, 2, 498, False),
    ("technical", HOME, 2, 480, False),
    ("technical", AWAY, 2, 480, False),
    ("FGA_3", HOME, 2, 460, True),
]

ALL_SEQUENCES = {
    "tov_insert": TOV_INSERT, "pf_insert": PF_INSERT, "direct": DIRECT,
    "real_bonus_trip": REAL_BONUS_TRIP, "offsetting": OFFSETTING,
}


# ===========================================================================
# 1. THE DEFAULT PATH IS UNCHANGED
# ===========================================================================
def test_default_flag_is_off():
    assert DEFAULT_TECH_LOOKAHEAD is False
    assert VERSION_TECH_LOOKAHEAD["v1"] is False
    assert VERSION_TECH_LOOKAHEAD["v2"] is False
    assert VERSION_TECH_LOOKAHEAD["v3"] is True
    assert ES.DEFAULT_TECH_LOOKAHEAD is False


def test_machine_defaults_to_off():
    """Constructing the machine without the keyword must give the old build."""
    m = _GameMachine(META, ev(TOV_INSERT))
    assert m.tech_lookahead is False
    m.run()
    assert summary(m) == summary(run(TOV_INSERT, tech_lookahead=False))


@pytest.mark.parametrize("name", sorted(ALL_SEQUENCES))
def test_flag_off_reproduces_the_documented_defect(name):
    """OFF, the two inserted-row sequences must still MIS-tag the trip -- the
    defect is the documented current behaviour of the tables on disk, and a
    default-path change would silently invalidate them."""
    s = summary(run(ALL_SEQUENCES[name], tech_lookahead=False))
    if name in ("tov_insert", "pf_insert"):
        assert s["n_tech_trips"] == 0            # the trip is missed
        assert s["fta"] == 2 and s["ftm"] == 2   # charged as an ordinary trip
        assert s["tech_off"] == 0 and s["tech_def"] == 0
        assert "FT_trip_shooting" in s["terminals"] or "FT_trip_bonus" in s["terminals"]
    if name == "direct":
        assert s["n_tech_trips"] == 1 and s["fta"] == 0


# ===========================================================================
# 2. THE FIX
# ===========================================================================
def test_lookahead_recovers_the_inserted_turnover_case():
    """game 401591429: the two free throws become a technical trip."""
    on = summary(run(TOV_INSERT, tech_lookahead=True))
    assert on["n_tech_trips"] == 1
    assert on["fta"] == 0 and on["ftm"] == 0      # excluded from trip counts
    assert on["tech_off"] + on["tech_def"] == 2   # but the points still balance
    assert "FT_trip_shooting" not in on["terminals"]
    assert "FT_trip_bonus" not in on["terminals"]


def test_the_inserted_turnover_is_kept_not_deleted():
    """The `Lost Ball Turnover` is a REAL, box-counted turnover (module
    docstring): the fix must still charge it, and to the technical'd team."""
    off = summary(run(TOV_INSERT, tech_lookahead=False))
    on = summary(run(TOV_INSERT, tech_lookahead=True))
    assert on["terminals"].count("TOV") == 1
    assert off["terminals"].count("TOV") == 1
    assert on["offense"][on["terminals"].index("TOV")] == HOME


def test_lookahead_recovers_the_duplicate_personal_foul_case():
    """game 401583794: the duplicate `PersonalFoul` row still counts toward the
    team total, but no longer claims the technical's free throws."""
    off = summary(run(PF_INSERT, tech_lookahead=False))
    on = summary(run(PF_INSERT, tech_lookahead=True))
    assert on["n_tech_trips"] == 1 and off["n_tech_trips"] == 0
    assert on["fta"] == 0 and off["fta"] == 2
    assert on["tech_off"] + on["tech_def"] == 2
    assert on["team_fouls"] == off["team_fouls"]      # the foul still counts
    assert on["team_fouls"][HOME] == 1


def test_direct_case_is_bit_identical_under_both_flags():
    assert summary(run(DIRECT, True)) == summary(run(DIRECT, False))


def test_real_foul_trip_is_untouched():
    """No technical in the sequence: the flag must change nothing."""
    assert summary(run(REAL_BONUS_TRIP, True)) == summary(run(REAL_BONUS_TRIP, False))
    assert summary(run(REAL_BONUS_TRIP, True))["fta"] == 2


def test_offsetting_technicals_produce_no_trip_under_either_flag():
    assert summary(run(OFFSETTING, True)) == summary(run(OFFSETTING, False))
    assert summary(run(OFFSETTING, True))["n_tech_trips"] == 0


def test_points_reconcile_under_both_flags():
    """Whatever bucket the free throws land in, the total points a game
    accounts for must not move -- that is the score reconciliation the module
    docstring promises."""
    for rows in ALL_SEQUENCES.values():
        off, on = summary(run(rows, False)), summary(run(rows, True))
        assert off["points"] + off["tech_off"] + off["tech_def"] == \
               on["points"] + on["tech_off"] + on["tech_def"]


# ===========================================================================
# 3. THE SCAN ITSELF
# ===========================================================================
def test_scan_stops_when_the_clock_moves():
    rows = [("technical", HOME, 2, 563, False), ("TOV", HOME, 2, 562, False),
            ("FT_made", AWAY, 2, 562, True)]
    e = ev(rows)
    assert technical_ft_index(0, e["cls"], e["team"], e["period"], e["sec"], AWAY, 3) == -1


def test_scan_stops_on_a_live_ball_shot():
    rows = [("technical", HOME, 2, 563, False), ("FGA_3", AWAY, 2, 563, False),
            ("FT_made", AWAY, 2, 563, True)]
    e = ev(rows)
    assert technical_ft_index(0, e["cls"], e["team"], e["period"], e["sec"], AWAY, 3) == -1


def test_scan_stops_at_a_free_throw_by_the_wrong_team():
    rows = [("technical", HOME, 2, 563, False), ("TOV", HOME, 2, 563, False),
            ("FT_made", HOME, 2, 563, True)]
    e = ev(rows)
    assert technical_ft_index(0, e["cls"], e["team"], e["period"], e["sec"], AWAY, 3) == -1


def test_scan_respects_its_row_budget():
    rows = ([("technical", HOME, 2, 563, False)]
            + [("DeadBallReb", HOME, 2, 563, False)] * (TECH_LOOKAHEAD_MAX_ROWS + 1)
            + [("FT_made", AWAY, 2, 563, True)])
    e = ev(rows)
    n = len(rows)
    assert technical_ft_index(0, e["cls"], e["team"], e["period"], e["sec"], AWAY, n) == -1
    assert technical_ft_index(0, e["cls"], e["team"], e["period"], e["sec"], AWAY, n,
                              max_rows=TECH_LOOKAHEAD_MAX_ROWS + 2) == n - 1


def test_scan_respects_a_game_boundary():
    rows = [("technical", HOME, 2, 563, False), ("TOV", HOME, 2, 563, False),
            ("FT_made", AWAY, 2, 563, True)]
    e = ev(rows)
    game = np.array([1, 1, 2], dtype="int64")
    assert technical_ft_index(0, e["cls"], e["team"], e["period"], e["sec"], AWAY, 3,
                              game=game) == -1
