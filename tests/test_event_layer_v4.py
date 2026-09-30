"""Event layer v4 switches (`cbb_sim.pbp.possessions`, "EVENT LAYER v4";
added 2026-09-30; docs/tests/event_layer_v4_2026-09-30.md).

Sequences are reduced from rows read in
docs/tests/g1_g5_possessions_corr_diagnostic_2026-09-30.md section 5.1
(`results/g1g5_diag/unknown_sample_rows.txt`).  Every test checks both the
default machine (switches off: the v2 behaviour, phantom present) and the v4
machine (phantom gone, events conserved).
"""
from __future__ import annotations

import numpy as np

from cbb_sim.pbp.possessions import (
    EVENT_FIX_SWITCHES,
    STRAY_REB_MAX_S,
    VERSION_EVENT_FIXES,
    _GameMachine,
)

HOME, AWAY = 0, 1
META = {"game_id": 1, "cbbd_game_id": 1, "season": 2025, "home_team_id": 100, "away_team_id": 200}
V4 = VERSION_EVENT_FIXES["v4"]


def ev(rows):
    return {
        "cls": np.array([r[0] for r in rows], dtype=object),
        "team": np.array([r[1] for r in rows], dtype="int64"),
        "period": np.array([r[2] for r in rows], dtype="int64"),
        "sec": np.array([r[3] for r in rows], dtype="int64"),
        "made": np.array([bool(r[4]) for r in rows], dtype=bool),
        "hs": np.zeros(len(rows), dtype="int64"),
        "as_": np.zeros(len(rows), dtype="int64"),
        "stolen": np.zeros(len(rows), dtype=bool),
        "on_floor": None,
    }


def run(rows, **kw):
    m = _GameMachine(META, ev(rows), **kw)
    m.run()
    return m


def events(m):
    ch = [c for p in m.possessions for c in p.chances]
    return {k: sum(getattr(c, k) for c in ch) for k in
            ("fga_rim", "fgm_rim", "fga_jump2", "fgm_jump2", "fga_3", "fgm_3", "fta", "ftm", "points")}


def test_switches_default_off_and_v4_on():
    assert set(V4) == set(EVENT_FIX_SWITCHES) and all(V4.values())
    assert not any(VERSION_EVENT_FIXES["v2"].values())
    assert STRAY_REB_MAX_S == 3


# game 401371721 (2022) p2 98 s: made layup, foul, missed and-one FT, DREB.
ANDONE_MISS_DREB = [
    ("FGA_rim", HOME, 2, 110, True),       # HOME scores (closes HOME, AWAY due)
    ("FGA_rim", AWAY, 2, 98, True),        # AWAY made layup ...
    ("foul", HOME, 2, 98, False),
    ("FT_missed", AWAY, 2, 98, False),     # ... and-one missed
    ("DREB", HOME, 2, 98, False),          # live rebound by HOME
    ("FGA_3", HOME, 2, 80, False),
    ("DREB", AWAY, 2, 78, False),
]


def test_class_A_phantom_removed():
    d = run(ANDONE_MISS_DREB)
    v = run(ANDONE_MISS_DREB, **V4)
    assert [p.chances[-1].terminal_event for p in d.possessions].count("unknown") == 1
    assert [p.chances[-1].terminal_event for p in v.possessions].count("unknown") == 0
    assert len(v.possessions) == len(d.possessions) - 1
    assert events(v) == events(d)
    # the possession after the rebound starts DREB, the AWAY and-one possession ends at the rebound
    offs = [p.offense_team_id for p in v.possessions]
    assert offs == [HOME, AWAY, HOME]
    assert v.possessions[2].start_reason == "DREB"


ANDONE_MISS_OWN_OREB = [
    ("FGA_rim", HOME, 1, 900, True),
    ("FGA_jump2", AWAY, 1, 880, True),
    ("foul", HOME, 1, 880, False),
    ("FT_missed", AWAY, 1, 880, False),
    ("OREB", AWAY, 1, 879, False),         # the shooter's own team rebounds
    ("FGA_rim", AWAY, 1, 875, True),
]


def test_class_E_restart_becomes_continuation():
    d = run(ANDONE_MISS_OWN_OREB)
    v = run(ANDONE_MISS_OWN_OREB, **V4)
    assert len(v.possessions) == len(d.possessions) - 1
    away = [p for p in v.possessions if p.offense_team_id == AWAY]
    assert len(away) == 1 and len(away[0].chances) == 2
    assert away[0].chances[1].start_reason == "OREB"
    assert events(v) == events(d)


ANDONE_MADE = [
    ("FGA_rim", HOME, 1, 900, True),
    ("FGA_rim", AWAY, 1, 880, True),
    ("foul", HOME, 1, 880, False),
    ("FT_made", AWAY, 1, 880, True),
    ("FGA_3", HOME, 1, 860, False),
    ("DREB", AWAY, 1, 858, False),
]


def test_made_andone_start_label_follows_the_floor():
    d = run(ANDONE_MADE)
    v = run(ANDONE_MADE, **V4)
    assert len(v.possessions) == len(d.possessions)
    assert d.possessions[2].start_reason == "made_FG"
    assert v.possessions[2].start_reason == "made_FT"
    assert events(v) == events(d)
    # clock round 6 arm L2a: phantom fix without the made-branch relabel
    a = run(ANDONE_MADE, andone_made_next="made_FG", **V4)
    assert a.possessions[2].start_reason == "made_FG"
    b = run(ANDONE_MISS_DREB, andone_made_next="made_FG", **V4)
    assert len(b.possessions) == len(run(ANDONE_MISS_DREB, **V4).possessions)


# B1: two made FTs, then a stray DREB row by the shooting team 0-1 s later
STRAY_DREB_AFTER_FT = [
    ("FGA_3", AWAY, 2, 60, False),
    ("DREB", HOME, 2, 58, False),
    ("foul", AWAY, 2, 34, False),
    ("FT_made", HOME, 2, 34, True),
    ("FT_made", HOME, 2, 34, True),
    ("DREB", HOME, 2, 34, False),          # stray
    ("FGA_3", AWAY, 2, 27, False),
    ("DREB", HOME, 2, 25, False),
]


def test_class_B1_stray_rebound_ignored():
    d = run(STRAY_DREB_AFTER_FT)
    v = run(STRAY_DREB_AFTER_FT, **V4)
    assert len(v.possessions) == len(d.possessions) - 1
    assert v.n_stray_dreb == 1
    assert events(v) == events(d)
    assert "unknown" not in [p.chances[-1].terminal_event for p in v.possessions]


# B2 > 3 s: a DREB 14 s after the last close with no shot logged: a REAL possession, kept
MISSING_SHOT = [
    ("FGA_3", HOME, 1, 1076, False),
    ("DREB", AWAY, 1, 1072, False),
    ("DREB", HOME, 1, 1058, False),        # AWAY's missed shot is absent from the feed
    ("FGA_rim", HOME, 1, 1053, True),
]


def test_class_B2_long_kept_as_real():
    d = run(MISSING_SHOT)
    v = run(MISSING_SHOT, **V4)
    assert len(v.possessions) == len(d.possessions)
    assert [p.chances[-1].terminal_event for p in v.possessions].count("unknown") == 1


# C1: HOME scores, then a stray HOME OREB, then AWAY shoots
STRAY_OREB = [
    ("FGA_rim", AWAY, 2, 80, True),
    ("FGA_3", HOME, 2, 74, True),
    ("OREB", HOME, 2, 59, False),          # stray: HOME just scored
    ("foul", HOME, 2, 56, False),
    ("FT_made", AWAY, 2, 56, True),
    ("FT_made", AWAY, 2, 56, True),
]


def test_class_C1_stray_oreb_ignored():
    d = run(STRAY_OREB)
    v = run(STRAY_OREB, **V4)
    assert len(v.possessions) == len(d.possessions) - 1
    assert v.n_stray_oreb == 1
    assert events(v) == events(d)
    assert "unknown" not in [p.chances[-1].terminal_event for p in v.possessions]
