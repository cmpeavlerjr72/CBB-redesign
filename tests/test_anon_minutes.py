"""Anonymous-minutes explanation: team_anon_minutes / publish notes. Explanation only; no existing number changes."""
import numpy as np
import pandas as pd

from cbb_sim.live import daily as D


def _frames():
    slate = pd.DataFrame({"game_id": [1, 2], "home_team_id": [10, 30], "away_team_id": [20, 40]})
    games = pd.DataFrame({"game_id": [1, 1, 2, 2], "seed": [0, 1, 0, 1], "n_periods": [2, 3, 2, 2]})
    # game 1 seed 0: home named 190 (anon 10), away named 200 (anon 0); seed 1 (1 OT = 225): home 225 (0), away 205 (20)
    # game 2: team 30 has no player rows at all (fully anonymous); team 40 named 200
    rows = [(1, 0, 10, 190.0), (1, 0, 20, 200.0), (1, 1, 10, 225.0), (1, 1, 20, 205.0), (2, 0, 40, 200.0), (2, 1, 40, 200.0)]
    pl = pd.DataFrame(rows, columns=["game_id", "seed", "team_id", "minutes"])
    return pl, games, slate


def test_anon_minutes_mean_over_seeds_and_ot():
    pl, games, slate = _frames()
    a = D.anon_minutes_by_team_game(pl, games, slate).set_index(["game_id", "team_id"])["team_anon_minutes"]
    assert a[(1, 10)] == 5.0 and a[(1, 20)] == 10.0
    assert a[(2, 30)] == 200.0 and a[(2, 40)] == 0.0       # fully anonymous team-game still explained


def test_note_string_and_empty():
    pl, games, slate = _frames()
    n = D.anon_minutes_note(D.anon_minutes_by_team_game(pl, games, slate), slate)
    assert list(n) == ["anon_minutes home=5.0 away=10.0", "anon_minutes home=200.0 away=0.0"]
    e = D.anon_minutes_by_team_game(None, games, slate)
    assert list(D.anon_minutes_note(e, slate)) == ["", ""]


def test_publish_id_unaffected_by_notes():
    pub = pd.DataFrame({"game_id": [1], "provider": ["x"], "spread": [1.0], "total": [2.0], "home_ml": [1.0], "away_ml": [2.0], "line_kind": ["close"]})
    assert D.publish_id(pub, "r") == D.publish_id(pub.assign(notes="anon_minutes home=1.0 away=1.0"), "r")
