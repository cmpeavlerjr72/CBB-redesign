"""Tests for `cbb_sim.features.opponent_adjust` and
`cbb_sim.features.conference`.

Every other sub-model will consume the adjustment (Decision 9), so the
properties that matter are tested on synthetic schedules where the right answer
is known by construction, not on real data where it is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cbb_sim.features import conference as CF
from cbb_sim.features import opponent_adjust as OA

RATE_DEFS = {"tov": ("tov", "poss")}
RATE_SCALE = {"tov": 100.0}


def _round_robin(n_teams: int = 6, n_rounds: int = 4, seed: int = 0,
                 team_rate: dict[int, float] | None = None,
                 allowed_rate: dict[int, float] | None = None) -> pd.DataFrame:
    """A synthetic team-game box. Each game gets both sides. The turnover count
    of team i against team j is `poss * (team_rate[i] + allowed_rate[j])`, i.e.
    the outcome is exactly additive in an offence effect and a defence effect,
    which is the model the adjustment is trying to invert."""
    rng = np.random.default_rng(seed)
    team_rate = team_rate or {i: 0.15 for i in range(n_teams)}
    allowed_rate = allowed_rate or {i: 0.0 for i in range(n_teams)}
    rows = []
    day = pd.Timestamp("2024-11-05")
    gid = 0
    for rd in range(n_rounds):
        order = rng.permutation(n_teams)
        for a, b in zip(order[::2], order[1::2]):
            gid += 1
            date = day + pd.Timedelta(days=3 * rd)
            for i, j in ((a, b), (b, a)):
                poss = 70.0
                rows.append({"season": 2025, "game_id": gid, "game_date": date,
                             "team_id": int(i), "opp_id": int(j),
                             "poss": poss,
                             "tov": poss * (team_rate[int(i)] + allowed_rate[int(j)])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def test_equal_teams_get_a_zero_correction():
    boxes = _round_robin()
    for method in ("one_pass", "iterative"):
        res = OA.adjust_team_form(boxes, RATE_DEFS, RATE_SCALE, method)
        assert np.abs(res.off_adj["tov"].to_numpy()).max() < 1e-6
        assert np.abs(res.def_adj["tov"].to_numpy()).max() < 1e-6


def test_none_is_a_no_op():
    boxes = _round_robin()
    res = OA.adjust_team_form(boxes, RATE_DEFS, RATE_SCALE, "none")
    assert (res.off_adj["tov"] == 0).all() and (res.def_adj["tov"] == 0).all()


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        OA.adjust_rate(_round_robin(), "tov", "poss", 100.0, "kenpom")


def test_a_team_that_played_only_generous_defences_is_adjusted_down():
    """Team 0 plays only team 5, whose defence allows 5 pp more turnovers than
    anyone else's. Team 0's raw offensive turnover deviation is inflated by the
    schedule; the correction must be POSITIVE (so that `dev - correction` moves
    it back toward the league)."""
    n = 6
    allowed = {i: 0.0 for i in range(n)}
    allowed[5] = 0.05
    rows = []
    day = pd.Timestamp("2024-11-05")
    gid = 0
    pairs = [(0, 5), (0, 5), (1, 2), (1, 2), (3, 4), (3, 4), (1, 3), (2, 4),
             (1, 4), (2, 3), (3, 5), (4, 5)]
    for rd, (a, b) in enumerate(pairs):
        gid += 1
        date = day + pd.Timedelta(days=2 * rd)
        for i, j in ((a, b), (b, a)):
            poss = 70.0
            rows.append({"season": 2025, "game_id": gid, "game_date": date,
                         "team_id": i, "opp_id": j, "poss": poss,
                         "tov": poss * (0.15 + allowed[j])})
    boxes = pd.DataFrame(rows)
    off, _dfn, _m = OA.adjust_rate(boxes, "tov", "poss", 100.0, "one_pass")
    last0 = boxes.index[(boxes["team_id"] == 0)][-1]
    last1 = boxes.index[(boxes["team_id"] == 1)][-1]
    assert off.loc[last0] > 0.5, "team 0's schedule inflation was not detected"
    assert off.loc[last0] > off.loc[last1]


def test_iterative_recovers_the_true_offence_effect_better_than_one_pass():
    """On an unbalanced schedule with a genuinely additive truth, the iterative
    solve should sit closer to the true offence effect than the single pass."""
    n = 8
    rng = np.random.default_rng(3)
    true_off = {i: 0.15 + 0.02 * (i - (n - 1) / 2) / n for i in range(n)}
    true_def = {i: 0.03 * (i - (n - 1) / 2) / n for i in range(n)}
    rows, gid = [], 0
    day = pd.Timestamp("2024-11-05")
    # deliberately unbalanced: low-index teams meet low-index teams more often
    for rd in range(14):
        order = rng.permutation(n) if rd % 3 else np.arange(n)
        for a, b in zip(order[::2], order[1::2]):
            gid += 1
            date = day + pd.Timedelta(days=2 * rd)
            for i, j in ((int(a), int(b)), (int(b), int(a))):
                poss = 70.0
                rows.append({"season": 2025, "game_id": gid, "game_date": date,
                             "team_id": i, "opp_id": j, "poss": poss,
                             "tov": poss * (true_off[i] + true_def[j])})
    boxes = pd.DataFrame(rows)
    dev = _final_raw_dev(boxes, "tov", "poss", 100.0)
    truth = np.array([100.0 * true_off[i] for i in range(n)])
    truth = truth - truth.mean()
    err = {}
    for method in ("one_pass", "iterative"):
        off, _d, _m = OA.adjust_rate(boxes, "tov", "poss", 100.0, method)
        last = {t: boxes.index[boxes["team_id"] == t][-1] for t in range(n)}
        adj = np.array([dev[t] - off.loc[last[t]] for t in range(n)])
        adj = adj - adj.mean()
        err[method] = float(np.abs(adj - truth).mean())
    raw_err = float(np.abs((np.array([dev[t] for t in range(n)])
                            - np.mean([dev[t] for t in range(n)])) - truth).mean())
    assert err["one_pass"] < raw_err
    assert err["iterative"] <= err["one_pass"] + 1e-9


def _final_raw_dev(boxes: pd.DataFrame, num: str, den: str, scale: float) -> dict:
    """The raw centred deviation each team carries into its LAST row, using the
    same strictly-before-the-date window the module uses."""
    teams = np.sort(boxes["team_id"].unique())
    dates = np.sort(boxes["game_date"].unique())
    off_dev, _dd, _od, _dd2, _lg = OA._snapshots(boxes, teams, dates, num, den, scale)
    out = {}
    for t in teams:
        k = pd.Index(dates).get_indexer(
            [boxes.loc[boxes["team_id"] == t, "game_date"].max()])[0]
        out[int(t)] = float(off_dev[k, pd.Index(teams).get_indexer([t])[0]])
    return out


def test_strictly_as_of_appending_future_games_changes_nothing():
    """The correction attached to a row must depend only on games strictly
    before that row's date. Appending later games must not move it."""
    boxes = _round_robin(n_teams=8, n_rounds=6, seed=11,
                         team_rate={i: 0.12 + 0.01 * i for i in range(8)},
                         allowed_rate={i: 0.005 * (i % 3) for i in range(8)})
    cut = boxes["game_date"].unique()[3]
    early = boxes[boxes["game_date"] <= cut].copy()
    for method in ("one_pass", "iterative"):
        full_off, full_def, _ = OA.adjust_rate(boxes, "tov", "poss", 100.0, method)
        e_off, e_def, _ = OA.adjust_rate(early, "tov", "poss", 100.0, method)
        np.testing.assert_allclose(full_off.loc[early.index].to_numpy(),
                                   e_off.to_numpy(), atol=1e-10)
        np.testing.assert_allclose(full_def.loc[early.index].to_numpy(),
                                   e_def.to_numpy(), atol=1e-10)


def test_first_date_correction_is_zero_because_there_is_no_history():
    boxes = _round_robin(n_teams=6, n_rounds=3, seed=5,
                         team_rate={i: 0.1 + 0.02 * i for i in range(6)})
    for method in ("one_pass", "iterative"):
        off, dfn, _ = OA.adjust_rate(boxes, "tov", "poss", 100.0, method)
        first = boxes["game_date"] == boxes["game_date"].min()
        assert np.abs(off[first].to_numpy()).max() == 0.0
        assert np.abs(dfn[first].to_numpy()).max() == 0.0


def test_iterative_converges_and_reports_it():
    boxes = _round_robin(n_teams=10, n_rounds=8, seed=7,
                         team_rate={i: 0.10 + 0.01 * i for i in range(10)},
                         allowed_rate={i: 0.004 * i for i in range(10)})
    _o, _d, meta = OA.adjust_rate(boxes, "tov", "poss", 100.0, "iterative")
    assert meta["max_iterations"] <= OA.MAX_ITER
    assert meta["n_dates_not_converged"] == 0
    assert meta["worst_final_delta"] < OA.TOL


def test_seasons_are_adjusted_independently():
    a = _round_robin(n_teams=6, n_rounds=3, seed=1)
    b = _round_robin(n_teams=6, n_rounds=3, seed=2,
                     team_rate={i: 0.30 for i in range(6)})
    b["season"] = 2024
    b["game_id"] = b["game_id"] + 10_000
    b["game_date"] = b["game_date"] - pd.Timedelta(days=365)
    both = pd.concat([a, b], ignore_index=True)
    r_both = OA.adjust_team_form(both, RATE_DEFS, RATE_SCALE, "one_pass")
    r_a = OA.adjust_team_form(a, RATE_DEFS, RATE_SCALE, "one_pass")
    np.testing.assert_allclose(
        r_both.off_adj["tov"].to_numpy()[: len(a)], r_a.off_adj["tov"].to_numpy(), atol=1e-10)


def test_adjust_team_form_returns_one_column_per_rate():
    boxes = _round_robin()
    boxes["fga"] = 55.0
    boxes["fga_3"] = 20.0
    defs = {"tov": ("tov", "poss"), "3pa": ("fga_3", "poss")}
    scale = {"tov": 100.0, "3pa": 100.0}
    res = OA.adjust_team_form(boxes, defs, scale, "one_pass")
    assert list(res.off_adj.columns) == ["tov", "3pa"]
    assert set(res.meta) == {"tov", "3pa"}
    assert 2025 in res.meta["tov"]


# ---------------------------------------------------------------------------
# conference.py
# ---------------------------------------------------------------------------
def test_weekly_boundaries_are_mondays_covering_every_game():
    dates = pd.Series(pd.to_datetime(
        ["2024-11-06", "2024-11-09", "2024-11-20", "2024-12-25", "2025-01-02"]))
    cuts = CF.weekly_boundaries(dates)
    assert all(c.weekday() == 0 for c in cuts)
    assert cuts[0] <= dates.min()
    idx = np.searchsorted(np.array([c.to_datetime64() for c in cuts]),
                          dates.to_numpy(), side="right") - 1
    assert (idx >= 0).all(), "every game must fall at or after some refit date"


def test_union_boundaries_dedupes_and_sorts():
    a = [pd.Timestamp("2024-12-01"), pd.Timestamp("2024-11-01")]
    b = [pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-28")]
    out = CF.union_boundaries(a, b)
    assert out == [pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-01"),
                   pd.Timestamp("2024-12-28")]


def test_first_conference_game_dates_ignores_postseason():
    conf = pd.DataFrame({
        "season": [2025, 2025],
        "game_id": [1, 2],
        "game_date": pd.to_datetime(["2025-03-12", "2025-01-04"]),
        "season_type": [3, 2],
        "home_team_id": [10, 10],
        "away_team_id": [11, 12],
        "home_conference_id": [1.0, 1.0],
        "away_conference_id": [1.0, 1.0],
        "is_conf_game": [True, True],
    })
    out = CF.first_conference_game_dates(conf)
    assert out.loc[out["team_id"] == 10, "first_conf_date"].iloc[0] == pd.Timestamp("2025-01-04")
    assert 11 not in set(out["team_id"]), "a postseason-only opponent has no boundary"
