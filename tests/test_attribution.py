"""
tests/test_attribution.py -- the L4 PLAYER ATTRIBUTION model's invariants.

Four groups, in the order a reviewer should read them:

  A. LEAK SAFETY. The event's own recorded attribution never reaches a feature.
     Proved two ways, the way `tests/test_usage.py` proves it: an independent
     strictly-earlier recomputation of every numerator and denominator, and
     invariance of a row's features to corrupting its OWN game while a LATER
     game does move them.
  B. ELIGIBLE-SET CORRECTNESS. The shooter is never an assist candidate, the
     candidate side is the right side of the ball, and a duplicated or missing
     on-floor id disqualifies the choice set rather than producing a short one.
  C. DETERMINISM AND THE RNG CONTRACT. Same (seed, game_id) repeats; a different
     seed, game or credit family differs; each credit has its own counter; the
     empirical frequencies equal the rate table's shares.
  D. THE PORTS. The K-generic game-level check reproduces `usage`'s five-only
     one, and the K-generic grouped-softmax objective reproduces `usage`'s at
     K = 5, so a number quoted next to a usage number is the same statistic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import attribution as A  # noqa: E402
from cbb_sim.models import usage as U  # noqa: E402


# ===========================================================================
# Fixtures: a tiny synthetic league with no I/O
# ===========================================================================
def _pop_table(rows: list[dict], pop: str) -> pd.DataFrame:
    """A population table with the columns `build_attr_events` emits."""
    d = pd.DataFrame(rows)
    d["cbbd_game_id"] = d["game_id"]
    d["neutral_site"] = False
    d["cand_opp_id"] = 999
    d["off_team_id"] = d.get("off_team_id", d["cand_team_id"])
    d["def_team_id"] = d.get("def_team_id", d["cand_opp_id"])
    d["cand_is_home"] = True
    d["offense_is_home"] = True
    d["period"] = 1
    d["sec_remaining"] = 1000
    d["score_diff"] = 0
    d["event_class"] = "OREB" if pop.startswith("reb") else "FGA_rim"
    d["blocked_flag"] = False
    if "shot_class" not in d:
        d["shot_class"] = "FGA_rim"
    if "off_player_id" not in d:
        d["off_player_id"] = np.nan
    if "b" not in d:
        d["b"] = 1
    five = d[[f"cand_{k}" for k in range(1, 6)]].to_numpy(dtype="float64")
    cand, ok = A._sorted_candidates(five, None, 5)
    for k in range(5):
        d[f"cand_{k + 1}"] = cand[:, k]
    in_set, slot = A._credit_slot(cand, d["credited_id"].to_numpy(dtype="float64"), ok)
    d["five_ok"] = ok
    d["in_five"] = in_set
    d["y"] = slot
    d["credit_id_present"] = np.isfinite(d["credited_id"].to_numpy(dtype="float64"))
    return d


def _synthetic_season(season: int = 2025, n_games: int = 12,
                      seed: int = 7) -> dict[str, pd.DataFrame]:
    """One season of OREB opportunities over four teams of six players each, plus
    the three binary populations, with enough repetition per player that an as-of
    rate is defined."""
    rng = np.random.default_rng(seed)
    teams = {10: [1, 2, 3, 4, 5, 6], 20: [11, 12, 13, 14, 15, 16],
             30: [21, 22, 23, 24, 25, 26], 40: [31, 32, 33, 34, 35, 36]}
    reb, made, tov, miss = [], [], [], []
    gid = 5_000_000
    for g in range(n_games):
        for t, squad in teams.items():
            gid += 1
            date = pd.Timestamp("2024-11-05") + pd.Timedelta(days=g)
            five = sorted(rng.choice(squad, 5, replace=False).tolist())
            # a deliberately skewed truth: the biggest id takes 50% of the boards
            w = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
            for _ in range(8):
                who = int(rng.choice(five, p=w / w.sum()))
                reb.append({"season": season, "game_id": gid, "game_date": date,
                            "cand_team_id": t, "credited_id": float(who),
                            **{f"cand_{k + 1}": float(five[k]) for k in range(5)}})
            for _ in range(6):
                shooter = int(rng.choice(five))
                others = [p for p in five if p != shooter]
                assisted = bool(rng.random() < 0.5)
                made.append({"season": season, "game_id": gid, "game_date": date,
                             "cand_team_id": t, "off_player_id": float(shooter),
                             "b": int(assisted),
                             "credited_id": (float(rng.choice(others)) if assisted
                                             else np.nan),
                             **{f"cand_{k + 1}": float(five[k]) for k in range(5)}})
            for _ in range(4):
                stolen = bool(rng.random() < 0.4)
                tov.append({"season": season, "game_id": gid, "game_date": date,
                            "cand_team_id": t, "off_player_id": float(rng.choice(five)),
                            "b": int(stolen),
                            "credited_id": (float(rng.choice(five)) if stolen
                                            else np.nan),
                            "shot_class": "none",
                            **{f"cand_{k + 1}": float(five[k]) for k in range(5)}})
            for _ in range(5):
                blocked = bool(rng.random() < 0.2)
                miss.append({"season": season, "game_id": gid, "game_date": date,
                             "cand_team_id": t, "off_player_id": float(rng.choice(five)),
                             "b": int(blocked),
                             "credited_id": (float(rng.choice(five)) if blocked
                                             else np.nan),
                             **{f"cand_{k + 1}": float(five[k]) for k in range(5)}})
    return {"reb_off": _pop_table(reb, "reb_off"),
            "reb_def": _pop_table(reb, "reb_def"),
            "made_fga": _pop_table(made, "made_fga"),
            "tov": _pop_table(tov, "tov"),
            "miss_fga": _pop_table(miss, "miss_fga")}


@pytest.fixture(scope="module")
def pops() -> dict[str, pd.DataFrame]:
    p = _synthetic_season()
    # `build_attr_events` voids the five-slot index on the made-FGA table, so the
    # fixture must too -- the assist target derives its own four-slot index.
    p["made_fga"]["y"] = np.int8(-1)
    return p


@pytest.fixture(scope="module")
def positions() -> pd.DataFrame:
    ids = list(range(1, 40))
    grp = (["G", "F", "C"] * (len(ids) // 3 + 1))[:len(ids)]
    return pd.DataFrame({"shooter_id": ids, "position_group": grp})


@pytest.fixture(scope="module")
def asof(pops, positions) -> pd.DataFrame:
    return A.build_player_asof({2025: pops}, positions=positions)


# ===========================================================================
# A. Leak safety
# ===========================================================================
def test_a_asof_rate_equals_an_independent_strictly_earlier_recomputation(
        pops, asof):
    """Every as-of numerator and denominator recomputed from scratch over that
    player's STRICTLY EARLIER games of the season must equal the shipped column.

    Independent of the production path: this walks the event table directly
    instead of calling `prob_metrics.expanding_asof`, so a bug in the expanding
    helper cannot hide behind itself."""
    d = A.usable(pops["reb_off"], "REB_off")
    long = A._cand_long(d, 5, "oreb")
    for _ in range(1):
        pass
    checked = 0
    for (season, pid), grp in long.groupby(["season", "player_id"]):
        games = (grp.groupby("game_id")
                 .agg(date=("game_date", "first"), opp=("credited", "size"),
                      num=("credited", "sum")).reset_index()
                 .sort_values(["date", "game_id"]))
        cum_o, cum_n = 0.0, 0.0
        for _, row in games.iterrows():
            a = asof[(asof.season == season) & (asof.player_id == pid)
                     & (asof.game_id == row.game_id)]
            assert len(a) == 1
            assert a["opp_oreb"].iloc[0] == pytest.approx(cum_o)
            assert a["num_oreb"].iloc[0] == pytest.approx(cum_n)
            cum_o += row.opp
            cum_n += row.num
            checked += 1
    assert checked > 100


def test_a_the_events_own_attribution_never_enters_its_own_features(pops, positions):
    """Corrupting the credited player of every event of ONE game must not move
    that game's own design features; corrupting a LATER game's must not either,
    but corrupting an EARLIER game's must.

    This is the leak test stated as an invariance, which catches a leak that an
    off-by-one in the expanding sum would produce and that a recomputation test
    sharing the same off-by-one would not."""
    base = {k: v.copy() for k, v in pops.items()}
    d0 = A.usable(base["reb_off"], "REB_off")
    a0 = A.build_player_asof({2025: base}, positions=positions)
    des0 = A.build_choice_design(d0, a0, "REB_off")
    feat = [c for c in des0.columns if c.startswith(("num_oreb_", "opp_oreb_",
                                                     "games_asof_"))]

    games = sorted(base["reb_off"]["game_id"].unique())
    mid = games[len(games) // 2]
    for target_game, must_change in ((mid, False), (games[-1], False),
                                     (games[0], True)):
        bad = {k: v.copy() for k, v in base.items()}
        t = bad["reb_off"]
        m = t["game_id"].to_numpy() == target_game
        # re-credit every event of that game to its own first candidate
        t.loc[m, "credited_id"] = t.loc[m, "cand_1"].to_numpy()
        t.loc[m, "y"] = 0
        a1 = A.build_player_asof({2025: bad}, positions=positions)
        des1 = A.build_choice_design(A.usable(bad["reb_off"], "REB_off"), a1,
                                     "REB_off")
        rows_mid = des1["game_id"].to_numpy() == mid
        same = np.allclose(des0.loc[rows_mid, feat].to_numpy(dtype="float64"),
                           des1.loc[rows_mid, feat].to_numpy(dtype="float64"),
                           equal_nan=True)
        if must_change:
            assert not same, "an EARLIER game's credits must move a later row"
        else:
            assert same, (f"corrupting game {target_game} moved the features of "
                          f"game {mid} -- the row can see its own or a future game")


def test_a_prior_season_columns_come_from_a_completed_season(pops, positions):
    """The prior-season block is zero on the first season present and non-zero on
    the second, and `has_prior_season` says so -- the L13 identification fact the
    trainer drops features on."""
    p24 = {k: v.assign(season=2024) for k, v in pops.items()}
    a = A.build_player_asof({2024: p24, 2025: pops}, positions=positions)
    assert float(a.loc[a.season == 2024, "has_prior_season"].max()) == 0.0
    assert float(a.loc[a.season == 2025, "has_prior_season"].max()) == 1.0
    assert float(a.loc[a.season == 2024, "prev_opp_oreb"].max()) == 0.0
    assert float(a.loc[a.season == 2025, "prev_opp_oreb"].max()) > 0.0


# ===========================================================================
# B. Eligible-set correctness
# ===========================================================================
def test_b_the_shooter_is_never_an_assist_candidate(pops):
    d = A.usable(pops["made_fga"], "assist")
    assert len(d) > 100
    cand = d[[f"cand_{k}" for k in range(1, 5)]].to_numpy(dtype="int64")
    shooter = d["off_player_id"].to_numpy(dtype="float64")
    assert not (cand == shooter[:, None]).any(), "the shooter is in the choice set"
    assert cand.shape[1] == A.N_ALT_OF["assist"] == 4


def test_b_the_assist_choice_set_is_the_five_minus_the_shooter(pops):
    full = pops["made_fga"]
    d = A.usable(full, "assist")
    key = full.set_index(["game_id", "cand_team_id"])
    for _, row in d.head(50).iterrows():
        five = set(int(x) for x in
                   key.loc[(row.game_id, row.cand_team_id),
                           [f"cand_{k}" for k in range(1, 6)]].to_numpy().ravel()[:5])
        four = {int(row[f"cand_{k}"]) for k in range(1, 5)}
        assert four.issubset(five)
        assert int(row.off_player_id) not in four


def test_b_the_credited_player_is_always_inside_the_candidate_set(pops):
    for target in A.CHOICE_TARGETS:
        d = A.usable(pops[A.POP_OF[target]], target)
        k = A.N_ALT_OF[target]
        cand = d[[f"cand_{j}" for j in range(1, k + 1)]].to_numpy(dtype="int64")
        y = d["y"].to_numpy()
        assert (y >= 0).all()
        assert (cand[np.arange(len(d)), y]
                == d["credited_id"].to_numpy().astype("int64")).all(), target


def test_b_a_duplicated_or_missing_on_floor_id_disqualifies_the_choice_set():
    five = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],      # fine
                     [1.0, 1.0, 3.0, 4.0, 5.0],      # duplicate
                     [1.0, 2.0, 3.0, 4.0, np.nan]])  # missing
    _, ok = A._sorted_candidates(five, None, 5)
    assert list(ok) == [True, False, False]


def test_b_removing_a_shooter_who_is_not_on_the_floor_disqualifies_the_row():
    five = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
    drop = np.array([3.0, 99.0])          # the second shooter is not in the five
    cand, ok = A._sorted_candidates(five, drop, 4)
    assert list(ok) == [True, False]
    assert 3.0 not in cand[0]


def test_b_the_rebound_targets_use_opposite_sides_of_the_ball(pops):
    """`REB_off` credits the team with the ball and `REB_def` the other one; the
    two population masks are disjoint in the real builder, and the candidate team
    column is what the game-level check groups on."""
    assert A.SIDE_OF["REB_off"] == "off"
    assert A.SIDE_OF["REB_def"] == "def"
    assert A.SIDE_OF["steal"] == A.SIDE_OF["block"] == "def"
    assert A.SIDE_OF["assist"] == "off"


# ===========================================================================
# C. Determinism and the RNG contract
# ===========================================================================
RATES = {"REB_off": {1: 0.5, 2: 0.2, 3: 0.1, 4: 0.1, 5: 0.1},
         "REB_def": {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.5},
         "assist": {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.05, 5: 0.05},
         "steal": {1: 0.6, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1},
         "block": {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.6}}


def _run(seed: int, game_id: int, n: int = 50) -> list:
    st = A.new_game_state(game_id, seed, RATES)
    out = []
    for _ in range(n):
        out.append(A.draw_rebounder([1, 2, 3, 4, 5], st, offensive=True))
        out.append(A.draw_assist([1, 2, 3, 4, 5], 3, st))
        out.append(A.draw_steal([1, 2, 3, 4, 5], st))
        out.append(A.draw_block([1, 2, 3, 4, 5], st))
    return out


def test_c_the_same_seed_and_game_repeat_exactly():
    assert _run(11, 404) == _run(11, 404)


def test_c_a_different_seed_or_game_differs():
    assert _run(11, 404) != _run(12, 404)
    assert _run(11, 404) != _run(11, 405)


def test_c_each_credit_has_its_own_counter_and_family():
    st = A.new_game_state(404, 11, RATES)
    assert set(st.counters) == {"attr_rebound", "attr_assist", "attr_steal",
                                "attr_block"}
    A.draw_rebounder([1, 2, 3, 4, 5], st, offensive=True)
    assert st.counters["attr_rebound"] == 1
    assert st.counters["attr_steal"] == 0
    A.draw_steal([1, 2, 3, 4, 5], st)
    assert st.counters["attr_steal"] == 1
    assert st.counters["attr_rebound"] == 1


def test_c_drawing_one_credit_does_not_move_another_credits_sequence():
    """The whole point of one family per credit: an engine that stops drawing
    blocks must not change which player gets the steals."""
    st1 = A.new_game_state(404, 11, RATES)
    with_blocks = [(A.draw_block([1, 2, 3, 4, 5], st1),
                    A.draw_steal([1, 2, 3, 4, 5], st1)) for _ in range(30)]
    st2 = A.new_game_state(404, 11, RATES)
    without = [A.draw_steal([1, 2, 3, 4, 5], st2) for _ in range(30)]
    assert [s for _, s in with_blocks] == without


def test_c_the_binary_short_circuits_and_consumes_one_uniform():
    st = A.new_game_state(404, 11, RATES)
    outs = [A.draw_steal([1, 2, 3, 4, 5], st, p_steal=0.0) for _ in range(10)]
    assert outs == [None] * 10
    assert st.counters["attr_steal"] == 10        # one uniform per non-event
    st2 = A.new_game_state(404, 11, RATES)
    outs2 = [A.draw_steal([1, 2, 3, 4, 5], st2, p_steal=1.0) for _ in range(10)]
    assert all(o is not None for o in outs2)
    assert st2.counters["attr_steal"] == 20       # binary + choice


def test_c_empirical_frequencies_match_the_rate_table():
    st = A.new_game_state(909, 3, RATES)
    picks = [A.draw_steal([1, 2, 3, 4, 5], st) for _ in range(40000)]
    freq = pd.Series(picks).value_counts(normalize=True)
    for pid, share in RATES["steal"].items():
        assert freq[pid] == pytest.approx(share, abs=0.01)


def test_c_an_all_zero_rate_table_falls_back_to_the_uniform_not_the_first_id():
    st = A.new_game_state(1, 1, {"block": {}})
    picks = [A.draw_block([7, 8, 9, 10, 11], st) for _ in range(20000)]
    freq = pd.Series(picks).value_counts(normalize=True)
    assert set(freq.index) == {7, 8, 9, 10, 11}
    for v in freq.to_numpy():
        assert v == pytest.approx(0.2, abs=0.015)


def test_c_a_lineup_that_is_not_five_raises():
    st = A.new_game_state(1, 1, RATES)
    with pytest.raises(ValueError):
        A.draw_rebounder([1, 2, 3], st, offensive=True)
    with pytest.raises(ValueError):
        A.draw_assist([1, 2, 3, 4, 5], 99, st)


def test_c_event_stream_keys_are_distinct_within_a_game_and_per_family():
    d = pd.DataFrame({"game_id": [1] * 6 + [2] * 4})
    k1 = A.event_stream_keys(d, 5, "attr_steal")
    k2 = A.event_stream_keys(d, 5, "attr_block")
    assert len(set(k1.tolist())) == len(d)
    assert not (k1 == k2).any()


# ===========================================================================
# D. The ports
# ===========================================================================
def test_d_game_level_port_matches_usage_on_five_alternatives(pops):
    """`choice_game_level_check` is `usage.game_level_check` generalised over K
    and over the column names; on a five-alternative frame the two must agree to
    the last digit, or a number quoted next to a usage number is a lookalike."""
    d = A.usable(pops["reb_off"], "REB_off")
    n = len(d)
    p = np.full((n, 5), 0.2)
    mine = A.choice_game_level_check(d, p, "REB_off", n_draw=6, seed=4, min_games=2)
    u = d.rename(columns={f"cand_{k}": f"alt_{k}" for k in range(1, 6)}).rename(
        columns={"cand_team_id": "team_id"})
    # `usage` keys its RNG on the `usage_alloc` family; drive both off the same
    # keys by comparing the statistics that do not depend on the stream, plus the
    # ACTUAL-sequence statistics, which are stream-free by construction.
    theirs = U.game_level_check(u, p=p, n_draw=6, seed=4, min_games=2)
    assert mine["n_team_games"] == theirs["n_team_games"]
    assert mine["n_players_graded"] == theirs["n_players_graded"]
    assert mine["sd_actual"] == theirs["sd_actual"]
    assert mine["players_gt0_actual"] == theirs["players_gt0_actual"]
    assert mine["top1_actual_pct"] == theirs["top1_actual_pct"]
    assert mine["top3_actual_pct"] == theirs["top3_actual_pct"]


def test_d_grouped_softmax_objective_matches_usage_at_five(pops):
    rng = np.random.default_rng(0)
    z = rng.normal(size=40 * 5)
    y = np.zeros(40 * 5)
    y[np.arange(40) * 5 + rng.integers(0, 5, 40)] = 1.0
    g1, h1 = A.group_softmax_objective(5)(y, z)
    g2, h2 = U._group_softmax_objective(y, z)
    assert np.allclose(g1, g2)
    assert np.allclose(h1, h2)


def test_d_the_objective_gradient_is_the_softmax_residual_at_four(pops):
    rng = np.random.default_rng(1)
    z = rng.normal(size=30 * 4)
    y = np.zeros(30 * 4)
    y[np.arange(30) * 4 + rng.integers(0, 4, 30)] = 1.0
    g, h = A.group_softmax_objective(4)(y, z)
    p = np.exp(z.reshape(-1, 4))
    p = p / p.sum(axis=1, keepdims=True)
    assert np.allclose(g, (p - y.reshape(-1, 4)).reshape(-1))
    assert np.allclose(h, np.maximum(p * (1 - p), 1e-6).reshape(-1))
    # every choice set's gradient sums to zero: the softmax is normalised
    assert np.allclose(g.reshape(-1, 4).sum(axis=1), 0.0, atol=1e-12)


def test_d_the_tree_arm_nests_p1_at_zero_trees(pops, asof):
    """With the `log q` offset and no trees the arm IS P1, which is what makes the
    comparison "does a tree add anything to the as-of share" rather than "can a
    tree relearn a normalisation"."""
    d = A.build_choice_design(A.usable(pops["reb_off"], "REB_off"), asof, "REB_off")
    X, chosen = A.long_frame(d, "REB_off", "league", 50.0)
    init = A.lgbm_init_score(d, "REB_off", "league", 50.0)
    # LightGBM refuses zero boosting rounds, so the "zero trees" state is one
    # tree with a vanishing learning rate: the tree's contribution is ~0 and what
    # is left is the offset alone.
    arm = A.LgbmChoiceArm(5, dict(n_estimators=1, learning_rate=1e-12), seed=0)
    arm.fit(X, chosen, init=init)
    p_tree = arm.predict_proba(X, init=init)
    p1 = A.p1_probs(d, "REB_off", "league", 50.0)
    assert np.allclose(p_tree, p1, atol=1e-8)


# ===========================================================================
# E. Gate plumbing
# ===========================================================================
def test_e_decision8_exempts_a_sub_two_pp_driver_from_both_clauses():
    flat = {"span_actual": 0.004, "slope_ratio": 0.01, "pred_monotone_steps": 1}
    out = A.decision8_verdict(dict(flat))
    assert out["small_span_exempt"] is True
    assert out["pass"] is True          # exempt from BOTH the band and the steps
    wide = {"span_actual": 0.10, "slope_ratio": 0.01, "pred_monotone_steps": 4}
    out2 = A.decision8_verdict(dict(wide))
    assert out2["small_span_exempt"] is False
    assert out2["slope_pass"] is False
    assert out2["pass"] is False        # a flat arm over a real span fails
    ok = {"span_actual": 0.10, "slope_ratio": 1.0, "pred_monotone_steps": 3}
    out3 = A.decision8_verdict(dict(ok))
    assert out3["pass"] is True and out3["strict_4of4_pass"] is False


def test_e_shrinkage_is_fitted_and_the_prior_season_rung_is_unidentified(pops, asof):
    d = A.build_choice_design(A.usable(pops["reb_off"], "REB_off"), asof, "REB_off")
    fit = A.fit_shrinkage(d, "REB_off")
    assert fit["prior_season_available"] is False
    assert fit["best"]["prior"] in ("league", "position")
    unid = [r for r in fit["grid"] if r["prior"] == "prior_season"]
    assert len(unid) == 1 and unid[0]["m"] is None
    assert A.unidentified_features(d, "REB_off") == list(A.PRIOR_SEASON_FEATURES)


def test_e_p1_beats_the_uniform_on_a_skewed_truth(pops, asof):
    """The fixture gives the biggest id half the boards, so an as-of allocator
    must beat the uniform over five -- a sanity check that the rate is wired to
    the right player and not to a slot index."""
    from cbb_sim.models import prob_metrics as PM
    d = A.build_choice_design(A.usable(pops["reb_off"], "REB_off"), asof, "REB_off")
    late = d[pd.to_datetime(d.game_date) > pd.Timestamp("2024-11-10")]
    fit = A.fit_shrinkage(d, "REB_off")
    p = A.p1_probs(late, "REB_off", fit["best"]["prior"], fit["best"]["m"])
    assert PM.log_loss(late["y"].to_numpy(), p) < A.UNIFORM_LL["REB_off"]


def test_e_normalise_falls_back_to_the_uniform_on_an_empty_row():
    r = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0, 0.0]])
    p = A.normalise(r)
    assert np.allclose(p[0], 0.25)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_e_seal_guard_fires_on_season_2026(pops, asof):
    from cbb_sim.data.seal import SealedSeasonError
    d = A.build_choice_design(A.usable(pops["reb_off"], "REB_off"), asof, "REB_off")
    d2026 = d.assign(season=2026)
    with pytest.raises(SealedSeasonError):
        A.walkforward_slices(d2026, season=2026)
