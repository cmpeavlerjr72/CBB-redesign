"""Unit gates for the L4 SHOT ALLOCATION (usage) layer.

Design and numbers: `docs/models/usage/model.md`, `docs/models/usage/experiments.md`.
Trainer: `scripts/train_usage_v1.py`.

Four things are proved here, in the order the pre-registration asks for them:

  MEAN PRESERVATION  the hierarchical arm's realised shares have expectation
                     exactly equal to the as-of weights, at both levels, in BOTH
                     sampler paths (the exact counter-based one the engine uses
                     and the batched one the trainer uses). This is the single
                     design constraint the CFB allocator exists to satisfy
                     (`cfb-props-sim/src/cfb_props_sim/sim/usage_alloc.py`), and
                     the constraint a naive varying-alpha Dirichlet violates.
  LEAK SAFETY        every as-of feature is proved strictly-earlier two ways: an
                     independent recomputation, and invariance to corrupting the
                     row's own game while a LATER game does move. Same two-way
                     shape as `tests/test_rebound.py` and
                     `tests/test_free_throw.py`.
  DETERMINISM        the sampler is a pure function of (seed, game_id) and its
                     per-event draws are distinct, which is the defect that made
                     the first game-level measurement read 3.42 players with a
                     three-point attempt per team-game against a real 6.69.
  THE DECISION RULE  an arm with the lowest log loss that fails the game-level
                     SD check is NOT eligible -- the CFB "too narrow" clause.

Single process, no network, no model training. Run:
    .venv/Scripts/python.exe -m pytest tests/test_usage.py -q -s
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import SealedSeasonError  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.models import usage as U  # noqa: E402

ARTIFACTS = ROOT / "data/processed/models/usage"


# ===========================================================================
# Toy fixtures
# ===========================================================================
def toy_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A nine-man profile with two handlers, five wings and two bigs."""
    players = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109])
    weights = np.array([0.26, 0.19, 0.14, 0.11, 0.09, 0.07, 0.06, 0.05, 0.03])
    roles = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2], dtype="int8")
    return players, weights / weights.sum(), roles


def toy_profile_set(players, weights, roles) -> U.ProfileSet:
    return U.ProfileSet(players=players, weights=weights, roles=roles,
                        offsets=np.array([0, len(weights)]),
                        codes=np.zeros(1, dtype="int64"),
                        gslot=np.arange(5)[None, :].astype("int64"),
                        game_ids=np.array([401_500_001], dtype="int64"))


def toy_events(n_games: int = 10, seed: int = 7) -> pd.DataFrame:
    """A synthetic credited-event table with the exact schema
    `build_player_asof` and `build_usage_design` consume.

    Two teams, six players each (so the five on the floor genuinely rotates),
    two seasons of `n_games` games, with a deliberately lopsided usage pattern
    so a leak would show up as a feature that knows its own game."""
    rng = np.random.default_rng(seed)
    rows = []
    rosters = {10: [1, 2, 3, 4, 5, 6], 20: [11, 12, 13, 14, 15, 16]}
    weight = {p: w for p, w in zip([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], strict=False)}
    weight.update({p: w for p, w in zip([11, 12, 13, 14, 15, 16], [1, 2, 3, 4, 5, 6],
                                        strict=False)})
    gid = 401_000_000
    for season in (2024, 2025):
        for g in range(n_games):
            gid += 1
            date = pd.Timestamp(f"{season - 1}-11-10") + pd.Timedelta(days=3 * g)
            for team, opp in ((10, 20), (20, 10)):
                roster = rosters[team]
                for ev in range(14):
                    five = sorted(rng.choice(roster, 5, replace=False).tolist())
                    w = np.array([weight[p] for p in five], dtype="float64")
                    credited = int(rng.choice(five, p=w / w.sum()))
                    rows.append({
                        "game_id": gid, "cbbd_game_id": gid, "season": season,
                        "game_date": date, "team_id": team, "opp_id": opp,
                        "offense_is_home": team == 10, "neutral_site": False,
                        "event_class": U.EVENT_CLASSES[ev % len(U.EVENT_CLASSES)],
                        "player_id": credited, "period": 1 + ev // 7,
                        "sec_in_period": 1200 - 80 * (ev % 7),
                        "sec_remaining": 2400 - 170 * ev,
                        "score_diff": ev - 7, "chance_number": 1 + (ev % 3 == 0),
                        **{f"alt_{k + 1}": five[k] for k in range(5)},
                        "five_ok": True, "in_five": True,
                        "y": five.index(credited),
                    })
    d = pd.DataFrame(rows)
    d["y"] = d["y"].astype("int8")
    return d


def toy_positions() -> pd.DataFrame:
    ids = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
    grp = ["G", "G", "F", "F", "C", "C"] * 2
    return pd.DataFrame({"shooter_id": ids, "position_group": grp})


@pytest.fixture(scope="module")
def toy():
    ev = toy_events()
    by_season = {s: ev[ev.season == s].reset_index(drop=True) for s in (2024, 2025)}
    asof = U.build_player_asof(by_season, minutes=None, positions=toy_positions())
    return ev, by_season, asof


# ===========================================================================
# (a) MEAN PRESERVATION -- the whole point of the hierarchical form
# ===========================================================================
@pytest.mark.parametrize("alphas", [
    U.hier_alphas(30.0, 6.0, 9.0, 4.0),
    U.hier_alphas(8.0, 3.0, 3.0, 3.0),
    U.single_level_alphas(20.0),
])
def test_a_mean_preserving_exact_path(alphas):
    """`realize_shares` (the engine path) has E[share] = weight at every level.

    A Dirichlet with concentration `a_i = alpha_i * w_i` has mean
    `a_i / sum_j a_j`, so a VARYING alpha silently inflates whoever carries the
    larger one -- measured in CFB as a top-1 share 1.04-1.13x real, "an outright
    fail". The two-level form is centred on its own conditional means at both
    levels, which is what this asserts."""
    players, w, roles = toy_profile()
    if "a_between" in alphas and not np.isfinite(alphas["a_between"]):
        roles = np.zeros_like(roles)          # U2 is the one-role case
    R = 60_000
    u = np.random.default_rng(20260910).random((R, len(w) + U.N_ROLE))
    S = U.realize_shares(w, roles, alphas, u)
    err = np.abs(S.mean(axis=0) - w)
    se = S.std(axis=0) / np.sqrt(R)
    print(f"\nmean preservation, exact path, alphas={alphas}")
    print(f"{'player':>8} {'weight':>9} {'mean':>9} {'|diff|':>9} {'3 x SE':>9}")
    for p, wi, mi, ei, si in zip(players, w, S.mean(axis=0), err, se, strict=False):
        print(f"{p:>8} {wi:9.5f} {mi:9.5f} {ei:9.5f} {3 * si:9.5f}")
    assert (err <= np.maximum(4 * se, 1e-4)).all(), "the Dirichlet is not mean-preserving"
    assert np.isclose(S.sum(axis=1), 1.0).all()


def test_a_mean_preserving_batch_path_and_agreement():
    """`realize_batch` (the trainer path) is the same distribution.

    The batched sampler is what makes a season-wide Monte Carlo affordable, so
    it has to be pinned to the shipped per-call one -- the same batched-twin
    discipline `cfb-props-sim/tests/test_usage_alloc.py::test_a_batch_matches_
    realize_shares` applies."""
    players, w, roles = toy_profile()
    alphas = U.hier_alphas(30.0, 6.0, 9.0, 4.0)
    R = 60_000
    u = np.random.default_rng(11).random((R, len(w) + U.N_ROLE))
    S = U.realize_shares(w, roles, alphas, u)
    B = U.realize_batch(toy_profile_set(players, w, roles), alphas, R, seed=13)
    print("\nbatched twin vs exact path")
    print(f"{'player':>8} {'weight':>9} {'exact':>9} {'batch':>9} "
          f"{'sd exact':>9} {'sd batch':>9}")
    for i, p in enumerate(players):
        print(f"{p:>8} {w[i]:9.5f} {S[:, i].mean():9.5f} {B[:, i].mean():9.5f} "
              f"{S[:, i].std():9.5f} {B[:, i].std():9.5f}")
    assert np.abs(B.mean(axis=0) - w).max() < 0.004
    assert np.abs(B.mean(axis=0) - S.mean(axis=0)).max() < 0.005
    assert np.abs(B.std(axis=0) - S.std(axis=0)).max() < 0.005


def test_a_single_level_matches_the_analytic_dirichlet_sd():
    """U2's one-knob form really is `Dirichlet(a * w)`: its marginal SD is the
    closed form `sqrt(w (1 - w) / (a + 1))`. This is what catches a role
    structure leaking into the arm that is supposed not to have one -- keeping
    the roles would pin a one-man role to zero variance."""
    players, w, roles = toy_profile()
    a = 25.0
    ps = toy_profile_set(players, w, roles).single_role()
    B = U.realize_batch(ps, U.single_level_alphas(a), 60_000, seed=5)
    want = np.sqrt(w * (1 - w) / (a + 1))
    print("\nsingle-level SD vs the analytic Dirichlet")
    for p, wi, got, exp in zip(players, w, B.std(axis=0), want, strict=False):
        print(f"{p:>8} w={wi:7.4f} sd={got:8.5f} analytic={exp:8.5f}")
    assert np.abs(B.std(axis=0) - want).max() < 0.004


def test_a_infinite_alphas_are_a_noop():
    """Every knob at infinity returns the as-of weights unchanged, in both
    paths: the Dirichlet arms nest U1 exactly, which is what lets the fitted
    concentration say "no dispersion" instead of the arm being a different
    model."""
    players, w, roles = toy_profile()
    u = np.random.default_rng(3).random((7, len(w) + U.N_ROLE))
    S = U.realize_shares(w, roles, U.NO_DISPERSION, u)
    B = U.realize_batch(toy_profile_set(players, w, roles), U.NO_DISPERSION, 7, seed=1)
    assert np.allclose(S, w[None, :])
    assert np.allclose(B, w[None, :])


# ===========================================================================
# (b) LEAK SAFETY -- proved two ways
# ===========================================================================
def test_b_asof_features_are_an_independent_strictly_earlier_recomputation(toy):
    """Recompute the as-of numerator and denominator from scratch for a sample of
    (player, game) rows and require an exact match.

    "From scratch" means: take every event of that season whose game_date is
    strictly earlier than this row's game, count the ones where the player was
    one of the five (exposure) and the ones credited to him (numerator). No
    expanding sum, no groupby machinery from the module under test."""
    ev, by_season, asof = toy
    rng = np.random.default_rng(4)
    pick = asof.iloc[rng.choice(len(asof), 40, replace=False)]
    checked = 0
    for _, r in pick.iterrows():
        season, pid, date = int(r["season"]), int(r["player_id"]), r["game_date"]
        e = by_season[season]
        earlier = e[pd.to_datetime(e["game_date"]) < pd.Timestamp(date)]
        alts = earlier[[f"alt_{k}" for k in range(1, 6)]].to_numpy()
        on_floor = (alts == pid).any(axis=1)
        exposure = int(on_floor.sum())
        assert exposure == int(r["exposure_asof"]), (
            f"exposure leak for player {pid} in {season}: "
            f"{r['exposure_asof']} vs an independent {exposure}")
        for c in U.EVENT_CLASSES:
            want = int(((earlier["player_id"].to_numpy() == pid)
                        & (earlier["event_class"].to_numpy() == c)).sum())
            assert want == int(r[f"ev_{c}"]), f"{c} numerator leak for {pid}"
        assert int(r["ev_total"]) == int((earlier["player_id"].to_numpy() == pid).sum())
        checked += 1
    print(f"\n(b) recomputed {checked} (player, game) as-of rows independently")
    assert checked == 40


def test_b_asof_features_cannot_see_their_own_game_but_do_see_an_earlier_one(toy):
    """Corrupt a game's own events and require every as-of feature of that game
    to be bit-identical, while a LATER game's features DO move.

    The second half is what makes the first half meaningful: a feature that
    never moves is not leak-safe, it is broken."""
    ev, by_season, asof = toy
    season = 2025
    games = sorted(by_season[season]["game_id"].unique())
    target = games[len(games) // 2]
    later = games[-1]

    corrupt = {s: d.copy() for s, d in by_season.items()}
    c = corrupt[season]
    msk = c["game_id"].to_numpy() == target
    # credit every one of the target game's events to its FIRST alternative
    c.loc[msk, "player_id"] = c.loc[msk, "alt_1"].to_numpy()
    c.loc[msk, "y"] = 0
    corrupted = U.build_player_asof(corrupt, minutes=None, positions=toy_positions())

    cols = ["exposure_asof", "ev_total", *[f"ev_{x}" for x in U.EVENT_CLASSES],
            *[f"rate_{x}" for x in U.EVENT_CLASSES]]
    key = ["season", "player_id", "game_id"]
    a = asof.set_index(key)[cols].sort_index()
    b = corrupted.set_index(key)[cols].sort_index()
    common = a.index.intersection(b.index)
    same_game = [i for i in common if i[0] == season and i[2] == target]
    later_game = [i for i in common if i[0] == season and i[2] == later]
    assert same_game and later_game

    d_same = (a.loc[same_game] - b.loc[same_game]).abs().to_numpy().max()
    d_later = (a.loc[later_game] - b.loc[later_game]).abs().to_numpy().max()
    print(f"\n(b) corrupting game {target}: max |delta| on its OWN as-of rows "
          f"{d_same:.10f}; on a LATER game's rows {d_later:.6f}")
    assert d_same == 0.0, "an as-of feature moved when its own game was corrupted"
    assert d_later > 0.0, "a LATER game's as-of features did not move -- the " \
                          "invariance test above proves nothing"


def test_b_prior_season_is_a_completed_season(toy):
    """The prior-season block is absent for the first season present and
    populated for the second -- i.e. it can only ever read a finished season.
    This is also the L13 asymmetry the trainer has to drop features for."""
    _, _, asof = toy
    by = asof.groupby("season")["has_prior_season"].mean()
    print(f"\n(b) share of player-games with a prior season: {by.to_dict()}")
    assert by.loc[2024] == 0.0
    assert by.loc[2025] > 0.5


# ===========================================================================
# (c) DETERMINISM of the sampler
# ===========================================================================
def _state(game_id, seed, alphas=None):
    players, w, roles = toy_profile()
    prof = {c: U.Profile(players=players, weights=w, roles=roles)
            for c in U.EVENT_CLASSES}
    return U.new_game_state(game_id, seed, profiles=prof, alphas=alphas)


def test_c_sampler_is_a_pure_function_of_seed_and_game():
    five = [101, 102, 103, 108, 109]
    al = U.hier_alphas(30.0, 6.0, 9.0, 4.0)
    seq = lambda st: [U.draw_player(five, "FGA_3", st) for _ in range(25)]  # noqa: E731
    a = seq(_state(401_500_001, 7, al))
    b = seq(_state(401_500_001, 7, al))
    c = seq(_state(401_500_001, 8, al))
    d = seq(_state(401_500_002, 7, al))
    print(f"\n(c) same (game, seed) draws: {a[:10]}")
    assert a == b, "same (game, seed) must repeat exactly"
    assert a != c, "a different seed must give a different stream"
    assert a != d, "a different game must give a different stream"


def test_c_consecutive_draws_are_independent_not_one_uniform_reused():
    """Each call consumes its own stream position.

    This is the bug the game-level measurement caught: keying the allocation
    uniform on the GAME alone handed every event of a team-game the same
    uniform, which collapsed the whole game onto one player (3.42 players with
    a three-point attempt per team-game against a real 6.69)."""
    five = [101, 102, 103, 108, 109]
    st = _state(401_500_003, 5, U.hier_alphas(30.0, 6.0, 9.0, 4.0))
    draws = [U.draw_player(five, "FGA_rim", st) for _ in range(400)]
    assert st.counter > 400, "the stream counter did not advance per draw"
    assert len(set(draws)) >= 4, f"400 draws produced only {set(draws)}"


def test_c_event_stream_keys_are_distinct_within_a_game():
    d = pd.DataFrame({"game_id": [401_000_001] * 6 + [401_000_002] * 4})
    k = U.event_stream_keys(d, seed=3)
    assert len(set(k.tolist())) == len(d), "two events of one game share a key"
    assert len(set(U.event_stream_keys(d, seed=4).tolist()) & set(k.tolist())) == 0


def test_c_draw_player_reproduces_the_no_dispersion_shares():
    """With no dispersion the sampler's empirical frequencies are U1's shares
    over the five -- the sampler and the scored arm are the same model."""
    players, w, roles = toy_profile()
    prof = {"FGA_3": U.Profile(players=players, weights=w, roles=roles)}
    five = [101, 103, 105, 108, 109]
    want = np.array([w[list(players).index(p)] for p in five])
    want = want / want.sum()
    counts = np.zeros(5)
    for g in range(4000):
        st = U.new_game_state(401_600_000 + g, 1, profiles=prof, alphas=None)
        counts[five.index(U.draw_player(five, "FGA_3", st))] += 1
    got = counts / counts.sum()
    print(f"\n(c) sampler frequencies {np.round(got, 4)} vs U1 shares {np.round(want, 4)}")
    assert np.abs(got - want).max() < 0.02


def test_c_a_lineup_with_no_history_falls_back_to_the_uniform():
    st = U.new_game_state(401_700_001, 1, rates={"TOV": {}})
    counts = {}
    for i in range(5000):
        st.counter = i
        counts[U.draw_player([7, 8, 9, 10, 11], "TOV", st)] = \
            counts.get(U.draw_player([7, 8, 9, 10, 11], "TOV", st), 0) + 1
    assert set(counts) == {7, 8, 9, 10, 11}, "the uniform fallback is not uniform"


def test_c_draw_player_requires_exactly_five():
    st = U.new_game_state(1, 1, rates={"TOV": {1: 1.0}})
    with pytest.raises(ValueError):
        U.draw_player([1, 2, 3], "TOV", st)


# ===========================================================================
# (d) The target, the design and the seal
# ===========================================================================
def test_d_the_five_is_sorted_and_y_points_at_the_credited_player(toy):
    ev, _, _ = toy
    alts = ev[[f"alt_{k}" for k in range(1, 6)]].to_numpy()
    assert (np.diff(alts, axis=1) > 0).all(), "the five is not sorted ascending"
    assert (alts[np.arange(len(ev)), ev["y"].to_numpy()]
            == ev["player_id"].to_numpy()).all()


def test_d_design_carries_every_alternative_and_normalises_to_one(toy):
    ev, _, asof = toy
    d = U.build_usage_design(ev, asof, "FGA_3")
    assert len(d) == int((ev["event_class"] == "FGA_3").sum())
    p = U.u1_probs(d, "FGA_3", "position", 50.0)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()
    assert not np.isnan(U.shrunk_rate(d, "FGA_3", "league", 50.0)).any()


def test_d_roles_put_a_centre_in_the_big_family_and_rank_the_rest(toy):
    ev, _, asof = toy
    d = U.build_usage_design(ev, asof, "FGA_rim")
    roles = U.assign_roles(d, "position", 50.0)
    pos = d[[f"position_code_{k}" for k in range(1, 6)]].to_numpy()
    is_c = pos == U.POSITION_LEVELS.index("C")
    assert (roles[is_c] == U.ROLES.index("big")).all(), "a centre is not a big"
    n_handler = (roles == U.ROLES.index("handler")).sum(axis=1)
    assert n_handler.max() <= U.HANDLER_RANK
    print(f"\n(d) role mix per lineup: handlers {n_handler.mean():.2f}, "
          f"wings {(roles == 1).sum(axis=1).mean():.2f}, "
          f"bigs {(roles == 2).sum(axis=1).mean():.2f}")


def test_d_the_prior_season_block_is_reported_unidentified_when_absent(toy):
    ev, _, asof = toy
    d = U.build_usage_design(ev, asof, "FGA_3")
    tr = d[d.season == 2024]
    te = d[d.season == 2025]
    assert not U.prior_season_available(tr)
    assert U.prior_season_available(te)
    assert set(U.unidentified_features(tr)) == set(U.PRIOR_SEASON_FEATURES)
    assert U.unidentified_features(te) == []
    # the shrinkage grid must not pick a prior it cannot identify
    fit = U.fit_shrinkage(tr, "FGA_3")
    assert fit["best"]["prior"] != "prior_season"
    assert any(r["status"].startswith("unidentified") for r in fit["grid"]
               if r["prior"] == "prior_season")


def test_d_the_2026_season_is_sealed(toy):
    """Every path that could put season 2026 into a fit or a selection raises.

    Two of them raise -- a fold whose season list names 2026 and a walk-forward
    call pointed at 2026 -- and the third is shown to be structurally impossible:
    a design that CARRIES 2026 rows still yields slices with none in them,
    because the fold selects on its own season list."""
    ev, _, asof = toy
    d = U.build_usage_design(ev, asof, "FGA_3")
    U.FOLDS["_seal_probe"] = {"train": [2024], "test": [2026]}
    try:
        with pytest.raises(SealedSeasonError):
            U.fold_slices(d, "_seal_probe")
    finally:
        del U.FOLDS["_seal_probe"]
    with pytest.raises(SealedSeasonError):
        U.walkforward_slices(d, season=2026)
    smuggled = d.assign(season=np.where(np.arange(len(d)) < 5, 2026,
                                        d["season"].to_numpy()))
    tr, te = U.fold_slices(smuggled)
    assert not (tr["season"] == 2026).any() and not (te["season"] == 2026).any()
    wtr, wte = U.walkforward_slices(smuggled)
    assert not (wtr["season"] == 2026).any() and not (wte["season"] == 2026).any()


def test_d_the_thresholds_are_arguments_so_a_role_rule_change_is_representable():
    """`HANDLER_RANK` is a module constant, not a literal buried in the role
    rule: a different definition of "primary handler" has to be expressible
    without editing the allocator."""
    import inspect
    src = inspect.getsource(U.assign_roles)
    assert "HANDLER_RANK" in src
    assert "rank < 2" not in src


def test_d_the_tree_objective_is_the_grouped_softmax_likelihood():
    """U5's objective is the conditional likelihood of the choice among five, not
    a per-row binary one.

    Two properties pin it: the gradient sums to zero inside every group of five
    (it is `p - y` on a simplex), and feeding it `log q` recovers `q` exactly --
    which is what makes `lgbm_init_score` nest U1 at zero trees. The binary
    objective this replaced came out systematically over-sharpened (F1 TOV top
    decile 30.13% predicted against 27.89% real, worst gap 2.24 pp, a
    calibration FAIL on an arm whose log loss looked good)."""
    q = np.array([[0.40, 0.25, 0.20, 0.10, 0.05],
                  [0.21, 0.20, 0.20, 0.20, 0.19]])
    y = np.zeros_like(q)
    y[0, 2] = 1.0
    y[1, 0] = 1.0
    grad, hess = U._group_softmax_objective(y.reshape(-1), np.log(q).reshape(-1))
    g = grad.reshape(-1, U.N_ALT)
    print("")
    print("(d) grouped-softmax gradient")
    print(np.round(g, 5))
    assert np.abs(g.sum(axis=1)).max() < 1e-12, "the gradient leaves the simplex"
    assert np.abs((g + y) - q).max() < 1e-12, "log q must map back to q"
    assert (hess > 0).all()


def test_d_the_responsiveness_gate_rejects_a_flat_arm(toy):
    """Decision 8 (`ARCHITECTURE_DECISIONS.md`): the responsiveness gate needs a
    slope ratio in [0.8, 1.2] as well as monotone steps.

    A flat arm -- every choice set predicted uniform -- is monotone in zero steps
    and has a slope ratio of 0, so it fails both halves. An arm that is perfectly
    monotone but holds only a tenth of the realised spread passes the steps half
    and must still FAIL: that is the fg_make failure (shooter slope 0.0086) the
    amendment exists to prevent."""
    ev, _, asof = toy
    d = U.build_usage_design(ev, asof, "FGA_rim")
    drv = U.shrunk_rate(d, "FGA_rim", "position", 50.0)
    real = U.u1_probs(d, "FGA_rim", "position", 50.0)

    flat = np.full_like(real, 1.0 / U.N_ALT)
    rf = U.share_responsiveness(d, flat, drv)
    assert not rf["slope_pass"] and not rf["pass"]

    # 10% of the real spread around the uniform: monotone everywhere, nearly flat
    damped = U.normalise(1.0 / U.N_ALT + 0.1 * (real - 1.0 / U.N_ALT))
    rd = U.share_responsiveness(d, damped, drv)
    rr = U.share_responsiveness(d, real, drv)
    print("")
    print(f"(d) slope ratios -- flat {rf['slope_ratio']}, damped {rd['slope_ratio']}, "
          f"real {rr['slope_ratio']}; steps {rf['pred_monotone_steps']}/"
          f"{rd['pred_monotone_steps']}/{rr['pred_monotone_steps']}")
    assert rd["steps_pass"], "the damped arm is still monotone -- that is the point"
    assert not rd["slope_pass"], "a near-flat arm must fail the slope half"
    assert not rd["pass"]
    assert rr["slope_pass"] and rr["steps_pass"] and rr["pass"]
    # the superseded steps-only reading stays on record and would have passed it
    assert rd["steps_only_pass"]


def test_d_the_small_span_clause_relaxes_the_step_count():
    """The other half of Decision 8: when the driver's REALISED quintile span is
    under 2 pp the steps are noise, so 4 of 4 relaxes to 3 of 4. Asserted on the
    constants and on the branch, because no driver in this model is anywhere near
    that span (the smallest realised span is 11.95 pp) and the clause would
    otherwise never be exercised."""
    assert U.RESP_MIN_STEPS_SMALL_SPAN < U.RESP_MIN_STEPS
    n = 4000
    rng = np.random.default_rng(5)
    drv = rng.random((n, U.N_ALT))
    # a target whose quintile span is tiny: the chosen slot is almost independent
    # of the driver
    y = rng.integers(0, U.N_ALT, n)
    te = pd.DataFrame({"y": y.astype("int8")})
    p = U.normalise(1.0 / U.N_ALT + 0.0005 * (drv - drv.mean()))
    r = U.share_responsiveness(te, p, drv)
    print("")
    print(f"(d) tiny-span driver: realised span {r['span_actual_pp']} pp, "
          f"steps required {r['steps_required']}")
    assert r["span_actual_pp"] < U.SMALL_SPAN_PP
    assert r["steps_required"] == U.RESP_MIN_STEPS_SMALL_SPAN


# ===========================================================================
# (e) The game-level checks and the decision rule
# ===========================================================================
def test_e_game_level_check_is_reproducible_and_sane(toy):
    ev, _, asof = toy
    d = U.build_usage_design(ev, asof, "FGA_rim").sort_values(
        ["game_id", "team_id"]).reset_index(drop=True)
    p = U.u1_probs(d, "FGA_rim", "position", 50.0)
    a = U.game_level_check(d, p=p, n_draw=12, seed=2, min_games=3)
    b = U.game_level_check(d, p=p, n_draw=12, seed=2, min_games=3)
    c = U.game_level_check(d, p=p, n_draw=12, seed=3, min_games=3)
    assert a == b, "the same seed must reproduce the game-level check exactly"
    assert a["sd_ratio"] != c["sd_ratio"]
    assert 0.0 < a["players_gt0_actual"] <= 5.0
    assert 0.0 < a["top1_actual_pct"] <= 100.0
    print(f"\n(e) toy game-level check: sd_ratio {a['sd_ratio']}, "
          f"players>=1 {a['players_gt0_sim']} vs {a['players_gt0_actual']}")


def test_e_an_arm_that_fails_the_sd_check_is_not_eligible():
    """The CFB "too narrow" clause: the pre-registration makes the game-level SD
    ratio an ELIGIBILITY condition, so the best log loss does not win if its
    dispersion is wrong."""
    import train_usage_v1 as T

    def arm(ll, sd, calib=0.5, resp=4, gt0=0.1, top=True):
        return {"log_loss": ll, "calib_pass": calib <= 2.0, "resp_pass": resp >= 4,
                "calib_worst_gap_pp": calib, "resp_steps": resp,
                "bootstrap_se": 0.001,
                "game_level": {"sd_pass": 0.9 <= sd <= 1.1, "sd_ratio": sd,
                               "gt0_pass": abs(gt0) <= 0.5,
                               "players_gt0_delta": gt0, "top_pass": top}}
    res = {"arms": {"proportional": arm(1.50, 1.02),
                    "dirichlet": arm(1.40, 0.55),          # best ll, too narrow
                    "hier_dirichlet": arm(1.60, 1.00),
                    "cond_logit": arm(1.52, 1.01),
                    "lgbm": arm(1.45, 1.03, calib=3.0)}}   # best ll, bad calibration
    dec = T.decide(res)
    print(f"\n(e) decision: winner {dec['winner']}, eligible {dec['eligible']}")
    assert "dirichlet" not in dec["eligible"], "a too-narrow arm must be ineligible"
    assert "lgbm" not in dec["eligible"], "a mis-calibrated arm must be ineligible"
    assert dec["winner"] == "proportional"


def test_e_the_tree_must_clear_the_floor_to_win():
    import train_usage_v1 as T

    def arm(ll, se):
        return {"log_loss": ll, "calib_pass": True, "resp_pass": True,
                "calib_worst_gap_pp": 0.4, "resp_steps": 4, "bootstrap_se": se,
                "game_level": {"sd_pass": True, "sd_ratio": 1.0, "gt0_pass": True,
                               "players_gt0_delta": 0.0, "top_pass": True}}
    base = {"proportional": arm(1.5000, 0.002), "dirichlet": arm(1.6, 0.002),
            "hier_dirichlet": arm(1.6, 0.002), "cond_logit": arm(1.6, 0.002)}
    inside = T.decide({"arms": {**base, "lgbm": arm(1.4990, 0.002)}})
    outside = T.decide({"arms": {**base, "lgbm": arm(1.4900, 0.002)}})
    print(f"\n(e) lgbm inside the floor -> {inside['winner']}; "
          f"outside -> {outside['winner']}")
    assert inside["winner"] == "proportional"
    assert outside["winner"] == "lgbm"


def test_e_when_the_tree_misses_the_floor_the_rest_still_tie_break_on_simplicity():
    """Knocking the tree out must not crown whichever non-tree arm scored lowest
    by a hair -- the ordinary tie-break still applies to what is left.

    Caught on the real run: F1 `FGA_3`'s within-2025 fold had lgbm inside the
    floor and the first version of the rule handed the win to `hier_dirichlet`,
    which led `proportional` by 0.00009 against a floor of 0.00169."""
    import train_usage_v1 as T

    def arm(ll):
        return {"log_loss": ll, "calib_pass": True, "resp_pass": True,
                "calib_worst_gap_pp": 0.4, "resp_steps": 4, "bootstrap_se": 0.002,
                "game_level": {"sd_pass": True, "sd_ratio": 1.0, "gt0_pass": True,
                               "players_gt0_delta": 0.0, "top_pass": True}}
    dec = T.decide({"arms": {"proportional": arm(1.5010), "dirichlet": arm(1.5009),
                             "hier_dirichlet": arm(1.5008), "cond_logit": arm(1.5011),
                             "lgbm": arm(1.5000)}})
    print("")
    print(f"(e) tree inside the floor -> {dec['winner']} ({dec['reason']})")
    assert dec["winner"] == "proportional"
    assert "lgbm" in dec["eligible"]


def test_e_ties_go_to_the_simpler_arm():
    import train_usage_v1 as T

    def arm(ll):
        return {"log_loss": ll, "calib_pass": True, "resp_pass": True,
                "calib_worst_gap_pp": 0.4, "resp_steps": 4, "bootstrap_se": 0.01,
                "game_level": {"sd_pass": True, "sd_ratio": 1.0, "gt0_pass": True,
                               "players_gt0_delta": 0.0, "top_pass": True}}
    dec = T.decide({"arms": {"proportional": arm(1.5005), "dirichlet": arm(1.5002),
                             "hier_dirichlet": arm(1.5001), "cond_logit": arm(1.5000),
                             "lgbm": arm(1.6)}})
    assert dec["winner"] == "proportional", dec["reason"]


# ===========================================================================
# (f) The shipped artifacts reproduce what the report claims
# ===========================================================================
@pytest.mark.skipif(not (ARTIFACTS / "usage_params_v1.json").exists(),
                    reason="run scripts/train_usage_v1.py first")
def test_f_the_shipped_parameters_reproduce_the_reported_log_loss():
    """Rebuild each class's winning arm from `usage_params_v1.json` alone and
    require the F1 test log loss in `results_v1.json` back to 1e-6.

    This is the reproducibility clause of the documentation standard: the params
    file plus the cached design must be enough to regenerate the table."""
    import json
    params = json.loads((ARTIFACTS / "usage_params_v1.json").read_text())
    results = json.loads((ARTIFACTS / "results_v1.json").read_text())
    version = params["possessions_version"]
    events = pd.read_parquet(ARTIFACTS / f"events_{version}.parquet")
    asof = pd.read_parquet(ARTIFACTS / f"asof_{version}.parquet")
    checked = 0
    for cls, spec in params["per_class"].items():
        f1 = next(r for r in results if r["fold"] == "F1" and r["event_class"] == cls)
        d = U.build_usage_design(events, asof, cls)
        _, te = U.fold_slices(d)
        p = U.u1_probs(te, cls, spec["prior_kind"], spec["shrink_m"])
        got = PM.log_loss(te["y"].to_numpy(), p)
        want = f1["arms"]["proportional"]["log_loss"]
        print(f"\n(f) {cls}: proportional log loss {got:.6f} vs reported {want:.6f}")
        assert abs(got - want) < 1e-6
        checked += 1
    assert checked == len(params["per_class"])
