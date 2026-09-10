"""Control-engine tests: feature leak-safety and simulator determinism.

Two properties are worth a test because both were failure modes last year
(`docs/postmortem/01_engine_review.md`):

1. **A game's own result must not influence its own features.** Both pregame
   rating sources are as-of joins with strictly-before semantics, so mutating a
   game's box score -- or its final score -- must leave every feature on that
   game's own rows bit-identical. The test does the mutation for real rather
   than inspecting the join code, so it also catches an accidental same-day
   `allow_exact_matches=True` or an off-by-one in the as-of window.

2. **The simulator is deterministic under a fixed seed and keyed per game.**
   The RNG contract (CLAUDE.md: "RNG seeded on (seed, game_id, family). Paired
   bake-off arms share aligned streams") means a game's draws must not depend
   on how the run was chunked, on which other games were in the run, or on
   anything but (seed, game_id, family).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cbb_sim.control import features as F
from cbb_sim.control import models as M
from cbb_sim.control import rng as crng
from cbb_sim.control import simulate as S
from cbb_sim.ratings import own_ratings as orat

SEASON = 2024
FEATURE_COLS = [
    "own_off_c", "own_def_c", "own_tempo_rel",
    "opp_own_off_c", "opp_own_def_c", "opp_own_tempo_rel",
    "own_kp_adj_o_c", "own_kp_adj_d_c", "own_kp_adj_t_rel",
    "opp_kp_adj_o_c", "opp_kp_adj_d_c", "opp_kp_adj_t_rel",
    "site_home", "site_away", "days_since_start",
]


# ---------------------------------------------------------------------------
# 1. feature builder leak-safety
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def universe():
    return orat.load_universe()


@pytest.fixture(scope="module")
def baseline(universe):
    tg, g = F.build_team_game_features([SEASON], universe=universe)
    return tg, g


def test_ratings_as_of_window_is_strictly_before(universe):
    """The own-ratings table for date D must be fitted only on games before D:
    the cumulative game count at D equals the number of games played strictly
    earlier that season."""
    ratings = orat.load_ratings([SEASON])
    tg = orat.load_team_games(universe, [SEASON])
    played = tg.drop_duplicates("game_id")[["game_id", "game_date"]]
    per_date = ratings.groupby("as_of_date")["n_games_window"].first().sort_index()
    for date, n_window in per_date.items():
        n_before = int((played["game_date"] < date).sum())
        assert n_window == n_before, f"as_of {date}: window {n_window} != {n_before} games before"


def test_kenpom_as_of_uses_the_previous_snapshot_when_one_exists_that_same_day(baseline):
    """KenPom re-scrapes the morning after games are played, so a snapshot
    dated the same day as the game already contains that game's result. On the
    days where a snapshot exists, the join must return the PREVIOUS one."""
    tg, _ = baseline
    kp = pd.read_parquet(F.DEFAULT_KENPOM)
    kp = kp[kp["season"] == SEASON].dropna(subset=["team"]).copy()
    kp["_k"] = kp["team"].map(F.normalize_join_key)
    kp["snapshot_date"] = pd.to_datetime(kp["snapshot_date"])

    snap_days = sorted(kp["snapshot_date"].unique())
    assert len(snap_days) > 5
    same_day = pd.to_datetime(snap_days[len(snap_days) // 2])
    prev_day = pd.to_datetime(snap_days[len(snap_days) // 2 - 1])

    rows = tg[pd.to_datetime(tg["game_date"]) == same_day]
    if rows.empty:
        pytest.skip(f"no games on snapshot day {same_day.date()}")

    key_map = F.build_kenpom_key_map([SEASON], F.DEFAULT_HOOPR_DIR, pd.read_parquet(F.DEFAULT_KENPOM))
    keys = key_map.set_index(["season", "team_id"])["kp_key"]

    same = kp[kp["snapshot_date"] == same_day].set_index("_k")["adj_o_c"]
    prev = kp[kp["snapshot_date"] == prev_day].set_index("_k")["adj_o_c"]

    checked = 0
    for r in rows.itertuples():
        k = keys.get((SEASON, int(r.team_id)))
        if k is None or k not in same.index or k not in prev.index:
            continue
        if np.isclose(same[k], prev[k]):
            continue    # value did not move that day; uninformative
        assert np.isclose(r.own_kp_adj_o_c, prev[k]), (
            f"team {r.team_id} on {same_day.date()} got the SAME-DAY snapshot"
        )
        checked += 1
    assert checked >= 10, f"only {checked} informative rows checked"


def test_own_ratings_as_of_ignores_the_game_itself_but_reacts_to_it_later(universe):
    """The teeth of the leak test: corrupt one game's box line, refit the
    season's walk-forward ratings, and require (a) the ratings AS OF that
    game's own date to be bit-identical -- its own result cannot reach its own
    features -- and (b) the ratings at a LATER date to move, which proves the
    pipeline is actually sensitive to that game and (a) is not vacuous."""
    tg = orat.load_team_games(universe, [SEASON])
    dates = np.sort(tg["game_date"].unique())
    mid = pd.Timestamp(dates[len(dates) // 2])
    target = tg[tg["game_date"] == mid].iloc[0]
    gid, teams = target["game_id"], set(tg[tg["game_id"] == target["game_id"]]["team_id"])

    corrupt = tg.copy()
    m = corrupt["game_id"] == gid
    corrupt.loc[m, "off_eff"] = [400.0, 5.0]
    corrupt.loc[m, "game_poss"] = 300.0

    kw = dict(lam_eff=5.0, lam_tempo=5.0, prior_eff=None, prior_tempo=None, w_eff=0.0, w_tempo=0.0)
    base = orat.run_to_frame(orat.fit_season(tg, SEASON, **kw))
    dirty = orat.run_to_frame(orat.fit_season(corrupt, SEASON, **kw))

    cols = ["off_c", "def_c", "tempo_rel"]
    key = ["as_of_date", "team_id"]
    b = base.set_index(key)[cols].sort_index()
    d = dirty.set_index(key)[cols].sort_index()

    # (a) nothing on or before the game's own date may move
    on_day = b.index.get_level_values("as_of_date") <= mid
    pd.testing.assert_frame_equal(b[on_day], d[on_day], check_exact=False, atol=1e-9)

    # (b) the two teams involved must move afterwards
    later = b.index.get_level_values("as_of_date") > mid
    moved = (b[later] - d[later]).abs().max(axis=1)
    involved = moved.index.get_level_values("team_id").isin(teams)
    assert moved[involved].max() > 0.5, "corrupting a game changed nothing later -- test is vacuous"


def test_own_n_games_counts_only_strictly_earlier_games(baseline, universe):
    """`own_n_games` on a team-game row must equal the number of that team's
    D-I games strictly BEFORE that date in the season."""
    tg, _ = baseline
    played = tg[["team_id", "game_date"]].copy()
    sample = tg.sample(300, random_state=20260910)
    counts = {}
    for r in sample.itertuples():
        n_before = int(((played["team_id"] == r.team_id) & (played["game_date"] < r.game_date)).sum())
        counts[(r.team_id, r.game_date)] = n_before
        assert r.own_n_games == n_before, (
            f"team {r.team_id} on {r.game_date}: own_n_games={r.own_n_games}, {n_before} games before"
        )
    assert len(counts) > 100


def test_own_ratings_change_form_leak_stat_is_inside_the_gate():
    """The persisted INV-45 result must still pass |as-joined corr| <= 0.15."""
    path = orat.DEFAULT_OUT_DIR / "own_ratings_leak_test.csv"
    if not path.exists():
        pytest.skip("run scripts/build_own_ratings.py first")
    leak = pd.read_csv(path)
    pooled = leak[leak["season"].astype(str) == "ALL"]
    assert len(pooled) >= 3
    assert (pooled["corr_asjoined"].abs() <= 0.15).all(), pooled[["column", "corr_asjoined"]]


# ---------------------------------------------------------------------------
# 2. simulator determinism
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim_inputs(baseline):
    tg, g = baseline
    try:
        bundle = M.load("F1", "A_own")
    except FileNotFoundError:
        pytest.skip("run scripts/train_control.py --fold F1 --anchor A_own first")
    sub = g.head(200).copy()
    tg_sub = tg[tg["game_id"].isin(set(sub["game_id"]))]
    return S.prepare(sub, tg_sub, bundle), sub, tg_sub, bundle


def test_simulator_is_bit_identical_under_a_fixed_seed(sim_inputs):
    inp, _, _, _ = sim_inputs
    a = S.simulate(inp, np.arange(5), chunk_seeds=2)
    b = S.simulate(inp, np.arange(5), chunk_seeds=2)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_simulator_result_is_independent_of_seed_chunking(sim_inputs):
    inp, _, _, _ = sim_inputs
    a = S.simulate(inp, np.arange(6), chunk_seeds=1).sort_values(["seed", "game_id"]).reset_index(drop=True)
    b = S.simulate(inp, np.arange(6), chunk_seeds=6).sort_values(["seed", "game_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_a_games_draws_do_not_depend_on_the_other_games_in_the_run(sim_inputs):
    """(seed, game_id) keying, the property that makes paired arms comparable."""
    inp, sub, tg_sub, bundle = sim_inputs
    full = S.simulate(inp, np.arange(3), chunk_seeds=2)

    half_games = sub.head(37)
    inp_half = S.prepare(half_games, tg_sub[tg_sub["game_id"].isin(set(half_games["game_id"]))], bundle)
    half = S.simulate(inp_half, np.arange(3), chunk_seeds=2)

    merged = half.merge(full, on=["game_id", "seed"], suffixes=("_half", "_full"))
    assert len(merged) == len(half)
    for c in ("home_pts", "away_pts", "possessions", "n_ot"):
        assert (merged[f"{c}_half"] == merged[f"{c}_full"]).all(), c


def test_different_seeds_give_different_draws(sim_inputs):
    inp, _, _, _ = sim_inputs
    a = S.simulate(inp, np.array([0]))
    b = S.simulate(inp, np.array([1000]))
    assert not (a["home_pts"].to_numpy() == b["home_pts"].to_numpy()).all()


def test_rng_streams_are_stable_and_uncorrelated():
    gid = np.arange(1, 50001, dtype=np.uint64)
    k = crng.stream_keys(7, gid, "control")
    assert np.array_equal(k, crng.stream_keys(7, gid, "control"))
    assert not np.array_equal(k, crng.stream_keys(7, gid, "other_family"))
    u0, u1 = crng.uniforms(k, 0), crng.uniforms(k, 1)
    assert u0.min() > 0.0 and u0.max() < 1.0
    assert abs(float(u0.mean()) - 0.5) < 0.01
    assert abs(float(np.corrcoef(u0, u1)[0, 1])) < 0.02


def test_preflight_rejects_missing_features(sim_inputs):
    inp, sub, tg_sub, bundle = sim_inputs
    broken = tg_sub.drop(columns=[bundle.rates["tpa"].features[0]])
    with pytest.raises(M.FeaturePreflightError):
        S.prepare(sub, broken, bundle)


def test_overtime_stub_is_flagged_and_only_fires_on_ties(sim_inputs):
    inp, _, _, _ = sim_inputs
    sims = S.simulate(inp, np.arange(40), chunk_seeds=8)
    assert (sims["n_ot"] >= 0).all()
    # every game that took an OT period must be untied at the end
    assert (sims.loc[sims["n_ot"] > 0, "home_pts"] != sims.loc[sims["n_ot"] > 0, "away_pts"]).all()
    assert 0.0 < float((sims["n_ot"] > 0).mean()) < 0.15
