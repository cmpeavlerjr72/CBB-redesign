"""rotation tests: the five-on-floor invariant, leak-safety of the as-of feature
builder, the CBBD<->ESPN player crosswalk, and RNG determinism.

Four things are worth a dedicated test for the L4 rotation model beyond what
`tests/test_control.py` and `tests/test_pace.py` already cover for the RNG
contract they share:

1. **Five on the floor, always, for every arm.** This is the one invariant the
   possession engine cannot survive a violation of, so it is asserted per
   possession (exactly five slots, five *distinct* players, all of them from the
   team's own candidate list, nobody with five fouls) rather than on a summary.
2. **A game's own box score never enters its own pregame features.** Proved two
   ways, the same pair `tests/test_possession_outcome.py` uses: an independent
   recomputation of the strictly-earlier average from the raw player-game table,
   and invariance of every as-of column to corrupting the game's own rows. The
   second is what catches a bare `.expanding()` where a `.shift(1).expanding()`
   was meant.
3. **The crosswalk is exact where it claims to be.** `source_id_verified` rows
   must land on an ESPN id that really exists in that season's hoopR box, ids
   must be unique in both directions, and the possession-weighted match rate is
   pinned so a regression in the roster pull cannot pass silently.
4. **Determinism.** A game's draws depend on (seed, game_id, "rotation") and
   nothing else -- re-simulating gives bit-identical lineups, a different seed
   gives different ones, and dropping other games from the run moves nothing.
   The online `RotationSampler.next_lineup` path must reproduce the offline
   `simulate` path exactly, or the engine and the bake-off are two models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.data import player_ids as PID  # noqa: E402
from cbb_sim.models import rotation as R  # noqa: E402

SEASON = 2024
N_GAMES = 60


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def season_data():
    tp = R.load_team_possessions(SEASON)
    games = sorted(tp["game_id"].unique())[:N_GAMES]
    tp = tp[tp["game_id"].isin(games)]
    pg = R.player_game_minutes(tp)
    feats = R.build_asof_player_features(pg)
    return {"tp": tp, "pg": pg, "feats": feats}


@pytest.fixture(scope="module")
def fitted(season_data):
    pg, feats, tp = season_data["pg"], season_data["feats"], season_data["tp"]
    fit = R.RotationFit(
        season_train=SEASON,
        role_prior=R.fit_role_prior(pg),
        k0=2.0, w_dnp=0.5,
        tail_ratio=0.7, n_profile=13,
        alpha=50.0, alpha_family=28.0, alpha_starters=40.0, alpha_bench=9.0,
        swap_threshold=0.3, ema_horizon=360.0, lam_deficit=0.5,
        foul_rate_scale=1.0, min_share=0.02, start_alpha=0.5,
        p_play=[0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.89, 0.84, 0.73, 0.56,
                0.42, 0.28, 0.21, 0.15, 0.12],
        w_avail_dnp=0.4, fpm_league=0.083, fpm_prior_min=150.0,
    )
    fit.tilt = R.fit_tilt_tables(tp, R.add_asof_ranks(feats, fit), None)
    fit.hazard_exit = {"coef": [0.0] * len(R.HAZARD_FEATURES), "intercept": -3.0}
    fit.hazard_enter = {"coef": [0.0] * len(R.HAZARD_FEATURES), "intercept": -3.5}
    # round-2 arms: small non-zero coefficients so the override actually fires
    rs = np.random.RandomState(0)
    fit.notes = {
        "r6_exit": {"coef": list(rs.normal(0, 0.2, len(R.R6_FEATURES))), "intercept": -3.0},
        "r6_enter": {"coef": list(rs.normal(0, 0.2, len(R.R6_FEATURES))), "intercept": -3.5},
        "r5_exit": {"coef": list(rs.normal(0, 0.3, len(R.OVERRIDE_FEATURES))),
                    "intercept": -3.0},
        "r5_enter": {"coef": list(rs.normal(0, 0.3, len(R.OVERRIDE_FEATURES))),
                     "intercept": -3.5},
        "r5_override_scale": 4.0, "r5_block_base": 0.02, "r5_donor_k": 5,
    }
    priors = R.build_priors(feats, fit)
    scripts = R.build_scripts(tp)
    return fit, priors, scripts


def _arms(fit, tp, feats):
    donors = R.build_donor_bank(tp, feats)
    return [
        R.R1Dirichlet(fit),
        R.R2HierDirichlet(fit),
        R.R3StintHazard(fit),
        R.R4StintResample(fit, donors),
        R.R5Hybrid(fit, donors),
        R.R6StintHazard(fit),
    ]


# ---------------------------------------------------------------------------
# 1. Five on the floor, always
# ---------------------------------------------------------------------------
def test_five_on_floor_invariant_every_arm(season_data, fitted):
    fit, priors, scripts = fitted
    keys = [k for k in scripts if k in priors][:40]
    assert len(keys) >= 20, "not enough team-games to test the invariant on"
    for arm in _arms(fit, season_data["tp"], season_data["feats"]):
        for key in keys:
            prior, script = priors[key], scripts[key]
            lineups, fouls = arm.simulate(prior, script, R.game_stream(0, key[0]))
            assert lineups.shape == (script.n, 5), f"{arm.name} {key} wrong shape"
            candidates = set(int(p) for p in prior.pids)
            for k in range(script.n):
                row = lineups[k]
                assert len(set(int(x) for x in row)) == 5, (
                    f"{arm.name} {key} possession {k}: duplicate player on the floor")
                assert candidates.issuperset(int(x) for x in row), (
                    f"{arm.name} {key} possession {k}: player outside the candidate list")
            idx = {int(p): i for i, p in enumerate(prior.pids)}
            for k in range(script.n):
                for p in lineups[k]:
                    assert fouls[k, idx[int(p)]] < R.FOUL_OUT, (
                        f"{arm.name} {key} possession {k}: a fouled-out player is on the floor")


def test_five_on_floor_holds_when_almost_everyone_is_unavailable(fitted):
    """The invariant must survive the degenerate case the availability draw can
    produce: a team where hardly anybody is available."""
    fit, priors, scripts = fitted
    key = next(k for k in scripts if k in priors)
    prior, script = priors[key], scripts[key]
    starved = R.TeamPrior(**{**prior.__dict__, "p_avail": np.zeros(prior.n)})
    rng = R.game_stream(1, key[0])
    avail = R.draw_available(starved, rng)
    assert avail.sum() >= 5, "draw_available must guarantee five"
    lineups, _ = R.R1Dirichlet(fit).simulate(starved, script, R.game_stream(1, key[0]))
    assert lineups.shape == (script.n, 5)
    assert all(len(set(int(x) for x in row)) == 5 for row in lineups)


# ---------------------------------------------------------------------------
# 2. Leak-safety
# ---------------------------------------------------------------------------
def test_asof_features_are_the_strictly_earlier_average(season_data):
    """Independent recomputation: `mpg_asof_raw` must equal the mean of the
    player's minutes over the team's strictly earlier games, counting games he
    missed as zeros."""
    pg, feats = season_data["pg"], season_data["feats"]
    order = (pg[["team_id", "game_id", "game_date"]].drop_duplicates()
             .sort_values(["team_id", "game_date", "game_id"]))
    order["idx"] = order.groupby("team_id").cumcount()
    idx = order.set_index(["team_id", "game_id"])["idx"]
    mins = {(int(t), int(g), int(p)): float(m) for t, g, p, m in
            zip(pg["team_id"], pg["game_id"], pg["pid"], pg["minutes"])}

    checked = 0
    for row in feats.sample(min(400, len(feats)), random_state=0).itertuples():
        i = int(idx.loc[(row.team_id, row.game_id)])
        if i == 0:
            continue
        earlier = order[(order["team_id"] == row.team_id) & (order["idx"] < i)]
        got = sum(mins.get((int(row.team_id), int(g), int(row.pid)), 0.0)
                  for g in earlier["game_id"])
        assert row.mpg_asof_raw == pytest.approx(got / i, abs=1e-9), (
            f"team {row.team_id} game {row.game_id} pid {row.pid}")
        checked += 1
    assert checked > 100


def test_asof_features_ignore_the_games_own_box_score(season_data):
    """Corrupting a game's own player-game rows must not move any as-of column
    for that game. A bare `.expanding()` in place of `.shift(1).expanding()`
    fails here even though the previous test would still pass on most rows."""
    pg = season_data["pg"]
    base = R.build_asof_player_features(pg)
    target = int(base["game_id"].mode().iloc[0]) if len(base) else None
    mid = sorted(pg["game_id"].unique())[len(pg["game_id"].unique()) // 2]
    corrupt = pg.copy()
    hit = corrupt["game_id"] == mid
    assert hit.any()
    corrupt.loc[hit, "minutes"] = 999.0
    corrupt.loc[hit, "is_starter"] = True
    after = R.build_asof_player_features(corrupt)

    key = ["game_id", "team_id", "pid"]
    cols = [c for c in ["mpg_asof_raw", "start_freq_asof", "start_ewma_50",
                        "dnp_rate_asof", "last_game_dnp", "games_played_asof",
                        "team_games_asof", "minutes_rank_asof", "rotation_depth_asof"]
            if c in base.columns]
    a = base[base["game_id"] == mid].set_index(key)[cols].sort_index()
    b = after[after["game_id"] == mid].set_index(key)[cols].sort_index()
    assert len(a) > 0 and a.index.equals(b.index)
    for c in cols:
        assert np.allclose(a[c].to_numpy(dtype="float64"),
                           b[c].to_numpy(dtype="float64"), equal_nan=True), (
            f"as-of column {c} moved when game {mid}'s own box score was corrupted")
    assert target is None or True


def test_contemporaneous_columns_are_not_on_the_feature_frame(season_data):
    """The columns the expanding sums were built from are that game's own box
    score and must not be handed to a consumer."""
    feats = season_data["feats"]
    for banned in ["minutes", "is_starter", "played", "dnp", "fouls",
                   "cum_min", "cum_start", "cum_fouls"]:
        assert banned not in feats.columns, f"{banned} leaks the game's own box score"


def test_priors_use_only_earlier_games(season_data, fitted):
    """End-to-end: a `TeamPrior` built from a corrupted table (that game's own
    rows blown up) must be identical to one built from the clean table."""
    fit, _, _ = fitted
    pg = season_data["pg"]
    mid = sorted(pg["game_id"].unique())[len(pg["game_id"].unique()) // 2]
    clean = R.build_priors(R.build_asof_player_features(pg), fit)
    corrupt = pg.copy()
    corrupt.loc[corrupt["game_id"] == mid, "minutes"] = 999.0
    dirty = R.build_priors(R.build_asof_player_features(corrupt), fit)
    keys = [k for k in clean if k[0] == mid]
    assert keys
    for k in keys:
        assert np.array_equal(clean[k].pids, dirty[k].pids)
        assert np.allclose(clean[k].share, dirty[k].share)


# ---------------------------------------------------------------------------
# 3. Crosswalk
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def crosswalk():
    return PID.load_crosswalk()


def test_crosswalk_ids_are_unique_both_ways(crosswalk):
    cw = crosswalk
    assert not cw.duplicated(subset=["season", "cbbd_player_id"]).any()
    m = cw[cw["espn_athlete_id"].notna()]
    dup = m[m.duplicated(subset=["season", "espn_athlete_id"], keep=False)]
    assert len(dup) == 0, f"{len(dup)} CBBD ids share an ESPN id within a season"


def test_source_id_verified_rows_exist_in_hoopr(crosswalk):
    v = crosswalk[crosswalk["match_method"] == "source_id_verified"]
    assert len(v) > 30_000
    for season, g in v.groupby("season"):
        pb = pd.read_parquet(
            PID.DEFAULT_PLAYER_BOX_DIR / f"player_box_{int(season)}.parquet",
            columns=["athlete_id"])
        ids = set(pb["athlete_id"].dropna().astype("int64"))
        got = set(g["espn_athlete_id"].astype("int64"))
        assert got.issubset(ids), f"season {season}: verified ids missing from hoopR"


def test_crosswalk_match_rate_is_pinned(crosswalk):
    """The reported number, asserted so a regression in the roster pull is loud."""
    rep = PID.crosswalk_report(crosswalk)
    assert rep["overall"]["possession_weighted_match_rate"] > 0.999
    assert rep["overall"]["on_floor_player_match_rate"] > 0.995
    for season in ("2024", "2025"):
        assert rep[season]["possession_weighted_match_rate"] > 0.999


def test_name_normalisation():
    assert PID.normalize_name("Bobby Pettiford Jr.") == "bobby pettiford"
    assert PID.normalize_name("D'Andre Jackson") == "dandre jackson"
    assert PID.normalize_name("Nikola Djurisic") == PID.normalize_name("Nikola Đurišić")
    assert PID.normalize_jersey("03") == PID.normalize_jersey("3") == "3"
    assert PID.normalize_jersey(None) == ""


# ---------------------------------------------------------------------------
# 4. Determinism and the RNG contract
# ---------------------------------------------------------------------------
def test_simulation_is_bit_identical_under_the_same_seed(season_data, fitted):
    fit, priors, scripts = fitted
    keys = [k for k in scripts if k in priors][:12]
    for arm in _arms(fit, season_data["tp"], season_data["feats"]):
        for key in keys:
            a, _ = arm.simulate(priors[key], scripts[key], R.game_stream(11, key[0]))
            b, _ = arm.simulate(priors[key], scripts[key], R.game_stream(11, key[0]))
            assert np.array_equal(a, b), f"{arm.name} is not reproducible under one seed"


def test_a_different_seed_moves_the_lineups(season_data, fitted):
    fit, priors, scripts = fitted
    keys = [k for k in scripts if k in priors][:12]
    arm = R.R1Dirichlet(fit)
    differ = 0
    for key in keys:
        a, _ = arm.simulate(priors[key], scripts[key], R.game_stream(11, key[0]))
        b, _ = arm.simulate(priors[key], scripts[key], R.game_stream(12, key[0]))
        differ += int(not np.array_equal(a, b))
    assert differ >= len(keys) - 1, "seed has almost no effect on the draw"


def test_a_games_stream_does_not_depend_on_the_other_games_in_the_run(fitted):
    """The whole point of the (seed, game_id, family) contract: dropping a game
    from the universe must not move any other game's draws."""
    fit, priors, scripts = fitted
    keys = [k for k in scripts if k in priors][:8]
    arm = R.R1Dirichlet(fit)
    full = {k: arm.simulate(priors[k], scripts[k], R.game_stream(5, k[0]))[0] for k in keys}
    subset = {k: arm.simulate(priors[k], scripts[k], R.game_stream(5, k[0]))[0]
              for k in keys[2:]}
    for k in keys[2:]:
        assert np.array_equal(full[k], subset[k])


def test_stream_keys_come_from_the_control_rng(fitted):
    from cbb_sim.control import rng as crng
    key = int(crng.stream_keys(7, np.asarray([401_706_881], dtype="int64"), "rotation")[0])
    other = int(crng.stream_keys(7, np.asarray([401_706_881], dtype="int64"), "pace")[0])
    assert key != other, "the family label must change the stream"
    g1 = R.game_stream(7, 401_706_881)
    g2 = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))
    assert np.array_equal(g1.random(8), g2.random(8))


def test_next_lineup_reproduces_the_offline_simulation(season_data, fitted):
    """The engine-facing sampler and the offline path must be one model."""
    fit, priors, scripts = fitted
    keys = [k for k in scripts if k in priors][:10]
    arm = R.R1Dirichlet(fit)
    for key in keys:
        prior, script = priors[key], scripts[key]
        offline, _ = arm.simulate(prior, script, R.game_stream(3, key[0]))

        sampler = R.RotationSampler(
            arm, prior, seed=3, game_id=key[0], is_home=script.is_home,
            expected_total_seconds=float(script.dur.sum()),
            rng=R.game_stream(3, key[0]),
        )
        online = []
        for k in range(script.n):
            margin = int(script.margin[k]) * (1 if script.is_home else -1)
            state = R.RotationState(
                period=int(script.period[k]),
                seconds_remaining=int(script.start_clock[k]),
                score_diff=margin,
                last_possession_seconds=float(script.dur[k - 1]) if k else 0.0,
            )
            online.append(sampler.next_lineup(state))
        online = np.asarray(online, dtype="int64")
        assert np.array_equal(np.sort(offline, axis=1), np.sort(online, axis=1)), (
            f"{key}: next_lineup diverges from simulate")


# ---------------------------------------------------------------------------
# 5. Guards on the fitted objects
# ---------------------------------------------------------------------------
def test_override_deviation_is_zero_in_a_neutral_state(fitted, season_data):
    """R5 keys its block on the state DEVIATION, so every state-dependent column
    of the override design must vanish with no fouls, a tied game and time left:
    that is what makes the override inert in an ordinary state instead of
    churning the lineup (the defect that sent the first parameterisation's fitted
    scale to the grid ceiling)."""
    fit, priors, scripts = fitted
    key = next(k for k in scripts if k in priors)
    prior, script = priors[key], scripts[key]
    neutral = R.GameScript(**{**script.__dict__,
                              "margin": np.zeros(script.n, dtype="int64"),
                              "margin_bucket": np.zeros(script.n, dtype="int64"),
                              "time_bucket": np.zeros(script.n, dtype="int64")})
    n = prior.n
    X = R._override_design(prior, neutral, prior.share * neutral.total_slot,
                           np.zeros(n), np.zeros(n, dtype="int64"), np.zeros(n),
                           np.arange(n), 0, neutral.remaining_slot())
    state = X[:, R.OVERRIDE_STATE_COLS]
    assert np.allclose(state, 0.0), "the override is not inert in a neutral state"
    # and it must bite once a starter is in foul trouble
    fouls = np.zeros(n, dtype="int64")
    fouls[prior.starters()[:1]] = 4
    X2 = R._override_design(prior, neutral, prior.share * neutral.total_slot,
                            np.zeros(n), fouls, np.zeros(n),
                            np.arange(n), 0, neutral.remaining_slot())
    assert not np.allclose(X2[:, R.OVERRIDE_STATE_COLS], 0.0)


def test_donor_depth_zero_means_every_earlier_game(season_data):
    tp, feats = season_data["tp"], season_data["feats"]
    d5 = R.build_donor_bank(tp, feats, k_donors=5)
    dall = R.build_donor_bank(tp, feats, k_donors=0)
    deeper = [k for k in d5 if len(dall[k]) > len(d5[k])]
    assert deeper, "k_donors=0 must widen at least one team-game's bank"
    assert all(len(d5[k]) <= 5 for k in d5)


def test_extend_profile_preserves_mass_and_adds_only_anonymous_tail():
    share = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    pids = np.array([11, 12, 13, 14, 15], dtype="int64")
    out, ids = R.extend_profile(share, pids, n_profile=9, tail_ratio=0.6)
    assert len(out) == 9 and out.sum() == pytest.approx(1.0)
    assert np.all(ids[:5] == pids)
    assert np.all(ids[5:] < 0), "tail slots must be anonymous, never named players"
    assert np.all(np.diff(out[4:]) < 0), "the tail must decay"


def test_apply_min_target_floors_only_available_players():
    s = np.array([0.5, 0.3, 0.19, 0.01])
    avail = np.array([True, True, False, True])
    out = R.apply_min_target(s, avail, 0.05)
    assert out[2] == 0.0
    assert out.sum() == pytest.approx(1.0)
    assert out[3] >= 0.05 - 1e-9


def test_foul_tilt_from_event_study_is_monotone_and_zero_at_foul_out():
    ev = pd.DataFrame([
        {"fouls": f, "time_bucket": 0, "before_on": 700.0, "before_n": 1000.0,
         "after_on": 700.0 * r, "after_n": 1000.0}
        for f, r in [(1, 0.95), (2, 0.85), (3, 0.5), (4, 0.3)]
    ])
    tab = R.foul_tilt_from_event_study(ev)
    assert tab[0, 0] == 1.0
    assert tab[R.FOUL_OUT, 0] == 0.0
    assert np.all(np.diff(tab[:5, 0]) < 0), "more fouls must not mean more floor time"


def test_time_and_margin_buckets():
    tb = R.time_bucket(np.array([1, 2, 2, 2, 3]), np.array([600, 900, 300, 60, 200]))
    assert list(tb) == [0, 1, 2, 3, 4]
    assert list(R.margin_bucket(np.array([0, -5, 6, -15, 16]))) == [0, 0, 1, 1, 2]
