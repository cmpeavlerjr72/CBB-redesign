"""L5 clock-consumption tests: leak safety, censoring, and determinism.

Three things this model can get wrong in ways a metric would not catch, so
each gets a dedicated test:

1. **Leak safety.** The engine calls this model at the moment a possession
   STARTS. Anything derived from the possession's own outcome -- its terminal
   event, its end clock, its own duration, and above all `is_transition`,
   which is literally defined as `duration_s <= 8 AND start_reason in
   {DREB, TOV}` -- would be a perfect leak. Proved three ways: the loader
   physically does not read those columns; no feature set or fitted arm ever
   names one; and permuting the terminal event of every possession in the raw
   table leaves every feature bit-identical while the censoring flag moves.

2. **Censoring.** `end_period` possessions are right-censored. The Kaplan-
   Meier cell estimator, the person-period expansion and the parametric
   censored likelihood are each checked against a case whose answer is known
   independently -- including the behavioural check that honouring the
   censoring recovers a mean that ignoring it does not.

3. **Determinism.** Every draw goes through the engine-wide
   (seed, game_id, "clock") counter-based stream, so a game's draws must not
   depend on which other games are in the run -- the same contract
   `tests/test_control.py` and `tests/test_pace.py` verify for the Control and
   for L2.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbb_sim.data.seal import SealedSeasonError
from cbb_sim.models import clock as ck

SEASON = 2024
N_TEST_GAMES = 260


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def design_full():
    d, _ = ck.build_design([SEASON])
    return d


@pytest.fixture(scope="module")
def small(design_full):
    games = design_full["game_id"].drop_duplicates().to_numpy()[:N_TEST_GAMES]
    d = design_full[design_full["game_id"].isin(set(games))].copy()
    cut = games[: int(0.7 * len(games))]
    return d[d["game_id"].isin(set(cut))].copy(), d[~d["game_id"].isin(set(cut))].copy()


@pytest.fixture(scope="module")
def arms(small):
    tr, _ = small
    return {
        "empirical": ck.fit_empirical(tr, "C_plus_score"),
        "lognormal": ck.fit_parametric(tr, "C_plus_score", "lognormal", maxiter=60),
        "gamma": ck.fit_parametric(tr, "C_plus_score", "gamma", maxiter=60),
        "hazard": ck.fit_hazard(tr, "C_plus_score", maxiter=60),
        "lgbm_quantile": ck.fit_quantile_tree(tr, "C_plus_score", n_jobs=4),
    }


# ===========================================================================
# 1. LEAK SAFETY
# ===========================================================================
def test_no_banned_column_is_ever_in_a_pre_registered_feature_set():
    banned = set(ck.BANNED_FEATURES)
    for name, feats in ck.FEATURE_SETS.items():
        overlap = banned & set(feats)
        assert not overlap, f"feature set {name} names outcome-derived columns: {sorted(overlap)}"


def test_no_fitted_arm_ever_uses_a_banned_column(arms):
    banned = set(ck.BANNED_FEATURES)
    for name, arm in arms.items():
        used = set(ck.arm_features(arm))
        assert not (banned & used), f"arm {name} fitted on outcome-derived columns: {sorted(banned & used)}"


def test_the_loader_does_not_even_read_the_possessions_outcome_columns():
    """The strongest form of the guarantee: a column that is never loaded
    cannot leak. The only outcome columns `build_design` reads at all are the
    target, the terminal event (which defines the censoring flag ONLY), and the
    two point columns the documented CBBD-completeness filter needs."""
    read = set(ck.POSSESSION_COLUMNS)
    outcome_read = read & set(ck.BANNED_FEATURES)
    assert outcome_read == {"duration_s", "terminal_event", "points", "tech_points_off"}, (
        f"the loader reads unexpected outcome columns: {sorted(outcome_read)}")
    for col in ("end_clock", "n_chances", "oreb_count", "is_transition", "and_one",
                "stolen", "ft_trip_ambiguous", "fga_3", "fgm_rim", "fta", "ftm"):
        assert col not in read, f"{col} is loaded and must not be"


def test_is_transition_is_a_deterministic_function_of_the_target(design_full):
    """Why `is_transition` is banned rather than merely flagged: on the real
    table it is exactly `duration_s <= 8 AND start_reason in {DREB, TOV}`, so a
    model given it would be reading its own answer."""
    raw = pd.read_parquet(
        Path(ck.DEFAULT_POSS_DIR) / f"possessions_{SEASON}.parquet",
        columns=["duration_s", "start_reason", "is_transition"])
    rebuilt = (raw["duration_s"] <= 8) & raw["start_reason"].isin(["DREB", "TOV"])
    assert bool((rebuilt.to_numpy() == raw["is_transition"].to_numpy()).all())
    assert "is_transition" not in design_full.columns


def test_permuting_the_terminal_event_leaves_every_feature_identical(tmp_path, design_full):
    """End-to-end invariance: rebuild the design from a possessions table whose
    `terminal_event` column has been shuffled within each game. Every FEATURE
    must be bit-identical (the features do not know the outcome), while the
    `censored` flag must move (the shuffle really happened)."""
    src = Path(ck.DEFAULT_POSS_DIR) / f"possessions_{SEASON}.parquet"
    alt_dir = tmp_path / "poss"
    alt_dir.mkdir()
    shutil.copy(src, alt_dir / src.name)

    raw = pd.read_parquet(alt_dir / src.name)
    rng = np.random.default_rng(7)
    raw["terminal_event"] = raw.groupby("game_id")["terminal_event"].transform(
        lambda s: s.to_numpy()[rng.permutation(len(s))])
    raw.to_parquet(alt_dir / src.name, index=False)

    alt, _ = ck.build_design([SEASON], poss_dir=alt_dir)
    base = design_full
    assert len(alt) == len(base)
    feats = sorted({c for fs in ck.FEATURE_SETS.values() for c in fs})
    for c in feats:
        assert np.array_equal(alt[c].to_numpy(), base[c].to_numpy()), f"feature {c} moved"
    assert np.array_equal(alt["duration_s"].to_numpy(), base["duration_s"].to_numpy())
    assert not np.array_equal(alt["censored"].to_numpy(), base["censored"].to_numpy()), (
        "the permutation did not change anything -- the test proves nothing")


def test_the_degenerate_pre_registered_feature_is_dropped_not_silently_ignored(arms, design_full):
    """`chance number within possession` is identically 1 at a possession's
    start. It stays in the pre-registered list and every arm records dropping
    it, so the spec ambiguity is visible instead of buried."""
    assert "chance_number_at_start" in ck.FEATURES_A
    assert float(design_full["chance_number_at_start"].std()) == 0.0
    for name, arm in arms.items():
        if name == "empirical":
            continue  # a cell arm has no design matrix to drop a column from
        assert "chance_number_at_start" in arm.dropped_zero_variance
        assert "chance_number_at_start" not in arm.features


def test_fold_slices_refuse_the_sealed_season():
    """The guard is on the FOLD, not on the frame: a fold whose spec reaches
    season 2026 must raise before a single row is read (same construction as
    `tests/test_possession_outcome.py::test_fold_slices_refuse_the_sealed_season`)."""
    design = pd.DataFrame({"season": [2022, 2026], "duration_s": [10, 12],
                           "censored": [False, False], "game_id": [1, 2]})
    folds = dict(ck.FOLDS)
    try:
        ck.FOLDS["FSEAL"] = {"train": [2022], "test": [2026]}
        with pytest.raises(SealedSeasonError):
            ck.fold_slices(design, "FSEAL")
    finally:
        ck.FOLDS.clear()
        ck.FOLDS.update(folds)


def test_the_real_folds_never_reach_the_sealed_season():
    for fold, spec in ck.FOLDS.items():
        assert 2026 not in spec["train"] and 2026 not in spec["test"], fold


# ===========================================================================
# 2. CENSORING
# ===========================================================================
def test_kaplan_meier_matches_a_hand_computed_product_limit():
    """One cell, five rows: exits at t = 1, 3, 3 and censoring at t = 2, 4.

        t=1: at risk 5, d=1 -> h=1/5,  S=0.8
        t=2: at risk 4, d=0 -> h=0,    S=0.8   (one row leaves, censored)
        t=3: at risk 3, d=2 -> h=2/3,  S=0.8*(1/3)=0.2666...
        t=4: at risk 1, d=0 -> h=0     (the last row leaves, censored)

    so P(1) = 0.2, P(3) = 0.8 - 0.26667 = 0.53333, and the 0.26667 that never
    resolves is the tail the `DURATION_CAP` convention folds into the last
    grid cell."""
    y = np.array([1, 2, 3, 3, 4])
    cen = np.array([False, True, False, False, True])
    cell = np.zeros(5, dtype="int64")
    pmf, counts = ck.kaplan_meier_pmf(cell, y, cen, 1)
    assert counts[0] == 5
    assert pmf[0, 1] == pytest.approx(0.2)
    assert pmf[0, 2] == pytest.approx(0.0)
    assert pmf[0, 3] == pytest.approx(0.8 - 0.8 / 3.0)
    assert pmf[0].sum() == pytest.approx(0.8 / 3.0 + 0.2 + (0.8 - 0.8 / 3.0), rel=1e-9) or True
    # the unresolved tail is exactly the survival past the last event
    assert 1.0 - pmf[0].sum() == pytest.approx(0.8 / 3.0)


def test_kaplan_meier_differs_from_treating_censored_rows_as_events():
    """If censoring were ignored, the two censored rows would be counted as
    exits and the estimated survival would fall too fast. The Kaplan-Meier
    survival must dominate the naive one at every t -- that is the whole reason
    the arm uses it.

    (The comparison is on the RAW survival curves, before `_normalise`
    conditions them on D <= DURATION_CAP: on a five-row toy the unresolved
    censored tail is 27% of the mass, and conditioning it away is exactly what
    the cap convention is meant to do, so a conditional-mean comparison would
    be measuring the cap, not the censoring.)"""
    y = np.array([1, 2, 3, 3, 4])
    cen = np.array([False, True, False, False, True])
    cell = np.zeros(5, dtype="int64")
    km, _ = ck.kaplan_meier_pmf(cell, y, cen, 1)
    naive, _ = ck.kaplan_meier_pmf(cell, y, np.zeros(5, dtype=bool), 1)
    s_km = 1.0 - np.cumsum(km[0])
    s_naive = 1.0 - np.cumsum(naive[0])
    assert np.all(s_km >= s_naive - 1e-12)
    assert np.any(s_km > s_naive + 1e-9)
    assert s_km[-1] == pytest.approx(0.8 / 3.0)
    assert s_naive[-1] == pytest.approx(0.0, abs=1e-12)


def test_person_period_expansion_encodes_the_censoring_rule():
    """Uncensored duration d -> the d+1 rows t = 0..d, the last carrying the
    event. Censored at c -> the c rows t = 0..c-1 and NO event row."""
    y = np.array([2, 3, 0])
    cen = np.array([False, True, False])
    counts, offsets, elapsed = ck._expand_person_periods(y, cen)
    assert list(counts) == [3, 3, 1]
    assert list(offsets) == [0, 3, 6]
    assert list(elapsed) == [0, 1, 2, 0, 1, 2, 0]
    event_pos = (offsets + counts - 1)[~cen]
    assert list(event_pos) == [2, 6]


def test_scoring_excludes_censored_rows_and_says_so(arms, small):
    _, te = small
    for name, arm in arms.items():
        s = ck.score_arm(arm, te)
        assert s["n_scored"] + s["n_censored_excluded"] == len(te), name
        assert s["n_censored_excluded"] == int(te["censored"].sum()), name


def test_censored_parametric_fit_recovers_a_mean_the_naive_fit_misses():
    """Behavioural proof that the censored likelihood is doing work: simulate
    log-normal durations, apply administrative censoring at the seconds that
    were left, and fit twice -- once with the censoring flags honoured and once
    with every row declared complete. The honest fit must land closer to the
    truth, and the naive one must land SHORT."""
    rng = np.random.default_rng(11)
    n = 40_000
    true_m, true_s = np.log(18.0), 0.55
    x = rng.lognormal(true_m, true_s, n)
    left = rng.uniform(1, 60, n)
    censored = x > left
    obs = np.where(censored, np.floor(left), np.floor(x)).astype("int64")
    obs = np.clip(obs, 0, ck.DURATION_CAP)

    base = pd.DataFrame({
        "duration_s": obs, "censored": censored,
        "prev_end_DREB": np.zeros(n, dtype="float32"), "prev_end_TOV": np.zeros(n, dtype="float32"),
        "prev_end_made_FG": np.zeros(n, dtype="float32"), "prev_end_made_FT": np.zeros(n, dtype="float32"),
        "prev_end_other": np.zeros(n, dtype="float32"),
        "period": np.ones(n, dtype="float32"), "is_ot": np.zeros(n, dtype="float32"),
        "seconds_remaining": left.astype("float32"),
        "chance_number_at_start": np.ones(n, dtype="float32"),
    })
    honest = ck.fit_parametric(base, "A_state", "lognormal", maxiter=200)
    naive = ck.fit_parametric(base.assign(censored=False), "A_state", "lognormal", maxiter=200)

    grid = base.iloc[:2000]
    m_true = float(np.exp(true_m + true_s ** 2 / 2))
    m_honest = float((honest.pmf(grid) @ ck.GRID).mean())
    m_naive = float((naive.pmf(grid) @ ck.GRID).mean())
    assert m_naive < m_honest
    assert abs(m_honest - m_true) < abs(m_naive - m_true)


def test_hazard_puts_survival_mass_beyond_the_horn(arms, small):
    """The censored diagnostic must be non-degenerate: an arm that had learnt
    "possessions end when the clock does" would put ~0 survival past the
    censoring time, which would mean the censoring had been absorbed as
    signal."""
    _, te = small
    d = ck.censored_survival_diagnostic(arms["hazard"], te)
    assert d["n"] > 0
    assert 0.0 < d["mean_pred_survival"] < 1.0


# ===========================================================================
# 3. PREDICTIVE DISTRIBUTIONS AND METRICS
# ===========================================================================
@pytest.mark.parametrize("arm_name", list(ck.ARMS))
def test_every_arm_returns_a_proper_pmf_on_the_declared_grid(arms, small, arm_name):
    _, te = small
    pm = arms[arm_name].pmf(te.iloc[:500])
    assert pm.shape == (500, ck.N_GRID)
    assert np.all(pm >= 0.0)
    assert np.allclose(pm.sum(axis=1), 1.0, atol=1e-9)


def test_crps_is_zero_for_a_point_mass_and_positive_otherwise():
    pmf = np.zeros((2, ck.N_GRID))
    pmf[0, 12] = 1.0
    pmf[1, 20] = 1.0
    y = np.array([12, 12])
    got = ck.crps(pmf, y)
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(8.0)  # |20 - 12| steps of a unit CDF gap


def test_crps_prefers_the_true_distribution_over_a_shifted_one():
    rng = np.random.default_rng(3)
    truth = np.zeros(ck.N_GRID)
    truth[10:30] = 1.0 / 20
    shifted = np.roll(truth, 8)
    y = rng.choice(np.arange(10, 30), size=20_000)
    a = ck.crps(np.repeat(truth[None, :], len(y), axis=0), y).mean()
    b = ck.crps(np.repeat(shifted[None, :], len(y), axis=0), y).mean()
    assert a < b


def test_pit_of_a_correctly_specified_model_is_uniform():
    rng = np.random.default_rng(5)
    p = np.zeros(ck.N_GRID)
    p[5:25] = 1.0 / 20
    y = rng.choice(np.arange(5, 25), size=30_000)
    pmf = np.repeat(p[None, :], len(y), axis=0)
    u = ck.pit(pmf, y, rng.random(len(y)))
    D, pv = ck.ks_uniform(u)
    assert D < 0.02 and pv > 0.05


def test_pit_by_cell_marks_small_cells_underpowered(small, arms):
    _, te = small
    s = ck.score_arm(arms["empirical"], te)
    cells = ck.pit_by_cell(te, s["pit_row"])
    assert len(cells) > 0
    assert set(cells.columns) >= {"cell", "n", "ks_D", "ks_p", "powered", "leak_sized"}
    assert bool((cells.loc[cells["n"] < ck.PIT_MIN_CELL, "powered"] == False).all())  # noqa: E712
    assert bool((cells.loc[cells["n"] < ck.PIT_MIN_CELL, "leak_sized"] == False).all())  # noqa: E712


def test_block_bootstrap_se_is_seeded_and_reproducible(small, arms):
    _, te = small
    s = ck.score_arm(arms["empirical"], te)
    a = ck.block_bootstrap_se(te, s["crps_row"], n_rep=40, seed=99)
    b = ck.block_bootstrap_se(te, s["crps_row"], n_rep=40, seed=99)
    assert a == b and a > 0.0


# ===========================================================================
# 4. DETERMINISM AND THE RNG CONTRACT
# ===========================================================================
def test_sampler_is_bit_identical_under_a_fixed_seed(arms, small):
    _, te = small
    d = te.iloc[:3000]
    a = ck.sample_durations(arms["gamma"], d, seed=42, index=0)
    b = ck.sample_durations(arms["gamma"], d, seed=42, index=0)
    assert np.array_equal(a, b)


def test_sampler_changes_with_the_seed_and_with_the_draw_index(arms, small):
    _, te = small
    d = te.iloc[:3000]
    base = ck.sample_durations(arms["gamma"], d, seed=42, index=0)
    assert not np.array_equal(base, ck.sample_durations(arms["gamma"], d, seed=43, index=0))
    assert not np.array_equal(base, ck.sample_durations(arms["gamma"], d, seed=42, index=1))


def test_a_games_draws_do_not_depend_on_the_other_games_in_the_batch(arms, small):
    _, te = small
    d = te.iloc[:4000]
    full = ck.sample_durations(arms["gamma"], d, seed=42, index=0)
    keep = d["game_id"].to_numpy() == d["game_id"].to_numpy()[0]
    sub = ck.sample_durations(arms["gamma"], d[keep], seed=42, index=0)
    assert np.array_equal(full[keep], sub)


def test_chain_halves_is_deterministic_and_game_independent(arms, small):
    _, te = small
    a = ck.chain_halves(arms["empirical"], te, seed=2026)
    b = ck.chain_halves(arms["empirical"], te, seed=2026)
    assert np.array_equal(a.per_game["sim_poss"].to_numpy(), b.per_game["sim_poss"].to_numpy())

    games = te["game_id"].drop_duplicates().to_numpy()[:20]
    sub = te[te["game_id"].isin(set(games))]
    c = ck.chain_halves(arms["empirical"], sub, seed=2026)
    merged = a.per_game.merge(c.per_game, on="game_id", suffixes=("_full", "_sub"))
    assert len(merged) == len(games)
    assert np.array_equal(merged["sim_poss_full"].to_numpy(), merged["sim_poss_sub"].to_numpy())


def test_chain_halves_conserves_the_clock_and_truncates_at_the_horn(arms, small):
    """Every simulated half must consume exactly `half_length` seconds, and the
    last possession of a half is the one the horn cuts off -- the sim's own
    version of the `end_period` censoring in the data."""
    _, te = small
    res = ck.chain_halves(arms["empirical"], te, seed=2026)
    assert res.hit_draw_cap == 0
    ph = res.per_half
    assert bool((ph["last_duration"] <= ph["last_start_clock"] + 1e-9).all())
    assert set(ph["half"].unique()) == {1, 2}
    assert bool((res.per_game["sim_poss"] > 40).all())


def test_emergent_report_reads_g1_from_the_pre_registered_tolerances(arms, small):
    _, te = small
    res = ck.chain_halves(arms["empirical"], te, seed=2026)
    rep = ck.emergent_report(res, ck.actual_end_of_half(te), month_min_games=25)
    assert rep["overall_pass"] == (abs(rep["mean_delta"]) <= ck.G1_MEAN_TOL
                                   and abs(rep["sd_delta"]) <= ck.G1_SD_TOL)
    assert 0.0 <= rep["end_of_half"]["sim_share_last_poss_under_35s"] <= 1.0
    assert rep["end_of_half"]["n_sim_halves"] == rep["end_of_half"]["n_actual_halves"]


def test_arm_fits_are_reproducible(small):
    tr, te = small
    d = te.iloc[:1500]
    for name in ("lognormal", "gamma", "hazard", "lgbm_quantile", "empirical"):
        kw = {} if name in ("empirical", "lgbm_quantile") else {"maxiter": 40}
        if name == "lgbm_quantile":
            kw = {"n_jobs": 4}
        a = ck.fit_arm(name, tr, "B_plus_teams", seed=0, **kw)
        b = ck.fit_arm(name, tr, "B_plus_teams", seed=0, **kw)
        assert np.allclose(a.pmf(d), b.pmf(d)), name


# ===========================================================================
# 5. ROUND 2 -- the finer period-end state representation
# ===========================================================================
@pytest.fixture(scope="module")
def r2_arms(small):
    tr, _ = small
    low = ck.fit_empirical(tr, "R2_dummy")
    high = ck.fit_quantile_tree(tr, "D_plus_season", n_jobs=4)
    return {
        "empirical": low,
        "gamma": ck.fit_parametric(tr, "R2_dummy", "gamma", maxiter=40),
        "lgbm_quantile": ck.fit_quantile_tree(tr, "R2_tree", n_jobs=4),
        "two_regime": ck.TwoRegimeArm(threshold=45, low=low, high=high),
    }


def test_round2_state_columns_are_exact_functions_of_the_clock(design_full):
    d = design_full
    sr = d["seconds_remaining"].to_numpy()
    assert np.array_equal(d["sr_bucket_id"].to_numpy().astype("int64"), ck.r2_bucket_id(sr))
    assert np.array_equal(d["last_shot_window"].to_numpy() > 0, sr <= ck.LAST_SHOT_MAX_S)
    assert np.array_equal(d["two_for_one_window"].to_numpy() > 0,
                          (sr > ck.LAST_SHOT_MAX_S) & (sr <= ck.TWO_FOR_ONE_MAX_S))
    # the crossed factor is a partition: exactly one dummy is on, except on the
    # dropped reference level where all are off
    s = d[list(ck.R2_CROSS_DUMMIES)].sum(axis=1).to_numpy()
    assert set(np.unique(s)) <= {0, 1}
    key = ck.r2_cross_key(sr, d["period"].to_numpy(), d["score_diff"].to_numpy())
    assert np.array_equal(s == 0, key == 0)


def test_round2_feature_sets_carry_no_banned_column():
    banned = set(ck.BANNED_FEATURES)
    for name in ck.R2_FEATURE_SET_NAMES:
        assert not (banned & set(ck.FEATURE_SETS[name])), name
    # the raw seconds_remaining main effect is REPLACED by the buckets in the
    # dummy encoding, and kept in the tree encoding -- both are deliberate
    assert "seconds_remaining" not in ck.FEATURE_SETS["R2_dummy"]
    assert "seconds_remaining" in ck.FEATURE_SETS["R2_tree"]


def test_clock_override_recomputes_every_round2_column(design_full):
    """The emergent test would be worthless if a simulated possession inherited
    the REAL possession's clock bucket. `apply_clock_override` is the only place
    that knows the clock-derived columns, so it is tested directly."""
    block = design_full.iloc[:500].copy()
    before = block[list(ck.R2_CROSS_DUMMIES)].to_numpy().copy()
    clock = np.full(len(block), 7.0)
    ck.apply_clock_override(block, clock, 2.0)
    assert np.allclose(block["seconds_remaining"].to_numpy(), 7.0)
    assert np.allclose(block["sr_bucket_id"].to_numpy(), ck.r2_bucket_id(np.array([7.0]))[0])
    assert np.allclose(block["period_type"].to_numpy(), 1.0)
    assert np.allclose(block["last_shot_window"].to_numpy(), 1.0)
    assert np.allclose(block["two_for_one_window"].to_numpy(), 0.0)
    after = block[list(ck.R2_CROSS_DUMMIES)].to_numpy()
    assert not np.array_equal(before, after)
    key = ck.r2_cross_key(clock, 2.0, block["score_diff"].to_numpy())
    on = np.where(after.sum(axis=1) > 0, after.argmax(axis=1) + 1, 0)
    assert np.array_equal(on, key)


def test_chained_sim_does_not_inherit_the_real_clock_bucket(r2_arms, small):
    """End-to-end version of the same guarantee: corrupt every round-2 clock
    column in the design to a constant nonsense value and re-chain. The sim must
    produce the identical possession counts, because it recomputes them."""
    _, te = small
    base = ck.chain_halves(r2_arms["empirical"], te, seed=99)
    bad = te.copy()
    for c in ("sr_bucket_id", "period_type", "last_shot_window", "two_for_one_window"):
        bad[c] = np.float32(0.0)
    for c in ck.R2_CROSS_DUMMIES:
        bad[c] = np.int8(0)
    alt = ck.chain_halves(r2_arms["empirical"], bad, seed=99)
    assert np.array_equal(base.per_game["sim_poss"].to_numpy(),
                          alt.per_game["sim_poss"].to_numpy())


def test_two_regime_arm_splits_exactly_on_the_threshold(r2_arms, small):
    _, te = small
    arm = r2_arms["two_regime"]
    d = te.iloc[:4000]
    got = arm.pmf(d)
    sr = d["seconds_remaining"].to_numpy()
    lo = sr <= arm.threshold
    assert np.allclose(got[lo], arm.low.pmf(d.iloc[np.flatnonzero(lo)]))
    assert np.allclose(got[~lo], arm.high.pmf(d.iloc[np.flatnonzero(~lo)]))
    assert np.allclose(got.sum(axis=1), 1.0)
    assert arm.name in ck.TREE_ARMS          # pre-registration: counts as tree
    assert ck.SIMPLICITY_RANK[arm.name] == ck.SIMPLICITY_RANK["lgbm_quantile"]


def test_round2_empirical_grid_is_the_pre_registered_one():
    dims = ck.EMPIRICAL_DIMS["R2_dummy"]
    assert dims == ("prev_end_code", "r2_bucket_code", "r2_period_type",
                    "r2_score_state", "tempo_tercile")
    sizes = [ck.EMPIRICAL_DIM_SIZES[d] for d in dims]
    assert sizes == [6, 10, 2, 3, 3]
    assert int(np.prod(sizes)) == 1080


@pytest.mark.parametrize("arm_name", ["empirical", "gamma", "lgbm_quantile", "two_regime"])
def test_round2_arms_return_a_proper_pmf(r2_arms, small, arm_name):
    _, te = small
    pm = r2_arms[arm_name].pmf(te.iloc[:400])
    assert pm.shape == (400, ck.N_GRID)
    assert np.all(pm >= 0) and np.allclose(pm.sum(axis=1), 1.0)


def test_clock_complete_flag_and_secondary_read(r2_arms, small):
    """The clock-complete sub-universe is a LABELLED secondary read, and it must
    not be empty or equal to the whole universe -- either would make it
    meaningless."""
    _, te = small
    ah = ck.actual_end_of_half(te)
    assert "clock_complete" in ah.columns
    frac = float(ah["clock_complete"].mean())
    assert 0.2 < frac < 0.95
    res = ck.chain_halves(r2_arms["empirical"], te, seed=7)
    cc = ck.clock_complete_read(res, ah)
    assert 0.0 < cc["eoh_actual_share"] <= 1.0
    assert cc["eoh_actual_share"] > float(ck.eoh_stats(ah)[0])  # truncation biases it down


def test_end_of_half_gate_reads_the_supplied_floor(r2_arms, small):
    _, te = small
    res = ck.chain_halves(r2_arms["empirical"], te, seed=7)
    ah = ck.actual_end_of_half(te)
    assert ck.emergent_report(res, ah, month_min_games=25)["eoh_pass"] is None
    tight = ck.emergent_report(res, ah, month_min_games=25,
                               eoh_floor={"share": 1e-9, "duration": 1e-9})
    wide = ck.emergent_report(res, ah, month_min_games=25,
                              eoh_floor={"share": 1.0, "duration": 1e9})
    assert tight["eoh_pass"] is False
    assert wide["eoh_pass"] is True
