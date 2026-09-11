"""rotation round-3 tests: the keep override.

Three things need a dedicated test beyond what `tests/test_rotation.py` already
asserts for the shared machinery:

1. **The fast keep score is the design matrix.** `_KeepLogistic.p_keep` writes
   the state deviation out term by term instead of building
   `keep_design(...)[:, KEEP_STATE_COLS] @ w`, because it runs once per
   possession per candidate and the column-stack dominates the arm's cost. The
   two must agree to floating-point noise or the fitted coefficients mean
   something different at simulation time than they did at fit time.
2. **The keep override is inert in a neutral state.** Every state column of
   `keep_design` must vanish with no fouls, a tied game and the first half --
   that is what makes the override bite only where it was fitted to bite,
   instead of churning the lineup everywhere (the failure mode `model.md`
   decision 9 records for the first block parameterisation).
3. **Five on the floor survives the keep.** The keep can displace a player the
   donor put on, so the invariant the engine cannot survive a violation of is
   re-asserted per possession for both round-3 arms, including that no
   fouled-out and no unavailable player is ever kept.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import rotation as R  # noqa: E402
from cbb_sim.models import rotation_v3 as V3  # noqa: E402

SEASON = 2024
N_GAMES = 40


@pytest.fixture(scope="module")
def setup():
    tp = R.load_team_possessions(SEASON)
    games = sorted(tp["game_id"].unique())[:N_GAMES]
    tp = tp[tp["game_id"].isin(games)]
    pg = R.player_game_minutes(tp)
    feats = R.build_asof_player_features(pg)
    fit = R.RotationFit(
        season_train=SEASON, role_prior=R.fit_role_prior(pg), k0=2.0, w_dnp=0.5,
        tail_ratio=0.7, n_profile=13, alpha=50.0, alpha_family=28.0,
        alpha_starters=40.0, alpha_bench=9.0, swap_threshold=0.3,
        ema_horizon=360.0, lam_deficit=0.5, foul_rate_scale=1.0, min_share=0.02,
        start_alpha=0.5,
        p_play=[0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.89, 0.84, 0.73, 0.56,
                0.42, 0.28, 0.21, 0.15, 0.12],
        w_avail_dnp=0.4, fpm_league=0.083, fpm_prior_min=150.0)
    fit.tilt = R.fit_tilt_tables(tp, R.add_asof_ranks(feats, fit), None)
    rs = np.random.RandomState(0)
    fit.notes = {
        "r5_exit": {"coef": list(rs.normal(0, 0.3, len(R.OVERRIDE_FEATURES))),
                    "intercept": -3.0},
        "r5_enter": {"coef": list(rs.normal(0, 0.3, len(R.OVERRIDE_FEATURES))),
                     "intercept": -3.5},
        "r5_override_scale": 1.0, "r5_block_base": 0.06, "r5_donor_k": 5,
        "r7_keep": {"coef": list(rs.normal(0, 0.4, len(V3.KEEP_FEATURES))),
                    "intercept": -1.0, "features": list(V3.KEEP_FEATURES)},
        "r7_keep_scale": 2.0, "r7_keep_base": 0.15,
        "r8_state_starter_table": V3.fit_state_starter_table(tp).tolist(),
        "r8_keep_theta": 0.6,
    }
    priors = R.build_priors(feats, fit)
    scripts = R.build_scripts(tp)
    donors = R.build_donor_bank(tp, k_donors=5)
    return fit, priors, scripts, donors


def test_fast_keep_score_matches_the_design_matrix(setup):
    fit, priors, scripts, _ = setup
    keep = V3._KeepLogistic(fit)
    w_full = np.asarray(fit.notes["r7_keep"]["coef"], dtype="float64")
    rs = np.random.RandomState(3)
    checked = 0
    for key in [k for k in scripts if k in priors][:8]:
        prior, script = priors[key], scripts[key]
        n = prior.n
        rem = script.remaining_slot()
        targets = prior.share * script.total_slot
        st_f = np.zeros(n)
        st_f[prior.starters()] = 1.0
        share_v = (targets / max(script.total_slot, 1.0)) * 5.0
        for k in rs.choice(script.n, size=min(12, script.n), replace=False):
            fouls = rs.randint(0, 6, size=n)
            X = V3.keep_design(prior, script, targets, np.zeros(n), fouls,
                               np.zeros(n), np.arange(n), int(k), rem)
            slow = X[:, V3.KEEP_STATE_COLS] @ w_full[V3.KEEP_STATE_COLS]
            slow_p = 1.0 / (1.0 + np.exp(-(keep.scale * slow + keep.keep_logit)))
            fast_p = keep.p_keep(script, int(k), rem, fouls, st_f, share_v,
                                 st_f.astype(bool))
            assert np.allclose(slow_p, fast_p, atol=1e-12), (
                f"{key} possession {k}: fast keep score diverges from keep_design")
            checked += 1
    assert checked >= 50


def test_keep_is_inert_in_a_neutral_state(setup):
    fit, priors, scripts, _ = setup
    key = next(k for k in scripts if k in priors)
    prior, script = priors[key], scripts[key]
    neutral = R.GameScript(**{**script.__dict__,
                              "margin": np.zeros(script.n, dtype="int64"),
                              "margin_bucket": np.zeros(script.n, dtype="int64"),
                              "time_bucket": np.zeros(script.n, dtype="int64"),
                              "period": np.ones(script.n, dtype="int64")})
    n = prior.n
    X = V3.keep_design(prior, neutral, prior.share * neutral.total_slot,
                       np.zeros(n), np.zeros(n, dtype="int64"), np.zeros(n),
                       np.arange(n), 0, neutral.remaining_slot())
    assert np.allclose(X[:, V3.KEEP_STATE_COLS], 0.0), (
        "the keep override is not inert in a neutral state")
    fouls = np.zeros(n, dtype="int64")
    fouls[prior.starters()[:1]] = 4
    X2 = V3.keep_design(prior, neutral, prior.share * neutral.total_slot,
                        np.zeros(n), fouls, np.zeros(n), np.arange(n), 0,
                        neutral.remaining_slot())
    assert not np.allclose(X2[:, V3.KEEP_STATE_COLS], 0.0)


def test_five_on_floor_invariant_with_the_keep(setup):
    fit, priors, scripts, donors = setup
    keys = [k for k in scripts if k in priors][:25]
    assert len(keys) >= 10
    arms = [V3.R5HybridV3(fit, donors),
            V3.R8KeepCell(fit, donors),
            V3.R7KeepLogistic(fit, donors)]
    for arm in arms:
        for key in keys:
            prior, script = priors[key], scripts[key]
            lineups, fouls = arm.simulate(prior, script, R.game_stream(0, key[0]))
            assert lineups.shape == (script.n, 5)
            cand = set(int(p) for p in prior.pids)
            idx = {int(p): i for i, p in enumerate(prior.pids)}
            for k in range(script.n):
                row = [int(x) for x in lineups[k]]
                assert len(set(row)) == 5, f"{arm.name} {key} poss {k}: duplicate"
                assert cand.issuperset(row), f"{arm.name} {key} poss {k}: outside pool"
                for p in row:
                    assert fouls[k, idx[p]] < R.FOUL_OUT, (
                        f"{arm.name} {key} poss {k}: fouled-out player kept on the floor")


def test_theta_zero_reproduces_r5_exactly(setup):
    """R8 with `theta = 0` must be R5, bit for bit apart from the extra tolerance
    draw -- the grid's zero point has to be the baseline or the fit cannot say
    the override earned anything."""
    fit, priors, scripts, donors = setup
    r5 = V3.R5HybridV3(fit, donors)
    r8 = V3.R8KeepCell(fit, donors, theta=0.0)
    for key in [k for k in scripts if k in priors][:10]:
        prior, script = priors[key], scripts[key]
        a, _ = r5.simulate(prior, script, R.game_stream(0, key[0]))
        b, _ = r8.simulate(prior, script, R.game_stream(0, key[0]))
        assert (a == b).all(), f"{key}: theta=0 is not R5"
