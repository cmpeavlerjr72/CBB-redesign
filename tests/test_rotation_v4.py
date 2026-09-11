"""
test_rotation_v4.py -- contract tests for the round-4 rotation family.

The round-4 pre-registration (`docs/models/rotation/experiments.md` section 10)
rests on one claim the engine has to honour: `rotation_v4.design()` is written
over (M, S) arrays and is called with M = 1 by the offline sampler and M = 2N by
`engine/rotation_adapter.py`, so the two cannot drift apart. These tests pin
that, the feature order, and the sampler's hard invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbb_sim.models import rotation as R
from cbb_sim.models import rotation_v4 as V4


def _prior(n: int = 12, seed: int = 0) -> R.TeamPrior:
    rng = np.random.default_rng(seed)
    share = np.sort(rng.random(n))[::-1]
    share = share / share.sum()
    srank = np.arange(1, n + 1)
    return R.TeamPrior(
        game_id=1, team_id=7, pids=np.arange(100, 100 + n, dtype="int64"),
        share=share, rank=np.arange(1, n + 1), srank=srank,
        start_order=np.arange(n, dtype="int64"),
        fpm=np.full(n, 0.09), p_avail=np.ones(n), n_prior_games=10)


def _script(npos: int = 140) -> R.GameScript:
    dur = np.full(npos, 17.0)
    period = np.where(np.arange(npos) < npos // 2, 1, 2)
    clock = np.empty(npos)
    for p in (1, 2):
        m = period == p
        clock[m] = np.linspace(1200, 0, int(m.sum()))
    margin = np.round(np.linspace(0, 12, npos))
    return R.GameScript(game_id=1, team_id=7, dur=dur, period=period,
                        start_clock=clock.astype("int64"),
                        margin=margin.astype("int64"),
                        time_bucket=R.time_bucket(period, clock.astype("int64")),
                        margin_bucket=R.margin_bucket(margin.astype("int64")),
                        is_home=True, final_margin=12)


def _fit() -> R.RotationFit:
    f = R.RotationFit(season_train=2024)
    f.foul_rate_scale = 1.0
    return f


def _sub(n_feat: int) -> V4.SubHazardFit:
    rng = np.random.default_rng(3)
    return V4.SubHazardFit(
        kind="logistic",
        out_coef=(rng.normal(size=n_feat) * 0.2).tolist(), out_intercept=-2.5,
        in_coef=(rng.normal(size=n_feat) * 0.2).tolist(), in_intercept=-3.2)


def test_feature_order_is_the_declared_one():
    assert V4.N_FEATURES == len(V4.SUB_FEATURES)
    assert V4.SUB_FEATURES[0] == "is_starter"
    # the design is saturated in (time cell x margin band x is_starter)
    assert sum(f.startswith("tc_") for f in V4.SUB_FEATURES) == V4.N_TIME_CELLS - 1
    assert sum(f.startswith("is_starter_x_tc_") for f in V4.SUB_FEATURES) \
        == V4.N_TIME_CELLS - 1
    assert "period_boundary" in V4.SUB_FEATURES
    assert "is_starter_x_period_boundary" in V4.SUB_FEATURES


def test_design_is_the_same_function_at_M_equals_1_and_M_equals_many():
    rng = np.random.default_rng(11)
    M, S = 6, 13
    args = dict(
        is_starter=(rng.random((M, S)) < 0.4).astype(float),
        share=rng.random((M, S)) * 2.0,
        fouls=rng.integers(0, 5, (M, S)).astype(float),
        state_min=rng.random((M, S)) * 6,
        half_min=rng.random((M, S)) * 18,
        period=np.array([1, 1, 2, 2, 2, 3]),
        sec_left=np.array([1150, 400, 1199, 470, 90, 240]),
        margin=np.array([0, -8, 3, 20, -2, 1]),
        prev_end=np.array([0, 3, 0, 4, 2, 5]),
        team_fouls=np.array([0.0, 4.0, 0.0, 7.0, 9.0, 2.0]),
    )
    big = V4.design(**args)
    assert big.shape == (M, S, V4.N_FEATURES)
    for i in range(M):
        one = V4.design(
            args["is_starter"][i:i + 1], args["share"][i:i + 1], args["fouls"][i:i + 1],
            args["state_min"][i:i + 1], args["half_min"][i:i + 1],
            args["period"][i:i + 1], args["sec_left"][i:i + 1], args["margin"][i:i + 1],
            args["prev_end"][i:i + 1], args["team_fouls"][i:i + 1])
        assert np.allclose(one[0], big[i])


def test_time_cell_matches_the_audit_definition():
    per = np.array([1, 1, 2, 2, 2, 2, 2, 2, 3])
    sec = np.array([1200, 600, 1200, 960, 720, 480, 240, 120, 300])
    assert V4.time_cell(per, sec).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_period_boundary_fires_only_on_the_period_start_code():
    base = dict(is_starter=np.ones((1, 3)), share=np.ones((1, 3)),
                fouls=np.zeros((1, 3)), state_min=np.zeros((1, 3)),
                half_min=np.zeros((1, 3)), period=np.array([2]),
                sec_left=np.array([1200]), margin=np.array([0]),
                team_fouls=np.array([0.0]))
    j = V4.SUB_FEATURES.index("period_boundary")
    for code, name in enumerate(V4.PREV_END_LEVELS):
        X = V4.design(prev_end=np.array([code]), **base)
        assert X[0, 0, j] == (1.0 if name == "period_start" else 0.0)


@pytest.mark.parametrize("hard_reset", [True, False])
def test_sampler_invariants(hard_reset):
    prior, script, fit = _prior(), _script(), _fit()
    sub = _sub(V4.N_FEATURES)
    rng = np.random.default_rng(5)
    avail = np.ones(prior.n, dtype=bool)
    lu, fh = V4.run_sub_hazard(prior, script, avail, sub, fit, rng,
                              hard_reset=hard_reset)
    assert lu.shape == (script.n, 5)
    # five distinct players on the floor at every possession
    for row in lu:
        assert len(set(int(x) for x in row)) == 5
    # nobody plays with five fouls while an eligible substitute exists
    idx = {int(p): i for i, p in enumerate(prior.pids)}
    for k in range(script.n):
        for p in lu[k]:
            assert fh[k][idx[int(p)]] < R.FOUL_OUT


def test_the_hard_reset_puts_the_predicted_starters_back_and_H2_does_not_have_to():
    prior, script, fit = _prior(), _script(), _fit()
    # a hazard that never substitutes voluntarily: only the reset can move the five
    sub = V4.SubHazardFit(kind="logistic",
                          out_coef=[0.0] * V4.N_FEATURES, out_intercept=-40.0,
                          in_coef=[0.0] * V4.N_FEATURES, in_intercept=-40.0)
    avail = np.ones(prior.n, dtype=bool)
    starters = set(int(x) for x in prior.pids[prior.starters()[:5]])
    k0 = int(np.flatnonzero(script.period == 2)[0])
    for hard, expect in ((True, True), (False, True)):
        rng = np.random.default_rng(1)
        lu, _ = V4.run_sub_hazard(prior, script, avail, sub, fit, rng, hard_reset=hard)
        on = set(int(x) for x in lu[k0])
        assert (on == starters) is expect       # nobody ever left, so both hold
    # now a hazard that empties the floor of starters before the half
    sub2 = V4.SubHazardFit(kind="logistic",
                           out_coef=[0.0] * V4.N_FEATURES, out_intercept=2.0,
                           in_coef=[0.0] * V4.N_FEATURES, in_intercept=2.0)
    rng = np.random.default_rng(2)
    lu_h1, _ = V4.run_sub_hazard(prior, script, avail, sub2, fit, rng, hard_reset=True)
    assert set(int(x) for x in lu_h1[k0]) == starters


def test_race_pick_is_weighted_sampling_without_replacement():
    w = np.array([10.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(2000):
        pick = V4._race_pick(w, rng.random(4), 1)
        hits += int(pick[0] == 0)
    assert 0.70 < hits / 2000 < 0.80          # 10 / 13 = 0.769
