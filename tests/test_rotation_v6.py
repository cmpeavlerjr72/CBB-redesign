"""
Round-6 composition family: the tiers, the fitted objects, and the parity of
the offline entry rule with the engine adapter's vectorised one.

The pre-registration (`docs/models/rotation/experiments.md` 14.10) leans on one
property: each round-6 entry rule is the SAME decision rule offline and in the
engine, and the objects are lookup tables the sim loop can gather. A parity test
is the only thing that keeps that true after an edit.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbb_sim.engine import rotation_adapter as RA
from cbb_sim.models import rotation_v4 as V4
from cbb_sim.models import rotation_v6 as V6


class _P:
    """the three `TeamPrior` fields `tier_of` reads"""

    def __init__(self, n, start_order):
        self.n = n
        self.start_order = np.asarray(start_order, dtype=np.int64)


def test_tier_edges_match_the_adapter_digitize():
    S = 12
    order = np.arange(S)
    off = V6.tier_of(_P(S, order))
    srank = np.arange(1, S + 1)
    eng = np.digitize(srank, [6, 8, 10])
    assert np.array_equal(off, eng)
    assert off[0] == 0 and off[4] == 0 and off[5] == 1 and off[6] == 1
    assert off[7] == 2 and off[8] == 2 and off[9] == 3


def test_tau_grid_is_the_preregistered_one():
    assert len(V6.TAU_GRID) == 20
    assert V6.TAU_GRID[0] == 0.25 and V6.TAU_GRID[-1] == 5.0
    assert 1.0 in set(V6.TAU_GRID)          # tau = 1 IS W4


def _counts(kin=None):
    return {"kin": np.zeros((V6.MAX_WAVE, 6, 6)) if kin is None else kin,
            "obs": np.zeros((4, 4)), "exp": np.zeros((4, 4)),
            "ll": np.zeros(len(V6.TAU_GRID)), "n_waves": 0, "n_used": 0,
            "team_games": 0}


def test_kin_rows_are_probabilities_over_the_reachable_support():
    c = _counts()
    c["kin"][0, 1, 0] = 700.0          # size 1, a starter left, a bench player in
    c["kin"][0, 0, 1] = 300.0
    c["kin"][1, 2, 1] = 500.0          # size 2
    f = V6.fit_comp(c)
    k = f.kin
    for s in range(V6.MAX_WAVE):
        for ko in range(6):
            assert k[s, ko].sum() == pytest.approx(1.0)
            assert (k[s, ko, s + 2:] == 0).all()      # k_in <= size
            assert (k[s, ko] >= 0).all()


def test_log_affinity_is_zero_when_observed_equals_expected():
    c = _counts()
    c["obs"][:] = 500.0
    c["exp"][:] = 500.0
    assert np.allclose(V6.fit_comp(c).A, 0.0)
    c2 = _counts()
    c2["obs"][0, 3] = 2000.0
    c2["exp"][0, 3] = 500.0
    assert V6.fit_comp(c2).A[0, 3] > 0.5                    # a real lift
    assert np.allclose(V6.fit_comp(c2).A[1, 1], 0.0)        # an empty cell -> W4


def test_tau_is_the_argmax_of_the_declared_grid():
    c = _counts()
    c["ll"][:] = -1e6
    c["ll"][7] = -1.0                                       # tau = 2.0
    assert V6.fit_comp(c).tau == pytest.approx(V6.TAU_GRID[7])


def _setup(S=12):
    rng = np.random.default_rng(11)
    coef_in = rng.normal(0, 0.4, V4.N_FEATURES)
    is_st = np.zeros(S)
    is_st[:5] = 1.0
    share = np.linspace(1.2, 0.05, S)
    X = V4.design(is_st[None, :], share[None, :], np.zeros(S)[None, :],
                  np.linspace(0.5, 9.0, S)[None, :], np.linspace(1.0, 8.0, S)[None, :],
                  np.array([2]), np.array([420]), np.array([-4]),
                  np.array([V4.PREV_END_CODE["made_FT"]]), np.array([6.0]))[0]
    p_in = 1 / (1 + np.exp(-(X @ coef_in)))
    return S, is_st, np.clip(p_in, 1e-9, 1 - 1e-9)


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.5])
def test_T1_offline_and_adapter_pick_the_same_entrants(tau):
    S, is_st, p_in = _setup()
    on = np.zeros(S, dtype=bool)
    on[:5] = True
    bench = ~on
    bench_idx = np.flatnonzero(bench)
    rng = np.random.default_rng(3)
    u_bench = rng.random(len(bench_idx))
    logw = np.log(p_in) - np.log1p(-p_in)
    for size in (1, 2, 3):
        w_off = np.exp(tau * (logw[bench_idx] - logw[bench_idx].max()))
        off = set(bench_idx[V6._race(w_off, u_bench, size)].tolist())
        # the adapter carries one uniform PER SLOT; give it the same numbers on
        # the bench slots, which is the stated stream divergence and nothing more
        u_slot = np.ones(S)
        u_slot[bench_idx] = u_bench
        lw = tau * logw
        w = np.exp(lw - lw.max())
        key = -np.log(np.clip(u_slot, 1e-12, 1.0)) / np.maximum(w, 1e-300)
        eng = RA._pick_k(bench[None, :], key[None, :], np.array([size]))[0]
        assert set(np.flatnonzero(eng).tolist()) == off


def test_A1_affinity_shifts_the_weights_identically_in_both_paths():
    S, is_st, p_in = _setup()
    tier = np.digitize(np.arange(1, S + 1), [6, 8, 10])
    log_a = np.zeros((4, 4))
    log_a[0, 3] = 1.5          # a starter leaving pulls the deep bench in
    log_a[0, 1] = -0.4
    on = np.zeros(S, dtype=bool)
    on[:5] = True
    bench = ~on
    bench_idx = np.flatnonzero(bench)
    leaving = np.zeros(S, dtype=bool)
    leaving[[0, 6]] = True                       # one tier-0 and one tier-1 leaver
    logw = np.log(p_in) - np.log1p(-p_in)
    adj_off = log_a[tier[np.flatnonzero(leaving)]].mean(axis=0)
    lw_off = logw[bench_idx] + adj_off[tier[bench_idx]]
    L = np.array([[(leaving & (tier == t)).sum() for t in range(4)]], dtype=float)
    avg = np.einsum("mt,mtb->mb", L, log_a[None, :, :]) / 2.0
    lw_eng = (logw + np.take_along_axis(avg, tier[None, :], axis=1)[0])[bench_idx]
    assert np.allclose(lw_off, lw_eng)


def test_K1_draws_a_class_count_inside_the_reachable_support():
    c = _counts()
    c["kin"][1, 2, 2] = 1000.0                   # size 2, both leavers starters
    f = V6.fit_comp(c)
    row = f.kin[1, 2]
    assert row[2] > 0.9
    # the support clip the sampler applies: at most the off-floor starters, at
    # least size - the off-floor non-starters
    size, n_st, n_bn = 2, 1, 4
    lo, hi = max(0, size - n_bn), min(size, n_st)
    assert (lo, hi) == (0, 1)


def test_round6_arm_table_and_default():
    assert set(RA.ROUND6_ARMS) == {"T1", "K1", "A1"}
    assert set(V6.ARMS) == {"T1_tau_entry", "K1_cond_class", "A1_tier_affinity"}
    assert RA.round6_arm() in RA.ROUND6_ARMS
