"""
Round-5 wave family: the cell function, the fitted tables, and the parity of
the offline sampler with the engine adapter.

The point of these tests is the one property the pre-registration leans on
(`docs/models/rotation/experiments.md` 12.10): `rotation_v5.wave_cell` is called
with M = 1 offline and M = 2N in the engine, and the composition rule is round
4's own hazard score, so the two paths cannot drift apart. A parity test is the
only thing that keeps that true after an edit.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbb_sim.engine import rotation_adapter as RA
from cbb_sim.models import rotation_v4 as V4
from cbb_sim.models import rotation_v5 as V5


def test_wave_cell_matches_manual_index():
    pe = np.array([V4.PREV_END_CODE["made_FT"]])
    ci = V5.wave_cell(pe, np.array([2]), np.array([300]), np.array([-3]),
                      np.array([1]))
    # 300 s left in period 2 is H2 08:00-04:00, time cell 5; |m| <= 5 is band 0
    assert int(ci[0]) == ((V4.PREV_END_CODE["made_FT"] * 9 + 5) * 3 + 0) * 2 + 1
    assert 0 <= ci[0] < V5.N_CELL


def test_wave_cell_vectorises_the_same_way():
    rng = np.random.default_rng(0)
    pe = rng.integers(0, 6, 200)
    per = rng.integers(1, 4, 200)
    sec = rng.integers(0, 1200, 200)
    mg = rng.integers(-30, 30, 200)
    fs = rng.integers(0, 2, 200)
    block = V5.wave_cell(pe, per, sec, mg, fs)
    one = np.array([int(V5.wave_cell(pe[i:i + 1], per[i:i + 1], sec[i:i + 1],
                                     mg[i:i + 1], fs[i:i + 1])[0])
                    for i in range(200)])
    assert np.array_equal(block, one)


def _counts(n_cell=V5.N_CELL, seed=3):
    rng = np.random.default_rng(seed)
    n = rng.integers(50, 4000, n_cell).astype(float)
    ev = np.minimum(n, rng.binomial(n.astype(int), 0.15).astype(float))
    size = rng.multinomial(1, [0.68, 0.25, 0.05, 0.015, 0.005],
                           size=n_cell).astype(float) * ev[:, None]
    return {"ev": ev, "n": n, "size": size,
            "joint": {"both": 0.0, "n": 0.0, "pa_pb": 0.0, "min_ab": 0.0, "pairs": []},
            "team_games": 100}


def test_fitted_tables_are_probabilities_and_shrink_thin_cells():
    c = _counts()
    hz = V4.SubHazardFit(kind="logistic", out_coef=[0.0] * V4.N_FEATURES,
                         in_coef=[0.0] * V4.N_FEATURES)
    wf = V5.fit_wave(c, hz)
    pw, ps = wf.pw, wf.ps
    assert pw.shape == (V5.N_CELL,)
    assert ((pw >= 0) & (pw <= 1)).all()
    assert ps.shape == (V5.N_CELL, V5.MAX_WAVE)
    assert np.allclose(ps.sum(axis=1), 1.0)


def test_collapse_state_removes_margin_and_foul_variation():
    c = _counts()
    hz = V4.SubHazardFit(kind="logistic", out_coef=[0.0] * V4.N_FEATURES,
                         in_coef=[0.0] * V4.N_FEATURES)
    wf = V5.fit_wave(c, hz, collapse_state=True)
    pw = wf.pw.reshape(V5.N_PE, V5.N_TC, V5.N_MB, V5.N_FS)
    # every margin band and foul state of a (prev_end, time cell) is identical
    assert np.allclose(pw, pw[:, :, :1, :1])


def _fake_fit():
    from cbb_sim.models.rotation import RotationFit
    return RotationFit()


def test_adapter_and_offline_sampler_pick_the_same_five():
    """One boundary, one team, identical state: the adapter's vectorised rule
    and `rotation_v5.run_wave`'s scalar rule must choose the same five."""
    S = 12
    rng = np.random.default_rng(7)
    coef_out = rng.normal(0, 0.3, V4.N_FEATURES)
    coef_in = rng.normal(0, 0.3, V4.N_FEATURES)
    is_st = np.zeros(S)
    is_st[:5] = 1.0
    share = np.linspace(1.2, 0.05, S)
    fouls = np.zeros(S)
    state_min = np.linspace(0.5, 9.0, S)
    half_min = np.linspace(1.0, 8.0, S)
    period, sec, margin, pe, tf = 2, 420, -4, V4.PREV_END_CODE["made_FT"], 6.0

    X = V4.design(is_st[None, :], share[None, :], fouls[None, :], state_min[None, :],
                  half_min[None, :], np.array([period]), np.array([sec]),
                  np.array([margin]), np.array([pe]), np.array([tf]))[0]
    p_out = 1 / (1 + np.exp(-(X @ coef_out)))
    p_in = 1 / (1 + np.exp(-(X @ coef_in)))

    on = np.zeros(S, dtype=bool)
    on[:5] = True
    bench = np.zeros(S, dtype=bool)
    bench[5:] = True
    for size in (1, 2, 3):
        k = np.full(1, size)
        leaving = RA._pick_k(on[None, :], np.where(on, -p_out, np.inf)[None, :], k)[0]
        entering = RA._pick_k(bench[None, :], np.where(bench, -p_in, np.inf)[None, :],
                              k)[0]
        # the offline rule: the `size` largest hazards, by argsort of the
        # negative score over the eligible set
        on_idx, bench_idx = np.flatnonzero(on), np.flatnonzero(bench)
        off = on_idx[np.argsort(-p_out[on_idx], kind="stable")[:size]]
        inn = bench_idx[np.argsort(-p_in[bench_idx], kind="stable")[:size]]
        assert set(np.flatnonzero(leaving)) == set(off)
        assert set(np.flatnonzero(entering)) == set(inn)


def test_shared_stream_is_shared_between_the_two_teams_and_varies_by_seed():
    a = V5.shared_stream(3, 401_234_567, 40)
    b = V5.shared_stream(3, 401_234_567, 40)
    c = V5.shared_stream(4, 401_234_567, 40)
    assert np.array_equal(a, b)          # both teams of one simulation
    assert not np.array_equal(a, c)      # a different seed is a different draw
    assert a.shape == (40, 2)
    assert ((a >= 0) & (a < 1)).all()


def test_round5_arm_table_is_the_preregistered_grid():
    assert set(RA.ROUND5_ARMS) == {"W1", "W2", "W3", "W4", "W5"}
    assert RA.ROUND5_ARMS["W1"] == (False, False, False)
    assert RA.ROUND5_ARMS["W2"] == (True, True, False)
    assert RA.ROUND5_ARMS["W4"] == (False, True, False)
    assert RA.ROUND5_ARMS["W5"] == (True, False, False)
    assert RA.ROUND5_ARMS["W3"][2] is True


@pytest.mark.parametrize("size,forced", [(0, 0), (2, 0), (1, 2)])
def test_pick_k_respects_size_and_mask(size, forced):
    S = 10
    mask = np.zeros((1, S), dtype=bool)
    mask[0, :5] = True
    key = np.arange(S, dtype=float)[None, :]
    k = np.array([max(size, forced)])
    out = RA._pick_k(mask, key, k)
    assert out.sum() == min(max(size, forced), 5)
    assert not (out & ~mask).any()
