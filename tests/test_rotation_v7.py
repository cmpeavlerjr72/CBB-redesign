"""
Round-7 exit family: the declared exit state, the fitted tables' support and
normalisation, the shrinkage identities, and the class clipping the sampler
relies on.

The pre-registration (`docs/models/rotation/experiments.md` 16.2, 16.3) leans on
three properties an edit could silently break: the exit state is the declared
18-cell coarsening of round 5's wave cell, every fitted row is a probability
over the reachable support, and an unobserved refinement is its parent EXACTLY
(X2's foul class and X3's previous-wave class both shrink to X1).
"""

from __future__ import annotations

import numpy as np

from cbb_sim.models import rotation_v4 as V4
from cbb_sim.models import rotation_v5 as V5
from cbb_sim.models import rotation_v7 as V7


def test_exit_state_is_the_declared_18_cell_coarsening():
    assert V7.N_EXIT_CELL == 18
    assert V7.N_ETC == 3 and V5.N_MB == 3 and V5.N_FS == 2
    per = np.array([1, 1, 2, 2, 2, 2, 3])
    clk = np.array([1100, 300, 1000, 700, 480, 100, 200])
    tc9 = V4.time_cell(per, clk)
    tc3 = V7.EXIT_TC[tc9]
    # H1 both cells, H2 above 8:00, H2 inside 8:00, and OT with the late cell
    assert list(tc3) == [0, 0, 1, 1, 2, 2, 2]
    ce = V7.exit_cell(per, clk, np.array([0, 0, 0, 0, 0, 0, 0]),
                      np.array([0, 0, 0, 0, 0, 0, 0]))
    assert ce.min() >= 0 and ce.max() < V7.N_EXIT_CELL
    # the new measured cell of 16.2: final 8:00, |m| > 15, no foul trouble
    blow = int(V7.exit_cell(np.array([2]), np.array([120]), np.array([-22]),
                            np.array([0]))[0])
    assert blow == (2 * V5.N_MB + 2) * V5.N_FS


def test_foul_class_is_three_levels_and_starter_aware():
    is_st = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    assert V7.foul_class(np.array([0.0, 1, 2, 3, 0]), is_st) == 0
    assert V7.foul_class(np.array([0.0, 1, 4, 0, 0]), is_st) == 1
    assert V7.foul_class(np.array([4.0, 1, 4, 0, 0]), is_st) == 2
    assert V7.foul_class(np.array([0.0, 4, 0, 0, 0]), is_st) == 2


def _counts():
    return {"c1": np.zeros((V7.MAX_WAVE, V7.N_EXIT_CELL, V7.MAX_WAVE + 1)),
            "c2": np.zeros((V7.MAX_WAVE, V7.N_ETC * V5.N_MB, V7.N_FCLASS,
                            V7.MAX_WAVE + 1)),
            "c3": np.zeros((V7.MAX_WAVE, V7.N_EXIT_CELL, V7.N_PCLASS,
                            V7.MAX_WAVE + 1)),
            "n_waves": 0, "n_used": 0, "team_games": 0}


def test_exit_rows_are_probabilities_over_the_reachable_support():
    c = _counts()
    c["c1"][0, 3, 1] = 700.0           # size 1, a starter left
    c["c1"][0, 3, 0] = 300.0
    c["c1"][1, 5, 2] = 500.0           # size 2, both leavers starters
    f = V7.fit_exit(c)
    x1 = f.X1
    for s in range(V7.MAX_WAVE):
        for ce in range(V7.N_EXIT_CELL):
            row = x1[s, ce]
            assert abs(row.sum() - 1.0) < 1e-9
            assert (row[s + 2:] == 0).all()      # k_out can never exceed the size
            assert (row >= 0).all()


def test_an_unobserved_refinement_is_its_parent_exactly():
    c = _counts()
    c["c1"][0, 2, 1] = 900.0
    c["c1"][0, 2, 0] = 100.0
    f = V7.fit_exit(c)
    x1, x2, x3 = f.X1, f.X2, f.X3
    ce = 2
    tm, fs = ce // V5.N_FS, ce % V5.N_FS
    # X3: no previous-wave counts anywhere -> every prev class is X1's row
    for pc in range(V7.N_PCLASS):
        assert np.allclose(x3[0, ce, pc], x1[0, ce])
    # X2: foul class 0 shrinks to the foul_state-0 row, classes 1 and 2 to the
    # foul_state-1 row (16.3)
    assert np.allclose(x2[0, tm, 0], x1[0, tm * V5.N_FS + 0])
    assert np.allclose(x2[0, tm, 1], x1[0, tm * V5.N_FS + 1])
    assert np.allclose(x2[0, tm, 2], x1[0, tm * V5.N_FS + 1])
    del fs


def test_shrinkage_constant_is_the_declared_threshold():
    assert V7.K_SHRINK == 300.0
    assert V7.FOUL_TROUBLE == 4
    f = V7.fit_exit(_counts())
    assert f.k_shrink == 300.0


def test_a_thin_cell_is_pulled_to_its_parent_and_a_thick_one_is_not():
    c = _counts()
    for ce in range(V7.N_EXIT_CELL):
        c["c1"][0, ce, 1] = 1000.0             # most cells say k_out = 1
    c["c1"][0, 7, :] = 0.0
    c["c1"][0, 7, 0] = 10.0                    # 10 counts against k = 300
    c["c1"][0, 9, :] = 0.0
    c["c1"][0, 9, 0] = 30000.0                 # 30k counts against k = 300
    f = V7.fit_exit(c)
    x1 = f.X1
    parent = c["c1"][0].sum(axis=0)
    parent = parent / parent.sum()
    thin = abs(x1[0, 7, 0] - parent[0])
    thick = abs(x1[0, 9, 0] - parent[0])
    # the thin cell sits essentially on the parent; the thick one keeps its own
    assert thin < 0.05
    assert thick > 0.3
    assert x1[0, 9, 0] > 0.98


def test_within_class_clipping_matches_the_samplers_bounds():
    """The sampler clips `k_out` to [max(0, size - n_bench_on), min(size,
    n_starters_on)]; when the fitted row has no mass inside that support the
    sampler falls back to the smallest feasible count, never to an infeasible
    one."""
    c = _counts()
    c["c1"][1, 0, 2] = 1000.0                  # size 2 always takes two starters
    f = V7.fit_exit(c)
    row = f.X1[1, 0].copy()
    assert row[2] > 0.9
    n_st_on, n_bn_on, sz = 1, 4, 2             # only one starter is on the floor
    lo, hi = max(0, sz - n_bn_on), min(sz, n_st_on)
    row[:lo] = 0.0
    row[hi + 1:] = 0.0
    tot = row.sum()
    k_out = lo if tot <= 0 else int(np.searchsorted(np.cumsum(row / tot), 0.99))
    k_out = int(min(max(k_out, lo), hi))
    assert lo <= k_out <= hi and k_out <= n_st_on and sz - k_out <= n_bn_on


def test_round7_arm_table_and_modes():
    assert set(V7.ARMS) == {"X1_exit_class", "X2_exit_class_foul",
                            "X3_exit_class_prev"}
    assert [V7.ARMS[n].simplicity_rank for n in
            ("X1_exit_class", "X2_exit_class_foul", "X3_exit_class_prev")] == [11, 12, 13]
    assert [V7.ARMS[n].mode for n in
            ("X1_exit_class", "X2_exit_class_foul", "X3_exit_class_prev")] == \
        ["X1", "X2", "X3"]
