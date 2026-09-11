"""Round-8 exit-rate object: the composition axis, its shrinkage and its support.

Pre-registration: `docs/models/rotation/experiments.md` section 18.
"""

import numpy as np

from cbb_sim.models import rotation_v7 as V7
from cbb_sim.models import rotation_v8 as V8
from cbb_sim.models.rotation_v5 import MAX_WAVE, N_FS, N_MB


def _counts(rng_seed: int = 0, scale: float = 1.0) -> dict:
    rs = np.random.RandomState(rng_seed)
    c1 = np.zeros((MAX_WAVE, V7.N_EXIT_CELL, MAX_WAVE + 1))
    cy1 = np.zeros((MAX_WAVE, V7.N_EXIT_CELL, V8.N_ST, MAX_WAVE + 1))
    cy2 = np.zeros((MAX_WAVE, V7.N_ETC * N_MB, V7.N_FCLASS, V8.N_ST, MAX_WAVE + 1))
    for s in range(MAX_WAVE):
        for ce in range(V7.N_EXIT_CELL):
            for ns in range(V8.N_ST):
                row = rs.poisson(200 * scale, size=MAX_WAVE + 1).astype(float)
                row[min(ns, s + 1) + 1:] = 0.0
                cy1[s, ce, ns] = row
            c1[s, ce] = cy1[s, ce].sum(axis=0)
        for tm in range(V7.N_ETC * N_MB):
            for fc in range(V7.N_FCLASS):
                for ns in range(V8.N_ST):
                    cy2[s, tm, fc, ns] = rs.poisson(50 * scale, size=MAX_WAVE + 1)
    return {"c1": c1, "cy1": cy1, "cy2": cy2, "n_waves": 10, "n_used": 9,
            "team_games": 3}


def test_shapes_and_normalisation():
    xf = V8.fit_exit8(_counts())
    assert xf.Y1.shape == (MAX_WAVE, V7.N_EXIT_CELL, V8.N_ST, MAX_WAVE + 1)
    assert xf.Y2.shape == (MAX_WAVE, V7.N_ETC * N_MB, V7.N_FCLASS, V8.N_ST,
                           MAX_WAVE + 1)
    assert np.allclose(xf.Y1.sum(axis=-1), 1.0)
    assert np.allclose(xf.Y2.sum(axis=-1), 1.0)
    assert np.allclose(xf.X1.sum(axis=-1), 1.0)


def test_support_clipped_to_size():
    """`k_out` can never exceed the wave size, at any composition (18.3)."""
    xf = V8.fit_exit8(_counts())
    for s in range(MAX_WAVE):
        assert np.all(xf.Y1[s, :, :, s + 2:] == 0.0)
        assert np.all(xf.Y2[s, :, :, :, s + 2:] == 0.0)


def test_y1_empty_cell_is_x1_exactly():
    """A composition cell with no rows shrinks to X1's own row (18.3)."""
    counts = _counts()
    counts["cy1"][0, 3, 2] = 0.0
    xf = V8.fit_exit8(counts)
    assert np.allclose(xf.Y1[0, 3, 2], xf.X1[0, 3])


def test_y2_empty_cell_is_y1_exactly():
    """A foul class with no rows shrinks to Y1's own row at the matching
    foul_state: class 0 -> foul_state 0, classes 1 and 2 -> foul_state 1."""
    counts = _counts()
    counts["cy2"][0, 4, 0, 1] = 0.0
    counts["cy2"][0, 4, 2, 1] = 0.0
    xf = V8.fit_exit8(counts)
    assert np.allclose(xf.Y2[0, 4, 0, 1], xf.Y1[0, 4 * N_FS + 0, 1])
    assert np.allclose(xf.Y2[0, 4, 2, 1], xf.Y1[0, 4 * N_FS + 1, 1])


def test_x1_parent_matches_round7_fit_on_the_same_counts():
    """Round 8's X1 parent is round 7's table by construction, not a re-fit."""
    counts = _counts()
    c7 = {"c1": counts["c1"], "c2": np.zeros((MAX_WAVE, V7.N_ETC * N_MB,
                                              V7.N_FCLASS, MAX_WAVE + 1)),
          "c3": np.zeros((MAX_WAVE, V7.N_EXIT_CELL, V7.N_PCLASS, MAX_WAVE + 1)),
          "n_waves": 10, "n_used": 9, "team_games": 3}
    assert np.allclose(V8.fit_exit8(counts).X1, V7.fit_exit(c7).X1)


def test_composition_axis_moves_the_rate():
    """A count table whose starter exits rise with the composition produces a
    fitted rate that rises with it: the round is identified on this axis."""
    rs = np.random.RandomState(7)
    c1 = np.zeros((MAX_WAVE, V7.N_EXIT_CELL, MAX_WAVE + 1))
    cy1 = np.zeros((MAX_WAVE, V7.N_EXIT_CELL, V8.N_ST, MAX_WAVE + 1))
    cy2 = np.zeros((MAX_WAVE, V7.N_ETC * N_MB, V7.N_FCLASS, V8.N_ST, MAX_WAVE + 1))
    for ce in range(V7.N_EXIT_CELL):
        for ns in range(V8.N_ST):
            n = 5000
            p = ns / 5.0
            k = rs.binomial(n, p)
            cy1[0, ce, ns, 1] = k
            cy1[0, ce, ns, 0] = n - k
        c1[0, ce] = cy1[0, ce].sum(axis=0)
    xf = V8.fit_exit8({"c1": c1, "cy1": cy1, "cy2": cy2, "n_waves": 1, "n_used": 1,
                       "team_games": 1})
    rate = xf.Y1[0, 0, :, 1]
    assert np.all(np.diff(rate) > 0.1)
    assert rate[1] < 0.3 < rate[4]
    # the level form has no such spread: X1's row is the mixture
    assert abs(xf.X1[0, 0, 1] - 0.5) < 0.05


def test_underpowered_threshold_is_the_project_constant():
    assert V8.K_SHRINK == 300.0
    assert V8.N_ST == 6
    assert V8.FOUL_TROUBLE == 4


def test_arm_grid():
    assert set(V8.ARMS) == {"Y1_exit_rate", "Y2_exit_rate_foul"}
    assert V8.ARMS["Y1_exit_rate"].mode == "Y1"
    assert V8.ARMS["Y2_exit_rate_foul"].mode == "Y2"
    assert V8.ARMS["Y1_exit_rate"].simplicity_rank == 14
    assert V8.ARMS["Y2_exit_rate_foul"].simplicity_rank == 15
