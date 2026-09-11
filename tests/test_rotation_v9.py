"""Round-9 exit objects: the hierarchy inverted onto the composition marginal.

Offline, on synthetic counts. Pre-registration: `experiments.md` section 20.
"""

from __future__ import annotations

import numpy as np

from cbb_sim.models.rotation_v5 import MAX_WAVE
from cbb_sim.models.rotation_v7 import N_EXIT_CELL
from cbb_sim.models import rotation_v8 as V8
from cbb_sim.models import rotation_v9 as V9


def _counts(seed: int = 0, scale: float = 1.0) -> dict:
    """Synthetic (size, exit_cell, n_st, k_out) counts with a REAL composition
    gradient and a state split thin enough that the parent matters."""
    rs = np.random.RandomState(seed)
    c1 = np.zeros((MAX_WAVE, N_EXIT_CELL, MAX_WAVE + 1))
    cy1 = np.zeros((MAX_WAVE, N_EXIT_CELL, V9.N_ST, MAX_WAVE + 1))
    for s in range(MAX_WAVE):
        for ce in range(N_EXIT_CELL):
            for ns in range(V9.N_ST):
                # P(a starter leaves) rises with the composition, 19.7's shape
                p = np.clip(ns / 5.0, 0.02, 0.98)
                n = int(scale * (5 + 40 * ns) * (1 + (ce % 3)))
                k = rs.binomial(min(s + 1, max(ns, 0)), p, size=n) if n else np.array([])
                for kk in k:
                    cy1[s, ce, ns, int(kk)] += 1.0
                    c1[s, ce, int(kk)] += 1.0
    return {"c1": c1, "cy1": cy1, "n_waves": 1, "n_used": 1, "team_games": 1}


def test_table_shapes_and_normalisation():
    zf = V9.fit_exit9(_counts(), k=300.0)
    assert zf.Z1.shape == (MAX_WAVE, N_EXIT_CELL, V9.N_ST, MAX_WAVE + 1)
    assert zf.Z2.shape == zf.Z1.shape
    assert zf.M1.shape == (MAX_WAVE, V9.N_ST, MAX_WAVE + 1)
    for t in (zf.Z1, zf.Z2):
        assert np.allclose(t.sum(axis=-1), 1.0)
    assert np.allclose(zf.M1.sum(axis=-1), 1.0)


def test_support_clipped_to_wave_size_at_every_composition():
    zf = V9.fit_exit9(_counts(), k=300.0)
    for s in range(MAX_WAVE):
        assert np.allclose(zf.Z1[s, :, :, s + 2:], 0.0)
        assert np.allclose(zf.Z2[s, :, :, s + 2:], 0.0)
        assert np.allclose(zf.M1[s, :, s + 2:], 0.0)


def test_empty_state_cell_is_the_composition_marginal_exactly():
    """20.3: a state cell with no signal is M1(s, n_st), NOT the level."""
    c = _counts()
    c["cy1"][0, 7, 3] = 0.0
    zf = V9.fit_exit9(c, k=300.0)
    assert np.allclose(zf.Z1[0, 7, 3], zf.M1[0, 3])
    assert np.allclose(zf.Z2[0, 7, 3], zf.M1[0, 3])


def test_z2_gate_is_the_300_row_threshold():
    """Z2 equals Z1 where the cell is powered and M1 exactly where it is not."""
    c = _counts()
    zf = V9.fit_exit9(c, k=300.0)
    n = np.asarray(zf.n_cell)
    powered = n >= V9.MIN_CELL
    assert powered.any() and (~powered).any(), "the fixture must exercise both sides"
    idx = np.argwhere(powered)
    s, ce, ns = idx[0]
    assert np.allclose(zf.Z2[s, ce, ns], zf.Z1[s, ce, ns])
    idx = np.argwhere(~powered)
    s, ce, ns = idx[0]
    assert np.allclose(zf.Z2[s, ce, ns], zf.M1[s, ns])


def test_marginal_is_pooled_over_exit_cells_and_slopes_with_composition():
    """The level-1 parent is the POWERED marginal and carries the gradient."""
    c = _counts()
    zf = V9.fit_exit9(c, k=300.0)
    assert np.allclose(np.asarray(zf.n_marg), c["cy1"].sum(axis=(1, 3)))
    p = zf.M1[0][:, 1]        # single swaps: P(the leaver is a starter)
    assert np.all(np.diff(p[1:6]) > 0), f"marginal must slope in n_st: {p}"


def test_hierarchy_is_inverted_relative_to_round_8():
    """The round-9 parent is the marginal, round 8's was the level, and on a
    thin cell the two give different answers -- the round's whole content."""
    c = _counts()
    c["cy1"][0, 5, 1] *= 0.02          # make one composition cell very thin
    z = V9.fit_exit9(c, k=300.0)
    y = V8.fit_exit8({"c1": c["c1"], "cy1": c["cy1"],
                      "cy2": np.zeros((MAX_WAVE, 9 * 2, 3, V9.N_ST, MAX_WAVE + 1)),
                      "n_waves": 1, "n_used": 1, "team_games": 1}, k=300.0)
    assert not np.allclose(z.Z1[0, 5, 1], y.Y1[0, 5, 1])
    # round 9's thin cell sits on the marginal; round 8's sits on the level
    assert abs(z.Z1[0, 5, 1, 1] - z.M1[0, 1, 1]) < abs(
        z.Z1[0, 5, 1, 1] - y.X1[0, 5, 1])


def test_select_k_is_a_heldout_likelihood_over_the_declared_grid():
    folds = [_counts(seed=i, scale=0.4) for i in range(3)]
    sel = V9.select_k(folds)
    assert sel["k"] in V9.K_GRID
    assert sorted(float(x) for x in sel["scores"]) == sorted(V9.K_GRID)
    assert all(v < 0 for v in sel["scores"].values())          # log-likelihoods
    assert sel["n_folds"] == 3 and sel["tie_nats"] == V9.K_TIE_NATS
    # the argmax is taken, with ties going to the LARGER k
    best = max(sel["scores"].values())
    assert best - sel["scores"][str(sel["k"])] <= V9.K_TIE_NATS


def test_sampler_delegates_to_round_8_with_the_round_9_table():
    """The mechanism and the uniform order are round 8's byte for byte, so a
    round-9 arm driven with Y1's table reproduces Y1 exactly (20.13 item 1)."""
    shim = V9._TableShim(np.zeros((MAX_WAVE, N_EXIT_CELL, V9.N_ST, MAX_WAVE + 1)))
    assert shim.Y1 is shim.Y2
    assert V9.run_wave9.__module__ == "cbb_sim.models.rotation_v9"
    import inspect
    assert "run_wave8" in inspect.getsource(V9.run_wave9)


def test_declared_constants_and_arm_grid():
    assert V9.MIN_CELL == 300
    assert V9.K_GRID == (30.0, 100.0, 300.0, 1000.0, 3000.0)
    assert set(V9.ARMS) == {"Z1_exit_marg", "Z2_exit_interact"}
    assert V9.ARMS["Z1_exit_marg"].simplicity_rank == 16
    assert V9.ARMS["Z2_exit_interact"].simplicity_rank == 17
    assert V9.ARMS["Z1_exit_marg"].mode == "Z1"
    assert V9.ARMS["Z2_exit_interact"].mode == "Z2"
