"""Round-9 rotation adapter (`ENGINE_ROTATION=round9`, experiments.md 21.15).

These tests do not touch the served default: every one of them sets
`ENGINE_ROTATION` explicitly and restores the environment, and the served path
(`reference`) is asserted to be what an unset environment still selects.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cbb_sim.engine import rotation_adapter as RA


@pytest.fixture(autouse=True)
def _clean_env():
    keep = {k: os.environ.get(k) for k in
            ("ENGINE_ROTATION", "ENGINE_ROTATION_ARM", "ENGINE_ROTATION_FREEZE",
             "ENGINE_ROT9_AUDIT")}
    for k in keep:
        os.environ.pop(k, None)
    yield
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_served_default_is_still_reference():
    assert RA.rotation_mode() == "reference"
    assert not RA.freeze_enabled()
    assert not RA.round9_audit_on()


def test_round9_arm_selection():
    assert RA.round9_arm() == "Z1"
    os.environ["ENGINE_ROTATION_ARM"] = "Z2"
    assert RA.round9_arm() == "Z2"
    os.environ["ENGINE_ROTATION_ARM"] = "T1"      # a round-6 arm name
    assert RA.round9_arm() == "Z1"                # falls back, never crashes


def test_round9_batch_is_dispatched_before_round6():
    """`Round9Batch` subclasses `Round6Batch`, so the isinstance ladder in
    `next_lineup` must test it FIRST or every round-9 run silently becomes a
    round-6 run."""
    import inspect
    src = inspect.getsource(RA.next_lineup)
    assert src.index("Round9Batch") < src.index("Round6Batch")
    assert issubclass(RA.Round9Batch, RA.Round6Batch)


def test_round9_draw_block_is_two_s_plus_five():
    """The stated RNG divergence from round 6 (21.15): the `rotation_sub` block
    is `2S + 5` wide and the new scalar at `2S + 4` is the `k_out` uniform."""
    import inspect
    s9 = inspect.getsource(RA.next_lineup_round9)
    s6 = inspect.getsource(RA.next_lineup_round6)
    assert 'draw_block("rotation_sub", rows, 2 * S + 5)' in s9
    assert 'draw_block("rotation_sub", rows, 2 * S + 4)' in s6
    assert "u_x = u[:, 2 * S + 3], u[:, 2 * S + 4]" in s9


def test_round6_entry_block_is_unchanged_by_round9():
    """Round 9 replaces the EXIT block only; round 6's K1 entry arithmetic must
    appear verbatim in both."""
    import inspect
    s9 = inspect.getsource(RA.next_lineup_round9)
    s6 = inspect.getsource(RA.next_lineup_round6)
    for line in ("k_in = np.clip(k_in, klo, np.maximum(khi, klo))"
                 .replace("klo", "lo").replace("khi", "hi"),):
        pass
    assert "rb.kin[rb.seg, np.clip(size, 1, 5) - 1" in s9
    assert "rb.kin[rb.seg, np.clip(size, 1, 5) - 1" in s6
    assert "_pick_k(bench_ok & is_st, key_in, k_in)" in s9
    assert "_pick_k(bench_ok & is_st, key_in, k_in)" in s6


# ---------------------------------------------------------------------------
def _offline_k_out(zt, size, n_st_on, n_bn_on, f_st, f_bn, ce, u_x) -> int:
    """`rotation_v8.run_wave8`'s exit block, scalar, copied clip for clip."""
    sz = int(size)
    row = zt[sz - 1, int(ce), int(min(n_st_on, RA.ROUND9_N_ST - 1))].copy()
    lo = max(0, sz - int(n_bn_on), int(f_st))
    hi = min(sz, int(n_st_on), sz - int(f_bn))
    if hi < lo:
        hi = lo = int(min(max(lo, 0), sz))
    row[:lo] = 0.0
    row[hi + 1:] = 0.0
    tot = row.sum()
    if tot <= 0:
        return lo
    k = int(np.searchsorted(np.cumsum(row / tot), float(u_x)))
    return int(min(max(k, lo), hi))


def _vector_k_out(zt, size, n_st_on, n_bn_on, f_st, f_bn, ce, u_x):
    """The adapter's vectorised exit block, lifted out of `next_lineup_round9`
    unchanged, so a divergence between the two is a real divergence."""
    zrow = zt[np.clip(size, 1, 5) - 1, ce, np.minimum(n_st_on, RA.ROUND9_N_ST - 1)]
    lo = np.maximum(np.maximum(0, size - n_bn_on), f_st)
    hi = np.minimum(np.minimum(size, n_st_on), size - f_bn)
    bad = hi < lo
    if bad.any():
        fix = np.minimum(np.maximum(lo, 0), size)
        lo = np.where(bad, fix, lo)
        hi = np.where(bad, fix, hi)
    j = np.arange(zrow.shape[1])[None, :]
    zrow = np.where((j >= lo[:, None]) & (j <= hi[:, None]), zrow, 0.0)
    tot = zrow.sum(axis=1, keepdims=True)
    cdf = np.cumsum(np.where(tot > 0, zrow / np.where(tot > 0, tot, 1.0), 0.0), axis=1)
    k = (u_x[:, None] > cdf).sum(axis=1)
    k = np.where(tot[:, 0] > 0, k, lo)
    return np.clip(k, lo, np.maximum(hi, lo))


def test_exit_block_matches_the_offline_sampler_exactly():
    """Floor A for an identity check is 0: on randomised states over a random
    exit table the vectorised block and the offline scalar block must agree on
    every row, including the degenerate supports."""
    rng = np.random.default_rng(20260911)
    n_cell = 18
    zt = rng.random((5, n_cell, RA.ROUND9_N_ST, 6))
    zt[:, :, :, :] /= zt.sum(axis=-1, keepdims=True)
    zt[3, 4, 2] = 0.0                     # an all-zero row: the `tot <= 0` branch
    m = 20000
    n_st_on = rng.integers(0, 6, m)
    n_bn_on = 5 - n_st_on
    size = rng.integers(1, 6, m)
    f_st = np.minimum(rng.integers(0, 3, m), n_st_on)
    f_bn = np.minimum(rng.integers(0, 3, m), n_bn_on)
    ce = rng.integers(0, n_cell, m)
    u_x = rng.random(m)
    got = _vector_k_out(zt, size, n_st_on, n_bn_on, f_st, f_bn, ce, u_x)
    want = np.array([_offline_k_out(zt, size[i], n_st_on[i], n_bn_on[i],
                                    f_st[i], f_bn[i], ce[i], u_x[i])
                     for i in range(m)])
    assert np.array_equal(got, want)
    assert (got <= size).all() and (got >= 0).all()


def test_audit_hook_is_default_off():
    assert not RA.round9_audit_on()
    os.environ["ENGINE_ROT9_AUDIT"] = "1"
    assert RA.round9_audit_on()
    os.environ["ENGINE_ROT9_AUDIT"] = "0"
    assert not RA.round9_audit_on()


def test_freeze_flag_reads_the_environment():
    assert not RA.freeze_enabled()
    os.environ["ENGINE_ROTATION_FREEZE"] = "1"
    assert RA.freeze_enabled()
    import inspect
    s9 = inspect.getsource(RA.next_lineup_round9)
    # the freeze must zero margin AND both foul counts, for the rotation only
    assert "if rb.freeze:" in s9
    assert "margin = np.zeros_like(margin)" in s9
    assert "f_model = np.zeros_like(f_model)" in s9
    assert "tf_model = np.zeros_like(tf_model)" in s9
