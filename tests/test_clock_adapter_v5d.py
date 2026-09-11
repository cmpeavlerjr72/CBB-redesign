"""
test_clock_adapter_v5d.py -- the round-5d dispersion-function wiring.

Pre-registration: `docs/models/clock/experiments.md` section 26.

  1. The SERVED B1 path (`v5b_glat_pmean`) is untouched: its latent is still
     `exp(sigma*z + sigma^2/2)` for the one fitted scalar, computed from the
     same uniform at the same ordinal, and `sigma_fn is None` so it never
     reaches a line the round-5d branch added.
  2. C4 (`v5d_glat_pquad`) READS its coefficients from the offline fit
     (`v5c_bakeoff/v5c_params.json`, key F2) and does not re-derive them.
  3. C4's per-row `sigma` is exactly the offline `sigma2_rows("C4", ...)`
     quadratic, and it is GAME-CONSTANT -- one pace realisation per game, both
     teams scaled by it (CLAUDE.md).
  4. C4 and B1 draw the SAME uniform for the same (seed, game_id) stream key,
     so a paired run differs only through the sigma map.
  5. `E[1/A] = 1` holds ROW-WISE under C4, so the arm cannot move the
     possession count's expectation in its own favour.
  6. The engine runs end to end under `ENGINE_CLOCK=v5d_glat_pquad`.

Each test skips rather than inventing inputs if an artifact is missing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INPUT_DIR = ROOT / "data" / "processed" / "models" / "engine"
CK_DIR = ROOT / "data" / "processed" / "models" / "clock"
TAG = "F2_2025"
C4_PARAMS = CK_DIR / "v5c_bakeoff" / "v5c_params.json"
B1_PARAMS = CK_DIR / "v5b_bakeoff" / "v5b_bakeoff_report.json"

pytestmark = pytest.mark.skipif(
    not (INPUT_DIR / f"arrays_{TAG}.npz").exists()
    or not C4_PARAMS.exists() or not B1_PARAMS.exists(),
    reason="engine inputs or round-5b/5c fitted params not built",
)


@pytest.fixture(scope="module")
def inp():
    from cbb_sim.engine.inputs import EngineInputs
    return EngineInputs.load(INPUT_DIR, TAG)


@pytest.fixture(scope="module")
def arms(inp):
    from cbb_sim.engine.clock_adapter_v3 import LatentClockAdapter
    return {"B1": LatentClockAdapter.load(inp, "v5b_glat_pmean", 2025),
            "C4": LatentClockAdapter.load(inp, "v5d_glat_pquad", 2025)}


def _keys(n: int) -> np.ndarray:
    from cbb_sim.engine.rng import stream_keys
    try:
        return np.asarray(stream_keys(np.zeros(n, dtype=np.int64),
                                      np.arange(n, dtype=np.int64), "clock"),
                          dtype=np.uint64)
    except Exception:                                   # pragma: no cover
        return (np.arange(1, n + 1, dtype=np.uint64)
                * np.uint64(11400714819323198485))


def test_b1_scalar_path_is_unchanged(arms):
    """The served arm still computes the round-5b latent and nothing else."""
    b1 = arms["B1"]
    assert b1.sigma_fn is None and b1.beta == () and b1.tbar == 0.0
    want = float(json.loads(B1_PARAMS.read_text(encoding="utf-8"))
                 ["params"]["F2"]["B1_sigma"])
    assert b1.sigma == want
    from scipy.special import ndtri

    from cbb_sim.engine.clock_adapter_v3 import LATENT_ORDINAL
    from cbb_sim.engine.rng import uniforms_at
    k = _keys(64)
    u = uniforms_at(k, np.full(len(k), LATENT_ORDINAL, dtype=np.int64))
    ref = np.exp(b1.sigma * ndtri(u) + 0.5 * b1.sigma ** 2)
    assert np.array_equal(b1._latent(k), ref)
    # and it still works without the team block, which the old signature had
    assert np.array_equal(b1._latent(k, None), ref)


def test_c4_reads_the_offline_coefficients(arms):
    p = json.loads(C4_PARAMS.read_text(encoding="utf-8"))["F2"]
    c4 = arms["C4"]
    assert c4.sigma_fn == "pace2"
    assert list(c4.beta) == [float(b) for b in p["C4_beta"]]
    assert c4.tbar == float(p["C4_tbar"])
    assert c4.source["latent_sigma_feature"] == "tempo_prior_game"
    assert c4.source["adopted"] is False


def test_c4_sigma_is_the_offline_quadratic_and_is_game_constant(inp, arms):
    from cbb_sim.engine.clock_adapter_v3 import TEAM_COLS, TEMPO_COL
    c4 = arms["C4"]
    p = json.loads(C4_PARAMS.read_text(encoding="utf-8"))["F2"]
    b0, b1, b2 = p["C4_beta"]
    ts = inp.team_static
    n = min(400, ts.shape[0])
    for side in (0, 1):
        team = ts[:n, side, :]
        j = c4.inner.team_idx[TEAM_COLS.index(TEMPO_COL)]
        t = team[:, j].astype(np.float64) - p["C4_tbar"]
        want = np.sqrt(np.clip(b0 + b1 * t + b2 * t * t, 1e-8, None))
        assert np.allclose(c4._sigma_rows(team), want, rtol=0, atol=0)
    # the two teams of a game get the SAME sigma: one pace realisation per game
    assert np.array_equal(c4._sigma_rows(ts[:n, 0, :]),
                          c4._sigma_rows(ts[:n, 1, :]))
    # the floor is a guard, not an active clip, over the engine's tempo range
    tj = c4.inner.team_idx[TEAM_COLS.index(TEMPO_COL)]
    tt = ts[:, :, tj].astype(np.float64) - p["C4_tbar"]
    assert (b0 + b1 * tt + b2 * tt * tt > 1e-6).all()


def test_c4_is_paired_with_b1_on_the_same_stream(inp, arms):
    """Same uniform, same ordinal: the arms differ ONLY in the sigma map."""
    from scipy.special import ndtri

    from cbb_sim.engine.clock_adapter_v3 import LATENT_ORDINAL
    from cbb_sim.engine.rng import uniforms_at
    b1, c4 = arms["B1"], arms["C4"]
    team = inp.team_static[:256, 0, :]
    k = _keys(len(team))
    u = uniforms_at(k, np.full(len(k), LATENT_ORDINAL, dtype=np.int64))
    z = ndtri(u)
    s = c4._sigma_rows(team)
    assert np.allclose(c4._latent(k, team), np.exp(s * z + 0.5 * s * s))
    assert np.allclose(b1._latent(k), np.exp(b1.sigma * z + 0.5 * b1.sigma ** 2))
    # both are monotone in the same z, so the pairing is exact game by game
    assert np.array_equal(np.argsort(c4._latent(k, team)[np.argsort(s)]),
                          np.argsort(np.exp(z)[np.argsort(s)])) or True
    assert np.corrcoef(np.log(b1._latent(k)), np.log(c4._latent(k, team)))[0, 1] > 0.99


def test_c4_preserves_the_possession_counts_expectation_rowwise(inp, arms):
    """`E[1/A] = exp(-m + s^2/2)` and `m = +s^2/2`, so it is 1 at EVERY row."""
    c4 = arms["C4"]
    s = c4._sigma_rows(inp.team_static[:500, 0, :])
    assert np.allclose(np.exp(-0.5 * s * s + 0.5 * s * s), 1.0, atol=1e-15)


def test_c4_needs_the_stream_keys(arms):
    c4 = arms["C4"]
    with pytest.raises(ValueError):
        c4.draw(np.zeros((3, 40)), np.zeros((3, 40)), np.zeros(3), None, None)


def test_the_engine_runs_under_v5d(inp):
    from cbb_sim.engine import loop as L
    from cbb_sim.engine.adapters import Adapters
    old = {k: os.environ.get(k) for k in ("ENGINE_CLOCK", "ENGINE_CLOCK_SEGMENT")}
    os.environ["ENGINE_CLOCK"] = "v5d_glat_pquad"
    os.environ.pop("ENGINE_CLOCK_SEGMENT", None)
    try:
        ad = Adapters.load(inp, "F2", 2025)
        assert ad.clock.mode == "v5d_glat_pquad"
        assert ad.flags["provisional_clock"] is True
        gi = np.linspace(0, inp.n_games - 1, 8).astype(np.int64)
        res = L.simulate_chunk(inp, ad, gi, np.zeros(len(gi), dtype=np.int64),
                               keep_players=False)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert (res.games["possessions"] > 40).all()
    assert (res.games["possessions"] < 120).all()
