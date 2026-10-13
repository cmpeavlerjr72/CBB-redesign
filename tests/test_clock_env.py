"""test_clock_env.py -- clock round 9 (experiments.md section 41) law module and wiring, synthetic data only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import clock as ck  # noqa: E402

# clock_env imports clock_r8, which registers extra cell grids in clock.EMPIRICAL_DIMS at import time; importing it at
# collection would change what tests/test_clock.py's fixtures iterate over, so every test imports it lazily.


def _CE():
    from cbb_sim.models import clock_env
    return clock_env


def _df(n=6):
    return pd.DataFrame({"off_rating_off_c": np.linspace(-5, 5, n), "off_rating_def_c": np.linspace(2, -2, n),
                         "def_rating_off_c": np.zeros(n), "def_rating_def_c": np.linspace(-1, 1, n),
                         "prev_end": [ck.PREV_END_LEVELS[i % len(ck.PREV_END_LEVELS)] for i in range(n)]})


def test_raw_env_definitions():
    CE = _CE()
    d = _df()
    e, g = CE.raw_env(d)
    assert np.allclose(e, d["off_rating_off_c"] + d["def_rating_def_c"])
    net_o = d["off_rating_off_c"] - d["off_rating_def_c"]
    net_d = d["def_rating_off_c"] - d["def_rating_def_c"]
    assert np.allclose(g, np.abs(net_o - net_d))


class _Flat:
    parametrisation = "P3"

    def pmf(self, df):
        p = np.zeros((len(df), ck.DURATION_CAP + 1))
        p[:, 10:31] = 1.0 / 21
        return p


def test_zero_coef_is_identity_and_positive_coef_lengthens():
    CE = _CE()
    norms = {"sd_off": 8.0, "mu_gap": 5.0, "sd_gap": 4.0}
    d = _df()
    for arm in CE.ARM_SPEC:
        ncoef = CE.design_matrix(CE.add_z(d.copy(), norms), arm).shape[1]
        law0 = CE.EnvAFT(_Flat(), arm, np.zeros(ncoef), norms, "marginal")
        assert np.abs(law0.pmf(d) - _Flat().pmf(d)).max() < 1e-12
    law = CE.EnvAFT(_Flat(), "E1", np.array([0.05]), norms, "marginal")
    p = law.pmf(d)
    assert np.allclose(p.sum(axis=1), 1.0)
    g = np.arange(ck.DURATION_CAP + 1)
    m = p @ g
    z = d["off_rating_off_c"] + d["def_rating_def_c"]
    assert np.all(np.diff(m[np.argsort(z.to_numpy())]) >= -1e-9)


def test_cond_cell_key_separates_end_class():
    CE = _CE()
    from cbb_sim.models import clock_r8 as r8
    d = pd.DataFrame({"prev_end": ["DREB"] * 5, "seconds_remaining": [600.0] * 5, "period": [1] * 5,
                      "score_diff": [0.0] * 5, "tempo_prior_game": [68.0] * 5, "end1_code": np.arange(5)})
    k = CE.cond_cell_key(d, (67.0, 69.0))
    assert len(np.unique(k)) == 5
    assert len(r8.END1_LEVELS) == 5


def test_mode_keys_default_off():
    from cbb_sim.engine import adapters, clock_adapter_v3 as CA
    for a in ("E1", "E2", "E3"):
        assert f"v5b_r8K2_env{a}_glat_pmean" in CA.V5_MODES
        assert f"v3c_r8K2_env{a}_P3_s1" in CA.V3C_MODES
        assert f"v5b_r8K2_env{a}_glat_pmean" not in CA.ADOPTED_MODES
    assert adapters.CLOCK_DEFAULT == "v5b_r8K2_glat_pmean"


def test_engine_frame_carries_the_rating_columns():
    """The adapter's frame is built from TEAM_COLS plus prev_end; the env law reads only those."""
    from cbb_sim.engine import clock_adapter_v3 as CA
    CE = _CE()
    assert set(CE.RATING_COLS) <= set(CA.TEAM_COLS)
