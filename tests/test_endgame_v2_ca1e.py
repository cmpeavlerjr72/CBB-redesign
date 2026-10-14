"""clock experiments.md s49: CA1E law pieces in `cbb_sim.models.endgame_v2` (synthetic)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cbb_sim.models import endgame_v2 as E2


def test_within_cell_env_scale_recovers_coefficient_not_cell_level():
    rng = np.random.default_rng(0)
    n = 40000
    cell = rng.integers(0, 50, n)
    z = rng.normal(size=n) + 0.5 * (cell % 5)          # z correlated with the cell level
    mu = np.exp(1.5 + 0.1 * (cell % 7)) * np.exp(-0.04 * z)
    y = rng.poisson(mu)
    c, _ = E2.fit_env_within(y, z[:, None], cell)
    assert abs(c[0] + 0.04) < 0.01
    se = E2.poisson_fe_cluster_se(y, z[:, None], cell, np.arange(n) // 20, c)
    assert 0 < se[0] < 0.01


def test_ca1e_zero_coef_equals_ca1_and_descale():
    rng = np.random.default_rng(1)
    n = 500
    p0 = rng.random((n, 91))
    p0 /= p0.sum(1, keepdims=True)
    per, sec, m = np.full(n, 2), rng.integers(1, 241, n), rng.integers(-15, 16, n)
    y, cen = rng.integers(0, 40, n), np.zeros(n, bool)
    ca1 = E2.fit_ca1(per, sec, m, y, cen, p0, p0, 50)
    df = pd.DataFrame({"off_rating_off_c": rng.normal(size=n), "def_rating_def_c": rng.normal(size=n)})
    law = E2.CA1ELaw(ca1=ca1, cols=["zoff"], coef=np.zeros(1), info={})
    assert np.allclose(law.apply(p0, per, sec, m, df=df), ca1.apply(p0, per, sec, m), atol=1e-9)
    assert E2.descale(np.array([10, 90, 0]), np.array([2.0, 0.5, 1.0])).tolist() == [5, 90, 0]
    X = E2.env_design(pd.DataFrame({"off_rating_off_c": [1.0, np.nan], "def_rating_def_c": [2.0, 1.0]}), ["zoff", "defz"])
    assert X.tolist() == [[3.0, 2.0], [0.0, 1.0]]
