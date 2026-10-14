"""clock experiments.md s51: CA1R law pieces in `cbb_sim.models.endgame_v2` (synthetic)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cbb_sim.models import endgame_v2 as E2


def _df(rng, n):
    return pd.DataFrame({"off_rating_off_c": rng.normal(size=n), "def_rating_def_c": rng.normal(size=n)})


def test_role_design_and_role_specific_recovery():
    rng = np.random.default_rng(0)
    n = 60000
    m = rng.integers(-15, 16, n)
    df = _df(rng, n)
    X = E2.env_design_role(df, m)
    base = E2.env_design(df, ["zoff", "defz"])
    assert np.allclose(X[m > 0][:, :2], base[m > 0]) and np.all(X[m > 0][:, 2:] == 0)
    assert np.allclose(X[m < 0][:, 2:4], base[m < 0]) and np.allclose(X[m == 0][:, 4:], base[m == 0])
    true = np.array([0.03, 0.0, -0.05, -0.02, 0.0, 0.01])
    cell = E2.roleband(m) * 10 + rng.integers(0, 10, n)        # FE nested in role (as the window cell is)
    mu = np.exp(1.8 + 0.05 * (cell % 9)) * np.exp(X @ true)
    c, _ = E2.fit_env_within(rng.poisson(mu), X, cell)
    assert np.allclose(c, true, atol=0.012)


def test_ca1r_zero_coef_equals_ca1_and_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    n = 400
    p0 = rng.random((n, 91))
    p0 /= p0.sum(1, keepdims=True)
    per, sec, m = np.full(n, 2), rng.integers(1, 241, n), rng.integers(-15, 16, n)
    ca1 = E2.fit_ca1(per, sec, m, rng.integers(0, 40, n), np.zeros(n, bool), p0, p0, 50)
    df = _df(rng, n)
    law = E2.CA1RLaw(ca1=ca1, coef=np.zeros(6), info={})
    assert np.allclose(law.apply(p0, per, sec, m, df=df), ca1.apply(p0, per, sec, m), atol=1e-9)
    law2 = E2.CA1RLaw(ca1=ca1, coef=np.array([0.1, 0.0, -0.1, 0.0, 0.0, 0.0]), info={"x": 1})
    law2.save(tmp_path)
    back = E2.CA1RLaw.load(tmp_path)
    assert np.allclose(back.apply_end(p0, per, sec, m, df=df), law2.apply_end(p0, per, sec, m, df=df))
    # a leader with z_off > 0 is scaled up, a trailer with z_off > 0 is scaled down
    k = np.exp(law2.log_k(df, m))
    z = E2.env_design(df, ["zoff"])[:, 0]
    assert np.all(k[(m > 0) & (z > 0)] > 1) and np.all(k[(m < 0) & (z > 0)] < 1) and np.allclose(k[m == 0], 1)
