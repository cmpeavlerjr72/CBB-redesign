"""CA1 diagnostic wiring (ENGINE_ENDGAME=CA1, default off; clock s46 block A, labelled diagnostic 2026-10-14)."""
from __future__ import annotations

import numpy as np
import pytest

from cbb_sim.engine import endgame_adapter as EA
from cbb_sim.models import endgame_v2 as E2


def test_parse_mode_accepts_ca1_as_clock_arm():
    assert EA.parse_mode("CA1") == ("CA1", None)
    with pytest.raises(ValueError):
        EA.parse_mode("CA1+GT1")


def test_ca1_law_identity_ratio_keeps_pmf():
    r = np.ones((2, 9, 5, E2.N_BIN))
    law = E2.CA1Law(rm=r, re=r, n=np.zeros((2, 9, 5)), m=100.0, info={})
    p0 = np.random.default_rng(0).dirichlet(np.ones(91), size=4)
    out = law.apply(p0, np.array([2, 2, 2, 2]), np.array([10.0, 50.0, 100.0, 200.0]), np.array([-5, 0, 3, 14]))
    assert np.allclose(out, p0)


def test_ca1_loads_from_artifact_when_present():
    p = EA.ART_V2 / "F2" / "ca1_law.npz"
    if not p.exists():
        pytest.skip("CA1 artifact not synced (HF model_artifacts)")
    law, src = EA.load_law("CA1", "F2")
    assert isinstance(law, E2.CA1Law) and src.endswith("ca1_law.npz")
