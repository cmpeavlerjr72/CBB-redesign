"""clock s46 block C feature construction (cbb_sim.models.clock_env_v2; offline arms, not wired)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# clock_env_v2 imports clock_r8 / clock_v3, which register extra cell grids in clock.EMPIRICAL_DIMS at import time;
# importing it at collection changes what tests/test_clock.py's fixtures iterate over, so it is imported lazily
# (as tests/test_clock_env.py does).


def _ce2():
    from cbb_sim.models import clock_env_v2 as CE2
    return CE2


NORMS = {"sd_off": 8.0, "mu_gap": 6.0, "sd_gap": 5.0}


def _pair():
    # one game, two offence rows: home offence then away offence (columns from the offence's view)
    return pd.DataFrame({"off_rating_off_c": [3.0, -1.0], "off_rating_def_c": [-2.0, 4.0],
                         "def_rating_off_c": [-1.0, 3.0], "def_rating_def_c": [4.0, -2.0], "period": [1, 2]})


def test_eg_is_identical_for_both_offences():
    CE2 = _ce2()
    x = CE2.features(_pair(), "EG", NORMS)
    assert x.shape == (2, 1) and x[0, 0] == pytest.approx(x[1, 0])
    assert x[0, 0] == pytest.approx((3 - 2 - 1 + 4) / 16.0)


def test_eh_splits_z_off_by_half_and_em_is_centred_gap():
    CE2 = _ce2()
    eh = CE2.features(_pair(), "EH", NORMS)
    assert eh[0, 1] == 0.0 and eh[1, 0] == 0.0
    assert eh[0, 0] == pytest.approx((3 + 4) / 8.0) and eh[1, 1] == pytest.approx((-1 - 2) / 8.0)
    em = CE2.features(_pair(), "EM", NORMS)
    assert em[0, 0] == pytest.approx((abs((3 + 2) - (-1 - 4)) - 6.0) / 5.0)


def test_q_scores_within_snapshot_range():
    CE2 = _ce2()
    if not list(CE2.RATINGS_DIR.glob("own_ratings_20*.parquet")):
        pytest.skip("own ratings batch files absent")
    t = CE2.q_table()
    assert t["q_off"].between(-0.5, 0.5).all() and t["q_def"].between(-0.5, 0.5).all()
    g = t.groupby(["season", "as_of_date"])["q_off"].mean()
    assert np.allclose(g.to_numpy(), 0.0, atol=1e-9)
    assert int(t["season"].max()) <= 2025
