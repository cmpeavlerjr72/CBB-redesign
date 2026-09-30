"""Tests for `cbb_sim.season_anchor` (season-drift anchor O).

The slow test refits the round-1 rebound F1 `O` cell through the module and
compares to the stored prediction (`results/season_drift/round1/preds/`); it
runs only with CBB_SLOW=1 and when that artifact exists (~2-4 min, 1 thread).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbb_sim import season_anchor as SA

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "data/processed/models/rebound/round3/design_round3.parquet"
PRED = ROOT / "results/season_drift/round1/preds/rebound_F1_O_s0.npy"


def test_day0_is_prior_season_end_and_strictly_before():
    season = np.array([2023] * 4 + [2024] * 4)
    date = pd.to_datetime(["2022-11-07", "2022-11-07", "2022-11-08", "2022-11-09",
                           "2023-11-06", "2023-11-06", "2023-11-07", "2023-11-08"])
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
    a = SA.anchor_O(season, date, y, np.ones(8), train_seasons=[2023], kind="binary")
    L = a.levels[:, 0]
    assert L[0] == pytest.approx(0.75)          # 2023 day 0: pooled train (no 2022)
    assert L[2] == pytest.approx(0.5)           # strictly before 11-08: games of 11-07
    assert L[3] == pytest.approx(2 / 3)
    assert L[4] == pytest.approx(0.75) and L[5] == pytest.approx(0.75)  # 2024 day 0 = L_end(2023)
    assert L[6] == pytest.approx(0.0)
    assert np.allclose(a.offset()[:, 0], SA.link(L, "binary") - SA.link(0.75, "binary"))


def test_matches_round1_fitter_offsets():
    if not DESIGN.exists():
        pytest.skip("rebound round-3 design not on disk")
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import exp_season_drift_anchor_v1 as X
    from cbb_sim.models import rebound as RB
    d = pd.read_parquet(DESIGN, columns=["season", "game_date", "y"])
    d = d[d["season"].isin([2022, 2023, 2024, 2025])].reset_index(drop=True)
    num, den = SA.rebound_inputs(d, RB.CLASS_INDEX["OREB"], RB.CLASS_INDEX["DEAD"])
    a = SA.anchor_O(d["season"].to_numpy(), d["game_date"].to_numpy(), num, den,
                    [2022, 2023, 2024], "binary")
    ref, _ = X.build_anchor(d["season"].to_numpy(), d["game_date"].to_numpy(), num, den,
                            [2022, 2023, 2024], "binary")
    assert np.array_equal(a.levels, ref["levels"]["O"])
    assert np.array_equal(a.Lbar, ref["Lbar"])


@pytest.mark.skipif(os.environ.get("CBB_SLOW") != "1" or not PRED.exists() or not DESIGN.exists(),
                    reason="slow refit; set CBB_SLOW=1 with round-1 artifacts on disk")
def test_reproduces_round1_rebound_O_cell():
    import lightgbm as lgb
    from cbb_sim.models import rebound as RB
    feats = RB.feature_set("C_plus_state")
    d = pd.read_parquet(DESIGN, columns=list(dict.fromkeys(feats + ["season", "game_date", "y"])))
    d = d[d["season"].isin([2022, 2023, 2024])].reset_index(drop=True)
    num, den = SA.rebound_inputs(d, RB.CLASS_INDEX["OREB"], RB.CLASS_INDEX["DEAD"])
    a = SA.anchor_O(d["season"].to_numpy(), d["game_date"].to_numpy(), num, den, [2022, 2023], "binary")
    init = SA.lgbm_init_score(a.offset(), 3, [RB.CLASS_INDEX["OREB"]])
    tr = d["season"].isin([2022, 2023]).to_numpy()
    X = d[feats].to_numpy(dtype="float32")
    clf = lgb.LGBMClassifier(random_state=0, **{**RB.LgbmArm.PARAMS, "n_jobs": 1})
    clf.fit(np.ascontiguousarray(X[tr]), d.loc[tr, "y"].to_numpy().astype(int), init_score=init[tr])
    p = SA.predict_proba_with_offset(clf, np.ascontiguousarray(X[~tr]), init[~tr])
    assert np.abs(p - np.load(PRED)).max() < 1e-9
