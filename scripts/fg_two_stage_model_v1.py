"""fg_two_stage_model_v1.py -- the served-shape object of the two-stage fg_make (fg_make/experiments.md s25).

predict_proba(X): X's LAST column is the stage-B team-game logit offset (`fg_team_offset`); the other columns are
stage A's features in its training order. P(make) = sigmoid(logit(pA) + offset). Classes (MISS, MAKE) like
`FG.LgbmArm`. Pure function of its inputs; no state is read at predict time.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


class TwoStageModel:
    def __init__(self, clf):
        self.clf_ = clf

    def predict_proba(self, X):
        X = np.asarray(X)
        pa = np.clip(self.clf_.predict_proba(X[:, :-1])[:, 1], EPS, 1 - EPS)
        eta = np.log(pa / (1 - pa)) + X[:, -1].astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-eta))
        return np.column_stack([1 - p, p])
