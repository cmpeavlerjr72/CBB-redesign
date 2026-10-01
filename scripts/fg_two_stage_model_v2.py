"""fg_two_stage_model_v2.py -- served-shape object of fg_make round TS-R (fg_make/experiments.md s27).

predict_proba(X): the last TWO columns are `lg_make_asof` (the league's as-of make rate on the type) and
`fg_team_offset` (stage B's team-game logit term). The trees score the other columns as a RAW margin, trained with
init_score = logit(league rate), so P(make) = sigmoid(raw + logit(lg_make_asof) + fg_team_offset).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


class TwoStageModelR:
    def __init__(self, clf):
        self.clf_ = clf

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        raw = self.clf_.predict(X[:, :-2], raw_score=True)
        lg = np.clip(X[:, -2], EPS, 1 - EPS)
        eta = raw + np.log(lg / (1 - lg)) + X[:, -1]
        p = 1.0 / (1.0 + np.exp(-eta))
        return np.column_stack([1 - p, p])
