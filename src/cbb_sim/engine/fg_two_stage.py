"""fg_two_stage -- the served object of the two-stage fg_make (fg_make experiments.md sections 27 / 29; lane A
2026-10-01). NOT a served default: reached only through a default-off `ENGINE_FG_MAKE=g9ts_<arm>` value.

predict_proba(X): the last two feature columns are `lg_make_asof` (the league's as-of make rate on the shot type)
and `fg_team_offset` (the stage-B team-game logit term, computed pre-game). The trees score the remaining columns
as a raw margin trained with init_score = logit(league rate):
    P(make) = sigmoid(raw + logit(lg_make_asof) + fg_team_offset).
Pure function of its inputs (no state read at predict time).
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
