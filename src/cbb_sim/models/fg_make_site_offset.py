"""
fg_make_site_offset.py -- the artifact class for fg_make arm G4 (Lane G home-site
round, `docs/models/fg_make/experiments.md` sections 21-22). NOT ADOPTED, NOT
SERVED: no engine path loads it unless a default-off flag is added.

G4 = the served B1 trees fitted WITHOUT site columns, plus a site LOGIT OFFSET
identified against team-season fixed effects on the refit's own training rows:

    logit P(make) = trees(x_without_site) + (b_home*site_home + b_away*site_away)
                                            / (pbar * (1 - pbar))

The wrapper takes the declared feature order `features_without_site +
["site_home", "site_away"]` (the last two columns are the site one-hots the
engine's fg team block already carries), so it presents the same
`predict_proba(X) -> (n, 2)` interface as every other dated fg_make artifact.
"""
from __future__ import annotations

import numpy as np


class SiteOffsetModel:
    def __init__(self, clf, n_base: int, b_home: float, b_away: float, pbar: float):
        self.clf_ = clf
        self.n_base = int(n_base)
        self.b_home = float(b_home)
        self.b_away = float(b_away)
        self.pbar = float(pbar)

    def offset(self, site_home: np.ndarray, site_away: np.ndarray) -> np.ndarray:
        return (self.b_home * np.asarray(site_home, float)
                + self.b_away * np.asarray(site_away, float)) / (self.pbar * (1.0 - self.pbar))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        raw = self.clf_.predict(np.ascontiguousarray(X[:, :self.n_base]), raw_score=True)
        z = raw + self.offset(X[:, self.n_base], X[:, self.n_base + 1])
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])
