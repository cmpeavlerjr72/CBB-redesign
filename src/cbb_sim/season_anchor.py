"""
season_anchor.py -- the season-drift anchor `O`: a model predicts the DEVIATION
from the as-of league level, carried as a link-scale offset at fit AND predict
time. Reference: `docs/models/season_drift/experiments.md` sections 1.2-1.3
(definition), 2 (round-1 results), 3 (PM ruling: `O` enters Stage B as the `TO`
arm with the E3 team features, rebound and possession_outcome).

Definition (arm `O`), per season `s` and game date `t`, for a target with
counts `num` (n, K) over exposure `den` (n,):

    L_asof(s, t) = sum(num over season s, dates < t) / sum(den, same rows)
    L_asof(s, t) = L_end(s-1)   when no earlier game of season s exists (day 0)

`L_end(s-1)` is the previous COMPLETED season's realised level. When the
previous season is not a training season of the fold (the first panel season),
the day-0 value is the fold's pooled TRAIN level `Lbar`. Offsets are centred on
`Lbar`: `offset = link(L_asof) - link(Lbar)` (logit for binary, log for class
shares / Poisson rates), a constant the model absorbs.

Nothing here is served; no engine path imports it. It exists so the Stage B
trainers and any future engine adapter compute the anchor one way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-9
KINDS = ("binary", "multi", "poisson")


def link(L: np.ndarray, kind: str) -> np.ndarray:
    L = np.asarray(L, dtype="float64")
    if kind == "binary":
        L = np.clip(L, 1e-6, 1 - 1e-6)
        return np.log(L / (1 - L))
    if kind in ("multi", "poisson"):
        return np.log(np.clip(L, 1e-12, None))
    raise KeyError(kind)


@dataclass
class AnchorO:
    levels: np.ndarray            # (n, K) L_asof per row
    Lbar: np.ndarray              # (K,) pooled TRAIN level
    kind: str
    meta: dict = field(default_factory=dict)

    def offset(self) -> np.ndarray:
        """(n, K) link-scale offset, centred on Lbar."""
        return link(self.levels, self.kind) - link(self.Lbar[None, :], self.kind)


def _daily(season, date, num, den):
    key = pd.DataFrame({"season": np.asarray(season).astype("int64"),
                        "date": pd.to_datetime(date).values.astype("datetime64[D]")})
    gid = key.groupby(["season", "date"], sort=True).ngroup().to_numpy()
    ng = int(gid.max()) + 1
    K = num.shape[1]
    N = np.column_stack([np.bincount(gid, weights=num[:, k], minlength=ng) for k in range(K)])
    D = np.bincount(gid, weights=den, minlength=ng)
    gs = key.assign(g=gid).drop_duplicates("g").sort_values("g")["season"].to_numpy()
    return gid, gs, N, D


def season_end_levels(season, num, den) -> dict[int, np.ndarray]:
    """Realised level of each season present (use only for COMPLETED seasons)."""
    num = np.asarray(num, dtype="float64").reshape(len(den), -1)
    den = np.asarray(den, dtype="float64")
    s = np.asarray(season).astype("int64")
    return {int(x): num[s == x].sum(axis=0) / max(den[s == x].sum(), EPS) for x in np.unique(s)}


def anchor_O(season, date, num, den, train_seasons: list[int], kind: str) -> AnchorO:
    """Arm `O` for every row of a fold (train + test rows together, so a test
    season's own earlier games feed its as-of level). Reproduces
    `scripts/exp_season_drift_anchor_v1.build_anchor(...)["levels"]["O"]`."""
    if kind not in KINDS:
        raise KeyError(kind)
    num = np.asarray(num, dtype="float64")
    num = num[:, None] if num.ndim == 1 else num
    den = np.asarray(den, dtype="float64")
    gid, gs, N, D = _daily(season, date, num, den)
    Nb, Db = np.zeros_like(N), np.zeros_like(D)
    for s in np.unique(gs):
        m = np.where(gs == s)[0]
        Nb[m[1:]] = np.cumsum(N[m], axis=0)[:-1]
        Db[m[1:]] = np.cumsum(D[m])[:-1]
    tr = np.isin(gs, train_seasons)
    Lbar = N[tr].sum(axis=0) / max(D[tr].sum(), EPS)
    Lend = {int(s): N[gs == s].sum(axis=0) / max(D[gs == s].sum(), EPS) for s in np.unique(gs)}
    prior = np.vstack([Lend[int(s) - 1] if (int(s) - 1) in train_seasons else Lbar for s in gs])
    with np.errstate(invalid="ignore", divide="ignore"):
        L = Nb / Db[:, None]
    empty = Db <= 0
    L[empty] = prior[empty]
    meta = {"Lbar": Lbar.tolist(),
            "prior_by_season": {int(s): (Lend[int(s) - 1] if (int(s) - 1) in train_seasons
                                         else Lbar).tolist() for s in np.unique(gs)}}
    return AnchorO(levels=L[gid], Lbar=Lbar, kind=kind, meta=meta)


def asof_level_live(num_by_date: pd.DataFrame, den_by_date: pd.Series, as_of: pd.Timestamp,
                    prior_end: np.ndarray) -> np.ndarray:
    """Serving form: the anchor for a game on `as_of`, from the current season's
    per-date totals (index = game date) and last season's end level."""
    before = pd.to_datetime(den_by_date.index) < pd.Timestamp(as_of)
    d = float(den_by_date[before].sum())
    if d <= 0:
        return np.asarray(prior_end, dtype="float64")
    return num_by_date[before].sum(axis=0).to_numpy(dtype="float64") / d


# ---------------------------------------------------------------------------
# fit / predict helpers (LightGBM multiclass, the rebound and PO models)
# ---------------------------------------------------------------------------
def lgbm_init_score(offset: np.ndarray, n_classes: int, cols: list[int]) -> np.ndarray:
    """Place the (n, len(cols)) offset into an (n, n_classes) init_score.
    Rebound: cols = [OREB] (DREB is the live reference, DEAD untouched).
    possession_outcome: cols = range(6), one per class."""
    offset = np.asarray(offset, dtype="float64")
    offset = offset[:, None] if offset.ndim == 1 else offset
    out = np.zeros((offset.shape[0], n_classes), dtype="float64")
    for j, c in enumerate(cols):
        out[:, c] = offset[:, j]
    return out


def predict_proba_with_offset(clf, X: np.ndarray, init_score: np.ndarray) -> np.ndarray:
    """LightGBM `predict` never adds an init_score; add it back and softmax."""
    raw = np.asarray(clf.predict(X, raw_score=True), dtype="float64") + init_score
    raw = raw - raw.max(axis=1, keepdims=True)
    e = np.exp(raw)
    return e / e.sum(axis=1, keepdims=True)


def rebound_inputs(design: pd.DataFrame, oreb_idx: int, dead_idx: int):
    """(num, den) for rebound's anchor: OREB share of LIVE (non-dead) misses."""
    y = design["y"].to_numpy()
    live = (y != dead_idx).astype("float64")
    return ((y == oreb_idx).astype("float64") * live)[:, None], live


def po_inputs(design: pd.DataFrame, n_classes: int):
    """(num, den) for possession_outcome's anchor: the class shares."""
    y = design["y"].to_numpy().astype(int)
    num = np.zeros((len(y), n_classes))
    num[np.arange(len(y)), y] = 1.0
    return num, np.ones(len(y))


__all__ = ["AnchorO", "anchor_O", "asof_level_live", "link", "lgbm_init_score",
           "predict_proba_with_offset", "po_inputs", "rebound_inputs", "season_end_levels"]
