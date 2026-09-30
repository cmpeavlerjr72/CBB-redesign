"""season_anchor_serving.py -- serve the season-drift anchor `O` at predict time. DEFAULT OFF.

Lane M, 2026-09-30. Model definition: `cbb_sim.season_anchor` (arm `O`), ruling
`docs/models/season_drift/experiments.md` section 3; evidence
`docs/tests/season_anchor_serving_2026-09-30.md`.

An anchored model is a LightGBM multiclass fitted with an init_score offset
(`link(L_asof) - link(Lbar)`, the as-of in-season league level of its own
target rows). LightGBM's `predict` never adds an init_score back, so a model
served without its offset is a different (wrong) model. This module feeds the
offset:

    ENGINE_SEASON_ANCHOR = off (default) | <path to an offsets .npz>

The .npz is written by `scripts/build_engine_anchor_offsets_v1.py` next to a
tagged input directory, precomputed per GAME (the as-of level on the game's
date), never computed in the sim loop:

    game_ids   (G,)     rows are selected by inp.games.game_id (a missing game is an error)
    po_first   (G, 6)   possession_outcome `first` offset, one per class (log shares), optional
    rb_oreb    (G,)     rebound OREB offset (logit of the live OREB share), optional

A family is anchored iff its array is present. At predict time the adapter adds
the game's offset row to the booster's raw score and takes the softmax, exactly
`cbb_sim.season_anchor.predict_proba_with_offset` (possession_outcome: all six
classes; rebound: OREB column only, as `train_rebound_v3_round3.predict_arm`).
`cont` (cascade) takes no offset: the anchored trainer fits it unchanged.

Rows whose offset row is exactly zero take the model's own `predict_proba`,
so an all-zero offsets file reproduces the unanchored path bit for bit (the
softmax of raw + 0 equals it only to ~1e-16, not bitwise).

Guards: an artifact marked `anchor` (the anchored artifact writers put
`{"anchor": {...}}` in each joblib) is refused when this mode is off, and an
offsets array with any non-zero entry is refused on artifacts NOT marked
anchored. Both errors mean "wrong model served".
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ENV = "ENGINE_SEASON_ANCHOR"
_CACHE: dict[str, "AnchorOffsets"] = {}


def mode() -> str:
    v = os.environ.get(ENV, "off")
    return "off" if v in ("", "off") else v


@dataclass
class AnchorOffsets:
    game_ids: np.ndarray
    po_first: np.ndarray | None
    rb_oreb: np.ndarray | None
    meta: dict = field(default_factory=dict)
    path: str = ""


def load(path: str | os.PathLike) -> AnchorOffsets:
    p = str(Path(path))
    if p not in _CACHE:
        if not Path(p).exists():
            raise FileNotFoundError(f"{ENV}={p}: offsets file missing; build it with "
                                    "scripts/build_engine_anchor_offsets_v1.py")
        z = np.load(p)
        meta_p = Path(p).with_suffix(".json")
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        get = lambda k: (np.asarray(z[k], dtype=np.float64) if k in z.files else None)  # noqa: E731
        _CACHE[p] = AnchorOffsets(game_ids=np.asarray(z["game_ids"], dtype=np.int64),
                                  po_first=get("po_first"), rb_oreb=get("rb_oreb"), meta=meta, path=p)
    return _CACHE[p]


def predict_with_offset(model, X: np.ndarray, off: np.ndarray, cols: list[int]) -> np.ndarray:
    """(n, K) class probabilities of an anchored LightGBM arm (`model.clf_`).

    `off` is (n, len(cols)), placed on raw-score columns `cols`. A row whose
    offset is all zero is predicted by `model.predict_proba` itself."""
    off = np.asarray(off, dtype=np.float64)
    off = off[:, None] if off.ndim == 1 else off
    nz = (off != 0.0).any(axis=1)
    if not nz.any():
        return model.predict_proba(X)
    raw = np.asarray(model.clf_.predict(X[nz], raw_score=True), dtype=np.float64)
    init = np.zeros_like(raw)
    for j, c in enumerate(cols):
        init[:, c] = off[nz, j]
    z = raw + init
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out = np.empty((len(X), raw.shape[1]), dtype=np.float64)
    out[nz] = e / e.sum(axis=1, keepdims=True)
    if (~nz).any():
        out[~nz] = model.predict_proba(X[~nz])
    return out


def _check_classes(models, k: int, label: str) -> None:
    for m in models:
        clf = getattr(m, "clf_", None)
        if clf is None or list(getattr(clf, "classes_", [])) != list(range(k)):
            raise ValueError(f"{ENV}: {label} artifact is not a LightGBM arm with classes 0..{k - 1}; "
                             "the raw-score offset cannot be placed")


def refuse_unserved_anchor(event, reb) -> None:
    """Mode OFF: an artifact marked anchored must not be served without its offset."""
    marks = [("possession_outcome first", any(event.anchor_marks)),
             ("rebound", any(reb.anchor_marks))]
    bad = [n for n, m in marks if m]
    if bad:
        raise ValueError(f"{bad}: artifacts are anchored (season-drift O) but {ENV} is off; "
                         "an anchored model served without its offset is a different model. "
                         f"Set {ENV}=<offsets .npz> (scripts/build_engine_anchor_offsets_v1.py).")


def attach(inp, event, reb, value: str) -> dict:
    """Bind the offsets to the adapters. Returns the provenance block for run_meta."""
    from cbb_sim.models import possession_outcome as PO
    from cbb_sim.models import rebound as RB

    a = load(value)
    gids = inp.games["game_id"].to_numpy().astype(np.int64)
    # Rows are selected BY GAME ID, so a sliced input set (run_engine_live --max-games, a
    # sample) gets exactly its own games' offsets; a game the file lacks is an error.
    pos = {int(g): i for i, g in enumerate(a.game_ids)}
    miss = [int(g) for g in gids if int(g) not in pos]
    if miss:
        raise ValueError(f"{ENV}={value}: {len(miss)} engine game(s) have no offset row (e.g. {miss[:3]})")
    rows = np.array([pos[int(g)] for g in gids], dtype=np.int64)
    src = {"path": a.path, "families": [], "n_games": int(len(gids)),
           "rows_realigned": bool(len(rows) != len(a.game_ids) or not np.array_equal(rows, np.arange(len(rows)))),
           "meta": {k: a.meta.get(k) for k in ("builder", "created_at", "fold", "season", "families")}}
    if a.po_first is not None:
        if a.po_first.shape != (len(a.game_ids), len(PO.CLASSES)) or not np.isfinite(a.po_first).all():
            raise ValueError(f"{ENV}: po_first must be finite (G, {len(PO.CLASSES)})")
        if event.mode != "round2_s1":
            raise ValueError(f"{ENV}: po_first needs ENGINE_EVENT=round2_s1, got {event.mode}")
        po = a.po_first[rows]
        if np.any(po != 0.0):
            if not all(event.anchor_marks):
                raise ValueError(f"{ENV}: non-zero po_first offsets on possession_outcome `first` "
                                 "artifacts that are not marked anchored")
            _check_classes(event.models_first, len(PO.CLASSES), "possession_outcome first")
        event.anchor_first = po
        src["families"].append("po_first")
    if a.rb_oreb is not None:
        if a.rb_oreb.shape != (len(a.game_ids),) or not np.isfinite(a.rb_oreb).all():
            raise ValueError(f"{ENV}: rb_oreb must be finite (G,)")
        if reb.manifest is None:
            raise ValueError(f"{ENV}: rb_oreb needs a dated rebound schedule (ENGINE_REBOUND=s1_weekly)")
        rb = a.rb_oreb[rows]
        if np.any(rb != 0.0):
            if not all(reb.anchor_marks):
                raise ValueError(f"{ENV}: non-zero rb_oreb offsets on rebound artifacts that are "
                                 "not marked anchored")
            _check_classes(reb.models_by_seg, len(RB.CLASSES), "rebound")
        reb.anchor_oreb = rb
        src["families"].append("rb_oreb")
    if not src["families"]:
        raise ValueError(f"{ENV}={value}: the offsets file carries neither po_first nor rb_oreb")
    unfed = [n for n, fam, m in (("possession_outcome first", "po_first", event.anchor_marks),
                                 ("rebound", "rb_oreb", reb.anchor_marks))
             if any(m) and fam not in src["families"]]
    if unfed:
        raise ValueError(f"{ENV}: {unfed} artifacts are anchored but the offsets file has no array for them")
    return src


__all__ = ["ENV", "AnchorOffsets", "attach", "load", "mode", "predict_with_offset",
           "refuse_unserved_anchor"]
