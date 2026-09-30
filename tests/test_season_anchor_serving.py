"""ENGINE_SEASON_ANCHOR (engine/season_anchor_serving.py): the predict-time offset hook."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cbb_sim import season_anchor as SA
from cbb_sim.engine import season_anchor_serving as SAS

lgb = pytest.importorskip("lightgbm")


class _Arm:
    """Stand-in for PO.LgbmArm / RB.LgbmArm: `clf_` + class-aligned predict_proba."""

    def __init__(self, clf):
        self.clf_ = clf

    def predict_proba(self, X):
        return self.clf_.predict_proba(X)


def _fit(k=3, n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype("float32")
    y = rng.integers(0, k, size=n)
    off = rng.normal(scale=0.1, size=(n, k))
    clf = lgb.LGBMClassifier(n_estimators=8, num_leaves=4, min_child_samples=5, verbose=-1,
                             n_jobs=1, random_state=seed)
    clf.fit(X, y, init_score=off)
    return _Arm(clf), X, rng


def test_offset_equals_offline_formula_all_classes():
    arm, X, rng = _fit(k=6)
    off = rng.normal(scale=0.2, size=(len(X), 6))
    want = SA.predict_proba_with_offset(arm.clf_, X, SA.lgbm_init_score(off, 6, list(range(6))))
    got = SAS.predict_with_offset(arm, X, off, list(range(6)))
    assert np.max(np.abs(got - want)) < 1e-15


def test_offset_one_column_equals_offline_formula():
    arm, X, rng = _fit(k=3)
    o = rng.normal(scale=0.2, size=len(X))
    want = SA.predict_proba_with_offset(arm.clf_, X, SA.lgbm_init_score(o, 3, [0]))
    got = SAS.predict_with_offset(arm, X, o, [0])
    assert np.max(np.abs(got - want)) < 1e-15


def test_zero_offset_is_bit_identical_to_predict_proba():
    arm, X, _ = _fit(k=6)
    got = SAS.predict_with_offset(arm, X, np.zeros((len(X), 6)), list(range(6)))
    assert np.array_equal(got, arm.predict_proba(X))
    # mixed rows: zero rows bit-identical, non-zero rows offset
    off = np.zeros((len(X), 6))
    off[::2, 1] = 0.3
    got = SAS.predict_with_offset(arm, X, off, list(range(6)))
    assert np.array_equal(got[1::2], arm.predict_proba(X)[1::2])
    assert not np.allclose(got[::2], arm.predict_proba(X)[::2])


def _inp(gids):
    return SimpleNamespace(games=pd.DataFrame({"game_id": np.asarray(gids, dtype=np.int64)}))


def test_attach_guards(tmp_path):
    arm, _, _ = _fit(k=6)
    gids = [11, 12, 13]
    p = tmp_path / "off.npz"
    np.savez(p, game_ids=np.array(gids), po_first=np.full((3, 6), 0.1))
    ev = SimpleNamespace(mode="round2_s1", anchor_marks=(False,), models_first=(arm,), anchor_first=None)
    rb = SimpleNamespace(manifest=None, anchor_marks=(), models_by_seg=(), anchor_oreb=None)
    with pytest.raises(ValueError, match="not marked anchored"):
        SAS.attach(_inp(gids), ev, rb, str(p))
    ev.anchor_marks = (True,)
    src = SAS.attach(_inp(gids), ev, rb, str(p))
    assert src["families"] == ["po_first"] and ev.anchor_first.shape == (3, 6)
    with pytest.raises(ValueError, match="game order"):
        SAS.attach(_inp([11, 13, 12]), ev, rb, str(p))
    # zero offsets are allowed on unmarked (served) artifacts: the parity proof
    pz = tmp_path / "zero.npz"
    np.savez(pz, game_ids=np.array(gids), po_first=np.zeros((3, 6)))
    ev2 = SimpleNamespace(mode="round2_s1", anchor_marks=(False,), models_first=(arm,), anchor_first=None)
    SAS.attach(_inp(gids), ev2, rb, str(pz))
    # anchored artifact with the mode off is refused
    with pytest.raises(ValueError, match="served without its offset"):
        SAS.refuse_unserved_anchor(ev, rb)
