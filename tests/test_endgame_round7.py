"""Late-game round 7 (late_game/experiments.md s15): window laws and the default-off wiring."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cbb_sim.engine import endgame_adapter as EA
from cbb_sim.engine import late_foul as LFL
from cbb_sim.models import endgame as EG


def _pmf(n, seed=0):
    p = np.random.default_rng(seed).random((n, 91))
    return p / p.sum(1, keepdims=True)


def test_cells():
    m = np.array([0, 1, 3, 4, 7, 8, 12, 13, 30, -1, -4, -8, -13])
    assert EG.roleband(m).tolist() == [8, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7]
    s = np.array([240, 181, 180, 121, 120, 61, 60, 31, 30, 1, 0])
    assert EG.tbucket(s).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4]
    assert EG.in_window([2, 2, 1, 3], [240, 241, 10, 10], (2,)).tolist() == [True, False, False, False]


def test_gt2_zero_is_identity_and_direction():
    p0 = _pmf(6)
    law = EG.GT2Law(b=np.zeros((2, 2)), info={})
    per, sec, m = np.full(6, 2), np.full(6, 60.0), np.array([15, 15, -15, -15, 0, 0])
    assert np.allclose(law.apply(p0, per, sec, m), p0)
    slow = EG.GT2Law(b=np.array([[0.0, 0.0], [0.3, -0.2]]), info={})
    g = np.arange(91)
    e0, e1 = p0 @ g, slow.apply(p0, per, sec, m) @ g
    assert (e1[:2] > e0[:2]).all() and (e1[2:4] < e0[2:4]).all() and np.allclose(e1[4:], e0[4:])
    # no response at 240 s (taper w = 0)
    assert np.allclose(slow.apply(p0, per, np.full(6, 240.0), m), p0)


def test_gt1_shrinks_to_served_law_when_empty():
    n = 4000
    rng = np.random.default_rng(1)
    per = rng.integers(1, 3, n)
    sec = rng.integers(0, 241, n)
    m = rng.integers(-20, 21, n)
    end1 = rng.integers(0, 5, n)
    y = rng.integers(1, 30, n)
    cen = np.zeros(n, bool)
    p0 = _pmf(n, 2)
    law = EG.fit_gt1(per, sec, m, end1, y, cen, p0, p0)
    q = law.apply(p0[:5], per[:5], sec[:5], m[:5])
    assert np.allclose(q.sum(1), 1.0)
    empty = EG.GT1Law(pb=law.pb, nb=np.zeros_like(law.nb), pc=law.pc, nc=np.zeros_like(law.nc), pbe=law.pbe,
                      nbe=np.zeros_like(law.nbe), pce=law.pce, nce=np.zeros_like(law.nce), m=50.0, info={})
    assert np.allclose(empty.apply(p0[:5], per[:5], sec[:5], m[:5]), p0[:5])
    assert np.allclose(empty.apply_end(p0[:5], per[:5], sec[:5], m[:5], end1[:5]), p0[:5])


def test_ev1_ratio_and_shrink():
    n = 3000
    rng = np.random.default_rng(3)
    per, sec, m = np.full(n, 2), np.full(n, 20.0), np.full(n, 15)
    p0 = np.tile([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], (n, 1))
    y = np.where(rng.random(n) < 0.3, 0, 3)          # TOV over-represented vs p0
    law = EG.fit_ev1(per, sec, m, y, p0)
    q = law.apply(p0[:1], per[:1], sec[:1], m[:1])
    assert np.isclose(q.sum(), 1.0) and q[0, 0] > 0.2
    # an untouched cell keeps the served law exactly
    assert np.allclose(law.apply(p0[:1], [2], [200.0], [-5]), p0[:1])


def test_parse_mode():
    assert EA.parse_mode("GT2") == ("GT2", None)
    assert EA.parse_mode("GT1+EV1") == ("GT1", "EV1")
    with pytest.raises(ValueError):
        EA.parse_mode("GT1+GT2")
    with pytest.raises(ValueError):
        EA.parse_mode("XX")


def test_late_foul_eg_arm_reads_own_dir(tmp_path, monkeypatch):
    d = tmp_path / "F2"
    d.mkdir()
    probs = np.full((5, 3, 4, 4), 0.25).tolist()
    (d / "late_foul_LF2.json").write_text(json.dumps({"probs": probs, "clock_edges": [30, 60, 120],
                                                      "foul_edges": [3, 5, 8]}), encoding="utf-8")
    monkeypatch.setattr(LFL, "EG_DIR", tmp_path)
    monkeypatch.setenv("ENGINE_LATE_FOUL", "eg_LF2")
    lf = LFL.load()
    k = lf.count(np.array([0.1, 0.9, 0.5]), np.array([2, 2, 1]), np.array([10, 10, 10]), np.array([2, 2, 2]),
                 np.array([4, 4, 4]))
    assert k.tolist() == [0, 3, -1]
    monkeypatch.setenv("ENGINE_LATE_FOUL", "off")
    assert LFL.load() is None


def test_flag_off_never_imports(monkeypatch):
    monkeypatch.delenv("ENGINE_ENDGAME", raising=False)
    assert EA.mode_from_env() == "off"
