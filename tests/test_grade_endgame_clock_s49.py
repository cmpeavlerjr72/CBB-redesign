"""clock experiments.md s49: actual-referenced gate lines of `scripts/grade_endgame_clock_v2.py` (synthetic inputs)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import grade_endgame_clock_v2 as G  # noqa: E402


def _truth(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"game_id": rng.integers(0, 200, n), "period": np.full(n, 2), "poss_index": np.arange(n),
                      "cen": np.zeros(n, bool), "R": rng.integers(1, 241, n).astype(float),
                      "m": rng.integers(-20, 21, n), "tempo": rng.normal(size=n), "strength": rng.normal(size=n),
                      "zoff": rng.normal(size=n), "defz": rng.normal(size=n)})
    t["rb"] = G.W.roleband(t["m"].to_numpy())
    t["tb"] = G.W.tbucket(t["R"].to_numpy())
    return t


def test_slope_lines_recover_known_response():
    t = _truth()
    # seconds = 12 + 2 * z_off exactly: the z_off quintile slope is 2 s per unit and the within-cell Q5 - Q1 is the
    # whole spread (cells carry no z_off composition by construction)
    y = 12 + 2.0 * t["zoff"].to_numpy()
    L = G.LinesS49(t, 2)
    sa = L.sums(y)
    v = L.lines(sa, np.ones(L.G), sa)
    assert abs(v["zoff"] - 2.0) < 1e-9
    assert abs(v["tempo"]) < 0.3
    flat = L.lines(L.sums(np.full(len(t), 12.0)), np.ones(L.G), sa)
    assert abs(flat["zoff"]) < 1e-12
    # a flat model's within-cell z_off spread is minus the between term; actual minus model = actual total spread
    assert v["wc_zoff"] - flat["wc_zoff"] > 3.0


def test_band_lines_use_margin():
    t = _truth(seed=1)
    m = t["m"].to_numpy()
    y = 10 + 0.5 * np.abs(m) * (m > 0) - 0.2 * np.abs(m) * (m < 0)
    L = G.LinesS49(t, 2)
    sa = L.sums(y)
    v = L.lines(sa, np.ones(L.G), sa)
    assert abs(v["band_lead"] - 0.5) < 1e-9
    assert abs(v["band_trail"] + 0.2) < 1e-9


def test_gate_actual_referenced_and_floor_includes_actual_se():
    rng = np.random.default_rng(0)
    Bc = rng.normal(0, 0.01, 200)
    Ba = Bc + rng.normal(0, 0.01, 200)
    Bact = rng.normal(0, 0.05, 200)
    g = G.s49_gate(La=-0.02, Lc=0.0, Lact=-0.03, Ls1=0.0, Ba=Ba, Bc=Bc, Bact=Bact)
    assert g["excess"] < 0 and g["pass"]                       # arm closer to actual
    assert g["floor"] >= g["se_actual"] and g["floor"] >= g["paired_floor"]
    assert abs(g["floor"] - np.sqrt(g["paired_floor"] ** 2 + g["se_actual"] ** 2)) < 1e-12
    # an arm on the far side of the actual, further than the control, beyond the floor fails
    bad = G.s49_gate(La=-1.0, Lc=0.0, Lact=-0.03, Ls1=0.0, Ba=Ba, Bc=Bc, Bact=Bact)
    assert not bad["pass"]
    # the retired "not below control" form would have failed this arm; the actual-referenced form passes it
    ok = G.s49_gate(La=-0.05, Lc=0.0, Lact=-0.04, Ls1=0.0, Ba=Ba, Bc=Bc, Bact=Bact)
    assert ok["pass"]


def test_cell_lines_per_cell_slope():
    t = _truth(seed=2)
    y = 12 + 1.5 * t["strength"].to_numpy()
    L = G.LinesS49(t, 2)
    c = L.cell_lines(L.sums(y))
    ok = c["n"] > 20
    assert np.allclose(c["slope"][ok], 1.5)


def _row(arm, ctrl, floor):
    return {"arm": arm, "ctrl": ctrl, "floor": floor, "delta": arm - ctrl}


def test_s49_full_verdict_needs_both_primaries_and_not_past(monkeypatch):
    act = {"act_MD5_pts13_last2": 7.0, "act_lsec_r2_13p": 16.0, "act_tie_rate": 0.068,
           "act_MD4_h2_last4_poss": 15.4}
    rows = {"MD5_pts13_last2": _row(9.0, 9.7, 0.2), "lsec_r2_13p": _row(14.2, 12.7, 0.1),
            "tie_rate": _row(0.042, 0.036, 0.004), "MD4_h2_last4_poss": _row(16.75, 16.75, 0.08),
            "MD1_gap": _row(15.0, 15.5, 2.5), "MD6_margin_sd_ratio": _row(1.009, 1.009, 0.02),
            **{f"MD2_q{i}": _row(1.0, 1.0, 0.05) for i in range(1, 6)}}
    monkeypatch.setattr(G, "judge_loop", lambda b, r, a, c: {"vetoes": {}, "reported": {}, "md": {}})
    v = G.s49_judge(rows, act, {}, "full")
    assert v["full_verdict"] == "CANDIDATE"
    rows["lsec_r2_13p"] = _row(19.0, 12.7, 0.1)               # past the actual beyond floor
    assert G.s49_judge(rows, act, {}, "full")["full_verdict"] == "NOT A CANDIDATE"
    rows["lsec_r2_13p"] = _row(14.2, 12.7, 0.1)
    rows["MD4_h2_last4_poss"] = _row(17.5, 16.75, 0.08)        # must-not-regress line regresses
    assert G.s49_judge(rows, act, {}, "full")["full_verdict"] == "NOT A CANDIDATE"


def test_s49_screen_any_one_line(monkeypatch):
    act = {"act_MD5_pts13_last2": 7.0, "act_lsec_r2_13p": 16.0, "act_tie_rate": 0.068,
           "act_MD4_h2_last4_poss": 15.4}
    rows = {"MD5_pts13_last2": _row(9.7, 9.7, 0.2), "lsec_r2_13p": _row(12.7, 12.7, 0.1),
            "tie_rate": _row(0.045, 0.036, 0.004), "MD4_h2_last4_poss": _row(16.75, 16.75, 0.08),
            "MD1_gap": _row(15.0, 15.5, 2.5), "MD6_margin_sd_ratio": _row(1.009, 1.009, 0.02),
            **{f"MD2_q{i}": _row(1.0, 1.0, 0.05) for i in range(1, 6)}}
    monkeypatch.setattr(G, "judge_loop", lambda b, r, a, c: {"vetoes": {}, "reported": {}, "md": {}})
    assert G.s49_judge(rows, act, {}, "loop")["screen"] == "proceed"
    rows["tie_rate"] = _row(0.038, 0.036, 0.004)               # toward but inside floor -> rerun at 50
    assert G.s49_judge(rows, act, {}, "loop")["screen"] == "stop (rerun at 50 seeds)"
    rows["tie_rate"] = _row(0.045, 0.036, 0.004)
    rows["MD2_q1"] = _row(1.2, 1.0, 0.05)                      # MD regression vetoes
    assert G.s49_judge(rows, act, {}, "loop")["screen"] == "stop"
