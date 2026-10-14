"""clock experiments.md s51: per-role within-cell gates and the s51 decision of `scripts/grade_endgame_clock_v2.py`
(synthetic inputs)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import grade_endgame_clock_v2 as G  # noqa: E402
from test_grade_endgame_clock_s49 import _truth  # noqa: E402


def test_role_within_cell_lines_split_by_role():
    t = _truth(n=9000, seed=3)
    m, z = t["m"].to_numpy(), t["zoff"].to_numpy()
    y = 12 + 2.0 * z * (m > 0) - 1.0 * z * (m < 0)
    L = G.LinesS51(t, 2)
    sa = L.sums(y)
    v = L.lines(sa, np.ones(L.G), sa)
    assert v["wc_zoff_lead"] > 3.0 and v["wc_zoff_trail"] < -1.5
    assert abs(v["wc_zoff_lead"] / v["wc_zoff_trail"] + 2.0) < 0.3
    assert abs(v["wc_zoff_tied"]) < 1e-9          # flat in z_off on tied rows
    assert set(G.S51_WC) <= set(v) and "team_tied" in v
    assert sum(L.role_n.values()) == len(t)


def test_wc_str_trail_is_the_s49_trailer_line():
    t = _truth(seed=4)
    y = 12 + 1.3 * t["strength"].to_numpy() + 0.1 * t["zoff"].to_numpy()
    L = G.LinesS51(t, 2)
    sa = L.sums(y)
    e = L.sums(np.full(len(t), 12.0) + 0.5 * t["strength"].to_numpy())
    v = L.lines(e, np.ones(L.G), sa)
    assert abs(v["wc_str_trail"] - v["wc_trail"]) < 1e-12


def _gate(ok=True):
    return {"pass": ok, "excess": -0.1 if ok else 0.5, "floor": 0.05}


def _F(fail=None, pair=None, beats=True):
    cmp = {a: {p: {"delta": -1.0 if beats else -0.001, "floor": 0.01} for p in ("dev_pos", "cal")}
           for a in ("CA1E", "CA1R")}
    gates = {a: {k: _gate(not (fail and a in fail and k in fail[a])) for k in G.S51_LINES} for a in ("CA1E", "CA1R")}
    p2 = {"compare": cmp, "s49_gates": gates}
    if pair:
        p2["pair_compare"] = pair
    g1 = {a: {"pass": False} for a in ("CA1E", "CA1R")}      # G1 proxy fails but is reported only in s51
    return {"F2": {"periods": {"2": p2}, "g1_sd": g1},
            "F1": {"periods": {"2": {"compare": cmp, "s49_gates": gates}}, "g1_sd": g1}}


def test_s51_decide_role_gate_blocks_and_g1_is_reported_only():
    arms = ["CA0", "CA0s1", "CA1E", "CA1R"]
    d = G.s51_decide(_F(fail={"CA1E": {"team_lead"}, "CA1R": {"wc_defz_tied"}}), arms)
    assert d["verdict"] == "offline refuted" and d["winners"] == []
    assert d["per_arm"]["CA1R"]["gates_failing"] == ["wc_defz_tied"]
    d = G.s51_decide(_F(fail={"CA1E": {"wc_zoff_lead"}}), arms)
    assert d["selected"] == "CA1R"                                    # G1 proxy fail does not block
    assert "g1_sd" not in d["per_arm"]["CA1R"]["gates_pass"]
    assert G.s51_decide(_F(beats=False), arms)["verdict"] == "offline refuted"


def test_s51_tie_break_needs_p1_and_two_wc_role_lines():
    arms = ["CA0", "CA0s1", "CA1E", "CA1R"]
    two = {k: (k in ("wc_zoff_lead", "wc_defz_trail")) for k in G.S51_WC}
    one = {k: (k == "wc_zoff_lead") for k in G.S51_WC}
    win = {"dev_pos_delta": -0.02, "dev_pos_floor": 0.005, "wc_closer_beyond_floor": two}
    assert G.s51_decide(_F(pair=win), arms)["selected"] == "CA1R"
    assert G.s51_decide(_F(pair={**win, "wc_closer_beyond_floor": one}), arms)["selected"] == "CA1E"
    assert G.s51_decide(_F(pair={**win, "dev_pos_delta": -0.004}), arms)["selected"] == "CA1E"
