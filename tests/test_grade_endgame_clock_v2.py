"""clock experiments.md s46: the blind grader `scripts/grade_endgame_clock_v2.py` (synthetic inputs only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import grade_endgame_clock_v2 as G  # noqa: E402


def _pmf(n, seed=0):
    p = np.random.default_rng(seed).random((n, 91))
    return p / p.sum(1, keepdims=True)


def _truth_a(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"game_id": rng.integers(0, 150, n), "period": np.full(n, 2), "poss_index": np.arange(n),
                      "y": rng.integers(1, 30, n), "cen": rng.random(n) < 0.05, "R": rng.integers(1, 241, n).astype(float),
                      "m": rng.integers(-20, 21, n), "tempo": rng.normal(size=n), "strength": rng.normal(size=n),
                      "zoff": rng.normal(size=n), "defz": rng.normal(size=n), "team": rng.integers(0, 10, n),
                      "start": rng.choice(["DREB", "TOV"], n)})
    t["rb"] = G.W.roleband(t["m"].to_numpy())
    t["tb"] = G.W.tbucket(t["R"].to_numpy())
    return t


def test_fps_cell():
    assert G.fps_cell(np.array([-1, -3, -4, -6, 0, -7, 1])).tolist() == [0, 0, 1, 1, 2, -1, -1]


def test_dual_deviance_excludes_control_zero_mass_rows():
    t = _truth_a()
    p = _pmf(len(t))
    p0 = p.copy()
    zero = np.arange(len(t)) < 30
    ii = np.flatnonzero(zero & ~t["cen"].to_numpy())
    p0[ii, t["y"].to_numpy()[ii]] = 0.0                      # control gives zero mass to the observed outcome
    r0, r1 = G.rows_A(t, {"pmf": p0}), G.rows_A(t, {"pmf": p})
    pos = r0["raw"] > G.EPS
    assert (~pos).sum() == len(ii)
    L = G.LinesA(t, 2, pos)
    one = np.ones(L.G)
    v0, v1 = L.lines(L.sums(r0), one), L.lines(L.sums(r1), one)
    # the full-set deviance rewards the arm for smoothing the zero-mass rows; the positive-mass deviance does not see them
    assert v0["dev"] > v1["dev"]
    assert abs(v0["dev_pos"] - v1["dev_pos"]) < abs(v0["dev"] - v1["dev"])
    assert v0["cal45"] >= 0          # synthetic sample: no cell reaches 200 rows, so `cal` is nan by construction


def test_identical_arm_has_zero_delta_and_reseed_floor():
    t = _truth_a(seed=3)
    p = _pmf(len(t), 4)
    rows = {a: G.rows_A(t, {"pmf": p}) for a in ("CA0", "CA0s1", "CA1")}
    L = G.LinesA(t, 2, rows["CA0"]["raw"] > G.EPS)
    sums = {a: L.sums(r) for a, r in rows.items()}
    wb = G.EG.boot_weights(L.G, n_boot=G.N_BOOT)
    val = {a: L.lines(s, np.ones(L.G)) for a, s in sums.items()}
    boot = {a: [L.lines(s, wb[b]) for b in range(G.N_BOOT)] for a, s in sums.items()}
    c = G.compare(val, boot, "CA0", "CA0s1", list(rows))
    fin = [x for x in c["CA1"].values() if np.isfinite(x["arm"])]
    assert fin and all(abs(x["delta"]) < 1e-12 and x["floor"] == 0.0 for x in fin)


def _res_A(delta_pos, delta_cal, tempo_delta=0.0, f1_sign=1.0, floor=0.01):
    def arm(dp, dc, td):
        d = {"dev_pos": {"arm": 6.0 + dp, "delta": dp, "floor": floor},
             "cal": {"arm": 2.0 + dc, "delta": dc, "floor": floor},
             "dev": {"arm": 6.0, "delta": dp, "floor": floor}}
        for s in G.GATES_NOT_BELOW["A"]:
            d[s] = {"arm": 0.4 + (td if s == "slope_tempo" else 0), "delta": td if s == "slope_tempo" else 0.0,
                    "floor": floor}
        for s in G.GATES_ABS1["A"]:
            d[s] = {"arm": 1.0, "delta": 0.0, "floor": floor, "d_abs_err": 0.0, "floor_abs_err": floor}
        return d
    f2 = {"CA0s1": arm(0.0, 0.0, 0.0), "CA1": arm(delta_pos, delta_cal, tempo_delta),
          "CA3": arm(delta_pos * 0.999, delta_cal * 0.999, 0.0)}
    f1 = {k: {p: {"delta": f1_sign * v[p]["delta"]} for p in G.PRIM["A"]} for k, v in f2.items()}
    return {"arms": ["CA0", "CA0s1", "CA1", "CA3"],
            "folds": {"F2": {"periods": {"2": {"compare": f2}}}, "F1": {"periods": {"2": {"compare": f1}}}}}


def test_decide_A_simplest_within_floor_and_tempo_gate():
    d = G.decide("A", _res_A(-0.05, -1.0))
    assert set(d["winners"]) == {"CA1", "CA3"} and d["selected"] == "CA3"
    assert d["per_arm"]["CA0s1"]["wins"] is False
    d = G.decide("A", _res_A(-0.05, -1.0, tempo_delta=-0.5))
    assert "CA1" not in d["winners"] and d["per_arm"]["CA1"]["gates_hold"]["slope_tempo"] is False
    d = G.decide("A", _res_A(-0.05, -1.0, f1_sign=-1.0))
    assert d["winners"] == [] and d["selected"] is None
    d = G.decide("A", _res_A(-0.005, -1.0))                  # dev_pos inside its floor: no win
    assert d["winners"] == []


def test_r4_veto_more_than_two_quintiles():
    rows = {f"envq{k}_poss_gap": {"ctrl": -1.0, "arm": -1.5 if k <= 3 else -1.0, "floor": 0.1} for k in range(1, 6)}
    assert G.r4_veto(rows)["pass"] is False and G.r4_veto(rows)["n_further"] == 3
    rows["envq3_poss_gap"]["arm"] = -1.0
    assert G.r4_veto(rows)["pass"] is True


def test_recode_blinds_arm_keys():
    codes = G.code_arms("A", ["CA0", "CA1", "CA2", "CA3", "CA0s1"])
    assert codes["CA0"] == "CTRL" and len(set(codes.values())) == 5
    x = G.recode({"compare": {"CA1": {"dev": 1}}, "cells": [{"CA2": 3.0}]}, codes)
    assert "CA1" not in x["compare"] and codes["CA1"] in x["compare"] and codes["CA2"] in x["cells"][0]
