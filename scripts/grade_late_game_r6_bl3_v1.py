"""grade_late_game_r6_bl3_v1.py -- late-game ROUND 6 offline grader (experiments.md 13.2 / 13.4). One code path.

    .venv/Scripts/python.exe scripts/grade_late_game_r6_bl3_v1.py --run lg4_R9_s25 --out results/late_game/round6/bl3_grade.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim.models import possession_outcome as PO                  # noqa: E402
from grade_late_game_v1 import block_se                              # noqa: E402

R1 = ROOT / "data/processed/models/late_game/round1"
R6 = ROOT / "data/processed/models/late_game/round6"
CELLS = {"F1": {"A0": "c000", "A1": "c012", "B0": "c003", "B1": None},
         "F2": {"A0": "c024", "A1": "c036", "B0": "c027", "B1": R6 / "bl3_F2_seed1.npy"}}
JB = PO.CLASS_INDEX["FT_trip_bonus"]
BANDS = (("lead_1_3", 1, 3), ("lead_4_6", 4, 6))
BUCK = (("(30,60]", 30, 60), ("(10,30]", 10, 30), ("(0,10]", -1, 10))


def ll(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))


def grade(fold, run):
    te = pd.read_parquet(R1 / f"window_test_{fold}.parquet")
    m = (te["population"] == "first").to_numpy() & te["in_window"].to_numpy().astype(bool)
    t = te[m].reset_index(drop=True)
    y = t["y"].to_numpy().astype(int)
    g = t["game_id"].to_numpy()
    P = {}
    for k, c in CELLS[fold].items():
        if c is None:
            continue
        if isinstance(c, Path) and not c.exists():
            continue
        arr = np.load(c) if isinstance(c, Path) else np.load(R1 / "pred" / f"{c}.npy")
        P[k] = arr[m].astype("float64")
    sd, sec = t["score_diff"].to_numpy(), t["seconds_remaining"].to_numpy()
    foul = (sd > 0) & (sec <= 60)
    L = {k: ll(p, y) for k, p in P.items()}
    out = {"fold": fold, "n_window_first": int(len(t)), "n_foul_cells": int(foul.sum())}
    for scope, msk in (("intentional_foul_cells", foul), ("all_window_first", np.ones(len(t), bool))):
        d = np.where(msk, L["B0"] - L["A0"], np.nan)
        se = block_se(d, g)
        seed_b = float(abs(L["B0"][msk].mean() - L["B1"][msk].mean())) if "B1" in L else None
        seed_a = float(abs(L["A0"][msk].mean() - L["A1"][msk].mean()))
        floor = max(se, seed_b if seed_b is not None else 0.0)
        out[scope] = {"ll_A": float(L["A0"][msk].mean()), "ll_BL3": float(L["B0"][msk].mean()),
                      "delta": float(np.nanmean(d)), "block_se": se, "seed_floor_BL3": seed_b, "seed_floor_A": seed_a,
                      "floor": floor, "floors_better": float(-np.nanmean(d) / floor),
                      "floor_note": None if seed_b is not None else "BL3 seed-1 refit not run on this fold: block SE only"}
    cal = {}
    Y = (y == JB).astype(float)
    for bn, lo, hi in BANDS:
        for kn, blo, bhi in BUCK:
            c = (sd >= lo) & (sd <= hi) & (sec > blo) & (sec <= bhi)
            cal[f"{bn}|{kn}"] = {"n": int(c.sum()), "act": float(Y[c].mean()), "A": float(P["A0"][c, JB].mean()),
                                 "BL3": float(P["B0"][c, JB].mean())}
    out["ft_bonus_calibration"] = cal
    # responsiveness: defence prior late-foul quintile on the foul cells
    q = pd.qcut(t.loc[foul, "def_late_foul_c"].rank(method="first"), 5, labels=False).to_numpy()
    rs = []
    for k in range(5):
        c = np.flatnonzero(foul)[q == k]
        rs.append({"q": k, "n": int(len(c)), "act": float(Y[c].mean()), "A": float(P["A0"][c, JB].mean()),
                   "BL3": float(P["B0"][c, JB].mean())})
    sp = lambda key: rs[-1][key] - rs[0][key]  # noqa: E731
    out["responsiveness"] = {"rows": rs, "span_act": sp("act"), "span_A": sp("A"), "span_BL3": sp("BL3")}
    # 13.4 state-gap attribution (fold 2 only, served-bundle model A)
    if fold == "F2" and run:
        lead30 = (sd > 0) & (sec <= 30)
        b = t["in_bonus"].to_numpy() > 0
        pin, pout = P["A0"][lead30 & b, JB].mean(), P["A0"][lead30 & ~b, JB].mean()
        occ_act = b[lead30].mean()
        tp = pd.read_parquet(ROOT / "results/engine_v0" / run / "tap_poss.parquet")
        if "per" in tp:
            tp = tp[tp["per"] == 2]
        sgn = np.where(tp["off"] == 0, 1, -1)
        om = (tp["hp"] - tp["ap"]).to_numpy() * sgn
        sl = (om >= 1) & (om <= 6) & (tp["sec"].to_numpy() <= 30)
        occ_sim = float(tp.loc[sl, "bon"].mean())
        out["state_attribution"] = {"p_trip_in_bonus": float(pin), "p_trip_not_in_bonus": float(pout),
                                    "occupancy_actual": float(occ_act), "occupancy_sim": occ_sim,
                                    "trip_rate_change_from_occupancy": float((occ_sim - occ_act) * (pin - pout))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lg4_R9_s25")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {f: grade(f, a.run) for f in ("F1", "F2")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for f, r in res.items():
        print(f"== {f}")
        for s in ("intentional_foul_cells", "all_window_first"):
            x = r[s]
            print(f"   {s:24s} ll A {x['ll_A']:.5f} BL3 {x['ll_BL3']:.5f} d {x['delta']:+.5f} block {x['block_se']:.5f} "
                  f"seedB {x['seed_floor_BL3']} seedA {x['seed_floor_A']:.5f} => {x['floors_better']:+.2f} floors")
        for c, v in r["ft_bonus_calibration"].items():
            print(f"   FT_bonus {c:18s} n={v['n']:4d} act {v['act']:.3f} A {v['A']:.3f} BL3 {v['BL3']:.3f}")
        rr = r["responsiveness"]
        print(f"   responsiveness span (q5-q1) act {rr['span_act']:+.3f} A {rr['span_A']:+.3f} BL3 {rr['span_BL3']:+.3f}")
        if "state_attribution" in r:
            print("   state:", r["state_attribution"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
