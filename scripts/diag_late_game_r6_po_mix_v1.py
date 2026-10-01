"""diag_late_game_r6_po_mix_v1.py -- late-game ROUND 6 part 2: is possession_outcome the owner of the leading
team's late outcome mix (fouled vs turnover vs shot)? Reported lines; one code path.

OFFLINE, both folds: round 1's arm-A predictions (`L0_reference` = the served `C_plus_state` first-chance
bundle, S0; seeds 0 and 1 -> the reseed floor) on round 1's held-out window first chances. By offence role band
(trail 4-6, trail 1-3, tied, lead 1-3, lead 4-6) x seconds left ((60,120] (30,60] (10,30] (0,10]): mean predicted
probability of each class vs the realised class frequency, with a game-block bootstrap SE of (pred - actual).
CLOSED LOOP: the served model's probabilities in the sim on window first chances (round-2/3 event tap:
leading / tied / trailing x the same buckets; P(bonus trip), P(three), P(any FGA)) vs the offline prediction
and the actual, so a gap can be split into "model" (offline pred vs actual) and "state fed" (sim vs offline).

    .venv/Scripts/python.exe scripts/diag_late_game_r6_po_mix_v1.py --run lg4_R9_s25 --out results/late_game/round6/po_mix.json
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
CELLS = {"F1": ("c000", "c012"), "F2": ("c024", "c036")}
BANDS = (("trail_4_6", -6, -4), ("trail_1_3", -3, -1), ("tied", 0, 0), ("lead_1_3", 1, 3), ("lead_4_6", 4, 6))
BUCK = (("(60,120]", 60, 120), ("(30,60]", 30, 60), ("(10,30]", 10, 30), ("(0,10]", -1, 10))
K = list(PO.CLASSES)


def offline(fold):
    te = pd.read_parquet(R1 / f"window_test_{fold}.parquet")
    m = (te["population"] == "first").to_numpy() & te["in_window"].to_numpy().astype(bool)
    p0 = np.load(R1 / "pred" / f"{CELLS[fold][0]}.npy")[m]
    p1 = np.load(R1 / "pred" / f"{CELLS[fold][1]}.npy")[m]
    t = te[m].reset_index(drop=True)
    y = t["y"].to_numpy().astype(int)
    Y = np.eye(len(K))[y]
    sd, sec, g = t["score_diff"].to_numpy(), t["seconds_remaining"].to_numpy(), t["game_id"].to_numpy()
    out = {}
    for bn, lo, hi in BANDS:
        for kn, blo, bhi in BUCK:
            c = (sd >= lo) & (sd <= hi) & (sec > blo) & (sec <= bhi)
            n = int(c.sum())
            if not n:
                continue
            row = {"n": n, "label": "UNDERPOWERED" if n < 200 else ""}
            for j, k in enumerate(K):
                d = np.where(c, p0[:, j] - Y[:, j], np.nan)
                row[k] = {"pred": float(p0[c, j].mean()), "pred_seed1": float(p1[c, j].mean()),
                          "act": float(Y[c, j].mean()), "se_gap": block_se(d, g),
                          "seed_floor": float(abs(p0[c, j].mean() - p1[c, j].mean()))}
            out[f"{bn}|{kn}"] = row
    return out


def closed_loop(tag):
    w = np.load(ROOT / "results/engine_v0" / tag / "tap_window.npz")["event"]   # role(0 trail,1 tied,2 lead) x bucket x (n, p_bonus, p_3, p_fga)
    bk = ["(0,10]", "(10,30]", "(30,60]", "(60,120]"]
    out = {}
    for ri, rn in enumerate(("trailing", "tied", "leading")):
        for bi, b in enumerate(bk):
            n = w[ri, bi, 0]
            if n:
                out[f"{rn}|{b}"] = {"n": int(n), "FT_trip_bonus": float(w[ri, bi, 1] / n),
                                    "FGA_3": float(w[ri, bi, 2] / n), "any_FGA": float(w[ri, bi, 3] / n)}
    return out


def pooled_role(off, roles):
    """offline rows pooled to role3 so they line up with the closed-loop tap."""
    return off


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"offline": {f: offline(f) for f in ("F1", "F2")}, "closed_loop": closed_loop(a.run)}
    # offline pooled to role3 x bucket (F2), for the model-vs-state split
    te = pd.read_parquet(R1 / "window_test_F2.parquet")
    m = (te["population"] == "first").to_numpy() & te["in_window"].to_numpy().astype(bool)
    p0 = np.load(R1 / "pred" / "c024.npy")[m]
    t = te[m].reset_index(drop=True)
    Y = np.eye(len(K))[t["y"].to_numpy().astype(int)]
    sd, sec = t["score_diff"].to_numpy(), t["seconds_remaining"].to_numpy()
    pool = {}
    for rn, r in (("trailing", sd < 0), ("tied", sd == 0), ("leading", sd > 0)):
        for kn, blo, bhi in BUCK:
            c = r & (sec > blo) & (sec <= bhi)
            fga = [K.index(x) for x in ("FGA_rim", "FGA_jump2", "FGA_3")]
            pool[f"{rn}|{kn}"] = {"n": int(c.sum()),
                                  "pred": {"FT_trip_bonus": float(p0[c, 5].mean()), "FGA_3": float(p0[c, 3].mean()),
                                           "any_FGA": float(p0[c][:, fga].sum(1).mean()), "TOV": float(p0[c, 0].mean())},
                                  "act": {"FT_trip_bonus": float(Y[c, 5].mean()), "FGA_3": float(Y[c, 3].mean()),
                                          "any_FGA": float(Y[c][:, fga].sum(1).mean()), "TOV": float(Y[c, 0].mean())}}
    res["offline_F2_role3"] = pool
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for f in ("F1", "F2"):
        print(f"== offline {f}: leading / trailing cells, pred vs act (TOV, FT_bonus, FT_shoot, FGA_3), gap in SE")
        for c, r in res["offline"][f].items():
            if not (c.startswith("lead") or c.startswith("trail")):
                continue
            s = "  ".join(f"{k[:8]} {r[k]['pred']:.3f}/{r[k]['act']:.3f} ({(r[k]['pred'] - r[k]['act']) / max(r[k]['se_gap'], 1e-9):+.1f})"
                          for k in ("TOV", "FT_trip_bonus", "FT_trip_shooting", "FGA_3"))
            print(f"  {c:20s} n={r['n']:5d} {s} {r['label']}")
    print("== model vs state (F2 offline pooled to role3 | closed loop | actual): FT_bonus, FGA_3, any_FGA")
    for c, r in pool.items():
        cl = res["closed_loop"].get(c, {})
        print(f"  {c:20s} pred {r['pred']['FT_trip_bonus']:.3f} {r['pred']['FGA_3']:.3f} {r['pred']['any_FGA']:.3f} | "
              f"sim {cl.get('FT_trip_bonus', float('nan')):.3f} {cl.get('FGA_3', float('nan')):.3f} {cl.get('any_FGA', float('nan')):.3f} | "
              f"act {r['act']['FT_trip_bonus']:.3f} {r['act']['FGA_3']:.3f} {r['act']['any_FGA']:.3f}  (TOV pred {r['pred']['TOV']:.3f} act {r['act']['TOV']:.3f}) n={r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
