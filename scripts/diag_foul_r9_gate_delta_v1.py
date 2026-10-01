"""diag_foul_r9_gate_delta_v1.py -- foul round 9 vetoes: every headline gate line of `eval_gates.py` reports, arm vs
the R8b reference on the same games and seeds, judged toward / away from the line's target in units of the
multi-draw floor (Decision 12): floor = SD over the S0 spec-identical reruns given (S0 + its seed-offset draws).
Parses the reports (diag_pair_gate_reports.parse); recomputes nothing.

    diag_foul_r9_gate_delta_v1.py --dir D --ref TAG --arms TAG,TAG --floor-draws TAG,TAG,TAG --out OUT.md
"""
import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_pair_gate_reports as D  # noqa: E402

TARGET_ONE = {"margin SD ratio", "total SD ratio", "calibration slope", "rotation minutes SD ratio"}

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--arms", required=True)
ap.add_argument("--floor-draws", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = Path(a.dir)
P = lambda t: D.parse(d / f"{t}__verified.md")["rows"]  # noqa: E731
ref = P(a.ref)
arms = [t for t in a.arms.split(",") if t]
AR = {t: P(t) for t in arms}
FL = [P(t) for t in a.floor_draws.split(",") if t]
L = [f"Reference `{a.ref}`; floor = SD over {len(FL)} S0 draws ({a.floor_draws}). "
     "toward = |ref - target| - |arm - target| in floor units (positive = closer). AWAY > 2 floors = veto.", "",
     "| gate | line | target | ref | " + " | ".join(f"{t} | toward (fl)" for t in arms) + " | floor SD | verdict |",
     "|---|---|---|---|" + "---|---:|" * len(arms) + "---:|---|"]
for k, r in ref.items():
    v0 = D.first_num(r["value"])
    tg = D.first_num(r["target"]) if D.first_num(r["target"]) is not None else (1.0 if k[1] in TARGET_ONE else None)
    if k[1] == "PIT K-S p" or v0 is None or tg is None:
        continue
    fv = [D.first_num(f[k]["value"]) for f in FL if k in f and D.first_num(f[k]["value"]) is not None]
    sd = st.stdev(fv) if len(fv) > 1 else float("nan")
    cells, verdict = [], "inside"
    for t in arms:
        v = D.first_num(AR[t][k]["value"]) if k in AR[t] else None
        if v is None:
            cells.append("n/a | n/a"); continue
        tw = (abs(v0 - tg) - abs(v - tg)) / sd if sd and sd == sd and sd > 0 else float("nan")
        if tw == tw and tw < -2:
            verdict = "AWAY > 2 fl"
        cells.append(f"{v:.4f} | {tw:+.2f}")
    L.append(f"| {k[0]} | {k[1]} | {tg:g} | {v0:.4f} | " + " | ".join(cells) + f" | {sd:.5f} | {verdict} |")
Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
