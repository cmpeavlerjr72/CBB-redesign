"""ops_v3_pair_table_v1.py -- operator 2026-09-30. Paired gate-line table with a MULTI-DRAW floor. Decides nothing.

    python scripts/ops_v3_pair_table_v1.py --dir <dir of eval_gates md> --truth verified|current \
        --ref v3full_S0_s200_o0 --floors v3full_S0f1_s200_o1000,... [--arms v3full_S1_s200_o0,...] --out <md>

Files read: <dir>/<tag>__<truth>.md (scripts/eval_gates.py output, parsed by diag_pair_gate_reports.parse; nothing recomputed).
Per headline gate line (first number of the value cell):
  ref value, each arm value, movement = arm - ref, floor draws = [ref, floors...] -> SD over draws (n-1) and the largest |floor - ref|,
  movement in SD units and in single-draw-floor units (|f1 - ref|), and a flag:
    BEYOND  |movement| > 2 x SD of the multi-draw floor       (a candidate difference)
    inside  otherwise                                          (a non-finding, labelled so)
Status column shows ref status -> arm status. The floor draws are spec-identical reruns of the REFERENCE on other seeds (unpaired).
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_pair_gate_reports as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--truth", required=True, choices=["verified", "current"])
ap.add_argument("--ref", required=True)
ap.add_argument("--floors", required=True)
ap.add_argument("--arms", default="")
ap.add_argument("--labels", default="", help="comma list of display names for arms (default: tags)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = Path(a.dir)
P = lambda t: D.parse(d / f"{t}__{a.truth}.md")
ref = P(a.ref)
floors = [P(t) for t in a.floors.split(",") if t]
arms = [t for t in a.arms.split(",") if t]
labels = a.labels.split(",") if a.labels else arms
armp = [P(t) for t in arms]
num = lambda s: D.first_num(s)

hdr = ["gate", "line", "target", "ref (" + a.ref + ")"]
for lb in labels:
    hdr += [lb, f"{lb} move", f"{lb} / SD", f"{lb} / 1-draw", "flag"]
hdr += ["floor SD (n=%d draws)" % (len(floors) + 1), "max|f-ref|", "status ref -> arms"]
L = [f"# Paired gate table ({a.truth} truth), multi-draw floor from {len(floors)} spec-identical reruns of the reference\n",
     "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for key, r in ref["rows"].items():
    rv = num(r["value"])
    if rv is None:
        continue
    fv = [num(f["rows"][key]["value"]) for f in floors if key in f["rows"]]
    fv = [x for x in fv if x is not None]
    draws = [rv] + fv
    sd = st.stdev(draws) if len(draws) > 2 else float("nan")
    mx = max((abs(x - rv) for x in fv), default=float("nan"))
    one = abs(fv[0] - rv) if fv else float("nan")
    cells = [key[0], key[1], r["target"], r["value"]]
    stat = [r["status"]]
    for p in armp:
        row = p["rows"].get(key)
        av = num(row["value"]) if row else None
        if av is None:
            cells += ["n/a", "", "", "", ""]; stat.append("n/a"); continue
        mv = av - rv
        flag = "BEYOND" if (sd == sd and abs(mv) > 2 * sd) else "inside"
        cells += [row["value"], f"{mv:+.4g}", f"{mv / sd:+.2f}" if sd and sd == sd else "n/a", f"{mv / one:+.2f}" if one and one == one else "n/a", flag]
        stat.append(row["status"])
    cells += [f"{sd:.4g}", f"{mx:.4g}", " -> ".join(stat)]
    L.append("| " + " | ".join(cells) + " |")
Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {a.out} ({len(L) - 3} lines)")
