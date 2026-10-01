#!/usr/bin/env python
"""grade_shared_shooting_gatepair_v1.py -- pair eval_gates check lines: arm vs reference,
floor (a) = SD of each line across the reference draws (ref + floor draws, same seed count).
    python scripts/grade_shared_shooting_gatepair_v1.py ARM.md REF.md DRAW1.md ... > out.md
"""
import re, sys
import numpy as np

def parse(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 5 and c[4] in ("PASS", "FAIL", "NEEDS-INSTRUMENTATION") and not re.match(r"^[-0-9.]+$", c[0]):
            m = re.match(r"^([-+]?[0-9.]+)", c[1])
            if c[0] not in out:
                out[c[0]] = (float(m.group(1)) if m else None, c[4], c[1], c[2])
    return out

arm, ref, *draws = sys.argv[1:]
A, R = parse(arm), parse(ref)
D = [R] + [parse(d) for d in draws]
print("| line | target | S0 (ref) | arm | delta | draw SD (n=%d) | delta / draw SD | status ref -> arm |" % len(D))
print("|---|---|---|---|---|---|---|---|")
for k, (v, st, raw, tgt) in A.items():
    if k not in R or v is None or R[k][0] is None:
        continue
    vals = [d[k][0] for d in D if k in d and d[k][0] is not None]
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    dl = v - R[k][0]
    z = dl / sd if sd > 0 else float("nan")
    print(f"| {k} | {tgt} | {R[k][0]:.4f} | {v:.4f} | {dl:+.4f} | {sd:.4f} | {z:+.1f} | {R[k][1]} -> {st} |")
