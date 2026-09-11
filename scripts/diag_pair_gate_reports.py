"""diag_pair_gate_reports.py -- put two or three `eval_gates.py` reports side by
side, with a seed-offset noise band.

A paired engine read is only readable when the SAME grader's own lines are
lined up; re-deriving a number by hand is how two "improved" variants got
believed last year. So this parses the reports themselves -- the headline
`| quantity | value | target | tolerance | status |` table under each `## G<n>`
heading -- and never recomputes anything.

Column meaning, when three reports are given:

    A   the reference arm
    B   the arm under test, SAME seed values as A
    N   a spec-identical second draw of A under DIFFERENT seeds

`|B - A|` is the movement being judged and `|N - A|` is the seed-offset noise
floor it must beat. A line whose movement does not exceed its own floor is
reported as INSIDE FLOOR, which is a non-finding and is labelled one.

    .venv/Scripts/python.exe scripts/diag_pair_gate_reports.py \
        --a docs/tests/gates_A.md --b docs/tests/gates_B.md --noise docs/tests/gates_N.md \
        --label-a "clock v3c, tree @4503c52" --label-b "clock v3c, tree @905b902" \
        --out docs/tests/pair.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse(path: Path) -> dict:
    gate, rows = None, {}
    overall = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+(G\d)\s", line)
        if m:
            gate = m.group(1)
            continue
        m = re.match(r"^\*\*(G\d) overall:\s*([A-Z-]+)\*\*", line)
        if m:
            overall[m.group(1)] = m.group(2)
            continue
        if gate and line.startswith("|") and line.count("|") >= 6:
            c = [x.strip() for x in line.strip("|").split("|")]
            if len(c) < 5 or c[0] in ("quantity", "---") or set(c[0]) <= {"-"}:
                continue
            if c[4] not in ("PASS", "FAIL", "NEEDS-INSTRUMENTATION", "UNDERPOWERED"):
                continue
            rows[(gate, c[0])] = {"value": c[1], "target": c[2],
                                  "tol": c[3], "status": c[4]}
    return {"rows": rows, "overall": overall}


def first_num(s: str) -> float | None:
    m = NUM.search(s.replace(",", ""))
    return float(m.group(0)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--label-n", default="noise (seeds 20-39)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = parse(Path(args.a)), parse(Path(args.b))
    N = parse(Path(args.noise)) if args.noise else None

    out: list[str] = []
    out.append(f"| gate | line | {args.label_a} | {args.label_b} | B-A | "
               f"{args.label_n} | N-A (floor) | verdict |")
    out.append("|---|---|---|---|---:|---|---:|---|")
    keys = list(A["rows"].keys())
    for k in B["rows"]:
        if k not in keys:
            keys.append(k)
    for k in keys:
        ra = A["rows"].get(k)
        rb = B["rows"].get(k)
        rn = N["rows"].get(k) if N else None
        va = ra["value"] if ra else "--"
        vb = rb["value"] if rb else "--"
        vn = rn["value"] if rn else "--"
        na, nb, nn = first_num(va), first_num(vb), first_num(vn)
        d = f"{nb - na:+.4f}" if (na is not None and nb is not None) else "--"
        dn = f"{nn - na:+.4f}" if (na is not None and nn is not None) else "--"
        verdict = "--"
        if na is not None and nb is not None and nn is not None:
            mv, fl = abs(nb - na), abs(nn - na)
            verdict = "**MOVED**" if mv > fl else "inside floor"
            if mv == 0.0:
                verdict = "identical"
        sa = ra["status"] if ra else "--"
        sb = rb["status"] if rb else "--"
        flip = "" if sa == sb else f"  ({sa} -> {sb})"
        out.append(f"| {k[0]} | {k[1]} | {va} [{sa}] | {vb} [{sb}] | {d} | "
                   f"{vn} | {dn} | {verdict}{flip} |")
    out.append("")
    out.append("Gate-level verdicts: " + ", ".join(
        f"{g} {A['overall'].get(g,'?')}/{B['overall'].get(g,'?')}"
        + (f"/{N['overall'].get(g,'?')}" if N else "")
        for g in sorted(set(A["overall"]) | set(B["overall"]))))
    txt = "\n".join(out)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
