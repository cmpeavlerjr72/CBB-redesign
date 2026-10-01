"""Compact gate table: headline quantity rows of ARM grade md vs REF grade md (operator 2026-10-01). usage: gatetab.py ARM.md REF.md"""
import re, sys
def parse(p):
    rows = {}; gate = None; inq = False
    for ln in open(p, encoding="utf-8"):
        m = re.match(r"## (G\d+\w*) --", ln)
        if m: gate = m.group(1); inq = False; continue
        if ln.startswith("| quantity |"): inq = True; continue
        if inq and ln.startswith("|---"): continue
        if inq and ln.startswith("|"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if len(c) >= 5: rows[(gate, c[0])] = (c[1], c[-1])
        elif inq: inq = False
    return rows
a, r = parse(sys.argv[1]), parse(sys.argv[2])
print("| gate | quantity | arm (value, status) | reference (value, status) |\n|---|---|---|---|")
for k in a:
    print(f"| {k[0]} | {k[1]} | {a[k][0]} {a[k][1]} | {r.get(k, ('-','-'))[0]} {r.get(k, ('-','-'))[1]} |")
