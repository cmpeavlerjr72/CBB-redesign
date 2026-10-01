#!/usr/bin/env python
"""build_design_overlay_v1.py -- write a NEW design file with feature-table columns overlaid (lane J's recipe).

`train_par_common_v1.overlay_columns` (imported, not edited) applied for each `--feature-table PATH:KEYS[:COLS]` in
order; interaction columns `x_<a>__<b>` are recomputed when an input is overlaid, unmatched rows are refused (strict).
Used by the full-retrain chain where a trainer takes `--design` but not `--feature-table` (the season-anchor
possession_outcome trainer), so that every PO arm sees the same corrected in_bonus state.

    .venv/Scripts/python.exe scripts/build_design_overlay_v1.py --design <root>/po_design/design.parquet \
        --feature-table <root>/foul_state/in_bonus_overlay.parquet:game_id,poss_index,chance_number:in_bonus \
        --out <root>/po_design/design_inbonus.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def parse_spec(spec: str):
    parts = spec.split(":")
    if len(parts) >= 2 and len(parts[0]) == 1 and parts[1].startswith(("\\", "/")):
        parts = [parts[0] + ":" + parts[1], *parts[2:]]
    cols = parts[2].split(",") if len(parts) > 2 and parts[2] else None
    return Path(parts[0]), parts[1].split(","), cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--feature-table", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() or a.out.resolve() == a.design.resolve():
        raise SystemExit(f"{a.out} exists or is the input")
    import pandas as pd
    import train_par_common_v1 as C
    d = pd.read_parquet(a.design)
    reps = []
    for spec in a.feature_table:
        p, keys, cols = parse_spec(spec)
        d, r = C.overlay_columns(d, p, keys=keys, cols=cols)
        reps.append(r)
    tmp = a.out.with_suffix(".parquet.tmp")
    d.to_parquet(tmp, index=False)
    tmp.replace(a.out)
    rep = {"design": str(a.design), "out": str(a.out), "overlays": reps}
    (a.out.parent / (a.out.stem + "_overlay_report.json")).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
