#!/usr/bin/env python
"""build_po_design_v4_v1.py -- possession_outcome round-2 design on a chosen possessions version and ratings dir.

Versioned sibling of the design step of `scripts/train_possession_outcome_v2.py` (NOT edited), whose
`POSSESSIONS_VERSION = "v2"` constant and default ratings dir are the only reason it cannot build the
full-retrain design. This calls the SAME function with the SAME arguments
(`PO.build_design(TRAIN_SEASONS, version=..., style_source="first_chance", require_pbp_complete=True)`)
and only exposes `--poss-version` and `--ratings-dir`. No feature, arm or filter changes.

Identity (must reproduce the served cache `round2/design.parquet` column for column):
    .venv/Scripts/python.exe scripts/build_po_design_v4_v1.py --poss-version v2 \
        --ratings-dir data/processed/ratings --out <scratch>/design.parquet --identity
Full retrain:
    .venv/Scripts/python.exe scripts/build_po_design_v4_v1.py --poss-version v4 \
        --ratings-dir data/processed/ratings_C_v1 --out <root>/po_design/design.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    import os
    os.environ.setdefault(_k, "1")

SERVED = ROOT / "data/processed/models/possession_outcome/round2/design.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poss-version", default="v4", choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--ratings-dir", default="data/processed/ratings")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--identity", action="store_true", help="compare to the served round2 design.parquet")
    a = ap.parse_args()
    if a.out.resolve() == SERVED.resolve():
        raise SystemExit("refusing to overwrite the served round-2 design")
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; never overwrite (delete your own scratch first)")
    import pandas as pd
    import train_possession_outcome_v2 as T2
    from cbb_sim.data.seal import assert_not_sealed
    from cbb_sim.models import possession_outcome as PO

    t0 = time.time()
    assert_not_sealed(T2.TRAIN_SEASONS, context="po design v4 sibling")
    d = PO.build_design(T2.TRAIN_SEASONS, version=a.poss_version, ratings_dir=a.ratings_dir,
                        style_source="first_chance", require_pbp_complete=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(a.out, index=False)
    rep = {"poss_version": a.poss_version, "ratings_dir": str(a.ratings_dir), "rows": int(len(d)),
           "n_first": int((d["population"] == "first").sum()), "n_cont": int((d["population"] == "cont").sum()),
           "seasons": T2.TRAIN_SEASONS, "seconds": round(time.time() - t0, 1),
           "wrapped": "PO.build_design as called by train_possession_outcome_v2.build_or_load_design"}
    if a.identity:
        new = pd.read_parquet(a.out)
        old = pd.read_parquet(SERVED)
        rep["identity_equal"] = bool(new.equals(old))
        rep["identity_rows"] = [int(len(new)), int(len(old))]
        if not rep["identity_equal"]:
            rep["cols_only_new"] = sorted(set(new.columns) - set(old.columns))
            rep["cols_only_old"] = sorted(set(old.columns) - set(new.columns))
            if len(new) == len(old):
                rep["cols_differing"] = [c for c in old.columns if c in new.columns and not new[c].equals(old[c])]
    (a.out.parent / (a.out.stem + "_report.json")).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0 if rep.get("identity_equal", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
