"""diag_chain_clock_vs_L2_v1.py -- does the retrain chain's clock stage equal the adopted clock L2 refit? (lane D, 2026-10-01)

The adopted clock `ENGINE_CLOCK=v5b_r6L2_glat_pmean` is the served v5b spec refitted on possessions v4 by lane B
(`exp_clk6_r6_arms_v1.py --arm L2`, artifacts `data/processed/models/clock/r6_L2/`). The chain's clock stage is
`train_clock_chain_v1.py --poss-version v4 --ratings-dir <dir>`. This script compares a chain clock root with r6_L2:
design table, the six S1 pickles (bytes), the S1 manifests, the censoring tables and the v5b latent sigma.

    .venv/Scripts/python.exe scripts/train_clock_chain_v1.py --root <ROOT> --poss-version v4 --ratings-dir data/processed/ratings
    .venv/Scripts/python.exe scripts/diag_chain_clock_vs_L2_v1.py --root <ROOT> [--out <json>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
L2 = ROOT / "data/processed/models/clock/r6_L2"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=L2)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = a.root if a.root.is_absolute() else ROOT / a.root
    ref = a.ref if a.ref.is_absolute() else ROOT / a.ref
    rep: dict = {"root": str(root), "ref": str(ref)}
    has_design = (root / "design_v2.parquet").exists()
    d1 = pd.read_parquet(root / "design_v2.parquet") if has_design else None
    d0 = pd.read_parquet(ref / "design_v2.parquet")
    same_cols = has_design and list(d1.columns) == list(d0.columns)
    rep["design_rows"] = [len(d1) if has_design else "not present", len(d0)]
    rep["design_equals"] = bool(same_cols and d1.equals(d0)) if has_design else "not present"
    if has_design and not rep["design_equals"] and same_cols and len(d1) == len(d0):
        diff = {c: int((~((d1[c] == d0[c]) | (d1[c].isna() & d0[c].isna()))).sum()) for c in d1.columns}
        rep["design_cols_differing"] = {c: n for c, n in diff.items() if n}
    pk = sorted(p.name for p in (ref / "v3c_s1").glob("*.pkl"))
    rep["s1_pickles"] = {n: (sha(root / "v3c_s1" / n) == sha(ref / "v3c_s1" / n)) if (root / "v3c_s1" / n).exists()
                         else "missing" for n in pk}
    m1 = json.loads((root / "v3c_s1/manifest_srfloor_P3.json").read_text())
    m0 = json.loads((ref / "v3c_s1/manifest_srfloor_P3.json").read_text())
    strip = lambda m: [{k: v for k, v in e.items() if k not in ("model_file", "path")} for e in m.get("months", [])]
    rep["manifest_months_equal_ignoring_paths"] = strip(m1) == strip(m0)
    cz = sorted(p.name for p in (ref / "clock_censoring").glob("*.parquet"))
    rep["censoring"] = {n: bool(pd.read_parquet(root / "clock_censoring" / n).equals(
        pd.read_parquet(ref / "clock_censoring" / n))) if (root / "clock_censoring" / n).exists() else "not present"
        for n in cz}
    s1 = json.loads((root / "v5b_bakeoff/v5b_bakeoff_report.json").read_text())["params"]["F2"]["B1_sigma"]
    s0 = json.loads((ref / "v5b_bakeoff/v5b_bakeoff_report.json").read_text())["params"]["F2"]["B1_sigma"]
    rep["B1_sigma"] = [s1, s0]
    rep["B1_sigma_equal"] = s1 == s0
    rep["IDENTICAL"] = bool(rep["design_equals"] is True and all(v is True for v in rep["s1_pickles"].values())
                            and all(v is True for v in rep["censoring"].values()) and rep["B1_sigma_equal"])
    #: what the engine SERVES from a clock root: the S1 pickles (via the manifest) and B1 sigma
    rep["SERVED_OBJECTS_IDENTICAL"] = bool(all(v is True for v in rep["s1_pickles"].values()) and rep["B1_sigma_equal"])
    txt = json.dumps(rep, indent=1, default=str)
    print(txt)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
