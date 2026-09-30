#!/usr/bin/env python
"""
build_possessions_v4.py -- EVENT LAYER v4 as a VERSIONED SIBLING.

    .venv/Scripts/python.exe scripts/build_possessions_v4.py                 # 2022-2026
    .venv/Scripts/python.exe scripts/build_possessions_v4.py --seasons 2025

v4 = the served v2 event layer + the 09-18 technical-FT lookahead (v3) + the
three phantom-possession switches (`cbb_sim.pbp.possessions.EVENT_FIX_SWITCHES`;
design and evidence: docs/tests/event_layer_v4_2026-09-30.md).

Per season this script
  1. rebuilds the DEFAULT machine (every switch off) and asserts it reproduces
     `data/processed/possessions_v2/{possessions,chances}_{season}.parquet`
     BIT-IDENTICALLY (DataFrame.equals, same columns, same dtypes);
  2. builds v4 and writes ONLY
       data/processed/possessions_v4/possessions_{season}.parquet
       data/processed/possessions_v4/chances_{season}.parquet
       data/processed/possessions_v4/build_report.json
Nothing existing is overwritten; no consumer is switched.  2026 (the sealed
season) may be BUILT -- it is data preparation -- and only its row counts are
reported.

HF sync: `data/processed/possessions_v4/` is gitignored bulk; its sync key is
recorded in the v4 doc (section 2). This script does not sync.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.pbp import possessions as PZ  # noqa: E402

UNIVERSE = ROOT / "data/processed/games_universe.parquet"
PBP_DIR = ROOT / "data/raw/cbbd/pbp"
V2 = ROOT / "data/processed/possessions_v2"


def load_universe() -> pd.DataFrame:
    u = pd.read_parquet(UNIVERSE)
    u = u[u["is_d1_game"] & ~u["pbp_truncated"] & u["cbbd_game_id"].notna()].copy()
    u["cbbd_game_id"] = u["cbbd_game_id"].astype("int64")
    return u


def same(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = {"rows_a": int(len(a)), "rows_b": int(len(b)),
           "columns_equal": list(a.columns) == list(b.columns),
           "dtypes_equal": bool((a.dtypes.astype(str).to_numpy() == b.dtypes.astype(str).to_numpy()).all())
           if list(a.columns) == list(b.columns) else False}
    out["bit_identical"] = bool(out["columns_equal"] and a.reset_index(drop=True).equals(b.reset_index(drop=True)))
    if not out["bit_identical"] and out["columns_equal"] and len(a) == len(b):
        diff = [c for c in a.columns if not a[c].reset_index(drop=True).equals(b[c].reset_index(drop=True))]
        out["differing_columns"] = diff
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=[2025, 2024, 2023, 2022, 2026])
    args = ap.parse_args()
    t0 = time.time()
    u = load_universe()
    out_dir = PZ.possessions_dir("v4")
    out_dir.mkdir(parents=True, exist_ok=True)
    rp = out_dir / "build_report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {"seasons": {}}
    report["switches"] = PZ.VERSION_EVENT_FIXES["v4"]
    report["tech_lookahead"] = PZ.VERSION_TECH_LOOKAHEAD["v4"]
    report["stray_reb_max_s"] = PZ.STRAY_REB_MAX_S
    for s in args.seasons:
        ts = time.time()
        rec: dict = {}
        # 1. default path parity against the served v2 table
        p_d, c_d, _ = PZ.segment_season(s, u, pbp_dir=PBP_DIR)
        rec["default_parity_possessions"] = same(p_d, pd.read_parquet(V2 / f"possessions_{s}.parquet"))
        rec["default_parity_chances"] = same(c_d, pd.read_parquet(V2 / f"chances_{s}.parquet"))
        del p_d, c_d
        # 2. v4
        p4, c4, diag = PZ.segment_season(s, u, pbp_dir=PBP_DIR, tech_lookahead=PZ.VERSION_TECH_LOOKAHEAD["v4"],
                                         **PZ.VERSION_EVENT_FIXES["v4"])
        p4.to_parquet(out_dir / f"possessions_{s}.parquet", index=False)
        c4.to_parquet(out_dir / f"chances_{s}.parquet", index=False)
        diag.update({"n_possessions": int(len(p4)), "n_chances": int(len(c4)),
                     "wall_clock_s": round(time.time() - ts, 1)})
        rec["v4_diag"] = diag
        report["seasons"][str(s)] = rec
        report["built_at"] = time.strftime("%Y-%m-%d %H:%M")
        rp.write_text(json.dumps(report, indent=1, default=str))
        print(f"[{time.time() - t0:7.1f}s] {s}: parity poss={rec['default_parity_possessions']['bit_identical']} "
              f"chances={rec['default_parity_chances']['bit_identical']}; v4 {len(p4):,} possessions "
              f"{len(c4):,} chances", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
