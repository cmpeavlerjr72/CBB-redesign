#!/usr/bin/env python
"""
build_possessions_v4otc_v1.py -- event layer v4 + OVERTIME TEAM-FOUL CARRY, as a VERSIONED SIBLING (lane F, 2026-10-01).

    .venv/Scripts/python.exe scripts/build_possessions_v4otc_v1.py --seasons 2022 2023 2024 2025 [--foul-state]

Writes ONLY data/processed/possessions_v4otc/{possessions,chances}_{season}.parquet (+ build_report.json) and, with --foul-state,
data/processed/models/possession_outcome/round6_v4otc/foul_accrual_poss.parquet (lane D's `build_foul_state_v4_v1.build_state`, imported
unedited, machine label `v4otc`). Overwrites nothing; no trainer default or served artifact changes. 2026 (sealed) is refused.

Proof written to the build report, per season: regulation rows (period <= 2) of both tables are `DataFrame.equals` to
data/processed/possessions_v4 (bit-identical, same columns and dtypes); the OT rows that differ are counted per column.
Machine change (`cbb_sim.pbp.possessions`, switch `ot_foul_carry`, default off): team fouls are NOT reset at the 2->3 boundary or any
later one (NCAA men's: reset at the end of the first half only). Terminal classes of OT free-throw trips can change because
`classify_ft_trip` reads the prior foul count; that is the intended correction and is counted as a column diff.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from cbb_sim.pbp import possessions as PZ  # noqa: E402
import build_possessions_v4 as B4  # noqa: E402

V4 = ROOT / "data/processed/possessions_v4"


def diff_report(a: pd.DataFrame, b: pd.DataFrame, key: list[str]) -> dict:
    """a = v4 (reference), b = v4otc. Regulation identity and OT per-column diffs (row-aligned; asserts the same row order and keys)."""
    out = {"rows_v4": int(len(a)), "rows_v4otc": int(len(b)), "columns_equal": list(a.columns) == list(b.columns)}
    out["dtypes_equal"] = bool((a.dtypes.astype(str).to_numpy() == b.dtypes.astype(str).to_numpy()).all()) if out["columns_equal"] else False
    out["same_row_count"] = len(a) == len(b)
    if not (out["columns_equal"] and out["same_row_count"]):
        return out
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    out["keys_equal"] = bool(a[key].equals(b[key]))
    reg = (a["period"] <= 2).to_numpy()
    out["regulation_rows"] = int(reg.sum())
    out["regulation_bit_identical"] = bool(a[reg].reset_index(drop=True).equals(b[reg].reset_index(drop=True)))
    ot = ~reg
    out["ot_rows"] = int(ot.sum())
    d = {}
    anyd = np.zeros(len(a), dtype=bool)
    for c in a.columns:
        x, y = a.loc[ot, c], b.loc[ot, c]
        ne = ~((x == y) | (x.isna() & y.isna())).to_numpy()
        if ne.any():
            d[c] = int(ne.sum()); anyd[np.flatnonzero(ot)[ne]] = True
    out["ot_rows_differing_by_column"] = d
    out["ot_rows_any_difference"] = int(anyd.sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--foul-state", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if 2026 in a.seasons:
        raise SystemExit("2026 is sealed; this sibling is built for 2022-2025 only")
    t0 = time.time()
    u = B4.load_universe()
    out_dir = PZ.possessions_dir("v4otc")
    out_dir.mkdir(parents=True, exist_ok=True)
    rp = out_dir / "build_report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {"seasons": {}}
    report["switches"] = PZ.VERSION_EVENT_FIXES["v4otc"]
    for s in a.seasons:
        ts = time.time()
        p, c, diag = PZ.segment_season(s, u, pbp_dir=B4.PBP_DIR, tech_lookahead=PZ.VERSION_TECH_LOOKAHEAD["v4otc"],
                                       andone_made_next="made_FT", **{k: v for k, v in PZ.VERSION_EVENT_FIXES["v4otc"].items() if k != "ot_foul_carry"},
                                       ot_foul_carry=True)
        p.to_parquet(out_dir / f"possessions_{s}.parquet", index=False)
        c.to_parquet(out_dir / f"chances_{s}.parquet", index=False)
        rec = {"possessions": diff_report(pd.read_parquet(V4 / f"possessions_{s}.parquet"), p, ["game_id", "period", "poss_index"]),
               "chances": diff_report(pd.read_parquet(V4 / f"chances_{s}.parquet"), c, ["game_id", "period", "poss_index", "chance_number"]),
               "wall_clock_s": round(time.time() - ts, 1)}
        report["seasons"][str(s)] = rec
        rp.write_text(json.dumps(report, indent=1, default=str))
        print(f"[{time.time()-t0:6.1f}s] {s}: poss reg identical={rec['possessions'].get('regulation_bit_identical')} "
              f"OT rows {rec['possessions'].get('ot_rows')} differing {rec['possessions'].get('ot_rows_any_difference')}; chances reg identical="
              f"{rec['chances'].get('regulation_bit_identical')} OT rows {rec['chances'].get('ot_rows')} differing {rec['chances'].get('ot_rows_any_difference')}", flush=True)
    if a.foul_state:
        import build_foul_state_v4_v1 as FS
        fo = ROOT / "data/processed/models/possession_outcome/round6_v4otc"
        FS.build_state("v4otc", fo, a.seasons, a.workers)
        print("foul state ->", fo, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
