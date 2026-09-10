#!/usr/bin/env python
"""
build_kenpom_snapshots.py -- build the point-in-time KenPom snapshot table.

Loads every available KenPom snapshot (weekly 2022-2025, daily 2026, preseason
2027 if present -- see `src/cbb_sim/data/kenpom.py` for source paths and the
name-matching writeup), computes the CENTERED features CLAUDE.md's modeling
rules require ("every rating feature is expressed relative to its own
snapshot's league mean"), and writes the long-format table to
data/processed/kenpom_snapshots.parquet.

Prints, to confirm the postmortem's drift number (100 -> 109.3):
  - snapshot counts per season
  - the league-mean AdjO drift table (first vs. last snapshot per season)
  - the team-name match report (KenPom name -> hoopR team_location)

Run: .venv/Scripts/python.exe scripts/build_kenpom_snapshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.data import kenpom  # noqa: E402

OUT_PATH = ROOT / "data" / "processed" / "kenpom_snapshots.parquet"


def drift_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in df.groupby("season"):
        dates = sorted(g["snapshot_date"].unique())
        first, last = dates[0], dates[-1]
        gf = g[g["snapshot_date"] == first]
        gl = g[g["snapshot_date"] == last]
        rows.append(
            {
                "season": int(season),
                "n_snapshots": len(dates),
                "first_date": pd.Timestamp(first).date(),
                "last_date": pd.Timestamp(last).date(),
                "mean_adj_o_first": round(float(gf["adj_o"].mean()), 2),
                "mean_adj_o_last": round(float(gl["adj_o"].mean()), 2),
                "mean_adj_d_first": round(float(gf["adj_d"].mean()), 2),
                "mean_adj_d_last": round(float(gl["adj_d"].mean()), 2),
                "n_teams_first": int(gf["adj_o"].notna().sum()),
                "n_teams_last": int(gl["adj_o"].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("season")


def main() -> None:
    print("[build_kenpom_snapshots] loading sources ...")
    df, report = kenpom.build_snapshots()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[build_kenpom_snapshots] wrote {OUT_PATH} ({len(df):,} rows)")

    print("\n[snapshot counts per season]")
    counts = df.groupby("season")["snapshot_date"].nunique().rename("n_snapshot_dates")
    rows_per_season = df.groupby("season").size().rename("n_rows")
    print(pd.concat([counts, rows_per_season], axis=1).to_string())

    print("\n[league-mean drift, first vs. last snapshot per season]")
    dt = drift_table(df)
    print(dt.to_string(index=False))

    print("\n[name match report]")
    print(f"  unique raw KenPom names : {report['n_unique_raw_names']}")
    print(f"  matched to hoopR team   : {report['n_matched']}")
    print(f"  UNMATCHED               : {report['n_unmatched']}")
    if report["unmatched_names"]:
        print("  unmatched names:")
        for n in report["unmatched_names"]:
            print(f"    - {n}")

    n_null_team = int(df["team"].isna().sum())
    print(f"\n[rows with unmatched team] {n_null_team:,} / {len(df):,} ({n_null_team / len(df):.2%})")


if __name__ == "__main__":
    main()
