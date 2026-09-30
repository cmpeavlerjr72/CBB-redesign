#!/usr/bin/env python
"""
diag_truth_flow_scan_v1.py -- where do unplayed / disputed game_ids appear? (Lane H, 2026-09-30)

Reads the flagged set from data/processed/truth/truth_audit_unplayed_v1.parquet
(diag_truth_finals_audit_v1.py), then scans the `game_id` column of every parquet
under data/processed, data/reference and results/ (column-only reads; sealed season 2026
ids are NOT in the flagged set used here, so nothing sealed is touched) and counts how many
flagged ids each table holds.  Writes data/processed/truth/truth_flow_scan_v1.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
aud = pd.read_parquet(ROOT / "data/processed/truth/truth_audit_unplayed_v1.parquet")
aud = aud[~aud["sealed_season"]]
nonfinal = aud[aud["cls"].str.contains("nonfinal|0_0|missing")]
U = set(nonfinal["game_id"].dropna().astype("int64"))
D = set(aud.loc[aud["cls"].str.contains("score_disagree"), "game_id"].astype("int64"))
uni = pd.read_parquet(ROOT / "data/processed/games_universe.parquet")
g = uni[uni["is_d1_game"] & ~uni["pbp_truncated"]]
UG = U & set(g["game_id"])  # in the graded universe rule
print(f"unplayed/nonfinal ids (2022-2025): {len(U)}; of which in graded universe rule: {len(UG)}; disputed-score ids: {len(D)}")

rows = []
for base in ("data/processed", "data/reference", "results"):
    for p in sorted((ROOT / base).rglob("*.parquet")):
        try:
            sch = pq.read_schema(p)
            if "game_id" not in sch.names:
                continue
            ids = pq.read_table(p, columns=["game_id"]).column(0).to_pandas()
            ids = pd.to_numeric(ids, errors="coerce").dropna().astype("int64")
        except Exception as e:  # noqa: BLE001
            rows.append({"path": str(p.relative_to(ROOT)), "err": str(e)[:80]})
            continue
        uq = set(ids.unique())
        rows.append({"path": str(p.relative_to(ROOT)).replace("\\", "/"), "rows": len(ids), "n_games": len(uq),
                     "n_unplayed_ids": len(uq & U), "unplayed_rows": int(ids.isin(U).sum()),
                     "n_in_graded_unplayed": len(uq & UG), "n_disputed_ids": len(uq & D),
                     "n_forfeit_or_zero_scored": None})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "data/processed/truth/truth_flow_scan_v1.csv", index=False)
hit = out[(out["n_unplayed_ids"].fillna(0) > 0) | (out["n_disputed_ids"].fillna(0) > 0)]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 90)
print(f"scanned {len(out)} parquet files with game_id; {len(hit)} hold flagged ids")
print(hit.drop(columns=["n_forfeit_or_zero_scored"]).to_string(index=False))
