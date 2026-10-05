#!/usr/bin/env python
"""build_lines_open_close_v1.py -- per-game open / close from our own line snapshots (lines ops, 2026-10-05).

Reads data/raw/lines_snapshots/lines_snapshots_*.parquet and tip_times_2027. Honesty rule: only captures with captured_at STRICTLY
before tipoff are used (captured_at >= tipoff excluded). Per (game_id, provider, source): open = first qualifying capture,
close = last qualifying capture; lead times in minutes before tip. A single capture gives open == close
(n_captures = 1; not a real movement). Output data/processed/lines/lines_open_close_v1.parquet.
  .venv/Scripts/python.exe scripts/build_lines_open_close_v1.py [--tips PATH]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "raw" / "lines_snapshots"
TIPS = ROOT / "data" / "processed" / "ingest" / "tip_times_2027.parquet"
OUT = ROOT / "data" / "processed" / "lines" / "lines_open_close_v1.parquet"
VALS = ["spread_home", "total", "home_ml", "away_ml"]
COLS = (["game_id", "provider", "source", "n_captures", "tipoff_utc", "tip_time_is_placeholder", "open_captured_at",
         "open_lead_min", "close_captured_at", "close_lead_min"] + [f"{p}_{v}" for p in ("open", "close") for v in VALS])


def build(snaps: pd.DataFrame, tips: pd.DataFrame) -> pd.DataFrame:
    t = tips[["game_id", "tipoff_utc", "tip_time_is_placeholder"]].copy()
    t["tipoff_utc"] = pd.to_datetime(t["tipoff_utc"], utc=True)
    s = snaps.dropna(subset=["game_id"]).merge(t, on="game_id", how="inner")
    s["captured_at"] = pd.to_datetime(s["captured_at"], utc=True)
    s = s[s["captured_at"] < s["tipoff_utc"]]
    if s.empty:
        return pd.DataFrame(columns=COLS)
    s = s.sort_values("captured_at", kind="stable")
    k = ["game_id", "provider", "source"]
    g = s.groupby(k, sort=False)
    first, last = g.head(1).set_index(k), g.tail(1).set_index(k)
    out = pd.DataFrame(index=first.index)
    out["n_captures"] = g.size()
    out["tipoff_utc"] = first["tipoff_utc"]
    out["tip_time_is_placeholder"] = first["tip_time_is_placeholder"]
    for nm, df in (("open", first), ("close", last)):
        df = df.reindex(out.index)
        out[f"{nm}_captured_at"] = df["captured_at"]
        out[f"{nm}_lead_min"] = (df["tipoff_utc"] - df["captured_at"]).dt.total_seconds() / 60
        for v in VALS:
            out[f"{nm}_{v}"] = df[v]
    return out.reset_index()[COLS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tips", default=str(TIPS))
    ap.add_argument("--snap-dir", default=str(SNAP))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    files = sorted(Path(a.snap_dir).glob("lines_snapshots_*.parquet"))
    if not files or not Path(a.tips).exists():
        print("no snapshots or no tip table; nothing built")
        return 0
    res = build(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), pd.read_parquet(a.tips))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out, index=False)
    print(f"wrote {a.out} rows={len(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
