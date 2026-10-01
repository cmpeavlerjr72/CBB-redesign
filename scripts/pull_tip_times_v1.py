#!/usr/bin/env python
"""
pull_tip_times_v1.py -- tip-time refresh step of the daily ingestion (lane F2, 2026-09-30).

Takes real tip times from the sources already used in the repo: the hoopR season schedule (raw GitHub parquet, `time_valid`) and the
CBBD `/games` endpoint (`startTimeTbd`), merges them with `cbb_sim.live.tips.merge_tips`, and writes

    data/processed/ingest/tip_times_{season}.parquet        latest table, one row per CBBD game in the window
    data/processed/ingest/tip_times/{season}/<UTC stamp>.parquet   immutable snapshot of this run (so the tip at publish time is auditable)

columns: game_id, cbbd_game_id, tipoff_utc, tip_source, tip_time_is_placeholder, tip_disagree_min, tip_fetched_at.
A placeholder (midnight ET, TBD) is recorded and flagged, never promoted to a real time. ESPN's public scoreboard API is NOT used: hoopR and
CBBD are the sources; they simply fill in late (see docs/ops/day1_readiness_2027_2026-09-30.md).

    .venv/Scripts/python.exe scripts/pull_tip_times_v1.py --season 2027 --from 2026-11-02 --to 2026-11-08 [--dry-run] [--hoopr-local]

Cost: 1 CBBD /games call per run (window <= 14 days) plus one hoopR schedule download. `refresh_tip_times` is importable by the chain.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cbb_sim.live import tips as T  # noqa: E402

HOOPR_URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/schedules/parquet/mbb_schedule_{season}.parquet"
OUT_DIR = REPO / "data/processed/ingest"


def fetch_cbbd_games(session, tracker, season: int, lo: date, hi: date) -> pd.DataFrame:
    import chain_daily as CD
    start = CD.pull_cbbd.iso(datetime(lo.year, lo.month, lo.day, tzinfo=timezone.utc) - timedelta(hours=12))
    end = CD.pull_cbbd.iso(datetime(hi.year, hi.month, hi.day, tzinfo=timezone.utc) + timedelta(days=1, hours=12))
    rows = CD.pull_cbbd.http_get(session, tracker, "/games", {"season": season, "startDateRange": start, "endDateRange": end})
    return pd.DataFrame(rows)


def fetch_hoopr_schedule(season: int, local: Path | None = None) -> pd.DataFrame | None:
    if local is not None:
        return pd.read_parquet(local)
    import io
    import requests
    r = requests.get(HOOPR_URL.format(season=season), timeout=120, headers={"User-Agent": "cbb-clean-sheet-tip-refresh/1.0"})
    if r.status_code != 200:
        return None
    return pd.read_parquet(io.BytesIO(r.content))


def refresh_tip_times(season: int, lo: date, hi: date, cbbd_games: pd.DataFrame, hoopr_sched: pd.DataFrame | None, now,
                      out_dir: Path = OUT_DIR, write: bool = True) -> tuple[pd.DataFrame, dict]:
    tips = T.merge_tips(T.cbbd_tip_frame(cbbd_games) if len(cbbd_games) else pd.DataFrame(
        columns=["game_id", "cbbd_game_id", "cbbd_tip", "cbbd_tbd"]),
        T.hoopr_tip_frame(hoopr_sched) if hoopr_sched is not None else None, now)
    summ = {"season": season, "window": [str(lo), str(hi)], "n_games": int(len(tips)),
            "n_placeholder": int(tips["tip_time_is_placeholder"].sum()),
            "n_real": int((~tips["tip_time_is_placeholder"]).sum()),
            "by_source": tips["tip_source"].value_counts().to_dict(),
            "n_real_sources_disagree_gt5min": int((tips["tip_disagree_min"] > 5).sum()),
            "hoopr_available": hoopr_sched is not None, "fetched_at": str(T._utc(now))}
    if write and len(tips):
        out_dir.mkdir(parents=True, exist_ok=True)
        snap = out_dir / "tip_times" / str(season)
        snap.mkdir(parents=True, exist_ok=True)
        stamp = T._utc(now).strftime("%Y%m%dT%H%M%SZ")
        tips.to_parquet(snap / f"{stamp}.parquet", index=False)
        latest = out_dir / f"tip_times_{season}.parquet"
        if latest.exists():                       # upsert by game id; this run's rows win
            old = pd.read_parquet(latest)
            old = old[~old["game_id"].isin(tips["game_id"])]
            tips_all = pd.concat([old, tips], ignore_index=True)
        else:
            tips_all = tips
        tips_all.to_parquet(latest, index=False)
        summ["latest"] = str(latest)
    return tips, summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2027)
    ap.add_argument("--from", dest="lo", required=True)
    ap.add_argument("--to", dest="hi", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hoopr-local", default=None, help="use this local hoopR schedule parquet instead of downloading")
    a = ap.parse_args(argv)
    import chain_daily as CD
    ctx = CD.build_context(date.today(), a.dry_run, max_calls=5)
    lo, hi = date.fromisoformat(a.lo), date.fromisoformat(a.hi)
    cg = fetch_cbbd_games(ctx.session, ctx.tracker, a.season, lo, hi)
    hs = fetch_hoopr_schedule(a.season, Path(a.hoopr_local) if a.hoopr_local else None)
    now = pd.Timestamp.now("UTC")
    _, summ = refresh_tip_times(a.season, lo, hi, cg, hs, now, write=not a.dry_run)
    summ["cbbd_calls"] = ctx.tracker.count
    print(json.dumps(summ, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
