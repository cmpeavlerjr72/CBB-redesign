#!/usr/bin/env python
"""
chain_ingest_daily_v1.py -- the post-game ingestion step for the daily chain (Lane K, 2026-09-30).

chain_daily.py is not edited today; this file exposes a step with the same shape as its steps so the PM can add
`("ingest", step_ingest)` to `STEPS` (after the schedule step, before any ratings / live-input step) in one line:

    import chain_ingest_daily_v1 as CI ; STEPS.insert(1, CI.STEP)

It ingests YESTERDAY's games (US/Eastern date = ctx.yesterday) and retries every pending game still inside the retry
window (pull_daily_ingest_v1.py). Idempotent, exception-isolated by the chain's run_step. Standalone:

    .venv/Scripts/python.exe scripts/chain_ingest_daily_v1.py --date 2026-11-03     # ingests 2026-11-02 + retries pending
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pull_daily_ingest_v1 as PI  # noqa: E402


def step_ingest(ctx) -> dict:
    """ctx: chain_daily.ChainContext (uses .yesterday, .dry_run, .season). Returns a small status dict."""
    argv = ["--dates", str(ctx.yesterday), "--season", str(ctx.season)]
    if ctx.dry_run:
        argv.append("--dry-run")
    rc = PI.main(argv)
    return {"rc": rc, "date": str(ctx.yesterday), "season": ctx.season}


STEP = ("ingest", step_ingest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date.today()), help="run date (YYYY-MM-DD); ingests the day before")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d = date.fromisoformat(a.date)
    argv = ["--dates", str(d - timedelta(days=1)), "--season", str(PI.season_of(d - timedelta(days=1)))]
    if a.dry_run:
        argv.append("--dry-run")
    return PI.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
