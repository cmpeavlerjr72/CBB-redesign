#!/usr/bin/env python
"""
pull_cbbd_lines_v1.py -- CBBD `/lines` pull for the free-lines-source task
(Task A, docs/tests/lines_cbbd_validation_2026-09-10.md).

Seasons: 2023, 2024, 2025 (real-book seasons, validated) and 2026 (pulled for
later use only -- 2026 is sealed, this script writes it to disk and reports a
row count but nothing about its content is read or summarised anywhere else).

Schema check performed before writing this script (one live `/lines` call,
season 2023, a 2-day window): every per-line record CBBD returns has exactly
these 7 fields, no timestamp of any kind beyond the game-level `startDate`:
    provider, spread, overUnder, homeMoneyline, awayMoneyline,
    spreadOpen, overUnderOpen
So "every field CBBD returns" == the flatten_lines() schema scripts/pull_cbbd.py
already used. There is no separate line-level timestamp anywhere in the
payload -- flagged in the validation doc.

Efficiency: scripts/pull_cbbd.py already pulled all four seasons in the
identical windowed fashion into data/raw/cbbd/lines_{season}.parquet on
2026-09-10. Per the "CBBD API use without waste" instruction, this script
REUSES that cached pull (0 new API calls) when the legacy file exists and its
schema matches, copying it into the new data/raw/cbbd/lines/ layout this task
asks for. It only calls the live API for a season whose legacy file is
missing or malformed.

Usage:
    .venv/Scripts/python.exe scripts/pull_cbbd_lines_v1.py
    .venv/Scripts/python.exe scripts/pull_cbbd_lines_v1.py --force-pull   # ignore cache

Writes:
    data/raw/cbbd/lines/lines_{season}.parquet   (2023, 2024, 2025, 2026)
    data/raw/cbbd/lines/manifest_lines_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import dotenv_values

BASE_URL = "https://api.collegebasketballdata.com"
ENV_PATH = Path(r"C:\Users\devuser\cfb-props-sim\.env")
ROOT = Path(__file__).resolve().parent.parent
LEGACY_DIR = ROOT / "data" / "raw" / "cbbd"
OUT_DIR = LEGACY_DIR / "lines"

SLEEP_SECS = 0.3
UNFILTERED_CAP = 3000
WINDOW_DAYS = 14
SAFETY_FLOOR = 500

SEASONS = [2023, 2024, 2025, 2026]

LINE_FIELDS = ["provider", "spread", "overUnder", "homeMoneyline",
               "awayMoneyline", "spreadOpen", "overUnderOpen"]
GAME_FIELDS = ["gameId", "season", "seasonType", "startDate",
               "homeTeamId", "homeTeam", "homeConference", "homeScore",
               "awayTeamId", "awayTeam", "awayConference", "awayScore"]
FULL_SCHEMA = GAME_FIELDS + LINE_FIELDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("pull_cbbd_lines_v1")


def load_api_key() -> str:
    vals = dotenv_values(str(ENV_PATH))
    key = vals.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError(f"CFBD_API_KEY not found in {ENV_PATH}")
    return key


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def season_windows(season: int, window_days: int = WINDOW_DAYS) -> list[tuple[str, str]]:
    start = datetime(season - 1, 11, 1, tzinfo=timezone.utc)
    end = datetime(season, 5, 1, tzinfo=timezone.utc)
    windows = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=window_days), end)
        windows.append((iso(cur), iso(nxt)))
        cur = nxt
    return windows


def flatten_lines(games: list[dict]) -> pd.DataFrame:
    rows = []
    for g in games:
        base = {c: g.get(c) for c in GAME_FIELDS}
        lines = g.get("lines") or []
        if not lines:
            row = dict(base)
            for f in LINE_FIELDS:
                row[f] = None
            rows.append(row)
        else:
            for l in lines:
                row = dict(base)
                for f in LINE_FIELDS:
                    row[f] = l.get(f)
                rows.append(row)
    return pd.DataFrame(rows, columns=FULL_SCHEMA)


def http_get(session, state, endpoint, params) -> Any:
    if state["remaining"] is not None and state["remaining"] < SAFETY_FLOOR:
        raise RuntimeError(f"X-CallLimit-Remaining below safety floor ({state['remaining']})")
    for attempt in range(4):
        try:
            resp = session.get(f"{BASE_URL}{endpoint}", params=params, timeout=60)
        except requests.RequestException as exc:
            log.warning("request error (attempt %d): %s", attempt, exc)
            time.sleep(1.0 + attempt)
            continue
        state["count"] += 1
        rem = resp.headers.get("X-CallLimit-Remaining")
        if rem is not None:
            state["remaining"] = int(rem)
        if resp.status_code == 429:
            time.sleep(2.0 + attempt * 2)
            continue
        resp.raise_for_status()
        time.sleep(SLEEP_SECS)
        return resp.json()
    raise RuntimeError(f"failed after retries: {endpoint} {params}")


def pull_season_live(session, state, season: int) -> pd.DataFrame:
    windows = season_windows(season)
    all_games, seen = [], set()
    for start, end in windows:
        data = http_get(session, state, "/lines",
                         {"season": season, "startDateRange": start, "endDateRange": end})
        if len(data) >= UNFILTERED_CAP:
            log.warning("season %d window %s..%s hit cap (%d)", season, start, end, len(data))
        for g in data:
            if g["gameId"] not in seen:
                seen.add(g["gameId"])
                all_games.append(g)
    return flatten_lines(all_games)


def try_reuse_legacy(season: int) -> Optional[pd.DataFrame]:
    legacy = LEGACY_DIR / f"lines_{season}.parquet"
    if not legacy.exists():
        return None
    df = pd.read_parquet(legacy)
    if list(df.columns) != FULL_SCHEMA:
        log.warning("legacy %s has schema %s, expected %s -- will re-pull",
                    legacy.name, list(df.columns), FULL_SCHEMA)
        return None
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    ap.add_argument("--force-pull", action="store_true",
                     help="ignore the legacy data/raw/cbbd/lines_{season}.parquet cache")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = {"count": 0, "remaining": None}
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "seasons": {}}
    key = None
    session = None

    for season in args.seasons:
        reused = None if args.force_pull else try_reuse_legacy(season)
        if reused is not None:
            df = reused
            source = "reused_legacy_pull_2026-09-10"
            log.info("season %d: reused legacy pull (%d rows, 0 API calls)", season, len(df))
        else:
            if session is None:
                key = load_api_key()
                session = requests.Session()
                session.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
            df = pull_season_live(session, state, season)
            source = "live_pull"
            log.info("season %d: live-pulled (%d rows, running call count %d)",
                     season, len(df), state["count"])
        out_path = OUT_DIR / f"lines_{season}.parquet"
        df.to_parquet(out_path, index=False)
        manifest["seasons"][str(season)] = {
            "rows": int(len(df)),
            "unique_games": int(df["gameId"].nunique()) if len(df) else 0,
            "source": source,
            "output": str(out_path.relative_to(ROOT)),
        }

    manifest["api_calls_used"] = state["count"]
    manifest["call_limit_remaining_final"] = state["remaining"]
    manifest["schema"] = FULL_SCHEMA
    (OUT_DIR / "manifest_lines_v1.json").write_text(json.dumps(manifest, indent=2))

    print("=" * 70)
    print("pull_cbbd_lines_v1 summary")
    print("=" * 70)
    for season, info in manifest["seasons"].items():
        print(f"  season {season}: {info['rows']:>6} rows, {info['unique_games']:>5} games, "
              f"source={info['source']}")
    print(f"  API calls used this run: {state['count']}")
    print(f"  X-CallLimit-Remaining: {state['remaining']}")


if __name__ == "__main__":
    main()
