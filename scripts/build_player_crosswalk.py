#!/usr/bin/env python
"""
build_player_crosswalk.py -- CBBD <-> ESPN player id crosswalk.

Two stages:

1. `--pull` (default on when the roster dumps are missing): one CBBD
   `/teams/roster?season=S` call per season. The endpoint returns **every**
   team's roster for the season when `team` is omitted (1,519 team-rows for
   2025), so the entire crosswalk costs 3 API calls -- far inside the 300-call
   budget. The API key is read from the same `.env` `scripts/pull_cbbd.py`
   uses and is never printed or logged. Written to
   `data/raw/cbbd/rosters/roster_{season}.parquet`.

2. Build + report: `cbb_sim.data.player_ids.build_crosswalk` ->
   `data/processed/player_crosswalk.parquet` and
   `data/processed/player_crosswalk_report.json`.

Usage:
    .venv/Scripts/python.exe scripts/build_player_crosswalk.py
    .venv/Scripts/python.exe scripts/build_player_crosswalk.py --seasons 2024 2025 2026 --pull
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.data.player_ids import (  # noqa: E402
    DEFAULT_OUT,
    build_crosswalk,
)

BASE_URL = "https://api.collegebasketballdata.com"
ENV_PATH = Path(r"C:\Users\devuser\cfb-props-sim\.env")
ROSTER_DIR = ROOT / "data" / "raw" / "cbbd" / "rosters"
REPORT_PATH = ROOT / "data" / "processed" / "player_crosswalk_report.json"
MAX_CALLS = 300
SLEEP_SECS = 0.3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_player_crosswalk")


def load_api_key() -> str:
    vals = dotenv_values(str(ENV_PATH))
    key = vals.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError(f"CFBD_API_KEY not found in {ENV_PATH}")
    return key


def pull_rosters(seasons: list[int]) -> int:
    """One /teams/roster call per season (whole-season, all teams). Returns calls used."""
    ROSTER_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {load_api_key()}",
                            "Accept": "application/json"})
    calls = 0
    for season in seasons:
        if calls >= MAX_CALLS:
            raise RuntimeError(f"call budget exhausted ({calls}/{MAX_CALLS})")
        resp = session.get(f"{BASE_URL}/teams/roster", params={"season": season}, timeout=180)
        calls += 1
        remaining = resp.headers.get("X-CallLimit-Remaining")
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for team in data:
            for p in team.get("players") or []:
                rows.append({
                    "season": int(season),
                    "cbbd_team_id": team.get("teamId"),
                    "cbbd_team": team.get("team"),
                    "team_source_id": team.get("teamSourceId"),
                    "conference": team.get("conference"),
                    "cbbd_player_id": p.get("id"),
                    "source_id": p.get("sourceId"),
                    "name": p.get("name"),
                    "first_name": p.get("firstName"),
                    "last_name": p.get("lastName"),
                    "jersey": p.get("jersey"),
                    "position": p.get("position"),
                    "height": p.get("height"),
                    "weight": p.get("weight"),
                    "start_season": p.get("startSeason"),
                    "end_season": p.get("endSeason"),
                })
        df = pd.DataFrame(rows)
        df = df[df["cbbd_player_id"].notna()]
        df["cbbd_player_id"] = df["cbbd_player_id"].astype("int64")
        df["cbbd_team_id"] = df["cbbd_team_id"].astype("int64")
        out = ROSTER_DIR / f"roster_{season}.parquet"
        df.to_parquet(out, index=False)
        log.info("season %d: %d teams, %d players -> %s (call-limit remaining %s)",
                 season, len(data), len(df), out.name, remaining)
        time.sleep(SLEEP_SECS)
    return calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--pull", action="store_true", help="force a re-pull of the roster dumps")
    ap.add_argument("--out", type=Path, default=ROOT / DEFAULT_OUT)
    args = ap.parse_args()

    seasons = sorted(args.seasons)
    missing = [s for s in seasons if not (ROSTER_DIR / f"roster_{s}.parquet").exists()]
    if args.pull or missing:
        used = pull_rosters(seasons if args.pull else missing)
        log.info("CBBD API calls used: %d (budget %d)", used, MAX_CALLS)
    else:
        log.info("roster dumps present for %s -- no API calls", seasons)

    cw, report = build_crosswalk(
        seasons,
        roster_dir=ROSTER_DIR,
        player_box_dir=ROOT / "data" / "raw" / "hoopr" / "player_box",
        poss_dir=ROOT / "data" / "processed" / "possessions",
        team_crosswalk_path=ROOT / "data" / "reference" / "team_crosswalk.parquet",
        pbp_dir=ROOT / "data" / "raw" / "cbbd" / "pbp",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cw.to_parquet(args.out, index=False)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    log.info("crosswalk -> %s (%d rows)", args.out, len(cw))
    print()
    print("=" * 78)
    print("CBBD <-> ESPN PLAYER CROSSWALK -- MATCH RATE")
    print("=" * 78)
    hdr = f"{'scope':>10} {'rows':>8} {'row match':>10} {'on-floor players':>18} {'poss-weighted':>15}"
    print(hdr)
    for scope in ["overall"] + [str(s) for s in seasons]:
        b = report[scope]
        print(f"{scope:>10} {b['rows']:>8} {b['row_match_rate']*100:>9.2f}% "
              f"{b['on_floor_matched']}/{b['on_floor_players']} "
              f"({b['on_floor_player_match_rate']*100:.2f}%) "
              f"{b['possession_weighted_match_rate']*100:>13.4f}%")
    print()
    print("by match method (overall):", report["overall"]["by_method"])
    if report["unmatched_on_floor_top20"]:
        print()
        print("largest unmatched on-floor ids:")
        for r in report["unmatched_on_floor_top20"][:10]:
            print(f"  season {r['season']} cbbd_id {r['cbbd_player_id']:>7} "
                  f"{str(r['name'])[:28]:<28} {r['on_floor_poss']} poss")


if __name__ == "__main__":
    main()
