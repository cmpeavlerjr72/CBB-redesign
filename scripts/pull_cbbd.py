#!/usr/bin/env python
"""
pull_cbbd.py -- download CollegeBasketballData (CBBD) API data into data/raw/cbbd.

API: https://api.collegebasketballdata.com  (Authorization: Bearer <CFBD_API_KEY>)
Key is read from C:\\Users\\devuser\\cfb-props-sim\\.env (CFBD_API_KEY=...) via
python-dotenv. The key is never printed or logged.

Unfiltered list endpoints (/lines, /games) cap out at 3000 rows, so both are
paged by 14-day windows spanning Nov 1 (season-1) through May 1 (season) --
calibrated against live counts (peak ~650 games / 14-day window, well under
the 3000 cap even during conference-tournament season).

Season convention: CBBD "season" is the *ending* year (2024-25 season -> 2025),
same convention as hoopR.

Rate limiting: sleeps SLEEP_SECS (0.3s) between calls and tracks the
X-CallLimit-Remaining response header. Stops early (raises) if the total call
count would exceed --max-calls (default 1500) or remaining quota drops below
a safety floor.

Usage:
    python scripts/pull_cbbd.py
    python scripts/pull_cbbd.py --max-calls 1500

Writes:
    data/raw/cbbd/lines_providers.json
    data/raw/cbbd/lines_{season}.parquet
    data/raw/cbbd/games_{season}.parquet
    data/raw/cbbd/ratings_adjusted_{season}.parquet
    data/raw/cbbd/ratings_srs_{season}.parquet
    data/raw/cbbd/ratings_elo_{season}.parquet
    data/raw/cbbd/samples/*.json
    data/raw/cbbd/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import dotenv_values

BASE_URL = "https://api.collegebasketballdata.com"
ENV_PATH = Path(r"C:\Users\devuser\cfb-props-sim\.env")
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "cbbd"
SAMPLES_DIR = OUT_DIR / "samples"

SLEEP_SECS = 0.3
UNFILTERED_CAP = 3000
WINDOW_DAYS = 14
SAFETY_FLOOR = 500  # stop pulling if X-CallLimit-Remaining drops below this

LINES_SEASONS = list(range(2013, 2027))  # 2013..2026 inclusive
GAMES_SEASONS = list(range(2013, 2027))  # extended back since budget allows
RATINGS_SEASONS = list(range(2022, 2027))  # 2022..2026 inclusive
SAMPLE_SEASON = 2025
SAMPLE_TEAM = "Duke"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_cbbd")


@dataclass
class CallTracker:
    max_calls: int
    count: int = 0
    remaining: Optional[int] = None
    manifest: list[dict[str, Any]] = field(default_factory=list)

    def note(self, endpoint: str, params: dict, rows: int, extra: Optional[dict] = None):
        entry = {
            "endpoint": endpoint,
            "params": params,
            "rows": rows,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "call_limit_remaining": self.remaining,
        }
        if extra:
            entry.update(extra)
        self.manifest.append(entry)


def load_api_key() -> str:
    vals = dotenv_values(str(ENV_PATH))
    key = vals.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError(f"CFBD_API_KEY not found in {ENV_PATH}")
    return key


def http_get(session: requests.Session, tracker: CallTracker, endpoint: str, params: dict) -> Any:
    if tracker.count >= tracker.max_calls:
        raise RuntimeError(f"Call budget exhausted ({tracker.count}/{tracker.max_calls})")
    if tracker.remaining is not None and tracker.remaining < SAFETY_FLOOR:
        raise RuntimeError(f"X-CallLimit-Remaining below safety floor ({tracker.remaining})")

    url = f"{BASE_URL}{endpoint}"
    for attempt in range(4):
        try:
            resp = session.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            log.warning("request error on %s %s (attempt %d): %s", endpoint, params, attempt, exc)
            time.sleep(1.0 + attempt)
            continue
        tracker.count += 1
        remaining_hdr = resp.headers.get("X-CallLimit-Remaining")
        if remaining_hdr is not None:
            tracker.remaining = int(remaining_hdr)
        if resp.status_code == 429:
            log.warning("429 rate limited on %s %s, backing off", endpoint, params)
            time.sleep(2.0 + attempt * 2)
            continue
        if resp.status_code != 200:
            log.error("HTTP %d on %s %s: %s", resp.status_code, endpoint, params, resp.text[:300])
            resp.raise_for_status()
        time.sleep(SLEEP_SECS)
        return resp.json()
    raise RuntimeError(f"Failed after retries: {endpoint} {params}")


def season_windows(season: int, window_days: int = WINDOW_DAYS) -> list[tuple[str, str]]:
    """14-day UTC windows from Nov 1 (season-1) to May 1 (season)."""
    start = datetime(season - 1, 11, 1, tzinfo=timezone.utc)
    end = datetime(season, 5, 1, tzinfo=timezone.utc)
    windows = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=window_days), end)
        windows.append((iso(cur), iso(nxt)))
        cur = nxt
    return windows


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# --------------------------------------------------------------------------
# Pull functions
# --------------------------------------------------------------------------

def pull_lines_providers(session: requests.Session, tracker: CallTracker):
    log.info("Pulling /lines/providers")
    data = http_get(session, tracker, "/lines/providers", {})
    (OUT_DIR / "lines_providers.json").write_text(json.dumps(data, indent=2))
    tracker.note("/lines/providers", {}, len(data))
    log.info("  -> %d providers", len(data))


def flatten_lines(games: list[dict]) -> pd.DataFrame:
    rows = []
    keep_cols = [
        "gameId", "season", "seasonType", "startDate",
        "homeTeamId", "homeTeam", "homeConference", "homeScore",
        "awayTeamId", "awayTeam", "awayConference", "awayScore",
    ]
    for g in games:
        base = {c: g.get(c) for c in keep_cols}
        lines = g.get("lines") or []
        if not lines:
            row = dict(base)
            for f in ["provider", "spread", "overUnder", "homeMoneyline",
                      "awayMoneyline", "spreadOpen", "overUnderOpen"]:
                row[f] = None
            rows.append(row)
        else:
            for l in lines:
                row = dict(base)
                row["provider"] = l.get("provider")
                row["spread"] = l.get("spread")
                row["overUnder"] = l.get("overUnder")
                row["homeMoneyline"] = l.get("homeMoneyline")
                row["awayMoneyline"] = l.get("awayMoneyline")
                row["spreadOpen"] = l.get("spreadOpen")
                row["overUnderOpen"] = l.get("overUnderOpen")
                rows.append(row)
    return pd.DataFrame(rows)


def pull_lines(session: requests.Session, tracker: CallTracker, seasons: list[int]):
    for season in seasons:
        windows = season_windows(season)
        all_games: list[dict] = []
        seen_ids: set = set()
        for start, end in windows:
            params = {"season": season, "startDateRange": start, "endDateRange": end}
            try:
                data = http_get(session, tracker, "/lines", params)
            except RuntimeError as exc:
                log.error("Stopping lines pull for season %d: %s", season, exc)
                raise
            if len(data) >= UNFILTERED_CAP:
                log.warning("season %d window %s..%s hit cap (%d rows) -- consider narrower window",
                            season, start, end, len(data))
            for g in data:
                if g["gameId"] not in seen_ids:
                    seen_ids.add(g["gameId"])
                    all_games.append(g)
        df = flatten_lines(all_games)
        out_path = OUT_DIR / f"lines_{season}.parquet"
        df.to_parquet(out_path, index=False)
        tracker.note("/lines", {"season": season, "windows": len(windows)}, len(df),
                     {"output": str(out_path.name), "unique_games": len(seen_ids)})
        log.info("  season %d: %d games, %d line-rows -> %s", season, len(seen_ids), len(df), out_path.name)


def pull_games(session: requests.Session, tracker: CallTracker, seasons: list[int]):
    for season in seasons:
        windows = season_windows(season)
        all_games: dict[int, dict] = {}
        for start, end in windows:
            params = {"season": season, "startDateRange": start, "endDateRange": end}
            try:
                data = http_get(session, tracker, "/games", params)
            except RuntimeError as exc:
                log.error("Stopping games pull for season %d: %s", season, exc)
                raise
            if len(data) >= UNFILTERED_CAP:
                log.warning("season %d window %s..%s hit cap (%d rows) -- consider narrower window",
                            season, start, end, len(data))
            for g in data:
                all_games[g["id"]] = g
        df = pd.DataFrame(list(all_games.values()))
        out_path = OUT_DIR / f"games_{season}.parquet"
        df.to_parquet(out_path, index=False)
        tracker.note("/games", {"season": season, "windows": len(windows)}, len(df),
                     {"output": str(out_path.name)})
        log.info("  season %d: %d games -> %s", season, len(df), out_path.name)


def pull_ratings(session: requests.Session, tracker: CallTracker, seasons: list[int]):
    for endpoint, stem in [("/ratings/adjusted", "ratings_adjusted"),
                            ("/ratings/srs", "ratings_srs"),
                            ("/ratings/elo", "ratings_elo")]:
        for season in seasons:
            data = http_get(session, tracker, endpoint, {"season": season})
            df = pd.DataFrame(data)
            out_path = OUT_DIR / f"{stem}_{season}.parquet"
            df.to_parquet(out_path, index=False)
            tracker.note(endpoint, {"season": season}, len(df), {"output": str(out_path.name)})
            log.info("  %s season %d: %d rows -> %s", endpoint, season, len(df), out_path.name)


def pull_samples(session: requests.Session, tracker: CallTracker):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Sample game ids: 3 completed Duke games in season 2025
    duke_games = http_get(session, tracker, "/games", {"season": SAMPLE_SEASON, "team": SAMPLE_TEAM})
    tracker.note("/games", {"season": SAMPLE_SEASON, "team": SAMPLE_TEAM}, len(duke_games),
                 {"purpose": "sample_game_id_source"})
    final_ids = [g["id"] for g in duke_games if g.get("status") == "final"][:3]
    if len(final_ids) < 3:
        final_ids = [g["id"] for g in duke_games][:3]
    log.info("Sample game ids (Duke, %d): %s", SAMPLE_SEASON, final_ids)
    (SAMPLES_DIR / "sample_game_ids.json").write_text(json.dumps(final_ids, indent=2))

    for gid in final_ids:
        for endpoint, stem in [(f"/plays/game/{gid}", "plays_game"),
                                (f"/substitutions/game/{gid}", "substitutions_game"),
                                (f"/lineups/game/{gid}", "lineups_game")]:
            data = http_get(session, tracker, endpoint, {})
            out_path = SAMPLES_DIR / f"{stem}_{gid}.json"
            out_path.write_text(json.dumps(data, indent=2))
            tracker.note(endpoint, {"gameId": gid}, len(data) if isinstance(data, list) else 1,
                         {"output": str(out_path.name)})
            log.info("  %s -> %d records", endpoint, len(data) if isinstance(data, list) else 1)

    # games/players
    data = http_get(session, tracker, "/games/players", {"season": SAMPLE_SEASON, "team": SAMPLE_TEAM})
    out_path = SAMPLES_DIR / "games_players_duke_2025.json"
    out_path.write_text(json.dumps(data, indent=2))
    tracker.note("/games/players", {"season": SAMPLE_SEASON, "team": SAMPLE_TEAM}, len(data),
                 {"output": str(out_path.name)})
    log.info("  /games/players -> %d game-rows", len(data))

    # teams/roster
    data = http_get(session, tracker, "/teams/roster", {"season": SAMPLE_SEASON, "team": SAMPLE_TEAM})
    out_path = SAMPLES_DIR / "teams_roster_duke_2025.json"
    out_path.write_text(json.dumps(data, indent=2))
    tracker.note("/teams/roster", {"season": SAMPLE_SEASON, "team": SAMPLE_TEAM}, len(data),
                 {"output": str(out_path.name)})
    log.info("  /teams/roster -> %d team-rows", len(data))

    # recruiting/portal -- NB: this endpoint filters on `year`, NOT `season`
    # (verified live: season=2025 param is ignored and returns years
    # 2021-2026 unfiltered [6679 rows]; year=2025 correctly filters to 1611
    # rows). We pull with the correct `year` param and note the discrepancy.
    data = http_get(session, tracker, "/recruiting/portal", {"year": SAMPLE_SEASON})
    out_path = SAMPLES_DIR / "recruiting_portal_2025.json"
    out_path.write_text(json.dumps(data, indent=2))
    tracker.note("/recruiting/portal", {"year": SAMPLE_SEASON}, len(data),
                 {"output": str(out_path.name),
                  "note": "endpoint param is `year` not `season`; season= is silently ignored"})
    log.info("  /recruiting/portal -> %d rows", len(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=1500)
    ap.add_argument("--skip-lines", action="store_true")
    ap.add_argument("--skip-games", action="store_true")
    ap.add_argument("--skip-ratings", action="store_true")
    ap.add_argument("--skip-samples", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    key = load_api_key()
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "accept": "application/json"})

    tracker = CallTracker(max_calls=args.max_calls)

    try:
        pull_lines_providers(session, tracker)
        if not args.skip_lines:
            pull_lines(session, tracker, LINES_SEASONS)
        if not args.skip_games:
            pull_games(session, tracker, GAMES_SEASONS)
        if not args.skip_ratings:
            pull_ratings(session, tracker, RATINGS_SEASONS)
        if not args.skip_samples:
            pull_samples(session, tracker)
    finally:
        manifest_path = OUT_DIR / "manifest.json"
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_url": BASE_URL,
            "total_calls": tracker.count,
            "call_limit_remaining_final": tracker.remaining,
            "entries": tracker.manifest,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log.info("Wrote manifest: %s", manifest_path)
        log.info("Total API calls made: %d", tracker.count)
        log.info("X-CallLimit-Remaining (final): %s", tracker.remaining)


if __name__ == "__main__":
    sys.exit(main())
