#!/usr/bin/env python
"""
pull_cbbd_pbp.py -- bulk-pull play-by-play (PBP) data from CollegeBasketballData
(CBBD) via /plays/date, one calendar date at a time, for seasons 2022..2026.

API: https://api.collegebasketballdata.com  (Authorization: Bearer <CFBD_API_KEY>)
Key is read from C:\\Users\\devuser\\cfb-props-sim\\.env (CFBD_API_KEY=...) via
python-dotenv, reusing pull_cbbd.load_api_key(). The key is never printed or
logged. BASE_URL is also reused from pull_cbbd.

------------------------------------------------------------------------------
DATE SEMANTIC OF /plays/date -- VERIFICATION FINDINGS (documented per spec)
------------------------------------------------------------------------------
Before bulk-pulling, we verified what calendar date /plays/date?date=YYYY-MM-DD
actually buckets a game's plays under, using two known games from season 2025
whose games_2025.parquet `startDate` (always UTC, ISO8601 with "Z") sits right
at a date boundary:

  * Game id 8: Gonzaga (home) vs Baylor, tipped ~8:30pm Pacific on 2024-11-04.
    startDate = "2024-11-05T04:30:00.000Z" (a precise, real broadcast time).
    -> ALL 337 of its plays came back under /plays/date?date=2024-11-05, and
       NONE under date=2024-11-04, even though 8:30pm Pacific on Nov 4 is
       still "Nov 4" in every US time zone (ET, CT, MT, PT all agree).
       i.e. this game's bucket == the raw UTC calendar date of `startDate`,
       verbatim, with NO local-US-timezone shift applied.

  * Game id 98: Louisville (home) vs Morehead State.
    startDate = "2024-11-05T00:00:00.000Z" -- exactly midnight UTC. Games with
    an exact 00:00:00.000Z time are extremely common (1374/6299 = 22% of
    season-2025 games) and are almost certainly a synthesized placeholder for
    "exact tip time unknown" (CBBD's `startTimeTbd` flag is unfortunately
    always False in this dataset, so it can't be used to confirm this
    directly) -- 00:00:00Z is exactly what you get by taking a default 7:00pm
    US/Eastern tip and converting to UTC during EST (UTC-5).
    -> ALL of this game's plays (confirmed via its real wallclock timestamps,
       19:47-21:08 UTC on 2024-11-04) came back under
       /plays/date?date=2024-11-04, i.e. ONE DAY EARLIER than the raw UTC
       date of its own `startDate` field.

  We also confirmed (game id 8) that /plays/date assigns a game's plays to
  exactly ONE date bucket as a whole -- even though 89 of its 337 plays have
  a `wallclock` timestamp that is itself UTC-dated 2024-11-04 (pre-midnight)
  and 248 are UTC-dated 2024-11-05, /plays/date?date=2024-11-04 returned ZERO
  of this game's plays and /plays/date?date=2024-11-05 returned all 337. So
  the bucketing key is a single per-game date, not a per-play wallclock date.

  CONCLUSION: the two verification games disagree under every single-formula
  hypothesis we tried (raw UTC date of startDate; startDate shifted to
  US/Eastern local date; startDate shifted to US/Pacific local date; a fixed
  -1-day shift). Real, precisely-timed games (game 8) keep the raw UTC date;
  placeholder-time games with startDate exactly at 00:00:00.000Z (game 98)
  bucket one day earlier. This is consistent with CBBD storing an internal,
  authoritative "game date" that predates the "startTimeTbd -> synthesize a
  7pm-ET startDate" step for placeholder games, and just uses the raw
  `startDate` UTC date for everything else -- but this cannot be verified
  from the public API surface, and we did not find a rule confirmed safe for
  100% of games.

  DECISION: rather than compute one inferred local date per game (risking
  silently dropped games whenever the inference is wrong), for each season we
  query EVERY calendar date in a padded continuous range:
      [min(UTC date of startDate) - 3 days, max(UTC date of startDate) + 3 days]
  This guarantees both the "same as raw UTC date" and "one day earlier"
  buckets (the only two behaviors observed) are queried regardless of which
  applies to any given game, at the cost of a handful of extra always-empty
  calendar-day calls at the very edges of the season. Completeness is then
  verified empirically post-hoc: docs/tests/data_audit_cbbd_pbp_*.md reports
  games_{season}.parquet coverage (missing games + example gameIds) per
  season, which would surface any residual gap in this strategy.

------------------------------------------------------------------------------
Payload size verification
------------------------------------------------------------------------------
/plays/date?date=2024-11-05 (a 174-scheduled-game UTC-date, 115 of which had
plays data) returned 39,202 play rows / ~53MB JSON in one response -- i.e.
the endpoint is NOT silently truncating large days (no 3000-row cap like the
unfiltered /games or /lines endpoints in pull_cbbd.py). A separate check
during initial dev against a ~43-game day matched the expected ~21,066
rows / ~28.6MB order of magnitude. No pagination is implemented by this
script because none was observed to be needed.

------------------------------------------------------------------------------
Flattening
------------------------------------------------------------------------------
Each play record's scalar fields are kept as-is (gameId, gameSourceId,
gameStartDate, season, seasonType, gameType, tournament, id, sourceId,
playType, isHomeTeam, teamId, team, conference, teamSeed, opponentId,
opponent, opponentConference, opponentSeed, homeScore, awayScore,
homeWinProbability, period, clock, secondsRemaining, scoringPlay,
shootingPlay, scoreValue, wallclock, playText), plus:

  * participants (list of {id,name}, length observed 0/1/2): kept whole as a
    JSON string in `participants_json`, plus `participant_1_id` /
    `participant_2_id` scalar columns for the first two entries (order as
    returned by the API).
  * shotInfo (single observed key-set: shooter, made, range, assisted,
    assistedBy, location{x,y}; null for non-shooting plays): flattened to
    shot_shooter_id, shot_shooter_name, shot_made, shot_range, shot_assisted,
    shot_assisted_by_id, shot_assisted_by_name, shot_location_x,
    shot_location_y.
  * onFloor (list of {id,name,team}, length observed mostly 10 but also 0/8/
    9/11 -- a real data-quality wrinkle, see the audit's onFloor-completeness
    metric): kept whole as a JSON string in `on_floor_json`, plus
    home_on_1..home_on_5 / away_on_1..away_on_5 id columns. Home/away side
    for each onFloor entry is assigned by comparing its `team` name against
    the home/away team names from games_{season}.parquet joined on gameId
    (NOT the play's own team/opponent/isHomeTeam fields, which are null for
    game-administrative plays like timeouts and period/game-end markers that
    can still carry a populated onFloor list). Within a side, entries are
    sorted by player id ascending for determinism before filling slots 1..5;
    if a side has other than exactly 5 entries the extra slots are left null
    (fewer than 5) or the lowest-id 5 are kept (more than 5).

------------------------------------------------------------------------------
Explicitly NOT pulled: /substitutions, /lineups (derivable from onFloor deltas).
------------------------------------------------------------------------------

Rate limiting: sleeps SLEEP_SECS (0.5s) between calls, retries with
exponential backoff on 429 and 5xx responses, and stops the whole run
cleanly (writes manifest + partial audit, exits 0) the moment
X-CallLimit-Remaining drops below CALL_LIMIT_FLOOR (20,000).

Resumable: a per-date file is skipped (no API call) if
data/raw/cbbd/pbp/{season}/plays_{date}.parquet already exists.

Usage:
    python scripts/pull_cbbd_pbp.py
    python scripts/pull_cbbd_pbp.py --seasons 2025 2026
    python scripts/pull_cbbd_pbp.py --audit-only   # regenerate the audit md
                                                    # from files already on disk

Writes:
    data/raw/cbbd/pbp/{season}/plays_{date}.parquet   (per-date, resumable)
    data/raw/cbbd/pbp/plays_{season}.parquet          (per-season concat, zstd)
    data/raw/cbbd/manifest.json                       (appended, not overwritten)
    docs/tests/data_audit_cbbd_pbp_2026-09-10.md
"""

from __future__ import annotations

import argparse
import hashlib
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_cbbd import BASE_URL, load_api_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "cbbd"
PBP_DIR = OUT_DIR / "pbp"
MANIFEST_PATH = OUT_DIR / "manifest.json"
AUDIT_PATH = ROOT / "docs" / "tests" / "data_audit_cbbd_pbp_2026-09-10.md"

SEASONS = list(range(2022, 2027))  # 2022..2026 inclusive
SLEEP_SECS = 0.5
CALL_LIMIT_FLOOR = 20_000
DATE_PAD_DAYS = 3  # padding on each side of the season's raw-UTC-date span
MAX_RETRIES = 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_cbbd_pbp")


class QuotaExhausted(Exception):
    pass


@dataclass
class CallTracker:
    count: int = 0
    remaining: Optional[int] = None
    manifest: list[dict[str, Any]] = field(default_factory=list)

    def note(self, entry: dict[str, Any]) -> None:
        self.manifest.append(entry)


def http_get(session: requests.Session, tracker: CallTracker, endpoint: str, params: dict) -> Any:
    """GET with retry/backoff on 429 + 5xx, tracking X-CallLimit-Remaining.

    Raises QuotaExhausted if remaining quota is already below CALL_LIMIT_FLOOR
    before making the call (checked pre-flight so we never fire a call we
    can't afford to report cleanly).
    """
    if tracker.remaining is not None and tracker.remaining < CALL_LIMIT_FLOOR:
        raise QuotaExhausted(f"X-CallLimit-Remaining={tracker.remaining} < floor {CALL_LIMIT_FLOOR}")

    url = f"{BASE_URL}{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=120)
        except requests.RequestException as exc:
            log.warning("request error on %s %s (attempt %d): %s", endpoint, params, attempt, exc)
            time.sleep(2.0 + attempt * 2)
            continue
        tracker.count += 1
        remaining_hdr = resp.headers.get("X-CallLimit-Remaining")
        if remaining_hdr is not None:
            tracker.remaining = int(remaining_hdr)
        if resp.status_code == 429:
            log.warning("429 rate limited on %s %s, backing off (attempt %d)", endpoint, params, attempt)
            time.sleep(3.0 + attempt * 3)
            continue
        if 500 <= resp.status_code < 600:
            log.warning("HTTP %d on %s %s, backing off (attempt %d)", resp.status_code, endpoint, params, attempt)
            time.sleep(3.0 + attempt * 3)
            continue
        if resp.status_code != 200:
            log.error("HTTP %d on %s %s: %s", resp.status_code, endpoint, params, resp.text[:300])
            resp.raise_for_status()
        time.sleep(SLEEP_SECS)
        return resp.json()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {endpoint} {params}")


# --------------------------------------------------------------------------
# Date range
# --------------------------------------------------------------------------

def season_date_range(season: int) -> list[str]:
    """Every calendar date to query for a season, padded DATE_PAD_DAYS on
    each side of the season's raw-UTC-date span (see module docstring for
    why padding, rather than per-game local-date inference, is used)."""
    games_path = OUT_DIR / f"games_{season}.parquet"
    df = pd.read_parquet(games_path)
    utc_dates = pd.to_datetime(df["startDate"].str.slice(0, 10))
    start = utc_dates.min() - timedelta(days=DATE_PAD_DAYS)
    end = utc_dates.max() + timedelta(days=DATE_PAD_DAYS)
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def load_home_away_map(season: int) -> dict[int, tuple[str, str]]:
    games_path = OUT_DIR / f"games_{season}.parquet"
    df = pd.read_parquet(games_path)
    return {
        int(row.id): (row.homeTeam, row.awayTeam)
        for row in df.itertuples(index=False)
    }


# --------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------

SCALAR_FIELDS = [
    "gameId", "gameSourceId", "gameStartDate", "season", "seasonType",
    "gameType", "tournament", "id", "sourceId", "playType", "isHomeTeam",
    "teamId", "team", "conference", "teamSeed", "opponentId", "opponent",
    "opponentConference", "opponentSeed", "homeScore", "awayScore",
    "homeWinProbability", "period", "clock", "secondsRemaining",
    "scoringPlay", "shootingPlay", "scoreValue", "wallclock", "playText",
]

SHOT_INFO_FIELDS = ["shooter", "made", "range", "assisted", "assistedBy", "location"]


def flatten_play(play: dict, home_away: dict[int, tuple[str, str]], query_date: str) -> dict:
    row: dict[str, Any] = {f: play.get(f) for f in SCALAR_FIELDS}
    row["query_date"] = query_date

    participants = play.get("participants") or []
    row["participants_json"] = json.dumps(participants)
    row["participant_1_id"] = participants[0].get("id") if len(participants) > 0 else None
    row["participant_2_id"] = participants[1].get("id") if len(participants) > 1 else None

    shot_info = play.get("shotInfo") or {}
    shooter = shot_info.get("shooter") or {}
    assisted_by = shot_info.get("assistedBy") or {}
    location = shot_info.get("location") or {}
    row["shot_shooter_id"] = shooter.get("id")
    row["shot_shooter_name"] = shooter.get("name")
    row["shot_made"] = shot_info.get("made")
    row["shot_range"] = shot_info.get("range")
    row["shot_assisted"] = shot_info.get("assisted")
    row["shot_assisted_by_id"] = assisted_by.get("id")
    row["shot_assisted_by_name"] = assisted_by.get("name")
    row["shot_location_x"] = location.get("x")
    row["shot_location_y"] = location.get("y")

    on_floor = play.get("onFloor") or []
    row["on_floor_json"] = json.dumps(on_floor)
    home_team, away_team = home_away.get(play.get("gameId"), (None, None))
    home_ids = sorted(p["id"] for p in on_floor if p.get("team") == home_team)
    away_ids = sorted(p["id"] for p in on_floor if p.get("team") == away_team)
    for i in range(5):
        row[f"home_on_{i + 1}"] = home_ids[i] if i < len(home_ids) else None
        row[f"away_on_{i + 1}"] = away_ids[i] if i < len(away_ids) else None

    return row


def flatten_plays(plays: list[dict], home_away: dict[int, tuple[str, str]], query_date: str) -> pd.DataFrame:
    rows = [flatten_play(p, home_away, query_date) for p in plays]
    if not rows:
        # Preserve schema for empty days so downstream concat/read still works.
        cols = SCALAR_FIELDS + [
            "query_date", "participants_json", "participant_1_id", "participant_2_id",
            "shot_shooter_id", "shot_shooter_name", "shot_made", "shot_range",
            "shot_assisted", "shot_assisted_by_id", "shot_assisted_by_name",
            "shot_location_x", "shot_location_y", "on_floor_json",
        ] + [f"home_on_{i + 1}" for i in range(5)] + [f"away_on_{i + 1}" for i in range(5)]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"generated_at": None, "base_url": BASE_URL, "total_calls": 0,
            "call_limit_remaining_final": None, "entries": []}


def save_manifest(manifest: dict[str, Any]) -> None:
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_entry_for_file(path: Path, endpoint: str, params: dict, rows: int) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "params": params,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "output": str(path.relative_to(OUT_DIR)).replace("\\", "/"),
    }


# --------------------------------------------------------------------------
# Per-season pull
# --------------------------------------------------------------------------

def process_season(session: requests.Session, tracker: CallTracker, manifest: dict[str, Any], season: int) -> None:
    season_dir = PBP_DIR / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    home_away = load_home_away_map(season)
    dates = season_date_range(season)
    log.info("Season %d: querying %d calendar dates (%s .. %s)", season, len(dates), dates[0], dates[-1])

    total_rows = 0
    new_calls = 0
    for date_str in dates:
        out_path = season_dir / f"plays_{date_str}.parquet"
        if out_path.exists():
            continue
        try:
            data = http_get(session, tracker, "/plays/date", {"date": date_str})
        except QuotaExhausted as exc:
            log.warning("Season %d: stopping early at date %s -- %s", season, date_str, exc)
            manifest["entries"].extend(tracker.manifest)
            tracker.manifest.clear()
            manifest["total_calls"] = manifest.get("total_calls", 0) + tracker.count
            manifest["call_limit_remaining_final"] = tracker.remaining
            save_manifest(manifest)
            raise
        new_calls += 1
        df = flatten_plays(data, home_away, date_str)
        df.to_parquet(out_path, index=False)
        entry = manifest_entry_for_file(out_path, "/plays/date", {"date": date_str}, len(df))
        entry["call_limit_remaining"] = tracker.remaining
        manifest["entries"].append(entry)
        total_rows += len(df)
        if new_calls % 20 == 0:
            log.info("  season %d: %d/%d dates done (%d new calls, remaining=%s)",
                      season, dates.index(date_str) + 1, len(dates), new_calls, tracker.remaining)
            # Persist manifest periodically so a crash doesn't lose provenance.
            manifest["total_calls"] = manifest.get("total_calls", 0) + tracker.count
            tracker.count = 0
            manifest["call_limit_remaining_final"] = tracker.remaining
            save_manifest(manifest)

    manifest["total_calls"] = manifest.get("total_calls", 0) + tracker.count
    tracker.count = 0
    manifest["call_limit_remaining_final"] = tracker.remaining
    save_manifest(manifest)

    # Concatenate the season's per-date files into one zstd parquet.
    per_date_files = sorted(season_dir.glob("plays_*.parquet"))
    frames = [pd.read_parquet(p) for p in per_date_files]
    season_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    season_out = PBP_DIR / f"plays_{season}.parquet"
    season_df.to_parquet(season_out, index=False, compression="zstd")
    entry = manifest_entry_for_file(
        season_out, "/plays/date", {"season": season, "dates": len(per_date_files)}, len(season_df)
    )
    manifest["entries"].append(entry)
    save_manifest(manifest)
    log.info("Season %d done: %d dates on disk, %d new calls this run, %d total rows -> %s",
              season, len(per_date_files), new_calls, len(season_df), season_out.name)


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def audit_season(season: int) -> dict[str, Any]:
    games_path = OUT_DIR / f"games_{season}.parquet"
    games = pd.read_parquet(games_path)
    final_games = games[games["status"] == "final"]

    season_pbp_path = PBP_DIR / f"plays_{season}.parquet"
    if not season_pbp_path.exists():
        return {"season": season, "error": f"missing {season_pbp_path.name}"}
    pbp = pd.read_parquet(season_pbp_path)

    covered_ids = set(pbp["gameId"].dropna().unique().tolist())
    final_ids = set(final_games["id"].tolist())
    missing_ids = sorted(final_ids - covered_ids)

    n_rows = len(pbp)
    home_cols = [f"home_on_{i+1}" for i in range(5)]
    away_cols = [f"away_on_{i+1}" for i in range(5)]
    home_complete = pbp[home_cols].notna().all(axis=1) if n_rows else pd.Series([], dtype=bool)
    away_complete = pbp[away_cols].notna().all(axis=1) if n_rows else pd.Series([], dtype=bool)
    both_complete = (home_complete & away_complete)
    on_floor_completeness = float(both_complete.mean()) if n_rows else None

    play_type_counts = pbp["playType"].value_counts(dropna=False).to_dict() if n_rows else {}
    shooting_share = float(pbp["shootingPlay"].mean()) if n_rows else None

    shot_fields = ["shot_shooter_id", "shot_shooter_name", "shot_made", "shot_range",
                   "shot_assisted", "shot_assisted_by_id", "shot_assisted_by_name",
                   "shot_location_x", "shot_location_y"]
    shot_coverage = {f: float(pbp[f].notna().mean()) for f in shot_fields} if n_rows else {}

    null_clock = int(pbp["clock"].isna().sum()) if n_rows else 0

    return {
        "season": season,
        "n_final_games": len(final_ids),
        "n_covered_games": len(final_ids & covered_ids),
        "n_missing_games": len(missing_ids),
        "example_missing_ids": missing_ids[:10],
        "n_rows": n_rows,
        "on_floor_completeness": on_floor_completeness,
        "play_type_counts": play_type_counts,
        "shooting_share": shooting_share,
        "shot_coverage": shot_coverage,
        "null_clock": null_clock,
    }


def write_audit_report(manifest: dict[str, Any]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audits = [audit_season(s) for s in SEASONS]

    lines = [
        "# CBBD play-by-play data audit -- 2026-09-10",
        "",
        f"Source: `data/raw/cbbd/pbp/plays_{{season}}.parquet`, pulled via "
        f"`scripts/pull_cbbd_pbp.py` from `{BASE_URL}/plays/date`.",
        "",
        "Coverage is checked against games with `status == \"final\"` in "
        "`data/raw/cbbd/games_{season}.parquet` (scheduled/cancelled/postponed "
        "games are expected to have no plays).",
        "",
    ]

    all_play_types: dict[str, int] = {}
    total_rows = 0
    for a in audits:
        if "error" in a:
            lines.append(f"## Season {a['season']}\n\nERROR: {a['error']}\n")
            continue
        total_rows += a["n_rows"]
        for k, v in a["play_type_counts"].items():
            all_play_types[k] = all_play_types.get(k, 0) + v

        lines.append(f"## Season {a['season']}")
        lines.append("")
        lines.append(f"- Games covered: {a['n_covered_games']} / {a['n_final_games']} final games "
                      f"(missing: {a['n_missing_games']})")
        if a["n_missing_games"]:
            lines.append(f"  - Example missing gameIds: {a['example_missing_ids']}")
        lines.append(f"- Rows: {a['n_rows']:,}")
        oc = a["on_floor_completeness"]
        lines.append(f"- onFloor completeness (exactly 5 home + 5 away ids): "
                      f"{oc:.2%}" if oc is not None else "- onFloor completeness: n/a (no rows)")
        lines.append(f"- shootingPlay share: {a['shooting_share']:.2%}" if a["shooting_share"] is not None else
                      "- shootingPlay share: n/a")
        lines.append(f"- Rows with null clock: {a['null_clock']:,}")
        lines.append("- shotInfo field coverage (share non-null):")
        for f, cov in a["shot_coverage"].items():
            lines.append(f"  - {f}: {cov:.2%}")
        lines.append("- playType counts:")
        for pt, cnt in sorted(a["play_type_counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {pt}: {cnt:,}")
        lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Total rows across all seasons: {total_rows:,}")
    lines.append(f"- Union playType vocabulary size (event dictionary): {len(all_play_types)}")
    for pt, cnt in sorted(all_play_types.items(), key=lambda kv: -kv[1]):
        lines.append(f"  - {pt}: {cnt:,}")
    lines.append("")
    lines.append(f"- Total API calls used (this manifest's running total): {manifest.get('total_calls')}")
    lines.append(f"- X-CallLimit-Remaining (final observed): {manifest.get('call_limit_remaining_final')}")
    lines.append("")

    AUDIT_PATH.write_text("\n".join(lines))
    log.info("Wrote audit report: %s", AUDIT_PATH)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    ap.add_argument("--audit-only", action="store_true", help="Skip pulling; just (re)write the audit report.")
    args = ap.parse_args()

    PBP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    if args.audit_only:
        write_audit_report(manifest)
        return 0

    key = load_api_key()
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "accept": "application/json"})
    tracker = CallTracker()

    stopped_early = False
    try:
        for season in args.seasons:
            process_season(session, tracker, manifest, season)
    except QuotaExhausted as exc:
        log.warning("STOPPING: %s", exc)
        stopped_early = True
    finally:
        manifest["entries"].extend(tracker.manifest)
        manifest["total_calls"] = manifest.get("total_calls", 0) + tracker.count
        if tracker.remaining is not None:
            manifest["call_limit_remaining_final"] = tracker.remaining
        save_manifest(manifest)
        log.info("Manifest saved: %s (total_calls=%s, remaining=%s)",
                  MANIFEST_PATH, manifest.get("total_calls"), manifest.get("call_limit_remaining_final"))

    write_audit_report(manifest)

    if stopped_early:
        log.warning("Run stopped early due to quota floor. Re-run the same command to resume "
                    "(completed per-date files are skipped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
