#!/usr/bin/env python
"""
pull_preseason.py -- 2026-27 (season 2027) preseason data layer.

Season convention (hoopR/CBBD, ending-year): 2026-27 season == season 2027,
same as everywhere else in this repo (see `src/cbb_sim/data/universe.py`,
`data/reference/coaches.parquet`). The transfer portal and HS recruiting
endpoints below key on CBBD's `year` param, which follows the *starting*
year of the just-finished season whose offseason produced the movement
(year=2026 == the window that runs from the end of the 2025-26 season into
the 2026-27 season) -- confirmed live, see the per-endpoint notes below.

Reuses helpers from `scripts/pull_cbbd.py` (CallTracker, http_get,
load_api_key, season_windows, iso, BASE_URL, ENV_PATH) rather than
duplicating the auth/retry/rate-limit machinery -- loaded by file path since
`scripts/` is a flat, unpackaged directory (see `scripts/README.md`). The key
itself (CFBD_API_KEY, read from `C:\\Users\\devuser\\cfb-props-sim\\.env`) is
never printed or logged, exactly as in pull_cbbd.py.

Endpoints pulled, and what was verified live on 2026-09-10 before writing
this script (see docs/tests/preseason_2027_2026-09-10.md for the full
writeup):

- `/games?season=2027`, paged by the same 14-day date windows as
  pull_cbbd.py's `season_windows()` (Nov 1 2026 - May 1 2027). ~13 calls.
- `/teams/roster?season=2027` (no `team` filter): DOES support a season-only
  call -- one call returns one row per team (1519 rows, all schools CBBD
  tracks, not just the 367 D-I-ever crosswalk teams). As of the pull date
  every team's `players` list is empty (verified both on the season-only
  response and by spot-checking 8 individual power-conference teams with
  `team=` filters, each also empty) -- CBBD has not populated 2026-27
  rosters yet this early in the offseason. We still take the one
  season-only call (cheapest possible check) plus a small paid-for spot
  check (8 calls) rather than looping over all 367 D-I teams individually,
  since the spot check already shows the emptiness is a data-availability
  fact, not a season-only-param limitation.
- `/recruiting/portal?year=2026`: the endpoint filters on `year`, not
  `season` (see pull_cbbd.py's own note); year=2026 returns entries from the
  transfer window feeding 2026-27 rosters. 1 call.
- `/recruiting/players?year=2026`: NOT in the original task's endpoint list,
  but pulled anyway (1 extra call) because item 2's roster-continuity table
  needs a freshmen count and neither CBBD roster (empty, above) nor the
  transfer portal (transfers only) carries incoming high-school signees.
  This is CBBD's recruiting *board* (ranked/scouted prospects only), so it
  is a lower bound on true incoming freshmen, not a full count -- see the
  caveat in the continuity doc.
- `/teams?season=2027` and `/teams?season=2026`: conference membership for
  both seasons, to detect realignment. 2 calls.

Total CBBD calls: ~13 (games) + 1 (roster season-only) + 8 (roster spot
check) + 1 (portal) + 1 (recruiting players) + 2 (teams) = ~26, tracked
exactly in the run manifest. Budget ceiling: --max-calls (default 600).

Usage:
    .venv/Scripts/python.exe scripts/pull_preseason.py

Writes (all under data/raw/preseason/2027/):
    games_2027.parquet
    roster_teams_2027.parquet          (one row per team, n_players from the season-only call)
    roster_players_2027.parquet        (flattened players; empty schema until CBBD populates)
    roster_spot_check_2027.csv
    portal_2026.parquet
    recruiting_players_2026.parquet
    teams_2026.parquet, teams_2027.parquet
    conference_changes.csv
    manifest.json                      (CBBD call log, same shape as pull_cbbd.py's)
    report.json                        (summary numbers used in the final report / doc)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "preseason" / "2027"
HOOPR_SCHEDULE_PATH = REPO_ROOT / "data" / "raw" / "hoopr" / "schedules" / "mbb_schedule_2027.parquet"
TEAM_CROSSWALK_PATH = REPO_ROOT / "data" / "reference" / "team_crosswalk.parquet"

SEASON = 2027
PRIOR_SEASON = 2026
PORTAL_YEAR = 2026
RECRUITING_YEAR = 2026

SPOT_CHECK_TEAMS = [
    "Duke", "Kansas", "Gonzaga", "Kentucky", "UConn", "Houston", "Purdue", "UCLA",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod  # dataclass field resolution needs the module registered
    spec.loader.exec_module(mod)
    return mod


pull_cbbd = _load_module("_pull_cbbd_for_preseason", SCRIPTS_DIR / "pull_cbbd.py")

log = pull_cbbd.log


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created_at"] = now_iso()
    return df


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------
def pull_games(session, tracker) -> pd.DataFrame:
    windows = pull_cbbd.season_windows(SEASON)
    games: dict[int, dict] = {}
    for start, end in windows:
        params = {"season": SEASON, "startDateRange": start, "endDateRange": end}
        data = pull_cbbd.http_get(session, tracker, "/games", params)
        for g in data:
            games[g["id"]] = g
    df = _stamp(pd.DataFrame(list(games.values())))
    out_path = OUT_DIR / "games_2027.parquet"
    df.to_parquet(out_path, index=False)
    tracker.note("/games", {"season": SEASON, "windows": len(windows)}, len(df), {"output": out_path.name})
    log.info("games_2027: %d scheduled games across %d windows -> %s", len(df), len(windows), out_path.name)
    return df


def pull_rosters(session, tracker) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pull_cbbd.http_get(session, tracker, "/teams/roster", {"season": SEASON})
    teams_rows = [
        {"teamId": d["teamId"], "team": d["team"], "conference": d.get("conference"), "n_players": len(d.get("players") or [])}
        for d in data
    ]
    teams_df = _stamp(pd.DataFrame(teams_rows))
    teams_df.to_parquet(OUT_DIR / "roster_teams_2027.parquet", index=False)
    tracker.note(
        "/teams/roster", {"season": SEASON}, len(data), {
            "output": "roster_teams_2027.parquet",
            "note": "season-only call returns one row per team; verified below that every team's players list is empty as of the pull date",
        },
    )

    player_rows = []
    for d in data:
        for p in (d.get("players") or []):
            row = dict(p)
            row["teamId"] = d["teamId"]
            row["team"] = d["team"]
            player_rows.append(row)
    players_df = _stamp(pd.DataFrame(player_rows))
    players_df.to_parquet(OUT_DIR / "roster_players_2027.parquet", index=False)

    log.info(
        "roster_teams_2027: %d teams from the season-only call, %d with a non-empty players list, %d total player rows",
        len(teams_df), int((teams_df["n_players"] > 0).sum()), len(players_df),
    )

    spot_rows = []
    for team in SPOT_CHECK_TEAMS:
        d2 = pull_cbbd.http_get(session, tracker, "/teams/roster", {"season": SEASON, "team": team})
        n = len(d2[0]["players"]) if d2 else None
        spot_rows.append({"team": team, "n_players": n})
        tracker.note(
            "/teams/roster", {"season": SEASON, "team": team}, len(d2) if isinstance(d2, list) else 1,
            {"purpose": "spot_check_season_only_emptiness"},
        )
    spot_df = _stamp(pd.DataFrame(spot_rows))
    spot_df.to_csv(OUT_DIR / "roster_spot_check_2027.csv", index=False)
    log.info("roster spot check (per-team calls): %s", dict(zip(spot_df["team"], spot_df["n_players"])))

    return teams_df, players_df, spot_df


def pull_portal(session, tracker) -> pd.DataFrame:
    data = pull_cbbd.http_get(session, tracker, "/recruiting/portal", {"year": PORTAL_YEAR})
    df = pd.json_normalize(data)
    if "destination.id" in df.columns:
        df["resolved_to_2027_destination"] = df["destination.id"].notna()
    else:
        df["resolved_to_2027_destination"] = False
    df = _stamp(df)
    out_path = OUT_DIR / f"portal_{PORTAL_YEAR}.parquet"
    df.to_parquet(out_path, index=False)
    tracker.note("/recruiting/portal", {"year": PORTAL_YEAR}, len(df), {"output": out_path.name})
    n_resolved = int(df["resolved_to_2027_destination"].sum())
    log.info("portal_%d: %d entries, %d resolved to a destination team, %d still unresolved",
              PORTAL_YEAR, len(df), n_resolved, len(df) - n_resolved)
    return df


def pull_recruiting_players(session, tracker) -> pd.DataFrame:
    data = pull_cbbd.http_get(session, tracker, "/recruiting/players", {"year": RECRUITING_YEAR})
    df = pd.json_normalize(data)
    df = _stamp(df)
    out_path = OUT_DIR / f"recruiting_players_{RECRUITING_YEAR}.parquet"
    df.to_parquet(out_path, index=False)
    tracker.note(
        "/recruiting/players", {"year": RECRUITING_YEAR}, len(df), {
            "output": out_path.name,
            "note": "extra endpoint beyond the item-1 list, added for the item-2 continuity table's freshmen count; "
                    "CBBD's ranked recruiting board only -- undercounts unranked/late signees",
        },
    )
    n_committed = int(df["committedTo.id"].notna().sum()) if "committedTo.id" in df.columns else 0
    log.info("recruiting_players_%d: %d ranked prospects, %d committed to a 2027 destination",
              RECRUITING_YEAR, len(df), n_committed)
    return df


def pull_teams(session, tracker) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for season in (PRIOR_SEASON, SEASON):
        data = pull_cbbd.http_get(session, tracker, "/teams", {"season": season})
        df = _stamp(pd.DataFrame(data))
        df.to_parquet(OUT_DIR / f"teams_{season}.parquet", index=False)
        tracker.note("/teams", {"season": season}, len(df), {"output": f"teams_{season}.parquet"})
        frames[season] = df
        log.info("teams_%d: %d teams", season, len(df))

    m = frames[PRIOR_SEASON][["id", "school", "conference"]].merge(
        frames[SEASON][["id", "school", "conference"]], on="id", how="outer",
        suffixes=(f"_{PRIOR_SEASON}", f"_{SEASON}"),
    )
    changed = m[m[f"conference_{PRIOR_SEASON}"] != m[f"conference_{SEASON}"]].copy()
    changed = _stamp(changed)
    changed.to_csv(OUT_DIR / "conference_changes.csv", index=False)
    log.info("conference_changes: %d teams differ (incl. teams present only one of the two seasons)", len(changed))
    return frames, changed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-calls", type=int, default=600)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    key = pull_cbbd.load_api_key()
    import requests
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "accept": "application/json"})

    tracker = pull_cbbd.CallTracker(max_calls=args.max_calls)

    report: dict = {"generated_at": now_iso(), "season": SEASON}

    try:
        games_df = pull_games(session, tracker)
        report["games_scheduled"] = int(len(games_df))

        teams_df, players_df, spot_df = pull_rosters(session, tracker)
        report["roster_teams_checked"] = int(len(teams_df))
        report["roster_teams_with_players"] = int((teams_df["n_players"] > 0).sum())
        report["roster_size_distribution"] = (
            teams_df["n_players"].describe().to_dict() if len(teams_df) else {}
        )
        report["roster_spot_check"] = dict(zip(spot_df["team"], spot_df["n_players"].astype(object)))

        portal_df = pull_portal(session, tracker)
        report["portal_entries"] = int(len(portal_df))
        report["portal_resolved_to_2027"] = int(portal_df["resolved_to_2027_destination"].sum())

        recruiting_df = pull_recruiting_players(session, tracker)
        report["recruiting_players"] = int(len(recruiting_df))
        report["recruiting_players_committed"] = (
            int(recruiting_df["committedTo.id"].notna().sum()) if "committedTo.id" in recruiting_df.columns else 0
        )

        teams_frames, changed_df = pull_teams(session, tracker)
        report["conference_changes_2026_to_2027"] = int(len(changed_df))

        if HOOPR_SCHEDULE_PATH.exists():
            hsched = pd.read_parquet(HOOPR_SCHEDULE_PATH, columns=["game_id", "game_date"])
            report["hoopr_schedule_games"] = int(len(hsched))
            report["hoopr_schedule_date_range"] = [str(hsched["game_date"].min()), str(hsched["game_date"].max())]
        else:
            report["hoopr_schedule_games"] = None
            log.warning("hoopR schedule file not found at %s", HOOPR_SCHEDULE_PATH)

    finally:
        manifest_path = OUT_DIR / "manifest.json"
        manifest = {
            "generated_at": now_iso(),
            "base_url": pull_cbbd.BASE_URL,
            "total_calls": tracker.count,
            "call_limit_remaining_final": tracker.remaining,
            "entries": tracker.manifest,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        report["cbbd_calls_used"] = tracker.count
        report_path = OUT_DIR / "report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        log.info("Wrote manifest: %s", manifest_path)
        log.info("Wrote report: %s", report_path)
        log.info("Total CBBD API calls made: %d / %d budget", tracker.count, args.max_calls)

    return 0


if __name__ == "__main__":
    sys.exit(main())
