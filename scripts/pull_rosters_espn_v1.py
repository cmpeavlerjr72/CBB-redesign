#!/usr/bin/env python
"""
pull_rosters_espn_v1.py -- season-S rosters from ESPN team roster endpoints (free, allowed), in the CBBD roster_{S}.parquet schema.

Why: CBBD /teams/roster?season=2027 returns 1535 team rows and 0 players (checked 2026-10-05; season label 2027 = 2026-27, 2026 = 5643 players).
ESPN publishes 2026-27 preseason rosters (athletes list, season.year 2027).

Output: data/raw/cbbd/rosters/roster_{S}.parquet by default (so build_engine_inputs_day1prior_v1 / make_seed_fn("A3") read it unchanged).
An existing file is replaced only after the new pull passes the hard checks (>= MIN_TEAMS teams, >= MIN_PLAYERS players); a failed pull writes nothing.
Columns = CBBD schema (season, cbbd_team_id, cbbd_team, team_source_id [= ESPN team id], conference, cbbd_player_id, source_id [= ESPN athlete id],
name, first_name, last_name, jersey, position, height, weight, start_season, end_season) plus: espn_player_id, experience, cbbd_id_mapped,
prev_team_source_id (team in the most recent earlier CBBD roster file), is_transfer_in (prev team exists and differs), new_to_d1 (no earlier roster row).
cbbd_player_id is mapped from the ESPN athlete id via source_id in the earlier roster files; players with no earlier row (freshmen, internationals,
non-D-I arrivals) have a null cbbd_player_id (nullable Int64). They have no S-1 D-I minutes, so A3 cannot name them by construction.

Usage: .venv/Scripts/python.exe scripts/pull_rosters_espn_v1.py [--season 2027] [--out PATH] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
ROSTER_DIR = ROOT / "data" / "raw" / "cbbd" / "rosters"
TEAM_XWALK = ROOT / "data" / "reference" / "team_crosswalk.parquet"
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{tid}/roster"
MIN_TEAMS, MIN_PLAYERS = 250, 3000   # ESPN rolls teams over to the new season one by one; stale teams are excluded and listed in the report
SCHEMA = ["season", "cbbd_team_id", "cbbd_team", "team_source_id", "conference", "cbbd_player_id", "source_id", "name", "first_name", "last_name",
          "jersey", "position", "height", "weight", "start_season", "end_season"]
EXTRA = ["espn_player_id", "experience", "cbbd_id_mapped", "prev_team_source_id", "is_transfer_in", "new_to_d1"]


def fetch_team(tid: int, session: requests.Session, season: int) -> tuple[int, list, str | None]:
    err = None
    for attempt in range(3):
        try:
            r = session.get(URL.format(tid=tid), timeout=30)
            if r.status_code == 200:
                d = r.json()
                yr = (d.get("season") or {}).get("year")
                if yr is not None and int(yr) != season:
                    return tid, [], f"season.year {yr} != {season}"
                return tid, d.get("athletes") or [], None
            err = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as e:
            err = str(e)[:80]
        time.sleep(1 + attempt)
    return tid, [], err


def parse_athletes(ath: list) -> list[dict]:
    out = []
    for a in ath:
        try:
            pid = int(a["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if ((a.get("status") or {}).get("type") or "active") != "active":
            continue
        out.append({"espn_player_id": pid, "name": a.get("displayName") or a.get("fullName"), "first_name": a.get("firstName"),
                    "last_name": a.get("lastName"), "jersey": a.get("jersey"), "position": (a.get("position") or {}).get("displayName"),
                    "height": a.get("height"), "weight": a.get("weight"), "experience": (a.get("experience") or {}).get("abbreviation")})
    return out


def build(season: int, raw: dict, xw: pd.DataFrame, prior: pd.DataFrame, conf: dict) -> pd.DataFrame:
    rows = [{"team_source_id": tid, **p} for tid, ath in raw.items() for p in parse_athletes(ath)]
    df = pd.DataFrame(rows)
    x = xw.set_index("espn_team_id")
    df["cbbd_team_id"] = df["team_source_id"].map(x["cbbd_team_id"]).astype("int64")
    df["cbbd_team"] = df["team_source_id"].map(x["cbbd_name"])
    df["conference"] = df["cbbd_team_id"].map(conf)
    df["season"] = season
    df["team_source_id"] = df["team_source_id"].astype(str)
    df["source_id"] = df["espn_player_id"].astype(str)
    pr = prior.dropna(subset=["source_id"]).copy()
    pr["source_id"] = pd.to_numeric(pr["source_id"], errors="coerce"); pr = pr.dropna(subset=["source_id"]); pr["source_id"] = pr["source_id"].astype("int64")
    pr = pr.sort_values("season").drop_duplicates("source_id", keep="last").set_index("source_id")
    df["cbbd_player_id"] = df["espn_player_id"].map(pr["cbbd_player_id"]).astype("Int64")
    df["prev_team_source_id"] = pd.to_numeric(df["espn_player_id"].map(pr["team_source_id"]), errors="coerce").astype("Int64")
    df["cbbd_id_mapped"] = df["cbbd_player_id"].notna()
    df["new_to_d1"] = df["prev_team_source_id"].isna()
    df["is_transfer_in"] = df["prev_team_source_id"].notna() & (df["prev_team_source_id"] != pd.to_numeric(df["team_source_id"]).astype("Int64"))
    df["start_season"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["end_season"] = pd.array([season] * len(df), dtype="Int64")
    df = df.drop_duplicates(["team_source_id", "espn_player_id"])
    return df[SCHEMA + EXTRA].reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2027)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    out = a.out or ROSTER_DIR / f"roster_{a.season}.parquet"
    xw = pd.read_parquet(TEAM_XWALK)
    xw = xw[xw["last_season"] >= a.season - 1].dropna(subset=["cbbd_team_id"]).copy()
    xw["cbbd_team_id"] = xw["cbbd_team_id"].astype("int64")
    ids = sorted(int(t) for t in xw["espn_team_id"])
    prior = pd.concat([pd.read_parquet(p) for p in sorted(ROSTER_DIR.glob("roster_*.parquet")) if int(p.stem.split("_")[1]) < a.season],
                      ignore_index=True)
    conf = prior.sort_values("season").drop_duplicates("cbbd_team_id", keep="last").set_index("cbbd_team_id")["conference"].to_dict()
    s = requests.Session()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda t: fetch_team(t, s, a.season), ids))
    raw = {t: ath for t, ath, e in res if e is None and ath}
    errs = {t: e for t, ath, e in res if e is not None}
    empty = [t for t, ath, e in res if e is None and not ath]
    if not raw:
        print(f"HARD ERROR: ESPN returned no roster for any of {len(ids)} teams ({len(errs)} errors). Nothing written.", file=sys.stderr)
        return 2
    df = build(a.season, raw, xw, prior, conf)
    if df["team_source_id"].nunique() < MIN_TEAMS or len(df) < MIN_PLAYERS:
        print(f"HARD ERROR: only {df['team_source_id'].nunique()} teams / {len(df)} players (min {MIN_TEAMS}/{MIN_PLAYERS}); "
              f"{len(errs)} fetch errors, {len(empty)} empty. Nothing written.", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    rep = {"season": a.season, "pulled_at": datetime.now(timezone.utc).isoformat(), "source": "ESPN site.api teams/{id}/roster",
           "teams_requested": len(ids), "teams_with_roster": int(df["team_source_id"].nunique()), "players": len(df),
           "cbbd_id_mapped": int(df["cbbd_id_mapped"].sum()), "is_transfer_in": int(df["is_transfer_in"].sum()),
           "new_to_d1": int(df["new_to_d1"].sum()), "fetch_errors": {str(k): v for k, v in errs.items()}, "empty_rosters": empty, "stale_teams_excluded": sorted(k for k, v in errs.items() if "season.year" in v)}
    out.with_suffix(".report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("fetch_errors", "empty_rosters")}), f"errors={len(errs)} empty={len(empty)} -> {out}")
    return 0



DAILY_FROM = "2026-11-02"       # opening day: refresh daily from here, weekly before


def roster_due(today, season: int = 2027, out: Path | None = None) -> tuple[bool, str]:
    """Cadence: weekly (>= 7 days since the last pull) until DAILY_FROM, daily from then on. A missing file is always due."""
    out = out or ROSTER_DIR / f"roster_{season}.parquet"
    rep = out.with_suffix(".report.json")
    if not out.exists() or not rep.exists():
        return True, "no roster file yet"
    last = pd.Timestamp(json.loads(rep.read_text())["pulled_at"]).date()
    age = (today - last).days
    need = 1 if str(today) >= DAILY_FROM else 7
    return age >= need, f"last pull {last} ({age} d ago, cadence {need} d)"


def run_roster_stage(today, season: int, dry_run: bool, force: bool = False) -> dict:
    """Chain stage. Dry run reports the cadence decision and writes nothing. A failed pull raises (stage = error), never leaves an empty file."""
    due, why = roster_due(today, season)
    if not (due or force):
        return {"_status": "skipped", "why": f"not due: {why}"}
    if dry_run:
        return {"would_pull": True, "why": why}
    rc = main(["--season", str(season)])
    if rc != 0:
        raise RuntimeError("ESPN roster pull failed its hard checks (see stderr); previous file left untouched")
    rep = json.loads((ROSTER_DIR / f"roster_{season}.report.json").read_text())
    return {k: rep[k] for k in ("teams_with_roster", "players", "cbbd_id_mapped", "is_transfer_in", "new_to_d1")} | {"stale_teams": len(rep["stale_teams_excluded"])}


if __name__ == "__main__":
    sys.exit(main())
