#!/usr/bin/env python
"""
pull_injuries_player_out_v1.py -- ESPN injuries -> a minimal "player out" table (free, allowed source; one league-wide call).

Source: https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/injuries (league-wide, all teams, 1 call). The chain's
step_injuries (chain_daily.py) separately stores raw per-team /teams/{id}/injuries payloads under data/raw/injuries/espn/{date}.json; this module can parse
that envelope too (`parse_team_envelope`). Both shapes are parsed defensively: ESPN college hoops injury coverage is sparse and the payload was EMPTY on
2026-10-05 (preseason), so the field names below follow the ESPN site API convention used for other leagues and are unverified against a live college
payload. A payload that is non-empty but yields zero parsed rows is a hard error (schema drift), never a silent "nobody is out".

Output (as-of stamped; created_at is the pull clock): data/processed/injuries/player_out_{date}.csv with the override columns
`athlete_id,date,status,note` (the data/overrides/availability.csv format; athlete_id = ESPN athlete id) plus espn_team_id, cbbd_team_id, cbbd_player_id,
raw_status, out (bool), source, created_at.

OUT rule (minimal, conservative, documented): out = raw status in {Out, Out For Season, Suspension, Suspended, Injured Reserve}. Doubtful / Questionable /
Day-To-Day / Probable are kept as rows with out=False (audit; a probabilistic availability override is an engine-side decision, not made here).
Manual rows in data/overrides/availability.csv with status 'out' for the date are merged in (source=manual) by `player_out_for`.

Usage: .venv/Scripts/python.exe scripts/pull_injuries_player_out_v1.py [--date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/injuries"
OUT_DIR = ROOT / "data" / "processed" / "injuries"
OVERRIDES = ROOT / "data" / "overrides" / "availability.csv"
ROSTER_DIR = ROOT / "data" / "raw" / "cbbd" / "rosters"
TEAM_XWALK = ROOT / "data" / "reference" / "team_crosswalk.parquet"
OUT_STATUSES = {"out", "out for season", "suspension", "suspended", "injured reserve"}
COLS = ["athlete_id", "date", "status", "note", "espn_team_id", "cbbd_team_id", "cbbd_player_id", "raw_status", "out", "source", "created_at"]


def _row(inj: dict, team_id, day: str, src: str, now: str) -> dict | None:
    ath = inj.get("athlete") or {}
    aid = ath.get("id")
    if aid is None:
        return None
    raw = str(inj.get("status") or (inj.get("type") or {}).get("description") or "").strip()
    det = inj.get("details") or {}
    note = " | ".join(str(x) for x in (det.get("type"), det.get("location"), inj.get("shortComment")) if x)[:200]
    return {"athlete_id": int(aid), "date": day, "status": "out" if raw.lower() in OUT_STATUSES else raw.lower(), "note": note,
            "espn_team_id": int(team_id) if team_id not in (None, "") else None, "raw_status": raw,
            "out": raw.lower() in OUT_STATUSES, "source": src, "created_at": now}


def parse_league(payload: dict, day: str, now: str) -> pd.DataFrame:
    """League-wide shape: {"injuries": [{"id": team, "injuries": [ {status, athlete:{id}, details, ...} ]}]}."""
    rows = []
    for t in payload.get("injuries") or []:
        for inj in t.get("injuries") or []:
            r = _row(inj, t.get("id") or (inj.get("athlete") or {}).get("team", {}).get("id"), day, "espn_league", now)
            if r:
                rows.append(r)
    return pd.DataFrame(rows, columns=[c for c in COLS if c in ("athlete_id", "date", "status", "note", "espn_team_id", "raw_status", "out", "source", "created_at")])


def parse_team_envelope(env: dict, day: str, now: str) -> pd.DataFrame:
    """chain_daily step_injuries envelope: {"teams": {espn_team_id: {"status_code", "body": {"injuries": [...]}|{}}}}."""
    rows = []
    for tid, v in (env.get("teams") or {}).items():
        body = v.get("body") if isinstance(v, dict) else None
        if not isinstance(body, dict):
            continue
        for inj in body.get("injuries") or []:
            r = _row(inj, tid, day, "espn_team", now)
            if r:
                rows.append(r)
    return pd.DataFrame(rows, columns=["athlete_id", "date", "status", "note", "espn_team_id", "raw_status", "out", "source", "created_at"])


def attach_ids(df: pd.DataFrame, season: int = 2027) -> pd.DataFrame:
    """cbbd_team_id (team crosswalk) and cbbd_player_id (ESPN athlete id = roster source_id; newest roster file wins)."""
    df = df.copy()
    xw = pd.read_parquet(TEAM_XWALK, columns=["espn_team_id", "cbbd_team_id"]).dropna()
    df["cbbd_team_id"] = pd.to_numeric(df["espn_team_id"], errors="coerce").map(dict(zip(xw["espn_team_id"].astype(int), xw["cbbd_team_id"].astype("int64")))).astype("Int64")
    ros = []
    for s in range(season, 2021, -1):
        p = ROSTER_DIR / f"roster_{s}.parquet"
        if p.exists():
            ros.append(pd.read_parquet(p, columns=["source_id", "cbbd_player_id"]).dropna())
    m = {}
    if ros:
        r = pd.concat(ros, ignore_index=True)
        r["source_id"] = pd.to_numeric(r["source_id"], errors="coerce")
        r = r.dropna(subset=["source_id"]).drop_duplicates("source_id")        # newest season first -> keep first
        m = dict(zip(r["source_id"].astype("int64"), r["cbbd_player_id"].astype("int64")))
    df["cbbd_player_id"] = df["athlete_id"].map(m).astype("Int64")
    return df.reindex(columns=COLS)


def fetch_league(session: requests.Session | None = None) -> dict:
    r = (session or requests).get(URL, timeout=30)
    r.raise_for_status()
    return r.json()


def pull(day: date, now: str | None = None, session=None, payload: dict | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now("UTC").isoformat()
    payload = payload if payload is not None else fetch_league(session)
    n_raw = sum(len(t.get("injuries") or []) for t in payload.get("injuries") or [])
    df = parse_league(payload, day.isoformat(), now)
    if n_raw and df.empty:
        raise RuntimeError(f"ESPN injuries payload had {n_raw} entries but none parsed (athlete.id missing): schema drift, fix the parser")
    return attach_ids(df)


def player_out_for(day: date, out_dir: Path = OUT_DIR, overrides: Path = OVERRIDES) -> pd.DataFrame:
    """The 'player out' set the overrides stage hands to the engine: auto rows (out=True) for `day` plus manual availability.csv rows with status out."""
    parts = []
    p = out_dir / f"player_out_{day.isoformat()}.csv"
    if p.exists():
        a = pd.read_csv(p)
        parts.append(a[a["out"].astype(bool)])
    if overrides.exists():
        m = pd.read_csv(overrides)
        if len(m) and {"athlete_id", "date", "status"} <= set(m.columns):
            m = m[(m["date"].astype(str) == day.isoformat()) & (m["status"].astype(str).str.lower() == "out")].copy()
            if len(m):
                m["source"], m["out"] = "manual", True
                parts.append(m)
    if not parts:
        return pd.DataFrame(columns=COLS)
    return pd.concat(parts, ignore_index=True).drop_duplicates("athlete_id").reindex(columns=COLS)


def run_injury_parse_stage(day: date, dry_run: bool, now: str | None = None, session=None, out_dir: Path = OUT_DIR) -> dict:
    df = pull(day, now, session)
    path = out_dir / f"player_out_{day.isoformat()}.csv"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    out = player_out_for(day, out_dir) if not dry_run else df[df["out"]]
    return {"rows": len(df), "players_out": int(len(out)), "unmapped_to_cbbd": int(out["cbbd_player_id"].isna().sum()) if len(out) else 0,
            "output": str(path) if not dry_run else f"(dry-run, would write {path})",
            "note": "ESPN college injury coverage is sparse; zero rows means 'no reported injuries', not 'everyone healthy'. No engine consumer yet."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    day = date.fromisoformat(a.date) if a.date else date.today()
    print(json.dumps(run_injury_parse_stage(day, a.dry_run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
