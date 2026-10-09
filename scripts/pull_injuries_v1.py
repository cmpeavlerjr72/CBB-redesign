#!/usr/bin/env python
"""
pull_injuries_v1.py -- free injury feed -> versioned parquet + manifest -> the Out set the daily sim removes from the roster
(ops, 2026-10-09; docs/ops/injury_feed_2026-10-09.md). Wiring only: no model, parameter or probability weighting.

Source (the only compliant free source found): ESPN public JSON, one league-wide call
    https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/injuries
CBBD (api.collegebasketballdata.com) has no injury / player-status endpoint (probed 2026-10-09, all 404). hoopR carries none either.
ESPN college coverage is sparse and the payload is EMPTY in the preseason; zero rows means "nothing reported", not "everybody healthy".

Output (as-of stamped; `pulled_at` is the pull clock, `snapshot_date` the chain's calendar date):
    data/processed/injuries/injuries_<YYYY-MM-DD>.parquet   latest pull of that date (the chain is its only writer)
    data/processed/injuries/manifests/injuries_<YYYY-MM-DD>_<HHMMSS>.json   one manifest per pull (counts, sha1, source url, schema check)
columns: season, team_id (ESPN team id), cbbd_team_id, espn_player_id, cbbd_player_id, player_name, status (raw ESPN text), status_norm,
detail, source, pulled_at, snapshot_date.

APPLIED = status_norm in {"out", "out for season"} and ONLY that. Doubtful / Questionable / Day-To-Day / Probable / Suspension are
recorded, never applied in this version. Applying = the player is removed from the available roster before usage allocation, the way a
player missing from the roster is handled (see `run_daily_sim_v1` + `build_engine_inputs_day1prior_v1.make_seed_fn(out_pids=...)` for the day-1
seed, `build_engine_inputs_live.apply_availability` for in-season rotation priors).

Schema safety: the ESPN athlete id is read from `athlete.id`, else from the player-card link (`/id/<n>/`), else the athlete `$ref`.
A non-empty payload that yields no parsed athlete id is a HARD ERROR (schema drift), never a silent "nobody is out".

    .venv/Scripts/python.exe scripts/pull_injuries_v1.py [--date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/injuries"
OUT_DIR = ROOT / "data" / "processed" / "injuries"
ROSTER_DIR = ROOT / "data" / "raw" / "cbbd" / "rosters"
TEAM_XWALK = ROOT / "data" / "reference" / "team_crosswalk.parquet"
OVERRIDES = ROOT / "data" / "overrides" / "availability.csv"
UA = "cbb-sim-research/1.0 (personal research; one request per pass)"
COLS = ["season", "team_id", "cbbd_team_id", "espn_player_id", "cbbd_player_id", "player_name", "status", "status_norm", "detail",
        "source", "pulled_at", "snapshot_date"]
OUT_NORM = {"out", "out for season"}              # APPLIED statuses (exactly Out; ESPN's season-long variant is still Out)
_ID_RE = re.compile(r"/id/(\d+)")


def norm_status(raw) -> str:
    return re.sub(r"\s+", " ", str(raw or "").strip().lower())


def athlete_id(inj: dict):
    ath = inj.get("athlete") or {}
    if ath.get("id") not in (None, ""):
        return int(ath["id"])
    for ln in ath.get("links") or []:
        m = _ID_RE.search(str(ln.get("href") or ""))
        if m and "playercard" in (ln.get("rel") or []):
            return int(m.group(1))
    m = _ID_RE.search(str(ath.get("$ref") or ""))
    return int(m.group(1)) if m else None


def _detail(inj: dict) -> str:
    d = inj.get("details") or {}
    parts = [d.get("type"), d.get("detail"), d.get("side"), d.get("returnDate"), inj.get("shortComment")]
    return " | ".join(str(x) for x in parts if x)[:240]


def parse_league(payload: dict, pulled_at: str, snapshot_date: str) -> tuple[pd.DataFrame, dict]:
    """League-wide shape {"season": {"year"}, "injuries": [{"id": team, "injuries": [ {status, athlete, details, ...} ]}]}."""
    season = (payload.get("season") or {}).get("year")
    rows, n_raw, n_noid = [], 0, 0
    for t in payload.get("injuries") or []:
        for inj in t.get("injuries") or []:
            n_raw += 1
            aid = athlete_id(inj)
            if aid is None:
                n_noid += 1
                continue
            ath = inj.get("athlete") or {}
            tid = t.get("id") or (ath.get("team") or {}).get("id")
            raw = str(inj.get("status") or (inj.get("type") or {}).get("description") or "").strip()
            rows.append({"season": int(season) if season is not None else None, "team_id": int(tid) if tid not in (None, "") else None,
                         "espn_player_id": aid, "player_name": ath.get("displayName") or ath.get("fullName"), "status": raw,
                         "status_norm": norm_status(raw), "detail": _detail(inj), "source": "espn_league_injuries",
                         "pulled_at": pulled_at, "snapshot_date": snapshot_date})
    if n_raw and not rows:
        raise RuntimeError(f"ESPN injuries payload had {n_raw} entries but no athlete id parsed: schema drift, fix the parser")
    df = pd.DataFrame(rows, columns=[c for c in COLS if c not in ("cbbd_team_id", "cbbd_player_id")])
    return df, {"entries_raw": n_raw, "entries_without_athlete_id": n_noid, "payload_season": season, "payload_status": payload.get("status")}


def attach_ids(df: pd.DataFrame, season: int, roster_dir: Path = ROSTER_DIR, xwalk: Path = TEAM_XWALK) -> pd.DataFrame:
    """cbbd_team_id (team crosswalk) and cbbd_player_id (ESPN athlete id = roster espn_player_id / source_id; newest season wins)."""
    df = df.copy()
    tm = {}
    if Path(xwalk).exists():
        xw = pd.read_parquet(xwalk, columns=["espn_team_id", "cbbd_team_id"]).dropna()
        tm = dict(zip(xw["espn_team_id"].astype(int), xw["cbbd_team_id"].astype("int64")))
    df["cbbd_team_id"] = pd.to_numeric(df["team_id"], errors="coerce").map(tm).astype("Int64")
    m: dict = {}
    for s in range(int(season), 2021, -1):                       # newest season first; setdefault keeps the newest
        p = Path(roster_dir) / f"roster_{s}.parquet"
        if not p.exists():
            continue
        cols = pd.read_parquet(p).columns
        keycol = "espn_player_id" if "espn_player_id" in cols else "source_id"
        r = pd.read_parquet(p, columns=[keycol, "cbbd_player_id"]).dropna()
        r[keycol] = pd.to_numeric(r[keycol], errors="coerce")
        for k, v in zip(r[keycol].dropna().astype("int64"), r["cbbd_player_id"].astype("int64")):
            m.setdefault(int(k), int(v))
    df["cbbd_player_id"] = df["espn_player_id"].map(m).astype("Int64")
    return df.reindex(columns=COLS)


def fetch_league(session=None) -> dict:
    r = (session or requests).get(URL, timeout=30, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.json()


def pull(day: date, now: str | None = None, session=None, payload: dict | None = None, **kw) -> tuple[pd.DataFrame, dict]:
    now = now or pd.Timestamp.now("UTC").isoformat()
    payload = payload if payload is not None else fetch_league(session)
    df, info = parse_league(payload, now, day.isoformat())
    season = info["payload_season"] or (day.year + (1 if day.month >= 5 else 0))
    return attach_ids(df, int(season), **kw), info


def write_pull(df: pd.DataFrame, info: dict, day: date, out_dir: Path = OUT_DIR, pulled_at: str | None = None) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    path = out_dir / f"injuries_{day.isoformat()}.parquet"
    df.to_parquet(path, index=False)
    sha = hashlib.sha1(path.read_bytes()).hexdigest()
    ts = pd.Timestamp(pulled_at or pd.Timestamp.now("UTC")).strftime("%H%M%S")
    outm = df["status_norm"].isin(OUT_NORM) if len(df) else pd.Series(dtype=bool)
    man = {"source": URL, "snapshot_date": day.isoformat(), "pulled_at": pulled_at, "rows": int(len(df)), "parquet": str(path), "sha1": sha,
           "by_status": df["status"].value_counts().to_dict() if len(df) else {}, "applied_statuses": sorted(OUT_NORM),
           "out_rows": int(outm.sum()) if len(df) else 0,
           "out_rows_unmapped_to_cbbd": int((outm & df["cbbd_player_id"].isna()).sum()) if len(df) else 0,
           **info, "terms": "ESPN public JSON; see docs/ops/injury_feed_2026-10-09.md"}
    (out_dir / "manifests" / f"injuries_{day.isoformat()}_{ts}.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    return man


def manual_out(day: date, roster_dir: Path = ROSTER_DIR, overrides: Path = OVERRIDES, season: int = 2027) -> pd.DataFrame:
    """Manual `status == out` rows of data/overrides/availability.csv for `day` (athlete_id = ESPN id), in the same frame shape."""
    if not Path(overrides).exists():
        return pd.DataFrame(columns=COLS)
    m = pd.read_csv(overrides)
    if not len(m) or not {"athlete_id", "date", "status"} <= set(m.columns):
        return pd.DataFrame(columns=COLS)
    m = m[(m["date"].astype(str) == day.isoformat()) & (m["status"].astype(str).str.lower() == "out")]
    if not len(m):
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame({"season": season, "team_id": None, "espn_player_id": m["athlete_id"].astype("int64").to_numpy(), "player_name": None,
                       "status": "Out", "status_norm": "out", "detail": "manual override",
                       "source": "manual", "snapshot_date": day.isoformat(),
                       "pulled_at": pd.Timestamp(Path(overrides).stat().st_mtime, unit="s", tz="UTC").isoformat()})   # a row cannot predate its file edit
    return attach_ids(df, season, roster_dir=roster_dir)


def out_players(day: date, now, out_dir: Path = OUT_DIR, roster_dir: Path = ROSTER_DIR, overrides: Path = OVERRIDES,
                season: int = 2027) -> pd.DataFrame:
    """The Out set for the sim at clock `now`: `day`'s feed rows with status_norm in OUT_NORM plus manual 'out' rows, one row per ESPN player.
    Columns: cbbd_player_id (NA = cannot be matched to any roster: reported, not applied), espn_player_id, team_id, source, created_at.
    LEAK GUARD: a row pulled after `now` raises (the sim must not see the future)."""
    from cbb_sim.live import guards as G
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    parts = []
    p = Path(out_dir) / f"injuries_{day.isoformat()}.parquet"
    if p.exists():
        a = pd.read_parquet(p)
        parts.append(a[a["status_norm"].isin(OUT_NORM)])
    parts.append(manual_out(day, roster_dir, overrides, season))
    parts = [x for x in parts if len(x)]
    if not parts:
        return pd.DataFrame(columns=["cbbd_player_id", "espn_player_id", "team_id", "source", "created_at"])
    df = pd.concat(parts, ignore_index=True).drop_duplicates("espn_player_id")
    ts = pd.to_datetime(df["pulled_at"], utc=True, errors="coerce")
    if (ts > now).any():
        raise G.LeakGuardError(f"{int((ts > now).sum())} injury rows pulled after the sim clock {now}")
    return df.assign(created_at=ts)[["cbbd_player_id", "espn_player_id", "team_id", "source", "created_at"]]


def run_injury_feed_stage(day: date, dry_run: bool, now: str | None = None, session=None, out_dir: Path = OUT_DIR) -> dict:
    """Chain stage (before engine inputs). Writes the parquet + manifest."""
    df, info = pull(day, now, session)
    outm = df["status_norm"].isin(OUT_NORM) if len(df) else pd.Series(dtype=bool)
    res = {"rows": int(len(df)), "by_status": df["status"].value_counts().to_dict() if len(df) else {},
           "out_rows": int(outm.sum()) if len(df) else 0,
           "out_unmapped_to_cbbd": int((outm & df["cbbd_player_id"].isna()).sum()) if len(df) else 0, **info,
           "note": "ESPN college injury coverage is sparse (empty in the preseason): zero rows means 'nothing reported', not 'everyone healthy'."}
    if dry_run:
        return {**res, "output": "(dry-run, nothing written)"}
    man = write_pull(df, info, day, out_dir, pulled_at=now)
    return {**res, "output": man["parquet"], "sha1": man["sha1"][:12]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    day = date.fromisoformat(a.date) if a.date else date.today()
    print(json.dumps(run_injury_feed_stage(day, a.dry_run), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
