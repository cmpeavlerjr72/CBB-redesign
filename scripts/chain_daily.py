#!/usr/bin/env python
"""
chain_daily.py -- the in-season daily-pull chain skeleton.

WHY THIS SHAPE. `docs/postmortem/03_data_inventory.md`: last year's
`daily_run.py` was a straight-line call graph (enrich schedule -> scrape
gamelogs -> build averages -> pull KenPom -> ESPN lines -> save finals ->
publish) with no per-step isolation, and it "crashed 82 times on one
unretried scraper and died in June" (`docs/FRAMEWORK_PLAN.md` section 0,
root cause 7). One brittle step took the whole day's pipeline down with it.
This chain fixes that structurally: every step is wrapped by `run_step()`,
which catches any exception the step raises, records it, and moves on --
a raising step can never stop a later step from running (see
`tests/test_chain_daily.py`). Every step is also independently idempotent:
rerunning the chain (or just one step) for the same date overwrites/upserts
that date's artifacts rather than duplicating or corrupting them.

STEPS
  (a) schedule/scoreboard, today + yesterday        -> data/raw/schedule_daily/{date}.parquet
  (b) lines, today=open snapshot, yesterday=close    -> data/raw/lines_daily/{date}.parquet (upserted)
  (c) KenPom snapshot (requests -> Playwright)        -> data/raw/kenpom/2027/{date}_kenpom.csv
  (d) injuries: ESPN per-team + Covers.com (raw)      -> data/raw/injuries/{espn,covers}/{date}.*
  (e) manual availability overrides (read-only)       <- data/overrides/availability.csv

Every written artifact carries a `created_at` column/field (see `_stamp_df`).

Season convention: 2026-27 == season 2027 (hoopR/CBBD ending-year
convention). `current_season(date)` rolls over on July 1 (mid-offseason,
never during a live season), matching `scripts/pull_cbbd.py`'s Nov1-May1
season windows.

Reuses helpers from `scripts/pull_cbbd.py` (CallTracker, http_get,
load_api_key, iso, flatten_lines, BASE_URL) and `scripts/pull_hoopr.py`
(make_session, Target, download_one) -- loaded by file path since
`scripts/` is flat/unpackaged (see `scripts/README.md`). The CFBD_API_KEY
(from `C:\\Users\\devuser\\cfb-props-sim\\.env`) is never printed or logged.
KenPom parsing (CANON_COLS / parse_best_table / requests->Playwright
fallback) is ported from `C:\\Users\\devuser\\CBB-Monte\\pull_kenpom_table.py`
-- parsing logic only, one HTTP request per run (with its own internal
retry-on-transient-error loop against that same URL, not multiple distinct
requests).

Usage:
    .venv/Scripts/python.exe scripts/chain_daily.py --dry-run
    .venv/Scripts/python.exe scripts/chain_daily.py
    .venv/Scripts/python.exe scripts/chain_daily.py --date 2026-11-15
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

SEASON_FOLDER = 2027  # KenPom snapshot folder; re-derived per-run from the date too (see current_season)

SCHEDULE_DAILY_DIR = REPO_ROOT / "data" / "raw" / "schedule_daily"
LINES_DAILY_DIR = REPO_ROOT / "data" / "raw" / "lines_daily"
KENPOM_DIR = REPO_ROOT / "data" / "raw" / "kenpom"
INJURIES_ESPN_DIR = REPO_ROOT / "data" / "raw" / "injuries" / "espn"
INJURIES_COVERS_DIR = REPO_ROOT / "data" / "raw" / "injuries" / "covers"
OVERRIDES_PATH = REPO_ROOT / "data" / "overrides" / "availability.csv"
TEAM_CROSSWALK_PATH = REPO_ROOT / "data" / "reference" / "team_crosswalk.parquet"
HOOPR_SCHEDULE_DIR = REPO_ROOT / "data" / "raw" / "hoopr" / "schedules"
RUN_LOG_DIR = REPO_ROOT / "data" / "raw" / "chain_daily_runs"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pull_cbbd = _load_module("_pull_cbbd_for_chain", SCRIPTS_DIR / "pull_cbbd.py")
pull_hoopr = _load_module("_pull_hoopr_for_chain", SCRIPTS_DIR / "pull_hoopr.py")

log = pull_cbbd.log


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_season(d: date) -> int:
    """hoopR/CBBD ending-year season for a given calendar date. Rolls over
    July 1, squarely in the offseason, never during a live season."""
    return d.year + 1 if d.month >= 7 else d.year


def _stamp_df(df: pd.DataFrame) -> pd.DataFrame:
    """Every written artifact carries `created_at` -- this is the one place
    that happens, so every step routes its output frame through here."""
    df = df.copy()
    df["created_at"] = now_iso()
    return df


def day_window(d: date) -> tuple[str, str]:
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return pull_cbbd.iso(start), pull_cbbd.iso(end)


# --------------------------------------------------------------------------
# Step harness -- the fix for last year's single-point-of-failure chain
# --------------------------------------------------------------------------
@dataclass
class StepResult:
    name: str
    status: str  # "ok" | "error"
    detail: dict = field(default_factory=dict)
    error: Optional[str] = None
    seconds: float = 0.0


def run_step(name: str, fn: Callable[[], dict], results: list[StepResult]) -> StepResult:
    """Run one chain step. ANY exception it raises is caught here, recorded
    as an "error" StepResult, and swallowed -- it never propagates, so the
    next step always runs. This is the direct fix for last year's failure
    mode (docs/postmortem/03_data_inventory.md: one unretried scraper killed
    the whole daily pipeline)."""
    t0 = time.monotonic()
    try:
        detail = fn()
        res = StepResult(name=name, status="ok", detail=detail or {}, seconds=time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001 -- deliberate: isolate every step
        log.error("step %s FAILED: %s", name, exc)
        res = StepResult(
            name=name, status="error",
            error=f"{type(exc).__name__}: {exc}",
            detail={"traceback": traceback.format_exc(limit=6)},
            seconds=time.monotonic() - t0,
        )
    results.append(res)
    log.info("step %-12s %-5s (%.2fs) %s", name, res.status, res.seconds,
              "" if res.status == "ok" else res.error)
    return res


# --------------------------------------------------------------------------
# (a) schedule / scoreboard
# --------------------------------------------------------------------------
def step_schedule(ctx: "ChainContext") -> dict:
    today, yesterday, season = ctx.today, ctx.yesterday, ctx.season

    rows = []
    per_day_cbbd: dict[str, list[dict]] = {}
    for label, d in (("today", today), ("yesterday", yesterday)):
        start, end = day_window(d)
        games = pull_cbbd.http_get(ctx.session, ctx.tracker, "/games", {
            "season": season, "startDateRange": start, "endDateRange": end,
        })
        per_day_cbbd[label] = games
        for g in games:
            rows.append({
                "date": d.isoformat(), "which": label, "source": "cbbd",
                "game_id": g.get("sourceId"), "cbbd_game_id": g.get("id"),
                "home_team": g.get("homeTeam"), "away_team": g.get("awayTeam"),
                "home_points": g.get("homePoints"), "away_points": g.get("awayPoints"),
                "status": g.get("status"),
            })
    ctx.state["games_today_cbbd"] = per_day_cbbd["today"]
    ctx.state["games_yesterday_cbbd"] = per_day_cbbd["yesterday"]

    # hoopR second source: refresh the current-season schedule file, then
    # filter to today/yesterday. Downloaded to a scratch path so a dry run
    # never touches the canonical on-disk file; a real run refreshes it.
    scratch = ctx.scratch_dir / f"mbb_schedule_{season}_refresh.parquet"
    hsession = pull_hoopr.make_session()
    target = pull_hoopr.Target(
        dataset="schedules", season=season, dirname="schedules", stem="mbb_schedule",
        url=pull_hoopr.BASE_URL.format(dirname="schedules", stem="mbb_schedule", season=season),
        dest=scratch, optional=True,
    )
    dl = pull_hoopr.download_one(hsession, target, max_retries=3, timeout=60, backoff_base=1.5)
    hoopr_today = hoopr_yesterday = 0
    hoopr_game_ids_today: set = set()
    if dl.status in ("downloaded", "skip") and scratch.exists():
        hsched = pd.read_parquet(scratch, columns=["game_id", "game_date"])
        hsched["game_date"] = pd.to_datetime(hsched["game_date"]).dt.date
        for label, d in (("today", today), ("yesterday", yesterday)):
            sub = hsched[hsched["game_date"] == d]
            for gid in sub["game_id"].tolist():
                rows.append({
                    "date": d.isoformat(), "which": label, "source": "hoopr",
                    "game_id": gid, "cbbd_game_id": None,
                    "home_team": None, "away_team": None,
                    "home_points": None, "away_points": None, "status": None,
                })
            if label == "today":
                hoopr_today = len(sub)
                hoopr_game_ids_today = set(sub["game_id"].tolist())
            else:
                hoopr_yesterday = len(sub)
        if not ctx.dry_run:
            HOOPR_SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
            scratch.replace(HOOPR_SCHEDULE_DIR / f"mbb_schedule_{season}.parquet")
    else:
        log.warning("hoopR schedule refresh failed/unavailable: %s", dl.error)

    schedule_cols = ["date", "which", "source", "game_id", "cbbd_game_id", "home_team", "away_team",
                      "home_points", "away_points", "status"]
    out = pd.DataFrame(rows, columns=schedule_cols)
    # CBBD's sourceId comes back as a numeric string, hoopR's game_id as a
    # real int64 -- coerce both to the same nullable int dtype so pyarrow
    # doesn't choke on a mixed str/int object column (caught by
    # tests/test_chain_daily.py against a real historical date with games).
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
    out["cbbd_game_id"] = pd.to_numeric(out["cbbd_game_id"], errors="coerce").astype("Int64")
    out = _stamp_df(out)
    out_path = SCHEDULE_DAILY_DIR / f"{today.isoformat()}.parquet"
    if not ctx.dry_run:
        # always written, even with 0 rows (e.g. offseason) -- a stable,
        # schema'd artifact every run is what makes this idempotent rather
        # than "sometimes the file exists, sometimes it doesn't."
        SCHEDULE_DAILY_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)  # idempotent: same date -> same filename, full overwrite

    cbbd_today_ids = {g.get("sourceId") for g in per_day_cbbd["today"]}
    return {
        "cbbd_games_today": len(per_day_cbbd["today"]),
        "cbbd_games_yesterday": len(per_day_cbbd["yesterday"]),
        "hoopr_games_today": hoopr_today,
        "hoopr_games_yesterday": hoopr_yesterday,
        "id_overlap_today": len(cbbd_today_ids & hoopr_game_ids_today),
        "output": str(out_path) if not ctx.dry_run else f"(dry-run, would write {out_path})",
    }


# --------------------------------------------------------------------------
# (b) lines: today = open snapshot, yesterday = close snapshot
# --------------------------------------------------------------------------
def _upsert_parquet(df: pd.DataFrame, path: Path, keys: list[str]) -> int:
    """Append-with-upsert: replace any row sharing `keys` with a newer one,
    keep everything else. This is what makes the "append" lines file
    idempotent under a same-day rerun instead of duplicating rows."""
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=keys, keep="last")
    else:
        combined = df
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return len(combined)


LINES_COLS = ["gameId", "season", "seasonType", "startDate", "homeTeamId", "homeTeam", "homeConference",
              "homeScore", "awayTeamId", "awayTeam", "awayConference", "awayScore", "provider", "spread",
              "overUnder", "homeMoneyline", "awayMoneyline", "spreadOpen", "overUnderOpen",
              "snapshot_type", "game_date", "fetched_at"]


def step_lines(ctx: "ChainContext") -> dict:
    today, yesterday, season = ctx.today, ctx.yesterday, ctx.season

    frames = []
    counts = {}
    for snapshot_type, d in (("open", today), ("close", yesterday)):
        start, end = day_window(d)
        games = pull_cbbd.http_get(ctx.session, ctx.tracker, "/lines", {
            "season": season, "startDateRange": start, "endDateRange": end,
        })
        flat = pull_cbbd.flatten_lines(games)  # may be a 0-row, 0-col frame when `games` is empty
        flat = flat.reindex(columns=LINES_COLS[:-3])  # the CBBD/game columns flatten_lines defines
        flat["snapshot_type"] = snapshot_type
        flat["game_date"] = d.isoformat()
        flat["fetched_at"] = now_iso()
        frames.append(flat)
        counts[f"{snapshot_type}_rows"] = len(flat)

    combined = _stamp_df(pd.concat(frames, ignore_index=True, sort=False))
    out_path = LINES_DAILY_DIR / f"{today.isoformat()}.parquet"
    total_rows = len(combined)
    if not ctx.dry_run:
        # always upserted, even with 0 rows for today -- see step_schedule's
        # note on why a stable schema beats a sometimes-missing file.
        total_rows = _upsert_parquet(combined, out_path, keys=["gameId", "provider", "snapshot_type", "game_date"])

    counts["output"] = str(out_path) if not ctx.dry_run else f"(dry-run, would upsert into {out_path})"
    counts["total_rows_in_file_after_run"] = total_rows if not ctx.dry_run else None
    return counts


# --------------------------------------------------------------------------
# (c) KenPom snapshot -- parsing ported from CBB-Monte/pull_kenpom_table.py
# --------------------------------------------------------------------------
CANON_COLS = [
    "Rk", "Team", "Conf", "W-L",
    "NetRtg",
    "ORtg", "ORnk",
    "DRtg", "DRnk",
    "AdjT", "AdjTRnk",
    "Luck", "LuckRnk",
    "SOSNetRtg", "SOSNetRnk",
    "SOSORtg", "SOSORnk",
    "SOSDRtg", "SOSDRnk",
    "NCSOSNetRtg", "NCSOSNetRnk",
]
KENPOM_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
KENPOM_TIMEOUT = 30
KENPOM_MAX_RETRIES = 3
KENPOM_BASE_BACKOFF = 3.0


def kenpom_season_from_date(d: date) -> int:
    """Ported verbatim from pull_kenpom_table.py's season_from_today: CBB
    season is named for the spring year; May-Dec rolls to next year."""
    return d.year + 1 if d.month >= 5 else d.year


def kenpom_url_for_season(season: int) -> str:
    return f"https://kenpom.com/index.php?y={season}"


def kenpom_fetch_with_requests(url: str) -> str:
    s = requests.Session()
    s.headers.update({
        "User-Agent": KENPOM_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    })
    last_exc: Optional[Exception] = None
    for attempt in range(KENPOM_MAX_RETRIES):
        try:
            resp = s.get(url, timeout=KENPOM_TIMEOUT)
            if resp.status_code == 403:
                raise requests.exceptions.HTTPError("403 Forbidden", response=resp)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 504):
                wait = KENPOM_BASE_BACKOFF * (2 ** attempt)
                log.warning("kenpom [%s] throttle/server error, retrying in %.0fs", status, wait)
                time.sleep(wait)
                last_exc = e
                continue
            raise
        except requests.exceptions.RequestException as e:
            wait = KENPOM_BASE_BACKOFF * (attempt + 1)
            log.warning("kenpom [NET] %s, retrying in %.0fs", e, wait)
            time.sleep(wait)
            last_exc = e
            continue
    raise RuntimeError(f"Failed to fetch {url} after {KENPOM_MAX_RETRIES} attempts: {last_exc}")


def kenpom_fetch_with_playwright(url: str) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "Playwright fallback unavailable (requests path failed and `playwright` is not "
            "installed in this venv). Install with: pip install playwright && "
            "python -m playwright install chromium"
        ) from e
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(user_agent=KENPOM_USER_AGENT, viewport={"width": 1400, "height": 900})
            page = context.new_page()
            page.goto(url, timeout=60_000, wait_until="load")
            page.wait_for_timeout(1500)
            return page.content()
        finally:
            browser.close()


def kenpom_parse_best_table(html_text: str) -> pd.DataFrame:
    """Ported verbatim (logic-for-logic) from pull_kenpom_table.py."""
    tables = pd.read_html(io.StringIO(html_text))
    if not tables:
        raise RuntimeError("No HTML tables found.")

    candidates = [(i, t.shape[0] * t.shape[1]) for i, t in enumerate(tables) if t.shape[0] >= 10 and t.shape[1] >= 5]
    idx = max(candidates, key=lambda x: x[1])[0] if candidates else 0
    df = tables[idx].copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if str(x) != "None"]).strip() for tup in df.columns]
    df.columns = [str(col).strip() if str(col).strip() else "" for col in df.columns]

    first_row = df.iloc[0].astype(str).tolist()
    if first_row[:4] == ["Rk", "Team", "Conf", "W-L"]:
        df.columns = first_row
        df = df.iloc[1:].reset_index(drop=True)

    while len(df.columns) > len(CANON_COLS) and (str(df.columns[0]).lower().startswith("unnamed") or df.columns[0] == ""):
        df = df.drop(columns=df.columns[0])
    if len(df.columns) > len(CANON_COLS):
        df = df.iloc[:, :len(CANON_COLS)]
    while len(df.columns) < len(CANON_COLS):
        df[f"_pad_{len(df.columns)}"] = pd.NA
    df.columns = CANON_COLS

    for c in ["Team", "Conf", "W-L"]:
        df[c] = df[c].astype(str).str.strip()
    for c in [c for c in CANON_COLS if c not in ("Team", "Conf", "W-L")]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Rk"] = pd.to_numeric(df["Rk"], errors="coerce").astype("Int64")
    return df


def step_kenpom(ctx: "ChainContext") -> dict:
    today = ctx.today
    season = kenpom_season_from_date(today)
    url = kenpom_url_for_season(season)

    fetch_method = "requests"
    try:
        html_text = kenpom_fetch_with_requests(url)
    except Exception as exc:
        log.warning("kenpom requests path failed (%s); falling back to Playwright", exc)
        fetch_method = "playwright"
        html_text = kenpom_fetch_with_playwright(url)

    df = kenpom_parse_best_table(html_text)
    df.insert(0, "snapshot_date", today.isoformat())
    df = _stamp_df(df)

    out_dir = KENPOM_DIR / str(season)
    out_path = out_dir / f"{today.isoformat()}_kenpom.csv"
    if not ctx.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return {
        "fetch_method": fetch_method,
        "season": season,
        "n_teams": len(df),
        "n_cols": df.shape[1],
        "output": str(out_path) if not ctx.dry_run else f"(dry-run, would write {out_path})",
    }


# --------------------------------------------------------------------------
# (d) injuries -- ESPN per-team + Covers.com, stored raw
# --------------------------------------------------------------------------
def step_injuries(ctx: "ChainContext") -> dict:
    today = ctx.today

    # scope ESPN per-team calls to teams actually playing today (reuse step
    # (a)'s fetch if it already ran this chain; otherwise fetch independently
    # so this step stays correct even if step (a) failed).
    games_today = ctx.state.get("games_today_cbbd")
    if games_today is None:
        start, end = day_window(today)
        games_today = pull_cbbd.http_get(ctx.session, ctx.tracker, "/games", {
            "season": ctx.season, "startDateRange": start, "endDateRange": end,
        })

    cbbd_ids = set()
    for g in games_today:
        cbbd_ids.add(g.get("homeTeamId"))
        cbbd_ids.add(g.get("awayTeamId"))
    cbbd_ids.discard(None)

    espn_ids: set[int] = set()
    if cbbd_ids and TEAM_CROSSWALK_PATH.exists():
        tc = pd.read_parquet(TEAM_CROSSWALK_PATH, columns=["espn_team_id", "cbbd_team_id"])
        tc = tc[tc["cbbd_team_id"].isin(cbbd_ids)]
        espn_ids = set(int(x) for x in tc["espn_team_id"].tolist())

    espn_payload: dict[str, Any] = {}
    for eid in sorted(espn_ids):
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{eid}/injuries"
        try:
            r = requests.get(url, timeout=15)
            espn_payload[str(eid)] = {"status_code": r.status_code, "body": r.json() if r.ok else r.text}
        except requests.RequestException as e:
            espn_payload[str(eid)] = {"error": str(e)}

    espn_envelope = {"fetched_at": now_iso(), "created_at": now_iso(), "date": today.isoformat(),
                      "n_teams": len(espn_ids), "teams": espn_payload}
    espn_out_path = INJURIES_ESPN_DIR / f"{today.isoformat()}.json"
    if not ctx.dry_run:
        INJURIES_ESPN_DIR.mkdir(parents=True, exist_ok=True)
        espn_out_path.write_text(json.dumps(espn_envelope, indent=2, default=str))

    # Covers.com NCAAB injuries page: fetched whole (one page, all teams),
    # stored raw ("no parsing beyond a table; unstructured is fine"). Try an
    # opportunistic table parse; fall back to raw HTML if no HTML-table
    # parser is available in this environment.
    covers_url = "https://www.covers.com/sport/basketball/ncaab/injuries"
    covers_status = "raw_html"
    covers_rows = None
    r = requests.get(covers_url, timeout=20, headers={"User-Agent": KENPOM_USER_AGENT})
    r.raise_for_status()
    parsed_csv = None
    try:
        tables = pd.read_html(io.StringIO(r.text))
        if tables:
            best = max(tables, key=lambda t: t.shape[0] * t.shape[1])
            best = _stamp_df(best)
            parsed_csv = best.to_csv(index=False)
            covers_rows = len(best)
            covers_status = "parsed_table"
    except Exception as exc:  # noqa: BLE001 -- unstructured fallback is explicitly allowed
        log.info("covers.com table parse unavailable (%s); storing raw HTML only", exc)

    covers_html_path = INJURIES_COVERS_DIR / f"{today.isoformat()}.html"
    covers_meta_path = INJURIES_COVERS_DIR / f"{today.isoformat()}.meta.json"
    covers_csv_path = INJURIES_COVERS_DIR / f"{today.isoformat()}.csv"
    if not ctx.dry_run:
        INJURIES_COVERS_DIR.mkdir(parents=True, exist_ok=True)
        covers_html_path.write_text(r.text, encoding="utf-8", errors="replace")
        covers_meta_path.write_text(json.dumps({
            "fetched_at": now_iso(), "created_at": now_iso(), "source_url": covers_url,
            "status_code": r.status_code, "content_length": len(r.text), "parse_status": covers_status,
        }, indent=2))
        if parsed_csv is not None:
            covers_csv_path.write_text(parsed_csv)

    return {
        "espn_teams_checked": len(espn_ids),
        "espn_output": str(espn_out_path) if not ctx.dry_run else f"(dry-run, would write {espn_out_path})",
        "covers_status": covers_status,
        "covers_rows_parsed": covers_rows,
        "covers_output": str(covers_html_path) if not ctx.dry_run else f"(dry-run, would write {covers_html_path})",
    }


# --------------------------------------------------------------------------
# (e) manual availability overrides -- read-only
# --------------------------------------------------------------------------
def step_overrides(ctx: "ChainContext") -> dict:
    if not OVERRIDES_PATH.exists():
        return {"exists": False, "n_rows": 0, "n_for_today": 0,
                "note": f"{OVERRIDES_PATH} not found -- treated as no overrides"}
    df = pd.read_csv(OVERRIDES_PATH)
    n_today = 0
    if "date" in df.columns:
        n_today = int((df["date"].astype(str) == ctx.today.isoformat()).sum())
    return {"exists": True, "n_rows": len(df), "n_for_today": n_today, "path": str(OVERRIDES_PATH)}


# --------------------------------------------------------------------------
# Chain orchestration
# --------------------------------------------------------------------------
@dataclass
class ChainContext:
    today: date
    yesterday: date
    season: int
    dry_run: bool
    session: requests.Session
    tracker: Any
    scratch_dir: Path
    state: dict = field(default_factory=dict)


def build_context(run_date: date, dry_run: bool, max_calls: int = 600) -> ChainContext:
    key = pull_cbbd.load_api_key()
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "accept": "application/json"})
    tracker = pull_cbbd.CallTracker(max_calls=max_calls)
    scratch_dir = REPO_ROOT / "data" / "raw" / "_chain_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return ChainContext(
        today=run_date, yesterday=run_date - timedelta(days=1),
        season=current_season(run_date), dry_run=dry_run,
        session=session, tracker=tracker, scratch_dir=scratch_dir,
    )


STEPS: list[tuple[str, Callable[[ChainContext], dict]]] = [
    ("schedule", step_schedule),
    ("lines", step_lines),
    ("kenpom", step_kenpom),
    ("injuries", step_injuries),
    ("overrides", step_overrides),
]


def run_chain(ctx: ChainContext) -> list[StepResult]:
    results: list[StepResult] = []
    for name, fn in STEPS:
        run_step(name, lambda fn=fn: fn(ctx), results)
    return results


def print_report(ctx: ChainContext, results: list[StepResult]) -> None:
    mode = "DRY RUN" if ctx.dry_run else "LIVE RUN"
    print(f"\n=== chain_daily {mode} -- date={ctx.today.isoformat()} season={ctx.season} ===")
    for r in results:
        print(f"  [{r.status.upper():5}] {r.name:10} ({r.seconds:5.2f}s)  {r.detail if r.status == 'ok' else r.error}")
    n_ok = sum(1 for r in results if r.status == "ok")
    print(f"--- {n_ok}/{len(results)} steps ok. CBBD calls used: {ctx.tracker.count} ---\n")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="run every step but write nothing; report status only")
    ap.add_argument("--date", type=str, default=None, help="run date YYYY-MM-DD (default: today)")
    ap.add_argument("--max-calls", type=int, default=600)
    args = ap.parse_args(argv)

    run_date = date.fromisoformat(args.date) if args.date else date.today()
    ctx = build_context(run_date, dry_run=args.dry_run, max_calls=args.max_calls)
    results = run_chain(ctx)
    print_report(ctx, results)

    if not ctx.dry_run:
        RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RUN_LOG_DIR / f"{run_date.isoformat()}.json"
        log_path.write_text(json.dumps({
            "created_at": now_iso(), "date": run_date.isoformat(), "season": ctx.season,
            "cbbd_calls_used": ctx.tracker.count,
            "steps": [{"name": r.name, "status": r.status, "detail": r.detail, "error": r.error, "seconds": r.seconds}
                      for r in results],
        }, indent=2, default=str))

    return 0  # the chain itself always "succeeds" -- per-step status is the signal, not the process exit code


if __name__ == "__main__":
    sys.exit(main())
