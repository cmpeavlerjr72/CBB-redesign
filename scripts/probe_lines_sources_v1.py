#!/usr/bin/env python
"""probe_lines_sources_v1.py -- free lines source probe (ops, 2026-10-05).

Per source reports: provider list, games covered over a date range, open/close availability, timestamps.
Sources (free; no banned sites; ESPN API hosts site.api.espn.com, not www.espn.com):
  cbbd         CBBD /games (schedule denominator, sourceId = ESPN event id), /lines, /lines/providers (key CFBD_API_KEY).
               A per-line record has no capture timestamp; open = spreadOpen / overUnderOpen.
  espn         site.api.espn.com scoreboard competitions[].odds (current line + provider, no history) and
               summary pickcenter for a sample of events (open/close fields when present).
  local_history  on-disk data/raw/cbbd/lines parquet: first startDate with a line per provider and open-field
               fraction (metadata only; no score columns are read).

Usage:
  .venv/Scripts/python.exe scripts/probe_lines_sources_v1.py --start 2026-11-02 --end 2026-11-09
  .venv/Scripts/python.exe scripts/probe_lines_sources_v1.py --dry-run    # no network; prints the plan (chain dry run)
  --history-season 2026            # local_history season label (2026 = 2025-26)
Writes data/processed/lines/probe_lines_sources_v1_<UTC date>.json and always exits 0 (never fails the chain).
Schedule: run daily from ~Oct 20 with default range (today..+7d); wall-clock fetched_at is recorded per source.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = Path(r"C:\Users\devuser\cfb-props-sim\.env")
CBBD = "https://api.collegebasketballdata.com"
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
OUT_DIR = ROOT / "data" / "processed" / "lines"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def season_for(d: date) -> int:
    return d.year + 1 if d.month >= 5 else d.year


def get(url, params=None, headers=None):
    for a in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=45)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 + 2 * a)
                continue
            return {"_err": f"HTTP {r.status_code}"}
        except requests.RequestException:
            time.sleep(1 + a)
    return {"_err": "request failed"}


def probe_cbbd(start: date, end: date) -> dict:
    out = {"fetched_at": now_iso()}
    key = dotenv_values(str(ENV_PATH)).get("CFBD_API_KEY")
    if not key:
        return {"_err": "no CFBD_API_KEY"}
    h = {"Authorization": f"Bearer {key}"}
    p = {"season": season_for(start), "startDateRange": f"{start}T00:00:00Z",
         "endDateRange": f"{end + timedelta(days=1)}T00:00:00Z"}
    games = get(f"{CBBD}/games", p, h)
    lines = get(f"{CBBD}/lines", p, h)
    prov_list = get(f"{CBBD}/lines/providers", None, h)
    out["providers_listed"] = [x["name"] for x in prov_list] if isinstance(prov_list, list) else prov_list
    if isinstance(games, dict) or isinstance(lines, dict):
        out["_err"] = f"games={games if isinstance(games, dict) else 'ok'} lines={lines if isinstance(lines, dict) else 'ok'}"
        return out
    out["games_scheduled"] = len(games)
    by_date = {}
    for g in games:
        d = g["startDate"][:10]
        by_date[d] = by_date.get(d, 0) + 1
    out["games_by_date"] = by_date
    prov = {}
    for g in lines:
        for l in g.get("lines") or []:
            s = prov.setdefault(l.get("provider"), {"games": 0, "spread": 0, "total": 0, "spreadOpen": 0,
                                                    "totalOpen": 0, "moneyline": 0})
            s["games"] += 1
            s["spread"] += l.get("spread") is not None
            s["total"] += l.get("overUnder") is not None
            s["spreadOpen"] += l.get("spreadOpen") is not None
            s["totalOpen"] += l.get("overUnderOpen") is not None
            s["moneyline"] += l.get("homeMoneyline") is not None
    out["games_with_any_line"] = sum(1 for g in lines if g.get("lines"))
    out["by_provider"] = prov
    out["line_timestamp_field"] = None  # schema is provider, spread, overUnder, moneylines, spreadOpen, overUnderOpen
    return out


def _num(x):
    return x not in (None, "", {})


def probe_espn(start: date, end: date, summary_max: int) -> dict:
    out = {"fetched_at": now_iso(), "scoreboard": {}}
    ev_ids, prov = [], {}
    d = start
    while d <= end:
        sb = get(f"{ESPN_SITE}/scoreboard", {"dates": d.strftime("%Y%m%d"), "limit": 400})
        if "_err" in sb:
            out["scoreboard"][str(d)] = sb
            d += timedelta(days=1)
            continue
        evs = sb.get("events", [])
        with_odds = 0
        for ev in evs:
            for c in ev.get("competitions", []):
                o = c.get("odds") or []
                with_odds += bool(o)
                for x in o:
                    nm = (x.get("provider") or {}).get("name")
                    prov[nm] = prov.get(nm, 0) + 1
            ev_ids.append(ev["id"])
        out["scoreboard"][str(d)] = {"events": len(evs), "with_odds": with_odds}
        d += timedelta(days=1)
        time.sleep(0.2)
    out["scoreboard_providers"] = prov
    sp, n = {}, 0
    for e in ev_ids[:summary_max]:
        s = get(f"{ESPN_SITE}/summary", {"event": e})
        time.sleep(0.2)
        n += 1
        for x in s.get("pickcenter") or []:
            nm = (x.get("provider") or {}).get("name")
            ps = (x.get("pointSpread") or {}).get("home") or {}
            r = sp.setdefault(nm, {"events": 0, "has_open": 0, "has_close": 0})
            r["events"] += 1
            r["has_open"] += _num(ps.get("open")) or _num(x.get("open"))
            r["has_close"] += _num(ps.get("close")) or _num(x.get("close"))
    out["summary"] = {"sampled": n, "by_provider": sp}
    return out


def local_history(season: int) -> dict:
    """First startDate with a line per provider and open-field fraction. No score columns are read."""
    import pandas as pd
    p = ROOT / "data" / "raw" / "cbbd" / "lines" / f"lines_{season}.parquet"
    if not p.exists():
        return {"_err": f"missing {p}"}
    d = pd.read_parquet(p, columns=["gameId", "startDate", "provider", "spreadOpen", "overUnderOpen"])
    d = d[d.provider.notna()].copy()
    d["startDate"] = pd.to_datetime(d["startDate"], utc=True)
    res = {}
    for prov, g in d.groupby("provider"):
        res[prov] = {"first_start": str(g.startDate.min()), "last_start": str(g.startDate.max()),
                     "rows": int(len(g)), "spreadOpen_frac": round(float(g.spreadOpen.notna().mean()), 3),
                     "totalOpen_frac": round(float(g.overUnderOpen.notna().mean()), 3)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    today = datetime.now(timezone.utc).date()
    ap.add_argument("--start", default=str(today))
    ap.add_argument("--end", default=str(today + timedelta(days=7)))
    ap.add_argument("--summary-max", type=int, default=15)
    ap.add_argument("--history-season", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    if a.dry_run:
        print(json.dumps({"dry_run": True, "range": [a.start, a.end], "would_call": [
            "CBBD /games, /lines, /lines/providers (3 calls)",
            f"ESPN site scoreboard x{(end - start).days + 1} + summary x<={a.summary_max}",
            "local_history parquet (no network)"]}, indent=1))
        return 0
    rep = {"run_at": now_iso(), "range": [a.start, a.end], "cbbd": probe_cbbd(start, end),
           "espn": probe_espn(start, end, a.summary_max),
           "local_history_season": a.history_season, "local_history": local_history(a.history_season)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f = OUT_DIR / f"probe_lines_sources_v1_{today}.json"
    f.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    print("wrote", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
