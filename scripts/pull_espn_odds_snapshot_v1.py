#!/usr/bin/env python
"""pull_espn_odds_snapshot_v1.py -- one lines snapshot per run for a slate date (lines ops, 2026-10-05).

Sources per run (one capture timestamp for the whole run):
  espn  site.api.espn.com scoreboard competitions[].odds (current line only, no history -- our own snapshots build open/close)
  cbbd  CBBD /games (id map + startDate) and /lines for the slate; CBBD rows carry no capture time, so captured_at = fetch time
Output: data/raw/lines_snapshots/lines_snapshots_<slate date>.parquet (gitignored; append only).
  Columns: game_id (ESPN event id), cbbd_game_id, provider, spread_home (home perspective, negative = home favoured), total,
           home_ml, away_ml, spread_open, total_open (CBBD only), captured_at (UTC), source
Idempotent: key (game_id, provider, source, captured_at); a re-run with the same captured_at adds nothing. Prior rows are never edited
or removed. Fetch failures write nothing and exit 0 (never fail the chain). Usage:
  .venv/Scripts/python.exe scripts/pull_espn_odds_snapshot_v1.py --date 2026-11-02 [--dry-run] [--sources espn,cbbd]
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT_DIR = ROOT / "data" / "raw" / "lines_snapshots"
COLS = ["game_id", "cbbd_game_id", "provider", "spread_home", "total", "home_ml", "away_ml", "spread_open", "total_open",
        "captured_at", "source"]
KEY = ["game_id", "provider", "source", "captured_at"]


def out_path(d: date, root: Path = OUT_DIR) -> Path:
    return Path(root) / f"lines_snapshots_{d}.parquet"


def _f(x):
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


def parse_espn_scoreboard(sb: dict, captured_at: pd.Timestamp) -> list[dict]:
    rows = []
    for ev in sb.get("events", []) or []:
        for c in ev.get("competitions", []) or []:
            for o in c.get("odds") or []:
                sp = _f(o.get("spread"))
                h, a = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
                if sp is not None:   # ESPN gives the favourite's spread; convert to home perspective
                    if h.get("favorite"):
                        sp = -abs(sp)
                    elif a.get("favorite"):
                        sp = abs(sp)
                tot, hm, am = _f(o.get("overUnder")), _f(h.get("moneyLine")), _f(a.get("moneyLine"))
                if sp is None and tot is None and hm is None and am is None:
                    continue
                rows.append(dict(game_id=int(ev["id"]), cbbd_game_id=None, provider=(o.get("provider") or {}).get("name"),
                                 spread_home=sp, total=tot, home_ml=hm, away_ml=am, spread_open=None, total_open=None,
                                 captured_at=captured_at, source="espn"))
    return rows


def parse_cbbd_lines(games: list, cbbd_to_espn: dict, captured_at: pd.Timestamp) -> list[dict]:
    rows = []
    for g in games or []:
        cid = g.get("gameId")
        for l in g.get("lines") or []:
            rows.append(dict(game_id=cbbd_to_espn.get(cid), cbbd_game_id=cid, provider=l.get("provider"),
                             spread_home=_f(l.get("spread")), total=_f(l.get("overUnder")), home_ml=_f(l.get("homeMoneyline")),
                             away_ml=_f(l.get("awayMoneyline")), spread_open=_f(l.get("spreadOpen")),
                             total_open=_f(l.get("overUnderOpen")), captured_at=captured_at, source="cbbd"))
    return rows


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    df["game_id"] = df["game_id"].astype("Int64")
    df["cbbd_game_id"] = df["cbbd_game_id"].astype("Int64")
    for c in ("spread_home", "total", "home_ml", "away_ml", "spread_open", "total_open"):
        df[c] = df[c].astype("float64")
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    return df


def append_snapshot(new: pd.DataFrame, path: Path) -> int:
    """Append only rows whose key is not already present; existing rows are carried over untouched. Returns rows added."""
    if new.empty:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        have = set(map(tuple, old[KEY].astype(str).values))
        keep = [tuple(r) not in have for r in new[KEY].astype(str).values]
        add = new[keep]
        if add.empty:
            return 0
        both = pd.concat([old, add], ignore_index=True)
    else:
        add, both = new, new
    tmp = path.with_suffix(".tmp")
    both.to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(add)


def _get(url, params=None, headers=None):
    import requests
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


def fetch_espn(d: date) -> dict:
    import probe_lines_sources_v1 as P
    return _get(f"{P.ESPN_SITE}/scoreboard", {"dates": d.strftime("%Y%m%d"), "limit": 400})


def fetch_cbbd(d: date):
    import probe_lines_sources_v1 as P
    from dotenv import dotenv_values
    key = dotenv_values(str(P.ENV_PATH)).get("CFBD_API_KEY")
    if not key:
        return {"_err": "no CFBD_API_KEY"}, {}
    h = {"Authorization": f"Bearer {key}"}
    # slate = ET date; window padded one day each side, rows filtered to the ET date by startDate below
    p = {"season": P.season_for(d), "startDateRange": f"{d - timedelta(days=1)}T00:00:00Z",
         "endDateRange": f"{d + timedelta(days=2)}T00:00:00Z"}
    games, lines = _get(f"{P.CBBD}/games", p, h), _get(f"{P.CBBD}/lines", p, h)
    if isinstance(games, dict) or isinstance(lines, dict):
        return {"_err": f"games={games if isinstance(games, dict) else 'ok'} lines={lines if isinstance(lines, dict) else 'ok'}"}, {}
    et = lambda s: pd.Timestamp(s).tz_convert("America/New_York").date()
    ids = {g["id"]: g.get("sourceId") for g in games if et(g["startDate"]) == d}
    ids = {k: (int(v) if v not in (None, "") else None) for k, v in ids.items()}
    return [g for g in lines if g.get("gameId") in ids], ids


def run(d: date, sources=("espn", "cbbd"), root: Path = OUT_DIR, now=None, fetchers=None) -> dict:
    fe, fc = (fetchers or {}).get("espn", fetch_espn), (fetchers or {}).get("cbbd", fetch_cbbd)
    cap = (pd.Timestamp(now) if now is not None else pd.Timestamp.now("UTC"))
    cap = (cap.tz_localize("UTC") if cap.tzinfo is None else cap.tz_convert("UTC")).floor("s")
    rows, rep = [], {"date": str(d), "captured_at": str(cap)}
    if "espn" in sources:
        sb = fe(d)
        if "_err" in sb:
            rep["espn_err"] = sb["_err"]
        else:
            r = parse_espn_scoreboard(sb, cap)
            rep["espn_events"], rep["espn_rows"] = len(sb.get("events", [])), len(r)
            rows += r
    if "cbbd" in sources:
        lines, ids = fc(d)
        if isinstance(lines, dict):
            rep["cbbd_err"] = lines["_err"]
        else:
            r = parse_cbbd_lines(lines, ids, cap)
            rep["cbbd_games"], rep["cbbd_rows"] = len(ids), len(r)
            rows += r
    rep["rows_added"] = append_snapshot(to_frame(rows), out_path(d, root))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(pd.Timestamp.now("America/New_York").date()))
    ap.add_argument("--sources", default="espn,cbbd")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d = date.fromisoformat(a.date)
    if a.dry_run:
        print(json.dumps({"dry_run": True, "date": a.date, "would_write": str(out_path(d)),
                          "would_call": ["ESPN scoreboard x1" if "espn" in a.sources else None,
                                         "CBBD /games + /lines x2" if "cbbd" in a.sources else None]}))
        return 0
    try:
        print(json.dumps(run(d, tuple(a.sources.split(","))), default=str))
    except Exception as e:  # never fail the chain
        print(json.dumps({"error": repr(e)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
