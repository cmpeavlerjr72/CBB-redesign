"""Kalshi leaderboard / top-trader collector.

Public endpoints only. No login, no API keys. One process, one thread per task,
one SQLite database (WAL) plus gzipped JSONL for the raw trade tape.

Tasks
  leaderboard  every board (metric x period x category) -> leaderboard_snapshots, handles
  metrics      profile/metrics for every handle seen -> handle_metrics (profit time series)
  profile      profile (followers, joined, top_categories) daily -> handle_profiles
  fills        social/trades for Full-sharing handles -> fills (backfill to exhaustion)
  social_feed  global social trade feed -> social_feed (only rows carrying a nickname)
  tape         global /markets/trades -> data/tape/YYYY-MM-DD.jsonl.gz + tape_5min
  markets      series + markets snapshots -> series, markets (ticker -> category, expiry)

Run:  python collector.py            (foreground, all tasks)
      python collector.py --once     (one pass of each task, then exit; smoke test)
      python collector.py --tasks leaderboard,metrics
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://kalshi.com/",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- rate limiter
class Bucket:
    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def take(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.05)


class Client:
    def __init__(self, cfg, log):
        self.base = cfg["base_url"]
        self.buckets = {k: Bucket(v) for k, v in cfg["rate_per_sec"].items()}
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.log = log
        self.backoff_until = 0.0

    def get(self, path: str, params: dict | None = None, kind: str = "social"):
        """Returns (status, json_or_text). Handles 429 with a global pause."""
        for attempt in range(4):
            wait = self.backoff_until - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.buckets[kind].take()
            try:
                r = self.s.get(self.base + path, params=params, timeout=25)
            except requests.RequestException as e:
                self.log.warning("net error %s %s: %s", path, params, e)
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429:
                self.backoff_until = time.monotonic() + 30
                self.log.warning("429 on %s; pausing 30s", path)
                continue
            if r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, r.text[:200]
        return -1, None


# ---------------------------------------------------------------- storage
SCHEMA = """
CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
  ts TEXT, metric TEXT, period TEXT, category TEXT, rank INTEGER, nickname TEXT,
  value REAL, is_anonymous INTEGER, social_id TEXT, covered_users INTEGER, provisional INTEGER
);
CREATE INDEX IF NOT EXISTS ix_lb_nick ON leaderboard_snapshots(nickname, ts);
CREATE INDEX IF NOT EXISTS ix_lb_board ON leaderboard_snapshots(metric, period, category, ts);

CREATE TABLE IF NOT EXISTS handles (
  nickname TEXT PRIMARY KEY, social_id TEXT, first_seen TEXT, last_seen TEXT,
  best_rank INTEGER, visibility TEXT, visibility_checked TEXT, profile_checked TEXT,
  metrics_checked TEXT, fills_cursor_done INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS handle_metrics (
  ts TEXT, nickname TEXT, social_id TEXT, period TEXT, volume REAL, dollars_traded REAL,
  pnl_raw INTEGER, pnl_usd REAL, num_markets INTEGER, open_interest REAL
);
CREATE INDEX IF NOT EXISTS ix_hm ON handle_metrics(nickname, period, ts);

CREATE TABLE IF NOT EXISTS handle_profiles (
  ts TEXT, nickname TEXT, social_id TEXT, follower_count INTEGER, following_count INTEGER,
  posts_count INTEGER, joined_at TEXT, profile_view_count INTEGER, top_categories TEXT, description TEXT
);

CREATE TABLE IF NOT EXISTS fills (
  trade_id TEXT PRIMARY KEY, nickname TEXT, role TEXT, ticker TEXT, market_id TEXT,
  price INTEGER, price_dollars TEXT, count REAL, taker_side TEXT, taker_action TEXT,
  maker_nickname TEXT, taker_nickname TEXT, create_date TEXT, fetched_ts TEXT
);
CREATE INDEX IF NOT EXISTS ix_fills_nick ON fills(nickname, create_date);
CREATE INDEX IF NOT EXISTS ix_fills_ticker ON fills(ticker, create_date);

CREATE TABLE IF NOT EXISTS social_feed (
  trade_id TEXT PRIMARY KEY, ticker TEXT, market_id TEXT, price INTEGER, count REAL,
  taker_side TEXT, taker_action TEXT, maker_nickname TEXT, taker_nickname TEXT,
  create_date TEXT, fetched_ts TEXT
);
CREATE INDEX IF NOT EXISTS ix_sf_maker ON social_feed(maker_nickname, create_date);
CREATE INDEX IF NOT EXISTS ix_sf_taker ON social_feed(taker_nickname, create_date);

CREATE TABLE IF NOT EXISTS tape_5min (
  bucket TEXT, ticker TEXT, trades INTEGER, contracts REAL, taker_yes_contracts REAL,
  block_trades INTEGER, yes_px_x_contracts REAL, max_print REAL,
  PRIMARY KEY (bucket, ticker)
);

CREATE TABLE IF NOT EXISTS series (
  ticker TEXT PRIMARY KEY, title TEXT, category TEXT, categories TEXT, frequency TEXT,
  tags TEXT, settlement_sources TEXT, snapshot_ts TEXT
);
CREATE TABLE IF NOT EXISTS markets (
  ticker TEXT PRIMARY KEY, event_ticker TEXT, series_ticker TEXT, title TEXT, status TEXT,
  open_time TEXT, close_time TEXT, expiration_time TEXT, result TEXT, volume REAL,
  open_interest REAL, last_price INTEGER, snapshot_ts TEXT
);
CREATE TABLE IF NOT EXISTS task_state (task TEXT PRIMARY KEY, state TEXT, updated TEXT);
"""


class DB:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        c = sqlite3.connect(path)
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)
        c.commit()
        c.close()

    def conn(self):
        c = sqlite3.connect(self.path, timeout=60)
        c.execute("PRAGMA busy_timeout=60000")
        return c

    def run(self, fn):
        with self.lock:
            c = self.conn()
            try:
                out = fn(c)
                c.commit()
                return out
            finally:
                c.close()

    def get_state(self, task):
        return self.run(lambda c: (c.execute("SELECT state FROM task_state WHERE task=?", (task,)).fetchone() or [None])[0])

    def set_state(self, task, state):
        self.run(lambda c: c.execute("INSERT OR REPLACE INTO task_state VALUES (?,?,?)", (task, state, utcnow())))


# ---------------------------------------------------------------- tasks
class Task:
    name = "task"

    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.db = ctx.db
        self.api = ctx.api
        self.log = logging.getLogger(self.name)

    def interval(self):
        return self.cfg[self.name]["interval_sec"]

    def run_once(self):
        raise NotImplementedError

    def loop(self, stop: threading.Event):
        while not stop.is_set():
            t0 = time.time()
            try:
                self.run_once()
            except Exception:
                self.log.exception("task crashed; continuing")
            dt = time.time() - t0
            stop.wait(max(1.0, self.interval() - dt))


class Leaderboard(Task):
    name = "leaderboard"

    def run_once(self):
        c = self.cfg["leaderboard"]
        ts = utcnow()
        rows, seen = [], {}
        for metric in c["metrics"]:
            for period in c["periods"]:
                for cat in c["categories"]:
                    params = {"metric_name": metric, "time_period": period, "limit": c["limit"]}
                    if cat:
                        params["category"] = cat
                    st, d = self.api.get("/v1/social/leaderboard", params)
                    if st != 200 or not isinstance(d, dict):
                        self.log.warning("board %s/%s/%s -> %s", metric, period, cat, st)
                        continue
                    cov = d.get("coverage") or {}
                    for r in d.get("rank_list", []):
                        rows.append((ts, metric, period, cat or "ALL", r.get("rank"), r.get("nickname"),
                                     r.get("value"), int(bool(r.get("is_anonymous"))), r.get("social_id") or None,
                                     cov.get("covered_users"), int(bool(cov.get("provisional")))))
                        n = r.get("nickname")
                        if n:
                            seen[n] = min(seen.get(n, 10**9), r.get("rank") or 10**9)

        def write(conn):
            conn.executemany("INSERT INTO leaderboard_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            for n, rk in seen.items():
                conn.execute(
                    "INSERT INTO handles(nickname, first_seen, last_seen, best_rank) VALUES (?,?,?,?) "
                    "ON CONFLICT(nickname) DO UPDATE SET last_seen=excluded.last_seen, "
                    "best_rank=MIN(COALESCE(best_rank, 1000000), excluded.best_rank)", (n, ts, ts, rk))
        self.db.run(write)
        self.log.info("boards: %d rows, %d distinct handles", len(rows), len(seen))


class Metrics(Task):
    name = "metrics"

    def _handles(self):
        """Hot set every cycle: handles ranked <= hot_rank on any projected_pnl board in the last
        hot_days, plus Full-sharing handles. Everyone else on the cold interval."""
        c = self.cfg["metrics"]
        now = datetime.now(timezone.utc)
        hot_cut = (now - timedelta(days=c["hot_days"])).strftime("%Y-%m-%dT%H:%M:%SZ")
        cold_cut = (now - timedelta(seconds=c["cold_interval_sec"])).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.db.run(lambda conn: conn.execute(
            "SELECT h.nickname, h.last_seen, h.metrics_checked, h.profile_checked FROM handles h "
            "WHERE h.visibility='Full' OR h.metrics_checked IS NULL OR h.metrics_checked < ? "
            "OR h.nickname IN (SELECT DISTINCT nickname FROM leaderboard_snapshots "
            "   WHERE metric='projected_pnl' AND rank <= ? AND ts >= ?)",
            (cold_cut, c["hot_rank"], hot_cut)).fetchall())

    def run_once(self):
        c = self.cfg["metrics"]
        handles = self._handles()
        now = datetime.now(timezone.utc)
        day_key = now.strftime("%Y-%m-%d")
        n_ok = 0
        for nick, last_seen, mchk, pchk in handles:
            periods = [c["period_every_cycle"]]
            if not mchk or mchk[:10] != day_key:
                periods += c["periods_daily"]
            ts = utcnow()
            rows = []
            sid = None
            for p in periods:
                st, d = self.api.get("/v1/social/profile/metrics", {"nickname": nick, "time_period": p})
                if st != 200 or not isinstance(d, dict):
                    continue
                m = d.get("metrics") or {}
                sid = d.get("social_id") or sid
                pnl_raw = m.get("pnl")
                rows.append((ts, nick, sid, p, m.get("volume"), m.get("dollars_traded"), pnl_raw,
                             (pnl_raw / 10000.0) if isinstance(pnl_raw, (int, float)) else None,
                             m.get("num_markets_traded"), m.get("open_interest")))
            prof = None
            if not pchk or (now - datetime.fromisoformat(pchk.replace("Z", "+00:00"))).total_seconds() > c["profile_interval_sec"]:
                st, d = self.api.get("/v1/social/profile", {"nickname": nick})
                if st == 200 and isinstance(d, dict):
                    sp = d.get("social_profile") or {}
                    sid = sp.get("social_id") or sid
                    prof = (ts, nick, sid, sp.get("follower_count"), sp.get("following_count"), sp.get("posts_count"),
                            sp.get("joined_at"), sp.get("profile_view_count"), json.dumps(sp.get("top_categories")),
                            sp.get("description"))

            def write(conn):
                conn.executemany("INSERT INTO handle_metrics VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
                if prof:
                    conn.execute("INSERT INTO handle_profiles VALUES (?,?,?,?,?,?,?,?,?,?)", prof)
                    conn.execute("UPDATE handles SET profile_checked=? WHERE nickname=?", (ts, nick))
                conn.execute("UPDATE handles SET metrics_checked=?, social_id=COALESCE(?, social_id) WHERE nickname=?",
                             (ts, sid, nick))
            self.db.run(write)
            n_ok += bool(rows)
        self.log.info("metrics: %d/%d handles updated", n_ok, len(handles))


class Fills(Task):
    name = "fills"

    def run_once(self):
        c = self.cfg["fills"]
        recheck = (datetime.now(timezone.utc) - timedelta(seconds=c["visibility_recheck_sec"])).strftime("%Y-%m-%dT%H:%M:%SZ")
        handles = self.db.run(lambda conn: conn.execute(
            "SELECT nickname, visibility, visibility_checked, fills_cursor_done FROM handles "
            "WHERE visibility='Full' OR visibility IS NULL OR visibility_checked < ?", (recheck,)).fetchall())
        n_new = 0
        for nick, vis, vchk, done in handles:
            known = set()
            if vis == "Full":
                known = set(r[0] for r in self.db.run(lambda conn: conn.execute(
                    "SELECT trade_id FROM fills WHERE nickname=? ORDER BY create_date DESC LIMIT 2000", (nick,)).fetchall()))
            cursor, pages, rows, hit_known = None, 0, [], False
            new_vis = vis
            while pages < c["max_pages_per_run"]:
                params = {"nickname": nick, "page_size": c["page_size"]}
                if cursor:
                    params["cursor"] = cursor
                st, d = self.api.get("/v1/social/trades", params)
                if st != 200 or not isinstance(d, dict):
                    break
                vstate = d.get("visibility_state")
                trades = d.get("trades") or []
                if pages == 0:
                    new_vis = "Full" if trades or vstate in ("visible", "full") else ("hidden" if vstate == "hidden" else (vis or "unknown"))
                    if not trades:
                        break
                ts = utcnow()
                for t in trades:
                    tid = t.get("trade_id")
                    if tid in known:
                        hit_known = True
                        break
                    role = "maker" if t.get("maker_nickname") == nick else ("taker" if t.get("taker_nickname") == nick else "unknown")
                    rows.append((tid, nick, role, t.get("ticker"), t.get("market_id"), t.get("price"), t.get("price_dollars"),
                                 t.get("count_fp") or t.get("count"), t.get("taker_side"), t.get("taker_action"),
                                 t.get("maker_nickname"), t.get("taker_nickname"), t.get("create_date"), ts))
                pages += 1
                cursor = d.get("cursor")
                if hit_known or not cursor or (done and vis == "Full" and pages >= 3 and not hit_known and False):
                    break
            exhausted = int(bool(done) or (pages > 0 and not cursor) or hit_known)

            def write(conn):
                conn.executemany("INSERT OR IGNORE INTO fills VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                conn.execute("UPDATE handles SET visibility=?, visibility_checked=?, fills_cursor_done=? WHERE nickname=?",
                             (new_vis, utcnow(), exhausted, nick))
            self.db.run(write)
            n_new += len(rows)
            if rows:
                self.log.info("fills %s: +%d (pages %d, exhausted=%d)", nick, len(rows), pages, exhausted)
        self.log.info("fills: +%d rows across %d handles", n_new, len(handles))


class SocialFeed(Task):
    name = "social_feed"

    def run_once(self):
        c = self.cfg["social_feed"]
        last = self.db.get_state("social_feed_last_id")
        cursor, pages, rows, newest = None, 0, [], None
        stop = False
        while pages < c["max_pages_per_cycle"] and not stop:
            params = {"page_size": c["page_size"]}
            if cursor:
                params["cursor"] = cursor
            st, d = self.api.get("/v1/social/trades", params)
            if st != 200 or not isinstance(d, dict):
                break
            trades = d.get("trades") or []
            if not trades:
                break
            ts = utcnow()
            for t in trades:
                tid = t.get("trade_id")
                newest = newest or tid
                if tid == last:
                    stop = True
                    break
                if t.get("maker_nickname") or t.get("taker_nickname"):
                    rows.append((tid, t.get("ticker"), t.get("market_id"), t.get("price"), t.get("count_fp") or t.get("count"),
                                 t.get("taker_side"), t.get("taker_action"), t.get("maker_nickname"), t.get("taker_nickname"),
                                 t.get("create_date"), ts))
            pages += 1
            cursor = d.get("cursor")
            if not cursor:
                break
        if rows:
            self.db.run(lambda conn: conn.executemany("INSERT OR IGNORE INTO social_feed VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows))
        if newest:
            self.db.set_state("social_feed_last_id", newest)
        if rows:
            self.log.debug("social_feed: +%d named rows (%d pages)", len(rows), pages)


class Tape(Task):
    name = "tape"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.dir = os.path.join(ROOT, self.cfg["tape_dir"])
        os.makedirs(self.dir, exist_ok=True)
        self._fh, self._fh_day = None, None
        self.agg = {}

    def _writer(self, day):
        """One gzip member per cycle: the file is opened in append mode per cycle and closed at
        the end of it, so a kill mid-write truncates at most the last member. Readers use
        scripts/read_tape.py, which tolerates a truncated final member."""
        if self._fh_day != day or self._fh is None:
            if self._fh:
                self._fh.close()
            self._fh = gzip.open(os.path.join(self.dir, f"{day}.jsonl.gz"), "at", encoding="utf-8")
            if self._fh_day != day:
                self._prune()
            self._fh_day = day
        return self._fh

    def _close_writer(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def _prune(self):
        """Delete raw tape files older than raw_retention_days (aggregates in SQLite are kept)."""
        days = self.cfg["tape"].get("raw_retention_days")
        if not days:
            return
        cut = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        for fn in os.listdir(self.dir):
            if fn.endswith(".jsonl.gz") and fn[:10] < cut:
                try:
                    os.remove(os.path.join(self.dir, fn))
                    self.log.info("pruned %s", fn)
                except OSError:
                    pass

    def _flush_agg(self, force=False):
        """Write every 5-minute bucket older than 10 minutes (wall clock); keep recent ones open."""
        cut = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M")
        rows = [(b, t, v["n"], v["c"], v["ty"], v["blk"], v["pv"], v["mx"])
                for (b, t), v in self.agg.items() if force or b < cut]
        if rows:
            self.db.run(lambda conn: conn.executemany(
                "INSERT INTO tape_5min VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(bucket,ticker) DO UPDATE SET "
                "trades=trades+excluded.trades, contracts=contracts+excluded.contracts, "
                "taker_yes_contracts=taker_yes_contracts+excluded.taker_yes_contracts, "
                "block_trades=block_trades+excluded.block_trades, "
                "yes_px_x_contracts=yes_px_x_contracts+excluded.yes_px_x_contracts, "
                "max_print=MAX(max_print, excluded.max_print)", rows))
            flushed = {(r[0], r[1]) for r in rows}
            self.agg = {k: v for k, v in self.agg.items() if k not in flushed}

    def run_once(self):
        c = self.cfg["tape"]
        last = self.db.get_state("tape_last_id")
        cursor, pages, newest, n, stop = None, 0, None, 0, False
        cur_bucket = None
        while pages < c["max_pages_per_cycle"] and not stop:
            params = {"limit": c["limit"]}
            if cursor:
                params["cursor"] = cursor
            st, d = self.api.get("/trade-api/v2/markets/trades", params, kind="trade")
            if st != 200 or not isinstance(d, dict):
                break
            trades = d.get("trades") or []
            if not trades:
                break
            for t in trades:
                tid = t.get("trade_id")
                newest = newest or tid
                if tid == last:
                    stop = True
                    break
                ct = t.get("created_time") or ""
                day = ct[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")
                cnt = float(t.get("count_fp") or t.get("count") or 0)
                # Raw retention. Series whose category is in raw_filter_categories (sports game
                # markets, the volume firehose) keep only prints >= raw_min_count plus a
                # deterministic sample. Every other series keeps every print raw. Every print
                # feeds the 5-minute aggregates below regardless.
                cat = self.ctx.series_category((t.get("ticker") or "").split("-")[0])
                if cat in c.get("raw_filter_categories", []):
                    min_cnt = c.get("raw_min_count_by_category", {}).get(cat, c.get("raw_min_count", 0))
                    keep = cnt >= min_cnt or (
                        c.get("raw_sample_rate", 0) > 0 and (hash(tid) % 10000) < c["raw_sample_rate"] * 10000)
                else:
                    keep = True
                if keep:
                    self._writer(day).write(json.dumps(t, separators=(",", ":")) + "\n")
                n += 1
                # 5-minute aggregate
                try:
                    mins = int(ct[14:16]) // 5 * 5
                    bucket = f"{ct[:13]}:{mins:02d}"
                except ValueError:
                    continue
                cur_bucket = cur_bucket or bucket
                yp = float(t.get("yes_price_dollars") or 0)
                k = (bucket, t.get("ticker"))
                v = self.agg.setdefault(k, {"n": 0, "c": 0.0, "ty": 0.0, "blk": 0, "pv": 0.0, "mx": 0.0})
                v["n"] += 1
                v["c"] += cnt
                v["ty"] += cnt if t.get("taker_side") == "yes" else 0.0
                v["blk"] += int(bool(t.get("is_block_trade")))
                v["pv"] += yp * cnt
                v["mx"] = max(v["mx"], cnt)
            pages += 1
            cursor = d.get("cursor")
            if not cursor:
                break
        self._close_writer()
        if newest:
            self.db.set_state("tape_last_id", newest)
        self._flush_agg()
        if n:
            self.log.debug("tape: +%d trades (%d pages)", n, pages)
        if pages >= c["max_pages_per_cycle"] and not stop:
            self.log.warning("tape: page cap hit, likely missed trades (burst)")


class Markets(Task):
    name = "markets"

    def run_once(self):
        ts = utcnow()
        # series
        cursor, srows = None, []
        for _ in range(50):
            params = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            st, d = self.api.get("/trade-api/v2/series", params, kind="trade")
            if st != 200 or not isinstance(d, dict):
                break
            for s in d.get("series") or []:
                srows.append((s.get("ticker"), s.get("title"), s.get("category"), json.dumps(s.get("categories")),
                              s.get("frequency"), json.dumps(s.get("tags")), json.dumps(s.get("settlement_sources")), ts))
            cursor = d.get("cursor")
            if not cursor:
                break
        # markets: open + recently settled
        mrows = []
        for status in ("open", "settled"):
            cursor = None
            for _ in range(200):
                params = {"limit": 1000, "status": status}
                if cursor:
                    params["cursor"] = cursor
                st, d = self.api.get("/trade-api/v2/markets", params, kind="trade")
                if st != 200 or not isinstance(d, dict):
                    break
                for m in d.get("markets") or []:
                    mrows.append((m.get("ticker"), m.get("event_ticker"), (m.get("event_ticker") or m.get("ticker") or "").split("-")[0],
                                  m.get("title"), m.get("status"), m.get("open_time"), m.get("close_time"),
                                  m.get("expiration_time"), m.get("result"), m.get("volume_fp") or m.get("volume"),
                                  m.get("open_interest_fp") or m.get("open_interest"), m.get("last_price"), ts))
                cursor = d.get("cursor")
                if not cursor:
                    break
                if status == "settled" and len(mrows) > 60000:
                    break

        def write(conn):
            conn.executemany("INSERT OR REPLACE INTO series VALUES (?,?,?,?,?,?,?,?)", srows)
            conn.executemany("INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", mrows)
        self.db.run(write)
        self.log.info("markets: %d series, %d markets", len(srows), len(mrows))


class Orderbook(Task):
    """Order-book snapshots for the most active non-sports markets: the resting-order layer that
    trade prints never show. Top-N tickers by contracts over the lookback window (from tape_5min),
    excluding configured categories, plus every open market in always_series."""
    name = "orderbook"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.dir = os.path.join(ROOT, "data", "orderbook")
        os.makedirs(self.dir, exist_ok=True)

    def _tickers(self):
        c = self.cfg["orderbook"]
        since = (datetime.now(timezone.utc) - timedelta(hours=c["lookback_hours"])).strftime("%Y-%m-%dT%H:%M")
        rows = self.db.run(lambda conn: conn.execute(
            "SELECT ticker, SUM(contracts) v FROM tape_5min WHERE bucket >= ? GROUP BY ticker ORDER BY v DESC LIMIT 2000",
            (since,)).fetchall())
        excl = set(c.get("exclude_categories", []))
        out = []
        for tk, v in rows:
            if self.ctx.series_category(tk.split("-")[0]) in excl:
                continue
            out.append(tk)
            if len(out) >= c["top_n"]:
                break
        always = tuple(c.get("always_series", []))
        if always:
            extra = self.db.run(lambda conn: conn.execute(
                "SELECT ticker FROM markets WHERE status IN ('active','open') AND series_ticker IN (%s)" % ",".join("?" * len(always)),
                always).fetchall())
            out += [t for (t,) in extra if t not in out]
        return out

    def run_once(self):
        c = self.cfg["orderbook"]
        tickers = self._tickers()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        n = 0
        with gzip.open(os.path.join(self.dir, f"{day}.jsonl.gz"), "at", encoding="utf-8") as fh:
            for tk in tickers:
                st, d = self.api.get(f"/trade-api/v2/markets/{tk}/orderbook", {"depth": c["depth"]}, kind="trade")
                if st != 200 or not isinstance(d, dict):
                    continue
                ob = d.get("orderbook_fp") or d.get("orderbook") or d
                rec = {"ts": utcnow(), "ticker": tk,
                       "yes": (ob.get("yes_dollars") or ob.get("yes") or [])[: c["depth"]],
                       "no": (ob.get("no_dollars") or ob.get("no") or [])[: c["depth"]]}
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n += 1
        self.log.info("orderbook: %d/%d snapshots", n, len(tickers))


TASKS = {t.name: t for t in (Leaderboard, Metrics, Fills, SocialFeed, Tape, Markets, Orderbook)}


class Ctx:
    """Shared state: config, db, api client, and a cached series -> category map."""

    def __init__(self):
        self._cats, self._cats_ts = {}, 0.0
        self._lock = threading.Lock()

    def series_category(self, series_ticker: str):
        with self._lock:
            if time.monotonic() - self._cats_ts > 600:
                try:
                    rows = self.db.run(lambda conn: conn.execute("SELECT ticker, category FROM series").fetchall())
                    self._cats = {t: c for t, c in rows}
                except Exception:
                    pass
                self._cats_ts = time.monotonic()
            return self._cats.get(series_ticker)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(os.path.join(ROOT, cfg["log_dir"]), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(ROOT, cfg["log_dir"], f"collector_{datetime.now():%Y-%m-%d}.log"), encoding="utf-8")])
    log = logging.getLogger("main")
    ctx = Ctx()
    ctx.cfg = cfg
    ctx.db = DB(os.path.join(ROOT, cfg["db_path"]))
    ctx.api = Client(cfg, log)
    names = [n for n in a.tasks.split(",") if n in TASKS and cfg.get(n, {}).get("enabled", True)]
    tasks = [TASKS[n](ctx) for n in names]
    if a.once:
        for t in tasks:
            log.info("once: %s", t.name)
            t.run_once()
        return
    stop = threading.Event()
    threads = [threading.Thread(target=t.loop, args=(stop,), name=t.name, daemon=True) for t in tasks]
    for th in threads:
        th.start()
    log.info("collector running: %s", names)
    try:
        while any(th.is_alive() for th in threads):
            time.sleep(5)
    except KeyboardInterrupt:
        stop.set()
        log.info("stopping")


if __name__ == "__main__":
    main()
