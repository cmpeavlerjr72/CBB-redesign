"""Quick views over the collector database.

  python report.py status                 row counts, last timestamps, task state
  python report.py boards [period]        current top 15 per category board (default monthly)
  python report.py movers [hours]         biggest all_time pnl deltas per handle over the window (default 24)
  python report.py handle <nickname>      pnl series, board appearances, category attribution, fills summary
  python report.py categories             category attribution for every handle seen on an ALL pnl board
  python report.py full                   Full-sharing handles with fill counts and series mix
"""
import os
import sqlite3
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "data", "collector.sqlite")


def q(sql, *a):
    c = sqlite3.connect(DB)
    try:
        return c.execute(sql, a).fetchall()
    finally:
        c.close()


def status():
    for t in ["leaderboard_snapshots", "handles", "handle_metrics", "handle_profiles", "fills", "social_feed", "tape_5min", "series", "markets"]:
        n = q(f"SELECT COUNT(*) FROM {t}")[0][0]
        print(f"{t:22s} {n:>10,}")
    print("snapshots:", q("SELECT MIN(ts), MAX(ts), COUNT(DISTINCT ts) FROM leaderboard_snapshots")[0])
    print("visibility:", q("SELECT visibility, COUNT(*) FROM handles GROUP BY visibility"))
    for r in q("SELECT task, state, updated FROM task_state"):
        print("state:", r[0], (r[1] or "")[:24], r[2])


def boards(period="monthly"):
    ts = q("SELECT MAX(ts) FROM leaderboard_snapshots")[0][0]
    for (cat,) in q("SELECT DISTINCT category FROM leaderboard_snapshots ORDER BY category"):
        rows = q("SELECT rank, nickname, value FROM leaderboard_snapshots WHERE ts=? AND metric='projected_pnl' "
                 "AND period=? AND category=? ORDER BY rank LIMIT 15", ts, period, cat)
        print(f"\n== {cat} / projected_pnl / {period} @ {ts}")
        for rk, n, v in rows:
            print(f"{rk:3d} {n:30s} {v:>14,.0f}")


def movers(hours=24):
    rows = q("""
      WITH s AS (SELECT nickname, ts, pnl_usd FROM handle_metrics WHERE period='all_time' AND pnl_usd IS NOT NULL),
      first AS (SELECT nickname, pnl_usd FROM s WHERE ts = (SELECT MIN(ts) FROM s s2 WHERE s2.nickname=s.nickname AND ts >= datetime('now', ?))),
      last AS (SELECT nickname, pnl_usd FROM s WHERE ts = (SELECT MAX(ts) FROM s s2 WHERE s2.nickname=s.nickname))
      SELECT l.nickname, f.pnl_usd, l.pnl_usd, l.pnl_usd - f.pnl_usd AS d FROM last l JOIN first f USING(nickname)
      ORDER BY ABS(d) DESC LIMIT 40""", f"-{int(hours)} hours")
    print(f"{'handle':30s} {'start':>14s} {'now':>14s} {'delta':>12s}")
    for n, a, b, d in rows:
        print(f"{n:30s} {a:>14,.0f} {b:>14,.0f} {d:>+12,.0f}")


def attribution(nick):
    """Category attribution: which category boards a handle appears on, and with what value share."""
    rows = q("SELECT category, period, MIN(rank), MAX(value) FROM leaderboard_snapshots "
             "WHERE nickname=? AND metric='projected_pnl' GROUP BY category, period", nick)
    out = defaultdict(dict)
    for cat, per, rk, val in rows:
        out[per][cat] = (rk, val)
    return out


def handle(nick):
    print("profile:", q("SELECT ts, follower_count, joined_at, top_categories FROM handle_profiles WHERE nickname=? ORDER BY ts DESC LIMIT 1", nick))
    print("visibility:", q("SELECT visibility, best_rank, first_seen, last_seen FROM handles WHERE nickname=?", nick))
    print("\nboards (period -> category: best rank, value):")
    for per, cats in attribution(nick).items():
        print(" ", per, {c: (rk, round(v)) for c, (rk, v) in sorted(cats.items(), key=lambda x: x[1][0])})
    print("\nall_time pnl series (last 30):")
    for ts, p, v, nm in q("SELECT ts, pnl_usd, volume, num_markets FROM handle_metrics WHERE nickname=? AND period='all_time' ORDER BY ts DESC LIMIT 30", nick):
        print(f"  {ts}  pnl {p:>14,.0f}  vol {v:>14,.0f}  mkts {nm}")
    fills = q("SELECT ticker, count, taker_side, taker_action, role, price, create_date FROM fills WHERE nickname=? ORDER BY create_date DESC LIMIT 100000", nick)
    if fills:
        ser = Counter(f[0].split("-")[0] for f in fills)
        roles = Counter(f[4] for f in fills)
        print(f"\nfills: {len(fills):,} rows, {q('SELECT MIN(create_date), MAX(create_date) FROM fills WHERE nickname=?', nick)[0]}")
        print("  roles:", dict(roles))
        print("  series:", ser.most_common(12))
    named = q("SELECT COUNT(*) FROM social_feed WHERE maker_nickname=? OR taker_nickname=?", nick, nick)[0][0]
    print("social_feed named rows:", named)


def categories():
    """For every handle on an ALL projected_pnl board, the category boards it also appears on."""
    hs = q("SELECT DISTINCT nickname FROM leaderboard_snapshots WHERE metric='projected_pnl' AND category='ALL'")
    print(f"{'handle':30s} {'vis':8s} categories (best rank)")
    for (n,) in hs:
        vis = (q("SELECT visibility FROM handles WHERE nickname=?", n) or [[None]])[0][0]
        cats = {}
        for per, cs in attribution(n).items():
            for c, (rk, v) in cs.items():
                if c != "ALL":
                    cats[c] = min(cats.get(c, 999), rk)
        print(f"{n:30s} {str(vis):8s} {sorted(cats.items(), key=lambda x: x[1])}")


def full():
    for n, in q("SELECT nickname FROM handles WHERE visibility='Full' ORDER BY nickname"):
        fl = q("SELECT COUNT(*), MIN(create_date), MAX(create_date) FROM fills WHERE nickname=?", n)[0]
        ser = Counter(r[0].split("-")[0] for r in q("SELECT ticker FROM fills WHERE nickname=?", n))
        print(f"{n:30s} fills {fl[0]:>8,} {fl[1] or ''} .. {fl[2] or ''}  {ser.most_common(5)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    args = sys.argv[2:]
    {"status": status, "boards": boards, "movers": movers, "handle": handle, "categories": categories, "full": full}[cmd](*args)
