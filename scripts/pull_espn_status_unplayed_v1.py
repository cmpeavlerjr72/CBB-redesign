#!/usr/bin/env python
"""Lane H: third-source status for the unverified-final games in the graded rule (2022-2025).
Source: ESPN public core API (sports.core.api.espn.com, the same ESPN feed family hoopR wraps), one
request per game for competition status, plus competitor scores. No banned site. Output:
data/processed/truth/truth_third_source_espn_v1.csv"""
import json, sys, time, urllib.request
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
a = pd.read_parquet(ROOT / "data/processed/truth/truth_audit_unplayed_v1.parquet"); a = a[~a.sealed_season]
g = a[a.cls.str.contains("nonfinal") & a.is_d1_game.fillna(False) & ~a.pbp_truncated.fillna(False)]
BASE = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/{e}/competitions/{e}"
def get(u):
    for i in range(3):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=25))
        except Exception as e:
            err = str(e)[:80]; time.sleep(1.5)
    return {"_err": err}
rows = []
for _, r in g.iterrows():
    e = int(r.game_id); st = get(BASE.format(e=e) + "/status"); t = st.get("type", {})
    comp = get(BASE.format(e=e)); sc = {}
    for c in comp.get("competitors", []):
        ref = c.get("score", {}).get("$ref")
        v = get(ref) if ref else {}
        sc[c.get("homeAway")] = v.get("value")
    rows.append({"season": r.season, "game_id": e, "h_status": r.h_status, "c_status": r.c_status, "espn_status": t.get("name"), "espn_completed": t.get("completed"),
                 "espn_period": st.get("period"), "espn_clock": st.get("clock"), "espn_home": sc.get("home"), "espn_away": sc.get("away"), "err": st.get("_err") or comp.get("_err")})
    time.sleep(0.25)
d = pd.DataFrame(rows); d.to_csv(ROOT / "data/processed/truth/truth_third_source_espn_v1.csv", index=False)
print(d.groupby(["season", "espn_status", "espn_completed"], dropna=False).size().to_string())
print("errors", d.err.notna().sum()); print(d[d.espn_completed.fillna(False) | (d.espn_period.fillna(0) > 0) | d.err.notna()].to_string())
