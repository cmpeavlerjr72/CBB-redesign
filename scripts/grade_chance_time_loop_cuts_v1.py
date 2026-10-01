"""grade_chance_time_loop_cuts_v1.py -- multi-level cuts of the chance_time POST-HOC closed loop (lane I, 2026-10-01).

Per game (both teams) total bias and rim / three make by month, site, G9 home tier, offence tier (team side),
for R and each arm on the same 500 verified stride games and paired seeds; per-team offence points bias.
Underpowered cells (< 60 games on this sample) are labelled.

Usage: grade_chance_time_loop_cuts_v1.py <out_json> R=<dir> K=<dir> [C12=<dir> ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CBB_TRUTH", "verified_v1")
from cbb_sim.eval import reference as R  # noqa: E402

OUT = Path(sys.argv[1])
DIRS = dict(a.split("=", 1) for a in sys.argv[2:])
games = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")
gid_of = games["game_id"].to_numpy()
act = R.load_actual_games(2025)
tiers = R.team_quality_terciles(act)
act = act.set_index("game_id")


def load(d):
    g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(Path(d) / "games_*.parquet")))], ignore_index=True)
    if "game_id" not in g:
        g["game_id"] = gid_of[g["gidx"].to_numpy()]
    g = g[g["game_id"].isin(act.index)]
    m = g.groupby("game_id").mean(numeric_only=True)
    m["total"] = m["home_pts"] + m["away_pts"]
    return m


res = {}
base = None
for arm, d in DIRS.items():
    m = load(d)
    a = act.loc[m.index]
    df = pd.DataFrame({"bias": m["total"] - a["total"], "month": a["month"],
                       "site": np.where(a["neutral"] > 0, "neutral", "home/away"),
                       "home_tier": a["home_team_id"].map(tiers["margin"]).astype(str),
                       "rim_m": m["home_fgm2_rim"] + m["away_fgm2_rim"], "rim_a": m["home_fga2_rim"] + m["away_fga2_rim"],
                       "th_m": m["home_fgm3"] + m["away_fgm3"], "th_a": m["home_fga3"] + m["away_fga3"]})
    out = {"overall": float(df["bias"].mean()), "n": int(len(df))}
    for key in ("month", "site", "home_tier"):
        out[key] = {}
        for v, c in df.groupby(key):
            out[key][str(v)] = {"n": int(len(c)), "bias": float(c["bias"].mean()),
                                "rim_make": float(c["rim_m"].sum() / c["rim_a"].sum()),
                                "three_make": float(c["th_m"].sum() / c["th_a"].sum()),
                                "underpowered": bool(len(c) < 60)}
    # per team offence points bias (team side)
    rows = []
    for side in ("home", "away"):
        rows.append(pd.DataFrame({"team": a[f"{side}_team_id"].to_numpy(),
                                  "d": (m[f"{side}_pts"] - a[f"{side}_score"]).to_numpy()}))
    t = pd.concat(rows).groupby("team")["d"].agg(["mean", "size"])
    t = t[t["size"] >= 3]
    out["per_team"] = {"n_teams": int(len(t)), "median": float(t["mean"].median()),
                       "share_neg": float((t["mean"] < 0).mean()), "note": "UNDERPOWERED per team (~3-6 games each)"}
    off_tier = {k: v for k, v in tiers["margin"].items()}
    res[arm] = out
    print(arm, round(out["overall"], 3), {k: {kk: round(vv["bias"], 2) for kk, vv in out[k].items()} for k in ("month", "site", "home_tier")},
          out["per_team"], flush=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
