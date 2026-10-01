"""diag_laneI_rbto_levels_v1.py -- multi-level OREB% evidence for the RBTO full-size read (lane I, 2026-10-01).

OREB share (OREB / (OREB + opponent DREB)) by month and by site, reference (served v2) vs arm vs the 2025 box, plus the
per-game paired distribution (arm - reference mean OREB share per game) and per-game MAE vs the box. Cells under 300
games are labelled UNDERPOWERED. No new simulation.

Usage: diag_laneI_rbto_levels_v1.py <out_json> <ref_tag> <arm_tag>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/engine_v0"
OUT, REF, ARM = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
gm = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")[["game_id", "game_date", "neutral"]]
box = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2025.parquet",
                      columns=["game_id", "team_id", "offensive_rebounds", "defensive_rebounds"])
b = box.groupby("game_id").agg(oreb=("offensive_rebounds", "sum"), dreb=("defensive_rebounds", "sum"))
act = (b["oreb"] / (b["oreb"] + b["dreb"])).rename("act")


def per_game(tag):
    g = pd.read_parquet(R / tag / "games.parquet", columns=["game_id", "home_oreb", "away_oreb", "home_dreb", "away_dreb"])
    s = g.groupby("game_id").sum()
    return ((s["home_oreb"] + s["away_oreb"]) / (s["home_oreb"] + s["away_oreb"] + s["home_dreb"] + s["away_dreb"])).rename(tag), s


r0, s0 = per_game(REF)
r1, s1 = per_game(ARM)
d = pd.concat([r0, r1, act], axis=1, join="inner").join(gm.set_index("game_id"))
d["month"] = pd.to_datetime(d["game_date"]).dt.month
d["site"] = np.where(d["neutral"].astype(bool), "neutral", "home/away")
res = {"n_games": int(len(d))}
for key in ("month", "site"):
    rows = {}
    for k, x in d.groupby(key):
        rows[str(k)] = {"n": int(len(x)), "underpowered": bool(len(x) < 300), "ref": float(x[REF].mean()),
                        "arm": float(x[ARM].mean()), "actual": float(x["act"].mean()),
                        "gap_ref_pp": 100 * float(x[REF].mean() - x["act"].mean()),
                        "gap_arm_pp": 100 * float(x[ARM].mean() - x["act"].mean())}
    res[f"by_{key}"] = rows
diff = d[ARM] - d[REF]
res["per_game"] = {"arm_minus_ref_mean_pp": 100 * float(diff.mean()), "share_games_up": float((diff > 0).mean()),
                   "mae_vs_actual_ref_pp": 100 * float((d[REF] - d["act"]).abs().mean()),
                   "mae_vs_actual_arm_pp": 100 * float((d[ARM] - d["act"]).abs().mean())}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
