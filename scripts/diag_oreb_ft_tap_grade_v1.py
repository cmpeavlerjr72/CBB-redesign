"""diag_oreb_ft_tap_grade_v1.py -- grade the served-v2 tap (diag_oreb_ft_tap_v1.py), lane I 2026-10-01.

1. Bit-identity: the tap's games rows vs the box adopted run (v3full_COMB9GCTKD_s200_o0) on the shared (game, seed).
2. K2_Ocell team responsiveness (rebound experiments.md 12.4, descriptive): each team's expected block rate on missed
   FGAs (sum of P(blocked | miss) / missed FGA), defence (blocking) and offence (being blocked), by 2024-prior quintile,
   vs the real 2025 box rate (blocks / opponent missed FGA). Slope = sim span / actual span.

Usage: diag_oreb_ft_tap_grade_v1.py <tap_dir> <out_json>
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
games = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet").reset_index(drop=True)
res = {}

tg = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(TAP / "games_*.parquet")))], ignore_index=True)
box = pd.read_parquet(ROOT / "results/engine_v0/v3full_COMB9GCTKD_s200_o0/games.parquet")
box = box[box["seed"].isin(tg["seed"].unique())]
cols = [c for c in box.columns if c in tg.columns and c not in ("game_id", "seed")]
m = tg.merge(box, on=["game_id", "seed"], suffixes=("_t", "_b"))
mism = {c: int((m[f"{c}_t"] != m[f"{c}_b"]).sum()) for c in cols}
res["bit_identity"] = {"rows_compared": int(len(m)), "columns": len(cols), "mismatching_cells": int(sum(mism.values())),
                       "by_column": {k: v for k, v in mism.items() if v}}


def rates(b):
    b = b.copy()
    b["miss"] = b["field_goals_attempted"] - b["field_goals_made"]
    o = b[["game_id", "team_id", "blocks", "miss"]].merge(b[["game_id", "team_id", "blocks", "miss"]], on="game_id",
                                                           suffixes=("", "_o"))
    o = o[o["team_id"] != o["team_id_o"]].groupby("team_id").sum(numeric_only=True)
    return o["blocks"] / o["miss_o"], o["blocks_o"] / o["miss"]          # def blocking rate, off blocked rate


tb25 = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2025.parquet")
tb25 = tb25[tb25["game_id"].isin(set(games["game_id"]))]
act_def, act_off = rates(tb25)
pri_def, pri_off = rates(pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2024.parquet"))
bk = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(TAP / "blk_*.parquet")))], ignore_index=True)
g = bk["gidx"].to_numpy()
offt = np.where(bk["off"] == 0, games["home_team_id"].to_numpy()[g], games["away_team_id"].to_numpy()[g])
deft = np.where(bk["off"] == 0, games["away_team_id"].to_numpy()[g], games["home_team_id"].to_numpy()[g])
bk["off_team"], bk["def_team"] = offt, deft
sim_def = bk.groupby("def_team")[["p", "n"]].sum()
sim_off = bk.groupby("off_team")[["p", "n"]].sum()
res["pooled"] = {"sim_block_share_of_missed_fga": float(bk["p"].sum() / bk["n"].sum()),
                 "actual_blocks_over_missed_fga": float(tb25["blocks"].sum() / (tb25["field_goals_attempted"]
                                                                                - tb25["field_goals_made"]).sum())}
for side, sim, act, pri in (("def", sim_def["p"] / sim_def["n"], act_def, pri_def),
                            ("off", sim_off["p"] / sim_off["n"], act_off, pri_off)):
    d = pd.DataFrame({"sim": sim, "act": act}).dropna().join(pri.rename("prior"), how="inner")
    d["q"] = pd.qcut(d["prior"].rank(method="first"), 5, labels=False)
    qq = d.groupby("q")[["sim", "act"]].mean()
    res[f"{side}_block"] = {"n_teams": int(len(d)), "sim_by_q": [round(float(x), 4) for x in qq["sim"]],
                            "act_by_q": [round(float(x), 4) for x in qq["act"]],
                            "slope": float((qq["sim"].iloc[-1] - qq["sim"].iloc[0]) / (qq["act"].iloc[-1] - qq["act"].iloc[0])),
                            "team_corr": float(np.corrcoef(d["sim"], d["act"])[0, 1])}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
