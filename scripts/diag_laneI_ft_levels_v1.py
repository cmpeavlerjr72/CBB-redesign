"""diag_laneI_ft_levels_v1.py -- multi-level FT% evidence for the N1 / A1 full-size reads (lane I, 2026-10-01).

Pooled FTM/FTA by month and by site, reference vs arms vs the 2025 box (graded games only), and by team prior quintile
(2024 team FT%): sim span / actual span. Cells under 300 games are labelled UNDERPOWERED. No new simulation.

Usage: diag_laneI_ft_levels_v1.py <out_json> <ref_tag> <arm_tag> [<arm_tag> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/engine_v0"
OUT, TAGS = Path(sys.argv[1]), sys.argv[2:]
gm = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")[
    ["game_id", "game_date", "neutral", "home_team_id", "away_team_id"]]
box = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2025.parquet",
                      columns=["game_id", "team_id", "free_throws_made", "free_throws_attempted"])
b24 = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2024.parquet",
                      columns=["team_id", "free_throws_made", "free_throws_attempted"]).groupby("team_id").sum()
prior = b24["free_throws_made"] / b24["free_throws_attempted"]
graded = set(box["game_id"])
meta = gm[gm["game_id"].isin(graded)].set_index("game_id")
meta["month"] = pd.to_datetime(meta["game_date"]).dt.month
meta["site"] = np.where(meta["neutral"].astype(bool), "neutral", "home/away")
ab = box.groupby("game_id")[["free_throws_made", "free_throws_attempted"]].sum()
res = {"tags": TAGS, "by_month": {}, "by_site": {}, "team_quintile": {}}


def team_rows(tag):
    g = pd.read_parquet(R / tag / "games.parquet", columns=["game_id", "home_ftm", "home_fta", "away_ftm", "away_fta"])
    g = g[g["game_id"].isin(graded)].merge(gm[["game_id", "home_team_id", "away_team_id"]], on="game_id")
    t = pd.concat([pd.DataFrame({"game_id": g["game_id"], "team": g[f"{s}_team_id"], "ftm": g[f"{s}_ftm"], "fta": g[f"{s}_fta"]})
                   for s in ("home", "away")])
    return t


T = {t: team_rows(t) for t in TAGS}
for key in ("month", "site"):
    for k, ids in meta.groupby(key).groups.items():
        ids = set(ids)
        row = {"n": len(ids), "underpowered": len(ids) < 300}
        a = ab.loc[ab.index.isin(ids)].sum()
        row["actual"] = float(a["free_throws_made"] / a["free_throws_attempted"])
        for t in TAGS:
            x = T[t][T[t]["game_id"].isin(ids)]
            row[t] = float(x["ftm"].sum() / x["fta"].sum())
        res[f"by_{key}"][str(k)] = row
abt = box.groupby("team_id")[["free_throws_made", "free_throws_attempted"]].sum()
act_t = abt["free_throws_made"] / abt["free_throws_attempted"]
for t in TAGS:
    s = T[t].groupby("team")[["ftm", "fta"]].sum()
    d = pd.DataFrame({"sim": s["ftm"] / s["fta"], "act": act_t}).dropna().join(prior.rename("prior"), how="inner")
    d["q"] = pd.qcut(d["prior"].rank(method="first"), 5, labels=False)
    qq = d.groupby("q")[["sim", "act"]].mean()
    res["team_quintile"][t] = {"sim_by_q": [round(float(v), 4) for v in qq["sim"]], "act_by_q": [round(float(v), 4) for v in qq["act"]],
                               "slope": float((qq["sim"].iloc[-1] - qq["sim"].iloc[0]) / (qq["act"].iloc[-1] - qq["act"].iloc[0]))}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
