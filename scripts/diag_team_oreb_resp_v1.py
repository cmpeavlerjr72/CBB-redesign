"""diag_team_oreb_resp_v1.py -- team OREB% responsiveness at FULL SIZE (all 5,710 games x 200 seeds), lane I 2026-10-01.

DIAGNOSTIC ONLY, no new simulation. The shot-block grader's team line (`grade_shot_block_closed_loop_v1.team_oreb`,
imported) is evaluated on the full-size box runs instead of a 500-game sample, for the offence AND the defence side:
teams in quintiles of their 2024 OREB% (offence) or 2024 OREB% allowed (defence), slope = sim span / actual span.
Floors: the four full-size seed-offset draws of S0 (v3full_S0f1..f4) against S0, max |draw - S0| (Decision 12).

Usage: diag_team_oreb_resp_v1.py <out_json> <run> [<run> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/engine_v0"
OUT, RUNS = Path(sys.argv[1]), sys.argv[2:]
FLOORS = ["v3full_S0f1_s200_o1000", "v3full_S0f2_s200_o2000", "v3full_S0f3_s200_o3000", "v3full_S0f4_s200_o4000"]
REF = "v3full_S0_s200_o0"
gmap = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")[["game_id", "home_team_id", "away_team_id"]]
box = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2025.parquet")
b24 = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2024.parquet")


def opp_join(b):
    o = b[["game_id", "team_id", "offensive_rebounds", "defensive_rebounds"]]
    o = o.merge(o, on="game_id", suffixes=("", "_o"))
    return o[o["team_id"] != o["team_id_o"]]


o24 = opp_join(b24).groupby("team_id").sum(numeric_only=True)
prior_off = o24["offensive_rebounds"] / (o24["offensive_rebounds"] + o24["defensive_rebounds_o"])
prior_def = o24["offensive_rebounds_o"] / (o24["offensive_rebounds_o"] + o24["defensive_rebounds"])
games_all = gmap["game_id"].unique()
act = opp_join(box[box["game_id"].isin(games_all)]).groupby("team_id").sum(numeric_only=True)
act_off = act["offensive_rebounds"] / (act["offensive_rebounds"] + act["defensive_rebounds_o"])
act_def = act["offensive_rebounds_o"] / (act["offensive_rebounds_o"] + act["defensive_rebounds"])


def lines(tag):
    g = pd.read_parquet(R / tag / "games.parquet", columns=["game_id", "home_oreb", "away_oreb", "home_dreb", "away_dreb"])
    g = g.merge(gmap, on="game_id")
    g = g[g["game_id"].isin(set(box["game_id"]))]
    rows = []
    for s, o in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({"team": g[f"{s}_team_id"], "oreb": g[f"{s}_oreb"], "opp_dreb": g[f"{o}_dreb"],
                                  "dreb": g[f"{s}_dreb"], "opp_oreb": g[f"{o}_oreb"]}))
    t = pd.concat(rows).groupby("team").sum()
    sim_off = t["oreb"] / (t["oreb"] + t["opp_dreb"])
    sim_def = t["opp_oreb"] / (t["opp_oreb"] + t["dreb"])
    out = {"pooled_oreb": float(t["oreb"].sum() / (t["oreb"].sum() + t["opp_dreb"].sum()))}
    for side, sim, a, pr in (("off", sim_off, act_off, prior_off), ("def", sim_def, act_def, prior_def)):
        d = pd.DataFrame({"sim": sim, "act": a}).dropna().join(pr.rename("prior"), how="inner")
        d["q"] = pd.qcut(d["prior"].rank(method="first"), 5, labels=False)
        qq = d.groupby("q")[["sim", "act"]].mean()
        out[f"{side}_slope"] = float((qq["sim"].iloc[-1] - qq["sim"].iloc[0]) / (qq["act"].iloc[-1] - qq["act"].iloc[0]))
        out[f"{side}_sim_by_q"] = [round(float(x), 4) for x in qq["sim"]]
        out[f"{side}_act_by_q"] = [round(float(x), 4) for x in qq["act"]]
        out[f"{side}_gap_by_q_pp"] = [round(100 * float(x - y), 2) for x, y in zip(qq["sim"], qq["act"])]
        out[f"{side}_team_mae_pp"] = float(100 * (d["sim"] - d["act"]).abs().mean())
        out[f"{side}_team_corr"] = float(np.corrcoef(d["sim"], d["act"])[0, 1])
        b = np.polyfit(d["sim"] - d["sim"].mean(), d["act"], 1)[0]
        out[f"{side}_beta_act_on_sim"] = float(b)
        out[f"{side}_n_teams"] = int(len(d))
    return out


res = {"ref": REF, "floor_draws": FLOORS, "runs": {}}
L0 = lines(REF)
FL = {t: lines(t) for t in FLOORS}
keys = [k for k, v in L0.items() if isinstance(v, float)]
res["floors"] = {k: max(abs(FL[t][k] - L0[k]) for t in FLOORS) for k in keys}
res["runs"][REF] = L0
for t in RUNS:
    res["runs"][t] = lines(t)
for t, v in res["runs"].items():
    print(f"{t:30s} pooled {v['pooled_oreb']:.4f} off slope {v['off_slope']:.3f} def slope {v['def_slope']:.3f} "
          f"off gap/q {v['off_gap_by_q_pp']} def gap/q {v['def_gap_by_q_pp']} offMAE {v['off_team_mae_pp']:.2f} "
          f"off beta {v['off_beta_act_on_sim']:.3f}", flush=True)
print("floors", {k: round(v, 4) for k, v in res["floors"].items()})
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
