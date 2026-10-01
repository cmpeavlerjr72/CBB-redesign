"""diag_team_ft_slope_owner_v1.py -- who owns the team FT-rate slope drop 0.958 (v2) -> 0.617 (v3)? (lane I, 2026-10-01)

DIAGNOSTIC ONLY. The line is `grade_foul_joint_closed_loop_v2.team_stats` (imported, not re-derived): teams binned
into quintiles of their 2024 FTA/FGA (>= 10 games), slope = (sim top - sim bottom) / (actual top - actual bottom) of the
team FTA/FGA over the graded games. It is evaluated on a 2 x 3 grid, with no new simulation:

    inputs  : v2 (F2_2025_s200_v5b_A_full, 09-18 served stack, v2 inputs)  vs  v3 (v3full_S0_s200_o0, same stack on v3)
    sample  : legacy stride (fj_R_s25's 500 games) / verified stride (e3_fj_R_s25's 500 games) / all games
plus the served v2 stack (v3full_COMB9GCTKD_s200_o0) and the two 500 x 25 reference runs as recorded. A paired game
bootstrap (200) gives the sampling SD of the slope on a 500-game sample. Quintile tables (sim and actual) per cell.

Usage: diag_team_ft_slope_owner_v1.py <out_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import grade_foul_joint_closed_loop_v2 as G2  # noqa: E402

OUT = Path(sys.argv[1])
R = ROOT / "results/engine_v0"
box = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2025.parquet")
box_g = box.rename(columns={"free_throws_attempted": "fta", "field_goals_attempted": "fga"})[["game_id", "team_id", "fta", "fga"]]
b24 = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2024.parquet")
p = b24.groupby("team_id").agg(n=("game_id", "size"), fta=("free_throws_attempted", "sum"), fga=("field_goals_attempted", "sum"))
prior = (p["fta"] / p["fga"])[p["n"] >= 10]
s25 = box.groupby("team_id")[["free_throws_attempted", "field_goals_attempted"]].sum()
season_rate = s25["free_throws_attempted"] / s25["field_goals_attempted"]
gmeta = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")[["game_id", "home_team_id", "away_team_id"]]


def with_teams(g):
    if "home_team_id" in g.columns:
        return g
    return g.merge(gmeta, on="game_id", how="left")


def quint(tp, games):
    w = pd.Series(games).value_counts()
    t = tp[tp["game_id"].isin(w.index)]
    sim = t.groupby("team_id")[["fta", "fga"]].sum()
    b = box_g[box_g["game_id"].isin(w.index)]
    act = b.groupby("team_id")[["fta", "fga"]].sum()
    df = pd.DataFrame({"sim": sim["fta"] / sim["fga"], "act": act["fta"] / act["fga"]}).dropna()
    q = df.join(prior.rename("prior"), how="inner")
    q["q"] = pd.qcut(q["prior"], 5, labels=False)
    qq = q.groupby("q")[["sim", "act"]].mean()
    return {"sim_by_q": [round(float(x), 4) for x in qq["sim"]], "act_by_q": [round(float(x), 4) for x in qq["act"]],
            "n_teams": int(len(q)), "team_games_per_team": float(len(t) / max(t["team_id"].nunique(), 1))}


runs = {"v2_S0_full": "F2_2025_s200_v5b_A_full", "v3_S0_full": "v3full_S0_s200_o0",
        "v3_servedv2_full": "v3full_COMB9GCTKD_s200_o0"}
TP = {k: G2.team_parts(with_teams(pd.read_parquet(R / v / "games.parquet"))) for k, v in runs.items()}
small = {"v2_ref_legacy_500x25": "fj_R_s25", "v3_ref_verified_500x25": "e3_fj_R_s25"}
for k, v in small.items():
    TP[k] = G2.team_parts(pd.read_parquet(R / v / "games_v2.parquet"))
samples = {"legacy": np.sort(pd.read_parquet(R / "fj_R_s25/games.parquet")["game_id"].unique()),
           "verified": np.sort(pd.read_parquet(R / "e3_fj_R_s25/games.parquet")["game_id"].unique()),
           "all": np.sort(gmeta["game_id"].unique())}
res = {"samples_n": {k: int(len(v)) for k, v in samples.items()},
       "legacy_verified_overlap": int(len(set(samples["legacy"]) & set(samples["verified"])))}
rng = np.random.default_rng(20261001)
cells = {}
for rk, tp in TP.items():
    for sk, gl in samples.items():
        if rk.startswith("v2_ref") and sk != "legacy":
            continue
        if rk.startswith("v3_ref") and sk != "verified":
            continue
        gl2 = gl[np.isin(gl, tp["game_id"].unique())]
        sl, sdr = G2.team_stats(tp, box_g, prior, season_rate, gl2)
        c = {"slope": sl, "team_sd_ratio": sdr, "n_games": int(len(gl2)), **quint(tp, gl2)}
        if sk != "all":
            bs = [G2.team_stats(tp, box_g, prior, season_rate, rng.choice(gl2, len(gl2), replace=True))[0] for _ in range(200)]
            c["boot_slope_sd"] = float(np.std(bs))
            c["boot_slope_mad_sd"] = float(1.4826 * np.median(np.abs(np.array(bs) - np.median(bs))))
            c["boot_slope_q05_q95"] = [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))]
        cells[f"{rk}|{sk}"] = c
        print(f"{rk:24s} {sk:9s} slope {sl:.3f} sdR {sdr:.3f} n={len(gl2)} "
              f"{c.get('boot_slope_sd', float('nan')):.3f} sim {c['sim_by_q']} act {c['act_by_q']}", flush=True)
res["cells"] = cells
# the sim's own span with the season actual (the slope numerator vs a fixed denominator): does the SIM span move?
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
