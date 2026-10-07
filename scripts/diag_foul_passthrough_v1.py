#!/usr/bin/env python
"""diag_foul_passthrough_v1.py -- why a +0.026 H1 in-bonus share move is worth only +0.15 pts (foul accrual round 2 prelude).
Paired d0-14 deltas (arm - ref, per game, both teams) from round-1 tap runs: possessions, in-bonus possessions, bonus trips,
FTA, FTM, FGA by type, FGM, TOV, points; points identity dPTS = dFTM + 2 dFGM2 + 3 dFGM3.
    .venv/Scripts/python.exe scripts/diag_foul_passthrough_v1.py --season 2025 --input-dir data/processed/models/engine_v3 \
        --ref results/engine_v0/fcal_F2_ctrl_s50 --arm results/engine_v0/fcal_F2_A2dbk_s50
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from cbb_sim.data.seal import assert_not_sealed
from cbb_sim.engine.inputs import EngineInputs

def game(p):
    g = pd.read_parquet(p / "games.parquet")
    o = pd.DataFrame({"game_id": g.game_id})
    for k in ("pts", "fta", "ftm", "fga3", "fga2_rim", "fga2_jump", "fgm3", "fgm2_rim", "fgm2_jump", "tov", "oreb"):
        o[k] = g[f"home_{k}"] + g[f"away_{k}"]
    o["poss"] = g["possessions"]
    a = pd.read_parquet(p / "half_agg.parquet")
    h = a.groupby(["game_id", "seed", "half"])[["poss", "in_bonus", "n_bonus_trip", "n_shoot_trip", "fta", "silent"]].sum().unstack("half")
    h.columns = [f"{c}_h{x}" for c, x in h.columns]
    h = h.reset_index().groupby("game_id").mean(numeric_only=True).drop(columns="seed")
    return o.groupby("game_id").mean().join(h)

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, required=True); ap.add_argument("--input-dir", required=True)
ap.add_argument("--ref", required=True); ap.add_argument("--arm", required=True)
a = ap.parse_args(); assert_not_sealed(a.season)
fold = "F2" if a.season == 2025 else "F1"
inp = EngineInputs.load(a.input_dir, f"{fold}_{a.season}")
dss = pd.Series(inp.team_static[:, 0, inp.team_names["days_since_start"]], index=inp.games["game_id"].to_numpy())
R, A = game(ROOT / a.ref), game(ROOT / a.arm)
ids = R.index.intersection(A.index)
for nm, lo, hi in (("d0-14", 0, 14), ("d46+", 46, 1e9)):
    ix = ids[(dss.reindex(ids) >= lo).to_numpy() & (dss.reindex(ids) <= hi).to_numpy()]
    d = (A.loc[ix] - R.loc[ix]).mean(); r = R.loc[ix].mean()
    print(f"\n== {fold} {nm} n={len(ix)} (per game, both teams; ref level | arm-ref)")
    for k in d.index:
        print(f"{k:16s} {r[k]:9.3f} | {d[k]:+.4f}")
    fgm2 = d.fgm2_rim + d.fgm2_jump
    print(f"identity: dFTM {d.ftm:+.3f} + 2dFGM2 {2*fgm2:+.3f} + 3dFGM3 {3*d.fgm3:+.3f} = {d.ftm+2*fgm2+3*d.fgm3:+.3f} vs dPTS {d.pts:+.3f}")
    print(f"FTA per extra bonus trip {(d.fta)/(d.n_bonus_trip_h1+d.n_bonus_trip_h2):.2f}; pts per extra FTA {d.pts/d.fta:.3f}; FT% ref {r.ftm/r.fta:.3f}")
