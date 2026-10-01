"""diag_laneI_tap_compare_v1.py -- local 500 x 25 taps, arm minus reference (lane I, 2026-10-01). DIRECTION ONLY:
Decision 12 says no line is decided at this size; every number here is labelled underpowered.

Gate lines through `grade_shot_block_closed_loop_v1.gate_lines` (the gate report's own numbers), pooled FT% and OREB% from
games.parquet, and the reference's bit-identity with the box adopted run on the shared (game, seed).

Usage: diag_laneI_tap_compare_v1.py <out_json> <ref_tag> <arm_tag> [<arm_tag> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import grade_shot_block_closed_loop_v1 as SB  # noqa: E402

R = ROOT / "results/engine_v0"
OUT, REF, ARMS = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]


def pooled(tag):
    g = pd.read_parquet(R / tag / "games.parquet")
    fta = g["home_fta"].sum() + g["away_fta"].sum()
    ftm = g["home_ftm"].sum() + g["away_ftm"].sum()
    o = g["home_oreb"].sum() + g["away_oreb"].sum()
    d = g["home_dreb"].sum() + g["away_dreb"].sum()
    return {"ft_pct": float(ftm / fta), "oreb_share": float(o / (o + d)), "pts_per_game": float((g["home_pts"] + g["away_pts"]).mean()),
            "n_rows": int(len(g))}


res = {"underpowered": "500 x 25 local tap: direction and parity only (Decision 12)", "ref": REF, "arms": {}}
g0 = pd.read_parquet(R / REF / "games.parquet")
box = pd.read_parquet(R / "v3full_COMB9GCTKD_s200_o0/games.parquet")
m = g0.merge(box, on=["game_id", "seed"], suffixes=("_a", "_b"))
cols = [c for c in g0.columns if c not in ("game_id", "seed") and c in box.columns]
res["ref_vs_box_adopted"] = {"rows": int(len(m)), "mismatching_cells": int(sum((m[f"{c}_a"] != m[f"{c}_b"]).sum() for c in cols))}
L0, P0 = SB.gate_lines(REF), pooled(REF)
res["ref_lines"] = {**{k: v["value"] for k, v in L0.items()}, **P0}
for t in ARMS:
    L, P = SB.gate_lines(t), pooled(t)
    res["arms"][t] = {"lines": {k: v["value"] for k, v in L.items()}, **P,
                      "delta": {**{k: L[k]["value"] - L0[k]["value"] for k in L if k in L0},
                                **{k: P[k] - P0[k] for k in ("ft_pct", "oreb_share", "pts_per_game")}},
                      "status_flips": {k: f"{L0[k]['status']}->{L[k]['status']}" for k in L if k in L0 and L[k]["status"] != L0[k]["status"]}}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
