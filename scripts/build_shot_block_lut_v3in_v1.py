#!/usr/bin/env python
"""
build_shot_block_lut_v3in_v1.py -- versioned SIBLING of the shot-block K2_Ocell backtest table, built against the v3 engine inputs
(lane F, 2026-10-01). Nothing is overwritten and the engine does not read the sibling (no default changed).

Why: `data/processed/models/engine/shot_block_K2_Ocell_F2_2025.npz` was built against the v2 inputs' roster slots
(`build_engine_shot_block_lut_v1.main`: EngineInputs.load(engine, "F2_2025")). The v3 inputs (served by the adopted full-size read and
equal, array for array, to the live builder's inputs) name players in 334 games where v2 used the anonymous fallback (those games'
team-games have no row in the on-floor table, so the v2 builder had no stub row for them; the live / v3 builder adds one). The engine
loads the v2-built table for v3 inputs (the file is keyed by tag `F2_2025`), so in those 334 games the shot-block shooter features read
"unknown shooter" while every other slot-keyed input reads the named slots.

This builds the table from the v3 slots with the live builder's own function (`build_shot_block_lut_live_v1.build_table`, full-season
events, per-date as-of), writes `engine_v3/shot_block_K2_Ocell_v3in_F2_2025.npz`, and checks it against (a) the v2 table on the games whose
slots agree (must be bit-identical) and (b) the live builder on dates with affected games.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import build_shot_block_lut_live_v1 as SBL
from cbb_sim.engine.inputs import EngineInputs

V3 = REPO / "data/processed/models/engine_v3"
V2 = REPO / "data/processed/models/engine"
inp3 = EngineInputs.load(V3, "F2_2025")
tab = SBL.build_table(inp3, "K2_Ocell", as_of=None)
out = V2 / "shot_block_K2_Ocell_v3in_F2_2025.npz"      # tracked engine dir, next to the v2-built table (engine flag K2_Ocell_v3in)
np.savez_compressed(out, roster_cbbd=inp3.roster_cbbd, **tab)
old = np.load(V2 / "shot_block_K2_Ocell_F2_2025.npz")
assert np.array_equal(old["game_id"], tab["game_id"])
v2r = np.load(V2 / "arrays_F2_2025.npz")["roster_cbbd"]
aff = (v2r != inp3.roster_cbbd).any(axis=(1, 2))
rep = {"n_games": int(len(aff)), "n_affected_games": int(aff.sum())}
for k in ("team", "anchor", "shooter", "known", "coef", "mu", "sd"):
    a, b = np.asarray(old[k]), np.asarray(tab[k])
    if k in ("coef", "mu", "sd"):
        rep[f"{k}_equal_v2_table"] = bool(np.array_equal(a, b)); continue
    rep[f"{k}_unaffected_games_bit_identical"] = bool(np.array_equal(a[~aff], b[~aff], equal_nan=True))
    rep[f"{k}_affected_games_max_abs_diff"] = float(np.abs(a[aff].astype(float) - b[aff].astype(float)).max())
# size of the difference on the affected games, through the fitted model
feats = [str(f) for f in tab["features"]]
c = tab["coef"][1:] / tab["sd"]
jk, js = feats.index("shooter_known"), feats.index("shooter_blocked_c")
named = (inp3.roster_cbbd[aff] >= 0)
d_eta = c[jk] * (tab["known"][aff].astype(float) - old["known"][aff]) + c[js] * (tab["shooter"][aff].astype(float) - old["shooter"][aff])
share = inp3.rot_share[aff].astype(float)
w = share / share.sum(axis=2, keepdims=True)                       # who takes the shot ~ rotation share (illustrative weight)
rep["affected_slot_sides_changed"] = int((d_eta != 0).sum()); rep["affected_slot_sides_total"] = int(d_eta.size)
rep["d_eta_mean_abs_over_changed"] = float(np.abs(d_eta[d_eta != 0]).mean()) if (d_eta != 0).any() else 0.0
rep["d_eta_min_max"] = [float(d_eta.min()), float(d_eta.max())]
rep["share_weighted_d_eta_per_team_game_mean"] = float((w * d_eta).sum(axis=2).mean())
rep["share_weighted_d_eta_per_team_game_mean_abs"] = float(np.abs((w * d_eta).sum(axis=2)).mean())
rep["table"] = str(out.relative_to(REPO))
(REPO / "results/shot_block_v3in_sibling.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps(rep, indent=1))
