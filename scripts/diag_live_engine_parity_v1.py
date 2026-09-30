"""Engine-output parity: engine on LIVE inputs vs engine on the backtest builder's arrays for the SAME games/seeds.

    .venv/Scripts/python.exe scripts/diag_live_engine_parity_v1.py --date 2024-11-12 --seeds 3

Requires the three runs written by run_engine_live.py (live, --slice-from-backtest, and
--slice-from-backtest --reverse). Reports (1) RNG alignment: backtest arrays run in reversed game order
must be bit-identical to forward order; (2) games whose live inputs equal the backtest inputs in EVERY
array must give bit-identical output; (3) for the rest, the size of the output shift next to seed noise.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
tag = f"LIVE_F2_2025_{a.date}"; R = ROOT / "results/engine_live"
live = pd.read_parquet(R / f"{tag}_s{a.seeds}/games.parquet")
bt = pd.read_parquet(R / f"{tag}__bt_arrays_s{a.seeds}/games.parquet")
rev = pd.read_parquet(R / f"{tag}__bt_arrays__rev_s{a.seeds}/games.parquet")
key = ["game_id", "seed"]
num = [c for c in live.columns if c not in key and live[c].dtype.kind in "fiu" and c in bt.columns]
def align(x): return x.sort_values(key).reset_index(drop=True)
live, bt, rev = align(live), align(bt), align(rev)
rep = {"date": a.date, "seeds": a.seeds, "n_games": int(live.game_id.nunique()), "n_rows": int(len(live))}
rep["rng_alignment_reverse_order_bit_identical"] = bool(all(np.array_equal(bt[c].to_numpy(), rev[c].to_numpy()) for c in num))
# identical-input games
ed = ROOT / "data/processed/models/engine"; ld = ROOT / "data/processed/models/engine_live"
zb = dict(np.load(ed / "arrays_F2_2025_v2.npz")); gb = pd.read_parquet(ed / "games_F2_2025_v2.parquet")
zl = dict(np.load(ld / f"arrays_{tag}.npz")); gl = pd.read_parquet(ld / f"games_{tag}.parquet")
evb = np.load(ed / "event_round2_s1_F2_2025/team_block.npz")["team_block"]; evl = np.load(ld / f"event_block_{tag}.npz")["team_block"]
pos = {int(g): i for i, g in enumerate(gb.game_id)}; bi = np.array([pos[int(g)] for g in gl.game_id])
ident = np.ones(len(gl), bool); diffarr = {}
for k in zl:
    d = (zl[k] != zb[k][bi]).reshape(len(gl), -1).any(axis=1); ident &= ~d; diffarr[k] = int(d.sum())
d = (evl != evb[bi]).reshape(len(gl), -1).any(axis=1); ident &= ~d; diffarr["event_block_r2"] = int(d.sum())
rep["games_with_identical_inputs"] = int(ident.sum()); rep["games_with_any_input_difference_by_array"] = diffarr
ids_ident = set(gl.game_id[ident].astype(int))
m = live.merge(bt, on=key, suffixes=("_live", "_bt"))
mi = m[m.game_id.isin(ids_ident)]
rep["identical_input_games_bit_identical_output"] = bool(all(np.array_equal(mi[f"{c}_live"], mi[f"{c}_bt"]) for c in num)) if len(mi) else None
rep["identical_input_rows"] = int(len(mi))
md = m[~m.game_id.isin(ids_ident)]
def sh(col):
    dd = (md[f"{col}_live"] - md[f"{col}_bt"]).to_numpy(float)
    return {"mean_abs_diff": float(np.abs(dd).mean()), "mean_diff_live_minus_bt": float(dd.mean()),
            "sd_of_col_within_game_across_seeds": float(md.groupby("game_id")[f"{col}_bt"].std().mean())}
rep["differing_input_games"] = int(md.game_id.nunique())
rep["output_shift"] = {c: sh(c) for c in ("home_pts", "away_pts", "possessions") if f"{c}_live" in md.columns}
hy = align(pd.read_parquet(R / f"{tag}__hybrid_s{a.seeds}/games.parquet"))
core = np.ones(len(gl), bool)
for k in ("team_static", "roster_cbbd", "roster_espn", "roster_valid", "rot_share", "rot_srank", "rot_start", "rot_fpm", "rot_pavail"):
    core &= ~(zl[k] != zb[k][bi]).reshape(len(gl), -1).any(axis=1)
core &= ~(evl != evb[bi]).reshape(len(gl), -1).any(axis=1)
core_ids = set(gl.game_id[core].astype(int))
mh = hy.merge(bt, on=key, suffixes=("_hy", "_bt")); mh = mh[mh.game_id.isin(core_ids)]
rep["hybrid_(live team/roster/rotation/event + backtest slot arrays)"] = {
    "games_with_identical_team_roster_rotation_event_inputs": len(core_ids),
    "bit_identical_output_on_those_games": bool(all(np.array_equal(mh[f"{c}_hy"], mh[f"{c}_bt"]) for c in num)),
    "rows": int(len(mh))}
ml = live.merge(hy, on=key, suffixes=("_live", "_hy")); ml = ml[ml.game_id.isin(core_ids)]
rep["slot_array_effect_on_output_(live vs hybrid, same games)"] = {c: {
    "mean_abs_diff": float(np.abs(ml[f"{c}_live"] - ml[f"{c}_hy"]).mean())} for c in ("home_pts", "away_pts", "possessions")}
print(json.dumps(rep, indent=1))
(R / f"parity_{tag}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
