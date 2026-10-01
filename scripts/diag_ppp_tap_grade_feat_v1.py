"""diag_ppp_tap_grade_feat_v1.py -- per-feature swaps (sibling of diag_ppp_tap_grade_v1.py).

(v1 docstring follows.) diag_ppp_tap_grade_v1.py -- grade the lane-I fg_make tap: is the sim's make deficit the model or its inputs?

DIAGNOSTIC ONLY (lane I, 2026-09-30). Reads results/ppp_decomp/tap_<arm>/ (diag_ppp_tap_v1.py) and the
fg_make design table (real 2024-25 shots, the model's own held-out fold-2 rows). Nothing is fitted:
every probability is the SERVED round4_B1 S1 artifact evaluated on a feature row.

1. Bit-identity: the tap's games rows vs the box run's rows on every shared (game_id, seed).
2. Per shot class, on the SAME games (the 500-game verified stride sample), a closed chain from the
   sim's realised make rate to the real make rate:
     sim realised -> sim mean p                    (RNG)
     -> swap the SHOOTER feature                    (who shoots: sim shooter mix vs real shooters)
     -> swap the possession STATE features          (chance number / elapsed / transition / clock / bonus)
     -> swap the TEAM features (serve -> train)     (train/serve skew of as-of team inputs, site, season)
     -> cell reweighting                            (which game/offence takes the attempts)
     -> real mean p (offline)                       == the model's held-out prediction on real shots
     -> real realised                               (offline model error)
   A swap gives each sim row the feature values of a real attempt drawn uniformly from the SAME
   (game, offence, class) cell (fixed RNG), so team strength is held fixed by construction. The
   reverse order (state before shooter) is reported to bound the path dependence.
3. Feature means sim vs real; points per game per step.

Usage: diag_ppp_tap_grade_v1.py <tap_dir> <box_run_dir> <out_json>
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
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
from cbb_sim.engine.adapters import Adapters  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402

TAP, BOX, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PV = {"FGA_rim": 2.0, "FGA_jump2": 2.0, "FGA_3": 3.0}
SHOOTER = ["shooter_shrunk_dev_c"]
STATE = ["period", "seconds_remaining", "in_bonus", "chance_number", "chance_elapsed_s", "is_transition_f"]
BOXCOL = {"FGA_rim": ("fga2_rim", "fgm2_rim"), "FGA_jump2": ("fga2_jump", "fgm2_jump"), "FGA_3": ("fga3", "fgm3")}

inp = EngineInputs.load(str(ROOT / "data/processed/models/engine_v3"), "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
games = inp.games.reset_index(drop=True)
gid_of = games["game_id"].to_numpy()
pos = {int(g): k for k, g in enumerate(gid_of)}
TS = inp.team_static
TN = inp.team_names


def predict(cls, X: np.ndarray, gidx: np.ndarray) -> np.ndarray:
    out = np.empty(len(X))
    segs = ad.fg.manifests[cls].segments(gidx.astype(np.int64))
    for k in np.unique(segs):
        r = np.flatnonzero(segs == k)
        out[r] = ad.fg.models_by_seg[cls][k].predict_proba(np.ascontiguousarray(X[r], dtype=np.float32))[:, 1]
    return out


res: dict = {"tap": str(TAP), "box": str(BOX)}
# ---------------------------------------------------------------- 1. bit identity
tg = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(TAP / "games_*.parquet")))], ignore_index=True)
if "game_id" not in tg:
    tg["game_id"] = gid_of[tg["gidx"].to_numpy()]
bx = pd.read_parquet(BOX / "games.parquet")
bx = bx[bx["game_id"].isin(tg["game_id"]) & bx["seed"].isin(tg["seed"].unique())]
cols = [c for c in bx.columns if c in tg.columns and c not in ("game_id", "seed")]
m = tg.merge(bx, on=["game_id", "seed"], suffixes=("_t", "_b"))
mism = int(sum((m[f"{c}_t"].astype(float) != m[f"{c}_b"].astype(float)).sum() for c in cols))
res["bit_identity"] = {"rows_tap": int(len(tg)), "rows_matched": int(len(m)), "columns": len(cols),
                       "mismatching_cells": mism}
print("bit identity", res["bit_identity"], flush=True)
n_sims = len(tg)
n_seeds = tg["seed"].nunique()
sample_games = np.sort(tg["game_id"].unique())

# ---------------------------------------------------------------- real rows (design, 2025, sample games)
design = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet")
extra = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v4_extra_v2.parquet",
                        columns=["shooter_shrunk_dev_c"])
design["shooter_shrunk_dev_c"] = extra["shooter_shrunk_dev_c"].to_numpy()
del extra
design = design[design["season"] == 2025]
real_all = design
real = design[design["game_id"].isin(sample_games)].copy()
real["gidx"] = real["game_id"].map(pos).astype(np.int64)

rng = np.random.default_rng(20260930)
res["features"] = {}
GROUPS = {"chance_elapsed_s": ["chance_elapsed_s"], "is_transition_f": ["is_transition_f"],
          "elapsed+transition": ["chance_elapsed_s", "is_transition_f"],
          "clock(period,sec)": ["period", "seconds_remaining"], "in_bonus": ["in_bonus"],
          "chance_number": ["chance_number"], "shooter": SHOOTER, "all_state": STATE}
for cls in ("FGA_rim", "FGA_jump2", "FGA_3"):
    feats = json.loads((Path(ad.fg.source[cls]["path"]) / f"manifest_{cls}.json").read_text())["features"]
    s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(TAP / f"fg_{cls}_*.parquet")))],
                  ignore_index=True)
    gidx_s = s["gidx"].to_numpy().astype(np.int64)
    j = TN["off_rating_off_c"]
    v = s["off_rating_off_c"].to_numpy()
    side = np.where(np.abs(TS[gidx_s, 0, j] - v) <= np.abs(TS[gidx_s, 1, j] - v), 0, 1)
    s["off_team_id"] = np.where(side == 0, games["home_team_id"].to_numpy()[gidx_s],
                                games["away_team_id"].to_numpy()[gidx_s])
    s["game_id"] = gid_of[gidx_s]
    Xs = s[feats].to_numpy(np.float64)
    rc = real[real["shot_class"] == cls].reset_index(drop=True)
    Xr = np.asarray(FG.design_matrix(rc, feats), dtype=np.float64)
    key_r = rc["game_id"].astype(np.int64).to_numpy() * 100000 + rc["off_team_id"].astype(np.int64).to_numpy()
    key_s = s["game_id"].astype(np.int64).to_numpy() * 100000 + s["off_team_id"].astype(np.int64).to_numpy()
    fi = {f: feats.index(f) for f in feats}
    # match WITHIN (game, offence, chance-number bucket 1 / 2+) so chance-number mix is held fixed
    cb_s = (Xs[:, fi["chance_number"]] >= 2).astype(np.int64)
    cb_r = (Xr[:, fi["chance_number"]] >= 2).astype(np.int64)
    key_r2, key_s2 = key_r * 2 + cb_r, key_s * 2 + cb_s
    order = np.argsort(key_r2, kind="stable")
    uk, start, cnt = np.unique(key_r2[order], return_index=True, return_counts=True)
    loc = np.clip(np.searchsorted(uk, key_s2), 0, len(uk) - 1)
    hs = uk[loc] == key_s2
    draw = np.full(len(s), -1, dtype=np.int64)
    u = rng.random(len(s))
    draw[hs] = order[start[loc[hs]] + (u[hs] * cnt[loc[hs]]).astype(np.int64)]
    base = predict(cls, Xs[hs], gidx_s[hs])
    a_col, m_col = BOXCOL[cls]
    fga_pg = float((tg[f"home_{a_col}"].sum() + tg[f"away_{a_col}"].sum()) / n_sims)
    out = {"matched_share": float(hs.mean()), "base_p": float(base.mean()), "fga_pg": fga_pg, "swaps": {}}
    cbm = cb_s[hs]
    for gname, grp in GROUPS.items():
        X = Xs[hs].copy()
        for f in grp:
            X[:, fi[f]] = Xr[draw[hs], fi[f]]
        p = predict(cls, X, gidx_s[hs])
        d = p - base
        out["swaps"][gname] = {"d_pp": 100 * float(d.mean()),
                               "pts_per_game_sim_minus_actual": -float(d.mean()) * fga_pg * PV[cls],
                               "d_pp_chance1": 100 * float(d[cbm == 0].mean()),
                               "d_pp_chance2plus": 100 * float(d[cbm == 1].mean()),
                               "share_chance2plus": float(cbm.mean())}
        print(cls, gname, {k: round(x, 3) for k, x in out["swaps"][gname].items()}, flush=True)
    # elapsed distributions by chance bucket, sim vs real
    for b in (0, 1):
        es, er = Xs[cb_s == b, fi["chance_elapsed_s"]], Xr[cb_r == b, fi["chance_elapsed_s"]]
        out[f"elapsed_q_chance{'1' if b == 0 else '2plus'}"] = {
            "sim": np.quantile(es, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            "real": np.quantile(er, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()}
        print(cls, b, out[f"elapsed_q_chance{'1' if b == 0 else '2plus'}"], flush=True)
    res["features"][cls] = out
OUT.write_text(json.dumps(res, indent=1, default=float))
