"""diag_ppp_tap_grade_v1.py -- grade the lane-I fg_make tap: is the sim's make deficit the model or its inputs?

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
res["classes"] = {}
pts_rows = []
for cls in ("FGA_rim", "FGA_jump2", "FGA_3"):
    feats = json.loads((Path(ad.fg.source[cls]["path"]) / f"manifest_{cls}.json").read_text())["features"]
    s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(TAP / f"fg_{cls}_*.parquet")))],
                  ignore_index=True)
    gidx_s = s["gidx"].to_numpy().astype(np.int64)
    # offence side: match the row's off_rating_off_c against both sides' team block
    j = TN["off_rating_off_c"]
    v = s["off_rating_off_c"].to_numpy()
    side = np.where(np.abs(TS[gidx_s, 0, j] - v) <= np.abs(TS[gidx_s, 1, j] - v), 0, 1)
    home_id = games["home_team_id"].to_numpy()[gidx_s]
    away_id = games["away_team_id"].to_numpy()[gidx_s]
    s["off_team_id"] = np.where(side == 0, home_id, away_id)
    s["game_id"] = gid_of[gidx_s]
    Xs = s[feats].to_numpy(np.float64)
    p_sim_tap = s["p"].to_numpy(np.float64)
    p_sim_re = predict(cls, Xs, gidx_s)
    rc = real[real["shot_class"] == cls].reset_index(drop=True)
    Xr = FG.design_matrix(rc, feats).astype(np.float64)
    Xr = np.asarray(Xr, dtype=np.float64)
    p_real = predict(cls, Xr, rc["gidx"].to_numpy())
    y_real = rc["y"].to_numpy(np.float64)
    # cell match
    key_r = rc["game_id"].astype(np.int64).to_numpy() * 100000 + rc["off_team_id"].astype(np.int64).to_numpy()
    key_s = s["game_id"].astype(np.int64).to_numpy() * 100000 + s["off_team_id"].astype(np.int64).to_numpy()
    order = np.argsort(key_r, kind="stable")
    ks = key_r[order]
    uk, start, cnt = np.unique(ks, return_index=True, return_counts=True)
    loc = np.searchsorted(uk, key_s)
    loc = np.clip(loc, 0, len(uk) - 1)
    has = uk[loc] == key_s
    draw = np.full(len(s), -1, dtype=np.int64)
    u = rng.random(len(s))
    draw[has] = order[start[loc[has]] + (u[has] * cnt[loc[has]]).astype(np.int64)]
    hs = has
    fi = {f: feats.index(f) for f in feats}

    def swap(group):
        X = Xs[hs].copy()
        for f in group:
            X[:, fi[f]] = Xr[draw[hs], fi[f]]
        return predict(cls, X, gidx_s[hs])

    team_f = [f for f in feats if f not in SHOOTER + STATE]
    p_sh = swap(SHOOTER)
    p_sh_st = swap(SHOOTER + STATE)
    p_st = swap(STATE)
    p_all = swap(feats)                             # == the drawn real rows' own features, sim gidx
    p_real_drawn = p_real[draw[hs]]                 # same rows, real gidx (identical games)
    # sim realised make rate on the same games/seeds from the tap games
    a_col, m_col = BOXCOL[cls]
    sim_real = float((tg[f"home_{m_col}"].sum() + tg[f"away_{m_col}"].sum())
                     / (tg[f"home_{a_col}"].sum() + tg[f"away_{a_col}"].sum()))
    fga_pg = float((tg[f"home_{a_col}"].sum() + tg[f"away_{a_col}"].sum()) / n_sims)
    chain = [
        ("sim realised make", sim_real),
        ("sim mean p (tap)", float(p_sim_tap.mean())),
        ("sim mean p, matched-cell rows only", float(p_sim_re[hs].mean())),
        ("+ shooter swapped to real", float(p_sh.mean())),
        ("+ state swapped to real", float(p_sh_st.mean())),
        ("+ team features swapped (serve -> train)", float(p_all.mean())),
        ("real mean p, real cell weights (offline)", float(p_real.mean())),
        ("real realised", float(y_real.mean())),
    ]
    steps = []
    for (a, va), (b, vb) in zip(chain[:-1], chain[1:]):
        steps.append({"step": f"{a} -> {b}", "d_pp": 100 * (vb - va),
                      "pts_per_game": -(vb - va) * fga_pg * PV[cls]})
    # NOTE sign: pts_per_game here is the contribution to SIM - ACTUAL (moving from sim to real by +x
    # means the sim is short by x), so it is -(vb - va) * attempts * value.
    feat_means = {f: {"sim": float(Xs[:, fi[f]].mean()), "real": float(Xr[:, fi[f]].mean())} for f in feats}
    extra_cut = {}
    # by chance number (1, 2+) and transition on the sim vs real side, own features
    for nm, sel_s, sel_r in (
        ("chance1", Xs[:, fi["chance_number"]] == 1, Xr[:, fi["chance_number"]] == 1),
        ("chance2plus", Xs[:, fi["chance_number"]] >= 2, Xr[:, fi["chance_number"]] >= 2),
        ("transition", Xs[:, fi["is_transition_f"]] == 1, Xr[:, fi["is_transition_f"]] == 1),
        ("late_2h_last5min", (Xs[:, fi["period"]] >= 2) & (Xs[:, fi["seconds_remaining"]] <= 300),
         (Xr[:, fi["period"]] >= 2) & (Xr[:, fi["seconds_remaining"]] <= 300)),
    ):
        extra_cut[nm] = {"sim_share": float(sel_s.mean()), "real_share": float(sel_r.mean()),
                         "sim_mean_p": float(p_sim_tap[sel_s].mean()) if sel_s.any() else None,
                         "real_mean_p": float(p_real[sel_r].mean()) if sel_r.any() else None,
                         "real_y": float(y_real[sel_r].mean()) if sel_r.any() else None}
    # shooter dev quintiles (real cut points): share of attempts sim vs real
    qs = np.quantile(Xr[:, fi["shooter_shrunk_dev_c"]], [0.2, 0.4, 0.6, 0.8])
    qsim = np.bincount(np.searchsorted(qs, Xs[:, fi["shooter_shrunk_dev_c"]]), minlength=5) / len(Xs)
    qreal = np.bincount(np.searchsorted(qs, Xr[:, fi["shooter_shrunk_dev_c"]]), minlength=5) / len(Xr)
    # full-season offline calibration (all 2025 real rows of the class)
    ra = real_all[real_all["shot_class"] == cls]
    ra_g = ra["game_id"].map(pos)
    ok = ra_g.notna().to_numpy()
    p_ra = predict(cls, np.asarray(FG.design_matrix(ra[ok], feats), dtype=np.float64),
                   ra_g[ok].astype(np.int64).to_numpy())
    res["classes"][cls] = {
        "n_sim_rows": int(len(s)), "matched_share": float(hs.mean()), "n_real_rows": int(len(rc)),
        "fga_per_game_sim": fga_pg,
        "tap_p_vs_recomputed_maxabs": float(np.max(np.abs(p_sim_tap - p_sim_re))),
        "chain": chain, "steps": steps,
        "reverse_order": {"state_first_d_pp": 100 * float(p_st.mean() - p_sim_re[hs].mean()),
                          "then_shooter_d_pp": 100 * float(p_sh_st.mean() - p_st.mean())},
        "feature_means": feat_means, "cuts": extra_cut,
        "shooter_quintile_share": {"sim": qsim.tolist(), "real": qreal.tolist()},
        "full_season_offline": {"n": int(ok.sum()), "mean_p": float(p_ra.mean()),
                                "mean_y": float(ra[ok]["y"].mean())},
    }
    print(cls, json.dumps({"chain": [(a, round(b, 4)) for a, b in chain]}), flush=True)
    for st_ in steps:
        print("   ", st_["step"], round(st_["d_pp"], 3), "pp", round(st_["pts_per_game"], 3), "pts", flush=True)

res["n_sims"] = int(n_sims)
res["n_seeds"] = int(n_seeds)
OUT.write_text(json.dumps(res, indent=1, default=float))
