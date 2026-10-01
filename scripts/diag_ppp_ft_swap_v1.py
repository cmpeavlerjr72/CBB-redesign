"""diag_ppp_ft_swap_v1.py -- FT% channel: is the sim's FT deficit the free-throw model or its inputs? (lane I, 2026-10-01)

DIAGNOSTIC ONLY. Real 2024-25 FT attempts with the served features (FT.build_ft_design over 2022-2025, as
scripts/train_free_throw_v2_site.py builds it) are scored by the SERVED S1 artifacts (adapter models). The
sim's FT attempts (diag_ppp_tap_v1.py ft_*.parquet, COMB) are re-scored after swapping feature groups to the
values of a real attempt drawn from the same (game, shooting team) cell: shooter block (shooter_ft_asof,
shooter_fta_asof, prior_season_ft, has_prior_season) and state (seconds_remaining, period, score_diff,
in_bonus). Neutral-site games are skipped (the tap records the shooting side only through site_home).

Usage: diag_ppp_ft_swap_v1.py <tap_dir> <out_json>
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
os.chdir(ROOT)
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
from cbb_sim.engine.adapters import Adapters  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402
from cbb_sim.models import event_stream as ES  # noqa: E402

TAP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
FEATS = ["shooter_ft_asof", "shooter_fta_asof", "prior_season_ft", "has_prior_season", "season_idx",
         "seconds_remaining", "period", "score_diff", "in_bonus"]
SHOOTER = FEATS[:4]
STATE = ["seconds_remaining", "period", "score_diff", "in_bonus"]

inp = EngineInputs.load("data/processed/models/engine_v3", "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
games = inp.games.reset_index(drop=True)
gid_of = games["game_id"].to_numpy()
pos = {int(g): i for i, g in enumerate(gid_of)}


def predict(X, gidx):
    out = np.empty(len(X))
    segs = ad.ft.manifest.segments(gidx.astype(np.int64))
    for k in np.unique(segs):
        r = np.flatnonzero(segs == k)
        out[r] = ad.ft.models_by_seg[k].predict_proba(np.ascontiguousarray(X[r], dtype=np.float32))[:, FT.CLASS_INDEX["MAKE"]]
    return out


ES.load_universe()
att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
att = att[att["season"].isin([2022, 2023, 2024, 2025])]
d = FT.build_ft_design(att)
d = d[d["season"] == 2025].copy()
d = d[d["game_id"].map(pos).notna()]
d["gidx"] = d["game_id"].map(pos).astype(np.int64)
Xr = np.asarray(FT.design_matrix(d, tuple(FEATS)), dtype=np.float64)
p_real = predict(Xr, d["gidx"].to_numpy())
y = d["y"].to_numpy(np.float64)
res = {"real_full_season": {"n": int(len(d)), "mean_p": float(p_real.mean()), "mean_y": float(y.mean())}}
s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(TAP / "ft_*.parquet")))], ignore_index=True)
s.columns = ["seed", "gidx", "off_is_home", *FEATS, "p"]
gidx_s = s["gidx"].to_numpy().astype(np.int64)
neutral = games["neutral"].to_numpy()[gidx_s] > 0
s = s[~neutral].reset_index(drop=True)
gidx_s = s["gidx"].to_numpy().astype(np.int64)
s["team_id"] = np.where(s["off_is_home"] > 0.5, games["home_team_id"].to_numpy()[gidx_s],
                        games["away_team_id"].to_numpy()[gidx_s])
s["game_id"] = gid_of[gidx_s]
samp = set(s["game_id"].unique())
dr = d[d["game_id"].isin(samp)].reset_index(drop=True)
Xrs = np.asarray(FT.design_matrix(dr, tuple(FEATS)), dtype=np.float64)
p_rs = predict(Xrs, dr["gidx"].to_numpy())
Xs = s[FEATS].to_numpy(np.float64)
p_sim = predict(Xs, gidx_s)
key_r = dr["game_id"].astype(np.int64).to_numpy() * 100000 + dr["team_id"].astype(np.int64).to_numpy()
key_s = s["game_id"].astype(np.int64).to_numpy() * 100000 + s["team_id"].astype(np.int64).to_numpy()
order = np.argsort(key_r, kind="stable")
uk, start, cnt = np.unique(key_r[order], return_index=True, return_counts=True)
loc = np.clip(np.searchsorted(uk, key_s), 0, len(uk) - 1)
hs = uk[loc] == key_s
rng = np.random.default_rng(20261001)
draw = np.full(len(s), -1, dtype=np.int64)
u = rng.random(len(s))
draw[hs] = order[start[loc[hs]] + (u[hs] * cnt[loc[hs]]).astype(np.int64)]


def swap(group):
    X = Xs[hs].copy()
    for f in group:
        j = FEATS.index(f)
        X[:, j] = Xrs[draw[hs], j]
    return predict(X, gidx_s[hs])


base = p_sim[hs]
res["sample"] = {
    "n_sim_attempts": int(len(s)), "matched_share": float(hs.mean()), "n_real_attempts": int(len(dr)),
    "sim_mean_p_tap": float(s["p"].mean()), "sim_mean_p_recomputed": float(p_sim.mean()),
    "sim_mean_p_matched": float(base.mean()), "real_mean_p": float(p_rs.mean()),
    "real_mean_y": float(dr["y"].mean()),
    "swap_shooter_d_pp": 100 * float(swap(SHOOTER).mean() - base.mean()),
    "swap_state_d_pp": 100 * float(swap(STATE).mean() - base.mean()),
    "swap_shooter_state_d_pp": 100 * float(swap(SHOOTER + STATE).mean() - base.mean()),
    "swap_all_d_pp": 100 * float(swap(FEATS).mean() - base.mean()),
    "feature_means": {f: {"sim": float(Xs[:, j].mean()), "real": float(Xrs[:, j].mean())} for j, f in enumerate(FEATS)},
}
for f in STATE + SHOOTER:
    res["sample"][f"swap_{f}_d_pp"] = 100 * float(swap([f]).mean() - base.mean())
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
