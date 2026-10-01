"""diag_ft_late_leader_v1.py -- late-game free throws by the LEADING team: who shoots them, sim vs real (lane I, 2026-10-01).

DIAGNOSTIC ONLY (PM request: lane L reads the leading team's late FT make at 0.64 sim vs 0.77 actual). Sim = the served-v2
tap (diag_oreb_ft_tap_v1.py, all 5,710 games x 2 seeds; per attempt: the model's assembled features and p). Real = the 2025
attempts of FT.build_ft_design. Window: period 2, seconds_remaining <= 120 (and <= 60), shooting team ahead (score_diff > 0).

Per window, sim vs real: attempt share, realised FT% (real) / mean p (sim and real), and the shooter block. "Who shoots" is
isolated by re-scoring every attempt with the served model at its OWN shooter block and a FIXED state (the real window's
median seconds, period 2, score_diff 0, in_bonus 1), so a gap there is the shooter mix alone; the rest is the state the model
is fed (score_diff carries most of it).

Usage: diag_ft_late_leader_v1.py <tap_dir> <out_json>
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from cbb_sim.engine.adapters import Adapters  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402

TAP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
FEATS = ["shooter_ft_asof", "shooter_fta_asof", "prior_season_ft", "has_prior_season", "season_idx",
         "seconds_remaining", "period", "score_diff", "in_bonus"]
inp = EngineInputs.load("data/processed/models/engine_v3", "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
pos = {int(g): i for i, g in enumerate(inp.games["game_id"].to_numpy())}


def predict(X, gidx):
    out = np.empty(len(X))
    segs = ad.ft.manifest.segments(gidx.astype(np.int64))
    for k in np.unique(segs):
        r = np.flatnonzero(segs == k)
        out[r] = ad.ft.models_by_seg[k].predict_proba(np.ascontiguousarray(X[r], dtype=np.float32))[:, 1]
    return out


s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(TAP / "ft_*.parquet")))], ignore_index=True)
s.columns = ["seed", "gidx", "off_is_home", *FEATS, "p"]
ES.load_universe()
att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
att = att[att["season"].isin([2022, 2023, 2024, 2025])]
d = FT.build_ft_design(att)
d = d[(d["season"] == 2025) & d["game_id"].isin(pos)].reset_index(drop=True)
d["gidx"] = d["game_id"].map(pos).astype(np.int64)
Xr = np.asarray(FT.design_matrix(d, tuple(FEATS)), dtype=np.float64)
real = pd.DataFrame(Xr, columns=FEATS)
real["gidx"], real["y"] = d["gidx"].to_numpy(), d["y"].to_numpy()
real["p"] = predict(Xr, real["gidx"].to_numpy())
res = {"sim_attempts": int(len(s)), "real_attempts": int(len(real))}
for name, sec in (("last120", 120), ("last60", 60)):
    out = {}
    for lab, df in (("sim", s), ("real", real)):
        late = (df["period"] == 2) & (df["seconds_remaining"] <= sec)
        for side, m in (("leading", late & (df["score_diff"] > 0)), ("trailing", late & (df["score_diff"] < 0)),
                        ("tied", late & (df["score_diff"] == 0))):
            x = df[m]
            if not len(x):
                continue
            Xf = x[FEATS].to_numpy(np.float64).copy()
            Xf[:, FEATS.index("seconds_remaining")] = sec / 2
            Xf[:, FEATS.index("period")] = 2
            Xf[:, FEATS.index("score_diff")] = 0
            Xf[:, FEATS.index("in_bonus")] = 1
            pf = predict(Xf, x["gidx"].to_numpy().astype(np.int64))
            out[f"{lab}_{side}"] = {
                "n": int(len(x)), "share_of_all": float(len(x) / len(df)), "mean_p": float(x["p"].mean()),
                "realised": float(x["y"].mean()) if "y" in x else None,
                "p_fixed_state_who_shoots": float(pf.mean()),
                "shooter_ft_asof": float(x["shooter_ft_asof"].mean()), "has_prior": float(x["has_prior_season"].mean()),
                "prior_season_ft": float(x["prior_season_ft"].mean()), "fta_asof": float(x["shooter_fta_asof"].mean()),
                "score_diff_mean": float(x["score_diff"].mean()), "in_bonus": float(x["in_bonus"].mean())}
    res[name] = out
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
