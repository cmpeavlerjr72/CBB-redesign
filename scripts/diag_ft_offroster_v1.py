"""diag_ft_offroster_v1.py -- who are the real 2024-25 free-throw shooters the engine roster does not carry? (lane I, 2026-10-01)

DIAGNOSTIC ONLY. Real attempts (FT.build_ft_design, as the FT trainer) whose shooter is not in the game's engine roster
(v3 inputs roster_cbbd) stand, in the sim, for the anonymous tail slots, which the engine serves with an all-zero FT
shooter block (= no prior season, 0 attempts, league-mean as-of). This reports their real features, their FT%, the
served model's p on their REAL features, and on the all-zero block the anonymous slots receive (same real states).
Also: the game's anonymous-slot count vs the real off-roster attempts, by month (early-season vs late).

Usage: diag_ft_offroster_v1.py <input_dir> <out_json>
"""
from __future__ import annotations

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

IN_DIR, OUT = sys.argv[1], Path(sys.argv[2])
FEATS = ["shooter_ft_asof", "shooter_fta_asof", "prior_season_ft", "has_prior_season", "season_idx",
         "seconds_remaining", "period", "score_diff", "in_bonus"]
inp = EngineInputs.load(IN_DIR, "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
games = inp.games.reset_index(drop=True)
pos = {int(g): i for i, g in enumerate(games["game_id"].to_numpy())}


def predict(X, gidx):
    out = np.empty(len(X))
    segs = ad.ft.manifest.segments(gidx.astype(np.int64))
    for k in np.unique(segs):
        r = np.flatnonzero(segs == k)
        out[r] = ad.ft.models_by_seg[k].predict_proba(np.ascontiguousarray(X[r], dtype=np.float32))[:, 1]
    return out


ES.load_universe()
att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
att = att[att["season"].isin([2022, 2023, 2024, 2025])]
d = FT.build_ft_design(att)
d = d[(d["season"] == 2025) & d["game_id"].isin(pos)].reset_index(drop=True)
g = d["game_id"].map(pos).to_numpy().astype(np.int64)
side = np.where(d["team_id"].to_numpy() == games["home_team_id"].to_numpy()[g], 0, 1)
ros = inp.roster_cbbd[g, side]
found = (ros == d["shooter_id"].to_numpy().astype(np.int64)[:, None]).any(axis=1)
n_anon = (inp.roster_cbbd[g, side] < 0).sum(axis=1)
X = np.asarray(FT.design_matrix(d, tuple(FEATS)), dtype=np.float64)
y = d["y"].to_numpy(np.float64)
p_real = predict(X, g)
Xz = X.copy()
for c in FEATS[:4]:
    Xz[:, FEATS.index(c)] = 0.0
p_zero = predict(Xz, g)
res = {}
for lab, m in (("on_roster", found), ("off_roster", ~found)):
    res[lab] = {"n": int(m.sum()), "share": float(m.mean()), "ft_pct": float(y[m].mean()),
                "p_real_features": float(p_real[m].mean()), "p_all_zero_block": float(p_zero[m].mean()),
                "has_prior_share": float(d["has_prior_season"].to_numpy()[m].mean()),
                "fta_asof_zero_share": float((d["shooter_fta_asof"].to_numpy()[m] == 0).mean()),
                "fta_asof_median": float(np.median(d["shooter_fta_asof"].to_numpy()[m])),
                "game_has_anon_slot_share": float((n_anon[m] > 0).mean())}
mo = pd.to_datetime(d["game_date"]).dt.month.to_numpy()
res["off_roster_by_month"] = {int(k): {"share": float((~found)[mo == k].mean()), "n": int((mo == k).sum()),
                                       "ft_pct_off": float(y[(mo == k) & ~found].mean()) if ((mo == k) & ~found).any() else None}
                              for k in sorted(set(mo))}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
