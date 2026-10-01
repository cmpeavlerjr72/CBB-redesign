"""diag_tov_real_v1.py -- possession_outcome's turnover rate on REAL chances, realised vs the served model (lane I, 2026-10-01).

DIAGNOSTIC ONLY. Rows: the round-2 PO design (chance level; `y` = class, `population` first/cont). For 2025 (F2 test) every
row is scored by the SERVED event artifacts (`engine/event_round2_s1_F2_2025`, monthly S1 segments: first = lgbm, cont =
cascade) on its own design features, so mean P(TOV) vs the realised TOV share is the model's offline level per cell.
Cells: chance 1 / 2 / 3+; chance 1 transition vs half-court; chance 2+ putback (duration <= 4 s) vs reset (> 4 s);
offence team prior quintile (`off_tov_c`, the served as-of team TOV feature) and season (2024 realised for context).

Usage: diag_tov_real_v1.py <out_json>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from cbb_sim.models import possession_outcome as PO  # noqa: E402

OUT = Path(sys.argv[1])
TOV = PO.CLASS_INDEX["TOV"]
ED = Path("data/processed/models/engine/event_round2_s1_F2_2025")
idx = json.loads((ED / "index.json").read_text())
d = pd.read_parquet("data/processed/models/possession_outcome/round2/design.parquet")
d = d[d["season"].isin([2024, 2025])].reset_index(drop=True)
d["tov"] = (d["y"] == TOV).astype(float)
d["ch"] = np.minimum(d["chance_number"], 3)
d["putback"] = (d["ch"] >= 2) & (d["duration_s"] <= 4)
d["gd"] = pd.to_datetime(d["game_date"])
d["p"] = np.nan
x = d[d["season"] == 2025]
refits = [pd.Timestamp(r) for r in idx["refit_dates"]]
for pop in ("first", "cont"):
    for k, r in enumerate(refits):
        nxt = refits[k + 1] if k + 1 < len(refits) else None
        m = (x["population"] == pop) & (x["gd"] >= r if k else True) & ((x["gd"] < nxt) if nxt is not None else True)
        if not m.any():
            continue
        art = joblib.load(ED / f"{pop}_{r.date()}.joblib")
        X = x.loc[m, art["features"]].to_numpy(dtype=np.float32 if pop == "first" else np.float64)
        d.loc[x.index[m], "p"] = art["model"].predict_proba(X)[:, TOV]
d["q"] = d.groupby("season")["off_tov_c"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))


def cell(z):
    out = {"n": int(len(z)), "tov_rate": float(z["tov"].mean())}
    if z["p"].notna().any():
        out["model_p"] = float(z["p"].mean())
        out["gap_pp"] = 100 * float(z["p"].mean() - z["tov"].mean())
    return out


res = {}
for s, z in d.groupby("season"):
    r = {"all": cell(z), "population": {p: cell(w) for p, w in z.groupby("population")},
         "chance": {int(c): cell(w) for c, w in z.groupby("ch")},
         "chance1_transition": {int(t): cell(w) for t, w in z[z["ch"] == 1].groupby("is_transition")},
         "chance2p_putback": {str(bool(t)): cell(w) for t, w in z[z["ch"] >= 2].groupby("putback")},
         "chances_per_possession": float(len(z) / z.groupby(["game_id", "period", "poss_index"]).ngroups),
         "team_tov_prior_quintile_first": {int(q): cell(w) for q, w in z[z["population"] == "first"].groupby("q")},
         "team_tov_prior_quintile_cont": {int(q): cell(w) for q, w in z[z["population"] == "cont"].groupby("q")}}
    res[int(s)] = r
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
