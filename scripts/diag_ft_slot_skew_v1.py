"""diag_ft_slot_skew_v1.py -- FT shooter block: does the ENGINE serve the features the FT model was trained on? (lane I, 2026-10-01)

DIAGNOSTIC ONLY. Nothing in src/ is changed.

For every real 2024-25 free-throw attempt (FT.build_ft_design over 2022-2025, as the FT trainer builds it) whose game is in
the v3 engine inputs, find the shooter's roster slot in that game (inputs.roster_cbbd) and compare the slot's served FT
features (shooter_ft_asof, shooter_fta_asof, prior_season_ft, has_prior_season_ft) with the design row's own values.
Both feature sets are scored by the SERVED S1_conf_aligned artifacts with the real state, so the difference in mean p is
the train/serve skew of the shooter block in probability points, holding WHO shoots fixed.

Also: offline calibration of the served model on the real rows by has_prior_season and by prior-season-FT quintile
(is the model's newcomer prior too low / too high?).

Usage: diag_ft_slot_skew_v1.py <input_dir> <out_json>
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
SHOOTER = FEATS[:4]
SLOT_COL = {"shooter_ft_asof": "shooter_ft_asof", "shooter_fta_asof": "shooter_fta_asof",
            "prior_season_ft": "prior_season_ft", "has_prior_season": "has_prior_season_ft"}

inp = EngineInputs.load(IN_DIR, "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
games = inp.games.reset_index(drop=True)
pos = {int(g): i for i, g in enumerate(games["game_id"].to_numpy())}


def predict(X, gidx):
    out = np.empty(len(X))
    segs = ad.ft.manifest.segments(gidx.astype(np.int64))
    for k in np.unique(segs):
        r = np.flatnonzero(segs == k)
        out[r] = ad.ft.models_by_seg[k].predict_proba(
            np.ascontiguousarray(X[r], dtype=np.float32))[:, FT.CLASS_INDEX["MAKE"]]
    return out


ES.load_universe()
att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
att = att[att["season"].isin([2022, 2023, 2024, 2025])]
d = FT.build_ft_design(att)
d = d[d["season"] == 2025].copy()
d = d[d["game_id"].map(pos).notna()].reset_index(drop=True)
d["gidx"] = d["game_id"].map(pos).astype(np.int64)
g = d["gidx"].to_numpy()
home = games["home_team_id"].to_numpy()[g]
away = games["away_team_id"].to_numpy()[g]
side = np.where(d["team_id"].to_numpy() == home, 0, np.where(d["team_id"].to_numpy() == away, 1, -1))
res: dict = {"n_attempts_2025_in_inputs": int(len(d)), "side_unresolved": int((side < 0).sum())}
sid = d["shooter_id"].to_numpy().astype(np.int64)
ros = inp.roster_cbbd[g, np.clip(side, 0, 1)]              # (n, S)
hit = (ros == sid[:, None]) & (side[:, None] >= 0)
found = hit.any(axis=1)
slot = np.where(found, hit.argmax(axis=1), -1)
# other side, to prove the side mapping
ros_o = inp.roster_cbbd[g, 1 - np.clip(side, 0, 1)]
res["shooter_on_roster_share"] = float(found.mean())
res["shooter_on_other_side_share"] = float((ros_o == sid[:, None]).any(axis=1).mean())
res["slot_names_ft"] = {k: inp.slot_names.get(v) for k, v in SLOT_COL.items()}

Xd = np.asarray(FT.design_matrix(d, tuple(FEATS)), dtype=np.float64)
y = d["y"].to_numpy(np.float64)
p_d = predict(Xd, g)
res["offline_full"] = {"mean_p": float(p_d.mean()), "mean_y": float(y.mean())}

# served slot values for the found attempts
f = np.flatnonzero(found)
Xs = Xd.copy()
for c, sc in SLOT_COL.items():
    j = FEATS.index(c)
    Xs[f, j] = inp.slot_static[g[f], side[f], slot[f], inp.slot_names[sc]]
p_s = predict(Xs[f], g[f])
res["found"] = {"n": int(len(f)), "mean_y": float(y[f].mean()), "mean_p_design": float(p_d[f].mean()),
                "mean_p_served_slot": float(p_s.mean()),
                "skew_pp": 100 * float(p_s.mean() - p_d[f].mean())}
cmp = {}
for c in SHOOTER:
    j = FEATS.index(c)
    a, b = Xd[f, j], Xs[f, j]
    cmp[c] = {"design_mean": float(a.mean()), "served_mean": float(b.mean()),
              "exact_share": float(np.isclose(a, b, atol=1e-5).mean()),
              "mean_abs_diff": float(np.abs(a - b).mean())}
hp_d, hp_s = Xd[f, FEATS.index("has_prior_season")], Xs[f, FEATS.index("has_prior_season")]
cmp["has_prior_crosstab"] = {f"design{int(i)}_served{int(k)}": int(((hp_d == i) & (hp_s == k)).sum())
                             for i in (0, 1) for k in (0, 1)}
res["feature_compare"] = cmp
# single-feature swaps (served -> design) on the found rows: which served column carries the skew
sw = {}
for c in SHOOTER:
    j = FEATS.index(c)
    X = Xs[f].copy()
    X[:, j] = Xd[f, j]
    sw[c] = 100 * float(predict(X, g[f]).mean() - p_s.mean())
res["swap_served_to_design_pp"] = sw
# where the served has_prior is 0 but design 1: what is the served row? (stale / zero fill)
m01 = (hp_d == 1) & (hp_s == 0)
if m01.any():
    ff = f[m01]
    res["design1_served0"] = {
        "n": int(m01.sum()),
        "served_fta_asof_zero_share": float((Xs[ff, FEATS.index("shooter_fta_asof")] == 0).mean()),
        "design_fta_asof_zero_share": float((Xd[ff, FEATS.index("shooter_fta_asof")] == 0).mean()),
        "served_prior_ft_zero_share": float((Xs[ff, FEATS.index("prior_season_ft")] == 0).mean()),
        "mean_p_design": float(p_d[ff].mean()), "mean_p_served": float(predict(Xs[ff], g[ff]).mean()),
        "mean_y": float(y[ff].mean())}
# fta_asof: served lag (stale by the last FT game?)
ja = FEATS.index("shooter_fta_asof")
res["fta_asof_served_minus_design_quantiles"] = [float(x) for x in np.quantile(Xs[f, ja] - Xd[f, ja], [0.05, .25, .5, .75, .95])]

# offline calibration by group (design features)
cal = {}
hp = Xd[:, FEATS.index("has_prior_season")]
for k in (0, 1):
    m = hp == k
    cal[f"has_prior_{k}"] = {"n": int(m.sum()), "share": float(m.mean()), "mean_p": float(p_d[m].mean()),
                             "mean_y": float(y[m].mean()), "gap_pp": 100 * float(p_d[m].mean() - y[m].mean())}
m = hp == 0
fa = Xd[:, ja]
for lo, hi in ((0, 1), (1, 10), (10, 30), (30, 1e9)):
    mm = m & (fa >= lo) & (fa < hi)
    if mm.any():
        cal[f"no_prior_fta_asof_{lo}_{int(min(hi, 9999))}"] = {
            "n": int(mm.sum()), "mean_p": float(p_d[mm].mean()), "mean_y": float(y[mm].mean()),
            "gap_pp": 100 * float(p_d[mm].mean() - y[mm].mean())}
mp = hp == 1
q = pd.qcut(Xd[mp, FEATS.index("prior_season_ft")], 5, labels=False, duplicates="drop")
cal["prior_quintile"] = [{"q": int(k), "mean_p": float(p_d[mp][q == k].mean()), "mean_y": float(y[mp][q == k].mean())}
                         for k in range(5)]
res["offline_calibration"] = cal
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
