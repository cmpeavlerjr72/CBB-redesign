"""train_shot_location_v1.py -- shot-location round 1, Read B (experiments.md 1.4): is the within-class band mix of a
team-game predictable before tipoff, and does the defence's allowed profile add to the offence's own?

Unit: LOCATED attempt of FGA_jump2 (target short2 / long2) or FGA_3 (target corner3 / wing3 / top3). Static fit on the
fold's training seasons; scored on the located attempts of the test season. Arms:

  B0  league-average band shares by class (training seasons pooled) -- the null, what the engine assumes today
  B1  multinomial logit on off_band_c + site + state (team profile only)
  B2  B1 + def_band_c (offence x defence)
  B3  LightGBM multiclass on B2's columns, fg_make frozen ladder params of the class (lgbm_ladder_v2)
  B2alt  B2 on the as-of profiles built with the fold-1 SECOND-best (k, rho) (the registered sensitivity fit)

Columns (from `build_shot_location_bands_v1.py --part build`, row-aligned to the fg_make design):
  off_band_c : jump2 -> off_short2_c (long2 is exactly its negative in log-odds); three -> off_corner3_c,
               off_wing3_c, off_top3_c; def_band_c likewise with def_.
  site       : site_home, site_away (neutral = reference)
  state      : the C_plus_state game-state columns as carried on the fg_make design: period, seconds_remaining,
               score_diff, in_bonus, is_transition_f, chance_number
GLM: sklearn LogisticRegression (lbfgs, multinomial, C = 1.0) on features standardised on the training slice (the
`RidgeLogitArm` convention of the rebound module). Deterministic: the seed floor is zero.

Predictions are written for ALL test-season attempts of the class (located or not), because Read C needs P(band) for
every attempt; Read B itself is graded on located rows only.

    .venv/Scripts/python.exe scripts/train_shot_location_v1.py --fold F2 --n-jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402

DESIGN = ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet"
OUT = ROOT / "data/processed/models/shot_location/round1"
LOC = OUT / "design_loc_cols_v1.parquet"
BANDS = {"FGA_jump2": ("short2", "long2"), "FGA_3": ("corner3", "wing3", "top3")}
PROF = {"FGA_jump2": ("short2",), "FGA_3": ("corner3", "wing3", "top3")}
SITE = ["site_home", "site_away"]
STATE = ["period", "seconds_remaining", "score_diff", "in_bonus", "is_transition_f", "chance_number"]
ARMS = ("B0", "B1", "B2", "B3", "B2alt")


def feats(cls: str, arm: str) -> list[str]:
    off = [f"off_{b}_c" for b in PROF[cls]]
    dfn = [f"def_{b}_c" for b in PROF[cls]]
    if arm == "B1":
        return off + SITE + STATE
    if arm in ("B2", "B3"):
        return off + dfn + SITE + STATE
    if arm == "B2alt":
        return [c + "_alt" for c in off + dfn] + SITE + STATE
    raise KeyError(arm)


def fit_predict(arm, tr, te_all, cls, n_jobs, seed=0):
    k = len(BANDS[cls])
    y = tr["y_band"].to_numpy()
    if arm == "B0":
        p0 = np.bincount(y, minlength=k) / len(y)
        return np.repeat(p0[None, :], len(te_all), axis=0), {"shares": p0.tolist()}
    f = feats(cls, arm)
    X = tr[f].to_numpy(dtype="float64")
    Xt = te_all[f].to_numpy(dtype="float64")
    if arm in ("B1", "B2", "B2alt"):
        from sklearn.linear_model import LogisticRegression
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-8] = 1.0
        clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs").fit((X - mu) / sd, y)
        p = clf.predict_proba((Xt - mu) / sd)
        coef = clf.coef_.tolist()
        return p, {"features": f, "coef_std": coef, "intercept": clf.intercept_.tolist(), "n_iter": int(clf.n_iter_[0])}
    import lightgbm as lgb
    ladder = json.loads((ROOT / "data/processed/models/fg_make/lgbm_ladder_v2.json").read_text(encoding="utf-8"))
    prm = {kk: v for kk, v in ladder["frozen_params"][cls].items() if kk != "n_jobs"}
    base = {**FG.LGBM_BASE, **prm}
    if k > 2:
        base.update(objective="multiclass", num_class=k)
    clf = lgb.LGBMClassifier(random_state=seed, n_jobs=n_jobs, **base).fit(X, y)
    p = clf.predict_proba(Xt)
    return p, {"features": f, "params": base}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(a.n_jobs)
    t0 = time.time()
    spec = FG.FOLDS[a.fold]
    assert_not_sealed(spec["train"] + spec["test"], context="shot_location Read B")
    od = OUT / "readB" / a.fold
    if od.exists() and any(od.iterdir()):
        raise SystemExit(f"{od} not empty")
    od.mkdir(parents=True, exist_ok=True)
    des = pd.read_parquet(DESIGN, columns=["game_id", "season", "shot_class", "shooter_id", "made", *SITE, *STATE])
    loc = pd.read_parquet(LOC)
    for k in ("game_id", "shooter_id", "shot_class", "made"):
        assert (loc[k].to_numpy() == des[k].to_numpy()).all(), k
    for c in loc.columns:
        if c.startswith(("off_", "def_")) or c in ("band", "located"):
            des[c] = loc[c].to_numpy()
    des["in_bonus"] = des["in_bonus"].astype("float64")
    rep = {"fold": a.fold, "created_at": pd.Timestamp.now("UTC").isoformat(), "by_class": {}}
    outs = []
    for cls, bands in BANDS.items():
        C = des[des["shot_class"] == cls]
        tr = C[C["season"].isin(spec["train"]) & (C["band"] != "")].copy()
        te_all = C[C["season"].isin(spec["test"])].copy()
        tr["y_band"] = tr["band"].map({b: i for i, b in enumerate(bands)}).astype("int64")
        o = pd.DataFrame({"fg_row": te_all.index.to_numpy(), "shot_class": cls,
                          "located": (te_all["band"] != "").to_numpy(), "band": te_all["band"].to_numpy()})
        rep["by_class"][cls] = {"n_train": int(len(tr)), "n_test_all": int(len(te_all)),
                                "n_test_located": int(o["located"].sum()), "arms": {}}
        for arm in ARMS:
            p, info = fit_predict(arm, tr, te_all, cls, a.n_jobs)
            for i, b in enumerate(bands):
                o[f"{arm}__p_{b}"] = p[:, i]
            rep["by_class"][cls]["arms"][arm] = info
            print(f"[{time.time() - t0:6.1f}s] {a.fold} {cls} {arm} fitted", flush=True)
        outs.append(o)
    pd.concat(outs, ignore_index=True).to_parquet(od / "preds.parquet", index=False)
    rep["runtime_s"] = round(time.time() - t0, 1)
    (od / "run_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"wrote {od}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
