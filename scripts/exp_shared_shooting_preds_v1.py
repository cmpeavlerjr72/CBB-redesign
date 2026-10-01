#!/usr/bin/env python
"""
exp_shared_shooting_preds_v1.py -- per-shot held-out fg_make predictions for the
shared-shooting-latent measurement (lane B, overnight 2026-09-30).

    .venv/Scripts/python.exe scripts/exp_shared_shooting_preds_v1.py

For every FGA in seasons 2023, 2024, 2025 the make probability of the SERVED
fg_make spec (round 4 arm B1: TEAM_FEATURES + R2_SAFE_STATE +
shooter_shrunk_dev_c, frozen ladder params, S1 monthly refits), each one fitted
ONLY on data before the shot's month:

  * 2025 (fold-2 test): the served artifacts themselves
    (`data/processed/models/fg_make/round4/B1/<class>_<refit>.joblib`), read,
    never refitted.
  * 2024 (fold-1 test): the same spec refitted on F1 (train 2022-2023 plus the
    test season's earlier months), seed 0.
  * 2023: the same spec refitted on 2022 plus 2023's earlier months, seed 0
    (the out-of-sample residual source for the training-side variance fits).

LightGBM is built directly (LGBM_BASE + frozen params, random_state=0) with
n_jobs = 4 so the lane's core cap holds; `FG.LgbmArm` hard-codes n_jobs=-1.
Nothing served is written: output goes to results/shared_shooting/ (gitignored).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

FG_DIR = Path("data/processed/models/fg_make")
B1_DIR = FG_DIR / "round4" / "B1"
OUT = Path("results/shared_shooting")
FEATS = list(FG.TEAM_FEATURES) + list(FG.R2_SAFE_STATE) + ["shooter_shrunk_dev_c"]
KEEP = ["game_id", "season", "game_date", "neutral_site", "period", "seconds_remaining",
        "off_team_id", "def_team_id", "offense_is_home", "shot_class", "class_key", "y",
        "chance_number", "chance_elapsed_s", "is_transition_f"]


def fit_lgbm(X, y, params):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(random_state=0, n_jobs=4, **{**FG.LGBM_BASE, **params})
    clf.fit(X, y)
    return clf


def p_make(clf, X):
    pr = clf.predict_proba(X)
    j = list(clf.classes_).index(1)
    return pr[:, j]


def refit_season(design, season, params, t0):
    """S1 monthly schedule for `season`, training on every earlier season."""
    out = []
    for c in FG.SHOT_CLASSES:
        tr = design[(design["season"] < season) & (design["shot_class"] == c)]
        te = design[(design["season"] == season) & (design["shot_class"] == c)]
        te_dates = pd.to_datetime(te["game_date"])
        cuts = PO.month_boundaries(te_dates)
        p = np.full(len(te), np.nan)
        for k, cut in enumerate(cuts):
            nxt = cuts[k + 1] if k + 1 < len(cuts) else None
            seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
            if not seg.any():
                continue
            before = (te_dates < cut).to_numpy()
            rows = pd.concat([tr[FEATS + ["y"]], te.loc[before, FEATS + ["y"]]], ignore_index=True)
            clf = fit_lgbm(FG.design_matrix(rows, FEATS), rows["y"].to_numpy(), params[c])
            p[seg] = p_make(clf, FG.design_matrix(te.loc[seg], FEATS))
            print(f"[{time.time()-t0:7.1f}s] {season} {c} refit {cut.date()} n_train {len(rows)} "
                  f"scored {int(seg.sum())}", flush=True)
        assert np.isfinite(p).all(), "S1 partition left rows unscored"
        o = te[KEEP].copy()
        o["p"] = p
        o["pred_source"] = f"refit_B1_S1_train_lt_{season}"
        out.append(o)
    return pd.concat(out, ignore_index=True)


def served_2025(design, t0):
    out = []
    for c in FG.SHOT_CLASSES:
        man = json.loads((B1_DIR / f"manifest_{c}.json").read_text(encoding="utf-8"))
        assert man["features"] == FEATS, (man["features"], FEATS)
        te = design[(design["season"] == 2025) & (design["shot_class"] == c)]
        te_dates = pd.to_datetime(te["game_date"])
        arts = man["artifacts"]
        cuts = [pd.Timestamp(a["refit_date"]) for a in arts]
        p = np.full(len(te), np.nan)
        for k, a in enumerate(arts):
            cut = cuts[k]
            nxt = cuts[k + 1] if k + 1 < len(cuts) else None
            seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
            if k == 0:
                seg = seg | (te_dates < cut).to_numpy()
            obj = joblib.load(B1_DIR / a["path"])
            p[seg] = obj["model"].predict_proba(FG.design_matrix(te.loc[seg], FEATS))[:, FG.CLASS_INDEX["MAKE"]]
        assert np.isfinite(p).all()
        o = te[KEEP].copy()
        o["p"] = p
        o["pred_source"] = "served_round4_B1"
        out.append(o)
        print(f"[{time.time()-t0:7.1f}s] 2025 {c} served predictions {len(te)}", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    design = pd.read_parquet(FG_DIR / "design_v2_shotshooter.parquet")
    extra = pd.read_parquet(FG_DIR / "design_v4_extra_v2.parquet", columns=["shooter_shrunk_dev_c"])
    design["shooter_shrunk_dev_c"] = extra["shooter_shrunk_dev_c"].to_numpy()
    design = design[design["season"].isin([2022, 2023, 2024, 2025])]
    ladder = json.loads((FG_DIR / "lgbm_ladder_v2.json").read_text(encoding="utf-8"))
    params = {c: dict(v) for c, v in ladder["frozen_params"].items()}
    print(f"[{time.time()-t0:7.1f}s] design {design.shape}", flush=True)
    parts = [served_2025(design, t0)]
    for s in (2024, 2023):
        parts.append(refit_season(design, s, params, t0))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT / "preds_v1.parquet", index=False)
    for s, g in out.groupby("season"):
        print(s, len(g), "mean y", round(g["y"].mean(), 4), "mean p", round(g["p"].mean(), 4))
    print(f"[{time.time()-t0:7.1f}s] wrote {OUT/'preds_v1.parquet'}", flush=True)


if __name__ == "__main__":
    main()
