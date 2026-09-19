#!/usr/bin/env python
"""
build_foul_accrual_lut_v1.py -- export round-6 arm `F5` as a dense LOOKUP TABLE
the engine can index without a live model call (CLAUDE.md, sim loop rules).

Pre-registration: `docs/models/possession_outcome/experiments.md` section 13 /
15; offline result section 17.

    .venv/Scripts/python.exe scripts/build_foul_accrual_lut_v1.py --fold F2

The table is indexed by
`(period_idx, clock_bucket, margin_bucket, def_team_fouls, off_team_fouls, site)`
= (3, 5, 8, 11, 11, 3) = 43,560 cells, every one of them a probability that the
possession produces a non-shooting foul on the DEFENCE that awards no trip.

TWO LIMITATIONS, STATED IN THE ARTIFACT ITSELF, not discovered later:

1. `F5`'s two as-of team-rate features (`def_foul_c`, `off_drawn_c`) are
   evaluated at **0.0, the league mean**, because the engine's `team_static`
   block does not carry them and adding them is an `EngineInputs` rebuild. The
   served table therefore reproduces `F5`'s CLOCK-and-STATE law and NOT its
   team responsiveness (offline prior-quintile slope 0.252 -> 0 by construction).
2. The engine charges every silent foul to the DEFENCE and has no offensive-foul
   mechanism. **No mechanism is invented here.** `--target` says which quantity
   the table serves under that one-sided attribution:
     `def`            the defence-side non-trip foul only (0.0769/possession) --
                      the literal event the served scalar draws;
     `engine_attrib`  BOTH non-trip channels (0.0769 + 0.0222 = 0.0991) charged
                      to the defence, which is what the served constant 0.123346
                      was implicitly doing and what the engine's team-foul
                      accounting actually needs.
   The closed loop prices both; neither is a tuned scalar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_foul_accrual_v1 import F3C, FOLDS, ROUND_DIR, build_features  # noqa: E402

CLOCK_CUTS = np.array([120, 300, 600, 900], dtype=np.float64)
MARGIN_CUTS = np.array([-15, -7, -3, 0, 3, 7, 15], dtype=np.float64)
CLOCK_MID = np.array([60.0, 210.0, 450.0, 750.0, 1050.0])
MARGIN_MID = np.array([-22.0, -11.0, -5.0, -1.5, 1.5, 5.0, 11.0, 22.0])
MAX_FOULS = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="def", choices=["def", "engine_attrib"],
                    help="'def' = the defence-side non-trip foul only; "
                         "'engine_attrib' = BOTH non-trip channels charged to the "
                         "defence, which is the attribution loop.py actually has")
    a = ap.parse_args()
    t0 = time.time()
    df = build_features(pd.read_parquet(ROUND_DIR / "foul_accrual_poss_v1.parquet"))
    df["y_def"] = ((df["def_silent"] + (df["off_silent"] if a.target == "engine_attrib" else 0))
                   >= 1).astype("int8")
    tr = df[df["season"].isin(FOLDS[a.fold]["train"]) & (df["in_fit_window"] == 1)]

    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05,
                           num_leaves=31, min_child_samples=500, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=a.seed,
                           n_jobs=1, verbose=-1)
    m.fit(tr[F3C].to_numpy(dtype="float32"), tr["y_def"].to_numpy())

    shape = (3, len(CLOCK_MID), len(MARGIN_MID), MAX_FOULS + 1, MAX_FOULS + 1, 3)
    rows = []
    for pi in range(3):
        period = 1.0 if pi == 0 else 2.0 if pi == 1 else 3.0
        for ci, sec in enumerate(CLOCK_MID):
            for mi, mg in enumerate(MARGIN_MID):
                for dfl in range(MAX_FOULS + 1):
                    for ofl in range(MAX_FOULS + 1):
                        for si in range(3):
                            gs = (1200 - sec) if pi == 0 else (2400 - sec) if pi == 1 else (2700 - sec)
                            rows.append({
                                "period": period, "sec_rem": sec, "margin": mg,
                                "abs_margin": abs(mg), "game_seconds": gs,
                                "def_team_fouls": float(dfl), "off_team_fouls": float(ofl),
                                "site_home": 1.0 if si == 1 else 0.0,
                                "site_away": 1.0 if si == 2 else 0.0,
                                "is_ot": 1.0 if pi == 2 else 0.0,
                                "off_in_bonus": 1.0 if dfl >= 6 else 0.0,
                                "def_in_bonus": 1.0 if ofl >= 6 else 0.0,
                                "off_in_double_bonus": 1.0 if dfl >= 9 else 0.0,
                                "def_foul_c": 0.0, "off_drawn_c": 0.0})
    grid = pd.DataFrame(rows)
    p = m.predict_proba(grid[F3C].to_numpy(dtype="float32"))[:, 1]
    lut = p.reshape(shape)

    # binning cost, measured not assumed: the LUT scored on the fold's own test
    # rows against the unbinned model
    te = df[df["season"].isin(FOLDS[a.fold]["test"]) & (df["in_fit_window"] == 1)]
    p_model = m.predict_proba(te[F3C].to_numpy(dtype="float32"))[:, 1]
    pi = np.where(te["period"].to_numpy() <= 1, 0, np.where(te["period"].to_numpy() == 2, 1, 2))
    ci = np.searchsorted(CLOCK_CUTS, te["sec_rem"].to_numpy(), side="right")
    mi = np.searchsorted(MARGIN_CUTS, te["margin"].to_numpy(), side="right")
    dfl = np.clip(te["def_team_fouls"].to_numpy().astype(int), 0, MAX_FOULS)
    ofl = np.clip(te["off_team_fouls"].to_numpy().astype(int), 0, MAX_FOULS)
    si = np.where(te["site_home"].to_numpy() > 0, 1,
                  np.where(te["site_away"].to_numpy() > 0, 2, 0))
    p_lut = lut[pi, ci, mi, dfl, ofl, si]
    y = te["y_def"].to_numpy()

    def ll(pv):
        pv = np.clip(pv, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(pv) + (1 - y) * np.log(1 - pv)))

    out = ROUND_DIR / (f"foul_accrual_lut_F5_{a.fold}.npz" if a.target == "def" else f"foul_accrual_lut_F5e_{a.fold}.npz")
    np.savez_compressed(out, lut=lut, clock_cuts=CLOCK_CUTS, margin_cuts=MARGIN_CUTS,
                        max_fouls=MAX_FOULS)
    rep = {"fold": a.fold, "seed": a.seed, "cells": int(lut.size),
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "logloss_model": round(ll(p_model), 8), "logloss_lut": round(ll(p_lut), 8),
           "binning_cost": round(ll(p_lut) - ll(p_model), 8),
           "mean_p_lut": round(float(p_lut.mean()), 6),
           "mean_actual": round(float(y.mean()), 6),
           "served_constant": 0.123346, "seconds": round(time.time() - t0, 1),
           "team_rate_features_at_league_mean": True,
           "target": a.target,
           "offensive_foul_channel": ("served inside the defence-side draw (engine attribution)"
                                      if a.target == "engine_attrib" else
                                      "not served; the engine has no mechanism and none is invented")}
    (ROUND_DIR / ((f"foul_accrual_lut_F5_{a.fold}.json") if a.target == "def" else f"foul_accrual_lut_F5e_{a.fold}.json")).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
