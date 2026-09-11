#!/usr/bin/env python
"""
train_usage_v3_lgbm_refit.py -- L4 SHOT ALLOCATION (usage), ROUND 3 targeted
refit: the adopted U5 LightGBM arm, PAIRED on corrected vs leaked `score_diff`,
same train/test split, same fixed hyperparameters, same shrinkage prior/m, same
seed, on the F1 selection fold.

Round 2 / round 2b already decided the ARM (lgbm wins every class; TOV
unconfirmed) and the shrinkage prior/m and LightGBM hyperparameters
(`data/processed/models/usage_v2/usage_params_v2.json`, "searched on 2024 only,
same rung selected on every class"). This script does not re-litigate that
choice -- it answers the narrower, round-3 question `docs/tests/
usage_state_confound_2026-09-11.md` raises: how much of the ADOPTED arm's
measured performance was manufactured by `score_diff` being post-outcome, and
does correcting it change the decision.

Because the shrinkage prior/m and the tree hyperparameters are HELD FIXED at
their round-2 values (no grid search re-run), this is much cheaper than a full
round-2-style bake-off: one LightGBM fit per class per arm (leaked / corrected)
= 10 fits total, all other arms (U1 baseline is scored too, unaffected by
score_diff) held as reported in round 2.

Artifacts: `data/processed/models/usage_v3/lgbm_refit_v3.json`,
`data/processed/models/usage_v3/events_v3_prestate.parquet` (corrected events
table, the "v2 sibling" the state-confound fix writes; nothing under `usage/`
or `usage_v2/` is touched).

Usage:
    .venv/Scripts/python.exe scripts/train_usage_v3_lgbm_refit.py
    .venv/Scripts/python.exe scripts/train_usage_v3_lgbm_refit.py --classes FGA_3
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import train_usage_v1 as V1  # noqa: E402

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import usage as U  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402

SEASONS = V1.SEASONS
SHOOTER_KEY = "shot_shooter_id"
SEED = 20260910
V2_DIR = ROOT / "data/processed/models/usage_v2"
V3_DIR = ROOT / "data/processed/models/usage_v3"
PARAMS_V2 = json.loads((V2_DIR / "usage_params_v2.json").read_text())["per_class"]


def build_events(score_diff_mode: str) -> pd.DataFrame:
    universe = ES.load_universe(require_pbp_complete=True)
    rim = ES.rim_override_for_version("v2")
    per_season = {}
    for s in SEASONS:
        raw = U.build_usage_events(s, universe, rim_override_max_ft=rim,
                                   shooter_key=SHOOTER_KEY,
                                   score_diff_mode=score_diff_mode)
        per_season[s] = U.usable_events(raw)
    return pd.concat([per_season[s] for s in SEASONS], ignore_index=True)


def score_one(cls: str, tr: pd.DataFrame, te: pd.DataFrame, pk: str, m: float,
             params: dict, log) -> dict:
    t0 = time.time()
    Ltr, ctr, Lte, kept, dropped, itr, ite = V1.lgbm_matrices(tr, te, cls, pk, m)
    arm = U.LgbmChoiceArm(params, seed=SEED).fit(Ltr, ctr, init=itr)
    p = arm.predict_proba(Lte, init=ite)
    driver = U.shrunk_rate(te, cls, pk, m)
    sc = U.score_arm(te, p, driver)
    roles_te = U.assign_roles(te, pk, m)
    ps_te = U.build_profiles(te, driver, roles_te)
    prole_te = U.profile_player_roles(ps_te)
    sc["game_level"] = U.game_level_check(te, p=p, n_draw=40, seed=SEED,
                                          player_role=prole_te)
    imp = dict(zip(kept, arm.clf_.feature_importances_.tolist(), strict=False))
    tot = sum(imp.values()) or 1.0
    sc["feature_importance_pct"] = {k: round(100.0 * v / tot, 2) for k, v in imp.items()}
    sc["state_feature_importance_pct"] = round(
        sum(v for k, v in sc["feature_importance_pct"].items()
            if k in U.LGBM_STATE_FEATURES), 2)
    sc["dropped_unidentified"] = dropped
    sc["seconds"] = round(time.time() - t0, 1)
    log(f"    {cls}: ll={sc['log_loss']:.6f} calib={sc['calib_worst_gap_pp']:.3f}pp "
        f"top3_gap={sc['game_level']['top3_gap_pp']:+.3f}pp "
        f"state_imp={sc['state_feature_importance_pct']:.1f}% "
        f"({sc['seconds']}s)")
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=list(U.EVENT_CLASSES))
    args = ap.parse_args()
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    V3_DIR.mkdir(parents=True, exist_ok=True)

    log("building LEAKED (own_row) events -- reproduces round 2's events_v2_shotshooter")
    events_leaked = build_events("own_row")
    log(f"  {len(events_leaked):,} rows ({time.time()-t0:.1f}s)")
    log("building CORRECTED (pre_outcome) events -- round 3 data fix")
    events_fixed = build_events("pre_outcome")
    events_fixed.to_parquet(V3_DIR / "events_v3_prestate.parquet", index=False)
    log(f"  {len(events_fixed):,} rows ({time.time()-t0:.1f}s)")

    asof = pd.read_parquet(V2_DIR / "asof_v2_shotshooter.parquet")

    results_path = V3_DIR / "lgbm_refit_v3.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for cls in args.classes:
        pc = PARAMS_V2[cls]
        pk, m, params = pc["prior_kind"], pc["shrink_m"], pc["lgbm_params"]
        log(f"  {cls}: prior={pk} m={m} params={params} (fixed from round 2)")

        d_leaked = U.build_usage_design(events_leaked, asof, cls)
        tr_l, te_l = U.fold_slices(d_leaked)
        d_fixed = U.build_usage_design(events_fixed, asof, cls)
        tr_f, te_f = U.fold_slices(d_fixed)

        sc_leaked = score_one(cls, tr_l, te_l, pk, m, params, log)
        sc_fixed = score_one(cls, tr_f, te_f, pk, m, params, log)
        results[cls] = {"prior_kind": pk, "shrink_m": m, "lgbm_params": params,
                        "n_train": int(len(tr_l)), "n_test": int(len(te_l)),
                        "leaked_own_row": sc_leaked, "corrected_pre_outcome": sc_fixed,
                        "log_loss_delta_corrected_minus_leaked": round(
                            sc_fixed["log_loss"] - sc_leaked["log_loss"], 6)}
        (V3_DIR / "lgbm_refit_v3.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")
        (V3_DIR / "train_log_v3_lgbm_refit.txt").write_text(
            "\n".join(log_lines), encoding="utf-8")

    log(f"\ndone in {time.time()-t0:.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
