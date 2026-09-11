#!/usr/bin/env python
"""
train_usage_v3_save_tree.py -- persist the round-3 (corrected score_diff)
LightGBM tree per class, for the Decision-10 closed-loop gate
(`scripts/run_usage_tree_closed_loop.py`).

`scripts/train_usage_v3_lgbm_refit.py` already fit and scored this exact arm
(same fixed hyperparameters/prior from round 2, corrected `score_diff`) but
did not persist the fitted booster. This script repeats ONLY that one fit per
class (the CORRECTED arm; the LEAKED comparison arm is not needed for engine
wiring) and saves it in the `usage_s1` joblib shape
(`{"arm", "event_class", "model", "features", "prior_kind", "shrink_m"}`) so
the closed-loop adapter can load it the same way `UsageAdapter` reads
`usage_s1` artifacts.

Also fits a `nostate` sibling (LGBM_ALT_FEATURES only, no score_diff/
sec_remaining/period/chance_number) for the Decision-10 refit-without arm
(L31/L33's "the closed-loop gate reports both the freeze and the refit-without
arm").

Output: `data/processed/models/usage_v3/lgbm_tree/{class}_corrected.joblib`,
`data/processed/models/usage_v3/lgbm_tree/{class}_nostate.joblib`.

Usage:
    .venv/Scripts/python.exe scripts/train_usage_v3_save_tree.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import train_usage_v1 as V1  # noqa: E402

from cbb_sim.models import usage as U  # noqa: E402

SEED = 20260910
V2_DIR = ROOT / "data/processed/models/usage_v2"
V3_DIR = ROOT / "data/processed/models/usage_v3"
OUT_DIR = V3_DIR / "lgbm_tree"
PARAMS_V2 = json.loads((V2_DIR / "usage_params_v2.json").read_text())["per_class"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["both", "corrected", "nostate"], default="both")
    ap.add_argument("--classes", nargs="*", default=list(U.EVENT_CLASSES))
    args = ap.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(V3_DIR / "events_v3_prestate.parquet")
    asof = pd.read_parquet(V2_DIR / "asof_v2_shotshooter.parquet")

    for cls in args.classes:
        pc = PARAMS_V2[cls]
        pk, m, params = pc["prior_kind"], pc["shrink_m"], pc["lgbm_params"]
        design = U.build_usage_design(events, asof, cls)
        tr, te = U.fold_slices(design)
        Ltr, ctr, Lte, kept, dropped, itr, ite = V1.lgbm_matrices(tr, te, cls, pk, m)

        if args.only in ("both", "corrected"):
            arm = U.LgbmChoiceArm(params, seed=SEED).fit(Ltr, ctr, init=itr)
            joblib.dump({"arm": "lgbm", "event_class": cls, "refit_date": None,
                        "model": arm, "features": kept, "prior_kind": pk, "shrink_m": m,
                        "score_diff_mode": "pre_outcome", "dropped_unidentified": dropped},
                       OUT_DIR / f"{cls}_corrected.joblib", compress=3)
            print(f"{cls} corrected: features={kept} n_train={len(tr):,} "
                 f"({time.time()-t0:.1f}s)", flush=True)

        if args.only in ("both", "nostate"):
            state = set(U.LGBM_STATE_FEATURES)
            keep_idx = [i for i, n in enumerate(kept) if n not in state]
            kept_ns = [kept[i] for i in keep_idx]
            arm_ns = U.LgbmChoiceArm(params, seed=SEED).fit(
                Ltr[:, keep_idx], ctr, init=itr)
            joblib.dump({"arm": "lgbm_nostate", "event_class": cls, "refit_date": None,
                        "model": arm_ns, "features": kept_ns, "prior_kind": pk, "shrink_m": m,
                        "score_diff_mode": "n/a (no state features)",
                        "dropped_unidentified": dropped},
                       OUT_DIR / f"{cls}_nostate.joblib", compress=3)
            print(f"{cls} nostate: features={kept_ns} ({time.time()-t0:.1f}s)", flush=True)

    print(f"done in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
