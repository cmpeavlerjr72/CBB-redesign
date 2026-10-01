#!/usr/bin/env python
"""
train_free_throw_v2_site.py -- Lane G (overnight 2026-09-30): FT% site-term
bake-off. Pre-registration: `docs/models/free_throw/experiments.md` section 11
(committed e5dd38c BEFORE this ran).

    .venv/Scripts/python.exe scripts/train_free_throw_v2_site.py --arms FT0,FT1,FT2 --folds F2,F1 --seeds 0
    .venv/Scripts/python.exe scripts/train_free_throw_v2_site.py --arms FT0 --folds F2,F1 --seeds 1

Arms (S1_monthly calendar for all, a stated cost deviation from the served
S1_conf_aligned):
    FT0  served FT_FEATURES
    FT1  + site_home, site_away  (cat)
    FT2  + site_signed           (signed)

Writes predictions only, results/home_site/ft/preds_<arm>_<fold>_s<seed>.parquet
in the shared grading schema of `grade_home_site_v1.py`. Sibling of
`train_free_throw_v2_s1.py`, which is not edited. Nothing goes to an engine
directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

NJ = int(os.environ.get("CBB_NJOBS", "2"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(NJ)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

FT.LgbmArm.PARAMS = {**FT.LgbmArm.PARAMS, "n_jobs": NJ}
OUT = Path("results/home_site/ft")
SEASONS = [2022, 2023, 2024, 2025]
BASE = list(FT.FT_FEATURES)
ARMS = {"FT0": BASE, "FT1": BASE + ["site_home", "site_away"], "FT2": BASE + ["site_signed"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="FT0,FT1,FT2")
    ap.add_argument("--folds", default="F2,F1")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    t0 = time.time()
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="free_throw site round")
    OUT.mkdir(parents=True, exist_ok=True)
    ES.load_universe()
    att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
    att = att[att["season"].isin(SEASONS)]
    d = FT.build_ft_design(att)
    neu = d["neutral_site"].fillna(False).astype(bool).to_numpy()
    home = d["shooter_is_home"].fillna(False).astype(bool).to_numpy()
    d["site_home"] = (~neu & home).astype("float32")
    d["site_away"] = (~neu & ~home).astype("float32")
    d["site_signed"] = (d["site_home"] - d["site_away"]).astype("float32")
    print(f"[{time.time() - t0:5.0f}s] design {d.shape}; site shares "
          f"home {d['site_home'].mean():.3f} away {d['site_away'].mean():.3f}", flush=True)
    for fold in a.folds.split(","):
        tr, te = FT.fold_slices(d, fold)
        te_dates = pd.to_datetime(te["game_date"])
        cuts = PO.month_boundaries(te_dates)
        for arm in a.arms.split(","):
            feats = ARMS[arm]
            for seed in (int(s) for s in a.seeds.split(",")):
                t1 = time.time()
                cols = [*feats, "y"]
                p = np.zeros(len(te))
                for k, cut in enumerate(cuts):
                    nxt = cuts[k + 1] if k + 1 < len(cuts) else None
                    seg = ((te_dates >= cut) if nxt is None
                           else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
                    if not seg.any():
                        continue
                    before = (te_dates < cut).to_numpy()
                    prior = te.loc[before, cols]
                    rows = tr[cols] if not len(prior) else pd.concat([tr[cols], prior], ignore_index=True)
                    m = FT.LgbmArm(seed=seed).fit(FT.design_matrix(rows, tuple(feats)), rows["y"].to_numpy())
                    p[seg] = m.predict_proba(FT.design_matrix(te.loc[seg], tuple(feats)))[:, FT.CLASS_INDEX["MAKE"]]
                if (p == 0).any():
                    raise AssertionError("calendar is not a cover")
                pm = np.column_stack([1 - p, p]) if FT.CLASS_INDEX["MAKE"] == 1 else np.column_stack([p, 1 - p])
                s = FT.score(te, pm)
                out = pd.DataFrame({
                    "game_id": te["game_id"].to_numpy(), "off_id": te["team_id"].to_numpy(),
                    "def_id": te["opp_id"].to_numpy(),
                    "site": te["site_signed"].to_numpy().astype("int8"),
                    "y": te["y"].to_numpy().astype("float64"), "p": p, "w": 1.0,
                    "game_date": pd.to_datetime(te["game_date"]).to_numpy(),
                    "driver": te["shooter_ft_asof"].to_numpy(),
                    "driver_defined": (te["shooter_fta_asof"].to_numpy() > 0)})
                path = OUT / f"preds_{arm}_{fold}_s{seed}.parquet"
                out.to_parquet(path, index=False)
                meta = {"arm": arm, "fold": fold, "seed": seed, "features": feats,
                        "n_refits": len(cuts), "log_loss": s["log_loss"],
                        "calib_pass": bool(s["calib_pass"]), "calib_worst_gap_pp": s["calib_worst_gap_pp"],
                        "resp_pass": bool(s["resp_pass"]),
                        "slope_ratio": s["responsiveness"]["shooter_ft_asof->MAKE"]["slope_ratio"],
                        "fit_s": round(time.time() - t1, 1)}
                (OUT / f"meta_{arm}_{fold}_s{seed}.json").write_text(json.dumps(meta, indent=1, default=float))
                print(f"[{time.time() - t0:5.0f}s] {arm} {fold} s{seed}: ll {s['log_loss']:.6f} "
                      f"calib {s['calib_worst_gap_pp']} resp {s['resp_pass']} ({meta['fit_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
