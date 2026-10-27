"""train_rebound_v2_loc_v1.py -- shot-location round 1, Read A rebound channel (experiments.md 1.3).

The served rebound model class and feature bundle (`lgbm` / `C_plus_state`, `cbb_sim.models.rebound` fixed PARAMS,
the served design columns as reproduced by rebound round 3's `design_round3.parquet`, whose A0B0C0 cell reproduces
round 1's S0 log loss exactly) with location columns of the MISS added. NOTHING existing is edited; the module's
`feature_set`, `LgbmArm.PARAMS`, `fold_slices` and `score` are used unchanged (only `n_jobs` is pinned in this
process, as `train_rebound_v2_s1.py` did).

  --arm R | A1 | A2   R  = C_plus_state (control; under --seed 1 it is R_seed, the floor)
                      A1 = R + band one-hots of the miss (b_short2, b_long2, b_corner3, b_wing3, b_top3)
                      A2 = R + d (ft) and theta (deg) of the miss
Population: rebound-opportunity rows whose miss is FGA_jump2 or FGA_3 AND whose miss is located (rim misses and
free-throw misses excluded; unlocated misses are never imputed), joined to the fg design row by
`build_shot_location_bands_v1.py` (`rebound_loc_cols_v1.parquet`).

Refit calendar: S0 (one static fit per fold on the fold's training seasons). DISCLOSED: the served calendar is
S1_weekly (23 refits per cell at 189-631 s per fit, measured in rebound s7-s11), which does not fit the round's
budget; every arm here shares the S0 calendar, so the arms differ ONLY in the added columns.

Extra output (no effect on any fit): `cf_<fold>_s<seed>.parquet`, the prediction on ALL jump2 / three miss rows of
the test season (located or not): R plain; A1 with the band one-hots set to each band of the miss class in turn.

    .venv/Scripts/python.exe scripts/train_rebound_v2_loc_v1.py --arm A1 --fold F2 --seed 0 --n-jobs 4 \
        --out-dir data/processed/models/shot_location/round1/rebound/A1_s0_F2
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

from cbb_sim.models import rebound as RB  # noqa: E402

DESIGN = ROOT / "data/processed/models/rebound/round3/design_round3.parquet"
LOC = ROOT / "data/processed/models/shot_location/round1/rebound_loc_cols_v1.parquet"
BANDS = {"jump2": ("short2", "long2"), "three": ("corner3", "wing3", "top3")}
BAND_COLS = ["b_short2", "b_long2", "b_corner3", "b_wing3", "b_top3"]
ARM_FEATURES = {"R": [], "A1": BAND_COLS, "A2": ["loc_d", "loc_theta"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_FEATURES), required=True)
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise SystemExit(f"{a.out_dir} not empty")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(a.n_jobs)
    RB.LgbmArm.PARAMS = {**RB.LgbmArm.PARAMS, "n_jobs": a.n_jobs}
    feats_r = RB.feature_set("C_plus_state")
    feats = feats_r + ARM_FEATURES[a.arm]
    d = pd.read_parquet(DESIGN, columns=sorted(set(["game_id", "season", "game_date", "off_team_id", "def_team_id",
                                                    "miss_type", "y", *feats_r])))
    loc = pd.read_parquet(LOC)
    if len(loc) != len(d) or not (loc["miss_type"].to_numpy() == d["miss_type"].to_numpy()).all():
        raise SystemExit("rebound loc cols misaligned with the design")
    d["located"] = loc["located"].to_numpy().astype(bool)
    d["fg_row"] = loc["fg_row"].to_numpy()
    d["band"] = loc["band"].to_numpy()
    for b in ("short2", "long2", "corner3", "wing3", "top3"):
        d[f"b_{b}"] = (d["band"] == b).astype("float32")
    d["loc_d"] = loc["d"].to_numpy(dtype="float32")
    d["loc_theta"] = loc["theta"].to_numpy(dtype="float32")
    jt = d["miss_type"].isin(["jump2", "three"]).to_numpy()
    pop = d[jt & d["located"].to_numpy()]
    tr, te = RB.fold_slices(pop, a.fold)
    print(f"[{time.time() - t0:6.1f}s] arm {a.arm} fold {a.fold}: train {len(tr):,} test {len(te):,}", flush=True)
    mdl = RB.fit_arm("lgbm", tr, feats, seed=a.seed)
    p = RB.predict_arm("lgbm", mdl, te, feats)
    sc = RB.score(te, p)
    print(f"[{time.time() - t0:6.1f}s] ll {sc['log_loss']:.6f} calib {sc['calib_worst_gap_pp']:.3f}pp", flush=True)
    out = pd.DataFrame({"row": te.index.to_numpy(), "fg_row": te["fg_row"].to_numpy(),
                        "game_id": te["game_id"].to_numpy(), "miss_type": te["miss_type"].to_numpy(),
                        "y": te["y"].to_numpy()})
    for i, c in enumerate(RB.CLASSES):
        out[f"p_{c}"] = p[:, i]
    out.to_parquet(a.out_dir / f"preds_{a.fold}_s{a.seed}.parquet", index=False)
    # counterfactual predictions on ALL jump2 / three miss rows of the test season
    _, te_all = RB.fold_slices(d[jt], a.fold)
    cf = pd.DataFrame({"row": te_all.index.to_numpy(), "fg_row": te_all["fg_row"].to_numpy(),
                       "miss_type": te_all["miss_type"].to_numpy(), "located": te_all["located"].to_numpy()})
    if a.arm == "R":
        cf["p_OREB"] = RB.predict_arm("lgbm", mdl, te_all, feats)[:, RB.CLASS_INDEX["OREB"]]
    elif a.arm == "A1":
        for mt, bands in BANDS.items():
            m = (te_all["miss_type"] == mt).to_numpy()
            for b in bands:
                z = te_all.loc[m].copy()
                for c in BAND_COLS:
                    z[c] = np.float32(1.0 if c == f"b_{b}" else 0.0)
                col = cf.get(f"p_OREB_{b}")
                v = np.full(len(cf), np.nan) if col is None else col.to_numpy()
                v[m] = RB.predict_arm("lgbm", mdl, z, feats)[:, RB.CLASS_INDEX["OREB"]]
                cf[f"p_OREB_{b}"] = v
    if a.arm in ("R", "A1"):
        cf.to_parquet(a.out_dir / f"cf_{a.fold}_s{a.seed}.parquet", index=False)
    rep = {"arm": a.arm, "fold": a.fold, "seed": a.seed, "features": feats, "scheme": "S0 (disclosed; served S1_weekly)",
           "n_train": int(len(tr)), "n_test": int(len(te)), "log_loss": sc["log_loss"],
           "calib_worst_gap_pp": sc["calib_worst_gap_pp"], "calib_pass": sc["calib_pass"],
           "runtime_s": round(time.time() - t0, 1)}
    (a.out_dir / "run_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"[{time.time() - t0:6.1f}s] wrote {a.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
