#!/usr/bin/env python
"""build_foul_cal_lut_v1.py -- export the foul-calendar round's seed-0 TRAIN fits (possession_outcome
experiments.md s32; F2 train 2022-24, F1 train 2022-23) as engine tables, on round 7's grid
(`build_foul_joint_lut_v1.grid / index`, imported unedited).

  round10cal/lut_acc_A2dbk_{F}.npz   lut (5 days buckets, 3, 5, 8, 11, 11, 3) + dss_edges
  round10cal/lut_acc_A2_{F}.npz      the control refit on the same grid (identity check vs the served table)

Binning cost and identity are measured, not assumed (report json).
    .venv/Scripts/python.exe scripts/build_foul_cal_lut_v1.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_foul_joint_lut_v1 as BL7  # noqa: E402
import train_foul_joint_v1 as FJ  # noqa: E402
from train_foul_cal_v1 import DSS_EDGES, dss_bucket  # noqa: E402

D = ROOT / "data/processed/models/possession_outcome/round10cal"
SERVED = {"F2": ROOT / "data/processed/models/possession_outcome/round7/lut_acc_A2_F2.npz",
          "F1": ROOT / "data/processed/models/fold1_v1/loop/lut_acc_A2_F1.npz"}
SHAPE = (3, 5, 8, BL7.MAXF + 1, BL7.MAXF + 1, 3)


def main() -> None:
    d = FJ.possession_design()
    d["dss_bkt"] = dss_bucket(d["days_since_start"].to_numpy()).astype("float32")
    g0 = BL7.grid(False)
    rep = {}
    for fold, sp in FJ.FOLDS.items():
        fits = joblib.load(D / f"fits_{fold}.joblib")
        te = d[d["season"].isin(sp["test"]) & (d["in_fit_window"] == 1)]
        ix = BL7.index(te)
        lut_a2 = fits["A2"].predict_proba(g0[FJ.STATE_T].to_numpy("float32"))[:, 1].reshape(SHAPE)
        np.savez_compressed(D / f"lut_acc_A2_{fold}.npz", lut=lut_a2, clock_cuts=BL7.CLOCK_CUTS,
                            margin_cuts=BL7.MARGIN_CUTS, max_fouls=BL7.MAXF)
        srv = np.load(SERVED[fold])["lut"]
        lut_b = np.empty((5,) + SHAPE)
        for b in range(5):
            X = g0[FJ.STATE_T].assign(dss_bkt=float(b))[FJ.STATE_T + ["dss_bkt"]].to_numpy("float32")
            lut_b[b] = fits["A2dbk"].predict_proba(X)[:, 1].reshape(SHAPE)
        np.savez_compressed(D / f"lut_acc_A2dbk_{fold}.npz", lut=lut_b, clock_cuts=BL7.CLOCK_CUTS,
                            margin_cuts=BL7.MARGIN_CUTS, max_fouls=BL7.MAXF, dss_edges=DSS_EDGES)
        bk = te["dss_bkt"].to_numpy().astype(int)
        y = te["y_nt"].to_numpy()
        p_model = fits["A2dbk"].predict_proba(te[FJ.STATE_T + ["dss_bkt"]].to_numpy("float32"))[:, 1]
        rep[fold] = {
            "A2_refit_vs_served_max_abs": float(np.abs(lut_a2 - srv).max()),
            "A2_refit_vs_served_mean_abs": float(np.abs(lut_a2 - srv).mean()),
            "ll_served_lut": BL7.ll(y, srv[ix]), "ll_A2_lut": BL7.ll(y, lut_a2[ix]),
            "ll_A2dbk_model": BL7.ll(y, p_model), "ll_A2dbk_lut": BL7.ll(y, lut_b[(bk,) + ix]),
            "mean_lut_by_bucket": [float(lut_b[b][ix].mean()) for b in range(5)],
            "lut_ratio_dbk_over_A2_by_bucket": [float(lut_b[b].mean() / lut_a2.mean()) for b in range(5)],
        }
    (D / "lut_report_v1.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
