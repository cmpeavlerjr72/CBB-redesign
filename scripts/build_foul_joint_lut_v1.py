#!/usr/bin/env python
"""
build_foul_joint_lut_v1.py -- export the round-7 FOLD-2 fits (train 2022-2024
only) as the lookup tables `src/cbb_sim/engine/foul_joint.py` serves.
Pre-registration: possession_outcome experiments.md section 20.3.

    .venv/Scripts/python.exe scripts/build_foul_joint_lut_v1.py

Accrual tables: (period idx 3, clock bucket 5, margin bucket 8, def fouls 0..10,
off fouls 0..10, site 3) -- round 6's grid and cuts, the state the ENGINE's
definition (fouls before the possession's own fouls). The offence table adds
`ended_tov` (0/1). Trip tables: logit offsets (half 2, live def fouls 0..10,
foul-differential bucket 3) for FT_trip_bonus and FT_trip_shooting.
Binning cost is measured on fold-2 test rows, not assumed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_foul_joint_v1 import STATE_T, possession_design  # noqa: E402

D = ROOT / "data/processed/models/possession_outcome/round7"
CLOCK_CUTS = np.array([120, 300, 600, 900], dtype=np.float64)
MARGIN_CUTS = np.array([-15, -7, -3, 0, 3, 7, 15], dtype=np.float64)
CLOCK_MID = np.array([60.0, 210.0, 450.0, 750.0, 1050.0])
MARGIN_MID = np.array([-22.0, -11.0, -5.0, -1.5, 1.5, 5.0, 11.0, 22.0])
MAXF = 10


def grid(with_tov: bool) -> pd.DataFrame:
    rows = []
    for pi in range(3):
        period = 1.0 if pi == 0 else 2.0 if pi == 1 else 3.0
        for sec in CLOCK_MID:
            for mg in MARGIN_MID:
                for dfl in range(MAXF + 1):
                    for ofl in range(MAXF + 1):
                        for si in range(3):
                            for tv in ((0, 1) if with_tov else (0,)):
                                gs = (1200 - sec) if pi == 0 else (2400 - sec) if pi == 1 else (2700 - sec)
                                rows.append((period, sec, mg, abs(mg), gs, dfl, ofl,
                                             float(dfl >= 6), float(ofl >= 6), float(dfl >= 9),
                                             float(si == 1), float(si == 2), float(pi == 2), tv))
    return pd.DataFrame(rows, columns=STATE_T + ["ended_tov"])


def index(te: pd.DataFrame):
    pi = np.where(te["period"].to_numpy() <= 1, 0, np.where(te["period"].to_numpy() == 2, 1, 2))
    ci = np.searchsorted(CLOCK_CUTS, te["sec_rem"].to_numpy(), side="right")
    mi = np.searchsorted(MARGIN_CUTS, te["margin"].to_numpy(), side="right")
    dfl = np.clip(te["def_f_t"].to_numpy().astype(int), 0, MAXF)
    ofl = np.clip(te["off_f_t"].to_numpy().astype(int), 0, MAXF)
    si = np.where(te["site_home"].to_numpy() > 0, 1, np.where(te["site_away"].to_numpy() > 0, 2, 0))
    return pi, ci, mi, dfl, ofl, si


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    fits = joblib.load(D / "fits_F2_poss_trip.joblib")
    tfits = joblib.load(D / "fits_F2_trip.joblib")
    d = possession_design()
    te = d[(d["season"] == 2025) & (d["in_fit_window"] == 1)]
    rep = {}
    shape = (3, 5, 8, MAXF + 1, MAXF + 1, 3)
    g0, g1 = grid(False), grid(True)
    for arm, y in (("A2", "y_nt"), ("D2", "y_dnt")):
        m = fits[arm]
        lut = m.predict_proba(g0[STATE_T].to_numpy("float32"))[:, 1].reshape(shape)
        ix = index(te)
        rep[arm] = {"logloss_model": ll(te[y].to_numpy(), m.predict_proba(te[STATE_T].to_numpy("float32"))[:, 1]),
                    "logloss_lut": ll(te[y].to_numpy(), lut[ix]), "mean_lut_on_test": float(lut[ix].mean()),
                    "mean_actual": float(te[y].mean())}
        np.savez_compressed(D / f"lut_acc_{arm}_F2.npz", lut=lut, clock_cuts=CLOCK_CUTS,
                            margin_cuts=MARGIN_CUTS, max_fouls=MAXF)
    m = fits["O2"]
    feats = STATE_T + ["ended_tov"]
    lut = m.predict_proba(g1[feats].to_numpy("float32"))[:, 1].reshape(shape + (2,))
    ix = index(te) + (te["ended_tov"].to_numpy().astype(int),)
    rep["O2"] = {"logloss_model": ll(te["y_off"].to_numpy(), m.predict_proba(te[feats].to_numpy("float32"))[:, 1]),
                 "logloss_lut": ll(te["y_off"].to_numpy(), lut[ix]), "mean_lut_on_test": float(lut[ix].mean()),
                 "mean_actual": float(te["y_off"].mean())}
    np.savez_compressed(D / "lut_off_O2_F2.npz", lut=lut, clock_cuts=CLOCK_CUTS,
                        margin_cuts=MARGIN_CUTS, max_fouls=MAXF)
    # A1: half x def-count table broadcast onto the same grid
    t = fits["A1_table"]
    tab = dict(zip(t["keys"], t["p"]))
    lut = np.full(shape, t["prior"])
    for pi in range(3):
        h = 0 if pi == 0 else 1
        for dfl in range(MAXF + 1):
            lut[pi, :, :, dfl, :, :] = tab.get(h * 100 + dfl, t["prior"])
    rep["A1"] = {"logloss_lut": ll(te["y_nt"].to_numpy(), lut[index(te)]),
                 "mean_lut_on_test": float(lut[index(te)].mean())}
    np.savez_compressed(D / "lut_acc_A1_F2.npz", lut=lut, clock_cuts=CLOCK_CUTS,
                        margin_cuts=MARGIN_CUTS, max_fouls=MAXF)
    # trip offsets
    for arm in ("T2c", "T2lab"):
        arrs = {}
        for y, nm in (("y_bonus", "delta_bonus"), ("y_shoot", "delta_shoot")):
            dl = tfits[f"delta_{arm}_{y}"]
            a = np.zeros((2, MAXF + 1, 3))
            for k, v in dl.items():
                h, rem = divmod(int(k), 1000)
                dc, db = divmod(rem, 10)
                a[h, dc, db] = v
            arrs[nm] = a
        np.savez_compressed(D / f"lut_trip_{arm}_F2.npz", max_fouls=MAXF, **arrs)
        rep[arm] = {k: {"min": float(v.min()), "max": float(v.max())} for k, v in arrs.items()}
    rep["note"] = "fold-2 TRAIN fits (2022-2024); scored on 2025 fit-window rows for binning cost only"
    (D / "lut_report_v1.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
