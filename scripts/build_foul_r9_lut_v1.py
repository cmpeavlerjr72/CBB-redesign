#!/usr/bin/env python
"""build_foul_r9_lut_v1.py -- export the round-9 FOLD-2 TRAIN and-one fits (train 2022-2024 only)
as the lookup tables `src/cbb_sim/engine/foul_r9.py` serves. Pre-registration: possession_outcome
experiments.md section 26.

    .venv/Scripts/python.exe scripts/build_foul_r9_lut_v1.py

AO1: (class 3, period index 3, clock bucket 5) cell table. AO3: the same table + `team_b` and the
prior-season team table (`ao_team_prior_v1.parquet`; season s holds season s-1 rates).
The binning cost (table vs the fit's own test predictions) is measured on fold-2 test rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

D = Path("data/processed/models/possession_outcome/round9")
CLOCK_CUTS = np.array([120, 300, 600, 900], dtype=np.float64)


def main() -> None:
    fits = joblib.load(D / "fits_F2_ao_trip.joblib") if (D / "fits_F2_ao_trip.joblib").exists() \
        else joblib.load(D / "fits_F2_ao.joblib")
    tab = fits["AO1"]
    prior = fits["AO0"]
    lut = np.zeros((3, 3, 5))
    for c in range(3):
        for pi in range(3):
            for ci in range(5):
                lut[c, pi, ci] = tab.get(c * 100 + pi * 10 + ci, prior[c])
    np.savez_compressed(D / "lut_ao_AO1_F2.npz", lut=lut, clock_cuts=CLOCK_CUTS)
    np.savez_compressed(D / "lut_ao_AO3_F2.npz", lut=lut, clock_cuts=CLOCK_CUTS,
                        team_b=np.array(fits["AO3_b"]), team_table=np.array("ao_team_prior_v1.parquet"))
    te = pd.read_parquet(D / "preds_ao_F2_seed0.parquet")
    pi = np.where(te["period"] <= 1, 0, np.where(te["period"] == 2, 1, 2))
    ci = np.searchsorted(CLOCK_CUTS, te["sec_rem"].to_numpy(float), side="right")
    p_lut = lut[te["cls"].to_numpy(), pi, ci]
    rep = {"AO1_max_abs_diff_lut_vs_fit_on_test": float(np.abs(p_lut - te["y_ao__AO1"]).max()),
           "lut": lut.round(5).tolist(), "class_prior": {int(k): float(v) for k, v in prior.items()},
           "AO3_b": fits["AO3_b"], "note": "fold-2 TRAIN fits (2022-2024)"}
    (D / "lut_report_v1.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    sys.exit(main())
