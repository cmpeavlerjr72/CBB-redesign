#!/usr/bin/env python
"""build_ft_exposure_serving_v1.py -- S1_conf_aligned F2 serving artifacts for a free_throw section-18 arm.

    .venv/Scripts/python.exe scripts/build_ft_exposure_serving_v1.py <ARM>      # X1 | X2 | X3

The served trainer `train_free_throw_v2_s1.py` is imported unedited; its module feature constant is swapped in this
process only (as `exp_ft_scorediff_v1.py build` does). Artifacts go to
data/processed/models/free_throw/s1_scorediff/<ARM>/S1_conf_aligned/F2/, which the existing default-off loader
`ENGINE_FT_SCORE=<ARM>` serves (adapters.FreeThrowAdapter.load). Nothing served changes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.chdir(ROOT)
import pandas as pd  # noqa: E402

import train_free_throw_v2_s1 as T  # noqa: E402
import train_free_throw_v4_exposure as X  # noqa: E402

FT = T.FT
ART = ROOT / "data/processed/models/free_throw/s1_scorediff"
_orig_dm = FT.design_matrix


def main(arm: str) -> None:
    feats = tuple(X.ARMS[arm])
    FT.FT_FEATURES = feats
    FT.design_matrix = lambda d, features=None: _orig_dm(d, tuple(features) if features is not None else feats)
    T.S1_DIR = ART / arm
    T.ES.load_universe()
    att = pd.read_parquet(T.OUT_DIR / "attempts_v1_era.parquet")
    att = att[att["season"].isin(T.SEASONS)]
    d = X.add_exposure(FT.build_ft_design(att))
    conf_all = T.CF.build_conference_flags(T.SEASONS)
    firsts = T.CF.first_conference_game_dates(conf_all)
    tr, te = FT.fold_slices(d, "F2")
    row = T.run_cell("S1_conf_aligned", "F2", tr, te, conf_all, firsts, 2025, seed=0)
    row = {k: v for k, v in row.items() if k not in ("conf4", "clean_trip")}
    row["features"] = list(feats)
    (ART / arm / "build_report.json").write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
    print(json.dumps(row, default=str)[:1500])


if __name__ == "__main__":
    main(sys.argv[1])
