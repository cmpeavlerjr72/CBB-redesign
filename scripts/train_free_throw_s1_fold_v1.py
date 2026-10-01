"""train_free_throw_s1_fold_v1.py -- ONE cell of the served free_throw S1 trainer (`train_free_throw_v2_s1.py`,
NOT edited) for a chosen fold and scheme, written under a chosen root (lane D, 2026-10-01; fold-1 confirmation,
docs/models/engine/experiments.md section 2).

The served artifact is `free_throw/s1_confirm/S1_conf_aligned/F2/manifest.json`; the served trainer's plan never
fits S1_conf_aligned on fold 1. This calls the trainer's own `run_cell(scheme, fold, ...)` (design, conference
flags, cuts, fit, manifest writer: all the trainer's code) with its module constant `S1_DIR` re-pointed to --out-root,
so nothing under the served tree is written. Identity: `--fold F2` reproduces the served cell (log loss and every
booster) up to LightGBM thread count.

    .venv/Scripts/python.exe scripts/train_free_throw_s1_fold_v1.py --fold F1 --out-root data/processed/models/fold1_v1/ft
    -> <out-root>/S1_conf_aligned/F1/manifest.json  (serve with adapters.FT_S1_MANIFEST)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd  # noqa: E402

import train_free_throw_v2_s1 as T  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--scheme", default="S1_conf_aligned")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--attempts", type=Path, default=None,
                    help="default off: an attempts_v1_era-format parquet (the OT-carry sibling) instead of the stored one")
    a = ap.parse_args()
    T.S1_DIR = a.out_root if a.out_root.is_absolute() else ROOT / a.out_root
    t0 = time.time()
    T.ES.load_universe()
    att_all = pd.read_parquet(a.attempts if a.attempts else T.OUT_DIR / "attempts_v1_era.parquet")
    attempts = att_all[att_all["season"].isin(T.SEASONS)]
    design = T.FT.build_ft_design(attempts)
    conf_all = T.CF.build_conference_flags(T.SEASONS)
    firsts = T.CF.first_conference_game_dates(conf_all)
    tr, te = T.FT.fold_slices(design, a.fold)
    season = {"F1": 2024, "F2": 2025}[a.fold]
    row = T.run_cell(a.scheme, a.fold, tr, te, conf_all, firsts, season, seed=0)
    row = {k: v for k, v in row.items() if k not in ("conf4", "clean_trip")} | {"seconds": round(time.time() - t0, 1)}
    print(json.dumps(row, default=str))
    (T.S1_DIR / a.scheme / a.fold / "fold_cell_report.json").write_text(json.dumps(row, indent=1, default=str),
                                                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
