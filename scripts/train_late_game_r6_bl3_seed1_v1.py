"""train_late_game_r6_bl3_seed1_v1.py -- late-game ROUND 6 (experiments.md 13.2): the spec-identical seed-1
refit of round 1's `B / L3_gates` first-chance LightGBM on fold 2, for the reseed floor. Round 2's fit code
(`train_late_game_r2_v1.event_cells` data path and `PO.fit_arm("lgbm", ...)`), seed = 1, n_jobs = 3.
Writes predictions on round 1's window test rows (first chances; other rows NaN). Computes no metric.

Output: data/processed/models/late_game/round6/bl3_F2_seed1.npy
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "3"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

import train_late_game_r2_v1 as T2                                   # noqa: E402
from cbb_sim.data.seal import assert_not_sealed                      # noqa: E402
from cbb_sim.models import possession_outcome as PO                  # noqa: E402

OUT = ROOT / "data/processed/models/late_game/round6"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feats = T2.LGD.bundle("L3_gates", "first")
    need = {"season", "game_id", "game_date", "period", "poss_index", "chance_number", "population", "y",
            "is_regulation", "in_window"} | set(feats)
    d = pd.read_parquet(T2.DESIGN_EV, columns=sorted(need))
    d = d[d["is_regulation"].to_numpy()].reset_index(drop=True)
    tr = d[d["season"].isin(PO.FOLDS["F2"]["train"])]
    assert_not_sealed(tr, context="F2 train slice")
    tr = tr[tr["population"] == "first"]
    model = PO.fit_arm("lgbm", tr, feats, seed=1)
    te = pd.read_parquet(T2.R1 / "round1" / "window_test_F2.parquet")
    first = (te["population"] == "first").to_numpy()
    p = np.full((len(te), len(PO.CLASSES)), np.nan, dtype="float32")
    p[first] = model.predict_proba(np.ascontiguousarray(te.loc[first, feats].to_numpy(dtype="float32")))
    np.save(OUT / "bl3_F2_seed1.npy", p)
    print(f"BL3 F2 seed 1: {len(tr):,} train rows, {int(first.sum()):,} test first chances, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
