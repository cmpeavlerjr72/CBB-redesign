"""build_chance_time_lut_v2.py -- chance_time addendum 1b (arm K) tables (lane I, 2026-10-01; experiments.md s3).

Versioned sibling of build_chance_time_lut_v1.py (unchanged). Writes data/processed/models/chance_time/<fold>/lut_v2.npz
= every lut_v1 array (bit-identical, recomputed by v1's own function) plus
  c1_q[g, k, :]  1001 quantiles of the fg_make design's chance-1 `chance_elapsed_s`, start group g
                 (0: DREB/TOV, 1: other), shot class k (0 rim, 1 jump2, 2 three). Training seasons only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from build_chance_time_lut_v1 import FOLDS, QS, CLASSES  # noqa: E402

for fold, seasons in FOLDS.items():
    assert_not_sealed(seasons, context="chance_time lut v2")
    d = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet",
                        columns=["season", "shot_class", "chance_number", "chance_elapsed_s", "chance_start_reason"])
    d = d[d["season"].isin(seasons) & (d["chance_number"] == 1)]
    g = np.where(d["chance_start_reason"].isin(["DREB", "TOV"]), 0, 1)
    c1_q = np.zeros((2, 3, len(QS)))
    c1_n = np.zeros((2, 3), dtype=np.int64)
    for gi in (0, 1):
        for k, c in enumerate(CLASSES):
            v = d.loc[(g == gi) & (d["shot_class"] == c).to_numpy(), "chance_elapsed_s"].to_numpy(np.float64)
            c1_n[gi, k] = len(v)
            c1_q[gi, k] = np.quantile(v, QS)
    out = ROOT / "data/processed/models/chance_time" / fold
    v1 = dict(np.load(out / "lut_v1.npz"))
    np.savez(out / "lut_v2.npz", **v1, c1_q=c1_q, c1_n=c1_n)
    meta = json.loads((out / "lut_v1.json").read_text())
    meta.update({"builder_v2": "scripts/build_chance_time_lut_v2.py", "c1_n": c1_n.tolist(),
                 "c1_median": c1_q[:, :, 500].tolist(), "created_at_v2": pd.Timestamp.now("UTC").isoformat()})
    (out / "lut_v2.json").write_text(json.dumps(meta, indent=1))
    print(fold, c1_n.tolist(), c1_q[:, :, [100, 250, 500, 750, 900]].round(1).tolist(), flush=True)
