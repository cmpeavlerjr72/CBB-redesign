"""build_chance_time_lut_v3.py -- chance_time addendum 1c (arm KD) tables (lane I, 2026-10-01; experiments.md s6).

Versioned sibling (v1 / v2 unchanged). Writes data/processed/models/chance_time/<fold>/lut_v3.npz = every lut_v2
array plus
  c1d_q[g, k, b, :]  1001 quantiles of the design's chance-1 `chance_elapsed_s` by start group g (0 DREB/TOV,
                     1 other), shot class k, possession-duration bin b (round-1 bins); cells < 200 rows use
                     lut_v2's c1_q[g, k]. Training rows joined to their possession as exp_chance_time_offline_v1.
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
from build_chance_time_lut_v1 import FOLDS, QS, CLASSES, dur_bin, MIN_N  # noqa: E402
sys.argv = [sys.argv[0], "unused"]  # exp_chance_time_offline_v1 reads argv[1] at import
from exp_chance_time_offline_v1 import real_rows  # noqa: E402

frames = {}
for s in sorted({x for v in FOLDS.values() for x in v}):
    assert_not_sealed([s], context="chance_time lut v3")
    m = real_rows(s)
    m = m[(m["chance_number"] == 1) & m["duration_s"].notna()]
    frames[s] = pd.DataFrame({"k": m["shot_class"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(),
                              "g": np.where(m["start_reason"].isin(["DREB", "TOV"]), 0, 1),
                              "b": dur_bin(m["duration_s"].to_numpy().astype(np.int64)),
                              "e": m["chance_elapsed_s"].to_numpy(np.float64)})
    print(s, len(frames[s]), flush=True)
for fold, seasons in FOLDS.items():
    out = ROOT / "data/processed/models/chance_time" / fold
    v2 = dict(np.load(out / "lut_v2.npz"))
    f = pd.concat([frames[s] for s in seasons], ignore_index=True)
    nb = len(v2["bin_edges"])
    c1d_q = np.zeros((2, 3, nb, len(QS)))
    c1d_n = np.zeros((2, 3, nb), dtype=np.int64)
    for g in (0, 1):
        for k in range(3):
            for b in range(nb):
                v = f.loc[(f["g"] == g) & (f["k"] == k) & (f["b"] == b), "e"].to_numpy()
                c1d_n[g, k, b] = len(v)
                c1d_q[g, k, b] = np.quantile(v, QS) if len(v) >= MIN_N else v2["c1_q"][g, k]
    np.savez(out / "lut_v3.npz", **v2, c1d_q=c1d_q, c1d_n=c1d_n)
    meta = json.loads((out / "lut_v2.json").read_text())
    meta.update({"builder_v3": "scripts/build_chance_time_lut_v3.py", "c1d_cells_thin": int((c1d_n < MIN_N).sum()),
                 "c1d_rows": int(c1d_n.sum()), "created_at_v3": pd.Timestamp.now("UTC").isoformat()})
    (out / "lut_v3.json").write_text(json.dumps(meta, indent=1))
    print(fold, "rows", int(c1d_n.sum()), "thin cells", int((c1d_n < MIN_N).sum()), flush=True)
