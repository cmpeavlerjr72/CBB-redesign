#!/usr/bin/env python
"""
diag_shot_block_seed_noise_v1.py -- SUPPLEMENTARY (not pre-registered): the
SEED noise of the served engine's mean-total correlation and G5 total SD ratio
on the 500-game subset, from four served runs with disjoint seed ranges
(0-24, 1000-1024, 2000-2024, 3000-3024), set beside the drawn-flag arms'
changes. The game bootstrap holds seeds fixed, so it cannot see this noise.

    .venv/Scripts/python.exe scripts/diag_shot_block_seed_noise_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import contract as C  # noqa: E402
from cbb_sim.eval import gates as G  # noqa: E402

SERVED = ["po4b_R_s25", "po4b_R_s25_floor", "sb_R_s25_off2000", "sb_R_s25_off3000"]
ARMS = ["sb_K2O_s25", "sb_K2_s25", "sb_K2Onoteam_s25"]


def read(tag: str) -> dict:
    eng = C.load_engine_results(str(ROOT / "results/engine_v0" / tag))
    s, _ = G.build_grading_frame(eng.games, 2025)
    ok = s["total"] > 0
    res = s["total"] - s["sim_total_mean"]
    return {"corr": float(np.corrcoef(s["total"], s["sim_total_mean"])[0, 1]),
            "corr_played": float(np.corrcoef(s.loc[ok, "total"], s.loc[ok, "sim_total_mean"])[0, 1]),
            "g5_total_ratio": float(s["sim_total_sd"].mean() / res.std()),
            "total_mean": float(s["sim_total_mean"].mean())}


def main() -> int:
    v = {t: read(t) for t in SERVED + ARMS}
    sd = {k: float(np.std([v[t][k] for t in SERVED], ddof=1)) for k in v[SERVED[0]]}
    ref = v[SERVED[0]]
    out = {"runs": v, "served_seed_sd": sd,
           "arm_minus_served_seed0": {a: {k: v[a][k] - ref[k] for k in ref} for a in ARMS},
           "arm_in_seed_sd_units": {a: {k: (v[a][k] - ref[k]) / sd[k] for k in ("corr", "corr_played",
                                                                                  "g5_total_ratio")}
                                    for a in ARMS}}
    (ROOT / "results/shot_block_round2/seed_noise_v1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
