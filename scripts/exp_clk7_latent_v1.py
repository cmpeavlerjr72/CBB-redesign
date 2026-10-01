"""exp_clk7_latent_v1.py -- clock round 7 (experiments.md section 32/33): refit the
served v5b B1 game-latent sigma on a round-7 law, exactly as lane B's
`exp_clk6_r6_arms_v1.py --step latent` does for L2: the UNEDITED
`exp_clk5b_mean_consistent.py --fit-only`, with only its design path and the
schedule it reads redirected.

    .venv/Scripts/python.exe scripts/exp_clk7_latent_v1.py --arm A2
writes data/processed/models/clock/r7_<arm>/v5b_bakeoff/v5b_bakeoff_report.json
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CK = ROOT / "data/processed/models/clock"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A2"], required=True)
    a = ap.parse_args()
    from cbb_sim.models import clock_r7  # noqa: F401  (unpickling)
    spec = importlib.util.spec_from_file_location("exp_clk5b_r7", ROOT / "scripts" / "exp_clk5b_mean_consistent.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["exp_clk5b_r7"] = m
    spec.loader.exec_module(m)
    m.r5.DESIGN = CK / "r6_L2" / "design_v2.parquet"
    served_load = m.r5.load_schedule
    mode = f"v3c_r7{a.arm}_P3_s1"
    m.r5.load_schedule = lambda _mode: served_load(mode)
    sys.argv = [sys.argv[0], "--out", str((CK / f"r7_{a.arm}" / "v5b_bakeoff").relative_to(ROOT)), "--fit-only"]
    m.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
