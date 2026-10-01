"""exp_clk8_latent_v1.py -- clock round 8 (experiments.md section 36): refit the served v5b
B1 game-latent sigma on a round-8 law, exactly as `exp_clk7_latent_v1.py` does for A2: the
UNEDITED `exp_clk5b_mean_consistent.py --fit-only`, with only its design path and the
schedule it reads redirected. One addition: the design's `days_since_start` is replaced by
the engine-input definition (`clock_r8.season_day`; 673 rows in 5 games differ, the
builder's median fills) as the design is read, so the latent is fitted on the feature the
engine serves.

    .venv/Scripts/python.exe scripts/exp_clk8_latent_v1.py --arm M2D
writes data/processed/models/clock/r8_<arm>/v5b_bakeoff/v5b_bakeoff_report.json
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


class _PdShim:
    """pandas, except that reading the clock design recomputes days_since_start."""

    def __init__(self, pd, design_path, season_day):
        self._pd, self._path, self._sd = pd, Path(design_path), season_day

    def __getattr__(self, k):
        return getattr(self._pd, k)

    def read_parquet(self, path, *args, **kw):
        df = self._pd.read_parquet(path, *args, **kw)
        if Path(path) == self._path and "game_date" in df.columns and "season" in df.columns:
            df["days_since_start"] = self._sd(df)
        return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["M2D"], required=True)
    a = ap.parse_args()
    import pandas as pd
    from cbb_sim.models import clock_r8  # noqa: F401  (unpickling)
    spec = importlib.util.spec_from_file_location("exp_clk5b_r8", ROOT / "scripts" / "exp_clk5b_mean_consistent.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["exp_clk5b_r8"] = m
    spec.loader.exec_module(m)
    m.r5.DESIGN = CK / "r6_L2" / "design_v2.parquet"
    m.pd = _PdShim(pd, m.r5.DESIGN, clock_r8.season_day)
    served_load = m.r5.load_schedule
    mode = f"v3c_r8{a.arm}_P3_s1"
    m.r5.load_schedule = lambda _mode: served_load(mode)
    sys.argv = [sys.argv[0], "--out", str((CK / f"r8_{a.arm}" / "v5b_bakeoff").relative_to(ROOT)), "--fit-only"]
    m.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
