"""exp_clk10_latent_env_v1.py -- clock round 10 (experiments.md section 43): refit the B1
game-latent sigma on the round-9 env arm E1 FIRST-CHANCE law (K2 x env AFT). Versioned sibling of `exp_clk8_latent_v2.py`
(unchanged; identical body, only the schedule and output point at r9_env_<arm>): the UNEDITED `exp_clk5b_mean_consistent.py --fit-only`, with its design path and
schedule redirected; as the design is read, `days_since_start` takes the engine-input definition
and `duration_s` is replaced by the first-chance duration d1 (chances_v4), the quantity the K
arm's law describes and the latent scales.

    .venv/Scripts/python.exe scripts/exp_clk10_latent_env_v1.py --arm E1
writes data/processed/models/clock/r9_env_<arm>/v5b_bakeoff/v5b_bakeoff_report.json
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
sys.path.insert(0, str(ROOT / "scripts"))
CK = ROOT / "data/processed/models/clock"


class _PdShim:
    def __init__(self, pd, design_path, fix):
        self._pd, self._path, self._fix = pd, Path(design_path), fix

    def __getattr__(self, k):
        return getattr(self._pd, k)

    def read_parquet(self, path, *args, **kw):
        df = self._pd.read_parquet(path, *args, **kw)
        if Path(path) == self._path and "game_date" in df.columns and "season" in df.columns:
            df = self._fix(df)
        return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["E1", "E2", "E3"], required=True)
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    from cbb_sim.models import clock_r8
    import train_clock_r8_chance_v1 as TK
    first, _ = TK.chance_tables(TK.T.ALL_SEASONS)
    f = first.reset_index()
    f["_p"] = f["period"].astype("int64"); f["_i"] = f["poss_index"].astype("int64")
    f = f[["game_id", "_p", "_i", "d1", "end1_code"]]

    def fix(df):
        df["days_since_start"] = clock_r8.season_day(df)
        k = df[["game_id", "period", "poss_index"]].copy()
        k["_p"] = k["period"].astype("int64"); k["_i"] = k["poss_index"].astype("int64")
        m = k[["game_id", "_p", "_i"]].merge(f, on=["game_id", "_p", "_i"], how="left")
        d1 = m["d1"].to_numpy()
        ok = np.isfinite(d1)
        df = df.loc[ok].copy()
        df["duration_s"] = np.clip(d1[ok].astype("int64"), 0, 90)
        df["end1_code"] = m["end1_code"].to_numpy()[ok].astype("int64")
        return df

    spec = importlib.util.spec_from_file_location("exp_clk5b_r10e", ROOT / "scripts" / "exp_clk5b_mean_consistent.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["exp_clk5b_r10e"] = m
    spec.loader.exec_module(m)
    m.r5.DESIGN = CK / "r6_L2" / "design_v2.parquet"
    m.pd = _PdShim(pd, m.r5.DESIGN, fix)
    served_load = m.r5.load_schedule
    mode = f"v3c_r8K2_env{a.arm}_P3_s1"
    m.r5.load_schedule = lambda _mode: served_load(mode)
    sys.argv = [sys.argv[0], "--out", str((CK / f"r9_env_{a.arm}" / "v5b_bakeoff").relative_to(ROOT)), "--fit-only"]
    m.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
