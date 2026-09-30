#!/usr/bin/env python
"""
diag_shot_block_default_parity_v1.py -- proof (ii) of shot_block section 5.5:
with ENGINE_SHOT_BLOCK unset, the edited engine reproduces the 09-18 reference
run `po4b_R_s25` bit-for-bit on a handful of its games (all 25 seeds each).

    .venv/Scripts/python.exe scripts/diag_shot_block_default_parity_v1.py --n-games 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=6)
    ap.add_argument("--ref", default="po4b_R_s25")
    a = ap.parse_args()
    import run_po4b_closed_loop as RUN
    os.environ.pop("ENGINE_SHOT_BLOCK", None)
    for k, v in {**RUN.PINNED_SUBMODELS, "ENGINE_EVENT": "round2_s1"}.items():
        os.environ[k] = v
    from cbb_sim.engine import loop as L
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(RUN.INPUT_DIR, "F2_2025")
    ad = Adapters.load(inp, "F2", 2025)
    rows = RUN.subset_rows(inp.games)[:: max(1, 500 // a.n_games)][: a.n_games]
    seeds = np.arange(25)
    gi = np.repeat(rows, len(seeds))
    sd = np.tile(seeds, len(rows))
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=True)
    ref = pd.read_parquet(ROOT / "results/engine_v0" / a.ref / "games.parquet")
    new = res.games
    keys = ["game_id", "seed"]
    ref = ref[ref["game_id"].isin(new["game_id"].unique())]
    cols = [c for c in ref.columns if c in new.columns]
    m = new[cols].merge(ref[cols], on=keys, suffixes=("_n", "_r"))
    diffs = {}
    for c in cols:
        if c in keys:
            continue
        x, y = m[f"{c}_n"], m[f"{c}_r"]
        eq = (x.astype(str) == y.astype(str)) if x.dtype == object else np.isclose(
            x.astype("float64"), y.astype("float64"), rtol=0, atol=0, equal_nan=True)
        if not bool(np.all(eq)):
            diffs[c] = int((~np.asarray(eq)).sum())
    out = {"ref": a.ref, "n_games": int(len(rows)), "n_rows_compared": int(len(m)),
           "n_rows_new": int(len(new)), "columns_compared": len(cols) - 2,
           "columns_differing": diffs, "bit_identical": not diffs and len(m) == len(new)}
    print(json.dumps(out, indent=1))
    Path(ROOT / "results/shot_block_round2").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/shot_block_round2/default_parity_po4b_v1.json").write_text(json.dumps(out, indent=1))
    return 0 if out["bit_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
