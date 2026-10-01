"""ops_r9_slice_runs_v1.py -- foul round 9 (lane C): copy the first N seeds of a box 500-game run into a sibling
results dir, so a local 25-seed arm can be graded and paired on exactly the same (game, seed) rows.

    ops_r9_slice_runs_v1.py SRC_TAG DST_TAG N_SEEDS
"""
import json
import sys
from pathlib import Path

import pandas as pd

R = Path("results/engine_v0")
src, dst, n = R / sys.argv[1], R / sys.argv[2], int(sys.argv[3])
dst.mkdir(parents=True, exist_ok=False)
g = pd.read_parquet(src / "games.parquet")
seeds = sorted(g["seed"].unique())[:n]
g[g["seed"].isin(seeds)].to_parquet(dst / "games.parquet", index=False)
if (src / "players.parquet").exists():
    p = pd.read_parquet(src / "players.parquet")
    p[p["seed"].isin(seeds)].to_parquet(dst / "players.parquet", index=False)
m = json.loads((src / "run_meta.json").read_text(encoding="utf-8"))
m["seeds"] = [int(s) for s in seeds]
m["n_seeds"] = len(seeds)
m["sliced_from"] = str(src)
m["n_rows"] = int(g["seed"].isin(seeds).sum())
(dst / "run_meta.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
print(dst, len(seeds), "seeds", seeds[0], "-", seeds[-1])
