"""diag_foul_r9_parity_v1.py -- foul round 9: is a LOCAL run bit-identical to the box run on the
rows they share? Compares games.parquet of two result dirs on the (game_id, seed) rows common
to both, every common column.

    diag_foul_r9_parity_v1.py LOCAL_DIR BOX_DIR
"""
import sys

import pandas as pd

a = pd.read_parquet(f"{sys.argv[1]}/games.parquet")
b = pd.read_parquet(f"{sys.argv[2]}/games.parquet")
k = ["game_id", "seed"]
m = a.merge(b, on=k, suffixes=("_a", "_b"), how="inner")
cols = [c for c in a.columns if c not in k and c in b.columns]
bad = [c for c in cols if not (m[f"{c}_a"].to_numpy() == m[f"{c}_b"].to_numpy()).all()]
print(f"rows local {len(a)}, box {len(b)}, common {len(m)}; columns compared {len(cols)}; "
      f"mismatching columns: {bad if bad else 'NONE (bit-identical)'}")
