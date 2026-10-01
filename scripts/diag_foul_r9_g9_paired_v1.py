"""diag_foul_r9_g9_paired_v1.py -- foul round 9: paired game-bootstrap SE (Decision 12's second floor component)
of arm - reference on the per-game mean simulated total points, margin and FTA/FTM (same games, same seeds).
The G9 total-bias delta of two arms graded on the same games equals the delta of their per-game mean totals.

    diag_foul_r9_g9_paired_v1.py REF_TAG ARM_TAG [ARM_TAG ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/engine_v0")


def per_game(t):
    g = pd.read_parquet(R / t / "games.parquet")
    g["total"] = g["home_pts"] + g["away_pts"]
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["ftm"] = g["home_ftm"] + g["away_ftm"]
    g["fta"] = g["home_fta"] + g["away_fta"]
    g["fg_pts"] = g["total"] - g["ftm"]
    return g.groupby("game_id")[["total", "margin", "ftm", "fta", "fg_pts"]].mean()


ref = per_game(sys.argv[1])
rng = np.random.default_rng(20260930)
for t in sys.argv[2:]:
    a = per_game(t).reindex(ref.index)
    d = a - ref
    idx = [rng.integers(0, len(d), len(d)) for _ in range(500)]
    for c in d.columns:
        v = d[c].to_numpy()
        se = np.std([v[i].mean() for i in idx])
        print(f"{t} - {sys.argv[1]}  {c:7s} delta {v.mean():+.4f}  paired boot SE {se:.4f}  ({v.mean() / se:+.1f} SE)")
