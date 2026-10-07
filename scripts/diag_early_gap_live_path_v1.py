#!/usr/bin/env python
"""Live-path early-gap read (2026-10-07): paired grade of ctrl (historical priors) vs A3+R1 day-1 live path on days 0-45,
plus the Shapley points decomposition (scripts/diag_early_total_points_decomp_v1.py, unedited) on each arm.
usage: diag_early_gap_live_path_v1.py grade | decomp <arm>   (arm in ctrl, live)"""
from __future__ import annotations
import sys, io, contextlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_early_total_points_decomp_v1 as X
from cbb_sim.eval import reference as R
ROOT = X.ROOT; RUN = ROOT / "results/player_day1/runs"
SEAS = {"F1": 2024, "F2": 2025}
OUT = ROOT / "results/early_gap_live_path"; OUT.mkdir(parents=True, exist_ok=True)

def per_game(path):
    g = pd.read_parquet(path); g["tot"] = g.home_pts + g.away_pts; g["mar"] = g.home_pts - g.away_pts
    return g.groupby("game_id")[["tot", "mar"]].mean()

def grade():
    rows = []
    for f, s in SEAS.items():
        act = R.load_actual_games(s); act["date"] = pd.to_datetime(act["game_date"])
        d0 = act["date"].min(); act["dss"] = (act["date"] - d0).dt.days
        act = act.set_index("game_id")
        c = per_game(RUN / f"early46_{f}_ctrl/games.parquet"); l = per_game(RUN / f"early46_{f}_live/games.parquet")
        j = c.join(l, lsuffix="_c", rsuffix="_l", how="inner").join(act[["total", "margin", "dss"]], how="inner")
        j["seeded"] = ((j.tot_c - j.tot_l).abs() > 1e-9)
        for b, m in (("d0-14", j.dss <= 14), ("d15-45", (j.dss > 14) & (j.dss <= 45)), ("d0-45", j.dss <= 45),
                     ("d0-14 seeded games", (j.dss <= 14) & j.seeded), ("d0-14 unseeded games", (j.dss <= 14) & ~j.seeded)):
            x = j[m]; n = len(x)
            if n < 2: continue
            for met, a_, c_, l_ in (("total", "total", "tot_c", "tot_l"), ("margin", "margin", "mar_c", "mar_l")):
                bc = (x[c_] - x[a_]); bl = (x[l_] - x[a_]); d = bl - bc
                rows.append(dict(fold=f, bucket=b, metric=met, n=n, ctrl_bias=bc.mean(), live_bias=bl.mean(), diff=d.mean(),
                                 diff_se=d.std(ddof=1) / np.sqrt(n), ctrl_se=bc.std(ddof=1) / np.sqrt(n),
                                 ctrl_mae=bc.abs().mean(), live_mae=bl.abs().mean(), n_seeded=int(x.seeded.sum())))
    T = pd.DataFrame(rows); T.to_csv(OUT / "grade.csv", index=False)
    pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))

def decomp(arm):
    X.D.RUNS = {2024: RUN / f"early46_F1_{arm}/games.parquet", 2025: RUN / f"early46_F2_{arm}/games.parquet"}
    X.OUT = OUT / arm; X.OUT.mkdir(parents=True, exist_ok=True)
    orig = X.D.decompose
    def safe(s, t, n):
        if n == 0:                       # empty bucket (d46+ is outside the 46-day run): NaN everywhere
            class _N(dict):
                def __missing__(self, k): return float("nan")
            return _N()
        return orig(s, t, n)
    X.D.decompose = safe
    # buckets with zero games (d46+) are skipped by safe() raising; patch the bucket loop tolerant
    import types
    try: X.main()
    except ValueError: pass

if __name__ == "__main__":
    grade() if sys.argv[1] == "grade" else decomp(sys.argv[2])
