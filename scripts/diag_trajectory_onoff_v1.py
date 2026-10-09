"""diag_trajectory_onoff_v1.py -- proofs (b) and (c) for the trajectory side-channel.

Runs the same F2/2025 (game, seed) set twice in-process, trajectory OFF then ON (CBB_TRAJECTORY=1, so the env switch and
the writer path are both exercised), and reports
  (b) max abs diff over every numeric column of the games and players frames (must be 0.0),
  (c) per game-seed: replaying the trajectory's per-possession points reproduces the final score exactly.
    .venv/Scripts/python.exe scripts/diag_trajectory_onoff_v1.py [--games 60] [--seeds 4]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
for k in [k for k in os.environ if k.startswith("ENGINE_")]:
    os.environ.pop(k)


def maxdiff(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, int]:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return float("inf"), -1
    worst = 0.0
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in "fiub":
            worst = max(worst, float(np.max(np.abs(x.astype(float) - y.astype(float)))) if len(x) else 0.0)
        elif not (x == y).all():
            return float("inf"), -1
    return worst, len(a)


def replay_check(traj: pd.DataFrame, games: pd.DataFrame) -> dict:
    t = traj.copy()
    t["h"] = np.where(t.off_side == 0, t.points, 0)
    t["a"] = np.where(t.off_side == 1, t.points, 0)
    g = t.groupby(["game_id", "seed"]).agg(h=("h", "sum"), a=("a", "sum"), last_h=("home_score", "last"),
                                           last_a=("away_score", "last"), n=("poss_idx", "size")).reset_index()
    m = g.merge(games[["game_id", "seed", "home_pts", "away_pts"]], on=["game_id", "seed"], how="outer", indicator=True)
    both = m[m._merge == "both"]
    bad = both[(both.h != both.home_pts) | (both.a != both.away_pts) | (both.last_h != both.home_pts)
               | (both.last_a != both.away_pts)]
    return {"game_seeds_games": len(games), "game_seeds_traj": len(g), "matched": len(both), "mismatch": len(bad),
            "missing_in_traj": int((m._merge == "right_only").sum()), "extra_in_traj": int((m._merge == "left_only").sum()),
            "rows": len(t), "rows_per_game_seed": len(t) / max(len(g), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--input-dir", default="data/processed/models/engine_v3")
    a = ap.parse_args()
    from cbb_sim.engine import loop as L
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(a.input_dir, "F2_2025")
    ad = Adapters.load(inp, "F2", 2025)
    gi = np.repeat(np.arange(a.games, dtype=np.int64), a.seeds)
    sd = np.tile(np.arange(a.seeds, dtype=np.int64), a.games)
    os.environ.pop("CBB_TRAJECTORY", None)
    off = L.simulate_chunk(inp, ad, gi, sd, keep_players=True)
    assert off.trajectory is None
    os.environ["CBB_TRAJECTORY"] = "1"
    on = L.simulate_chunk(inp, ad, gi, sd, keep_players=True)
    gd, gn = maxdiff(off.games, on.games)
    pdf, pn = maxdiff(off.players, on.players)
    print(f"(b) games  rows={gn} max abs diff={gd}")
    print(f"(b) players rows={pn} max abs diff={pdf}")
    print(f"    diag identical: {off.diag == on.diag}  n_possessions {off.n_possessions} vs {on.n_possessions}")
    rc = replay_check(on.trajectory, on.games)
    print("(c) replay:", rc)
    ok = gd == 0.0 and pdf == 0.0 and rc["mismatch"] == 0 and rc["missing_in_traj"] == 0 and off.diag == on.diag
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
