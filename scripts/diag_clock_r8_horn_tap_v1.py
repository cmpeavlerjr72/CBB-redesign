"""diag_clock_r8_horn_tap_v1.py -- possessions STARTED by time-left bucket at the end of each half,
sim (one clock mode) vs actual (lane L's line, PM message 10:3x EDT 2026-10-01; clock round 8 doc).

Lane H, 2026-10-01. In-process tap: the clock adapter is wrapped so every `draw` call (one per
possession start) records (period, seconds left at the start); the wrapper delegates every call and
attribute (including the K arms' `cont_mode` / `redraw_end` / `draw_cont`) unchanged, so the simulation
is bit-identical to a plain run. `simulate_chunk` runs one seed at a time over the sample's games,
exactly as `run_engine.py`'s worker does (same inputs dir, same adapter load, threads pinned to 1).

    ENGINE_CLOCK=<mode> .venv/Scripts/python.exe scripts/diag_clock_r8_horn_tap_v1.py --label K2 --seeds 10
Writes results/clock_r8/horn_<label>.json: per half, possessions per game starting in each bucket
(0-3, 3-6, 6-10, 10-20, 20-35 s left), sim and actual (possessions_v4, the same games).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EDGES = [0, 3, 6, 10, 20, 35]
LABELS = ["0-3", "3-6", "6-10", "10-20", "20-35"]
SAMPLE = ROOT / "data/processed/truth/stride500_verified_v1_F2_2025.parquet"


class ClockTap:
    def __init__(self, real, rec, idx):
        self._r, self._rec, self._i = real, rec, idx

    def __getattr__(self, k):
        return getattr(self._r, k)

    def draw(self, team, x, u, gidx=None, keys=None):
        if keys is not None:
            d = self._r.draw(team, x, u, gidx, keys)
        elif gidx is not None:
            d = self._r.draw(team, x, u, gidx)
        else:
            d = self._r.draw(team, x, u)
        self._rec.append(np.column_stack([np.asarray(gidx if gidx is not None else np.full(len(x), -1)),
                                          x[:, self._i["period"]], x[:, self._i["seconds_remaining"]]]))
        return d


def bucket_counts(per, sec):
    out = {}
    for p in (1, 2):
        m = per == p
        out[f"H{p}"] = np.histogram(sec[m], bins=EDGES)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()
    from cbb_sim.engine import loop as L
    from cbb_sim.engine.adapters import STATE_INDEX, Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(str(ROOT / "data/processed/models/engine_v3"), "F2_2025")
    ad = Adapters.load(inp, "F2", 2025)
    clock_mode = ad.flags.get("ENGINE_CLOCK")
    rec: list = []
    ad.clock = ClockTap(ad.clock, rec, STATE_INDEX)
    ids = pd.read_parquet(SAMPLE)["game_id"].astype(int).tolist()
    gpos = pd.Series(np.arange(inp.n_games), index=inp.games["game_id"].astype(int).to_numpy())
    rows = np.sort(gpos.reindex(ids).dropna().astype(int).to_numpy())
    t0 = time.time()
    sims = {k: np.zeros(len(LABELS)) for k in ("H1", "H2")}
    n_sim = 0
    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        rec.clear()
        L.simulate_chunk(inp, ad, rows, np.full(len(rows), s, dtype=np.int64), keep_players=False)
        r = np.vstack(rec)
        c = bucket_counts(r[:, 1], r[:, 2])
        for k in sims:
            sims[k] += c[k]
        n_sim += len(rows)
        print(f"seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    gids = inp.games["game_id"].astype(int).to_numpy()[rows]
    p = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2025.parquet",
                        columns=["game_id", "period", "start_clock"])
    p = p[p["game_id"].isin(set(gids))]
    act = bucket_counts(p["period"].to_numpy(), p["start_clock"].to_numpy())
    n_act = p["game_id"].nunique()
    rep = {"label": a.label, "clock": clock_mode, "seeds": [a.seed_offset, a.seed_offset + a.seeds - 1],
           "n_games": int(len(rows)), "n_actual_games": int(n_act), "buckets_s": LABELS,
           "sim_per_game": {k: (v / n_sim).round(4).tolist() for k, v in sims.items()},
           "actual_per_game": {k: (v / n_act).round(4).tolist() for k, v in act.items()},
           "sim_counts": {k: v.tolist() for k, v in sims.items()}, "n_sim_games": n_sim}
    out = ROOT / f"results/clock_r8/horn_{a.label}.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("clock", "sim_per_game", "actual_per_game")}))


if __name__ == "__main__":
    main()
