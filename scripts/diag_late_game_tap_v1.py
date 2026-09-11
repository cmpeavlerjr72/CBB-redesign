"""diag_late_game_tap_v1.py -- opt-in per-possession LATE-GAME state tap.

Lane: late-game regime (2026-09-11).  MEASUREMENT ONLY.  NOTHING IN
`src/cbb_sim/` IS CHANGED BY THIS SCRIPT and no served default is touched.
It follows the pattern `diag_pace_efficiency_poss_log_v1.py` established: the
clock and event adapters are wrapped IN THIS PROCESS ONLY, one seed at a time
so `game_index` identifies the simulation row, and the wrappers delegate to
the real adapter and return its value unchanged, so the simulation is
bit-identical to a normal run.

Flags are pinned to those of the 200-seed run A
(results/engine_v0/F2_2025_s200_v1_clockv3c_A/run_meta.json) so the tap is
comparable with the gate read and the variance/OT diagnostic, NOT to the
current served default (ENGINE_CLOCK=v5b_glat_pmean).  Recorded per
possession: drawn duration, period, seconds_remaining, offense score_diff,
in_bonus, and the event model's class probabilities.

Usage: diag_late_game_tap_v1.py [n_games] [n_seeds] [out_dir]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FLAGS = {
    "ENGINE_EVENT": "round2_s1", "ENGINE_CLOCK": "v3c_srfloor_P3_s1",
    "ENGINE_ROTATION": "reference", "ENGINE_FG3": "decision8",
    "ENGINE_FG_MAKE": "round4_B1", "ENGINE_REBOUND": "s1_weekly",
    "ENGINE_FREE_THROW": "s1_conf_aligned", "ENGINE_ROTATION_SCHEME": "s1",
    "ENGINE_INPUTS_VERSION": "v2",
}
for k, v in FLAGS.items():
    os.environ[k] = v
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "6"

NGAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NSEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUTD = Path(sys.argv[3] if len(sys.argv) > 3 else "results/late_game/tap_s5")
OUTD.mkdir(parents=True, exist_ok=True)

from cbb_sim.engine import loop as L                                 # noqa: E402
from cbb_sim.engine.adapters import STATE_INDEX, Adapters            # noqa: E402
from cbb_sim.engine.inputs import EngineInputs                       # noqa: E402
from cbb_sim.models import possession_outcome as PO                  # noqa: E402

I = STATE_INDEX
inp = EngineInputs.load(str(ROOT / "data/processed/models/engine"), "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
REC = {"clock": [], "event": []}
SEED = [0]


class ClockTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def draw(self, team, x, u, gidx=None):
        dur = (self._r.draw(team, x, u, gidx) if gidx is not None
               else self._r.draw(team, x, u))
        g = gidx if gidx is not None else np.full(len(x), -1, dtype=np.int64)
        REC["clock"].append(np.column_stack([
            np.full(len(x), SEED[0]), g, np.asarray(dur, float),
            x[:, I["period"]], x[:, I["seconds_remaining"]],
            x[:, I["score_diff"]], x[:, I["in_bonus"]]]))
        return dur


class EventTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def predict(self, team, state, is_first, gidx, off):
        p = self._r.predict(team, state, is_first, gidx, off)
        REC["event"].append(np.column_stack([
            np.full(len(state), SEED[0]), gidx, np.asarray(is_first, float),
            state[:, I["period"]], state[:, I["seconds_remaining"]],
            state[:, I["score_diff"]], state[:, I["in_bonus"]],
            state[:, I["chance_number"]], np.asarray(p, float)]))
        return p


ad.clock = ClockTap(ad.clock)
ad.event = EventTap(ad.event)

rng = np.random.default_rng(11)
rows = np.sort(rng.choice(len(inp.games), size=NGAMES, replace=False)).astype(np.int64)
print("=== late-game tap: %d games x %d seeds (F2/2025, run-A flags) ===" % (NGAMES, NSEEDS))
t0 = time.time()
games = []
for s in range(NSEEDS):
    SEED[0] = s
    res = L.simulate_chunk(inp, ad, rows, np.full(NGAMES, s, dtype=np.int64),
                           keep_players=False)
    games.append(res.games.assign(gidx=rows))
    print("  seed %d done  %.1f s" % (s, time.time() - t0))

G = pd.concat(games, ignore_index=True)
clk = pd.DataFrame(np.vstack(REC["clock"]),
                   columns=["seed", "gidx", "dur", "period", "sec", "sd", "bonus"])
ev = pd.DataFrame(np.vstack(REC["event"]),
                  columns=["seed", "gidx", "is_first", "period", "sec", "sd",
                           "bonus", "chance"] + list(PO.CLASSES))
G.to_parquet(OUTD / "games.parquet")
clk.to_parquet(OUTD / "clock_tap.parquet")
ev.to_parquet(OUTD / "event_tap.parquet")
print("  possessions %d  chances %d  sims %d  in %.1f s"
      % (len(clk), len(ev), len(G), time.time() - t0))
