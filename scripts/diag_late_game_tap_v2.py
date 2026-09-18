"""diag_late_game_tap_v2.py -- the late-game state tap against the SERVED stack.

Lane: late-game regime, round 1 (2026-09-18).  MEASUREMENT ONLY.  NOTHING IN
`src/cbb_sim/` IS CHANGED BY THIS SCRIPT and no served default is touched.

This is the versioned sibling of `diag_late_game_tap_v1.py`, which pinned run
A's flags (`ENGINE_CLOCK=v3c_srfloor_P3_s1`).  `docs/models/late_game/
experiments.md` section 1.0 makes a re-read against the SERVED default
(`ENGINE_CLOCK=v5b_glat_pmean`, adopted 2026-09-11) a BLOCKING precondition on
the round.  v2 therefore pins NOTHING: it lets `Adapters.load` resolve every
default and records the resolved flag set beside the tap, so the file itself
says what stack produced it.

Two mechanical differences from v1, both forced by the served clock:

  1. `LatentClockAdapter.draw` takes a fifth argument (the per-simulation
     stream keys) and declares `wants_sim_keys`.  The wrapper forwards *args
     unchanged rather than naming them, so it is signature-agnostic and
     delegates bit-identically under either adapter.
  2. The tap is SHARDED.  Each process takes its own disjoint slice of the
     game universe (`--shard k/n`) and its own seed offset, so four processes
     under the 4-worker compute cap cover four times the simulations without
     two of them writing the same file.  `diag_late_game_compare_v2.py`
     concatenates the shards.

AND ONE CORRECTNESS FIX, which is the reason this file records a column v1 did
not.  v1 recorded no offence-side flag, so `diag_late_game_compare_v1.py` had to
build the "role at the 2:00 mark" from the offence-perspective `score_diff` of
the FIRST window possession and then apply that ONE signed number to every
possession of the simulation -- including the opponent's, which carry the
OPPOSITE sign.  The sim's anchored trailing and leading cells were therefore a
~50/50 mixture of the two roles and any real split cancelled inside them.  The
ACTUAL side never had the defect (`diag_late_game_actual_v1.py` line 62 signs
the anchor per possession with `sgn`).  v2 records the offence's home flag on
BOTH taps -- the event adapter is handed `off` directly; the clock adapter is
handed the team-static row, which carries `site_home` -- so the comparison can
sign the anchor per possession on the sim side too.

Usage:
  diag_late_game_tap_v2.py [n_games] [n_seeds] [out_dir] [shard_k] [n_shards]
                           [seed_offset]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# CLAUDE.md: "Pin thread-count env vars in any container."  One thread per sim
# worker so four shards under the compute cap cannot oversubscribe the box.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

NGAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 250
NSEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUTD = Path(sys.argv[3] if len(sys.argv) > 3 else "results/late_game/tap_v5b")
SHARD = int(sys.argv[4]) if len(sys.argv) > 4 else 0
NSHARD = int(sys.argv[5]) if len(sys.argv) > 5 else 1
SEED0 = int(sys.argv[6]) if len(sys.argv) > 6 else 0
OUTD.mkdir(parents=True, exist_ok=True)

from cbb_sim.engine import loop as L                                 # noqa: E402
from cbb_sim.engine.adapters import STATE_INDEX, Adapters            # noqa: E402
from cbb_sim.engine.inputs import EngineInputs                       # noqa: E402
from cbb_sim.models import possession_outcome as PO                  # noqa: E402

I = STATE_INDEX
inp = EngineInputs.load(str(ROOT / "data/processed/models/engine"), "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
#: `site_home` in the shared team-static block: 1.0 when the OFFENCE is the
#: home team.  `loop.py` hands `inp.team_static[gidx, off]` to the clock, so
#: this column is the offence's side on the clock tap's own rows.
J_HOME = inp.team_names["site_home"]
REC = {"clock": [], "event": []}
SEED = [0]


class ClockTap:
    """Delegates every call to the real adapter and returns its value
    unchanged.  `*a` rather than named arguments so the v5b five-argument
    `draw` and the v3c four-argument one both pass straight through."""

    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def draw(self, team, x, u, *a):
        dur = self._r.draw(team, x, u, *a)
        gidx = a[0] if a and a[0] is not None else np.full(len(x), -1, dtype=np.int64)
        REC["clock"].append(np.column_stack([
            np.full(len(x), SEED[0]), gidx, np.asarray(dur, float),
            x[:, I["period"]], x[:, I["seconds_remaining"]],
            x[:, I["score_diff"]], x[:, I["in_bonus"]], team[:, J_HOME]]))
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
            state[:, I["chance_number"]],
            (np.asarray(off) == 0).astype(float), np.asarray(p, float)]))
        return p


ad.clock = ClockTap(ad.clock)
ad.event = EventTap(ad.event)

# One shared game sample, split into disjoint shards, so the shards together
# are one stratified sample of the universe rather than four overlapping ones.
rng = np.random.default_rng(11)
pool = np.sort(rng.choice(len(inp.games), size=NGAMES * NSHARD, replace=False))
rows = np.ascontiguousarray(pool[SHARD::NSHARD]).astype(np.int64)
n = len(rows)
print("=== late-game tap v2: SERVED defaults, shard %d/%d, %d games x %d seeds ==="
      % (SHARD, NSHARD, n, NSEEDS))
print("   flags: " + json.dumps({k: v for k, v in ad.flags.items()
                                 if k.startswith("ENGINE_")}))
t0 = time.time()
games = []
for s in range(SEED0, SEED0 + NSEEDS):
    SEED[0] = s
    res = L.simulate_chunk(inp, ad, rows, np.full(n, s, dtype=np.int64),
                           keep_players=False)
    games.append(res.games.assign(gidx=rows))
    print("  seed %d done  %.1f s" % (s, time.time() - t0), flush=True)

G = pd.concat(games, ignore_index=True)
clk = pd.DataFrame(np.vstack(REC["clock"]),
                   columns=["seed", "gidx", "dur", "period", "sec", "sd", "bonus",
                            "off_is_home"])
ev = pd.DataFrame(np.vstack(REC["event"]),
                  columns=["seed", "gidx", "is_first", "period", "sec", "sd",
                           "bonus", "chance", "off_is_home"] + list(PO.CLASSES))
sfx = "_s%d" % SHARD
G.to_parquet(OUTD / ("games%s.parquet" % sfx))
clk.to_parquet(OUTD / ("clock_tap%s.parquet" % sfx))
ev.to_parquet(OUTD / ("event_tap%s.parquet" % sfx))
(OUTD / ("flags%s.json" % sfx)).write_text(
    json.dumps({"flags": ad.flags, "n_games": int(n), "n_seeds": int(NSEEDS),
                "seed0": SEED0, "shard": SHARD, "n_shards": NSHARD,
                "game_rows": rows.tolist()}, indent=1, default=str),
    encoding="utf-8")
print("  possessions %d  chances %d  sims %d  in %.1f s"
      % (len(clk), len(ev), len(G), time.time() - t0))
