"""diag_g1g5_tap_v2.py -- per-possession clock/state tap, v2 (clock round 6 closed loop).

Versioned sibling of diag_g1g5_tap_v1.py.  Differences: (1) the round-6
switch ENGINE_ANDONE_LABEL is passed through from the caller's environment and
recorded (every other flag pinned to the served stack as in v1); (2) the
halftime score of every simulation is recorded (for the G7 half-share veto) by
wrapping `loop._state_block`, which returns its value unchanged.

(v1 docstring follows.)

Lane B, 2026-09-30.  MEASUREMENT ONLY.  NOTHING IN `src/cbb_sim/` IS CHANGED
BY THIS SCRIPT.  Same pattern as `diag_g4_tap_v1.py` / `diag_late_game_tap_v2.py`:
the clock adapter is wrapped IN THIS PROCESS ONLY, the wrapper delegates to the
real `draw` and returns its value unchanged, one seed per `simulate_chunk`
call.  Bit-identity with the served 200-seed run is CHECKED by
`diag_g1g5_possessions_v1.py` (games.parquet compared row for row on the
shared (game_id, seed) pairs), not asserted.

Flags pinned to the SERVED stack (the v5b 200-seed gate read).

Recorded, one row per possession: seed, gidx, off_is_home, period,
seconds_remaining at the start, prev_end code (loop.PREV), the drawn duration
and the consumed duration min(dur, seconds_remaining).

Usage: diag_g1g5_tap_v1.py [n_games] [n_seeds] [out_dir] [seed0]
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

FLAGS = {
    "ENGINE_EVENT": "round2_s1", "ENGINE_CLOCK": "v5b_glat_pmean",
    "ENGINE_ROTATION": "reference", "ENGINE_FG3": "decision8",
    "ENGINE_FG_MAKE": "round4_B1", "ENGINE_REBOUND": "s1_weekly",
    "ENGINE_FREE_THROW": "s1_conf_aligned", "ENGINE_ROTATION_SCHEME": "s1",
    "ENGINE_INPUTS_VERSION": "v2",
}
for k, v in FLAGS.items():
    os.environ[k] = v
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[k] = "1"

NGAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 500
NSEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUTD = Path(sys.argv[3] if len(sys.argv) > 3 else "results/g1g5_diag/tap_r6")
SEED0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
OUTD.mkdir(parents=True, exist_ok=True)

from cbb_sim.engine import loop as L                                 # noqa: E402
from cbb_sim.engine.adapters import STATE_INDEX, Adapters            # noqa: E402
from cbb_sim.engine.inputs import EngineInputs                       # noqa: E402

I = STATE_INDEX
inp = EngineInputs.load(str(ROOT / "data/processed/models/engine"), "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
J_HOME = inp.team_names["site_home"]
PREV_COLS = [I["prev_end_DREB"], I["prev_end_TOV"], I["prev_end_made_FG"],
             I["prev_end_made_FT"], I["prev_end_other"]]
REC: list[np.ndarray] = []
SEED = [0]


class ClockTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def draw(self, team, x, u, *a):
        dur = self._r.draw(team, x, u, *a)
        gidx = a[0] if a and a[0] is not None else np.full(len(x), -1, dtype=np.int64)
        oh = x[:, PREV_COLS]
        # loop.PREV: period_start 0, DREB 1, TOV 2, made_FG 3, made_FT 4, other 5
        code = np.where(oh.sum(axis=1) > 0.5, oh.argmax(axis=1) + 1, 0)
        sec = x[:, I["seconds_remaining"]]
        d = np.asarray(dur, float)
        REC.append(np.column_stack([
            np.full(len(x), SEED[0]), np.asarray(gidx, float), team[:, J_HOME],
            x[:, I["period"]], sec, code.astype(float), d, np.minimum(d, sec),
            x[:, I["score_diff"]], x[:, I["in_bonus"]]]))
        return dur


ad.clock = ClockTap(ad.clock)

HALF: dict = {}
_real_sb = L._state_block


def state_block_tap(st, act, n_state, off_sd, bonus):
    per = st.period[act]
    sel = act[per >= 2]
    if len(sel):
        for r in sel[st.seconds_remaining[sel] == 1200]:
            k = (SEED[0], int(st.game_index[r]))
            if k not in HALF and int(st.period[r]) == 2:
                HALF[k] = (int(st.pts[r, 0]), int(st.pts[r, 1]))
    return _real_sb(st, act, n_state, off_sd, bonus)


L._state_block = state_block_tap

NG = len(inp.games)
step = max(1, NG // NGAMES)
rows_sel = np.arange(0, NG, step, dtype=np.int64)[:NGAMES]
print(f"=== g1g5 tap: {len(rows_sel)} games x {NSEEDS} seeds (seed0={SEED0}) ===", flush=True)
t0 = time.time()
games = []
for s in range(SEED0, SEED0 + NSEEDS):
    SEED[0] = s
    res = L.simulate_chunk(inp, ad, rows_sel, np.full(len(rows_sel), s, dtype=np.int64),
                           keep_players=False)
    games.append(res.games.assign(gidx=rows_sel))
    print(f"  seed {s} done {time.time() - t0:.1f}s", flush=True)

G = pd.concat(games, ignore_index=True)
P = pd.DataFrame(np.vstack(REC), columns=[
    "seed", "gidx", "off_is_home", "period", "sec", "prev_end", "dur", "used", "sd", "bonus"])
G["h1_home_pts"] = [HALF.get((int(a), int(b)), (np.nan, np.nan))[0] for a, b in zip(G["seed"], G["gidx"])]
G["h1_away_pts"] = [HALF.get((int(a), int(b)), (np.nan, np.nan))[1] for a, b in zip(G["seed"], G["gidx"])]
G.to_parquet(OUTD / f"games_{SEED0}.parquet")
P.to_parquet(OUTD / f"poss_{SEED0}.parquet")
(OUTD / f"flags_{SEED0}.json").write_text(json.dumps(
    {"flags": {k: v for k, v in ad.flags.items() if str(k).startswith("ENGINE_")},
     "ENGINE_ANDONE_LABEL": os.environ.get("ENGINE_ANDONE_LABEL", "engine"),
     "n_games": len(rows_sel), "seeds": [SEED0, SEED0 + NSEEDS - 1]}, default=str, indent=1),
    encoding="utf-8")
print(f"  sims {len(G)} possessions {len(P)} in {time.time() - t0:.1f}s", flush=True)
