"""diag_fgctx_tap_v1.py -- per-FGA make-probability tap on the SERVED stack v2 (lane B, 2026-10-01).

MEASUREMENT ONLY. Nothing in `src/cbb_sim/` changes. `state.new_state` and
`SharedShooting.shift` (the last step that sets the served p_make) are wrapped IN
THIS PROCESS ONLY and return their values unchanged. One row per FGA: sim row ->
(game_id, seed), side, type, period, seconds remaining, both scores before the
shot, final p_make. Made / missed is not visible to the wrapper; the grader takes
luck at the game level from the box (FGM - sum p) and composition by context
(sum p - p0 A). Usage: diag_fgctx_tap_v1.py <game_lo> <game_hi> <n_seeds> <out_dir>
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from cbb_sim.engine import loop as L  # noqa: E402
from cbb_sim.engine import shared_shooting as SSL  # noqa: E402
from cbb_sim.engine.adapters import Adapters  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402

LO, HI, NS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
OUTD = Path(sys.argv[4])
OUTD.mkdir(parents=True, exist_ok=True)
inp = EngineInputs.load("data/processed/models/engine_v3", "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
REC: list[pd.DataFrame] = []
CUR: dict = {}
_orig_new = L.S.new_state
_orig_shift = SSL.SharedShooting.shift


def new_wrap(*a, **k):
    st = _orig_new(*a, **k)
    CUR["st"] = st
    return st


def shift_wrap(self, p, rows, side, type_idx):
    out = _orig_shift(self, p, rows, side, type_idx)
    st = CUR["st"]
    rows = np.asarray(rows)
    side = np.asarray(side).astype(np.int64)
    REC.append(pd.DataFrame({
        "gidx": CUR["gi"][rows], "seed": CUR["sd"][rows], "side": side, "type": int(type_idx),
        "period": st.period[rows].astype(np.int64), "sec": st.seconds_remaining[rows].astype(np.int64),
        "pts_off": st.pts[rows, side].astype(np.int64), "pts_def": st.pts[rows, 1 - side].astype(np.int64),
        "p": np.asarray(out, dtype=np.float64)}))
    return out


L.S.new_state = new_wrap
SSL.SharedShooting.shift = shift_wrap
t0 = time.time()
seeds = np.arange(NS, dtype=np.int64)
games_all = []
for blk in range(LO, HI, 50):
    rows_g = np.arange(blk, min(HI, blk + 50), dtype=np.int64)
    gi = np.repeat(rows_g, NS)
    sd = np.tile(seeds, len(rows_g))
    CUR["gi"], CUR["sd"] = gi, sd
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=False)
    games_all.append(res.games)
    print(f"[{time.time()-t0:7.1f}s] games {rows_g[-1]+1}/{HI}", flush=True)
f = pd.concat(REC, ignore_index=True)
f["game_id"] = inp.games["game_id"].to_numpy()[f["gidx"].to_numpy()]
f.to_parquet(OUTD / f"fga_{LO}_{HI}.parquet", index=False)
pd.concat(games_all, ignore_index=True).to_parquet(OUTD / f"games_{LO}_{HI}.parquet", index=False)
print(f"done {time.time()-t0:.1f}s fga {len(f)}")
