"""diag_tov_tap_v1.py -- in-process tap of possession_outcome on served v2 (or RBTO): P(TOV) by chance (lane I, 2026-10-01).

MEASUREMENT ONLY. Every ENGINE_* flag is unset except the ones passed for the RBTO arm (ENGINE_SEASON_ANCHOR via the
environment of the caller) and an optional rebound manifest override (the same module attribute the overlay runner sets).
Records, per (seed, gidx, off side, chance number, transition flag): the number of chances and the sum of P(TOV) the
served event adapter returns, plus the games rows. One seed per simulate_chunk call.

Usage: diag_tov_tap_v1.py <input_dir> <n_seeds> <out_dir> <seed0> [rb_manifest_override]
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
for _k in [k for k in os.environ if k.startswith("ENGINE_") and k != "ENGINE_SEASON_ANCHOR"]:
    os.environ.pop(_k, None)
os.environ["CBB_TRUTH"] = "verified_v1"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[k] = "1"
INPUT_DIR, NSEEDS, OUTD, SEED0 = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
OUTD.mkdir(parents=True, exist_ok=True)

from cbb_sim.engine import adapters as A  # noqa: E402
if len(sys.argv) > 5:
    A.RB_S1_MANIFEST = Path(sys.argv[5])
from cbb_sim.engine import loop as L  # noqa: E402
from cbb_sim.engine.adapters import Adapters, STATE_INDEX  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

inp = EngineInputs.load(str(ROOT / INPUT_DIR), "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
SEED = [0]
REC: list = []
_pred = ad.event.predict
J_CH, J_TR = STATE_INDEX["chance_number"], STATE_INDEX["is_transition"]
TOV = PO.CLASS_INDEX["TOV"]


def tap(team, state, is_first, gidx, off, *a, **k):
    p = _pred(team, state, is_first, gidx, off, *a, **k)
    REC.append(pd.DataFrame({"seed": SEED[0], "gidx": np.asarray(gidx, np.int64), "off": np.asarray(off, np.int8),
                             "chance": np.minimum(state[:, J_CH], 3).astype(np.int8),
                             "trans": state[:, J_TR].astype(np.int8), "p": p[:, TOV]})
               .groupby(["seed", "gidx", "off", "chance", "trans"], as_index=False).agg(n=("p", "size"), p=("p", "sum")))
    return p


ad.event.predict = tap
rows = np.arange(len(inp.games), dtype=np.int64)
t0 = time.time()
games = []
for s in range(SEED0, SEED0 + NSEEDS):
    SEED[0] = s
    res = L.simulate_chunk(inp, ad, rows, np.full(len(rows), s, dtype=np.int64), keep_players=False)
    games.append(res.games)
    print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)
pd.concat(games, ignore_index=True).to_parquet(OUTD / f"games_{SEED0}.parquet")
pd.concat(REC).groupby(["seed", "gidx", "off", "chance", "trans"], as_index=False)[["n", "p"]].sum().to_parquet(OUTD / f"tov_{SEED0}.parquet")
(OUTD / f"meta_{SEED0}.json").write_text(json.dumps({"flags": {k: v for k, v in ad.flags.items() if str(k).startswith("ENGINE_")},
                                                     "rb_manifest": str(A.RB_S1_MANIFEST), "runtime_s": time.time() - t0},
                                                    default=str, indent=1))
print("done", flush=True)
