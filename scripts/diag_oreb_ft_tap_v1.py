"""diag_oreb_ft_tap_v1.py -- in-process tap of the SERVED v2 stack: free-throw inputs and shot-block probabilities (lane I, 2026-10-01)

MEASUREMENT ONLY. Nothing in src/ is changed. Every ENGINE_* flag is unset, so the run is the served default (served
stack v2, event team block v3), on the v3 inputs. Wrappers live in THIS PROCESS ONLY and return the real value
unchanged; one seed per simulate_chunk call (as diag_ppp_tap_v1.py). Bit-identity of the games rows with the box
adopted run (v3full_COMB9GCTKD_s200_o0) is checked afterwards by the grader.

Recorded:
  ft_<seed0>.parquet   per free-throw attempt: seed, gidx, off_is_home, the model's assembled features, p
                       (same layout as diag_ppp_tap_v1.py, so diag_ppp_ft_swap_v1.py reads it)
  blk_<seed0>.parquet  per (seed, gidx, off side, shot type): missed FGAs n and the sum of P(blocked | miss)
  games_<seed0>.parquet

Usage: diag_oreb_ft_tap_v1.py <input_dir> <sample_file|all> <n_seeds> <out_dir> <seed0>
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
for _k in [k for k in os.environ if k.startswith("ENGINE_")]:
    os.environ.pop(_k, None)
os.environ["CBB_TRUTH"] = "verified_v1"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
          "LIGHTGBM_NUM_THREADS"):
    os.environ[k] = "1"

INPUT_DIR, SAMPLE = sys.argv[1], sys.argv[2]
NSEEDS = int(sys.argv[3])
OUTD = Path(sys.argv[4])
SEED0 = int(sys.argv[5])
OUTD.mkdir(parents=True, exist_ok=True)

from cbb_sim.engine import loop as L                                  # noqa: E402
from cbb_sim.engine import shot_block as SBK                          # noqa: E402
from cbb_sim.engine.adapters import Adapters, _assemble               # noqa: E402
from cbb_sim.engine.inputs import EngineInputs                        # noqa: E402

inp = EngineInputs.load(str(ROOT / INPUT_DIR), "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
J_HOME = inp.team_names["site_home"]
SEED = [0]
FT_REC: list = []
BLK_REC: list = []
_ft_predict = ad.ft.predict
_sb_prob = SBK.ShotBlock.prob


def ft_tap(team, slot, state, gidx=None, *a, **k):
    p = _ft_predict(team, slot, state, gidx, *a, **k)
    m = _assemble(ad.ft.plan, team, slot, state)
    FT_REC.append(np.column_stack([
        np.full(len(p), SEED[0], dtype=np.float64), np.asarray(gidx, float), team[:, J_HOME],
        m, p]).astype(np.float32))
    return p


def sb_tap(self, team_static, gidx, off, slot, type_idx, *a, **k):
    p = _sb_prob(self, team_static, gidx, off, slot, type_idx, *a, **k)
    BLK_REC.append(pd.DataFrame({"seed": SEED[0], "gidx": np.asarray(gidx, np.int64), "off": np.asarray(off, np.int8),
                                 "type": np.asarray(type_idx, np.int8), "p": p})
                   .groupby(["seed", "gidx", "off", "type"], as_index=False).agg(n=("p", "size"), p=("p", "sum")))
    return p


ad.ft.predict = ft_tap
SBK.ShotBlock.prob = sb_tap

if SAMPLE == "all":
    rows_sel = np.arange(len(inp.games), dtype=np.int64)
else:
    _ids = np.sort(pd.read_parquet(ROOT / SAMPLE)["game_id"].to_numpy().astype("int64"))
    _pos = {int(g): k for k, g in enumerate(inp.games["game_id"].to_numpy())}
    rows_sel = np.array([_pos[int(g)] for g in _ids if int(g) in _pos], dtype=np.int64)
print(f"=== oreb/ft tap served v2: {len(rows_sel)} games x {NSEEDS} seeds (seed0={SEED0}) ===", flush=True)
t0 = time.time()
games = []
for s in range(SEED0, SEED0 + NSEEDS):
    SEED[0] = s
    res = L.simulate_chunk(inp, ad, rows_sel, np.full(len(rows_sel), s, dtype=np.int64), keep_players=False)
    games.append(res.games.assign(gidx=rows_sel))
    print(f"  seed {s} done {time.time() - t0:.1f}s", flush=True)

pd.concat(games, ignore_index=True).to_parquet(OUTD / f"games_{SEED0}.parquet")
ft_cols = [f"f{i}" for i in range(ad.ft.plan.width)]
pd.DataFrame(np.vstack(FT_REC), columns=["seed", "gidx", "off_is_home", *ft_cols, "p"]).to_parquet(OUTD / f"ft_{SEED0}.parquet")
pd.concat(BLK_REC).groupby(["seed", "gidx", "off", "type"], as_index=False)[["n", "p"]].sum().to_parquet(
    OUTD / f"blk_{SEED0}.parquet")
(OUTD / f"meta_{SEED0}.json").write_text(json.dumps(
    {"flags": {k: v for k, v in ad.flags.items() if str(k).startswith("ENGINE_")},
     "n_games": len(rows_sel), "seeds": [SEED0, SEED0 + NSEEDS - 1], "runtime_s": time.time() - t0},
    default=str, indent=1), encoding="utf-8")
print(f"  done in {time.time() - t0:.1f}s", flush=True)
