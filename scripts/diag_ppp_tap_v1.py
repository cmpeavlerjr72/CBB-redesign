"""diag_ppp_tap_v1.py -- fg_make / free-throw input tap on the COMB stack (lane I, 2026-09-30).

MEASUREMENT ONLY. Nothing in src/ is changed. Same pattern as diag_g1g5_tap_v3.py: adapters are
wrapped IN THIS PROCESS ONLY; each wrapper delegates to the real predict and returns its value
unchanged; one seed per simulate_chunk call. The v3 event team block is swapped in exactly as
scripts/run_engine_v3evb_v1.py does (proved equal to the box S0 overlay by lane B).
Bit-identity of the games rows with the box COMB run is CHECKED by diag_ppp_tap_grade_v1.py.

Stack (COMB, the Decision 11 combined set): ENGINE_CLOCK=v5b_r6L2_glat_pmean,
ENGINE_SHOT_BLOCK=K2_Ocell, ENGINE_FOUL_JOINT=R8b on the served rest.

Recorded per fg_make call row: seed, gidx, off side, shot class, the model's exact 16-feature
row (as assembled for the model), p. Per free-throw attempt: seed, gidx, side, assembled
features, p.

Usage: diag_ppp_tap_v1.py <input_dir> <sample_file> <n_seeds> <out_dir> <seed0> [arm=COMB|S0]
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

for _k in ("ENGINE_ROTATION_SCHEME", "ENGINE_INPUTS_VERSION", "ENGINE_SHARED_SHOOTING",
           "ENGINE_ANDONE_LABEL", "ENGINE_LATE_GAME"):
    os.environ.pop(_k, None)
ARM = sys.argv[6] if len(sys.argv) > 6 else "COMB"
FLAGS = {"ENGINE_EVENT": "round2_s1", "ENGINE_CLOCK": "v5b_glat_pmean",
         "ENGINE_ROTATION": "reference", "ENGINE_FG3": "decision8", "CBB_TRUTH": "verified_v1"}
if ARM == "COMB":
    FLAGS.update({"ENGINE_CLOCK": "v5b_r6L2_glat_pmean", "ENGINE_SHOT_BLOCK": "K2_Ocell",
                  "ENGINE_FOUL_JOINT": "R8b"})
else:
    os.environ.pop("ENGINE_SHOT_BLOCK", None)
    os.environ.pop("ENGINE_FOUL_JOINT", None)
for k, v in FLAGS.items():
    os.environ[k] = v
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[k] = "1"

INPUT_DIR, SAMPLE = sys.argv[1], sys.argv[2]
NSEEDS = int(sys.argv[3])
OUTD = Path(sys.argv[4])
SEED0 = int(sys.argv[5])
OUTD.mkdir(parents=True, exist_ok=True)

from cbb_sim.engine import adapters as A                              # noqa: E402
from cbb_sim.engine import loop as L                                  # noqa: E402
from cbb_sim.engine.adapters import Adapters, _assemble               # noqa: E402
from cbb_sim.engine.inputs import EngineInputs                        # noqa: E402

inp = EngineInputs.load(str(ROOT / INPUT_DIR), "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
blk = np.load(ROOT / INPUT_DIR / "event_block_F2_2025.npz")["team_block"]
assert blk.shape == ad.event.team_block.shape
ad.event.team_block = blk
J_HOME = inp.team_names["site_home"]

SEED = [0]
FG_REC: dict[str, list] = {"FGA_rim": [], "FGA_jump2": [], "FGA_3": []}
FT_REC: list = []
_fg_predict = ad.fg.predict
_ft_predict = ad.ft.predict


def fg_tap(cls_name, team, slot, state, gidx=None):
    p = _fg_predict(cls_name, team, slot, state, gidx)
    m = _assemble(ad.fg.plans[cls_name], team, slot, state)
    FG_REC[cls_name].append(np.column_stack([
        np.full(len(p), SEED[0], dtype=np.float64), np.asarray(gidx, float), team[:, J_HOME],
        m, p]).astype(np.float32))
    return p


def ft_tap(team, slot, state, gidx=None, *a, **k):
    p = _ft_predict(team, slot, state, gidx, *a, **k)
    m = _assemble(ad.ft.plan, team, slot, state)
    FT_REC.append(np.column_stack([
        np.full(len(p), SEED[0], dtype=np.float64), np.asarray(gidx, float), team[:, J_HOME],
        m, p]).astype(np.float32))
    return p


ad.fg.predict = fg_tap
ad.ft.predict = ft_tap

_ids = np.sort(pd.read_parquet(ROOT / SAMPLE)["game_id"].to_numpy().astype("int64"))
_pos = {int(g): k for k, g in enumerate(inp.games["game_id"].to_numpy())}
rows_sel = np.array([_pos[int(g)] for g in _ids if int(g) in _pos], dtype=np.int64)
print(f"=== ppp tap {ARM}: {len(rows_sel)} games x {NSEEDS} seeds (seed0={SEED0}) ===", flush=True)
t0 = time.time()
games = []
for s in range(SEED0, SEED0 + NSEEDS):
    SEED[0] = s
    res = L.simulate_chunk(inp, ad, rows_sel, np.full(len(rows_sel), s, dtype=np.int64),
                           keep_players=False)
    games.append(res.games.assign(gidx=rows_sel))
    print(f"  seed {s} done {time.time() - t0:.1f}s", flush=True)

G = pd.concat(games, ignore_index=True)
G.to_parquet(OUTD / f"games_{SEED0}.parquet")
fg_names = {}
for c, recs in FG_REC.items():
    names = list(json.loads((Path(ad.fg.source[c]["path"]) / f"manifest_{c}.json")
                            .read_text(encoding="utf-8"))["features"])
    fg_names[c] = names
    df = pd.DataFrame(np.vstack(recs), columns=["seed", "gidx", "off_is_home", *names, "p"])
    df.to_parquet(OUTD / f"fg_{c}_{SEED0}.parquet")
ft_cols = [f"f{i}" for i in range(ad.ft.plan.width)]
pd.DataFrame(np.vstack(FT_REC), columns=["seed", "gidx", "off_is_home", *ft_cols, "p"]).to_parquet(
    OUTD / f"ft_{SEED0}.parquet")
(OUTD / f"meta_{SEED0}.json").write_text(json.dumps(
    {"arm": ARM, "flags": {k: v for k, v in ad.flags.items() if str(k).startswith("ENGINE_")},
     "env": {k: os.environ.get(k) for k in FLAGS}, "fg_features": fg_names,
     "n_games": len(rows_sel), "seeds": [SEED0, SEED0 + NSEEDS - 1],
     "runtime_s": time.time() - t0}, default=str, indent=1), encoding="utf-8")
print(f"  sims {len(G)} in {time.time() - t0:.1f}s", flush=True)
