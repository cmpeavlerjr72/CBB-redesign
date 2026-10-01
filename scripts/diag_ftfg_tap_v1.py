"""diag_ftfg_tap_v1.py -- per-trip free-throw tap on the SERVED stack v2 (lane B, 2026-10-01).

MEASUREMENT ONLY. Nothing in `src/cbb_sim/` changes: `loop._shoot_trip` and
`ad.ft.predict` are wrapped IN THIS PROCESS ONLY; the wrappers call the real
functions and return their values unchanged. Bit-identity with the served
200-seed run (`v3full_COMB9GCTKD_s200_o0`) is CHECKED on games.parquet by the
grader (`diag_ftfg_report_v1.py`), not asserted.

One row per free-throw trip: sim row -> (game_id, seed), shooting side, period,
seconds remaining, both scores BEFORE the trip, team fouls, attempts, makes,
sum of the attempts' make probabilities, one-and-one flag, and the shooter's
as-of FT feature (who shoots).

Usage: diag_ftfg_tap_v1.py <game_lo> <game_hi> <n_seeds> <out_dir>
Engine defaults = served v2; inputs engine_v3 / F2_2025; 1 thread.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[k] = "1"
os.environ.setdefault("CBB_TRUTH", "verified_v1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from cbb_sim.engine import loop as L  # noqa: E402
from cbb_sim.engine.adapters import Adapters  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402

LO, HI, NS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
OUTD = Path(sys.argv[4])
OUTD.mkdir(parents=True, exist_ok=True)

inp = EngineInputs.load("data/processed/models/engine_v3", "F2_2025")
ad = Adapters.load(inp, "F2", 2025)

REC: list[dict] = []
CUR = {"gi": None, "sd": None, "p": [], "slot0": []}
_orig_trip = L._shoot_trip
_orig_pred = ad.ft.predict


def pred_wrap(team, slot, x, gidx, *a, **k):
    p = _orig_pred(team, slot, x, gidx, *a, **k)
    CUR["p"].append(np.asarray(p, dtype=np.float64).copy())
    CUR["slot0"].append(np.asarray(slot)[:, 0].astype(np.float64).copy() if np.ndim(slot) == 2 else None)
    return p


def trip_wrap(st, inp_, ad_, book, rows, shooter, n_att, one_and_one, n_state):
    rows = np.asarray(rows)
    side = st.off[rows].astype(np.int64)
    pts_off = st.pts[rows, side].astype(np.int64)
    pts_def = st.pts[rows, 1 - side].astype(np.int64)
    fta0 = st.box["fta"][rows, side].astype(np.int64)
    ftm0 = st.box["ftm"][rows, side].astype(np.int64)
    tf_def = st.team_fouls[rows, 1 - side].astype(np.int64)
    per = st.period[rows].astype(np.int64)
    sec = st.seconds_remaining[rows].astype(np.int64)
    CUR["p"], CUR["slot0"] = [], []
    out = _orig_trip(st, inp_, ad_, book, rows, shooter, n_att, one_and_one, n_state)
    fta = st.box["fta"][rows, side].astype(np.int64) - fta0
    ftm = st.box["ftm"][rows, side].astype(np.int64) - ftm0
    # attempt-level p arrive in rounds (attempt 1 for all rows, attempt 2 for the alive subset ...)
    psum = np.zeros(len(rows))
    p1 = np.full(len(rows), np.nan)
    alive = np.ones(len(rows), dtype=bool)
    na = np.asarray(n_att)
    oao = np.asarray(one_and_one)
    # replay the alive pattern from realised fta to map p arrays to rows
    take_sets = []
    for a in range(3):
        take = (fta > a)
        take_sets.append(np.flatnonzero(take))
    for a, pa in enumerate(CUR["p"]):
        ks = take_sets[a]
        if len(ks) != len(pa):           # should not happen; keep the row-sum unknown
            psum[:] = np.nan
            break
        psum[ks] += pa
        if a == 0:
            p1[ks] = pa
    gi, sd = CUR["gi"][rows], CUR["sd"][rows]
    REC.append(pd.DataFrame({"gidx": gi, "seed": sd, "side": side, "period": per, "sec": sec,
                             "pts_off": pts_off, "pts_def": pts_def, "tf_def": tf_def,
                             "n_att": na, "oao": oao, "fta": fta, "ftm": ftm, "psum": psum, "p1": p1}))
    return out


L._shoot_trip = trip_wrap
ad.ft.predict = pred_wrap

t0 = time.time()
games_all = []
seeds = np.arange(NS, dtype=np.int64)
for blk in range(LO, HI, 50):
    rows_g = np.arange(blk, min(HI, blk + 50), dtype=np.int64)
    gi = np.repeat(rows_g, NS)
    sd = np.tile(seeds, len(rows_g))
    CUR["gi"], CUR["sd"] = gi, sd
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=False)
    games_all.append(res.games)
    print(f"[{time.time()-t0:7.1f}s] games {rows_g[-1]+1}/{HI}", flush=True)
trips = pd.concat(REC, ignore_index=True)
trips["game_id"] = inp.games["game_id"].to_numpy()[trips["gidx"].to_numpy()]
trips.to_parquet(OUTD / f"trips_{LO}_{HI}.parquet", index=False)
pd.concat(games_all, ignore_index=True).to_parquet(OUTD / f"games_{LO}_{HI}.parquet", index=False)
print(f"done {time.time()-t0:.1f}s trips {len(trips)}")
