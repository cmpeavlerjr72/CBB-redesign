#!/usr/bin/env python
"""diag_rot9_adapter_parity.py -- round-9 engine adapter vs the OFFLINE sampler.

The round-9 adapter (`engine/rotation_adapter.next_lineup_round9`, scoped in
`docs/models/rotation/experiments.md` 21.15) is round 6's K1 path with ONE block
replaced: the exit-side `k_out` draw. Everything else is round 6's code, already
parity-checked in round 6. So the parity question this script answers is exactly
the scope of the change:

    on a 150-game x 3-seed sample of REAL engine states, does the adapter's
    vectorized exit block draw the same `k_out` -- and therefore the same
    exit-side STARTER SHARE by `n_starters` -- as the offline scalar block in
    `rotation_v8.run_wave8` that `rotation_v9.run_wave9` delegates to?

Method: `ENGINE_ROT9_AUDIT=1` records every wave the engine resolves (size,
n_st_on, n_bn_on, forced_st, forced_bn, exit_cell, seg, u_x, k_out). The offline
block is then re-run in a scalar Python loop, on those SAME states and the SAME
uniform, from the SAME fitted tables. A tolerance would be meaningless here --
the two must be equal, so floor A for this check is 0.

    .venv/Scripts/python.exe scripts/diag_rot9_adapter_parity.py --games 150 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def offline_k_out(zt: np.ndarray, size, n_st_on, n_bn_on, f_st, f_bn, ce, u_x) -> int:
    """`rotation_v8.run_wave8`'s exit block, scalar, copied clip for clip."""
    sz = int(size)
    row = zt[sz - 1, int(ce), int(min(n_st_on, 5))].copy()
    lo = max(0, sz - int(n_bn_on), int(f_st))
    hi = min(sz, int(n_st_on), sz - int(f_bn))
    if hi < lo:
        hi = lo = int(min(max(lo, 0), sz))
    row[:lo] = 0.0
    row[hi + 1:] = 0.0
    tot = row.sum()
    if tot <= 0:
        return lo
    k = int(np.searchsorted(np.cumsum(row / tot), float(u_x)))
    return int(min(max(k, lo), hi))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arm", default="Z1", choices=["Z1", "Z2"])
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--input-dir", default="data/processed/models/engine")
    ap.add_argument("--out", default="data/processed/models/rotation/"
                                     "round9_adapter_parity_2026-09-11.json")
    a = ap.parse_args()

    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("ENGINE_EVENT", "round2_s1")
    os.environ.setdefault("ENGINE_FG_MAKE", "round3_shooter_S_C_s1")
    os.environ.setdefault("ENGINE_CLOCK", "reference")
    os.environ["ENGINE_ROTATION"] = "reference"

    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs

    t0 = time.time()
    inp = EngineInputs.load(a.input_dir, f"{a.fold}_{a.season}")
    ad = Adapters.load(inp, a.fold, int(a.season))
    os.environ["ENGINE_ROTATION"] = "round9"
    os.environ["ENGINE_ROTATION_ARM"] = a.arm
    os.environ["ENGINE_ROT9_AUDIT"] = "1"
    ad.flags["ENGINE_ROTATION"] = "round9"

    from cbb_sim.engine import loop as L
    from cbb_sim.engine import rotation_adapter as RA

    r9 = RA.load_round9(inp.games, arm=a.arm)
    ZEXIT = r9["ZEXIT"]                      # (n_seg, MW, N_EXIT_CELL, N_ST, MW+1)

    rows = np.argsort(inp.games["game_id"].to_numpy(), kind="stable")[:a.games]
    gi = np.repeat(rows.astype(np.int64), a.seeds)
    sd = np.tile(np.arange(a.seeds, dtype=np.int64), len(rows))
    RA.ROUND9_AUDIT.clear()
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=False)
    rec = np.concatenate(RA.ROUND9_AUDIT, axis=1).T
    n = len(rec)

    off = np.empty(n, dtype=np.int64)
    for i in range(n):
        size, nst, nbn, fst, fbn, ce, seg, u_x, _ = rec[i]
        off[i] = offline_k_out(ZEXIT[int(seg)], size, nst, nbn, fst, fbn, ce, u_x)
    eng = rec[:, 8].astype(np.int64)
    size = rec[:, 0].astype(np.int64)
    nst = rec[:, 1].astype(np.int64)

    mism = int((eng != off).sum())
    by = []
    for b in range(6):
        m = nst == b
        if not m.any():
            by.append({"n_starters": b, "n_waves": 0, "underpowered": True})
            continue
        e = float(eng[m].sum() / size[m].sum())
        o = float(off[m].sum() / size[m].sum())
        by.append({"n_starters": b, "n_waves": int(m.sum()),
                   "engine_starter_share": round(e, 6),
                   "offline_starter_share": round(o, 6),
                   "abs_delta": round(abs(e - o), 6),
                   "underpowered": bool(m.sum() < 300)})

    out = {
        "check": "round-9 adapter exit block vs rotation_v8.run_wave8 scalar block",
        "scope": "experiments.md 21.15; floor A for an identity check is 0",
        "arm": a.arm, "n_games": int(a.games), "n_seeds": int(a.seeds),
        "n_sims": int(len(gi)), "n_waves_audited": n,
        "n_possessions": int(res.n_possessions),
        "k_out_mismatches": mism,
        "max_abs_share_delta": round(float(max(
            (d["abs_delta"] for d in by if d.get("n_waves")), default=0.0)), 6),
        "exit_side_starter_share_by_n_starters": by,
        "runtime_s": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
