#!/usr/bin/env python
"""
diag_rot10_sampler_parity.py -- the bit-exact identity test of 23.9, floor 0.

`run_wave10` fed with round 5's `p_wave` and `p_size` and round 6's `kin`
broadcast over the `n_st` axis must reproduce `run_wave9`'s Z1 lineup and foul
arrays ELEMENT FOR ELEMENT. Two samplers are the same sampler or they are not;
a failure stops round 10.

    .venv/Scripts/python.exe scripts/diag_rot10_sampler_parity.py --games 30 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.models import rotation as R  # noqa: E402
from cbb_sim.models import rotation_v10 as V10  # noqa: E402

OUT = (ROOT / "data" / "processed" / "models" / "rotation"
       / "round10_sampler_parity_2026-09-18.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()

    import train_rotation_v5 as V5T
    import train_rotation_v9 as V9T
    args_d = {"test_games": args.games, "seeds": 1, "noise_seeds": 3,
              "noise_games": 30, "wave_team_games": 6000, "min_prior_games": 3,
              "fit_seed": 11, "floor_fit_seed": 101, "floor_seed": 23,
              "floor_b_arm": "Z1_exit_marg", "workers": 1, "skip_floor_a": True,
              "skip_fit": True, "k_folds": 5, "mode": "bakeoff", "smoke": False,
              "tag": "rot10parity"}
    c = V9T.ensure_ctx(args_d)

    n_tg = n_pos = 0
    bad_lu = bad_fh = 0
    for tag, gl in c["games_by_window"].items():
        if not gl:
            continue
        pw = V5T.priors_for("S1", tag)
        a9 = V9T.make_arm9("Z1_exit_marg", tag, "")
        a10 = V10.ARMS["VREF_z1_broadcast"](
            a9.fit, a9.wave, a9.comp, a9.exit, None, a9.side_state)
        for gid in gl:
            keys = [k for k in c["keys_by_game"][gid] if k in pw]
            if len(keys) != 2:
                continue
            for s in range(args.seeds):
                r9 = R.game_stream(s, gid)
                r10 = R.game_stream(s, gid)
                for key in keys:
                    pr, sc = pw[key], c["scripts"][key]
                    lu9, fh9 = a9.simulate(pr, sc, r9)
                    lu10, fh10 = a10.simulate(pr, sc, r10)
                    n_tg += 1
                    n_pos += int(lu9.shape[0])
                    bad_lu += int((lu9 != lu10).sum())
                    bad_fh += int((np.asarray(fh9) != np.asarray(fh10)).sum())

    ok = (bad_lu == 0 and bad_fh == 0)
    res = {"games": args.games, "seeds": args.seeds, "team_game_sims": n_tg,
           "possessions": n_pos, "lineup_mismatches": bad_lu,
           "foul_mismatches": bad_fh, "identical": ok,
           "seconds": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    if not ok:
        raise SystemExit("IDENTITY TEST FAILED -- round 10 stops here")
    print("PASS: run_wave10(broadcast) == run_wave9(Z1), element for element")


if __name__ == "__main__":
    main()
