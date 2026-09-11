#!/usr/bin/env python
"""
diag_rotation_exit_v9.py -- the round-9 report-only diagnostics (20.8).

Two objects, both measured through ONE function applied to the SIMULATED and to
the ACTUAL on-floor sequence with the SAME as-of predicted starter set on both
sides, so every row is like for like:

  1. round 7's exit-side starter share (16.7), by swap size, so round 9's arms
     drop straight into 17.10's table;
  2. **the object round 9 exists to move**: the realised
     `P(a starter is the man who leaves | single swap, starters on the floor)`
     in the SIM against the same rate on the ACTUAL sequences -- 19.7's table
     with a round-9 column.

`scripts/diag_rotation_exit_v7.py` supplies `_walk`, `_acc`, `_empty`, `_report`
and `wilson` unchanged; nothing in it is modified.

    .venv/Scripts/python.exe scripts/diag_rotation_exit_v9.py --sim-games 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.models import rotation as R  # noqa: E402
import diag_rotation_exit_v7 as D7  # noqa: E402

OUT = ROOT / "data" / "processed" / "models" / "rotation" / "exit_audit_round9_2026-09-11.json"
ARMS = ["Y1_exit_rate", "Z1_exit_marg", "Z2_exit_interact"]


def _walk_comp(lu: np.ndarray, starters: set) -> np.ndarray:
    """(6, 2) accumulator over SINGLE swaps: for each number of predicted
    starters on the floor BEFORE the swap, [starters among the leavers,
    leavers]. The convention of 17.17, with the as-of predicted five in place of
    the game's own so the simulated and actual columns are comparable."""
    acc = np.zeros((6, 2))
    prev = set(int(x) for x in lu[0])
    for i in range(1, lu.shape[0]):
        cur = set(int(x) for x in lu[i])
        if cur == prev:
            continue
        off, on_ = prev - cur, cur - prev
        s = len(off)
        if s == 1 and len(on_) == 1:
            n_on = sum(1 for p in prev if p in starters)
            ko = sum(1 for p in off if p in starters)
            acc[min(n_on, 5), 0] += ko
            acc[min(n_on, 5), 1] += s
        prev = cur
    return acc


def _report_comp(acc: np.ndarray) -> list:
    rows = []
    for ns in range(6):
        num, den = acc[ns]
        rows.append({"starters_on_floor": ns, "n_leavers": int(den),
                     "p_starter_leaves": round(float(num / den), 4) if den else None,
                     "proportional": round(ns / 5.0, 4),
                     "ci": round(D7.wilson(num, den), 4) if den else None,
                     "underpowered": bool(den < 300)})
    return rows


def sim_check(n_games: int, seed: int) -> dict:
    import train_rotation_v5 as V5T
    import train_rotation_v9 as V9T
    args_d = {"test_games": n_games, "seeds": 1, "noise_seeds": 3, "noise_games": 30,
              "wave_team_games": 6000, "min_prior_games": 3, "fit_seed": 11,
              "floor_fit_seed": 101, "floor_seed": 23, "floor_b_arm": "Z1_exit_marg",
              "workers": 1, "skip_floor_a": True, "skip_fit": True,
              "k_folds": 5, "mode": "bakeoff", "smoke": False, "tag": "exitcheck9"}
    c = V9T.ensure_ctx(args_d)
    out, comp = {}, {}
    for name in ARMS:
        tot = D7._empty()
        cacc = np.zeros((6, 2))
        for tag, gl in c["games_by_window"].items():
            if not gl:
                continue
            pw = V5T.priors_for("S1", tag)
            arm = V9T.make_arm9(name, tag, "")
            for g in gl:
                keys = [k for k in c["keys_by_game"][g] if k in pw]
                if len(keys) != 2:
                    continue
                rng = R.game_stream(seed, g)
                for key in keys:
                    pr, sc = pw[key], c["scripts"][key]
                    lu, _fh = arm.simulate(pr, sc, rng)
                    st = set(int(x) for x in pr.pids[pr.starters()[:5]])
                    tot = D7._acc(tot, D7._walk(lu, st))
                    cacc += _walk_comp(lu, st)
        out[name] = D7._report(tot)
        comp[name] = _report_comp(cacc)
    # ACTUAL on the same games, with the same as-of predicted starter set
    tot = D7._empty()
    cacc = np.zeros((6, 2))
    kset = {k for g in c["sel"] for k in c["keys_by_game"][g]}
    tps = c["test"]["tp"]
    tps = tps[[(int(g), int(t)) in kset for g, t in zip(tps["game_id"], tps["team_id"])]]
    pr_all = c["priors_static"]
    for key, g in tps.groupby(["game_id", "team_id"], sort=False):
        key = (int(key[0]), int(key[1]))
        if key not in pr_all:
            continue
        pr = pr_all[key]
        st = set(int(x) for x in pr.pids[pr.starters()[:5]])
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        tot = D7._acc(tot, D7._walk(lu, st))
        cacc += _walk_comp(lu, st)
    out["ACTUAL_asof_starters"] = D7._report(tot)
    comp["ACTUAL_asof_starters"] = _report_comp(cacc)
    return {"exit_side": out, "by_composition": comp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-games", type=int, default=200)
    ap.add_argument("--sim-seed", type=int, default=0)
    args = ap.parse_args()
    res = sim_check(args.sim_games, args.sim_seed)
    res["config"] = {"sim_games": args.sim_games, "sim_seed": args.sim_seed}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n=== exit-side starter share (200 games, seed 0) ===")
    for k, v in res["exit_side"].items():
        print(f"{k:26s} overall {v['starter_share_of_leavers_overall']:.4f} "
              f"by size {v['starter_share_of_leavers_by_size']} "
              f"entrants {v['starter_share_of_entrants_overall']:.4f} "
              f"joint {v['p_starter_in_given_bench_out']:.4f}/"
              f"{v['p_starter_in_given_starter_out']:.4f} spread {v['spread_pp']}")
    print("\n=== P(starter leaves | single swap, starters on floor) ===")
    for k, rows in res["by_composition"].items():
        s = "  ".join(f"{r['starters_on_floor']}:{r['p_starter_leaves']}"
                      f"({r['n_leavers']})" for r in rows if r["n_leavers"])
        print(f"{k:26s} {s}")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
