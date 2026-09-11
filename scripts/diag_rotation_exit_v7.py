#!/usr/bin/env python
"""
diag_rotation_exit_v7.py -- the round-7 report-only diagnostic (16.7): WHO LEAVES
the floor, by swap size.

Round 6's audit section 4 measured the exit side on SINGLE swaps only and found
the binding defect: every arm since round 5 takes a starter off on 0.456-0.467 of
single swaps against a real 0.587. This script measures the same object for the
round-7 arms and extends it to every swap size, through ONE function applied to
the simulated and to the actual on-floor sequence with the SAME as-of predicted
starter set on both sides, so every row is like for like.

Reported per arm and for ACTUAL:
    * starter share of ALL departing players, overall and by swap size 1 / 2 / 3+
    * the share of swaps that take at least one starter off, by size
    * the size-1 joint of audit 4 (P(starter in | bench out), P(starter in |
      starter out)), so round 6's table extends by three rows

    .venv/Scripts/python.exe scripts/diag_rotation_exit_v7.py --sim-games 200
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

OUT = ROOT / "data" / "processed" / "models" / "rotation" / "exit_audit_2026-09-11.json"
ARMS = ["K1_cond_class", "X1_exit_class", "X2_exit_class_foul", "X3_exit_class_prev"]


def wilson(k: float, n: float) -> float:
    """95% Wilson half-width, the interval every descriptive rate in the round-6
    and round-7 audits carries."""
    if n <= 0:
        return float("nan")
    p = k / n
    z = 1.96
    return float(z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n))


def _walk(lu: np.ndarray, starters: set) -> dict:
    """Exit- and entry-side class counts over one on-floor sequence, by swap
    size. `lu` is (npos, 5) of player ids; `starters` is the as-of predicted
    five. Sizes are bucketed 1 / 2 / 3+ (index 0 / 1 / 2)."""
    n_wave = np.zeros(3)
    n_leave = np.zeros(3)
    k_out = np.zeros(3)
    k_in = np.zeros(3)
    any_st = np.zeros(3)
    joint = np.zeros((2, 2))
    prev = set(int(x) for x in lu[0])
    for i in range(1, lu.shape[0]):
        cur = set(int(x) for x in lu[i])
        if cur == prev:
            continue
        off, on_ = prev - cur, cur - prev
        s = len(off)
        if s and s == len(on_):
            b = 0 if s == 1 else (1 if s == 2 else 2)
            ko = sum(1 for p in off if p in starters)
            ki = sum(1 for p in on_ if p in starters)
            n_wave[b] += 1
            n_leave[b] += s
            k_out[b] += ko
            k_in[b] += ki
            any_st[b] += 1 if ko else 0
            if s == 1:
                joint[1 if ko else 0, 1 if ki else 0] += 1
        prev = cur
    return {"n_wave": n_wave, "n_leave": n_leave, "k_out": k_out, "k_in": k_in,
            "any_st": any_st, "joint": joint}


def _acc(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def _empty() -> dict:
    return {"n_wave": np.zeros(3), "n_leave": np.zeros(3), "k_out": np.zeros(3),
            "k_in": np.zeros(3), "any_st": np.zeros(3), "joint": np.zeros((2, 2))}


def _report(d: dict) -> dict:
    nw, nl, ko, ki, ay, j = (d["n_wave"], d["n_leave"], d["k_out"], d["k_in"],
                             d["any_st"], d["joint"])
    out = {"n_waves_by_size": [int(x) for x in nw],
           "starter_share_of_leavers_by_size":
               [round(float(ko[i] / nl[i]), 4) if nl[i] else float("nan")
                for i in range(3)],
           "starter_share_of_entrants_by_size":
               [round(float(ki[i] / nl[i]), 4) if nl[i] else float("nan")
                for i in range(3)],
           "p_any_starter_out_by_size":
               [round(float(ay[i] / nw[i]), 4) if nw[i] else float("nan")
                for i in range(3)],
           "starter_share_of_leavers_overall":
               round(float(ko.sum() / nl.sum()), 4) if nl.sum() else float("nan"),
           "starter_share_of_entrants_overall":
               round(float(ki.sum() / nl.sum()), 4) if nl.sum() else float("nan"),
           "p_starter_in_given_bench_out":
               round(float(j[0, 1] / max(j[0].sum(), 1)), 4),
           "p_starter_in_given_starter_out":
               round(float(j[1, 1] / max(j[1].sum(), 1)), 4),
           "n_bench_out": int(j[0].sum()), "n_starter_out": int(j[1].sum()),
           "underpowered_sizes": [bool(nw[i] < 300) for i in range(3)]}
    out["spread_pp"] = round(100.0 * (out["p_starter_in_given_bench_out"]
                                      - out["p_starter_in_given_starter_out"]), 1)
    return out


def sim_check(n_games: int, seed: int) -> dict:
    import train_rotation_v5 as V5T
    import train_rotation_v6 as V6T
    import train_rotation_v7 as V7T
    args_d = {"test_games": n_games, "seeds": 1, "noise_seeds": 3, "noise_games": 30,
              "wave_team_games": 6000, "min_prior_games": 3, "fit_seed": 11,
              "floor_fit_seed": 101, "floor_seed": 23, "floor_b_arm": "X3_exit_class_prev",
              "workers": 1, "skip_floor_a": True, "skip_floor_b": True,
              "skip_fit": True, "mode": "bakeoff", "smoke": False, "tag": "exitcheck"}
    c = V7T.ensure_ctx(args_d)
    out = {}
    for name in ARMS:
        tot = _empty()
        for tag, gl in c["games_by_window"].items():
            if not gl:
                continue
            pw = V5T.priors_for("S1", tag)
            arm = V7T.make_arm7(name, tag, "")
            for g in gl:
                keys = [k for k in c["keys_by_game"][g] if k in pw]
                if len(keys) != 2:
                    continue
                rng = R.game_stream(seed, g)
                for key in keys:
                    pr, sc = pw[key], c["scripts"][key]
                    lu, _fh = arm.simulate(pr, sc, rng)
                    st = set(int(x) for x in pr.pids[pr.starters()[:5]])
                    tot = _acc(tot, _walk(lu, st))
        out[name] = _report(tot)
    # ACTUAL on the same games, with the same as-of predicted starter set
    tot = _empty()
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
        tot = _acc(tot, _walk(g[R.SLOTS].to_numpy(dtype="int64"), st))
    out["ACTUAL_asof_starters"] = _report(tot)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-games", type=int, default=200)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--by-composition", action="store_true")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    args = ap.parse_args()
    if args.by_composition:
        prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        for s in args.seasons:
            r = by_composition(s)
            prev.setdefault(str(s), {})["by_composition"] = r
            print(f"--- season {s}: P(a starter leaves | size, starters on floor)")
            for row in r["by_size"]:
                flag = "  UNDERPOWERED" if row["underpowered"] else ""
                print(f"  size {row['size']} on {row['starters_on_floor']} "
                      f"n {row['n_leavers']:>7} p {row['p_starter_leaves']:.4f} "
                      f"+/-{row['ci']:.4f} prop {row['proportional']:.2f} "
                      f"ratio {row['ratio_to_proportional']}{flag}")
            for row in r["by_time"]:
                flag = "  UNDERPOWERED" if row["underpowered"] else ""
                print(f"  {row['time']:<16} on {row['starters_on_floor']} "
                      f"n {row['n_leavers']:>7} p {row['p_starter_leaves']:.4f} "
                      f"prop {row['proportional']:.2f}{flag}")
        OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print("wrote", OUT)
        return
    res = sim_check(args.sim_games, args.sim_seed)
    print(f"--- exit side, {args.sim_games} games, seed {args.sim_seed}")
    print("    (as-of predicted starters on BOTH sides, so like for like)")
    for k, v in res.items():
        print(f"  {k:<24} starter share of leavers "
              f"{v['starter_share_of_leavers_overall']:.4f} "
              f"by size {v['starter_share_of_leavers_by_size']} "
              f"n {v['n_waves_by_size']} "
              f"| size-1 joint {v['p_starter_in_given_bench_out']:.4f} / "
              f"{v['p_starter_in_given_starter_out']:.4f}")
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev[f"exit_check_{args.sim_games}g_seed{args.sim_seed}"] = res
    OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print("wrote", OUT)




# ===========================================================================
# Descriptive: does the exit class depend on the COMPOSITION on the floor?
# ===========================================================================
def by_composition(season: int) -> dict:
    """P(a starter leaves | size 1, starters on the floor) on the ACTUAL
    sequences, with the game's OWN starting five and its own participant pool --
    the descriptive convention of the round-6 composition audit, a measurement of
    coaching behaviour and never a bake-off result.

    This is the support check for the object round 7's failure names: an exit
    class count that does not know how many starters are on the floor is a level,
    not a rate. `prop` is the proportional (composition-neutral) rate the draw
    would need if leaving were independent of class.
    """
    tp = R.load_team_possessions(season)
    cells = {}
    tcells = {}
    for _key, g in tp.groupby(["game_id", "team_id"], sort=False):
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        if lu.shape[0] < 20:
            continue
        per = g["period"].to_numpy(dtype="int64")
        clk = g["start_clock"].to_numpy(dtype="int64")
        starters = set(int(x) for x in lu[0])
        prev = set(int(x) for x in lu[0])
        for i in range(1, lu.shape[0]):
            cur = set(int(x) for x in lu[i])
            if cur == prev:
                continue
            off, on_ = prev - cur, cur - prev
            s = len(off)
            if s and s == len(on_):
                n_on = sum(1 for p in prev if p in starters)
                ko = sum(1 for p in off if p in starters)
                c = cells.setdefault((s if s <= 3 else 3, n_on), [0, 0])
                c[0] += ko
                c[1] += s
                late = 2 if (per[i] >= 2 and clk[i] <= 480) else (0 if per[i] == 1 else 1)
                t = tcells.setdefault((late, n_on), [0, 0])
                t[0] += ko
                t[1] += s
            prev = cur
    rows = []
    for (s, n_on), (num, den) in sorted(cells.items()):
        rows.append({"size": s, "starters_on_floor": n_on, "n_leavers": den,
                     "p_starter_leaves": round(num / den, 4) if den else float("nan"),
                     "proportional": round(n_on / 5.0, 4),
                     "ratio_to_proportional": round((num / den) / (n_on / 5.0), 4)
                     if den and n_on else float("nan"),
                     "ci": round(wilson(num, den), 4),
                     "underpowered": bool(den < 300)})
    trows = []
    names = {0: "H1", 1: "H2 20:00-08:00", 2: "final 8:00"}
    for (t, n_on), (num, den) in sorted(tcells.items()):
        trows.append({"time": names[t], "starters_on_floor": n_on, "n_leavers": den,
                      "p_starter_leaves": round(num / den, 4) if den else float("nan"),
                      "proportional": round(n_on / 5.0, 4),
                      "ci": round(wilson(num, den), 4),
                      "underpowered": bool(den < 300)})
    return {"by_size": rows, "by_time": trows}


if __name__ == "__main__":
    main()
