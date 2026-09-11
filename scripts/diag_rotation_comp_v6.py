#!/usr/bin/env python
"""
diag_rotation_comp_v6.py -- the round-6 evidence audit: WHO comes in, given who
went off.

`docs/tests/rotation_wave_audit_2026-09-11.md` section 4 measured the two
MARGINALS -- a single swap takes a starter off 61% of the time and puts a
starter on 48% of the time -- and said in its own words that the joint is what a
model choice would need. This script measures the JOINT, descriptively, on both
seasons:

    P(k_in starters enter | wave size, k_out starters leave)

with the game's OWN starting five (the first on-floor set) and the game's own
participant pool, which is a measurement of coaching behaviour and is never a
bake-off result. The bake-off's own fit uses the as-of pool and the as-of
predicted starters instead (`rotation_v6.build_comp_training`), so the two are
reported side by side and never mixed.

    .venv/Scripts/python.exe scripts/diag_rotation_comp_v6.py --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import rotation as R  # noqa: E402

MAXW = 5
OUT = ROOT / "data" / "processed" / "models" / "rotation" / "comp_audit_2026-09-11.json"


def wilson(k: float, n: float) -> float:
    if n <= 0:
        return float("nan")
    p = k / n
    z = 1.96
    return float(z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n))


def season_joint_by_state(season: int) -> dict:
    """The size-1 joint split by game state, which is what a round-7 state term
    would have to carry. Same descriptive definitions as `season_joint`."""
    tp = R.load_team_possessions(season)
    cells = {}
    for _key, g in tp.groupby(["game_id", "team_id"], sort=False):
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        if lu.shape[0] < 20:
            continue
        per = g["period"].to_numpy(dtype="int64")
        clk = g["start_clock"].to_numpy(dtype="int64")
        mg = np.abs(g["margin"].to_numpy(dtype="int64"))
        starters = set(int(x) for x in lu[0])
        pids = np.unique(lu)
        on = np.zeros((lu.shape[0], len(pids)), dtype=bool)
        for s in range(5):
            on[np.arange(lu.shape[0]), np.searchsorted(pids, lu[:, s])] = True
        is_st = np.array([1.0 if int(p) in starters else 0.0 for p in pids])
        leave = on[:-1] & ~on[1:]
        enter = ~on[:-1] & on[1:]
        sz = leave.sum(axis=1)
        w = np.flatnonzero((sz == 1) & (enter.sum(axis=1) == 1))
        if not len(w):
            continue
        k = w + 1
        late = (per[k] >= 2) & (clk[k] <= 480)
        half1 = per[k] == 1
        tcell = np.where(late, 2, np.where(half1, 0, 1))
        mb = np.where(mg[k] <= 5, 0, np.where(mg[k] <= 15, 1, 2))
        ko = (leave[w] * is_st[None, :]).sum(axis=1).astype(int)
        ki = (enter[w] * is_st[None, :]).sum(axis=1).astype(int)
        for t, m, a, b in zip(tcell, mb, ko, ki):
            c = cells.setdefault((int(t), int(m), int(a)), [0, 0])
            c[0] += int(b)
            c[1] += 1
    names = {0: "H1", 1: "H2 20:00-08:00", 2: "final 8:00"}
    bands = {0: "|m|<=5", 1: "|m| 6-15", 2: "|m|>15"}
    out = []
    for (t, m, a), (num, den) in sorted(cells.items()):
        out.append({"time": names[t], "margin": bands[m], "k_out": a, "n": den,
                    "p_starter_in": round(num / den, 4) if den else float("nan"),
                    "ci": round(wilson(num, den), 4),
                    "underpowered": bool(den < 300)})
    return {"rows": out}


def season_joint(season: int) -> dict:
    tp = R.load_team_possessions(season)
    kin = np.zeros((MAXW, MAXW + 1, MAXW + 1))
    n_tg = 0
    size_hist = np.zeros(MAXW)
    for _key, g in tp.groupby(["game_id", "team_id"], sort=False):
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        if lu.shape[0] < 20:
            continue
        n_tg += 1
        starters = set(int(x) for x in lu[0])
        pids = np.unique(lu)
        idx = {int(p): i for i, p in enumerate(pids)}
        on = np.zeros((lu.shape[0], len(pids)), dtype=bool)
        for s in range(5):
            on[np.arange(lu.shape[0]), np.searchsorted(pids, lu[:, s])] = True
        is_st = np.array([1.0 if int(p) in starters else 0.0 for p in pids])
        del idx
        leave = on[:-1] & ~on[1:]
        enter = ~on[:-1] & on[1:]
        sz = leave.sum(axis=1)
        w = np.flatnonzero((sz > 0) & (sz == enter.sum(axis=1)) & (sz <= MAXW))
        if not len(w):
            continue
        ko = (leave[w] * is_st[None, :]).sum(axis=1).astype(int)
        ki = (enter[w] * is_st[None, :]).sum(axis=1).astype(int)
        np.add.at(kin, (sz[w] - 1, ko, ki), 1.0)
        np.add.at(size_hist, np.clip(sz[w], 1, MAXW) - 1, 1.0)
    return {"kin": kin, "team_games": n_tg, "size_hist": size_hist}


def sim_check(n_games: int, seed: int) -> dict:
    """Does an arm REPRODUCE the joint it was built on? Simulate the same games
    under W4 (round 5's unconditional race) and K1 (the conditional class count)
    and measure the size-1 joint the same way on both, plus on the ACTUAL
    sequence of the same games graded with the SAME as-of starter set, so all
    three columns are like for like."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_rotation_v5 as V5T
    import train_rotation_v6 as V6T
    args_d = {"test_games": n_games, "seeds": 1, "noise_seeds": 3, "noise_games": 30,
              "wave_team_games": 6000, "min_prior_games": 3, "fit_seed": 11,
              "floor_fit_seed": 101, "floor_seed": 23, "workers": 1,
              "skip_floor_a": True, "skip_floor_b": True, "skip_fit": True,
              "mode": "bakeoff", "smoke": False, "tag": "simcheck"}
    c = V6T.ensure_ctx(args_d)
    out = {}
    for name in ("W4_wave_rank_draw", "K1_cond_class", "A1_tier_affinity"):
        num = np.zeros((2, 2))
        for tag, gl in c["games_by_window"].items():
            if not gl:
                continue
            pw = V5T.priors_for("S1", tag)
            arm = V6T.make_arm6(name, tag, "")
            for g in gl:
                keys = [k for k in c["keys_by_game"][g] if k in pw]
                if len(keys) != 2:
                    continue
                rng = R.game_stream(seed, g)
                for key in keys:
                    pr, sc = pw[key], c["scripts"][key]
                    lu, _fh = arm.simulate(pr, sc, rng)
                    st = set(int(x) for x in pr.pids[pr.starters()[:5]])
                    num += _joint_size1(lu, st)
        out[name] = num
    # ACTUAL on the same games, with the same as-of starter set
    num = np.zeros((2, 2))
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
        num += _joint_size1(g[R.SLOTS].to_numpy(dtype="int64"), st)
    out["ACTUAL_asof_starters"] = num
    return out


def _joint_size1(lu: np.ndarray, starters: set) -> np.ndarray:
    """(k_out, k_in) counts over single swaps, k in {0, 1} starters."""
    num = np.zeros((2, 2))
    prev = set(int(x) for x in lu[0])
    for i in range(1, lu.shape[0]):
        cur = set(int(x) for x in lu[i])
        if cur == prev:
            continue
        off, on_ = prev - cur, cur - prev
        if len(off) == 1 and len(on_) == 1:
            a = 1 if next(iter(off)) in starters else 0
            b = 1 if next(iter(on_)) in starters else 0
            num[a, b] += 1
        prev = cur
    return num


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--by-state", action="store_true")
    ap.add_argument("--sim-check", action="store_true")
    ap.add_argument("--sim-games", type=int, default=200)
    ap.add_argument("--sim-seed", type=int, default=0)
    args = ap.parse_args()
    if args.sim_check:
        res = sim_check(args.sim_games, args.sim_seed)
        prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        block = {}
        print(f"--- realised size-1 joint, {args.sim_games} games, seed {args.sim_seed}")
        print("    (as-of predicted starters on BOTH sides, so like for like)")
        for k, m in res.items():
            p0 = m[0, 1] / max(m[0].sum(), 1)
            p1 = m[1, 1] / max(m[1].sum(), 1)
            block[k] = {"p_starter_in_given_bench_out": round(float(p0), 4),
                        "p_starter_in_given_starter_out": round(float(p1), 4),
                        "n_bench_out": int(m[0].sum()), "n_starter_out": int(m[1].sum())}
            print(f"  {k:<24} P(starter in | bench out) {p0:.4f}  "
                  f"P(starter in | starter out) {p1:.4f}  "
                  f"n {int(m[0].sum())}/{int(m[1].sum())}")
        prev["sim_check_size1"] = block
        OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print("wrote", OUT)
        return
    if args.by_state:
        prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        for s in args.seasons:
            r = season_joint_by_state(s)
            prev.setdefault(str(s), {})["by_state_size1"] = r["rows"]
            print(f"--- season {s}: P(a starter enters | size 1, k_out) by state")
            for row in r["rows"]:
                flag = "  UNDERPOWERED" if row["underpowered"] else ""
                print(f"  {row['time']:<16} {row['margin']:<9} k_out {row['k_out']} "
                      f"n {row['n']:>7} p {row['p_starter_in']:.4f} "
                      f"+/-{row['ci']:.4f}{flag}")
        OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print("wrote", OUT)
        return
    out = {}
    for s in args.seasons:
        r = season_joint(s)
        kin = r["kin"]
        rows = []
        for size in range(1, MAXW + 1):
            for ko in range(size + 1):
                n = kin[size - 1, ko, :size + 1].sum()
                if n <= 0:
                    continue
                p = kin[size - 1, ko, :size + 1] / n
                rows.append({"size": size, "k_out": ko, "n": int(n),
                             "underpowered": bool(n < 300),
                             "p_k_in": [round(float(x), 4) for x in p],
                             "ci_halfwidth_top": round(wilson(float(p.max() * n), n), 4)})
        out[str(s)] = {"team_games": r["team_games"],
                       "waves": int(kin.sum()),
                       "size_hist": [int(x) for x in r["size_hist"]],
                       "rows": rows}
        print(f"--- season {s}: {r['team_games']} team-games, {int(kin.sum())} waves")
        for row in rows:
            if row["size"] <= 3:
                flag = "  UNDERPOWERED" if row["underpowered"] else ""
                print(f"  size {row['size']} k_out {row['k_out']} n {row['n']:>7} "
                      f"P(k_in) {row['p_k_in']}{flag}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
