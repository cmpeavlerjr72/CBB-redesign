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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    args = ap.parse_args()
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
