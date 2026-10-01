"""grade_d1001A_v1.py -- lane A 2026-10-01: the two lines box request d1001_A_1 is read on, for closed-loop reads.

Per read: G9 slope (OLS of verified actual margin on the sim's seed-mean margin), |1 - slope|, SD of the seed-mean
margin, margin bias. Arm minus reference, with Decision 12 floors: SD over [ref, floor draws] (unpaired seed-offset
reruns of the reference) and a paired game bootstrap (2,000 draws, games resampled, all seeds kept); floor = max(2 x
draw SD, half-width of the bootstrap 95% interval). Decides nothing.

    CBB_TRUTH=verified_v1 python scripts/grade_d1001A_v1.py --ref v3full_COMB9GCTKD_s200_o0 \
        --floors a,b,c,d --arms d1001A_TSRJ1_s200_o0 --out results/d1001A/abs_slope.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CBB_TRUTH", "verified_v1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402


def frame(results: Path, tag: str, season: int) -> pd.DataFrame:
    g = pd.read_parquet(results / tag / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts",
                                                                   "possessions", "n_periods"])
    s, _ = G.build_grading_frame(g, season)
    return s[["game_id", "margin", "sim_margin_mean"]].set_index("game_id")


def lines(y, x):
    sl = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
    return {"slope": sl, "abs_1_minus_slope": abs(1 - sl), "sd_sim_margin": float(np.std(x, ddof=1)),
            "margin_bias": float(np.mean(x - y))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results/engine_v0")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--floors", default="")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    ref = frame(a.results_dir, a.ref, a.season)
    floors = [frame(a.results_dir, t, a.season) for t in a.floors.split(",") if t]
    out = {"ref": a.ref, "floors": a.floors, "lines": {}}
    rl = lines(ref["margin"].to_numpy(), ref["sim_margin_mean"].to_numpy())
    out["ref_lines"] = rl
    draws = [rl] + [lines(f["margin"].to_numpy(), f["sim_margin_mean"].to_numpy()) for f in floors]
    rng = np.random.default_rng(20261001)
    for arm in a.arms.split(","):
        af = frame(a.results_dir, arm, a.season)
        ids = ref.index.intersection(af.index)
        y = ref.loc[ids, "margin"].to_numpy(); xr = ref.loc[ids, "sim_margin_mean"].to_numpy()
        xa = af.loc[ids, "sim_margin_mean"].to_numpy()
        la, lr = lines(y, xa), lines(y, xr)
        boots = []
        for _ in range(a.n_boot):
            i = rng.integers(0, len(ids), len(ids))
            b1, b0 = lines(y[i], xa[i]), lines(y[i], xr[i])
            boots.append({k: b1[k] - b0[k] for k in b1})
        bt = pd.DataFrame(boots)
        res = {}
        for k in la:
            mv = la[k] - lr[k]
            dsd = float(np.std([d[k] for d in draws], ddof=1)) if len(draws) > 1 else 0.0
            lo, hi = np.percentile(bt[k], [2.5, 97.5])
            fl = max(2 * dsd, (hi - lo) / 2)
            res[k] = {"arm": la[k], "ref": lr[k], "move": mv, "draw_sd": dsd, "boot_ci": [float(lo), float(hi)],
                      "floor": fl, "flag": "BEYOND" if abs(mv) > fl else "inside"}
        out["lines"][arm] = {"n_games": int(len(ids)), **res}
        print(arm, {k: (round(v["arm"], 4), round(v["move"], 4), round(v["floor"], 4), v["flag"]) for k, v in res.items()})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
