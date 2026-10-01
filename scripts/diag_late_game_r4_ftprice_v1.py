"""diag_late_game_r4_ftprice_v1.py -- late-game ROUND 4 (c): price the leading-team late FT make defect.

experiments.md 9.5. NOT an arm, never applied to any sim output: a first-order re-scoring of a run's
possession log for Decision 11 bookkeeping. For every period-2 possession in the final 2:00 whose offence
LEADS, by lead band (1-3, 4-6, 7+) x clock bucket where the sim's FT make is below the 2024-25 actual, each
missed FT becomes a make with probability (act - sim) / (1 - sim) on a fixed seeded stream (20 replicates
averaged); the extra points are carried to the end-of-regulation margin with NO behavioural response.
Reported: P(0), P(1), P(0)/P(1) before and after, and the change with a game-bootstrap SE.

    .venv/Scripts/python.exe scripts/diag_late_game_r4_ftprice_v1.py --run lg4_R9_s25 --out results/late_game/round4/ftprice_R9.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/engine_v0"
BANDS = ((1, 3), (4, 6), (7, 99))
BUCKETS = ((60, 120), (30, 60), (10, 30), (-1, 10))


def cells(off_margin, sec):
    b = np.full(len(sec), -1)
    for i, (lo, hi) in enumerate(BANDS):
        for j, (blo, bhi) in enumerate(BUCKETS):
            b[(off_margin >= lo) & (off_margin <= hi) & (sec > blo) & (sec <= bhi)] = i * 4 + j
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    d = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2025.parquet",
                        columns=["period", "start_clock", "start_score_diff", "fta", "ftm"])
    d = d[(d["period"] == 2) & (d["start_clock"] <= 120)]
    ca = cells(d["start_score_diff"].to_numpy(), d["start_clock"].to_numpy())
    act = {k: (d["ftm"][ca == k].sum() / max(d["fta"][ca == k].sum(), 1), int(d["fta"][ca == k].sum())) for k in range(12)}

    t = pd.read_parquet(RES / a.run / "tap_poss.parquet")
    if "per" in t:
        t = t[t["per"] == 2]
    t = t.sort_values(["game_id", "seed", "sec"], ascending=[True, True, False]).reset_index(drop=True)
    sgn = np.where(t["off"].to_numpy() == 0, 1, -1)
    hm = (t["hp"] - t["ap"]).to_numpy()
    om = hm * sgn
    cs = cells(om, t["sec"].to_numpy())
    cs[t["sec"].to_numpy() > 120] = -1
    sim = {k: (t["d_ftm"][cs == k].sum() / max(t["d_fta"][cs == k].sum(), 1), int(t["d_fta"][cs == k].sum())) for k in range(12)}
    conv = np.zeros(13)
    table = []
    for k in range(12):
        s, a_ = sim[k][0], act[k][0]
        c = max(0.0, (a_ - s) / (1 - s)) if s < 1 else 0.0
        conv[k] = c
        table.append({"cell": f"lead {BANDS[k // 4]} sec {BUCKETS[k % 4]}", "sim_ft": s, "sim_fta": sim[k][1],
                      "act_ft": a_, "act_fta": act[k][1], "conversion": c})
    g = pd.read_parquet(RES / a.run / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts", "n_periods"])
    last = t.groupby(["game_id", "seed"]).tail(1)
    lsg = np.where(last["off"].to_numpy() == 0, 1, -1)
    endm = pd.DataFrame({"game_id": last["game_id"].to_numpy(), "seed": last["seed"].to_numpy(),
                         "m_end": (last["hp"] - last["ap"]).to_numpy() + lsg * (last["d_pts_off"] - last["d_pts_def"]).to_numpy()})
    g = g.merge(endm, on=["game_id", "seed"], how="left")
    ot = (g["n_periods"] > 2).to_numpy()
    m0 = np.where(ot, 0, (g["home_pts"] - g["away_pts"]).to_numpy())
    miss = (t["d_fta"] - t["d_ftm"]).to_numpy()
    p_row = conv[np.where(cs >= 0, cs, 12)]
    key = pd.MultiIndex.from_arrays([t["game_id"], t["seed"]])
    gi = pd.MultiIndex.from_arrays([g["game_id"], g["seed"]])
    pos = gi.get_indexer(key)
    rng = np.random.default_rng(20261001)
    p0n, p1n = [], []
    for _ in range(a.reps):
        extra = rng.binomial(miss, p_row) * sgn
        add = np.bincount(pos, weights=extra, minlength=len(g))
        m1 = m0 + add
        p0n.append(m1 == 0)
        p1n.append(np.abs(m1) == 1)
    P0n, P1n = np.mean(p0n, axis=0), np.mean(p1n, axis=0)
    P00, P10 = (m0 == 0).astype(float), (np.abs(m0) == 1).astype(float)
    pg = pd.DataFrame({"game_id": g["game_id"], "p00": P00, "p10": P10, "p0n": P0n, "p1n": P1n}).groupby("game_id").sum()
    A = pg.to_numpy()
    r = np.random.default_rng(7)
    bs = []
    for _ in range(1000):
        s = A[r.integers(0, len(A), len(A))].sum(0)
        bs.append([(s[2] - s[0]) / len(g), s[2] / s[3] - s[0] / s[1]])
    bs = np.array(bs)
    out = {"run": a.run, "cells": table,
           "before": {"P0": float(P00.mean()), "P1": float(P10.mean()), "ratio": float(P00.mean() / P10.mean())},
           "after": {"P0": float(P0n.mean()), "P1": float(P1n.mean()), "ratio": float(P0n.mean() / P1n.mean())},
           "dP0": float(P0n.mean() - P00.mean()), "se_dP0": float(bs[:, 0].std()),
           "d_ratio": float(P0n.mean() / P1n.mean() - P00.mean() / P10.mean()), "se_d_ratio": float(bs[:, 1].std()),
           "label": "first-order re-scoring, no behavioural response; an estimate, never applied"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for c in table:
        print(f"  {c['cell']:32s} sim {c['sim_ft']:.3f} ({c['sim_fta']}) act {c['act_ft']:.3f} ({c['act_fta']}) conv {c['conversion']:.3f}")
    print({k: out[k] for k in ("before", "after", "dP0", "se_dP0", "d_ratio", "se_d_ratio")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
