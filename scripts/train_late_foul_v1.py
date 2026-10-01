"""train_late_foul_v1.py -- late foul accrual (possession_outcome experiments.md section 29): fit + grade.

Window: period 2, possession start <= 120 s, regulation. Target K = the defence's non-trip fouls in the possession
(`def_silent`, engine-definition table `round6/foul_accrual_poss_v2.parquet`), clipped at 3.
    LF0  served LUT (`lut_acc_A2_F2`) as a Bernoulli: P(0) = 1-p, P(1) = p
    LF1  window refit, binary P(K >= 1) by cell
    LF2  window refit, count P(K = 0..3) by cell
Cells: defence role (trail 1-3 / trail 4-6 / trail 7+ / tied / leading) x clock ((0,30], (30,60], (60,120]) x
defence fouls (0-3, 4-5, 6-8, 9+); hierarchy clock -> clock x fouls -> full, m = 50.
Metric: multinomial log loss of min(K, 3), class probabilities floored at 1e-4; floor = game-block bootstrap SE.

    .venv/Scripts/python.exe scripts/train_late_foul_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from grade_late_game_v1 import block_se                              # noqa: E402

ACC = ROOT / "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet"
LUT = ROOT / "data/processed/models/possession_outcome/round7/lut_acc_A2_F2.npz"
OUT = ROOT / "data/processed/models/possession_outcome/late_foul"
FOLDS = {"F1": ([2022, 2023], 2024), "F2": ([2022, 2023, 2024], 2025)}
CLOCK_EDGES = np.array([30, 60, 120])
FOUL_EDGES = np.array([3, 5, 8])          # 0-3, 4-5, 6-8, 9+
NK, M, EPS = 4, 50.0, 1e-4


def role_code(om):
    a = np.abs(om)
    return np.where(om > 0, np.where(a <= 3, 0, np.where(a <= 6, 1, 2)), np.where(om == 0, 3, 4))


def codes(d):
    return (role_code(d["start_score_diff"].to_numpy()),
            np.searchsorted(CLOCK_EDGES, d["start_clock"].to_numpy(), side="left").clip(0, 2),
            np.searchsorted(FOUL_EDGES, d["def_team_fouls_true"].to_numpy(), side="left").clip(0, 3))


def load(seasons):
    d = pd.read_parquet(ACC)
    d = d[d["season"].isin(seasons) & (d["period"] == 2) & (d["start_clock"] <= 120) & (d["is_ot"] == 0)]
    return d.reset_index(drop=True)


def lut_p(d):
    t = np.load(LUT)
    ci = np.searchsorted(t["clock_cuts"], d["start_clock"].to_numpy(), side="right")
    mi = np.searchsorted(t["margin_cuts"], d["start_score_diff"].to_numpy(), side="right")
    m = int(t["max_fouls"])
    site = np.where(d["neutral_site"], 0, np.where(d["offense_is_home"], 1, 2))
    return t["lut"][1, ci, mi, np.clip(d["def_team_fouls_true"].to_numpy(), 0, m),
                    np.clip(d["off_team_fouls_true"].to_numpy(), 0, m), site]


def fit(tr, binary: bool):
    r, c, f = codes(tr)
    k = np.clip(tr["def_silent"].to_numpy(), 0, 3)
    if binary:
        k = np.minimum(k, 1)
    Y = np.eye(NK)[k]
    base = Y.mean(0)
    # level 1: clock
    l1 = np.zeros((3, NK))
    for ci in range(3):
        m = c == ci
        l1[ci] = (Y[m].sum(0) + M * base) / (m.sum() + M)
    l2 = np.zeros((3, 4, NK))
    for ci in range(3):
        for fi in range(4):
            m = (c == ci) & (f == fi)
            l2[ci, fi] = (Y[m].sum(0) + M * l1[ci]) / (m.sum() + M)
    l3 = np.zeros((5, 3, 4, NK))
    for ri in range(5):
        for ci in range(3):
            for fi in range(4):
                m = (r == ri) & (c == ci) & (f == fi)
                l3[ri, ci, fi] = (Y[m].sum(0) + M * l2[ci, fi]) / (m.sum() + M)
    return l3


def probs(lut, d):
    r, c, f = codes(d)
    return lut[r, c, f]


def mll(P, k):
    P = np.clip(P, EPS, None)
    P = P / P.sum(1, keepdims=True)
    return -np.log(P[np.arange(len(k)), k])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    for fold, (trs, tes) in FOLDS.items():
        tr, te = load(trs), load([tes])
        k = np.clip(te["def_silent"].to_numpy(), 0, 3)
        g = te["game_id"].to_numpy()
        p0 = lut_p(te)
        P = {"LF0": np.column_stack([1 - p0, p0, np.zeros_like(p0), np.zeros_like(p0)])}
        luts = {"LF1": fit(tr, True), "LF2": fit(tr, False)}
        for a, l in luts.items():
            P[a] = probs(l, te)
        L = {a: mll(p, k) for a, p in P.items()}
        Lb = {a: -np.log(np.clip(np.where(k >= 1, 1 - p[:, 0], p[:, 0]), EPS, 1)) for a, p in P.items()}
        rows = {}
        for a in P:
            row = {"mll": float(L[a].mean()), "binary_ll": float(Lb[a].mean()),
                   "expected_K": float((P[a] * np.arange(NK)).sum(1).mean())}
            for b in ("LF0", "LF1"):
                if a != b and not (a == "LF1" and b == "LF1"):
                    dd = L[a] - L[b]
                    se = block_se(dd, g)
                    row[f"vs_{b}"] = {"d": float(dd.mean()), "se": se, "floors": float(-dd.mean() / se)}
            rows[a] = row
        r, c, f = codes(te)
        cal = []
        for ri, rn in enumerate(("trail1-3", "trail4-6", "trail7+", "tied", "leading")):
            for ci, cn in enumerate(("(0,30]", "(30,60]", "(60,120]")):
                for fi, fn in enumerate(("0-3", "4-5", "6-8", "9+")):
                    m = (r == ri) & (c == ci) & (f == fi)
                    if m.sum() < 20:
                        continue
                    cal.append({"cell": f"{rn}|{cn}|{fn}", "n": int(m.sum()), "act_K": float(k[m].mean()),
                                "LF0": float(P["LF0"][m, 1].mean()), "LF2_K": float((P["LF2"][m] * np.arange(NK)).sum(1).mean()),
                                "label": "UNDERPOWERED" if m.sum() < 200 else ""})
        rep[fold] = {"n_train": len(tr), "n_test": len(te), "actual_mean_K": float(k.mean()), "arms": rows, "cells": cal}
        if fold == "F2":
            for a, l in luts.items():
                (OUT / f"late_foul_{a}_F2.json").write_text(json.dumps({
                    "arm": a, "probs": l.tolist(), "clock_edges": CLOCK_EDGES.tolist(), "foul_edges": FOUL_EDGES.tolist(),
                    "roles": ["def_trail_1_3", "def_trail_4_6", "def_trail_7p", "tied", "def_leading"],
                    "window": "period 2, start <= 120 s, regulation", "train_seasons": trs, "m": M}), encoding="utf-8")
    (OUT / "late_foul_grade.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    for f, r in rep.items():
        print(f"== {f}: train {r['n_train']:,} test {r['n_test']:,} actual mean K {r['actual_mean_K']:.4f}")
        for a, x in r["arms"].items():
            print(f"   {a}: mll {x['mll']:.5f} binary {x['binary_ll']:.5f} E[K] {x['expected_K']:.4f} "
                  + " ".join(f"{k}: d {v['d']:+.5f} se {v['se']:.5f} ({v['floors']:+.1f} fl)" for k, v in x.items() if k.startswith("vs_")))
        for c in r["cells"]:
            if c["cell"].split("|")[2] in ("4-5", "0-3") or c["cell"].startswith("trail1-3"):
                print(f"   {c['cell']:26s} n={c['n']:5d} act K {c['act_K']:.3f} LF0 p {c['LF0']:.3f} LF2 E[K] {c['LF2_K']:.3f} {c['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
