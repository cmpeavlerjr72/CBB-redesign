"""grade_free_throw_newcomer_v1.py -- ONE blind grader for every arm of free_throw experiments.md section 13.2 (lane I, 2026-10-01).

Reads results/laneI_1001/ft/preds_<arm>_<fold>_s<seed>.parquet + meta_*.json (train_free_throw_v3_newcomer.py) and applies
the registered rule: primary F2 log loss vs N0, floor max(|N0s1 - N0| F2, 0.000147); fold 1 same sign and no loss beyond the
floor; FT.score gates (meta), Decision 8 slope band [0.8, 1.2]; newcomer guard (|gap| growth <= 0.25 pp); simplest winner
within the floor of the best (N2 < N1 < N3). Segments: has_prior 0/1, newcomers by as-of attempts, November,
as-of FT% quintile and prior-season FT% quintile (responsiveness: predicted span / realised span).

    .venv/Scripts/python.exe scripts/grade_free_throw_newcomer_v1.py --out results/laneI_1001/ft/grade_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("results/laneI_1001/ft")
ORDER = ["N2", "N1", "N3"]
REG_FLOOR = 0.000147


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def seg(df, m):
    x = df[m]
    if not len(x):
        return None
    return {"n": int(len(x)), "log_loss": ll(x["y"].to_numpy(), x["p"].to_numpy()), "mean_p": float(x["p"].mean()),
            "mean_y": float(x["y"].mean()), "gap_pp": 100 * float(x["p"].mean() - x["y"].mean()),
            "underpowered": bool(len(x) < 1000)}


def span(df, col, mask=None):
    x = df if mask is None else df[mask]
    q = pd.qcut(x[col].rank(method="first"), 5, labels=False)
    g = x.groupby(q)[["p", "y"]].mean()
    return {"pred_by_q": [round(float(v), 4) for v in g["p"]], "act_by_q": [round(float(v), 4) for v in g["y"]],
            "slope": float((g["p"].iloc[-1] - g["p"].iloc[0]) / (g["y"].iloc[-1] - g["y"].iloc[0]))}


def lines(df):
    nov = pd.to_datetime(df["game_date"]).dt.month.to_numpy() == 11
    hp = df["has_prior"].to_numpy() > 0.5
    fa = df["fta_asof"].to_numpy()
    out = {"log_loss": ll(df["y"].to_numpy(), df["p"].to_numpy()), "n": int(len(df)),
           "all": seg(df, np.ones(len(df), bool)), "has_prior_0": seg(df, ~hp), "has_prior_1": seg(df, hp),
           "new_asof_0": seg(df, ~hp & (fa == 0)), "new_asof_1_10": seg(df, ~hp & (fa >= 1) & (fa <= 10)),
           "new_asof_11_30": seg(df, ~hp & (fa >= 11) & (fa <= 30)), "new_asof_31p": seg(df, ~hp & (fa > 30)),
           "november": seg(df, nov), "resp_ft_asof_q": span(df, "ft_asof", fa > 0),
           "resp_prior_ft_q": span(df, "prior_ft", hp)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"cells": {}}
    for f in sorted(D.glob("preds_*.parquet")):
        _, arm, fold, s = f.stem.split("_")
        meta = json.loads((D / f"meta_{arm}_{fold}_{s}.json").read_text())
        res["cells"][f"{arm}|{fold}|{s}"] = {**lines(pd.read_parquet(f)), "meta": {k: meta[k] for k in (
            "calib_pass", "calib_worst_gap_pp", "resp_pass", "slope_ratio", "features", "fit_s")}}
    C = res["cells"]

    def L(arm, fold, s="s0"):
        c = C.get(f"{arm}|{fold}|{s}")
        return None if c is None else c["log_loss"]
    s1 = L("N0", "F2", "s1")
    res["seed_spread_F2"] = None if s1 is None else abs(s1 - L("N0", "F2"))
    fl = max(REG_FLOOR, res["seed_spread_F2"] or 0.0)
    res["floor"] = fl
    dec = {}
    for arm in ORDER:
        if L(arm, "F2") is None:
            dec[arm] = "NOT RUN"
            continue
        d2 = L(arm, "F2") - L("N0", "F2")
        d1 = None if L(arm, "F1") is None or L("N0", "F1") is None else L(arm, "F1") - L("N0", "F1")
        c2 = C[f"{arm}|F2|s0"]
        gates = all(C[f"{arm}|{f}|s0"]["meta"]["calib_pass"] and C[f"{arm}|{f}|s0"]["meta"]["resp_pass"]
                    and 0.8 <= C[f"{arm}|{f}|s0"]["meta"]["slope_ratio"] <= 1.2
                    for f in ("F2", "F1") if f"{arm}|{f}|s0" in C)
        guard = all(abs(C[f"{arm}|{f}|s0"]["has_prior_0"]["gap_pp"]) - abs(C[f"N0|{f}|s0"]["has_prior_0"]["gap_pp"]) <= 0.25
                    for f in ("F2", "F1") if f"{arm}|{f}|s0" in C and f"N0|{f}|s0" in C)
        win = (d2 < -fl) and (d1 is not None and d1 < 0 and d1 <= fl) and gates and guard
        dec[arm] = {"d_F2": d2, "floors_F2": d2 / fl, "d_F1": d1, "floors_F1": None if d1 is None else d1 / fl,
                    "gates": gates, "newcomer_guard": guard, "wins": bool(win),
                    "newcomer_gap_F2_pp": c2["has_prior_0"]["gap_pp"]}
    winners = [k for k in ORDER if isinstance(dec.get(k), dict) and dec[k]["wins"]]
    if winners:
        best = min(L(k, "F2") for k in winners)
        pick = next(k for k in ORDER if k in winners and L(k, "F2") - best <= fl)
    else:
        pick = "N0"
    res["decisions"] = dec
    res["winner"] = pick
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"floor {fl:.6f}; winner {pick}")
    for k, v in C.items():
        print(f"{k:14s} ll {v['log_loss']:.6f} newc gap {v['has_prior_0']['gap_pp']:+.2f} asof0 {v['new_asof_0']['gap_pp']:+.2f} "
              f"nov {v['november']['gap_pp']:+.2f} calib {v['meta']['calib_worst_gap_pp']} slope {v['meta']['slope_ratio']:.3f} "
              f"respPrior {v['resp_prior_ft_q']['slope']:.3f}")
    for k, v in dec.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
