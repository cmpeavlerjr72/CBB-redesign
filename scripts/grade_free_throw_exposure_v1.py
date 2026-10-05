"""grade_free_throw_exposure_v1.py -- ONE blind grader for every arm of free_throw experiments.md section 18.

Reads results/ft_exposure/preds_<arm>_<fold>_s<seed>.parquet + meta_*.json (train_free_throw_v4_exposure.py) and applies
the registered rule: primary = F2 |window calibration gap| (days_since_start <= 14); gap floor = max(0.25 pp,
2 x |X0s1 - X0| window gap); log-loss floor = max(0.000147, |X0s1 - X0| F2 log loss); gates from meta (FT.score
calibration + responsiveness, Decision 8 slope [0.8, 1.2]); vetoes: F2 log loss, F2 newcomer |gap| +0.25 pp, d46+ |gap|
+0.25 pp either fold; fold 1 window |gap| may not grow beyond the gap floor; simplest winner within the gap floor of the
best (X1 < X2 < X3).

    .venv/Scripts/python.exe scripts/grade_free_throw_exposure_v1.py --out results/ft_exposure/grade_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("results/ft_exposure")
ORDER = ["X1", "X2", "X3"]
REG_LL_FLOOR = 0.000147
GAP_MIN = 0.25


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
    dss = df["days_since_start"].to_numpy()
    w, mid, late = dss <= 14, (dss > 14) & (dss <= 45), dss > 45
    hp = df["has_prior"].to_numpy() > 0.5
    nov = pd.to_datetime(df["game_date"]).dt.month.to_numpy() == 11
    fa = df["fta_asof"].to_numpy()
    return {"log_loss": ll(df["y"].to_numpy(), df["p"].to_numpy()), "n": int(len(df)),
            "all": seg(df, np.ones(len(df), bool)), "W": seg(df, w), "d15_45": seg(df, mid), "d46p": seg(df, late),
            "W_new": seg(df, w & ~hp), "W_ret": seg(df, w & hp), "has_prior_0": seg(df, ~hp),
            "has_prior_1": seg(df, hp), "november": seg(df, nov),
            "resp_ft_asof_q": span(df, "ft_asof", fa > 0), "resp_prior_ft_q": span(df, "prior_ft", hp)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C = {}
    for f in sorted(D.glob("preds_*.parquet")):
        _, arm, fold, s = f.stem.split("_")
        meta = json.loads((D / f"meta_{arm}_{fold}_{s}.json").read_text())
        C[f"{arm}|{fold}|{s}"] = {**lines(pd.read_parquet(f)), "meta": {k: meta[k] for k in (
            "calib_pass", "calib_worst_gap_pp", "resp_pass", "slope_ratio", "features", "coverage")}}
    res = {"cells": C}

    def g(arm, fold, key, s="s0"):
        return C[f"{arm}|{fold}|{s}"][key]

    ll_spread = abs(g("X0", "F2", "log_loss", "s1") - g("X0", "F2", "log_loss"))
    gap_spread = abs(g("X0", "F2", "W", "s1")["gap_pp"] - g("X0", "F2", "W")["gap_pp"])
    fl_ll, fl_gap = max(REG_LL_FLOOR, ll_spread), max(GAP_MIN, 2 * gap_spread)
    res.update(ll_floor=fl_ll, gap_floor_pp=fl_gap, ll_seed_spread=ll_spread, gap_seed_spread_pp=gap_spread)
    dec = {}
    for arm in ORDER:
        if f"{arm}|F2|s0" not in C or f"{arm}|F1|s0" not in C:
            dec[arm] = {"status": "NOT RUN"}
            continue
        r = {}
        r["F2_W_absgap"] = abs(g(arm, "F2", "W")["gap_pp"])
        r["F2_W_absgap_X0"] = abs(g("X0", "F2", "W")["gap_pp"])
        r["F2_W_improve_pp"] = r["F2_W_absgap_X0"] - r["F2_W_absgap"]
        r["F1_W_growth_pp"] = abs(g(arm, "F1", "W")["gap_pp"]) - abs(g("X0", "F1", "W")["gap_pp"])
        r["F2_ll_delta"] = g(arm, "F2", "log_loss") - g("X0", "F2", "log_loss")
        r["F1_ll_delta"] = g(arm, "F1", "log_loss") - g("X0", "F1", "log_loss")
        r["F2_new_growth_pp"] = abs(g(arm, "F2", "has_prior_0")["gap_pp"]) - abs(g("X0", "F2", "has_prior_0")["gap_pp"])
        r["d46_growth_pp"] = {f: abs(g(arm, f, "d46p")["gap_pp"]) - abs(g("X0", f, "d46p")["gap_pp"]) for f in ("F2", "F1")}
        gates = all(C[f"{arm}|{f}|s0"]["meta"]["calib_pass"] and C[f"{arm}|{f}|s0"]["meta"]["resp_pass"]
                    and 0.8 <= C[f"{arm}|{f}|s0"]["meta"]["slope_ratio"] <= 1.2 for f in ("F2", "F1"))
        vetoes = []
        if r["F2_ll_delta"] > fl_ll:
            vetoes.append("F2 log loss")
        if r["F2_new_growth_pp"] > GAP_MIN:
            vetoes.append("F2 newcomer gap")
        if max(r["d46_growth_pp"].values()) > GAP_MIN:
            vetoes.append("d46+ gap")
        r.update(gates_pass=gates, vetoes=vetoes)
        r["win"] = bool(r["F2_W_improve_pp"] > fl_gap and r["F1_W_growth_pp"] <= fl_gap and gates and not vetoes)
        dec[arm] = r
    res["decision"] = dec
    winners = [a for a in ORDER if dec.get(a, {}).get("win")]
    if winners:
        best = min(dec[a]["F2_W_absgap"] for a in winners)
        res["winner"] = next(a for a in ORDER if a in winners and dec[a]["F2_W_absgap"] - best <= fl_gap)
    else:
        res["winner"] = "X0 (no winner)"
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"floors": [fl_ll, fl_gap], "winner": res["winner"], "decision": dec}, indent=1, default=float))
    for k in sorted(C):
        c = C[k]
        print(k, f"ll {c['log_loss']:.6f}", " ".join(f"{s} {c[s]['gap_pp']:+.2f}" for s in
              ("all", "W", "W_new", "W_ret", "d15_45", "d46p", "november")),
              f"calib {c['meta']['calib_worst_gap_pp']} slope {c['meta']['slope_ratio']:.3f}",
              "respPriorQ", c["resp_prior_ft_q"]["slope"].__round__(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
