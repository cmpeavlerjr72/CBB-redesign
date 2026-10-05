"""grade_free_throw_prior_v1.py -- ONE blind grader for every arm of free_throw experiments.md section 20.

Reads results/ft_prior/preds_<arm>_<fold>_s<seed>.parquet + meta. Registered rule (s20.2):
  primary F2 full-season log loss beats P0 by > LL floor = max(0.000147, |P0s1-P0| F2 LL);
  co-primary F2 |T gap| shrinks by > T floor (T = prior-season FTA 1-40, days 0-45; floor = max(0.25pp, 2|P0s1-P0|));
  F1 confirms (LL not worse beyond floor, |T gap| not growing);
  vetoes either fold: any of 12 cells (prior FTA 0/1-10/11-40/41+ x d0-14/d15-45/d46+) |gap| grows > its cell floor,
  FT.score calibration / responsiveness gates, Decision 8 slope [0.8,1.2], prior-FT% quintile span ratio in [0.8,1.2];
  simplest winner within the LL floor of the best (P1 < P2 < P3).

    .venv/Scripts/python.exe scripts/grade_free_throw_prior_v1.py --out results/ft_prior/grade_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("results/ft_prior")
ORDER = ["P1", "P2", "P3"]
REG_LL = 0.000147
GAP_MIN = 0.25
FB = ["0", "1-10", "11-40", "41+"]
DB = ["d0-14", "d15-45", "d46+"]


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def lines(df):
    pf, dd = df["prior_fta"].to_numpy(), df["days_since_start"].to_numpy()
    fb = np.select([pf <= 0, pf <= 10, pf <= 40], FB[:3], FB[3])
    db = np.select([dd <= 14, dd <= 45], DB[:2], DB[2])
    gap = lambda m: 100 * float(df["p"].to_numpy()[m].mean() - df["y"].to_numpy()[m].mean())  # noqa: E731
    cells = {f"{a}|{b}": {"gap": gap((fb == a) & (db == b)), "n": int(((fb == a) & (db == b)).sum())}
             for a in FB for b in DB}
    T = (pf >= 1) & (pf <= 40) & (dd <= 45)
    hp = df["has_prior"].to_numpy() > 0.5
    x = df[hp]
    q = pd.qcut(x["prior_ft"].rank(method="first"), 5, labels=False)
    g = x.groupby(q)[["p", "y"]].mean()
    span = float((g["p"].iloc[-1] - g["p"].iloc[0]) / (g["y"].iloc[-1] - g["y"].iloc[0]))
    return {"log_loss": ll(df["y"].to_numpy(), df["p"].to_numpy()), "T_gap": gap(T), "T_n": int(T.sum()),
            "all_gap": gap(np.ones(len(df), bool)), "W_gap": gap(dd <= 14), "cells": cells,
            "prior_ft_q": {"pred": g["p"].round(4).tolist(), "act": g["y"].round(4).tolist(), "span_ratio": span}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C = {}
    for f in sorted(D.glob("preds_*.parquet")):
        _, arm, fold, s = f.stem.split("_")
        meta = json.loads((D / f"meta_{arm}_{fold}_{s}.json").read_text())
        C[f"{arm}|{fold}|{s}"] = {**lines(pd.read_parquet(f)), "meta": {k: meta[k] for k in (
            "calib_pass", "calib_worst_gap_pp", "resp_pass", "slope_ratio", "features")}}
    g = lambda arm, fold, s="s0": C[f"{arm}|{fold}|{s}"]  # noqa: E731
    fl_ll = max(REG_LL, abs(g("P0", "F2", "s1")["log_loss"] - g("P0", "F2")["log_loss"]))

    def floor_of(fold, key=None):
        if key is None:
            return max(GAP_MIN, 2 * abs(g("P0", fold, "s1")["T_gap"] - g("P0", fold)["T_gap"]))
        return max(GAP_MIN, 2 * abs(g("P0", fold, "s1")["cells"][key]["gap"] - g("P0", fold)["cells"][key]["gap"]))

    res = {"ll_floor": fl_ll, "T_floor": {f: floor_of(f) for f in ("F2", "F1")}, "cells_raw": C, "arms": {}}
    winners = []
    for arm in ORDER:
        v = {"reasons": []}
        for fold in ("F2", "F1"):
            r, r0 = g(arm, fold), g("P0", fold)
            v[fold] = {"dLL": r["log_loss"] - r0["log_loss"], "T_gap": r["T_gap"], "T_gap_P0": r0["T_gap"],
                       "W_gap": r["W_gap"], "W_gap_P0": r0["W_gap"], "calib": r["meta"]["calib_worst_gap_pp"],
                       "d8_slope": r["meta"]["slope_ratio"], "span_ratio": r["prior_ft_q"]["span_ratio"]}
            for key in r["cells"]:
                grow = abs(r["cells"][key]["gap"]) - abs(r0["cells"][key]["gap"])
                if grow > floor_of(fold, key):
                    v["reasons"].append(f"{fold} cell {key} |gap| +{grow:.2f}pp > floor {floor_of(fold, key):.2f}")
            if not (r["meta"]["calib_pass"] and r["meta"]["resp_pass"]):
                v["reasons"].append(f"{fold} FT.score gate fail (calib {r['meta']['calib_worst_gap_pp']})")
            if not 0.8 <= r["meta"]["slope_ratio"] <= 1.2:
                v["reasons"].append(f"{fold} Decision 8 slope {r['meta']['slope_ratio']}")
            if not 0.8 <= r["prior_ft_q"]["span_ratio"] <= 1.2:
                v["reasons"].append(f"{fold} prior-FT quintile span {r['prior_ft_q']['span_ratio']:.3f}")
        if not v["F2"]["dLL"] < -fl_ll:
            v["reasons"].append(f"F2 dLL {v['F2']['dLL']:.6f} not beyond floor")
        if not abs(v["F2"]["T_gap_P0"]) - abs(v["F2"]["T_gap"]) > res["T_floor"]["F2"]:
            v["reasons"].append("F2 |T gap| does not shrink beyond floor")
        if v["F1"]["dLL"] > fl_ll:
            v["reasons"].append("F1 LL worse beyond floor")
        if abs(v["F1"]["T_gap"]) > abs(v["F1"]["T_gap_P0"]):
            v["reasons"].append("F1 |T gap| grows")
        v["win"] = not v["reasons"]
        if v["win"]:
            winners.append(arm)
        res["arms"][arm] = v
    if winners:
        best = min(g(w, "F2")["log_loss"] for w in winners)
        res["winner"] = next(w for w in ORDER if w in winners and g(w, "F2")["log_loss"] - best <= fl_ll)
    else:
        res["winner"] = "P0"
    res["winners"] = winners
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"LL floor {fl_ll:.6f}  T floors {res['T_floor']}")
    for fold in ("F2", "F1"):
        print(f"\n{fold} cell gaps (pp)")
        rows = {arm: {k: round(v['gap'], 2) for k, v in g(arm, fold)['cells'].items()} for arm in ["P0", *ORDER]}
        rows["P0s1"] = {k: round(v['gap'], 2) for k, v in g("P0", fold, "s1")['cells'].items()}
        print(pd.DataFrame(rows).to_string())
    for arm in ORDER:
        v = res["arms"][arm]
        print(f"\n{arm}: win={v['win']}  F2 {json.dumps({k: round(x, 6) for k, x in v['F2'].items()})}")
        print(f"     F1 {json.dumps({k: round(x, 6) for k, x in v['F1'].items()})}")
        for r in v["reasons"]:
            print("     -", r)
    print(f"\nWINNER: {res['winner']}  (winners {winners})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
