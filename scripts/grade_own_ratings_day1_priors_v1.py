#!/usr/bin/env python
"""
grade_own_ratings_day1_priors_v1.py -- Lane N (2026-09-30): one blind grader for every arm of the own_ratings day-1
prior bake-off (docs/models/own_ratings/experiments.md section 1.4-1.5). Reads results/own_ratings_day1/preds_v1.parquet
(keyed by arm) and writes grade_v1.json and grade_v1_tables.md next to it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
OUT = REPO / "results/own_ratings_day1"
ARMS = ["Z", "R", "F", "C", "K", "D"]
BANDS = ["0-1", "2-3", "4-7", "all"]
SIG_GRID = np.round(np.arange(5.0, 20.0001, 0.1), 1)
N_BOOT = 1000
SEED = 20260930


def band_of(week: int) -> str:
    return "0-1" if week <= 1 else ("2-3" if week <= 3 else "4-7")


def logloss(pred, win, sig):
    p = np.clip(norm.cdf(pred / sig), 1e-12, 1 - 1e-12)
    return -(win * np.log(p) + (1 - win) * np.log(1 - p))


def fit_sigma(pred, win):
    ll = [logloss(pred, win, s).mean() for s in SIG_GRID]
    return float(SIG_GRID[int(np.argmin(ll))])


def team_boot(diff: np.ndarray, home: np.ndarray, away: np.ndarray, rng) -> tuple[float, float, float]:
    teams, inv = np.unique(np.concatenate([home, away]), return_inverse=True)
    n = len(diff)
    ih, ia = inv[:n], inv[n:]
    T = len(teams)
    dsum = np.bincount(ih, 0.5 * diff, T) + np.bincount(ia, 0.5 * diff, T)
    gsum = np.bincount(ih, np.full(n, 0.5), T) + np.bincount(ia, np.full(n, 0.5), T)
    point = dsum.sum() / gsum.sum()
    draws = rng.integers(0, T, size=(N_BOOT, T))
    cnt = np.zeros((N_BOOT, T))
    for b in range(N_BOOT):
        cnt[b] = np.bincount(draws[b], minlength=T)
    st = (cnt @ dsum) / (cnt @ gsum)
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def quintile_slope(df: pd.DataFrame, day1: pd.DataFrame) -> float:
    """Team-perspective: mean predicted vs mean realised margin per quintile of the R day-1 net rating."""
    h = df[["home_team_id", "pred", "margin"]].rename(columns={"home_team_id": "team_id"})
    a = df[["away_team_id", "pred", "margin"]].rename(columns={"away_team_id": "team_id"})
    a = a.assign(pred=-a["pred"], margin=-a["margin"])
    t = pd.concat([h, a]).merge(day1[["team_id", "q"]], on="team_id")
    g = t.groupby("q")[["pred", "margin"]].mean()
    if len(g) < 5:
        return float("nan")
    return float(np.polyfit(g["pred"], g["margin"], 1)[0])


def main() -> int:
    pr = pd.read_parquet(OUT / "preds_v1.parquet")
    day1 = pd.read_parquet(OUT / "day1_v1.parquet")
    pr["band"] = pr["week"].map(band_of)
    pr["win"] = (pr["margin"] > 0).astype(float)
    ln = pd.read_parquet("data/processed/lines/lines_close_v2_verified.parquet", columns=["game_id", "close_spread_home"])
    close = (-ln.groupby("game_id")["close_spread_home"].median()).rename("close_margin")
    pr = pr.merge(close, left_on="game_id", right_index=True, how="left")
    rng = np.random.default_rng(SEED)
    res = {"sigma": {}, "metrics": [], "paired": [], "close_benchmark": {}}

    for fold in ["F1", "F2"]:
        f = pr[pr["fold"] == fold]
        te_season = int(f.loc[f["role"] == "test", "season"].iloc[0])
        d1 = day1[day1["season"] == te_season].copy()
        d1["q"] = pd.qcut(d1["net_R_day1"].rank(method="first"), 5, labels=False)
        # sigma per arm and band, training rows only
        sig = {}
        for arm in ARMS:
            tr = f[(f["role"] == "train") & (f["arm"] == arm)]
            for band in BANDS[:3]:
                m = tr["band"] == band
                sig[(arm, band)] = fit_sigma(tr.loc[m, "pred"].to_numpy(), tr.loc[m, "win"].to_numpy())
        res["sigma"][fold] = {f"{a}|{b}": v for (a, b), v in sig.items()}
        te = f[f["role"] == "test"].copy()
        te["sig"] = [sig[(a, b)] for a, b in zip(te["arm"], te["band"])]
        te["ae"] = (te["pred"] - te["margin"]).abs()
        te["ll"] = logloss(te["pred"].to_numpy(), te["win"].to_numpy(), te["sig"].to_numpy())
        te["ae_close"] = (te["pred"] - te["close_margin"]).abs()
        base = te[te["arm"] == "R"].set_index("game_id")
        for band in BANDS:
            for cell in ["all", "new_d1", "old"]:
                for arm in ARMS:
                    x = te[te["arm"] == arm]
                    if band != "all":
                        x = x[x["band"] == band]
                    if cell == "new_d1":
                        x = x[x["new_d1"]]
                    elif cell == "old":
                        x = x[~x["new_d1"]]
                    if len(x) == 0:
                        continue
                    hasl = x["close_margin"].notna()
                    slope_g = float(np.polyfit(x["pred"], x["margin"], 1)[0]) if len(x) > 2 else float("nan")
                    row = {"fold": fold, "band": band, "cell": cell, "arm": arm, "n": int(len(x)),
                           "mae": float(x["ae"].mean()), "logloss": float(x["ll"].mean()),
                           "mae_vs_close": float(x.loc[hasl, "ae_close"].mean()) if hasl.any() else float("nan"),
                           "n_close": int(hasl.sum()), "slope_game": slope_g,
                           "slope_quintile": quintile_slope(x, d1) if cell == "all" else float("nan")}
                    res["metrics"].append(row)
                    if arm != "R" and cell in ("all", "new_d1"):
                        b = base.loc[x["game_id"]]
                        for met, col in (("mae", "ae"), ("logloss", "ll"), ("mae_vs_close", "ae_close")):
                            xx = x.set_index("game_id")
                            ok = xx[col].notna().to_numpy() & b[col].notna().to_numpy()
                            if ok.sum() < 3:
                                continue
                            dd = xx[col].to_numpy()[ok] - b[col].to_numpy()[ok]
                            pt, lo, hi = team_boot(dd, xx["home_team_id"].to_numpy()[ok], xx["away_team_id"].to_numpy()[ok], rng)
                            res["paired"].append({"fold": fold, "band": band, "cell": cell, "arm": arm, "metric": met,
                                                  "diff": pt, "lo": lo, "hi": hi, "n": int(ok.sum())})
            xb = base if band == "all" else base[base["band"] == band]
            hl = xb["close_margin"].notna()
            res["close_benchmark"][f"{fold}|{band}"] = {"close_mae_vs_final": float((xb.loc[hl, "close_margin"] - xb.loc[hl, "margin"]).abs().mean()) if hl.any() else None,
                                                         "n": int(hl.sum())}
    (OUT / "grade_v1.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # ---- decision rule (section 1.5) ----
    P = pd.DataFrame(res["paired"])
    M = pd.DataFrame(res["metrics"])
    rank = {"Z": 0, "R": 1, "F": 2, "C": 3, "K": 4, "D": 5}
    q = []
    for arm in ARMS:
        if arm == "R":
            continue
        g = lambda fo, me: P[(P.fold == fo) & (P.band == "all") & (P.cell == "all") & (P.arm == arm) & (P.metric == me)].iloc[0]
        f2, f1, ll2 = g("F2", "mae"), g("F1", "mae"), g("F2", "logloss")
        ok = (f2.hi < 0) and (f1["diff"] < 0) and not (ll2.lo > 0)
        q.append({"arm": arm, "f2_diff": f2["diff"], "f2_lo": f2.lo, "f2_hi": f2.hi, "f1_diff": f1["diff"], "ll_lo": ll2.lo, "qualifies": bool(ok)})
    Q = pd.DataFrame(q)
    quals = Q[Q.qualifies].sort_values("f2_diff")
    winner = "R (none qualifies)"
    if len(quals):
        best = quals.iloc[0]
        simpler = quals[quals.arm.map(rank) < rank[best.arm]]
        winner = best.arm
        for _, s in simpler.sort_values("arm", key=lambda a: a.map(rank)).iterrows():
            if best.f2_lo <= s.f2_diff <= best.f2_hi:
                winner = s.arm
                break
    res["decision"] = {"table": Q.to_dict("records"), "winner": winner}
    (OUT / "grade_v1.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(Q.to_string())
    print("WINNER:", winner)
    pd.set_option("display.width", 250)
    print(M[M.cell == "all"].pivot_table(index=["fold", "arm"], columns="band", values=["mae", "logloss"]).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
