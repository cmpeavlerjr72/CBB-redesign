#!/usr/bin/env python
"""grade_foul_r9_v1.py -- ONE blind grader for foul round 9 (possession_outcome
experiments.md section 26). Scores every arm column of round9/preds_{ao,trip}_{fold}_seed{s}
the same way; it knows no arm by name except the block references (AO0, T2c).

Per block, fold, arm: test log loss (AO: y_ao; T: y_bonus + y_shoot summed), delta vs the
block reference in APPLIED floors (applied floor = max(|seed0 - seed7| measured per arm,
0.000804)), a paired game-bootstrap SE of the delta (200 draws), calibration gap pp by half,
minute bucket, class / rule state, site; team SD ratio and prior-quintile slope (responsiveness).

    .venv/Scripts/python.exe scripts/grade_foul_r9_v1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data/processed/models/possession_outcome/round9")
FLOOR_MIN = 0.000804
NB = 200


def ll_vec(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_se(game_ids, lvec_delta, rng):
    g = pd.Series(lvec_delta).groupby(pd.Series(game_ids)).agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    k = len(s)
    out = []
    for _ in range(NB):
        i = rng.integers(0, k, k)
        out.append(s[i].sum() / n[i].sum())
    return float(np.std(out))


def calib(df, ycols, pcols, by):
    g = df.groupby(by, observed=True)
    res = {}
    for y, p in zip(ycols, pcols):
        t = (g[p].mean() - g[y].mean()) * 100
        res[y] = {str(k): round(float(v), 3) for k, v in t.items()}
    res["n"] = {str(k): int(v) for k, v in g.size().items()}
    return res


def team_stats(df, ycol, pcol, team_col, prior_col=None, min_n=300):
    g = df.groupby(team_col).agg(y=(ycol, "mean"), p=(pcol, "mean"), n=(ycol, "size"))
    g = g[g["n"] >= min_n]
    sd_ratio = float(g["p"].std() / g["y"].std()) if len(g) > 2 else float("nan")
    out = {"team_sd_ratio": sd_ratio, "n_teams": int(len(g))}
    if prior_col is not None:
        pr = df.groupby(team_col)[prior_col].first().reindex(g.index)
        q = pd.qcut(pr.rank(method="first"), 5, labels=False)
        t = g.groupby(q)[["y", "p"]].mean()
        span_y = float(t["y"].iloc[-1] - t["y"].iloc[0])
        span_p = float(t["p"].iloc[-1] - t["p"].iloc[0])
        out.update({"quintile_y": t["y"].round(5).tolist(), "quintile_p": t["p"].round(5).tolist(),
                    "slope_ratio": span_p / span_y if span_y != 0 else float("nan")})
    return out


def grade_block(block, ycols, ref):
    rng = np.random.default_rng(20260930)
    res = {}
    for fold in ("F1", "F2"):
        f0 = pd.read_parquet(D / f"preds_{block}_{fold}_seed0.parquet")
        f7 = pd.read_parquet(D / f"preds_{block}_{fold}_seed7.parquet")
        arms = sorted({c.split("__")[1] for c in f0.columns if "__" in c})
        lv0 = {a: sum(ll_vec(f0[y].to_numpy(float), f0[f"{y}__{a}"].to_numpy(float)) for y in ycols) for a in arms}
        lv7 = {a: sum(ll_vec(f7[y].to_numpy(float), f7[f"{y}__{a}"].to_numpy(float)) for y in ycols) for a in arms}
        floors = {a: max(abs(float(lv0[a].mean() - lv7[a].mean())), FLOOR_MIN) for a in arms}
        rf = {}
        for a in arms:
            fl = max(floors[a], floors[ref])
            dl = lv0[ref] - lv0[a]           # positive = arm better
            rf[a] = {"logloss": float(lv0[a].mean()), "seed_floor_measured": abs(float(lv0[a].mean() - lv7[a].mean())),
                     "applied_floor": fl, "delta_vs_ref": float(dl.mean()), "floors_vs_ref": float(dl.mean() / fl),
                     "boot_se_delta": boot_se(f0["game_id"].to_numpy(), dl, rng) if a != ref else 0.0}
        res[fold] = {"arms": rf}
        if fold == "F2":
            df = f0.copy()
            df["minute_b"] = pd.cut(np.where(df["period"] <= 2, (df["period"] - 1) * 20
                                             + (1200 - df["sec_rem" if "sec_rem" in df else "start_clock"]) / 60, np.nan),
                                    [0, 5, 10, 15, 20, 25, 30, 35, 38, 40.001], right=False).astype(str)
            seg = {}
            for a in arms:
                pc = [f"{y}__{a}" for y in ycols]
                sa = {"half": calib(df, ycols, pc, "half"), "minute_b": calib(df, ycols, pc, "minute_b")}
                if block == "ao":
                    sa["cls"] = calib(df, ycols, pc, "cls")
                    df["site"] = np.where(df["site_home"] > 0, "home", np.where(df["site_away"] > 0, "away", "neutral"))
                    sa["site"] = calib(df, ycols, pc, "site")
                    sa["state"] = calib(df.assign(st=np.clip(df["def_f_t"], 0, 10)), ycols, pc, "st")
                    sa["team_off"] = team_stats(df, ycols[0], pc[0], "offense_team_id", "ao_off_prior_c")
                    sa["team_def"] = team_stats(df, ycols[0], pc[0], "defense_team_id", "ao_def_prior_c")
                else:
                    df["site"] = np.where(df["site_home"] > 0, "home", np.where(df["site_away"] > 0, "away", "neutral"))
                    sa["site"] = calib(df, ycols, pc, "site")
                    stt = np.where(df["def_f_t"] >= 9, "dbl", np.where(df["def_f_t"] >= 6, "1-1", "pre"))
                    sa["state_x_half"] = calib(df.assign(sth=pd.Series(stt).str.cat(df["half"].astype(str).to_numpy(), sep="_").to_numpy()),
                                               ycols, pc, "sth")
                    for y, p in zip(ycols, pc):
                        sa[f"team_def_{y}"] = team_stats(df, y, p, "defense_team_id")
                seg[a] = sa
            res[fold]["segments"] = seg
    return res


def main():
    out = {"ao": grade_block("ao", ["y_ao"], "AO0"),
           "trip": grade_block("trip", ["y_bonus", "y_shoot"], "T2c")}
    (D / "grade_round9.json").write_text(json.dumps(out, indent=1, default=float))
    for b in ("ao", "trip"):
        for fold in ("F2", "F1"):
            print(f"\n== {b} {fold} ==")
            print(pd.DataFrame(out[b][fold]["arms"]).T.round(6).to_string())
    print("wrote", D / "grade_round9.json")


if __name__ == "__main__":
    main()
