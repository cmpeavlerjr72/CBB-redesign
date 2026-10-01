#!/usr/bin/env python
"""
grade_home_site_v1.py -- Lane G (overnight 2026-09-30): ONE blind grader for
every arm of the home-site round (free_throw, clock, foul accrual, fg_make).

    .venv/Scripts/python.exe scripts/grade_home_site_v1.py --model ft --arms FT0,FT1,FT2 --ref FT0
    .venv/Scripts/python.exe scripts/grade_home_site_v1.py --model fg --arms S0,G1 --ref S0 --by-class

Input schema (results/home_site/<model>/preds_<arm>_<fold>_s<seed>.parquet):
    game_id, off_id, def_id, site (+1 home / 0 neutral / -1 away, offence),
    y, p, w [, game_date, driver, driver_defined, shot_class, crps_trunc]

Primary (pre-registered): G_site = |HCA_pred - HCA_real|, HCA from
    rate = mu + off_FE + def_FE + b_home [site=+1] + b_away [site=-1]
on (game, offence) aggregates weighted by sum(w). Floor = max(seed-refit
|dG|, 2 x game-block bootstrap SE of the paired dG), 200 reps, seed 20260930.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

OUTROOT = Path("results/home_site")
N_BOOT = 200


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(yw=df["y"] * df["w"], pw=df["p"] * df["w"])
    g = d.groupby(["game_id", "off_id"], sort=True).agg(
        def_id=("def_id", "first"), site=("site", "first"),
        y=("yw", "sum"), p=("pw", "sum"), w=("w", "sum")).reset_index()
    return g


class FE:
    """Sparse weighted FE fit, reusable across bootstrap reweightings."""

    def __init__(self, g: pd.DataFrame):
        teams = np.unique(np.concatenate([g["off_id"].to_numpy(), g["def_id"].to_numpy()]))
        ti = {t: i for i, t in enumerate(teams)}
        n, T = len(g), len(teams)
        oi = g["off_id"].map(ti).to_numpy()
        di = g["def_id"].map(ti).to_numpy()
        s = g["site"].to_numpy()
        rows = np.concatenate([np.arange(n)] * 5)
        cols = np.concatenate([np.zeros(n, int), np.ones(n, int), np.full(n, 2),
                               3 + oi, 3 + T + di])
        vals = np.concatenate([np.ones(n), (s == 1).astype(float), (s == -1).astype(float),
                               np.ones(n), np.ones(n)])
        self.X = sp.csr_matrix((vals, (rows, cols)), shape=(n, 3 + 2 * T))
        self.k = 3 + 2 * T

    def hca(self, rate: np.ndarray, w: np.ndarray) -> tuple[float, float]:
        W = sp.diags(w)
        XtW = self.X.T @ W
        A = (XtW @ self.X).toarray()
        A[np.arange(3, self.k), np.arange(3, self.k)] += 1e-6 * float(w.mean())
        beta = np.linalg.solve(A, XtW @ rate)
        r = rate - self.X @ beta
        s2 = float((w * r * r).sum() / max((w > 0).sum() - self.k, 1))
        Ai = np.linalg.inv(A)
        var = s2 * (Ai[1, 1] + Ai[2, 2] - 2 * Ai[1, 2])
        return float(beta[1] - beta[2]), float(np.sqrt(max(var, 0)))


def site_metrics(g: pd.DataFrame, fe: FE, mult: np.ndarray | None = None) -> dict:
    w = g["w"].to_numpy(float) * (1.0 if mult is None else mult)
    ok = w > 0
    yr = np.where(ok, g["y"].to_numpy() / np.maximum(g["w"].to_numpy(), 1e-12), 0.0)
    pr = np.where(ok, g["p"].to_numpy() / np.maximum(g["w"].to_numpy(), 1e-12), 0.0)
    h_real, se_real = fe.hca(yr, w)
    h_pred, _ = fe.hca(pr, w)
    return {"hca_real": h_real, "hca_real_se": se_real, "hca_pred": h_pred,
            "gap": h_pred - h_real, "G_site": abs(h_pred - h_real)}


def raw_site_calib(df: pd.DataFrame) -> dict:
    out = {}
    for v, nm in ((1, "home"), (-1, "away"), (0, "neutral")):
        m = df["site"].to_numpy() == v
        if m.any():
            w = df["w"].to_numpy()[m]
            out[nm] = {"n": int(m.sum()), "real": float((df["y"].to_numpy()[m] * w).sum() / w.sum()),
                       "pred": float((df["p"].to_numpy()[m] * w).sum() / w.sum())}
            out[nm]["pred_minus_real"] = out[nm]["pred"] - out[nm]["real"]
    if "home" in out and "away" in out:
        out["home_minus_away_resid"] = out["home"]["pred_minus_real"] - out["away"]["pred_minus_real"]
    return out


def responsiveness(df: pd.DataFrame) -> dict | None:
    if "driver" not in df:
        return None
    d = df[df["driver_defined"].astype(bool)] if "driver_defined" in df else df
    q = pd.qcut(d["driver"].rank(method="first"), 5, labels=False)
    t = d.groupby(q).agg(y=("y", "mean"), p=("p", "mean"))
    dy = t["y"].iloc[-1] - t["y"].iloc[0]
    return {"q_real": t["y"].round(5).tolist(), "q_pred": t["p"].round(5).tolist(),
            "slope_ratio": float((t["p"].iloc[-1] - t["p"].iloc[0]) / dy) if dy else None}


def per_team(df: pd.DataFrame) -> dict:
    """Per-team home-minus-away residual (pred - real), teams with >= 5 home and
    >= 5 away games: mean and mean-absolute over teams."""
    d = df[df["site"] != 0].assign(r=(df["p"] - df["y"]) * df["w"])
    t = d.groupby(["off_id", "site"]).agg(r=("r", "sum"), w=("w", "sum"),
                                          ng=("game_id", "nunique")).reset_index()
    t["rr"] = t["r"] / t["w"]
    h = t[t["site"] == 1].set_index("off_id")
    a = t[t["site"] == -1].set_index("off_id")
    ix = h.index.intersection(a.index)
    ix = ix[(h.loc[ix, "ng"] >= 5).to_numpy() & (a.loc[ix, "ng"] >= 5).to_numpy()]
    diff = h.loc[ix, "rr"] - a.loc[ix, "rr"]
    return {"n_teams": int(len(ix)), "mean": float(diff.mean()), "mean_abs": float(diff.abs().mean())}


def is_binary(df):
    return set(np.unique(df["y"].to_numpy()[:1000])).issubset({0.0, 1.0})


def row_ll(df):
    p = np.clip(df["p"].to_numpy(), 1e-9, 1 - 1e-9)
    y = df["y"].to_numpy()
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def grade_slice(frames: dict[str, pd.DataFrame], ref: str, seed1: pd.DataFrame | None,
                extra_metric: str | None) -> dict:
    res = {}
    aggs = {a: aggregate(f) for a, f in frames.items()}
    g0 = aggs[ref]
    fe = FE(g0)
    games = g0["game_id"].to_numpy()
    ug, inv = np.unique(games, return_inverse=True)
    for a, f in frames.items():
        ga = aggs[a]
        assert (ga["game_id"].to_numpy() == games).all() and (ga["off_id"].to_numpy() == g0["off_id"].to_numpy()).all()
        r = site_metrics(ga, fe)
        r["raw_site_calib"] = raw_site_calib(f)
        r["responsiveness"] = responsiveness(f)
        r["per_team"] = per_team(f)
        r["n_rows"] = int(len(f))
        if is_binary(f):
            r["log_loss"] = float(row_ll(f).mean())
        if extra_metric and extra_metric in f:
            r[extra_metric] = float(np.nanmean(f[extra_metric].to_numpy()))
        res[a] = r
    # seed floor
    floor_seed = None
    if seed1 is not None:
        g1 = aggregate(seed1)
        r1 = site_metrics(g1, FE(g1))
        floor_seed = abs(r1["G_site"] - res[ref]["G_site"])
        res["_seed1_ref"] = {"G_site": r1["G_site"], "hca_pred": r1["hca_pred"]}
        if "log_loss" in res[ref]:
            res["_seed1_ref"]["log_loss"] = float(row_ll(seed1).mean())
            res["_seed1_ref"]["ll_floor"] = abs(res["_seed1_ref"]["log_loss"] - res[ref]["log_loss"])
    # paired game bootstrap
    rng = np.random.default_rng(20260930)
    # per-game loss sums for the guard
    gsum = {}
    for a, f in frames.items():
        key = extra_metric if (extra_metric and extra_metric in f) else ("ll" if is_binary(f) else None)
        if key is None:
            continue
        v = row_ll(f) if key == "ll" else f[key].to_numpy()
        ok = np.isfinite(v)
        s = pd.Series(v[ok]).groupby(f["game_id"].to_numpy()[ok]).agg(["sum", "size"])
        gsum[a] = s.reindex(ug).fillna(0.0)
    boots = {a: [] for a in frames}
    lboots = {a: [] for a in gsum}
    for _ in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
        mult = cnt[inv]
        gref = site_metrics(g0, fe, mult)["G_site"]
        for a in frames:
            ga = aggs[a]
            boots[a].append(site_metrics(ga, fe, mult)["G_site"] - gref)
        if ref in gsum:
            lr = (gsum[ref]["sum"].to_numpy() * cnt).sum() / (gsum[ref]["size"].to_numpy() * cnt).sum()
            for a in gsum:
                la = (gsum[a]["sum"].to_numpy() * cnt).sum() / (gsum[a]["size"].to_numpy() * cnt).sum()
                lboots[a].append(la - lr)
    for a in frames:
        if a == ref:
            continue
        se = float(np.std(boots[a], ddof=1))
        floor = max(2 * se, floor_seed or 0.0)
        d = res[a]["G_site"] - res[ref]["G_site"]
        res[a]["dG_vs_ref"] = d
        res[a]["dG_boot_se"] = se
        res[a]["floor"] = floor
        res[a]["floor_seed_component"] = floor_seed
        res[a]["beats_ref_on_primary"] = bool(d < -floor)
        if a in lboots and lboots[a]:
            lse = float(np.std(lboots[a], ddof=1))
            key = "crps_trunc" if extra_metric in frames[a] else "log_loss"
            dl = res[a][key] - res[ref][key]
            res[a]["d_loss_vs_ref"] = dl
            res[a]["d_loss_boot_se"] = lse
    return res


def load(model: str, arm: str, fold: str, seed: int) -> pd.DataFrame | None:
    p = OUTROOT / model / f"preds_{arm}_{fold}_s{seed}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if "off_team_id" in d:          # fg_make schema -> shared schema
        d = d.rename(columns={"off_team_id": "off_id", "def_team_id": "def_id"})
        d["site"] = (d["site_home"] - d["site_away"]).astype("int8")
        d["w"] = 1.0
        d["driver"] = d["off_make_c"]          # team (offence) prior quintiles
        d["driver_defined"] = d["n_prior_off"] > 0
        d = d.sort_values(["shot_class", "game_id", "off_id"], kind="stable").reset_index(drop=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--folds", default="F2,F1")
    ap.add_argument("--by-class", action="store_true")
    ap.add_argument("--extra-metric", default=None)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    out = {}
    for fold in a.folds.split(","):
        frames = {}
        for arm in a.arms.split(","):
            f = load(a.model, arm, fold, 0)
            if f is not None:
                frames[arm] = f
        if a.ref not in frames:
            print(f"{fold}: reference missing, skipped")
            continue
        s1 = load(a.model, a.ref, fold, 1)
        if a.by_class:
            for c in sorted(frames[a.ref]["shot_class"].unique()):
                fr = {k: v[v["shot_class"] == c].reset_index(drop=True) for k, v in frames.items()}
                s1c = None if s1 is None else s1[s1["shot_class"] == c].reset_index(drop=True)
                out[f"{fold}|{c}"] = grade_slice(fr, a.ref, s1c, a.extra_metric)
        else:
            out[fold] = grade_slice(frames, a.ref, s1, a.extra_metric)
    path = OUTROOT / a.model / f"grade_{a.tag}.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for k, v in out.items():
        print(f"== {k}")
        for arm, r in v.items():
            if arm.startswith("_"):
                print(f"  {arm}: {json.dumps(r, default=float)}")
                continue
            line = (f"  {arm}: HCA real {r['hca_real']:+.5f} (SE {r['hca_real_se']:.5f}) pred {r['hca_pred']:+.5f} "
                    f"G {r['G_site']:.5f}")
            if "dG_vs_ref" in r:
                line += f" dG {r['dG_vs_ref']:+.5f} floor {r['floor']:.5f} beats {r['beats_ref_on_primary']}"
            for key in ("log_loss", "crps_trunc"):
                if key in r:
                    line += f" {key} {r[key]:.6f}"
            if "d_loss_vs_ref" in r:
                line += f" dloss {r['d_loss_vs_ref']:+.6f} (se {r['d_loss_boot_se']:.6f})"
            rs = r.get("responsiveness")
            if rs:
                line += f" resp {rs['slope_ratio']:.3f}"
            rc = r["raw_site_calib"]
            line += f" rawHmA {rc.get('home_minus_away_resid', float('nan')):+.5f}"
            line += f" team|d| {r['per_team']['mean_abs']:.4f}"
            print(line)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
