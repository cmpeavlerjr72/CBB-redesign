#!/usr/bin/env python
"""
train_foul_cal_v1.py -- possession-outcome foul-accrual CALENDAR round (experiments.md section 32,
pre-registered and pushed in commit 3337d13 BEFORE this script ran).

    .venv/Scripts/python.exe scripts/train_foul_cal_v1.py --n-jobs 8

Arms (32.2): A2 (round-7 GBM on STATE_T, the served LUT's source spec), A2dec (A2 offset +
b*exp(-dss/tau)), A2decH (b by half, shared tau), A2dbk (GBM on STATE_T + days bucket). Seeds 0 and 7
for every arm (A2dec* seed s sits on the seed-s A2 offset). Design, target, fit mask and GBM settings
are round 7's (`train_foul_joint_v1`, imported unedited).

Writes data/processed/models/possession_outcome/round10cal/preds_{F1,F2}_seed{0,7}.parquet,
fits_{fold}.joblib (seed-0 GBMs + decay params, for LUT export) and train_meta.json.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_foul_joint_v1 as FJ  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

OUT = ROOT / "data/processed/models/possession_outcome/round10cal"
SEEDS = (0, 7)
DSS_EDGES = np.array([7, 14, 30, 45])        # bucket = searchsorted(edges, dss, 'left'): 0-7,8-14,15-30,31-45,46+
TAU_BOUNDS = (1.0, 150.0)


def dss_bucket(dss: np.ndarray) -> np.ndarray:
    return np.searchsorted(DSS_EDGES, np.asarray(dss, dtype=float), side="left").astype(np.int64)


def gbm_task(Xtr, ytr, Xq_list, seed):
    m = FJ.fit_gbm(Xtr, ytr, seed)
    return [m.predict_proba(X)[:, 1] for X in Xq_list], m


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def fit_decay(off: np.ndarray, y: np.ndarray, dss: np.ndarray, grp: np.ndarray, n_grp: int) -> dict:
    """ML fit of logit p = off + b[grp] * exp(-dss / tau), tau in TAU_BOUNDS."""
    from scipy.optimize import minimize
    y = y.astype(float)

    def nll(theta):
        b = theta[:n_grp]
        tau = np.exp(theta[n_grp])
        e = np.exp(-dss / tau)
        z = off + b[grp] * e
        p = 1.0 / (1.0 + np.exp(-z))
        ll = np.sum(np.logaddexp(0, z) - y * z)
        r = p - y
        g = np.zeros(n_grp + 1)
        g[:n_grp] = np.bincount(grp, weights=r * e, minlength=n_grp)
        g[n_grp] = np.sum(r * b[grp] * e * dss / tau)       # d/dlog tau
        return ll, g

    best = None
    for tau0 in (5.0, 15.0, 40.0):
        x0 = np.r_[np.zeros(n_grp), np.log(tau0)]
        r = minimize(nll, x0, jac=True, method="L-BFGS-B",
                     bounds=[(-3, 3)] * n_grp + [tuple(np.log(TAU_BOUNDS))])
        if best is None or r.fun < best.fun:
            best = r
    return {"b": best.x[:n_grp].tolist(), "tau": float(np.exp(best.x[n_grp])),
            "nll": float(best.fun), "converged": bool(best.success), "n": int(len(y))}


def apply_decay(off, dss, grp, prm) -> np.ndarray:
    b = np.asarray(prm["b"])
    z = off + b[grp] * np.exp(-dss / prm["tau"])
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = FJ.possession_design()
    d["dss"] = d["days_since_start"].astype("float64")
    if d["dss"].isna().any():
        raise SystemExit("days_since_start has NaN rows")
    d["dss_bkt"] = dss_bucket(d["dss"].to_numpy()).astype("float32")
    print(f"design {len(d):,} rows ({time.time() - t0:.0f}s)", flush=True)
    feats_b = FJ.STATE_T + ["dss_bkt"]
    from joblib import Parallel, delayed
    parts, tasks, keys = {}, [], []
    for fold, sp in FJ.FOLDS.items():
        assert_not_sealed(sp["train"], context=f"{fold} train")
        assert_not_sealed(sp["test"], context=f"{fold} test")
        tr = d[d["season"].isin(sp["train"]) & (d["in_fit_window"] == 1)]
        te = d[d["season"].isin(sp["test"])]
        parts[fold] = (tr, te)
        for arm, ft in (("A2", FJ.STATE_T), ("A2dbk", feats_b)):
            Xtr = tr[ft].to_numpy("float32")
            for s in SEEDS:
                tasks.append(delayed(gbm_task)(Xtr, tr["y_nt"].to_numpy(),
                                               [Xtr, te[ft].to_numpy("float32")], s))
                keys.append((fold, arm, s))
    print(f"{len(tasks)} GBM fits on {a.n_jobs} workers", flush=True)
    res = dict(zip(keys, Parallel(n_jobs=a.n_jobs, backend="loky")(tasks)))
    print(f"GBM fits done ({time.time() - t0:.0f}s)", flush=True)
    meta = {"pre_registration": "possession_outcome experiments.md section 32 (commit 3337d13)",
            "gbm": FJ.GBM_KW, "seeds": SEEDS, "dss_edges": DSS_EDGES.tolist(), "decay": {}}
    for fold, (tr, te) in parts.items():
        fits = {}
        dtr, dte = tr["dss"].to_numpy(), te["dss"].to_numpy()
        htr = (tr["period"].to_numpy() >= 2).astype(np.int64)
        hte = (te["period"].to_numpy() >= 2).astype(np.int64)
        for s in SEEDS:
            (p_tr, p_te), m = res[(fold, "A2", s)]
            (_, pb_te), mb = res[(fold, "A2dbk", s)]
            o_tr, o_te = logit(p_tr), logit(p_te)
            prm1 = fit_decay(o_tr, tr["y_nt"].to_numpy(), dtr, np.zeros(len(dtr), np.int64), 1)
            prm2 = fit_decay(o_tr, tr["y_nt"].to_numpy(), dtr, htr, 2)
            meta["decay"][f"{fold}_s{s}"] = {"A2dec": prm1, "A2decH": prm2}
            print(f"{fold} s{s} A2dec {prm1}  A2decH {prm2}", flush=True)
            out = te[["game_id", "season", "period", "poss_index", "offense_team_id", "defense_team_id",
                      "in_fit_window", "site_home", "site_away", "def_f_t", "off_f_t", "dss", "y_nt"]].copy()
            out["p__A2"] = p_te
            out["p__A2dec"] = apply_decay(o_te, dte, np.zeros(len(dte), np.int64), prm1)
            out["p__A2decH"] = apply_decay(o_te, dte, hte, prm2)
            out["p__A2dbk"] = pb_te
            out.to_parquet(OUT / f"preds_{fold}_seed{s}.parquet", index=False)
            if s == 0:
                fits = {"A2": m, "A2dbk": mb, "A2dec": prm1, "A2decH": prm2}
        import joblib
        joblib.dump(fits, OUT / f"fits_{fold}.joblib")
    meta["seconds"] = round(time.time() - t0, 1)
    (OUT / "train_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta["decay"], indent=1))


if __name__ == "__main__":
    main()
