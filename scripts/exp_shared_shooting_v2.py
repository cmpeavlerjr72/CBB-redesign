#!/usr/bin/env python
"""exp_shared_shooting_v2.py -- shared_shooting round 2: walk-forward refit of G3's Sigma (lane B, 2026-10-01).

Spec: docs/models/shared_shooting/experiments.md section 3 (committed 5a62b8e before this stage ran).

    .venv/Scripts/python.exe scripts/exp_shared_shooting_v2.py wf

Arms: G3P (pooled train seasons, served spec), G3L (last train season only),
G3A (as-of in-season update of G3P from completed earlier ISO weeks of the test
season, prior weight = one train season's between-team mass). Reuses the round-1
data and moment code (`exp_shared_shooting_v1`). Writes results/shared_shooting/wf_v2.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_shared_shooting_v1 as V1  # noqa: E402

ARMS = ("G3P", "G3L", "G3A")
SIMPLICITY = {"G3P": 0, "G3L": 1, "G3A": 2}
FOLDS = V1.FOLDS


def between_mass(R, W, w=None):
    if w is None:
        w = np.ones(len(R))
    num = np.einsum("g,gk,gl->kl", w, R[:, 0], R[:, 1])
    den = np.einsum("g,gk,gl->kl", w, W[:, 0], W[:, 1])
    return 0.5 * (num + num.T), 0.5 * (den + den.T)


def ll6(R, W, S, T):
    """Per-game Gaussian log density of [R_h(3), R_a(3)]; S is (G,3,3) or (3,3)."""
    G = len(R)
    S = np.broadcast_to(S, (G, 3, 3))
    V = np.zeros((G, 6, 6))
    for i in range(2):
        Wi = W[:, i]
        V[:, 3 * i:3 * i + 3, 3 * i:3 * i + 3] = (np.einsum("gk,gkl,gl->gkl", Wi, S + T[None], Wi)
                                                  + np.einsum("gk,kl->gkl", Wi, np.eye(3)))
    V[:, 0:3, 3:6] = np.einsum("gk,gkl,gl->gkl", W[:, 0], S, W[:, 1])
    V[:, 3:6, 0:3] = np.transpose(V[:, 0:3, 3:6], (0, 2, 1))
    V += np.eye(6)[None] * 1e-6
    x = np.concatenate([R[:, 0], R[:, 1]], axis=1)
    _, logdet = np.linalg.slogdet(V)
    sol = np.linalg.solve(V, x[..., None])[..., 0]
    return -0.5 * (logdet + (x * sol).sum(1) + 6 * np.log(2 * np.pi))


def fit_static(tr_games, w=None):
    R, W = V1.arrays(tr_games, centre=True)
    if w is None:
        w = np.ones(len(R))
    S = V1.psd(V1.sigma_between(R, W, w))
    T = V1.psd(V1.within_cross(R, W, w) - S)
    return S, T, R, W


def g3a_path(te, S_P, n0_den):
    """Per-game as-of Sigma for the test season: prior S_P with mass n0_den (3x3),
    updated with centred between-team cross products from ISO weeks strictly
    before the game's week (season-ordered week index)."""
    R, W = V1.arrays(te, centre=True)
    d = pd.to_datetime(te["game_date"])
    wk = (d - d.min()).dt.days.to_numpy() // 7          # season-relative week; same ordering as ISO weeks
    # ISO-week key as used by the centring
    iso = te["week"].to_numpy()
    yr = d.dt.isocalendar().year.to_numpy()
    key = yr * 100 + iso
    order = np.argsort(key, kind="stable")
    ukeys = np.unique(key)
    S_g = np.zeros((len(R), 3, 3))
    num_acc = np.zeros((3, 3))
    den_acc = np.zeros((3, 3))
    path = []
    for k in ukeys:
        m = key == k
        S_k = V1.psd((n0_den * S_P + num_acc) / (n0_den + den_acc))
        S_g[m] = S_k
        path.append({"iso_key": int(k), "games": int(m.sum()), "Sigma_diag": np.diag(S_k).tolist()})
        n_, d_ = between_mass(R[m], W[m])
        num_acc += n_
        den_acc += d_
    return S_g, R, W, path


def run_fold(games, spec, boot_seed=2):
    tr = games[games["season"].isin(spec["train"])]
    te = games[games["season"] == spec["test"]]
    last = games[games["season"] == max(spec["train"])]
    S_P, T_P, Rtr, Wtr = fit_static(tr)
    S_L, T_L, _, _ = fit_static(last)
    # prior mass: mean per-season between-team denominator over train seasons
    dens = []
    for s in spec["train"]:
        Rs, Ws = V1.arrays(games[games["season"] == s], centre=True)
        dens.append(between_mass(Rs, Ws)[1])
    n0_den = np.mean(dens, axis=0)
    S_A, Rte, Wte, path = g3a_path(te, S_P, n0_den)
    lls = {"G3P": ll6(Rte, Wte, S_P, T_P), "G3L": ll6(Rte, Wte, S_L, T_L), "G3A": ll6(Rte, Wte, S_A, T_P)}
    # spec-identical refit under another seed: training-game bootstrap draw (seed 1)
    w1 = V1.boot_w(len(Rtr), 1, seed=1)[0]
    S_P1, T_P1, _, _ = fit_static(tr, w1)
    Rl, _ = V1.arrays(last, centre=True)
    wl = V1.boot_w(len(Rl), 1, seed=1)[0]
    S_L1, T_L1, _, _ = fit_static(last, wl)
    S_A1, _, _, _ = g3a_path(te, S_P1, n0_den)
    lls_b1 = {"G3P": ll6(Rte, Wte, S_P1, T_P1), "G3L": ll6(Rte, Wte, S_L1, T_L1),
              "G3A": ll6(Rte, Wte, S_A1, T_P1)}
    base = lls["G3P"]
    BW = V1.boot_w(len(base), V1.N_BOOT, seed=boot_seed)
    res = {}
    obs = 0.5 * (np.einsum("gk,gl->kl", Rte[:, 0], Rte[:, 1]) + np.einsum("gk,gl->kl", Rte[:, 1], Rte[:, 0]))
    for arm in ARMS:
        d = lls[arm] - base
        dboot = (BW @ d) / BW.sum(1)
        refit_gap = float(abs(lls[arm].mean() - lls_b1[arm].mean()))
        bsd = float(dboot.std())
        floor = max(bsd, refit_gap)
        Sg = {"G3P": np.broadcast_to(S_P, (len(Rte), 3, 3)), "G3L": np.broadcast_to(S_L, (len(Rte), 3, 3)),
              "G3A": S_A}[arm]
        pred = np.einsum("gk,gkl,gl->kl", Wte[:, 0], Sg, Wte[:, 1])
        pred = 0.5 * (pred + pred.T)
        # vs N (no latent), for reference: within T = within_cross (all unshared)
        res[arm] = {"gain_vs_G3P": float(d.mean()), "boot_sd": bsd, "refit_seed1_gap": refit_gap,
                    "floor": floor, "floors": float(d.mean() / floor) if floor > 0 else 0.0,
                    "between_obs_over_pred_total": float(obs.sum() / pred.sum()),
                    "mean_ll": float(lls[arm].mean())}
    months = pd.to_datetime(te["game_date"]).dt.month.to_numpy()
    seg = {}
    for mth in (11, 12, 1, 2, 3):
        m = months == mth
        seg[f"month_{mth}"] = {a: float((lls[a][m] - base[m]).mean()) for a in ARMS}
        seg[f"month_{mth}"]["n"] = int(m.sum())
    return {"n_test": int(len(Rte)), "arms": res, "segments": seg, "Sigma_P": S_P.tolist(),
            "Sigma_L": S_L.tolist(), "n0_den": n0_den.tolist(), "G3A_path": path}


def main():
    t0 = time.time()
    games = V1.load_games()
    print(f"[{time.time()-t0:6.1f}s] games {len(games)}", flush=True)
    rep = {"created_at": pd.Timestamp.now("UTC").isoformat(), "folds": {}}
    for fold, spec in FOLDS.items():
        rep["folds"][fold] = run_fold(games, spec)
    f1, f2 = rep["folds"]["F1"]["arms"], rep["folds"]["F2"]["arms"]
    elig = [a for a in ARMS if a != "G3P" and f2[a]["gain_vs_G3P"] > 2 * f2[a]["floor"]
            and f1[a]["gain_vs_G3P"] > 0]
    if not elig:
        winner = "G3P"
    else:
        best = max(elig, key=lambda a: f2[a]["gain_vs_G3P"])
        tied = [a for a in elig if f2[best]["gain_vs_G3P"] - f2[a]["gain_vs_G3P"] <= f2[best]["floor"]]
        winner = min(tied, key=lambda a: SIMPLICITY[a])
    rep["eligible"], rep["winner"] = elig, winner
    out = V1.OUT / "wf_v2.json"
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    for fold in ("F1", "F2"):
        print(fold, json.dumps(rep["folds"][fold]["arms"], indent=0, default=float))
        print("  segments", rep["folds"][fold]["segments"])
        print("  path", [(p["iso_key"], [round(x, 4) for x in p["Sigma_diag"]]) for p in rep["folds"][fold]["G3A_path"][::4]])
    print("eligible", elig, "winner", winner, f"[{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "wf") != "wf":
        raise SystemExit("only stage: wf")
    main()
