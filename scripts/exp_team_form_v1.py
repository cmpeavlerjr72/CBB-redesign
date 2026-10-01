#!/usr/bin/env python
"""exp_team_form_v1.py -- round 16 offline stage: FTn guards and the per-team-game form latent fit (lane B, 2026-10-01).

Spec: docs/models/shared_shooting/experiments.md section 5 (committed 26e3b0b BEFORE this ran).

    .venv/Scripts/python.exe scripts/exp_team_form_v1.py

1. FTn out-of-sample FT predictions (static fits, seeds 0 and 1): train <= 2022 -> 2023, <= 2023 -> 2024, <= 2024 -> 2025.
2. Offline guards on FTn (section 5.3) for 2024 (fold 1) and 2025 (fold 2): FT% level, shooter-prior responsiveness,
   pregame margin tier calibration; both seeds.
3. Form latent: M (within-team) and B (between-team) 4x4 moments over [rim, jump2, three, ft] on train seasons,
   Omega = psd(M - B); FL1 = rank-1 factor of Omega; held-out validation (fold 1 fit 2023 -> test 2024; fold 2
   fit 2023-24 -> test 2025): within-team variance obs/pred by type, FT x own-FG cross moment obs/pred; seed-1
   floor (FTn seed-1 residuals + a seed-1 game-bootstrap refit of Omega).
4. Writes data/processed/models/shared_shooting/team_form_params_v1.json (fold-2 fit) and
   results/team_form/offline_v1.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_v] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp_ft_scorediff_v1 as XS  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402

FT = XS.FT
OUT = ROOT / "results/team_form"
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = ROOT / "data/processed/models/shared_shooting/team_form_params_v1.json"
KEYS = ("rim", "jump2", "three", "ft")
NB = 200


def psd(S):
    v, Q = np.linalg.eigh(0.5 * (S + S.T))
    return (Q * np.clip(v, 0, None)) @ Q.T


def ft_preds(d):
    """FTn out-of-sample p for 2023, 2024, 2025, seeds 0 and 1."""
    XS.use_arm("FTn")
    out = []
    for te_season in (2023, 2024, 2025):
        tr = d[d["season"] < te_season]
        te = d[d["season"] == te_season].copy()
        for seed in (0, 1):
            m = FT.LgbmArm(seed=seed).fit(FT.design_matrix(tr), tr["y"].to_numpy())
            te[f"p{seed}"] = m.predict_proba(FT.design_matrix(te))[:, -1]
        out.append(te)
        print(f"FTn preds {te_season} n={len(te)}", flush=True)
    return pd.concat(out, ignore_index=True)


def pregame_tier(seasons):
    """As-of point-differential gap per (game, team): team's season-to-date mean margin minus opponent's."""
    rows = []
    for s in seasons:
        g = R.load_actual_games(s)[["game_id", "game_date", "home_team_id", "away_team_id", "home_score", "away_score"]]
        long = pd.concat([
            pd.DataFrame({"game_id": g.game_id, "date": g.game_date, "team": g.home_team_id, "opp": g.away_team_id,
                          "m": g.home_score - g.away_score}),
            pd.DataFrame({"game_id": g.game_id, "date": g.game_date, "team": g.away_team_id, "opp": g.home_team_id,
                          "m": g.away_score - g.home_score})]).sort_values(["date", "game_id"])
        long["asof"] = long.groupby("team")["m"].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.0)
        mp = long.set_index(["game_id", "team"])["asof"]
        long["gap"] = long["asof"].to_numpy() - mp.reindex(pd.MultiIndex.from_arrays([long.game_id, long.opp])).fillna(0.0).to_numpy()
        rows.append(long[["game_id", "team", "gap"]])
    return pd.concat(rows, ignore_index=True).rename(columns={"team": "team_id"})


def guards(te, pcol):
    y = te["y"].to_numpy().astype(float)
    p = te[pcol].to_numpy()
    resp = FT.responsiveness(te, np.column_stack([1 - p, p]))
    ok, worst = FT.PM.responsiveness_verdict(resp, FT.RESPONSIVENESS_MIN_STEPS)
    tier = pd.qcut(te["gap"], 3, labels=["weaker", "even", "stronger"])
    cal = pd.DataFrame({"t": tier, "g": y - p}).groupby("t", observed=True)["g"].mean() * 100
    return {"level_pp": float((y - p).mean() * 100), "resp_pass": bool(ok), "resp_worst": worst,
            "tier_cal_pp": {str(k): float(v) for k, v in cal.items()}, "tier_max_pp": float(cal.abs().max()),
            "pass": bool(abs((y - p).mean() * 100) <= 0.5 and ok and cal.abs().max() <= 1.0)}


def team_game_arrays(ftp, pcol):
    """R, W (G, 2 sides, 4 types) per verified game, FG from preds_v1 and FT from FTn p."""
    pr = pd.read_parquet(ROOT / "results/shared_shooting/preds_v1.parquet",
                         columns=["game_id", "season", "game_date", "offense_is_home", "class_key", "y", "p"])
    pr["side"] = np.where(pr["offense_is_home"].astype(bool), 0, 1)
    pr["r"] = pr["y"] - pr["p"]
    pr["w"] = pr["p"] * (1 - pr["p"])
    fg = pr.groupby(["game_id", "side", "class_key"])[["r", "w"]].sum()
    ft = ftp.copy()
    ft["side"] = np.where(ft["shooter_is_home"].astype(bool), 0, 1)
    ft["r"] = ft["y"] - ft[pcol]
    ft["w"] = ft[pcol] * (1 - ft[pcol])
    ft["class_key"] = "ft"
    ftg = ft.groupby(["game_id", "side", "class_key"])[["r", "w"]].sum()
    allg = pd.concat([fg, ftg])
    wide = allg.unstack(["side", "class_key"]).fillna(0.0)
    meta = pr.drop_duplicates("game_id").set_index("game_id")[["season", "game_date"]]
    ver = set()
    for s in (2023, 2024, 2025):
        ver |= set(R.load_actual_games(s, verified_finals=True)["game_id"])
    ok = [g for g in wide.index if g in ver and g in meta.index]
    wide = wide.loc[ok]
    meta = meta.loc[ok]
    Rr = np.zeros((len(wide), 2, 4))
    Ww = np.zeros((len(wide), 2, 4))
    for i in (0, 1):
        for j, k in enumerate(KEYS):
            if ("r", i, k) in wide.columns:
                Rr[:, i, j] = wide[("r", i, k)].to_numpy()
                Ww[:, i, j] = wide[("w", i, k)].to_numpy()
    # (season, ISO week, type) centring
    wk = pd.to_datetime(meta["game_date"]).dt.isocalendar()
    key = meta["season"].astype(str).to_numpy() + "_" + wk["week"].astype(str).to_numpy()
    for kk in np.unique(key):
        m = key == kk
        c = Rr[m].sum(axis=(0, 1)) / np.maximum(Ww[m].sum(axis=(0, 1)), 1e-9)
        Rr[m] = Rr[m] - Ww[m] * c[None, None, :]
    return Rr, Ww, meta["season"].to_numpy()


def moments(Rr, Ww, w=None):
    if w is None:
        w = np.ones(len(Rr))
    M = (np.einsum("g,gik,gil->kl", w, Rr, Rr) - np.diag(np.einsum("g,gik->k", w, Ww))) \
        / np.einsum("g,gik,gil->kl", w, Ww, Ww)
    num = np.einsum("g,gk,gl->kl", w, Rr[:, 0], Rr[:, 1])
    den = np.einsum("g,gk,gl->kl", w, Ww[:, 0], Ww[:, 1])
    B = 0.5 * (num + num.T) / (0.5 * (den + den.T))
    return M, B


def arms_from(M, B):
    Om = psd(M - B)
    v, Q = np.linalg.eigh(Om)
    lam = Q[:, -1] * np.sqrt(max(v[-1], 0.0))
    if lam[3] < 0:
        lam = -lam
    return {"FL": Om, "FL1": np.outer(lam, lam)}, lam


def validate(Om, B, Rt, Wt):
    """held-out within-team: variance obs/pred by type and FT x FG cross obs/pred (summed over FG types)."""
    S = B + Om
    obs_v = (Rt ** 2).sum(axis=(0, 1))
    pred_v = np.einsum("gik,k->k", Wt ** 2, np.diag(S)) + Wt.sum(axis=(0, 1))
    obs_c = sum((Rt[:, :, k] * Rt[:, :, 3]).sum() for k in range(3))
    pred_c = sum((Wt[:, :, k] * Wt[:, :, 3]).sum() * S[k, 3] for k in range(3))
    # points-scale own FG x FT (pts weights 2, 2, 3 x 1), per side, mean per game
    pts = np.array([2.0, 2.0, 3.0])
    obs_pts = float(sum(pts[k] * (Rt[:, :, k] * Rt[:, :, 3]).sum() for k in range(3)) / (2 * len(Rt)))
    pred_pts = float(sum(pts[k] * (Wt[:, :, k] * Wt[:, :, 3]).sum() * S[k, 3] for k in range(3)) / (2 * len(Rt)))
    return {"var_obs_over_pred": (obs_v / pred_v).tolist(), "cross_obs_over_pred": float(obs_c / pred_c) if pred_c else None,
            "own_fg_x_ft_pts_obs_per_side": obs_pts, "own_fg_x_ft_pts_pred_per_side": pred_pts}


def main():
    t0 = time.time()
    rep = {"created_at": pd.Timestamp.now("UTC").isoformat()}
    d = XS.design()
    tier = pregame_tier([2023, 2024, 2025])
    ftp = ft_preds(d)
    ftp = ftp.merge(tier, on=["game_id", "team_id"], how="left").fillna({"gap": 0.0})
    rep["guards"] = {}
    for fold, s in (("F1", 2024), ("F2", 2025)):
        te = ftp[ftp["season"] == s]
        rep["guards"][fold] = {"seed0": guards(te, "p0"), "seed1": guards(te, "p1")}
    print(json.dumps(rep["guards"], indent=1, default=float), flush=True)
    folds = {"F1": ([2023], 2024), "F2": ([2023, 2024], 2025)}
    rep["fits"] = {}
    params = None
    for pcol in ("p0", "p1"):
        Rr, Ww, seas = team_game_arrays(ftp, pcol)
        for fold, (trs, tes) in folds.items():
            mtr = np.isin(seas, trs)
            mte = seas == tes
            M, B = moments(Rr[mtr], Ww[mtr])
            arms, lam = arms_from(M, B)
            wb = np.random.default_rng(1).poisson(1.0, int(mtr.sum())).astype(float)
            Mb, Bb = moments(Rr[mtr], Ww[mtr], wb)
            arms_b, lam_b = arms_from(Mb, Bb)
            Mt, Bt = moments(Rr[mte], Ww[mte])
            row = {"M": M.tolist(), "B": B.tolist(), "lambda_FL1": lam.tolist(), "lambda_FL1_boot1": lam_b.tolist(),
                   "test_M": Mt.tolist(), "test_B": Bt.tolist()}
            for a, Om in arms.items():
                row[a] = {"Omega": Om.tolist(), "validate": validate(Om, B, Rr[mte], Ww[mte]),
                          "validate_boot1": validate(arms_b[a], Bb, Rr[mte], Ww[mte])}
            row["G3_only_validate"] = validate(np.zeros((4, 4)), B * np.array([[1, 1, 1, 0]] * 3 + [[0, 0, 0, 0]]),
                                               Rr[mte], Ww[mte])
            rep["fits"][f"{fold}_{pcol}"] = row
            if fold == "F2" and pcol == "p0":
                params = {"FL": arms["FL"].tolist(), "FL1_lambda": lam.tolist(), "types": list(KEYS),
                          "source": "scripts/exp_team_form_v1.py, fold-2 TRAIN fit (2023-24), Omega = psd(M - B); "
                                    "never tuned on sim output",
                          "spec": "docs/models/shared_shooting/experiments.md section 5",
                          "created_at": rep["created_at"]}
    PARAMS.write_text(json.dumps(params, indent=1), encoding="utf-8")
    (OUT / "offline_v1.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    for k, v in rep["fits"].items():
        print(k, "lambda", np.round(v["lambda_FL1"], 4), "boot1", np.round(v["lambda_FL1_boot1"], 4))
        print("   M diag", np.round(np.diag(v["M"]), 4), "B diag", np.round(np.diag(v["B"]), 4),
              "M ft-row", np.round(v["M"][3], 4), "B ft-row", np.round(v["B"][3], 4))
        for a in ("FL1", "FL"):
            print("  ", a, json.dumps(v[a]["validate"], default=float))
        print("   G3only", json.dumps(v["G3_only_validate"], default=float))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
