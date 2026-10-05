#!/usr/bin/env python
"""diag_ft_prior_buckets_v1.py -- player-FT Phase 1 (2026-10-05): where is the thin/no-history shooter FT prior set low?

Diagnostic only, nothing served. Served FT model X0 (served FT_FEATURES, S1_monthly calendar, seed 0, the same fits as
results/ft_exposure/preds_X0_*) refit here so the anonymous-slot counterfactual can be scored: the same real attempt with
its shooter block zeroed (has_prior_season 0, shooter_fta_asof 0, shooter_ft_asof 0, prior_season_ft 0), which is what
an engine anonymous slot feeds the model.

Buckets: shooter prior-season FTA (0, 1-10, 11-40, 41+) and known FTA = prior-season + as-of (same cuts) x days bucket
(0-14, 15-45, 46+). Per cell: n, raw prior FT% fed (prior-season raw when has_prior else league), raw as-of FT%
(league when 0 attempts), model p, anon-block p, realised; gap = mean - realised (pp); binomial SE.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/diag_ft_prior_buckets_v1.py
Writes results/ft_prior_buckets/phase1_v1.json (+ csv).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CBB_NJOBS", "6")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_free_throw_v4_exposure as X  # noqa: E402  (sets threads, chdir, sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FT, PO, ES = X.FT, X.PO, X.ES
OUT = Path("results/ft_prior_buckets")
FEATS = list(FT.FT_FEATURES)
BLOCK = ["shooter_ft_asof", "shooter_fta_asof", "prior_season_ft", "has_prior_season"]


def bucket(v):
    return np.select([v <= 0, v <= 10, v <= 40], ["0", "1-10", "11-40"], "41+")


def dbucket(v):
    return np.select([v <= 14, v <= 45], ["d0-14", "d15-45"], "d46+")


def fit_monthly(tr, te, seed=0):
    te_dates = pd.to_datetime(te["game_date"])
    cuts = PO.month_boundaries(te_dates)
    p, pa = np.zeros(len(te)), np.zeros(len(te))
    anon = te[FEATS].copy()
    anon[BLOCK] = 0.0
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not seg.any():
            continue
        prior = te.loc[(te_dates < cut).to_numpy(), FEATS + ["y"]]
        rows = pd.concat([tr[FEATS + ["y"]], prior], ignore_index=True)
        m = FT.LgbmArm(seed=seed).fit(FT.design_matrix(rows, tuple(FEATS)), rows["y"].to_numpy())
        k1 = FT.CLASS_INDEX["MAKE"]
        p[seg] = m.predict_proba(FT.design_matrix(te.loc[seg], tuple(FEATS)))[:, k1]
        pa[seg] = m.predict_proba(FT.design_matrix(anon.loc[seg], tuple(FEATS)))[:, k1]
    return p, pa


def cells(df, key):
    rows = []
    for (b, db), g in df.groupby([key, "dbk"]):
        y = g["y"].mean()
        rows.append({"bucket_kind": key, "bucket": b, "days": db, "n": len(g), "n_shooters": g["shooter_id"].nunique(),
                     "prior_raw": g["prior_raw"].mean(), "asof_raw": g["asof_raw"].mean(), "p": g["p"].mean(),
                     "p_anon": g["p_anon"].mean(), "y": y, "gap_p_pp": 100 * (g["p"].mean() - y),
                     "gap_anon_pp": 100 * (g["p_anon"].mean() - y),
                     "se_pp": 100 * float(np.sqrt(y * (1 - y) / len(g))), "underpowered": len(g) < 1000})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ES.load_universe()
    att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
    att = att[att["season"].isin(X.SEASONS)]
    d = X.add_exposure(FT.build_ft_design(att))
    res = {}
    allrows = []
    for fold in ("F2", "F1"):
        tr, te = FT.fold_slices(d, fold)
        te = te.copy()
        te["p"], te["p_anon"] = fit_monthly(tr, te)
        cached = pd.read_parquet(f"results/ft_exposure/preds_X0_{fold}_s0.parquet")
        repro = float(np.abs(cached["p"].to_numpy() - te["p"].to_numpy()).max())
        te["prior_raw"] = te["prior_season_ft_raw"]
        te["asof_raw"] = te["shooter_ft_raw"]
        te["dbk"] = dbucket(te["days_since_start"].to_numpy())
        te["prior_fta_bk"] = bucket(te["prior_season_fta"].to_numpy())
        te["known_fta_bk"] = bucket((te["prior_season_fta"] + te["shooter_fta_asof"]).to_numpy())
        rows = cells(te, "prior_fta_bk") + cells(te, "known_fta_bk")
        for r in rows:
            r["fold"] = fold
        allrows += rows
        y = te["y"].to_numpy()
        res[fold] = {"repro_max_abs_vs_cached": repro, "n": len(te), "mean_p": float(te["p"].mean()),
                     "mean_y": float(y.mean()), "mean_p_anon": float(te["p_anon"].mean())}
        print(fold, res[fold], flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(OUT / "phase1_v1.csv", index=False)
    res["cells"] = allrows
    (OUT / "phase1_v1.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(df[["fold", "bucket_kind", "bucket", "days", "n", "prior_raw", "asof_raw", "p", "p_anon", "y", "gap_p_pp",
              "gap_anon_pp", "se_pp"]].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
