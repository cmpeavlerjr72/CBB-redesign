#!/usr/bin/env python
"""
exp_foul_accrual_site_v1.py -- Lane G (overnight 2026-09-30): foul-accrual
site-term bake-off. Pre-registration: `docs/models/possession_outcome/
experiments.md` section 25 (committed e5dd38c BEFORE this ran).

    .venv/Scripts/python.exe scripts/exp_foul_accrual_site_v1.py

Target y_def = def_silent >= 1 per possession (round 6 definition), fit mask
in_fit_window == 1 on the fold's train seasons, static fit (as served).
    A0  constant (pooled train rate)
    A1  three site cell means, Laplace pseudo-count 200 toward the pooled rate
    A2  logistic a + b * site_signed, maximum likelihood
Writes predictions in the shared grading schema to results/home_site/foul/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

DESIGN = Path("data/processed/models/possession_outcome/round6/foul_accrual_poss_v1.parquet")
FOLDS = {"F1": {"train": [2022, 2023], "test": [2024]},
         "F2": {"train": [2022, 2023, 2024], "test": [2025]}}
OUT = Path("results/home_site/foul")
K = 200.0


def logit_fit(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """MLE of P(y) = sigmoid(a + b s) by Newton on the 3-level sufficient stats."""
    lv = np.array([-1.0, 0.0, 1.0])
    n = np.array([(s == v).sum() for v in lv], float)
    k = np.array([y[s == v].sum() for v in lv], float)
    a, b = np.log(k.sum() / (n.sum() - k.sum())), 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(a + b * lv)))
        g = np.array([(k - n * p).sum(), ((k - n * p) * lv).sum()])
        w = n * p * (1 - p)
        H = np.array([[w.sum(), (w * lv).sum()], [(w * lv).sum(), (w * lv * lv).sum()]])
        step = np.linalg.solve(H, g)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-12:
            break
    return float(a), float(b)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["game_id", "season", "game_date", "offense_team_id", "defense_team_id",
            "site_home", "site_away", "def_silent", "in_fit_window"]
    df = pd.read_parquet(DESIGN, columns=cols)
    df["y"] = (df["def_silent"] >= 1).astype("float64")
    df["s"] = (df["site_home"].astype(int) - df["site_away"].astype(int)).astype("int8")
    meta = {}
    for fold, spec in FOLDS.items():
        assert_not_sealed(spec["train"], context=f"{fold} train")
        assert_not_sealed(spec["test"], context=f"{fold} test")
        tr = df[df["season"].isin(spec["train"]) & (df["in_fit_window"] == 1)]
        te = df[df["season"].isin(spec["test"])]
        prior = float(tr["y"].mean())
        g = tr.groupby("s")["y"].agg(["sum", "size"])
        cell = ((g["sum"] + K * prior) / (g["size"] + K)).to_dict()
        a, b = logit_fit(tr["s"].to_numpy(), tr["y"].to_numpy())
        s = te["s"].to_numpy()
        preds = {"A0": np.full(len(te), prior),
                 "A1": np.array([cell[v] for v in s]),
                 "A2": 1 / (1 + np.exp(-(a + b * s)))}
        meta[fold] = {"prior": prior, "cells": {int(k): v for k, v in cell.items()},
                      "logit_a": a, "logit_b": b, "n_train": int(len(tr)), "n_test": int(len(te)),
                      "train_rate_by_site": tr.groupby("s")["y"].mean().to_dict()}
        for arm, p in preds.items():
            out = pd.DataFrame({"game_id": te["game_id"].to_numpy(),
                                "off_id": te["offense_team_id"].to_numpy(),
                                "def_id": te["defense_team_id"].to_numpy(), "site": s,
                                "y": te["y"].to_numpy(), "p": p, "w": 1.0,
                                "game_date": pd.to_datetime(te["game_date"]).to_numpy()})
            out.to_parquet(OUT / f"preds_{arm}_{fold}_s0.parquet", index=False)
        print(fold, json.dumps(meta[fold], default=float))
    (OUT / "meta_v1.json").write_text(json.dumps(meta, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
