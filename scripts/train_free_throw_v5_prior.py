#!/usr/bin/env python
"""
train_free_throw_v5_prior.py -- FT thin-sample shooter prior round. Pre-registration:
`docs/models/free_throw/experiments.md` section 20 (committed c1cd2ef BEFORE this ran).

    CBB_NJOBS=6 .venv/Scripts/python.exe scripts/train_free_throw_v5_prior.py --arms P0,P1,P2,P3 --folds F2,F1 --seeds 0
    CBB_NJOBS=6 .venv/Scripts/python.exe scripts/train_free_throw_v5_prior.py --arms P0 --folds F2,F1 --seeds 1

Arms (S1_monthly calendar for all, stated cost deviation of sections 11/13/18/20):
    P0  served FT_FEATURES
    P1  + prior_season_fta
    P2  prior_season_ft -> prior_season_ft_eb = (prev_ftm + k lg)/(prev_fta + k) - lg  (0 with no prior season)
    P3  P2 with target lg + m_c, m_c = WLS of (y - lg) on CBBD bio (pos, height_c, d1_years, missing flag, is_transfer)
k (FT.SHRINK_GRID) and m_c are fitted once per fold on that fold's TRAINING seasons only.

Writes results/ft_prior/preds_<arm>_<fold>_s<seed>.parquet, meta_*.json, prior_params_<arm>_<fold>.json. Sibling of
train_free_throw_v4_exposure.py (not edited). Nothing goes to an engine directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_free_throw_v4_exposure as X  # noqa: E402  (threads, chdir, src path)
import train_free_throw_v3_newcomer as N  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FT, PO, ES = X.FT, X.PO, X.ES
FT.LgbmArm.PARAMS = {**FT.LgbmArm.PARAMS, "n_jobs": X.NJ}
OUT = Path("results/ft_prior")
SEASONS = [2022, 2023, 2024, 2025]
BASE = list(FT.FT_FEATURES)
EBF = [("prior_season_ft_eb" if f == "prior_season_ft" else f) for f in BASE]
ARMS = {"P0": BASE, "P1": BASE + ["prior_season_fta"], "P2": EBF, "P3": EBF}
BIO_X = ["pos_G", "pos_F", "pos_C", "height_c0", "d1_years0", "d1_missing", "is_transfer_f"]


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bio_matrix(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(d))] + [d[c].to_numpy(dtype="float64") for c in BIO_X])


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = N.add_bio(d)
    d["height_c0"] = d["height_c"].fillna(0.0).astype("float32")
    d["d1_missing"] = d["d1_years"].isna().astype("float32")
    d["d1_years0"] = d["d1_years"].fillna(0.0).astype("float32")
    d["is_transfer_f"] = d["is_transfer"].astype("float32")
    d["prev_fta0"] = d["prev_fta"].fillna(0.0).astype("float64")
    d["prev_ftm0"] = d["prev_ftm"].fillna(0.0).astype("float64")
    return d


def fit_prior(tr: pd.DataFrame, conditioned: bool) -> dict:
    lg = tr["lg_ft_asof"].to_numpy(dtype="float64")
    coef = np.zeros(1 + len(BIO_X))
    if conditioned:
        Xb, r = bio_matrix(tr), tr["y"].to_numpy(dtype="float64") - lg
        coef = np.linalg.lstsq(Xb, r, rcond=None)[0]
    hp = tr["has_prior_season"].to_numpy() > 0.5
    h = tr[hp]
    m = lg[hp] + bio_matrix(h) @ coef
    y = h["y"].to_numpy(dtype="float64")
    grid = {}
    for k in FT.SHRINK_GRID:
        rate = (h["prev_ftm0"].to_numpy() + k * m) / (h["prev_fta0"].to_numpy() + k)
        grid[float(k)] = ll(y, rate)
    k = min(grid, key=grid.get)
    return {"k": k, "coef": coef.tolist(), "bio_x": BIO_X, "conditioned": conditioned, "grid_ll": grid}


def eb_feature(d: pd.DataFrame, prm: dict) -> np.ndarray:
    lg = d["lg_ft_asof"].to_numpy(dtype="float64")
    mc = bio_matrix(d) @ np.asarray(prm["coef"])
    rate = (d["prev_ftm0"].to_numpy() + prm["k"] * (lg + mc)) / (d["prev_fta0"].to_numpy() + prm["k"])
    hp = d["has_prior_season"].to_numpy() > 0.5
    return np.where(hp, rate - lg, mc).astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="P0,P1,P2,P3")
    ap.add_argument("--folds", default="F2,F1")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    t0 = time.time()
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="free_throw prior round")
    OUT.mkdir(parents=True, exist_ok=True)
    ES.load_universe()
    att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
    att = att[att["season"].isin(SEASONS)]
    d = prepare(X.add_exposure(FT.build_ft_design(att)))
    print(f"[{time.time() - t0:5.0f}s] design {d.shape}; bio found {d['_bio_found'].mean():.4f}", flush=True)
    for fold in a.folds.split(","):
        tr, te = FT.fold_slices(d, fold)
        tr, te = tr.copy(), te.copy()
        te_dates = pd.to_datetime(te["game_date"])
        cuts = PO.month_boundaries(te_dates)
        for arm in a.arms.split(","):
            feats = ARMS[arm]
            if arm in ("P2", "P3"):
                prm = fit_prior(tr, conditioned=(arm == "P3"))
                (OUT / f"prior_params_{arm}_{fold}.json").write_text(json.dumps(prm, indent=1), encoding="utf-8")
                tr["prior_season_ft_eb"] = eb_feature(tr, prm)
                te["prior_season_ft_eb"] = eb_feature(te, prm)
                print(f"  {arm} {fold}: k={prm['k']} coef={np.round(prm['coef'], 4).tolist()}", flush=True)
            for seed in (int(s) for s in a.seeds.split(",")):
                path = OUT / f"preds_{arm}_{fold}_s{seed}.parquet"
                if path.exists():
                    print(f"SKIP {path}", flush=True)
                    continue
                t1 = time.time()
                cols = [*feats, "y"]
                p = np.zeros(len(te))
                for kk, cut in enumerate(cuts):
                    nxt = cuts[kk + 1] if kk + 1 < len(cuts) else None
                    seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
                    if not seg.any():
                        continue
                    prior = te.loc[(te_dates < cut).to_numpy(), cols]
                    rows = tr[cols] if not len(prior) else pd.concat([tr[cols], prior], ignore_index=True)
                    m = FT.LgbmArm(seed=seed).fit(FT.design_matrix(rows, tuple(feats)), rows["y"].to_numpy())
                    p[seg] = m.predict_proba(FT.design_matrix(te.loc[seg], tuple(feats)))[:, FT.CLASS_INDEX["MAKE"]]
                if (p == 0).any():
                    raise AssertionError("calendar is not a cover")
                pm = np.column_stack([1 - p, p]) if FT.CLASS_INDEX["MAKE"] == 1 else np.column_stack([p, 1 - p])
                s = FT.score(te, pm)
                out = pd.DataFrame({
                    "game_id": te["game_id"].to_numpy(), "shooter_id": te["shooter_id"].to_numpy(),
                    "game_date": pd.to_datetime(te["game_date"]).to_numpy(),
                    "y": te["y"].to_numpy().astype("float64"), "p": p,
                    "has_prior": te["has_prior_season"].to_numpy(), "fta_asof": te["shooter_fta_asof"].to_numpy(),
                    "prior_fta": te["prior_season_fta"].to_numpy(), "prior_ft": te["prior_season_ft"].to_numpy(),
                    "days_since_start": te["days_since_start"].to_numpy()})
                out.to_parquet(path, index=False)
                meta = {"arm": arm, "fold": fold, "seed": seed, "features": feats, "n_refits": len(cuts),
                        "log_loss": s["log_loss"], "calib_pass": bool(s["calib_pass"]),
                        "calib_worst_gap_pp": s["calib_worst_gap_pp"], "resp_pass": bool(s["resp_pass"]),
                        "slope_ratio": s["responsiveness"]["shooter_ft_asof->MAKE"]["slope_ratio"],
                        "fit_s": round(time.time() - t1, 1)}
                (OUT / f"meta_{arm}_{fold}_s{seed}.json").write_text(json.dumps(meta, indent=1, default=float))
                print(f"[{time.time() - t0:5.0f}s] {arm} {fold} s{seed}: ll {s['log_loss']:.6f} "
                      f"calib {s['calib_worst_gap_pp']} resp {s['resp_pass']} ({meta['fit_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
