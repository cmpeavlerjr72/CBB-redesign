#!/usr/bin/env python
"""
train_free_throw_v4_exposure.py -- FT early-season exposure round. Pre-registration:
`docs/models/free_throw/experiments.md` section 18 (committed 120bee9 BEFORE this ran).

    CBB_NJOBS=4 .venv/Scripts/python.exe scripts/train_free_throw_v4_exposure.py --arms X0,X1,X2,X3 --folds F2,F1 --seeds 0
    CBB_NJOBS=4 .venv/Scripts/python.exe scripts/train_free_throw_v4_exposure.py --arms X0 --folds F2,F1 --seeds 1

Arms (S1_monthly calendar for all, the stated cost deviation of sections 11/13/18):
    X0  served FT_FEATURES
    X1  + days_since_start   (engine team column; source = possession_outcome/design.parquet, as build_engine_inputs)
    X2  + shooter_games_asof (engine slot column; fg_make design_v2_shotshooter, joined backward as N2 in section 13)
    X3  + both

Writes predictions only: results/ft_exposure/preds_<arm>_<fold>_s<seed>.parquet and meta_*.json. Sibling of
`train_free_throw_v3_newcomer.py` (not edited). Nothing goes to an engine directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

NJ = int(os.environ.get("CBB_NJOBS", "1"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(NJ)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

FT.LgbmArm.PARAMS = {**FT.LgbmArm.PARAMS, "n_jobs": NJ}
OUT = Path("results/ft_exposure")
SEASONS = [2022, 2023, 2024, 2025]
BASE = list(FT.FT_FEATURES)
ARMS = {"X0": BASE, "X1": BASE + ["days_since_start"], "X2": BASE + ["shooter_games_asof"],
        "X3": BASE + ["days_since_start", "shooter_games_asof"]}


def add_exposure(d: pd.DataFrame) -> pd.DataFrame:
    po = pd.read_parquet("data/processed/models/possession_outcome/design.parquet",
                         columns=["game_id", "days_since_start"]).drop_duplicates("game_id")
    out = d.merge(po, on="game_id", how="left")
    assert len(out) == len(d)
    out["_dss_found"] = out["days_since_start"].notna()
    out["days_since_start"] = out["days_since_start"].astype("float32")
    fg = pd.read_parquet("data/processed/models/fg_make/design_v2_shotshooter.parquet",
                         columns=["season", "game_id", "game_date", "shooter_id", "shooter_games_asof"])
    fg = fg.dropna(subset=["shooter_id"])
    fg["shooter_id"] = fg["shooter_id"].astype("int64")
    fg["game_date"] = pd.to_datetime(fg["game_date"])
    right = fg.drop_duplicates(["season", "shooter_id", "game_date"]).sort_values("game_date", kind="stable")[
        ["season", "shooter_id", "game_date", "shooter_games_asof"]]
    left = out[["season", "shooter_id", "game_date"]].copy()
    left["_row"] = np.arange(len(out))
    left["game_date"] = pd.to_datetime(left["game_date"])
    left = left.sort_values("game_date", kind="stable")
    m = pd.merge_asof(left, right, on="game_date", by=["season", "shooter_id"], direction="backward",
                      allow_exact_matches=True).sort_values("_row")
    out["_sga_found"] = m["shooter_games_asof"].notna().to_numpy()
    out["shooter_games_asof"] = m["shooter_games_asof"].fillna(0.0).to_numpy().astype("float32")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="X0,X1,X2,X3")
    ap.add_argument("--folds", default="F2,F1")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    t0 = time.time()
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="free_throw exposure round")
    OUT.mkdir(parents=True, exist_ok=True)
    ES.load_universe()
    att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
    att = att[att["season"].isin(SEASONS)]
    d = FT.build_ft_design(att)
    d = add_exposure(d)
    cov = {"dss_found": float(d["_dss_found"].mean()), "sga_found": float(d["_sga_found"].mean())}
    print(f"[{time.time() - t0:5.0f}s] design {d.shape}; coverage {cov}", flush=True)
    for fold in a.folds.split(","):
        tr, te = FT.fold_slices(d, fold)
        te_dates = pd.to_datetime(te["game_date"])
        cuts = PO.month_boundaries(te_dates)
        for arm in a.arms.split(","):
            feats = ARMS[arm]
            for seed in (int(s) for s in a.seeds.split(",")):
                path = OUT / f"preds_{arm}_{fold}_s{seed}.parquet"
                if path.exists():
                    print(f"SKIP {path}", flush=True)
                    continue
                t1 = time.time()
                cols = [*feats, "y"]
                p = np.zeros(len(te))
                for k, cut in enumerate(cuts):
                    nxt = cuts[k + 1] if k + 1 < len(cuts) else None
                    seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
                    if not seg.any():
                        continue
                    before = (te_dates < cut).to_numpy()
                    prior = te.loc[before, cols]
                    rows = tr[cols] if not len(prior) else pd.concat([tr[cols], prior], ignore_index=True)
                    m = FT.LgbmArm(seed=seed).fit(FT.design_matrix(rows, tuple(feats)), rows["y"].to_numpy())
                    p[seg] = m.predict_proba(FT.design_matrix(te.loc[seg], tuple(feats)))[:, FT.CLASS_INDEX["MAKE"]]
                if (p == 0).any():
                    raise AssertionError("calendar is not a cover")
                pm = np.column_stack([1 - p, p]) if FT.CLASS_INDEX["MAKE"] == 1 else np.column_stack([p, 1 - p])
                s = FT.score(te, pm)
                out = pd.DataFrame({
                    "game_id": te["game_id"].to_numpy(), "team_id": te["team_id"].to_numpy(),
                    "shooter_id": te["shooter_id"].to_numpy(), "season": te["season"].to_numpy(),
                    "game_date": pd.to_datetime(te["game_date"]).to_numpy(),
                    "y": te["y"].to_numpy().astype("float64"), "p": p,
                    "has_prior": te["has_prior_season"].to_numpy(), "fta_asof": te["shooter_fta_asof"].to_numpy(),
                    "ft_asof": te["shooter_ft_asof"].to_numpy(), "prior_ft": te["prior_season_ft"].to_numpy(),
                    "days_since_start": te["days_since_start"].to_numpy()})
                out.to_parquet(path, index=False)
                meta = {"arm": arm, "fold": fold, "seed": seed, "features": feats, "n_refits": len(cuts),
                        "log_loss": s["log_loss"], "calib_pass": bool(s["calib_pass"]),
                        "calib_worst_gap_pp": s["calib_worst_gap_pp"], "resp_pass": bool(s["resp_pass"]),
                        "slope_ratio": s["responsiveness"]["shooter_ft_asof->MAKE"]["slope_ratio"],
                        "coverage": cov, "fit_s": round(time.time() - t1, 1)}
                (OUT / f"meta_{arm}_{fold}_s{seed}.json").write_text(json.dumps(meta, indent=1, default=float))
                print(f"[{time.time() - t0:5.0f}s] {arm} {fold} s{seed}: ll {s['log_loss']:.6f} "
                      f"calib {s['calib_worst_gap_pp']} resp {s['resp_pass']} ({meta['fit_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
