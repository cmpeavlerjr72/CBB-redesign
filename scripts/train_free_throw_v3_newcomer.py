#!/usr/bin/env python
"""
train_free_throw_v3_newcomer.py -- lane I (2026-10-01): FT newcomer-prior arms. Pre-registration:
`docs/models/free_throw/experiments.md` section 13 (committed 4716a08 BEFORE this ran).

    CBB_NJOBS=1 .venv/Scripts/python.exe scripts/train_free_throw_v3_newcomer.py --arms N0,N2 --folds F2,F1 --seeds 0
    CBB_NJOBS=1 .venv/Scripts/python.exe scripts/train_free_throw_v3_newcomer.py --arms N0 --folds F2,F1 --seeds 1

Arms (S1_monthly calendar for all, the stated cost deviation of section 11/13):
    N0  served FT_FEATURES
    N2  + shooter_make_c__three, shooter_att_c__three, prior_season_make_c__three (fg_make design FGA_3 rows, as-of)
    N1  + height_c, pos_G, pos_F, pos_C, d1_years (CBBD rosters)
    N3  N1 + N2

Writes predictions only: results/laneI_1001/ft/preds_<arm>_<fold>_s<seed>.parquet and meta_*.json. Sibling of
`train_free_throw_v2_site.py` (not edited). Nothing goes to an engine directory.
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
OUT = Path("results/laneI_1001/ft")
SEASONS = [2022, 2023, 2024, 2025]
BASE = list(FT.FT_FEATURES)
SHOOT = ["shooter_make_c__three", "shooter_att_c__three", "prior_season_make_c__three"]
BIO = ["height_c", "pos_G", "pos_F", "pos_C", "d1_years"]
ARMS = {"N0": BASE, "N2": BASE + SHOOT, "N1": BASE + BIO, "N3": BASE + BIO + SHOOT}


def add_shooting(d: pd.DataFrame) -> pd.DataFrame:
    fg = pd.read_parquet("data/processed/models/fg_make/design_v2_shotshooter.parquet",
                         columns=["season", "game_id", "game_date", "shooter_id", "shot_class", "shooter_make_c",
                                  "shooter_att_c", "prior_season_make_c"])
    cls = [c for c in fg["shot_class"].unique() if "3" in str(c)]
    assert len(cls) == 1, cls
    fg = fg[fg["shot_class"] == cls[0]].dropna(subset=["shooter_id"])
    fg["shooter_id"] = fg["shooter_id"].astype("int64")
    fg["game_date"] = pd.to_datetime(fg["game_date"])
    per_game = fg.drop_duplicates(["season", "shooter_id", "game_id"])
    # prior-season make: a constant of (season, shooter), entirely in the past -> fill for every row of that pair
    prior = per_game.groupby(["season", "shooter_id"])["prior_season_make_c"].first().rename("prior_season_make_c__three")
    left = d[["season", "shooter_id", "game_date"]].copy()
    left["_row"] = np.arange(len(d))
    left["game_date"] = pd.to_datetime(left["game_date"])
    left = left.sort_values("game_date", kind="stable")
    right = per_game.sort_values("game_date", kind="stable")[["season", "shooter_id", "game_date", "shooter_make_c",
                                                               "shooter_att_c"]]
    m = pd.merge_asof(left, right, on="game_date", by=["season", "shooter_id"], direction="backward",
                      allow_exact_matches=True).sort_values("_row")
    out = d.copy()
    out["shooter_make_c__three"] = m["shooter_make_c"].fillna(0.0).to_numpy().astype("float32")
    out["shooter_att_c__three"] = m["shooter_att_c"].fillna(0.0).to_numpy().astype("float32")
    pr = out[["season", "shooter_id"]].merge(prior.reset_index(), on=["season", "shooter_id"], how="left")
    out["prior_season_make_c__three"] = pr["prior_season_make_c__three"].fillna(0.0).to_numpy().astype("float32")
    return out


def add_bio(d: pd.DataFrame) -> pd.DataFrame:
    rs = []
    for s in SEASONS:
        r = pd.read_parquet(f"data/raw/cbbd/rosters/roster_{s}.parquet",
                            columns=["season", "cbbd_player_id", "position", "height", "start_season"])
        rs.append(r)
    r = pd.concat(rs).dropna(subset=["cbbd_player_id"])
    r["cbbd_player_id"] = r["cbbd_player_id"].astype("int64")
    r = r.drop_duplicates(["season", "cbbd_player_id"])
    r["height_c"] = r["height"] - r.groupby("season")["height"].transform("mean")
    p = r["position"].fillna("").str.upper().str[0]
    r["pos_G"], r["pos_F"], r["pos_C"] = [(p == k).astype("float32") for k in ("G", "F", "C")]
    r["d1_years"] = np.clip(r["season"] - r["start_season"], 0, 4).astype("float32")
    r.loc[r["start_season"].isna(), "d1_years"] = np.nan
    j = d[["season", "shooter_id"]].merge(r.rename(columns={"cbbd_player_id": "shooter_id"})[
        ["season", "shooter_id", "height_c", "pos_G", "pos_F", "pos_C", "d1_years"]], on=["season", "shooter_id"], how="left")
    out = d.copy()
    for c in BIO:
        v = j[c].to_numpy().astype("float32")
        out[c] = np.where(np.isnan(v), 0.0, v).astype("float32") if c.startswith("pos_") else v
    out["_bio_found"] = j["height_c"].notna().to_numpy() | j["pos_G"].notna().to_numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="N0,N2,N1,N3")
    ap.add_argument("--folds", default="F2,F1")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    t0 = time.time()
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="free_throw newcomer round")
    OUT.mkdir(parents=True, exist_ok=True)
    ES.load_universe()
    att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
    att = att[att["season"].isin(SEASONS)]
    d = FT.build_ft_design(att)
    d = add_bio(add_shooting(d))
    cov = {"bio_found": float(d["_bio_found"].mean()),
           "three_asof_nonzero": float((d["shooter_att_c__three"] != 0).mean()),
           "three_prior_nonzero": float((d["prior_season_make_c__three"] != 0).mean())}
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
                    "ft_asof": te["shooter_ft_asof"].to_numpy(), "prior_ft": te["prior_season_ft"].to_numpy()})
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
