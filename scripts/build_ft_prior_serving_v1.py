#!/usr/bin/env python
"""build_ft_prior_serving_v1.py -- free_throw section 20 arm P1 (served FT_FEATURES + prior_season_fta):
S1_conf_aligned serving artifacts (F2 and F1) and versioned engine-input siblings with the appended slot column.

P1 did NOT win section 20 (no arm did); this serves it for a DESCRIPTIVE closed-loop read only.

    .venv/Scripts/python.exe scripts/build_ft_prior_serving_v1.py artifacts F2|F1
    .venv/Scripts/python.exe scripts/build_ft_prior_serving_v1.py inputs F2|F1

artifacts -> data/processed/models/free_throw/s1_scorediff/P1/S1_conf_aligned/<fold>/ (served by the existing default-off
             ENGINE_FT_SCORE=P1 loader; nothing served changes)
inputs    -> data/processed/models/engine_v3_FTP_P1 (copy of engine_v3) / engine_v3_f1_FTP_P1 (copy of engine_v3_f1),
             slot column `prior_season_fta` appended: the completed prior season's non-technical FTA of the slot's
             cbbd player (the design's prev_fta); 0 for anonymous slots and players with no prior season.
             Served input dirs are read only.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.chdir(ROOT)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train_free_throw_v2_s1 as T  # noqa: E402

FT = T.FT
ART = ROOT / "data/processed/models/free_throw/s1_scorediff"
ARM = "P1"
FEATS = (*FT.FT_FEATURES, "prior_season_fta")
SEASON = {"F2": 2025, "F1": 2024}
SRC_IN = {"F2": "data/processed/models/engine_v3", "F1": "data/processed/models/engine_v3_f1"}
DST_IN = {"F2": "data/processed/models/engine_v3_FTP_P1", "F1": "data/processed/models/engine_v3_f1_FTP_P1"}
_orig_dm = FT.design_matrix


def design() -> pd.DataFrame:
    T.ES.load_universe()
    att = pd.read_parquet(T.OUT_DIR / "attempts_v1_era.parquet")
    att = att[att["season"].isin(T.SEASONS)]
    return FT.build_ft_design(att)


def artifacts(fold: str) -> None:
    FT.FT_FEATURES = FEATS
    FT.design_matrix = lambda d, features=None: _orig_dm(d, tuple(features) if features is not None else FEATS)
    T.S1_DIR = ART / ARM
    d = design()
    conf_all = T.CF.build_conference_flags(T.SEASONS)
    firsts = T.CF.first_conference_game_dates(conf_all)
    tr, te = FT.fold_slices(d, fold)
    row = T.run_cell("S1_conf_aligned", fold, tr, te, conf_all, firsts, SEASON[fold], seed=0)
    row = {k: v for k, v in row.items() if k not in ("conf4", "clean_trip")}
    row["features"] = list(FEATS)
    (ART / ARM / f"build_report_{fold}.json").write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
    print(json.dumps(row, default=str)[:1200])


def inputs(fold: str) -> None:
    season = SEASON[fold]
    src, dst = ROOT / SRC_IN[fold], ROOT / DST_IN[fold]
    tag = f"{fold}_{season}"
    if dst.exists():
        raise SystemExit(f"{dst} exists; versioned dirs are never overwritten")
    att = pd.read_parquet(T.OUT_DIR / "attempts_v1_era.parquet")
    att = att[(att["season"] == season - 1) & (att["foul_class"] != "technical")]
    att = att[np.isfinite(att["shooter_id"].to_numpy())]
    prev = att.groupby(att["shooter_id"].astype("int64")).size()
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in (f"arrays_{tag}.npz", f"names_{tag}.json"):
            shutil.copy2(f, dst / f.name)
    z = dict(np.load(src / f"arrays_{tag}.npz"))
    names = json.loads((src / f"names_{tag}.json").read_text(encoding="utf-8"))
    ss = z["slot_static"]
    pid = z["roster_cbbd"].astype("int64")
    col = np.where(pid > 0, pd.Series(pid.reshape(-1)).map(prev).fillna(0).to_numpy().reshape(pid.shape), 0.0)
    z["slot_static"] = np.concatenate([ss, col[..., None].astype(np.float32)], axis=3)
    names["slot_names"]["prior_season_fta"] = ss.shape[3]
    np.savez(dst / f"arrays_{tag}.npz", **z)
    (dst / f"names_{tag}.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    # serving check: slot value == design prior_season_fta for named real shooters of this season
    d = design()
    d = d[d["season"] == season][["game_id", "shooter_id", "prior_season_fta"]].drop_duplicates()
    games = pd.read_parquet(src / f"games_{tag}.parquet")
    flat = pd.DataFrame({"game_id": np.repeat(games["game_id"].to_numpy(), pid.shape[1] * pid.shape[2]),
                         "shooter_id": pid.reshape(-1), "slot_val": col.reshape(-1)})
    j = d.merge(flat[flat["shooter_id"] > 0], on=["game_id", "shooter_id"], how="inner")
    rep = {"fold": fold, "dst": str(dst), "slot_names_width": int(z["slot_static"].shape[3]),
           "checked_rows": int(len(j)), "max_abs_diff": float(np.abs(j["slot_val"] - j["prior_season_fta"]).max()),
           "unchanged_columns_identical": bool(np.array_equal(z["slot_static"][..., :ss.shape[3]], ss)),
           "nonzero_slot_share": float((col > 0).mean())}
    (dst / "builder_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    {"artifacts": artifacts, "inputs": inputs}[sys.argv[1]](sys.argv[2])
