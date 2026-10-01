"""train_clock_r7_pace_v1.py -- clock round 7 (experiments.md section 32): S1 schedules
for C0 (L2 refit), A1 (continuous symmetric AFT tempo scale), A2 (offence/defence
asymmetric AFT scale) and A3 (A2 + walk-forward partial-pooled team effects).

Lane H, 2026-09-30. Builds on lane B's `exp_clk6_r6_arms_v1.py` / lane D's
`train_clock_chain_v1.py` inputs: the v4 L2 design and its own horn-censoring
table. Writes ONLY under data/processed/models/clock/r7_<arm>/<fold>/ (gitignored);
nothing served is touched. The fold-2 manifest has the same format as
`train_clock_v3c_s1.py`'s so `clock_adapter_v3` can serve it unchanged.

Usage (one arm and fold per process; threads pinned to 1):
    .venv/Scripts/python.exe scripts/train_clock_r7_pace_v1.py --arm A2 --fold F2
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.models import clock as ck  # noqa: E402
from cbb_sim.models import clock_v3 as c3  # noqa: E402
from cbb_sim.models import clock_r7 as r7  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

CKD = ROOT / "data/processed/models/clock"
L2 = CKD / "r6_L2"
ALL_SEASONS = [2022, 2023, 2024, 2025]
BASE_SEED = 20260910


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_design() -> pd.DataFrame:
    design = pd.read_parquet(L2 / "design_v2.parquet")
    design, cdiag = c3.attach_horn_censoring(design, ALL_SEASONS, censor_dir=L2 / "clock_censoring")
    c3.set_flag_inplace(design, "horn")
    log(f"design {len(design):,} rows; horn-censored {cdiag['censored_horn_pct']}%")
    return design


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["C0", "A1", "A2", "A3"], required=True)
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    a = ap.parse_args()
    out_dir = CKD / f"r7_{a.arm}" / a.fold
    out_dir.mkdir(parents=True, exist_ok=True)
    design = load_design()
    test_season = ck.FOLDS[a.fold]["test"][0]
    params = {}
    if a.arm == "A3":
        re, params = r7.team_re(design, ck.FOLDS[a.fold]["train"])
        design["re_o"] = re["re_o"].to_numpy()
        design["re_d"] = re["re_d"].to_numpy()
        log(f"A3 team effects: {json.dumps(params)}")
        design[["game_id", "offense_team_id", "defense_team_id", "season", "game_date",
                "re_o", "re_d"]].drop_duplicates(["game_id", "offense_team_id"]).to_parquet(
            out_dir / "team_re_side.parquet", index=False)
    tr, te = ck.fold_slices(design, a.fold)
    del design
    if a.arm == "C0":
        cols = c3.s1_fit_columns("empirical_km3")
        cols = [c for c in dict.fromkeys([*cols, *ck.feature_set(c3.P_FEATURES["P3"])]) if c in tr.columns]
    else:
        cols = [c for c in dict.fromkeys([*c3.s1_fit_columns("empirical_km3"), "score_diff",
                                          "re_o", "re_d"]) if c in tr.columns]
    tr_small = tr[cols]
    te_dates = pd.to_datetime(te["game_date"])
    tr_dates = pd.to_datetime(tr["game_date"])
    cuts = PO.month_boundaries(te_dates)
    months = []
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not seg.any():
            continue
        before = (te_dates < cut).to_numpy()
        prior = te.loc[before, cols]
        fit_rows = tr_small if not len(prior) else pd.concat([tr_small, prior], ignore_index=True)
        t0 = time.time()
        if a.arm == "C0":
            arm = c3.fit_arm_v3b("empirical_km3_srfloor", "P3", fit_rows, seed=BASE_SEED)
            info = {}
        else:
            arm = r7.fit_arm(a.arm, fit_rows)
            info = {"coef": arm.coef.tolist(), **arm.info}
        stamp = f"{pd.Timestamp(cut).year:04d}-{pd.Timestamp(cut).month:02d}"
        mfile = out_dir / f"r7{a.arm}_S1_{test_season}_{stamp}.pkl"
        with open(mfile, "wb") as f:
            pickle.dump(arm, f)
        mx = tr_dates.max() if not before.any() else max(tr_dates.max(), te_dates[before].max())
        months.append({
            "refit_date": str(pd.Timestamp(cut).date()), "valid_from": str(pd.Timestamp(cut).date()),
            "valid_to": None if nxt is None else str((pd.Timestamp(nxt) - pd.Timedelta(days=1)).date()),
            "n_train": int(len(fit_rows)), "n_train_from_test_season": int(len(prior)),
            "n_scored": int(seg.sum()), "max_train_date": str(pd.Timestamp(mx).date()),
            "season": test_season, "month": stamp,
            "model_file": str(mfile.relative_to(CKD)).replace("\\", "/"),
            "fit_seconds": round(time.time() - t0, 1), **info})
        log(f"{a.arm} {a.fold} refit {stamp}: {len(fit_rows):,} rows, {months[-1]['fit_seconds']}s")
        del fit_rows, prior, arm
    manifest = {"base_arm": "empirical_km3_srfloor" if a.arm == "C0" else f"clock_r7_{a.arm}",
                "parametrisation": "P3", "tag": f"r7{a.arm}", "scheme": "S1", "season": test_season,
                "fold": a.fold, "round": "7", "arm_adopted": False, "team_re_params": params,
                "note": "clock round 7 (experiments.md section 32); NOT adopted, NOT a default",
                "months": months}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
