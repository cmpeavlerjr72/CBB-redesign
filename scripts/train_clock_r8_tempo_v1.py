"""train_clock_r8_tempo_v1.py -- clock round 8 (experiments.md section 36): S1 monthly
schedules for M1, M2, M2D, G2D (mean-scale AFT team tempo, in-season term, game-level
elasticity factor). C0 and A2 are round 7's schedules, unchanged.

Lane H, 2026-10-01. Same inputs and S1 scheme as `train_clock_r7_pace_v1.py` (the v4 L2
design with its own horn censoring). Writes ONLY under
data/processed/models/clock/r8_<arm>/<fold>[_seed<k>]/ (gitignored). Threads pinned to 1.

    .venv/Scripts/python.exe scripts/train_clock_r8_tempo_v1.py --arm M2D --fold F2
    .venv/Scripts/python.exe scripts/train_clock_r8_tempo_v1.py --arm M2D --fold F2 --seed 1   # reseed floor proof
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
from cbb_sim.models import clock_r8 as r8  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

CKD = ROOT / "data/processed/models/clock"
L2 = CKD / "r6_L2"
ALL_SEASONS = [2022, 2023, 2024, 2025]
CHUNK = 20000


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_design() -> pd.DataFrame:
    design = pd.read_parquet(L2 / "design_v2.parquet")
    design, cdiag = c3.attach_horn_censoring(design, ALL_SEASONS, censor_dir=L2 / "clock_censoring")
    c3.set_flag_inplace(design, "horn")
    design["days_since_start"] = r8.season_day(design)
    log(f"design {len(design):,} rows; horn-censored {cdiag['censored_horn_pct']}%")
    return design


def expected_min(arm, d: pd.DataFrame) -> np.ndarray:
    grid = np.arange(ck.DURATION_CAP + 1, dtype=np.float64)
    e = np.empty(len(d))
    for lo in range(0, len(d), CHUNK):
        blk = d.iloc[lo:lo + CHUNK].reset_index(drop=True)
        p = np.asarray(arm.pmf(blk), dtype=np.float64)
        R = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
        e[lo:lo + len(blk)] = (p * np.minimum(grid[None, :], R)).sum(axis=1)
    return e


def fold_lambda(tr: pd.DataFrame) -> dict:
    """G2D: lambda = e_act / e_law on the fold's TRAINING seasons' clock-complete regulation
    games, the law being M2D fitted on the same training rows (the fold's first refit)."""
    univ = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet")
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    law = r8.fit_arm("M2D", tr)
    d = tr[(tr["period"] <= 2) & tr["game_id"].isin(cc)].reset_index(drop=True)
    e = expected_min(law, d)
    ge = r8.game_elasticity(d, e)
    ge["lambda"] = ge["e_act"] / ge["e_law"]
    return ge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(r8.ARM_SPEC), required=True)
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="spec-identical reseed (the fit has no RNG; proves the floor is 0)")
    a = ap.parse_args()
    np.random.seed(a.seed)
    out_dir = CKD / f"r8_{a.arm}" / (a.fold if a.seed == 0 else f"{a.fold}_seed{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    design = load_design()
    test_season = ck.FOLDS[a.fold]["test"][0]
    tr, te = ck.fold_slices(design, a.fold)
    del design
    cols = [c for c in dict.fromkeys([*c3.s1_fit_columns("empirical_km3"), "score_diff",
                                      "days_since_start", "game_id"]) if c in tr.columns]
    tr_small = tr[cols]
    lam_info = {}
    if a.arm == "G2D":
        t0 = time.time()
        lam_info = fold_lambda(tr_small)
        log(f"G2D lambda {json.dumps(lam_info)} ({time.time() - t0:.0f}s)")
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
        arm = r8.fit_arm(a.arm, fit_rows, lam=lam_info.get("lambda"))
        info = {"coef": arm.coef.tolist(), **arm.info}
        stamp = f"{pd.Timestamp(cut).year:04d}-{pd.Timestamp(cut).month:02d}"
        mfile = out_dir / f"r8{a.arm}_S1_{test_season}_{stamp}.pkl"
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
        log(f"{a.arm} {a.fold} refit {stamp}: {len(fit_rows):,} rows, {months[-1]['fit_seconds']}s, "
            f"coef {np.round(arm.coef, 3).tolist()}")
        del fit_rows, prior, arm
    manifest = {"base_arm": f"clock_r8_{a.arm}", "parametrisation": "P3", "tag": f"r8{a.arm}",
                "scheme": "S1", "season": test_season, "fold": a.fold, "round": "8", "seed": a.seed,
                "arm_adopted": False, "lambda": lam_info,
                "note": "clock round 8 (experiments.md section 36); NOT adopted, NOT a default",
                "months": months}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
