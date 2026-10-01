"""train_clock_s1_fold_v1.py -- the served clock S1 schedule (srfloor_P3, monthly refits) for ANY fold, into a root
that the engine serves by rebinding CK_DIR (lane D, 2026-10-01; fold-1 confirmation, engine experiments.md s2).

Fit loop = lane H's `train_clock_r7_pace_v1.py --arm C0` (whose fold-2 pickles are byte-equal to `clock/r6_L2`),
with the design root as a parameter instead of hard-coded r6_L2:
  --spec v5b  design `clock/design_v2.parquet` (possessions v1, served ratings) + `data/processed/clock_censoring`
              -> <out>/v3c_s1/manifest_srfloor_P3.json + pickles   (served by ENGINE_CLOCK=v5b_glat_pmean)
  --spec L2   design `clock/r6_L2/design_v2.parquet` (possessions v4) + `clock/r6_L2/clock_censoring`
              -> <out>/r6_L2/v3c_s1/manifest_srfloor_P3.json + pickles (served by ENGINE_CLOCK=v5b_r6L2_glat_pmean)
The latent sigma is NOT refitted: the served reports (`clock/v5b_bakeoff/...`, `clock/r6_L2/v5b_bakeoff/...`) already
carry a fold-1 training-row fit under params.F1; they are copied into the root and read at
`clock_adapter_v3.PARAMS_FOLD = "F1"` (overlay). `v5_bakeoff/v5_bakeoff_report.json` is copied (existence check only).

Identity: `--fold F2 --identity` must give pickles byte-equal to the served schedule of the spec.

    .venv/Scripts/python.exe scripts/train_clock_s1_fold_v1.py --spec v5b --fold F1 --out data/processed/models/fold1_v1/clock
    .venv/Scripts/python.exe scripts/train_clock_s1_fold_v1.py --spec L2  --fold F1 --out data/processed/models/fold1_v1/clock
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.models import clock as ck  # noqa: E402
from cbb_sim.models import clock_v3 as c3  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

CKD = ROOT / "data/processed/models/clock"
ALL_SEASONS = [2022, 2023, 2024, 2025]
BASE_SEED = 20260910
SPECS = {
    "v5b": {"design": CKD / "design_v2.parquet", "censor": ROOT / "data/processed/clock_censoring", "sub": "",
            "served": CKD / "v3c_s1", "report": CKD / "v5b_bakeoff/v5b_bakeoff_report.json"},
    "L2": {"design": CKD / "r6_L2/design_v2.parquet", "censor": CKD / "r6_L2/clock_censoring", "sub": "r6_L2",
           "served": CKD / "r6_L2/v3c_s1", "report": CKD / "r6_L2/v5b_bakeoff/v5b_bakeoff_report.json"},
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=sorted(SPECS), required=True)
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--out", type=Path, required=True, help="clock root (served layout)")
    ap.add_argument("--identity", action="store_true")
    a = ap.parse_args()
    sp = SPECS[a.spec]
    out = a.out if a.out.is_absolute() else ROOT / a.out
    base = out / sp["sub"] if sp["sub"] else out
    s1 = base / "v3c_s1"
    s1.mkdir(parents=True, exist_ok=True)
    design = pd.read_parquet(sp["design"])
    design, cdiag = c3.attach_horn_censoring(design, ALL_SEASONS, censor_dir=sp["censor"])
    c3.set_flag_inplace(design, "horn")
    log(f"{a.spec}: design {len(design):,} rows; horn-censored {cdiag['censored_horn_pct']}%")
    test_season = ck.FOLDS[a.fold]["test"][0]
    tr, te = ck.fold_slices(design, a.fold)
    del design
    cols = c3.s1_fit_columns("empirical_km3")
    cols = [c for c in dict.fromkeys([*cols, *ck.feature_set(c3.P_FEATURES["P3"])]) if c in tr.columns]
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
        arm = c3.fit_arm_v3b("empirical_km3_srfloor", "P3", fit_rows, seed=BASE_SEED)
        stamp = f"{pd.Timestamp(cut).year:04d}-{pd.Timestamp(cut).month:02d}"
        mfile = s1 / f"srfloor_P3_S1_{test_season}_{stamp}.pkl"
        with open(mfile, "wb") as f:
            pickle.dump(arm, f)
        mx = tr_dates.max() if not before.any() else max(tr_dates.max(), te_dates[before].max())
        months.append({
            "refit_date": str(pd.Timestamp(cut).date()), "valid_from": str(pd.Timestamp(cut).date()),
            "valid_to": None if nxt is None else str((pd.Timestamp(nxt) - pd.Timedelta(days=1)).date()),
            "n_train": int(len(fit_rows)), "n_train_from_test_season": int(len(prior)),
            "n_scored": int(seg.sum()), "max_train_date": str(pd.Timestamp(mx).date()),
            "season": test_season, "month": stamp,
            "model_file": str(mfile.relative_to(out)).replace("\\", "/"),
            "fit_seconds": round(time.time() - t0, 1)})
        log(f"{a.spec} {a.fold} refit {stamp}: {len(fit_rows):,} rows, {months[-1]['fit_seconds']}s")
        del fit_rows, prior, arm
    manifest = {"base_arm": "empirical_km3_srfloor", "parametrisation": "P3", "tag": f"fold_{a.spec}",
                "scheme": "S1", "season": test_season, "fold": a.fold, "round": "3c-spec (lane D fold sibling)",
                "arm_adopted": False, "note": f"served {a.spec} clock spec refitted for fold {a.fold} by "
                                              "scripts/train_clock_s1_fold_v1.py; model_file is relative to the root",
                "months": months}
    (s1 / "manifest_srfloor_P3.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (base / "v5b_bakeoff").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sp["report"], base / "v5b_bakeoff/v5b_bakeoff_report.json")
    (out / "v5_bakeoff").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CKD / "v5_bakeoff/v5_bakeoff_report.json", out / "v5_bakeoff/v5_bakeoff_report.json")
    rep = {"spec": a.spec, "fold": a.fold, "out": str(out), "n_months": len(months)}
    if a.identity:
        h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        rep["identity"] = {m["month"]: h(s1 / Path(m["model_file"]).name) ==
                           h(sp["served"] / f"srfloor_P3_S1_{test_season}_{m['month']}.pkl") for m in months}
        rep["IDENTICAL"] = all(rep["identity"].values())
    log(json.dumps(rep))
    (base / f"fold_clock_{a.spec}_{a.fold}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
