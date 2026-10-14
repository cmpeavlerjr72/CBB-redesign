"""train_clock_env_v2.py -- clock experiments.md section 46 block C: S1 monthly schedules for the bounded env arms
ED, EG, EM, EH (`cbb_sim.models.clock_env_v2`) on the served K2 law family (copy of train_clock_env_v1.py; only the arm module, the extra id columns and the output directory differ).

Opus engine worker, 2026-10-14. NOT ADOPTED, NOT A DEFAULT. Same design, horn censoring, first-chance
target, continuation pmf and S1 scheme as `train_clock_r8_chance_v1.py` (K2's trainer, imported, not
edited). Fold 1 trains <= 2022-23 and tests 2023-24 monthly; fold 2 trains <= 2023-24 and tests 2024-25.
2025-26 is never read (the design stops at 2025). Honesty asserted in code:
  * every refit's rows are dated strictly before the refit's first scored date;
  * every feature row's rating snapshot is `as_of_date == game date` (fitted on games strictly before it;
    `own_ratings.join_as_of`, proved identical to the design columns by `diag_clock_env_q1_wiring_v1.py`),
    and that as-of cut (00:00 America/New_York of the game date) is < tipoff_utc for every game that has one
    (created_at < tipoff).
Writes ONLY under data/processed/models/clock/s46C_<arm>/<fold>[_seed<k>]/ (gitignored; HF model_artifacts).

    .venv/Scripts/python.exe scripts/train_clock_env_v2.py --arm ED --fold F2
    .venv/Scripts/python.exe scripts/train_clock_env_v2.py --arm ED --fold F2 --seed 1   # reseed proof
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
sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim.models import clock as ck  # noqa: E402
from cbb_sim.models import clock_v3 as c3  # noqa: E402
from cbb_sim.models import clock_r8 as r8  # noqa: E402
from cbb_sim.models import clock_env as CE  # noqa: E402
from cbb_sim.models import clock_env_v2 as CE2  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
import train_clock_r8_chance_v1 as TK  # noqa: E402

CKD = ROOT / "data/processed/models/clock"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def assert_pretip(design: pd.DataFrame) -> dict:
    """created_at < tipoff: the as-of cut of every row's features is the start of its game date."""
    u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet", columns=["game_id", "tipoff_utc"])
    tip = pd.to_datetime(u.drop_duplicates("game_id").set_index("game_id")["tipoff_utc"], utc=True)
    g = design.drop_duplicates("game_id")[["game_id", "game_date"]]
    cut = pd.to_datetime(g["game_date"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    t = g["game_id"].map(tip)
    have = t.notna().to_numpy()
    bad = int((cut[have].to_numpy() >= t[have].to_numpy()).sum())
    if bad:
        raise AssertionError(f"{bad} games whose rating as-of cut is not before tipoff")
    return {"n_games": int(len(g)), "n_with_tipoff": int(have.sum()), "n_violations": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(CE2.ARMS), required=True)
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_dir = CKD / f"s46C_{a.arm}" / (a.fold if a.seed == 0 else f"{a.fold}_seed{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    design = TK.T.load_design()
    if int(design["season"].max()) > 2025:
        raise AssertionError("sealed season 2025-26 present in the design")
    pretip = assert_pretip(design)
    log(f"pre-tip assertion: {pretip}")
    first, cont = TK.chance_tables(TK.T.ALL_SEASONS)
    n0 = len(design)
    design = TK.build_d1_design(design, first)
    log(f"d1 design {len(design):,} of {n0:,} rows joined; d1-censored {design['censored'].mean():.4%}")
    pinfo = design[[*TK.KEY, "season", "game_date", "censored_poss"]].copy()
    for c_ in ("period", "poss_index"):
        pinfo[c_] = pinfo[c_].astype("int64")
        cont[c_] = cont[c_].astype("int64")
    cont = cont.merge(pinfo, on=TK.KEY, how="inner")
    cont = cont[(cont["period"] <= 2) & (~cont["censored_poss"])]
    test_season = ck.FOLDS[a.fold]["test"][0]
    tr, te = ck.fold_slices(design, a.fold)
    del design
    cols = [c for c in dict.fromkeys([*c3.s1_fit_columns("empirical_km3"), "score_diff", "days_since_start",
                                      "game_id", "end1_code", "n_ch", *CE.RATING_COLS, *CE2.EXTRA_COLS]) if c in tr.columns]
    missing = [c for c in CE.RATING_COLS if c not in cols]
    if missing:
        raise KeyError(f"design lacks {missing}")
    tr_small = tr[cols]
    norms = CE.fold_norms(tr_small)
    log(f"fold norms (training seasons {ck.FOLDS[a.fold]['train']}): {norms}")
    te_dates = pd.to_datetime(te["game_date"])
    tr_dates = pd.to_datetime(tr["game_date"])
    cont_dates = pd.to_datetime(cont["game_date"])
    cont_train = cont["season"].isin(ck.FOLDS[a.fold]["train"]).to_numpy()
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
        mx = pd.to_datetime(fit_rows["game_date"]).max() if "game_date" in fit_rows.columns else (
            tr_dates.max() if not before.any() else max(tr_dates.max(), te_dates[before].max()))
        if not pd.Timestamp(mx) < pd.Timestamp(cut):
            raise AssertionError(f"refit {cut}: training row dated {mx} not before the refit cut")
        cm = cont_train | ((cont["season"] == test_season).to_numpy() & (cont_dates < cut).to_numpy())
        if not (cont_dates[cm] < cut).all():
            raise AssertionError("continuation rows not before the refit cut")
        cpmf = r8.cont_pmf_from(cont.loc[cm, "duration_s"].to_numpy(), cont.loc[cm, "grp"].to_numpy())
        t0 = time.time()
        arm = CE2.fit_env_arm(a.arm, fit_rows, cpmf, norms, seed=a.seed)
        stamp = f"{pd.Timestamp(cut).year:04d}-{pd.Timestamp(cut).month:02d}"
        mfile = out_dir / f"s46C{a.arm}_S1_{test_season}_{stamp}.pkl"
        with open(mfile, "wb") as f:
            pickle.dump(arm, f)
        months.append({
            "refit_date": str(pd.Timestamp(cut).date()), "valid_from": str(pd.Timestamp(cut).date()),
            "valid_to": None if nxt is None else str((pd.Timestamp(nxt) - pd.Timedelta(days=1)).date()),
            "n_train": int(len(fit_rows)), "n_train_from_test_season": int(len(prior)),
            "n_cont_rows": int(cm.sum()), "mean_cont": arm.mean_cont().round(3).tolist(),
            "n_scored": int(seg.sum()), "max_train_date": str(pd.Timestamp(mx).date()),
            "season": test_season, "month": stamp,
            "model_file": str(mfile.relative_to(CKD)).replace("\\", "/"),
            "fit_seconds": round(time.time() - t0, 1), **arm.info})
        log(f"{a.arm} {a.fold} refit {stamp}: {len(fit_rows):,} rows, coef {arm.info['coef']}, "
            f"k_sd {arm.info['k_sd']:.4f}, {months[-1]['fit_seconds']}s")
        del fit_rows, prior, arm
    manifest = {"base_arm": f"clock_env_v2_{a.arm}", "parametrisation": "P3", "tag": f"s46C{a.arm}",
                "scheme": "S1", "season": test_season, "fold": a.fold, "round": "s46 block C",
                "seed": a.seed, "arm_adopted": False, "pretip_assertion": pretip, "fold_norms": norms,
                "note": "clock s46 block C bounded env AFT on the K2 law; NOT adopted, NOT a default, not wired",
                "months": months}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
