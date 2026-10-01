"""train_clock_r8_chance_v1.py -- clock round 8 amendment (experiments.md section 38): S1 monthly
schedules for K1, K2, K2M (outcome-conditioned possession time).

Lane H, 2026-10-01. Same design / censoring / S1 scheme as `train_clock_r8_tempo_v1.py`; the
target is the FIRST-CHANCE duration d1 from chances_v4 (censored only when the possession has one
chance and was horn-censored), with the chance-1 end class; the continuation pmf comes from the
chance >= 2 rows (regulation, possessions not horn-censored) on the same rows the refit may see.
Writes ONLY under data/processed/models/clock/r8_<arm>/<fold>[_seed<k>]/ (gitignored unless added).

    .venv/Scripts/python.exe scripts/train_clock_r8_chance_v1.py --arm K2 --fold F2
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
from cbb_sim.models import possession_outcome as PO  # noqa: E402
import train_clock_r8_tempo_v1 as T  # noqa: E402

CKD = ROOT / "data/processed/models/clock"
P4 = ROOT / "data/processed/possessions_v4"
KEY = ["game_id", "period", "poss_index"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def chance_tables(seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per possession: d1, end1_code, n_ch. Continuation rows: (game_id, period, poss_index, grp, d)."""
    firsts, conts = [], []
    for s in seasons:
        c = pd.read_parquet(P4 / f"chances_{s}.parquet",
                            columns=[*KEY, "chance_number", "duration_s", "terminal_event", "fgm_rim", "fgm_jump2", "fgm_3"])
        n = c.groupby(KEY)["chance_number"].max().rename("n_ch")
        f = c[c["chance_number"] == 1].copy()
        f["end1_code"] = r8.end1_code_from_chance(f["terminal_event"].to_numpy(),
                                                  (f["fgm_rim"] + f["fgm_jump2"] + f["fgm_3"]).to_numpy())
        f = f.set_index(KEY)[["duration_s", "end1_code"]].rename(columns={"duration_s": "d1"}).join(n)
        firsts.append(f)
        k = c[c["chance_number"] >= 2][[*KEY, "chance_number", "duration_s"]].copy()
        k["grp"] = np.where(k["chance_number"] >= 3, 1, 0)
        conts.append(k)
    return pd.concat(firsts), pd.concat(conts, ignore_index=True)


def build_d1_design(design: pd.DataFrame, first: pd.DataFrame) -> pd.DataFrame:
    f = first.reset_index()
    f["_p"] = f["period"].astype("int64")
    f["_i"] = f["poss_index"].astype("int64")
    f = f.drop(columns=["period", "poss_index"])
    design = design.assign(_p=design["period"].astype("int64"), _i=design["poss_index"].astype("int64"))
    d = design.merge(f, on=["game_id", "_p", "_i"], how="inner", validate="1:1").drop(columns=["_p", "_i"])
    d = d[d["d1"].notna()].copy()
    d["censored_poss"] = d["censored"].to_numpy(dtype=bool)
    d["duration_s"] = np.clip(d["d1"].to_numpy(dtype="int64"), 0, ck.DURATION_CAP)
    d["censored"] = d["censored_poss"] & (d["n_ch"].to_numpy() == 1)
    d["end1_code"] = d["end1_code"].astype("int64")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["K1", "K2", "K2M"], required=True)
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_dir = CKD / f"r8_{a.arm}" / (a.fold if a.seed == 0 else f"{a.fold}_seed{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    design = T.load_design()
    first, cont = chance_tables(T.ALL_SEASONS)
    n0 = len(design)
    design = build_d1_design(design, first)
    log(f"d1 design {len(design):,} of {n0:,} rows joined; d1-censored {design['censored'].mean():.4%}")
    # continuation rows: regulation, possession not horn-censored, with the possession's date
    pinfo = design[[*KEY, "season", "game_date", "censored_poss"]].copy()
    for c_ in ("period", "poss_index"):
        pinfo[c_] = pinfo[c_].astype("int64")
        cont[c_] = cont[c_].astype("int64")
    cont = cont.merge(pinfo, on=KEY, how="inner")
    cont = cont[(cont["period"] <= 2) & (~cont["censored_poss"])]
    test_season = ck.FOLDS[a.fold]["test"][0]
    tr, te = ck.fold_slices(design, a.fold)
    del design
    cols = [c for c in dict.fromkeys([*c3.s1_fit_columns("empirical_km3"), "score_diff", "days_since_start",
                                      "game_id", "end1_code", "n_ch"]) if c in tr.columns]
    tr_small = tr[cols]
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
        cm = cont_train | ((cont["season"] == test_season).to_numpy() & (cont_dates < cut).to_numpy())
        cpmf = r8.cont_pmf_from(cont.loc[cm, "duration_s"].to_numpy(), cont.loc[cm, "grp"].to_numpy())
        t0 = time.time()
        arm = r8.fit_k_arm(a.arm, fit_rows, cpmf, seed=a.seed)
        stamp = f"{pd.Timestamp(cut).year:04d}-{pd.Timestamp(cut).month:02d}"
        mfile = out_dir / f"r8{a.arm}_S1_{test_season}_{stamp}.pkl"
        with open(mfile, "wb") as f:
            pickle.dump(arm, f)
        mx = tr_dates.max() if not before.any() else max(tr_dates.max(), te_dates[before].max())
        months.append({
            "refit_date": str(pd.Timestamp(cut).date()), "valid_from": str(pd.Timestamp(cut).date()),
            "valid_to": None if nxt is None else str((pd.Timestamp(nxt) - pd.Timedelta(days=1)).date()),
            "n_train": int(len(fit_rows)), "n_train_from_test_season": int(len(prior)),
            "n_cont_rows": int(cm.sum()), "mean_cont": arm.mean_cont().round(3).tolist(),
            "n_scored": int(seg.sum()), "max_train_date": str(pd.Timestamp(mx).date()),
            "season": test_season, "month": stamp,
            "model_file": str(mfile.relative_to(CKD)).replace("\\", "/"),
            "fit_seconds": round(time.time() - t0, 1), **arm.info})
        log(f"{a.arm} {a.fold} refit {stamp}: {len(fit_rows):,} rows, cont {int(cm.sum()):,} "
            f"(mean {arm.mean_cont().round(2).tolist()}), {months[-1]['fit_seconds']}s")
        del fit_rows, prior, arm
    manifest = {"base_arm": f"clock_r8_{a.arm}", "parametrisation": "P3", "tag": f"r8{a.arm}",
                "scheme": "S1", "season": test_season, "fold": a.fold, "round": "8 amendment (section 38)",
                "seed": a.seed, "arm_adopted": False,
                "note": "clock round 8 amendment, outcome-conditioned possession time; NOT adopted, NOT a default",
                "months": months}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
