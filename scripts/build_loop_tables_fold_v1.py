"""build_loop_tables_fold_v1.py -- fold tables for three of the adopted loop-level members (lane D, 2026-10-01;
fold-1 confirmation, docs/models/engine/experiments.md section 2). Nothing served is read-modified or overwritten.

  foul R8b base   lut_acc_A2_<F>.npz, lut_trip_T2c_<F>.npz  (round 7: `train_foul_joint_v1` design / fit / offset
                  functions and `build_foul_joint_lut_v1` grid / index, imported and called unedited; only the
                  fold's train seasons differ)
  foul R9 AO3     lut_ao_AO3_<F>.npz                       (round 9: `train_foul_r9_v1.ao_design / team_prior_table /
                  add_team_prior / fit_team_terms` and the AO1 Laplace cell table, as in `run_ao`; the team-prior
                  table is the served one, season s holds season s-1 rates)
  shared_shooting params_G3_<F>.json                       (G3 Sigma READ from the bake-off's own fold fit,
                  `results/shared_shooting/bakeoff_v1.json` folds.<F>.arms.G3.fit.Sigma; the served params_v1.json
                  is folds.F2 of the same file, checked here)
chance_time KD needs nothing: `chance_time/F1/lut_v3.npz` exists (built by `build_chance_time_lut_v3.py` per fold).

Identity: `--fold F2` must reproduce the served A2 / T2c / AO3 tables (A2 up to LightGBM thread count; reported).

    .venv/Scripts/python.exe scripts/build_loop_tables_fold_v1.py --fold F1 --out data/processed/models/fold1_v1/loop
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import build_foul_joint_lut_v1 as BL7  # noqa: E402
import train_foul_joint_v1 as FJ  # noqa: E402

R7D = ROOT / "data/processed/models/possession_outcome/round7"
R9D = ROOT / "data/processed/models/possession_outcome/round9"
SS = ROOT / "data/processed/models/shared_shooting/params_v1.json"
BAKE = ROOT / "results/shared_shooting/bakeoff_v1.json"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def foul_r7(fold: str, out: Path, rep: dict) -> None:
    sp = FJ.FOLDS[fold]
    d = FJ.possession_design()
    tr = d[d["season"].isin(sp["train"]) & (d["in_fit_window"] == 1)]
    log(f"A2 fit on {len(tr):,} rows (train {sp['train']})")
    m = FJ.fit_gbm(tr[FJ.STATE_T].to_numpy("float32"), tr["y_nt"].to_numpy(), 0)
    shape = (3, 5, 8, BL7.MAXF + 1, BL7.MAXF + 1, 3)
    lut = m.predict_proba(BL7.grid(False)[FJ.STATE_T].to_numpy("float32"))[:, 1].reshape(shape)
    np.savez_compressed(out / f"lut_acc_A2_{fold}.npz", lut=lut, clock_cuts=BL7.CLOCK_CUTS,
                        margin_cuts=BL7.MARGIN_CUTS, max_fouls=BL7.MAXF)
    rep["A2"] = {"n_train": int(len(tr)), "mean_lut": float(lut.mean())}
    del d, tr
    ch, base = FJ.chance_design()
    tr = ch[ch["season"].isin(sp["train"]) & (ch["in_fit_window"] == 1)]
    Xtr_l = tr[base].to_numpy("float64")
    Xtr_t = tr[base].assign(in_bonus=tr["inb_t"])[base].to_numpy("float64")
    arrs = {}
    for y, nm in (("y_bonus", "delta_bonus"), ("y_shoot", "delta_shoot")):
        (p_tr_l, p_tr_t), _ = FJ.t0_task(Xtr_l, tr[y].to_numpy(), [Xtr_l, Xtr_t])
        dl = FJ.fit_offsets(FJ.t_cells(tr, "t", "T2c"), tr[y].to_numpy(), FJ.logit(p_tr_t))
        a = np.zeros((2, BL7.MAXF + 1, 3))
        for k, v in dl.items():
            h, rem = divmod(int(k), 1000)
            dc, db = divmod(rem, 10)
            a[h, dc, db] = v
        arrs[nm] = a
    np.savez_compressed(out / f"lut_trip_T2c_{fold}.npz", max_fouls=BL7.MAXF, **arrs)
    rep["T2c"] = {"n_train": int(len(tr)), **{k: [float(v.min()), float(v.max())] for k, v in arrs.items()}}


def foul_r9(fold: str, out: Path, rep: dict) -> None:
    import train_foul_r9_v1 as R9
    sp = R9.FOLDS[fold]
    d = R9.ao_design()
    tp = pd.read_parquet(R9D / "ao_team_prior_v1.parquet")          # served table (season s holds s-1 rates)
    d = R9.add_team_prior(d, tp)
    tr = d[d["season"].isin(sp["train"])]
    cls_rate = tr.groupby("cls")["y_ao"].mean()
    ktr = R9.ao_cells_key(tr)
    s_ = pd.DataFrame({"k": ktr, "y": tr["y_ao"].to_numpy(), "c": tr["cls"].to_numpy()})
    agg = s_.groupby("k").agg(sm=("y", "sum"), n=("y", "size"), c=("c", "first"))
    pr = agg["c"].map(cls_rate)
    tab = (agg["sm"] + 200 * pr) / (agg["n"] + 200)
    ao1_tr = pd.Series(ktr).map(tab).to_numpy(dtype=float)
    tc = ["ao_off_prior_c", "ao_def_prior_c"]
    b = R9.fit_team_terms(ao1_tr, tr[tc].to_numpy(float), tr["y_ao"].to_numpy(float))
    tabd, prior = tab.to_dict(), cls_rate.to_dict()
    lut = np.zeros((3, 3, 5))
    for c in range(3):
        for pi in range(3):
            for ci in range(5):
                lut[c, pi, ci] = tabd.get(c * 100 + pi * 10 + ci, prior[c])
    # the engine reads team_table relative to foul_r9.LUT_DIR: copy the served table next to the LUT
    shutil.copyfile(R9D / "ao_team_prior_v1.parquet", out / "ao_team_prior_v1.parquet")
    np.savez_compressed(out / f"lut_ao_AO3_{fold}.npz", lut=lut, clock_cuts=np.array([120, 300, 600, 900], float),
                        team_b=np.array(b.tolist()), team_table=np.array("ao_team_prior_v1.parquet"))
    rep["AO3"] = {"n_train": int(len(tr)), "team_b": b.tolist(), "class_prior": {int(k): float(v) for k, v in prior.items()}}


def g3(fold: str, out: Path, rep: dict) -> None:
    r = json.loads(BAKE.read_text(encoding="utf-8"))
    served = json.loads(SS.read_text(encoding="utf-8"))
    ok = np.array_equal(np.array(r["folds"]["F2"]["arms"]["G3"]["fit"]["Sigma"]), np.array(served["G3_Sigma"]))
    if not ok:
        raise SystemExit("served params_v1.json G3_Sigma is not the bake-off's F2 fit: provenance broken")
    p = dict(served)
    p["G3_Sigma"] = r["folds"][fold]["arms"]["G3"]["fit"]["Sigma"]
    p["source"] = (f"G3 Sigma of fold {fold} from {BAKE.relative_to(ROOT).as_posix()} (folds.{fold}.arms.G3.fit; train "
                   "seasons of that fold); every other key copied from params_v1.json and NOT used by G3")
    p["created_at"] = pd.Timestamp.now("UTC").isoformat()
    (out / f"params_G3_{fold}.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    rep["G3"] = {"served_is_bakeoff_F2": ok, "Sigma": p["G3_Sigma"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", default="g3,r9,r7")
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"fold": a.fold}
    t0 = time.time()
    parts = a.parts.split(",")
    if "g3" in parts:
        g3(a.fold, out, rep)
    if "r9" in parts:
        foul_r9(a.fold, out, rep)
    if "r7" in parts:
        foul_r7(a.fold, out, rep)
    if a.fold == "F2":                       # identity vs the served tables
        idn = {}
        for stem, d in (("lut_acc_A2_F2", R7D), ("lut_trip_T2c_F2", R7D), ("lut_ao_AO3_F2", R9D)):
            if (out / f"{stem}.npz").exists():
                x, y = np.load(out / f"{stem}.npz"), np.load(d / f"{stem}.npz")
                idn[stem] = {k: float(np.abs(x[k].astype(float) - y[k].astype(float)).max())
                             for k in x.files if x[k].dtype.kind in "fi" and k in y.files}
        rep["identity_max_abs_vs_served"] = idn
    rep["seconds"] = round(time.time() - t0, 1)
    (out / f"loop_tables_{a.fold}_{'_'.join(parts)}.json").write_text(json.dumps(rep, indent=1, default=float),
                                                                     encoding="utf-8")
    log(json.dumps(rep, default=float)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
