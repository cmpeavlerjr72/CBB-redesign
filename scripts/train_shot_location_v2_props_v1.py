"""train_shot_location_v2_props_v1.py -- shot-location round 2, block P arms (docs/models/shot_location/experiments.md
section 3.2; PRE-REGISTERED at commit 5f6d6b0 BEFORE any fit).

One trainer, two paths, selected by --arm:

  P-A (make ceiling; wrapper of the served fg_make B1 trainer exactly as `train_fg_make_v6_loc_v1.py` wraps it; it
       adds columns and changes nothing else; population = LOCATED FGA_jump2 / FGA_3 attempts, rim excluded)
      --arm R    the served B1 spec refit on the located population (run with --seed 1 it is R_seed, the floor)
      --arm P0   R + band one-hots (round-1 A1, re-run)
      --arm P1   P0 + sh_band_make_c (the shooter's centred shrunk make deviation WITHIN the attempt's own band)
      Extra output: `cf_<fold>_s<seed>.parquet`, the out-of-sample make probability on ALL test-season rows of the
      class (located or not): R = plain p; P0 = p_<band> with the band one-hots set to each band in turn; P1 = the
      same AND sh_band_make_c set to that shooter's value for that band (`smk_<band>`).

  P-B (band predictability; light, deterministic GLM; the round-1 `train_shot_location_v1.py` path)
      --arm B2   round-1 carried arm (re-fit; asserted equal to the round-1 predictions)
      --arm BP   B2 + sh_band_c (the shooter's centred shrunk band log-odds within class: short2 for jumpers;
                 corner3, wing3, top3 for threes)
      Predictions are written for ALL test-season attempts of the class (located or not).

Row-aligned inputs: round-1 `design_loc_cols_v1.parquet` (bands, located) and round-2
`design_props_cols_v1.parquet` (from `build_shooter_band_v1.py`); both are asserted equal to the fg design on
game_id / shooter_id / shot_class / made before use.

    .venv/Scripts/python.exe scripts/train_shot_location_v2_props_v1.py --arm P1 --fold F2 --seed 0 --arms B1 \
        --no-floor --no-leak --n-jobs 6 --out-dir data/processed/models/shot_location/round2/make/P1_s0_F2
    .venv/Scripts/python.exe scripts/train_shot_location_v2_props_v1.py --arm BP --fold F2 --n-jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

BANDS = {"FGA_jump2": ("short2", "long2"), "FGA_3": ("corner3", "wing3", "top3")}
PROF = {"FGA_jump2": ("short2",), "FGA_3": ("corner3", "wing3", "top3")}
BAND_COLS = ["b_short2", "b_long2", "b_corner3", "b_wing3", "b_top3"]
ARM_FEATURES = {"R": [], "P0": BAND_COLS, "P1": BAND_COLS + ["sh_band_make_c"]}
LIGHT = ("B2", "BP")
R1 = ROOT / "data/processed/models/shot_location/round1"
R2 = ROOT / "data/processed/models/shot_location/round2"
LOC1 = R1 / "design_loc_cols_v1.parquet"
LOC2 = R2 / "design_props_cols_v1.parquet"

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--arm", choices=[*ARM_FEATURES, *LIGHT], required=True)
OWN, REST = _ap.parse_known_args()
HEAVY = OWN.arm in ARM_FEATURES

if HEAVY:
    if any(x.startswith("--extra-features") for x in REST):
        raise SystemExit("--extra-features is set by --arm in this wrapper")
    _extra = ARM_FEATURES[OWN.arm]
    sys.argv = [sys.argv[0], *REST, *([f"--extra-features={','.join(_extra)}"] if _extra else [])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if HEAVY:
    import train_fg_make_v4_par_g9_v1 as G9  # noqa: E402  (parses sys.argv at import, by design)
    import train_fg_make_v4_par_v1 as P  # noqa: E402
    import train_fg_make_v4_shooter_block as R4M  # noqa: E402
    from cbb_sim.models import fg_make as FG  # noqa: E402

    CLEAN = ("FGA_jump2", "FGA_3")
    FG.SHOT_CLASSES = CLEAN
    _DES: dict = {}
    _orig_load_design = P.load_design

    def _load_design_loc(a, t0):
        design, rep = _orig_load_design(a, t0)
        c1 = pd.read_parquet(LOC1)
        c2 = pd.read_parquet(LOC2)
        if len(c1) != len(design) or len(c2) != len(design):
            raise SystemExit("loc cols misaligned in length")
        for cols in (c1, c2):
            for k in ("game_id", "shooter_id", "shot_class", "made"):
                if not (cols[k].to_numpy() == design[k].to_numpy()).all():
                    raise SystemExit(f"loc cols misaligned with the design on {k}")
        for c in BAND_COLS:
            design[c] = c1[c].to_numpy(dtype="float32")
        for b in ("short2", "long2", "corner3", "wing3", "top3"):
            design[f"smk_{b}"] = c2[f"smk_{b}"].to_numpy(dtype="float32")
        design["sh_band_make_c"] = c2["sh_band_make_c"].to_numpy(dtype="float32")
        design["loc_located"] = c1["located"].to_numpy().astype(bool)
        keep = design["loc_located"].to_numpy() & design["shot_class"].isin(CLEAN).to_numpy()
        design["loc_keep"] = keep
        for c in _extra:
            v = design.loc[keep, c].to_numpy(dtype="float64")
            if not np.isfinite(v).all():
                raise SystemExit(f"feature {c} has non-finite values on the population")
        _DES["design"] = design
        print(f"props wrapper: arm {OWN.arm}, extra {_extra}, population {int(keep.sum()):,} rows", flush=True)
        return design, rep

    P.load_design = _load_design_loc
    _orig_build_tasks = G9.build_tasks

    def _build_tasks_loc(design, arm, params, seed, cuts_hook=None):
        d = design[design["loc_keep"].to_numpy()].copy()
        return _orig_build_tasks(d, arm, params, seed, cuts_hook)

    G9.build_tasks = _build_tasks_loc

    def counterfactual(out_dir: Path, fold: str, seed: int) -> None:
        design = _DES["design"]
        feats = R4M.COMMON + R4M.SHOOTER_BLOCK["B1"]
        _, te_all = FG.fold_slices(design, fold)
        rows_out, cf_out = [], []
        for cls in CLEAN:
            plan = G9._CTX["plans"][cls]
            rows_out.append(pd.DataFrame({"fg_row": plan["te"].index.to_numpy(), "shot_class": cls}))
            allr = FG.class_slice(te_all, cls)
            dates = pd.to_datetime(allr["game_date"]).to_numpy()
            cuts = [np.datetime64(pd.Timestamp(s["cut"])) for s in plan["segs"]]
            seg_of = np.clip(np.searchsorted(np.array(cuts), dates, side="right") - 1, 0, len(cuts) - 1)
            fr = pd.DataFrame({"fg_row": allr.index.to_numpy(), "shot_class": cls,
                               "located": allr["loc_located"].to_numpy()})
            res = {}
            for i, s in enumerate(plan["segs"]):
                m = seg_of == i
                if not m.any():
                    continue
                mdl = G9._CTX["results"][s["key"]]["model"]
                sub = allr.loc[m]
                if OWN.arm in ("P0", "P1"):
                    for b in BANDS[cls]:
                        z = sub.copy()
                        for c in BAND_COLS:
                            z[c] = np.float32(1.0 if c == f"b_{b}" else 0.0)
                        if OWN.arm == "P1":
                            z["sh_band_make_c"] = z[f"smk_{b}"].to_numpy(dtype="float32")
                        res.setdefault(f"p_{b}", np.full(len(allr), np.nan))[m] = \
                            mdl.predict_proba(FG.design_matrix(z, feats))[:, FG.CLASS_INDEX["MAKE"]]
                else:
                    res.setdefault("p", np.full(len(allr), np.nan))[m] = \
                        mdl.predict_proba(FG.design_matrix(sub, feats))[:, FG.CLASS_INDEX["MAKE"]]
            for k, v in res.items():
                fr[k] = v
            cf_out.append(fr)
        pd.concat(rows_out, ignore_index=True).to_parquet(out_dir / f"rows_{fold}_s{seed}.parquet", index=False)
        pd.concat(cf_out, ignore_index=True).to_parquet(out_dir / f"cf_{fold}_s{seed}.parquet", index=False)
        print("props wrapper: rows + counterfactual predictions written", flush=True)

    def main_heavy() -> int:
        rc = G9.main()
        if rc:
            return rc
        pa = argparse.ArgumentParser(add_help=False)
        pa.add_argument("--out-dir", type=Path, required=True)
        pa.add_argument("--seed", type=int, default=0)
        k, _ = pa.parse_known_args(REST)
        counterfactual(k.out_dir, G9.G9.fold, k.seed)
        return 0


# ============================================================================ P-B (light path)
STATE = ["period", "seconds_remaining", "score_diff", "in_bonus", "is_transition_f", "chance_number"]
SITE = ["site_home", "site_away"]


def main_light() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    a, _ = ap.parse_known_args(REST)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(a.n_jobs)
    import time
    from sklearn.linear_model import LogisticRegression

    from cbb_sim.data.seal import assert_not_sealed
    from cbb_sim.models import fg_make as FG

    t0 = time.time()
    spec = FG.FOLDS[a.fold]
    assert_not_sealed(spec["train"] + spec["test"], context="shot_location round 2 P-B")
    od = R2 / "readB" / f"{OWN.arm}_{a.fold}"
    if od.exists() and any(od.iterdir()):
        raise SystemExit(f"{od} not empty")
    od.mkdir(parents=True, exist_ok=True)
    DESIGN = ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet"
    des = pd.read_parquet(DESIGN, columns=["game_id", "season", "shot_class", "shooter_id", "made", *SITE, *STATE])
    c1 = pd.read_parquet(LOC1)
    c2 = pd.read_parquet(LOC2)
    for cols in (c1, c2):
        for k in ("game_id", "shooter_id", "shot_class", "made"):
            assert (cols[k].to_numpy() == des[k].to_numpy()).all(), k
    for c in c1.columns:
        if c.startswith(("off_", "def_")) or c in ("band", "located"):
            des[c] = c1[c].to_numpy()
    for c in c2.columns:
        if c.startswith("sh_"):
            des[c] = c2[c].to_numpy()
    des["in_bonus"] = des["in_bonus"].astype("float64")
    rep = {"arm": OWN.arm, "fold": a.fold, "created_at": pd.Timestamp.now("UTC").isoformat(), "by_class": {}}
    outs = []
    for cls, bands in BANDS.items():
        C = des[des["shot_class"] == cls]
        tr = C[C["season"].isin(spec["train"]) & (C["band"] != "")].copy()
        te_all = C[C["season"].isin(spec["test"])].copy()
        tr["y_band"] = tr["band"].map({b: i for i, b in enumerate(bands)}).astype("int64")
        off = [f"off_{b}_c" for b in PROF[cls]]
        dfn = [f"def_{b}_c" for b in PROF[cls]]
        shc = [f"sh_{b}_c" for b in PROF[cls]]
        f = off + dfn + SITE + STATE + (shc if OWN.arm == "BP" else [])
        X = tr[f].to_numpy(dtype="float64")
        Xt = te_all[f].to_numpy(dtype="float64")
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-8] = 1.0
        clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs").fit((X - mu) / sd, tr["y_band"].to_numpy())
        p = clf.predict_proba((Xt - mu) / sd)
        o = pd.DataFrame({"fg_row": te_all.index.to_numpy(), "shot_class": cls,
                          "located": (te_all["band"] != "").to_numpy(), "band": te_all["band"].to_numpy()})
        for i, b in enumerate(bands):
            o[f"{OWN.arm}__p_{b}"] = p[:, i]
        rep["by_class"][cls] = {"features": f, "n_train": int(len(tr)), "n_test_all": int(len(te_all)),
                                "coef_std": clf.coef_.tolist(), "n_iter": int(clf.n_iter_[0])}
        outs.append(o)
        print(f"[{time.time() - t0:6.1f}s] {a.fold} {cls} {OWN.arm} fitted", flush=True)
    pr = pd.concat(outs, ignore_index=True)
    pr.to_parquet(od / "preds.parquet", index=False)
    if OWN.arm == "B2":
        ref = pd.read_parquet(R1 / "readB" / a.fold / "preds.parquet")
        ref = ref.set_index("fg_row")
        mine = pr.set_index("fg_row")
        diffs = []
        for c in [c for c in mine.columns if c.startswith("B2__p_")]:
            diffs.append(float(np.nanmax(np.abs(mine[c].to_numpy() - ref.loc[mine.index, c].to_numpy()))))
        rep["max_abs_diff_vs_round1_B2"] = max(diffs)
        print(f"B2 re-fit max abs diff vs round-1 preds: {max(diffs):.3e}", flush=True)
    rep["runtime_s"] = round(time.time() - t0, 1)
    (od / "run_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_heavy() if HEAVY else main_light())
