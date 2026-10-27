"""train_fg_make_v6_loc_v1.py -- shot-location round 1, Read A (docs/models/shot_location/experiments.md 1.3): the
served fg_make B1 structure (lane-A features, S1 monthly schedule, frozen ladder params) with location columns added.

Versioned wrapper of `train_fg_make_v4_par_g9_v1.py` (which wraps `train_fg_make_v4_par_v1.py`), built exactly as
`train_fg_make_v5_pae_v1.py` was; NOTHING existing is edited. What this wrapper changes, and nothing else:

  --arm R | A1 | A2   R  = the served B1 spec, unchanged (control; run under --seed 1 it is R_seed, the floor)
                      A1 = R + band one-hots b_short2, b_long2, b_corner3, b_wing3, b_top3 (section 1.2 bands; within a
                           class the other class's columns are constant 0 and never split on)
                      A2 = R + cell_p (the attempt's prior-season da15 cell make rate, fg_make 31.2 recipe)
  --target make | and_one
                      make    : population = LOCATED FGA_jump2 / FGA_3 attempts (rim excluded by ruling 11)
                      and_one : population = MADE located FGA_jump2 / FGA_3 attempts, target = `and_one` (a shooting
                                foul on a made shot; the foul channel is PARTIAL by construction, section 1.3). Used by
                                `train_foul_draw_v1.py`. Same B1 features and ladder params.
  --loc-cols PATH     the row-aligned table written by `build_shot_location_bands_v1.py --part build`; attached BY
                      POSITION after asserting game_id / shooter_id / shot_class / made are equal row for row.

Population filter: applied INSIDE the task builder only (after `fit_m`, which therefore sees the same full design as
the served trainer), by wrapping the g9 module's `build_tasks`; the rim class is not fitted (FG.SHOT_CLASSES is
restricted to the two clean classes in this process only). Unlocated attempts are never imputed: they are not in the
population.

Extra output (no effect on any fit): `cf_<fold>_s<seed>.parquet`, the out-of-sample prediction of each S1 cut model
on ALL test-season rows of the class (located or not; for and_one, all made rows): for R the plain prediction, for A1
the prediction with the band one-hots set to each band of the class in turn. Read C marginalises these over the Read B
band probabilities. And `rows_<fold>_s<seed>.parquet`: the fg design row index of every scored test row.

    .venv/Scripts/python.exe scripts/train_fg_make_v6_loc_v1.py --arm A1 --target make --fold F2 --seed 0 \
        --arms B1 --no-floor --no-leak --n-jobs 6 --out-dir data/processed/models/shot_location/round1/make/A1_s0_F2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

BANDS = {"FGA_jump2": ("short2", "long2"), "FGA_3": ("corner3", "wing3", "top3")}
BAND_COLS = ["b_short2", "b_long2", "b_corner3", "b_wing3", "b_top3"]
ARM_FEATURES = {"R": [], "A1": BAND_COLS, "A2": ["cell_p"]}
LOC_COLS = ROOT / "data/processed/models/shot_location/round1/design_loc_cols_v1.parquet"

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--arm", choices=list(ARM_FEATURES), required=True)
_ap.add_argument("--target", choices=["make", "and_one"], default="make")
_ap.add_argument("--loc-cols", type=Path, default=LOC_COLS)
OWN, REST = _ap.parse_known_args()
if any(x.startswith("--extra-features") for x in REST):
    raise SystemExit("--extra-features is set by --arm in this wrapper")
_extra = ARM_FEATURES[OWN.arm]
sys.argv = [sys.argv[0], *REST, *([f"--extra-features={','.join(_extra)}"] if _extra else [])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train_fg_make_v4_par_g9_v1 as G9  # noqa: E402  (parses sys.argv at import, by design)
import train_fg_make_v4_par_v1 as P  # noqa: E402
import train_fg_make_v4_shooter_block as R4M  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402

CLEAN = ("FGA_jump2", "FGA_3")
FG.SHOT_CLASSES = CLEAN               # this process only: the rim class is excluded by ruling 11
_DES: dict = {}
_orig_load_design = P.load_design


def _load_design_loc(a, t0):
    design, rep = _orig_load_design(a, t0)
    cols = pd.read_parquet(OWN.loc_cols)
    if len(cols) != len(design):
        raise SystemExit(f"loc cols {len(cols)} rows vs design {len(design)}")
    for k in ("game_id", "shooter_id", "shot_class", "made"):
        if not (cols[k].to_numpy() == design[k].to_numpy()).all():
            raise SystemExit(f"loc cols misaligned with the design on {k}")
    for c in [*BAND_COLS, "cell_p"]:
        design[c] = cols[c].to_numpy(dtype="float32")
    design["loc_located"] = cols["located"].to_numpy().astype(bool)
    keep = design["loc_located"].to_numpy() & design["shot_class"].isin(CLEAN).to_numpy()
    if OWN.target == "and_one":
        keep &= design["made"].to_numpy().astype(bool)
    design["loc_keep"] = keep
    for c in _extra:
        v = design.loc[keep, c].to_numpy(dtype="float64")
        if not np.isfinite(v).all():
            raise SystemExit(f"feature {c} has non-finite values on the population")
    _DES["design"] = design
    print(f"loc wrapper: arm {OWN.arm}, target {OWN.target}, extra {_extra}, population {int(keep.sum()):,} rows",
          flush=True)
    return design, rep


P.load_design = _load_design_loc
_orig_build_tasks = G9.build_tasks


def _build_tasks_loc(design, arm, params, seed, cuts_hook=None):
    d = design[design["loc_keep"].to_numpy()].copy()
    if OWN.target == "and_one":
        d["y"] = d["and_one"].astype("int64").to_numpy()
    return _orig_build_tasks(d, arm, params, seed, cuts_hook)


G9.build_tasks = _build_tasks_loc


def counterfactual(out_dir: Path, fold: str, seed: int) -> None:
    design = _DES["design"]
    feats = R4M.COMMON + R4M.SHOOTER_BLOCK["B1"]
    _, te_all = FG.fold_slices(design, fold)
    rows_out, cf_out = [], []
    for cls in CLEAN:
        plan = G9._CTX["plans"][cls]
        te_pop = plan["te"]
        rows_out.append(pd.DataFrame({"fg_row": te_pop.index.to_numpy(), "shot_class": cls}))
        allr = FG.class_slice(te_all, cls)
        if OWN.target == "and_one":
            allr = allr[allr["made"].to_numpy().astype(bool)]
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
            if OWN.arm == "A1":
                for b in BANDS[cls]:
                    z = sub.copy()
                    for c in BAND_COLS:
                        z[c] = np.float32(1.0 if c == f"b_{b}" else 0.0)
                    res.setdefault(f"p_{b}", np.full(len(allr), np.nan))[m] = \
                        mdl.predict_proba(FG.design_matrix(z, feats))[:, FG.CLASS_INDEX["MAKE"]]
            else:
                res.setdefault("p", np.full(len(allr), np.nan))[m] = \
                    mdl.predict_proba(FG.design_matrix(sub, feats))[:, FG.CLASS_INDEX["MAKE"]]
        for k, v in res.items():
            fr[k] = v
        cf_out.append(fr)
    pd.concat(rows_out, ignore_index=True).to_parquet(out_dir / f"rows_{fold}_s{seed}.parquet", index=False)
    if OWN.arm in ("R", "A1"):
        pd.concat(cf_out, ignore_index=True).to_parquet(out_dir / f"cf_{fold}_s{seed}.parquet", index=False)
    print("loc wrapper: rows + counterfactual predictions written", flush=True)


def main() -> int:
    rc = G9.main()
    if rc:
        return rc
    pa = argparse.ArgumentParser(add_help=False)
    pa.add_argument("--out-dir", type=Path, required=True)
    pa.add_argument("--seed", type=int, default=0)
    k, _ = pa.parse_known_args(REST)
    counterfactual(k.out_dir, G9.G9.fold, k.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
