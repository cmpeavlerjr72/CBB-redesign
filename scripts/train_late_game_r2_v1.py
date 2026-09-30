"""train_late_game_r2_v1.py -- late-game ROUND 2: refit and persist the served window objects.

Pre-registration: `docs/models/late_game/experiments.md` section 4 (committed at
1c7aa68 BEFORE this file existed). Round 1 persisted predictions, not fitted
objects, so the four cells round 2 serves are refit here spec-identically and
VERIFIED against round 1's saved test-row predictions before anything is
written (section 4.2 item 1):

    C2_clk   window KM cell law, P3R dims, no floor      == round1_clock/pmf/C2_clk_F2.npy  (exact)
    D_clk    window KM cell law, LGD dims, no floor      == round1_clock/pmf/D_clk_F2.npy   (exact)
    ev_BL3   full-scope lgbm, first chances, L3_gates    ~= round1/pred/c027.npy (max|d| <= 1e-4)
    ev_L0S0  full-scope lgbm, first chances, L0_reference ~= round1/pred/c024.npy (max|d| <= 1e-4)

All fold 2, seed 0, static S0 (train seasons 2022-2024). Computes no metric.

Output: data/processed/models/late_game/round2/{clk_C2,clk_D,ev_BL3,ev_L0S0}.pkl + manifest.json

Usage: train_late_game_r2_v1.py [--only clk_C2,clk_D,ev_BL3,ev_L0S0]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Round 1 ran with 3 OpenMP threads; the same count keeps LightGBM's histogram
# summation order as close as possible. Lane cap is 4 cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "3"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from cbb_sim.data.seal import assert_not_sealed                      # noqa: E402
from cbb_sim.models import clock as ck                               # noqa: E402
from cbb_sim.models import clock_v3 as c3                            # noqa: E402
from cbb_sim.models import late_game as LGM                          # noqa: E402
from cbb_sim.models import possession_outcome as PO                  # noqa: E402

import importlib.util as _ilu                                        # noqa: E402
_spec = _ilu.spec_from_file_location(
    "build_late_game_design_v1", ROOT / "scripts" / "build_late_game_design_v1.py")
LGD = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(LGD)

DESIGN_V2 = ROOT / "data/processed/models/clock/design_v2.parquet"
DESIGN_EV = ROOT / "data/processed/models/late_game/design_v1.parquet"
R1 = ROOT / "data/processed/models/late_game"
OUT = R1 / "round2"
ALL_SEASONS = [2022, 2023, 2024, 2025]
FOLD = "F2"
LGBM_TOL = 1e-4


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def clock_cells(want: set[str], manifest: list) -> None:
    log(f"loading {DESIGN_V2.name}")
    d = pd.read_parquet(DESIGN_V2)
    d, _ = c3.attach_horn_censoring(d, ALL_SEASONS)
    c3.set_flag_inplace(d, "horn")
    d = c3.add_p_state(d)
    per2 = d["period"].to_numpy() == 2
    sec = d["seconds_remaining"].to_numpy()
    asd = np.abs(d["score_diff"].to_numpy())
    d["in_window"] = per2 & (sec <= LGM.GATE_SEC) & (asd <= LGM.GATE_MARGIN)
    tr_all, te_all = ck.fold_slices(d, FOLD)
    order = ["game_id", "period", "poss_index"]
    te = te_all[te_all["in_window"].to_numpy()].sort_values(order, kind="stable")
    tr = tr_all[tr_all["in_window"].to_numpy()]
    for name, fs in (("clk_C2", "P3R_dummy"), ("clk_D", "LGD_dummy")):
        if name not in want:
            continue
        cell = name.split("_", 1)[1] + "_clk"
        arm = LGM.fit_cell_arm(tr, fs, 0, f"{cell}|{fs}|floor0")
        pmf = arm.pmf(te.copy()).astype("float32")
        ref = np.load(R1 / "round1_clock" / "pmf" / f"{cell}_{FOLD}.npy")
        same = pmf.shape == ref.shape and bool(np.array_equal(pmf, ref))
        maxd = float(np.abs(pmf - ref).max()) if pmf.shape == ref.shape else float("nan")
        log(f"  {name}: {len(tr):,} train rows, test {len(te):,}; round-1 PMF exact={same} max|d|={maxd:.3g}")
        if not same:
            raise SystemExit(f"{name} does NOT reproduce round 1 exactly; not served (section 4.2.1)")
        p = OUT / f"{name}.pkl"
        with open(p, "wb") as f:
            pickle.dump(arm, f)
        manifest.append({"name": name, "kind": "clock_window_cell_law", "round1_cell": cell,
                         "feature_set": fs, "dims": list(arm.dims), "sr_floor_bucket": 0,
                         "fold": FOLD, "scheme": "S0", "n_train": int(len(tr)),
                         "max_train_date": str(pd.Timestamp(tr["game_date"].max()).date()),
                         "verified_vs": f"round1_clock/pmf/{cell}_{FOLD}.npy",
                         "verify_exact": same, "verify_max_abs_diff": maxd, "path": str(p)})


def event_cells(want: set[str], manifest: list) -> None:
    need = {"season", "game_id", "game_date", "period", "poss_index", "chance_number",
            "population", "y", "is_regulation", "in_window"}
    for b in ("L0_reference", "L3_gates"):
        need |= set(LGD.bundle(b, "first"))
    log(f"loading {DESIGN_EV.name} ({len(need)} columns)")
    d = pd.read_parquet(DESIGN_EV, columns=sorted(need))
    d["game_date"] = pd.to_datetime(d["game_date"])
    d = d[d["is_regulation"].to_numpy()].reset_index(drop=True)
    spec = PO.FOLDS[FOLD]
    tr_all = d[d["season"].isin(spec["train"])]
    assert_not_sealed(tr_all, context=f"{FOLD} train slice")
    tr = tr_all[tr_all["population"] == "first"]
    te = pd.read_parquet(R1 / "round1" / f"window_test_{FOLD}.parquet")
    first = (te["population"] == "first").to_numpy()
    for name, bundle, cid in (("ev_BL3", "L3_gates", "c027"), ("ev_L0S0", "L0_reference", "c024")):
        if name not in want:
            continue
        feats = LGD.bundle(bundle, "first")
        t0 = time.time()
        model = PO.fit_arm("lgbm", tr, feats, seed=0)
        p = model.predict_proba(np.ascontiguousarray(te.loc[first, feats].to_numpy(dtype="float32")))
        ref = np.load(R1 / "round1" / "pred" / f"{cid}.npy")[first]
        maxd = float(np.abs(p.astype("float32") - ref).max())
        log(f"  {name}: {len(tr):,} train rows, {time.time() - t0:.0f}s; vs round-1 {cid} "
            f"max|d|={maxd:.3g} on {int(first.sum()):,} rows")
        if not maxd <= LGBM_TOL:
            raise SystemExit(f"{name} does NOT reproduce round 1 within {LGBM_TOL}; not served")
        try:
            model.clf_.set_params(n_jobs=1)
        except Exception:                                            # noqa: BLE001
            pass
        pth = OUT / f"{name}.pkl"
        with open(pth, "wb") as f:
            pickle.dump({"model": model, "features": feats, "bundle": bundle}, f)
        manifest.append({"name": name, "kind": "event_first_chance_lgbm", "round1_cell": cid,
                         "bundle": bundle, "features": feats, "fold": FOLD, "scheme": "S0",
                         "seed": 0, "n_train": int(len(tr)),
                         "max_train_date": str(pd.Timestamp(tr["game_date"].max()).date()),
                         "verified_vs": f"round1/pred/{cid}.npy",
                         "verify_max_abs_diff": maxd, "path": str(pth)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="clk_C2,clk_D,ev_BL3,ev_L0S0")
    a = ap.parse_args()
    want = {x.strip() for x in a.only.split(",") if x.strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else []
    manifest = [m for m in manifest if m["name"] not in want]
    if want & {"clk_C2", "clk_D"}:
        clock_cells(want, manifest)
    if want & {"ev_BL3", "ev_L0S0"}:
        event_cells(want, manifest)
    mpath.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    log(f"wrote {mpath} ({len(manifest)} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
