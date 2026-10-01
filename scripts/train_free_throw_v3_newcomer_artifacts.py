#!/usr/bin/env python
"""
train_free_throw_v3_newcomer_artifacts.py -- serving artifacts for a section-13 FT arm on the SERVED calendar
(lane I, 2026-10-01; free_throw experiments.md 13.2 "A winner is served behind a default-off mode").

Fits the arm on the S1_conf_aligned refit dates of the served manifest (same dates, same pool rule: all train seasons plus
test-season attempts strictly before the refit date), writes `seg_<k>_<date>.joblib` + `manifest.json` in the served format
to data/processed/models/free_throw/laneI_<arm>/S1_conf_aligned/F2/, and scores F2 with them (log loss next to the served
0.575210). It also runs the TRAIN/SERVE PARITY check: for real 2025 attempts whose shooter is on the engine roster, the
engine_v3 slot values of the arm's extra columns must equal the training join (share equal to 1e-5 reported; PASS >= 99%).

The engine serves the result with NO code change: `scripts/run_engine_overlay_v1.py` with an overrides file
{"adapters.FT_S1_MANIFEST": "<that manifest>"} (the adapter's dated loader reads the feature list from the artifacts).

    CBB_NJOBS=1 .venv/Scripts/python.exe scripts/train_free_throw_v3_newcomer_artifacts.py --arm N2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

NJ = int(os.environ.get("CBB_NJOBS", "1"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(NJ)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)
import train_free_throw_v3_newcomer as T  # noqa: E402
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402

SERVED = Path("data/processed/models/free_throw/s1_confirm/S1_conf_aligned/F2/manifest.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parity-input-dir", default="data/processed/models/engine_v3")
    a = ap.parse_args()
    feats = T.ARMS[a.arm]
    out = Path(f"data/processed/models/free_throw/laneI_{a.arm}/S1_conf_aligned/F2")
    if out.exists():
        raise SystemExit(f"{out} exists: never overwrite")
    t0 = time.time()
    ES.load_universe()
    att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
    att = att[att["season"].isin(T.SEASONS)]
    d = FT.build_ft_design(att)
    d = T.add_shooting(d)
    if any(f in T.BIO for f in feats):
        d = T.add_bio(d)
    tr, te = FT.fold_slices(d, "F2")
    te_dates = pd.to_datetime(te["game_date"])
    sv = json.loads(SERVED.read_text(encoding="utf-8"))
    cols = [*feats, "y"]
    p = np.zeros(len(te))
    out.mkdir(parents=True)
    arts = []
    refits = [pd.Timestamp(e["refit_date"]) for e in sv["artifacts"]]
    for k, cut in enumerate(refits):
        nxt = refits[k + 1] if k + 1 < len(refits) else None
        seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if k == 0:
            seg = seg | (te_dates < cut).to_numpy()
        before = (te_dates < cut).to_numpy()
        rows = tr[cols] if not before.any() else pd.concat([tr[cols], te.loc[before, cols]], ignore_index=True)
        m = FT.LgbmArm(seed=a.seed).fit(FT.design_matrix(rows, tuple(feats)), rows["y"].to_numpy())
        if seg.any():
            p[seg] = m.predict_proba(FT.design_matrix(te.loc[seg], tuple(feats)))[:, FT.CLASS_INDEX["MAKE"]]
        fn = f"seg_{k:02d}_{cut.date()}.joblib"
        joblib.dump({"arm": "lgbm", "features": list(feats), "model": m, "scheme": "S1_conf_aligned", "fold": "F2",
                     "refit_date": str(cut.date()), "max_train_date": sv["artifacts"][k]["max_train_date"],
                     "note": f"lane I free_throw experiments.md section 13 arm {a.arm} (NOT served by default)"}, out / fn)
        arts.append({"refit_date": str(cut.date()), "path": fn, "max_train_date": sv["artifacts"][k]["max_train_date"],
                     "n_train": int(len(rows))})
        print(f"[{time.time() - t0:5.0f}s] seg {k} {cut.date()} n_train {len(rows)}", flush=True)
    (out / "manifest.json").write_text(json.dumps({**{k: v for k, v in sv.items() if k != "artifacts"},
                                                   "key": f"FT2_lgbm_{a.arm}", "artifacts": arts}, indent=1),
                                       encoding="utf-8")
    y = te["y"].to_numpy().astype(float)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    rep = {"arm": a.arm, "features": feats, "F2_log_loss_conf_aligned": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
           "served_N0_conf_aligned": 0.57521}
    # train/serve parity of the extra columns against the engine's slot values
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(a.parity_input_dir, "F2_2025")
    games = inp.games.reset_index(drop=True)
    pos = {int(g): i for i, g in enumerate(games["game_id"].to_numpy())}
    x = te[te["game_id"].isin(pos)].reset_index(drop=True)
    g = x["game_id"].map(pos).to_numpy().astype(np.int64)
    side = np.where(x["team_id"].to_numpy() == games["home_team_id"].to_numpy()[g], 0, 1)
    hit = inp.roster_cbbd[g, side] == x["shooter_id"].to_numpy().astype(np.int64)[:, None]
    f = hit.any(axis=1)
    sl = hit.argmax(axis=1)
    par = {}
    for c in feats:
        if c in FT.FT_FEATURES or c not in inp.slot_names:
            continue
        sv_ = inp.slot_static[g[f], side[f], sl[f], inp.slot_names[c]].astype(np.float64)
        tr_ = x.loc[f, c].to_numpy(np.float64)
        eq = np.isclose(sv_, tr_, atol=1e-5, rtol=0, equal_nan=True)
        ok = np.isfinite(sv_) & np.isfinite(tr_)
        par[c] = {"share_equal_1e5": float(np.mean(eq)), "nan_served": float(np.mean(~np.isfinite(sv_))),
                  "nan_train": float(np.mean(~np.isfinite(tr_))),
                  "mean_abs_diff_finite": float(np.mean(np.abs(sv_[ok] - tr_[ok]))) if ok.any() else None}
    rep["parity"] = par
    rep["parity_PASS"] = bool(par) and all(v["share_equal_1e5"] >= 0.99 for v in par.values())
    (out / "build_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
