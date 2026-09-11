#!/usr/bin/env python
"""
train_rotation_v5_nostate.py -- the L31 refit-WITHOUT-the-feature arm for the
rotation round-5 Decision-10 gate.

Pre-registration: `docs/models/rotation/experiments.md` section 12.9. L31:
*"every Decision-10 gate runs both the frozen arm and the refit-without-feature
arm before any magnitude is quoted"*, because a freeze DETECTS a loop and only a
refit SIZES one (clock round 3c: the freeze read 4.47 possessions, the refit
0.42).

The rotation's engine-produced state enters round 5 in two places, and both are
removed here, per window, on the same rows the live arm was fitted on:

  1. the WAVE CELL -- the margin band and the foul state are marginalised out of
     the counts before the shrinkage, which is the maximum-likelihood fit of the
     state-free cell model, not an ablation of a fitted number;
  2. round 4's COMPOSITION HAZARDS -- refitted on the same design rows with
     every margin and foul column DROPPED, and the coefficients scattered back
     into the 45-long vector with zeros in the dropped positions, so the
     adapter's design matrix is untouched and only the model changes.

Writes `rotation_v5_wave_nostate_{YYYYMM}.json` and
`rotation_v5_manifest_nostate.json` under
`data/processed/models/rotation/round5/`; nothing else is written or overwritten.

    .venv/Scripts/python.exe scripts/train_rotation_v5_nostate.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import rotation as R  # noqa: E402
from cbb_sim.models import rotation_v4 as V4  # noqa: E402
from cbb_sim.models import rotation_v5 as V5  # noqa: E402

import train_rotation_v1 as V1  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "models" / "rotation"
R5_DIR = OUT_DIR / "round5"
TRAIN_SEASON, TEST_SEASON = 2024, 2025

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rot5_nostate")

#: every feature of `rotation_v4.SUB_FEATURES` that is a margin term or a foul
#: term. These are the engine-produced state the Decision-10 refit removes.
STATE_FEATURES = {
    "fouls", "foul_out", "fouls_x_is_starter", "fouls_x_late", "team_fouls_frac",
    "mb_6_15", "mb_gt15", "is_starter_x_mb_6_15", "is_starter_x_mb_gt15",
    "is_starter_x_close_x_late", "is_starter_x_blowout_x_late", "share_x_mb_gt15",
    "abs_margin", "is_starter_x_abs_margin",
}


def refit_hazards_without_state(Xo, yo, Xi, yi, seed: int = 0) -> V4.SubHazardFit:
    from sklearn.linear_model import LogisticRegression
    keep = np.array([i for i, f in enumerate(V4.SUB_FEATURES)
                     if f not in STATE_FEATURES], dtype=np.int64)
    out = []
    for X, y in ((Xo, yo), (Xi, yi)):
        m = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", random_state=seed)
        m.fit(X[:, keep], y)
        full = np.zeros(len(V4.SUB_FEATURES), dtype=np.float64)
        full[keep] = m.coef_[0]
        out.append((full.tolist(), float(m.intercept_[0])))
    return V4.SubHazardFit(kind="logistic", out_coef=out[0][0], out_intercept=out[0][1],
                           in_coef=out[1][0], in_intercept=out[1][1],
                           n_out=int(len(yo)), n_in=int(len(yi)),
                           base_out=float(np.mean(yo)), base_in=float(np.mean(yi)),
                           notes={"dropped_features": sorted(STATE_FEATURES),
                                  "kept": int(len(keep))})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave-team-games", type=int, default=6000)
    ap.add_argument("--hazard-games", type=int, default=1500)
    ap.add_argument("--hazard-team-games", type=int, default=800)
    ap.add_argument("--fit-seed", type=int, default=11)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-prior-games", type=int, default=3)
    args = ap.parse_args()
    assert_not_sealed([TRAIN_SEASON, TEST_SEASON], context="rotation round 5 nostate")
    t0 = time.time()
    R5_DIR.mkdir(parents=True, exist_ok=True)

    train24 = V1.load_season(TRAIN_SEASON)
    test = V1.load_season(TEST_SEASON)
    ss_all = pd.concat([V4.load_side_state(TRAIN_SEASON), V4.load_side_state(TEST_SEASON)],
                       ignore_index=True)
    static = R.RotationFit.from_json(OUT_DIR / "rotation_fit_v3.json")

    live = json.loads((R5_DIR / "rotation_v5_manifest.json").read_text(encoding="utf-8"))
    tp25, ev25 = test["tp"], test["fouls"]
    entries = []
    for e in live["artifacts"]:
        m = pd.Timestamp(e["refit_date"])
        tag = m.strftime("%Y%m")
        t1 = time.time()
        if e is live["artifacts"][0]:
            tp_w = train24["tp"]
            ev_w = train24["fouls"]
            fe_w = train24["feats"]
            base = static
        else:
            mask = pd.to_datetime(tp25["game_date"]) < m
            gids = set(tp25.loc[mask, "game_id"].astype("int64"))
            tp_w = pd.concat([train24["tp"], tp25[tp25["game_id"].isin(gids)]],
                             ignore_index=True)
            ev_w = pd.concat([train24["fouls"], ev25[ev25["game_id"].isin(gids)]],
                             ignore_index=True)
            fe_w = pd.concat([train24["feats"],
                              test["feats"][test["feats"]["game_id"].isin(gids)]],
                             ignore_index=True)
            bp = OUT_DIR / f"rotation_fit_v3_S1_{tag}.json"
            base = R.RotationFit.from_json(bp) if bp.exists() else static

        # (1) the wave tables, state marginalised out of the same counts
        counts = V5.build_wave_training(tp_w, sorted(tp_w["game_id"].unique()),
                                        side_state=ss_all, fouls=ev_w,
                                        max_team_games=args.wave_team_games,
                                        seed=args.fit_seed)
        # (2) round 4's hazards, refitted on the same rows without the state
        gl = sorted(tp_w["game_id"].unique())
        rs = np.random.RandomState(args.fit_seed)
        use = list(rs.choice(gl, size=min(args.hazard_games, len(gl)), replace=False))
        Xo, yo, Xi, yi = V4.build_sub_training(
            tp_w, fe_w, base, use, fouls=ev_w, side_state=ss_all,
            max_team_games=args.hazard_team_games, seed=args.fit_seed)
        hz = refit_hazards_without_state(Xo, yo, Xi, yi, seed=args.seed)

        wf = V5.fit_wave(counts, hz, hazard_source=f"refit_without_state_{tag}",
                         collapse_state=True)
        wf.notes.update({"window": tag, "max_train_date": e["max_train_date"],
                         "fit_seed": args.fit_seed, "refit_without_state": True,
                         "hazard_rows": [int(len(yo)), int(len(yi))]})
        p = R5_DIR / f"rotation_v5_wave_nostate_{tag}.json"
        wf.to_json(p)
        entries.append({"refit_date": e["refit_date"], "path": p.name,
                        "max_train_date": e["max_train_date"],
                        "hazards": "refit_without_state",
                        "rho": round(wf.rho, 4), "n_boundaries": wf.n_boundaries})
        log.info("  nostate window %s: %d boundaries, rho %.3f, hazard rows %s/%s "
                 "(%.1f min)", tag, wf.n_boundaries, wf.rho, len(yo), len(yi),
                 (time.time() - t1) / 60)

    (R5_DIR / "rotation_v5_manifest_nostate.json").write_text(json.dumps(
        {"model": "rotation", "scheme": "S1", "fold": "F1", "season": TEST_SEASON,
         "arm": "round5_wave_refit_without_state", "artifacts": entries}, indent=2),
        encoding="utf-8")
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60,
             R5_DIR / "rotation_v5_manifest_nostate.json")


if __name__ == "__main__":
    main()
