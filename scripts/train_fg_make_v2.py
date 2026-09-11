#!/usr/bin/env python
"""
train_fg_make_v2.py -- run the pre-registered L3 FIELD-GOAL MAKE **round 2**:
state parametrisation (`docs/models/fg_make/experiments.md` section 13).

    .venv/Scripts/python.exe scripts/train_fg_make_v2.py
    .venv/Scripts/python.exe scripts/train_fg_make_v2.py --no-append

`train_fg_make_v1.py` is NEVER overwritten and is not imported: round 1's
numbers stand as that script produced them. This script executes section 13's
spec, which was written and committed before any of it ran.

WHAT IS FIXED, AND WHERE IT CAME FROM
  * model class      LightGBM per shot class (round 1, sections 9 and 12)
  * parameters       round 1's F1-ONLY frozen ladder winner per class, read
                     from `lgbm_ladder_v2.json`. No search happens here.
  * scheme           static, as round 1. L21's S1 is a separate round.
  * folds            F1 reported, F2 SELECTION, 2026 sealed.
  * floors           max(round-1 block bootstrap SE, five-seed S-A refit SD)
  * arms             the five state parametrisations of `fg_make.R2_ARMS`

The ONLY thing that differs between arms is the state block. The team block and
the shooter block are byte-identical across all five, which is what makes a
log-loss difference attributable to the state parametrisation and nothing else.

Artifacts, none of which touch round 1 or the engine's own directory:
    data/processed/models/fg_make/round2/<arm>/fg_make_<class>_F2.joblib
    data/processed/models/fg_make/round2/grid_results.csv
    data/processed/models/fg_make/round2/run_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402

OUT_DIR = Path("data/processed/models/fg_make")
R2_DIR = OUT_DIR / "round2"
DOC = Path("docs/models/fg_make/experiments.md")
SEASONS = [2022, 2023, 2024, 2025]
SEED_GRID = (0, 1, 2, 3, 4)

#: Round 1's per-class game-block bootstrap SE on `ridge` (experiments.md
#: sections 3.1-3.3). Half of the pre-registered floor; the other half is the
#: five-seed S-A refit SD computed below.
R1_BOOTSTRAP_FLOOR = {"FGA_rim": 0.000539, "FGA_jump2": 0.000578, "FGA_3": 0.000849}

#: Round 1's F2 log loss for the `lgbm` / `C_plus_state` arm. S-A must
#: reproduce these, which is the harness's own validity check.
R1_F2_LGBM_LOGLOSS = {"FGA_rim": 0.641605, "FGA_jump2": 0.642003, "FGA_3": 0.561085}


def build_design(version: str) -> pd.DataFrame:
    uni = ES.load_universe(FG.DEFAULT_UNIVERSE, require_pbp_complete=True)
    ev_path = OUT_DIR / f"events_{version}.parquet"
    if not ev_path.exists():
        raise FileNotFoundError(
            f"{ev_path} missing; run scripts/train_fg_make_v1.py --version {version} first "
            "(this script never rebuilds round 1's event cache)")
    ev = pd.read_parquet(ev_path)
    ev.attrs["rim_override_max_ft"] = ES.rim_override_for_version(version)
    ev.attrs["possessions_version"] = version
    d = FG.build_design(SEASONS, universe=uni, version=version, events=ev)
    d = FG.add_round2_state(d)
    # L22: a feature builder never fills a missing column with a constant, and
    # a trainer asserts before fitting so a failure is attributed to the builder.
    for col in [FG.R2_MARGIN_COL, *FG.R2_INDICATORS]:
        v = d[col].to_numpy()
        assert np.isfinite(v).all(), f"{col} carries non-finite values"
        assert v.std() > 0, f"{col} is constant across the design -- builder defect"
    return d


def flat_row(s: dict, fold: str, shot_class: str, arm: str, fs: str,
             fit_s: float) -> dict:
    gated = {c: v for c, v in s["calibration"].items()
             if v["share_pct"] >= PM.CALIB_MIN_SHARE * 100}
    level = max((abs(v["level_shift_pp"]) for v in gated.values()), default=0.0)
    shape = max((v["max_abs_gap_pp_after_level_shift"] for v in gated.values()), default=0.0)
    row = {
        "shot_class": shot_class, "fold": fold, "arm": arm, "feature_set": fs,
        "n": s["n"], "log_loss": round(s["log_loss"], 6), "brier": round(s["brier"], 6),
        "pred_make_pct": round(s["pred_make_rate"] * 100, 3),
        "actual_make_pct": round(s["actual_make_rate"] * 100, 3),
        "calib": "PASS" if s["calib_pass"] else "FAIL",
        "worst_gap_pp": s["calib_worst_gap_pp"],
        "level_pp": round(level, 3), "shape_pp": round(shape, 3),
        "respons_d8": "PASS" if s.get("resp_pass_decision8") else "FAIL",
        "resp_d8_failed": ",".join(x.split("->")[0]
                                   for x in s.get("resp_failed_drivers_decision8", [])),
        "fit_s": round(fit_s, 1),
    }
    for k, v in s["responsiveness"].items():
        tag = k.split("->")[0]
        row[f"steps_{tag}"] = v["pred_monotone_steps"]
        row[f"slope_{tag}"] = v["slope_ratio"]
    for k, v in s["by_chance"].items():
        row[f"chancegap_{k}"] = v["max_abs_gap_pp"]
    return row


def fit_one(arm: str, tr: pd.DataFrame, te: pd.DataFrame, params: dict,
            seed: int = 0) -> tuple[dict, object, float, np.ndarray]:
    fs = FG.R2_ARM_FEATURE_SET[arm]
    t0 = time.time()
    model = FG.fit_arm("lgbm", tr, fs, seed=seed, params=params)
    p = FG.predict_arm("lgbm", model, te, fs)
    return FG.score(te, p), model, time.time() - t0, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--export", action="store_true", default=True,
                    help="write the per-arm joblibs the engine adapter reads")
    ap.add_argument("--skip-floor", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    R2_DIR.mkdir(parents=True, exist_ok=True)
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="fg_make round 2")

    ladder = json.loads((OUT_DIR / f"lgbm_ladder_{a.version}.json").read_text(encoding="utf-8"))
    params = {c: dict(v) for c, v in ladder["frozen_params"].items()}
    print("frozen F1 params (round 1, not re-searched): " + json.dumps(params), flush=True)

    design = build_design(a.version)
    print(f"design {design.shape} in {time.time() - t0:.1f}s", flush=True)

    report: dict = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "possessions_version": a.version,
        "frozen_params": params,
        "arms": list(FG.R2_ARMS),
        "feature_sets": {k: FG.feature_set(v) for k, v in FG.R2_ARM_FEATURE_SET.items()},
        "ineligible": sorted(FG.R2_INELIGIBLE),
        "thresholds": {"gt_margin": FG.R2_GT_MARGIN, "gt_seconds": FG.R2_GT_SECONDS,
                       "eg_seconds": FG.R2_EG_SECONDS,
                       "eg_lo": FG.R2_EG_LO, "eg_hi": FG.R2_EG_HI},
        "detail": {}, "indicator_exposure": {},
    }

    # how often each indicator fires, per class per fold -- reported so a null
    # result cannot be mistaken for a feature that never turned on
    for fold in ("F1", "F2"):
        tr_all, te_all = FG.fold_slices(design, fold)
        for c in FG.SHOT_CLASSES:
            te = FG.class_slice(te_all, c)
            report["indicator_exposure"][f"{fold}|{c}"] = {
                "n_test": int(len(te)),
                **{k: {"n": int(te[k].sum()),
                       "pct": round(float(te[k].mean() * 100), 4),
                       "make_pct": (round(float(te.loc[te[k] > 0, "y"].mean() * 100), 3)
                                    if te[k].sum() else None)}
                   for k in FG.R2_INDICATORS},
                "score_diff_pre_sd": round(float(te[FG.R2_MARGIN_COL].std()), 4),
            }

    rows: list[dict] = []
    for fold in ("F1", "F2"):
        tr_all, te_all = FG.fold_slices(design, fold)
        print(f"\n=== {fold}: train {sorted(tr_all['season'].unique())} "
              f"test {sorted(te_all['season'].unique())} ===", flush=True)
        for c in FG.SHOT_CLASSES:
            tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
            for arm in FG.R2_ARMS:
                s, model, fit_s, _ = fit_one(arm, tr, te, params[c])
                rows.append(flat_row(s, fold, c, arm, FG.R2_ARM_FEATURE_SET[arm], fit_s))
                report["detail"][f"{fold}|{c}|{arm}"] = {
                    k: s[k] for k in ("n", "log_loss", "brier", "calibration",
                                      "calib_pass", "calib_worst_gap_pp",
                                      "responsiveness", "resp_decision8",
                                      "resp_pass_decision8", "by_chance",
                                      "actual_make_rate", "pred_make_rate")}
                print(f"  {c:<10} {arm}  ll={s['log_loss']:.6f}  "
                      f"calib={'PASS' if s['calib_pass'] else 'FAIL'} "
                      f"({s['calib_worst_gap_pp']:.3f} pp)  "
                      f"D8={'PASS' if s['resp_pass_decision8'] else 'FAIL'}  "
                      f"{fit_s:.1f}s", flush=True)
                if fold == "F2" and a.export:
                    d = R2_DIR / arm
                    d.mkdir(parents=True, exist_ok=True)
                    joblib.dump({
                        "arm": "lgbm", "round2_arm": arm,
                        "feature_set": FG.R2_ARM_FEATURE_SET[arm],
                        "features": FG.feature_set(FG.R2_ARM_FEATURE_SET[arm]),
                        "model": model, "fold": fold, "shot_class": c,
                        "adopted": False,
                        "possessions_version": a.version,
                        "note": "fg_make round 2, experiments.md section 13; "
                                "PROVISIONAL until the round-2 decision",
                    }, d / f"fg_make_{c}_{fold}.joblib")

    grid = pd.DataFrame(rows)
    grid.to_csv(R2_DIR / "grid_results.csv", index=False)

    # ---- S-A reproduction check ------------------------------------------
    repro = {}
    for c in FG.SHOT_CLASSES:
        got = float(grid[(grid.fold == "F2") & (grid.shot_class == c)
                         & (grid.arm == "S_A")]["log_loss"].iloc[0])
        repro[c] = {"round1": R1_F2_LGBM_LOGLOSS[c], "round2_S_A": got,
                    "delta": round(got - R1_F2_LGBM_LOGLOSS[c], 8)}
        print(f"S-A reproduction {c}: round1 {R1_F2_LGBM_LOGLOSS[c]:.6f} vs "
              f"{got:.6f} (delta {got - R1_F2_LGBM_LOGLOSS[c]:+.2e})", flush=True)
    report["s_a_reproduction"] = repro

    # ---- noise floor: five seed-varied refits of S-A on F2 ----------------
    floors = {}
    if not a.skip_floor:
        tr_all, te_all = FG.fold_slices(design, "F2")
        for c in FG.SHOT_CLASSES:
            tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
            lls = []
            for sd in SEED_GRID:
                m = FG.fit_arm("lgbm", tr, FG.R2_ARM_FEATURE_SET["S_A"],
                               seed=sd, params=params[c])
                p = FG.predict_arm("lgbm", m, te, FG.R2_ARM_FEATURE_SET["S_A"])
                lls.append(PM.log_loss(te["y"].to_numpy(), p))
            seed_sd = float(np.std(lls, ddof=1))
            floors[c] = {"seed_refit_sd": seed_sd, "seed_log_losses": lls,
                         "round1_block_bootstrap_se": R1_BOOTSTRAP_FLOOR[c],
                         "floor": max(seed_sd, R1_BOOTSTRAP_FLOOR[c])}
            print(f"floor {c}: seed SD {seed_sd:.6f}, round-1 bootstrap "
                  f"{R1_BOOTSTRAP_FLOOR[c]:.6f} -> floor "
                  f"{floors[c]['floor']:.6f}", flush=True)
    report["floors"] = floors

    report["runtime_s"] = round(time.time() - t0, 1)
    (R2_DIR / "run_report.json").write_text(json.dumps(report, indent=2, default=str),
                                            encoding="utf-8")
    print(f"\nwrote {R2_DIR}/grid_results.csv and run_report.json "
          f"({report['runtime_s']}s)")
    print("OFFLINE ONLY. The closed-loop gate (section 13.5) is run by "
          "scripts/run_engine.py with ENGINE_FG_MAKE=round2_<arm>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
