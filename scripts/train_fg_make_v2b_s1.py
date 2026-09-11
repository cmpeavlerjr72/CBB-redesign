#!/usr/bin/env python
"""
train_fg_make_v2b_s1.py -- fg_make ROUND 2b: the round-2 winner (S-C) under the
standing S1 scheme vs the same arm fitted once.

    .venv/Scripts/python.exe scripts/train_fg_make_v2b_s1.py

Pre-registration: `docs/models/fg_make/experiments.md` section 15, written and
appended after round 2 decided and before this ran.

S1 IS NOT REIMPLEMENTED HERE. The refit schedule is
`possession_outcome.month_boundaries`, called directly, so "monthly refit" means
the same thing in this model as in the model that adopted it (L21). What this
script adds is the per-segment PERSISTENCE the engine needs: one joblib per
(shot class, refit date) plus a `manifest.py`-format manifest declaring
`refit_date`, `path` and `max_train_date` for each.

`train_fg_make_v2.py` and `train_fg_make_v1.py` are not modified or imported for
their side effects; the S-C static arm is read back from its round-2 artifacts
so the comparison is against the exact object round 2 decided on.

Artifacts (a new directory; nothing existing is overwritten):
    data/processed/models/fg_make/round2b/S_C_s1/<class>_<refit_date>.joblib
    data/processed/models/fg_make/round2b/S_C_s1/manifest_<class>.json
    data/processed/models/fg_make/round2b/run_report.json
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
from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402

OUT_DIR = Path("data/processed/models/fg_make")
R2 = OUT_DIR / "round2"
R2B = OUT_DIR / "round2b" / "S_C_s1"
ARM = "S_C"
FS = FG.R2_ARM_FEATURE_SET[ARM]
SEASONS = [2022, 2023, 2024, 2025]
FLOOR = {"FGA_rim": 0.000539, "FGA_jump2": 0.000578, "FGA_3": 0.000849}


def build_design(version: str) -> pd.DataFrame:
    uni = ES.load_universe(FG.DEFAULT_UNIVERSE, require_pbp_complete=True)
    ev = pd.read_parquet(OUT_DIR / f"events_{version}.parquet")
    ev.attrs["rim_override_max_ft"] = ES.rim_override_for_version(version)
    ev.attrs["possessions_version"] = version
    return FG.add_round2_state(
        FG.build_design(SEASONS, universe=uni, version=version, events=ev))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    a = ap.parse_args()
    t0 = time.time()
    R2B.mkdir(parents=True, exist_ok=True)
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="fg_make round 2b")

    ladder = json.loads((OUT_DIR / f"lgbm_ladder_{a.version}.json").read_text(encoding="utf-8"))
    params = {c: dict(v) for c, v in ladder["frozen_params"].items()}
    feats = FG.feature_set(FS)
    design = build_design(a.version)
    tr_all, te_all = FG.fold_slices(design, "F2")
    print(f"design {design.shape}; F2 train {len(tr_all):,} test {len(te_all):,} "
          f"({time.time() - t0:.1f}s)", flush=True)

    report: dict = {"created_at": pd.Timestamp.now("UTC").isoformat(), "arm": ARM,
                    "feature_set": FS, "features": feats, "floors": FLOOR,
                    "frozen_params": params, "by_class": {}}

    for c in FG.SHOT_CLASSES:
        tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
        y = te["y"].to_numpy()

        # ---- static arm: read back the object round 2 decided on ----------
        w = joblib.load(R2 / ARM / f"fg_make_{c}_F2.joblib")
        assert w["feature_set"] == FS and w["features"] == feats, \
            "the round-2 S_C artifact is not the arm this round is comparing"
        p_static = FG.predict_arm("lgbm", w["model"], te, FS)
        s_static = FG.score(te, p_static)

        # ---- S1: the possession_outcome schedule, one fit per month -------
        te_dates = pd.to_datetime(te["game_date"])
        tr_dates = pd.to_datetime(tr["game_date"])
        cuts = PO.month_boundaries(te_dates)
        fit_cols = [*feats, "y"]
        p_s1 = np.zeros((len(te), len(FG.CLASSES)), dtype="float64")
        segments = []
        for k, cut in enumerate(cuts):
            nxt = cuts[k + 1] if k + 1 < len(cuts) else None
            seg = ((te_dates >= cut) if nxt is None
                   else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
            if not seg.any():
                continue
            before = (te_dates < cut).to_numpy()
            prior = te.loc[before, fit_cols]
            rows = tr[fit_cols] if not len(prior) else pd.concat([tr[fit_cols], prior],
                                                                 ignore_index=True)
            t1 = time.time()
            m = FG.fit_arm("lgbm", rows, FS, seed=0, params=params[c])
            p_s1[seg] = FG.predict_arm("lgbm", m, te.loc[seg], FS)
            max_train = (tr_dates.max() if not before.any()
                         else max(tr_dates.max(), te_dates[before].max()))
            fname = f"{c}_{cut.date()}.joblib"
            joblib.dump({"arm": "lgbm", "round2_arm": ARM, "scheme": "S1",
                         "feature_set": FS, "features": feats, "model": m,
                         "fold": "F2", "shot_class": c, "adopted": False,
                         "refit_date": str(cut.date()),
                         "max_train_date": str(pd.Timestamp(max_train).date()),
                         "possessions_version": a.version,
                         "note": "fg_make round 2b, experiments.md section 15"},
                        R2B / fname)
            segments.append({"refit_date": str(cut.date()), "path": fname,
                             "max_train_date": str(pd.Timestamp(max_train).date()),
                             "n_train": int(len(rows)),
                             "n_train_from_test_season": int(len(prior)),
                             "n_scored": int(seg.sum()),
                             "fit_s": round(time.time() - t1, 1)})
            print(f"  {c} refit {cut.date()}: train {len(rows):,} "
                  f"(test-season {len(prior):,}), scores {int(seg.sum()):,}, "
                  f"max_train {segments[-1]['max_train_date']}, "
                  f"{segments[-1]['fit_s']}s", flush=True)
        if (p_s1.sum(axis=1) == 0).any():
            raise AssertionError("S1 left test rows unscored; the month partition is not a cover")
        s_s1 = FG.score(te, p_s1)

        (R2B / f"manifest_{c}.json").write_text(json.dumps({
            "model": "fg_make", "scheme": "S1", "fold": "F2", "season": 2025,
            "key": c, "arm": ARM, "feature_set": FS, "features": feats,
            "artifacts": [{k: v for k, v in s.items()
                           if k in ("refit_date", "path", "max_train_date", "n_train")}
                          for s in segments],
        }, indent=2), encoding="utf-8")

        delta = s_s1["log_loss"] - s_static["log_loss"]
        report["by_class"][c] = {
            "n_test": int(len(te)), "n_refits": len(segments), "segments": segments,
            "static": {k: s_static[k] for k in ("log_loss", "brier", "calib_pass",
                                                "calib_worst_gap_pp", "resp_pass_decision8",
                                                "pred_make_rate", "actual_make_rate")},
            "s1": {k: s_s1[k] for k in ("log_loss", "brier", "calib_pass",
                                        "calib_worst_gap_pp", "resp_pass_decision8",
                                        "pred_make_rate", "actual_make_rate")},
            "delta_log_loss": round(delta, 6),
            "delta_in_floors": round(delta / FLOOR[c], 3),
            "static_detail": {"calibration": s_static["calibration"],
                              "resp_decision8": s_static["resp_decision8"],
                              "by_chance": s_static["by_chance"]},
            "s1_detail": {"calibration": s_s1["calibration"],
                          "resp_decision8": s_s1["resp_decision8"],
                          "by_chance": s_s1["by_chance"]},
        }
        print(f"{c}: static ll {s_static['log_loss']:.6f} calib "
              f"{s_static['calib_worst_gap_pp']:.3f} D8 {s_static['resp_pass_decision8']} | "
              f"S1 ll {s_s1['log_loss']:.6f} calib {s_s1['calib_worst_gap_pp']:.3f} "
              f"D8 {s_s1['resp_pass_decision8']} | delta {delta:+.6f} "
              f"({delta / FLOOR[c]:+.2f} floors)", flush=True)

    report["runtime_s"] = round(time.time() - t0, 1)
    (R2B.parent / "run_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {R2B.parent}/run_report.json ({report['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
