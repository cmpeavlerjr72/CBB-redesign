#!/usr/bin/env python
"""
train_fg_make_v3_shooter.py -- fg_make ROUND 3: the round-2b winner (S-C under
S1) retrained with the shooter keyed on `shot_shooter_id` instead of the
defective `participant_1_id` (`docs/models/change_ledger.md`, "CBBD's
`participant_1_id` is the ASSISTER...").

    .venv/Scripts/python.exe scripts/train_fg_make_v3_shooter.py

Pre-registration: `docs/models/fg_make/experiments.md` section 17, committed
BEFORE this ran (fa7c982). Evidence: `docs/tests/fg_make_shooter_key_2026-09-10.md`
(`scripts/diag_fg_make_shooter_key_v1.py`).

WHAT IS FIXED, per section 17.1
  * arm            the round-2b winner, S-C `R2_C_safe_state`, all three classes
  * scheme         S1 (monthly walk-forward), reused from
                   `possession_outcome.month_boundaries`, not reimplemented
  * model class    LightGBM per shot class, round-1 frozen F1-only parameters
  * folds          F1 reported (selects nothing), F2 SELECTION, 2026 sealed
  * FG.score()     unchanged

WHAT CHANGES: only the shooter key fed to `fg_make.build_fg_events` /
`build_design` (this round's own addition to `fg_make.py`, mirroring
`event_stream.build_stream`'s / `usage.build_usage_events`'s existing
`shooter_key` parameter): `"shot_shooter_id"` instead of the default
`"participant_1_id"`. Rows with no `shot_shooter_id` are DROPPED, never
imputed, and the drop is reported by season and by team (never just pooled).

The REFERENCE is round 2b's own already-computed numbers
(`data/processed/models/fg_make/round2b/run_report.json`), read back rather
than recomputed, exactly as `train_fg_make_v2b_s1.py` read round 2's own
artifacts back as ITS reference. `train_fg_make_v2.py` and
`train_fg_make_v2b_s1.py` are not modified or imported for side effects.

NOISE FLOOR: a second-seed refit of the ENTIRE new arm (section 17.3), not a
reused round-1/round-2 floor -- the data changed this round, so the floor is
recomputed on it. `abs(log_loss(seed=0) - log_loss(seed=1))` per class.

Artifacts (a new directory; nothing existing is overwritten):
    data/processed/models/fg_make/round3_shooter/S_C_s1/<class>_<refit_date>.joblib
    data/processed/models/fg_make/round3_shooter/S_C_s1/manifest_<class>.json
    data/processed/models/fg_make/round3_shooter/run_report.json
    data/processed/models/fg_make/events_v2_shotshooter.parquet   (cache)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Worker discipline: five other workers share this machine. Cap every
# thread-pool library at 4 threads, set BEFORE numpy/lightgbm import anything
# that reads these at import time.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402

OUT_DIR = Path("data/processed/models/fg_make")
R2B = OUT_DIR / "round2b"
R3 = OUT_DIR / "round3_shooter" / "S_C_s1"
EVENTS_CACHE = OUT_DIR / "events_v2_shotshooter.parquet"
ARM = "S_C"
FS = FG.R2_ARM_FEATURE_SET[ARM]
SEASONS = [2022, 2023, 2024, 2025]
SHOOTER_KEY = "shot_shooter_id"


def build_events(version: str) -> pd.DataFrame:
    """The shot_shooter_id-keyed event cache. Built once, reused across the
    two seed refits (seed 0 and the seed-1 floor refit) so the drop report and
    the design are computed on exactly one build."""
    uni = ES.load_universe(FG.DEFAULT_UNIVERSE, require_pbp_complete=True)
    if EVENTS_CACHE.exists():
        ev = pd.read_parquet(EVENTS_CACHE)
        ev.attrs["rim_override_max_ft"] = ES.rim_override_for_version(version)
        ev.attrs["possessions_version"] = version
        ev.attrs["shooter_key"] = SHOOTER_KEY
        return ev, uni
    ev = FG.build_fg_events(SEASONS, universe=uni, version=version, shooter_key=SHOOTER_KEY)
    ev.to_parquet(EVENTS_CACHE)
    return ev, uni


def drop_report(events: pd.DataFrame, min_rows: int = 50) -> dict:
    """Rows lost to a missing `shot_shooter_id`, by class, by season, by team
    (never imputed) -- the report this round's pre-registration requires
    (section 17.0 item 4)."""
    miss = ~np.isfinite(events["shooter_id"].to_numpy(dtype="float64"))
    out: dict = {"pooled_by_class": {}, "by_season_class": {}, "by_team": {}}
    for c in FG.SHOT_CLASSES:
        m = events["shot_class"].to_numpy() == c
        n = int(m.sum())
        out["pooled_by_class"][c] = {
            "n": n, "dropped": int(miss[m].sum()),
            "drop_pct": round(float(miss[m].mean() * 100), 4) if n else None,
        }
        for s in SEASONS:
            ms = m & (events["season"].to_numpy() == s)
            ns = int(ms.sum())
            if not ns:
                continue
            out["by_season_class"][f"{s}|{c}"] = {
                "n": ns, "dropped": int(miss[ms].sum()),
                "drop_pct": round(float(miss[ms].mean() * 100), 4),
            }
        tid = events["off_team_id"].to_numpy()
        g = pd.DataFrame({"t": tid[m], "miss": miss[m]}).groupby("t")["miss"].agg(["size", "mean"])
        big = g[g["size"] >= min_rows]
        if len(big):
            q = big["mean"] * 100
            out["by_team"][c] = {
                "teams": int(len(big)), "underpowered_teams": int(len(g) - len(big)),
                "min_pct": round(float(q.min()), 4), "median_pct": round(float(q.median()), 4),
                "p95_pct": round(float(q.quantile(0.95)), 4), "max_pct": round(float(q.max()), 4),
            }
    return out


def fit_s1_schedule(design: pd.DataFrame, params: dict, seed: int,
                    export: bool) -> tuple[dict, dict]:
    """One full S1 schedule (all classes, all monthly refits) at one seed.

    Returns `(scores_by_class, segments_by_class)`. When `export`, also writes
    the joblibs and manifests to `R3` (seed 0 only -- the seed-1 refit is the
    noise floor and is not exported as a served artifact)."""
    tr_all, te_all = FG.fold_slices(design, "F2")
    scores: dict = {}
    segs_by_class: dict = {}
    for c in FG.SHOT_CLASSES:
        tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
        feats = FG.feature_set(FS)
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
            rows = tr[fit_cols] if not len(prior) else pd.concat(
                [tr[fit_cols], prior], ignore_index=True)
            t1 = time.time()
            m = FG.fit_arm("lgbm", rows, FS, seed=seed, params=params[c])
            p_s1[seg] = FG.predict_arm("lgbm", m, te.loc[seg], FS)
            max_train = (tr_dates.max() if not before.any()
                         else max(tr_dates.max(), te_dates[before].max()))
            seg_info = {"refit_date": str(cut.date()), "n_train": int(len(rows)),
                       "n_train_from_test_season": int(len(prior)),
                       "n_scored": int(seg.sum()),
                       "max_train_date": str(pd.Timestamp(max_train).date()),
                       "fit_s": round(time.time() - t1, 1)}
            if export:
                fname = f"{c}_{cut.date()}.joblib"
                joblib.dump({"arm": "lgbm", "round2_arm": ARM, "scheme": "S1",
                            "shooter_key": SHOOTER_KEY,
                            "feature_set": FS, "features": feats, "model": m,
                            "fold": "F2", "shot_class": c, "adopted": False,
                            "refit_date": str(cut.date()),
                            "max_train_date": seg_info["max_train_date"],
                            "possessions_version": "v2",
                            "note": "fg_make round 3, experiments.md section 17: "
                                    "round-2b winner (S-C, S1) retrained on "
                                    "shot_shooter_id"},
                            R3 / fname)
                seg_info["path"] = fname
            segments.append(seg_info)
            print(f"  seed={seed} {c} refit {cut.date()}: train {len(rows):,} "
                  f"(test-season {len(prior):,}), scores {int(seg.sum()):,}, "
                  f"{seg_info['fit_s']}s", flush=True)
        if (p_s1.sum(axis=1) == 0).any():
            raise AssertionError("S1 left test rows unscored; the month partition is not a cover")
        scores[c] = FG.score(te, p_s1)
        segs_by_class[c] = segments
        if export:
            (R3 / f"manifest_{c}.json").write_text(json.dumps({
                "model": "fg_make", "scheme": "S1", "fold": "F2", "season": 2025,
                "key": c, "arm": ARM, "feature_set": FS, "features": feats,
                "shooter_key": SHOOTER_KEY,
                "artifacts": [{k: v for k, v in s.items()
                              if k in ("refit_date", "path", "max_train_date", "n_train")}
                             for s in segments],
            }, indent=2), encoding="utf-8")
    return scores, segs_by_class


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    a = ap.parse_args()
    t0 = time.time()
    R3.mkdir(parents=True, exist_ok=True)
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="fg_make round 3")

    ladder = json.loads((OUT_DIR / f"lgbm_ladder_{a.version}.json").read_text(encoding="utf-8"))
    params = {c: dict(v) for c, v in ladder["frozen_params"].items()}
    # NOTE: `fg_make.LgbmArm.fit` hardcodes `n_jobs=-1`, so a per-class
    # `n_jobs` override in `params` would collide with it (duplicate kwarg).
    # The thread cap for this script is the OMP_NUM_THREADS/... env vars set
    # at import time above, which LightGBM's native thread pool reads when
    # `n_jobs`/`num_threads` is left at its auto (-1) setting.

    events, uni = build_events(a.version)
    drops = drop_report(events)
    print("drop report (pooled by class): "
          + json.dumps(drops["pooled_by_class"]), flush=True)

    design = FG.add_round2_state(FG.build_design(SEASONS, universe=uni, version=a.version,
                                                 events=events))
    for col in [FG.R2_MARGIN_COL, *FG.R2_INDICATORS]:
        v = design[col].to_numpy()
        assert np.isfinite(v).all(), f"{col} carries non-finite values"
    print(f"design {design.shape}; shooter_key={SHOOTER_KEY} ({time.time() - t0:.1f}s)",
          flush=True)

    print("=== seed 0 (the served arm) ===", flush=True)
    scores0, segs0 = fit_s1_schedule(design, params, seed=0, export=True)
    print("=== seed 1 (the noise floor) ===", flush=True)
    scores1, _ = fit_s1_schedule(design, params, seed=1, export=False)

    ref = json.loads((R2B / "run_report.json").read_text(encoding="utf-8"))

    report: dict = {"created_at": pd.Timestamp.now("UTC").isoformat(), "arm": ARM,
                    "scheme": "S1", "shooter_key": SHOOTER_KEY, "feature_set": FS,
                    "reference": "data/processed/models/fg_make/round2b/run_report.json",
                    "drop_report": drops, "by_class": {}}

    decisions = {}
    for c in FG.SHOT_CLASSES:
        s0, s1seed = scores0[c], scores1[c]
        floor = abs(s0["log_loss"] - s1seed["log_loss"])
        rc = ref["by_class"][c]["s1"]
        rc_detail = ref["by_class"][c]["s1_detail"]
        delta_ll = s0["log_loss"] - rc["log_loss"]

        shooter_before = rc_detail["resp_decision8"]["by_driver"]["shooter_make_c->MAKE"]
        shooter_after = s0["resp_decision8"]["by_driver"]["shooter_make_c->MAKE"]

        calib_ok = s0["calib_pass"] or not rc["calib_pass"]
        d8_ok = s0["resp_pass_decision8"] or not rc["resp_pass_decision8"]
        ll_ok = delta_ll <= floor
        adopt = bool(calib_ok and d8_ok and ll_ok)
        reason = []
        if not calib_ok:
            reason.append("calibration regressed vs reference")
        if not d8_ok:
            reason.append("Decision 8 regressed vs reference")
        if not ll_ok:
            reason.append(f"log loss worse than reference by {delta_ll:.6f} "
                          f"({delta_ll / floor:.2f} floors), beyond the {floor:.6f} floor")
        decisions[c] = {"adopt": adopt, "reason": reason or ["no gate regressed beyond the floor"]}

        report["by_class"][c] = {
            "n_test": s0["n"], "n_refits": len(segs0[c]), "segments": segs0[c],
            "reference_s1": {k: rc[k] for k in ("log_loss", "brier", "calib_pass",
                                                "calib_worst_gap_pp", "resp_pass_decision8",
                                                "pred_make_rate", "actual_make_rate")},
            "new_s1_shooterfix": {k: s0[k] for k in ("log_loss", "brier", "calib_pass",
                                                     "calib_worst_gap_pp", "resp_pass_decision8",
                                                     "pred_make_rate", "actual_make_rate")},
            "seed1_floor_refit": {"log_loss": s1seed["log_loss"]},
            "noise_floor_second_seed": round(floor, 6),
            "delta_log_loss_vs_reference": round(delta_ll, 6),
            "delta_in_floors": round(delta_ll / floor, 3) if floor else None,
            "shooter_make_c_decision8": {
                "reference": shooter_before, "new": shooter_after,
            },
            "new_detail": {"calibration": s0["calibration"], "resp_decision8": s0["resp_decision8"],
                          "by_chance": s0["by_chance"]},
            "decision": decisions[c],
        }
        print(f"{c}: ref ll {rc['log_loss']:.6f} -> new ll {s0['log_loss']:.6f} "
              f"(delta {delta_ll:+.6f}, floor {floor:.6f}, "
              f"{delta_ll / floor:+.2f} floors) | shooter slope {shooter_before['slope_ratio']} "
              f"-> {shooter_after['slope_ratio']} | ADOPT={adopt}", flush=True)

    report["decision"] = decisions
    report["runtime_s"] = round(time.time() - t0, 1)
    (R3.parent / "run_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {R3.parent}/run_report.json ({report['runtime_s']}s)")
    print("OFFLINE ONLY. The Decision-10 closed-loop for this arm needs an "
          "adapters.py change (experiments.md section 17.4) and is not run here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
