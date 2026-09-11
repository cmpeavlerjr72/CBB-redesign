"""train_clock_v4.py -- ROUND 4: fit and score the calendar-level clock arms.

Pre-registration: `docs/models/clock/experiments.md` section 14, committed
(421b97b) BEFORE this file was written. Diagnosis:
`docs/tests/clock_duration_shortfall_2026-09-11.md`.

Three stages, in this order and no other:

  1. **F1 ONLY**: choose A1's recency half-life from the pre-registered grid
     {120, 365, 730} days by CRPS_trunc on the F1 test season (2024), under the
     same S1 schedule the F2 arms use. The choice is written to disk before F2
     is touched.
  2. **F2**: fit an S1 schedule per arm (R, A1, A2, A3, A4) and persist it with
     a manifest the engine adapter can read.
  3. **Offline scoring**, one blind path for every arm: CRPS_trunc (primary
     offline), censored log-likelihood, PIT by cell, the `chain_halves`
     emergent G1 on clock-complete games, the end-of-half pair, the segment
     table and the tempo-quintile responsiveness check.

The closed-loop stage is a separate script (`scripts/run_clk4_closed_loop.py`)
because it is the deciding read and must run per arm under pinned sub-models.

Usage (threads capped at 4; three other workers share this machine):
    OMP_NUM_THREADS=4 .venv/Scripts/python.exe scripts/train_clock_v4.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cbb_sim.models import clock as ck  # noqa: E402
from cbb_sim.models import clock_v3 as c3  # noqa: E402
from cbb_sim.models import clock_v4 as c4  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

OUT = Path("data/processed/models/clock")
S1_DIR = OUT / "v4_s1"
DESIGN_V2 = OUT / "design_v2.parquet"
ALL_SEASONS = [2022, 2023, 2024, 2025]
BASE_SEED = 20260911
TEST_SEASON = 2025
F1_TEST_SEASON = 2024

#: (tag, engine mode suffix). `ref` is the served round-3c arm refitted inside
#: this harness so one code path scores every arm; the ENGINE reference is the
#: existing `v3c_srfloor_P3_s1` artifacts, not these.
ARMS: tuple[str, ...] = ("ref", "recency", "curseason", "calpart", "nofloor")


def log(m: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {m}", flush=True)


def fit_schedule(tag: str, tr: pd.DataFrame, te: pd.DataFrame, season: int,
                 half_life: float | None, persist: Path | None) -> tuple[c3.MonthlyArm, dict]:
    """One S1 schedule: refit at every month boundary of the test season on all
    prior seasons plus the test season strictly before that boundary."""
    te_dates = pd.to_datetime(te["game_date"])
    tr_dates = pd.to_datetime(tr["game_date"])
    cuts = PO.month_boundaries(te_dates)
    cols = [c for c in dict.fromkeys([*c4.s1_fit_columns_v4(tag),
                                      *ck.feature_set(c3.P_FEATURES["P3"])])
            if c in tr.columns]
    tr_small = tr[cols]

    arms, months, kept = [], [], []
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = ((te_dates >= cut) if nxt is None
               else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not seg.any():
            continue
        before = (te_dates < cut).to_numpy()
        prior = te.loc[before, cols]
        fit_rows = tr_small if not len(prior) else pd.concat([tr_small, prior],
                                                             ignore_index=True)
        t0 = time.time()
        arm = c4.fit_arm_v4(tag, fit_rows, parametrisation="P3", cur_season=season,
                            half_life_days=half_life, refit_date=cut)
        arms.append(arm)
        kept.append(pd.Timestamp(cut))
        stamp = f"{pd.Timestamp(cut).year:04d}-{pd.Timestamp(cut).month:02d}"
        mfile = None
        if persist is not None:
            persist.mkdir(parents=True, exist_ok=True)
            mfile = persist / f"{tag}_S1_{season}_{stamp}.pkl"
            with open(mfile, "wb") as f:
                pickle.dump(arm, f)
        mx = tr_dates.max() if not before.any() else max(tr_dates.max(),
                                                         te_dates[before].max())
        months.append({
            "refit_date": str(pd.Timestamp(cut).date()),
            "valid_from": str(pd.Timestamp(cut).date()),
            "valid_to": None if nxt is None
            else str((pd.Timestamp(nxt) - pd.Timedelta(days=1)).date()),
            "n_train": int(len(fit_rows)),
            "n_train_from_test_season": int(len(prior)),
            "n_scored": int(seg.sum()),
            "max_train_date": str(pd.Timestamp(mx).date()),
            "season": int(season), "month": stamp,
            "half_life_days": None if half_life is None else float(half_life),
            "model_file": None if mfile is None
            else str(mfile.relative_to(OUT)).replace("\\", "/"),
            "fit_seconds": round(time.time() - t0, 1),
        })
        log(f"  {tag} refit {months[-1]['refit_date']}: {len(fit_rows):,} rows "
            f"({len(prior):,} from the test season), scores {int(seg.sum()):,}, "
            f"{months[-1]['fit_seconds']:.0f}s")
        del fit_rows, prior
    if not months:
        raise AssertionError(f"{tag}: S1 produced no refits")
    spec = c4.V4_ARMS[tag]
    manifest = {
        "base_arm": spec["base"], "parametrisation": "P3", "tag": tag, "scheme": "S1",
        "calendar_mode": spec["cal"],
        "half_life_days": None if half_life is None else float(half_life),
        "season": int(season), "fold": "F2" if season == TEST_SEASON else "F1",
        "round": "4",
        "selection_rule": ("pick the row whose [valid_from, valid_to] contains the "
                           "game's own date; equivalently the latest refit_date at or "
                           "before it. valid_to null means the last month of the season."),
        "naming": "{tag}_S1_{season}_{YYYY-MM}.pkl, relative to "
                  "data/processed/models/clock/",
        "arm_adopted": False,
        "note": ("fitted for the round-4 bake-off (experiments.md section 14); "
                 "adoption is decided by the closed-loop run, not here. No lookup "
                 "table is exported: the engine runs the live fitted object."),
        "months": months,
    }
    if persist is not None:
        (persist / f"manifest_{tag}.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return c3.MonthlyArm(cuts=kept, arms=arms, base_arm=spec["base"],
                         name=f"v4_{tag}_s1"), manifest


def score(tag: str, arm, te: pd.DataFrame, actual_half: pd.DataFrame,
          design: pd.DataFrame, out_dir: Path, seed: int = BASE_SEED) -> dict:
    """The one blind scoring path, identical for every arm."""
    t0 = time.time()
    s = c3.score_arm_v3(arm, te)
    pit = c3.pit_by_cell_v3(te, s["_pit_rows"])
    pit.to_csv(out_dir / f"v4_pit_cells_F2_{tag}.csv", index=False)
    seg = c3.segment_table(te, s["_crps_trunc_rows"], s["_loglik_rows"])
    seg.to_csv(out_dir / f"v4_segments_F2_{tag}.csv", index=False)

    res = ck.chain_halves(arm, te, seed=seed)
    em = c3.emergent_cc(res, actual_half)
    pd.DataFrame(em["months"]).to_csv(out_dir / f"v4_cc_months_F2_{tag}.csv", index=False)
    resp = c3.responsiveness(res, te)
    resp.to_csv(out_dir / f"v4_responsiveness_F2_{tag}.csv", index=False)
    band = c3.pred_mean_by_band(arm, te)
    band.to_csv(out_dir / f"v4_band_means_F2_{tag}.csv", index=False)
    allg = ck.emergent_report(res, actual_half)

    powered = pit[pit["powered"]]
    row = {
        "arm": tag,
        "crps_trunc": float(s["crps_trunc"]),
        "crps_r2def": float(s["crps_r2def"]),
        "censored_loglik": float(s["censored_loglik"]),
        "pred_mean_duration": float(s["pred_mean_duration"]),
        "actual_mean_duration_uncensored": float(s["actual_mean_duration_uncensored"]),
        "pit_worst_D": float(powered["ks_D"].max()) if len(powered) else float("nan"),
        "pit_leak_failures": int(powered["leak_sized"].sum()) if len(powered) else 0,
        "pit_powered": int(len(powered)), "pit_underpowered": int((~pit["powered"]).sum()),
        "pit_pass": bool(len(powered) and not powered["leak_sized"].any()),
        "cc_n_games": em["n_games"], "cc_sim_mean": em["sim_mean"],
        "cc_actual_mean": em["actual_mean"], "cc_mean_delta": em["mean_delta"],
        "cc_sd_delta": em["sd_delta"], "cc_months_pass": em["months_pass"],
        "cc_n_powered_months": em["n_powered_months"], "cc_g1_pass": em["g1_pass"],
        "cc_eoh_share_gap": em["eoh_share_gap"],
        "cc_eoh_duration_gap": em["eoh_duration_gap"],
        "cc_eoh_sim_dur": em["eoh_sim_dur"], "cc_eoh_actual_dur": em["eoh_actual_dur"],
        "resp_slope_ratio": float(resp["slope_ratio"].iloc[0]),
        "resp_steps_agreeing": int(resp["steps_agreeing"].iloc[0]),
        "all_mean_delta": float(allg.get("mean_delta", np.nan)),
        "all_sd_delta": float(allg.get("sd_delta", np.nan)),
        "score_seconds": round(time.time() - t0, 1),
    }
    log(f"  {tag}: CRPS_trunc {row['crps_trunc']:.5f}, G1-cc {row['cc_mean_delta']:+.3f}, "
        f"PIT worst D {row['pit_worst_D']:.4f}, eoh dur gap "
        f"{row['cc_eoh_duration_gap']:+.3f}s, {row['score_seconds']:.0f}s")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated arm tags")
    ap.add_argument("--skip-f1", action="store_true",
                    help="reuse the half-life already written to v4_f1_halflife.json")
    a = ap.parse_args()

    S1_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()

    log(f"loading {DESIGN_V2}")
    design = pd.read_parquet(DESIGN_V2)
    design, cdiag = c3.attach_horn_censoring(design, ALL_SEASONS)
    c3.set_flag_inplace(design, "horn")
    log(f"horn-censored {cdiag['censored_horn_pct']}%")

    # ---- stage 1: F1 ONLY, the half-life -------------------------------
    hl_path = OUT / "v4_f1_halflife.json"
    if a.skip_f1 and hl_path.exists():
        hl_doc = json.loads(hl_path.read_text(encoding="utf-8"))
        log(f"reusing F1 half-life {hl_doc['chosen_half_life_days']} days")
    else:
        tr1, te1 = ck.fold_slices(design, "F1")
        log(f"F1 train {len(tr1):,} / test {len(te1):,}; choosing the half-life")
        rows = []
        for hl in c4.HALF_LIFE_GRID_DAYS:
            marm, _ = fit_schedule("recency", tr1, te1, F1_TEST_SEASON, float(hl), None)
            s = c3.score_arm_v3(marm, te1)
            rows.append({"half_life_days": int(hl),
                         "f1_crps_trunc": float(s["crps_trunc"]),
                         "f1_censored_loglik": float(s["censored_loglik"]),
                         "f1_pred_mean_duration": float(s["pred_mean_duration"]),
                         "f1_actual_mean_duration": float(s["actual_mean_duration_uncensored"])})
            log(f"  half-life {hl}d: CRPS_trunc {rows[-1]['f1_crps_trunc']:.5f}, "
                f"pred mean {rows[-1]['f1_pred_mean_duration']:.3f} vs actual "
                f"{rows[-1]['f1_actual_mean_duration']:.3f}")
            del marm
        best = min(rows, key=lambda r: r["f1_crps_trunc"])
        hl_doc = {"grid": list(c4.HALF_LIFE_GRID_DAYS), "rows": rows,
                  "chosen_half_life_days": best["half_life_days"],
                  "chosen_on": "F1 only, by CRPS_trunc (pre-registration 14.2)",
                  "written_at": datetime.now(UTC).isoformat()}
        hl_path.write_text(json.dumps(hl_doc, indent=2), encoding="utf-8")
        log(f"CHOSEN half-life: {best['half_life_days']} days (F1 only)")
        del tr1, te1

    half_life = float(hl_doc["chosen_half_life_days"])

    # ---- stage 2 + 3: F2 ------------------------------------------------
    tr, te = ck.fold_slices(design, "F2")
    hc = c3.load_clock_complete()
    actual_half = c3.actual_end_of_half_cc(te, hc)
    log(f"F2 train {len(tr):,} / test {len(te):,}; "
        f"{int(actual_half.groupby('game_id')['clock_complete'].all().sum())} clock-complete games")
    del design

    want = {s.strip() for s in a.only.split(",") if s.strip()}
    grid, manifests = [], {}
    for tag in ARMS:
        if want and tag not in want:
            continue
        log(f"S1 schedule {tag} (calendar {c4.V4_ARMS[tag]['cal']}, "
            f"half-life {half_life if c4.V4_ARMS[tag]['recency'] else None})")
        marm, man = fit_schedule(tag, tr, te, TEST_SEASON,
                                 half_life if c4.V4_ARMS[tag]["recency"] else None, S1_DIR)
        manifests[tag] = man
        grid.append(score(tag, marm, te, actual_half, te, OUT))
        del marm

    g = pd.DataFrame(grid).sort_values("crps_trunc").reset_index(drop=True)
    g.to_csv(OUT / "v4_grid_F2.csv", index=False)
    (S1_DIR / "index.json").write_text(json.dumps({
        "round": "4", "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "half_life_days": half_life,
        "schedules": {k: {"base_arm": v["base_arm"], "calendar_mode": v["calendar_mode"],
                          "half_life_days": v["half_life_days"],
                          "n_months": len(v["months"])} for k, v in manifests.items()},
    }, indent=2, default=str), encoding="utf-8")
    log("offline grid:\n" + g.to_string(index=False))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
