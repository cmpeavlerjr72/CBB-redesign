#!/usr/bin/env python
"""
train_rebound_v2_s1.py -- L3 REBOUND S1 SCHEME CONFIRMATION: which refit
calendar (Decision 9c), model class and feature set held fixed.

    .venv/Scripts/python.exe scripts/train_rebound_v2_s1.py
    .venv/Scripts/python.exe scripts/train_rebound_v2_s1.py --no-append

Pre-registration: `docs/models/rebound/experiments.md` section 7, written and
COMMITTED before this ran. This script executes it and APPENDS its results;
it never edits what is already there.

Model class and feature set are HELD FIXED at round 1's winner, `lgbm` /
`C_plus_state`. Four refit calendars are compared: `S0` (static, the
reference -- must reproduce round 1's F2 log loss 0.645565), `S1_monthly`,
`S1_conf_aligned`, `S1_weekly`. The refit-calendar CONSTRUCTION is imported
from `scripts/train_possession_outcome_v3.py` (which itself calls
`cbb_sim.features.conference` and `cbb_sim.models.possession_outcome.
month_boundaries`) rather than reimplemented, so "weekly" and
"conference-aligned" are the same objects in every sub-model that tests them.

THREAD CAP. Eight other workers share this machine; every LightGBM fit here
runs at `n_jobs=3` (monkeypatched onto `RB.LgbmArm.PARAMS`, not onto the
shared module), and `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`
/`NUMEXPR_NUM_THREADS` are pinned the same way, matching
`train_possession_outcome_v3.py`'s own convention.

WALL CLOCK. Section 7.4's measured 189.2 s/fit gives a projected ~3.9 h for
the full stage list; the hard cap here is 4.5 h, checked before each cell
starts. Anything the clock does not reach is written to the report and the
appended section as NOT RUN, never as a result.

Artifacts (gitignored, HF-synced, never `git add`ed):
    data/processed/models/rebound/s1_confirm/<scheme>/<fold>/*.joblib
    data/processed/models/rebound/s1_confirm/<scheme>/<fold>/manifest.json
    data/processed/models/rebound/s1_confirm_run_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "3")

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.features import conference as CF  # noqa: E402
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

RB.LgbmArm.PARAMS = {**RB.LgbmArm.PARAMS, "n_jobs": 3}

OUT_DIR = Path("data/processed/models/rebound")
S1_DIR = OUT_DIR / "s1_confirm"
DOC = Path("docs/models/rebound/experiments.md")
SEASONS = [2022, 2023, 2024, 2025]
ARM, FS = "lgbm", "C_plus_state"
SCHEMES = ("S0", "S1_monthly", "S1_conf_aligned", "S1_weekly")
FIT_S_MEASURED = 189.2          # section 7.4
BUDGET_S = 4.5 * 3600           # section 7.4
CONF4_MIN_N = 1000              # section 7.3
CONF4_FLOOR_PP = 0.25           # section 7.4 / 7.5
# S0's F1 number is READ from section 3, never refit (section 7.2).
CITED_S0_F1 = {"log_loss": 0.620446, "calib_worst_gap_pp": 1.629, "calib_pass": True,
              "resp_pass": True, "resp_min_steps": 4,
              "slope_off_oreb_c": 0.9138, "slope_opp_def_dreb_c": 1.087, "fit_s": 31.8,
              "source": "experiments.md section 3, F1 row lgbm/C_plus_state"}
CITED_S0_F2 = {"log_loss": 0.645565}


def get_cuts(scheme: str, te_dates: pd.Series, conf_all: pd.DataFrame, season: int):
    if scheme == "S0":
        return None
    monthly = PO.month_boundaries(te_dates)
    if scheme == "S1_monthly":
        return monthly
    if scheme == "S1_conf_aligned":
        bounds = CF.conference_boundary_dates(conf_all, season)
        return CF.union_boundaries(monthly, bounds)
    if scheme == "S1_weekly":
        weekly = CF.weekly_boundaries(te_dates)
        return CF.union_boundaries(weekly, monthly[:1])
    raise KeyError(scheme)


def estimate_cost_s(cuts) -> float:
    n = 1 if cuts is None else len(cuts)
    pad = 1.0 if n <= 1 else 1.2
    return n * FIT_S_MEASURED * pad


def fit_predict_scheme(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str],
                       cuts, seed: int = 0):
    tr_dates = pd.to_datetime(tr["game_date"])
    if cuts is None:
        model = RB.fit_arm(ARM, tr, feats, seed=seed)
        p = RB.predict_arm(ARM, model, te, feats)
        seg = [{"refit_date": str(tr_dates.max().date()), "model": model,
               "max_train_date": str(tr_dates.max().date()),
               "n_train": int(len(tr)), "n_scored": int(len(te))}]
        return p, seg
    te_dates = pd.to_datetime(te["game_date"])
    fit_cols = [*feats, "y"]
    p = np.zeros((len(te), len(RB.CLASSES)), dtype="float64")
    segments = []
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = ((te_dates >= cut) if nxt is None
               else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not seg.any():
            continue
        before = (te_dates < cut).to_numpy()
        prior = te.loc[before, fit_cols]
        rows = tr[fit_cols] if not len(prior) else pd.concat([tr[fit_cols], prior], ignore_index=True)
        model = RB.fit_arm(ARM, rows, feats, seed=seed)
        p[seg] = RB.predict_arm(ARM, model, te.loc[seg], feats)
        max_train = (tr_dates.max() if not before.any()
                    else max(tr_dates.max(), te_dates[before].max()))
        segments.append({"refit_date": str(pd.Timestamp(cut).date()), "model": model,
                         "max_train_date": str(pd.Timestamp(max_train).date()),
                         "n_train": int(len(rows)), "n_scored": int(seg.sum())})
    if (p.sum(axis=1) == 0).any():
        raise AssertionError("refit calendar is not a cover of the test slice")
    return p, segments


def conf4_calibration(te: pd.DataFrame, p: np.ndarray, firsts: pd.DataFrame,
                      min_n: int = CONF4_MIN_N) -> dict:
    t = te[["season", "off_team_id", "game_date"]].reset_index(drop=True).copy()
    t["game_date"] = pd.to_datetime(t["game_date"])
    t = t.merge(firsts.rename(columns={"team_id": "off_team_id"}),
               on=["season", "off_team_id"], how="left")
    wk = (t["game_date"] - t["first_conf_date"]).dt.days / 7.0
    mask = ((wk >= 0) & (wk < 4)).fillna(False).to_numpy()
    n = int(mask.sum())
    if n < min_n:
        return {"n": n, "worst_gap_pp": None, "underpowered": True}
    y = te["y"].to_numpy()
    calib = PM.decile_calibration(y[mask], p[mask], RB.CLASSES)
    ok, worst, who = PM.calibration_verdict(calib)
    return {"n": n, "worst_gap_pp": worst, "worst_class": who,
           "gate_pass": bool(ok), "underpowered": False}


def write_manifest(scheme: str, fold: str, segments: list[dict], season: int) -> Path:
    d = S1_DIR / scheme / fold
    d.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for i, s in enumerate(segments):
        fname = f"seg_{i:02d}_{s['refit_date']}.joblib"
        joblib.dump({"arm": ARM, "feature_set": FS, "features": RB.feature_set(FS),
                    "model": s["model"], "scheme": scheme, "fold": fold,
                    "refit_date": s["refit_date"], "max_train_date": s["max_train_date"],
                    "note": "rebound S1 scheme confirmation, experiments.md section 7"},
                   d / fname)
        artifacts.append({"refit_date": s["refit_date"], "path": fname,
                          "max_train_date": s["max_train_date"], "n_train": s["n_train"]})
    manifest = {"model": "rebound", "scheme": scheme, "fold": fold, "season": season,
               "key": FS, "artifacts": artifacts}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return d / "manifest.json"


def run_cell(scheme: str, fold: str, tr: pd.DataFrame, te: pd.DataFrame, feats: list[str],
            conf_all: pd.DataFrame, firsts: pd.DataFrame, season: int, seed: int = 0) -> dict:
    te_dates = pd.to_datetime(te["game_date"])
    cuts = get_cuts(scheme, te_dates, conf_all, season)
    t0 = time.time()
    p, segments = fit_predict_scheme(tr, te, feats, cuts, seed=seed)
    fit_s = round(time.time() - t0, 1)
    s = RB.score(te, p)
    conf4 = conf4_calibration(te, p, firsts)
    manifest_path = write_manifest(scheme, fold, segments, season) if seed == 0 else None
    row = {
        "scheme": scheme, "fold": fold, "seed": seed, "n_fits": len(segments),
        "n_test": int(len(te)), "fit_seconds": fit_s,
        "log_loss": round(s["log_loss"], 6), "calib_pass": bool(s["calib_pass"]),
        "calib_worst_gap_pp": s["calib_worst_gap_pp"], "calib_worst_class": s["calib_worst_class"],
        "resp_pass": bool(s["resp_pass"]), "resp_min_steps": s["resp_min_steps"],
        "slope_off_oreb_c": s["responsiveness"]["off_oreb_c->OREB"]["slope_ratio"],
        "slope_opp_def_dreb_c": s["responsiveness"]["opp_def_dreb_c->OREB"]["slope_ratio"],
        "conf4": conf4,
        "manifest_path": str(manifest_path) if manifest_path else None,
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--budget-hours", type=float, default=BUDGET_S / 3600)
    args = ap.parse_args()
    budget_s = args.budget_hours * 3600

    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="rebound S1 confirmation")

    t_start = time.time()
    universe = ES.load_universe()
    ev = pd.read_parquet(OUT_DIR / "events_v1.parquet")
    ev.attrs["rim_override_max_ft"] = ES.rim_override_for_version("v1")
    ev.attrs["possessions_version"] = "v1"
    design = RB.build_design(SEASONS, universe=universe, version="v1", events=ev)
    feats = RB.feature_set(FS)
    conf_all = CF.build_conference_flags(SEASONS)
    firsts = CF.first_conference_game_dates(conf_all)
    print(f"design ready {time.time() - t_start:.1f}s; conf flags "
          f"{conf_all.attrs.get('n_missing_conference_id')} missing", flush=True)

    tr2, te2 = RB.fold_slices(design, "F2")
    tr1, te1 = RB.fold_slices(design, "F1")

    plan = [
        ("F2", "S0", 0), ("F2", "S0", 1),
        ("F2", "S1_monthly", 0), ("F1", "S1_monthly", 0),
        ("F2", "S1_conf_aligned", 0),
        ("F2", "S1_weekly", 0),
    ]
    fold_data = {"F2": (tr2, te2, 2025), "F1": (tr1, te1, 2024)}

    results: dict[str, dict] = {}
    not_run: list[str] = []
    for fold, scheme, seed in plan:
        key = f"{fold}|{scheme}|s{seed}"
        tr, te, season = fold_data[fold]
        cuts_preview = get_cuts(scheme, pd.to_datetime(te["game_date"]), conf_all, season)
        cost = estimate_cost_s(cuts_preview)
        elapsed = time.time() - t_start
        if elapsed + cost > budget_s:
            print(f"SKIP {key}: projected {cost:.0f}s would exceed budget "
                  f"({elapsed:.0f}s used of {budget_s:.0f}s) -- NOT RUN", flush=True)
            not_run.append(key)
            continue
        row = run_cell(scheme, fold, tr, te, feats, conf_all, firsts, season, seed=seed)
        results[key] = row
        print(f"{key}: ll={row['log_loss']:.6f} calib={'PASS' if row['calib_pass'] else 'FAIL'}"
              f"({row['calib_worst_gap_pp']}) resp={'PASS' if row['resp_pass'] else 'FAIL'} "
              f"conf4_n={row['conf4']['n']} conf4_gap={row['conf4']['worst_gap_pp']} "
              f"n_fits={row['n_fits']} {row['fit_seconds']}s", flush=True)

    # -- reproduction check --------------------------------------------------
    s0f2 = results.get("F2|S0|s0")
    repro_ok = s0f2 is not None and abs(s0f2["log_loss"] - CITED_S0_F2["log_loss"]) < 1e-6

    # -- noise floor ----------------------------------------------------------
    s0f2_seed1 = results.get("F2|S0|s1")
    primary_floor = (abs(s0f2["log_loss"] - s0f2_seed1["log_loss"])
                     if s0f2 and s0f2_seed1 else None)
    conf4_seed_spread = None
    if (s0f2 and s0f2_seed1 and s0f2["conf4"]["worst_gap_pp"] is not None
            and s0f2_seed1["conf4"]["worst_gap_pp"] is not None):
        conf4_seed_spread = round(abs(s0f2["conf4"]["worst_gap_pp"]
                                      - s0f2_seed1["conf4"]["worst_gap_pp"]), 3)

    # -- decision -------------------------------------------------------------
    ladder = ["S1_monthly", "S1_conf_aligned", "S1_weekly"]
    beaters = []
    if s0f2 and primary_floor is not None:
        for scheme in ladder:
            r = results.get(f"F2|{scheme}|s0")
            if r is None:
                continue
            ll_gain = s0f2["log_loss"] - r["log_loss"]
            c4_gain = (None if (r["conf4"]["worst_gap_pp"] is None
                                or s0f2["conf4"]["worst_gap_pp"] is None)
                      else s0f2["conf4"]["worst_gap_pp"] - r["conf4"]["worst_gap_pp"])
            gates_ok = r["calib_pass"] and r["resp_pass"]
            beats = gates_ok and (ll_gain > primary_floor
                                  or (c4_gain is not None and c4_gain > CONF4_FLOOR_PP))
            beaters.append({"scheme": scheme, "log_loss": r["log_loss"], "ll_gain": round(ll_gain, 6),
                           "conf4_gain_pp": (round(c4_gain, 3) if c4_gain is not None else None),
                           "gates_ok": gates_ok, "beats_reference": beats})
    winning = [b for b in beaters if b["beats_reference"]]
    if winning:
        best = min(winning, key=lambda b: b["log_loss"])
        within_floor = [b for b in winning if b["log_loss"] - best["log_loss"] <= (primary_floor or 0)]
        # simplest by ladder order among those within the floor of the best beater
        chosen = next(b for b in ladder if any(w["scheme"] == b for w in within_floor))
        decision = {"winner": chosen, "reason": "beats the S0 reference beyond the floor; "
                   "simplest arm within the floor of the best beater", "beaters": beaters}
    else:
        decision = {"winner": "S0", "reason": "nothing beats the S0 reference beyond the floor "
                   "or the conf4 threshold; cadence remains PENDING EVIDENCE for this sub-model",
                   "beaters": beaters}

    report = {
        "created_at": pd.Timestamp.now("UTC").isoformat(), "arm": ARM, "feature_set": FS,
        "s0_f2_reproduction_ok": bool(repro_ok),
        "cited_s0_f1": CITED_S0_F1, "cited_s0_f2": CITED_S0_F2,
        "primary_metric_floor": (round(primary_floor, 6) if primary_floor is not None else None),
        "conf4_floor_pp_fixed": CONF4_FLOOR_PP, "conf4_seed_spread_pp": conf4_seed_spread,
        "cells": {k: {kk: vv for kk, vv in v.items() if kk != "manifest_path_obj"}
                 for k, v in results.items()},
        "not_run": not_run, "decision": decision,
        "runtime_min": round((time.time() - t_start) / 60, 1),
    }
    (OUT_DIR / "s1_confirm_run_report.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8")

    section = render_section(report)
    if args.no_append:
        print(section)
    else:
        with DOC.open("a", encoding="utf-8") as fh:
            fh.write("\n" + section)
        print(f"\nappended results to {DOC}")
    print(f"\ndone in {report['runtime_min']} min; winner={decision['winner']}; "
         f"not_run={not_run}")
    return 0


def _table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def render_section(report: dict) -> str:
    L: list[str] = []
    L.append(f"## 8. S1 scheme confirmation: results (run {report['created_at']}, "
             "`scripts/train_rebound_v2_s1.py`)\n")
    L.append(f"S0/F2 reproduction of round 1's adopted number: "
             f"{'CONFIRMED' if report['s0_f2_reproduction_ok'] else '**MISMATCH -- SEE BELOW**'} "
             f"(cited {report['cited_s0_f2']['log_loss']}).\n")
    rows = []
    for key, r in report["cells"].items():
        rows.append({
            "cell": key, "n_fits": r["n_fits"], "log_loss": r["log_loss"],
            "calib": "PASS" if r["calib_pass"] else "FAIL", "worst_gap_pp": r["calib_worst_gap_pp"],
            "respons": "PASS" if r["resp_pass"] else "FAIL",
            "slope_off_oreb_c": r["slope_off_oreb_c"], "slope_opp_def_dreb_c": r["slope_opp_def_dreb_c"],
            "conf4_n": r["conf4"]["n"],
            "conf4_gap_pp": ("underpowered" if r["conf4"]["underpowered"] else r["conf4"]["worst_gap_pp"]),
            "fit_s": r["fit_seconds"],
        })
    L.append(_table(rows, ["cell", "n_fits", "log_loss", "calib", "worst_gap_pp", "respons",
                           "slope_off_oreb_c", "slope_opp_def_dreb_c", "conf4_n", "conf4_gap_pp",
                           "fit_s"]))
    L.append("")
    L.append(f"S0 F1 (cited, not refit, section 3): log loss {report['cited_s0_f1']['log_loss']}, "
             f"calib gap {report['cited_s0_f1']['calib_worst_gap_pp']} pp "
             f"({'PASS' if report['cited_s0_f1']['calib_pass'] else 'FAIL'}), slopes "
             f"{report['cited_s0_f1']['slope_off_oreb_c']} / "
             f"{report['cited_s0_f1']['slope_opp_def_dreb_c']}.\n")
    L.append(f"Noise floor: primary-metric floor (S0 second-seed refit) = "
             f"**{report['primary_metric_floor']}**. conf4 seed spread (context only; the decision "
             f"rule uses the fixed 0.25 pp threshold) = {report['conf4_seed_spread_pp']} pp.\n")
    if report["not_run"]:
        L.append(f"**NOT RUN** (wall-clock budget): {', '.join(report['not_run'])}.\n")
    L.append("### 8.1 Decision\n")
    d = report["decision"]
    for b in d["beaters"]:
        L.append(f"- `{b['scheme']}`: log loss {b['log_loss']} (gain {b['ll_gain']:+.6f} vs S0), "
                 f"conf4 gain {b['conf4_gain_pp']} pp, gates "
                 f"{'PASS' if b['gates_ok'] else 'FAIL'}, beats reference: {b['beats_reference']}")
    L.append("")
    L.append(f"**WINNER: {d['winner']}** -- {d['reason']}\n")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
