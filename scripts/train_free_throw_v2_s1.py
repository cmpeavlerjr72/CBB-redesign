#!/usr/bin/env python
"""
train_free_throw_v2_s1.py -- L3 FREE THROW (FT-2) S1 SCHEME CONFIRMATION:
which refit calendar (Decision 9c), model class and features held fixed.

    .venv/Scripts/python.exe scripts/train_free_throw_v2_s1.py
    .venv/Scripts/python.exe scripts/train_free_throw_v2_s1.py --no-append

Pre-registration: `docs/models/free_throw/experiments.md` section 7, written
and COMMITTED before this ran. Executes it and APPENDS results; never edits
what is already there. FT-1 (trip structure) is untouched -- a rule, not a
fit -- and this script does not read or write `bonus_era.json`.

Model class and feature bundle are HELD FIXED at round 1's FT-2 winner,
`lgbm` on `FT_FEATURES`. Four refit calendars are compared: `S0` (static, the
reference -- must reproduce round 1's F2 log loss 0.575281), `S1_monthly`,
`S1_conf_aligned`, `S1_weekly`. Refit-calendar construction is imported from
`scripts/train_possession_outcome_v3.py` / `cbb_sim.features.conference` /
`cbb_sim.models.possession_outcome.month_boundaries`, matching the rebound
S1 confirmation exactly.

THREAD CAP and WALL CLOCK, same convention as the rebound script: `n_jobs=3`
monkeypatched onto `FT.LgbmArm.PARAMS`, thread env vars pinned to 3. Section
7.4's measured 51.0 s/fit projects ~65 min for the full stage list; hard cap
here is 2.0 h, checked before each cell starts.

Artifacts (gitignored, HF-synced, never `git add`ed):
    data/processed/models/free_throw/s1_confirm/<scheme>/<fold>/*.joblib
    data/processed/models/free_throw/s1_confirm/<scheme>/<fold>/manifest.json
    data/processed/models/free_throw/s1_confirm_run_report.json
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
from cbb_sim.models import free_throw as FT  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402

FT.LgbmArm.PARAMS = {**FT.LgbmArm.PARAMS, "n_jobs": 3}

OUT_DIR = Path("data/processed/models/free_throw")
S1_DIR = OUT_DIR / "s1_confirm"
DOC = Path("docs/models/free_throw/experiments.md")
SEASONS = [2022, 2023, 2024, 2025]
SCHEMES = ("S0", "S1_monthly", "S1_conf_aligned", "S1_weekly")
FIT_S_MEASURED = 51.0           # section 7.4
BUDGET_S = 2.0 * 3600           # section 7.4
CONF4_MIN_N = 1000              # section 7.3
CONF4_FLOOR_PP = 0.25           # section 7.4 / 7.5
CITED_S0_F1 = {"log_loss": 0.578489, "calib_worst_gap_pp": 2.812, "calib_pass": False,
              "resp_pass": True, "resp_min_steps": 4, "slope_ratio": 0.9962, "fit_s": 10.5,
              "source": "experiments.md section 4, F1 row lgbm"}
CITED_S0_F2 = {"log_loss": 0.575281}


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


def fit_predict_scheme(tr: pd.DataFrame, te: pd.DataFrame, cuts, seed: int = 0):
    tr_dates = pd.to_datetime(tr["game_date"])
    if cuts is None:
        m = FT.LgbmArm(seed=seed).fit(FT.design_matrix(tr), tr["y"].to_numpy())
        p = m.predict_proba(FT.design_matrix(te))
        seg = [{"refit_date": str(tr_dates.max().date()), "model": m,
               "max_train_date": str(tr_dates.max().date()),
               "n_train": int(len(tr)), "n_scored": int(len(te))}]
        return p, seg
    te_dates = pd.to_datetime(te["game_date"])
    fit_cols = [*FT.FT_FEATURES, "y"]
    p = np.zeros((len(te), len(FT.CLASSES)), dtype="float64")
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
        m = FT.LgbmArm(seed=seed).fit(FT.design_matrix(rows), rows["y"].to_numpy())
        p[seg] = m.predict_proba(FT.design_matrix(te.loc[seg]))
        max_train = (tr_dates.max() if not before.any()
                    else max(tr_dates.max(), te_dates[before].max()))
        segments.append({"refit_date": str(pd.Timestamp(cut).date()), "model": m,
                         "max_train_date": str(pd.Timestamp(max_train).date()),
                         "n_train": int(len(rows)), "n_scored": int(seg.sum())})
    if (p.sum(axis=1) == 0).any():
        raise AssertionError("refit calendar is not a cover of the test slice")
    return p, segments


def conf4_calibration(te: pd.DataFrame, p: np.ndarray, firsts: pd.DataFrame,
                      min_n: int = CONF4_MIN_N) -> dict:
    t = te[["season", "team_id", "game_date"]].reset_index(drop=True).copy()
    t["game_date"] = pd.to_datetime(t["game_date"])
    t = t.merge(firsts, on=["season", "team_id"], how="left")
    wk = (t["game_date"] - t["first_conf_date"]).dt.days / 7.0
    mask = ((wk >= 0) & (wk < 4)).fillna(False).to_numpy()
    n = int(mask.sum())
    if n < min_n:
        return {"n": n, "worst_gap_pp": None, "underpowered": True}
    y = te["y"].to_numpy()
    calib = PM.decile_calibration(y[mask], p[mask], FT.CLASSES)
    ok, worst, who = PM.calibration_verdict(calib)
    return {"n": n, "worst_gap_pp": worst, "worst_class": who,
           "gate_pass": bool(ok), "underpowered": False}


def clean_trip_calibration(te: pd.DataFrame, p: np.ndarray) -> dict:
    """Calibration on CLEAN trips only -- technical already excluded from the
    universe; ambiguous = a two-attempt trip whose foul_class is
    bonus_one_and_one or double_bonus (section 3.3's own ambiguous-mass
    definition, L24). Reported, never gated."""
    ambiguous = (te["foul_class"].isin(["bonus_one_and_one", "double_bonus"]).to_numpy()
                & (te["trip_len"].to_numpy() == 2))
    clean = ~ambiguous
    y = te["y"].to_numpy()
    calib = PM.decile_calibration(y[clean], p[clean], FT.CLASSES)
    ok, worst, who = PM.calibration_verdict(calib)
    return {"n_clean": int(clean.sum()), "n_ambiguous": int(ambiguous.sum()),
           "ambiguous_share_pct": round(float(ambiguous.mean() * 100), 3),
           "clean_worst_gap_pp": worst, "clean_gate_pass": bool(ok)}


def write_manifest(scheme: str, fold: str, segments: list[dict], season: int) -> Path:
    d = S1_DIR / scheme / fold
    d.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for i, s in enumerate(segments):
        fname = f"seg_{i:02d}_{s['refit_date']}.joblib"
        joblib.dump({"arm": "lgbm", "features": list(FT.FT_FEATURES), "model": s["model"],
                    "scheme": scheme, "fold": fold, "refit_date": s["refit_date"],
                    "max_train_date": s["max_train_date"],
                    "note": "free_throw S1 scheme confirmation, experiments.md section 7"},
                   d / fname)
        artifacts.append({"refit_date": s["refit_date"], "path": fname,
                          "max_train_date": s["max_train_date"], "n_train": s["n_train"]})
    manifest = {"model": "free_throw", "scheme": scheme, "fold": fold, "season": season,
               "key": "FT2_lgbm", "artifacts": artifacts}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return d / "manifest.json"


def run_cell(scheme: str, fold: str, tr: pd.DataFrame, te: pd.DataFrame,
            conf_all: pd.DataFrame, firsts: pd.DataFrame, season: int, seed: int = 0) -> dict:
    te_dates = pd.to_datetime(te["game_date"])
    cuts = get_cuts(scheme, te_dates, conf_all, season)
    t0 = time.time()
    p, segments = fit_predict_scheme(tr, te, cuts, seed=seed)
    fit_s = round(time.time() - t0, 1)
    s = FT.score(te, p)
    conf4 = conf4_calibration(te, p, firsts)
    clean = clean_trip_calibration(te, p)
    manifest_path = write_manifest(scheme, fold, segments, season) if seed == 0 else None
    row = {
        "scheme": scheme, "fold": fold, "seed": seed, "n_fits": len(segments),
        "n_test": int(len(te)), "fit_seconds": fit_s,
        "log_loss": round(s["log_loss"], 6), "calib_pass": bool(s["calib_pass"]),
        "calib_worst_gap_pp": s["calib_worst_gap_pp"],
        "resp_pass": bool(s["resp_pass"]), "resp_min_steps": s["resp_min_steps"],
        "slope_ratio": s["responsiveness"]["shooter_ft_asof->MAKE"]["slope_ratio"],
        "conf4": conf4, "clean_trip": clean,
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
    assert_not_sealed(SEASONS, context="free_throw S1 confirmation")

    t_start = time.time()
    ES.load_universe()
    att_all = pd.read_parquet(OUT_DIR / "attempts_v1_era.parquet")
    attempts = att_all[att_all["season"].isin(SEASONS)]
    design = FT.build_ft_design(attempts)
    conf_all = CF.build_conference_flags(SEASONS)
    firsts = CF.first_conference_game_dates(conf_all)
    print(f"design ready {time.time() - t_start:.1f}s", flush=True)

    tr2, te2 = FT.fold_slices(design, "F2")
    tr1, te1 = FT.fold_slices(design, "F1")

    # -- descriptive: technical + ambiguous share of the TRAINING pool (2022-2024) --
    train_pool_all = att_all[att_all["season"].isin([2022, 2023, 2024])]
    technical_share = round(float(
        (train_pool_all["foul_class"] == "technical").mean() * 100), 3)
    ambiguous_share = round(float(
        (train_pool_all["foul_class"].isin(["bonus_one_and_one", "double_bonus"])
         & (train_pool_all["trip_len"] == 2)).mean() * 100), 3)

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
        row = run_cell(scheme, fold, tr, te, conf_all, firsts, season, seed=seed)
        results[key] = row
        print(f"{key}: ll={row['log_loss']:.6f} calib={'PASS' if row['calib_pass'] else 'FAIL'}"
              f"({row['calib_worst_gap_pp']}) resp={'PASS' if row['resp_pass'] else 'FAIL'} "
              f"conf4_n={row['conf4']['n']} conf4_gap={row['conf4']['worst_gap_pp']} "
              f"clean_gap={row['clean_trip']['clean_worst_gap_pp']} "
              f"n_fits={row['n_fits']} {row['fit_seconds']}s", flush=True)

    s0f2 = results.get("F2|S0|s0")
    repro_ok = s0f2 is not None and abs(s0f2["log_loss"] - CITED_S0_F2["log_loss"]) < 1e-6

    s0f2_seed1 = results.get("F2|S0|s1")
    primary_floor = (abs(s0f2["log_loss"] - s0f2_seed1["log_loss"])
                     if s0f2 and s0f2_seed1 else None)
    conf4_seed_spread = None
    if (s0f2 and s0f2_seed1 and s0f2["conf4"]["worst_gap_pp"] is not None
            and s0f2_seed1["conf4"]["worst_gap_pp"] is not None):
        conf4_seed_spread = round(abs(s0f2["conf4"]["worst_gap_pp"]
                                      - s0f2_seed1["conf4"]["worst_gap_pp"]), 3)

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
        chosen = next(b for b in ladder if any(w["scheme"] == b for w in within_floor))
        decision = {"winner": chosen, "reason": "beats the S0 reference beyond the floor; "
                   "simplest arm within the floor of the best beater", "beaters": beaters}
    else:
        decision = {"winner": "S0", "reason": "nothing beats the S0 reference beyond the floor "
                   "or the conf4 threshold; cadence remains PENDING EVIDENCE for this sub-model",
                   "beaters": beaters}

    report = {
        "created_at": pd.Timestamp.now("UTC").isoformat(), "arm": "lgbm",
        "s0_f2_reproduction_ok": bool(repro_ok),
        "cited_s0_f1": CITED_S0_F1, "cited_s0_f2": CITED_S0_F2,
        "primary_metric_floor": (round(primary_floor, 6) if primary_floor is not None else None),
        "conf4_floor_pp_fixed": CONF4_FLOOR_PP, "conf4_seed_spread_pp": conf4_seed_spread,
        "l24_segment": {"training_pool_seasons": [2022, 2023, 2024],
                       "technical_share_pct": technical_share,
                       "ft_trip_ambiguous_share_pct": ambiguous_share},
        "cells": results, "not_run": not_run, "decision": decision,
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
             "`scripts/train_free_throw_v2_s1.py`)\n")
    L.append(f"S0/F2 reproduction of round 1's adopted number: "
             f"{'CONFIRMED' if report['s0_f2_reproduction_ok'] else '**MISMATCH -- SEE BELOW**'} "
             f"(cited {report['cited_s0_f2']['log_loss']}).\n")
    l24 = report["l24_segment"]
    L.append(f"L24 segment (training pool {l24['training_pool_seasons']}): technical attempts "
             f"{l24['technical_share_pct']}% of all attempts (excluded from every FT-2 arm's "
             f"universe already); `ft_trip_ambiguous`-equivalent attempts (two-attempt trips whose "
             f"foul class is bonus_one_and_one or double_bonus) {l24['ft_trip_ambiguous_share_pct']}% "
             "-- both reported segments, neither a gate.\n")
    rows = []
    for key, r in report["cells"].items():
        rows.append({
            "cell": key, "n_fits": r["n_fits"], "log_loss": r["log_loss"],
            "calib": "PASS" if r["calib_pass"] else "FAIL", "worst_gap_pp": r["calib_worst_gap_pp"],
            "respons": "PASS" if r["resp_pass"] else "FAIL", "slope_ratio": r["slope_ratio"],
            "conf4_n": r["conf4"]["n"],
            "conf4_gap_pp": ("underpowered" if r["conf4"]["underpowered"] else r["conf4"]["worst_gap_pp"]),
            "clean_gap_pp": r["clean_trip"]["clean_worst_gap_pp"],
            "ambiguous_pct": r["clean_trip"]["ambiguous_share_pct"],
            "fit_s": r["fit_seconds"],
        })
    L.append(_table(rows, ["cell", "n_fits", "log_loss", "calib", "worst_gap_pp", "respons",
                           "slope_ratio", "conf4_n", "conf4_gap_pp", "clean_gap_pp",
                           "ambiguous_pct", "fit_s"]))
    L.append("")
    L.append(f"S0 F1 (cited, not refit, section 4): log loss {report['cited_s0_f1']['log_loss']}, "
             f"calib gap {report['cited_s0_f1']['calib_worst_gap_pp']} pp "
             f"({'PASS' if report['cited_s0_f1']['calib_pass'] else 'FAIL'}), slope "
             f"{report['cited_s0_f1']['slope_ratio']}.\n")
    L.append(f"Noise floor: primary-metric floor (S0 second-seed refit) = "
             f"**{report['primary_metric_floor']}**. conf4 seed spread (context only) = "
             f"{report['conf4_seed_spread_pp']} pp.\n")
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
