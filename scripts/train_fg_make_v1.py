#!/usr/bin/env python
"""
train_fg_make_v1.py -- run the pre-registered L3 FIELD-GOAL MAKE bake-off.

    .venv/Scripts/python.exe scripts/train_fg_make_v1.py
    .venv/Scripts/python.exe scripts/train_fg_make_v1.py --version v2 --no-append

The pre-registration is section 1 of `docs/models/fg_make/experiments.md` and
was written before any of this ran. This script executes it and APPENDS its
results to that file; it never edits what is already there.

THREE MODELS, NOT ONE. Every fit in this script is on one shot class's rows
alone (`fg_make.fit_by_class` / `class_slice`), every metric is reported per
class, and the decision rule is applied per class. The only place the three are
combined is the implied-eFG% check, which is a GATE (G4) on the three of them
together and not a fourth model.

`--version` selects the possessions build. Unlike the rebound and free-throw
models, this one genuinely depends on it: the rim-location override (L16) is
what moves ESPN's 2025 putback mistag out of `FGA_jump2`, so the target classes
themselves change. The pre-registration names v2 and the run report records the
threshold that build used.

Artifacts under `data/processed/models/fg_make/`:
    events_{version}.parquet    one row per field-goal attempt (cached)
    grid_results.csv            every (class, fold, arm) row
    run_report.json             everything the markdown section renders from
    winner_{class}.joblib       the selected arm per class, if one passes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402

OUT_DIR = Path("data/processed/models/fg_make")
DOC = Path("docs/models/fg_make/experiments.md")
SEASONS = [2022, 2023, 2024, 2025]
SEED_GRID = (0, 1, 2, 3, 4)
MAKE = FG.CLASS_INDEX["MAKE"]


# ---------------------------------------------------------------------------
# Build / cache
# ---------------------------------------------------------------------------
def build_or_load_events(version: str, rebuild: bool, universe: pd.DataFrame) -> pd.DataFrame:
    path = OUT_DIR / f"events_{version}.parquet"
    if path.exists() and not rebuild:
        ev = pd.read_parquet(path)
        ev.attrs["rim_override_max_ft"] = ES.rim_override_for_version(version)
        ev.attrs["possessions_version"] = version
        print(f"loaded cached attempts: {path} ({len(ev):,} rows)", flush=True)
        return ev
    t0 = time.time()
    ev = FG.build_fg_events(SEASONS, universe=universe, version=version)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(path, index=False)
    print(f"built {len(ev):,} attempts in {time.time() - t0:.1f}s -> {path}", flush=True)
    return ev


def run_arm(arm: str, tr: pd.DataFrame, te: pd.DataFrame, eb_best: dict | None = None,
            seed: int = 0, params: dict | None = None,
            feature_set_name: str | None = None) -> tuple[dict, np.ndarray]:
    t0 = time.time()
    model = FG.fit_arm(arm, tr, feature_set_name, seed=seed, eb_best=eb_best, params=params)
    p = FG.predict_arm(arm, model, te, feature_set_name)
    s = FG.score(te, p)
    s["arm"] = arm
    s["feature_set"] = feature_set_name or FG.ARM_FEATURE_SET[arm] or "eb"
    s["fit_s"] = round(time.time() - t0, 1)
    return s, p


def flat_row(s: dict, fold: str, shot_class: str) -> dict:
    gated = {c: v for c, v in s["calibration"].items()
             if v["share_pct"] >= PM.CALIB_MIN_SHARE * 100}
    level = max((abs(v["level_shift_pp"]) for v in gated.values()), default=0.0)
    shape = max((v["max_abs_gap_pp_after_level_shift"] for v in gated.values()), default=0.0)
    row = {
        "shot_class": shot_class, "fold": fold, "arm": s["arm"],
        "feature_set": s["feature_set"], "n": s["n"],
        "log_loss": round(s["log_loss"], 6), "brier": round(s["brier"], 6),
        "pred_make_pct": round(s["pred_make_rate"] * 100, 3),
        "actual_make_pct": round(s["actual_make_rate"] * 100, 3),
        "calib": "PASS" if s["calib_pass"] else "FAIL",
        "worst_gap_pp": s["calib_worst_gap_pp"],
        "level_pp": round(level, 3), "shape_pp": round(shape, 3),
        "respons": "PASS" if s["resp_pass"] else "FAIL",
        "resp_min_steps": s["resp_min_steps"],
        "fit_s": s["fit_s"],
    }
    for k, v in s["responsiveness"].items():
        tag = k.split("->")[0]
        row[f"steps_{tag}"] = v["pred_monotone_steps"]
        row[f"slope_{tag}"] = v["slope_ratio"]
    for k, v in s["by_chance"].items():
        row[f"chancegap_{k}"] = v["max_abs_gap_pp"]
    return row


# ---------------------------------------------------------------------------
# Blocks of the run
# ---------------------------------------------------------------------------
def descriptives(events: pd.DataFrame, design: pd.DataFrame) -> dict:
    out: dict = {"by_season": {}, "by_class": {}}
    for s in SEASONS:
        e = events[events["season"] == s]
        d = design[design["season"] == s]
        row = {"n_attempts": int(len(e)), "n_games": int(e["game_id"].nunique()),
               "attempts_per_game": round(len(e) / max(e["game_id"].nunique(), 1), 2),
               "espn_id_coverage_pct": round(float(d["espn_athlete_id"].notna().mean() * 100), 2),
               "position_coverage_pct": round(float((d["position_group"] != "UNK").mean() * 100), 2),
               "on_floor_complete_pct": round(float(np.isfinite(
                   e[list(ES.ON_FLOOR_COLS)].to_numpy()).all(axis=1).mean() * 100), 2)}
        for c in FG.SHOT_CLASSES:
            m = e["shot_class"] == c
            row[f"{c}_share_pct"] = round(float(m.mean() * 100), 3)
            row[f"{c}_make_pct"] = round(float(e.loc[m, "made"].mean() * 100), 3)
        row["efg_pct"] = round(float(
            (e["made"].to_numpy() * (1.0 + 0.5 * (e["class_key"].to_numpy() == "three"))).sum()
            / max(len(e), 1) * 100), 3)
        out["by_season"][str(s)] = row
    for c in FG.SHOT_CLASSES:
        e = events[events["shot_class"] == c]
        d = design[design["shot_class"] == c]
        out["by_class"][c] = {
            "n": int(len(e)),
            "share_of_attempts_pct": round(float(len(e) / max(len(events), 1) * 100), 3),
            "make_pct": round(float(e["made"].mean() * 100), 3),
            "blocked_pct": round(float(e["blocked"].mean() * 100), 3),
            "and_one_pct_of_makes": round(float(
                e.loc[e["made"], "and_one"].mean() * 100), 3),
            "continuation_pct": round(float((e["chance_number"] > 1).mean() * 100), 3),
            "transition_pct": round(float(e["is_transition"].mean() * 100), 3),
            "shooter_no_prior_attempt_pct": round(float((d["shooter_att_c"] == 0).mean() * 100), 3),
            "n_shooters": int(d["shooter_id"].nunique()),
        }
    e = events
    out["population_rules"] = {
        "blocked_attempts": int(e["blocked"].sum()),
        "blocked_and_made": int((e["blocked"] & e["made"]).sum()),
        "blocked_and_made_pct_of_blocks": round(float(
            (e["blocked"] & e["made"]).sum() / max(int(e["blocked"].sum()), 1) * 100), 4),
        "and_one_attempts": int(e["and_one"].sum()),
        "and_one_pct_of_attempts": round(float(e["and_one"].mean() * 100), 3),
        "and_one_make_rate": round(float(e.loc[e["and_one"], "made"].mean()), 5),
        "n_no_shooter_dropped": int(design.attrs["n_no_shooter"]),
        "no_shooter_pct": round(float(design.attrs["n_no_shooter"]
                                      / max(design.attrs["n_attempts"], 1) * 100), 4),
    }
    out["chance_state"] = {
        "chance_number_dist": {str(int(k)): int(v) for k, v in
                               events["chance_number"].value_counts().sort_index().head(8).items()},
        "start_reason_share_pct": {k: round(v * 100, 3) for k, v in
                                   events["chance_start_reason"].value_counts(
                                       normalize=True).items()},
        "elapsed_mean_s": round(float(events["chance_elapsed_s"].mean()), 3),
        "elapsed_clipped_pct": round(float(events["chance_elapsed_clipped"].mean() * 100), 4),
        "derived_offence_matches_shooter_pct": round(float(events["chance_side_ok"].mean() * 100), 3),
        "derived_offence_known_pct": round(float(events["chance_side_known"].mean() * 100), 3),
        "transition_pct": round(float(events["is_transition"].mean() * 100), 3),
    }
    return out


def lgbm_ladder_on_F1(design: pd.DataFrame) -> dict:
    """The pre-registration allows a LightGBM parameter search on F1 ONLY. The
    winner per class is frozen here and every later LightGBM fit -- F2, the seed
    refits, the lineup fold -- uses it, so F2 never sees a tuning decision."""
    tr, te = FG.fold_slices(design, "F1")
    out: dict = {"fold": "F1 (train 2022+2023, test 2024)", "by_class": {}}
    best: dict = {}
    for c in FG.SHOT_CLASSES:
        trc, tec = FG.class_slice(tr, c), FG.class_slice(te, c)
        rows = []
        for i, over in enumerate(FG.LGBM_LADDER):
            params = {**FG.LGBM_BASE, **over}
            t0 = time.time()
            m = FG.LgbmArm(seed=0, params=over).fit(
                FG.design_matrix(trc, FG.feature_set("C_plus_state")), trc["y"].to_numpy())
            p = m.predict_proba(FG.design_matrix(tec, FG.feature_set("C_plus_state")))
            ll = PM.log_loss(tec["y"].to_numpy(), p)
            rows.append({"rung": i, "override": json.dumps(over) if over else "(base)",
                         "num_leaves": params["num_leaves"],
                         "min_child_samples": params["min_child_samples"],
                         "n_estimators": params["n_estimators"],
                         "learning_rate": params["learning_rate"],
                         "reg_lambda": params["reg_lambda"],
                         "F1_log_loss": round(ll, 6), "fit_s": round(time.time() - t0, 1)})
            print(f"  ladder {c} rung {i}: ll={ll:.6f} ({rows[-1]['fit_s']}s)", flush=True)
        win = min(rows, key=lambda r: r["F1_log_loss"])
        best[c] = dict(FG.LGBM_LADDER[win["rung"]])
        out["by_class"][c] = {"rows": rows, "winner_rung": win["rung"],
                              "winner_override": win["override"],
                              "winner_F1_log_loss": win["F1_log_loss"]}
    out["frozen_params"] = {c: best[c] for c in FG.SHOT_CLASSES}
    return out, best


def transfer_block(te: pd.DataFrame, preds: dict, shot_class: str) -> dict:
    """The L15 natural experiment: players whose modal team changed since the
    prior season, where a prior-season-based prior is least trustworthy."""
    y = te["y"].to_numpy()
    tsub = te["is_transfer"].to_numpy()
    has_prior = te["has_prior_season"].to_numpy() > 0
    out = {}
    for name, mask in (("transfer", tsub),
                       ("non_transfer", ~tsub & has_prior),
                       ("no_prior_season", ~has_prior)):
        if mask.sum() == 0:
            continue
        sub = {"n": int(mask.sum()), "actual_make_rate": round(float(y[mask].mean()), 5)}
        for arm in FG.ARMS:
            p = preds[(shot_class, arm)]
            sub[f"{arm}_log_loss"] = round(PM.log_loss(y[mask], p[mask]), 6)
        sub["eb_pred_make_rate"] = round(float(preds[(shot_class, "eb_shrink")][mask, MAKE].mean()), 5)
        out[name] = sub
    return out


def lineup_block(design: pd.DataFrame, events: pd.DataFrame, lgbm_params: dict,
                 floors: dict, bootstrap_reps: int) -> dict:
    """C vs D on the lineup fold (train 2024, test 2025), per class, on the rows
    that carry all ten on-floor ids, with the defender-rate shrinkage strength
    FITTED on the train season (L13)."""
    seasons = FG.LINEUP_FOLDS["L2"]["train"] + FG.LINEUP_FOLDS["L2"]["test"]
    ev = events[events["season"].isin(seasons)]
    dd = design[design["season"].isin(seasons)]
    out: dict = {"fold": "L2 (train 2024, test 2025)",
                 "prior_grid": list(FG.LINEUP_PRIOR_GRID), "by_class": {}}

    fits, best_k, best_ll, best_design = [], None, None, None
    for k in FG.LINEUP_PRIOR_GRID:
        t0 = time.time()
        rates = FG.defender_rates(ev, prior_att=k)
        d = FG.attach_lineup_features(dd, rates)
        sub = d[d["lineup_on_floor_ok"]]
        tr, _ = FG.fold_slices(sub, "L2")
        ll_tot, n_tot = 0.0, 0
        for c in FG.SHOT_CLASSES:
            trc = FG.class_slice(tr, c)
            feats = FG.feature_set("D_plus_lineup")
            m = FG.RidgeArm().fit(FG.design_matrix(trc, feats), trc["y"].to_numpy())
            p = m.predict_proba(FG.design_matrix(trc, feats))
            ll_tot += PM.log_loss(trc["y"].to_numpy(), p) * len(trc)
            n_tot += len(trc)
        ll = ll_tot / max(n_tot, 1)
        fits.append({"prior_att": int(k), "train_log_loss": round(ll, 6),
                     "fit_s": round(time.time() - t0, 1)})
        print(f"  lineup shrinkage k={k}: train ll={ll:.6f} ({fits[-1]['fit_s']}s)", flush=True)
        if best_ll is None or ll < best_ll:
            best_k, best_ll, best_design = int(k), ll, d
    out["shrinkage_fit"] = fits
    out["prior_att_fitted"] = best_k

    sub = best_design[best_design["lineup_on_floor_ok"]]
    out["n_rows_on_floor_complete"] = int(len(sub))
    out["on_floor_coverage_pct"] = round(float(len(sub) / max(len(best_design), 1) * 100), 2)
    tr, te = FG.fold_slices(sub, "L2")
    out["n_train"] = int(len(tr))
    out["n_test"] = int(len(te))
    for c in FG.SHOT_CLASSES:
        trc, tec = FG.class_slice(tr, c), FG.class_slice(te, c)
        rows, p_by = [], {}
        for arm in ("ridge", "lgbm"):
            for fs in ("C_plus_state", "D_plus_lineup"):
                s, p = run_arm(arm, trc, tec, params=lgbm_params.get(c),
                               feature_set_name=fs)
                p_by[(arm, fs)] = p
                rows.append({"arm": arm, "feature_set": fs, "n": s["n"],
                             "log_loss": round(s["log_loss"], 6), "brier": round(s["brier"], 6),
                             "calib": "PASS" if s["calib_pass"] else "FAIL",
                             "worst_gap_pp": s["calib_worst_gap_pp"],
                             "respons": "PASS" if s["resp_pass"] else "FAIL",
                             "resp_min_steps": s["resp_min_steps"], "fit_s": s["fit_s"]})
                print(f"  lineup {c} {arm:6s} {fs:14s} ll={s['log_loss']:.6f}", flush=True)
        floor_l2 = PM.block_bootstrap_se(tec["game_id"].to_numpy(), tec["y"].to_numpy(),
                                         p_by[("ridge", "C_plus_state")], n_rep=bootstrap_reps)
        gains = {}
        for arm in ("ridge", "lgbm"):
            cc = next(r for r in rows if r["arm"] == arm and r["feature_set"] == "C_plus_state")
            dd_ = next(r for r in rows if r["arm"] == arm and r["feature_set"] == "D_plus_lineup")
            g = cc["log_loss"] - dd_["log_loss"]
            gains[arm] = {"gain_D_over_C": round(g, 6),
                          "gain_in_floors": round(g / floor_l2, 2) if floor_l2 else None}
        verdict = "LINEUP-LEVEL" if gains["ridge"]["gain_D_over_C"] > floor_l2 else "TEAM-LEVEL"
        out["by_class"][c] = {"rows": rows, "block_bootstrap_se_on_this_fold": round(floor_l2, 6),
                              "gains": gains, "verdict": verdict}
    vs = [v["verdict"] for v in out["by_class"].values()]
    out["verdict_overall"] = ("LINEUP-LEVEL" if all(v == "LINEUP-LEVEL" for v in vs)
                              else "TEAM-LEVEL" if all(v == "TEAM-LEVEL" for v in vs)
                              else "SPLIT -- see per class")
    return out


def winner_mix_only(args) -> int:
    """`--efg-winner-mix-only`: the G4 check on the adopted trio, from the run
    report the main pass already wrote. Separate entry point so the expensive
    grid is not re-run and so `experiments.md` gains one appended section rather
    than a duplicate of the whole results block."""
    rp = OUT_DIR / "run_report.json"
    report = json.loads(rp.read_text())
    ladder = json.loads((OUT_DIR / f"lgbm_ladder_{args.version}.json").read_text())
    lgbm_params = {c: dict(v) for c, v in ladder["frozen_params"].items()}
    universe = ES.load_universe(require_pbp_complete=True)
    events = build_or_load_events(args.version, False, universe)
    design = FG.build_design(SEASONS, universe=universe, version=args.version, events=events)
    mx = efg_winner_mix(design, report, lgbm_params)
    report["efg_gate_winner_mix"] = mx
    rp.write_text(json.dumps(report, indent=1, default=str))
    section = render_winner_mix(mx)
    if args.no_append:
        print(section)
    else:
        with DOC.open("a", encoding="utf-8") as fh:
            fh.write("\n" + section)
        print(f"appended the adopted-trio G4 section to {DOC}", flush=True)
    print(json.dumps({k: v for k, v in mx.items() if k not in ("offense", "defense")}, indent=1))
    for side in ("offense", "defense"):
        if side in mx:
            print(side, {k: v for k, v in mx[side].items() if not k.startswith("terciles")})
    return 0


def efg_winner_mix(design: pd.DataFrame, report: dict, lgbm_params: dict) -> dict:
    """The G4 check on the TRIO THAT WAS ACTUALLY ADOPTED.

    The per-arm table in section 6 holds one arm fixed across all three classes,
    which is the right way to compare arms but is not the engine's model when
    the three classes choose differently. The pre-registration asks for "the
    implied team eFG% ... computed from the three class models", so when the
    winners are mixed the adopted trio gets its own row, refit on F2 train and
    scored on the same F2 test rows."""
    tr_all, te_all = FG.fold_slices(design, "F2")
    mix = {c: report["decision"][c]["winner"] for c in FG.SHOT_CLASSES}
    parts, preds = [], []
    for c in FG.SHOT_CLASSES:
        arm = mix[c]
        if arm is None:
            return {"mix": mix, "computable": False,
                    "note": ("at least one class adopted nothing, so there is no adopted trio to "
                             "compute an implied eFG% from")}
        tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
        model = FG.fit_arm(arm, tr, eb_best=report["eb_fits"][f"F2|{c}"]["best"],
                           params=lgbm_params.get(c))
        preds.append(FG.predict_arm(arm, model, te)[:, MAKE])
        parts.append(te)
    te_cat = pd.concat(parts, ignore_index=True)
    p = np.concatenate(preds)
    return {"mix": mix, "computable": True,
            "offense": FG.efg_table(te_cat, p, side="offense"),
            "defense": FG.efg_table(te_cat, p, side="defense")}


def render_winner_mix(mx: dict) -> str:
    L = ["## 10. G4 on the adopted trio (appended after the decision, same run)\n"]
    L.append("Section 6 holds one arm fixed across all three classes, which is how arms are "
             "compared but is not the engine's model when the classes choose differently. The trio "
             "the decision rule actually adopted is "
             + ", ".join(f"`{c}` -> `{a}`" for c, a in mx["mix"].items())
             + ". Refit on F2 train, scored on the same F2 test rows and the same actual shot "
             "mix:\n")
    if not mx.get("computable"):
        L.append(mx.get("note", "") + "\n")
        return "\n".join(L)
    rows = []
    for side in ("offense", "defense"):
        e = mx[side]
        row = {"side": side, "n_teams": e["n_teams"],
               "actual eFG%": e["overall_actual_efg_pct"],
               "implied eFG%": e["overall_implied_efg_pct"],
               "overall gap pp": e["overall_gap_pp"],
               "team MAE pp": e["team_level_mae_pp"], "team corr": e["team_level_corr"]}
        for label in ("asof", "actual"):
            for r in e[f"terciles_{label}"]:
                row[f"T{r['tercile']} gap pp ({label})"] = r["gap_pp"]
            row[f"worst gap pp ({label})"] = e[f"worst_gap_pp_{label}"]
            row[f"G4 ({label})"] = "PASS" if e[f"pass_{label}"] else "FAIL"
        rows.append(row)
    L.append(_table(rows, list(rows[0].keys())))
    L.append("")
    return "\n".join(L)


def decide(rows: list[dict], floor: float, shot_class: str) -> dict:
    # `ridge(diagnostic)` is a reported feature-block row, NOT a pre-registered
    # arm, and can never win the bake-off.
    f2 = [r for r in rows if r["fold"] == "F2" and r["shot_class"] == shot_class
          and r["arm"] in FG.ARMS]
    passing = [r for r in f2 if r["calib"] == "PASS" and r["respons"] == "PASS"]
    v: dict = {"n_arms": len(f2), "n_passing": len(passing),
               "passing_arms": [r["arm"] for r in passing], "floor": round(floor, 6)}
    if not passing:
        v["winner"] = None
        v["note"] = ("no arm passes both pre-registered gates on F2, so nothing is adopted for "
                     "this class")
        return v
    best = min(passing, key=lambda r: (r["log_loss"], FG.ARM_SIMPLICITY[r["arm"]]))
    v["winner"] = best["arm"]
    v["log_loss"] = best["log_loss"]
    non_tree = [r for r in passing if r["arm"] not in FG.TREE_ARMS]
    best_non_tree = min(non_tree, key=lambda r: r["log_loss"]) if non_tree else None
    if best["arm"] in FG.TREE_ARMS:
        if best_non_tree is None:
            v["tree_clause"] = ("no non-tree arm passes both gates, so the pre-registered 'a tree "
                                "arm must beat the best passing non-tree arm by more than the "
                                "floor' clause has nothing to bind against")
        else:
            gain = best_non_tree["log_loss"] - best["log_loss"]
            v["best_passing_non_tree"] = best_non_tree["arm"]
            v["tree_gain_over_non_tree"] = round(gain, 6)
            v["tree_gain_in_floors"] = round(gain / floor, 2) if floor else None
            if gain <= floor:
                v["winner"] = best_non_tree["arm"]
                v["log_loss"] = best_non_tree["log_loss"]
                v["note"] = ("the tree arm's edge is inside the noise floor, so the "
                             "pre-registered simplicity tie-break selects the simpler arm")
    return v


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=FG.DEFAULT_VERSION)
    ap.add_argument("--rebuild-events", action="store_true")
    ap.add_argument("--rebuild-ladder", action="store_true")
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--bootstrap-reps", type=int, default=200)
    ap.add_argument("--skip-lineup", action="store_true")
    ap.add_argument("--skip-first-chance-check", action="store_true")
    ap.add_argument("--efg-winner-mix-only", action="store_true",
                    help="re-open the last run report, compute G4 on the adopted trio and append "
                         "that section only (the rest of the run is untouched)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.efg_winner_mix_only:
        return winner_mix_only(args)
    t_start = time.time()
    universe = ES.load_universe(require_pbp_complete=True)
    universe_loose = ES.load_universe()
    events = build_or_load_events(args.version, args.rebuild_events, universe)
    t0 = time.time()
    design = FG.build_design(SEASONS, universe=universe, version=args.version, events=events)
    print(f"design: {len(design):,} attempts in {time.time() - t0:.1f}s "
          f"({design.attrs['n_no_shooter']:,} with no shooter id dropped)", flush=True)

    report: dict = {
        "run_at": time.strftime("%Y-%m-%d %H:%M"),
        "possessions_version": args.version,
        "rim_override_max_ft": design.attrs["rim_override_max_ft"],
        "seasons": SEASONS,
        "universe": {
            "n_games_d1_nontruncated_pbp_complete": int(
                universe[universe["season"].isin(SEASONS)].shape[0]),
            "n_games_d1_nontruncated": int(
                universe_loose[universe_loose["season"].isin(SEASONS)].shape[0]),
            "pbp_complete_share_pct": round(float(
                universe[universe["season"].isin(SEASONS)].shape[0]
                / universe_loose[universe_loose["season"].isin(SEASONS)].shape[0] * 100), 2),
        },
        "n_attempts": int(design.attrs["n_attempts"]),
        "n_modelled": int(len(design)),
    }
    report["descriptives"] = descriptives(events, design)

    # ---------------- LightGBM parameter search, F1 only ------------------
    # Cached: it is the most expensive block in the run and its answer is frozen
    # by definition, so a re-run should not re-pay for it.
    ladder_path = OUT_DIR / f"lgbm_ladder_{args.version}.json"
    if ladder_path.exists() and not args.rebuild_ladder:
        ladder = json.loads(ladder_path.read_text())
        lgbm_params = {c: dict(v) for c, v in ladder["frozen_params"].items()}
        print(f"loaded cached F1 ladder: {ladder_path}", flush=True)
    else:
        print("\nLightGBM parameter ladder on F1 only", flush=True)
        ladder, lgbm_params = lgbm_ladder_on_F1(design)
        ladder_path.write_text(json.dumps(ladder, indent=1))
    report["lgbm_ladder"] = ladder

    # ---------------- the grid --------------------------------------------
    rows: list[dict] = []
    detail: dict = {}
    preds: dict = {}
    eb_fits: dict = {}
    for fold in ("F1", "F2"):
        tr_all, te_all = FG.fold_slices(design, fold)
        for c in FG.SHOT_CLASSES:
            tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
            eb = FG.fit_eb(tr)
            eb_fits[f"{fold}|{c}"] = eb
            print(f"\n{fold} {c}: train {len(tr):,} test {len(te):,} | eb best "
                  f"prior={eb['best']['prior']} m={eb['best']['m']} m_def={eb['best']['m_def']}",
                  flush=True)
            for arm in FG.ARMS:
                s, p = run_arm(arm, tr, te, eb_best=eb["best"], params=lgbm_params.get(c))
                detail[f"{fold}|{c}|{arm}"] = s
                if fold == FG.SELECTION_FOLD:
                    preds[(c, arm)] = p
                rows.append(flat_row(s, fold, c))
                print(f"  {arm:14s} ll={s['log_loss']:.6f} brier={s['brier']:.6f} "
                      f"calib={'PASS' if s['calib_pass'] else 'FAIL'}({s['calib_worst_gap_pp']}) "
                      f"resp={'PASS' if s['resp_pass'] else 'FAIL'}({s['resp_min_steps']}) "
                      f"{s['fit_s']}s", flush=True)
            # the feature-block diagnostic: ridge on B, so the shooter block's
            # and the state block's contributions are separable. NOT a decision
            # arm -- the pre-registration's ridge arm is the full bundle.
            s, _ = run_arm("ridge", tr, te, feature_set_name="B_plus_shooter")
            s["arm"] = "ridge"
            r = flat_row(s, fold, c)
            r["arm"] = "ridge(diagnostic)"
            detail[f"{fold}|{c}|ridge|B_plus_shooter"] = s
            rows.append(r)
            print(f"  {'ridge/B (diag)':14s} ll={s['log_loss']:.6f}", flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "grid_results.csv", index=False)
    report["eb_fits"] = {k: v for k, v in eb_fits.items()}

    # ---------------- noise floors, per class ------------------------------
    tr2_all, te2_all = FG.fold_slices(design, "F2")
    floors: dict = {}
    for c in FG.SHOT_CLASSES:
        tr2, te2 = FG.class_slice(tr2_all, c), FG.class_slice(te2_all, c)
        fb = PM.block_bootstrap_se(te2["game_id"].to_numpy(), te2["y"].to_numpy(),
                                   preds[(c, "ridge")], n_rep=args.bootstrap_reps)
        seed_ll = []
        for sd in SEED_GRID:
            s, _ = run_arm("lgbm", tr2, te2, seed=sd, params=lgbm_params.get(c))
            seed_ll.append(s["log_loss"])
        fs = float(np.std(seed_ll, ddof=1))
        floors[c] = {"block_bootstrap_se": round(fb, 6), "block_bootstrap_arm": "ridge",
                     "block_bootstrap_reps": args.bootstrap_reps,
                     "lgbm_seed_sd": round(fs, 6),
                     "lgbm_seed_log_losses": [round(x, 6) for x in seed_ll],
                     "floor": round(max(fb, fs), 6)}
        print(f"floor {c}: bootstrap {fb:.6f}, lgbm seed SD {fs:.6f} -> {max(fb, fs):.6f}",
              flush=True)
    report["noise_floor"] = floors

    # ---------------- the implied eFG% gate (G4) ---------------------------
    efg: dict = {}
    for arm in FG.ARMS:
        p_all = np.concatenate([preds[(c, arm)][:, MAKE] for c in FG.SHOT_CLASSES])
        te_cat = pd.concat([FG.class_slice(te2_all, c) for c in FG.SHOT_CLASSES],
                           ignore_index=True)
        efg[arm] = {"offense": FG.efg_table(te_cat, p_all, side="offense"),
                    "defense": FG.efg_table(te_cat, p_all, side="defense")}
        print(f"eFG {arm}: implied {efg[arm]['offense']['overall_implied_efg_pct']} vs actual "
              f"{efg[arm]['offense']['overall_actual_efg_pct']} "
              f"(tercile worst asof {efg[arm]['offense']['worst_gap_pp_asof']} pp, "
              f"actual {efg[arm]['offense']['worst_gap_pp_actual']} pp)", flush=True)
    report["efg_gate"] = efg

    # ---------------- shrinkage + transfer --------------------------------
    shrink: dict = {}
    transfer: dict = {}
    for c in FG.SHOT_CLASSES:
        te2 = FG.class_slice(te2_all, c)
        b = eb_fits[f"F2|{c}"]["best"]
        m = float(b["m"])
        shrink[c] = {
            "F1": eb_fits[f"F1|{c}"]["best"], "F2": b,
            "attempts_for_50pct_own_weight": m,
            "attempts_for_75pct_own_weight": 3 * m,
            "attempts_for_90pct_own_weight": 9 * m,
            "median_shooter_attempts_at_prediction_time_F2": float(
                np.median(te2["shooter_att_c"].to_numpy())),
            "share_of_F2_attempts_with_own_rate_dominant_pct": round(float(
                (te2["shooter_att_c"].to_numpy() > m).mean() * 100), 2),
        }
        transfer[c] = transfer_block(te2, preds, c)
    report["shrinkage"] = shrink
    report["transfer_check"] = transfer

    # ---------------- first-chance form diagnostic ------------------------
    if not args.skip_first_chance_check:
        print("\nfirst-chance-only team form diagnostic", flush=True)
        d_fc = FG.build_design(SEASONS, universe=universe, version=args.version,
                               events=events, first_chance_form=True)
        fc: dict = {}
        trf, tef = FG.fold_slices(d_fc, "F2")
        for c in FG.SHOT_CLASSES:
            trc, tec = FG.class_slice(trf, c), FG.class_slice(tef, c)
            s, _ = run_arm("ridge", trc, tec)
            base = next(r for r in rows if r["fold"] == "F2" and r["shot_class"] == c
                        and r["arm"] == "ridge")
            corr_off = float(np.corrcoef(
                d_fc.loc[d_fc["shot_class"] == c, "off_make_c"].to_numpy(),
                design.loc[design["shot_class"] == c, "off_make_c"].to_numpy())[0, 1])
            fc[c] = {"ridge_log_loss_first_chance_form": round(s["log_loss"], 6),
                     "ridge_log_loss_all_chance_form": base["log_loss"],
                     "delta": round(s["log_loss"] - base["log_loss"], 6),
                     "delta_in_floors": round((s["log_loss"] - base["log_loss"])
                                              / floors[c]["floor"], 2),
                     "off_make_c_correlation_between_sources": round(corr_off, 5)}
            print(f"  {c}: {fc[c]}", flush=True)
        report["first_chance_form_check"] = fc
        del d_fc

    # A checkpoint before the longest remaining block, so a failure there
    # cannot cost the whole grid.
    (OUT_DIR / "run_report_partial.json").write_text(json.dumps(report, indent=1, default=str))

    # ---------------- the lineup question ---------------------------------
    if not args.skip_lineup:
        print("\nlineup fold L2 (train 2024, test 2025)", flush=True)
        report["lineup"] = lineup_block(design, events, lgbm_params, floors, args.bootstrap_reps)
        print(f"lineup verdict: {report['lineup']['verdict_overall']}", flush=True)

    # ---------------- decision, per class ---------------------------------
    report["decision"] = {c: decide(rows, floors[c]["floor"], c) for c in FG.SHOT_CLASSES}
    for c, v in report["decision"].items():
        print(f"decision {c}: winner={v['winner']} ({v.get('log_loss')})", flush=True)

    # ---------------- persist the winners ---------------------------------
    winners = {}
    for c, v in report["decision"].items():
        if not v["winner"]:
            continue
        arm = v["winner"]
        tr2 = FG.class_slice(tr2_all, c)
        model = FG.fit_arm(arm, tr2, eb_best=eb_fits[f"F2|{c}"]["best"],
                           params=lgbm_params.get(c))
        fs = FG.ARM_FEATURE_SET[arm] or "eb"
        fitted = FG.FittedFgMake(
            arm=arm, feature_set=fs, fold="F2",
            features=FG.feature_set(fs) if fs != "eb" else [],
            models={c: model}, classes=FG.CLASSES,
            meta={"shot_class": c, "possessions_version": args.version,
                  "rim_override_max_ft": report["rim_override_max_ft"],
                  "log_loss_F2": v.get("log_loss"),
                  "lgbm_params": lgbm_params.get(c) if arm == "lgbm" else None,
                  "eb_best": eb_fits[f"F2|{c}"]["best"] if arm == "eb_shrink" else None})
        try:
            import joblib

            p = OUT_DIR / f"winner_{c}.joblib"
            joblib.dump(fitted, p)
            winners[c] = str(p)
        except Exception as exc:  # pragma: no cover - joblib is a soft dependency
            winners[c] = f"NOT WRITTEN: {exc}"
    report["winner_artifacts"] = winners

    report["runtime_min"] = round((time.time() - t_start) / 60, 1)
    report["grid_rows"] = rows
    report["detail"] = detail
    (OUT_DIR / "run_report.json").write_text(json.dumps(report, indent=1, default=str))

    section = render_section(report, grid)
    if args.no_append:
        print(section)
    else:
        with DOC.open("a", encoding="utf-8") as fh:
            fh.write("\n" + section)
        print(f"\nappended results to {DOC}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


GRID_COLS = ["arm", "feature_set", "n", "log_loss", "brier", "pred_make_pct", "actual_make_pct",
             "calib", "worst_gap_pp", "level_pp", "shape_pp", "respons",
             "steps_shooter_make_c", "slope_shooter_make_c", "steps_def_allow_c",
             "slope_def_allow_c", "chancegap_first", "chancegap_continuation", "fit_s"]


def render_section(report: dict, grid: pd.DataFrame) -> str:
    v = report["possessions_version"]
    L: list[str] = []
    L.append(f"## 2. Grid configuration (as executed, run {report['run_at']}, "
             f"`scripts/train_fg_make_v1.py --version {v}`)\n")
    L.append("| Dimension | Values |")
    L.append("|---|---|")
    L.append("| Target | one field-goal attempt: made vs missed, **three separate binary "
             "models** by shot class |")
    L.append(f"| Shot classes | {', '.join(f'`{c}`' for c in FG.SHOT_CLASSES)} -- no fit is ever "
             "shared between them (`fg_make.fit_by_class`) |")
    L.append("| Arms | `team_baseline` (logistic ridge on A_team only), `eb_shrink` (shooter EB "
             "x defence allowed, on the logit scale; prior and both strengths FITTED), `ridge` "
             "(logistic ridge on the full bundle), `lgbm` (LightGBM on the full bundle, "
             "parameters searched on F1 only) |")
    L.append("| Feature sets | `A_team`, `B_plus_shooter`, `C_plus_state` (the full bundle) on "
             "F1/F2; `D_plus_lineup` on its own fold (section 8) |")
    L.append("| Folds | F1: train 2022+2023, test 2024. F2: train 2022+2023+2024, test 2025 "
             "(selection). L2: train 2024, test 2025 (lineup bundle only) |")
    L.append("| Sealed | 2026 -- `assert_not_sealed` on every train and test slice |")
    L.append("| Primary metric | attempt-level log loss on F2, per class |")
    L.append(f"| Possessions version | `{v}`, rim-location override "
             f"{report['rim_override_max_ft']} ft (L16). This model's TARGET depends on it |")
    L.append(f"| Noise floor | per class -- linear: "
             f"{report['noise_floor'][FG.SHOT_CLASSES[0]]['block_bootstrap_reps']}-replicate "
             f"game-level block bootstrap SE on `ridge`; tree: SD over {len(SEED_GRID)} "
             "seed-varied refits |")
    L.append("")
    u = report["universe"]
    dd = report["descriptives"]
    pr = dd["population_rules"]
    L.append(f"Universe: the pre-registration's **D-I, non-truncated, `pbp_complete`** games -- "
             f"{u['n_games_d1_nontruncated_pbp_complete']:,} of the "
             f"{u['n_games_d1_nontruncated']:,} D-I non-truncated games over "
             f"{report['seasons']} ({u['pbp_complete_share_pct']}%). That is STRICTER than the "
             "universe the rebound and free-throw bake-offs ran on, so row counts here are not "
             "comparable to theirs.\n")
    L.append(f"{report['n_attempts']:,} field-goal attempts, of which {report['n_modelled']:,} "
             f"are modelled; {pr['n_no_shooter_dropped']:,} ({pr['no_shooter_pct']}%) carry no "
             "shooter id on the row and are dropped rather than imputed.\n")
    L.append("The two population rules the pre-registration states, as measured rather than "
             f"asserted: **blocked shots are misses** -- {pr['blocked_attempts']:,} attempts "
             f"carry an adjacent block row and {pr['blocked_and_made']} of them "
             f"({pr['blocked_and_made_pct_of_blocks']}% of blocks) are logged as MADE, a feed "
             "artefact that is left exactly as the feed has it and never patched; **and-one "
             f"attempts are makes** -- {pr['and_one_attempts']:,} attempts "
             f"({pr['and_one_pct_of_attempts']}% of all, make rate "
             f"{pr['and_one_make_rate']}) match the and-one signature and are kept as the makes "
             "they are. Both `blocked` and `and_one` are POST-OUTCOME and are in "
             "`fg_make.BANNED_FEATURES`; the assisted flag the pre-registration excludes is not "
             "built at all.\n")

    L.append("### 2.1 The attempt population by season\n")
    rws = [{"season": s, **{k: vv for k, vv in d.items()}}
           for s, d in dd["by_season"].items()]
    cols = ["season", "n_attempts", "n_games", "attempts_per_game", "FGA_rim_share_pct",
            "FGA_jump2_share_pct", "FGA_3_share_pct", "FGA_rim_make_pct",
            "FGA_jump2_make_pct", "FGA_3_make_pct", "efg_pct", "espn_id_coverage_pct",
            "position_coverage_pct", "on_floor_complete_pct"]
    L.append(_table(rws, cols))
    L.append("")
    L.append("The ESPN-id column is why shooter identity is keyed on the CBBD player id and why "
             "the pre-registered `minutes-to-date` feature could not be built: minutes live in "
             "hoopR `player_box` behind the ESPN athlete id, and the crosswalk that reaches them "
             "resolves 0% of two of the three training seasons. `shooter_games_asof` and "
             "`shooter_fga_asof` carry the exposure instead; `features.md` section 3 records the "
             "substitution.\n")

    L.append("### 2.2 By shot class\n")
    rws = [{"shot_class": c, **d} for c, d in dd["by_class"].items()]
    L.append(_table(rws, ["shot_class", "n", "share_of_attempts_pct", "make_pct", "blocked_pct",
                          "and_one_pct_of_makes", "continuation_pct", "transition_pct",
                          "shooter_no_prior_attempt_pct", "n_shooters"]))
    L.append("")
    cs = dd["chance_state"]
    L.append(f"Chance-state derivation (the `C_plus_state` block): mean elapsed time at release "
             f"{cs['elapsed_mean_s']} s, {cs['elapsed_clipped_pct']}% of attempts hit the "
             f"{FG.MAX_CHANCE_ELAPSED_S:.0f} s clip, {cs['transition_pct']}% are transition. The "
             "derivation is validated against the feed rather than trusted: the offence implied "
             f"by the chance's own start event matches the shooter's team on "
             f"{cs['derived_offence_matches_shooter_pct']}% of attempts "
             f"({cs['derived_offence_known_pct']}% have an implied offence at all). Chance-number "
             f"distribution: {cs['chance_number_dist']}.\n")

    L.append("---\n")
    L.append("## 3. Full results, per class\n")
    L.append("`ridge(diagnostic)` is logistic ridge on `B_plus_shooter` -- reported so the "
             "shooter block and the state block are separable, NOT a decision arm (the "
             "pre-registration's ridge arm is the full bundle).\n")
    for c in FG.SHOT_CLASSES:
        L.append(f"### 3.{FG.SHOT_CLASSES.index(c) + 1} `{c}`\n")
        for fold, label in (("F1", "train [2022, 2023], test [2024]"),
                            ("F2", "train [2022, 2023, 2024], test [2025]  (SELECTION)")):
            sub = grid[(grid["fold"] == fold) & (grid["shot_class"] == c)].sort_values("log_loss")
            L.append(f"**{fold}** -- {label}\n")
            L.append(_table(sub.to_dict("records"), GRID_COLS))
            L.append("")
        nf = report["noise_floor"][c]
        L.append(f"Noise floor: game-block bootstrap SE {nf['block_bootstrap_se']} on `ridge`; "
                 f"LightGBM seed-refit SD {nf['lgbm_seed_sd']} over seeds {list(SEED_GRID)} "
                 f"({nf['lgbm_seed_log_losses']}). The floor the decision rule uses is the "
                 f"larger, **{nf['floor']}**.\n")

    L.append("---\n")
    L.append("## 4. The LightGBM parameter ladder (F1 ONLY)\n")
    L.append("The pre-registration allows a parameter search on F1 and nowhere else. The winner "
             "per class is frozen before F2 is touched and is used for every later LightGBM fit "
             "(F2, the seed refits, the lineup fold).\n")
    for c in FG.SHOT_CLASSES:
        b = report["lgbm_ladder"]["by_class"][c]
        L.append(f"**`{c}`** -- winner rung {b['winner_rung']} (`{b['winner_override']}`), F1 log "
                 f"loss {b['winner_F1_log_loss']}\n")
        L.append(_table(b["rows"], ["rung", "override", "num_leaves", "min_child_samples",
                                    "n_estimators", "learning_rate", "reg_lambda",
                                    "F1_log_loss", "fit_s"]))
        L.append("")

    L.append("---\n")
    L.append("## 5. The fitted shrinkage, per class\n")
    L.append("`eb_shrink` is `logit(p) = logit(shooter_EB) + [logit(defence_allowed_EB) - "
             "logit(league_asof)]`: the defence enters as its log-odds deviation from the league "
             "on this shot class, so a league-average defence contributes exactly zero. Both "
             "strengths are fitted on the training fold (the defence's too -- its as-of rate "
             "rests on a handful of attempts in November and thousands in March).\n")
    rws = []
    for c in FG.SHOT_CLASSES:
        s = report["shrinkage"][c]
        rws.append({"shot_class": c,
                    "F1 prior": s["F1"]["prior"], "F1 m": s["F1"]["m"], "F1 m_def": s["F1"]["m_def"],
                    "F2 prior": s["F2"]["prior"], "F2 m": s["F2"]["m"], "F2 m_def": s["F2"]["m_def"],
                    "median shooter attempts at prediction time (F2)":
                        s["median_shooter_attempts_at_prediction_time_F2"],
                    "% of F2 attempts where the shooter's own rate outweighs the prior":
                        s["share_of_F2_attempts_with_own_rate_dominant_pct"]})
    L.append(_table(rws, list(rws[0].keys())))
    L.append("")
    for c in FG.SHOT_CLASSES:
        s = report["shrinkage"][c]
        L.append(f"- `{c}`: at m = {s['F2']['m']} attempts a shooter's own rate carries half the "
                 f"weight at {s['attempts_for_50pct_own_weight']:.0f} attempts, three quarters at "
                 f"{s['attempts_for_75pct_own_weight']:.0f} and nine tenths at "
                 f"{s['attempts_for_90pct_own_weight']:.0f}.")
    L.append("")
    L.append("Full grid (train log loss per prior x shooter strength x defence strength) is in "
             "`data/processed/models/fg_make/run_report.json` under `eb_fits`.\n")

    L.append("### 5.1 Transfer subset\n")
    L.append("The L15 natural experiment: players whose modal team changed since the prior "
             "season, where a prior-season-based prior is least trustworthy.\n")
    for c in FG.SHOT_CLASSES:
        rws = [{"subset": k, **vv} for k, vv in report["transfer_check"][c].items()]
        if not rws:
            continue
        L.append(f"**`{c}`**\n")
        cols = ["subset", "n", "actual_make_rate", "eb_pred_make_rate"] + \
               [f"{a}_log_loss" for a in FG.ARMS]
        L.append(_table(rws, cols))
        L.append("")

    L.append("---\n")
    L.append("## 6. G4: the implied team eFG%, from the three class models on the test season's "
             "own shot mix\n")
    L.append("eFG% = (FGM + 0.5 x 3PM) / FGA, with every attempt the team actually took weighted "
             "by its class model's predicted make probability instead of by the outcome -- so the "
             "shot MIX is taken as given and this is a check of the make models alone. Teams with "
             "fewer than 200 attempts in the test season are excluded. `asof` terciles bucket "
             "teams by their own pregame as-of form (the honest, matchup-specific grouping); "
             "`actual` terciles bucket by realised eFG% and are an ORACLE grouping, labelled as "
             "such, reported because it is the one that exposes a flat model. Gate: +/- 1.0 pp.\n")
    for side in ("offense", "defense"):
        L.append(f"**{side}**\n")
        rws = []
        for arm in FG.ARMS:
            e = report["efg_gate"][arm][side]
            row = {"arm": arm, "n_teams": e["n_teams"],
                   "actual eFG%": e["overall_actual_efg_pct"],
                   "implied eFG%": e["overall_implied_efg_pct"],
                   "overall gap pp": e["overall_gap_pp"],
                   "team MAE pp": e["team_level_mae_pp"],
                   "team corr": e["team_level_corr"]}
            for label in ("asof", "actual"):
                for r in e[f"terciles_{label}"]:
                    row[f"T{r['tercile']} gap pp ({label})"] = r["gap_pp"]
                row[f"worst gap pp ({label})"] = e[f"worst_gap_pp_{label}"]
                row[f"G4 ({label})"] = "PASS" if e[f"pass_{label}"] else "FAIL"
            rws.append(row)
        L.append(_table(rws, list(rws[0].keys())))
        L.append("")

    if "first_chance_form_check" in report:
        L.append("---\n")
        L.append("## 7. Team form from ALL chances vs FIRST chances only\n")
        L.append("The rebound model restricts its team rates to first-chance opportunities "
                 "(ledger row B5: pooling continuation chances lets a continuation-chance "
                 "labelling defect into a first-chance model's predictors). Here the target "
                 "population is ALL attempts and the L16 defect is repaired at the source by the "
                 "v2 override, so the default is all chances -- and the cost of that choice is "
                 "measured rather than argued.\n")
        rws = [{"shot_class": c, **d} for c, d in report["first_chance_form_check"].items()]
        L.append(_table(rws, list(rws[0].keys())))
        L.append("")

    if "lineup" in report:
        ln = report["lineup"]
        L.append("---\n")
        L.append("## 8. Is the DEFENCE lineup-level or team-level? (`D_plus_lineup` on its own "
                 "fold)\n")
        L.append(f"Fold {ln['fold']} (CBBD `onFloor` is empty at the source before 2024, L13). "
                 f"{ln['n_rows_on_floor_complete']:,} of the fold's attempts "
                 f"({ln['on_floor_coverage_pct']}%) carry all ten on-floor ids; C and D are "
                 "scored on exactly those rows so the comparison is like-for-like. The defender "
                 "shrinkage strength is FITTED on the train season over "
                 f"{ln['prior_grid']} pseudo-attempts (L13) and landed on "
                 f"**{ln['prior_att_fitted']}**.\n")
        L.append(_table(ln["shrinkage_fit"], ["prior_att", "train_log_loss", "fit_s"]))
        L.append("")
        for c in FG.SHOT_CLASSES:
            b = ln["by_class"][c]
            L.append(f"**`{c}`** -- fold block-bootstrap floor "
                     f"{b['block_bootstrap_se_on_this_fold']}; ridge gain D over C "
                     f"{b['gains']['ridge']['gain_D_over_C']} "
                     f"({b['gains']['ridge']['gain_in_floors']}x the floor), lgbm gain "
                     f"{b['gains']['lgbm']['gain_D_over_C']} "
                     f"({b['gains']['lgbm']['gain_in_floors']}x) -> **{b['verdict']}**\n")
            L.append(_table(b["rows"], ["arm", "feature_set", "n", "log_loss", "brier", "calib",
                                        "worst_gap_pp", "respons", "resp_min_steps", "fit_s"]))
            L.append("")
        L.append(f"**Verdict: {ln['verdict_overall']}.**\n")

    L.append("---\n")
    L.append("## 9. Decision, by the pre-registered rule, per class\n")
    L.append("An arm passes only if its worst gated decile calibration gap is <= 2.00 pp AND "
             "BOTH responsiveness drivers -- the shooter's as-of class make rate and the "
             "defence's as-of allowed rate -- are monotone in 4 of 4 quintile steps.\n")
    for c in FG.SHOT_CLASSES:
        d = report["decision"][c]
        L.append(f"### `{c}`\n")
        L.append(f"{d['n_passing']} of {d['n_arms']} F2 arms pass both gates"
                 + (f": {', '.join(f'`{a}`' for a in d['passing_arms'])}." if d['passing_arms']
                    else ".") + "\n")
        if d.get("tree_gain_over_non_tree") is not None:
            L.append(f"The tree arm beats the best passing non-tree arm "
                     f"(`{d['best_passing_non_tree']}`) by {d['tree_gain_over_non_tree']} log "
                     f"loss = {d['tree_gain_in_floors']}x the noise floor ({d['floor']}).\n")
        if d.get("tree_clause"):
            L.append(d["tree_clause"] + "\n")
        if d.get("note"):
            L.append(d["note"] + "\n")
        L.append(f"**WINNER ({c}): {d.get('winner') or 'NONE -- nothing adopted for this class'}**"
                 + (f", F2 log loss {d.get('log_loss')}." if d.get("winner") else ".") + "\n")
    L.append(f"Runtime {report['runtime_min']} min.\n")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
