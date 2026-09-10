#!/usr/bin/env python
"""
train_rebound_v1.py -- run the pre-registered L3 REBOUND bake-off.

    .venv/Scripts/python.exe scripts/train_rebound_v1.py
    .venv/Scripts/python.exe scripts/train_rebound_v1.py --version v2
    .venv/Scripts/python.exe scripts/train_rebound_v1.py --no-append

The pre-registration is section 1 of `docs/models/rebound/experiments.md` and
was written before any of this ran. This script executes it and APPENDS its
results to that file; it never edits what is already there.

`--version` selects the possessions build. For this model it changes exactly
one thing -- whether ESPN's 2025 putback mistag is repaired before a miss is
labelled rim vs jumper (L16) -- so a re-run on `possessions_v2` is one flag,
which is what the PM asked for. The rebound OUTCOME, the opportunity set, and
every team rate are identical under both versions by construction (the override
can only move an `FGA_jump2` to an `FGA_rim`), and the run report says so with
the numbers.

Artifacts under `data/processed/models/rebound/`:
    events_{version}.parquet      one row per rebound opportunity (cached)
    grid_results.csv              every (arm, feature set, fold) row
    run_report.json               everything the markdown section is rendered from
    winner.joblib                 the selected arm, if one passes the gates
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
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

OUT_DIR = Path("data/processed/models/rebound")
DOC = Path("docs/models/rebound/experiments.md")
SEASONS = [2022, 2023, 2024, 2025]
#: Grid feature sets for the main folds. `D_plus_lineup` is excluded here by
#: the pre-registration itself: CBBD `onFloor` is empty at the source before
#: 2024 (L13), so the bundle cannot be built for an F1 or F2 TRAIN window. It
#: is run on its own fold in section 5.
MAIN_SETS = ("A_team", "B_plus_miss", "C_plus_state")
SEED_GRID = (0, 1, 2, 3, 4)


def build_or_load_events(version: str, rebuild: bool, universe: pd.DataFrame) -> pd.DataFrame:
    path = OUT_DIR / f"events_{version}.parquet"
    if path.exists() and not rebuild:
        ev = pd.read_parquet(path)
        ev.attrs["rim_override_max_ft"] = ES.rim_override_for_version(version)
        ev.attrs["possessions_version"] = version
        print(f"loaded cached opportunities: {path} ({len(ev):,} rows)")
        return ev
    t0 = time.time()
    ev = RB.build_rebound_events(SEASONS, universe=universe, version=version)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(path, index=False)
    print(f"built {len(ev):,} rebound opportunities in {time.time() - t0:.1f}s -> {path}")
    return ev


def run_arm(arm: str, fs_name: str, tr: pd.DataFrame, te: pd.DataFrame, seed: int = 0) -> dict:
    feats = RB.feature_set(fs_name)
    t0 = time.time()
    model = RB.fit_arm(arm, tr, feats, seed=seed)
    p = RB.predict_arm(arm, model, te, feats)
    s = RB.score(te, p)
    s["arm"] = arm
    s["feature_set"] = fs_name
    s["fit_s"] = round(time.time() - t0, 1)
    return s, p, model


def flat_row(s: dict, fold: str) -> dict:
    calib = s["calibration"]
    resp = s["responsiveness"]
    gated = {c: v for c, v in calib.items() if v["share_pct"] >= PM.CALIB_MIN_SHARE * 100}
    level = max((abs(v["level_shift_pp"]) for v in gated.values()), default=0.0)
    shape = max((v["max_abs_gap_pp_after_level_shift"] for v in gated.values()), default=0.0)
    row = {
        "fold": fold, "arm": s["arm"], "feature_set": s["feature_set"], "n": s["n"],
        "log_loss": round(s["log_loss"], 6), "brier": round(s["brier"], 6),
        "calib": "PASS" if s["calib_pass"] else "FAIL",
        "worst_gap_pp": s["calib_worst_gap_pp"], "worst_class": s["calib_worst_class"],
        "level_pp": round(level, 3), "shape_pp": round(shape, 3),
        "respons": "PASS" if s["resp_pass"] else "FAIL",
        "resp_min_steps": s["resp_min_steps"], "fit_s": s["fit_s"],
    }
    for c, v in s["brier_by_class"].items():
        row[f"brier_{c}"] = round(v, 6)
    for k, v in resp.items():
        row[f"slope_{k.split('->')[0]}"] = v["slope_ratio"]
    for m, v in s["by_miss_type"].items():
        row[f"missgap_{m}"] = v["max_abs_gap_pp"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1",
                    help="possessions build whose rim/jumper labelling the miss-type feature uses")
    ap.add_argument("--rebuild-events", action="store_true")
    ap.add_argument("--no-append", action="store_true", help="print the section, do not append it")
    ap.add_argument("--bootstrap-reps", type=int, default=200)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    universe = ES.load_universe()
    events = build_or_load_events(args.version, args.rebuild_events, universe)
    design = RB.build_design(SEASONS, universe=universe, version=args.version, events=events)
    print(f"design: {len(design):,} modelled opportunities "
          f"({design.attrs['n_unresolved']:,} unresolved dropped)")

    report: dict = {
        "run_at": time.strftime("%Y-%m-%d %H:%M"),
        "possessions_version": args.version,
        "rim_override_max_ft": design.attrs["rim_override_max_ft"],
        "n_opportunities": design.attrs["n_opportunities"],
        "n_modelled": int(len(design)),
        "n_unresolved": design.attrs["n_unresolved"],
        "seasons": SEASONS,
    }

    # --- descriptive: class shares by season and miss type -----------------
    shares = {}
    for s in SEASONS:
        d = design[design["season"] == s]
        ev = events[events["season"] == s]
        shares[str(s)] = {
            "n_opportunities": int(len(ev)),
            "unresolved_pct": round(float((ev["outcome"] == "unresolved").mean() * 100), 3),
            "OREB_pct": round(float((d["y"] == RB.CLASS_INDEX["OREB"]).mean() * 100), 3),
            "DREB_pct": round(float((d["y"] == RB.CLASS_INDEX["DREB"]).mean() * 100), 3),
            "DEAD_pct": round(float((d["y"] == RB.CLASS_INDEX["DEAD"]).mean() * 100), 3),
            "on_floor_complete_pct": round(float(
                ev[list(ES.ON_FLOOR_COLS)].notna().all(axis=1).mean() * 100), 2),
        }
    report["season_shares"] = shares
    by_miss = {}
    for m in RB.MISS_TYPES:
        e = events[events["miss_type"] == m]
        d = design[design["miss_type"] == m]
        by_miss[m] = {
            "n": int(len(e)),
            "share_of_opportunities_pct": round(float(len(e) / max(len(events), 1) * 100), 3),
            "OREB_pct": round(float((d["y"] == 0).mean() * 100), 3),
            "DREB_pct": round(float((d["y"] == 1).mean() * 100), 3),
            "DEAD_pct": round(float((d["y"] == 2).mean() * 100), 3),
            "unresolved_pct": round(float((e["outcome"] == "unresolved").mean() * 100), 3),
        }
    report["by_miss_type"] = by_miss

    # ------------------------------------------------------------------
    # The grid
    # ------------------------------------------------------------------
    rows: list[dict] = []
    detail: dict = {}
    preds: dict = {}
    for fold in ("F1", "F2"):
        tr, te = RB.fold_slices(design, fold)
        print(f"\n{fold}: train {len(tr):,} test {len(te):,}")
        for arm in RB.ARMS:
            sets = ("A_team",) if arm == "baseline" else MAIN_SETS
            for fs_name in sets:
                s, p, _ = run_arm(arm, fs_name, tr, te)
                key = f"{fold}|{arm}|{fs_name}"
                detail[key] = s
                if fold == RB.SELECTION_FOLD:
                    preds[key] = p
                rows.append(flat_row(s, fold))
                print(f"  {arm:12s} {fs_name:14s} ll={s['log_loss']:.6f} "
                      f"calib={'PASS' if s['calib_pass'] else 'FAIL'}({s['calib_worst_gap_pp']}) "
                      f"resp={'PASS' if s['resp_pass'] else 'FAIL'} {s['fit_s']}s")
        # the labelled-unattainable oracle baseline (L3 round-1 convention)
        if fold == RB.SELECTION_FOLD:
            orc = RB.BaselineArm().fit(te["y"].to_numpy(), te["miss_type"].to_numpy(),
                                       te["season"].to_numpy())
            p_orc = orc.predict_proba(te["miss_type"].to_numpy())
            report["oracle_baseline_F2_log_loss"] = round(PM.log_loss(te["y"].to_numpy(), p_orc), 6)

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "grid_results.csv", index=False)

    # ------------------------------------------------------------------
    # Noise floors
    # ------------------------------------------------------------------
    tr2, te2 = RB.fold_slices(design, "F2")
    best_lin = min((r for r in rows if r["fold"] == "F2" and r["arm"] == "ridge_logit"),
                   key=lambda r: r["log_loss"])
    p_lin = preds[f"F2|ridge_logit|{best_lin['feature_set']}"]
    floor_boot = PM.block_bootstrap_se(te2["game_id"].to_numpy(), te2["y"].to_numpy(),
                                       p_lin, n_rep=args.bootstrap_reps)
    seed_ll = []
    for sd in SEED_GRID:
        s, _, _ = run_arm("lgbm", "C_plus_state", tr2, te2, seed=sd)
        seed_ll.append(s["log_loss"])
    floor_seed = float(np.std(seed_ll, ddof=1))
    report["noise_floor"] = {
        "block_bootstrap_se": round(floor_boot, 6),
        "block_bootstrap_reps": args.bootstrap_reps,
        "block_bootstrap_arm": f"ridge_logit/{best_lin['feature_set']}",
        "lgbm_seed_sd": round(floor_seed, 6),
        "lgbm_seed_log_losses": [round(x, 6) for x in seed_ll],
        "floor": round(max(floor_boot, floor_seed), 6),
    }
    floor = max(floor_boot, floor_seed)
    print(f"\nnoise floor: bootstrap {floor_boot:.6f}, lgbm seed SD {floor_seed:.6f} -> {floor:.6f}")

    # ------------------------------------------------------------------
    # The dead-ball question
    # ------------------------------------------------------------------
    dead_share = RB.deterministic_dead_share(tr2)
    live_tr = tr2[tr2["y"] != RB.CLASS_INDEX["DEAD"]]
    feats = RB.feature_set("C_plus_state")
    from sklearn.linear_model import LogisticRegression
    Xtr = live_tr[feats].to_numpy(dtype="float32")
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=0).fit(
        (Xtr - mu) / sd, (live_tr["y"].to_numpy() == RB.CLASS_INDEX["OREB"]).astype(int))
    Xte = (te2[feats].to_numpy(dtype="float32") - mu) / sd
    p_oreb = clf.predict_proba(Xte)[:, 1]
    p_fixed = RB.compose_binary_plus_fixed_dead(p_oreb, te2["miss_type"].to_numpy(), dead_share)
    ll_fixed = PM.log_loss(te2["y"].to_numpy(), p_fixed)
    ll_full = detail["F2|ridge_logit|C_plus_state"]["log_loss"]
    calib_fixed = PM.decile_calibration(te2["y"].to_numpy(), p_fixed, RB.CLASSES)
    report["dead_ball"] = {
        "train_share_by_miss_type": {k: round(v, 5) for k, v in dead_share.items()},
        "test_share_by_miss_type": {
            m: round(float((te2.loc[te2["miss_type"] == m, "y"] == 2).mean()), 5)
            for m in RB.MISS_TYPES},
        "log_loss_three_class_ridge": round(ll_full, 6),
        "log_loss_binary_plus_fixed_share": round(ll_fixed, 6),
        "delta": round(ll_fixed - ll_full, 6),
        "delta_in_floors": round((ll_fixed - ll_full) / floor, 2) if floor else None,
        "dead_calibration": {k: {kk: vv for kk, vv in v.items() if kk != "bins"}
                             for k, v in calib_fixed.items()},
        "next_action_by_shooting_team_share": None,
    }

    # ------------------------------------------------------------------
    # The lineup question: its own fold, its own rows
    # ------------------------------------------------------------------
    report["lineup"] = run_lineup_block(design, events, floor)

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    f2 = [r for r in rows if r["fold"] == "F2"]
    passing = [r for r in f2 if r["calib"] == "PASS" and r["respons"] == "PASS"]
    all_lin = [r for r in f2 if r["arm"] == "ridge_logit"]
    best_lin_any = min(all_lin, key=lambda r: r["log_loss"]) if all_lin else None
    best_tree_any = min((r for r in f2 if r["arm"] == "lgbm"), key=lambda r: r["log_loss"])
    verdict: dict = {
        "n_arms": len(f2),
        "n_arms_passing_both_gates": len(passing),
        "best_linear_any": f"{best_lin_any['arm']}/{best_lin_any['feature_set']}" if best_lin_any else None,
        "best_linear_any_log_loss": best_lin_any["log_loss"] if best_lin_any else None,
        "best_tree_any": f"{best_tree_any['arm']}/{best_tree_any['feature_set']}",
        "best_tree_any_log_loss": best_tree_any["log_loss"],
    }
    if best_lin_any is not None:
        gap = best_lin_any["log_loss"] - best_tree_any["log_loss"]
        verdict["tree_minus_linear_all_arms"] = round(gap, 6)
        verdict["tree_minus_linear_in_floors"] = round(gap / floor, 2) if floor else None
    if passing:
        best = min(passing, key=lambda r: r["log_loss"])
        lin = [r for r in passing if r["arm"] in ("ridge_logit", "baseline")]
        best_lin_pass = min(lin, key=lambda r: r["log_loss"]) if lin else None
        verdict["winner"] = f"{best['arm']}/{best['feature_set']}"
        verdict["log_loss"] = best["log_loss"]
        verdict["passing_arms"] = [f"{r['arm']}/{r['feature_set']}" for r in passing]
        if best["arm"] == "lgbm":
            if best_lin_pass is None:
                verdict["tree_clause"] = (
                    "no LINEAR arm passes both gates, so the pre-registered "
                    "'a tree arm must beat the best linear arm by more than the floor' clause has "
                    "no passing linear arm to bind against. The gap to the best linear arm OF ANY "
                    f"gate status is reported above ({verdict.get('tree_minus_linear_all_arms')} = "
                    f"{verdict.get('tree_minus_linear_in_floors')}x the floor) so the PM can apply "
                    "the stricter reading if they prefer it.")
            else:
                gain = best_lin_pass["log_loss"] - best["log_loss"]
                verdict["tree_gain_over_linear"] = round(gain, 6)
                verdict["tree_gain_in_floors"] = round(gain / floor, 2) if floor else None
                if gain <= floor:
                    verdict["winner"] = f"{best_lin_pass['arm']}/{best_lin_pass['feature_set']}"
                    verdict["note"] = ("the tree arm's edge is inside the noise floor, so the "
                                       "pre-registered simplicity tie-break selects the linear arm")
    else:
        verdict["winner"] = None
        verdict["note"] = "no arm passes both pre-registered gates on F2"
    report["decision"] = verdict
    report["runtime_min"] = round((time.time() - t_start) / 60, 1)
    report["grid_rows"] = rows
    report["detail"] = {k: {kk: vv for kk, vv in v.items()} for k, v in detail.items()}

    (OUT_DIR / "run_report.json").write_text(json.dumps(report, indent=1, default=str))
    section = render_section(report, grid)
    if args.no_append:
        print(section)
    else:
        with DOC.open("a", encoding="utf-8") as fh:
            fh.write("\n" + section)
        print(f"\nappended results to {DOC}")
    return 0


def run_lineup_block(design: pd.DataFrame, events: pd.DataFrame, floor: float) -> dict:
    """C vs D on the lineup fold (train 2024, test 2025), on the rows that
    carry all ten on-floor ids, with the shrinkage strength FITTED on the train
    season (L13)."""
    seasons = RB.LINEUP_FOLDS["L2"]["train"] + RB.LINEUP_FOLDS["L2"]["test"]
    ev = events[events["season"].isin(seasons)]
    out: dict = {"fold": "L2 (train 2024, test 2025)", "prior_grid": list(RB.LINEUP_PRIOR_GRID)}
    fits = []
    best = None
    for k in RB.LINEUP_PRIOR_GRID:
        rates = RB.player_rebound_rates(ev, prior_opps=k)
        d = RB.attach_lineup_features(design[design["season"].isin(seasons)], rates)
        sub = d[d["lineup_on_floor_ok"]]
        tr, te = RB.fold_slices(sub, "L2")
        feats = RB.feature_set("D_plus_lineup")
        m = RB.fit_arm("ridge_logit", tr, feats)
        p_tr = RB.predict_arm("ridge_logit", m, tr, feats)
        ll_tr = PM.log_loss(tr["y"].to_numpy(), p_tr)
        fits.append({"prior_opps": int(k), "train_log_loss": round(ll_tr, 6)})
        if best is None or ll_tr < best[1]:
            best = (k, ll_tr, d)
    out["shrinkage_fit"] = fits
    k_best, _, d_best = best
    out["prior_opps_fitted"] = int(k_best)

    sub = d_best[d_best["lineup_on_floor_ok"]]
    out["n_rows_on_floor_complete"] = int(len(sub))
    out["on_floor_coverage_pct"] = round(float(len(sub) / max(len(d_best), 1) * 100), 2)
    tr, te = RB.fold_slices(sub, "L2")
    out["n_train"] = int(len(tr))
    out["n_test"] = int(len(te))
    rows = []
    p_by = {}
    for arm in ("ridge_logit", "lgbm"):
        for fs_name in ("C_plus_state", "D_plus_lineup"):
            feats = RB.feature_set(fs_name)
            t0 = time.time()
            m = RB.fit_arm(arm, tr, feats)
            p = RB.predict_arm(arm, m, te, feats)
            s = RB.score(te, p)
            p_by[f"{arm}|{fs_name}"] = p
            rows.append({"arm": arm, "feature_set": fs_name, "n": s["n"],
                         "log_loss": round(s["log_loss"], 6), "brier": round(s["brier"], 6),
                         "calib": "PASS" if s["calib_pass"] else "FAIL",
                         "worst_gap_pp": s["calib_worst_gap_pp"],
                         "respons": "PASS" if s["resp_pass"] else "FAIL",
                         "resp_min_steps": s["resp_min_steps"],
                         "fit_s": round(time.time() - t0, 1)})
    out["rows"] = rows
    floor_l2 = PM.block_bootstrap_se(te["game_id"].to_numpy(), te["y"].to_numpy(),
                                     p_by["ridge_logit|C_plus_state"], n_rep=200)
    out["block_bootstrap_se_on_this_fold"] = round(floor_l2, 6)
    c = next(r for r in rows if r["arm"] == "ridge_logit" and r["feature_set"] == "C_plus_state")
    dd = next(r for r in rows if r["arm"] == "ridge_logit" and r["feature_set"] == "D_plus_lineup")
    gain = c["log_loss"] - dd["log_loss"]
    out["ridge_gain_D_over_C"] = round(gain, 6)
    out["ridge_gain_in_floors"] = round(gain / floor_l2, 2) if floor_l2 else None
    cl = next(r for r in rows if r["arm"] == "lgbm" and r["feature_set"] == "C_plus_state")
    dl = next(r for r in rows if r["arm"] == "lgbm" and r["feature_set"] == "D_plus_lineup")
    out["lgbm_gain_D_over_C"] = round(cl["log_loss"] - dl["log_loss"], 6)
    out["lgbm_gain_in_floors"] = round((cl["log_loss"] - dl["log_loss"]) / floor_l2, 2) if floor_l2 else None
    out["verdict"] = ("LINEUP-LEVEL" if gain > floor_l2 else "TEAM-LEVEL")
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


GRID_COLS = ["arm", "feature_set", "n", "log_loss", "brier", "calib", "worst_gap_pp",
             "worst_class", "level_pp", "shape_pp", "respons", "resp_min_steps",
             "brier_OREB", "brier_DREB", "brier_DEAD", "slope_off_oreb_c",
             "slope_opp_def_dreb_c", "missgap_rim", "missgap_jump2", "missgap_three",
             "missgap_ft", "fit_s"]


def render_section(report: dict, grid: pd.DataFrame) -> str:
    v = report["possessions_version"]
    lines: list[str] = []
    lines.append(f"## 2. Grid configuration (as executed, run {report['run_at']}, "
                 f"`scripts/train_rebound_v1.py --version {v}`)\n")
    lines.append("| Dimension | Values |")
    lines.append("|---|---|")
    lines.append("| Target | rebound outcome of an opportunity, 3 classes: OREB, DREB, DEAD |")
    lines.append("| Opportunity | a missed FGA, or the missed LAST free throw of a non-technical trip |")
    lines.append(f"| Feature sets | {', '.join(MAIN_SETS)} on F1/F2; `D_plus_lineup` on its own fold (section 5) |")
    lines.append("| Model classes | `baseline` (league shares BY MISS TYPE, most recent training season), "
                 "`ridge_logit` (multinomial), `lgbm` (LightGBM multiclass) |")
    lines.append("| Folds | F1: train 2022+2023, test 2024. F2: train 2022+2023+2024, test 2025 (selection). "
                 "L2: train 2024, test 2025 (lineup bundle only) |")
    lines.append("| Sealed | 2026 -- `assert_not_sealed` is called on every train and test slice |")
    lines.append("| Primary metric | three-class log loss on F2 |")
    lines.append(f"| Possessions version | `{v}`, rim-location override "
                 f"{report['rim_override_max_ft']} ft |")
    lines.append(f"| Noise floor | linear: {report['noise_floor']['block_bootstrap_reps']}-replicate "
                 f"game-level block bootstrap SE; tree: SD over {len(SEED_GRID)} seed-varied refits |")
    lines.append("")
    lines.append(f"Design: {report['n_opportunities']:,} rebound opportunities over "
                 f"{report['seasons']}, of which {report['n_modelled']:,} are modelled and "
                 f"{report['n_unresolved']:,} ({report['n_unresolved'] / report['n_opportunities'] * 100:.2f}%) "
                 "are `unresolved` -- the rebound row never appears in the feed, so the row is dropped "
                 "with its count reported rather than imputed. Runtime "
                 f"{report['runtime_min']} min. CSV alongside: "
                 "`data/processed/models/rebound/grid_results.csv`.\n")

    lines.append("### 2.1 Class shares by season\n")
    rows = [{"season": k, **v2} for k, v2 in report["season_shares"].items()]
    lines.append(_table(rows, ["season", "n_opportunities", "OREB_pct", "DREB_pct", "DEAD_pct",
                               "unresolved_pct", "on_floor_complete_pct"]))
    lines.append("")
    lines.append("### 2.2 Class shares by miss type (pooled 2022-2025)\n")
    rows = [{"miss_type": k, **v2} for k, v2 in report["by_miss_type"].items()]
    lines.append(_table(rows, ["miss_type", "n", "share_of_opportunities_pct", "OREB_pct",
                               "DREB_pct", "DEAD_pct", "unresolved_pct"]))
    lines.append("")

    lines.append("---\n")
    lines.append("## 3. Full results\n")
    for fold, label in (("F1", "train [2022, 2023], test [2024]"),
                        ("F2", "train [2022, 2023, 2024], test [2025]  (SELECTION)")):
        sub = grid[grid["fold"] == fold].sort_values("log_loss")
        lines.append(f"**{fold}** -- {label}\n")
        lines.append(_table(sub.to_dict("records"), GRID_COLS))
        lines.append("")
    nf = report["noise_floor"]
    lines.append(f"Noise floor: game-block bootstrap SE {nf['block_bootstrap_se']} on "
                 f"`{nf['block_bootstrap_arm']}`; LightGBM seed-refit SD {nf['lgbm_seed_sd']} over seeds "
                 f"{list(SEED_GRID)} ({nf['lgbm_seed_log_losses']}). The floor the decision rule uses is "
                 f"the larger, **{nf['floor']}**.\n")
    lines.append(f"Oracle (unattainable) baseline on F2 -- the test season's OWN shares by miss type -- "
                 f"log loss {report.get('oracle_baseline_F2_log_loss')}. The gap between it and the "
                 "honest baseline is the season-drift cost L11 measures, reported so it is visible "
                 "rather than hidden.\n")

    lines.append("---\n")
    lines.append("## 4. Dead-ball rebounds: third class, or a fixed share?\n")
    db = report["dead_ball"]
    lines.append(f"Dead balls are {report['by_miss_type']['rim']['DEAD_pct']}-"
                 f"{report['by_miss_type']['three']['DEAD_pct']}% of opportunities by miss type "
                 f"(train shares {db['train_share_by_miss_type']}, F2 test shares "
                 f"{db['test_share_by_miss_type']}).\n")
    lines.append(f"Full three-class `ridge_logit`/`C_plus_state`: log loss "
                 f"{db['log_loss_three_class_ridge']}. A LIVE-only binary model of the same shape, "
                 f"composed with a deterministic dead-ball share by miss type taken from the training "
                 f"fold: {db['log_loss_binary_plus_fixed_share']}. Difference "
                 f"{db['delta']} = {db['delta_in_floors']}x the noise floor.\n")

    lines.append("---\n")
    lines.append("## 5. Lineup vs team: does `D_plus_lineup` beat `C_plus_state`?\n")
    lb = report["lineup"]
    lines.append(f"Fold {lb['fold']}. {lb['n_rows_on_floor_complete']:,} of the fold's opportunities "
                 f"carry all ten on-floor ids ({lb['on_floor_coverage_pct']}%); C and D are scored on "
                 f"exactly those rows, so the comparison is like-for-like. Train {lb['n_train']:,}, "
                 f"test {lb['n_test']:,}.\n")
    lines.append(f"Individual-rate shrinkage strength FITTED on the train season over "
                 f"{lb['prior_grid']} pseudo-opportunities (L13: never assumed): "
                 f"**{lb['prior_opps_fitted']}**. Train log loss per rung: "
                 f"{[(f['prior_opps'], f['train_log_loss']) for f in lb['shrinkage_fit']]}.\n")
    lines.append(_table(lb["rows"], ["arm", "feature_set", "n", "log_loss", "brier", "calib",
                                     "worst_gap_pp", "respons", "resp_min_steps", "fit_s"]))
    lines.append("")
    lines.append(f"Block-bootstrap SE on this fold: {lb['block_bootstrap_se_on_this_fold']}. "
                 f"`ridge_logit` D - C gain {lb['ridge_gain_D_over_C']} "
                 f"({lb['ridge_gain_in_floors']}x floor); `lgbm` D - C gain "
                 f"{lb['lgbm_gain_D_over_C']} ({lb['lgbm_gain_in_floors']}x floor). "
                 f"**Verdict: {lb['verdict']}.**\n")

    lines.append("---\n")
    lines.append("## 6. Decision, by the pre-registered rule\n")
    d = report["decision"]
    lines.append(f"{d['n_arms_passing_both_gates']} of {d['n_arms']} F2 arms pass BOTH the "
                 "calibration gate (worst decile gap <= 2.00 pp on classes with a >= 5% share) "
                 "and the responsiveness gate (4 of 4 monotone quintile steps on BOTH the offence "
                 "as-of OREB% and the defence as-of DREB% drivers).\n")
    if d.get("passing_arms"):
        lines.append(f"Passing: {', '.join(f'`{a}`' for a in d['passing_arms'])}.\n")
    lines.append(f"Best linear arm of any gate status: `{d['best_linear_any']}` at "
                 f"{d['best_linear_any_log_loss']}. Best tree arm: `{d['best_tree_any']}` at "
                 f"{d['best_tree_any_log_loss']}. Difference "
                 f"{d.get('tree_minus_linear_all_arms')} = "
                 f"{d.get('tree_minus_linear_in_floors')}x the noise floor.\n")
    if d.get("tree_clause"):
        lines.append(d["tree_clause"] + "\n")
    if d.get("note"):
        lines.append(d["note"] + "\n")
    lines.append(f"**WINNER: {d.get('winner') or 'NONE -- no arm passes both gates'}**"
                 + (f", F2 log loss {d.get('log_loss')}." if d.get("winner") else ".") + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
