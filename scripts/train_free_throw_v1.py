#!/usr/bin/env python
"""
train_free_throw_v1.py -- run the pre-registered L3 FREE-THROW bake-off, both
sub-models.

    .venv/Scripts/python.exe scripts/train_free_throw_v1.py
    .venv/Scripts/python.exe scripts/train_free_throw_v1.py --version v2
    .venv/Scripts/python.exe scripts/train_free_throw_v1.py --no-append

The pre-registration is section 1 of `docs/models/free_throw/experiments.md`
and was written before any of this ran. This script executes it and APPENDS its
results; it never edits what is already there.

FT-1 (trip structure) is a RULE CHECK, not a fit: the rule table in
`cbb_sim.models.free_throw.TRIP_RULES` says how many attempts each foul class
produces, the foul class is derived from context only, and this script reports
the per-season violation counts plus the bonus thresholds RE-DERIVED from each
season's own data. The era question the pre-registration asks -- did the
2024-25 bonus structure change -- is answered by whether those derived
thresholds move, not by an assumption about the rulebook.

FT-2 (make probability) is the bake-off proper.

`--version` selects the possessions build. Nothing in this model depends on it;
the script proves that by rebuilding the attempt table under both versions and
asserting the tables are identical (`--check-versions`).

Artifacts under `data/processed/models/free_throw/`:
    attempts_{version}.parquet   one row per free-throw attempt (cached)
    trips_{version}.parquet      one row per trip
    bonus_era.json               the per-season thresholds the ENGINE reads
                                 into GameState
    grid_results.csv             every (arm, fold) row
    run_report.json
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
from cbb_sim.models import free_throw as FT  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402

OUT_DIR = Path("data/processed/models/free_throw")
DOC = Path("docs/models/free_throw/experiments.md")
SEASONS = [2022, 2023, 2024, 2025]
ERA_SEASONS = [2022, 2023, 2024, 2025, 2026]   # 2026 is DESCRIPTIVE only, never a fold
SEED_GRID = (0, 1, 2, 3, 4)


def build_or_load(version: str, rebuild: bool, universe: pd.DataFrame,
                  seasons: list[int], tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tp = OUT_DIR / f"trips_{version}_{tag}.parquet"
    ap = OUT_DIR / f"attempts_{version}_{tag}.parquet"
    if tp.exists() and ap.exists() and not rebuild:
        print(f"loaded cached {tag} tables")
        return pd.read_parquet(tp), pd.read_parquet(ap)
    t0 = time.time()
    trips, att = FT.build_trips_and_attempts(seasons, universe=universe, version=version)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trips.to_parquet(tp, index=False)
    att.to_parquet(ap, index=False)
    print(f"built {len(trips):,} trips / {len(att):,} attempts ({tag}) in {time.time() - t0:.1f}s")
    return trips, att


# ---------------------------------------------------------------------------
# FT-1
# ---------------------------------------------------------------------------
def ft1_block(trips: pd.DataFrame) -> dict:
    out: dict = {"by_season": {}, "pooled": {}}
    for s in sorted(trips["season"].unique()):
        t = trips[trips["season"] == s]
        derived = FT.derive_bonus_thresholds(t)
        bp = derived["bonus_prior_fouls"]
        dp = derived["double_bonus_prior_fouls"]
        # re-classify with THIS season's own derived thresholds, so a moved
        # threshold shows up as a moved class boundary and not as a violation
        t = t.assign(foul_class=FT.classify_foul(
            t["trip_cause"], t["trip_andone_ctx"], t["trip_prior_fouls"],
            bonus_prior=bp if bp is not None else FT.BONUS_THRESHOLDS_DEFAULT[0],
            double_prior=dp if dp is not None else FT.BONUS_THRESHOLDS_DEFAULT[1]))
        rules = FT.verify_trip_rules(t)
        n_games = int(t["game_id"].nunique())
        bonus_trips = int((t["foul_class"] == "bonus_one_and_one").sum())
        dbl_trips = int((t["foul_class"] == "double_bonus").sum())
        out["by_season"][str(int(s))] = {
            "bonus_prior_fouls": bp,
            "double_bonus_prior_fouls": dp,
            "n_trips": int(len(t)),
            "n_games": n_games,
            "trips_per_game": round(len(t) / max(n_games, 1), 3),
            "attempts_per_game": round(float(t["trip_len"].sum()) / max(n_games, 1), 3),
            "bonus_trips_per_game": round(bonus_trips / max(n_games, 1), 3),
            "double_bonus_trips_per_game": round(dbl_trips / max(n_games, 1), 3),
            "one_and_one_share_of_trips_pct": round(bonus_trips / max(len(t), 1) * 100, 3),
            "rules": rules,
            "derivation": derived,
        }
    return out


def bonus_era_json(ft1: dict) -> dict:
    by = {}
    for s, v in ft1["by_season"].items():
        by[s] = {"bonus_prior_fouls": v["bonus_prior_fouls"],
                 "double_bonus_prior_fouls": v["double_bonus_prior_fouls"],
                 "derived_from": "one-attempt-trip miss share by prior foul count"}
    thresholds = {(v["bonus_prior_fouls"], v["double_bonus_prior_fouls"])
                  for v in by.values()}
    return {
        "built_at": time.strftime("%Y-%m-%d %H:%M"),
        "by_season": by,
        "n_distinct_threshold_pairs": len(thresholds),
        "era_boundary_detected": len(thresholds) > 1,
        "note": ("THE ENGINE READS THIS INTO GameState (CLAUDE.md: rule-era flags live in "
                 "GameState, not baked into sub-models). No fitted object in "
                 "cbb_sim.models.free_throw carries a season-specific rule constant."),
    }


# ---------------------------------------------------------------------------
# FT-2
# ---------------------------------------------------------------------------
def run_arm(arm: str, tr: pd.DataFrame, te: pd.DataFrame, eb_best: dict | None = None,
            seed: int = 0) -> tuple[dict, np.ndarray]:
    t0 = time.time()
    if arm == "team_asof":
        p = FT.team_predict(te)
    elif arm == "eb_shrink":
        p = FT.eb_predict(te, eb_best["prior"], eb_best["m"])
    elif arm == "ridge":
        m = FT.RidgeArm(seed=seed).fit(FT.design_matrix(tr), tr["y"].to_numpy())
        p = m.predict_proba(FT.design_matrix(te))
    elif arm == "lgbm":
        m = FT.LgbmArm(seed=seed).fit(FT.design_matrix(tr), tr["y"].to_numpy())
        p = m.predict_proba(FT.design_matrix(te))
    else:
        raise KeyError(arm)
    s = FT.score(te, p)
    s["arm"] = arm
    s["fit_s"] = round(time.time() - t0, 1)
    return s, p


def flat_row(s: dict, fold: str) -> dict:
    r = s["responsiveness"]["shooter_ft_asof->MAKE"]
    gated = {c: v for c, v in s["calibration"].items() if v["share_pct"] >= 5.0}
    level = max((abs(v["level_shift_pp"]) for v in gated.values()), default=0.0)
    shape = max((v["max_abs_gap_pp_after_level_shift"] for v in gated.values()), default=0.0)
    return {
        "fold": fold, "arm": s["arm"], "n": s["n"],
        "log_loss": round(s["log_loss"], 6), "brier": round(s["brier"], 6),
        "calib": "PASS" if s["calib_pass"] else "FAIL",
        "worst_gap_pp": s["calib_worst_gap_pp"],
        "level_pp": round(level, 3), "shape_pp": round(shape, 3),
        "respons": "PASS" if s["resp_pass"] else "FAIL",
        "resp_min_steps": s["resp_min_steps"],
        "slope_ratio": r["slope_ratio"],
        "bonus_gap_pp": s["by_bonus"].get("bonus", {}).get("max_abs_gap_pp"),
        "shooting_gap_pp": s["by_bonus"].get("shooting", {}).get("max_abs_gap_pp"),
        "fit_s": s["fit_s"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--check-versions", action="store_true",
                    help="rebuild under v1 and v2 and assert the attempt tables are identical")
    ap.add_argument("--bootstrap-reps", type=int, default=200)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    universe = ES.load_universe()

    # era block needs 2026 as a DESCRIPTIVE season (a rule check is data, not a
    # fit -- the same standing `scripts/build_possessions.py` gives the 2026
    # possession tables). No fold ever sees it: `fold_slices` guards that.
    trips_all, att_all = build_or_load(args.version, args.rebuild, universe, ERA_SEASONS, "era")
    trips = trips_all[trips_all["season"].isin(SEASONS)]
    attempts = att_all[att_all["season"].isin(SEASONS)]

    report: dict = {
        "run_at": time.strftime("%Y-%m-%d %H:%M"),
        "possessions_version": args.version,
        "rim_override_max_ft": ES.rim_override_for_version(args.version),
        "seasons": SEASONS,
        "era_seasons": ERA_SEASONS,
    }

    if args.check_versions:
        other = "v2" if args.version == "v1" else "v1"
        _, att_other = FT.build_trips_and_attempts(SEASONS, universe=universe, version=other)
        cols = ["season", "game_id", "shooter_id", "made", "trip_id", "trip_pos",
                "trip_len", "foul_class"]
        same = att_other[cols].reset_index(drop=True).equals(
            attempts[cols].reset_index(drop=True))
        report["version_invariance"] = {
            "compared_against": other, "identical": bool(same),
            "n_rows": int(len(attempts)),
        }
        print(f"version invariance vs {other}: {'IDENTICAL' if same else 'DIFFERENT'}")

    # ---------------- FT-1 ----------------
    ft1 = ft1_block(trips_all)
    report["ft1"] = ft1
    era = bonus_era_json(ft1)
    (OUT_DIR / "bonus_era.json").write_text(json.dumps(era, indent=1))
    report["bonus_era"] = era
    print("FT-1 derived thresholds:",
          {k: (v["bonus_prior_fouls"], v["double_bonus_prior_fouls"])
           for k, v in ft1["by_season"].items()})

    # ---------------- FT-2 ----------------
    t0 = time.time()
    design = FT.build_ft_design(attempts)
    print(f"design: {len(design):,} modelled attempts in {time.time() - t0:.1f}s")
    tech = attempts[attempts["foul_class"] == "technical"]
    report["ft2_universe"] = {
        "n_attempts_all": int(len(attempts)),
        "n_modelled": int(len(design)),
        "n_technical_excluded": int(len(tech)),
        "technical_make_rate": round(float(tech["made"].mean()), 5) if len(tech) else None,
        "non_technical_make_rate": round(float(design["y"].mean()), 5),
        "n_shooters": int(design["shooter_id"].nunique()),
        "espn_id_coverage_pct": round(float(design["espn_athlete_id"].notna().mean() * 100), 2),
        "position_coverage_pct": round(float((design["position_group"] != "UNK").mean() * 100), 2),
        "position_shares": {k: round(v * 100, 2) for k, v in
                            design["position_group"].value_counts(normalize=True).items()},
        "by_season": {str(int(s)): {
            "n": int((design["season"] == s).sum()),
            "make_rate": round(float(design.loc[design["season"] == s, "y"].mean()), 5),
            "espn_id_coverage_pct": round(float(
                design.loc[design["season"] == s, "espn_athlete_id"].notna().mean() * 100), 2),
            "position_coverage_pct": round(float(
                (design.loc[design["season"] == s, "position_group"] != "UNK").mean() * 100), 2),
        } for s in SEASONS},
    }

    rows: list[dict] = []
    detail: dict = {}
    preds: dict = {}
    eb_fits: dict = {}
    for fold in ("F1", "F2"):
        tr, te = FT.fold_slices(design, fold)
        print(f"\n{fold}: train {len(tr):,} test {len(te):,}")
        eb = FT.fit_eb(tr)
        eb_fits[fold] = eb
        print(f"  eb grid best: prior={eb['best']['prior']} m={eb['best']['m']} "
              f"(train ll {eb['best']['train_log_loss']})")
        for arm in FT.ARMS:
            s, p = run_arm(arm, tr, te, eb_best=eb["best"])
            detail[f"{fold}|{arm}"] = s
            if fold == FT.SELECTION_FOLD:
                preds[arm] = p
            rows.append(flat_row(s, fold))
            print(f"  {arm:10s} ll={s['log_loss']:.6f} brier={s['brier']:.6f} "
                  f"calib={'PASS' if s['calib_pass'] else 'FAIL'}({s['calib_worst_gap_pp']}) "
                  f"resp={'PASS' if s['resp_pass'] else 'FAIL'} {s['fit_s']}s")
    report["eb_fits"] = {k: {"best": v["best"], "grid": v["grid"]} for k, v in eb_fits.items()}
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "grid_results.csv", index=False)

    tr2, te2 = FT.fold_slices(design, "F2")
    floor_boot = PM.block_bootstrap_se(te2["game_id"].to_numpy(), te2["y"].to_numpy(),
                                       preds["ridge"], n_rep=args.bootstrap_reps)
    seed_ll = []
    for sd in SEED_GRID:
        s, _ = run_arm("lgbm", tr2, te2, seed=sd)
        seed_ll.append(s["log_loss"])
    floor_seed = float(np.std(seed_ll, ddof=1))
    floor = max(floor_boot, floor_seed)
    report["noise_floor"] = {
        "block_bootstrap_se": round(floor_boot, 6),
        "block_bootstrap_reps": args.bootstrap_reps,
        "lgbm_seed_sd": round(floor_seed, 6),
        "lgbm_seed_log_losses": [round(x, 6) for x in seed_ll],
        "floor": round(floor, 6),
    }
    print(f"\nnoise floor: bootstrap {floor_boot:.6f}, lgbm seed SD {floor_seed:.6f} -> {floor:.6f}")

    # --- shrinkage interpretation ------------------------------------------
    best = eb_fits["F2"]["best"]
    m = float(best["m"])
    report["shrinkage"] = {
        "prior": best["prior"],
        "m_attempts": m,
        "attempts_for_50pct_own_weight": m,
        "attempts_for_75pct_own_weight": 3 * m,
        "attempts_for_90pct_own_weight": 9 * m,
        "median_shooter_attempts_at_prediction_time_F2": float(
            np.median(te2["shooter_fta_prior"].to_numpy())),
        "share_of_F2_attempts_with_own_rate_dominant": round(float(
            (te2["shooter_fta_prior"].to_numpy() > m).mean() * 100), 2),
    }

    # --- transfer subset ----------------------------------------------------
    tsub = te2["is_transfer"].to_numpy()
    trans = {}
    for name, mask in (("transfer", tsub), ("non_transfer", ~tsub & (te2["has_prior_season"].to_numpy() > 0)),
                       ("no_prior_season", te2["has_prior_season"].to_numpy() == 0)):
        if mask.sum() == 0:
            continue
        sub = {"n": int(mask.sum())}
        for arm in FT.ARMS:
            sub[f"{arm}_log_loss"] = round(PM.log_loss(te2["y"].to_numpy()[mask], preds[arm][mask]), 6)
        sub["actual_make_rate"] = round(float(te2["y"].to_numpy()[mask].mean()), 5)
        sub["eb_pred_make_rate"] = round(float(preds["eb_shrink"][mask, 1].mean()), 5)
        trans[name] = sub
    report["transfer_check"] = trans

    # --- decision -----------------------------------------------------------
    f2 = [r for r in rows if r["fold"] == "F2"]
    passing = [r for r in f2 if r["calib"] == "PASS" and r["respons"] == "PASS"]
    if passing:
        best_row = min(passing, key=lambda r: r["log_loss"])
        simple = [r for r in passing if r["arm"] in ("team_asof", "eb_shrink", "ridge")]
        best_simple = min(simple, key=lambda r: r["log_loss"]) if simple else None
        verdict = {"winner": best_row["arm"], "log_loss": best_row["log_loss"]}
        if best_row["arm"] == "lgbm" and best_simple is not None:
            gain = best_simple["log_loss"] - best_row["log_loss"]
            verdict["tree_gain_over_simpler"] = round(gain, 6)
            verdict["tree_gain_in_floors"] = round(gain / floor, 2) if floor else None
            if gain <= floor:
                verdict["winner"] = best_simple["arm"]
                verdict["note"] = ("the tree arm's edge is inside the noise floor, so the "
                                   "pre-registered simplicity tie-break selects the simpler arm")
    else:
        verdict = {"winner": None, "note": "no arm passes both pre-registered gates on F2"}
    report["decision"] = verdict
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
        print(f"\nappended results to {DOC}")
    return 0


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


GRID_COLS = ["arm", "n", "log_loss", "brier", "calib", "worst_gap_pp", "level_pp", "shape_pp",
             "respons", "resp_min_steps", "slope_ratio", "bonus_gap_pp", "shooting_gap_pp", "fit_s"]


def render_section(report: dict, grid: pd.DataFrame) -> str:
    v = report["possessions_version"]
    L: list[str] = []
    L.append(f"## 2. Grid configuration (as executed, run {report['run_at']}, "
             f"`scripts/train_free_throw_v1.py --version {v}`)\n")
    L.append("| Dimension | Values |")
    L.append("|---|---|")
    L.append("| FT-1 target | number of attempts in a trip, checked against the rule table "
             "`cbb_sim.models.free_throw.TRIP_RULES` |")
    L.append("| FT-1 foul class | derived from CONTEXT ONLY (technical row / and-one signature / "
             "the fouling team's prior period foul count), never from the attempt count |")
    L.append("| FT-2 target | one free-throw attempt: made vs missed |")
    L.append("| FT-2 arms | `team_asof` (team-level floor), `eb_shrink` (prior and strength FITTED), "
             "`ridge` (logistic ridge), `lgbm` |")
    L.append(f"| FT-2 features | {', '.join(FT.FT_FEATURES)} |")
    L.append("| Folds | F1: train 2022+2023, test 2024. F2: train 2022+2023+2024, test 2025 (selection) |")
    L.append("| Sealed | 2026 -- `assert_not_sealed` on every train and test slice. The FT-1 rule "
             "check reports 2026 DESCRIPTIVELY (a rule check is data, not a fit) and no fold sees it |")
    L.append("| Primary metric | attempt-level log loss on F2 |")
    L.append(f"| Possessions version | `{v}` (rim override {report['rim_override_max_ft']} ft); "
             "nothing in this model depends on it |")
    L.append(f"| Noise floor | linear: {report['noise_floor']['block_bootstrap_reps']}-replicate "
             f"game-level block bootstrap SE; tree: SD over {len(SEED_GRID)} seed-varied refits |")
    L.append("")
    if "version_invariance" in report:
        vi = report["version_invariance"]
        L.append(f"Version invariance check: the attempt table built under `{v}` and under "
                 f"`{vi['compared_against']}` are "
                 f"{'IDENTICAL' if vi['identical'] else '**DIFFERENT**'} over all "
                 f"{vi['n_rows']:,} rows. The rim-location override can only move an `FGA_jump2` "
                 "to an `FGA_rim`, and no free-throw attempt or trip structure reads a field-goal "
                 "class, so this is the assertion rather than the assumption.\n")
    u = report["ft2_universe"]
    L.append(f"FT-2 universe: {u['n_attempts_all']:,} free-throw attempts over "
             f"{report['seasons']}, of which {u['n_modelled']:,} are modelled and "
             f"{u['n_technical_excluded']:,} technical attempts are excluded (their make rate is "
             f"{u['technical_make_rate']} against {u['non_technical_make_rate']} for the rest -- the "
             "shooter on a technical is chosen by the coach, so the two are drawn from different "
             f"shooter distributions). {u['n_shooters']:,} distinct shooters. ESPN athlete id "
             f"resolves on {u['espn_id_coverage_pct']}% of modelled attempts; roster position "
             f"resolves on {u['position_coverage_pct']}%. Runtime {report['runtime_min']} min.\n")

    L.append("---\n")
    L.append("## 3. FT-1: trip structure against the rule\n")
    L.append("### 3.1 Bonus thresholds RE-DERIVED from each season's own data\n")
    L.append("The signature is the one `cbb_sim.pbp.possessions` validated: a one-attempt trip "
             "inside the one-and-one window is a MISSED front end (a made front end earns a second "
             "shot), whereas a one-attempt trip outside the window is an and-one, made about two "
             "thirds of the time. `r(p)` below is the share of one-attempt trips at prior foul "
             "count `p` whose only attempt missed. The onset is the first `p` where it crosses "
             "0.55; the double-bonus onset is the first `p` after that where it falls back.\n")
    rows = []
    for s, d in report["ft1"]["by_season"].items():
        r = d["derivation"]["one_attempt_miss_share_by_prior"]
        rows.append({
            "season": s,
            "bonus onset (prior fouls)": d["bonus_prior_fouls"],
            "double-bonus onset": d["double_bonus_prior_fouls"],
            "r(5)": r.get("5"), "r(6)": r.get("6"), "r(7)": r.get("7"),
            "r(8)": r.get("8"), "r(9)": r.get("9"), "r(10)": r.get("10"),
        })
    L.append(_table(rows, list(rows[0].keys())))
    L.append("")
    era = report["bonus_era"]
    L.append(f"**Era boundary: {'DETECTED' if era['era_boundary_detected'] else 'NOT DETECTED'}** "
             f"({era['n_distinct_threshold_pairs']} distinct threshold pair(s) across "
             f"{report['era_seasons']}). Written to "
             "`data/processed/models/free_throw/bonus_era.json`, which the ENGINE reads into "
             "GameState -- per `CLAUDE.md`, rule-era flags live in GameState and never inside a "
             "fitted sub-model.\n")

    L.append("### 3.2 Free-throw volume by season (the L4 scoring trend, at the trip level)\n")
    rows = [{"season": s, **{k: d[k] for k in
                             ("n_trips", "trips_per_game", "attempts_per_game",
                              "bonus_trips_per_game", "double_bonus_trips_per_game",
                              "one_and_one_share_of_trips_pct")}}
            for s, d in report["ft1"]["by_season"].items()]
    L.append(_table(rows, list(rows[0].keys())))
    L.append("")

    L.append("### 3.3 Rule-derived attempt counts vs the data\n")
    L.append("| season | trips | shooting w/ 1 attempt | and-one w/ >1 | 1-and-1 single attempt "
             "that was MADE | double bonus w/ 1 attempt | >= 4 attempts | violation rate % |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s, d in report["ft1"]["by_season"].items():
        v2 = d["rules"]["violations"]
        L.append(f"| {s} | {d['rules']['n_trips']:,} | "
                 f"{v2['shooting_with_one_attempt']:,} | "
                 f"{v2['and_one_with_more_than_one_attempt']:,} | "
                 f"{v2['one_and_one_single_attempt_that_was_MADE']:,} | "
                 f"{v2['double_bonus_with_one_attempt']:,} | "
                 f"{v2['any_class_with_four_or_more_attempts']:,} | "
                 f"{d['rules']['violation_rate_pct']} |")
    L.append("")
    L.append("Ambiguous mass (NOT violations -- the feed cannot separate a bonus trip from a "
             "two-shot shooting foul once the bonus is in force, data-defect row D5 of the change "
             "ledger):")
    for s, d in report["ft1"]["by_season"].items():
        a = d["rules"]["ambiguous"]
        L.append(f"- {s}: {a['one_and_one_two_attempts']:,} two-attempt trips in the one-and-one, "
                 f"{a['double_bonus_two_attempts']:,} in the double bonus")
    L.append("")

    L.append("---\n")
    L.append("## 4. FT-2: full results\n")
    for fold, label in (("F1", "train [2022, 2023], test [2024]"),
                        ("F2", "train [2022, 2023, 2024], test [2025]  (SELECTION)")):
        sub = grid[grid["fold"] == fold].sort_values("log_loss")
        L.append(f"**{fold}** -- {label}\n")
        L.append(_table(sub.to_dict("records"), GRID_COLS))
        L.append("")
    nf = report["noise_floor"]
    L.append(f"Noise floor: game-block bootstrap SE {nf['block_bootstrap_se']} on `ridge`; "
             f"LightGBM seed-refit SD {nf['lgbm_seed_sd']} over seeds {list(SEED_GRID)} "
             f"({nf['lgbm_seed_log_losses']}). The floor the decision rule uses is the larger, "
             f"**{nf['floor']}**.\n")

    L.append("---\n")
    L.append("## 5. The fitted shrinkage\n")
    for fold in ("F1", "F2"):
        b = report["eb_fits"][fold]["best"]
        L.append(f"- **{fold}**: prior = `{b['prior']}`, strength m = **{b['m']}** "
                 f"pseudo-attempts (train log loss {b['train_log_loss']}).")
    sh = report["shrinkage"]
    L.append("")
    L.append(f"At m = {sh['m_attempts']} pseudo-attempts, a shooter's OWN rate carries half the "
             f"weight once they have {sh['attempts_for_50pct_own_weight']:.0f} attempts on the "
             f"season, three quarters at {sh['attempts_for_75pct_own_weight']:.0f} and nine tenths "
             f"at {sh['attempts_for_90pct_own_weight']:.0f}. The median F2 attempt is taken by a "
             f"shooter with {sh['median_shooter_attempts_at_prediction_time_F2']:.0f} prior "
             f"attempts this season, and {sh['share_of_F2_attempts_with_own_rate_dominant']}% of "
             "F2 attempts are taken by a shooter whose own rate already outweighs the prior.\n")
    L.append("Full grid (train log loss per prior x strength) is in "
             "`data/processed/models/free_throw/run_report.json` under `eb_fits`.\n")

    L.append("### 5.1 Transfer subset\n")
    L.append("The natural experiment L15 used: players whose modal team changed since the prior "
             "season, where a prior-season-based prior is least trustworthy.\n")
    rows = [{"subset": k, **v2} for k, v2 in report["transfer_check"].items()]
    cols = ["subset", "n", "actual_make_rate", "eb_pred_make_rate"] + \
           [f"{a}_log_loss" for a in FT.ARMS]
    L.append(_table(rows, cols))
    L.append("")

    L.append("---\n")
    L.append("## 6. Decision, by the pre-registered rule\n")
    d = report["decision"]
    f2 = [r for r in report["grid_rows"] if r["fold"] == "F2"]
    passing = [r["arm"] for r in f2 if r["calib"] == "PASS" and r["respons"] == "PASS"]
    L.append(f"{len(passing)} of {len(f2)} F2 arms pass BOTH the calibration gate (worst decile "
             "gap <= 2.00 pp on classes with a >= 5% share) and the responsiveness gate "
             "(monotone in at least 3 of the 4 shooter-as-of-FT% quintile steps -- the L3 "
             f"round-1 reading of \"4 of 5 quintile steps\"): {', '.join(f'`{a}`' for a in passing)}.\n")
    if d.get("tree_gain_over_simpler") is not None:
        L.append(f"The tree arm beats the best simpler PASSING arm by "
                 f"{d['tree_gain_over_simpler']} log loss = {d['tree_gain_in_floors']}x the noise "
                 "floor, so the pre-registered \"a tree must beat the simpler arm by more than "
                 "the floor\" clause is satisfied and the simplicity tie-break does not fire.\n")
    if d.get("note"):
        L.append(d["note"] + "\n")
    L.append(f"**WINNER: {d.get('winner') or 'NONE -- no arm passes both gates'}**"
             + (f", F2 log loss {d.get('log_loss')}." if d.get("winner") else ".") + "\n")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
