"""
train_clock_v1.py -- the L5 CLOCK-CONSUMPTION bake-off, exactly the grid
pre-registered in `docs/models/clock/experiments.md` section 1.

    5 arms  x  4 feature sets  x  2 folds

    arms         empirical (Kaplan-Meier state cells)
                 lognormal / gamma (heteroscedastic, censored MLE)
                 hazard    (discrete-time per-second exit, logistic on state)
                 lgbm_quantile (9 quantiles, inverse CDF, linear interpolation)
    feature sets A_state, B_plus_teams, C_plus_score, D_plus_season
    folds        F1 train {2022,2023} test 2024
                 F2 train {2022,2023,2024} test 2025   <- SELECTION

Every arm is scored blind by the same code path (`cbb_sim.models.clock.score_arm`)
on one CRPS definition, then the pre-registered decision rule is applied
mechanically in `decide()`. Nothing in this script chooses a winner by hand.

Artifacts -> `data/processed/models/clock/`
    design.parquet          the one design table every arm shares
    build_diagnostics.json  exclusion counts for the universe filters
    grid_results.csv        one row per (fold, arm, feature set)
    pit_cells_F2.csv        the PIT K-S table by state cell, every F2 arm
    emergent_F2.csv         the emergent-G1 table, every F2 arm
    duration_by_terminal_F2.csv
    noise_floor.json
    verdict.json
    winner.pkl              only written if an arm actually wins

Results are APPENDED to `docs/models/clock/experiments.md` (append-only).

Usage:
    .venv/Scripts/python.exe scripts/train_clock_v1.py
    .venv/Scripts/python.exe scripts/train_clock_v1.py --quick   # smoke path
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.models import clock as ck

OUT_DIR = Path("data/processed/models/clock")
DOC = Path("docs/models/clock/experiments.md")
ALL_SEASONS = [2022, 2023, 2024, 2025]
BASE_SEED = 20260910
NOISE_SEED_OFFSET = 1000


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Fitting / scoring one cell of the grid
# ---------------------------------------------------------------------------
def fit_one(arm_name: str, tr: pd.DataFrame, fs: str, seed: int):
    t0 = time.time()
    arm = ck.fit_arm(arm_name, tr, fs, seed=seed)
    return arm, time.time() - t0


def score_one(arm, te: pd.DataFrame) -> dict:
    s = ck.score_arm(arm, te)
    s["censored_diag"] = ck.censored_survival_diagnostic(arm, te)
    return s


def pit_summary(cells: pd.DataFrame) -> dict:
    powered = cells[cells["powered"]]
    return {
        "n_cells": int(len(cells)),
        "n_powered": int(len(powered)),
        "n_underpowered": int((~cells["powered"]).sum()),
        "pit_leak_failures": int(powered["leak_sized"].sum()),
        "pit_worst_D_powered": round(float(powered["ks_D"].max()), 5) if len(powered) else float("nan"),
        "pit_worst_cell": (powered.loc[powered["ks_D"].idxmax(), "cell"] if len(powered) else ""),
        "pit_median_D_powered": round(float(powered["ks_D"].median()), 5) if len(powered) else float("nan"),
    }


# ---------------------------------------------------------------------------
# The pre-registered decision rule, applied mechanically
# ---------------------------------------------------------------------------
def decide(grid: pd.DataFrame, floor: float) -> dict:
    """winner = lowest F2 CRPS among arms that pass the emergent G1 and carry
    no LEAK-sized PIT failure in any powered cell; a tree arm must beat the
    best non-tree arm by more than the floor; ties (within the floor) go to the
    simpler arm; the feature set is then chosen inside the winning class by the
    same rule. No arm passing the emergent G1 => adopt nothing."""
    f2 = grid[grid["fold"] == "F2"].copy()
    f2["eligible"] = f2["g1_pass"] & (f2["pit_leak_failures"] == 0)
    elig = f2[f2["eligible"]]
    out = {
        "floor": floor,
        "n_arms": int(len(f2)),
        "n_eligible": int(len(elig)),
        "n_pass_g1": int(f2["g1_pass"].sum()),
        "n_pit_clean": int((f2["pit_leak_failures"] == 0).sum()),
    }
    if not len(elig):
        out.update({
            "winner": None,
            "verdict": "NO ARM ADOPTED",
            "reason": ("no (arm, feature set) combination passed the pre-registered emergent G1 "
                       "and PIT gates on F2"),
            "best_crps_overall": float(f2["crps"].min()),
            "best_crps_arm": f2.loc[f2["crps"].idxmin(), "arm"],
            "best_crps_feature_set": f2.loc[f2["crps"].idxmin(), "feature_set"],
        })
        return out

    best_crps = float(elig["crps"].min())
    non_tree = elig[~elig["arm"].isin(ck.TREE_ARMS)]
    best_non_tree = float(non_tree["crps"].min()) if len(non_tree) else np.inf

    # class-level tie-break: every arm whose best CRPS is within the floor of
    # the overall best eligible CRPS is a tie, and the simplest of them wins.
    per_arm = elig.groupby("arm", as_index=False)["crps"].min()
    per_arm["rank"] = per_arm["arm"].map(ck.SIMPLICITY_RANK)
    tied = per_arm[per_arm["crps"] <= best_crps + floor]
    tied = tied.sort_values(["rank", "crps"], kind="stable")
    win_arm = str(tied.iloc[0]["arm"])

    # "a tree arm must beat the best non-tree arm by more than the floor"
    tree_note = ""
    if (win_arm in ck.TREE_ARMS and np.isfinite(best_non_tree)
            and float(per_arm.loc[per_arm["arm"] == win_arm, "crps"].iloc[0]) >= best_non_tree - floor):
        drop = tied[~tied["arm"].isin(ck.TREE_ARMS)]
        tree_note = (f"tree arm did not beat the best non-tree arm by more than the floor "
                     f"({best_non_tree:.5f} - {best_crps:.5f} <= {floor:.5f}); demoted")
        win_arm = str(drop.iloc[0]["arm"]) if len(drop) else str(
            non_tree.sort_values("crps").iloc[0]["arm"])

    inside = elig[elig["arm"] == win_arm].copy()
    fs_order = {f: i for i, f in enumerate(ck.FEATURE_SET_NAMES)}
    inside["fs_rank"] = inside["feature_set"].map(fs_order)
    fs_best = float(inside["crps"].min())
    fs_tied = inside[inside["crps"] <= fs_best + floor].sort_values(["fs_rank", "crps"], kind="stable")
    win_fs = str(fs_tied.iloc[0]["feature_set"])

    row = inside[inside["feature_set"] == win_fs].iloc[0]
    out.update({
        "winner": {"arm": win_arm, "feature_set": win_fs, "crps": float(row["crps"])},
        "verdict": f"ADOPT {win_arm} / {win_fs}",
        "best_eligible_crps": best_crps,
        "best_non_tree_crps": None if not np.isfinite(best_non_tree) else best_non_tree,
        "tree_note": tree_note,
    })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="one feature set, one fold -- a smoke path, not a bake-off")
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()

    # ---- design ----------------------------------------------------------
    design_path = OUT_DIR / "design.parquet"
    _log("building the design table ...")
    design, diag = ck.build_design(ALL_SEASONS)
    design.to_parquet(design_path, index=False)
    (OUT_DIR / "build_diagnostics.json").write_text(json.dumps(diag, indent=2, default=str))
    _log(f"design {len(design):,} rows / {diag['final_games']:,} games -> {design_path}")

    zero_var = {}
    for fs in ck.FEATURE_SET_NAMES:
        tr0, _ = ck.fold_slices(design, "F2")
        _, dropped = ck._drop_zero_variance(tr0, ck.feature_set(fs))
        zero_var[fs] = dropped
    _log(f"zero-variance pre-registered features (dropped, never silently): {zero_var['D_plus_season']}")

    folds = ["F2"] if args.quick else ["F1", "F2"]
    fsets = ["C_plus_score"] if args.quick else list(ck.FEATURE_SET_NAMES)
    arms = list(ck.ARMS)

    rows: list[dict] = []
    detail: dict = {"zero_variance_dropped": zero_var, "build": diag}
    pit_frames: list[pd.DataFrame] = []
    emerg_rows: list[dict] = []
    emerg_months: list[dict] = []
    dbt_frames: list[pd.DataFrame] = []
    fitted_f2: dict[tuple[str, str], object] = {}
    crps_rows_f2: dict[tuple[str, str], np.ndarray] = {}

    for fold in folds:
        tr, te = ck.fold_slices(design, fold)
        _log(f"--- fold {fold}: train {len(tr):,} rows ({sorted(tr.season.unique())}), "
             f"test {len(te):,} rows ({sorted(te.season.unique())})")
        actual_half = ck.actual_end_of_half(te)
        for arm_name in arms:
            for fs in fsets:
                arm, fit_s = fit_one(arm_name, tr, fs, seed=args.seed)
                s = score_one(arm, te)
                cells = ck.pit_by_cell(te, s["pit_row"])
                row = {
                    "fold": fold, "arm": arm_name, "feature_set": fs,
                    "n_train": int(len(tr)), "n_test_scored": s["n_scored"],
                    "n_test_censored_excluded": s["n_censored_excluded"],
                    "fit_seconds": round(fit_s, 2),
                    "crps": round(s["crps"], 6),
                    "logscore_defined": round(s["logscore_defined"], 6),
                    "logscore_undef_pct": s["logscore_undef_pct"],
                    "pred_mean_duration": round(s["pred_mean"], 4),
                    "actual_mean_duration": round(s["actual_mean"], 4),
                    "mean_duration_gap": round(s["pred_mean"] - s["actual_mean"], 4),
                    "censored_mean_pred_survival": round(
                        s["censored_diag"].get("mean_pred_survival", float("nan")), 4),
                    "n_features_used": len(ck.arm_features(arm)),
                    "dropped_zero_variance": ";".join(getattr(arm, "dropped_zero_variance", [])),
                }
                row.update(pit_summary(cells))

                if fold == ck.SELECTION_FOLD:
                    cells2 = cells.copy()
                    cells2.insert(0, "arm", arm_name)
                    cells2.insert(1, "feature_set", fs)
                    pit_frames.append(cells2)

                    res = ck.chain_halves(arm, te, seed=args.seed)
                    rep = ck.emergent_report(res, actual_half)
                    row.update({
                        "g1_sim_mean": rep["sim_mean"], "g1_actual_mean": rep["actual_mean"],
                        "g1_mean_delta": rep["mean_delta"],
                        "g1_sim_sd": rep["sim_sd"], "g1_actual_sd": rep["actual_sd"],
                        "g1_sd_delta": rep["sd_delta"],
                        "g1_count_ks_D": rep["count_ks_D"], "g1_count_ks_p": rep["count_ks_p"],
                        "g1_overall_pass": rep["overall_pass"],
                        "g1_months_powered": rep["n_powered_months"],
                        "g1_months_pass": int(sum(m["pass"] for m in rep["months"] if m["powered"])),
                        "g1_pass": rep["g1_pass"],
                        "eoh_sim_share_lt35": rep["end_of_half"]["sim_share_last_poss_under_35s"],
                        "eoh_actual_share_lt35": rep["end_of_half"]["actual_share_last_poss_under_35s"],
                        "eoh_sim_mean_dur": rep["end_of_half"]["sim_mean_duration_under_35s"],
                        "eoh_actual_mean_dur": rep["end_of_half"]["actual_mean_duration_under_35s"],
                        "chain_wrapped_halves": rep["wrapped_halves"],
                    })
                    emerg_rows.append({"arm": arm_name, "feature_set": fs,
                                       **{k: v for k, v in rep.items() if k not in ("months", "end_of_half")},
                                       **{f"eoh_{k}": v for k, v in rep["end_of_half"].items()}})
                    for m in rep["months"]:
                        emerg_months.append({"arm": arm_name, "feature_set": fs, **m})

                    dbt = ck.duration_by_terminal(te, s["pred_mean_row"], s["pred_m2_row"])
                    dbt.insert(0, "arm", arm_name)
                    dbt.insert(1, "feature_set", fs)
                    dbt_frames.append(dbt)

                    fitted_f2[(arm_name, fs)] = arm
                    crps_rows_f2[(arm_name, fs)] = s["crps_row"]
                else:
                    row.update({k: np.nan for k in (
                        "g1_sim_mean", "g1_actual_mean", "g1_mean_delta", "g1_sim_sd",
                        "g1_actual_sd", "g1_sd_delta", "g1_count_ks_D", "g1_count_ks_p")})
                    row["g1_pass"] = False

                rows.append(row)
                _log(f"  {fold} {arm_name:14s} {fs:14s} fit {fit_s:6.1f}s  "
                     f"CRPS {row['crps']:.4f}  PITfail {row['pit_leak_failures']}/{row['n_powered']}"
                     + (f"  G1 dmean {row.get('g1_mean_delta')} dsd {row.get('g1_sd_delta')}"
                        if fold == ck.SELECTION_FOLD else ""))

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "grid_results.csv", index=False)
    if pit_frames:
        pd.concat(pit_frames, ignore_index=True).to_csv(OUT_DIR / "pit_cells_F2.csv", index=False)
    if emerg_rows:
        pd.DataFrame(emerg_rows).to_csv(OUT_DIR / "emergent_F2.csv", index=False)
        pd.DataFrame(emerg_months).to_csv(OUT_DIR / "emergent_F2_by_month.csv", index=False)
    if dbt_frames:
        pd.concat(dbt_frames, ignore_index=True).to_csv(
            OUT_DIR / "duration_by_terminal_F2.csv", index=False)

    # ---- noise floor -----------------------------------------------------
    _log("noise floor: seed-varied refit of the tree arm; game-block bootstrap for the rest")
    tr2, te2 = ck.fold_slices(design, ck.SELECTION_FOLD)
    floor_detail: dict = {"tree_seed_refit": {}, "block_bootstrap_se": {}}
    tree_deltas = []
    for fs in fsets:
        base = grid[(grid.fold == ck.SELECTION_FOLD) & (grid.arm == "lgbm_quantile")
                    & (grid.feature_set == fs)]
        if not len(base):
            continue
        alt, _ = fit_one("lgbm_quantile", tr2, fs, seed=args.seed + NOISE_SEED_OFFSET)
        s_alt = ck.score_arm(alt, te2)
        d = abs(float(s_alt["crps"]) - float(base["crps"].iloc[0]))
        tree_deltas.append(d)
        floor_detail["tree_seed_refit"][fs] = {
            "crps_seed_base": float(base["crps"].iloc[0]),
            "crps_seed_alt": round(float(s_alt["crps"]), 6), "abs_delta": round(d, 6)}
        _log(f"  tree seed refit {fs:14s} |dCRPS| = {d:.6f}")
    for key, cr in crps_rows_f2.items():
        se = ck.block_bootstrap_se(te2, cr)
        floor_detail["block_bootstrap_se"][f"{key[0]}/{key[1]}"] = round(se, 6)

    tree_floor = float(max(tree_deltas)) if tree_deltas else 0.0
    boot_max = float(max(floor_detail["block_bootstrap_se"].values())) if floor_detail["block_bootstrap_se"] else 0.0
    floor = float(max(tree_floor, boot_max))
    floor_detail.update({"tree_floor": round(tree_floor, 6), "max_block_bootstrap_se": round(boot_max, 6),
                         "floor_used": round(floor, 6)})
    (OUT_DIR / "noise_floor.json").write_text(json.dumps(floor_detail, indent=2))
    _log(f"floor = max(tree seed refit {tree_floor:.6f}, block bootstrap SE {boot_max:.6f}) = {floor:.6f}")

    # ---- decision --------------------------------------------------------
    verdict = decide(grid, floor)
    verdict.update({"created_at": started, "finished_at": datetime.now(UTC).isoformat(),
                    "seed": args.seed, "quick": bool(args.quick),
                    "selection_fold": ck.SELECTION_FOLD})
    (OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    _log(f"VERDICT: {verdict['verdict']}")

    if verdict.get("winner"):
        w = verdict["winner"]
        with open(OUT_DIR / "winner.pkl", "wb") as fh:
            pickle.dump(fitted_f2[(w["arm"], w["feature_set"])], fh)

    detail["noise_floor"] = floor_detail
    detail["verdict"] = verdict
    (OUT_DIR / "metrics_detail.json").write_text(json.dumps(detail, indent=2, default=str))

    write_experiments_section(grid, verdict, floor_detail, diag, zero_var, args)
    _log("done")


# ---------------------------------------------------------------------------
# experiments.md (append-only)
# ---------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isfinite(v):
                vals.append(f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}")
            elif isinstance(v, (bool, np.bool_)):
                vals.append("PASS" if v else "FAIL")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_experiments_section(grid, verdict, floor_detail, diag, zero_var, args) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    f2 = grid[grid.fold == "F2"].sort_values("crps")
    f1 = grid[grid.fold == "F1"].sort_values("crps")
    lines: list[str] = []
    a = lines.append
    a("")
    a(f"## 2. Run R1 -- the full pre-registered grid ({stamp})")
    a("")
    a(f"`scripts/train_clock_v1.py`, seed {args.seed}. Artifacts in "
      "`data/processed/models/clock/`. Every number below is written by that script; "
      "none is typed in by hand.")
    a("")
    a("### 2.1 Universe and exclusions")
    a("")
    a("| item | value |")
    a("|---|---|")
    for k in ("raw_rows", "raw_games", "games_in_universe", "games_cbbd_complete",
              "games_dropped_cbbd_incomplete", "rows_over_duration_cap",
              "rows_over_duration_cap_pct", "censored_rows", "censored_pct",
              "final_rows", "final_games"):
        a(f"| {k} | {diag[k]} |")
    a("")
    a(f"Rows per season: {diag['rows_by_season']}.")
    a("")
    a("`chance number within possession` is a pre-registered A_state feature that is "
      "identically 1 at a POSSESSION's start (a possession begins on its first chance), so it "
      "is a zero-variance column in every feature set and is dropped by every arm rather than "
      f"silently ignored: {zero_var['A_state']}. See `model.md` section 9.")
    a("")
    a("### 2.2 F2 (selection fold: train 2022-2024, test 2025)")
    a("")
    a(_md_table(f2, ["arm", "feature_set", "crps", "logscore_defined", "logscore_undef_pct",
                     "pit_leak_failures", "n_powered", "n_underpowered", "pit_worst_D_powered",
                     "g1_mean_delta", "g1_sd_delta", "g1_months_pass", "g1_months_powered", "g1_pass"],
                ["arm", "features", "CRPS", "log score", "undef %", "PIT fails", "powered cells",
                 "underpowered", "worst K-S D", "G1 d-mean", "G1 d-SD", "months pass",
                 "months powered", "emergent G1"]))
    a("")
    a("### 2.3 F1 (train 2022-2023, test 2024) -- robustness only")
    a("")
    a(_md_table(f1, ["arm", "feature_set", "crps", "logscore_defined", "logscore_undef_pct",
                     "pit_leak_failures", "n_powered", "pit_worst_D_powered"],
                ["arm", "features", "CRPS", "log score", "undef %", "PIT fails", "powered cells",
                 "worst K-S D"]))
    a("")
    a("### 2.4 Noise floor")
    a("")
    a(f"Tree seed-varied refit (seed {args.seed} vs {args.seed + NOISE_SEED_OFFSET}), worst "
      f"|dCRPS| across feature sets: **{floor_detail['tree_floor']:.6f}**. "
      f"Worst game-block bootstrap SE of the mean CRPS across all other arms: "
      f"**{floor_detail['max_block_bootstrap_se']:.6f}**. "
      f"Floor used by the decision rule: **{floor_detail['floor_used']:.6f}**.")
    a("")
    a("### 2.5 Verdict")
    a("")
    a("```json")
    a(json.dumps({k: v for k, v in verdict.items() if k != "months"}, indent=2, default=str))
    a("```")
    a("")
    with DOC.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
