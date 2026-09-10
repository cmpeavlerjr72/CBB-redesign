#!/usr/bin/env python
"""
train_possession_outcome_v1.py -- the pre-registered L3 POSSESSION-OUTCOME
bake-off, run blind over the full grid.

    .venv/Scripts/python.exe scripts/train_possession_outcome_v1.py
    .venv/Scripts/python.exe scripts/train_possession_outcome_v1.py --quick     # 1 fold, no noise floor
    .venv/Scripts/python.exe scripts/train_possession_outcome_v1.py --no-append # do not touch experiments.md

Pre-registration (written BEFORE this script ran):
`docs/models/possession_outcome/experiments.md` section 1.

WHAT IT DOES, in order:
  1. Builds the chance-level design matrix (`cbb_sim.models.possession_outcome.
     build_design`) for seasons 2022-2025. Season 2026 is never requested; the
     seal guard fires on every fold slice regardless.
  2. Fits every (population x fold x arm x feature set) cell of the grid with
     ONE scoring function, so no arm is scored by its own code.
  3. Measures the noise floor: 5 seed-varied refits for the tree arm, a
     200-replicate GAME-BLOCK bootstrap SE for the linear arms.
  4. Applies the pre-registered decision rule mechanically and writes the
     verdict, including which arms failed which gate.
  5. Writes artifacts to `data/processed/models/possession_outcome/` and
     APPENDS every result table to `experiments.md` (append-only, per
     `CLAUDE.md`).

The decision rule is implemented in `decide()` and is a direct transcription of
the pre-registration; it is not re-tuned after seeing the numbers.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

OUT_DIR = Path("data/processed/models/possession_outcome")
EXPERIMENTS_MD = Path("docs/models/possession_outcome/experiments.md")

TRAIN_SEASONS = [2022, 2023, 2024, 2025]

#: Simplicity order for the "ties go to the simpler arm" rule.
ARM_SIMPLICITY = {"baseline": 0, "ridge_logit": 1, "cascade": 2, "lgbm": 3}
FS_SIMPLICITY = {"A_team": 0, "B_plus_season": 1, "C_plus_state": 2, "D_plus_interactions": 3}

LINEAR_ARMS = ("ridge_logit", "cascade")
TREE_ARMS = ("lgbm",)

CAL_GATE_PP = 2.0          # max absolute decile miscalibration, percentage points
CAL_GATE_MIN_SHARE = 5.0   # only classes with >= 5% share are gated
N_SEEDS = 5                # tree-arm noise floor
N_BOOTSTRAP = 200          # linear-arm noise floor


# ---------------------------------------------------------------------------
def arm_feature_sets(arm: str, population: str) -> list[str]:
    """Which feature sets each arm runs. The baseline is matchup-naive so it
    has no feature set; `D_plus_interactions` is defined for the linear arms
    only (the pre-registration: the tree arm finds interactions itself, so
    running it on D would duplicate C)."""
    if arm == "baseline":
        return ["none"]
    if arm in TREE_ARMS:
        return [f for f in PO.FEATURE_SETS if f != "D_plus_interactions"]
    return list(PO.FEATURE_SETS)


def score(te: pd.DataFrame, p: np.ndarray) -> dict:
    """The ONE scoring function every arm goes through."""
    y = te["y"].to_numpy()
    cal = PO.decile_calibration(y, p)
    resp = PO.responsiveness(te, p)
    gated = {c: v for c, v in cal.items() if v["share_pct"] >= CAL_GATE_MIN_SHARE}
    worst = max((v["max_abs_gap_pp"] for v in gated.values()), default=0.0)
    worst_shape = max((v["max_abs_gap_pp_after_level_shift"] for v in gated.values()), default=0.0)
    worst_level = max((abs(v["level_shift_pp"]) for v in gated.values()), default=0.0)
    resp_ok = all(v["pred_monotone_steps"] >= PO.RESPONSIVENESS_MIN_STEPS for v in resp.values())
    return {
        "log_loss": PO.log_loss(y, p),
        "brier": PO.per_class_brier(y, p),
        "calibration": cal,
        "worst_gated_gap_pp": worst,
        "worst_gated_level_pp": worst_level,
        "worst_gated_shape_pp": worst_shape,
        "calibration_pass": bool(worst <= CAL_GATE_PP),
        "responsiveness": resp,
        "responsiveness_pass": bool(resp_ok),
        "by_state": PO.by_state_calibration(te, p),
    }


def run_grid(design: pd.DataFrame, folds: list[str], populations: list[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    detail: dict = {}
    for pop in populations:
        for fold in folds:
            tr, te = PO.fold_slices(design, fold, pop)
            for arm in PO.ARMS:
                for fs in arm_feature_sets(arm, pop):
                    feats = [] if fs == "none" else PO.feature_set(fs, pop)
                    t0 = time.time()
                    model = PO.fit_arm(arm, tr, feats, seed=0)
                    p = PO.predict_arm(arm, model, te, feats)
                    s = score(te, p)
                    dt = time.time() - t0
                    key = f"{pop}|{fold}|{arm}|{fs}"
                    detail[key] = s
                    rows.append({
                        "population": pop, "fold": fold, "arm": arm, "feature_set": fs,
                        "n_train": len(tr), "n_test": len(te),
                        "log_loss": round(s["log_loss"], 6),
                        "worst_gated_gap_pp": round(s["worst_gated_gap_pp"], 3),
                        "worst_gated_level_pp": round(s["worst_gated_level_pp"], 3),
                        "worst_gated_shape_pp": round(s["worst_gated_shape_pp"], 3),
                        "calibration_pass": s["calibration_pass"],
                        "responsiveness_pass": s["responsiveness_pass"],
                        **{f"brier_{c}": round(v, 6) for c, v in s["brier"].items()},
                        "fit_seconds": round(dt, 1),
                    })
                    print(f"  {key:52s} logloss {s['log_loss']:.5f}  "
                          f"cal {'PASS' if s['calibration_pass'] else 'FAIL'} "
                          f"({s['worst_gated_gap_pp']:.2f}pp)  "
                          f"resp {'PASS' if s['responsiveness_pass'] else 'FAIL'}  "
                          f"[{dt:.0f}s]", flush=True)
    return rows, detail


def noise_floor(design: pd.DataFrame, grid: pd.DataFrame, population: str,
                fold: str = PO.SELECTION_FOLD, n_seeds: int = N_SEEDS,
                n_boot: int = N_BOOTSTRAP) -> dict:
    """Tree arm: SD of the fold's log loss over `n_seeds` spec-identical refits.
    Linear arms: game-block bootstrap SE of the same quantity.

    Both answer the same question -- how much of a log-loss difference is
    indistinguishable from re-running the identical spec -- and the winner has
    to clear it (`CLAUDE.md`: "A winner must beat a spec-identical retrain
    under another seed")."""
    tr, te = PO.fold_slices(design, fold, population)
    sub = grid[(grid["population"] == population) & (grid["fold"] == fold)]

    out: dict = {"population": population, "fold": fold}

    tree = sub[sub["arm"].isin(TREE_ARMS)]
    if len(tree):
        best = tree.sort_values("log_loss").iloc[0]
        feats = PO.feature_set(best["feature_set"], population)
        losses = []
        for seed in range(n_seeds):
            m = PO.fit_arm(best["arm"], tr, feats, seed=seed)
            losses.append(PO.log_loss(te["y"].to_numpy(), PO.predict_arm(best["arm"], m, te, feats)))
        out["tree"] = {
            "arm": best["arm"], "feature_set": best["feature_set"],
            "seed_losses": [round(x, 6) for x in losses],
            "mean": round(float(np.mean(losses)), 6),
            "sd": round(float(np.std(losses, ddof=1)), 6),
        }

    lin = sub[sub["arm"].isin(LINEAR_ARMS)]
    if len(lin):
        best = lin.sort_values("log_loss").iloc[0]
        feats = PO.feature_set(best["feature_set"], population)
        m = PO.fit_arm(best["arm"], tr, feats, seed=0)
        p = PO.predict_arm(best["arm"], m, te, feats)
        out["linear"] = {
            "arm": best["arm"], "feature_set": best["feature_set"],
            "log_loss": round(PO.log_loss(te["y"].to_numpy(), p), 6),
            "block_bootstrap_se": round(PO.block_bootstrap_se(te, p, n_rep=n_boot), 6),
            "n_replicates": n_boot,
        }
    return out


def decide(grid: pd.DataFrame, floor: dict, population: str,
           fold: str = PO.SELECTION_FOLD) -> dict:
    """The pre-registered decision rule, transcribed:

      winner = lowest F2 log loss among arms passing per-class calibration
      (max absolute decile miscalibration <= 2 pp on classes with share >= 5%)
      and responsiveness (monotone in 4 of 5 quintile steps for each of 3PA,
      rim, TOV); a tree arm must beat the best linear arm by more than the
      noise floor; ties go to the simpler arm. Feature set is chosen by the
      same rule within the winning class."""
    sub = grid[(grid["population"] == population) & (grid["fold"] == fold)
               & (grid["arm"] != "baseline")].copy()
    sub["arm_rank"] = sub["arm"].map(ARM_SIMPLICITY)
    sub["fs_rank"] = sub["feature_set"].map(FS_SIMPLICITY)
    eligible = sub[sub["calibration_pass"] & sub["responsiveness_pass"]].copy()
    failed = sub[~(sub["calibration_pass"] & sub["responsiveness_pass"])]
    tree_sd = float(floor.get("tree", {}).get("sd", np.nan))
    lin_se = float(floor.get("linear", {}).get("block_bootstrap_se", np.nan))
    nf = float(np.nanmax([tree_sd, lin_se]))
    ranked = sub.sort_values(["log_loss", "arm_rank", "fs_rank"])
    top = ranked.iloc[0]
    verdict: dict = {
        "population": population, "fold": fold,
        "n_arms_scored": int(len(sub)), "n_eligible": int(len(eligible)),
        "noise_floor": {"tree_seed_sd": tree_sd, "linear_block_bootstrap_se": lin_se,
                        "applied": nf},
        "best_by_log_loss_ignoring_gates": {
            "arm": str(top["arm"]), "feature_set": str(top["feature_set"]),
            "log_loss": float(top["log_loss"]),
            "worst_gated_gap_pp": float(top["worst_gated_gap_pp"]),
            "worst_gated_level_pp": float(top["worst_gated_level_pp"]),
            "worst_gated_shape_pp": float(top["worst_gated_shape_pp"]),
        },
        "failed_gates": [
            {"arm": r["arm"], "feature_set": r["feature_set"],
             "calibration_pass": bool(r["calibration_pass"]),
             "responsiveness_pass": bool(r["responsiveness_pass"]),
             "worst_gated_gap_pp": float(r["worst_gated_gap_pp"]),
             "worst_gated_level_pp": float(r["worst_gated_level_pp"]),
             "worst_gated_shape_pp": float(r["worst_gated_shape_pp"]),
             "log_loss": float(r["log_loss"])}
            for _, r in failed.sort_values("log_loss").iterrows()
        ],
    }
    if not len(eligible):
        verdict["winner"] = None
        verdict["reason"] = ("no arm passed both pre-registered gates, so the pre-registration "
                             "yields NO WINNER and nothing is adopted")
        return verdict

    lin = eligible[eligible["arm"].isin(LINEAR_ARMS)].sort_values(
        ["log_loss", "arm_rank", "fs_rank"])
    tree = eligible[eligible["arm"].isin(TREE_ARMS)].sort_values(
        ["log_loss", "arm_rank", "fs_rank"])

    best_lin = lin.iloc[0] if len(lin) else None
    best_tree = tree.iloc[0] if len(tree) else None

    if best_lin is None:
        chosen, why = best_tree, "no linear arm passed the gates"
    elif best_tree is None:
        chosen, why = best_lin, "no tree arm passed the gates"
    else:
        margin = float(best_lin["log_loss"] - best_tree["log_loss"])
        verdict["tree_margin_over_best_linear"] = round(margin, 6)
        if margin > nf:
            chosen = best_tree
            why = (f"tree arm beats the best linear arm by {margin:.5f} log loss, "
                   f"more than the noise floor {nf:.5f}")
        else:
            chosen = best_lin
            why = (f"tree arm's edge over the best linear arm ({margin:.5f}) does not exceed the "
                   f"noise floor ({nf:.5f}); the pre-registration hands the win to the simpler arm")

    # feature set inside the winning class, by the same rule
    same_class = eligible[eligible["arm"] == chosen["arm"]].sort_values(
        ["log_loss", "fs_rank"])
    fs_best = same_class.iloc[0]
    ll_span = float(same_class["log_loss"].max() - same_class["log_loss"].min())
    simplest_within_floor = same_class[
        same_class["log_loss"] <= fs_best["log_loss"] + nf].sort_values("fs_rank").iloc[0]
    verdict["feature_set_ladder"] = [
        {"feature_set": r["feature_set"], "log_loss": float(r["log_loss"])}
        for _, r in same_class.sort_values("fs_rank").iterrows()
    ]
    verdict["feature_set_span"] = round(ll_span, 6)
    verdict["winner"] = {
        "arm": str(fs_best["arm"]),
        "feature_set_lowest_loss": str(fs_best["feature_set"]),
        "feature_set_selected": str(simplest_within_floor["feature_set"]),
        "log_loss": float(simplest_within_floor["log_loss"]),
        "log_loss_lowest": float(fs_best["log_loss"]),
    }
    verdict["reason"] = why
    return verdict


def rejected_state_features(grid: pd.DataFrame, verdict: dict, population: str,
                            fold: str = PO.SELECTION_FOLD) -> dict:
    """'State features that do not reduce log loss beyond the floor are
    recorded as rejected in features.md' -- computed here so features.md can
    quote it instead of asserting it."""
    # When no arm passes the gates there is no winner to adopt, but the
    # feature-block question is still answerable and still belongs in
    # features.md, so fall back to the lowest-loss arm for this comparison and
    # say so.
    arm = (verdict["winner"]["arm"] if verdict.get("winner")
           else verdict.get("best_by_log_loss_ignoring_gates", {}).get("arm"))
    if not arm:
        return {}
    nf = float(verdict["noise_floor"]["applied"])
    fold_rows = grid[(grid["population"] == population) & (grid["fold"] == fold)]
    sub = fold_rows[fold_rows["arm"] == arm].set_index("feature_set")["log_loss"]
    # The tree arm has no D_plus_interactions row by design (the
    # pre-registration: a tree finds interactions itself). The interaction
    # block is still a real question for the linear arms, so it is answered on
    # the best linear arm rather than silently omitted.
    lin = fold_rows[fold_rows["arm"].isin(LINEAR_ARMS)].sort_values("log_loss")
    lin_arm = str(lin.iloc[0]["arm"]) if len(lin) else None
    lin_sub = (fold_rows[fold_rows["arm"] == lin_arm].set_index("feature_set")["log_loss"]
               if lin_arm else None)
    out = {}
    for lo, hi, label in (("A_team", "B_plus_season", "season block"),
                          ("B_plus_season", "C_plus_state", "state block"),
                          ("C_plus_state", "D_plus_interactions", "explicit interaction block")):
        series, on_arm = sub, arm
        if (lo not in sub.index or hi not in sub.index) and lin_sub is not None:
            series, on_arm = lin_sub, lin_arm
        if series is None or lo not in series.index or hi not in series.index:
            continue
        gain = float(series[lo] - series[hi])
        out[label] = {"from": lo, "to": hi, "measured_on_arm": on_arm,
                      "log_loss_gain": round(gain, 6), "noise_floor": round(nf, 6),
                      "verdict": "KEPT" if gain > nf else "REJECTED"}
    return out


# ---------------------------------------------------------------------------
# experiments.md rendering
# ---------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def render_results(grid: pd.DataFrame, detail: dict, floors: dict, verdicts: dict,
                   rejects: dict, run_meta: dict) -> str:
    L: list[str] = []
    A = L.append
    A("")
    A("---")
    A("")
    A(f"## 3. Full results (run {run_meta['run_at']}, `scripts/train_possession_outcome_v1.py`)")
    A("")
    A(f"Design: {run_meta['n_rows']:,} modelled chances over seasons "
      f"{run_meta['seasons']} ({run_meta['n_first']:,} first, {run_meta['n_cont']:,} continuation). "
      f"`end_period` and `unknown` chances excluded per the pre-registration. "
      f"Total grid runtime {run_meta['runtime_min']:.1f} min. "
      f"CSV alongside: `data/processed/models/possession_outcome/grid_results.csv`.")
    A("")
    for pop in sorted(grid["population"].unique(), key=lambda x: 0 if x == "first" else 1):
        A(f"### 3.{1 if pop == 'first' else 2} Population `{pop}`"
          f"{' (the primary metric)' if pop == 'first' else ' (chances after an offensive rebound)'}")
        A("")
        for fold in ("F1", "F2"):
            g = grid[(grid["population"] == pop) & (grid["fold"] == fold)]
            if not len(g):
                continue
            g = g.sort_values(["log_loss"])
            g = g.assign(
                cal=lambda d: np.where(d["calibration_pass"], "PASS", "FAIL"),
                resp=lambda d: np.where(d["responsiveness_pass"], "PASS", "FAIL"),
            )
            A(f"**{fold}** -- train {PO.FOLDS[fold]['train']}, test {PO.FOLDS[fold]['test']}"
              f"{'  (SELECTION)' if fold == PO.SELECTION_FOLD else ''}")
            A("")
            A(_md_table(g, ["arm", "feature_set", "log_loss", "cal", "worst_gated_gap_pp",
                            "worst_gated_level_pp", "worst_gated_shape_pp", "resp",
                            "brier_TOV", "brier_FGA_rim", "brier_FGA_jump2", "brier_FGA_3",
                            "brier_FT_trip_shooting", "brier_FT_trip_bonus", "fit_seconds"],
                        ["arm", "feature set", "log loss", "calib", "worst gap (pp)",
                         "of which level (pp)", "residual shape (pp)", "respons.",
                         "Brier TOV", "Brier rim", "Brier jump2", "Brier 3", "Brier FT-shoot",
                         "Brier FT-bonus", "fit s"]))
            A("")
    A("`worst gap (pp)` is the pre-registered gate quantity: the largest absolute "
      "predicted-minus-actual gap over the ten predicted-probability deciles, maximised over the "
      "classes with a >= 5% share. It is then split, as a DIAGNOSTIC and never as a correction, "
      "into the part that is the same in every decile (`of which level` -- the class's overall rate "
      "is wrong) and the part that varies across deciles (`residual shape` -- the model orders "
      "chances wrongly). They point at different fixes, which is what "
      "`docs/SIM_GUARDRAILS.md` section 5 asks for.")
    A("")
    A("### 3.3 Noise floor")
    A("")
    for pop, f in floors.items():
        A(f"**{pop}**, fold {f['fold']}:")
        A("")
        if "tree" in f:
            A(f"* Tree arm (`{f['tree']['arm']}`, `{f['tree']['feature_set']}`), {N_SEEDS} "
              f"spec-identical refits under different seeds: log loss "
              f"{f['tree']['seed_losses']}, mean {f['tree']['mean']:.6f}, "
              f"**SD {f['tree']['sd']:.6f}**.")
        if "linear" in f:
            A(f"* Linear arm (`{f['linear']['arm']}`, `{f['linear']['feature_set']}`), "
              f"{f['linear']['n_replicates']}-replicate GAME-BLOCK bootstrap of the test set: "
              f"log loss {f['linear']['log_loss']:.6f}, **SE "
              f"{f['linear']['block_bootstrap_se']:.6f}**. The resampling unit is the game, not the "
              f"chance, because chances inside one game share lineups, officials and pace.")
        A("")
    A("### 3.4 Verdict under the pre-registered decision rule")
    A("")
    for pop, v in verdicts.items():
        A(f"**{pop}** (fold {v['fold']}):")
        A("")
        if not v.get("winner"):
            b = v["best_by_log_loss_ignoring_gates"]
            A(f"* Arms scored: {v['n_arms_scored']}; passing both gates: **{v['n_eligible']}**.")
            A(f"* **NO WINNER. {v['reason']}.**")
            A(f"* Noise floor: {v['noise_floor']['applied']:.6f} "
              f"(tree seed SD {v['noise_floor']['tree_seed_sd']:.6f}, linear block-bootstrap SE "
              f"{v['noise_floor']['linear_block_bootstrap_se']:.6f}).")
            A(f"* For reference only, the lowest F2 log loss regardless of the gates was "
              f"`{b['arm']}` + `{b['feature_set']}` at {b['log_loss']:.6f}, with a worst gated "
              f"decile gap of {b['worst_gated_gap_pp']:.2f} pp -- of which {b['worst_gated_level_pp']:.2f} pp "
              f"is a flat level shift and only {b['worst_gated_shape_pp']:.2f} pp is residual shape. "
              f"That decomposition is the finding: the arms order chances well and get the LEVEL of "
              f"the class rates wrong, which is L11 (`docs/LEARNINGS.md`) reappearing one layer down. "
              f"Nothing is adopted on this run and no post-hoc level correction is applied "
              f"(`CLAUDE.md`, standing rule 'no hand tuning on engine output').")
            A("")
            if v["failed_gates"]:
                A("Every arm and why it failed:")
                A("")
                fg = pd.DataFrame(v["failed_gates"])
                fg["calib"] = np.where(fg["calibration_pass"], "PASS", "FAIL")
                fg["respons"] = np.where(fg["responsiveness_pass"], "PASS", "FAIL")
                A(_md_table(fg, ["arm", "feature_set", "log_loss", "calib", "worst_gated_gap_pp",
                                 "worst_gated_level_pp", "worst_gated_shape_pp", "respons"],
                            ["arm", "feature set", "F2 log loss", "calib", "worst gap (pp)",
                             "level (pp)", "shape (pp)", "respons."]))
                A("")
            continue
        w = v["winner"]
        A(f"* Arms scored: {v['n_arms_scored']}; passing both gates: {v['n_eligible']}.")
        A(f"* Noise floor applied: **{v['noise_floor']['applied']:.6f}** "
          f"(tree seed SD {v['noise_floor']['tree_seed_sd']:.6f}, "
          f"linear block-bootstrap SE {v['noise_floor']['linear_block_bootstrap_se']:.6f}; the larger "
          f"of the two is used, which is the conservative choice).")
        if "tree_margin_over_best_linear" in v:
            A(f"* Tree arm's margin over the best linear arm: "
              f"**{v['tree_margin_over_best_linear']:+.6f}** log loss.")
        A(f"* **Winner: `{w['arm']}` + `{w['feature_set_selected']}`, F2 log loss "
          f"{w['log_loss']:.6f}.** {v['reason']}.")
        if w["feature_set_selected"] != w["feature_set_lowest_loss"]:
            A(f"* The lowest-loss feature set inside the winning class was "
              f"`{w['feature_set_lowest_loss']}` ({w['log_loss_lowest']:.6f}), but it does not clear "
              f"the noise floor over `{w['feature_set_selected']}`, so the tie goes to the simpler "
              f"bundle exactly as pre-registered.")
        A("")
        A("Feature-set ladder inside the winning class:")
        A("")
        A(_md_table(pd.DataFrame(v["feature_set_ladder"]), ["feature_set", "log_loss"],
                    ["feature set", "F2 log loss"]))
        A("")
        if v["failed_gates"]:
            A("Arms that failed a gate (excluded from selection before any log loss was compared):")
            A("")
            fg = pd.DataFrame(v["failed_gates"])
            fg["calib"] = np.where(fg["calibration_pass"], "PASS", "FAIL")
            fg["respons"] = np.where(fg["responsiveness_pass"], "PASS", "FAIL")
            A(_md_table(fg.sort_values("log_loss"),
                        ["arm", "feature_set", "log_loss", "calib", "worst_gated_gap_pp", "respons"],
                        ["arm", "feature set", "F2 log loss", "calib", "worst gap (pp)", "respons."]))
        else:
            A("No arm failed a gate.")
        A("")
    A("### 3.5 State and interaction blocks: kept or rejected")
    A("")
    for pop, r in rejects.items():
        if not r:
            continue
        A(f"**{pop}**:")
        A("")
        df = pd.DataFrame([{"block": k, **v} for k, v in r.items()])
        cols = ["block", "from", "to", "log_loss_gain", "noise_floor", "verdict"]
        hdrs = ["block", "from", "to", "F2 log loss gain", "noise floor", "verdict"]
        if "measured_on_arm" in df.columns:
            cols.insert(3, "measured_on_arm")
            hdrs.insert(3, "measured on arm")
        A(_md_table(df, cols, hdrs))
        A("")
    wv = verdicts.get("first", {})
    if wv.get("winner"):
        label = "the winner"
        key = f"first|{PO.SELECTION_FOLD}|{wv['winner']['arm']}|{wv['winner']['feature_set_selected']}"
    elif wv.get("best_by_log_loss_ignoring_gates"):
        b = wv["best_by_log_loss_ignoring_gates"]
        label = f"the lowest-loss arm `{b['arm']}` + `{b['feature_set']}` (NOT ADOPTED -- it failed a gate)"
        key = f"first|{PO.SELECTION_FOLD}|{b['arm']}|{b['feature_set']}"
    else:
        key = None
    A(f"### 3.6 Per-class decile calibration of {label if key else 'the winner'} (F2, `first`)")
    A("")
    if key:
        cal = detail[key]["calibration"]
        rows = [{"class": c, "share %": v["share_pct"], "max abs decile gap (pp)": v["max_abs_gap_pp"],
                 "level shift (pp)": v["level_shift_pp"],
                 "residual shape (pp)": v["max_abs_gap_pp_after_level_shift"],
                 "gated": "yes" if v["share_pct"] >= CAL_GATE_MIN_SHARE else "no (share < 5%)"}
                for c, v in cal.items()]
        A(_md_table(pd.DataFrame(rows), ["class", "share %", "max abs decile gap (pp)",
                                         "level shift (pp)", "residual shape (pp)", "gated"]))
        A("")
        A(f"### 3.7 Responsiveness of {label} (F2, `first`)")
        A("")
        A("The matchup-specific rule: bucket the test chances by the offence's own as-of rate "
          "quintile and check the predicted class share slopes with the actual one instead of "
          "sitting flat at the league mean.")
        A("")
        for spec, v in detail[key]["responsiveness"].items():
            A(f"`{spec}` -- predicted {v['pred_share']}, actual {v['actual_share']}; "
              f"predicted moves with the actual direction in {v['pred_monotone_steps']}/"
              f"{v['n_steps']} steps (gate: >= {PO.RESPONSIVENESS_MIN_STEPS}); "
              f"span predicted {v['span_pred']:+.5f} vs actual {v['span_actual']:+.5f}, "
              f"slope ratio {v['slope_ratio']}.")
        A("")
        A(f"### 3.8 By-state calibration of {label} (F2, `first`)")
        A("")
        bs = detail[key]["by_state"]
        rows = [{"segment": k, "n": v["n"], "log loss": round(v["log_loss"], 5),
                 "max abs gap (pp)": v["max_abs_gap_pp"],
                 "TOV pred/act %": f'{v["TOV_pred_pct"]}/{v["TOV_actual_pct"]}',
                 "rim pred/act %": f'{v["FGA_rim_pred_pct"]}/{v["FGA_rim_actual_pct"]}',
                 "3 pred/act %": f'{v["FGA_3_pred_pct"]}/{v["FGA_3_actual_pct"]}'}
                for k, v in bs.items()]
        A(_md_table(pd.DataFrame(rows),
                    ["segment", "n", "log loss", "max abs gap (pp)", "TOV pred/act %",
                     "rim pred/act %", "3 pred/act %"]))
        A("")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--quick", action="store_true", help="F2 only, first population, no noise floor")
    ap.add_argument("--no-append", action="store_true", help="do not append to experiments.md")
    ap.add_argument("--cache", type=Path, default=None,
                    help="parquet path to cache/reuse the design matrix")
    args = ap.parse_args()

    assert_not_sealed(TRAIN_SEASONS, context="possession_outcome bake-off seasons")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if args.cache and Path(args.cache).exists():
        print(f"reusing design cache {args.cache}")
        design = pd.read_parquet(args.cache)
    else:
        print("building design matrix ...", flush=True)
        design = PO.build_design(TRAIN_SEASONS)
        if args.cache:
            design.to_parquet(args.cache, index=False)
    assert_not_sealed(design, context="possession_outcome design matrix")

    run_meta = {
        "run_at": time.strftime("%Y-%m-%d %H:%M"),
        "seasons": TRAIN_SEASONS,
        "n_rows": int(len(design)),
        "n_first": int((design["population"] == "first").sum()),
        "n_cont": int((design["population"] == "cont").sum()),
    }
    print(f"design: {run_meta['n_rows']:,} chances "
          f"({run_meta['n_first']:,} first / {run_meta['n_cont']:,} continuation)")

    folds = ["F2"] if args.quick else ["F1", "F2"]
    populations = ["first"] if args.quick else ["first", "cont"]

    rows, detail = run_grid(design, folds, populations)
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "grid_results.csv", index=False)

    floors, verdicts, rejects = {}, {}, {}
    if not args.quick:
        for pop in populations:
            print(f"noise floor [{pop}] ...", flush=True)
            floors[pop] = noise_floor(design, grid, pop)
            verdicts[pop] = decide(grid, floors[pop], pop)
            rejects[pop] = rejected_state_features(grid, verdicts[pop], pop)

    run_meta["runtime_min"] = (time.time() - t_start) / 60.0

    # --- artifacts --------------------------------------------------------
    (out_dir / "noise_floor.json").write_text(json.dumps(floors, indent=2, default=str))
    (out_dir / "verdict.json").write_text(json.dumps(verdicts, indent=2, default=str))
    (out_dir / "rejected_blocks.json").write_text(json.dumps(rejects, indent=2, default=str))
    (out_dir / "metrics_detail.json").write_text(json.dumps(detail, indent=2, default=str))
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))

    for pop, v in verdicts.items():
        if v.get("winner"):
            w = v["winner"]
            arm, fs, adopted = w["arm"], w["feature_set_selected"], True
            stem = f"winner_{pop}"
        elif v.get("best_by_log_loss_ignoring_gates"):
            # No arm passed the gates, so nothing is adopted. The lowest-loss
            # arm is still persisted -- clearly labelled NOT ADOPTED -- because
            # the next iteration needs something to diff against, and because a
            # claim in experiments.md should be reproducible from an artifact.
            b = v["best_by_log_loss_ignoring_gates"]
            arm, fs, adopted = b["arm"], b["feature_set"], False
            stem = f"reference_not_adopted_{pop}"
        else:
            continue
        feats = PO.feature_set(fs, pop)
        tr, te = PO.fold_slices(design, PO.SELECTION_FOLD, pop)
        model = PO.fit_arm(arm, tr, feats, seed=0)
        with open(out_dir / f"{stem}.pkl", "wb") as fh:
            pickle.dump({"arm": arm, "feature_set": fs, "features": feats,
                         "classes": PO.CLASSES, "population": pop, "adopted": adopted,
                         "fold": PO.SELECTION_FOLD,
                         "train_seasons": PO.FOLDS[PO.SELECTION_FOLD]["train"],
                         "model": model}, fh)
        p = PO.predict_arm(arm, model, te, feats)
        np.save(out_dir / f"{stem}_F2_pred.npy", p.astype("float32"))
        print(f"wrote {stem}.pkl ({arm} + {fs}; adopted={adopted})")

    if not args.no_append:
        md = render_results(grid, detail, floors, verdicts, rejects, run_meta)
        with open(EXPERIMENTS_MD, "a", encoding="utf-8") as fh:
            fh.write(md)
        print(f"appended results to {EXPERIMENTS_MD}")
    print(f"done in {run_meta['runtime_min']:.1f} min")


if __name__ == "__main__":
    main()
