#!/usr/bin/env python
"""
train_possession_outcome_v2.py -- ROUND 2 of the pre-registered L3
POSSESSION-OUTCOME bake-off, run blind over the round-2 grid.

    .venv/Scripts/python.exe scripts/train_possession_outcome_v2.py
    .venv/Scripts/python.exe scripts/train_possession_outcome_v2.py --quick
    .venv/Scripts/python.exe scripts/train_possession_outcome_v2.py --no-append

Pre-registration (written to `docs/models/possession_outcome/experiments.md`
BEFORE this script ran): section "Round 2 pre-registration (PM-authored,
2026-09-10)".

WHAT CHANGES FROM ROUND 1, and nothing else does:

  (i)   the EVENT LAYER is corrected -- the rim-location override
        (`cbb_sim.pbp.events`) and first-chance-only style rates
        (`cbb_sim.models.possession_outcome.STYLE_SOURCES`). Both are data
        fixes upstream of the model, not adjustments to its output.
  (ii)  the UNIVERSE is restricted to `pbp_complete` games
        (`cbb_sim.data.universe`).
  (iii) a TRAINING-SCHEME dimension is added: S0 static, S1 in-season
        walk-forward, S2 exponential recency weighting
        (`cbb_sim.models.possession_outcome.SCHEMES`).

Feature set is `C_plus_state` for every arm -- round 1's best for every model
class -- and A / B / D are not rerun.

GRADING IS LITERALLY UNCHANGED. `score()` is imported from
`train_possession_outcome_v1` rather than reimplemented, so "metrics and gates
unchanged from round 1" is enforced by the call graph instead of asserted in
prose. The decision rule is transcribed below with one addition the round-1
rule could not have: a tie-break over the training schemes.

CHECKPOINTING. Every unit of work -- each S2 half-life, each grid cell, each
population's noise floor -- is written to `round2/checkpoint.json` as soon as it
finishes, and `--resume` (the default) skips anything already in it. This is not
convenience: two earlier attempts at this run were killed mid-LightGBM by the
environment with no Python traceback and no stderr, after 45 and 15 minutes of
work respectively, and the cost of the tree arm's S1 cells plus its 5 seed-varied
refits is measured in hours. A resumed run recomputes nothing and fabricates
nothing -- the checkpoint holds finished measurements only, keyed by
`population|fold|arm|scheme`, and `--no-resume` forces a clean run.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import train_possession_outcome_v1 as R1  # noqa: E402

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

OUT_DIR = Path("data/processed/models/possession_outcome/round2")
EXPERIMENTS_MD = Path("docs/models/possession_outcome/experiments.md")

TRAIN_SEASONS = [2022, 2023, 2024, 2025]
FEATURE_SET = "C_plus_state"
ROUND2_ARMS = ("ridge_logit", "cascade", "lgbm")
POSSESSIONS_VERSION = "v2"

#: Simplicity order for the "ties go to the simpler arm" rule. The scheme rank
#: is the round-2 addition: S0 is one static fit, S2 is one weighted fit (a
#: single extra fitted scalar, and that scalar is fitted on F1, not F2), S1 is
#: a fit per month of the test season and needs the pipeline to rerun in
#: production every month. That is a genuine ordering of complexity, not a
#: preference.
SCHEME_SIMPLICITY = {"S0": 0, "S2": 1, "S1": 2}


# ---------------------------------------------------------------------------
class Checkpoint:
    """Finished measurements, on disk, keyed so a resumed run can skip them.

    Holds only completed units of work. Nothing in here is ever derived from
    anything else in here, so a partial checkpoint is a partial RUN rather than
    an inconsistent one."""

    def __init__(self, path: Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self.d: dict = {"half_lives": {}, "grid": {}, "detail": {}, "scheme_meta": {}, "floors": {}}
        if enabled and self.path.exists():
            loaded = json.loads(self.path.read_text())
            self.d.update({k: loaded.get(k, v) for k, v in self.d.items()})
            print(f"resuming from {self.path}: {len(self.d['half_lives'])} half-lives, "
                  f"{len(self.d['grid'])} grid cells, {len(self.d['floors'])} noise floors",
                  flush=True)

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.d, indent=1, default=str))
        os.replace(tmp, self.path)

    # -- half-lives ---------------------------------------------------------
    def half_life(self, pop: str, arm: str):
        return self.d["half_lives"].get(f"{pop}|{arm}")

    def put_half_life(self, pop: str, arm: str, value: dict) -> None:
        self.d["half_lives"][f"{pop}|{arm}"] = value
        self.save()

    def half_lives_nested(self) -> dict:
        out: dict = {}
        for k, v in self.d["half_lives"].items():
            pop, arm = k.split("|")
            out.setdefault(pop, {})[arm] = v
        return out

    # -- grid ---------------------------------------------------------------
    def cell(self, key: str):
        return self.d["grid"].get(key)

    def put_cell(self, key: str, row: dict, detail: dict, meta: dict) -> None:
        self.d["grid"][key] = row
        self.d["detail"][key] = detail
        self.d["scheme_meta"][key] = meta
        self.save()

    # -- noise floor --------------------------------------------------------
    def floor(self, pop: str):
        return self.d["floors"].get(pop)

    def put_floor(self, pop: str, value: dict) -> None:
        self.d["floors"][pop] = value
        self.save()


#: Columns the grid, the gates and the noise floor actually read. The design
#: carries another twenty (raw attempt counts, clocks, terminal labels) that no
#: arm and no metric touches, and carrying them through the fold slices and the
#: S1 monthly concatenations costs gigabytes for nothing. Dropping them cannot
#: change a number -- the list below is the union of every feature set in play,
#: the target, and the keys the metrics group by.
def design_columns() -> list[str]:
    cols = set(PO.feature_set(FEATURE_SET, "first")) | set(PO.feature_set(FEATURE_SET, "cont"))
    cols |= {f for f, _c in PO.RESPONSIVENESS_SPECS}
    cols |= {"y", "season", "game_id", "game_date", "population"}
    return sorted(cols)


def build_or_load_design(cache: Path | None) -> pd.DataFrame:
    if cache and Path(cache).exists():
        print(f"reusing design cache {cache}")
        d = pd.read_parquet(cache)
    else:
        print("building design matrix (possessions v2, first-chance style rates, "
              "pbp_complete universe) ...", flush=True)
        d = PO.build_design(
            TRAIN_SEASONS,
            version=POSSESSIONS_VERSION,
            style_source="first_chance",
            require_pbp_complete=True,
        )
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            d.to_parquet(cache, index=False)
    return d[design_columns()]


def fit_half_lives(design: pd.DataFrame, populations: list[str],
                   ckpt: Checkpoint) -> dict:
    """S2's half-life, fitted on F1 only, per (population, model class).

    Per model class rather than once globally because the schemes are
    per-class arms and a half-life is a property of how a given estimator
    weights its history; per population because the pre-registration fits the
    two populations separately throughout."""
    out: dict[str, dict] = {}
    for pop in populations:
        feats = PO.feature_set(FEATURE_SET, pop)
        out[pop] = {}
        for arm in ROUND2_ARMS:
            done = ckpt.half_life(pop, arm)
            if done is not None:
                out[pop][arm] = done
                print(f"  half-life [{pop}|{arm}] -> {done['half_life_days']}d  (from checkpoint)",
                      flush=True)
                continue
            t0 = time.time()
            h = PO.fit_half_life(arm, feats, design, pop, fold="F1")
            out[pop][arm] = h
            ckpt.put_half_life(pop, arm, h)
            print(f"  half-life [{pop}|{arm}] -> {h['half_life_days']}d  "
                  f"F1 grid {h['grid_log_loss']}  [{time.time() - t0:.0f}s]", flush=True)
    return out


def run_grid(design: pd.DataFrame, folds: list[str], populations: list[str],
             half_lives: dict, schemes: list[str],
             ckpt: Checkpoint) -> tuple[list[dict], dict, dict]:
    rows: list[dict] = []
    detail: dict = {}
    scheme_meta: dict = {}
    for pop in populations:
        feats = PO.feature_set(FEATURE_SET, pop)
        for fold in folds:
            tr, te = PO.fold_slices(design, fold, pop)
            # the matchup-naive floor, S0 only. It is NOT a round-2 arm (the
            # pre-registration lists three model classes) and `decide_v2`
            # excludes it exactly as round 1 did; it is carried because a grid
            # with no floor cannot say how much of the log loss is matchup
            # information at all.
            for arm in ("baseline", *ROUND2_ARMS):
                for scheme in (["S0"] if arm == "baseline" else schemes):
                    key = f"{pop}|{fold}|{arm}|{scheme}"
                    cached = ckpt.cell(key)
                    if cached is not None:
                        rows.append(cached)
                        detail[key] = ckpt.d["detail"][key]
                        scheme_meta[key] = ckpt.d["scheme_meta"][key]
                        print(f"  {key:34s} logloss {cached['log_loss']:.5f}  (from checkpoint)",
                              flush=True)
                        continue
                    hl = None
                    if scheme == "S2":
                        hl = half_lives[pop][arm]["half_life_days"]
                    t0 = time.time()
                    if arm == "baseline":
                        model = PO.fit_arm(arm, tr, [], seed=0)
                        p = PO.predict_arm(arm, model, te, [])
                        meta = {"scheme": "S0", "n_fits": 1}
                    else:
                        p, meta = PO.fit_predict_scheme(
                            arm, scheme, tr, te, feats, seed=0, half_life_days=hl)
                    s = R1.score(te, p)
                    dt = time.time() - t0
                    detail[key] = s
                    scheme_meta[key] = meta
                    row = {
                        "population": pop, "fold": fold, "arm": arm,
                        "feature_set": "none" if arm == "baseline" else FEATURE_SET,
                        "scheme": scheme,
                        "half_life_days": hl if hl is not None else "",
                        "n_fits": meta.get("n_fits", 1),
                        "n_train": len(tr), "n_test": len(te),
                        "log_loss": round(s["log_loss"], 6),
                        "worst_gated_gap_pp": round(s["worst_gated_gap_pp"], 3),
                        "worst_gated_level_pp": round(s["worst_gated_level_pp"], 3),
                        "worst_gated_shape_pp": round(s["worst_gated_shape_pp"], 3),
                        "calibration_pass": s["calibration_pass"],
                        "responsiveness_pass": s["responsiveness_pass"],
                        **{f"brier_{c}": round(v, 6) for c, v in s["brier"].items()},
                        "fit_seconds": round(dt, 1),
                    }
                    rows.append(row)
                    ckpt.put_cell(key, row, s, meta)
                    print(f"  {key:34s} logloss {s['log_loss']:.5f}  "
                          f"cal {'PASS' if s['calibration_pass'] else 'FAIL'} "
                          f"({s['worst_gated_gap_pp']:.2f}pp)  "
                          f"resp {'PASS' if s['responsiveness_pass'] else 'FAIL'}  "
                          f"[{dt:.0f}s]", flush=True)
    return rows, detail, scheme_meta


def noise_floor(design: pd.DataFrame, grid: pd.DataFrame, population: str,
                half_lives: dict, fold: str = PO.SELECTION_FOLD,
                n_seeds: int = R1.N_SEEDS, n_boot: int = R1.N_BOOTSTRAP) -> dict:
    """Unchanged from round 1: the tree arm's SD over spec-identical refits
    under different seeds, and a game-block bootstrap SE for the linear arms.
    The only round-2 addition is that the refits reproduce the arm's SCHEME as
    well as its feature set -- a seed-varied refit of an S1 arm has to redo
    every monthly refit, or it would be measuring a different spec."""
    tr, te = PO.fold_slices(design, fold, population)
    feats = PO.feature_set(FEATURE_SET, population)
    sub = grid[(grid["population"] == population) & (grid["fold"] == fold)
               & (grid["arm"] != "baseline")]
    out: dict = {"population": population, "fold": fold}

    tree = sub[sub["arm"].isin(R1.TREE_ARMS)]
    if len(tree):
        best = tree.sort_values("log_loss").iloc[0]
        hl = half_lives[population][best["arm"]]["half_life_days"] if best["scheme"] == "S2" else None
        losses = []
        for seed in range(n_seeds):
            p, _ = PO.fit_predict_scheme(best["arm"], best["scheme"], tr, te, feats,
                                         seed=seed, half_life_days=hl)
            losses.append(PO.log_loss(te["y"].to_numpy(), p))
        out["tree"] = {"arm": best["arm"], "feature_set": FEATURE_SET, "scheme": best["scheme"],
                       "n_seeds": int(n_seeds), "seed_losses": [round(x, 6) for x in losses],
                       "mean": round(float(np.mean(losses)), 6) if losses else None,
                       "sd": (round(float(np.std(losses, ddof=1)), 6) if len(losses) > 1 else None),
                       "complete": bool(n_seeds >= R1.N_SEEDS),
                       "partial_reason": (None if n_seeds >= R1.N_SEEDS else
                                          f"only {n_seeds} of {R1.N_SEEDS} pre-registered "
                                          f"seed-varied refits were run")}
        if n_seeds < R1.N_SEEDS:
            out["partial"] = True

    lin = sub[sub["arm"].isin(R1.LINEAR_ARMS)]
    if len(lin):
        best = lin.sort_values("log_loss").iloc[0]
        hl = half_lives[population][best["arm"]]["half_life_days"] if best["scheme"] == "S2" else None
        p, _ = PO.fit_predict_scheme(best["arm"], best["scheme"], tr, te, feats,
                                     seed=0, half_life_days=hl)
        out["linear"] = {"arm": best["arm"], "feature_set": FEATURE_SET, "scheme": best["scheme"],
                         "log_loss": round(PO.log_loss(te["y"].to_numpy(), p), 6),
                         "block_bootstrap_se": round(PO.block_bootstrap_se(te, p, n_rep=n_boot), 6),
                         "n_replicates": n_boot}
    return out


def decide_v2(grid: pd.DataFrame, floor: dict, population: str,
              fold: str = PO.SELECTION_FOLD) -> dict:
    """The pre-registered decision rule, transcribed -- identical to round 1
    (`train_possession_outcome_v1.decide`) except that the tie-break ladder
    gains the training scheme, since the feature-set dimension is now a
    constant and the scheme dimension is new:

      winner = lowest F2 log loss among arms passing per-class calibration
      (max absolute decile miscalibration <= 2 pp on classes with share >= 5%)
      and responsiveness (monotone in 4 of 5 quintile steps for each of 3PA,
      rim, TOV); a tree arm must beat the best linear arm by more than the
      noise floor; ties go to the simpler arm, and among equally simple arms to
      the simpler training scheme."""
    sub = grid[(grid["population"] == population) & (grid["fold"] == fold)
               & (grid["arm"] != "baseline")].copy()
    sub["arm_rank"] = sub["arm"].map(R1.ARM_SIMPLICITY)
    sub["scheme_rank"] = sub["scheme"].map(SCHEME_SIMPLICITY)
    eligible = sub[sub["calibration_pass"] & sub["responsiveness_pass"]].copy()
    failed = sub[~(sub["calibration_pass"] & sub["responsiveness_pass"])]
    # A missing tree SD (a PARTIAL floor) must not silently become 0 and it must
    # not become NaN either: `nanmax` falls back to whichever component exists,
    # and `noise_floor_partial` records that the applied number is a lower
    # bound on the pre-registered floor rather than the floor itself.
    tsd = floor.get("tree", {}).get("sd")
    tree_sd = float(tsd) if tsd is not None else float("nan")
    lse = floor.get("linear", {}).get("block_bootstrap_se")
    lin_se = float(lse) if lse is not None else float("nan")
    nf = float(np.nanmax([tree_sd, lin_se]))
    order = ["log_loss", "arm_rank", "scheme_rank"]
    top = sub.sort_values(order).iloc[0]
    verdict: dict = {
        "population": population, "fold": fold,
        "n_arms_scored": int(len(sub)), "n_eligible": int(len(eligible)),
        "noise_floor": {"tree_seed_sd": tree_sd, "linear_block_bootstrap_se": lin_se,
                        "applied": nf, "partial": bool(floor.get("partial", False)),
                        "partial_reason": floor.get("tree", {}).get("partial_reason")},
        "best_by_log_loss_ignoring_gates": {
            "arm": str(top["arm"]), "scheme": str(top["scheme"]),
            "feature_set": str(top["feature_set"]),
            "log_loss": float(top["log_loss"]),
            "worst_gated_gap_pp": float(top["worst_gated_gap_pp"]),
            "worst_gated_level_pp": float(top["worst_gated_level_pp"]),
            "worst_gated_shape_pp": float(top["worst_gated_shape_pp"]),
        },
        "failed_gates": [
            {"arm": r["arm"], "scheme": r["scheme"], "feature_set": r["feature_set"],
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

    lin = eligible[eligible["arm"].isin(R1.LINEAR_ARMS)].sort_values(order)
    tree = eligible[eligible["arm"].isin(R1.TREE_ARMS)].sort_values(order)
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
            why = (f"tree arm beats the best linear arm by {margin:.5f} log loss, more than the "
                   f"noise floor {nf:.5f}")
        else:
            chosen = best_lin
            why = (f"tree arm's edge over the best linear arm ({margin:.5f}) does not exceed the "
                   f"noise floor ({nf:.5f}); the pre-registration hands the win to the simpler arm")

    same_class = eligible[eligible["arm"] == chosen["arm"]].sort_values(order)
    best_cell = same_class.iloc[0]
    simplest_within_floor = same_class[
        same_class["log_loss"] <= best_cell["log_loss"] + nf].sort_values("scheme_rank").iloc[0]
    verdict["scheme_ladder"] = [
        {"scheme": r["scheme"], "log_loss": float(r["log_loss"]),
         "calibration_pass": bool(r["calibration_pass"])}
        for _, r in eligible[eligible["arm"] == chosen["arm"]].sort_values("scheme_rank").iterrows()
    ]
    verdict["winner"] = {
        "arm": str(best_cell["arm"]),
        "feature_set": FEATURE_SET,
        "scheme_lowest_loss": str(best_cell["scheme"]),
        "scheme_selected": str(simplest_within_floor["scheme"]),
        "half_life_days": (simplest_within_floor["half_life_days"]
                           if simplest_within_floor["scheme"] == "S2" else None),
        "log_loss": float(simplest_within_floor["log_loss"]),
        "log_loss_lowest": float(best_cell["log_loss"]),
    }
    verdict["reason"] = why
    return verdict


def scheme_ladder_all(grid: pd.DataFrame, population: str, nf: float,
                      fold: str = PO.SELECTION_FOLD) -> list[dict]:
    """Scheme-vs-S0 log-loss gain for EVERY model class, gated or not.

    The pre-registration's question ("does the fit need to follow the season?")
    is a question about the schemes, and it has an answer whether or not any
    arm clears the calibration gate. Reported for all three classes so the
    answer is not read off a single arm."""
    sub = grid[(grid["population"] == population) & (grid["fold"] == fold)
               & (grid["arm"] != "baseline")]
    out = []
    for arm in ROUND2_ARMS:
        a = sub[sub["arm"] == arm].set_index("scheme")["log_loss"]
        if "S0" not in a.index:
            continue
        for scheme in ("S1", "S2"):
            if scheme not in a.index:
                continue
            gain = float(a["S0"] - a[scheme])
            out.append({"arm": arm, "scheme": scheme, "S0_log_loss": float(a["S0"]),
                        "scheme_log_loss": float(a[scheme]), "gain_vs_S0": round(gain, 6),
                        "noise_floor": round(nf, 6),
                        "verdict": "BETTER" if gain > nf else
                                   ("WORSE" if gain < -nf else "INSIDE THE FLOOR")})
    return out


# ---------------------------------------------------------------------------
# experiments.md rendering
# ---------------------------------------------------------------------------
def render_results(grid: pd.DataFrame, detail: dict, floors: dict, verdicts: dict,
                   half_lives: dict, ladders: dict, scheme_meta: dict, run_meta: dict) -> str:
    T = R1._md_table
    L: list[str] = []
    A = L.append
    A("")
    A("---")
    A("")
    A(f"## 5. Round 2 full results (run {run_meta['run_at']}, "
      f"`scripts/train_possession_outcome_v2.py`)")
    A("")
    A(f"Design: {run_meta['n_rows']:,} modelled chances over seasons {run_meta['seasons']} "
      f"({run_meta['n_first']:,} first, {run_meta['n_cont']:,} continuation), built from "
      f"possessions **{run_meta['possessions_version']}**, style rates from "
      f"**{run_meta['style_source']}**, universe restricted to **pbp_complete** games. "
      f"Round 1's design over the same seasons carried {run_meta['round1_n_rows']:,} chances, so "
      f"the completeness restriction and the event-layer rebuild together cost "
      f"{100 * (1 - run_meta['n_rows'] / run_meta['round1_n_rows']):.1f}% of the rows. "
      f"Total grid runtime {run_meta['runtime_min']:.1f} min. CSV alongside: "
      f"`{OUT_DIR.as_posix()}/grid_results.csv`.")
    A("")
    A("`baseline` is the matchup-naive floor and is NOT one of the three pre-registered model "
      "classes; it is carried for scale and excluded from selection, exactly as in round 1.")
    A("")
    A("### 5.1 S2's half-life, fitted on F1 only")
    A("")
    A("The pre-registration fits the half-life on F1 and applies it to F2 unchanged, so the "
      "selection fold never sees it chosen. Every grid point is shown, not just the argmin:")
    A("")
    rows = []
    for pop, per_arm in half_lives.items():
        for arm, h in per_arm.items():
            rows.append({"population": pop, "arm": arm,
                         **{f"{k}d": f"{v:.6f}" for k, v in h["grid_log_loss"].items()},
                         "chosen": f'**{h["half_life_days"]}d**',
                         "grid spread": f'{h["spread"]:.6f}'})
    A(T(pd.DataFrame(rows), list(rows[0].keys())))
    A("")
    for pop in sorted(grid["population"].unique(), key=lambda x: 0 if x == "first" else 1):
        idx = 2 if pop == "first" else 3
        A(f"### 5.{idx} Population `{pop}`"
          f"{' (the primary metric)' if pop == 'first' else ' (chances after an offensive rebound)'}")
        A("")
        for fold in ("F1", "F2"):
            g = grid[(grid["population"] == pop) & (grid["fold"] == fold)]
            if not len(g):
                continue
            g = g.sort_values(["log_loss"]).assign(
                cal=lambda d: np.where(d["calibration_pass"], "PASS", "FAIL"),
                resp=lambda d: np.where(d["responsiveness_pass"], "PASS", "FAIL"),
                hl=lambda d: d["half_life_days"].astype(str),
            )
            A(f"**{fold}** -- train {PO.FOLDS[fold]['train']}, test {PO.FOLDS[fold]['test']}"
              f"{'  (SELECTION)' if fold == PO.SELECTION_FOLD else ''}")
            A("")
            A(T(g, ["arm", "scheme", "hl", "n_fits", "log_loss", "cal", "worst_gated_gap_pp",
                    "worst_gated_level_pp", "worst_gated_shape_pp", "resp",
                    "brier_TOV", "brier_FGA_rim", "brier_FGA_jump2", "brier_FGA_3",
                    "brier_FT_trip_shooting", "brier_FT_trip_bonus", "fit_seconds"],
                  ["arm", "scheme", "half-life", "fits", "log loss", "calib", "worst gap (pp)",
                   "of which level (pp)", "residual shape (pp)", "respons.",
                   "Brier TOV", "Brier rim", "Brier jump2", "Brier 3", "Brier FT-shoot",
                   "Brier FT-bonus", "fit s"]))
            A("")
    A("### 5.4 Noise floor")
    A("")
    for pop, f in floors.items():
        A(f"**{pop}**, fold {f['fold']}:")
        A("")
        if "tree" in f and f["tree"].get("sd") is not None:
            A(f"* Tree arm (`{f['tree']['arm']}` + `{f['tree']['feature_set']}`, scheme "
              f"`{f['tree']['scheme']}`), {f['tree'].get('n_seeds', R1.N_SEEDS)} spec-identical "
              f"refits under different seeds (each refit reproducing the scheme in full): log loss "
              f"{f['tree']['seed_losses']}, mean {f['tree']['mean']:.6f}, "
              f"**SD {f['tree']['sd']:.6f}**"
              + ("" if f["tree"].get("complete", True) else
                 f" -- **PARTIAL: {f['tree']['partial_reason']}**") + ".")
        elif "tree" in f:
            A(f"* Tree arm (`{f['tree']['arm']}` + `{f['tree']['feature_set']}`, scheme "
              f"`{f['tree']['scheme']}`): **NOT MEASURED -- PARTIAL.** "
              f"{f['tree']['partial_reason']}. The pre-registered quantity is the SD of "
              f"{R1.N_SEEDS} spec-identical refits, and for an S1 arm each refit replays every "
              f"monthly refit of the test season, so this is the single most expensive number in "
              f"the run. The floor applied below is therefore the LINEAR component alone, which is "
              f"a lower bound on the pre-registered floor. Command to finish it is in section 5.10.")
        if "linear" in f:
            A(f"* Linear arm (`{f['linear']['arm']}` + `{f['linear']['feature_set']}`, scheme "
              f"`{f['linear']['scheme']}`), {f['linear']['n_replicates']}-replicate GAME-BLOCK "
              f"bootstrap of the test set: log loss {f['linear']['log_loss']:.6f}, **SE "
              f"{f['linear']['block_bootstrap_se']:.6f}**.")
        A("")
    A("### 5.5 Does the fit need to follow the season? (scheme vs S0, every model class)")
    A("")
    A("The question the scheme dimension was added to answer, reported for all three model classes "
      "on F2 regardless of whether any of them clears a gate -- a scheme effect read off one arm "
      "would be an arm effect:")
    A("")
    for pop, lad in ladders.items():
        if not lad:
            continue
        A(f"**{pop}**:")
        A("")
        A(T(pd.DataFrame(lad), ["arm", "scheme", "S0_log_loss", "scheme_log_loss", "gain_vs_S0",
                                "noise_floor", "verdict"],
            ["arm", "scheme", "S0 F2 log loss", "scheme F2 log loss", "gain vs S0", "noise floor",
             "verdict"]))
        A("")
    A("### 5.6 Verdict under the pre-registered decision rule")
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
              f"`{b['arm']}` + `{b['scheme']}` at {b['log_loss']:.6f}, with a worst gated decile gap "
              f"of {b['worst_gated_gap_pp']:.2f} pp -- of which {b['worst_gated_level_pp']:.2f} pp is "
              f"a flat level shift and {b['worst_gated_shape_pp']:.2f} pp is residual shape. Nothing "
              f"is adopted on this run and no post-hoc level correction is applied (`CLAUDE.md`, "
              f"standing rule 'no hand tuning on engine output').")
            A("")
            if v["failed_gates"]:
                A("Every arm and why it failed:")
                A("")
                fg = pd.DataFrame(v["failed_gates"])
                fg["calib"] = np.where(fg["calibration_pass"], "PASS", "FAIL")
                fg["respons"] = np.where(fg["responsiveness_pass"], "PASS", "FAIL")
                A(T(fg, ["arm", "scheme", "log_loss", "calib", "worst_gated_gap_pp",
                         "worst_gated_level_pp", "worst_gated_shape_pp", "respons"],
                      ["arm", "scheme", "F2 log loss", "calib", "worst gap (pp)", "level (pp)",
                       "shape (pp)", "respons."]))
                A("")
            continue
        w = v["winner"]
        A(f"* Arms scored: {v['n_arms_scored']}; passing both gates: {v['n_eligible']}.")
        A(f"* Noise floor applied: **{v['noise_floor']['applied']:.6f}** (the larger of the tree "
          f"seed SD {v['noise_floor']['tree_seed_sd']:.6f} and the linear block-bootstrap SE "
          f"{v['noise_floor']['linear_block_bootstrap_se']:.6f}, which is the conservative choice).")
        if "tree_margin_over_best_linear" in v:
            A(f"* Tree arm's margin over the best linear arm: "
              f"**{v['tree_margin_over_best_linear']:+.6f}** log loss.")
        A(f"* **Winner: `{w['arm']}` + `{w['feature_set']}` + scheme `{w['scheme_selected']}`"
          + (f" (half-life {w['half_life_days']} days)" if w["scheme_selected"] == "S2" else "")
          + f", F2 log loss {w['log_loss']:.6f}.** {v['reason']}.")
        if w["scheme_selected"] != w["scheme_lowest_loss"]:
            A(f"* The lowest-loss scheme inside the winning class was `{w['scheme_lowest_loss']}` "
              f"({w['log_loss_lowest']:.6f}), but it does not clear the noise floor over "
              f"`{w['scheme_selected']}`, so the tie goes to the simpler scheme exactly as "
              f"pre-registered.")
        A("")
        A("Scheme ladder inside the winning class (arms that passed the gates):")
        A("")
        A(T(pd.DataFrame(v["scheme_ladder"]), ["scheme", "log_loss"], ["scheme", "F2 log loss"]))
        A("")
        if v["failed_gates"]:
            A("Arms that failed a gate (excluded from selection before any log loss was compared):")
            A("")
            fg = pd.DataFrame(v["failed_gates"])
            fg["calib"] = np.where(fg["calibration_pass"], "PASS", "FAIL")
            fg["respons"] = np.where(fg["responsiveness_pass"], "PASS", "FAIL")
            A(T(fg.sort_values("log_loss"),
                ["arm", "scheme", "log_loss", "calib", "worst_gated_gap_pp", "respons"],
                ["arm", "scheme", "F2 log loss", "calib", "worst gap (pp)", "respons."]))
        else:
            A("No arm failed a gate.")
        A("")
    # per-class calibration and responsiveness of the reference/winning arm
    for pop in sorted(verdicts, key=lambda x: 0 if x == "first" else 1):
        v = verdicts[pop]
        if v.get("winner"):
            w = v["winner"]
            label = "the winner"
            key = f"{pop}|{PO.SELECTION_FOLD}|{w['arm']}|{w['scheme_selected']}"
        else:
            b = v["best_by_log_loss_ignoring_gates"]
            label = (f"the lowest-loss arm `{b['arm']}` + `{b['scheme']}` "
                     f"(NOT ADOPTED -- it failed a gate)")
            key = f"{pop}|{PO.SELECTION_FOLD}|{b['arm']}|{b['scheme']}"
        idx = 7 if pop == "first" else 8
        A(f"### 5.{idx} Per-class decile calibration and responsiveness of {label} "
          f"(F2, `{pop}`)")
        A("")
        cal = detail[key]["calibration"]
        A(T(pd.DataFrame([
            {"class": c, "share %": x["share_pct"], "max abs decile gap (pp)": x["max_abs_gap_pp"],
             "level shift (pp)": x["level_shift_pp"],
             "residual shape (pp)": x["max_abs_gap_pp_after_level_shift"],
             "gated": "yes" if x["share_pct"] >= R1.CAL_GATE_MIN_SHARE else "no (share < 5%)"}
            for c, x in cal.items()]),
            ["class", "share %", "max abs decile gap (pp)", "level shift (pp)",
             "residual shape (pp)", "gated"]))
        A("")
        for spec, x in detail[key]["responsiveness"].items():
            A(f"`{spec}` -- predicted {x['pred_share']}, actual {x['actual_share']}; predicted moves "
              f"with the actual direction in {x['pred_monotone_steps']}/{x['n_steps']} steps "
              f"(gate: >= {PO.RESPONSIVENESS_MIN_STEPS}); span predicted {x['span_pred']:+.5f} vs "
              f"actual {x['span_actual']:+.5f}, slope ratio {x['slope_ratio']}.")
        A("")
        bs = detail[key]["by_state"]
        A(T(pd.DataFrame([
            {"segment": k, "n": x["n"], "log loss": round(x["log_loss"], 5),
             "max abs gap (pp)": x["max_abs_gap_pp"],
             "TOV pred/act %": f'{x["TOV_pred_pct"]}/{x["TOV_actual_pct"]}',
             "rim pred/act %": f'{x["FGA_rim_pred_pct"]}/{x["FGA_rim_actual_pct"]}',
             "3 pred/act %": f'{x["FGA_3_pred_pct"]}/{x["FGA_3_actual_pct"]}'}
            for k, x in bs.items()]),
            ["segment", "n", "log loss", "max abs gap (pp)", "TOV pred/act %", "rim pred/act %",
             "3 pred/act %"]))
        A("")
    A("### 5.9 What S1 actually did")
    A("")
    A("The refit schedule of the S1 arms on F2, so the scheme's cost is a number rather than an "
      "idea. Each test chance is scored by the most recent refit AT OR BEFORE its own game date, "
      "and every refit uses only games STRICTLY BEFORE its date, so no game is in its own fit and "
      "the test set is identical to S0's:")
    A("")
    for pop in sorted(verdicts, key=lambda x: 0 if x == "first" else 1):
        key = f"{pop}|{PO.SELECTION_FOLD}|cascade|S1"
        meta = scheme_meta.get(key)
        if not meta or "segments" not in meta:
            continue
        A(f"**{pop}** (shown on `cascade`; the schedule is identical for every arm):")
        A("")
        A(T(pd.DataFrame(meta["segments"]),
            ["refit_date", "n_train", "n_train_from_test_season", "max_train_date", "n_scored"],
            ["refit date", "train rows", "of which from the test season", "last train game",
             "test chances scored"]))
        A("")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--quick", action="store_true",
                    help="F2 only, first population, S0/S2 only, no noise floor")
    ap.add_argument("--no-append", action="store_true", help="do not append to experiments.md")
    ap.add_argument("--tree-floor-seeds", type=int, default=R1.N_SEEDS,
                    help="how many seed-varied refits to run for the TREE arm's noise floor "
                         "(pre-registered: 5). A smaller number, including 0, marks the floor "
                         "PARTIAL in every artifact and in the write-up; use it only under a hard "
                         "wall-clock limit, and finish the floor afterwards")
    ap.add_argument("--no-pred-dump", action="store_true",
                    help="skip the {stem}_F2_pred.npy dump. It re-runs the whole scheme, which for "
                         "an S1 winner means replaying every monthly refit")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore round2/checkpoint.json and recompute every cell from scratch")
    ap.add_argument("--cache", type=Path,
                    default=OUT_DIR / "design.parquet",
                    help="parquet path to cache/reuse the design matrix")
    args = ap.parse_args()

    assert_not_sealed(TRAIN_SEASONS, context="possession_outcome round-2 bake-off seasons")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    design = build_or_load_design(args.cache)
    assert_not_sealed(design, context="possession_outcome round-2 design matrix")

    run_meta = {
        "run_at": time.strftime("%Y-%m-%d %H:%M"),
        "seasons": TRAIN_SEASONS,
        "possessions_version": POSSESSIONS_VERSION,
        "style_source": "first_chance",
        "require_pbp_complete": True,
        "feature_set": FEATURE_SET,
        "n_rows": int(len(design)),
        "n_first": int((design["population"] == "first").sum()),
        "n_cont": int((design["population"] == "cont").sum()),
    }
    r1_meta = Path("data/processed/models/possession_outcome/run_meta.json")
    run_meta["round1_n_rows"] = (json.loads(r1_meta.read_text())["n_rows"]
                                 if r1_meta.exists() else run_meta["n_rows"])
    print(f"design: {run_meta['n_rows']:,} chances "
          f"({run_meta['n_first']:,} first / {run_meta['n_cont']:,} continuation); "
          f"round 1 had {run_meta['round1_n_rows']:,}")

    folds = ["F2"] if args.quick else ["F1", "F2"]
    populations = ["first"] if args.quick else ["first", "cont"]
    schemes = ["S0", "S2"] if args.quick else list(PO.SCHEMES)

    ckpt = Checkpoint(out_dir / "checkpoint.json", enabled=not args.no_resume)

    print("fitting S2 half-lives on F1 ...", flush=True)
    half_lives = fit_half_lives(design, populations, ckpt)
    (out_dir / "half_lives.json").write_text(json.dumps(half_lives, indent=2, default=str))

    rows, detail, scheme_meta = run_grid(design, folds, populations, half_lives, schemes, ckpt)
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "grid_results.csv", index=False)
    (out_dir / "metrics_detail.json").write_text(json.dumps(detail, indent=2, default=str))
    (out_dir / "scheme_meta.json").write_text(json.dumps(scheme_meta, indent=2, default=str))

    floors, verdicts, ladders = {}, {}, {}
    if not args.quick:
        for pop in populations:
            cached = ckpt.floor(pop)
            if cached is not None:
                print(f"noise floor [{pop}] (from checkpoint)", flush=True)
                floors[pop] = cached
            else:
                print(f"noise floor [{pop}] ...", flush=True)
                floors[pop] = noise_floor(design, grid, pop, half_lives,
                                          n_seeds=args.tree_floor_seeds)
                ckpt.put_floor(pop, floors[pop])
            verdicts[pop] = decide_v2(grid, floors[pop], pop)
            ladders[pop] = scheme_ladder_all(grid, pop,
                                             float(verdicts[pop]["noise_floor"]["applied"]))

    run_meta["runtime_min"] = (time.time() - t_start) / 60.0

    (out_dir / "noise_floor.json").write_text(json.dumps(floors, indent=2, default=str))
    (out_dir / "verdict.json").write_text(json.dumps(verdicts, indent=2, default=str))
    (out_dir / "scheme_ladder.json").write_text(json.dumps(ladders, indent=2, default=str))
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))

    for pop, v in verdicts.items():
        feats = PO.feature_set(FEATURE_SET, pop)
        if v.get("winner"):
            w = v["winner"]
            arm, scheme, adopted = w["arm"], w["scheme_selected"], True
            stem = f"winner_{pop}"
        elif v.get("best_by_log_loss_ignoring_gates"):
            b = v["best_by_log_loss_ignoring_gates"]
            arm, scheme, adopted = b["arm"], b["scheme"], False
            stem = f"reference_not_adopted_{pop}"
        else:
            continue
        hl = half_lives[pop][arm]["half_life_days"] if scheme == "S2" else None
        tr, te = PO.fold_slices(design, PO.SELECTION_FOLD, pop)
        if scheme == "S1":
            # An S1 arm is a SEQUENCE of fits. What is persisted is the LAST
            # refit -- the one a deployment would carry forward -- plus the
            # full refit schedule, and the payload says so rather than
            # pretending a single pickle is the whole scheme.
            cuts = PO.month_boundaries(pd.to_datetime(te["game_date"]))
            last_cut = cuts[-1]
            prior = te[(pd.to_datetime(te["game_date"]) < last_cut).to_numpy()]
            fit_rows = tr if not len(prior) else pd.concat([tr, prior], ignore_index=True)
            model = PO.fit_arm(arm, fit_rows, feats, seed=0)
            extra = {"s1_last_refit_date": str(last_cut.date()),
                     "s1_refit_schedule": scheme_meta[f"{pop}|F2|{arm}|S1"]["segments"],
                     "note": "this pickle is the LAST monthly refit only; S1 is a schedule"}
        elif scheme == "S2":
            ref = pd.to_datetime(te["game_date"]).min()
            wts = PO.recency_weights(tr["game_date"], ref, hl)
            model = PO.fit_arm(arm, tr, feats, seed=0, sample_weight=wts)
            extra = {"half_life_days": hl, "reference_date": str(ref.date())}
        else:
            model = PO.fit_arm(arm, tr, feats, seed=0)
            extra = {}
        with open(out_dir / f"{stem}.pkl", "wb") as fh:
            pickle.dump({"arm": arm, "feature_set": FEATURE_SET, "features": feats,
                         "classes": PO.CLASSES, "population": pop, "adopted": adopted,
                         "scheme": scheme, "fold": PO.SELECTION_FOLD,
                         "possessions_version": POSSESSIONS_VERSION,
                         "style_source": "first_chance", "require_pbp_complete": True,
                         "train_seasons": PO.FOLDS[PO.SELECTION_FOLD]["train"],
                         "model": model, **extra}, fh)
        if not args.no_pred_dump:
            p, _ = PO.fit_predict_scheme(arm, scheme, tr, te, feats, seed=0, half_life_days=hl)
            np.save(out_dir / f"{stem}_F2_pred.npy", p.astype("float32"))
        print(f"wrote {stem}.pkl ({arm} + {scheme}; adopted={adopted})")

    if not args.no_append:
        md = render_results(grid, detail, floors, verdicts, half_lives, ladders,
                            scheme_meta, run_meta)
        with open(EXPERIMENTS_MD, "a", encoding="utf-8") as fh:
            fh.write(md)
        print(f"appended results to {EXPERIMENTS_MD}")
    print(f"done in {run_meta['runtime_min']:.1f} min")


if __name__ == "__main__":
    main()
