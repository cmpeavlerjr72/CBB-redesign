"""
train_clock_v2.py -- ROUND 2 of the L5 CLOCK-CONSUMPTION bake-off, exactly the
grid pre-registered in `docs/models/clock/experiments.md` section 5.

What changed from round 1 (`scripts/train_clock_v1.py`, never overwritten):
`seconds_remaining` is no longer linear or a single 0-34 s bucket. It enters as
ten FINE buckets crossed with the period type {first half, second half/OT} and
the offence's score state {trailing, tied, leading}, plus a `last_shot_window`
(<= 30 s) and a `two_for_one_window` (30-45 s). Target, universe, folds,
metrics, noise floor and decision rules are unchanged.

    arms         empirical      fine Kaplan-Meier cell grid
                 lognormal      heteroscedastic, censored MLE, new dummies
                 gamma          same
                 hazard         discrete-time per-second, new dummies
                 lgbm_quantile  9 quantiles on raw clock + bucket id, with a
                                tree-parameter search ON F1 ONLY
                 two_regime     NEW: round-1 lgbm/D_plus_season above T,
                                the fine empirical table at or below T,
                                T chosen ON F1 ONLY from {30, 45, 60, 90}
    gates        emergent G1 (mean +/- 1.0, SD +/- 0.75, all powered months)
                 PIT cell K-S D <= 0.05 in every powered cell
                 END-OF-HALF, now a GATE: sim within the F1-derived noise floor
                 of actual, on both the share and the mean duration

Artifacts -> `data/processed/models/clock/` with a `v2_` prefix so round 1's
files are never overwritten. Results are APPENDED to
`docs/models/clock/experiments.md`.

Usage:
    .venv/Scripts/python.exe scripts/train_clock_v2.py
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

#: Feature set per arm, fixed by the round-2 pre-registration (there is no
#: feature-set grid in round 2: it specifies ONE state representation, in two
#: encodings -- dummies for the linear/hazard arms, raw + bucket id for the
#: tree).
ARM_FEATURES: dict[str, str] = {
    "empirical": "R2_dummy",
    "lognormal": "R2_dummy",
    "gamma": "R2_dummy",
    "hazard": "R2_dummy",
    "lgbm_quantile": "R2_tree",
}
#: Arm 5's upper regime is literally the round-1 best-CRPS model.
TWO_REGIME_HIGH_FEATURES = "D_plus_season"
TWO_REGIME_T_GRID: tuple[int, ...] = (30, 45, 60, 90)

#: Seeds used for the end-of-half noise floor. A spec-identical re-chain under
#: another seed is what `CLAUDE.md` means by a noise floor.
EOH_FLOOR_SEEDS: tuple[int, ...] = (20260910, 20260911, 20260912, 20260913, 20260914)

#: The pre-registration says "within the F1-derived noise floor" without fixing
#: a multiplier, so the multiplier is stated here rather than assumed silently:
#: a two-standard-error band. The one-SE version is reported alongside it.
EOH_FLOOR_K = 2.0


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Round-2 fitting helpers
# ---------------------------------------------------------------------------
def fit_r2_arm(arm: str, tr: pd.DataFrame, seed: int, tree_params: dict | None,
               high_model=None, threshold: int | None = None):
    if arm == "two_regime":
        low = ck.fit_empirical(tr, ARM_FEATURES["empirical"], seed=seed)
        return ck.TwoRegimeArm(threshold=int(threshold), low=low, high=high_model)
    fs = ARM_FEATURES[arm]
    if arm == "lgbm_quantile":
        return ck.fit_quantile_tree(tr, fs, seed=seed, params=tree_params)
    return ck.fit_arm(arm, tr, fs, seed=seed)


def tune_tree_on_f1(tr: pd.DataFrame, te: pd.DataFrame, seed: int) -> tuple[dict, list]:
    """The tree-parameter search the pre-registration allows ON F1 ONLY."""
    rows = []
    best, best_crps = None, np.inf
    for params in ck.R2_TREE_PARAM_GRID:
        t0 = time.time()
        arm = ck.fit_quantile_tree(tr, "R2_tree", seed=seed, params=params)
        s = ck.score_arm(arm, te)
        rows.append({**params, "f1_crps": round(float(s["crps"]), 6),
                     "fit_seconds": round(time.time() - t0, 1)})
        _log(f"  tree tune {params} -> F1 CRPS {s['crps']:.5f} ({time.time() - t0:.0f}s)")
        if s["crps"] < best_crps:
            best, best_crps = dict(params), float(s["crps"])
    return best, rows


def eoh_floor_from_f1(arms_f1: dict, te_f1: pd.DataFrame, actual_half_f1: pd.DataFrame) -> dict:
    """Two sources of noise in the end-of-half comparison, and the floor is the
    larger of them: the seed-to-seed SD of the SIM statistic (a spec-identical
    re-chain under another seed) and the game-block bootstrap SE of the ACTUAL
    statistic. Multiplied by `EOH_FLOOR_K`, which is stated, not assumed."""
    seed_share, seed_dur = [], []
    for name, n_seeds in (("empirical", 5), ("gamma", 3)):
        if name not in arms_f1:
            continue
        sh, du = ck.eoh_seed_noise(arms_f1[name], te_f1, EOH_FLOOR_SEEDS[:n_seeds])
        seed_share.append(sh)
        seed_dur.append(du)
        _log(f"  eoh seed noise ({name}, {n_seeds} seeds): share SD {sh:.5f}, duration SD {du:.4f}")
    b_share, b_dur = ck.eoh_actual_block_bootstrap_se(actual_half_f1)
    _log(f"  eoh actual block-bootstrap SE: share {b_share:.5f}, duration {b_dur:.4f}")
    se_share = max([*seed_share, b_share])
    se_dur = max([*seed_dur, b_dur])
    return {
        "share_se": round(se_share, 6), "duration_se": round(se_dur, 5),
        "share": round(EOH_FLOOR_K * se_share, 6), "duration": round(EOH_FLOOR_K * se_dur, 5),
        "k": EOH_FLOOR_K,
        "sim_seed_sd_share": [round(x, 6) for x in seed_share],
        "sim_seed_sd_duration": [round(x, 5) for x in seed_dur],
        "actual_block_bootstrap_se_share": round(b_share, 6),
        "actual_block_bootstrap_se_duration": round(b_dur, 5),
        "seeds": list(EOH_FLOOR_SEEDS),
    }


def choose_threshold_on_f1(low, high, te_f1, actual_half_f1, floor, seed) -> tuple[int, list]:
    """T from {30, 45, 60, 90} by emergent G1 + end-of-half error, ON F1 ONLY.

    The four errors are put on one scale by dividing each by its own tolerance
    (the G1 mean and SD tolerances, and the two end-of-half floors), so the
    score is "how many tolerances of error, summed". The formula is stated here
    because the pre-registration says only "by emergent G1 + end-of-half
    error"."""
    rows = []
    best, best_score = None, np.inf
    for T in TWO_REGIME_T_GRID:
        arm = ck.TwoRegimeArm(threshold=T, low=low, high=high)
        res = ck.chain_halves(arm, te_f1, seed=seed)
        rep = ck.emergent_report(res, actual_half_f1, eoh_floor=floor)
        e = rep["end_of_half"]
        score = (abs(rep["mean_delta"]) / ck.G1_MEAN_TOL
                 + abs(rep["sd_delta"]) / ck.G1_SD_TOL
                 + abs(e["share_gap"]) / floor["share"]
                 + abs(e["duration_gap"]) / floor["duration"])
        rows.append({"T": T, "f1_mean_delta": rep["mean_delta"], "f1_sd_delta": rep["sd_delta"],
                     "f1_eoh_share_gap": e["share_gap"], "f1_eoh_duration_gap": e["duration_gap"],
                     "score": round(float(score), 4)})
        _log(f"  T={T}: dmean {rep['mean_delta']:+.3f} dsd {rep['sd_delta']:+.3f} "
             f"eoh dshare {e['share_gap']:+.4f} ddur {e['duration_gap']:+.3f} -> score {score:.3f}")
        if score < best_score:
            best, best_score = T, float(score)
    return int(best), rows


def pit_summary(cells: pd.DataFrame) -> dict:
    powered = cells[cells["powered"]]
    return {
        "n_cells": int(len(cells)), "n_powered": int(len(powered)),
        "n_underpowered": int((~cells["powered"]).sum()),
        "pit_leak_failures": int(powered["leak_sized"].sum()),
        "pit_worst_D_powered": round(float(powered["ks_D"].max()), 5) if len(powered) else float("nan"),
        "pit_worst_cell": (powered.loc[powered["ks_D"].idxmax(), "cell"] if len(powered) else ""),
    }


def decide(grid: pd.DataFrame, floor: float) -> dict:
    """Round-1 decision rules, with the end-of-half check added as a gate and
    `two_regime` counting as a tree arm."""
    f2 = grid[grid["fold"] == "F2"].copy()
    f2["eligible"] = f2["g1_pass"] & (f2["pit_leak_failures"] == 0) & f2["eoh_pass"]
    elig = f2[f2["eligible"]]
    out = {"floor": floor, "n_arms": int(len(f2)), "n_eligible": int(len(elig)),
           "n_pass_g1": int(f2["g1_pass"].sum()),
           "n_pit_clean": int((f2["pit_leak_failures"] == 0).sum()),
           "n_pass_eoh": int(f2["eoh_pass"].sum())}
    if not len(elig):
        out.update({
            "winner": None, "verdict": "NO ARM ADOPTED",
            "reason": ("no arm passed all three pre-registered round-2 gates on F2 "
                       "(emergent G1, PIT by state cell, end-of-half)"),
            "best_crps_arm": f2.loc[f2["crps"].idxmin(), "arm"],
            "best_crps": float(f2["crps"].min()),
        })
        return out

    per_arm = elig.groupby("arm", as_index=False)["crps"].min()
    per_arm["rank"] = per_arm["arm"].map(ck.SIMPLICITY_RANK)
    best_crps = float(per_arm["crps"].min())
    non_tree = per_arm[~per_arm["arm"].isin(ck.TREE_ARMS)]
    best_non_tree = float(non_tree["crps"].min()) if len(non_tree) else np.inf
    tied = per_arm[per_arm["crps"] <= best_crps + floor].sort_values(["rank", "crps"], kind="stable")
    win_arm = str(tied.iloc[0]["arm"])
    tree_note = ""
    if (win_arm in ck.TREE_ARMS and np.isfinite(best_non_tree)
            and float(per_arm.loc[per_arm["arm"] == win_arm, "crps"].iloc[0]) >= best_non_tree - floor):
        drop = tied[~tied["arm"].isin(ck.TREE_ARMS)]
        tree_note = (f"tree arm did not beat the best non-tree arm by more than the floor "
                     f"({best_non_tree:.5f} - {best_crps:.5f} <= {floor:.5f}); demoted")
        win_arm = str(drop.iloc[0]["arm"]) if len(drop) else str(
            non_tree.sort_values("crps").iloc[0]["arm"])
    row = elig[elig["arm"] == win_arm].sort_values("crps").iloc[0]
    out.update({
        "winner": {"arm": win_arm, "feature_set": str(row["feature_set"]), "crps": float(row["crps"])},
        "verdict": f"ADOPT {win_arm} / {row['feature_set']}",
        "best_eligible_crps": best_crps,
        "best_non_tree_crps": None if not np.isfinite(best_non_tree) else best_non_tree,
        "tree_note": tree_note,
    })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
F1_CACHE = OUT_DIR / "v2_f1_stage.json"
CACHE = OUT_DIR / "v2_cache"


def _load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _cached_pickle(path: Path, build, label: str):
    """Fit-or-load. Every expensive object round 2 builds goes through this, so a
    killed run resumes instead of restarting."""
    if path.exists():
        _log(f"  {label} loaded from cache")
        return _load_pickle(path)
    t0 = time.time()
    obj = build()
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    _log(f"  {label} fitted in {time.time() - t0:.0f}s and cached")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--refit-f1", action="store_true",
                    help="ignore the cached F1 stage and redo the tree search, the "
                         "end-of-half noise floor and the threshold search")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()

    _log("building the round-2 design table ...")
    design, diag = ck.build_design(ALL_SEASONS)
    design.to_parquet(OUT_DIR / "design_v2.parquet", index=False)
    (OUT_DIR / "v2_build_diagnostics.json").write_text(json.dumps(diag, indent=2, default=str))
    _log(f"design {len(design):,} rows / {design.shape[1]} cols -> design_v2.parquet")

    tr1, te1 = ck.fold_slices(design, "F1")
    tr2, te2 = ck.fold_slices(design, "F2")
    ah1 = ck.actual_end_of_half(te1)
    ah2 = ck.actual_end_of_half(te2)
    _log(f"F1 train {len(tr1):,} / test {len(te1):,}; F2 train {len(tr2):,} / test {len(te2):,}")

    # ---- F1: every choice the pre-registration allows on F1 only ----------
    # Cached as a whole, because it is deterministic given (design, seed) and it
    # costs ~30 minutes: a re-run of the F2 stage must not silently re-decide
    # the tree parameters, the threshold or the noise floor.
    if F1_CACHE.exists() and not args.refit_f1:
        cache = json.loads(F1_CACHE.read_text())
        if cache.get("seed") != args.seed:
            raise SystemExit(f"{F1_CACHE} was produced with seed {cache.get('seed')}, "
                             f"not {args.seed}; pass --refit-f1")
        tree_params = cache["tree_params"]
        tune_rows = cache["tree_param_search_f1"]
        floor_eoh = cache["end_of_half_floor"]
        T = int(cache["threshold_chosen"])
        t_rows = cache["threshold_search_f1"]
        rows = list(cache["f1_rows"])
        _log(f"reusing cached F1 stage: tree {tree_params}, T = {T}, "
             f"eoh floor share +/-{floor_eoh['share']}, duration +/-{floor_eoh['duration']}")
    else:
        _log("F1 tree-parameter search (allowed on F1 only) ...")
        tree_params, tune_rows = tune_tree_on_f1(tr1, te1, args.seed)
        _log(f"F1 chose tree params {tree_params}")

        rows = []
        arms_f1: dict = {}
        high_f1 = None
        for arm in ck.R2_ARMS:
            t0 = time.time()
            if arm == "two_regime":
                high_f1 = ck.fit_quantile_tree(tr1, TWO_REGIME_HIGH_FEATURES, seed=args.seed)
                _log(f"  F1 round-1 high component fitted in {time.time() - t0:.0f}s")
                continue  # built after T is chosen
            arms_f1[arm] = fit_r2_arm(arm, tr1, args.seed, tree_params)
            s = ck.score_arm(arms_f1[arm], te1)
            cells = ck.pit_by_cell(te1, s["pit_row"])
            rows.append({"fold": "F1", "arm": arm, "feature_set": ARM_FEATURES[arm],
                         "crps": round(s["crps"], 6),
                         "logscore_defined": round(s["logscore_defined"], 6),
                         "logscore_undef_pct": s["logscore_undef_pct"],
                         "fit_seconds": round(time.time() - t0, 1), **pit_summary(cells)})
            _log(f"  F1 {arm:14s} fit {time.time() - t0:6.0f}s CRPS {s['crps']:.5f} "
                 f"PITfail {rows[-1]['pit_leak_failures']}/{rows[-1]['n_powered']}")

        _log("F1 end-of-half noise floor ...")
        floor_eoh = eoh_floor_from_f1(arms_f1, te1, ah1)
        _log(f"  floor: share +/-{floor_eoh['share']:.5f}, duration +/-{floor_eoh['duration']:.4f}")

        _log("F1 threshold search for the two-regime arm ...")
        T, t_rows = choose_threshold_on_f1(arms_f1["empirical"], high_f1, te1, ah1,
                                           floor_eoh, args.seed)
        _log(f"F1 chose T = {T}")
        two_f1 = ck.TwoRegimeArm(threshold=T, low=arms_f1["empirical"], high=high_f1)
        s = ck.score_arm(two_f1, te1)
        cells = ck.pit_by_cell(te1, s["pit_row"])
        rows.append({"fold": "F1", "arm": "two_regime", "feature_set": f"R2_two_regime(T={T})",
                     "crps": round(s["crps"], 6),
                     "logscore_defined": round(s["logscore_defined"], 6),
                     "logscore_undef_pct": s["logscore_undef_pct"], "fit_seconds": None,
                     **pit_summary(cells)})
        F1_CACHE.write_text(json.dumps({
            "seed": args.seed, "tree_params": tree_params,
            "tree_param_search_f1": tune_rows, "end_of_half_floor": floor_eoh,
            "threshold_chosen": T, "threshold_search_f1": t_rows, "f1_rows": rows,
            "written_at": datetime.now(UTC).isoformat(),
        }, indent=2, default=str))
        del arms_f1, high_f1, two_f1
        _log(f"F1 stage cached to {F1_CACHE}")

    # ---- F2: the selection fold ------------------------------------------
    # Cached PER ARM. Another agent force-killed every python process on this
    # machine twice while round 2 was running, so the run has to be able to pick
    # up where it was cut off; each arm's fitted model and every frame it
    # produces are written as soon as they exist.
    _log("--- F2 (selection) ---")
    CACHE.mkdir(parents=True, exist_ok=True)
    arms_f2: dict = {}
    boot_se: dict = {}
    pit_frames, emerg_rows, month_rows, dbt_frames, resp_rows = [], [], [], [], []
    high_f2 = _cached_pickle(CACHE / "high_f2.pkl",
                             lambda: ck.fit_quantile_tree(tr2, TWO_REGIME_HIGH_FEATURES,
                                                          seed=args.seed),
                             "F2 round-1 high component")
    for arm in ck.R2_ARMS:
        done = CACHE / f"{arm}_row.json"
        if done.exists():
            payload = json.loads(done.read_text())
            rows.append(payload["row"])
            boot_se[arm] = payload["block_bootstrap_se"]
            pit_frames.append(pd.read_csv(CACHE / f"{arm}_pit.csv"))
            emerg_rows.append(payload["emergent"])
            month_rows.extend(payload["months"])
            dbt_frames.append(pd.read_csv(CACHE / f"{arm}_dbt.csv"))
            resp_rows.append(payload["responsiveness"])
            arms_f2[arm] = _load_pickle(CACHE / f"{arm}_model.pkl")
            _log(f"  F2 {arm:14s} loaded from cache (CRPS {payload['row']['crps']:.5f})")
            continue
        t0 = time.time()
        a = fit_r2_arm(arm, tr2, args.seed, tree_params, high_model=high_f2, threshold=T)
        arms_f2[arm] = a
        s = ck.score_arm(a, te2)
        boot_se[arm] = round(ck.block_bootstrap_se(te2, s["crps_row"]), 6)
        cells = ck.pit_by_cell(te2, s["pit_row"])
        c2 = cells.copy()
        c2.insert(0, "arm", arm)
        pit_frames.append(c2)

        res = ck.chain_halves(a, te2, seed=args.seed)
        rep = ck.emergent_report(res, ah2, eoh_floor=floor_eoh)
        e = rep["end_of_half"]
        cc = rep["clock_complete"]
        row = {
            "fold": "F2", "arm": arm,
            "feature_set": (f"R2_two_regime(T={T})" if arm == "two_regime" else ARM_FEATURES[arm]),
            "crps": round(s["crps"], 6), "logscore_defined": round(s["logscore_defined"], 6),
            "logscore_undef_pct": s["logscore_undef_pct"],
            "fit_seconds": round(time.time() - t0, 1),
            "pred_mean_duration": round(s["pred_mean"], 4),
            "actual_mean_duration": round(s["actual_mean"], 4),
            "censored_mean_pred_survival": round(
                ck.censored_survival_diagnostic(a, te2).get("mean_pred_survival", float("nan")), 4),
            **pit_summary(cells),
            "g1_mean_delta": rep["mean_delta"], "g1_sd_delta": rep["sd_delta"],
            "g1_sim_mean": rep["sim_mean"], "g1_actual_mean": rep["actual_mean"],
            "g1_sim_sd": rep["sim_sd"], "g1_actual_sd": rep["actual_sd"],
            "g1_count_ks_D": rep["count_ks_D"],
            "g1_overall_pass": rep["overall_pass"],
            "g1_months_pass": int(sum(m["pass"] for m in rep["months"] if m["powered"])),
            "g1_months_powered": rep["n_powered_months"], "g1_pass": rep["g1_pass"],
            "eoh_sim_share": e["sim_share_last_poss_under_35s"],
            "eoh_actual_share": e["actual_share_last_poss_under_35s"],
            "eoh_share_gap": e["share_gap"], "eoh_share_pass": e["share_pass"],
            "eoh_sim_dur": e["sim_mean_duration_under_35s"],
            "eoh_actual_dur": e["actual_mean_duration_under_35s"],
            "eoh_duration_gap": e["duration_gap"], "eoh_duration_pass": e["duration_pass"],
            "eoh_pass": e["pass"],
            "cc_eoh_actual_share": cc.get("eoh_actual_share"),
            "cc_eoh_sim_share": cc.get("eoh_sim_share"),
            "cc_eoh_share_gap": cc.get("eoh_share_gap"),
            "cc_eoh_actual_dur": cc.get("eoh_actual_mean_duration"),
            "cc_eoh_sim_dur": cc.get("eoh_sim_mean_duration"),
            "cc_eoh_duration_gap": cc.get("eoh_duration_gap"),
            "cc_mean_delta": cc.get("mean_delta"), "cc_sd_delta": cc.get("sd_delta"),
            "cc_n_games": cc.get("n_games"),
            "chain_wrapped_halves": rep["wrapped_halves"],
        }
        rows.append(row)
        emergent = {"arm": arm, **{k: v for k, v in rep.items()
                                   if k not in ("months", "end_of_half", "clock_complete")},
                    **{f"eoh_{k}": v for k, v in e.items()},
                    **{f"cc_{k}": v for k, v in cc.items()}}
        emerg_rows.append(emergent)
        months = [{"arm": arm, **m} for m in rep["months"]]
        month_rows.extend(months)

        dbt = ck.duration_by_terminal(te2, s["pred_mean_row"], s["pred_m2_row"])
        dbt.insert(0, "arm", arm)
        dbt_frames.append(dbt)

        # responsiveness (reported, per the pre-registration)
        tempo = te2.groupby("game_id")["tempo_prior_game"].first()
        q = pd.qcut(tempo, 5, labels=[1, 2, 3, 4, 5])
        pg = res.per_game.set_index("game_id")
        j = pd.DataFrame({"tempo_q": q.reindex(pg.index), "sim": pg["sim_poss"],
                          "actual": pg["actual_poss"]}).dropna()
        agg = j.groupby("tempo_q", observed=True).mean()
        sp_s = float(agg["sim"].iloc[-1] - agg["sim"].iloc[0])
        sp_a = float(agg["actual"].iloc[-1] - agg["actual"].iloc[0])
        steps = int(np.sum(np.sign(np.diff(agg["sim"])) == np.sign(np.diff(agg["actual"]))))
        resp = {"arm": arm, "span_sim": round(sp_s, 3), "span_actual": round(sp_a, 3),
                "slope_ratio": round(sp_s / sp_a, 4) if sp_a else None,
                "steps_agreeing": steps, "n_steps": 4,
                **{f"q{i + 1}_sim": round(float(v), 3) for i, v in enumerate(agg["sim"])},
                **{f"q{i + 1}_actual": round(float(v), 3) for i, v in enumerate(agg["actual"])}}
        resp_rows.append(resp)

        c2.to_csv(CACHE / f"{arm}_pit.csv", index=False)
        dbt.to_csv(CACHE / f"{arm}_dbt.csv", index=False)
        with open(CACHE / f"{arm}_model.pkl", "wb") as fh:
            pickle.dump(a, fh)
        (CACHE / f"{arm}_row.json").write_text(json.dumps(
            {"seed": args.seed, "row": row, "emergent": emergent, "months": months,
             "responsiveness": resp, "block_bootstrap_se": boot_se[arm]},
            indent=2, default=str))

        _log(f"  F2 {arm:14s} fit {row['fit_seconds']:7.1f}s CRPS {row['crps']:.5f} "
             f"PITfail {row['pit_leak_failures']}/{row['n_powered']} "
             f"G1 dmean {row['g1_mean_delta']:+.3f} dsd {row['g1_sd_delta']:+.3f} "
             f"EOH dshare {row['eoh_share_gap']:+.4f} ddur {row['eoh_duration_gap']:+.3f} "
             f"[cc dshare {row['cc_eoh_share_gap']} ddur {row['cc_eoh_duration_gap']}]")
        # written after every arm, so a killed run loses at most one arm
        pd.DataFrame(rows).to_csv(OUT_DIR / "v2_grid_results.csv", index=False)

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "v2_grid_results.csv", index=False)
    pd.concat(pit_frames, ignore_index=True).to_csv(OUT_DIR / "v2_pit_cells_F2.csv", index=False)
    pd.DataFrame(emerg_rows).to_csv(OUT_DIR / "v2_emergent_F2.csv", index=False)
    pd.DataFrame(month_rows).to_csv(OUT_DIR / "v2_emergent_F2_by_month.csv", index=False)
    pd.concat(dbt_frames, ignore_index=True).to_csv(OUT_DIR / "v2_duration_by_terminal_F2.csv", index=False)
    pd.DataFrame(resp_rows).to_csv(OUT_DIR / "v2_responsiveness_F2.csv", index=False)

    # ---- noise floor ------------------------------------------------------
    _log("noise floor ...")
    base = float(grid[(grid.fold == "F2") & (grid.arm == "lgbm_quantile")]["crps"].iloc[0])
    tf_path = CACHE / "tree_seed_refit.json"
    if tf_path.exists():
        tree_floor = float(json.loads(tf_path.read_text())["abs_delta"])
        alt_crps = float(json.loads(tf_path.read_text())["crps_seed_alt"])
        _log(f"  tree seed refit loaded from cache: |dCRPS| = {tree_floor:.6f}")
    else:
        alt = ck.fit_quantile_tree(tr2, "R2_tree", seed=args.seed + NOISE_SEED_OFFSET,
                                   params=tree_params)
        alt_crps = float(ck.score_arm(alt, te2)["crps"])
        tree_floor = abs(alt_crps - base)
        tf_path.write_text(json.dumps({"crps_seed_base": base, "crps_seed_alt": round(alt_crps, 6),
                                       "abs_delta": round(tree_floor, 6)}, indent=2))
    boot = dict(boot_se)
    floor = float(max(tree_floor, max(boot.values())))
    nf = {"tree_seed_refit": {"crps_seed_base": base, "crps_seed_alt": round(alt_crps, 6),
                              "abs_delta": round(tree_floor, 6)},
          "block_bootstrap_se": boot, "floor_used": round(floor, 6),
          "end_of_half_floor": floor_eoh,
          "tree_param_search_f1": tune_rows, "tree_params_chosen": tree_params,
          "threshold_search_f1": t_rows, "threshold_chosen": T}
    (OUT_DIR / "v2_noise_floor.json").write_text(json.dumps(nf, indent=2, default=str))
    _log(f"floor = {floor:.6f} (tree seed refit {tree_floor:.6f}, worst bootstrap SE {max(boot.values()):.6f})")

    verdict = decide(grid, floor)
    verdict.update({"created_at": started, "finished_at": datetime.now(UTC).isoformat(),
                    "seed": args.seed, "round": 2, "tree_params": tree_params, "threshold_T": T,
                    "end_of_half_floor": floor_eoh})
    (OUT_DIR / "v2_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    _log(f"VERDICT: {verdict['verdict']}")

    if verdict.get("winner"):
        with open(OUT_DIR / "v2_winner.pkl", "wb") as fh:
            pickle.dump({"adopted": True, **verdict["winner"],
                         "model": arms_f2[verdict["winner"]["arm"]]}, fh)
    else:
        best = str(grid[grid.fold == "F2"].sort_values("crps").iloc[0]["arm"])
        with open(OUT_DIR / f"v2_reference_not_adopted_{best}.pkl", "wb") as fh:
            pickle.dump({"adopted": False, "arm": best, "round": 2,
                         "why": "round 2 adopted nothing; see docs/models/clock/model.md",
                         "model": arms_f2[best]}, fh)

    write_section(grid, verdict, nf, diag, args, T, tree_params)
    _log("done")


def _md(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                vals.append("PASS" if v else "FAIL")
            elif isinstance(v, float) and np.isfinite(v):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_section(grid, verdict, nf, diag, args, T, tree_params) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    f2 = grid[grid.fold == "F2"].sort_values("crps")
    f1 = grid[grid.fold == "F1"].sort_values("crps")
    e = nf["end_of_half_floor"]
    L: list[str] = []
    a = L.append
    a("")
    a(f"## 6. Run R3 -- the round-2 grid ({stamp})")
    a("")
    a(f"`scripts/train_clock_v2.py`, seed {args.seed}. Artifacts are `v2_*` in "
      "`data/processed/models/clock/`; round 1's files are untouched. Every number is written by "
      "that script.")
    a("")
    a("### 6.1 Choices made on F1 only")
    a("")
    a("Tree-parameter search (allowed on F1 only by the pre-registration):")
    a("")
    a(_md(pd.DataFrame(nf["tree_param_search_f1"]), ["num_leaves", "min_child_samples", "f1_crps", "fit_seconds"]))
    a("")
    a(f"Chosen: `{tree_params}`.")
    a("")
    a("Two-regime threshold search (F1 only). Score = |G1 mean delta| / 1.0 + |G1 SD delta| / 0.75 "
      "+ |end-of-half share gap| / share floor + |end-of-half duration gap| / duration floor, i.e. "
      "how many tolerances of error, summed; the formula is stated here because the "
      "pre-registration says only \"by emergent G1 + end-of-half error\".")
    a("")
    a(_md(pd.DataFrame(nf["threshold_search_f1"]),
          ["T", "f1_mean_delta", "f1_sd_delta", "f1_eoh_share_gap", "f1_eoh_duration_gap", "score"]))
    a("")
    a(f"**Chosen T = {T}.**")
    a("")
    a("End-of-half noise floor, derived on F1: the larger of the seed-to-seed SD of the SIM "
      f"statistic (spec-identical re-chains at {len(e['seeds'])} seeds) and the game-block bootstrap "
      "SE of the ACTUAL statistic, times k = "
      f"{e['k']} (the multiplier is stated, not assumed). "
      f"Sim seed SD: share {e['sim_seed_sd_share']}, duration {e['sim_seed_sd_duration']}. "
      f"Actual block-bootstrap SE: share {e['actual_block_bootstrap_se_share']}, duration "
      f"{e['actual_block_bootstrap_se_duration']}. "
      f"**Floor: share +/- {e['share']}, duration +/- {e['duration']} s.**")
    a("")
    a("### 6.2 F2 (selection fold) -- the three gates")
    a("")
    a(_md(f2, ["arm", "crps", "logscore_defined", "logscore_undef_pct", "pit_leak_failures",
               "n_powered", "pit_worst_D_powered", "g1_mean_delta", "g1_sd_delta",
               "g1_months_pass", "g1_pass", "eoh_share_gap", "eoh_duration_gap", "eoh_pass"],
          ["arm", "CRPS", "log score", "undef %", "PIT fails", "powered", "worst D",
           "G1 d-mean", "G1 d-SD", "months pass", "G1", "EOH d-share", "EOH d-dur", "EOH"]))
    a("")
    a("### 6.3 The end-of-half statistic is contaminated by feed truncation (measured, not assumed)")
    a("")
    a("The CBBD event stream stops before the horn in a large minority of halves: only 44.6% of 2025 "
      "halves have a last logged possession ending at exactly 0:00, the median unaccounted time is "
      "1 s, the 90th percentile 17 s and the mean 5.29 s. The 11.7% of halves whose last LOGGED "
      "possession starts with 35+ seconds left are mostly these -- their mean end clock is 20.4 s, "
      "and only 16.3% of them reach 0:00. A sim that always runs its clock to zero cannot reproduce "
      "a half that simply stops at 0:20, so the pre-registered actual value of 0.8832 is not a "
      "quantity any correct model can match.")
    a("")
    a("Restricted to CLOCK-COMPLETE halves (last possession ending within "
      f"{ck.CLOCK_COMPLETE_TOL_S} s of the horn, 63.5% of 2025 halves) the actual share is "
      "**0.9556** and the actual mean duration **12.41 s**. Both readings are reported below; the "
      "pre-registered all-halves number is the one the GATE uses, because a gate is not re-based "
      "after the fact.")
    a("")
    a(_md(f2, ["arm", "eoh_actual_share", "eoh_sim_share", "eoh_share_gap", "eoh_actual_dur",
               "eoh_sim_dur", "eoh_duration_gap", "cc_eoh_actual_share", "cc_eoh_sim_share",
               "cc_eoh_share_gap", "cc_eoh_actual_dur", "cc_eoh_sim_dur", "cc_eoh_duration_gap"],
          ["arm", "actual share", "sim share", "d-share", "actual dur", "sim dur", "d-dur",
           "CC actual share", "CC sim share", "CC d-share", "CC actual dur", "CC sim dur",
           "CC d-dur"]))
    a("")
    a("`CC` = the clock-complete sub-universe. The same secondary read on the emergent count "
      "(clock-complete GAMES, both halves) is in the `cc_mean_delta` / `cc_sd_delta` columns of "
      "`v2_grid_results.csv`.")
    a("")
    a("### 6.4 F1 (robustness only)")
    a("")
    a(_md(f1, ["arm", "feature_set", "crps", "logscore_defined", "logscore_undef_pct",
               "pit_leak_failures", "n_powered", "pit_worst_D_powered"],
          ["arm", "features", "CRPS", "log score", "undef %", "PIT fails", "powered", "worst D"]))
    a("")
    a("### 6.5 Noise floor and verdict")
    a("")
    a(f"Tree seed-varied refit |dCRPS| = {nf['tree_seed_refit']['abs_delta']}. "
      f"Worst game-block bootstrap SE = {max(nf['block_bootstrap_se'].values())}. "
      f"Floor used = **{nf['floor_used']}**.")
    a("")
    a("```json")
    a(json.dumps(verdict, indent=2, default=str))
    a("```")
    a("")
    with DOC.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
