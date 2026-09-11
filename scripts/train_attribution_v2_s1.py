#!/usr/bin/env python
"""
train_attribution_v2_s1.py -- L4 attribution ROUND 2: the score_diff pre-play
fix (data fix only) x the S1 in-season monthly walk-forward scheme.

    .venv/Scripts/python.exe scripts/train_attribution_v2_s1.py

Pre-registration: `docs/models/attribution/experiments.md` section 4,
committed BEFORE this ran. Leak quantification:
`docs/tests/attribution_score_diff_leak_2026-09-10.md`.

WHAT THIS COMPARES, per target, round-1's winning arm CLASS FIXED (`assist`
and `stolen` carry their best round-1 arm as a REFERENCE, since round 1
adopted no winner for either):

    REB_off -> cond_logit   REB_def -> lgbm     assist  -> lgbm (reference)
    steal   -> proportional block   -> lgbm     assisted -> aware_ridge
    stolen  -> lgbm (reference)                 blocked  -> aware_ridge

(a) STATIC-FIXED. Reuses `train_attribution_v1`'s OWN tri-arm, two-fold
    pipeline (`run_choice_fold` / `run_binary_fold`, UNMODIFIED) on population
    tables rebuilt with `score_diff_mode="pre_play"` (now the default --
    `attribution.py` this session). This is free cross-checking:
    `docs/tests/attribution_score_diff_leak_2026-09-10.md` proves
    `REB_off`/`REB_def`/`steal`/`stolen`/`block`/`blocked`'s populations are
    UNCHANGED by the fix, so their numbers here should reproduce round 1's
    bit for bit; only `assist`/`assisted` should move. Written to
    `round2_s1/static_results.json` / `round2_s1/static_report.md`.

(b) S1-FIXED. NOT a tri-arm rerun: ONE arm class per target (the table
    above), refit at each calendar-month boundary of the F1 2025 test season
    on all of 2024 plus 2025-to-date, on the same pre-play-fixed data.
    `possession_outcome.month_boundaries` is called directly -- S1 is NOT
    reimplemented, exactly the `train_fg_make_v2b_s1.py` pattern. The arm's
    own hyperparameter (cond_logit's l2 / the lgbm params / the ridge C) is
    FROZEN at round 1's F1-searched value (read back from `results_v1.json`)
    and reused unchanged across every refit; the shrinkage prior is refit
    fresh each month (cheap, and shrinkage does not depend on `score_diff`).
    A second LightGBM seed (seed=1, not persisted) supplies this arm's own
    noise floor on the four targets whose arm class is `lgbm`. The WF2025
    robustness read is the SAME S1 predictions sliced to
    `game_date >= 2025-01-15`, per the pre-registration -- not a second
    monthly schedule fit inside an already-partial season.

Artifacts, all under `data/processed/models/attribution/round2_s1/`
(gitignored; never `git add`-ed):
    events_{pop}_v2.parquet            rebuilt population tables (pre-play score_diff)
    asof_v2.parquet / team_asof_v2.parquet   rebuilt (byte-for-byte vs round 1: neither reads score_diff)
    static_results.json / static_report.md   (a) full tri-arm rerun, both folds
    {target}_s1/<refit_date>.joblib          (b) one persisted arm per S1 refit month (seed 0)
    manifest_{target}.json                   (b) engine.manifest-format manifest (seed 0)
    round2_summary.json                      per-target comparison + decision
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.models import attribution as A  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

import train_attribution_v1 as V1  # noqa: E402

OUT = ROOT / "data/processed/models/attribution/round2_s1"
ROUND1 = ROOT / "data/processed/models/attribution"
RESULTS_V1_PATH = ROUND1 / "results_v1.json"

TARGET_ARM: dict[str, str] = {
    "REB_off": "cond_logit", "REB_def": "lgbm", "assist": "lgbm",
    "steal": "proportional", "block": "lgbm",
    "assisted": "aware_ridge", "stolen": "lgbm", "blocked": "aware_ridge",
}
#: targets adopted nothing on F1 in round 1 -- carried as a REFERENCE class,
#: never re-decided here.
REFERENCE_ONLY: frozenset[str] = frozenset({"assist", "stolen"})

WF_CUT = pd.Timestamp(A.WF_SPLIT_DATE)


# ---------------------------------------------------------------------------
# 0. data: rebuild the five population tables with the fix, reuse asof/team
# ---------------------------------------------------------------------------
def build_fixed_tables(rebuild: bool):
    """`train_attribution_v1.build_tables`, UNCHANGED, pointed at a NEW
    directory. It calls `attribution.build_attr_events` without a
    `score_diff_mode` argument, so it now gets this session's new DEFAULT
    (`"pre_play"`) -- round 1's own cache under `data/processed/models/
    attribution/` is never touched (different `out_dir`)."""
    return V1.build_tables("v2", OUT, rebuild)


# ---------------------------------------------------------------------------
# 1. frozen hyperparameters, read back from round 1's F1 result
# ---------------------------------------------------------------------------
def load_frozen_params() -> dict[str, dict]:
    results = json.loads(RESULTS_V1_PATH.read_text(encoding="utf-8"))
    by = {(r["fold"], r["target"]): r for r in results}
    out = {}
    for target, arm in TARGET_ARM.items():
        r = by[("F1", target)]["arms"][arm]
        if arm == "cond_logit":
            out[target] = {"l2": r["l2"]}
        elif arm == "lgbm":
            out[target] = {"params": r["params"]}
        elif arm == "aware_ridge":
            out[target] = {"C": r["C"]}
        else:
            out[target] = {}
    return out, by


# ---------------------------------------------------------------------------
# 2. (a) STATIC-FIXED: reuse train_attribution_v1's tri-arm pipeline verbatim
# ---------------------------------------------------------------------------
def run_static_fixed(pops, asof, team, args, log) -> list[dict]:
    results: list[dict] = []
    for target in A.TARGETS:
        if target in A.CHOICE_TARGETS:
            design = A.build_choice_design(A.usable(pops[A.POP_OF[target]], target),
                                           asof, target)
            tr, te = A.fold_slices(design)
            f1 = V1.run_choice_fold(target, tr, te, "F1", args, log)
            results.append(f1)
            wtr, wte = A.walkforward_slices(design)
            results.append(V1.run_choice_fold(target, wtr, wte, "WF2025", args, log,
                                               lgbm_params=f1["arms"]["lgbm"]["params"]))
        else:
            p = pops[A.POP_OF[target]]
            design = A.build_binary_design(p[p["five_ok"].to_numpy()], asof, team, target)
            tr, te = A.fold_slices(design)
            results.append(V1.run_binary_fold(target, tr, te, "F1", args, log))
            wtr, wte = A.walkforward_slices(design)
            results.append(V1.run_binary_fold(target, wtr, wte, "WF2025", args, log))
    return results


# ---------------------------------------------------------------------------
# 3. (b) S1-FIXED: one arm class per target, monthly refit
# ---------------------------------------------------------------------------
def build_target_design(target: str, pops: dict, asof: pd.DataFrame,
                        team: pd.DataFrame) -> pd.DataFrame:
    if target in A.CHOICE_TARGETS:
        return A.build_choice_design(A.usable(pops[A.POP_OF[target]], target), asof, target)
    p = pops[A.POP_OF[target]]
    return A.build_binary_design(p[p["five_ok"].to_numpy()], asof, team, target)


def fit_one_choice_arm(arm: str, target: str, tr: pd.DataFrame, pk: str, m: float,
                       frozen: dict, seed: int):
    if arm == "proportional":
        return None, {}
    if arm == "cond_logit":
        unid = A.unidentified_features(tr, target)
        X, names = A.cl_design(tr, target, pk, m)
        keep = [i for i, n in enumerate(names) if n not in unid]
        X, names = X[:, :, keep], [names[i] for i in keep]
        X, names, _const = A.drop_constant_features(X, names)
        model = A.CondLogitArm(l2=frozen["l2"]).fit(X, tr["y"].to_numpy())
        return model, {"names": names}
    if arm == "lgbm":
        k_alt = A.N_ALT_OF[target]
        unid = set(A.unidentified_features(tr, target))
        keep = [i for i, n in enumerate(A.LGBM_FEATURES) if n not in unid]
        L, c = A.long_frame(tr, target, pk, m)
        init = A.lgbm_init_score(tr, target, pk, m)
        model = A.LgbmChoiceArm(k_alt, frozen["params"], seed=seed).fit(L[:, keep], c, init=init)
        return model, {"keep": keep}
    raise KeyError(arm)


def predict_one_choice_arm(arm: str, target: str, model, meta: dict, pk: str, m: float,
                           te: pd.DataFrame) -> np.ndarray:
    if arm == "proportional":
        return A.p1_probs(te, target, pk, m)
    if arm == "cond_logit":
        Xte, allnames = A.cl_design(te, target, pk, m)
        Xte = Xte[:, :, [allnames.index(n) for n in meta["names"]]]
        return model.predict_proba(Xte)
    if arm == "lgbm":
        Lte, _ = A.long_frame(te, target, pk, m)
        init = A.lgbm_init_score(te, target, pk, m)
        return model.predict_proba(Lte[:, meta["keep"]], init=init)
    raise KeyError(arm)


def fit_one_binary_arm(arm: str, target: str, tr: pd.DataFrame, frozen: dict, seed: int):
    fit = A.fit_binary_shrinkage(tr, target)
    feats = A.BINARY_FEATURES[arm]
    X = A.binary_matrix(tr, target, fit, feats)
    y = tr["b"].to_numpy()
    if arm == "lgbm":
        model = A.BinaryLgbmArm(seed=seed).fit(X, y)
    else:
        model = A.BinaryRidgeArm(C=frozen["C"], seed=seed).fit(X, y)
    return model, {"fit": fit, "feats": feats}


def predict_one_binary_arm(model, meta: dict, target: str, te: pd.DataFrame) -> np.ndarray:
    X = A.binary_matrix(te, target, meta["fit"], meta["feats"])
    return model.predict_proba(X)


def _persist_payload(model, arm: str, target: str) -> dict:
    """`LgbmChoiceArm.fit` passes a Python closure (`group_softmax_objective`'s
    `_obj`) as LightGBM's `objective=`, and the sklearn wrapper keeps a
    reference to it that `pickle`/`joblib` cannot serialise (`PicklingError:
    Can't pickle <function group_softmax_objective.<locals>._obj ...>`). The
    fitted booster itself carries no such reference, so choice-target lgbm
    arms are persisted as the booster's own portable text dump instead of the
    wrapper; every other arm class pickles directly (proven on `REB_off`
    [cond_logit], `blocked` [aware_ridge] and `steal` [proportional] before
    this was found)."""
    if arm == "lgbm" and target in A.CHOICE_TARGETS:
        return {"booster_model_str": model.clf_.booster_.model_to_string(),
               "n_alt": model.n_alt, "lgbm_params": model.params}
    return {"model": model}


def run_s1(target: str, design: pd.DataFrame, frozen: dict, seed: int,
          persist: bool, log) -> tuple[np.ndarray, pd.DataFrame, list[dict]]:
    """Monthly S1 refit of `TARGET_ARM[target]` across the F1 2025 test season.
    `possession_outcome.month_boundaries` decides the refit dates -- NOT
    reimplemented here, same as `train_fg_make_v2b_s1.py`."""
    arm = TARGET_ARM[target]
    is_choice = target in A.CHOICE_TARGETS
    tr_all, te_all = A.fold_slices(design)
    te_dates = pd.to_datetime(te_all["game_date"])
    tr_dates = pd.to_datetime(tr_all["game_date"])
    cuts = PO.month_boundaries(te_dates)
    k_alt = A.N_ALT_OF[target] if is_choice else 1
    p_s1 = (np.zeros((len(te_all), k_alt), dtype="float64") if is_choice
            else np.zeros(len(te_all), dtype="float64"))
    covered = np.zeros(len(te_all), dtype=bool)
    segments = []
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = ((te_dates >= cut) if nxt is None
               else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not seg.any():
            continue
        before = (te_dates < cut).to_numpy()
        prior = te_all.loc[before]
        rows = tr_all if not len(prior) else pd.concat([tr_all, prior], ignore_index=True)
        te_seg = te_all.loc[seg]
        t1 = time.time()
        if is_choice:
            shrink = A.fit_shrinkage(rows, target)
            pk, m = shrink["best"]["prior"], shrink["best"]["m"]
            model, meta = fit_one_choice_arm(arm, target, rows, pk, m, frozen, seed)
            p_s1[seg.nonzero()[0]] = predict_one_choice_arm(arm, target, model, meta, pk, m, te_seg)
            save = {"pk": pk, "m": m, "meta": meta}
        else:
            model, meta = fit_one_binary_arm(arm, target, rows, frozen, seed)
            p_s1[seg.nonzero()[0]] = predict_one_binary_arm(model, meta, target, te_seg)
            save = {"meta": meta}
        covered[seg.nonzero()[0]] = True
        max_train = (tr_dates.max() if not before.any()
                     else max(tr_dates.max(), te_dates[before].max()))
        rec = {"refit_date": str(cut.date()), "max_train_date": str(pd.Timestamp(max_train).date()),
               "n_train": int(len(rows)), "n_scored": int(seg.sum()),
               "fit_s": round(time.time() - t1, 1)}
        if persist:
            d = OUT / f"{target}_s1"
            d.mkdir(parents=True, exist_ok=True)
            fname = f"{cut.date()}.joblib"
            joblib.dump({**_persist_payload(model, arm, target),
                        "scheme": "S1", "fold": "F1", "season": 2025,
                        "target": target, "arm": arm, "seed": seed, **save,
                        "refit_date": rec["refit_date"], "max_train_date": rec["max_train_date"]},
                       d / fname)
            rec["path"] = f"{target}_s1/{fname}"
        segments.append(rec)
        log(f"    [S1 seed{seed}] {target} refit {cut.date()}: train {len(rows):,} "
            f"(test-season {len(prior):,}), scores {int(seg.sum()):,}, "
            f"{rec['fit_s']}s")
    if not covered.all():
        raise AssertionError(f"{target}: S1 (seed {seed}) left "
                             f"{int((~covered).sum())} test rows unscored")
    return p_s1, te_all, segments


def score_s1(target: str, p: np.ndarray, te: pd.DataFrame, driver) -> dict:
    if target in A.CHOICE_TARGETS:
        sc = A.score_choice_arm(te, p, driver, target)
        sc["bootstrap_se"] = round(A.bootstrap_se_choice(te, p, n_rep=200), 6)
        sc["game_level"] = A.choice_game_level_check(te, p, target, n_draw=40, seed=20260910)
    else:
        sc = A.score_binary_arm(te, p, driver, target)
        sc["bootstrap_se"] = round(A.bootstrap_se_binary(te, p, n_rep=200), 6)
        sc["game_level"] = A.binary_game_level_check(te, p, target, n_draw=40, seed=20260910)
    return sc


def eligible(sc: dict, target: str) -> bool:
    """Mirrors `train_attribution_v1.decide`'s own per-kind branching exactly.
    BUG FOUND POST-RUN: `binary_game_level_check` ALSO carries an `sd_pass` key
    (plus `gt0_pass`/`top_pass` set to `None`, "not applicable"), so sniffing
    for `"sd_pass" in gl` sends every binary target down the CHOICE branch,
    where `bool(None)` makes `gt0_pass`/`top_pass` always fail -- which is why
    the first run's `assisted`/`stolen`/`blocked` decisions all came back
    "neither passes" regardless of their real numbers. Branch on `target`'s
    OWN kind instead, the same way `decide()` does (`res["kind"]`). Fixed
    post-hoc from the already-computed scores; no retraining needed."""
    gl = sc["game_level"]
    ok = bool(sc["calib_pass"]) and bool(sc["resp_pass"])
    if target in A.CHOICE_TARGETS:
        ok = ok and bool(gl["sd_pass"]) and bool(gl["gt0_pass"]) and bool(gl["top_pass"])
    else:
        ok = ok and bool(gl["count_pass"]) and bool(gl["sd_pass"])
    return ok


def run_s1_for_target(target: str, pops, asof, team, frozen: dict, log) -> dict:
    design = build_target_design(target, pops, asof, team)
    arm = TARGET_ARM[target]

    # -- the reference driver for scoring: ONE global F1 shrinkage, computed on
    # the FULL 2024 train slice, held fixed across the whole S1 test season so
    # every month's rows are bucketed on the SAME definition of the driver.
    tr_all, te_all = A.fold_slices(design)
    if target in A.CHOICE_TARGETS:
        shrink0 = A.fit_shrinkage(tr_all, target)
        pk0, m0 = shrink0["best"]["prior"], shrink0["best"]["m"]
        driver_full = A.shrunk_rate(te_all, target, pk0, m0)
    else:
        fit0 = A.fit_binary_shrinkage(tr_all, target)
        driver_full = {}
        for side in ("off", "def"):
            lg = te_all[f"{side}_{target}_lg"].to_numpy(dtype="float64")
            driver_full[f"{side}_rate_c"] = A.team_shrunk(te_all, target, side, fit0["m_team"]) - lg

    t0 = time.time()
    p0, te0, seg0 = run_s1(target, design, frozen, seed=0, persist=True, log=log)
    sc_full = score_s1(target, p0, te0, driver_full)
    wf_mask = (pd.to_datetime(te0["game_date"]) >= WF_CUT).to_numpy()
    if target in A.CHOICE_TARGETS:
        driver_wf = driver_full[wf_mask]
    else:
        driver_wf = {k: v[wf_mask] for k, v in driver_full.items()}
    sc_wf = score_s1(target, p0[wf_mask], te0.loc[wf_mask].reset_index(drop=True), driver_wf)

    seed_sd = None
    if arm == "lgbm":
        p1, te1, _seg1 = run_s1(target, design, frozen, seed=1, persist=False, log=log)
        sc1_full = score_s1(target, p1, te1, driver_full)
        seed_sd = round(abs(sc1_full["log_loss"] - sc_full["log_loss"]), 6)

    (OUT / f"manifest_{target}.json").write_text(json.dumps({
        "model": "attribution", "scheme": "S1", "fold": "F1", "season": 2025,
        "key": target, "arm": arm,
        "artifacts": [{k: v for k, v in s.items()
                       if k in ("refit_date", "path", "max_train_date", "n_train")}
                      for s in seg0],
    }, indent=2), encoding="utf-8")

    log(f"  [S1] {target} ({arm}): F1 ll {sc_full['log_loss']:.6f} calib "
        f"{sc_full['calib_worst_gap_pp']:.3f}pp | WF-window ll {sc_wf['log_loss']:.6f} "
        f"calib {sc_wf['calib_worst_gap_pp']:.3f}pp | seed_sd {seed_sd} "
        f"({time.time() - t0:.1f}s)")
    return {"target": target, "arm": arm, "n_refits": len(seg0), "segments": seg0,
           "F1": sc_full, "WF2025_window": sc_wf, "seed_sd": seed_sd}


# ---------------------------------------------------------------------------
# 4. decision, per target: static-fixed vs S1-fixed vs round-1 (leaked)
# ---------------------------------------------------------------------------
def decide_round2(target: str, static_by, s1_res: dict, by_v1) -> dict:
    arm = TARGET_ARM[target]
    leaked_f1 = by_v1[("F1", target)]["arms"][arm]
    leaked_wf = by_v1[("WF2025", target)]["arms"][arm]
    static_f1 = static_by[("F1", target)]["arms"][arm]
    static_wf = static_by[("WF2025", target)]["arms"][arm]
    s1_f1, s1_wf = s1_res["F1"], s1_res["WF2025_window"]

    floor_static = static_f1["bootstrap_se"]
    floor_s1 = max(s1_f1["bootstrap_se"], s1_res["seed_sd"] or 0.0)
    floor = max(floor_static, floor_s1)

    # Eligibility gates the SELECTION fold (F1), exactly as round 1's own
    # `decide()` gates one fold at a time -- WF2025 is the ROBUSTNESS read and
    # feeds only the tie-break's calibration-gap clause, per the
    # pre-registration ("adopt the simplest of the arms that pass every gate
    # ... unless a more complex arm beats it ... on the within-2025
    # calibration gap"), never a second hard requirement layered on top.
    elig_static = eligible(static_f1, target)
    elig_s1 = eligible(s1_f1, target)

    ll_gap = s1_f1["log_loss"] - static_f1["log_loss"]
    calib_gap_static_wf = static_wf["calib_worst_gap_pp"]
    calib_gap_s1_wf = s1_wf["calib_worst_gap_pp"]
    calib_improves = calib_gap_s1_wf < calib_gap_static_wf - 0.1  # pp, well above float noise

    if elig_static and not (elig_s1 and (ll_gap < -floor or calib_improves)):
        winner, reason = "static_fixed", (
            "static-fixed passes F1; S1-fixed does not beat it beyond the floor "
            f"on log loss ({ll_gap:+.6f} vs floor {floor:.6f}) or on the WF2025 "
            f"calibration gap ({calib_gap_static_wf:.3f} -> {calib_gap_s1_wf:.3f} pp) "
            "-- simplest wins" + ("" if elig_s1 else "; S1-fixed also fails its own F1 gate"))
    elif elig_s1:
        winner, reason = "s1_fixed", (
            f"S1-fixed passes F1 and beats static-fixed beyond the floor on log "
            f"loss ({ll_gap:+.6f} vs floor {floor:.6f}) or closes the WF2025 "
            f"calibration gap ({calib_gap_static_wf:.3f} -> {calib_gap_s1_wf:.3f} pp)"
            + ("" if elig_static else "; static-fixed fails its own F1 gate"))
    else:
        winner, reason = None, "neither static-fixed nor S1-fixed passes F1 (the selection fold)"

    return {
        "target": target, "arm_class": arm, "reference_only": target in REFERENCE_ONLY,
        "round1_leaked": {"F1_log_loss": leaked_f1["log_loss"],
                          "F1_calib_pp": leaked_f1["calib_worst_gap_pp"],
                          "WF2025_log_loss": leaked_wf["log_loss"],
                          "WF2025_calib_pp": leaked_wf["calib_worst_gap_pp"]},
        "static_fixed": {"F1_log_loss": static_f1["log_loss"],
                        "F1_calib_pp": static_f1["calib_worst_gap_pp"],
                        "F1_eligible": eligible(static_f1, target),
                        "WF2025_log_loss": static_wf["log_loss"],
                        "WF2025_calib_pp": static_wf["calib_worst_gap_pp"],
                        "WF2025_eligible": eligible(static_wf, target),
                        "bootstrap_se": floor_static},
        "s1_fixed": {"F1_log_loss": s1_f1["log_loss"], "F1_calib_pp": s1_f1["calib_worst_gap_pp"],
                    "F1_eligible": eligible(s1_f1, target),
                    "WFwindow_log_loss": s1_wf["log_loss"], "WFwindow_calib_pp": s1_wf["calib_worst_gap_pp"],
                    "WFwindow_eligible": eligible(s1_wf, target), "seed_sd": s1_res["seed_sd"],
                    "floor": floor_s1},
        "floor": floor, "decision": winner, "reason": reason,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--targets", nargs="*", default=list(A.TARGETS))
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([2024, 2025], context="attribution round 2")

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    def flush_log() -> None:
        (OUT / "train_log_v2.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log("building pre-play-fixed population tables (asof/team reused from round 1)...")
    pops, asof, team, meta = build_fixed_tables(a.rebuild)
    log(f"tables built: {meta['n_rows']} asof {len(asof):,} team {len(team):,} "
        f"({time.time() - t0:.1f}s)")
    flush_log()

    frozen, by_v1 = load_frozen_params()

    args_static = argparse.Namespace(
        version="v2", seed=20260910, sim_draws=40, composed_draws=20, boot_reps=200,
        lgbm_grid=len(A.LGBM_PARAM_GRID), lgbm_seeds=len(V1.LGBM_SEEDS),
        inner_split=V1.INNER_SPLIT_DATE)
    log("\n=== (a) STATIC-FIXED: tri-arm rerun on pre-play-fixed data ===")
    static_results = run_static_fixed(pops, asof, team, args_static, log)
    (OUT / "static_results.json").write_text(
        json.dumps(static_results, indent=2, default=float), encoding="utf-8")
    V1.write_report(meta, static_results, OUT / "static_report.md", args_static,
                    partial_note="Round 2, arm (a): static fit, score_diff rebuilt "
                                 "pre-play. Tri-arm rerun of train_attribution_v1's "
                                 "own pipeline; only assist/assisted are expected to "
                                 "move vs round 1's leaked numbers.")
    static_by = {(r["fold"], r["target"]): r for r in static_results}
    flush_log()

    log("\n=== (b) S1-FIXED: one arm per target, monthly refit ===")
    s1_results, decisions = {}, {}
    for target in a.targets:
        s1_results[target] = run_s1_for_target(target, pops, asof, team, frozen[target], log)
        decisions[target] = decide_round2(target, static_by, s1_results[target], by_v1)
        log(f"  DECISION {target}: {decisions[target]['decision']} -- "
            f"{decisions[target]['reason']}")
        (OUT / "round2_summary.json").write_text(
            json.dumps({"targets": decisions, "s1_detail": s1_results}, indent=2, default=float),
            encoding="utf-8")
        flush_log()

    log(f"\nwrote {OUT}/round2_summary.json ({time.time() - t0:.1f}s total)")
    flush_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
