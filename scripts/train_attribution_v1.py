#!/usr/bin/env python
"""
train_attribution_v1.py -- the L4 PLAYER ATTRIBUTION bake-off.

Runs the pre-registration in `docs/models/attribution/experiments.md` section 1
end to end: builds the five population tables and their pregame as-of inputs,
fits the three choice arms on each of the five choice targets and the three
binary arms on each of the three binaries on F1 (train 2024, test 2025), scores
them on the pre-registered metric battery including the game-level checks
carried over from `usage`, repeats the whole thing on the within-2025
walk-forward fold, and writes a markdown report plus the fitted parameters.

Nothing in here decides anything by hand: the shrinkage prior and strength, the
ridge penalty and the LightGBM parameters are all fitted on training data only,
and the per-target winner falls out of the pre-registered decision rule in
`decide()`.

Usage:
    .venv/Scripts/python.exe scripts/train_attribution_v1.py --version v2
    .venv/Scripts/python.exe scripts/train_attribution_v1.py --targets steal --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import attribution as A  # noqa: E402
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.models import usage as U  # noqa: E402

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEASONS = (2024, 2025)
#: The internal split of the TRAINING fold, used to pick the ridge penalty and
#: the LightGBM parameters. "Parameter search on 2024 only" in the
#: pre-registration means exactly this: the test season is never seen.
INNER_SPLIT_DATE = "2024-01-15"
L2_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
C_GRID = (0.01, 0.1, 1.0, 10.0)
LGBM_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_tables(version: str, out_dir: Path, rebuild: bool):
    paths = {p: out_dir / f"events_{p}_{version}.parquet" for p in A.POPULATIONS}
    asof_path = out_dir / f"asof_{version}.parquet"
    team_path = out_dir / f"team_asof_{version}.parquet"
    meta_path = out_dir / f"build_report_{version}.json"
    if (all(p.exists() for p in paths.values()) and asof_path.exists()
            and team_path.exists() and meta_path.exists() and not rebuild):
        pops = {p: pd.read_parquet(paths[p]) for p in A.POPULATIONS}
        return (pops, pd.read_parquet(asof_path), pd.read_parquet(team_path),
                json.loads(meta_path.read_text()))

    universe = ES.load_universe(require_pbp_complete=True)
    rim = ES.rim_override_for_version(version)
    by_season, coverage, stream_meta = {}, {}, {}
    for s in SEASONS:
        by_season[s] = A.build_attr_events(s, universe, rim_override_max_ft=rim)
        coverage[str(s)] = A.coverage_report(by_season[s])
        stream_meta[str(s)] = by_season[s]["reb_off"].attrs.get("stream", {})
    minutes = U.load_minutes([SEASONS[0] - 1, *SEASONS])
    asof = A.build_player_asof(by_season, minutes=minutes)
    team = A.build_team_asof(by_season)
    pops = {p: pd.concat([by_season[s][p] for s in SEASONS], ignore_index=True)
            for p in A.POPULATIONS}
    meta = {
        "possessions_version": version,
        "rim_override_max_ft": rim,
        "coverage": coverage,
        "stream": stream_meta,
        "minutes_join_pct": asof.attrs.get("minutes_join_pct"),
        "n_rows": {p: int(len(pops[p])) for p in A.POPULATIONS},
        "n_player_games": int(len(asof)),
        "n_team_games": int(len(team)),
        "position_known_pct": round(
            float((asof["position_group"] != "UNK").mean() * 100), 4),
        "prior_season_pct": {str(s): round(float(
            asof.loc[asof.season == s, "has_prior_season"].mean() * 100), 4)
            for s in SEASONS},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, path in paths.items():
        pops[p].to_parquet(path, index=False)
    asof.to_parquet(asof_path, index=False)
    team.to_parquet(team_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=float))
    return pops, asof, team, meta


def inner_split(tr: pd.DataFrame, split_date: str):
    dt = pd.to_datetime(tr["game_date"])
    cut = pd.Timestamp(split_date)
    return tr[dt < cut].reset_index(drop=True), tr[dt >= cut].reset_index(drop=True)


def inner_for(tr: pd.DataFrame, fold: str, split: str):
    """The parameter-search split. On F1 it is the pre-registered "2024 only"
    date; on the within-season fold the training slice is already short, so it is
    halved at its own median date rather than at a date that would leave one side
    empty. If either side is too small the search falls back to in-fold, which is
    recorded in the report."""
    if fold != "F1":
        split = str(pd.to_datetime(tr["game_date"]).median().date())
    i_tr, i_va = inner_split(tr, split)
    if len(i_va) < 500 or len(i_tr) < 500:
        return tr, tr, split + " (too small; in-fold)"
    return i_tr, i_va, split


# ---------------------------------------------------------------------------
# One choice target, one fold
# ---------------------------------------------------------------------------
def cl_matrices(tr, te, target, pk, m):
    unid = A.unidentified_features(tr, target)
    Xtr, names = A.cl_design(tr, target, pk, m)
    keep = [i for i, n in enumerate(names) if n not in unid]
    Xtr, names = Xtr[:, :, keep], [names[i] for i in keep]
    Xtr, names, const = A.drop_constant_features(Xtr, names)
    Xte, allnames = A.cl_design(te, target, pk, m)
    Xte = Xte[:, :, [allnames.index(n) for n in names]]
    return Xtr, Xte, names, unid, const


def lgbm_matrices(tr, te, target, pk, m):
    unid = set(A.unidentified_features(tr, target))
    keep = [i for i, n in enumerate(A.LGBM_FEATURES) if n not in unid]
    kept = [A.LGBM_FEATURES[i] for i in keep]
    Ltr, ctr = A.long_frame(tr, target, pk, m)
    Lte, _ = A.long_frame(te, target, pk, m)
    return (Ltr[:, keep], ctr, Lte[:, keep], kept,
            sorted(unid & set(A.LGBM_FEATURES)),
            A.lgbm_init_score(tr, target, pk, m), A.lgbm_init_score(te, target, pk, m))


def run_choice_fold(target, tr, te, fold, args, log, lgbm_params=None) -> dict:
    t0 = time.time()
    k_alt = A.N_ALT_OF[target]
    log(f"  [{fold}] {target}: train {len(tr):,} test {len(te):,} (K={k_alt})")
    shrink = A.fit_shrinkage(tr, target)
    pk, m = shrink["best"]["prior"], shrink["best"]["m"]
    log(f"    shrinkage: prior={pk} m={m:g} (train ll {shrink['best']['train_log_loss']})")

    y_te = te["y"].to_numpy()
    driver = A.shrunk_rate(te, target, pk, m)
    probs, extra = {}, {}

    probs["proportional"] = A.p1_probs(te, target, pk, m)

    Xtr, Xte, names, unid, const = cl_matrices(tr, te, target, pk, m)
    i_tr, i_va, split = inner_for(tr, fold, args.inner_split)
    Xi_tr, _, inames, _, _ = cl_matrices(i_tr, i_va, target, pk, m)
    l2_rows, best_l2, best_ll = [], None, np.inf
    for l2 in L2_GRID:
        arm = A.CondLogitArm(l2=l2).fit(Xi_tr, i_tr["y"].to_numpy())
        Xv, allv = A.cl_design(i_va, target, pk, m)
        Xv = Xv[:, :, [allv.index(n) for n in inames]]
        ll = PM.log_loss(i_va["y"].to_numpy(), arm.predict_proba(Xv))
        l2_rows.append({"l2": l2, "inner_val_log_loss": round(ll, 6)})
        if ll < best_ll:
            best_l2, best_ll = l2, ll
    cl = A.CondLogitArm(l2=best_l2).fit(Xtr, tr["y"].to_numpy())
    probs["cond_logit"] = cl.predict_proba(Xte)
    extra["cond_logit"] = {"l2": best_l2, "l2_grid": l2_rows, "features": names,
                           "dropped_unidentified": unid, "dropped_constant": const,
                           "coefficients": {n: round(float(b), 5) for n, b
                                            in zip(names, cl.beta_, strict=False)},
                           "converged": cl.converged_}
    log(f"    cond_logit: l2={best_l2} ll {PM.log_loss(y_te, probs['cond_logit']):.6f}"
        f" dropped {unid + const}")

    Ltr, ctr, Lte, kept, dropped3, itr, ite = lgbm_matrices(tr, te, target, pk, m)
    Li_tr, ci_tr, Li_va, _, _, ii_tr, ii_va = lgbm_matrices(i_tr, i_va, target, pk, m)
    grid = (A.LGBM_PARAM_GRID[:args.lgbm_grid] if lgbm_params is None
            else (lgbm_params,))
    grid_rows, best_params, best_ll = [], None, np.inf
    for params in grid:
        arm = A.LgbmChoiceArm(k_alt, params, seed=args.seed).fit(Li_tr, ci_tr, init=ii_tr)
        ll = PM.log_loss(i_va["y"].to_numpy(), arm.predict_proba(Li_va, init=ii_va))
        grid_rows.append({**params, "inner_val_log_loss": round(ll, 6)})
        log(f"      lgbm {params}: inner val ll {ll:.6f}")
        if ll < best_ll:
            best_params, best_ll = params, ll
    seed_lls, imp = [], {}
    seeds = LGBM_SEEDS[:args.lgbm_seeds] if fold == "F1" else LGBM_SEEDS[:1]
    for sd in seeds:
        a3 = A.LgbmChoiceArm(k_alt, best_params, seed=sd).fit(Ltr, ctr, init=itr)
        p3 = a3.predict_proba(Lte, init=ite)
        seed_lls.append(PM.log_loss(y_te, p3))
        if sd == seeds[0]:
            probs["lgbm"] = p3
            imp = dict(zip(kept, a3.clf_.feature_importances_.tolist(), strict=False))
    extra["lgbm"] = {"params": best_params, "grid": grid_rows,
                     "dropped_unidentified": dropped3,
                     "seed_log_losses": [round(x, 6) for x in seed_lls],
                     "seed_sd": (round(float(np.std(seed_lls, ddof=1)), 6)
                                 if len(seed_lls) > 1 else None),
                     "feature_importance": imp}
    log(f"    lgbm: {best_params} ll {PM.log_loss(y_te, probs['lgbm']):.6f} "
        f"seed sd {extra['lgbm']['seed_sd']}")

    transfer = te[[f"is_transfer_{k}" for k in range(1, k_alt + 1)]].to_numpy()[
        np.arange(len(te)), y_te].astype(bool)
    has_prev = te[[f"has_prior_season_{k}" for k in range(1, k_alt + 1)]].to_numpy()[
        np.arange(len(te)), y_te] > 0
    rows = {}
    for arm in A.CHOICE_ARMS:
        p = probs[arm]
        sc = A.score_choice_arm(te, p, driver, target)
        # The tree's own floor is its seed-varied refit SD; the linear arms get
        # the game-block bootstrap. Both are reported for every arm.
        sc["bootstrap_se"] = round(A.bootstrap_se_choice(te, p, n_rep=args.boot_reps), 6)
        sc["game_level"] = A.choice_game_level_check(
            te, p, target, n_draw=args.sim_draws, seed=args.seed)
        sc["transfer"] = {
            "transfer_n": int(transfer.sum()),
            "transfer_log_loss": (round(PM.log_loss(y_te[transfer], p[transfer]), 6)
                                  if transfer.any() else None),
            "continuing_log_loss": (round(PM.log_loss(
                y_te[has_prev & ~transfer], p[has_prev & ~transfer]), 6)
                if (has_prev & ~transfer).any() else None),
            "no_prior_season_log_loss": (round(PM.log_loss(y_te[~has_prev],
                                                           p[~has_prev]), 6)
                                         if (~has_prev).any() else None)}
        sc.update(extra.get(arm, {}))
        rows[arm] = sc
        gl = sc["game_level"]
        log(f"    -> {arm:<13} ll {sc['log_loss']:.6f} calib {sc['calib_worst_gap_pp']:.3f}pp "
            f"resp {sc['resp_steps']}/4 slope {sc['resp_slope_ratio']} "
            f"sd {gl['sd_ratio']} gt0 {gl['players_gt0_delta']:+.3f} "
            f"top3 {gl['top3_gap_pp']:+.2f}pp")

    out = {"fold": fold, "target": target, "kind": "choice", "n_alt": k_alt,
           "uniform_log_loss": round(A.UNIFORM_LL[target], 6),
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "shrinkage": shrink, "prior_kind": pk, "shrink_m": m,
           "inner_split": split, "arms": rows,
           "seconds": round(time.time() - t0, 1)}
    out["decision"] = decide(out)
    log(f"    DECISION: {out['decision']['winner'] or 'NO WINNER'} -- "
        f"{out['decision']['reason']}")
    return out


# ---------------------------------------------------------------------------
# One binary target, one fold
# ---------------------------------------------------------------------------
def run_binary_fold(binary, tr, te, fold, args, log) -> dict:
    t0 = time.time()
    log(f"  [{fold}] {binary}: train {len(tr):,} test {len(te):,}")
    fit = A.fit_binary_shrinkage(tr, binary)
    log(f"    shrinkage: m_team={fit['m_team']:g} own_prior={fit['own_prior']} "
        f"m_own={fit['m_own']:g}")
    y_te = te["b"].to_numpy()
    drivers = {}
    for side in ("off", "def"):
        lg = te[f"{side}_{binary}_lg"].to_numpy(dtype="float64")
        drivers[f"{side}_rate_c"] = A.team_shrunk(te, binary, side, fit["m_team"]) - lg

    i_tr, i_va, split = inner_for(tr, fold, args.inner_split)
    probs, extra = {}, {}
    for arm in A.BINARY_ARMS:
        feats = A.BINARY_FEATURES[arm]
        Xtr = A.binary_matrix(tr, binary, fit, feats)
        Xte = A.binary_matrix(te, binary, fit, feats)
        if arm == "lgbm":
            seed_lls = []
            seeds = LGBM_SEEDS[:args.lgbm_seeds] if fold == "F1" else LGBM_SEEDS[:1]
            for sd in seeds:
                mdl = A.BinaryLgbmArm(seed=sd).fit(Xtr, y_tr := tr["b"].to_numpy())
                p = mdl.predict_proba(Xte)
                seed_lls.append(PM.log_loss(y_te, np.column_stack([1 - p, p])))
                if sd == seeds[0]:
                    probs[arm] = p
                    imp = dict(zip(feats, mdl.clf_.feature_importances_.tolist(),
                                   strict=False))
            extra[arm] = {"params": A.BinaryLgbmArm.PARAMS, "features": list(feats),
                          "seed_log_losses": [round(x, 6) for x in seed_lls],
                          "seed_sd": (round(float(np.std(seed_lls, ddof=1)), 6)
                                      if len(seed_lls) > 1 else None),
                          "feature_importance": imp}
        else:
            Xi_tr = A.binary_matrix(i_tr, binary, fit, feats)
            Xi_va = A.binary_matrix(i_va, binary, fit, feats)
            c_rows, best_c, best_ll = [], None, np.inf
            for C in C_GRID:
                mdl = A.BinaryRidgeArm(C=C, seed=args.seed).fit(Xi_tr, i_tr["b"].to_numpy())
                pv = mdl.predict_proba(Xi_va)
                ll = PM.log_loss(i_va["b"].to_numpy(), np.column_stack([1 - pv, pv]))
                c_rows.append({"C": C, "inner_val_log_loss": round(ll, 6)})
                if ll < best_ll:
                    best_c, best_ll = C, ll
            mdl = A.BinaryRidgeArm(C=best_c, seed=args.seed).fit(Xtr, tr["b"].to_numpy())
            probs[arm] = mdl.predict_proba(Xte)
            extra[arm] = {"C": best_c, "C_grid": c_rows, "features": list(feats),
                          "coefficients": {n: round(float(b), 5) for n, b
                                           in zip(feats, mdl.clf_.coef_[0], strict=False)}}
    rows = {}
    for arm in A.BINARY_ARMS:
        p = probs[arm]
        sc = A.score_binary_arm(te, p, drivers, binary)
        sc["bootstrap_se"] = round(A.bootstrap_se_binary(te, p, n_rep=args.boot_reps), 6)
        sc["game_level"] = A.binary_game_level_check(
            te, p, binary, n_draw=args.sim_draws, seed=args.seed)
        sc.update(extra.get(arm, {}))
        rows[arm] = sc
        gl = sc["game_level"]
        log(f"    -> {arm:<13} ll {sc['log_loss']:.6f} calib {sc['calib_worst_gap_pp']:.3f}pp "
            f"resp {sc['resp_steps']}/4 slopes {sc['resp_slope_ratio']} "
            f"count {gl['count_sim']:.3f}/{gl['count_actual']:.3f} sd {gl['sd_ratio']}")
    out = {"fold": fold, "target": binary, "kind": "binary",
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "base_rate_pct": round(float(y_te.mean() * 100), 4),
           "shrinkage": fit, "inner_split": split, "arms": rows,
           "seconds": round(time.time() - t0, 1)}
    out["decision"] = decide(out)
    log(f"    DECISION: {out['decision']['winner'] or 'NO WINNER'} -- "
        f"{out['decision']['reason']}")
    return out


# ---------------------------------------------------------------------------
# The pre-registered decision rule
# ---------------------------------------------------------------------------
def decide(res: dict) -> dict:
    """Per target, the winner is the lowest log loss among the arms that pass
    calibration, responsiveness AND the game-level checks; the tree must
    additionally beat the best non-tree eligible arm by more than the floor; ties
    go to the simpler arm (P1 < P2 < P3; team ridge < aware ridge < tree). If no
    arm is eligible, adopt nothing."""
    target = res["target"]
    order = A.arm_order(target)
    elig, why = [], {}
    for arm, r in res["arms"].items():
        gl = r["game_level"]
        checks = {"calibration": bool(r["calib_pass"]),
                  "responsiveness": bool(r["resp_pass"])}
        if res["kind"] == "choice":
            checks.update({"sd_ratio": bool(gl["sd_pass"]),
                           "players_gt0": bool(gl["gt0_pass"]),
                           "top1_top3": bool(gl["top_pass"])})
        else:
            # A binary credits no player, so the per-player trio has no player
            # axis; the same-family statistic one level up is the per-team-game
            # COUNT and its dispersion (`attribution.binary_game_level_check`).
            checks.update({"count_level": bool(gl["count_pass"]),
                           "count_sd_ratio": bool(gl["sd_pass"])})
        why[arm] = checks
        if all(checks.values()):
            elig.append(arm)
    if not elig:
        return {"winner": None, "eligible": [], "checks": why,
                "reason": "no arm passes every pre-registered gate"}
    floor = max(res["arms"][a]["bootstrap_se"] for a in elig)
    tree_sd = res["arms"].get(A.TREE_ARM, {}).get("seed_sd")
    if tree_sd:
        floor = max(floor, float(tree_sd))
    ranked = sorted(elig, key=lambda a: (res["arms"][a]["log_loss"], order[a]))
    best = ranked[0]
    non_tree = [a for a in ranked if a != A.TREE_ARM]
    if best == A.TREE_ARM and non_tree:
        alt = non_tree[0]
        gap = res["arms"][alt]["log_loss"] - res["arms"][best]["log_loss"]
        if gap <= floor:
            nt_tied = [a for a in non_tree
                       if res["arms"][a]["log_loss"] - res["arms"][alt]["log_loss"] <= floor]
            nt_winner = min(nt_tied, key=lambda a: order[a])
            return {"winner": nt_winner, "eligible": elig, "checks": why,
                    "floor": floor, "tied_within_floor": nt_tied,
                    "reason": (f"the tree leads by {gap:.6f}, inside the {floor:.6f} "
                               f"floor, so the pre-registration's requirement that a "
                               f"tree beat the best passing non-tree arm by more than "
                               f"the floor is not met; among the non-tree arms "
                               f"{', '.join(nt_tied)} are inside the floor of each "
                               f"other and the tie-break takes the simplest")}
    tied = [a for a in ranked
            if res["arms"][a]["log_loss"] - res["arms"][best]["log_loss"] <= floor]
    winner = min(tied, key=lambda a: order[a])
    gap_txt = ", ".join(f"{a} {res['arms'][a]['log_loss']:.6f}" for a in ranked)
    if len(tied) == 1:
        if len(ranked) > 1:
            d = res["arms"][ranked[1]]["log_loss"] - res["arms"][best]["log_loss"]
            tail = (f"clear of the next eligible arm by {d:.6f} "
                    f"({d / floor:.1f} floors)")
        else:
            tail = "the only eligible arm"
    else:
        tail = (f"{', '.join(tied)} are inside the floor of each other; the "
                f"pre-registered tie-break takes the simplest")
    return {"winner": winner, "eligible": elig, "checks": why, "floor": floor,
            "tied_within_floor": tied,
            "reason": f"eligible: {gap_txt}; floor {floor:.6f}; {tail}"}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_report(meta: dict, results: list[dict], path: Path, args,
                 partial_note: str | None = None) -> None:
    L: list[str] = []
    A_ = L.append
    if partial_note:
        A_(partial_note + "\n")
    A_(f"## 2. Run configuration ({time.strftime('%Y-%m-%d')})\n")
    A_("| item | value |")
    A_("|---|---|")
    A_("| trainer | `scripts/train_attribution_v1.py` |")
    A_(f"| possessions version | `{meta['possessions_version']}` "
       f"(rim override {meta['rim_override_max_ft']} ft) |")
    A_("| universe | D-I, non-truncated, `pbp_complete` |")
    A_("| population rows (2024 + 2025) | "
       + ", ".join(f"{k} {v:,}" for k, v in meta["n_rows"].items()) + " |")
    A_(f"| player-games with as-of inputs | {meta['n_player_games']:,} |")
    A_(f"| team-games with as-of inputs | {meta['n_team_games']:,} |")
    A_(f"| roster position known | {meta['position_known_pct']}% |")
    A_(f"| hoopR minutes joined through the player crosswalk | {meta['minutes_join_pct']}% |")
    A_("| player-games with a prior season of history | "
       + ", ".join(f"{k}: {v}%" for k, v in meta["prior_season_pct"].items()) + " |")
    A_(f"| game-level / composed draws, bootstrap reps | {args.sim_draws} / "
       f"{args.composed_draws} / {args.boot_reps} |")
    A_("")
    A_("### 2.1 Target coverage (reported, not silently filtered)\n")
    A_("| season | target | population | binary rate | candidate set resolved | "
       "credited id present | modelled |")
    A_("|---|---|---:|---:|---:|---:|---:|")
    for s, cov in meta["coverage"].items():
        for t, r in cov.items():
            A_(f"| {s} | {t} | {r['n_population']:,} | {r['binary_rate_pct']}% | "
               f"{r['five_resolved_pct']}% | {r['credit_id_present_pct']}% | "
               f"{r['modelled_pct']}% |")
    A_("")

    for fold, title in (("F1", "## 3. F1 results (train 2024, test 2025) -- the "
                               "selection fold"),
                        ("WF2025", f"## 4. Robustness fold: within-2025 walk-forward "
                                   f"(train before {A.WF_SPLIT_DATE}, test after)")):
        rs = [r for r in results if r["fold"] == fold]
        if not rs:
            continue
        A_(title + "\n")
        ch = [r for r in rs if r["kind"] == "choice"]
        if ch:
            A_("### Choice targets\n")
            for res in ch:
                A_(f"**{res['target']}** (K = {res['n_alt']}, uniform log loss "
                   f"{res['uniform_log_loss']:.6f}). Train {res['n_train']:,}, test "
                   f"{res['n_test']:,}. Fitted shrinkage: prior "
                   f"`{res['prior_kind']}`, m = {res['shrink_m']:g} pseudo "
                   f"opportunities.\n")
                A_("| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | "
                   "resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) "
                   "| top-1 (sim / real) | top-3 gap | eligible |")
                A_("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|")
                for arm in A.CHOICE_ARMS:
                    r = res["arms"][arm]
                    gl = r["game_level"]
                    ok = res["decision"]["checks"][arm]
                    A_(f"| {arm} | {r['log_loss']:.6f} | {r['brier']:.4f} | "
                       f"{r['top1']:.4f} | {r['top3']:.4f} | "
                       f"{r['calib_worst_gap_pp']:.3f} | {r['resp_steps']}/4 | "
                       f"{fmt(r['resp_slope_ratio'], 3)} | {r['bootstrap_se']:.6f} | "
                       f"{fmt(gl['sd_ratio'], 4)} | "
                       f"{fmt(gl['players_gt0_sim'], 3)} / "
                       f"{fmt(gl['players_gt0_actual'], 3)} | "
                       f"{fmt(gl['top1_sim_pct'], 2)}% / "
                       f"{fmt(gl['top1_actual_pct'], 2)}% | "
                       f"{fmt(gl['top3_gap_pp'], 2)} | "
                       f"{'yes' if all(ok.values()) else 'NO (' + ', '.join(k for k, v in ok.items() if not v) + ')'} |")
                A_("")
                A_(f"**Decision: {res['decision']['winner'] or 'NO WINNER'}.** "
                   f"{res['decision']['reason']}\n")
                cl = res["arms"]["cond_logit"]
                A_(f"P2 ridge penalty {cl['l2']} (searched on "
                   f"`{res['inner_split']}`); dropped as unidentified: "
                   f"`{cl['dropped_unidentified']}`; dropped as constant in this "
                   f"fold: `{cl['dropped_constant']}`. Coefficients: "
                   f"`{cl['coefficients']}`.\n")
                lg = res["arms"]["lgbm"]
                A_(f"P3 parameters {lg['params']}; seed-varied refits "
                   f"{lg['seed_log_losses']}, seed SD {lg['seed_sd']}; feature "
                   f"importance `{lg['feature_importance']}`.\n")
                A_("Transfer subset:\n")
                A_("| arm | transfers | continuing | no prior season |")
                A_("|---|---:|---:|---:|")
                for arm in A.CHOICE_ARMS:
                    t = res["arms"][arm]["transfer"]
                    A_(f"| {arm} | {fmt(t['transfer_log_loss'], 6)} | "
                       f"{fmt(t['continuing_log_loss'], 6)} | "
                       f"{fmt(t['no_prior_season_log_loss'], 6)} |")
                A_(f"\n(n transfers = "
                   f"{res['arms']['proportional']['transfer']['transfer_n']:,})\n")
        bn = [r for r in rs if r["kind"] == "binary"]
        if bn:
            A_("### Binary targets\n")
            A_("| target | base rate | arm | log loss | Brier | calib worst (pp) | "
               "resp steps | slopes (off / def) | boot SE | count (sim / real) | "
               "count SD ratio | eligible |")
            A_("|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|---|")
            for res in bn:
                for arm in A.BINARY_ARMS:
                    r = res["arms"][arm]
                    gl = r["game_level"]
                    ok = res["decision"]["checks"][arm]
                    sl = r["resp_slope_ratio"]
                    A_(f"| {res['target']} | {res['base_rate_pct']}% | {arm} | "
                       f"{r['log_loss']:.6f} | {r['brier']:.6f} | "
                       f"{r['calib_worst_gap_pp']:.3f} | {r['resp_steps']}/4 | "
                       f"{fmt(sl.get('off_rate_c'), 3)} / {fmt(sl.get('def_rate_c'), 3)} | "
                       f"{r['bootstrap_se']:.6f} | {gl['count_sim']:.3f} / "
                       f"{gl['count_actual']:.3f} | {fmt(gl['sd_ratio'], 4)} | "
                       f"{'yes' if all(ok.values()) else 'NO (' + ', '.join(k for k, v in ok.items() if not v) + ')'} |")
            A_("")
            for res in bn:
                A_(f"**{res['target']} decision: "
                   f"{res['decision']['winner'] or 'NO WINNER'}.** "
                   f"{res['decision']['reason']} Fitted shrinkage: m_team "
                   f"{res['shrinkage']['m_team']:g}, own-share prior "
                   f"`{res['shrinkage']['own_prior']}` m_own "
                   f"{res['shrinkage']['m_own']:g}.\n")
    comp = [r for r in results if r.get("kind") == "composed"]
    if comp:
        A_("## 5. Composed per-player check (REPORTED DIAGNOSTIC, not a gate)\n")
        A_("Binary then choice, drawn over the actual event sequence with the "
           "actual fives, accumulated per (team-game, player). This is the "
           "quantity a player prop needs; it is not in the pre-registration's "
           "decision rule.\n")
        A_("| target | arms | credits per team-game (sim / real) | SD ratio | "
           "players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) |")
        A_("|---|---|---|---:|---|---|---|")
        for r in comp:
            c = r["check"]
            A_(f"| {r['target']} | {r['binary_arm']} + {r['choice_arm']} | "
               f"{c['credits_per_team_game_sim']} / "
               f"{c['credits_per_team_game_actual']} | {fmt(c['sd_ratio'], 4)} | "
               f"{c['players_gt0_sim']} / {c['players_gt0_actual']} | "
               f"{c['top1_sim_pct']}% / {c['top1_actual_pct']}% | "
               f"{c['top3_sim_pct']}% / {c['top3_actual_pct']}% |")
        A_("")
    path.write_text("\n".join(L), encoding="utf-8")


def write_results(out_dir: Path, report: Path, meta: dict, results: list[dict],
                  args, partial_note: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results_v1.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    params = {"possessions_version": args.version, "seed": args.seed,
              "per_target": {}}
    for r in results:
        if r["fold"] != "F1" or r.get("kind") == "composed":
            continue
        e = {"kind": r["kind"], "winner": r["decision"]["winner"],
             "eligible": r["decision"]["eligible"]}
        if r["kind"] == "choice":
            e.update({"n_alt": r["n_alt"], "prior_kind": r["prior_kind"],
                      "shrink_m": r["shrink_m"],
                      "cond_logit_features": r["arms"]["cond_logit"]["features"],
                      "cond_logit_l2": r["arms"]["cond_logit"]["l2"],
                      "cond_logit_coefficients": r["arms"]["cond_logit"]["coefficients"],
                      "lgbm_params": r["arms"]["lgbm"]["params"],
                      "rng_family": A.FAMILY_OF[r["target"]]})
        else:
            e.update({"m_team": r["shrinkage"]["m_team"],
                      "m_own": r["shrinkage"]["m_own"],
                      "own_prior": r["shrinkage"]["own_prior"],
                      "features": {a: list(A.BINARY_FEATURES[a])
                                   for a in A.BINARY_ARMS},
                      "ridge_coefficients": {
                          a: r["arms"][a].get("coefficients")
                          for a in ("team_ridge", "aware_ridge")},
                      "rng_family": A.FAMILY_OF[r["target"]]})
        params["per_target"][r["target"]] = e
    (out_dir / "attribution_params_v1.json").write_text(
        json.dumps(params, indent=2, default=float), encoding="utf-8")
    write_report(meta, results, report, args, partial_note)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--out", default="data/processed/models/attribution")
    ap.add_argument("--report", default=None)
    ap.add_argument("--targets", nargs="*", default=list(A.TARGETS))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--sim-draws", type=int, default=40)
    ap.add_argument("--composed-draws", type=int, default=20)
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--lgbm-grid", type=int, default=len(A.LGBM_PARAM_GRID))
    ap.add_argument("--lgbm-seeds", type=int, default=len(LGBM_SEEDS))
    ap.add_argument("--inner-split", default=INNER_SPLIT_DATE)
    ap.add_argument("--skip-wf", action="store_true")
    ap.add_argument("--skip-composed", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--partial-note", default=None,
                    help="banner prepended to the report when the run is known "
                         "to be incomplete")
    args = ap.parse_args()
    if args.quick:
        args.sim_draws, args.composed_draws, args.boot_reps = 8, 5, 50
        args.lgbm_grid, args.lgbm_seeds = 1, 2

    out_dir = ROOT / args.out
    report = Path(args.report) if args.report else out_dir / "report_v1.md"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    def flush_log() -> None:
        (out_dir / "train_log_v1.txt").write_text("\n".join(log_lines),
                                                  encoding="utf-8")

    t0 = time.time()
    pops, asof, team, meta = build_tables(args.version, out_dir, args.rebuild)
    log(f"tables built: {meta['n_rows']} asof {len(asof):,} team {len(team):,} "
        f"({time.time() - t0:.1f}s)")
    if args.build_only:
        flush_log()
        return 0

    results: list[dict] = []
    for target in args.targets:
        try:
            if target in A.CHOICE_TARGETS:
                design = A.build_choice_design(
                    A.usable(pops[A.POP_OF[target]], target), asof, target)
                tr, te = A.fold_slices(design)
                f1 = run_choice_fold(target, tr, te, "F1", args, log)
                results.append(f1)
                if not args.skip_wf:
                    wtr, wte = A.walkforward_slices(design)
                    results.append(run_choice_fold(
                        target, wtr, wte, "WF2025", args, log,
                        lgbm_params=f1["arms"]["lgbm"]["params"]))
                del design
            else:
                p = pops[A.POP_OF[target]]
                design = A.build_binary_design(
                    p[p["five_ok"].to_numpy()], asof, team, target)
                tr, te = A.fold_slices(design)
                results.append(run_binary_fold(target, tr, te, "F1", args, log))
                if not args.skip_wf:
                    wtr, wte = A.walkforward_slices(design)
                    results.append(run_binary_fold(target, wtr, wte, "WF2025",
                                                   args, log))
                del design
        except Exception as exc:                       # noqa: BLE001
            log(f"  [FAILED] {target}: {type(exc).__name__}: {exc}")
        write_results(out_dir, report, meta, results, args, args.partial_note)
        flush_log()
    log(f"\nwrote {report} and {out_dir}/attribution_params_v1.json "
        f"({time.time() - t0:.1f}s total)")
    flush_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
