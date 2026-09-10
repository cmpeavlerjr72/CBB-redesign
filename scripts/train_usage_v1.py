#!/usr/bin/env python
"""
train_usage_v1.py -- the L4 SHOT ALLOCATION (usage) bake-off.

Runs the pre-registration in `docs/models/usage/experiments.md` section 1 end to
end: builds the credited-event table and its pregame as-of inputs, fits the five
arms per event class on F1 (train 2024, test 2025), scores them on the
pre-registered metric battery including the CFB "too narrow / too short" pair,
repeats the whole thing on the within-2025 walk-forward fold, and writes a
markdown report plus the fitted parameters.

Nothing in here decides anything by hand: the shrinkage prior and strength, the
Dirichlet concentrations, the ridge penalty and the LightGBM parameters are all
fitted on training data only, and the per-class winner falls out of the
pre-registered decision rule in `decide()`.

Usage:
    .venv/Scripts/python.exe scripts/train_usage_v1.py --version v2
    .venv/Scripts/python.exe scripts/train_usage_v1.py --classes FGA_3 --quick
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

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.models import usage as U  # noqa: E402

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SEASONS = (2024, 2025)
#: The internal split of the TRAINING fold, used to pick the ridge penalty and
#: the LightGBM parameters. "Parameter search on 2024 only" in the
#: pre-registration means exactly this: the test season is never seen.
INNER_SPLIT_DATE = "2024-01-15"
L2_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
LGBM_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_tables(version: str, out_dir: Path, rebuild: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ev_path = out_dir / f"events_{version}.parquet"
    asof_path = out_dir / f"asof_{version}.parquet"
    meta_path = out_dir / f"build_report_{version}.json"
    if ev_path.exists() and asof_path.exists() and meta_path.exists() and not rebuild:
        return (pd.read_parquet(ev_path), pd.read_parquet(asof_path),
                json.loads(meta_path.read_text()))

    universe = ES.load_universe(require_pbp_complete=True)
    rim = ES.rim_override_for_version(version)
    per_season, coverage = {}, {}
    for s in SEASONS:
        raw = U.build_usage_events(s, universe, rim_override_max_ft=rim)
        coverage[str(s)] = U.coverage_report(raw)
        per_season[s] = U.usable_events(raw)
    minutes = U.load_minutes([SEASONS[0] - 1, *SEASONS])
    asof = U.build_player_asof(per_season, minutes=minutes)
    events = pd.concat([per_season[s] for s in SEASONS], ignore_index=True)
    meta = {
        "possessions_version": version,
        "rim_override_max_ft": rim,
        "coverage": coverage,
        "minutes_join_pct": asof.attrs.get("minutes_join_pct"),
        "n_events": int(len(events)),
        "n_player_games": int(len(asof)),
        "position_known_pct": round(
            float((asof["position_group"] != "UNK").mean() * 100), 4),
        "prior_season_pct": {str(s): round(float(
            asof.loc[asof.season == s, "has_prior_season"].mean() * 100), 4)
            for s in SEASONS},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(ev_path, index=False)
    asof.to_parquet(asof_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    return events, asof, meta


# ---------------------------------------------------------------------------
# Arm fitting
# ---------------------------------------------------------------------------
def inner_split(tr: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt = pd.to_datetime(tr["game_date"])
    cut = pd.Timestamp(split_date)
    return tr[dt < cut].reset_index(drop=True), tr[dt >= cut].reset_index(drop=True)


def cl_matrices(tr, te, cls, pk, m):
    """Conditional-logit designs with the unidentified and constant columns
    dropped on the TRAINING fold and the same columns removed from the test."""
    unid = U.unidentified_features(tr)
    Xtr, names = U.cl_design(tr, cls, pk, m)
    keep = [i for i, n in enumerate(names) if n not in unid]
    Xtr, names = Xtr[:, :, keep], [names[i] for i in keep]
    Xtr, names, const = U.drop_constant_features(Xtr, names)
    Xte, allnames = U.cl_design(te, cls, pk, m)
    Xte = Xte[:, :, [allnames.index(n) for n in names]]
    return Xtr, Xte, names, unid, const


def lgbm_matrices(tr, te, cls, pk, m):
    unid = set(U.unidentified_features(tr))
    keep = [i for i, n in enumerate(U.LGBM_FEATURES) if n not in unid]
    kept = [U.LGBM_FEATURES[i] for i in keep]
    Ltr, ctr = U.long_frame(tr, cls, pk, m)
    Lte, _ = U.long_frame(te, cls, pk, m)
    return (Ltr[:, keep], ctr, Lte[:, keep], kept, sorted(unid & set(U.LGBM_FEATURES)),
            U.lgbm_init_score(tr, cls, pk, m), U.lgbm_init_score(te, cls, pk, m))


def fit_alphas(tr: pd.DataFrame, cls: str, pk: str, m: float, hier: bool,
               n_draw: int, max_tg: int, seed: int, log) -> dict:
    """Fit the Dirichlet concentration(s) by matching the TRAINING fold's own
    per-player per-game count dispersion.

    Per-event log loss cannot fit these: a mean-preserving Dirichlet is centred
    on U1's shares, so its marginal predictive is U1's up to the subsetting term
    and the likelihood is nearly flat in the concentration. The dispersion IS
    what the layer exists to change, so the dispersion is what it is fitted
    against -- the CLAUDE.md rule that a model's variance function is validated
    against realised residual SD, applied as a fit rather than as a check.

    U2 (single level) fits one concentration against the pooled SD ratio.
    U3 fits each role's concentration against THAT ROLE's own SD ratio with the
    between level off, then the between concentration against the pooled ratio
    with the within levels fixed -- one moment per parameter, so nothing is
    fitted twice against the same number.
    """
    sub = subsample_team_games(tr, max_tg, seed)
    rates = U.shrunk_rate(sub, cls, pk, m)
    roles = U.assign_roles(sub, pk, m)
    ps_h = U.build_profiles(sub, rates, roles)
    prole = U.profile_player_roles(ps_h)
    grid_rows = []

    def ratio(ps, alphas, role=None):
        r = U.game_level_check(sub, ps=ps, alphas=alphas, n_draw=n_draw, seed=seed,
                               player_role=prole)
        if role is None:
            return r["sd_ratio"], r
        return r["sd_ratio_by_role"].get(role, {}).get("sd_ratio", np.nan), r

    if not hier:
        ps = ps_h.single_role()
        best, best_gap = None, np.inf
        for a in U.ALPHA_GRID:
            v, rep = ratio(ps, U.single_level_alphas(a))
            grid_rows.append({"knob": "a", "value": a, "sd_ratio": v,
                              "players_gt0": rep["players_gt0_sim"],
                              "top1_pp": rep["top1_gap_pp"]})
            log(f"      a={a}: sd_ratio {v:.4f}")
            if abs(v - 1.0) < best_gap:
                best, best_gap = a, abs(v - 1.0)
        return {"alphas": U.single_level_alphas(best), "grid": grid_rows,
                "fitted": {"a": best}, "single_level": True}

    fitted = {}
    for role in U.ROLES:
        best, best_gap = None, np.inf
        for a in U.ALPHA_GRID:
            al = {"a_between": float("inf"),
                  **{f"a_{r}": (float(a) if r == role else float("inf"))
                     for r in U.ROLES}}
            v, _ = ratio(ps_h, al, role=role)
            grid_rows.append({"knob": f"a_{role}", "value": a, "sd_ratio_role": v})
            log(f"      a_{role}={a}: role sd_ratio {v:.4f}")
            if np.isfinite(v) and abs(v - 1.0) < best_gap:
                best, best_gap = a, abs(v - 1.0)
        if best is None:
            log(f"      a_{role}: role SD ratio undefined on every rung "
                f"(too few graded players); left at no dispersion")
        fitted[f"a_{role}"] = float(best) if best is not None else float("inf")
    best_b, gap_b = None, np.inf
    for a in U.ALPHA_GRID:
        al = {"a_between": float(a), **fitted}
        v, rep = ratio(ps_h, al)
        grid_rows.append({"knob": "a_between", "value": a, "sd_ratio": v,
                          "players_gt0": rep["players_gt0_sim"]})
        log(f"      a_between={a}: sd_ratio {v:.4f}")
        if abs(v - 1.0) < gap_b:
            best_b, gap_b = a, abs(v - 1.0)
    alphas = {"a_between": float(best_b), **fitted}
    return {"alphas": alphas, "grid": grid_rows, "fitted": alphas,
            "single_level": False}


def subsample_team_games(d: pd.DataFrame, max_tg: int, seed: int) -> pd.DataFrame:
    """A random subset of whole TEAMS, sized to about `max_tg` team-games.

    The unit is the TEAM, not the team-game: the moment being fitted is the SD
    across a player's own games of his per-game count, so a subsample that keeps
    a scattered 30% of each team's games would leave every player below the
    minimum-games threshold and the statistic would be measured on nothing but
    outliers (observed: the per-role ratio came back undefined for every role
    under a team-game subsample). Sampling whole teams keeps every sampled
    player's full game log."""
    teams = d["team_id"].drop_duplicates().to_numpy()
    n_tg = d.groupby(["game_id", "team_id"], sort=False).ngroups
    if n_tg <= max_tg:
        return d
    rng = np.random.default_rng(seed)
    per_team = max(n_tg / max(len(teams), 1), 1e-9)
    n_keep = int(np.clip(round(max_tg / per_team), 1, len(teams)))
    keep = rng.choice(teams, n_keep, replace=False)
    return d[d["team_id"].isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------
def run_fold(cls: str, tr: pd.DataFrame, te: pd.DataFrame, fold: str,
             args, log, lgbm_params: dict | None = None) -> dict:
    t0 = time.time()
    log(f"  [{fold}] {cls}: train {len(tr):,} test {len(te):,}")
    shrink = U.fit_shrinkage(tr, cls)
    pk, m = shrink["best"]["prior"], shrink["best"]["m"]
    log(f"    shrinkage: prior={pk} m={m} (train ll {shrink['best']['train_log_loss']})")

    y_te = te["y"].to_numpy()
    driver = U.shrunk_rate(te, cls, pk, m)
    rates_te = driver
    roles_te = U.assign_roles(te, pk, m)
    ps_te = U.build_profiles(te, rates_te, roles_te)
    prole_te = U.profile_player_roles(ps_te)

    probs: dict[str, np.ndarray] = {}
    extra: dict[str, dict] = {}

    # ---- U1 ---------------------------------------------------------------
    probs["proportional"] = U.u1_probs(te, cls, pk, m)

    # ---- U2 / U3 ----------------------------------------------------------
    alpha_fits = {}
    for arm, hier in (("dirichlet", False), ("hier_dirichlet", True)):
        fit = fit_alphas(tr, cls, pk, m, hier, args.alpha_draws, args.alpha_max_tg,
                         args.seed, log)
        alpha_fits[arm] = fit
        ps = ps_te.single_role() if not hier else ps_te
        probs[arm] = U.marginal_probs(ps, fit["alphas"], n_draw=args.mc_draws,
                                      seed=args.seed)
        seq = U.sequential_probs(ps, y_te, fit["alphas"], n_draw=args.seq_draws,
                                 seed=args.seed)
        extra[arm] = {"alphas": {k: (None if not np.isfinite(v) else v)
                                 for k, v in fit["alphas"].items()},
                      "alpha_grid": fit["grid"],
                      "polya_urn_log_loss": round(PM.log_loss(y_te, seq), 6)}
        log(f"    {arm}: alphas {fit['fitted']} marginal ll "
            f"{PM.log_loss(y_te, probs[arm]):.6f} polya {extra[arm]['polya_urn_log_loss']}")

    # ---- U4 ---------------------------------------------------------------
    Xtr, Xte, names, unid, const = cl_matrices(tr, te, cls, pk, m)
    # The parameter search never sees the test fold. On F1 that is the
    # pre-registered "2024 only" split; on the within-season fold the training
    # slice is already short, so it is halved at its own median date rather than
    # at a date that would leave one side empty.
    split = args.inner_split if fold == "F1" else str(
        pd.to_datetime(tr["game_date"]).median().date())
    i_tr, i_va = inner_split(tr, split)
    if len(i_va) < 500 or len(i_tr) < 500:
        i_tr, i_va = tr, tr
    Xi_tr, Xi_va, inames, _, _ = cl_matrices(i_tr, i_va, cls, pk, m)
    l2_rows = []
    best_l2, best_ll = None, np.inf
    for l2 in L2_GRID:
        arm = U.CondLogitArm(l2=l2).fit(Xi_tr, i_tr["y"].to_numpy())
        # the validation design must use the columns the INNER fit kept
        Xv, allv = U.cl_design(i_va, cls, pk, m)
        Xv = Xv[:, :, [allv.index(n) for n in inames]]
        ll = PM.log_loss(i_va["y"].to_numpy(), arm.predict_proba(Xv))
        l2_rows.append({"l2": l2, "inner_val_log_loss": round(ll, 6)})
        if ll < best_ll:
            best_l2, best_ll = l2, ll
    cl = U.CondLogitArm(l2=best_l2).fit(Xtr, tr["y"].to_numpy())
    probs["cond_logit"] = cl.predict_proba(Xte)
    extra["cond_logit"] = {
        "l2": best_l2, "l2_grid": l2_rows, "features": names,
        "dropped_unidentified": unid, "dropped_constant": const,
        "coefficients": {n: round(float(b), 5)
                         for n, b in zip(names, cl.beta_, strict=False)},
        "converged": cl.converged_}
    log(f"    cond_logit: l2={best_l2} ll {PM.log_loss(y_te, probs['cond_logit']):.6f}")

    # ---- U5 ---------------------------------------------------------------
    Ltr, ctr, Lte, kept, dropped5, itr, ite = lgbm_matrices(tr, te, cls, pk, m)
    Li_tr, ci_tr, Li_va, _, _, ii_tr, ii_va = lgbm_matrices(i_tr, i_va, cls, pk, m)
    grid_rows, best_params, best_ll = [], None, np.inf
    grid = U.LGBM_PARAM_GRID if lgbm_params is None else (lgbm_params,)
    for params in grid:
        arm = U.LgbmChoiceArm(params, seed=args.seed).fit(Li_tr, ci_tr, init=ii_tr)
        ll = PM.log_loss(i_va["y"].to_numpy(), arm.predict_proba(Li_va, init=ii_va))
        grid_rows.append({**params, "inner_val_log_loss": round(ll, 6)})
        log(f"      lgbm {params}: inner val ll {ll:.6f}")
        if ll < best_ll:
            best_params, best_ll = params, ll
    seed_lls = []
    for sd in (LGBM_SEEDS if fold == "F1" else LGBM_SEEDS[:1]):
        arm5 = U.LgbmChoiceArm(best_params, seed=sd).fit(Ltr, ctr, init=itr)
        p5 = arm5.predict_proba(Lte, init=ite)
        seed_lls.append(PM.log_loss(y_te, p5))
        if sd == LGBM_SEEDS[0]:
            probs["lgbm"] = p5
            imp = dict(zip(kept, arm5.clf_.feature_importances_.tolist(), strict=False))
    extra["lgbm"] = {"params": best_params, "grid": grid_rows,
                     "dropped_unidentified": dropped5,
                     "seed_log_losses": [round(x, 6) for x in seed_lls],
                     "seed_sd": round(float(np.std(seed_lls, ddof=1)), 6)
                     if len(seed_lls) > 1 else None,
                     "feature_importance": imp}
    log(f"    lgbm: {best_params} ll {PM.log_loss(y_te, probs['lgbm']):.6f} "
        f"seed sd {extra['lgbm']['seed_sd']}")

    # ---- score every arm --------------------------------------------------
    transfer_credit = te[[f"is_transfer_{k}" for k in range(1, 6)]].to_numpy()[
        np.arange(len(te)), y_te].astype(bool)
    has_prev = te[[f"has_prior_season_{k}" for k in range(1, 6)]].to_numpy()[
        np.arange(len(te)), y_te] > 0
    rows = {}
    for arm in U.ARMS:
        p = probs[arm]
        sc = U.score_arm(te, p, driver)
        sc["bootstrap_se"] = round(U.bootstrap_se(te, p, n_rep=args.boot_reps), 6)
        ps = ps_te.single_role() if arm == "dirichlet" else ps_te
        if arm in U.DIRICHLET_ARMS:
            sc["game_level"] = U.game_level_check(
                te, ps=ps, alphas=alpha_fits[arm]["alphas"], n_draw=args.sim_draws,
                seed=args.seed, player_role=prole_te)
        else:
            sc["game_level"] = U.game_level_check(
                te, p=p, n_draw=args.sim_draws, seed=args.seed, player_role=prole_te)
        sc["transfer"] = {
            "transfer_n": int(transfer_credit.sum()),
            "transfer_log_loss": round(PM.log_loss(
                y_te[transfer_credit], p[transfer_credit]), 6)
            if transfer_credit.any() else None,
            "continuing_log_loss": round(PM.log_loss(
                y_te[has_prev & ~transfer_credit], p[has_prev & ~transfer_credit]), 6)
            if (has_prev & ~transfer_credit).any() else None,
            "no_prior_season_log_loss": round(PM.log_loss(
                y_te[~has_prev], p[~has_prev]), 6) if (~has_prev).any() else None,
        }
        sc.update(extra.get(arm, {}))
        rows[arm] = sc
        log(f"    -> {arm:<15} ll {sc['log_loss']:.6f} calib {sc['calib_worst_gap_pp']:.3f}pp "
            f"resp {sc['resp_steps']}/4 sd {sc['game_level']['sd_ratio']:.3f} "
            f"gt0 {sc['game_level']['players_gt0_delta']:+.3f}")

    out = {"fold": fold, "event_class": cls, "n_train": int(len(tr)),
           "n_test": int(len(te)), "shrinkage": shrink, "inner_split": split,
           "prior_kind": pk, "shrink_m": m, "arms": rows,
           "seconds": round(time.time() - t0, 1)}
    out["decision"] = decide(out)
    log(f"    DECISION: {out['decision']['winner'] or 'NO WINNER'} -- "
        f"{out['decision']['reason']}")
    return out


# ---------------------------------------------------------------------------
# The pre-registered decision rule
# ---------------------------------------------------------------------------
def decide(res: dict) -> dict:
    """Per class, the winner is the lowest log loss among the arms that pass
    calibration, responsiveness AND the game-level dispersion checks; the tree
    arm must additionally beat the best non-tree eligible arm by more than the
    floor; ties go to the simpler arm."""
    elig, why = [], {}
    for arm, r in res["arms"].items():
        gl = r["game_level"]
        checks = {"calibration": r["calib_pass"], "responsiveness": r["resp_pass"],
                  "sd_ratio": gl["sd_pass"], "players_gt0": gl["gt0_pass"],
                  "top1_top3": gl["top_pass"]}
        why[arm] = checks
        if all(checks.values()):
            elig.append(arm)
    if not elig:
        return {"winner": None, "eligible": [], "checks": why,
                "reason": "no arm passes every pre-registered gate"}
    floor = max(res["arms"][a]["bootstrap_se"] for a in elig)
    ranked = sorted(elig, key=lambda a: (res["arms"][a]["log_loss"], U.ARM_ORDER[a]))
    best = ranked[0]
    non_tree = [a for a in ranked if a != U.TREE_ARM]
    if best == U.TREE_ARM and non_tree:
        alt = non_tree[0]
        gap = res["arms"][alt]["log_loss"] - res["arms"][best]["log_loss"]
        if gap <= floor:
            # The tree is not clear of the floor, so it is out -- and the arms
            # that remain then go through the ORDINARY tie-break, which takes the
            # SIMPLEST arm inside the floor of the best non-tree one. Handing the
            # win straight to `non_tree[0]` would skip that and crown whichever
            # arm happened to score lowest by a hair.
            nt_tied = [a for a in non_tree
                       if res["arms"][a]["log_loss"] - res["arms"][alt]["log_loss"] <= floor]
            nt_winner = min(nt_tied, key=lambda a: U.ARM_ORDER[a])
            return {"winner": nt_winner, "eligible": elig, "checks": why,
                    "floor": floor, "tied_within_floor": nt_tied,
                    "reason": (f"lgbm leads by {gap:.6f}, inside the {floor:.6f} floor, so "
                               f"the pre-registration's requirement that the tree beat the "
                               f"best non-tree arm by more than the floor is not met; among "
                               f"the non-tree arms {', '.join(nt_tied)} are inside the floor "
                               f"of each other and the tie-break takes the simplest")}
    # ties to the simpler arm
    tied = [a for a in ranked
            if res["arms"][a]["log_loss"] - res["arms"][best]["log_loss"] <= floor]
    winner = min(tied, key=lambda a: U.ARM_ORDER[a])
    gap_txt = ", ".join(f"{a} {res['arms'][a]['log_loss']:.6f}" for a in ranked)
    if len(tied) == 1:
        tail = (f"clear of the next eligible arm by "
                f"{res['arms'][ranked[1]]['log_loss'] - res['arms'][best]['log_loss']:.6f} "
                f"({(res['arms'][ranked[1]]['log_loss'] - res['arms'][best]['log_loss']) / floor:.1f} floors)"
                if len(ranked) > 1 else "the only eligible arm")
    else:
        tail = (f"{', '.join(tied)} are inside the floor of each other; "
                f"the pre-registered tie-break takes the simplest")
    return {"winner": winner, "eligible": elig, "checks": why, "floor": floor,
            "tied_within_floor": tied,
            "reason": f"eligible: {gap_txt}; floor {floor:.6f}; {tail}"}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def split_of(r: dict) -> str:
    return r.get("inner_split", "")


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_report(meta: dict, results: list[dict], path: Path, args) -> None:
    L: list[str] = []
    A = L.append
    A(f"## 2. Run configuration ({time.strftime('%Y-%m-%d')})\n")
    A("| item | value |")
    A("|---|---|")
    A("| trainer | `scripts/train_usage_v1.py` |")
    A(f"| possessions version | `{meta['possessions_version']}` "
      f"(rim override {meta['rim_override_max_ft']} ft) |")
    A("| universe | D-I, non-truncated, `pbp_complete` |")
    A(f"| credited events (modelled) | {meta['n_events']:,} |")
    A(f"| player-games with as-of inputs | {meta['n_player_games']:,} |")
    A(f"| roster position known | {meta['position_known_pct']}% |")
    A(f"| hoopR minutes joined through the player crosswalk | {meta['minutes_join_pct']}% |")
    A("| player-games with a prior season of on-floor history | "
      + ", ".join(f"{k}: {v}%" for k, v in meta["prior_season_pct"].items()) + " |")
    A(f"| Monte-Carlo draws (marginal / Polya / game-level / alpha fit) | "
      f"{args.mc_draws} / {args.seq_draws} / {args.sim_draws} / {args.alpha_draws} |")
    A("")
    A("### 2.1 Event coverage (reported, not silently filtered)\n")
    A("| season | class | events | five resolved | credited id present | modelled |")
    A("|---|---|---:|---:|---:|---:|")
    for s, cov in meta["coverage"].items():
        for c, r in cov.items():
            A(f"| {s} | {c} | {r['n']:,} | {r['five_resolved_pct']}% | "
              f"{r['player_id_present_pct']}% | {r['modelled_pct']}% |")
    A("")

    A("## 3. F1 results (train 2024, test 2025) -- the selection fold\n")
    for res in [r for r in results if r["fold"] == "F1"]:
        A(f"### 3.{1 + list(U.EVENT_CLASSES).index(res['event_class'])} "
          f"{res['event_class']}\n")
        A(f"Train {res['n_train']:,} events, test {res['n_test']:,}. "
          f"Fitted shrinkage: prior `{res['prior_kind']}`, m = {res['shrink_m']:g} "
          f"pseudo on-floor events. Uniform-over-five log loss = 1.609438.\n")
        A("| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp | "
          "boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | "
          "top-3 (sim / real) | eligible |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
        for arm in U.ARMS:
            r = res["arms"][arm]
            gl = r["game_level"]
            ok = res["decision"]["checks"][arm]
            A(f"| {arm} | {r['log_loss']:.6f} | {r['brier']:.4f} | {r['top1']:.4f} | "
              f"{r['top3']:.4f} | {r['calib_worst_gap_pp']:.3f} | {r['resp_steps']}/4 | "
              f"{r['bootstrap_se']:.6f} | {fmt(gl['sd_ratio'], 4)} | "
              f"{fmt(gl['players_gt0_sim'], 3)} / {fmt(gl['players_gt0_actual'], 3)} | "
              f"{fmt(gl['top1_sim_pct'], 2)}% / {fmt(gl['top1_actual_pct'], 2)}% | "
              f"{fmt(gl['top3_sim_pct'], 2)}% / {fmt(gl['top3_actual_pct'], 2)}% | "
              f"{'yes' if all(ok.values()) else 'NO (' + ', '.join(k for k, v in ok.items() if not v) + ')'} |")
        A("")
        A(f"**Decision: {res['decision']['winner'] or 'NO WINNER'}.** "
          f"{res['decision']['reason']}\n")
        d = res["arms"]["dirichlet"]
        h = res["arms"]["hier_dirichlet"]
        A(f"Fitted concentrations -- U2 `{d['alphas']}`; U3 `{h['alphas']}` "
          f"(`null` = no dispersion at that level, the top rung of the grid). "
          f"Polya-urn (within-game, NOT a pregame quantity, never a decision input) "
          f"log loss: U2 {d['polya_urn_log_loss']}, U3 {h['polya_urn_log_loss']}.\n")
        cl = res["arms"]["cond_logit"]
        A(f"U4 ridge penalty {cl['l2']}; dropped as unidentified in this fold: "
          f"`{cl['dropped_unidentified']}`; dropped as constant: `{cl['dropped_constant']}`. "
          f"Coefficients: `{cl['coefficients']}`.\n")
        lg = res["arms"]["lgbm"]
        A(f"U5 parameters {lg['params']} (searched on 2024 only); seed-varied refits "
          f"{lg['seed_log_losses']}, seed SD {lg['seed_sd']}.\n")
        A("Transfer subset (2025 credited players whose modal team changed):\n")
        A("| arm | transfers | continuing | no prior season |")
        A("|---|---:|---:|---:|")
        for arm in U.ARMS:
            t = res["arms"][arm]["transfer"]
            A(f"| {arm} | {fmt(t['transfer_log_loss'], 6)} | "
              f"{fmt(t['continuing_log_loss'], 6)} | "
              f"{fmt(t['no_prior_season_log_loss'], 6)} |")
        A(f"\n(n transfers = {res['arms']['proportional']['transfer']['transfer_n']:,})\n")
        by_role = res["arms"]["proportional"]["game_level"].get("sd_ratio_by_role", {})
        if by_role:
            A("Per-role \"too narrow\" statistic for the proportional arm:\n")
            A("| role | players | sim SD | real SD | ratio |")
            A("|---|---:|---:|---:|---:|")
            for role, v in by_role.items():
                A(f"| {role} | {v['n_players']} | {v['sd_sim']} | {v['sd_actual']} | "
                  f"{v['sd_ratio']} |")
            A("")

    wf = [r for r in results if r["fold"] == "WF2025"]
    if wf:
        A("## 4. Robustness fold: within-2025 walk-forward "
          f"(train before {U.WF_SPLIT_DATE}, test after)\n")
        A("| class | arm | log loss | boot SE | calib (pp) | resp | SD ratio | "
          "players >=1 delta | eligible |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for res in wf:
            for arm in U.ARMS:
                r = res["arms"][arm]
                gl = r["game_level"]
                ok = res["decision"]["checks"][arm]
                A(f"| {res['event_class']} | {arm} | {r['log_loss']:.6f} | "
                  f"{r['bootstrap_se']:.6f} | {r['calib_worst_gap_pp']:.3f} | "
                  f"{r['resp_steps']}/4 | {fmt(gl['sd_ratio'], 4)} | "
                  f"{fmt(gl['players_gt0_delta'], 3)} | "
                  f"{'yes' if all(ok.values()) else 'no'} |")
        A("")
        A("| class | winner | reason |")
        A("|---|---|---|")
        for res in wf:
            A(f"| {res['event_class']} | {res['decision']['winner'] or 'NO WINNER'} | "
              f"{res['decision']['reason']} |")
        A("")
        A("On this fold the prior-season block IS identified on both sides "
          "(2024 is a completed season of on-floor history), so the shrinkage grid "
          "can choose it and the logit/tree can use it. The fitted priors are in "
          "the per-class rows above.\n")
        A("| class | fitted prior | m | prior-season rung available |")
        A("|---|---|---:|---|")
        for res in wf:
            A(f"| {res['event_class']} | {res['prior_kind']} | {res['shrink_m']:g} | "
              f"{res['shrinkage']['prior_season_available']} |")
        A("")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", help="possessions build to read the rim override from")
    ap.add_argument("--out", default="data/processed/models/usage")
    ap.add_argument("--report", default=None)
    ap.add_argument("--classes", nargs="*", default=list(U.EVENT_CLASSES))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--mc-draws", type=int, default=200)
    ap.add_argument("--seq-draws", type=int, default=100)
    ap.add_argument("--sim-draws", type=int, default=40)
    ap.add_argument("--alpha-draws", type=int, default=15)
    ap.add_argument("--alpha-max-tg", type=int, default=3000)
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--inner-split", default=INNER_SPLIT_DATE)
    ap.add_argument("--skip-wf", action="store_true")
    ap.add_argument("--quick", action="store_true", help="small draw counts, for a smoke run")
    ap.add_argument("--redecide", action="store_true",
                    help="re-run the DECISION RULE over an existing results_v1.json and "
                         "rewrite the report and the params file; refits nothing")
    args = ap.parse_args()
    if args.quick:
        args.mc_draws, args.seq_draws, args.sim_draws = 40, 30, 10
        args.alpha_draws, args.alpha_max_tg, args.boot_reps = 6, 800, 50

    out_dir = ROOT / args.out
    report = Path(args.report) if args.report else out_dir / "report_v1.md"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    if args.redecide:
        results = json.loads((out_dir / "results_v1.json").read_text())
        meta = json.loads((out_dir / f"build_report_{args.version}.json").read_text())
        for r in results:
            r["decision"] = decide(r)
            log(f"  [{r['fold']}] {r['event_class']}: "
                f"{r['decision']['winner'] or 'NO WINNER'} -- {r['decision']['reason']}")
        write_results(out_dir, report, meta, results, args)
        log(f"re-decided {len(results)} folds and rewrote {report}")
        return 0

    events, asof, meta = build_tables(args.version, out_dir, args.rebuild)
    log(f"events {len(events):,} asof {len(asof):,} ({time.time() - t0:.1f}s)")

    results = []
    for cls in args.classes:
        design = U.build_usage_design(events, asof, cls)
        tr, te = U.fold_slices(design)
        f1 = run_fold(cls, tr, te, "F1", args, log)
        results.append(f1)
        if not args.skip_wf:
            wtr, wte = U.walkforward_slices(design)
            # the tree's parameters were searched on 2024 only, as
            # pre-registered; the robustness fold REUSES them rather than
            # searching again on data that includes its own test window.
            results.append(run_fold(cls, wtr, wte, "WF2025", args, log,
                                    lgbm_params=f1["arms"]["lgbm"]["params"]))
        del design

    write_results(out_dir, report, meta, results, args)
    (out_dir / "train_log_v1.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nwrote {report} and {out_dir}/usage_params_v1.json "
        f"({time.time() - t0:.1f}s total)")
    return 0


def write_results(out_dir: Path, report: Path, meta: dict, results: list[dict],
                  args) -> None:
    """Persist the results JSON, the sim-facing parameters, and the report.

    Split out of `main` so that `--redecide` can re-run the DECISION RULE over a
    finished results file and rewrite all three without refitting anything."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results_v1.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    params = {
        "possessions_version": args.version,
        "seed": args.seed,
        "per_class": {r["event_class"]: {
            "prior_kind": r["prior_kind"], "shrink_m": r["shrink_m"],
            "winner": r["decision"]["winner"],
            "dirichlet_alphas": r["arms"]["dirichlet"]["alphas"],
            "hier_alphas": r["arms"]["hier_dirichlet"]["alphas"],
            "cond_logit_l2": r["arms"]["cond_logit"]["l2"],
            "inner_split": split_of(r),
            "cond_logit_features": r["arms"]["cond_logit"]["features"],
            "cond_logit_coefficients": r["arms"]["cond_logit"]["coefficients"],
            "lgbm_params": r["arms"]["lgbm"]["params"],
        } for r in results if r["fold"] == "F1"},
    }
    (out_dir / "usage_params_v1.json").write_text(json.dumps(params, indent=2),
                                                  encoding="utf-8")
    write_report(meta, results, report, args)


if __name__ == "__main__":
    raise SystemExit(main())
