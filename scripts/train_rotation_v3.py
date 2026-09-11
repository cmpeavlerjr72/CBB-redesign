#!/usr/bin/env python
"""
train_rotation_v3.py -- ROUND 3 of the pre-registered L4 rotation bake-off
(`docs/models/rotation/experiments.md` section 6), F1: train 2024, test 2025.

Rounds 1 and 2 adopted nothing. Round 2's binding failure is the opposite of
round 1's: R5 and R6 are inside tolerance in a blowout and 4-7 pp short of the
starters' share in the final eight minutes when the game is close or moderate,
because R5's override is one-sided -- it can only push a player onto the bench,
never put one back on the floor. Round 3 adds the missing half as a FITTED model
component, in two functional forms, and re-runs R5 with the corrected hazard
matrix (`experiments.md` section 5, status OPEN).

Arms:
  R2_hier_dirichlet   incumbent reference, unchanged.
  R5_hybrid           round 2's hybrid with the CORRECTED hazard matrix. This
                      column closes section 5's OPEN item.
  R7_keep_logistic    R5 + a logistic keep-probability fitted on the training
                      season over (fouls, |margin|, time remaining, is_starter,
                      the team prior), applied as a state DEVIATION with one
                      per-player-per-game tolerance draw.
  R8_keep_cell        R5 + the simpler threshold form: one fitted scalar against
                      the training season's own starters'-share-by-(time x
                      margin) table.

Everything else -- data, folds, metrics, tolerances, the eligibility veto and the
decision rule -- is unchanged from rounds 1 and 2 and is imported from
`train_rotation_v1`, so all three rounds are graded by literally the same code.

Usage:
    .venv/Scripts/python.exe scripts/train_rotation_v3.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.data.player_ids import load_crosswalk  # noqa: E402
from cbb_sim.models import rotation as R  # noqa: E402
from cbb_sim.models import rotation_v3 as V3  # noqa: E402

import train_rotation_v1 as V1  # noqa: E402
import train_rotation_v2 as V2  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "models" / "rotation"
EXPERIMENTS = ROOT / "docs" / "models" / "rotation" / "experiments.md"
BASE_FIT = OUT_DIR / "rotation_fit_round2_corrected_hazards.json"
TRAIN_SEASON, TEST_SEASON = 2024, 2025
TEST_SUBSET_SEED = 2025

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_rotation_v3")

STATE_CELLS_FIT = V2.STATE_CELLS_FIT
SIMPLICITY = {"R2_hier_dirichlet": 1, "R5_hybrid": 2, "R8_keep_cell": 3,
              "R7_keep_logistic": 4}

KEEP_SCALE_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
KEEP_Q0_GRID = (0.02, 0.05, 0.15, 0.35)
THETA_GRID = tuple(round(0.1 * i, 2) for i in range(11))
BLOCK_SCALE_GRID = (0.5, 1.0, 2.0, 4.0)
BLOCK_P0_GRID = (0.005, 0.02, 0.06)


# ---------------------------------------------------------------------------
# Knob fitting (coordinate descent over the block pair and the keep pair)
# ---------------------------------------------------------------------------
def _score_arm(arm, priors, scripts, keys, rotation_sets, state_targets, seed):
    recs, chg, tot = [], 0, 0
    for key in keys:
        prior, script = priors[key], scripts[key]
        lu, fh = arm.simulate(prior, script, R.game_stream(seed, key[0]))
        starters = set(int(x) for x in prior.pids[prior.starters()[:5]])
        recs.append(R.team_game_stats(lu, script.dur, script.time_bucket,
                                      script.margin_bucket, starters, fh,
                                      prior.pids, rotation_sets.get(key, set())))
        ks = [tuple(sorted(int(x) for x in r)) for r in lu]
        for i in range(1, len(ks)):
            tot += 1
            chg += ks[i] != ks[i - 1]
    got = R.aggregate_stats(recs)
    err = 0.0
    for c in STATE_CELLS_FIT:
        a, b = got.get(c, np.nan), state_targets.get(c, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            err += (a - b) ** 2
    return ({**{c: float(got.get(c, np.nan)) for c in STATE_CELLS_FIT},
             "change_rate": chg / max(tot, 1), "err": float(err)}, err)


def fit_knobs(make_arm, keep_grid, fit, donors, priors, scripts, keys, rotation_sets,
              state_targets, seed=7, passes=2, start=None):
    """Coordinate descent: fit the keep knob(s) with the block held at its
    round-2 corrected value, then the block pair at that keep, then the keep
    again. Every grid point evaluated is returned and reported."""
    block = (float(fit.notes.get("r5_override_scale", 1.0)),
             float(fit.notes.get("r5_block_base", 0.02)))
    keep = start
    grids: dict[str, dict] = {}
    for p in range(passes):
        # --- keep pass ------------------------------------------------------
        g, best, best_err = {}, keep, np.inf
        for kk in keep_grid:
            arm = make_arm(fit, donors, block, kk)
            sc, err = _score_arm(arm, priors, scripts, keys, rotation_sets,
                                 state_targets, seed)
            g[f"keep={kk}"] = sc
            if err < best_err:
                best, best_err = kk, err
        keep = best
        grids[f"pass{p + 1}_keep"] = g
        log.info("  pass %d keep -> %s (err %.6f)", p + 1, keep, best_err)
        # --- block pass -----------------------------------------------------
        g, best, best_err = {}, block, np.inf
        for bs in BLOCK_SCALE_GRID:
            for bp in BLOCK_P0_GRID:
                arm = make_arm(fit, donors, (bs, bp), keep)
                sc, err = _score_arm(arm, priors, scripts, keys, rotation_sets,
                                     state_targets, seed)
                g[f"block_scale={bs},p0={bp}"] = sc
                if err < best_err:
                    best, best_err = (bs, bp), err
        block = best
        grids[f"pass{p + 1}_block"] = g
        log.info("  pass %d block -> %s (err %.6f)", p + 1, block, best_err)
    return block, keep, grids


def make_r7(fit, donors, block, keep):
    return V3.R7KeepLogistic(fit, donors, block_scale=block[0], block_p0=block[1],
                             keep_scale=keep[0], keep_q0=keep[1])


def make_r8(fit, donors, block, keep):
    return V3.R8KeepCell(fit, donors, block_scale=block[0], block_p0=block[1],
                         theta=keep)


# ---------------------------------------------------------------------------
def fit_round3(train, fit, args, seed=7, fit_seed=11):
    """Fit everything round 3 adds. `fit_seed` selects the training-game sample
    and the logistic's random_state; `seed` is the sim seed inside the knob grid.
    The noise floor re-runs this function with both changed."""
    tp, feats = train["tp"], train["feats"]
    notes = dict(fit.notes)

    # --- R8's data table ---------------------------------------------------
    t0 = time.time()
    tab = V3.fit_state_starter_table(tp)
    notes["r8_state_starter_table"] = tab.tolist()
    log.info("R8 state starter table (%.1fs): late cells %s",
             time.time() - t0, np.round(tab[2], 4).tolist())

    # --- R7's on-floor propensity -----------------------------------------
    tr_games = sorted(tp["game_id"].unique())
    rs = np.random.RandomState(fit_seed)
    kg = rs.choice(tr_games, size=min(args.keep_games, len(tr_games)), replace=False)
    t0 = time.time()
    X, y = V3.build_keep_training(tp, feats, fit, list(kg), fouls=train["fouls"])
    km = V3.fit_keep_model(X, y, seed=fit_seed)
    notes["r7_keep"] = km
    log.info("R7 keep model: %d rows, base %.4f, %.1fs", km["n"], km["base_rate"],
             time.time() - t0)
    for f, c in zip(V3.KEEP_FEATURES, km["coef"]):
        log.info("    %-32s %+0.4f", f, c)

    fit.notes = notes

    # --- the knob grids ----------------------------------------------------
    sched_ids = list(kg)[: args.knob_games]
    tp_s = tp[tp["game_id"].isin(sched_ids)]
    feats_s = feats[feats["game_id"].isin(sched_ids)]
    scripts_s = R.build_scripts(tp_s)
    priors_s = R.build_priors(feats_s, fit)
    keys = [k for k in list(scripts_s)[: args.knob_keys] if k in priors_s]
    rot_s = {(int(a), int(b)): set(int(x) for x in g.loc[g["mpg_asof_raw"] >= 10.0, "pid"])
             for (a, b), g in feats_s.groupby(["game_id", "team_id"], sort=False)}
    asof_s = {k: set(int(x) for x in v.pids[v.starters()[:5]]) for k, v in priors_s.items()}
    st_targets = V2.train_state_targets(train, keys, rot_s, asof_s)
    notes["r3_state_targets_train"] = {c: float(st_targets[c]) for c in STATE_CELLS_FIT}
    log.info("train state targets %s",
             {c[-2:]: round(st_targets[c], 4) for c in STATE_CELLS_FIT})

    donors_tr = R.build_donor_bank(tp, k_donors=notes["r5_donor_k"])

    log.info("fitting R7 knobs on %d train team-games", len(keys))
    kgrid7 = [(s, q) for s in KEEP_SCALE_GRID for q in KEEP_Q0_GRID]
    b7, k7, g7 = fit_knobs(make_r7, kgrid7, fit, donors_tr, priors_s, scripts_s, keys,
                           rot_s, st_targets, seed=seed,
                           start=(KEEP_SCALE_GRID[0], KEEP_Q0_GRID[0]))
    notes["r7_block"] = list(b7)
    notes["r7_keep_scale"], notes["r7_keep_base"] = float(k7[0]), float(k7[1])
    notes["r7_knob_grid"] = g7

    log.info("fitting R8 knobs on %d train team-games", len(keys))
    b8, k8, g8 = fit_knobs(make_r8, list(THETA_GRID), fit, donors_tr, priors_s, scripts_s,
                           keys, rot_s, st_targets, seed=seed, start=0.0)
    notes["r8_block"] = list(b8)
    notes["r8_keep_theta"] = float(k8)
    notes["r8_knob_grid"] = g8

    fit.notes = notes
    return fit


# ---------------------------------------------------------------------------
def build_arms(fit, donors):
    return {
        "R2_hier_dirichlet": R.R2HierDirichlet(fit),
        "R5_hybrid": V3.R5HybridV3(fit, donors),
        "R8_keep_cell": V3.R8KeepCell(
            fit, donors, block_scale=fit.notes["r8_block"][0],
            block_p0=fit.notes["r8_block"][1], theta=fit.notes["r8_keep_theta"]),
        "R7_keep_logistic": V3.R7KeepLogistic(
            fit, donors, block_scale=fit.notes["r7_block"][0],
            block_p0=fit.notes["r7_block"][1], keep_scale=fit.notes["r7_keep_scale"],
            keep_q0=fit.notes["r7_keep_base"]),
    }


def slope_table(arm_lus, actual_by_key, priors, keys):
    """Team-quintile responsiveness: the prior starter-minutes share against the
    close-and-late starters' share, actual and simulated."""
    ps = {k: float(priors[k].share[priors[k].starters()[:5]].sum()) for k in keys}
    ks = sorted(ps)
    vals = np.array([ps[k] for k in ks])
    qs = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
    qidx = np.searchsorted(qs, vals, side="right")
    rows = []
    for q in range(5):
        sel = [k for k, qq in zip(ks, qidx) if qq == q]
        row = {"quintile": q + 1, "n_team_games": len(sel),
               "prior_starter_share": float(np.mean([ps[k] for k in sel]))}
        num = sum(actual_by_key[k][0] for k in sel if k in actual_by_key)
        den = sum(actual_by_key[k][1] for k in sel if k in actual_by_key)
        row["ACTUAL"] = float(num / den) if den else float("nan")
        row["n_poss"] = float(den / 5.0)
        for name, per_key in arm_lus.items():
            num = sum(per_key[k][0] for k in sel if k in per_key)
            den = sum(per_key[k][1] for k in sel if k in per_key)
            row[name] = float(num / den) if den else float("nan")
        rows.append(row)
    return rows


def close_late_cell(lu, script_or_g, starters, is_actual):
    """(starter slots, total slots) in the final 8:00 at |margin| <= 5 -- the
    cell the round-3 override exists to repair."""
    ids, inv = np.unique(lu, return_inverse=True)
    st = np.array([1.0 if int(i) in starters else 0.0 for i in ids])
    ss = st[inv.reshape(lu.shape)].sum(axis=1)
    tb = script_or_g[0]
    mb = script_or_g[1]
    m = np.isin(tb, R.LATE_TIME_BUCKETS) & (mb == 0)
    return float(ss[m].sum()), float(5 * m.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-games", type=int, default=1600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--noise-seeds", type=int, default=20)
    ap.add_argument("--noise-games", type=int, default=150)
    ap.add_argument("--keep-games", type=int, default=300)
    ap.add_argument("--knob-games", type=int, default=250)
    ap.add_argument("--knob-keys", type=int, default=120)
    ap.add_argument("--min-prior-games", type=int, default=3)
    ap.add_argument("--noise-refit", action="store_true", default=True)
    ap.add_argument("--tag", type=str, default="F1-round3")
    args = ap.parse_args()

    t0 = time.time()
    assert_not_sealed([TRAIN_SEASON, TEST_SEASON], context="rotation bake-off F1 round 3")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = V1.load_season(TRAIN_SEASON)
    test = V1.load_season(TEST_SEASON)

    cw = load_crosswalk()
    av = R.attach_hoopr_availability(test["feats"], TEST_SEASON, cw, pg=test["pg"])
    ok = av["hoopr_join_ok"]
    cw_cov = {"asof_rows": int(len(av)),
              "crosswalk_hit": float(av["espn_id"].notna().mean()),
              "hoopr_box_join": float(ok.mean()),
              "dnp_agreement": float((av.loc[ok, "did_not_play"].astype(bool)
                                      == (av.loc[ok, "onfloor_minutes"] <= 0)).mean())}

    base = R.RotationFit.from_json(BASE_FIT)
    fit = fit_round3(train, base, args, seed=7, fit_seed=11)
    fit.to_json(OUT_DIR / "rotation_fit_v3.json")

    # ---- test universe: identical to the graded round-2 run ---------------
    priors_te = R.build_priors(test["feats"], fit, min_prior_games=args.min_prior_games)
    scripts_te = R.build_scripts(test["tp"])
    both: dict[int, list] = {}
    for (gid, tid) in scripts_te:
        if (gid, tid) in priors_te:
            both.setdefault(gid, []).append((gid, tid))
    eligible = sorted([g for g, v in both.items() if len(v) == 2])
    if len(eligible) > args.test_games:
        rs = np.random.RandomState(TEST_SUBSET_SEED)
        sel = sorted(rs.choice(eligible, size=args.test_games, replace=False))
        subset_note = (f"random subset of {args.test_games} of {len(eligible)} eligible "
                       f"games, numpy RandomState seed {TEST_SUBSET_SEED} -- the same "
                       f"game set the graded round-2 run used")
    else:
        sel, subset_note = eligible, f"all {len(eligible)} eligible games"
    log.info("test universe: %s", subset_note)

    keys_by_game = {g: ([k for k in both[g] if scripts_te[k].is_home]
                        + [k for k in both[g] if not scripts_te[k].is_home]) for g in sel}
    keys = [k for g in sel for k in keys_by_game[g]]

    rotation_sets = {}
    fs = test["feats"]
    kset = set(keys)
    fs = fs[[(int(g), int(t)) in kset for g, t in zip(fs["game_id"], fs["team_id"])]]
    for key, g in fs.groupby(["game_id", "team_id"], sort=False):
        rotation_sets[(int(key[0]), int(key[1]))] = set(
            int(p) for p in g.loc[g["mpg_asof_raw"] >= 10.0, "pid"])
    asof_starters = {k: set(int(x) for x in v.pids[v.starters()[:5]])
                     for k, v in priors_te.items()}

    log.info("computing actual metrics on %d team-games", len(keys))
    a_recs, a_rot, (a_diag, a_overlap) = V1.actual_records(test, keys, rotation_sets,
                                                           asof_starters)
    a_row = V1.build_row("ACTUAL", a_recs, a_rot, {}, np.array([]))
    act_minutes = V1.pooled_and_within_sd(a_rot)[3]
    a_row["ks_minutes_d"], a_row["ks_minutes_p"] = np.nan, np.nan
    diag_row = R.aggregate_stats(a_diag) if a_diag else {}
    diag_row["starter_overlap_of5"] = float(np.mean(a_overlap)) if a_overlap else float("nan")

    # actual close-late cell per team-game, for the slope check
    tps = test["tp"]
    tps = tps[[(int(g), int(t)) in kset for g, t in zip(tps["game_id"], tps["team_id"])]]
    actual_cl = {}
    for key, g in tps.groupby(["game_id", "team_id"], sort=False):
        key = (int(key[0]), int(key[1]))
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        actual_cl[key] = close_late_cell(lu, (g["time_bucket"].to_numpy(),
                                              g["margin_bucket"].to_numpy()),
                                         set(int(x) for x in lu[0]), True)

    donors_full = R.build_donor_bank(test["tp"], k_donors=fit.notes["r5_donor_k"])
    arms = build_arms(fit, donors_full)
    seeds = list(range(args.seeds))
    rows, verdicts, lineup_top1, per_key_cl = {}, {}, {}, {}
    for name, arm in arms.items():
        ts = time.time()
        recs, rot, lus = V1.sim_records(arm, priors_te, scripts_te, keys_by_game,
                                        rotation_sets, seeds)
        row = V1.build_row(name, recs, rot, a_row, act_minutes)
        row["change_rate"] = R.sim_change_rate(lus)
        rows[name] = row
        verdicts[name] = V1.verdict(row, a_row)
        lineup_top1[name] = np.array([r["lu_top1"] for r in recs])
        # slope cell, accumulated over seeds
        acc: dict = {}
        i = 0
        for _seed in seeds:
            for gid, sides in keys_by_game.items():
                for key in sides:
                    lu = lus[i]
                    i += 1
                    sc = scripts_te[key]
                    st = set(int(x) for x in priors_te[key].pids[priors_te[key].starters()[:5]])
                    c = close_late_cell(lu, (sc.time_bucket, sc.margin_bucket), st, False)
                    prev = acc.get(key, (0.0, 0.0))
                    acc[key] = (prev[0] + c[0], prev[1] + c[1])
        per_key_cl[name] = acc
        log.info("%-20s %.1f min  nonzero %.2f top5 %.4f top8 %.4f lu1 %.4f min %.2f "
                 "late b0/b1/b2 %.4f/%.4f/%.4f ft %.4f f4 %.4f",
                 name, (time.time() - ts) / 60, row["n_nonzero_mean"], row["top5_share"],
                 row["top8_share"], row["lu_top1"], row["minutes_mean"],
                 row["late_starter_share_b0"], row["late_starter_share_b1"],
                 row["late_starter_share_b2"], row["foul_trouble_share"],
                 row["foul4_share"])

    a_top1 = np.array([r["lu_top1"] for r in a_recs])
    for name in rows:
        d, p = R.ks_2samp(lineup_top1[name], a_top1)
        rows[name]["ks_lineup_top1_d"] = d
        rows[name]["ks_lineup_top1_p"] = p

    slope = slope_table(per_key_cl, actual_cl, priors_te, [k for k in keys if k in actual_cl])

    # ---- noise floor A: seed-varied sim runs ------------------------------
    log.info("noise floor A: %d seeds x %d games", args.noise_seeds, args.noise_games)
    nsel = sel[: args.noise_games]
    nkeys = {g: keys_by_game[g] for g in nsel}
    noise = {}
    for name, arm in arms.items():
        per_seed = []
        for sd in range(args.noise_seeds):
            recs, rot, _ = V1.sim_records(arm, priors_te, scripts_te, nkeys,
                                          rotation_sets, [1000 + sd])
            per_seed.append(V1.build_row(name, recs, rot, a_row, act_minutes))
        noise[name] = {k: float(np.std([r[k] for r in per_seed], ddof=1))
                       for k in ["minutes_mean", "minutes_sd_pooled", "minutes_sd_within",
                                 "top5_share", "top8_share", "n_nonzero_mean", "lu_top1",
                                 "late_starter_share_b0", "late_starter_share_b1",
                                 "late_starter_share_b2", "foul_trouble_share"]}
        noise[name]["n_seeds"] = args.noise_seeds
        noise[name]["n_games"] = len(nsel)

    # ---- noise floor B: spec-identical refit under a second seed ----------
    refit_rows = {}
    if args.noise_refit:
        log.info("noise floor B: spec-identical refit under a second seed")
        base2 = R.RotationFit.from_json(BASE_FIT)
        fit2 = fit_round3(train, base2, args, seed=23, fit_seed=101)
        fit2.to_json(OUT_DIR / "rotation_fit_v3_noisefloor_seed2.json")
        arms2 = build_arms(fit2, donors_full)
        for name in ("R7_keep_logistic", "R8_keep_cell"):
            recs, rot, _ = V1.sim_records(arms2[name], priors_te, scripts_te,
                                          nkeys, rotation_sets, seeds)
            refit_rows[name] = V1.build_row(name, recs, rot, a_row, act_minutes)
        # the first-seed arms on the SAME 150-game universe, so the comparison is
        # refit-vs-refit and not universe-vs-universe
        for name in ("R7_keep_logistic", "R8_keep_cell"):
            recs, rot, _ = V1.sim_records(arms[name], priors_te, scripts_te,
                                          nkeys, rotation_sets, seeds)
            refit_rows[name + "_seed1"] = V1.build_row(name, recs, rot, a_row, act_minutes)
        refit_rows["_fit2_knobs"] = {
            "r7_block": fit2.notes["r7_block"], "r7_keep_scale": fit2.notes["r7_keep_scale"],
            "r7_keep_base": fit2.notes["r7_keep_base"], "r8_block": fit2.notes["r8_block"],
            "r8_keep_theta": fit2.notes["r8_keep_theta"]}
        a_recs_n, a_rot_n, _ = V1.actual_records(
            test, [k for g in nsel for k in keys_by_game[g]], rotation_sets, None)
        refit_rows["_actual_noise_universe"] = V1.build_row("ACTUAL", a_recs_n, a_rot_n,
                                                            {}, np.array([]))

    decision = V2.decide2(rows, verdicts) if len(rows) == 3 else decide3(rows, verdicts)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fold": args.tag, "round": 3,
        "train_season": TRAIN_SEASON, "test_season": TEST_SEASON,
        "config": vars(args), "test_universe": subset_note,
        "test_subset_seed": TEST_SUBSET_SEED, "crosswalk_coverage": cw_cov,
        "n_eligible_games": len(eligible), "n_simulated_games": len(sel),
        "actual": a_row, "actual_asof_starters": diag_row,
        "arms": rows, "verdicts": verdicts, "noise": noise,
        "noise_refit": refit_rows, "slope": slope, "decision": decision,
        "fit": {k: v for k, v in fit.__dict__.items() if k not in ("tilt", "notes")},
        "fit_notes": {k: v for k, v in fit.notes.items()
                      if k not in ("scheduler_grid", "r5_override_scale_grid",
                                   "r5_donor_k_grid")},
    }
    stem = ("rotation_F1_round3" if args.tag.startswith("F1-round3")
            else f"rotation_round3_{args.tag}")
    (OUT_DIR / f"{stem}_results.json").write_text(
        json.dumps(payload, indent=2, default=V1._json_default))
    pd.DataFrame([a_row] + [rows[n] for n in rows]).to_csv(
        OUT_DIR / f"{stem}_table.csv", index=False)
    (OUT_DIR / f"{stem}_section.md").write_text(render_round3(payload), encoding="utf-8")
    print(render_round3(payload))
    log.info("done in %.1f min", (time.time() - t0) / 60)


def decide3(rows: dict, verdicts: dict) -> dict:
    scored = []
    for name, v in verdicts.items():
        g8 = sum(1 for c in V1.G8_CELLS if v.get(c))
        state = sum(1 for c in V1.STATE_CELLS if v.get(c))
        scored.append({
            "arm": name, "g8_pass": g8, "g8_cells": len(V1.G8_CELLS),
            "state_pass": state, "state_cells": len(V1.STATE_CELLS),
            "total_pass": g8 + state,
            "eligible": bool(all(v.get(c) for c in V1.STATE_CELLS)),
            "ks_lineup_top1_d": rows[name]["ks_lineup_top1_d"],
            "simplicity_rank": SIMPLICITY[name]})
    scored.sort(key=lambda r: (-r["total_pass"], r["ks_lineup_top1_d"], r["simplicity_rank"]))
    winners = [r for r in scored if r["eligible"]]
    return {"ranking": scored, "winner": winners[0]["arm"] if winners else None,
            "diagnosis": None if winners else
            "no arm has every state-dependence cell inside +/- 3 pp"}


def render_round3(p: dict) -> str:
    a, rows, v, noise = p["actual"], p["arms"], p["verdicts"], p["noise"]
    dg = p.get("actual_asof_starters") or {}
    names = list(rows)
    L: list[str] = []
    W = L.append
    W(f"\n## 7. Round-3 results (train {p['train_season']}, test {p['test_season']}) -- "
      f"run {p['generated_at'][:19]}Z\n")
    W(f"Test universe: {p['test_universe']}, {p['config']['seeds']} seeds per arm; "
      f"noise floor A {p['config']['noise_seeds']} seeds x {p['config']['noise_games']} "
      f"games, noise floor B a spec-identical refit under a second seed. Rotation "
      f"player-games per arm {rows[names[0]]['n_rot_player_games']:,} (actual "
      f"{a['n_rot_player_games']:,}); every cell is far above the n < 300 "
      f"UNDERPOWERED threshold unless flagged.\n")

    W("### 7.1 G8 cells\n")
    W("| cell | tol | ACTUAL | " + " | ".join(names) + " |")
    W("|---|---|---:|" + "---:|" * len(names))

    def line(label, key, tol, vkey, fmt="{:.4f}"):
        W(f"| {label} | {tol} | " + fmt.format(a[key]) + " | " +
          " | ".join(fmt.format(rows[n][key]) + "  " + ("PASS" if v[n].get(vkey) else "FAIL")
                     for n in names) + " |")

    line("minutes mean (rotation players)", "minutes_mean", "+/- 2.0", "minutes_mean", "{:.2f}")
    W("| minutes SD ratio, pooled | 0.9-1.1 | 1.000 | " + " | ".join(
        f"{v[n]['_sd_ratio_pooled_val']:.3f}  " +
        ("PASS" if v[n]['sd_ratio_pooled'] else "FAIL") for n in names) + " |")
    W("| minutes SD ratio, within-player | 0.9-1.1 | 1.000 | " + " | ".join(
        f"{v[n]['_sd_ratio_within_val']:.3f}  " +
        ("PASS" if v[n]['sd_ratio_within'] else "FAIL") for n in names) + " |")
    line("top-5 share of team minutes", "top5_share", "+/- 2 pp", "top5_share")
    line("top-8 share of team minutes", "top8_share", "+/- 2 pp", "top8_share")
    line("players with > 0 minutes", "n_nonzero_mean", "+/- 1.0", "n_nonzero", "{:.2f}")

    W("\n### 7.2 State-dependence cells (an arm missing ANY of these is ineligible)\n")
    W("| cell | ACTUAL | " + " | ".join(names) + " |")
    W("|---|---:|" + "---:|" * len(names))
    lab = {0: r"\|margin\| <= 5", 1: r"\|margin\| 6-15", 2: r"\|margin\| > 15"}
    for b in range(R.N_MARGIN_BUCKETS):
        k = f"late_starter_share_b{b}"
        W(f"| final 8:00 starters' share, {lab[b]} (n={a[f'late_slots_b{b}']:,.0f} poss) | "
          f"{a[k]:.4f} | " + " | ".join(
              f"{rows[n][k]:.4f} ({(rows[n][k]-a[k])*100:+.1f} pp)  " +
              ("PASS" if v[n][k] else "FAIL") for n in names) + " |")
    k = "foul_trouble_share"
    W(f"| starters on floor while carrying >= 4 fouls (n={a['foul_trouble_opps']:,.0f}) | "
      f"{a[k]:.4f} | " + " | ".join(
          f"{rows[n][k]:.4f} ({(rows[n][k]-a[k])*100:+.1f} pp)  " +
          ("PASS" if v[n][k] else "FAIL") for n in names) + " |")
    W(f"| (diagnostic) at exactly 4 fouls (n={a['foul4_opps']:,.0f}) | {a['foul4_share']:.4f} | "
      + " | ".join(f"{rows[n]['foul4_share']:.4f}" for n in names) + " |")
    if dg:
        W(f"\nAs-of starter set overlaps the real starting five on "
          f"**{dg['starter_overlap_of5']:.2f} of 5**; the ACTUAL sequence re-graded "
          f"with the MODEL's starter set separates \"wrong five\" from \"wrong "
          f"rotation\":\n")
        W("| cell | ACTUAL (own starters) | ACTUAL (as-of starter set) | " +
          " | ".join(names) + " |")
        W("|---|---:|---:|" + "---:|" * len(names))
        for b in range(R.N_MARGIN_BUCKETS):
            k = f"late_starter_share_b{b}"
            W(f"| final 8:00, {lab[b]} | {a[k]:.4f} | {dg[k]:.4f} | " +
              " | ".join(f"{rows[n][k]:.4f}" for n in names) + " |")
        W(f"| >= 4 fouls | {a['foul_trouble_share']:.4f} | {dg['foul_trouble_share']:.4f} | "
          + " | ".join(f"{rows[n]['foul_trouble_share']:.4f}" for n in names) + " |")

    W("\n### 7.3 Lineup concentration\n")
    W("| metric | ACTUAL | " + " | ".join(names) + " |")
    W("|---|---:|" + "---:|" * len(names))
    for label, key, fmt in (("top-1 lineup share of possessions", "lu_top1", "{:.4f}"),
                            ("top-3 lineup share", "lu_top3", "{:.4f}"),
                            ("top-5 lineup share", "lu_top5", "{:.4f}"),
                            ("distinct lineups per team-game", "n_lineups_mean", "{:.2f}"),
                            ("K-S of per-player minutes (D)", "ks_minutes_d", "{:.4f}")):
        av = a.get(key, float("nan"))
        W(f"| {label} | " + (fmt.format(av) if np.isfinite(av) else "--") + " | " +
          " | ".join(fmt.format(rows[n][key]) for n in names) + " |")
    W("| K-S of the top-1 lineup share distribution (D) | -- | " +
      " | ".join(f"{rows[n]['ks_lineup_top1_d']:.4f}" for n in names) + " |")
    W("| substitution rate at a possession boundary | " +
      f"{p['fit_notes']['observed_change_rate_train']:.4f} (train) | " +
      " | ".join(f"{rows[n]['change_rate']:.4f}" for n in names) + " |")

    W(f"\n### 7.4 Noise floor A ({p['config']['noise_seeds']} seeds x "
      f"{p['config']['noise_games']} games)\n")
    W("| metric | " + " | ".join(names) + " |")
    W("|---|" + "---:|" * len(names))
    for k in ["minutes_mean", "top5_share", "n_nonzero_mean", "lu_top1",
              "late_starter_share_b0", "late_starter_share_b1",
              "late_starter_share_b2", "foul_trouble_share"]:
        W(f"| {k} | " + " | ".join(f"{noise[n][k]:.5f}" for n in names) + " |")

    nr = p.get("noise_refit") or {}
    if nr:
        W("\n### 7.5 Noise floor B -- spec-identical refit under a second seed\n")
        W(f"Both fits use the same specification; the second draws a different "
          f"training-game sample (fit seed 101 vs 11), a different logistic "
          f"`random_state`, and a different sim seed inside the knob grid (23 vs 7). "
          f"Graded on the {p['config']['noise_games']}-game noise universe so the "
          f"two fits are compared on identical games.\n")
        W(f"Refit knobs: {json.dumps(nr.get('_fit2_knobs', {}))}\n")
        an = nr.get("_actual_noise_universe", {})
        W("| cell | ACTUAL | R7 seed 1 | R7 seed 2 | |delta| pp | R8 seed 1 | R8 seed 2 | |delta| pp |")
        W("|---|---:|---:|---:|---:|---:|---:|---:|")
        for c in V1.STATE_CELLS + ["top5_share", "top8_share"]:
            r7a = nr["R7_keep_logistic_seed1"][c]
            r7b = nr["R7_keep_logistic"][c]
            r8a = nr["R8_keep_cell_seed1"][c]
            r8b = nr["R8_keep_cell"][c]
            W(f"| {c} | {an.get(c, float('nan')):.4f} | {r7a:.4f} | {r7b:.4f} | "
              f"{abs(r7b - r7a) * 100:.1f} | {r8a:.4f} | {r8b:.4f} | "
              f"{abs(r8b - r8a) * 100:.1f} |")

    W("\n### 7.6 Slope check -- team quintile of the as-of starter-minutes share\n")
    W("Cell = starters' share of on-floor slots in the final 8:00 at |margin| <= 5.\n")
    W("| quintile | team-games | prior starter share | ACTUAL | " + " | ".join(names)
      + " | close-late possessions |")
    W("|---|---:|---:|---:|" + "---:|" * len(names) + "---:|")
    for r in p["slope"]:
        flag = "" if r["n_poss"] >= 300 else " UNDERPOWERED"
        W(f"| Q{r['quintile']} | {r['n_team_games']:,} | {r['prior_starter_share']:.4f} | "
          f"{r['ACTUAL']:.4f} | " + " | ".join(f"{r[n]:.4f}" for n in names) +
          f" | {r['n_poss']:,.0f}{flag} |")
    x = np.array([r["prior_starter_share"] for r in p["slope"]])
    W("")
    for labn in ["ACTUAL"] + names:
        y = np.array([r[labn] for r in p["slope"]])
        W(f"- `{labn}` slope vs the prior: **{np.polyfit(x, y, 1)[0]:+.3f}**, "
          f"Q5 - Q1 = {(y[-1] - y[0]) * 100:+.1f} pp")

    W("\n### 7.7 Fitted round-3 components\n")
    fn = p["fit_notes"]
    W("R7's on-floor propensity (logit scale; positive = more likely on the floor):\n")
    W("| feature | coefficient |")
    W("|---|---:|")
    for f, c in zip(fn["r7_keep"]["features"], fn["r7_keep"]["coef"]):
        W(f"| `{f}` | {c:+.4f} |")
    W(f"| _intercept_ | {fn['r7_keep']['intercept']:+.4f} |")
    W(f"\nFitted on {fn['r7_keep']['n']:,} (team-game, possession, candidate) rows, "
      f"base rate {fn['r7_keep']['base_rate']:.4f}.\n")
    W(f"\nR7 knobs: block (scale {fn['r7_block'][0]}, p0 {fn['r7_block'][1]}), "
      f"keep (scale {fn['r7_keep_scale']}, q0 {fn['r7_keep_base']}). "
      f"R8 knobs: block (scale {fn['r8_block'][0]}, p0 {fn['r8_block'][1]}), "
      f"theta {fn['r8_keep_theta']}.\n")
    tab = np.asarray(fn["r8_state_starter_table"], dtype="float64")
    W("R8's fitted `s*` table (training-season starters' share of on-floor slots):\n")
    W("| time bucket | \\|m\\| <= 5 | \\|m\\| 6-15 | \\|m\\| > 15 |")
    W("|---|---:|---:|---:|")
    tlab = ["1st half", "2nd half > 8:00", "2nd half 8:00-2:00", "final 2:00", "OT"]
    for i in range(tab.shape[0]):
        W(f"| {tlab[i]} | " + " | ".join(
            (f"{tab[i, j]:.4f}" if np.isfinite(tab[i, j]) else "--") for j in range(3)) + " |")
    for tag, gk in (("R7", "r7_knob_grid"), ("R8", "r8_knob_grid")):
        W(f"\n{tag} knob grid (state-cell squared error; every point evaluated):\n")
        W("| pass | point | late b0 | late b1 | late b2 | >= 4 fouls | sub rate | sq. err |")
        W("|---|---|---:|---:|---:|---:|---:|---:|")
        for ph, g in fn[gk].items():
            for pt, sc in g.items():
                W(f"| {ph} | {pt} | {sc['late_starter_share_b0']:.4f} | "
                  f"{sc['late_starter_share_b1']:.4f} | {sc['late_starter_share_b2']:.4f} | "
                  f"{sc['foul_trouble_share']:.4f} | {sc['change_rate']:.4f} | "
                  f"{sc['err']:.6f} |")

    W("\n### 7.8 Decision\n")
    W("| arm | G8 cells | state cells | total | eligible (all 4 state cells) | "
      "lineup K-S D | simplicity |")
    W("|---|---:|---:|---:|---|---:|---:|")
    for r in p["decision"]["ranking"]:
        W(f"| {r['arm']} | {r['g8_pass']}/{r['g8_cells']} | {r['state_pass']}/"
          f"{r['state_cells']} | {r['total_pass']} | {'yes' if r['eligible'] else 'NO'} | "
          f"{r['ks_lineup_top1_d']:.4f} | {r['simplicity_rank']} |")
    if p["decision"]["winner"]:
        W(f"\n**Winner: `{p['decision']['winner']}`** by the pre-registered rule.\n")
    else:
        W(f"\n**No arm adopted.** {p['decision']['diagnosis']}. Cell-by-cell misses:\n")
        W("| arm | cell | sim | actual | miss |")
        W("|---|---|---:|---:|---:|")
        for n in names:
            for c in V1.STATE_CELLS:
                if not v[n].get(c):
                    W(f"| {n} | {c} | {rows[n][c]:.4f} | {a[c]:.4f} | "
                      f"{(rows[n][c]-a[c])*100:+.1f} pp |")
        W("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
