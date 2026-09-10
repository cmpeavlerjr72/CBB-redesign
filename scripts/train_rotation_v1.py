#!/usr/bin/env python
"""
train_rotation_v1.py -- the pre-registered L4 ROTATION bake-off (F1: train 2024,
test 2025), per `docs/models/rotation/experiments.md` section 1.

Runs all four arms (R1 dirichlet, R2 hier_dirichlet, R3 stint_hazard,
R4 stint_resample) through one grading path, writes artifacts to
`data/processed/models/rotation/`, and appends the results section to
`docs/models/rotation/experiments.md` (append-only, per CLAUDE.md).

Usage:
    .venv/Scripts/python.exe scripts/train_rotation_v1.py
    .venv/Scripts/python.exe scripts/train_rotation_v1.py --test-games 400 --seeds 3 --noise-seeds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.data.player_ids import load_crosswalk  # noqa: E402
from cbb_sim.models import rotation as R  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "models" / "rotation"
EXPERIMENTS = ROOT / "docs" / "models" / "rotation" / "experiments.md"

TRAIN_SEASON = 2024
TEST_SEASON = 2025

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_rotation_v1")

# Pre-registered tolerances
TOL = {
    "minutes_mean": 2.0,
    "sd_ratio": (0.9, 1.1),
    "share_pp": 2.0,
    "state_pp": 3.0,
}
UNDERPOWERED_N = 300


# ---------------------------------------------------------------------------
# Season assembly
# ---------------------------------------------------------------------------
def load_season(season: int) -> dict:
    log.info("loading season %d", season)
    tp = R.load_team_possessions(season)
    pg = R.player_game_minutes(tp)
    gu = pd.read_parquet(ROOT / "data" / "processed" / "games_universe.parquet")
    gu = gu[gu["season"] == season][["game_id", "cbbd_game_id"]]
    ev = R.player_game_fouls(season)
    ev = ev.merge(gu, on="cbbd_game_id", how="inner")
    ev = ev.merge(pg[["game_id", "team_id", "pid"]].drop_duplicates(),
                  on=["game_id", "pid"], how="inner")
    feats = R.build_asof_player_features(pg, fouls=ev)
    log.info("  %d games, %d team-poss, %d player-games, %d as-of rows",
             tp["game_id"].nunique(), len(tp), len(pg), len(feats))
    return {"season": season, "tp": tp, "pg": pg, "fouls": ev, "feats": feats}


def foul_event_study(tp: pd.DataFrame, feats: pd.DataFrame, fouls: pd.DataFrame,
                     game_ids, window: int = 20) -> pd.DataFrame:
    """Within-player on-floor rate in the `window` possessions before vs after
    each personal foul -- the input to the fitted foul tilt. Rows are
    (fouls, time_bucket, before_on, before_n, after_on, after_n)."""
    gset = set(int(g) for g in game_ids)
    tps = tp[tp["game_id"].isin(gset)]
    ev = fouls[fouls["game_id"].isin(gset)]
    evg = {k: g for k, g in ev.groupby(["game_id", "team_id"], sort=False)}
    acc: dict[tuple[int, int], list[float]] = {}
    for key, g in tps.groupby(["game_id", "team_id"], sort=False):
        e = evg.get(key)
        if e is None or not len(e):
            continue
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        tb = g["time_bucket"].to_numpy()
        period = g["period"].to_numpy()
        clock = g["start_clock"].to_numpy()
        key_poss = period.astype("int64") * 10000 - clock.astype("int64")
        n = lu.shape[0]
        on_by_pid: dict[int, np.ndarray] = {}
        for pid in np.unique(lu):
            on_by_pid[int(pid)] = (lu == pid).any(axis=1)
        counts: dict[int, int] = {}
        order = np.argsort(e["period"].to_numpy() * 10000 - e["start_clock"].to_numpy())
        for i in order:
            pid = int(e["pid"].to_numpy()[i])
            on = on_by_pid.get(pid)
            if on is None:
                continue
            counts[pid] = counts.get(pid, 0) + 1
            f = counts[pid]
            if f >= R.FOUL_OUT:
                continue
            k = int(e["period"].to_numpy()[i]) * 10000 - int(e["start_clock"].to_numpy()[i])
            idx = int(np.searchsorted(key_poss, k))
            lo0, lo1 = max(0, idx - window), max(0, idx)
            hi0, hi1 = min(n, idx + 1), min(n, idx + 1 + window)
            if lo1 - lo0 < 5 or hi1 - hi0 < 5:
                continue
            b = acc.setdefault((f, int(tb[min(idx, n - 1)])), [0.0, 0.0, 0.0, 0.0])
            b[0] += float(on[lo0:lo1].sum())
            b[1] += float(lo1 - lo0)
            b[2] += float(on[hi0:hi1].sum())
            b[3] += float(hi1 - hi0)
    rows = [{"fouls": f, "time_bucket": tbb, "before_on": v[0], "before_n": v[1],
             "after_on": v[2], "after_n": v[3]} for (f, tbb), v in acc.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_all(train: dict, args) -> tuple[R.RotationFit, dict]:
    tp, pg, feats, fouls = train["tp"], train["pg"], train["feats"], train["fouls"]
    notes: dict = {}

    role_prior = R.fit_role_prior(pg)
    k0, k0_scores = R.fit_k0(feats, pg, role_prior)
    w_dnp = R.fit_w_dnp(feats, pg, role_prior, k0)
    tail_ratio, _n_profile_p90 = R.fit_tail(pg)
    min_share = R.fit_min_share(pg)
    notes["k0_grid_mae"] = k0_scores
    notes["role_prior"] = [round(v, 3) for v in role_prior]
    log.info("role prior top-8 %s", [round(v, 1) for v in role_prior[:8]])

    start_alpha, start_scores = R.fit_start_predictor(
        feats, pg, R.RotationFit(season_train=TRAIN_SEASON, role_prior=role_prior, k0=k0,
                                 w_dnp=w_dnp))
    notes["start_predictor_overlap"] = start_scores
    log.info("starter predictor overlap (of 5): %s -> alpha %.2f",
             {k: round(v, 3) for k, v in start_scores.items()}, start_alpha)
    p_play, w_avail_dnp = R.fit_availability(feats, pg)
    sched_targets = R.observed_scheduler_targets(tp)
    n_profile = R.fit_n_profile(p_play, sched_targets["n_nonzero"])  # grid seed
    notes["scheduler_targets_train"] = sched_targets
    fpm_league, fpm_prior_min, fpm_scores = R.fit_fpm(feats, pg, fouls)
    notes["p_play_by_rank"] = [round(v, 4) for v in p_play]
    notes["fpm_grid_mae"] = fpm_scores
    log.info("k0=%.1f  w_dnp=%.3f  tail_ratio=%.3f  n_profile=%d (p90 heuristic %d)",
             k0, w_dnp, tail_ratio, n_profile, _n_profile_p90)
    log.info("p_play by rank %s", [round(v, 3) for v in p_play[:14]])
    log.info("w_avail_dnp=%.3f  fpm_league=%.4f  fpm_prior_min=%.0f",
             w_avail_dnp, fpm_league, fpm_prior_min)

    fit = R.RotationFit(season_train=TRAIN_SEASON, role_prior=role_prior, k0=k0,
                        w_dnp=w_dnp, tail_ratio=tail_ratio, n_profile=int(n_profile),
                        p_play=p_play, w_avail_dnp=w_avail_dnp,
                        fpm_league=fpm_league, fpm_prior_min=fpm_prior_min,
                        min_share=min_share, start_alpha=start_alpha)

    # --- Dirichlet concentrations, method of moments on the training season ---
    priors_tr = R.build_priors(feats, fit)
    tot = pg.groupby(["game_id", "team_id"])["minutes"].sum()
    pred, real, fam_pred, fam_real, sp, sr, bp, br = [], [], [], [], [], [], [], []
    obs = {k: dict(zip(g["pid"].astype("int64"), g["minutes"]))
           for k, g in pg.groupby(["game_id", "team_id"], sort=False)}
    for key, pr in priors_tr.items():
        o = obs.get(key)
        t = tot.get(key)
        if not o or not t or t <= 0:
            continue
        rs = np.array([o.get(int(p), 0.0) for p in pr.pids]) / t
        # tail slots absorb whatever the named candidates did not cover
        miss = max(0.0, 1.0 - rs.sum())
        tail = pr.pids < 0
        if tail.any() and miss > 0:
            rs[tail] += miss * (pr.share[tail] / max(pr.share[tail].sum(), 1e-9))
        pred.append(pr.share)
        real.append(rs)
        st = np.zeros(pr.n, dtype=bool)
        st[pr.starters()] = True
        fam_pred.append(np.array([pr.share[st].sum(), pr.share[~st].sum()]))
        fam_real.append(np.array([rs[st].sum(), rs[~st].sum()]))
        if pr.share[st].sum() > 0 and rs[st].sum() > 0:
            sp.append(pr.share[st] / pr.share[st].sum()); sr.append(rs[st] / rs[st].sum())
        if pr.share[~st].sum() > 0 and rs[~st].sum() > 0:
            bp.append(pr.share[~st] / pr.share[~st].sum()); br.append(rs[~st] / rs[~st].sum())
    fit.alpha = R.fit_dirichlet_alpha(np.concatenate(pred), np.concatenate(real))
    fit.alpha_family = R.fit_dirichlet_alpha(np.concatenate(fam_pred), np.concatenate(fam_real))
    fit.alpha_starters = R.fit_dirichlet_alpha(np.concatenate(sp), np.concatenate(sr))
    fit.alpha_bench = R.fit_dirichlet_alpha(np.concatenate(bp), np.concatenate(br))
    log.info("alpha=%.1f  alpha_family=%.1f  alpha_starters=%.1f  alpha_bench=%.1f",
             fit.alpha, fit.alpha_family, fit.alpha_starters, fit.alpha_bench)

    # --- fitted state / foul tilt tables --------------------------------------
    tr_games = sorted(tp["game_id"].unique())
    rs = np.random.RandomState(11)
    tilt_games = rs.choice(tr_games, size=min(args.tilt_games, len(tr_games)), replace=False)
    fp = foul_event_study(tp, feats, fouls, tilt_games)
    ranked = R.add_asof_ranks(feats, fit)
    fit.tilt = R.fit_tilt_tables(tp, ranked, fp)
    notes["foul_event_study_rows"] = int(len(fp))
    notes["foul_tilt"] = fit.tilt.foul.tolist()

    # --- foul-rate calibration -------------------------------------------------
    fouls_t = fouls.merge(tp[["game_id", "team_id"]].drop_duplicates(),
                          on=["game_id", "team_id"], how="inner")
    fit.foul_rate_scale = R.fit_foul_rate_scale(tp, feats, fouls_t, fit)
    log.info("foul_rate_scale=%.3f", fit.foul_rate_scale)

    # --- scheduler stickiness -------------------------------------------------
    sched_ids = list(tilt_games)[: args.theta_games]
    scripts_tr = R.build_scripts(tp[tp["game_id"].isin(sched_ids)])
    feats_tr = feats[feats["game_id"].isin(sched_ids)]
    keys = list(scripts_tr)[: args.theta_keys]
    bestp, sched_scores = R.fit_scheduler_params(fit, feats_tr, scripts_tr, keys,
                                                 sched_targets)
    fit.ema_horizon = bestp["ema_horizon"]
    fit.swap_threshold = bestp["swap_threshold"]
    fit.lam_deficit = bestp["lam_deficit"]
    fit.n_profile = bestp["n_profile"]
    notes["observed_change_rate_train"] = float(sched_targets["change_rate"])
    notes["scheduler_grid_best"] = bestp
    notes["scheduler_grid"] = {k: v for k, v in
                               sorted(sched_scores.items(), key=lambda kv: kv[1]["err"])[:20]}
    key = (f"n={bestp['n_profile']},H={bestp['ema_horizon']:.0f},"
           f"theta={bestp['swap_threshold']},lam={bestp['lam_deficit']}")
    b = sched_scores[key]
    log.info("scheduler fit %s -> change %.4f/%.4f  lineups %.2f/%.2f  top5 %.4f/%.4f  "
             "nonzero %.2f/%.2f", key, b["change_rate"], sched_targets["change_rate"],
             b["n_lineups"], sched_targets["n_lineups"], b["top5_share"],
             sched_targets["top5_share"], b["n_nonzero"], sched_targets["n_nonzero"])

    # --- R3 hazards ------------------------------------------------------------
    hz_games = rs.choice(tr_games, size=min(args.hazard_games, len(tr_games)), replace=False)
    Xe, ye, Xn, yn = R.build_hazard_training(tp, feats, fit, list(hz_games), fouls=fouls)
    fit.hazard_exit, fit.hazard_enter = R.fit_hazard_models(Xe, ye, Xn, yn)
    log.info("hazard exit rows %d (base %.4f); enter rows %d (base %.4f)",
             fit.hazard_exit["n"], fit.hazard_exit["base_rate"],
             fit.hazard_enter["n"], fit.hazard_enter["base_rate"])
    fit.notes = notes
    return fit, notes


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def actual_records(test: dict, keys, rotation_sets, asof_starters: dict | None = None
                   ) -> tuple[list, list, list]:
    tp, feats, fouls = test["tp"], test["feats"], test["fouls"]
    kset = set(keys)
    tps = tp[[(int(g), int(t)) in kset for g, t in zip(tp["game_id"], tp["team_id"])]]
    evg = {k: g for k, g in fouls.groupby(["game_id", "team_id"], sort=False)}
    recs, rot_rows, diag = [], [], []
    overlaps = []
    for key, g in tps.groupby(["game_id", "team_id"], sort=False):
        key = (int(key[0]), int(key[1]))
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        dur = g["duration_s"].to_numpy(dtype="float64")
        tb = g["time_bucket"].to_numpy()
        mb = g["margin_bucket"].to_numpy()
        starters = set(int(x) for x in lu[0])
        pids = np.unique(lu)
        pid_index = {int(p): i for i, p in enumerate(pids)}
        fm = R.actual_foul_matrix(evg.get(key), g["period"].to_numpy(),
                                  g["start_clock"].to_numpy(), pid_index)
        rot = rotation_sets.get(key, set())
        st = R.team_game_stats(lu, dur, tb, mb, starters, fm, pids, rot)
        recs.append(st)
        for p, m in st["rot_minutes"].items():
            rot_rows.append((key[0], key[1], p, m))
        if asof_starters is not None and key in asof_starters:
            pred = asof_starters[key]
            overlaps.append(len(pred & starters))
            diag.append(R.team_game_stats(lu, dur, tb, mb, pred, fm, pids, rot))
    return recs, rot_rows, (diag, overlaps)


def sim_records(arm, priors, scripts, keys_by_game, rotation_sets, seeds) -> tuple[list, list, list]:
    recs, rot_rows, lus = [], [], []
    for seed in seeds:
        for gid, sides in keys_by_game.items():
            rng = R.game_stream(seed, gid)
            for key in sides:                      # home first, then away
                pr, sc = priors[key], scripts[key]
                lu, fh = arm.simulate(pr, sc, rng)
                starters = set(int(x) for x in pr.pids[pr.starters()[:5]])
                rot = rotation_sets.get(key, set())
                st = R.team_game_stats(lu, sc.dur, sc.time_bucket, sc.margin_bucket,
                                       starters, fh, pr.pids, rot)
                recs.append(st)
                lus.append(lu)
                for p, m in st["rot_minutes"].items():
                    rot_rows.append((key[0], key[1], p, m, seed))
    return recs, rot_rows, lus


def pooled_and_within_sd(rot_rows, min_games: int = 5) -> tuple[float, float, float, np.ndarray]:
    """Mean minutes, pooled SD, and the mean *within-player* game-to-game SD --
    the CFB "too narrow" diagnostic."""
    if not rot_rows:
        return np.nan, np.nan, np.nan, np.array([])
    has_seed = len(rot_rows[0]) == 5
    df = pd.DataFrame(rot_rows, columns=["game_id", "team_id", "pid", "minutes"] +
                      (["seed"] if has_seed else []))
    m = df["minutes"].to_numpy(dtype="float64")
    # within-player SD is computed per seed and then averaged: pooling seeds
    # would add Monte-Carlo noise the actual side has no counterpart for.
    keys = (["seed"] if has_seed else []) + ["team_id", "pid"]
    g = df.groupby(keys)["minutes"]
    sd, n = g.std(ddof=1), g.size()
    within = float(sd[n >= min_games].mean())
    return float(m.mean()), float(m.std(ddof=1)), within, m


def build_row(name: str, recs: list, rot_rows: list, actual_agg: dict,
              actual_minutes: np.ndarray) -> dict:
    agg = R.aggregate_stats(recs)
    mean_m, pooled_sd, within_sd, minutes = pooled_and_within_sd(rot_rows)
    ks_d, ks_p = R.ks_2samp(minutes, actual_minutes)
    agg.update({
        "arm": name,
        "minutes_mean": mean_m,
        "minutes_sd_pooled": pooled_sd,
        "minutes_sd_within": within_sd,
        "ks_minutes_d": ks_d,
        "ks_minutes_p": ks_p,
        "n_rot_player_games": len(rot_rows),
    })
    return agg


def verdict(row: dict, act: dict) -> dict:
    """PASS/FAIL against the pre-registered tolerances."""
    v = {}
    v["minutes_mean"] = abs(row["minutes_mean"] - act["minutes_mean"]) <= TOL["minutes_mean"]
    ratio_p = row["minutes_sd_pooled"] / act["minutes_sd_pooled"]
    ratio_w = row["minutes_sd_within"] / act["minutes_sd_within"]
    v["sd_ratio_pooled"] = TOL["sd_ratio"][0] <= ratio_p <= TOL["sd_ratio"][1]
    v["sd_ratio_within"] = TOL["sd_ratio"][0] <= ratio_w <= TOL["sd_ratio"][1]
    v["top5_share"] = abs(row["top5_share"] - act["top5_share"]) * 100 <= TOL["share_pp"]
    v["top8_share"] = abs(row["top8_share"] - act["top8_share"]) * 100 <= TOL["share_pp"]
    v["n_nonzero"] = abs(row["n_nonzero_mean"] - act["n_nonzero_mean"]) <= 1.0
    for b in range(R.N_MARGIN_BUCKETS):
        k = f"late_starter_share_b{b}"
        v[k] = abs(row[k] - act[k]) * 100 <= TOL["state_pp"]
    v["foul_trouble_share"] = abs(row["foul_trouble_share"] - act["foul_trouble_share"]) * 100 \
        <= TOL["state_pp"]
    v["_sd_ratio_pooled_val"] = ratio_p
    v["_sd_ratio_within_val"] = ratio_w
    return v


G8_CELLS = ["minutes_mean", "sd_ratio_pooled", "sd_ratio_within", "top5_share",
            "top8_share", "n_nonzero"]
STATE_CELLS = ["late_starter_share_b0", "late_starter_share_b1", "late_starter_share_b2",
               "foul_trouble_share"]
BLOWOUT_CELLS = ["late_starter_share_b2", "foul_trouble_share"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-games", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--noise-seeds", type=int, default=50)
    ap.add_argument("--noise-games", type=int, default=250)
    ap.add_argument("--tilt-games", type=int, default=1200)
    ap.add_argument("--hazard-games", type=int, default=400)
    ap.add_argument("--theta-keys", type=int, default=200)
    ap.add_argument("--theta-games", type=int, default=400)
    ap.add_argument("--min-prior-games", type=int, default=3)
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--tag", type=str, default="F1")
    ap.add_argument("--append-experiments", action="store_true",
                    help="append the results section to docs/models/rotation/experiments.md; "
                         "off by default so development runs cannot pollute an append-only doc")
    args = ap.parse_args()

    t0 = time.time()
    assert_not_sealed([TRAIN_SEASON, TEST_SEASON], context="rotation bake-off F1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_season(TRAIN_SEASON)
    test = load_season(TEST_SEASON)

    # crosswalk coverage of the modelling path (reported, per the guardrails)
    cw = load_crosswalk()
    av = R.attach_hoopr_availability(test["feats"], TEST_SEASON, cw, pg=test["pg"])
    ok = av["hoopr_join_ok"]
    cw_cov = {
        "asof_rows": int(len(av)),
        "crosswalk_hit": float(av["espn_id"].notna().mean()),
        "hoopr_box_join": float(ok.mean()),
        "dnp_agreement": float((av.loc[ok, "did_not_play"].astype(bool)
                                == (av.loc[ok, "onfloor_minutes"] <= 0)).mean()),
    }
    log.info("crosswalk coverage on the as-of feature table: %s", cw_cov)

    fit, notes = fit_all(train, args)
    fit.to_json(OUT_DIR / "rotation_fit.json")

    # ---------------- test universe -------------------------------------------
    priors_te = R.build_priors(test["feats"], fit, min_prior_games=args.min_prior_games)
    scripts_te = R.build_scripts(test["tp"])
    both = {}
    for (gid, tid) in scripts_te:
        if (gid, tid) in priors_te:
            both.setdefault(gid, []).append((gid, tid))
    eligible = sorted([g for g, v in both.items() if len(v) == 2])
    n_all_games = test["tp"]["game_id"].nunique()
    log.info("test games with complete on-floor data: %d; both teams with >= %d prior games: %d",
             n_all_games, args.min_prior_games, len(eligible))

    rs = np.random.RandomState(2025)
    sel = sorted(rs.choice(eligible, size=min(args.test_games, len(eligible)), replace=False)) \
        if len(eligible) > args.test_games else eligible
    keys_by_game = {}
    for g in sel:
        home = [k for k in both[g] if scripts_te[k].is_home]
        away = [k for k in both[g] if not scripts_te[k].is_home]
        keys_by_game[g] = home + away
    keys = [k for g in sel for k in keys_by_game[g]]
    log.info("simulating %d games / %d team-games x %d seeds x 4 arms",
             len(sel), len(keys), args.seeds)

    rotation_sets = {}
    fs = test["feats"]
    fs = fs[[(int(g), int(t)) in set(keys) for g, t in zip(fs["game_id"], fs["team_id"])]]
    for key, g in fs.groupby(["game_id", "team_id"], sort=False):
        rotation_sets[(int(key[0]), int(key[1]))] = set(
            int(p) for p in g.loc[g["mpg_asof_raw"] >= 10.0, "pid"])

    # ---------------- actual ---------------------------------------------------
    log.info("computing actual metrics")
    asof_starters = {k: set(int(x) for x in v.pids[v.starters()[:5]])
                     for k, v in priors_te.items()}
    a_recs, a_rot, (a_diag, a_overlap) = actual_records(test, keys, rotation_sets,
                                                        asof_starters)
    a_row = build_row("ACTUAL", a_recs, a_rot, {}, np.array([]))
    act_minutes = pooled_and_within_sd(a_rot)[3]
    a_row["ks_minutes_d"], a_row["ks_minutes_p"] = np.nan, np.nan
    diag_row = R.aggregate_stats(a_diag) if a_diag else {}
    diag_row["starter_overlap_of5"] = float(np.mean(a_overlap)) if a_overlap else float("nan")
    log.info("starter prediction: %.3f of 5 correct; ACTUAL sequence graded with the "
             "as-of starter set -> late b0/b1/b2 %.4f / %.4f / %.4f",
             diag_row["starter_overlap_of5"], diag_row.get("late_starter_share_b0", np.nan),
             diag_row.get("late_starter_share_b1", np.nan),
             diag_row.get("late_starter_share_b2", np.nan))
    log.info("actual: nonzero %.2f top5 %.4f top8 %.4f lu1 %.4f minutes mean %.2f sd %.2f/%.2f",
             a_row["n_nonzero_mean"], a_row["top5_share"], a_row["top8_share"],
             a_row["lu_top1"], a_row["minutes_mean"], a_row["minutes_sd_pooled"],
             a_row["minutes_sd_within"])

    donors = R.build_donor_bank(test["tp"], test["feats"])
    arms = {
        "R1_dirichlet": R.R1Dirichlet(fit),
        "R2_hier_dirichlet": R.R2HierDirichlet(fit),
        "R3_stint_hazard": R.R3StintHazard(fit),
        "R4_stint_resample": R.R4StintResample(fit, donors),
    }

    seeds = list(range(args.seeds))
    rows, verdicts, lineup_top1 = {}, {}, {}
    for name, arm in arms.items():
        ts = time.time()
        recs, rot, lus = sim_records(arm, priors_te, scripts_te, keys_by_game,
                                     rotation_sets, seeds)
        row = build_row(name, recs, rot, a_row, act_minutes)
        row["change_rate"] = R.sim_change_rate(lus)
        rows[name] = row
        verdicts[name] = verdict(row, a_row)
        lineup_top1[name] = np.array([r["lu_top1"] for r in recs])
        log.info("%-20s %.1fs  nonzero %.2f top5 %.4f top8 %.4f lu1 %.4f min %.2f "
                 "sd %.2f/%.2f late_b2 %.4f ft %.4f",
                 name, time.time() - ts, row["n_nonzero_mean"], row["top5_share"],
                 row["top8_share"], row["lu_top1"], row["minutes_mean"],
                 row["minutes_sd_pooled"], row["minutes_sd_within"],
                 row["late_starter_share_b2"], row["foul_trouble_share"])

    a_top1 = np.array([r["lu_top1"] for r in a_recs])
    for name in rows:
        d, p = R.ks_2samp(lineup_top1[name], a_top1)
        rows[name]["ks_lineup_top1_d"] = d
        rows[name]["ks_lineup_top1_p"] = p

    # ---------------- noise floor ---------------------------------------------
    log.info("noise floor: %d seeds x %d games", args.noise_seeds, args.noise_games)
    nsel = sel[: args.noise_games]
    nkeys_by_game = {g: keys_by_game[g] for g in nsel}
    noise = {}
    for name, arm in arms.items():
        per_seed = []
        for s in range(args.noise_seeds):
            recs, rot, _ = sim_records(arm, priors_te, scripts_te, nkeys_by_game,
                                       rotation_sets, [1000 + s])
            r = build_row(name, recs, rot, a_row, act_minutes)
            per_seed.append(r)
        noise[name] = {
            k: float(np.std([r[k] for r in per_seed], ddof=1))
            for k in ["minutes_mean", "minutes_sd_pooled", "minutes_sd_within", "top5_share",
                      "top8_share", "n_nonzero_mean", "lu_top1", "lu_top3", "lu_top5",
                      "late_starter_share_b0", "late_starter_share_b1",
                      "late_starter_share_b2", "foul_trouble_share"]
        }
        noise[name]["n_seeds"] = args.noise_seeds
        noise[name]["n_games"] = len(nsel)
        log.info("%-20s noise SD: minutes_mean %.4f top5 %.5f late_b2 %.5f",
                 name, noise[name]["minutes_mean"], noise[name]["top5_share"],
                 noise[name]["late_starter_share_b2"])

    # ---------------- robustness: within-2025 walk-forward --------------------
    robust = {}
    if not args.skip_robustness:
        log.info("robustness: within-2025 walk-forward (train < 2025-01-15, test after)")
        cut = pd.Timestamp("2025-01-15")
        d = pd.to_datetime(test["tp"]["game_date"])
        tr_ids = set(test["tp"].loc[d < cut, "game_id"].unique())
        te_ids = set(test["tp"].loc[d >= cut, "game_id"].unique())
        tr = {k: (v[v["game_id"].isin(tr_ids)] if isinstance(v, pd.DataFrame) else v)
              for k, v in test.items()}
        fit2, _ = fit_all(tr, argparse.Namespace(
            tilt_games=min(args.tilt_games, len(tr_ids)),
            hazard_games=min(args.hazard_games, len(tr_ids)),
            theta_keys=args.theta_keys, theta_games=args.theta_games))
        te_sel = [g for g in sel if g in te_ids][: max(200, args.test_games // 3)]
        if te_sel:
            p2 = R.build_priors(test["feats"][test["feats"]["game_id"].isin(te_sel)], fit2,
                                min_prior_games=args.min_prior_games)
            kbg = {g: [k for k in keys_by_game[g] if k in p2] for g in te_sel}
            kbg = {g: v for g, v in kbg.items() if len(v) == 2}
            k2 = [k for g in kbg for k in kbg[g]]
            ar, arot, _ = actual_records(test, k2, rotation_sets)
            arow = build_row("ACTUAL", ar, arot, {}, np.array([]))
            amin = pooled_and_within_sd(arot)[3]
            donors2 = R.build_donor_bank(test["tp"], test["feats"])
            arms2 = {"R1_dirichlet": R.R1Dirichlet(fit2),
                     "R2_hier_dirichlet": R.R2HierDirichlet(fit2),
                     "R3_stint_hazard": R.R3StintHazard(fit2),
                     "R4_stint_resample": R.R4StintResample(fit2, donors2)}
            robust["ACTUAL"] = arow
            for name, arm in arms2.items():
                recs, rot, lus = sim_records(arm, p2, scripts_te, kbg, rotation_sets,
                                             list(range(min(3, args.seeds))))
                r = build_row(name, recs, rot, arow, amin)
                robust[name] = r
                robust[name + "_verdict"] = verdict(r, arow)
            robust["_n_games"] = len(kbg)

    # ---------------- decision --------------------------------------------------
    decision = decide(rows, verdicts, a_row)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fold": args.tag, "train_season": TRAIN_SEASON, "test_season": TEST_SEASON,
        "config": vars(args),
        "crosswalk_coverage": cw_cov,
        "n_test_games_complete_onfloor": int(n_all_games),
        "n_eligible_games": len(eligible), "n_simulated_games": len(sel),
        "actual": a_row, "actual_asof_starters": diag_row,
        "arms": rows, "verdicts": verdicts, "noise": noise,
        "robustness": robust, "decision": decision,
        "fit": {k: v for k, v in fit.__dict__.items() if k not in ("tilt", "notes")},
        "fit_notes": notes,
    }
    stem = "rotation_F1" if args.tag == "F1" else f"rotation_{args.tag}"
    (OUT_DIR / f"{stem}_results.json").write_text(
        json.dumps(payload, indent=2, default=_json_default))
    pd.DataFrame([a_row] + [rows[n] for n in rows]).to_csv(
        OUT_DIR / f"{stem}_table.csv", index=False)

    md = render_markdown(payload)
    if args.append_experiments:
        with EXPERIMENTS.open("a", encoding="utf-8") as fh:
            fh.write(md)
    else:
        log.info("--append-experiments not set: results NOT appended to %s "
                 "(experiments.md is append-only and is for graded runs only)", EXPERIMENTS)
    print(md)
    log.info("done in %.1f min", (time.time() - t0) / 60)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return str(o)


def decide(rows: dict, verdicts: dict, act: dict) -> dict:
    scored = []
    for name, v in verdicts.items():
        g8 = sum(1 for c in G8_CELLS if v.get(c))
        state = sum(1 for c in STATE_CELLS if v.get(c))
        eligible = all(v.get(c) for c in BLOWOUT_CELLS)
        scored.append({
            "arm": name, "g8_pass": g8, "g8_cells": len(G8_CELLS),
            "state_pass": state, "state_cells": len(STATE_CELLS),
            "total_pass": g8 + state, "eligible": bool(eligible),
            "ks_lineup_top1_d": rows[name]["ks_lineup_top1_d"],
            "simplicity_rank": {"R1_dirichlet": 1, "R2_hier_dirichlet": 2,
                                "R4_stint_resample": 3, "R3_stint_hazard": 4}[name],
        })
    scored.sort(key=lambda r: (-r["total_pass"], r["ks_lineup_top1_d"], r["simplicity_rank"]))
    winners = [r for r in scored if r["eligible"]]
    return {
        "ranking": scored,
        "winner": winners[0]["arm"] if winners else None,
        "diagnosis": None if winners else
        "no arm reproduces both blowout and foul-trouble behaviour inside tolerance",
    }


def _pf(ok) -> str:
    return "PASS" if ok else "FAIL"


def render_markdown(p: dict) -> str:
    a, rows, v, noise = p["actual"], p["arms"], p["verdicts"], p["noise"]
    names = list(rows)
    L = []
    W = L.append
    W(f"\n## 2. F1 results (train {p['train_season']}, test {p['test_season']}) -- "
      f"run {p['generated_at'][:19]}Z\n")
    W(f"Config: {p['n_simulated_games']} test games (of {p['n_eligible_games']} eligible, "
      f"{p['n_test_games_complete_onfloor']} with complete on-floor data), "
      f"{p['config']['seeds']} seeds per arm for the comparison and "
      f"{p['config']['noise_seeds']} seeds x {p['config']['noise_games']} games for the "
      f"noise floor. Rotation player-games per arm: "
      f"{rows[names[0]]['n_rot_player_games']:,} (actual {a['n_rot_player_games']:,}) -- "
      f"every cell below is far above the n < {UNDERPOWERED_N} UNDERPOWERED threshold "
      f"unless flagged.\n")
    W("### 2.1 Player crosswalk coverage of the modelling path\n")
    c = p["crosswalk_coverage"]
    W(f"| as-of feature rows | CBBD id -> ESPN id | joined to a hoopR box row | "
      f"hoopR `did_not_play` agrees with zero on-floor minutes |")
    W("|---:|---:|---:|---:|")
    W(f"| {c['asof_rows']:,} | {c['crosswalk_hit']*100:.3f}% | {c['hoopr_box_join']*100:.3f}% "
      f"| {c['dnp_agreement']*100:.2f}% |\n")

    W("### 2.2 G8 and the CFB narrow/short pair\n")
    W("| metric | ACTUAL | " + " | ".join(names) + " | tol |")
    W("|---|---:|" + "---:|" * len(names) + "---|")

    def line(label, key, fmt="{:.4f}", scale=1.0, tol="", vkey=None):
        cells = []
        for n in names:
            val = rows[n][key] * scale
            mark = "" if vkey is None else "  " + _pf(v[n].get(vkey))
            cells.append(fmt.format(val) + mark)
        W(f"| {label} | " + fmt.format(a[key] * scale) + " | " + " | ".join(cells) +
          f" | {tol} |")

    line("minutes mean (rotation players)", "minutes_mean", "{:.2f}", 1.0,
         "+/- 2.0", "minutes_mean")
    line("minutes SD, pooled", "minutes_sd_pooled", "{:.2f}", 1.0, "ratio 0.9-1.1")
    W("| SD ratio, pooled | 1.000 | " + " | ".join(
        f"{v[n]['_sd_ratio_pooled_val']:.3f}  {_pf(v[n]['sd_ratio_pooled'])}" for n in names) +
      " | 0.9-1.1 |")
    line("minutes SD, within-player", "minutes_sd_within", "{:.2f}", 1.0, "ratio 0.9-1.1")
    W("| SD ratio, within-player | 1.000 | " + " | ".join(
        f"{v[n]['_sd_ratio_within_val']:.3f}  {_pf(v[n]['sd_ratio_within'])}" for n in names) +
      " | 0.9-1.1 |")
    line("top-5 share of team minutes", "top5_share", "{:.4f}", 1.0, "+/- 2 pp", "top5_share")
    line("top-8 share of team minutes", "top8_share", "{:.4f}", 1.0, "+/- 2 pp", "top8_share")
    line("players with > 0 minutes", "n_nonzero_mean", "{:.2f}", 1.0, "+/- 1.0", "n_nonzero")
    W("| K-S of per-player minutes (D / p) | -- | " + " | ".join(
        f"{rows[n]['ks_minutes_d']:.4f} / {rows[n]['ks_minutes_p']:.2e}" for n in names) +
      " | report |")

    W("\n### 2.3 Lineup concentration\n")
    W("| metric | ACTUAL | " + " | ".join(names) + " |")
    W("|---|---:|" + "---:|" * len(names))
    for label, key in (("top-1 lineup share of possessions", "lu_top1"),
                       ("top-3 lineup share", "lu_top3"),
                       ("top-5 lineup share", "lu_top5"),
                       ("distinct lineups per team-game", "n_lineups_mean")):
        fmt = "{:.2f}" if key == "n_lineups_mean" else "{:.4f}"
        W(f"| {label} | " + fmt.format(a[key]) + " | " +
          " | ".join(fmt.format(rows[n][key]) for n in names) + " |")
    W("| K-S of the top-1 lineup share distribution (D) | -- | " +
      " | ".join(f"{rows[n]['ks_lineup_top1_d']:.4f}" for n in names) + " |")
    W("| substitution rate at a possession boundary | " +
      f"{p['fit_notes']['observed_change_rate_train']:.4f} (train) | " +
      " | ".join(f"{rows[n].get('change_rate', float('nan')):.4f}" for n in names) + " |")

    W("\n### 2.4 State dependence\n")
    W("| metric | ACTUAL | " + " | ".join(names) + " | tol |")
    W("|---|---:|" + "---:|" * len(names) + "---|")
    band_label = {0: r"\|margin\| <= 5", 1: r"\|margin\| 6-15", 2: r"\|margin\| > 15"}
    for b in range(R.N_MARGIN_BUCKETS):
        key = f"late_starter_share_b{b}"
        n_pg = a[f"late_slots_b{b}"]
        flag = "  **UNDERPOWERED**" if n_pg < UNDERPOWERED_N else ""
        W(f"| starters' share of on-floor slots, final 8:00, {band_label[b]}"
          f" (n={n_pg:,.0f} poss){flag} | {a[key]:.4f} | " +
          " | ".join(f"{rows[n][key]:.4f}  {_pf(v[n][key])}" for n in names) + " | +/- 3 pp |")
    n_ft = a["foul_trouble_opps"]
    flag = "  **UNDERPOWERED**" if n_ft < UNDERPOWERED_N else ""
    W(f"| starters on floor while carrying >= 4 fouls (n={n_ft:,.0f}){flag} | "
      f"{a['foul_trouble_share']:.4f} | " +
      " | ".join(f"{rows[n]['foul_trouble_share']:.4f}  {_pf(v[n]['foul_trouble_share'])}"
                 for n in names) + " | +/- 3 pp |")
    W(f"| (diagnostic) starters on floor at exactly 4 fouls (n={a['foul4_opps']:,.0f}) | "
      f"{a['foul4_share']:.4f} | " +
      " | ".join(f"{rows[n]['foul4_share']:.4f}" for n in names) + " | report |")
    dg = p.get("actual_asof_starters") or {}
    if dg:
        W(f"\n**Diagnostic decomposition.** The rows above count, on each side, the five "
          f"that side actually started. The model's as-of starter set overlaps the real "
          f"starting five on **{dg['starter_overlap_of5']:.2f} of 5** players, so part of "
          f"any gap is picking the wrong fifth man rather than rotating him wrongly. "
          f"Re-grading the **actual** on-floor sequence with the **model's** as-of starter "
          f"set isolates that:\n")
        W("| metric | ACTUAL (own starters) | ACTUAL (as-of starter set) | " +
          " | ".join(names) + " |")
        W("|---|---:|---:|" + "---:|" * len(names))
        for b in range(R.N_MARGIN_BUCKETS):
            k = f"late_starter_share_b{b}"
            W(f"| final 8:00 starters' share, {band_label[b]} | {a[k]:.4f} | {dg[k]:.4f} | " +
              " | ".join(f"{rows[n][k]:.4f}" for n in names) + " |")
        W(f"| starters on floor while carrying >= 4 fouls | {a['foul_trouble_share']:.4f} | "
          f"{dg['foul_trouble_share']:.4f} | " +
          " | ".join(f"{rows[n]['foul_trouble_share']:.4f}" for n in names) + " |")
        W("")

    W("\n### 2.5 Noise floor (seed-varied runs)\n")
    W(f"SD across {p['config']['noise_seeds']} seeds on {p['config']['noise_games']} games:\n")
    W("| metric | " + " | ".join(names) + " |")
    W("|---|" + "---:|" * len(names))
    for k in ["minutes_mean", "minutes_sd_pooled", "top5_share", "top8_share",
              "n_nonzero_mean", "lu_top1", "late_starter_share_b2", "foul_trouble_share"]:
        W(f"| {k} | " + " | ".join(f"{noise[n][k]:.5f}" for n in names) + " |")

    if p.get("robustness"):
        rb = p["robustness"]
        W(f"\n### 2.6 Robustness -- within-2025 walk-forward "
          f"(train games before 2025-01-15, test after; {rb.get('_n_games', 0)} games)\n")
        W("| metric | ACTUAL | " + " | ".join(names) + " |")
        W("|---|---:|" + "---:|" * len(names))
        for label, key, fmt in (("minutes mean", "minutes_mean", "{:.2f}"),
                                ("SD ratio pooled", "minutes_sd_pooled", "{:.2f}"),
                                ("top-5 share", "top5_share", "{:.4f}"),
                                ("players > 0 min", "n_nonzero_mean", "{:.2f}"),
                                ("late starter share, |m| > 15",
                                 "late_starter_share_b2", "{:.4f}"),
                                ("foul-trouble share", "foul_trouble_share", "{:.4f}")):
            if key not in rb.get("ACTUAL", {}):
                continue
            W(f"| {label} | " + fmt.format(rb["ACTUAL"][key]) + " | " +
              " | ".join(fmt.format(rb[n][key]) if n in rb else "--" for n in names) + " |")

    W("\n### 2.7 Decision\n")
    W("| arm | G8 cells passed | state cells passed | total | eligible "
      "(blowout + foul trouble) | lineup-concentration K-S D | simplicity |")
    W("|---|---:|---:|---:|---|---:|---:|")
    for r in p["decision"]["ranking"]:
        W(f"| {r['arm']} | {r['g8_pass']}/{r['g8_cells']} | "
          f"{r['state_pass']}/{r['state_cells']} | {r['total_pass']} | "
          f"{'yes' if r['eligible'] else 'NO'} | {r['ks_lineup_top1_d']:.4f} | "
          f"{r['simplicity_rank']} |")
    if p["decision"]["winner"]:
        W(f"\n**Winner: `{p['decision']['winner']}`** by the pre-registered rule "
          f"(most G8 + state-dependence cells inside tolerance, ties broken by "
          f"lineup-concentration K-S then simplicity).\n")
    else:
        W(f"\n**No arm adopted.** {p['decision']['diagnosis']}. Per the pre-registration, "
          f"we adopt nothing and report the diagnosis.\n")
    W("\n### 2.8 Fitted parameters\n")
    f = p["fit"]
    W("| parameter | value | fitted how |")
    W("|---|---:|---|")
    W(f"| `k0` (shrinkage weight) | {f['k0']:.2f} | grid, min next-game minutes MAE on train |")
    W(f"| `w_dnp` | {f['w_dnp']:.3f} | realised/predicted minutes of last-game-DNP players |")
    W(f"| `tail_ratio` | {f['tail_ratio']:.3f} | median share ratio of successive tail ranks |")
    W(f"| `n_profile` | {f['n_profile']} | joint scheduler grid (nonzero-count target) |")
    W(f"| `alpha` (R1) | {f['alpha']:.1f} | method of moments on share residual variance |")
    W(f"| `alpha_family` (R2) | {f['alpha_family']:.1f} | same, family level |")
    W(f"| `alpha_starters` (R2) | {f['alpha_starters']:.1f} | same, within starters |")
    W(f"| `alpha_bench` (R2) | {f['alpha_bench']:.1f} | same, within bench |")
    W(f"| `ema_horizon` (s) | {f['ema_horizon']:.0f} | joint grid on train "
      f"(sub rate, distinct lineups, top-5 share) |")
    W(f"| `swap_threshold` | {f['swap_threshold']:.4g} | same joint grid |")
    W(f"| `lam_deficit` | {f['lam_deficit']:.2f} | same joint grid |")
    W(f"| `foul_rate_scale` | {f['foul_rate_scale']:.3f} | match train team fouls per game |")
    W(f"| `fpm_league` (fouls per on-floor minute) | {f['fpm_league']:.4f} | train pooled |")
    W(f"| `fpm_prior_min` | {f['fpm_prior_min']:.0f} | grid, min next-game foul MAE |")
    W(f"| `w_avail_dnp` | {f['w_avail_dnp']:.3f} | realised play rate of last-game-DNP players |")
    W(f"| `min_share` | {f['min_share']:.5f} | mean share of the smallest nonzero-minutes "
      f"player in a team-game |")
    W(f"| `start_alpha` (starter-predictor EWMA decay) | {f['start_alpha']:.2f} | "
      f"grid, max overlap with the real starting five on train |")
    W(f"| `p_play` by as-of rank 1-12 | {', '.join(f'{v:.2f}' for v in f['p_play'][:12])} | "
      f"train P(records any minutes) |")
    W("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
