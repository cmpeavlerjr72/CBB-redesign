#!/usr/bin/env python
"""
train_rotation_v4.py -- ROUND 4 of the L4 rotation bake-off: per-player
discrete-time substitution hazards, against the incumbent R2.

Pre-registration: `docs/models/rotation/experiments.md` section 10, written and
committed BEFORE this script was run. Evidence it is built on:
`docs/tests/rotation_sub_hazard_audit_2026-09-10.md`.

Arms
----
    R2_static   the incumbent, fit once on 2024 (rounds 1-3's scheme)
    R2_S1       the incumbent under the adopted S1 scheme (round 3b's column)
    H1          logistic sub-out / sub-in hazards + a HARD reset to the
                predicted starting five at the start of period 2
    H2          the same hazards, NO hard reset -- the reset must be EARNED
    H3          H1 with a LightGBM hazard

Every arm is graded under S1 with its static column reported alongside, through
the one blind grading path rounds 1-3 used (`train_rotation_v1.build_row` /
`verdict` / `rotation.aggregate_stats`), plus the two new pre-registered cells.

BASE FITS ARE REUSED, NOT REFITTED. `rotation_fit_v3.json` and the six
`rotation_fit_v3_S1_{YYYYMM}.json` written by round 3b are the base parameter
sets (role prior, shrinkage, availability, foul rate, the tilt tables and the
scheduler grid R2 needs). Round 4 fits ONLY what it adds -- the two hazards --
per window, on that window's own training data. Reusing them is what makes the
round-4 `R2_S1` column literally round 3b's `R2 S1` column, so a difference
between rounds cannot be a difference in R2's fit. Nothing is written to any of
those files.

Usage:
    .venv/Scripts/python.exe scripts/train_rotation_v4.py --smoke
    .venv/Scripts/python.exe scripts/train_rotation_v4.py
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
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import rotation as R  # noqa: E402
from cbb_sim.models import rotation_v4 as V4  # noqa: E402

import train_rotation_v1 as V1  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "models" / "rotation"
TRAIN_SEASON, TEST_SEASON = 2024, 2025
TEST_SUBSET_SEED = 2025

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_rotation_v4")

#: the two cells round 4 adds, from the audit. Tolerance +/- 3 pp, the same the
#: four round-3 state cells carry.
NEW_CELLS = ["h2tip_starter_share_b0", "h2tip_starter_share_b1",
             "h2tip_starter_share_b2", "opentip_starter_share_close"]


# ---------------------------------------------------------------------------
# the two new cells -- ONE code path for sim and actual
# ---------------------------------------------------------------------------
def extra_cells(lineups: np.ndarray, dur: np.ndarray, period: np.ndarray,
                start_clock: np.ndarray, margin: np.ndarray,
                starter_pids: set) -> dict:
    """Starter share of on-floor slots at the second-half tip (by margin band)
    and over the opening ten minutes of a close game.

    `h2_tip` is the FIRST possession of period 2 -- the cell the audit shows no
    arm in three rounds could reach. `opentip` is H1 20:00-10:00 at |m| <= 5,
    where R2 is -17 pp."""
    npos = lineups.shape[0]
    st = np.array([[1.0 if int(p) in starter_pids else 0.0 for p in row]
                   for row in lineups]).sum(axis=1)
    tc = V4.time_cell(np.asarray(period), np.asarray(start_clock))
    mb = R.margin_bucket(np.asarray(margin))
    out = {f"h2tip_b{b}": (0.0, 0.0) for b in range(3)}
    h2 = np.flatnonzero(np.asarray(period) == 2)
    if len(h2):
        k = int(h2[0])
        out[f"h2tip_b{int(mb[k])}"] = (float(st[k]), 5.0)
    m = (tc == 0) & (mb == 0)
    out["opentip_close"] = (float(st[m].sum()), float(5 * m.sum()))
    return out


def agg_extra(records: list[dict]) -> dict:
    o = {}
    for b in range(3):
        num = sum(r["extra"][f"h2tip_b{b}"][0] for r in records)
        den = sum(r["extra"][f"h2tip_b{b}"][1] for r in records)
        o[f"h2tip_starter_share_b{b}"] = float(num / den) if den else float("nan")
        o[f"h2tip_slots_b{b}"] = float(den)
    num = sum(r["extra"]["opentip_close"][0] for r in records)
    den = sum(r["extra"]["opentip_close"][1] for r in records)
    o["opentip_starter_share_close"] = float(num / den) if den else float("nan")
    o["opentip_slots_close"] = float(den)
    return o


def verdict_v4(row: dict, act: dict) -> dict:
    v = V1.verdict(row, act)
    for k in NEW_CELLS:
        v[k] = abs(row[k] - act[k]) * 100 <= V1.TOL["state_pp"]
    return v


STATE_CELLS_V4 = V1.STATE_CELLS + NEW_CELLS


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def actual_records_v4(test: dict, keys, rotation_sets, asof_starters=None):
    tp, fouls = test["tp"], test["fouls"]
    kset = set(keys)
    tps = tp[[(int(g), int(t)) in kset for g, t in zip(tp["game_id"], tp["team_id"])]]
    evg = {k: g for k, g in fouls.groupby(["game_id", "team_id"], sort=False)}
    recs, rot_rows, diag, overlaps = [], [], [], []
    for key, g in tps.groupby(["game_id", "team_id"], sort=False):
        key = (int(key[0]), int(key[1]))
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        dur = g["duration_s"].to_numpy(dtype="float64")
        tb, mb = g["time_bucket"].to_numpy(), g["margin_bucket"].to_numpy()
        per = g["period"].to_numpy()
        clk = g["start_clock"].to_numpy()
        mg = g["margin"].to_numpy()
        starters = set(int(x) for x in lu[0])
        pids = np.unique(lu)
        pid_index = {int(p): i for i, p in enumerate(pids)}
        fm = R.actual_foul_matrix(evg.get(key), per, clk, pid_index)
        rot = rotation_sets.get(key, set())
        stt = R.team_game_stats(lu, dur, tb, mb, starters, fm, pids, rot)
        stt["extra"] = extra_cells(lu, dur, per, clk, mg, starters)
        stt["key"] = key
        recs.append(stt)
        for p, m in stt["rot_minutes"].items():
            rot_rows.append((key[0], key[1], p, m))
        if asof_starters is not None and key in asof_starters:
            pred = asof_starters[key]
            overlaps.append(len(pred & starters))
            d = R.team_game_stats(lu, dur, tb, mb, pred, fm, pids, rot)
            d["extra"] = extra_cells(lu, dur, per, clk, mg, pred)
            diag.append(d)
    return recs, rot_rows, (diag, overlaps)


def sim_records_v4(arm, priors, scripts, keys_by_game, rotation_sets, seeds):
    recs, rot_rows, lus = [], [], []
    for seed in seeds:
        for gid, sides in keys_by_game.items():
            rng = R.game_stream(seed, gid)
            for key in sides:
                pr, sc = priors[key], scripts[key]
                lu, fh = arm.simulate(pr, sc, rng)
                starters = set(int(x) for x in pr.pids[pr.starters()[:5]])
                rot = rotation_sets.get(key, set())
                stt = R.team_game_stats(lu, sc.dur, sc.time_bucket, sc.margin_bucket,
                                        starters, fh, pr.pids, rot)
                stt["extra"] = extra_cells(lu, sc.dur, sc.period, sc.start_clock,
                                           sc.margin, starters)
                stt["key"] = key
                recs.append(stt)
                lus.append(lu)
                for p, m in stt["rot_minutes"].items():
                    rot_rows.append((key[0], key[1], p, m, seed))
    return recs, rot_rows, lus


def minutes_mae(sim_rot, act_rot) -> tuple[float, float]:
    """Per-player minutes MAE -- the round-4 PRIMARY metric.

    Outer join on (game_id, team_id, pid) so a player the arm never plays and a
    player the arm invents both count their full minutes as error; per seed,
    then averaged over seeds so Monte-Carlo noise is not pooled into the level.
    """
    a = pd.DataFrame(act_rot, columns=["game_id", "team_id", "pid", "act"])
    s = pd.DataFrame(sim_rot, columns=["game_id", "team_id", "pid", "sim", "seed"])
    per = []
    for seed, g in s.groupby("seed"):
        m = g.merge(a, on=["game_id", "team_id", "pid"], how="outer")
        m["sim"] = m["sim"].fillna(0.0)
        m["act"] = m["act"].fillna(0.0)
        per.append(float((m["sim"] - m["act"]).abs().mean()))
    return float(np.mean(per)), float(np.std(per, ddof=1) if len(per) > 1 else 0.0)


def build_row_v4(name, recs, rot_rows, a_row, act_minutes, act_rot) -> dict:
    row = V1.build_row(name, recs, rot_rows, a_row, act_minutes)
    row.update(agg_extra(recs))
    mae, mae_sd = minutes_mae(rot_rows, act_rot)
    row["minutes_mae"] = mae
    row["minutes_mae_seed_sd"] = mae_sd
    return row


# ---------------------------------------------------------------------------
# fitting what round 4 adds
# ---------------------------------------------------------------------------
def fit_hazards(win: dict, base: R.RotationFit, side_state: pd.DataFrame,
                args, seed: int, fit_seed: int, want_lgbm: bool = True) -> dict:
    gids = sorted(win["tp"]["game_id"].unique())
    rs = np.random.RandomState(fit_seed)
    use = list(rs.choice(gids, size=min(args.hazard_games, len(gids)), replace=False))
    t0 = time.time()
    Xo, yo, Xi, yi = V4.build_sub_training(
        win["tp"], win["feats"], base, use, fouls=win["fouls"],
        side_state=side_state, max_team_games=args.hazard_team_games, seed=fit_seed)
    log.info("  hazard rows: out %s (base %.4f), in %s (base %.4f) in %.1fs",
             Xo.shape, float(np.mean(yo)), Xi.shape, float(np.mean(yi)),
             time.time() - t0)
    out = {"logistic": V4.fit_sub_models(Xo, yo, Xi, yi, kind="logistic", seed=seed)}
    if want_lgbm:
        t1 = time.time()
        out["lgbm"] = V4.fit_sub_models(Xo, yo, Xi, yi, kind="lgbm", seed=seed,
                                        n_estimators=args.lgbm_rounds,
                                        num_leaves=args.lgbm_leaves, threads=4)
        log.info("  lgbm hazards fitted in %.1fs", time.time() - t1)
    return out


def build_arms_v4(base: R.RotationFit, subs: dict, ss_index: dict) -> dict:
    lg = subs["logistic"]
    arms = {
        "R2_hier_dirichlet": R.R2HierDirichlet(base),
        "H1_sub_hazard": V4.H1SubHazard(base, lg, ss_index),
        "H2_sub_hazard_noreset": V4.H2SubHazardNoReset(base, lg, ss_index),
    }
    if "lgbm" in subs:
        arms["H3_sub_hazard_lgbm"] = V4.H3SubHazardLGBM(base, subs["lgbm"], ss_index)
    return arms


# ---------------------------------------------------------------------------
def month_windows(dates) -> list[pd.Timestamp]:
    d = pd.to_datetime(pd.Series(dates).dropna().unique())
    return sorted({pd.Timestamp(x.year, x.month, 1) for x in d})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-games", type=int, default=1600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--static-seeds", type=int, default=3)
    ap.add_argument("--noise-seeds", type=int, default=20)
    ap.add_argument("--noise-games", type=int, default=150)
    ap.add_argument("--hazard-games", type=int, default=1500,
                    help="game pool the hazard training team-games are drawn from")
    ap.add_argument("--hazard-team-games", type=int, default=800)
    ap.add_argument("--lgbm-rounds", type=int, default=300)
    ap.add_argument("--lgbm-leaves", type=int, default=31)
    ap.add_argument("--min-prior-games", type=int, default=3)
    ap.add_argument("--fit-seed", type=int, default=11)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--floor-fit-seed", type=int, default=101)
    ap.add_argument("--floor-seed", type=int, default=23)
    ap.add_argument("--skip-lgbm", action="store_true")
    ap.add_argument("--skip-floor-b", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", type=str, default="F1-round4")
    args = ap.parse_args()
    if args.smoke:
        args.test_games, args.seeds, args.static_seeds = 60, 1, 1
        args.noise_seeds, args.noise_games = 3, 30
        args.hazard_games, args.hazard_team_games = 200, 120
        args.lgbm_rounds = 60

    t0 = time.time()
    assert_not_sealed([TRAIN_SEASON, TEST_SEASON], context="rotation round 4")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train24 = V1.load_season(TRAIN_SEASON)
    test = V1.load_season(TEST_SEASON)
    ss24 = V4.load_side_state(TRAIN_SEASON)
    ss25 = V4.load_side_state(TEST_SEASON)
    ss_all = pd.concat([ss24, ss25], ignore_index=True)

    static = R.RotationFit.from_json(OUT_DIR / "rotation_fit_v3.json")

    # ---- test universe: identical to rounds 2, 3 and 3b -------------------
    priors_static = R.build_priors(test["feats"], static,
                                   min_prior_games=args.min_prior_games)
    scripts_te = R.build_scripts(test["tp"])
    both: dict[int, list] = {}
    for (gid, tid) in scripts_te:
        if (gid, tid) in priors_static:
            both.setdefault(gid, []).append((gid, tid))
    eligible = sorted([g for g, v in both.items() if len(v) == 2])
    rs = np.random.RandomState(TEST_SUBSET_SEED)
    sel = sorted(rs.choice(eligible, size=min(args.test_games, len(eligible)),
                           replace=False))
    keys_by_game = {g: ([k for k in both[g] if scripts_te[k].is_home]
                        + [k for k in both[g] if not scripts_te[k].is_home]) for g in sel}
    keys = [k for g in sel for k in keys_by_game[g]]
    kset = set(keys)
    log.info("test universe: %d of %d eligible games (subset seed %d)",
             len(sel), len(eligible), TEST_SUBSET_SEED)

    rotation_sets = {}
    fs = test["feats"]
    fs = fs[[(int(g), int(t)) in kset for g, t in zip(fs["game_id"], fs["team_id"])]]
    for key, g in fs.groupby(["game_id", "team_id"], sort=False):
        rotation_sets[(int(key[0]), int(key[1]))] = set(
            int(p) for p in g.loc[g["mpg_asof_raw"] >= 10.0, "pid"])
    asof_starters = {k: set(int(x) for x in v.pids[v.starters()[:5]])
                     for k, v in priors_static.items()}

    a_recs, a_rot, (a_diag, a_overlap) = actual_records_v4(test, keys, rotation_sets,
                                                           asof_starters)
    a_row = V1.build_row("ACTUAL", a_recs, a_rot, {}, np.array([]))
    a_row.update(agg_extra(a_recs))
    act_minutes = V1.pooled_and_within_sd(a_rot)[3]
    a_row["ks_minutes_d"], a_row["ks_minutes_p"] = np.nan, np.nan
    a_row["minutes_mae"] = 0.0
    diag_row = R.aggregate_stats(a_diag) if a_diag else {}
    if a_diag:
        diag_row.update(agg_extra(a_diag))
    diag_row["starter_overlap_of5"] = float(np.mean(a_overlap)) if a_overlap else np.nan
    log.info("ACTUAL: late b0/b1/b2 %.4f/%.4f/%.4f  ft %.4f  h2tip %.4f/%.4f/%.4f "
             "opentip %.4f", a_row["late_starter_share_b0"], a_row["late_starter_share_b1"],
             a_row["late_starter_share_b2"], a_row["foul_trouble_share"],
             a_row["h2tip_starter_share_b0"], a_row["h2tip_starter_share_b1"],
             a_row["h2tip_starter_share_b2"], a_row["opentip_starter_share_close"])

    # ---- S1 windows -------------------------------------------------------
    gdates = (test["tp"][["game_id", "game_date"]].drop_duplicates()
              .set_index("game_id")["game_date"])
    months = month_windows(gdates.loc[[g for g in sel if g in gdates.index]])
    games_by_window: dict[str, list[int]] = {m.strftime("%Y%m"): [] for m in months}
    for g in sel:
        d = pd.Timestamp(gdates.loc[g])
        w = max([m for m in months if m <= d], default=months[0])
        games_by_window[w.strftime("%Y%m")].append(g)
    log.info("S1 windows: %s", {t: len(v) for t, v in games_by_window.items()})

    tp25, feats25, pg25, ev25 = test["tp"], test["feats"], test["pg"], test["fouls"]

    def window_data(m: pd.Timestamp) -> dict:
        mask = pd.to_datetime(tp25["game_date"]) < m
        gids = set(tp25.loc[mask, "game_id"].astype("int64"))
        return {
            "tp": pd.concat([train24["tp"], tp25[tp25["game_id"].isin(gids)]],
                            ignore_index=True),
            "pg": pd.concat([train24["pg"], pg25[pg25["game_id"].isin(gids)]],
                            ignore_index=True),
            "feats": pd.concat([train24["feats"], feats25[feats25["game_id"].isin(gids)]],
                               ignore_index=True),
            "fouls": pd.concat([train24["fouls"], ev25[ev25["game_id"].isin(gids)]],
                               ignore_index=True),
            # the window trained on games with `game_date < m`, so the latest
            # date it can have seen is the day before. `manifest.py` checks
            # `max_train_date < game_date`; an upper bound is the safe side.
            "max_train_date": str((pd.Timestamp(m) - pd.Timedelta(days=1)).date()),
        }

    # ---- static fit of the two hazards (2024 only) -----------------------
    log.info("fitting the round-4 hazards, STATIC (2024 only)")
    subs_static = fit_hazards({"tp": train24["tp"], "feats": train24["feats"],
                               "fouls": train24["fouls"]}, static, ss24, args,
                              seed=args.seed, fit_seed=args.fit_seed,
                              want_lgbm=not args.skip_lgbm)
    subs_static["logistic"].notes["train_seasons"] = [TRAIN_SEASON]
    subs_static["logistic"].to_json(OUT_DIR / "rotation_v4_sub_static.json")

    # ---- per-window fits --------------------------------------------------
    base_fits: dict[str, R.RotationFit] = {}
    subs_by_window: dict[str, dict] = {}
    manifest_entries = []
    for m in months:
        tag = m.strftime("%Y%m")
        if not games_by_window[tag]:
            continue
        bp = OUT_DIR / f"rotation_fit_v3_S1_{tag}.json"
        base_fits[tag] = R.RotationFit.from_json(bp) if bp.exists() else static
        if m == months[0]:
            subs_by_window[tag] = subs_static
            manifest_entries.append({"refit_date": str(m.date()),
                                     "path": "rotation_v4_sub_static.json",
                                     "max_train_date": f"{TRAIN_SEASON}-04-30",
                                     "base_fit": bp.name if bp.exists() else
                                     "rotation_fit_v3.json"})
            log.info("window %s: training data is 2024 alone -> reuse the static hazards",
                     tag)
            continue
        ts = time.time()
        win = window_data(m)
        subs_by_window[tag] = fit_hazards(win, base_fits[tag], ss_all, args,
                                          seed=args.seed, fit_seed=args.fit_seed,
                                          want_lgbm=not args.skip_lgbm)
        p = OUT_DIR / f"rotation_v4_sub_S1_{tag}.json"
        subs_by_window[tag]["logistic"].notes["train_seasons"] = [TRAIN_SEASON, TEST_SEASON]
        subs_by_window[tag]["logistic"].notes["max_train_date"] = win["max_train_date"]
        subs_by_window[tag]["logistic"].to_json(p)
        manifest_entries.append({"refit_date": str(m.date()), "path": p.name,
                                 "max_train_date": win["max_train_date"],
                                 "base_fit": bp.name if bp.exists() else
                                 "rotation_fit_v3.json"})
        log.info("window %s hazards fitted in %.1f min", tag, (time.time() - ts) / 60)

    (OUT_DIR / "rotation_v4_manifest.json").write_text(json.dumps({
        "model": "rotation", "scheme": "S1", "fold": "F1", "season": TEST_SEASON,
        "arm": "round4_sub_hazard", "artifacts": manifest_entries}, indent=2),
        encoding="utf-8")

    # ---- caches (pure memoisation; changes no result) ---------------------
    _prior_cache: dict = {}
    _ssi_cache: dict = {}

    def priors_for(scheme: str, tag: str):
        k = (scheme, tag)
        if k not in _prior_cache:
            base = base_fits[tag] if scheme == "S1" else static
            _prior_cache[k] = R.build_priors(test["feats"], base,
                                             min_prior_games=args.min_prior_games)
        return _prior_cache[k]

    def ssi_for(tag: str, gl=None):
        if tag not in _ssi_cache:
            _ssi_cache[tag] = V4.side_state_index(ss25, games_by_window[tag])
        return _ssi_cache[tag]

    # ---- grade ------------------------------------------------------------
    arm_names = ["R2_hier_dirichlet", "H1_sub_hazard", "H2_sub_hazard_noreset"]
    if not args.skip_lgbm:
        arm_names.append("H3_sub_hazard_lgbm")

    def grade(scheme: str, seeds: list[int]) -> dict:
        pooled = {n: {"recs": [], "rot": [], "lus": []} for n in arm_names}
        for tag, gl in games_by_window.items():
            if not gl:
                continue
            base = base_fits[tag] if scheme == "S1" else static
            subs = subs_by_window[tag] if scheme == "S1" else subs_static
            priors_w = priors_for(scheme, tag)
            arms = build_arms_v4(base, subs, ssi_for(tag, gl))
            kbg = {g: keys_by_game[g] for g in gl
                   if all(k in priors_w for k in keys_by_game[g])}
            for n in arm_names:
                recs, rot, lus = sim_records_v4(arms[n], priors_w, scripts_te, kbg,
                                                rotation_sets, seeds)
                pooled[n]["recs"] += recs
                pooled[n]["rot"] += rot
                pooled[n]["lus"] += lus
            log.info("  [%s] window %s graded: %d games", scheme, tag, len(kbg))
        rows = {}
        for n in arm_names:
            row = build_row_v4(n, pooled[n]["recs"], pooled[n]["rot"], a_row,
                               act_minutes, a_rot)
            row["change_rate"] = R.sim_change_rate(pooled[n]["lus"])
            d, p = R.ks_2samp(np.array([r["lu_top1"] for r in pooled[n]["recs"]]),
                              np.array([r["lu_top1"] for r in a_recs]))
            row["ks_lineup_top1_d"], row["ks_lineup_top1_p"] = d, p
            rows[n] = row
            log.info("%-6s %-22s late %.4f/%.4f/%.4f ft %.4f h2tip %.4f/%.4f/%.4f "
                     "open %.4f top5 %.4f nz %.2f MAE %.3f", scheme, n,
                     row["late_starter_share_b0"], row["late_starter_share_b1"],
                     row["late_starter_share_b2"], row["foul_trouble_share"],
                     row["h2tip_starter_share_b0"], row["h2tip_starter_share_b1"],
                     row["h2tip_starter_share_b2"], row["opentip_starter_share_close"],
                     row["top5_share"], row["n_nonzero_mean"], row["minutes_mae"])
        return {"rows": rows, "pooled": pooled}

    log.info("grading STATIC")
    res_static = grade("static", list(range(args.static_seeds)))
    log.info("grading S1")
    res_s1 = grade("S1", list(range(args.seeds)))

    rows_s1 = res_s1["rows"]
    verdicts = {n: verdict_v4(rows_s1[n], a_row) for n in arm_names}
    verdicts_static = {n: verdict_v4(res_static["rows"][n], a_row) for n in arm_names}

    # ---- noise floor A: seed-varied sim runs -----------------------------
    log.info("noise floor A: %d seeds x %d games", args.noise_seeds, args.noise_games)
    nsel = sel[: args.noise_games]
    ntag = {}
    for tag, gl in games_by_window.items():
        keep = [g for g in gl if g in set(nsel)]
        if keep:
            ntag[tag] = keep
    noise_a = {}
    floor_cells = (V1.G8_CELLS[:0] + ["minutes_mean", "top5_share", "top8_share",
                                      "n_nonzero_mean", "lu_top1", "minutes_mae"]
                   + STATE_CELLS_V4)
    for n in arm_names:
        per_seed = []
        for s in range(args.noise_seeds):
            recs, rot = [], []
            for tag, gl in ntag.items():
                base, subs = base_fits[tag], subs_by_window[tag]
                priors_w = priors_for("S1", tag)
                arms = build_arms_v4(base, subs, ssi_for(tag, gl))
                kbg = {g: keys_by_game[g] for g in gl
                       if all(k in priors_w for k in keys_by_game[g])}
                r1, r2, _ = sim_records_v4(arms[n], priors_w, scripts_te, kbg,
                                           rotation_sets, [1000 + s])
                recs += r1
                rot += r2
            per_seed.append(build_row_v4(n, recs, rot, a_row, act_minutes, a_rot))
        noise_a[n] = {k: float(np.std([r[k] for r in per_seed], ddof=1))
                      for k in floor_cells}
        noise_a[n]["n_seeds"] = args.noise_seeds
        noise_a[n]["n_games"] = len(nsel)
        log.info("  %-22s floor: MAE %.4f late_b0 %.5f h2tip_b0 %.5f",
                 n, noise_a[n]["minutes_mae"], noise_a[n]["late_starter_share_b0"],
                 noise_a[n]["h2tip_starter_share_b0"])

    # ---- noise floor B: spec-identical refit under a second seed ----------
    floor_b = {}
    if not args.skip_floor_b:
        log.info("noise floor B: spec-identical hazard refit under seed %d/%d",
                 args.floor_fit_seed, args.floor_seed)
        subs2 = fit_hazards({"tp": train24["tp"], "feats": train24["feats"],
                             "fouls": train24["fouls"]}, static, ss24, args,
                            seed=args.floor_seed, fit_seed=args.floor_fit_seed,
                            want_lgbm=False)
        for lbl, sb in (("seed1", subs_static), ("seed2", subs2)):
            recs, rot = [], []
            for tag, gl in ntag.items():
                priors_w = priors_for("static", tag)
                arms = build_arms_v4(static, sb, ssi_for(tag, gl))
                kbg = {g: keys_by_game[g] for g in gl
                       if all(k in priors_w for k in keys_by_game[g])}
                r1, r2, _ = sim_records_v4(arms["H1_sub_hazard"], priors_w, scripts_te,
                                           kbg, rotation_sets, [0])
                recs += r1
                rot += r2
            floor_b[lbl] = build_row_v4("H1_" + lbl, recs, rot, a_row, act_minutes, a_rot)
        an, arot_n = [], []
        nkeys = [k for tag, gl in ntag.items() for g in gl for k in keys_by_game[g]]
        an, arot_n, _ = actual_records_v4(test, nkeys, rotation_sets)
        floor_b["ACTUAL"] = V1.build_row("ACTUAL", an, arot_n, {}, np.array([]))
        floor_b["ACTUAL"].update(agg_extra(an))
        floor_b["coefs_seed1"] = subs_static["logistic"].out_coef
        floor_b["coefs_seed2"] = subs2["logistic"].out_coef

    # ---- slope check (Decision 8) ----------------------------------------
    prior_share = {}
    for k, v in priors_static.items():
        s5 = float(v.share[v.starters()].sum())
        prior_share[k] = s5
    slope = slope_check(a_recs, res_s1["pooled"], arm_names, prior_share)

    decision = decide_v4(rows_s1, verdicts, a_row, noise_a)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fold": args.tag, "round": 4, "scheme": "S1 (static reported alongside)",
        "config": vars(args),
        "windows": {t: len(g) for t, g in games_by_window.items()},
        "manifest": manifest_entries,
        "n_eligible_games": len(eligible), "n_simulated_games": len(sel),
        "actual": a_row, "actual_asof_starters": diag_row,
        "arms_S1": rows_s1, "arms_static": res_static["rows"],
        "verdicts_S1": verdicts, "verdicts_static": verdicts_static,
        "noise_A": noise_a, "noise_B": floor_b,
        "slope": slope, "decision": decision,
        "hazard_features": list(V4.SUB_FEATURES),
        "hazard_coefs_static": {
            "out": subs_static["logistic"].out_coef,
            "out_intercept": subs_static["logistic"].out_intercept,
            "in": subs_static["logistic"].in_coef,
            "in_intercept": subs_static["logistic"].in_intercept,
            "n_out": subs_static["logistic"].n_out,
            "n_in": subs_static["logistic"].n_in,
            "base_out": subs_static["logistic"].base_out,
            "base_in": subs_static["logistic"].base_in,
        },
    }
    stem = "rotation_F1_round4" + ("_SMOKE" if args.smoke else "")
    (OUT_DIR / f"{stem}_results.json").write_text(
        json.dumps(payload, indent=2, default=V1._json_default), encoding="utf-8")
    pd.DataFrame([a_row] + [rows_s1[n] for n in arm_names]).to_csv(
        OUT_DIR / f"{stem}_table.csv", index=False)
    log.info("decision: %s", json.dumps(decision, default=V1._json_default))
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60,
             OUT_DIR / f"{stem}_results.json")


def slope_check(a_recs, pooled, arm_names, prior_share) -> dict:
    """Close-and-late starters' share by quintile of the pregame team prior."""
    def cell(recs):
        num, den = {}, {}
        for r in recs:
            k = r["key"]
            num[k] = num.get(k, 0.0) + r["late_bands"][0][0]
            den[k] = den.get(k, 0.0) + r["late_bands"][0][1]
        return num, den

    keys = sorted({r["key"] for r in a_recs if r["key"] in prior_share})
    pr = np.array([prior_share[k] for k in keys])
    q = np.quantile(pr, [0.2, 0.4, 0.6, 0.8])
    bucket = np.searchsorted(q, pr)
    out = {"quintile_prior": [], "n": []}
    for i in range(5):
        m = bucket == i
        out["quintile_prior"].append(float(pr[m].mean()) if m.any() else np.nan)
        out["n"].append(int(m.sum()))
    for lbl, recs in [("ACTUAL", a_recs)] + [(n, pooled[n]["recs"]) for n in arm_names]:
        num, den = cell(recs)
        vals = []
        for i in range(5):
            ks = [k for k, b in zip(keys, bucket) if b == i]
            nn = sum(num.get(k, 0.0) for k in ks)
            dd = sum(den.get(k, 0.0) for k in ks)
            vals.append(float(nn / dd) if dd else np.nan)
        out[lbl] = vals
        ok = ~np.isnan(np.array(vals)) & ~np.isnan(np.array(out["quintile_prior"]))
        out[lbl + "_slope"] = float(np.polyfit(np.array(out["quintile_prior"])[ok],
                                               np.array(vals)[ok], 1)[0])
        out[lbl + "_q5_q1"] = float(vals[-1] - vals[0])
    return out


def decide_v4(rows, verdicts, act, noise) -> dict:
    """The pre-registered rule: the SIMPLEST arm passing every state cell (the
    four round-3 cells and the two new ones) AND beating R2_S1 beyond the floor
    on per-player minutes MAE. Ties to the simpler arm, R2 < H1 < H2 < H3."""
    order = {"R2_hier_dirichlet": 1, "H1_sub_hazard": 2,
             "H2_sub_hazard_noreset": 3, "H3_sub_hazard_lgbm": 4}
    ref = rows.get("R2_hier_dirichlet", {})
    ref_mae = ref.get("minutes_mae", np.nan)
    out = {"table": [], "winner": None, "why": ""}
    eligible = []
    for n, row in rows.items():
        v = verdicts[n]
        state_ok = all(v[c] for c in STATE_CELLS_V4)
        g8 = sum(1 for c in V1.G8_CELLS if v[c])
        fl = max(noise.get(n, {}).get("minutes_mae", 0.0),
                 noise.get("R2_hier_dirichlet", {}).get("minutes_mae", 0.0))
        beats = (ref_mae - row["minutes_mae"]) > fl
        out["table"].append({
            "arm": n, "simplicity": order.get(n, 99),
            "state_cells_passed": sum(1 for c in STATE_CELLS_V4 if v[c]),
            "state_cells": len(STATE_CELLS_V4), "eligible": bool(state_ok),
            "g8_cells_passed": g8, "minutes_mae": row["minutes_mae"],
            "mae_vs_R2": float(ref_mae - row["minutes_mae"]),
            "mae_floor": float(fl), "beats_ref_beyond_floor": bool(beats),
            "lineup_ks_d": row.get("ks_lineup_top1_d"),
        })
        if state_ok and (n == "R2_hier_dirichlet" or beats):
            eligible.append((order.get(n, 99), n))
    out["table"].sort(key=lambda r: r["simplicity"])
    if eligible:
        eligible.sort()
        out["winner"] = eligible[0][1]
        out["why"] = ("simplest arm passing every state cell and (unless it is the "
                      "incumbent) beating R2_S1 on per-player minutes MAE beyond the floor")
    else:
        out["why"] = "no arm passes every pre-registered state cell; adopt nothing"
    return out


if __name__ == "__main__":
    main()
