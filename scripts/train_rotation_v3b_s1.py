#!/usr/bin/env python
"""
train_rotation_v3b_s1.py -- ROUND 3b: does the standing S1 training scheme
change any round-3 conclusion?

S1 (L21, `docs/models/README.md`) is the project's default training scheme: at
each month boundary of the test season, refit on every prior season plus the
test season to date, strictly before the refit date, and simulate each game with
the parameter set whose window closed before its tipoff. Rounds 1-3 of the L4
rotation bake-off all used the static scheme (fit once on 2024), because they
were specified before S1 became the default. This script runs the same arms,
the same test universe and the same grading code under S1 and reports both
columns.

Windows are the calendar months of the 2024-25 season. The first window's
training data is 2024 alone, which IS the static fit, so it is reused rather
than refitted -- an S1 run whose first window differed from the static fit would
be measuring two things at once.

Usage:
    .venv/Scripts/python.exe scripts/train_rotation_v3b_s1.py
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
from cbb_sim.models import rotation as R  # noqa: E402
from cbb_sim.models import rotation_v3 as V3  # noqa: E402

import train_rotation_v1 as V1  # noqa: E402
import train_rotation_v3 as V3T  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "models" / "rotation"
TRAIN_SEASON, TEST_SEASON = 2024, 2025
TEST_SUBSET_SEED = 2025

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_rotation_v3b_s1")


def month_windows(dates: pd.Series) -> list[pd.Timestamp]:
    """Calendar month starts of the test season, excluding the first (whose
    training data is the prior season alone = the static fit)."""
    d = pd.to_datetime(pd.Series(dates).dropna().unique())
    months = sorted({pd.Timestamp(x.year, x.month, 1) for x in d})
    return months


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-games", type=int, default=1600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tilt-games", type=int, default=1500)
    ap.add_argument("--hazard-games", type=int, default=400)
    ap.add_argument("--theta-keys", type=int, default=150)
    ap.add_argument("--theta-games", type=int, default=250)
    ap.add_argument("--keep-games", type=int, default=300)
    ap.add_argument("--knob-games", type=int, default=250)
    ap.add_argument("--knob-keys", type=int, default=120)
    ap.add_argument("--min-prior-games", type=int, default=3)
    ap.add_argument("--tag", type=str, default="F1-round3b-S1")
    args = ap.parse_args()

    t0 = time.time()
    assert_not_sealed([TRAIN_SEASON, TEST_SEASON], context="rotation round 3b (S1)")

    train24 = V1.load_season(TRAIN_SEASON)
    test = V1.load_season(TEST_SEASON)

    # ---- the test universe: identical to rounds 2 and 3 -------------------
    static = R.RotationFit.from_json(OUT_DIR / "rotation_fit_v3.json")
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

    a_recs, a_rot, _ = V1.actual_records(test, keys, rotation_sets, asof_starters)
    a_row = V1.build_row("ACTUAL", a_recs, a_rot, {}, np.array([]))
    act_minutes = V1.pooled_and_within_sd(a_rot)[3]
    a_row["ks_minutes_d"], a_row["ks_minutes_p"] = np.nan, np.nan

    # ---- windows ---------------------------------------------------------
    gdates = (test["tp"][["game_id", "game_date"]].drop_duplicates()
              .set_index("game_id")["game_date"])
    months = month_windows(gdates.loc[[g for g in sel if g in gdates.index]])
    log.info("S1 windows: %s", [str(m.date()) for m in months])

    fits: dict[str, R.RotationFit] = {}
    games_by_window: dict[str, list[int]] = {m.strftime("%Y%m"): [] for m in months}
    for g in sel:
        d = pd.Timestamp(gdates.loc[g])
        w = max([m for m in months if m <= d], default=months[0])
        games_by_window[w.strftime("%Y%m")].append(g)

    tp25, feats25, pg25, ev25 = (test["tp"], test["feats"], test["pg"], test["fouls"])
    for m in months:
        tag = m.strftime("%Y%m")
        if not games_by_window[tag]:
            continue
        path = OUT_DIR / f"rotation_fit_v3_S1_{tag}.json"
        if m == months[0]:
            # training data is the prior season alone -- that IS the static fit
            fits[tag] = static
            static.to_json(path)
            log.info("window %s: prior season only -> reuse the static fit", tag)
            continue
        mask = pd.to_datetime(tp25["game_date"]) < m
        gids = set(tp25.loc[mask, "game_id"].astype("int64"))
        win = {
            "season": TEST_SEASON,
            "tp": pd.concat([train24["tp"], tp25[tp25["game_id"].isin(gids)]],
                            ignore_index=True),
            "pg": pd.concat([train24["pg"], pg25[pg25["game_id"].isin(gids)]],
                            ignore_index=True),
            "feats": pd.concat([train24["feats"], feats25[feats25["game_id"].isin(gids)]],
                               ignore_index=True),
            "fouls": pd.concat([train24["fouls"], ev25[ev25["game_id"].isin(gids)]],
                               ignore_index=True),
        }
        ts = time.time()
        f, _ = V1.fit_all(win, args)
        f.notes["r5_donor_k"] = static.notes["r5_donor_k"]
        f.notes["r5_exit"] = static.notes["r5_exit"]
        f.notes["r5_enter"] = static.notes["r5_enter"]
        f.notes["r5_features"] = static.notes["r5_features"]
        f.notes["r5_override_scale"] = static.notes["r5_override_scale"]
        f.notes["r5_block_base"] = static.notes["r5_block_base"]
        # the R5 override hazards, refitted on this window
        Xe, ye, Xn, yn = R.build_hazard_training(
            win["tp"], win["feats"], f,
            list(np.random.RandomState(11).choice(
                sorted(win["tp"]["game_id"].unique()),
                size=min(args.hazard_games, win["tp"]["game_id"].nunique()),
                replace=False)),
            fouls=win["fouls"], design=R._override_design)
        oex, oen = R.fit_hazard_models(Xe, ye, Xn, yn)
        f.notes["r5_exit"], f.notes["r5_enter"] = oex, oen
        f = V3T.fit_round3(win, f, args, seed=7, fit_seed=11)
        f.to_json(path)
        fits[tag] = f
        log.info("window %s refitted in %.1f min (%d train games) -> %s",
                 tag, (time.time() - ts) / 60, win["tp"]["game_id"].nunique(), path.name)

    # ---- grade, per window, pooled ---------------------------------------
    donors_full = R.build_donor_bank(test["tp"], k_donors=static.notes["r5_donor_k"])
    arm_names = ["R2_hier_dirichlet", "R5_hybrid", "R7_keep_logistic"]
    pooled: dict[str, dict] = {n: {"recs": [], "rot": [], "lus": []} for n in arm_names}
    for tag, gl in games_by_window.items():
        if not gl:
            continue
        f = fits[tag]
        priors_w = R.build_priors(test["feats"], f, min_prior_games=args.min_prior_games)
        arms = V3T.build_arms(f, donors_full)
        kbg = {g: keys_by_game[g] for g in gl if all(k in priors_w for k in keys_by_game[g])}
        for n in arm_names:
            recs, rot, lus = V1.sim_records(arms[n], priors_w, scripts_te, kbg,
                                            rotation_sets, list(range(args.seeds)))
            pooled[n]["recs"] += recs
            pooled[n]["rot"] += rot
            pooled[n]["lus"] += lus
        log.info("window %s graded: %d games", tag, len(kbg))

    rows, verdicts = {}, {}
    for n in arm_names:
        row = V1.build_row(n, pooled[n]["recs"], pooled[n]["rot"], a_row, act_minutes)
        row["change_rate"] = R.sim_change_rate(pooled[n]["lus"])
        d, p = R.ks_2samp(np.array([r["lu_top1"] for r in pooled[n]["recs"]]),
                          np.array([r["lu_top1"] for r in a_recs]))
        row["ks_lineup_top1_d"], row["ks_lineup_top1_p"] = d, p
        rows[n] = row
        verdicts[n] = V1.verdict(row, a_row)
        log.info("S1 %-20s late b0/b1/b2 %.4f/%.4f/%.4f ft %.4f top5 %.4f nonzero %.2f",
                 n, row["late_starter_share_b0"], row["late_starter_share_b1"],
                 row["late_starter_share_b2"], row["foul_trouble_share"],
                 row["top5_share"], row["n_nonzero_mean"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fold": args.tag, "round": "3b", "scheme": "S1",
        "config": vars(args),
        "windows": {t: len(g) for t, g in games_by_window.items()},
        "fit_files": {t: f"rotation_fit_v3_S1_{t}.json" for t in games_by_window if
                      games_by_window[t]},
        "window_knobs": {t: {"r7_block": fits[t].notes.get("r7_block"),
                             "r7_keep_scale": fits[t].notes.get("r7_keep_scale"),
                             "r7_keep_base": fits[t].notes.get("r7_keep_base"),
                             "r8_keep_theta": fits[t].notes.get("r8_keep_theta")}
                         for t in games_by_window if games_by_window[t]},
        "actual": a_row, "arms": rows, "verdicts": verdicts,
    }
    (OUT_DIR / "rotation_F1_round3b_S1_results.json").write_text(
        json.dumps(payload, indent=2, default=V1._json_default))
    print(json.dumps({n: {k: rows[n][k] for k in
                          ["late_starter_share_b0", "late_starter_share_b1",
                           "late_starter_share_b2", "foul_trouble_share",
                           "top5_share", "top8_share", "n_nonzero_mean",
                           "minutes_mean", "lu_top1"]} for n in arm_names}, indent=2))
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
