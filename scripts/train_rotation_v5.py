#!/usr/bin/env python
"""
train_rotation_v5.py -- ROUND 5 of the L4 rotation bake-off: the JOINT
substitution wave at a dead ball.

Pre-registration: `docs/models/rotation/experiments.md` section 12, written and
committed BEFORE this script was run (commit a14a569). Evidence it is built on:
`docs/tests/rotation_wave_audit_2026-09-11.md`.

Arms
----
    W1_wave_rank        wave Bernoulli + size categorical; composition RANKED
                        on round 4's hazards (largest p_out leaves, largest
                        p_in enters)
    W2_wave_draw        the same, composition DRAWN on both sides
    W4_wave_rank_draw   rank the exits, draw the entries
    W5_wave_draw_rank   draw the exits, rank the entries
    W3_wave_coupled     W1 plus one shared dead-ball draw, coupling the two
                        teams at the fitted `rho`
    H1_sub_hazard       round 4's arm, 1 seed, as a REPRODUCTION CHECK of the
                        reference columns taken from round 4's results JSON

ROUND 4'S HAZARDS ARE REUSED, NOT REFITTED, and so are the round-3b base fits.
Round 5 fits only the two wave tables and the coupling scalar, per window, so a
difference between a round-5 arm and H1 is a difference in the DRAW alone.

Usage:
    .venv/Scripts/python.exe scripts/train_rotation_v5.py --smoke
    .venv/Scripts/python.exe scripts/train_rotation_v5.py --workers 3
    .venv/Scripts/python.exe scripts/train_rotation_v5.py --mode nostate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from cbb_sim.models import rotation_v5 as V5  # noqa: E402

import train_rotation_v1 as V1  # noqa: E402
import train_rotation_v4 as V4T  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "models" / "rotation"
R5_DIR = OUT_DIR / "round5"
TRAIN_SEASON, TEST_SEASON = 2024, 2025
TEST_SUBSET_SEED = 2025

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_rotation_v5")

#: the two cells round 5 adds. Tolerances are pre-registered in 12.6 as the
#: LARGER of a fixed number and 3x the floor-A seed SD.
NEW_CELLS_V5 = ["sub_rate_per_boundary", "distinct_lineups_per_game"]
FIXED_TOL_V5 = {"sub_rate_per_boundary": 0.015, "distinct_lineups_per_game": 1.5}
STATE_CELLS_V5 = list(V4T.STATE_CELLS_V4)

WAVE_ARMS = ["W1_wave_rank", "W2_wave_draw", "W4_wave_rank_draw",
             "W5_wave_draw_rank", "W3_wave_coupled"]
SIMPLICITY = {"R2_hier_dirichlet": 1, "H1_sub_hazard": 2, "W1_wave_rank": 3,
              "W2_wave_draw": 4, "W4_wave_rank_draw": 5, "W5_wave_draw_rank": 6,
              "W3_wave_coupled": 7}


# ---------------------------------------------------------------------------
# the two new cells -- ONE code path for sim and actual
# ---------------------------------------------------------------------------
def wave_cells(lineups_by_team_game: list[np.ndarray]) -> dict:
    """`sub_rate_per_boundary` and `distinct_lineups_per_game`, computed by the
    SAME functions on the simulated and on the actual on-floor sequence."""
    return {
        "sub_rate_per_boundary": float(R.sim_change_rate(lineups_by_team_game)),
        "distinct_lineups_per_game": float(np.mean(
            [len({tuple(sorted(int(x) for x in row)) for row in lu})
             for lu in lineups_by_team_game])),
    }


def verdict_v5(row: dict, act: dict, tol: dict) -> dict:
    v = V4T.verdict_v4(row, act)
    for c in NEW_CELLS_V5:
        v[c] = bool(abs(row[c] - act[c]) <= tol[c])
    return v


def build_row_v5(name, recs, rot_rows, lus, a_row, act_minutes, act_rot) -> dict:
    row = V4T.build_row_v4(name, recs, rot_rows, a_row, act_minutes, act_rot)
    row.update(wave_cells(lus))
    row["n_lineups_mean"] = row.get("n_lineups_mean", np.nan)
    d, p = R.ks_2samp(np.array([r["lu_top1"] for r in recs]),
                      np.array(a_row.get("_lu_top1_actual", [0.0])))
    row["ks_lineup_top1_d"], row["ks_lineup_top1_p"] = d, p
    return row


def sim_records_v5(arm, priors, scripts, keys_by_game, rotation_sets, seeds):
    """`train_rotation_v4.sim_records_v4` plus the game-level shared stream W3
    needs. The extra draw comes from its own generator, so every other arm's
    stream is byte-identical to round 4's."""
    recs, rot_rows, lus = [], [], []
    has_shared = hasattr(arm, "set_shared")
    for seed in seeds:
        for gid, sides in keys_by_game.items():
            rng = R.game_stream(seed, gid)
            if has_shared:
                npos = max(scripts[k].n for k in sides)
                arm.set_shared(V5.shared_stream(seed, gid, npos))
            for key in sides:
                pr, sc = priors[key], scripts[key]
                lu, fh = arm.simulate(pr, sc, rng)
                starters = set(int(x) for x in pr.pids[pr.starters()[:5]])
                rot = rotation_sets.get(key, set())
                stt = R.team_game_stats(lu, sc.dur, sc.time_bucket, sc.margin_bucket,
                                        starters, fh, pr.pids, rot)
                stt["extra"] = V4T.extra_cells(lu, sc.dur, sc.period, sc.start_clock,
                                               sc.margin, starters)
                stt["key"] = key
                recs.append(stt)
                lus.append(lu)
                for p, m in stt["rot_minutes"].items():
                    rot_rows.append((key[0], key[1], p, m, seed))
    return recs, rot_rows, lus


# ---------------------------------------------------------------------------
# context: everything both the parent and a worker derive from disk
# ---------------------------------------------------------------------------
_CTX: dict = {}


def build_context(args_d: dict) -> dict:
    if _CTX:
        return _CTX
    args = argparse.Namespace(**args_d)
    train24 = V1.load_season(TRAIN_SEASON)
    test = V1.load_season(TEST_SEASON)
    ss25 = V4.load_side_state(TEST_SEASON)
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

    rotation_sets = {}
    fs = test["feats"]
    fs = fs[[(int(g), int(t)) in kset for g, t in zip(fs["game_id"], fs["team_id"])]]
    for key, g in fs.groupby(["game_id", "team_id"], sort=False):
        rotation_sets[(int(key[0]), int(key[1]))] = set(
            int(p) for p in g.loc[g["mpg_asof_raw"] >= 10.0, "pid"])

    a_recs, a_rot, _ = V4T.actual_records_v4(test, keys, rotation_sets)
    a_row = V1.build_row("ACTUAL", a_recs, a_rot, {}, np.array([]))
    a_row.update(V4T.agg_extra(a_recs))
    a_row["_lu_top1_actual"] = [r["lu_top1"] for r in a_recs]
    act_minutes = V1.pooled_and_within_sd(a_rot)[3]
    a_row["minutes_mae"] = 0.0

    # ACTUAL on the two new cells, on THIS universe, through the same functions
    tps = test["tp"]
    tps = tps[[(int(g), int(t)) in kset for g, t in zip(tps["game_id"], tps["team_id"])]]
    a_lus = [g[R.SLOTS].to_numpy(dtype="int64")
             for _, g in tps.groupby(["game_id", "team_id"], sort=False)]
    a_row.update(wave_cells(a_lus))

    gdates = (test["tp"][["game_id", "game_date"]].drop_duplicates()
              .set_index("game_id")["game_date"])
    d = pd.to_datetime(pd.Series(gdates.loc[[g for g in sel if g in gdates.index]])
                       .dropna().unique())
    months = sorted({pd.Timestamp(x.year, x.month, 1) for x in d})
    games_by_window: dict[str, list[int]] = {m.strftime("%Y%m"): [] for m in months}
    for g in sel:
        dd = pd.Timestamp(gdates.loc[g])
        w = max([m for m in months if m <= dd], default=months[0])
        games_by_window[w.strftime("%Y%m")].append(g)

    base_fits = {}
    for m in months:
        tag = m.strftime("%Y%m")
        bp = OUT_DIR / f"rotation_fit_v3_S1_{tag}.json"
        base_fits[tag] = R.RotationFit.from_json(bp) if bp.exists() else static

    _CTX.update(dict(args=args, train24=train24, test=test, ss25=ss25, static=static,
                     scripts=scripts_te, keys_by_game=keys_by_game, sel=sel,
                     rotation_sets=rotation_sets, a_row=a_row, a_rot=a_rot,
                     a_recs=a_recs, act_minutes=act_minutes, months=months,
                     games_by_window=games_by_window, base_fits=base_fits,
                     priors_static=priors_static, eligible=len(eligible),
                     _priors={}, _ssi={}, _wave={}, _haz={}))
    return _CTX


def priors_for(scheme: str, tag: str):
    c = _CTX
    k = (scheme, tag)
    if k not in c["_priors"]:
        base = c["base_fits"][tag] if scheme == "S1" else c["static"]
        c["_priors"][k] = R.build_priors(c["test"]["feats"], base,
                                         min_prior_games=c["args"].min_prior_games)
    return c["_priors"][k]


def ssi_for(tag: str):
    c = _CTX
    if tag not in c["_ssi"]:
        c["_ssi"][tag] = V4.side_state_index(c["ss25"], c["games_by_window"][tag])
    return c["_ssi"][tag]


def wave_for(tag: str, suffix: str = ""):
    c = _CTX
    k = (tag, suffix)
    if k not in c["_wave"]:
        p = R5_DIR / f"rotation_v5_wave{suffix}_{tag}.json"
        c["_wave"][k] = V5.WaveFit.from_json(p)
    return c["_wave"][k]


def hazards_for(tag: str):
    c = _CTX
    if tag not in c["_haz"]:
        p = OUT_DIR / f"rotation_v4_sub_S1_{tag}.json"
        if not p.exists():
            p = OUT_DIR / "rotation_v4_sub_static.json"
        c["_haz"][tag] = V4.SubHazardFit.from_json(p)
    return c["_haz"][tag]


def make_arm(name: str, tag: str, scheme: str = "S1", suffix: str = ""):
    c = _CTX
    base = c["base_fits"][tag] if scheme == "S1" else c["static"]
    ss = ssi_for(tag)
    if name == "H1_sub_hazard":
        return V4.H1SubHazard(base, hazards_for(tag), ss)
    if name == "R2_hier_dirichlet":
        return R.R2HierDirichlet(base)
    return V5.ARMS[name](base, wave_for(tag, suffix), ss)


# ---------------------------------------------------------------------------
# the jobs a worker runs
# ---------------------------------------------------------------------------
def _init_worker(args_d: dict) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    build_context(args_d)


def run_job(job: tuple) -> dict:
    kind, name, scheme, seeds, suffix = job
    c = _CTX
    t0 = time.time()
    if kind == "grade":
        pooled_r, pooled_rot, pooled_lu = [], [], []
        for tag, gl in c["games_by_window"].items():
            if not gl:
                continue
            pw = priors_for(scheme, tag)
            arm = make_arm(name, tag, scheme, suffix)
            kbg = {g: c["keys_by_game"][g] for g in gl
                   if all(k in pw for k in c["keys_by_game"][g])}
            r1, r2, r3 = sim_records_v5(arm, pw, c["scripts"], kbg,
                                        c["rotation_sets"], seeds)
            pooled_r += r1
            pooled_rot += r2
            pooled_lu += r3
        row = build_row_v5(name, pooled_r, pooled_rot, pooled_lu, c["a_row"],
                           c["act_minutes"], c["a_rot"])
        cell = {}
        for r in pooled_r:
            k = r["key"]
            n, d = cell.get(k, (0.0, 0.0))
            cell[k] = (n + r["late_bands"][0][0], d + r["late_bands"][0][1])
        return {"kind": kind, "arm": name, "scheme": scheme, "suffix": suffix,
                "row": row, "late_by_key": {f"{a}_{b}": v for (a, b), v in cell.items()},
                "seconds": round(time.time() - t0, 1)}
    if kind == "floorA":
        nsel = set(c["sel"][: c["args"].noise_games])
        per_seed = []
        for s in range(c["args"].noise_seeds):
            recs, rot, lus = [], [], []
            for tag, gl in c["games_by_window"].items():
                keep = [g for g in gl if g in nsel]
                if not keep:
                    continue
                pw = priors_for("S1", tag)
                arm = make_arm(name, tag, "S1", suffix)
                kbg = {g: c["keys_by_game"][g] for g in keep
                       if all(k in pw for k in c["keys_by_game"][g])}
                r1, r2, r3 = sim_records_v5(arm, pw, c["scripts"], kbg,
                                            c["rotation_sets"], [1000 + s])
                recs += r1
                rot += r2
                lus += r3
            per_seed.append(build_row_v5(name, recs, rot, lus, c["a_row"],
                                         c["act_minutes"], c["a_rot"]))
        cells = (["minutes_mean", "top5_share", "top8_share", "n_nonzero_mean",
                  "lu_top1", "minutes_mae", "n_lineups_mean"]
                 + STATE_CELLS_V5 + NEW_CELLS_V5)
        out = {k: float(np.std([r[k] for r in per_seed], ddof=1)) for k in cells}
        out["n_seeds"] = c["args"].noise_seeds
        out["n_games"] = len(nsel)
        return {"kind": kind, "arm": name, "floor": out,
                "seconds": round(time.time() - t0, 1)}
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# fitting what round 5 adds
# ---------------------------------------------------------------------------
def fit_windows(c: dict, args, fit_seed: int, suffix: str = "",
                collapse_state: bool = False) -> list[dict]:
    """One wave fit per S1 window; the round-4 hazards are copied in unchanged."""
    R5_DIR.mkdir(parents=True, exist_ok=True)
    train24, test = c["train24"], c["test"]
    ss_all = pd.concat([V4.load_side_state(TRAIN_SEASON), c["ss25"]], ignore_index=True)
    tp25, ev25 = test["tp"], test["fouls"]
    entries = []
    for m in c["months"]:
        tag = m.strftime("%Y%m")
        if not c["games_by_window"][tag]:
            continue
        t0 = time.time()
        if m == c["months"][0]:
            tp_w, ev_w = train24["tp"], train24["fouls"]
            max_train = f"{TRAIN_SEASON}-04-30"
            hz_name = "rotation_v4_sub_static.json"
        else:
            mask = pd.to_datetime(tp25["game_date"]) < m
            gids = set(tp25.loc[mask, "game_id"].astype("int64"))
            tp_w = pd.concat([train24["tp"], tp25[tp25["game_id"].isin(gids)]],
                             ignore_index=True)
            ev_w = pd.concat([train24["fouls"], ev25[ev25["game_id"].isin(gids)]],
                             ignore_index=True)
            max_train = str((pd.Timestamp(m) - pd.Timedelta(days=1)).date())
            hz_name = f"rotation_v4_sub_S1_{tag}.json"
        hp = OUT_DIR / hz_name
        if not hp.exists():
            hp = OUT_DIR / "rotation_v4_sub_static.json"
            hz_name = hp.name
        counts = V5.build_wave_training(tp_w, sorted(tp_w["game_id"].unique()),
                                        side_state=ss_all, fouls=ev_w,
                                        max_team_games=args.wave_team_games,
                                        seed=fit_seed)
        wf = V5.fit_wave(counts, V4.SubHazardFit.from_json(hp), hazard_source=hz_name,
                         collapse_state=collapse_state)
        wf.notes.update({"window": tag, "max_train_date": max_train,
                         "fit_seed": fit_seed, "team_games": counts["team_games"]})
        p = R5_DIR / f"rotation_v5_wave{suffix}_{tag}.json"
        wf.to_json(p)
        entries.append({"refit_date": str(m.date()), "path": p.name,
                        "max_train_date": max_train, "hazards": hz_name,
                        "rho": round(wf.rho, 4), "n_boundaries": wf.n_boundaries,
                        "wave_rate": round(wf.n_waves / max(wf.n_boundaries, 1), 4)})
        log.info("  wave window %s: %d team-games, %d boundaries, rate %.4f, rho %.3f "
                 "(%.1fs)", tag, counts["team_games"], wf.n_boundaries,
                 wf.n_waves / max(wf.n_boundaries, 1), wf.rho, time.time() - t0)
    man = {"model": "rotation", "scheme": "S1", "fold": "F1", "season": TEST_SEASON,
           "arm": "round5_wave" + suffix, "artifacts": entries}
    (R5_DIR / f"rotation_v5_manifest{suffix}.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8")
    return entries


# ---------------------------------------------------------------------------
def slope_check(c: dict, late_by_key: dict, prior_share: dict) -> dict:
    keys = sorted({r["key"] for r in c["a_recs"] if r["key"] in prior_share})
    pr = np.array([prior_share[k] for k in keys])
    q = np.quantile(pr, [0.2, 0.4, 0.6, 0.8])
    bucket = np.searchsorted(q, pr)
    out = {"quintile_prior": [], "n": []}
    for i in range(5):
        m = bucket == i
        out["quintile_prior"].append(float(pr[m].mean()) if m.any() else np.nan)
        out["n"].append(int(m.sum()))
    act = {}
    for r in c["a_recs"]:
        k = r["key"]
        n, d = act.get(k, (0.0, 0.0))
        act[k] = (n + r["late_bands"][0][0], d + r["late_bands"][0][1])
    srcs = {"ACTUAL": {f"{a}_{b}": v for (a, b), v in act.items()}}
    srcs.update(late_by_key)
    for lbl, cell in srcs.items():
        vals = []
        for i in range(5):
            ks = [k for k, b in zip(keys, bucket) if b == i]
            nn = sum(cell.get(f"{k[0]}_{k[1]}", (0.0, 0.0))[0] for k in ks)
            dd = sum(cell.get(f"{k[0]}_{k[1]}", (0.0, 0.0))[1] for k in ks)
            vals.append(float(nn / dd) if dd else np.nan)
        out[lbl] = vals
        ok = ~np.isnan(np.array(vals))
        out[lbl + "_slope"] = float(np.polyfit(np.array(out["quintile_prior"])[ok],
                                               np.array(vals)[ok], 1)[0])
        out[lbl + "_q5_q1"] = float(vals[-1] - vals[0])
    return out


def decide_v5(rows: dict, verdicts: dict, noise: dict, ref_mae: float,
              ref_floor: float = 0.0) -> dict:
    out = {"table": [], "winner": None, "why": ""}
    eligible = []
    for n, row in rows.items():
        v = verdicts[n]
        state_ok = all(v[cc] for cc in STATE_CELLS_V5)
        new_ok = all(v[cc] for cc in NEW_CELLS_V5)
        # the pre-registered floor on the MAE comparison: the larger of the
        # arm's own seed floor and the reference's, which is round 4's rule.
        # R2 is not re-simulated here, so its floor is round 4's own noise A.
        fl = max(noise.get(n, {}).get("minutes_mae", 0.0), ref_floor)
        beats = (ref_mae - row["minutes_mae"]) > fl
        out["table"].append({
            "arm": n, "simplicity": SIMPLICITY.get(n, 99),
            "state_cells_passed": sum(1 for cc in STATE_CELLS_V5 if v[cc]),
            "state_cells": len(STATE_CELLS_V5),
            "new_cells_passed": sum(1 for cc in NEW_CELLS_V5 if v[cc]),
            "eligible": bool(state_ok and new_ok),
            "g8_cells_passed": sum(1 for cc in V1.G8_CELLS if v[cc]),
            "minutes_mae": row["minutes_mae"],
            "mae_vs_R2": float(ref_mae - row["minutes_mae"]),
            "mae_floor": float(fl), "beats_ref_beyond_floor": bool(beats),
            "sub_rate_per_boundary": row["sub_rate_per_boundary"],
            "distinct_lineups_per_game": row["distinct_lineups_per_game"],
        })
        if state_ok and new_ok and beats and n in WAVE_ARMS:
            eligible.append((SIMPLICITY.get(n, 99), n))
    out["table"].sort(key=lambda r: r["simplicity"])
    if eligible:
        eligible.sort()
        out["winner"] = eligible[0][1]
        out["why"] = ("simplest arm passing all eight state cells and both new cells "
                      "and beating R2_S1 on per-player minutes MAE beyond the floor")
    else:
        out["why"] = "no arm passes every pre-registered cell; adopt nothing"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-games", type=int, default=1600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise-seeds", type=int, default=20)
    ap.add_argument("--noise-games", type=int, default=150)
    ap.add_argument("--wave-team-games", type=int, default=6000)
    ap.add_argument("--min-prior-games", type=int, default=3)
    ap.add_argument("--fit-seed", type=int, default=11)
    ap.add_argument("--floor-fit-seed", type=int, default=101)
    ap.add_argument("--floor-seed", type=int, default=23)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-floor-a", action="store_true")
    ap.add_argument("--skip-floor-b", action="store_true")
    ap.add_argument("--mode", default="bakeoff", choices=["bakeoff", "nostate"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", type=str, default="F1-round5")
    args = ap.parse_args()
    if args.smoke:
        args.test_games, args.seeds = 60, 1
        args.noise_seeds, args.noise_games = 3, 30
        args.wave_team_games = 600
        args.workers = 1
    t0 = time.time()
    assert_not_sealed([TRAIN_SEASON, TEST_SEASON], context="rotation round 5")
    args_d = vars(args)
    c = build_context(args_d)
    log.info("test universe: %d of %d eligible games (subset seed %d); windows %s",
             len(c["sel"]), c["eligible"], TEST_SUBSET_SEED,
             {t: len(v) for t, v in c["games_by_window"].items()})
    log.info("ACTUAL: sub_rate %.4f distinct %.2f late %.4f/%.4f/%.4f h2tip "
             "%.4f/%.4f/%.4f open %.4f",
             c["a_row"]["sub_rate_per_boundary"], c["a_row"]["distinct_lineups_per_game"],
             c["a_row"]["late_starter_share_b0"], c["a_row"]["late_starter_share_b1"],
             c["a_row"]["late_starter_share_b2"], c["a_row"]["h2tip_starter_share_b0"],
             c["a_row"]["h2tip_starter_share_b1"], c["a_row"]["h2tip_starter_share_b2"],
             c["a_row"]["opentip_starter_share_close"])

    if args.mode == "nostate":
        log.info("fitting the L31 refit-WITHOUT-state wave tables")
        ent = fit_windows(c, args, args.fit_seed, suffix="_nostate", collapse_state=True)
        log.info("wrote %d no-state windows to %s", len(ent), R5_DIR)
        return

    log.info("fitting the round-5 wave tables (%d windows)", len(c["months"]))
    entries = fit_windows(c, args, args.fit_seed)

    jobs = [("grade", n, "S1", list(range(args.seeds)), "") for n in WAVE_ARMS]
    jobs.append(("grade", "H1_sub_hazard", "S1", [0], ""))
    jobs.append(("grade", "W1_wave_rank", "static", list(range(args.seeds)), ""))
    if not args.skip_floor_a:
        jobs += [("floorA", n, "S1", None, "") for n in WAVE_ARMS]

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(args_d,)) as ex:
            futs = {ex.submit(run_job, j): j for j in jobs}
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                log.info("done %s %s %s in %.1fs", r["kind"], r["arm"],
                         r.get("scheme", ""), r["seconds"])
    else:
        for j in jobs:
            r = run_job(j)
            results.append(r)
            log.info("done %s %s %s in %.1fs", r["kind"], r["arm"], r.get("scheme", ""),
                     r["seconds"])

    rows_s1 = {r["arm"]: r["row"] for r in results
               if r["kind"] == "grade" and r["scheme"] == "S1"}
    rows_static = {r["arm"] + "_static": r["row"] for r in results
                   if r["kind"] == "grade" and r["scheme"] == "static"}
    late_by_key = {r["arm"]: r["late_by_key"] for r in results
                   if r["kind"] == "grade" and r["scheme"] == "S1"}
    noise_a = {r["arm"]: r["floor"] for r in results if r["kind"] == "floorA"}

    # ---- reference columns from round 4, and the reproduction check --------
    r4 = json.loads((OUT_DIR / "rotation_F1_round4_results.json").read_text(
        encoding="utf-8"))
    ref = {}
    for n in ("R2_hier_dirichlet", "H1_sub_hazard"):
        row = dict(r4["arms_S1"][n])
        row["sub_rate_per_boundary"] = row.get("change_rate", np.nan)
        row["distinct_lineups_per_game"] = row.get("n_lineups_mean", np.nan)
        ref[n] = row
    repro = {"cells": {}, "ok": True}
    if "H1_sub_hazard" in rows_s1:
        h1 = rows_s1.pop("H1_sub_hazard")
        for cc in STATE_CELLS_V5 + ["minutes_mae", "sub_rate_per_boundary",
                                    "distinct_lineups_per_game"]:
            a, b = ref["H1_sub_hazard"].get(cc, np.nan), h1.get(cc, np.nan)
            fl = r4["noise_A"]["H1_sub_hazard"].get(cc, np.nan)
            repro["cells"][cc] = {"round4_3seed": a, "round5_1seed": b,
                                  "delta": float(b - a) if a == a else None,
                                  "floorA": fl}
            if cc in STATE_CELLS_V5 and a == a and fl == fl and abs(b - a) > fl:
                repro["ok"] = False
        repro["h1_reproduction_row"] = h1

    # ---- tolerances, verdicts, decision -----------------------------------
    tol = {}
    for cc in NEW_CELLS_V5:
        fl = max([noise_a.get(n, {}).get(cc, 0.0) for n in noise_a] or [0.0])
        tol[cc] = float(max(FIXED_TOL_V5[cc], 3.0 * fl))
        tol[cc + "_floor3x"] = float(3.0 * fl)
    all_rows = dict(ref)
    all_rows.update(rows_s1)
    verdicts = {n: verdict_v5(r, c["a_row"], tol) for n, r in all_rows.items()}
    decision = decide_v5(all_rows, verdicts, noise_a,
                         ref["R2_hier_dirichlet"]["minutes_mae"],
                         ref_floor=float(r4["noise_A"]["R2_hier_dirichlet"]["minutes_mae"]))

    prior_share = {k: float(v.share[v.starters()].sum())
                   for k, v in c["priors_static"].items()}
    slope = slope_check(c, late_by_key, prior_share)

    # ---- floor B: spec-identical wave refit under a second seed -----------
    floor_b = {}
    if not args.skip_floor_b:
        log.info("floor B: wave refit under fit seed %d", args.floor_fit_seed)
        fit_windows(c, args, args.floor_fit_seed, suffix="_seed2")
        nsel = set(c["sel"][: args.noise_games])
        for lbl, suf, sd in (("seed1", "", 0), ("seed2", "_seed2", args.floor_seed)):
            recs, rot, lus = [], [], []
            for tag, gl in c["games_by_window"].items():
                keep = [g for g in gl if g in nsel]
                if not keep:
                    continue
                pw = priors_for("S1", tag)
                arm = make_arm("W1_wave_rank", tag, "S1", suf)
                kbg = {g: c["keys_by_game"][g] for g in keep
                       if all(k in pw for k in c["keys_by_game"][g])}
                r1, r2, r3 = sim_records_v5(arm, pw, c["scripts"], kbg,
                                            c["rotation_sets"], [sd])
                recs += r1
                rot += r2
                lus += r3
            floor_b[lbl] = build_row_v5("W1_" + lbl, recs, rot, lus, c["a_row"],
                                        c["act_minutes"], c["a_rot"])
        nkeys = [k for tag, gl in c["games_by_window"].items() for g in gl
                 if g in nsel for k in c["keys_by_game"][g]]
        an, arot_n, _ = V4T.actual_records_v4(c["test"], nkeys, c["rotation_sets"])
        fb = V1.build_row("ACTUAL", an, arot_n, {}, np.array([]))
        fb.update(V4T.agg_extra(an))
        floor_b["ACTUAL"] = fb

    a_out = {k: v for k, v in c["a_row"].items() if not k.startswith("_")}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fold": args.tag, "round": 5, "scheme": "S1",
        "prereg_commit": "a14a569",
        "config": vars(args),
        "windows": {t: len(g) for t, g in c["games_by_window"].items()},
        "manifest": entries,
        "n_eligible_games": c["eligible"], "n_simulated_games": len(c["sel"]),
        "actual": a_out,
        "arms_S1": {n: r for n, r in all_rows.items()},
        "arms_static": rows_static,
        "reference_from_round4": list(ref),
        "h1_reproduction_check": repro,
        "verdicts_S1": verdicts, "tolerances_new_cells": tol,
        "noise_A": noise_a, "noise_B": floor_b,
        "slope": slope, "decision": decision,
    }
    stem = "rotation_F1_round5" + ("_SMOKE" if args.smoke else "")
    (OUT_DIR / f"{stem}_results.json").write_text(
        json.dumps(payload, indent=2, default=V1._json_default), encoding="utf-8")
    pd.DataFrame([a_out] + [all_rows[n] for n in all_rows]).to_csv(
        OUT_DIR / f"{stem}_table.csv", index=False)
    log.info("decision: %s", json.dumps(decision["table"], default=V1._json_default))
    log.info("winner: %s (%s)", decision["winner"], decision["why"])
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60,
             OUT_DIR / f"{stem}_results.json")


if __name__ == "__main__":
    main()
