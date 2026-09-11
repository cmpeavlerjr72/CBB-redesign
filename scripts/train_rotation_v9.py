#!/usr/bin/env python
"""
train_rotation_v9.py -- ROUND 9 of the L4 rotation bake-off: the exit hierarchy
INVERTED onto the powered composition marginal.

Pre-registration: `docs/models/rotation/experiments.md` section 20, committed
(3525996) BEFORE `rotation_v9.py` existed and before anything was fitted.
Evidence it is built on: round 8's results (section 19) and its support
measurement 19.1.

Arms
----
    Z1_exit_marg      P(k_out | size, ce, n_st) shrunk to the MARGINAL
                      P(k_out | size, n_st), at the FITTED k of 20.4
    Z2_exit_interact  Z1 where the cell carries >= 300 rows, the marginal exactly
                      otherwise (a hard gate on the state x composition term)
    Y1_exit_rate      round 8's arm, 1 seed, as a REPRODUCTION CHECK of the
                      reference columns taken from round 8's results JSON

ROUND 8'S COUNTS AND SAMPLER, ROUND 7'S MECHANISM, ROUND 6'S COMPOSITION TABLES,
ROUND 5'S WAVE TABLES, ROUND 4'S HAZARDS AND THE ROUND-3b BASE FITS ARE REUSED,
NOT REFITTED. Round 9 fits only the two exit objects per window, so a difference
between a round-9 arm and Y1 is a difference in the SHRINKAGE PARENT alone.

The shrinkage constant is fitted first, by leave-one-fold-out multinomial
log-likelihood on the 2024 training season over the declared grid (20.4); it
never sees a gate cell, an MAE or a test row.

Usage:
    .venv/Scripts/python.exe scripts/train_rotation_v9.py --workers 6
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
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import rotation_v5 as V5  # noqa: E402
from cbb_sim.models import rotation_v6 as V6  # noqa: E402
from cbb_sim.models import rotation_v7 as V7  # noqa: E402
from cbb_sim.models import rotation_v8 as V8  # noqa: E402
from cbb_sim.models import rotation_v9 as V9  # noqa: E402

import train_rotation_v1 as V1  # noqa: E402
import train_rotation_v4 as V4T  # noqa: E402
import train_rotation_v5 as V5T  # noqa: E402
import train_rotation_v6 as V6T  # noqa: E402
import train_rotation_v7 as V7T  # noqa: E402
import train_rotation_v8 as V8T  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "models" / "rotation"
R9_DIR = OUT_DIR / "round9"
TRAIN_SEASON, TEST_SEASON = 2024, 2025

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_rotation_v9")

ARMS9 = ["Z1_exit_marg", "Z2_exit_interact"]
REFS = ["R2_hier_dirichlet", "K1_cond_class", "X1_exit_class", "Y1_exit_rate"]
SIMPLICITY = {"R2_hier_dirichlet": 1, "K1_cond_class": 9, "X1_exit_class": 11,
              "Y1_exit_rate": 14, "Z1_exit_marg": 16, "Z2_exit_interact": 17}
STATE_CELLS = list(V5T.STATE_CELLS_V5)
NEW_CELLS = list(V5T.NEW_CELLS_V5)

_X9: dict = {}


# ---------------------------------------------------------------------------
def ensure_ctx(args_d: dict) -> dict:
    """round 6's context builder, unchanged (it carries the as-of mpg map the
    player quintile check buckets on)."""
    return V6T.ensure_ctx(args_d)


def exit9_for(tag: str, suffix: str = "") -> V9.ExitFit9:
    key = (tag, suffix)
    if key not in _X9:
        _X9[key] = V9.ExitFit9.from_json(R9_DIR / f"rotation_v9_exit{suffix}_{tag}.json")
    return _X9[key]


def make_arm9(name: str, tag: str, suffix: str = ""):
    """Rounds 5-8's arms come from their OWN artifacts (suffix always ""); only
    the round-9 exit object follows the floor-B suffix."""
    c = V5T._CTX
    base = c["base_fits"][tag]
    ss = V5T.ssi_for(tag)
    wave = V5T.wave_for(tag, "")
    if name in V5.ARMS:
        return V5.ARMS[name](base, wave, ss)
    if name in V6.ARMS:
        return V6.ARMS[name](base, wave, V6T.comp_for(tag, ""), ss)
    if name in V7.ARMS:
        return V7.ARMS[name](base, wave, V6T.comp_for(tag, ""), V7T.exit_for(tag, ""), ss)
    if name in V8.ARMS:
        return V8.ARMS[name](base, wave, V6T.comp_for(tag, ""),
                             V8T.exit8_for(tag, ""), ss)
    return V9.ARMS[name](base, wave, V6T.comp_for(tag, ""), exit9_for(tag, suffix), ss)


# ---------------------------------------------------------------------------
def _init_worker(args_d: dict) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    ensure_ctx(args_d)


def run_job(job: tuple) -> dict:
    """Round 8's `run_job`, byte for byte, with `make_arm9` in place of
    `make_arm8`."""
    kind, name, seeds, suffix = job
    c = V5T._CTX
    t0 = time.time()
    if kind in ("grade", "floorB"):
        nsel = set(c["sel"][: c["args"].noise_games]) if kind == "floorB" else None
        recs, rot, lus = [], [], []
        for tag, gl in c["games_by_window"].items():
            keep = [g for g in gl if nsel is None or g in nsel]
            if not keep:
                continue
            pw = V5T.priors_for("S1", tag)
            arm = make_arm9(name, tag, suffix)
            kbg = {g: c["keys_by_game"][g] for g in keep
                   if all(k in pw for k in c["keys_by_game"][g])}
            r1, r2, r3 = V5T.sim_records_v5(arm, pw, c["scripts"], kbg,
                                            c["rotation_sets"], seeds)
            recs += r1
            rot += r2
            lus += r3
        row = V5T.build_row_v5(name, recs, rot, lus, c["a_row"], c["act_minutes"],
                               c["a_rot"])
        res = {"kind": kind, "arm": name, "suffix": suffix, "row": row,
               "seconds": round(time.time() - t0, 1)}
        if kind == "grade":
            cell = {}
            for r in recs:
                k = r["key"]
                n, d = cell.get(k, (0.0, 0.0))
                cell[k] = (n + r["late_bands"][0][0], d + r["late_bands"][0][1])
            res["late_by_key"] = {f"{a}_{b}": v for (a, b), v in cell.items()}
            res["quintile"] = V6T.quintile_mae(rot, c["a_rot"], c["_mpg"])
        return res
    if kind == "floorA":
        nsel = set(c["sel"][: c["args"].noise_games])
        per_seed = []
        for s in range(c["args"].noise_seeds):
            recs, rot, lus = [], [], []
            for tag, gl in c["games_by_window"].items():
                keep = [g for g in gl if g in nsel]
                if not keep:
                    continue
                pw = V5T.priors_for("S1", tag)
                arm = make_arm9(name, tag, suffix)
                kbg = {g: c["keys_by_game"][g] for g in keep
                       if all(k in pw for k in c["keys_by_game"][g])}
                r1, r2, r3 = V5T.sim_records_v5(arm, pw, c["scripts"], kbg,
                                                c["rotation_sets"], [1000 + s])
                recs += r1
                rot += r2
                lus += r3
            per_seed.append(V5T.build_row_v5(name, recs, rot, lus, c["a_row"],
                                             c["act_minutes"], c["a_rot"]))
        cells = (["minutes_mean", "top5_share", "top8_share", "n_nonzero_mean",
                  "lu_top1", "minutes_mae", "n_lineups_mean"]
                 + STATE_CELLS + NEW_CELLS)
        out = {k: float(np.std([r[k] for r in per_seed], ddof=1)) for k in cells}
        out["n_seeds"] = c["args"].noise_seeds
        out["n_games"] = len(nsel)
        return {"kind": kind, "arm": name, "floor": out,
                "seconds": round(time.time() - t0, 1)}
    raise ValueError(kind)


# ---------------------------------------------------------------------------
def fit_one_window(payload: tuple) -> dict:
    """One S1 window's exit objects, on rounds 6-8's own training rows and the
    shrinkage constant `select_k` fitted on the 2024 training season (20.4)."""
    tag, m_iso, args_d, fit_seed, suffix, k_sel, k_scores = payload
    c = ensure_ctx(args_d)
    args = argparse.Namespace(**args_d)
    m = pd.Timestamp(m_iso)
    train24, test = c["train24"], c["test"]
    if m == c["months"][0]:
        tp_w, feats_w, ev_w = train24["tp"], train24["feats"], train24["fouls"]
        max_train = f"{TRAIN_SEASON}-04-30"
        hz_name = "rotation_v4_sub_static.json"
    else:
        mask = pd.to_datetime(test["tp"]["game_date"]) < m
        gids = set(test["tp"].loc[mask, "game_id"].astype("int64"))
        tp_w = pd.concat([train24["tp"], test["tp"][test["tp"]["game_id"].isin(gids)]],
                         ignore_index=True)
        feats_w = pd.concat([train24["feats"],
                             test["feats"][test["feats"]["game_id"].isin(gids)]],
                            ignore_index=True)
        ev_w = pd.concat([train24["fouls"],
                          test["fouls"][test["fouls"]["game_id"].isin(gids)]],
                         ignore_index=True)
        max_train = str((pd.Timestamp(m) - pd.Timedelta(days=1)).date())
        hz_name = f"rotation_v4_sub_S1_{tag}.json"
    base = c["base_fits"][tag]
    t0 = time.time()
    counts = V9.build_exit8_training(tp_w, feats_w, base,
                                     sorted(tp_w["game_id"].unique()), fouls=ev_w,
                                     max_team_games=args.wave_team_games, seed=fit_seed)
    xf = V9.fit_exit9(counts, k=k_sel, wave_source=f"rotation_v5_wave_{tag}.json",
                      comp_source=f"rotation_v6_comp_{tag}.json",
                      hazard_source=hz_name,
                      k_selected_by="leave-one-fold-out multinomial log-likelihood, "
                                    "2024 training season, grid "
                                    + str(list(V9.K_GRID)),
                      k_grid_scores=k_scores)
    xf.notes.update({"window": tag, "max_train_date": max_train, "fit_seed": fit_seed})
    R9_DIR.mkdir(parents=True, exist_ok=True)
    p = R9_DIR / f"rotation_v9_exit{suffix}_{tag}.json"
    xf.to_json(p)
    ncell = np.asarray(xf.n_cell)
    nmarg = np.asarray(xf.n_marg)
    return {"refit_date": str(m.date()), "path": p.name, "max_train_date": max_train,
            "hazards": hz_name, "wave": f"rotation_v5_wave_{tag}.json",
            "comp": f"rotation_v6_comp_{tag}.json",
            "exit_parent": "the composition marginal P(k_out | size, n_st) (20.2)",
            "k_shrink": float(xf.k_shrink),
            "n_waves": xf.n_waves, "n_pairs": xf.n_pairs,
            "team_games": counts["team_games"],
            # the cells the pre-registration asks to be published (20.3)
            "p_kout1_size1_marginal_by_nst": np.round(xf.M1[0][:, 1], 4).tolist(),
            "p_kout1_size1_Z1_by_nst": np.round(xf.Z1[0].mean(axis=0)[:, 1], 4).tolist(),
            "p_kout1_size1_Z2_by_nst": np.round(xf.Z2[0].mean(axis=0)[:, 1], 4).tolist(),
            "n_size1_by_nst": nmarg[0].tolist(),
            "z_size1_cells_under300": int((ncell[0] < V9.MIN_CELL).sum()),
            "z_size1_cells_total": int(ncell[0].size),
            "seconds": round(time.time() - t0, 1)}


def fit_windows(args_d: dict, fit_seed: int, suffix: str, workers: int,
                k_sel: float, k_scores: dict) -> list:
    c = V5T._CTX
    payloads = [(m.strftime("%Y%m"), m.isoformat(), args_d, fit_seed, suffix,
                 k_sel, k_scores)
                for m in c["months"] if c["games_by_window"][m.strftime("%Y%m")]]
    ents = []
    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads)),
                                 initializer=_init_worker,
                                 initargs=(args_d,)) as ex:
            futs = [ex.submit(fit_one_window, p) for p in payloads]
            for f in as_completed(futs):
                e = f.result()
                ents.append(e)
                log.info("  exit window %s: %d waves, %d used (%.0fs)", e["path"],
                         e["n_waves"], e["n_pairs"], e["seconds"])
    else:
        for p in payloads:
            e = fit_one_window(p)
            ents.append(e)
            log.info("  exit window %s: %d waves (%.0fs)", e["path"], e["n_waves"],
                     e["seconds"])
    ents.sort(key=lambda e: e["refit_date"])
    man = {"model": "rotation", "scheme": "S1", "fold": "F1", "season": TEST_SEASON,
           "arm": "round9_exit" + suffix, "k_shrink": float(k_sel),
           "k_selection": k_scores, "artifacts": ents}
    (R9_DIR / f"rotation_v9_manifest{suffix}.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8")
    return ents


# ---------------------------------------------------------------------------
def _fold_counts(payload: tuple) -> dict:
    """One leave-one-fold-out fold's exit counts on the 2024 TRAINING season."""
    i, n_folds, args_d, fit_seed = payload
    c = ensure_ctx(args_d)
    args = argparse.Namespace(**args_d)
    tr = c["train24"]
    gids = np.sort(tr["tp"]["game_id"].unique())
    rs = np.random.RandomState(fit_seed)
    fold = rs.randint(0, n_folds, size=len(gids))
    keep = set(int(g) for g in gids[fold == i])
    tp_i = tr["tp"][tr["tp"]["game_id"].isin(keep)]
    ft_i = tr["feats"][tr["feats"]["game_id"].isin(keep)]
    ev_i = tr["fouls"][tr["fouls"]["game_id"].isin(keep)]
    cnt = V9.build_exit8_training(
        tp_i, ft_i, c["base_fits"][c["months"][0].strftime("%Y%m")],
        sorted(keep), fouls=ev_i,
        max_team_games=max(1, args.wave_team_games // n_folds), seed=fit_seed)
    return {"c1": cnt["c1"], "cy1": cnt["cy1"], "n_used": cnt["n_used"]}


def fit_k(args_d: dict, fit_seed: int, workers: int, n_folds: int = 5) -> dict:
    """20.4: the shrinkage constant, fitted by leave-one-fold-out multinomial
    log-likelihood on the 2024 training season over the declared grid. No gate
    cell, no MAE and no 2025 row is visible to this."""
    t0 = time.time()
    payloads = [(i, n_folds, args_d, fit_seed) for i in range(n_folds)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, n_folds),
                                 initializer=_init_worker,
                                 initargs=(args_d,)) as ex:
            folds = [f.result() for f in [ex.submit(_fold_counts, p)
                                          for p in payloads]]
    else:
        folds = [_fold_counts(p) for p in payloads]
    sel = V9.select_k(folds)
    sel["fold_rows"] = [int(f["n_used"]) for f in folds]
    sel["seconds"] = round(time.time() - t0, 1)
    log.info("fitted k = %g by %s; scores %s", sel["k"], sel["criterion"],
             json.dumps(sel["scores"]))
    return sel


# ---------------------------------------------------------------------------
def decide_v9(rows: dict, verdicts: dict, noise: dict, quint: dict, slope: dict,
              ref_mae: dict, ref_floor: dict, r6_quint: dict) -> dict:
    """Round 8's rule, byte for byte, with the arm grid of 20.5 (the quintile
    condition is still applied against K1 and W4, 20.9)."""
    out = {"table": [], "winner": None, "why": ""}
    eligible = []
    for n, row in rows.items():
        v = verdicts[n]
        state_ok = all(v[cc] for cc in STATE_CELLS)
        new_ok = all(v[cc] for cc in NEW_CELLS)
        fl_r2 = max(noise.get(n, {}).get("minutes_mae", 0.0),
                    ref_floor.get("R2_hier_dirichlet", 0.0))
        fl_k1 = max(noise.get(n, {}).get("minutes_mae", 0.0),
                    ref_floor.get("K1_cond_class", 0.0))
        beats_r2 = (ref_mae["R2_hier_dirichlet"] - row["minutes_mae"]) > fl_r2
        beats_k1 = (ref_mae["K1_cond_class"] - row["minutes_mae"]) > fl_k1
        q_ok, q_worse = True, []
        if n in quint:
            for ref_name, qref in r6_quint.items():
                for i in range(5):
                    a = quint[n]["q_mae"][i]
                    b = qref["q_mae"][i]
                    if a == a and b == b and (a - b) > fl_k1:
                        q_ok = False
                        q_worse.append(f"Q{i + 1} vs {ref_name}")
        sl = slope.get(n + "_slope")
        act = slope.get("ACTUAL_slope")
        ratio = float(sl / act) if sl is not None and act else float("nan")
        d8 = bool(ratio == ratio and 0.8 <= ratio <= 1.2)
        out["table"].append({
            "arm": n, "simplicity": SIMPLICITY.get(n, 99),
            "state_cells_passed": sum(1 for cc in STATE_CELLS if v[cc]),
            "state_cells": len(STATE_CELLS),
            "new_cells_passed": sum(1 for cc in NEW_CELLS if v[cc]),
            "eligible": bool(state_ok and new_ok),
            "g8_cells_passed": sum(1 for cc in V1.G8_CELLS if v[cc]),
            "minutes_mae": row["minutes_mae"],
            "mae_vs_R2": float(ref_mae["R2_hier_dirichlet"] - row["minutes_mae"]),
            "mae_vs_K1": float(ref_mae["K1_cond_class"] - row["minutes_mae"]),
            "mae_vs_X1": float(ref_mae["X1_exit_class"] - row["minutes_mae"]),
            "mae_vs_Y1": float(ref_mae["Y1_exit_rate"] - row["minutes_mae"]),
            "mae_floor_r2": float(fl_r2), "mae_floor_k1": float(fl_k1),
            "beats_R2_beyond_floor": bool(beats_r2),
            "beats_K1_beyond_floor": bool(beats_k1),
            "quintile_ok": bool(q_ok), "quintiles_worse": q_worse,
            "d8_slope_ratio": ratio, "d8_ok": d8,
            "sub_rate_per_boundary": row["sub_rate_per_boundary"],
            "distinct_lineups_per_game": row["distinct_lineups_per_game"],
        })
        if (state_ok and new_ok and beats_r2 and beats_k1 and q_ok and d8
                and n in ARMS9):
            eligible.append((SIMPLICITY.get(n, 99), n))
    out["table"].sort(key=lambda r: r["simplicity"])
    if eligible:
        eligible.sort()
        out["winner"] = eligible[0][1]
        out["why"] = ("simplest arm passing all eight state cells, both round-5 "
                      "cells, both MAE comparisons beyond the floor, every player "
                      "quintile against BOTH K1 and W4, and the Decision 8 band; "
                      "condition 6 (the Decision 10 freeze) is reported separately")
    else:
        out["why"] = "no arm passes every pre-registered condition; adopt nothing"
    return out


# ---------------------------------------------------------------------------
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
    ap.add_argument("--floor-b-arm", type=str, default="Z1_exit_marg")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-floor-a", action="store_true")
    ap.add_argument("--skip-fit", action="store_true")
    ap.add_argument("--mode", default="bakeoff", choices=["bakeoff", "floorb"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", type=str, default="F1-round9")
    args = ap.parse_args()
    if args.smoke:
        args.test_games, args.seeds = 60, 1
        args.noise_seeds, args.noise_games = 3, 30
        args.wave_team_games = 400
        args.workers = 2
    t0 = time.time()
    assert_not_sealed([TRAIN_SEASON, TEST_SEASON], context="rotation round 9")
    args_d = vars(args)
    c = ensure_ctx(args_d)
    log.info("test universe: %d of %d eligible games; windows %s",
             len(c["sel"]), c["eligible"], {t: len(v) for t, v in
                                            c["games_by_window"].items()})

    if args.mode == "floorb":
        log.info("floor B: exit refit under fit seed %d, arm %s",
                 args.floor_fit_seed, args.floor_b_arm)
        ksel_fb = json.loads((R9_DIR / "rotation_v9_kselect.json").read_text(
            encoding="utf-8"))
        ents2 = fit_windows(args_d, args.floor_fit_seed, "_seed2", args.workers,
                            ksel_fb["k"], ksel_fb)
        fb_jobs = [("floorB", args.floor_b_arm, [0], ""),
                   ("floorB", args.floor_b_arm, [args.floor_seed], "_seed2")]
        with ProcessPoolExecutor(max_workers=2, initializer=_init_worker,
                                 initargs=(args_d,)) as ex:
            fb = [f.result() for f in [ex.submit(run_job, j) for j in fb_jobs]]
        nsel = set(c["sel"][: args.noise_games])
        nkeys = [k for tag, gl in c["games_by_window"].items() for g in gl
                 if g in nsel for k in c["keys_by_game"][g]]
        an, arot_n, _ = V4T.actual_records_v4(c["test"], nkeys, c["rotation_sets"])
        fbr = V1.build_row("ACTUAL", an, arot_n, {}, np.array([]))
        fbr.update(V4T.agg_extra(an))
        out = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "arm": args.floor_b_arm, "fit_seed_2": args.floor_fit_seed,
               "sim_seed_2": args.floor_seed, "n_games": args.noise_games,
               "manifest_seed2": ents2,
               "seed1": fb[0]["row"], "seed2": fb[1]["row"], "ACTUAL": fbr}
        stem = "rotation_F1_round9_floorB_" + args.floor_b_arm.split("_")[0]
        (OUT_DIR / f"{stem}.json").write_text(
            json.dumps(out, indent=2, default=V1._json_default), encoding="utf-8")
        log.info("floor B written in %.1f min -> %s", (time.time() - t0) / 60, stem)
        return

    if args.skip_fit:
        ksel = json.loads((R9_DIR / "rotation_v9_kselect.json").read_text(
            encoding="utf-8"))
        entries = json.loads((R9_DIR / "rotation_v9_manifest.json").read_text(
            encoding="utf-8"))["artifacts"]
    else:
        ksel = fit_k(args_d, args.fit_seed, args.workers, args.k_folds)
        R9_DIR.mkdir(parents=True, exist_ok=True)
        (R9_DIR / "rotation_v9_kselect.json").write_text(
            json.dumps(ksel, indent=2, default=V1._json_default), encoding="utf-8")
        entries = fit_windows(args_d, args.fit_seed, "", args.workers,
                              ksel["k"], ksel)

    jobs = [("grade", n, list(range(args.seeds)), "") for n in ARMS9]
    jobs.append(("grade", "Y1_exit_rate", [0], ""))
    if not args.skip_floor_a:
        jobs += [("floorA", n, None, "") for n in ARMS9]

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(args_d,)) as ex:
            futs = {ex.submit(run_job, j): j for j in jobs}
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                log.info("done %s %s in %.1fs", r["kind"], r["arm"], r["seconds"])
    else:
        for j in jobs:
            r = run_job(j)
            results.append(r)
            log.info("done %s %s in %.1fs", r["kind"], r["arm"], r["seconds"])

    rows8 = {r["arm"]: r["row"] for r in results if r["kind"] == "grade"}
    late_by_key = {r["arm"]: r["late_by_key"] for r in results if r["kind"] == "grade"}
    quint = {r["arm"]: r["quintile"] for r in results if r["kind"] == "grade"}
    noise_a = {r["arm"]: r["floor"] for r in results if r["kind"] == "floorA"}

    # ---- reference columns from rounds 6 and 7, and the X1 reproduction ----
    r6 = json.loads((OUT_DIR / "rotation_F1_round6_results.json").read_text(
        encoding="utf-8"))
    r7 = json.loads((OUT_DIR / "rotation_F1_round7_results.json").read_text(
        encoding="utf-8"))
    r8 = json.loads((OUT_DIR / "rotation_F1_round8_results.json").read_text(
        encoding="utf-8"))
    ref = {"R2_hier_dirichlet": dict(r6["arms_S1"]["R2_hier_dirichlet"]),
           "K1_cond_class": dict(r6["arms_S1"]["K1_cond_class"]),
           "X1_exit_class": dict(r7["arms_S1"]["X1_exit_class"]),
           "Y1_exit_rate": dict(r8["arms_S1"]["Y1_exit_rate"])}
    ref_floors = {"K1_cond_class": r6["noise_A"].get("K1_cond_class", {}),
                  "R2_hier_dirichlet": r6["noise_A_round5"].get("R2_hier_dirichlet", {}),
                  "X1_exit_class": r7["noise_A"].get("X1_exit_class", {}),
                  "Y1_exit_rate": r8["noise_A"].get("Y1_exit_rate", {})}
    repro = {"cells": {}, "ok": True, "arm": "Y1_exit_rate"}
    y1r = rows8.pop("Y1_exit_rate", None)
    y1_quint_repro = quint.pop("Y1_exit_rate", None)
    late_by_key.pop("Y1_exit_rate", None)
    if y1r is not None:
        for cc in STATE_CELLS + ["minutes_mae"] + NEW_CELLS:
            a = ref["Y1_exit_rate"].get(cc, np.nan)
            b = y1r.get(cc, np.nan)
            fl = ref_floors["Y1_exit_rate"].get(cc, np.nan)
            repro["cells"][cc] = {"round8_3seed": a, "round9_1seed": b,
                                  "delta": float(b - a) if a == a else None,
                                  "floorA": fl}
            if cc in STATE_CELLS and a == a and fl == fl and abs(b - a) > fl:
                repro["ok"] = False
        repro["y1_reproduction_row"] = y1r
        repro["y1_reproduction_quintile"] = y1_quint_repro

    # ---- the as-of starter-set benchmark (report only, 18.7) --------------
    asof_st = {k: set(int(x) for x in v.pids[v.starters()[:5]])
               for k, v in c["priors_static"].items()}
    keys = [k for g in c["sel"] for k in c["keys_by_game"][g]]
    b_recs, b_rot, (diag, overlaps) = V4T.actual_records_v4(
        c["test"], keys, c["rotation_sets"], asof_starters=asof_st)
    bench = V1.build_row("ACTUAL_asof_starters", diag, b_rot, {}, np.array([]))
    bench.update(V4T.agg_extra(diag))
    bench["starter_overlap_of_5"] = float(np.mean(overlaps)) if overlaps else np.nan

    tol = {}
    for cc in NEW_CELLS:
        fl = max([noise_a.get(n, {}).get(cc, 0.0) for n in noise_a] or [0.0])
        tol[cc] = float(max(V5T.FIXED_TOL_V5[cc], 3.0 * fl))
        tol[cc + "_floor3x"] = float(3.0 * fl)
    all_rows = dict(ref)
    all_rows.update(rows8)
    verdicts = {n: V5T.verdict_v5(r, c["a_row"], tol) for n, r in all_rows.items()}

    prior_share = {k: float(v.share[v.starters()].sum())
                   for k, v in c["priors_static"].items()}
    slope = V5T.slope_check(c, dict(late_by_key), prior_share)
    for n, src in (("R2_hier_dirichlet", r6), ("K1_cond_class", r6),
                   ("X1_exit_class", r7), ("Y1_exit_rate", r8)):
        if n in src.get("slope", {}):
            slope[n] = src["slope"][n]
            slope[n + "_slope"] = src["slope"].get(n + "_slope")
            slope[n + "_q5_q1"] = src["slope"].get(n + "_q5_q1")

    ref_mae = {n: ref[n]["minutes_mae"] for n in REFS}
    ref_floor = {n: float(ref_floors[n].get("minutes_mae", 0.0)) for n in REFS}
    r6_quint = {"K1": r6["quintile_mae"]["K1_cond_class"],
                "W4": r6["quintile_mae"]["W4_wave_rank_draw"]}
    decision = decide_v9(all_rows, verdicts, noise_a, quint, slope, ref_mae,
                         ref_floor, r6_quint)

    # ---- per-team evidence -------------------------------------------------
    per_team = {}
    act_cell = {}
    for r in c["a_recs"]:
        k = r["key"]
        n, d = act_cell.get(k[1], (0.0, 0.0))
        act_cell[k[1]] = (n + r["late_bands"][0][0], d + r["late_bands"][0][1])
    for name, cell in late_by_key.items():
        agg = {}
        for kk, (n, d) in cell.items():
            tid = int(kk.split("_")[1])
            a, b = agg.get(tid, (0.0, 0.0))
            agg[tid] = (a + n, b + d)
        tids = [t for t in agg if agg[t][1] >= 300 and act_cell.get(t, (0, 0))[1] >= 300]
        sim = np.array([agg[t][0] / agg[t][1] for t in tids])
        act = np.array([act_cell[t][0] / act_cell[t][1] for t in tids])
        per_team[name] = {
            "n_teams_powered": len(tids),
            "n_teams_underpowered": len(agg) - len(tids),
            "sim_mean": float(sim.mean()) if len(sim) else np.nan,
            "act_mean": float(act.mean()) if len(act) else np.nan,
            "sim_sd": float(sim.std(ddof=1)) if len(sim) > 1 else np.nan,
            "act_sd": float(act.std(ddof=1)) if len(act) > 1 else np.nan,
            "corr": float(np.corrcoef(sim, act)[0, 1]) if len(sim) > 2 else np.nan,
            "mean_abs_dev": float(np.abs(sim - act).mean()) if len(sim) else np.nan,
        }

    a_out = {k: v for k, v in c["a_row"].items() if not k.startswith("_")}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fold": args.tag, "round": 9, "scheme": "S1",
        "prereg_commit": "3525996",
        "k_selection": ksel,
        "config": vars(args),
        "windows": {t: len(g) for t, g in c["games_by_window"].items()},
        "manifest": entries,
        "n_eligible_games": c["eligible"], "n_simulated_games": len(c["sel"]),
        "actual": a_out,
        "asof_starter_benchmark": bench,
        "arms_S1": all_rows,
        "reference_from_rounds_6_7_8": REFS,
        "y1_reproduction_check": repro,
        "verdicts_S1": verdicts, "tolerances_new_cells": tol,
        "noise_A": noise_a, "noise_A_refs": ref_floors,
        "quintile_mae": quint, "quintile_mae_round6": r6_quint,
        "quintile_mae_round7_X1": r7["quintile_mae"].get("X1_exit_class"),
        "quintile_mae_round8_Y1": r8["quintile_mae"].get("Y1_exit_rate"),
        "per_team": per_team,
        "slope": slope, "decision": decision,
    }
    stem = "rotation_F1_round9" + ("_SMOKE" if args.smoke else "")
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
