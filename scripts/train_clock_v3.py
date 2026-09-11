"""train_clock_v3.py -- ROUND 3 of the L5 clock bake-off.

Runs exactly the grid pre-registered in `docs/models/clock/experiments.md`
section 8, which was committed (0c7cc21) before this file existed. Rounds 1 and
2 (`scripts/train_clock_v1.py`, `scripts/train_clock_v2.py`) are never
overwritten and neither are their artifacts; everything here takes a `v3_`
prefix in `data/processed/models/clock/`.

What round 3 changes and nothing else (L20): horn-ending possessions are
RIGHT-CENSORED, the model targets the duration the offence INTENDED, and the
engine truncates at the horn. The gate universe moves to CLOCK-COMPLETE games,
which the audit measured as a grading-truth fix worth 0.00 to -0.20 possessions
per team-game -- it makes the round-2 arms look slightly worse, not better.

    arms   A1 empirical_km3          KM + corrected flag + the 8.3 tail rule
           A2 empirical_km3_srfloor  A1, clock bucket floored at 45-59 s
           A3 gamma_aft              censored MLE, corrected flag
           A4 lognormal_aft          censored MLE, corrected flag
           A5 hazard3                discrete-time hazard, corrected flag
           A6 xgb_aft                XGBoost survival:aft (censored TREE loss)
           B1 lgbm_quantile_r2       ROUND-2 REFERENCE, unchanged spec
           B2 empirical_r2           round-2 empirical (old flag) -- control
           B3 gamma_r2               round-2 gamma (old flag) -- control

    gates  G1-CC (emergent, clock-complete games), PIT (truncated law),
           end-of-half on clock-complete halves

Usage (threads capped at 4; four other workers share this machine):
    OMP_NUM_THREADS=4 .venv/Scripts/python.exe scripts/train_clock_v3.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cbb_sim.models import clock as ck  # noqa: E402
from cbb_sim.models import clock_v3 as c3  # noqa: E402

OUT = Path("data/processed/models/clock")
DESIGN_V2 = OUT / "design_v2.parquet"          # READ-ONLY: written by round 2
DESIGN_V3 = OUT / "v3_design.parquet"          # only if round 2's is absent
ALL_SEASONS = [2022, 2023, 2024, 2025]
BASE_SEED = 20260910
NOISE_SEED_OFFSET = 1000
EOH_FLOOR_SEEDS = (20260910, 20260911, 20260912, 20260913, 20260914)
EOH_FLOOR_K = 2.0
N_JOBS = 4


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
def load_design() -> tuple[pd.DataFrame, dict]:
    if DESIGN_V2.exists():
        log(f"reading round-2 design (read-only) {DESIGN_V2}")
        d = pd.read_parquet(DESIGN_V2)
        diag = {"design_source": str(DESIGN_V2)}
    elif DESIGN_V3.exists():
        d = pd.read_parquet(DESIGN_V3)
        diag = {"design_source": str(DESIGN_V3)}
    else:
        log("building design from scratch")
        d, diag = ck.build_design(ALL_SEASONS)
        d.to_parquet(DESIGN_V3, index=False)
        diag["design_source"] = str(DESIGN_V3)
    d, cdiag = c3.attach_horn_censoring(d, ALL_SEASONS)
    diag.update(cdiag)
    return d, diag


def block_bootstrap_se_rows(game_ids: np.ndarray, rows: np.ndarray, n_rep: int = 200,
                            seed: int = 4242) -> float:
    """Game-block bootstrap SE of the mean of a per-row metric. The resampling
    unit is the GAME, as in rounds 1 and 2."""
    ok = np.isfinite(rows)
    games = np.asarray(game_ids)[ok]
    vals = rows[ok]
    order = np.argsort(games, kind="stable")
    g, v = games[order], vals[order]
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    ends = np.concatenate([starts[1:], [len(g)]])
    sums = np.add.reduceat(v, starts)
    ns = (ends - starts).astype("float64")
    rng = np.random.default_rng(seed)
    n_g = len(starts)
    out = np.empty(n_rep)
    for r in range(n_rep):
        pick = rng.integers(0, n_g, n_g)
        out[r] = sums[pick].sum() / ns[pick].sum()
    return float(out.std(ddof=1))


def fit_and_score(arm_name: str, design: pd.DataFrame, fold: str, seed: int,
                  hc: pd.DataFrame, do_chain: bool, extra: dict) -> dict:
    """One blind path: same fit dispatch, same scorer, same chain, every arm."""
    c3.set_flag_inplace(design, c3.ARM_FLAG[arm_name])
    tr, te = ck.fold_slices(design, fold)
    t0 = time.time()
    arm = c3.fit_arm_v3(arm_name, tr, seed=seed, n_jobs=N_JOBS, **extra)
    fit_s = time.time() - t0
    log(f"  {fold} {arm_name}: fit {fit_s:.1f}s")

    sc = c3.score_arm_v3(arm, te, pit_seed=BASE_SEED)
    row = {k: v for k, v in sc.items() if not k.startswith("_")}
    row.update(fold=fold, arm=arm_name, flag=c3.ARM_FLAG[arm_name],
               features=c3.ARM_FEATURES_V3[arm_name], fit_seconds=round(fit_s, 1),
               n_train=int(len(tr)), n_train_censored=int(tr["censored"].sum()))

    pit_cells = c3.pit_by_cell_v3(te, sc["_pit_rows"])
    row["pit_leak_failures"] = int(pit_cells["leak_sized"].sum())
    row["pit_powered"] = int(pit_cells["powered"].sum())
    row["pit_underpowered"] = int((~pit_cells["powered"]).sum())
    pw = pit_cells[pit_cells["powered"]]
    row["pit_worst_D"] = float(pw["ks_D"].max()) if len(pw) else float("nan")
    row["pit_worst_cell"] = str(pw["cell"].iloc[0]) if len(pw) else ""
    row["pit_pass"] = bool(row["pit_leak_failures"] == 0)

    seg = c3.segment_table(te, sc["_crps_trunc_rows"], sc["_loglik_rows"])
    row["_segments"] = seg
    row["_pit_cells"] = pit_cells
    row["_crps_rows"] = sc["_crps_trunc_rows"]
    row["_arm"] = arm
    row["_game_ids"] = te["game_id"].to_numpy()
    row["_bands"] = c3.pred_mean_by_band(arm, te)

    if do_chain:
        t0 = time.time()
        res = ck.chain_halves(arm, te, seed=seed)
        ah = c3.actual_end_of_half_cc(te, hc)
        cc = c3.emergent_cc(res, ah)
        allg = ck.emergent_report(res, ah)
        row.update({f"cc_{k}": v for k, v in cc.items() if k != "months"})
        row["all_mean_delta"] = allg["mean_delta"]
        row["all_sd_delta"] = allg["sd_delta"]
        row["all_sim_mean"] = allg["sim_mean"]
        row["all_actual_mean"] = allg["actual_mean"]
        row["chain_seconds"] = round(time.time() - t0, 1)
        row["chain_wrapped_halves"] = res.wrapped_halves
        row["_cc_months"] = pd.DataFrame(cc["months"])
        row["_responsiveness"] = c3.responsiveness(res, te)
        row["_chain"] = res
        row["_actual_half"] = ah
        log(f"  {fold} {arm_name}: CRPS_trunc {row['crps_trunc']:.4f} "
            f"G1-CC {cc['mean_delta']:+.3f}/{cc['sd_delta']:+.3f} "
            f"EOH-CC {cc['eoh_share_gap']:+.4f}/{cc['eoh_duration_gap']:+.3f}")
    return row


def strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="F2 only, no F1 stage")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    design, diag = load_design()
    hc = c3.load_clock_complete()
    log(f"design {len(design):,} rows; censored old {diag['censored_old_pct']}% "
        f"-> horn {diag['censored_horn_pct']}%")
    (OUT / "v3_build_diagnostics.json").write_text(json.dumps(diag, indent=2, default=str),
                                                   encoding="utf-8")

    f1_stage: dict = {}

    # =====================================================================
    # F1 stage: the only choices the pre-registration allows off F1
    # =====================================================================
    if not args.quick:
        log("F1 stage: xgb_aft error distribution (F1 ONLY)")
        c3.set_flag_inplace(design, "horn")
        tr1, te1 = ck.fold_slices(design, "F1")
        dist_rows = []
        for dist in c3.AFT_DISTS:
            t0 = time.time()
            a = c3.fit_xgb_aft(tr1, "R2_tree", dist=dist, seed=BASE_SEED, nthread=N_JOBS)
            s = c3.score_arm_v3(a, te1, pit_seed=BASE_SEED)
            dist_rows.append({"dist": dist, "scale": a.scale,
                              "f1_censored_loglik": s["censored_loglik"],
                              "f1_crps_trunc": s["crps_trunc"],
                              "fit_seconds": round(time.time() - t0, 1)})
            log(f"  xgb_aft {dist}: scale {a.scale:.4f} "
                f"loglik {s['censored_loglik']:.5f} CRPS_t {s['crps_trunc']:.4f}")
            del a, s
        dist_df = pd.DataFrame(dist_rows)
        chosen_dist = str(dist_df.sort_values("f1_censored_loglik", ascending=False)["dist"].iloc[0])
        f1_stage["xgb_dist_search"] = dist_rows
        f1_stage["xgb_dist_chosen"] = chosen_dist
        log(f"  chosen AFT distribution (F1 only): {chosen_dist}")

        log("F1 stage: end-of-half noise floor (5-seed re-chain + block bootstrap)")
        ref = c3.fit_arm_v3("empirical_km3", tr1, seed=BASE_SEED)
        ah1 = c3.actual_end_of_half_cc(te1, hc)
        shares, durs = [], []
        for s in EOH_FLOOR_SEEDS:
            r = ck.chain_halves(ref, te1, seed=s)
            sh, du = c3.eoh_cc_stats(r, ah1)
            shares.append(sh)
            durs.append(du)
            del r
        sim_sd_share, sim_sd_dur = float(np.std(shares, ddof=1)), float(np.std(durs, ddof=1))
        cc_halves = ah1[ah1["clock_complete"]]
        a_se_share, a_se_dur = ck.eoh_actual_block_bootstrap_se(cc_halves)
        floor = {"share": EOH_FLOOR_K * max(sim_sd_share, a_se_share),
                 "duration": EOH_FLOOR_K * max(sim_sd_dur, a_se_dur),
                 "k": EOH_FLOOR_K, "sim_seed_sd_share": sim_sd_share,
                 "sim_seed_sd_duration": sim_sd_dur,
                 "actual_block_bootstrap_se_share": a_se_share,
                 "actual_block_bootstrap_se_duration": a_se_dur,
                 "seeds": list(EOH_FLOOR_SEEDS), "n_cc_halves": int(len(cc_halves))}
        f1_stage["end_of_half_floor"] = floor
        log(f"  EOH floor: share +/-{floor['share']:.5f}, duration +/-{floor['duration']:.4f}s")
        del ref, tr1, te1
        (OUT / "v3_f1_stage.json").write_text(json.dumps(f1_stage, indent=2, default=str),
                                              encoding="utf-8")

        log("F1 robustness grid (no chain)")
        f1_rows = []
        for arm in c3.ARMS_V3:
            extra = {"dist": chosen_dist} if arm == "xgb_aft" else {}
            r = fit_and_score(arm, design, "F1", BASE_SEED, hc, do_chain=False, extra=extra)
            f1_rows.append(strip(r))
            del r
        pd.DataFrame(f1_rows).to_csv(OUT / "v3_grid_F1.csv", index=False)
    else:
        chosen_dist = "normal"
        floor = {"share": 0.006338, "duration": 0.19424, "k": EOH_FLOOR_K,
                 "note": "quick mode: round-2 floor reused"}

    # =====================================================================
    # F2: the selection fold
    # =====================================================================
    log("F2 selection grid")
    f2: list[dict] = []
    for arm in c3.ARMS_V3:
        extra = {"dist": chosen_dist} if arm == "xgb_aft" else {}
        f2.append(fit_and_score(arm, design, "F2", BASE_SEED, hc, do_chain=True, extra=extra))

    # ---- noise floor -----------------------------------------------------
    log("noise floor")
    floors = {}
    for r in f2:
        arm = r["arm"]
        if arm in c3.STOCHASTIC_ARMS:
            extra = {"dist": chosen_dist} if arm == "xgb_aft" else {}
            r2 = fit_and_score(arm, design, "F2", BASE_SEED + NOISE_SEED_OFFSET, hc,
                               do_chain=False, extra=extra)
            floors[arm] = {"kind": "seed_refit",
                           "value": abs(r2["crps_trunc"] - r["crps_trunc"])}
            del r2
        else:
            floors[arm] = {"kind": "block_bootstrap_se",
                           "value": block_bootstrap_se_rows(r["_game_ids"], r["_crps_rows"])}
        log(f"  {arm}: floor {floors[arm]['value']:.6f} ({floors[arm]['kind']})")
    crps_floor = max(v["value"] for v in floors.values())

    # ---- gates and verdict ----------------------------------------------
    for r in f2:
        r["eoh_share_pass"] = bool(abs(r["cc_eoh_share_gap"]) <= floor["share"])
        r["eoh_dur_pass"] = bool(abs(r["cc_eoh_duration_gap"]) <= floor["duration"])
        r["eoh_pass"] = bool(r["eoh_share_pass"] and r["eoh_dur_pass"])
        r["g1_pass"] = bool(r["cc_g1_pass"])
        r["all_gates_pass"] = bool(r["g1_pass"] and r["pit_pass"] and r["eoh_pass"])

    grid = pd.DataFrame([strip(r) for r in f2]).sort_values("crps_trunc")
    grid.to_csv(OUT / "v3_grid_F2.csv", index=False)

    elig = [r for r in f2 if r["all_gates_pass"]]
    verdict: dict = {
        "round": 3, "seed": BASE_SEED, "selection_fold": "F2",
        "floor": crps_floor, "floors_by_arm": floors,
        "end_of_half_floor": floor, "xgb_dist_chosen": chosen_dist,
        "n_arms": len(f2), "n_pass_g1": int(sum(r["g1_pass"] for r in f2)),
        "n_pit_clean": int(sum(r["pit_pass"] for r in f2)),
        "n_pass_eoh": int(sum(r["eoh_pass"] for r in f2)),
        "n_eligible": len(elig),
        "best_crps_arm": str(grid["arm"].iloc[0]),
        "best_crps": float(grid["crps_trunc"].iloc[0]),
        "created_at": started, "finished_at": datetime.now(UTC).isoformat(),
    }
    if elig:
        elig_sorted = sorted(elig, key=lambda r: r["crps_trunc"])
        best = elig_sorted[0]
        # tie-break inside the floor -> simpler arm
        tied = [r for r in elig_sorted if r["crps_trunc"] - best["crps_trunc"] <= crps_floor]
        win = sorted(tied, key=lambda r: (c3.SIMPLICITY_RANK_V3[r["arm"]], r["crps_trunc"]))[0]
        non_tree = [r for r in elig_sorted if r["arm"] not in c3.TREE_ARMS_V3]
        if win["arm"] in c3.TREE_ARMS_V3 and non_tree:
            if non_tree[0]["crps_trunc"] - win["crps_trunc"] <= crps_floor:
                win = non_tree[0]
                verdict["tree_rule_applied"] = True
        verdict["winner"] = win["arm"]
        verdict["verdict"] = "ADOPTED"
        verdict["tied_within_floor"] = [r["arm"] for r in tied]
    else:
        win = min(f2, key=lambda r: r["crps_trunc"])
        verdict["winner"] = None
        verdict["verdict"] = "NO ARM ADOPTED"
        verdict["reason"] = ("no arm passed all three pre-registered round-3 gates on F2 "
                             "(emergent G1 on clock-complete games, PIT, end-of-half)")

    # ---- artifacts -------------------------------------------------------
    for r in f2:
        tag = r["arm"]
        r["_pit_cells"].to_csv(OUT / f"v3_pit_cells_F2_{tag}.csv", index=False)
        r["_segments"].to_csv(OUT / f"v3_segments_F2_{tag}.csv", index=False)
        r["_cc_months"].to_csv(OUT / f"v3_cc_months_F2_{tag}.csv", index=False)
        r["_responsiveness"].to_csv(OUT / f"v3_responsiveness_F2_{tag}.csv", index=False)
        r["_bands"].to_csv(OUT / f"v3_band_means_F2_{tag}.csv", index=False)

    # ---- lookup-table export (pre-registration 8.9) ----------------------
    export_arm = win
    log(f"lookup export for {export_arm['arm']} "
        f"({'ADOPTED' if elig else 'NOT ADOPTED'})")
    c3.set_flag_inplace(design, c3.ARM_FLAG[export_arm["arm"]])
    tr2, te2 = ck.fold_slices(design, "F2")
    tempo = tr2["tempo_prior_game"].to_numpy(dtype="float64")
    tempo_edges = (float(np.quantile(tempo, 1 / 3)), float(np.quantile(tempo, 2 / 3)))
    t0 = time.time()
    lut, ldiag = c3.build_lookup(export_arm["_arm"], tr2, tempo_edges)
    ldiag["build_seconds"] = round(time.time() - t0, 1)

    t0 = time.time()
    lsc = c3.score_arm_v3(lut, te2, pit_seed=BASE_SEED)
    ldiag["lookup_rows_per_s"] = round(len(te2) / max(time.time() - t0, 1e-9), 1)
    t0 = time.time()
    _ = export_arm["_arm"].pmf(te2.iloc[:200_000])
    ldiag["live_rows_per_s"] = round(200_000 / max(time.time() - t0, 1e-9), 1)
    del _

    ldiag.update(c3.binning_error(export_arm["_arm"], lut, te2))
    ldiag["live_crps_trunc"] = export_arm["crps_trunc"]
    ldiag["lookup_crps_trunc"] = lsc["crps_trunc"]
    ldiag["d_crps_trunc"] = lsc["crps_trunc"] - export_arm["crps_trunc"]
    rl = ck.chain_halves(lut, te2, seed=BASE_SEED)
    cl = c3.emergent_cc(rl, export_arm["_actual_half"])
    ldiag["lookup_cc_mean_delta"] = cl["mean_delta"]
    ldiag["lookup_cc_sd_delta"] = cl["sd_delta"]
    ldiag["live_cc_mean_delta"] = export_arm["cc_mean_delta"]
    ldiag["live_cc_sd_delta"] = export_arm["cc_sd_delta"]
    ldiag["d_cc_mean"] = cl["mean_delta"] - export_arm["cc_mean_delta"]
    ldiag["lookup_cc_eoh_duration_gap"] = cl["eoh_duration_gap"]
    ldiag["adopted"] = bool(elig)
    (OUT / "v3_lookup_report.json").write_text(json.dumps(ldiag, indent=2, default=str),
                                               encoding="utf-8")
    np.savez_compressed(OUT / "v3_lookup_table.npz", table=lut.table,
                        tempo_edges=np.array(tempo_edges),
                        dims=np.array(c3.LOOKUP_DIMS), sizes=np.array(c3.LOOKUP_SIZES))
    log(f"  lookup: dCRPS {ldiag['d_crps_trunc']:+.6f}, TV mean {ldiag['tv_mean']:.5f} "
        f"max {ldiag['tv_max']:.4f}, G1-CC delta {ldiag['d_cc_mean']:+.3f}")

    suffix = "winner" if elig else "reference_not_adopted"
    with open(OUT / f"v3_{suffix}_{export_arm['arm']}.pkl", "wb") as f:
        pickle.dump(export_arm["_arm"], f)

    (OUT / "v3_verdict.json").write_text(json.dumps(verdict, indent=2, default=str),
                                         encoding="utf-8")
    log(json.dumps({k: v for k, v in verdict.items() if k != "floors_by_arm"}, indent=2,
                   default=str))


if __name__ == "__main__":
    main()
