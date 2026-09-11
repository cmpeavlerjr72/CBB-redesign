"""train_clock_v3b.py -- ROUND 3b: state parametrisation x censoring, then S1.

Pre-registration: `docs/models/clock/experiments.md` section 10, appended AFTER
round 3's verdict and BEFORE this script ran. Section 8 (round 3) is
STATIC-ONLY and is not edited.

PART A. Round 3 CONFIRMED that horn censoring is a real defect worth about 0.45
possessions per team-game, and still failed every gate with +0.9 to +1.4
possessions left. L23 / Decision 10 attribute an effect of that size to
`score_diff` FEEDBACK inside the clock model: `score_diff` is produced by the
simulation, so conditioning on it closes a loop offline scoring cannot see.
Part A crosses the censoring fix with three state parametrisations --

    P1  as designed    round-2 state verbatim
    P2  score removed  no simulation-produced score information at all
    P3  engine-safe    margin only as three coarse end-game indicators inside
                       the last 120 s of H2/OT

-- on the two arms that carried the fix furthest (`empirical_km3`, `gamma_aft`).
Round 3's P1 rows are reused verbatim, so the comparison is paired.

PART B. The winner is refit under S1 (in-season monthly walk-forward refit,
`docs/LEARNINGS.md` L21), with S0 as the reference, and its monthly artifacts
are persisted as per-month files the engine selects by game date.

GATES. Round 3's three carry over unchanged, plus the Decision-10 CLOSED-LOOP
gate: a paired-stream run of each arm inside the engine
(`scripts/diag_engine_multilevel.py`, 5 seeds) that must hold possessions per
game inside the G1 tolerance and must not move margin SD. That gate is launched
separately once Part A has written its arms, because it needs the engine
worker's harness.

Artifacts take a `v3b_` prefix; rounds 1-3 files are never overwritten.

Usage (threads capped at 4):
    OMP_NUM_THREADS=4 .venv/Scripts/python.exe scripts/train_clock_v3b.py
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
S1_DIR = OUT / "v3b_s1"
DESIGN_V2 = OUT / "design_v2.parquet"
ALL_SEASONS = [2022, 2023, 2024, 2025]
BASE_SEED = 20260910
N_JOBS = 4
TEST_SEASON = 2025


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def evaluate(arm, te, hc, seed: int = BASE_SEED) -> dict:
    """The round-3 scorer and chain, unchanged, so Part A rows are comparable
    row-by-row with section 9."""
    sc = c3.score_arm_v3(arm, te, pit_seed=BASE_SEED)
    row = {k: v for k, v in sc.items() if not k.startswith("_")}
    pit = c3.pit_by_cell_v3(te, sc["_pit_rows"])
    row["pit_leak_failures"] = int(pit["leak_sized"].sum())
    row["pit_powered"] = int(pit["powered"].sum())
    pw = pit[pit["powered"]]
    row["pit_worst_D"] = float(pw["ks_D"].max()) if len(pw) else float("nan")
    row["pit_pass"] = bool(row["pit_leak_failures"] == 0)

    res = ck.chain_halves(arm, te, seed=seed)
    ah = c3.actual_end_of_half_cc(te, hc)
    cc = c3.emergent_cc(res, ah)
    row.update({f"cc_{k}": v for k, v in cc.items() if k != "months"})
    allg = ck.emergent_report(res, ah)
    row["all_mean_delta"] = allg["mean_delta"]
    row["all_sd_delta"] = allg["sd_delta"]
    row["_pit_cells"] = pit
    row["_cc_months"] = pd.DataFrame(cc["months"])
    row["_segments"] = c3.segment_table(te, sc["_crps_trunc_rows"], sc["_loglik_rows"])
    row["_responsiveness"] = c3.responsiveness(res, te)
    row["_bands"] = c3.pred_mean_by_band(arm, te)
    row["_crps_rows"] = sc["_crps_trunc_rows"]
    row["_game_ids"] = te["game_id"].to_numpy()
    row["_actual_half"] = ah
    row["_arm"] = arm
    return row


def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def block_bootstrap_se_rows(game_ids, rows, n_rep: int = 200, seed: int = 4242) -> float:
    ok = np.isfinite(rows)
    g0, v = np.asarray(game_ids)[ok], rows[ok]
    order = np.argsort(g0, kind="stable")
    g, v = g0[order], v[order]
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    sums = np.add.reduceat(v, starts)
    ends = np.concatenate([starts[1:], [len(g)]])
    ns = (ends - starts).astype("float64")
    rng = np.random.default_rng(seed)
    out = np.empty(n_rep)
    for r in range(n_rep):
        pick = rng.integers(0, len(starts), len(starts))
        out[r] = sums[pick].sum() / ns[pick].sum()
    return float(out.std(ddof=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-s1", action="store_true")
    args = ap.parse_args()

    S1_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()

    v3 = json.loads((OUT / "v3_verdict.json").read_text(encoding="utf-8"))
    floor = float(v3["floor"])
    eoh_floor = v3["end_of_half_floor"]
    log(f"round-3 verdict {v3['verdict']}; CRPS floor {floor:.6f}")

    design = pd.read_parquet(DESIGN_V2)
    design, cdiag = c3.attach_horn_censoring(design, ALL_SEASONS)
    hc = c3.load_clock_complete()
    c3.set_flag_inplace(design, "horn")
    tr, te = ck.fold_slices(design, "F2")
    log(f"F2 train {len(tr):,} / test {len(te):,}; horn-censored "
        f"{cdiag['censored_horn_pct']}%")

    # ---- Part A: 2 base arms x 3 parametrisations ------------------------
    rows: list[dict] = []
    for base in c3.V3B_BASE_ARMS:
        for par in c3.PARAMETRISATIONS:
            t0 = time.time()
            arm = c3.fit_arm_v3b(base, par, tr, seed=BASE_SEED)
            fit_s = time.time() - t0
            r = evaluate(arm, te, hc)
            r.update(base_arm=base, parametrisation=par, arm=f"{base}|{par}",
                     features=c3.P_FEATURES[par], fit_seconds=round(fit_s, 1),
                     n_train=int(len(tr)))
            r["eoh_share_pass"] = bool(abs(r["cc_eoh_share_gap"]) <= eoh_floor["share"])
            r["eoh_dur_pass"] = bool(abs(r["cc_eoh_duration_gap"]) <= eoh_floor["duration"])
            r["eoh_pass"] = bool(r["eoh_share_pass"] and r["eoh_dur_pass"])
            r["g1_pass"] = bool(r["cc_g1_pass"])
            r["offline_gates_pass"] = bool(r["g1_pass"] and r["pit_pass"] and r["eoh_pass"])
            r["crps_floor_se"] = block_bootstrap_se_rows(r["_game_ids"], r["_crps_rows"])
            rows.append(r)
            log(f"  {base}|{par}: CRPS_t {r['crps_trunc']:.4f} "
                f"loglik {r['censored_loglik']:.4f} "
                f"G1-CC {r['cc_mean_delta']:+.3f}/{r['cc_sd_delta']:+.3f} "
                f"EOH-CC {r['cc_eoh_share_gap']:+.5f}/{r['cc_eoh_duration_gap']:+.3f} "
                f"PITfail {r['pit_leak_failures']} fit {fit_s:.0f}s")

    grid = pd.DataFrame([strip(r) for r in rows]).sort_values("crps_trunc")
    grid.to_csv(OUT / "v3b_grid_F2_partA.csv", index=False)
    for r in rows:
        t = r["arm"].replace("|", "_")
        r["_pit_cells"].to_csv(OUT / f"v3b_pit_cells_F2_{t}.csv", index=False)
        r["_cc_months"].to_csv(OUT / f"v3b_cc_months_F2_{t}.csv", index=False)
        r["_segments"].to_csv(OUT / f"v3b_segments_F2_{t}.csv", index=False)
        r["_responsiveness"].to_csv(OUT / f"v3b_responsiveness_F2_{t}.csv", index=False)
        r["_bands"].to_csv(OUT / f"v3b_band_means_F2_{t}.csv", index=False)
        with open(OUT / f"v3b_arm_{t}.pkl", "wb") as f:
            pickle.dump(r["_arm"], f)

    elig = [r for r in rows if r["offline_gates_pass"]]
    if elig:
        best = min(elig, key=lambda r: r["crps_trunc"])
        tied = [r for r in elig if r["crps_trunc"] - best["crps_trunc"] <= floor]
        pick = sorted(tied, key=lambda r: (c3.P_SIMPLICITY[r["parametrisation"]],
                                           0 if r["base_arm"] == "empirical_km3" else 1,
                                           r["crps_trunc"]))[0]
    else:
        pick = min(rows, key=lambda r: r["cc_mean_delta"] ** 2)

    verdict = {
        "round": "3b-A", "seed": BASE_SEED, "floor": floor,
        "end_of_half_floor": eoh_floor,
        "n_arms": len(rows),
        "n_pass_g1": int(sum(r["g1_pass"] for r in rows)),
        "n_pit_clean": int(sum(r["pit_pass"] for r in rows)),
        "n_pass_eoh": int(sum(r["eoh_pass"] for r in rows)),
        "n_offline_eligible": len(elig),
        "offline_winner": pick["arm"] if elig else None,
        "closed_loop_candidate": pick["arm"],
        "closed_loop_gate": "PENDING -- scripts/diag_engine_multilevel.py, 5 seeds, "
                            "possessions within G1 tolerance and margin SD unmoved "
                            "(pre-registration 10.3). No arm is adopted until it runs.",
        "verdict": "OFFLINE STAGE ONLY -- NO ARM ADOPTED",
        "created_at": started, "finished_at": datetime.now(UTC).isoformat(),
    }
    (OUT / "v3b_verdict_partA.json").write_text(json.dumps(verdict, indent=2, default=str),
                                                encoding="utf-8")
    log(json.dumps({k: v for k, v in verdict.items()
                    if k != "end_of_half_floor"}, indent=2, default=str))

    if args.skip_s1:
        return

    # ---- Part B: S1 on the closed-loop candidate -------------------------
    base, par = pick["base_arm"], pick["parametrisation"]
    log(f"Part B: S1 scheme confirmation on {base}|{par}")

    def fit_wrapper(train_rows, seed=BASE_SEED, **kw):
        return c3.fit_arm_v3b(base, par, train_rows, seed=seed)

    # monthly walk-forward, reusing possession_outcome's month boundaries
    from cbb_sim.models import possession_outcome as PO
    te_dates = pd.to_datetime(te["game_date"])
    tr_dates = pd.to_datetime(tr["game_date"])
    cols = c3.s1_fit_columns("empirical_km3" if base == "empirical_km3" else "gamma_aft")
    cols = [c for c in dict.fromkeys([*cols, *ck.feature_set(c3.P_FEATURES[par])])
            if c in tr.columns]
    cuts = PO.month_boundaries(te_dates)
    tr_small = tr[cols]
    arms, kept, segments = [], [], []
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = ((te_dates >= cut) if nxt is None
               else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not seg.any():
            continue
        before = (te_dates < cut).to_numpy()
        prior = te.loc[before, cols]
        fit_rows = tr_small if not len(prior) else pd.concat([tr_small, prior], ignore_index=True)
        arms.append(fit_wrapper(fit_rows))
        kept.append(pd.Timestamp(cut))
        mx = tr_dates.max() if not before.any() else max(tr_dates.max(), te_dates[before].max())
        segments.append({"refit_date": str(pd.Timestamp(cut).date()),
                         "valid_from": str(pd.Timestamp(cut).date()),
                         "valid_to": None if nxt is None
                         else str((pd.Timestamp(nxt) - pd.Timedelta(days=1)).date()),
                         "n_train": int(len(fit_rows)),
                         "n_train_from_test_season": int(len(prior)),
                         "n_scored": int(seg.sum()),
                         "max_train_date": str(pd.Timestamp(mx).date())})
        log(f"  refit {segments[-1]['refit_date']}: {len(fit_rows):,} train rows "
            f"({len(prior):,} from the test season), {int(seg.sum()):,} scored")
        del fit_rows, prior

    ma = c3.MonthlyArm(cuts=kept, arms=arms, base_arm=f"{base}|{par}")
    r_s1 = evaluate(ma, te, hc)
    r_s1.update(arm=f"{base}|{par}", scheme="S1", n_fits=len(arms))
    r_s0 = {**pick, "scheme": "S0", "n_fits": 1}
    log(f"  S0 CRPS_t {r_s0['crps_trunc']:.4f} G1-CC {r_s0['cc_mean_delta']:+.3f}")
    log(f"  S1 CRPS_t {r_s1['crps_trunc']:.4f} G1-CC {r_s1['cc_mean_delta']:+.3f}")

    regressions = []
    if r_s1["crps_trunc"] - r_s0["crps_trunc"] > floor:
        regressions.append(f"CRPS_trunc worse by {r_s1['crps_trunc'] - r_s0['crps_trunc']:.6f}")
    if r_s0["pit_pass"] and not r_s1["pit_pass"]:
        regressions.append("PIT: S0 clean, S1 not")
    if r_s0["cc_g1_pass"] and not r_s1["cc_g1_pass"]:
        regressions.append("G1-CC: S0 passes, S1 does not")
    if abs(r_s1["cc_mean_delta"]) - abs(r_s0["cc_mean_delta"]) > ck.G1_MEAN_TOL:
        regressions.append("G1-CC mean drifted by more than a whole tolerance")
    for key, fl in (("cc_eoh_share_gap", eoh_floor["share"]),
                    ("cc_eoh_duration_gap", eoh_floor["duration"])):
        if abs(r_s1[key]) - abs(r_s0[key]) > fl:
            regressions.append(f"{key} worse by more than its floor")

    # persist the monthly artifacts + manifest the engine selects by date
    tempo = tr["tempo_prior_game"].to_numpy(dtype="float64")
    tempo_edges = (float(np.quantile(tempo, 1 / 3)), float(np.quantile(tempo, 2 / 3)))
    tag = f"{base}_{par}"
    manifest = []
    for k, (cut, sub) in enumerate(zip(kept, arms, strict=False)):
        stamp = f"{pd.Timestamp(cut).year:04d}-{pd.Timestamp(cut).month:02d}"
        mfile = S1_DIR / f"{tag}_S1_{TEST_SEASON}_{stamp}.pkl"
        with open(mfile, "wb") as f:
            pickle.dump(sub, f)
        before = (te_dates < pd.Timestamp(cut)).to_numpy()
        bin_rows = tr if not before.any() else pd.concat([tr, te.loc[before]], ignore_index=True)
        lut, ldiag = c3.build_lookup(sub, bin_rows, tempo_edges)
        lfile = S1_DIR / f"lookup_{tag}_S1_{TEST_SEASON}_{stamp}.npz"
        np.savez_compressed(lfile, table=lut.table, tempo_edges=np.array(tempo_edges),
                            dims=np.array(c3.LOOKUP_DIMS), sizes=np.array(c3.LOOKUP_SIZES))
        manifest.append({**segments[k], "season": TEST_SEASON, "month": stamp,
                         "model_file": str(mfile.relative_to(OUT)).replace("\\", "/"),
                         "lookup_file": str(lfile.relative_to(OUT)).replace("\\", "/"),
                         "lookup_empty_cells": ldiag["n_empty_cells"],
                         "lookup_cells": ldiag["n_cells"]})
        del bin_rows, lut

    (S1_DIR / "manifest.json").write_text(json.dumps({
        "base_arm": base, "parametrisation": par, "scheme": "S1",
        "season": TEST_SEASON,
        "selection_rule": ("pick the row whose [valid_from, valid_to] contains the game's "
                           "own date; equivalently the latest refit_date at or before it. "
                           "valid_to null means the last month of the season."),
        "naming": ("{arm}_S1_{season}_{YYYY-MM}.pkl and "
                   "lookup_{arm}_S1_{season}_{YYYY-MM}.npz, relative to "
                   "data/processed/models/clock/"),
        "lookup_grid": {"dims": list(c3.LOOKUP_DIMS), "sizes": list(c3.LOOKUP_SIZES),
                        "durations": int(c3.N_GRID)},
        "adopted_scheme": "S0" if regressions else "S1",
        "arm_adopted": False,
        "note": "Adopting S1 as the SCHEME does not adopt the arm; round 3 adopted nothing "
                "and the Decision-10 closed-loop gate has not run.",
        "months": manifest,
    }, indent=2, default=str), encoding="utf-8")

    pd.DataFrame([strip(r_s0), strip(r_s1)]).to_csv(OUT / "v3b_grid_F2_partB.csv", index=False)
    (OUT / "v3b_verdict_partB.json").write_text(json.dumps({
        "round": "3b-B", "base_arm": base, "parametrisation": par,
        "floor": floor, "n_fits_S1": len(arms),
        "S0": {k: strip(r_s0)[k] for k in ("crps_trunc", "censored_loglik", "cc_mean_delta",
                                           "cc_sd_delta", "cc_eoh_share_gap",
                                           "cc_eoh_duration_gap", "pit_leak_failures")},
        "S1": {k: strip(r_s1)[k] for k in ("crps_trunc", "censored_loglik", "cc_mean_delta",
                                           "cc_sd_delta", "cc_eoh_share_gap",
                                           "cc_eoh_duration_gap", "pit_leak_failures")},
        "d_crps_trunc": r_s1["crps_trunc"] - r_s0["crps_trunc"],
        "d_crps_in_floors": (r_s1["crps_trunc"] - r_s0["crps_trunc"]) / max(floor, 1e-12),
        "regressions": regressions,
        "scheme_adopted": "S0" if regressions else "S1",
        "s1_schedule": segments,
        "finished_at": datetime.now(UTC).isoformat(),
    }, indent=2, default=str), encoding="utf-8")
    log(f"Part B done: scheme adopted = {'S0' if regressions else 'S1'}; "
        f"regressions = {regressions}")


if __name__ == "__main__":
    main()
