"""
exp_clk5b_mean_consistent.py -- round 5b: a MEAN-CONSISTENT per-game pace latent.

Pre-registration: `docs/models/clock/experiments.md` section 21, committed at
`58d5050` BEFORE this script, before the module change and before any fitted
round-5b parameter existed.

WHAT IS NEW, AND WHAT IS REUSED BYTE FOR BYTE
---------------------------------------------
Round 5 (sections 16-20) fitted the per-game duration latent as
`A ~ LogN(-s2/2, s2)`, i.e. `E[A] = 1` on the DURATION scale. Every gate reads
the possession COUNT `P = 1200/Dbar`, which is the RECIPROCAL of the duration
mean, and `E[1/A] = e^{s2} > 1`: the latent necessarily adds `~0.15`
possessions per team-game (measured +0.095 clock-complete / +0.230 all-500 in
the closed loop, section 20.3). Round 5b changes ONLY the LOCATION `m` of the
latent, `A = exp(sigma*z + m)`, and asks whether the mean cost disappears.

Everything else is round 5's, imported from
`scripts/exp_clk5_dispersion_bakeoff.py` and NOT edited: the design table, the
S1 schedule loader, the per-unit variance algebra (`per_unit`, `var_sum`), the
per-game / per-offence summaries, round 5's own `fit_params` (which supplies the
A1 reference sigma), and `clock_v3.score_arm_v3` as the blind scorer. Only the
LOCATION-aware node moments, the three new fits and the location-aware grade are
new, and they run over R, A1, B1, B2 and B3 through ONE code path so no arm gets
a bespoke scorer.

WHAT IS AND IS NOT A HAND TUNE
------------------------------
`m` is either an analytic identity of the family (B1: `m = +s2/2` is the unique
location making `E[1/A] = 1`) or a method-of-moments estimate on TRAINING rows
(B2: `E[A] = c` matched to the training marginal mean duration). Neither reads
the test fold, the engine, or any gate value. A rescale of simulated possessions
to restore the mean would be the banned pattern (`docs/SIM_GUARDRAILS.md`
section 5); the mean here comes out of the fitted law and the grader reports
round 4's mean gate for every arm so the claim is checked, not asserted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import clock_v3 as c3  # noqa: E402
from cbb_sim.models import clock_v5 as c5  # noqa: E402


def _load_r5():
    """Round 5's bake-off script, imported as a module and NOT edited."""
    p = ROOT / "scripts" / "exp_clk5_dispersion_bakeoff.py"
    spec = importlib.util.spec_from_file_location("exp_clk5", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp_clk5"] = mod
    spec.loader.exec_module(mod)
    return mod


r5 = _load_r5()
GRID = r5.GRID
CHUNK = r5.CHUNK
log = r5.log


# ---------------------------------------------------------------------------
# node moments WITH a latent location (the only new piece of algebra)
# ---------------------------------------------------------------------------
def node_moments_loc(arm, df: pd.DataFrame, sigma: np.ndarray, m: np.ndarray,
                     z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`m1[i,j] = E[min(round(a_ij T), R_i)]`, `m2 = E[.^2]`, with
    `a_ij = exp(sigma_i z_j + m_i)`. Identical to round 5's `node_moments`
    except that the location is passed instead of being fixed at `-s2/2`."""
    n, K = len(df), len(z)
    m1 = np.empty((n, K)); m2 = np.empty((n, K))
    const = float(np.ptp(sigma)) < 1e-12 and float(np.ptp(m)) < 1e-12
    a_const = (c5.latent_values(float(sigma[0]), z, float(m[0]))[0]
               if const else None)
    for lo in range(0, n, CHUNK):
        blk = df.iloc[lo:lo + CHUNK]
        hi = lo + len(blk)
        p = np.asarray(arm.pmf(blk.reset_index(drop=True)), dtype=np.float64)
        R = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
        a_rows = (None if const
                  else c5.latent_values(sigma[lo:hi], z, m[lo:hi]))
        for j in range(K):
            g = (np.rint(a_const[j] * GRID)[None, :] if const
                 else np.rint(a_rows[:, j][:, None] * GRID[None, :]))
            c = np.minimum(g, R)
            m1[lo:hi, j] = (p * c).sum(axis=1)
            m2[lo:hi, j] = (p * c * c).sum(axis=1)
    return m1, m2


# ---------------------------------------------------------------------------
# the three new fits, method of moments, TRAINING rows only
# ---------------------------------------------------------------------------
def _var_pred(s2: float, c: float, meanV: float, meanM2: float,
              var_mbar: float) -> float:
    """Model-implied `Var_g(Dbar - mbar)` for a location-`m` latent.

        Dbar | A  has mean  A*mbar_g  and variance  A^2 * V_iid(g)
        =>  Var(Dbar) = E[A^2] V_iid + mbar^2 (E[A^2] - E[A]^2)
        with E[A] = c and E[A^2] = c^2 e^{s2}; the across-game spread of the
        mean shift (c-1)*mbar_g adds (c-1)^2 Var_g(mbar).
    Round 5's A1 is the c = 1 special case and reduces to its own equation."""
    k = np.exp(s2)
    return c * c * k * meanV + meanM2 * c * c * (k - 1.0) + (c - 1.0) ** 2 * var_mbar


def fit_b(kind: str, gt: pd.DataFrame, mean_dur: float, mean_emin: float) -> dict:
    """`kind` in {"b1", "b2"}. Returns sigma2, sigma, c, log_c, m."""
    T_var = float(np.var(gt["rbar"].to_numpy(), ddof=0))
    meanV = float(gt["V_iid"].mean())
    meanM2 = float(np.mean(gt["mbar"].to_numpy() ** 2))
    var_mbar = float(np.var(gt["mbar"].to_numpy(), ddof=0))

    if kind == "b1":
        # E[1/A] = 1  =>  m = +s2/2  =>  c = E[A] = e^{s2}.
        def f(s2):
            return _var_pred(s2, float(np.exp(s2)), meanV, meanM2, var_mbar) - T_var
        c_of = lambda s2: float(np.exp(s2))            # noqa: E731
    elif kind == "b2":
        # The law's MEAN level is re-estimated with the latent present, on
        # TRAINING rows: E[A] * E[min(T,R)] must equal the training marginal
        # mean duration. That is a moment condition of the fitted law, not an
        # adjustment applied to output.
        c_hat = mean_dur / mean_emin

        def f(s2):
            return _var_pred(s2, c_hat, meanV, meanM2, var_mbar) - T_var
        c_of = lambda s2: c_hat                        # noqa: E731
    else:
        raise ValueError(kind)

    lo, hi = 1e-12, 0.05
    s2 = float(brentq(f, lo, hi)) if f(lo) * f(hi) < 0 else max(
        0.0, (T_var - meanV) / (3 * meanV + meanM2))
    c = c_of(s2)
    return {"sigma2": s2, "sigma": float(np.sqrt(s2)), "c": float(c),
            "log_c": float(np.log(c)), "m": float(np.log(c) - 0.5 * s2)}


def fit_b3(gt: pd.DataFrame) -> dict:
    """B3: `sigma^2(x) = a + b * tempo_prior_game` under the B1 location.

    Round 5's A6 estimator, with the first-order variance relation corrected for
    the B1 location: at `c = e^{s2}`, `Var(Dbar) ~= V(1+3 s2) + s2 mbar^2`, so
    the per-game latent estimate is `(rc^2 - V) / (3V + mbar^2)`. Weighted least
    squares on the pregame tempo, weight = possessions, exactly as A6."""
    rc = gt["rbar"].to_numpy() - gt["rbar"].mean()
    V = gt["V_iid"].to_numpy()
    mb = gt["mbar"].to_numpy()
    s2_hat = (rc ** 2 - V) / (3.0 * V + mb ** 2)
    tv = gt["tempo"].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones_like(tv), tv])
    wt = gt["M"].to_numpy(dtype=np.float64)
    beta = np.linalg.lstsq(X * wt[:, None], s2_hat * wt, rcond=None)[0]
    return {"beta0": float(beta[0]), "beta1": float(beta[1])}


# ---------------------------------------------------------------------------
# one grade path for every arm
# ---------------------------------------------------------------------------
def sigma_m_rows(spec: dict, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(te)
    if spec["kind"] == "none":
        return np.zeros(n), np.zeros(n)
    if spec["kind"] == "b3":
        s2 = np.clip(spec["beta0"] + spec["beta1"]
                     * te["tempo_prior_game"].to_numpy(), 0.0, None)
        return np.sqrt(s2), 0.5 * s2                     # B1 location, per row
    s = np.full(n, float(spec["sigma"]))
    return s, np.full(n, float(spec["m"]))


def grade(name: str, spec: dict, te: pd.DataFrame, seg: np.ndarray,
          sched: list, z: np.ndarray, w: np.ndarray,
          needed_var: float, kbridge: float) -> tuple[dict, np.ndarray]:
    sigma, m = sigma_m_rows(spec, te)
    n = len(te)
    m1 = np.empty((n, len(z))); m2 = np.empty((n, len(z)))
    for k in np.unique(seg):
        r = np.flatnonzero(seg == k)
        a, b = node_moments_loc(sched[int(k)]["arm"], te.iloc[r],
                                sigma[r], m[r], z)
        m1[r] = a; m2[r] = b
    u = r5.per_unit(te, m1, m2, ["game_id"])
    eH, vF = r5.var_sum(u, w)
    M = u["n"].to_numpy(dtype=np.float64)
    var_dbar = float(np.mean((eH + vF) / M ** 2))
    e_min_mix = m1 @ w
    actual_mean = float(te["duration_s"].mean())
    row = {
        "arm": name, "kind": spec["kind"],
        "sigma": float(spec.get("sigma", 0.0)),
        "m": float(spec.get("m", 0.0)),
        "c_EA": float(np.exp(spec.get("m", 0.0) + 0.5 * spec.get("sigma", 0.0) ** 2)),
        "E_inv_A": float(np.exp(-spec.get("m", 0.0)
                                + 0.5 * spec.get("sigma", 0.0) ** 2)),
        "var_dbar": var_dbar, "sd_dbar": float(np.sqrt(var_dbar)),
        "P_sd_produced": kbridge * float(np.sqrt(var_dbar)),
        "P_sd_needed": kbridge * float(np.sqrt(needed_var)),
        "E_min_TR": float(e_min_mix.mean()),
        "n_test": n, "n_games": int(te["game_id"].nunique()),
    }
    row["P_sd_ratio"] = row["P_sd_produced"] / row["P_sd_needed"]
    row["primary_abs_dev"] = abs(row["P_sd_ratio"] - 1.0)
    row["E_min_gap_s"] = row["E_min_TR"] - actual_mean
    row["implied_poss_per_team_game_delta"] = (
        1200.0 / row["E_min_TR"] - 1200.0 / actual_mean)
    return row, e_min_mix


def inner_for(spec: dict, arm):
    if spec["kind"] == "none":
        return arm
    if spec["kind"] == "b3":
        b0, b1 = spec["beta0"], spec["beta1"]
        f = (lambda df, b0=b0, b1=b1: np.sqrt(np.clip(
            b0 + b1 * df["tempo_prior_game"].to_numpy(), 0.0, None)))
        return c5.LatentArm(arm, f, loc_kind="plus_half")
    return c5.LatentArm(arm, float(spec["sigma"]), loc_kind=spec["loc_kind"],
                        log_c=float(spec.get("log_c", 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/models/clock/v5b_bakeoff")
    ap.add_argument("--nodes", type=int, default=9)
    ap.add_argument("--fit-only", action="store_true",
                    help="write the fitted report and stop (lets the closed "
                         "loop start while the offline grade runs)")
    ap.add_argument("--no-score", action="store_true",
                    help="skip CRPS/PIT (the primary metric does not need it)")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    z, w = c5.gh_nodes(a.nodes)
    univ = pd.read_parquet(r5.UNIV_V2)
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    log("loading design")
    design = pd.read_parquet(r5.DESIGN)
    design = design[(design["period"] <= 2.0) & (design["game_id"].isin(cc))].copy()
    sched = r5.load_schedule("v3c_srfloor_P3_s1")
    fit_arm = sched[0]["arm"]

    def annotate(df: pd.DataFrame, by_seg: bool) -> pd.DataFrame:
        df = df.sort_values(["game_id", "period", "poss_index"],
                            kind="stable").reset_index(drop=True)
        e = np.empty(len(df)); v = np.empty(len(df))
        if by_seg:
            cuts = np.array([np.datetime64(r["refit_date"], "ns") for r in sched])
            sg = np.searchsorted(cuts, pd.to_datetime(df["game_date"]).to_numpy(),
                                 side="right") - 1
            assert (sg >= 0).all()
            for k in np.unique(sg):
                r = np.flatnonzero(sg == k)
                a1, a2 = node_moments_loc(sched[int(k)]["arm"], df.iloc[r],
                                          np.zeros(len(r)), np.zeros(len(r)),
                                          np.array([0.0]))
                e[r] = a1[:, 0]; v[r] = a2[:, 0] - a1[:, 0] ** 2
            df["_seg"] = sg
        else:
            for lo in range(0, len(df), CHUNK):
                blk = df.iloc[lo:lo + CHUNK]
                p = np.asarray(fit_arm.pmf(blk.reset_index(drop=True)),
                               dtype=np.float64)
                R = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
                c = np.minimum(GRID[None, :], R)
                mm = (p * c).sum(axis=1)
                e[lo:lo + len(blk)] = mm
                v[lo:lo + len(blk)] = (p * c * c).sum(axis=1) - mm ** 2
        df["e_min"] = e; df["v_min"] = v
        df["resid"] = df["duration_s"].to_numpy(dtype=np.float64) - e
        return df

    # ------------------------- fits (TRAINING only) -------------------------
    params = {}
    tr_by_fold = {}
    for fold, seasons in (("F2", (2022, 2023, 2024)), ("F1", (2022, 2023))):
        tr = annotate(design[design["season"].isin(seasons)].copy(), False)
        gt, _ = r5.summarise(tr)
        pr = dict(r5.fit_params(tr, fold))          # round 5's own A1 reference
        md = float(tr["duration_s"].mean()); me = float(tr["e_min"].mean())
        b1 = fit_b("b1", gt, md, me)
        b2 = fit_b("b2", gt, md, me)
        b3 = fit_b3(gt)
        pr.update({
            "train_mean_duration": md, "train_mean_e_min": me,
            "B1_sigma2": b1["sigma2"], "B1_sigma": b1["sigma"],
            "B1_m": b1["m"], "B1_c": b1["c"], "B1_log_c": b1["log_c"],
            "B2_sigma2": b2["sigma2"], "B2_sigma": b2["sigma"],
            "B2_m": b2["m"], "B2_c": b2["c"], "B2_log_c": b2["log_c"],
            "B3_beta0": b3["beta0"], "B3_beta1": b3["beta1"],
        })
        params[fold] = pr
        tr_by_fold[fold] = tr
        log(f"{fold}: A1 sigma {pr['A1_sigma']:.6f} | B1 sigma {b1['sigma']:.6f} "
            f"m {b1['m']:+.6f} c {b1['c']:.6f} | B2 sigma {b2['sigma']:.6f} "
            f"m {b2['m']:+.6f} c {b2['c']:.6f} | B3 {b3['beta0']:.6g},"
            f"{b3['beta1']:.6g}")

    # ------------- floor: spec-identical refits, game-block bootstrap -------
    floor_params = []
    tr_f2 = tr_by_fold["F2"]
    gids = tr_f2["game_id"].unique()
    idx = pd.Series(np.arange(len(tr_f2)), index=tr_f2["game_id"]).groupby(level=0)
    pos = {g: v.to_numpy() for g, v in idx}
    for seed in (20260911, 20260912):
        rng = np.random.default_rng(seed)
        pick = rng.choice(gids, size=len(gids), replace=True)
        rows = np.concatenate([pos[g] for g in pick])
        bs = tr_f2.iloc[rows].copy()
        bs["game_id"] = np.repeat(np.arange(len(pick)), [len(pos[g]) for g in pick])
        gtb, _ = r5.summarise(bs)
        mdb = float(bs["duration_s"].mean()); meb = float(bs["e_min"].mean())
        fb1 = fit_b("b1", gtb, mdb, meb)
        fb2 = fit_b("b2", gtb, mdb, meb)
        floor_params.append({"label": f"boot{seed}",
                             "B1_sigma": fb1["sigma"], "B1_m": fb1["m"],
                             "B2_sigma": fb2["sigma"], "B2_m": fb2["m"],
                             "B2_c": fb2["c"]})
        log(f"floor refit {seed}: B1 sigma {fb1['sigma']:.6f} "
            f"B2 sigma {fb2['sigma']:.6f} c {fb2['c']:.6f}")

    rep = {"params": params, "floor_params": floor_params,
           "pre_registration": "docs/models/clock/experiments.md section 21",
           "round5_report": "data/processed/models/clock/v5_bakeoff/v5_bakeoff_report.json"}
    (out / "v5b_bakeoff_report.json").write_text(
        json.dumps(rep, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out/'v5b_bakeoff_report.json'}")
    if a.fit_only:
        return

    # ------------------------- test slice (F2 = 2025) -----------------------
    te = annotate(design[design["season"] == 2025].copy(), True)
    seg = te["_seg"].to_numpy()
    gt, ot = r5.summarise(te)
    needed_var = float(np.var(gt["rbar"].to_numpy(), ddof=0))
    kbridge = float(gt["P"].mean()) / float(te["duration_s"].mean())
    log(f"test {len(te)} poss / {len(gt)} games, needed Var(rbar) {needed_var:.5f}, "
        f"needed P SD {kbridge*np.sqrt(needed_var):.4f}")
    gt["tq"] = pd.qcut(gt["tempo"], 5,
                       labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
    te = te.merge(gt[["game_id", "tq"]], on="game_id", how="left")

    p2 = params["F2"]
    arms = [
        ("R  v3c_srfloor_P3_s1", {"kind": "none"}),
        ("A1 v5_glat_shared", {"kind": "b", "loc_kind": "minus_half",
                               "sigma": p2["A1_sigma"],
                               "m": -0.5 * p2["A1_sigma2"]}),
        ("B1 v5b_glat_pmean", {"kind": "b", "loc_kind": "plus_half",
                               "sigma": p2["B1_sigma"], "m": p2["B1_m"]}),
        ("B2 v5b_glat_joint", {"kind": "b", "loc_kind": "log_c",
                               "sigma": p2["B2_sigma"], "m": p2["B2_m"],
                               "log_c": p2["B2_log_c"]}),
        ("B3 v5b_glat_pmean_tempo", {"kind": "b3", "beta0": p2["B3_beta0"],
                                     "beta1": p2["B3_beta1"]}),
    ]

    rows, resp_rows, cell_rows = [], [], []
    for name, spec in arms:
        t0 = time.time()
        row, e_mix = grade(name, spec, te, seg, sched, z, w, needed_var, kbridge)
        if not a.no_score:
            crps_r = np.empty(len(te)); pit_r = np.empty(len(te))
            ll_r = np.empty(len(te))
            for k in np.unique(seg):
                r = np.flatnonzero(seg == k)
                s = c3.score_arm_v3(inner_for(spec, sched[int(k)]["arm"]),
                                    te.iloc[r].reset_index(drop=True))
                crps_r[r] = s["_crps_trunc_rows"]
                pit_r[r] = s["_pit_rows"]
                ll_r[r] = s["_loglik_rows"]
            unc = ~te["censored"].to_numpy(dtype=bool)
            row["crps_trunc"] = float(np.nanmean(crps_r[unc]))
            row["censored_loglik"] = float(np.mean(ll_r))
            cells = c3.pit_by_cell_v3(te, pit_r)
            pw = cells[cells["powered"]]
            row["pit_worst_D"] = float(pw["ks_D"].max()) if len(pw) else float("nan")
            row["pit_leak_cells"] = int(pw["leak_sized"].sum()) if len(pw) else 0
        rows.append(row)
        log(f"{name}: P_sd {row['P_sd_produced']:.4f} ratio {row['P_sd_ratio']:.4f} "
            f"E[min] {row['E_min_TR']:.4f} gap {row['E_min_gap_s']:+.4f} "
            f"poss {row['implied_poss_per_team_game_delta']:+.4f} "
            f"({time.time()-t0:.0f}s)")

        # multi-level: per pregame-tempo quintile (responsiveness)
        for q in sorted(gt["tq"].unique()):
            gsub = gt[gt["tq"] == q]
            sel = te["tq"].to_numpy() == q
            sub = te[sel].reset_index(drop=True)
            r2, _ = grade(name, spec, sub, seg[sel], sched, z, w,
                          float(np.var(gsub["rbar"].to_numpy(), ddof=0)),
                          float(gsub["P"].mean()) / float(sub["duration_s"].mean()))
            resp_rows.append({"arm": name, "tempo_quintile": q,
                              "n_games": int(len(gsub)),
                              "P_sd_produced": r2["P_sd_produced"],
                              "P_sd_needed": r2["P_sd_needed"],
                              "ratio": r2["P_sd_ratio"]})
        # multi-level: per prev_end cell (mean gate only; a cell is not a game)
        for cell, sub in te.groupby("prev_end"):
            n = int(len(sub))
            s_r, m_r = sigma_m_rows(spec, sub)
            e = np.empty(len(sub))
            sg = seg[te["prev_end"].to_numpy() == cell]
            for k in np.unique(sg):
                r = np.flatnonzero(sg == k)
                a1, _ = node_moments_loc(sched[int(k)]["arm"], sub.iloc[r],
                                         s_r[r], m_r[r], z)
                e[r] = a1 @ w
            cell_rows.append({"arm": name, "prev_end": str(cell), "n": n,
                              "E_min_TR": float(e.mean()),
                              "actual": float(sub["duration_s"].mean()),
                              "gap": float(e.mean() - sub["duration_s"].mean()),
                              "powered": n >= 300})

    # ------------------------- the floor, as a metric delta -----------------
    floor_rows = []
    for fp in floor_params:
        for tag, spec in (("B1", {"kind": "b", "loc_kind": "plus_half",
                                  "sigma": fp["B1_sigma"], "m": fp["B1_m"]}),):
            r, _ = grade(f"floor {tag} {fp['label']}", spec, te, seg, sched,
                         z, w, needed_var, kbridge)
            floor_rows.append({"label": fp["label"], "arm": tag,
                               "sigma": fp[f"{tag}_sigma"],
                               "P_sd_produced": r["P_sd_produced"],
                               "P_sd_ratio": r["P_sd_ratio"],
                               "E_min_TR": r["E_min_TR"],
                               "implied_poss_delta":
                                   r["implied_poss_per_team_game_delta"]})

    res = pd.DataFrame(rows)
    res.to_csv(out / "v5b_bakeoff_F2.csv", index=False)
    pd.DataFrame(resp_rows).to_csv(out / "v5b_responsiveness_F2.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(out / "v5b_prev_end_F2.csv", index=False)
    pd.DataFrame(floor_rows).to_csv(out / "v5b_floor.csv", index=False)
    rep.update({"floor_rows": floor_rows, "needed_var_rbar": needed_var,
                "bridge_k": kbridge,
                "P_sd_needed": kbridge * float(np.sqrt(needed_var)),
                "n_test": int(len(te)), "n_test_games": int(len(gt))})
    (out / "v5b_bakeoff_report.json").write_text(
        json.dumps(rep, indent=2, default=str), encoding="utf-8")
    cols = [c for c in ["arm", "sigma", "m", "c_EA", "E_inv_A", "P_sd_produced",
                        "P_sd_ratio", "crps_trunc", "censored_loglik",
                        "pit_worst_D", "pit_leak_cells", "E_min_TR",
                        "E_min_gap_s", "implied_poss_per_team_game_delta"]
            if c in res.columns]
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(res[cols].to_string(index=False))
        print()
        print(pd.DataFrame(floor_rows).to_string(index=False))
        print()
        print(pd.DataFrame(resp_rows).pivot(index="arm",
                                            columns="tempo_quintile",
                                            values="ratio").to_string())


if __name__ == "__main__":
    main()
