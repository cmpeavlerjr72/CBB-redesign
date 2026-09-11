"""exp_clk5_dispersion_bakeoff.py -- round 5's fit + blind grade, one code path.

Pre-registration: `docs/models/clock/experiments.md` section 16, committed at
`28b5f17` BEFORE `clock_v5.py` or this script existed. Every arm -- the served
reference included -- is fitted by the same fitter, scored by the same scorer
(`clock_v3.score_arm_v3`, the round-3 blind path, untouched) and reduced to the
same primary metric by the same variance algebra. No arm has a bespoke scorer.

THE PRIMARY METRIC (pre-registration 16.5)
------------------------------------------
On a clock-complete regulation game the possessions tile 2400 s exactly, so

    P = 1200 / Dbar      and     SD(P) ~= (Pbar/mu) * SD(Dbar) = 3.851 * SD(Dbar)

(the delta-method step is checked at ratio 0.995 against the realised counts by
`scripts/diag_clk5_dispersion.py`). Each arm's `Var(Dbar)` is computed EXACTLY
from its own law by conditioning on the latent:

    Var(sum_i D_i | g) = E_A[ sum_i Var(D_i | A) ] + Var_A( sum_i E[D_i | A] )

with `E[D_i | A] = sum_t p_i(t) min(round(A t), R_i)` -- the quantity `loop.py`
actually subtracts from the clock -- evaluated on a 9-node Gauss-Hermite grid.
The reference is the A -> 1 special case of the same formula, so the arms are
not compared through two different pieces of arithmetic.

FITTING (pre-registration 16.2)
-------------------------------
Method of moments on TRAINING rows only. Fold 2 trains {2022, 2023, 2024} and
tests 2025; fold 1 trains {2022, 2023} and is robustness. The training rows are
predicted by the S1 artifact whose `max_train_date` precedes the whole training
window, so NO test-season row enters any fit and `created_at < tipoff` holds for
every graded row. The remaining in-sample-ness (a training game contributes
~1/N of its own cell mean, cells being >= 300 rows) is reported, not hidden.

Usage:
    .venv/Scripts/python.exe scripts/exp_clk5_dispersion_bakeoff.py \
        --out data/processed/models/clock/v5_bakeoff
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import clock as ck          # noqa: E402
from cbb_sim.models import clock_v3 as c3       # noqa: E402
from cbb_sim.models import clock_v5 as c5       # noqa: E402

CK_DIR = ROOT / "data" / "processed" / "models" / "clock"
UNIV_V2 = ROOT / "data" / "processed" / "games_universe_v2.parquet"
DESIGN = CK_DIR / "design_v2.parquet"
CHUNK = 40_000
GRID = np.arange(ck.DURATION_CAP + 1, dtype=np.float64)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_schedule(mode: str) -> list[dict]:
    from cbb_sim.engine.clock_adapter_v3 import V3C_MODES
    doc = json.loads((CK_DIR / V3C_MODES[mode]["manifest"]).read_text(encoding="utf-8"))
    out = []
    for m in doc["months"]:
        with open(CK_DIR / m["model_file"], "rb") as f:
            out.append({"refit_date": pd.Timestamp(m["refit_date"]),
                        "max_train_date": pd.Timestamp(m.get("max_train_date")),
                        "arm": pickle.load(f)})
    out.sort(key=lambda r: r["refit_date"])
    return out


# ---------------------------------------------------------------------------
# node moments: m1[i,j] = E[min(round(a_ij T), R_i)], m2 = E[ . ^2 ]
# ---------------------------------------------------------------------------
def node_moments(arm, df: pd.DataFrame, sigma: np.ndarray,
                 z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, K = len(df), len(z)
    m1 = np.empty((n, K)); m2 = np.empty((n, K))
    const = float(np.ptp(sigma)) < 1e-12
    a_const = c5.latent_values(float(sigma[0]), z)[0] if const else None
    for lo in range(0, n, CHUNK):
        blk = df.iloc[lo:lo + CHUNK]
        hi = lo + len(blk)
        p = np.asarray(arm.pmf(blk.reset_index(drop=True)), dtype=np.float64)
        R = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
        a_rows = (None if const else c5.latent_values(sigma[lo:hi], z))
        for j in range(K):
            g = (np.rint(a_const[j] * GRID)[None, :] if const
                 else np.rint(a_rows[:, j][:, None] * GRID[None, :]))
            c = np.minimum(g, R)
            m1[lo:hi, j] = (p * c).sum(axis=1)
            m2[lo:hi, j] = (p * c * c).sum(axis=1)
    return m1, m2


def per_unit(df: pd.DataFrame, m1: np.ndarray, m2: np.ndarray,
             keys: list[str]) -> pd.DataFrame:
    """Sum the node moments over a unit (a game, or a game x offence)."""
    K = m1.shape[1]
    d = df[keys].copy()
    for j in range(K):
        d[f"F{j}"] = m1[:, j]
        d[f"H{j}"] = m2[:, j] - m1[:, j] ** 2
    g = d.groupby(keys, sort=True)
    out = g.sum()
    out["n"] = g.size()
    return out.reset_index()


def var_sum(u: pd.DataFrame, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(E_A[sum Var(D|A)], Var_A(sum E[D|A])) per unit."""
    K = len(w)
    F = u[[f"F{j}" for j in range(K)]].to_numpy()
    H = u[[f"H{j}" for j in range(K)]].to_numpy()
    eH = H @ w
    eF = F @ w
    vF = (F ** 2) @ w - eF ** 2
    return eH, vF


# ---------------------------------------------------------------------------
# per-game / per-offence summaries used by both the fitter and the grader
# ---------------------------------------------------------------------------
def summarise(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = d.assign(off=np.where(d["offense_is_home"] > 0, "H", "A"))
    g = d.groupby("game_id")
    gt = pd.DataFrame({
        "M": g.size(), "dbar": g["duration_s"].mean(), "mbar": g["e_min"].mean(),
        "sum_v": g["v_min"].sum(), "tempo": g["tempo_prior_game"].first(),
    }).reset_index()
    gt["rbar"] = gt["dbar"] - gt["mbar"]
    gt["V_iid"] = gt["sum_v"] / gt["M"] ** 2
    gt["P"] = 1200.0 / gt["dbar"]
    o = d.groupby(["game_id", "off"])
    ot = pd.DataFrame({
        "n": o.size(), "dbar": o["duration_s"].mean(), "mbar": o["e_min"].mean(),
        "sum_v": o["v_min"].sum(),
    }).reset_index()
    ot["rbar"] = ot["dbar"] - ot["mbar"]
    ot["V_o"] = ot["sum_v"] / ot["n"] ** 2
    return gt, ot


def offence_pivot(ot: pd.DataFrame) -> pd.DataFrame:
    w = ot.pivot(index="game_id", columns="off",
                 values=["n", "mbar", "V_o", "rbar", "sum_v"]).dropna()
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.reset_index()


def fit_params(d: pd.DataFrame, label: str) -> dict:
    """Method of moments on one TRAINING slice. Nothing here sees a test row."""
    gt, ot = summarise(d)
    w = offence_pivot(ot)
    T_var = float(np.var(gt["rbar"].to_numpy(), ddof=0))
    meanV = float(gt["V_iid"].mean())
    meanM2 = float(np.mean(gt["mbar"].to_numpy() ** 2))
    s2_a1 = max((T_var - meanV) / (meanV + meanM2), 0.0)

    M = (w["n_H"] + w["n_A"]).to_numpy(dtype=np.float64)
    Q = float(np.mean((w["n_H"].to_numpy() ** 2 * w["mbar_H"].to_numpy() ** 2
                       + w["n_A"].to_numpy() ** 2 * w["mbar_A"].to_numpy() ** 2) / M ** 2))
    s2_a2 = max((T_var - meanV) / (meanV + Q), 0.0)

    rh = w["rbar_H"].to_numpy(); ra = w["rbar_A"].to_numpy()
    own_h = float(np.var(rh, ddof=0) - w["V_o_H"].mean())
    own_a = float(np.var(ra, ddof=0) - w["V_o_A"].mean())
    denom = float(np.mean(w["mbar_H"].to_numpy() ** 2 + w["V_o_H"].to_numpy())
                  + np.mean(w["mbar_A"].to_numpy() ** 2 + w["V_o_A"].to_numpy())) / 2.0
    s2_a3 = max((own_h + own_a) / 2.0 / denom, 0.0)
    cov = float(np.mean((rh - rh.mean()) * (ra - ra.mean())))
    mm = float(np.mean(w["mbar_H"].to_numpy() * w["mbar_A"].to_numpy()))
    rho_t = float(np.clip(cov / (s2_a3 * mm), -0.99, 0.99)) if s2_a3 > 0 else 0.0

    # AR(1) within (game, offence): consecutive possessions of the SAME offence
    dd = d.assign(off=np.where(d["offense_is_home"] > 0, "H", "A")).sort_values(
        ["game_id", "off", "period", "poss_index"], kind="stable")
    key = dd["game_id"].to_numpy() * 10 + (dd["off"].to_numpy() == "H")
    per = dd["period"].to_numpy()
    x = dd["resid"].to_numpy(dtype=np.float64)
    x = x - x.mean()
    ok = (key[:-1] == key[1:]) & (per[:-1] == per[1:])
    rho_ar = float(np.corrcoef(x[:-1][ok], x[1:][ok])[0, 1])

    # A6: per-game latent estimate regressed on pregame tempo (WLS, weight = M)
    rc = gt["rbar"].to_numpy() - gt["rbar"].mean()
    s2_hat = (rc ** 2 - gt["V_iid"].to_numpy()) / (gt["V_iid"].to_numpy()
                                                   + gt["mbar"].to_numpy() ** 2)
    tv = gt["tempo"].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones_like(tv), tv])
    wt = gt["M"].to_numpy(dtype=np.float64)
    beta = np.linalg.lstsq(X * wt[:, None], s2_hat * wt, rcond=None)[0]

    return {
        "label": label, "n_games": int(len(gt)), "n_possessions": int(len(d)),
        "T_var_train": T_var, "meanV_train": meanV,
        "A1_sigma2": s2_a1, "A1_sigma": float(np.sqrt(s2_a1)),
        "A2_sigma2": s2_a2, "A2_sigma": float(np.sqrt(s2_a2)),
        "A3_sigma2": s2_a3, "A3_sigma": float(np.sqrt(s2_a3)), "A3_rho_t": rho_t,
        "A4_rho": rho_ar,
        "A6_beta0": float(beta[0]), "A6_beta1": float(beta[1]),
        "train_own_latent_var_home": own_h, "train_own_latent_var_away": own_a,
        "train_offence_cov": cov,
    }


# ---------------------------------------------------------------------------
# grading one arm on the TEST slice
# ---------------------------------------------------------------------------
def grade(name: str, base_arm_of_row, te: pd.DataFrame, seg: np.ndarray,
          sched: list[dict], kind: str, par: dict, z: np.ndarray, w: np.ndarray,
          needed_var: float, kbridge: float, score: bool = True) -> dict:
    n = len(te)
    if kind in ("none", "a4"):
        sigma = np.zeros(n)
    elif kind == "a6":
        s2 = par["A6_beta0"] + par["A6_beta1"] * te["tempo_prior_game"].to_numpy()
        sigma = np.sqrt(np.clip(s2, 0.0, None))
    else:
        sigma = np.full(n, par["sigma"], dtype=np.float64)

    m1 = np.empty((n, len(z))); m2 = np.empty((n, len(z)))
    for k in np.unique(seg):
        r = np.flatnonzero(seg == k)
        a, b = node_moments(sched[int(k)]["arm"], te.iloc[r], sigma[r], z)
        m1[r] = a; m2[r] = b

    te = te.assign(off=np.where(te["offense_is_home"] > 0, "H", "A"))
    Mg = te.groupby("game_id").size()

    if kind in ("none", "a1", "a6"):
        u = per_unit(te, m1, m2, ["game_id"])
        eH, vF = var_sum(u, w)
        var_sum_g = eH + vF
        M = u["n"].to_numpy(dtype=np.float64)
        var_dbar = float(np.mean(var_sum_g / M ** 2))
        vF_off = None
    elif kind in ("a2", "a3"):
        u = per_unit(te, m1, m2, ["game_id", "off"])
        eH, vF = var_sum(u, w)
        u = u.assign(eH=eH, vF=vF)
        pv = u.pivot(index="game_id", columns="off", values=["eH", "vF", "n"]).dropna()
        pv.columns = [f"{a}_{b}" for a, b in pv.columns]
        tot = pv["eH_H"] + pv["eH_A"] + pv["vF_H"] + pv["vF_A"]
        if kind == "a3":
            tot = tot + 2.0 * par["rho_t"] * np.sqrt(pv["vF_H"] * pv["vF_A"])
        M = (pv["n_H"] + pv["n_A"]).to_numpy(dtype=np.float64)
        var_dbar = float(np.mean(tot.to_numpy() / M ** 2))
        vF_off = (float(pv["vF_H"].mean()), float(pv["vF_A"].mean()))
    elif kind == "a4":
        o = te.groupby(["game_id", "off"])
        ot = pd.DataFrame({"n": o.size(), "vbar": o["v_min"].mean()}).reset_index()
        f = c5.ar1_mean_inflation(par["rho"], ot["n"].to_numpy())
        ot["vs"] = ot["n"].to_numpy() * ot["vbar"].to_numpy() * f
        tot = ot.groupby("game_id")["vs"].sum()
        M = te.groupby("game_id").size().reindex(tot.index).to_numpy(dtype=np.float64)
        var_dbar = float(np.mean(tot.to_numpy() / M ** 2))
        vF_off = None
    else:
        raise ValueError(kind)

    e_min_mix = m1 @ w
    row = {
        "arm": name, "kind": kind,
        "param": json.dumps({k: v for k, v in par.items() if k != "label"}),
        "var_dbar": var_dbar,
        "sd_dbar": float(np.sqrt(var_dbar)),
        "P_sd_produced": kbridge * float(np.sqrt(var_dbar)),
        "P_sd_needed": kbridge * float(np.sqrt(needed_var)),
        "E_min_TR": float(e_min_mix.mean()),
        "n_test": n, "n_games": int(len(Mg)),
    }
    row["P_sd_ratio"] = row["P_sd_produced"] / row["P_sd_needed"]
    row["primary_abs_dev"] = abs(row["P_sd_ratio"] - 1.0)
    row["E_min_gap_s"] = row["E_min_TR"] - float(te["duration_s"].mean())
    row["implied_poss_per_team_game_delta"] = (
        1200.0 / row["E_min_TR"] - 1200.0 / float(te["duration_s"].mean()))
    if vF_off is not None:
        row["implied_offence_latent_var_home"] = vF_off[0] / (
            te[te["off"] == "H"].groupby("game_id").size().mean() ** 2)
        row["implied_offence_latent_var_away"] = vF_off[1] / (
            te[te["off"] == "A"].groupby("game_id").size().mean() ** 2)
        row["implied_offence_corr"] = par.get("rho_t", 0.0)
    return row, m1, sigma


def seg_for(sched: list[dict], te: pd.DataFrame) -> np.ndarray:
    """Each test row's own S1 segment under THIS schedule. Computed per
    schedule, never shared: the gamma arm's manifest is round 3b's and its
    refit dates are its own."""
    cuts = np.array([np.datetime64(r["refit_date"], "ns") for r in sched])
    seg = np.searchsorted(cuts, pd.to_datetime(te["game_date"]).to_numpy(),
                          side="right") - 1
    assert (seg >= 0).all(), "a game tips before the first refit date"
    return seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/models/clock/v5_bakeoff")
    ap.add_argument("--nodes", type=int, default=9)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    z, w = c5.gh_nodes(a.nodes)
    univ = pd.read_parquet(UNIV_V2)
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))

    log("loading design")
    design = pd.read_parquet(DESIGN)
    design = design[(design["period"] <= 2.0) & (design["game_id"].isin(cc))].copy()

    sched = load_schedule("v3c_srfloor_P3_s1")
    sched_g = load_schedule("v3c_gamma_P3_s1")
    fit_arm = sched[0]["arm"]        # max_train_date precedes the whole train window
    log(f"fitting artifact refit_date={sched[0]['refit_date'].date()} "
        f"max_train_date={sched[0]['max_train_date']}")

    # ---------------- training slices (no test-season row) ------------------
    params = {}
    for fold, seasons in (("F2", (2022, 2023, 2024)), ("F1", (2022, 2023))):
        tr = design[design["season"].isin(seasons)].copy()
        tr = tr.sort_values(["game_id", "period", "poss_index"],
                            kind="stable").reset_index(drop=True)
        e = np.empty(len(tr)); v = np.empty(len(tr))
        for lo in range(0, len(tr), CHUNK):
            blk = tr.iloc[lo:lo + CHUNK]
            p = np.asarray(fit_arm.pmf(blk.reset_index(drop=True)), dtype=np.float64)
            R = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
            c = np.minimum(GRID[None, :], R)
            mm = (p * c).sum(axis=1)
            e[lo:lo + len(blk)] = mm
            v[lo:lo + len(blk)] = (p * c * c).sum(axis=1) - mm ** 2
        tr["e_min"] = e; tr["v_min"] = v
        tr["resid"] = tr["duration_s"].to_numpy(dtype=np.float64) - e
        params[fold] = fit_params(tr, fold)
        log(f"{fold} fitted: " + json.dumps({k: round(vv, 6) for k, vv in params[fold].items()
                                             if isinstance(vv, float)}))
        if fold == "F2":
            tr_f2 = tr

    # ---------------- the floor: game-block bootstrap refits ----------------
    floor_params = []
    gids = tr_f2["game_id"].unique()
    for seed in (20260911, 20260912):
        rng = np.random.default_rng(seed)
        pick = rng.choice(gids, size=len(gids), replace=True)
        idx = pd.Series(np.arange(len(tr_f2)), index=tr_f2["game_id"]).groupby(level=0)
        pos = {g: v.to_numpy() for g, v in idx}
        rows = np.concatenate([pos[g] for g in pick])
        bs = tr_f2.iloc[rows].copy()
        bs["game_id"] = np.repeat(np.arange(len(pick)), [len(pos[g]) for g in pick])
        floor_params.append(fit_params(bs, f"boot{seed}"))
        log(f"floor refit seed {seed}: A1 sigma {floor_params[-1]['A1_sigma']:.5f}")

    # ---------------- test slice (F2 = 2025) --------------------------------
    te = design[design["season"] == 2025].copy()
    te = te.sort_values(["game_id", "period", "poss_index"],
                        kind="stable").reset_index(drop=True)
    cuts = np.array([np.datetime64(r["refit_date"], "ns") for r in sched])
    seg = np.searchsorted(cuts, pd.to_datetime(te["game_date"]).to_numpy(),
                          side="right") - 1
    assert (seg >= 0).all()
    e = np.empty(len(te)); v = np.empty(len(te))
    for k in np.unique(seg):
        r = np.flatnonzero(seg == k)
        m1r, m2r = node_moments(sched[int(k)]["arm"], te.iloc[r],
                                np.zeros(len(r)), np.array([0.0]))
        e[r] = m1r[:, 0]; v[r] = m2r[:, 0] - m1r[:, 0] ** 2
    te["e_min"] = e; te["v_min"] = v
    te["resid"] = te["duration_s"].to_numpy(dtype=np.float64) - e
    gt, ot = summarise(te)
    needed_var = float(np.var(gt["rbar"].to_numpy(), ddof=0))
    kbridge = float(gt["P"].mean()) / float(te["duration_s"].mean())
    log(f"test: {len(te)} possessions, {len(gt)} games, needed Var(rbar) "
        f"{needed_var:.4f}, bridge k {kbridge:.4f}, needed P SD "
        f"{kbridge*np.sqrt(needed_var):.4f}")

    p2 = params["F2"]
    arms = [
        ("R  v3c_srfloor_P3_s1", "none", {}, sched),
        ("A1 v5_glat_shared", "a1", {"sigma": p2["A1_sigma"]}, sched),
        ("A2 v5_glat_team", "a2", {"sigma": p2["A2_sigma"]}, sched),
        ("A3 v5_glat_biv", "a3", {"sigma": p2["A3_sigma"], "rho_t": p2["A3_rho_t"]}, sched),
        ("A4 v5_ar1", "a4", {"rho": p2["A4_rho"]}, sched),
        ("A5 v5_gamma_P3_s1", "none", {}, sched_g),  # round 3b's own manifest
        ("A6 v5_glat_tempo", "a6",
         {"A6_beta0": p2["A6_beta0"], "A6_beta1": p2["A6_beta1"]}, sched),
    ]

    rows = []
    quint = pd.qcut(gt["tempo"], 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
    gt["tq"] = quint
    te = te.merge(gt[["game_id", "tq"]], on="game_id", how="left")
    resp_rows = []

    seg_cache = {id(sched): seg, id(sched_g): seg_for(sched_g, te)}
    for name, kind, par, sc in arms:
        t0 = time.time()
        sg = seg_cache[id(sc)]
        row, m1, sigma = grade(name, None, te, sg, sc, kind, par, z, w,
                               needed_var, kbridge)
        # scoring: CRPS_trunc / PIT / censored loglik through the round-3 path
        if kind in ("none", "a4"):
            inner_by_seg = {int(k): sc[int(k)]["arm"] for k in np.unique(sg)}
        else:
            inner_by_seg = {}
            for k in np.unique(sg):
                if kind == "a6":
                    b0, b1 = par["A6_beta0"], par["A6_beta1"]
                    s_k = (lambda df, b0=b0, b1=b1: np.sqrt(np.clip(
                        b0 + b1 * df["tempo_prior_game"].to_numpy(), 0.0, None)))
                else:
                    s_k = float(sigma[0])
                inner_by_seg[int(k)] = c5.LatentArm(sc[int(k)]["arm"], s_k)
        crps_r = np.empty(len(te)); pit_r = np.empty(len(te)); ll_r = np.empty(len(te))
        for k in np.unique(sg):
            r = np.flatnonzero(sg == k)
            s = c3.score_arm_v3(inner_by_seg[int(k)], te.iloc[r].reset_index(drop=True))
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
            f"CRPS {row['crps_trunc']:.5f} E[min] {row['E_min_TR']:.4f} "
            f"({time.time()-t0:.0f}s)")

        # responsiveness: produced / needed by pregame-tempo quintile
        for q in sorted(gt["tq"].unique()):
            gsub = gt[gt["tq"] == q]
            sel = te["tq"].to_numpy() == q
            sub = te[sel].reset_index(drop=True)
            r2, _, _ = grade(name, None, sub, sg[sel], sc, kind, par, z, w,
                             float(np.var(gsub["rbar"].to_numpy(), ddof=0)),
                             float(gsub["P"].mean()) / float(sub["duration_s"].mean()))
            resp_rows.append({"arm": name, "tempo_quintile": q,
                              "n_games": int(len(gsub)),
                              "P_sd_produced": r2["P_sd_produced"],
                              "P_sd_needed": r2["P_sd_needed"],
                              "ratio": r2["P_sd_ratio"]})

    # ---------------- the floor, as a metric delta --------------------------
    floor_rows = []
    for fp in floor_params:
        r, _, _ = grade("floor " + fp["label"], None, te, seg, sched, "a1",
                        {"sigma": fp["A1_sigma"]}, z, w, needed_var, kbridge)
        floor_rows.append({"label": fp["label"], "A1_sigma": fp["A1_sigma"],
                           "P_sd_produced": r["P_sd_produced"],
                           "P_sd_ratio": r["P_sd_ratio"],
                           "E_min_TR": r["E_min_TR"]})

    res = pd.DataFrame(rows)
    res.to_csv(out / "v5_bakeoff_F2.csv", index=False)
    pd.DataFrame(resp_rows).to_csv(out / "v5_responsiveness_F2.csv", index=False)
    pd.DataFrame(floor_rows).to_csv(out / "v5_floor.csv", index=False)
    rep = {"params": params, "floor_params": floor_params,
           "floor_rows": floor_rows,
           "needed_var_rbar": needed_var, "bridge_k": kbridge,
           "P_sd_needed": kbridge * float(np.sqrt(needed_var)),
           "n_test": int(len(te)), "n_test_games": int(len(gt))}
    (out / "v5_bakeoff_report.json").write_text(json.dumps(rep, indent=2, default=str),
                                                encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(res[["arm", "P_sd_produced", "P_sd_needed", "P_sd_ratio",
                   "crps_trunc", "censored_loglik", "pit_worst_D", "pit_leak_cells",
                   "E_min_TR", "E_min_gap_s",
                   "implied_poss_per_team_game_delta"]].to_string(index=False))
        print()
        print(pd.DataFrame(floor_rows).to_string(index=False))


if __name__ == "__main__":
    main()
