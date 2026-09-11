"""
exp_clk5c_dispersion_function.py -- round 5c: the latent's DISPERSION FUNCTION.

Pre-registration: `docs/models/clock/experiments.md` section 24, committed at
`98f8db5` BEFORE this script and before any fitted round-5c parameter existed,
together with section 23, the diagnosis it is written against.

REUSED BYTE FOR BYTE: round 5's design loader, S1 schedule, `per_unit`,
`var_sum`, `summarise` and `fit_params`; round 5b's `node_moments_loc`, `fit_b`
(which supplies the B1 reference sigma and location) and its per-game latent
estimator `fit_b3`'s algebra. NEW: only the three dispersion functions and the
per-row sigma they produce. Every arm is graded through ONE code path.

ARMS (section 24.1): B1 reference (constant sigma), C1 per-team hierarchical,
C2 as-of possession-count SD (failure PRE-REGISTERED), C4 quadratic in pregame
pace. C3 (the per-offence latent) is round 5's A2 re-located and is NOT fitted
here; section 25 records that.

Every arm uses the B1 location `m = +sigma^2(x)/2` row-wise, so `E[1/A] = 1`
holds row-wise and no arm can move the possession-count mean in its own favour.
All fitting is on TRAINING rows only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, fn: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fn)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r5 = _load("exp_clk5", "exp_clk5_dispersion_bakeoff.py")
r5b = _load("exp_clk5b", "exp_clk5b_mean_consistent.py")
d1 = _load("diag_clk5c_q2", "diag_clk5c_q2.py")
log = r5.log
OUT = ROOT / "data" / "processed" / "models" / "clock" / "v5c_bakeoff"
MIN_PRIOR = 5
NBOOT = 2000
BOOT_SEED = 20260911


def annotate(df: pd.DataFrame, sched: list, by_seg: bool) -> pd.DataFrame:
    """Identical to round 5b's nested `annotate` (section 22 code path)."""
    df = df.sort_values(["game_id", "period", "poss_index"],
                        kind="stable").reset_index(drop=True)
    n = len(df)
    e = np.empty(n); v = np.empty(n)
    if by_seg:
        cuts = np.array([np.datetime64(r["refit_date"], "ns") for r in sched])
        sg = np.searchsorted(cuts, pd.to_datetime(df["game_date"]).to_numpy(),
                             side="right") - 1
        assert (sg >= 0).all()
        df["_seg"] = sg
    else:
        sg = np.zeros(n, dtype=int)
        df["_seg"] = 0
    for k in np.unique(sg):
        r = np.flatnonzero(sg == k)
        a1, a2 = r5b.node_moments_loc(sched[int(k)]["arm"], df.iloc[r],
                                      np.zeros(len(r)), np.zeros(len(r)),
                                      np.array([0.0]))
        e[r] = a1[:, 0]; v[r] = a2[:, 0] - a1[:, 0] ** 2
    df["e_min"] = e; df["v_min"] = v
    df["resid"] = df["duration_s"].to_numpy(dtype=np.float64) - e
    return df


def game_table(d: pd.DataFrame) -> pd.DataFrame:
    """Per-game summary + the per-game latent estimate + the pregame features."""
    gt, _ = r5.summarise(d)
    rc = gt["rbar"].to_numpy() - gt["rbar"].mean()
    V = gt["V_iid"].to_numpy(); mb = gt["mbar"].to_numpy()
    gt["s2_hat"] = (rc ** 2 - V) / (3.0 * V + mb ** 2)     # round 5b fit_b3
    tp = (d.groupby(["game_id", "offense_team_id"]).size().rename("poss")
          .reset_index()
          .merge(d.groupby("game_id")[["game_date", "season"]].first()
                 .reset_index(), on="game_id")
          .sort_values(["offense_team_id", "season", "game_date"],
                       kind="stable"))
    g = tp.groupby(["offense_team_id", "season"])
    tp["asof_sd"] = g["poss"].transform(
        lambda s: s.shift(1).expanding(MIN_PRIOR).std(ddof=0))
    wide = tp.groupby("game_id").agg(
        team_a=("offense_team_id", "min"), team_b=("offense_team_id", "max"),
        asof_sd=("asof_sd", "mean"), asof_n=("asof_sd", "count"))
    gt = gt.merge(wide.reset_index(), on="game_id", how="left")
    gt["asof_ok"] = gt["asof_n"] == 2
    return gt, tp


def _wls(gt: pd.DataFrame, cols: list[np.ndarray]) -> np.ndarray:
    X = np.column_stack([np.ones(len(gt))] + cols)
    w = gt["M"].to_numpy(dtype=np.float64)
    y = gt["s2_hat"].to_numpy()
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return np.linalg.lstsq(X[ok] * w[ok, None], y[ok] * w[ok], rcond=None)[0]


def fit_c1(gt: pd.DataFrame, s2_global: float) -> dict:
    """Per-team latent variance, empirical-Bayes shrunk to the global."""
    long = pd.concat([gt[["team_a", "s2_hat"]].rename(columns={"team_a": "t"}),
                      gt[["team_b", "s2_hat"]].rename(columns={"team_b": "t"})])
    g = long.groupby("t")["s2_hat"]
    tm = pd.DataFrame({"n": g.size(), "mean": g.mean()}).reset_index()
    v_w = float(np.var(long["s2_hat"].to_numpy(), ddof=1))      # within-team
    tm = tm[tm["n"] >= 5]
    tau_b2 = max(float(np.var(tm["mean"].to_numpy(), ddof=1))
                 - v_w * float(np.mean(1.0 / tm["n"].to_numpy())), 0.0)
    k = v_w / tau_b2 if tau_b2 > 0 else np.inf
    w = tm["n"].to_numpy() / (tm["n"].to_numpy() + k)
    tm["s2_team"] = s2_global + w * (tm["mean"].to_numpy() - s2_global)
    return {"k": float(k), "v_within": v_w, "tau_between2": tau_b2,
            "n_teams": int(len(tm)), "s2_global": s2_global,
            "table": dict(zip(tm["t"].astype(int), tm["s2_team"].astype(float)))}


def sigma2_rows(arm: str, gt: pd.DataFrame, p: dict) -> np.ndarray:
    """Per-GAME sigma^2 under one arm. Nothing here reads an outcome."""
    n = len(gt)
    if arm == "B1":
        return np.full(n, p["B1_sigma2"])
    if arm == "C1":
        tab = p["C1"]["table"]; g0 = p["B1_sigma2"]
        a = np.array([tab.get(int(t), g0) for t in gt["team_a"]])
        b = np.array([tab.get(int(t), g0) for t in gt["team_b"]])
        return 0.5 * (a + b)
    if arm == "C2":
        x = gt["asof_sd"].to_numpy(dtype=np.float64).copy()
        x[~np.isfinite(x)] = p["C2_xbar"]        # imputed at the TRAIN mean
        b = p["C2_beta"]
        return b[0] + b[1] * x
    if arm == "C4":
        t = gt["tempo"].to_numpy(dtype=np.float64) - p["C4_tbar"]
        b = p["C4_beta"]
        return b[0] + b[1] * t + b[2] * t ** 2
    raise ValueError(arm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=NBOOT)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from cbb_sim.models import clock_v5 as c5
    z, w = c5.gh_nodes(9)

    univ = pd.read_parquet(r5.UNIV_V2)
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    design = pd.read_parquet(r5.DESIGN)
    design = design[(design["period"] <= 2.0) & (design["game_id"].isin(cc))].copy()
    sched = r5.load_schedule("v3c_srfloor_P3_s1")

    params = {}
    for fold, seasons in (("F2", (2022, 2023, 2024)), ("F1", (2022, 2023))):
        tr = annotate(design[design["season"].isin(seasons)].copy(), sched, False)
        gt_tr, _ = game_table(tr)
        b1 = r5b.fit_b("b1", gt_tr, float(tr["duration_s"].mean()),
                       float(tr["e_min"].mean()))
        xok = gt_tr[gt_tr["asof_ok"]]
        xbar = float(xok["asof_sd"].mean())
        tbar = float(gt_tr["tempo"].mean())
        c2b = _wls(xok, [xok["asof_sd"].to_numpy(dtype=np.float64)])
        tc = gt_tr["tempo"].to_numpy(dtype=np.float64) - tbar
        c4b = _wls(gt_tr, [tc, tc ** 2])
        p = {"B1_sigma2": b1["sigma2"], "B1_sigma": b1["sigma"], "B1_m": b1["m"],
             "C2_beta": c2b.tolist(), "C2_xbar": xbar,
             "C4_beta": c4b.tolist(), "C4_tbar": tbar,
             "C1": fit_c1(gt_tr, b1["sigma2"]),
             "n_train_games": int(len(gt_tr)),
             "asof_coverage": float(gt_tr["asof_ok"].mean())}
        params[fold] = p
        log(f"{fold}: B1 s2 {b1['sigma2']:.6g} | C2 {c2b[0]:.4g},{c2b[1]:.4g} "
            f"(cov {p['asof_coverage']:.2f}) | C4 {c4b[0]:.4g},{c4b[1]:.4g},"
            f"{c4b[2]:.4g} | C1 k {p['C1']['k']:.2f} tau_b2 "
            f"{p['C1']['tau_between2']:.3g} teams {p['C1']['n_teams']}")

    te = annotate(design[design["season"] == 2025].copy(), sched, True)
    seg = te["_seg"].to_numpy()
    gt, _tp = game_table(te)
    needed_var = float(np.var(gt["rbar"].to_numpy(), ddof=0))
    kbridge = float(gt["P"].mean()) / float(te["duration_s"].mean())
    gt["tq"] = pd.qcut(gt["tempo"], 5,
                       labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
    log(f"test {len(te)} poss / {len(gt)} games; needed P SD "
        f"{kbridge*np.sqrt(needed_var):.4f}")

    p2 = params["F2"]
    rows, resp = [], []
    for arm in ("B1", "C1", "C2", "C4"):
        s2g = np.clip(sigma2_rows(arm, gt, p2), 1e-8, None)
        mp = dict(zip(gt["game_id"].astype(int), s2g))
        s2_row = np.array([mp[int(g)] for g in te["game_id"]])
        sigma = np.sqrt(s2_row); m = 0.5 * s2_row
        n = len(te)
        m1 = np.empty((n, len(z))); m2 = np.empty((n, len(z)))
        for k in np.unique(seg):
            r = np.flatnonzero(seg == k)
            aa, bb = r5b.node_moments_loc(sched[int(k)]["arm"], te.iloc[r],
                                          sigma[r], m[r], z)
            m1[r] = aa; m2[r] = bb
        u = r5.per_unit(te, m1, m2, ["game_id"])
        eH, vF = r5.var_sum(u, w)
        M = u["n"].to_numpy(dtype=np.float64)
        comp = pd.DataFrame({"game_id": u["game_id"].to_numpy(),
                             "within_g": eH / M ** 2, "between_g": vF / M ** 2})
        g = gt.merge(comp, on="game_id", how="inner")
        e_mix = m1 @ w
        actual_mean = float(te["duration_s"].mean())
        s_all = d1._stats(d1._arrs(g))
        row = {"arm": arm, "P_sd_produced": kbridge * np.sqrt(s_all["produced"]),
               "P_sd_needed": kbridge * np.sqrt(needed_var),
               "ratio": np.sqrt(s_all["produced"] / needed_var),
               "sigma2_mean": float(s2g.mean()), "sigma2_sd": float(s2g.std()),
               "E_min_TR": float(e_mix.mean()),
               "E_min_gap_s": float(e_mix.mean() - actual_mean),
               "implied_poss_delta": float(1200.0 / e_mix.mean()
                                           - 1200.0 / actual_mean),
               "within_ratio": s_all["within_ratio"],
               "between_ratio": s_all["between_ratio"]}
        rows.append(row)
        for q, sub in g.groupby("tq"):
            aa = d1._arrs(sub)
            st = d1._stats(aa)
            bt = d1.boot(aa, a.nboot, BOOT_SEED + 3)
            resp.append({"arm": arm, "tempo_quintile": q,
                         "n_games": int(len(sub)), "ratio": st["ratio"],
                         "ratio_se": bt["ratio_se"], "ratio_lo": bt["ratio_lo"],
                         "ratio_hi": bt["ratio_hi"],
                         "band_exceed_se": max(st["ratio"] - 1.15,
                                               0.85 - st["ratio"], 0.0)
                         / bt["ratio_se"],
                         "needed_tau2": st["needed_tau2"],
                         "produced_between": st["produced_between"],
                         "between_ratio": st["between_ratio"],
                         "within_ratio": st["within_ratio"]})
        log(f"{arm}: ratio {row['ratio']:.4f} E[min] {row['E_min_TR']:.4f} "
            f"gap {row['E_min_gap_s']:+.4f} poss {row['implied_poss_delta']:+.4f}")

    res = pd.DataFrame(rows); rr = pd.DataFrame(resp)
    res.to_csv(OUT / "v5c_bakeoff_F2.csv", index=False)
    rr.to_csv(OUT / "v5c_responsiveness_F2.csv", index=False)
    (OUT / "v5c_params.json").write_text(json.dumps(params, indent=2,
                                                    default=str), encoding="utf-8")
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(res.to_string(index=False)); print()
        print(rr.pivot(index="arm", columns="tempo_quintile",
                       values="ratio").to_string()); print()
        print(rr.pivot(index="arm", columns="tempo_quintile",
                       values="band_exceed_se").to_string()); print()
        print(rr.to_string(index=False))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
