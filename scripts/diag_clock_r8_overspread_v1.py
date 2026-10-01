"""diag_clock_r8_overspread_v1.py -- clock round 8 step 1 (lane H, 2026-10-01): where does
round-7 arm A2's extra GAME-LEVEL possession variance come from, and does the pace x
efficiency channel have a clock-structure owner? READ-ONLY: fits nothing that is served.

Sections (results/clock_r8/diag_overspread.json):
  A  sim variance budget (full size 5,705 x 200, L2 vs A2): Var(per-game sim mean) +
     E[within-game var] = pooled; the CALIBRATED benchmark corr^2 * Var(actual)
  B  game-level elasticity of log possessions on the served tempo sum X = log rel_h + log rel_a
     (month fixed effects): actual v4 counts, offline C0 / A2 counts (actual composition),
     sim L2 / A2 means. Ratio arm / actual = the over-response of the game level.
  C  train/serve skew: design's per-game tempo_rel vs the engine inputs' team_static
  D  log-scale vs mean-scale elasticity of possession duration (cell fixed effects,
     uncensored rows with >= 60 s, the round-7 estimation rows): OLS on log(d+0.5)
     (round 7's estimator) vs a Poisson pseudo-likelihood (mean scale), per start type;
     implied game-level elasticity of each
  E  in-season drift: actual mean duration by month (2023-2025) and the as-of league mean
  F  pace x efficiency: per-possession Cov(duration, points | start type) in the data, split
     transition / non-transition and by OREB; what it implies for the game-level
     Cov(possessions, points per possession); the sim's within-game Cov from the seeds.

    CBB_TRUTH=verified_v1 .venv/Scripts/python.exe scripts/diag_clock_r8_overspread_v1.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import reference as REF  # noqa: E402
from cbb_sim.models import clock as ck  # noqa: E402
from cbb_sim.models import clock_r7 as r7  # noqa: E402

if os.environ.get("CBB_TRUTH") != "verified_v1":
    raise SystemExit("set CBB_TRUTH=verified_v1")
EV = ROOT / "results/engine_v0"
OUT = ROOT / "results/clock_r8"
OUT.mkdir(parents=True, exist_ok=True)
L2D = ROOT / "data/processed/models/clock/r6_L2/design_v2.parquet"


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, r


def month_fe(m):
    return pd.get_dummies(pd.Series(m).astype(int), drop_first=True).to_numpy(dtype=float)


def main():
    rep = {}
    # ------------------------------------------------------------------ truth
    truth = REF.load_actual_games(2025).set_index("game_id")
    p4 = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2025.parquet",
                         columns=["game_id", "offense_team_id", "period", "duration_s", "points",
                                  "start_reason", "is_transition", "oreb_count", "start_clock"])
    nt = p4.groupby("game_id")["offense_team_id"].nunique()
    cnt4 = p4.groupby(["game_id", "offense_team_id"]).size().groupby("game_id").mean()
    u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet").set_index("game_id")
    truth["cnt_v4"] = cnt4
    truth["month"] = pd.to_datetime(truth["game_date"]).dt.month
    # ------------------------------------------------------------------ engine features
    a = np.load(ROOT / "data/processed/models/engine_v3/arrays_F2_2025.npz")
    ts = a["team_static"]
    gi = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")
    names = json.loads((ROOT / "data/processed/models/engine_v3/names_F2_2025.json").read_text())
    tn = names["team_names"] if isinstance(names["team_names"], dict) else json.loads(names["team_names"])
    eng = pd.DataFrame({"game_id": gi["game_id"].to_numpy(),
                        "rel_h": ts[:, 0, tn["off_tempo_rel"]], "rel_a": ts[:, 1, tn["off_tempo_rel"]],
                        "prior": ts[:, 0, tn["tempo_prior_game"]],
                        "dss": ts[:, 0, tn["days_since_start"]]}).set_index("game_id")
    eng["X"] = np.log(eng["rel_h"]) + np.log(eng["rel_a"])
    eng["filled"] = (eng["rel_h"] == 1.0) | (eng["rel_a"] == 1.0)
    # ------------------------------------------------------------------ sim runs
    def sim(tag):
        g = pd.read_parquet(EV / tag / "games.parquet",
                            columns=["game_id", "seed", "home_pts", "away_pts", "possessions"])
        g = g[g["seed"] < 200]
        g["tot"] = g["home_pts"] + g["away_pts"]
        g["ppp"] = g["tot"] / (2 * g["possessions"])
        return g
    runs = {"L2": sim("v3full_L2_s200_o0"), "A2": sim("laneH_v3full_A2_s200_o0")}
    cs = set(runs["L2"]["seed"]) & set(runs["A2"]["seed"])
    runs = {k: v[v["seed"].isin(cs)] for k, v in runs.items()}
    common = sorted(set(runs["L2"]["game_id"]) & set(runs["A2"]["game_id"]) & set(truth.index) & set(eng.index))
    pc = [g for g in common if g in nt.index and nt[g] == 2 and bool(u.at[g, "pbp_complete"])
          and pd.notna(truth.at[g, "cnt_v4"])]
    rep["n_games_pc"] = len(pc)
    T = truth.loc[pc]
    act = T["cnt_v4"].to_numpy(dtype=float)
    # A: variance budget
    A = {"actual_var": float(act.var()), "actual_sd": float(act.std())}
    gm = {}
    for k, g in runs.items():
        g = g[g["game_id"].isin(pc)]
        agg = g.groupby("game_id").agg(m=("possessions", "mean"), v=("possessions", "var"),
                                       pm=("ppp", "mean"))
        agg = agg.loc[pc]
        gm[k] = agg
        corr = float(np.corrcoef(agg["m"], act)[0, 1])
        A[k] = {"var_means": float(agg["m"].var(ddof=0)), "sd_means": float(agg["m"].std(ddof=0)),
                "mean_within_var": float(agg["v"].mean()),
                "pooled_var": float(agg["m"].var(ddof=0) + agg["v"].mean()),
                "corr_with_actual": corr,
                "calibrated_var_means": corr ** 2 * float(act.var()),
                "calibration_slope": float(np.polyfit(agg["m"], act, 1)[0]),
                "needed_within_var_if_calibrated": float(act.var() - corr ** 2 * act.var())}
    rep["A_variance_budget"] = A
    # B: game-level elasticity
    E = eng.loc[pc]
    X = E["X"].to_numpy()
    MF = month_fe(T["month"])
    full = np.column_stack([X, MF])
    B = {}
    rows = pd.read_parquet(ROOT / "results/clock_r7/rows_F2.parquet",
                           columns=["game_id", "duration_s", "e_C0", "e_A2"])
    off = rows.groupby("game_id").agg(a=("duration_s", "mean"), c0=("e_C0", "mean"), a2=("e_A2", "mean"))
    ys = {"actual_v4": np.log(act), "sim_L2": np.log(gm["L2"]["m"].to_numpy()),
          "sim_A2": np.log(gm["A2"]["m"].to_numpy())}
    for k, y in ys.items():
        b, r = ols(y, full)
        B[k] = {"elasticity": float(b[1]), "resid_sd_log": float(r.std()), "n": int(len(y))}
    # offline on clock-complete games (actual composition)
    oc = [g for g in pc if g in off.index]
    Eo = eng.loc[oc]
    fo = np.column_stack([Eo["X"].to_numpy(), month_fe(truth.loc[oc, "month"])])
    for k, col in (("offline_actual_dur", "a"), ("offline_C0", "c0"), ("offline_A2", "a2")):
        b, r = ols(np.log(1200.0 / off.loc[oc, col].to_numpy()), fo)
        B[k] = {"elasticity": float(b[1]), "resid_sd_log": float(r.std()), "n": int(len(oc))}
    b, _ = ols(np.log(truth.loc[oc, "cnt_v4"].to_numpy(dtype=float)), fo)
    B["actual_v4_on_offline_games"] = {"elasticity": float(b[1])}
    # by fill status and by month (actual vs A2 sim)
    for lab, msk in (("not_filled", ~E["filled"].to_numpy()), ("filled", E["filled"].to_numpy())):
        if msk.sum() > 50:
            B[f"actual_{lab}"] = float(ols(ys["actual_v4"][msk], X[msk])[0][1])
            B[f"simA2_{lab}"] = float(ols(ys["sim_A2"][msk], X[msk])[0][1])
            B[f"simL2_{lab}"] = float(ols(ys["sim_L2"][msk], X[msk])[0][1])
            B[f"n_{lab}"] = int(msk.sum())
    for mo in sorted(T["month"].unique()):
        msk = (T["month"] == mo).to_numpy()
        if msk.sum() > 100:
            B[f"month_{mo}"] = {k: float(ols(y[msk], X[msk])[0][1]) for k, y in ys.items()}
    # the part of the sim mean that is NOT the tempo sum
    for k in ("sim_L2", "sim_A2", "actual_v4"):
        bb, r = ols(ys[k], full)
        B[k]["var_tempo_part_log"] = float((bb[1] * X).var())
    rep["B_game_elasticity"] = B
    # C: train/serve skew of the tempo feature
    des = pd.read_parquet(L2D, columns=["season", "game_id", "offense_is_home", "off_tempo_rel",
                                        "def_tempo_rel", "days_since_start"])
    des = des[des["season"] == 2025]
    dh = des[des["offense_is_home"]].groupby("game_id").agg(dh=("off_tempo_rel", "first"),
                                                              da=("def_tempo_rel", "first"),
                                                              ddss=("days_since_start", "first"))
    j = eng.join(dh, how="inner")
    C = {"n_games": int(len(j)),
         "max_abs_diff_rel_h": float((j["rel_h"] - j["dh"]).abs().max()),
         "share_exact_rel_h": float(((j["rel_h"] - j["dh"]).abs() < 1e-5).mean()),
         "max_abs_diff_rel_a": float((j["rel_a"] - j["da"]).abs().max()),
         "max_abs_diff_dss": float((j["dss"] - j["ddss"]).abs().max()),
         "sd_log_rel_engine": float(np.log(j["rel_h"]).std()), "sd_log_rel_design": float(np.log(j["dh"]).std()),
         "engine_filled_games": int(eng["filled"].sum())}
    rep["C_train_serve"] = C
    # D: log vs mean elasticity (training rows of F2 = 2022-2024, and 2025 for reference)
    cols = ["season", "game_id", "period", "duration_s", "censored", "prev_end", "seconds_remaining",
            "score_diff", "off_tempo_rel", "def_tempo_rel", "month", "days_since_start", "game_date"]
    dz = pd.read_parquet(L2D, columns=cols)
    D = {}
    for lab, seasons in (("train_F2", [2022, 2023, 2024]), ("test_2025", [2025])):
        t = dz[dz["season"].isin(seasons) & (dz["period"] <= 2)].copy()
        m = (~t["censored"].to_numpy(dtype=bool)) & (t["seconds_remaining"].to_numpy() >= 60)
        t = r7.add_x(t.loc[m].copy())
        key = r7.cell_key(t)
        _, inv = np.unique(key, return_inverse=True)
        s = r7.start_code(t)
        y = t["duration_s"].to_numpy(dtype=float)
        Xd = r7.design_matrix(t, ("x_o", "x_d"))
        # (1) OLS on log, cell FE (round 7)
        ly = np.log(y + 0.5)
        cnt = np.bincount(inv).astype(float)
        lyc = ly - (np.bincount(inv, weights=ly) / cnt)[inv]
        Xc = Xd - np.stack([(np.bincount(inv, weights=Xd[:, j]) / cnt)[inv] for j in range(Xd.shape[1])], 1)
        keep = np.abs(Xc).sum(0) > 0
        b_log = np.zeros(Xd.shape[1]); b_log[keep] = np.linalg.lstsq(Xc[:, keep], lyc, rcond=None)[0]
        # (2) Poisson pseudo-likelihood with cell FE: E[y] = mu_c exp(Xb)
        b = b_log.copy() * 0.5
        for _ in range(50):
            eta = Xd @ b
            mu_c = np.bincount(inv, weights=y) / np.bincount(inv, weights=np.exp(eta))
            mu = mu_c[inv] * np.exp(eta)
            g = Xd.T @ (y - mu)
            H = (Xd * mu[:, None]).T @ Xd
            # profile out the cell FE (Schur complement)
            wsum = np.bincount(inv, weights=mu)
            Xm = np.stack([np.bincount(inv, weights=mu * Xd[:, j]) for j in range(Xd.shape[1])], 1)
            H = H - (Xm / wsum[:, None]).T @ Xm
            step = np.zeros_like(b)
            step[keep] = np.linalg.solve(H[np.ix_(keep, keep)], g[keep])
            b = b + step
            if np.abs(step).max() < 1e-9:
                break
        # (3) start-type weights for the implied game-level elasticity: share of possessions
        w = np.bincount(s, minlength=r7.N_START) / len(s)
        mean_by_s = np.bincount(s, weights=y, minlength=r7.N_START) / np.maximum(np.bincount(s, minlength=r7.N_START), 1)
        tw = w * mean_by_s / (w * mean_by_s).sum()     # time weights: elasticity of the mean duration
        bl = b_log.reshape(r7.N_START, 2); bp = b.reshape(r7.N_START, 2)
        D[lab] = {"n_rows": int(len(t)), "start_levels": list(ck.PREV_END_LEVELS),
                  "b_log_offence": bl[:, 0].round(4).tolist(), "b_log_defence": bl[:, 1].round(4).tolist(),
                  "b_mean_offence": bp[:, 0].round(4).tolist(), "b_mean_defence": bp[:, 1].round(4).tolist(),
                  "time_weights": tw.round(4).tolist(),
                  "game_elasticity_log_fit": float(-(tw * (bl[:, 0] + bl[:, 1])).sum() / 2 * 2),
                  "game_elasticity_mean_fit": float(-(tw * (bp[:, 0] + bp[:, 1])).sum() / 2 * 2)}
        # per-team-game elasticity on X = x_A + x_B: each team's offence term b_o x_own + b_d x_opp;
        # summing both offences gives (b_o + b_d)(x_A + x_B) / 2 per team-game of time, so the
        # elasticity of the possession count on X is -(time-weighted mean of (b_o + b_d)) / 2... * 2 / 2.
        # Reported as -(sum_s tw_s (b_o,s + b_d,s)) / 2, the coefficient on X = x_A + x_B.
        D[lab]["game_elasticity_log_fit"] /= 2
        D[lab]["game_elasticity_mean_fit"] /= 2
    rep["D_log_vs_mean"] = D
    # E: drift
    dd = dz[(dz["period"] <= 2)].copy()
    dd["mo"] = pd.to_datetime(dd["game_date"]).dt.month
    Edr = {}
    for sea, g in dd.groupby("season"):
        Edr[str(int(sea))] = g.groupby("mo")["duration_s"].mean().round(3).to_dict()
    rep["E_drift_actual_mean_duration_by_month"] = Edr
    # F: pace x efficiency
    p = p4[(p4["period"] <= 2) & p4["game_id"].isin(pc)].copy()
    p["y"] = p["points"].astype(float)
    p["d"] = p["duration_s"].astype(float)
    F = {}
    def ccov(df):
        dm = df["d"] - df.groupby("start_reason")["d"].transform("mean")
        ym = df["y"] - df.groupby("start_reason")["y"].transform("mean")
        return float((dm * ym).mean()), float(np.corrcoef(dm, ym)[0, 1])
    F["cov_d_y_within_start"], F["corr_d_y_within_start"] = ccov(p)
    for lab, msk in (("transition", p["is_transition"].astype(bool)), ("non_transition", ~p["is_transition"].astype(bool)),
                     ("oreb0", p["oreb_count"] == 0), ("oreb_pos", p["oreb_count"] > 0)):
        F[f"cov_{lab}"], F[f"corr_{lab}"] = ccov(p[msk])
        F[f"share_rows_{lab}"] = float(msk.mean())
    # between-group part: transition indicator alone (what the sim reproduces structurally)
    q = p.copy()
    q["d_tr"] = q.groupby(["start_reason", "is_transition"])["d"].transform("mean")
    q["y_tr"] = q.groupby(["start_reason", "is_transition"])["y"].transform("mean")
    dm = q["d_tr"] - q.groupby("start_reason")["d"].transform("mean")
    ym = q["y_tr"] - q.groupby("start_reason")["y"].transform("mean")
    F["cov_between_transition_strata"] = float((dm * ym).mean())
    q["d_or"] = q.groupby(["start_reason", "is_transition", "oreb_count"])["d"].transform("mean")
    q["y_or"] = q.groupby(["start_reason", "is_transition", "oreb_count"])["y"].transform("mean")
    dm2 = q["d_or"] - q.groupby("start_reason")["d"].transform("mean")
    ym2 = q["y_or"] - q.groupby("start_reason")["y"].transform("mean")
    F["cov_between_transition_x_oreb_strata"] = float((dm2 * ym2).mean())
    # game-level implication: Cov(N_t, e) from independent possessions = -Cov(d,y)/Dbar * (N_t / N)
    # with N = 2 N_t possessions in the game: dN/N = -dDbar/Dbar, Cov(Dbar, Ybar) = Cov(d,y)/N.
    Dbar = float(p["d"].mean())
    Nt = float(T["cnt_v4"].mean())
    F["implied_cov_Nt_e_from_coupling"] = float(-Nt * F["cov_d_y_within_start"] / (Dbar * 2 * Nt) * 1.0)
    # the actual residual covariance (against the L2 sim per-game mean, as lane B's diagnostic) and the
    # sim's within-game covariance
    e_act = (T["total"] / (2 * T["cnt_v4"])).to_numpy(dtype=float)
    for k in ("L2", "A2"):
        g = runs[k][runs[k]["game_id"].isin(pc)]
        mm = g.groupby("game_id")[["possessions", "ppp"]].transform("mean")
        w_n = g["possessions"] - mm["possessions"]
        w_e = g["ppp"] - mm["ppp"]
        F[f"sim_{k}_within_cov_Nt_e"] = float((w_n * w_e).mean())
        F[f"sim_{k}_within_corr_Nt_e"] = float(np.corrcoef(w_n, w_e)[0, 1])
        r_n = act - gm[k]["m"].to_numpy()
        r_e = e_act - gm[k]["pm"].to_numpy()
        F[f"actual_resid_cov_Nt_e_vs_{k}"] = float(np.cov(r_n, r_e)[0, 1])
        F[f"actual_resid_corr_Nt_e_vs_{k}"] = float(np.corrcoef(r_n, r_e)[0, 1])
        F[f"between_game_corr_simmean_N_e_{k}"] = float(np.corrcoef(gm[k]["m"], gm[k]["pm"])[0, 1])
    rep["F_pace_efficiency"] = F
    (OUT / "diag_overspread.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
