#!/usr/bin/env python
"""
exp_own_ratings_day1_priors_v1.py -- Lane N (2026-09-30): day-1 prior bake-off for our own ridge team ratings.

Pre-registration: docs/models/own_ratings/experiments.md section 1 (committed 7cdb01e before this ran).

For every season S in {2023, 2024, 2025} and every date D in weeks 0-7 of S, re-solves the efficiency ridge exactly as
cbb_sim.ratings.own_ratings.fit_season does (same accumulator, same row order, same re-centring), but with several
prior right-hand sides at once. The as-of solution, the re-centring and the predicted margin are LINEAR in the prior,
so every arm's prediction is a linear combination of basis predictions:

  m0   : fixed-term priors only (intercept/home/away from S-1 final), team prior 0          (arm Z)
  m1   : team prior = c_prev (S-1 final off/def)                                            (R = m0 + 0.8 m1)
  m2   : team prior = cont0 * c_prev (cont with NaN -> 0)
  m3   : team prior = miss * c_prev (cont missing)
  m4   : team prior = coach_change * c_prev
  m5   : team prior = cm (season-S conference mean of c_prev)
at lambda 5, and m0 + 0.8 m1 at every lambda of the D grid.

Tempo (predicted possessions) is the library's current-rule as-of tempo prediction in every arm.
The R arm is asserted equal to the library's pred_off_eff (home minus away) to 1e-8.
Fitted parameters use ONLY the fold's training seasons. 2026 is never loaded.
Writes results/own_ratings_day1/preds_v1.parquet, params_v1.json, day1_v1.parquet.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd
from scipy import linalg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

from cbb_sim.ratings import own_ratings as orat  # noqa: E402

OUT = REPO / "results/own_ratings_day1"
SEASONS = [2022, 2023, 2024, 2025]
LAM = 5.0
W_R = 0.8
LAM_GRID = [0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0]
N_WEEKS = 8
FOLDS = {"F1": {"train": [2023], "test": 2024}, "F2": {"train": [2023, 2024], "test": 2025}}
BASIS = ["m0", "m1", "m2", "m3", "m4", "m5"]


def band_of(week: int) -> str:
    return "0-1" if week <= 1 else ("2-3" if week <= 3 else "4-7")


def conference_map(season: int) -> pd.Series:
    s = pd.read_parquet(f"data/raw/hoopr/schedules/mbb_schedule_{season}.parquet",
                        columns=["home_id", "away_id", "home_conference_id", "away_conference_id"])
    a = pd.concat([s[["home_id", "home_conference_id"]].set_axis(["team_id", "conf"], axis=1),
                   s[["away_id", "away_conference_id"]].set_axis(["team_id", "conf"], axis=1)]).dropna()
    a["team_id"] = a["team_id"].astype("int64")
    return a.groupby("team_id")["conf"].agg(lambda x: x.mode().iloc[0])


def season_basis(tg: pd.DataFrame, season: int, prior_final: dict, run_R, cont: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s, games, teams = orat._season_frames(tg, season)
    t_index = {int(t): i for i, t in enumerate(teams)}
    n_t = len(teams)
    F = orat.EFF_FIXED
    n_p = F + 2 * n_t
    eff = orat.RidgeProblem(n_p)

    off_prev_s = prior_final["team_off"]
    def_prev_s = prior_final["team_def"]
    has_prev = pd.Series(teams).isin(off_prev_s.index).to_numpy()
    off_prev = off_prev_s.reindex(teams).fillna(0.0).to_numpy(float)
    def_prev = def_prev_s.reindex(teams).fillna(0.0).to_numpy(float)
    cs = cont[cont["season"] == season].set_index("team_id")
    cont_raw = cs["cont"].reindex(teams).to_numpy(float)
    miss = ~np.isfinite(cont_raw)
    cont0 = np.where(miss, 0.0, cont_raw)
    coach = cs["coach_change"].reindex(teams).fillna(0.0).to_numpy(float)
    conf = conference_map(season).reindex(teams)
    cm_off = np.zeros(n_t)
    cm_def = np.zeros(n_t)
    df = pd.DataFrame({"conf": conf.to_numpy(), "o": off_prev, "d": def_prev, "hp": has_prev})
    g = df[df["hp"] & df["conf"].notna()].groupby("conf")[["o", "d"]].mean()
    ok = df["conf"].notna().to_numpy() & df["conf"].isin(g.index).to_numpy()
    cm_off[ok] = g["o"].reindex(df.loc[ok, "conf"]).to_numpy()
    cm_def[ok] = g["d"].reindex(df.loc[ok, "conf"]).to_numpy()

    p_fixed = np.zeros(n_p)
    p_fixed[0], p_fixed[1], p_fixed[2] = prior_final["intercept"], prior_final["home"], prior_final["away"]

    def team_vec(o, d):
        v = np.zeros(n_p)
        v[F:F + n_t] = o
        v[F + n_t:] = d
        return v

    P = np.column_stack([
        p_fixed,
        team_vec(off_prev, def_prev),
        team_vec(cont0 * off_prev, cont0 * def_prev),
        team_vec(miss * off_prev, miss * def_prev),
        team_vec(coach * off_prev, coach * def_prev),
        team_vec(cm_off, cm_def),
    ])

    s = s.sort_values(["game_date", "game_id", "team_id"], kind="mergesort")
    off_col = s["team_id"].map(t_index).to_numpy() + F
    def_col = s["opp_team_id"].map(t_index).to_numpy() + F + n_t
    eff_idx = np.column_stack([np.zeros(len(s), dtype=int), np.where(s["site_home"].to_numpy() > 0, 1, -1),
                               np.where(s["site_away"].to_numpy() > 0, 2, -1), off_col, def_col])
    eff_val = np.ones_like(eff_idx, dtype=float)
    eff_y = s["off_eff"].to_numpy(float)
    eff_dates = s["game_date"].to_numpy()

    first = pd.Timestamp(games["game_date"].min())
    games = games.copy()
    games["week"] = (games["game_date"] - first).dt.days // 7
    poss = pd.Series(run_R.pred_poss, index=run_R.game_rows["game_id"].to_numpy())
    # library R prediction (home row minus away row) for the parity assert
    er = run_R.eff_rows.assign(p=run_R.pred_off_eff)
    lib_h = er[er["team_id"] == er["home_team_id"]].set_index("game_id")["p"]
    lib_a = er[er["team_id"] == er["away_team_id"]].set_index("game_id")["p"]

    dates = np.unique(games["game_date"].to_numpy())
    dates = dates[(pd.to_datetime(dates) - first).days // 7 < N_WEEKS]
    out = []
    ptr = 0
    day1 = None
    for d in dates:
        assert ptr == 0 or eff_dates[ptr - 1] < d          # strictly before D
        e_end = ptr
        while e_end < len(s) and eff_dates[e_end] == d:
            e_end += 1
        gd = games[games["game_date"] == d]
        hi = gd["home_team_id"].map(t_index).to_numpy()
        ai = gd["away_team_id"].map(t_index).to_numpy()
        neu = gd["neutral_site"].to_numpy().astype(bool)
        ph = poss.reindex(gd["game_id"]).to_numpy()
        rec = {"game_id": gd["game_id"].to_numpy(), "season": season, "game_date": gd["game_date"].to_numpy(),
               "week": gd["week"].to_numpy(), "home_team_id": gd["home_team_id"].to_numpy(),
               "away_team_id": gd["away_team_id"].to_numpy(), "neutral_site": neu, "poss_hat": ph,
               "new_d1": ~has_prev[hi] | ~has_prev[ai]}

        def margins(B):
            off = B[F:F + n_t] - B[F:F + n_t].mean(axis=0)
            dfn = B[F + n_t:] - B[F + n_t:].mean(axis=0)
            site = (B[1] - B[2])[None, :] * (~neu)[:, None]
            diff = off[hi] + dfn[ai] - off[ai] - dfn[hi] + site
            return ph[:, None] * diff / 100.0, off, dfn

        for lam in LAM_GRID:
            A = eff.xtx + lam * np.eye(n_p)
            c = linalg.cho_factor(A, lower=True, check_finite=False)
            if lam == LAM:
                rhs = np.column_stack([eff.xty + lam * P[:, 0]] + [lam * P[:, k] for k in range(1, P.shape[1])])
                B = linalg.cho_solve(c, rhs, check_finite=False)
                M, off, dfn = margins(B)
                for k, nm in enumerate(BASIS):
                    rec[nm] = M[:, k]
                if day1 is None:
                    net = (off[:, 0] + W_R * off[:, 1]) - (dfn[:, 0] + W_R * dfn[:, 1])
                    day1 = pd.DataFrame({"season": season, "team_id": teams, "net_R_day1": net, "has_prev": has_prev,
                                         "conf": conf.to_numpy(), "cont": cont_raw, "coach_change": coach})
                # parity with the library's R
                rR = M[:, 0] + W_R * M[:, 1]
                lib = (lib_h.reindex(gd["game_id"]).to_numpy() - lib_a.reindex(gd["game_id"]).to_numpy()) * ph / 100.0
                assert np.nanmax(np.abs(rR - lib)) < 1e-8, (season, d, np.nanmax(np.abs(rR - lib)))
            else:
                rhs = np.column_stack([eff.xty + lam * (P[:, 0] + W_R * P[:, 1])])
                B = linalg.cho_solve(c, rhs, check_finite=False)
                M, _, _ = margins(B)
                rec[f"D_lam{lam:g}"] = M[:, 0]
        r5 = rec["m0"] + W_R * rec["m1"]
        rec["D_lam5"] = r5
        out.append(pd.DataFrame(rec))
        if e_end > ptr:
            eff.add_rows(eff_idx[ptr:e_end], eff_val[ptr:e_end], eff_y[ptr:e_end])
            ptr = e_end
    return pd.concat(out, ignore_index=True), day1


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    uni = orat.load_universe()
    uni = uni[uni["season"].isin(SEASONS)]
    tg = orat.load_team_games(uni, SEASONS)
    cont = pd.read_parquet("results/team_rate_estimator/continuity_v2.parquet")
    assert not (tg["season"] >= 2026).any()
    runs, prior_e, prior_t = {}, None, None
    for s in SEASONS:
        runs[s] = orat.fit_season(tg, s, LAM, LAM, prior_e, prior_t, W_R, W_R)
        prior_e, prior_t = runs[s].final["eff"], runs[s].final["tempo"]
    basis, day1 = [], []
    for s in SEASONS[1:]:
        b, d1 = season_basis(tg, s, runs[s - 1].final["eff"], runs[s], cont)
        basis.append(b)
        day1.append(d1)
        print(f"season {s}: {len(b)} games weeks 0-7, {b['new_d1'].sum()} with a new-D-I team, {time.time()-t0:.0f}s", flush=True)
    basis = pd.concat(basis, ignore_index=True)
    day1 = pd.concat(day1, ignore_index=True)

    fin = pd.read_parquet("data/processed/truth/game_finals_v2.parquet", columns=["game_id", "home_score", "away_score"])
    fin["margin"] = fin["home_score"] - fin["away_score"]
    n0 = len(basis)
    basis = basis.merge(fin[["game_id", "margin"]], on="game_id", how="inner")
    print(f"dropped {n0 - len(basis)} games without a verified final")
    basis = basis[np.isfinite(basis["poss_hat"])]
    basis.to_parquet(OUT / "basis_v1.parquet", index=False)

    # ---- fit per fold on training seasons only, build arm predictions ----
    rows, params = [], {}
    for fold, spec in FOLDS.items():
        tr = basis[basis["season"].isin(spec["train"])]
        y = tr["margin"].to_numpy()
        base = tr["m0"].to_numpy()
        # continuity centring constant: training mean of cont over teams with a value
        cm_ = day1[day1["season"].isin(spec["train"])]["cont"]
        mcont = float(np.nanmean(cm_))

        def kcols(df):
            return np.column_stack([df["m1"], df["m2"] - mcont * (df["m1"] - df["m3"]), df["m4"]])

        w_F = float(np.linalg.lstsq(tr[["m1"]].to_numpy(), y - base, rcond=None)[0][0])
        xc = (tr["m1"] - tr["m5"]).to_numpy()[:, None]
        w_C = float(np.linalg.lstsq(xc, y - base - tr["m5"].to_numpy(), rcond=None)[0][0])
        r = np.linalg.lstsq(kcols(tr), y - base, rcond=None)[0]
        lam_b = {}
        for band in ["0-1", "2-3", "4-7"]:
            m = tr["week"].map(band_of) == band
            sse = {lam: float(((tr.loc[m, "margin"] - tr.loc[m, f"D_lam{lam:g}"]) ** 2).sum()) for lam in LAM_GRID}
            lam_b[band] = min(sse, key=sse.get)
        params[fold] = {"w_F": w_F, "w_C": w_C, "K_r0_r1_r2": [float(v) for v in r], "K_cont_train_mean": mcont,
                        "D_lambda_by_band": lam_b, "train_seasons": spec["train"], "test_season": spec["test"]}
        sub = basis[basis["season"].isin(spec["train"] + [spec["test"]])].copy()
        preds = {
            "Z": sub["m0"],
            "R": sub["m0"] + W_R * sub["m1"],
            "F": sub["m0"] + w_F * sub["m1"],
            "C": sub["m0"] + sub["m5"] + w_C * (sub["m1"] - sub["m5"]),
            "K": sub["m0"] + kcols(sub) @ r,
            "D": pd.Series([sub.iloc[i][f"D_lam{lam_b[band_of(int(wk))]:g}"] for i, wk in enumerate(sub["week"])], index=sub.index),
        }
        keep = ["season", "game_id", "game_date", "week", "home_team_id", "away_team_id", "neutral_site", "new_d1", "margin"]
        for arm, p in preds.items():
            o = sub[keep].copy()
            o["fold"] = fold
            o["role"] = np.where(o["season"] == spec["test"], "test", "train")
            o["arm"] = arm
            o["pred"] = p.to_numpy()
            rows.append(o)
        print(fold, json.dumps(params[fold]))
    pr = pd.concat(rows, ignore_index=True)
    pr.to_parquet(OUT / "preds_v1.parquet", index=False)
    day1.to_parquet(OUT / "day1_v1.parquet", index=False)
    (OUT / "params_v1.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"done {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
