#!/usr/bin/env python
"""
train_foul_team_v1.py -- possession-outcome foul-accrual ROUND 2 (experiments.md section 34, pre-registered and pushed
in commit 0f83dee BEFORE this script ran).

    .venv/Scripts/python.exe scripts/train_foul_team_v1.py --n-jobs 8

Arms (34.2): A2 (round-7 GBM on STATE_T, seeds 0/7), offset arms on each seed's A2 logit
  A2t   : a + b_d D + b_o O
  A2tn  : A2t + c_d D u_d + c_o O u_o + g Lv
  A2tnc : A2tn + e_k (days buckets 0-3; d46+ reference)
and the diagnostic ceiling A2tG (GBM on STATE_T + [D, O, u_d, u_o, Lv, dss_bkt]).
Features: `build_foul_team_feats_v1` (one function for training rows and engine slates).

Writes data/processed/models/possession_outcome/round11team/
  preds_{F1,F2}_seed{0,7}.parquet, coef_{arm}_{fold}.json (seed-0 TRAIN fits), slate_feats_{2024,2025}.parquet,
  train_meta.json
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_foul_team_feats_v1 as BF  # noqa: E402
import train_foul_joint_v1 as FJ  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from train_foul_cal_v1 import dss_bucket  # noqa: E402

OUT = ROOT / "data/processed/models/possession_outcome/round11team"
SEEDS = (0, 7)
N0_GRID = [0.0] + [float(10 ** (k / 2)) for k in range(2, 17)] + [np.inf]     # 0, 1e1, 3.16e1, ..., 1e8, inf
OFFSET_ARMS = {
    "A2t": ["one", "D", "O"],
    "A2tn": ["one", "D", "O", "Du", "Ou", "Lv"],
    "A2tnc": ["one", "D", "O", "Du", "Ou", "Lv", "b0", "b1", "b2", "b3"],
}
GBM_FEATS = FJ.STATE_T + ["D", "O", "u_d", "u_o", "Lv", "dss_bkt"]
SLATES = {2025: ("F2", "data/processed/models/engine_v3", "F2_2025"),
          2024: ("F1", "data/processed/models/engine_v3_f1", "F1_2024")}


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_offset(off: np.ndarray, y: np.ndarray, X: np.ndarray) -> dict:
    from scipy.optimize import minimize
    y = y.astype(float)

    def nll(beta):
        z = off + X @ beta
        return float(np.sum(np.logaddexp(0, z) - y * z)), X.T @ (expit(z) - y)

    r = minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    return {"beta": r.x.tolist(), "nll": float(r.fun), "converged": bool(r.success), "n": int(len(y))}


def design_X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    m = {"one": np.ones(len(df)), "D": df["D"].to_numpy(), "O": df["O"].to_numpy(),
         "Du": (df["D"] * df["u_d"]).to_numpy(), "Ou": (df["O"] * df["u_o"]).to_numpy(), "Lv": df["Lv"].to_numpy()}
    for b in range(4):
        m[f"b{b}"] = (df["dss_bkt"].to_numpy() == b).astype(float)
    return np.column_stack([m[c] for c in cols]).astype("float64")


def gbm_task(Xtr, ytr, Xq_list, seed):
    m = FJ.fit_gbm(Xtr, ytr, seed)
    return [m.predict_proba(X)[:, 1] for X in Xq_list], m


def attach_team(d: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    d["gdate"] = pd.to_datetime(d["game_date"]).dt.normalize()
    for side, col in (("d", "defense_team_id"), ("o", "offense_team_id")):
        q = d[["season", col, "gdate"]].drop_duplicates().rename(columns={col: "team_id", "gdate": "game_date"})
        q = q.reset_index(drop=True)
        f = BF.team_priors_asof(tg, q)
        q = pd.concat([q, f], axis=1).rename(columns={"team_id": col, "game_date": "gdate"})
        keep = {"d": {"D": "D", "u": "u_d", "n": "n_d"}, "o": {"O": "O", "u": "u_o", "n": "n_o"}}[side]
        d = d.merge(q[["season", col, "gdate"] + list(keep)].rename(columns=keep), on=["season", col, "gdate"],
                    how="left", validate="many_to_one")
    return d


def choose_n0(daily, d_tr, lbar, prior_by_season, seasons_with_prior) -> tuple[float, dict]:
    w = d_tr[(d_tr["in_fit_window"] == 1) & d_tr["season"].isin(seasons_with_prior)]
    g = w.groupby(["season", "gdate"])["y_nt"].agg(["sum", "size"]).reset_index().rename(columns={"gdate": "game_date"})
    ll = {}
    for n0 in N0_GRID:
        L = BF.level_asof(daily, g[["season", "game_date"]], n0, prior_by_season)
        L = np.clip(L, 1e-9, 1 - 1e-9)
        ll[str(n0)] = float(np.sum(g["sum"] * np.log(L) + (g["size"] - g["sum"]) * np.log(1 - L)))
    best = max(N0_GRID, key=lambda n: ll[str(n)])
    return best, ll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = FJ.possession_design()
    d["dss"] = d["days_since_start"].astype("float64")
    d["dss_bkt"] = dss_bucket(d["dss"].to_numpy()).astype("float32")
    tg = BF.team_games()
    d = attach_team(d, tg)
    for c in ("D", "O", "u_d", "u_o"):
        if d[c].isna().any():
            raise SystemExit(f"{c} has NaN rows after merge")
    daily = BF.league_daily_y(d)
    seas_rate = d[d["in_fit_window"] == 1].groupby("season")["y_nt"].mean()
    print(f"design {len(d):,} rows ({time.time() - t0:.0f}s)", flush=True)
    meta = {"pre_registration": "possession_outcome experiments.md section 34 (commit 0f83dee)", "gbm": FJ.GBM_KW,
            "seeds": SEEDS, "K_poss": BF.PRIOR_GAMES, "folds": {}}
    parts, tasks, keys = {}, [], []
    for fold, sp in FJ.FOLDS.items():
        assert_not_sealed(sp["train"], context=f"{fold} train")
        assert_not_sealed(sp["test"], context=f"{fold} test")
        tr_all = d[d["season"].isin(sp["train"])]
        lbar = float(tr_all.loc[tr_all["in_fit_window"] == 1, "y_nt"].mean())
        prior = {int(s): (float(seas_rate[s - 1]) if (s - 1) in seas_rate.index and (s - 1) >= min(sp["train"]) else lbar)
                 for s in sp["train"] + sp["test"]}
        n0, ll_grid = choose_n0(daily, tr_all, lbar, prior, [s for s in sp["train"] if s - 1 >= min(sp["train"])])
        dd = d[d["season"].isin(sp["train"] + sp["test"])].copy()
        L = BF.level_asof(daily, dd[["season", "gdate"]].rename(columns={"gdate": "game_date"}), n0, prior)
        dd["Lv"] = logit(L) - logit(lbar)
        tr = dd[dd["season"].isin(sp["train"]) & (dd["in_fit_window"] == 1)]
        te = dd[dd["season"].isin(sp["test"])]
        parts[fold] = (tr, te)
        meta["folds"][fold] = {"lbar": lbar, "prior_by_season": prior, "n0": n0, "n0_loglik": ll_grid,
                               "Lv_test_by_bkt": te.groupby("dss_bkt")["Lv"].mean().round(4).tolist(),
                               "Lv_train_by_season": tr.groupby("season")["Lv"].mean().round(4).to_dict()}
        print(f"{fold}: lbar {lbar:.5f} n0 {n0} prior {prior}", flush=True)
        for arm, ft in (("A2", FJ.STATE_T), ("A2tG", GBM_FEATS)):
            Xtr = tr[ft].to_numpy("float32")
            for s in SEEDS:
                tasks.append(delayed_task(Xtr, tr["y_nt"].to_numpy(), [Xtr, te[ft].to_numpy("float32")], s))
                keys.append((fold, arm, s))
    from joblib import Parallel
    print(f"{len(tasks)} GBM fits on {a.n_jobs} workers", flush=True)
    res = dict(zip(keys, Parallel(n_jobs=a.n_jobs, backend="loky")(tasks)))
    print(f"GBM fits done ({time.time() - t0:.0f}s)", flush=True)
    for fold, (tr, te) in parts.items():
        meta["folds"][fold]["offset"] = {}
        for s in SEEDS:
            (p_tr, p_te), m = res[(fold, "A2", s)]
            (_, pg_te), _mg = res[(fold, "A2tG", s)]
            o_tr, o_te = logit(p_tr), logit(p_te)
            out = te[["game_id", "season", "period", "poss_index", "offense_team_id", "defense_team_id",
                      "in_fit_window", "site_home", "site_away", "def_f_t", "off_f_t", "dss", "y_nt",
                      "D", "O", "n_d", "n_o", "Lv"]].copy()
            out["p__A2"] = p_te
            for arm, cols in OFFSET_ARMS.items():
                prm = fit_offset(o_tr, tr["y_nt"].to_numpy(), design_X(tr, cols))
                prm["cols"] = cols
                meta["folds"][fold]["offset"][f"{arm}_s{s}"] = prm
                print(f"{fold} s{s} {arm}: " + ", ".join(f"{c}={b:+.4f}" for c, b in zip(cols, prm["beta"])), flush=True)
                out[f"p__{arm}"] = expit(o_te + design_X(te, cols) @ np.asarray(prm["beta"]))
                if s == 0:
                    cj = {"arm": arm, "fold": fold, "cols": cols, "beta": prm["beta"], "n0": n0,
                          "lbar": meta["folds"][fold]["lbar"], "prior_by_season": meta["folds"][fold]["prior_by_season"],
                          "dss_edges": [7, 14, 30, 45], "K_prior_games": BF.PRIOR_GAMES,
                          "pre_registration": meta["pre_registration"]}
                    (OUT / f"coef_{arm}_{fold}.json").write_text(json.dumps(cj, indent=1), encoding="utf-8")
            out["p__A2tG"] = pg_te
            out.to_parquet(OUT / f"preds_{fold}_seed{s}.parquet", index=False)
    # engine slates: the same builder, as-of each game's date; Lv with the slate fold's n0 / prior
    meta["slates"] = {}
    for season, (fold, idir, tag) in SLATES.items():
        from cbb_sim.engine.inputs import EngineInputs
        inp = EngineInputs.load(idir, tag)
        g = inp.games[["game_id", "season", "game_date", "home_team_id", "away_team_id"]].copy()
        g["game_date"] = pd.to_datetime(g["game_date"]).dt.normalize()
        sl = g[["game_id"]].copy()
        for side, col in (("h", "home_team_id"), ("a", "away_team_id")):
            f = BF.team_priors_asof(tg, g[["season", col, "game_date"]].rename(columns={col: "team_id"}))
            for c in ("D", "O", "u", "n"):
                sl[f"{c}_{side}"] = f[c].to_numpy()
        fm = meta["folds"][fold]
        sl["Lv"] = logit(BF.level_asof(daily, g[["season", "game_date"]], fm["n0"], fm["prior_by_season"])) - logit(fm["lbar"])
        dss = inp.team_static[:, 0, inp.team_names["days_since_start"]].astype(float)
        sl["dss_bkt"] = dss_bucket(dss)
        sl.to_parquet(OUT / f"slate_feats_{season}.parquet", index=False)
        # identity vs training rows on matched games (defence side home)
        chk = d[(d["season"] == season)][["game_id", "defense_team_id", "D", "O"]].drop_duplicates(["game_id", "defense_team_id"])
        chk = chk.merge(g[["game_id", "home_team_id"]], on="game_id")
        chk = chk[chk["defense_team_id"] == chk["home_team_id"]].merge(sl[["game_id", "D_h"]], on="game_id")
        meta["slates"][season] = {"n_games": int(len(sl)), "n_matched_design": int(len(chk)),
                                  "max_abs_D_diff_vs_design": float((chk["D"] - chk["D_h"]).abs().max()),
                                  "share_D_zero": float((sl["D_h"] == 0).mean())}
        print(f"slate {season}: {meta['slates'][season]}", flush=True)
    meta["seconds"] = round(time.time() - t0, 1)
    (OUT / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")


def delayed_task(Xtr, ytr, Xq, s):
    from joblib import delayed
    return delayed(gbm_task)(Xtr, ytr, Xq, s)


if __name__ == "__main__":
    main()
