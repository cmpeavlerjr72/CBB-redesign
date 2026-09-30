#!/usr/bin/env python
"""
exp_foul_whistle_var_v1.py -- round 8 (possession_outcome experiments.md s22.2 as amended by
s23): FIT the variance of the shared per-game whistle latent on TRAINING seasons only.

Per fold: the round-7 specs refit on the fold's train window on the v2 (engine-definition)
state -- `A2` (non-trip accrual, GBM) and `T0` + `T2c` offsets (both FT-trip classes). For every
team-game (the team as the FOULING side = the defence):
    W = non-trip events (y_nt) on its defensive possessions + FT trips (bonus + shooting)
        awarded against it,
    E = the same models' probabilities summed over the same rows,
    r = W / E - 1.
c = cov(r_home, r_away) across games (independent sampling noise cancels in the covariance);
s^2 = ln(1 + c) so that A = exp(s z - s^2/2) has E[A] = 1 and Var[A] = c.
Game-bootstrap SE (200). Train seasons in-sample (the fit); the test season out of sample is
the pre-registered CHECK (within 2 SE). A fit-window-only (regulation outside the final 2:00)
version is reported as a sensitivity.

    .venv/Scripts/python.exe scripts/exp_foul_whistle_var_v1.py
Writes data/processed/models/possession_outcome/round7/whistle_var_v1.json (gitignored dir) and
a copy results/foul_joint/whistle_var_v1.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import train_foul_joint_v1 as TJ  # noqa: E402

OUT = ROOT / "data/processed/models/possession_outcome/round7"
FOLDS = TJ.FOLDS
NBOOT = 200


def team_game(poss: pd.DataFrame, ch: pd.DataFrame, mask_poss, mask_ch) -> pd.DataFrame:
    a = poss[mask_poss].groupby(["season", "game_id", "defense_team_id"]).agg(
        W=("y_nt", "sum"), E=("p_nt", "sum"))
    b = ch[mask_ch].assign(yt=ch["y_bonus"] + ch["y_shoot"], pt=ch["p_b"] + ch["p_s"]).groupby(
        ["season", "game_id", "defense_team_id"]).agg(W=("yt", "sum"), E=("pt", "sum"))
    t = a.add(b, fill_value=0).reset_index()
    t["r"] = t["W"] / t["E"] - 1.0
    return t


def cov_stats(t: pd.DataFrame, home: pd.Series, rng) -> dict:
    t = t.merge(home.rename("home_team_id"), left_on="game_id", right_index=True)
    h = t[t["defense_team_id"] == t["home_team_id"]].set_index("game_id")
    w = t[t["defense_team_id"] != t["home_team_id"]].drop_duplicates("game_id").set_index("game_id")
    j = h[["r", "E"]].join(w[["r", "E"]], lsuffix="_h", rsuffix="_a", how="inner")
    rh, ra = j["r_h"].to_numpy(), j["r_a"].to_numpy()

    def c_of(ix):
        return float(np.cov(rh[ix], ra[ix])[0, 1])
    n = len(j)
    c = c_of(np.arange(n))
    boots = [c_of(rng.integers(0, n, n)) for _ in range(NBOOT)]
    var_r = float(np.var(np.concatenate([rh, ra]), ddof=1))
    pois = float(np.mean(np.concatenate([1 / j["E_h"].to_numpy(), 1 / j["E_a"].to_numpy()])))
    return {"n_games": int(n), "c": c, "se": float(np.std(boots, ddof=1)),
            "corr": float(np.corrcoef(rh, ra)[0, 1]), "var_r": var_r,
            "poisson_part": pois, "extra_marginal_var": var_r - pois,
            "s2": float(np.log1p(max(c, 0.0)))}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(20260930)
    d = TJ.possession_design()
    ch, base = TJ.chance_design()
    u = pd.read_parquet(ROOT / "data/processed/games_universe.parquet", columns=["game_id", "home_team_id"])
    home = u.drop_duplicates("game_id").set_index("game_id")["home_team_id"]
    from joblib import Parallel, delayed
    tasks = []
    for fold, sp in FOLDS.items():
        tr = d[d["season"].isin(sp["train"]) & (d["in_fit_window"] == 1)]
        tasks.append(delayed(TJ.fit_gbm)(tr[TJ.STATE_T].to_numpy("float32"), tr["y_nt"].to_numpy(), 0))
    models = dict(zip(FOLDS, Parallel(n_jobs=2, backend="loky")(tasks)))
    print(f"A2 fits {time.time() - t0:.0f}s", flush=True)
    out = {"definition": __doc__.split("\n\n")[1], "folds": {}}
    for fold, sp in FOLDS.items():
        seasons = sp["train"] + sp["test"]
        P = d[d["season"].isin(seasons)].copy()
        P["p_nt"] = models[fold].predict_proba(P[TJ.STATE_T].to_numpy("float32"))[:, 1]
        C = ch[ch["season"].isin(seasons)].copy()
        trc = C[C["season"].isin(sp["train"]) & (C["in_fit_window"] == 1)]
        Xall_t = C[base].assign(in_bonus=C["inb_t"])[base].to_numpy("float64")
        for y, col in (("y_bonus", "p_b"), ("y_shoot", "p_s")):
            (ptr_l, ptr_t, pall), _ = TJ.t0_task(trc[base].to_numpy("float64"), trc[y].to_numpy(),
                                                [trc[base].to_numpy("float64"),
                                                 trc[base].assign(in_bonus=trc["inb_t"])[base].to_numpy("float64"),
                                                 Xall_t])
            dl = TJ.fit_offsets(TJ.t_cells(trc, "t", "T2c"), trc[y].to_numpy(), TJ.logit(ptr_t))
            dd = pd.Series(TJ.t_cells(C, "t", "T2c")).map(dl).fillna(0.0).to_numpy()
            C[col] = 1.0 / (1.0 + np.exp(-(TJ.logit(pall) + dd)))
        res = {}
        for label, seas in (("train", sp["train"]), ("test", sp["test"])):
            for mname, mp, mc in (("all_rows", np.ones(len(P), bool), np.ones(len(C), bool)),
                                  ("fit_window", P["in_fit_window"].to_numpy() == 1,
                                   C["in_fit_window"].to_numpy() == 1)):
                sp_ = P["season"].isin(seas).to_numpy() & mp
                sc_ = C["season"].isin(seas).to_numpy() & mc
                res[f"{label}_{mname}"] = cov_stats(team_game(P, C, sp_, sc_), home, rng)
        tr_, te_ = res["train_all_rows"], res["test_all_rows"]
        res["check_test_within_2se"] = bool(abs(te_["c"] - tr_["c"]) <= 2 * tr_["se"])
        out["folds"][fold] = res
        print(fold, json.dumps({k: (v if not isinstance(v, dict) else
                                    {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in v.items()})
                                for k, v in res.items()}), flush=True)
    out["served_s2_F2"] = out["folds"]["F2"]["train_all_rows"]["s2"]
    out["seconds"] = round(time.time() - t0, 1)
    for p in (OUT / "whistle_var_v1.json", ROOT / "results/foul_joint/whistle_var_v1.json"):
        p.write_text(json.dumps(out, indent=1))
    print("served s2 (F2 train):", out["served_s2_F2"])


if __name__ == "__main__":
    main()
