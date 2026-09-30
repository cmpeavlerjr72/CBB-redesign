"""exp_team_rate_estimator_v4.py -- closes the `off_3pa_c` gap like for like.

This adds 3PA per POSSESSION (`pa3`, offence and defence) with the identical E3 procedure, and emits
table v2. The procedure: an E3 deviance fit of (q, P0, rho) on training seasons, which is round 1's
E3 fit and equals round 2's V0 + P0. The PM ruling of 2026-09-30 is in
docs/models/team_rate_estimator/experiments.md section 7. The v1 tables and every v1/v2/v3 output
are untouched.

Parts:
  fit    fit pa3 for both folds -> results/team_rate_estimator/params_pa3_v4.json and
         estimates_pa3_v4.parquet (test-season rows, arm "V0", for the v3 grader's --guard)
  emit   [--opp] -> data/processed/team_rate_features_E3_v2.parquet (or E3opp_v2): 17 rates x 2 sides
         x {c, v, L}, strictly as-of (asserted, by tipoff order)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import exp_team_rate_estimator_v3 as X3  # noqa: E402
from cbb_sim.live.team_rate_estimator import RATES_V2  # noqa: E402

OUT = Path("results/team_rate_estimator")
X3.RATES = RATES_V2          # X3.build_all / opp_adj iterate this name


def fit_pa3(Ps):
    params, est = {}, []
    for side in ("off", "def"):
        P = Ps[("pa3", side)]
        for fold, fd in X3.FOLDS.items():
            tr = np.isin(P.season, fd["train"])
            cm = float(np.nanmean(P.cont[tr])) if np.isfinite(P.cont[tr]).any() else 0.0
            th, fval = X3.X2.fit(P, "E3", tr, cm)
            prm = X3.base_params(th, cm)
            params[f"{fold}|pa3|{side}"] = X3.prm_to_json(prm) | {"train_dev": fval}
            est.append(X3.rows(P, prm, P.season == fd["test"], fold, "pa3", side, "V0"))
    json.dump(params, open(OUT / "params_pa3_v4.json", "w"), indent=1)
    pd.concat(est, ignore_index=True).to_parquet(OUT / "estimates_pa3_v4.parquet", index=False)
    print(json.dumps(params, indent=1))


def merged_params():
    pm = X3.final_params("q2:P0")
    p = json.load(open(OUT / "params_pa3_v4.json"))
    for k, v in p.items():
        v = {kk: vv for kk, vv in v.items() if kk != "train_dev"}
        pm[k] = X3.prm_from_json(v)
    return pm


def emit(Ps, with_opp: bool, label: str):
    u = pd.read_parquet("data/processed/games_universe.parquet", columns=["game_id", "tipoff_utc", "season"])
    tip_all = u[u["season"] <= 2025][["game_id", "tipoff_utc"]].assign(tipoff_utc=lambda x: pd.to_datetime(x["tipoff_utc"], utc=True))
    pm = merged_params()
    wide = []
    for fold, fd in X3.FOLDS.items():
        parts = []
        for rate in RATES_V2:
            adj = X3.opp_adj(Ps, pm, fold, rate) if with_opp else {"off": None, "def": None}
            for side in ("off", "def"):
                P = Ps[(rate, side)]
                tip = P.df[["season", "team_id", "j", "game_id"]].merge(tip_all, on="game_id", how="left")
                assert tip["tipoff_utc"].notna().all(), "missing tipoff time"
                td = tip.sort_values(["season", "team_id", "j"]).groupby(["season", "team_id"])["tipoff_utc"].diff().dropna()
                assert (td > pd.Timedelta(0)).all(), "game index order disagrees with tipoff order: as-of unsafe"
                sel = (P.season <= fd["test"])[P.ts]
                c, v = X3.run_filter(P, pm[f"{fold}|{rate}|{side}"], adj[side])
                d = P.df.loc[sel, ["season", "game_id", "team_id", "game_date", "L"]].copy()
                d[f"{rate}_{side}_c"] = c[P.ts[sel], P.j[sel]]
                d[f"{rate}_{side}_v"] = v[P.ts[sel], P.j[sel]]
                d = d.rename(columns={"L": f"{rate}_{side}_L"})
                parts.append(d.set_index(["season", "game_id", "team_id", "game_date"]))
        w = pd.concat(parts, axis=1).reset_index()
        w.insert(0, "fold", fold)
        wide.append(w)
    out = pd.concat(wide, ignore_index=True)
    assert not out.duplicated(["fold", "game_id", "team_id"]).any()
    assert (out["season"] <= 2025).all() and (out.loc[out["fold"] == "F1", "season"] <= 2024).all()
    assert not out.isna().any().any()
    path = Path(f"data/processed/team_rate_features_{label}_v2.parquet")
    assert not path.exists(), f"{path} exists; write a new version instead of overwriting"
    out.to_parquet(path, index=False)
    json.dump({k: X3.prm_to_json(v) for k, v in pm.items()}, open(OUT / f"params_emit_{label}_v4.json", "w"), indent=1)
    # the v1 columns must be reproduced exactly (only pa3 is new)
    v1 = Path(f"data/processed/team_rate_features_{label}_v1.parquet")
    if v1.exists():
        a = pd.read_parquet(v1)
        m = a.merge(out, on=["fold", "season", "game_id", "team_id"], suffixes=("", "__v2"))
        worst = max(float(np.abs(m[c] - m[c + "__v2"]).max()) for c in a.columns if c.endswith(("_c", "_v", "_L")))
        print("max |v1 - v2| over the 48 shared columns:", worst)
        assert worst == 0.0
    print("wrote", path, out.shape)


def o1a_variance(Ps):
    """Stage C arm S3's variance source (section 7): O1a's estimation variance v and overdispersion phi for all
    17 rate-sides. 16 come from round 2's O1a fit (params_q1_v3.json); pa3 is fitted now with the identical
    procedure (V1 = P0 refit on first-6 Gaussian likelihood, then O1a = (P0, phi) jointly on all training rows,
    q and rho from the pa3 E3 fit). The Kalman variance path depends only on exposures, L, q, P0 and phi, never on
    outcomes, so v is as-of by construction. -> data/processed/team_rate_variance_O1a_v1.parquet"""
    p1 = json.load(open(OUT / "params_q1_v3.json")); pp = json.load(open(OUT / "params_pa3_v4.json"))
    prm = {k.rsplit("|", 1)[0]: X3.prm_from_json(v) for k, v in p1.items() if k.endswith("|O1a")}
    for side in ("off", "def"):
        P = Ps[("pa3", side)]
        for fold, fd in X3.FOLDS.items():
            tr = np.isin(P.season, fd["train"])[:, None] & P.mask
            first = tr & (np.arange(P.J)[None, :] <= X3.FIRST_J)
            V0 = X3.prm_from_json({k: v for k, v in pp[f"{fold}|pa3|{side}"].items() if k != "train_dev"})
            a0 = X3.fit_nm(lambda x: X3.gauss_nll(P, X3.replace(V0, log_p0=(x[0], 0.0, 0.0)), first), [V0.log_p0[0]])
            V1 = X3.replace(V0, log_p0=(float(a0[0]), 0.0, 0.0))
            x = X3.fit_nm(lambda x: X3.gauss_nll(P, X3.replace(V1, log_p0=(x[0], 0.0, 0.0), phi=float(np.exp(x[1]))), tr),
                          [V1.log_p0[0], 0.0])
            prm[f"{fold}|pa3|{side}"] = X3.replace(V1, log_p0=(float(x[0]), 0.0, 0.0), phi=float(np.exp(x[1])))
    json.dump({k: X3.prm_to_json(v) for k, v in prm.items()}, open(OUT / "params_o1a_v4.json", "w"), indent=1)
    wide = []
    for fold, fd in X3.FOLDS.items():
        parts = []
        for rate in RATES_V2:
            for side in ("off", "def"):
                P = Ps[(rate, side)]
                sel = (P.season <= fd["test"])[P.ts]
                _, v = X3.run_filter(P, prm[f"{fold}|{rate}|{side}"])
                d = P.df.loc[sel, ["season", "game_id", "team_id", "game_date"]].copy()
                d[f"{rate}_{side}_v_o1a"] = v[P.ts[sel], P.j[sel]]
                d[f"{rate}_{side}_phi_o1a"] = prm[f"{fold}|{rate}|{side}"].phi
                parts.append(d.set_index(["season", "game_id", "team_id", "game_date"]))
        w = pd.concat(parts, axis=1).reset_index(); w.insert(0, "fold", fold); wide.append(w)
    out = pd.concat(wide, ignore_index=True)
    assert not out.isna().any().any() and not out.duplicated(["fold", "game_id", "team_id"]).any()
    path = Path("data/processed/team_rate_variance_O1a_v1.parquet")
    assert not path.exists()
    out.to_parquet(path, index=False)
    print("wrote", path, out.shape, {k: round(v.phi, 3) for k, v in prm.items() if "pa3" in k})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["fit", "emit", "o1a"])
    ap.add_argument("--opp", action="store_true")
    a = ap.parse_args()
    Ps, _, _ = X3.build_all()
    if a.part == "fit":
        fit_pa3(Ps)
    elif a.part == "o1a":
        o1a_variance(Ps)
    else:
        emit(Ps, a.opp, "E3opp" if a.opp else "E3")


if __name__ == "__main__":
    main()
