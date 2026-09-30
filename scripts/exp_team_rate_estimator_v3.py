"""exp_team_rate_estimator_v3.py -- team-rate estimator ROUND 2.

The round is pre-registered in docs/models/team_rate_estimator/experiments.md section 5, committed as
8694748 before any fit. The E3 family is fixed. The filter is the shared module
`cbb_sim.live.team_rate_estimator`.

Parts:
  q1    variance arms V0, V1, V2 and O1 (O1a on the V1 base, O1b on the V2 base)
          -> results/team_rate_estimator/estimates_q1_v3.parquet
  q2    --base ARM: prior-mean arms P0 (E3 carry) and P1 (E3c carry) on the Q1 winner's variance
        spec, plus the diagnostic Pd (carry fitted by the first-6-games likelihood)
          -> estimates_q2_v3.parquet
  q3    --final ARM: the final arm and the final arm + opponent adjustment
          -> estimates_q3_v3.parquet
  emit  --final ARM [--opp]: data/processed/team_rate_features_<arm>_v1.parquet
        (one row per fold, season, game and team). As-of is asserted.

Hyper-parameters are fitted on training seasons only (F1: 2023; F2: 2023+2024). 2026 is SEALED and is
never loaded.

Usage:
    .venv/Scripts/python.exe scripts/exp_team_rate_estimator_v3.py --part q1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import exp_team_rate_estimator_v2 as X2  # noqa: E402  (data build + Panel; reproduced exactly)
from cbb_sim.live.team_rate_estimator import RATES, RateParams, kalman_filter, sigma2  # noqa: E402

OUT = Path("results/team_rate_estimator")
FOLDS = X2.FOLDS
FIRST_J = 5          # "first 6 games" = j in 0..5


def build_all():
    tg = X2.team_games()
    cont = X2.continuity()
    panel = X2.build_panel(tg)
    out = {}
    for rate, (num, den, fam) in RATES.items():
        lg, final = X2.league_asof(tg, num, den)
        for side in ("off", "def"):
            d = X2.side_frame(panel, num, den, side).merge(lg, on=["season", "game_date"], how="left")
            fs = d.groupby(["season", "team_id"])[[num, den]].sum()
            fs["c"] = fs[num] / fs[den] - fs.index.get_level_values("season").map(final).to_numpy()
            fs = fs.reset_index()[["season", "team_id", "c"]]
            fs["season"] = fs["season"] + 1
            carry = fs.rename(columns={"c": "c_prev"}).merge(cont[["season", "team_id", "cont", "coach_change"]],
                                                           on=["season", "team_id"], how="outer")
            out[(rate, side)] = X2.Panel(d, num, den, fam, carry)
    return out, tg, cont


def run_filter(P, prm: RateParams, adj=None):
    return kalman_filter(P.num, P.den, P.L, P.mask, P.fam, prm.q, prm.p0(P.cont, P.coach),
                         prm.rho(P.cont, P.coach) * P.cprev, prm.phi, adj)


def gauss_nll(P, prm: RateParams, rows_mask):
    c, v = run_filter(P, prm)
    L = np.nan_to_num(P.L, nan=0.5)
    m = rows_mask & (P.den > 0)
    y = P.num[m] / P.den[m]
    p = np.clip(L[m] + c[m], 1e-4, 1 - 1e-4) if P.fam == "binom" else np.clip(L[m] + c[m], 1e-4, None)
    s2 = (p * (1 - p) if P.fam == "binom" else p) / P.den[m]
    var = v[m] + prm.phi * s2
    val = float(0.5 * np.sum(np.log(var) + (y - p) ** 2 / var))
    return val if np.isfinite(val) else 1e18


def base_params(th_e3, cm):
    return RateParams(q=float(np.exp(th_e3[0])), log_p0=(float(th_e3[1]), 0.0, 0.0),
                      carry=(float(th_e3[2]), 0.0, 0.0), phi=1.0, cont_mean=cm)


def replace(prm: RateParams, **kw):
    d = dict(q=prm.q, log_p0=prm.log_p0, carry=prm.carry, phi=prm.phi, cont_mean=prm.cont_mean); d.update(kw)
    return RateParams(**d)


def fit_nm(f, x0):
    r = minimize(f, np.array(x0, float), method="Nelder-Mead",
                 options={"maxiter": 400 * len(x0), "xatol": 1e-4, "fatol": 1e-3})
    return r.x


def rows(P, prm, te, fold, rate, side, arm, adj=None):
    c, v = run_filter(P, prm, adj)
    sel = te[P.ts]
    num, den = P.df.columns[7], P.df.columns[8]
    dd = P.df.loc[sel, ["season", "game_id", "team_id", "opp_id", "j", "week", num, den, "L"]].rename(
        columns={num: "num", den: "den"}).copy()
    dd["c"] = c[P.ts[sel], P.j[sel]]; dd["v"] = v[P.ts[sel], P.j[sel]]
    dd["c_lag1"] = np.where(P.j[sel] > 0, c[P.ts[sel], np.maximum(P.j[sel] - 1, 0)], np.nan)
    dd["phi"] = prm.phi
    dd["fold"] = fold; dd["rate"] = rate; dd["side"] = side; dd["arm"] = arm; dd["fam"] = P.fam
    return dd


def prm_to_json(prm):
    return {"q": prm.q, "log_p0": list(prm.log_p0), "carry": list(prm.carry), "phi": prm.phi, "cont_mean": prm.cont_mean}


def prm_from_json(d):
    return RateParams(q=d["q"], log_p0=tuple(d["log_p0"]), carry=tuple(d["carry"]), phi=d["phi"], cont_mean=d["cont_mean"])


def part_q1(Ps):
    p2 = json.load(open(OUT / "params_v2.json"))
    est, params = [], {}
    t0 = time.time()
    for (rate, side), P in Ps.items():
        for fold, fd in FOLDS.items():
            tr = np.isin(P.season, fd["train"])[:, None] & P.mask
            first = tr & (np.arange(P.J)[None, :] <= FIRST_J)
            te = P.season == fd["test"]
            cm = p2[f"{fold}|{rate}|{side}|E3"]["cont_mean"]
            V0 = base_params(p2[f"{fold}|{rate}|{side}|E3"]["theta"], cm)
            a0 = fit_nm(lambda x: gauss_nll(P, replace(V0, log_p0=(x[0], 0.0, 0.0)), first), [V0.log_p0[0]])
            V1 = replace(V0, log_p0=(float(a0[0]), 0.0, 0.0))
            a = fit_nm(lambda x: gauss_nll(P, replace(V0, log_p0=(x[0], x[1], x[2])), first), [a0[0], 0.0, 0.0])
            V2 = replace(V0, log_p0=tuple(float(t) for t in a))
            arms = {"V0": V0, "V1": V1, "V2": V2}
            for nm, base in (("O1a", V1), ("O1b", V2)):
                if nm == "O1a":
                    x = fit_nm(lambda x: gauss_nll(P, replace(base, log_p0=(x[0], 0.0, 0.0), phi=float(np.exp(x[1]))), tr),
                               [base.log_p0[0], 0.0])
                    arms[nm] = replace(base, log_p0=(float(x[0]), 0.0, 0.0), phi=float(np.exp(x[1])))
                else:
                    x = fit_nm(lambda x: gauss_nll(P, replace(base, log_p0=(x[0], x[1], x[2]), phi=float(np.exp(x[3]))), tr),
                               list(base.log_p0) + [0.0])
                    arms[nm] = replace(base, log_p0=(float(x[0]), float(x[1]), float(x[2])), phi=float(np.exp(x[3])))
            for nm, prm in arms.items():
                params[f"{fold}|{rate}|{side}|{nm}"] = prm_to_json(prm)
                est.append(rows(P, prm, te, fold, rate, side, nm))
        print(f"q1 {rate}/{side} ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(est, ignore_index=True).to_parquet(OUT / "estimates_q1_v3.parquet", index=False)
    json.dump(params, open(OUT / "params_q1_v3.json", "w"), indent=1)


def part_q2(Ps, base_arm):
    p2 = json.load(open(OUT / "params_v2.json")); p1 = json.load(open(OUT / "params_q1_v3.json"))
    est, params = [], {}
    for (rate, side), P in Ps.items():
        for fold, fd in FOLDS.items():
            tr = np.isin(P.season, fd["train"])[:, None] & P.mask
            first = tr & (np.arange(P.J)[None, :] <= FIRST_J)
            te = P.season == fd["test"]
            B = prm_from_json(p1[f"{fold}|{rate}|{side}|{base_arm}"])
            th_c = p2[f"{fold}|{rate}|{side}|E3c"]["theta"]
            P1 = replace(B, carry=(float(th_c[2]), float(th_c[3]), float(th_c[4])),
                         cont_mean=p2[f"{fold}|{rate}|{side}|E3c"]["cont_mean"])
            r = fit_nm(lambda x: gauss_nll(P, replace(B, carry=(x[0], 0.0, 0.0)), first), [B.carry[0]])
            Pd = replace(B, carry=(float(r[0]), 0.0, 0.0))
            for nm, prm in (("P0", B), ("P1", P1), ("Pd", Pd)):
                params[f"{fold}|{rate}|{side}|{nm}"] = prm_to_json(prm)
                est.append(rows(P, prm, te, fold, rate, side, nm))
        print(f"q2 {rate}/{side}", flush=True)
    pd.concat(est, ignore_index=True).to_parquet(OUT / "estimates_q2_v3.parquet", index=False)
    json.dump(params, open(OUT / "params_q2_v3.json", "w"), indent=1)


def final_params(final):
    """final = '<file>:<arm>', e.g. 'q2:P0' or 'q1:V1'."""
    src, arm = final.split(":")
    p = json.load(open(OUT / f"params_{src}_v3.json"))
    return {k.rsplit("|", 1)[0]: prm_from_json(v) for k, v in p.items() if k.endswith("|" + arm)}


def opp_adj(Ps, prm_map, fold, rate):
    """den x the opponent's opposite-side estimate at the same game, from the unadjusted final arm."""
    cs = {s: run_filter(Ps[(rate, s)], prm_map[f"{fold}|{rate}|{s}"])[0] for s in ("off", "def")}
    adj = {}
    for side, other in (("off", "def"), ("def", "off")):
        Po = Ps[(rate, other)]
        lut = pd.Series(cs[other][Po.ts, Po.j], index=pd.MultiIndex.from_arrays([Po.df["team_id"].to_numpy(), Po.df["game_id"].to_numpy()]))
        P = Ps[(rate, side)]
        look = lut.reindex(pd.MultiIndex.from_arrays([P.df["opp_id"].to_numpy(), P.df["game_id"].to_numpy()])).fillna(0.0).to_numpy()
        a = np.zeros_like(P.num); a[P.ts, P.j] = P.den[P.ts, P.j] * look
        adj[side] = a
    return adj


def part_q3(Ps, final):
    pm = final_params(final)
    est = []
    for rate in RATES:
        for fold, fd in FOLDS.items():
            adj = opp_adj(Ps, pm, fold, rate)
            for side in ("off", "def"):
                P = Ps[(rate, side)]; te = P.season == fd["test"]; prm = pm[f"{fold}|{rate}|{side}"]
                est.append(rows(P, prm, te, fold, rate, side, "final"))
                est.append(rows(P, prm, te, fold, rate, side, "final+opp", adj[side]))
        print(f"q3 {rate}", flush=True)
    pd.concat(est, ignore_index=True).to_parquet(OUT / "estimates_q3_v3.parquet", index=False)


TIPOFF = None


def part_emit(Ps, final, with_opp: bool, label: str):
    global TIPOFF
    u = pd.read_parquet("data/processed/games_universe.parquet", columns=["game_id", "tipoff_utc", "season"])
    TIPOFF = u[u["season"] <= 2025][["game_id", "tipoff_utc"]].assign(tipoff_utc=lambda x: pd.to_datetime(x["tipoff_utc"], utc=True))
    pm = final_params(final)
    wide = []
    for fold, fd in FOLDS.items():
        parts = []
        for rate in RATES:
            adj = opp_adj(Ps, pm, fold, rate) if with_opp else {"off": None, "def": None}
            for side in ("off", "def"):
                P = Ps[(rate, side)]
                # ---- strict as-of: dates strictly increasing within team-season, so index j uses only earlier dates
                # (dates are calendar dates; one 2023 team played twice on 2022-11-25, so the check uses tipoff time)
                tip = P.df[["season", "team_id", "j", "game_id"]].merge(TIPOFF, on="game_id", how="left")
                assert tip["tipoff_utc"].notna().all(), "missing tipoff time"
                tdiff = tip.sort_values(["season", "team_id", "j"]).groupby(["season", "team_id"])["tipoff_utc"].diff().dropna()
                assert (tdiff > pd.Timedelta(0)).all(), "game index order disagrees with tipoff order: as-of unsafe"
                keep = P.season <= fd["test"]
                sel = keep[P.ts]
                c, v = run_filter(P, pm[f"{fold}|{rate}|{side}"], adj[side])
                d = P.df.loc[sel, ["season", "game_id", "team_id", "game_date", "L"]].copy()
                # league level is the as-of cumulative over dates strictly before (X2.league_asof): assert monotone use
                d[f"{rate}_{side}_c"] = c[P.ts[sel], P.j[sel]]
                d[f"{rate}_{side}_v"] = v[P.ts[sel], P.j[sel]]
                d = d.rename(columns={"L": f"{rate}_{side}_L"})
                parts.append(d.set_index(["season", "game_id", "team_id", "game_date"]))
        w = pd.concat(parts, axis=1).reset_index()
        w.insert(0, "fold", fold)
        wide.append(w)
    out = pd.concat(wide, ignore_index=True)
    assert not out.duplicated(["fold", "game_id", "team_id"]).any()
    assert (out.loc[out["fold"] == "F1", "season"] <= 2024).all() and (out["season"] <= 2025).all()
    path = Path(f"data/processed/team_rate_features_{label}_v1.parquet")
    assert not path.exists(), f"{path} exists; write a new version instead of overwriting"
    out.to_parquet(path, index=False)
    json.dump({f"{k}": prm_to_json(v) for k, v in pm.items()}, open(OUT / f"params_emit_{label}_v3.json", "w"), indent=1)
    print("wrote", path, out.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["q1", "q2", "q3", "emit"])
    ap.add_argument("--base", default=None)
    ap.add_argument("--final", default=None)
    ap.add_argument("--opp", action="store_true")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    Ps, tg, cont = build_all()
    if a.part == "q1":
        part_q1(Ps)
    elif a.part == "q2":
        part_q2(Ps, a.base)
    elif a.part == "q3":
        part_q3(Ps, a.final)
    else:
        part_emit(Ps, a.final, a.opp, a.label)


if __name__ == "__main__":
    main()
