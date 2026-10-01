"""diag_g3_delivery_v1.py -- why the engine delivers ~58% of G3's fitted make covariance (lane B, 2026-10-01).

DIAGNOSTIC ONLY. Paired 200-seed runs with and without G3 (same seeds, same stack otherwise):
  served v2 (COMB9GCTKD) vs COMB9CTKD;  S0+G3 (laneB_v3full_G3_s200_o0) vs S0.

For each type pair (k, l) the within-game between-team covariance of made FG is split exactly:
  FGM_ik = A_ik p0_ik + R_ik,   R_ik = FGM_ik - A_ik p0_ik  (p0 = the arm's own per-game make rate)
  Cov(FGM_hk, FGM_al) = p0p0 Cov(A_hk, A_al)       volume x volume
                      + p0 Cov(A_hk, R_al) + ...   volume x rate (both orders)
                      + Cov(R_hk, R_al)            rate x rate
Delta (G3 minus no-G3) of each part against the first-order prediction E_g[W_hk Sigma_kl W_al],
W = A0 p0 (1 - p0) from the no-G3 arm's per-game means (the sim's own W, as in the 09-30 doc 5.3).
Rate x rate / predicted tests the SCALE (logit Sigma applied on the scale it was fitted);
the volume terms are the DILUTION (fewer misses -> fewer OREB -> fewer attempts; clock).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
K = ["rim", "jump", "three"]
COL = {"rim": ("fga2_rim", "fgm2_rim"), "jump": ("fga2_jump", "fgm2_jump"), "three": ("fga3", "fgm3")}
SIG = np.array(json.loads((ROOT / "data/processed/models/shared_shooting/params_v1.json").read_text())["G3_Sigma"])


def load(run):
    cols = ["game_id", "seed"] + [f"{s}_{c}" for s in ("home", "away") for k in K for c in COL[k]] + \
        ["home_pts", "away_pts", "home_oreb", "away_oreb", "possessions",
         "home_tov", "away_tov", "home_fta", "away_fta"]
    return pd.read_parquet(ROOT / "results/engine_v0" / run / "games.parquet", columns=cols) \
        .sort_values(["game_id", "seed"]).reset_index(drop=True)


def parts(d):
    gid = d["game_id"].to_numpy()
    out = {}
    A, M, P0 = {}, {}, {}
    for s in ("home", "away"):
        for k in K:
            a = d[f"{s}_{COL[k][0]}"].to_numpy(float)
            m = d[f"{s}_{COL[k][1]}"].to_numpy(float)
            ga = pd.Series(a).groupby(gid).transform("mean").to_numpy()
            gm = pd.Series(m).groupby(gid).transform("mean").to_numpy()
            p0 = gm / np.maximum(ga, 1e-9)
            A[s, k], M[s, k], P0[s, k] = a, m, p0
    def wc(x, y):
        xc = x - pd.Series(x).groupby(gid).transform("mean").to_numpy()
        yc = y - pd.Series(y).groupby(gid).transform("mean").to_numpy()
        return float(np.mean(xc * yc))
    for k in K:
        for l in K:
            ah, al = A["home", k], A["away", l]
            ph, pl = P0["home", k], P0["away", l]
            rh, rl = M["home", k] - ah * ph, M["away", l] - al * pl
            out[f"{k}x{l}"] = {"count": wc(M["home", k], M["away", l]),
                               "vol_vol": wc(ah * ph, al * pl),
                               "vol_rate": wc(ah * ph, rl) + wc(rh, al * pl),
                               "rate_rate": wc(rh, rl)}
    out["_pts_cov"] = wc(d["home_pts"].to_numpy(float), d["away_pts"].to_numpy(float))
    out["_oreb_cov"] = wc(d["home_oreb"].to_numpy(float), d["away_oreb"].to_numpy(float))
    out["_poss_var"] = wc(d["possessions"].to_numpy(float), d["possessions"].to_numpy(float))
    fga = {s: sum(A[s, k] for k in K) for s in ("home", "away")}
    out["_fga_cov"] = wc(fga["home"], fga["away"])
    out["_fga_var_h"] = wc(fga["home"], fga["home"])
    # route of the volume response: Cov(FGA_i, R_j total), FGA = N_i + OREB - TOV - 0.44 FTA (exact)
    Rt = {s: sum(M[s, k] - A[s, k] * P0[s, k] for k in K) for s in ("home", "away")}
    for i, j in (("home", "away"), ("away", "home")):
        oreb = d[f"{i}_oreb"].to_numpy(float)
        tov = d[f"{i}_tov"].to_numpy(float)
        fta = d[f"{i}_fta"].to_numpy(float)
        ni = fga[i] - oreb + tov + 0.44 * fta
        for nm, x in (("N", ni), ("OREB", oreb), ("TOV", -tov), ("FTA", -0.44 * fta), ("FGA", fga[i])):
            out[f"_route_{nm}"] = out.get(f"_route_{nm}", 0.0) + wc(x, Rt[j])
        out["_own_FGA_x_own_R"] = out.get("_own_FGA_x_own_R", 0.0) + wc(fga[i], Rt[i])
        out["_own_OREB_x_own_R"] = out.get("_own_OREB_x_own_R", 0.0) + wc(oreb, Rt[i])
    return out, A, P0, gid


def predicted(A, P0, gid):
    g = pd.Index(pd.unique(gid))
    W = {}
    for s in ("home", "away"):
        for k in K:
            a0 = pd.Series(A[s, k]).groupby(gid).mean().loc[g].to_numpy()
            p0 = pd.Series(P0[s, k]).groupby(gid).mean().loc[g].to_numpy()
            W[s, k] = a0 * p0 * (1 - p0)
    return {f"{k}x{l}": float(np.mean(W["home", k] * SIG[i, j] * W["away", l]))
            for i, k in enumerate(K) for j, l in enumerate(K)}


def main():
    pairs = [("v3full_COMB9GCTKD_s200_o0", "v3full_COMB9CTKD_s200_o0"),
             ("laneB_v3full_G3_s200_o0", "v3full_S0_s200_o0")]
    rep = {}
    for on, off in pairs:
        p_on, *_ = parts(load(on))
        p_off, A, P0, gid = parts(load(off))
        pred = predicted(A, P0, gid)
        rows = {}
        tot = {"count": 0, "vol_vol": 0, "vol_rate": 0, "rate_rate": 0, "pred": 0}
        for key in pred:
            r = {c: p_on[key][c] - p_off[key][c] for c in ("count", "vol_vol", "vol_rate", "rate_rate")}
            r["pred"] = pred[key]
            rows[key] = r
            for c in tot:
                tot[c] += r[c]
        rows["total"] = tot
        rows["ratio_count_over_pred"] = tot["count"] / tot["pred"]
        rows["ratio_rate_over_pred"] = tot["rate_rate"] / tot["pred"]
        rows["delta_pts_cov"] = p_on["_pts_cov"] - p_off["_pts_cov"]
        rows["delta_oreb_cov"] = p_on["_oreb_cov"] - p_off["_oreb_cov"]
        rows["delta_fga_cov"] = p_on["_fga_cov"] - p_off["_fga_cov"]
        rows["delta_fga_var_home"] = p_on["_fga_var_h"] - p_off["_fga_var_h"]
        rows["delta_poss_var"] = p_on["_poss_var"] - p_off["_poss_var"]
        for kk in [x for x in p_on if x.startswith("_route_") or x.startswith("_own_")]:
            rows["delta" + kk] = p_on[kk] - p_off[kk]
        rep[f"{on} - {off}"] = rows
    out = ROOT / "results/g5_channels/g3_delivery_v1.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for k, v in rep.items():
        print("==", k)
        for kk, vv in v.items():
            if isinstance(vv, dict):
                print(f"  {kk:10s} " + " ".join(f"{c}={x:+.4f}" for c, x in vv.items()))
            else:
                print(f"  {kk}: {vv:+.4f}")


if __name__ == "__main__":
    main()
