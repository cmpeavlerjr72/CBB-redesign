"""Addendum D analysis (lane A): closed-loop full reads X_F, X_PR vs S0, S1 (and R2 if present)."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import diag_aggregation_overspread_v1 as A
d = A.base_frame(); gid = d["game_id"].to_numpy()
Y = d["margin"].to_numpy(float); C = d["close"].to_numpy(float); lined = d["close"].notna().to_numpy()
runs = {"S0": "results/engine_v0/v3full_S0_s200_o0", "S1": "results/engine_v0/v3full_S1_s200_o0",
        "X_F": "results/aggregation_v1/X_F_FULL_s200_o0", "X_PR": "results/aggregation_v1/X_PR_FULL_s200_o0",
        "R2": "results/aggregation_v1/R2_FULL_s200_o0",
        "X_Tfix": "results/aggregation_v1/X_Tfix_FULL_s200_o0", "S1fix": "results/aggregation_v1/S1fix_FULL_s200_o0",
        "X_Tfix1": "results/aggregation_v1/X_Tfix1_FULL_s200_o0"}
M = {}
for k, p in runs.items():
    if not Path(p, "games.parquet").exists():
        continue
    g = pd.read_parquet(Path(p, "games.parquet"), columns=["game_id", "seed", "home_pts", "away_pts"])
    g["m"] = g["home_pts"].astype(float) - g["away_pts"].astype(float)
    M[k] = g.pivot_table(index="game_id", columns="seed", values="m").reindex(gid)
    print(k, M[k].shape, int(M[k].isna().sum().sum()))
seeds = np.arange(200)
def stats(idx, sel=None):
    o = {}
    for k, m in M.items():
        a = m.to_numpy()[:, seeds if sel is None else sel]
        x = a.mean(axis=1)[idx]
        mc = float(np.mean(a.var(axis=1, ddof=1)[idx]) / a.shape[1])
        o[f"slope_{k}"] = A.slope(Y[idx], x)
        o[f"slopemc_{k}"] = o[f"slope_{k}"] * np.var(x, ddof=1) / (np.var(x, ddof=1) - mc)
        li = idx[lined[idx]]
        o[f"slopeC_{k}"] = A.slope(C[li], m.to_numpy()[:, seeds if sel is None else sel].mean(axis=1)[li])
        o[f"sd_{k}"] = float(np.std(x, ddof=1))
    for k in M:
        if k != "S0":
            for s in ("slope", "slopeC", "sd"):
                o[f"d_{s}_{k}"] = o[f"{s}_{k}"] - o[f"{s}_S0"]
    if "X_F" in M and "S1" in M:
        o["share_XF"] = o["d_slope_X_F"] / o["d_slope_S1"]
        o["share_XPR"] = o["d_slope_X_PR"] / o["d_slope_S1"] if "X_PR" in M else np.nan
        o["shareC_XF"] = o["d_slopeC_X_F"] / o["d_slopeC_S1"]
        o["shareC_XPR"] = o["d_slopeC_X_PR"] / o["d_slopeC_S1"] if "X_PR" in M else np.nan
    for k in ("X_Tfix", "S1fix", "X_Tfix1"):
        if k in M and "S1" in M:
            o[f"share_{k}"] = o[f"d_slope_{k}"] / o["d_slope_S1"]
            o[f"shareC_{k}"] = o[f"d_slopeC_{k}"] / o["d_slopeC_S1"]
    return o
idx = np.arange(len(Y))
est = stats(idx)
draws = pd.DataFrame([stats(idx, np.arange(q * 50, q * 50 + 50)) for q in range(4)])
se = A.boot(lambda i: stats(i), len(Y), 200)
res = {"est": est, "boot_se": se, "draw_sd": draws.std(ddof=1).to_dict(), "draws": draws.to_dict(orient="list")}
for k in sorted(est):
    fl = max(se.get(k, 0), res["draw_sd"].get(k, 0) / 2)
    print(f"{k:16s} {est[k]:+.4f}  boot {se.get(k,0):.4f}  drawSD {res['draw_sd'].get(k,0):.4f}  floor {fl:.4f}")
json.dump(res, open("results/aggregation_v1/analysis_addendumD_v1.json", "w"), indent=1, default=float)
