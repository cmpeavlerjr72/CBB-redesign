"""diag_aggregation_addendumC_v1.py -- lane A 2026-09-30, docs/models/aggregation/experiments.md (analysis helper; run from repo root with CBB_TRUTH=verified_v1)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import diag_aggregation_overspread_v1 as A
d = A.base_frame(); Y = d["margin"].to_numpy(float); C = d["close"].to_numpy(float); lined = d["close"].notna().to_numpy()
res = {}
M = {}
for st in ("S0", "S1", "X_F"):
    h = pd.read_parquet(f"results/aggregation_v1/harness_{st}_addC.parquet")
    w = h.pivot_table(index=["game_id", "arm"], columns="side", values="ppp")
    m = (67.875 * (w[0] - w[1])).unstack("arm").reindex(d["game_id"]).reset_index(drop=True)
    M[st] = m
def comps(m):
    full = m["FULL"].to_numpy(float)
    z = {"rest": m["FG"].to_numpy(float), "FG_OFF": full - m["FG_OFF"].to_numpy(float), "FG_DEF": full - m["FG_DEF"].to_numpy(float)}
    z["FG_I"] = full - z["rest"] - z["FG_OFF"] - z["FG_DEF"]
    return full, z
def stats(idx):
    o = {}
    for st, m in M.items():
        X, Z = comps(m)
        for ref, R, mask in (("Y", Y, np.ones(len(Y), bool)), ("C", C, lined)):
            ii = idx[mask[idx]]
            k = A.kdecomp(R[ii], X[ii], {q: v[ii] for q, v in Z.items()})
            for q in Z: o[f"{st}_{ref}_k_{q}"] = k[f"k_{q}"]; o[f"{st}_{ref}_b_{q}"] = k[f"b_{q}"]; o[f"{st}_{ref}_share_{q}"] = k[f"share_{q}"]
    for ref in ("Y", "C"):
        for q in ("rest", "FG_OFF", "FG_DEF", "FG_I"):
            o[f"dS1_{ref}_k_{q}"] = o[f"S1_{ref}_k_{q}"] - o[f"S0_{ref}_k_{q}"]
            o[f"dXF_{ref}_k_{q}"] = o[f"X_F_{ref}_k_{q}"] - o[f"S0_{ref}_k_{q}"]
    return o
est = stats(np.arange(len(Y))); se = A.boot(stats, len(Y), 200)
for k in sorted(est):
    if "_k_" in k or "_b_" in k or "_share_" in k: print(f"{k:24s} {est[k]:+.4f} ({se[k]:.4f})")
json.dump({"est": est, "se": se}, open("results/aggregation_v1/analysis_addendumC_v1.json", "w"), indent=1)
