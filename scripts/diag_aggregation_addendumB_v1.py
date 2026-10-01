"""diag_aggregation_addendumB_v1.py -- lane A 2026-09-30, docs/models/aggregation/experiments.md (analysis helper; run from repo root with CBB_TRUTH=verified_v1)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import diag_aggregation_overspread_v1 as A
import diag_g9_g6_margin_v1 as M1
d = A.base_frame(); Y = d["margin"].to_numpy(float); C = d["close"].to_numpy(float); lined = d["close"].notna().to_numpy()
def xh(name):
    h = pd.read_parquet(f"results/aggregation_v1/harness_{name}.parquet"); h = h[h.arm == "FULL"]
    w = h.pivot_table(index="game_id", columns="side", values="ppp")
    return (67.875 * (w[0] - w[1])).reindex(d["game_id"]).to_numpy(float), h
X = {n: xh(n) for n in ("S0", "S1", "X_F", "S0_fgfirst", "S1_fgfirst", "X_F_fgfirst")}
def stats(idx):
    o = {}
    for n, (x, _) in X.items():
        o[f"oms_{n}"] = 1 - A.slope(Y[idx], x[idx]); li = idx[lined[idx]]; o[f"omsC_{n}"] = 1 - A.slope(C[li], x[li]); o[f"sd_{n}"] = float(np.std(x[idx], ddof=1))
    for suf in ("", "_fgfirst"):
        for ref in ("oms", "omsC"):
            o[f"d_{ref}_S1{suf}"] = o[f"{ref}_S1{suf}"] - o[f"{ref}_S0{suf}"]; o[f"d_{ref}_XF{suf}"] = o[f"{ref}_X_F{suf}"] - o[f"{ref}_S0{suf}"]
    return o
est = stats(np.arange(len(Y))); se = A.boot(stats, len(Y), 200)
for k in sorted(est): print(f"{k:24s} {est[k]:+.4f} ({se[k]:.4f})")
# by-month make slopes under first refit
Yact, _, _ = M1.build_actual(); YA = Yact.set_index("game_id")
mon = pd.to_datetime(d.set_index("game_id")["game_date"]).dt.month
res = {"est": est, "se": se, "month_slopes": {}}
for k, rk, wk in (("rim", "p_rim", "n_fga_rim"), ("jump2", "p_jump", "n_fga_jump"), ("three", "p_3", "n_fga_3")):
    for n in ("S0", "S1", "S0_fgfirst", "S1_fgfirst"):
        h = X[n][1]; rows = []
        for side, pre in ((0, "h"), (1, "a")):
            f = h[h.side == side].set_index("game_id"); ids = f.index.intersection(YA.index).intersection(mon.index)
            rows.append(pd.DataFrame({"y": YA.loc[ids, f"{pre}_{rk}"], "w": YA.loc[ids, f"{pre}_{wk}"], "p": f.loc[ids, f"p_make_{k}"], "m": mon.loc[ids].to_numpy()}))
        dd = pd.concat(rows); dd = dd[dd.w > 0]; out = {}
        for m in (11, 12, 1, 2, 3, 0):
            s = dd if m == 0 else dd[dd.m == m]; w = s.w.to_numpy(); x = s.p.to_numpy(); y = s.y.to_numpy()
            xm, ym = np.average(x, weights=w), np.average(y, weights=w)
            out[m] = float(np.sum(w*(x-xm)*(y-ym))/np.sum(w*(x-xm)**2))
        res["month_slopes"][f"{k}_{n}"] = out
        print(f"make_{k} {n:12s} " + " ".join(f"{m}:{v:.2f}" for m, v in out.items()))
json.dump(res, open("results/aggregation_v1/analysis_addendumB_v1.json", "w"), indent=1)
