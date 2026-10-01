"""diag_aggregation_tfix_v1.py -- lane A 2026-09-30, experiments.md section 2.5 (addendum G): harness read of the
skew-free fg_make arm Tfix (seeds 0/1) as X_Tfix (S0 + fg Tfix) and S1fix (PO T + RB T + fg Tfix)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import diag_aggregation_overspread_v1 as A
import diag_g9_g6_margin_v1 as M1
d = A.base_frame(); Y = d["margin"].to_numpy(float); C = d["close"].to_numpy(float); lined = d["close"].notna().to_numpy()
X, H = {}, {}
for n in ("S0", "S1", "X_F", "X_Tfix", "X_Tfix1", "S1fix", "X_PR"):
    h = pd.read_parquet(f"results/aggregation_v1/harness_{n}.parquet"); h = h[h.arm == "FULL"]; H[n] = h
    w = h.pivot_table(index="game_id", columns="side", values="ppp")
    X[n] = (67.875 * (w[0] - w[1])).reindex(d["game_id"]).to_numpy(float)
def stats(idx):
    o = {}
    for n, x in X.items():
        o[f"oms_{n}"] = 1 - A.slope(Y[idx], x[idx]); li = idx[lined[idx]]; o[f"omsC_{n}"] = 1 - A.slope(C[li], x[li]); o[f"sd_{n}"] = float(np.std(x[idx], ddof=1))
    for n in ("S1", "X_F", "X_Tfix", "X_Tfix1", "S1fix", "X_PR"):
        o[f"d_{n}"] = o[f"oms_{n}"] - o["oms_S0"]; o[f"dC_{n}"] = o[f"omsC_{n}"] - o["omsC_S0"]
    o["seedfloor_Tfix"] = o["oms_X_Tfix1"] - o["oms_X_Tfix"]
    return o
est = stats(np.arange(len(Y))); se = A.boot(stats, len(Y), 200)
for k in sorted(est): print(f"{k:16s} {est[k]:+.4f} ({se[k]:.4f})")
Yact, _, _ = M1.build_actual(); YA = Yact.set_index("game_id")
mon = pd.to_datetime(d.set_index("game_id")["game_date"]).dt.month
ms = {}
for k, rk, wk in (("rim", "p_rim", "n_fga_rim"), ("jump2", "p_jump", "n_fga_jump"), ("three", "p_3", "n_fga_3")):
    for n in ("S0", "X_F", "X_Tfix"):
        h = H[n]; rows = []
        for side, pre in ((0, "h"), (1, "a")):
            f = h[h.side == side].set_index("game_id"); ids = f.index.intersection(YA.index).intersection(mon.index)
            rows.append(pd.DataFrame({"y": YA.loc[ids, f"{pre}_{rk}"], "w": YA.loc[ids, f"{pre}_{wk}"], "p": f.loc[ids, f"p_make_{k}"], "m": mon.loc[ids].to_numpy()}))
        dd = pd.concat(rows); dd = dd[dd.w > 0]; out = {}
        for m in (0, 11, 12, 1, 2, 3):
            s = dd if m == 0 else dd[dd.m == m]; w = s.w.to_numpy(); x = s.p.to_numpy(); y = s.y.to_numpy()
            xm, ym = np.average(x, weights=w), np.average(y, weights=w)
            out[m] = float(np.sum(w*(x-xm)*(y-ym))/np.sum(w*(x-xm)**2))
        ms[f"{k}_{n}"] = out
        print(f"make_{k} {n:8s} " + " ".join(f"{m}:{v:.2f}" for m, v in out.items()))
json.dump({"est": est, "se": se, "month_slopes": ms}, open("results/aggregation_v1/analysis_tfix_v1.json", "w"), indent=1)
