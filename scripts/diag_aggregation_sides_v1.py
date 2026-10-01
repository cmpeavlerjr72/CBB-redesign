"""diag_aggregation_sides_v1.py -- lane A 2026-09-30, PM question 23:40: is the margin case the same mechanism as
clock round 7's pace case (each side individually calibrated, the two-side combination over-spread)?
Per rate and at margin level (harness, S0 and S1): the offence-team part (FULL - OFF-swapped) and the defence-team
part (FULL - DEF-swapped), each regressed ALONE (with the all-team-swapped base) and JOINTLY."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import diag_aggregation_overspread_v1 as A
import diag_g9_g6_margin_v1 as M1
d = A.base_frame()
Yact, _, _ = M1.build_actual(); YA = Yact.set_index("game_id")
RATES = {"tov": (lambda f: f["p_tov"], "t", "P"),
         "make_rim": (lambda f: f["p_make_rim"], "p_rim", "n_fga_rim"),
         "make_jump": (lambda f: f["p_make_jump2"], "p_jump", "n_fga_jump"),
         "make_3": (lambda f: f["p_make_three"], "p_3", "n_fga_3"),
         "oreb": (lambda f: f["p_oreb"], "rho", "chances"),
         "share_3": (lambda f: f["p_fga_three"] / (f["p_fga_rim"] + f["p_fga_jump2"] + f["p_fga_three"]), "s_3", "fga"),
         "ppp": (lambda f: f["ppp"], "PPP", "P")}
def wls(y, cols, w):
    Z = np.column_stack([np.ones(len(y))] + cols) * np.sqrt(w)[:, None]
    return np.linalg.lstsq(Z, y * np.sqrt(w), rcond=None)[0][1:]
out = {}
for st in ("S0", "S1"):
    h = pd.read_parquet(f"results/aggregation_v1/harness_{st}.parquet")
    r = {}
    for name, (fn, rk, wk) in RATES.items():
        parts = []
        for side, pre in ((0, "h"), (1, "a")):
            g = {a: h[(h.arm == a) & (h.side == side)].set_index("game_id") for a in ("FULL", "OFF", "DEF", "TEAM")}
            ids = g["FULL"].index.intersection(YA.index).intersection(d["game_id"])
            if wk == "P": w = YA.loc[ids, f"{pre}_P"]
            elif wk == "fga": w = YA.loc[ids, [f"{pre}_n_fga_rim", f"{pre}_n_fga_jump", f"{pre}_n_fga_3"]].sum(axis=1)
            elif wk == "chances": w = YA.loc[ids, f"{pre}_n_oreb"] + YA.loc[ids, f"{pre}_n_oppdreb"]
            else: w = YA.loc[ids, f"{pre}_{wk}"]
            v = {a: fn(g[a].loc[ids]).to_numpy(float) for a in g}
            parts.append(pd.DataFrame({"y": YA.loc[ids, f"{pre}_{rk}"].to_numpy(float), "w": w.to_numpy(float),
                                       "base": v["TEAM"], "off": v["FULL"] - v["OFF"], "dfn": v["FULL"] - v["DEF"],
                                       "full": v["FULL"]}))
        f = pd.concat(parts, ignore_index=True)
        f = f[(f.w > 0) & np.isfinite(f[["y", "base", "off", "dfn"]]).all(axis=1)]
        y, w = f.y.to_numpy(), f.w.to_numpy()
        rest = f.full - f.base - f.off - f.dfn
        b_joint = wls(y, [f.base.to_numpy(), f.off.to_numpy(), f.dfn.to_numpy(), rest.to_numpy()], w)
        b_off = wls(y, [f.base.to_numpy(), f.off.to_numpy()], w)
        b_def = wls(y, [f.base.to_numpy(), f.dfn.to_numpy()], w)
        b_sum = wls(y, [f.base.to_numpy(), (f.off + f.dfn).to_numpy()], w)
        r[name] = {"b_off_alone": float(b_off[1]), "b_def_alone": float(b_def[1]), "b_off_joint": float(b_joint[1]),
                   "b_def_joint": float(b_joint[2]), "b_sum": float(b_sum[1]),
                   "corr_off_def": float(np.corrcoef(f.off, f.dfn)[0, 1]),
                   "sd_off": float(f.off.std()), "sd_def": float(f.dfn.std())}
    # margin level: off part / def part of the margin (harness)
    w2 = h.pivot_table(index=["game_id", "arm"], columns="side", values="ppp")
    m = (67.875 * (w2[0] - w2[1])).unstack("arm").reindex(d["game_id"]).reset_index(drop=True)
    Y = d["margin"].to_numpy(float)
    off, dfn, base = (m["FULL"] - m["OFF"]).to_numpy(), (m["FULL"] - m["DEF"]).to_numpy(), m["TEAM"].to_numpy()
    rest = m["FULL"].to_numpy() - base - off - dfn
    one = np.ones(len(Y))
    ols = lambda cols: np.linalg.lstsq(np.column_stack([one] + cols), Y, rcond=None)[0][1:]
    r["margin"] = {"b_off_alone": float(ols([base, off])[1]), "b_def_alone": float(ols([base, dfn])[1]),
                   "b_off_joint": float(ols([base, off, dfn, rest])[1]), "b_def_joint": float(ols([base, off, dfn, rest])[2]),
                   "b_sum": float(ols([base, off + dfn])[1]), "corr_off_def": float(np.corrcoef(off, dfn)[0, 1]),
                   "sd_off": float(off.std()), "sd_def": float(dfn.std())}
    out[st] = r
    for k, v in r.items():
        print(st, f"{k:9s}", " ".join(f"{q} {x:+.3f}" for q, x in v.items()))
json.dump(out, open("results/aggregation_v1/analysis_sides_v1.json", "w"), indent=1)
