"""diag_aggregation_sim_vs_harness_v1.py -- lane A 2026-09-30, docs/models/aggregation/experiments.md (analysis helper; run from repo root with CBB_TRUTH=verified_v1)."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import diag_g9_g6_margin_v1 as M1
out = {}
MAP = {"t": lambda f: f["p_tov"], "rp": lambda f: f["p_trip"],
       "s_rim": lambda f: f["p_fga_rim"] / (f["p_fga_rim"] + f["p_fga_jump2"] + f["p_fga_three"]),
       "s_3": lambda f: f["p_fga_three"] / (f["p_fga_rim"] + f["p_fga_jump2"] + f["p_fga_three"]),
       "p_rim": lambda f: f["p_make_rim"], "p_jump": lambda f: f["p_make_jump2"], "p_3": lambda f: f["p_make_three"],
       "rho": lambda f: f["p_oreb"], "PPP": lambda f: f["ppp"]}
for st in ("S0", "S1"):
    X = M1.build_sim(Path(f"results/engine_v0/v3full_{st}_s200_o0"), 67.875, st).set_index("game_id")
    h = pd.read_parquet(f"results/aggregation_v1/harness_{st}.parquet"); h = h[h.arm == "FULL"]
    r = {}
    for k, fn in MAP.items():
        xs, ys = [], []
        for side, pre in ((0, "h"), (1, "a")):
            f = h[h.side == side].set_index("game_id"); ids = f.index.intersection(X.index)
            xs.append(fn(f.loc[ids]).to_numpy(float)); ys.append(X.loc[ids, f"{pre}_{k}"].to_numpy(float))
        x = np.concatenate(xs); y = np.concatenate(ys); ok = np.isfinite(x) & np.isfinite(y)
        r[k] = {"slope_sim_on_h": float(np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok], ddof=1)), "corr": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                "sd_h": float(np.std(x[ok])), "sd_sim": float(np.std(y[ok]))}
    out[st] = r
    print(st, {k: (round(v["slope_sim_on_h"], 3), round(v["corr"], 3)) for k, v in r.items()})
json.dump(out, open("results/aggregation_v1/analysis_sim_vs_harness_v1.json", "w"), indent=1)
