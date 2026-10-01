"""
diag_aggregation_overspread_v1.py -- analysis for docs/models/aggregation/experiments.md section 1
(lane A, 2026-09-30). DIAGNOSTIC ONLY.

Parts:
  identity  Part A: exact channel identity on the 200-seed full reads of S0 and S1; each channel's
            k split into OWN and CROSS terms.
  harness   Part C: deterministic expected-points harness (results/aggregation_v1/harness_<stack>.parquet)
            partitions L1 / L2 / L3, against realised (Y) and the close (C), harness-only and
            sim-anchored (with `not_harnessed`); stacks S0, S1, R, R2.
  loop      closed-loop swap arms (results/aggregation_v1/<stack>_<arm>_s<n>_o0), IV-on-split-seeds k's.

    CBB_TRUTH=verified_v1 .venv/Scripts/python.exe scripts/diag_aggregation_overspread_v1.py --part identity --part harness
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim.eval import gates as G  # noqa: E402
import diag_g9_g6_margin_v1 as M1  # noqa: E402

OUT = ROOT / "results" / "aggregation_v1"
SEASON = 2025
FULLREAD = {"S0": ROOT / "results/engine_v0/v3full_S0_s200_o0", "S1": ROOT / "results/engine_v0/v3full_S1_s200_o0"}
LINES = ROOT / "data/processed/lines/lines_close_v2_verified.parquet"
NB = 300
RNG = np.random.default_rng(20260930)
L1 = ["PO", "FG", "RB", "RAT", "PLY"]          # harness L1 (PACE is not harnessed)
L1_LOOP = ["PO", "FG", "RB", "RAT", "PACE", "PLY"]
L2 = ["OFF", "DEF"]


# --------------------------------------------------------------------------- common frame
def base_frame() -> pd.DataFrame:
    g = pd.read_parquet(FULLREAD["S0"] / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts",
                                                                    "possessions", "n_periods"])
    summ, _ = G.build_grading_frame(g, SEASON)
    d = summ[["game_id", "game_date", "month", "home_team_id", "away_team_id", "neutral", "margin",
              "sim_margin_mean", "sim_margin_sd", "n_seeds"]].rename(
        columns={"sim_margin_mean": "X_S0", "sim_margin_sd": "XSD_S0", "n_seeds": "NS_S0"})
    g1 = pd.read_parquet(FULLREAD["S1"] / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts",
                                                                     "possessions", "n_periods"])
    s1, _ = G.build_grading_frame(g1, SEASON)
    d = d.merge(s1[["game_id", "sim_margin_mean", "sim_margin_sd", "n_seeds"]].rename(
        columns={"sim_margin_mean": "X_S1", "sim_margin_sd": "XSD_S1", "n_seeds": "NS_S1"}), on="game_id")
    ln = pd.read_parquet(LINES)
    ln = ln[(ln["season"] == SEASON) & (ln["provider"] == "ESPN BET")].drop_duplicates("game_id")
    d = d.merge(ln[["game_id", "close_spread_home"]], on="game_id", how="left")
    d["close"] = -d["close_spread_home"]
    # team strength: as-of own-rating net of each team (home row = side 0 offence view)
    inp_dir = ROOT / "data/processed/models/engine_v3_S0_laneA"
    z = np.load(inp_dir / "arrays_F2_2025.npz")["team_static"]
    names = json.load(open(inp_dir / "names_F2_2025.json"))["team_names"]
    gm = pd.read_parquet(inp_dir / "games_F2_2025.parquet")[["game_id"]]
    ro, rd = names["off_rating_off_c"], names["off_rating_def_c"]
    gm["net_h"] = z[:, 0, ro] - z[:, 0, rd]
    gm["net_a"] = z[:, 1, ro] - z[:, 1, rd]
    d = d.merge(gm, on="game_id", how="left")
    sign = np.sign(np.corrcoef(d["net_h"] - d["net_a"], d["margin"])[0, 1])
    allnet = np.concatenate([d["net_h"], d["net_a"]]) * sign
    edges = np.quantile(allnet, [0.2, 0.4, 0.6, 0.8])
    d["q_h"] = np.searchsorted(edges, d["net_h"] * sign)
    d["q_a"] = np.searchsorted(edges, d["net_a"] * sign)
    d["qgap"] = (d["q_h"] - d["q_a"]).abs()
    d["site"] = np.where(d["neutral"] > 0, "neutral", "home/away")
    d["mon"] = pd.to_datetime(d["game_date"]).dt.month
    return d.reset_index(drop=True)


def slope(y, x):
    return float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))


def kdecomp(R: np.ndarray, X: np.ndarray, Z: dict, iv: tuple | None = None) -> dict:
    """1 - slope(R on X) = sum_j (1-b_j) cov(Z_j,X)/var(X) + k_res. Z_j must sum to X.
    iv = (Z_even, Z_odd) dicts for split-seed IV of b (averaged over both orientations)."""
    names = list(Z)
    vx = np.var(X, ddof=1)
    M = np.column_stack([np.ones(len(X))] + [Z[n] for n in names])
    if iv is None:
        b = np.linalg.lstsq(M, R, rcond=None)[0]
    else:
        ze, zo = iv
        Me = np.column_stack([np.ones(len(X))] + [ze[n] for n in names])
        Mo = np.column_stack([np.ones(len(X))] + [zo[n] for n in names])
        b = 0.5 * (np.linalg.solve(Mo.T @ Me, Mo.T @ R) + np.linalg.solve(Me.T @ Mo, Me.T @ R))
    out = {"slope": slope(R, X), "sd_X": float(np.sqrt(vx))}
    tot = 0.0
    for j, n in enumerate(names):
        share = float(np.cov(Z[n], X)[0, 1] / vx)
        out[f"k_{n}"] = (1 - b[j + 1]) * share
        out[f"b_{n}"] = float(b[j + 1])
        out[f"share_{n}"] = share
        tot += out[f"k_{n}"]
    out["k_res"] = (1 - out["slope"]) - tot
    out["one_minus_slope"] = 1 - out["slope"]
    return out


def boot(fn, n: int, nb: int = NB) -> dict:
    """fn(idx) -> dict of floats. Returns SE per key."""
    reps = []
    for _ in range(nb):
        idx = RNG.integers(0, n, n)
        reps.append(fn(idx))
    df = pd.DataFrame(reps)
    return df.std(ddof=1).to_dict()


# --------------------------------------------------------------------------- Part C: harness
def harness_margins(stack: str, P_ref: float) -> pd.DataFrame:
    h = pd.read_parquet(OUT / f"harness_{stack}.parquet")
    w = h.pivot_table(index=["game_id", "arm"], columns="side", values="ppp")
    m = (P_ref * (w[0] - w[1])).unstack("arm")
    return m


def harness_components(m: pd.DataFrame) -> tuple[dict, dict, dict]:
    full = m["FULL"]
    D = {a: full - m[a] for a in m.columns if a != "FULL"}
    l1 = {"base_ALL": m["ALL"]}
    for a in L1:
        l1[a] = D[a]
    l1["I1"] = full - m["ALL"] - sum(D[a] for a in L1)
    l2 = {"base_TEAM": m["TEAM"], "OFF": D["OFF"], "DEF": D["DEF"], "I2": full - m["TEAM"] - D["OFF"] - D["DEF"]}
    l3 = {"rest": full - D["RAT"], "RAT_PO": D["RAT_PO"], "RAT_other": D["RAT"] - D["RAT_PO"]}
    return l1, l2, l3


def part_harness(d: pd.DataFrame, P_ref: float) -> dict:
    res = {}
    mats = {}
    for st in ("S0", "S1", "R", "R2"):
        p = OUT / f"harness_{st}.parquet"
        if not p.exists():
            print("missing", p)
            continue
        m = harness_margins(st, P_ref)
        mats[st] = m.reindex(d["game_id"]).reset_index(drop=True)
    lined = d["close"].notna().to_numpy()
    for st, m in mats.items():
        l1, l2, l3 = harness_components(m)
        Xh = m["FULL"].to_numpy()
        r = {"sd_Xh": float(np.std(Xh, ddof=1)), "corr_X_Xh": None}
        if f"X_{st}" in d:
            r["corr_X_Xh"] = float(np.corrcoef(Xh, d[f"X_{st}"])[0, 1])
            r["slope_X_on_Xh"] = slope(d[f"X_{st}"].to_numpy(), Xh)
        for arm in m.columns:
            r[f"arm_sd_{arm}"] = float(np.std(m[arm], ddof=1))
            r[f"arm_slopeY_{arm}"] = slope(d["margin"].to_numpy(), m[arm].to_numpy())
            r[f"arm_slopeC_{arm}"] = slope(d.loc[lined, "close"].to_numpy(), m.loc[lined, arm].to_numpy())
        for pname, Z in (("L1", l1), ("L2", l2), ("L3", l3)):
            Zn = {k: np.asarray(v, float) for k, v in Z.items()}
            for ref, mask in (("Y", np.ones(len(d), bool)), ("C", lined)):
                R = (d["margin"] if ref == "Y" else d["close"]).to_numpy(float)
                kk = kdecomp(R[mask], Xh[mask], {k: v[mask] for k, v in Zn.items()})
                se = boot(lambda idx: {k: v for k, v in kdecomp(R[mask][idx], Xh[mask][idx],
                                                                 {q: z[mask][idx] for q, z in Zn.items()}).items()
                                       if k.startswith("k_") or k == "slope"}, int(mask.sum()), 200)
                r[f"harness_{pname}_{ref}"] = kk
                r[f"harness_{pname}_{ref}_se"] = se
            # sim-anchored (S0, S1 only): add not_harnessed = X - Xh
            if f"X_{st}" in d:
                X = d[f"X_{st}"].to_numpy(float)
                Zs = dict(Zn)
                Zs["not_harnessed"] = X - Xh
                for ref, mask in (("Y", np.ones(len(d), bool)), ("C", lined)):
                    R = (d["margin"] if ref == "Y" else d["close"]).to_numpy(float)
                    kk = kdecomp(R[mask], X[mask], {k: v[mask] for k, v in Zs.items()})
                    se = boot(lambda idx: {k: v for k, v in kdecomp(R[mask][idx], X[mask][idx],
                                                                     {q: z[mask][idx] for q, z in Zs.items()}).items()
                                           if k.startswith("k_") or k == "slope"}, int(mask.sum()), 200)
                    mc = float(np.mean(d.loc[mask, f"XSD_{st}"] ** 2 / d.loc[mask, f"NS_{st}"]) / np.var(X[mask], ddof=1))
                    kk["k_mc_expected"] = mc
                    r[f"sim_{pname}_{ref}"] = kk
                    r[f"sim_{pname}_{ref}_se"] = se
        # segments (harness-only L1/L2, Y lens)
        seg = {}
        for sname, col in (("month", "mon"), ("qgap", "qgap"), ("site", "site")):
            for val, sub in d.groupby(col):
                ix = sub.index.to_numpy()
                if len(ix) < 60:
                    continue
                e = {"n": int(len(ix)), "underpowered": bool(len(ix) < 300)}
                for pname, Z in (("L1", l1), ("L2", l2)):
                    kk = kdecomp(d.loc[ix, "margin"].to_numpy(float), Xh[ix], {k: np.asarray(v, float)[ix] for k, v in Z.items()})
                    e[pname] = {k: v for k, v in kk.items() if k.startswith("k_") or k in ("slope", "sd_X")}
                if lined[ix].sum() > 60:
                    ixl = ix[lined[ix]]
                    e["slopeC"] = slope(d.loc[ixl, "close"].to_numpy(float), Xh[ixl])
                seg[f"{sname}={val}"] = e
        r["segments"] = seg
        # per team: SD of per-team mean predicted margin (team view) vs realised and close
        tm = []
        for side, sg in (("home_team_id", 1.0), ("away_team_id", -1.0)):
            tm.append(pd.DataFrame({"team": d[side], "x": sg * Xh, "y": sg * d["margin"], "c": sg * d["close"]}))
        tm = pd.concat(tm).groupby("team").mean()
        r["per_team"] = {"sd_x": float(tm["x"].std()), "sd_y": float(tm["y"].std()), "sd_c": float(tm["c"].std()),
                         "slope_y_on_x": slope(tm["y"].to_numpy(), tm["x"].to_numpy()),
                         "slope_c_on_x": slope(tm["c"].dropna().to_numpy(), tm.loc[tm["c"].notna(), "x"].to_numpy())}
        res[st] = r
        print(f"[harness] {st}: sd_Xh {r['sd_Xh']:.3f} corr(X,Xh) {r['corr_X_Xh']} "
              f"slope Y~Xh {r['harness_L1_Y']['slope']:.4f}", flush=True)
    # paired S1 - S0 (and R2 - R as the retrain-seed floor; T vs R as the trainer-matched comparison)
    pairs = {}
    for a, b in (("S1", "S0"), ("R2", "R"), ("S1", "R"), ("R", "S0")):
        if a not in mats or b not in mats:
            continue
        la, l2a, l3a = harness_components(mats[a]); lb, l2b, l3b = harness_components(mats[b])
        Xa, Xb = mats[a]["FULL"].to_numpy(float), mats[b]["FULL"].to_numpy(float)
        Y = d["margin"].to_numpy(float)
        out = {}
        for pname, Za, Zb in (("L1", la, lb), ("L2", l2a, l2b), ("L3", l3a, l3b)):
            Za = {k: np.asarray(v, float) for k, v in Za.items()}
            Zb = {k: np.asarray(v, float) for k, v in Zb.items()}

            def dk(idx):
                ka = kdecomp(Y[idx], Xa[idx], {k: v[idx] for k, v in Za.items()})
                kb = kdecomp(Y[idx], Xb[idx], {k: v[idx] for k, v in Zb.items()})
                return {k: ka[k] - kb[k] for k in ka if k.startswith("k_") or k == "slope" or k == "sd_X"}
            out[pname] = dk(np.arange(len(Y)))
            out[pname + "_se"] = boot(dk, len(Y), 200)
        pairs[f"{a}-{b}"] = out
    res["pairs"] = pairs
    return res


# --------------------------------------------------------------------------- Part A: identity own / cross
def sim_channels_frame(g: pd.DataFrame, P_ref: float) -> pd.DataFrame:
    h, a = M1.side_rates(M1.sim_counts(g, "home")), M1.side_rates(M1.sim_counts(g, "away"))
    ch, _ = M1.decompose(h, a, P_ref)
    df = pd.DataFrame({"game_id": g["game_id"].to_numpy(), "seed": g["seed"].to_numpy()})
    for c in M1.CHANNELS:
        df[c] = ch[c]
    return df


def own_cross(Xc: pd.DataFrame, Yc: pd.DataFrame, X: np.ndarray) -> dict:
    vx = np.var(X, ddof=1)
    out = {}
    for c in M1.CHANNELS + ["final_vs_box"]:
        e = Xc[c].to_numpy() - Yc[c].to_numpy()
        own = float(np.cov(e, Xc[c])[0, 1] / vx)
        tot = float(np.cov(e, X)[0, 1] / vx)
        out[c] = {"own": own, "cross": tot - own, "total": tot}
    out["_sum_total"] = sum(v["total"] for k, v in out.items() if not k.startswith("_"))
    out["_sum_own"] = sum(v["own"] for k, v in out.items() if not k.startswith("_"))
    out["_sum_cross"] = sum(v["cross"] for k, v in out.items() if not k.startswith("_"))
    return out


def part_identity(d: pd.DataFrame) -> dict:
    Yact, P_ref, nmiss = M1.build_actual()
    res = {"P_ref": P_ref, "n_missing_ev": nmiss}
    Y = Yact.set_index("game_id").reindex(d["game_id"]).reset_index()
    ok = Y["tov"].notna().to_numpy()
    Yc = Y.loc[ok].reset_index(drop=True)
    Yc["final_vs_box"] = d.loc[ok, "margin"].to_numpy() - Yc[M1.CHANNELS].sum(axis=1).to_numpy()
    res["n_games"] = int(ok.sum())
    chan_means = {}
    for st in ("S0", "S1"):
        cols = ["game_id", "seed", "home_pts", "away_pts"] + [f"{s}_{k}" for s in ("home", "away") for k in (
            "fga3", "fga2_rim", "fga2_jump", "fta", "tov", "oreb", "dreb", "fgm2_rim", "fgm2_jump", "fgm3", "ftm")]
        g = pd.read_parquet(FULLREAD[st] / "games.parquet", columns=cols)
        g = g[g["game_id"].isin(d.loc[ok, "game_id"])]
        sc = sim_channels_frame(g, P_ref)
        sc["margin_sim"] = (g["home_pts"].astype(int) - g["away_pts"].astype(int)).to_numpy()
        draws = {}
        for q in range(4):
            sub = sc[(sc["seed"] // 50) == q]
            Xq = sub.groupby("game_id")[M1.CHANNELS + ["margin_sim"]].mean().reindex(Yc["game_id"]).reset_index()
            Xq["final_vs_box"] = Xq["margin_sim"] - Xq[M1.CHANNELS].sum(axis=1)
            draws[q] = own_cross(Xq, Yc, Xq["margin_sim"].to_numpy())
        Xm = sc.groupby("game_id")[M1.CHANNELS + ["margin_sim"]].mean().reindex(Yc["game_id"]).reset_index()
        Xm["final_vs_box"] = Xm["margin_sim"] - Xm[M1.CHANNELS].sum(axis=1)
        X = Xm["margin_sim"].to_numpy()
        oc = own_cross(Xm, Yc, X)
        oc["_slope"] = slope(Yc[M1.CHANNELS].sum(axis=1).to_numpy() + Yc["final_vs_box"].to_numpy(), X)
        oc["_draw_sd"] = {c: {t: float(np.std([draws[q][c][t] for q in range(4)], ddof=1)) for t in ("own", "cross", "total")}
                          for c in M1.CHANNELS}

        def bt(idx):
            o = own_cross(Xm.iloc[idx].reset_index(drop=True), Yc.iloc[idx].reset_index(drop=True), X[idx])
            return {f"{c}_{t}": o[c][t] for c in M1.CHANNELS for t in ("own", "cross")}
        oc["_boot_se"] = boot(bt, len(X), 200)
        # cross matrix (rows: channel c error, cols: channel d prediction)
        vx = np.var(X, ddof=1)
        mat = {c: {dd: float(np.cov(Xm[c] - Yc[c], Xm[dd])[0, 1] / vx) for dd in M1.CHANNELS} for c in M1.CHANNELS}
        oc["_matrix"] = mat
        # prediction-vs-realised cross covariance of channels (the "co-movement" test)
        oc["_pred_cov_vs_real"] = {f"{c}|{dd}": [float(np.cov(Xm[c], Xm[dd])[0, 1]), float(np.cov(Yc[c], Xm[dd])[0, 1])]
                                   for i, c in enumerate(M1.CHANNELS) for dd in M1.CHANNELS[i + 1:]}
        res[st] = oc
        chan_means[st] = Xm
        print(f"[identity] {st}: slope {oc['_slope']:.4f} sum_total {oc['_sum_total']:.4f} own {oc['_sum_own']:.4f} "
              f"cross {oc['_sum_cross']:.4f}", flush=True)
    # paired difference S1 - S0 with bootstrap
    if "S0" in chan_means and "S1" in chan_means:
        def dd(idx):
            a = own_cross(chan_means["S1"].iloc[idx].reset_index(drop=True), Yc.iloc[idx].reset_index(drop=True),
                          chan_means["S1"]["margin_sim"].to_numpy()[idx])
            b = own_cross(chan_means["S0"].iloc[idx].reset_index(drop=True), Yc.iloc[idx].reset_index(drop=True),
                          chan_means["S0"]["margin_sim"].to_numpy()[idx])
            o = {f"{c}_{t}": a[c][t] - b[c][t] for c in M1.CHANNELS for t in ("own", "cross", "total")}
            o["sum_own"] = a["_sum_own"] - b["_sum_own"]; o["sum_cross"] = a["_sum_cross"] - b["_sum_cross"]
            o["sum_total"] = a["_sum_total"] - b["_sum_total"]
            return o
        res["S1-S0"] = dd(np.arange(len(Yc)))
        res["S1-S0_se"] = boot(dd, len(Yc), 200)
    return res


# --------------------------------------------------------------------------- addendum A: replacement factorial
def part_factorial(d: pd.DataFrame, P_ref: float) -> dict:
    stacks = {"": "S0", "P": "X_P", "F": "X_F", "R": "X_R", "PF": "X_PF", "PR": "X_PR", "FR": "X_FR", "PFR": "S1"}
    seeds = {"P": "Z_P", "F": "Z_F", "R": "Z_R"}
    Y = d["margin"].to_numpy(float)
    lined = d["close"].notna().to_numpy()
    C = d["close"].to_numpy(float)
    X = {}
    for lab, st in list(stacks.items()) + [("z" + k, v) for k, v in seeds.items()]:
        p = OUT / f"harness_{st}.parquet"
        if not p.exists():
            print("missing", p)
            continue
        h = pd.read_parquet(p)
        h = h[h["arm"] == "FULL"]
        w = h.pivot_table(index="game_id", columns="side", values="ppp")
        X[lab] = (P_ref * (w[0] - w[1])).reindex(d["game_id"]).to_numpy(float)

    def stats(idx):
        o = {}
        for lab, x in X.items():
            o[f"oms_Y_{lab}"] = 1 - slope(Y[idx], x[idx])
            li = idx[lined[idx]]
            o[f"oms_C_{lab}"] = 1 - slope(C[li], x[li])
            o[f"sd_{lab}"] = float(np.std(x[idx], ddof=1))
        for ref in ("Y", "C"):
            f = lambda lab: o[f"oms_{ref}_{lab}"]  # noqa: E731
            if all(k in X for k in stacks):
                for m in "PFR":
                    others = [c for c in "PFR" if c != m]
                    states = ["", others[0], others[1], others[0] + others[1]]
                    key = lambda base, add: "".join(sorted(base + add, key="PFR".index))  # noqa: E731
                    o[f"main_{ref}_{m}"] = float(np.mean([f(key(sb, m)) - f(key(sb, "")) for sb in states]))
                    o[f"alone_{ref}_{m}"] = f(m) - f("")
                tot = f("PFR") - f("")
                o[f"total_{ref}"] = tot
                o[f"interaction_{ref}"] = tot - sum(f(m) - f("") for m in "PFR")
            for m in "PFR":
                if "z" + m in X:
                    o[f"seedfloor_{ref}_{m}"] = f("z" + m) - f("")
        return o
    full = stats(np.arange(len(Y)))
    se = boot(stats, len(Y), 200)
    print("[factorial] " + ", ".join(f"{k} {v:+.4f}({se.get(k, 0):.4f})" for k, v in full.items()
                                      if k.startswith(("main_Y", "alone_Y", "total_Y", "interaction_Y", "seedfloor_Y"))), flush=True)
    return {"est": full, "se": se}


# --------------------------------------------------------------------------- per-rate game-level calibration
RATE_DEF = {  # name: (predicted fn of harness row frame, realised key, weight key, rate-feature arm)
    "tov": (lambda f: f["p_tov"], "t", "P", "PO"),
    "trip": (lambda f: f["p_trip"], "rp", "P", "PO"),
    "share_rim": (lambda f: f["p_fga_rim"] / (f["p_fga_rim"] + f["p_fga_jump2"] + f["p_fga_three"]), "s_rim", "fga", "PO"),
    "share_3": (lambda f: f["p_fga_three"] / (f["p_fga_rim"] + f["p_fga_jump2"] + f["p_fga_three"]), "s_3", "fga", "PO"),
    "make_rim": (lambda f: f["p_make_rim"], "p_rim", "n_fga_rim", "FG"),
    "make_jump": (lambda f: f["p_make_jump2"], "p_jump", "n_fga_jump", "FG"),
    "make_3": (lambda f: f["p_make_three"], "p_3", "n_fga_3", "FG"),
    "oreb": (lambda f: f["p_oreb"], "rho", "chances", "RB"),
}


def part_rates(d: pd.DataFrame) -> dict:
    Yact, P_ref, _ = M1.build_actual()
    Y = Yact.set_index("game_id")
    res = {}
    for st in ("S0", "S1", "R2"):
        p = OUT / f"harness_{st}.parquet"
        if not p.exists():
            continue
        h = pd.read_parquet(p)
        r = {}
        for name, (fn, rk, wk, arm) in RATE_DEF.items():
            rows = []
            for side, pre in ((0, "h"), (1, "a")):
                full = h[(h["arm"] == "FULL") & (h["side"] == side)].set_index("game_id")
                sw = h[(h["arm"] == arm) & (h["side"] == side)].set_index("game_id")
                rat = h[(h["arm"] == "RAT") & (h["side"] == side)].set_index("game_id")
                ids = full.index.intersection(Y.index).intersection(d["game_id"])
                if wk == "P":
                    w = Y.loc[ids, f"{pre}_P"]
                elif wk == "fga":
                    w = Y.loc[ids, [f"{pre}_n_fga_rim", f"{pre}_n_fga_jump", f"{pre}_n_fga_3"]].sum(axis=1)
                elif wk == "chances":
                    w = Y.loc[ids, f"{pre}_n_oreb"] + Y.loc[ids, f"{pre}_n_oppdreb"]
                else:
                    w = Y.loc[ids, f"{pre}_{wk}"]
                rows.append(pd.DataFrame({"y": Y.loc[ids, f"{pre}_{rk}"].to_numpy(float), "w": w.to_numpy(float),
                                          "pf": fn(full.loc[ids]).to_numpy(float), "ps": fn(sw.loc[ids]).to_numpy(float),
                                          "pr": fn(rat.loc[ids]).to_numpy(float)}))
            f = pd.concat(rows, ignore_index=True)
            f = f[(f["w"] > 0) & np.isfinite(f[["y", "pf", "ps", "pr"]]).all(axis=1)]
            w = f["w"].to_numpy()
            def wslope(y, x):
                xm, ym = np.average(x, weights=w), np.average(y, weights=w)
                return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))
            b, _ = wls_(f["y"].to_numpy(), np.column_stack([f["ps"], f["pf"] - f["ps"]]), w)
            b2, _ = wls_(f["y"].to_numpy(), np.column_stack([f["pr"], f["pf"] - f["pr"]]), w)
            r[name] = {"n": int(len(f)), "slope_full": wslope(f["y"].to_numpy(), f["pf"].to_numpy()),
                       "sd_pred": float(np.sqrt(np.cov(f["pf"], aweights=w))),
                       "b_without_rates": float(b[1]), "b_rate_part": float(b[2]),
                       "sd_rate_part": float(np.sqrt(np.cov(f["pf"] - f["ps"], aweights=w))),
                       "b_without_ratings": float(b2[1]), "b_ratings_part": float(b2[2]),
                       "corr_rate_part_ratings_part": float(np.corrcoef(f["pf"] - f["ps"], f["pf"] - f["pr"])[0, 1])}
        res[st] = r
        print(f"[rates] {st}: " + ", ".join(f"{k} slope {v['slope_full']:.2f} b_rate {v['b_rate_part']:.2f}" for k, v in r.items()), flush=True)
    return res


def wls_(y, X, w):
    sw = np.sqrt(w)
    Z = np.column_stack([np.ones(len(y)), X]) * sw[:, None]
    b, *_ = np.linalg.lstsq(Z, y * sw, rcond=None)
    return b, None


# --------------------------------------------------------------------------- closed loop
def load_arm(stack, arm, nseeds, games, suffix=""):
    if arm == "FULL":
        g = pd.read_parquet(FULLREAD[stack] / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts"])
        g = g[g["seed"] < nseeds]
    else:
        p = OUT / f"{stack}_{arm}_s{nseeds}_o0{suffix}" / "games.parquet"
        if not p.exists():
            return None
        g = pd.read_parquet(p, columns=["game_id", "seed", "home_pts", "away_pts"])
    g = g[g["game_id"].isin(games)]
    g = g.assign(m=g["home_pts"].astype(float) - g["away_pts"].astype(float))
    return g.pivot_table(index="game_id", columns="seed", values="m")


def part_loop(d: pd.DataFrame, nseeds: int, suffix: str) -> dict:
    arms = ["FULL", "TEAM", "OFF", "DEF", "ALL", "PO", "FG", "RB", "RAT", "RAT_PO", "PACE", "PLY"]
    res = {"nseeds": nseeds, "suffix": suffix}
    for st in ("S0", "S1"):
        M = {}
        for a in arms:
            m = load_arm(st, a, nseeds, set(d["game_id"]), suffix)
            if m is not None:
                M[a] = m
        if "FULL" not in M or len(M) < 2:
            continue
        common = sorted(set.intersection(*[set(m.index) for m in M.values()]))
        dd = d.set_index("game_id").loc[common].reset_index()
        seeds = np.array(sorted(M["FULL"].columns))
        Ms = {a: m.loc[common, seeds].to_numpy(float) for a, m in M.items()}
        Y = dd["margin"].to_numpy(float)
        lined = dd["close"].notna().to_numpy()
        r = {"n_games": len(common), "arms": sorted(Ms)}
        ev, od = seeds % 2 == 0, seeds % 2 == 1

        def comps(sel, names):
            X = {a: Ms[a][:, sel].mean(axis=1) for a in names}
            out = {}
            full = X["FULL"]
            if all(a in X for a in ["ALL"] + L1_LOOP):
                z = {"base_ALL": X["ALL"]}
                for a in L1_LOOP:
                    z[a] = full - X[a]
                z["I1"] = full - X["ALL"] - sum(full - X[a] for a in L1_LOOP)
                out["L1"] = z
            if all(a in X for a in ("TEAM", "OFF", "DEF")):
                out["L2"] = {"base_TEAM": X["TEAM"], "OFF": full - X["OFF"], "DEF": full - X["DEF"],
                             "I2": full - X["TEAM"] - (full - X["OFF"]) - (full - X["DEF"])}
            return X, out
        Xall, Pall = comps(np.ones(len(seeds), bool), list(Ms))
        _, Pe = comps(ev, list(Ms)); _, Po = comps(od, list(Ms))
        X = Xall["FULL"]
        for a, xa in Xall.items():
            mcv = float(np.mean(Ms[a].var(axis=1, ddof=1)) / len(seeds))
            r[f"arm_{a}"] = {"sd": float(np.std(xa, ddof=1)), "slopeY": slope(Y, xa),
                             "slopeY_mc_corr": slope(Y, xa) * np.var(xa, ddof=1) / (np.var(xa, ddof=1) - mcv),
                             "slopeC": slope(dd.loc[lined, "close"].to_numpy(float), xa[lined]), "mae": float(np.mean(np.abs(Y - xa)))}
        for pname in Pall:
            for ref, mask in (("Y", np.ones(len(common), bool)), ("C", lined)):
                R = (dd["margin"] if ref == "Y" else dd["close"]).to_numpy(float)
                f = lambda idx, P=Pall[pname], E=Pe[pname], O=Po[pname]: kdecomp(  # noqa: E731
                    R[mask][idx], X[mask][idx], {k: v[mask][idx] for k, v in P.items()},
                    iv=({k: v[mask][idx] for k, v in E.items()}, {k: v[mask][idx] for k, v in O.items()}))
                kk = f(np.arange(int(mask.sum())))
                # MC expectation of k_res from split-half noise
                hX = (np.mean(Ms["FULL"][:, ev], 1) - np.mean(Ms["FULL"][:, od], 1)) / 2
                mc = 0.0
                for j, (k, v) in enumerate(Pall[pname].items()):
                    hj = (Pe[pname][k] - Po[pname][k]) / 2
                    mc += kk[f"b_{k}"] * float(np.cov(hj[mask], hX[mask])[0, 1])
                kk["k_mc_expected"] = mc / np.var(X[mask], ddof=1)
                # reliability per component (split-half corr)
                kk["reliability"] = {k: float(np.corrcoef(Pe[pname][k][mask], Po[pname][k][mask])[0, 1]) for k in Pall[pname]}
                se = boot(lambda idx: {k: v for k, v in f(idx).items() if k.startswith("k_")}, int(mask.sum()), 200)
                # seed-quarter draws
                qs = []
                for q in range(4):
                    sel = (np.arange(len(seeds)) % 4) == q
                    selq = np.where(sel)[0]
                    e_, o_ = np.zeros(len(seeds), bool), np.zeros(len(seeds), bool)
                    e_[selq[::2]] = True; o_[selq[1::2]] = True
                    Xq, Pq = comps(sel, list(Ms)); _, Pqe = comps(e_, list(Ms)); _, Pqo = comps(o_, list(Ms))
                    if o_.sum() == 0:
                        continue
                    qs.append(kdecomp(R[mask], Xq["FULL"][mask], {k: v[mask] for k, v in Pq[pname].items()},
                                      iv=({k: v[mask] for k, v in Pqe[pname].items()},
                                          {k: v[mask] for k, v in Pqo[pname].items()})))
                dsd = pd.DataFrame(qs).std(ddof=1).to_dict() if len(qs) >= 2 else {}
                kk["floor"] = {k: max(se.get(k, 0), dsd.get(k, 0) / 2) for k in se}
                kk["boot_se"] = se
                kk["draw_sd"] = {k: v for k, v in dsd.items() if k.startswith("k_")}
                r[f"{pname}_{ref}"] = kk
        res[st] = r
        print(f"[loop] {st}: n {len(common)} arms {sorted(Ms)}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", choices=["identity", "harness", "rates", "factorial", "loop"])
    ap.add_argument("--loop-seeds", type=int, default=48)
    ap.add_argument("--loop-suffix", default="")
    args = ap.parse_args()
    os.environ.setdefault("CBB_TRUTH", "verified_v1")
    d = base_frame()
    print(f"frame: {len(d)} graded games, lined {int(d['close'].notna().sum())}; slope S0 {slope(d['margin'], d['X_S0']):.4f} "
          f"S1 {slope(d['margin'], d['X_S1']):.4f}", flush=True)
    P_ref = 67.875
    allres = {"n_games": len(d), "slope_S0": slope(d["margin"].to_numpy(), d["X_S0"].to_numpy()),
              "slope_S1": slope(d["margin"].to_numpy(), d["X_S1"].to_numpy())}
    for part in args.part or []:
        if part == "identity":
            r = part_identity(d)
            P_ref = r["P_ref"]
        elif part == "harness":
            r = part_harness(d, P_ref)
        elif part == "rates":
            r = part_rates(d)
        elif part == "factorial":
            r = part_factorial(d, P_ref)
        else:
            r = part_loop(d, args.loop_seeds, args.loop_suffix)
        allres[part] = r
        tag = part if part != "loop" else f"loop_s{args.loop_seeds}{args.loop_suffix}"
        (OUT / f"analysis_{tag}_v1.json").write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
        print("wrote", OUT / f"analysis_{tag}_v1.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
