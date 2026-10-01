"""
diag_g9_team_response_v1.py -- lane A day 2026-10-01: G9 slope of SERVED STACK v2 and the team-rate
response of possession_outcome (PO) and fg_make. DIAGNOSTIC ONLY; nothing is adopted or re-defaulted.

Part `attrib` (step 1 of the brief): reproduce last night's attribution
(docs/tests/aggregation_overspread_decomposition_2026-09-30.md sections 2.2 / 2.4 / 5.3) on the CURRENT
served stack v2, at the same level (deterministic harness components, sim-anchored with the served-v2
200-seed mean margin), and show it at several levels:
  overall; by month; by site (home/away vs neutral); per TEAM PRIOR QUINTILE (team view, as-of own-rating
  net); per possession type (team-game rate calibration of the sim and the harness rate-part / ratings-part
  betas); responsiveness (predicted vs realised margin by team prior quintile).

The harness rows are `results/aggregation_v1/harness_S0.parquet`: the served v2 stack serves the SAME
possession_outcome / fg_make / rebound / free_throw artifacts and the SAME input arrays and event team block
(v3, sha 228794fc) as `engine_v3_S0_laneA` (checked in this script). The five v2 loop-level changes (clock L2,
shot block, foul joint, shared shooting G3, chance time KD) are not in the reference-state harness; they sit in
`not_harnessed` = X_sim - X_h, which is carried as its own component.

Part `arms`: the pre-registered fix round (docs/models/aggregation/experiments.md section 3): harness margin
1 - slope(Y on X_h) with one sub-model replaced by an arm, paired against the served harness, with bootstrap
SEs and the arm's retrain-seed floor; per-rate betas; responsiveness by team prior quintile.

    CBB_TRUTH=verified_v1 .venv/Scripts/python.exe scripts/diag_g9_team_response_v1.py --part attrib
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("CBB_TRUTH", "verified_v1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim.eval import gates as G  # noqa: E402
import diag_aggregation_overspread_v1 as A  # noqa: E402
import diag_g9_g6_margin_v1 as M1  # noqa: E402

OUT = ROOT / "results" / "g9_team_response_v1"
AGG = ROOT / "results" / "aggregation_v1"
V2READ = ROOT / "results/engine_v0/v3full_COMB9GCTKD_s200_o0"
S0READ = ROOT / "results/engine_v0/v3full_S0_s200_o0"
P_REF = 67.875
NB = 200


def sha8(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def check_inputs() -> dict:
    a = np.load(ROOT / "data/processed/models/engine_v3_S0_laneA/arrays_F2_2025.npz")
    b = np.load(ROOT / "data/processed/models/engine_v3/arrays_F2_2025.npz")
    arrays_equal = all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
    blk_h = sha8(ROOT / "data/processed/models/engine_v3_S0_laneA/overlay/data/processed/models/engine/"
                 "event_round2_s1_F2_2025/team_block.npz")
    blk_s = sha8(ROOT / "data/processed/models/engine/event_team_block_v3_F2_2025.npz")
    return {"arrays_equal": bool(arrays_equal), "harness_event_block": blk_h, "served_v3_block": blk_s,
            "block_equal": blk_h == blk_s}


def sim_mean(run: Path) -> pd.DataFrame:
    g = pd.read_parquet(run / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts",
                                                        "possessions", "n_periods"])
    s, _ = G.build_grading_frame(g, A.SEASON)
    return s[["game_id", "sim_margin_mean", "sim_margin_sd", "n_seeds"]]


def frame() -> pd.DataFrame:
    d = A.base_frame()
    v2 = sim_mean(V2READ).rename(columns={"sim_margin_mean": "X_V2", "sim_margin_sd": "XSD_V2", "n_seeds": "NS_V2"})
    d = d.merge(v2, on="game_id", how="left")
    assert d["X_V2"].notna().all()
    return d


def harness_X(stack: str, d: pd.DataFrame, arm: str = "FULL", path: Path | None = None) -> np.ndarray:
    h = pd.read_parquet(path or (AGG / f"harness_{stack}.parquet"))
    h = h[h["arm"] == arm]
    w = h.pivot_table(index="game_id", columns="side", values="ppp")
    return (P_REF * (w[0] - w[1])).reindex(d["game_id"]).to_numpy(float)


def l1_components(stack: str, d: pd.DataFrame, path: Path | None = None):
    m = A.harness_margins(stack, P_REF) if path is None else None
    if path is not None:
        h = pd.read_parquet(path)
        w = h.pivot_table(index=["game_id", "arm"], columns="side", values="ppp")
        m = (P_REF * (w[0] - w[1])).unstack("arm")
    m = m.reindex(d["game_id"]).reset_index(drop=True)
    l1, l2, _ = A.harness_components(m)
    return m, {k: np.asarray(v, float) for k, v in l1.items()}, {k: np.asarray(v, float) for k, v in l2.items()}


def kd_with_se(R, X, Z, mask, nb=NB):
    kk = A.kdecomp(R[mask], X[mask], {k: v[mask] for k, v in Z.items()})
    se = A.boot(lambda idx: {k: v for k, v in A.kdecomp(R[mask][idx], X[mask][idx],
                                                         {q: z[mask][idx] for q, z in Z.items()}).items()
                             if k.startswith(("k_", "b_")) or k == "slope"}, int(mask.sum()), nb)
    return kk, se


# --------------------------------------------------------------------------- per-rate (possession type)
RATE_SIM = {  # name: (sim/actual rate key, weight key in actual frame)
    "tov": ("t", "P"), "ft_trip": ("rp", "P"), "share_rim": ("s_rim", "fga"), "share_jump": ("s_jump", "fga"),
    "share_3": ("s_3", "fga"), "make_rim": ("p_rim", "n_fga_rim"), "make_jump": ("p_jump", "n_fga_jump"),
    "make_3": ("p_3", "n_fga_3"), "oreb": ("rho", "chances"), "ft_pct": ("f", "n_fta"),
}


def rate_frame(Yact: pd.DataFrame, Xs: dict[str, pd.DataFrame], d: pd.DataFrame) -> pd.DataFrame:
    """team-game rows (one per side): realised rate, weight, sim mean rate per stack, team quintile, site."""
    Y = Yact.set_index("game_id")
    dd = d.set_index("game_id")
    rows = []
    for pre, qcol in (("h", "q_h"), ("a", "q_a")):
        ids = Y.index.intersection(dd.index)
        for X in Xs.values():
            ids = ids.intersection(X.set_index("game_id").index)
        f = pd.DataFrame({"game_id": ids, "side": pre, "q": dd.loc[ids, qcol].to_numpy(),
                          "site": dd.loc[ids, "site"].to_numpy(), "mon": dd.loc[ids, "mon"].to_numpy()})
        for name, (rk, wk) in RATE_SIM.items():
            f[f"y_{name}"] = Y.loc[ids, f"{pre}_{rk}"].to_numpy(float)
            if wk == "P":
                w = Y.loc[ids, f"{pre}_P"]
            elif wk == "fga":
                w = Y.loc[ids, [f"{pre}_n_fga_rim", f"{pre}_n_fga_jump", f"{pre}_n_fga_3"]].sum(axis=1)
            elif wk == "chances":
                w = Y.loc[ids, f"{pre}_n_oreb"] + Y.loc[ids, f"{pre}_n_oppdreb"]
            else:
                w = Y.loc[ids, f"{pre}_{wk}"]
            f[f"w_{name}"] = w.to_numpy(float)
            for st, X in Xs.items():
                f[f"x_{st}_{name}"] = X.set_index("game_id").loc[ids, f"{pre}_{rk}"].to_numpy(float)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def wslope(y, x, w):
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))


def rate_table(rf: pd.DataFrame, stacks) -> dict:
    out = {}
    for name in RATE_SIM:
        for st in stacks:
            f = rf[["q", "site", "mon", f"y_{name}", f"w_{name}", f"x_{st}_{name}"]].dropna()
            f = f[f[f"w_{name}"] > 0]
            y, x, w = f[f"y_{name}"].to_numpy(), f[f"x_{st}_{name}"].to_numpy(), f[f"w_{name}"].to_numpy()
            e = {"n": int(len(f)), "slope": wslope(y, x, w), "bias": float(np.average(x - y, weights=w)),
                 "sd_pred": float(np.sqrt(np.cov(x, aweights=w)))}
            bs = []
            rng = np.random.default_rng(7)
            for _ in range(100):
                i = rng.integers(0, len(f), len(f))
                bs.append(wslope(y[i], x[i], w[i]))
            e["slope_se"] = float(np.std(bs, ddof=1))
            # responsiveness by team prior quintile (offence team's quintile): weighted mean pred vs actual
            qs = {}
            for q, s in f.groupby("q"):
                ws = s[f"w_{name}"].to_numpy()
                qs[int(q)] = {"pred": float(np.average(s[f"x_{st}_{name}"], weights=ws)),
                              "act": float(np.average(s[f"y_{name}"], weights=ws)), "n": int(len(s))}
            e["by_quintile"] = qs
            pq = np.array([qs[q]["pred"] for q in sorted(qs)]); aq = np.array([qs[q]["act"] for q in sorted(qs)])
            e["quintile_slope"] = float(np.polyfit(pq, aq, 1)[0]) if np.ptp(pq) > 0 else float("nan")
            e["by_site"] = {k: wslope(s[f"y_{name}"].to_numpy(), s[f"x_{st}_{name}"].to_numpy(), s[f"w_{name}"].to_numpy())
                            for k, s in f.groupby("site")}
            out[f"{name}|{st}"] = e
    return out


# --------------------------------------------------------------------------- margin responsiveness
def team_view(d: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({"q": d["q_h"], "x": X, "y": d["margin"], "c": d["close"]}),
                      pd.DataFrame({"q": d["q_a"], "x": -X, "y": -d["margin"], "c": -d["close"]})], ignore_index=True)


def margin_by_quintile(d: pd.DataFrame, X: np.ndarray) -> dict:
    tv = team_view(d, X)
    out = {}
    for q, s in tv.groupby("q"):
        out[int(q)] = {"n": int(len(s)), "pred": float(s["x"].mean()), "act": float(s["y"].mean()),
                       "close": float(s["c"].mean()), "slope_within": A.slope(s["y"].to_numpy(), s["x"].to_numpy())}
    pq = np.array([out[q]["pred"] for q in sorted(out)]); aq = np.array([out[q]["act"] for q in sorted(out)])
    out["quintile_slope"] = float(np.polyfit(pq, aq, 1)[0])
    return out


def kd_team_quintile(d: pd.DataFrame, X: np.ndarray, Z: dict) -> dict:
    """L1 k decomposition in the TEAM view, within each team prior quintile (games appear once per team)."""
    out = {}
    tv_y = np.concatenate([d["margin"].to_numpy(float), -d["margin"].to_numpy(float)])
    tv_x = np.concatenate([X, -X])
    tv_q = np.concatenate([d["q_h"].to_numpy(), d["q_a"].to_numpy()])
    tz = {k: np.concatenate([v, -v]) for k, v in Z.items()}
    for q in range(5):
        m = tv_q == q
        kk = A.kdecomp(tv_y[m], tv_x[m], {k: v[m] for k, v in tz.items()})
        out[q] = {"n": int(m.sum()), "underpowered": bool(m.sum() < 600),
                  **{k: v for k, v in kk.items() if k.startswith(("k_", "b_")) or k in ("slope", "sd_X")}}
    return out


def part_attrib() -> dict:
    chk = check_inputs()
    print("input check:", chk, flush=True)
    d = frame()
    Y = d["margin"].to_numpy(float); C = d["close"].to_numpy(float); lined = d["close"].notna().to_numpy()
    allm = np.ones(len(d), bool)
    res = {"input_check": chk, "n_games": len(d), "n_lined": int(lined.sum())}
    for st in ("V2", "S0"):
        X = d[f"X_{st}"].to_numpy(float)
        res[f"slope_{st}"] = {"Y": A.slope(Y, X), "C": A.slope(C[lined], X[lined]), "sd": float(np.std(X, ddof=1))}
    m, l1, l2 = l1_components("S0", d)
    Xh = m["FULL"].to_numpy(float)
    res["harness"] = {"sd": float(np.std(Xh, ddof=1)), "slope_Y": A.slope(Y, Xh), "slope_C": A.slope(C[lined], Xh[lined]),
                      "corr_V2": float(np.corrcoef(Xh, d["X_V2"])[0, 1]), "slope_V2_on_Xh": A.slope(d["X_V2"].to_numpy(), Xh)}
    # harness-only and sim-anchored L1 / L2
    for st in ("V2", "S0"):
        X = d[f"X_{st}"].to_numpy(float)
        Z1 = dict(l1); Z1["not_harnessed"] = X - Xh
        Z2 = dict(l2); Z2["not_harnessed"] = X - Xh
        for pname, Z in (("L1", Z1), ("L2", Z2)):
            for ref, R, mask in (("Y", Y, allm), ("C", C, lined)):
                kk, se = kd_with_se(R, X, Z, mask)
                mc = float(np.mean(d.loc[mask, f"XSD_{st}"] ** 2 / d.loc[mask, f"NS_{st}"]) / np.var(X[mask], ddof=1))
                kk["k_mc_expected"] = mc
                res[f"sim_{st}_{pname}_{ref}"] = {"est": kk, "se": se}
        print(f"[attrib] {st}: slope {res[f'slope_{st}']['Y']:.4f}; " + ", ".join(
            f"{k} {res[f'sim_{st}_L1_Y']['est'][f'k_{k}']:+.4f}(b {res[f'sim_{st}_L1_Y']['est'][f'b_{k}']:.2f})"
            for k in ("PO", "FG", "RB", "RAT", "not_harnessed")), flush=True)
    for pname, Z in (("L1", l1),):
        for ref, R, mask in (("Y", Y, allm), ("C", C, lined)):
            kk, se = kd_with_se(R, Xh, Z, mask)
            res[f"harness_{pname}_{ref}"] = {"est": kk, "se": se}
    # paired V2 - S0 (sim-anchored, same harness components; only not_harnessed and X differ)
    X2, X0 = d["X_V2"].to_numpy(float), d["X_S0"].to_numpy(float)

    def dk(idx):
        Z2 = {k: v[idx] for k, v in l1.items()}; Z2["not_harnessed"] = (X2 - Xh)[idx]
        Z0 = {k: v[idx] for k, v in l1.items()}; Z0["not_harnessed"] = (X0 - Xh)[idx]
        a = A.kdecomp(Y[idx], X2[idx], Z2); b = A.kdecomp(Y[idx], X0[idx], Z0)
        return {k: a[k] - b[k] for k in a if k.startswith(("k_", "b_")) or k == "slope"}
    res["V2_minus_S0_L1_Y"] = {"est": dk(np.arange(len(Y))), "se": A.boot(dk, len(Y), NB)}
    # segments (sim-anchored, V2, Y lens)
    X = X2
    Z1 = dict(l1); Z1["not_harnessed"] = X - Xh
    seg = {}
    for sname, col in (("month", "mon"), ("site", "site"), ("qgap", "qgap")):
        for val, sub in d.groupby(col):
            ix = sub.index.to_numpy()
            if len(ix) < 60:
                continue
            kk = A.kdecomp(Y[ix], X[ix], {k: v[ix] for k, v in Z1.items()})
            e = {"n": int(len(ix)), "underpowered": bool(len(ix) < 300),
                 **{k: v for k, v in kk.items() if k.startswith(("k_", "b_")) or k in ("slope", "sd_X")}}
            se = A.boot(lambda idx: {k: v for k, v in A.kdecomp(Y[ix][idx], X[ix][idx],
                                                                 {q: z[ix][idx] for q, z in Z1.items()}).items()
                                     if k in ("slope", "k_PO", "k_FG", "b_PO", "b_FG")}, len(ix), 100)
            e["se"] = se
            seg[f"{sname}={val}"] = e
    res["segments_V2"] = seg
    res["team_quintile_V2"] = kd_team_quintile(d, X, Z1)
    res["margin_by_quintile"] = {"V2": margin_by_quintile(d, X2), "S0": margin_by_quintile(d, X0),
                                 "harness": margin_by_quintile(d, Xh)}
    # per possession type: sim team-game rate calibration (V2 vs S0) and harness rate / ratings parts
    Yact, _, _ = M1.build_actual(P_REF)
    Xs = {"V2": M1.build_sim(V2READ, P_REF, "V2"), "S0": M1.build_sim(S0READ, P_REF, "S0")}
    rf = rate_frame(Yact, Xs, d)
    res["rates_sim"] = rate_table(rf, ("V2", "S0"))
    res["rates_harness"] = A.part_rates(d)["S0"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attrib_v2_v1.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("wrote", OUT / "attrib_v2_v1.json", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["attrib"], required=True)
    a = ap.parse_args()
    if a.part == "attrib":
        part_attrib()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
