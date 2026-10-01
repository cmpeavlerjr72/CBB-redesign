"""diag_home_site_audit_v1.py -- Lane G (overnight 2026-09-30): home/away/neutral
audit of every served scoring-stage sub-model, and the fg_make home-excess trace.

DIAGNOSTIC ONLY. Reads existing outputs; fits nothing that is served; writes
only under results/home_site/.

Part `audit`:
  For each served sub-model rate, the team-strength-adjusted (offence FE +
  defence FE) home and away site coefficients of (a) the REALISED per-game rate
  (hoopR box, results/g9g6_diag/actual_channels_v1.parquet), (b) the served
  model's OFFLINE prediction (results/g9ws_diag/preds_v1.parquet variant F0:
  served artifacts at fixed reference states), and (c) the 200-seed SIM
  expected rate (results/g9g6_diag/sim_channels_A_v1.parquet). HCA = b_home -
  b_away (offence at home vs offence away; one game contributes one row per
  side, so the offence AND defence perspective are both inside the fit).
  Points = P_ref * dPPP/drate * dHCA, as in diag_g9_g6_margin_v1 part analyse.

Usage:
    .venv/Scripts/python.exe scripts/diag_home_site_audit_v1.py --part audit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import diag_g9_g6_margin_v1 as D  # noqa: E402

OUT = Path("results/home_site")
PREDS = Path("results/g9ws_diag/preds_v1.parquet")

# offline prediction column -> (actual rate key, weight key, owner)
OFF_MAP = {
    "p_tov": ("t", "P", "possession_outcome TOV"),
    "p_trip": ("rp", "P", "possession_outcome FT-trip class (+ foul accrual in sim)"),
    "s_rim": ("s_rim", None, "possession_outcome shot mix (rim)"),
    "s_three": ("s_3", None, "possession_outcome shot mix (3)"),
    "p_oreb": ("rho", "oreb_chances", "rebound OREB%"),
    "p_ft": ("f", "fta", "free_throw FT%"),
    "p_make_rim": ("p_rim", "fga_rim", "fg_make rim"),
    "p_make_jump2": ("p_jump", "fga_jump", "fg_make jump2"),
    "p_make_three": ("p_3", "fga_3", "fg_make three"),
}


def fe_site(df: pd.DataFrame, src: str, key: str, wkey: str | None, mask=None):
    """offence FE + defence FE + home + away (non-neutral) site terms, weighted.
    Returns b_home, b_away, se of (b_home - b_away) (OLS, weighted)."""
    recs = []
    for side in ("h", "a"):
        sub = pd.DataFrame({
            "off": df["home_team_id"] if side == "h" else df["away_team_id"],
            "dfn": df["away_team_id"] if side == "h" else df["home_team_id"],
            "y": df[f"{src}_{side}_{key}"],
            "w": 1.0 if wkey is None else (df[f"Y_{side}_P"] if wkey == "P" else df[f"Y_{side}_n_{wkey}"]),
            "sh": ((df["neutral"] == 0) & (side == "h")).astype(float),
            "sa": ((df["neutral"] == 0) & (side == "a")).astype(float),
        })
        if mask is not None:
            sub = sub[np.asarray(mask)]
        recs.append(sub)
    s = pd.concat(recs, ignore_index=True).dropna()
    s = s[s["w"] > 0]
    teams = np.unique(np.concatenate([s["off"], s["dfn"]]))
    ti = {t: i for i, t in enumerate(teams)}
    n, T = len(s), len(teams)
    X = np.zeros((n, 3 + 2 * T))
    X[:, 0] = 1; X[:, 1] = s["sh"]; X[:, 2] = s["sa"]
    X[np.arange(n), 3 + s["off"].map(ti).to_numpy()] = 1
    X[np.arange(n), 3 + T + s["dfn"].map(ti).to_numpy()] = 1
    w = s["w"].to_numpy(float)
    sw = np.sqrt(w)
    A = X * sw[:, None]; b = s["y"].to_numpy(float) * sw
    reg = np.zeros(X.shape[1]); reg[3:] = 1e-6 * float(w.mean())
    XtX = A.T @ A + np.diag(reg)
    beta = np.linalg.solve(XtX, A.T @ b)
    r = b - A @ beta
    s2 = float(r @ r / max(n - X.shape[1], 1))
    cov = s2 * np.linalg.inv(XtX)
    var_d = cov[1, 1] + cov[2, 2] - 2 * cov[1, 2]
    return float(beta[1]), float(beta[2]), float(np.sqrt(max(var_d, 0.0)))


def load_audit_frame():
    df, meta = D.load_frame("A")
    ok = df.dropna(subset=["Y_tov"]).copy()
    for s in ("X", "Y"):
        for sd in ("h", "a"):
            ok[f"{s}_{sd}_n_oreb_chances"] = ok[f"{s}_{sd}_n_oreb"] + ok[f"{s}_{sd}_n_oppdreb"]
    p = pd.read_parquet(PREDS)
    p = p[p["variant"] == "F0"]
    cols = list(OFF_MAP)
    h = p[p["side"] == 0].set_index("game_id")[cols].add_prefix("O_h_")
    a = p[p["side"] == 1].set_index("game_id")[cols].add_prefix("O_a_")
    ok = ok.merge(h, left_on="game_id", right_index=True, how="left")
    ok = ok.merge(a, left_on="game_id", right_index=True, how="left")
    # rename offline columns to the actual-rate key
    for c, (k, _, _) in OFF_MAP.items():
        for sd in ("h", "a"):
            ok[f"O_{sd}_{k}"] = ok.pop(f"O_{sd}_{c}")
    return ok.reset_index(drop=True), meta


def deriv_of(ok: pd.DataFrame):
    ref = {}
    for k in ("t", "rp", "f", "rho", "m", "s_rim", "s_jump", "s_3", "p_rim", "p_jump", "p_3"):
        ref[k] = float(pd.concat([ok[f"Y_h_{k}"], ok[f"Y_a_{k}"]]).mean())
    V = D.V
    Qr = sum(V[k] * ref[f"s_{k}"] * ref[f"p_{k}"] for k in D.KS)
    Ar = 1 + ref["rho"] * ref["m"] - ref["t"] - 0.44 * ref["rp"]
    return ref, {"t": -Qr, "rp": ref["f"] - 0.44 * Qr, "f": ref["rp"], "rho": Qr * ref["m"],
                 "s_3": Ar * (3 * ref["p_3"] - 2 * ref["p_jump"]),
                 "s_rim": Ar * (2 * ref["p_rim"] - 2 * ref["p_jump"]),
                 "p_rim": Ar * 2 * ref["s_rim"], "p_jump": Ar * 2 * ref["s_jump"],
                 "p_3": Ar * 3 * ref["s_3"]}


def part_audit():
    OUT.mkdir(parents=True, exist_ok=True)
    ok, meta = load_audit_frame()
    P_ref = meta["P_ref"]
    ref, dv = deriv_of(ok)
    rows = []
    for c, (k, w, owner) in OFF_MAP.items():
        # offline predictions are at a fixed reference state: rescale to the
        # realised league level so a level offset cannot masquerade as site
        lvl_o = float(pd.concat([ok[f"O_h_{k}"], ok[f"O_a_{k}"]]).mean())
        scale = ref[k] / lvl_o
        for sd in ("h", "a"):
            ok[f"Os_{sd}_{k}"] = ok[f"O_{sd}_{k}"] * scale
        bhY, baY, seY = fe_site(ok, "Y", k, w)
        bhO, baO, seO = fe_site(ok, "Os", k, w)
        bhX, baX, seX = fe_site(ok, "X", k, w)
        r = {"rate": k, "owner": owner, "offline_scale": scale,
             "act_hca": bhY - baY, "act_se": seY, "off_hca": bhO - baO, "sim_hca": bhX - baX,
             "act_home": bhY, "act_away": baY, "off_home": bhO, "off_away": baO,
             "sim_home": bhX, "sim_away": baX, "deriv": dv[k]}
        for nm in ("act", "off", "sim"):
            r[f"{nm}_pts"] = P_ref * dv[k] * r[f"{nm}_hca"]
        r["gap_off_pts"] = r["off_pts"] - r["act_pts"]
        r["gap_sim_pts"] = r["sim_pts"] - r["act_pts"]
        r["gap_off_se_pts"] = abs(P_ref * dv[k] * seY)
        rows.append(r)
    # rates the offline harness does not carry: jump share, m (chances), P
    for k, w in (("s_jump", None), ("m", "P")):
        bhY, baY, seY = fe_site(ok, "Y", k, w)
        bhX, baX, seX = fe_site(ok, "X", k, w)
        rows.append({"rate": k, "owner": "derived" if k == "m" else "possession_outcome shot mix (jump)",
                     "act_hca": bhY - baY, "act_se": seY, "sim_hca": bhX - baX})
    res = {"P_ref": P_ref, "ref_rates": ref, "deriv": dv, "rows": rows, "n_games": len(ok)}
    # by conference / non-conference for the fg_make rates
    seg = {}
    for nm, m in (("conf", ok["conferenceGame"].fillna(False).astype(bool).to_numpy()),
                  ("nonconf", ~ok["conferenceGame"].fillna(False).astype(bool).to_numpy())):
        srows = []
        for c, (k, w, owner) in OFF_MAP.items():
            bhY, baY, seY = fe_site(ok, "Y", k, w, mask=m)
            bhO, baO, _ = fe_site(ok, "Os", k, w, mask=m)
            bhX, baX, _ = fe_site(ok, "X", k, w, mask=m)
            srows.append({"rate": k, "act_hca": bhY - baY, "act_se": seY, "off_hca": bhO - baO,
                          "sim_hca": bhX - baX,
                          "gap_off_pts": P_ref * dv[k] * ((bhO - baO) - (bhY - baY)),
                          "gap_sim_pts": P_ref * dv[k] * ((bhX - baX) - (bhY - baY)),
                          "se_pts": abs(P_ref * dv[k] * seY)})
        seg[nm] = {"n_games": int(m.sum()), "rows": srows}
    res["by_conf"] = seg
    (OUT / "audit_v1.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t[["rate", "owner", "act_hca", "act_se", "off_hca", "sim_hca", "act_pts", "off_pts",
             "sim_pts", "gap_off_pts", "gap_sim_pts", "gap_off_se_pts"]].round(4).to_string())
    for nm, v in seg.items():
        print(nm, v["n_games"])
        print(pd.DataFrame(v["rows"]).round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", required=True)
    a = ap.parse_args()
    for p in a.part:
        {"audit": part_audit}[p]()


if __name__ == "__main__":
    main()
