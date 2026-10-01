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


def _fe_hca(df: pd.DataFrame, ycol: str) -> tuple[float, float]:
    """FE HCA on (game, offence) aggregates of a row-level frame with columns
    game_id, off_team_id, def_team_id, site (+1/0/-1), ycol."""
    import grade_home_site_v1 as GR
    d = pd.DataFrame({"game_id": df["game_id"].to_numpy(), "off_id": df["off_team_id"].to_numpy(),
                      "def_id": df["def_team_id"].to_numpy(), "site": df["site"].to_numpy(),
                      "y": df[ycol].to_numpy(dtype=float), "p": df[ycol].to_numpy(dtype=float), "w": 1.0})
    g = GR.aggregate(d)
    fe = GR.FE(g)
    return fe.hca(g["y"].to_numpy() / g["w"].to_numpy(), g["w"].to_numpy())


def part_fgdiag():
    """Step 2: why fg_make over-produces home advantage. Uses the S0 (served B1
    spec) held-out predictions written by train_fg_make_v4_site.py, with the
    per-row counterfactuals p_neutral (site columns zeroed) and p_flip."""
    meta = json.loads((OUT / "audit_v1.json").read_text())
    P_ref, dv = meta["P_ref"], meta["deriv"]
    dkey = {"FGA_rim": "p_rim", "FGA_jump2": "p_jump", "FGA_3": "p_3"}
    res = {}
    for fold in ("F2", "F1"):
        f = Path(f"results/home_site/fg/preds_S0_{fold}_s0.parquet")
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        d["site"] = (d["site_home"] - d["site_away"]).astype(int)
        d["month"] = pd.to_datetime(d["game_date"]).dt.month
        d["eff"] = d["p"] - d["p_neutral"]          # the model's own site term, per row
        d["resid"] = d["p"] - d["y"]
        # game-level home-minus-away rating gap, for strength-mismatch segments
        hg = d[d["site"] == 1].groupby("game_id")["rating_gap"].mean()
        d["home_gap"] = d["game_id"].map(hg)
        nn = d[d["site"] != 0]
        q = pd.qcut(nn["home_gap"].rank(method="first"), 5, labels=False)
        d.loc[nn.index, "gap_q"] = q
        out = {}
        for c, dc in d.groupby("shot_class"):
            k = dkey[c]
            pts = P_ref * dv[k]
            r = {}
            h_real, se = _fe_hca(dc, "y")
            h_pred, _ = _fe_hca(dc, "p")
            h_neu, _ = _fe_hca(dc, "p_neutral")
            r["fe"] = {"hca_real": h_real, "se": se, "hca_pred": h_pred, "hca_pred_site_zeroed": h_neu,
                       "gap_pts": pts * (h_pred - h_real), "se_pts": pts * se,
                       "model_site_term_pts": pts * (h_pred - h_neu),
                       "non_site_part_pts": pts * (h_neu - h_real)}
            nnc = dc[dc["site"] != 0]
            def seg_rows(key):
                rows = []
                for gk, s in nnc.groupby(key):
                    hm = s["site"] == 1
                    if hm.sum() < 500 or (~hm).sum() < 500:
                        continue
                    rows.append({"seg": str(gk), "n_home": int(hm.sum()), "n_away": int((~hm).sum()),
                                 "real_HmA": float(s.loc[hm, "y"].mean() - s.loc[~hm, "y"].mean()),
                                 "pred_HmA": float(s.loc[hm, "p"].mean() - s.loc[~hm, "p"].mean()),
                                 "resid_HmA": float(s.loc[hm, "resid"].mean() - s.loc[~hm, "resid"].mean()),
                                 "resid_HmA_se": float(np.sqrt(s.loc[hm, "y"].var() / hm.sum() + s.loc[~hm, "y"].var() / (~hm).sum())),
                                 "model_site_term_HmA": float(s.loc[hm, "eff"].mean() - s.loc[~hm, "eff"].mean())})
                return rows
            r["by_conf"] = seg_rows("conf_game")
            r["by_gap_quintile"] = seg_rows("gap_q")
            r["by_month"] = seg_rows("month")
            r["by_season_type"] = seg_rows("season_type")
            # FE HCA conf vs nonconf
            for nm, m in (("conf", dc["conf_game"].to_numpy()), ("nonconf", ~dc["conf_game"].to_numpy())):
                sub = dc[m]
                if len(sub) > 10000:
                    hr, hse = _fe_hca(sub, "y"); hp, _ = _fe_hca(sub, "p"); hn, _ = _fe_hca(sub, "p_neutral")
                    r[f"fe_{nm}"] = {"hca_real": hr, "se": hse, "hca_pred": hp, "hca_site_zeroed": hn,
                                     "gap_pts": pts * (hp - hr), "se_pts": pts * hse,
                                     "model_site_term_pts": pts * (hp - hn)}
            out[c] = r
        res[fold] = out
    # realised FE HCA by SEASON over the full design (training-sample check)
    import sys as _s
    _s.path.insert(0, str(ROOT / "scripts"))
    des = pd.read_parquet("data/processed/models/fg_make/design_v2_shotshooter.parquet",
                          columns=["game_id", "cbbd_game_id", "season", "off_team_id", "def_team_id",
                                   "site_home", "site_away", "shot_class", "y", "off_rating_off_c",
                                   "off_rating_def_c", "def_rating_off_c", "def_rating_def_c"])
    des["site"] = (des["site_home"] - des["site_away"]).astype(int)
    cg = pd.concat([pd.read_parquet(f"data/raw/cbbd/games_{s}.parquet", columns=["id", "conferenceGame"])
                    for s in (2022, 2023, 2024, 2025)]).drop_duplicates("id")
    des = des.merge(cg.rename(columns={"id": "cbbd_game_id"}), on="cbbd_game_id", how="left")
    des["conf_game"] = des["conferenceGame"].fillna(False).astype(bool)
    by_season = []
    for (s, c), dsc in des.groupby(["season", "shot_class"]):
        h, se = _fe_hca(dsc, "y")
        row = {"season": int(s), "class": c, "hca_real": h, "se": se,
               "share_rows_nonconf_home": float(((dsc["site"] == 1) & ~dsc["conf_game"]).mean()),
               "share_rows_nonconf": float((~dsc["conf_game"]).mean())}
        for nm, m in (("conf", dsc["conf_game"]), ("nonconf", ~dsc["conf_game"])):
            hh, ss = _fe_hca(dsc[m.to_numpy()], "y")
            row[f"hca_{nm}"] = hh; row[f"se_{nm}"] = ss
        # strength mismatch of home vs away rows: mean offence-minus-defence rating gap
        off_net = dsc["off_rating_off_c"] - dsc["off_rating_def_c"]
        def_net = dsc["def_rating_off_c"] - dsc["def_rating_def_c"]
        gap = off_net - def_net
        row["mean_gap_home_rows"] = float(gap[dsc["site"] == 1].mean())
        row["mean_gap_home_rows_nonconf"] = float(gap[(dsc["site"] == 1) & ~dsc["conf_game"]].mean())
        by_season.append(row)
    res["by_season_realised"] = by_season
    (OUT / "fgdiag_v1.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    for fold in ("F2", "F1"):
        if fold not in res:
            continue
        for c, r in res[fold].items():
            print(f"== {fold} {c} FE: {json.dumps({k: round(v, 5) for k, v in r['fe'].items()})}")
            for k in ("fe_conf", "fe_nonconf"):
                if k in r:
                    print(f"   {k}: {json.dumps({kk: round(v, 5) for kk, v in r[k].items()})}")
            for key in ("by_conf", "by_gap_quintile", "by_month", "by_season_type"):
                print(f"   {key}:")
                print(pd.DataFrame(r[key]).round(5).to_string(index=False))
    print(pd.DataFrame(by_season).round(5).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", required=True)
    a = ap.parse_args()
    for p in a.part:
        {"audit": part_audit, "fgdiag": part_fgdiag}[p]()


if __name__ == "__main__":
    main()
