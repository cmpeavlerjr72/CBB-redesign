#!/usr/bin/env python
"""diag_gamezone_frame_v1.py -- GameZone frame calibration to the CBBD folded frame, and within-vendor
reliability of team band profiles.  DESCRIPTIVE ONLY: no recommendation.  PRIVATE INTERNAL RESEARCH DATA.

(1) Per-season least-squares maps from GameZone (loc_x, loc_y) to the CBBD folded frame, fit on joined
    jump2 + three attempts (2022-2024; rim pixels are canned): per-axis scale+offset and a full 2x2 affine.
    Residual distances, then part (c) and (d) of the agreement doc re-measured before/after.
(2) Split-half (odd/even games, Spearman-Brown) reliability, between-team SD vs within-team binomial SD of
    team band shares, offence and defence, CBBD 2022-2025, GameZone 2022-2024, and CBBD restricted to the
    GameZone-matched games.

Needs: diag_gamezone_inventory_v1.py and diag_gamezone_vs_espn_v1.py outputs (results/gamezone/*), audit masters.
Frame note: CBBD folded frame = (u, v) = (xf / 10, y / 10) ft, u from the baseline, basket (5.25, 25.0). The fold
mirrors the lateral axis at the far end, so the CBBD lateral coordinate is first aligned to GameZone's orientation
(v_al = y/10 if x > 470 else 50 - y/10; the correlation with loc_x is then positive at both ends).

Run from the repo root:  .venv/Scripts/python.exe -I scripts/diag_gamezone_frame_v1.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DOCS = ROOT / "docs/tests"


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


V = _load("gz_vs_espn", "diag_gamezone_vs_espn_v1.py")
I = V.I
A = V.A
T = A.md_table
pct = I.pct
SEASONS = V.SEASONS
BASKET = (5.25, 25.0)
BANDS = {"jump2": ["short2", "long2"], "three": ["corner3", "wing3", "top3"]}
MIN200 = 200
HALF_MIN = 20


def flag(n, k=MIN200):
    return "UNDERPOWERED" if n < k else ""


# ------------------------------------------------------------------ calibration
def fit_maps(d):
    """d: joined jump2+three, one season. Returns per-axis and full affine parameters."""
    P, Q = d["loc_y"].to_numpy(float), d["loc_x"].to_numpy(float)       # GameZone along, lateral
    U, W = d["u"].to_numpy(float), d["v_al"].to_numpy(float)            # CBBD along, aligned lateral
    X1 = np.c_[P, np.ones(len(P))]
    X2 = np.c_[Q, np.ones(len(Q))]
    au, bu = np.linalg.lstsq(X1, U, rcond=None)[0]
    av, bv = np.linalg.lstsq(X2, W, rcond=None)[0]
    Xf = np.c_[P, Q, np.ones(len(P))]
    cu = np.linalg.lstsq(Xf, U, rcond=None)[0]
    cv = np.linalg.lstsq(Xf, W, rcond=None)[0]
    def deming(x, y):
        sxx, syy = np.var(x), np.var(y)
        sxy = np.mean((x - x.mean()) * (y - y.mean()))
        sl = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
        return sl, y.mean() - sl * x.mean()
    tu, tbu = deming(P, U)
    tv, tbv = deming(Q, W)
    return {"axis": {"u_scale": au, "u_offset": bu, "v_scale": av, "v_offset": bv},
            "axis_tls": {"u_scale": tu, "u_offset": tbu, "v_scale": tv, "v_offset": tbv},
            "full": {"u_from_along": cu[0], "u_from_lateral": cu[1], "u_offset": cu[2],
                     "v_from_along": cv[0], "v_from_lateral": cv[1], "v_offset": cv[2]}}


def apply_map(kind, par, loc_x, loc_y):
    P, Q = np.asarray(loc_y, float), np.asarray(loc_x, float)
    if kind == "identity":
        return P, Q
    if kind in ("axis", "axis_tls"):
        p = par[kind]
        return p["u_scale"] * P + p["u_offset"], p["v_scale"] * Q + p["v_offset"]
    p = par["full"]
    return p["u_from_along"] * P + p["u_from_lateral"] * Q + p["u_offset"], p["v_from_along"] * P + p["v_from_lateral"] * Q + p["v_offset"]


def geom(u, v):
    du, dv = u - BASKET[0], v - BASKET[1]
    return np.hypot(du, dv), np.degrees(np.arctan2(dv, du)), (v <= 4.0) | (v >= 46.0)


def qs(x):
    x = np.asarray(x, float)
    return {f"p{q}": float(np.quantile(x, q / 100)) for q in (25, 50, 75, 90, 95)} | {"mean": float(x.mean())}


# ------------------------------------------------------------------ band metrics
def band_metrics(cbl, gzm, jb, bandcol):
    """cbl: CBBD located attempts in matched games (with band, team); gzm: GameZone located attempts in matched
    games (with `bandcol`, team_o); jb: joined both-located same-class frame with band_c and band_g."""
    out = {}
    d = jb[(jb["cls"] == jb["gz_cls"]) & jb["cls"].isin(["jump2", "three"])]
    out["attempt_band_differs_pct"] = {cl: pct((d.loc[d["cls"] == cl, "band_c"] != d.loc[d["cls"] == cl, "band_g"]).sum(), (d["cls"] == cl).sum())
                                       for cl in ("jump2", "three")}
    uc = V.I.band  # noqa
    def units(df, col, team):
        t = df[df[col] != ""].groupby(["gameId", team, col]).size().unstack(fill_value=0)
        for b in ["short2", "long2", "corner3", "wing3", "top3"]:
            if b not in t:
                t[b] = 0
        t["n_j2"] = t["short2"] + t["long2"]
        t["n_3"] = t["corner3"] + t["wing3"] + t["top3"]
        return t.reset_index().rename(columns={team: "team"})
    U = units(cbl, "band", "team").merge(units(gzm, bandcol, "team"), on=["gameId", "team"], suffixes=("_c", "_g"))
    U["season"] = U["gameId"].map(V.GAME_SEASON)
    mad, corr = {}, {}
    for cl, bl, ncol in (("jump2", ["short2", "long2"], "n_j2"), ("three", ["corner3", "wing3", "top3"], "n_3")):
        ok = U[(U[ncol + "_c"] >= 5) & (U[ncol + "_g"] >= 5)]
        ts = U.groupby(["season", "team"])[[b + s for b in bl for s in ("_c", "_g")] + [ncol + "_c", ncol + "_g"]].sum().reset_index()
        okt = ts[(ts[ncol + "_c"] >= 100) & (ts[ncol + "_g"] >= 100)]
        for b in bl:
            sc, sg = ok[b + "_c"] / ok[ncol + "_c"], ok[b + "_g"] / ok[ncol + "_g"]
            mad[b] = 100 * float((sc - sg).abs().mean())
            tc, tg = okt[b + "_c"] / okt[ncol + "_c"], okt[b + "_g"] / okt[ncol + "_g"]
            corr[b] = float(np.corrcoef(tc, tg)[0, 1])
    out["team_game_mean_abs_diff_pp"] = mad
    out["team_season_corr"] = corr
    out["pooled_share_pct"] = {b: pct((gzm[bandcol] == b).sum(), gzm[bandcol].isin(BANDS["jump2" if b in BANDS["jump2"] else "three"]).sum())
                               for b in ["short2", "long2", "corner3", "wing3", "top3"]}
    return out


# ------------------------------------------------------------------ reliability
def reliability(df, vendor, side):
    """df: attempts with season, team, gameId, date, cls3 in {jump2, three}, band. One row per band."""
    gm = df[["season", "team", "gameId", "date"]].drop_duplicates().sort_values(["season", "team", "date", "gameId"])
    gm["par"] = gm.groupby(["season", "team"]).cumcount() % 2
    d = df.merge(gm[["season", "team", "gameId", "par"]], on=["season", "team", "gameId"])
    rows = []
    for cl, bl in BANDS.items():
        dc = d[d["cls3"] == cl]
        cnt = dc.groupby(["season", "team", "par", "band"]).size().unstack(["par", "band"], fill_value=0)
        for b in bl[:1] if cl == "jump2" else bl:
            ko, ke = cnt.get((0, b), 0), cnt.get((1, b), 0)
            no = sum(cnt.get((0, x), 0) for x in bl)
            ne = sum(cnt.get((1, x), 0) for x in bl)
            t = pd.DataFrame({"ko": ko, "ke": ke, "no": no, "ne": ne}).reset_index()
            t["n"] = t["no"] + t["ne"]
            t["p"] = (t["ko"] + t["ke"]) / t["n"]
            t["po"], t["pe"] = t["ko"] / t["no"], t["ke"] / t["ne"]
            for scope, s in (("n>=200", t[(t["n"] >= MIN200) & (t["no"] >= HALF_MIN) & (t["ne"] >= HALF_MIN)]),
                             ("n<200 (UNDERPOWERED)", t[(t["n"] < MIN200) & (t["no"] >= HALF_MIN) & (t["ne"] >= HALF_MIN)])):
                if len(s) < 5:
                    rows.append({"vendor": vendor, "side": side, "class": cl, "band": b, "scope": scope, "team_seasons": len(s)})
                    continue
                dm = lambda c: s[c] - s.groupby("season")[c].transform("mean")
                r = float(np.corrcoef(dm("po"), dm("pe"))[0, 1])
                sb = 2 * r / (1 + r) if r > -1 else float("nan")
                pf = dm("p") + s["p"].mean()
                nmed = float(s["n"].median())
                sd_b = float(dm("p").std())
                pbar = float(s["p"].mean())
                sd_w = float(np.sqrt(pbar * (1 - pbar) / nmed))
                mean_var = float((s["p"] * (1 - s["p"]) / s["n"]).mean())
                rows.append({"vendor": vendor, "side": side, "class": cl, "band": b, "scope": scope, "team_seasons": len(s),
                             "median_attempts": nmed, "split_half_r": r, "spearman_brown": sb, "between_team_sd_pp": 100 * sd_b,
                             "binomial_sd_at_median_n_pp": 100 * sd_w, "sd_ratio": sd_b / sd_w if sd_w > 0 else float("nan"),
                             "signal_variance_share": float(max(0.0, 1 - mean_var / sd_b ** 2)) if sd_b > 0 else float("nan"),
                             "flag": "UNDERPOWERED" if scope.startswith("n<200") or len(s) < 30 else ""})
    return rows


def main() -> int:
    out = {"script": "scripts/diag_gamezone_frame_v1.py"}
    gz = V.load_gz()
    cb = V.load_cbbd()
    games = pd.concat([pd.read_parquet(V.CBBD / f"games_{s}.parquet", columns=["id", "season", "homeTeamId", "awayTeamId"]) for s in (2022, 2023, 2024, 2025)]).rename(columns={"id": "gameId"})
    V.GAME_SEASON = games.set_index("gameId")["season"]
    matched = gz[gz["gameId"].notna() & gz["dseason"].isin(SEASONS)].copy()
    matched["gameId"] = matched["gameId"].astype(int)
    mg = set(matched["gameId"])
    c = cb[cb["gameId"].isin(mg)].reset_index(drop=True)
    g = matched.reset_index(drop=True)
    pr, _ = V.join_plays(c, g)
    c = c.assign(cid=np.arange(len(c)))
    g = g.assign(gid=np.arange(len(g)))
    J = pr.merge(c, on="cid").merge(g, on="gid", suffixes=("_c", "_g"))
    both = J[J["located_c"] & J["located_g"]].copy()
    both["u"] = both["xf"] / 10.0
    both["v_al"] = np.where(both["x"] > 470, both["y"], 500.0 - both["y"]) / 10.0
    fitset = both[both["cls"].isin(["jump2", "three"])]
    A_ = A
    out["n_joined_both_located"] = len(both)
    out["n_fit_jump2_three"] = len(fitset)

    pars, par_rows, res_rows = {}, [], []
    for s in SEASONS:
        d = fitset[fitset["season"] == s]
        par = fit_maps(d)
        pars[s] = par
        par_rows.append({"season": s, "n_fit": len(d), **{f"axis_{k}": v for k, v in par["axis"].items()}})
        par_rows[-1].update({f"full_{k}": v for k, v in par["full"].items()})
        par_rows[-1].update({f"tls_{k}": v for k, v in par["axis_tls"].items()})
        for kind in ("identity", "axis", "full", "axis_tls"):
            u, v = apply_map(kind, par, d["loc_x"], d["loc_y"])
            res = np.hypot(u - d["u"].to_numpy(), v - d["v_al"].to_numpy())
            dg, _, _ = geom(u, v)
            dd = np.abs(dg - np.hypot(d["u"].to_numpy() - BASKET[0], d["v_al"].to_numpy() - BASKET[1]))
            res_rows.append({"season": s, "map": kind, "n": len(d), "resid_dist_ft": qs(res), "abs_basket_distance_diff_ft": qs(dd),
                             "basket_dist_within_1ft_pct": pct((dd <= 1).sum(), len(dd)), "basket_dist_within_3ft_pct": pct((dd <= 3).sum(), len(dd)),
                             "basket_dist_gt6ft_pct": pct((dd > 6).sum(), len(dd)),
                             "u_resid_sd": float(np.std(u - d["u"].to_numpy())), "v_resid_sd": float(np.std(v - d["v_al"].to_numpy()))})
    out["fit_parameters"] = par_rows
    flatres = [{"season": r["season"], "map": r["map"], "n": r["n"], **{"resid_" + k: v for k, v in r["resid_dist_ft"].items()},
                **{"abs_dbasket_" + k: v for k, v in r["abs_basket_distance_diff_ft"].items()},
                "d_within_1ft_pct": r["basket_dist_within_1ft_pct"], "d_within_3ft_pct": r["basket_dist_within_3ft_pct"], "d_gt6ft_pct": r["basket_dist_gt6ft_pct"],
                "u_resid_sd": r["u_resid_sd"], "v_resid_sd": r["v_resid_sd"]} for r in res_rows]
    out["residuals"] = flatres

    # ---- (c) before / after
    cbl = cb[cb["located"] & cb["gameId"].isin(mg)].copy()
    cbl["band"] = I.band(cbl["cls"], cbl["dist"], np.degrees(np.arctan2(cbl["y"] - 250.0, cbl["xf"] - 52.5)), (cbl["y"] <= 40) | (cbl["y"] >= 460))
    gzm = gz[gz["located"] & gz["gameId"].notna() & gz["dseason"].isin(SEASONS)].copy()
    gzm["gameId"] = gzm["gameId"].astype(int)
    gzm["side_o"] = np.where(gzm["flip"].astype(bool), np.where(gzm["side"] == "home", "away", "home"), gzm["side"])
    gg = games.drop_duplicates("gameId")
    gzm = gzm.merge(gg[["gameId", "homeTeamId", "awayTeamId"]], on="gameId", how="left")
    gzm["team"] = np.where(gzm["side_o"] == "home", gzm["homeTeamId"], gzm["awayTeamId"])
    gzm["opp"] = np.where(gzm["side_o"] == "home", gzm["awayTeamId"], gzm["homeTeamId"])
    for kind in ("identity", "axis", "full", "axis_tls"):
        uu = np.full(len(gzm), np.nan)
        vv = np.full(len(gzm), np.nan)
        for s in SEASONS:
            m = (gzm["dseason"] == s).to_numpy()
            uu[m], vv[m] = apply_map(kind, pars[s], gzm.loc[m, "loc_x"], gzm.loc[m, "loc_y"])
        dg, th, sl = geom(uu, vv)
        gzm["dist_" + kind], gzm["theta_" + kind], gzm["sl_" + kind] = dg, th, sl
        gzm["band_" + kind] = I.band(gzm["gz_cls"], dg, th, sl)
        # joined attempts
        uj = np.full(len(both), np.nan)
        vj = np.full(len(both), np.nan)
        for s in SEASONS:
            m = (both["season"] == s).to_numpy()
            uj[m], vj[m] = apply_map(kind, pars[s], both.loc[m, "loc_x"], both.loc[m, "loc_y"])
        dj, tj, sj = geom(uj, vj)
        both["band_g_" + kind] = I.band(both["gz_cls"], dj, tj, sj)
        both["dist_g_" + kind] = dj
    both["band_c"] = I.band(both["cls"], both["dist_c"], np.degrees(np.arctan2(both["y"] - 250.0, both["xf"] - 52.5)), (both["y"] <= 40) | (both["y"] >= 460))
    cmp_ = {}
    for kind, lab in (("identity", "before (textbook basket, raw GameZone frame)"), ("axis", "after: per-axis scale+offset"), ("full", "after: full 2x2 affine"), ("axis_tls", "after: per-axis total least squares (Deming, equal error variances; added arm, not an LS fit)")):
        jb = both.assign(band_g=both["band_g_" + kind])
        cmp_[lab] = band_metrics(cbl, gzm, jb, "band_" + kind)
    out["c_before_after"] = cmp_
    rows = []
    for lab, m in cmp_.items():
        rows.append({"frame": lab,
                     "jump2_attempt_band_differs_pct": m["attempt_band_differs_pct"]["jump2"], "three_attempt_band_differs_pct": m["attempt_band_differs_pct"]["three"],
                     **{f"mad_pp_{b}": m["team_game_mean_abs_diff_pp"][b] for b in ["short2", "corner3", "wing3", "top3"]},
                     **{f"corr_ts_{b}": m["team_season_corr"][b] for b in ["short2", "corner3", "wing3", "top3"]},
                     **{f"share_{b}": m["pooled_share_pct"][b] for b in ["short2", "corner3", "wing3", "top3"]}})
    cb_share = {b: pct((cbl["band"] == b).sum(), cbl["band"].isin(BANDS["jump2" if b in BANDS["jump2"] else "three"]).sum()) for b in ["short2", "corner3", "wing3", "top3"]}
    out["c_cbbd_pooled_share_pct"] = cb_share
    out["c_table"] = rows

    # ---- (d) calibrated
    drows = []
    gl = gz[gz["located"] & gz["dseason"].isin(SEASONS)].copy()
    for kind in ("identity", "axis", "full", "axis_tls"):
        for s in SEASONS:
            d = gl[gl["dseason"] == s]
            u, v = apply_map(kind, pars[s], d["loc_x"], d["loc_y"])
            dist, _, _ = geom(u, v)
            th = d["text_cls"].to_numpy() == "three"
            tj = d["text_cls"].to_numpy() == "twojump"
            drows.append({"frame": kind, "season": s, "n_text_three": int(th.sum()), "three_inside_22.15_pct": pct((dist[th] < 22.15).sum(), th.sum()),
                          "three_inside_21.65_pct": pct((dist[th] < 21.65).sum(), th.sum()), "three_inside_20.65_pct": pct((dist[th] < 20.65).sum(), th.sum()),
                          "n_text_twojump": int(tj.sum()), "twojump_outside_22.15_pct": pct((dist[tj] >= 22.15).sum(), tj.sum()),
                          "median_three_dist_ft": float(np.median(dist[th])), "flag": flag(min(th.sum(), tj.sum()), 500)})
    cbj = both[both["text_cls_c"] == "three"]
    for s in SEASONS:
        d = cbj[cbj["season"] == s]
        drows.append({"frame": "CBBD (joined text-threes, own frame)", "season": s, "n_text_three": len(d), "three_inside_22.15_pct": pct((d["dist_c"] < 22.15).sum(), len(d)),
                      "three_inside_21.65_pct": pct((d["dist_c"] < 21.65).sum(), len(d)), "three_inside_20.65_pct": pct((d["dist_c"] < 20.65).sum(), len(d)),
                      "median_three_dist_ft": float(d["dist_c"].median())})
    out["d_calibrated"] = drows

    # ---- (2) reliability
    cbs = []
    for s in (2022, 2023, 2024, 2025):
        m = pd.read_parquet(ROOT / f"results/coordinate_audit/master_{s}.parquet", columns=["season", "gameId", "teamId", "opponentId", "cls", "xf", "y", "dist", "located", "game_date"])
        m = m[m["located"] & m["cls"].isin(["jump2", "three"])]
        cbs.append(m)
    cbr = pd.concat(cbs, ignore_index=True)
    cbr["band"] = I.band(cbr["cls"], cbr["dist"], np.degrees(np.arctan2(cbr["y"] - 250.0, cbr["xf"] - 52.5)), (cbr["y"] <= 40) | (cbr["y"] >= 460))
    cbr["cls3"] = cbr["cls"]
    cbr["date"] = cbr["game_date"]
    cbr = cbr[cbr["band"] != ""]
    rel = []
    for side, tc in (("offence", "teamId"), ("defence", "opponentId")):
        rel += reliability(cbr.assign(team=cbr[tc]), "CBBD 2022-2025", side)
    sub = cbr[cbr["season"].isin(SEASONS) & cbr["gameId"].isin(mg)]
    for side, tc in (("offence", "teamId"), ("defence", "opponentId")):
        rel += reliability(sub.assign(team=sub[tc]), "CBBD 2022-2024, GameZone-matched games only", side)
    gzr = gzm[gzm["band_identity"] != ""].copy()
    for kind, lab in (("identity", "GameZone 2022-2024 (textbook raw frame)"), ("full", "GameZone 2022-2024 (full-affine calibrated)")):
        z = gzr if kind == "identity" else gzm[gzm["band_full"] != ""].copy()
        z["band"] = z["band_" + kind]
        z["cls3"] = z["gz_cls"]
        z["season"] = z["dseason"]
        z = z[z["cls3"].isin(["jump2", "three"])]
        for side, tc in (("offence", "team"), ("defence", "opp")):
            rel += reliability(z.assign(team=z[tc]), lab, side)
    out["reliability"] = rel
    write_doc(out, pd.DataFrame(par_rows), pd.DataFrame(flatres), pd.DataFrame(rows), pd.DataFrame(drows), pd.DataFrame(rel))
    return 0


VERDICTS = {
 "c": ("Calibration by least squares does not bring the vendors closer; it makes the band comparison worse, and the reason is shot-level noise, not a global frame offset. "
       "Even with GameZone read as-is, the mapped point is a median 4.9-5.8 ft (p90 21-26 ft) from the CBBD point for jump2 + three attempts. Ordinary least squares shrinks every slope (along 0.80-0.83, lateral 0.59-0.69) toward the middle of the court, so corner threes disappear (GameZone corner3 share 18.1% before, 0.0% after) and the share of joined attempts whose band differs rises from 26.7% to 29.4% for jumpers and from 37.5% to 47.4% for threes; the full 2x2 affine is the same (off-diagonal terms are about 0.02). Team-game mean absolute band-share differences go from 19.5 / 10.8 / 16.9 / 14.6 points (short2 / corner3 / wing3 / top3) to 19.5 / 21.7 / 19.7 / 19.1; team-season correlations from 0.64 / 0.51 / 0.26 / 0.39 to 0.59 / undefined / 0.01 / 0.35. "
       "The added total-least-squares arm (no shrinkage; slopes along 1.20-1.33, lateral 1.03) is roughly neutral: band differs 27.0% (jumpers) and 40.3% (threes); team-game differences 18.6 / 12.8 / 18.2 / 14.5 points; team-season correlations 0.65 / 0.44 / 0.22 / 0.39; GameZone pooled shares 36.5 / 11.9 / 55.3 / 32.8 against CBBD 41.3 / 21.8 / 44.3 / 33.9. No affine map tried makes the two vendors agree better than the raw textbook frame on corner and wing threes."),
 "d": ("Under the least-squares maps 83-91% of text-threes fall inside 22.15 ft, which shows the map is invalid for this purpose (it compresses the court). Under the total-least-squares map 3.4% / 7.3% / 4.6% of text-threes (2022 / 2023 / 2024) are inside 22.15 ft and 0.4-0.5% of text two-point jumpers outside it, against 2.7% / 2.2% / 3.3% in the raw textbook frame and 1.8% / 1.6% / 1.8% for CBBD on its joined text-threes. Calibration did not improve the inside-the-line share."),
 "rel_cbbd": ("CBBD (training source), team-seasons with at least 200 located attempts in the class: team band profiles are a stable trait at a season's attempt count, most clearly on offence and least for wing threes. Offence Spearman-Brown reliability is 0.89 for short2 (split-half r 0.80, median 421 attempts), 0.79 for corner3, 0.67 for wing3 and 0.80 for top3 (median 649 attempts); the between-team SD of the share (9.9 / 4.5 / 3.9 / 4.7 points) is 4.1 / 2.8 / 2.0 / 2.6 times the binomial SD at the median count (2.4 / 1.6 / 2.0 / 1.9 points). Defence (bands allowed) is weaker: 0.69 / 0.76 / 0.52 / 0.75, ratios 2.6 / 2.5 / 1.7 / 2.4. Restricted to the 40%-of-games GameZone-matched subset, reliabilities fall (offence 0.86 / 0.77 / 0.57 / 0.76; defence 0.55 / 0.69 / 0.57 / 0.67), which shows how fast it drops with fewer attempts. Team-seasons under 200 attempts (UNDERPOWERED) are mostly noise: offence threes 0.15-0.23, defence short2 and wing3 about 0."),
 "rel_gz": ("GameZone alone, at a similar attempt count, the profiles are mostly noise except for short two-point jumpers. Offence reliability is 0.72 for short2, 0.44 for corner3, -0.03 for wing3 and 0.24 for top3; defence 0.35 / 0.36 / -0.03 / 0.30, with 295-337 team-seasons and medians of 470-650 attempts. The between-team SDs are close to CBBD's on the same games (short2 7.4 vs 9.1 points offence) but the split-half correlation is far lower (CBBD on the same games: 0.86 / 0.77 / 0.57 / 0.76 offence), so GameZone adds within-team variation that does not repeat across halves of a season; the full-affine calibrated version does not help (0.70 short2, 0.31 for wing3 and top3, corner3 empty). Under-200 team-seasons are UNDERPOWERED and near zero."),
}


def write_doc(out, par, res, ctab, dtab, rel):
    A.jdump(out, DOCS / "gamezone_frame_calibration_2026-10-27.json")
    cb = out["c_cbbd_pooled_share_pct"]
    r1 = rel[rel["scope"] == "n>=200"].drop(columns=["scope", "flag"], errors="ignore")
    r2 = rel[rel["scope"] != "n>=200"]
    md = f"""# GameZone frame calibration and team band-profile reliability (2026-10-27)

Descriptive only: no recommendation. Private internal research data. Script `scripts/diag_gamezone_frame_v1.py`; follows `docs/tests/gamezone_vs_espn_agreement_2026-10-27.md`. Tables also in `docs/tests/gamezone_frame_calibration_2026-10-27.json`.

## 1. Frame calibration

Per-season least squares from GameZone (loc_y along the court from the baseline, loc_x lateral) to the CBBD folded frame in feet (u = xf/10 along, v = lateral). The CBBD fold mirrors the lateral axis at the far end (raw x > 470), so v is first aligned to GameZone's orientation (v = y/10 at the far end, 50 - y/10 otherwise; the correlation with loc_x is then positive at both ends). Fit set: joined jump2 + three attempts located by both vendors ({out['n_fit_jump2_three']:,} of {out['n_joined_both_located']:,}; rim attempts excluded because their pixels are canned). Two least-squares maps as asked: per-axis scale + offset (u = a*loc_y + b, v = c*loc_x + d) and a full 2x2 affine plus offset. Because the two vendors' coordinates for the same shot differ by several feet in either direction (residuals below), ordinary least squares is attenuated (slopes 0.6-0.8, pulling every shot toward the middle of the court) and is not a valid frame calibration; a third arm, per-axis total least squares (Deming, equal error variance in both vendors, so the slope is not shrunk), is added and labelled as such. Least squares with noise in both coordinates shrinks slopes below their true value (errors-in-variables), so the fitted scales are lower bounds on any true stretch. "identity" = GameZone raw coordinates read directly as the CBBD frame.

Fitted parameters:

{T(par)}
Residual distance between the mapped GameZone point and the CBBD point (ft), and agreement of the distance to the basket (5.25, 25.0) after mapping, fit-set attempts:

{T(res)}
### (c) re-measured, before and after calibration

Joined attempts of the same class (attempt-level), co-located team-games with at least 5 attempts of the class under both vendors (team-game mean absolute band-share difference, points), team-seasons with at least 100 under both (team-season correlation), and GameZone pooled band shares (CBBD pooled: short2 {cb['short2']:.1f}%, corner3 {cb['corner3']:.1f}%, wing3 {cb['wing3']:.1f}%, top3 {cb['top3']:.1f}%):

{T(ctab)}
Verdict (c): {VERDICTS['c']}

### (d) re-reported under the calibrated frame

Text threes (vendor `desc`) inside the line and text jumpers outside it, all located GameZone attempts of the season, distance to the CBBD basket in the mapped frame; the CBBD rows are the joined text-threes in CBBD's own frame:

{T(dtab)}
Verdict (d): {VERDICTS['d']}

## 2. Within-vendor reliability of team band profiles

Unit = team-season. Attempts: located jump2 (bands short2 / long2; short2 shown, long2 is its complement) and located three (corner3 / wing3 / top3), bands as in the spec (section 2 of the draft). Offence = the team's own attempts; defence = bands allowed (opponent attempts). Games are ordered by date and split odd / even; each half needs at least {HALF_MIN} attempts of the class. Shares are demeaned by season before correlating so that league drift does not inflate the correlation. `spearman_brown` = 2r/(1+r) on the split-half correlation r. `between_team_sd_pp` = SD across team-seasons of the full-season share; `binomial_sd_at_median_n_pp` = sqrt(p(1-p)/n) at the median attempt count; `sd_ratio` is their ratio; `signal_variance_share` = 1 - mean binomial variance / between-team variance. Team-seasons under {MIN200} located attempts in the class are in a separate scope labelled UNDERPOWERED. Vendors: CBBD 2022-2025 (training source); GameZone 2022-2024 (matched games only, so fewer attempts per team-season); CBBD restricted to the same GameZone-matched games for a like-for-like attempt count.

Team-seasons with at least {MIN200} attempts:

{T(r1)}
Team-seasons under {MIN200} attempts (UNDERPOWERED):

{T(r2)}
Plain words, CBBD: {VERDICTS['rel_cbbd']}

Plain words, GameZone: {VERDICTS['rel_gz']}
"""
    (DOCS / "gamezone_frame_calibration_2026-10-27.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
