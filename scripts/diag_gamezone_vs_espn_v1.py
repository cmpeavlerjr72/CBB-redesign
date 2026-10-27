#!/usr/bin/env python
"""diag_gamezone_vs_espn_v1.py -- vendor-vs-ESPN agreement diagnostic (GameZone vs CBBD located shots).

DESCRIPTIVE ONLY: no fit, no recommendation. PRIVATE INTERNAL RESEARCH DATA (user ruling 2026-10-27).

Needs scripts/diag_gamezone_inventory_v1.py to have run first (results/gamezone/*.parquet, basket.json) and
results/coordinate_audit/master_<season>.parquet (scripts/diag_coordinate_audit_v1.py --part master).

(a) play-level join on matched games 2022-2024 (period, clock +-2 s, shooter last name + first initial, outcome):
    distance agreement by CBBD text class and outcome; per-vendor text consistency by outcome.
(b) GameZone alone, all seasons: rim-text charted > 6 ft from the rim, makes vs misses (paired, game bootstrap),
    canned rim pixels by outcome, spike pixels.
(c) band shares per team-game, GameZone vs CBBD, co-located games.
(d) three-point placement accuracy under GameZone by season (text from the vendor `desc`).

Season 2026 is not touched (GameZone ends 2023-24). Underpowered = fewer than 500 attempts in a cell.

Run from the repo root:  .venv/Scripts/python.exe -I scripts/diag_gamezone_vs_espn_v1.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.pbp import events as E  # noqa: E402

RES = ROOT / "results/gamezone"
DOCS = ROOT / "docs/tests"
CBBD = ROOT / "data/raw/cbbd"
MIN_N = 500
DELTAS = [0] + [x for d in range(1, 11) for x in (d, -d)]      # nearest-first clock offsets, +-10 s
ARC_FT = 22.15
SEASONS = (2022, 2023, 2024)
RIM_TEXT = ("Layup Shot", "Dunk Shot", "Tip Shot")


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


A = _load("coord_audit", "diag_coordinate_audit_v1.py")
I = _load("gz_inventory", "diag_gamezone_inventory_v1.py")
log = A.log
T = A.md_table
pct = I.pct
RNG = np.random.default_rng(20261027)


def flag(n):
    return "UNDERPOWERED" if n < MIN_N else ""


# ------------------------------------------------------------------ loading
SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def name_key(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    t = [x for x in re.sub(r"[^a-z ]+", " ", s).split() if x not in SUFFIX]
    if not t:
        return ""
    return t[-1] + "|" + t[0][0]


def load_gz():
    sh = pd.read_parquet(RES / "gz_shots.parquet")
    bsk = json.load(open(RES / "basket.json"))
    sh = sh[sh["fga"]].copy()
    sh["dseason"] = sh["dseason"].astype(int)
    bx, by = 25.0, 5.25                       # PRIMARY basket = textbook rim centre (= canned pixel A)
    cx = sh["dseason"].map(lambda y: bsk["dunk_centroid"][str(y)][0]).to_numpy()
    cy = sh["dseason"].map(lambda y: bsk["dunk_centroid"][str(y)][1]).to_numpy()
    sh["dist"] = np.hypot(sh["loc_y"] - by, sh["loc_x"] - bx)
    sh["dist_alt"] = np.hypot(sh["loc_y"] - cy, sh["loc_x"] - cx)      # sensitivity: per-season dunk-centroid basket
    sh["theta"] = np.degrees(np.arctan2(sh["loc_x"] - bx, sh["loc_y"] - by))
    sh["theta_alt"] = np.degrees(np.arctan2(sh["loc_x"] - cx, sh["loc_y"] - cy))
    sh["sideline"] = (sh["loc_x"] <= 4.0) | (sh["loc_x"] >= 46.0)
    d = sh["desc"].fillna("").str.lower()
    sh["text_cls"] = np.where(d.str.contains("3-point|three point"), "three",
                              np.where(d.str.contains("layup|dunk|tip"), "rimtext",
                                       np.where(d.str.contains("jump shot"), "twojump", "other")))
    sh["nk"] = sh["shooter"].map(name_key)
    sh["secs"] = sh["half_secs_remaining"]
    sh["canned"] = (((sh["loc_x"] == 25.0) & (sh["loc_y"] == 5.25)) | ((sh["loc_x"] == 26.0) & (sh["loc_y"] == 6.25)))
    return sh


def load_cbbd():
    ms = []
    for s in SEASONS:
        m = pd.read_parquet(ROOT / f"results/coordinate_audit/master_{s}.parquet")
        n = E.load_plays(s, columns=("gameId", "id", "shot_shooter_name"))[["id", "shot_shooter_name"]]
        m = m.merge(n, on="id", how="left")
        g = pd.read_parquet(CBBD / f"games_{s}.parquet", columns=["id", "homeTeamId", "awayTeamId"]).rename(columns={"id": "gameId"})
        m = m.merge(g, on="gameId", how="left")
        ms.append(m)
    m = pd.concat(ms, ignore_index=True)
    m["nk"] = m["shot_shooter_name"].map(name_key)
    m["side"] = np.where(m["isHomeTeam"], "home", "away")
    m["team"] = np.where(m["isHomeTeam"], m["homeTeamId"], m["awayTeamId"])
    return m


# ------------------------------------------------------------------ join
def join_plays(c: pd.DataFrame, g: pd.DataFrame):
    """One-to-one join of CBBD attempts to GameZone attempts inside matched games."""
    c = c.copy()
    g = g.copy()
    c["gk"] = c["gameId"]
    g["gk"] = g["gameId"]
    c["cid"] = np.arange(len(c))
    g["gid"] = np.arange(len(g))
    c["pair_gid"] = -1
    pairs = []
    cu = c[["cid", "gk", "period", "secondsRemaining", "nk", "made"]].copy()
    gu = g[["gid", "gk", "half", "secs", "nk", "made"]].rename(columns={"half": "period"}).copy()
    gu["period"] = gu["period"].astype(int)
    cu = cu[cu["nk"] != ""]
    gu = gu[gu["nk"] != ""]
    for delta in DELTAS:
        cu["t"] = cu["secondsRemaining"] + delta
        keyc = ["gk", "period", "t", "nk", "made"]
        gu["t"] = gu["secs"]
        cu["r"] = cu.groupby(keyc).cumcount()
        gu["r"] = gu.groupby(["gk", "period", "t", "nk", "made"]).cumcount()
        mm = cu.merge(gu, on=["gk", "period", "t", "nk", "made", "r"], suffixes=("", "_g"))
        pairs.append(mm[["cid", "gid"]].assign(delta=delta))
        cu = cu[~cu["cid"].isin(mm["cid"])]
        gu = gu[~gu["gid"].isin(mm["gid"])]
    pr = pd.concat(pairs, ignore_index=True)
    # leftover with outcome ignored (diagnostic of outcome disagreement)
    left_c, left_g = cu.copy(), gu.copy()
    left_c["t"] = left_c["secondsRemaining"]
    left_c["r"] = left_c.groupby(["gk", "period", "t", "nk"]).cumcount()
    left_g["r"] = left_g.groupby(["gk", "period", "t", "nk"]).cumcount()
    lm = left_c.merge(left_g, on=["gk", "period", "t", "nk", "r"], suffixes=("", "_g"))
    info = {"pairs_by_clock_delta": {int(k): int(v) for k, v in pr["delta"].value_counts().items()},
            "leftover_same_clock_name_different_outcome": int(len(lm))}
    return pr, info


# ------------------------------------------------------------------ main
def main() -> int:
    out = {"script": "scripts/diag_gamezone_vs_espn_v1.py"}
    gz = load_gz()
    log(f"gz fga {len(gz):,}")
    cb = load_cbbd()
    log(f"cbbd fga {len(cb):,}")

    # ============================== (a)
    matched = gz[gz["gameId"].notna() & gz["dseason"].isin(SEASONS)].copy()
    matched["gameId"] = matched["gameId"].astype(int)
    mg = set(matched["gameId"])
    c = cb[cb["gameId"].isin(mg)].reset_index(drop=True)
    g = matched.reset_index(drop=True)
    pr, jinfo = join_plays(c, g)
    log(f"joined {len(pr):,} of cbbd {len(c):,} / gz {len(g):,}")
    c = c.assign(cid=np.arange(len(c)))
    g = g.assign(gid=np.arange(len(g)))
    J = pr.merge(c, on="cid").merge(g, on="gid", suffixes=("_c", "_g"))
    # columns: c: x,y,xf,dist,located,cls,text_cls,made; g: loc_x, loc_y, dist, located, text_cls (suffix)
    J["c_located"] = J["located_c"]
    J["g_located"] = J["located_g"]
    J["season"] = J["season"]
    rate = []
    for s in SEASONS:
        cs = c[c["season"] == s]
        gs = g[g["dseason"] == s]
        js = J[J["season"] == s]
        rate.append({"season": s, "matched_games": cs["gameId"].nunique(), "cbbd_fga": len(cs), "cbbd_located": int(cs["located"].sum()),
                     "gz_fga": len(gs), "joined": len(js), "join_rate_of_cbbd_pct": pct(len(js), len(cs)),
                     "join_rate_of_gz_pct": pct(len(js), len(gs)),
                     "join_rate_of_cbbd_located_pct": pct(js["c_located"].sum(), cs["located"].sum()),
                     "joined_both_located": int((js["c_located"] & js["g_located"]).sum())})
    out["a_join_rate"] = rate
    out["a_join_info"] = jinfo
    both = J[J["c_located"] & J["g_located"]].copy()
    both["d_c"] = both["dist_c"]
    both["d_g"] = both["dist_g"]
    both["ad"] = (both["d_c"] - both["d_g"]).abs()
    both["ad_tb"] = (both["d_c"] - both["dist_alt"]).abs()
    both["outcome"] = np.where(both["made_c"] == 1, "made", "missed")
    both["cls_c"] = both["cls"]
    out["a_n_both_located"] = int(len(both))

    def agr(d, keys):
        rows = []
        for k, grp in d.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            rows.append({**dict(zip(keys, k)), "n": len(grp), "within_1ft_pct": pct((grp["ad"] <= 1).sum(), len(grp)),
                         "within_3ft_pct": pct((grp["ad"] <= 3).sum(), len(grp)), "gt_6ft_pct": pct((grp["ad"] > 6).sum(), len(grp)),
                         "median_abs_diff_ft": float(grp["ad"].median()), "flag": flag(len(grp))})
        return pd.DataFrame(rows)
    both["all"] = "all"
    t_all = agr(both, ["all"])
    t_cls = agr(both, ["cls_c"])
    t_out = agr(both, ["outcome"])
    t_clsout = agr(both, ["cls_c", "outcome"])
    t_season_cls = agr(both, ["season", "cls_c"])
    out["a_distance_agreement"] = {"all": A.recs(t_all), "by_class": A.recs(t_cls), "by_outcome": A.recs(t_out),
                                   "by_class_x_outcome": A.recs(t_clsout), "by_season_x_class": A.recs(t_season_cls)}
    # per-vendor text consistency by outcome on the SAME joined, both-located attempts (text class from CBBD play text)
    tc = []
    for oc in ("all", "made", "missed"):
        d = both if oc == "all" else both[both["outcome"] == oc]
        for vend, dcol in (("cbbd", "d_c"), ("gamezone", "d_g")):
            th = d[d["text_cls_c"] == "three"]
            rt = d[d["text_cls_c"] == "rimtext"]
            tj = d[d["text_cls_c"] == "twojump"]
            tc.append({"outcome": oc, "vendor": vend, "n_three": len(th), "n_rimtext": len(rt), "n_twojump": len(tj),
                       "three_inside_arc_pct": pct((th[dcol] < ARC_FT).sum(), len(th)),
                       "twojump_outside_arc_pct": pct((tj[dcol] >= ARC_FT).sum(), len(tj)),
                       "rimtext_gt6ft_pct": pct((rt[dcol] > 6).sum(), len(rt)),
                       "flag": flag(min(len(th), len(rt), len(tj)))})
    out["a_text_consistency_same_attempts"] = tc
    # paired outcome gap in the joined rim-text >6 ft share, per vendor
    rim = both[both["text_cls_c"] == "rimtext"].copy()
    rim_gap = []
    for vend, dcol in (("cbbd", "d_c"), ("gamezone", "d_g")):
        r = A.paired_diff(rim.rename(columns={"gameId_c": "gameId"}) if "gameId_c" in rim else rim,
                          (rim["outcome"] == "missed").to_numpy(), (rim["outcome"] == "made").to_numpy(),
                          (rim[dcol] > 6).astype(float).to_numpy(), RNG)
        rim_gap.append({"vendor": vend, "miss_gt6_rate": r["rate_a"], "make_gt6_rate": r["rate_b"], "pooled_diff": r["pooled_diff"],
                        "within_game_diff": r["within_diff"], "boot_se": r["boot_se"], "z": r["z"], "n_miss": r["n_a"], "n_make": r["n_b"],
                        "n_games_paired": r["n_games_paired"], "flag": flag(min(r["n_a"], r["n_b"]))})
    out["a_rim_gap_same_attempts"] = rim_gap
    # robustness: exact-ish clock joins only; and both vendors agree the attempt is a rim shot
    rob = []
    rim2 = both[(both["text_cls_c"] == "rimtext")].copy()
    rim2["both_rim"] = rim2["gz_cls"] == "rim"
    for lab, sub in (("all joined rim-text", rim2), ("clock offset <= 1 s", rim2[rim2["delta"].abs() <= 1]),
                     ("GameZone also calls it a rim shot", rim2[rim2["both_rim"]])):
        sub = sub.rename(columns={"gameId_c": "gameId"})
        for vend, dcol in (("cbbd", "d_c"), ("gamezone", "d_g")):
            r = A.paired_diff(sub, (sub["outcome"] == "missed").to_numpy(), (sub["outcome"] == "made").to_numpy(),
                              (sub[dcol] > 6).astype(float).to_numpy(), RNG)
            rob.append({"subset": lab, "vendor": vend, "miss_gt6_pct": 100 * r["rate_a"], "make_gt6_pct": 100 * r["rate_b"],
                        "within_game_diff_pp": 100 * r["within_diff"], "boot_se_pp": 100 * r["boot_se"], "n_miss": r["n_a"], "n_make": r["n_b"],
                        "flag": flag(min(r["n_a"], r["n_b"]))})
    out["a_rim_gap_robustness"] = rob
    # class disagreement between vendors on the joined attempts
    both["gz_three_flag"] = both["gz_cls"] == "three"
    both["c_three"] = both["cls_c"] == "three"
    out["a_class_agreement"] = {"n_joined_both_located": len(both),
                                "three_flag_disagree_pct": pct((both["gz_three_flag"] != both["c_three"]).sum(), len(both)),
                                "rim_class_agree_pct": pct(((both["gz_cls"] == "rim") == (both["cls_c"] == "rim")).sum(), len(both))}
    # displacement geometry (bias in distance)
    out["a_mean_signed_diff_cbbd_minus_gz_ft"] = A.recs(both.assign(sd=both["d_c"] - both["d_g"]).groupby(["cls_c", "outcome"]).agg(
        n=("sd", "size"), mean_signed=("sd", "mean"), sd=("sd", "std")).reset_index())

    # ============================== (b)
    gzl = gz[gz["located"]].copy()
    gzl["gameId"] = gzl["game_id"]
    rim = gzl[gzl["shot_desc"].isin(RIM_TEXT)].copy()
    rows, pix = [], []
    for s in sorted(rim["dseason"].unique()):
        r = rim[rim["dseason"] == s].reset_index(drop=True)
        mk, ms_ = (r["made"] == 1).to_numpy(), (r["made"] == 0).to_numpy()
        for lab, ind in (("gt6ft_from_rim", (r["dist"] > 6).astype(float)), ("on_canned_pixel", r["canned"].astype(float)),
                         ("gt6ft_dunk_centroid_basket", (r["dist_alt"] > 6).astype(float))):
            p = A.paired_diff(r, ms_, mk, ind.to_numpy(), RNG)
            rows.append({"date_season": int(s), "measure": lab, "n_miss": p["n_a"], "n_make": p["n_b"], "miss_rate_pct": 100 * p["rate_a"],
                         "make_rate_pct": 100 * p["rate_b"], "pooled_diff_pp": 100 * p["pooled_diff"], "within_game_diff_pp": 100 * p["within_diff"],
                         "boot_se_pp": 100 * p["boot_se"], "z": p["z"], "games_paired": p["n_games_paired"], "flag": flag(min(p["n_a"], p["n_b"]))})
    out["b_rim_asymmetry"] = rows
    # spike pixels
    allx = gzl.assign(px=(gzl["loc_x"] * 20).round() / 20, py=(gzl["loc_y"] * 20).round() / 20)
    for s in sorted(allx["dseason"].unique()):
        d = allx[allx["dseason"] == s]
        vc = d.groupby(["px", "py"]).agg(n=("made", "size"), mk=("made", "mean"), rimsh=("shot_desc", lambda x: x.isin(RIM_TEXT).mean())).reset_index()
        top = vc.sort_values("n", ascending=False).head(20)
        px_arr, py_arr, mk_arr = d["px"].to_numpy(), d["py"].to_numpy(), d["made"].to_numpy()
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            box = (np.abs(px_arr - r["px"]) <= 1.0 + 1e-9) & (np.abs(py_arr - r["py"]) <= 1.0 + 1e-9) & ~((px_arr == r["px"]) & (py_arr == r["py"]))
            nb = int(box.sum())
            isrim = d["shot_desc"].isin(RIM_TEXT).to_numpy()
            pix.append({"date_season": int(s), "rank": rank, "x": r["px"], "y": r["py"], "n": int(r["n"]), "share_of_located_pct": 100 * r["n"] / len(d),
                        "make_rate": r["mk"], "rim_text_share": r["rimsh"], "n_neighbours_3x3_1ft": nb,
                        "neighbour_make_rate": float(mk_arr[box].mean()) if nb else float("nan"),
                        "neighbour_rim_text_share": float(isrim[box].mean()) if nb else float("nan"),
                        "make_rate_minus_neighbours": r["mk"] - float(mk_arr[box].mean()) if nb else float("nan"),
                        "flag": flag(min(int(r["n"]), nb))})
    pixdf = pd.DataFrame(pix)
    out["b_top20_pixels"] = A.recs(pixdf)
    spike_sum = (pixdf[pixdf["rank"] <= 3].pivot_table(index="date_season", columns="rank", values=["share_of_located_pct", "make_rate"]).round(3))
    spike_sum.columns = [f"{a}_rank{b}" for a, b in spike_sum.columns]
    spike_sum = spike_sum.reset_index()
    out["b_top3_pixels_summary"] = A.recs(spike_sum)
    ab = []
    for sn in sorted(gzl["dseason"].unique()):
        d = gzl[gzl["dseason"] == sn]
        pa = (d["loc_x"] == 25.0) & (d["loc_y"] == 5.25)
        pb = (d["loc_x"] == 26.0) & (d["loc_y"] == 6.25)
        ab.append({"date_season": int(sn), "n_A": int(pa.sum()), "A_make_rate": float(d.loc[pa, "made"].mean()), "n_B": int(pb.sum()),
                   "B_make_rate": float(d.loc[pb, "made"].mean()), "rim_text_make_rate_not_on_A_or_B": float(d.loc[~pa & ~pb & d["shot_desc"].isin(RIM_TEXT), "made"].mean()),
                   "share_of_pair_that_is_A_among_makes_pct": pct((pa & (d["made"] == 1)).sum(), ((pa | pb) & (d["made"] == 1)).sum()),
                   "share_of_pair_that_is_A_among_misses_pct": pct((pa & (d["made"] == 0)).sum(), ((pa | pb) & (d["made"] == 0)).sum())})
    out["b_pixels_A_B_by_outcome"] = ab

    # ============================== (c)
    cbl = cb[cb["located"] & cb["gameId"].isin(mg)].copy()
    cbl["band"] = I.band(cbl["cls"], cbl["dist"], np.degrees(np.arctan2(cbl["y"] - 250.0, cbl["xf"] - 52.5)),
                         (cbl["y"] <= 40) | (cbl["y"] >= 460))
    cbl["cls3"] = cbl["cls"]
    gzm = gz[gz["located"] & gz["gameId"].notna() & gz["dseason"].isin(SEASONS)].copy()
    gzm["gameId"] = gzm["gameId"].astype(int)
    gzm["band"] = I.band(gzm["gz_cls"], gzm["dist"], gzm["theta"], gzm["sideline"])
    gzm["band_alt"] = I.band(gzm["gz_cls"], gzm["dist_alt"], gzm["theta_alt"], gzm["sideline"])
    gzm["side_o"] = np.where(gzm["flip"].astype(bool), np.where(gzm["side"] == "home", "away", "home"), gzm["side"])
    games = pd.read_parquet(CBBD / "games_2022.parquet", columns=["id", "homeTeamId", "awayTeamId"])
    games = pd.concat([pd.read_parquet(CBBD / f"games_{s}.parquet", columns=["id", "homeTeamId", "awayTeamId"]) for s in SEASONS]).rename(columns={"id": "gameId"})
    gzm = gzm.merge(games, on="gameId", how="left")
    gzm["team"] = np.where(gzm["side_o"] == "home", gzm["homeTeamId"], gzm["awayTeamId"])
    bands = {"jump2": ["short2", "long2"], "three": ["corner3", "wing3", "top3"]}

    def unit_shares(d, bandcol, teamcol, sidecol):
        t = d[d[bandcol] != ""].groupby(["gameId", teamcol, "band"]).size().unstack(fill_value=0)
        for b in ["short2", "long2", "corner3", "wing3", "top3"]:
            if b not in t:
                t[b] = 0
        t["n_j2"] = t["short2"] + t["long2"]
        t["n_3"] = t["corner3"] + t["wing3"] + t["top3"]
        return t.reset_index().rename(columns={teamcol: "team"})
    uc = unit_shares(cbl, "band", "team", "side")
    ug = unit_shares(gzm, "band", "team", "side_o")
    U = uc.merge(ug, on=["gameId", "team"], suffixes=("_c", "_g"))
    U["season"] = U["gameId"].map(pd.concat([pd.read_parquet(CBBD / f"games_{s}.parquet", columns=["id", "season"]) for s in SEASONS]).set_index("id")["season"])
    crow, team_rows = [], []
    for grp, bl, ncol in (("jump2", ["short2", "long2"], "n_j2"), ("three", ["corner3", "wing3", "top3"], "n_3")):
        for minn in (5,):
            ok = U[(U[ncol + "_c"] >= minn) & (U[ncol + "_g"] >= minn)]
            for b in bl:
                sc, sg = ok[b + "_c"] / ok[ncol + "_c"], ok[b + "_g"] / ok[ncol + "_g"]
                crow.append({"class": grp, "band": b, "min_attempts_each_vendor": minn, "team_games": len(ok), "mean_share_cbbd_pct": 100 * sc.mean(),
                             "mean_share_gz_pct": 100 * sg.mean(), "mean_abs_diff_pp": 100 * (sc - sg).abs().mean(),
                             "mean_signed_diff_cbbd_minus_gz_pp": 100 * (sc - sg).mean(), "corr_team_game": float(np.corrcoef(sc, sg)[0, 1]),
                             "flag": flag(int(ok[ncol + "_c"].sum()))})
        # team-season shares
        ts = U.groupby(["season", "team"])[[b + s for b in bl for s in ("_c", "_g")] + [ncol + "_c", ncol + "_g"]].sum().reset_index()
        for minn in (100,):
            ok = ts[(ts[ncol + "_c"] >= minn) & (ts[ncol + "_g"] >= minn)]
            for b in bl:
                sc, sg = ok[b + "_c"] / ok[ncol + "_c"], ok[b + "_g"] / ok[ncol + "_g"]
                team_rows.append({"class": grp, "band": b, "min_attempts_each_vendor": minn, "team_seasons": len(ok),
                                  "corr_team_season": float(np.corrcoef(sc, sg)[0, 1]) if len(ok) > 3 else float("nan"),
                                  "sd_share_cbbd_pp": 100 * sc.std(), "sd_share_gz_pp": 100 * sg.std(),
                                  "mean_abs_diff_pp": 100 * (sc - sg).abs().mean(),
                                  "flag": "UNDERPOWERED" if len(ok) < 30 else ""})
    out["c_team_game_band_shares"] = crow
    out["c_team_season_band_shares"] = team_rows
    # attempt-level band disagreement on joined attempts (both located, both text-compatible class)
    jb = both.copy()
    jb["xf_c"] = jb["xf"]
    jb["band_c"] = I.band(jb["cls_c"], jb["d_c"], np.degrees(np.arctan2(jb["y"] - 250.0, jb["xf"] - 52.5)), (jb["y"] <= 40) | (jb["y"] >= 460))
    jb["band_g"] = I.band(jb["gz_cls"], jb["d_g"], jb["theta"], jb["sideline"])
    jb = jb[(jb["band_c"] != "") | (jb["band_g"] != "")]
    same_cls = jb[(jb["cls_c"] == jb["gz_cls"]) & jb["cls_c"].isin(["jump2", "three"])]
    ar = []
    for cl in ("jump2", "three"):
        d = same_cls[same_cls["cls_c"] == cl]
        ar.append({"class": cl, "scope": "same class under both vendors", "n": len(d), "band_differs_pct": pct((d["band_c"] != d["band_g"]).sum(), len(d)),
                   "flag": flag(len(d))})
        for b in bands[cl]:
            db = d[(d["band_c"] == b) | (d["band_g"] == b)]
            ar.append({"class": cl, "scope": f"either vendor says {b}", "n": len(db), "band_differs_pct": pct((db["band_c"] != db["band_g"]).sum(), len(db)),
                       "flag": flag(len(db))})
    for oc in ("made", "missed"):
        for cl in ("jump2", "three"):
            d = same_cls[(same_cls["cls_c"] == cl) & (same_cls["outcome"] == oc)]
            ar.append({"class": cl, "scope": f"same class, {oc}", "n": len(d), "band_differs_pct": pct((d["band_c"] != d["band_g"]).sum(), len(d)), "flag": flag(len(d))})
    out["c_attempt_band_disagreement"] = ar
    ct = pd.crosstab(same_cls[same_cls["cls_c"] == "three"]["band_c"], same_cls[same_cls["cls_c"] == "three"]["band_g"])
    out["c_three_band_confusion_rows_cbbd_cols_gz"] = {r: {c_: int(v) for c_, v in row.items()} for r, row in ct.iterrows()}
    ct2 = pd.crosstab(same_cls[same_cls["cls_c"] == "jump2"]["band_c"], same_cls[same_cls["cls_c"] == "jump2"]["band_g"])
    out["c_jump2_band_confusion_rows_cbbd_cols_gz"] = {r: {c_: int(v) for c_, v in row.items()} for r, row in ct2.iterrows()}
    # band-share by vendor, pooled on co-located located attempts
    out["c_pooled_band_shares_pct"] = [{"vendor": v, **{b: pct((d["band"] == b).sum(), (d["band"].isin(bands["jump2"] if b in bands["jump2"] else bands["three"])).sum()) for b in
                                                      ["short2", "long2", "corner3", "wing3", "top3"]}, "n_jump2": int(d["band"].isin(bands["jump2"]).sum()),
                                       "n_three": int(d["band"].isin(bands["three"]).sum())} for v, d in (("cbbd", cbl), ("gamezone", gzm))] + [
        {"vendor": "gamezone (dunk-centroid basket sensitivity)", **{b: pct((gzm["band_alt"] == b).sum(), gzm["band_alt"].isin(bands["jump2"] if b in bands["jump2"] else bands["three"]).sum()) for b in
                                                             ["short2", "long2", "corner3", "wing3", "top3"]}, "n_jump2": int(gzm["band_alt"].isin(bands["jump2"]).sum()),
         "n_three": int(gzm["band_alt"].isin(bands["three"]).sum())}]

    # ============================== (d)
    gd = gz[gz["located"]].copy()
    gd["outcome"] = np.where(gd["made"] == 1, "made", "missed")
    drows = []
    for s in sorted(gd["dseason"].unique()):
        d = gd[gd["dseason"] == s]
        th = d[d["text_cls"] == "three"]
        tj = d[d["text_cls"] == "twojump"]
        arc = 20.75 if s <= 2019 else 22.15       # NCAA men's line: 20.75 ft through 2018-19, 22.15 ft (22' 1 3/4") from 2019-20
        row = {"date_season": int(s), "n_text_three": len(th), "n_text_twojump": len(tj), "rule_arc_ft": arc,
               "three_inside_rule_arc_textbook_pct": pct((th["dist"] < arc).sum(), len(th)),
               "twojump_outside_rule_arc_textbook_pct": pct((tj["dist"] >= arc).sum(), len(tj))}
        for lab, col in (("textbook", "dist"), ("dunkcentroid", "dist_alt")):
            row[f"three_inside_22.15_{lab}_pct"] = pct((th[col] < ARC_FT).sum(), len(th))
            row[f"three_inside_21.65_{lab}_pct"] = pct((th[col] < ARC_FT - 0.5).sum(), len(th))
            row[f"three_inside_20.65_{lab}_pct"] = pct((th[col] < ARC_FT - 1.5).sum(), len(th))
            row[f"three_inside_20.75_{lab}_pct"] = pct((th[col] < 20.75).sum(), len(th))
            row[f"twojump_outside_22.15_{lab}_pct"] = pct((tj[col] >= ARC_FT).sum(), len(tj))
            row[f"twojump_outside_22.65_{lab}_pct"] = pct((tj[col] >= ARC_FT + 0.5).sum(), len(tj))
            row[f"twojump_outside_23.65_{lab}_pct"] = pct((tj[col] >= ARC_FT + 1.5).sum(), len(tj))
        row["vendor_three_flag_vs_text_disagree_pct"] = pct(((d["three_pt"].astype(bool)) != (d["text_cls"] == "three")).sum(), len(d))
        row["flag"] = flag(min(len(th), len(tj)))
        drows.append(row)
    out["d_by_season"] = drows
    dm = []
    for oc in ("made", "missed"):
        for s in sorted(gd["dseason"].unique()):
            d = gd[(gd["dseason"] == s) & (gd["outcome"] == oc)]
            th, tj = d[d["text_cls"] == "three"], d[d["text_cls"] == "twojump"]
            arc = 20.75 if s <= 2019 else 22.15
            dm.append({"date_season": int(s), "outcome": oc, "rule_arc_ft": arc, "n_three": len(th), "three_inside_22.15_pct": pct((th["dist"] < ARC_FT).sum(), len(th)),
                       "three_inside_rule_arc_pct": pct((th["dist"] < arc).sum(), len(th)),
                       "n_twojump": len(tj), "twojump_outside_22.15_pct": pct((tj["dist"] >= ARC_FT).sum(), len(tj)),
                       "twojump_outside_rule_arc_pct": pct((tj["dist"] >= arc).sum(), len(tj)),
                       "flag": flag(min(len(th), len(tj)))})
    out["d_by_season_outcome"] = dm
    gd2 = gd[gd["dseason"].isin(SEASONS)]
    out["d_gz_vs_cbbd_coloc_note"] = "CBBD figures on the same joined attempts are in a_text_consistency_same_attempts."
    write_doc(out, locals())
    return 0


# Plain-words answers, written after reading the tables of the run dated 2026-10-27 (numbers quoted from them).
VERDICTS = {
 "a": ("The independent vendor does not reproduce ESPN rim charting by outcome at ESPN size, but it also cannot test it well, because it puts almost every rim attempt on one of two fixed pixels. "
       "Join: 91.6-94.3% of CBBD attempts per season join one-to-one to a GameZone attempt (475,554 joined attempts located by both vendors). On those, the two vendors' distances to the basket are within 1 ft for 30% of attempts, within 3 ft for 71%, and more than 6 ft apart for 7.4%; threes agree best (44% within 1 ft, 87% within 3 ft, 1.5% over 6 ft), two-point jumpers worst (21%, 56%, 17.0%), rim shots in between (22%, 65%, 6.7%). By outcome the disagreement is not symmetric: rim attempts are over 6 ft apart for 10.3% of misses against 4.1% of makes, jumpers 18.3% vs 14.9%, threes 1.7% vs 1.1%. "
       "Where the rim error sits: for layups, dunks and tips (CBBD play text), ESPN charts 14.5% of misses against 5.6% of makes more than 6 ft from the rim (gap 8.8 points within game, floor 0.17); GameZone on the same attempts charts 10.7% of misses against 5.1% of makes (gap 5.5 points). Restricted to attempts GameZone itself calls rim shots, ESPN keeps its gap (13.5% vs 5.1%, 8.3 points) while GameZone shows 4.1% vs 2.8% (1.7 points, floor 0.10). So the outcome-dependent far placement is an ESPN pattern, and GameZone own rim placement barely moves with outcome (mostly because it is canned). Three-point placement does not depend on outcome in either vendor (text threes charted inside 22.15 ft: ESPN 1.6% made vs 1.8% missed, GameZone 2.7% vs 2.9%)."),
 "b": ("On its own, GameZone shows almost no outcome-dependent rim charting, except in one file, and its rim placement is mostly not a measurement. In 15 of 16 date seasons 93-99% of layups, dunks and tips sit on one of two fixed pixels, (25.0, 5.25) and (26.0, 6.25), which together hold 29-36% of ALL located attempts (18% in the 2022-23 file); only 0.0-0.7% of rim-text attempts are charted more than 6 ft from the rim, and misses minus makes is 0.02-0.37 points within game (floors 0.03-0.07; many are statistically nonzero because n is huge, but the size is a few tenths of a point against ESPN 9 points). The exception is the 2022-23 file (date season 2023): only 45% of misses but 53% of makes are on the fixed pixels, the rest carry real coordinates, and there misses are charted over 6 ft from the rim 10.5% of the time against 7.1% for makes (4.4 points within game, floor 0.22; about half of ESPN gap on its own data). Which rim attempts get a real coordinate in that file depends on the outcome (makes are more often canned, -7.6 points, z -31). "
       "A different problem: in the three earliest date seasons (2008-09 to 2010-11 files) the pixel itself encodes the outcome: (25.0, 5.25) has a 97.8% make rate and (26.0, 6.25) a 0.2% make rate (99.9% of makes on the pair use the first pixel, 96-97% of misses use the second), then from 2011-12 the two pixels are used 50/50 with make rates 57-64% each. Spike pixels exist in every season (the top two pixels are the pair; the third pixel holds at most 0.07% of attempts); from 2011-12 the two pixels make rate is within 5 points of their neighbours (a third pixel holds at most 0.07% of attempts and is too small to read)."),
 "c": ("The two vendors do not agree on where jumpers and threes are taken at the band level, though they agree on which teams lean one way for some bands. Pooled over co-located games, GameZone has fewer short two-point jumpers (31.7% of two-point jumpers vs 41.3% in CBBD) and more wing threes (50.9% vs 44.3%); corner threes 18.1% vs 21.8%, top threes 31.0% vs 33.9% (using the dunk-centroid basket instead moves GameZone short2 to 34.4%, so part of the gap is frame calibration, which cannot be resolved here). Per team-game the mean absolute difference in band share is 19.5 points for short2/long2, 10.8 for corner3, 16.9 for wing3 and 14.6 for top3, with team-game correlations of 0.30, 0.35, 0.17 and 0.25. Per team-season (at least 100 attempts each) the band shares correlate 0.64 for short2, 0.51 for corner3, 0.39 for top3 and only 0.26 for wing3, with similar spread across teams in the two vendors (SD 9.7 vs 8.7 points for short2). On individual joined attempts of the same class, the band differs for 26.7% of two-point jumpers and 37.5% of threes (of the attempts either vendor puts in a given band, 56.9% corner, 52.8% wing, 55.4% top and 59.2% short2 differ), with no difference by outcome (made 36.6% vs missed 37.9% for threes). The main confusions are wing vs top and corner vs wing for threes, short vs long for jumpers. What the vendors share is a team-level tendency, not an attempt-level band."),
 "d": ("GameZone places text-threes on the right side of the line almost always, with a few percent inside it. Against the line in force (20.75 ft through 2018-19, 22.15 ft after) charted from the textbook rim, 0.01-0.10% of text-threes fall inside it in 2012-13 to 2018-19, 1.8-4.0% in 2008-09 to 2011-12 (a different vintage), and 2.2-4.4% in 2019-20 to 2023-24 (2.7%, 2.2%, 3.3% in 2022, 2023, 2024). The audit convention of 22.15 ft for every season gives 4.1-6.6% for 2012-13 to 2018-19, which reflects the old line, not a placement error. ESPN (CBBD) on the same joined 2022-2024 attempts is at 1.7% inside 22.15 ft, GameZone at 2.8%; neither depends on outcome (GameZone made vs missed 2.2-3.5% vs 2.3-3.2% by season). Text two-point jumpers charted outside the line: 0.00% in 2019-20 to 2023-24 and 0.36-1.07% in 2008-09 to 2018-19 against the 20.75 ft line. The vendor three flag and its own text agree on every row. With the dunk-centroid basket instead of the textbook rim the inside-line share roughly doubles in 2012-13 to 2023-24 (5.8% vs 2.7% in 2022), so basket choice matters as much as vendor."),
}


def write_doc(out, L):
    A.jdump(out, DOCS / "gamezone_vs_espn_agreement_2026-10-27.json")
    P = lambda k: pd.DataFrame(out[k])
    VER = lambda k: VERDICTS[k]
    da = out["a_distance_agreement"]
    md = f"""# GameZone vs ESPN (CBBD) agreement diagnostic (2026-10-27)

Descriptive only: no fit, no recommendation. Private internal research data (user ruling 2026-10-27). Script `scripts/diag_gamezone_vs_espn_v1.py` (needs `scripts/diag_gamezone_inventory_v1.py` first); tables also in `docs/tests/gamezone_vs_espn_agreement_2026-10-27.json`. "ESPN" below means the CBBD located shots (ESPN-sourced placement).

## Scope, definitions, floors

- Seasons: (a), (c): 2022, 2023, 2024 (the co-located seasons; GameZone ends 15 Jan 2024; 2025 and 2026 are not in GameZone and 2026 is untouched). (b), (d): all GameZone seasons by date season. Season = ending year.
- Games: GameZone games matched to CBBD games (inventory doc section 4). CBBD side: the audit master tables (FGA rows, de-duplicated, rim override 2.27 ft, classes rim / jump2 / three; text classes from `playText`, as the audit). GameZone side: FGA = shot rows with `free_throw == False`; `gz_cls` = three if `three_pt`, rim if `shot_desc` in {{Layup, Dunk, Tip}}, else jump2 (hook shots are jump2 here); text classes from the vendor `desc` (3-point; layup / dunk / tip; jump shot).
- Frames: CBBD folded to the left half, basket (52.5, 250) tenths of ft. GameZone: PRIMARY basket = textbook rim centre (25.0, 5.25), which is canned pixel A (inventory section 3 explains why the dunk-median calibration is degenerate); the per-date-season dunk centroid is shown as a sensitivity (`dunkcentroid` columns). Distances in feet. Arc 22.15 ft; slack variants as in the audit.
- Join (a): one-to-one on (game, period, clock within 10 s, shooter, outcome). Period: CBBD period = GameZone half (3+ = overtime). Clock: CBBD `secondsRemaining` vs GameZone `half_secs_remaining`, exact first, then nearest-first up to +-10 s (the two clocks differ by 3-10 s on many shots, symmetric, no per-game constant). Shooter: CBBD `shot_shooter_name` vs GameZone `shooter`, normalised (accents, punctuation, suffix removed) to last name + first initial. Outcome must match. Leftover pairs agreeing on everything except outcome are counted.
- Floors: paired comparisons are within game (weights n_a n_b / (n_a + n_b)) with a 300-resample game bootstrap, as the audit. Any cell under {MIN_N} attempts is UNDERPOWERED and is not read as signal or as absence of signal.

## (a) Play-level join on matched games, 2022-2024

Join rate (of CBBD attempts, of GameZone attempts, of CBBD located attempts), and the clock offsets that were needed ({out['a_join_info']['pairs_by_clock_delta']}; {out['a_join_info']['leftover_same_clock_name_different_outcome']} leftover pairs agree on game, period, clock and shooter but not on outcome):

{T(P('a_join_rate'))}
Distance-to-basket agreement on joined attempts located by BOTH vendors ({out['a_n_both_located']:,} attempts), |CBBD distance - GameZone distance|, by CBBD shot class and by outcome separately:

All: {T(pd.DataFrame(da['all']).drop(columns=['all']))}
By class:

{T(pd.DataFrame(da['by_class']))}
By outcome:

{T(pd.DataFrame(da['by_outcome']))}
Class x outcome:

{T(pd.DataFrame(da['by_class_x_outcome']))}
By season x class:

{T(pd.DataFrame(da['by_season_x_class']))}
Signed difference (CBBD minus GameZone distance, ft) by class and outcome: json `a_mean_signed_diff_cbbd_minus_gz_ft`. Class agreement on the joined attempts: {out['a_class_agreement']}.

Text consistency of each vendor on the SAME joined, both-located attempts (text class from the CBBD play text; threes charted inside 22.15 ft, text jumpers charted outside it, layups/dunks/tips charted more than 6 ft from the rim), by outcome:

{T(P('a_text_consistency_same_attempts'))}
Rim-text charted > 6 ft from the rim, missed minus made, within game with game-bootstrap floor, same attempts, per vendor:

{T(P('a_rim_gap_same_attempts'))}
Robustness (same measure; only joins with clock offset <= 1 s; only attempts GameZone also classes as a rim shot):

{T(P('a_rim_gap_robustness'))}
Verdict (a): {VER('a')}

## (b) Rim asymmetry re-measured on GameZone alone, all seasons

Layups, dunks and tips (`shot_desc`), located. For each date season: share charted > 6 ft from the rim, share sitting exactly on one of the two canned rim pixels (25.0, 5.25) / (26.0, 6.25), and the same with the per-season dunk-centroid basket, for misses vs makes, within game (the audit found 6-7% for makes vs 15-16% for misses on ESPN):

{T(P('b_rim_asymmetry'))}
Spike pixels: top 20 pixels per season (exact vendor coordinate), share of located attempts, make rate versus the attempts in the 3x3 block of 1-ft cells around it (excluding itself), in the json `b_top20_pixels`. Pixel A = (25.0, 5.25) and B = (26.0, 6.25): make rate at each, and which pixel makes and misses use:

{T(P('b_pixels_A_B_by_outcome'))}
Top three pixels per season:

{T(pd.DataFrame(out['b_top3_pixels_summary']))}
Verdict (b): {VER('b')}

## (c) Band shares per team-game, GameZone vs CBBD, co-located games 2022-2024

Bands from spec section 2 (short2 <= 10 ft; long2; corner3 = within 4 ft of a sideline and <= 24 ft; wing3 |theta| >= 30 deg; top3), each vendor in its own frame and own class. Pooled shares over all co-located located attempts:

{T(P('c_pooled_band_shares_pct'))}
Per team-game (both vendors with at least 5 located attempts of the class), share of the class in each band:

{T(P('c_team_game_band_shares'))}
Per team-season (both vendors with at least 100 attempts of the class), correlation of the band share between vendors:

{T(P('c_team_season_band_shares'))}
Share of individual joined attempts whose band differs (both located, same class under both vendors):

{T(P('c_attempt_band_disagreement'))}
Three-point band confusion on joined attempts (rows CBBD, columns GameZone): {out['c_three_band_confusion_rows_cbbd_cols_gz']}. Jump2 (rows CBBD, columns GameZone): {out['c_jump2_band_confusion_rows_cbbd_cols_gz']}.

Verdict (c): {VER('c')}

## (d) Three-point placement accuracy under GameZone

The men's three-point line was 20.75 ft through 2018-19 and 22.15 ft from 2019-20; the audit used 22.15 ft for every season by instruction, so both are shown (`rule_arc_ft` columns use the line in force; the 22.15 columns are the audit's convention and the only like-for-like comparison with the audit for 2022-2024). Text threes (vendor `desc` contains 3-point) charted inside the arc, and text two-point jumpers (`desc` contains "jump shot", no 3-point) charted outside it, by date season, with the textbook (primary) and dunk-centroid baskets and the audit's slack variants (CBBD on the same co-located attempts is in (a)):

{T(P('d_by_season'))}
By outcome (primary textbook basket):

{T(P('d_by_season_outcome'))}
Verdict (d): {VER('d')}
"""
    (DOCS / "gamezone_vs_espn_agreement_2026-10-27.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
