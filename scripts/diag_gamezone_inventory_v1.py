#!/usr/bin/env python
"""diag_gamezone_inventory_v1.py -- inventory of the frozen GameZone pbp copy (JackLich10/gamezoneR-data).

PRIVATE INTERNAL RESEARCH DATA (user ruling 2026-10-27): research input only, never synced to the public
HF dataset, never copied into rfq-collab. Descriptive; no fit, no recommendation.

Reads data/raw/gamezone/parquet/gamezone_pbp_<season>.parquet (scripts/pull_gamezone_v1.py).
Writes results/gamezone/{gz_shots,gz_games,gz_match}.parquet (gitignored; used by diag_gamezone_vs_espn_v1.py)
and docs/tests/gamezone_inventory_2026-10-27.{md,json}.

Season = ending year. "File season" = the file a row came from; "date season" = season implied by the
game date (month >= 8 -> year + 1). All match statistics use the date season after de-duplicating a
game_id that appears in several files (keep the copy in the file matching its date season).

Run from the repo root:  .venv/Scripts/python.exe -I scripts/diag_gamezone_inventory_v1.py
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GZ = ROOT / "data/raw/gamezone/parquet"
CBBD = ROOT / "data/raw/cbbd"
RES = ROOT / "results/gamezone"
DOCS = ROOT / "docs/tests"
FILE_SEASONS = list(range(2009, 2025))
MIN_N = 500
ARC_FT = 22.15
RIM_TEXT = ("Layup Shot", "Dunk Shot", "Tip Shot")

_spec = importlib.util.spec_from_file_location("coord_audit", ROOT / "scripts/diag_coordinate_audit_v1.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)           # reuse md_table, jdump, paired_diff, conf_tier, constants


def log(m):
    A.log(m)


# ------------------------------------------------------------------ load
def load_gz() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["season", "date", "game_id", "play_id", "neutral", "half", "half_secs_remaining", "home", "away", "event_team",
            "desc", "shot_outcome", "free_throw", "three_pt", "shot_desc", "loc_x", "loc_y", "shooter", "shooter_id"]
    sh, gm = [], []
    for y in FILE_SEASONS:
        d = pd.read_parquet(GZ / f"gamezone_pbp_{y}.parquet", columns=cols)
        d["file_season"] = y
        d["date"] = pd.to_datetime(d["date"])
        g = d.groupby("game_id").agg(date=("date", "first"), home=("home", "first"), away=("away", "first"),
                                     neutral=("neutral", "first"), rows=("date", "size")).reset_index()
        g["file_season"] = y
        gm.append(g)
        s = d[d["shot_outcome"].notna()].copy()
        sh.append(s.drop(columns=["season"]))
        log(f"gz {y}: rows {len(d):,}, games {len(g):,}, shot rows {len(s):,}")
    gm = pd.concat(gm, ignore_index=True)
    sh = pd.concat(sh, ignore_index=True)
    for t in (gm, sh):
        t["dseason"] = np.where(t["date"].dt.month >= 8, t["date"].dt.year + 1, t["date"].dt.year)
    return gm, sh


def dedupe(gm: pd.DataFrame, sh: pd.DataFrame):
    dup = gm[gm["game_id"].duplicated(keep=False)]
    dup_info = [{"game_id": int(g), "date": str(r["date"].iloc[0].date()), "date_season": int(r["dseason"].iloc[0]),
                 "in_file_seasons": [int(x) for x in sorted(r["file_season"])]} for g, r in dup.groupby("game_id")]
    gm = gm.assign(_pref=(gm["file_season"] == gm["dseason"]).astype(int)).sort_values(["game_id", "_pref"], ascending=[True, False])
    keep = gm.drop_duplicates("game_id")[["game_id", "file_season"]].rename(columns={"file_season": "keep_fs"})
    gm2 = gm.drop_duplicates("game_id").drop(columns="_pref").reset_index(drop=True)
    sh2 = sh.merge(keep, on="game_id", how="left")
    sh2 = sh2[sh2["file_season"] == sh2["keep_fs"]].drop(columns="keep_fs").reset_index(drop=True)
    return gm2, sh2, dup_info


# ------------------------------------------------------------------ frame
def classify_gz(sh: pd.DataFrame) -> pd.DataFrame:
    sh = sh.copy()
    sh["fga"] = ~sh["free_throw"].astype(bool)
    sh["located"] = sh["loc_x"].notna() & sh["loc_y"].notna()
    sh["made"] = (sh["shot_outcome"] == "made").astype(float)
    sh["gz_cls"] = np.where(sh["three_pt"].astype(bool), "three", np.where(sh["shot_desc"].isin(RIM_TEXT), "rim", "jump2"))
    sh.loc[sh["shot_desc"].isna() & ~sh["three_pt"].astype(bool), "gz_cls"] = "unk"
    return sh


def add_geometry(d: pd.DataFrame, bx: float, by: float) -> pd.DataFrame:
    d = d.copy()
    dx = d["loc_y"].to_numpy(float) - by          # forward (away from baseline)
    dy = d["loc_x"].to_numpy(float) - bx          # lateral
    d["dist"] = np.hypot(dx, dy)
    d["theta"] = np.degrees(np.arctan2(dy, dx))
    d["near_sideline"] = (d["loc_x"] <= 4.0) | (d["loc_x"] >= 46.0)
    return d


def band(cls, dist, theta, sideline):
    """Spec section 2 bands, GameZone frame (identical geometry; sideline = within 4 ft of a sideline)."""
    cls = np.asarray(cls)
    dist = np.asarray(dist, float)
    out = np.full(len(cls), "", dtype=object)
    out[(cls == "jump2") & (dist <= 10)] = "short2"
    out[(cls == "jump2") & (dist > 10)] = "long2"
    th = np.abs(np.asarray(theta, float))
    sideline = np.asarray(sideline, bool)
    t = cls == "three"
    corner = sideline & (dist <= 24)
    out[t & corner] = "corner3"
    out[t & ~corner & (th >= 30)] = "wing3"
    out[t & ~corner & (th < 30)] = "top3"
    return out


# ------------------------------------------------------------------ name matching
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("'", "")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    t = s.split()
    if not t:
        return s
    if t[0] == "st":
        t[0] = "saint"
    if len(t) > 1 and t[-1] == "st":
        t[-1] = "state"
    return " ".join(t)


def load_ours() -> pd.DataFrame:
    gs = []
    for s in range(2013, 2027):
        g = pd.read_parquet(CBBD / f"games_{s}.parquet", columns=[
            "id", "sourceId", "season", "seasonType", "startDate", "neutralSite", "homeTeam", "awayTeam",
            "homeConference", "awayConference", "status"])
        gs.append(g)
    g = pd.concat(gs, ignore_index=True)
    g = g[g["status"] == "final"].copy()
    ts = pd.to_datetime(g["startDate"], utc=True).dt.tz_convert("US/Eastern")
    g["date"] = ts.dt.tz_localize(None).dt.normalize()
    g["d1"] = g["homeConference"].notna() & g["awayConference"].notna()
    g["h"] = g["homeTeam"].map(norm)
    g["a"] = g["awayTeam"].map(norm)
    g["neutralSite"] = g["neutralSite"].fillna(False).astype(bool)
    return g.rename(columns={"id": "gameId"}).reset_index(drop=True)


def match_games(gm: pd.DataFrame, ours: pd.DataFrame):
    """Three tiers: (1) exact normalised names, either orientation, date +0/+-1; (2) one side matches by
    normalised/learned name and the date window holds exactly one candidate; (3) both names unknown: unique
    candidate on the date window with neutral flag equal and min side difflib ratio >= 0.70. Learned map
    gz-name -> our-name is built from tiers 1-2 (mode). flip = GameZone home is our away."""
    by_date = defaultdict(list)
    for i, dt in enumerate(ours["date"]):
        by_date[dt].append(i)
    oh_, oa_ = ours["h"].to_numpy(), ours["a"].to_numpy()
    used = {}
    res = {}
    nmap = defaultdict(Counter)

    def learn(gh, ga, i, flip):
        oh, oa = oh_[i], oa_[i]
        if flip:
            oh, oa = oa, oh
        nmap[gh][oh] += 1
        nmap[ga][oa] += 1

    g = gm.assign(gh=gm["home"].map(norm), ga=gm["away"].map(norm)).reset_index(drop=True)
    todo = list(g.index)
    for k in todo:                                                  # tier 1
        r = g.loc[k]
        done = False
        for o in (0, 1, -1):
            for i in by_date.get(r["date"] + pd.Timedelta(days=o), []):
                if i in used:
                    continue
                if (oh_[i], oa_[i]) == (r["gh"], r["ga"]):
                    flip = False
                elif (oa_[i], oh_[i]) == (r["gh"], r["ga"]):
                    flip = True
                else:
                    continue
                used[i] = k
                res[k] = (i, 1, flip, o)
                learn(r["gh"], r["ga"], i, flip)
                done = True
                break
            if done:
                break
    for _ in range(4):                                              # tier 2
        mp = {k: c.most_common(1)[0][0] for k, c in nmap.items()}
        new = 0
        for k in [t for t in todo if t not in res]:
            r = g.loc[k]
            kh, ka = mp.get(r["gh"], r["gh"]), mp.get(r["ga"], r["ga"])
            hits = []
            for o in (0, 1, -1):
                for i in by_date.get(r["date"] + pd.Timedelta(days=o), []):
                    if i in used:
                        continue
                    oh, oa = oh_[i], oa_[i]
                    if {oh, oa} == {kh, ka} and kh != ka:
                        hits.append((i, oh != kh, o))
                    elif kh in (oh, oa) or ka in (oh, oa):
                        hits.append((i, (oa == kh) or (oh == ka), o))
                if hits:
                    break
            if len(hits) == 1:
                i, flip, o = hits[0]
                used[i] = k
                res[k] = (i, 2, bool(flip), o)
                learn(r["gh"], r["ga"], i, bool(flip))
                new += 1
        if new == 0:
            break
    for k in [t for t in todo if t not in res]:                     # tier 3
        r = g.loc[k]
        best = []
        for o in (0, 1, -1):
            for i in by_date.get(r["date"] + pd.Timedelta(days=o), []):
                if i in used or bool(ours["neutralSite"].iat[i]) != bool(r["neutral"]):
                    continue
                s_f = min(difflib.SequenceMatcher(None, r["gh"], oh_[i]).ratio(), difflib.SequenceMatcher(None, r["ga"], oa_[i]).ratio())
                s_r = min(difflib.SequenceMatcher(None, r["gh"], oa_[i]).ratio(), difflib.SequenceMatcher(None, r["ga"], oh_[i]).ratio())
                best.append((max(s_f, s_r), i, s_r > s_f, o))
            if best:
                break
        best = [b for b in best if b[0] >= 0.70]
        if len(best) == 1:
            _, i, flip, o = best[0]
            used[i] = k
            res[k] = (i, 3, bool(flip), o)
    rows = [{"game_id": g.loc[k, "game_id"], "our_idx": i, "tier": t, "flip": f, "date_offset": o} for k, (i, t, f, o) in res.items()]
    m = pd.DataFrame(rows)
    m["gameId"] = ours["gameId"].to_numpy()[m["our_idx"].to_numpy()]
    mp = {k: c.most_common(1)[0][0] for k, c in nmap.items()}
    return m.drop(columns="our_idx"), mp


def pct(a, b):
    return float(100 * a / b) if b else float("nan")


# ------------------------------------------------------------------ main
def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    gm_raw, sh_raw = load_gz()
    gm, sh, dup_info = dedupe(gm_raw, sh_raw)
    sh = classify_gz(sh)
    sh_raw = classify_gz(sh_raw)
    out = {"script": "scripts/diag_gamezone_inventory_v1.py", "duplicate_games_across_files": dup_info}

    rows = []
    for y in FILE_SEASONS:
        g = gm_raw[gm_raw["file_season"] == y]
        s = sh_raw[sh_raw["file_season"] == y]
        f = s[s["fga"]]
        lo, hi = pd.Timestamp(y - 1, 10, 25), pd.Timestamp(y, 4, 15)
        stray = g[(g["date"] < lo) | (g["date"] > hi)]
        fg = f.groupby("game_id")["located"]
        rows.append({"file_season": y, "rows": int(g["rows"].sum()), "games": len(g), "date_min": str(g["date"].min().date()),
                     "date_max": str(g["date"].max().date()), "games_outside_window": len(stray),
                     "stray_dates": ",".join(sorted({str(d.date()) for d in stray["date"]}))[:60],
                     "games_dateseason_ne_file": int((g["dseason"] != y).sum()),
                     "fga_rows": len(f), "fga_located_pct": pct(f["located"].sum(), len(f)),
                     "ft_rows": int((~s["fga"]).sum()), "ft_located_pct": pct(s.loc[~s["fga"], "located"].sum(), (~s["fga"]).sum()),
                     "fga_games": f["game_id"].nunique(),
                     "games_ge90_located_pct": pct((fg.mean() >= 0.9).sum(), f["game_id"].nunique()),
                     "games_zero_located": int((fg.sum() == 0).sum())})
    inv = pd.DataFrame(rows)
    out["per_file"] = A.recs(inv)

    fga_raw = sh_raw[sh_raw["fga"]]
    vocab = fga_raw.assign(sd=fga_raw["shot_desc"].fillna("(null)")).groupby(["file_season", "sd"]).size().unstack(fill_value=0)
    vocab_pct = (100 * vocab.div(vocab.sum(axis=1), axis=0)).round(2).reset_index()
    vocab_pct["three_pt_pct"] = [pct(fga_raw[fga_raw["file_season"] == y]["three_pt"].sum(), vocab.loc[y].sum()) for y in vocab.index]
    out["shot_desc_vocab_pct_of_fga"] = A.recs(vocab_pct)
    out["shot_type_values_all_files"] = {k: int(v) for k, v in fga_raw["shot_desc"].fillna("(null)").value_counts().items()}
    out["three_flag_on_nonjump"] = A.recs(fga_raw[fga_raw["three_pt"].astype(bool) & (fga_raw["shot_desc"] != "Jump Shot")]
                                          .groupby(["file_season", "shot_desc"]).size().rename("n").reset_index())

    loc = sh[sh["fga"] & sh["located"]].copy()
    dunk = loc[loc["shot_desc"] == "Dunk Shot"]
    cen = {int(y): (round(float(g["loc_x"].mean()) * 20) / 20, round(float(g["loc_y"].mean()) * 20) / 20)
           for y, g in dunk.groupby("dseason")}
    bsk = {y: (25.0, 5.25) for y in cen}          # primary basket = textbook rim centre (see doc section 3)
    bx, by = round(float(dunk["loc_x"].mean()) * 20) / 20, round(float(dunk["loc_y"].mean()) * 20) / 20
    out["basket_calibration"] = {"pooled_dunk_centroid_x": float(dunk["loc_x"].mean()), "pooled_dunk_centroid_y": float(dunk["loc_y"].mean()),
                                 "dunk_centroid_by_date_season": {str(k): list(v) for k, v in cen.items()},
                                 "primary_basket_all_seasons": [25.0, 5.25], "n_dunks": len(dunk)}
    loc = pd.concat([add_geometry(g, *bsk[int(y)]) for y, g in loc.groupby("dseason")], ignore_index=True)
    qs = [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]
    fr, dq, pk = [], [], []
    for y in sorted(loc["dseason"].unique()):
        d = loc[loc["dseason"] == y]
        dk = d[d["shot_desc"] == "Dunk Shot"]
        ly = d[d["shot_desc"] == "Layup Shot"]
        th = d[d["gz_cls"] == "three"]
        fr.append({"date_season": int(y), "n_located_fga": len(d), "x_min": d["loc_x"].min(), "x_max": d["loc_x"].max(),
                   "y_min": d["loc_y"].min(), "y_max": d["loc_y"].max(), "y_gt_47_pct": pct((d["loc_y"] > 47).sum(), len(d)),
                   "dunk_mean_x": dk["loc_x"].mean(), "dunk_mean_y": dk["loc_y"].mean(),
                   "three_median_dist": th["dist"].median(), "three_p05_dist": th["dist"].quantile(.05)})
        for nm_, t in (("Dunk", dk), ("Layup", ly)):
            dq.append({"date_season": int(y), "type": nm_, "n": len(t), **{f"p{int(q * 100):02d}": float(t["dist"].quantile(q)) for q in qs}})
        pa = (d["loc_x"] == 25.0) & (d["loc_y"] == 5.25)
        pb = (d["loc_x"] == 26.0) & (d["loc_y"] == 6.25)
        rim = d["shot_desc"].isin(RIM_TEXT)
        both = pa | pb
        home = d["event_team"] == d["home"]
        pk.append({"date_season": int(y), "pixelA_pct_of_located": pct(pa.sum(), len(d)), "pixelB_pct_of_located": pct(pb.sum(), len(d)),
                   "canned_pair_pct_of_rim_text": pct((both & rim).sum(), rim.sum()),
                   "A_share_of_pair": pct(pa.sum(), both.sum()),
                   "A_share_half1": pct((pa & (d["half"] == 1)).sum(), (both & (d["half"] == 1)).sum()),
                   "A_share_half2": pct((pa & (d["half"] == 2)).sum(), (both & (d["half"] == 2)).sum()),
                   "A_share_home": pct((pa & home).sum(), (both & home).sum()),
                   "A_share_away": pct((pa & ~home).sum(), (both & ~home).sum())})
    out["frame_by_date_season"] = fr
    out["dunk_layup_distance_quantiles_ft"] = dq
    out["canned_rim_pixels"] = pk
    side = loc.assign(lat=loc["loc_x"] - bx, home_flag=loc["event_team"] == loc["home"], hh=loc["half"].clip(upper=3))
    side_t = side.groupby(["hh", "home_flag"]).agg(n=("lat", "size"), mean_loc_x=("loc_x", "mean"), mean_loc_y=("loc_y", "mean"),
                                                   mean_lateral_offset=("lat", "mean")).reset_index()
    out["side_check"] = A.recs(side_t)

    ours = load_ours()
    m, nmap = match_games(gm, ours)
    gm = gm.merge(m, on="game_id", how="left")
    gm["matched"] = gm["gameId"].notna()
    sh = sh.merge(gm[["game_id", "gameId", "flip", "tier"]], on="game_id", how="left")
    sh["side"] = np.where(sh["event_team"] == sh["home"], "home", np.where(sh["event_team"] == sh["away"], "away", ""))
    nm_pairs = {k: v for k, v in nmap.items() if k != v}
    out["name_map_gz_to_cbbd_normalised_nonidentical"] = dict(sorted(nm_pairs.items()))
    out["name_map_n_entries"] = {"identical": sum(1 for k, v in nmap.items() if k == v), "nonidentical": len(nm_pairs)}
    ms = []
    for y in sorted(gm["dseason"].unique()):
        g = gm[gm["dseason"] == y]
        o = ours[ours["season"] == y]
        o1 = o[o["d1"]]
        mt = g[g["matched"]]
        has = bool(len(o))
        ms.append({"date_season": int(y), "gz_games": len(g), "matched": len(mt),
                   "match_rate_pct": pct(len(mt), len(g)) if has else float("nan"),
                   "tier1": int((mt["tier"] == 1).sum()), "tier2": int((mt["tier"] == 2).sum()), "tier3_fuzzy": int((mt["tier"] == 3).sum()),
                   "date_offset_nonzero": int((mt["date_offset"] != 0).sum()),
                   "our_final_d1proxy_games": len(o1), "our_matched": int(o1["gameId"].isin(mt["gameId"]).sum()),
                   "matched_share_of_ours_pct": pct(o1["gameId"].isin(mt["gameId"]).sum(), len(o1)) if has else float("nan"),
                   "matched_to_nonD1proxy": int((~o["gameId"].isin(o1["gameId"]) & o["gameId"].isin(mt["gameId"])).sum())})
    out["match_by_date_season"] = ms
    unm = gm[~gm["matched"] & gm["dseason"].between(2013, 2024)]
    out["unmatched_gz_examples"] = A.recs(unm.sample(min(40, len(unm)), random_state=0)[["dseason", "date", "home", "away", "neutral"]].astype({"date": str}))
    out["unmatched_name_counts_top"] = dict(Counter(list(unm["home"]) + list(unm["away"])).most_common(25))
    gm.drop(columns=["rows"]).to_parquet(RES / "gz_games.parquet", index=False)
    keep = ["file_season", "dseason", "date", "game_id", "gameId", "flip", "tier", "play_id", "neutral", "half", "half_secs_remaining", "home", "away",
            "event_team", "side", "desc", "shot_outcome", "made", "free_throw", "three_pt", "shot_desc", "gz_cls", "loc_x", "loc_y", "located",
            "fga", "shooter", "shooter_id"]
    sh[keep].to_parquet(RES / "gz_shots.parquet", index=False)
    m.to_parquet(RES / "gz_match.parquet", index=False)
    json.dump({"primary": {str(k): list(v) for k, v in bsk.items()}, "dunk_centroid": {str(k): list(v) for k, v in cen.items()}}, open(RES / "basket.json", "w"))
    log("match done")

    uni = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet",
                          columns=["season", "cbbd_game_id", "game_id", "is_d1_game", "season_type", "neutral_site", "has_pbp"])
    matched_ids = set(m["gameId"])
    oi = ours.set_index("gameId")
    un_rows, cell_rows = [], []
    for s in (2022, 2023, 2024, 2025):
        mst = pd.read_parquet(ROOT / f"results/coordinate_audit/master_{s}.parquet", columns=["gameId", "located", "h_located"])
        gg = mst.groupby("gameId").agg(n=("located", "size"), c=("located", "mean"), h=("h_located", "mean")).reset_index()
        u = uni[(uni["season"] == s) & uni["is_d1_game"]].copy()
        u = u.merge(gg, left_on="cbbd_game_id", right_on="gameId", how="left")
        u["ours_located"] = (u[["c", "h"]].max(axis=1) > 0.5).fillna(False)
        u["gz"] = u["cbbd_game_id"].isin(matched_ids)
        hc = u["cbbd_game_id"].map(oi["homeConference"]).fillna("none")
        u["power"] = [A.conf_tier(c, s) == "power" for c in hc]
        dt = pd.to_datetime(u["cbbd_game_id"].map(oi["date"]))
        u["march"] = dt.dt.month == 3
        u["post"] = u["season_type"] == 3
        u["neutral"] = u["neutral_site"].fillna(False).astype(bool)
        un = u[~u["ours_located"]]
        un_rows.append({"season": s, "d1_games": len(u), "ours_unlocated": len(un), "ours_unlocated_pct": pct(len(un), len(u)),
                        "gz_covers_all_games_pct": pct(u["gz"].sum(), len(u)),
                        "gz_covers_unlocated": int(un["gz"].sum()), "gz_covers_unlocated_pct": pct(un["gz"].sum(), len(un))})
        for nm_, col in (("power", "power"), ("neutral", "neutral"), ("postseason", "post"), ("march", "march")):
            for flag in (True, False):
                a_ = u[u[col] == flag]
                b_ = a_[~a_["ours_located"]]
                cell_rows.append({"season": s, "cell": f"{nm_}={flag}", "games": len(a_), "gz_cover_all_pct": pct(a_["gz"].sum(), len(a_)),
                                  "unlocated": len(b_), "unlocated_pct": pct(len(b_), len(a_)),
                                  "gz_covers_unlocated": int(b_["gz"].sum()), "gz_covers_unlocated_pct": pct(b_["gz"].sum(), len(b_)),
                                  "flag": "UNDERPOWERED (<5 games)" if len(b_) < 5 else ""})
    out["unlocated_overlap_2022_2025"] = un_rows
    out["unlocated_overlap_cells"] = cell_rows
    out["unlocated_definition"] = ("games_universe_v2 is_d1_game; ours located = max(CBBD located share, hoopR located share) > 0.5 over FGA rows "
                                   "(results/coordinate_audit master tables); a universe game with no CBBD pbp rows counts as unlocated")

    gain = []
    for s in (2019, 2020, 2021):
        o = ours[(ours["season"] == s) & ours["d1"]]
        h = pd.read_parquet(ROOT / f"data/raw/hoopr_hist/pbp/play_by_play_{s}.parquet", columns=["game_id", "shooting_play", "type_text", "coordinate_x"])
        h = h[h["shooting_play"].fillna(False).astype(bool) & ~h["type_text"].fillna("").str.contains("Free Throw")]
        cx = pd.to_numeric(h["coordinate_x"], errors="coerce")
        h = h.assign(loc=cx.notna() & (cx.abs() <= 1e6))
        hg = h.groupby("game_id")["loc"].mean()
        hg.index = pd.to_numeric(hg.index, errors="coerce")
        o = o.assign(hoopr_loc=pd.to_numeric(o["sourceId"], errors="coerce").map(hg).fillna(0).gt(0.5))
        o = o.assign(gz=o["gameId"].isin(matched_ids))
        gain.append({"season": s, "our_final_d1proxy_games": len(o), "gz_matched": int(o["gz"].sum()), "gz_matched_pct": pct(o["gz"].sum(), len(o)),
                     "ours_hoopr_hist_located_games": int(o["hoopr_loc"].sum()), "ours_located_pct": pct(o["hoopr_loc"].sum(), len(o)),
                     "gain_games_gz_and_not_located": int((o["gz"] & ~o["hoopr_loc"]).sum()),
                     "gain_pct_of_ours": pct((o["gz"] & ~o["hoopr_loc"]).sum(), len(o)),
                     "gain_pct_of_unlocated": pct((o["gz"] & ~o["hoopr_loc"]).sum(), (~o["hoopr_loc"]).sum())})
    out["gain_2019_2021"] = gain
    out["gain_note"] = ("CBBD plays are on disk only from 2021-11 (season 2022), so for 2019-2021 the only on-disk coordinates are hoopR_hist pbp "
                        "(ESPN, roughly 11-14% located); 'unlocated' there means hoopR_hist located share <= 0.5. CBBD coordinates for those seasons "
                        "(sampled earlier at 28-71%) are not on disk, so the true gain over a full CBBD pull would be smaller.")
    write_doc(out, inv, vocab_pct, pd.DataFrame(fr), pd.DataFrame(dq), pd.DataFrame(pk), pd.DataFrame(ms),
              pd.DataFrame(un_rows), pd.DataFrame(cell_rows), pd.DataFrame(gain), side_t)
    return 0


def write_doc(out, inv, vocab, fr, dq, pk, ms, un, cells, gain, side_t):
    A.jdump(out, DOCS / "gamezone_inventory_2026-10-27.json")
    T = A.md_table
    bc = out["basket_calibration"]
    dups = "; ".join(f"game {d['game_id']} dated {d['date']} (date season {d['date_season']}) appears in files {d['in_file_seasons']}" for d in out["duplicate_games_across_files"])
    stray = "; ".join(f"file {r['file_season']}: {r['stray_dates']}" for _, r in inv.iterrows() if r["stray_dates"])
    md = f"""# GameZone (gamezoneR-data) inventory (2026-10-27)

Private internal research data (user ruling 2026-10-27): research input only, not synced to the public HF dataset, not in rfq-collab. Descriptive; nothing is fit or adopted. Script `scripts/diag_gamezone_inventory_v1.py`; pull `scripts/pull_gamezone_v1.py` (16 `.rds` files read with pyreadr 0.5.8, all converted, no failures). Tables are also in `docs/tests/gamezone_inventory_2026-10-27.json`. Season = ending year. "File season" = file of origin; "date season" = from the game date (Aug or later = next year). Attempt cells under {MIN_N} are marked UNDERPOWERED; game-level cells in section 5 hold about 130 attempts per game, so a cell is flagged only below 5 games.

## 1. Files as delivered

Rows are all pbp rows (lineups, subs, fouls, shots). FGA = rows with a shot outcome and `free_throw == False`. Free-throw rows never carry coordinates. "Window" = 25 Oct (y-1) to 15 Apr (y).

{T(inv.drop(columns=["stray_dates"]))}
Dates outside the window: {stray or "none"}.

Duplicate game_ids across files: {dups or "none"}. Each duplicate is a single game copied into other files; that one stray game is what makes the minimum date of several files look wrong (e.g. 2013-12-27 in the 2014-15 file). It is not mixed-season data: after de-duplicating on game_id (keep the copy in the file matching the date season) no date/season mismatch remains. The 2008-09 file holds only part of a season (see its date range); 2019-20 ends in March 2020; 2023-24 ends 15 Jan 2024 (feed stops).

## 2. Shot-type vocabulary (percent of FGA rows per file; `three_pt_pct` = vendor three flag)

Vocabulary: Jump Shot, Layup Shot, Dunk Shot, Hook Shot, Tip Shot (plus 3 `Slam Dunk Shot` rows in all files and a few null `shot_desc`). Threes are flagged by `three_pt`; a few non-jumpers carry that flag (json `three_flag_on_nonjump`).

{T(vocab)}
## 3. Frame

Feet, half-court frame, one basket for every attempt: `loc_x` 0 to 50 across the court, `loc_y` from the baseline outward. There is no half-time side switching: attempts of both teams in both halves are placed at the same basket near the baseline (see `y_min`, dunk centroid below, and the half/home split in the json `side_check`).

Basket. The possessions v2 calibration for CBBD checked that the median dunk lands 2-2.5 ft from a stated basket. That check is degenerate here because GameZone dunks (and most rim attempts) are not spread around the rim: they sit on two canned pixels, A = (25.0, 5.25) and B = (26.0, 6.25), so the dunk median is 0 or 1.41 ft from A by construction. The per-date-season located `Dunk Shot` centroid is {bc['dunk_centroid_by_date_season']} (pooled ({bc['pooled_dunk_centroid_x']:.3f}, {bc['pooled_dunk_centroid_y']:.3f}) over {bc['n_dunks']:,} dunks): it equals pixel A in 2008-09 to 2010-11 (A dominates) and sits about 0.7 ft to the upper right of it from 2011-12 (A and B used about equally). PRIMARY basket for every GameZone distance here and in the vendor comparison = the textbook rim centre (25.0, 5.25) = pixel A; the dunk centroid is carried as a sensitivity. The text-threes-inside-the-arc yardstick (vendor comparison, part d) is lower with the textbook basket from 2012-13 on (2.2-6.6% vs 5.0-12.0% for 2012-13 onward), which supports it. Distances are therefore not on the same footing as CBBD's calibrated frame, and any band built from them inherits the vendor's frame offset. Frame by date season:

{T(fr)}
Distance to the primary (textbook) basket for dunks and layups (ft; quantiles; the CBBD calibration had a dunk median of 2.27 ft):

{T(dq)}
### Canned rim pixels

Two pixels, A = (25.0, 5.25) and B = (26.0, 6.25), hold a large share of ALL located attempts and most rim-text attempts. Which of the two is used does not follow half, home/away or neutral (columns `A_share_*` near 50%), so the pair behaves like a rim placeholder rather than a frame switch.

{T(pk)}
## 4. Match to our game universe

Our universe: CBBD `games_<season>.parquet`, status final. D1 proxy = both conferences non-null (section 5 uses `games_universe_v2` `is_d1_game`, as the audit did). CBBD games exist on disk from season 2013, so earlier GameZone files cannot be matched (NaN). Matching is on date (+0 then +-1 day; US/Eastern date of the CBBD start time) and team names in either orientation, in three tiers: (1) exact normalised names; (2) one side matches by normalised or learned name and the date window holds exactly one candidate (this learns the other side's name); (3) both names unknown: unique candidate with the same neutral flag and both-side difflib ratio >= 0.70. The learned name map has {out['name_map_n_entries']['nonidentical']} non-identical entries (json key `name_map_gz_to_cbbd_normalised_nonidentical`) and {out['name_map_n_entries']['identical']} identical ones.

{T(ms)}
`match_rate_pct` = matched GameZone games / GameZone games. `matched_share_of_ours_pct` = our D1-proxy games with a GameZone counterpart. `matched_to_nonD1proxy` = matches to CBBD games with a null conference. Unmatched GameZone games (name list and 40 examples) are in the json.

## 5. Overlap with our unlocated games, 2022-2025

Our located game = more than 50% of FGA rows located in CBBD or hoopR (audit master tables); universe = `games_universe_v2` D1 games (a game with no CBBD pbp rows counts as unlocated).

{T(un)}
Breakouts (power = home conference in the audit's power list; neutral; postseason = season_type 3; March = game month 3):

{T(cells)}
## 6. 2019-2021: games that would gain coordinates if the training window were extended

{T(gain)}
{out['gain_note']}
"""
    ms_ = ms.set_index("date_season")
    un_ = un.set_index("season")
    pw = cells[(cells["cell"] == "power=True") & cells["season"].isin([2022, 2023])]
    ne = cells[(cells["cell"] == "neutral=True") & cells["season"].isin([2022, 2023, 2024])]
    md += f"""
## Verdict, inventory (plain words)

All 16 files read without R and all carry coordinates on 99.97-99.99% of field-goal attempts (free throws never), in every game (no game below 90% located). The one odd minimum date (2013-12-27) is a single game copied into several files, not mixed seasons. GameZone games match our CBBD universe by date and names at {ms_.loc[2013:2024, 'match_rate_pct'].min():.1f}-{ms_.loc[2013:2024, 'match_rate_pct'].max():.1f}% per season (2013-2024; {int(ms_.loc[2013:2024, 'tier3_fuzzy'].sum())} fuzzy matches in total), and it covers about 38-40% of our D1 games each full season ({ms_.loc[2022, 'matched_share_of_ours_pct']:.1f}% in 2022, {ms_.loc[2023, 'matched_share_of_ours_pct']:.1f}% in 2023, {ms_.loc[2024, 'matched_share_of_ours_pct']:.1f}% in 2024 because the feed stops 15 Jan 2024, 0% in 2025 and 2026). Coverage is skewed to the games we lack: it covers {pw['gz_covers_unlocated_pct'].min():.0f}% of our unlocated power-conference games and 100% of unlocated postseason games in 2022 and 2023, 34-43% of unlocated neutral-site games, but only {un_.loc[2024, 'gz_covers_unlocated_pct']:.0f}-{un_.loc[2023, 'gz_covers_unlocated_pct']:.0f}% of all our unlocated games ({int(un_.loc[2022, 'gz_covers_unlocated'])}, {int(un_.loc[2023, 'gz_covers_unlocated'])}, {int(un_.loc[2024, 'gz_covers_unlocated'])} games in 2022-2024), because most of our unlocated games are low-tier games GameZone does not chart (power-conference home games: GameZone covers about 100%; the rest about 8-17%). Nothing for 2024-25 or 2025-26. The placement caveat that matters for any use: rim attempts sit on two canned pixels (about 94-98% of rim-text attempts in most files, 50% in the 2022-23 file), and the CBBD-style dunk-median basket calibration is degenerate (the textbook rim centre is used instead). If the window were extended to 2019-2021, GameZone would add coordinates to {int(gain['gain_games_gz_and_not_located'].min()):,}-{int(gain['gain_games_gz_and_not_located'].max()):,} games per season ({gain['gain_pct_of_ours'].min():.0f}-{gain['gain_pct_of_ours'].max():.0f}% of our D1 games) over what hoopR_hist locates today, with the caveat above on CBBD coverage for those years.
"""
    (DOCS / "gamezone_inventory_2026-10-27.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
