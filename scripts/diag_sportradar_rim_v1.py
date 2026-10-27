#!/usr/bin/env python
"""diag_sportradar_rim_v1.py -- rim-placement asymmetry on an independent vendor (Sportradar).

DIAGNOSTIC, research-only data (user ruling 2026-10-27; never synced to HF / rfq-collab). No model is
fit, nothing is wired. Repeats the rim measurements of docs/tests/coordinate_audit_2026-10-27.md on the
Sportradar men's pbp (BigQuery public dataset, seasons 2013-2017 = 2013-14 .. 2017-18), and on the
games that Sportradar and the ESPN hoopR history files both locate.

  (a) share of rim attempts (layup / dunk / tip) charted more than 6 ft from the rim, makes vs misses,
      within game, paired bootstrap over games (300 resamples) as the floor;
  (b) share of rim attempts WITHOUT a coordinate, makes vs misses;
  (c) top 20 pixels per season with share of located attempts and make rate vs the 8 neighbours;
  (d) rim make rate and make/miss placement by distance bin;
  (e) ESPN-vs-Sportradar play-level agreement on the commonly located 2014-2018 games, and the same
      measurements on the ESPN side of those games.

Output: docs/tests/sportradar_rim_asymmetry_2026-10-27.{md,json}.
The shared loader and the ESPN match come from diag_sportradar_inventory_v1.py (run that first).

    .venv/Scripts/python.exe scripts/diag_sportradar_rim_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import diag_sportradar_inventory_v1 as INV  # noqa: E402
from diag_coordinate_audit_v1 import paired_diff  # noqa: E402  (within-game paired difference + game bootstrap)

SEASONS = INV.SEASONS
MIN_N = 500
RIM_FT = 6.0
RES, DOCS = INV.RES, INV.DOCS
rng = np.random.default_rng(20261027)
log = INV.log
md_table, jdump, recs = INV.md_table, INV.jdump, INV.recs
BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 1e9)]
BIN_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-6", "6+"]

VERDICT = (
    "**Does an independent vendor show the same outcome-dependent rim charting as ESPN? Yes for the placement asymmetry, and it cannot be tested for the other two symptoms.** "
    "Sportradar charts layups, dunks and tips more than 6 ft from the rim far more often when the shot missed: 13.6 / 5.2 / 8.2 / 6.6% of makes against 25.3 / 13.7 / 18.8 / 16.5% of misses "
    "(2014-15 to 2017-18), a within-game gap of -11.1 / -8.3 / -10.4 / -9.7 points against a bootstrap floor of 0.25-0.30 (z -34 to -40), the same sign and about the same size as the ESPN/CBBD "
    "-8.4 to -8.9 points of 2022-25. It survives moving the basket to each season's dunk centroid (-4.2 to -5.7 points, floor 0.15-0.21, z -27 to -29), holds for layups and, smaller, for dunks "
    "(-2.8 to -6.2 points) and tips (-2.7 to -3.7), and make rate falls smoothly with distance (79-82% inside 1 ft to 36-43% beyond 6 ft). The two vendors are genuinely independent "
    "(on 72-108k joined plays both locate per season their positions differ by a median of 2.0-2.4 ft for rim plays and only 36-41% of rim distances agree within 1 ft), and rim MISSES are the less "
    "consistent placements: 32-37% of miss distances agree within 1 ft against 39-45% for makes, and 17.8-25.4% of misses differ by more than 3 ft against 9.9-17.6% of makes. The sharpest test "
    "conditions on ESPN: among rim plays ESPN puts within 3 ft of the rim (ESPN puts 99.5-99.9% of layups within 6 ft), Sportradar places the same play more than 6 ft away for 3.8-10.7% of makes but "
    "8.8-17.1% of misses (gap -5.2 to -9.3 points, floor 0.4-0.65), so the asymmetry is not just misses being truly farther out. Two things are not reproduced. (b) Sportradar carries a coordinate on every "
    "attempt (0 of 92-108k rim attempts lack one, no (0,0) or sentinel rows), so the outcome-dependent omission of makes cannot occur and cannot be tested; ESPN's own 2014-18 files lose 1.2-2.3 points "
    "more makes than misses from the located set pooled, but that is between games (located-or-not is decided per game there, within-game gap 0). (c) There is no placeholder pixel: the largest 1-inch "
    "pixel holds 0.07-0.09% of rim attempts (50-94 attempts, UNDERPOWERED, neighbour ratio at most 5), and the largest 6-inch cell holds 1.0-1.3% with a neighbour ratio at most 1.4 and "
    "make-rate differences from its ring of -4.3 to +3.5 points against floors of 1.4-2.3. Caveats that matter: the ESPN 2014-18 vintage on the SAME games is nearly clean "
    "(rim >6 ft, 2014-15 to 2017-18: 0.06-0.24% of makes vs 0.25-0.52% of misses, gap -0.2 to -0.3 points, floor 0.04-0.06), so outcome-dependent charting is a property of a vendor and vintage, not of the sport; "
    "its integer-foot grid puts 6-7% of rim attempts on the basket pixel, which makes 3.5-5.9 points less than its ring in every season (floor about 1.2-1.4), the same direction as the 2022-25 spike pixel but on a grid too coarse "
    "to separate from geometry; Sportradar is noisier than ESPN in general (dunk median distance 2.4-3.0 ft from the textbook basket, with a 0.5-1 ft frame offset that moves between 2014-15 and 2015-16), "
    "and I cannot tell from these data whether Sportradar's charter sees the outcome or misses are simply harder to place. Season 2013-14 has no shot type, so the rim measurements are not defined there. "
    "Net: a second, independent vendor shows the same pattern, so rim x/y should still not be used as an outcome-blind input.")



def row_pd(tag: dict, r: dict) -> dict:
    out = dict(tag)
    out.update({"n_makes": r["n_a"], "n_misses": r["n_b"], "rate_makes_pct": 100 * r["rate_a"], "rate_misses_pct": 100 * r["rate_b"],
                "pooled_diff_pts": 100 * r["pooled_diff"], "games_paired": r["n_games_paired"],
                "within_game_diff_pts": 100 * r["within_diff"] if np.isfinite(r["within_diff"]) else np.nan,
                "boot_floor_pts": 100 * r["boot_se"] if np.isfinite(r["boot_se"]) else np.nan,
                "z": r["z"], "flag": "UNDERPOWERED" if r["underpowered"] else ""})
    return out


# ------------------------------------------------------------------------------------- (a)
def asym_gt6(d: pd.DataFrame, label: str, dist: str, extra: dict | None = None) -> list[dict]:
    """d: rim attempts with columns gameId, made (bool), <dist> (ft), located (bool)."""
    rows = []
    d = d[d["located"]]
    for grp, m in (("rim (layup+dunk+tip)", np.ones(len(d), bool)), ("layup", (d["rimtype"] == "layup").to_numpy()),
                   ("dunk", (d["rimtype"] == "dunk").to_numpy()), ("tip", (d["rimtype"] == "tip").to_numpy())):
        g = d[m]
        if len(g) == 0:
            continue
        ind = (g[dist].to_numpy() > RIM_FT).astype(float)
        r = paired_diff(g, g["made"].to_numpy(), ~g["made"].to_numpy(), ind, rng)
        tag = {"label": label, "group": grp}
        tag.update(extra or {})
        rows.append(row_pd(tag, r))
    return rows


# ------------------------------------------------------------------------------------- (b)
def missing_coord(d: pd.DataFrame, label: str, extra: dict | None = None) -> list[dict]:
    """Share of rim attempts WITHOUT a coordinate, makes vs misses (located indicator inverted)."""
    rows = []
    ind = (~d["located"].to_numpy()).astype(float)
    r = paired_diff(d, d["made"].to_numpy(), ~d["made"].to_numpy(), ind, rng)
    tag = {"label": label}
    tag.update(extra or {})
    rows.append(row_pd(tag, r))
    return rows


# ------------------------------------------------------------------------------------- (c)
def pixel_rows(d: pd.DataFrame, label: str, kx: np.ndarray, ky: np.ndarray, grid: str, extra: dict | None = None, topn: int = 20) -> list[dict]:
    """Top pixels by count among located attempts; neighbours = the 8 adjacent cells on the same grid."""
    L = d.assign(kx=kx, ky=ky)
    L = L[L["located"]].copy()
    OFF = 1000                                                   # cells can be negative (ESPN basket-relative frame)
    L["key"] = (L["kx"].astype(np.int64) + OFF) * 100000 + (L["ky"].astype(np.int64) + OFF)
    cnt = L.groupby("key").agg(n=("made", "size"), mk=("made", "sum"))
    n_all = len(L)
    rows = []
    top = cnt.sort_values("n", ascending=False).head(topn)
    for rank, (key, r) in enumerate(top.iterrows(), 1):
        kxv, kyv = key // 100000, key % 100000                  # still offset by OFF
        ring_keys = [(kxv + i) * 100000 + (kyv + j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0)]
        rc = cnt.reindex(ring_keys).fillna(0.0)
        ring_n, ring_mk = float(rc["n"].sum()), float(rc["mk"].sum())
        sub = L[L["key"].isin([key] + ring_keys)]
        pm = (sub["key"] == key).to_numpy()
        res = paired_diff(sub, pm, ~pm, sub["made"].to_numpy().astype(float), rng, B=200)
        pix = L[L["key"] == key]
        rows.append({**(extra or {}), "label": label, "grid": grid, "rank": rank, "kx": int(kxv - OFF), "ky": int(kyv - OFF),
                     "dist_from_basket_ft": float(pix["dist"].median()), "n": int(r["n"]), "share_of_located_pct": 100 * r["n"] / n_all,
                     "neighbour_ratio": float(r["n"] / (ring_n / 8)) if ring_n > 0 else np.inf,
                     "pct_rim": 100 * float((pix["cls"] == "rim").mean()) if "cls" in pix else np.nan,
                     "make_rate_pct": 100 * r["mk"] / r["n"], "ring_make_rate_pct": 100 * ring_mk / ring_n if ring_n else np.nan,
                     "ring_n": int(ring_n), "within_game_diff_pts": 100 * res["within_diff"] if np.isfinite(res["within_diff"]) else np.nan,
                     "boot_floor_pts": 100 * res["boot_se"] if np.isfinite(res["boot_se"]) else np.nan,
                     "flag": "UNDERPOWERED" if min(r["n"], ring_n) < MIN_N else ""})
    return rows


# ------------------------------------------------------------------------------------- (d)
def dist_bins(d: pd.DataFrame, label: str, dist: str, extra: dict | None = None) -> list[dict]:
    d = d[d["located"]]
    mk, ms = d["made"].to_numpy(), ~d["made"].to_numpy()
    rows = []
    for (lo, hi), nm in zip(BINS, BIN_LABELS):
        inb = ((d[dist] >= lo) & (d[dist] < hi)).to_numpy()
        n = int(inb.sum())
        rows.append({**(extra or {}), "label": label, "bin_ft": nm, "attempts": n, "pct_of_rim_attempts": 100 * n / len(d),
                     "make_rate_pct": 100 * float(mk[inb].mean()) if n else np.nan,
                     "pct_of_makes_in_bin": 100 * float(inb[mk].mean()), "pct_of_misses_in_bin": 100 * float(inb[ms].mean()),
                     "flag": "UNDERPOWERED" if n < MIN_N else ""})
    return rows


# ------------------------------------------------------------------------------------- (e) join
SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def lastname_key(s: pd.Series) -> pd.Series:
    def f(n):
        if not isinstance(n, str):
            return ""
        t = [w for w in n.lower().replace(".", " ").replace("'", "").replace("-", "").split() if w and w not in SUFFIX]
        return "".join(ch for ch in (t[-1] if t else "") if ch.isalpha())
    return s.map(f)


def join_play_level(sr: pd.DataFrame, ef: pd.DataFrame, m: pd.DataFrame, tol: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """sr: Sportradar FGA of one season; ef: ESPN FGA of the matched games; m: game match. Key =
    (ESPN game, period, outcome, shooter last name) plus the nearest clock within `tol` seconds (the two vendors
    stamp clocks a second or two apart). Pairs must be one-to-one: when a play has more than one candidate in
    the group the closest wins and an ESPN row claimed twice is dropped from both."""
    s = sr.merge(m[["sr_game_id", "espn_game_id"]], left_on="game_id", right_on="sr_game_id", how="inner")
    s["nk"] = lastname_key(s["player"])
    e = ef.copy()
    e["nk"] = lastname_key(e["athlete_name_1"])
    e = e.rename(columns={"game_id": "espn_game_id", "period_number": "period"})
    e["period"] = e["period"].astype(float)
    s["period"] = s["period"].astype(float)
    s["clock_sec"] = s["clock_sec"].astype(float)
    e["clock_sec"] = e["clock_sec"].astype(float)
    by = ["espn_game_id", "period", "made", "nk"]
    s["espn_game_id"] = s["espn_game_id"].astype(str)
    e["espn_game_id"] = e["espn_game_id"].astype(str)
    s = s[s["clock_sec"].notna()].sort_values("clock_sec").reset_index(drop=True)
    e = e[e["clock_sec"].notna()].sort_values("clock_sec").reset_index(drop=True)
    e["_eid"] = np.arange(len(e))
    e["clock_es"] = e["clock_sec"]
    e2 = e[by + ["clock_sec", "clock_es", "_eid", "located", "dist", "rimtype", "is3", "coordinate_y_raw", "coordinate_y", "coordinate_x"]].rename(
        columns={"located": "located_es", "dist": "dist_es", "rimtype": "rimtype_es", "is3": "is3_es"})
    j = pd.merge_asof(s, e2, on="clock_sec", by=by, direction="nearest", tolerance=tol)
    j = j[j["_eid"].notna()].copy()
    j["_dt"] = (j["clock_sec"] - j["clock_es"]).abs()
    n_claim = int(j["_eid"].duplicated(keep=False).sum())
    j = j.sort_values("_dt").drop_duplicates("_eid", keep="first")           # one-to-one: closest SR play keeps the ESPN row
    j = j.rename(columns={"located": "located_sr", "dist": "dist_sr", "rimtype": "rimtype_sr", "is3": "is3_sr"})
    info = {"sr_fga_in_matched_games": int(len(s)), "espn_fga_in_matched_games": int(len(e)), "joined": int(len(j)),
            "joined_pct_of_sr": 100 * len(j) / max(len(s), 1), "joined_exact_clock_pct": 100 * float((j["_dt"] == 0).mean()),
            "joined_within_1s_pct": 100 * float((j["_dt"] <= 1).mean()), "espn_rows_claimed_twice": n_claim,
            "clock_tolerance_s": tol}
    j["gameId"] = j["espn_game_id"]
    return j, info


def agreement(j: pd.DataFrame) -> tuple[list[dict], dict]:
    b = j[j["located_sr"] & j["located_es"]].copy()
    # positional comparison in basket-relative feet: along = distance out from the baseline basket line, lateral = |offset|
    sr_along = (b["xf"] - INV.BASKET_TEXTBOOK[0]) / 12.0
    sr_lat = (b["y"] - INV.BASKET_TEXTBOOK[1]).abs() / 12.0
    es_along = b["coordinate_y_raw"]
    es_lat = b["coordinate_y"].abs()
    b["pos_err"] = np.hypot(sr_along - es_along, sr_lat - es_lat)
    b["dist_err"] = (b["dist_sr"] - b["dist_es"]).abs()
    b["cls_a"] = np.where(b["typed"], b["cls"], np.where(b["rimtype_es"] != "", "rim", np.where(b["is3_es"], "three", "jump2")))
    rows = []
    for outcome, om in (("all", np.ones(len(b), bool)), ("makes", b["made"].to_numpy()), ("misses", ~b["made"].to_numpy())):
        for cl in ("rim", "jump2", "three", "all classes"):
            g = b[om & ((b["cls_a"] == cl) if cl != "all classes" else True)]
            n = len(g)
            if n == 0:
                continue
            rows.append({"outcome": outcome, "class": cl, "pairs": n, "dist_median_abs_diff_ft": float(g["dist_err"].median()),
                         "dist_within_1ft_pct": 100 * float((g["dist_err"] <= 1).mean()), "dist_gt3ft_pct": 100 * float((g["dist_err"] > 3).mean()),
                         "pos_within_1ft_pct": 100 * float((g["pos_err"] <= 1).mean()), "pos_gt3ft_pct": 100 * float((g["pos_err"] > 3).mean()),
                         "pos_median_err_ft": float(g["pos_err"].median()),
                         "dist_corr": float(np.corrcoef(g["dist_sr"], g["dist_es"])[0, 1]) if n > 20 else np.nan,
                         "mean_dist_sr_ft": float(g["dist_sr"].mean()), "mean_dist_es_ft": float(g["dist_es"].mean()),
                         "flag": "UNDERPOWERED" if n < MIN_N else ""})
    # Sharp test of outcome-dependent noise: among rim plays that ESPN (text-consistent, ~99.7% of layups within 6 ft)
    # places within 3 ft of the rim, how often does Sportradar place the same play more than 6 ft away, makes vs misses?
    g = b[(b["cls_a"] == "rim") & (b["dist_es"] <= 3)].copy()
    r = paired_diff(g, g["made"].to_numpy(), ~g["made"].to_numpy(), (g["dist_sr"].to_numpy() > RIM_FT).astype(float), rng)
    far = row_pd({"test": "SR > 6 ft | ESPN <= 3 ft (rim plays)"}, r)
    extra = {"sr_far_given_espn_near": far, "pairs_both_located": int(len(b)),
             "rimtype_agree_pct(sr layup/dunk/tip == espn)": 100 * float((b["rimtype_sr"] == b["rimtype_es"]).mean()),
             "is3_agree_pct": 100 * float((b["is3_sr"] == b["is3_es"]).mean()),
             "exact_same_position_pct(pos_err<0.05ft)": 100 * float((b["pos_err"] < 0.05).mean())}
    return rows, extra


# ------------------------------------------------------------------------------------- main
def sr_frame(s: int) -> pd.DataFrame:
    f = INV.load_sr_fga(s)
    f = f.rename(columns={"game_id": "gameId"})
    return f


def main() -> None:
    R: dict = {"created": pd.Timestamp.now("UTC").isoformat()}
    A, B, C, Dd, E_agree, E_info = [], [], [], [], [], []
    sens = []
    ESPN_A, ESPN_B, ESPN_C, ESPN_D = [], [], [], []
    join_info = []
    FAR = []
    SRm_A, SRm_B, SRm_D = [], [], []
    for s in SEASONS:
        f = sr_frame(s)
        if not f["typed"].iloc[0]:
            A.append({"label": "sportradar", "season": s, "group": "rim", "flag": "NOT DEFINED: season carries no shot_type and no layup/dunk text"})
            log(f"{s}: untyped season, rim measurements not defined")
        else:
            rim = f[f["cls"] == "rim"].copy()
            # basket sensitivity: dunk-centroid basket (the median folded dunk position)
            dk = f[f["rimtype"] == "dunk"]
            cx, cy = float(dk["xf"].median()), float(dk["y"].median())
            rim["dist_cent"] = np.hypot(rim["xf"] - cx, rim["y"] - cy) / 12.0
            A += asym_gt6(rim, "sportradar", "dist", {"season": s, "basket": "textbook (63, 300) in"})
            sens += asym_gt6(rim, "sportradar", "dist_cent", {"season": s, "basket": f"dunk centroid ({cx:.0f}, {cy:.0f}) in"})
            B += missing_coord(rim, "sportradar", {"season": s})
            xi, yi = np.round(rim["xf"].to_numpy()), np.round(rim["y"].to_numpy())
            # (c) native 1-inch pixel: rim attempts only, and all attempts
            C += pixel_rows(rim, "sportradar rim attempts", xi, yi, "1 in pixel", {"season": s})
            allf = f[f["cls"] != "two_untyped"]
            C += pixel_rows(allf, "sportradar all attempts", np.round(allf["xf"].to_numpy()), np.round(allf["y"].to_numpy()), "1 in pixel", {"season": s}, topn=10)
            C += pixel_rows(rim, "sportradar rim attempts", np.floor(rim["xf"].to_numpy() / 6), np.floor(rim["y"].to_numpy() / 6), "6 in cell", {"season": s}, topn=10)
            Dd += dist_bins(rim, "sportradar", "dist", {"season": s})
            log(f"{s}: SR rim parts a-d done ({len(rim):,} rim attempts)")
        # ---- ESPN side of the matched games
        mp = RES / f"game_match_{s}.parquet"
        es = s + 1
        if not mp.exists():
            continue
        m = pd.read_parquet(mp)
        e = INV.espn_fga(es)
        efa = INV.espn_fga_rows(e)
        ef = efa[efa["game_id"].isin(set(m["espn_game_id"]))].copy()
        gl = ef.groupby("game_id")["located"].mean()
        loc_games = set(gl.index[gl > 0])
        efl = ef[ef["game_id"].isin(loc_games)].copy()            # matched games ESPN locates
        efl["gameId"] = efl["game_id"]
        efl["cls"] = np.where(efl["rimtype"] != "", "rim", np.where(efl["is3"], "three", "jump2"))
        er = efl[efl["cls"] == "rim"].copy()
        # ESPN (a)-(d) on the commonly located games
        ESPN_A += asym_gt6(er, "espn (matched, espn-located games)", "dist", {"season": s})
        er_all = er.copy()
        er_file = efa[efa["rimtype"] != ""].copy()
        er_file["gameId"] = er_file["game_id"]
        ESPN_B += missing_coord(er_file, "espn (all games in file)", {"season": s})
        ei, ej = np.round(er["coordinate_y_raw"].to_numpy()), np.round(er["coordinate_y"].to_numpy())
        ESPN_C += pixel_rows(er, "espn rim attempts", ei, ej, "1 ft pixel (along, lateral)", {"season": s}, topn=10)
        ESPN_D += dist_bins(er, "espn (matched)", "dist", {"season": s})
        # same games on the Sportradar side
        if f["typed"].iloc[0]:
            sm = f[f["gameId"].isin(set(m.loc[m["espn_game_id"].isin(loc_games), "sr_game_id"]))]
            smr = sm[sm["cls"] == "rim"].copy()
            SRm_A += asym_gt6(smr, "sportradar (same games)", "dist", {"season": s})
            SRm_D += dist_bins(smr, "sportradar (same games)", "dist", {"season": s})
        # ESPN all located games of the file, for the era reference
        eall = efa[efa["located"]].copy()
        eall["gameId"] = eall["game_id"]
        eall = eall[eall["rimtype"] != ""]
        ESPN_A += asym_gt6(eall.assign(located=True), "espn (all located games in file)", "dist", {"season": s})
        # ---- play-level join
        sr_g = f.rename(columns={"gameId": "game_id"})
        j, jinfo = join_play_level(sr_g, ef, m)
        jinfo["season"] = s
        rows, extra = agreement(j)
        far = extra.pop("sr_far_given_espn_near")
        far["season"] = s
        FAR.append(far)
        jinfo.update(extra)
        jinfo["joined_both_located"] = extra["pairs_both_located"]
        join_info.append(jinfo)
        for r in rows:
            r["season"] = s
        E_agree += rows
        log(f"{s}: ESPN side + join done (joined {len(j):,}, both located {jinfo['joined_both_located']:,})")
    R["a_sr"], R["a_sr_dunk_centroid_basket_sensitivity"], R["b_sr"], R["c_sr"], R["d_sr"] = A, sens, B, C, Dd
    R["e_join"], R["e_agreement"], R["e_sr_far_given_espn_near"] = join_info, E_agree, FAR
    R["e_espn_a"], R["e_espn_b"], R["e_espn_c"], R["e_espn_d"] = ESPN_A, ESPN_B, ESPN_C, ESPN_D
    R["e_sr_same_games_a"], R["e_sr_same_games_d"] = SRm_A, SRm_D
    jdump(R, DOCS / "sportradar_rim_asymmetry_2026-10-27.json")

    md = ["# Sportradar rim-placement asymmetry (2026-10-27)\n",
          "Research-only data (user ruling 2026-10-27; never synced to HF or rfq-collab). Diagnostic; nothing is fit or adopted. "
          "Script `scripts/diag_sportradar_rim_v1.py` (loader and ESPN match from `scripts/diag_sportradar_inventory_v1.py`, "
          "inventory in `docs/tests/sportradar_inventory_2026-10-27.md`); tables are also in `docs/tests/sportradar_rim_asymmetry_2026-10-27.json`. "
          "It repeats the rim measurements of `docs/tests/coordinate_audit_2026-10-27.md` (ESPN/CBBD) on an independent vendor.\n",
          "## Scope, definitions, floors\n",
          "- **Seasons.** Sportradar season 2013 = 2013-14 ... 2017 = 2017-18. Season 2013 carries no `shot_type` and its descriptions say only 'two point shot', "
          "so layups, dunks and tips cannot be identified: every rim measurement is NOT DEFINED for 2013 (it appears only in the distance-agreement table, with ESPN's shot type). "
          "No outcome data from the sealed 2025-26 season is used.\n",
          "- **Rim class (mapping).** layup = `shot_type == 'layup'` or description contains 'layup'; dunk = `shot_type == 'dunk'` or 'dunk'; tip = `shot_type == 'tip shot'` or 'tip shot'. "
          "`hook shot` is NOT rim (it stays in jump2, as ESPN's JumpShot does). `shot_subtype` (2016-17 only: driving, putback, alley-oop, finger roll...) is not used for the class; "
          "it is a modifier of layup / dunk / jump shot. The ESPN side uses `type_text` LayUpShot / DunkShot / TipShot.\n",
          "- **Frame and distance.** Inches; court folded onto one half (x > 564 -> 1128 - x); distance = feet from the textbook basket (63 in from the baseline, 300 in across), "
          "dunk median distance 2.4-3.0 ft per season (inventory doc). ESPN distance = sqrt(raw_y^2 + y^2) (basket-relative frame, as in the audit). "
          "Section (a) also reports the 6-ft rate with the basket moved to each season's dunk centroid (sensitivity).\n",
          "- **Floors.** Paired comparisons are within game, weights n_a n_b / (n_a + n_b), floor = SD over 300 bootstrap resamples of games (200 for pixel rows), "
          "exactly `paired_diff` from `scripts/diag_coordinate_audit_v1.py`. Cells under 500 attempts in either group are flagged UNDERPOWERED and not read.\n",
          "- **Reference (ESPN/CBBD, 2022-2025 audit).** Layups/tips charted more than 6 ft from the rim: 6-7% of makes vs 15-16% of misses, within-game gap -8.4 to -8.9 pts, floor 0.2; "
          "rim makes 1.3-2.1 pts less likely to carry a coordinate; one spike pixel at the basket making 2.3-3.2 pts less than its ring.\n"]
    def sec(title, df, note=""):
        md.append(f"## {title}\n")
        if note:
            md.append(note + "\n")
        md.append(md_table(df))
    sec("(a) Rim attempts charted more than 6 ft from the rim, makes vs misses", pd.DataFrame(A)[
        [c for c in ["label", "season", "group", "n_makes", "n_misses", "rate_makes_pct", "rate_misses_pct", "pooled_diff_pts", "games_paired",
                     "within_game_diff_pts", "boot_floor_pts", "z", "flag"] if c in pd.DataFrame(A).columns]],
        "`rate_*_pct` = percent of that group's located rim attempts with distance > 6 ft. `within_game_diff_pts` = makes minus misses within game.")
    sec("(a) sensitivity: same measurement with the basket at each season's dunk centroid", pd.DataFrame(sens)[
        ["season", "basket", "group", "n_makes", "n_misses", "rate_makes_pct", "rate_misses_pct", "within_game_diff_pts", "boot_floor_pts", "z", "flag"]])
    sec("(b) Rim attempts WITHOUT a coordinate, makes vs misses (Sportradar)", pd.DataFrame(B)[
        ["season", "n_makes", "n_misses", "rate_makes_pct", "rate_misses_pct", "pooled_diff_pts", "within_game_diff_pts", "boot_floor_pts", "flag"]],
        "Sportradar carries a coordinate on every attempt it records (and no (0,0) or sentinel rows), so there is nothing to compare: the asymmetry in WHICH attempts get a coordinate cannot occur here.")
    cc = pd.DataFrame(C)
    for lab, grid in (("sportradar rim attempts", "1 in pixel"), ("sportradar all attempts", "1 in pixel"), ("sportradar rim attempts", "6 in cell")):
        sub = cc[(cc["label"] == lab) & (cc["grid"] == grid)].drop(columns=["label", "grid"])
        md.append(f"## (c) Top pixels: {lab}, grid {grid}\n")
        md.append("`neighbour_ratio` = count / mean count of the 8 adjacent cells on the same grid; the ESPN/CBBD spike pixel had a ratio of 8-9 and share 2.7-3.0% of located attempts. "
                  "`within_game_diff_pts` = pixel minus ring make rate within game.\n")
        for s in SEASONS:
            ss = sub[sub["season"] == s].drop(columns=["season"])
            if len(ss):
                md.append(f"**Season {s}**\n")
                md.append(md_table(ss.head(10 if grid != "1 in pixel" or "all" in lab else 20)))
    sec("(d) Rim make rate and placement by distance bin (Sportradar)", pd.DataFrame(Dd).drop(columns=["label"]))
    md.append("## (e) Sportradar vs ESPN on the commonly located games, 2013-14 to 2017-18\n")
    md.append("ESPN file = Sportradar season + 1. Games matched on team pair and date (inventory doc). ESPN locates a game entirely or not at all (the located-share columns of the "
              "inventory match table are equal at >0, >=50% and >=90%), so 'commonly located' = matched games in which ESPN has coordinates. Play-level key = (ESPN game, period, "
              "clock seconds, made/missed, shooter last name), required to be unique on both sides.\n")
    md.append(md_table(pd.DataFrame(join_info)))
    md.append("\n**Position and distance agreement on joined plays that both vendors locate** (position error compares basket-relative along-court and |lateral| feet; "
              "distance error compares the two distances from the basket):\n")
    md.append(md_table(pd.DataFrame(E_agree)[["season", "outcome", "class", "pairs", "dist_median_abs_diff_ft", "dist_within_1ft_pct", "dist_gt3ft_pct", "pos_median_err_ft",
                                              "pos_within_1ft_pct", "pos_gt3ft_pct", "dist_corr", "mean_dist_sr_ft", "mean_dist_es_ft", "flag"]]))
    md.append("\n**Outcome-dependent noise test.** Among rim plays that ESPN places within 3 ft of the rim (ESPN places 99.5-99.9% of layups within 6 ft in these seasons), the share Sportradar places more than 6 ft away, makes vs misses, within game:\n")
    fr_ = pd.DataFrame(FAR)
    md.append(md_table(fr_[["season", "n_makes", "n_misses", "rate_makes_pct", "rate_misses_pct", "within_game_diff_pts", "boot_floor_pts", "z", "flag"]]))
    md.append("\n**The same measurements on the ESPN side of those games (and on the Sportradar side of the same games)**\n")
    ea = pd.DataFrame(ESPN_A)
    ea = ea[[c for c in ["label", "season", "group", "n_makes", "n_misses", "rate_makes_pct", "rate_misses_pct", "within_game_diff_pts", "boot_floor_pts", "z", "flag"] if c in ea.columns]]
    md.append("(a) rim > 6 ft:\n")
    md.append(md_table(ea))
    sa = pd.DataFrame(SRm_A)
    md.append("(a) Sportradar, same games:\n")
    md.append(md_table(sa[[c for c in ea.columns if c in sa.columns]]))
    md.append("(b) ESPN rim attempts without a coordinate ALL games in the file (the located-or-not decision is made per game in this vintage, so within-game pairs are almost all identical; the pooled column mixes games):\n")
    md.append(md_table(pd.DataFrame(ESPN_B)[["season", "n_makes", "n_misses", "rate_makes_pct", "rate_misses_pct", "pooled_diff_pts", "within_game_diff_pts", "boot_floor_pts", "flag"]]))
    md.append("(c) ESPN top rim pixels (1 ft grid in the basket-relative frame (along, lateral)):\n")
    ec = pd.DataFrame(ESPN_C)
    for s in SEASONS:
        ss = ec[ec["season"] == s].drop(columns=["label", "grid", "season"])
        if len(ss):
            md.append(f"**Season {s} (ESPN)**\n")
            md.append(md_table(ss.head(6)))
    md.append("(d) rim make rate and placement by distance bin, ESPN (matched) and Sportradar (same games):\n")
    md.append(md_table(pd.concat([pd.DataFrame(ESPN_D), pd.DataFrame(SRm_D)], ignore_index=True)[
        ["label", "season", "bin_ft", "attempts", "pct_of_rim_attempts", "make_rate_pct", "pct_of_makes_in_bin", "pct_of_misses_in_bin", "flag"]]))
    md.append("\n## Verdict\n")
    md.append(VERDICT or "(verdict text pending)")
    (DOCS / "sportradar_rim_asymmetry_2026-10-27.md").write_text("\n".join(md), encoding="utf-8")
    log("rim doc written")


if __name__ == "__main__":
    main()
