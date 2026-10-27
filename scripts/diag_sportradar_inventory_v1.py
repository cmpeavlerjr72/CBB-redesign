#!/usr/bin/env python
"""diag_sportradar_inventory_v1.py -- Sportradar men's pbp (BigQuery public dataset) inventory.

DIAGNOSTIC, research-only data (user ruling 2026-10-27: never synced to the public HF dataset
or rfq-collab). Nothing is fit, nothing is wired into the engine.

Parts written by this script (docs/tests/sportradar_inventory_2026-10-27.{md,json}):
  PART 1  inventory: rows, games, dates, vocabularies, coordinate coverage, the frame (units, where
          the baskets sit, side switching, dunk calibration, three-point arc check), per-team located
          attempts, and the game match to the ESPN hoopR history files (name map documented below).
  PART 3  bands (the shot-location spec's short2 / long2 and corner3 / wing3 / top3) per
          team-season and their drift 2013-2017, next to CBBD 2022-2025 for reference.

The shared loader (`load_sr_fga`, `fold_dist`, `match_games`) is imported by diag_sportradar_rim_v1.py.

Season convention: Sportradar `season` 2013 = 2013-14 (earliest). ESPN hoopR-hist file N is the
season ENDING in year N, so Sportradar season s <-> ESPN file s+1.

    .venv/Scripts/python.exe scripts/diag_sportradar_inventory_v1.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SR_DIR = ROOT / "data/raw/bigquery_ncaa/mbb_pbp_sr"
ESPN_DIR = ROOT / "data/raw/hoopr_hist/pbp"
RES = ROOT / "results" / "sportradar"
DOCS = ROOT / "docs" / "tests"
SEASONS = [2013, 2014, 2015, 2016, 2017]
FGA_EV = ["twopointmade", "twopointmiss", "threepointmade", "threepointmiss"]
COURT_X = 1128.0                 # 94 ft in inches
BASKET_TEXTBOOK = (63.0, 300.0)  # 5.25 ft from the baseline, mid-court (inches)
ARC_FT = 20.75                   # NCAA men's arc 2008-09 .. 2018-19
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def md_table(df: pd.DataFrame, floatfmt: int = 3) -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else (f"{int(v)}" if float(v).is_integer() and abs(v) < 1e7 else f"{v:.{floatfmt}f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def jdump(obj, path: Path) -> None:
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=conv), encoding="utf-8")


def recs(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


# =====================================================================================
# loader
# =====================================================================================
def parse_clock(s: pd.Series) -> np.ndarray:
    """'m:ss' (time remaining in the period) -> seconds."""
    p = s.astype(str).str.extract(r"^\s*(\d+):(\d+(?:\.\d+)?)")
    return (pd.to_numeric(p[0], errors="coerce") * 60 + pd.to_numeric(p[1], errors="coerce")).to_numpy(dtype=float)


def load_sr_raw(season: int, columns=None) -> pd.DataFrame:
    return pd.read_parquet(SR_DIR / f"mbb_pbp_sr_{season}.parquet", columns=columns)


def load_sr_fga(season: int, rebuild: bool = False) -> pd.DataFrame:
    """Field-goal attempts (event_type in twopoint/threepoint made/miss), de-duplicated on event_id.
    Adds folded coordinates (inches), distance (ft, textbook basket) and the rim/jump2/three class.

    Class mapping (documented in the docs): rimtype = dunk if shot_type == 'dunk' or the description
    contains 'dunk'; layup if shot_type == 'layup' or 'layup'; tip if shot_type == 'tip shot' or the
    description contains 'tip shot'. rim = any of the three. 'hook shot' is NOT rim (it joins jump2,
    as ESPN's JumpShot type does). three = three_point_shot flag or a threepoint event type. jump2 = every
    other two-point attempt. Season 2013 carries no shot_type and no layup/dunk text ('two point shot'), so
    its two-point attempts are 'two_untyped' (rim and jump2 cannot be separated)."""
    f = RES / f"fga_{season}.parquet"
    if f.exists() and not rebuild:
        return pd.read_parquet(f)
    d = load_sr_raw(season)
    n_all = len(d)
    d = d[d["event_type"].isin(FGA_EV)].copy()
    n_fga_raw = len(d)
    d = d.drop_duplicates("event_id").copy()
    out = pd.DataFrame({
        "game_id": d["game_id"].to_numpy(), "season": season,
        "date_et": d["scheduled_date"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize().to_numpy(),
        "period": d["period"].astype("float").to_numpy(), "clock_sec": parse_clock(d["game_clock"]),
        "elapsed": d["elapsed_time_sec"].astype("float").to_numpy(),
        "team_id": d["team_id"].to_numpy(), "team_market": d["team_market"].to_numpy(),
        "home_id": d["home_id"].to_numpy(), "away_id": d["away_id"].to_numpy(),
        "home_market": d["home_market"].to_numpy(), "away_market": d["away_market"].to_numpy(),
        "team_basket": d["team_basket"].to_numpy(), "player_id": d["player_id"].to_numpy(),
        "player": d["player_full_name"].to_numpy(), "event_id": d["event_id"].to_numpy(),
        "event_type": d["event_type"].to_numpy(), "desc": d["event_description"].fillna("").str.lower().to_numpy(),
        "shot_type": d["shot_type"].to_numpy(), "shot_subtype": d["shot_subtype"].to_numpy(),
        "x": d["event_coord_x"].astype(float).to_numpy(), "y": d["event_coord_y"].astype(float).to_numpy(),
        "three_flag": d["three_point_shot"].astype("boolean").fillna(False).astype(bool).to_numpy(),
    })
    out["made"] = out["event_type"].str.endswith("made")
    out["is3"] = out["event_type"].str.startswith("three")
    st = out["shot_type"].fillna("")
    ds = out["desc"]
    dunk = (st == "dunk") | ds.str.contains("dunk")
    lay = (st == "layup") | ds.str.contains("layup|lay up")
    tip = (st == "tip shot") | ds.str.contains("tip shot")
    out["rimtype"] = np.where(dunk, "dunk", np.where(lay, "layup", np.where(tip, "tip", "")))
    typed = (st != "").any() and (st != "").mean() > 0.5
    out["typed"] = bool(typed)
    out["cls"] = np.where(out["is3"], "three", np.where(out["rimtype"] != "", "rim",
                          np.where(typed, "jump2", "two_untyped")))
    out["located"] = out["x"].notna() & out["y"].notna()
    xf, dd, th = fold_dist(out["x"].to_numpy(), out["y"].to_numpy())
    out["xf"], out["dist"], out["theta"] = xf, dd, th
    out.attrs.update(rows_all=n_all, fga_raw=n_fga_raw)
    out["rows_all"], out["fga_raw"] = n_all, n_fga_raw       # carried so the cache keeps them
    RES.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f, index=False)
    return out


def fold_dist(x, y, basket=BASKET_TEXTBOOK):
    """Fold the court onto the left half; return folded x (in), distance (ft) and angle (deg, 0 = straight out)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xf = np.where(x > COURT_X / 2, COURT_X - x, x)
    dx, dy = xf - basket[0], y - basket[1]
    return xf, np.hypot(dx, dy) / 12.0, np.degrees(np.arctan2(dy, dx))


# =====================================================================================
# name matching, Sportradar `market` -> ESPN hoopR `team_name`
# =====================================================================================
def norm_name(n) -> str:
    n = str(n).lower().replace("&", "and").replace("'", "").replace(".", "")
    n = re.sub(r"\bsaint\b", "st", n)
    n = re.sub(r"\bstate\b", "st", n)
    return re.sub(r"[^a-z0-9]+", "", n)


# Explicit aliases for Division I names that differ between the vendors (Sportradar -> ESPN spelling).
# Found by listing the Sportradar names with no normalised ESPN match and reviewing close candidates.
NAME_ALIASES = {
    "Arkansas-Little Rock": "Little Rock", "Brigham Young": "BYU", "Bryant University": "Bryant",
    "Central Connecticut State": "Central Connecticut", "Citadel": "The Citadel", "Connecticut": "UConn",
    "Detroit": "Detroit Mercy", "Grambling State": "Grambling", "Houston Baptist": "Houston Christian",
    "IPFW": "Purdue Fort Wayne", "Illinois-Chicago": "UIC", "LIU Brooklyn": "Long Island University",
    "Louisiana-Lafayette": "Louisiana", "Louisiana-Monroe": "UL Monroe", "Loyola (IL)": "Loyola Chicago",
    "Loyola (MD)": "Loyola Maryland", "Massachusetts": "UMass", "Massachusetts Lowell": "UMass Lowell",
    "McNeese State": "McNeese", "Miami (FL)": "Miami", "Nebraska-Omaha": "Omaha", "Nicholls State": "Nicholls",
    "North Carolina State": "NC State", "North Carolina-Wilmington": "UNC Wilmington",
    "Saint Joseph's (PA)": "Saint Joseph's", "Sam Houston State": "Sam Houston", "San Jose State": "San Jose State",
    "Seattle": "Seattle U", "Southeastern Louisiana": "SE Louisiana", "Southern Methodist": "SMU",
    "Southern University": "Southern", "St. Francis (BKN)": "St. Francis Brooklyn", "Tennessee-Martin": "UT Martin",
    "Texas A&M-CC": "Texas A&M-Corpus Christi", "Texas Rio Grande Valley": "UT Rio Grande Valley",
    "Texas-Arlington": "UT Arlington", "Texas-Pan American": "UT Rio Grande Valley", "UMKC": "Kansas City",
    "USC Upstate": "South Carolina Upstate", "Virginia Commonwealth": "VCU", "Virginia Military": "VMI",
}
ALIAS_KEYS = {norm_name(k): norm_name(v) for k, v in NAME_ALIASES.items()}


def team_key(n) -> str:
    k = norm_name(n)
    return ALIAS_KEYS.get(k, k)


def espn_fga(espn_season: int) -> pd.DataFrame:
    cols = ["id", "game_id", "game_date", "period_number", "clock_display_value", "type_text", "text", "scoring_play",
            "shooting_play", "team_id", "athlete_id_1", "athlete_name_1", "home_team_name", "away_team_name",
            "home_team_id", "coordinate_x", "coordinate_y", "coordinate_x_raw", "coordinate_y_raw"]
    e = pd.read_parquet(ESPN_DIR / f"play_by_play_{espn_season}.parquet", columns=cols)
    return e


ESPN_FG_TYPES = {"JumpShot", "LayUpShot", "DunkShot", "TipShot", "Three Point Jump Shot"}


def espn_fga_rows(e: pd.DataFrame) -> pd.DataFrame:
    f = e[e["shooting_play"].fillna(False).astype(bool) & e["type_text"].isin(ESPN_FG_TYPES)].copy()
    for c in ("coordinate_x", "coordinate_y", "coordinate_x_raw", "coordinate_y_raw"):
        v = pd.to_numeric(f[c], errors="coerce")
        f[c] = v.where(v.abs() <= 1e6)
    f["located"] = f["coordinate_x"].notna() & f["coordinate_y"].notna()
    f["made"] = f["scoring_play"].fillna(False).astype(bool)
    f["is3"] = f["type_text"] == "Three Point Jump Shot"
    t = f["text"].fillna("").str.lower()
    f["rimtype"] = np.where(f["type_text"] == "DunkShot", "dunk", np.where(f["type_text"] == "LayUpShot", "layup",
                            np.where(f["type_text"] == "TipShot", "tip", "")))
    f["dist"] = np.hypot(f["coordinate_y_raw"], f["coordinate_y"])       # raw_y is basket-relative (audit part 1a)
    f["clock_sec"] = parse_clock(f["clock_display_value"])
    f["date"] = pd.to_datetime(f["game_date"]).dt.normalize()
    return f


def match_games(sr: pd.DataFrame, e: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Match Sportradar games to ESPN games on the unordered team pair and date (within +/- 1 day)."""
    g = sr.drop_duplicates("game_id")[["game_id", "date_et", "home_market", "away_market"]].copy()
    g["hk"], g["ak"] = g["home_market"].map(team_key), g["away_market"].map(team_key)
    eg = e.drop_duplicates("game_id")[["game_id", "game_date", "home_team_name", "away_team_name"]].copy()
    eg["date"] = pd.to_datetime(eg["game_date"]).dt.normalize()
    eg["hk"], eg["ak"] = eg["home_team_name"].map(team_key), eg["away_team_name"].map(team_key)
    names = set(eg["hk"]) | set(eg["ak"])
    g["resolvable"] = g["hk"].isin(names) & g["ak"].isin(names)
    eg["pair"] = [frozenset(p) for p in zip(eg["hk"], eg["ak"])]
    g["pair"] = [frozenset(p) for p in zip(g["hk"], g["ak"])]
    by_pair: dict = {}
    for r in eg.itertuples(index=False):
        by_pair.setdefault(r.pair, []).append((r.date, r.game_id, r.hk))
    rows = []
    for r in g.itertuples(index=False):
        best = None
        for (d, gid, hk) in by_pair.get(r.pair, []):
            dd = abs((d - r.date_et).days)
            if dd <= 1 and (best is None or dd < best[0]):
                best = (dd, gid, hk)
        if best is not None:
            rows.append({"sr_game_id": r.game_id, "espn_game_id": best[1], "date_diff": best[0], "home_swapped": best[2] != r.hk})
    m = pd.DataFrame(rows)
    dup = int(m["espn_game_id"].duplicated().sum()) if len(m) else 0
    if dup:                                         # one ESPN game claimed twice: keep exact-date match, drop the rest
        m = m.sort_values("date_diff").drop_duplicates("espn_game_id").reset_index(drop=True)
    info = {"sr_games": int(len(g)), "sr_games_both_names_in_espn": int(g["resolvable"].sum()), "matched": int(len(m)),
            "espn_games": int(len(eg)), "dup_espn_claims_dropped": dup,
            "unmatched_unresolvable_names": sorted(set(g.loc[~g["resolvable"], "home_market"]) | set(g.loc[~g["resolvable"], "away_market"]))}
    return m, info


# =====================================================================================
# PART 1
# =====================================================================================
def vocab_counts(s: pd.Series, top: int = 40) -> dict:
    return {str(k): int(v) for k, v in s.fillna("(null)").value_counts().head(top).items()}


def season_inventory(s: int) -> dict:
    raw = load_sr_raw(s)
    f = load_sr_fga(s)
    ev = raw["event_type"]
    games = raw["game_id"].nunique()
    d_et = raw["scheduled_date"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    inv = {"season": s, "rows": len(raw), "games": int(games), "date_min": str(d_et.min().date()), "date_max": str(d_et.max().date()),
           "teams": int(raw["team_id"].nunique()), "fga_rows_raw": int(ev.isin(FGA_EV).sum()), "fga_rows": len(f),
           "fga_duplicate_event_ids_dropped": int(ev.isin(FGA_EV).sum() - len(f)),
           "fga_per_game": len(f) / games, "periods": vocab_counts(raw["period"].astype("Int64").astype(str), 10),
           "event_type": vocab_counts(ev, 60), "type": vocab_counts(raw["type"], 30),
           "shot_type_fga": vocab_counts(f["shot_type"], 20), "shot_subtype_fga": vocab_counts(f["shot_subtype"], 20),
           "class_fga": vocab_counts(f["cls"], 10), "rimtype_fga": vocab_counts(f["rimtype"].replace("", "(none)"), 10),
           "located_pct_fga": 100 * float(f["located"].mean()),
           "x_y_min_max": [float(f["x"].min()), float(f["x"].max()), float(f["y"].min()), float(f["y"].max())],
           "x_p01_p99": [float(f["x"].quantile(.01)), float(f["x"].quantile(.99))],
           "y_p01_p99": [float(f["y"].quantile(.01)), float(f["y"].quantile(.99))],
           "distinct_x": int(f["x"].nunique()), "distinct_y": int(f["y"].nunique()),
           "xy_integer_pct": 100 * float(((f["x"] % 1 == 0) & (f["y"] % 1 == 0)).mean()),
           "zero_zero_rows": int(((f["x"] == 0) & (f["y"] == 0)).sum()),
           "three_flag_vs_event_agree_pct": 100 * float((f["three_flag"] == f["is3"]).mean()),
           "team_basket_null_pct_fga": 100 * float(f["team_basket"].isna().mean())}
    # ---- frame
    dk = f[f["rimtype"] == "dunk"]
    fr = {"season": s, "dunks": len(dk)}
    if len(dk) >= 200:
        cx, cy = float(dk["xf"].median()), float(dk["y"].median())
        fr.update(dunk_centroid_x_in=cx, dunk_centroid_y_in=cy, dunk_centroid_x_ft=cx / 12, dunk_centroid_y_ft=cy / 12,
                  dunk_median_dist_textbook_ft=float(dk["dist"].median()),
                  dunk_median_dist_at_own_centroid_ft=float(np.median(np.hypot(dk["xf"] - cx, dk["y"] - cy)) / 12))
        fr["dunk_dist_q10_q25_q50_q75_q90"] = [float(v) for v in dk["dist"].quantile([.1, .25, .5, .75, .9])]
    side_pos = (f["x"] > COURT_X / 2)
    fr["x_right_half_pct"] = 100 * float(side_pos.mean())
    ok = f["team_basket"].notna()
    fr["team_basket_matches_x_side_pct"] = 100 * float((side_pos[ok] == (f.loc[ok, "team_basket"] == "right")).mean())
    # side switching at half, per team-game: majority team_basket in P1 vs P2, and majority coordinate side
    h = f[ok & f["period"].isin([1, 2])].copy()
    h["tb_right"] = (h["team_basket"] == "right").astype(float)
    h["x_right"] = (h["x"] > COURT_X / 2).astype(float)
    a = h.groupby(["game_id", "team_id", "period"]).agg(tb=("tb_right", "mean"), xr=("x_right", "mean"), n=("x", "size")).reset_index()
    a = a[a["n"] >= 5]
    pv_tb = a.pivot_table(index=["game_id", "team_id"], columns="period", values="tb").dropna()
    pv_x = a.pivot_table(index=["game_id", "team_id"], columns="period", values="xr").dropna()
    fr["team_games_p1_p2"] = int(len(pv_tb))
    fr["switch_at_half_team_basket_pct"] = 100 * float(((pv_tb[1] - .5) * (pv_tb[2] - .5) < 0).mean())
    fr["switch_at_half_coord_side_pct"] = 100 * float(((pv_x[1] - .5) * (pv_x[2] - .5) < 0).mean())
    # home team's basket in P1: does home always attack the same end?
    hh = f[ok & (f["period"] == 1) & (f["team_id"] == f["home_id"])]
    fr["home_attacks_right_p1_pct"] = 100 * float((hh["team_basket"] == "right").mean())
    # ---- arc
    t3 = f[f["is3"]]
    t2 = f[~f["is3"]]
    arc = {"season": s, "n_three": len(t3), "n_two": len(t2),
           "three_inside_20.75_pct": 100 * float((t3["dist"] < ARC_FT).mean()),
           "three_inside_19.75_pct(1ft slack)": 100 * float((t3["dist"] < ARC_FT - 1).mean()),
           "three_dist_q01_q05_q50": [float(v) for v in t3["dist"].quantile([.01, .05, .5])],
           "two_outside_20.75_pct": 100 * float((t2["dist"] > ARC_FT).mean()),
           "two_outside_21.75_pct(1ft slack)": 100 * float((t2["dist"] > ARC_FT + 1).mean()),
           "three_inside_22.15_pct(modern arc, reference)": 100 * float((t3["dist"] < 22.15).mean())}
    # ---- per-team located attempts
    pt = {"season": s}
    for side, key in (("offence", f["team_id"]), ("defence", np.where(f["team_id"] == f["home_id"], f["away_id"], f["home_id"]))):
        c = f[f["located"]].groupby(key).size()
        pt[side] = {"teams": int(len(c)), "min": int(c.min()), "p10": float(c.quantile(.1)), "p50": float(c.median()),
                    "p90": float(c.quantile(.9)), "teams_under_100": int((c < 100).sum())}
    gm = f.groupby("game_id").size()
    pt["fga_per_game_p10_p50_p90"] = [float(v) for v in gm.quantile([.1, .5, .9])]
    return {"inv": inv, "frame": fr, "arc": arc, "per_team": pt}


def part1_match() -> tuple[list[dict], list[dict]]:
    mrows, minfo = [], []
    for s in SEASONS:
        es = s + 1
        if not (ESPN_DIR / f"play_by_play_{es}.parquet").exists():
            minfo.append({"sr_season": s, "espn_file": es, "note": "no ESPN file"})
            continue
        sr = load_sr_fga(s)
        e = espn_fga(es)
        m, info = match_games(sr, e)
        m["sr_season"] = s
        m.to_parquet(RES / f"game_match_{s}.parquet", index=False)
        ef = espn_fga_rows(e)
        ef = ef[ef["game_id"].isin(set(m["espn_game_id"]))]
        gl = ef.groupby("game_id").agg(n=("located", "size"), loc=("located", "sum"))
        gl["frac"] = gl["loc"] / gl["n"]
        info.update(sr_season=s, espn_file=es, match_rate_all_sr_games_pct=100 * info["matched"] / info["sr_games"],
                    match_rate_resolvable_pct=100 * info["matched"] / max(info["sr_games_both_names_in_espn"], 1),
                    matched_date_diff_0_pct=100 * float((m["date_diff"] == 0).mean()),
                    matched_games_espn_fga_located_gt0=int((gl["loc"] > 0).sum()),
                    matched_games_espn_fga_located_ge50pct=int((gl["frac"] >= .5).sum()),
                    matched_games_espn_fga_located_ge90pct=int((gl["frac"] >= .9).sum()),
                    espn_fga_in_matched_games=int(len(ef)), espn_fga_located_in_matched_games=int(ef["located"].sum()),
                    espn_located_pct_matched=100 * float(ef["located"].mean()) if len(ef) else np.nan,
                    espn_located_pct_all_games=100 * float(espn_fga_rows(e)["located"].mean()),
                    espn_dunk_median_dist_ft=float(ef.loc[(ef["rimtype"] == "dunk") & ef["located"], "dist"].median()))
        info["unmatched_unresolvable_names"] = info["unmatched_unresolvable_names"][:80]
        minfo.append(info)
        log(f"match {s}->{es}: {info['matched']}/{info['sr_games']} games ({info['match_rate_all_sr_games_pct']:.1f}%)")
    return mrows, minfo


# =====================================================================================
# PART 3: bands
# =====================================================================================
def bands(cls: pd.Series, d: np.ndarray, theta: np.ndarray, y_off_ft: np.ndarray) -> pd.Series:
    """Spec section 2: short2 d<=10 / long2 d>10 for jump2; corner3 (within 4.0 ft of a sideline, i.e. |lateral|>=21 ft,
    and d<=24 ft), wing3 (|theta|>=30), top3 (|theta|<30) for threes. Everything else ''."""
    b = np.full(len(cls), "", dtype=object)
    j = (cls == "jump2").to_numpy()
    t = (cls == "three").to_numpy()
    b[j & (d <= 10)] = "short2"
    b[j & (d > 10)] = "long2"
    corner = t & (np.abs(y_off_ft) >= 21.0) & (d <= 24)
    b[corner] = "corner3"
    b[t & ~corner & (np.abs(theta) >= 30)] = "wing3"
    b[t & ~corner & (np.abs(theta) < 30)] = "top3"
    return pd.Series(b, index=cls.index)


def band_tables(df: pd.DataFrame, team_col: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """df has season, team (team_col), band. Pooled shares by season and team-season distribution of shares."""
    pooled, team = [], []
    for s, g in df.groupby("season"):
        for cl, bl in (("jump2", ["short2", "long2"]), ("three", ["corner3", "wing3", "top3"])):
            gg = g[g["band"].isin(bl)]
            n = len(gg)
            if not n:
                continue
            row = {"source": label, "season": s, "class": cl, "attempts": n}
            for b in bl:
                row[f"{b}_pct"] = 100 * float((gg["band"] == b).mean())
            pooled.append(row)
            t = gg.groupby(team_col)["band"].agg(["size"]).join(
                pd.concat([(gg["band"] == b).groupby(gg[team_col]).sum().rename(b) for b in bl], axis=1))
            t = t[t["size"] >= 50]
            for b in bl:
                sh = t[b] / t["size"]
                p = float(gg["band"].eq(b).mean())
                binom_sd = float(np.sqrt((p * (1 - p) / t["size"]).mean()))     # SD of team shares if all teams had the pooled share
                team.append({"source": label, "season": s, "class": cl, "band": b, "teams_ge50": int(len(t)),
                             "median_attempts_per_team": float(t["size"].median()),
                             "share_p10_pct": 100 * float(sh.quantile(.1)), "share_median_pct": 100 * float(sh.median()),
                             "share_p90_pct": 100 * float(sh.quantile(.9)), "share_min_pct": 100 * float(sh.min()),
                             "team_share_sd_pct": 100 * float(sh.std()), "binomial_sd_if_equal_pct": 100 * binom_sd,
                             "excess_sd_pct": 100 * float(np.sqrt(max(sh.var() - binom_sd ** 2, 0)))})
    return pd.DataFrame(pooled), pd.DataFrame(team)


def part3() -> dict:
    out = {}
    sr_all = []
    for s in SEASONS:
        f = load_sr_fga(s)
        f = f[f["located"]].copy()
        f["band"] = bands(f["cls"], f["dist"].to_numpy(), f["theta"].to_numpy(), (f["y"].to_numpy() - 300.0) / 12.0)
        sr_all.append(f[["season", "team_id", "cls", "band", "made"]])
    SR = pd.concat(sr_all, ignore_index=True)
    p_sr, t_sr = band_tables(SR, "team_id", "sportradar")
    # CBBD reference from the coordinate-audit master tables (2022-2025 only; located rows only; 2026 stays sealed)
    cb = []
    for s in (2022, 2023, 2024, 2025):
        pth = ROOT / f"results/coordinate_audit/master_{s}.parquet"
        if not pth.exists():
            continue
        m = pd.read_parquet(pth, columns=["season", "teamId", "cls", "xf", "y", "located"])
        m = m[m["located"] & m["cls"].isin(["jump2", "three"])].copy()
        dx, dy = m["xf"].to_numpy() - 52.5, m["y"].to_numpy() - 250.0
        d = np.hypot(dx, dy) / 10.0
        th = np.degrees(np.arctan2(dy, dx))
        m["band"] = bands(m["cls"], d, th, dy / 10.0)
        cb.append(m.rename(columns={"teamId": "team_id"})[["season", "team_id", "cls", "band"]])
    if cb:
        p_cb, t_cb = band_tables(pd.concat(cb, ignore_index=True), "team_id", "cbbd_located")
        p_sr, t_sr = pd.concat([p_sr, p_cb], ignore_index=True), pd.concat([t_sr, t_cb], ignore_index=True)
    out["pooled"], out["team"] = p_sr, t_sr
    return out


# =====================================================================================
# main
# =====================================================================================
def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    R = {"created": pd.Timestamp.now("UTC").isoformat(), "seasons": SEASONS, "basket_textbook_in": BASKET_TEXTBOOK}
    invs, frames, arcs, pts = [], [], [], []
    for s in SEASONS:
        r = season_inventory(s)
        invs.append(r["inv"]); frames.append(r["frame"]); arcs.append(r["arc"]); pts.append(r["per_team"])
        log(f"season {s}: {r['inv']['rows']:,} rows, {r['inv']['games']} games, {r['inv']['fga_rows']:,} FGA")
    R["inventory"], R["frame"], R["arc"], R["per_team"] = invs, frames, arcs, pts
    _, minfo = part1_match()
    R["name_aliases"] = NAME_ALIASES
    R["match"] = minfo
    p3 = part3()
    R["part3_pooled"], R["part3_team"] = recs(p3["pooled"]), recs(p3["team"])
    jdump(R, DOCS / "sportradar_inventory_2026-10-27.json")

    md = ["# Sportradar men's pbp (BigQuery public dataset) inventory (2026-10-27)\n",
          "Research-only data (user ruling 2026-10-27): never synced to the public HF dataset or rfq-collab. Descriptive; nothing is fit or adopted. "
          "Script `scripts/diag_sportradar_inventory_v1.py`; tables are also in `docs/tests/sportradar_inventory_2026-10-27.json`. "
          "Files `data/raw/bigquery_ncaa/mbb_pbp_sr/mbb_pbp_sr_<season>.parquet`, season 2013 = 2013-14 ... 2017 = 2017-18. "
          "FGA = `event_type` in twopointmade / twopointmiss / threepointmade / threepointmiss, de-duplicated on `event_id`. "
          "Distances are feet from the textbook basket (5.25 ft from the baseline, mid-width) after folding the court onto one half.\n"]
    md.append("## 1. Files as delivered\n")
    t = pd.DataFrame([{k: v for k, v in i.items() if k in ("season", "rows", "games", "date_min", "date_max", "teams", "fga_rows_raw", "fga_rows",
                                                           "fga_duplicate_event_ids_dropped", "fga_per_game", "located_pct_fga", "distinct_x", "distinct_y",
                                                           "xy_integer_pct", "zero_zero_rows", "three_flag_vs_event_agree_pct", "team_basket_null_pct_fga")} for i in invs])
    md.append(md_table(t))
    md.append("Games per season are 1,745-2,598, roughly a third of Division I (CBBD has about 6,000 per season): Sportradar here is a sample of games, "
              "and the file includes non-D-I exhibition opponents. The 2017-18 file holds 970 exact duplicate FGA rows in five games (same event_id, different load_timestamp), removed.\n")
    md.append("### Vocabularies\n")
    for key, title in (("event_type", "event_type (all rows, count)"), ("type", "type (all rows)"),
                       ("shot_type_fga", "shot_type (FGA rows)"), ("shot_subtype_fga", "shot_subtype (FGA rows)"),
                       ("class_fga", "Derived class (FGA rows; rim = shot_type layup/dunk/tip or description text, jump2 includes hook shots, 2013 has no shot type)"),
                       ("rimtype_fga", "Derived rim sub-type (FGA rows)")):
        keys = sorted({k for i in invs for k in i[key]}, key=lambda k: -sum(i[key].get(k, 0) for i in invs))
        md.append(f"**{title}**\n")
        md.append(md_table(pd.DataFrame([{"season": i["season"], **{k: i[key].get(k, 0) for k in keys}} for i in invs])))
    md.append("## 2. Coordinates and the frame\n")
    c = pd.DataFrame([{"season": i["season"], "x_min": i["x_y_min_max"][0], "x_max": i["x_y_min_max"][1], "y_min": i["x_y_min_max"][2],
                       "y_max": i["x_y_min_max"][3], "x_p01": i["x_p01_p99"][0], "x_p99": i["x_p01_p99"][1], "y_p01": i["y_p01_p99"][0],
                       "y_p99": i["y_p01_p99"][1], "located_pct_fga": i["located_pct_fga"]} for i in invs])
    md.append(md_table(c))
    fr = pd.DataFrame([{k: (v if not isinstance(v, list) else str([round(z, 2) for z in v])) for k, v in f.items()} for f in frames])
    md.append("Frame facts per season (inches unless the name says ft):\n")
    md.append(md_table(fr))
    md.append("\n### Three-point arc check (arc of the era = 20.75 ft; the 22.15 ft modern arc is shown for reference only)\n")
    md.append(md_table(pd.DataFrame([{k: (v if not isinstance(v, list) else str([round(z, 2) for z in v])) for k, v in a.items()} for a in arcs])))
    md.append("\n### Per-team located attempts per season (offence = shooter's team; defence = the opponent)\n")
    rows = []
    for p in pts:
        for side in ("offence", "defence"):
            rows.append({"season": p["season"], "side": side, **p[side]})
    md.append(md_table(pd.DataFrame(rows)))
    md.append("\n## 3. Match to the ESPN hoopR history files\n")
    md.append("Sportradar season s is the season starting in autumn of s; the ESPN file `play_by_play_<s+1>.parquet` is the same season. "
              "Games match on the UNORDERED team pair (so neutral-site home/away swaps do not matter) and date within one day "
              "(Sportradar `scheduled_date` converted UTC -> America/New_York; ESPN `game_date`). Names are matched after a normaliser "
              "(lower case, `&` -> `and`, apostrophes and dots dropped, `Saint`/`State` -> `st`, non-alphanumerics dropped) plus an explicit alias "
              "map for the Division I names the two vendors spell differently (the map is in the json under `name_aliases`: "
              + ", ".join(f"{k} = {v}" for k, v in list(NAME_ALIASES.items())[:12]) + ", ...). "
              "'Resolvable' means both Sportradar team names exist in that ESPN file's name universe (the rest are mostly non-D-I exhibition opponents).\n")
    mt = pd.DataFrame([{k: v for k, v in m.items() if k not in ("unmatched_unresolvable_names",)} for m in minfo])
    mt = mt[["sr_season", "espn_file"] + [c for c in mt.columns if c not in ("sr_season", "espn_file")]]
    md.append(md_table(mt))
    md.append("\nSportradar locates every attempt it records, so the number of games BOTH vendors locate equals the number of matched games in which "
              "ESPN locates something: `matched_games_espn_fga_located_gt0` (any), `_ge50pct`, `_ge90pct`. The play-level counts (plays both vendors locate) are in the rim-asymmetry doc, part (e).\n")
    md.append("\n## 4. Bands (PART 3): short2 / long2 and corner3 / wing3 / top3, per team-season and drift 2013-2017\n")
    md.append("Band definitions are section 2 of `docs/models/shot_location/experiments_DRAFT_shot_location_stage_2026-10-27.md`, applied in the folded frame "
              "with the textbook basket: short2 d <= 10 ft, long2 d > 10 ft (jump2 = non-rim two-point attempts, hook shots included); "
              "corner3 = within 4.0 ft of a sideline (|lateral offset| >= 21 ft) and d <= 24 ft; wing3 = not corner and |theta| >= 30 deg from the centre line; top3 = |theta| < 30 deg. "
              "Season 2013 carries no shot type, so jump2 cannot be separated from layups and dunks: its jump2 bands are not computed (threes are). "
              "Rows are located attempts only (Sportradar locates 100%). The CBBD rows are the 2022-2025 located attempts from the coordinate-audit master tables "
              "(2026 stays sealed and is not read); CBBD located attempts are a non-random 80-99% of attempts, so they are a reference, not a truth. "
              "`teams_ge50` counts teams with at least 50 attempts in the class. Sportradar coverage is lopsided by team (per-team located attempts per season: median 171-360, "
              "p90 about 1,800-2,000, so a minority of teams have nearly a full schedule and most have a few games), so team shares are noisy for most teams. `binomial_sd_if_equal_pct` is the team-share SD expected if every team had the pooled share; "
              "`excess_sd_pct` = sqrt(max(observed var - binomial var, 0)) is the real between-team spread.\n")
    md.append("**Reading (drift).** Corner3 steps from 18.1 / 18.2% of located threes in 2013-14 and 2014-15 to 21.5 / 21.5 / 22.6% in 2015-16 to 2017-18, "
              "landing on CBBD's 21.6-21.9% for 2022-25; wing3 falls 47.9 -> 41.5% (CBBD 44.3-44.8%) and top3 stays at 34.0-35.8% (CBBD 33.6-33.8%). "
              "Short2 (non-rim twos within 10 ft) is 39.4 / 44.3 / 42.0 / 43.2% in 2014-15 to 2017-18 against CBBD's 42.8-45.7%. The step between the 2014-15 and 2015-16 files "
              "coincides with the frame shift seen in the dunk centroid (92 in in 2014-15, 84-88 in later files), so a vendor-side re-charting or recalibration is a candidate "
              "explanation as well as a real change in shot selection; the two earliest seasons are the least comparable. "
              "Team-level spread is real in both eras (`excess_sd_pct` 3.7-4.4 points for corner3, 7.4-8.2 for short2/long2 in Sportradar; CBBD 4.2-4.5 and 9.4-10.7), "
              "but Sportradar has far fewer attempts per team, so its team shares are noisy and it is a population-level, not a team-level, window.\n")
    md.append("### Pooled shares by season\n")
    md.append(md_table(p3["pooled"].sort_values(["class", "source", "season"])))
    md.append("\n### Team-season share distribution (teams with >= 50 attempts in the class)\n")
    md.append(md_table(p3["team"].sort_values(["class", "band", "source", "season"])))
    (DOCS / "sportradar_inventory_2026-10-27.md").write_text("\n".join(md), encoding="utf-8")
    log("inventory doc written")


if __name__ == "__main__":
    main()
