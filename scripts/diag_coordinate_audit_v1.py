#!/usr/bin/env python
"""diag_coordinate_audit_v1.py -- coordinate-placement audit (PM ruling 4 of 2026-10-25).

DIAGNOSTIC, not a fit. No model is trained, nothing is wired into the engine.

Question: can shot x/y be trusted as an upstream input (a shot-location stage that would
drive make, foul and rebound models), or is the location signal partly outcome-dependent
charting?

Parts (select with --part; each writes results/coordinate_audit/part*.json + .md fragment):

  master  build the per-attempt master tables (CBBD FGA rows + hoopR coordinates attached)
  1       inventory and reconciliation of the on-disk coordinate sources
  2       coverage structure (which games lack coordinates)
  3       placement audit (canned pixels, outcome-dependent charting, canned response to
          team skill, regime change, text-vs-coordinate charting accuracy)
  report  assemble docs/tests/coordinate_audit_2026-10-27.{md,json}
  all     master, 1, 2, 3, report

SEAL. Season 2026 (= 2025-26) is sealed for selection. This audit reads it only for
non-outcome descriptive statistics (coverage, pixel/charting structure; the module docstring of
cbb_sim.data.seal allows pure data audits). `made` is set to NaN for season 2026 at load time, so
no make rate, no outcome comparison and no rating test can touch it. All outcome tables are 2022-2025.

Season convention: season = ending year (2025 = 2024-25).

    .venv/Scripts/python.exe scripts/diag_coordinate_audit_v1.py --part all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.pbp import events as E  # noqa: E402

ALL_SEASONS = [2022, 2023, 2024, 2025, 2026]
OUT_SEASONS = [2022, 2023, 2024, 2025]          # outcome-bearing (2026 sealed)
CLASSES = ("rim", "jump2", "three")
FGA = ("FGA_rim", "FGA_jump2", "FGA_3")
CLS_SHORT = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}
RIM_MAX_FT = 2.27
HALF_X = 470.0
BASKET = (52.5, 250.0)
ARC_FT = 22.15
CANNED_SHARE = 0.005
MIN_N = 500
BOOT = 300
RES = ROOT / "results" / "coordinate_audit"
DOCS = ROOT / "docs" / "tests"
HOOPR = ROOT / "data/raw/hoopr"
CBBD = ROOT / "data/raw/cbbd"
PAE_DIR = ROOT / "data/processed/models/fg_make/pae_v1"
DESIGN = ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet"
T0 = time.time()

POWER = {"ACC", "Big 12", "Big East", "Big Ten", "SEC"}
POWER_EXTRA = {"Pac-12"}          # power in 2022-2024 only
MID = {"American Athletic", "Atlantic 10", "Mountain West", "West Coast", "Missouri Valley"}


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
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, pd.Timestamp):
            return str(o)
        return str(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=conv), encoding="utf-8")


def recs(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def fold_x(x):
    return np.where(x > HALF_X, 940.0 - x, x)


def dist_ft(xf, y):
    return np.sqrt((xf - BASKET[0]) ** 2 + (y - BASKET[1]) ** 2) / 10.0


# =====================================================================================
# master table
# =====================================================================================
def _hoopr_pbp(season: int) -> pd.DataFrame:
    cols = ["id", "game_id", "season_type", "shooting_play", "type_text", "text", "period_number", "team_id",
            "coordinate_x", "coordinate_y", "coordinate_x_raw", "coordinate_y_raw", "game_date"]
    h = pd.read_parquet(HOOPR / f"pbp/play_by_play_{season}.parquet", columns=cols)
    h = h[h["shooting_play"].fillna(False).astype(bool)].copy()
    for c in ("coordinate_x", "coordinate_y", "coordinate_x_raw", "coordinate_y_raw"):
        v = pd.to_numeric(h[c], errors="coerce")
        h[c + "_sentinel"] = v.abs() > 1e6
        h[c] = v.where(v.abs() <= 1e6)
    return h


def build_master(season: int) -> pd.DataFrame:
    cols = tuple(E.PLAY_COLUMNS) + ("gameStartDate", "sourceId", "gameSourceId", "conference")
    p = E.load_plays(season, columns=cols)
    cls = E.classify_frame(p, rim_override_max_ft=RIM_MAX_FT)
    cls0 = E.classify_frame(p, rim_override_max_ft=0.0)
    keep = cls.isin(list(FGA)).to_numpy()
    f = p.loc[keep].copy()
    f["cls"] = cls[keep].map(CLS_SHORT).astype(str).to_numpy()
    f["cls0"] = cls0[keep].map(CLS_SHORT).astype(str).to_numpy()
    f["x"] = pd.to_numeric(f["shot_location_x"], errors="coerce")
    f["y"] = pd.to_numeric(f["shot_location_y"], errors="coerce")
    f["located"] = (f["x"].notna() & f["y"].notna()).to_numpy()
    if season in OUT_SEASONS:
        made = f["shot_made"]
        if made.dtype == object:
            made = made.map({True: True, False: False})
        f["made"] = made.astype("boolean").fillna(f["scoringPlay"].astype("boolean")).fillna(False).astype(bool).astype(float)
    else:
        f["made"] = np.nan                      # sealed season: outcomes are never read
    f["season"] = season
    f["game_date"] = pd.to_datetime(f["gameStartDate"], utc=True).dt.tz_convert("US/Eastern").dt.tz_localize(None).dt.normalize()
    f["month"] = f["game_date"].dt.month
    xf = fold_x(f["x"].to_numpy(dtype=float))
    f["xf"] = xf
    f["dist"] = dist_ft(xf, f["y"].to_numpy(dtype=float))
    f["pix"] = np.where(f["located"], np.round(xf * 10).astype("float") * 1000 + np.round(f["y"].to_numpy(dtype=float)), np.nan)
    t = f["playText"].fillna("").str.lower()
    tt = np.where(t.str.contains("three point"), "three",
         np.where(t.str.contains("layup|dunk|tip|lay up"), "rimtext",
         np.where(t.str.contains("jumper|jump shot"), "twojump", "other")))
    f["text_cls"] = tt
    f["is_dunk"] = t.str.contains("dunk").to_numpy()
    f["sid"] = pd.to_numeric(f["sourceId"], errors="coerce").astype("Int64")
    # ---- attach hoopR coordinates by ESPN play id (hoopR `id` == CBBD `sourceId`)
    h = _hoopr_pbp(season)
    hh = h[["id", "coordinate_x", "coordinate_y", "coordinate_x_raw", "coordinate_y_raw", "type_text",
            "coordinate_x_sentinel"]].rename(columns={
        "id": "sid", "coordinate_x": "hx", "coordinate_y": "hy", "coordinate_x_raw": "hrx",
        "coordinate_y_raw": "hry", "type_text": "h_type", "coordinate_x_sentinel": "h_sentinel"})
    hh["sid"] = hh["sid"].astype("Int64")
    f = f.merge(hh, on="sid", how="left")
    f["in_hoopr"] = f["h_type"].notna()
    f["h_located"] = f["hx"].notna() & f["hy"].notna()
    keepc = ["season", "gameId", "id", "sid", "gameSourceId", "teamId", "opponentId", "isHomeTeam", "period",
             "secondsRemaining", "shot_shooter_id", "cls", "cls0", "playType", "text_cls", "is_dunk", "made",
             "x", "y", "xf", "dist", "pix", "located", "game_date", "month", "hx", "hy", "hrx", "hry",
             "in_hoopr", "h_located", "h_sentinel", "playText"]
    f = f[keepc].reset_index(drop=True)
    return f


def master_path(season: int) -> Path:
    return RES / f"master_{season}.parquet"


def part_master(a) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    for s in ALL_SEASONS:
        if master_path(s).exists() and not a.rebuild:
            log(f"master {s}: cached")
            continue
        m = build_master(s)
        m.to_parquet(master_path(s), index=False)
        log(f"master {s}: {len(m):,} FGA rows, CBBD located {100 * m['located'].mean():.2f}%, "
            f"hoopR-joined {100 * m['in_hoopr'].mean():.2f}%, hoopR located {100 * m['h_located'].mean():.2f}%")


def load_master(seasons=None) -> pd.DataFrame:
    seasons = seasons or ALL_SEASONS
    d = pd.concat([pd.read_parquet(master_path(s)) for s in seasons], ignore_index=True)
    d["isHome"] = d["isHomeTeam"].astype("boolean").fillna(False).astype(bool)
    return d


# =====================================================================================
# generic paired (within-game) comparison with game bootstrap
# =====================================================================================
def paired_diff(df: pd.DataFrame, mask_a: np.ndarray, mask_b: np.ndarray, ind: np.ndarray, rng, B: int = BOOT) -> dict:
    """Mean of `ind` in group a vs group b. Pooled (unpaired) difference, plus a within-game
    estimate over games holding both groups: sum w_g d_g / sum w_g, w_g = na nb / (na + nb),
    with a nonparametric bootstrap over games (floor)."""
    mask_a = np.asarray(mask_a, bool)
    mask_b = np.asarray(mask_b, bool)
    ind = np.asarray(ind, float)
    ok = np.isfinite(ind)
    mask_a, mask_b = mask_a & ok, mask_b & ok
    na, nb = int(mask_a.sum()), int(mask_b.sum())
    out = {"n_a": na, "n_b": nb, "rate_a": float(ind[mask_a].mean()) if na else np.nan,
           "rate_b": float(ind[mask_b].mean()) if nb else np.nan}
    out["pooled_diff"] = out["rate_a"] - out["rate_b"] if na and nb else np.nan
    out["underpowered"] = bool(min(na, nb) < MIN_N)
    g = df["gameId"].to_numpy()
    sub = mask_a | mask_b
    t = pd.DataFrame({"g": g[sub], "a": mask_a[sub], "v": ind[sub]})
    t["va"] = np.where(t["a"], t["v"], 0.0)
    t["vb"] = np.where(~t["a"], t["v"], 0.0)
    t["ia"] = t["a"].astype(float)
    t["ib"] = (~t["a"]).astype(float)
    gg = t.groupby("g")[["ia", "ib", "va", "vb"]].sum()
    gg = gg[(gg["ia"] > 0) & (gg["ib"] > 0)]
    out["n_games_paired"] = int(len(gg))
    if len(gg) < 5:
        out.update(within_diff=np.nan, boot_se=np.nan, z=np.nan)
        return out
    d = (gg["va"] / gg["ia"] - gg["vb"] / gg["ib"]).to_numpy()
    w = (gg["ia"] * gg["ib"] / (gg["ia"] + gg["ib"])).to_numpy()
    est = float((w * d).sum() / w.sum())
    G = len(gg)
    cnt = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)
    be = (cnt @ (w * d)) / (cnt @ w)
    se = float(be.std(ddof=1))
    out.update(within_diff=est, boot_se=se, z=est / se if se > 0 else np.nan)
    return out


def slope_stats(x: np.ndarray, y: np.ndarray, games: np.ndarray, rng, B: int = 200) -> dict:
    """OLS slope of y on x with game-bootstrap SE (per-game sufficient statistics)."""
    n = len(x)
    if n < 50:
        return {"n": n, "slope": np.nan, "se": np.nan}
    t = pd.DataFrame({"g": games, "n": 1.0, "sx": x, "sy": y, "sxy": x * y, "sxx": x * x})
    gg = t.groupby("g").sum()
    S = gg[["n", "sx", "sy", "sxy", "sxx"]].to_numpy()

    def sl(v):
        nn, sx, sy, sxy, sxx = v
        den = sxx - sx * sx / nn
        return (sxy - sx * sy / nn) / den if den > 0 else np.nan
    est = sl(S.sum(axis=0))
    G = len(S)
    cnt = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)
    bs = np.array([sl(c @ S) for c in cnt])
    return {"n": n, "slope": float(est), "se": float(np.nanstd(bs, ddof=1))}



# =====================================================================================
# canned pixels
# =====================================================================================
def canned_flag(d: pd.DataFrame, thresh: float = CANNED_SHARE) -> pd.Series:
    """Per season: a located attempt is `canned` if its (folded x, y) pixel holds more than
    `thresh` of that season's located attempts (all classes pooled)."""
    out = pd.Series(False, index=d.index)
    for s, g in d[d["located"]].groupby("season"):
        vc = g["pix"].value_counts()
        big = set(vc.index[vc / len(g) > thresh])
        out.loc[g.index] = g["pix"].isin(big).to_numpy()
    return out


def spike_flags(d: pd.DataFrame, ratio: float = 3.0, thresh: float = CANNED_SHARE) -> tuple[pd.Series, pd.Series]:
    """Per season: `spike` = a canned (>thresh) pixel whose count is >= `ratio` x the mean of its 8 grid
    neighbours (an isolated spike rather than a dense real region); `ring` = the 8 neighbours of a spike pixel."""
    spike = pd.Series(False, index=d.index)
    ring = pd.Series(False, index=d.index)
    for s, g in d[d["located"]].groupby("season"):
        ix = np.round(g["xf"] / 9.4).astype(int).clip(0, 50).to_numpy()
        iy = np.round(g["y"] / 5.0).astype(int).clip(0, 100).to_numpy()
        grid = np.zeros((51, 101))
        np.add.at(grid, (ix, iy), 1)
        pad = np.pad(grid, 1, constant_values=np.nan)
        nb = np.nanmean(np.stack([pad[1 + dx:52 + dx, 1 + dy:102 + dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                  if not (dx == 0 and dy == 0)]), axis=0)
        sp = (grid / len(g) > thresh) & (grid >= ratio * nb)
        rg = np.zeros_like(sp)
        for (i, j) in zip(*np.nonzero(sp)):
            rg[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = True
        rg &= ~sp
        spike.loc[g.index] = sp[ix, iy]
        ring.loc[g.index] = rg[ix, iy]
    return spike, ring


def with_flags(D: pd.DataFrame) -> pd.DataFrame:
    D["canned"] = canned_flag(D)
    D["spike"], D["ring"] = spike_flags(D)
    D["dh"] = np.sqrt(D["hry"] ** 2 + D["hy"] ** 2)       # hoopR distance from basket (raw frame is basket-relative)
    return D


# =====================================================================================
# PART 1
# =====================================================================================
def hoopr_shots_check(season: int) -> dict:
    sh = pd.read_parquet(HOOPR / f"shots/shots_{season}.parquet")
    pb = pd.read_parquet(HOOPR / f"pbp/play_by_play_{season}.parquet")
    pb = pb[pb["shooting_play"].fillna(False).astype(bool)]
    cols = ["game_id", "period_number", "clock_display_value", "team_id", "athlete_id_1", "type_text",
            "scoring_play", "score_value", "coordinate_x", "coordinate_y", "coordinate_x_raw", "coordinate_y_raw"]
    def norm(d):
        n = d[cols].copy()
        for c in cols:
            if c not in ("clock_display_value", "type_text"):
                n[c] = pd.to_numeric(n[c].astype(int) if n[c].dtype == bool else n[c], errors="coerce").astype(float).round(3)
        return n.astype(str)
    ha = pd.util.hash_pandas_object(norm(sh), index=False).to_numpy()
    hb = pd.util.hash_pandas_object(norm(pb), index=False).to_numpy()
    sa, sb = np.sort(ha), np.sort(hb)
    same = bool(len(sa) == len(sb) and (sa == sb).all())
    ca = pd.to_numeric(sh["coordinate_x"], errors="coerce")
    return {"season": season, "shots_rows": int(len(sh)), "pbp_shooting_play_rows": int(len(pb)),
            "multiset_identical_on_12_cols": same,
            "shots_rows_in_pbp_share": round(float(np.isin(ha, hb).mean()), 5),
            "shots_coordinate_x_nonnull_pct": round(100 * float(ca.notna().mean()), 2),
            "shots_coordinate_x_sentinel_pct": round(100 * float((ca.abs() > 1e6).mean()), 3),
            "shots_has_play_id": "id" in sh.columns,
            "shots_columns": list(sh.columns)}


def part1(a) -> None:
    out: dict = {}
    md: list[str] = []
    D = with_flags(load_master())
    # ---------------- (a) hoopR pbp frame and coverage
    rows, frame_rows, orient = [], [], []
    for s in ALL_SEASONS:
        h = _hoopr_pbp(s)
        fg = h[h["type_text"].isin(["JumpShot", "LayUpShot", "DunkShot", "TipShot"])]
        loc = fg["coordinate_x"].notna() & fg["coordinate_y"].notna()
        sent = fg["coordinate_x_sentinel"] | fg["coordinate_y_sentinel"]
        rows.append({"season": s, "fga_rows": len(fg), "located_pct": round(100 * loc.mean(), 2),
                     "int32_sentinel_rows": int(sent.sum()),
                     "null_pct": round(100 * float((~loc & ~sent).mean()), 2)})
        L = fg[loc]
        dunk = L[L["type_text"] == "DunkShot"]
        frame_rows.append({"season": s, "x_min": L["coordinate_x"].min(), "x_max": L["coordinate_x"].max(),
                           "y_min": L["coordinate_y"].min(), "y_max": L["coordinate_y"].max(),
                           "raw_x_min": L["coordinate_x_raw"].min(), "raw_x_max": L["coordinate_x_raw"].max(),
                           "raw_y_min": L["coordinate_y_raw"].min(), "raw_y_max": L["coordinate_y_raw"].max(),
                           "dunk_median_abs_x": float(dunk["coordinate_x"].abs().median()),
                           "dunk_median_abs_y": float(dunk["coordinate_y"].abs().median()),
                           "distinct_x": int(L["coordinate_x"].nunique()), "distinct_y": int(L["coordinate_y"].nunique())})
        L = L.assign(pos=(L["coordinate_x"] > 0))
        e = L.groupby(["game_id", "team_id", "period_number"])["pos"].agg(["mean", "size"])
        e = e[(e["size"] >= 5) & e.index.get_level_values(2).isin([1, 2])]
        pv = e["mean"].unstack("period_number")
        sw = np.nan
        if {1, 2} <= set(pv.columns):
            pv = pv.dropna()
            sw = float((((pv[1] - .5) * (pv[2] - .5)) < 0).mean()) if len(pv) else np.nan
        hm = D[(D["season"] == s) & D["h_located"]]
        side = ((hm["hx"] > 0).to_numpy() == hm["isHome"].to_numpy())
        orient.append({"season": s, "team_periods": int(len(e)),
                       "x_sign_matches_home_flag_pct": round(100 * float(max(side.mean(), 1 - side.mean())), 2),
                       "share_team_periods_one_side_ge95pct": round(float(((e["mean"] >= .95) | (e["mean"] <= .05)).mean()), 4),
                       "share_team_games_switch_side_at_half": round(sw, 4)})
    A1 = pd.DataFrame(rows)
    F1 = pd.DataFrame(frame_rows)
    O1 = pd.DataFrame(orient)
    out["1a_hoopr_pbp_coverage"] = recs(A1)
    out["1a_hoopr_pbp_frame"] = recs(F1)
    out["1a_hoopr_pbp_switching"] = recs(O1)
    crow = []
    for s in ALL_SEASONS:
        g = D[(D["season"] == s) & D["located"]]
        dk = g[g["playType"] == "DunkShot"]
        crow.append({"season": s, "x_min": g["x"].min(), "x_max": g["x"].max(), "y_min": g["y"].min(),
                     "y_max": g["y"].max(), "distinct_x": int(g["x"].nunique()), "distinct_y": int(g["y"].nunique()),
                     "x_on_9.4_grid_pct": round(100 * float((np.abs(g["x"] / 9.4 - np.round(g["x"] / 9.4)) < 0.011).mean()), 2),
                     "y_on_5_grid_pct": round(100 * float((g["y"] % 5 == 0).mean()), 2),
                     "dunk_median_folded_x": float(dk["xf"].median()), "dunk_median_y": float(dk["y"].median()),
                     "share_x_left_half": round(float((g["x"] < HALF_X).mean()), 3)})
    out["1c_cbbd_frame"] = crow
    # ---------------- (b) hoopR shots_YYYY
    out["1b_shots_file"] = [hoopr_shots_check(s) for s in ALL_SEASONS]
    man = json.loads((HOOPR / "manifest.json").read_text())
    out["1b_shots_manifest"] = {k: {"url": v["url"], "downloaded_at": v["downloaded_at"], "bytes": v["bytes"]}
                                for k, v in man.items() if k.startswith("shots/")}
    log("part 1a/1b done")
    # ---------------- (c) CBBD coverage
    cov = D.groupby("season").agg(fga=("located", "size"), located=("located", "sum"),
                                  hoopr_joined=("in_hoopr", "sum"), hoopr_located=("h_located", "sum")).reset_index()
    cov["cbbd_located_pct"] = 100 * cov["located"] / cov["fga"]
    cov["hoopr_joined_pct"] = 100 * cov["hoopr_joined"] / cov["fga"]
    cov["hoopr_located_pct"] = 100 * cov["hoopr_located"] / cov["fga"]
    out["1c_cbbd_coverage"] = recs(cov)
    cc = D.groupby(["season", "cls"]).agg(n=("located", "size"), located_pct=("located", lambda v: 100 * v.mean())).reset_index()
    out["1c_cbbd_coverage_by_class"] = recs(cc)
    # ---------------- (d) reconciliation
    rec, acc, dis = [], [], []
    for s, g in D.groupby("season"):
        c, h = g["located"], g["h_located"]
        both = g[c & h]
        e = np.sqrt(((both["xf"] / 10) - (47 - both["hx"].abs())) ** 2
                    + (((both["y"] / 10 - 25).abs()) - both["hy"].abs()) ** 2)
        dd = (both["dist"] - both["dh"]).abs()
        pc = lambda m: round(100 * float(np.mean(m)), 2) if len(both) else np.nan
        rec.append({
            "season": s, "attempts": len(g), "cbbd_located": int(c.sum()), "hoopr_located": int(h.sum()),
            "both": int((c & h).sum()), "cbbd_only": int((c & ~h).sum()), "hoopr_only": int((~c & h).sum()),
            "neither": int((~c & ~h).sum()), "cbbd_pct": round(100 * c.mean(), 2), "hoopr_pct": round(100 * h.mean(), 2),
            "union_pct": round(100 * (c | h).mean(), 2),
            "pair_exact_pct": pc(e <= 0.011), "pair_le0.5ft_pct": pc(e <= 0.5), "pair_le1ft_pct": pc(e <= 1.0),
            "pair_1_to_3ft_pct": pc((e > 1) & (e <= 3)), "pair_gt3ft_pct": pc(e > 3), "pair_gt10ft_pct": pc(e > 10),
            "pair_dist_corr": round(float(np.corrcoef(both["dist"], both["dh"])[0, 1]), 3) if len(both) > 10 else np.nan,
            "pair_median_abs_dist_diff_ft": round(float(dd.median()), 3) if len(both) else np.nan,
            "cbbd_rows_without_hoopr_row": int((~g["in_hoopr"]).sum())})
        subs = [("all", both)] + ([("makes", both[both["made"] == 1]), ("misses", both[both["made"] == 0])] if s in OUT_SEASONS else [])
        for sub, bb in subs:
            for src, col in (("cbbd", "dist"), ("hoopr", "dh")):
                t3, tr, tj = (bb[bb["text_cls"] == k] for k in ("three", "rimtext", "twojump"))
                acc.append({"season": s, "subset": sub, "source": src, "n_three": len(t3), "n_rimtext": len(tr), "n_twojump": len(tj),
                            "three_inside_arc_pct": round(100 * float((t3[col] < ARC_FT).mean()), 2) if len(t3) else np.nan,
                            "twojump_outside_arc_pct": round(100 * float((tj[col] > ARC_FT).mean()), 2) if len(tj) else np.nan,
                            "rimtext_gt6ft_pct": round(100 * float((tr[col] > 6).mean()), 2) if len(tr) else np.nan})
        for flag in (True, False):
            m = (both["canned"] == flag).to_numpy()
            dis.append({"season": s, "cbbd_pixel_canned": flag, "n_pairs": int(m.sum()),
                        "pair_le1ft_pct": round(100 * float((e[m] <= 1).mean()), 2) if m.sum() else np.nan,
                        "pair_gt3ft_pct": round(100 * float((e[m] > 3).mean()), 2) if m.sum() else np.nan})
    R = pd.DataFrame(rec)
    out["1d_reconciliation"] = recs(R)
    out["1d_text_accuracy_on_shared_attempts"] = acc
    out["1d_disagreement_by_cbbd_canned"] = dis
    gl = D.groupby(["season", "gameId"]).agg(h=("h_located", "mean"), c=("located", "mean")).reset_index()
    sel = pd.DataFrame([{"season": s, "games": len(g), "hoopr_games_zero_pct": 100 * (g["h"] == 0).mean(),
                         "hoopr_games_ge90_pct": 100 * (g["h"] >= .9).mean(), "cbbd_games_zero_pct": 100 * (g["c"] == 0).mean(),
                         "cbbd_games_ge90_pct": 100 * (g["c"] >= .9).mean(),
                         "corr_game_cov_hoopr_vs_cbbd": float(np.corrcoef(g["h"], g["c"])[0, 1])}
                        for s, g in gl.groupby("season")])
    out["1d_game_coverage_hoopr_vs_cbbd"] = recs(sel)
    md += ["### 1a. hoopR pbp coordinate frame and FGA coverage\n", md_table(A1.merge(O1, on="season")), md_table(F1),
           "### 1b. hoopR shots_YYYY.parquet vs pbp shooting rows\n",
           md_table(pd.DataFrame(out["1b_shots_file"]).drop(columns=["shots_columns"])),
           "### 1c. CBBD plays: coverage and frame\n",
           md_table(cov[["season", "fga", "cbbd_located_pct", "hoopr_joined_pct", "hoopr_located_pct"]]),
           md_table(pd.DataFrame(crow)),
           "### 1d. Reconciliation (CBBD attempts are the universe; join key hoopR `id` == CBBD `sourceId`)\n",
           md_table(R), "Text-consistency of each source on the SAME attempts (both located)\n",
           md_table(pd.DataFrame(acc)), "Pair disagreement split by whether the CBBD pixel is canned\n",
           md_table(pd.DataFrame(dis)), "Game-level coverage, hoopR vs CBBD\n", md_table(sel)]
    jdump(out, RES / "part1.json")
    (RES / "part1.md").write_text("\n".join(md), encoding="utf-8")
    log("part 1 written")


# =====================================================================================
# PART 2: coverage structure
# =====================================================================================
def conf_tier(conf: str, season: int) -> str:
    if conf in POWER or conf in POWER_EXTRA:
        return "power"
    if conf in MID:
        return "mid"
    return "low"


def game_attrs() -> pd.DataFrame:
    gs, ls = [], []
    for s in ALL_SEASONS:
        g = pd.read_parquet(CBBD / f"games_{s}.parquet", columns=["id", "season", "seasonType", "neutralSite", "homeConference",
                                                                  "awayConference", "conferenceGame", "startDate"])
        g["has_line"] = g["id"].isin(set(pd.read_parquet(CBBD / f"lines_{s}.parquet", columns=["gameId", "spread", "overUnder"])
                                         .dropna(subset=["spread", "overUnder"], how="all")["gameId"]))
        g["tier"] = [conf_tier(c, s) for c in g["homeConference"].fillna("none")]
        gs.append(g)
    G = pd.concat(gs, ignore_index=True).rename(columns={"id": "gameId"})
    G["neutralSite"] = G["neutralSite"].fillna(False).astype(bool)
    return G[["gameId", "seasonType", "neutralSite", "homeConference", "conferenceGame", "has_line", "tier"]]


def cov_by(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    gg = d.groupby(by + ["gameId"]).agg(n=("located", "size"), c=("located", "sum"), u=("union", "sum")).reset_index()
    out = gg.groupby(by).apply(lambda g: pd.Series({
        "games": len(g), "attempts": int(g["n"].sum()),
        "cbbd_located_pct": 100 * g["c"].sum() / g["n"].sum(),
        "union_located_pct": 100 * g["u"].sum() / g["n"].sum(),
        "games_0pct": 100 * (g["c"] == 0).mean(),
        "games_1_99pct": 100 * ((g["c"] > 0) & (g["c"] < g["n"])).mean(),
        "games_100pct": 100 * (g["c"] == g["n"]).mean()}), include_groups=False).reset_index()
    return out


def part2(a) -> None:
    rng = np.random.default_rng(20261027)
    D = with_flags(load_master())
    D["union"] = D["located"] | D["h_located"]
    D = D.merge(game_attrs(), on="gameId", how="left")
    out, md = {}, []
    # ---- 2a: per-game structure
    gg = D.groupby(["season", "gameId"]).agg(n=("located", "size"), c=("located", "sum")).reset_index()
    gg["frac"] = gg["c"] / gg["n"]
    rows = []
    for s, g in gg.groupby("season"):
        p = g["c"].sum() / g["n"].sum()
        nbar = g["n"].mean()
        exp_var = p * (1 - p) / nbar                  # per-game coverage variance if each attempt were located independently
        rows.append({"season": s, "games": len(g), "attempt_cov_pct": 100 * p, "games_0pct": 100 * (g["c"] == 0).mean(),
                     "games_1_99pct": 100 * ((g["c"] > 0) & (g["c"] < g["n"])).mean(),
                     "games_100pct": 100 * (g["c"] == g["n"]).mean(),
                     "attempts_in_partial_games_pct": 100 * g.loc[(g["c"] > 0) & (g["c"] < g["n"]), "n"].sum() / g["n"].sum(),
                     "obs_over_indep_variance_ratio": float(g["frac"].var() / exp_var)})
    T2a = pd.DataFrame(rows)
    out["2a_per_game_structure"] = recs(T2a)
    # partial games: what do the partial games look like (period / first-half)? located share by period
    per = D.assign(per=np.minimum(D["period"].astype(float), 3)).groupby(["season", "per"]).agg(n=("located", "size"), pct=("located", lambda v: 100 * v.mean())).reset_index()
    out["2a_located_by_period_(3=OT)"] = recs(per)
    # purity of units: share of attempts sitting in units that are 0% or 100% located, observed vs the
    # share expected if every attempt were located independently at the season rate
    D["per3"] = np.minimum(D["period"].astype(float), 3)
    D["blk"] = D["secondsRemaining"] // 300
    pur = []
    for s, d in D.groupby("season"):
        p = d["located"].mean()
        for nm, keys in (("game", ["gameId"]), ("game x period", ["gameId", "per3"]), ("game x team", ["gameId", "teamId"]),
                         ("game x team x period", ["gameId", "teamId", "per3"]),
                         ("game x period x 5-min block", ["gameId", "per3", "blk"]), ("game x shooter", ["gameId", "shot_shooter_id"])):
            u = d.groupby(keys)["located"].agg(["mean", "size"])
            obs = u.loc[(u["mean"] == 0) | (u["mean"] == 1), "size"].sum() / u["size"].sum()
            exp = (u["size"] * (p ** u["size"] + (1 - p) ** u["size"])).sum() / u["size"].sum()
            pur.append({"season": s, "unit": nm, "units": len(u), "mean_attempts_per_unit": float(u["size"].mean()),
                        "pure_share_observed_pct": 100 * obs, "pure_share_if_independent_pct": 100 * exp})
    T2p = pd.DataFrame(pur)
    out["2a_unit_purity"] = recs(T2p)
    md += ["### 2a. Is missingness decided per game?\n", md_table(T2a),
           "Share of attempts that sit in a unit located 0% or 100% (observed vs expected if each attempt were located independently at the season rate)\n",
           md_table(T2p)]
    # ---- 2b
    base = D.assign(all="all")
    tabs = {}
    tabs["by_season"] = cov_by(base, ["season"])
    tabs["by_season_month"] = cov_by(D, ["season", "month"])
    tabs["by_season_tier"] = cov_by(D, ["season", "tier"])
    tabs["by_season_neutral"] = cov_by(D, ["season", "neutralSite"])
    tabs["by_season_has_line"] = cov_by(D, ["season", "has_line"])
    tabs["by_season_seasontype"] = cov_by(D, ["season", "seasonType"])
    tabs["by_season_conference_game"] = cov_by(D.assign(conferenceGame=D["conferenceGame"].fillna(False).astype(bool)), ["season", "conferenceGame"])
    tabs["by_season_tier_has_line"] = cov_by(D, ["season", "tier", "has_line"])
    for k, v in tabs.items():
        out["2b_" + k] = recs(v)
        md += [f"### 2b. coverage {k}\n", md_table(v)]
    # missing-game profile: composition of games with 0% located vs the rest (is the missing set a biased subset?)
    prof = []
    gm = D.groupby(["season", "gameId"]).agg(c=("located", "sum"), n=("located", "size"), tier=("tier", "first"),
                                              neutral=("neutralSite", "first"), has_line=("has_line", "first"),
                                              month=("month", "first")).reset_index()
    gm["group"] = np.where(gm["c"] == 0, "zero", np.where(gm["c"] == gm["n"], "full", "partial"))
    for s, g in gm.groupby("season"):
        for grp, h in g.groupby("group"):
            prof.append({"season": s, "group": grp, "games": len(h), "power_pct": 100 * (h["tier"] == "power").mean(),
                         "mid_pct": 100 * (h["tier"] == "mid").mean(), "low_pct": 100 * (h["tier"] == "low").mean(),
                         "neutral_pct": 100 * h["neutral"].mean(), "has_line_pct": 100 * h["has_line"].fillna(False).astype(bool).mean(),
                         "march_apr_pct": 100 * h["month"].isin([3, 4]).mean(), "mean_attempts": h["n"].mean()})
    P = pd.DataFrame(prof)
    out["2b_missing_game_profile"] = recs(P)
    md += ["### 2b. Composition of games by located status (0% / partial / full)\n", md_table(P)]
    # ---- 2c: per-team counts
    rows = []
    for s, g in D.groupby("season"):
        off = g.groupby("teamId").agg(n=("located", "size"), loc=("located", "sum"))
        dfn = g.groupby("opponentId").agg(n=("located", "size"), loc=("located", "sum"))
        for side, t in (("offence", off), ("defence", dfn)):
            t = t[t["n"] >= 500]
            pct = 100 * t["loc"] / t["n"]
            rows.append({"season": s, "side": side, "teams": len(t), "located_attempts_min": int(t["loc"].min()),
                         "located_attempts_p10": float(t["loc"].quantile(.1)), "located_attempts_p50": float(t["loc"].median()),
                         "located_pct_min": float(pct.min()), "located_pct_p10": float(pct.quantile(.1)),
                         "located_pct_p50": float(pct.median()),
                         "teams_below_50pct": int((pct < 50).sum())})
    T2c = pd.DataFrame(rows)
    out["2c_per_team"] = recs(T2c)
    md += ["### 2c. Per-team located attempts per season (teams with >= 500 attempts)\n", md_table(T2c)]
    jdump(out, RES / "part2.json")
    (RES / "part2.md").write_text("\n".join(md), encoding="utf-8")
    log("part 2 written")


# =====================================================================================
# PART 3: placement audit
# =====================================================================================
def up(n) -> str:
    return "UNDERPOWERED" if n < MIN_N else ""


def pdiff_row(tag: dict, r: dict) -> dict:
    row = dict(tag)
    row.update({"n_a": r["n_a"], "n_b": r["n_b"], "rate_a": r["rate_a"], "rate_b": r["rate_b"],
                "pooled_diff": r["pooled_diff"], "games_paired": r["n_games_paired"],
                "within_game_diff": r["within_diff"], "boot_se": r["boot_se"], "z": r["z"],
                "flag": "UNDERPOWERED" if r["underpowered"] else ""})
    return row


def pixel_tables(D: pd.DataFrame) -> tuple[dict, list]:
    out, md = {}, []
    L = D[D["located"]]
    top_rows, dist_rows, thr_rows = [], [], []
    for s, g in L.groupby("season"):
        vc = g["pix"].value_counts()
        n = len(g)
        share = vc / n
        ix = np.round(g["xf"] / 9.4).astype(int).clip(0, 50)
        iy = np.round(g["y"] / 5.0).astype(int).clip(0, 100)
        grid = np.zeros((51, 101))
        np.add.at(grid, (ix.to_numpy(), iy.to_numpy()), 1)
        pad = np.pad(grid, 1, constant_values=np.nan)
        nb = np.nanmean(np.stack([pad[1 + dx:52 + dx, 1 + dy:102 + dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                  if not (dx == 0 and dy == 0)]), axis=0)
        for rank, (pix, c) in enumerate(vc.head(20).items(), 1):
            h = g[g["pix"] == pix]
            gi, gj = int(np.round(h["xf"].iloc[0] / 9.4)), int(np.round(h["y"].iloc[0] / 5.0))
            row = {"neighbour_ratio": float(grid[gi, gj] / nb[gi, gj]), "season": s, "rank": rank, "x_ft_folded": float(h["xf"].iloc[0] / 10), "y_ft": float(h["y"].iloc[0] / 10),
                   "dist_from_basket_ft": float(h["dist"].iloc[0]), "n": int(c), "share_of_located_pct": 100 * c / n,
                   "pct_rim": 100 * float((h["cls"] == "rim").mean()), "pct_jump2": 100 * float((h["cls"] == "jump2").mean()),
                   "pct_three": 100 * float((h["cls"] == "three").mean()),
                   "pct_dunk_playtype": 100 * float((h["playType"] == "DunkShot").mean())}
            if s in OUT_SEASONS:
                row["make_rate"] = float(h["made"].mean())
            top_rows.append(row)
        with np.errstate(divide="ignore", invalid="ignore"):
            spike = np.where(nb > 0, grid / nb, np.nan)
        flat_share = grid.ravel() / n
        big = flat_share > CANNED_SHARE
        dist_rows.append({
            "season": s, "located": n, "distinct_pixels": int((grid > 0).sum()),
            "uniform_pixel_share_pct": 100 / int((grid > 0).sum()),
            "pixel_share_p50_pct": 100 * float(np.quantile(flat_share[grid.ravel() > 0], .5)),
            "pixel_share_p99_pct": 100 * float(np.quantile(flat_share[grid.ravel() > 0], .99)),
            "pixel_share_p999_pct": 100 * float(np.quantile(flat_share[grid.ravel() > 0], .999)),
            "max_pixel_share_pct": 100 * float(flat_share.max()),
            "pixels_over_0.5pct": int(big.sum()), "attempts_on_canned_pct": 100 * float(flat_share[big].sum()),
            "median_neighbour_spike_ratio_canned": float(np.nanmedian(spike.ravel()[big])) if big.any() else np.nan,
            "max_neighbour_ratio_noncanned(n>=20)": float(np.nanmax(np.where(big | (grid.ravel() < 20), np.nan, spike.ravel()))),
            "p99_neighbour_ratio_noncanned(n>=20)": float(np.nanquantile(np.where(big | (grid.ravel() < 20), np.nan, spike.ravel()), .99)),
            "pixels_canned_with_ratio_ge3": int((big & (spike.ravel() >= 3)).sum())})
        for th in (0.0025, 0.005, 0.01):
            b2 = flat_share > th
            thr_rows.append({"season": s, "threshold_pct": 100 * th, "pixels": int(b2.sum()),
                             "attempts_on_them_pct": 100 * float(flat_share[b2].sum())})
    out["3a_top20_pixels"] = top_rows
    out["3a_pixel_distribution"] = dist_rows
    out["3a_threshold_sensitivity"] = thr_rows
    md += ["### 3a. Pixel-frequency distribution (threshold justification)\n", md_table(pd.DataFrame(dist_rows)),
           "Sensitivity of the canned set to the threshold\n", md_table(pd.DataFrame(thr_rows)),
           "### 3a. Top 20 pixels per season (folded frame, feet from baseline / from sideline)\n"]
    for s in ALL_SEASONS:
        md += [f"**Season {s}**\n", md_table(pd.DataFrame([r for r in top_rows if r["season"] == s]).drop(columns=["season"]))]
    return out, md


def part3(a) -> None:
    rng = np.random.default_rng(20261027)
    assert_not_sealed(OUT_SEASONS, context="coordinate audit outcome tables")
    D = with_flags(load_master())
    D = D.merge(game_attrs(), on="gameId", how="left")
    out, md = {}, []
    o, m = pixel_tables(D)
    out.update(o)
    md += m
    DO = D[D["season"].isin(OUT_SEASONS)].copy()                    # outcome-bearing seasons only
    LO = DO[DO["located"]]
    # ------------------------------------------------------------------ 3a: canned make rates
    rows, rows_cls = [], []
    for s in OUT_SEASONS:
        for c in CLASSES:
            g = LO[(LO["season"] == s) & (LO["cls"] == c)]
            r = paired_diff(g, g["canned"], ~g["canned"], g["made"], rng)
            rows.append(pdiff_row({"season": s, "class": c, "a": "canned", "b": "non-canned"}, r))
            rows_cls.append({"season": s, "class": c, "located": len(g), "canned_share_pct": 100 * float(g["canned"].mean())})
    T3a = pd.DataFrame(rows)
    out["3a_make_canned_vs_not"] = recs(T3a)
    out["3a_canned_share_by_class"] = rows_cls
    # spike pixel vs its 8-neighbour ring (same place on the floor, the cleanest like-for-like)
    rows = []
    for s in OUT_SEASONS:
        for c in CLASSES:
            g = LO[(LO["season"] == s) & (LO["cls"] == c) & (LO["spike"] | LO["ring"])]
            if len(g) == 0:
                continue
            r = paired_diff(g, g["spike"], g["ring"], g["made"], rng)
            rows.append(pdiff_row({"season": s, "class": c, "a": "spike pixel", "b": "its 8-neighbour ring"}, r))
    T3as = pd.DataFrame(rows)
    out["3a_spike_vs_ring"] = recs(T3as)
    # distance-matched canned (>0.5%) vs non-canned, rim class
    rows = []
    edges = [0, .5, 1, 1.5, 2, 2.5, 3, 4, 6]
    for s in OUT_SEASONS:
        g = LO[(LO["season"] == s) & (LO["cls"] == "rim")].copy()
        g["db"] = pd.cut(g["dist"], edges, right=False)
        num = den = 0.0
        var = 0.0
        for db, h in g.groupby("db", observed=True):
            c_, n_ = h[h["canned"]], h[~h["canned"]]
            if len(c_) and len(n_):
                rows.append({"season": s, "dist_bin_ft": str(db), "n_canned": len(c_), "n_non_canned": len(n_),
                             "make_canned": float(c_["made"].mean()), "make_non_canned": float(n_["made"].mean()),
                             "diff": float(c_["made"].mean() - n_["made"].mean()),
                             "flag": "UNDERPOWERED" if min(len(c_), len(n_)) < MIN_N else ""})
                if min(len(c_), len(n_)) >= MIN_N:
                    w = len(c_)
                    num += w * (c_["made"].mean() - n_["made"].mean())
                    den += w
                    var += w * w * (c_["made"].var() / len(c_) + n_["made"].var() / len(n_))
        rows.append({"season": s, "dist_bin_ft": "STANDARDISED (weights = canned n, bins >= 500 both)", "n_canned": int(den),
                     "n_non_canned": np.nan, "make_canned": np.nan, "make_non_canned": np.nan,
                     "diff": num / den if den else np.nan, "flag": f"binomial SE {np.sqrt(var) / den:.4f}" if den else ""})
    T3ad = pd.DataFrame(rows)
    out["3a_distance_matched_rim"] = recs(T3ad)
    # does the within-game rim result move with the canned threshold?
    rows = []
    for th in (0.0025, 0.005, 0.01):
        cf = canned_flag(D, th)
        for s in OUT_SEASONS:
            g = LO[(LO["season"] == s) & (LO["cls"] == "rim")]
            cc = cf.loc[g.index]
            r = paired_diff(g, cc, ~cc, g["made"], rng)
            rows.append(pdiff_row({"season": s, "threshold_pct": 100 * th, "class": "rim"}, r))
    T3at = pd.DataFrame(rows)
    out["3a_threshold_sensitivity_rim"] = recs(T3at)
    md += ["### 3a. Make rate on canned vs non-canned pixels (a = canned, b = non-canned). within_game_diff is the paired, within-game estimate; boot_se is the game bootstrap floor\n",
           md_table(T3a), "Canned share of located attempts by class\n", md_table(pd.DataFrame(rows_cls)),
           "Spike pixel (canned AND >= 3x its 8 neighbours) vs the ring of its 8 neighbours, same class, within game\n", md_table(T3as),
           "Distance-matched: rim attempts, canned vs non-canned within 0.5 ft distance-from-basket bins\n", md_table(T3ad),
           "Rim canned-vs-non-canned within-game gap under other canned thresholds\n", md_table(T3at)]
    # ------------------------------------------------------------------ 3b: outcome-dependent charting
    # (i) located share, makes vs misses, within game and class (feed class so the location-based rim override cannot leak in)
    rows, rows_p = [], []
    gcov = DO.groupby("gameId")["located"].mean()
    DO["game_cov"] = DO["gameId"].map(gcov)
    for s in OUT_SEASONS:
        for c in CLASSES:
            g = DO[(DO["season"] == s) & (DO["cls0"] == c)]
            r = paired_diff(g, g["made"] == 1, g["made"] == 0, g["located"].astype(float), rng)
            rows.append(pdiff_row({"season": s, "class(feed)": c, "a": "makes", "b": "misses", "metric": "located share"}, r))
            gp = g[(g["game_cov"] > 0) & (g["game_cov"] < 1)]
            r = paired_diff(gp, gp["made"] == 1, gp["made"] == 0, gp["located"].astype(float), rng)
            rows_p.append(pdiff_row({"season": s, "class(feed)": c, "a": "makes", "b": "misses", "metric": "located share (partial games only)"}, r))
    T3b1, T3b1p = pd.DataFrame(rows), pd.DataFrame(rows_p)
    out["3b_located_share_make_vs_miss"] = recs(T3b1)
    out["3b_located_share_make_vs_miss_partial_games"] = recs(T3b1p)
    md += ["### 3b(i). Located share, MAKES vs MISSES (outcome-blind charting => equal within the floor)\n", md_table(T3b1),
           "Same, restricted to games with partial coverage (the only games that can show the effect)\n", md_table(T3b1p)]
    # (ii) make rate, located vs unlocated (build class reproduces the round-1 numbers; feed class is like-for-like)
    rows = []
    for s in OUT_SEASONS:
        for cname in ("cls", "cls0"):
            for c in CLASSES:
                g = DO[(DO["season"] == s) & (DO[cname] == c)]
                r = paired_diff(g, g["located"], ~g["located"], g["made"], rng)
                rows.append(pdiff_row({"season": s, "class": c, "class_def": "build(rim override)" if cname == "cls" else "feed", "a": "located", "b": "unlocated"}, r))
    T3b2 = pd.DataFrame(rows)
    out["3b_make_rate_located_vs_unlocated"] = recs(T3b2)
    md += ["### 3b(ii). Make rate, LOCATED (a) vs UNLOCATED (b)\n", md_table(T3b2)]
    # explanation of the 2025 rim gap
    ex = {}
    for s in OUT_SEASONS:
        g = DO[(DO["season"] == s) & (DO["cls"] == "rim")]
        u, l = g[~g["located"]], g[g["located"]]
        mix = []
        for pt, h in g.groupby("playType"):
            mix.append({"season": s, "playType": pt, "unlocated_n": int((~h["located"]).sum()), "located_n": int(h["located"].sum()),
                        "unlocated_make": float(h.loc[~h["located"], "made"].mean()) if (~h["located"]).sum() else np.nan,
                        "located_make": float(h.loc[h["located"], "made"].mean()) if h["located"].sum() else np.nan})
        ex.setdefault("by_playtype", []).extend(mix)
        # mix-standardised located make rate to the unlocated playType mix
        umix = u["playType"].value_counts(normalize=True)
        lm = l.groupby("playType")["made"].mean()
        ex.setdefault("mix_adjusted", []).append({
            "season": s, "unlocated_n": len(u), "unlocated_make": float(u["made"].mean()), "located_make": float(l["made"].mean()),
            "located_make_at_unlocated_playtype_mix": float((umix * lm.reindex(umix.index)).sum()),
            "located_close_jumper_share_pct": 100 * float((l["cls0"] != "rim").mean())})
        # concentration
        per_game = u.groupby("gameId").agg(n=("made", "size"), m=("made", "mean"))
        top = per_game.sort_values("n", ascending=False)
        t10 = top.head(10)
        rest = u[~u["gameId"].isin(t10.index)]
        per_team = u.groupby("teamId")["made"].agg(["size", "mean"]).sort_values("size", ascending=False)
        per_month = u.groupby("month")["made"].agg(["size", "mean"])
        ex.setdefault("concentration", []).append({
            "season": s, "unlocated_rim": len(u), "games_with_unlocated_rim": int(len(per_game)),
            "games_in_season": int(g["gameId"].nunique()),
            "top10_games_share_pct": 100 * float(t10["n"].sum() / len(u)),
            "unlocated_make_excl_top10_games": float(rest["made"].mean()) if len(rest) else np.nan,
            "n_excl_top10": len(rest),
            "teams_with_unlocated_rim": int(len(per_team)), "top10_teams_share_pct": 100 * float(per_team["size"].head(10).sum() / len(u)),
            "mean_make_rate_of_games_with_unlocated_rim_all_attempts": float(g[g["gameId"].isin(per_game.index)]["made"].mean()),
            "make_rate_games_without_unlocated_rim": float(g[~g["gameId"].isin(per_game.index) & g["located"]]["made"].mean())})
        for mo, r_ in per_month.iterrows():
            ex.setdefault("by_month", []).append({"season": s, "month": int(mo), "unlocated_n": int(r_["size"]), "unlocated_make": float(r_["mean"]),
                                                  "located_n": int(((l["month"] == mo)).sum()),
                                                  "located_make": float(l.loc[l["month"] == mo, "made"].mean())})
        for per, h_ in g.assign(per=np.minimum(g["period"].astype(float), 3)).groupby("per"):
            ex.setdefault("by_period(3=OT)", []).append({"season": s, "period": int(per), "unlocated_n": int((~h_["located"]).sum()),
                                                  "unlocated_make": float(h_.loc[~h_["located"], "made"].mean()) if (~h_["located"]).sum() else np.nan,
                                                  "located_n": int(h_["located"].sum()), "located_make": float(h_.loc[h_["located"], "made"].mean())})
        # unlocated rim attempts: same-game located attempts on the same offence team (within-team contrast)
        ex.setdefault("by_tier", []).extend([
            {"season": s, "tier": t, "unlocated_n": int((u["tier"] == t).sum()), "unlocated_make": float(u.loc[u["tier"] == t, "made"].mean()) if (u["tier"] == t).sum() else np.nan,
             "located_n": int((l["tier"] == t).sum()), "located_make": float(l.loc[l["tier"] == t, "made"].mean())} for t in ("power", "mid", "low")])
    for k, v in ex.items():
        out["3b_explain_rim_gap_" + k] = v
        md += [f"### 3b(ii). Explaining the rim located-vs-unlocated make-rate gap: {k}\n", md_table(pd.DataFrame(v))]
    # ------------------------------------------------------------------ 3c: canned response to team skill
    hh = pd.concat([pd.read_parquet(HOOPR / f"pbp/play_by_play_{s}.parquet", columns=["game_id", "home_team_id", "away_team_id"])
                    .drop_duplicates("game_id") for s in OUT_SEASONS], ignore_index=True)
    hh["espn_game"] = hh["game_id"].astype("int64")
    tr = pd.read_parquet(PAE_DIR / "team_ratings_v1.parquet", columns=["game_id", "team_id", "side", "rating", "N"])
    tr = tr[tr["side"] == "off"].rename(columns={"game_id": "espn_game", "team_id": "espn_team", "rating": "pae_off_c", "N": "pae_off_n"})[
        ["espn_game", "espn_team", "pae_off_c", "pae_off_n"]]
    LR = LO.copy()
    LR["espn_game"] = pd.to_numeric(LR["gameSourceId"], errors="coerce").astype("Int64").astype("float").astype("int64")
    LR = LR.merge(hh[["espn_game", "home_team_id", "away_team_id"]], on="espn_game", how="left")
    LR["espn_team"] = np.where(LR["isHome"], LR["home_team_id"], LR["away_team_id"]).astype("float")
    LR["espn_team"] = LR["espn_team"].astype("Int64")
    tr["espn_team"] = tr["espn_team"].astype("Int64")
    LR = LR.merge(tr, on=["espn_game", "espn_team"], how="left")
    out["3c_rating_join"] = {"located_attempts": len(LR), "with_rating_pct": 100 * float(LR["pae_off_c"].notna().mean()),
                             "with_prior_attempts_pct": 100 * float((LR["pae_off_n"] > 0).mean())}
    LR = LR[LR["pae_off_n"] > 0]
    qrows, srows = [], []
    for s in OUT_SEASONS:
        for c in CLASSES:
            g = LR[(LR["season"] == s) & (LR["cls"] == c)].copy()
            if len(g) < 200:
                continue
            g["q"] = pd.qcut(g["pae_off_c"], 5, labels=False, duplicates="drop") + 1
            for grp, msk in (("spike pixel", g["spike"]), ("ring of spike pixel", g["ring"]), ("canned (>0.5%)", g["canned"]), ("non-canned", ~g["canned"]),
                             ("all located", g["canned"] | ~g["canned"])):
                h = g[msk]
                for q, hq in h.groupby("q"):
                    qrows.append({"season": s, "class": c, "group": grp, "quintile": int(q), "n": len(hq),
                                  "mean_rating": float(hq["pae_off_c"].mean()), "make_rate": float(hq["made"].mean()),
                                  "flag": up(len(hq))})
                sl = slope_stats(h["pae_off_c"].to_numpy(), h["made"].to_numpy(), h["gameId"].to_numpy(), rng)
                q1 = h[h["q"] == 1]["made"].mean() if (h["q"] == 1).any() else np.nan
                q5 = h[h["q"] == 5]["made"].mean() if (h["q"] == 5).any() else np.nan
                srows.append({"season": s, "class": c, "group": grp, "n": sl["n"], "slope_make_per_rating_pt": sl["slope"],
                              "boot_se": sl["se"], "make_Q5_minus_Q1": q5 - q1, "flag": up(sl["n"])})
    TQ, TS = pd.DataFrame(qrows), pd.DataFrame(srows)
    # slope difference canned - non-canned
    dd = []
    for (s, c), g in TS.groupby(["season", "class"]):
        gn = g[g["group"] == "non-canned"]
        for gname in ("canned (>0.5%)", "spike pixel", "ring of spike pixel"):
            gc = g[g["group"] == gname]
            if len(gc) and len(gn):
                d_ = gc["slope_make_per_rating_pt"].iloc[0] - gn["slope_make_per_rating_pt"].iloc[0]
                se = float(np.sqrt(gc["boot_se"].iloc[0] ** 2 + gn["boot_se"].iloc[0] ** 2))
                dd.append({"season": s, "class": c, "group": gname, "slope_group": gc["slope_make_per_rating_pt"].iloc[0],
                           "slope_reference": gn["slope_make_per_rating_pt"].iloc[0], "diff": d_, "se_diff_(indep. approx.)": se,
                           "z": d_ / se if se > 0 else np.nan, "n_group": int(gc["n"].iloc[0]), "flag": up(int(gc["n"].iloc[0]))})
    for (s, c), g in TS.groupby(["season", "class"]):
        gs_, gr_ = g[g["group"] == "spike pixel"], g[g["group"] == "ring of spike pixel"]
        if len(gs_) and len(gr_) and np.isfinite(gs_["slope_make_per_rating_pt"].iloc[0]) and np.isfinite(gr_["slope_make_per_rating_pt"].iloc[0]):
            d_ = gs_["slope_make_per_rating_pt"].iloc[0] - gr_["slope_make_per_rating_pt"].iloc[0]
            se = float(np.sqrt(gs_["boot_se"].iloc[0] ** 2 + gr_["boot_se"].iloc[0] ** 2))
            dd.append({"season": s, "class": c, "group": "spike pixel MINUS its ring", "slope_group": gs_["slope_make_per_rating_pt"].iloc[0],
                       "slope_reference": gr_["slope_make_per_rating_pt"].iloc[0], "diff": d_, "se_diff_(indep. approx.)": se,
                       "z": d_ / se if se > 0 else np.nan, "n_group": int(gs_["n"].iloc[0]), "flag": up(int(gs_["n"].iloc[0]))})
    TD = pd.DataFrame(dd)
    out["3c_quintiles"], out["3c_slopes"], out["3c_slope_diff"] = recs(TQ), recs(TS), recs(TD)
    md += ["### 3c. Make rate by as-of OFFENCE-PAE quintile, canned vs non-canned (located attempts, rated teams)\n",
           md_table(TQ[TQ["class"] == "rim"]), "Slopes (make probability per rating point; SE = game bootstrap)\n", md_table(TS),
           "Group minus non-canned slope (SE adds the two bootstrap SEs in quadrature; the groups share games, so this is approximate)\n", md_table(TD)]
    # ------------------------------------------------------------------ 3d: regime change
    LA = D[D["located"]].copy()                       # includes sealed 2026: structure only, no outcomes used below
    bins = [0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 20, 22, 24, 26, 30, 40, 100]
    LA["dbin"] = pd.cut(LA["dist"], bins, right=False).astype(str)
    reg = []
    for s, g in LA.groupby("season"):
        n = len(g)
        vc = g["pix"].value_counts()
        reg.append({"season": s, "located": n, "located_share_of_attempts_pct": 100 * n / int((D["season"] == s).sum()),
                    "canned_share_pct": 100 * float(g["canned"].mean()), "spike_pixel_share_pct": 100 * float(g["spike"].mean()),
                    "spike_share_of_rim_located_pct": 100 * float(g.loc[g["cls"] == "rim", "spike"].mean()), "top3_pixel_share_pct": 100 * float(vc.head(3).sum() / n),
                    "distinct_pixels_per_10k_located": 10000 * g["pix"].nunique() / n,
                    "within_1ft_of_basket_pct": 100 * float((g["dist"] < 1).mean()),
                    "rim_mean_dist_ft": float(g.loc[g["cls"] == "rim", "dist"].mean()),
                    "jump2_mean_dist_ft": float(g.loc[g["cls"] == "jump2", "dist"].mean()),
                    "three_mean_dist_ft": float(g.loc[g["cls"] == "three", "dist"].mean())})
    T3d = pd.DataFrame(reg)
    out["3d_by_season"] = recs(T3d)
    hist = LA.groupby(["season", "dbin"]).size().unstack("season").fillna(0)
    hist = hist.reindex([str(pd.Interval(bins[i], bins[i + 1], closed="left")) for i in range(len(bins) - 1)])
    hist_pct = (100 * hist / hist.sum()).round(2).reset_index().rename(columns={"dbin": "dist_bin_ft"})
    out["3d_distance_hist_pct"] = recs(hist_pct)
    # TV distance season-to-season vs split-half noise floor (random halves of games within a season)
    tvr = []
    prob = (hist / hist.sum())
    for s in ALL_SEASONS[1:]:
        tv = 0.5 * float((prob[s] - prob[s - 1]).abs().sum())
        gs = LA[LA["season"] == s]["gameId"].unique()
        fl = []
        for _ in range(30):
            pm = rng.permutation(gs)
            h1 = LA[(LA["season"] == s) & LA["gameId"].isin(pm[: len(pm) // 2])]["dbin"].value_counts(normalize=True)
            h2 = LA[(LA["season"] == s) & LA["gameId"].isin(pm[len(pm) // 2:])]["dbin"].value_counts(normalize=True)
            idx = h1.index.union(h2.index)
            fl.append(0.5 * float((h1.reindex(idx).fillna(0) - h2.reindex(idx).fillna(0)).abs().sum()))
        tvr.append({"from": s - 1, "to": s, "tv_distance": tv, "split_half_noise_mean": float(np.mean(fl)),
                    "split_half_noise_p95": float(np.quantile(fl, .95))})
    T3dtv = pd.DataFrame(tvr)
    out["3d_tv_distance"] = recs(T3dtv)
    # month by month
    mrows = []
    for (s, mo), g in LA.groupby(["season", "month"]):
        if mo not in (11, 12, 1, 2, 3):
            continue
        t3 = g[g["text_cls"] == "three"]
        tr = g[g["text_cls"] == "rimtext"]
        allm = D[(D["season"] == s) & (D["month"] == mo)]
        pr = allm[allm["located"] & allm["h_located"]]
        e = np.sqrt(((pr["xf"] / 10) - (47 - pr["hx"].abs())) ** 2 + (((pr["y"] / 10 - 25).abs()) - pr["hy"].abs()) ** 2)
        mrows.append({"season": s, "month": mo, "attempts": len(allm), "located_pct": 100 * len(g) / len(allm),
                      "canned_pct": 100 * float(g["canned"].mean()), "spike_pct": 100 * float(g["spike"].mean()),
                      "within_1ft_basket_pct": 100 * float((g["dist"] < 1).mean()),
                      "rim_mean_dist": float(g.loc[g["cls"] == "rim", "dist"].mean()),
                      "three_mean_dist": float(g.loc[g["cls"] == "three", "dist"].mean()),
                      "text3_inside_arc_pct": 100 * float((t3["dist"] < ARC_FT).mean()) if len(t3) else np.nan,
                      "layup_gt6ft_pct": 100 * float((tr["dist"] > 6).mean()) if len(tr) else np.nan,
                      "hoopr_pair_n": len(pr), "hoopr_pair_le1ft_pct": 100 * float((e <= 1).mean()) if len(pr) else np.nan})
    T3dm = pd.DataFrame(mrows)
    out["3d_by_season_month"] = recs(T3dm)
    # weekly view of 2024-25 (and 2023-24 for contrast): when does CBBD's value start to equal ESPN's (hoopR)?
    wrows = []
    for s in (2024, 2025):
        a_ = D[D["season"] == s].copy()
        a_["week"] = a_["game_date"] - pd.to_timedelta(a_["game_date"].dt.weekday, unit="D")
        for wk, g in a_.groupby("week"):
            pr = g[g["located"] & g["h_located"]]
            e = np.sqrt(((pr["xf"] / 10) - (47 - pr["hx"].abs())) ** 2 + (((pr["y"] / 10 - 25).abs()) - pr["hy"].abs()) ** 2)
            lg = g[g["located"]]
            tr = lg[lg["text_cls"] == "rimtext"]
            wrows.append({"season": s, "week_start": str(wk.date()), "attempts": len(g), "cbbd_located_pct": 100 * float(g["located"].mean()),
                          "hoopr_located_pct": 100 * float(g["h_located"].mean()), "pairs": len(pr),
                          "pair_le1ft_pct": 100 * float((e <= 1).mean()) if len(pr) >= 30 else np.nan,
                          "layup_gt6ft_pct": 100 * float((tr["dist"] > 6).mean()) if len(tr) else np.nan})
    T3dw = pd.DataFrame(wrows)
    out["3d_weekly_2024_2025"] = recs(T3dw)
    md += ["### 3d. Regime: season-level structure of located shots\n", md_table(T3d),
           "Distance-from-basket histogram of located attempts (% of season's located)\n", md_table(hist_pct),
           "Total-variation distance of the histogram between consecutive seasons vs a split-half (random halves of games) noise floor\n", md_table(T3dtv),
           "Month by month (months 11-3; the `hoopr_pair` columns compare CBBD with hoopR's ESPN value on the same plays)\n", md_table(T3dm),
           "Weekly, seasons 2023-24 and 2024-25: the CBBD value vs the ESPN (hoopR) value on the same plays\n", md_table(T3dw)]
    # ------------------------------------------------------------------ 3e: charting accuracy against the play text
    rows = []

    def acc_rows(g, tag):
        t3 = g[g["text_cls"] == "three"]
        tj = g[g["text_cls"] == "twojump"]
        tr = g[g["text_cls"] == "rimtext"]
        td = g[g["is_dunk"] & (g["text_cls"] == "rimtext")]
        for nm, h, col, thr, op in (("text three charted inside 22.15 ft", t3, "dist", ARC_FT, "<"),
                                    ("text three charted inside 20.75 ft", t3, "dist", 20.75, "<"),
                                    ("text three charted inside 21.65 ft (0.5 ft slack)", t3, "dist", 21.65, "<"),
                                    ("text two-jumper charted outside 22.15 ft", tj, "dist", ARC_FT, ">"),
                                    ("text two-jumper charted outside 23.65 ft (1.5 ft slack)", tj, "dist", 23.65, ">"),
                                    ("layup/tip charted > 6 ft from rim", tr[~tr["is_dunk"]], "dist", 6, ">"),
                                    ("dunk charted > 6 ft from rim", td, "dist", 6, ">"),
                                    ("layup/dunk/tip charted > 10 ft from rim", tr, "dist", 10, ">")):
            bad = (h[col] < thr) if op == "<" else (h[col] > thr)
            rows.append({**tag, "check": nm, "n": len(h), "misplaced_pct": 100 * float(bad.mean()) if len(h) else np.nan, "flag": up(len(h))})

    for s in ALL_SEASONS:
        g = LA[LA["season"] == s]
        acc_rows(g, {"season": s, "subset": "all located"})
        if s in OUT_SEASONS:
            acc_rows(g[g["made"] == 1], {"season": s, "subset": "makes"})
            acc_rows(g[g["made"] == 0], {"season": s, "subset": "misses"})
        acc_rows(g[g["canned"]], {"season": s, "subset": "canned pixels"})
        acc_rows(g[g["spike"]], {"season": s, "subset": "spike pixel"})
        acc_rows(g[~g["canned"]], {"season": s, "subset": "non-canned"})
    T3e = pd.DataFrame(rows)
    out["3e_text_vs_coordinate"] = recs(T3e)
    # make vs miss paired test on misplacement
    prow = []
    for s in OUT_SEASONS:
        g = LA[LA["season"] == s]
        for nm, sel, ind in (("text three inside 22.15 ft", g["text_cls"] == "three", (g["dist"] < ARC_FT)),
                             ("layup/dunk/tip > 6 ft", g["text_cls"] == "rimtext", (g["dist"] > 6)),
                             ("text two-jumper outside 22.15 ft", g["text_cls"] == "twojump", (g["dist"] > ARC_FT))):
            h = g[sel]
            r = paired_diff(h, h["made"] == 1, h["made"] == 0, ind[sel].astype(float), rng)
            prow.append(pdiff_row({"season": s, "check": nm, "a": "makes", "b": "misses"}, r))
    T3ep = pd.DataFrame(prow)
    out["3e_misplacement_make_vs_miss_paired"] = recs(T3ep)
    # hoopR coordinates on their own located attempts
    hrows = []
    for s in ALL_SEASONS:
        g = D[(D["season"] == s) & D["h_located"]]
        t3, tj, tr = (g[g["text_cls"] == k] for k in ("three", "twojump", "rimtext"))
        hrows.append({"season": s, "hoopr_located": len(g),
                      "text three inside 22.15 ft (%)": 100 * float((t3["dh"] < ARC_FT).mean()) if len(t3) else np.nan,
                      "text two-jumper outside 22.15 ft (%)": 100 * float((tj["dh"] > ARC_FT).mean()) if len(tj) else np.nan,
                      "layup/dunk/tip > 6 ft (%)": 100 * float((tr["dh"] > 6).mean()) if len(tr) else np.nan,
                      "n_three": len(t3), "n_rimtext": len(tr)})
    T3eh = pd.DataFrame(hrows)
    out["3e_hoopr_own_located"] = recs(T3eh)
    md += ["### 3e. Charting accuracy against the play text (CBBD coordinates; distances folded to the near basket)\n",
           md_table(T3e[T3e["subset"] == "all located"].drop(columns=["subset"])),
           "Makes vs misses, and canned vs non-canned\n", md_table(T3e[T3e["subset"].isin(["makes", "misses", "canned pixels", "spike pixel", "non-canned"])]),
           "Paired test: is misplacement different for makes and misses within game (a = makes)?\n", md_table(T3ep),
           "hoopR (ESPN) coordinates on hoopR's own located attempts\n", md_table(T3eh)]
    jdump(out, RES / "part3.json")
    (RES / "part3.md").write_text("\n".join(md), encoding="utf-8")
    log("part 3 written")


# =====================================================================================
# REPORT
# =====================================================================================
VERDICT_1 = """**Verdict, part 1 (what the on-disk sources are).** There are two independent coordinate sources on disk, not three. `data/raw/hoopr/shots/shots_YYYY.parquet` is a verbatim copy of the hoopR `pbp` rows with `shooting_play == True` (same 12 key columns identical as a multiset in all five seasons; it came from `sportsdataverse/hoopR-mbb-data` `mbb/shots/parquet/`, pulled by `scripts/pull_hoopr.py` on 2026-09-10); it adds no coordinates and no play id, and it includes free-throw rows. So the two sources are hoopR pbp (ESPN values, integer feet, centre-court frame, basket at |x| = 41.75 ft, x sign fixed by home/away so there is no half-time switching) and CBBD plays (a 101 x 101 grid on 94 x 50 ft, steps of 0.94 ft by 0.5 ft, x switching ends at half-time in 99.98% of team-games, per the 10-25 coverage report). They are NOT copies of each other. In 2022-2024 they locate different attempts (game-level coverage correlation -0.66 to -0.70) and on the 7-11k plays both locate, only 9-10% agree within 1 ft and 43-48% disagree by more than 3 ft; in 2025-26 they agree within 1 ft on 91-95% of 190k-650k pairs (the same ESPN value, re-quantised), but the 2025 agreement exists only from the week of 2025-02-10 (hoopR's own located share jumps from 3.9% to 87.6% that week; part 3d). The share of exactly equal pairs is 0.0-0.9%, which is a quantisation fact, not disagreement. Neither source locates every attempt: the UNION coverage is 90.5 / 91.9 / 92.8 / 99.2% for 2022-2025 (CBBD alone 80.0 / 84.3 / 87.5 / 98.6%; the 2025-26 sealed season 99.9% union vs 87.8% CBBD, descriptive only). hoopR-only attempts add 10.5 / 7.6 / 5.3 / 0.6 points, but in 2022-2024 they are a different measurement vintage than CBBD's values, so a union table would mix two incompatible placements. On the same plays, hoopR's 2022-24 values are far more consistent with the play text than CBBD's (layups/dunks/tips more than 6 ft from the rim: 0.1-0.2% vs 10.5-13.7%; text-threes charted inside the arc: 0.1-0.3% vs 2.4-3.1%); in 2025-26 the two sources are equally (im)precise. Which source is closer to physical truth cannot be settled with files on disk; text consistency is the only truth-independent yardstick and it favours hoopR in 2022-24 on the 6-12% of attempts it covers."""

VERDICT_2 = """**Verdict, part 2 (which games lack coordinates).** CBBD is the best source for 2022-2025 (hoopR in the sealed 2026 season), so tables use CBBD with the union alongside. Missingness is NOT decided per game alone. Whole games at 0% are rare (5.1 / 3.6 / 2.6 / 0.2%) and fully located games are only 30-44% of games in 2022-2025; 56-67% of games are partially located. But missingness is strongly clustered at every nested level: the share of attempts in units that are entirely located or entirely missing is 37% for games (0.0% expected if each attempt were located independently), 55% for game x period, 72% for game x team x period, 76% for game x period x 5-minute block and 86% for game x shooter (20% expected) in 2022. In other words it comes in runs inside a game (a period, a team's stretch, a player's stretch). The missing set is a biased subset in 2022-2024 and nearly unbiased in 2025. By CBBD coverage (2022 / 2023 / 2024 / 2025): power-conference home teams 65.5 / 73.5 / 81.0 / 97.9% vs low-tier 84.8 / 87.8 / 90.0 / 98.9%; neutral-site games 51.7 / 60.9 / 61.9 / 98.0% vs 83.7 / 87.6 / 90.8 / 98.7%; postseason 31.7 / 27.1 / 46.5 / 94.5%; March 63.7 / 64.4 / 75.7 / 98.5% vs January 84.8 / 89.3 / 92.4 / 98.2%; games with no line in the lines files 58.1 / 62.0 / 76.2 / 98.6% vs 82.1 / 85.9 / 89.6 / 98.6% (and 29.6% of no-line games in 2022 are entirely unlocated vs 2.9% of lined games). The games CBBD lacks are disproportionately the high-profile games (power conference, neutral site, March, tournament) and those are the games hoopR does locate (union coverage for power-conference games 95.9 / 96.4 / 96.0%), while the lowest-profile games with no line are the ones neither source locates. Per team, the located offence attempts are p10 1,035 / 1,283 / 1,482 / 1,672, median 1,536 / 1,667 / 1,739 / 1,860 (minimum 0 / 0 / 0 / 761, and 29 / 18 / 10 / 0 teams below 50% located); defence p10 1,051 / 1,265 / 1,460 / 1,663 (minimum 510 / 676 / 525 / 1,009). Only 2025 is close to complete and even; before that, any per-game or per-team location feature is built on a non-random 80-88%."""

VERDICT_3 = """**Verdict, part 3 (is x/y trustworthy as an upstream input?).** Partly. Three- and two-point jumpers are placed consistently with the play text (text-threes charted inside 22.15 ft: 1.83 / 1.62 / 1.81 / 1.86%, inside 20.75 ft only 0.13-0.14%; text two-point jumpers charted outside 22.15 ft: 0.39 / 0.34 / 0.21 / 0.16%), and make rate falls with distance as it should. The rim is the problem, and the problem is outcome-dependent. (i) Charting accuracy depends on the result: layups/tips charted more than 6 ft from the rim are 6.1-6.8% of makes but 14.6-15.7% of misses in every season (within-game difference -8.4 to -8.9 points against a bootstrap floor of 0.2 points); the same asymmetry appears in hoopR's values for 2025-26 (5.8% vs 13.8% in 2025), so it is in the ESPN feed and not a CBBD processing artefact. (ii) Which attempts get a coordinate depends on the result: the located share of rim makes is 1.3 / 1.6 / 1.7 / 2.1 points lower than that of rim misses within game (floor 0.1 point; z = -13 to -36; 2.1 to 3.6 points in games with partial coverage); jump2 is smaller (0.3-1.1 points) and threes are outcome-blind (within +/-0.2 points). (iii) The 2025 gap is reproduced and explained: located rim attempts make 58.1% vs 81.5% for the 5,510 unlocated ones (2022-24: 58.4 / 58.3 / 57.7% vs 61.4 / 62.3 / 61.7%, sign opposite and 3-4 points). It lives entirely in layups (unlocated 81.1% vs located 54.0%; dunks are equal, 91.1% vs 87.8%), is NOT a few games (2,860 games, top 10 hold 8.2%; excluding them 83.3%), not a few teams (468), not a month, a tier or a half, and not shot mix (located rate re-weighted to the unlocated play-type mix is 56.6%). In 2025 about 3.0% of rim makes lack a coordinate against 1.0% of rim misses. (iv) Canned pixels. Using the instructed rule (> 0.5% of located attempts) there are 15 / 14 / 16 / 19 canned pixels holding 10.8 / 10.6 / 11.8 / 13.6% of located attempts (28-35% of located rim attempts, 1.3-2.2% of jump2, 0.03% of threes); 0.5% is 20 times the uniform pixel share (0.025%) and sits just under the 99.9th percentile of pixel shares (0.60-0.70%). Only ONE pixel per season is an isolated spike (8-9 times its eight neighbours; 5.64 ft from the baseline, 25.0 ft from the sideline, 0.39 ft from the basket centre): 2.7 / 2.9 / 2.9 / 3.0% of located attempts and about 7.6-8.0% of located rim attempts; the other canned pixels have neighbour ratios of mostly 0.6-1.8 and are the dense layup region. Canned rim attempts make 66.7-68.1% vs 53.4-54.5% elsewhere (+13 pooled, +14 to +15 within game), but matched on distance from the basket (0.5 ft bins to 4 ft) the gap falls to +1.4 to +1.9 points (SE 0.003), and the spike pixel makes LESS than its own eight neighbours (-2.3 to -3.2 points within game, z -3.6 to -5.9). The spike pixel (about 70% vs 57% elsewhere in the PM's numbers) is therefore mostly geometry. (v) Response to team skill: for rim shots on the 2022-25 as-of offence-PAE rating, make rate rises by 5.5 / 6.4 / 6.9 / 6.7 points from quintile 1 to 5 (slope 0.70 / 0.59 / 0.63 / 0.52 per rating point); canned (> 0.5%) pixels respond like the rest in 2023-25 (slope difference -0.07 / +0.02 / -0.02, all |z| < 1; 2022 -0.29, z -2.8), but the isolated spike pixel responds at about 0.2-0.36 against 0.63-0.96 for its own ring (spike minus ring z = -2.0 / -2.2 / -4.3 / -4.8): a damped response, consistent with a mixture of true at-the-rim shots and uninformative default placements, though a physical explanation cannot be excluded. Jump2 canned pixels (1.5-3k attempts) have slopes of -1.2 / -0.2 / -0.7 / +0.3, below the non-canned 0.24-0.35 in 2022 and 2024. (vi) Regime change: nothing in the play-text accuracy moves between 2023-24 and 2024-25 (text-three inside the arc 1.7-2.0% in every month of both seasons; layups > 6 ft 9.8-10.7% vs 8.3-10.5%, drifting down through 2025), and no break is visible inside 2024-25. What does step is the season boundary: CBBD located share (Nov 85.0% -> Nov 98.7%), canned share (11.9% -> 13.3% in November), and the distance histogram (total-variation distance 2024 vs 2025 = 0.022 against a split-half noise floor of 0.005-0.007 and 0.009-0.011 for the two earlier transitions). The hoopR/CBBD agreement break on 2025-02-10 is a hoopR scrape-vintage artefact (hoopR located share 3.9% -> 87.6% -> 99.99% in consecutive weeks) and not a charting change. We cannot tell from these tables whether ESPN changed charting at the start of 2024-25 or CBBD changed what it ingests; both are consistent with a step at the season boundary, and the higher 2025 coverage alone changes the composition of the located set."""

FINAL = """## What this means for a shot-location stage (descriptive only; no recommendation to adopt anything)

- **Where the signal looks reliable.** Jumper and three placements agree with the play text on more than 98% of attempts in all four seasons, they are outcome-blind in whether they are located, and their misplacement differs between makes and misses by at most 0.5 points. Location-by-make-rate curves for those classes are measuring where shots are taken.
- **Where it is contaminated.** For rim attempts, distance from the basket is partly a function of the result: misses are charted more than 6 ft from the rim 2.3-2.4 times as often as makes, and makes are 1.3-2.1 points less likely to carry any coordinate. A make model that takes a located rim distance (or the per-attempt cell rate of the earlier round) as an input can therefore learn "charted far from the rim implies miss". A shot-location stage that samples locations from the located-shot distribution would reproduce the same coupling in the simulation, in the direction that flatters the make model offline. The evidence of this audit is that the outcome dependence is present in the ESPN values themselves in 2025-26 (hoopR and CBBD agree), not only in one vendor's processing.
- **Coverage is season-dependent and not random before 2025.** 2022-2024 coordinates exist for 80-88% of attempts, missing in runs inside games, with the gaps concentrated in power-conference, neutral-site, March and postseason games; 2025 is 98.6% and even. The 2025-26 season has a different gap pattern (11% of games entirely unlocated in CBBD, mostly low-tier games with no line), and hoopR fills it. Per-team rating features built on located shots rest on 1.0-1.7k located attempts per team-season at the 10th percentile.
- **Two measurement vintages.** For 2022-2024 the two available sources disagree on the same plays (about 9-10% within 1 ft), and the less-covered source (hoopR) is the one that agrees with the play text. For 2025-26 the sources agree and share the same text-inconsistency profile. There is no second-source confirmation of CBBD's 2022-2024 positions, which is where a stage trained on folds 1 and 2 would get most of its data.
- **One placeholder-like pixel.** A single pixel next to the basket centre holds about 3% of located attempts (8% of located rim attempts) in every season including the sealed one; its make rate is below that of its neighbours and its response to team shooting skill is about 0.3-0.45 times its neighbours', which is the only canned-pixel signature in the data. The other 14-18 pixels above 0.5% are the dense layup region and are not placeholders on any test here.
- **Stage mechanics already known.** The engine samples no location; a stage would need class-conditional location sampling that respects the missing-by-outcome pattern, and its own bottom-up gate. Nothing in this audit evaluates such a stage.
- **Not tested here.** Whether the rim outcome dependence is symmetric on the defence side, whether it varies by arena/broadcast crew, and whether the same asymmetry exists in a non-ESPN source; none is available on disk.
"""

METHOD = """## Scope, definitions, floors

- **Diagnostic only.** No model is fit, nothing is wired into the engine. Script: `scripts/diag_coordinate_audit_v1.py` (`--part master|1|2|3|report|all`). Intermediate tables: `results/coordinate_audit/` (gitignored).
- **Seasons.** season = ending year (2025 = 2024-25). Outcome tables use 2022-2025 only. Season 2026 (2025-26) is sealed for selection: it is read for non-outcome structure only (coverage, pixel and distance structure, text-vs-position accuracy), `made` is set to NaN for it at load time, and no make rate, no outcome comparison and no rating test touches it. Rows labelled 2026 are descriptive and play no role in any verdict.
- **Universe.** CBBD FGA rows (de-duplicated, `E.classify_frame`), the same universe and classes as `build_pae_shot_rating_v1.py` (rim includes jumpers located within 2.27 ft of the rim: "build" class). The "feed" class has no location override and is used where the located share is the outcome (it cannot depend on location). hoopR rows are joined by ESPN play id (hoopR `id` == CBBD `sourceId`; 99.9-100% of CBBD FGA rows have a hoopR row).
- **Frame.** CBBD x is folded to the left half (`x > 470 -> 940 - x`), so the basket is at (52.5, 250) in tenths of feet; distances are Euclidean feet from that point. hoopR distances are `sqrt(raw_y^2 + y^2)` (hoopR's raw frame is basket-relative; verified by the dunk medians). hoopR vs CBBD position comparisons use folded baseline distance and |sideline offset| because the two frames flip left/right differently.
- **Pixel.** A (folded x, y) pair on CBBD's 101 x 101 grid. **Canned** (as instructed) = a pixel holding more than 0.5% of a season's located attempts (all classes pooled). **Spike pixel** = a canned pixel whose count is at least 3 times the mean of its 8 grid neighbours; **ring** = those 8 neighbours.
- **Floors.** Paired comparisons are within game: the difference is computed in every game that holds both groups and averaged with weights n_a n_b / (n_a + n_b); the floor is the standard deviation of that estimate over 300 bootstrap resamples of games (slopes: 200 resamples). Pooled (unpaired) rates are reported next to it. Slope differences add the two bootstrap SEs in quadrature, which is approximate because the groups share games.
- **Underpowered.** Any cell with fewer than 500 attempts in either group is flagged UNDERPOWERED in the `flag` column and is not read as signal or as absence of signal.
- **Text classes (3e).** From `playText` only, independent of position: "three point" -> three; layup / dunk / tip -> rim text; "jumper" without "three point" -> two-point jumper. Arc = 22.15 ft for all seasons (instructed); the 20.75 ft and 0.5 / 1.5 ft slack variants are shown because coordinates sit on a 0.94 x 0.5 ft grid.
"""


def report(a) -> None:
    p1 = json.loads((RES / "part1.json").read_text())
    p2 = json.loads((RES / "part2.json").read_text())
    p3 = json.loads((RES / "part3.json").read_text())
    md = ["# Coordinate-placement audit (2026-10-27)\n",
          "Diagnostic worker output for PM ruling 4 of 2026-10-25 (`PROJECT_STATUS.md`, fg_make PAE round): can shot x/y be trusted as an upstream input, or is the location signal partly outcome-dependent charting? Script `scripts/diag_coordinate_audit_v1.py`; tables below are also in `docs/tests/coordinate_audit_2026-10-27.json`. Descriptive; nothing is adopted or wired.\n",
          METHOD,
          "## Part 1. Inventory and reconciliation of on-disk coordinate sources\n", VERDICT_1 + "\n", (RES / "part1.md").read_text(encoding="utf-8"),
          "\n## Part 2. Coverage structure\n", VERDICT_2 + "\n", (RES / "part2.md").read_text(encoding="utf-8"),
          "\n## Part 3. Placement audit\n", VERDICT_3 + "\n", (RES / "part3.md").read_text(encoding="utf-8"),
          "\n" + FINAL]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "coordinate_audit_2026-10-27.md").write_text("\n".join(md), encoding="utf-8")
    jdump({"created_at": pd.Timestamp.now("UTC").isoformat(), "script": "scripts/diag_coordinate_audit_v1.py",
           "seasons_outcome": OUT_SEASONS, "seasons_descriptive_only": [2026], "canned_share_threshold": CANNED_SHARE,
           "arc_ft": ARC_FT, "min_n_underpowered": MIN_N, "bootstrap_reps": BOOT,
           "verdicts": {"part1": VERDICT_1, "part2": VERDICT_2, "part3": VERDICT_3, "final": FINAL},
           "part1": p1, "part2": p2, "part3": p3}, DOCS / "coordinate_audit_2026-10-27.json")
    log("report written")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["master", "1", "2", "3", "report", "all"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    parts = ["master", "1", "2", "3", "report"] if a.part == "all" else [a.part]
    for p in parts:
        {"master": part_master, "1": part1, "2": part2, "3": part3, "report": report}[p](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
