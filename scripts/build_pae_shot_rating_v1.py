#!/usr/bin/env python
"""build_pae_shot_rating_v1.py -- fg_make PAE round (experiments.md section 31).

A shot-location "points above expected" (PAE) team rating built from CBBD
`shot_location_x/y`, entered as a feature arm against the served fg_make
reference. The idea is the collaborator's (rfq-collab "CBB Model/Heuermann"
README, Layer 2); this is our own construction under our own as-of discipline,
not his code.

Parts (run in this order):

  --part coverage   STEP 1 diagnostic (no fit). Coordinate coverage by season /
                    class / team / site / month, made rate on located vs
                    unlocated attempts, the coordinate frame, and the
                    design<->pbp row join. Writes --coverage-out.
  --part build      Everything that is FITTED (only after the pre-registration
                    commit): the grid choice g on fold-1 training seasons
                    (leave-one-season-out), prior-season cell baselines, per-shot
                    residuals, k / rho on fold-1 training season 2023, as-of team
                    ratings with estimation variance v, the change-form leak test,
                    and one row-aligned column table for the design. Writes under
                    --out-dir (data/processed/models/fg_make/pae_v1/).

Season 2026 (= 2025-26) is SEALED and never opened: `assert_not_sealed` on the
season list before any file is read.

    .venv/Scripts/python.exe scripts/build_pae_shot_rating_v1.py --part coverage
    .venv/Scripts/python.exe scripts/build_pae_shot_rating_v1.py --part build
"""
from __future__ import annotations

import argparse
import hashlib
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

SEASONS = [2022, 2023, 2024, 2025]
FGA = ("FGA_rim", "FGA_jump2", "FGA_3")
DESIGN = ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet"
OUT_DIR = ROOT / "data/processed/models/fg_make/pae_v1"
COVERAGE_OUT = ROOT / "docs/tests/fg_make_pae_coverage_2026-10-25.json"
RIM_MAX_FT = 2.27          # the v2 build's override threshold (model.md section 2)
HALF_X = 470.0             # half of the 0-940 full-court x axis
BASKET = (52.5, 250.0)     # the near basket after folding x to the left half

# ---- pre-registered constants (experiments.md section 31) -----------------------
GRIDS = {"g10": ("xy", 10, 10), "g15": ("xy", 15, 15), "g25": ("xy", 25, 25),
         "da15": ("da", 15, 15)}          # da = distance x angle bins of similar count
CELL_SMOOTH_A = 20.0       # pseudo-attempts toward the (value) league mean per cell
K_GRID = (50, 100, 200, 400, 800, 1600, 3200)
RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# =====================================================================================
# loading
# =====================================================================================
def load_fga(season: int) -> pd.DataFrame:
    """De-duplicated CBBD plays for one season, FGA rows only, with the project's
    own classifier (v2 rim override) and the raw coordinate columns."""
    cols = tuple(E.PLAY_COLUMNS) + ("gameStartDate",)
    p = E.load_plays(season, columns=cols)
    cls = E.classify_frame(p, rim_override_max_ft=RIM_MAX_FT)
    n_shooting_all = int(p["shootingPlay"].fillna(False).astype(bool).sum())
    x_all = pd.to_numeric(p["shot_location_x"], errors="coerce")
    y_all = pd.to_numeric(p["shot_location_y"], errors="coerce")
    sp = p["shootingPlay"].fillna(False).astype(bool)
    loc_all = (x_all.notna() & y_all.notna())
    keep = cls.isin(list(FGA)).to_numpy()
    f = p.loc[keep].copy()
    f["shot_class"] = cls[keep].astype(str).to_numpy()
    f["x"] = pd.to_numeric(f["shot_location_x"], errors="coerce")
    f["y"] = pd.to_numeric(f["shot_location_y"], errors="coerce")
    f["located"] = (f["x"].notna() & f["y"].notna()).to_numpy()
    made = f["shot_made"]
    if made.dtype == object:
        made = made.map({True: True, False: False})
    f["made"] = made.astype("boolean").fillna(f["scoringPlay"].astype("boolean")).fillna(False).astype(bool)
    f["value"] = np.where(f["shot_class"].to_numpy() == "FGA_3", 3, 2)
    f["game_date"] = pd.to_datetime(f["gameStartDate"], utc=True).dt.tz_convert("US/Eastern").dt.tz_localize(None).dt.normalize()
    f.attrs["shooting_rows_all"] = n_shooting_all
    f.attrs["shooting_rows_all_located"] = int((sp & loc_all).sum())
    return f


def fold_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold the full court onto the left half: the near basket is then (52.5, 250).
    The feed does not say which end a team attacks; folding makes that moot."""
    xf = np.where(x > HALF_X, 940.0 - x, x)
    return xf, y


def dist_angle(xf: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = xf - BASKET[0], y - BASKET[1]
    d = np.sqrt(dx ** 2 + dy ** 2) / 10.0                 # feet
    a = np.degrees(np.arctan2(dy, dx))                     # -180..180, 0 = straight out
    return d, a


# =====================================================================================
# STEP 1: coverage
# =====================================================================================
def _cov(df: pd.DataFrame, by: list[str]) -> list[dict]:
    g = df.groupby(by, observed=True).agg(n=("located", "size"), located=("located", "sum"))
    g["coverage_pct"] = (100 * g["located"] / g["n"]).round(2)
    return g.reset_index().to_dict(orient="records")


def coverage(a, t0) -> int:
    assert_not_sealed(SEASONS, context="PAE coverage")
    out: dict = {"created_at": pd.Timestamp.now("UTC").isoformat(), "seasons": SEASONS,
                 "rim_override_max_ft": RIM_MAX_FT}
    frames = []
    recon = []
    for s in SEASONS:
        f = load_fga(s)
        recon.append({"season": s, "shooting_rows_all_incl_ft": f.attrs["shooting_rows_all"],
                      "shooting_rows_all_located_pct": round(100 * f.attrs["shooting_rows_all_located"]
                                                             / f.attrs["shooting_rows_all"], 2),
                      "fga_rows": int(len(f)), "fga_located_pct": round(100 * f["located"].mean(), 2)})
        f["season"] = s
        frames.append(f[["gameId", "id", "season", "teamId", "opponentId", "isHomeTeam", "period",
                         "secondsRemaining", "shot_shooter_id", "shot_class", "made", "value", "x", "y",
                         "located", "game_date", "playType"]])
        log(f"season {s}: {len(f):,} FGA rows, located {100 * f['located'].mean():.2f}%", t0)
    d = pd.concat(frames, ignore_index=True)
    out["reconciliation"] = recon
    out["by_season"] = _cov(d, ["season"])
    out["by_season_class"] = _cov(d, ["season", "shot_class"])
    d["site"] = np.where(d["isHomeTeam"].astype("boolean").fillna(False), "home", "away")
    out["by_season_site"] = _cov(d, ["season", "site"])
    d["month"] = d["game_date"].dt.month
    out["by_season_month"] = _cov(d, ["season", "month"])
    out["by_playtype_season"] = _cov(d, ["season", "playType"])
    tm = d.groupby(["season", "teamId"]).agg(n=("located", "size"), cov=("located", "mean"))
    tm = tm[tm["n"] >= 200]
    out["per_team_coverage"] = [
        {"season": int(s), "n_teams": int(len(g)),
         **{k: round(100 * float(np.quantile(g["cov"], q)), 2)
            for k, q in (("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1))},
         "share_teams_below_40pct": round(100 * float((g["cov"] < 0.40).mean()), 2)}
        for s, g in tm.groupby(level="season")]
    gm = d.groupby(["season", "gameId"])["located"].mean()
    out["per_game_coverage"] = [
        {"season": int(s), "n_games": int(len(g)),
         "share_games_zero": round(100 * float((g == 0).mean()), 2),
         "share_games_full": round(100 * float((g >= 0.99).mean()), 2),
         "p10": round(100 * float(g.quantile(.1)), 2), "median": round(100 * float(g.median()), 2)}
        for s, g in gm.groupby(level="season")]
    # made rate located vs unlocated
    mr = d.groupby(["season", "shot_class", "located"]).agg(n=("made", "size"), make=("made", "mean"))
    mr["make"] = (100 * mr["make"]).round(3)
    out["make_rate_located_vs_not"] = mr.reset_index().to_dict(orient="records")
    # within located games only (games with partial coverage): is missingness within a game informative?
    part = gm[(gm > 0.05) & (gm < 0.95)].reset_index()[["season", "gameId"]]
    dp = d.merge(part, on=["season", "gameId"])
    mr2 = dp.groupby(["season", "shot_class", "located"]).agg(n=("made", "size"), make=("made", "mean"))
    mr2["make"] = (100 * mr2["make"]).round(3)
    out["make_rate_located_vs_not_partial_games"] = mr2.reset_index().to_dict(orient="records")
    # coordinate frame
    L = d[d["located"]].copy()
    x, y = L["x"].to_numpy(), L["y"].to_numpy()
    out["frame"] = {"x_min": float(x.min()), "x_max": float(x.max()), "y_min": float(y.min()),
                    "y_max": float(y.max()), "share_x_left_half": round(float((x < HALF_X).mean()), 4)}
    xf, yy = fold_xy(x, y)
    dist, ang = dist_angle(xf, yy)
    # distance to each raw basket, unfolded
    d_left = np.sqrt((x - 52.5) ** 2 + (y - 250) ** 2) / 10
    d_right = np.sqrt((x - 887.5) ** 2 + (y - 250) ** 2) / 10
    bins = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 47, 100]
    L["dist_ft"] = dist
    L["dbin"] = pd.cut(dist, bins, right=False)
    curve = L.groupby(["dbin", "value"], observed=True).agg(n=("made", "size"), make=("made", "mean")).reset_index()
    curve["dbin"] = curve["dbin"].astype(str)
    curve["make"] = (100 * curve["make"]).round(2)
    out["frame"]["make_by_folded_distance"] = curve.to_dict(orient="records")
    # what if we had NOT folded: distance to the LEFT basket only
    L["dbin_left"] = pd.cut(d_left, bins, right=False)
    c2 = L.groupby("dbin_left", observed=True).agg(n=("made", "size"), make=("made", "mean")).reset_index()
    c2["dbin_left"] = c2["dbin_left"].astype(str)
    c2["make"] = (100 * c2["make"]).round(2)
    out["frame"]["make_by_left_basket_distance_unfolded"] = c2.to_dict(orient="records")
    out["frame"]["share_nearer_right_basket"] = round(float((d_right < d_left).mean()), 4)
    # which end does a team attack: per (game, team, period) share of shots in left half
    L["left"] = (L["x"] < HALF_X)
    end = L.groupby(["season", "gameId", "teamId", "period"])["left"].agg(["mean", "size"])
    end = end[end["size"] >= 8]
    out["frame"]["per_team_period_left_share_quantiles"] = {
        k: round(float(end["mean"].quantile(q)), 4) for k, q in
        (("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))}
    out["frame"]["share_team_periods_one_end_ge95pct"] = round(
        float(((end["mean"] >= .95) | (end["mean"] <= .05)).mean()), 4)
    hv = L.merge(end.reset_index()[["season", "gameId", "teamId", "period", "mean"]],
                 on=["season", "gameId", "teamId", "period"])
    sw = hv[hv["period"].isin([1, 2])].pivot_table(index=["season", "gameId", "teamId"], columns="period",
                                                   values="mean", aggfunc="first").dropna()
    out["frame"]["share_team_games_switch_end_at_half"] = round(
        float(((sw[1] - 0.5) * (sw[2] - 0.5) < 0).mean()), 4) if len(sw) else None
    med = L.groupby("playType")["dist_ft"].median()
    out["frame"]["median_folded_distance_by_playtype"] = {k: round(float(v), 2) for k, v in med.items()}
    # 3PA distance sanity (the 3-pt line is 22.15 ft corner, 22.15 / 22.15 since 2019-20 NCAA men)
    out["frame"]["three_pt_distance_quantiles_ft"] = {
        k: round(float(np.quantile(dist[L["value"].to_numpy() == 3], q)), 2)
        for k, q in (("p01", .01), ("p05", .05), ("p50", .5))}
    # ---- design join
    out["design_join"] = design_join(d, t0, report_only=True)[1]
    COV = Path(a.coverage_out)
    COV.parent.mkdir(parents=True, exist_ok=True)
    COV.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log(f"wrote {COV}", t0)
    return 0


# =====================================================================================
# design <-> pbp row join
# =====================================================================================
JOIN_KEYS = ["cbbd_game_id", "period", "sec", "shooter", "shot_class", "made"]


def design_join(fga: pd.DataFrame, t0: float, report_only: bool = False):
    """Attach (x, y) from the pbp FGA rows to every design row.

    Key (cbbd game, period, seconds remaining, shooter, class, made) plus an
    ordinal within duplicate keys in event order (pbp `id` order; the design
    preserves the stream's event order). `made` is in the key so the join cannot
    match a make to a miss; the MATCH RATE is reported split by made so a
    coverage asymmetry by outcome (the round-4 sh_assisted_share lesson,
    features.md section 6) is visible. Returns (aligned frame, report)."""
    des = pd.read_parquet(DESIGN, columns=["cbbd_game_id", "season", "period", "seconds_remaining",
                                           "shooter_id", "shot_class", "made", "off_team_id",
                                           "def_team_id", "game_date", "game_id"])
    des = des.reset_index(drop=True)
    des["row"] = np.arange(len(des))
    L = pd.DataFrame({
        "cbbd_game_id": fga["gameId"].astype("int64").to_numpy(),
        "period": pd.to_numeric(fga["period"], errors="coerce").fillna(1).astype("int64").to_numpy(),
        "sec": pd.to_numeric(fga["secondsRemaining"], errors="coerce").fillna(0).astype("int64").to_numpy(),
        "shooter": pd.to_numeric(fga["shot_shooter_id"], errors="coerce").to_numpy(),
        "shot_class": fga["shot_class"].to_numpy(),
        "made": fga["made"].to_numpy(dtype=bool),
        "x": fga["x"].to_numpy(), "y": fga["y"].to_numpy(), "pid": fga["id"].to_numpy()})
    L = L[np.isfinite(L["shooter"])].sort_values(["cbbd_game_id", "pid"], kind="stable")
    L["shooter"] = L["shooter"].astype("int64")
    R = pd.DataFrame({
        "cbbd_game_id": des["cbbd_game_id"].astype("int64").to_numpy(),
        "period": des["period"].astype("int64").to_numpy(),
        "sec": des["seconds_remaining"].astype("int64").to_numpy(),
        "shooter": des["shooter_id"].astype("int64").to_numpy(),
        "shot_class": des["shot_class"].astype(str).to_numpy(),
        "made": des["made"].astype(bool).to_numpy(), "row": des["row"].to_numpy()})
    L["ord"] = L.groupby(JOIN_KEYS).cumcount()
    R["ord"] = R.groupby(JOIN_KEYS).cumcount()
    m = R.merge(L[[*JOIN_KEYS, "ord", "x", "y"]], on=[*JOIN_KEYS, "ord"], how="left", indicator=True)
    m = m.sort_values("row").reset_index(drop=True)
    assert len(m) == len(des) and (m["row"].to_numpy() == des["row"].to_numpy()).all()
    matched = (m["_merge"] == "both").to_numpy()
    located = matched & np.isfinite(m["x"].to_numpy()) & np.isfinite(m["y"].to_numpy())
    rep = {"n_design_rows": int(len(des)), "matched_pct": round(100 * matched.mean(), 4),
           "unmatched_rows": int((~matched).sum()),
           "matched_pct_made": round(100 * matched[des["made"].to_numpy()].mean(), 4),
           "matched_pct_missed": round(100 * matched[~des["made"].to_numpy()].mean(), 4),
           "located_pct": round(100 * located.mean(), 3),
           "located_pct_by_season": {int(s): round(100 * float(located[(des["season"] == s).to_numpy()].mean()), 3)
                                     for s in SEASONS},
           "located_pct_by_season_class": {
               f"{s}_{c}": round(100 * float(located[((des["season"] == s) & (des["shot_class"] == c)).to_numpy()].mean()), 3)
               for s in SEASONS for c in FGA}}
    log(f"design join: matched {rep['matched_pct']}%, located {rep['located_pct']}%", t0)
    des["x"] = np.where(located, m["x"].to_numpy(), np.nan)
    des["y"] = np.where(located, m["y"].to_numpy(), np.nan)
    des["located"] = located
    des["join_matched"] = matched
    return des, rep


# =====================================================================================
# BUILD (fitted; run only after the pre-registration commit)
# =====================================================================================
def cell_index(xf: np.ndarray, y: np.ndarray, grid: str) -> np.ndarray:
    kind, nx, ny = GRIDS[grid]
    if kind == "xy":
        ix = np.clip((xf / HALF_X * nx).astype("int64"), 0, nx - 1)
        iy = np.clip((y / 500.0 * ny).astype("int64"), 0, ny - 1)
        return ix * ny + iy
    d, a = dist_angle(xf, y)
    # distance bins: 15 bins of 2 ft to 28 ft, last bin 28+ ; angle: 15 equal bins over [-90, 90], clipped
    idd = np.clip((d / 2.0).astype("int64"), 0, nx - 1)
    ia = np.clip(((np.clip(a, -90, 90) + 90) / 180.0 * ny).astype("int64"), 0, ny - 1)
    return idd * ny + ia


def fit_cells(df: pd.DataFrame, grid: str) -> pd.DataFrame:
    """Make rate per (cell, value), smoothed toward the value-level mean with
    CELL_SMOOTH_A pseudo-attempts. `df` = located shots of the baseline season."""
    c = cell_index(df["xf"].to_numpy(), df["y"].to_numpy(), grid)
    t = pd.DataFrame({"cell": c, "value": df["value"].to_numpy(), "made": df["made"].to_numpy(dtype=float)})
    p0 = t.groupby("value")["made"].mean()
    g = t.groupby(["cell", "value"])["made"].agg(["sum", "size"]).reset_index()
    g["p"] = (g["sum"] + CELL_SMOOTH_A * g["value"].map(p0)) / (g["size"] + CELL_SMOOTH_A)
    g.attrs["p0"] = p0.to_dict()
    return g[["cell", "value", "p"]], p0.to_dict()


def apply_cells(df: pd.DataFrame, grid: str, cells: pd.DataFrame, p0: dict) -> np.ndarray:
    c = cell_index(df["xf"].to_numpy(), df["y"].to_numpy(), grid)
    k = pd.DataFrame({"cell": c, "value": df["value"].to_numpy()})
    m = k.merge(cells, on=["cell", "value"], how="left")["p"].to_numpy()
    fb = df["value"].map(p0).to_numpy(dtype=float)
    return np.where(np.isfinite(m), m, fb)


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def choose_grid(shots: pd.DataFrame, t0) -> dict:
    """Fold-1 TRAINING seasons only (2022, 2023): leave-one-season-out log loss
    of the cell baseline alone (fit on one season, score the other)."""
    res = {}
    for g in GRIDS:
        lls, ns = [], []
        for fit_s, ev_s in ((2022, 2023), (2023, 2022)):
            A = shots[(shots["season"] == fit_s) & shots["located"]]
            B = shots[(shots["season"] == ev_s) & shots["located"]]
            cells, p0 = fit_cells(A, g)
            p = apply_cells(B, g, cells, p0)
            lls.append(_ll(B["made"].to_numpy(dtype=float), p))
            ns.append(len(B))
        res[g] = {"loso_log_loss": float(np.average(lls, weights=ns)), "by_eval_season": dict(zip(["2023", "2022"], lls))}
        log(f"grid {g}: LOSO ll {res[g]['loso_log_loss']:.6f}", t0)
    best = min(res, key=lambda k: res[k]["loso_log_loss"])
    return {"grid_results": res, "chosen": best,
            "rule": "min LOSO log loss of the cell baseline within fold-1 training seasons {2022, 2023}"}


def baselines_and_residuals(shots: pd.DataFrame, grid: str) -> tuple[pd.DataFrame, dict]:
    """Baseline for season S = located shots of season S-1; for S = 2022 (no
    earlier CBBD season on disk) the 2023 season (training rows only, disclosed in
    the pre-registration). Residual r = points - value * p_cell on located shots."""
    shots = shots.copy()
    shots["p_cell"] = np.nan
    meta = {}
    for s in SEASONS:
        bs = s - 1 if s > 2022 else 2023
        A = shots[(shots["season"] == bs) & shots["located"]]
        cells, p0 = fit_cells(A, grid)
        m = (shots["season"] == s).to_numpy()
        sub = shots.loc[m]
        p = apply_cells(sub, grid, cells, p0)
        shots.loc[m, "p_cell"] = np.where(sub["located"].to_numpy(), p, np.nan)
        cls_mean = A.groupby("shot_class")["made"].mean().to_dict()
        shots.loc[m, "cls_mean_prior"] = sub["shot_class"].map(cls_mean).to_numpy()
        meta[s] = {"baseline_season": bs, "n_baseline_located": int(len(A)),
                   "class_mean_prior": {k: round(float(v), 5) for k, v in cls_mean.items()}}
    pts = shots["made"].to_numpy(dtype=float) * shots["value"].to_numpy()
    shots["r"] = np.where(shots["located"].to_numpy(), pts - shots["value"].to_numpy() * shots["p_cell"].to_numpy(), np.nan)
    return shots, meta


def team_game_panel(shots: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, game, team, side) with sum / count of residuals on
    located shots: side 'off' = the team's own shots, 'def' = shots it faced."""
    L = shots[shots["located"]]
    rows = []
    for side, col in (("off", "off_team_id"), ("def", "def_team_id")):
        g = L.groupby(["season", "game_id", "game_date", col]).agg(rsum=("r", "sum"), n=("r", "size"),
                                                                    r2=("r", lambda v: float((v ** 2).sum())))
        g = g.reset_index().rename(columns={col: "team_id"})
        g["side"] = side
        rows.append(g)
    # all games a team played (so a game with zero located shots still exists as a row)
    allg = []
    for side, col in (("off", "off_team_id"), ("def", "def_team_id")):
        a = shots.groupby(["season", "game_id", "game_date", col]).size().reset_index(name="n_fga")
        a = a.rename(columns={col: "team_id"})
        a["side"] = side
        allg.append(a)
    P = pd.concat(allg, ignore_index=True).merge(pd.concat(rows, ignore_index=True),
                                                 on=["season", "game_id", "game_date", "team_id", "side"], how="left")
    P[["rsum", "n", "r2"]] = P[["rsum", "n", "r2"]].fillna(0.0)
    return P.sort_values(["side", "season", "team_id", "game_date", "game_id"]).reset_index(drop=True)


def league_asof(P: pd.DataFrame) -> pd.DataFrame:
    """League mean residual per located attempt over games strictly BEFORE each
    date, per (season, side). Offence and defence league means are the same
    shots, so they agree; computed per side for symmetry."""
    out = []
    for (s, side), g in P.groupby(["season", "side"]):
        d = g.groupby("game_date")[["rsum", "n"]].sum().sort_index()
        cs = d.cumsum().shift(1).fillna(0.0)
        lg = np.where(cs["n"] > 0, cs["rsum"] / cs["n"].where(cs["n"] > 0, 1), 0.0)
        out.append(pd.DataFrame({"season": s, "side": side, "game_date": d.index, "lg": lg}))
    return pd.concat(out, ignore_index=True)


def ratings(P: pd.DataFrame, LG: pd.DataFrame, k: dict, rho: dict, s2: float) -> pd.DataFrame:
    """As-of shrunk team PAE, centred on the league as-of mean of the same date.

    rating = (k * (lg + mu) + S) / (k + n) - lg,   mu = rho * prev_final_c
    v      = s2 / (k + n)                          (G-A1b form, sigma^2 / (D + k))
    S, n = residual sum / located attempts over the team's games strictly before
    this one in the season. prev_final_c = the team's previous-season FINAL
    centred PAE (raw season mean minus that season's league mean), 0 when absent."""
    P = P.merge(LG, on=["season", "side", "game_date"], how="left")
    g = P.groupby(["side", "season", "team_id"], sort=False)
    P["S"] = g["rsum"].cumsum() - P["rsum"]
    P["N"] = g["n"].cumsum() - P["n"]
    fin = P.groupby(["side", "season", "team_id"]).agg(rs=("rsum", "sum"), nn=("n", "sum")).reset_index()
    lgs = P.groupby(["side", "season"]).agg(lrs=("rsum", "sum"), lnn=("n", "sum")).reset_index()
    fin = fin.merge(lgs, on=["side", "season"])
    fin["final_c"] = np.where(fin["nn"] > 0, fin["rs"] / fin["nn"].where(fin["nn"] > 0, 1), 0.0) - fin["lrs"] / fin["lnn"]
    prev = fin[["side", "season", "team_id", "final_c"]].copy()
    prev["season"] = prev["season"] + 1
    P = P.merge(prev.rename(columns={"final_c": "prev_c"}), on=["side", "season", "team_id"], how="left")
    P["has_prev"] = P["prev_c"].notna()
    P["prev_c"] = P["prev_c"].fillna(0.0)
    kk = P["side"].map(k).to_numpy(dtype=float)
    rr = P["side"].map(rho).to_numpy(dtype=float)
    mu = rr * P["prev_c"].to_numpy()
    lg = P["lg"].to_numpy()
    P["rating"] = (kk * (lg + mu) + P["S"].to_numpy()) / (kk + P["N"].to_numpy()) - lg
    P["v"] = s2 / (kk + P["N"].to_numpy())
    return P


def choose_k_rho(P: pd.DataFrame, LG: pd.DataFrame, s2: float, t0) -> dict:
    """Fold-1 TRAINING season 2023 only (its prior is 2022's final PAE): per side,
    the (k, rho) minimising the located-attempt-weighted squared error of the
    as-of rating against the game's own realised centred mean residual,
    over games with >= 1 located attempt."""
    res = {}
    for side in ("off", "def"):
        tab = {}
        for k in K_GRID:
            for rho in RHO_GRID:
                R = ratings(P[P["side"] == side], LG, {side: k}, {side: rho}, s2)
                R = R[(R["season"] == 2023) & (R["n"] > 0)]
                obs = R["rsum"] / R["n"] - R["lg"]
                tab[f"{k}|{rho}"] = float(np.average((obs - R["rating"]) ** 2, weights=R["n"]))
        best = min(tab, key=tab.get)
        res[side] = {"grid_wmse": tab, "k": int(best.split("|")[0]), "rho": float(best.split("|")[1])}
        log(f"k/rho {side}: k={res[side]['k']} rho={res[side]['rho']}", t0)
    return res


def leak(Rt: pd.DataFrame) -> dict:
    """Change-form leak test (INV-45) on both ratings, team-grouped, margin from
    the VERIFIED finals (eval.reference.load_actual_games), per season and pooled."""
    from cbb_sim.analysis import leak_test as LK
    from cbb_sim.eval import reference as REF
    act = pd.concat([REF.load_actual_games(s)[["game_id", "home_score", "away_score", "home_team_id", "away_team_id"]]
                     for s in SEASONS], ignore_index=True).drop_duplicates("game_id")
    out = {}
    for side, col in (("off", "pae_off_c"), ("def", "pae_def_c")):
        p = Rt[Rt["side"] == side][["season", "game_id", "game_date", "team_id", "rating"]].rename(columns={"rating": col})
        p = p.merge(act, on="game_id", how="inner")
        home = (p["team_id"].astype(str) == p["home_team_id"].astype(str)).to_numpy()
        p["margin"] = np.where(home, p["home_score"] - p["away_score"], p["away_score"] - p["home_score"]).astype(float)
        p = p.rename(columns={"team_id": "team"})
        res = LK.run_leak_test(p, [col], team_col="team", season_col="season", date_col="game_date", margin_col="margin")
        out[col] = res.to_dict(orient="records")
        out[col + "__n_panel"] = int(len(p))
    return out


def ga1b(Rt: pd.DataFrame, s2: float) -> dict:
    """G-A1b(ii)-style check: SD of z = (obs_c - rating) / sqrt(v + s2 / n) over
    team-games with >= 10 located attempts, per side and season (reported only)."""
    out = {}
    for side in ("off", "def"):
        R = Rt[(Rt["side"] == side) & (Rt["n"] >= 10)]
        obs = R["rsum"] / R["n"] - R["lg"]
        z = (obs - R["rating"]) / np.sqrt(R["v"] + s2 / R["n"])
        out[side] = {str(s): round(float(z[R["season"] == s].std()), 4) for s in SEASONS}
        out[side]["pooled"] = round(float(z.std()), 4)
    return out


def build(a, t0) -> int:
    assert_not_sealed(SEASONS, context="PAE build")
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; versioned outputs are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    frames = [load_fga(s).assign(season=s) for s in SEASONS]
    fga = pd.concat(frames, ignore_index=True)
    des, rep = design_join(fga, t0)
    des["value"] = np.where(des["shot_class"].to_numpy() == "FGA_3", 3, 2)
    xf, yy = fold_xy(des["x"].to_numpy(), des["y"].to_numpy())
    des["xf"], des["y"] = xf, yy
    gsel = choose_grid(des, t0)
    grid = gsel["chosen"]
    shots, bmeta = baselines_and_residuals(des, grid)
    # sigma^2 of the per-shot residual: fold-1 training seasons only (2022-2023)
    s2 = float(np.nanvar(shots.loc[shots["season"].isin([2022, 2023]), "r"]))
    P = team_game_panel(shots)
    LG = league_asof(P)
    kr = choose_k_rho(P, LG, s2, t0)
    Rt = ratings(P, LG, {s: kr[s]["k"] for s in kr}, {s: kr[s]["rho"] for s in kr}, s2)
    Rt.to_parquet(out / "team_ratings_v1.parquet", index=False)
    lk = leak(Rt)
    z = ga1b(Rt, s2)
    # ---- row-aligned design columns
    off = Rt[Rt["side"] == "off"][["game_id", "team_id", "rating", "v", "N"]].rename(
        columns={"team_id": "off_team_id", "rating": "pae_off_c", "v": "pae_off_v", "N": "pae_off_n"})
    dfn = Rt[Rt["side"] == "def"][["game_id", "team_id", "rating", "v", "N"]].rename(
        columns={"team_id": "def_team_id", "rating": "pae_def_c", "v": "pae_def_v", "N": "pae_def_n"})
    cols = shots[["row", "game_id", "off_team_id", "def_team_id", "shooter_id", "shot_class", "made", "season",
                  "located", "join_matched", "p_cell", "cls_mean_prior"]].copy()
    cols = cols.merge(off, on=["game_id", "off_team_id"], how="left").merge(dfn, on=["game_id", "def_team_id"], how="left")
    cols = cols.sort_values("row").reset_index(drop=True)
    miss = int(cols[["pae_off_c", "pae_def_c"]].isna().any(axis=1).sum())
    if miss:
        raise SystemExit(f"{miss} design rows without a rating key (coverage miss raises; features.md s6)")
    cols["cell_p"] = np.where(cols["located"], cols["p_cell"], cols["cls_mean_prior"]).astype("float32")
    cols["loc_missing"] = (~cols["located"]).astype("float32")
    for c in ("pae_off_c", "pae_def_c", "pae_off_v", "pae_def_v", "pae_off_n", "pae_def_n"):
        cols[c] = cols[c].astype("float32")
    keep = ["row", "game_id", "shooter_id", "shot_class", "made", "season", "located", "join_matched",
            "pae_off_c", "pae_def_c", "pae_off_v", "pae_def_v", "pae_off_n", "pae_def_n", "cell_p", "loc_missing"]
    cp = out / "design_pae_cols_v1.parquet"
    cols[keep].to_parquet(cp, index=False)
    h = hashlib.sha256(cp.read_bytes()).hexdigest()
    desc = {s: {c: {k: round(float(v), 5) for k, v in cols.loc[cols["season"] == s, c].describe().items()}
                for c in ("pae_off_c", "pae_def_c", "cell_p")} for s in SEASONS}
    rep_all = {"created_at": pd.Timestamp.now("UTC").isoformat(), "design_join": rep, "grid": gsel,
               "baselines": bmeta, "residual_var_s2_fold1_train": s2, "k_rho": kr, "leak_test": lk,
               "ga1b_ii_z_sd": z, "design_cols_sha256": h, "design_cols_path": str(cp.relative_to(ROOT)),
               "column_describe_by_season": desc, "cell_smooth_a": CELL_SMOOTH_A}
    (out / "build_report_v1.json").write_text(json.dumps(rep_all, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out}/build_report_v1.json; design cols sha256 {h[:12]}", t0)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["coverage", "build"], required=True)
    ap.add_argument("--coverage-out", default=str(COVERAGE_OUT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    t0 = time.time()
    return coverage(a, t0) if a.part == "coverage" else build(a, t0)


if __name__ == "__main__":
    raise SystemExit(main())
