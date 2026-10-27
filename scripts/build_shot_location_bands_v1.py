#!/usr/bin/env python
"""build_shot_location_bands_v1.py -- shot-location stage round 1 (docs/models/shot_location/experiments.md
section 1; pre-registered at commit ea73296 BEFORE any fit).

Parts (run in this order):

  --part build   bands (section 1.2, fixed thresholds), the prior-season `da15` cell make rate (fg_make 31.2 recipe,
                 re-used from `build_pae_shot_rating_v1.py` unchanged; da15 is NOT re-chosen), the as-of team band
                 profiles `off_band_c` / `def_band_c` (31.2 recipe on band shares: shrunk to the as-of league share,
                 prior-season carry rho, separate k per side; k / rho chosen on fold-1 TRAINING seasons 2022-2023
                 by located-attempt-weighted squared error against the game's realised shares, before any arm),
                 the change-form leak test on both families (REFUSED on failure), the band shares per season and
                 month, and two row-aligned tables:
                   design_loc_cols_v1.parquet   aligned to fg_make design_v2_shotshooter (2.24M rows)
                   rebound_loc_cols_v1.parquet  aligned to rebound round3/design_round3 (1.54M rows): the band,
                                                 d and theta of a jump2 / three MISS, joined to the fg design row
  --part diag    the registered diagnostics of section 1.7 (rim override make vs miss; hoopR vs CBBD band shares on
                 2025 from 2025-02-10; band drift with the Sportradar reference; split-half reliability on this
                 round's own design table). Descriptive, not gated.

Season 2026 is SEALED and never opened.

    .venv/Scripts/python.exe scripts/build_shot_location_bands_v1.py --part build
    .venv/Scripts/python.exe scripts/build_shot_location_bands_v1.py --part diag
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
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
import build_pae_shot_rating_v1 as PAE  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]
OUT = ROOT / "data/processed/models/shot_location/round1"
REB_DESIGN = ROOT / "data/processed/models/rebound/round3/design_round3.parquet"
AUDIT = ROOT / "results/coordinate_audit"
GRID = "da15"                       # fixed by fg_make 31.2; NOT re-chosen
BANDS = {"FGA_jump2": ("short2", "long2"), "FGA_3": ("corner3", "wing3", "top3")}
ALL_BANDS = ("short2", "long2", "corner3", "wing3", "top3")
CK = {"FGA_jump2": "j", "FGA_3": "t"}
K_GRID = (50, 100, 200, 400, 800, 1600, 3200)
RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EPS = 1e-3


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ bands (section 1.2)
def assign_band(shot_class: np.ndarray, xf: np.ndarray, y: np.ndarray, located: np.ndarray):
    """Returns (band array of str or '', d ft, theta deg). theta = PAE.dist_angle convention:
    atan2(y - 250, xf - 52.5) in degrees; 0 = straight out from the basket toward half court, +/-90 = along the
    baseline, |theta| > 90 behind the basket."""
    d, th = PAE.dist_angle(xf, y)
    band = np.full(len(shot_class), "", dtype=object)
    j = located & (shot_class == "FGA_jump2")
    band[j & (d <= 10.0)] = "short2"
    band[j & (d > 10.0)] = "long2"
    t = located & (shot_class == "FGA_3")
    corner = t & (np.abs(y - 250.0) >= 210.0) & (d <= 24.0)
    band[corner] = "corner3"
    band[t & ~corner & (np.abs(th) >= 30.0)] = "wing3"
    band[t & ~corner & (np.abs(th) < 30.0)] = "top3"
    return band, d, th


# ------------------------------------------------------------------ as-of band profiles
def team_panel(des: pd.DataFrame) -> pd.DataFrame:
    """One row per (side, season, game, team): located attempts per band and per class."""
    L = des[des["band"] != ""]
    rows = []
    for side, col in (("off", "off_team_id"), ("def", "def_team_id")):
        g = pd.crosstab([L["season"], L["game_id"], L["game_date"], L[col]], L["band"])
        g = g.reindex(columns=list(ALL_BANDS), fill_value=0).reset_index().rename(columns={col: "team_id"})
        a = des.groupby(["season", "game_id", "game_date", col]).size().reset_index(name="n_fga").rename(
            columns={col: "team_id"})
        g = a.merge(g, on=["season", "game_id", "game_date", "team_id"], how="left")
        g[list(ALL_BANDS)] = g[list(ALL_BANDS)].fillna(0).astype("float64")
        g["side"] = side
        rows.append(g)
    P = pd.concat(rows, ignore_index=True)
    P["n_j"] = P["short2"] + P["long2"]
    P["n_t"] = P["corner3"] + P["wing3"] + P["top3"]
    return P.sort_values(["side", "season", "team_id", "game_date", "game_id"]).reset_index(drop=True)


def league_asof(P: pd.DataFrame) -> pd.DataFrame:
    """As-of league band share per (season, date): located attempts of games strictly BEFORE the date (offence
    side; the defence side is the same shots). When nothing precedes the date (season opener) the previous
    season's final league share is used; for 2022 (no earlier season) the 2022 final share is used -- this has NO
    effect on any rating because on that date S = N = 0 and prev_c = 0, so the centred value is exactly 0."""
    O = P[P["side"] == "off"]
    fin = O.groupby("season")[list(ALL_BANDS)].sum()
    out = []
    for s, g in O.groupby("season"):
        d = g.groupby("game_date")[list(ALL_BANDS)].sum().sort_index()
        cs = d.cumsum().shift(1).fillna(0.0)
        prev = fin.loc[s - 1] if (s - 1) in fin.index else fin.loc[s]
        for cls, bands in BANDS.items():
            n = cs[list(bands)].sum(axis=1)
            pn = float(prev[list(bands)].sum())
            for b in bands:
                cs["lg_" + b] = np.where(n > 0, cs[b] / n.where(n > 0, 1), float(prev[b]) / pn)
        cs = cs[["lg_" + b for b in ALL_BANDS]].reset_index()
        cs["season"] = s
        out.append(cs)
    return pd.concat(out, ignore_index=True)


def prev_final(P: pd.DataFrame) -> pd.DataFrame:
    """Previous-season FINAL centred share per (side, team): raw season share minus that season's league share."""
    f = P.groupby(["side", "season", "team_id"])[list(ALL_BANDS)].sum().reset_index()
    lgs = P[P["side"] == "off"].groupby("season")[list(ALL_BANDS)].sum()
    out = f[["side", "season", "team_id"]].copy()
    for cls, bands in BANDS.items():
        n = f[list(bands)].sum(axis=1)
        for b in bands:
            ln = lgs[list(bands)].sum(axis=1)
            lg = f["season"].map(lgs[b] / ln)
            out["prev_" + b] = np.where(n > 0, f[b] / n.where(n > 0, 1) - lg, 0.0)
    out["season"] = out["season"] + 1
    return out


def prepare(P: pd.DataFrame, LG: pd.DataFrame) -> pd.DataFrame:
    P = P.merge(LG, on=["season", "game_date"], how="left")
    g = P.groupby(["side", "season", "team_id"], sort=False)
    for b in (*ALL_BANDS, "n_j", "n_t"):
        P["S_" + b] = g[b].cumsum() - P[b]
    P = P.merge(prev_final(P), on=["side", "season", "team_id"], how="left")
    P["has_prev"] = P["prev_short2"].notna()
    for b in ALL_BANDS:
        P["prev_" + b] = P["prev_" + b].fillna(0.0)
    return P


def shares(P: pd.DataFrame, k: float, rho: float) -> dict[str, np.ndarray]:
    """As-of shrunk share per band: (k (lg + rho prev_c) + S_b) / (k + N_class)."""
    out = {}
    for cls, bands in BANDS.items():
        N = P["S_n_" + CK[cls]].to_numpy()
        for b in bands:
            lg = P["lg_" + b].to_numpy()
            out[b] = (k * (lg + rho * P["prev_" + b].to_numpy()) + P["S_" + b].to_numpy()) / (k + N)
    return out


def centred(P: pd.DataFrame, sh: dict) -> dict[str, np.ndarray]:
    lo = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))  # noqa: E731
    return {b: lo(sh[b]) - lo(P["lg_" + b].to_numpy()) for b in ALL_BANDS}


def choose_k_rho(P: pd.DataFrame, t0) -> dict:
    """Fold-1 TRAINING seasons 2022-2023 only, per side: located-attempt-weighted squared error of the as-of
    share against the game's realised share, summed over the bands of each class (weight = the game's located
    attempts in the class), over games with >= 1 located attempt in the class."""
    res = {}
    for side in ("off", "def"):
        Q = P[(P["side"] == side) & P["season"].isin([2022, 2023])]
        tab = {}
        for k in K_GRID:
            for rho in RHO_GRID:
                sh = shares(Q, k, rho)
                num = den = 0.0
                for cls, bands in BANDS.items():
                    n = Q["n_" + CK[cls]].to_numpy()
                    m = n > 0
                    for b in bands:
                        obs = Q[b].to_numpy()[m] / n[m]
                        num += float(np.sum(n[m] * (obs - sh[b][m]) ** 2))
                    den += float(n[m].sum())
                tab[f"{k}|{rho}"] = num / den
        order = sorted(tab, key=tab.get)
        res[side] = {"grid_wmse": tab, "k": int(order[0].split("|")[0]), "rho": float(order[0].split("|")[1]),
                     "second_k": int(order[1].split("|")[0]), "second_rho": float(order[1].split("|")[1])}
        log(f"k/rho {side}: best {order[0]} ({tab[order[0]]:.6f}), second {order[1]} ({tab[order[1]]:.6f})", t0)
    return res


def leak(P: pd.DataFrame, cols: dict) -> dict:
    """Change-form leak test (INV-45) per side on the four non-redundant band columns (long2 = -short2 exactly in
    log-odds), team-grouped, margin from the VERIFIED finals; per season and pooled."""
    from cbb_sim.analysis import leak_test as LK
    from cbb_sim.eval import reference as REF
    act = pd.concat([REF.load_actual_games(s)[["game_id", "home_score", "away_score", "home_team_id",
                                                "away_team_id"]] for s in SEASONS], ignore_index=True)
    act = act.drop_duplicates("game_id")
    out = {}
    for side in ("off", "def"):
        m = (P["side"] == side).to_numpy()
        p = P.loc[m, ["season", "game_id", "game_date", "team_id"]].copy()
        names = []
        for b in ("short2", "corner3", "wing3", "top3"):
            nm = f"{side}_{b}_c"
            p[nm] = cols[b][m]
            names.append(nm)
        p = p.merge(act, on="game_id", how="inner")
        home = (p["team_id"].astype(str) == p["home_team_id"].astype(str)).to_numpy()
        p["margin"] = np.where(home, p["home_score"] - p["away_score"],
                               p["away_score"] - p["home_score"]).astype(float)
        p = p.rename(columns={"team_id": "team"})
        r = LK.run_leak_test(p, names, team_col="team", season_col="season", date_col="game_date",
                             margin_col="margin")
        out[side] = r.to_dict(orient="records")
        out[side + "__n_panel"] = int(len(p))
        out[side + "__family_verdict"] = "LEAK" if (r["verdict"] == "LEAK").any() else "PASS"
    out["refused"] = any(out[s + "__family_verdict"] == "LEAK" for s in ("off", "def"))
    return out


# ------------------------------------------------------------------ rebound link
def rebound_link(des: pd.DataFrame, t0) -> tuple[pd.DataFrame, dict]:
    """Row-aligned to the served rebound design (round3): for a jump2 / three MISS opportunity, the fg design row
    of the miss. Key (cbbd game, period, seconds remaining, offence team, class, blocked) + ordinal within the key
    in event order (both tables preserve the stream's event order)."""
    R = pd.read_parquet(REB_DESIGN, columns=["cbbd_game_id", "season", "period", "seconds_remaining",
                                             "off_team_id", "miss_type", "blocked"])
    R = R.reset_index(drop=True)
    R["row"] = np.arange(len(R))
    cls = R["miss_type"].map({"jump2": "FGA_jump2", "three": "FGA_3"})
    Rk = pd.DataFrame({"g": R["cbbd_game_id"].astype("int64"), "per": R["period"].astype("int64"),
                       "sec": R["seconds_remaining"].astype("int64"), "team": R["off_team_id"].astype(str),
                       "cls": cls, "blk": R["blocked"].astype(bool), "row": R["row"]})
    Rk = Rk[Rk["cls"].notna()]
    m = (~des["made"].astype(bool)) & des["shot_class"].isin(["FGA_jump2", "FGA_3"])
    D = des.loc[m]
    Dk = pd.DataFrame({"g": D["cbbd_game_id"].astype("int64").to_numpy(), "per": D["period"].astype("int64").to_numpy(),
                       "sec": D["seconds_remaining"].astype("int64").to_numpy(),
                       "team": D["off_team_id"].astype(str).to_numpy(), "cls": D["shot_class"].to_numpy(),
                       "blk": D["blocked"].astype(bool).to_numpy(), "fg_row": D.index.to_numpy()})
    keys = ["g", "per", "sec", "team", "cls", "blk"]
    Rk["ord"] = Rk.groupby(keys).cumcount()
    Dk["ord"] = Dk.groupby(keys).cumcount()
    J = Rk.merge(Dk, on=[*keys, "ord"], how="left")
    out = pd.DataFrame({"row": R["row"], "miss_type": R["miss_type"], "season": R["season"]})
    out["fg_row"] = -1
    ok = J["fg_row"].notna()
    out.loc[J.loc[ok, "row"].to_numpy(), "fg_row"] = J.loc[ok, "fg_row"].astype("int64").to_numpy()
    fr = out["fg_row"].to_numpy()
    has = fr >= 0
    for c, fill, dt in (("band", "", object), ("d", np.nan, "float64"), ("theta", np.nan, "float64"),
                        ("located", False, bool)):
        v = np.full(len(out), fill, dtype=dt)
        v[has] = des[c].to_numpy()[fr[has]]
        out[c] = v
    jt = out["miss_type"].isin(["jump2", "three"]).to_numpy()
    rep = {"n_rebound_rows": int(len(out)), "n_jump2_three_miss_rows": int(jt.sum()),
           "matched_pct": round(100 * float(has[jt].mean()), 3),
           "located_pct_of_matched": round(100 * float(out["located"].to_numpy()[jt & has].astype(bool).mean()), 3),
           "matched_pct_by_season": {int(s): round(100 * float(has[jt & (out["season"] == s).to_numpy()].mean()), 3)
                                     for s in SEASONS}}
    log(f"rebound link: {rep}", t0)
    return out, rep


# ------------------------------------------------------------------ build
def build(a, t0) -> int:
    assert_not_sealed(SEASONS, context="shot_location build")
    if (OUT / "design_loc_cols_v1.parquet").exists():
        raise SystemExit(f"{OUT} already holds the build; versioned outputs are never overwritten")
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {"created_at": pd.Timestamp.now("UTC").isoformat(), "spec": "experiments.md section 1 (ea73296)"}
    fga = pd.concat([PAE.load_fga(s).assign(season=s) for s in SEASONS], ignore_index=True)
    log(f"loaded {len(fga):,} FGA rows", t0)
    des, jrep = PAE.design_join(fga, t0)
    del fga
    rep["design_join"] = jrep
    extra = pd.read_parquet(PAE.DESIGN, columns=["blocked"])
    des["blocked"] = extra["blocked"].to_numpy()
    des["value"] = np.where(des["shot_class"].to_numpy() == "FGA_3", 3, 2)
    xf, yy = PAE.fold_xy(des["x"].to_numpy(), des["y"].to_numpy())
    des["xf"], des["y"] = xf, yy
    loc = des["located"].to_numpy().astype(bool)
    band, d, th = assign_band(des["shot_class"].to_numpy(), xf, yy, loc)
    des["band"], des["d"], des["theta"] = band, np.where(loc, d, np.nan), np.where(loc, th, np.nan)
    # prior-season da15 cell make rate (31.2 recipe, PAE code unchanged)
    shots, bmeta = PAE.baselines_and_residuals(des, GRID)
    des["cell_p"] = shots["p_cell"].to_numpy()
    rep["cell_baselines"] = {str(k): v for k, v in bmeta.items()}
    chk = pd.read_parquet(ROOT / "data/processed/models/fg_make/pae_v1/design_pae_cols_v1.parquet",
                          columns=["cell_p", "located"])
    lm = loc & chk["located"].to_numpy().astype(bool)
    rep["cell_p_vs_pae_table_max_abs_diff_located"] = float(np.nanmax(np.abs(des["cell_p"].to_numpy()[lm]
                                                                             - chk["cell_p"].to_numpy()[lm])))
    log(f"cell_p reproduces the PAE table: max abs diff {rep['cell_p_vs_pae_table_max_abs_diff_located']}", t0)
    # band shares per season (pooled-training view is computed by the grader / diag)
    sh = des[des["band"] != ""].groupby(["season", "shot_class", "band"]).size().rename("n").reset_index()
    sh["share"] = sh["n"] / sh.groupby(["season", "shot_class"])["n"].transform("sum")
    rep["band_shares_by_season"] = sh.to_dict(orient="records")
    # as-of band profiles
    P0 = team_panel(des)
    P = prepare(P0, league_asof(P0))
    kr = choose_k_rho(P, t0)
    rep["k_rho"] = kr
    cols = {}
    for tag, kk, rr in (("", "k", "rho"), ("_alt", "second_k", "second_rho")):
        for side in ("off", "def"):
            m = (P["side"] == side).to_numpy()
            s_ = shares(P[m], kr[side][kk], kr[side][rr])
            c_ = centred(P[m], s_)
            for b in ALL_BANDS:
                key = f"{side}_{b}_c{tag}"
                if key not in P:
                    P[key] = np.nan
                P.loc[m, key] = c_[b]
                if tag == "":
                    P.loc[m, f"{side}_{b}_share"] = s_[b]
    for b in ALL_BANDS:
        cols[b] = np.where(P["side"] == "off", P[f"off_{b}_c"], P[f"def_{b}_c"])
    lk = leak(P, cols)
    rep["leak_test"] = lk
    log(f"leak test: off {lk['off__family_verdict']} def {lk['def__family_verdict']} -> refused={lk['refused']}", t0)
    keep = ["side", "season", "game_id", "game_date", "team_id", "n_fga", *ALL_BANDS, "n_j", "n_t", "S_n_j",
            "S_n_t", "has_prev"] + [c for c in P.columns if c.startswith(("off_", "def_")) or c.startswith("lg_")]
    P[keep].to_parquet(OUT / "team_band_profiles_v1.parquet", index=False)
    # row-aligned design columns
    out = des[["game_id", "shooter_id", "shot_class", "made", "season", "game_date", "off_team_id", "def_team_id",
               "located", "xf", "y", "d", "theta", "band", "cell_p"]].copy()
    for side, col in (("off", "off_team_id"), ("def", "def_team_id")):
        Q = P[P["side"] == side][["game_id", "team_id", "S_n_j", "S_n_t",
                                  *[f"{side}_{b}_c" for b in ALL_BANDS], *[f"{side}_{b}_c_alt" for b in ALL_BANDS],
                                  *[f"{side}_{b}_share" for b in ALL_BANDS]]]
        Q = Q.rename(columns={"team_id": col, "S_n_j": f"{side}_n_j", "S_n_t": f"{side}_n_t"})
        n0 = len(out)
        out = out.merge(Q, on=["game_id", col], how="left")
        assert len(out) == n0
    miss = int(out[[f"off_{b}_c" for b in ALL_BANDS] + [f"def_{b}_c" for b in ALL_BANDS]].isna().any(axis=1).sum())
    rep["design_rows_without_profile"] = miss
    if miss:
        raise SystemExit(f"{miss} design rows without a band profile")
    for b in ALL_BANDS:
        out[f"b_{b}"] = (out["band"] == b).astype("float32")
    out.to_parquet(OUT / "design_loc_cols_v1.parquet", index=False)
    rep["design_loc_cols_sha256"] = sha(OUT / "design_loc_cols_v1.parquet")
    log(f"wrote design_loc_cols_v1 ({len(out):,} rows)", t0)
    # rebound link (needs the pbp keys of the design)
    keys = pd.read_parquet(PAE.DESIGN, columns=["cbbd_game_id", "period", "seconds_remaining", "made", "shot_class",
                                                "off_team_id", "blocked"])
    for c in ("band", "d", "theta", "located"):
        keys[c] = out[c].to_numpy()
    rb, rrep = rebound_link(keys, t0)
    rb.to_parquet(OUT / "rebound_loc_cols_v1.parquet", index=False)
    rep["rebound_link"] = rrep
    (OUT / "build_report_v1.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    log("wrote build_report_v1.json", t0)
    return 0


# ------------------------------------------------------------------ diagnostics (section 1.7)
def split_half(des: pd.DataFrame) -> list[dict]:
    """Registered method (gamezone_frame_calibration section 2): team-season, games by date split odd / even,
    each half >= 20 attempts of the class, shares demeaned by season, Spearman-Brown on r; team-seasons with
    >= 200 located attempts in the class."""
    L = des[des["band"] != ""]
    out = []
    for side, col in (("offence", "off_team_id"), ("defence", "def_team_id")):
        for cls, bands in BANDS.items():
            C = L[L["shot_class"] == cls]
            g = C.groupby(["season", col, "game_date", "game_id", "band"]).size().unstack(fill_value=0)
            g = g.reindex(columns=list(bands), fill_value=0).reset_index().sort_values(["season", col, "game_date"])
            g["half"] = g.groupby(["season", col]).cumcount() % 2
            h = g.groupby(["season", col, "half"])[list(bands)].sum()
            h["n"] = h[list(bands)].sum(axis=1)
            tot = h.groupby(level=[0, 1])["n"].sum()
            for b in bands[:-1] if cls == "FGA_jump2" else bands:
                w = (h[b] / h["n"]).unstack("half")
                n = h["n"].unstack("half")
                ok = (n[0] >= 20) & (n[1] >= 20) & (tot.reindex(n.index) >= 200)
                w = w[ok]
                w = w - w.groupby(level=0).transform("mean")
                r = float(np.corrcoef(w[0], w[1])[0, 1])
                out.append({"side": side, "class": cls, "band": b, "team_seasons": int(ok.sum()),
                            "split_half_r": round(r, 3), "spearman_brown": round(2 * r / (1 + r), 3)})
    return out


def diag(a, t0) -> int:
    assert_not_sealed(SEASONS, context="shot_location diag")
    rep: dict = {"created_at": pd.Timestamp.now("UTC").isoformat()}
    cols = pd.read_parquet(OUT / "design_loc_cols_v1.parquet",
                           columns=["game_id", "season", "game_date", "shot_class", "band", "off_team_id",
                                    "def_team_id", "made", "located"])
    # 4. split-half reliability on the round's own design table
    rep["split_half"] = split_half(cols)
    log("split-half done", t0)
    # 3. drift by season and month
    L = cols[cols["band"] != ""].copy()
    L["month"] = pd.to_datetime(L["game_date"]).dt.month
    dm = L.groupby(["season", "month", "shot_class", "band"]).size().rename("n").reset_index()
    dm["share"] = dm["n"] / dm.groupby(["season", "month", "shot_class"])["n"].transform("sum")
    rep["drift_by_season_month"] = dm.to_dict(orient="records")
    # 1. rim override (JumpShot within 2.27 ft -> FGA_rim) for makes vs misses within game, all FGA from the
    # coordinate-audit master tables (CBBD playType, same classifier); paired within game, game bootstrap
    rows = []
    rng = np.random.default_rng(20261027)
    for s in SEASONS:
        m = pd.read_parquet(AUDIT / f"master_{s}.parquet", columns=["gameId", "playType", "cls", "cls0", "made",
                                                                    "located"])
        js = m[(m["playType"] == "JumpShot") & m["located"]]
        js = js.assign(ovr=((js["cls"] == "rim") & (js["cls0"] != "rim")).astype(float))
        g = js.groupby(["gameId", "made"])["ovr"].agg(["sum", "size"]).unstack("made")
        g = g.dropna()
        rm = g[("sum", 1.0)] / g[("size", 1.0)]
        rx = g[("sum", 0.0)] / g[("size", 0.0)]
        dd = (rm - rx).to_numpy()
        w = (g[("size", 1.0)] + g[("size", 0.0)]).to_numpy()
        est = float(np.average(dd, weights=w))
        reps = []
        for _ in range(200):
            k = rng.integers(0, len(dd), len(dd))
            reps.append(np.average(dd[k], weights=w[k]))
        rows.append({"season": s, "jumpshot_located": int(len(js)),
                     "override_rate_makes_pct": round(100 * float(js.loc[js["made"] == 1, "ovr"].mean()), 3),
                     "override_rate_misses_pct": round(100 * float(js.loc[js["made"] == 0, "ovr"].mean()), 3),
                     "within_game_diff_pp": round(100 * est, 3), "boot_se_pp": round(100 * float(np.std(reps, ddof=1)), 3),
                     "games": int(len(dd))})
    rep["rim_override_make_vs_miss"] = rows
    log(f"rim override: {rows}", t0)
    # 2. hoopR vs CBBD band shares, 2025 from 2025-02-10 (descriptive)
    m = pd.read_parquet(AUDIT / "master_2025.parquet")
    m = m[(m["game_date"] >= "2025-02-10") & m["cls"].isin(["jump2", "three"])].copy()
    cls = m["cls"].map({"jump2": "FGA_jump2", "three": "FGA_3"}).to_numpy()
    bc, _, _ = assign_band(cls, m["xf"].to_numpy(dtype=float), m["y"].to_numpy(dtype=float),
                           m["located"].to_numpy(dtype=bool))
    hx = m["hx"].to_numpy(dtype=float)
    hy = m["hy"].to_numpy(dtype=float)
    hxf = (47.0 - np.abs(hx)) * 10.0
    hyy = 250.0 + hy * 10.0
    hl = m["h_located"].to_numpy(dtype=bool) & ~m["h_sentinel"].fillna(False).to_numpy(dtype=bool)
    bh, _, _ = assign_band(cls, hxf, hyy, hl)
    m["band_c"], m["band_h"] = bc, bh
    both = m[(m["band_c"] != "") & (m["band_h"] != "")]
    agree = []
    for c in ("jump2", "three"):
        b = both[both["cls"] == c]
        agree.append({"class": c, "pairs": int(len(b)), "band_differs_pct": round(100 * float((b["band_c"] != b["band_h"]).mean()), 3),
                      "confusion": pd.crosstab(b["band_c"], b["band_h"]).to_dict()})
    tg = []
    for src, col in (("cbbd", "band_c"), ("hoopr", "band_h")):
        x = both.groupby(["gameId", "teamId", col]).size().unstack(fill_value=0)
        tg.append(x.add_prefix(src + "_"))
    T = tg[0].join(tg[1], how="inner")
    tgr = []
    for c, bands in (("jump2", ("short2", "long2")), ("three", ("corner3", "wing3", "top3"))):
        for src in ("cbbd", "hoopr"):
            for b in bands:
                if f"{src}_{b}" not in T:
                    T[f"{src}_{b}"] = 0
        nc = sum(T[f"cbbd_{b}"] for b in bands)
        ok = nc >= 5
        for b in bands:
            sc = T.loc[ok, f"cbbd_{b}"] / nc[ok]
            nh = sum(T.loc[ok, f"hoopr_{b2}"] for b2 in bands)
            shh = T.loc[ok, f"hoopr_{b}"] / nh
            tgr.append({"class": c, "band": b, "team_games": int(ok.sum()),
                        "mean_share_cbbd": round(float(sc.mean()), 4), "mean_share_hoopr": round(float(shh.mean()), 4),
                        "mae_share_pp": round(100 * float((sc - shh).abs().mean()), 3),
                        "corr": round(float(np.corrcoef(sc, shh)[0, 1]), 4)})
    rep["hoopr_vs_cbbd_2025_from_0210"] = {"attempt_agreement": agree, "team_game_shares": tgr,
                                           "rows_in_window": int(len(m)), "both_located": int(len(both))}
    log(f"hoopR vs CBBD: {agree[0]['band_differs_pct']} / {agree[1]['band_differs_pct']} % band differs", t0)
    (OUT / "diag_report_v1.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    log("wrote diag_report_v1.json", t0)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["build", "diag"], required=True)
    a = ap.parse_args()
    t0 = time.time()
    return build(a, t0) if a.part == "build" else diag(a, t0)


if __name__ == "__main__":
    raise SystemExit(main())
