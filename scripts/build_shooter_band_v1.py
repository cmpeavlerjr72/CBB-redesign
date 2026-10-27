#!/usr/bin/env python
"""build_shooter_band_v1.py -- shot-location round 2, block P (docs/models/shot_location/experiments.md section 3.2;
PRE-REGISTERED at commit 5f6d6b0 BEFORE any fit).

Builds, before any arm is fitted:
  1. split-half (odd / even games, Spearman-Brown) reliability of (a) the shooter's band shares within class and (b)
     the shooter's make rate by band, over shooter-seasons with >= 100 located attempts in the class (40-99 are
     reported separately and labelled UNDERPOWERED);
  2. the as-of shooter features
       sh_<band>_c    the shooter's centred shrunk band log-odds within class; prior = the shooter's team profile
                      (round-1 `off_<band>_share`), then league (`lg_*`): p = (k_s T_b + S_b) / (k_s + N)
       smk_<band>     the shooter's centred shrunk make deviation WITHIN band; prior = his overall
                      `shooter_shrunk_dev_c` (served; m = 200 jumper / 300 three): (k_m prior + D_b) / (k_m + S_b)
       sh_band_make_c smk of the attempt's OWN band (located attempts)
     S_b / D_b = the shooter's located counts / sums of (made - as-of league band make rate) over games STRICTLY
     BEFORE this one in the season plus RHO x the previous season's final totals (RHO = 0.5, the round-1 team value,
     not re-chosen). k_s, k_m are chosen per class on fold-1 TRAINING seasons (2022-2023) by located-attempt-weighted
     squared error, before any arm;
  3. the change-form leak test (team-game mean over the team's attempts vs verified own-game margin, |corr| <= 0.15),
     REFUSED on failure;
  4. the CBBD-shooter -> hoopR-athlete map (`player_crosswalk_v2`, key (season, cbbd_player_id) -> espn_athlete_id)
     and its join rates.
Season 2026 is sealed and never opened.

    .venv/Scripts/python.exe scripts/build_shooter_band_v1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]
R1 = ROOT / "data/processed/models/shot_location/round1"
OUT = ROOT / "data/processed/models/shot_location/round2"
DESIGN = ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet"
CROSSWALK = ROOT / "data/processed/player_crosswalk_v2.parquet"
M_FIT = ROOT / "data/processed/models/fg_make/round4/m_fitted.json"
BANDS = {"FGA_jump2": ("short2", "long2"), "FGA_3": ("corner3", "wing3", "top3")}
ALL = ("short2", "long2", "corner3", "wing3", "top3")
K_S_GRID = (10, 20, 50, 100, 200, 400, 800)
K_M_GRID = (25, 50, 100, 200, 400, 800, 1600)
RHO = 0.5
K0_LEAGUE_MAKE = 500.0
EPS = 1e-3


def log(m, t0):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def lo(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ league as-of make rate per band
def league_make_asof(D: pd.DataFrame) -> pd.DataFrame:
    """(season, game_date) -> lgmk_<band>: (K0 prev_final + cum made before date) / (K0 + cum n before date).
    Season opener: previous season's final (2022: its own final, no effect on anything graded)."""
    L = D[D["band"] != ""]
    g = L.groupby(["season", "game_date", "band"])["made"].agg(["sum", "size"]).reset_index()
    fin = L.groupby(["season", "band"])["made"].mean().unstack()
    out = []
    for s in SEASONS:
        gs = g[g["season"] == s]
        dates = np.sort(gs["game_date"].unique())
        row = {"season": s, "game_date": dates}
        for b in ALL:
            gb = gs[gs["band"] == b].set_index("game_date")
            n = gb["size"].reindex(dates).fillna(0).cumsum().shift(1).fillna(0).to_numpy()
            m = gb["sum"].reindex(dates).fillna(0).cumsum().shift(1).fillna(0).to_numpy()
            prev = fin.loc[s - 1, b] if (s - 1) in fin.index else fin.loc[s, b]
            row["lgmk_" + b] = (K0_LEAGUE_MAKE * prev + m) / (K0_LEAGUE_MAKE + n)
        out.append(pd.DataFrame(row))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ shooter panel
def build_panel(D: pd.DataFrame, LGM: pd.DataFrame, prof: pd.DataFrame, t0) -> pd.DataFrame:
    """One row per (shooter, class, game) over ALL jump2 / three attempts (located or not)."""
    D = D.merge(LGM, on=["season", "game_date"], how="left")
    L = D[D["band"] != ""].copy()
    L["dev"] = L["made"].astype("float64") - np.select([L["band"] == b for b in ALL],
                                                        [L["lgmk_" + b] for b in ALL])
    key = ["shooter_id", "shot_class", "season", "game_id", "game_date"]
    keyp = D.groupby(key, sort=False).agg(off_team_id=("off_team_id", "first"), prior=("prior", "first"),
                                         n_all=("made", "size")).reset_index()
    cnt = L.groupby(key + ["band"]).agg(n=("made", "size"), d=("dev", "sum")).reset_index()
    for b in ALL:
        x = cnt[cnt["band"] == b][key + ["n", "d"]].rename(columns={"n": "g_n_" + b, "d": "g_d_" + b})
        keyp = keyp.merge(x, on=key, how="left")
        keyp[["g_n_" + b, "g_d_" + b]] = keyp[["g_n_" + b, "g_d_" + b]].fillna(0.0)
    keyp = keyp.sort_values(["shooter_id", "shot_class", "season", "game_date", "game_id"]).reset_index(drop=True)
    grp = keyp.groupby(["shooter_id", "shot_class", "season"], sort=False)
    for b in ALL:
        for p in ("n", "d"):
            keyp[f"c_{p}_{b}"] = grp[f"g_{p}_{b}"].cumsum() - keyp[f"g_{p}_{b}"]
    tot = grp[[f"g_{p}_{b}" for b in ALL for p in ("n", "d")]].sum().reset_index()
    tot["season"] = tot["season"] + 1
    tot = tot.rename(columns={c: "p_" + c[2:] for c in tot.columns if c.startswith("g_")})
    keyp = keyp.merge(tot, on=["shooter_id", "shot_class", "season"], how="left")
    for b in ALL:
        for p in ("n", "d"):
            keyp[f"p_{p}_{b}"] = keyp[f"p_{p}_{b}"].fillna(0.0)
            keyp[f"S{p}_{b}"] = keyp[f"c_{p}_{b}"] + RHO * keyp[f"p_{p}_{b}"]
    # team prior shares + league shares
    tp = prof[prof["side"] == "off"][["game_id", "team_id", *[f"off_{b}_share" for b in ALL]]].rename(
        columns={"team_id": "off_team_id", **{f"off_{b}_share": "T_" + b for b in ALL}})
    keyp = keyp.merge(tp, on=["game_id", "off_team_id"], how="left")
    lg = prof[prof["side"] == "off"].drop_duplicates(["season", "game_date"])[["season", "game_date",
                                                                              *[f"lg_{b}" for b in ALL]]]
    keyp = keyp.merge(lg, on=["season", "game_date"], how="left")
    keyp = keyp.merge(LGM, on=["season", "game_date"], how="left")
    miss = keyp[["T_" + b for b in ALL] + ["lg_" + b for b in ALL]].isna().any(axis=1).sum()
    assert miss == 0, f"{miss} panel rows without team/league prior"
    log(f"panel: {len(keyp):,} shooter-class-games", t0)
    return keyp


def features(P: pd.DataFrame, k_s: dict, k_m: dict) -> dict:
    """k_s / k_m: per class (FGA_jump2 / FGA_3). Returns dict band -> (sh_c, smk) arrays on panel rows (0 where the
    band does not belong to the row's class)."""
    out = {}
    for cls, bands in BANDS.items():
        m = (P["shot_class"] == cls).to_numpy()
        N = sum(P["Sn_" + b].to_numpy() for b in bands)
        for b in bands:
            sh = np.zeros(len(P)); mk = np.zeros(len(P))
            T = P["T_" + b].to_numpy()[m]
            p = (k_s[cls] * T + P["Sn_" + b].to_numpy()[m]) / (k_s[cls] + N[m])
            sh[m] = lo(p) - lo(P["lg_" + b].to_numpy()[m])
            mk[m] = (k_m[cls] * P["prior"].to_numpy()[m] + P["Sd_" + b].to_numpy()[m]) / (k_m[cls] + P["Sn_" + b].to_numpy()[m])
            out[b] = (sh, mk)
    return out


def choose_k(P: pd.DataFrame, A: pd.DataFrame, t0) -> tuple[dict, dict, dict]:
    """Fold-1 TRAINING seasons (2022-2023), per class, located-attempt-weighted squared error. A: located attempts
    with panel row index `pi`, band, made, lgmk of the own band."""
    k_s, k_m, rep = {}, {}, {}
    for cls, bands in BANDS.items():
        a = A[(A["shot_class"] == cls) & A["season"].isin([2022, 2023])]
        pi = a["pi"].to_numpy()
        res_s, res_m = {}, {}
        for k in K_S_GRID:
            ks = {c: 100 for c in BANDS}; ks[cls] = k
            f = features(P, ks, {c: 100 for c in BANDS})
            err = 0.0
            for b in bands:
                tgt = (a["band"].to_numpy() == b).astype(float)
                pb = 1 / (1 + np.exp(-(f[b][0][pi] + lo(P["lg_" + b].to_numpy()[pi]))))
                pb = np.clip(pb, EPS, 1 - EPS)
                err += np.sum((tgt - pb) ** 2)
            res_s[k] = err / len(a)
        for k in K_M_GRID:
            km = {c: 100 for c in BANDS}; km[cls] = k
            f = features(P, {c: 100 for c in BANDS}, km)
            pred = np.zeros(len(a))
            for b in bands:
                mb = a["band"].to_numpy() == b
                pred[mb] = a["lgmk"].to_numpy()[mb] + f[b][1][pi[mb]]
            res_m[k] = float(np.mean((a["made"].to_numpy().astype(float) - pred) ** 2))
        k_s[cls] = min(res_s, key=res_s.get)
        k_m[cls] = min(res_m, key=res_m.get)
        rep[cls] = {"n_train_located": int(len(a)), "share_wmse": {str(k): v for k, v in res_s.items()},
                    "make_wmse": {str(k): v for k, v in res_m.items()}, "k_s": k_s[cls], "k_m": k_m[cls]}
        log(f"{cls}: k_s {k_s[cls]} k_m {k_m[cls]}", t0)
    return k_s, k_m, rep


# ------------------------------------------------------------------ reliability (registered first)
def sb(r):
    return 2 * r / (1 + r)


def reliability(D: pd.DataFrame) -> dict:
    L = D[D["band"] != ""].copy()
    rows = []
    for cls, bands in BANDS.items():
        C = L[L["shot_class"] == cls].copy()
        C = C.sort_values(["shooter_id", "season", "game_date", "game_id"])
        kk = C[["shooter_id", "season"]].to_numpy()
        newg = np.ones(len(C), dtype=np.int64)
        same = (kk[1:] == kk[:-1]).all(axis=1) & (C["game_id"].to_numpy()[1:] == C["game_id"].to_numpy()[:-1])
        newg[1:] = (~same).astype(np.int64)
        C["_ng"] = newg
        gi = C.groupby(["shooter_id", "season"])["_ng"].cumsum() - 1
        C["half"] = gi.to_numpy() % 2
        tot = C.groupby(["shooter_id", "season"]).size().rename("tot")
        cells = {}
        for b in bands:
            cells[b] = C.assign(isb=(C["band"] == b).astype(float), mk=C["made"].astype(float))
        # share reliability
        h = C.groupby(["shooter_id", "season", "half"]).size().unstack(fill_value=0)
        for nm, grp in (("share", bands),):
            for b in grp:
                hb = C[C["band"] == b].groupby(["shooter_id", "season", "half"]).size().unstack(fill_value=0)
                hb = hb.reindex(h.index, fill_value=0)
                for lab, lo_, hi_ in (("powered(>=100)", 100, 10 ** 9), ("UNDERPOWERED(40-99)", 40, 99)):
                    t = tot.reindex(h.index)
                    ok = (h[0] >= 20) & (h[1] >= 20) & (t >= lo_) & (t <= hi_)
                    s0 = (hb[0] / h[0])[ok]; s1 = (hb[1] / h[1])[ok]
                    s0 = s0 - s0.groupby(level=1).transform("mean"); s1 = s1 - s1.groupby(level=1).transform("mean")
                    r = float(np.corrcoef(s0, s1)[0, 1]) if ok.sum() > 3 else float("nan")
                    rows.append({"quantity": "band_share", "class": cls, "band": b, "stratum": lab,
                                 "shooter_seasons": int(ok.sum()), "split_half_r": r, "spearman_brown": sb(r)})
        # make-rate reliability by band cell
        cellsets = {"FGA_jump2": {"short2": ["short2"], "long2": ["long2"]},
                    "FGA_3": {"corner3": ["corner3"], "noncorner3": ["wing3", "top3"]}}[cls]
        t = tot
        res = {}
        for nm, bl in cellsets.items():
            x = C[C["band"].isin(bl)].groupby(["shooter_id", "season", "half"])["made"].agg(["sum", "size"])
            n_ = x["size"].unstack(fill_value=0).reindex(h.index, fill_value=0)
            m_ = x["sum"].unstack(fill_value=0).reindex(h.index, fill_value=0)
            res[nm] = (n_, m_)
        for nm, (n_, m_) in res.items():
            for lab, lo_, hi_ in (("powered(>=100)", 100, 10 ** 9), ("UNDERPOWERED(40-99)", 40, 99)):
                tt = t.reindex(n_.index)
                ok = (n_[0] >= 10) & (n_[1] >= 10) & (tt >= lo_) & (tt <= hi_)
                a0 = (m_[0] / n_[0])[ok]; a1 = (m_[1] / n_[1])[ok]
                a0 = a0 - a0.groupby(level=1).transform("mean"); a1 = a1 - a1.groupby(level=1).transform("mean")
                r = float(np.corrcoef(a0, a1)[0, 1]) if ok.sum() > 3 else float("nan")
                rows.append({"quantity": "make_rate", "class": cls, "band": nm, "stratum": lab,
                             "shooter_seasons": int(ok.sum()), "split_half_r": r, "spearman_brown": sb(r)})
        # contrast of the two cells
        (nA, mA), (nB, mB) = list(res.values())
        names = list(res.keys())
        for lab, lo_, hi_ in (("powered(>=100)", 100, 10 ** 9), ("UNDERPOWERED(40-99)", 40, 99)):
            tt = t.reindex(nA.index)
            ok = (nA[0] >= 10) & (nA[1] >= 10) & (nB[0] >= 10) & (nB[1] >= 10) & (tt >= lo_) & (tt <= hi_)
            c0 = (mA[0] / nA[0] - mB[0] / nB[0])[ok]; c1 = (mA[1] / nA[1] - mB[1] / nB[1])[ok]
            c0 = c0 - c0.groupby(level=1).transform("mean"); c1 = c1 - c1.groupby(level=1).transform("mean")
            r = float(np.corrcoef(c0, c1)[0, 1]) if ok.sum() > 3 else float("nan")
            rows.append({"quantity": "make_rate_contrast", "class": cls, "band": f"{names[0]} - {names[1]}",
                         "stratum": lab, "shooter_seasons": int(ok.sum()), "split_half_r": r,
                         "spearman_brown": sb(r)})
    return {"rows": rows, "n_shooter_seasons_ge100": {
        cls: int((L[L["shot_class"] == cls].groupby(["shooter_id", "season"]).size() >= 100).sum()) for cls in BANDS},
        "n_shooter_seasons_lt100": {
        cls: int((L[L["shot_class"] == cls].groupby(["shooter_id", "season"]).size() < 100).sum()) for cls in BANDS}}


# ------------------------------------------------------------------ leak test
def leak(D: pd.DataFrame, cols: pd.DataFrame) -> dict:
    from cbb_sim.analysis import leak_test as LK
    from cbb_sim.eval import reference as REF
    act = pd.concat([REF.load_actual_games(s)[["game_id", "home_score", "away_score", "home_team_id", "away_team_id"]]
                     for s in SEASONS], ignore_index=True).drop_duplicates("game_id")
    base = D[["season", "game_id", "game_date", "off_team_id", "shot_class", "band"]].copy()
    for c in cols.columns:
        base[c] = cols[c].to_numpy()
    g = {}
    for nm, cls in (("sh_short2_c", "FGA_jump2"), ("sh_corner3_c", "FGA_3"), ("sh_wing3_c", "FGA_3"),
                    ("sh_top3_c", "FGA_3")):
        x = base[base["shot_class"] == cls].groupby(["season", "game_id", "game_date", "off_team_id"])[nm].mean()
        g[nm] = x
    x = base[base["band"] != ""].groupby(["season", "game_id", "game_date", "off_team_id"])["sh_band_make_c"].mean()
    g["sh_band_make_c"] = x
    p = pd.concat(g, axis=1).reset_index().rename(columns={"off_team_id": "team"})
    p = p.merge(act, on="game_id", how="inner")
    home = (p["team"].astype(str) == p["home_team_id"].astype(str)).to_numpy()
    p["margin"] = np.where(home, p["home_score"] - p["away_score"], p["away_score"] - p["home_score"]).astype(float)
    names = list(g.keys())
    p[names] = p[names].fillna(0.0)
    r = LK.run_leak_test(p, names, team_col="team", season_col="season", date_col="game_date", margin_col="margin")
    return {"rows": r.to_dict(orient="records"), "n_panel": int(len(p)),
            "verdict": "LEAK" if (r["verdict"] == "LEAK").any() else "PASS",
            "worst_abs_corr": float(np.nanmax(np.abs(r[["corr_asjoined", "corr_update"]].to_numpy(dtype=float))))}


def main() -> int:
    t0 = time.time()
    assert_not_sealed(SEASONS, context="shot_location round 2 build_shooter_band")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "design_props_cols_v1.parquet").exists():
        raise SystemExit("round-2 props columns already built; versioned outputs are never overwritten")
    des = pd.read_parquet(DESIGN, columns=["game_id", "season", "game_date", "off_team_id", "shooter_id",
                                           "shot_class", "made", "off_make_raw", "shooter_mk_c", "shooter_att_c"])
    loc = pd.read_parquet(R1 / "design_loc_cols_v1.parquet", columns=["game_id", "shooter_id", "shot_class", "made",
                                                                      "band", "located"])
    for k in ("game_id", "shooter_id", "shot_class", "made"):
        assert (loc[k].to_numpy() == des[k].to_numpy()).all(), k
    des["band"] = loc["band"].to_numpy()
    des["made"] = des["made"].astype(bool)
    m = json.loads(M_FIT.read_text(encoding="utf-8"))
    mvec = des["shot_class"].map({"FGA_jump2": m["FGA_jump2"]["m"], "FGA_3": m["FGA_3"]["m"]}).to_numpy(dtype="float64")
    team = des["off_make_raw"].to_numpy(dtype="float64")
    des["prior"] = ((mvec * team + des["shooter_mk_c"].to_numpy(dtype="float64")) /
                    (mvec + des["shooter_att_c"].to_numpy(dtype="float64")) - team)
    des["prior"] = np.where(np.isfinite(des["prior"]), des["prior"], 0.0)
    D = des[des["shot_class"].isin(BANDS)].copy()
    log(f"jump2 / three attempts: {len(D):,}; located {int((D['band'] != '').sum()):,}", t0)
    prof = pd.read_parquet(R1 / "team_band_profiles_v1.parquet")
    rep: dict = {"created_at": pd.Timestamp.now("UTC").isoformat(), "spec": "experiments.md section 3 (5f6d6b0)",
                 "rho": RHO, "k0_league_make": K0_LEAGUE_MAKE, "m_overall": {c: m[c]["m"] for c in BANDS}}
    # ---- 4. map and join rates
    cw = pd.read_parquet(CROSSWALK)
    cw = cw[cw["season"].isin(SEASONS)][["season", "cbbd_player_id", "espn_athlete_id", "match_method"]]
    mp = D[["season", "shooter_id"]].merge(cw.rename(columns={"cbbd_player_id": "shooter_id"}),
                                           on=["season", "shooter_id"], how="left")
    rep["player_map"] = {"source": "data/processed/player_crosswalk_v2.parquet, key (season, cbbd_player_id) = "
                                   "design shooter_id -> espn_athlete_id (hoopR player_box athlete_id)",
                         "attempt_join_rate_by_season": {int(s): float(g["espn_athlete_id"].notna().mean())
                                                         for s, g in mp.groupby("season")},
                         "by_method": mp["match_method"].fillna("none").value_counts().to_dict()}
    # ---- 1. reliability FIRST
    rel = reliability(D)
    rep["reliability"] = rel
    log("reliability done", t0)
    # ---- 2. panel, k selection
    LGM = league_make_asof(D)
    P = build_panel(D, LGM, prof, t0)
    pidx = pd.MultiIndex.from_frame(P[["shooter_id", "shot_class", "game_id"]])
    A = D[D["band"] != ""].copy()
    A["pi"] = pidx.get_indexer(pd.MultiIndex.from_frame(A[["shooter_id", "shot_class", "game_id"]]))
    assert (A["pi"] >= 0).all()
    A = A.merge(LGM, on=["season", "game_date"], how="left")
    A["lgmk"] = np.select([A["band"] == b for b in ALL], [A["lgmk_" + b] for b in ALL])
    A = A.reset_index(drop=False).rename(columns={"index": "fg_row"})
    k_s, k_m, krep = choose_k(P, A, t0)
    rep["k"] = krep
    F = features(P, k_s, k_m)
    # ---- attach to every design row (jump2 / three); rim rows 0
    Dall = pd.DataFrame({"fg_row": np.arange(len(des))})
    Dp = D[["shooter_id", "shot_class", "game_id", "band"]].copy()
    Dp["fg_row"] = D.index.to_numpy()
    Dp["pi"] = pidx.get_indexer(pd.MultiIndex.from_frame(Dp[["shooter_id", "shot_class", "game_id"]]))
    assert (Dp["pi"] >= 0).all()
    pi = Dp["pi"].to_numpy()
    out = pd.DataFrame(0.0, index=np.arange(len(des)), columns=[*[f"sh_{b}_c" for b in ALL],
                                                                *[f"smk_{b}" for b in ALL]], dtype="float32")
    for b in ALL:
        out.loc[Dp["fg_row"].to_numpy(), f"sh_{b}_c"] = F[b][0][pi].astype("float32")
        out.loc[Dp["fg_row"].to_numpy(), f"smk_{b}"] = F[b][1][pi].astype("float32")
    own = np.zeros(len(des), dtype="float32")
    ownm = np.full(len(des), np.nan, dtype="float32")
    for b in ALL:
        mb = (Dp["band"].to_numpy() == b)
        ownm[Dp["fg_row"].to_numpy()[mb]] = F[b][1][pi[mb]].astype("float32")
    out["sh_band_make_c"] = ownm
    out["sh_n_class"] = np.float32(0)
    Nc = np.zeros(len(P))
    for cls, bands in BANDS.items():
        mm = (P["shot_class"] == cls).to_numpy()
        Nc[mm] = sum(P["Sn_" + b].to_numpy()[mm] for b in bands)
    out.loc[Dp["fg_row"].to_numpy(), "sh_n_class"] = Nc[pi].astype("float32")
    out["game_id"] = des["game_id"].to_numpy(); out["shooter_id"] = des["shooter_id"].to_numpy()
    out["shot_class"] = des["shot_class"].to_numpy(); out["made"] = des["made"].to_numpy()
    log("columns attached", t0)
    # ---- 3. leak test
    lk = leak(des.assign(band=des["band"].fillna("")), out)
    rep["leak_test"] = lk
    rep["refused"] = lk["verdict"] == "LEAK"
    log(f"leak test: {lk['verdict']} worst |corr| {lk['worst_abs_corr']}", t0)
    out.to_parquet(OUT / "design_props_cols_v1.parquet", index=False)
    (OUT / "build_report_v1.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    # distribution summary of the new features on located attempts
    L = D["band"] != ""
    rep_s = {}
    for b in ALL:
        v = out.loc[D.index[L.to_numpy()], f"sh_{b}_c"]
        rep_s[b] = [float(v.mean()), float(v.std())]
    log(f"done; sh_c mean/sd on located: {rep_s}", t0)
    return 0 if not rep["refused"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
