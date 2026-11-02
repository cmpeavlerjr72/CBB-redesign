#!/usr/bin/env python
"""
diag_opening_day_totals_v1.py -- opening-day (2026-11-02) totals diagnostic. REPORT ONLY.

Owner report: the 2027 opening-day sim totals sit ~11 points under the market on every game.
Nothing here changes the engine, its tables or configs; no adjustment is applied to any output.

Sections (each writes CSVs under results/diag/opening_day_totals_2026-11-02/):
  q1  today's slate: pace vs efficiency split of sim - market, bucketed.
  q2  history (2022-23, 2023-24, 2024-25; 2025-26 is SEALED and not read): actual total / possessions / PPP /
      market total by day-of-season bucket (d0, d1-3, d4-7, d8-14, d15-30, d31+).
  q3  the frozen engine (served v3 = clock K2 closed-loop 200-seed reads, F1 2023-24 and F2 2024-25) by the same
      buckets: sim - actual, sim - market, pace vs PPP split, Shapley channels on d0.
  q4  2027-build checks: league level of the 2027 ratings vs history; slate per-possession channel rates and a
      Shapley decomposition of today's slate vs real opening days and vs the engine's own past opening days;
      anonymous-minutes regression on today's team-games.
  q5  any completed 2027 finals for today's games.

Possessions everywhere are box possessions P = FGA - OREB + TOV + 0.44 FTA per team (truth and sim alike);
PPP = points / P per team; a game total = 2 * P_game * PPP_game with P_game the two-team mean.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/diag_opening_day_totals_v1.py [--run-replay]

--run-replay first runs the CURRENT daily path on 2024-11-04 (fold-2 opening day) at 200 seeds into OUT (about 3 min at 8 jobs);
without it, an existing replay under OUT is graded if present.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.eval import reference as R  # noqa: E402
import diag_total_bias_decomp_v1 as DEC  # noqa: E402  (factors / shapley / truth_team_games, imported unchanged)

OUT = ROOT / "results/diag/opening_day_totals_2026-11-02"
SLATE = ROOT / "results/daily/publish/2026-11-02/s2500_o0/da1e4c7591/slate.parquet"
SLATE_GAMES = ROOT / "results/daily/sim/2026-11-02/s2500_o0/games.parquet"
RATINGS27 = ROOT / "data/processed/ratings_asof/2026-11-02/own_ratings_2027.parquet"
SCHED27 = ROOT / "data/raw/preseason/2027_v2_20261008/games_2027.parquet"
LINES = ROOT / "data/processed/lines/lines_close_v2_verified.parquet"
RUNS = {  # served v3 (clock K2) closed-loop reads, 200 seeds
    2024: ROOT / "results/engine_v0/f1c_K2_full_s200_o0/games.parquet",
    2025: ROOT / "results/engine_v0/laneH_v3full_K2_s200_o0/games.parquet",
}
HIST = [2023, 2024, 2025]  # 2025-26 (2026) is sealed: not read
CNT = DEC.CNT
POWER = {"ACC", "Big Ten", "Big 12", "SEC", "Big East"}
HIMID = {"Mountain West", "WCC", "A-10", "American", "Pac-12"}
BUCKETS = [(-99, 0, "d0"), (1, 3, "d1-3"), (4, 7, "d4-7"), (8, 14, "d8-14"), (15, 30, "d15-30"), (31, 999, "d31+")]


def tier(c):
    if c in POWER:
        return "P"
    if c in HIMID:
        return "M"
    return "L"


def matchup(a, b):
    t = "".join(sorted([tier(a), tier(b)], key="PML".index))
    return {"PP": "power-power", "PM": "power-mid", "PL": "power-low", "MM": "mid-mid", "ML": "mid-low",
            "LL": "low-low"}[t]


def dbucket(d):
    for lo, hi, name in BUCKETS:
        if lo <= d <= hi:
            return name
    return "pre"


def market_totals(season: int) -> pd.Series:
    L = pd.read_parquet(LINES)
    L = L[(L.season == season) & L.close_over_under.notna()]
    espn = L[L.provider == "ESPN BET"].groupby("game_id").close_over_under.first()
    med = L.groupby("game_id").close_over_under.median()
    return espn.combine_first(med).rename("market_total")


def market_spreads(season: int) -> pd.Series:
    L = pd.read_parquet(LINES)
    L = L[(L.season == season) & L.close_spread_home.notna()]
    espn = L[L.provider == "ESPN BET"].groupby("game_id").close_spread_home.first()
    return espn.combine_first(L.groupby("game_id").close_spread_home.median()).rename("market_spread")


def conferences(season: int) -> pd.DataFrame:
    g = pd.read_parquet(ROOT / f"data/raw/cbbd/games_{season}.parquet", columns=["id", "homeConference", "awayConference"])
    return g.rename(columns={"id": "cbbd_game_id"})


def hist_games(season: int) -> pd.DataFrame:
    """Verified finals + box possessions + market + day-of-season, one row per D-I game."""
    g = R.load_actual_games(season)
    g = g[g.get("season_type", 2) == 2] if "season_type" in g else g
    tb = R.load_actual_team_box(season)
    gp = tb.groupby("game_id").agg(P=("poss_team", "mean"), n=("poss_team", "size"))
    gp = gp[gp.n == 2].drop(columns="n")
    g = g.merge(gp, left_on="game_id", right_index=True, how="inner")
    g["date"] = pd.to_datetime(g["game_date"])
    cnt = g.groupby("date").size()
    d0 = cnt[cnt >= 50].index.min()
    g["dss"] = (g["date"] - d0).dt.days
    g = g[g.dss >= 0].copy()
    g["bucket"] = g["dss"].map(dbucket)
    g = g.merge(market_totals(season), left_on="game_id", right_index=True, how="left")
    g = g.merge(market_spreads(season), left_on="game_id", right_index=True, how="left")
    cf = conferences(season)
    g["cbbd_game_id"] = g["cbbd_game_id"].astype("Int64")
    g = g.merge(cf, on="cbbd_game_id", how="left")
    g["matchup"] = [matchup(a, b) for a, b in zip(g.homeConference, g.awayConference)]
    g["ppp"] = g["total"] / (2 * g["P"])
    g["season_label"] = f"{season - 1}-{str(season)[2:]}"
    g["d0_date"] = d0.date()
    return g


def bucket_table(df: pd.DataFrame, keys, cols_extra=None) -> pd.DataFrame:
    rows = []
    for k, x in df.groupby(keys, observed=True):
        lined = x[x.market_total.notna()]
        r = dict(zip(keys if isinstance(keys, list) else [keys], k if isinstance(k, tuple) else [k]))
        r.update(n=len(x), actual_total=x.total.mean(), actual_P=x.P.mean(),
                 actual_ppp=x.total.sum() / (2 * x.P.sum()), n_lined=len(lined),
                 market_total=lined.market_total.mean(),
                 actual_minus_market=(lined.total - lined.market_total).mean(),
                 se_amm=(lined.total - lined.market_total).std() / np.sqrt(max(len(lined), 1)),
                 abs_spread=lined.market_spread.abs().mean())
        if cols_extra:
            for c in cols_extra:
                r[c] = x[c].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def split_total_gap(Pa, ppa, Pb, ppb):
    """total = 2 P PPP; midpoint split of (a - b) into pace and PPP parts."""
    pace = 2 * (Pa - Pb) * (ppa + ppb) / 2
    eff = 2 * (ppa - ppb) * (Pa + Pb) / 2
    return pace, eff


# ------------------------------------------------------------------ Q2
def q2():
    allg = []
    for s in HIST:
        allg.append(hist_games(s))
    G = pd.concat(allg, ignore_index=True)
    G.to_parquet(OUT / "q2_hist_games.parquet")
    order = [b[2] for b in BUCKETS]
    G["bucket"] = pd.Categorical(G["bucket"], order)
    t = bucket_table(G, ["season_label", "bucket"])
    t.to_csv(OUT / "q2_by_season_bucket.csv", index=False)
    tp = bucket_table(G, ["bucket"])
    tp.to_csv(OUT / "q2_pooled_bucket.csv", index=False)
    # d0 vs rest-of-season (d15+) split into pace and PPP, per season and pooled
    rows = []
    for lab, x in list(G.groupby("season_label")) + [("pooled", G)]:
        a = x[x.bucket == "d0"]
        b = x[x.dss >= 15]
        Pa, Pb = a.P.mean(), b.P.mean()
        ppa, ppb = a.total.sum() / (2 * a.P.sum()), b.total.sum() / (2 * b.P.sum())
        pace, eff = split_total_gap(Pa, ppa, Pb, ppb)
        la, lb = a[a.market_total.notna()], b[b.market_total.notna()]
        rows.append(dict(season=lab, n_d0=len(a), d0_total=a.total.mean(), rest_total=b.total.mean(),
                         d0_minus_rest=a.total.mean() - b.total.mean(), pace_part=pace, ppp_part=eff,
                         d0_P=Pa, rest_P=Pb, d0_ppp=ppa, rest_ppp=ppb,
                         d0_market=la.market_total.mean(), rest_market=lb.market_total.mean(),
                         d0_act_minus_mkt=(la.total - la.market_total).mean(),
                         rest_act_minus_mkt=(lb.total - lb.market_total).mean(),
                         d0_abs_spread=la.market_spread.abs().mean(), rest_abs_spread=lb.market_spread.abs().mean()))
    pd.DataFrame(rows).to_csv(OUT / "q2_d0_vs_rest.csv", index=False)
    # d0 by matchup class and by |spread| tercile (pooled seasons)
    d0 = G[G.bucket == "d0"].copy()
    bucket_table(d0, ["matchup"]).to_csv(OUT / "q2_d0_by_matchup.csv", index=False)
    rest = G[G.dss >= 15].copy()
    bucket_table(rest, ["matchup"]).to_csv(OUT / "q2_rest_by_matchup.csv", index=False)
    edges = [0, 5, 10, 17, 25, 100]
    for nm, x in (("d0", d0), ("rest", rest)):
        x = x[x.market_spread.notna()].copy()
        x["abs_spread_b"] = pd.cut(x.market_spread.abs(), edges, right=False).astype(str)
        bucket_table(x, ["abs_spread_b"]).to_csv(OUT / f"q2_{nm}_by_abs_spread.csv", index=False)
    return G


# ------------------------------------------------------------------ Q3
def sim_game_means(path: Path) -> pd.DataFrame:
    g = pd.read_parquet(path)
    cols = ["home_pts", "away_pts", "possessions"] + [f"{s}_{c}" for s in ("home", "away") for c in CNT]
    m = g.groupby("game_id")[cols].mean().reset_index()
    m["game_id"] = m["game_id"].astype("int64")
    for s in ("home", "away"):
        m[f"{s}_P"] = (m[f"{s}_fga3"] + m[f"{s}_fga2_rim"] + m[f"{s}_fga2_jump"] - m[f"{s}_oreb"] + m[f"{s}_tov"]
                       + 0.44 * m[f"{s}_fta"])
    m["sim_P"] = (m.home_P + m.away_P) / 2
    m["sim_total"] = m.home_pts + m.away_pts
    m["sim_ppp"] = m.sim_total / (2 * m.sim_P)
    return m


def team_rows(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ("home", "away"):
        d = m[["game_id", f"{side}_pts"] + [f"{side}_{c}" for c in CNT]].copy()
        d.columns = ["game_id", "pts"] + CNT
        d["side"] = side
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def per_game_factors(d: pd.DataFrame, n_games: int) -> dict:
    f = DEC.factors(d)
    f["P"] = f["P"] / n_games
    f["pts"] = f["pts"] / n_games
    return f


def shap_row(sim_t: pd.DataFrame, ref_t: pd.DataFrame, n_sim: int, n_ref: int, label: str) -> dict:
    fs, fa = per_game_factors(sim_t, n_sim), per_game_factors(ref_t, n_ref)
    sh = DEC.shapley(fs, fa)
    r = dict(comparison=label, n_sim_games=n_sim, n_ref_games=n_ref, pts_gap=fs["pts"] - fa["pts"],
             ident_gap=DEC.ident(fs) - DEC.ident(fa))
    r.update({f"c_{k}": v for k, v in sh.items()})
    for k in DEC.FACT:
        r[f"sim_{k}"], r[f"ref_{k}"] = fs[k], fa[k]
    r["sim_P_tg"], r["ref_P_tg"] = fs["P"] / 2, fa["P"] / 2
    r["sim_ppp"], r["ref_ppp"] = fs["pts"] / fs["P"], fa["pts"] / fa["P"]
    return r


def q3(G: pd.DataFrame):
    order = [b[2] for b in BUCKETS]
    rows, shap = [], []
    simd0 = {}
    for season, path in RUNS.items():
        m = sim_game_means(path)
        g = G[G.season == season].merge(m[["game_id", "sim_total", "sim_P", "sim_ppp", "possessions"]], on="game_id")
        g["bucket"] = pd.Categorical(g["bucket"], order)
        for b, x in g.groupby("bucket", observed=True):
            lx = x[x.market_total.notna()]
            ppa = x.total.sum() / (2 * x.P.sum())
            pps = x.sim_total.sum() / (2 * x.sim_P.sum())
            pace, eff = split_total_gap(x.sim_P.mean(), pps, x.P.mean(), ppa)
            rows.append(dict(season=f"{season - 1}-{str(season)[2:]}", bucket=b, n=len(x),
                             sim_total=x.sim_total.mean(), actual_total=x.total.mean(),
                             sim_minus_actual=(x.sim_total - x.total).mean(),
                             se_sma=(x.sim_total - x.total).std() / np.sqrt(len(x)),
                             n_lined=len(lx), market_total=lx.market_total.mean(),
                             sim_minus_market=(lx.sim_total - lx.market_total).mean(),
                             se_smm=(lx.sim_total - lx.market_total).std() / np.sqrt(max(len(lx), 1)),
                             actual_minus_market=(lx.total - lx.market_total).mean(),
                             sim_P=x.sim_P.mean(), actual_P=x.P.mean(), sim_engine_poss=x.possessions.mean(),
                             sim_ppp=pps, actual_ppp=ppa, pace_part=pace, ppp_part=eff))
        # Shapley on d0, d1-3 and d15+ (truth needs event shot classes)
        games_s = R.load_actual_games(season)
        tru = DEC.truth_team_games(season, games_s)
        st = team_rows(m)
        for lab, ids in (("d0", set(g.game_id[g.bucket == "d0"])), ("d1-3", set(g.game_id[g.bucket == "d1-3"])),
                         ("d0-14", set(g.game_id[g.dss <= 14])), ("d15+", set(g.game_id[g.dss >= 15]))):
            ok = set(tru.groupby("game_id").filter(lambda z: len(z) == 2).game_id) & ids
            r = shap_row(st[st.game_id.isin(ok)], tru[tru.game_id.isin(ok)], len(ok), len(ok),
                         f"{season - 1}-{str(season)[2:]} {lab}: engine - actual")
            shap.append(r)
        d0ids = set(g.game_id[g.bucket == "d0"])
        simd0[season] = (st[st.game_id.isin(d0ids)], len(d0ids))
    pd.DataFrame(rows).to_csv(OUT / "q3_engine_by_bucket.csv", index=False)
    pd.DataFrame(shap).to_csv(OUT / "q3_engine_shapley.csv", index=False)
    return simd0


# ------------------------------------------------------------------ Q1 / Q4
def slate_frame():
    s = pd.read_parquet(SLATE)
    m = sim_game_means(SLATE_GAMES)
    s = s.merge(m[["game_id", "sim_P", "sim_ppp", "home_P", "away_P", "home_tov", "away_tov", "home_oreb",
                   "away_oreb"]], on="game_id", how="left")
    r = pd.read_parquet(RATINGS27)[["team_id", "off_c", "def_c", "tempo_rel"]]
    r["net"] = r.off_c - r.def_c
    for side in ("home", "away"):
        s = s.merge(r.add_prefix(f"{side}_r_").rename(columns={f"{side}_r_team_id": f"{side}_team_id"}),
                    on=f"{side}_team_id", how="left")
    sc = pd.read_parquet(SCHED27, columns=["id", "homeConference", "awayConference"]).rename(columns={"id": "cbbd_game_id"})
    s = s.merge(sc, on="cbbd_game_id", how="left")
    s["matchup"] = [matchup(a, b) for a, b in zip(s.homeConference, s.awayConference)]
    an = s["notes"].fillna("").str.extract(r"anon_minutes home=([\d.]+) away=([\d.]+)").astype(float)
    s["anon_home"], s["anon_away"] = an[0], an[1]
    return s, m


def q1(s: pd.DataFrame, G: pd.DataFrame):
    x = s[s.total.notna()].copy()
    x["gap"] = x.sim_total_mean - x.total
    x["ppp_mkt_at_simP"] = x.total / (2 * x.sim_P)
    x["P_mkt_at_simppp"] = x.total / (2 * x.sim_ppp)
    # reference pace: real opening-day possessions (pooled 3 seasons), and that by matchup class
    d0 = G[G.bucket == "d0"]
    P_ref = d0.P.mean()
    P_ref_by_m = d0.groupby("matchup").P.mean()
    x["P_ref_d0"] = x.matchup.map(P_ref_by_m).fillna(P_ref)
    x["pace_part_ref"] = 2 * (x.sim_P - x.P_ref_d0) * x.sim_ppp
    x["eff_part_ref"] = x.gap - x.pace_part_ref
    x["home_q"] = pd.qcut(x.home_r_net, 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
    x["abs_spread_q"] = pd.qcut(x.spread.abs(), 5, labels=[f"S{i}" for i in range(1, 6)]).astype(str)
    x["site"] = np.where(x.neutral, "neutral", "home-court")
    x.to_csv(OUT / "q1_slate_games.csv", index=False)
    cols = ["gap", "sim_total_mean", "total", "sim_P", "sim_poss_mean", "sim_ppp", "ppp_mkt_at_simP",
            "P_mkt_at_simppp", "P_ref_d0", "pace_part_ref", "eff_part_ref"]
    out = []
    for dim in ("all", "home_q", "abs_spread_q", "site", "matchup"):
        grp = [("all", x)] if dim == "all" else list(x.groupby(dim))
        for k, y in grp:
            r = dict(dim=dim, bucket=k, n=len(y))
            r.update({c: y[c].mean() for c in cols})
            r["share_below_line"] = (y.gap < 0).mean()
            r["abs_spread"] = y.spread.abs().mean()
            out.append(r)
    pd.DataFrame(out).to_csv(OUT / "q1_buckets.csv", index=False)
    dist = x[["gap", "sim_ppp", "ppp_mkt_at_simP", "sim_P", "P_mkt_at_simppp"]].describe(percentiles=[.1, .5, .9]).T
    dist.to_csv(OUT / "q1_distributions.csv")
    return x


def q4(s: pd.DataFrame, m_slate: pd.DataFrame, G: pd.DataFrame, simd0: dict):
    # (a) league level by season from truth vs ratings league means
    rows = []
    for season in HIST:
        x = G[G.season == season]
        rt = pd.read_parquet(ROOT / f"data/processed/ratings/own_ratings_{season}.parquet",
                             columns=["as_of_date", "league_off_mean", "league_tempo_mean"])
        rt["as_of_date"] = pd.to_datetime(rt.as_of_date)
        lm = rt.groupby("as_of_date").first()
        nov = x[x.date.dt.month == 11]
        rows.append(dict(season=f"{season - 1}-{str(season)[2:]}",
                         truth_ppp_season=100 * x.total.sum() / (2 * x.P.sum()),
                         truth_ppp_nov=100 * nov.total.sum() / (2 * nov.P.sum()),
                         truth_ppp_d0=100 * x[x.bucket == "d0"].total.sum() / (2 * x[x.bucket == "d0"].P.sum()),
                         truth_P_season=x.P.mean(), truth_P_nov=nov.P.mean(),
                         truth_total_season=x.total.mean(), truth_total_nov=nov.total.mean(),
                         ratings_league_off_d0=lm.league_off_mean.iloc[0],
                         ratings_league_off_final=lm.league_off_mean.iloc[-1],
                         ratings_league_tempo_d0=lm.league_tempo_mean.iloc[0],
                         ratings_league_tempo_final=lm.league_tempo_mean.iloc[-1]))
    r27 = pd.read_parquet(RATINGS27)
    rows.append(dict(season="2026-27 day-1 (= 2025-26 final carried)", ratings_league_off_d0=r27.league_off_mean.iloc[0],
                     ratings_league_tempo_d0=r27.league_tempo_mean.iloc[0]))
    pd.DataFrame(rows).to_csv(OUT / "q4a_league_level.csv", index=False)

    # (b) slate channel rates and Shapley vs references
    st = team_rows(m_slate[m_slate.game_id.isin(s.game_id)])
    n_sl = st.game_id.nunique()
    shap = []
    d0g = G[G.bucket == "d0"]
    tru_all, sim_all, n_tr = [], [], 0
    for season in HIST:
        games_s = R.load_actual_games(season)
        tru = DEC.truth_team_games(season, games_s)
        ok = set(tru.groupby("game_id").filter(lambda z: len(z) == 2).game_id) & set(d0g.game_id[d0g.season == season])
        tru_all.append(tru[tru.game_id.isin(ok)])
        n_tr += len(ok)
        shap.append(shap_row(st, tru[tru.game_id.isin(ok)], n_sl, len(ok), f"today vs real d0 {season - 1}-{str(season)[2:]}"))
        rest = set(tru.groupby("game_id").filter(lambda z: len(z) == 2).game_id) & set(G.game_id[(G.season == season) & (G.dss >= 15)])
        shap.append(shap_row(st, tru[tru.game_id.isin(rest)], n_sl, len(rest), f"today vs real d15+ {season - 1}-{str(season)[2:]}"))
    shap.append(shap_row(st, pd.concat(tru_all), n_sl, n_tr, "today vs real d0 pooled 3 seasons"))
    for season, (sd, n) in simd0.items():
        shap.append(shap_row(st, sd, n_sl, n, f"today vs ENGINE d0 {season - 1}-{str(season)[2:]} (closed loop K2)"))
        sim_all.append((sd, n))
    pd.DataFrame(shap).to_csv(OUT / "q4b_shapley.csv", index=False)

    # (c) anonymous minutes: team-level regression on today's slate
    tg = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = pd.DataFrame(dict(game_id=s.game_id, pts=s[f"sim_{side}_pts_mean"], P=s[f"{side}_P"],
                              off=s[f"{side}_r_off_c"], odef=s[f"{opp}_r_def_c"], anon=s[f"anon_{side}"],
                              home=(side == "home") & ~s.neutral, away=(side == "away") & ~s.neutral))
        tg.append(d)
    tg = pd.concat(tg, ignore_index=True).dropna()
    tg["ppp100"] = 100 * tg.pts / tg.P
    X = np.column_stack([np.ones(len(tg)), tg.off, tg.odef, tg.home.astype(float), tg.away.astype(float), tg.anon])
    beta, *_ = np.linalg.lstsq(X, tg.ppp100.values, rcond=None)
    resid = tg.ppp100.values - X @ beta
    sig2 = resid @ resid / (len(tg) - X.shape[1])
    se = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
    reg = pd.DataFrame(dict(term=["const", "off_c", "opp_def_c", "home", "away", "anon_minutes"], beta=beta, se=se))
    reg.to_csv(OUT / "q4c_anon_regression.csv", index=False)
    tg.to_csv(OUT / "q4c_team_games.csv", index=False)
    return reg, tg


def q5(s: pd.DataFrame):
    """Verified 2027 finals for today's games; falls back to the chain's single-source CBBD schedule pull (UNVERIFIED)."""
    f = pd.read_parquet(ROOT / "data/processed/ingest/finals_verified_2027.parquet")
    j = s.merge(f[["game_id", "hoopr_home", "hoopr_away"]], on="game_id", how="inner")
    j["final_total"] = j.hoopr_home + j.hoopr_away
    j["source"] = "finals_verified_2027"
    sd = ROOT / "data/raw/schedule_daily/2026-11-02.parquet"
    if sd.exists():
        d = pd.read_parquet(sd)
        d = d[(d.status == "final") & ~d.game_id.isin(j.game_id)]
        k = s.merge(d[["game_id", "home_points", "away_points"]], on="game_id", how="inner")
        k["final_total"] = k.home_points + k.away_points
        k["source"] = "schedule_daily CBBD status=final (UNVERIFIED, single source)"
        j = pd.concat([j, k], ignore_index=True)
    j[["game_id", "home_name", "away_name", "sim_total_mean", "sim_total_q05", "sim_total_q95", "total", "final_total",
       "source"]].to_csv(OUT / "q5_finals.csv", index=False)
    return j


REPLAY = OUT / "replay_2024-11-04/sim/2024-11-04/s200_o0/games.parquet"


def run_replay():
    """Current daily path (run_daily_sim_v1.run_sim_stage, replay=True) on 2024-11-04 at 200 seeds, 8 jobs, into OUT."""
    import run_daily_sim_v1 as SIM
    now = pd.Timestamp("2024-11-04", tz="America/New_York") - pd.Timedelta(hours=4)
    return SIM.run_sim_stage("2024-11-04", 2025, "F2", 200, 0, now, OUT / "replay_2024-11-04", None, "universe",
                             replay=True, jobs=8, force=True)


def q3_replay(G: pd.DataFrame):
    if not REPLAY.exists():
        return None
    m = sim_game_means(REPLAY)
    x = G[G.season == 2025].merge(m[["game_id", "sim_total", "sim_P", "possessions"]], on="game_id")
    lx = x[x.market_total.notna()]
    pps, ppa = x.sim_total.sum() / (2 * x.sim_P.sum()), x.total.sum() / (2 * x.P.sum())
    pace, eff = split_total_gap(x.sim_P.mean(), pps, x.P.mean(), ppa)
    k = sim_game_means(RUNS[2025])
    jj = m.merge(k, on="game_id", suffixes=("_r", "_k"))
    r = dict(n=len(x), sim_total=x.sim_total.mean(), actual_total=x.total.mean(),
             sim_minus_actual=(x.sim_total - x.total).mean(), se=(x.sim_total - x.total).std() / np.sqrt(len(x)),
             n_lined=len(lx), sim_minus_market=(lx.sim_total - lx.market_total).mean(), sim_P=x.sim_P.mean(),
             actual_P=x.P.mean(), sim_ppp=pps, actual_ppp=ppa, pace_part=pace, ppp_part=eff,
             max_abs_diff_vs_closed_loop_total=float((jj.sim_total_r - jj.sim_total_k).abs().max()),
             max_abs_diff_vs_closed_loop_P=float((jj.sim_P_r - jj.sim_P_k).abs().max()))
    pd.DataFrame([r]).to_csv(OUT / "q3_replay_2024-11-04.csv", index=False)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--run-replay" in sys.argv:
        print(run_replay())
    G = q2()
    simd0 = q3(G)
    q3_replay(G)
    s, m_slate = slate_frame()
    q1(s, G)
    q4(s, m_slate, G, simd0)
    j = q5(s)
    meta = dict(slate=str(SLATE.relative_to(ROOT)), slate_games=str(SLATE_GAMES.relative_to(ROOT)),
                runs={k: str(v.relative_to(ROOT)) for k, v in RUNS.items()}, hist_seasons=HIST,
                sealed_not_read=[2026], n_q5_finals=len(j),
                d0_dates={int(k): str(v) for k, v in G.groupby("season").d0_date.first().items()})
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
