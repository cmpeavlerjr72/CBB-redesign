"""Total-bias decomposition of served stack v2 (diagnostic only, no model).

Splits sim-minus-actual game total into pace (box possession estimate) and the
per-possession channels via an exact Shapley decomposition of the accounting identity

    P   = FGA - OREB + TOV + 0.44 FTA          (per team, both sim and truth)
    FGA = P * (1 - tov_r + oreb_r - 0.44 fta_r)
    pts = 2*FGA*(1-s3)*(r*p_rim + (1-r)*p_jump) + 3*FGA*s3*p3 + P*fta_r*p_ft

factors: P, tov_r, oreb_r, fta_r, s3 (3PA share), r (rim share of 2PA), p_rim, p_jump, p3, p_ft.
Applied to bucket-level means of team-game counts (sim = mean over seeds). Residual =
(final total bias) - (identity bias): technical FTs / box-vs-event class drift / and-ones.

Buckets: season, days since season start (0-14 / 15-45 / 46+), team's first game, site, quintiles.
Truth: verified finals (load_actual_games), hoopR team box (TOV/OREB/FTA/FTM/FGA), event shot
classes (team_game_shots_v2) for rim/jump/3 attempts and makes. 2025-26 never read.

Run: .venv/Scripts/python.exe scripts/diag_total_bias_decomp_v1.py
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.eval import reference as R

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    2024: ROOT / "results/engine_v0/f1c_V2_full_s200_o0/games.parquet",
    2025: ROOT / "results/engine_v0/v3full_COMB9GCTKD_s200_o0/games.parquet",
}
OUT = ROOT / "results/total_bias_decomp"
FACT = ["P", "tov_r", "oreb_r", "fta_r", "s3", "rim_sh", "p_rim", "p_jump", "p3", "p_ft"]
CNT = ["fga3", "fga2_rim", "fga2_jump", "fgm3", "fgm2_rim", "fgm2_jump", "fta", "ftm", "tov", "oreb"]


def sim_team_games(season: int) -> pd.DataFrame:
    g = pd.read_parquet(RUNS[season])
    cols = ["home_pts", "away_pts"] + [f"{s}_{c}" for s in ("home", "away") for c in CNT]
    m = g.groupby("game_id")[cols].mean().reset_index()
    m["game_id"] = m["game_id"].astype("int64")
    rows = []
    for side in ("home", "away"):
        d = m[["game_id", f"{side}_pts"] + [f"{side}_{c}" for c in CNT]].copy()
        d.columns = ["game_id", "pts"] + CNT
        d["side"] = side
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def truth_team_games(season: int, games: pd.DataFrame) -> pd.DataFrame:
    tb = R.load_actual_team_box(season)
    sh = pd.read_parquet(ROOT / "data/processed/truth/team_game_shots_v2.parquet")
    sh = sh[sh["season"] == season]
    t = tb[["game_id", "team_id", "tov", "oreb", "fta", "ftm"]].merge(
        sh[["game_id", "team_id", "ev_fga_rim", "ev_fgm_rim", "ev_fga_jump2", "ev_fgm_jump2", "ev_fga_3", "ev_fgm_3"]],
        on=["game_id", "team_id"], how="inner")
    t = t.rename(columns={"ev_fga_rim": "fga2_rim", "ev_fgm_rim": "fgm2_rim", "ev_fga_jump2": "fga2_jump",
                          "ev_fgm_jump2": "fgm2_jump", "ev_fga_3": "fga3", "ev_fgm_3": "fgm3"})
    hm = games[["game_id", "home_team_id", "away_team_id", "home_score", "away_score"]]
    t = t.merge(hm, on="game_id", how="inner")
    t["side"] = np.where(t["team_id"] == t["home_team_id"], "home", "away")
    t["pts"] = np.where(t["side"] == "home", t["home_score"], t["away_score"])
    t = t[["game_id", "team_id", "side", "pts"] + CNT]
    bad = set(t.loc[t[CNT].isna().any(axis=1), "game_id"])  # box_only rows (no event shot classes)
    return t[~t.game_id.isin(bad)]


def factors(d: pd.DataFrame) -> dict:
    s = d[CNT + ["pts"]].sum()
    fga = s.fga3 + s.fga2_rim + s.fga2_jump
    P = fga - s.oreb + s.tov + 0.44 * s.fta
    fga2 = s.fga2_rim + s.fga2_jump
    return dict(P=P, tov_r=s.tov / P, oreb_r=s.oreb / P, fta_r=s.fta / P, s3=s.fga3 / fga,
                rim_sh=s.fga2_rim / fga2, p_rim=s.fgm2_rim / s.fga2_rim, p_jump=s.fgm2_jump / s.fga2_jump,
                p3=s.fgm3 / s.fga3, p_ft=s.ftm / s.fta, pts=s.pts)


def ident(f: dict) -> float:
    fga = f["P"] * (1 - f["tov_r"] + f["oreb_r"] - 0.44 * f["fta_r"])
    p2 = f["rim_sh"] * f["p_rim"] + (1 - f["rim_sh"]) * f["p_jump"]
    return 2 * fga * (1 - f["s3"]) * p2 + 3 * fga * f["s3"] * f["p3"] + f["P"] * f["fta_r"] * f["p_ft"]


def shapley(fs: dict, fa: dict) -> dict:
    n = len(FACT)
    cache = {}

    def v(mask):
        if mask not in cache:
            cache[mask] = ident({k: (fs[k] if mask >> i & 1 else fa[k]) for i, k in enumerate(FACT)})
        return cache[mask]
    out = {}
    for i, k in enumerate(FACT):
        tot = 0.0
        for mask in range(1 << n):
            if mask >> i & 1:
                continue
            sz = bin(mask).count("1")
            w = math.factorial(sz) * math.factorial(n - sz - 1) / math.factorial(n)
            tot += w * (v(mask | 1 << i) - v(mask))
        out[k] = tot
    return out


def decompose(sim: pd.DataFrame, tru: pd.DataFrame, n_games: int) -> dict:
    fs, fa = factors(sim), factors(tru)
    sh = {k: v / n_games for k, v in shapley(fs, fa).items()}
    res = dict(n_games=n_games, total_bias=(fs["pts"] - fa["pts"]) / n_games,
               ident_bias=(ident(fs) - ident(fa)) / n_games)
    res["resid"] = res["total_bias"] - res["ident_bias"]
    res.update({f"c_{k}": v for k, v in sh.items()})
    res.update({f"sim_{k}": fs[k] for k in FACT if k != "P"})
    res.update({f"act_{k}": fa[k] for k in FACT if k != "P"})
    n_tg = len(sim)
    res["sim_P_tg"], res["act_P_tg"] = fs["P"] / n_tg, fa["P"] / n_tg
    res["sim_ppp"], res["act_ppp"] = fs["pts"] / fs["P"], fa["pts"] / fa["P"]
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    allrows = []
    gl_all = []
    for season in (2024, 2025):
        games = R.load_actual_games(season)
        sim = sim_team_games(season)
        tru = truth_team_games(season, games)
        common = set(sim.game_id) & set(tru.groupby("game_id").filter(lambda x: len(x) == 2).game_id)
        games = games[games.game_id.isin(common)].copy()
        games["date"] = pd.to_datetime(games["game_date"])
        start = games["date"].min()
        games["dss"] = (games["date"] - start).dt.days
        games["dss_b"] = pd.cut(games["dss"], [-1, 14, 45, 10_000], labels=["d00-14", "d15-45", "d46+"]).astype(str)
        # team's first game of season (either team)
        long = pd.concat([games[["game_id", "date", "home_team_id"]].rename(columns={"home_team_id": "t"}),
                          games[["game_id", "date", "away_team_id"]].rename(columns={"away_team_id": "t"})])
        long = long.sort_values(["t", "date", "game_id"])
        long["k"] = long.groupby("t").cumcount()
        first_any = set(long.loc[long.k == 0, "game_id"])
        games["opener"] = games.game_id.isin(first_any)
        games["site"] = np.where(games["neutral"] > 0, "neutral", "home-court")
        # sim mean total per game (prediction) quintile -> responsiveness
        gs = sim.groupby("game_id")["pts"].sum().rename("sim_total")
        games = games.merge(gs, left_on="game_id", right_index=True)
        games["predq"] = pd.qcut(games["sim_total"], 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
        # prior-season (S-1) team scoring env (pts for + against per game) quintile, grading-only
        prev = R.load_actual_games(season - 1)
        pl = pd.concat([prev[["home_team_id", "total"]].rename(columns={"home_team_id": "t"}),
                        prev[["away_team_id", "total"]].rename(columns={"away_team_id": "t"})])
        penv = pl.groupby("t")["total"].mean()
        games["prior_env"] = games["home_team_id"].map(penv) + games["away_team_id"].map(penv)
        games["priorq"] = pd.qcut(games["prior_env"], 5, labels=[f"P{i}" for i in range(1, 6)]).astype(str)
        games.loc[games["prior_env"].isna(), "priorq"] = "P_na"
        sim = sim[sim.game_id.isin(common)].merge(games[["game_id", "dss_b", "opener", "site", "predq", "priorq"]], on="game_id")
        tru = tru[tru.game_id.isin(common)].merge(games[["game_id", "dss_b", "opener", "site", "predq", "priorq"]], on="game_id")
        gl = games[["game_id", "dss", "dss_b", "opener", "site", "predq", "priorq", "total", "sim_total"]].copy()
        gl["season"] = season
        gl_all.append(gl)

        def add(dim, key, s_, t_):
            n = s_.game_id.nunique()
            r = decompose(s_, t_, n)
            r.update(season=season, dim=dim, bucket=key)
            allrows.append(r)
        add("all", "all", sim, tru)
        for dim in ("dss_b", "opener", "site", "predq", "priorq"):
            for key, s_ in sim.groupby(dim):
                add(dim, str(key), s_, tru[tru[dim] == key])
        for key in ("d00-14", "d15-45", "d46+"):
            for side in ("home", "away"):
                s_ = sim[(sim.dss_b == key) & (sim.side == side) & (sim.site == "home-court")]
                t_ = tru[(tru.dss_b == key) & (tru.side == side) & (tru.site == "home-court")]
                add("dss_x_side", f"{key}|{side}", s_, t_)
    df = pd.DataFrame(allrows)
    df.to_csv(OUT / "decomp_v1.csv", index=False)
    pd.concat(gl_all).to_parquet(OUT / "game_level_v1.parquet")
    show = ["season", "dim", "bucket", "n_games", "total_bias", "c_P", "c_tov_r", "c_oreb_r", "c_fta_r", "c_s3",
            "c_rim_sh", "c_p_rim", "c_p_jump", "c_p3", "c_p_ft", "resid"]
    pd.set_option("display.width", 250)
    print(df[show].round(2).to_string(index=False))
    rates = ["season", "dim", "bucket", "sim_P_tg", "act_P_tg", "sim_ppp", "act_ppp", "sim_p_ft", "act_p_ft",
             "sim_p3", "act_p3", "sim_p_rim", "act_p_rim", "sim_p_jump", "act_p_jump", "sim_tov_r", "act_tov_r",
             "sim_oreb_r", "act_oreb_r", "sim_fta_r", "act_fta_r"]
    print(df[df.dim.isin(["all", "dss_b", "opener"])][rates].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
