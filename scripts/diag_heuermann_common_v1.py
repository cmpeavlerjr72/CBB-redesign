"""Shared loaders and as-of / shrinkage helpers for the Heuermann sim_v2 data tests (2026-11-08).

Seasons are labelled by ending year (2024 = 2023-24). ONLY seasons 2022-2025 are ever loaded: 2026 (sealed) and 2027 (live)
are dropped at load time. Diagnostics only: no model, no tuning.
"""
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUTCSV = pathlib.Path(r"C:\Users\devuser\rfq-collab\CBB Model\Handoffs - Communication\2026-11-08_Peavler_REPLY_cbb-sim-v2")
RES = ROOT / "results/diag/heuermann_2026-11-08"
SEASONS = [2022, 2023, 2024, 2025]


def save_csv(df, name):
    OUTCSV.mkdir(parents=True, exist_ok=True)
    f = OUTCSV / name
    df.to_csv(f, index=False)
    assert f.stat().st_size < 1_000_000, (name, f.stat().st_size)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / name, index=False)
    return f


def load_universe():
    gu = pd.read_parquet(P / "games_universe_v2.parquet",
                         columns=["game_id", "season", "season_type", "game_date", "home_team_id", "away_team_id",
                                  "neutral_site", "is_d1_game", "n_periods", "pbp_complete", "has_player_box"])
    gu = gu[gu.season.isin(SEASONS)].copy()
    gu["date"] = pd.to_datetime(gu.game_date)
    return gu


def load_team_games(d1_only=True, regular_only=True):
    """Team-game shot table (box counts) with date, home flag. One row per team per game."""
    t = pd.read_parquet(P / "truth/team_game_shots_v2.parquet",
                        columns=["game_id", "team_id", "opp_team_id", "season", "box_fgm", "box_fga", "box_fgm3", "box_fga3",
                                 "box_ftm", "box_fta", "box_fgm2", "box_fga2"])
    t = t[t.season.isin(SEASONS)]
    gu = load_universe()
    t = t.merge(gu[["game_id", "date", "season_type", "is_d1_game", "home_team_id", "neutral_site", "n_periods"]],
                on="game_id", how="inner")
    if d1_only:
        t = t[t.is_d1_game]
    if regular_only:
        t = t[t.season_type == 2]
    t["is_home"] = (t.team_id == t.home_team_id) & ~t.neutral_site
    t = t.dropna(subset=["box_fga", "box_fta", "box_fga2", "box_fga3"])
    # opponent columns (other row of the same game)
    o = t[["game_id", "team_id", "box_fga", "box_fta", "box_fga2", "box_fga3", "box_fgm2", "box_fgm3"]].rename(
        columns={"team_id": "opp_team_id", "box_fga": "opp_fga", "box_fta": "opp_fta", "box_fga2": "opp_fga2",
                 "box_fga3": "opp_fga3", "box_fgm2": "opp_fgm2", "box_fgm3": "opp_fgm3"})
    t = t.merge(o, on=["game_id", "opp_team_id"], how="inner")
    t = t.sort_values(["season", "date", "game_id", "team_id"]).reset_index(drop=True)
    t["gnum"] = t.groupby(["season", "team_id"]).cumcount()          # prior games of this team this season
    t["gtot"] = t.groupby(["season", "team_id"]).game_id.transform("size")
    return t


# metric -> (numerator col, denominator col); defence flavour handled by the caller via columns
METRICS = {
    "2P%": ("box_fgm2", "box_fga2"),
    "3P%": ("box_fgm3", "box_fga3"),
    "3PA_share": ("box_fga3", "box_fga"),
    "FTA_per_FGA": ("box_fta", "box_fga"),
}


def team_season_stats(t, num, den):
    """Per team-season: rate R, denominator total N, pooled within-team game variance s2 (of den-weighted deviations)."""
    d = t[t[den] > 0].copy()
    d["r"] = d[num] / d[den]
    g = d.groupby(["season", "team_id"])
    R = g[num].sum() / g[den].sum()
    N = g[den].sum()
    G = g.size()
    d = d.join(R.rename("R"), on=["season", "team_id"])
    d["ss"] = d[den] * (d.r - d.R) ** 2
    SS = d.groupby(["season", "team_id"]).ss.sum()
    out = pd.DataFrame({"R": R, "N": N, "G": G, "SS": SS}).reset_index()
    return out


def estimate_k(t, num, den, seasons, min_games=15):
    """Empirical-Bayes prior strength k (in denominator units) from the given (prior) seasons only.
    s2 = within-team per-attempt variance; tau2 = between-team variance of true rate; k = s2/tau2."""
    ts = team_season_stats(t[t.season.isin(seasons)], num, den)
    ts = ts[ts.G >= min_games]
    s2 = ts.SS.sum() / (ts.G - 1).sum()
    tau2s, mus = [], []
    for s, g in ts.groupby("season"):
        tau2s.append(g.R.var(ddof=1) - np.mean(s2 / g.N))
        mus.append(g.R.mean())
    tau2 = float(np.mean(tau2s))
    return dict(k=float(s2 / tau2), s2=float(s2), tau2=tau2, seasons=list(seasons), tau2_by_season=[float(x) for x in tau2s])


def asof_rates(t, num, den, team_col="team_id", k=None, prior_mu_by_season=None):
    """Adds as-of columns for one numerator/denominator pair, using ONLY strictly earlier dates in the same season.
    Returns frame with: y (game rate), mu (as-of league rate), cnum, cden (team's prior totals), xraw, xsh (centred ratings).
    Also leave-one-out (hindsight, NOT as-of) ratings: xloo_raw, xloo_sh.
    prior_mu_by_season: league rate of the previous season, used for league mu before any league games exist."""
    d = t.sort_values(["season", "date", "game_id"]).copy()
    d["_n"], d["_d"] = d[num].astype(float), d[den].astype(float)
    # league as-of: cumulative of all earlier DATES
    day = d.groupby(["season", "date"])[["_n", "_d"]].sum().reset_index()
    day["ln"] = day.groupby("season")._n.cumsum() - day._n
    day["ld"] = day.groupby("season")._d.cumsum() - day._d
    d = d.merge(day[["season", "date", "ln", "ld"]], on=["season", "date"], how="left")
    d["mu"] = np.where(d.ld > 200, d.ln / d.ld.replace(0, np.nan), np.nan)
    if prior_mu_by_season is not None:
        d["mu"] = d.mu.fillna(d.season.map(prior_mu_by_season))
    g = d.groupby(["season", team_col])
    d["cnum"] = g._n.cumsum() - d._n
    d["cden"] = g._d.cumsum() - d._d
    d["y"] = np.where(d._d > 0, d._n / d._d.replace(0, np.nan), np.nan)
    d["xraw"] = np.where(d.cden > 0, d.cnum / d.cden.replace(0, np.nan) - d.mu, np.nan)
    if k is not None:
        w = d.cden / (d.cden + k)
        d["xsh"] = np.where(d.cden > 0, w * d.xraw, 0.0)           # no prior games -> league mean (0)
        d["xsh"] = np.where(d.mu.isna(), np.nan, d.xsh)
    # hindsight leave-one-out
    tn, td = g._n.transform("sum"), g._d.transform("sum")
    on, od = tn - d._n, td - d._d
    ls = d.groupby("season")[["_n", "_d"]].transform("sum")
    muf = (ls._n / ls._d)
    d["xloo_raw"] = np.where(od > 0, on / od.replace(0, np.nan) - muf, np.nan)
    if k is not None:
        d["xloo_sh"] = od / (od + k) * d.xloo_raw
    d["muf"] = muf
    d["yc"] = d.y - d.mu
    d["w"] = d._d
    d["nprior"] = d.gnum
    return d


def wls_slope(y, x, w, cluster):
    """Weighted least squares y = a + b x; returns b and cluster-robust SE (CR1)."""
    y, x, w = map(np.asarray, (y, x, w))
    cl = pd.factorize(np.asarray(cluster))[0]
    X = np.column_stack([np.ones_like(x), x])
    W = w[:, None]
    A = X.T @ (W * X)
    beta = np.linalg.solve(A, X.T @ (w * y))
    e = y - X @ beta
    Ainv = np.linalg.inv(A)
    G = cl.max() + 1
    S = np.zeros((G, 2))
    np.add.at(S, cl, X * (w * e)[:, None])
    meat = S.T @ S
    n, p = len(y), 2
    adj = G / (G - 1) * (n - 1) / (n - p)
    V = adj * Ainv @ meat @ Ainv
    return float(beta[1]), float(np.sqrt(V[1, 1])), float(beta[0])
