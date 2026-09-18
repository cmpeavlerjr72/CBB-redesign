"""grade_ft_technical_round1b_v1.py -- L3 free-throw technicals round 1b.

Runs `docs/models/free_throw/experiments.md` section 10.2 (pre-registered
and pushed BEFORE this script ran, commit is named in the results doc).
OFFLINE ONLY, one grading script, blind -- no engine wiring, no closed-loop
run this round (same concurrency conditions as round 1, section 6).

Target: `data/processed/models/free_throw/technical_target_verified_v1.parquet`
/ `technical_target_verified_trips_v1.parquet` (section 10.1's adjudicated
target), replacing round 1's `trips_v1_era.parquet` technical count
everywhere. 2025-26 never loaded (`assert_not_sealed` on every season list).

Reuses round 1's own bucketing/scoring helpers by IMPORTING
`exp_free_throw_technicals_v1` rather than re-deriving them (game_phase,
margin_bucket, site_of, poisson_deviance, cell_tables, grid_search_shrinkage,
block_bootstrap_se, segment_table, fit_x1, fit_x5, team_asof_rate) -- round 1
is not re-run, its helper functions are shared code.

Output: results/free_throw_technicals/round1b_offline.json (gitignored).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_free_throw_technicals_v1 as R1  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402

RNG_SEED0 = 0
RNG_SEED1 = 1

FT_DIR = Path("data/processed/models/free_throw")
CHANCES_DIR = Path("data/processed/possessions_v2")
UNIVERSE_PATH = Path("data/processed/games_universe.parquet")
OUT_DIR = Path("results/free_throw_technicals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = FT.FOLDS
ALL_SEASONS = [2022, 2023, 2024, 2025]  # 2026 never touched

HALFLIFE_GRID = [0.5, 1.0, 2.0, 4.0, 999.0]  # seasons; 999 ~= uniform (X1)
SHRINK_GRID = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0,
               3200.0, 6400.0, 12800.0, 51200.0, 1e7]
LEVEL_CALIB_REL_TOL = 0.10  # eligibility line 2, section 10.2


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# verified target loading
# ---------------------------------------------------------------------------
def load_verified_trips(seasons: list[int]) -> pd.DataFrame:
    """Verified technical trips (section 10.1), one row per trip, with
    game_phase/margin_bucket/site attached the same way round 1's
    `load_technical_events` attached them to `trips_v1_era` rows -- except
    margin needs a fresh lookup here (the verified target's trip-grain file
    does not carry score_diff; round 1's did, straight off `trips_v1_era`).
    Margin is recovered from CBBD's own raw pbp running score
    (`homeScore`/`awayScore`, already loaded and matched by `game_id` in
    `build_ft_technical_target_v1.build_cbbd_raw_scan`'s universe join) at
    the exact technical-foul row -- a pre-outcome quantity (`_Chance`-style:
    the score AT THE MOMENT the technical was called, before its own free
    throws land, since a technical never changes `scoreValue`)."""
    assert_not_sealed(seasons, context="verified trips load")
    trips = pd.read_parquet(FT_DIR / "technical_target_verified_trips_v1.parquet")
    trips = trips[trips["season"].isin(seasons)].copy()
    trips = trips.rename(columns={"n_attempts_verified": "n_attempts"})

    universe = pd.read_parquet(UNIVERSE_PATH)
    margin_rows = []
    for s in seasons:
        cols = ["gameId", "id", "playType", "period", "secondsRemaining", "homeScore", "awayScore"]
        p = pd.read_parquet(Path("data/raw/cbbd/pbp") / f"plays_{s}.parquet", columns=cols)
        p = p.drop_duplicates(subset=["gameId", "id"], keep="first")
        p = p[p["playType"] == "Technical Foul"]
        u = universe[universe["season"] == s][
            ["cbbd_game_id", "game_id", "home_team_id", "away_team_id", "neutral_site", "game_date"]
        ].drop_duplicates("cbbd_game_id")
        p = p.merge(u, left_on="gameId", right_on="cbbd_game_id", how="inner")
        p = p.drop_duplicates(subset=["game_id", "period", "secondsRemaining"], keep="first")
        p = p.rename(columns={"secondsRemaining": "clock"})
        p["season"] = s
        margin_rows.append(p[["season", "game_id", "period", "clock", "homeScore", "awayScore",
                               "home_team_id", "away_team_id", "neutral_site", "game_date"]])
    margins = pd.concat(margin_rows, ignore_index=True)

    trips = trips.merge(margins, on=["season", "game_id", "period", "clock"], how="left")
    # `beneficiary` is the team that SHOOTS the technical FTs -- the OFFENDER
    # is the other team in the game. R1's own convention (`load_technical_events`,
    # 9.9.2) keys margin_bucket/site/the team-quintile rate on the OFFENDER's
    # own state ("does a trailing/away team commit more technicals"), so that
    # convention is reproduced here exactly, not the beneficiary's.
    trips["offender_team_id"] = np.where(trips["beneficiary"] == trips["home_team_id"],
                                          trips["away_team_id"], trips["home_team_id"])
    trips["offender_is_home"] = np.where(trips["neutral_site"].fillna(False), False,
                                          trips["offender_team_id"] == trips["home_team_id"])
    trips["margin"] = np.where(trips["offender_is_home"],
                                trips["homeScore"] - trips["awayScore"],
                                trips["awayScore"] - trips["homeScore"])
    trips["game_phase"] = R1.game_phase(trips["period"].to_numpy(), trips["clock"].to_numpy())
    trips["margin_bucket"] = R1.margin_bucket(trips["margin"].fillna(0).to_numpy())
    trips["site"] = R1.site_of(trips["offender_is_home"].to_numpy(), trips["neutral_site"].fillna(False).to_numpy())
    # R1's imported helpers (`fit_x5`'s internal grid search, `cell_tables`)
    # are hard-coded to a `team_id` column exactly the way
    # `load_technical_events` left `trips_v1_era`'s OWN `team_id` (the
    # beneficiary/shooter's team) in place while ADDING `offender_team_id`
    # alongside it, rather than renaming it -- reproduced here unchanged so
    # the imported functions run without modification and round 1b's `X5`
    # inherits round 1's exact (if slightly inconsistent -- the shrinkage
    # STRENGTH search keys on the beneficiary while the applied team rate,
    # `team_asof_rate`, keys on `offender_team_id`) convention rather than
    # silently changing it.
    trips["team_id"] = trips["beneficiary"]
    return trips


def load_exposure(seasons: list[int]) -> pd.DataFrame:
    assert_not_sealed(seasons, context="exposure load (round 1b)")
    universe = pd.read_parquet(UNIVERSE_PATH)
    frames = []
    for s in seasons:
        df = pd.read_parquet(CHANCES_DIR / f"chances_{s}.parquet",
                              columns=["game_id", "season", "period", "start_clock",
                                       "start_score_diff", "offense_team_id",
                                       "defense_team_id", "offense_is_home"])
        frames.append(df)
    chances = pd.concat(frames, ignore_index=True)
    exposure = R1.build_exposure(chances, universe)
    u_small = universe[["game_id", "season", "game_date"]].copy()
    exposure = exposure.merge(u_small, on=["game_id", "season"], how="left")
    return exposure


# ---------------------------------------------------------------------------
# X6: recency-weighted training window
# ---------------------------------------------------------------------------
def season_rate(exposure: pd.DataFrame, tech: pd.DataFrame, season: int) -> tuple[int, int]:
    e = int((exposure["season"] == season).sum())
    c = int((tech["season"] == season).sum())
    return e, c


def choose_halflife(exposure_tr: pd.DataFrame, tech_tr: pd.DataFrame, train_seasons: list[int],
                     grid: list[float]) -> tuple[float, dict]:
    if len(train_seasons) < 2:
        return grid[-1], {"note": "single train season; recency weighting degenerates to X1 (halflife=inf)"}
    val_season = max(train_seasons)
    fit_seasons = [s for s in train_seasons if s != val_season]
    rates = {s: season_rate(exposure_tr, tech_tr, s) for s in fit_seasons}
    e_val, c_val = season_rate(exposure_tr, tech_tr, val_season)

    best_hl, best_dev, devs = grid[0], np.inf, {}
    for hl in grid:
        w = {s: 2.0 ** (-(val_season - s) / hl) for s in fit_seasons}
        num = sum(w[s] * rates[s][1] for s in fit_seasons)
        den = sum(w[s] * rates[s][0] for s in fit_seasons)
        rate = num / den if den > 0 else 0.0
        dev = R1.poisson_deviance(np.array([c_val]), np.array([rate * e_val]))
        devs[hl] = dev
        if dev < best_dev:
            best_dev, best_hl = dev, hl
    return best_hl, {"val_season": val_season, "fit_seasons": fit_seasons, "deviance_by_halflife": devs}


def fit_x6_rate(exposure_tr: pd.DataFrame, tech_tr: pd.DataFrame, train_seasons: list[int],
                 halflife: float, relative_to_season: int) -> float:
    """Final recency-weighted pooled rate, weights taken relative to the
    season being PREDICTED (the real test season), not to the last train
    season -- the halflife itself was chosen on an internal (train-only)
    validation split above; this call only fixes the weighting anchor."""
    w = {s: 2.0 ** (-(relative_to_season - s) / halflife) for s in train_seasons}
    num = sum(w[s] * season_rate(exposure_tr, tech_tr, s)[1] for s in train_seasons)
    den = sum(w[s] * season_rate(exposure_tr, tech_tr, s)[0] for s in train_seasons)
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# X7: league-level in-season as-of rate x site
# ---------------------------------------------------------------------------
def fit_x7(exposure_tr: pd.DataFrame, tech_tr: pd.DataFrame, exposure_te: pd.DataFrame, tech_te: pd.DataFrame,
           train_seasons: list[int], test_season: int, x6_rate: float, grid: list[float]) -> tuple[pd.Series, dict]:
    """Per-test-row predicted rate = EB blend of (a) the test season's own
    cumulative rate strictly before that row's game_date and (b) `x6_rate`
    (the recency-weighted prior level), pseudo-exposure k grid-searched,
    times a site ratio fit on the training pool."""
    # site ratio, training pool
    site_overall = len(tech_tr) / max(len(exposure_tr), 1)
    site_rates = {}
    for site in ["home", "away", "neutral"]:
        e = int((exposure_tr["site"] == site).sum())
        c = int((tech_tr["site"] == site).sum())
        site_rates[site] = (c / e if e > 0 else site_overall) / site_overall if site_overall > 0 else 1.0

    # daily cumulative test-season counts (pre-outcome: date < row's date)
    exp_by_date = exposure_te.groupby("game_date").size().sort_index()
    tech_by_date = tech_te.groupby("game_date").size().sort_index()
    all_dates = exp_by_date.index.union(tech_by_date.index).sort_values()
    cum_exp = exp_by_date.reindex(all_dates, fill_value=0).cumsum()
    cum_tech = tech_by_date.reindex(all_dates, fill_value=0).cumsum()
    # shift by one date so "before this date" excludes the date's own games
    cum_exp_before = cum_exp.shift(1, fill_value=0)
    cum_tech_before = cum_tech.shift(1, fill_value=0)

    # k fit: one pseudo-fold using the LAST training season as a stand-in
    # test season (same thin-data convention `grid_search_shrinkage`
    # already uses elsewhere in this lane when <2 seasons are available for
    # a proper internal split -- here there is always >=2 train seasons, but
    # only ONE of them can stand in for "a season not yet used to build the
    # prior", so one pseudo-fold is what the data supports).
    pseudo_season = max(train_seasons)
    pseudo_prior_seasons = [s for s in train_seasons if s != pseudo_season]
    if pseudo_prior_seasons:
        pseudo_prior_rate = sum(season_rate(exposure_tr, tech_tr, s)[1] for s in pseudo_prior_seasons) / \
            max(sum(season_rate(exposure_tr, tech_tr, s)[0] for s in pseudo_prior_seasons), 1)
        exp_pseudo = exposure_tr[exposure_tr["season"] == pseudo_season]
        tech_pseudo = tech_tr[tech_tr["season"] == pseudo_season]
        pe = exp_pseudo.groupby("game_date").size().sort_index()
        pt = tech_pseudo.groupby("game_date").size().sort_index()
        pd_dates = pe.index.union(pt.index).sort_values()
        pce = pe.reindex(pd_dates, fill_value=0).cumsum().shift(1, fill_value=0)
        pct = pt.reindex(pd_dates, fill_value=0).cumsum().shift(1, fill_value=0)
        best_k, best_dev, devs = grid[0], np.inf, {}
        for k in grid:
            blended = (pct + k * pseudo_prior_rate) / (pce + k).replace(0, np.nan)
            blended = blended.fillna(pseudo_prior_rate)
            # score against that date's OWN day count (Poisson per date-bucket)
            mu = (blended * pe.reindex(pd_dates, fill_value=0)).sum()
            actual = pt.reindex(pd_dates, fill_value=0).sum()
            dev = R1.poisson_deviance(np.array([actual]), np.array([mu]))
            devs[k] = dev
            if dev < best_dev:
                best_dev, best_k = dev, k
        k_meta = {"pseudo_season": pseudo_season, "deviance_by_k": devs, "chosen_k": best_k}
    else:
        best_k = grid[len(grid) // 2]
        k_meta = {"note": "no prior training season available for the pseudo-fold; mid-grid k used"}

    blended_by_date = (cum_tech_before + best_k * x6_rate) / (cum_exp_before + best_k).replace(0, np.nan)
    blended_by_date = blended_by_date.fillna(x6_rate)

    return blended_by_date, {"k": best_k, "k_meta": k_meta, "site_rates": site_rates}


# ---------------------------------------------------------------------------
# X3 shooter-rule variants (train/test-safe, unlike round 1's pooled version)
# ---------------------------------------------------------------------------
def who_shoots_fold(train_seasons: list[int], test_season: int) -> dict:
    attempts = pd.read_parquet(FT_DIR / "attempts_v1_era.parquet")
    design_tr = FT.build_ft_design(attempts[attempts["season"].isin(train_seasons)], include_technical=True)
    design_te = FT.build_ft_design(attempts[attempts["season"] == test_season], include_technical=True)

    def context(design):
        tech = design[design["foul_class"] == "technical"].copy()
        non_tech = design[design["foul_class"] != "technical"].copy()
        team_avg = non_tech.groupby(["season", "team_id", "game_id"])["shooter_ft_raw"].mean().rename(
            "avg").reset_index()
        team_best = non_tech.groupby(["season", "team_id", "game_id"])["shooter_ft_raw"].max().rename(
            "best").reset_index()
        tech1 = tech.drop_duplicates(["season", "team_id", "game_id", "trip_id"])
        m = tech1.merge(team_avg, on=["season", "team_id", "game_id"], how="left")
        m = m.merge(team_best, on=["season", "team_id", "game_id"], how="left")
        return m.dropna(subset=["avg", "best"])

    m_tr = context(design_tr)
    m_te = context(design_te)
    if len(m_tr) == 0 or len(m_te) == 0:
        return {"n_train": len(m_tr), "n_test": len(m_te), "note": "insufficient context, skipped"}

    # fit blend fraction f on TRAIN only: minimize sum of squared error
    # between (avg + f*(best-avg)) and the actual chosen shooter's as-of rate
    diff = (m_tr["best"] - m_tr["avg"]).to_numpy()
    resid = (m_tr["shooter_ft_raw"] - m_tr["avg"]).to_numpy()
    denom = float((diff ** 2).sum())
    f_fit = float((diff * resid).sum() / denom) if denom > 0 else 0.5
    f_fit = float(np.clip(f_fit, 0.0, 1.0))

    out = {"n_train": len(m_tr), "n_test": len(m_te), "fitted_blend_f": f_fit,
           "actual_test_shooter_rate": float(m_te["shooter_ft_raw"].mean())}
    for label, pred in [("X3_avg", m_te["avg"]), ("X3_best", m_te["best"]),
                        ("X3_blend", m_te["avg"] + f_fit * (m_te["best"] - m_te["avg"]))]:
        out[f"{label}_predicted_mean"] = float(pred.mean())
        out[f"{label}_gap_pp"] = float((pred.mean() - m_te["shooter_ft_raw"].mean()) * 100)
    return out


# ---------------------------------------------------------------------------
# per-fold run
# ---------------------------------------------------------------------------
def run_fold(fold: str, exposure_all: pd.DataFrame, tech_all: pd.DataFrame) -> dict:
    spec = FOLDS[fold]
    train_seasons, test_seasons = spec["train"], spec["test"]
    test_season = test_seasons[0]
    assert_not_sealed(train_seasons, context=f"{fold} train")
    assert_not_sealed(test_seasons, context=f"{fold} test")

    exposure_tr = exposure_all[exposure_all["season"].isin(train_seasons)]
    exposure_te = exposure_all[exposure_all["season"].isin(test_seasons)]
    tech_tr = tech_all[tech_all["season"].isin(train_seasons)]
    tech_te = tech_all[tech_all["season"].isin(test_seasons)]

    # ---- X1 ----
    x1_rate = R1.fit_x1(exposure_tr, tech_tr)

    # ---- X5 (shrunk toward X1) ----
    x5_k, x5_meta = R1.fit_x5(exposure_tr, tech_tr, train_seasons, x1_rate, SHRINK_GRID)
    x5_team = R1.team_asof_rate(exposure_tr, tech_tr, x5_k, x1_rate, test_season)

    # ---- X6 (recency-weighted) ----
    halflife, hl_meta = choose_halflife(exposure_tr, tech_tr, train_seasons, HALFLIFE_GRID)
    x6_rate = fit_x6_rate(exposure_tr, tech_tr, train_seasons, halflife, test_season)

    # ---- X5r (X5's shrinkage, but toward X6's level) ----
    x5r_k, x5r_meta = R1.fit_x5(exposure_tr, tech_tr, train_seasons, x6_rate, SHRINK_GRID)
    x5r_team = R1.team_asof_rate(exposure_tr, tech_tr, x5r_k, x6_rate, test_season)

    # ---- X7 (league in-season as-of x site) ----
    blended_by_date, x7_meta = fit_x7(exposure_tr, tech_tr, exposure_te, tech_te,
                                       train_seasons, test_season, x6_rate, SHRINK_GRID)

    def predict_x0(exp_df):
        return np.zeros(len(exp_df))

    def predict_x1(exp_df):
        return np.full(len(exp_df), x1_rate)

    def predict_x5(exp_df):
        m = exp_df.merge(x5_team[["team_id", "rate"]], on="team_id", how="left")
        return m["rate"].fillna(x1_rate).to_numpy()

    def predict_x6(exp_df):
        return np.full(len(exp_df), x6_rate)

    def predict_x5r(exp_df):
        m = exp_df.merge(x5r_team[["team_id", "rate"]], on="team_id", how="left")
        return m["rate"].fillna(x6_rate).to_numpy()

    def predict_x7(exp_df):
        base = exp_df["game_date"].map(blended_by_date).to_numpy(dtype=float)
        base = np.where(np.isnan(base), x6_rate, base)
        site_mult = exp_df["site"].map(x7_meta["site_rates"]).to_numpy(dtype=float)
        return base * site_mult

    arms = {"X0": predict_x0, "X1": predict_x1, "X5": predict_x5,
            "X6": predict_x6, "X5r": predict_x5r, "X7": predict_x7}
    overall = {name: R1.score_predicted_vs_actual(exposure_te, tech_te, fn, name) for name, fn in arms.items()}

    # ---- eligibility line 2: level calibration ----
    actual_test_rate = len(tech_te) / max(len(exposure_te), 1)
    level_calib = {}
    for name, row in overall.items():
        pred_rate = row["predicted_rate_per_chance"]
        rel_gap = (pred_rate - actual_test_rate) / actual_test_rate if actual_test_rate > 0 else np.nan
        level_calib[name] = {"predicted_rate": pred_rate, "actual_rate": actual_test_rate,
                              "rel_gap": rel_gap, "passes": bool(abs(rel_gap) <= LEVEL_CALIB_REL_TOL)}

    # ---- segments ----
    segments = {}
    for name, fn in arms.items():
        segments[name] = {
            "game_phase": R1.segment_table(exposure_te, tech_te, fn, "game_phase").to_dict("records"),
            "margin_bucket": R1.segment_table(exposure_te, tech_te, fn, "margin_bucket").to_dict("records"),
            "site": R1.segment_table(exposure_te, tech_te, fn, "site").to_dict("records"),
        }
    u_small = pd.read_parquet(UNIVERSE_PATH)[["game_id", "season", "game_date"]].copy()
    u_small["month"] = pd.to_datetime(u_small["game_date"]).dt.month
    exposure_te_ctx = exposure_te.merge(u_small, on=["game_id", "season"], how="left", suffixes=("", "_u"))
    tech_te_ctx = tech_te.merge(u_small, on=["game_id", "season"], how="left", suffixes=("", "_u"))
    for name, fn in arms.items():
        segments[name]["month"] = R1.segment_table(exposure_te_ctx, tech_te_ctx, fn, "month").to_dict("records")

    # ---- eligibility line 1: team prior-quintile responsiveness (POWERED) ----
    prior_szn = test_season - 1
    exp_prior = exposure_all[exposure_all["season"] == prior_szn]
    tech_prior = tech_all[tech_all["season"] == prior_szn]
    prior_team = exp_prior.groupby("team_id").size().rename("exposure").reset_index().merge(
        tech_prior.groupby("offender_team_id").size().rename("count").reset_index().rename(
            columns={"offender_team_id": "team_id"}), on="team_id", how="left")
    prior_team["count"] = prior_team["count"].fillna(0)
    prior_team["prior_rate"] = prior_team["count"] / prior_team["exposure"].replace(0, np.nan)
    prior_team = prior_team.dropna(subset=["prior_rate"])
    prior_team = prior_team[prior_team["exposure"] > 500]
    try:
        prior_team["quintile"] = pd.qcut(prior_team["prior_rate"], 5, labels=False, duplicates="drop")
    except ValueError:
        prior_team["quintile"] = 0

    test_team_exp = exposure_te.groupby("team_id").size().rename("exposure_test").reset_index()
    test_team_cnt = tech_te.groupby("offender_team_id").size().rename("count_test").reset_index().rename(
        columns={"offender_team_id": "team_id"})
    qtab = prior_team.merge(test_team_exp, on="team_id", how="left").merge(test_team_cnt, on="team_id", how="left")
    qtab["count_test"] = qtab["count_test"].fillna(0)
    qtab["exposure_test"] = qtab["exposure_test"].fillna(0)

    team_arms = {"X0": predict_x0, "X1": predict_x1, "X5": predict_x5, "X5r": predict_x5r}
    exp_te_team = exposure_te[["team_id"]].drop_duplicates().reset_index(drop=True)
    for aname, afn in team_arms.items():
        exp_te_team[f"pred_rate_{aname}"] = afn(exp_te_team)
    qtab = qtab.merge(exp_te_team, on="team_id", how="left")

    qsum = qtab.groupby("quintile").agg(
        n_teams=("team_id", "size"), prior_rate_mean=("prior_rate", "mean"),
        exposure_test=("exposure_test", "sum"), count_test=("count_test", "sum")).reset_index()
    for aname in team_arms:
        pred_trips = (qtab[f"pred_rate_{aname}"] * qtab["exposure_test"]).groupby(qtab["quintile"]).sum()
        qsum[f"pred_rate_{aname}"] = (pred_trips / qsum.set_index("quintile")["exposure_test"]).to_numpy()
    qsum["actual_test_rate"] = qsum["count_test"] / qsum["exposure_test"].replace(0, np.nan)
    qsum["expected_trips_at_league_rate"] = qsum["exposure_test"] * actual_test_rate
    qsum["poisson_se_trips"] = np.sqrt(qsum["expected_trips_at_league_rate"].clip(lower=0))
    qsum["poisson_se_rate"] = qsum["poisson_se_trips"] / qsum["exposure_test"].replace(0, np.nan)
    qsum["underpowered"] = qsum["count_test"] < 30

    def slope_ratio(pred_col):
        vals = qsum[pred_col].to_numpy()
        actual_vals = qsum["actual_test_rate"].to_numpy()
        if actual_vals[0] == 0 or (actual_vals[-1] - actual_vals[0]) == 0:
            return np.nan
        return float((vals[-1] - vals[0]) / (actual_vals[-1] - actual_vals[0]))

    def monotone_steps(pred_col):
        vals = qsum[pred_col].to_numpy()
        # a rate-constant arm (X0, X1) is flat BY CONSTRUCTION (round 1's own
        # finding, section 9.9.4/3.1): the aggregated per-quintile prediction
        # for a true constant is the same value repeated, up to floating-
        # point noise from summing over different-sized quintile exposure
        # pools. Without this guard a spuriously-passing or spuriously-
        # failing "monotone" count for X0/X1 is a coin flip on sub-ULP jitter
        # (observed directly: X1 read 4/4 on F1 and 2/4 on F2 before this fix,
        # for the identical construction-flat reason on both folds) --
        # reported here as constructed-flat rather than left to that noise.
        if np.ptp(vals) < 1e-9 * max(abs(vals).max(), 1e-12):
            return 0
        return int(sum(1 for i in range(len(vals) - 1) if vals[i + 1] >= vals[i]))

    responsiveness = {}
    for aname in team_arms:
        responsiveness[aname] = {
            "slope_ratio": slope_ratio(f"pred_rate_{aname}"),
            "monotone_steps_of_4": monotone_steps(f"pred_rate_{aname}"),
            "passes": bool(monotone_steps(f"pred_rate_{aname}") >= 3),  # 3-of-4, same reading as FT-2 elsewhere
        }
    end_gap = float(qsum["actual_test_rate"].iloc[-1] - qsum["actual_test_rate"].iloc[0])
    end_se = float(np.sqrt(qsum["poisson_se_rate"].iloc[0] ** 2 + qsum["poisson_se_rate"].iloc[-1] ** 2))
    power_sigma = end_gap / end_se if end_se > 0 else np.nan

    noise_floor_seed0 = R1.block_bootstrap_se(exposure_tr, tech_tr, n_boot=200, seed=RNG_SEED0)
    noise_floor_seed1 = R1.block_bootstrap_se(exposure_tr, tech_tr, n_boot=200, seed=RNG_SEED1)

    who_shoots = who_shoots_fold(train_seasons, test_season)

    return {
        "fold": fold, "train_seasons": train_seasons, "test_seasons": test_seasons,
        "x1_rate": x1_rate, "x6_rate": x6_rate, "halflife_chosen": halflife, "halflife_meta": hl_meta,
        "x5_k": x5_k, "x5r_k": x5r_k, "x7_meta": {k: v for k, v in x7_meta.items() if k != "site_rates"},
        "x7_site_rates": x7_meta["site_rates"],
        "overall": overall, "level_calibration": level_calib, "segments": segments,
        "quintile_responsiveness": qsum.to_dict("records"), "responsiveness": responsiveness,
        "quintile_power_sigma": power_sigma,
        "noise_floor_seed0": noise_floor_seed0, "noise_floor_seed1": noise_floor_seed1,
        "noise_floor_used": max(noise_floor_seed0, noise_floor_seed1),
        "who_shoots": who_shoots,
    }


def main():
    t0 = time.time()
    exposure_all = load_exposure(ALL_SEASONS)
    log(f"exposure loaded: {len(exposure_all):,} rows", t0)
    tech_all = load_verified_trips(ALL_SEASONS)
    log(f"verified technical trips loaded: {len(tech_all):,} rows", t0)

    results = {}
    for fold in ["F1", "F2"]:
        log(f"running fold {fold}", t0)
        results[fold] = run_fold(fold, exposure_all, tech_all)
        log(f"fold {fold} done", t0)

    out_path = OUT_DIR / "round1b_offline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"written {out_path}", t0)

    print("\n=== F2 (SELECTION) overall ===")
    for name, row in results["F2"]["overall"].items():
        print(name, row)
    print("\nlevel calibration (F2):")
    for name, row in results["F2"]["level_calibration"].items():
        print(name, row)
    print("\nresponsiveness (F2):")
    for name, row in results["F2"]["responsiveness"].items():
        print(name, row)
    print("\nquintile power sigma (F2):", results["F2"]["quintile_power_sigma"])
    print("\nwho-shoots (F2):", results["F2"]["who_shoots"])
    print("\nwall clock total:", time.time() - t0, "s")


if __name__ == "__main__":
    main()
