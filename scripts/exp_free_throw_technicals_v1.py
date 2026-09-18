"""exp_free_throw_technicals_v1.py -- L3 free-throw technicals round.

Runs docs/models/free_throw/experiments.md section 9 (PROPOSED 2026-09-18) as
amended by section 9.9 (PM conditions, same date, committed BEFORE this
script ran). OFFLINE ONLY: no engine wiring, no served default touched, no
closed-loop paired-seed run (see the results doc for why this round stops at
the offline table).

Arms: X0 (reference, no technicals), X1 (league-constant rate per team-chance),
X4 (in-game state: game_phase x margin_bucket x site, EB-shrunk), X5 (team
as-of rate, EB-shrunk toward X1). X2 (pre-game static covariates) and X3
(shooter-rule variant of X1) are also scored, X3 via a dedicated who-shoots
comparison (section 9.9.1 / 9.9.4) rather than a rate re-fit, since X3's rate
is identical to X1's by construction (9.1).

2025-26 (season 2026) is never loaded by this script. `assert_not_sealed` is
called on every train/test season list before any file for that season is
opened.

Output: results/free_throw_technicals/round1_offline.json (gitignored,
>20MB artifact dirs are not tracked; this one is small but is written to the
same results/ tree as every other engine run for consistency) and printed
tables captured into the same JSON for the results doc to quote verbatim.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.features import conference as CONF  # noqa: E402

RNG_SEED0 = 0
RNG_SEED1 = 1  # noise-floor second seed (bootstrap resample seed, not a model seed)

TRIPS_PATH = Path("data/processed/models/free_throw/trips_v1_era.parquet")
CHANCES_DIR = Path("data/processed/possessions_v2")
UNIVERSE_PATH = Path("data/processed/games_universe.parquet")
OUT_DIR = Path("results/free_throw_technicals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = FT.FOLDS  # {"F1": {"train":[2022,2023], "test":[2024]}, "F2": {"train":[2022,2023,2024], "test":[2025]}}
ALL_SEASONS = [2022, 2023, 2024, 2025]  # 2026 never touched


def log(msg: str, t0: float) -> None:
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# bucketing (shared by exposure rows and technical-event rows)
# ---------------------------------------------------------------------------
def game_minute(period: np.ndarray, start_clock: np.ndarray) -> np.ndarray:
    """Minutes elapsed since game start, from (period, seconds-remaining-in-period)."""
    period = np.asarray(period, dtype=np.int64)
    start_clock = np.asarray(start_clock, dtype=np.float64)
    reg = period <= 2
    elapsed = np.where(reg,
                        (period - 1) * 1200.0 + (1200.0 - start_clock),
                        2400.0 + (period - 3) * 300.0 + (300.0 - start_clock))
    return elapsed / 60.0


def game_phase(period: np.ndarray, start_clock: np.ndarray) -> np.ndarray:
    m = game_minute(period, start_clock)
    return np.select(
        [m < 10, (m >= 10) & (m < 20), (m >= 20) & (m < 30), m >= 30],
        ["H1_early", "H1_late", "H2_early", "H2_late_OT"],
        default="H2_late_OT")


def margin_bucket(margin: np.ndarray) -> np.ndarray:
    margin = np.asarray(margin, dtype=np.float64)
    return np.select([margin <= -5, margin >= 5], ["trailing", "leading"], default="close")


def site_of(is_home: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    is_home = np.asarray(is_home, dtype=bool)
    neutral = np.asarray(neutral, dtype=bool)
    return np.where(neutral, "neutral", np.where(is_home, "home", "away"))


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_chances(seasons: list[int]) -> pd.DataFrame:
    assert_not_sealed(seasons, context="chances load")
    frames = []
    for s in seasons:
        p = CHANCES_DIR / f"chances_{s}.parquet"
        df = pd.read_parquet(p, columns=["game_id", "season", "period", "start_clock",
                                          "start_score_diff", "offense_team_id",
                                          "defense_team_id", "offense_is_home"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_exposure(chances: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, chance) -- each chance exposes BOTH teams once."""
    u = universe[["game_id", "neutral_site"]].drop_duplicates("game_id")
    c = chances.merge(u, on="game_id", how="left")
    c["neutral_site"] = c["neutral_site"].fillna(False)

    off = pd.DataFrame({
        "game_id": c["game_id"], "season": c["season"], "team_id": c["offense_team_id"],
        "period": c["period"], "start_clock": c["start_clock"],
        "margin": c["start_score_diff"], "is_home": c["offense_is_home"],
        "neutral": c["neutral_site"],
    })
    deff = pd.DataFrame({
        "game_id": c["game_id"], "season": c["season"], "team_id": c["defense_team_id"],
        "period": c["period"], "start_clock": c["start_clock"],
        "margin": -c["start_score_diff"], "is_home": ~c["offense_is_home"] & ~c["neutral_site"],
        "neutral": c["neutral_site"],
    })
    both = pd.concat([off, deff], ignore_index=True)
    both["game_phase"] = game_phase(both["period"].to_numpy(), both["start_clock"].to_numpy())
    both["margin_bucket"] = margin_bucket(both["margin"].to_numpy())
    both["site"] = site_of(both["is_home"].to_numpy(), both["neutral"].to_numpy())
    return both


def load_technical_events(seasons: list[int]) -> pd.DataFrame:
    assert_not_sealed(seasons, context="technical trips load")
    t = pd.read_parquet(TRIPS_PATH)
    t = t[t["season"].isin(seasons) & (t["foul_class"] == "technical")].copy()
    # offender = opp_id (the team that was called); shooter's team_id is the BENEFICIARY
    t["offender_team_id"] = t["opp_id"]
    t["offender_is_home"] = np.where(t["neutral_site"].to_numpy(), False, ~t["shooter_is_home"].to_numpy())
    t["margin"] = -t["score_diff"].to_numpy()  # offender's own margin
    t["game_phase"] = game_phase(t["period"].to_numpy(), t["seconds_remaining"].to_numpy())
    t["margin_bucket"] = margin_bucket(t["margin"].to_numpy())
    t["site"] = site_of(t["offender_is_home"].to_numpy(), t["neutral_site"].to_numpy())
    return t


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def fit_x1(exposure_tr: pd.DataFrame, tech_tr: pd.DataFrame) -> float:
    return len(tech_tr) / max(len(exposure_tr), 1)


def cell_tables(exposure: pd.DataFrame, tech: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    exp_g = exposure.groupby(keys).size().rename("exposure").reset_index()
    tech_g = tech.groupby(keys).size().rename("count").reset_index()
    out = exp_g.merge(tech_g, on=keys, how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    return out


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-12)
    y = np.asarray(y, dtype=np.float64)
    term = np.where(y > 0, y * np.log(y / mu) - (y - mu), mu)
    return float(2.0 * term.sum())


def grid_search_shrinkage(exposure_tr: pd.DataFrame, tech_tr: pd.DataFrame, keys: list[str],
                           seasons_train: list[int], grid: list[float], league_rate: float) -> tuple[float, dict]:
    """Chronological internal split: fit cells on all-but-last train season,
    validate (Poisson deviance) on the last train season, pick k off that."""
    if len(seasons_train) < 2:
        return grid[len(grid) // 2], {"note": "single-season train fold; grid search skipped, mid-grid k used"}
    val_season = max(seasons_train)
    fit_seasons = [s for s in seasons_train if s != val_season]
    exp_fit = exposure_tr[exposure_tr["season"].isin(fit_seasons)]
    tech_fit = tech_tr[tech_tr["season"].isin(fit_seasons)]
    exp_val = exposure_tr[exposure_tr["season"] == val_season]
    tech_val = tech_tr[tech_tr["season"] == val_season]

    fit_cells = cell_tables(exp_fit, tech_fit, keys)
    val_cells = cell_tables(exp_val, tech_val, keys)
    merged = fit_cells.merge(val_cells, on=keys, how="outer", suffixes=("_fit", "_val")).fillna(0)

    results = {}
    best_k, best_dev = grid[0], np.inf
    for k in grid:
        rate = (merged["count_fit"] + k * league_rate) / (merged["exposure_fit"] + k)
        mu = rate.to_numpy() * merged["exposure_val"].to_numpy()
        dev = poisson_deviance(merged["count_val"].to_numpy(), mu)
        results[k] = dev
        if dev < best_dev:
            best_dev, best_k = dev, k
    return best_k, {"val_season": val_season, "fit_seasons": fit_seasons, "deviance_by_k": results, "chosen_k": best_k}


def fit_x4(exposure_tr: pd.DataFrame, tech_tr: pd.DataFrame, seasons_train: list[int],
           league_rate: float, grid: list[float]) -> tuple[pd.DataFrame, dict]:
    keys = ["game_phase", "margin_bucket", "site"]
    k, meta = grid_search_shrinkage(exposure_tr, tech_tr, keys, seasons_train, grid, league_rate)
    cells = cell_tables(exposure_tr, tech_tr, keys)
    cells["rate"] = (cells["count"] + k * league_rate) / (cells["exposure"] + k)
    cells["underpowered"] = cells["count"] < 20
    meta["k"] = k
    return cells, meta


def fit_x5(exposure_tr: pd.DataFrame, tech_tr: pd.DataFrame, seasons_train: list[int],
           league_rate: float, grid: list[float]) -> tuple[float, dict]:
    """Team as-of rate shrunk toward the pooled league rate (fold-level constant,
    not a moving daily target -- documented simplification, 9.9.3)."""
    keys = ["team_id"]
    k, meta = grid_search_shrinkage(exposure_tr, tech_tr, keys, seasons_train, grid, league_rate)
    meta["k"] = k
    return k, meta


def team_asof_rate(exposure_all: pd.DataFrame, tech_all: pd.DataFrame, k: float,
                    league_rate: float, upto_season: int) -> pd.DataFrame:
    """Each team's as-of rate on games strictly before `upto_season`'s test
    games -- built at the SEASON grain (train seasons pooled, cumulative),
    which is what a between-season EB rate needs here since exposure is by
    season already. Returns one row per team_id with a shrunk rate to apply
    uniformly to every one of that team's test-season chances."""
    exp_g = exposure_all.groupby(["team_id", "game_id"]).size().rename("exposure").reset_index()
    exp_g = exp_g.groupby("team_id")["exposure"].sum().rename("exposure").reset_index()
    tech_g = tech_all.groupby("offender_team_id").size().rename("count").reset_index().rename(
        columns={"offender_team_id": "team_id"})
    out = exp_g.merge(tech_g, on="team_id", how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    out["rate"] = (out["count"] + k * league_rate) / (out["exposure"] + k)
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_predicted_vs_actual(exposure_te: pd.DataFrame, tech_te: pd.DataFrame,
                               predict_fn, label: str) -> dict:
    n_exp = len(exposure_te)
    n_actual = len(tech_te)
    pred_rate = exposure_te.assign(p=predict_fn(exposure_te))
    mu_total = float(pred_rate["p"].sum())
    dev = poisson_deviance(np.array([n_actual]), np.array([mu_total]))
    actual_rate = n_actual / max(n_exp, 1)
    pred_pooled_rate = mu_total / max(n_exp, 1)
    return {
        "arm": label, "n_exposure": n_exp, "n_actual_trips": n_actual,
        "predicted_trips": round(mu_total, 2),
        "actual_rate_per_chance": actual_rate, "predicted_rate_per_chance": pred_pooled_rate,
        "pooled_poisson_deviance": dev,
    }


def segment_table(exposure_te: pd.DataFrame, tech_te: pd.DataFrame, predict_fn, key: str) -> pd.DataFrame:
    exp_g = exposure_te.assign(p=predict_fn(exposure_te)).groupby(key).agg(
        exposure=("p", "size"), predicted_trips=("p", "sum")).reset_index()
    tech_g = tech_te.groupby(key).size().rename("actual_trips").reset_index()
    out = exp_g.merge(tech_g, on=key, how="left")
    out["actual_trips"] = out["actual_trips"].fillna(0)
    out["actual_rate"] = out["actual_trips"] / out["exposure"]
    out["predicted_rate"] = out["predicted_trips"] / out["exposure"]
    out["underpowered"] = out["actual_trips"] < 20
    return out


def block_bootstrap_se(exposure_tr: pd.DataFrame, tech_tr: pd.DataFrame, n_boot: int = 200,
                        seed: int = RNG_SEED0) -> float:
    """Game-level block bootstrap SE on the pooled X1 rate (this project's
    standing noise-floor convention, applied OFFLINE since no closed-loop run
    exists this round -- see results doc)."""
    rng = np.random.default_rng(seed)
    games = exposure_tr["game_id"].unique()
    exp_by_game = exposure_tr.groupby("game_id").size()
    tech_by_game = tech_tr.groupby("game_id").size()
    rates = []
    for _ in range(n_boot):
        samp = rng.choice(games, size=len(games), replace=True)
        e = exp_by_game.reindex(samp, fill_value=0).sum()
        c = tech_by_game.reindex(samp, fill_value=0).sum()
        rates.append(c / max(e, 1))
    return float(np.std(rates, ddof=1))


# ---------------------------------------------------------------------------
# X3 who-shoots
# ---------------------------------------------------------------------------
def who_shoots_analysis(seasons: list[int]) -> dict:
    attempts = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
    attempts = attempts[attempts["season"].isin(seasons)]
    design = FT.build_ft_design(attempts, include_technical=True)

    tech = design[design["foul_class"] == "technical"].copy()
    non_tech = design[design["foul_class"] != "technical"].copy()

    # team's attempt-share-weighted average shooter that game (non-technical attempts only)
    team_avg = non_tech.groupby(["season", "team_id", "game_id"])["shooter_ft_raw"].mean().rename(
        "team_avg_shooter_rate").reset_index()
    team_best = non_tech.groupby(["season", "team_id", "game_id"])["shooter_ft_raw"].max().rename(
        "team_best_shooter_rate").reset_index()

    tech_one_row = tech.drop_duplicates(["season", "team_id", "game_id", "trip_id"])
    m = tech_one_row.merge(team_avg, on=["season", "team_id", "game_id"], how="left")
    m = m.merge(team_best, on=["season", "team_id", "game_id"], how="left")
    m = m.dropna(subset=["team_avg_shooter_rate", "team_best_shooter_rate"])

    return {
        "n_technical_trips_with_context": int(len(m)),
        "actual_technical_shooter_mean_asof_rate": float(m["shooter_ft_raw"].mean()),
        "team_avg_shooter_mean_asof_rate": float(m["team_avg_shooter_rate"].mean()),
        "team_best_shooter_mean_asof_rate": float(m["team_best_shooter_rate"].mean()),
        "actual_technical_make_rate_observed": float(tech["made"].mean()),
        "non_technical_make_rate_observed": float(non_tech["made"].mean()),
        "closer_to": ("best" if abs(m["shooter_ft_raw"].mean() - m["team_best_shooter_rate"].mean())
                       < abs(m["shooter_ft_raw"].mean() - m["team_avg_shooter_rate"].mean()) else "avg"),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_fold(fold: str, universe: pd.DataFrame, chances_all: pd.DataFrame,
             tech_all_events: pd.DataFrame) -> dict:
    spec = FOLDS[fold]
    train_seasons, test_seasons = spec["train"], spec["test"]
    assert_not_sealed(train_seasons, context=f"{fold} train")
    assert_not_sealed(test_seasons, context=f"{fold} test")

    exposure_all = build_exposure(chances_all[chances_all["season"].isin(train_seasons + test_seasons)], universe)
    # X2 (section 9.1, pre-game static covariates: conference-game flag, neutral
    # site; "season" cannot vary within a single test season and is dropped as
    # a conditioning axis for that reason, not omitted)
    conf_all = CONF.build_conference_flags(train_seasons + test_seasons)[["game_id", "is_conf_game"]]
    exposure_all = exposure_all.merge(conf_all, on="game_id", how="left")
    exposure_all["is_conf_game"] = exposure_all["is_conf_game"].fillna(False)

    exposure_tr = exposure_all[exposure_all["season"].isin(train_seasons)]
    exposure_te = exposure_all[exposure_all["season"].isin(test_seasons)]
    tech_tr = tech_all_events[tech_all_events["season"].isin(train_seasons)]
    tech_te = tech_all_events[tech_all_events["season"].isin(test_seasons)]
    tech_tr = tech_tr.merge(conf_all, on="game_id", how="left")
    tech_tr["is_conf_game"] = tech_tr["is_conf_game"].fillna(False)
    tech_te = tech_te.merge(conf_all, on="game_id", how="left")
    tech_te["is_conf_game"] = tech_te["is_conf_game"].fillna(False)

    x1_rate = fit_x1(exposure_tr, tech_tr)
    grid = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0,
            3200.0, 6400.0, 12800.0, 51200.0, 1e7]
    x4_cells, x4_meta = fit_x4(exposure_tr, tech_tr, train_seasons, x1_rate, grid)
    x5_k, x5_meta = fit_x5(exposure_tr, tech_tr, train_seasons, x1_rate, grid)
    x5_team = team_asof_rate(exposure_tr, tech_tr, x5_k, x1_rate, min(test_seasons))
    x2_keys = ["is_conf_game", "site"]
    x2_k, x2_meta = grid_search_shrinkage(exposure_tr, tech_tr, x2_keys, train_seasons, grid, x1_rate)
    x2_cells = cell_tables(exposure_tr, tech_tr, x2_keys)
    x2_cells["rate"] = (x2_cells["count"] + x2_k * x1_rate) / (x2_cells["exposure"] + x2_k)
    x2_cells["underpowered"] = x2_cells["count"] < 20
    x2_meta["k"] = x2_k

    def predict_x0(exp_df):
        return np.zeros(len(exp_df))

    def predict_x1(exp_df):
        return np.full(len(exp_df), x1_rate)

    def predict_x2(exp_df):
        m = exp_df.merge(x2_cells[["is_conf_game", "site", "rate"]], on=["is_conf_game", "site"], how="left")
        return m["rate"].fillna(x1_rate).to_numpy()

    def predict_x4(exp_df):
        m = exp_df.merge(x4_cells[["game_phase", "margin_bucket", "site", "rate"]],
                          on=["game_phase", "margin_bucket", "site"], how="left")
        return m["rate"].fillna(x1_rate).to_numpy()

    def predict_x5(exp_df):
        m = exp_df.merge(x5_team[["team_id", "rate"]], on="team_id", how="left")
        return m["rate"].fillna(x1_rate).to_numpy()

    arms = {"X0": predict_x0, "X1": predict_x1, "X2": predict_x2, "X4": predict_x4, "X5": predict_x5}
    overall = {name: score_predicted_vs_actual(exposure_te, tech_te, fn, name) for name, fn in arms.items()}

    segments = {}
    for name, fn in arms.items():
        segments[name] = {
            "game_phase": segment_table(exposure_te, tech_te, fn, "game_phase").to_dict("records"),
            "margin_bucket": segment_table(exposure_te, tech_te, fn, "margin_bucket").to_dict("records"),
            "site": segment_table(exposure_te, tech_te, fn, "site").to_dict("records"),
        }

    # month cut needs game_date; is_conf_game is already merged onto
    # exposure_te / tech_te above (X2's own conditioning feature)
    u_small = universe[["game_id", "season", "game_date"]].copy()
    u_small["month"] = pd.to_datetime(u_small["game_date"]).dt.month

    exposure_te_ctx = exposure_te.merge(u_small, on=["game_id", "season"], how="left")
    tech_te_ctx = tech_te.merge(u_small, on=["game_id", "season"], how="left")
    for name, fn in arms.items():
        segments[name]["month"] = segment_table(exposure_te_ctx, tech_te_ctx, fn, "month").to_dict("records")
        segments[name]["is_conf_game"] = segment_table(exposure_te_ctx, tech_te_ctx, fn, "is_conf_game").to_dict("records")

    # team-level prior-season quintile responsiveness + power calc (X1's flat
    # rate vs actual, and X5's own slope)
    prior_szn = min(test_seasons) - 1
    exp_prior = build_exposure(chances_all[chances_all["season"] == prior_szn], universe)
    tech_prior = tech_all_events[tech_all_events["season"] == prior_szn]
    prior_team = exp_prior.groupby("team_id").size().rename("exposure").reset_index().merge(
        tech_prior.groupby("offender_team_id").size().rename("count").reset_index().rename(
            columns={"offender_team_id": "team_id"}), on="team_id", how="left")
    prior_team["count"] = prior_team["count"].fillna(0)
    prior_team["prior_rate"] = prior_team["count"] / prior_team["exposure"].replace(0, np.nan)
    prior_team = prior_team.dropna(subset=["prior_rate"])
    prior_team = prior_team[prior_team["exposure"] > 500]  # need enough games to trust the prior rate
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
    # each arm's OWN predicted rate for the team, applied to the team's test
    # exposure -- the responsiveness check the standing rule requires (does the
    # arm's prediction slope with the team's own prior quintile, not just the
    # realised actual?)
    # responsiveness check restricted to arms whose prediction is well-defined
    # from team_id alone (X0/X1 constants; X5 is intrinsically team-keyed).
    # X2/X4 condition on in-game state, not team identity, so a per-team
    # predicted rate is not the natural check for them and is not attempted.
    team_arms = {k: v for k, v in arms.items() if k in ("X0", "X1", "X5")}
    exp_te_team = exposure_te[["team_id"]].drop_duplicates().reset_index(drop=True)
    for aname, afn in team_arms.items():
        exp_te_team[f"pred_rate_{aname}"] = afn(exp_te_team)
    qtab = qtab.merge(exp_te_team, on="team_id", how="left")

    qsum = qtab.groupby("quintile").agg(
        n_teams=("team_id", "size"), prior_rate_mean=("prior_rate", "mean"),
        exposure_test=("exposure_test", "sum"), count_test=("count_test", "sum")).reset_index()
    for aname in team_arms:
        pred_trips = (qtab["pred_rate_" + aname] * qtab["exposure_test"]).groupby(qtab["quintile"]).sum()
        qsum[f"pred_rate_{aname}"] = (pred_trips / qsum.set_index("quintile")["exposure_test"]).to_numpy()
    qsum["actual_test_rate"] = qsum["count_test"] / qsum["exposure_test"].replace(0, np.nan)
    league_rate_test = tech_te["season"].size and (len(tech_te) / len(exposure_te))
    qsum["expected_trips_at_league_rate"] = qsum["exposure_test"] * league_rate_test
    qsum["poisson_se_trips"] = np.sqrt(qsum["expected_trips_at_league_rate"].clip(lower=0))
    qsum["poisson_se_rate"] = qsum["poisson_se_trips"] / qsum["exposure_test"].replace(0, np.nan)
    qsum["underpowered"] = qsum["count_test"] < 30

    noise_floor_x1 = block_bootstrap_se(exposure_tr, tech_tr, n_boot=200, seed=RNG_SEED0)
    noise_floor_x1_seed1 = block_bootstrap_se(exposure_tr, tech_tr, n_boot=200, seed=RNG_SEED1)

    return {
        "fold": fold, "train_seasons": train_seasons, "test_seasons": test_seasons,
        "x1_rate": x1_rate, "x2_meta": x2_meta, "x4_meta": x4_meta, "x5_meta": x5_meta,
        "overall": overall, "segments": segments,
        "quintile_responsiveness": qsum.to_dict("records"),
        "noise_floor_seed0": noise_floor_x1, "noise_floor_seed1": noise_floor_x1_seed1,
        "noise_floor_used": max(noise_floor_x1, noise_floor_x1_seed1),
    }


def main():
    t0 = time.time()
    universe = pd.read_parquet(UNIVERSE_PATH)
    log("universe loaded", t0)

    chances_all = load_chances(ALL_SEASONS)
    log(f"chances loaded: {len(chances_all):,} rows", t0)

    tech_all_events = load_technical_events(ALL_SEASONS)
    log(f"technical events loaded: {len(tech_all_events):,} rows", t0)

    results = {}
    for fold in ["F1", "F2"]:
        log(f"running fold {fold}", t0)
        results[fold] = run_fold(fold, universe, chances_all, tech_all_events)
        log(f"fold {fold} done", t0)

    log("who-shoots analysis", t0)
    results["who_shoots"] = who_shoots_analysis(ALL_SEASONS)

    out_path = OUT_DIR / "round1_offline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"written {out_path}", t0)

    # ---- headline print ----
    f2 = results["F2"]
    print("\n=== F2 (SELECTION) overall ===")
    for name, row in f2["overall"].items():
        print(name, row)
    print("\nnoise floor (F2 train, game-block bootstrap SE on pooled rate):",
          f2["noise_floor_used"])
    print("\nwho-shoots:", results["who_shoots"])


if __name__ == "__main__":
    main()
