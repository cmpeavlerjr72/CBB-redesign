"""
own_ratings.py -- our own as-of-date team ratings from hoopR team_box.

WHY. CLAUDE.md data rules: "KenPom snapshots from last year stay on disk and are
one bake-off arm; our own ratings from hoopR box data are the other and the
compliant fallback." This module is that other arm. It is also the anchor arm
`A_own` of the control_engine bake-off (`docs/models/control_engine/model.md`
section 3) and is built once and reused by the L1 anchor bake-off later.

WHAT. For every (season, as_of_date) on which D-I games occur, three ridge
regressions on team dummies, fitted on the games of that season *strictly
before* as_of_date:

  offense/defense   y = 100 * team_score / game_poss   (points per 100 poss)
                    one row per team-game (both sides of every game)
                    X = [intercept, is_home, is_away] + off dummy(team)
                                                      + def dummy(opponent)
                    neutral-site is the reference level of the site term.

  tempo             y = game_poss  (FGA - OREB + TOV + 0.44 FTA, averaged over
                    both teams -- the definition in model.md section 2 and in
                    scripts/build_gate_reference.py)
                    one row per game
                    X = [intercept, neutral] + tempo dummy(home) + tempo dummy(away)

LEAK SAFETY. The as_of_date window is *strictly before* the date, so a game's
own result can never enter its own features. Because CBB teams play on
irregular schedules there is no "week" to align to; the window is the calendar
date, matching `cbb_sim.data.kenpom.as_of`'s strictly-before semantics and
`cbb_sim.analysis.leak_test`'s consecutive-game delta panel.

RIDGE TOWARD A PRIOR, NOT TOWARD ZERO. The penalty is

    lambda * || b - b_prior ||^2

with

  * team effects   b_prior = w * (previous season's FINAL own rating for that
                   team), i.e. last season's rating shrunk toward the league
                   mean (0 on the centred scale) by a fitted weight w. A team
                   with no previous-season rating (new to D-I) gets 0, which
                   IS the league mean, not a fabricated value.
  * site/intercept b_prior = previous season's FINAL fitted value. For the
                   earliest season available there is no previous season, so
                   the intercept prior is the training window's own mean of y
                   (a no-op pull that keeps the level out of the team dummies
                   when the window is thin) and the site priors are 0.

lambda and w are FITTED, not chosen by hand: `select_hyperparameters()` scores
a grid by one-step-ahead walk-forward prediction error on the fold-1 TRAINING
seasons only (2022 and 2023). Nothing later than 2023 is touched by selection,
so the same fitted values are honest for both fold 1 (test 2024) and fold 2
(test 2025). The chosen values are written to the ratings manifest and reported
in `docs/models/control_engine/experiments.md`.

OUTPUT (one row per season x as_of_date x team_id):
    season, as_of_date, team_id, off_c, def_c, tempo_rel, n_games,
    league_off_mean, league_tempo_mean, home_off_eff, away_off_eff,
    neutral_tempo_eff

`off_c` and `def_c` are points per 100 possessions relative to that as-of
league mean (positive off_c = better offense; positive def_c = *worse* defense,
i.e. allows more, the same sign convention as KenPom's AdjD). `tempo_rel` is
the team's own tempo divided by the as-of league mean tempo (KenPom AdjT
convention), so the expected possessions of a game between i and j is
league_tempo_mean * (tempo_rel_i + tempo_rel_j - 1) + neutral_tempo_eff * neutral.
Expected offensive efficiency of i against j is
league_off_mean + off_c_i + def_c_j + (home_off_eff | away_off_eff | 0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

DEFAULT_HOOPR_DIR = Path("data/raw/hoopr")
DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_OUT_DIR = Path("data/processed/ratings")

# Seasons whose *training* windows may be used to pick hyperparameters. Fold 1
# trains on {2022, 2023}; selection may see nothing later (fold 1's own test
# season 2024 included).
SELECTION_SEASONS: tuple[int, ...] = (2022, 2023)

# Grids are coarse and logarithmic on purpose: the criterion surface is flat
# and a fine grid would be fitting noise. Both are recorded with the winner.
LAMBDA_GRID_EFF: tuple[float, ...] = (0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0)
LAMBDA_GRID_TEMPO: tuple[float, ...] = (0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
PRIOR_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_universe(path: Path | str = DEFAULT_UNIVERSE) -> pd.DataFrame:
    u = pd.read_parquet(path)
    u["game_date"] = pd.to_datetime(u["game_date"])
    return u


def load_team_games(
    universe: pd.DataFrame,
    seasons: list[int],
    hoopr_dir: Path | str = DEFAULT_HOOPR_DIR,
) -> pd.DataFrame:
    """One row per team-game for D-I, non-truncated, completed games in
    `seasons`, with the box aggregates the rating models need.

    Games missing either side's box row are dropped (they cannot produce a
    possession estimate) and the count is reported by the caller.
    """
    hoopr_dir = Path(hoopr_dir)
    keep = universe[
        universe["is_d1_game"]
        & ~universe["pbp_truncated"]
        & universe["season"].isin(seasons)
        & universe["home_score"].notna()
        & universe["away_score"].notna()
    ][
        [
            "game_id", "season", "game_date", "tipoff_utc", "home_team_id", "away_team_id",
            "neutral_site", "home_score", "away_score", "n_periods", "cbbd_game_id",
        ]
    ].copy()

    frames = []
    for season in sorted(keep["season"].unique()):
        path = hoopr_dir / "team_box" / f"team_box_{int(season)}.parquet"
        cols = [
            "game_id", "team_id", "team_home_away", "team_score",
            "field_goals_made", "field_goals_attempted",
            "three_point_field_goals_made", "three_point_field_goals_attempted",
            "free_throws_made", "free_throws_attempted",
            "offensive_rebounds", "defensive_rebounds", "total_turnovers",
        ]
        tb = pd.read_parquet(path, columns=cols)
        tb["game_id"] = pd.to_numeric(tb["game_id"], errors="coerce").astype("int64")
        tb["team_id"] = pd.to_numeric(tb["team_id"], errors="coerce").astype("int64")
        frames.append(tb[tb["game_id"].isin(set(keep.loc[keep["season"] == season, "game_id"]))])
    tb = pd.concat(frames, ignore_index=True)
    tb = tb.rename(
        columns={
            "field_goals_made": "fgm", "field_goals_attempted": "fga",
            "three_point_field_goals_made": "tpm", "three_point_field_goals_attempted": "tpa",
            "free_throws_made": "ftm", "free_throws_attempted": "fta",
            "offensive_rebounds": "oreb", "defensive_rebounds": "dreb",
            "total_turnovers": "tov",
        }
    )
    for c in ["fgm", "fga", "tpm", "tpa", "ftm", "fta", "oreb", "dreb", "tov", "team_score"]:
        tb[c] = pd.to_numeric(tb[c], errors="coerce")

    tb["poss_team"] = tb["fga"] - tb["oreb"] + tb["tov"] + 0.44 * tb["fta"]

    # Only games with exactly two usable box rows.
    ok = tb.dropna(subset=["poss_team"]).groupby("game_id").size()
    good_games = set(ok[ok == 2].index)
    tb = tb[tb["game_id"].isin(good_games)]

    game_poss = tb.groupby("game_id")["poss_team"].mean().rename("game_poss")

    tg = tb.merge(keep, on="game_id", how="inner")
    tg = tg.merge(game_poss, on="game_id", how="left")
    tg["is_home"] = (tg["team_home_away"] == "home").astype(float)
    tg["neutral"] = tg["neutral_site"].astype(float)
    tg.loc[tg["neutral"] == 1.0, "is_home"] = np.nan  # site term is home/away/neutral
    tg["site_home"] = ((tg["team_home_away"] == "home") & (~tg["neutral_site"])).astype(float)
    tg["site_away"] = ((tg["team_home_away"] == "away") & (~tg["neutral_site"])).astype(float)
    tg = tg.drop(columns=["is_home"])

    opp = tg[["game_id", "team_id", "team_score"]].rename(
        columns={"team_id": "opp_team_id", "team_score": "opp_score"}
    )
    tg = tg.merge(opp, on="game_id")
    tg = tg[tg["team_id"] != tg["opp_team_id"]].copy()

    tg["off_eff"] = 100.0 * tg["team_score"] / tg["game_poss"].replace(0, np.nan)
    tg["margin"] = tg["team_score"] - tg["opp_score"]
    tg = tg.dropna(subset=["off_eff", "game_poss"])
    tg = tg.sort_values(["season", "game_date", "game_id", "team_id"], kind="mergesort").reset_index(drop=True)
    return tg


# ---------------------------------------------------------------------------
# Ridge-toward-a-prior machinery
# ---------------------------------------------------------------------------
@dataclass
class RidgeProblem:
    """Incremental normal-equation accumulator for a design matrix whose rows
    have only a handful of non-zero entries (an intercept, one or two fixed
    effects, and one or two team dummies). Rows are appended in date order so
    the walk-forward refits are cumulative sums, never re-scans."""

    n_params: int
    xtx: np.ndarray = field(init=False)
    xty: np.ndarray = field(init=False)
    yty: float = field(init=False, default=0.0)
    sy: float = field(init=False, default=0.0)
    n: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.xtx = np.zeros((self.n_params, self.n_params), dtype=np.float64)
        self.xty = np.zeros(self.n_params, dtype=np.float64)

    def add_rows(self, idx: np.ndarray, val: np.ndarray, y: np.ndarray) -> None:
        """`idx`/`val` are (n_rows, k) arrays of column indices and values;
        an index of -1 marks an unused slot."""
        n_rows, k = idx.shape
        for a in range(k):
            ia, va = idx[:, a], val[:, a]
            ok_a = ia >= 0
            for b in range(k):
                ib, vb = idx[:, b], val[:, b]
                m = ok_a & (ib >= 0)
                if m.any():
                    np.add.at(self.xtx, (ia[m], ib[m]), va[m] * vb[m])
            if ok_a.any():
                np.add.at(self.xty, ia[ok_a], va[ok_a] * y[ok_a])
        self.yty += float(np.dot(y, y))
        self.sy += float(y.sum())
        self.n += n_rows

    def solve(self, lam: float, prior: np.ndarray, rhs_extra: np.ndarray | None = None) -> np.ndarray:
        """(X'X + lam*I) b = X'y + lam*prior. `rhs_extra`, when given, is an
        extra (n_params, m) block of right-hand sides solved with the same
        factorisation (used for the affine-in-w hyperparameter sweep)."""
        a = self.xtx + lam * np.eye(self.n_params)
        rhs = self.xty + lam * prior
        b = np.column_stack([rhs]) if rhs_extra is None else np.column_stack([rhs, rhs_extra])
        c, low = linalg.cho_factor(a, lower=True, check_finite=False)
        sol = linalg.cho_solve((c, low), b, check_finite=False)
        return sol


@dataclass
class SeasonRatingRun:
    """Walk-forward ratings for one season, one (lambda, w) setting."""

    season: int
    dates: list[pd.Timestamp]
    team_ids: np.ndarray
    # coefficient blocks, one row per as_of_date
    off: np.ndarray
    dfn: np.ndarray
    tempo: np.ndarray
    league_off: np.ndarray
    league_tempo: np.ndarray
    home_off: np.ndarray
    away_off: np.ndarray
    neutral_tempo: np.ndarray
    n_games_team: np.ndarray
    n_games_window: np.ndarray
    # one-step-ahead predictions on this season's own team-games
    pred_off_eff: np.ndarray
    pred_poss: np.ndarray
    final: dict = field(default_factory=dict)
    eff_rows: pd.DataFrame | None = None
    game_rows: pd.DataFrame | None = None


# eff design: [0]=intercept, [1]=site_home, [2]=site_away, [3:3+T]=off, [3+T:]=def
EFF_FIXED = 3
# tempo design: [0]=intercept, [1]=neutral, [2:2+T]=tempo
TEMPO_FIXED = 2


def _season_frames(tg: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    s = tg[tg["season"] == season]
    teams = np.sort(np.union1d(s["team_id"].unique(), s["opp_team_id"].unique()))
    games = (
        s.drop_duplicates("game_id")[["game_id", "game_date", "neutral_site", "game_poss", "home_team_id", "away_team_id"]]
        .sort_values(["game_date", "game_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return s, games, teams


def fit_season(
    tg: pd.DataFrame,
    season: int,
    lam_eff: float,
    lam_tempo: float,
    prior_eff: dict | None,
    prior_tempo: dict | None,
    w_eff: float,
    w_tempo: float,
    do_eff: bool = True,
    do_tempo: bool = True,
) -> SeasonRatingRun:
    """Walk-forward ridge ratings for one season.

    `prior_eff` / `prior_tempo` are the previous season's FINAL fit, as dicts
    {"team_off": Series, "team_def": Series, "intercept": float,
     "home": float, "away": float} / {"team": Series, "intercept": float,
     "neutral": float}, or None when no previous season is available.
    """
    s, games, teams = _season_frames(tg, season)
    t_index = {int(t): i for i, t in enumerate(teams)}
    n_t = len(teams)

    eff = RidgeProblem(EFF_FIXED + 2 * n_t)
    tmp = RidgeProblem(TEMPO_FIXED + n_t)

    # ---- priors ----------------------------------------------------------
    p_eff = np.zeros(eff.n_params)
    p_tmp = np.zeros(tmp.n_params)
    if prior_eff is not None:
        off_prev = prior_eff["team_off"].reindex(teams).fillna(0.0).to_numpy(dtype=float)
        def_prev = prior_eff["team_def"].reindex(teams).fillna(0.0).to_numpy(dtype=float)
        p_eff[EFF_FIXED:EFF_FIXED + n_t] = w_eff * off_prev
        p_eff[EFF_FIXED + n_t:] = w_eff * def_prev
        p_eff[0] = prior_eff["intercept"]
        p_eff[1] = prior_eff["home"]
        p_eff[2] = prior_eff["away"]
    if prior_tempo is not None:
        tmp_prev = prior_tempo["team"].reindex(teams).fillna(0.0).to_numpy(dtype=float)
        p_tmp[TEMPO_FIXED:] = w_tempo * tmp_prev
        p_tmp[0] = prior_tempo["intercept"]
        p_tmp[1] = prior_tempo["neutral"]

    # ---- pre-built row index/value blocks --------------------------------
    s = s.sort_values(["game_date", "game_id", "team_id"], kind="mergesort")
    off_col = s["team_id"].map(t_index).to_numpy() + EFF_FIXED
    def_col = s["opp_team_id"].map(t_index).to_numpy() + EFF_FIXED + n_t
    eff_idx = np.column_stack([
        np.zeros(len(s), dtype=int),
        np.where(s["site_home"].to_numpy() > 0, 1, -1),
        np.where(s["site_away"].to_numpy() > 0, 2, -1),
        off_col,
        def_col,
    ])
    eff_val = np.ones_like(eff_idx, dtype=float)
    eff_y = s["off_eff"].to_numpy(dtype=float)
    eff_dates = s["game_date"].to_numpy()

    home_col = games["home_team_id"].map(t_index).to_numpy() + TEMPO_FIXED
    away_col = games["away_team_id"].map(t_index).to_numpy() + TEMPO_FIXED
    tmp_idx = np.column_stack([
        np.zeros(len(games), dtype=int),
        np.where(games["neutral_site"].to_numpy(), 1, -1),
        home_col,
        away_col,
    ])
    tmp_val = np.ones_like(tmp_idx, dtype=float)
    tmp_y = games["game_poss"].to_numpy(dtype=float)
    tmp_dates = games["game_date"].to_numpy()

    dates = np.unique(games["game_date"].to_numpy())

    n_dates = len(dates)
    off_out = np.zeros((n_dates, n_t))
    def_out = np.zeros((n_dates, n_t))
    tmp_out = np.zeros((n_dates, n_t))
    lo_out = np.full(n_dates, np.nan)
    lt_out = np.full(n_dates, np.nan)
    ho_out = np.zeros(n_dates)
    ao_out = np.zeros(n_dates)
    nt_out = np.zeros(n_dates)
    ng_out = np.zeros((n_dates, n_t), dtype=np.int32)
    nw_out = np.zeros(n_dates, dtype=np.int32)

    pred_off = np.full(len(s), np.nan)
    pred_poss = np.full(len(games), np.nan)

    team_counts = np.zeros(n_t, dtype=np.int32)
    eff_ptr = 0
    tmp_ptr = 0

    for di, d in enumerate(dates):
        # -- fit on everything strictly before d (already accumulated) -----
        prior_eff_vec = p_eff.copy()
        prior_tmp_vec = p_tmp.copy()
        if prior_eff is None and eff.n > 0:
            prior_eff_vec[0] = eff.sy / eff.n
        if prior_tempo is None and tmp.n > 0:
            prior_tmp_vec[0] = tmp.sy / tmp.n

        e_end = eff_ptr
        while e_end < len(s) and eff_dates[e_end] == d:
            e_end += 1
        t_end = tmp_ptr
        while t_end < len(games) and tmp_dates[t_end] == d:
            t_end += 1

        if do_eff:
            b_eff = eff.solve(lam_eff, prior_eff_vec)[:, 0]
            off_raw = b_eff[EFF_FIXED:EFF_FIXED + n_t]
            def_raw = b_eff[EFF_FIXED + n_t:]
            # Re-centre so every rating is expressed relative to the as-of
            # league mean (CLAUDE.md modelling rule 1). The level moves into
            # the intercept, which is what we report as the league mean, so
            # predictions are unchanged by the re-parameterisation.
            off_m, def_m = off_raw.mean(), def_raw.mean()
            off_c = off_raw - off_m
            def_c = def_raw - def_m
            # With an empty training window and no previous season there is no
            # league level to report: NaN, never a fabricated 0.
            has_level = (eff.n > 0) or (prior_eff is not None)
            league_off = (b_eff[0] + off_m + def_m) if has_level else np.nan
            off_out[di] = off_c
            def_out[di] = def_c
            lo_out[di] = league_off
            ho_out[di] = b_eff[1]
            ao_out[di] = b_eff[2]
            if e_end > eff_ptr:
                sl = slice(eff_ptr, e_end)
                oi = off_col[sl] - EFF_FIXED
                dj = def_col[sl] - EFF_FIXED - n_t
                pred_off[sl] = (
                    league_off + off_c[oi] + def_c[dj]
                    + b_eff[1] * s["site_home"].to_numpy()[sl]
                    + b_eff[2] * s["site_away"].to_numpy()[sl]
                )

        if do_tempo:
            b_tmp = tmp.solve(lam_tempo, prior_tmp_vec)[:, 0]
            tempo_raw = b_tmp[TEMPO_FIXED:]
            tempo_m = tempo_raw.mean()
            has_level_t = (tmp.n > 0) or (prior_tempo is not None)
            league_tempo = (b_tmp[0] + 2.0 * tempo_m) if has_level_t else np.nan
            tempo_team = tempo_raw - tempo_m  # additive deviation
            tmp_out[di] = tempo_team
            lt_out[di] = league_tempo
            nt_out[di] = b_tmp[1]
            if t_end > tmp_ptr:
                sl = slice(tmp_ptr, t_end)
                hi = home_col[sl] - TEMPO_FIXED
                ai = away_col[sl] - TEMPO_FIXED
                pred_poss[sl] = (
                    league_tempo + tempo_team[hi] + tempo_team[ai]
                    + b_tmp[1] * games["neutral_site"].to_numpy()[sl].astype(float)
                )

        ng_out[di] = team_counts
        nw_out[di] = tmp.n

        # -- now fold d's games into the accumulators ----------------------
        if e_end > eff_ptr:
            eff.add_rows(eff_idx[eff_ptr:e_end], eff_val[eff_ptr:e_end], eff_y[eff_ptr:e_end])
            np.add.at(team_counts, off_col[eff_ptr:e_end] - EFF_FIXED, 1)
            eff_ptr = e_end
        if t_end > tmp_ptr:
            tmp.add_rows(tmp_idx[tmp_ptr:t_end], tmp_val[tmp_ptr:t_end], tmp_y[tmp_ptr:t_end])
            tmp_ptr = t_end

    # ---- final (post-season) fit, the prior for the next season ----------
    prior_eff_vec = p_eff.copy()
    prior_tmp_vec = p_tmp.copy()
    if prior_eff is None and eff.n > 0:
        prior_eff_vec[0] = eff.sy / eff.n
    if prior_tempo is None and tmp.n > 0:
        prior_tmp_vec[0] = tmp.sy / tmp.n
    final: dict = {}
    if do_eff:
        b_eff = eff.solve(lam_eff, prior_eff_vec)[:, 0]
        off_raw = b_eff[EFF_FIXED:EFF_FIXED + n_t]
        def_raw = b_eff[EFF_FIXED + n_t:]
        off_m, def_m = off_raw.mean(), def_raw.mean()
        final["eff"] = {
            "team_off": pd.Series(off_raw - off_m, index=teams),
            "team_def": pd.Series(def_raw - def_m, index=teams),
            "intercept": float(b_eff[0] + off_m + def_m),
            "home": float(b_eff[1]),
            "away": float(b_eff[2]),
        }
    if do_tempo:
        b_tmp = tmp.solve(lam_tempo, prior_tmp_vec)[:, 0]
        tempo_raw = b_tmp[TEMPO_FIXED:]
        tempo_m = tempo_raw.mean()
        final["tempo"] = {
            "team": pd.Series(tempo_raw - tempo_m, index=teams),
            "intercept": float(b_tmp[0] + 2.0 * tempo_m),
            "neutral": float(b_tmp[1]),
        }

    run = SeasonRatingRun(
        season=season,
        dates=[pd.Timestamp(d) for d in dates],
        team_ids=teams,
        off=off_out, dfn=def_out, tempo=tmp_out,
        league_off=lo_out, league_tempo=lt_out,
        home_off=ho_out, away_off=ao_out, neutral_tempo=nt_out,
        n_games_team=ng_out, n_games_window=nw_out,
        pred_off_eff=pred_off, pred_poss=pred_poss,
        final=final, eff_rows=s, game_rows=games,
    )
    return run


# ---------------------------------------------------------------------------
# Hyperparameter selection (fold-1 training seasons only)
# ---------------------------------------------------------------------------
def _sse(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(p) & np.isfinite(y)
    return float(((y[m] - p[m]) ** 2).sum()), int(m.sum())


def _sweep_one_model(
    tg: pd.DataFrame,
    seasons: tuple[int, ...],
    lambda_grid: tuple[float, ...],
    prior_weight_grid: tuple[float, ...],
    which: str,
) -> pd.DataFrame:
    """Walk-forward one-step-ahead RMSE for one of the two rating models over
    a (lambda, prior weight) grid.

    The efficiency and tempo models are algebraically independent (disjoint
    designs, disjoint targets), so they are swept separately rather than as a
    product grid. The FIRST selection season has no previous season, so its
    fit -- and therefore its contribution to the criterion -- does not depend
    on the prior weight at all; it is computed once per lambda and reused.
    """
    do_eff = which == "eff"
    rows = []
    for lam in lambda_grid:
        run = fit_season(tg, seasons[0], lam, lam, None, None, 0.0, 0.0,
                         do_eff=do_eff, do_tempo=not do_eff)
        if do_eff:
            base_sse, base_n = _sse(run.eff_rows["off_eff"].to_numpy(dtype=float), run.pred_off_eff)
            prior0 = run.final["eff"]
        else:
            base_sse, base_n = _sse(run.game_rows["game_poss"].to_numpy(dtype=float), run.pred_poss)
            prior0 = run.final["tempo"]
        for w in prior_weight_grid:
            sse_tot, n_tot = base_sse, base_n
            prior_e = prior0 if do_eff else None
            prior_t = None if do_eff else prior0
            for season in seasons[1:]:
                run = fit_season(tg, season, lam, lam, prior_e, prior_t, w, w,
                                 do_eff=do_eff, do_tempo=not do_eff)
                if do_eff:
                    sse, n = _sse(run.eff_rows["off_eff"].to_numpy(dtype=float), run.pred_off_eff)
                    prior_e = run.final["eff"]
                else:
                    sse, n = _sse(run.game_rows["game_poss"].to_numpy(dtype=float), run.pred_poss)
                    prior_t = run.final["tempo"]
                sse_tot += sse
                n_tot += n
            rows.append({"model": which, "lambda": lam, "prior_weight": w,
                         "rmse": float(np.sqrt(sse_tot / n_tot)), "n": int(n_tot)})
    return pd.DataFrame(rows)


def select_hyperparameters(
    tg: pd.DataFrame,
    seasons: tuple[int, ...] = SELECTION_SEASONS,
    lambda_grid_eff: tuple[float, ...] = LAMBDA_GRID_EFF,
    lambda_grid_tempo: tuple[float, ...] = LAMBDA_GRID_TEMPO,
    prior_weight_grid: tuple[float, ...] = PRIOR_WEIGHT_GRID,
) -> dict:
    """Pick (lambda, prior weight) for the efficiency and tempo models by
    one-step-ahead walk-forward RMSE on `seasons` only.

    The criterion is the error of predicting each team-game's *actual*
    offensive efficiency (resp. each game's actual possessions) from the
    ratings that were available strictly before that game -- exactly how the
    ratings are used downstream. Nothing after `seasons` is read, so the
    chosen values are honest for fold 1 (test 2024) and fold 2 (test 2025).
    """
    seasons = tuple(sorted(seasons))
    eff_df = _sweep_one_model(tg, seasons, lambda_grid_eff, prior_weight_grid, "eff")
    tempo_df = _sweep_one_model(tg, seasons, lambda_grid_tempo, prior_weight_grid, "tempo")
    best_eff = eff_df.loc[eff_df["rmse"].idxmin()]
    best_tempo = tempo_df.loc[tempo_df["rmse"].idxmin()]
    return {
        "lambda_eff": float(best_eff["lambda"]),
        "prior_weight_eff": float(best_eff["prior_weight"]),
        "rmse_eff": float(best_eff["rmse"]),
        "lambda_tempo": float(best_tempo["lambda"]),
        "prior_weight_tempo": float(best_tempo["prior_weight"]),
        "rmse_tempo": float(best_tempo["rmse"]),
        "selection_seasons": list(seasons),
        "grid_eff": eff_df.to_dict("records"),
        "grid_tempo": tempo_df.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Build + persist
# ---------------------------------------------------------------------------
def run_to_frame(run: SeasonRatingRun) -> pd.DataFrame:
    n_dates, n_t = run.off.shape
    dates = np.repeat(np.array(run.dates, dtype="datetime64[ns]"), n_t)
    teams = np.tile(run.team_ids, n_dates)
    league_tempo = np.repeat(run.league_tempo, n_t)
    tempo_dev = run.tempo.reshape(-1)
    out = pd.DataFrame({
        "season": np.int64(run.season),
        "as_of_date": dates,
        "team_id": teams.astype("int64"),
        "off_c": run.off.reshape(-1),
        "def_c": run.dfn.reshape(-1),
        "tempo_rel": 1.0 + tempo_dev / league_tempo,
        "n_games": run.n_games_team.reshape(-1).astype("int32"),
        "league_off_mean": np.repeat(run.league_off, n_t),
        "league_tempo_mean": league_tempo,
        "home_off_eff": np.repeat(run.home_off, n_t),
        "away_off_eff": np.repeat(run.away_off, n_t),
        "neutral_tempo_eff": np.repeat(run.neutral_tempo, n_t),
        "n_games_window": np.repeat(run.n_games_window, n_t).astype("int32"),
    })
    return out


def build_all_seasons(
    tg: pd.DataFrame,
    seasons: list[int],
    lam_eff: float,
    lam_tempo: float,
    w_eff: float,
    w_tempo: float,
) -> tuple[dict[int, pd.DataFrame], dict[int, dict]]:
    """Walk-forward ratings for every season in `seasons`, chaining each
    season's final fit forward as the next season's prior."""
    frames: dict[int, pd.DataFrame] = {}
    finals: dict[int, dict] = {}
    prior_e = prior_t = None
    for season in sorted(seasons):
        run = fit_season(tg, season, lam_eff, lam_tempo, prior_e, prior_t, w_eff, w_tempo)
        frames[season] = run_to_frame(run)
        finals[season] = run.final  # type: ignore[attr-defined]
        prior_e = run.final["eff"]  # type: ignore[attr-defined]
        prior_t = run.final["tempo"]  # type: ignore[attr-defined]
    return frames, finals


def write_ratings(frames: dict[int, pd.DataFrame], out_dir: Path | str = DEFAULT_OUT_DIR) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for season, df in frames.items():
        p = out_dir / f"own_ratings_{season}.parquet"
        df.to_parquet(p, index=False)
        paths.append(p)
    return paths


def write_manifest(manifest: dict, out_dir: Path | str = DEFAULT_OUT_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "own_ratings_manifest.json"
    p.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# As-of lookup for the feature builder
# ---------------------------------------------------------------------------
def load_ratings(seasons: list[int], out_dir: Path | str = DEFAULT_OUT_DIR) -> pd.DataFrame:
    frames = []
    for season in seasons:
        p = Path(out_dir) / f"own_ratings_{season}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing own ratings: {p}")
        frames.append(pd.read_parquet(p))
    out = pd.concat(frames, ignore_index=True)
    out["as_of_date"] = pd.to_datetime(out["as_of_date"])
    return out


RATING_COLS: tuple[str, ...] = (
    "off_c", "def_c", "tempo_rel", "n_games",
    "league_off_mean", "league_tempo_mean",
    "home_off_eff", "away_off_eff", "neutral_tempo_eff",
)


def join_as_of(
    games: pd.DataFrame,
    ratings: pd.DataFrame,
    team_col: str,
    date_col: str = "game_date",
    season_col: str = "season",
    suffix: str = "",
    cols: tuple[str, ...] = RATING_COLS,
) -> pd.DataFrame:
    """Attach the ratings row for (season, team, as_of_date == the game's own
    date) to every row of `games`.

    The as-of table is BUILT with a strictly-before window, so joining on the
    game's own date is the correct, leak-free lookup: `as_of_date == D` means
    "fitted on games strictly before D". Teams with no row on that date (a
    date on which no D-I game of theirs is scheduled cannot happen here, since
    every date in the table carries every team) fall back to NaN and are never
    fabricated.
    """
    g = games.copy()
    g["_key_team"] = g[team_col].astype("int64")
    g["_key_date"] = pd.to_datetime(g[date_col]).astype("datetime64[ns]")
    r = ratings[["season", "as_of_date", "team_id", *cols]].rename(
        columns={"team_id": "_key_team", "as_of_date": "_key_date"}
    )
    merged = g.merge(r, on=[season_col, "_key_team", "_key_date"], how="left")
    merged = merged.rename(columns={c: f"{c}{suffix}" for c in cols})
    return merged.drop(columns=["_key_team", "_key_date"])
