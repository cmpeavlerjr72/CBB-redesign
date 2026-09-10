"""
clock.py -- L5 CLOCK CONSUMPTION: the number of seconds a possession takes.

Pre-registrations: `docs/models/clock/experiments.md` section 1 (round 1) and
section 5 (round 2), both PM-authored 2026-09-10 and both written before the
code they govern. Feature provenance: `docs/models/clock/features.md`.
Trainers: `scripts/train_clock_v1.py` (round 1) and `scripts/train_clock_v2.py`
(round 2, never overwriting round 1's script or artifacts). Diagnosis of the
round-1 failure: `scripts/diag_clock_v1.py`. Model doc and verdict:
`docs/models/clock/model.md`.

ROUND 2 in one line: round 1 measured that every arm smoothed the sharp bend in
conditional duration inside the last 45 seconds of a period, so round 2 replaces
the clock's state representation -- ten fine buckets crossed with the period
type and the offence's score state, plus a last-shot and a two-for-one window --
and adds a two-regime arm. Everything else is unchanged, which is what makes the
two rounds comparable row by row.

WHAT THIS MODEL IS. Pace is EMERGENT in the possession engine
(`ARCHITECTURE_DECISIONS.md` Decision 7, `docs/LEARNINGS.md` L14): the engine
never draws possessions per game. At each possession start it draws the
possession's DURATION in seconds from this model, conditioned on state, and
then draws the terminal event (L3) given the duration bucket and state. The
possession count of a game is whatever falls out of 2 x 1200 seconds of that
process plus the overtime model. G1 therefore depends on this model being
right BY STATE, not merely on average.

TARGET. `duration_s` on `data/processed/possessions/possessions_{season}.parquet`
-- the number of seconds between the previous possession's terminal event and
this one's. Integer, supported on 0..`DURATION_CAP`.

--------------------------------------------------------------------------
CENSORING
--------------------------------------------------------------------------
A possession whose terminal event is `end_period` did not end because the
offence gave the ball up; it ended because the horn sounded. Its `duration_s`
is the number of seconds that were left, and the true duration is strictly
LARGER. Those rows are RIGHT-CENSORED, and every arm here states how it
handles them:

  empirical      Kaplan-Meier inside the state cell (discrete-time product
                 limit): a censored row stays in the risk set up to its
                 observed time and leaves it afterwards.
  lognormal      censored likelihood contribution log S(c + 0.5) in the MLE.
  gamma          same.
  lgbm_quantile  EXCLUDED from training. LightGBM's quantile objective has no
                 censored form, and scoring a censored row as if complete is a
                 known-wrong imputation. The exclusion is 0.2-0.3% of rows and
                 removes short observations, so the fitted quantiles are
                 biased slightly LONG; the trainer reports the count.
  hazard         native: a row censored at c contributes the person-periods
                 t = 0..c-1 with no event, and nothing at t = c.

--------------------------------------------------------------------------
LEAK SAFETY -- what may NOT be a feature
--------------------------------------------------------------------------
The engine calls this model at the MOMENT THE POSSESSION STARTS, before the
terminal event exists. Anything computed from the possession's own outcome is
therefore banned, and `BANNED_FEATURES` names them so
`tests/test_clock.py` can assert it:

  terminal_event, end_clock, duration_s, n_chances, oreb_count, points, fga_*,
  fgm_*, fta, ftm, and_one, stolen, ft_trip_ambiguous, is_transition.

`is_transition` deserves its own sentence: it is defined as
`duration_s <= 8 AND start_reason in {DREB, TOV}` -- it is a FUNCTION OF THE
TARGET. Using it would be a perfect leak. The change ledger already carries it
as a CONFIRMED-DEFECT for L3 for the same reason; here it would be worse, so
it is banned outright rather than merely flagged.

Every team feature is as-of and strictly-before by construction: the own-ridge
ratings and the tempo prior come through `cbb_sim.ratings.own_ratings.join_as_of`
and `cbb_sim.models.pace.build_pace_table`, whose as-of windows are built
strictly before the game's own date (`docs/models/pace/features.md`).

--------------------------------------------------------------------------
SEAL
--------------------------------------------------------------------------
`cbb_sim.data.seal.assert_not_sealed` guards every fold slice. Season 2026
possession tables exist (they are data, not a fit) and cannot enter a fold.

--------------------------------------------------------------------------
REUSED, NOT REIMPLEMENTED
--------------------------------------------------------------------------
    cbb_sim.models.pace          build_pace_table (the as-of tempo prior
                                 feature loader) and fit_multiplicative /
                                 MultiplicativeModel (the L2 winner, used here
                                 as a pregame tempo prior exactly as Decision 7
                                 says it must be), FOLDS, SEASON_INDEX_ANCHOR
    cbb_sim.ratings.own_ratings  join_as_of for the own ratings
    cbb_sim.control.rng          the counter-based (seed, game_id, family)
                                 stream; this model's family label is "clock"
    cbb_sim.data.seal            assert_not_sealed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from cbb_sim.control import rng as crng
from cbb_sim.data.seal import assert_not_sealed
from cbb_sim.models import pace
from cbb_sim.ratings import own_ratings as orat

DEFAULT_POSS_DIR = Path("data/processed/possessions")
DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_RATINGS_DIR = Path("data/processed/ratings")
DEFAULT_MODEL_DIR = Path("data/processed/models/clock")

#: RNG family label. Every draw this model makes goes through
#: `cbb_sim.control.rng` keyed on (seed, game_id, RNG_FAMILY).
RNG_FAMILY = "clock"

#: Modelled support, in whole seconds. A possession cannot plausibly exceed
#: three shot clocks plus dead-ball time; `duration_s` above this is a CBBD
#: feed gap (a period whose first logged event is its last -- see
#: `docs/tests/possessions_build_2026-09-10.md` section 4, where `end_period`
#: and `unknown` rows carry maxima up to a full 1,200-second period). Those
#: rows are EXCLUDED from training and from every metric, and the trainer
#: reports the count per season. They are not clipped: clipping would invent a
#: duration for a possession whose duration is not in the feed.
DURATION_CAP = 90
GRID: np.ndarray = np.arange(0, DURATION_CAP + 1, dtype=np.int64)
N_GRID = len(GRID)

#: The previous possession's end type, i.e. `start_reason` on the possession
#: row. `period_start` is the reference level of the dummy coding.
PREV_END_LEVELS: tuple[str, ...] = (
    "period_start", "DREB", "TOV", "made_FG", "made_FT", "other",
)
PREV_END_INDEX = {v: i for i, v in enumerate(PREV_END_LEVELS)}

#: Seconds-remaining buckets. These define BOTH the empirical arm's cell grid
#: and the pre-registered PIT reporting cells (previous end type x
#: seconds-remaining bucket). The first edge is the end-of-period bucket the
#: pre-registration's end-of-half check also uses (< 35 s).
SECS_BUCKET_EDGES: tuple[int, ...] = (0, 35, 120, 300, 600, 1201)
SECS_BUCKET_LABELS: tuple[str, ...] = ("0-34", "35-119", "120-299", "300-599", "600+")

#: Score-difference buckets (offence view), used by the empirical arm's C grid.
SCORE_BUCKET_EDGES: tuple[float, ...] = (-np.inf, -15.5, -5.5, 5.5, 15.5, np.inf)
SCORE_BUCKET_LABELS: tuple[str, ...] = ("<=-16", "-15..-6", "-5..5", "6..15", ">=16")

#: A PIT cell with fewer than this many test rows is reported UNDERPOWERED and
#: is excluded from the decision rule (pre-registration).
PIT_MIN_CELL = 300

#: LEAK-sized PIT failure threshold from the pre-registration.
PIT_KS_D_GATE = 0.05

#: Minimum rows in an empirical cell before the hierarchical fallback fires.
EMPIRICAL_MIN_CELL = 300

#: Emergent-G1 tolerances (`docs/SIM_GUARDRAILS.md` G1).
G1_MEAN_TOL = 1.0
G1_SD_TOL = 0.75

#: A month is powered for the by-month G1 read if it carries at least this
#: many games. November-March carry thousands; April carries a handful.
G1_MONTH_MIN_GAMES = 100

FOLDS = pace.FOLDS
SELECTION_FOLD = "F2"
SEASON_INDEX_ANCHOR = pace.SEASON_INDEX_ANCHOR

#: Columns computed from the possession's OWN outcome. They can never be
#: features; `tests/test_clock.py` asserts it against every feature set and
#: against every fitted arm's stored feature list.
BANNED_FEATURES: tuple[str, ...] = (
    "terminal_event", "end_clock", "duration_s", "n_chances", "oreb_count",
    "points", "fga_rim", "fgm_rim", "fga_jump2", "fgm_jump2", "fga_3", "fgm_3",
    "fta", "ftm", "and_one", "stolen", "ft_trip_ambiguous", "is_transition",
    "tech_points_off", "tech_points_def", "censored",
)


# ===========================================================================
# 1. Feature sets
# ===========================================================================
PREV_END_DUMMIES: tuple[str, ...] = tuple(
    f"prev_end_{lv}" for lv in PREV_END_LEVELS if lv != "period_start"
)

#: A_state, exactly as pre-registered: previous end type, period, seconds
#: remaining, chance number.
#:
#: `chance_number_at_start` IS the pre-registered "chance number within
#: possession", and at a POSSESSION's start it is identically 1 -- a possession
#: begins on its first chance by the segmentation rule
#: (`cbb_sim/pbp/possessions.py`). It is kept in the list so the pre-registered
#: bundle is visibly honoured, and every arm drops it as a ZERO-VARIANCE column
#: and records the drop. See `docs/models/clock/model.md` section 9: if a
#: chance-level duration target was intended, that is a different target and
#: needs its own pre-registration.
FEATURES_A: tuple[str, ...] = (
    *PREV_END_DUMMIES, "period", "is_ot", "seconds_remaining",
    "chance_number_at_start",
)
FEATURES_B: tuple[str, ...] = FEATURES_A + (
    "off_tempo_rel", "def_tempo_rel", "tempo_prior_game",
    "off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c",
    "site_home", "site_away",
)
FEATURES_C: tuple[str, ...] = FEATURES_B + (
    "score_diff", "x_score_diff__seconds_remaining", "in_bonus",
)
FEATURES_D: tuple[str, ...] = FEATURES_C + ("season_idx", "days_since_start")

# ---------------------------------------------------------------------------
# ROUND 2 state representation (pre-registration section 5, 2026-09-10)
# ---------------------------------------------------------------------------
# Round 1's diagnosis (experiments.md section 3.1): the conditional mean of
# duration falls from 17.9 s to 3.1 s across the last 90 seconds of a period and
# every arm smoothed the bend, because `seconds_remaining` entered either
# linearly or in a single 0-34 s bucket. Round 2 replaces that with a FINE
# bucket crossed with the period type and the offence's score state, so that
# end-of-first-half and end-of-game are separate states and the late TRAILING
# team's short possessions are representable.
R2_SR_EDGES: tuple[int, ...] = (0, 5, 10, 20, 30, 45, 60, 90, 150, 300, 1201)
R2_SR_LABELS: tuple[str, ...] = (
    "0-4", "5-9", "10-19", "20-29", "30-44", "45-59", "60-89", "90-149",
    "150-299", "300+",
)
#: {first half, second half/OT}, exactly as pre-registered. OT is pooled with
#: the second half; the consequence is recorded in `model.md` section 6.
R2_PERIOD_TYPES: tuple[str, ...] = ("H1", "H2_OT")
R2_SCORE_STATES: tuple[str, ...] = ("trailing", "tied", "leading")

#: One shot clock, and the two-for-one window just outside it.
LAST_SHOT_MAX_S = 30
TWO_FOR_ONE_MAX_S = 45

#: The crossed factor, one-hot with level 0 as the reference so that a design
#: with an intercept stays full rank.
R2_CROSS_LEVELS: tuple[tuple[int, int, int], ...] = tuple(
    (b, p, s)
    for b in range(len(R2_SR_LABELS))
    for p in range(len(R2_PERIOD_TYPES))
    for s in range(len(R2_SCORE_STATES))
)
R2_CROSS_COLUMNS: tuple[str, ...] = tuple(
    f"srx_{R2_SR_LABELS[b]}__{R2_PERIOD_TYPES[p]}__{R2_SCORE_STATES[s]}"
    for (b, p, s) in R2_CROSS_LEVELS
)
#: Level 0 (`srx_0-4__H1__trailing`) is the dropped reference level.
R2_CROSS_DUMMIES: tuple[str, ...] = R2_CROSS_COLUMNS[1:]

R2_WINDOW_FEATURES: tuple[str, ...] = ("last_shot_window", "two_for_one_window")

#: Team + score content carried over from round 1's C_plus_score, which round 1
#: recorded as KEPT; the raw `seconds_remaining` main effect and the
#: `period`/`is_ot` columns are REPLACED by the crossed factor, and
#: `chance_number_at_start` is gone because round 1 measured it as
#: zero-variance.
R2_CARRIED: tuple[str, ...] = (
    *PREV_END_DUMMIES,
    "off_tempo_rel", "def_tempo_rel", "tempo_prior_game",
    "off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c",
    "site_home", "site_away",
    "score_diff", "x_score_diff__seconds_remaining", "in_bonus",
)

#: The dummy/interaction form the parametric and hazard arms take.
FEATURES_R2_DUMMY: tuple[str, ...] = R2_CARRIED + R2_CROSS_DUMMIES + R2_WINDOW_FEATURES

#: The tree form: "the tree arm takes the raw seconds_remaining plus the bucket
#: id". A tree splits on an ordered integer id by itself, so handing it 59
#: one-hot columns would only make the splits harder to find.
FEATURES_R2_TREE: tuple[str, ...] = R2_CARRIED + (
    "seconds_remaining", "sr_bucket_id", "period_type", "score_state",
    *R2_WINDOW_FEATURES,
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "A_state": FEATURES_A,
    "B_plus_teams": FEATURES_B,
    "C_plus_score": FEATURES_C,
    "D_plus_season": FEATURES_D,
    "R2_dummy": FEATURES_R2_DUMMY,
    "R2_tree": FEATURES_R2_TREE,
}
FEATURE_SET_NAMES: tuple[str, ...] = ("A_state", "B_plus_teams", "C_plus_score", "D_plus_season")
R2_FEATURE_SET_NAMES: tuple[str, ...] = ("R2_dummy", "R2_tree")


def feature_set(name: str) -> list[str]:
    return list(FEATURE_SETS[name])


def r2_bucket_id(seconds_remaining: np.ndarray) -> np.ndarray:
    """Fine seconds-remaining bucket id, 0..9."""
    return np.searchsorted(np.asarray(R2_SR_EDGES[1:-1]),
                           np.asarray(seconds_remaining), side="right").astype("int64")


def r2_period_type(period) -> np.ndarray:
    """0 = first half, 1 = second half or overtime."""
    return (np.asarray(period, dtype="float64") >= 2.0).astype("int64")


def r2_score_state(score_diff: np.ndarray) -> np.ndarray:
    """0 = offence trailing, 1 = tied, 2 = offence leading."""
    sd = np.asarray(score_diff, dtype="float64")
    return np.where(sd < 0, 0, np.where(sd == 0, 1, 2)).astype("int64")


def r2_cross_key(seconds_remaining, period, score_diff) -> np.ndarray:
    return np.ravel_multi_index(
        (r2_bucket_id(seconds_remaining),
         r2_period_type(np.broadcast_to(np.asarray(period, dtype="float64"),
                                        np.shape(seconds_remaining))),
         r2_score_state(score_diff)),
        (len(R2_SR_LABELS), len(R2_PERIOD_TYPES), len(R2_SCORE_STATES)))


def add_r2_state(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every round-2 state column in place and return the frame.

    Called once by `build_design` and again by `apply_clock_override` for each
    simulated second, so the sim can never inherit the REAL possession's clock
    bucket -- which would be a leak of the answer into the emergent test."""
    sr = df["seconds_remaining"].to_numpy(dtype="float64")
    sd = df["score_diff"].to_numpy(dtype="float64")
    per = df["period"].to_numpy(dtype="float64")
    df["sr_bucket_id"] = r2_bucket_id(sr).astype("float32")
    df["period_type"] = r2_period_type(per).astype("float32")
    df["score_state"] = r2_score_state(sd).astype("float32")
    df["last_shot_window"] = (sr <= LAST_SHOT_MAX_S).astype("float32")
    df["two_for_one_window"] = ((sr > LAST_SHOT_MAX_S) & (sr <= TWO_FOR_ONE_MAX_S)).astype("float32")
    key = np.ravel_multi_index(
        (r2_bucket_id(sr), r2_period_type(per), r2_score_state(sd)),
        (len(R2_SR_LABELS), len(R2_PERIOD_TYPES), len(R2_SCORE_STATES)))
    for j, name in enumerate(R2_CROSS_COLUMNS):
        if j == 0:
            continue  # reference level
        df[name] = (key == j).astype("int8")
    return df


def apply_clock_override(block: pd.DataFrame, clock: np.ndarray, period: float) -> pd.DataFrame:
    """Put the SIMULATED clock into a block of real possession rows.

    Everything else in the row -- previous end type, score difference, bonus,
    which team has the ball, both teams' priors -- stays as the real possession
    had it; only the clock and everything derived from it is replaced. This is
    the single place that knows which columns are clock-derived, so round 1 and
    round 2 cannot drift apart."""
    sc = np.asarray(clock, dtype="float64")
    block["seconds_remaining"] = sc.astype("float32")
    block["period"] = np.float32(period)
    block["is_ot"] = np.float32(1.0 if period >= 3 else 0.0)
    if "x_score_diff__seconds_remaining" in block.columns:
        block["x_score_diff__seconds_remaining"] = (
            block["score_diff"].to_numpy(dtype="float64") * sc / 1200.0).astype("float32")
    if "sr_bucket_id" in block.columns:
        add_r2_state(block)
    return block


#: Columns whose value changes as the SIM clock runs, and which therefore have
#: to be recomputed inside `chain_halves` rather than taken from the real
#: possession row. Everything else in a feature set is a property of the
#: matchup or of the real previous possession and is carried across unchanged.
CLOCK_DEPENDENT_FEATURES: tuple[str, ...] = (
    "period", "is_ot", "seconds_remaining", "x_score_diff__seconds_remaining",
)

ARMS: tuple[str, ...] = ("empirical", "lognormal", "gamma", "hazard", "lgbm_quantile")

#: Round 2 adds the two-regime composite. The pre-registration says it "counts
#: as tree for the tree-must-beat-linear rule", so it sits at the tree rank and
#: is listed in TREE_ARMS.
R2_ARMS: tuple[str, ...] = (
    "empirical", "lognormal", "gamma", "hazard", "lgbm_quantile", "two_regime",
)

#: Tie-break order from the pre-registration: empirical < parametric < hazard
#: < tree.
SIMPLICITY_RANK: dict[str, int] = {
    "empirical": 0, "lognormal": 1, "gamma": 1, "hazard": 2, "lgbm_quantile": 3,
    "two_regime": 3,
}
TREE_ARMS: tuple[str, ...] = ("lgbm_quantile", "two_regime")


# ===========================================================================
# 2. Design matrix
# ===========================================================================
POSSESSION_COLUMNS: tuple[str, ...] = (
    "game_id", "season", "period", "poss_index", "offense_team_id",
    "defense_team_id", "offense_is_home", "start_clock", "duration_s",
    "start_score_diff", "start_reason", "terminal_event", "off_in_bonus",
    "points", "tech_points_off",
)


def load_possessions(seasons: list[int], poss_dir: Path | str = DEFAULT_POSS_DIR) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = Path(poss_dir) / f"possessions_{int(s)}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing possessions table: {p} (run scripts/build_possessions.py)")
        frames.append(pd.read_parquet(p, columns=list(POSSESSION_COLUMNS)))
    return pd.concat(frames, ignore_index=True)


def cbbd_complete_games(poss: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Per game: does the CBBD event stream contain the whole game?

    `games_universe.pbp_truncated` is a hoopR-side flag and does NOT identify
    CBBD feed incompleteness (`docs/tests/possessions_build_2026-09-10.md`
    section 2: 19% of 2022-2023 games are short of the final score by a whole
    possession's worth of points). The pre-registration's universe is
    "CBBD-complete games", so the flag is built here with the same definition
    the build report uses: the possession table's own points, plus the
    technical free throws it carries separately, must equal the schedule's
    final score for BOTH teams.

    Returns one row per game with `points_complete`, the two accumulated
    scores, and the per-half clock-accounting deviation as a DIAGNOSTIC (it is
    reported, not used as a filter: the signed deviation is -0.75 s per half on
    2025, 0.06% of a half, and filtering on it would remove two thirds of the
    universe for an effect two orders of magnitude smaller than G1)."""
    tp = poss.groupby(["game_id", "offense_team_id"], as_index=False)[["points", "tech_points_off"]].sum()
    tp["scored"] = tp["points"] + tp["tech_points_off"]
    u = universe[["game_id", "home_team_id", "home_score", "away_score"]]
    m = tp.merge(u, on="game_id", how="inner")
    m["final"] = np.where(m["offense_team_id"] == m["home_team_id"], m["home_score"], m["away_score"])
    m["ok"] = m["scored"] == m["final"]
    per_game = m.groupby("game_id", as_index=False)["ok"].all().rename(columns={"ok": "points_complete"})

    reg = poss[poss["period"] <= 2]
    half = reg.groupby(["game_id", "period"], as_index=False)["duration_s"].sum()
    half["dev"] = half["duration_s"] - 1200
    dev = half.groupby("game_id", as_index=False)["dev"].agg(
        clock_dev_max_abs=lambda s: float(np.abs(s).max()),
        clock_dev_signed=lambda s: float(s.sum()),
    )
    return per_game.merge(dev, on="game_id", how="left")


def build_design(
    seasons: list[int],
    poss_dir: Path | str = DEFAULT_POSS_DIR,
    universe_path: Path | str = DEFAULT_UNIVERSE,
    ratings_dir: Path | str = DEFAULT_RATINGS_DIR,
    pace_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """One row per modelled possession, with every candidate feature, the
    target `duration_s`, and the `censored` flag.

    Rows kept: every possession of a D-I, non-truncated, CBBD-complete game
    whose `duration_s` is inside the modelled support 0..`DURATION_CAP`.
    Nothing is imputed and nothing is clipped; the diagnostics dict reports
    every exclusion count.
    """
    seasons = [int(s) for s in seasons]
    universe = pd.read_parquet(universe_path)
    universe = universe[universe["is_d1_game"] & ~universe["pbp_truncated"]].copy()
    universe["game_date"] = pd.to_datetime(universe["game_date"])

    poss = load_possessions(seasons, poss_dir)
    diag: dict = {"seasons": seasons, "raw_rows": int(len(poss)),
                  "raw_games": int(poss["game_id"].nunique())}

    poss = poss[poss["game_id"].isin(set(universe["game_id"]))].copy()
    diag["rows_in_universe"] = int(len(poss))
    diag["games_in_universe"] = int(poss["game_id"].nunique())

    complete = cbbd_complete_games(poss, universe)
    keep_games = set(complete.loc[complete["points_complete"], "game_id"])
    diag["games_cbbd_complete"] = int(len(keep_games))
    diag["games_dropped_cbbd_incomplete"] = int(diag["games_in_universe"] - len(keep_games))
    poss = poss[poss["game_id"].isin(keep_games)].copy()
    diag["rows_cbbd_complete"] = int(len(poss))

    over_cap = int((poss["duration_s"] > DURATION_CAP).sum())
    diag["rows_over_duration_cap"] = over_cap
    diag["rows_over_duration_cap_pct"] = round(100.0 * over_cap / max(len(poss), 1), 5)
    diag["over_cap_by_terminal"] = (
        poss.loc[poss["duration_s"] > DURATION_CAP, "terminal_event"].value_counts().to_dict()
    )
    poss = poss[poss["duration_s"] <= DURATION_CAP].copy()

    # ---- target and censoring -------------------------------------------
    poss["censored"] = (poss["terminal_event"] == "end_period").to_numpy()
    diag["censored_rows"] = int(poss["censored"].sum())
    diag["censored_pct"] = round(100.0 * float(poss["censored"].mean()), 4)

    # ---- calendar / site -------------------------------------------------
    poss = poss.merge(
        universe[["game_id", "game_date", "neutral_site", "home_team_id", "away_team_id"]],
        on="game_id", how="inner")
    poss["month"] = poss["game_date"].dt.month.astype("int16")

    neutral = poss["neutral_site"].to_numpy()
    off_home = poss["offense_is_home"].to_numpy()
    poss["site_home"] = ((~neutral) & off_home).astype("float32")
    poss["site_away"] = ((~neutral) & (~off_home)).astype("float32")

    # ---- state at possession start --------------------------------------
    sr = poss["start_reason"].to_numpy()
    unknown_reason = ~np.isin(sr, PREV_END_LEVELS)
    diag["rows_unknown_start_reason"] = int(unknown_reason.sum())
    if unknown_reason.any():  # pragma: no cover - guard, never seen in 2022-2026
        raise ValueError(
            f"unmapped start_reason values: {sorted(set(sr[unknown_reason]))} -- "
            "PREV_END_LEVELS must map every value the possession builder emits")
    for lv in PREV_END_LEVELS:
        if lv == "period_start":
            continue
        poss[f"prev_end_{lv}"] = (poss["start_reason"] == lv).to_numpy().astype("float32")

    poss["period"] = poss["period"].astype("float32")
    poss["is_ot"] = (poss["period"] >= 3).astype("float32")
    poss["seconds_remaining"] = poss["start_clock"].astype("float32")
    poss["chance_number_at_start"] = np.float32(1.0)
    poss["score_diff"] = poss["start_score_diff"].astype("float32")
    poss["in_bonus"] = poss["off_in_bonus"].astype("float32")
    # Scaled so the interaction sits on the same order of magnitude as the two
    # columns it multiplies; the scale is fixed, never fitted.
    poss["x_score_diff__seconds_remaining"] = (
        poss["score_diff"] * poss["seconds_remaining"] / 1200.0).astype("float32")

    # ---- team features: as-of tempo prior (pace.py) and own ratings ------
    if pace_table is None:
        pace_table = pace.build_pace_table(sorted(set(seasons)))
    pt = pace_table[["game_id", "home_tempo_rel", "away_tempo_rel",
                     "league_tempo_mean_asof", "days_since_start", "season_index"]].copy()
    poss = poss.merge(pt, on="game_id", how="left")

    is_home_off = poss["offense_is_home"].to_numpy()
    poss["off_tempo_rel"] = np.where(is_home_off, poss["home_tempo_rel"], poss["away_tempo_rel"]).astype("float32")
    poss["def_tempo_rel"] = np.where(is_home_off, poss["away_tempo_rel"], poss["home_tempo_rel"]).astype("float32")
    # The L2 winner, verbatim: tempo_A x tempo_B x the as-of league mean
    # (`cbb_sim.models.pace.MultiplicativeModel`). Order does not matter, so the
    # offence/defence split above does not change this product.
    poss["tempo_prior_game"] = (
        poss["home_tempo_rel"] * poss["away_tempo_rel"] * poss["league_tempo_mean_asof"]
    ).astype("float32")

    ratings = orat.load_ratings(sorted(set(seasons)), out_dir=ratings_dir)
    poss = orat.join_as_of(poss, ratings, team_col="offense_team_id", date_col="game_date",
                           suffix="__offteam", cols=("off_c", "def_c"))
    poss = orat.join_as_of(poss, ratings, team_col="defense_team_id", date_col="game_date",
                           suffix="__defteam", cols=("off_c", "def_c"))
    poss = poss.rename(columns={
        "off_c__offteam": "off_rating_off_c", "def_c__offteam": "off_rating_def_c",
        "off_c__defteam": "def_rating_off_c", "def_c__defteam": "def_rating_def_c",
    })

    poss["season_idx"] = (poss["season"] - SEASON_INDEX_ANCHOR).astype("float32")
    poss["days_since_start"] = poss["days_since_start"].astype("float32")

    # A small share of rows fail a team-feature join (a team-season whose
    # ratings row is missing on that date). On a league-CENTRED scale 0.0 IS
    # the league mean, and for the two tempo ratios 1.0 is; neither is a
    # fabricated level. The tempo prior falls back to the as-of league mean.
    fill_zero = ["off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c"]
    diag["rows_missing_ratings"] = int(poss[fill_zero].isna().any(axis=1).sum())
    for c in fill_zero:
        poss[c] = poss[c].astype("float32").fillna(0.0)
    diag["rows_missing_tempo"] = int(poss["off_tempo_rel"].isna().sum())
    for c in ("off_tempo_rel", "def_tempo_rel"):
        poss[c] = poss[c].astype("float32").fillna(1.0)
    lg_mean = float(poss["league_tempo_mean_asof"].median())
    poss["tempo_prior_game"] = poss["tempo_prior_game"].astype("float32").fillna(lg_mean)
    poss["days_since_start"] = poss["days_since_start"].fillna(
        float(poss["days_since_start"].median())).astype("float32")

    # ---- reporting cells --------------------------------------------------
    poss["secs_bucket"] = pd.cut(poss["seconds_remaining"], bins=list(SECS_BUCKET_EDGES),
                                 right=False, labels=list(SECS_BUCKET_LABELS)).astype("str")
    poss["score_bucket"] = pd.cut(poss["score_diff"], bins=list(SCORE_BUCKET_EDGES),
                                  labels=list(SCORE_BUCKET_LABELS)).astype("str")
    poss["prev_end"] = poss["start_reason"].astype("str")
    poss["pit_cell"] = poss["prev_end"] + " | " + poss["secs_bucket"]

    # ---- round-2 state representation (pre-registration section 5) --------
    add_r2_state(poss)

    keep: list[str] = []
    for c in [
        "season", "game_id", "period", "poss_index", "offense_team_id", "defense_team_id",
        "offense_is_home", "game_date", "month", "duration_s", "censored", "terminal_event",
        "start_reason", "prev_end", "secs_bucket", "score_bucket", "pit_cell",
        *sorted({c for fs in FEATURE_SETS.values() for c in fs}),
    ]:
        if c not in keep:
            keep.append(c)
    out = poss[keep].reset_index(drop=True)
    diag["final_rows"] = int(len(out))
    diag["final_games"] = int(out["game_id"].nunique())
    diag["rows_by_season"] = out.groupby("season").size().to_dict()
    return out, diag


def fold_slices(design: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) with the seal guard on both. Season 2026 can never enter."""
    spec = FOLDS[fold]
    assert_not_sealed(spec["train"], context=f"clock {fold} train seasons")
    assert_not_sealed(spec["test"], context=f"clock {fold} test seasons")
    tr = design[design["season"].isin(spec["train"])]
    te = design[design["season"].isin(spec["test"])]
    assert_not_sealed(tr, context=f"clock {fold} train slice")
    assert_not_sealed(te, context=f"clock {fold} test slice")
    return tr, te


# ===========================================================================
# 3. Shared predictive-distribution machinery
# ===========================================================================
#: Every arm exposes `pmf(df) -> (n, N_GRID) float64`, a proper probability
#: mass function over the integer seconds 0..DURATION_CAP whose rows sum to 1.
#: That single interface is what makes four very different model classes
#: comparable on one CRPS definition, and it is also what the sampler and
#: `chain_halves` consume, so the sim draws from EXACTLY the object that was
#: scored -- no second implementation to drift.
PMF_FLOOR = 1e-12


def _normalise(pmf: np.ndarray) -> np.ndarray:
    """Renormalise onto the modelled support.

    Any mass an arm places beyond `DURATION_CAP` is removed and the rest is
    rescaled, i.e. every arm's predictive law is CONDITIONED on
    D <= DURATION_CAP. That is exactly the conditioning the target's own
    exclusion rule applies (rows above the cap are dropped as feed gaps), so
    the scored object and the observed object live on the same support.
    Piling the excess onto the last cell instead would put visible mass on a
    90-second possession, which is precisely the artifact the cap exists to
    keep out."""
    pmf = np.clip(np.asarray(pmf, dtype="float64"), 0.0, None)
    s = pmf.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return pmf / s


def _pmf_from_cdf(cdf_at_points: np.ndarray) -> np.ndarray:
    """Discretise a predictive CDF onto the integer grid by differencing it at
    N_GRID+1 evaluation points. Mass beyond the last point is removed and the
    rest renormalised by `_normalise` -- the same "conditioned on
    D <= DURATION_CAP" convention every arm uses."""
    return _normalise(np.diff(cdf_at_points, axis=1))


#: Evaluation points for a CONTINUOUS predictive law of the underlying
#: duration X. A recorded `duration_s = t` means X spent the whole of second t
#: and the representative point of that interval is t + 0.5 (which is why the
#: parametric arms fit on y + `CONTINUITY`), so P(D = t) = F(t+1) - F(t).
KNOTS: np.ndarray = np.arange(0, DURATION_CAP + 2, dtype="float64")

#: Evaluation points for an arm that models the OBSERVED INTEGER directly (the
#: quantile-regression arm fits LightGBM on `duration_s` itself). The usual
#: continuity correction applies: P(D = t) = F(t + 0.5) - F(t - 0.5).
MIDPOINTS: np.ndarray = np.concatenate([[-0.5], GRID + 0.5]).astype("float64")


def crps(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row CRPS of an integer-supported predictive distribution:

        CRPS = sum_t ( F(t) - 1{y <= t} )^2 ,  t = 0..DURATION_CAP

    the discrete (ranked-probability-score) form, which is a strictly proper
    scoring rule for a distribution on the integers and is defined identically
    for the empirical, parametric, tree and hazard arms."""
    F = np.cumsum(pmf, axis=1)
    ind = (GRID[None, :] >= np.asarray(y)[:, None]).astype("float64")
    return ((F - ind) ** 2).sum(axis=1)


def log_score(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row -log P(D = y). Returns +inf where the arm puts zero mass on the
    observed value; the trainer reports the share of such rows separately and
    averages only over the rows where the score is DEFINED, which is what the
    pre-registration's "log score where defined" asks for."""
    p = pmf[np.arange(len(y)), np.asarray(y, dtype="int64")]
    with np.errstate(divide="ignore"):
        return -np.log(p)


def pit(pmf: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Randomised PIT for a discrete predictive distribution:
    F(y-1) + V (F(y) - F(y-1)) with V ~ U(0,1). Same construction as
    `cbb_sim.models.pace.pit_discrete` and `scripts/grade_control.py::gate_g5`.
    Uniform under a correctly specified model.

    `v` is supplied by the caller (one uniform per row, drawn once for the
    whole test slice) so the PIT does not depend on how the slice happens to
    be chunked, and so paired arms see the SAME randomisation."""
    F = np.cumsum(pmf, axis=1)
    idx = np.arange(len(y))
    yy = np.asarray(y, dtype="int64")
    hi = F[idx, yy]
    lo = hi - pmf[idx, yy]
    return lo + np.asarray(v) * (hi - lo)


def ks_uniform(u: np.ndarray) -> tuple[float, float]:
    """(D, p) of a one-sample K-S test of `u` against Uniform(0,1)."""
    if len(u) == 0:
        return float("nan"), float("nan")
    res = stats.kstest(np.asarray(u, dtype="float64"), "uniform")
    return float(res.statistic), float(res.pvalue)


def pmf_mean_sd(pmf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = pmf @ GRID.astype("float64")
    m2 = pmf @ (GRID.astype("float64") ** 2)
    return m, np.sqrt(np.maximum(m2 - m ** 2, 0.0))


def sample_from_pmf(pmf: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF draw: the smallest t with F(t) >= u. Deterministic given
    `u`, which is what makes the whole sim reproducible."""
    F = np.cumsum(pmf, axis=1)
    F[:, -1] = 1.0
    idx = (np.asarray(u)[:, None] > F).sum(axis=1)
    return GRID[np.minimum(idx, N_GRID - 1)]


def sample_durations(arm, df: pd.DataFrame, seed: int, index: int,
                     game_ids: np.ndarray | None = None) -> np.ndarray:
    """Draw one duration per row of `df` from `arm`, on the engine-wide
    (seed, game_id, "clock") counter-based stream (`cbb_sim.control.rng`).

    Nothing about a game's draw depends on which other games are in the run,
    so paired arms and paired seeds difference game by game -- the property
    CLAUDE.md's RNG rule exists to give."""
    gids = df["game_id"].to_numpy() if game_ids is None else np.asarray(game_ids)
    keys = crng.stream_keys(seed, gids, RNG_FAMILY)
    u = crng.uniforms(keys, index)
    return sample_from_pmf(arm.pmf(df), u)


def _matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.ascontiguousarray(df[features].to_numpy(dtype="float64"))


def _drop_zero_variance(train: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Split a pre-registered feature list into the columns that carry
    information on the training slice and the ZERO-VARIANCE ones. The dropped
    names are stored on the fitted arm and printed by the trainer: a
    pre-registered feature is never silently discarded."""
    keep, dropped = [], []
    for f in features:
        v = train[f].to_numpy(dtype="float64")
        (keep if np.nanstd(v) > 1e-12 else dropped).append(f)
    return keep, dropped


# ===========================================================================
# 4. Arm 1 -- empirical resampling from state cells (Kaplan-Meier per cell)
# ===========================================================================
#: The empirical arm realises each pre-registered feature set as a NESTED cell
#: grid, because a cell model has no other way to consume a continuous
#: feature. Dimensions are ordered most-informative first; the hierarchical
#: fallback drops them from the RIGHT until a cell has at least
#: `EMPIRICAL_MIN_CELL` training rows, ending at the single global cell. The
#: pre-registration's named grid (previous end type x seconds-remaining bucket
#: x tempo-prior tercile) is exactly `B_plus_teams` minus the site dimension.
EMPIRICAL_DIMS: dict[str, tuple[str, ...]] = {
    "A_state": ("prev_end_code", "secs_bucket_code", "period_group"),
    "B_plus_teams": ("prev_end_code", "secs_bucket_code", "period_group",
                     "tempo_tercile", "site_code"),
    "C_plus_score": ("prev_end_code", "secs_bucket_code", "period_group",
                     "tempo_tercile", "site_code", "score_bucket_code", "bonus_code"),
    "D_plus_season": ("prev_end_code", "secs_bucket_code", "period_group",
                      "tempo_tercile", "site_code", "score_bucket_code", "bonus_code",
                      "season_code"),
}
#: Round 2's grid, exactly as pre-registered: "prev end type x fine bucket x
#: period type x trailing/leading x tempo tercile". Dimensions are listed in
#: that stated order and the hierarchical fallback drops them from the RIGHT,
#: so the fine bucket -- the thing round 2 exists to add -- survives longer than
#: any of the coarser dimensions. 6 x 10 x 2 x 3 x 3 = 1,080 cells.
EMPIRICAL_DIMS["R2_dummy"] = (
    "prev_end_code", "r2_bucket_code", "r2_period_type", "r2_score_state", "tempo_tercile",
)
EMPIRICAL_DIMS["R2_tree"] = EMPIRICAL_DIMS["R2_dummy"]

EMPIRICAL_DIM_SIZES: dict[str, int] = {
    "prev_end_code": len(PREV_END_LEVELS),
    "secs_bucket_code": len(SECS_BUCKET_LABELS),
    "period_group": 2,
    "tempo_tercile": 3,
    "site_code": 3,
    "score_bucket_code": len(SCORE_BUCKET_LABELS),
    "bonus_code": 2,
    "season_code": 8,
    "r2_bucket_code": len(R2_SR_LABELS),
    "r2_period_type": len(R2_PERIOD_TYPES),
    "r2_score_state": len(R2_SCORE_STATES),
}


def kaplan_meier_pmf(cell: np.ndarray, y: np.ndarray, censored: np.ndarray,
                     n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time Kaplan-Meier pmf on the integer grid, per cell.

    d(t) = uncensored exits at t, c(t) = censored departures at t (a censored
    row survived t and leaves the risk set AFTER it), n(t) = rows still at risk
    at t. h(t) = d(t)/n(t), S(t) = prod_{s<=t} (1 - h(s)), P(D = t) = S(t-1) -
    S(t). The survival that never resolves inside the grid -- the censored
    rows that outlast every observed exit in their cell -- is dropped and the
    rest renormalised by `_normalise`, the same "conditioned on
    D <= DURATION_CAP" convention every other arm uses.

    This is the honest way for a resampling arm to use a right-censored
    observation: it contributes to the denominator up to its own time and is
    never counted as an exit."""
    d = np.zeros((n_cells, N_GRID), dtype="float64")
    c = np.zeros((n_cells, N_GRID), dtype="float64")
    np.add.at(d, (cell[~censored], y[~censored]), 1.0)
    np.add.at(c, (cell[censored], y[censored]), 1.0)
    n_total = (d + c).sum(axis=1, keepdims=True)
    left_before = np.concatenate([np.zeros((n_cells, 1)), np.cumsum(d + c, axis=1)[:, :-1]], axis=1)
    at_risk = n_total - left_before
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(at_risk > 0, d / np.maximum(at_risk, 1e-12), 0.0)
    surv = np.cumprod(1.0 - h, axis=1)
    prev = np.concatenate([np.ones((n_cells, 1)), surv[:, :-1]], axis=1)
    pmf = prev - surv
    counts = n_total.ravel()
    return pmf, counts


@dataclass
class EmpiricalArm:
    """Resampling from state cells, with Kaplan-Meier inside each cell and a
    documented hierarchical fallback.

    CENSORING: Kaplan-Meier (see `kaplan_meier_pmf`).
    FALLBACK: a row whose deepest cell holds fewer than `EMPIRICAL_MIN_CELL`
    TRAINING rows drops one dimension from the right and retries, down to the
    single global cell. The trainer reports the share of test rows served at
    each level."""

    feature_set_name: str
    dims: tuple[str, ...]
    sizes: tuple[int, ...]
    level_pmfs: list[np.ndarray]
    level_counts: list[np.ndarray]
    tempo_edges: tuple[float, float]
    season_map: dict[int, int]
    min_cell: int = EMPIRICAL_MIN_CELL
    features: list[str] = field(default_factory=list)
    dropped_zero_variance: list[str] = field(default_factory=list)
    name: str = "empirical"

    # -- cell coding -------------------------------------------------------
    def _codes(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        prev = df["prev_end"].map(PREV_END_INDEX).to_numpy()
        secs = np.searchsorted(np.asarray(SECS_BUCKET_EDGES[1:-1]),
                               df["seconds_remaining"].to_numpy(), side="right")
        score = np.searchsorted(np.asarray(SCORE_BUCKET_EDGES[1:-1]),
                                df["score_diff"].to_numpy(), side="right")
        site = np.where(df["site_home"].to_numpy() > 0, 0,
                        np.where(df["site_away"].to_numpy() > 0, 1, 2))
        tempo = np.searchsorted(np.asarray(self.tempo_edges),
                                df["tempo_prior_game"].to_numpy(), side="right")
        # The test season is not in the training seasons, so its own cell can
        # never exist. It is mapped to the MOST RECENT TRAINING season -- the
        # same honest realisation of "by season" that
        # `possession_outcome.BaselineArm` uses, and the reason the trainer
        # reports D_plus_season and C_plus_score as near-identical here.
        last = max(self.season_map.values()) if self.season_map else 0
        season = df["season"].map(self.season_map).fillna(last).to_numpy().astype("int64")
        return {
            "prev_end_code": prev.astype("int64"),
            "secs_bucket_code": secs.astype("int64"),
            "period_group": (df["is_ot"].to_numpy() > 0).astype("int64"),
            "tempo_tercile": tempo.astype("int64"),
            "site_code": site.astype("int64"),
            "score_bucket_code": score.astype("int64"),
            "bonus_code": (df["in_bonus"].to_numpy() > 0).astype("int64"),
            "season_code": season,
            # round 2: recomputed from the frame's own clock, so the emergent
            # chain's overridden clock is what selects the cell
            "r2_bucket_code": r2_bucket_id(df["seconds_remaining"].to_numpy()),
            "r2_period_type": r2_period_type(df["period"].to_numpy()),
            "r2_score_state": r2_score_state(df["score_diff"].to_numpy()),
        }

    def _keys(self, codes: dict[str, np.ndarray], level: int) -> np.ndarray:
        if level == 0:
            return np.zeros(len(next(iter(codes.values()))), dtype="int64")
        dims = self.dims[:level]
        sizes = self.sizes[:level]
        return np.ravel_multi_index([codes[d] for d in dims], sizes)

    def level_of(self, df: pd.DataFrame) -> np.ndarray:
        """The fallback level each row is served at (len(dims) == full grid)."""
        codes = self._codes(df)
        n = len(df)
        out = np.full(n, -1, dtype="int64")
        for lv in range(len(self.dims), -1, -1):
            todo = out < 0
            if not todo.any():
                break
            keys = self._keys({k: v[todo] for k, v in codes.items()}, lv)
            ok = self.level_counts[lv][keys] >= self.min_cell
            idx = np.flatnonzero(todo)[ok]
            out[idx] = lv
        return out

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        codes = self._codes(df)
        n = len(df)
        out = np.zeros((n, N_GRID), dtype="float64")
        done = np.zeros(n, dtype=bool)
        for lv in range(len(self.dims), -1, -1):
            todo = ~done
            if not todo.any():
                break
            keys = self._keys({k: v[todo] for k, v in codes.items()}, lv)
            ok = self.level_counts[lv][keys] >= self.min_cell
            idx = np.flatnonzero(todo)[ok]
            if len(idx):
                out[idx] = self.level_pmfs[lv][keys[ok]]
                done[idx] = True
        if not done.all():  # the level-0 global cell always qualifies
            out[~done] = self.level_pmfs[0][0]
        return _normalise(out)


def fit_empirical(train: pd.DataFrame, feature_set_name: str,
                  min_cell: int = EMPIRICAL_MIN_CELL, seed: int = 0) -> EmpiricalArm:
    dims = EMPIRICAL_DIMS[feature_set_name]
    sizes = tuple(EMPIRICAL_DIM_SIZES[d] for d in dims)
    tempo = train["tempo_prior_game"].to_numpy(dtype="float64")
    tempo_edges = (float(np.quantile(tempo, 1 / 3)), float(np.quantile(tempo, 2 / 3)))
    seasons = sorted(int(s) for s in train["season"].unique())
    season_map = {s: i for i, s in enumerate(seasons)}

    arm = EmpiricalArm(
        feature_set_name=feature_set_name, dims=dims, sizes=sizes,
        level_pmfs=[], level_counts=[], tempo_edges=tempo_edges,
        season_map=season_map, min_cell=min_cell,
        features=feature_set(feature_set_name),
    )
    codes = arm._codes(train)
    y = train["duration_s"].to_numpy(dtype="int64")
    cen = train["censored"].to_numpy(dtype=bool)
    for lv in range(len(dims) + 1):
        n_cells = 1 if lv == 0 else int(np.prod(sizes[:lv]))
        keys = arm._keys(codes, lv)
        pmf, counts = kaplan_meier_pmf(keys, y, cen, n_cells)
        arm.level_pmfs.append(_normalise(pmf))
        arm.level_counts.append(counts)
    # The global cell is the terminal fallback and must always qualify.
    arm.level_counts[0] = np.maximum(arm.level_counts[0], min_cell)
    return arm


# ===========================================================================
# 5. Arms 2-3 -- heteroscedastic parametric regression, censored MLE
# ===========================================================================
#: A continuity offset: the target is an integer count of whole seconds, and a
#: continuous density has to be evaluated somewhere inside the second. The
#: midpoint is the standard choice and keeps y = 0 (2% of possessions) inside
#: the support of a log-scale family.
CONTINUITY = 0.5


@dataclass
class ParametricArm:
    """log(mu) = X b, log(dispersion) = X g -- heteroscedastic by construction,
    which is what the pre-registration asks for ("heteroscedastic").

    Family `lognormal`: log D ~ Normal(m = X b, s = exp(X g)).
    Family `gamma`:     D ~ Gamma(mean = exp(X b), shape = exp(X g)).

    CENSORING: an `end_period` row at c contributes log S(c + CONTINUITY) --
    "the possession would have lasted longer than the c seconds that were
    left" -- to the same likelihood, so censored rows inform the fit without
    being scored as if they were complete. The uncensored block's gradient is
    analytic; the censored block's is a central finite difference over the same
    objective, which costs 2(2p+2) evaluations of a block that is 0.2-0.3% of
    the rows and avoids an approximation to d/dk of the incomplete gamma."""

    family: str
    feature_set_name: str
    features: list[str]
    dropped_zero_variance: list[str]
    mu_scale: np.ndarray
    sd_scale: np.ndarray
    beta: np.ndarray
    gamma_: np.ndarray
    n_obs: int
    n_censored: int
    converged: bool
    nll: float

    @property
    def name(self) -> str:
        return self.family

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        x = _matrix(df, self.features)
        z = (x - self.mu_scale) / self.sd_scale
        return np.concatenate([np.ones((len(z), 1)), z], axis=1)

    def _params(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z @ self.beta, z @ self.gamma_

    def cdf_at(self, df: pd.DataFrame, points: np.ndarray) -> np.ndarray:
        z = self._design(df)
        a, b = self._params(z)
        pts = np.maximum(np.asarray(points, dtype="float64"), 1e-9)
        if self.family == "lognormal":
            s = np.exp(np.clip(b, -8.0, 8.0))
            return stats.norm.cdf((np.log(pts)[None, :] - a[:, None]) / s[:, None])
        mu = np.exp(np.clip(a, -8.0, 8.0))
        k = np.exp(np.clip(b, -8.0, 8.0))
        return special.gammainc(k[:, None], k[:, None] * pts[None, :] / mu[:, None])

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        cdf = self.cdf_at(df, KNOTS)
        cdf[:, 0] = 0.0
        return _pmf_from_cdf(cdf)


def _lognormal_terms(a, b, w):
    s = np.exp(np.clip(b, -8.0, 8.0))
    lw = np.log(w)
    z = (lw - a) / s
    ll = -b - 0.5 * z * z - lw - 0.5 * np.log(2 * np.pi)
    return ll, z / s, z * z - 1.0


def _gamma_terms(a, b, w):
    mu = np.exp(np.clip(a, -8.0, 8.0))
    k = np.exp(np.clip(b, -8.0, 8.0))
    ll = k * (np.log(k) - a) + (k - 1.0) * np.log(w) - k * w / mu - special.gammaln(k)
    d_a = k * (w / mu - 1.0)
    d_b = k * (np.log(k) + 1.0 - a + np.log(w) - w / mu - special.digamma(k))
    return ll, d_a, d_b


def _censored_ll(family: str, a, b, c) -> np.ndarray:
    """log S(c): the possession consumed all c seconds that were left and
    would have gone on, so the only thing the data says is X > c. The knot is
    the integer c, matching `KNOTS`; c = 0 contributes log 1 = 0, which is
    correct -- a possession with no time left carries no information."""
    w = np.maximum(np.asarray(c, dtype="float64"), 1e-9)
    if family == "lognormal":
        s = np.exp(np.clip(b, -8.0, 8.0))
        return stats.norm.logsf((np.log(w) - a) / s)
    mu = np.exp(np.clip(a, -8.0, 8.0))
    k = np.exp(np.clip(b, -8.0, 8.0))
    return np.log(np.maximum(special.gammaincc(k, k * w / mu), 1e-300))


def fit_parametric(train: pd.DataFrame, feature_set_name: str, family: str,
                   seed: int = 0, maxiter: int = 200) -> ParametricArm:
    feats = feature_set(feature_set_name)
    keep, dropped = _drop_zero_variance(train, feats)
    x = _matrix(train, keep)
    mu_scale = x.mean(axis=0)
    sd_scale = x.std(axis=0)
    sd_scale[sd_scale < 1e-12] = 1.0
    z = np.concatenate([np.ones((len(x), 1)), (x - mu_scale) / sd_scale], axis=1)
    del x  # the round-2 dummy design is ~1.2 GB per copy; do not hold two
    p = z.shape[1]

    y = train["duration_s"].to_numpy(dtype="float64")
    cen = train["censored"].to_numpy(dtype=bool)
    zu, wu = z[~cen], y[~cen] + CONTINUITY
    zc, cc = z[cen], y[cen]
    n = len(y)
    terms = _lognormal_terms if family == "lognormal" else _gamma_terms

    lw = np.log(wu)
    beta0 = np.zeros(p)
    gamma0 = np.zeros(p)
    if family == "lognormal":
        beta0[0] = lw.mean()
        gamma0[0] = np.log(max(lw.std(), 1e-3))
    else:
        beta0[0] = np.log(max(wu.mean(), 1e-3))
        gamma0[0] = np.log(max((wu.mean() / max(wu.std(), 1e-6)) ** 2, 0.5))
    theta0 = np.concatenate([beta0, gamma0])

    def censored_block(theta: np.ndarray) -> float:
        if not len(zc):
            return 0.0
        a = zc @ theta[:p]
        b = zc @ theta[p:]
        return float(_censored_ll(family, a, b, cc).sum())

    def obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a = zu @ theta[:p]
        b = zu @ theta[p:]
        ll, d_a, d_b = terms(a, b, wu)
        total = float(ll.sum())
        grad = np.concatenate([zu.T @ d_a, zu.T @ d_b])
        if len(zc):
            total += censored_block(theta)
            eps = 1e-5
            gc = np.empty(2 * p)
            for j in range(2 * p):
                step = np.zeros(2 * p)
                step[j] = eps
                gc[j] = (censored_block(theta + step) - censored_block(theta - step)) / (2 * eps)
            grad = grad + gc
        return -total / n, -grad / n

    res = optimize.minimize(obj, theta0, jac=True, method="L-BFGS-B",
                            options={"maxiter": maxiter, "maxcor": 20})
    theta = res.x
    return ParametricArm(
        family=family, feature_set_name=feature_set_name, features=keep,
        dropped_zero_variance=dropped, mu_scale=mu_scale, sd_scale=sd_scale,
        beta=theta[:p], gamma_=theta[p:], n_obs=n, n_censored=int(cen.sum()),
        converged=bool(res.success), nll=float(res.fun),
    )


# ===========================================================================
# 6. Arm 4 -- LightGBM quantile regression, 9 quantiles
# ===========================================================================
QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: Fixed across every feature set and fold, so the grid compares feature sets
#: and model classes rather than tuning effort (the same rule
#: `possession_outcome.LgbmArm` states).
LGBM_PARAMS = dict(
    objective="quantile", n_estimators=300, learning_rate=0.07, num_leaves=63,
    min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
    reg_lambda=1.0, verbose=-1,
)


@dataclass
class QuantileTreeArm:
    """Nine LightGBM quantile regressions, turned into a predictive
    distribution by inverse-CDF with LINEAR INTERPOLATION between the fitted
    quantiles, exactly as pre-registered.

    The interpolated quantile function is inverted into a CDF on the integer
    grid so that the CRPS, the log score and the sampler all see one object.
    Below q(0.1) and above q(0.9) the CDF is extended with the slope of the
    first and last fitted segment; predicted quantiles that CROSS are sorted
    per row (a monotonicity repair, not a tuning knob).

    CENSORING: censored rows are dropped from training -- stated, counted, and
    its direction of bias recorded in `n_censored_dropped`."""

    feature_set_name: str
    features: list[str]
    dropped_zero_variance: list[str]
    boosters: list
    levels: tuple[float, ...]
    n_censored_dropped: int
    seed: int
    name: str = "lgbm_quantile"
    params: dict = field(default_factory=dict)

    def quantiles(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features]
        q = np.column_stack([b.predict(x) for b in self.boosters])
        return np.sort(np.clip(q, 0.0, float(DURATION_CAP)), axis=1)

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        q = self.quantiles(df)
        tau = np.asarray(self.levels, dtype="float64")
        first = np.maximum(q[:, 1] - q[:, 0], 1e-6)
        last = np.maximum(q[:, -1] - q[:, -2], 1e-6)
        q_lo = q[:, 0] - first * (tau[0] / (tau[1] - tau[0]))
        q_hi = q[:, -1] + last * ((1.0 - tau[-1]) / (tau[-1] - tau[-2]))
        Q = np.column_stack([q_lo, q, q_hi])
        T = np.concatenate([[0.0], tau, [1.0]])
        Q = np.maximum.accumulate(Q, axis=1)

        n = len(q)
        cdf = np.empty((n, len(MIDPOINTS)), dtype="float64")
        for j, e in enumerate(MIDPOINTS):
            k = np.clip((e >= Q).sum(axis=1) - 1, 0, Q.shape[1] - 2)
            rows = np.arange(n)
            lo, hi = Q[rows, k], Q[rows, k + 1]
            width = hi - lo
            frac = np.where(width > 1e-9, (e - lo) / np.maximum(width, 1e-9), 1.0)
            cdf[:, j] = T[k] + np.clip(frac, 0.0, 1.0) * (T[k + 1] - T[k])
        cdf = np.clip(np.maximum.accumulate(cdf, axis=1), 0.0, 1.0)
        cdf[:, 0] = 0.0
        return _pmf_from_cdf(cdf)


def fit_quantile_tree(train: pd.DataFrame, feature_set_name: str, seed: int = 0,
                      levels: tuple[float, ...] = QUANTILE_LEVELS,
                      n_jobs: int = 16, params: dict | None = None) -> QuantileTreeArm:
    """`params` overrides `LGBM_PARAMS` entries. It is None everywhere in round
    1 (fixed hyperparameters across the grid) and carries the F1-selected
    tree-parameter pair in round 2, where the pre-registration allows a search
    ON F1 ONLY."""
    import lightgbm as lgb

    feats = feature_set(feature_set_name)
    keep, dropped = _drop_zero_variance(train, feats)
    cen = train["censored"].to_numpy(dtype=bool)
    d = train.loc[~cen]
    x = d[keep]
    y = d["duration_s"].to_numpy(dtype="float64")
    use = dict(LGBM_PARAMS)
    use.update(params or {})
    boosters = []
    for tau in levels:
        m = lgb.LGBMRegressor(alpha=tau, random_state=seed, n_jobs=n_jobs, **use)
        m.fit(x, y)
        boosters.append(m)
    return QuantileTreeArm(
        feature_set_name=feature_set_name, features=keep, dropped_zero_variance=dropped,
        boosters=boosters, levels=tuple(levels), n_censored_dropped=int(cen.sum()), seed=seed,
        params=use,
    )


# ===========================================================================
# 7. Arm 5 -- discrete-time hazard (per-second exit probability)
# ===========================================================================
#: The baseline hazard is a free coefficient per elapsed second up to
#: `HAZARD_BASELINE_BINS - 1`, with everything beyond pooled into the last bin.
#: A fully flexible baseline is the point of a discrete-time hazard: the
#: shot-clock shape (a hump around 20-30 s) is not a parametric curve.
HAZARD_BASELINE_BINS = 61

#: A small ridge on every coefficient except the intercept, fixed across
#: feature sets and folds. It exists to keep the sparsely-populated late
#: baseline bins finite, not to tune fit quality.
HAZARD_L2 = 1e-4

#: Elapsed-time terms that are functions of the SIM clock rather than of the
#: possession's start state, and are therefore recomputed at every second.
#: `seconds_remaining_now = start_clock - elapsed` enters ONLY through these
#: two indicators: its linear part is an exact linear combination of the static
#: `seconds_remaining` feature and the baseline's elapsed-time coefficients, so
#: adding it would be collinear by construction.
HAZARD_CLOCK_TERMS: tuple[str, ...] = ("sr_now_lt35", "sr_now_lt10")


@dataclass
class HazardArm:
    """Discrete-time hazard: P(the possession ends at elapsed second t | it has
    lasted t seconds) = logistic(alpha_t + x b + clock terms).

    CENSORING is native and needs no adjustment: a possession censored at c
    contributes the person-periods t = 0..c-1 with no event and contributes
    NOTHING at t = c, which is exactly the statement "it had not ended when
    the horn went".

    Fitted by L-BFGS on the exact person-period log-likelihood without ever
    materialising the person-period design matrix: the static block is constant
    within a possession, so its gradient is `X' rsum` with rsum the per-
    possession residual sum (`np.add.reduceat`), and the baseline's gradient is
    a `bincount` over elapsed second. The full 2022-2024 expansion is 42M
    person-periods and is used in full -- no subsampling."""

    feature_set_name: str
    features: list[str]
    dropped_zero_variance: list[str]
    mu_scale: np.ndarray
    sd_scale: np.ndarray
    beta: np.ndarray
    alpha: np.ndarray
    clock_coef: np.ndarray
    n_obs: int
    n_person_periods: int
    converged: bool
    nll: float
    name: str = "hazard"

    def _static_eta(self, df: pd.DataFrame) -> np.ndarray:
        x = _matrix(df, self.features)
        z = (x - self.mu_scale) / self.sd_scale
        return self.beta[0] + z @ self.beta[1:]

    def hazard_matrix(self, df: pd.DataFrame) -> np.ndarray:
        eta = self._static_eta(df)[:, None] + self.alpha[np.minimum(GRID, HAZARD_BASELINE_BINS - 1)][None, :]
        sr_now = df["seconds_remaining"].to_numpy(dtype="float64")[:, None] - GRID[None, :]
        eta = eta + self.clock_coef[0] * (sr_now < 35) + self.clock_coef[1] * (sr_now < 10)
        return special.expit(eta)

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        h = self.hazard_matrix(df)
        surv = np.cumprod(1.0 - h, axis=1)
        prev = np.concatenate([np.ones((len(h), 1)), surv[:, :-1]], axis=1)
        return _normalise(prev * h)


def _expand_person_periods(y: np.ndarray, censored: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(counts, offsets, elapsed) of the person-period expansion.

    Uncensored duration d -> the d + 1 rows t = 0..d, the last carrying the
    event. Censored at c -> the c rows t = 0..c-1, none carrying an event."""
    counts = np.where(censored, y, y + 1).astype("int64")
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    total = int(counts.sum())
    elapsed = np.arange(total, dtype="int64") - np.repeat(offsets, counts)
    return counts, offsets, elapsed


def fit_hazard(train: pd.DataFrame, feature_set_name: str, seed: int = 0,
               maxiter: int = 150, l2: float = HAZARD_L2) -> HazardArm:
    feats = feature_set(feature_set_name)
    keep, dropped = _drop_zero_variance(train, feats)
    x = _matrix(train, keep)
    mu_scale = x.mean(axis=0)
    sd_scale = x.std(axis=0)
    sd_scale[sd_scale < 1e-12] = 1.0
    z = np.ascontiguousarray(((x - mu_scale) / sd_scale).astype("float64"))

    y = train["duration_s"].to_numpy(dtype="int64")
    cen = train["censored"].to_numpy(dtype=bool)
    sc = train["seconds_remaining"].to_numpy(dtype="float64")
    counts, offsets, elapsed = _expand_person_periods(y, cen)
    n_pp = len(elapsed)
    keep_rows = counts > 0
    reduce_offsets = offsets[keep_rows]

    bin_idx = np.minimum(elapsed, HAZARD_BASELINE_BINS - 1).astype("int64")
    sr_now = np.repeat(sc, counts) - elapsed
    m35 = (sr_now < 35).astype("float64")
    m10 = (sr_now < 10).astype("float64")
    del sr_now
    # The event indicator is 1 only on the final person-period of an
    # uncensored possession.
    event_pos = (offsets + counts - 1)[(~cen) & keep_rows]

    p = z.shape[1]
    n_beta = p + 1
    n_theta = n_beta + HAZARD_BASELINE_BINS + len(HAZARD_CLOCK_TERMS)
    theta0 = np.zeros(n_theta)
    base_rate = max(float((~cen).sum()) / max(n_pp, 1), 1e-6)
    theta0[0] = float(np.log(base_rate / (1.0 - base_rate)))

    def obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = theta[:n_beta]
        alpha = theta[n_beta:n_beta + HAZARD_BASELINE_BINS]
        cc = theta[n_beta + HAZARD_BASELINE_BINS:]
        static = beta[0] + z @ beta[1:]
        eta = np.repeat(static, counts) + alpha[bin_idx] + cc[0] * m35 + cc[1] * m10
        nll = np.logaddexp(0.0, eta).sum()
        nll -= eta[event_pos].sum()
        r = special.expit(eta)
        r[event_pos] -= 1.0
        rsum_rows = np.add.reduceat(r, reduce_offsets)
        rsum = np.zeros(len(counts))
        rsum[keep_rows] = rsum_rows
        g_beta = np.concatenate([[rsum.sum()], z.T @ rsum])
        g_alpha = np.bincount(bin_idx, weights=r, minlength=HAZARD_BASELINE_BINS)
        g_clock = np.array([float(r @ m35), float(r @ m10)])
        grad = np.concatenate([g_beta, g_alpha, g_clock])
        pen_mask = np.ones(n_theta)
        pen_mask[0] = 0.0
        nll = nll / n_pp + 0.5 * l2 * float((theta ** 2 * pen_mask).sum())
        grad = grad / n_pp + l2 * theta * pen_mask
        return float(nll), grad

    res = optimize.minimize(obj, theta0, jac=True, method="L-BFGS-B",
                            options={"maxiter": maxiter, "maxcor": 20})
    theta = res.x
    return HazardArm(
        feature_set_name=feature_set_name, features=keep, dropped_zero_variance=dropped,
        mu_scale=mu_scale, sd_scale=sd_scale, beta=theta[:n_beta],
        alpha=theta[n_beta:n_beta + HAZARD_BASELINE_BINS],
        clock_coef=theta[n_beta + HAZARD_BASELINE_BINS:],
        n_obs=int(len(y)), n_person_periods=int(n_pp),
        converged=bool(res.success), nll=float(res.fun),
    )


# ===========================================================================
# 8. Arm dispatch
# ===========================================================================
@dataclass
class TwoRegimeArm:
    """Round-2 arm 5, exactly as pre-registered: the round-1 best-CRPS model
    (`lgbm_quantile` on `D_plus_season`) wherever `seconds_remaining > T`, and
    the round-2 fine empirical table wherever `seconds_remaining <= T`.

    The split is on the clock alone, so which component serves a row is decided
    by the SIMULATED clock inside `chain_halves` and by the real one at scoring
    time -- the same rule in both places. `T` is chosen on F1 only.

    CENSORING: whichever component owns the row owns its censoring handling
    too -- Kaplan-Meier below `T` (where every censored row lives, since a
    censored possession by definition started with fewer than its own duration
    of seconds left), exclusion above it."""

    threshold: int
    low: EmpiricalArm
    high: object
    name: str = "two_regime"
    feature_set_name: str = "R2_two_regime"

    @property
    def features(self) -> list[str]:
        out = list(getattr(self.low, "features", []))
        for f in getattr(self.high, "features", []):
            if f not in out:
                out.append(f)
        return out

    @property
    def dropped_zero_variance(self) -> list[str]:
        return list(getattr(self.high, "dropped_zero_variance", []))

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        sr = df["seconds_remaining"].to_numpy(dtype="float64")
        low = sr <= self.threshold
        out = np.empty((len(df), N_GRID), dtype="float64")
        idx_lo = np.flatnonzero(low)
        idx_hi = np.flatnonzero(~low)
        if len(idx_lo):
            out[idx_lo] = self.low.pmf(df.iloc[idx_lo])
        if len(idx_hi):
            out[idx_hi] = self.high.pmf(df.iloc[idx_hi])
        return out


#: The tree-parameter search the round-2 pre-registration allows, on F1 ONLY.
#: A coarse complexity ladder rather than a full sweep: the point is to check
#: that the round-1 setting was not the thing holding the tree arm back, not to
#: tune it to the last decimal.
R2_TREE_PARAM_GRID: tuple[dict, ...] = (
    {"num_leaves": 31, "min_child_samples": 1500},
    {"num_leaves": 63, "min_child_samples": 500},
    {"num_leaves": 127, "min_child_samples": 200},
)


def fit_arm(arm: str, train: pd.DataFrame, feature_set_name: str, seed: int = 0, **kw):
    if arm == "empirical":
        return fit_empirical(train, feature_set_name, seed=seed, **kw)
    if arm in ("lognormal", "gamma"):
        return fit_parametric(train, feature_set_name, family=arm, seed=seed, **kw)
    if arm == "hazard":
        return fit_hazard(train, feature_set_name, seed=seed, **kw)
    if arm == "lgbm_quantile":
        return fit_quantile_tree(train, feature_set_name, seed=seed, **kw)
    raise KeyError(f"unknown arm {arm!r}")


def arm_features(arm) -> list[str]:
    return list(getattr(arm, "features", []))


# ===========================================================================
# 9. Metrics
# ===========================================================================
def score_arm(arm, te: pd.DataFrame, chunk: int = 100_000, pit_seed: int = 20260910) -> dict:
    """CRPS, log score and PIT for one arm on one test slice.

    Censored rows are EXCLUDED from CRPS / log score / PIT -- their true
    duration is unknown, so scoring them against the observed value would
    reward an arm for predicting the horn -- and are reported separately by
    `censored_survival_diagnostic`."""
    y = te["duration_s"].to_numpy(dtype="int64")
    cen = te["censored"].to_numpy(dtype=bool)
    n = len(te)
    crps_row = np.empty(n)
    ls_row = np.empty(n)
    pit_row = np.empty(n)
    mean_row = np.empty(n)
    m2_row = np.empty(n)
    v_all = np.random.default_rng(pit_seed).random(n)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sl = slice(start, stop)
        pm = arm.pmf(te.iloc[sl])
        crps_row[sl] = crps(pm, y[sl])
        ls_row[sl] = log_score(pm, y[sl])
        pit_row[sl] = pit(pm, y[sl], v_all[sl])
        mean_row[sl] = pm @ GRID.astype("float64")
        m2_row[sl] = pm @ (GRID.astype("float64") ** 2)
    ok = ~cen
    defined = ok & np.isfinite(ls_row)
    return {
        "n_scored": int(ok.sum()),
        "n_censored_excluded": int(cen.sum()),
        "crps": float(crps_row[ok].mean()),
        "crps_row": crps_row,
        "logscore_defined": float(ls_row[defined].mean()) if defined.any() else float("nan"),
        "logscore_undef_pct": round(100.0 * float((~np.isfinite(ls_row[ok])).mean()), 4),
        "pit_row": pit_row,
        "pred_mean_row": mean_row,
        "pred_m2_row": m2_row,
        "pred_mean": float(mean_row[ok].mean()),
        "actual_mean": float(y[ok].mean()),
    }


def censored_survival_diagnostic(arm, te: pd.DataFrame, chunk: int = 100_000) -> dict:
    """For the censored rows only: the arm's own P(D > c). A model that
    understood the end-of-period state would put NON-trivial survival mass
    beyond the seconds that were left; a model that puts ~0 there is telling us
    those possessions should already have ended."""
    cen = te["censored"].to_numpy(dtype=bool)
    d = te.loc[cen]
    if not len(d):
        return {"n": 0}
    c = d["duration_s"].to_numpy(dtype="int64")
    surv = np.empty(len(d))
    for start in range(0, len(d), chunk):
        stop = min(start + chunk, len(d))
        sl = slice(start, stop)
        pm = arm.pmf(d.iloc[sl])
        F = np.cumsum(pm, axis=1)
        surv[sl] = 1.0 - F[np.arange(stop - start), c[sl]]
    return {"n": int(len(d)), "mean_pred_survival": float(surv.mean()),
            "median_pred_survival": float(np.median(surv))}


def pit_by_cell(te: pd.DataFrame, pit_row: np.ndarray, min_cell: int = PIT_MIN_CELL) -> pd.DataFrame:
    """K-S of the PIT inside each pre-registered cell (previous end type x
    seconds-remaining bucket). Cells with n < `min_cell` are marked
    UNDERPOWERED and are excluded from the decision rule -- never reported as
    signal OR as absence of signal (`CLAUDE.md`, multi-level evidence)."""
    ok = ~te["censored"].to_numpy(dtype=bool)
    d = pd.DataFrame({"cell": te["pit_cell"].to_numpy()[ok], "u": pit_row[ok]})
    rows = []
    for cell, grp in d.groupby("cell", sort=True):
        n = len(grp)
        D, pv = ks_uniform(grp["u"].to_numpy())
        rows.append({"cell": cell, "n": int(n), "ks_D": D, "ks_p": pv,
                     "powered": n >= min_cell,
                     "leak_sized": bool(n >= min_cell and D > PIT_KS_D_GATE)})
    return pd.DataFrame(rows).sort_values("cell").reset_index(drop=True)


def duration_by_terminal(te: pd.DataFrame, pred_mean: np.ndarray, pred_m2: np.ndarray) -> pd.DataFrame:
    """Mean and SD of duration by TERMINAL EVENT, actual vs model-implied.

    The duration model never sees the terminal event, so this is a test of
    whether the STATE carries the information: if it does, the model's mean
    duration should already be longer for the possessions that ended in a
    two-point jumper than for those that ended in a bonus free-throw trip. The
    model-implied SD is the mixture SD, E[Var] + Var[E], not the SD of the
    point predictions."""
    ok = ~te["censored"].to_numpy(dtype=bool)
    d = pd.DataFrame({
        "terminal_event": te["terminal_event"].to_numpy()[ok],
        "y": te["duration_s"].to_numpy(dtype="float64")[ok],
        "m": pred_mean[ok], "m2": pred_m2[ok],
    })
    rows = []
    for ev, g in d.groupby("terminal_event", sort=True):
        mm = float(g["m"].mean())
        var = float(g["m2"].mean()) - mm ** 2
        rows.append({
            "terminal_event": ev, "n": int(len(g)),
            "actual_mean": round(float(g["y"].mean()), 3),
            "actual_sd": round(float(g["y"].std(ddof=1)), 3),
            "model_mean": round(mm, 3),
            "model_sd": round(float(np.sqrt(max(var, 0.0))), 3),
            "mean_gap": round(mm - float(g["y"].mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("terminal_event").reset_index(drop=True)


def block_bootstrap_se(te: pd.DataFrame, crps_row: np.ndarray, n_rep: int = 200,
                       seed: int = 12345) -> float:
    """Game-block bootstrap SE of the mean CRPS. Possessions inside one game
    share pace, officials and lineups, so the resampling unit is the GAME --
    the same construction as
    `possession_outcome.block_bootstrap_se`."""
    ok = ~te["censored"].to_numpy(dtype=bool)
    games = te["game_id"].to_numpy()[ok]
    vals = crps_row[ok]
    order = np.argsort(games, kind="stable")
    g_sorted = games[order]
    starts = np.flatnonzero(np.concatenate([[True], g_sorted[1:] != g_sorted[:-1]]))
    ends = np.concatenate([starts[1:], [len(g_sorted)]])
    v = vals[order]
    block_sums = np.array([v[s:e].sum() for s, e in zip(starts, ends, strict=False)])
    block_ns = np.array([e - s for s, e in zip(starts, ends, strict=False)], dtype="float64")
    rng = np.random.default_rng(seed)
    n_g = len(block_sums)
    out = np.empty(n_rep)
    for r in range(n_rep):
        pick = rng.integers(0, n_g, n_g)
        out[r] = block_sums[pick].sum() / block_ns[pick].sum()
    return float(out.std(ddof=1))


# ===========================================================================
# 10. The EMERGENT simulator
# ===========================================================================
HALF_LENGTH_S = 1200
MAX_DRAWS_PER_HALF = 400


@dataclass
class ChainResult:
    per_game: pd.DataFrame          # game_id, month, sim_poss, actual_poss
    per_half: pd.DataFrame          # game_id, half, last_start_clock, last_duration
    wrapped_halves: int
    hit_draw_cap: int


def chain_halves(arm, design: pd.DataFrame, seed: int = 20260910,
                 half_length: int = HALF_LENGTH_S,
                 max_draws: int = MAX_DRAWS_PER_HALF) -> ChainResult:
    """The pre-registered EMERGENT test: run the clock model, and only the
    clock model, over two `half_length`-second halves per game.

    THE STATE INPUT IS REAL, THE CLOCK IS THE SIM'S. Draw j of a half takes its
    whole feature row from the j-th REAL possession of that half -- previous
    end type, score difference, bonus, which team has the ball, both teams'
    priors -- and overrides only the clock-dependent columns
    (`CLOCK_DEPENDENT_FEATURES`) with the simulated clock. No event model, no
    score model and no L3 output is involved, so the possession count that
    comes out is a property of THIS model alone. That is what makes it a fair
    G1 read on a sub-model that cannot yet be run inside a full engine.

    If a half's real possession list is exhausted before the simulated clock
    runs out, the list WRAPS to the start of that half (counted in
    `wrapped_halves`): a faster sim needs more state rows than the game
    supplied, and re-using that half's own sequence is the least informative
    way to supply them.

    The final possession of a half is truncated at the horn, exactly as the
    censored possessions in the data are.

    RNG: the engine-wide counter-based stream keyed on
    (seed, game_id, "clock"); draw index `(half - 1) * max_draws + j`, so the
    two halves of a game never share a draw and no game's draws depend on which
    other games are in the run."""
    reg = design[design["period"] <= 2.0].copy()
    reg = reg.sort_values(["game_id", "period", "poss_index"], kind="stable")
    reg["_row"] = np.arange(len(reg))

    games = reg["game_id"].drop_duplicates().to_numpy()
    month = reg.groupby("game_id")["month"].first()
    actual = reg.groupby("game_id").size() / 2.0

    seqs: dict[tuple[int, int], np.ndarray] = {}
    for (gid, per), g in reg.groupby(["game_id", "period"], sort=False):
        seqs[(int(gid), int(per))] = g["_row"].to_numpy()

    base = reg.reset_index(drop=True)

    sim_counts = {int(g): 0 for g in games}
    last_start: list[dict] = []
    wrapped = 0
    hit_cap = 0

    for half in (1, 2):
        gh = [(int(g), seqs.get((int(g), half))) for g in games]
        gh = [(g, s) for g, s in gh if s is not None and len(s) > 0]
        if not gh:
            continue
        gids = np.array([g for g, _ in gh], dtype="int64")
        lens = np.array([len(s) for _, s in gh], dtype="int64")
        maxlen = int(lens.max())
        seq_mat = np.zeros((len(gh), maxlen), dtype="int64")
        for i, (_, s) in enumerate(gh):
            seq_mat[i, :len(s)] = s
        keys = crng.stream_keys(seed, gids, RNG_FAMILY)
        clock = np.full(len(gh), float(half_length))
        active = clock > 0
        last_sc = np.zeros(len(gh))
        last_dur = np.zeros(len(gh))
        counts = np.zeros(len(gh), dtype="int64")

        for j in range(max_draws):
            idx = np.flatnonzero(active)
            if not len(idx):
                break
            ln = lens[idx]
            # Wrap SKIPS index 0: the half's first possession is the only one
            # whose previous-end type is `period_start`, and injecting that
            # state in the middle of a half would be a state the game never
            # reaches. Everything from index 1 on is an ordinary mid-half
            # state and is safe to re-use.
            pos = np.where(j < ln, j, 1 + (j - ln) % np.maximum(ln - 1, 1))
            pos = np.minimum(pos, ln - 1)
            rows = seq_mat[idx, pos]
            block = base.iloc[rows].copy()
            sc = clock[idx]
            apply_clock_override(block, sc, float(half))
            u = crng.uniforms(keys[idx], (half - 1) * max_draws + j)
            draw = sample_from_pmf(arm.pmf(block), u).astype("float64")
            trunc = draw >= sc
            eff = np.where(trunc, sc, draw)
            last_sc[idx] = sc
            last_dur[idx] = eff
            counts[idx] += 1
            clock[idx] = sc - eff
            active = clock > 0
            if j == max_draws - 1 and active.any():
                hit_cap += int(active.sum())

        wrapped += int((counts > lens).sum())
        for i, g in enumerate(gids):
            sim_counts[int(g)] += int(counts[i])
            last_start.append({"game_id": int(g), "half": half,
                               "last_start_clock": float(last_sc[i]),
                               "last_duration": float(last_dur[i])})

    per_game = pd.DataFrame({
        "game_id": games,
        "month": month.reindex(games).to_numpy(),
        "sim_poss": np.array([sim_counts[int(g)] for g in games], dtype="float64") / 2.0,
        "actual_poss": actual.reindex(games).to_numpy(),
    })
    return ChainResult(per_game=per_game, per_half=pd.DataFrame(last_start),
                       wrapped_halves=int(wrapped), hit_draw_cap=int(hit_cap))


#: A half is CLOCK-COMPLETE if its logged possessions account for the whole
#: period, i.e. the last one ends within this many seconds of the horn.
#:
#: WHY THIS EXISTS (measured 2026-09-10, round 2). The CBBD event stream stops
#: before the horn in a substantial minority of halves: only 44.6% of 2025
#: halves have a last possession ending at exactly 0:00, the median unaccounted
#: time is 1 s but the 90th percentile is 17 s and the mean is 5.29 s. The
#: pre-registered end-of-half statistic is measured on the ACTUAL last LOGGED
#: possession, so those halves report a last possession that starts at 40 s and
#: "ends" at 20 s -- which the sim, which always runs its clock to zero, can
#: never reproduce. Restricting to clock-complete halves is the like-for-like
#: comparison, and it moves the actual share of halves whose last possession
#: starts under 35 s from 0.8832 to 0.9556 on 2025. Both are reported; the
#: pre-registered (all-halves) number remains the one the gate reads.
CLOCK_COMPLETE_TOL_S = 2


def actual_end_of_half(design: pd.DataFrame) -> pd.DataFrame:
    """The real last possession of each regulation half: its start clock, its
    duration, and the seconds still on the clock when the feed stopped."""
    reg = design[design["period"] <= 2.0]
    reg = reg.sort_values(["game_id", "period", "poss_index"], kind="stable")
    last = reg.groupby(["game_id", "period"]).tail(1)
    sc = last["seconds_remaining"].to_numpy(dtype="float64")
    du = last["duration_s"].to_numpy(dtype="float64")
    return pd.DataFrame({
        "game_id": last["game_id"].to_numpy(),
        "half": last["period"].to_numpy().astype("int64"),
        "last_start_clock": sc,
        "last_duration": du,
        "last_end_clock": sc - du,
        "clock_complete": (sc - du) <= CLOCK_COMPLETE_TOL_S,
    })


#: The end-of-half statistic the round-2 pre-registration promotes from a
#: diagnostic to a GATE: the share of halves whose last possession starts with
#: fewer than this many seconds left, and that possession's mean duration.
EOH_WINDOW_S = 35


def eoh_stats(half_frame: pd.DataFrame) -> tuple[float, float]:
    """(share of halves whose last possession starts under EOH_WINDOW_S,
    mean duration of those possessions)."""
    late = half_frame["last_start_clock"].to_numpy(dtype="float64") < EOH_WINDOW_S
    share = float(late.mean()) if len(half_frame) else float("nan")
    dur = float(half_frame.loc[late, "last_duration"].mean()) if late.any() else float("nan")
    return share, dur


def eoh_actual_block_bootstrap_se(actual_half: pd.DataFrame, n_rep: int = 400,
                                  seed: int = 4242) -> tuple[float, float]:
    """Game-block bootstrap SE of the two ACTUAL end-of-half statistics. The
    resampling unit is the GAME because a game contributes two halves that
    share pace, officials and lineups."""
    games = actual_half["game_id"].to_numpy()
    order = np.argsort(games, kind="stable")
    g = games[order]
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    ends = np.concatenate([starts[1:], [len(g)]])
    blocks = [order[s:e] for s, e in zip(starts, ends, strict=False)]
    sc = actual_half["last_start_clock"].to_numpy(dtype="float64")
    du = actual_half["last_duration"].to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    shares, durs = np.empty(n_rep), np.empty(n_rep)
    n_g = len(blocks)
    for r in range(n_rep):
        pick = np.concatenate([blocks[i] for i in rng.integers(0, n_g, n_g)])
        late = sc[pick] < EOH_WINDOW_S
        shares[r] = late.mean()
        durs[r] = du[pick][late].mean() if late.any() else np.nan
    return float(shares.std(ddof=1)), float(np.nanstd(durs, ddof=1))


def eoh_seed_noise(arm, design: pd.DataFrame, seeds: tuple[int, ...]) -> tuple[float, float]:
    """Seed-to-seed SD of the two SIM end-of-half statistics: a spec-identical
    re-chain under a different seed, which is what `CLAUDE.md` means by a noise
    floor."""
    shares, durs = [], []
    for s in seeds:
        res = chain_halves(arm, design, seed=s)
        sh, du = eoh_stats(res.per_half)
        shares.append(sh)
        durs.append(du)
    return float(np.std(shares, ddof=1)), float(np.std(durs, ddof=1))


def emergent_report(res: ChainResult, actual_half: pd.DataFrame,
                    month_min_games: int = G1_MONTH_MIN_GAMES,
                    eoh_floor: dict | None = None) -> dict:
    """G1 on the emergent possession count, plus the end-of-half check."""
    pg = res.per_game.dropna(subset=["sim_poss", "actual_poss"])
    sim, act = pg["sim_poss"].to_numpy(), pg["actual_poss"].to_numpy()
    ks_D, ks_p = stats.ks_2samp(sim, act)[:2]

    months = []
    for m, g in pg.groupby("month", sort=True):
        powered = len(g) >= month_min_games
        dm = float(g["sim_poss"].mean() - g["actual_poss"].mean())
        ds = float(g["sim_poss"].std(ddof=1) - g["actual_poss"].std(ddof=1))
        months.append({
            "month": int(m), "n_games": int(len(g)), "powered": bool(powered),
            "sim_mean": round(float(g["sim_poss"].mean()), 3),
            "actual_mean": round(float(g["actual_poss"].mean()), 3),
            "mean_delta": round(dm, 3),
            "sim_sd": round(float(g["sim_poss"].std(ddof=1)), 3),
            "actual_sd": round(float(g["actual_poss"].std(ddof=1)), 3),
            "sd_delta": round(ds, 3),
            "pass": bool(powered and abs(dm) <= G1_MEAN_TOL and abs(ds) <= G1_SD_TOL),
        })

    sim_half = res.per_half
    a_share, a_dur = eoh_stats(actual_half)
    s_share, s_dur = eoh_stats(sim_half)
    eoh = {
        "actual_share_last_poss_under_35s": round(a_share, 4),
        "sim_share_last_poss_under_35s": round(s_share, 4),
        "actual_mean_duration_under_35s": round(a_dur, 3),
        "sim_mean_duration_under_35s": round(s_dur, 3),
        "share_gap": round(s_share - a_share, 5),
        "duration_gap": round(s_dur - a_dur, 4),
        "n_actual_halves": int(len(actual_half)), "n_sim_halves": int(len(sim_half)),
    }
    # ROUND 2: this is a GATE, not a diagnostic. The tolerance is the F1-derived
    # noise floor passed in by the trainer; round 1 passes nothing and the gate
    # reads as None, which is how the same function serves both rounds.
    if eoh_floor:
        eoh["share_floor"] = eoh_floor["share"]
        eoh["duration_floor"] = eoh_floor["duration"]
        eoh["share_pass"] = bool(abs(s_share - a_share) <= eoh_floor["share"])
        eoh["duration_pass"] = bool(abs(s_dur - a_dur) <= eoh_floor["duration"])
        eoh["pass"] = bool(eoh["share_pass"] and eoh["duration_pass"])
    else:
        eoh["pass"] = None

    mean_delta = float(sim.mean() - act.mean())
    sd_delta = float(sim.std(ddof=1) - act.std(ddof=1))
    powered_months = [m for m in months if m["powered"]]
    return {
        "n_games": int(len(pg)),
        "sim_mean": round(float(sim.mean()), 3), "actual_mean": round(float(act.mean()), 3),
        "mean_delta": round(mean_delta, 3),
        "sim_sd": round(float(sim.std(ddof=1)), 3), "actual_sd": round(float(act.std(ddof=1)), 3),
        "sd_delta": round(sd_delta, 3),
        "count_ks_D": round(float(ks_D), 4), "count_ks_p": float(ks_p),
        "overall_pass": bool(abs(mean_delta) <= G1_MEAN_TOL and abs(sd_delta) <= G1_SD_TOL),
        "months": months,
        "all_powered_months_pass": bool(all(m["pass"] for m in powered_months)) if powered_months else False,
        "n_powered_months": len(powered_months),
        "end_of_half": eoh,
        "wrapped_halves": res.wrapped_halves,
        "hit_draw_cap": res.hit_draw_cap,
        "g1_pass": bool(abs(mean_delta) <= G1_MEAN_TOL and abs(sd_delta) <= G1_SD_TOL
                        and bool(powered_months) and all(m["pass"] for m in powered_months)),
        "eoh_pass": eoh["pass"],
        "clock_complete": clock_complete_read(res, actual_half),
    }


def clock_complete_read(res: ChainResult, actual_half: pd.DataFrame) -> dict:
    """SECONDARY, LABELLED read of the same two comparisons on the halves and
    games whose logged possessions account for the whole period.

    This is not a substitute for the pre-registered numbers and does not feed
    any gate. It exists because the actual end-of-half statistic is contaminated
    by CBBD feed truncation (see `CLOCK_COMPLETE_TOL_S`), and the project's rule
    is that grading truth is checked before a gate is read -- not that a gate is
    quietly re-based."""
    if "clock_complete" not in actual_half.columns:
        return {}
    ah = actual_half[actual_half["clock_complete"]]
    if not len(ah):
        return {}
    keys = set(zip(ah["game_id"].to_numpy(), ah["half"].to_numpy(), strict=False))
    sh = res.per_half
    sel = np.array([(g, h) in keys for g, h in
                    zip(sh["game_id"].to_numpy(), sh["half"].to_numpy(), strict=False)])
    a_share, a_dur = eoh_stats(ah)
    s_share, s_dur = eoh_stats(sh[sel])

    # a game is clock-complete only if BOTH of its halves are
    per_game = actual_half.groupby("game_id")["clock_complete"].all()
    good = set(per_game[per_game].index)
    pg = res.per_game[res.per_game["game_id"].isin(good)].dropna(subset=["sim_poss", "actual_poss"])
    out = {
        "n_halves": int(len(ah)), "half_share_of_all": round(len(ah) / len(actual_half), 4),
        "eoh_actual_share": round(a_share, 4), "eoh_sim_share": round(s_share, 4),
        "eoh_share_gap": round(s_share - a_share, 5),
        "eoh_actual_mean_duration": round(a_dur, 3), "eoh_sim_mean_duration": round(s_dur, 3),
        "eoh_duration_gap": round(s_dur - a_dur, 4),
    }
    if len(pg):
        out.update({
            "n_games": int(len(pg)),
            "sim_mean": round(float(pg["sim_poss"].mean()), 3),
            "actual_mean": round(float(pg["actual_poss"].mean()), 3),
            "mean_delta": round(float(pg["sim_poss"].mean() - pg["actual_poss"].mean()), 3),
            "sim_sd": round(float(pg["sim_poss"].std(ddof=1)), 3),
            "actual_sd": round(float(pg["actual_poss"].std(ddof=1)), 3),
            "sd_delta": round(float(pg["sim_poss"].std(ddof=1) - pg["actual_poss"].std(ddof=1)), 3),
        })
    return out
