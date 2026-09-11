"""
fg_make.py -- L3 FIELD-GOAL MAKE sub-model: does an attempt go in.

Pre-registration (verbatim, PM, 2026-09-10): `docs/models/fg_make/experiments.md`
section 1. Trainer: `scripts/train_fg_make_v1.py`. Feature provenance:
`docs/models/fg_make/features.md`.

===========================================================================
TARGET, AND WHY THERE ARE THREE OF THEM
===========================================================================
One row per FIELD-GOAL ATTEMPT; the target is made (1) or missed (0). The
pre-registration models the three shot classes SEPARATELY -- `FGA_rim`,
`FGA_jump2`, `FGA_3` -- because the base rate, the shooter-skill signal and
the defensive signal are different quantities in each. Nothing in this module
ever fits across classes: `fit_by_class` slices the frame by `shot_class` and
hands each slice its own estimator, and `tests/test_fg_make.py` proves that
corrupting one class's rows cannot move another class's predictions.

The class labels come from `cbb_sim.pbp.events.classify_frame` under the **v2**
possessions build, i.e. with the L16 rim-location override applied at the
threshold that build recorded (`data/processed/possessions_v2/build_report.json`,
`rim_override.max_ft`). This is the one place the possessions version matters
to this model and it matters a lot: the override moves ESPN's 2025 putback
mistag out of `FGA_jump2` and into `FGA_rim`, and a rim/jumper make model
trained on the un-repaired labels would be fitting a vendor defect.

Two population rules from the pre-registration, both verified by the trainer
rather than asserted here:

  * **Blocked shots are misses.** A `Block Shot` row is folded into the
    attempt as `blocked` by `cbb_sim.models.event_stream` and the attempt keeps
    its own `made = False`. `blocked` is therefore a POST-OUTCOME field -- it
    exists only because the shot missed -- and it is in `BANNED_FEATURES`.
  * **And-one attempts are makes.** An and-one is a made field goal plus a
    foul; the FGA row is already `made = True` and is kept exactly as it is.
    `and_one` is derived for the descriptive table only and is likewise banned
    as a feature (the foul that follows the shot cannot be known at release).

===========================================================================
LEAK SAFETY
===========================================================================
Three separate hazards, all closed by construction:

  1. **The attempt's own outcome.** Every as-of rate -- shooter, team, league,
     position, defender -- is an EXPANDING mean over GAMES STRICTLY BEFORE the
     current one within the season (`prob_metrics.expanding_asof`, which is
     `cumsum() - value`), so an attempt cannot see its own game, let alone its
     own shot. `prior_season_*` is a completed previous season.
  2. **Post-outcome fields.** `BANNED_FEATURES` names them: the target itself,
     `blocked`, `and_one`, and the assist. The pre-registration singles out the
     assisted flag: in the CBBD/ESPN feed an assist is logged as a property of
     a MADE basket, so "was this assisted" is knowable only after the ball goes
     in and would be a near-perfect predictor of its own target. It is not
     built, not stored, and not readable from any column this module produces.
  3. **The chance-state block is at-release, not post-outcome.** `chance_number`
     and `chance_elapsed_s` are read off the events that STARTED the current
     chance -- all strictly before the attempt -- plus the attempt's own clock,
     which is when the ball left the shooter's hands. This is deliberately NOT
     the `is_transition` the change ledger bans at L5 and flags at L3: that one
     is a function of the chance's own `duration_s`, i.e. of when the chance
     ENDS, which at L5 is the target and at L3 is contemporaneous with the
     terminal event being predicted. Here the attempt's existence and its
     release time are GIVEN and only make/miss is modelled, so elapsed-time-at-
     release is an honest conditioning variable. `tests/test_fg_make.py` pins
     it: flipping an attempt's own `made` flag leaves every one of its own
     features bit-identical.

===========================================================================
SHOOTER IDENTITY
===========================================================================
The key is the CBBD player id, exactly as in `cbb_sim.models.free_throw` and
for the reason measured there (change-ledger row B: the ESPN athlete id is
present on 100% of 2024-2025 attempts and 0% of 2022-2023, because CBBD
rosters were pulled for 2024-2026 only, while the CBBD id maps to the same
ESPN id on 100.0% of the players the crosswalk covers in two consecutive
seasons). The ESPN id is attached wherever the crosswalk resolves it and its
coverage is a reported number.

THE ONE PRE-REGISTERED FEATURE THAT COULD NOT BE BUILT: `minutes-to-date`.
Minutes live in hoopR `player_box`, keyed on the ESPN athlete id, and the
crosswalk that would reach them does not exist for two of the three training
seasons (above); CBBD `onFloor` cannot substitute because it is empty at the
source before 2024 (L13). Rather than feed the model a column that is real in
2024-2025 and structurally zero in 2022-2023 -- which would make it a season
dummy wearing a minutes label -- the bundle carries the two exposure counters
that ARE available in CBBD id space for every season, `shooter_games_asof` and
`shooter_fga_asof`, and the trainer reports the ESPN-id coverage per season so
the substitution is auditable. Recorded in `features.md` section 3.

===========================================================================
FOLDS AND THE SEAL
===========================================================================
F1 trains {2022, 2023} and tests 2024; F2 trains {2022, 2023, 2024} and tests
2025 and is the selection fold. Season 2026 is sealed and cannot enter a fold:
`fold_slices` calls `cbb_sim.data.seal.assert_not_sealed` on both slices.

`D_plus_lineup` has its OWN fold, `L2`: train 2024, test 2025, because CBBD
`onFloor` is empty at the source before 2024 (L13). C and D are re-scored
against each other on exactly the rows of that fold that carry all ten on-floor
ids, so the comparison is like-for-like -- the same construction the rebound
model's D arm used.

The universe is the pre-registration's: D-I, non-truncated, AND `pbp_complete`
(the CBBD event stream accounts for the final score). That is stricter than the
universe the rebound and free-throw bake-offs used, so row counts here are not
comparable to theirs; the trainer reports both.

===========================================================================
THE SHOOTER-KEY DEFECT (round 3, 2026-09-10) AND ITS FIX
===========================================================================
Every function in this module that resolves "who took this shot" ultimately
reads `event_stream.build_stream`'s `player_id` column, which up to and
including round 2b is `participant_1_id` on every row -- CBBD's field-goal
`participants` array is not ordered shooter-first, and on an ASSISTED made
field goal `participant_1_id` is the ASSISTER on 100.000% of the ~49% of rows
where it disagrees with the dedicated `shot_shooter_id` column
(`docs/tests/shooter_key_audit_2026-09-10.md`, `docs/models/change_ledger.md`
row "CBBD's `participant_1_id` is the ASSISTER..."). `usage` (L4) fixed this by
adding a `shooter_key` parameter to `event_stream.build_stream` /
`build_usage_events`, defaulting to the OLD column so every other model stays
byte-identical and opting a caller in explicitly. `build_fg_events` and
`_season_events` follow the identical pattern: `shooter_key` defaults to
`event_stream.DEFAULT_SHOOTER_KEY` ("participant_1_id", today's behaviour,
byte for byte) and threads straight through to `build_stream`; passing
`shooter_key="shot_shooter_id"` re-keys every FGA row's shooter (and only FGA
rows -- missed FGAs, free throws and every non-shooting event this module never
reads are untouched) and leaves a row's shooter id missing, never imputed and
never falling back to `participant_1_id`, wherever `shot_shooter_id` itself is
absent (0.04-0.27% of FGA rows per class). `build_design`'s existing
`np.isfinite(events["shooter_id"])` drop (module docstring above) is what
removes those rows; it needed no change because it already keys off whichever
column `events["shooter_id"]` was built from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.data.seal import assert_not_sealed
from cbb_sim.models import event_stream as ES
from cbb_sim.models import prob_metrics as PM
from cbb_sim.models.free_throw import load_espn_ids, load_positions
from cbb_sim.ratings import own_ratings as orat

DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_ARTIFACT_DIR = Path("data/processed/models/fg_make")
DEFAULT_RATINGS_DIR = Path("data/processed/ratings")

#: FIXED class order for the binary target. Every probability matrix in this
#: module and every artifact the trainer writes uses it.
CLASSES: tuple[str, ...] = ("MISS", "MAKE")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

#: The three shot classes, each of which gets its OWN model. `class_key` is the
#: short form used in every wide as-of column name.
SHOT_CLASSES: tuple[str, ...] = ("FGA_rim", "FGA_jump2", "FGA_3")
CLASS_KEY: dict[str, str] = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}
CLASS_KEYS: tuple[str, ...] = ("rim", "jump2", "three")

#: The possessions build whose rim/jumper labelling the target classes come
#: from. The pre-registration names v2 (the L16 override).
DEFAULT_VERSION = "v2"

FOLDS: dict[str, dict[str, list[int]]] = {
    "F1": {"train": [2022, 2023], "test": [2024]},
    "F2": {"train": [2022, 2023, 2024], "test": [2025]},
}
SELECTION_FOLD = "F2"

#: The lineup bundle's own fold (L13: CBBD `onFloor` is empty before 2024).
LINEUP_FOLDS: dict[str, dict[str, list[int]]] = {"L2": {"train": [2024], "test": [2025]}}
LINEUP_SELECTION_FOLD = "L2"

#: Shrinkage grid for a DEFENDER's as-of allowed rate, in attempts faced.
#: FITTED on the lineup fold's train season, never assumed (L13).
LINEUP_PRIOR_GRID: tuple[int, ...] = (0, 100, 250, 500, 1000)

#: Columns that must never reach a feature matrix, and why. The test suite
#: asserts that no bundle contains one and that none of them is even built.
BANNED_FEATURES: dict[str, str] = {
    "made": "the target",
    "y": "the target",
    "blocked": ("a Block Shot row exists only because the attempt missed; it is a "
                "post-outcome field, not a pre-release one"),
    "and_one": ("the foul that turns a make into an and-one is logged after the ball goes "
                "in; knowing it is knowing the target"),
    "assisted": ("the pre-registration's own exclusion: in this feed an assist is a property "
                 "of a MADE basket, so the flag is post-outcome. It is not built at all"),
    "shot_made": "the target under its raw feed name",
}

#: NCAA period lengths in seconds: two 20-minute halves, 5-minute overtimes.
PERIOD_SECONDS = {1: 1200, 2: 1200}
OT_SECONDS = 300

#: A chance that starts on a live defensive rebound or a turnover and produces
#: a shot within this many seconds is `is_transition`. The threshold is the one
#: `cbb_sim.pbp.possessions` already uses, kept identical so the two layers
#: mean the same thing by the word.
TRANSITION_MAX_S = 8.0
TRANSITION_START_REASONS: frozenset[str] = frozenset({"DREB", "TOV"})

#: Elapsed time is clipped to this before it becomes a feature. The NCAA shot
#: clock is 30 s; anything past 60 s is a feed artefact (a missing chance-start
#: event), and the trainer reports the share that hits the clip instead of the
#: clip hiding it.
MAX_CHANCE_ELAPSED_S = 60.0


# ===========================================================================
# 1. Field-goal attempts from the event stream
# ===========================================================================
def build_fg_events(
    seasons: list[int],
    universe: pd.DataFrame | None = None,
    version: str | None = DEFAULT_VERSION,
    poss_dir: Path | str | None = None,
    pbp_dir: Path | str = ES.DEFAULT_PBP_DIR,
    universe_path: Path | str = DEFAULT_UNIVERSE,
    require_pbp_complete: bool = True,
    shooter_key: str = ES.DEFAULT_SHOOTER_KEY,
) -> pd.DataFrame:
    """One row per field-goal attempt for `seasons` (module docstring).

    `version` selects the possessions build whose rim-location override
    threshold is applied when the stream is classified; it is read out of that
    build's own `build_report.json` by `event_stream.rim_override_for_version`,
    so the threshold is never typed in here.

    `shooter_key` (module docstring, "THE SHOOTER-KEY DEFECT") defaults to
    `event_stream.DEFAULT_SHOOTER_KEY` ("participant_1_id", today's behaviour,
    byte for byte) and is threaded straight through to
    `event_stream.build_stream`; pass `"shot_shooter_id"` to re-key the FGA
    shooter onto the dedicated, correct column."""
    if universe is None:
        universe = ES.load_universe(universe_path, require_pbp_complete=require_pbp_complete)
    max_ft = ES.rim_override_for_version(version, poss_dir)
    frames = [_season_events(int(s), universe, max_ft, pbp_dir, shooter_key=shooter_key)
              for s in seasons]
    out = pd.concat(frames, ignore_index=True)
    out.attrs["possessions_version"] = version or DEFAULT_VERSION
    out.attrs["rim_override_max_ft"] = float(max_ft)
    out.attrs["shooter_key"] = shooter_key
    return out


def _period_seconds(period: np.ndarray) -> np.ndarray:
    return np.where(period <= 2, PERIOD_SECONDS[1], OT_SECONDS).astype("float64")


def chance_state(st: pd.DataFrame) -> pd.DataFrame:
    """Chance number, chance start reason and elapsed-seconds-at-release for
    every row of an event stream.

    A CHANCE ends when the ball changes hands or is rebounded: a made field
    goal, a made last free throw of a non-technical trip, a defensive rebound,
    a turnover, a dead-ball rebound, or a period boundary. An OFFENSIVE rebound
    ends the chance too but keeps the possession, which is exactly the
    distinction `chance_number` carries: it is 1 on a possession's first chance
    and increments once per offensive rebound.

    Everything here is read off events STRICTLY BEFORE the row (the chance's
    own start) plus the row's own clock; the row's own class and outcome are
    excluded from its own chance by construction (`cumsum() - self`), which is
    what makes the block safe to feed a make model (module docstring, LEAK
    SAFETY item 3)."""
    cls = st["cls"].to_numpy(dtype=object)
    made = st["made"].to_numpy()
    g = st["cbbd_game_id"].to_numpy()
    sd = st["side"].to_numpy()
    sec = st["sec"].to_numpy().astype("float64")
    period = st["period"].to_numpy()

    is_fga = np.isin(cls, SHOT_CLASSES)
    made_last_ft = ((cls == "FT_made") & st["trip_last"].to_numpy()
                    & ~st["trip_is_technical"].to_numpy())
    poss_end = ((is_fga & made) | made_last_ft
                | np.isin(cls, ["DREB", "TOV", "DeadBallReb", "end_period", "end_game"]))
    is_oreb = cls == "OREB"
    start_ev = poss_end | is_oreb

    # Every GAME also opens a chance at its first row, otherwise that game's
    # first offensive rebound would chain to the PREVIOUS game's last chance
    # and be numbered 1 instead of 2. The synthetic entry is labelled
    # `game_start`, which is a reset by construction.
    first_of_game = np.concatenate([[True], g[1:] != g[:-1]])
    synth_start = first_of_game & ~start_ev
    start_all = start_ev | first_of_game

    # group 0 = "before any start event at all"; group k = the chance opened by
    # the k-th start. `cumsum - self` is what keeps a row out of the chance it
    # itself opens.
    grp = (np.cumsum(start_all) - start_all).astype("int64")
    starts = np.flatnonzero(start_all)
    synth = synth_start[starts]
    s_sec = np.concatenate([[np.nan], sec[starts]])
    s_cls = np.concatenate([np.array(["game_start"], dtype=object),
                            np.where(synth, "game_start", cls[starts]).astype(object)])
    s_side = np.concatenate([[-9], np.where(synth, -9, sd[starts])])
    s_game = np.concatenate([np.array([-1], dtype=g.dtype), g[starts]])
    s_period = np.concatenate([[-1], period[starts]])

    # chance number: consecutive OREB starts chain, anything else resets to 1.
    s_is_oreb = s_cls == "OREB"
    same_game_chain = np.concatenate([[False], s_game[1:] == s_game[:-1]])
    chain = s_is_oreb & same_game_chain
    ix = np.arange(len(s_cls))
    last_reset = np.maximum.accumulate(np.where(~chain, ix, -1))
    s_chance_no = (ix - last_reset + 1).astype("int32")

    # the offence of the chance, from the start event's own semantics
    s_off = np.where(np.isin(s_cls, ["OREB", "DREB"]), s_side,
                     np.where(np.isin(s_cls, ["TOV", "FGA_rim", "FGA_jump2", "FGA_3",
                                              "FT_made", "DeadBallReb"]), 1 - s_side, -9))
    s_off = np.where(s_side < 0, -9, s_off)

    start_sec = s_sec[grp]
    start_cls = s_cls[grp]
    start_period = s_period[grp]
    start_game = s_game[grp]
    start_off = s_off[grp]
    chance_no = s_chance_no[grp]

    # A chance whose start event belongs to another game or another period, or
    # which is a period boundary, actually began at the period's opening tip.
    fresh = ((start_game != g) | (start_period != period)
             | np.isin(start_cls, ["end_period", "end_game", "game_start"]))
    reason = np.where(fresh, "period_start", start_cls.astype(str))
    start_sec = np.where(fresh, _period_seconds(period), start_sec)
    chance_no = np.where(fresh, 1, chance_no).astype("int32")
    start_off = np.where(fresh, -9, start_off)

    elapsed = start_sec - sec
    over = elapsed > MAX_CHANCE_ELAPSED_S
    under = elapsed < 0
    elapsed = np.clip(elapsed, 0.0, MAX_CHANCE_ELAPSED_S)

    return pd.DataFrame({
        "chance_number": chance_no,
        "chance_start_reason": reason,
        "chance_elapsed_s": elapsed.astype("float32"),
        "chance_elapsed_clipped": (over | under),
        "chance_side_ok": (start_off == sd) | (start_off == -9),
        "chance_side_known": start_off != -9,
        "is_transition": (np.isin(reason, list(TRANSITION_START_REASONS))
                          & (elapsed <= TRANSITION_MAX_S)),
    }, index=st.index)


def _season_events(season: int, universe: pd.DataFrame, max_ft: float,
                   pbp_dir: Path | str,
                   shooter_key: str = ES.DEFAULT_SHOOTER_KEY) -> pd.DataFrame:
    st = ES.build_stream(season, universe, rim_override_max_ft=max_ft, pbp_dir=pbp_dir,
                         shooter_key=shooter_key)
    ch = chance_state(st)
    cls = st["cls"].to_numpy(dtype=object)
    made = st["made"].to_numpy()
    g = st["cbbd_game_id"].to_numpy()
    sd = st["side"].to_numpy()
    sec = st["sec"].to_numpy()
    n = len(st)

    # and-one, for the DESCRIPTIVE table only (it is a banned feature): the
    # attempt is made, the next row is a foul on the defence at the same clock,
    # and the row after that is a free throw by the shooting team. This is the
    # mirror image of the signature `event_stream` uses on the free-throw side.
    nxt_cls = np.concatenate([cls[1:], [None]])
    nxt_side = np.concatenate([sd[1:], [-9]])
    nxt_sec = np.concatenate([sec[1:], [-1]])
    nxt_game = np.concatenate([g[1:], np.array([-1], dtype=g.dtype)])
    pad = 2 if n >= 2 else n
    nn_cls = np.concatenate([cls[pad:], np.array([None] * pad, dtype=object)])
    nn_side = np.concatenate([sd[pad:], np.full(pad, -9)])
    nn_game = np.concatenate([g[pad:], np.full(pad, -1, dtype=g.dtype)])
    and_one = (made & (nxt_cls == "foul") & (nxt_side != sd) & (nxt_sec == sec)
               & (nxt_game == g) & np.isin(nn_cls, list(ES.FT_CLASSES))
               & (nn_side == sd) & (nn_game == g))

    idx = np.flatnonzero(np.isin(cls, SHOT_CLASSES))
    off_home = sd[idx] == 0
    hs, as_ = st["home_score"].to_numpy()[idx], st["away_score"].to_numpy()[idx]
    out = pd.DataFrame({
        "game_id": st["game_id"].to_numpy()[idx],
        "cbbd_game_id": g[idx],
        "season": np.full(len(idx), season, dtype="int16"),
        "game_date": st["game_date"].to_numpy()[idx],
        "neutral_site": st["neutral_site"].to_numpy()[idx],
        "period": st["period"].to_numpy()[idx],
        "seconds_remaining": sec[idx],
        "score_diff": np.where(off_home, hs - as_, as_ - hs).astype("int32"),
        "off_team_id": st["team_id"].to_numpy()[idx],
        "def_team_id": st["opp_id"].to_numpy()[idx],
        "offense_is_home": off_home,
        "shooter_id": st["player_id"].to_numpy()[idx],
        "shot_class": cls[idx],
        "made": made[idx],
        "blocked": st["blocked"].to_numpy()[idx],
        "and_one": and_one[idx],
        "off_in_bonus": ES.in_bonus(st["fouls_opp_prior"].to_numpy()[idx]),
        "off_in_double_bonus": ES.in_double_bonus(st["fouls_opp_prior"].to_numpy()[idx]),
    })
    for c in ch.columns:
        out[c] = ch[c].to_numpy()[idx]
    for c in ES.ON_FLOOR_COLS:
        out[c] = st[c].to_numpy()[idx]
    out["class_key"] = pd.Series(out["shot_class"]).map(CLASS_KEY).to_numpy()
    return out


# ===========================================================================
# 2. As-of form: team, league, shooter, position
# ===========================================================================
def _wide_counts(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-row attempt/make indicators, one pair of columns per shot class."""
    ck = df["class_key"].to_numpy()
    made = df["made"].to_numpy()
    out: dict[str, np.ndarray] = {}
    for k in CLASS_KEYS:
        m = ck == k
        out[f"att_{k}"] = m.astype("int32")
        out[f"mk_{k}"] = (m & made).astype("int32")
    out["att_all"] = np.ones(len(df), dtype="int32")
    out["mk_all"] = made.astype("int32")
    return out


COUNT_COLS: tuple[str, ...] = tuple(
    [f"att_{k}" for k in CLASS_KEYS] + [f"mk_{k}" for k in CLASS_KEYS] + ["att_all", "mk_all"])


def team_shot_form(events: pd.DataFrame, universe: pd.DataFrame,
                   first_chance_only: bool = False) -> pd.DataFrame:
    """As-of, league-centred shooting form for every (game, team): the team's
    own make rate per shot class, and the make rate it ALLOWS per shot class.

    Both are expanding means over the team's games strictly before this one,
    within season, so the team's own game contributes nothing to its own
    feature. The league's as-of rate on the same DATE travels with them, so the
    centring is against a snapshot mean rather than a season constant
    (`CLAUDE.md`: no raw levels).

    `first_chance_only` restricts the source to a possession's first chance.
    It is OFF by default here -- unlike the rebound model, whose targets are
    first-chance-safe by design -- because this model's target population is
    ALL attempts, and the L16 contamination channel that motivated the
    restriction there is closed at the source by the v2 rim override rather
    than by dropping rows. The trainer runs it both ways and reports what the
    restriction costs, so the default is a measured choice."""
    ev = events[events["chance_number"] == 1] if first_chance_only else events
    box = pd.DataFrame(_wide_counts(ev))
    box[["season", "game_id", "team_id", "opp_id"]] = ev[
        ["season", "game_id", "off_team_id", "def_team_id"]].to_numpy()
    box = box.groupby(["season", "game_id", "team_id", "opp_id"], as_index=False)[
        list(COUNT_COLS)].sum()

    dates = universe[["game_id", "game_date"]].copy()
    dates["game_date"] = pd.to_datetime(dates["game_date"])
    box = box.merge(dates, on="game_id", how="left")

    off = box.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    off_asof = PM.expanding_asof(off, ["season", "team_id"], list(COUNT_COLS))
    off_asof.columns = [f"off_{c}" for c in off_asof.columns]
    off = pd.concat([off[["season", "game_id", "team_id", "game_date"]], off_asof], axis=1)

    dfd = box.rename(columns={"team_id": "_off", "opp_id": "team_id"})
    dfd = dfd.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    def_asof = PM.expanding_asof(dfd, ["season", "team_id"], list(COUNT_COLS))
    def_asof.columns = [f"dal_{c}" for c in def_asof.columns]
    dfd = pd.concat([dfd[["season", "game_id", "team_id"]], def_asof], axis=1)

    form = off.merge(dfd, on=["season", "game_id", "team_id"], how="left")

    day = box.groupby(["season", "game_date"], as_index=False)[list(COUNT_COLS)].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PM.expanding_asof(day, ["season"], list(COUNT_COLS))
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)
    form = form.merge(day, on=["season", "game_date"], how="left")
    form = form.rename(columns={"off_n_prior": "n_prior_off", "dal_n_prior": "n_prior_def",
                                "lg_n_prior": "n_prior_lg"})
    return form


def shooter_form(events: pd.DataFrame, roster_dir: Path | str | None = None) -> pd.DataFrame:
    """As-of shooting form for every (season, shooter, game), wide over the
    three shot classes, plus the position as-of rate and the shooter's
    COMPLETED prior-season rates.

    Built exactly the way `cbb_sim.models.free_throw.build_ft_design` builds
    its shooter block, including the one trap that module documents: the
    per-game counts (`pg_raw`) are kept separate from the expanded ones,
    because expanding a column that is already an expanding sum would square
    the history."""
    a = events[np.isfinite(events["shooter_id"].to_numpy())].copy()
    a["shooter_id"] = a["shooter_id"].astype("int64")
    a["game_date"] = pd.to_datetime(a["game_date"])
    cnt = pd.DataFrame(_wide_counts(a), index=a.index)
    a = pd.concat([a[["season", "shooter_id", "game_id", "game_date", "off_team_id"]], cnt], axis=1)

    pg_raw = a.groupby(["season", "shooter_id", "game_id"], as_index=False).agg(
        {**{c: "sum" for c in COUNT_COLS}, "game_date": "first"})
    pg_raw = pg_raw.sort_values(["season", "shooter_id", "game_date", "game_id"],
                                kind="stable").reset_index(drop=True)
    asof = PM.expanding_asof(pg_raw, ["season", "shooter_id"], list(COUNT_COLS))
    pg = pd.concat([pg_raw[["season", "shooter_id", "game_id", "game_date"]], asof], axis=1)
    pg = pg.rename(columns={"n_prior": "shooter_games_asof"})

    pos = load_positions(roster_dir) if roster_dir is not None else load_positions()
    pg_raw = pg_raw.merge(pos, on="shooter_id", how="left")
    pg_raw["position_group"] = pg_raw["position_group"].fillna("UNK")
    pg = pg.merge(pg_raw[["season", "shooter_id", "game_id", "position_group"]],
                  on=["season", "shooter_id", "game_id"], how="left")

    posday = pg_raw.groupby(["season", "position_group", "game_date"], as_index=False)[
        list(COUNT_COLS)].sum()
    posday = posday.sort_values(["season", "position_group", "game_date"], kind="stable")
    pasof = PM.expanding_asof(posday, ["season", "position_group"], list(COUNT_COLS))
    pasof.columns = [f"pos_{c}" for c in pasof.columns]
    posday = pd.concat([posday[["season", "position_group", "game_date"]], pasof], axis=1)
    pg = pg.merge(posday, on=["season", "position_group", "game_date"], how="left")

    season_tot = pg_raw.groupby(["season", "shooter_id"], as_index=False)[list(COUNT_COLS)].sum()
    season_tot["season"] = season_tot["season"] + 1
    season_tot = season_tot.rename(columns={c: f"prev_{c}" for c in COUNT_COLS})
    pg = pg.merge(season_tot, on=["season", "shooter_id"], how="left")

    prev_team = a.groupby(["season", "shooter_id"])["off_team_id"].agg(
        lambda s: s.value_counts().index[0]).reset_index().rename(
        columns={"off_team_id": "prev_team_id"})
    prev_team["season"] = prev_team["season"] + 1
    pg = pg.merge(prev_team, on=["season", "shooter_id"], how="left")
    return pg


def league_asof_by_date(events: pd.DataFrame) -> pd.DataFrame:
    """League as-of make rate per class on each (season, date), from the same
    per-game counts every other as-of table is built from.

    The season's FIRST date has nothing strictly earlier, so its as-of rate is
    undefined and is back-filled from that season's own next available date --
    computed on the DATE-sorted table, because a back-fill along attempt order
    would mean nothing. This is the only forward-looking value in the module
    and it touches opening day alone (the same treatment, for the same reason,
    as `free_throw.build_ft_design`)."""
    ev = events.copy()
    ev["game_date"] = pd.to_datetime(ev["game_date"])
    cnt = pd.DataFrame(_wide_counts(ev), index=ev.index)
    ev = pd.concat([ev[["season", "game_date"]], cnt], axis=1)
    day = ev.groupby(["season", "game_date"], as_index=False)[list(COUNT_COLS)].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PM.expanding_asof(day, ["season"], list(COUNT_COLS))
    lg.columns = [f"lg_{c}" for c in lg.columns]
    out = pd.concat([day[["season", "game_date"]].reset_index(drop=True),
                     lg.reset_index(drop=True)], axis=1)
    for k in CLASS_KEYS:
        r = _rate(out[f"lg_mk_{k}"].to_numpy(), out[f"lg_att_{k}"].to_numpy())
        out[f"lg_rate_{k}"] = r
        out[f"lg_rate_{k}"] = out.groupby("season")[f"lg_rate_{k}"].bfill()
    return out


def _rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def _pick_class(d: pd.DataFrame, prefix: str, suffix: str = "") -> np.ndarray:
    """The value of a WIDE per-class column for each row's OWN class."""
    ck = d["class_key"].to_numpy()
    out = np.full(len(d), np.nan, dtype="float64")
    for k in CLASS_KEYS:
        m = ck == k
        if m.any():
            out[m] = d.loc[m, f"{prefix}{k}{suffix}"].to_numpy(dtype="float64")
    return out


# ===========================================================================
# 3. Defender (lineup) rates, 2024+ only
# ===========================================================================
def defender_rates(events: pd.DataFrame, prior_att: int = 250) -> pd.DataFrame:
    """As-of rim- and three-point make rate ALLOWED by each player while he was
    on the floor on DEFENCE, for every (season, player, game).

    Rim protection and perimeter defence are the two quantities the
    pre-registration names. The denominator is attempts of that class FACED
    while the player was one of the five defenders, so the aggregate over a
    lineup is a mean of five comparable rates rather than a sum of shares.

    `prior_att` shrinks each rate toward the league's own as-of rate with that
    many pseudo-attempts. L13 requires the strength be FITTED, not assumed;
    `scripts/train_fg_make_v1.py` fits it on the lineup fold's train season
    over `LINEUP_PRIOR_GRID`."""
    ev = events[np.isfinite(events[list(ES.ON_FLOOR_COLS)].to_numpy()).all(axis=1)]
    if not len(ev):
        return pd.DataFrame(columns=["season", "player_id", "game_id", "game_date",
                                     "rim_allow", "three_allow", "lg_rim_allow",
                                     "lg_three_allow"])
    off_home = ev["offense_is_home"].to_numpy()
    ck = ev["class_key"].to_numpy()
    made = ev["made"].to_numpy()
    att_rim = (ck == "rim").astype("int32")
    mk_rim = (att_rim.astype(bool) & made).astype("int32")
    att_three = (ck == "three").astype("int32")
    mk_three = (att_three.astype(bool) & made).astype("int32")

    rows = []
    for slot, col in enumerate(ES.ON_FLOOR_COLS):
        player_is_home = slot < 5
        on_defense = player_is_home != off_home
        if not on_defense.any():
            continue
        rows.append(pd.DataFrame({
            "season": ev["season"].to_numpy()[on_defense],
            "game_id": ev["game_id"].to_numpy()[on_defense],
            "player_id": ev[col].to_numpy()[on_defense].astype("int64"),
            "att_rim": att_rim[on_defense], "mk_rim": mk_rim[on_defense],
            "att_three": att_three[on_defense], "mk_three": mk_three[on_defense],
        }))
    long = pd.concat(rows, ignore_index=True)
    del rows
    cols = ["att_rim", "mk_rim", "att_three", "mk_three"]
    pg = long.groupby(["season", "player_id", "game_id"], as_index=False)[cols].sum()
    del long
    dates = ev[["game_id", "game_date"]].drop_duplicates()
    pg = pg.merge(dates, on="game_id", how="left")
    pg["game_date"] = pd.to_datetime(pg["game_date"])
    pg = pg.sort_values(["season", "player_id", "game_date", "game_id"],
                        kind="stable").reset_index(drop=True)
    asof = PM.expanding_asof(pg, ["season", "player_id"], cols)
    pg = pd.concat([pg[["season", "player_id", "game_id", "game_date"]], asof], axis=1)

    day = pg.groupby(["season", "game_date"], as_index=False)[cols].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PM.expanding_asof(day, ["season"], cols)
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]].reset_index(drop=True),
                     lg.reset_index(drop=True)], axis=1)
    pg = pg.merge(day, on=["season", "game_date"], how="left")

    k = float(prior_att)
    out = pg[["season", "player_id", "game_id", "game_date"]].copy()
    for name, a, m in (("rim", "att_rim", "mk_rim"), ("three", "att_three", "mk_three")):
        lg_rate = _rate(pg[f"lg_{m}"].to_numpy(), pg[f"lg_{a}"].to_numpy())
        lg_rate = np.where(np.isfinite(lg_rate), lg_rate, np.nan)
        # a season's opening day has no strictly-earlier league rate; it is
        # back-filled from the season's own next date, never invented
        lg_s = pd.Series(lg_rate).groupby(pg["season"].to_numpy()).bfill()
        lg_rate = lg_s.to_numpy()
        den = pg[a].to_numpy() + k
        val = np.where(den > 0, (pg[m].to_numpy() + k * lg_rate) / np.maximum(den, 1e-9), lg_rate)
        out[f"{name}_allow"] = np.where(np.isfinite(val), val, lg_rate).astype("float32")
        out[f"lg_{name}_allow"] = lg_rate.astype("float32")
    out["prior_att"] = int(prior_att)
    return out


def attach_lineup_features(design: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Mean of the DEFENSIVE five's as-of allowed rates, centred on the
    league's own as-of rate, plus the `lineup_on_floor_ok` mask.

    Both bundles are scored on the masked rows only, so C and D are compared
    on identical rows (the rebound model's construction)."""
    d = design.copy()
    key = rates.set_index(["season", "player_id", "game_id"])
    off_home = d["offense_is_home"].to_numpy()
    n = len(d)
    rim_sum = np.zeros(n)
    three_sum = np.zeros(n)
    cnt = np.zeros(n)
    ok = np.ones(n, dtype=bool)
    lg_rim = np.full(n, np.nan)
    lg_three = np.full(n, np.nan)
    for slot, col in enumerate(ES.ON_FLOOR_COLS):
        player_is_home = slot < 5
        on_defense = player_is_home != off_home
        pid = pd.to_numeric(d[col], errors="coerce").to_numpy()
        pid_i = np.where(np.isfinite(pid), pid, -1).astype("int64")
        idx = pd.MultiIndex.from_arrays(
            [d["season"].to_numpy(), pid_i, d["game_id"].to_numpy()])
        r = key.reindex(idx)
        rr = r["rim_allow"].to_numpy()
        tt = r["three_allow"].to_numpy()
        present = np.isfinite(rr) & np.isfinite(tt)
        ok &= present | ~on_defense
        take = on_defense & present
        rim_sum += np.where(take, np.nan_to_num(rr), 0.0)
        three_sum += np.where(take, np.nan_to_num(tt), 0.0)
        cnt += take.astype("float64")
        lg_rim = np.where(np.isfinite(r["lg_rim_allow"].to_numpy()),
                          r["lg_rim_allow"].to_numpy(), lg_rim)
        lg_three = np.where(np.isfinite(r["lg_three_allow"].to_numpy()),
                            r["lg_three_allow"].to_numpy(), lg_three)
    ok &= cnt == 5
    lg_rim = np.nan_to_num(lg_rim)
    lg_three = np.nan_to_num(lg_three)
    with np.errstate(invalid="ignore", divide="ignore"):
        d["def5_rim_allow_c"] = np.where(
            ok, rim_sum / np.maximum(cnt, 1e-9) - lg_rim, 0.0).astype("float32")
        d["def5_three_allow_c"] = np.where(
            ok, three_sum / np.maximum(cnt, 1e-9) - lg_three, 0.0).astype("float32")
    d["lineup_on_floor_ok"] = ok
    return d


# ===========================================================================
# 4. The design matrix and the pre-registered bundles
# ===========================================================================
TEAM_FEATURES: tuple[str, ...] = (
    "off_make_c",            # offence's as-of make rate on THIS class, league-centred
    "def_allow_c",           # defence's as-of make rate ALLOWED on this class, centred
    "off_rating_off_c", "off_rating_def_c",
    "def_rating_off_c", "def_rating_def_c",
    "site_home", "site_away",           # neutral is the reference level
    "season_idx",
)
SHOOTER_FEATURES: tuple[str, ...] = (
    "shooter_make_c",        # shooter's as-of make rate on this class, league-centred
    "shooter_att_c",         # attempts to date on this class
    "prior_season_make_c",   # completed prior season on this class, centred
    "has_prior_season",
    "pos_G", "pos_F", "pos_C",          # UNK is the reference level
    "shooter_games_asof", "shooter_fga_asof",   # the minutes-to-date stand-in
)
STATE_FEATURES: tuple[str, ...] = (
    "period", "seconds_remaining", "score_diff", "in_bonus",
    "chance_number", "chance_elapsed_s", "is_transition_f",
)
LINEUP_FEATURES: tuple[str, ...] = ("def5_rim_allow_c", "def5_three_allow_c")

FEATURE_SETS: tuple[str, ...] = ("A_team", "B_plus_shooter", "C_plus_state", "D_plus_lineup")


def feature_set(name: str) -> list[str]:
    a = list(TEAM_FEATURES)
    if name == "A_team":
        return a
    b = a + list(SHOOTER_FEATURES)
    if name == "B_plus_shooter":
        return b
    c = b + list(STATE_FEATURES)
    if name == "C_plus_state":
        return c
    if name == "D_plus_lineup":
        return c + list(LINEUP_FEATURES)
    if name in R2_FEATURE_SETS:
        return list(R2_FEATURE_SETS[name])
    raise KeyError(f"unknown feature set {name!r}")


# ===========================================================================
# ROUND 2 (2026-09-10): state parametrisations
# ---------------------------------------------------------------------------
# Pre-registration: `docs/models/fg_make/experiments.md` section 13.
# Evidence:         `docs/tests/fg_make_state_confound_2026-09-10.md`.
#
# THE DEFECT THIS EXISTS TO REPAIR. `_season_events` reads `score_diff` off the
# feed's `homeScore`/`awayScore` on the attempt's OWN row, and that column is
# the score AFTER the play: a made three already carries its own three points.
# The feature is therefore POST-OUTCOME, in the same family as `blocked` and
# `and_one`, and it manufactures 62-84% of the apparent margin effect. It is
# NOT added to `BANNED_FEATURES` because arm S-A has to reproduce round 1's
# numbers exactly; the round-2 decision and the change ledger carry its status.
# ===========================================================================

#: The corrected margin: the score difference BEFORE the attempt. Removing the
#: attempt's own points can only take outcome information OUT (a miss is
#: unchanged by construction), which is what makes the correction itself safe.
R2_MARGIN_COL = "score_diff_pre"

#: Thresholds FIXED IN THE PRE-REGISTRATION from the step-1 evidence grids, and
#: never re-tuned afterwards. Both effects are flat across their grids, which is
#: why an indicator is the right parametrisation and why the threshold is a
#: reading rather than a fitted parameter.
R2_GT_MARGIN = 15.0          # garbage time: |margin| at or beyond this
R2_GT_SECONDS = 480.0        # ... with this many seconds or fewer left in regulation
R2_EG_SECONDS = 120.0        # end game: seconds left in regulation
R2_EG_LO, R2_EG_HI = 1.0, 9.0    # ... trailing / leading by 1 to 9 (one to three possessions)

R2_SAFE_STATE: tuple[str, ...] = (
    "period", "seconds_remaining", "in_bonus",
    "chance_number", "chance_elapsed_s", "is_transition_f",
)
R2_INDICATORS: tuple[str, ...] = ("gt_flag", "eg_trail", "eg_lead")

R2_ARMS: tuple[str, ...] = ("S_A", "S_B", "S_C", "S_D", "S_E")
R2_ARM_FEATURE_SET: dict[str, str] = {
    "S_A": "R2_A_round1_leaked",
    "S_B": "R2_B_no_state",
    "S_C": "R2_C_safe_state",
    "S_D": "R2_D_safe_plus_indicators",
    "S_E": "R2_E_safe_plus_continuous",
}
#: Tie-break order of the pre-registration: simpler wins inside the floor.
R2_ARM_SIMPLICITY: dict[str, int] = {"S_B": 0, "S_C": 1, "S_D": 2, "S_E": 3, "S_A": 4}
#: S-A is declared ineligible in the pre-registration, BEFORE the round ran,
#: on the data-integrity ground above and not on any number it produced.
R2_INELIGIBLE: frozenset[str] = frozenset({"S_A"})

_R2_B = list(TEAM_FEATURES) + list(SHOOTER_FEATURES)
R2_FEATURE_SETS: dict[str, list[str]] = {
    "R2_A_round1_leaked": _R2_B + list(STATE_FEATURES),
    "R2_B_no_state": list(_R2_B),
    "R2_C_safe_state": _R2_B + list(R2_SAFE_STATE),
    "R2_D_safe_plus_indicators": _R2_B + list(R2_SAFE_STATE) + list(R2_INDICATORS),
    "R2_E_safe_plus_continuous": _R2_B + list(R2_SAFE_STATE) + [R2_MARGIN_COL],
}


def regulation_seconds_remaining(period: np.ndarray, sec: np.ndarray) -> np.ndarray:
    """Seconds left in REGULATION. `seconds_remaining` is per period in this
    feed (1200 in a half, 300 in overtime), so period 1 carries the second half
    with it. Overtime rows keep their own period clock and are excluded from
    every round-2 indicator by the `period <= 2` guard."""
    per = np.asarray(period, dtype="float64")
    s = np.asarray(sec, dtype="float64")
    return np.where(per <= 1.0, s + 1200.0, s)


def add_round2_state(d: pd.DataFrame) -> pd.DataFrame:
    """Add `score_diff_pre` and the three round-2 indicators, in place.

    The identical arithmetic lives in `cbb_sim.engine.loop._state_block` for the
    simulated side, where the margin is already pre-shot; the two definitions
    are pinned against each other by `tests/test_engine.py`."""
    pts = np.where(d["shot_class"].to_numpy() == "FGA_3", 3.0, 2.0)
    own = np.where(d["made"].to_numpy().astype(bool), pts, 0.0)
    sd = d["score_diff"].to_numpy(dtype="float64") - own
    d[R2_MARGIN_COL] = sd.astype("float32")
    per = d["period"].to_numpy(dtype="float64")
    gsr = regulation_seconds_remaining(per, d["seconds_remaining"].to_numpy())
    reg = per <= 2.0
    d["gt_flag"] = (reg & (np.abs(sd) >= R2_GT_MARGIN)
                    & (gsr <= R2_GT_SECONDS)).astype("float32")
    d["eg_trail"] = (reg & (sd <= -R2_EG_LO) & (sd >= -R2_EG_HI)
                     & (gsr <= R2_EG_SECONDS)).astype("float32")
    d["eg_lead"] = (reg & (sd >= R2_EG_LO) & (sd <= R2_EG_HI)
                    & (gsr <= R2_EG_SECONDS)).astype("float32")
    return d


def build_design(
    seasons: list[int],
    universe: pd.DataFrame | None = None,
    version: str | None = DEFAULT_VERSION,
    poss_dir: Path | str | None = None,
    ratings_dir: Path | str = DEFAULT_RATINGS_DIR,
    universe_path: Path | str = DEFAULT_UNIVERSE,
    pbp_dir: Path | str = ES.DEFAULT_PBP_DIR,
    require_pbp_complete: bool = True,
    first_chance_form: bool = False,
    events: pd.DataFrame | None = None,
    roster_dir: Path | str | None = None,
    crosswalk_path: Path | str | None = None,
    with_ratings: bool = True,
) -> pd.DataFrame:
    """One row per modelled field-goal attempt with every candidate feature.

    Attempts with no shooter id on the row are DROPPED (the shooter block is
    undefined for them); the count is left on `df.attrs['n_no_shooter']` so the
    trainer reports the share rather than it disappearing silently."""
    seasons = [int(s) for s in seasons]
    if universe is None:
        universe = ES.load_universe(universe_path, require_pbp_complete=require_pbp_complete)
    if events is None:
        events = build_fg_events(seasons, universe=universe, version=version,
                                 poss_dir=poss_dir, pbp_dir=pbp_dir)
    events = events[events["season"].isin(seasons)]

    n_all = len(events)
    d = events[np.isfinite(events["shooter_id"].to_numpy())].copy()
    n_no_shooter = n_all - len(d)
    d["shooter_id"] = d["shooter_id"].astype("int64")
    d["game_date"] = pd.to_datetime(d["game_date"])
    d["y"] = d["made"].astype("int8")

    form = team_shot_form(events, universe, first_chance_only=first_chance_form)
    off_cols = ["season", "game_id", "team_id", "n_prior_off"] + [f"off_{c}" for c in COUNT_COLS]
    off = form[off_cols].rename(columns={"team_id": "off_team_id"})
    d = d.merge(off, on=["season", "game_id", "off_team_id"], how="left")
    def_cols = ["season", "game_id", "team_id", "n_prior_def"] + [f"dal_{c}" for c in COUNT_COLS]
    dfn = form[def_cols].rename(columns={"team_id": "def_team_id"})
    d = d.merge(dfn, on=["season", "game_id", "def_team_id"], how="left")

    lgd = league_asof_by_date(events)
    d = d.merge(lgd[["season", "game_date"] + [f"lg_rate_{k}" for k in CLASS_KEYS]
                    + [f"lg_att_{k}" for k in CLASS_KEYS]],
                on=["season", "game_date"], how="left")

    sf = shooter_form(events, roster_dir=roster_dir)
    keep = (["season", "shooter_id", "game_id", "shooter_games_asof", "position_group",
             "prev_team_id"]
            + list(COUNT_COLS) + [f"pos_{c}" for c in COUNT_COLS]
            + [f"prev_{c}" for c in COUNT_COLS])
    d = d.merge(sf[keep], on=["season", "shooter_id", "game_id"], how="left")

    # ---- per-row, per-class rates ----------------------------------------
    lg_rate = _pick_class(d, "lg_rate_")
    lg_rate = np.where(np.isfinite(lg_rate), lg_rate, np.nan)
    d["lg_make_asof"] = lg_rate.astype("float32")

    # Every centred rate below is built as `where(the rate EXISTS, rate - league,
    # 0.0)`. The zero branch is not a convenience: "no prior attempts" means the
    # feature's value is the league mean, which on a centred scale is EXACTLY
    # 0.0 (the rebound model's convention). Subtracting instead of branching
    # would leave a float32 rounding residue of ~1e-8 in a column that is
    # supposed to be constant, and a standardiser would then divide by that
    # residue and blow the linear arms up -- which is exactly what it did on the
    # first smoke run of this trainer.
    off_att = _pick_class(d, "off_att_")
    off_mk = _pick_class(d, "off_mk_")
    off_rate = _rate(off_mk, off_att)
    off_ok = np.isfinite(off_rate)
    d["off_make_raw"] = np.where(off_ok, off_rate, lg_rate).astype("float32")
    d["off_make_c"] = np.where(off_ok, off_rate - lg_rate, 0.0).astype("float32")
    d["off_att_prior"] = np.nan_to_num(off_att).astype("float32")

    dal_att = _pick_class(d, "dal_att_")
    dal_mk = _pick_class(d, "dal_mk_")
    dal_rate = _rate(dal_mk, dal_att)
    dal_ok = np.isfinite(dal_rate)
    d["def_allow_raw"] = np.where(dal_ok, dal_rate, lg_rate).astype("float32")
    d["def_allow_c"] = np.where(dal_ok, dal_rate - lg_rate, 0.0).astype("float32")
    d["def_att_prior"] = np.nan_to_num(dal_att).astype("float32")

    sh_att = _pick_class(d, "att_")
    sh_mk = _pick_class(d, "mk_")
    sh_rate = _rate(sh_mk, sh_att)
    sh_ok = np.isfinite(sh_rate)
    d["shooter_make_raw"] = np.where(sh_ok, sh_rate, lg_rate).astype("float32")
    d["shooter_make_c"] = np.where(sh_ok, sh_rate - lg_rate, 0.0).astype("float32")
    d["shooter_att_c"] = np.nan_to_num(sh_att).astype("float32")
    d["shooter_mk_c"] = np.nan_to_num(sh_mk).astype("float32")
    d["shooter_games_asof"] = d["shooter_games_asof"].fillna(0).astype("float32")
    d["shooter_fga_asof"] = np.nan_to_num(d["att_all"].to_numpy()).astype("float32")

    pos_att = _pick_class(d, "pos_att_")
    pos_mk = _pick_class(d, "pos_mk_")
    pos_rate = _rate(pos_mk, pos_att)
    d["position_make_raw"] = np.where(np.isfinite(pos_rate), pos_rate, lg_rate).astype("float32")

    prev_att = _pick_class(d, "prev_att_")
    prev_mk = _pick_class(d, "prev_mk_")
    prev_rate = _rate(prev_mk, prev_att)
    prev_ok = np.isfinite(prev_rate)
    d["has_prior_season"] = prev_ok.astype("float32")
    d["prior_season_make_raw"] = np.where(prev_ok, prev_rate, lg_rate).astype("float32")
    d["prior_season_make_c"] = np.where(prev_ok, prev_rate - lg_rate, 0.0).astype("float32")
    d["prior_season_att_c"] = np.nan_to_num(prev_att).astype("float32")

    pg = d["position_group"].fillna("UNK").to_numpy()
    for g in ("G", "F", "C"):
        d[f"pos_{g}"] = (pg == g).astype("float32")
    d["position_group"] = pg

    prev_team_id = pd.to_numeric(d["prev_team_id"], errors="coerce").to_numpy()
    d["is_transfer"] = (np.isfinite(prev_team_id)
                        & (prev_team_id != d["off_team_id"].to_numpy())
                        & (d["has_prior_season"].to_numpy() > 0))

    # ---- ratings, site, state --------------------------------------------
    if with_ratings:
        ratings = orat.load_ratings(sorted(set(seasons)), out_dir=ratings_dir)
        d = orat.join_as_of(d, ratings, team_col="off_team_id", date_col="game_date",
                            suffix="__offteam", cols=("off_c", "def_c"))
        d = orat.join_as_of(d, ratings, team_col="def_team_id", date_col="game_date",
                            suffix="__defteam", cols=("off_c", "def_c"))
        d = d.rename(columns={
            "off_c__offteam": "off_rating_off_c", "def_c__offteam": "off_rating_def_c",
            "off_c__defteam": "def_rating_off_c", "def_c__defteam": "def_rating_def_c"})
    else:
        for c in ("off_rating_off_c", "off_rating_def_c",
                  "def_rating_off_c", "def_rating_def_c"):
            d[c] = 0.0

    neutral = d["neutral_site"].to_numpy().astype(bool)
    off_home = d["offense_is_home"].to_numpy().astype(bool)
    d["site_home"] = ((~neutral) & off_home).astype("float32")
    d["site_away"] = ((~neutral) & (~off_home)).astype("float32")

    d["season_idx"] = (d["season"] - 2022).astype("float32")
    d["period"] = d["period"].astype("float32")
    d["seconds_remaining"] = d["seconds_remaining"].astype("float32")
    d["score_diff"] = d["score_diff"].astype("float32")
    d["in_bonus"] = d["off_in_bonus"].astype("float32")
    d["chance_number"] = d["chance_number"].astype("float32")
    d["is_transition_f"] = d["is_transition"].astype("float32")

    for c in ("off_make_c", "def_allow_c", "shooter_make_c", "prior_season_make_c",
              "off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c"):
        d[c] = d[c].astype("float32").fillna(0.0)
    for c in ("lg_make_asof", "off_make_raw", "def_allow_raw", "shooter_make_raw",
              "position_make_raw", "prior_season_make_raw"):
        d[c] = d[c].astype("float32").fillna(d[c].astype("float32").median())

    cw = load_espn_ids(crosswalk_path) if crosswalk_path is not None else load_espn_ids()
    if len(cw):
        d = d.merge(cw, on=["season", "shooter_id"], how="left")
    else:
        d["espn_athlete_id"] = np.nan

    # The WIDE per-class working columns have done their job; drop them by
    # EXACT name rather than by prefix, because a prefix rule here would also
    # eat `off_att_prior` / `def_att_prior`, which are reported quantities.
    prefixes = ("off_att_", "off_mk_", "dal_att_", "dal_mk_", "att_", "mk_",
                "pos_att_", "pos_mk_", "prev_att_", "prev_mk_", "lg_rate_", "lg_att_")
    drop = [f"{p}{k}" for p in prefixes for k in (*CLASS_KEYS, "all")]
    d = d.drop(columns=[c for c in drop if c in d.columns])

    d.attrs["n_attempts"] = int(n_all)
    d.attrs["n_no_shooter"] = int(n_no_shooter)
    d.attrs["possessions_version"] = version or DEFAULT_VERSION
    d.attrs["rim_override_max_ft"] = float(events.attrs.get("rim_override_max_ft", np.nan))
    d.attrs["first_chance_form"] = bool(first_chance_form)
    return d.reset_index(drop=True)


def fold_slices(design: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) slices for a fold, with the seal guard on BOTH."""
    spec = FOLDS[fold] if fold in FOLDS else LINEUP_FOLDS[fold]
    assert_not_sealed(spec["train"], context=f"{fold} train seasons")
    assert_not_sealed(spec["test"], context=f"{fold} test seasons")
    tr = design[design["season"].isin(spec["train"])]
    te = design[design["season"].isin(spec["test"])]
    assert_not_sealed(tr, context=f"{fold} train slice")
    assert_not_sealed(te, context=f"{fold} test slice")
    return tr, te


def class_slice(design: pd.DataFrame, shot_class: str) -> pd.DataFrame:
    if shot_class not in SHOT_CLASSES:
        raise KeyError(f"unknown shot class {shot_class!r}")
    return design[design["shot_class"].to_numpy() == shot_class]


# ===========================================================================
# 5. Arms
# ===========================================================================
ARMS: tuple[str, ...] = ("team_baseline", "eb_shrink", "ridge", "lgbm")
#: The pre-registered simplicity order for the tie-break.
ARM_SIMPLICITY = {"team_baseline": 0, "eb_shrink": 1, "ridge": 2, "lgbm": 3}
TREE_ARMS: frozenset[str] = frozenset({"lgbm"})
#: The feature bundle each arm is scored on, per the pre-registration
#: ("team-level baseline (A only)"; "logistic ridge on the full bundle";
#: "LightGBM on the full bundle").
ARM_FEATURE_SET = {"team_baseline": "A_team", "eb_shrink": None,
                   "ridge": "C_plus_state", "lgbm": "C_plus_state"}

#: The three pre-registered priors for the shooter's own class rate.
PRIOR_KINDS: tuple[str, ...] = ("league", "position", "prior_season")
#: Shrinkage strength grid for the SHOOTER, in attempts.
SHRINK_GRID: tuple[float, ...] = (5, 10, 20, 30, 50, 75, 100, 150, 250, 400)
#: Shrinkage strength grid for the DEFENCE's allowed rate. It is fitted for the
#: same reason the shooter's is: a defence's as-of allowed rate is built on a
#: handful of attempts in November and on thousands in March, and the logit
#: combination is only stable if the early rows are pulled toward the league.
DEF_SHRINK_GRID: tuple[float, ...] = (0, 100, 300, 1000)

EPS_P = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS_P, 1.0 - EPS_P)
    return np.log(p / (1.0 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _prior_rate(d: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "league":
        return d["lg_make_asof"].to_numpy(dtype="float64")
    if kind == "position":
        return d["position_make_raw"].to_numpy(dtype="float64")
    if kind == "prior_season":
        # returning players use their own completed prior season; newcomers
        # have none and fall back to the league as-of rate, which is what "for
        # returning players" means and is not a hidden imputation
        return d["prior_season_make_raw"].to_numpy(dtype="float64")
    raise KeyError(f"unknown prior kind {kind!r}")


def eb_predict(d: pd.DataFrame, kind: str, m: float, m_def: float = 0.0) -> np.ndarray:
    """Empirical-Bayes shooter rate combined with the defence's allowed rate
    MULTIPLICATIVELY ON THE LOGIT SCALE, as the pre-registration specifies.

        logit(p) = logit(shooter_EB) + [logit(def_allowed_EB) - logit(league)]

    The bracket is the defence's log-odds deviation from the league on this
    shot class, so a league-average defence contributes exactly nothing and the
    arm degrades to the shooter's own shrunk rate -- which is the property that
    makes it a genuine matchup model rather than two blended levels."""
    prior = _prior_rate(d, kind)
    made = d["shooter_mk_c"].to_numpy(dtype="float64")
    att = d["shooter_att_c"].to_numpy(dtype="float64")
    p_sh = (m * prior + made) / (m + att)

    lg = d["lg_make_asof"].to_numpy(dtype="float64")
    d_mk = d["def_allow_raw"].to_numpy(dtype="float64") * d["def_att_prior"].to_numpy(dtype="float64")
    d_att = d["def_att_prior"].to_numpy(dtype="float64")
    p_def = (m_def * lg + d_mk) / np.maximum(m_def + d_att, 1e-9)
    p_def = np.where(m_def + d_att > 0, p_def, lg)

    z = _logit(p_sh) + (_logit(p_def) - _logit(lg))
    p = _expit(z)
    p = np.clip(p, EPS_P, 1.0 - EPS_P)
    return np.column_stack([1.0 - p, p])


def fit_eb(tr: pd.DataFrame, grid: tuple[float, ...] = SHRINK_GRID,
           kinds: tuple[str, ...] = PRIOR_KINDS,
           def_grid: tuple[float, ...] = DEF_SHRINK_GRID) -> dict:
    """Grid over (prior, shooter strength, defence strength) on the TRAINING
    fold.

    The selection metric is the training fold's own log loss, which is honest
    here because every quantity the arm uses is an as-of feature: a shooter's
    shrunk rate on 12 January is built from games strictly before 12 January,
    so the training log loss is already a walk-forward number."""
    y = tr["y"].to_numpy()
    rows, best = [], None
    for kind in kinds:
        for m in grid:
            for md in def_grid:
                p = eb_predict(tr, kind, float(m), float(md))
                ll = PM.log_loss(y, p)
                rows.append({"prior": kind, "m": float(m), "m_def": float(md),
                             "train_log_loss": round(ll, 6)})
                if best is None or ll < best["train_log_loss"]:
                    best = rows[-1]
    return {"grid": rows, "best": best}


class RidgeArm:
    """Binary logistic ridge. Features are standardised on the TRAIN slice only
    and the fitted means/SDs travel with the model, so the sim applies the
    identical transform."""

    def __init__(self, C: float = 1.0, max_iter: int = 300, seed: int = 0):
        self.C, self.max_iter, self.seed = C, max_iter, seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeArm:
        from sklearn.linear_model import LogisticRegression

        self.mu_ = X.mean(axis=0)
        sd = X.std(axis=0)
        # A column is treated as CONSTANT on a RELATIVE threshold, not an
        # absolute one. A feature that is structurally constant on a fold (a
        # prior-season rate in the first season of the data) can still carry a
        # float32 rounding residue of ~1e-8; dividing by that residue turns a
        # zero-information column into a 1e7-sized input and the fit diverges.
        # The threshold scales with the column's own level so a genuinely
        # small-scale feature is not swallowed.
        self.sd_ = np.where(sd < 1e-6 * np.maximum(np.abs(self.mu_), 1.0), 1.0, sd)
        self.clf_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs",
                                       random_state=self.seed).fit((X - self.mu_) / self.sd_, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _align(self.clf_.predict_proba((X - self.mu_) / self.sd_), self.clf_.classes_)


#: The LightGBM parameter ladder. The pre-registration allows a search on F1
#: ONLY; the winner per class is frozen before F2 is touched.
LGBM_BASE = dict(objective="binary", n_estimators=400, learning_rate=0.06, num_leaves=63,
                 min_child_samples=400, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.9, reg_lambda=1.0, verbose=-1)
LGBM_LADDER: tuple[dict, ...] = (
    {},
    {"num_leaves": 31, "min_child_samples": 200},
    {"num_leaves": 127, "min_child_samples": 800},
    {"n_estimators": 800, "learning_rate": 0.03},
    {"num_leaves": 15, "min_child_samples": 100, "n_estimators": 800, "learning_rate": 0.03},
    {"reg_lambda": 20.0, "num_leaves": 63, "min_child_samples": 800},
)


class LgbmArm:
    def __init__(self, seed: int = 0, params: dict | None = None):
        self.seed = seed
        self.params = {**LGBM_BASE, **(params or {})}

    def fit(self, X: np.ndarray, y: np.ndarray) -> LgbmArm:
        import lightgbm as lgb

        self.clf_ = lgb.LGBMClassifier(random_state=self.seed, n_jobs=-1, **self.params)
        self.clf_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _align(self.clf_.predict_proba(X), self.clf_.classes_)


def _align(p: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((p.shape[0], len(CLASSES)), dtype="float64")
    for j, c in enumerate(classes):
        out[:, int(c)] = p[:, j]
    s = out.sum(axis=1, keepdims=True)
    return out / np.maximum(s, 1e-12)


def design_matrix(d: pd.DataFrame, features: list[str] | tuple[str, ...]) -> np.ndarray:
    """The feature matrix in the DECLARED feature order.

    Public because the sim calls it, and because the banned-feature guard has
    to live where the matrix is actually built rather than in prose."""
    feats = list(features)
    bad = [f for f in feats if f in BANNED_FEATURES]
    if bad:
        raise ValueError(f"post-outcome features cannot enter the design matrix: {bad}")
    return np.ascontiguousarray(d[feats].to_numpy(dtype="float32"))


def fit_arm(arm: str, tr: pd.DataFrame, feature_set_name: str | None = None,
            seed: int = 0, eb_best: dict | None = None, params: dict | None = None):
    """Fit ONE arm on ONE shot class's rows. `tr` must already be a single-class
    slice; `fit_by_class` is the entry point that guarantees it."""
    fs = feature_set_name or ARM_FEATURE_SET[arm]
    y = tr["y"].to_numpy()
    if arm == "eb_shrink":
        return {"arm": arm, "eb": eb_best}
    X = design_matrix(tr, feature_set(fs))
    if arm in ("ridge", "team_baseline"):
        return RidgeArm(seed=seed).fit(X, y)
    if arm == "lgbm":
        return LgbmArm(seed=seed, params=params).fit(X, y)
    raise KeyError(f"unknown arm {arm!r}")


def predict_arm(arm: str, model, te: pd.DataFrame,
                feature_set_name: str | None = None) -> np.ndarray:
    fs = feature_set_name or ARM_FEATURE_SET[arm]
    if arm == "eb_shrink":
        b = model["eb"]
        return eb_predict(te, b["prior"], float(b["m"]), float(b.get("m_def", 0.0)))
    return model.predict_proba(design_matrix(te, feature_set(fs)))


def fit_by_class(arm: str, train: pd.DataFrame, feature_set_name: str | None = None,
                 seed: int = 0, eb_by_class: dict | None = None,
                 params_by_class: dict | None = None) -> dict:
    """One estimator per shot class, each fitted on that class's rows ALONE.

    This is the function that makes "the three classes never share a fit" a
    property of the code rather than of the caller's discipline, and it is what
    `tests/test_fg_make.py` binds its cross-class isolation proof to."""
    out = {}
    for c in SHOT_CLASSES:
        tr = class_slice(train, c)
        out[c] = fit_arm(arm, tr, feature_set_name, seed=seed,
                         eb_best=(eb_by_class or {}).get(c),
                         params=(params_by_class or {}).get(c))
    return out


# ===========================================================================
# 6. Metrics wired to this model's vocabulary
# ===========================================================================
#: The two pre-registered responsiveness drivers. The pre-registered gate was
#: "BOTH monotone in 4 of 4 quintile steps" ("the matchup-specific rule applies
#: to defense too"); `ARCHITECTURE_DECISIONS.md` **Decision 8** supersedes it
#: with the slope-and-steps gate below. `score` reports BOTH verdicts -- the
#: superseded one under `resp_pass` so the original run's tables stay readable,
#: and the live one under `resp_pass_decision8`, which is what the trainer
#: decides on.
RESPONSIVENESS_SPECS: tuple[tuple[str, str], ...] = (
    ("shooter_make_c", "shooter_att_c"),
    ("def_allow_c", "def_att_prior"),
)
RESPONSIVENESS_MIN_STEPS = 4


def _rank(v: np.ndarray) -> np.ndarray:
    """Rank transform, ties broken by position.

    The driver quintiles are taken on the RANK, not on the raw value, because
    both drivers carry a tie mass (a shooter with no prior attempts of this
    class has no as-of rate at all) and a repeated quantile edge collapses a
    quintile into an empty bin, which `quintile_responsiveness` would report as
    a NaN step and the gate would read as a failure that is really a tie. The
    transform is monotone, so every reported share, span and slope ratio is
    exactly what it would have been on the raw driver."""
    order = np.argsort(v, kind="stable")
    out = np.empty(len(v), dtype="float64")
    out[order] = np.arange(len(v), dtype="float64")
    return out


#: **Decision 8** (`ARCHITECTURE_DECISIONS.md`, 2026-09-10) supersedes the
#: steps-only responsiveness wording of this model's pre-registration: the gate
#: is (a) slope ratio within [0.8, 1.2] AND (b) monotone in at least 3 of 4
#: quintile steps, with the 4-of-4 requirement dropped when the driver's
#: realised quintile span is below 2 pp ("the steps are then noise"), on every
#: driver.
#:
#: THE ONE AMBIGUITY, AND WHY BOTH READINGS ARE COMPUTED. Clause (b) exempts a
#: low-span driver from the step count because its steps are noise. Clause (a)
#: does not say whether the SLOPE on such a driver is exempt too. It matters
#: exactly once: `FGA_3`'s defence driver has a realised span of 1.37 pp and
#: LightGBM's slope on it is 0.474, so
#:   * `strict`: the slope band applies to every driver -> no arm passes FGA_3;
#:   * `low_span_exempt`: a sub-2 pp driver is noise for BOTH clauses -> FGA_3
#:     goes to LightGBM, which is the outcome Decision 8 itself states.
#: `decision8_verdict` takes the reading as an argument so neither is hidden,
#: and `scripts/train_fg_make_v1.py --redecide` reports both.
SLOPE_BAND: tuple[float, float] = (0.8, 1.2)
LOW_SPAN_PP: float = 2.0
MIN_STEPS_DEFAULT: int = 4
MIN_STEPS_LOW_SPAN: int = 3
DECISION8_READINGS: tuple[str, ...] = ("strict", "low_span_exempt")
#: The reading Decision 8's own stated outcome implies, and therefore the one
#: `score` and the trainer's decision step use. Changing it changes which arms
#: pass on a sub-2 pp driver and nothing else; both readings are always reported
#: side by side by `scripts/train_fg_make_v1.py --redecide`.
DECISION8_ADOPTED_READING = "low_span_exempt"


def decision8_verdict(resp: dict[str, dict], reading: str = "low_span_exempt",
                      slope_band: tuple[float, float] = SLOPE_BAND,
                      low_span_pp: float = LOW_SPAN_PP) -> dict:
    """Decision 8's responsiveness gate, per driver, for one arm.

    Returns the verdict and the per-driver working, so a PASS/FAIL can always
    be traced to the driver and the clause that produced it."""
    if reading not in DECISION8_READINGS:
        raise KeyError(f"unknown reading {reading!r}")
    lo, hi = slope_band
    out: dict = {"reading": reading, "slope_band": [lo, hi],
                 "low_span_pp": low_span_pp, "by_driver": {}}
    ok = True
    for name, r in resp.items():
        span_pp = abs(float(r["span_actual"])) * 100.0
        low_span = span_pp < low_span_pp
        min_steps = MIN_STEPS_LOW_SPAN if low_span else MIN_STEPS_DEFAULT
        steps_ok = int(r["pred_monotone_steps"]) >= min_steps
        slope = r["slope_ratio"]
        slope_applies = not (low_span and reading == "low_span_exempt")
        if not slope_applies:
            slope_ok = True
        elif slope is None:
            slope_ok = False          # an undefined slope cannot clear a band
        else:
            slope_ok = lo <= float(slope) <= hi
        d_ok = steps_ok and slope_ok
        ok &= d_ok
        out["by_driver"][name] = {
            "realised_span_pp": round(span_pp, 3), "low_span": bool(low_span),
            "min_steps_required": min_steps, "steps": int(r["pred_monotone_steps"]),
            "steps_ok": bool(steps_ok), "slope_ratio": slope,
            "slope_clause_applies": bool(slope_applies), "slope_ok": bool(slope_ok),
            "pass": bool(d_ok),
        }
    out["pass"] = bool(ok)
    out["failed_drivers"] = [k for k, v in out["by_driver"].items() if not v["pass"]]
    return out


def responsiveness(te: pd.DataFrame, p: np.ndarray, n_q: int = 5) -> dict[str, dict]:
    """Both pre-registered drivers, each measured on the rows where the driver
    is DEFINED (the shooter/defence has at least one prior attempt of this
    class). The undefined rows have no as-of rate to bucket -- imputing the
    league mean and then bucketing on the imputation would be measuring the
    imputation -- so they are reported as their own labelled cell instead."""
    y = te["y"].to_numpy()
    out: dict[str, dict] = {}
    for feat, gate in RESPONSIVENESS_SPECS:
        defined = te[gate].to_numpy() > 0
        sub = te[defined]
        r = PM.quintile_responsiveness(_rank(sub[feat].to_numpy()), y[defined], p[defined],
                                       CLASS_INDEX["MAKE"], n_q=n_q)
        r["driver"] = feat
        r["n_defined"] = int(defined.sum())
        r["n_undefined"] = int((~defined).sum())
        if (~defined).any():
            r["undefined_pred_share"] = round(float(p[~defined, CLASS_INDEX["MAKE"]].mean()), 5)
            r["undefined_actual_share"] = round(float((y[~defined] == 1).mean()), 5)
        out[f"{feat}->MAKE"] = r
    return out


def score(te: pd.DataFrame, p: np.ndarray) -> dict:
    """The full pre-registered scorecard for one arm on one class on one fold."""
    y = te["y"].to_numpy()
    calib = PM.decile_calibration(y, p, CLASSES)
    calib_ok, calib_worst, calib_who = PM.calibration_verdict(calib)
    resp = responsiveness(te, p)
    resp_ok, resp_worst = PM.responsiveness_verdict(resp, RESPONSIVENESS_MIN_STEPS)
    d8 = decision8_verdict(resp, reading=DECISION8_ADOPTED_READING)
    return {
        "n": int(len(te)),
        "log_loss": PM.log_loss(y, p),
        "brier": float(((p[:, CLASS_INDEX["MAKE"]] - y) ** 2).mean()),
        "actual_make_rate": float(y.mean()),
        "pred_make_rate": float(p[:, CLASS_INDEX["MAKE"]].mean()),
        "calibration": calib,
        "calib_pass": bool(calib_ok),
        "calib_worst_gap_pp": calib_worst,
        "calib_worst_class": calib_who,
        "responsiveness": resp,
        "resp_pass": bool(resp_ok),              # the SUPERSEDED steps-only gate
        "resp_min_steps": resp_worst,
        "resp_decision8": d8,                    # the live gate (Decision 8)
        "resp_pass_decision8": bool(d8["pass"]),
        "resp_failed_drivers_decision8": d8["failed_drivers"],
        "by_chance": PM.segment_calibration(
            np.where(te["chance_number"].to_numpy() > 1, "continuation", "first"), y, p, CLASSES),
    }


# ---------------------------------------------------------------------------
# G4: the implied team eFG%
# ---------------------------------------------------------------------------
def efg_table(te: pd.DataFrame, p_make: np.ndarray, side: str = "offense",
              n_tiers: int = 3) -> dict:
    """Implied vs actual team eFG% by tercile (gate G4, +/- 1.0 pp).

    eFG% = (FGM + 0.5 * 3PM) / FGA, computed on the TEST SEASON'S OWN SHOT MIX:
    every attempt the team actually took, weighted by the class model's
    predicted make probability instead of by the outcome. That is what makes it
    a check of the three make models rather than of the shot-selection model --
    the mix is taken as given, exactly as the pre-registration asks.

    Two tercile definitions are reported. `asof` buckets teams by their own
    pregame as-of form, which is the honest, matchup-specific grouping; `actual`
    buckets by the team's realised eFG%, which is an ORACLE grouping and is
    labelled as such -- it cannot be used to select anything, but it is the one
    that exposes a shape miss (a model that is right on average and flat across
    the range)."""
    team_col = "off_team_id" if side == "offense" else "def_team_id"
    three = (te["class_key"].to_numpy() == "three")
    y = te["y"].to_numpy().astype("float64")
    w = 1.0 + 0.5 * three
    df = pd.DataFrame({
        "team": te[team_col].to_numpy(),
        "fga": 1.0,
        "act": y * w,
        "imp": p_make * w,
        "form": te["off_make_c"].to_numpy() if side == "offense" else -te["def_allow_c"].to_numpy(),
    })
    g = df.groupby("team", as_index=False).agg(fga=("fga", "sum"), act=("act", "sum"),
                                               imp=("imp", "sum"), form=("form", "mean"))
    g = g[g["fga"] >= 200]
    g["actual_efg"] = g["act"] / g["fga"]
    g["implied_efg"] = g["imp"] / g["fga"]
    out: dict = {
        "side": side,
        "n_teams": int(len(g)),
        "overall_actual_efg_pct": round(float(g["act"].sum() / g["fga"].sum() * 100), 3),
        "overall_implied_efg_pct": round(float(g["imp"].sum() / g["fga"].sum() * 100), 3),
        "team_level_mae_pp": round(float((g["implied_efg"] - g["actual_efg"]).abs().mean() * 100), 3),
        "team_level_corr": round(float(np.corrcoef(g["implied_efg"], g["actual_efg"])[0, 1]), 4)
        if len(g) > 2 else None,
    }
    out["overall_gap_pp"] = round(out["overall_implied_efg_pct"] - out["overall_actual_efg_pct"], 3)
    if len(g) < n_tiers:
        for label in ("asof", "actual"):
            out[f"terciles_{label}"] = []
            out[f"worst_gap_pp_{label}"] = None
            out[f"pass_{label}"] = None
        return out
    for label, key in (("asof", "form"), ("actual", "actual_efg")):
        q = pd.qcut(g[key].rank(method="first"), n_tiers, labels=False)
        rows = []
        for t in range(n_tiers):
            m = q == t
            if not m.any():
                continue
            a = float(g.loc[m, "act"].sum() / g.loc[m, "fga"].sum() * 100)
            i = float(g.loc[m, "imp"].sum() / g.loc[m, "fga"].sum() * 100)
            rows.append({"tercile": int(t + 1), "n_teams": int(m.sum()),
                         "actual_efg_pct": round(a, 3), "implied_efg_pct": round(i, 3),
                         "gap_pp": round(i - a, 3)})
        out[f"terciles_{label}"] = rows
        out[f"worst_gap_pp_{label}"] = round(max(abs(r["gap_pp"]) for r in rows), 3) if rows else None
        out[f"pass_{label}"] = bool(rows and max(abs(r["gap_pp"]) for r in rows) <= 1.0)
    return out


@dataclass
class FittedFgMake:
    """What the trainer persists for the sim to load: one entry per shot class."""
    arm: str
    feature_set: str
    fold: str
    features: list[str]
    models: dict
    classes: tuple[str, ...]
    meta: dict


__all__ = [
    "ARMS", "ARM_FEATURE_SET", "ARM_SIMPLICITY", "BANNED_FEATURES", "CLASSES",
    "CLASS_INDEX", "CLASS_KEY", "CLASS_KEYS", "COUNT_COLS",
    "DECISION8_ADOPTED_READING", "DECISION8_READINGS",
    "DEFAULT_VERSION", "DEF_SHRINK_GRID", "FEATURE_SETS", "FOLDS", "LGBM_LADDER",
    "LINEUP_FOLDS", "LINEUP_PRIOR_GRID", "LOW_SPAN_PP", "MIN_STEPS_DEFAULT",
    "MIN_STEPS_LOW_SPAN", "PRIOR_KINDS", "RESPONSIVENESS_MIN_STEPS",
    "RESPONSIVENESS_SPECS", "SELECTION_FOLD", "SHOT_CLASSES", "SHRINK_GRID",
    "SLOPE_BAND", "TREE_ARMS", "FittedFgMake", "LgbmArm", "RidgeArm",
    "R2_ARMS", "R2_ARM_FEATURE_SET", "R2_ARM_SIMPLICITY", "R2_EG_HI", "R2_EG_LO",
    "R2_EG_SECONDS", "R2_FEATURE_SETS", "R2_GT_MARGIN", "R2_GT_SECONDS",
    "R2_INDICATORS", "R2_INELIGIBLE", "R2_MARGIN_COL", "R2_SAFE_STATE",
    "add_round2_state", "regulation_seconds_remaining",
    "attach_lineup_features", "build_design", "build_fg_events", "chance_state",
    "class_slice", "decision8_verdict", "defender_rates", "design_matrix",
    "eb_predict", "efg_table", "feature_set", "fit_arm", "fit_by_class", "fit_eb",
    "fold_slices", "league_asof_by_date", "predict_arm", "responsiveness", "score",
    "shooter_form", "team_shot_form",
]
