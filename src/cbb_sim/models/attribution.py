"""
attribution.py -- L4 PLAYER ATTRIBUTION: after the engine has resolved a
TEAM-level event, which player is credited with the secondary stat.

Pre-registration (verbatim, PM, 2026-09-10):
`docs/models/attribution/experiments.md` section 1. Trainer:
`scripts/train_attribution_v1.py`. Feature provenance:
`docs/models/attribution/features.md`.

===========================================================================
WHAT THIS MODEL IS, AND WHAT IT IS NOT
===========================================================================
The cascade has already decided everything about the event except who gets the
box-score line. `possession_outcome` decided that the chance ended in a missed
shot or a turnover; `fg_make` decided made vs missed; `rebound` decided that the
OFFENCE (not the defence, not a dead ball) recovered the miss; `rotation`
decided which ten players were on the floor; `usage` decided which of the five
OFFENSIVE players took the shot or lost the ball. What is left is the SECONDARY
credit:

    (R) which of the rebounding team's five secured a live rebound
    (A) whether a made field goal was assisted, and by which team-mate
    (S) whether a turnover was a steal, and by which defender
    (B) whether a missed field goal was blocked, and by which defender

Every one of those is a choice among the five on the floor (four, for the
assist: a shooter cannot assist himself), optionally preceded by a binary. So
the model class is the one `usage` already settled: a conditional choice over
the actual lineup with per-player as-of rates as the only player-identity
input. This module is that same design applied to five more targets, which is
why it imports `usage`'s conditional logit, its normaliser and its allocation
cell machinery rather than re-deriving them.

It does NOT decide how many rebounds / assists / steals / blocks there are --
`rebound` owns the rebound count, `possession_outcome` owns the turnover count,
`fg_make` owns the miss count, and the three BINARIES here own only the
conditional share of those counts that carries a secondary credit. The
pre-registration's game-level checks re-attribute the ACTUAL event sequence over
the ACTUAL fives precisely so that a failure here cannot be confused with a
failure in any of those neighbours.

===========================================================================
THE TARGETS
===========================================================================
Five CHOICE targets and three BINARY targets:

  target      population                            candidates     credited
  ---------   -----------------------------------   ------------   --------------
  REB_off     live OREB rows                        offence five   rebounder
  REB_def     live DREB rows                        defence five   rebounder
  assist      made FGA rows that ARE assisted       offence five   assister
                                                    MINUS shooter
  steal       TOV rows that ARE steals              defence five   stealer
  block       missed FGA rows that ARE blocked      defence five   blocker

  assisted    made FGA rows                         --             was it assisted
  stolen      TOV rows                              --             was it a steal
  blocked     missed FGA rows                       --             was it blocked

Offensive and defensive rebounds are modelled SEPARATELY, as the
pre-registration requires: they are different skills on different bodies (an
offensive board is won against four opponents in the paint, a defensive board is
the default outcome of a miss), and pooling them would force one rate per player
where the data plainly carries two.

`DeadBallReb` is NOT a live rebound and is excluded from both R targets -- it
carries no player credit at the source and `rebound` already models it as a
fixed per-miss-type share (L17). Free-throw rebounds ARE live rebounds (a missed
last free throw is a live ball) and are included; ESPN's ADMINISTRATIVE reset
rebound between two free throws of one trip is dropped by exactly the rule
`event_stream` and `possessions._collect_trip` use.

===========================================================================
WHY THIS MODULE BUILDS ITS OWN EVENT STREAM
===========================================================================
`cbb_sim.models.event_stream` is the stream `rebound` and `free_throw` train on,
and three of the four credits here are not in it:

  * it DROPS the `Block Shot` rows after folding them into a `blocked` flag, so
    the BLOCKER's id is gone by the time the stream is returned;
  * it carries no assist columns at all (`shot_assisted` and
    `shot_assisted_by_id` are not in `pbp.events.PLAY_COLUMNS`);
  * `steal` rows do survive in it, but the stream keeps no play `id`, so there
    is no key to join a stealer back onto its turnover row with.

`event_stream.py` is the shared prerequisite layer `rebound` and `free_throw`
already ship against, so it is NOT edited here. `build_attr_stream` instead
re-does the same cleaning on the same source with the three extra columns
attached: identical side repair (`possessions._fix_flipped_sides`, the
majority-vote repair of data-defect D2), identical administrative-rebound drop,
and the adjacency rule `event_stream` uses for `blocked` extended to carry the
blocker's and the stealer's ids as well as the flag. It deliberately does NOT
re-derive the free-throw trip machinery or the running team-foul counts: no
target here needs either.

ADJACENCY, MEASURED. A `Steal` row follows its `Lost Ball Turnover` row at the
same `secondsRemaining`, and a `Block Shot` row sits adjacent to its attempt at
the same second; both are matched in BOTH directions, exactly as `event_stream`
matches `blocked`, and the share of flags with no resolvable id is reported by
`coverage_report` rather than dropped silently.

===========================================================================
THE CHOICE SET, AND WHY THE ASSIST HAS FOUR ALTERNATIVES
===========================================================================
Candidates are sorted ascending by CBBD player id, so the alternative order is a
function of the lineup and never of the feed's column order, and `y` is the slot
index of the credited player. For the assist target the SHOOTER is removed from
the offensive five first: a made field goal cannot be assisted by the player who
made it, so a five-alternative assist model would spend probability mass on an
impossible outcome and its log loss would not be comparable to a four-alternative
one. `tests/test_attribution.py` pins this.

Uniform baselines: 1.609438 over five, 1.386294 over four.

===========================================================================
THE AS-OF INPUTS
===========================================================================
One row per (season, player, game), every column an expanding sum over that
player's STRICTLY EARLIER games in the season (`prob_metrics.expanding_asof`),
plus a COMPLETED prior season and a position-group prior -- the structure
`usage.build_player_asof` established, generalised from five offensive event
classes to five attribution targets with per-target candidate sets.

EXPOSURE IS PER OPPORTUNITY OF THE TARGET'S OWN POPULATION, ON THE FLOOR. A
player's `REB_def` exposure is the number of live defensive rebounds his team won
while he was one of the defensive five; his numerator is how many of them he
secured. So the candidates' rates sum to approximately one over the candidate
set, which is exactly the quantity P1 normalises, and a fitted shrinkage
strength reads directly as "pseudo opportunities of history before a player's own
rate outweighs the prior". `rebound.player_rebound_rates` makes the other
defensible choice (denominator = every miss while on the floor, so the rate is a
share of AVAILABLE rebounds); both are documented in `features.md` section 2 and
the difference is a denominator, not a leak.

The three BINARIES are team-level questions and get team-level as-of inputs: the
attacking team's own as-of rate and the defending team's as-of rate allowed,
plus -- for the "aware" arm -- the individual offensive player's own as-of share
and the shot / miss class.

===========================================================================
THE SAMPLER
===========================================================================
Four samplers, one per credit, each on its OWN RNG family so that adding or
removing one credit from the engine does not move any other credit's draws:

    draw_rebounder -> family "attr_rebound"
    draw_assist    -> family "attr_assist"
    draw_steal     -> family "attr_steal"
    draw_block     -> family "attr_block"

Each consumes one uniform per call from the `(seed, game_id, family)` stream
(`cbb_sim.control.rng`), and the trainer's re-attribution path uses the
`(seed, game_id, family, ordinal)` keys `usage.event_stream_keys` established:
keying on the game alone would hand every event of a team-game the same uniform
and collapse the whole game onto one player (measured in the usage bake-off:
3.42 players with a three-point attempt per team-game against a real 6.69).

===========================================================================
FOLDS AND THE SEAL
===========================================================================
F1 trains 2024 and tests 2025 and is the selection fold -- on-floor ids are
empty at the source in 2022-2023 (L13), so there is no earlier fold to build.
The robustness fold is a within-2025 walk-forward, train before 2025-01-15, test
after. 2026 is sealed; `fold_slices` and `walkforward_slices` both call
`assert_not_sealed`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.control import rng as RNG
from cbb_sim.data.seal import assert_not_sealed
from cbb_sim.models import event_stream as ES
from cbb_sim.models import prob_metrics as PM
from cbb_sim.models import usage as U
from cbb_sim.models.free_throw import load_positions
from cbb_sim.pbp.events import PLAY_COLUMNS, classify_frame, load_plays
from cbb_sim.pbp.possessions import _fix_flipped_sides

DEFAULT_ARTIFACT_DIR = Path("data/processed/models/attribution")

# ===========================================================================
# 0. Vocabulary
# ===========================================================================
#: The five choice targets. Order is fixed: every artifact and every table in
#: the docs uses it.
CHOICE_TARGETS: tuple[str, ...] = ("REB_off", "REB_def", "assist", "steal", "block")
#: The three binary targets, in the order their choice partner appears above.
BINARY_TARGETS: tuple[str, ...] = ("assisted", "stolen", "blocked")
TARGETS: tuple[str, ...] = (*CHOICE_TARGETS, *BINARY_TARGETS)

#: Alternatives per choice target. The assist has FOUR: the shooter is removed
#: from the offensive five (module docstring).
N_ALT_OF: dict[str, int] = {"REB_off": 5, "REB_def": 5, "assist": 4,
                            "steal": 5, "block": 5}
#: Uniform-over-K log loss, the floor any choice arm has to beat.
UNIFORM_LL: dict[str, float] = {t: float(np.log(k)) for t, k in N_ALT_OF.items()}

#: Population table each target reads. `made_fga` serves both the `assisted`
#: binary and the `assist` choice (its assisted subset), and so on.
POPULATIONS: tuple[str, ...] = ("reb_off", "reb_def", "made_fga", "tov", "miss_fga")
POP_OF: dict[str, str] = {
    "REB_off": "reb_off", "REB_def": "reb_def", "assist": "made_fga",
    "steal": "tov", "block": "miss_fga",
    "assisted": "made_fga", "stolen": "tov", "blocked": "miss_fga",
}
#: The binary target each population defines, where it defines one.
BINARY_OF_POP: dict[str, str] = {"made_fga": "assisted", "tov": "stolen",
                                 "miss_fga": "blocked"}
#: Which side's five is the candidate set. `off` = the five with the ball.
SIDE_OF: dict[str, str] = {"REB_off": "off", "REB_def": "def", "assist": "off",
                           "steal": "def", "block": "def"}

#: Per-player as-of rate class behind each choice target.
RATE_OF: dict[str, str] = {"REB_off": "oreb", "REB_def": "dreb",
                           "assist": "assist", "steal": "steal", "block": "block"}
RATE_CLASSES: tuple[str, ...] = ("oreb", "dreb", "assist", "steal", "block")

#: The individual OFFENSIVE player's own as-of share behind each binary (the
#: "shooter-aware" half of the pre-registration's second binary arm).
OWN_SHARE_OF: dict[str, str] = {"assisted": "own_assisted", "stolen": "own_stolen",
                                "blocked": "own_blocked"}
OWN_SHARE_CLASSES: tuple[str, ...] = ("own_assisted", "own_stolen", "own_blocked")

POSITION_LEVELS: tuple[str, ...] = U.POSITION_LEVELS

CHOICE_ARMS: tuple[str, ...] = ("proportional", "cond_logit", "lgbm")
BINARY_ARMS: tuple[str, ...] = ("team_ridge", "aware_ridge", "lgbm")
#: Tie-break order of the pre-registration: P1 < P2 < P3, and
#: team ridge < aware ridge < tree.
CHOICE_ARM_ORDER = {a: i for i, a in enumerate(CHOICE_ARMS)}
BINARY_ARM_ORDER = {a: i for i, a in enumerate(BINARY_ARMS)}
TREE_ARM = "lgbm"


def arms_for(target: str) -> tuple[str, ...]:
    """The arm list of `target`: three choice arms or three binary arms."""
    return CHOICE_ARMS if target in CHOICE_TARGETS else BINARY_ARMS


def arm_order(target: str) -> dict[str, int]:
    """The pre-registered simplicity order used for tie-breaks."""
    return CHOICE_ARM_ORDER if target in CHOICE_TARGETS else BINARY_ARM_ORDER


FOLDS: dict[str, dict[str, list[int]]] = {"F1": {"train": [2024], "test": [2025]}}
SELECTION_FOLD = "F1"
WF_SEASON = 2025
WF_SPLIT_DATE = "2025-01-15"

#: Shrinkage grid in pseudo OPPORTUNITIES of the target's own population, and
#: the three priors the pre-registration names.
SHRINK_GRID: tuple[float, ...] = (2, 5, 10, 25, 50, 100, 200, 400, 800)
PRIOR_KINDS: tuple[str, ...] = ("league", "position", "prior_season")

#: Pre-registered gates. The responsiveness pair is `ARCHITECTURE_DECISIONS.md`
#: Decision 8 as written (slope ratio in [0.8, 1.2] AND monotone in at least
#: 3 of 4 quintile steps, with a sub-2-pp realised span exempting a driver from
#: BOTH clauses). This model's pre-registration cites Decision 8 directly rather
#: than restating a step count, so there is no earlier wording to supersede.
CALIB_GATE_PP = 2.0
SLOPE_BAND = (0.8, 1.2)
RESP_MIN_STEPS = 3
SMALL_SPAN_PP = 2.0
SD_RATIO_BAND = (0.9, 1.1)
GT0_TOL = 0.5
TOPSHARE_TOL_PP = 2.0

EPS = 1e-12


# ===========================================================================
# 1. The event stream (module docstring, "WHY THIS MODULE BUILDS ITS OWN")
# ===========================================================================
#: `participant_1_id` and the two assist columns are not in `PLAY_COLUMNS`; the
#: targets here need all three.
ATTR_PLAY_COLUMNS: tuple[str, ...] = (
    *PLAY_COLUMNS, "participant_1_id", "shot_assisted", "shot_assisted_by_id")

#: Classes with neither possession nor attribution information. `block` and
#: `steal` are NOT inert here (unlike `event_stream`, which keeps `steal` but
#: drops `block`): both carry the defensive credit this model is about, and both
#: are dropped only AFTER their player id has been moved onto the event they
#: decorate.
INERT: frozenset[str] = frozenset({"timeout", "sub", "jumpball", "challenge"})


def _bool_array(col: pd.Series, fallback: pd.Series | None = None) -> np.ndarray:
    """`event_stream._bool_array`, re-stated so this module's leak and coverage
    proofs bind to the code it actually calls (the `prob_metrics.expanding_asof`
    precedent)."""
    s = col
    if s.dtype == object:
        s = s.map({True: True, False: False})
    s = s.astype("boolean")
    if fallback is not None:
        s = s.fillna(fallback.astype("boolean"))
    return s.fillna(False).to_numpy(dtype=bool)


def _adjacent_credit(g: np.ndarray, sec: np.ndarray, is_credit: np.ndarray,
                     pid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(flag, credited id) for a decorating row that sits immediately before or
    after the event it decorates, in the same game at the same second.

    This is `event_stream`'s `blocked` rule with the player id carried through.
    The NEXT row is preferred over the previous one when both are credit rows,
    because ESPN emits the decoration after the event it decorates on the large
    majority of rows (`Lost Ball Turnover` then `Steal`; attempt then
    `Block Shot`); the preference only matters on the handful of rows with a
    credit on both sides."""
    same_prev = np.concatenate([[False], g[1:] == g[:-1]])
    same_next = np.concatenate([g[1:] == g[:-1], [False]])
    same_sec_prev = np.concatenate([[False], sec[1:] == sec[:-1]])
    same_sec_next = np.concatenate([sec[:-1] == sec[1:], [False]])
    prev_c = np.concatenate([[False], is_credit[:-1]]) & same_prev & same_sec_prev
    next_c = np.concatenate([is_credit[1:], [False]]) & same_next & same_sec_next
    prev_id = np.concatenate([[np.nan], pid[:-1]])
    next_id = np.concatenate([pid[1:], [np.nan]])
    flag = prev_c | next_c
    who = np.where(next_c, next_id, np.where(prev_c, prev_id, np.nan))
    return flag, who


def build_attr_stream(
    season: int,
    universe: pd.DataFrame,
    rim_override_max_ft: float = 0.0,
    pbp_dir: Path | str = ES.DEFAULT_PBP_DIR,
) -> pd.DataFrame:
    """The cleaned event stream for one season, with the four attribution
    credits attached to the rows they belong to (module docstring).

    Columns beyond `event_stream`'s: `assisted` / `assister_id` on made field
    goals, `stolen` / `stealer_id` on turnovers, `blocked` / `blocker_id` on
    attempts, and the ten on-floor ids split into an offensive and a defensive
    five by `side`."""
    season = int(season)
    u = universe[universe["season"] == season]
    if not len(u):
        raise ValueError(f"no universe rows for season {season}")
    gids = set(u["cbbd_game_id"].dropna().astype("int64").tolist())
    plays = load_plays(season, pbp_dir=pbp_dir, game_ids=gids,
                       columns=ATTR_PLAY_COLUMNS)
    cls = classify_frame(plays, rim_override_max_ft=rim_override_max_ft)

    keep = ~cls.isin(list(INERT)).to_numpy()
    plays = plays.loc[keep].reset_index(drop=True)
    cls = cls.loc[keep].reset_index(drop=True)

    is_home = plays["isHomeTeam"]
    if is_home.dtype == object:
        is_home = is_home.map({True: True, False: False})
    is_home = is_home.astype("boolean")
    has_team = (pd.to_numeric(plays["teamId"], errors="coerce").notna()
                & is_home.notna()).to_numpy()
    side = np.where(is_home.fillna(False).to_numpy(), 0, 1)
    side = _fix_flipped_sides(plays, side, has_team)
    side = np.where(has_team, side, -1).astype("int16")

    period = pd.to_numeric(plays["period"], errors="coerce").fillna(1).astype("int16")
    df = pd.DataFrame({
        "cbbd_game_id": plays["gameId"].to_numpy(),
        "season": np.full(len(plays), season, dtype="int16"),
        "cls": cls.to_numpy(dtype=object),
        "side": side,
        "period": period.to_numpy(),
        "sec": pd.to_numeric(plays["secondsRemaining"], errors="coerce").fillna(0)
                 .astype("int32").to_numpy(),
        "home_score": pd.to_numeric(plays["homeScore"], errors="coerce").ffill()
                        .fillna(0).astype("int32").to_numpy(),
        "away_score": pd.to_numeric(plays["awayScore"], errors="coerce").ffill()
                        .fillna(0).astype("int32").to_numpy(),
        "made": _bool_array(plays["shot_made"], plays["scoringPlay"]),
        "player_id": pd.to_numeric(plays["participant_1_id"], errors="coerce").to_numpy(),
        # THE SHOOTER IS `shot_shooter_id`, NOT `participant_1_id`. Measured on
        # 2025 made field goals: `participant_1_id` equals `shot_shooter_id` on
        # 74.5% of rows overall and on only 50.9% of ASSISTED ones, where it is
        # the ASSISTER on the other 48.9% -- CBBD's participant ORDER is not
        # stable on a two-participant row. Reading the shooter off
        # `participant_1_id` would therefore make the assister "the shooter" on
        # half the assist population, remove the wrong man from the choice set,
        # and hand the model an impossible target. `shot_shooter_id` is a
        # dedicated column and is populated on every shooting row.
        "shooter_id": pd.to_numeric(plays["shot_shooter_id"], errors="coerce").to_numpy(),
        "assisted_raw": _bool_array(plays["shot_assisted"]),
        "assister_raw": pd.to_numeric(plays["shot_assisted_by_id"],
                                      errors="coerce").to_numpy(),
        "assist_col_present": plays["shot_assisted"].notna().to_numpy(),
    })
    for c in ES.ON_FLOOR_COLS:
        df[c] = pd.to_numeric(plays[c], errors="coerce").to_numpy()

    g = df["cbbd_game_id"].to_numpy()
    c = df["cls"].to_numpy(dtype=object)
    sec = df["sec"].to_numpy()
    pid = df["player_id"].to_numpy()

    # --- blocker and stealer, then drop the decorating rows ------------------
    blk = c == "block"
    stl = c == "steal"
    df["blocked"], df["blocker_id"] = _adjacent_credit(g, sec, blk, pid)
    df["stolen"], df["stealer_id"] = _adjacent_credit(g, sec, stl, pid)
    n_block_rows, n_steal_rows = int(blk.sum()), int(stl.sum())
    df = df[~(blk | stl)].reset_index(drop=True)

    # --- administrative rebounds inside a free-throw trip --------------------
    # Exactly `event_stream`'s rule: a rebound row whose previous AND next rows
    # are free throws by the same team is a dead-ball reset, not a live rebound.
    g = df["cbbd_game_id"].to_numpy()
    c = df["cls"].to_numpy(dtype=object)
    sd = df["side"].to_numpy()
    is_ft = np.isin(c, ES.FT_CLASSES)
    same_prev = np.concatenate([[False], g[1:] == g[:-1]])
    same_next = np.concatenate([g[1:] == g[:-1], [False]])
    prev_ft = np.concatenate([[False], is_ft[:-1]]) & same_prev
    next_ft = np.concatenate([is_ft[1:], [False]]) & same_next
    prev_side = np.concatenate([[-9], sd[:-1]])
    next_side = np.concatenate([sd[1:], [-9]])
    admin = (np.isin(c, ["OREB", "DeadBallReb"]) & prev_ft & next_ft
             & (prev_side == next_side))
    n_admin = int(admin.sum())
    df = df[~admin].reset_index(drop=True)

    # --- the shot class the row is "about" -----------------------------------
    # On an attempt it is the attempt's own class. On a REBOUND row it is the
    # class of the miss immediately before it, which is the pre-registration's
    # "shot class where relevant" for the two R targets: an offensive board off
    # a rim miss is a different event from one off a three. `prev_miss_cls` is
    # read off the previous row in the same game, and is `other` when that row
    # is not a miss (which happens on the rebound of a blocked shot only after
    # the block row has already been dropped, i.e. never).
    c = df["cls"].to_numpy(dtype=object)
    g = df["cbbd_game_id"].to_numpy()
    made = df["made"].to_numpy()
    is_miss = (np.isin(c, ES.FGA_CLASSES) & ~made) | (c == "FT_missed")
    prev_same = np.concatenate([[False], g[1:] == g[:-1]])
    prev_cls = np.concatenate([[None], c[:-1]])
    prev_is_miss = np.concatenate([[False], is_miss[:-1]]) & prev_same
    df["prev_miss_cls"] = np.where(prev_is_miss, prev_cls, "other")

    # --- universe keys ------------------------------------------------------
    ucols = ["game_id", "cbbd_game_id", "game_date", "neutral_site",
             "home_team_id", "away_team_id"]
    df = df.merge(u[ucols], on="cbbd_game_id", how="inner")
    home = df["home_team_id"].to_numpy()
    away = df["away_team_id"].to_numpy()
    sd = df["side"].to_numpy()
    df["team_id"] = np.where(sd == 0, home, np.where(sd == 1, away, -1)).astype("int64")
    df["opp_id"] = np.where(sd == 0, away, np.where(sd == 1, home, -1)).astype("int64")
    df.attrs["n_admin_rebounds_dropped"] = n_admin
    df.attrs["n_block_rows"] = n_block_rows
    df.attrs["n_steal_rows"] = n_steal_rows
    df.attrs["rim_override_max_ft"] = float(rim_override_max_ft)
    return df


# ===========================================================================
# 2. The population tables
# ===========================================================================
#: Every population table carries these, plus `cand_1..cand_5`, `y`, and the
#: population's own binary and offensive-player columns.
BASE_COLUMNS: tuple[str, ...] = (
    "game_id", "cbbd_game_id", "season", "game_date", "neutral_site",
    "cand_team_id", "cand_opp_id", "off_team_id", "def_team_id",
    "cand_is_home", "offense_is_home", "period", "sec_remaining", "score_diff",
    "event_class", "shot_class", "credited_id", "off_player_id", "blocked_flag",
    "five_ok", "in_five", "y", "b", "credit_id_present",
)

#: The "shot class where relevant" vocabulary. `none` is the turnover
#: population (no shot exists); `other` is a rebound whose preceding row is not
#: a miss.
SHOT_CLASSES: tuple[str, ...] = ("FGA_rim", "FGA_jump2", "FGA_3", "FT_missed",
                                 "none", "other")
SHOT_CLASS_CODE = {c: i for i, c in enumerate(SHOT_CLASSES)}


def _five(stream: pd.DataFrame, which: str) -> np.ndarray:
    """(n, 5) CBBD ids of the OFFENSIVE (`off`) or DEFENSIVE (`def`) five.

    `side` is the team the ROW belongs to, so for a rebound row the row's team
    is the rebounding team; the caller passes the side it wants relative to the
    row's own team through `which`."""
    home = stream[[f"home_on_{k}" for k in range(1, 6)]].to_numpy(dtype="float64")
    away = stream[[f"away_on_{k}" for k in range(1, 6)]].to_numpy(dtype="float64")
    own = np.where((stream["side"].to_numpy() == 0)[:, None], home, away)
    opp = np.where((stream["side"].to_numpy() == 0)[:, None], away, home)
    return own if which == "own" else opp


def _sorted_candidates(five: np.ndarray, drop_id: np.ndarray | None,
                       n_alt: int) -> tuple[np.ndarray, np.ndarray]:
    """(sorted candidate ids, ok flag). `drop_id` removes one player (the
    shooter, for the assist target) before sorting.

    `ok` requires exactly `n_alt` finite DISTINCT ids, so a feed row with a
    duplicated on-floor id cannot masquerade as a valid choice set."""
    f = np.array(five, dtype="float64", copy=True)
    if drop_id is not None:
        f = np.where(np.isclose(f, np.where(np.isfinite(drop_id), drop_id,
                                            -1.0)[:, None]), np.nan, f)
    finite = np.isfinite(f)
    srt = np.sort(np.where(finite, f, np.inf), axis=1)[:, :n_alt]
    ok = (finite.sum(axis=1) == n_alt) & np.isfinite(srt).all(axis=1)
    distinct = np.zeros(len(f), dtype=bool)
    if ok.any():
        distinct[ok] = (np.diff(srt[ok], axis=1) > 0).all(axis=1)
    return srt, ok & distinct


def _credit_slot(cand: np.ndarray, credited: np.ndarray, ok: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(in_set, slot index) of the credited player inside the candidate set."""
    eq = np.isclose(cand, np.where(np.isfinite(credited), credited, -1.0)[:, None])
    in_set = ok & np.isfinite(credited) & eq.any(axis=1)
    return in_set, np.where(in_set, eq.argmax(axis=1), -1).astype("int8")


#: L27 / `docs/tests/attribution_score_diff_leak_2026-09-10.md`: CBBD's
#: `homeScore`/`awayScore` are the score AFTER the row's own play, exactly the
#: fg_make defect this ports the fix from. Of the five populations built here,
#: `made_fga` is the ONLY one whose own row is itself a scoring play (a live
#: rebound, a turnover and a missed/blocked attempt never change the score on
#: their own row), so it is the only population the leak can reach -- proved by
#: the own-row delta test in that doc, not assumed.
SCORE_DIFF_MODES: tuple[str, ...] = ("pre_play", "leaked")


def _state_block(s: pd.DataFrame, cand_is_home: np.ndarray, pop: str,
                 score_diff_mode: str = "pre_play") -> dict:
    """The event's state: `period`, `sec_remaining` and `score_diff`.

    `score_diff_mode`:
      - `"pre_play"` (default): the candidate team's margin BEFORE this row's
        own event. Ported from `fg_make.add_round2_state`'s `score_diff_pre`
        repair: subtract the row's own points when the row is a MADE field
        goal (`made_fga`'s own population), zero everywhere else, since a
        miss (or a rebound, or a turnover) is unchanged by construction --
        the correction can only remove outcome information, never add it.
      - `"leaked"`: the score AFTER the row's own play, the construction this
        replaces. Kept selectable for reproducing round 1's numbers; never
        the default.
    """
    if score_diff_mode not in SCORE_DIFF_MODES:
        raise ValueError(f"unknown score_diff_mode {score_diff_mode!r}, "
                         f"expected one of {SCORE_DIFF_MODES}")
    period = s["period"].to_numpy()
    sec = s["sec"].to_numpy()
    hs = s["home_score"].to_numpy()
    as_ = s["away_score"].to_numpy()
    score_diff = np.where(cand_is_home, hs - as_, as_ - hs).astype("int16")
    if score_diff_mode == "pre_play" and pop == "made_fga":
        made = s["made"].to_numpy().astype(bool)
        pts = np.where(s["cls"].to_numpy(dtype=object) == "FGA_3", 3, 2)
        own_points = np.where(made, pts, 0).astype("int16")
        score_diff = (score_diff - own_points).astype("int16")
    return {
        "period": period.astype("int16"),
        "sec_remaining": np.where(period <= 2, (2 - period) * 1200 + sec,
                                  sec).astype("int32"),
        "score_diff": score_diff,
    }


def build_attr_events(season: int, universe: pd.DataFrame,
                      rim_override_max_ft: float = 0.0,
                      pbp_dir: Path | str = ES.DEFAULT_PBP_DIR,
                      stream: pd.DataFrame | None = None,
                      score_diff_mode: str = "pre_play",
                      ) -> dict[str, pd.DataFrame]:
    """The five population tables of one season, keyed by `POPULATIONS`.

    Rows whose candidate five is incomplete, or whose credited player is not in
    it, are KEPT with `five_ok` / `in_five` False so `coverage_report` can
    report the loss instead of it happening silently; `usable` applies the
    filter.

    `score_diff_mode` ("pre_play" default, "leaked" selectable for
    reproduction) controls `_state_block`'s `score_diff` construction -- see
    `docs/tests/attribution_score_diff_leak_2026-09-10.md`."""
    st = build_attr_stream(season, universe, rim_override_max_ft, pbp_dir) \
        if stream is None else stream
    cls = st["cls"].to_numpy(dtype=object)
    made = st["made"].to_numpy()
    side = st["side"].to_numpy()
    own5 = _five(st, "own")
    opp5 = _five(st, "opp")

    is_fga = np.isin(cls, ES.FGA_CLASSES)
    masks = {
        "reb_off": (cls == "OREB") & (side >= 0),
        "reb_def": (cls == "DREB") & (side >= 0),
        "made_fga": is_fga & made & (side >= 0),
        "tov": (cls == "TOV") & (side >= 0),
        "miss_fga": is_fga & ~made & (side >= 0),
    }
    out: dict[str, pd.DataFrame] = {}
    for pop, m in masks.items():
        s = st.loc[m].reset_index(drop=True)
        # The CANDIDATE side. For a rebound row the row's team IS the
        # rebounding team, so the candidates are its own five. For a turnover
        # or a missed shot the row's team is the OFFENCE and the credit belongs
        # to the defence.
        cand_own = pop in ("reb_off", "reb_def", "made_fga")
        cand = (own5 if cand_own else opp5)[m]
        row_is_home = s["side"].to_numpy() == 0
        cand_is_home = row_is_home if cand_own else ~row_is_home
        # `offense_is_home` is about the team WITH THE BALL at the event. On a
        # defensive rebound the row's team (the rebounder's) is the defence.
        off_is_home = row_is_home if pop != "reb_def" else ~row_is_home

        # The offensive player of the row: the SHOOTER on a field-goal row (from
        # `shot_shooter_id`, see `build_attr_stream`), the charged player on a
        # turnover, and nobody on a defensive rebound (the row's team is the
        # defence there).
        if pop in ("made_fga", "miss_fga"):
            off_player = s["shooter_id"].to_numpy()
        elif pop == "tov":
            off_player = s["player_id"].to_numpy()
        else:
            off_player = np.full(len(s), np.nan)

        if pop in ("reb_off", "reb_def"):
            credited = s["player_id"].to_numpy()
            binary = None
        elif pop == "made_fga":
            binary = s["assisted_raw"].to_numpy() & s["assist_col_present"].to_numpy()
            credited = np.where(binary, s["assister_raw"].to_numpy(), np.nan)
        elif pop == "tov":
            binary = s["stolen"].to_numpy()
            credited = np.where(binary, s["stealer_id"].to_numpy(), np.nan)
        else:
            binary = s["blocked"].to_numpy()
            credited = np.where(binary, s["blocker_id"].to_numpy(), np.nan)

        # Every population table stores the candidate side's FULL five, so the
        # binary and the choice read the same rows. The assist target's four
        # alternatives are derived in `usable`, which is the single place the
        # "a shooter cannot assist himself" rule lives.
        cand_srt, ok = _sorted_candidates(cand, None, 5)
        in_set, slot = _credit_slot(cand_srt, credited, ok)
        if pop == "made_fga":
            # `in_five` stays (it measures "the assister resolved inside the
            # offensive five", a real coverage number) but the SLOT is voided:
            # a five-slot index is meaningless for a four-alternative target and
            # leaving it would invite a silent off-by-one downstream.
            slot = np.full(len(s), -1, dtype="int8")

        d = pd.DataFrame({
            "game_id": s["game_id"].to_numpy(),
            "cbbd_game_id": s["cbbd_game_id"].to_numpy(),
            "season": np.full(len(s), int(season), dtype="int16"),
            "game_date": s["game_date"].to_numpy(),
            "neutral_site": s["neutral_site"].to_numpy(),
            "cand_team_id": np.where(cand_own, s["team_id"].to_numpy(),
                                     s["opp_id"].to_numpy()),
            "cand_opp_id": np.where(cand_own, s["opp_id"].to_numpy(),
                                    s["team_id"].to_numpy()),
            "off_team_id": np.where(pop == "reb_def", s["opp_id"].to_numpy(),
                                    s["team_id"].to_numpy()),
            "def_team_id": np.where(pop == "reb_def", s["team_id"].to_numpy(),
                                    s["opp_id"].to_numpy()),
            "cand_is_home": cand_is_home,
            "offense_is_home": off_is_home,
            "event_class": s["cls"].to_numpy(dtype=object),
            "shot_class": (s["prev_miss_cls"].to_numpy(dtype=object)
                           if pop in ("reb_off", "reb_def")
                           else (s["cls"].to_numpy(dtype=object)
                                 if pop in ("made_fga", "miss_fga")
                                 else np.full(len(s), "none", dtype=object))),
            "credited_id": credited,
            "off_player_id": off_player,
            "blocked_flag": s["blocked"].to_numpy(),
            **_state_block(s, cand_is_home, pop, score_diff_mode=score_diff_mode),
        })
        for k in range(5):
            d[f"cand_{k + 1}"] = cand_srt[:, k]
        d["five_ok"] = ok
        d["in_five"] = in_set
        d["y"] = slot
        if binary is not None:
            d["b"] = binary.astype("int8")
            d["credit_id_present"] = np.isfinite(credited) | ~binary
        else:
            d["b"] = 1
            d["credit_id_present"] = np.isfinite(credited)
        d.attrs["season"] = int(season)
        d.attrs["stream"] = dict(st.attrs)
        out[pop] = d
    return out


def usable(pop_table: pd.DataFrame, target: str) -> pd.DataFrame:
    """The modelled universe of `target`.

    For a CHOICE target: rows whose binary fired, whose candidate set resolved,
    and whose credited player is in it. For a BINARY target: rows whose
    candidate five resolved, which is the population the team-level features and
    the game-level count check are defined on.

    The assist choice additionally re-derives its FOUR alternatives by removing
    the shooter from the stored five, which is where the eligible-set rule
    ("the shooter cannot assist himself") is enforced."""
    if target in BINARY_TARGETS:
        out = pop_table[pop_table["five_ok"].to_numpy()].reset_index(drop=True)
        for k in range(1, 6):
            out[f"cand_{k}"] = out[f"cand_{k}"].astype("int64")
        return out
    if target == "assist":
        t = pop_table[(pop_table["b"].to_numpy() == 1)
                      & pop_table["five_ok"].to_numpy()].reset_index(drop=True)
        five = t[[f"cand_{k}" for k in range(1, 6)]].to_numpy(dtype="float64")
        cand, ok = _sorted_candidates(five, t["off_player_id"].to_numpy(), 4)
        in_set, slot = _credit_slot(cand, t["credited_id"].to_numpy(), ok)
        t = t.drop(columns=[f"cand_{k}" for k in range(1, 6)])
        for k in range(4):
            t[f"cand_{k + 1}"] = cand[:, k].astype("int64")
        t["five_ok"] = ok
        t["in_five"] = in_set
        t["y"] = slot
        return t[t["in_five"].to_numpy()].reset_index(drop=True)
    out = pop_table[(pop_table["b"].to_numpy() == 1)
                    & pop_table["in_five"].to_numpy()].reset_index(drop=True)
    for k in range(1, 6):
        out[f"cand_{k}"] = out[f"cand_{k}"].astype("int64")
    return out


def coverage_report(pops: dict[str, pd.DataFrame]) -> dict:
    """Per target: rows in the population, the binary's realised share, the
    share with a resolved candidate set, the share whose credited id is present,
    and the modelled share. Reported numbers, never a silent filter."""
    out: dict[str, dict] = {}
    for t in TARGETS:
        p = pops[POP_OF[t]]
        if not len(p):
            continue
        if t in BINARY_TARGETS:
            out[t] = {
                "n_population": int(len(p)),
                "binary_rate_pct": round(float(p["b"].mean() * 100), 4),
                "five_resolved_pct": round(float(p["five_ok"].mean() * 100), 4),
                "credit_id_present_pct": round(
                    float(p["credit_id_present"].mean() * 100), 4),
                "modelled_pct": round(float(p["five_ok"].mean() * 100), 4),
            }
            continue
        fired = p["b"].to_numpy() == 1
        n = int(fired.sum())
        if not n:
            continue
        mod = usable(p, t)
        out[t] = {
            "n_population": int(len(p)),
            "n_fired": n,
            "binary_rate_pct": round(float(fired.mean() * 100), 4),
            "five_resolved_pct": round(float(p["five_ok"].to_numpy()[fired].mean() * 100), 4),
            "credit_id_present_pct": round(
                float(np.isfinite(p["credited_id"].to_numpy()[fired]).mean() * 100), 4),
            "modelled_pct": round(float(len(mod) / n * 100), 4),
        }
    return out


# ===========================================================================
# 3. As-of player features
# ===========================================================================
def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.maximum(den, EPS), np.nan)


def _cand_long(d: pd.DataFrame, n_alt: int, rate_class: str) -> pd.DataFrame:
    """(event x candidate) long frame of one target's MODELLED rows: one row per
    candidate, flagged with whether he was the credited one."""
    n = len(d)
    alt = d[[f"cand_{k + 1}" for k in range(n_alt)]].to_numpy(dtype="int64")
    rep = np.repeat(np.arange(n), n_alt)
    slot = np.tile(np.arange(n_alt), n)
    return pd.DataFrame({
        "season": d["season"].to_numpy()[rep],
        "player_id": alt.reshape(-1),
        "game_id": d["game_id"].to_numpy()[rep],
        "game_date": d["game_date"].to_numpy()[rep],
        "team_id": d["cand_team_id"].to_numpy()[rep],
        "rate_class": rate_class,
        "credited": slot == np.repeat(d["y"].to_numpy(), n_alt),
    })


def _own_long(pop: pd.DataFrame, share_class: str) -> pd.DataFrame:
    """(event) frame keyed on the OFFENSIVE player of the row: the denominator of
    his own as-of share (his made field goals, his turnovers, his misses) and its
    numerator (the ones that carried a secondary credit)."""
    keep = np.isfinite(pop["off_player_id"].to_numpy())
    p = pop.loc[keep]
    return pd.DataFrame({
        "season": p["season"].to_numpy(),
        "player_id": p["off_player_id"].to_numpy().astype("int64"),
        "game_id": p["game_id"].to_numpy(),
        "game_date": p["game_date"].to_numpy(),
        "team_id": p["off_team_id"].to_numpy(),
        "rate_class": share_class,
        "credited": p["b"].to_numpy() == 1,
    })


ALL_CLASSES: tuple[str, ...] = (*RATE_CLASSES, *OWN_SHARE_CLASSES)


def build_player_asof(pops_by_season: dict[int, dict[str, pd.DataFrame]],
                      minutes: pd.DataFrame | None = None,
                      positions: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (season, player, game) carrying every as-of player input.

    Leak safety: every column is an expanding sum over that player's games
    STRICTLY BEFORE the current one within the season
    (`prob_metrics.expanding_asof`), so no row can see its own game, and the
    prior-season columns are a COMPLETED season. This is
    `usage.build_player_asof` generalised from five offensive event classes to
    the five attribution targets (candidate-set exposures) plus the three
    own-share classes the binaries need.

    Seasons are pooled so that a season's prior-season columns can be built from
    the season before it when that season is present."""
    frames = []
    for season in sorted(pops_by_season):
        pops = pops_by_season[season]
        for t in CHOICE_TARGETS:
            d = usable(pops[POP_OF[t]], t)
            if len(d):
                frames.append(_cand_long(d, N_ALT_OF[t], RATE_OF[t]))
        for b in BINARY_TARGETS:
            p = pops[POP_OF[b]]
            p = p[p["five_ok"].to_numpy()]
            if len(p):
                frames.append(_own_long(p, OWN_SHARE_OF[b]))
    lg = pd.concat(frames, ignore_index=True)
    del frames

    pg = lg.groupby(["season", "player_id", "game_id", "rate_class"],
                    as_index=False).agg(opp=("credited", "size"),
                                        num=("credited", "sum"),
                                        game_date=("game_date", "first"),
                                        team_id=("team_id", "first"))
    del lg
    wide = pg.pivot_table(index=["season", "player_id", "game_id"],
                          columns="rate_class", values=["opp", "num"],
                          fill_value=0.0, aggfunc="sum")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for c in ALL_CLASSES:
        for stem in ("opp", "num"):
            col = f"{stem}_{c}"
            if col not in wide.columns:
                wide[col] = 0.0
    meta = pg.groupby(["season", "player_id", "game_id"], as_index=False).agg(
        game_date=("game_date", "first"), team_id=("team_id", "first"))
    pg = meta.merge(wide, on=["season", "player_id", "game_id"], how="left")
    del wide, meta

    if minutes is not None and len(minutes):
        pg = pg.merge(minutes, on=["season", "player_id", "game_id"], how="left")
    else:
        pg["minutes"] = np.nan
    minutes_join_pct = round(float(pg["minutes"].notna().mean() * 100), 4)
    pg["minutes_f"] = pg["minutes"].fillna(0.0)
    pg["minutes_known"] = pg["minutes"].notna().astype("float64")

    pg = pg.sort_values(["season", "player_id", "game_date", "game_id"],
                        kind="stable").reset_index(drop=True)
    cnt_cols = [f"{s}_{c}" for c in ALL_CLASSES for s in ("opp", "num")]
    sum_cols = [*cnt_cols, "minutes_f", "minutes_known"]
    asof = PM.expanding_asof(pg, ["season", "player_id"], sum_cols)
    out = pd.concat([pg[["season", "player_id", "game_id", "game_date"]], asof],
                    axis=1)
    out = out.rename(columns={"minutes_f": "minutes_asof",
                              "minutes_known": "minutes_games_asof",
                              "n_prior": "games_asof"})

    # ---- prior season totals (a COMPLETED season) --------------------------
    tot = pg.groupby(["season", "player_id"], as_index=False)[cnt_cols].sum()
    tot["season"] = tot["season"] + 1
    tot = tot.rename(columns={c: f"prev_{c}" for c in cnt_cols})
    out = out.merge(tot, on=["season", "player_id"], how="left")

    modal = (pg.groupby(["season", "player_id"])["team_id"]
             .agg(lambda x: x.value_counts().index[0] if len(x.dropna()) else np.nan)
             .reset_index().rename(columns={"team_id": "prev_team_id"}))
    modal["season"] = modal["season"] + 1
    out = out.merge(modal, on=["season", "player_id"], how="left")

    # ---- position group ----------------------------------------------------
    pos = positions if positions is not None else load_positions()
    pos = pos.rename(columns={"shooter_id": "player_id"})
    out = out.merge(pos, on="player_id", how="left")
    out["position_group"] = out["position_group"].fillna("UNK")

    # ---- league and position as-of priors, by calendar date ----------------
    day = pg.groupby(["season", "game_date"], as_index=False)[cnt_cols].sum()
    day = day.sort_values(["season", "game_date"], kind="stable").reset_index(drop=True)
    lga = PM.expanding_asof(day, ["season"], cnt_cols)
    lga.columns = [f"lg_{c}" for c in lga.columns]
    day = pd.concat([day[["season", "game_date"]], lga], axis=1)
    out = out.merge(day, on=["season", "game_date"], how="left")

    pgp = pg.merge(pos, on="player_id", how="left")
    pgp["position_group"] = pgp["position_group"].fillna("UNK")
    pday = pgp.groupby(["season", "position_group", "game_date"],
                       as_index=False)[cnt_cols].sum()
    pday = pday.sort_values(["season", "position_group", "game_date"],
                            kind="stable").reset_index(drop=True)
    pa = PM.expanding_asof(pday, ["season", "position_group"], cnt_cols)
    pa.columns = [f"pos_{c}" for c in pa.columns]
    pday = pd.concat([pday[["season", "position_group", "game_date"]], pa], axis=1)
    out = out.merge(pday, on=["season", "position_group", "game_date"], how="left")
    del pg, pgp, pday, day

    # ---- rates -------------------------------------------------------------
    for c in ALL_CLASSES:
        out[f"rate_{c}"] = _safe_div(out[f"num_{c}"].to_numpy(dtype="float64"),
                                     out[f"opp_{c}"].to_numpy(dtype="float64"))
        out[f"lg_rate_{c}"] = _safe_div(out[f"lg_num_{c}"].to_numpy(dtype="float64"),
                                        out[f"lg_opp_{c}"].to_numpy(dtype="float64"))
        out[f"pos_rate_{c}"] = _safe_div(out[f"pos_num_{c}"].to_numpy(dtype="float64"),
                                         out[f"pos_opp_{c}"].to_numpy(dtype="float64"))
        out[f"prev_rate_{c}"] = _safe_div(
            out[f"prev_num_{c}"].to_numpy(dtype="float64"),
            out[f"prev_opp_{c}"].to_numpy(dtype="float64"))
    # A season's opening day has nothing strictly earlier, so its as-of league
    # and position priors are undefined. They are back-filled from that season's
    # own next available date on the date-sorted table -- the same opening-day
    # rule `free_throw` and `usage` use, and the only forward-looking value in
    # the module. It touches opening day alone.
    order = out.sort_values(["season", "game_date"], kind="stable").index
    for c in ALL_CLASSES:
        for pre, grp in (("lg", ["season"]), ("pos", ["season", "position_group"])):
            col = f"{pre}_rate_{c}"
            out[col] = out.loc[order].groupby(grp)[col].bfill().reindex(out.index)
            out[col] = out[col].fillna(out[col].median())

    out["has_prior_season"] = np.isfinite(
        out[[f"prev_opp_{c}" for c in ALL_CLASSES]].to_numpy(dtype="float64")
    ).any(axis=1).astype("float64")
    for c in ALL_CLASSES:
        out[f"prev_opp_{c}"] = out[f"prev_opp_{c}"].fillna(0.0)
        out[f"prev_num_{c}"] = out[f"prev_num_{c}"].fillna(0.0)
    mpg = _safe_div(out["minutes_asof"].to_numpy(dtype="float64"),
                    out["minutes_games_asof"].to_numpy(dtype="float64"))
    out["minutes_per_game_asof"] = np.nan_to_num(mpg, nan=0.0)
    out.attrs["minutes_join_pct"] = minutes_join_pct
    return out


# ===========================================================================
# 4. As-of TEAM features (the three binaries)
# ===========================================================================
#: Per binary, the (numerator, denominator) count pair of the OFFENCE's own rate
#: and of the DEFENCE's rate allowed.
TEAM_RATES: dict[str, dict[str, tuple[str, str]]] = {
    "assisted": {"off": ("ast", "made"), "def": ("ast_allowed", "opp_made")},
    "stolen": {"off": ("tov_stolen", "tov"), "def": ("steals", "opp_tov")},
    "blocked": {"off": ("miss_blocked", "miss"), "def": ("blocks", "opp_miss")},
}
TEAM_COUNT_COLS: tuple[str, ...] = (
    "made", "ast", "opp_made", "ast_allowed", "tov", "tov_stolen", "opp_tov",
    "steals", "miss", "miss_blocked", "opp_miss", "blocks")


def build_team_asof(pops_by_season: dict[int, dict[str, pd.DataFrame]]
                    ) -> pd.DataFrame:
    """One row per (season, team, game) with the team's as-of offensive and
    defensive counts for all three binaries, plus the league as-of rate on the
    same date for the shrinkage prior.

    Built from the SAME population tables the targets are, so a team's
    denominator is exactly the set of events the binary is graded on, and every
    column is an expanding sum over that team's strictly earlier games.

    Each season's six (pop x side) groups are merged COLUMN-WISE first (their
    names are unique within one season -- `made`/`ast`/`opp_made`/... -- so an
    outer merge on the game key is correct); seasons are then stacked ROW-WISE.
    Merging every season's groups together in one flat sequential merge (the
    original code) re-uses the same 12 column names for 2024 and 2025 alike, so
    pandas silently suffixes every one of them `_x`/`_y`, the bare names never
    exist, the `if c not in tg.columns` fallback below manufactures an all-zero
    column for literally every count, and the league-rate `_safe_div(0, 0)`
    that follows is NaN on every row for all three binaries -- confirmed by
    reproducing the merge standalone and finding only `_x`/`_y`-suffixed
    columns in `tg`, never the bare ones."""
    spec = (("made_fga", "made", "ast", "opp_made", "ast_allowed"),
            ("tov", "tov", "tov_stolen", "opp_tov", "steals"),
            ("miss_fga", "miss", "miss_blocked", "opp_miss", "blocks"))
    season_frames = []
    for season in sorted(pops_by_season):
        pops = pops_by_season[season]
        rows = []
        for pop, off_den, off_num, def_den, def_num in spec:
            p = pops[pop]
            base = pd.DataFrame({
                "season": p["season"].to_numpy(), "game_id": p["game_id"].to_numpy(),
                "game_date": p["game_date"].to_numpy(),
                "b": p["b"].to_numpy().astype("float64")})
            for team_col, den, num in ((p["off_team_id"], off_den, off_num),
                                       (p["def_team_id"], def_den, def_num)):
                g = base.assign(team_id=team_col.to_numpy())
                g = g.groupby(["season", "team_id", "game_id", "game_date"],
                              as_index=False).agg(d=("b", "size"), n=("b", "sum"))
                rows.append(g.rename(columns={"d": den, "n": num}))
        tgs = rows[0]
        for r in rows[1:]:
            tgs = tgs.merge(r, on=["season", "team_id", "game_id", "game_date"], how="outer")
        season_frames.append(tgs)
    tg = pd.concat(season_frames, ignore_index=True)
    for c in TEAM_COUNT_COLS:
        if c not in tg.columns:
            tg[c] = 0.0
        tg[c] = tg[c].fillna(0.0)
    tg = tg.sort_values(["season", "team_id", "game_date", "game_id"],
                        kind="stable").reset_index(drop=True)
    asof = PM.expanding_asof(tg, ["season", "team_id"], list(TEAM_COUNT_COLS))
    out = pd.concat([tg[["season", "team_id", "game_id", "game_date"]], asof], axis=1)
    out = out.rename(columns={"n_prior": "team_games_asof"})

    day = tg.groupby(["season", "game_date"], as_index=False)[list(TEAM_COUNT_COLS)].sum()
    day = day.sort_values(["season", "game_date"], kind="stable").reset_index(drop=True)
    lg = PM.expanding_asof(day, ["season"], list(TEAM_COUNT_COLS))
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)
    out = out.merge(day, on=["season", "game_date"], how="left")

    for b, sides in TEAM_RATES.items():
        for side, (num, den) in sides.items():
            out[f"{side}_{b}_num"] = out[num].to_numpy(dtype="float64")
            out[f"{side}_{b}_den"] = out[den].to_numpy(dtype="float64")
            out[f"{side}_{b}_lg"] = _safe_div(out[f"lg_{num}"].to_numpy(dtype="float64"),
                                              out[f"lg_{den}"].to_numpy(dtype="float64"))
    order = out.sort_values(["season", "game_date"], kind="stable").index
    for b, sides in TEAM_RATES.items():
        for side in sides:
            col = f"{side}_{b}_lg"
            out[col] = out.loc[order].groupby("season")[col].bfill().reindex(out.index)
            out[col] = out[col].fillna(out[col].median())
    return out


def team_shrunk(d: pd.DataFrame, binary: str, side: str, m: float) -> np.ndarray:
    """One team's as-of rate for one binary, shrunk toward the LEAGUE as-of rate
    with `m` pseudo events. Same formula and same unit (pseudo opportunities) as
    the player-level `shrunk_rate`; a team with no history yet sits exactly at
    the league rate rather than at a NaN."""
    num = d[f"{side}_{binary}_num"].to_numpy(dtype="float64")
    den = d[f"{side}_{binary}_den"].to_numpy(dtype="float64")
    prior = d[f"{side}_{binary}_lg"].to_numpy(dtype="float64")
    return (m * prior + num) / (m + den)


# ===========================================================================
# 5. The per-target designs
# ===========================================================================
#: Columns of the player as-of table carried onto every candidate slot. The
#: pre-registration's player inputs are the as-of rate, the prior-season rate and
#: the position group; `games_asof` / `minutes_asof` / `prev_team_id` ride along
#: for the REPORTED subsets (transfer, no-prior-season) and are NOT features of
#: any arm -- `features.md` section 4 lists what each arm actually sees.
SLOT_COLS: tuple[str, ...] = (
    "games_asof", "minutes_asof", "minutes_per_game_asof", "has_prior_season",
    "prev_team_id", "position_group",
)


def _rate_cols(target: str) -> list[str]:
    c = RATE_OF[target]
    return [f"opp_{c}", f"num_{c}", f"lg_rate_{c}", f"pos_rate_{c}",
            f"prev_rate_{c}", f"prev_opp_{c}"]


def build_choice_design(events: pd.DataFrame, asof: pd.DataFrame,
                        target: str) -> pd.DataFrame:
    """One row per modelled event of `target`, with the candidates' as-of
    features attached as `*_1 .. *_K` columns.

    The wide form is `usage.build_usage_design`'s, for the same reason: the
    choice is among a fixed K, so an (n, K) block is the natural shape for the
    proportional and conditional-logit arms and keeps the design 1/K the size of
    the long form the tree needs (`long_frame` builds that on demand)."""
    if target not in CHOICE_TARGETS:
        raise KeyError(f"{target!r} is not a choice target")
    k_alt = N_ALT_OF[target]
    keep = ["season", "player_id", "game_id", *_rate_cols(target), *SLOT_COLS]
    a = asof[list(dict.fromkeys(keep))].copy()
    a["position_code"] = a["position_group"].map(
        {p: i for i, p in enumerate(POSITION_LEVELS)}).fillna(
        len(POSITION_LEVELS) - 1).astype("int8")
    a = a.drop(columns=["position_group"])

    out = events.reset_index(drop=True)
    for k in range(1, k_alt + 1):
        m = a.rename(columns={"player_id": f"cand_{k}"})
        m = m.rename(columns={c: f"{c}_{k}" for c in m.columns
                              if c not in ("season", "game_id", f"cand_{k}")})
        out = out.merge(m, on=["season", "game_id", f"cand_{k}"], how="left")

    c = RATE_OF[target]
    zero_fill = (f"opp_{c}", f"num_{c}", f"prev_opp_{c}", "games_asof",
                 "minutes_asof", "minutes_per_game_asof", "has_prior_season")
    for k in range(1, k_alt + 1):
        for col in zero_fill:
            out[f"{col}_{k}"] = out[f"{col}_{k}"].fillna(0.0)
        out[f"position_code_{k}"] = out[f"position_code_{k}"].fillna(
            len(POSITION_LEVELS) - 1)
    # A candidate with no as-of row at all (his first game of the season) has
    # zero exposure, so his shrunk rate is entirely the prior -- which is what
    # the formula does with a zero denominator anyway. The prior itself is taken
    # from the lineup's other members and then from the column median, so a
    # lineup of five debutants still has a defined prior rather than a NaN.
    for col in (f"lg_rate_{c}", f"pos_rate_{c}"):
        cols = [f"{col}_{k}" for k in range(1, k_alt + 1)]
        ref = out[cols].median(axis=1)
        med = float(np.nanmedian(out[cols].to_numpy(dtype="float64")))
        for cc in cols:
            out[cc] = out[cc].fillna(ref).fillna(med)
    out = out.copy()                 # de-fragment before the derived columns go on
    for k in range(1, k_alt + 1):
        out[f"prev_rate_{c}_{k}"] = out[f"prev_rate_{c}_{k}"].fillna(
            out[f"pos_rate_{c}_{k}"])
        out[f"is_transfer_{k}"] = (
            np.isfinite(out[f"prev_team_id_{k}"].to_numpy(dtype="float64"))
            & (out[f"prev_team_id_{k}"].to_numpy(dtype="float64")
               != out["cand_team_id"].to_numpy())
            & (out[f"has_prior_season_{k}"].to_numpy() > 0)).astype("int8")
    out["shot_class_code"] = out["shot_class"].map(SHOT_CLASS_CODE).fillna(
        SHOT_CLASS_CODE["other"]).astype("int8")
    out["target"] = target
    out["n_alt"] = k_alt
    return out


def _block(d: pd.DataFrame, stem: str, k_alt: int) -> np.ndarray:
    return d[[f"{stem}_{k}" for k in range(1, k_alt + 1)]].to_numpy(dtype="float64")


def _prior_block(d: pd.DataFrame, target: str, prior_kind: str,
                 k_alt: int) -> np.ndarray:
    c = RATE_OF[target]
    if prior_kind == "league":
        return _block(d, f"lg_rate_{c}", k_alt)
    if prior_kind == "position":
        return _block(d, f"pos_rate_{c}", k_alt)
    if prior_kind == "prior_season":
        # returners use their own completed prior season; newcomers have none and
        # fall back to the POSITION prior, which is exactly what "prior-season
        # rate for returners, position group prior for new players" says.
        return _block(d, f"prev_rate_{c}", k_alt)
    raise KeyError(f"unknown prior kind {prior_kind!r}")


def shrunk_rate(d: pd.DataFrame, target: str, prior_kind: str,
                m: float) -> np.ndarray:
    """(n, K) shrunk as-of rate for every candidate.

    `(m * prior + credits) / (m + opportunities)`: `m` is in pseudo
    OPPORTUNITIES OF THIS TARGET'S OWN POPULATION, the same unit as the
    denominator, so "m = 100" literally means "100 opportunities of history
    before a player's own rate outweighs the prior"."""
    k_alt = N_ALT_OF[target]
    c = RATE_OF[target]
    num = _block(d, f"num_{c}", k_alt)
    den = _block(d, f"opp_{c}", k_alt)
    return (m * _prior_block(d, target, prior_kind, k_alt) + num) / (m + den)


def normalise(r: np.ndarray) -> np.ndarray:
    """`usage.normalise`: rows normalised to sum to one, a non-finite or all-zero
    row falling back to the uniform over the candidate set -- the only honest
    answer when the model has no information about a lineup."""
    return U.normalise(r)


# ---------------------------------------------------------------------------
# P1: proportional
# ---------------------------------------------------------------------------
def p1_probs(d: pd.DataFrame, target: str, prior_kind: str, m: float) -> np.ndarray:
    return normalise(shrunk_rate(d, target, prior_kind, m))


def prior_season_available(d: pd.DataFrame, target: str) -> bool:
    """Does this slice carry ANY prior-season history? False on the F1 TRAINING
    fold by construction (L13: on-floor ids start in 2024), which is why
    `fit_shrinkage` reports the prior-season rung as unidentified there instead
    of scoring it."""
    return bool(_block(d, "has_prior_season", N_ALT_OF[target]).max() > 0)


def fit_shrinkage(tr: pd.DataFrame, target: str,
                  grid: tuple[float, ...] = SHRINK_GRID,
                  kinds: tuple[str, ...] = PRIOR_KINDS) -> dict:
    """Grid over (prior, strength) on the TRAINING fold's own log loss.

    Honest in-fold for the reason `usage.fit_shrinkage` and `free_throw.fit_eb`
    state: every quantity the arm uses is an as-of feature, so a training-fold
    log loss is already a walk-forward number. A prior whose column is
    identically absent in the training fold is reported as `unidentified` and
    excluded from the choice rather than silently winning a tie."""
    y = tr["y"].to_numpy()
    have_prev = prior_season_available(tr, target)
    rows, best = [], None
    for kind in kinds:
        if kind == "prior_season" and not have_prev:
            rows.append({"prior": kind, "m": None, "train_log_loss": None,
                         "status": "unidentified (no prior-season history in this fold)"})
            continue
        for mm in grid:
            ll = PM.log_loss(y, p1_probs(tr, target, kind, float(mm)))
            rows.append({"prior": kind, "m": float(mm),
                         "train_log_loss": round(ll, 6), "status": "ok"})
            if best is None or ll < best["train_log_loss"]:
                best = rows[-1]
    return {"grid": rows, "best": best, "prior_season_available": have_prev}


# ---------------------------------------------------------------------------
# P2: conditional logit
# ---------------------------------------------------------------------------
#: The alternative-varying design. State variables are alternative-INVARIANT and
#: cancel in a conditional logit, so they enter only as interactions with the
#: alternative's own share and its position indicator -- the only way the
#: pre-registered state features are identified at all (`usage` decision 9). The
#: feature list is exactly the pre-registration's: as-of rate (as the normalised
#: log share), prior-season rate, position group, shot class, score diff,
#: seconds remaining.
CL_FEATURES: tuple[str, ...] = (
    "log_share", "log_prior_share", "is_G", "is_F", "is_C",
    "log_share_x_scorediff", "log_share_x_sec",
    "log_share_x_rim", "log_share_x_three", "log_share_x_ft",
    "is_C_x_scorediff", "is_C_x_sec", "is_C_x_rim", "is_C_x_three", "is_C_x_ft",
)

#: Features that READ the prior-season block. On a fold whose training slice has
#: no prior-season history at all (F1 train = 2024, L13) these are not merely
#: constant: `build_choice_design` falls `prev_rate_*` back to the POSITION
#: prior, so the column still VARIES across the candidates and a naive fit
#: happily estimates a coefficient for a variable that MEANS something different
#: in the test fold. `usage` measured the cost of not catching this at 3.74
#: against 1.46 on one class.
PRIOR_SEASON_FEATURES: tuple[str, ...] = ("log_prior_share", "prior_rate")


def unidentified_features(tr: pd.DataFrame, target: str) -> list[str]:
    """Feature names whose MEANING is not identified by this training fold."""
    return [] if prior_season_available(tr, target) else list(PRIOR_SEASON_FEATURES)


def cl_design(d: pd.DataFrame, target: str, prior_kind: str,
              m: float) -> tuple[np.ndarray, list[str]]:
    """(n, K, F) conditional-logit design, centred within the choice set."""
    k_alt = N_ALT_OF[target]
    q = normalise(shrunk_rate(d, target, prior_kind, m))
    log_share = np.log(np.maximum(q, 1e-9))
    c = RATE_OF[target]
    log_prev = np.log(np.maximum(normalise(_block(d, f"prev_rate_{c}", k_alt)), 1e-9))
    pos = _block(d, "position_code", k_alt)
    is_c = (pos == POSITION_LEVELS.index("C")).astype("float64")
    sd = (d["score_diff"].to_numpy(dtype="float64") / 10.0)[:, None]
    sec = (d["sec_remaining"].to_numpy(dtype="float64") / 600.0)[:, None]
    sc = d["shot_class_code"].to_numpy()
    rim = (sc == SHOT_CLASS_CODE["FGA_rim"]).astype("float64")[:, None]
    three = (sc == SHOT_CLASS_CODE["FGA_3"]).astype("float64")[:, None]
    ft = (sc == SHOT_CLASS_CODE["FT_missed"]).astype("float64")[:, None]
    cols = {
        "log_share": log_share,
        "log_prior_share": log_prev,
        "is_G": (pos == POSITION_LEVELS.index("G")).astype("float64"),
        "is_F": (pos == POSITION_LEVELS.index("F")).astype("float64"),
        "is_C": is_c,
        "log_share_x_scorediff": log_share * sd,
        "log_share_x_sec": log_share * sec,
        "log_share_x_rim": log_share * rim,
        "log_share_x_three": log_share * three,
        "log_share_x_ft": log_share * ft,
        "is_C_x_scorediff": is_c * sd,
        "is_C_x_sec": is_c * sec,
        "is_C_x_rim": is_c * rim,
        "is_C_x_three": is_c * three,
        "is_C_x_ft": is_c * ft,
    }
    names = list(CL_FEATURES)
    X = np.stack([cols[n] for n in names], axis=2)
    return X - X.mean(axis=1, keepdims=True), names


def drop_constant_features(X: np.ndarray, names: list[str]
                           ) -> tuple[np.ndarray, list[str], list[str]]:
    """Remove columns with no within-choice-set variation in THIS fold -- a
    conditional logit cannot identify one (it cancels). The `steal` target has no
    shot class at all, so its three shot-class interactions are dropped here and
    the drop is recorded by the trainer rather than passing silently."""
    return U.drop_constant_features(X, names)


#: `usage.CondLogitArm` is shape-generic in K (its design is (n, K, F)), so it is
#: imported rather than re-derived.
CondLogitArm = U.CondLogitArm


# ---------------------------------------------------------------------------
# P3: LightGBM grouped-softmax ranker
# ---------------------------------------------------------------------------
#: The tree sees the same pre-registered inputs, with the state and the shot
#: class RAW: a tree can interact a raw state variable with an alternative
#: feature itself, so -- unlike in the logit -- nothing has to be pre-interacted
#: to be identified. `share` and `rank` are monotone transforms of the as-of rate
#: WITHIN the choice set, not extra inputs.
LGBM_ALT_FEATURES: tuple[str, ...] = ("share", "rate", "prior_rate",
                                      "position_code", "rate_rank")
LGBM_STATE_FEATURES: tuple[str, ...] = ("score_diff", "sec_remaining",
                                        "shot_class_code")
LGBM_FEATURES: tuple[str, ...] = (*LGBM_ALT_FEATURES, *LGBM_STATE_FEATURES)

LGBM_PARAM_GRID: tuple[dict, ...] = (
    dict(num_leaves=15, learning_rate=0.08, n_estimators=300, min_child_samples=200),
    dict(num_leaves=31, learning_rate=0.06, n_estimators=400, min_child_samples=400),
    dict(num_leaves=63, learning_rate=0.05, n_estimators=500, min_child_samples=800),
)


def long_frame(d: pd.DataFrame, target: str, prior_kind: str,
               m: float) -> tuple[np.ndarray, np.ndarray]:
    """(Kn, F) long design for the tree arm and its (Kn,) chosen flag. Row order
    is event-major, slot-minor, so `reshape(n, K)` recovers the choice sets."""
    k_alt = N_ALT_OF[target]
    c = RATE_OF[target]
    r = shrunk_rate(d, target, prior_kind, m)
    q = normalise(r)
    rank = np.argsort(np.argsort(-q, axis=1, kind="stable"), axis=1).astype("float64")
    blocks = {"share": q, "rate": r,
              "prior_rate": _block(d, f"prev_rate_{c}", k_alt),
              "position_code": _block(d, "position_code", k_alt),
              "rate_rank": rank}
    n = len(d)
    X = np.empty((n * k_alt, len(LGBM_FEATURES)), dtype="float32")
    for j, name in enumerate(LGBM_ALT_FEATURES):
        X[:, j] = blocks[name].reshape(-1)
    off = len(LGBM_ALT_FEATURES)
    for j, name in enumerate(LGBM_STATE_FEATURES):
        X[:, off + j] = np.repeat(d[name].to_numpy(dtype="float64"), k_alt)
    chosen = np.zeros(n * k_alt, dtype="int8")
    chosen[np.arange(n) * k_alt + d["y"].to_numpy()] = 1
    return X, chosen


def group_softmax_objective(k_alt: int):
    """LightGBM objective: the CONDITIONAL likelihood of a choice among `k_alt`.

    `usage._group_softmax_objective` is the same function hard-wired to five;
    this is the K-generic factory, because the assist target has four
    alternatives. Rows arrive event-major and slot-minor, so `reshape(-1, K)` is
    the choice set, and the gradient of the grouped-softmax log likelihood is
    `p - y` with diagonal Hessian `p (1 - p)` inside each group.

    This is what P3 has to optimise to be the arm the pre-registration names. A
    plain binary objective softmaxed afterwards optimises a DIFFERENT likelihood
    and comes out systematically over-sharpened -- measured in the usage bake-off
    at a 2.24 pp worst calibration gap on an arm whose log loss looked fine. The
    fix is the objective, not a temperature fitted on the model's own output;
    the latter is the shape `docs/SIM_GUARDRAILS.md` section 5 bans."""
    def _obj(y_true: np.ndarray, y_pred: np.ndarray):
        z = np.asarray(y_pred, dtype="float64").reshape(-1, k_alt)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        grad = p - np.asarray(y_true, dtype="float64").reshape(-1, k_alt)
        hess = np.maximum(p * (1.0 - p), 1e-6)
        return grad.reshape(-1), hess.reshape(-1)
    return _obj


class LgbmChoiceArm:
    """LightGBM scoring each alternative, softmaxed within the choice set.

    `usage.LgbmChoiceArm` with K a constructor argument instead of a module
    constant. Offset by `log q` (`lgbm_init_score`) so that at zero trees the arm
    IS P1 and what the tree fits is a multiplicative correction to the as-of
    share; without the offset the comparison would be "can a tree relearn a
    normalisation", which is not the pre-registered question.

    Row bagging is off: a row here is one alternative of a choice set, so
    `subsample` would split choice sets across the in-bag boundary. Column
    sampling is unaffected and is kept."""

    BASE = dict(colsample_bytree=0.9, reg_lambda=1.0, verbose=-1,
                boost_from_average=False)

    def __init__(self, n_alt: int, params: dict | None = None, seed: int = 0):
        self.n_alt = int(n_alt)
        self.params = dict(params or {})
        self.seed = int(seed)

    def fit(self, X: np.ndarray, chosen: np.ndarray,
            init: np.ndarray | None = None) -> LgbmChoiceArm:
        import lightgbm as lgb
        self.clf_ = lgb.LGBMRegressor(random_state=self.seed,
                                      objective=group_softmax_objective(self.n_alt),
                                      **{**self.BASE, **self.params})
        kw = {} if init is None else {"init_score": np.asarray(init, dtype="float64")}
        self.clf_.fit(X, np.asarray(chosen, dtype="float64"), **kw)
        return self

    def predict_proba(self, X: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
        raw = np.asarray(self.clf_.predict(X, raw_score=True), dtype="float64")
        if init is not None:
            raw = raw + np.asarray(init, dtype="float64")
        raw = raw.reshape(-1, self.n_alt)
        raw = raw - raw.max(axis=1, keepdims=True)
        e = np.exp(raw)
        return e / e.sum(axis=1, keepdims=True)


def lgbm_init_score(d: pd.DataFrame, target: str, prior_kind: str,
                    m: float) -> np.ndarray:
    """(Kn,) offset `log(P1 share)` for the tree arm (class docstring)."""
    q = normalise(shrunk_rate(d, target, prior_kind, m))
    return np.log(np.maximum(q, 1e-9)).reshape(-1)


# ===========================================================================
# 6. The binary designs and arms
# ===========================================================================
#: The pre-registration's binary inputs. `team_ridge` sees the team block only;
#: `aware_ridge` adds the individual offensive player's own as-of share and the
#: shot class; the tree sees the union. Home/neutral is in every arm --
#: `CLAUDE.md`, "home/away/neutral is a first-class feature in every
#: scoring-stage model".
TEAM_FEATURES: tuple[str, ...] = ("off_rate_c", "def_rate_c", "score_diff",
                                  "sec_remaining", "cand_is_home", "neutral_site")
AWARE_FEATURES: tuple[str, ...] = ("own_share_c", "sc_rim", "sc_three")
BINARY_FEATURES: dict[str, tuple[str, ...]] = {
    "team_ridge": TEAM_FEATURES,
    "aware_ridge": (*TEAM_FEATURES, *AWARE_FEATURES),
    "lgbm": (*TEAM_FEATURES, *AWARE_FEATURES),
}
BINARY_SHRINK_GRID: tuple[float, ...] = (5, 10, 25, 50, 100, 200, 400, 800)


def build_binary_design(pop: pd.DataFrame, asof: pd.DataFrame,
                        team_asof: pd.DataFrame, binary: str) -> pd.DataFrame:
    """One row per event of the binary's population, with the offence's and the
    defence's as-of team blocks and the offensive player's own as-of share."""
    if binary not in BINARY_TARGETS:
        raise KeyError(f"{binary!r} is not a binary target")
    d = pop.reset_index(drop=True).copy()
    t = team_asof[["season", "team_id", "game_id", "team_games_asof",
                   *[f"{s}_{binary}_{w}" for s in ("off", "def")
                     for w in ("num", "den", "lg")]]]
    for side, key in (("off", "off_team_id"), ("def", "def_team_id")):
        cols = [f"{side}_{binary}_{w}" for w in ("num", "den", "lg")]
        m = t[["season", "team_id", "game_id", "team_games_asof", *cols]].rename(
            columns={"team_id": key, "team_games_asof": f"{side}_games_asof"})
        d = d.merge(m, on=["season", key, "game_id"], how="left")
    share = OWN_SHARE_OF[binary]
    a = asof[["season", "player_id", "game_id", f"opp_{share}", f"num_{share}",
              f"lg_rate_{share}", f"pos_rate_{share}", f"prev_rate_{share}",
              "has_prior_season", "position_group"]].rename(
        columns={"player_id": "off_player_id"})
    d = d.merge(a, on=["season", "off_player_id", "game_id"], how="left")
    for c in (f"opp_{share}", f"num_{share}"):
        d[c] = d[c].fillna(0.0)
    for c in (f"lg_rate_{share}", f"pos_rate_{share}"):
        d[c] = d[c].fillna(d[c].median())
    d[f"prev_rate_{share}"] = d[f"prev_rate_{share}"].fillna(d[f"pos_rate_{share}"])
    d["has_prior_season"] = d["has_prior_season"].fillna(0.0)
    d["position_group"] = d["position_group"].fillna("UNK")
    # A team with no as-of history yet sits exactly at the league rate; the
    # league rate itself is back-filled on opening day by `build_team_asof`.
    for side in ("off", "def"):
        for w in ("num", "den"):
            d[f"{side}_{binary}_{w}"] = d[f"{side}_{binary}_{w}"].fillna(0.0)
        lgc = f"{side}_{binary}_lg"
        d[lgc] = d[lgc].fillna(d[lgc].median())
        d[f"{side}_games_asof"] = d[f"{side}_games_asof"].fillna(0.0)
    sc = d["shot_class"].map(SHOT_CLASS_CODE).fillna(SHOT_CLASS_CODE["other"])
    d["shot_class_code"] = sc.astype("int8")
    d["sc_rim"] = (sc == SHOT_CLASS_CODE["FGA_rim"]).astype("float64")
    d["sc_three"] = (sc == SHOT_CLASS_CODE["FGA_3"]).astype("float64")
    d["cand_is_home"] = d["cand_is_home"].astype("float64")
    d["neutral_site"] = d["neutral_site"].astype("float64")
    d["target"] = binary
    return d


def own_share_shrunk(d: pd.DataFrame, binary: str, prior_kind: str,
                     m: float) -> np.ndarray:
    """The offensive player's own as-of share for this binary, shrunk toward the
    prior the grid chose, in the same pseudo-opportunity unit as everything else."""
    share = OWN_SHARE_OF[binary]
    num = d[f"num_{share}"].to_numpy(dtype="float64")
    den = d[f"opp_{share}"].to_numpy(dtype="float64")
    if prior_kind == "league":
        prior = d[f"lg_rate_{share}"].to_numpy(dtype="float64")
    elif prior_kind == "position":
        prior = d[f"pos_rate_{share}"].to_numpy(dtype="float64")
    elif prior_kind == "prior_season":
        prior = d[f"prev_rate_{share}"].to_numpy(dtype="float64")
    else:
        raise KeyError(f"unknown prior kind {prior_kind!r}")
    return (m * prior + num) / (m + den)


def binary_matrix(d: pd.DataFrame, binary: str, fit: dict,
                  features: tuple[str, ...]) -> np.ndarray:
    """The feature matrix of one binary arm, every rate CENTRED on the league
    as-of rate of the same date (`CLAUDE.md`: "every rating feature is expressed
    relative to its own snapshot's league mean; raw levels are banned")."""
    cols = {}
    for side in ("off", "def"):
        lg = d[f"{side}_{binary}_lg"].to_numpy(dtype="float64")
        cols[f"{side}_rate_c"] = team_shrunk(d, binary, side, fit["m_team"]) - lg
    share = OWN_SHARE_OF[binary]
    cols["own_share_c"] = (own_share_shrunk(d, binary, fit["own_prior"], fit["m_own"])
                           - d[f"lg_rate_{share}"].to_numpy(dtype="float64"))
    for f in ("score_diff", "sec_remaining", "cand_is_home", "neutral_site",
              "sc_rim", "sc_three"):
        cols[f] = d[f].to_numpy(dtype="float64")
    return np.ascontiguousarray(
        np.column_stack([cols[f] for f in features]), dtype="float64")


class BinaryRidgeArm:
    """Logistic ridge. Features are standardised on the TRAIN slice only and the
    fitted means/SDs travel with the model, so the sim applies the identical
    transform (`rebound.RidgeLogitArm`'s contract)."""

    def __init__(self, C: float = 1.0, max_iter: int = 400, seed: int = 0):
        self.C, self.max_iter, self.seed = float(C), int(max_iter), int(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> BinaryRidgeArm:
        from sklearn.linear_model import LogisticRegression
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ < 1e-8] = 1.0
        self.clf_ = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                       solver="lbfgs", random_state=self.seed)
        self.clf_.fit((X - self.mu_) / self.sd_, np.asarray(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.clf_.predict_proba((X - self.mu_) / self.sd_)
        return p[:, list(self.clf_.classes_).index(1)]


class BinaryLgbmArm:
    """LightGBM binary. Hyperparameters are FIXED across the three binaries and
    both folds, so the comparison is of feature bundles and model classes rather
    than of tuning effort -- the discipline `rebound.LgbmArm` uses."""

    PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.06,
                  num_leaves=63, min_child_samples=400, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
                  verbose=-1)

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> BinaryLgbmArm:
        import lightgbm as lgb
        self.clf_ = lgb.LGBMClassifier(random_state=self.seed, **self.PARAMS)
        self.clf_.fit(X, np.asarray(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.clf_.predict_proba(X)
        return p[:, list(self.clf_.classes_).index(1)]


def fit_binary_shrinkage(tr: pd.DataFrame, binary: str,
                         grid: tuple[float, ...] = BINARY_SHRINK_GRID) -> dict:
    """Fit the team-rate and own-share shrinkage strengths on the TRAINING fold.

    Two one-moment fits rather than a joint grid: the team strength is chosen by
    the training log loss of a logistic on the two shrunk TEAM rates alone, and
    the own-share strength (and its prior) by a logistic on the shrunk own share
    alone. Fitting them jointly would cost 9x27 refits to move a number that
    neither arm's eligibility turns on; fitting them separately keeps one moment
    per parameter, which is the rule `usage.fit_alphas` follows."""
    from sklearn.linear_model import LogisticRegression
    y = tr["b"].to_numpy()
    have_prev = bool(tr["has_prior_season"].to_numpy().max() > 0)
    rows = []

    def _ll(X):
        X = np.column_stack(X)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-8] = 1.0
        clf = LogisticRegression(max_iter=300).fit((X - mu) / sd, y)
        p = clf.predict_proba((X - mu) / sd)[:, list(clf.classes_).index(1)]
        return float(-np.mean(y * np.log(np.clip(p, EPS, 1)) +
                              (1 - y) * np.log(np.clip(1 - p, EPS, 1))))

    best_t, best_tll = None, np.inf
    for mm in grid:
        ll = _ll([team_shrunk(tr, binary, "off", float(mm)),
                  team_shrunk(tr, binary, "def", float(mm))])
        rows.append({"knob": "m_team", "prior": "league", "m": float(mm),
                     "train_log_loss": round(ll, 6)})
        if ll < best_tll:
            best_t, best_tll = float(mm), ll
    best_o, best_p, best_oll = None, None, np.inf
    for kind in PRIOR_KINDS:
        if kind == "prior_season" and not have_prev:
            rows.append({"knob": "m_own", "prior": kind, "m": None,
                         "train_log_loss": None,
                         "status": "unidentified (no prior-season history in this fold)"})
            continue
        for mm in grid:
            ll = _ll([own_share_shrunk(tr, binary, kind, float(mm))])
            rows.append({"knob": "m_own", "prior": kind, "m": float(mm),
                         "train_log_loss": round(ll, 6)})
            if ll < best_oll:
                best_o, best_p, best_oll = float(mm), kind, ll
    return {"m_team": best_t, "m_own": best_o, "own_prior": best_p,
            "grid": rows, "prior_season_available": have_prev}


# ===========================================================================
# 7. Folds
# ===========================================================================
def fold_slices(design: pd.DataFrame, fold: str = SELECTION_FOLD
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = FOLDS[fold]
    assert_not_sealed(spec["train"], context=f"{fold} train seasons")
    assert_not_sealed(spec["test"], context=f"{fold} test seasons")
    tr = design[design["season"].isin(spec["train"])]
    te = design[design["season"].isin(spec["test"])]
    assert_not_sealed(tr, context=f"{fold} train slice")
    assert_not_sealed(te, context=f"{fold} test slice")
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def walkforward_slices(design: pd.DataFrame, season: int = WF_SEASON,
                       split_date: str = WF_SPLIT_DATE
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The within-season robustness fold of the pre-registration."""
    assert_not_sealed([season], context="within-season walk-forward")
    d = design[design["season"] == season]
    assert_not_sealed(d, context="walk-forward slice")
    cut = pd.Timestamp(split_date)
    dt = pd.to_datetime(d["game_date"])
    return d[dt < cut].reset_index(drop=True), d[dt >= cut].reset_index(drop=True)


# ===========================================================================
# 8. Metrics
# ===========================================================================
#: Decision 8, as this model's pre-registration cites it: "slope ratio in
#: [0.8, 1.2] AND >= 3 of 4 monotone steps, sub-2 pp drivers exempt". The
#: exemption covers BOTH clauses, because a slope ratio measured over a
#: noise-sized span is itself noise; the gate is then evaluated on the remaining
#: drivers only and the exemption is recorded per driver. `usage` reads clause
#: (b) as 4-of-4 relaxing to 3-of-4, because ITS OWN pre-registration specified
#: 4 of 4 and Decision 8 only relaxes it; this model's pre-registration names no
#: step count of its own, so the Decision-8 minimum (3) is the requirement here.
#: The 4-of-4 reading is reported alongside every verdict so the stricter
#: reading stays auditable.
def decision8_verdict(resp: dict) -> dict:
    """Annotate a `prob_metrics.quintile_responsiveness` block with the
    Decision-8 gate. Mutates and returns `resp`."""
    span_pp = abs(float(resp["span_actual"])) * 100
    sr = resp.get("slope_ratio")
    exempt = span_pp < SMALL_SPAN_PP
    steps = int(resp["pred_monotone_steps"])
    slope_ok = sr is not None and SLOPE_BAND[0] <= float(sr) <= SLOPE_BAND[1]
    resp.update({
        "span_actual_pp": round(span_pp, 4),
        "small_span_exempt": bool(exempt),
        "steps_required": RESP_MIN_STEPS,
        "steps_pass": bool(steps >= RESP_MIN_STEPS),
        "slope_pass": bool(slope_ok),
        "strict_4of4_pass": bool(steps >= 4 and slope_ok),
        "pass": True if exempt else bool(steps >= RESP_MIN_STEPS and slope_ok),
    })
    return resp


def _alt_long(te: pd.DataFrame, p: np.ndarray, driver: np.ndarray):
    """Flatten the choice sets to (event x alternative) rows -- the axis the
    pre-registration defines the calibration and responsiveness checks on (the
    PLAYER's as-of rate, not the event)."""
    y = te["y"].to_numpy()
    obs = np.zeros_like(p, dtype="int64")
    obs[np.arange(len(y)), y] = 1
    return obs.reshape(-1), p.reshape(-1), np.asarray(driver).reshape(-1)


def share_calibration(te: pd.DataFrame, p: np.ndarray, driver: np.ndarray,
                      n_bins: int = 10) -> dict:
    """Predicted share vs actual share inside deciles of the PREDICTED
    probability, in percentage points -- the pre-registration's "calibration by
    predicted-probability decile (<= 2 pp)". The driver-decile version of the
    same table is reported alongside it by `score_choice_arm`."""
    obs, pred, _ = _alt_long(te, p, driver)
    return _decile_table(obs, pred, pred, n_bins)


def driver_calibration(te: pd.DataFrame, p: np.ndarray, driver: np.ndarray,
                       n_bins: int = 10) -> dict:
    """The same table binned on the player's own as-of rate instead."""
    obs, pred, dr = _alt_long(te, p, driver)
    return _decile_table(obs, pred, dr, n_bins)


def _decile_table(obs: np.ndarray, pred: np.ndarray, binner: np.ndarray,
                  n_bins: int) -> dict:
    edges = np.quantile(binner, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, binner, side="right") - 1, 0, n_bins - 1)
    rows, worst = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.sum():
            continue
        pp = float(pred[m].mean() * 100)
        aa = float(obs[m].mean() * 100)
        rows.append({"bin": b + 1, "n": int(m.sum()),
                     "bin_mean": round(float(binner[m].mean()), 6),
                     "pred_pct": round(pp, 4), "actual_pct": round(aa, 4),
                     "gap_pp": round(pp - aa, 4)})
        worst = max(worst, abs(pp - aa))
    return {"bins": rows, "max_abs_gap_pp": round(worst, 4),
            "pass": bool(worst <= CALIB_GATE_PP)}


def share_responsiveness(te: pd.DataFrame, p: np.ndarray, driver: np.ndarray,
                         n_q: int = 5) -> dict:
    """Quintile responsiveness on the candidate's own as-of rate, under the
    Decision-8 gate. The slope half is what stops a FLAT arm passing: an arm can
    be monotone in every step and still sit at the league mean, which violates
    the standing matchup-specific rule."""
    obs, pred, dr = _alt_long(te, p, driver)
    out = PM.quintile_responsiveness(dr, obs, np.column_stack([1 - pred, pred]), 1,
                                     n_q=n_q)
    return decision8_verdict(out)


def top_k_accuracy(te: pd.DataFrame, p: np.ndarray, k: int) -> float:
    y = te["y"].to_numpy()
    order = np.argsort(-p, axis=1, kind="stable")
    return float((order[:, :k] == y[:, None]).any(axis=1).mean())


def score_choice_arm(te: pd.DataFrame, p: np.ndarray, driver: np.ndarray,
                     target: str) -> dict:
    k_alt = N_ALT_OF[target]
    y = te["y"].to_numpy()
    onehot = np.eye(k_alt)[y]
    cal = share_calibration(te, p, driver)
    dcal = driver_calibration(te, p, driver)
    resp = share_responsiveness(te, p, driver)
    return {
        "n": int(len(te)),
        "log_loss": round(PM.log_loss(y, p), 6),
        "brier": round(float(((p - onehot) ** 2).sum(axis=1).mean()), 6),
        "top1": round(top_k_accuracy(te, p, 1), 6),
        "top3": round(top_k_accuracy(te, p, min(3, k_alt)), 6),
        "calibration": cal,
        "driver_calibration": dcal,
        "calib_pass": bool(cal["pass"] and dcal["pass"]),
        "calib_worst_gap_pp": max(cal["max_abs_gap_pp"], dcal["max_abs_gap_pp"]),
        "responsiveness": {"player_asof_rate": resp},
        "resp_pass": resp["pass"],
        "resp_steps": int(resp["pred_monotone_steps"]),
        "resp_slope_ratio": resp["slope_ratio"],
        "resp_exempt": resp["small_span_exempt"],
    }


#: The two team drivers each binary is gated on. Both are reported; a driver
#: whose realised quintile span is under 2 pp is exempt per Decision 8, and the
#: verdict is the AND over the non-exempt ones.
BINARY_DRIVERS: dict[str, tuple[str, ...]] = {
    b: ("off_rate_c", "def_rate_c") for b in BINARY_TARGETS}


def score_binary_arm(te: pd.DataFrame, p: np.ndarray, drivers: dict[str, np.ndarray],
                     binary: str) -> dict:
    y = te["b"].to_numpy()
    P = np.column_stack([1.0 - p, p])
    cal = PM.decile_calibration(y, P, ("no", "yes"))
    ok, worst, who = PM.calibration_verdict(cal, gate_pp=CALIB_GATE_PP)
    resp, passes = {}, []
    for name in BINARY_DRIVERS[binary]:
        r = decision8_verdict(PM.quintile_responsiveness(drivers[name], y, P, 1))
        resp[name] = r
        if not r["small_span_exempt"]:
            passes.append(bool(r["pass"]))
    return {
        "n": int(len(te)),
        "base_rate_pct": round(float(y.mean() * 100), 4),
        "log_loss": round(PM.log_loss(y, P), 6),
        "brier": round(float(((p - y) ** 2).mean()), 6),
        "calibration": cal,
        "calib_pass": bool(ok),
        "calib_worst_gap_pp": worst,
        "calib_worst_class": who,
        "responsiveness": resp,
        "resp_pass": bool(all(passes)) if passes else True,
        "resp_all_exempt": not passes,
        "resp_steps": min(int(r["pred_monotone_steps"]) for r in resp.values()),
        "resp_slope_ratio": {k: v["slope_ratio"] for k, v in resp.items()},
    }


def bootstrap_se_choice(te: pd.DataFrame, p: np.ndarray, n_rep: int = 200,
                        seed: int = 12345) -> float:
    """Game-level block bootstrap SE of the K-way log loss -- the noise floor for
    the non-tree arms (`prob_metrics.block_bootstrap_se`). Events inside one game
    are not independent, so the resampling unit is the GAME."""
    return PM.block_bootstrap_se(te["game_id"].to_numpy(), te["y"].to_numpy(), p,
                                 n_rep=n_rep, seed=seed)


def bootstrap_se_binary(te: pd.DataFrame, p: np.ndarray, n_rep: int = 200,
                        seed: int = 12345) -> float:
    return PM.block_bootstrap_se(te["game_id"].to_numpy(), te["b"].to_numpy(),
                                 np.column_stack([1.0 - p, p]), n_rep=n_rep, seed=seed)


# ===========================================================================
# 9. The game-level checks (usage's trio, generalised over K)
# ===========================================================================
#: `usage.alloc_cells` and its three helpers are fixed at five alternatives and
#: at the `alt_*` column names and the `usage_alloc` RNG family. They are ported
#: here over K and over `cand_*`; the definitions are IDENTICAL, and
#: `tests/test_attribution.py::test_d_game_level_port_matches_usage` pins this
#: port to `usage.game_level_check` on a five-alternative frame so that a number
#: quoted next to a usage number is the same statistic, not a lookalike.
def alloc_cells(d: pd.DataFrame, k_alt: int) -> U.AllocCells:
    """The (team-game, player) cell index the game-level checks accumulate into.

    Every player who was a CANDIDATE in at least one of a team-game's events gets
    a cell, so a player the arm never credits contributes an explicit ZERO to his
    own count SD instead of dropping out of it -- which is the difference between
    measuring "too narrow" and measuring nothing."""
    gid = d["game_id"].to_numpy(dtype="int64")
    tid = d["cand_team_id"].to_numpy(dtype="int64")
    tg, _ = pd.factorize(pd.MultiIndex.from_arrays([gid, tid]), sort=False)
    alts = d[[f"cand_{k}" for k in range(1, k_alt + 1)]].to_numpy(dtype="int64")
    cell, uniq = pd.factorize(pd.MultiIndex.from_arrays(
        [np.repeat(tg, k_alt), alts.reshape(-1)]), sort=False)
    cell_tg = np.asarray(uniq.get_level_values(0), dtype="int64")
    cell_pl_id = np.asarray(uniq.get_level_values(1), dtype="int64")
    pcode, puniq = pd.factorize(cell_pl_id, sort=False)
    n_players = int(pcode.max()) + 1 if len(pcode) else 0
    return U.AllocCells(cell_of=cell.reshape(-1, k_alt), cell_tg=cell_tg,
                        cell_player=pcode, cell_player_id=cell_pl_id,
                        n_tg=int(tg.max()) + 1 if len(tg) else 0,
                        n_players=n_players,
                        player_games=np.bincount(pcode, minlength=n_players),
                        player_id=np.asarray(puniq, dtype="int64"))


#: Room for a team-game's events inside one game's key space, as in `usage`.
EVENT_KEY_STRIDE = 4096
#: One RNG family per credit (module docstring, THE SAMPLER).
FAMILY_OF: dict[str, str] = {
    "REB_off": "attr_rebound", "REB_def": "attr_rebound",
    "assist": "attr_assist", "steal": "attr_steal", "block": "attr_block",
    "assisted": "attr_assist", "stolen": "attr_steal", "blocked": "attr_block",
}


def event_stream_keys(d: pd.DataFrame, seed: int, family: str) -> np.ndarray:
    """One counter-based key per EVENT, off the `(seed, game_id, family)` stream
    with the event's ordinal within its game folded into the game id.

    Keying on the game alone would hand every event of a team-game the SAME
    uniform and collapse the whole game onto one player (measured in the usage
    bake-off: 3.42 players with a three-point attempt per team-game against a
    real 6.69). Folding the ordinal in keeps the per-game independence the RNG
    rule exists to give while making a game's own draws independent of each
    other."""
    gid = d["game_id"].to_numpy(dtype="int64")
    ordinal = d.groupby("game_id", sort=False).cumcount().to_numpy()
    if ordinal.max(initial=0) >= EVENT_KEY_STRIDE:
        raise ValueError("more events in one game than EVENT_KEY_STRIDE allows")
    return RNG.stream_keys(seed, (gid.astype("uint64") * np.uint64(EVENT_KEY_STRIDE)
                                  + ordinal.astype("uint64")), family)


def _cell_counts(cells: U.AllocCells, pick: np.ndarray) -> np.ndarray:
    n = len(pick)
    return np.bincount(cells.cell_of[np.arange(n), pick],
                       minlength=len(cells.cell_tg)).astype("float64")


def _tg_stats(cells: U.AllocCells, cnt: np.ndarray) -> tuple[float, float, float]:
    """(players with >= 1 credit per team-game, top-1 share, top-3 share)."""
    order = np.lexsort((-cnt, cells.cell_tg))
    tg_sorted = cells.cell_tg[order]
    starts = np.flatnonzero(np.concatenate([[True], tg_sorted[1:] != tg_sorted[:-1]]))
    pos = np.arange(len(order)) - np.repeat(starts, np.diff(np.append(starts, len(order))))
    c_sorted = cnt[order]
    tot = np.maximum(np.bincount(tg_sorted, weights=c_sorted, minlength=cells.n_tg), 1.0)
    gt0 = np.bincount(cells.cell_tg, weights=(cnt > 0).astype("float64"),
                      minlength=cells.n_tg)
    top1 = np.bincount(tg_sorted, weights=np.where(pos == 0, c_sorted, 0.0),
                       minlength=cells.n_tg) / tot
    top3 = np.bincount(tg_sorted, weights=np.where(pos < 3, c_sorted, 0.0),
                       minlength=cells.n_tg) / tot
    live = np.bincount(cells.cell_tg, minlength=cells.n_tg) > 0
    return float(gt0[live].mean()), float(top1[live].mean()), float(top3[live].mean())


def _player_sd(cells: U.AllocCells, cnt: np.ndarray, min_games: int) -> np.ndarray:
    """Per player, the SD across his team-games of his per-game credited count --
    CFB's "too narrow" statistic, on the same definition `usage` used."""
    s1 = np.bincount(cells.cell_player, weights=cnt, minlength=cells.n_players)
    s2 = np.bincount(cells.cell_player, weights=cnt ** 2, minlength=cells.n_players)
    ng = cells.player_games.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 / ng - (s1 / ng) ** 2) * np.where(ng > 1, ng / (ng - 1), np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))
    sd[cells.player_games < min_games] = np.nan
    return sd


def _pick(p: np.ndarray, u: np.ndarray, k_alt: int) -> np.ndarray:
    cum = np.cumsum(p, axis=1)
    cum = cum / cum[:, [-1]]
    return np.clip((u[:, None] > cum).sum(axis=1), 0, k_alt - 1)


def choice_game_level_check(d: pd.DataFrame, p: np.ndarray, target: str,
                            n_draw: int = 40, seed: int = 0,
                            min_games: int = 8) -> dict:
    """The "too narrow / too short" pair plus the top-1/top-3 share check.

    Re-attributes the ACTUAL event sequence with the ACTUAL fives `n_draw` times,
    which is the pre-registration's isolation clause: how many events a team-game
    has, and who was on the floor at each of them, come from reality, so what is
    measured is attribution alone -- not rotation, not the rebound model, not the
    possession-outcome model."""
    k_alt = N_ALT_OF[target]
    cells = alloc_cells(d, k_alt)
    keys = event_stream_keys(d, seed, FAMILY_OF[target])
    act = _cell_counts(cells, d["y"].to_numpy())
    gt0_a, t1_a, t3_a = _tg_stats(cells, act)
    sd_a = _player_sd(cells, act, min_games)

    sd_acc = np.zeros(cells.n_players)
    sd_n = np.zeros(cells.n_players)
    gt0_s, t1_s, t3_s = [], [], []
    for r in range(n_draw):
        cnt = _cell_counts(cells, _pick(p, RNG.uniforms(keys, r), k_alt))
        g0, a1, a3 = _tg_stats(cells, cnt)
        gt0_s.append(g0)
        t1_s.append(a1)
        t3_s.append(a3)
        sd = _player_sd(cells, cnt, min_games)
        ok = np.isfinite(sd)
        sd_acc[ok] += sd[ok]
        sd_n[ok] += 1

    with np.errstate(invalid="ignore"):
        sd_sim = np.where(sd_n > 0, sd_acc / np.maximum(sd_n, 1), np.nan)
    both = np.isfinite(sd_sim) & np.isfinite(sd_a)
    sd_ratio = (float(sd_sim[both].mean() / max(sd_a[both].mean(), EPS))
                if both.any() else np.nan)
    gt0_sim = float(np.mean(gt0_s))
    t1_sim, t3_sim = float(np.mean(t1_s)), float(np.mean(t3_s))
    return {
        "n_team_games": cells.n_tg, "n_draw": n_draw,
        "n_players_graded": int(both.sum()),
        "sd_sim": round(float(sd_sim[both].mean()), 5) if both.any() else None,
        "sd_actual": round(float(sd_a[both].mean()), 5) if both.any() else None,
        "sd_ratio": round(sd_ratio, 5),
        "sd_pass": bool(SD_RATIO_BAND[0] <= sd_ratio <= SD_RATIO_BAND[1]),
        "players_gt0_sim": round(gt0_sim, 5),
        "players_gt0_actual": round(gt0_a, 5),
        "players_gt0_delta": round(gt0_sim - gt0_a, 5),
        "gt0_pass": bool(abs(gt0_sim - gt0_a) <= GT0_TOL),
        "top1_sim_pct": round(t1_sim * 100, 4), "top1_actual_pct": round(t1_a * 100, 4),
        "top1_gap_pp": round((t1_sim - t1_a) * 100, 4),
        "top3_sim_pct": round(t3_sim * 100, 4), "top3_actual_pct": round(t3_a * 100, 4),
        "top3_gap_pp": round((t3_sim - t3_a) * 100, 4),
        "top_pass": bool(abs(t1_sim - t1_a) * 100 <= TOPSHARE_TOL_PP
                         and abs(t3_sim - t3_a) * 100 <= TOPSHARE_TOL_PP),
    }


def binary_game_level_check(d: pd.DataFrame, p: np.ndarray, binary: str,
                            n_draw: int = 40, seed: int = 0) -> dict:
    """The game-level check a BINARY admits.

    A binary credits no player, so usage's per-player trio (count SD, players
    with >= 1, top-k share) has no player axis to live on. The statistic of the
    same family that DOES exist is the per-TEAM-GAME COUNT of the credit: draw
    the Bernoulli over the actual event sequence and compare the mean and the
    dispersion of "assists / steals / blocks this team-game" against reality.
    That is the "too narrow" question asked one level up, and it is the level the
    binary actually controls. The three per-player checks are reported as `null`
    with this reason rather than silently omitted; the COMPOSED per-player check
    (`composed_check`) is where the player axis comes back."""
    gid = d["game_id"].to_numpy(dtype="int64")
    tid = d["cand_team_id"].to_numpy(dtype="int64")
    tg, _ = pd.factorize(pd.MultiIndex.from_arrays([gid, tid]), sort=False)
    n_tg = int(tg.max()) + 1 if len(tg) else 0
    keys = event_stream_keys(d, seed, FAMILY_OF[binary])
    y = d["b"].to_numpy(dtype="float64")
    act = np.bincount(tg, weights=y, minlength=n_tg)
    means, sds = [], []
    for r in range(n_draw):
        hit = (RNG.uniforms(keys, r) < p).astype("float64")
        c = np.bincount(tg, weights=hit, minlength=n_tg)
        means.append(float(c.mean()))
        sds.append(float(c.std(ddof=1)))
    sd_sim = float(np.mean(sds))
    sd_act = float(act.std(ddof=1))
    ratio = sd_sim / max(sd_act, EPS)
    return {
        "n_team_games": n_tg, "n_draw": n_draw,
        "count_sim": round(float(np.mean(means)), 5),
        "count_actual": round(float(act.mean()), 5),
        "count_gap": round(float(np.mean(means) - act.mean()), 5),
        "sd_sim": round(sd_sim, 5), "sd_actual": round(sd_act, 5),
        "sd_ratio": round(ratio, 5),
        "sd_pass": bool(SD_RATIO_BAND[0] <= ratio <= SD_RATIO_BAND[1]),
        "count_pass": bool(abs(float(np.mean(means)) - act.mean()) <= GT0_TOL),
        "players_gt0_sim": None, "players_gt0_actual": None,
        "gt0_pass": None, "top_pass": None,
        "per_player_checks": "not applicable -- a binary credits no player; see "
                             "composed_check for the player axis",
    }


def composed_check(pop_full: pd.DataFrame, p_binary: np.ndarray,
                   p_choice: np.ndarray, target: str, n_draw: int = 20,
                   seed: int = 0, min_games: int = 8) -> dict:
    """REPORTED DIAGNOSTIC, never a gate: the binary and the choice composed.

    Draws the binary, then (when it fires) the candidate, over the actual event
    sequence, and measures the per-player per-team-game count trio on the result.
    This is the quantity a player prop actually needs -- "assists by player X in
    this game" -- and it is the only place in this bake-off where the two halves
    of a target are graded together. It is NOT in the pre-registration's decision
    rule and does not enter `decide`."""
    k_alt = N_ALT_OF[target]
    cells = alloc_cells(pop_full, k_alt)
    bkeys = event_stream_keys(pop_full, seed, FAMILY_OF[target] + "_binary")
    ckeys = event_stream_keys(pop_full, seed, FAMILY_OF[target])
    fired = pop_full["b"].to_numpy() == 1
    credited = pop_full["in_five"].to_numpy() & fired
    act = _cell_counts(cells, np.where(credited, pop_full["y"].to_numpy(), 0))
    act = act - np.bincount(
        cells.cell_of[np.arange(len(pop_full)), 0],
        weights=(~credited).astype("float64"), minlength=len(cells.cell_tg))
    act = np.maximum(act, 0.0)
    gt0_a, t1_a, t3_a = _tg_stats(cells, act)
    sd_a = _player_sd(cells, act, min_games)

    sd_acc = np.zeros(cells.n_players)
    sd_n = np.zeros(cells.n_players)
    gt0_s, t1_s, t3_s, cnt_s = [], [], [], []
    n = len(pop_full)
    for r in range(n_draw):
        hit = RNG.uniforms(bkeys, r) < p_binary
        pick = _pick(p_choice, RNG.uniforms(ckeys, r), k_alt)
        cnt = np.bincount(cells.cell_of[np.arange(n), pick],
                          weights=hit.astype("float64"),
                          minlength=len(cells.cell_tg)).astype("float64")
        g0, a1, a3 = _tg_stats(cells, cnt)
        gt0_s.append(g0)
        t1_s.append(a1)
        t3_s.append(a3)
        cnt_s.append(float(cnt.sum() / max(cells.n_tg, 1)))
        sd = _player_sd(cells, cnt, min_games)
        ok = np.isfinite(sd)
        sd_acc[ok] += sd[ok]
        sd_n[ok] += 1
    with np.errstate(invalid="ignore"):
        sd_sim = np.where(sd_n > 0, sd_acc / np.maximum(sd_n, 1), np.nan)
    both = np.isfinite(sd_sim) & np.isfinite(sd_a)
    ratio = (float(sd_sim[both].mean() / max(sd_a[both].mean(), EPS))
             if both.any() else np.nan)
    return {
        "n_team_games": cells.n_tg, "n_draw": n_draw,
        "n_players_graded": int(both.sum()),
        "credits_per_team_game_sim": round(float(np.mean(cnt_s)), 5),
        "credits_per_team_game_actual": round(float(act.sum() / max(cells.n_tg, 1)), 5),
        "sd_ratio": round(ratio, 5),
        "players_gt0_sim": round(float(np.mean(gt0_s)), 5),
        "players_gt0_actual": round(gt0_a, 5),
        "top1_sim_pct": round(float(np.mean(t1_s)) * 100, 4),
        "top1_actual_pct": round(t1_a * 100, 4),
        "top3_sim_pct": round(float(np.mean(t3_s)) * 100, 4),
        "top3_actual_pct": round(t3_a * 100, 4),
    }


# ===========================================================================
# 10. The samplers
# ===========================================================================
@dataclass
class AttributionState:
    """One game's attribution rates plus its position in each credit's RNG
    stream.

    `rates[credit]` maps a CBBD player id to his relative rate for that credit in
    THIS game, where `credit` is one of `REB_off`, `REB_def`, `assist`, `steal`,
    `block`. Each credit has its OWN counter and its own stream key, so adding or
    removing one credit from the engine does not move any other credit's draws --
    the property the `(seed, game_id, family)` rule exists to give."""
    game_id: int
    seed: int
    rates: dict[str, dict[int, float]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    keys: dict[str, np.ndarray] = field(default_factory=dict)

    def next_uniform(self, family: str) -> float:
        u = float(RNG.uniforms(self.keys[family], self.counters[family])[0])
        self.counters[family] += 1
        return u


def new_game_state(game_id: int, seed: int,
                   rates: dict[str, dict[int, float]] | None = None
                   ) -> AttributionState:
    """Open one game's attribution streams, one per credit family."""
    fams = sorted(set(FAMILY_OF[t] for t in CHOICE_TARGETS))
    return AttributionState(
        game_id=int(game_id), seed=int(seed),
        rates={k: dict(v) for k, v in (rates or {}).items()},
        counters={f: 0 for f in fams},
        keys={f: RNG.stream_keys(seed, np.array([int(game_id)], dtype="uint64"), f)
              for f in fams})


def _choose(ids: list[int], credit: str, state: AttributionState) -> int:
    """Credit one event to one of `ids`, proportional to their rates for
    `credit`.

    A candidate set in which nobody carries any rate falls back to the uniform --
    the only honest answer when the model has no information -- never a silent
    pick of the first id. One uniform of the credit's own stream is consumed per
    call."""
    table = state.rates.get(credit, {})
    w = np.array([max(float(table.get(int(i), 0.0)), 0.0) for i in ids])
    tot = w.sum()
    w = np.full(len(ids), 1.0 / len(ids)) if (tot <= 0 or not np.isfinite(tot)) else w / tot
    u = state.next_uniform(FAMILY_OF[credit])
    return int(ids[int(np.clip((u > np.cumsum(w)).sum(), 0, len(ids) - 1))])


def _binary_then_choice(ids: list[int], credit: str, state: AttributionState,
                        p: float | None) -> int | None:
    """The binary (when `p` is given) and then the choice, off ONE stream.

    The binary uniform is drawn first and the choice uniform only when the binary
    fires, so a game in which no steal happens consumes exactly as many uniforms
    as there were turnovers and the stream stays a function of the event sequence
    alone."""
    if p is not None:
        if state.next_uniform(FAMILY_OF[credit]) >= float(p):
            return None
    return _choose(ids, credit, state)


def draw_rebounder(five, state: AttributionState, offensive: bool) -> int:
    """Credit a live rebound to one of the rebounding team's `five`.

    `offensive` selects the target: True for a rebound won by the team that shot
    (the `REB_off` rates), False for the defence (`REB_def`). Both run on the
    `attr_rebound` family, because they are the same credit asked on two sides of
    the ball and an engine that draws one never draws the other for the same
    miss."""
    ids = [int(x) for x in five]
    if len(ids) != 5:
        raise ValueError(f"expected 5 players on the floor, got {len(ids)}")
    return _choose(ids, "REB_off" if offensive else "REB_def", state)


def draw_assist(five, shooter_id: int, state: AttributionState,
                p_assisted: float | None = None) -> int | None:
    """Credit the assist on a made field goal, or `None` when it is unassisted.

    `five` is the OFFENSIVE five including the shooter; the shooter is removed
    here, which is the eligible-set rule ("a shooter cannot assist himself")
    enforced at sampling time as well as at training time. Passing
    `p_assisted` draws the binary first; omitting it asserts the caller already
    knows the field goal was assisted."""
    ids = [int(x) for x in five]
    if len(ids) != 5:
        raise ValueError(f"expected 5 players on the floor, got {len(ids)}")
    others = [i for i in ids if i != int(shooter_id)]
    if len(others) != 4:
        raise ValueError("the shooter must be exactly one of the five on the floor")
    return _binary_then_choice(others, "assist", state, p_assisted)


def draw_steal(five, state: AttributionState,
               p_steal: float | None = None) -> int | None:
    """Credit the steal on a turnover to one of the DEFENSIVE `five`, or `None`
    when the turnover was not a steal."""
    ids = [int(x) for x in five]
    if len(ids) != 5:
        raise ValueError(f"expected 5 players on the floor, got {len(ids)}")
    return _binary_then_choice(ids, "steal", state, p_steal)


def draw_block(five, state: AttributionState,
               p_block: float | None = None) -> int | None:
    """Credit the block on a missed field goal to one of the DEFENSIVE `five`, or
    `None` when the miss was not blocked."""
    ids = [int(x) for x in five]
    if len(ids) != 5:
        raise ValueError(f"expected 5 players on the floor, got {len(ids)}")
    return _binary_then_choice(ids, "block", state, p_block)


__all__ = [
    "ALL_CLASSES", "AWARE_FEATURES", "BINARY_ARMS", "BINARY_ARM_ORDER",
    "BINARY_DRIVERS", "BINARY_FEATURES", "BINARY_SHRINK_GRID", "BINARY_TARGETS",
    "BASE_COLUMNS", "CALIB_GATE_PP", "CHOICE_ARMS", "CHOICE_ARM_ORDER",
    "CHOICE_TARGETS", "CL_FEATURES", "EVENT_KEY_STRIDE", "FAMILY_OF", "FOLDS",
    "GT0_TOL", "LGBM_FEATURES", "LGBM_PARAM_GRID", "N_ALT_OF", "OWN_SHARE_OF",
    "POPULATIONS", "POP_OF", "PRIOR_KINDS", "PRIOR_SEASON_FEATURES",
    "RATE_CLASSES", "RATE_OF", "RESP_MIN_STEPS", "SD_RATIO_BAND",
    "SELECTION_FOLD", "SHOT_CLASSES", "SHOT_CLASS_CODE", "SHRINK_GRID",
    "SIDE_OF", "SLOPE_BAND", "SMALL_SPAN_PP", "TARGETS", "TEAM_FEATURES",
    "TEAM_RATES", "TOPSHARE_TOL_PP", "TREE_ARM", "UNIFORM_LL", "WF_SEASON",
    "WF_SPLIT_DATE", "AttributionState", "BinaryLgbmArm", "BinaryRidgeArm",
    "CondLogitArm", "LgbmChoiceArm", "alloc_cells", "arm_order", "arms_for",
    "binary_game_level_check", "binary_matrix", "bootstrap_se_binary",
    "bootstrap_se_choice", "build_attr_events", "build_attr_stream",
    "build_binary_design", "build_choice_design", "build_player_asof",
    "build_team_asof", "choice_game_level_check", "cl_design", "composed_check",
    "coverage_report", "decision8_verdict", "draw_assist", "draw_block",
    "draw_rebounder", "draw_steal", "driver_calibration",
    "drop_constant_features", "event_stream_keys", "fit_binary_shrinkage",
    "fit_shrinkage", "fold_slices", "group_softmax_objective", "lgbm_init_score",
    "long_frame", "new_game_state", "normalise", "own_share_shrunk", "p1_probs",
    "prior_season_available", "score_binary_arm", "score_choice_arm",
    "share_calibration", "share_responsiveness", "shrunk_rate", "team_shrunk",
    "top_k_accuracy", "unidentified_features", "usable", "walkforward_slices",
]
