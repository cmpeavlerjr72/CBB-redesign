"""
possessions.py -- CBBD play-by-play -> one row per possession (and one row per
chance), the event layer every L3 sub-model trains on.

Entry point: `scripts/build_possessions.py`. Output:
`data/processed/possessions{_v2}/possessions_{season}.parquet` (one row per
possession) and `.../chances_{season}.parquet` (one row per chance; a
possession with an offensive rebound has more than one).

VERSIONS. `v1` (`data/processed/possessions`) is the original build. `v2`
(`data/processed/possessions_v2`) is the same segmentation rule over a
corrected event layer: the rim-location override in `cbb_sim.pbp.events`
repairs ESPN's 2025 putback mistag, and the chance rows carry their own
`fga_rim`/`fga_jump2`/`fga_3` counts so a style rate can be built from FIRST
chances alone. The segmentation state machine itself is unchanged, and by
construction has to be: the override only ever turns an `FGA_jump2` into an
`FGA_rim`, and `_handle_fga` treats those two classes identically (same
points, same rebound handling), so v2's possession and chance BOUNDARIES are
bit-identical to v1's and only labels move. `possessions_dir()` resolves the
version; the default stays `v1` until it is switched in one place.

===========================================================================
THE SEGMENTATION RULE
===========================================================================

DEFINITION. A **possession** is the interval during which one team has the
ball, from the moment it gains it to the moment it gives it up. It ends on a
made field goal, a made last free throw of a trip, a defensive rebound, a
turnover, or the end of a period -- never on an offensive rebound. A
**chance** is one shot-clock attempt inside a possession: the first chance
begins with the possession, and each offensive rebound starts another. This
is the Kubatko/Oliver convention, and it is the one the L2 pace target and
the `FGA - OREB + TOV + 0.44*FTA` box formula both approximate; PART A of
`docs/tests/possessions_build_2026-09-10.md` measures the gap between them.

INPUT ORDER. `cbb_sim.pbp.events.load_plays` de-duplicates on `(gameId, id)`
(the raw files carry 18-19% duplicated rows from the day-by-day pull) and
sorts by `id` within `gameId`. `id` order is clock-consistent on 100.000% of
2022-2024 rows and 99.99% of 2025-2026 rows; `sourceId` is 22x worse.

EVENT VOCABULARY. `cbb_sim.pbp.events.PLAY_TYPE_TO_EVENT` maps all 24 CBBD
`playType` values explicitly and raises on an unknown one
(`docs/SIM_GUARDRAILS.md` section 4). Classes in `INERT_CLASSES` -- timeout,
sub, block, jumpball, challenge -- carry no possession information and are
dropped before the machine runs, so "the next event" always means the next
FGA / FT / rebound / turnover / foul / period boundary.

--------------------------------------------------------------------------
STATE MACHINE (one forward pass per game)
--------------------------------------------------------------------------
State: `offense` (team with the ball), `awaiting_reb` (a shot or last free
throw missed and the rebound has not been seen yet), `pending_terminal` (the
event class that will be charged if the possession ends now), the running
per-period team-foul counts, and the open possession/chance accumulators.

  FGA (FGA_rim | FGA_jump2 | FGA_3)
      The shooting team IS the offense. If a possession is open for the other
      team, close it first (an implicit, unlogged change of possession -- see
      MISMATCH GUARD). Record the attempt on the current chance.
      * made, and the next non-inert event is a `foul` followed by exactly one
        free throw by the same team at the same clock -> AND-ONE: the free
        throw's points/attempts join THIS possession, `and_one = True`, and
        the possession's terminal event stays the FGA class. (If that bonus
        free throw is missed and offensively rebounded, a NEW possession
        opens: the scoring possession already ended on the made basket.)
      * made otherwise -> possession ends now, terminal = the FGA class.
      * missed -> `awaiting_reb`, `pending_terminal` = the FGA class. The
        possession stays open until a rebound, turnover or period end.

  OREB
      Not terminal. `n_chances += 1`, a new chance opens, `awaiting_reb`
      clears. EXCEPTION -- the administrative rebound: ESPN logs an
      "Offensive Rebound" for the shooting team *between* two free throws of
      the same trip (a dead-ball reset, not a live rebound). An OREB is
      ignored when the immediately following non-inert event is another free
      throw by the same team at the same `secondsRemaining`.

  DREB
      Terminal. The possession is charged to `offense` (the team that just
      missed), NOT to the rebounder's own team, and the terminal event is
      `pending_terminal` -- the missed FGA class or FT trip class that
      created the rebound. Falls back to "the team that is not the
      rebounder's" if `offense` was never set (a feed gap).

  DeadBallReb
      No-op. Both of its meanings are already handled: the common case (ball
      out of bounds off the defence, offence retains) changes nothing, and
      the rarer case (awarded to the defence) is caught by the MISMATCH GUARD
      when the next real event belongs to the other team.

  TOV (Lost Ball Turnover; a `steal` with no adjacent turnover is promoted to
      one charged to the other team)
      Terminal, charged to the team on the event, terminal = `TOV`.

  foul (PersonalFoul)
      Always increments that team's period foul count -- including offensive
      fouls, which NCAA counts as team fouls and which the empirical bonus
      check below confirms must be counted. If free throws follow, the whole
      trip is consumed here; see FREE-THROW TRIPS.

  technical (Technical Foul)
      Never changes possession. The free throws that follow it are shot by
      the other team during a dead ball and belong to no possession; their
      points are recorded on the open possession as `tech_points_off` /
      `tech_points_def` (by which side shot them) so that the score
      reconciliation still balances, and they are excluded from `fta`,
      `ftm`, `points` and from every terminal-event class.

  end_period / end_game
      Any open possession closes with terminal = `end_period`. Period foul
      counts reset. (Two halves and up to four overtimes: `secondsRemaining`
      is per-period, 1200 in periods 1-2 and 300 in 3+.)

  MISMATCH GUARD. Before processing any FGA / real FT trip / TOV whose team
  differs from the open possession's `offense`, the open possession is closed
  first with its `pending_terminal` (or `unknown` if nothing was pending).
  Without this, a dropped or mislabelled rebound would delete one team's
  possession instead of merely omitting an administrative row.

--------------------------------------------------------------------------
FREE-THROW TRIPS: shooting vs bonus, and how the bonus state is derived
--------------------------------------------------------------------------
CBBD does not say whether a personal foul was a shooting foul, and the free
throw rows carry no "1 of 2" text (checked: the only `playText` suffix on all
281,470 free throws in 2025 is "Free Throw."). Both facts have to be
recovered from the sequence.

A **trip** is the run of consecutive free throws by one team, allowing
interleaved inert rows and administrative rebounds at the same
`secondsRemaining`.

BONUS STATE, DERIVED BY COUNTING. NCAA men's rules: the 7th team foul of a
half puts the other team in the one-and-one, the 10th puts it in the double
bonus. Counting `PersonalFoul` rows per (period, team) reproduces both
thresholds exactly in the data -- in a 600-game 2025 slice the number of
one-free-throw trips per foul index jumps from ~230 to 479 at the 7th foul
(prior count 6) and the share of those single free throws that were MISSED
jumps from ~45% to 97% (a missed one-and-one front end has no second shot),
then the single-free-throw trips collapse from 180 to 12 at the 10th foul
(prior count 9) when the double bonus removes the front end. The table thus
validates the counting rule rather than assuming it.

So, writing `pf` for the fouling team's prior foul count in this period:

    in_bonus         pf >= 6   (this foul is the 7th or later)
    in_double_bonus  pf >= 9   (this foul is the 10th or later)

and the possession row carries `off_in_bonus` / `off_in_double_bonus`
computed the same way from the DEFENCE's foul count at possession start.

TRIP CLASSIFICATION (deterministic, applied in this order):

  1. The trip follows a `technical` rather than a `foul`  -> technical trip,
     no terminal event, points to `tech_points_*`.
  2. exactly 1 free throw, and the last real event before the foul is a MADE
     field goal by the same team at the same clock -> AND-ONE, folded into
     that FGA's possession (verified: of 1,982 fouls that are both preceded
     by a shot at the same clock and followed by free throws, 1,942 follow a
     MADE shot -- ESPN does not log the field-goal attempt when a shooting
     foul occurs on a MISS, which is exactly why a shooting foul on a miss
     must be its own terminal class).
  3. 3 free throws -> `FT_trip_shooting` (fouled on a three-point attempt).
  4. `pf <= 5` (no bonus) -> `FT_trip_shooting`: with no bonus in force, a
     non-shooting foul produces no free throws at all.
  5. `pf` in 6..8 (one-and-one):
       - 1 free throw, missed  -> `FT_trip_bonus` (missed front end).
       - 2 free throws, first missed -> `FT_trip_shooting` (a one-and-one
         cannot produce a second shot after a missed first).
       - 2 free throws, first made -> `FT_trip_bonus`.
       - anything else -> `FT_trip_shooting`.
  6. `pf >= 9` (double bonus): 2 free throws -> `FT_trip_bonus`; otherwise
     `FT_trip_shooting`.

--------------------------------------------------------------------------
TECHNICAL FREE THROWS: the same-clock lookahead (added 2026-09-18, OFF by
default)
--------------------------------------------------------------------------
THE DEFECT. `_handle_technical` originally looked at exactly ONE row ahead for
the free throw a technical produces. CBBD's feed routinely inserts an
administrative row first, at the IDENTICAL frozen clock, credited to the
OFFENDING team:

  Technical Foul on Nick Pringle.      team 333, p2, 563s
  Nick Pringle Turnover.               team 333, p2, 563s   <-- inserted row
  Riley Minix made Free Throw.         team 2413, p2, 563s
  Riley Minix made Free Throw.         team 2413, p2, 563s

Census of the row immediately following every single-team technical moment,
2022-2025: 83-91% `Lost Ball Turnover`, 2-13% `PersonalFoul`, the rest a
handful of rebounds and shots. The gap is exactly one row on 99.0% of affected
moments and never more than six. The inserted row names the SAME player as the
technical on 99.5-100.0% of the rows where both carry a participant id, and
belongs to the offending team on 98.9-99.7%.

CONSEQUENCE OF THE DEFECT. The one-row lookahead misses the trip, so the free
throws fall through to the GENERIC free-throw handler: they are charged to a
possession as ordinary `FT_trip_shooting` / `FT_trip_bonus` attempts, with
`fta`/`ftm`/`points` on a real possession instead of `tech_points_*`, and they
close that possession with a free-throw terminal event. ~1,200-1,900 technical
attempts a season are mis-tagged this way.

THE INSERTED TURNOVER IS REAL, NOT A PHANTOM -- so the fix does NOT delete it.
Measured against hoopR's team box (`turnovers`), which disagrees with the event
stream on 2.69% of team-games and is therefore not a re-add of these same rows:
team-games carrying one inserted technical turnover run only **+0.007** event
turnovers above box turnovers relative to team-games carrying none (SE 0.005;
+1.00 is what a pure feed artefact would give). The box counts them. Deleting
them would create a fresh -1 box disagreement per technical, so the fix is a
LOOKAHEAD ONLY: every intervening row is still processed exactly as before,
and only the free throws' CLASS changes.

SEPARATELY MEASURED, NOT PATCHED: on the next live-ball event, the ball
belongs to the technical'd team on 65.8% of the moments that carry the inserted
turnover, against 22.4% of the technical moments that do not -- i.e. the
turnover row usually does NOT coincide with the ball changing hands, even
though the box counts it. That is a separate possible defect in the possession
segmentation, it is reported rather than patched, and it is unchanged by this
version.

THE FIX (`tech_lookahead=True`, version `v3` only). `_handle_technical` runs a
BOUNDED SAME-CLOCK scan (`technical_ft_index`) for the first free throw by the
beneficiary while `(period, secondsRemaining)` are unchanged, stepping over at
most `TECH_LOOKAHEAD_MAX_ROWS` non-free-throw rows and stopping at any field
goal attempt or period boundary. If that free throw is the very next row the
behaviour is the pre-fix behaviour. If it is further ahead, the machine ARMS a
pending marker holding that exact row index and returns, so the intervening
rows (the turnover, the duplicate personal foul) are processed normally by the
main loop; the free-throw branch then consumes that one row index as a
TECHNICAL trip. `_handle_foul` checks the same marker so that the duplicate
`PersonalFoul` row logged for a technical still increments the team foul count
but no longer claims the technical's free throws as an ordinary trip.

The marker is an EXACT ROW INDEX, not a clock predicate, so it can never claim
a different trip at the same frozen clock. Offsetting technicals (one on each
team at one clock) shoot no free throws by rule: the scan finds none, nothing
is armed, and nothing changes.

RESIDUAL. Two technicals on the same team at the same frozen clock still
resolve to one armed marker each in row order, and the verified target's own
same-clock scanner shares the same limitation; the residual against
`data/processed/models/free_throw/technical_target_verified_trips_v1.parquet`
is reported in `docs/tests/event_layer_technical_lookahead_2026-09-18.md`.

KNOWN AMBIGUITY, MEASURED NOT PATCHED. Rules 5 and 6 cannot separate a
genuine two-shot shooting foul from a bonus trip once the bonus is in force,
because both produce two free throws. `ft_trip_ambiguous` is written on every
such row, and `docs/tests/possessions_build_2026-09-10.md` reports the share
and a base-rate estimate of how many of them are really shooting fouls. No
row is reassigned on that estimate: the flag is the honest representation.

--------------------------------------------------------------------------
DERIVED COLUMNS
--------------------------------------------------------------------------
  start_clock / end_clock   `secondsRemaining` (per period) at the previous
                            possession's terminal event -- the period length
                            for the period's first possession -- and at this
                            possession's own terminal event.
  duration_s                start_clock - end_clock, clipped at 0.
  start_score_diff          offence score minus defence score at possession
                            start, from the feed's running `homeScore` /
                            `awayScore` after the previous terminal event.
  n_chances                 1 + offensive rebounds that started a new chance.
  start_reason              period_start | DREB | TOV | made_FG | made_FT |
                            end_of_and_one | other -- what handed this team
                            the ball.
  is_transition             `duration_s <= 8` AND `start_reason` in
                            {DREB, TOV}. A proxy, labelled as one.
  on_floor_h1..h5/a1..a5    the ten `home_on_*` / `away_on_*` ids on the
                            possession's first event. Null before 2024: CBBD
                            `onFloor` is empty at the source in 2022-2023 and
                            90 / 98 / 97% complete in 2024 / 2025 / 2026
                            (L13).

EXCLUSIONS. A game enters the table only if `games_universe.is_d1_game` and
not `pbp_truncated`, joined on `cbbd_game_id`. Season 2026 is built (it is
data, not a fit) but `cbb_sim.data.seal.assert_not_sealed` guards every
training and selection path that reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.pbp.events import INERT_CLASSES, classify_frame, load_plays

DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")

#: VERSIONED OUTPUT (2026-09-10). `v1` is the original build. `v2` adds the
#: rim-location override (`cbb_sim.pbp.events`, module docstring) and the
#: per-chance attempt counts the first-chance-only style rates need. The two
#: directories coexist deliberately: other workers hold long-running reads on
#: `v1` while `v2` is being built and validated, and the default stays `v1`
#: until the PM switches it in one place -- here.
POSSESSION_VERSIONS: dict[str, Path] = {
    "v1": Path("data/processed/possessions"),
    "v2": Path("data/processed/possessions_v2"),
    # v3 = v2's event layer PLUS the bounded same-clock technical-free-throw
    # lookahead (see TECHNICAL FREE THROWS below). A NEW SIBLING DIRECTORY:
    # v1 and v2 are untouched and remain what every current consumer reads.
    "v3": Path("data/processed/possessions_v3"),
}
DEFAULT_POSSESSION_VERSION = "v1"
DEFAULT_OUT_DIR = POSSESSION_VERSIONS[DEFAULT_POSSESSION_VERSION]

#: Which versions turn the technical lookahead ON. Keyed by version label so a
#: caller that only knows "v3" gets the right machine without a second flag.
#: v1/v2 are False, so the default path is bit-identical to what is on disk.
VERSION_TECH_LOOKAHEAD: dict[str, bool] = {"v1": False, "v2": False, "v3": True}
DEFAULT_TECH_LOOKAHEAD = False

#: Bound on the same-clock technical lookahead: how many non-free-throw rows
#: the scan may step over between a `Technical Foul` row and the free throw it
#: produced. DERIVED, not typed: the observed gap between the two rows is 1 row
#: on 99.0% of affected moments and never exceeds 6 in seasons 2022-2025
#: (census in `docs/tests/event_layer_technical_lookahead_2026-09-18.md`
#: section 1). The bound exists so a feed gap cannot make the scan walk an
#: unbounded distance; the frozen-clock condition is the primary guard.
TECH_LOOKAHEAD_MAX_ROWS = 6


def possessions_dir(version: str | None = None, poss_dir: Path | str | None = None) -> Path:
    """Resolve a possessions directory from a version label.

    An explicit `poss_dir` always wins (so a caller can point at a scratch
    build); otherwise `version` selects from `POSSESSION_VERSIONS`, defaulting
    to `DEFAULT_POSSESSION_VERSION`. An unknown label raises rather than
    falling back, for the same reason `map_play_type` raises: a silent default
    would read the wrong table and never say so."""
    if poss_dir is not None:
        return Path(poss_dir)
    v = DEFAULT_POSSESSION_VERSION if version is None else str(version)
    try:
        return POSSESSION_VERSIONS[v]
    except KeyError as exc:
        raise KeyError(
            f"unknown possessions version {v!r}; known versions: "
            f"{sorted(POSSESSION_VERSIONS)}"
        ) from exc

#: Terminal-event vocabulary of a possession / chance.
TERMINAL_EVENTS: tuple[str, ...] = (
    "TOV", "FGA_rim", "FGA_jump2", "FGA_3", "FT_trip_shooting", "FT_trip_bonus",
    "end_period", "unknown",
)

#: The six modelled classes (end_period is handled by the clock model; unknown
#: is a data-gap bucket).
MODELLED_TERMINALS: tuple[str, ...] = (
    "TOV", "FGA_rim", "FGA_jump2", "FGA_3", "FT_trip_shooting", "FT_trip_bonus",
)

#: NCAA men's bonus thresholds, expressed as the fouling team's PRIOR foul
#: count in the period (so `>= 6` means "this foul is the 7th"). Validated
#: against the free-throw-trip length distribution -- see the module docstring.
BONUS_PRIOR_FOULS = 6
DOUBLE_BONUS_PRIOR_FOULS = 9

PERIOD_LENGTH_S = {1: 1200, 2: 1200}
OT_LENGTH_S = 300

TRANSITION_MAX_S = 8


def period_length(period: int) -> int:
    return PERIOD_LENGTH_S.get(int(period), OT_LENGTH_S)


# ---------------------------------------------------------------------------
# Trip classification (importable and unit-testable on its own)
# ---------------------------------------------------------------------------
def technical_ft_index(
    i: int,
    cls,
    team,
    period,
    sec,
    beneficiary: int,
    n: int,
    game=None,
    max_rows: int = TECH_LOOKAHEAD_MAX_ROWS,
) -> int:
    """Row index of the free throw produced by the `technical` row at `i`, or
    -1 if there is none.

    THE BOUNDED SAME-CLOCK LOOKAHEAD (module docstring, "TECHNICAL FREE
    THROWS"). A technical free-throw trip is shot at a dead ball, so every one
    of its rows carries the IDENTICAL `(period, secondsRemaining)` as the
    technical itself -- the same invariant the verified target's own scanner
    (`scripts/build_ft_technical_target_v1.py::scan_technical_attempts`) is
    built on. The scan therefore walks forward only while the clock is frozen,
    steps over at most `max_rows` non-free-throw rows (the administrative
    turnover and/or the duplicate personal foul), and stops at:

      * the first free throw -- returned if it belongs to `beneficiary`, and
        -1 otherwise (a free throw by the OTHER team at this clock is a
        different trip and the technical's own trip, if any, is not reachable
        without guessing);
      * a field-goal attempt or a period boundary -- a live-ball event, so the
        clock was not really frozen and this is not one dead-ball sequence;
      * a game boundary, when `game` is supplied (it is not needed by the
        per-game state machine, which is already sliced to one game).

    Pure, index-only and importable so both event-layer builders -- this
    module's state machine and `cbb_sim.models.event_stream._attach_trips` --
    apply exactly one rule rather than two that drift apart."""
    stepped = 0
    k = i + 1
    while k < n and stepped <= max_rows:
        if game is not None and game[k] != game[i]:
            return -1
        if period[k] != period[i] or sec[k] != sec[i]:
            return -1
        c = cls[k]
        if c == "FT_made" or c == "FT_missed":
            return k if int(team[k]) == int(beneficiary) else -1
        if c in ("FGA_rim", "FGA_jump2", "FGA_3", "end_period", "end_game"):
            return -1
        k += 1
        stepped += 1
    return -1


def classify_ft_trip(n_ft: int, first_made: bool, prior_fouls: int) -> tuple[str, bool]:
    """Return `(terminal_event, ambiguous)` for a real (non-technical,
    non-and-one) free-throw trip. Rules 3-6 of the module docstring."""
    if n_ft >= 3:
        return "FT_trip_shooting", False
    if prior_fouls < BONUS_PRIOR_FOULS:
        return "FT_trip_shooting", False
    if prior_fouls < DOUBLE_BONUS_PRIOR_FOULS:
        if n_ft == 1:
            return ("FT_trip_bonus", False) if not first_made else ("FT_trip_shooting", True)
        if n_ft == 2:
            if not first_made:
                return "FT_trip_shooting", False
            return "FT_trip_bonus", True
        return "FT_trip_shooting", True
    if n_ft == 2:
        return "FT_trip_bonus", True
    return "FT_trip_shooting", True


# ---------------------------------------------------------------------------
# Per-game state machine
# ---------------------------------------------------------------------------
@dataclass
class _Chance:
    chance_number: int
    start_clock: int
    start_reason: str
    terminal_event: str = "unknown"
    end_clock: int = 0
    fga_rim: int = 0
    fgm_rim: int = 0
    fga_jump2: int = 0
    fgm_jump2: int = 0
    fga_3: int = 0
    fgm_3: int = 0
    fta: int = 0
    ftm: int = 0
    points: int = 0
    and_one: bool = False
    ft_trip_ambiguous: bool = False
    stolen: bool = False


@dataclass
class _Possession:
    game_id: int
    cbbd_game_id: int
    season: int
    period: int
    poss_index: int
    offense_team_id: int
    defense_team_id: int
    start_clock: int
    start_score_diff: int
    start_reason: str
    off_team_fouls: int
    def_team_fouls: int
    on_floor: tuple = ()
    chances: list = field(default_factory=list)
    tech_points_off: int = 0
    tech_points_def: int = 0


class _GameMachine:
    """One forward pass over one game's non-inert events."""

    def __init__(self, game_meta: dict, ev: dict[str, np.ndarray],
                 tech_lookahead: bool = DEFAULT_TECH_LOOKAHEAD) -> None:
        self.m = game_meta
        self.ev = ev
        self.tech_lookahead = bool(tech_lookahead)
        #: exact row index of a free throw already identified as a technical's
        #: own (module docstring, "TECHNICAL FREE THROWS"). -1 = nothing armed,
        #: which is always the case when `tech_lookahead` is False.
        self.pending_tech_ft_idx = -1
        self.n_tech_lookahead_hits = 0
        self.n = len(ev["cls"])
        # The machine works on SIDE, not on team ids: CBBD's `teamId` is
        # CBBD's own team key, not the ESPN/hoopR `team_id` the rest of the
        # project is keyed on, and `isHomeTeam` is populated on 100% of rows
        # that carry a `teamId`. `_emit` maps side -> the universe's ESPN ids.
        self.home = 0
        self.away = 1
        self.possessions: list[_Possession] = []
        self.cur: _Possession | None = None
        self.chance: _Chance | None = None
        self.awaiting_reb = False
        self.pending_terminal = "unknown"
        self.period = int(ev["period"][0]) if self.n else 1
        self.team_fouls = {self.home: 0, self.away: 0}
        self.prev_end_clock = period_length(self.period)
        self.prev_home_score = 0
        self.prev_away_score = 0
        self.poss_index = 0
        self.next_start_reason = "period_start"
        self.n_unknown_team = 0
        self.n_mismatch_closes = 0
        self.n_tech_trips = 0
        self.n_admin_orebs = 0
        # technical free throws shot while NO possession was open, buffered
        # until one is (or flushed onto the last possession at game end) so
        # that the score reconciliation always balances
        self.pending_tech = [0, 0]  # [home_side_points, away_side_points]

    # -- helpers ------------------------------------------------------------
    def _other(self, team: int) -> int:
        return self.away if team == self.home else self.home

    def _open(self, team: int, i: int) -> None:
        if self.cur is not None:
            self._close(self.pending_terminal, i)
        self.poss_index += 1
        on_floor = tuple(self.ev["on_floor"][i]) if self.ev["on_floor"] is not None else ()
        opp = self._other(team)
        diff = (self.prev_home_score - self.prev_away_score) if team == self.home else (
            self.prev_away_score - self.prev_home_score
        )
        self.cur = _Possession(
            game_id=self.m["game_id"],
            cbbd_game_id=self.m["cbbd_game_id"],
            season=self.m["season"],
            period=self.period,
            poss_index=self.poss_index,
            offense_team_id=team,
            defense_team_id=opp,
            start_clock=self.prev_end_clock,
            start_score_diff=int(diff),
            start_reason=self.next_start_reason,
            off_team_fouls=self.team_fouls.get(team, 0),
            def_team_fouls=self.team_fouls.get(opp, 0),
            on_floor=on_floor,
        )
        if self.pending_tech[0] or self.pending_tech[1]:
            self.cur.tech_points_off += self.pending_tech[team]
            self.cur.tech_points_def += self.pending_tech[opp]
            self.pending_tech = [0, 0]
        self.chance = _Chance(chance_number=1, start_clock=self.prev_end_clock,
                              start_reason=self.next_start_reason)
        self.cur.chances.append(self.chance)
        self.awaiting_reb = False
        self.pending_terminal = "unknown"

    def _close(self, terminal: str, i: int, reason_next: str = "other") -> None:
        if self.cur is None:
            return
        clock = int(self.ev["sec"][i]) if i < self.n else 0
        self.chance.terminal_event = terminal
        self.chance.end_clock = clock
        self.possessions.append(self.cur)
        self.prev_end_clock = clock
        self.prev_home_score = int(self.ev["hs"][i]) if i < self.n else self.prev_home_score
        self.prev_away_score = int(self.ev["as_"][i]) if i < self.n else self.prev_away_score
        self.cur = None
        self.chance = None
        self.awaiting_reb = False
        self.pending_terminal = "unknown"
        self.next_start_reason = reason_next

    def _ensure(self, team: int, i: int) -> None:
        """MISMATCH GUARD: make `team` the offense, closing a stale possession
        for the other team first."""
        if self.cur is None:
            self._open(team, i)
        elif self.cur.offense_team_id != team:
            self.n_mismatch_closes += 1
            self._close(self.pending_terminal, i, reason_next="other")
            self._open(team, i)

    def _next_real(self, i: int) -> int:
        """Index of the next event (inert classes are already dropped)."""
        return i + 1 if i + 1 < self.n else -1

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        ev = self.ev
        cls = ev["cls"]
        team = ev["team"]
        per = ev["period"]
        i = 0
        while i < self.n:
            c = cls[i]
            p = int(per[i])
            if p != self.period:
                # A period boundary that was not announced by end_period.
                if self.cur is not None:
                    self._close("end_period", i - 1 if i else 0, reason_next="period_start")
                self.period = p
                self.team_fouls = {self.home: 0, self.away: 0}
                self.prev_end_clock = period_length(p)
                self.next_start_reason = "period_start"

            if c in ("end_period", "end_game"):
                if self.cur is not None:
                    self._close("end_period", i, reason_next="period_start")
                self.team_fouls = {self.home: 0, self.away: 0}
                nxt = self._next_real(i)
                self.period = int(per[nxt]) if nxt >= 0 else self.period
                self.prev_end_clock = period_length(self.period)
                self.next_start_reason = "period_start"
                i += 1
                continue

            t = team[i]
            if t < 0:
                # No team on the row and it is not a period boundary: nothing
                # to do (should not happen -- inert rows are already dropped).
                self.n_unknown_team += 1
                i += 1
                continue
            t = int(t)

            if c in ("FGA_rim", "FGA_jump2", "FGA_3"):
                i = self._handle_fga(i, c, t)
                continue
            if c in ("FT_made", "FT_missed"):
                # `pending_tech_ft_idx` is an exact row index and stays -1 in
                # the default build, so this is a no-op there.
                is_tech = self.pending_tech_ft_idx == i
                if is_tech:
                    self.pending_tech_ft_idx = -1
                    self.n_tech_lookahead_hits += 1
                i = self._handle_ft_trip(i, t, prior_fouls=None, technical=is_tech)
                continue
            if c == "OREB":
                i = self._handle_oreb(i, t)
                continue
            if c == "DREB":
                self._handle_dreb(i, t)
                i += 1
                continue
            if c == "TOV":
                self._ensure(t, i)
                self.chance.stolen = bool(ev["stolen"][i])
                self._close("TOV", i, reason_next="TOV")
                i += 1
                continue
            if c == "steal":
                # unpaired steal: promote to a turnover by the other team
                self._ensure(self._other(t), i)
                self.chance.stolen = True
                self._close("TOV", i, reason_next="TOV")
                i += 1
                continue
            if c == "foul":
                i = self._handle_foul(i, t)
                continue
            if c == "technical":
                i = self._handle_technical(i, t)
                continue
            # DeadBallReb, unknown: no-op
            i += 1

        if self.cur is not None:
            self._close("end_period", self.n - 1, reason_next="period_start")
        if (self.pending_tech[0] or self.pending_tech[1]) and self.possessions:
            last = self.possessions[-1]
            last.tech_points_off += self.pending_tech[last.offense_team_id]
            last.tech_points_def += self.pending_tech[last.defense_team_id]
            self.pending_tech = [0, 0]

    # -- handlers -----------------------------------------------------------
    def _handle_fga(self, i: int, c: str, t: int) -> int:
        ev = self.ev
        self._ensure(t, i)
        made = bool(ev["made"][i])
        ch = self.chance
        if c == "FGA_rim":
            ch.fga_rim += 1
            ch.fgm_rim += int(made)
        elif c == "FGA_jump2":
            ch.fga_jump2 += 1
            ch.fgm_jump2 += int(made)
        else:
            ch.fga_3 += 1
            ch.fgm_3 += int(made)
        if made:
            ch.points += 3 if c == "FGA_3" else 2
            # and-one lookahead: foul (any team) then exactly one FT by t at
            # the same clock.
            j = i + 1
            if (
                j + 1 < self.n
                and ev["cls"][j] == "foul"
                and ev["cls"][j + 1] in ("FT_made", "FT_missed")
                and int(ev["team"][j + 1]) == t
                and int(ev["sec"][j + 1]) == int(ev["sec"][i])
            ):
                trip_end = j + 1
                n_ft = 0
                while trip_end < self.n and ev["cls"][trip_end] in ("FT_made", "FT_missed") \
                        and int(ev["team"][trip_end]) == t:
                    n_ft += 1
                    trip_end += 1
                if n_ft == 1:
                    k = j + 1
                    ch.and_one = True
                    ch.fta += 1
                    ft_made = ev["cls"][k] == "FT_made"
                    ch.ftm += int(ft_made)
                    ch.points += int(ft_made)
                    self.team_fouls[int(ev["team"][j])] = self.team_fouls.get(int(ev["team"][j]), 0) + 1
                    self._close(c, k, reason_next="made_FG")
                    return k + 1
            self._close(c, i, reason_next="made_FG")
            return i + 1
        self.awaiting_reb = True
        self.pending_terminal = c
        return i + 1

    def _handle_oreb(self, i: int, t: int) -> int:
        ev = self.ev
        # Administrative rebound between two free throws of one trip. The test
        # is "the very next event is another free throw by the same team":
        # a GENUINE offensive rebound can only be followed by a free throw if
        # a new foul is committed first, and that foul is its own row in
        # between, so adjacency alone is sufficient. (The clock is NOT part of
        # the test: the feed sometimes stamps the reset rebound a second or two
        # after the free throw it follows.)
        j = i + 1
        if (
            j < self.n
            and ev["cls"][j] in ("FT_made", "FT_missed")
            and int(ev["team"][j]) == t
        ):
            self.n_admin_orebs += 1
            return i + 1
        self._ensure(t, i)
        if self.cur is not None:
            self.chance.terminal_event = self.pending_terminal
            self.chance.end_clock = int(ev["sec"][i])
            new_ch = _Chance(
                chance_number=self.chance.chance_number + 1,
                start_clock=int(ev["sec"][i]),
                start_reason="OREB",
            )
            self.cur.chances.append(new_ch)
            self.chance = new_ch
            self.awaiting_reb = False
            self.pending_terminal = "unknown"
        return i + 1

    def _handle_dreb(self, i: int, t: int) -> None:
        if self.cur is None:
            # no open possession: charge the team that is not the rebounder
            self._open(self._other(t), i)
        self._close(self.pending_terminal, i, reason_next="DREB")

    def _handle_foul(self, i: int, t: int) -> int:
        ev = self.ev
        prior = self.team_fouls.get(t, 0)
        j = i + 1
        if j < self.n and ev["cls"][j] in ("FT_made", "FT_missed"):
            self.team_fouls[t] = prior + 1
            if self.pending_tech_ft_idx == j:
                # A duplicate `PersonalFoul` row logged for a technical foul
                # (10% of the affected moments: the same player's name appears
                # on both rows at the same frozen clock). The foul still counts
                # toward the team total -- that is what the row says, and the
                # bonus thresholds were validated on counting every
                # `PersonalFoul` row -- but the free throws belong to the
                # technical, so the main loop routes them, not this handler.
                return i + 1
            return self._handle_ft_trip(j, int(ev["team"][j]), prior_fouls=prior, technical=False)
        self.team_fouls[t] = prior + 1
        return i + 1

    def _handle_technical(self, i: int, t: int) -> int:
        ev = self.ev
        if self.tech_lookahead:
            ben = self._other(t)
            j = technical_ft_index(i, ev["cls"], ev["team"], ev["period"], ev["sec"],
                                   ben, self.n)
            if j == i + 1:
                # the free throw is the very next row: the pre-fix path, taken
                # unchanged so the common case stays byte for byte identical
                return self._handle_ft_trip(j, int(ev["team"][j]), prior_fouls=None,
                                            technical=True)
            if j > i + 1:
                # administrative rows in between. Arm the exact row index and
                # let the main loop process those rows normally -- the inserted
                # `Lost Ball Turnover` is a real, box-counted turnover (module
                # docstring) and must NOT be swallowed here.
                self.pending_tech_ft_idx = j
            return i + 1
        j = i + 1
        if j < self.n and ev["cls"][j] in ("FT_made", "FT_missed"):
            return self._handle_ft_trip(j, int(ev["team"][j]), prior_fouls=None, technical=True)
        return i + 1

    def _collect_trip(self, i: int, t: int) -> tuple[int, list[bool]]:
        """Run of consecutive FTs by `t`, allowing administrative rebounds and
        same-clock foul rows in between."""
        ev = self.ev
        made: list[bool] = []
        k = i
        while k < self.n:
            c = ev["cls"][k]
            if c in ("FT_made", "FT_missed") and int(ev["team"][k]) == t:
                made.append(c == "FT_made")
                k += 1
                continue
            if c in ("OREB", "DeadBallReb"):
                nxt = k + 1
                if nxt < self.n and ev["cls"][nxt] in ("FT_made", "FT_missed") and int(ev["team"][nxt]) == t:
                    k += 1
                    self.n_admin_orebs += 1
                    continue
            break
        return k, made

    def _handle_ft_trip(self, i: int, t: int, prior_fouls: int | None, technical: bool) -> int:
        k, made = self._collect_trip(i, t)
        if not made:
            return i + 1
        n_ft = len(made)
        pts = int(sum(made))

        if technical:
            self.n_tech_trips += 1
            if self.cur is not None:
                if t == self.cur.offense_team_id:
                    self.cur.tech_points_off += pts
                else:
                    self.cur.tech_points_def += pts
            else:
                self.pending_tech[t] += pts
            return k

        if prior_fouls is None:
            # A trip with no immediately preceding foul row (feed gap). Use the
            # DEFENCE's current foul count, which is the same quantity the
            # bonus rule needs.
            prior_fouls = self.team_fouls.get(self._other(t), 0)

        terminal, ambiguous = classify_ft_trip(n_ft, bool(made[0]), int(prior_fouls))
        self._ensure(t, i)
        ch = self.chance
        ch.fta += n_ft
        ch.ftm += pts
        ch.points += pts
        ch.ft_trip_ambiguous = ambiguous
        last_idx = k - 1
        if made[-1]:
            self._close(terminal, last_idx, reason_next="made_FT")
        else:
            self.awaiting_reb = True
            self.pending_terminal = terminal
        return k


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------
def _fix_flipped_sides(p: pd.DataFrame, side: np.ndarray, has_team: np.ndarray) -> np.ndarray:
    """Repair games where CBBD's `isHomeTeam` flag is inverted.

    CBBD's `homeScore` / `awayScore` running columns agree with the schedule's
    final score, but the per-row `isHomeTeam` flag is inverted on a small
    number of games (5 of 6,135 in 2025). The running score settles it without
    any external key: on a scoring play exactly one of the two score columns
    increases, and it must be the one belonging to the scoring team. Each
    game's rows vote, and the majority decides -- so a handful of stray rows
    cannot flip a correct game, and a genuinely inverted game flips wholesale.
    """
    hs = pd.to_numeric(p["homeScore"], errors="coerce").ffill().fillna(0).to_numpy()
    as_ = pd.to_numeric(p["awayScore"], errors="coerce").ffill().fillna(0).to_numpy()
    game = p["gameId"].to_numpy()
    same_game = np.concatenate([[False], game[1:] == game[:-1]])
    dh = np.concatenate([[0.0], np.diff(hs)]) * same_game
    da = np.concatenate([[0.0], np.diff(as_)]) * same_game
    is_h = has_team & (side == 0)
    is_a = has_team & (side == 1)
    ok = (is_h & (dh > 0)) | (is_a & (da > 0))
    swap = (is_h & (da > 0)) | (is_a & (dh > 0))
    votes = pd.DataFrame({"game": game, "ok": ok, "swap": swap}).groupby("game", sort=False).sum()
    flipped = set(votes.index[votes["swap"] > votes["ok"]].tolist())
    if not flipped:
        return side
    mask = pd.Series(game).isin(flipped).to_numpy()
    return np.where(mask & has_team, 1 - side, side)


def _prepare_events(plays: pd.DataFrame, with_on_floor: bool) -> dict:
    cls = classify_frame(plays)
    keep = ~cls.isin(list(INERT_CLASSES))
    p = plays.loc[keep.to_numpy()].reset_index(drop=True)
    c = cls.loc[keep.to_numpy()].reset_index(drop=True)

    is_home = p["isHomeTeam"]
    if is_home.dtype == object:
        is_home = is_home.map({True: True, False: False})
    is_home = is_home.astype("boolean")
    has_team = pd.to_numeric(p["teamId"], errors="coerce").notna() & is_home.notna()
    side = np.where(is_home.fillna(False).to_numpy(), 0, 1)
    side = _fix_flipped_sides(p, side, has_team.to_numpy())
    team = np.where(has_team.to_numpy(), side, -1).astype("int64")
    sec = pd.to_numeric(p["secondsRemaining"], errors="coerce").fillna(0).astype("int64").to_numpy()
    period = pd.to_numeric(p["period"], errors="coerce").fillna(1).astype("int64").to_numpy()
    hs = pd.to_numeric(p["homeScore"], errors="coerce").ffill().fillna(0).astype("int64").to_numpy()
    as_ = pd.to_numeric(p["awayScore"], errors="coerce").ffill().fillna(0).astype("int64").to_numpy()
    made_col = p["shot_made"]
    if made_col.dtype == object:
        made_col = made_col.map({True: True, False: False})
    made = made_col.astype("boolean").fillna(p["scoringPlay"].astype("boolean")).fillna(False).to_numpy(dtype=bool)

    cls_arr = c.to_numpy(dtype=object)
    # steal pairing: a `steal` adjacent to a TOV is decoration
    is_steal = cls_arr == "steal"
    is_tov = cls_arr == "TOV"
    prev_tov = np.concatenate([[False], is_tov[:-1]])
    next_tov = np.concatenate([is_tov[1:], [False]])
    paired = is_steal & (prev_tov | next_tov)
    stolen = np.zeros(len(cls_arr), dtype=bool)
    stolen[np.where(paired & prev_tov)[0] - 1] = True
    idx_np = np.where(paired & next_tov)[0] + 1
    stolen[idx_np[idx_np < len(stolen)]] = True
    cls_arr = cls_arr.copy()
    cls_arr[paired] = "_paired_steal"
    drop = cls_arr == "_paired_steal"

    on_floor = None
    if with_on_floor:
        cols = [f"home_on_{k}" for k in range(1, 6)] + [f"away_on_{k}" for k in range(1, 6)]
        on_floor = p[cols].to_numpy(dtype="float64")

    sel = ~drop
    out = {
        "cls": cls_arr[sel],
        "team": team[sel],
        "sec": sec[sel],
        "period": period[sel],
        "hs": hs[sel],
        "as_": as_[sel],
        "made": made[sel],
        "stolen": stolen[sel],
        "game": p["gameId"].to_numpy()[sel],
        "on_floor": on_floor[sel] if on_floor is not None else None,
    }
    return out


def segment_season(
    season: int,
    universe: pd.DataFrame,
    pbp_dir: Path | str = "data/raw/cbbd/pbp",
    progress_every: int = 1500,
    tech_lookahead: bool = DEFAULT_TECH_LOOKAHEAD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Segment one season into possessions and chances.

    `universe` must already be filtered to `is_d1_game & ~pbp_truncated`.

    `tech_lookahead` turns on the bounded same-clock technical free-throw
    lookahead (module docstring, "TECHNICAL FREE THROWS"). It defaults to
    False, so every existing caller gets the build that is already on disk;
    `VERSION_TECH_LOOKAHEAD` maps a version label to the right value."""
    u = universe[universe["season"] == int(season)]
    meta = {
        int(r.cbbd_game_id): {
            "game_id": int(r.game_id),
            "cbbd_game_id": int(r.cbbd_game_id),
            "season": int(r.season),
            "home_team_id": int(r.home_team_id),
            "away_team_id": int(r.away_team_id),
        }
        for r in u.itertuples()
    }
    plays = load_plays(season, pbp_dir=pbp_dir, game_ids=set(meta))
    with_on_floor = int(season) >= 2024
    ev = _prepare_events(plays, with_on_floor=with_on_floor)

    games = ev["game"]
    bounds = np.flatnonzero(np.concatenate([[True], games[1:] != games[:-1], [True]]))
    poss_rows: list[dict] = []
    chance_rows: list[dict] = []
    diag = {"n_games": 0, "n_mismatch_closes": 0, "n_tech_trips": 0, "n_admin_orebs": 0,
            "n_unknown_team": 0, "n_games_no_plays": 0, "n_tech_lookahead_hits": 0,
            "tech_lookahead": bool(tech_lookahead)}

    for b in range(len(bounds) - 1):
        lo, hi = int(bounds[b]), int(bounds[b + 1])
        gid = int(games[lo])
        gm = meta.get(gid)
        if gm is None:
            continue
        sub = {
            "cls": ev["cls"][lo:hi], "team": ev["team"][lo:hi], "sec": ev["sec"][lo:hi],
            "period": ev["period"][lo:hi], "hs": ev["hs"][lo:hi], "as_": ev["as_"][lo:hi],
            "made": ev["made"][lo:hi], "stolen": ev["stolen"][lo:hi],
            "on_floor": ev["on_floor"][lo:hi] if ev["on_floor"] is not None else None,
        }
        machine = _GameMachine(gm, sub, tech_lookahead=tech_lookahead)
        machine.run()
        diag["n_games"] += 1
        diag["n_mismatch_closes"] += machine.n_mismatch_closes
        diag["n_tech_trips"] += machine.n_tech_trips
        diag["n_tech_lookahead_hits"] += machine.n_tech_lookahead_hits
        diag["n_admin_orebs"] += machine.n_admin_orebs
        diag["n_unknown_team"] += machine.n_unknown_team
        _emit(machine, poss_rows, chance_rows)

    poss = pd.DataFrame(poss_rows)
    chances = pd.DataFrame(chance_rows)
    poss = _finalise(poss)
    if len(chances):
        chances = chances.astype({"chance_number": "int16"})
    return poss, chances, diag


def _emit(machine: _GameMachine, poss_rows: list, chance_rows: list) -> None:
    on_floor_ok = machine.ev["on_floor"] is not None
    side_to_team = {0: int(machine.m["home_team_id"]), 1: int(machine.m["away_team_id"])}
    for p in machine.possessions:
        chs = p.chances
        last = chs[-1]
        row = {
            "game_id": p.game_id,
            "cbbd_game_id": p.cbbd_game_id,
            "season": p.season,
            "period": p.period,
            "poss_index": p.poss_index,
            "offense_team_id": side_to_team[p.offense_team_id],
            "defense_team_id": side_to_team[p.defense_team_id],
            "offense_is_home": p.offense_team_id == 0,
            "start_clock": p.start_clock,
            "end_clock": last.end_clock,
            "duration_s": max(0, p.start_clock - last.end_clock),
            "start_score_diff": p.start_score_diff,
            "start_reason": p.start_reason,
            "n_chances": len(chs),
            "oreb_count": len(chs) - 1,
            "terminal_event": last.terminal_event,
            "fga_rim": sum(c.fga_rim for c in chs),
            "fgm_rim": sum(c.fgm_rim for c in chs),
            "fga_jump2": sum(c.fga_jump2 for c in chs),
            "fgm_jump2": sum(c.fgm_jump2 for c in chs),
            "fga_3": sum(c.fga_3 for c in chs),
            "fgm_3": sum(c.fgm_3 for c in chs),
            "fta": sum(c.fta for c in chs),
            "ftm": sum(c.ftm for c in chs),
            "points": sum(c.points for c in chs),
            "and_one": any(c.and_one for c in chs),
            "ft_trip_ambiguous": any(c.ft_trip_ambiguous for c in chs),
            "stolen": any(c.stolen for c in chs),
            "off_team_fouls": p.off_team_fouls,
            "def_team_fouls": p.def_team_fouls,
            "off_in_bonus": p.def_team_fouls >= BONUS_PRIOR_FOULS,
            "off_in_double_bonus": p.def_team_fouls >= DOUBLE_BONUS_PRIOR_FOULS,
            "is_transition": (max(0, p.start_clock - last.end_clock) <= TRANSITION_MAX_S)
                             and p.start_reason in ("DREB", "TOV"),
            "tech_points_off": p.tech_points_off,
            "tech_points_def": p.tech_points_def,
        }
        if on_floor_ok and len(p.on_floor) == 10:
            of = p.on_floor
            for k in range(5):
                row[f"on_floor_h{k + 1}"] = of[k]
            for k in range(5):
                row[f"on_floor_a{k + 1}"] = of[5 + k]
        poss_rows.append(row)

        for c in chs:
            chance_rows.append({
                "game_id": p.game_id,
                "season": p.season,
                "period": p.period,
                "poss_index": p.poss_index,
                "chance_number": c.chance_number,
                "offense_team_id": side_to_team[p.offense_team_id],
                "defense_team_id": side_to_team[p.defense_team_id],
                "offense_is_home": p.offense_team_id == 0,
                "terminal_event": c.terminal_event,
                "start_clock": c.start_clock,
                "end_clock": c.end_clock,
                "duration_s": max(0, c.start_clock - c.end_clock),
                "start_score_diff": p.start_score_diff,
                "points": c.points,
                # Per-CHANCE attempt counts. The possession row sums these
                # across every chance, first and continuation alike, which is
                # what let a continuation-chance labelling defect leak into a
                # FIRST-chance model's own predictors through `off_rim_c`
                # (docs/tests/shot_classification_diag_2026-09-10.md section 6).
                # Carrying them per chance is what makes a first-chance-only
                # style rate computable at all.
                "fga_rim": c.fga_rim,
                "fgm_rim": c.fgm_rim,
                "fga_jump2": c.fga_jump2,
                "fgm_jump2": c.fgm_jump2,
                "fga_3": c.fga_3,
                "fgm_3": c.fgm_3,
                "fta": c.fta,
                "ftm": c.ftm,
                "and_one": c.and_one,
                "ft_trip_ambiguous": c.ft_trip_ambiguous,
                "off_in_bonus": p.def_team_fouls >= BONUS_PRIOR_FOULS,
                "off_in_double_bonus": p.def_team_fouls >= DOUBLE_BONUS_PRIOR_FOULS,
                "is_transition": (max(0, p.start_clock - c.end_clock) <= TRANSITION_MAX_S)
                                 and c.start_reason in ("DREB", "TOV") and c.chance_number == 1,
                "start_reason": c.start_reason,
            })


def _finalise(poss: pd.DataFrame) -> pd.DataFrame:
    if not len(poss):
        return poss
    int_cols = ["game_id", "cbbd_game_id", "season", "period", "poss_index",
                "offense_team_id", "defense_team_id", "start_clock", "end_clock",
                "duration_s", "start_score_diff", "n_chances", "oreb_count",
                "fga_rim", "fgm_rim", "fga_jump2", "fgm_jump2", "fga_3", "fgm_3",
                "fta", "ftm", "points", "off_team_fouls", "def_team_fouls",
                "tech_points_off", "tech_points_def"]
    for c in int_cols:
        poss[c] = pd.to_numeric(poss[c], errors="coerce").astype("int64")
    return poss
