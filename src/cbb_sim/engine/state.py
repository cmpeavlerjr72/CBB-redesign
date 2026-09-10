"""
state.py -- GameState as a struct of arrays over N concurrent simulations.

Deliverable 1. Every field is a NumPy array whose row `i` is one simulation
(one (game, seed) pair), so every step of the possession loop is a vector
operation over the whole batch and there is no per-game Python object and no
per-game loop.

RULE-ERA FLAGS LIVE HERE (CLAUDE.md: "Rule-era flags live in GameState, not
baked into sub-models"; Decision 5; L18). `bonus_prior_fouls` and
`double_bonus_prior_fouls` are read from
`data/processed/models/free_throw/bonus_era.json` per season and carried as
per-simulation arrays, not as module constants and not baked into
`cbb_sim.models.free_throw` -- which is exactly why `load_bonus_era` exists
there at all. The 2022-2026 derivation lands on (6, 9) in every season (no era
boundary in the data, L18); the engine still carries it as state so a future
season whose thresholds differ needs no code change.

SIGN AND SCOPE CONVENTIONS. Three sub-models disagree, so the state stores the
primitive and each adapter derives its own view:

    seconds_remaining   WITHIN the period: 1200 in a half, 300 in an overtime.
                        clock.py, possession_outcome.py, fg_make.py,
                        free_throw.py and rotation.py all want this.
                        usage.py wants (2 - period) * 1200 + this during
                        regulation, which `usage_sec_remaining()` derives.
    score_diff          stored as home minus away. `off_score_diff()` returns
                        offence minus defence, which is what
                        possession_outcome, clock, fg_make, free_throw and
                        usage all define `score_diff` to be;
                        `home_score_diff()` is the only thing
                        `rotation.RotationState` accepts.
    off                 0 when the home team has the ball, 1 when the away
                        team does. Side index 0 is ALWAYS home.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Period lengths in seconds. Regulation halves are 20 minutes; an overtime
#: period is 5 minutes (deliverable 2f).
HALF_SECONDS = 1200
OT_SECONDS = 300

#: The previous possession's end type, in `cbb_sim.models.clock.PREV_END_LEVELS`
#: order. `period_start` is code 0 and is the dummy coding's reference level,
#: so the five dummies the clock arm reads are codes 1..5.
PREV_END_LEVELS: tuple[str, ...] = (
    "period_start", "DREB", "TOV", "made_FG", "made_FT", "other",
)
PREV_END_CODE = {v: i for i, v in enumerate(PREV_END_LEVELS)}

#: Personal fouls that disqualify a player (`rotation.FOUL_OUT`).
FOUL_OUT = 5

#: Hard stop on overtime periods. Ties resolve through overtime, never by
#: discarding the game (CLAUDE.md); the cap exists only so that a pathological
#: parameter set cannot hang a run, and the number of simulations that reach it
#: is reported rather than hidden.
MAX_OT_PERIODS = 12

DEFAULT_BONUS_ERA = Path("data/processed/models/free_throw/bonus_era.json")


def load_bonus_era(path: Path | str | None = None) -> dict[int, tuple[int, int]]:
    """(bonus_prior_fouls, double_bonus_prior_fouls) per season.

    Reads the same artifact `cbb_sim.models.free_throw.load_bonus_era` reads.
    It is re-read here rather than imported so that `state.py` has no
    import-time dependency on a model module: the ENGINE owns the rule era."""
    p = Path(path) if path is not None else DEFAULT_BONUS_ERA
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): (int(v["bonus_prior_fouls"]), int(v["double_bonus_prior_fouls"]))
            for k, v in raw["by_season"].items()}


#: Per-team box columns the results contract asks for (`contract.BOX_STATS`).
BOX_STATS: tuple[str, ...] = (
    "fga3", "fga2_rim", "fga2_jump", "fta", "tov", "oreb", "dreb",
)
#: Per-player box columns (`contract.REQUIRED_PLAYER_COLUMNS` minus the keys
#: and minutes). `ast` is a placeholder the engine has no model for and is
#: written as 0 with that stated, never as a fabricated number.
PLAYER_STATS: tuple[str, ...] = ("pts", "reb", "fga", "fg3a", "fta")


@dataclass
class GameState:
    """N concurrent simulations. Row `i` is (game_index[i], seed[i])."""

    # ---- identity --------------------------------------------------------
    game_index: np.ndarray      # (N,) int32 -> row of the per-game input tables
    seed: np.ndarray            # (N,) int32
    n_slots: int                # roster slots per team in the input tables

    # ---- clock and period ------------------------------------------------
    period: np.ndarray          # (N,) int16, 1 and 2 regulation, 3+ overtime
    seconds_remaining: np.ndarray   # (N,) int16, WITHIN the period
    n_ot: np.ndarray            # (N,) int8
    active: np.ndarray          # (N,) bool, game still in progress

    # ---- score and possession --------------------------------------------
    pts: np.ndarray             # (N, 2) int16, side 0 = home
    off: np.ndarray             # (N,) int8, 0 = home has the ball
    first_off: np.ndarray       # (N,) int8, who won the opening tip
    chance_number: np.ndarray   # (N,) int8, 1 on a first chance
    prev_end: np.ndarray        # (N,) int8, code into PREV_END_LEVELS
    poss_duration: np.ndarray   # (N,) int16, duration drawn for the live possession
    poss_count: np.ndarray      # (N, 2) int16, possessions used by each side

    # ---- fouls and the rule era ------------------------------------------
    team_fouls: np.ndarray      # (N, 2) int8, reset every period
    bonus_prior_fouls: np.ndarray          # (N,) int8   RULE ERA
    double_bonus_prior_fouls: np.ndarray   # (N,) int8   RULE ERA

    # ---- the ten on the floor --------------------------------------------
    on_floor: np.ndarray        # (N, 2, 5) int16, roster SLOT indices
    player_fouls: np.ndarray    # (N, 2, S) int8
    player_seconds: np.ndarray  # (N, 2, S) float32

    # ---- box score -------------------------------------------------------
    box: dict[str, np.ndarray] = field(default_factory=dict)         # stat -> (N, 2) int16
    player_box: dict[str, np.ndarray] = field(default_factory=dict)  # stat -> (N, 2, S) int16

    # ---- diagnostics: counted, reported, never corrected ------------------
    diag: dict[str, int] = field(default_factory=dict)

    # -- derived views -----------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.game_index)

    def rows(self) -> np.ndarray:
        return np.arange(self.n)

    def off_pts(self) -> np.ndarray:
        return self.pts[self.rows(), self.off]

    def def_pts(self) -> np.ndarray:
        return self.pts[self.rows(), 1 - self.off]

    def off_score_diff(self) -> np.ndarray:
        """Offence minus defence. possession_outcome, clock, fg_make,
        free_throw and usage all define `score_diff` this way."""
        return self.off_pts().astype(np.int32) - self.def_pts().astype(np.int32)

    def home_score_diff(self) -> np.ndarray:
        """Home minus away: the only convention `RotationState` accepts."""
        return self.pts[:, 0].astype(np.int32) - self.pts[:, 1].astype(np.int32)

    def usage_sec_remaining(self) -> np.ndarray:
        """usage.py's `sec_remaining`: seconds left in REGULATION during
        regulation, seconds left in the overtime period during overtime."""
        p = self.period.astype(np.int32)
        sr = self.seconds_remaining.astype(np.int32)
        return np.where(p <= 2, (2 - p) * HALF_SECONDS + sr, sr)

    def def_team_fouls(self) -> np.ndarray:
        return self.team_fouls[self.rows(), 1 - self.off]

    def in_bonus(self) -> np.ndarray:
        """`off_in_bonus`: the OFFENCE shoots bonus free throws because the
        DEFENCE has reached the era's team-foul threshold."""
        return (self.def_team_fouls() >= self.bonus_prior_fouls).astype(np.int8)

    def in_double_bonus(self) -> np.ndarray:
        return (self.def_team_fouls() >= self.double_bonus_prior_fouls).astype(np.int8)

    def possessions(self) -> np.ndarray:
        """The contract's `possessions`, PER TEAM, which is what
        `reference.load_actual_possessions` grades against (a box-derived count
        averaged over the two sides), so the two sides are averaged here too."""
        return 0.5 * (self.poss_count[:, 0].astype(np.float32)
                      + self.poss_count[:, 1].astype(np.float32))

    def n_periods(self) -> np.ndarray:
        """The contract's `n_periods`: 2 for regulation, 3 for one overtime."""
        return (2 + self.n_ot).astype(np.int16)

    def bump(self, key: str, amount: int = 1) -> None:
        self.diag[key] = self.diag.get(key, 0) + int(amount)


def new_state(game_index: np.ndarray, seed: np.ndarray, seasons: np.ndarray,
              n_slots: int, bonus_era: dict[int, tuple[int, int]],
              first_off: np.ndarray) -> GameState:
    """Open N simulations at the opening tip of period 1.

    `seasons[i]` is the season of simulation `i`'s game and selects that
    simulation's rule-era thresholds. A season absent from `bonus_era` falls
    back to the most recent season present; that fallback is stated here rather
    than hidden behind a module constant."""
    n = len(game_index)
    known = sorted(bonus_era)
    if not known:
        raise ValueError("bonus_era is empty; the engine cannot carry a rule era it has not read")
    b = np.empty(n, dtype=np.int8)
    d = np.empty(n, dtype=np.int8)
    for i, s in enumerate(np.asarray(seasons, dtype=np.int64)):
        key = int(s) if int(s) in bonus_era else known[-1]
        b[i], d[i] = bonus_era[key]

    return GameState(
        game_index=np.asarray(game_index, dtype=np.int32),
        seed=np.asarray(seed, dtype=np.int32),
        n_slots=int(n_slots),
        period=np.ones(n, dtype=np.int16),
        seconds_remaining=np.full(n, HALF_SECONDS, dtype=np.int16),
        n_ot=np.zeros(n, dtype=np.int8),
        active=np.ones(n, dtype=bool),
        pts=np.zeros((n, 2), dtype=np.int16),
        off=np.asarray(first_off, dtype=np.int8).copy(),
        first_off=np.asarray(first_off, dtype=np.int8).copy(),
        chance_number=np.ones(n, dtype=np.int8),
        prev_end=np.full(n, PREV_END_CODE["period_start"], dtype=np.int8),
        poss_duration=np.zeros(n, dtype=np.int16),
        poss_count=np.zeros((n, 2), dtype=np.int16),
        team_fouls=np.zeros((n, 2), dtype=np.int8),
        bonus_prior_fouls=b,
        double_bonus_prior_fouls=d,
        on_floor=np.zeros((n, 2, 5), dtype=np.int16),
        player_fouls=np.zeros((n, 2, n_slots), dtype=np.int8),
        player_seconds=np.zeros((n, 2, n_slots), dtype=np.float32),
        box={k: np.zeros((n, 2), dtype=np.int16) for k in BOX_STATS},
        player_box={k: np.zeros((n, 2, n_slots), dtype=np.int16) for k in PLAYER_STATS},
    )
