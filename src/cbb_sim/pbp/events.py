"""
events.py -- the canonical CBBD play-by-play event vocabulary.

WHY THIS FILE EXISTS. `docs/SIM_GUARDRAILS.md` section 4 ("Event vocabulary
drift") and `docs/LEARNINGS.md` L6 make one requirement non-negotiable: *every
pbp-derived feature must be built from an explicit mapping table with a test
that fails on an unknown type*. CBBD's `playType` vocabulary is 24 values over
seasons 2022-2026 and it is NOT stable across them (`Substitution` only from
2025; `Coach's Challenge (Stands|Overturned)` and a bare `Shot` only in 2026;
a `Not Available` placeholder only in 2022-2023). `PLAY_TYPE_TO_EVENT` below
lists all 24 explicitly; `map_play_type` raises `UnknownPlayTypeError` on
anything else, and `tests/test_possession_outcome.py` asserts that it does.

SOURCE. `data/raw/cbbd/pbp/plays_{season}.parquet`, pulled by
`scripts/pull_cbbd_pbp.py` from `https://api.collegebasketballdata.com/plays/date`.
Audit: `docs/tests/data_audit_cbbd_pbp_2026-09-10.md`.

--------------------------------------------------------------------------
THE TWO-LEVEL MAPPING
--------------------------------------------------------------------------
`playType` alone is not enough to name the event we need, for two reasons, so
the mapping is two-level: `playType -> EventFamily`, then a row-level resolver
that uses the shot columns.

  1. `JumpShot` covers both two-point jumpers and three-point attempts.
  2. `MadeFreeThrow` covers BOTH made and missed free throws -- the type name
     is a misnomer of the upstream ESPN feed. Verified against `playText`
     ("... made Free Throw." vs "... missed Free Throw." in 2022-2025,
     "makes"/"misses" in 2026) and against `shot_made` / `scoringPlay`.

Canonical event classes produced by `classify_row` / `classify_frame`:

    FGA_rim        DunkShot, LayUpShot, TipShot            (and 'Shot'/'Not
                   Available' rows that carry shot_range == 'rim')
    FGA_jump2      JumpShot that is not a three
    FGA_3          JumpShot that IS a three
    FT_made        MadeFreeThrow with shot_made == True
    FT_missed      MadeFreeThrow with shot_made != True
    OREB           Offensive Rebound
    DREB           Defensive Rebound
    DeadBallReb    Dead Ball Rebound
    TOV            Lost Ball Turnover  (+ the Steal pairing, see below)
    steal          Steal -- the defensive credit half of a TOV, not an event
                   of its own; see STEAL PAIRING
    foul           PersonalFoul -- shooting vs non-shooting is NOT in the
                   feed and is inferred downstream from the FT sequence that
                   follows plus the running team-foul count; see
                   `cbb_sim.pbp.possessions.classify_ft_trip`
    technical      Technical Foul
    timeout        OfficialTVTimeOut, ShortTimeOut, RegularTimeOut
    sub            Substitution
    block          Block Shot   (defensive credit attached to a missed FGA;
                   carries no possession information of its own)
    jumpball       Jumpball
    end_period     End Period
    end_game       End Game
    challenge      Coach's Challenge (Stands), Coach's Challenge (Overturned)
    unknown        Not Available rows that carry no shot information

--------------------------------------------------------------------------
HOW 2 VS 3 IS DETERMINED, AND ITS FAILURE RATE
--------------------------------------------------------------------------
Three independent signals exist on every `JumpShot` row:

  (a) `shot_range` in {'three_pointer', 'jumper', 'rim', 'free_throw'} --
      populated on 100.00% of rows with `shootingPlay == True` in all five
      seasons (2,261,348 de-duplicated JumpShot rows checked, zero nulls).
  (b) `playText` containing the case-insensitive substring "three"
      ("... missed Three Point Jumper.").
  (c) `scoreValue` in {0, 1, 2, 3}.

Measured agreement over all 2,261,348 de-duplicated JumpShot rows, seasons
2022-2026:

  * `shot_range` is NULL on **0** rows with `shootingPlay == True` in any of
    the five seasons, so (a) always resolves.
  * (a) and (b) agree on **100.000%** of rows that have a non-null
    `playText`: every `shot_range == 'three_pointer'` row contains "three",
    and no `shot_range == 'jumper'` row does. Zero disagreeing rows in any
    season.
  * `playText` is NULL on 2,845 JumpShot rows (0.126%): 0 in 2022, 25 in
    2023, 1,737 in 2024, 981 in 2025, 102 in 2026. All carry a populated
    `shot_range`, so (a) resolves every row that (b) cannot.
  * `scoreValue` is the unreliable signal and is NOT used: it is 0 or null on
    5,958 JumpShot rows (missed shots in 2023-2025 where the feed did not
    populate it) and 1 on 605 more, and it says 3 while `shot_range` and
    `playText` both say a two-point jumper on **152 rows** (2 in 2023, 22 in
    2024, 18 in 2025, 110 in 2026).

THE RULE USED HERE: **`shot_range == 'three_pointer'` -> FGA_3, otherwise
FGA_jump2.** `playText` is used only as a fallback when `shot_range` is null
(which happens on zero rows in the current five-season extract), and
`scoreValue` is used only as a last resort.

FAILURE RATE OF THE RULE: 152 rows in 2,261,348 = **0.0067%** -- the rows
where `scoreValue` claims 3 and `shot_range`/`playText` claim a two-point
jumper. Inspection of their `playText` ("Bryan Ndjonga misses 25-foot
turnaround jump shot", i.e. a shot from beyond the arc described as a jump
shot) suggests `scoreValue` is the correct signal on some of them and the
range label is the correct one on others; at 5 rows per 100,000 the choice
cannot move any per-100-possession rate by a measurable amount. It is
recorded, not patched. `three_point_signal_disagreement()` recomputes the
number from the raw frame so the claim is auditable rather than typed in.

--------------------------------------------------------------------------
STEAL PAIRING
--------------------------------------------------------------------------
`Lost Ball Turnover` is CBBD's single turnover bucket (bad pass, travel,
offensive foul, shot-clock violation, stolen ball all land there) and carries
`teamId` == the team that lost the ball. `Steal` is a separate row carrying
`teamId` == the team that took it, and is *adjacent* to its `Lost Ball
Turnover` row 99.92% of the time (83,274 of 83,337 checked in a 2025 slice).
It is therefore NOT an independent possession event: the possession machine
consumes `TOV` and treats `steal` as decoration, recording `stolen=True` on
the turnover. A `Steal` with no adjacent `Lost Ball Turnover` (0.08%) still
implies a change of possession and is promoted to a `TOV` charged to the
other team by `cbb_sim.pbp.possessions`.

--------------------------------------------------------------------------
DUPLICATE ROWS (a data-quality finding, handled here)
--------------------------------------------------------------------------
`scripts/pull_cbbd_pbp.py` pulls `/plays/date` day by day, and a game that
straddles local midnight is returned by two consecutive `query_date` calls.
The raw files therefore carry **18-19% exactly-duplicated rows** (449,518 of
2,310,520 in 2022 ... 626,041 of 3,537,141 in 2026), touching 21-25% of
games. `(gameId, id)` is unique after de-duplication in every season.
`load_plays` de-duplicates on `(gameId, id)`; nothing downstream should read
the raw files without it. NOTE: the per-playType counts printed in
`docs/tests/data_audit_cbbd_pbp_2026-09-10.md` are pre-de-duplication and are
inflated by that factor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PBP_DIR = Path("data/raw/cbbd/pbp")

# ---------------------------------------------------------------------------
# Event families
# ---------------------------------------------------------------------------
# The *family* is what `playType` alone can tell us. `classify_frame` resolves
# SHOT -> FGA_rim / FGA_jump2 / FGA_3 and FT -> FT_made / FT_missed using the
# shot columns.
FAM_SHOT_RIM = "SHOT_RIM"
FAM_SHOT_JUMP = "SHOT_JUMP"
FAM_SHOT_ANY = "SHOT_ANY"  # 'Shot' / 'Not Available' rows: range column decides
FAM_FT = "FT"
FAM_OREB = "OREB"
FAM_DREB = "DREB"
FAM_DEADBALL_REB = "DeadBallReb"
FAM_TOV = "TOV"
FAM_STEAL = "steal"
FAM_FOUL = "foul"
FAM_TECHNICAL = "technical"
FAM_TIMEOUT = "timeout"
FAM_SUB = "sub"
FAM_BLOCK = "block"
FAM_JUMPBALL = "jumpball"
FAM_END_PERIOD = "end_period"
FAM_END_GAME = "end_game"
FAM_CHALLENGE = "challenge"
FAM_UNKNOWN = "unknown"

#: THE mapping table. All 24 CBBD playType values observed in
#: data/raw/cbbd/pbp/plays_{2022..2026}.parquet, each with an explicit family.
#: Adding a season means re-running the audit and adding any new value HERE,
#: deliberately -- never by widening a fallback.
PLAY_TYPE_TO_EVENT: dict[str, str] = {
    # --- field goal attempts -------------------------------------------------
    "DunkShot": FAM_SHOT_RIM,
    "LayUpShot": FAM_SHOT_RIM,
    "TipShot": FAM_SHOT_RIM,
    "JumpShot": FAM_SHOT_JUMP,
    "Shot": FAM_SHOT_ANY,  # 2026 only, 1 row total; shot_range decides
    # --- free throws ---------------------------------------------------------
    "MadeFreeThrow": FAM_FT,  # covers makes AND misses (see module docstring)
    # --- rebounds ------------------------------------------------------------
    "Offensive Rebound": FAM_OREB,
    "Defensive Rebound": FAM_DREB,
    "Dead Ball Rebound": FAM_DEADBALL_REB,
    # --- turnovers -----------------------------------------------------------
    "Lost Ball Turnover": FAM_TOV,
    "Steal": FAM_STEAL,
    # --- fouls ---------------------------------------------------------------
    "PersonalFoul": FAM_FOUL,
    "Technical Foul": FAM_TECHNICAL,
    # --- stoppages / administration -----------------------------------------
    "OfficialTVTimeOut": FAM_TIMEOUT,
    "ShortTimeOut": FAM_TIMEOUT,
    "RegularTimeOut": FAM_TIMEOUT,
    "Substitution": FAM_SUB,
    "Block Shot": FAM_BLOCK,
    "Jumpball": FAM_JUMPBALL,
    "Coach's Challenge (Stands)": FAM_CHALLENGE,
    "Coach's Challenge (Overturned)": FAM_CHALLENGE,
    # --- period boundaries ---------------------------------------------------
    "End Period": FAM_END_PERIOD,
    "End Game": FAM_END_GAME,
    # --- explicit placeholder ------------------------------------------------
    # 105 rows, 2022-2023 only. ~25% of them carry shootingPlay == True with a
    # populated shot_range and are promoted to the matching FGA class by
    # classify_frame; the rest are genuinely uninterpretable.
    "Not Available": FAM_SHOT_ANY,
}

#: Canonical event class names emitted by classify_frame.
EVENT_CLASSES: tuple[str, ...] = (
    "FGA_rim", "FGA_jump2", "FGA_3",
    "FT_made", "FT_missed",
    "OREB", "DREB", "DeadBallReb",
    "TOV", "steal",
    "foul", "technical",
    "timeout", "sub", "block", "jumpball",
    "end_period", "end_game", "challenge", "unknown",
)

#: Classes that carry no possession information at all and are dropped from the
#: event stream before the possession state machine runs.
INERT_CLASSES: frozenset[str] = frozenset({"timeout", "sub", "block", "jumpball", "challenge"})

SHOT_CLASSES: frozenset[str] = frozenset({"FGA_rim", "FGA_jump2", "FGA_3"})
FT_CLASSES: frozenset[str] = frozenset({"FT_made", "FT_missed"})


class UnknownPlayTypeError(ValueError):
    """Raised when a CBBD `playType` is not in `PLAY_TYPE_TO_EVENT`.

    The guardrail (`docs/SIM_GUARDRAILS.md` section 4) is that an unseen event
    type must stop the pipeline, not fall through to a silent default: hoopR
    and CBBD both added event types mid-history, and a silent default would
    have mis-segmented every possession containing one.
    """


def map_play_type(play_type: str) -> str:
    """`playType` -> event family. Raises `UnknownPlayTypeError` on anything
    not explicitly listed in `PLAY_TYPE_TO_EVENT`."""
    try:
        return PLAY_TYPE_TO_EVENT[play_type]
    except KeyError as exc:
        raise UnknownPlayTypeError(
            f"unknown CBBD playType {play_type!r}. Add it to PLAY_TYPE_TO_EVENT in "
            f"{__name__} with an explicit event family before proceeding "
            "(docs/SIM_GUARDRAILS.md section 4: every pbp-derived feature is built "
            "from an explicit mapping table with a test that fails on an unknown type)."
        ) from exc


def assert_known_play_types(play_types) -> None:
    """Raise on the first unmapped playType in an iterable/Series of them."""
    for pt in pd.unique(pd.Series(list(play_types), dtype="object").dropna()):
        map_play_type(str(pt))


# ---------------------------------------------------------------------------
# Row-level classification
# ---------------------------------------------------------------------------
def _is_three(shot_range: pd.Series, play_text: pd.Series, score_value: pd.Series) -> np.ndarray:
    """Three-point indicator for shot rows. Rule and measured failure rate are
    documented in the module docstring: `shot_range == 'three_pointer'` first,
    the case-insensitive "three" substring of `playText` as the fallback when
    `shot_range` is null, `scoreValue == 3` as the last resort."""
    rng = shot_range.astype("string").str.lower()
    txt = play_text.astype("string").str.lower()
    sv = pd.to_numeric(score_value, errors="coerce")

    out = np.where(
        rng.notna().to_numpy(),
        (rng == "three_pointer").to_numpy(),
        np.where(
            txt.notna().to_numpy(),
            txt.str.contains("three", na=False).to_numpy(),
            (sv == 3).to_numpy(),
        ),
    )
    return out.astype(bool)


def classify_frame(plays: pd.DataFrame) -> pd.Series:
    """Canonical event class for every row of a CBBD plays frame.

    Requires columns: playType, shot_range, playText, scoreValue, shot_made,
    scoringPlay, shootingPlay. Raises `UnknownPlayTypeError` on an unmapped
    `playType`."""
    fam = plays["playType"].map(map_play_type)

    rng = plays["shot_range"].astype("string").str.lower()
    is_three = _is_three(plays["shot_range"], plays["playText"], plays["scoreValue"])

    ev = pd.Series(fam.to_numpy(), index=plays.index, dtype="object")

    # SHOT_RIM / SHOT_JUMP resolve directly.
    ev[fam == FAM_SHOT_RIM] = "FGA_rim"
    jump = fam == FAM_SHOT_JUMP
    ev[jump & is_three] = "FGA_3"
    ev[jump & ~is_three] = "FGA_jump2"

    # SHOT_ANY ('Shot', 'Not Available'): only a row that actually carries shot
    # information becomes an FGA; the rest are `unknown`.
    any_shot = fam == FAM_SHOT_ANY
    has_shot_info = any_shot & plays["shootingPlay"].fillna(False).astype(bool) & rng.notna()
    ev[any_shot] = "unknown"
    ev[has_shot_info & (rng == "rim")] = "FGA_rim"
    ev[has_shot_info & (rng == "free_throw")] = "FT_missed"  # resolved below
    ev[has_shot_info & (rng == "three_pointer")] = "FGA_3"
    ev[has_shot_info & (rng == "jumper") & is_three] = "FGA_3"
    ev[has_shot_info & (rng == "jumper") & ~is_three] = "FGA_jump2"

    # FT: made flag. `shot_made` is authoritative (matches `scoringPlay` on
    # 99.99% of rows and survives the 2026 playText format change from
    # "made/missed" to "makes/misses"); `scoringPlay` is the fallback.
    made = plays["shot_made"]
    if made.dtype == object:
        made = made.map({True: True, False: False})
    made = made.astype("boolean")
    made = made.fillna(plays["scoringPlay"].astype("boolean")).fillna(False).to_numpy(dtype=bool)
    ft = fam == FAM_FT
    ev[ft & made] = "FT_made"
    ev[ft & ~made] = "FT_missed"

    return ev.astype("string")


def three_point_signal_disagreement(plays: pd.DataFrame) -> dict[str, int]:
    """Audit helper: recompute the 2-vs-3 signal agreement numbers quoted in
    the module docstring straight off a plays frame, so the documented failure
    rate is reproducible rather than asserted."""
    jump = plays["playType"] == "JumpShot"
    j = plays.loc[jump]
    rng = j["shot_range"].astype("string").str.lower()
    txt = j["playText"].astype("string").str.lower()
    sv = pd.to_numeric(j["scoreValue"], errors="coerce")
    range_three = (rng == "three_pointer")
    text_three = txt.str.contains("three", na=False)
    return {
        "n_jumpshots": int(len(j)),
        "n_shot_range_null": int(rng.isna().sum()),
        "n_playtext_null": int(txt.isna().sum()),
        "n_range_vs_text_disagree": int((range_three & txt.notna() & ~text_three).sum()
                                        + (~range_three & text_three).sum()),
        "n_scorevalue3_but_not_three": int(((sv == 3) & ~range_three).sum()),
        "n_scorevalue_missing_or_zero": int(((sv == 0) | sv.isna()).sum()),
    }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
PLAY_COLUMNS: tuple[str, ...] = (
    "gameId", "season", "id", "playType", "isHomeTeam", "teamId", "opponentId",
    "homeScore", "awayScore", "period", "secondsRemaining",
    "scoringPlay", "shootingPlay", "scoreValue", "shot_made", "shot_range", "playText",
    "shot_shooter_id",
    "home_on_1", "home_on_2", "home_on_3", "home_on_4", "home_on_5",
    "away_on_1", "away_on_2", "away_on_3", "away_on_4", "away_on_5",
)


def load_plays(
    season: int,
    pbp_dir: Path | str = DEFAULT_PBP_DIR,
    game_ids: set[int] | None = None,
    columns: tuple[str, ...] = PLAY_COLUMNS,
) -> pd.DataFrame:
    """Load one season of CBBD plays, de-duplicated on `(gameId, id)` and
    sorted into event order.

    De-duplication is mandatory, not hygiene: the day-by-day pull returns a
    game that straddles local midnight twice, and 18-19% of raw rows in every
    season are exact duplicates (module docstring, "DUPLICATE ROWS").

    ORDER KEY. `id` within `gameId`. Validated against the clock: sorting by
    `id` produces a non-increasing `secondsRemaining` within (game, period) on
    100.000% of 2022-2024 rows and 99.99% of 2025-2026 rows (202 / 2.19M and
    768 / 2.91M inversions respectively). The alternative key `sourceId` is
    22x worse (4,430 and 17,491 inversions), so `id` is the key.
    """
    path = Path(pbp_dir) / f"plays_{int(season)}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing CBBD pbp file: {path}")
    df = pd.read_parquet(path, columns=list(columns))
    if game_ids is not None:
        df = df[df["gameId"].isin(game_ids)]
    df = df.drop_duplicates(subset=["gameId", "id"], keep="first")
    df = df.sort_values(["gameId", "id"], kind="stable").reset_index(drop=True)
    return df
