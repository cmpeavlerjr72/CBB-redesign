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
THE RIM-LOCATION OVERRIDE (added 2026-09-10)
--------------------------------------------------------------------------
ESPN's 2025-season feed mistags a large batch of true tip-in / putback
attempts `JumpShot` instead of `TipShot`/`LayUpShot`. Evidence, in full:
`docs/tests/shot_classification_diag_2026-09-10.md`. The short version:

  * The continuation-chance ("after an offensive rebound") rim share is
    37.9 / 38.5 / 38.1% in 2022-2024, collapses to 31.6% in 2025, and returns
    to 39.2% in 2026 -- a one-season round trip that is not a basketball
    change.
  * `shot_range` cannot catch it. It is generated in lockstep with `playType`
    at the source: it is `'rim'` on 100.000% of Dunk/LayUp/Tip rows and never
    `'rim'` on a `JumpShot` row, in any of the five seasons.
  * `playText` cannot catch it either: the mistagged rows carry the generic
    "made/missed Jumper." text, and only 42 of 462,118 `JumpShot` rows in 2025
    contain the substring "tip" (FEWER than the neighbouring seasons).
  * hoopR's independently-built parse of the same broadcast feed carries the
    IDENTICAL wrong tag on 99.6% of the identical plays, so this is an
    upstream vendor defect, not a CBBD parsing bug and not a bug here.
  * `shot_location_x` / `shot_location_y` -- a separate ESPN sub-system (the
    shot chart) -- DID get these plays right: they sit on the same canned
    at-the-rim placeholder pixel that legitimate `TipShot` rows use.

So the one signal the vendor got right is geometry, and the override uses it:
a row the feed calls a two-point `JumpShot` whose recorded release point is
within `RIM_OVERRIDE_MAX_FT` of a basket is an `FGA_rim`.

SCOPE, deliberately narrow. The override is applied ONLY to rows already
resolved to `FGA_jump2`. It can therefore only ever move a two-point jumper
to a rim attempt: it never touches `DunkShot`/`LayUpShot`/`TipShot` rows, and
it never touches three-point rows, so the 2-vs-3 split (and hence every
points total and the 3PA/2PA reconciliation against hoopR's box score) is
bit-identical before and after. Three-point rows are excluded by construction
rather than by argument, even though `JumpShot(3)` release distance is a
stable ~24.6 ft in every season with no near-rim rows to clip.

THE THRESHOLD IS DERIVED, NOT TYPED. `rim_family_distance_quantiles()`
recomputes, from any plays frame, the release-distance distribution of the
rows the feed ITSELF labels a rim attempt (`DunkShot`, `LayUpShot`,
`TipShot`). `RIM_OVERRIDE_MAX_FT` is a stated quantile of that distribution
-- the value, the quantile it is, and the measured per-season effect of every
candidate on the continuation-chance rim/jump2 shares are recorded in
`docs/tests/possessions_build_v2_2026-09-10.md` section 3.1. The selection
rule was fixed in advance: the threshold must return 2025 to the
2022-2024/2026 band AND move every clean season by less than 0.5 pp, or the
override is over-reaching and is not adopted.

COORDINATE SYSTEM. CBBD passes ESPN's shot-chart coordinates through
unchanged: a 0-940 x 0-500 full-court grid in tenths of a foot. The two
baskets sit at raw `(52.5, 250)` and `(887.5, 250)`; that calibration is
empirical, not assumed -- it is the pair of points that puts the median
`DunkShot` release at 2.27-2.32 ft from the hoop in every one of the five
seasons, which is what a dunk is. `shot_distance_ft` returns the distance to
the NEARER basket, so it needs no knowledge of which direction a team is
attacking.

COVERAGE. `shot_location_y` is populated on 78-88% of shooting rows in
2022-2024 and 97-99% in 2025-2026. A row with no location cannot be
overridden and keeps its feed label; that is a miss, not a false positive,
and it is the conservative direction.

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

# ---------------------------------------------------------------------------
# Rim-location override (module docstring, "THE RIM-LOCATION OVERRIDE")
# ---------------------------------------------------------------------------
#: The two baskets in ESPN's raw shot-chart coordinates (a 0-940 x 0-500
#: full-court grid in tenths of a foot). Empirical, not assumed: this is the
#: pair of points that puts the median `DunkShot` release at 2.27-2.32 ft in
#: every one of the five seasons. `basket_calibration()` recomputes the check.
BASKET_XY: tuple[tuple[float, float], ...] = ((52.5, 250.0), (887.5, 250.0))

#: playTypes the feed itself calls a rim attempt. The override's threshold is a
#: quantile of THEIR release-distance distribution, so the cutoff is derived
#: from the feed's own notion of "at the rim" rather than chosen by eye.
RIM_PLAY_TYPES: tuple[str, ...] = ("DunkShot", "LayUpShot", "TipShot")

#: DERIVED, not typed: **the 50th percentile of `DunkShot` release distance**,
#: pooled over 110,069 located `DunkShot` rows in seasons 2022-2026. It is
#: 2.27 ft in 2022/2023/2024 and 2.32 ft in 2025/2026, so the pooled median is
#: 2.27 and no season is doing the choosing.
#:
#: Chosen from a ladder of eight stated quantiles of that distribution by a
#: rule fixed before the rungs were measured -- every clean season (2022-2024,
#: 2026) must move by less than 0.5 pp on the continuation-chance `FGA_rim` and
#: `FGA_jump2` shares, and among the rungs that clear that gate the one that
#: leaves 2025 closest to the clean-season band wins. Measured effect at 2.27
#: ft: 2025's continuation rim share moves 31.63 -> 38.13 (the clean band is
#: 38.18-39.38, so it lands 0.06 pp outside it against the 6.27 pp gap it
#: started with) and the worst any clean season moves is 0.34 pp. The whole
#: ladder is in `docs/tests/possessions_build_v2_2026-09-10.md` section 3.1
#: and is regenerated by `scripts/build_possessions.py --threshold-ladder`.
#:
#: A value <= 0 disables the override entirely and reproduces v1.
RIM_OVERRIDE_MAX_FT: float = 2.27


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


def shot_distance_ft(plays: pd.DataFrame) -> np.ndarray:
    """Release distance in FEET from the nearer basket, per row.

    `NaN` where the shot-chart coordinates are absent (the row was never
    located) or where the frame does not carry the columns at all. Using the
    nearer of the two baskets means no knowledge of which direction a team is
    attacking is needed, which is the one thing the coordinate feed does not
    say."""
    if "shot_location_x" not in plays.columns or "shot_location_y" not in plays.columns:
        return np.full(len(plays), np.nan)
    x = pd.to_numeric(plays["shot_location_x"], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(plays["shot_location_y"], errors="coerce").to_numpy(dtype="float64")
    d = np.full(len(plays), np.inf)
    for bx, by in BASKET_XY:
        d = np.minimum(d, np.sqrt((x - bx) ** 2 + (y - by) ** 2))
    return d / 10.0  # the grid is in tenths of a foot


def classify_frame(plays: pd.DataFrame, rim_override_max_ft: float | None = None) -> pd.Series:
    """Canonical event class for every row of a CBBD plays frame.

    Requires columns: playType, shot_range, playText, scoreValue, shot_made,
    scoringPlay, shootingPlay. Uses shot_location_x / shot_location_y when
    present (they are in `PLAY_COLUMNS`, so the production path always has
    them). Raises `UnknownPlayTypeError` on an unmapped `playType`.

    `rim_override_max_ft` defaults to the module constant
    `RIM_OVERRIDE_MAX_FT`; pass an explicit number to sweep it, or a value
    <= 0 to reproduce the pre-override behaviour exactly."""
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

    # RIM-LOCATION OVERRIDE. See the module docstring section of that name for
    # the evidence, the scope and how the threshold is derived. It is applied
    # LAST, and keyed on the resolved class rather than on the `JumpShot`
    # play-type mask, so that its stated scope -- "a row already resolved to
    # FGA_jump2" -- is literally what the code does, including for the handful
    # of `Not Available` / `Shot` rows that reach FGA_jump2 through the
    # SHOT_ANY branch above. It can therefore only ever move a two-point
    # jumper toward FGA_rim: no rim-tagged row, no three-point row and no free
    # throw can change class here, so every points total and the whole 2-vs-3
    # split are bit-identical before and after.
    max_ft = RIM_OVERRIDE_MAX_FT if rim_override_max_ft is None else float(rim_override_max_ft)
    if max_ft > 0:
        dist_ft = shot_distance_ft(plays)
        near_rim = ((ev == "FGA_jump2").to_numpy()
                    & np.isfinite(dist_ft) & (dist_ft <= max_ft))
        ev[near_rim] = "FGA_rim"

    return ev.astype("string")


RIM_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def rim_family_distance_quantiles(
    plays: pd.DataFrame, quantiles: tuple[float, ...] = RIM_QUANTILES
) -> dict[str, dict]:
    """THE FUNCTION THE OVERRIDE'S THRESHOLD IS DERIVED FROM.

    Release-distance quantiles, in feet from the nearer basket, of the rows the
    feed ITSELF labels a rim attempt (`DunkShot`, `LayUpShot`, `TipShot`), plus
    the same quantiles for two-point and three-point `JumpShot` rows as the
    contrast. Every candidate cutoff quoted anywhere in the documentation is a
    number out of this table, recomputed from the parquet rather than typed.

    Also reports location coverage per type, because a threshold derived from a
    6%-covered `TipShot` distribution (2022-2024) would be derived from almost
    nothing -- which is exactly why `DunkShot`, covered on 78-99% of rows in
    every season, is the type the adopted quantile is taken from."""
    out: dict[str, dict] = {}
    d_all = shot_distance_ft(plays)
    pt = plays["playType"].astype("string").fillna("").to_numpy(dtype=object)
    rng = plays["shot_range"].astype("string").str.lower().fillna("").to_numpy(dtype=object)
    groups: list[tuple[str, np.ndarray]] = [(t, pt == t) for t in RIM_PLAY_TYPES]
    groups.append(("rim_family", np.isin(pt, list(RIM_PLAY_TYPES))))
    groups.append(("JumpShot(2)", (pt == "JumpShot") & (rng != "three_pointer")))
    groups.append(("JumpShot(3)", (pt == "JumpShot") & (rng == "three_pointer")))
    for name, m in groups:
        d = d_all[m & np.isfinite(d_all)]
        out[name] = {
            "n_rows": int(m.sum()),
            "n_located": int(len(d)),
            "located_pct": round(100.0 * len(d) / max(int(m.sum()), 1), 2),
            "quantiles_ft": {f"p{int(q * 100)}": (round(float(np.quantile(d, q)), 3) if len(d) else None)
                             for q in quantiles},
        }
    return out


def basket_calibration(plays: pd.DataFrame) -> dict[str, float]:
    """Audit helper: the median `DunkShot` release distance implied by
    `BASKET_XY`. A dunk is taken at the rim, so this number must sit around
    2-2.5 ft; if a future pull changes the coordinate convention it will not,
    and the calibration claim in the module docstring fails visibly instead of
    silently mis-scaling every distance."""
    d = shot_distance_ft(plays)
    m = (plays["playType"].astype("string").fillna("").to_numpy(dtype=object) == "DunkShot") & np.isfinite(d)
    return {"n": int(m.sum()),
            "median_dunk_distance_ft": round(float(np.median(d[m])), 3) if m.sum() else float("nan")}


def rim_override_counts(plays: pd.DataFrame,
                        rim_override_max_ft: float | None = None) -> dict[str, int]:
    """How many rows the override actually moves, and how many it could not
    reach. Reported per season rather than asserted."""
    max_ft = RIM_OVERRIDE_MAX_FT if rim_override_max_ft is None else float(rim_override_max_ft)
    base = classify_frame(plays, rim_override_max_ft=0.0).to_numpy()
    d = shot_distance_ft(plays)
    j2 = base == "FGA_jump2"
    j3 = base == "FGA_3"
    return {
        "n_jump2": int(j2.sum()),
        "n_jump2_located": int((j2 & np.isfinite(d)).sum()),
        "n_overridden": int((j2 & np.isfinite(d) & (d <= max_ft)).sum()),
        "n_jump2_unlocated": int((j2 & ~np.isfinite(d)).sum()),
        # reported so the "three-point rows are never touched" claim is a
        # measurement rather than a promise
        "n_three_within_threshold_not_touched": int((j3 & np.isfinite(d) & (d <= max_ft)).sum()),
    }


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
    # the rim-location override reads these two (module docstring)
    "shot_location_x", "shot_location_y",
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
