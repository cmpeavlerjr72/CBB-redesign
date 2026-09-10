"""
event_stream.py -- the cleaned CBBD play-by-play stream the REBOUND (L3) and
FREE-THROW (L3) sub-models train on.

WHY THIS EXISTS, AND WHY IT IS NOT THE POSSESSION TABLE
=======================================================
`cbb_sim.pbp.possessions` answers "whose possession is this and how did it
end". Neither of the two models here can be built from its output alone:

  * The possession/chance tables record that a chance ENDED on a missed shot,
    but not which of the three things happened to the ball afterwards. An
    offensive rebound is visible only as "a further chance exists in this
    possession", a defensive rebound only as "the possession ended", and a
    DEAD-BALL rebound is invisible: `possessions.py` treats `DeadBallReb` as a
    no-op by design, so a missed shot recovered at a dead ball looks exactly
    like a missed shot recovered live. The rebound pre-registration asks for
    dead balls as a THIRD CLASS with its own reported share, so the class has
    to be read off the event that actually carries it.
  * Free-throw attempts have no row of their own in the possession tables at
    all: a trip contributes `fta` / `ftm` COUNTS to a chance. The shooter's
    identity -- which L15 says is the dominant term -- lives only on the pbp
    row (`participant_1_id`, populated on 100.000% of free-throw rows in
    2022-2024 and 99.999% in 2025).

So both models read the event stream directly. What they do NOT re-derive is
the possession segmentation: the as-of team-form features come from the
versioned possession/chance tables through
`cbb_sim.models.possession_outcome.build_team_form`, exactly as L3 round 1
built them.

WHAT THIS MODULE PRODUCES
=========================
One row per NON-INERT event of one season, in `id` order within `gameId`
(`cbb_sim.pbp.events.load_plays` de-duplicates and sorts), restricted to the
D-I non-truncated universe, with:

  cls              canonical event class (`cbb_sim.pbp.events.classify_frame`)
  side             0 = home, 1 = away, -1 = no team on the row; the CBBD
                   `isHomeTeam` inversion is repaired by
                   `possessions._fix_flipped_sides`, the same majority-vote
                   repair the possession build uses (data-defect row D2 of
                   `docs/models/change_ledger.md`)
  team_id/opp_id   ESPN team ids, resolved from the universe through `side`
  made             shot_made, falling back to scoringPlay (same rule as
                   `possessions._prepare_events`)
  blocked          a `Block Shot` row sits adjacent to this row at the same
                   `secondsRemaining`. Block rows are then DROPPED, so the
                   defensive credit becomes a flag on the attempt rather than
                   an event that would break "the next event is the rebound".
  fouls_own_prior / fouls_opp_prior
                   each team's PRIOR personal-foul count in this period
                   (strictly before this row) -- the quantity the NCAA bonus
                   rule is defined on
  trip_*           free-throw trip structure, see below
  player_id        `participant_1_id` (CBBD player id: the free-throw shooter,
                   the rebounder, the fouler)
  home_on_1..5 / away_on_1..5   CBBD on-floor ids, null before 2024 (L13)

FREE-THROW TRIPS
================
A trip is a run of consecutive free throws by one team. ESPN interleaves an
ADMINISTRATIVE rebound row ("Offensive Rebound" or "Dead Ball Rebound") between
two free throws of the same trip -- a dead-ball reset, not a live rebound. Those
rows are identified exactly as `possessions._collect_trip` identifies them (the
row is a rebound, the previous and next rows are both free throws by the SAME
team) and dropped from the stream, which is what makes "the next event after the
last free throw of a trip is the rebound" true.

Each trip carries:
  trip_len          number of attempts
  trip_pos          1-based index of this attempt inside the trip
  trip_last         this is the last attempt of the trip
  trip_first_made   the first attempt of the trip was made
  trip_cause        `foul` | `technical` | `none` -- the class of the row
                    immediately before the trip's first attempt
  trip_prior_fouls  the FOULING team's prior foul count in the period, taken
                    off that `foul` row (-1 when `trip_cause` is not `foul`)
  trip_andone_ctx   the row before the causing foul is a MADE field goal by the
                    shooting team at the same `secondsRemaining` -- the
                    and-one signature `possessions._handle_fga` uses

THE VERSION ARGUMENT
====================
`rim_override_max_ft` is the ONE thing the possessions version changes for
these two models: it decides whether ESPN's 2025 putback mistag (L16) is
repaired before a miss is labelled rim vs jumper. It changes a FEATURE of the
rebound model (miss type) and nothing else -- no rebound outcome, no free-throw
attempt, no possession boundary depends on it, because the override can only
ever move an `FGA_jump2` to an `FGA_rim` (`cbb_sim.pbp.events`, "THE
RIM-LOCATION OVERRIDE"). `rim_override_for_version` reads the threshold that
was actually used to build a given possessions version out of that build's own
`build_report.json`, so the flag is read from the data rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.pbp.events import PLAY_COLUMNS, classify_frame, load_plays
from cbb_sim.pbp.possessions import (
    BONUS_PRIOR_FOULS,
    DEFAULT_POSSESSION_VERSION,
    DOUBLE_BONUS_PRIOR_FOULS,
    _fix_flipped_sides,
    possessions_dir,
)

DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_PBP_DIR = Path("data/raw/cbbd/pbp")

#: `participant_1_id` is not in `PLAY_COLUMNS` (the possession machine does not
#: need a player id); the two models here do.
STREAM_COLUMNS: tuple[str, ...] = (*PLAY_COLUMNS, "participant_1_id")

#: Classes with no possession information AND no rebound information. `block`
#: is NOT in this set: it is dropped separately, after its information has been
#: moved onto the attempt as `blocked`.
INERT: frozenset[str] = frozenset({"timeout", "sub", "jumpball", "challenge"})

FT_CLASSES: tuple[str, ...] = ("FT_made", "FT_missed")
FGA_CLASSES: tuple[str, ...] = ("FGA_rim", "FGA_jump2", "FGA_3")

ON_FLOOR_COLS: tuple[str, ...] = tuple(
    [f"home_on_{k}" for k in range(1, 6)] + [f"away_on_{k}" for k in range(1, 6)]
)


def rim_override_for_version(version: str | None = None,
                             poss_dir: Path | str | None = None) -> float:
    """The rim-location override threshold (feet) that the named possessions
    build actually used.

    Read out of that build's `build_report.json` (`rim_override.max_ft`), which
    `scripts/build_possessions.py` writes on every run. v1 predates the
    override and its report carries no such key, so the fallback is 0.0 --
    "no override", which is exactly what `classify_frame` does with a
    non-positive threshold. Nothing here assumes which version has the
    override; it reads what the build recorded."""
    d = possessions_dir(version, poss_dir)
    rp = d / "build_report.json"
    if not rp.exists():
        return 0.0
    try:
        report = json.loads(rp.read_text())
    except json.JSONDecodeError:
        return 0.0
    return float(report.get("rim_override", {}).get("max_ft", 0.0) or 0.0)


def load_universe(path: Path | str = DEFAULT_UNIVERSE,
                  require_pbp_complete: bool = False) -> pd.DataFrame:
    """The D-I, non-truncated game universe, with dates parsed.

    `require_pbp_complete` additionally drops the games whose CBBD event stream
    does not account for the final score (data-defect row D3 of the change
    ledger). It is OFF by default so that the headline tables are built on the
    same universe L3 round 1 used and the two are comparable."""
    u = pd.read_parquet(path)
    u = u[u["is_d1_game"] & ~u["pbp_truncated"]].copy()
    if require_pbp_complete:
        if "pbp_complete" not in u.columns:
            raise KeyError(
                "games_universe.parquet has no `pbp_complete` column; rebuild it with "
                "scripts/build_game_universe.py")
        u = u[u["pbp_complete"]].copy()
    u["game_date"] = pd.to_datetime(u["game_date"])
    return u


def _bool_array(col: pd.Series, fallback: pd.Series | None = None) -> np.ndarray:
    s = col
    if s.dtype == object:
        s = s.map({True: True, False: False})
    s = s.astype("boolean")
    if fallback is not None:
        s = s.fillna(fallback.astype("boolean"))
    return s.fillna(False).to_numpy(dtype=bool)


def build_stream(
    season: int,
    universe: pd.DataFrame,
    rim_override_max_ft: float = 0.0,
    pbp_dir: Path | str = DEFAULT_PBP_DIR,
) -> pd.DataFrame:
    """The cleaned event stream for one season (module docstring)."""
    season = int(season)
    u = universe[universe["season"] == season]
    if not len(u):
        raise ValueError(f"no universe rows for season {season}")
    gids = set(u["cbbd_game_id"].dropna().astype("int64").tolist())
    plays = load_plays(season, pbp_dir=pbp_dir, game_ids=gids, columns=STREAM_COLUMNS)
    cls = classify_frame(plays, rim_override_max_ft=rim_override_max_ft)

    keep = ~cls.isin(list(INERT)).to_numpy()
    plays = plays.loc[keep].reset_index(drop=True)
    cls = cls.loc[keep].reset_index(drop=True)

    is_home = plays["isHomeTeam"]
    if is_home.dtype == object:
        is_home = is_home.map({True: True, False: False})
    is_home = is_home.astype("boolean")
    has_team = (pd.to_numeric(plays["teamId"], errors="coerce").notna() & is_home.notna()).to_numpy()
    side = np.where(is_home.fillna(False).to_numpy(), 0, 1)
    side = _fix_flipped_sides(plays, side, has_team)
    side = np.where(has_team, side, -1).astype("int16")

    df = pd.DataFrame({
        "cbbd_game_id": plays["gameId"].to_numpy(),
        "season": np.full(len(plays), season, dtype="int16"),
        "cls": cls.to_numpy(dtype=object),
        "side": side,
        "period": pd.to_numeric(plays["period"], errors="coerce").fillna(1).astype("int16").to_numpy(),
        "sec": pd.to_numeric(plays["secondsRemaining"], errors="coerce").fillna(0).astype("int32").to_numpy(),
        "home_score": pd.to_numeric(plays["homeScore"], errors="coerce").ffill().fillna(0).astype("int32").to_numpy(),
        "away_score": pd.to_numeric(plays["awayScore"], errors="coerce").ffill().fillna(0).astype("int32").to_numpy(),
        "made": _bool_array(plays["shot_made"], plays["scoringPlay"]),
        "player_id": pd.to_numeric(plays["participant_1_id"], errors="coerce").to_numpy(),
    })
    for c in ON_FLOOR_COLS:
        df[c] = pd.to_numeric(plays[c], errors="coerce").to_numpy()

    g = df["cbbd_game_id"].to_numpy()
    c = df["cls"].to_numpy(dtype=object)
    sec = df["sec"].to_numpy()
    same_prev = np.concatenate([[False], g[1:] == g[:-1]])
    same_next = np.concatenate([g[1:] == g[:-1], [False]])
    same_sec_prev = np.concatenate([[False], sec[1:] == sec[:-1]])
    same_sec_next = np.concatenate([sec[:-1] == sec[1:], [False]])

    # --- blocked flag, then drop the block rows -----------------------------
    blk = c == "block"
    prev_blk = np.concatenate([[False], blk[:-1]]) & same_prev & same_sec_prev
    next_blk = np.concatenate([blk[1:], [False]]) & same_next & same_sec_next
    df["blocked"] = prev_blk | next_blk
    df = df[~blk].reset_index(drop=True)

    # --- administrative rebounds inside a free-throw trip -------------------
    g = df["cbbd_game_id"].to_numpy()
    c = df["cls"].to_numpy(dtype=object)
    sd = df["side"].to_numpy()
    is_ft = np.isin(c, FT_CLASSES)
    same_prev = np.concatenate([[False], g[1:] == g[:-1]])
    same_next = np.concatenate([g[1:] == g[:-1], [False]])
    prev_ft = np.concatenate([[False], is_ft[:-1]]) & same_prev
    next_ft = np.concatenate([is_ft[1:], [False]]) & same_next
    prev_side = np.concatenate([[-9], sd[:-1]])
    next_side = np.concatenate([sd[1:], [-9]])
    admin = np.isin(c, ["OREB", "DeadBallReb"]) & prev_ft & next_ft & (prev_side == next_side)
    n_admin = int(admin.sum())
    df = df[~admin].reset_index(drop=True)

    df = _attach_team_fouls(df)
    df = _attach_trips(df)

    # --- universe keys ------------------------------------------------------
    ucols = ["game_id", "cbbd_game_id", "game_date", "neutral_site", "home_team_id", "away_team_id"]
    df = df.merge(u[ucols], on="cbbd_game_id", how="inner")
    home = df["home_team_id"].to_numpy()
    away = df["away_team_id"].to_numpy()
    sd = df["side"].to_numpy()
    df["team_id"] = np.where(sd == 0, home, np.where(sd == 1, away, -1)).astype("int64")
    df["opp_id"] = np.where(sd == 0, away, np.where(sd == 1, home, -1)).astype("int64")
    df.attrs["n_admin_rebounds_dropped"] = n_admin
    df.attrs["rim_override_max_ft"] = float(rim_override_max_ft)
    return df


def _attach_team_fouls(df: pd.DataFrame) -> pd.DataFrame:
    """Each team's PRIOR personal-foul count in the current period.

    Counted the same way `cbb_sim.pbp.possessions` counts it -- every
    `PersonalFoul` row increments the fouling team's period count, offensive
    fouls included -- because that is the counting rule the bonus thresholds
    were empirically validated against (`possessions.py`, "BONUS STATE, DERIVED
    BY COUNTING"). Technical fouls do NOT count toward the team total and are a
    separate class here, so they are excluded automatically."""
    c = df["cls"].to_numpy(dtype=object)
    sd = df["side"].to_numpy()
    is_foul = c == "foul"
    key = pd.DataFrame({"g": df["cbbd_game_id"].to_numpy(), "p": df["period"].to_numpy()})
    for s, name in ((0, "fouls_home"), (1, "fouls_away")):
        f = (is_foul & (sd == s)).astype("int32")
        k = key.assign(f=f)
        cum = k.groupby(["g", "p"], sort=False)["f"].cumsum().to_numpy()
        df[name] = (cum - f).astype("int16")   # strictly-before count
    own = np.where(sd == 0, df["fouls_home"], np.where(sd == 1, df["fouls_away"], -1))
    opp = np.where(sd == 0, df["fouls_away"], np.where(sd == 1, df["fouls_home"], -1))
    df["fouls_own_prior"] = own.astype("int16")
    df["fouls_opp_prior"] = opp.astype("int16")
    return df


def _attach_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Free-throw trip structure (module docstring)."""
    g = df["cbbd_game_id"].to_numpy()
    c = df["cls"].to_numpy(dtype=object)
    sd = df["side"].to_numpy()
    sec = df["sec"].to_numpy()
    made = df["made"].to_numpy()
    n = len(df)

    is_ft = np.isin(c, FT_CLASSES)
    same_prev = np.concatenate([[False], g[1:] == g[:-1]])
    same_next = np.concatenate([g[1:] == g[:-1], [False]])
    prev_ft = np.concatenate([[False], is_ft[:-1]]) & same_prev
    next_ft = np.concatenate([is_ft[1:], [False]]) & same_next
    prev_side = np.concatenate([[-9], sd[:-1]])
    next_side = np.concatenate([sd[1:], [-9]])

    trip_start = is_ft & ~(prev_ft & (prev_side == sd))
    trip_last = is_ft & ~(next_ft & (next_side == sd))
    trip_no = np.where(is_ft, np.cumsum(trip_start) - 1, -1).astype("int64")

    # cause of the trip: the row immediately before its first attempt
    prev_cls = np.concatenate([[None], c[:-1]])
    cause_all = np.where(
        ~same_prev, "none",
        np.where(prev_cls == "foul", "foul",
                 np.where(prev_cls == "technical", "technical", "none")))
    prev_own_prior = np.concatenate([[-1], df["fouls_own_prior"].to_numpy()[:-1]])
    prior_all = np.where(cause_all == "foul", prev_own_prior, -1).astype("int16")

    # and-one context: two rows back is a MADE field goal by the shooting team
    # at the same clock (the signature `possessions._handle_fga` looks for).
    pad2 = 2 if n >= 2 else n
    pp_cls = np.concatenate([np.array([None] * pad2, dtype=object), c[:n - pad2]])
    pp_made = np.concatenate([np.zeros(pad2, dtype=bool), made[:n - pad2]])
    pp_side = np.concatenate([np.full(pad2, -9), sd[:n - pad2]])
    pp_sec = np.concatenate([np.full(pad2, -1), sec[:n - pad2]])
    pp_game = np.concatenate([np.full(pad2, -1, dtype=g.dtype), g[:n - pad2]])
    andone_all = (np.isin(pp_cls, FGA_CLASSES) & pp_made & (pp_side == sd)
                  & (pp_sec == sec) & (pp_game == g))

    starts = np.flatnonzero(trip_start)
    n_trips = len(starts)
    lens = np.bincount(trip_no[is_ft], minlength=max(n_trips, 1))[:max(n_trips, 1)]
    first_made = made[starts]
    cause_by_trip = cause_all[starts]
    prior_by_trip = prior_all[starts]
    andone_by_trip = andone_all[starts]

    idx = np.where(is_ft, trip_no, 0)
    zero_i16 = np.zeros(n, dtype="int16")
    df["trip_id"] = np.where(is_ft, trip_no, -1)
    if n_trips:
        df["trip_len"] = np.where(is_ft, lens[idx], 0).astype("int16")
        df["trip_first_made"] = np.where(is_ft, first_made[idx], False)
        df["trip_cause"] = np.where(is_ft, cause_by_trip[idx], "")
        df["trip_prior_fouls"] = np.where(is_ft, prior_by_trip[idx], -1).astype("int16")
        df["trip_andone_ctx"] = np.where(is_ft, andone_by_trip[idx], False)
        ft_idx = np.flatnonzero(is_ft)
        pos = zero_i16.copy()
        pos[ft_idx] = (ft_idx - starts[trip_no[ft_idx]] + 1).astype("int16")
        df["trip_pos"] = pos
    else:
        df["trip_len"] = zero_i16
        df["trip_first_made"] = np.zeros(n, dtype=bool)
        df["trip_cause"] = np.array([""] * n, dtype=object)
        df["trip_prior_fouls"] = zero_i16 - 1
        df["trip_andone_ctx"] = np.zeros(n, dtype=bool)
        df["trip_pos"] = zero_i16
    df["trip_last"] = trip_last
    df["trip_is_technical"] = df["trip_cause"].to_numpy() == "technical"
    return df


def in_bonus(prior_fouls) -> np.ndarray:
    """NCAA one-and-one state from the fouling team's prior period foul count.
    The thresholds are `cbb_sim.pbp.possessions`' constants, which that module
    derived from the trip-length distribution rather than assuming."""
    return np.asarray(prior_fouls) >= BONUS_PRIOR_FOULS


def in_double_bonus(prior_fouls) -> np.ndarray:
    return np.asarray(prior_fouls) >= DOUBLE_BONUS_PRIOR_FOULS


__all__ = [
    "BONUS_PRIOR_FOULS",
    "DEFAULT_POSSESSION_VERSION",
    "DOUBLE_BONUS_PRIOR_FOULS",
    "FGA_CLASSES",
    "FT_CLASSES",
    "ON_FLOOR_COLS",
    "build_stream",
    "in_bonus",
    "in_double_bonus",
    "load_universe",
    "rim_override_for_version",
]
