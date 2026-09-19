"""
free_throw.py -- L3 FREE THROW sub-model, in two parts.

Pre-registration (verbatim, PM, 2026-09-10):
`docs/models/free_throw/experiments.md` section 1. Trainer:
`scripts/train_free_throw_v1.py`. Feature provenance:
`docs/models/free_throw/features.md`.

===========================================================================
FT-1: TRIP STRUCTURE
===========================================================================
How many attempts a foul produces is a RULE, not a fit. The rule table is
`TRIP_RULES` below, keyed on a foul class that is derived from CONTEXT only --
never from the attempt count itself, because deriving the class from the count
and then checking the count against the class would verify nothing.

The context signals, in order:

  technical          the row before the trip's first attempt is a Technical
                     Foul. Technical free throws are shot during a dead ball,
                     do not count toward the team foul total, and never change
                     possession.
  and_one            the row before the causing foul is a MADE field goal by
                     the shooting team at the same `secondsRemaining`. This is
                     the same signature `possessions._handle_fga` uses, and it
                     was validated there: of 1,982 fouls both preceded by a
                     shot at the same clock and followed by free throws, 1,942
                     follow a MADE shot.
  shooting           the fouling team's PRIOR foul count in the period is below
                     the bonus threshold. With no bonus in force a non-shooting
                     foul produces no free throws at all, so a trip that exists
                     must be a shooting foul.
  bonus_one_and_one  prior count in the one-and-one window.
  double_bonus       prior count at or above the double-bonus threshold.

The last two are AMBIGUOUS by construction: once the bonus is in force a
genuine two-shot shooting foul and a bonus trip both produce two attempts and
the feed does not say which happened (this is `ft_trip_ambiguous` on the
possession tables, data-defect row D5 of `docs/models/change_ledger.md`). The
rule check therefore tests only the parts of the rule that ARE identified, and
`verify_trip_rules` reports the ambiguous mass separately instead of scoring it.

THE ERA FLAG. `BONUS_THRESHOLDS_DEFAULT` are the NCAA men's thresholds
`cbb_sim.pbp.possessions` derived from the trip-length distribution. This
module re-derives them PER SEASON from the data (`derive_bonus_thresholds`) so
that a rule change shows up as a moved threshold rather than as a silent
mis-labelling, and writes the per-season table to
`data/processed/models/free_throw/bonus_era.json`. Per the pre-registration the
era flag belongs in GameState, not in a fitted model: the engine reads
`load_bonus_era()` into GameState at game setup and the make-probability model
below never sees a season-specific rule constant.

===========================================================================
FT-2: MAKE PROBABILITY PER ATTEMPT
===========================================================================
One row per free-throw attempt. Target: made (1) or missed (0). L15 measured
that player identity is the dominant term in every player-game rate stat, so
the arms are ordered by how much shooter identity they are allowed to use:

  (a) team_asof   the shooting TEAM's as-of free-throw percentage. The
                  team-level floor: no shooter identity at all.
  (b) eb_shrink   empirical-Bayes shrinkage of the shooter's own as-of rate
                  toward a prior, with BOTH the prior (league mean / position
                  mean / the player's prior-season rate) and the shrinkage
                  strength fitted on the training fold.
  (c) ridge       logistic ridge on the shooter's as-of rate, attempts to date,
                  prior-season rate, season index and the late-game state.
  (d) lgbm        LightGBM on the same features.

LEAK SAFETY. Every as-of rate -- shooter, team, league, position -- is an
expanding mean over GAMES STRICTLY BEFORE the current one within the season
(`prob_metrics.expanding_asof`), so an attempt can never see its own game, let
alone its own trip. `prior_season_ft` is a completed previous season and is
therefore entirely in the past by construction. `tests/test_free_throw.py`
proves both two ways: an independent recomputation of the strictly-earlier
average, and invariance of every feature to corrupting the attempt's own game.

PLAYER IDENTITY. The key is the CBBD player id. The pre-registration prefers
the ESPN athlete id "via the crosswalk if present, else CBBD ids", and the
crosswalk (`cbb_sim.data.player_ids`) only exists for 2024-2026 because CBBD
rosters were pulled only for those seasons -- it cannot span the 2022-2025
window this model is fitted on. The CBBD id can: on the 9,421 players the
crosswalk covers in both 2024 and 2025, and the 3,730 covered in both 2025 and
2026, the CBBD id maps to the SAME ESPN athlete id 100.0% of the time, so it is
a stable cross-season identity and not a per-season surrogate. The ESPN id is
still attached wherever the crosswalk resolves it (that is what the L4 player
layer will join on) and the coverage is a reported number.

===========================================================================
FOLDS AND THE SEAL
===========================================================================
F1 trains {2022, 2023} and tests 2024; F2 trains {2022, 2023, 2024} and tests
2025 and is the selection fold. 2026 is sealed; `fold_slices` calls
`assert_not_sealed` on both slices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.data.seal import assert_not_sealed
from cbb_sim.models import event_stream as ES
from cbb_sim.models import prob_metrics as PM
from cbb_sim.pbp.possessions import BONUS_PRIOR_FOULS, DOUBLE_BONUS_PRIOR_FOULS

DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_ARTIFACT_DIR = Path("data/processed/models/free_throw")
DEFAULT_ROSTER_DIR = Path("data/raw/cbbd/rosters")
DEFAULT_CROSSWALK = Path("data/processed/player_crosswalk.parquet")

CLASSES: tuple[str, ...] = ("MISS", "MAKE")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

FOLDS: dict[str, dict[str, list[int]]] = {
    "F1": {"train": [2022, 2023], "test": [2024]},
    "F2": {"train": [2022, 2023, 2024], "test": [2025]},
}
SELECTION_FOLD = "F2"

BONUS_THRESHOLDS_DEFAULT = (BONUS_PRIOR_FOULS, DOUBLE_BONUS_PRIOR_FOULS)


# ===========================================================================
# FT-1: the rule table
# ===========================================================================
FOUL_CLASSES: tuple[str, ...] = (
    "shooting", "and_one", "bonus_one_and_one", "double_bonus", "technical", "unknown",
)

#: The rule, as attempts. `expected` is the set of attempt counts the rule
#: allows; `identified` says whether context alone pins the class down, which
#: is what decides if a mismatch is a VIOLATION or merely unidentified.
TRIP_RULES: dict[str, dict] = {
    "and_one": {
        "expected": (1,), "identified": True,
        "why": "one bonus free throw after a made field goal",
    },
    "shooting": {
        "expected": (2, 3), "identified": True,
        "why": ("a foul below the bonus threshold produces free throws only if it was a "
                "shooting foul: 2 on a two-point attempt, 3 on a three. Which of the two is "
                "NOT observable -- ESPN does not log the field-goal attempt when a shooting "
                "foul occurs on a miss -- so the identified prediction is 'at least 2'"),
    },
    "bonus_one_and_one": {
        "expected": (1, 2), "identified": False,
        "why": ("1 if the front end misses, 2 if it makes -- but a two-shot SHOOTING foul in "
                "the same window is indistinguishable in the feed, so only the sharp half of "
                "the rule is identified: a ONE-attempt trip here must have MISSED"),
    },
    "double_bonus": {
        "expected": (2, 3), "identified": False,
        "why": ("two shots with no front end; 3 if the foul was on a three-point attempt. "
                "A one-attempt trip is a violation, a two-attempt trip is ambiguous with a "
                "shooting foul"),
    },
    "technical": {
        "expected": (1, 2), "identified": False,
        "why": ("1 or 2 by infraction (administrative vs contact / flagrant), which the feed "
                "does not name; reported, not gated"),
    },
    "unknown": {
        "expected": (), "identified": False,
        "why": "no causing foul row in the feed (a feed gap); reported, not gated",
    },
}


def classify_foul(trip_cause: np.ndarray, andone_ctx: np.ndarray, prior_fouls: np.ndarray,
                  bonus_prior: int = BONUS_PRIOR_FOULS,
                  double_prior: int = DOUBLE_BONUS_PRIOR_FOULS) -> np.ndarray:
    """The foul class of each trip, from CONTEXT ONLY (module docstring).

    `bonus_prior` / `double_prior` are arguments and not constants so that the
    per-season thresholds `derive_bonus_thresholds` finds in the data can be
    used instead of the defaults -- which is what makes an era check possible
    rather than circular."""
    cause = np.asarray(trip_cause, dtype=object)
    ao = np.asarray(andone_ctx, dtype=bool)
    pf = np.asarray(prior_fouls)
    out = np.full(len(cause), "unknown", dtype=object)
    is_foul = cause == "foul"
    out[cause == "technical"] = "technical"
    out[is_foul & ao] = "and_one"
    rest = is_foul & ~ao
    out[rest & (pf < bonus_prior)] = "shooting"
    out[rest & (pf >= bonus_prior) & (pf < double_prior)] = "bonus_one_and_one"
    out[rest & (pf >= double_prior)] = "double_bonus"
    return out


def derive_bonus_thresholds(trips: pd.DataFrame, hi: float = 0.55,
                            max_prior: int = 15) -> dict:
    """Re-derive the two bonus thresholds FROM THE DATA for one season.

    The signature is the one `cbb_sim.pbp.possessions` validated: a one-attempt
    trip inside the one-and-one window is almost always a MISSED front end (a
    made front end earns a second shot), whereas a one-attempt trip outside the
    window is almost always an and-one, which is made about two thirds of the
    time. So `r(p)` -- the share of one-attempt trips at prior foul count `p`
    whose only attempt MISSED -- steps from about 0.32 to about 0.72 when the
    one-and-one starts and steps back down when the double bonus removes the
    front end. The onset is the first `p` where `r(p)` crosses `hi`; the
    double-bonus onset is the first `p` after that where it falls back below.

    Returns the two thresholds, the `r(p)` series they were read off, and the
    one-attempt trip counts, so the claim is auditable rather than typed in."""
    t = trips[(trips["trip_cause"] == "foul") & (trips["trip_len"] == 1)]
    r, n = {}, {}
    for p in range(0, max_prior + 1):
        s = t[t["trip_prior_fouls"] == p]
        n[p] = int(len(s))
        r[p] = float(1.0 - s["trip_first_made"].mean()) if len(s) else float("nan")
    onset = None
    for p in range(0, max_prior + 1):
        if n[p] >= 100 and r[p] > hi:
            onset = p
            break
    double = None
    if onset is not None:
        for p in range(onset + 1, max_prior + 1):
            if n[p] >= 50 and r[p] <= hi:
                double = p
                break
    return {
        "bonus_prior_fouls": onset,
        "double_bonus_prior_fouls": double,
        "one_attempt_miss_share_by_prior": {str(p): (round(r[p], 4) if n[p] else None)
                                            for p in range(0, max_prior + 1)},
        "one_attempt_trips_by_prior": {str(p): n[p] for p in range(0, max_prior + 1)},
        "threshold": hi,
    }


def verify_trip_rules(trips: pd.DataFrame) -> dict:
    """Rule-derived attempt counts vs the data, per the pre-registration.

    Every trip is scored against `TRIP_RULES` for the class its CONTEXT gives
    it. A mismatch on an `identified` class is a VIOLATION; a mismatch on an
    unidentified class is reported as ambiguous mass, never as a violation."""
    out: dict = {"n_trips": int(len(trips)), "by_class": {}, "violations": {}}
    fc = trips["foul_class"].to_numpy()
    n_ft = trips["trip_len"].to_numpy()
    first_made = trips["trip_first_made"].to_numpy()
    for cls in FOUL_CLASSES:
        m = fc == cls
        if m.sum() == 0:
            continue
        counts = pd.Series(n_ft[m]).value_counts().sort_index()
        out["by_class"][cls] = {
            "n": int(m.sum()),
            "share_pct": round(float(m.mean() * 100), 3),
            "attempts_hist": {str(int(k)): int(v) for k, v in counts.items()},
            "expected": list(TRIP_RULES[cls]["expected"]),
            "identified": TRIP_RULES[cls]["identified"],
        }
    v = {}
    m = fc == "shooting"
    v["shooting_with_one_attempt"] = int((m & (n_ft == 1)).sum())
    m = fc == "and_one"
    v["and_one_with_more_than_one_attempt"] = int((m & (n_ft != 1)).sum())
    m = fc == "bonus_one_and_one"
    v["one_and_one_single_attempt_that_was_MADE"] = int((m & (n_ft == 1) & first_made).sum())
    m = fc == "double_bonus"
    v["double_bonus_with_one_attempt"] = int((m & (n_ft == 1)).sum())
    v["any_class_with_four_or_more_attempts"] = int((n_ft >= 4).sum())
    out["violations"] = v
    out["violation_rate_pct"] = round(
        float(sum(v.values()) / max(len(trips), 1) * 100), 4)
    m = fc == "bonus_one_and_one"
    out["ambiguous"] = {
        "one_and_one_two_attempts": int((m & (n_ft == 2)).sum()),
        "double_bonus_two_attempts": int(((fc == "double_bonus") & (n_ft == 2)).sum()),
        "note": ("a two-attempt trip with the bonus in force is consistent with BOTH a bonus "
                 "trip and a two-shot shooting foul; the feed cannot separate them"),
    }
    return out


def load_bonus_era(path: Path | str | None = None) -> dict[int, tuple[int, int]]:
    """Per-season (bonus_prior_fouls, double_bonus_prior_fouls) as derived from
    the data by `scripts/train_free_throw_v1.py`.

    THE ENGINE READS THIS INTO GameState. Per the pre-registration and
    `CLAUDE.md` ("rule-era flags live in GameState, not baked into
    sub-models"), no fitted object in this module carries a season-specific
    rule constant; the state carries the era and the model reads the state."""
    p = Path(path) if path is not None else DEFAULT_ARTIFACT_DIR / "bonus_era.json"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}; run scripts/train_free_throw_v1.py to derive it from the data")
    raw = json.loads(p.read_text())
    return {int(k): (int(v["bonus_prior_fouls"]), int(v["double_bonus_prior_fouls"]))
            for k, v in raw["by_season"].items()}


# ===========================================================================
# FT-2: the attempt table
# ===========================================================================
def build_trips_and_attempts(
    seasons: list[int],
    universe: pd.DataFrame | None = None,
    version: str | None = None,
    poss_dir: Path | str | None = None,
    pbp_dir: Path | str = ES.DEFAULT_PBP_DIR,
    universe_path: Path | str = DEFAULT_UNIVERSE,
    require_pbp_complete: bool = False,
    tech_lookahead: bool = ES.DEFAULT_TECH_LOOKAHEAD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(trips, attempts) for `seasons`.

    `version` is accepted for symmetry with the rebound model and with the
    PM's one-flag re-run; it selects the possessions build whose rim-location
    override is applied when the stream is classified. NOTHING in either table
    depends on it -- the override can only move an `FGA_jump2` to an `FGA_rim`,
    and neither a free-throw attempt nor a trip's structure reads a field-goal
    class. `scripts/train_free_throw_v1.py` asserts that the two versions
    produce identical tables rather than asserting it here in prose."""
    if universe is None:
        universe = ES.load_universe(universe_path, require_pbp_complete=require_pbp_complete)
    max_ft = ES.rim_override_for_version(version, poss_dir)
    trips, attempts = [], []
    for s in seasons:
        t, a = _season_trips(int(s), universe, max_ft, pbp_dir,
                             tech_lookahead=tech_lookahead)
        trips.append(t)
        attempts.append(a)
    return (pd.concat(trips, ignore_index=True),
            pd.concat(attempts, ignore_index=True))


def _season_trips(season: int, universe: pd.DataFrame, max_ft: float,
                  pbp_dir: Path | str,
                  tech_lookahead: bool = ES.DEFAULT_TECH_LOOKAHEAD
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = ES.build_stream(season, universe, rim_override_max_ft=max_ft, pbp_dir=pbp_dir,
                         tech_lookahead=tech_lookahead)
    ft = st[st["cls"].isin(ES.FT_CLASSES)].copy()
    off_home = (ft["side"] == 0).to_numpy()
    hs, as_ = ft["home_score"].to_numpy(), ft["away_score"].to_numpy()
    att = pd.DataFrame({
        "season": ft["season"].to_numpy(),
        "game_id": ft["game_id"].to_numpy(),
        "cbbd_game_id": ft["cbbd_game_id"].to_numpy(),
        "game_date": ft["game_date"].to_numpy(),
        "neutral_site": ft["neutral_site"].to_numpy(),
        "shooter_is_home": off_home,
        "team_id": ft["team_id"].to_numpy(),
        "opp_id": ft["opp_id"].to_numpy(),
        "shooter_id": ft["player_id"].to_numpy(),
        "period": ft["period"].to_numpy(),
        "seconds_remaining": ft["sec"].to_numpy(),
        "score_diff": np.where(off_home, hs - as_, as_ - hs).astype("int32"),
        "made": (ft["cls"].to_numpy() == "FT_made"),
        "trip_id": ft["trip_id"].to_numpy(),
        "trip_pos": ft["trip_pos"].to_numpy(),
        "trip_len": ft["trip_len"].to_numpy(),
        "trip_cause": ft["trip_cause"].to_numpy(),
        "trip_prior_fouls": ft["trip_prior_fouls"].to_numpy(),
        "trip_andone_ctx": ft["trip_andone_ctx"].to_numpy(),
        "trip_first_made": ft["trip_first_made"].to_numpy(),
    })
    att["foul_class"] = classify_foul(att["trip_cause"], att["trip_andone_ctx"],
                                      att["trip_prior_fouls"])
    trips = att[att["trip_pos"] == 1].copy()
    trips["trip_made"] = att.groupby("trip_id")["made"].sum().reindex(trips["trip_id"]).to_numpy()
    return trips, att


# ---------------------------------------------------------------------------
# Player identity, position, and the as-of features
# ---------------------------------------------------------------------------
POSITION_GROUPS: dict[str, str] = {
    "guard": "G", "point guard": "G", "shooting guard": "G", "guard/forward": "G",
    "forward": "F", "small forward": "F", "power forward": "F",
    "center": "C",
}


def load_positions(roster_dir: Path | str = DEFAULT_ROSTER_DIR) -> pd.DataFrame:
    """(cbbd_player_id -> position group) from the CBBD rosters on disk.

    Rosters exist for 2024-2026 only, and the CBBD player id is stable across
    seasons (module docstring), so a player who appears on any of those rosters
    carries their position back to 2022. Everyone else is `UNK` and falls back
    to the league prior; `scripts/train_free_throw_v1.py` reports the coverage
    per season rather than letting an unmeasured fallback pass as a position."""
    frames = []
    for p in sorted(Path(roster_dir).glob("roster_*.parquet")):
        frames.append(pd.read_parquet(p, columns=["season", "cbbd_player_id", "position"]))
    if not frames:
        return pd.DataFrame({"shooter_id": pd.Series(dtype="int64"),
                             "position_group": pd.Series(dtype="object")})
    r = pd.concat(frames, ignore_index=True).sort_values("season")
    r["position_group"] = (r["position"].astype("string").str.lower()
                           .map(POSITION_GROUPS).fillna("UNK"))
    r = r.drop_duplicates(subset=["cbbd_player_id"], keep="first")
    return r.rename(columns={"cbbd_player_id": "shooter_id"})[["shooter_id", "position_group"]]


def load_espn_ids(path: Path | str = DEFAULT_CROSSWALK) -> pd.DataFrame:
    """(season, cbbd player id) -> ESPN athlete id, where the crosswalk has it.

    Attached to the attempt table for the L4 handoff and for the reported match
    rate; it is NOT the model's key, because the crosswalk covers 2024-2026
    only (module docstring, PLAYER IDENTITY)."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["season", "shooter_id", "espn_athlete_id"])
    cw = pd.read_parquet(p, columns=["season", "cbbd_player_id", "espn_athlete_id"])
    return cw.rename(columns={"cbbd_player_id": "shooter_id"})


FT_FEATURES: tuple[str, ...] = (
    "shooter_ft_asof",      # the shooter's as-of FT%, league-centred
    "shooter_fta_asof",     # attempts to date this season
    "prior_season_ft",      # the shooter's completed prior-season FT%, league-centred
    "has_prior_season",
    "season_idx",
    "seconds_remaining", "period", "score_diff", "in_bonus",
)
#: `period` is in the late-game block on purpose: `seconds_remaining` is
#: PER PERIOD in the feed (1200 in halves, 300 in overtime), so without the
#: period index it does not identify late-game at all. This is the one addition
#: to the pre-registered feature list and it is recorded as such in features.md.

TEAM_FEATURE = "team_ft_asof"


def build_ft_design(
    attempts: pd.DataFrame,
    roster_dir: Path | str = DEFAULT_ROSTER_DIR,
    crosswalk_path: Path | str = DEFAULT_CROSSWALK,
    include_technical: bool = False,
) -> pd.DataFrame:
    """The attempt-level design matrix with every as-of feature attached.

    Technical free throws are EXCLUDED by default: the shooter is chosen by the
    coach rather than by who was fouled, so the attempt is drawn from a
    different shooter distribution than every other free throw and pooling the
    two would bias both. Their count and make rate are reported by the trainer;
    the engine needs a separate rule for them and `model.md` section 9 says so.
    """
    a = attempts.copy()
    a["game_date"] = pd.to_datetime(a["game_date"])
    a["is_technical"] = a["foul_class"].to_numpy() == "technical"
    if not include_technical:
        a = a[~a["is_technical"]]
    a = a[np.isfinite(a["shooter_id"].to_numpy())]
    a["shooter_id"] = a["shooter_id"].astype("int64")
    a["y"] = a["made"].astype("int8")

    # --- shooter as-of, by GAME (never within the shooter's own game) ------
    # `pg_raw` keeps the PER-GAME counts. Every other as-of table below is
    # built from it and not from `pg`, whose `fta`/`ftm` are already cumulative
    # prior sums: expanding a column that is itself an expanding sum would
    # square the history and is exactly the bug this split exists to prevent.
    pg_raw = a.groupby(["season", "shooter_id", "game_id"], as_index=False).agg(
        fta=("y", "size"), ftm=("y", "sum"), game_date=("game_date", "first"))
    pg_raw = pg_raw.sort_values(["season", "shooter_id", "game_date", "game_id"],
                                kind="stable").reset_index(drop=True)
    asof = PM.expanding_asof(pg_raw, ["season", "shooter_id"], ["fta", "ftm"])
    pg = pd.concat([pg_raw[["season", "shooter_id", "game_id", "game_date"]], asof], axis=1)

    tg = a.groupby(["season", "team_id", "game_id"], as_index=False).agg(
        fta=("y", "size"), ftm=("y", "sum"), game_date=("game_date", "first"))
    tg = tg.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    tasof = PM.expanding_asof(tg, ["season", "team_id"], ["fta", "ftm"])
    tg = pd.concat([tg[["season", "team_id", "game_id"]], tasof], axis=1)
    tg = tg.rename(columns={"fta": "team_fta_prior", "ftm": "team_ftm_prior",
                            "n_prior": "team_games_prior"})

    day = pg_raw.groupby(["season", "game_date"], as_index=False)[["fta", "ftm"]].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PM.expanding_asof(day, ["season"], ["fta", "ftm"])
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)
    # The season's FIRST date has nothing strictly earlier, so its as-of league
    # rate is undefined. It is back-filled from the season's own next available
    # date -- computed HERE, on the date-sorted table, because a back-fill on
    # the attempt table would run along attempt order, not calendar order, and
    # would silently mean nothing. This is the only forward-looking value in
    # the module and it touches opening day alone; the alternative (a typed-in
    # constant) would be worse, and the 0.70 floor is only reached when a
    # season has no rate at all.
    with np.errstate(divide="ignore", invalid="ignore"):
        lg_day = np.where(day["lg_fta"].to_numpy() > 0,
                          day["lg_ftm"].to_numpy() / np.maximum(day["lg_fta"].to_numpy(), 1e-9),
                          np.nan)
    day["lg_ft_asof"] = lg_day
    day["lg_ft_asof"] = day.groupby("season")["lg_ft_asof"].bfill().fillna(0.70)
    pg = pg.merge(day, on=["season", "game_date"], how="left")

    pos = load_positions(roster_dir)
    pg_raw = pg_raw.merge(pos, on="shooter_id", how="left")
    pg_raw["position_group"] = pg_raw["position_group"].fillna("UNK")
    pg = pg.merge(pg_raw[["season", "shooter_id", "game_id", "position_group"]],
                  on=["season", "shooter_id", "game_id"], how="left")

    # position as-of rate: expanding over the position's own earlier games,
    # from the PER-GAME counts (see the `pg_raw` note above)
    posday = pg_raw.groupby(["season", "position_group", "game_date"], as_index=False)[["fta", "ftm"]].sum()
    posday = posday.sort_values(["season", "position_group", "game_date"], kind="stable")
    pasof = PM.expanding_asof(posday, ["season", "position_group"], ["fta", "ftm"])
    pasof.columns = [f"pos_{c}" for c in pasof.columns]
    posday = pd.concat([posday[["season", "position_group", "game_date"]], pasof], axis=1)
    pg = pg.merge(posday, on=["season", "position_group", "game_date"], how="left")

    # prior-season rate: the shooter's completed previous season
    season_tot = a.groupby(["season", "shooter_id"], as_index=False).agg(
        fta=("y", "size"), ftm=("y", "sum"))
    season_tot["season"] = season_tot["season"] + 1
    season_tot = season_tot.rename(columns={"fta": "prev_fta", "ftm": "prev_ftm"})
    pg = pg.merge(season_tot, on=["season", "shooter_id"], how="left")

    prev_team = a.groupby(["season", "shooter_id"])["team_id"].agg(
        lambda s: s.value_counts().index[0]).reset_index().rename(columns={"team_id": "prev_team_id"})
    prev_team["season"] = prev_team["season"] + 1
    pg = pg.merge(prev_team, on=["season", "shooter_id"], how="left")

    keep = ["season", "shooter_id", "game_id", "fta", "ftm", "n_prior",
            "lg_ft_asof", "pos_fta", "pos_ftm", "prev_fta", "prev_ftm",
            "prev_team_id", "position_group"]
    d = a.merge(pg[keep], on=["season", "shooter_id", "game_id"], how="left")
    d = d.merge(tg, on=["season", "team_id", "game_id"], how="left")

    def _rate(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

    d["lg_ft_asof"] = d["lg_ft_asof"].fillna(0.70).to_numpy().astype("float32")

    own = _rate(d["ftm"].to_numpy(), d["fta"].to_numpy())
    d["shooter_fta_prior"] = d["fta"].fillna(0).to_numpy().astype("float32")
    d["shooter_ftm_prior"] = d["ftm"].fillna(0).to_numpy().astype("float32")
    d["shooter_ft_raw"] = np.where(np.isfinite(own), own, d["lg_ft_asof"]).astype("float32")
    d["shooter_ft_asof"] = (d["shooter_ft_raw"] - d["lg_ft_asof"]).astype("float32")
    d["shooter_fta_asof"] = d["shooter_fta_prior"]

    pos_rate = _rate(d["pos_ftm"].to_numpy(), d["pos_fta"].to_numpy())
    d["position_ft_asof"] = np.where(np.isfinite(pos_rate), pos_rate,
                                     d["lg_ft_asof"]).astype("float32")

    prev = _rate(d["prev_ftm"].to_numpy(), d["prev_fta"].to_numpy())
    d["has_prior_season"] = np.isfinite(prev).astype("float32")
    d["prior_season_ft_raw"] = np.where(np.isfinite(prev), prev, d["lg_ft_asof"]).astype("float32")
    d["prior_season_ft"] = (d["prior_season_ft_raw"] - d["lg_ft_asof"]).astype("float32")
    d["prior_season_fta"] = d["prev_fta"].fillna(0).to_numpy().astype("float32")

    team_rate = _rate(d["team_ftm_prior"].to_numpy(), d["team_fta_prior"].to_numpy())
    d["team_ft_raw"] = np.where(np.isfinite(team_rate), team_rate, d["lg_ft_asof"]).astype("float32")
    d[TEAM_FEATURE] = (d["team_ft_raw"] - d["lg_ft_asof"]).astype("float32")

    d["season_idx"] = (d["season"] - 2022).astype("float32")
    d["seconds_remaining"] = d["seconds_remaining"].astype("float32")
    d["period"] = d["period"].astype("float32")
    d["score_diff"] = d["score_diff"].astype("float32")
    d["in_bonus"] = np.isin(d["foul_class"].to_numpy(),
                            ["bonus_one_and_one", "double_bonus"]).astype("float32")

    # transfer flag: the player's modal team changed since the prior season
    prev_team_id = d["prev_team_id"].to_numpy()
    d["is_transfer"] = (np.isfinite(prev_team_id) & (prev_team_id != d["team_id"].to_numpy())
                        & (d["has_prior_season"].to_numpy() > 0))

    cw = load_espn_ids(crosswalk_path)
    if len(cw):
        d = d.merge(cw, on=["season", "shooter_id"], how="left")
    else:
        d["espn_athlete_id"] = np.nan
    return d


def fold_slices(design: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = FOLDS[fold]
    assert_not_sealed(spec["train"], context=f"{fold} train seasons")
    assert_not_sealed(spec["test"], context=f"{fold} test seasons")
    tr = design[design["season"].isin(spec["train"])]
    te = design[design["season"].isin(spec["test"])]
    assert_not_sealed(tr, context=f"{fold} train slice")
    assert_not_sealed(te, context=f"{fold} test slice")
    return tr, te


# ===========================================================================
# Arms
# ===========================================================================
ARMS: tuple[str, ...] = ("team_asof", "eb_shrink", "ridge", "lgbm")

#: The shrinkage grid, in pseudo-attempts. Fitted on the training fold.
SHRINK_GRID: tuple[float, ...] = (5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300)
#: The three priors the pre-registration names.
PRIOR_KINDS: tuple[str, ...] = ("league", "position", "prior_season")


def _prior_rate(d: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "league":
        return d["lg_ft_asof"].to_numpy(dtype="float64")
    if kind == "position":
        return d["position_ft_asof"].to_numpy(dtype="float64")
    if kind == "prior_season":
        # returning players use their own completed prior season; newcomers
        # have none and fall back to the league as-of mean, which is exactly
        # what "for returning players" means and is not a hidden imputation.
        return d["prior_season_ft_raw"].to_numpy(dtype="float64")
    raise KeyError(f"unknown prior kind {kind!r}")


def eb_predict(d: pd.DataFrame, kind: str, m: float) -> np.ndarray:
    """Empirical-Bayes shrunk make probability, as a (n, 2) matrix."""
    prior = _prior_rate(d, kind)
    made = d["shooter_ftm_prior"].to_numpy(dtype="float64")
    att = d["shooter_fta_prior"].to_numpy(dtype="float64")
    p = (m * prior + made) / (m + att)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.column_stack([1.0 - p, p])


def fit_eb(tr: pd.DataFrame, grid: tuple[float, ...] = SHRINK_GRID,
           kinds: tuple[str, ...] = PRIOR_KINDS) -> dict:
    """Grid over (prior, shrinkage strength) on the TRAINING fold.

    The selection metric is the training fold's own log loss, which is honest
    here because every quantity the arm uses is an as-of feature: a shooter's
    shrunk rate on 12 January is built from games strictly before 12 January,
    so the training log loss is already a walk-forward number, not an in-sample
    one."""
    y = tr["y"].to_numpy()
    rows = []
    best = None
    for kind in kinds:
        for m in grid:
            p = eb_predict(tr, kind, float(m))
            ll = PM.log_loss(y, p)
            rows.append({"prior": kind, "m": float(m), "train_log_loss": round(ll, 6)})
            if best is None or ll < best["train_log_loss"]:
                best = rows[-1]
    return {"grid": rows, "best": best}


def team_predict(d: pd.DataFrame) -> np.ndarray:
    p = np.clip(d["team_ft_raw"].to_numpy(dtype="float64"), 1e-6, 1 - 1e-6)
    return np.column_stack([1.0 - p, p])


class RidgeArm:
    def __init__(self, C: float = 1.0, max_iter: int = 300, seed: int = 0):
        self.C, self.max_iter, self.seed = C, max_iter, seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeArm:
        from sklearn.linear_model import LogisticRegression

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ < 1e-8] = 1.0
        self.clf_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs",
                                       random_state=self.seed).fit((X - self.mu_) / self.sd_, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf_.predict_proba((X - self.mu_) / self.sd_)


class LgbmArm:
    PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.06, num_leaves=63,
                  min_child_samples=400, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.9, reg_lambda=1.0, verbose=-1)

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> LgbmArm:
        import lightgbm as lgb

        self.clf_ = lgb.LGBMClassifier(random_state=self.seed, **self.PARAMS)
        self.clf_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf_.predict_proba(X)


def design_matrix(d: pd.DataFrame, features: tuple[str, ...] = FT_FEATURES) -> np.ndarray:
    """The feature matrix in the declared feature order.

    Public because the sim calls it: `model.md` section 7 is the contract and
    the order is `FT_FEATURES`, not whatever order the caller's frame happens
    to have."""
    return np.ascontiguousarray(d[list(features)].to_numpy(dtype="float32"))


# ===========================================================================
# Metrics
# ===========================================================================
#: The L3 round-1 reading of "monotone in 4 of 5 quintile steps": five
#: quintiles give four steps, so one violation is allowed.
RESPONSIVENESS_MIN_STEPS = 3


def responsiveness(te: pd.DataFrame, p: np.ndarray, n_q: int = 5) -> dict[str, dict]:
    return {"shooter_ft_asof->MAKE": PM.quintile_responsiveness(
        te["shooter_ft_asof"].to_numpy(), te["y"].to_numpy(), p, CLASS_INDEX["MAKE"], n_q=n_q)}


def score(te: pd.DataFrame, p: np.ndarray) -> dict:
    y = te["y"].to_numpy()
    calib = PM.decile_calibration(y, p, CLASSES)
    calib_ok, calib_worst, calib_who = PM.calibration_verdict(calib)
    resp = responsiveness(te, p)
    resp_ok, resp_worst = PM.responsiveness_verdict(resp, RESPONSIVENESS_MIN_STEPS)
    seg = PM.segment_calibration(
        np.where(te["in_bonus"].to_numpy() > 0, "bonus", "shooting"), y, p, CLASSES)
    return {
        "n": int(len(te)),
        "log_loss": PM.log_loss(y, p),
        "brier": float(((p[:, CLASS_INDEX["MAKE"]] - y) ** 2).mean()),
        "calibration": calib,
        "calib_pass": bool(calib_ok),
        "calib_worst_gap_pp": calib_worst,
        "calib_worst_class": calib_who,
        "responsiveness": resp,
        "resp_pass": bool(resp_ok),
        "resp_min_steps": resp_worst,
        "by_bonus": seg,
    }


__all__ = [
    "ARMS", "BONUS_THRESHOLDS_DEFAULT", "CLASSES", "CLASS_INDEX", "FOLDS",
    "FOUL_CLASSES", "FT_FEATURES", "PRIOR_KINDS", "RESPONSIVENESS_MIN_STEPS",
    "SELECTION_FOLD", "SHRINK_GRID", "TRIP_RULES", "LgbmArm", "RidgeArm",
    "build_ft_design", "build_trips_and_attempts", "classify_foul", "design_matrix",
    "derive_bonus_thresholds", "eb_predict", "fit_eb", "fold_slices",
    "load_bonus_era", "load_positions", "responsiveness", "score",
    "team_predict", "verify_trip_rules",
]
