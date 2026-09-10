"""
universe.py -- L0 game universe builder.

Produces one row per game for hoopR/CBBD-covered seasons 2022-2026 (hoopR/CBBD
"season" = the *ending* year of the season, e.g. 2022 == the 2021-22 season).
The canonical `game_id` is the hoopR/ESPN game id. See
`scripts/build_game_universe.py` for the CLI entry point, and
`docs/tests/data_audit_hoopr_2026-09-10.md` / `data_audit_cbbd_2026-09-10.md`
for the underlying data-quality findings this module encodes.

Key design decisions
---------------------
- **D-I flag.** A `(season, team_id)` pair is D-I if that team_id appears in
  that season's hoopR schedules with a *non-null* conference_id in at least
  `MIN_D1_GAMES` (5) games that season (home or away side, whichever side it
  played on). `is_d1_game` requires both teams D-I for that game's season.
  This mirrors the hoopR audit's own D-I proxy (`home_conference_id` /
  `away_conference_id` nullness), but requires >=5 games rather than a single
  null observation, because the audit found a handful of true D-I teams with
  spurious single-game nulls (data gaps, not real non-D-I status) -- see
  Section 1 of the hoopR audit ("home-side nulls are a mix of legitimate
  non-D-I host venues ... and a couple of apparent data gaps for nominally
  D-I hosts").

- **cbbd_game_id.** CBBD's own internal `id` field (not `sourceId`), matched
  onto the hoopR `game_id` via CBBD's `sourceId` column. `sourceId` is the
  ESPN/hoopR game id verbatim: verified 100% (or near-100%, off by a couple of
  postponed/rescheduled games) row-count overlap and >=99.9% home-score
  agreement between hoopR schedules and CBBD games in every one of the 5
  seasons audited here. This is a much stronger join key than
  date+score+name-similarity and is used instead of that fuzzier approach.

- **n_periods / OT.** From schedule `home_linescores` (present 2023+: count of
  period entries in the serialized linescore list, same regex approach as
  `scripts/diag_hoopr_audit.py`'s `linescore_periods`). For 2022 (schema
  drift: `home_linescores` column is absent that season entirely) falls back
  to `max(pbp period_number)` for that game_id -- the same fallback the hoopR
  audit itself uses and validates in Section 4/6.

- **pbp_truncated.** True when the pbp feed's own last-recorded running score
  never reaches the schedule's final score for a completed game (a partial /
  truncated feed, not a full game_id-level gap) -- identical definition to
  the hoopR audit's Section 4 "truncated/partial pbp feed" check. This is a
  **hoopR-side** flag.

- **pbp_complete (added 2026-09-10).** The **CBBD-side** counterpart, and a
  strictly stronger statement than `~pbp_truncated`. True when the CBBD event
  stream's own SCORING EVENTS account for the schedule's final score exactly,
  for BOTH teams:

      pbp_complete = the game has CBBD pbp rows
                     AND sum(points of classified scoring events by team T)
                         == final score of T, for both T, with zero slack

  Points are taken from `cbb_sim.pbp.events.classify_frame`: 3 for a made
  `FGA_3`, 2 for a made `FGA_rim`/`FGA_jump2`, 1 for every `FT_made`
  **including technical free throws** (they belong to no possession, so the
  possession table carries them in `tech_points_off`/`tech_points_def`, but
  they are scoring and they count here). Sides are resolved with the same
  `_fix_flipped_sides` repair the possession layer uses, so a game with an
  inverted `isHomeTeam` flag is not counted incomplete for that reason.

  WHY THIS DEFINITION AND NOT THE RUNNING-SCORE ONE. Two different
  completeness numbers were in circulation for 2022 -- 19.5%
  (`docs/tests/possessions_build_2026-09-10.md` section 2, event-derived) and
  3.6% (`docs/tests/shot_classification_diag_2026-09-10.md` section 8, from
  the feed's own last running `homeScore`/`awayScore`). They are reconciled
  in `docs/tests/possessions_build_v2_2026-09-10.md` section 2: technical
  free throws are NOT the cause (they are already counted on both sides of
  that comparison, and removing them changes nothing), and neither are
  unteamed rows, unclassifiable `Not Available` rows, or made-flag nulls --
  all four of those are measured at exactly zero. The cause is that CBBD's
  2022-2023 stream is missing scoring **rows** while its running-score
  **column**, carried forward on every later row, still reaches the final
  score. So the running-score check cannot see a missing basket and the
  event-sum check can. A model that trains on events needs the event-level
  guarantee, which is why `pbp_complete` uses the strict one.

  It is added ALONGSIDE `pbp_truncated`, never replacing it: the two come
  from independently sourced feeds and the rows where they disagree are
  themselves informative about which single feed failed for a given game.

- **tipoff_utc.** hoopR's `game_date_time` is tz-aware (`America/New_York`
  already applied on ingest); converted to UTC here. `game_date` (a plain
  date, already US-local per ESPN's own convention) is carried through as-is
  for `game_date`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HOOPR_DIR = Path("data/raw/hoopr")
DEFAULT_CBBD_DIR = Path("data/raw/cbbd")
DEFAULT_SEASONS = [2022, 2023, 2024, 2025, 2026]
SEALED_SEASON = 2026

MIN_D1_GAMES = 5

HOOPR_STEM = {
    "schedules": "mbb_schedule",
    "pbp": "play_by_play",
    "team_box": "team_box",
    "player_box": "player_box",
}


def _hoopr_path(hoopr_dir: Path, dataset: str, season: int) -> Path:
    return Path(hoopr_dir) / dataset / f"{HOOPR_STEM[dataset]}_{season}.parquet"


def _to_int64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_schedules(hoopr_dir: Path, seasons: list[int]) -> pd.DataFrame:
    """Concatenate hoopR schedules across seasons. `game_id` is coerced to a
    nullable Int64 for reliable joins (dtype drifts int32/int64/str across
    seasons in the raw files)."""
    frames = []
    for season in seasons:
        path = _hoopr_path(hoopr_dir, "schedules", season)
        if not path.exists():
            raise FileNotFoundError(f"missing hoopR schedules file: {path}")
        df = pd.read_parquet(path)
        df["season"] = int(season)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["game_id"] = _to_int64(out["game_id"])
    out["home_id"] = _to_int64(out["home_id"])
    out["away_id"] = _to_int64(out["away_id"])
    return out


def load_cbbd_games(cbbd_dir: Path, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = Path(cbbd_dir) / f"games_{season}.parquet"
        if not path.exists():
            continue
        cols = ["id", "sourceId", "season", "homeTeamId", "awayTeamId", "homePoints", "awayPoints"]
        df = pd.read_parquet(path, columns=cols)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["id", "sourceId", "season", "homeTeamId", "awayTeamId", "homePoints", "awayPoints"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["sourceId"] = _to_int64(out["sourceId"])
    out["id"] = _to_int64(out["id"])
    return out


# --------------------------------------------------------------------------
# n_periods / OT
# --------------------------------------------------------------------------
def _linescore_periods(s) -> float:
    """Count period entries in a serialized hoopR linescores string. Matches
    scripts/diag_hoopr_audit.py's linescore_periods exactly."""
    if not isinstance(s, str):
        return np.nan
    return float(len(re.findall(r"'value'", s)))


def compute_n_periods(schedules: pd.DataFrame, hoopr_dir: Path) -> pd.Series:
    """Per-row n_periods: from home_linescores where the column exists and
    parses; else max(pbp period_number) for that game_id/season (2022 only,
    where home_linescores is absent from the schema entirely)."""
    if "home_linescores" in schedules.columns:
        out = schedules["home_linescores"].map(_linescore_periods)
    else:
        out = pd.Series(np.nan, index=schedules.index, dtype="float64")

    missing_mask = out.isna()
    if missing_mask.any():
        for season in schedules.loc[missing_mask, "season"].unique():
            path = _hoopr_path(hoopr_dir, "pbp", season)
            if not path.exists():
                continue
            pbp = pd.read_parquet(path, columns=["game_id", "period_number"])
            pbp["game_id"] = _to_int64(pbp["game_id"])
            period_max = pbp.groupby("game_id")["period_number"].max()
            sub = missing_mask & (schedules["season"] == season)
            out.loc[sub] = schedules.loc[sub, "game_id"].map(period_max).astype("float64")
    return out


# --------------------------------------------------------------------------
# D-I flag
# --------------------------------------------------------------------------
def compute_d1_team_seasons(schedules: pd.DataFrame, min_games: int = MIN_D1_GAMES) -> set[tuple[int, int]]:
    """Return the set of (season, team_id) pairs considered D-I: team_id
    appears in that season's schedules with a non-null conference_id in
    >= min_games games (home or away side, whichever side it played on)."""
    home = schedules[["season", "home_id", "home_conference_id"]].rename(
        columns={"home_id": "team_id", "home_conference_id": "conference_id"}
    )
    away = schedules[["season", "away_id", "away_conference_id"]].rename(
        columns={"away_id": "team_id", "away_conference_id": "conference_id"}
    )
    appearances = pd.concat([home, away], ignore_index=True)
    appearances = appearances.dropna(subset=["team_id"])
    appearances["has_conf"] = appearances["conference_id"].notna()
    counts = appearances.groupby(["season", "team_id"])["has_conf"].sum()
    d1 = counts[counts >= min_games]
    return {(int(season), int(team_id)) for season, team_id in d1.index.tolist()}


def compute_is_d1_game(schedules: pd.DataFrame, d1_team_seasons: set[tuple[int, int]]) -> pd.Series:
    def _in_d1(season, team_id) -> bool:
        if pd.isna(team_id):
            return False
        return (int(season), int(team_id)) in d1_team_seasons

    home_d1 = schedules.apply(lambda r: _in_d1(r["season"], r["home_id"]), axis=1)
    away_d1 = schedules.apply(lambda r: _in_d1(r["season"], r["away_id"]), axis=1)
    return home_d1 & away_d1


# --------------------------------------------------------------------------
# pbp flags: has_pbp, pbp_truncated
# --------------------------------------------------------------------------
def compute_pbp_flags(schedules: pd.DataFrame, hoopr_dir: Path) -> pd.DataFrame:
    """Per game_id: has_pbp, pbp_truncated. Definition matches
    scripts/diag_hoopr_audit.py Section 4 exactly (last pbp running score vs.
    schedule final score, completed games only)."""
    parts = []
    for season in sorted(schedules["season"].unique()):
        sch_season = schedules[schedules["season"] == season]
        path = _hoopr_path(hoopr_dir, "pbp", season)
        if not path.exists():
            parts.append(pd.DataFrame({
                "game_id": sch_season["game_id"].values,
                "has_pbp": False,
                "pbp_truncated": False,
            }))
            continue

        pbp = pd.read_parquet(path, columns=["game_id", "home_score", "away_score"])
        pbp["game_id"] = _to_int64(pbp["game_id"])
        pbp_games = set(pbp["game_id"].dropna().unique())
        last_running = pbp.groupby("game_id")[["home_score", "away_score"]].last()

        has_pbp = sch_season["game_id"].isin(pbp_games).to_numpy()
        completed = (
            sch_season["status_type_completed"].to_numpy()
            if "status_type_completed" in sch_season.columns
            else np.ones(len(sch_season), dtype=bool)
        )

        merged = sch_season[["game_id", "home_score", "away_score"]].merge(
            last_running, left_on="game_id", right_index=True, how="left", suffixes=("", "_pbp")
        )
        mismatch = (
            merged["home_score"].astype(float).to_numpy() != merged["home_score_pbp"].astype(float).to_numpy()
        ) | (
            merged["away_score"].astype(float).to_numpy() != merged["away_score_pbp"].astype(float).to_numpy()
        )
        truncated = has_pbp & completed & mismatch

        parts.append(pd.DataFrame({
            "game_id": sch_season["game_id"].values,
            "has_pbp": has_pbp,
            "pbp_truncated": truncated,
        }))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# CBBD feed completeness: pbp_complete
# --------------------------------------------------------------------------
def compute_pbp_complete(
    schedules: pd.DataFrame,
    cbbd_game_id: pd.Series,
    cbbd_pbp_dir: Path | str = Path("data/raw/cbbd/pbp"),
) -> pd.DataFrame:
    """Per game: `pbp_complete`, plus the two per-team point deltas it is built
    from, so the flag is auditable instead of opaque.

    Definition and the reconciliation it settles: module docstring,
    "pbp_complete". Returns a frame indexed like `schedules` with columns
    `pbp_complete`, `cbbd_pbp_rows`, `cbbd_pts_delta_home`,
    `cbbd_pts_delta_away` (event points MINUS the final score, so 0 is
    complete, negative is a short feed and positive is a feed that scores more
    than the box).

    LAYERING NOTE. This L0 builder imports the L3 event classifier on purpose.
    The flag has to be computed with the SAME `classify_frame` and the SAME
    side repair the possession layer uses, or it would certify a feed as
    complete under one definition of "a scoring event" and be consumed under
    another.
    """
    from cbb_sim.pbp.events import classify_frame, load_plays
    from cbb_sim.pbp.possessions import _fix_flipped_sides

    n = len(schedules)
    out = pd.DataFrame({
        "pbp_complete": np.zeros(n, dtype=bool),
        "cbbd_pbp_rows": np.zeros(n, dtype="int64"),
        "cbbd_pts_delta_home": pd.Series([pd.NA] * n, dtype="Int64"),
        "cbbd_pts_delta_away": pd.Series([pd.NA] * n, dtype="Int64"),
    }, index=schedules.index)

    gid = _to_int64(pd.Series(cbbd_game_id).reset_index(drop=True))
    for season in sorted(int(s) for s in schedules["season"].unique()):
        path = Path(cbbd_pbp_dir) / f"plays_{season}.parquet"
        sub = (schedules["season"] == season).to_numpy()
        if not path.exists():
            continue
        wanted = set(gid[sub].dropna().astype("int64").tolist())
        if not wanted:
            continue
        plays = load_plays(season, pbp_dir=Path(cbbd_pbp_dir), game_ids=wanted)
        if not len(plays):
            continue
        cls = classify_frame(plays).to_numpy()

        made = plays["shot_made"]
        if made.dtype == object:
            made = made.map({True: True, False: False})
        made = (made.astype("boolean")
                .fillna(plays["scoringPlay"].astype("boolean"))
                .fillna(False).to_numpy(dtype=bool))
        pv = np.where(cls == "FGA_3", 3,
                      np.where(np.isin(cls, ["FGA_rim", "FGA_jump2"]), 2,
                               np.where(cls == "FT_made", 1, 0)))
        pv = np.where(np.isin(cls, ["FGA_3", "FGA_rim", "FGA_jump2"]) & ~made, 0, pv)

        is_home = plays["isHomeTeam"]
        if is_home.dtype == object:
            is_home = is_home.map({True: True, False: False})
        is_home = is_home.astype("boolean")
        has_team = (pd.to_numeric(plays["teamId"], errors="coerce").notna()
                    & is_home.notna()).to_numpy()
        side = np.where(is_home.fillna(False).to_numpy(), 0, 1)
        side = _fix_flipped_sides(plays, side, has_team)

        pts = pd.DataFrame({
            "cbbd_game_id": plays["gameId"].to_numpy(),
            "home_pts": np.where(has_team & (side == 0), pv, 0),
            "away_pts": np.where(has_team & (side == 1), pv, 0),
            "rows": 1,
        }).groupby("cbbd_game_id", as_index=True).sum()

        idx = schedules.index[sub]
        # float64 with NaN for the games that have no CBBD id at all; the
        # lookup below is done on the integer ids only, so a NaN id can never
        # accidentally match a row of `pts`.
        g = gid[sub].astype("float64").to_numpy(dtype="float64", na_value=np.nan)
        key = pd.Series(np.where(np.isfinite(g), g, -1).astype("int64"))
        hp = key.map(pts["home_pts"]).to_numpy(dtype="float64")
        ap = key.map(pts["away_pts"]).to_numpy(dtype="float64")
        rows = key.map(pts["rows"]).fillna(0).to_numpy(dtype="int64")
        fh = pd.to_numeric(schedules.loc[idx, "home_score"], errors="coerce").to_numpy(dtype="float64")
        fa = pd.to_numeric(schedules.loc[idx, "away_score"], errors="coerce").to_numpy(dtype="float64")
        dh, da = hp - fh, ap - fa
        ok = (rows > 0) & (dh == 0) & (da == 0)
        out.loc[idx, "pbp_complete"] = ok
        out.loc[idx, "cbbd_pbp_rows"] = rows
        out.loc[idx, "cbbd_pts_delta_home"] = pd.array(dh, dtype="float64").astype("Int64")
        out.loc[idx, "cbbd_pts_delta_away"] = pd.array(da, dtype="float64").astype("Int64")
        del plays, cls
    return out


# --------------------------------------------------------------------------
# player_box flag
# --------------------------------------------------------------------------
def compute_has_player_box(schedules: pd.DataFrame, hoopr_dir: Path) -> pd.Series:
    out = pd.Series(False, index=schedules.index)
    for season in schedules["season"].unique():
        path = _hoopr_path(hoopr_dir, "player_box", season)
        sub = schedules["season"] == season
        if not path.exists():
            continue
        pb = pd.read_parquet(path, columns=["game_id"])
        pb["game_id"] = _to_int64(pb["game_id"])
        games = set(pb["game_id"].dropna().unique())
        out.loc[sub] = schedules.loc[sub, "game_id"].isin(games)
    return out


# --------------------------------------------------------------------------
# CBBD join
# --------------------------------------------------------------------------
def compute_cbbd_join(schedules: pd.DataFrame, cbbd_dir: Path) -> pd.DataFrame:
    """Join hoopR game_id -> CBBD internal game id via CBBD's `sourceId`
    column (verified == hoopR/ESPN game_id -- see module docstring)."""
    seasons = sorted(int(s) for s in schedules["season"].unique())
    cbbd_games = load_cbbd_games(cbbd_dir, seasons)
    join = schedules[["game_id"]].merge(
        cbbd_games[["id", "sourceId"]].rename(columns={"id": "cbbd_game_id", "sourceId": "game_id"}),
        on="game_id",
        how="left",
    )
    return join[["cbbd_game_id"]].set_axis(schedules.index)


def compute_has_cbbd_line(cbbd_game_ids: pd.Series, seasons: list[int], cbbd_dir: Path) -> pd.Series:
    lines_frames = []
    for season in seasons:
        path = Path(cbbd_dir) / f"lines_{season}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["gameId", "spread", "overUnder", "homeMoneyline"])
        lines_frames.append(df)
    if not lines_frames:
        return pd.Series(False, index=cbbd_game_ids.index)
    lines = pd.concat(lines_frames, ignore_index=True)
    has_line = lines[["spread", "overUnder", "homeMoneyline"]].notna().any(axis=1)
    line_game_ids = set(_to_int64(lines.loc[has_line, "gameId"]).dropna().unique())
    return cbbd_game_ids.isin(line_game_ids)


def compute_join_quality(schedules: pd.DataFrame, cbbd_dir: Path) -> pd.DataFrame:
    """Per-season CBBD<->hoopR join-quality report: overlap counts and
    home-score agreement rate on the sourceId==game_id join. Used only for
    the human-readable report, not for the universe table itself."""
    rows = []
    for season in sorted(int(s) for s in schedules["season"].unique()):
        sch = schedules[schedules["season"] == season]
        cbbd = load_cbbd_games(cbbd_dir, [season])
        sched_ids = set(sch["game_id"].dropna().unique())
        cbbd_ids = set(cbbd["sourceId"].dropna().unique())
        merged = sch[["game_id", "home_score", "away_score"]].merge(
            cbbd[["sourceId", "homePoints", "awayPoints"]], left_on="game_id", right_on="sourceId", how="inner"
        )
        score_ok = (
            (merged["home_score"].astype(float) == merged["homePoints"].astype(float))
            & (merged["away_score"].astype(float) == merged["awayPoints"].astype(float))
        )
        rows.append({
            "season": season,
            "hoopr_games": len(sched_ids),
            "cbbd_games": len(cbbd_ids),
            "matched_games": len(merged),
            "score_agreement_rate": float(score_ok.mean()) if len(merged) else float("nan"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main builder
# --------------------------------------------------------------------------
def build_universe(
    seasons: list[int] | None = None,
    hoopr_dir: Path | str = DEFAULT_HOOPR_DIR,
    cbbd_dir: Path | str = DEFAULT_CBBD_DIR,
    with_pbp_complete: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build the games_universe table plus a report dict of per-season flag
    counts and D-I-exclusion diagnostics (used by build_game_universe.py to
    print/write the summary)."""
    seasons = list(seasons or DEFAULT_SEASONS)
    hoopr_dir = Path(hoopr_dir)
    cbbd_dir = Path(cbbd_dir)

    sch = load_schedules(hoopr_dir, seasons)

    n_periods = compute_n_periods(sch, hoopr_dir)
    d1_team_seasons = compute_d1_team_seasons(sch)
    is_d1_game = compute_is_d1_game(sch, d1_team_seasons)
    pbp_flags = compute_pbp_flags(sch, hoopr_dir)
    has_player_box = compute_has_player_box(sch, hoopr_dir)
    cbbd_join = compute_cbbd_join(sch, cbbd_dir)

    cbbd_game_id = cbbd_join["cbbd_game_id"]
    has_cbbd_line = compute_has_cbbd_line(cbbd_game_id, seasons, cbbd_dir)
    if with_pbp_complete:
        completeness = compute_pbp_complete(sch, cbbd_game_id, Path(cbbd_dir) / "pbp")
    else:
        completeness = pd.DataFrame({
            "pbp_complete": np.zeros(len(sch), dtype=bool),
            "cbbd_pbp_rows": np.zeros(len(sch), dtype="int64"),
            "cbbd_pts_delta_home": pd.Series([pd.NA] * len(sch), dtype="Int64"),
            "cbbd_pts_delta_away": pd.Series([pd.NA] * len(sch), dtype="Int64"),
        }, index=sch.index)

    tipoff_utc = pd.to_datetime(sch["game_date_time"], utc=True, errors="coerce")

    out = pd.DataFrame({
        "game_id": sch["game_id"].astype("int64"),
        "cbbd_game_id": cbbd_game_id,
        "season": sch["season"].astype("int64"),
        "season_type": sch["season_type"],
        "game_date": pd.to_datetime(sch["game_date"]).dt.date,
        "tipoff_utc": tipoff_utc,
        "home_team_id": sch["home_id"].astype("int64"),
        "away_team_id": sch["away_id"].astype("int64"),
        "home_display_name": sch["home_display_name"],
        "away_display_name": sch["away_display_name"],
        "neutral_site": sch["neutral_site"].astype("bool"),
        "home_score": sch["home_score"],
        "away_score": sch["away_score"],
        "n_periods": n_periods,
        "is_d1_game": is_d1_game.to_numpy(),
        "has_pbp": pbp_flags["has_pbp"].to_numpy(),
        "pbp_truncated": pbp_flags["pbp_truncated"].to_numpy(),
        "has_player_box": has_player_box.to_numpy(),
        "has_cbbd_line": has_cbbd_line.to_numpy(),
        "sealed": (sch["season"] == SEALED_SEASON).to_numpy(),
        # CBBD-side feed completeness (module docstring, "pbp_complete")
        "pbp_complete": completeness["pbp_complete"].to_numpy(),
        "cbbd_pbp_rows": completeness["cbbd_pbp_rows"].to_numpy(),
        "cbbd_pts_delta_home": completeness["cbbd_pts_delta_home"].to_numpy(),
        "cbbd_pts_delta_away": completeness["cbbd_pts_delta_away"].to_numpy(),
    })

    # ---- report ----
    total_games = len(sch)
    non_d1_games = int((~is_d1_game).sum())
    join_quality = compute_join_quality(sch, cbbd_dir)

    flag_cols = ["is_d1_game", "has_pbp", "pbp_truncated", "has_player_box", "has_cbbd_line",
                 "pbp_complete"]
    per_season = out.groupby("season")[flag_cols + ["game_id"]].agg(
        n_games=("game_id", "count"),
        is_d1_game=("is_d1_game", "sum"),
        has_pbp=("has_pbp", "sum"),
        pbp_truncated=("pbp_truncated", "sum"),
        has_player_box=("has_player_box", "sum"),
        has_cbbd_line=("has_cbbd_line", "sum"),
        pbp_complete=("pbp_complete", "sum"),
    )

    # pbp_complete share among the games any L3 fold can actually use
    d1 = out[out["is_d1_game"]]
    complete_share = d1.groupby("season").agg(
        n_d1=("game_id", "count"),
        n_complete=("pbp_complete", "sum"),
    )
    complete_share["pct_complete"] = (
        100.0 * complete_share["n_complete"] / complete_share["n_d1"]).round(2)
    d1nt = d1[~d1["pbp_truncated"]]
    complete_share["n_d1_not_truncated"] = d1nt.groupby("season")["game_id"].count()
    complete_share["n_d1_not_truncated_complete"] = d1nt.groupby("season")["pbp_complete"].sum()
    complete_share["pct_complete_of_not_truncated"] = (
        100.0 * complete_share["n_d1_not_truncated_complete"]
        / complete_share["n_d1_not_truncated"]).round(2)

    report = {
        "total_games": total_games,
        "non_d1_games": non_d1_games,
        "d1_team_seasons_n": len(d1_team_seasons),
        "per_season_counts": per_season,
        "join_quality": join_quality,
        "n_periods_missing": int(n_periods.isna().sum()),
        "pbp_complete_by_season": complete_share,
    }
    return out, report
