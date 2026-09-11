"""
conference.py -- conference-game flags and conference-boundary dates.

WHY (ARCHITECTURE_DECISIONS.md Decision 9). The first four to six weeks of a
CBB season are mostly non-conference games against opponents of very different
quality from the conference schedule that follows. Decision 9 makes (b) a
conference-game flag a first-class feature candidate, audited like home/away,
and (c) the alignment of the in-season refit cadence a bake-off dimension --
both need one definition of "conference game" and one definition of "this
team's conference boundary", used identically by every sub-model.

DEFINITION. A game is a conference game when hoopR's schedule reports the same
non-null conference id on both sides (`home_conference_id == away_conference_id`,
both present). A team's conference boundary is the date of its FIRST
REGULAR-SEASON conference game of that season. Postseason games (`season_type`
3: conference tournaments and the NCAA/NIT fields) are excluded from the
boundary search -- a conference tournament game is a conference game but it is
not the start of conference play, and including it would hand a boundary date
in March to any team whose regular-season conference schedule the feed missed.

LEAK SAFETY. Both quantities come from the SCHEDULE, which is published before
the season is played: whether a given fixture is a conference game, and the
date of a team's first conference fixture, are knowable on day one and contain
no result. They are therefore legitimate pregame features and legitimate
refit-calendar inputs. Nothing here reads a score. The one thing a caller must
still not do is use a boundary date to select TRAINING ROWS from after it; the
refit machinery keeps its own strictly-before rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCHEDULE_DIR = Path("data/raw/hoopr/schedules")
UNIVERSE_PATH = Path("data/processed/games_universe.parquet")
REGULAR_SEASON_TYPE = 2


def load_schedule_conference(seasons: list[int],
                             schedule_dir: Path | str = SCHEDULE_DIR) -> pd.DataFrame:
    """`game_id, home_conference_id, away_conference_id` for `seasons`."""
    d = Path(schedule_dir)
    frames = []
    for s in seasons:
        p = d / f"mbb_schedule_{int(s)}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"no hoopR schedule for season {s} at {p}")
        frames.append(pd.read_parquet(
            p, columns=["game_id", "home_conference_id", "away_conference_id"]))
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("game_id")


def build_conference_flags(seasons: list[int],
                           universe_path: Path | str = UNIVERSE_PATH,
                           schedule_dir: Path | str = SCHEDULE_DIR) -> pd.DataFrame:
    """One row per D-I game in `seasons` with `is_conf_game`.

    Columns: season, game_id, game_date, season_type, home_team_id,
    away_team_id, home_conference_id, away_conference_id, is_conf_game.

    A game whose schedule row carries a null conference id on either side is
    `is_conf_game = False` and is COUNTED, not dropped: the count is reported by
    every caller so a feed gap can never masquerade as a modelling result
    (L22 -- a builder never fills a missing value with a convenient constant;
    False here is the observable "not known to be a conference game", and the
    count of such games is carried in `df.attrs`)."""
    seasons = [int(s) for s in seasons]
    u = pd.read_parquet(universe_path)
    u = u[u["season"].isin(seasons) & u["is_d1_game"]].copy()
    u["game_date"] = pd.to_datetime(u["game_date"])
    sched = load_schedule_conference(seasons, schedule_dir)
    m = u.merge(sched, on="game_id", how="left")
    both = m["home_conference_id"].notna() & m["away_conference_id"].notna()
    m["is_conf_game"] = (both & (m["home_conference_id"] == m["away_conference_id"]))
    m.attrs["n_missing_conference_id"] = int((~both).sum())
    m.attrs["n_games"] = int(len(m))
    keep = ["season", "game_id", "game_date", "season_type", "home_team_id",
            "away_team_id", "home_conference_id", "away_conference_id", "is_conf_game"]
    out = m[keep].copy()
    out.attrs.update(m.attrs)
    return out


def first_conference_game_dates(conf: pd.DataFrame) -> pd.DataFrame:
    """`season, team_id, first_conf_date` -- each team's conference boundary.

    Regular season only (see the module docstring). A team with no regular-season
    conference game in the feed gets NaT and is reported by the caller; it is
    never given a fabricated boundary."""
    reg = conf[(conf["season_type"] == REGULAR_SEASON_TYPE) & conf["is_conf_game"]]
    long = pd.concat([
        reg[["season", "home_team_id", "game_date"]].rename(columns={"home_team_id": "team_id"}),
        reg[["season", "away_team_id", "game_date"]].rename(columns={"away_team_id": "team_id"}),
    ], ignore_index=True)
    out = long.groupby(["season", "team_id"], as_index=False)["game_date"].min()
    return out.rename(columns={"game_date": "first_conf_date"})


def conference_boundary_dates(conf: pd.DataFrame, season: int) -> list[pd.Timestamp]:
    """The DISTINCT dates on which at least one team plays its first conference
    game of `season`, in order.

    These are the refit dates of the `S1-conf-aligned` scheme. Refitting once
    per distinct boundary date and scoring each game with the latest refit at or
    before its own date gives every team a fit that already knows about its own
    conference start, at the cost of one fit per date rather than one per team
    (Decision 9's "risk acknowledged")."""
    f = first_conference_game_dates(conf)
    f = f[(f["season"] == int(season)) & f["first_conf_date"].notna()]
    return sorted(pd.to_datetime(f["first_conf_date"]).unique().tolist())


def weekly_boundaries(dates: pd.Series, weekday: int = 0) -> list[pd.Timestamp]:
    """The refit dates of the `S1-weekly` scheme: every `weekday` (0 = Monday)
    on or before the first game date, through the last, that has at least one
    game in the week beginning on it."""
    d = pd.to_datetime(pd.Series(dates)).dropna()
    if not len(d):
        return []
    start = d.min().normalize()
    start = start - pd.Timedelta(days=int((start.weekday() - weekday) % 7))
    end = d.max().normalize()
    weeks = pd.date_range(start, end, freq="7D")
    idx = np.searchsorted(weeks.to_numpy(), d.to_numpy(), side="right") - 1
    used = sorted(set(int(i) for i in idx if i >= 0))
    return [weeks[i] for i in used]


def union_boundaries(*sets: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Sorted union of refit-date lists, de-duplicated to the day."""
    out: set[pd.Timestamp] = set()
    for s in sets:
        out.update(pd.Timestamp(x).normalize() for x in s)
    return sorted(out)
