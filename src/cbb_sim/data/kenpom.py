"""
kenpom.py -- point-in-time KenPom snapshot loader.

WHY. CLAUDE.md's "Modeling rules" section: "Every rating feature is expressed
relative to its own snapshot's league mean. Raw levels are banned (KenPom's
league-mean AdjO drifted 100 -> 109.3 across last year's data and inflated
every model)." This module builds the point-in-time snapshot table that makes
that centering possible, plus the `as_of()` / `join_as_of()` lookups a
feature-builder uses to join strictly BEFORE a game's tipoff -- KenPom
re-scrapes the morning after games are played, so a same-day snapshot already
contains that day's result. See docs/postmortem/05_cfb_methodology_extract.md
section 4 (the INV-45 method), ported to CBB in
`src/cbb_sim/analysis/leak_test.py`.

SOURCES.
  weekly    C:\\Users\\devuser\\CBB-Monte\\{season}_kenpom.csv         seasons 2022-2025
            columns: Date (YYYYMMDD int), Team, Conf, NetRtg, ORtg, DRtg, AdjT.
            No rank column -- rank_net is computed here (NetRtg descending,
            ties get the same rank, "min" method).
  daily     C:\\Users\\devuser\\CBB-Monte-storage\\data\\kenpom\\2026\\*.csv     season 2026
  preseason C:\\Users\\devuser\\CBB-Monte-storage\\data\\kenpom\\2027\\*.csv     season 2027 (if present)
            One file per snapshot date. The filename, not the internal
            `snapshot_date` column, is treated as the date of record: the
            very first 2026 file is stamped M/D/YYYY ("10/13/2025") while
            every later file is YYYY-MM-DD ("2026-04-30") -- a schema drift
            inside the column itself, sidestepped entirely by parsing the
            date out of the `YYYY-MM-DD_kenpom.csv` filename instead.
            Columns: snapshot_date, Rk, Team, Conf, W-L, NetRtg, ORtg, ORnk,
            DRtg, DRnk, AdjT, AdjTRnk, Luck, ..., NCSOSNetRnk. `Rk` is
            spot-checked to be the NetRtg-descending rank and used directly.

SEASON CONVENTION. hoopR/CBBD convention: season = the ending year of the
season (2025-26 season -> 2026). For weekly files the season is the file's
own year label (already that convention). For daily/preseason files the
season is the containing folder name -- note the 2027 (2026-27 preseason)
files carry calendar dates in May-June 2026; the folder name, not the
calendar year, is the season label.

NAME MATCHING. KenPom team names differ from ESPN/hoopR `team_box.team_location`
names in three systematic ways, resolved in order:
  1. Scrape artifacts fixed unconditionally: a stray ';' glued onto an
     ampersand abbreviation ("Texas A&M;" -> "Texas A&M") and a trailing
     " <digits>" NCAA-tournament seed suffix KenPom's site appends during
     March ("Michigan 1" -> "Michigan").
  2. `NAME_ALIASES`, a ~100-entry explicit table (renamed schools, "St." used
     as "State" vs. as "Saint", nickname vs. formal name -- e.g. "Ole Miss"
     for "Mississippi", "UConn" for "Connecticut") built by diffing the full
     KenPom 2022-2025 weekly name universe against hoopR's 2022-2026
     `team_location` universe and resolving every non-exact mismatch by hand.
  3. `normalize_join_key()`, a whitespace/hyphen/period/apostrophe-insensitive
     fallback key, used for BOTH sides of every match -- this absorbs
     spacing drift hoopR itself is not consistent about (e.g. "St. Thomas -
     Minnesota" in the 2022 team_box vs. "St. Thomas-Minnesota" from 2023 on).

Unmatched names are never silently dropped: they are kept with `team` = NaN,
and `build_snapshots()` returns a match report alongside the table so the
caller (`scripts/build_kenpom_snapshots.py`) can print the residual count.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

WEEKLY_DIR = Path(r"C:\Users\devuser\CBB-Monte")
DAILY_ROOT = Path(r"C:\Users\devuser\CBB-Monte-storage\data\kenpom")
HOOPR_ROOT = Path("data/raw/hoopR")

WEEKLY_SEASONS_DEFAULT: tuple[int, ...] = (2022, 2023, 2024, 2025)
DAILY_SEASONS_DEFAULT: tuple[int, ...] = (2026, 2027)
HOOPR_MATCH_SEASONS_DEFAULT: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)

_SEED_SUFFIX_RE = re.compile(r"\s+\d+$")
_DAILY_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_kenpom\.csv$", re.IGNORECASE)
_HYPHEN_SPACING_RE = re.compile(r"\s*-\s*")
_WHITESPACE_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Name normalization / crosswalk to hoopR team_box.team_location
# ---------------------------------------------------------------------------

# KenPom raw (post scrape-artifact cleanup) -> hoopR/ESPN team_location.
# Built by diffing the full 2022-2025 weekly KenPom name universe (377 unique
# raw names) against the 2022-2025 hoopR team_box team_location universe
# (1122 unique names): 271 matched exactly, these 100 cover every remaining
# non-exact mismatch (6 more collapse to exact matches after the ';' /
# trailing-digit cleanup and need no entry here).
NAME_ALIASES: dict[str, str] = {
    "Alabama St.": "Alabama State",
    "Alcorn St.": "Alcorn State",
    "American": "American University",
    "Appalachian St.": "Appalachian State",
    "Arizona St.": "Arizona State",
    "Arkansas Pine Bluff": "Arkansas-Pine Bluff",
    "Arkansas St.": "Arkansas State",
    "Ball St.": "Ball State",
    "Bethune Cookman": "Bethune-Cookman",
    "Boise St.": "Boise State",
    "CSUN": "Cal State Northridge",
    "Cal Baptist": "California Baptist",
    "Cal St. Bakersfield": "Cal State Bakersfield",
    "Cal St. Fullerton": "Cal State Fullerton",
    "Cal St. Northridge": "Cal State Northridge",
    "Chicago St.": "Chicago State",
    "Cleveland St.": "Cleveland State",
    "Colorado St.": "Colorado State",
    "Connecticut": "UConn",
    "Coppin St.": "Coppin State",
    "Delaware St.": "Delaware State",
    "Dixie St.": "Utah Tech",
    "East Tennessee St.": "East Tennessee State",
    "FIU": "Florida International",
    "Florida St.": "Florida State",
    "Fresno St.": "Fresno State",
    "Gardner Webb": "Gardner-Webb",
    "Georgia St.": "Georgia State",
    "Grambling St.": "Grambling",  # ESPN/hoopR team_location omits "State" for this school
    "Hawaii": "Hawai'i",
    "Houston Baptist": "Houston Christian",
    "IU Indy": "IU Indianapolis",
    "Idaho St.": "Idaho State",
    "Illinois Chicago": "UIC",
    "Illinois St.": "Illinois State",
    "Indiana St.": "Indiana State",
    "Iowa St.": "Iowa State",
    "Jackson St.": "Jackson State",
    "Jacksonville St.": "Jacksonville State",
    "Kansas St.": "Kansas State",
    "Kennesaw St.": "Kennesaw State",
    "Kent St.": "Kent State",
    "LIU": "Long Island University",
    "Long Beach St.": "Long Beach State",
    "Louisiana Monroe": "UL Monroe",
    "Loyola MD": "Loyola Maryland",
    "McNeese St.": "McNeese",  # ESPN/hoopR team_location omits "State" for this school
    "Miami FL": "Miami",
    "Miami OH": "Miami (OH)",
    "Michigan St.": "Michigan State",
    "Mississippi": "Ole Miss",
    "Mississippi St.": "Mississippi State",
    "Mississippi Valley St.": "Mississippi Valley State",
    "Missouri St.": "Missouri State",
    "Montana St.": "Montana State",
    "Morehead St.": "Morehead State",
    "Morgan St.": "Morgan State",
    "Murray St.": "Murray State",
    "N.C. State": "NC State",
    "Nebraska Omaha": "Omaha",
    "New Mexico St.": "New Mexico State",
    "Nicholls St.": "Nicholls",
    "Norfolk St.": "Norfolk State",
    "North Dakota St.": "North Dakota State",
    "Northwestern St.": "Northwestern State",
    "Ohio St.": "Ohio State",
    "Oklahoma St.": "Oklahoma State",
    "Oregon St.": "Oregon State",
    "Penn": "Pennsylvania",
    "Penn St.": "Penn State",
    "Portland St.": "Portland State",
    "Queens": "Queens University",
    "SIUE": "SIU Edwardsville",
    "Sacramento St.": "Sacramento State",
    "Sam Houston St.": "Sam Houston",
    "San Diego St.": "San Diego State",
    "San Jose St.": "San Jos\u00e9 State",
    "Seattle": "Seattle U",
    "South Carolina St.": "South Carolina State",
    "South Dakota St.": "South Dakota State",
    "Southeast Missouri": "Southeast Missouri State",
    "Southeast Missouri St.": "Southeast Missouri State",
    "Southeastern Louisiana": "SE Louisiana",
    "St. Francis NY": "St. Francis Brooklyn",
    "St. Francis PA": "St. Francis (PA)",
    "St. Thomas": "St. Thomas-Minnesota",
    "Tarleton St.": "Tarleton State",
    "Tennessee Martin": "UT Martin",
    "Tennessee St.": "Tennessee State",
    "Texas A&M Commerce": "Texas A&M-Commerce",
    "Texas A&M Corpus Chris": "Texas A&M-Corpus Christi",
    "Texas St.": "Texas State",
    "UMKC": "Kansas City",
    "USC Upstate": "South Carolina Upstate",
    "Utah St.": "Utah State",
    "Washington St.": "Washington State",
    "Weber St.": "Weber State",
    "Wichita St.": "Wichita State",
    "Wright St.": "Wright State",
    "Youngstown St.": "Youngstown State",
}


def _clean_raw_name(name: object) -> str:
    """Fix scrape artifacts common to every KenPom export: a stray ';' glued
    onto an ampersand abbreviation, a trailing NCAA-tournament seed digit,
    and incidental whitespace."""
    s = str(name).strip()
    s = s.replace(";", "")
    s = _SEED_SUFFIX_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def normalize_join_key(name: object) -> str:
    """Whitespace/hyphen/period/apostrophe-insensitive join key, applied to
    BOTH the KenPom side and the hoopR side of every match."""
    s = _clean_raw_name(name).casefold()
    s = s.replace(".", "").replace("'", "")
    s = _HYPHEN_SPACING_RE.sub("-", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _load_hoopr_team_index(hoopr_root: Path, seasons: Sequence[int]) -> dict[str, str]:
    """join_key -> a canonical hoopR team_location display name, unioned
    across `seasons` (first-seen wins; renamed schools resolve to whichever
    season's spelling is encountered first, which is fine -- the join key is
    what actually matters for merges)."""
    index: dict[str, str] = {}
    for season in seasons:
        path = hoopr_root / "team_box" / f"team_box_{season}.parquet"
        if not path.exists():
            continue
        tb = pd.read_parquet(path, columns=["team_location"])
        for name in tb["team_location"].dropna().unique():
            key = normalize_join_key(name)
            index.setdefault(key, name)
    return index


def match_names(
    raw_names: Sequence[str], hoopr_index: dict[str, str]
) -> tuple[dict[str, Optional[str]], list[str]]:
    """Map each distinct cleaned KenPom name to a hoopR team_location name
    (or None). Returns (name_map, sorted_unmatched)."""
    name_map: dict[str, Optional[str]] = {}
    unmatched: list[str] = []
    for raw in raw_names:
        cleaned = _clean_raw_name(raw)
        candidate = NAME_ALIASES.get(cleaned, cleaned)
        key = normalize_join_key(candidate)
        matched = hoopr_index.get(key)
        name_map[raw] = matched
        if matched is None:
            unmatched.append(cleaned)
    return name_map, sorted(set(unmatched))


# ---------------------------------------------------------------------------
# Per-file loaders
# ---------------------------------------------------------------------------


def _load_weekly_file(path: Path, season: int) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(df["Date"].astype(str), format="%Y%m%d"),
            "season": season,
            "kenpom_name_raw": df["Team"],
            "adj_o": pd.to_numeric(df["ORtg"], errors="coerce"),
            "adj_d": pd.to_numeric(df["DRtg"], errors="coerce"),
            "adj_t": pd.to_numeric(df["AdjT"], errors="coerce"),
            "net_rtg": pd.to_numeric(df["NetRtg"], errors="coerce"),
        }
    )
    out["rank_net"] = out.groupby("snapshot_date")["net_rtg"].rank(method="min", ascending=False)
    out["source"] = "weekly"
    return out


def _load_daily_file(path: Path, season: int) -> pd.DataFrame:
    m = _DAILY_FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"cannot parse a YYYY-MM-DD snapshot date from filename: {path}")
    snap_date = pd.Timestamp(m.group(1))
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    # The 2027 preseason export re-embeds a literal duplicate header row every
    # ~40 data rows (a pagination artifact of the scrape) -- e.g. 324 stray
    # rows across the 45 files in data/kenpom/2027, each with Team=="Team"
    # and every numeric column null. Drop them; a real team is never named
    # "Team" and never has a null Team.
    df = df[df["Team"].notna() & (df["Team"] != "Team")].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "snapshot_date": snap_date,
            "season": season,
            "kenpom_name_raw": df["Team"],
            "adj_o": pd.to_numeric(df["ORtg"], errors="coerce"),
            "adj_d": pd.to_numeric(df["DRtg"], errors="coerce"),
            "adj_t": pd.to_numeric(df["AdjT"], errors="coerce"),
            "rank_net": pd.to_numeric(df["Rk"], errors="coerce"),
            "net_rtg": pd.to_numeric(df["NetRtg"], errors="coerce"),
        }
    )
    out["source"] = "daily"
    return out


def _load_daily_dir(dir_path: Path, season: int) -> pd.DataFrame:
    files = sorted(dir_path.glob("*_kenpom.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([_load_daily_file(f, season) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_snapshots(
    weekly_dir: Path = WEEKLY_DIR,
    weekly_seasons: Sequence[int] = WEEKLY_SEASONS_DEFAULT,
    daily_root: Path = DAILY_ROOT,
    daily_seasons: Sequence[int] = DAILY_SEASONS_DEFAULT,
    hoopr_root: Path = HOOPR_ROOT,
    hoopr_match_seasons: Sequence[int] = HOOPR_MATCH_SEASONS_DEFAULT,
) -> tuple[pd.DataFrame, dict]:
    """Load every available KenPom snapshot, match team names to hoopR
    `team_location`, and compute the centered (relative-to-own-snapshot-mean)
    features. Returns (snapshot_df, match_report).

    snapshot_df columns: snapshot_date, season, kenpom_name, team, adj_o,
    adj_d, adj_t, rank_net, adj_o_c, adj_d_c, adj_t_rel, snap_mean_adj_o,
    snap_mean_adj_d, snap_mean_adj_t, snap_n_teams, source.

    `team` is NaN for any KenPom name that did not match a hoopR
    `team_location` -- these rows are KEPT (never silently dropped); see
    `match_report["unmatched_names"]`.
    """
    frames: list[pd.DataFrame] = []
    for season in weekly_seasons:
        p = weekly_dir / f"{season}_kenpom.csv"
        if p.exists():
            frames.append(_load_weekly_file(p, season))
    for season in daily_seasons:
        d = daily_root / str(season)
        if d.exists():
            df = _load_daily_dir(d, season)
            if not df.empty:
                frames.append(df)
    if not frames:
        raise FileNotFoundError("no KenPom snapshot files found in any configured source directory")

    raw = pd.concat(frames, ignore_index=True)
    raw["kenpom_name"] = raw["kenpom_name_raw"].map(_clean_raw_name)
    raw = raw.drop(columns=["kenpom_name_raw"])

    hoopr_index = _load_hoopr_team_index(hoopr_root, hoopr_match_seasons)
    name_map, unmatched = match_names(raw["kenpom_name"].unique(), hoopr_index)
    raw["team"] = raw["kenpom_name"].map(name_map)

    grp = raw.groupby(["season", "snapshot_date"])
    raw["snap_mean_adj_o"] = grp["adj_o"].transform("mean")
    raw["snap_mean_adj_d"] = grp["adj_d"].transform("mean")
    raw["snap_mean_adj_t"] = grp["adj_t"].transform("mean")
    raw["snap_n_teams"] = grp["adj_o"].transform("count")
    raw["adj_o_c"] = raw["adj_o"] - raw["snap_mean_adj_o"]
    raw["adj_d_c"] = raw["adj_d"] - raw["snap_mean_adj_d"]
    raw["adj_t_rel"] = raw["adj_t"] / raw["snap_mean_adj_t"]

    cols = [
        "snapshot_date", "season", "kenpom_name", "team",
        "adj_o", "adj_d", "adj_t", "rank_net",
        "adj_o_c", "adj_d_c", "adj_t_rel",
        "snap_mean_adj_o", "snap_mean_adj_d", "snap_mean_adj_t", "snap_n_teams",
        "source",
    ]
    out = raw[cols].sort_values(["season", "snapshot_date", "team"]).reset_index(drop=True)

    n_unique = raw["kenpom_name"].nunique()
    match_report = {
        "n_unique_raw_names": int(n_unique),
        "n_matched": int(n_unique - len(unmatched)),
        "n_unmatched": int(len(unmatched)),
        "unmatched_names": unmatched,
    }
    return out, match_report


def as_of(
    snapshot_df: pd.DataFrame,
    team: str,
    date,
    team_col: str = "team",
    date_col: str = "snapshot_date",
) -> Optional[pd.Series]:
    """The latest snapshot row for `team` strictly BEFORE `date`.

    Never same-day: KenPom re-scrapes the morning after games are played, so
    a same-day snapshot already contains that day's result -- joining it
    entering-day would leak the outcome (the exact defect this whole harness
    exists to catch, INV-45). Returns None if no such snapshot exists.

    This is a convenience single-lookup (filters the whole table per call);
    for bulk feature-building over many games, use `join_as_of()` instead.
    """
    ts = pd.Timestamp(date)
    key = normalize_join_key(team)
    keys = snapshot_df[team_col].map(normalize_join_key)
    mask = (keys == key) & (snapshot_df[date_col] < ts)
    sub = snapshot_df.loc[mask]
    if sub.empty:
        return None
    return sub.loc[sub[date_col].idxmax()]


DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    "adj_o", "adj_d", "adj_t", "rank_net",
    "adj_o_c", "adj_d_c", "adj_t_rel",
    "snap_mean_adj_o", "snap_mean_adj_d", "snap_mean_adj_t",
)


def join_as_of(
    games: pd.DataFrame,
    snapshots: pd.DataFrame,
    game_team_col: str,
    game_date_col: str,
    snap_team_col: str = "team",
    snap_date_col: str = "snapshot_date",
    feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
    suffix: str = "_kp",
) -> pd.DataFrame:
    """Vectorized as-of join: for every row of `games`, attach the latest
    `snapshots` row for the same team (matched via `normalize_join_key`)
    strictly BEFORE `game_date_col`. Implemented with `pd.merge_asof(...,
    allow_exact_matches=False)`, which is the efficient equivalent of calling
    `as_of()` once per row.

    Rows whose team has no matching hoopR-crosswalked KenPom snapshot (unmatched
    name, or genuinely no snapshot before that date -- e.g. the team's first
    game of a season it has no preseason snapshot for) get NaN feature columns,
    never a fabricated value.
    """
    g = games.copy()
    g["_team_key"] = g[game_team_col].map(normalize_join_key)
    g[game_date_col] = pd.to_datetime(g[game_date_col]).astype("datetime64[ns]")
    g["_row_id"] = np.arange(len(g))
    g_sorted = g.sort_values(game_date_col, kind="mergesort")

    s = snapshots.dropna(subset=[snap_team_col]).copy()
    s["_team_key"] = s[snap_team_col].map(normalize_join_key)
    s[snap_date_col] = pd.to_datetime(s[snap_date_col]).astype("datetime64[ns]")
    s_sorted = s.sort_values(snap_date_col, kind="mergesort")
    right = s_sorted[["_team_key", snap_date_col, *feature_cols]]

    merged = pd.merge_asof(
        g_sorted,
        right,
        left_on=game_date_col,
        right_on=snap_date_col,
        by="_team_key",
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_snap"),
    )
    merged = merged.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id", "_team_key"])
    if snap_date_col in merged.columns and snap_date_col != game_date_col:
        merged = merged.rename(columns={snap_date_col: f"{snap_date_col}{suffix}"})
    rename_map = {c: f"{c}{suffix}" for c in feature_cols}
    merged = merged.rename(columns=rename_map)
    return merged.reset_index(drop=True)
