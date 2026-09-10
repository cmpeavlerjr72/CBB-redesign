"""
ids.py -- ESPN <-> CBBD <-> KenPom team ID crosswalk builder.

Scope: **D-I-ever teams only** -- a team is included if it is D-I (per
`cbb_sim.data.universe.compute_d1_team_seasons`, >=5 games with a non-null
conference_id in some season) in at least one season 2022-2026. Non-D-I
opponents (buy games, exhibitions, D-II/III schools) are excluded; CBBD and
KenPom do not meaningfully cover them and matching effort there would mostly
produce noise.

Matching methodology
---------------------
- **ESPN <-> CBBD**: via the *game* join, not name similarity or date/score
  fuzzy-matching. CBBD's `sourceId` field on `/games` is the ESPN/hoopR
  game_id verbatim (verified in `cbb_sim.data.universe`: ~100% row overlap
  and >=99.9% score agreement every season). For every score-verified game
  (both home and away score match between hoopR and CBBD) we take the
  (espn_team_id, cbbd_team_id) pair from both sides, tally votes across all
  seasons, and assign each espn_team_id its modal cbbd_id. This resolved
  367/367 D-I-ever teams in the 2026-09-10 build; the handful of dissenting
  votes per team come from postponed/rescheduled doubleheaders that happen to
  share a final score with another game on the same join key.

- **ESPN <-> KenPom**: primarily via last year's crosswalk file
  (`espn_to_kp_from_massy_matches.csv`, built for CBB-Monte via a
  Massey-Ratings-id bridge), keyed on `espn_id` -> `kp_name`. This alone
  covers 362/367 D-I-ever teams (2026-09-10 build). Teams missing from that
  file (new D-I reclassifications after last year's file was built, or teams
  that stopped playing D-I ball) fall back, in order: (1) a small manual
  alias table for known KenPom-only naming quirks (e.g. ESPN "St. Francis
  Brooklyn" is KenPom's "St. Francis NY" -- same institution, two campus
  names); (2) an exact match of the normalized ESPN team *location* against
  the normalized KenPom team-name universe observed across every snapshot
  file on disk; (3) a substring match, only when exactly one KenPom name
  candidate contains (or is contained by) the normalized location.
  Anything still unresolved is written to
  `data/reference/team_crosswalk_unmatched.csv` rather than guessed at.

This module does not create a separate name-alias table: per-season ESPN
display-name history lives inside the crosswalk row itself
(`espn_names_by_season`, a small JSON blob), not a fourth table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_MASSY_KP_CROSSWALK = Path(r"C:\Users\devuser\CBB-Monte\massy_to_kp.csv")
DEFAULT_ESPN_TO_KP_MATCHES = Path(r"C:\Users\devuser\CBB-Monte-storage\data\processed\espn_to_kp_from_massy_matches.csv")
DEFAULT_KENPOM_WEEKLY_DIR = Path(r"C:\Users\devuser\CBB-Monte")
DEFAULT_KENPOM_WEEKLY_SEASONS = [2022, 2023, 2024, 2025]
DEFAULT_KENPOM_DAILY_DIR = Path(r"C:\Users\devuser\CBB-Monte-storage\data\kenpom\2026")

# Manual aliases: normalized ESPN team *location* -> KenPom `Team` name, for
# cases the automated matchers cannot bridge (verified by hand, 2026-09-10).
MANUAL_KENPOM_ALIASES = {
    "st francis brooklyn": "St. Francis NY",
}


def _to_int64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _norm(s) -> str:
    s = str(s).lower()
    s = s.replace("&", "and")
    s = re.sub(r"[.'()]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


# --------------------------------------------------------------------------
# ESPN name history
# --------------------------------------------------------------------------
def build_espn_team_names(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, espn_team_id) appearance: display_name, location.
    Used both for first/last season bounds and as the ESPN-side name pool for
    the KenPom fallback matcher."""
    home = schedules[["season", "home_id", "home_display_name", "home_location"]].rename(
        columns={"home_id": "team_id", "home_display_name": "display_name", "home_location": "location"}
    )
    away = schedules[["season", "away_id", "away_display_name", "away_location"]].rename(
        columns={"away_id": "team_id", "away_display_name": "display_name", "away_location": "location"}
    )
    out = pd.concat([home, away], ignore_index=True).dropna(subset=["team_id"])
    out["team_id"] = out["team_id"].astype("int64")
    return out.drop_duplicates(subset=["season", "team_id"])


# --------------------------------------------------------------------------
# ESPN <-> CBBD, via the score-verified game join
# --------------------------------------------------------------------------
def match_espn_to_cbbd(schedules: pd.DataFrame, cbbd_dir: Path) -> pd.DataFrame:
    """Modal-vote (espn_id -> cbbd_id, cbbd_name) over score-verified games.
    Returns columns: espn_id, cbbd_id, cbbd_name, n_votes, n_top_votes."""
    frames = []
    for season in sorted(int(s) for s in schedules["season"].unique()):
        path = Path(cbbd_dir) / f"games_{season}.parquet"
        if not path.exists():
            continue
        cbbd = pd.read_parquet(
            path, columns=["sourceId", "homeTeamId", "awayTeamId", "homeTeam", "awayTeam", "homePoints", "awayPoints"]
        )
        cbbd["sourceId"] = _to_int64(cbbd["sourceId"])
        sub = schedules[schedules["season"] == season][["game_id", "home_id", "away_id", "home_score", "away_score"]]
        merged = sub.merge(cbbd, left_on="game_id", right_on="sourceId", how="inner")
        ok = (merged["home_score"].astype(float) == merged["homePoints"].astype(float)) & (
            merged["away_score"].astype(float) == merged["awayPoints"].astype(float)
        )
        merged = merged[ok]
        home_pairs = merged[["home_id", "homeTeamId", "homeTeam"]].rename(
            columns={"home_id": "espn_id", "homeTeamId": "cbbd_id", "homeTeam": "cbbd_name"}
        )
        away_pairs = merged[["away_id", "awayTeamId", "awayTeam"]].rename(
            columns={"away_id": "espn_id", "awayTeamId": "cbbd_id", "awayTeam": "cbbd_name"}
        )
        frames.append(home_pairs)
        frames.append(away_pairs)

    if not frames:
        return pd.DataFrame(columns=["espn_id", "cbbd_id", "cbbd_name", "n_votes", "n_top_votes"])

    allpairs = pd.concat(frames, ignore_index=True)
    allpairs["espn_id"] = allpairs["espn_id"].astype("int64")
    allpairs["cbbd_id"] = _to_int64(allpairs["cbbd_id"])
    allpairs = allpairs.dropna(subset=["cbbd_id"])

    rows = []
    for espn_id, g in allpairs.groupby("espn_id"):
        vc = g["cbbd_id"].value_counts()
        top_id = vc.idxmax()
        top_name = g.loc[g["cbbd_id"] == top_id, "cbbd_name"].value_counts().idxmax()
        rows.append(
            {
                "espn_id": int(espn_id),
                "cbbd_id": int(top_id),
                "cbbd_name": top_name,
                "n_votes": int(len(g)),
                "n_top_votes": int((g["cbbd_id"] == top_id).sum()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# ESPN <-> KenPom
# --------------------------------------------------------------------------
def load_kenpom_team_names(
    weekly_seasons: list[int] = DEFAULT_KENPOM_WEEKLY_SEASONS,
    weekly_dir: Path = DEFAULT_KENPOM_WEEKLY_DIR,
    daily_dir: Path = DEFAULT_KENPOM_DAILY_DIR,
) -> set[str]:
    """Every distinct `Team` value seen across the weekly (2022-2025) and
    daily (2026) KenPom snapshot files on disk."""
    names: set[str] = set()
    weekly_dir = Path(weekly_dir)
    for season in weekly_seasons:
        path = weekly_dir / f"{season}_kenpom.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["Team"])
            names.update(df["Team"].dropna().unique())
    daily_dir = Path(daily_dir)
    if daily_dir.exists():
        for f in sorted(daily_dir.glob("*_kenpom.csv")):
            df = pd.read_csv(f, usecols=["Team"])
            names.update(df["Team"].dropna().unique())
    return names


def match_espn_to_kenpom(
    espn_ids: list[int],
    espn_names: pd.DataFrame,
    massy_matches_path: Path = DEFAULT_ESPN_TO_KP_MATCHES,
    kenpom_names: set[str] | None = None,
) -> pd.DataFrame:
    """Return one row per espn_id: kenpom_name, match_method, match_confidence.
    See module docstring for the fallback order."""
    matches_df = (
        pd.read_csv(massy_matches_path) if Path(massy_matches_path).exists() else pd.DataFrame(columns=["espn_id", "kp_name"])
    )
    if not matches_df.empty:
        matches_df["espn_id"] = pd.to_numeric(matches_df["espn_id"], errors="coerce").astype("Int64")
    direct = dict(zip(matches_df["espn_id"].dropna().astype(int), matches_df["kp_name"], strict=False))

    if kenpom_names is None:
        kenpom_names = load_kenpom_team_names()
    norm_map: dict[str, str] = {}
    for n in kenpom_names:
        norm_map.setdefault(_norm(n), n)

    loc_by_id = espn_names.sort_values("season").groupby("team_id")["location"].last()

    rows = []
    for tid in sorted(espn_ids):
        if tid in direct:
            rows.append(
                {"espn_id": tid, "kenpom_name": direct[tid], "match_method": "massy_crosswalk", "match_confidence": 1.0}
            )
            continue

        loc = loc_by_id.get(tid, "")
        n = _norm(loc)

        if n in MANUAL_KENPOM_ALIASES:
            rows.append(
                {
                    "espn_id": tid,
                    "kenpom_name": MANUAL_KENPOM_ALIASES[n],
                    "match_method": "manual_alias",
                    "match_confidence": 0.9,
                }
            )
            continue

        if n in norm_map:
            rows.append(
                {"espn_id": tid, "kenpom_name": norm_map[n], "match_method": "normalized_name", "match_confidence": 0.9}
            )
            continue

        cands = sorted({orig for key, orig in norm_map.items() if key and (n in key or key in n)})
        if len(cands) == 1:
            rows.append(
                {"espn_id": tid, "kenpom_name": cands[0], "match_method": "substring_name", "match_confidence": 0.6}
            )
            continue

        rows.append({"espn_id": tid, "kenpom_name": None, "match_method": "unmatched", "match_confidence": 0.0})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main builder
# --------------------------------------------------------------------------
def build_team_crosswalk(
    schedules: pd.DataFrame,
    d1_team_ids: set[int],
    cbbd_dir: Path,
    massy_matches_path: Path = DEFAULT_ESPN_TO_KP_MATCHES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the team_crosswalk table (D-I-ever teams only) plus an
    unmatched-teams report. Returns (crosswalk_df, unmatched_df, report)."""
    espn_names = build_espn_team_names(schedules)
    espn_names_d1 = espn_names[espn_names["team_id"].isin(d1_team_ids)]

    bounds = espn_names_d1.groupby("team_id")["season"].agg(first_season="min", last_season="max")

    names_by_season = (
        espn_names_d1.sort_values(["team_id", "season"])
        .groupby("team_id")
        .apply(lambda g: json.dumps({int(s): n for s, n in zip(g["season"], g["display_name"], strict=False)}))
        .rename("espn_names_by_season")
    )
    latest_name = espn_names_d1.sort_values("season").groupby("team_id")["display_name"].last().rename("espn_name")

    cbbd_match = match_espn_to_cbbd(schedules, cbbd_dir).set_index("espn_id")
    kp_match = match_espn_to_kenpom(sorted(d1_team_ids), espn_names_d1, massy_matches_path=massy_matches_path).set_index(
        "espn_id"
    )

    out = pd.DataFrame(index=sorted(d1_team_ids))
    out.index.name = "espn_team_id"
    out = out.join(latest_name).join(names_by_season).join(bounds)
    out = out.join(cbbd_match[["cbbd_id", "cbbd_name"]].rename(columns={"cbbd_id": "cbbd_team_id"}))
    out = out.join(kp_match[["kenpom_name", "match_method", "match_confidence"]])
    out = out.reset_index()

    unmatched = out[out["cbbd_team_id"].isna() | out["kenpom_name"].isna()].copy()
    unmatched["missing_cbbd"] = unmatched["cbbd_team_id"].isna()
    unmatched["missing_kenpom"] = unmatched["kenpom_name"].isna()

    report = {
        "n_teams": len(out),
        "n_cbbd_matched": int(out["cbbd_team_id"].notna().sum()),
        "n_kenpom_matched": int(out["kenpom_name"].notna().sum()),
        "kenpom_match_method_counts": out["match_method"].value_counts(dropna=False).to_dict(),
        "n_unmatched_any": len(unmatched),
    }
    return out, unmatched, report
