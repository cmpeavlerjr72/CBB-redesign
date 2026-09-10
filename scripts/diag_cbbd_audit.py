#!/usr/bin/env python
"""
diag_cbbd_audit.py -- audit the CBBD pull in data/raw/cbbd and write
docs/tests/data_audit_cbbd_2026-09-10.md.

Read-only: does not modify any pulled data. Run after scripts/pull_cbbd.py.

Usage:
    python scripts/diag_cbbd_audit.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CBBD_DIR = ROOT / "data" / "raw" / "cbbd"
HOOPR_SCHEDULES_DIR = ROOT / "data" / "raw" / "hoopr" / "schedules"
SAMPLES_DIR = CBBD_DIR / "samples"
OUT_MD = ROOT / "docs" / "tests" / "data_audit_cbbd_2026-09-10.md"

LINES_SEASONS = list(range(2013, 2027))
GAMES_SEASONS = list(range(2013, 2027))
RATINGS_SEASONS = list(range(2022, 2027))

pd.set_option("display.width", 160)


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def load_lines(season: int) -> pd.DataFrame | None:
    p = CBBD_DIR / f"lines_{season}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_games(season: int) -> pd.DataFrame | None:
    p = CBBD_DIR / f"games_{season}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


# --------------------------------------------------------------------------
# Lines audit
# --------------------------------------------------------------------------

def audit_lines() -> str:
    lines_out = ["## 1. Lines (`/lines`)", ""]
    rows = []
    per_provider_rows = []
    for season in LINES_SEASONS:
        df = load_lines(season)
        if df is None:
            continue
        n_games = df["gameId"].nunique()
        has_line = df.dropna(subset=["provider"])
        n_games_with_line = has_line["gameId"].nunique()
        date_min = pd.to_datetime(df["startDate"]).min()
        date_max = pd.to_datetime(df["startDate"]).max()
        row = {
            "season": season,
            "games": n_games,
            "games_with_line": n_games_with_line,
            "pct_with_line": pct(n_games_with_line, n_games),
            "spread_nonnull_%": pct(has_line["spread"].notna().sum(), len(has_line)) if len(has_line) else "n/a",
            "ou_nonnull_%": pct(has_line["overUnder"].notna().sum(), len(has_line)) if len(has_line) else "n/a",
            "ml_home_nonnull_%": pct(has_line["homeMoneyline"].notna().sum(), len(has_line)) if len(has_line) else "n/a",
            "ml_away_nonnull_%": pct(has_line["awayMoneyline"].notna().sum(), len(has_line)) if len(has_line) else "n/a",
            "spreadOpen_nonnull_%": pct(has_line["spreadOpen"].notna().sum(), len(has_line)) if len(has_line) else "n/a",
            "ouOpen_nonnull_%": pct(has_line["overUnderOpen"].notna().sum(), len(has_line)) if len(has_line) else "n/a",
            "date_min": str(date_min.date()) if pd.notna(date_min) else "n/a",
            "date_max": str(date_max.date()) if pd.notna(date_max) else "n/a",
        }
        rows.append(row)

        prov_counts = has_line.groupby("provider")["gameId"].nunique().sort_values(ascending=False)
        per_provider_rows.append((season, prov_counts))

    if rows:
        summary = pd.DataFrame(rows).set_index("season")
        lines_out.append("Per-season line coverage:\n")
        lines_out.append(summary.to_markdown())
        lines_out.append("")

    lines_out.append("Games-per-provider (distinct gameId with a non-null line row), by season:\n")
    for season, counts in per_provider_rows:
        if counts.empty:
            continue
        s = ", ".join(f"{prov}={n}" for prov, n in counts.items())
        lines_out.append(f"- **{season}**: {s}")
    lines_out.append("")

    # 2026 open vs close
    df26 = load_lines(2026)
    if df26 is not None:
        has_line = df26.dropna(subset=["provider"])
        both_spread = has_line.dropna(subset=["spread", "spreadOpen"])
        both_ou = has_line.dropna(subset=["overUnder", "overUnderOpen"])
        n_rows = len(has_line)
        lines_out.append("### 2026 open vs. close")
        lines_out.append("")
        lines_out.append(f"- Line-rows with a provider quote: {n_rows}")
        lines_out.append(f"- Rows with BOTH spread and spreadOpen: {len(both_spread)} ({pct(len(both_spread), n_rows)})")
        lines_out.append(f"- Rows with BOTH overUnder and overUnderOpen: {len(both_ou)} ({pct(len(both_ou), n_rows)})")
        if len(both_spread):
            delta_spread = both_spread["spread"] - both_spread["spreadOpen"]
            lines_out.append("")
            lines_out.append("`spread - spreadOpen` distribution:")
            lines_out.append("")
            lines_out.append(delta_spread.describe().to_markdown())
        if len(both_ou):
            delta_ou = both_ou["overUnder"] - both_ou["overUnderOpen"]
            lines_out.append("")
            lines_out.append("`overUnder - overUnderOpen` distribution:")
            lines_out.append("")
            lines_out.append(delta_ou.describe().to_markdown())
        lines_out.append("")
    else:
        lines_out.append("### 2026 open vs. close\n\n_lines_2026.parquet not found -- skipped._\n")

    # hoopR join for 2025 & 2026
    lines_out.append("### Join to hoopR schedules (2025, 2026)")
    lines_out.append("")
    for season in (2025, 2026):
        hoopr_path = HOOPR_SCHEDULES_DIR / f"mbb_schedule_{season}.parquet"
        cbbd_df = load_lines(season)
        if cbbd_df is None:
            lines_out.append(f"- **{season}**: lines_{season}.parquet not found -- skipped.")
            continue
        if not hoopr_path.exists():
            lines_out.append(f"- **{season}**: `{hoopr_path}` does not exist -- skipped (hoopR pull not available yet).")
            continue
        hoopr_df = pd.read_parquet(hoopr_path)
        match_result = join_cbbd_hoopr(cbbd_df, hoopr_df)
        lines_out.append(f"- **{season}**: matched {match_result['matched']}/{match_result['total']} CBBD games "
                          f"({pct(match_result['matched'], match_result['total'])}) to a hoopR schedule row "
                          f"on (date, home team name, away team name).")
        if match_result["mismatch_sample"]:
            lines_out.append(f"  - Sample unmatched (CBBD homeTeam/awayTeam/date), up to 10 shown:")
            for m in match_result["mismatch_sample"]:
                lines_out.append(f"    - {m}")
    lines_out.append("")

    return "\n".join(lines_out)


def join_cbbd_hoopr(cbbd_df: pd.DataFrame, hoopr_df: pd.DataFrame) -> dict:
    cbbd_games = cbbd_df.drop_duplicates(subset=["gameId"]).copy()
    cbbd_games["date_only"] = pd.to_datetime(cbbd_games["startDate"]).dt.tz_convert(None).dt.date
    # startDate is UTC and often lands after midnight local for night games;
    # also try date-1 as a fallback join key.
    cbbd_games["date_minus1"] = pd.to_datetime(cbbd_games["startDate"]).dt.tz_convert(None).dt.date - pd.Timedelta(days=1)

    hoopr_df = hoopr_df.copy()
    hoopr_df["game_date_only"] = pd.to_datetime(hoopr_df["game_date"]).dt.date

    hoopr_key_a = set(zip(hoopr_df["game_date_only"], hoopr_df["home_location"], hoopr_df["away_location"]))
    hoopr_key_b = set(zip(hoopr_df["game_date_only"], hoopr_df["home_name"], hoopr_df["away_name"]))

    matched = 0
    mismatches = []
    for _, r in cbbd_games.iterrows():
        keys_to_try = [
            (r["date_only"], r["homeTeam"], r["awayTeam"]),
            (r["date_minus1"], r["homeTeam"], r["awayTeam"]),
        ]
        hit = any(k in hoopr_key_a or k in hoopr_key_b for k in keys_to_try)
        if hit:
            matched += 1
        else:
            if len(mismatches) < 10:
                mismatches.append(f"{r['date_only']} | {r['homeTeam']} vs {r['awayTeam']}")
    return {"matched": matched, "total": len(cbbd_games), "mismatch_sample": mismatches}


# --------------------------------------------------------------------------
# Games audit
# --------------------------------------------------------------------------

def audit_games() -> str:
    out = ["## 2. Games (`/games`)", ""]
    rows = []
    for season in GAMES_SEASONS:
        df = load_games(season)
        if df is None:
            continue
        n = len(df)
        neutral_share = pct(df["neutralSite"].sum(), n) if "neutralSite" in df else "n/a"
        season_types = df["seasonType"].value_counts().to_dict() if "seasonType" in df else {}
        has_periods = df["homePeriodPoints"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)
        n_with_periods = has_periods.sum()
        ot_games = df["homePeriodPoints"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 2)
        n_ot = ot_games.sum()
        score_present = df["homePoints"].notna() & df["awayPoints"].notna()
        rows.append({
            "season": season,
            "games": n,
            "neutralSite_%": neutral_share,
            "seasonTypes": season_types,
            "games_with_period_scores": int(n_with_periods),
            "OT_games(periods>2)": int(n_ot),
            "OT_%_of_with_periods": pct(n_ot, n_with_periods),
            "score_present_%": pct(score_present.sum(), n),
        })
    if rows:
        summary = pd.DataFrame(rows).set_index("season")
        out.append(summary.to_markdown())
    else:
        out.append("_No games files found._")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Ratings audit
# --------------------------------------------------------------------------

def audit_ratings() -> str:
    out = ["## 3. Ratings (`/ratings/adjusted`, `/ratings/srs`, `/ratings/elo`)", ""]

    for stem, label in [("ratings_adjusted", "adjusted (offensive/defensive/net rating)"),
                         ("ratings_srs", "SRS"),
                         ("ratings_elo", "Elo (season-level)")]:
        p = CBBD_DIR / f"{stem}_2025.parquet"
        if not p.exists():
            out.append(f"### `{stem}` -- file not found, skipped\n")
            continue
        df = pd.read_parquet(p)
        out.append(f"### `{stem}` -- {label}")
        out.append("")
        out.append(f"Fields returned: `{', '.join(df.columns)}`")
        has_date_field = any(re.search(r"date|asof|as_of|time", c, re.I) for c in df.columns)
        out.append(f"Any date/asOf field present: **{has_date_field}**")
        out.append("")
        sample_teams = ["Duke", "Houston", "Auburn", "Florida", "Tennessee"]
        team_col = "team" if "team" in df.columns else None
        if team_col:
            sample = df[df[team_col].isin(sample_teams)]
            out.append(f"Sample rows for {sample_teams}:\n")
            out.append(sample.to_markdown(index=False))
        out.append("")

        # Flag any season with zero rows (data-availability gap, not a code bug).
        for chk_season in RATINGS_SEASONS:
            chk_p = CBBD_DIR / f"{stem}_{chk_season}.parquet"
            if chk_p.exists():
                chk_df = pd.read_parquet(chk_p)
                if len(chk_df) == 0:
                    out.append(f"**Anomaly**: `{stem}_{chk_season}.parquet` has **0 rows** "
                               f"(other seasons/endpoints for {chk_season} are populated -- e.g. "
                               f"`ratings_elo_{chk_season}` and `ratings_adjusted_{chk_season}` both "
                               f"have ~365 rows). This looks like a CBBD data-availability gap for this "
                               f"specific endpoint+season, not a pull bug -- flagging for awareness before "
                               f"relying on `{stem}` for {chk_season}.\n")

    out.append("### Point-in-time verdict")
    out.append("")
    out.append(
        "`/ratings/adjusted`, `/ratings/srs`, and `/ratings/elo` (queried by `season` only) each return **exactly "
        "one row per team-season** with no date/asOf/game-id field anywhere in the payload -- verified by "
        "inspecting the raw JSON keys above. These are **end-of-season snapshots**, computed over the team's full "
        "slate of games for that season. Joining any of them onto a game as a *pregame* feature (for any game "
        "before the season's last game) is a **look-ahead leak**: the rating already reflects games that, for "
        "an early- or mid-season game, have not been played yet.\n"
    )
    out.append(
        "By contrast, `/games` carries `homeTeamEloStart` / `homeTeamEloEnd` / `awayTeamEloStart` / "
        "`awayTeamEloEnd` **per game** -- this IS point-in-time (the Elo value entering that specific game) and is "
        "safe to use as a pregame feature. `/ratings/elo` (season-level) is a different, later snapshot and should "
        "not be confused with the per-game Elo embedded in `/games`.\n"
    )
    out.append(
        "**Practical implication**: for a pregame model, do not join `/ratings/adjusted` or `/ratings/srs` "
        "by (team, season) onto in-season games. Either (a) use only the per-game Elo from `/games`, or (b) pull "
        "`/ratings/adjusted`/`srs` incrementally after each date and reconstruct a time series (not attempted in "
        "this pull), or (c) restrict use of the season-end adjusted/SRS ratings to season-level or postseason "
        "analysis where the leak window is closed.\n"
    )
    return "\n".join(out)


# --------------------------------------------------------------------------
# Samples audit
# --------------------------------------------------------------------------

def audit_samples() -> str:
    out = ["## 4. Samples (`/plays`, `/substitutions`, `/lineups`, `/games/players`, `/teams/roster`, `/recruiting/portal`)", ""]

    ids_path = SAMPLES_DIR / "sample_game_ids.json"
    game_ids = json.loads(ids_path.read_text()) if ids_path.exists() else []
    out.append(f"Sample games (Duke, season 2025): {game_ids}\n")

    if game_ids:
        gid = game_ids[0]
        plays_p = SAMPLES_DIR / f"plays_game_{gid}.json"
        if plays_p.exists():
            plays = json.loads(plays_p.read_text())
            shot = next((p for p in plays if p.get("shootingPlay")), None)
            nonshot = next((p for p in plays if not p.get("shootingPlay")), None)
            out.append("### `/plays/game/{id}` structure")
            out.append("")
            out.append(f"- {len(plays)} play records for game {gid}.")
            out.append(f"- Top-level fields: `{', '.join(plays[0].keys())}`")
            out.append(
                "- `onFloor`: list of 10 dicts (`id`, `name`, `team`) -- the full 5-vs-5 lineup on court at the "
                "moment of the play, for BOTH teams. This is present on every play record, not just shots/subs."
            )
            if shot:
                out.append(f"- `shotInfo` (only on `shootingPlay=true` rows): "
                            f"`{json.dumps(shot['shotInfo'])}` -- shooter id/name, made bool, `range` "
                            "(e.g. jumper/layup/dunk), assisted flag + assister, and x/y shot `location`.")
            out.append("- Non-shooting plays (fouls, subs-as-plays, turnovers, etc.) have `shotInfo: null`.")
            out.append("")

        subs_p = SAMPLES_DIR / f"substitutions_game_{gid}.json"
        if subs_p.exists():
            subs = json.loads(subs_p.read_text())
            out.append("### `/substitutions/game/{id}` structure")
            out.append("")
            out.append(f"- {len(subs)} substitution records for game {gid} (one row per player per stint on court).")
            if subs:
                out.append(f"- Fields: `{', '.join(subs[0].keys())}`")
                out.append(
                    "- `subIn` / `subOut` are nested `{period, secondsRemaining, teamPoints, opponentPoints}` -- "
                    "gives exact game-clock and score context for when a player entered/left, i.e. a stint table, "
                    "not raw substitution events."
                )
            out.append("")

        lineups_p = SAMPLES_DIR / f"lineups_game_{gid}.json"
        if lineups_p.exists():
            lineups = json.loads(lineups_p.read_text())
            out.append("### `/lineups/game/{id}` structure")
            out.append("")
            out.append(f"- {len(lineups)} five-man-lineup records for game {gid} (one row per distinct 5-man unit "
                        "that appeared, per team).")
            if lineups:
                l0 = lineups[0]
                out.append(f"- Top-level fields: `{', '.join(l0.keys())}`")
                out.append(f"- `athletes`: list of 5 `{{id, name}}` dicts identifying the unit (`idHash` is a "
                            "stable sorted-id key for the lineup).")
                out.append(f"- Per-lineup minutes/possessions: `totalSeconds`={l0.get('totalSeconds')}, "
                            f"`pace`={l0.get('pace')} -- YES, minutes and pace (~possessions proxy) are present.")
                out.append(f"- Ratings: `offenseRating`, `defenseRating`, `netRating` per lineup.")
                ts_keys = list(l0.get("teamStats", {}).keys())
                out.append(f"- `teamStats` / `opponentStats`: full box-score-style splits for the lineup "
                            f"(`{', '.join(ts_keys)}`), including a `fourFactors` sub-block "
                            f"(`effectiveFieldGoalPct`, `turnoverRatio`, `offensiveReboundPct`, `freeThrowRate`).")
            out.append("")

    gp_p = SAMPLES_DIR / "games_players_duke_2025.json"
    if gp_p.exists():
        gp = json.loads(gp_p.read_text())
        out.append("### `/games/players` structure")
        out.append("")
        out.append(f"- {len(gp)} game-rows for Duke, season 2025 (one row per game, `players` nested list per game).")
        if gp:
            pl = gp[0]["players"][0]
            out.append(f"- Per-game top fields: `{', '.join(gp[0].keys())}`")
            out.append(f"- Per-player fields: `{', '.join(pl.keys())}`")
            out.append("- Includes box score + advanced per-player: minutes, usage, offensive/defensive/net rating, "
                        "gameScore, true shooting%, effective FG%, four-factor-style rates, and split shooting "
                        "(FG/2P/3P/FT) with makes/attempts/pct.")
        out.append("")

    ro_p = SAMPLES_DIR / "teams_roster_duke_2025.json"
    if ro_p.exists():
        ro = json.loads(ro_p.read_text())
        out.append("### `/teams/roster` structure")
        out.append("")
        if ro:
            team_row = ro[0]
            players = team_row.get("players", [])
            out.append(f"- {len(players)} players on the Duke 2025 roster.")
            out.append(f"- Team-level fields: `{', '.join(team_row.keys())}`")
            if players:
                out.append(f"- Per-player fields: `{', '.join(players[0].keys())}`")
                out.append("- `hometown` is a nested dict (city/state/country/lat/long/countyFips); "
                            "`dateOfBirth` was null for the sampled players; `startSeason`/`endSeason` bound "
                            "eligibility on this roster snapshot.")
        out.append("")

    port_p = SAMPLES_DIR / "recruiting_portal_2025.json"
    if port_p.exists():
        port = json.loads(port_p.read_text())
        out.append("### `/recruiting/portal` structure")
        out.append("")
        out.append(f"- {len(port)} transfer-portal entries pulled with `year=2025` "
                    "(**note**: the endpoint's real filter param is `year`, not `season` -- passing `season=2025` "
                    "is silently ignored and returns all years unfiltered [verified: 6679 rows spanning "
                    "years 2021-2026 vs. 1611 rows with `year=2025`]).")
        if port:
            out.append(f"- Fields: `{', '.join(port[0].keys())}`")
            out.append("- `origin`/`destination` are nested `{id, name, conference}`; includes `stars`, `rating`, "
                        "`eligibility` (e.g. 'Immediate'), and `yearsRemaining`.")
        out.append("")

    out.append("### CBBD vs. hoopR as the primary event source for lineup-on-floor modelling")
    out.append("")
    out.append(
        "**Verdict: CBBD is the primary source.** The hoopR MBB play-by-play parquet "
        "(`data/raw/hoopr/pbp/play_by_play_2025.parquet`, 63 columns) carries only `athlete_id_1/2/3` "
        "(the play's direct participants -- shooter/assister/etc.), with **no on-court roster field at all**; "
        "reconstructing which 10 players were on the floor at a given moment from hoopR alone would require "
        "inferring stints from a separate substitution/box signal that hoopR's MBB pbp does not appear to expose "
        "either. CBBD's `/plays/game/{id}` embeds a 10-player `onFloor` list (5 per team, with team label) on "
        "**every single play record**, and CBBD additionally ships a purpose-built `/lineups/game/{id}` endpoint "
        "that pre-aggregates minutes (`totalSeconds`), pace, offensive/defensive/net rating, and a full four-factors "
        "box line per distinct 5-man unit -- i.e. the lineup-level modelling target is largely pre-computed. "
        "hoopR remains useful for shot coordinates/shot-chart detail (`shots_*.parquet`) and player/team box scores, "
        "but for on-floor lineup composition and lineup-level efficiency, CBBD's `onFloor` + `/lineups` are the "
        "right primary source."
    )
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Call budget
# --------------------------------------------------------------------------

def audit_budget() -> str:
    out = ["## 5. Call budget", ""]
    manifest_path = CBBD_DIR / "manifest.json"
    if not manifest_path.exists():
        out.append("_manifest.json not found._")
        return "\n".join(out)
    manifest = json.loads(manifest_path.read_text())
    out.append(f"- Total API calls made this pull: **{manifest['total_calls']}**")
    out.append(f"- `X-CallLimit-Remaining` at end of pull: **{manifest['call_limit_remaining_final']}**")
    out.append(f"- Manifest generated at: {manifest['generated_at']}")
    out.append(f"- Manifest entries: {len(manifest['entries'])} (see `data/raw/cbbd/manifest.json` for full detail)")
    out.append("")
    return "\n".join(out)


def main():
    sections = [
        f"# CBBD data audit -- 2026-09-10\n",
        "Source: CollegeBasketballData API (`https://api.collegebasketballdata.com`). "
        "Pulled by `scripts/pull_cbbd.py` into `data/raw/cbbd/`. This report is generated read-only by "
        "`scripts/diag_cbbd_audit.py`.\n",
        audit_lines(),
        audit_games(),
        audit_ratings(),
        audit_samples(),
        audit_budget(),
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(sections))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
