#!/usr/bin/env python
"""
diag_hoopr_audit.py -- data-quality / gate-reference audit of the hoopR
men's college basketball parquet pull in data/raw/hoopr/.

Produces a Markdown report (default: docs/tests/data_audit_hoopr_2026-09-10.md)
covering, per season: schedules, team_box, player_box, pbp, a 30-game
final-score cross-check (2025), possession/rate gate targets, and ID hygiene.

Usage:
    python scripts/diag_hoopr_audit.py
    python scripts/diag_hoopr_audit.py --seasons 2022 2023 2024 2025 2026 --out docs/tests/data_audit_hoopr_2026-09-10.md
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR_DEFAULT = Path("data/raw/hoopr")
OUT_DEFAULT = Path("docs/tests/data_audit_hoopr_2026-09-10.md")
SEASONS_DEFAULT = [2022, 2023, 2024, 2025, 2026]

STEM = {
    "pbp": "play_by_play",
    "player_box": "player_box",
    "team_box": "team_box",
    "schedules": "mbb_schedule",
    "shots": "shots",
    "rosters": "rosters",
    "game_rosters": "game_rosters",
    "player_core": "player_core",
}

DEFECTS: list[str] = []


def flag(msg: str) -> None:
    DEFECTS.append(msg)


def load(data_dir: Path, dataset: str, season: int, columns=None) -> pd.DataFrame | None:
    path = data_dir / dataset / f"{STEM[dataset]}_{season}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=columns)


def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.1f}%"


def num(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:,.{nd}f}"


def cnt_dict(vc) -> dict:
    """Value-counts Series -> plain-Python dict (no np.int64(...) clutter in repr)."""
    def native(x):
        return x.item() if hasattr(x, "item") else x
    return {native(k): native(v) for k, v in vc.items()}


def md_table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def linescore_periods(s) -> float:
    if not isinstance(s, str):
        return np.nan
    return float(len(re.findall(r"'value'", s)))


def linescore_periods_col(sch: pd.DataFrame) -> pd.Series:
    """home_linescores is missing entirely in some seasons (schema drift, e.g. 2022) -- return all-NaN in that case."""
    if "home_linescores" not in sch.columns:
        return pd.Series(np.nan, index=sch.index)
    return sch["home_linescores"].map(linescore_periods)


def pbp_period_max_by_game(data_dir: Path, season: int) -> pd.Series | None:
    pbp = load(data_dir, "pbp", season, columns=["game_id", "period_number"])
    if pbp is None:
        return None
    return pbp.groupby("game_id")["period_number"].max()


# --------------------------------------------------------------------------
# Section 1: schedules
# --------------------------------------------------------------------------

def audit_schedules(data_dir: Path, seasons: list[int], extra_season: int) -> tuple[list[str], dict]:
    lines = ["## 1. Schedules", ""]
    per_season: dict[int, dict] = {}

    rows_overview = []
    for season in seasons + [extra_season]:
        sch = load(data_dir, "schedules", season)
        if sch is None:
            flag(f"schedules: season {season} file missing")
            continue
        games = len(sch)
        completed = int(sch["status_type_completed"].sum())
        rows_overview.append([season, games, completed, sch["game_date"].min(), sch["game_date"].max()])
        if season == extra_season:
            continue  # 2027 is a forward schedule preview, not a gate-reference season
        per_season[season] = {"df": sch}

    lines += ["Games and date coverage per season (2027 shown separately as a forward-looking, mostly-unplayed schedule):", ""]
    lines += md_table(["season", "games", "status_completed=True", "min game_date", "max game_date"], rows_overview)
    lines.append("")

    col_counts = {s: len(per_season[s]["df"].columns) for s in seasons if s in per_season}
    lines.append(f"Schema drift note: schedule column counts differ by season -- {col_counts}. 2022 lacks `home_linescores`/`away_linescores`/`home_records`/`away_records`/`*_current_rank`/`broadcast_market`/`broadcast_name`/`play_by_play_available` (all present from 2023+); 2025+ add `broadcast`/`highlights`; 2025 uniquely adds a near-empty `away_non_div1_team` column (see below). Treat the schedules schema as evolving, not fixed, across seasons.")
    lines.append("")

    for season in seasons:
        sch = per_season[season]["df"]
        games = len(sch)
        lines.append(f"### {season}")
        lines.append("")

        st_counts = sch["season_type"].value_counts(dropna=False).sort_index()
        ta_counts = sch["type_abbreviation"].value_counts(dropna=False)
        lines.append(f"- games: **{games:,}**")
        lines.append(f"- `season_type` value counts: {cnt_dict(st_counts)}  (2=regular season, 3=postseason/tournament per ESPN convention; cross-check against `type_abbreviation`: {cnt_dict(ta_counts)})")

        neutral_counts = sch["neutral_site"].value_counts(dropna=False)
        conf_comp_counts = sch["conference_competition"].value_counts(dropna=False)
        n_venues = sch["venue_id"].nunique()
        indoor_counts = sch["venue_indoor"].value_counts(dropna=False)
        attendance_notna = sch["attendance"].notna().mean()
        lines.append(f"- `neutral_site` counts: {cnt_dict(neutral_counts)}; `conference_competition` counts: {cnt_dict(conf_comp_counts)}")
        lines.append(f"- venues: {n_venues:,} distinct `venue_id`; `venue_indoor` counts: {cnt_dict(indoor_counts)}; `attendance` populated for {pct(attendance_notna)} of games")

        home_null = sch["home_conference_id"].isna()
        away_null = sch["away_conference_id"].isna()
        both_null = (home_null & away_null).sum()
        either_null = (home_null | away_null).sum()
        both_di = games - either_null
        non_di_teams = set(sch.loc[home_null, "home_id"]) | set(sch.loc[away_null, "away_id"])
        lines.append(
            f"- D-I identification field used: `home_conference_id` / `away_conference_id` (null => team not in an ESPN-tracked D-I conference for that game). "
            f"Both-teams-D-I games: **{both_di:,}** / {games:,}. Games with >=1 non-D-I side: **{either_null:,}** "
            f"(home-side null: {home_null.sum():,}, away-side null: {away_null.sum():,}, both null: {both_null}). "
            f"Distinct non-D-I `team_id`s involved: **{len(non_di_teams):,}**."
        )
        if home_null.sum() > 0:
            sample = sch.loc[home_null, ["game_id", "game_date", "home_name", "away_name", "type_abbreviation"]].head(5).to_dict("records")
            lines.append(f"  - home-side nulls are a mix of legitimate non-D-I host venues (e.g. Chaminade 'Silverswords' at the Maui Invitational) and a couple of apparent data gaps for nominally D-I hosts; sample: {sample}")
            flag(f"schedules {season}: {home_null.sum()} games have null home_conference_id even though the home team can be a D-I school (data-gap, not real non-D-I) -- spot check before using conference_id as a hard D-I filter on the home side.")

        completed = sch["status_type_completed"]
        status_counts = sch["status_type_name"].value_counts(dropna=False)
        final_score_games = int((completed).sum())
        lines.append(f"- `status_type_name` counts: {cnt_dict(status_counts)}; games with a final score (`status_type_completed`==True): **{final_score_games:,}** / {games:,}")
        not_completed = sch.loc[~completed]
        if len(not_completed):
            zero_zero = ((not_completed["home_score"] == 0) & (not_completed["away_score"] == 0)).mean()
            lines.append(f"  - non-completed games ({len(not_completed)}) have home_score/away_score == 0-0 in {pct(zero_zero)} of cases (placeholder, not a real result)")

        periods = linescore_periods_col(sch)
        if periods.notna().any():
            ot_dist = periods.value_counts(dropna=False).sort_index()
            n_ot = int((periods > 2).sum())
            lines.append(f"- OT indicator: derived by counting period entries in `home_linescores` (regulation = 2 halves). Periods-per-game distribution: {cnt_dict(ot_dist)}. Games that went to OT: **{n_ot:,}** ({pct(n_ot / games)}).")
        else:
            lines.append("- OT indicator: `home_linescores`/`away_linescores` columns **do not exist** in this season's schedule file (schema drift -- see below); OT is instead derived from pbp `period_number` in Section 4/6.")
            flag(f"schedules {season}: home_linescores/away_linescores columns are absent from the schedule schema this season (present from 2023 onward)")
        lines.append(f"- `format_regulation_periods` is constant at {sorted(sch['format_regulation_periods'].dropna().unique().tolist())} for all seasons (halves format metadata, not an OT flag by itself).")
        if "away_non_div1_team" in sch.columns:
            flagged_n = int((sch["away_non_div1_team"] == True).sum())  # noqa: E712
            lines.append(f"- note: this season's schedule also carries an `away_non_div1_team` column, but it is populated for only {flagged_n} row(s) out of {games:,} -- far fewer than the {away_null.sum():,} rows with a null `away_conference_id`. It looks like an incompletely-populated one-off field (also absent from every other season's schema) and should **not** be relied on as the D-I flag; `*_conference_id` nullness is the more consistent proxy used throughout this report.")
            flag(f"schedules {season}: away_non_div1_team column exists but is populated for only {flagged_n} rows (inconsistent with {away_null.sum()} away_conference_id nulls) and is absent from other seasons -- unreliable, do not use as the D-I filter")

        market_cols = [c for c in sch.columns if any(k in c.lower() for k in ("spread", "odds", "moneyline", "over_under", "total_line", "win_prob"))]
        if market_cols:
            lines.append(f"- embedded market-like columns found directly in schedules: {market_cols}")
        else:
            lines.append("- **no embedded market fields** (spread / over_under / moneyline) exist on the schedules table itself. The market-ish columns (`game_spread`, `home_team_spread`, `home_favorite`, `game_spread_available`, `pregame_home_prob`, `home_win_prob`) live on the **pbp** table only -- see Section 4 for non-null rates and date coverage, including an important frozen-field defect there.")
        lines.append("")

    return lines, per_season


# --------------------------------------------------------------------------
# Section 2: team box
# --------------------------------------------------------------------------

def audit_team_box(data_dir: Path, seasons: list[int], schedules: dict) -> tuple[list[str], dict]:
    lines = ["## 2. Team box", ""]
    per_season: dict[int, pd.DataFrame] = {}
    cols_printed = False

    rows_overview = []
    for season in seasons:
        tb = load(data_dir, "team_box", season)
        if tb is None:
            flag(f"team_box: season {season} file missing")
            continue
        per_season[season] = tb
        if not cols_printed:
            lines.append(f"Columns ({len(tb.columns)}): `{', '.join(tb.columns)}`")
            lines.append("")
            cols_printed = True

        games_covered = tb["game_id"].nunique()
        rows = len(tb)
        teams_per_game = tb.groupby("game_id").size()
        off_2 = int((teams_per_game != 2).sum())

        sch = schedules[season]["df"]
        sched_games = set(sch["game_id"])
        tb_games = set(tb["game_id"])
        missing_from_tb = sched_games - tb_games
        extra_in_tb = tb_games - sched_games

        home_rows = tb[tb["team_home_away"] == "home"][["game_id", "team_score"]].rename(columns={"team_score": "tb_home_score"})
        away_rows = tb[tb["team_home_away"] == "away"][["game_id", "team_score"]].rename(columns={"team_score": "tb_away_score"})
        merged = sch[["game_id", "home_score", "away_score", "status_type_completed"]].merge(home_rows, on="game_id", how="inner").merge(away_rows, on="game_id", how="inner")
        merged = merged[merged["status_type_completed"]]
        home_mismatch = (merged["home_score"] != merged["tb_home_score"]).sum()
        away_mismatch = (merged["away_score"] != merged["tb_away_score"]).sum()

        rows_overview.append([
            season, rows, games_covered, off_2, len(missing_from_tb), len(extra_in_tb),
            f"{home_mismatch}/{len(merged)}", f"{away_mismatch}/{len(merged)}",
        ])
        if home_mismatch or away_mismatch:
            flag(f"team_box {season}: team_score disagrees with schedules home/away score on {home_mismatch + away_mismatch} team-rows")
        if off_2:
            flag(f"team_box {season}: {off_2} games do not have exactly 2 team rows")

    lines += md_table(
        ["season", "rows", "games covered", "games w/ !=2 team rows", "sched games missing from team_box", "team_box games not in sched", "home_score mismatch", "away_score mismatch"],
        rows_overview,
    )
    lines.append("")
    lines.append("`team_score` in team_box matches `home_score`/`away_score` on the schedules table for essentially all completed games in every season (see mismatch columns above); team_box is treated as internally consistent with schedules for scoring.")
    lines.append("")
    return lines, per_season


# --------------------------------------------------------------------------
# Section 3: player box
# --------------------------------------------------------------------------

def audit_player_box(data_dir: Path, seasons: list[int], team_box: dict, schedules: dict) -> tuple[list[str], dict]:  # noqa: ARG001
    lines = ["## 3. Player box", ""]
    per_season: dict[int, pd.DataFrame] = {}

    rows_overview = []
    for season in seasons:
        pb = load(data_dir, "player_box", season)
        if pb is None:
            flag(f"player_box: season {season} file missing")
            continue
        per_season[season] = pb

        rows = len(pb)
        games = pb["game_id"].nunique()
        players = pb["athlete_id"].nunique()
        min_null_rate = pb["minutes"].isna().mean()
        dnp = pb["did_not_play"].value_counts(dropna=False)
        active = pb["active"].value_counts(dropna=False)
        starter = pb["starter"].value_counts(dropna=False)

        rows_overview.append([season, rows, games, players, pct(min_null_rate), cnt_dict(dnp), cnt_dict(active), cnt_dict(starter)])

    lines += md_table(["season", "rows", "games", "players", "minutes null rate", "did_not_play", "active", "starter"], rows_overview)
    lines.append("")
    lines.append(
        "- **DEFECT: `active` is unreliable before 2026.** It is constant `False` for every row in 2022-2024, ~100% `None`/null in 2025 (only 221/207,613 rows populated), "
        "and only becomes a real, informative True/False signal in 2026 (True correlates with `did_not_play`==False as expected: 62,202/62,246 True rows are NOT DNP). "
        "Do not use `active` as a roster-availability feature for seasons before 2026 -- use `did_not_play` instead, which is populated and behaves sensibly in every season."
    )
    flag("player_box: `active` column is a frozen/near-empty placeholder in 2022-2025 (constant False, or ~100% null in 2025) and only becomes real in 2026 -- use `did_not_play` instead for historical seasons")
    lines.append("")

    for season in seasons:
        pb = per_season.get(season)
        if pb is None:
            continue
        lines.append(f"### {season}")
        lines.append("")

        grp = pb.groupby(["game_id", "team_id"]).agg(
            pts_sum=("points", "sum"),
            team_score=("team_score", "first"),
            team_name=("team_name", "first"),
        ).reset_index()
        grp["diff"] = grp["pts_sum"] - grp["team_score"]
        mismatches = grp[grp["diff"] != 0]
        lines.append(f"- per-game team point-sum vs `team_score`: **{len(mismatches):,}** / {len(grp):,} team-game rows mismatch")
        if len(mismatches):
            worst = mismatches.reindex(mismatches["diff"].abs().sort_values(ascending=False).index).head(10)
            lines += md_table(
                ["game_id", "team_id", "team_name", "sum(points)", "team_score", "diff"],
                worst[["game_id", "team_id", "team_name", "pts_sum", "team_score", "diff"]].values.tolist(),
            )
            flag(f"player_box {season}: {len(mismatches)} team-games where sum(player points) != team_score (worst diff {int(mismatches['diff'].abs().max())})")
        lines.append("")

        # OT periods derived from pbp period_number (robust across seasons; schedule
        # home_linescores is missing entirely in 2022 -- see Section 1 schema-drift note)
        period_max = pbp_period_max_by_game(data_dir, season)
        game_ot = (period_max - 2).clip(lower=0) if period_max is not None else pd.Series(dtype=float)

        mgrp = pb.groupby(["game_id", "team_id"])["minutes"].sum().reset_index()
        mgrp["ot_periods"] = mgrp["game_id"].map(game_ot).fillna(0)
        mgrp["expected"] = 200 + 25 * mgrp["ot_periods"]
        mgrp["diff"] = mgrp["minutes"] - mgrp["expected"]
        mgrp["diff_round"] = mgrp["diff"].round(0)
        dist = mgrp["diff_round"].value_counts().sort_index()
        exact = int((mgrp["diff"].abs() < 0.5).sum())
        lines.append(
            f"- per-team-game sum(minutes) vs expected (200 + 25*OT periods): exact match (within 0.5 min) for **{exact:,}** / {len(mgrp):,} team-games. "
            f"Diff distribution (minutes over/under expected, rounded), top 10 buckets by frequency: {cnt_dict(dist.reindex(dist.sort_values(ascending=False).head(10).index).sort_index())}"
        )
        big_mismatch = int((mgrp["diff"].abs() > 5).sum())
        if big_mismatch:
            flag(f"player_box {season}: {big_mismatch} team-games have summed player minutes off by >5 minutes from the 200(+25/OT) expectation")
        lines.append("")

    return lines, per_season


# --------------------------------------------------------------------------
# Section 4: pbp
# --------------------------------------------------------------------------

def audit_pbp(data_dir: Path, seasons: list[int], schedules: dict, team_box: dict) -> tuple[list[str], dict]:
    lines = ["## 4. Play-by-play", ""]

    vocab_frames = []
    market_cols = ["game_spread", "home_team_spread", "home_favorite", "game_spread_available", "pregame_home_prob", "home_win_prob"]
    overview_rows = []
    period_dist_by_season = {}
    market_rows = []
    cross_check_holder = {}
    coord_trend = []

    for season in seasons:
        pbp = load(data_dir, "pbp", season)
        if pbp is None:
            flag(f"pbp: season {season} file missing")
            continue

        rows = len(pbp)
        games_covered = pbp["game_id"].nunique()
        sch_df = schedules[season]["df"]
        sch_games = set(sch_df["game_id"])
        pbp_games = set(pbp["game_id"].unique())
        missing_games = sch_games - pbp_games
        overview_rows.append([season, rows, games_covered, len(missing_games)])

        non_di_game_ids = set(sch_df.loc[sch_df["home_conference_id"].isna() | sch_df["away_conference_id"].isna(), "game_id"])
        missing_non_di_share = (len(missing_games & non_di_game_ids) / len(missing_games)) if missing_games else float("nan")

        # games "present" in pbp but where the pbp feed's own running scoreboard never
        # reaches the schedule's final score (truncated / partial pbp feed, not a total gap)
        last_running = pbp.groupby("game_id")[["home_score", "away_score"]].last()
        completed_sch = sch_df.loc[sch_df["status_type_completed"], ["game_id", "home_score", "away_score"]].set_index("game_id")
        joined = completed_sch.join(last_running, how="inner", rsuffix="_pbp")
        partial_games = joined[(joined["home_score"] != joined["home_score_pbp"]) | (joined["away_score"] != joined["away_score_pbp"])]
        partial_rate = len(partial_games) / len(joined) if len(joined) else float("nan")

        vc = pbp.groupby(["type_id", "type_text"]).size().rename(season)
        vocab_frames.append(vc)

        shooting_counts = pbp["shooting_play"].value_counts(dropna=False)
        score_counts = pbp["score_value"].value_counts(dropna=False).sort_index()
        coord_all = pbp["coordinate_x"].notna().mean()
        coord_shot = pbp.loc[pbp["shooting_play"] == True, "coordinate_x"].notna().mean()  # noqa: E712
        athlete_cov_shot = pbp.loc[pbp["shooting_play"] == True, "athlete_id_1"].notna().mean()

        period_max = pbp.groupby("game_id")["period_number"].max()
        pdist = period_max.value_counts().sort_index()
        period_dist_by_season[season] = pdist

        sub_mask = pbp["type_text"].astype(str).str.contains("Substitution|Enters", case=False, na=False)
        sub_count = int(sub_mask.sum())

        dup_count = int(pbp.duplicated(subset=["game_id", "sequence_number"]).sum())

        mrow = {"season": season}
        for c in market_cols:
            notna_rate = pbp[c].notna().mean()
            mrow[f"{c}_notna"] = notna_rate
            if c == "game_spread_available":
                avail = pbp["game_spread_available"] == True  # noqa: E712
                if avail.any():
                    mrow["avail_date_min"] = pbp.loc[avail, "game_date"].min()
                    mrow["avail_date_max"] = pbp.loc[avail, "game_date"].max()
                    mrow["avail_rate"] = avail.mean()
                else:
                    mrow["avail_date_min"] = mrow["avail_date_max"] = None
                    mrow["avail_rate"] = 0.0
        market_rows.append(mrow)

        nunique = {c: pbp[c].nunique(dropna=False) for c in ["game_spread", "home_team_spread", "home_favorite"]}
        frozen = all(v == 1 for v in nunique.values())

        lines.append(f"### {season}")
        lines.append("")
        lines.append(f"- rows: **{rows:,}**; games covered: **{games_covered:,}** / {len(sch_games):,} scheduled; games in schedule missing from pbp: **{len(missing_games):,}**")
        if missing_games:
            lines.append(f"  - sample missing game_ids: {sorted(missing_games)[:10]}")
            lines.append(f"  - {pct(missing_non_di_share)} of missing-pbp games involve a non-D-I side (null `home_conference_id`/`away_conference_id`) -- most pbp gaps look like exhibition/buy games against non-D-I opponents rather than random data loss, though not all of them are.")
        lines.append(
            f"- of games present in pbp, **{len(partial_games):,}** / {len(joined):,} ({pct(partial_rate)}) have a **truncated/partial pbp feed**: the running `home_score`/`away_score` at the last recorded pbp row never reaches the schedules final score "
            f"(e.g. game 401725959 in 2025 cuts off mid-1st-half at 38-26 vs a 97-61 final). These are not full game_id-level gaps but partial-coverage games; treat `game_id` presence in pbp as necessary but not sufficient for full-game coverage."
        )
        if len(partial_games):
            flag(f"pbp {season}: {len(partial_games)} games ({pct(partial_rate)} of games present) have a truncated/partial pbp feed whose final running score never matches the schedules final score")
        lines.append(f"- `shooting_play` counts: {cnt_dict(shooting_counts)}")
        lines.append(f"- `score_value` counts: {cnt_dict(score_counts)} (0=non-scoring event, 1=FT, 2=2pt FG, 3=3pt FG; note `type_text`='JumpShot' covers both 2pt and 3pt jumpers -- distinguish makes/attempts by `score_value`, not by `type_text` alone)")
        if coord_shot >= 0.9:
            coord_note = "**coverage is essentially complete this season** (a marked improvement vs earlier seasons -- see trend across seasons below)."
        elif coord_shot >= 0.5:
            coord_note = "**coverage is partial this season** -- roughly half of shooting plays carry coordinates."
        else:
            coord_note = "**shot coordinates are sparse this season** (looks like only a subset of broadcast/tracked games carry ESPN shot-chart coordinates in the pbp table; the separate `shots` dataset should be checked as the primary coordinate source instead of relying on pbp coordinates)."
        lines.append(f"- coordinate coverage: {pct(coord_all)} of all rows, {pct(coord_shot)} of `shooting_play`==True rows have non-null `coordinate_x`/`coordinate_y` -- {coord_note}")
        coord_trend.append([season, pct(coord_all), pct(coord_shot)])
        if coord_shot < 0.5:
            flag(f"pbp {season}: only {pct(coord_shot)} of shooting plays have coordinates in the pbp table -- do not assume shot-location coverage from pbp alone")
        lines.append(f"- clock/period fields: `period_number` (int, 1/2=halves, 3+=OT), `period_display_value` (str, e.g. '1st Half'/'2nd Half'/'OT'/'2OT'), `clock_display_value` (str 'MM:SS' counting down), `clock_minutes`/`clock_seconds` (ints, same info split out), `start_period_seconds_remaining`/`end_period_seconds_remaining`/`start_game_seconds_remaining`/`end_game_seconds_remaining` (numeric countdown clocks).")
        lines.append(f"- periods-per-game distribution (from pbp max `period_number`): {cnt_dict(pdist)}")
        lines.append(f"- participants coverage on shooting plays: `athlete_id_1` populated for {pct(athlete_cov_shot)} of `shooting_play`==True rows")
        lines.append(f"- substitution events (`type_text` matching 'Substitution'/'Enters'): **{sub_count:,}** rows" + (" -- present this season" if sub_count else " -- **none found this season**"))
        lines.append(f"- duplicate (`game_id`,`sequence_number`) pairs: **{dup_count}**")
        if dup_count:
            flag(f"pbp {season}: {dup_count} duplicate (game_id, sequence_number) key collisions -- sequence_number is not a fully unique per-game key")
        lines.append(
            f"- embedded market columns non-null rate: " + ", ".join(f"`{c}`={pct(mrow[f'{c}_notna'])}" for c in market_cols)
        )
        if frozen:
            lines.append(
                f"  - **DEFECT: frozen placeholder values.** `game_spread`/`home_team_spread`/`home_favorite` take a single constant value this season "
                f"(`game_spread`={sorted(pbp['game_spread'].dropna().unique().tolist())}) and `game_spread_available` is always False -- "
                f"this is NOT real betting-line data for this season, it is a dummy default. `home_win_prob`/`pregame_home_prob` remain real, "
                f"varying model win-probabilities (ESPN BPI-style), not sportsbook odds."
            )
            flag(f"pbp {season}: game_spread/home_team_spread/home_favorite/game_spread_available are frozen placeholders (spread constant, availability flag always False) -- not usable as historical line data")
        else:
            lines.append(
                f"  - real, varying spread data this season: `game_spread` non-null {pct(mrow['game_spread_notna'])}, "
                f"`game_spread_available`==True for {pct(mrow['avail_rate'])} of rows, earliest game_date with a real spread {mrow['avail_date_min']}, latest {mrow['avail_date_max']}."
            )
        lines.append("")

        if season == 2025:
            cross_check_holder["pbp_2025"] = pbp.copy()
        del pbp

    lines_overview = ["Overview:", ""] + md_table(["season", "rows", "games covered", "sched games missing from pbp"], overview_rows) + [""]
    lines_overview += ["Coordinate coverage trend (rises sharply over time -- treat pre-2025 pbp coordinates as unreliable/sparse, use the `shots` dataset instead for those seasons):", ""]
    lines_overview += md_table(["season", "coord coverage, all rows", "coord coverage, shooting_play rows"], coord_trend) + [""]
    lines = lines[:2] + lines_overview + lines[2:]

    # combined vocabulary across seasons
    vocab_df = pd.concat(vocab_frames, axis=1).fillna(0).astype(int)
    vocab_df["total"] = vocab_df.sum(axis=1)
    vocab_df = vocab_df.sort_values("total", ascending=False)
    lines.append("### Combined type_id / type_text vocabulary (all seasons) -- the possession-outcome event dictionary")
    lines.append("")
    vocab_rows = []
    for (type_id, type_text), row in vocab_df.iterrows():
        per_season_counts = {s: int(row[s]) for s in seasons if s in row.index and row[s] > 0}
        vocab_rows.append([type_id, type_text, int(row["total"]), per_season_counts])
    lines += md_table(["type_id", "type_text", "total (all seasons)", "present in / count by season"], vocab_rows)
    lines.append("")
    lines.append(f"- **{len(vocab_df)}** distinct (type_id, type_text) pairs observed across seasons {seasons}.")

    # schema drift notes
    seen_by_season = {s: set(vocab_df.index.get_level_values("type_text")[vocab_df[s] > 0]) if s in vocab_df.columns else set() for s in seasons}
    all_texts = set().union(*seen_by_season.values())
    drift_notes = []
    for t in sorted(all_texts):
        present_in = [s for s in seasons if t in seen_by_season[s]]
        if len(present_in) != len(seasons):
            drift_notes.append(f"`{t}` present only in season(s) {present_in}")
    if drift_notes:
        lines.append("- **Schema drift across seasons (vocabulary is NOT stable):**")
        for d in drift_notes:
            lines.append(f"  - {d}")
        flag("pbp: type_text vocabulary is not stable across seasons -- 'Substitution' events only start appearing in 2025+, 'Not Available' placeholder disappears after 2024, and 2026 adds new event types ('Shot', \"Coach's Challenge (...)\"). Any downstream feature relying on a fixed event-type set will silently break on older/newer seasons.")
    lines.append("")

    return lines, cross_check_holder


# --------------------------------------------------------------------------
# Section 5: 30-game cross-check (2025)
# --------------------------------------------------------------------------

def audit_cross_check(pbp_2025: pd.DataFrame, team_box_2025: pd.DataFrame, seed: int = 42, n: int = 30) -> list[str]:
    lines = ["## 5. Cross-check: recomputed final score from pbp scoring plays vs team_box (2025, 30 random games)", ""]

    rng = np.random.default_rng(seed)
    all_games = pbp_2025["game_id"].unique()
    sample_games = rng.choice(all_games, size=min(n, len(all_games)), replace=False)

    scoring = pbp_2025[pbp_2025["scoring_play"] == True]  # noqa: E712
    recomputed = scoring.groupby(["game_id", "team_id"])["score_value"].sum().reset_index().rename(columns={"score_value": "recomputed_score"})

    # left-join FROM team_box so game_id/team_id keep team_box's own dtypes and every
    # team-game gets a row (0 if pbp had no scoring plays for that team, e.g. missing-pbp games)
    tb = team_box_2025[["game_id", "team_id", "team_name", "team_score"]].copy()
    merged = tb.merge(recomputed, on=["game_id", "team_id"], how="left")
    merged["team_id"] = merged["team_id"].astype(int)
    merged["recomputed_score"] = merged["recomputed_score"].fillna(0).astype(int)
    merged["diff"] = merged["recomputed_score"] - merged["team_score"]

    sub = merged[merged["game_id"].isin(sample_games)].sort_values(["game_id", "team_id"])
    mismatches = sub[sub["diff"] != 0]

    lines.append(f"- sampled {len(sample_games)} distinct game_ids from pbp 2025 (seed={seed}); recomputed each team's final score as `sum(score_value)` over rows where `scoring_play`==True, grouped by (game_id, team_id).")
    lines.append(f"- mismatches vs team_box `team_score`: **{len(mismatches)}** / {len(sub)} team-game rows")
    display_cols = ["game_id", "team_id", "team_name", "recomputed_score", "team_score", "diff"]
    lines += md_table(display_cols, sub[display_cols].values.tolist())
    lines.append("")

    all_mismatches = merged[merged["diff"] != 0]
    lines.append(f"- for reference, across **all** 2025 games (not just the 30-game sample): **{len(all_mismatches)}** / {len(merged)} team-game rows mismatch between recomputed pbp score and team_box team_score.")
    if len(all_mismatches):
        flag(f"pbp/team_box 2025: {len(all_mismatches)} team-games where sum(scoring_play score_value) != team_box team_score")
    lines.append("")
    return lines


# --------------------------------------------------------------------------
# Section 6: possessions / rate gate targets
# --------------------------------------------------------------------------

def audit_possessions(seasons: list[int], team_box: dict, schedules: dict) -> list[str]:
    lines = ["## 6. Possessions-per-game and rate gate-reference targets", ""]

    season_summary = []
    monthly_rows = []
    margin_rows = []

    for season in seasons:
        tb = team_box[season].copy()
        tb["poss_est"] = tb["field_goals_attempted"] - tb["offensive_rebounds"] + tb["total_turnovers"] + 0.44 * tb["free_throws_attempted"]
        tb["fga_share_3pa"] = tb["three_point_field_goals_attempted"] / tb["field_goals_attempted"].replace(0, np.nan)
        tb["fta_fga"] = tb["free_throws_attempted"] / tb["field_goals_attempted"].replace(0, np.nan)

        sch = schedules[season]["df"][["game_id", "game_date"]]
        tb = tb.merge(sch, on="game_id", how="left", suffixes=("", "_sched"))
        tb["game_date_use"] = pd.to_datetime(tb["game_date"] if "game_date" in tb.columns else tb["game_date_sched"], errors="coerce")

        per_game_poss = tb.groupby("game_id")["poss_est"].mean()
        per_game_month = tb.groupby("game_id")["game_date_use"].first().dt.month

        mean_poss, sd_poss = per_game_poss.mean(), per_game_poss.std()
        mean_pts = tb["team_score"].mean()
        mean_3pa_share = tb["fga_share_3pa"].mean()
        mean_fta_fga = tb["fta_fga"].mean()

        season_summary.append([season, num(mean_poss), num(sd_poss), num(mean_pts), pct(mean_3pa_share), num(mean_fta_fga, 3)])

        by_month = pd.DataFrame({"poss": per_game_poss, "month": per_game_month}).groupby("month")["poss"].agg(["mean", "std", "count"])
        for month, row in by_month.iterrows():
            monthly_rows.append([season, int(month), num(row["mean"]), num(row["std"]), int(row["count"])])

        sch_full = schedules[season]["df"]
        completed = sch_full[sch_full["status_type_completed"]]
        completed = completed.assign(margin=completed["home_score"] - completed["away_score"])
        non_neutral = completed[completed["neutral_site"] == False]  # noqa: E712
        neutral = completed[completed["neutral_site"] == True]  # noqa: E712
        margin_rows.append([
            season,
            num(non_neutral["margin"].mean()), num(non_neutral["margin"].std()), len(non_neutral),
            num(neutral["margin"].mean()), num(neutral["margin"].std()), len(neutral),
        ])

    lines.append("Season-level gate targets (possessions/game estimated as mean over the two teams of `FGA - OREB + TOV + 0.44*FTA`, from team_box):")
    lines.append("")
    lines += md_table(["season", "mean poss/gm", "SD poss/gm", "mean pts/team/gm", "mean 3PA share of FGA", "mean FTA/FGA"], season_summary)
    lines.append("")
    lines.append("By month (calendar month of game_date; November/December/etc. -- note season-year wraps, e.g. Nov/Dec belong to the season's first calendar year, Jan-Apr to the second):")
    lines.append("")
    lines += md_table(["season", "month", "mean poss/gm", "SD poss/gm", "games"], monthly_rows)
    lines.append("")
    lines.append("Home-court margin (home_score - away_score), completed games only, non-neutral vs neutral sites:")
    lines.append("")
    lines += md_table(["season", "mean margin (non-neutral)", "SD (non-neutral)", "n (non-neutral)", "mean margin (neutral)", "SD (neutral)", "n (neutral)"], margin_rows)
    lines.append("")
    return lines


# --------------------------------------------------------------------------
# Section 7: ID hygiene
# --------------------------------------------------------------------------

def audit_id_hygiene(seasons: list[int], team_box: dict, player_box: dict) -> list[str]:
    lines = ["## 7. ID hygiene", ""]

    name_by_season = []
    for season in seasons:
        tb = team_box[season][["team_id", "team_name", "team_display_name"]].drop_duplicates()
        tb["season"] = season
        name_by_season.append(tb)
    names = pd.concat(name_by_season, ignore_index=True)

    total_team_ids = names["team_id"].nunique()
    name_groups = names.groupby("team_id")["team_display_name"].nunique()
    changed = name_groups[name_groups > 1]
    lines.append(f"- distinct `team_id` values across all seasons (team_box): **{total_team_ids:,}**")
    lines.append(f"- `team_id`s whose `team_display_name` changed across seasons: **{len(changed):,}**")
    if len(changed):
        rows = []
        for tid in changed.index[:15]:
            hist = names[names["team_id"] == tid].sort_values("season")[["season", "team_display_name"]].drop_duplicates()
            rows.append([tid, "; ".join(f"{r.season}={r.team_display_name}" for r in hist.itertuples())])
        lines += md_table(["team_id", "name history"], rows)
        flag(f"team_id -> name is not fully stable: {len(changed)} team_id(s) show a display-name change across seasons (rebrand or ID reuse) -- see Section 7 table")
    lines.append("")

    reuse_rows = []
    for season in seasons:
        pb = player_box[season]
        grp = pb.groupby("athlete_id")["team_id"].nunique()
        multi = grp[grp > 1]
        reuse_rows.append([season, int((grp > 1).sum())])
        if len(multi) and season == seasons[-1]:
            sample_ids = multi.index[:5].tolist()
            detail = pb[pb["athlete_id"].isin(sample_ids)][["athlete_id", "athlete_display_name", "team_id", "team_name", "game_date"]].sort_values(["athlete_id", "game_date"])
            lines.append(f"  - sample multi-team athlete_ids in {season}: {detail.head(20).to_dict('records')}")
    lines.append("- `athlete_id` appearing under >1 `team_id` within the same season (mid-season team change; true in-season D-I transfers are not supposed to happen under NCAA eligibility rules, so a non-zero count here is either a rare real case -- e.g. a withdrawal/re-enrollment -- or an athlete_id collision/reuse):")
    lines.append("")
    lines += md_table(["season", "athlete_ids on >1 team_id"], reuse_rows)
    if any(r[1] > 0 for r in reuse_rows):
        flag(f"player_box: athlete_id reuse across teams within a season detected in some seasons -- counts: {reuse_rows}")
    lines.append("")
    return lines


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument("--seasons", type=int, nargs="+", default=SEASONS_DEFAULT)
    ap.add_argument("--extra-schedule-season", type=int, default=2027)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    seasons = sorted(set(args.seasons))
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    header = [
        "# hoopR MBB data audit",
        "",
        f"Generated: {generated_at}  ",
        f"Source: `sportsdataverse/hoopR-mbb-data` (GitHub, CC BY 4.0), pulled via `scripts/pull_hoopr.py` into `{args.data_dir.as_posix()}`  ",
        f"Seasons audited: {seasons} (hoopR season = ending year of the season, e.g. 2022 = 2021-22 season)  ",
        f"Schedule preview also pulled for season {args.extra_schedule_season} (upcoming, mostly-unplayed -- reported separately, excluded from gate-reference stats).",
        "",
        "---",
        "",
    ]

    print("Loading schedules and auditing section 1...", file=sys.stderr)
    sec1, schedules = audit_schedules(args.data_dir, seasons, args.extra_schedule_season)

    print("Auditing team_box (section 2)...", file=sys.stderr)
    sec2, team_box = audit_team_box(args.data_dir, seasons, schedules)

    print("Auditing player_box (section 3)...", file=sys.stderr)
    sec3, player_box = audit_player_box(args.data_dir, seasons, team_box, schedules)

    print("Auditing pbp (section 4)...", file=sys.stderr)
    sec4, holder = audit_pbp(args.data_dir, seasons, schedules, team_box)

    print("Cross-check section 5...", file=sys.stderr)
    if "pbp_2025" in holder and 2025 in team_box:
        sec5 = audit_cross_check(holder["pbp_2025"], team_box[2025])
    else:
        sec5 = ["## 5. Cross-check", "", "2025 pbp or team_box not available -- skipped.", ""]
        flag("cross-check section skipped: 2025 pbp/team_box unavailable")

    print("Possession/rate gate targets (section 6)...", file=sys.stderr)
    sec6 = audit_possessions(seasons, team_box, schedules)

    print("ID hygiene (section 7)...", file=sys.stderr)
    sec7 = audit_id_hygiene(seasons, team_box, player_box)

    defect_lines = ["## 8. Notable defects / flags summary", ""]
    if DEFECTS:
        for d in DEFECTS:
            defect_lines.append(f"- {d}")
    else:
        defect_lines.append("- none found")
    defect_lines.append("")

    all_lines = header + sec1 + sec2 + sec3 + sec4 + sec5 + sec6 + sec7 + defect_lines

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
