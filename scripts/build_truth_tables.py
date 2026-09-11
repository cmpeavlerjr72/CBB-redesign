#!/usr/bin/env python
"""
build_truth_tables.py -- team/player/finals truth tables for engine gates
G3, G4 and G8 (`docs/tests/gates_engine_v0_F2_2025_s5_r2event_2026-09-10.md`).

    .venv/Scripts/python.exe scripts/build_truth_tables.py

Writes, for seasons 2022-2025 (2026 is sealed and is never read here):

    data/processed/truth/team_game_shots_v1.parquet
    data/processed/truth/player_game_v1.parquet
    data/processed/truth/game_finals_v1.parquet
    data/processed/truth/build_report.json

Every table carries BOTH independent sources side by side -- the CBBD
possession/pbp event layer and the hoopR box/schedule -- plus a diff and a
disagreement flag per stat. Neither source is dropped or silently preferred;
`scripts/eval_gates.py` / `src/cbb_sim/eval/gates.py` decide which column to
read for a given gate, and the choice is recorded there, not made silently
here.

Sources
-------
team_game_shots_v1   event layer: data/processed/possessions_v2/possessions_{season}.parquet
                      box:        data/raw/hoopr/team_box/team_box_{season}.parquet
player_game_v1        box:        data/raw/hoopr/player_box/player_box_{season}.parquet
                      event layer: data/raw/cbbd/pbp/plays_{season}.parquet, keyed on
                                   shot_shooter_id (NOT participant_1_id -- see
                                   docs/models/change_ledger.md, section A, the L4 usage
                                   ROUND 2 row: participant_1_id is the ASSISTER on ~half
                                   of assisted made FGAs)
                      crosswalk:  data/processed/player_crosswalk.parquet
                                  (`src/cbb_sim/data/player_ids.py`) -- CBBD rosters exist
                                  for 2024-2026 only, so the event-layer per-class columns
                                  and the cbbd_player_id key are NULL for 2022-2023 by
                                  construction, not a bug; reported, not hidden.
game_finals_v1         hoopR:      data/processed/games_universe.parquet (home_score,
                                   away_score, n_periods -- already hoopR-schedule-derived)
                      CBBD:       data/raw/cbbd/games_{season}.parquet (homePoints,
                                   awayPoints, homePeriodPoints, awayPeriodPoints)

Universe filter for all three tables: `is_d1_game & ~pbp_truncated`, matching
every other truth table in this repo (`src/cbb_sim/eval/reference.py`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.data.player_ids import load_crosswalk  # noqa: E402
from cbb_sim.pbp.events import classify_frame, load_plays  # noqa: E402

SEASONS: tuple[int, ...] = (2022, 2023, 2024, 2025)
SEALED_SEASON = 2026

UNIVERSE_PATH = Path("data/processed/games_universe.parquet")
POSSESSIONS_DIR = Path("data/processed/possessions_v2")
HOOPR_DIR = Path("data/raw/hoopr")
CBBD_DIR = Path("data/raw/cbbd")
OUT_DIR = Path("data/processed/truth")

SHOT_CLASSES = ("FGA_rim", "FGA_jump2", "FGA_3")
CROSSWALK_SEASONS = (2024, 2025, 2026)  # CBBD rosters exist for these only


def _diff_buckets(diff: pd.Series) -> dict[str, float]:
    d = diff.abs()
    n = int(d.notna().sum())
    if n == 0:
        return {"n": 0, "pct_0": float("nan"), "pct_1": float("nan"), "pct_2plus": float("nan")}
    return {
        "n": n,
        "pct_0": float((d == 0).sum()) / n,
        "pct_1": float((d == 1).sum()) / n,
        "pct_2plus": float((d >= 2).sum()) / n,
    }


def load_universe() -> pd.DataFrame:
    u = pd.read_parquet(UNIVERSE_PATH)
    if int(SEALED_SEASON) in set(u["season"].unique()):
        u = u[u["season"] != SEALED_SEASON].copy()
    return u


# ---------------------------------------------------------------------------
# 1. team_game_shots_v1
# ---------------------------------------------------------------------------
def build_team_game_event(universe: pd.DataFrame, season: int) -> pd.DataFrame:
    path = POSSESSIONS_DIR / f"possessions_{season}.parquet"
    poss = pd.read_parquet(
        path,
        columns=["game_id", "offense_team_id", "offense_is_home",
                 "fga_rim", "fgm_rim", "fga_jump2", "fgm_jump2",
                 "fga_3", "fgm_3", "fta", "ftm"],
    )
    uni = universe[(universe["season"] == season) & universe["is_d1_game"] & ~universe["pbp_truncated"]]
    game_ids = set(uni["game_id"].tolist())
    poss = poss[poss["game_id"].isin(game_ids)].copy()

    home_map = uni.set_index("game_id")["home_team_id"]
    away_map = uni.set_index("game_id")["away_team_id"]
    poss["team_id"] = np.where(
        poss["offense_is_home"].to_numpy(),
        poss["game_id"].map(home_map).to_numpy(),
        poss["game_id"].map(away_map).to_numpy(),
    )
    poss["team_id"] = poss["team_id"].astype("int64")

    ev_cols = ["fga_rim", "fgm_rim", "fga_jump2", "fgm_jump2", "fga_3", "fgm_3", "fta", "ftm"]
    agg = poss.groupby(["game_id", "team_id"], as_index=False)[ev_cols].sum()
    agg = agg.rename(columns={c: f"ev_{c}" for c in ev_cols})
    agg["ev_fga"] = agg["ev_fga_rim"] + agg["ev_fga_jump2"] + agg["ev_fga_3"]
    agg["ev_fgm"] = agg["ev_fgm_rim"] + agg["ev_fgm_jump2"] + agg["ev_fgm_3"]
    agg["season"] = int(season)
    return agg


def build_team_game_box(season: int) -> pd.DataFrame:
    path = HOOPR_DIR / "team_box" / f"team_box_{season}.parquet"
    cols = ["game_id", "team_id", "field_goals_made", "field_goals_attempted",
            "three_point_field_goals_made", "three_point_field_goals_attempted",
            "free_throws_made", "free_throws_attempted"]
    tb = pd.read_parquet(path, columns=cols)
    tb["game_id"] = pd.to_numeric(tb["game_id"], errors="coerce").astype("int64")
    tb["team_id"] = pd.to_numeric(tb["team_id"], errors="coerce").astype("int64")
    tb = tb.rename(columns={
        "field_goals_made": "box_fgm", "field_goals_attempted": "box_fga",
        "three_point_field_goals_made": "box_fgm3", "three_point_field_goals_attempted": "box_fga3",
        "free_throws_made": "box_ftm", "free_throws_attempted": "box_fta",
    })
    for c in ["box_fgm", "box_fga", "box_fgm3", "box_fga3", "box_ftm", "box_fta"]:
        tb[c] = pd.to_numeric(tb[c], errors="coerce")
    tb["box_fga2"] = tb["box_fga"] - tb["box_fga3"]
    tb["box_fgm2"] = tb["box_fgm"] - tb["box_fgm3"]
    return tb


def build_team_game_shots(universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames = []
    for season in SEASONS:
        uni_s = universe[(universe["season"] == season) & universe["is_d1_game"] & ~universe["pbp_truncated"]]
        game_ids = set(uni_s["game_id"].tolist())

        ev = build_team_game_event(universe, season)
        box = build_team_game_box(season)
        box = box[box["game_id"].isin(game_ids)].copy()
        m = ev.merge(box, on=["game_id", "team_id"], how="outer", indicator=True)
        m["season"] = season
        m["match_source"] = m["_merge"].map({"both": "event_and_box", "left_only": "event_only", "right_only": "box_only"})
        m = m.drop(columns=["_merge"])

        home_of = m["game_id"].map(uni_s.set_index("game_id")["home_team_id"])
        away_of = m["game_id"].map(uni_s.set_index("game_id")["away_team_id"])
        m["opp_team_id"] = np.where(m["team_id"] == home_of, away_of, home_of)

        for stat in ("fga", "fgm", "fga3", "fgm3", "fta", "ftm"):
            ev_col = "ev_fga_3" if stat == "fga3" else ("ev_fgm_3" if stat == "fgm3" else f"ev_{stat}")
            box_col = f"box_{stat}"
            m[f"diff_{stat}"] = m[ev_col] - m[box_col]

        m["rim_share_ev"] = m["ev_fga_rim"] / m["ev_fga"].replace(0, np.nan)
        m["jump2_share_ev"] = m["ev_fga_jump2"] / m["ev_fga"].replace(0, np.nan)
        m["three_share_ev"] = m["ev_fga_3"] / m["ev_fga"].replace(0, np.nan)
        m["ft_rate_ev"] = m["ev_fta"] / m["ev_fga"].replace(0, np.nan)
        m["three_share_box"] = m["box_fga3"] / m["box_fga"].replace(0, np.nan)
        m["ft_rate_box"] = m["box_fta"] / m["box_fga"].replace(0, np.nan)

        disagree_cols = [f"diff_{s}" for s in ("fga", "fgm", "fga3", "fgm3", "fta", "ftm")]
        m["any_disagreement"] = (m[disagree_cols].abs() > 0).any(axis=1)

        frames.append(m)

    out = pd.concat(frames, ignore_index=True)

    recon: dict = {}
    for season in SEASONS:
        s = out[out["season"] == season]
        both = s[s["match_source"] == "event_and_box"]
        recon[str(season)] = {
            "n_team_games": int(len(s)),
            "n_event_and_box": int(len(both)),
            "n_event_only": int((s["match_source"] == "event_only").sum()),
            "n_box_only": int((s["match_source"] == "box_only").sum()),
            "reconciliation": {stat: _diff_buckets(both[f"diff_{stat}"]) for stat in
                                ("fga", "fgm", "fga3", "fgm3", "fta", "ftm")},
        }
    return out, recon


# ---------------------------------------------------------------------------
# 2. player_game_v1
# ---------------------------------------------------------------------------
PLAYER_BOX_COLS = {
    "game_id": "game_id", "season": "season", "athlete_id": "athlete_id",
    "team_id": "team_id", "opponent_team_id": "opp_team_id",
    "minutes": "minutes", "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "orb", "defensive_rebounds": "drb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov",
    "fouls": "pf", "points": "pts", "starter": "starter", "did_not_play": "did_not_play",
}


def build_player_game_box(season: int) -> pd.DataFrame:
    path = HOOPR_DIR / "player_box" / f"player_box_{season}.parquet"
    pb = pd.read_parquet(path, columns=list(PLAYER_BOX_COLS.keys()))
    pb = pb.rename(columns=PLAYER_BOX_COLS)
    pb["game_id"] = pd.to_numeric(pb["game_id"], errors="coerce").astype("Int64")
    pb["athlete_id"] = pd.to_numeric(pb["athlete_id"], errors="coerce").astype("Int64")
    pb["team_id"] = pd.to_numeric(pb["team_id"], errors="coerce").astype("Int64")
    pb["opp_team_id"] = pd.to_numeric(pb["opp_team_id"], errors="coerce").astype("Int64")
    for c in ["minutes", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "orb", "drb",
              "ast", "stl", "blk", "tov", "pf", "pts"]:
        pb[c] = pd.to_numeric(pb[c], errors="coerce")
    pb["fg2m"] = pb["fgm"] - pb["fg3m"]
    pb["fg2a"] = pb["fga"] - pb["fg3a"]
    pb = pb[pb["athlete_id"].notna()].copy()
    pb["season"] = int(season)
    return pb


def _made_flag(plays: pd.DataFrame) -> np.ndarray:
    made = plays["shot_made"]
    if made.dtype == object:
        made = made.map({True: True, False: False})
    return (made.astype("boolean").fillna(plays["scoringPlay"].astype("boolean"))
            .fillna(False).to_numpy(dtype=bool))


def build_player_shots_event(season: int, universe: pd.DataFrame) -> pd.DataFrame:
    """Per (cbbd game_id, shot_shooter_id): FGA/FGM by class + FTA/FTM, keyed
    on `shot_shooter_id` -- never `participant_1_id`, the assister on ~half of
    assisted made FGAs (see module docstring)."""
    uni_s = universe[(universe["season"] == season) & universe["is_d1_game"] & ~universe["pbp_truncated"]]
    wanted_cbbd = set(pd.to_numeric(uni_s["cbbd_game_id"], errors="coerce").dropna().astype("int64").tolist())

    plays = load_plays(season, game_ids=wanted_cbbd)
    cls = classify_frame(plays).to_numpy()
    made = _made_flag(plays)
    shooter = pd.to_numeric(plays["shot_shooter_id"], errors="coerce")

    df = pd.DataFrame({
        "cbbd_game_id": plays["gameId"].to_numpy(),
        "shot_shooter_id": shooter.to_numpy(),
        "cls": cls,
        "made": made,
    })
    df = df.dropna(subset=["shot_shooter_id"])
    df["shot_shooter_id"] = df["shot_shooter_id"].astype("int64")

    is_shot = df["cls"].isin(SHOT_CLASSES)
    is_ft = df["cls"].isin(("FT_made", "FT_missed"))

    shot_rows = df[is_shot]
    att = (shot_rows.groupby(["cbbd_game_id", "shot_shooter_id", "cls"]).size()
           .unstack(fill_value=0).reindex(columns=SHOT_CLASSES, fill_value=0))
    att.columns = [f"ev_fga_{c.split('_')[-1].lower()}" for c in att.columns]
    mk = (shot_rows[shot_rows["made"]].groupby(["cbbd_game_id", "shot_shooter_id", "cls"]).size()
          .unstack(fill_value=0).reindex(columns=SHOT_CLASSES, fill_value=0))
    mk.columns = [f"ev_fgm_{c.split('_')[-1].lower()}" for c in mk.columns]

    ft_rows = df[is_ft]
    fta = ft_rows.groupby(["cbbd_game_id", "shot_shooter_id"]).size().rename("ev_fta")
    ftm = (ft_rows[ft_rows["cls"] == "FT_made"].groupby(["cbbd_game_id", "shot_shooter_id"]).size()
           .rename("ev_ftm"))

    out = att.join(mk, how="outer").join(fta, how="outer").join(ftm, how="outer").fillna(0)
    out = out.reset_index()
    for c in out.columns:
        if c not in ("cbbd_game_id", "shot_shooter_id"):
            out[c] = out[c].astype("int64")

    out["ev_fga"] = out["ev_fga_rim"] + out["ev_fga_jump2"] + out["ev_fga_3"]
    out["ev_fgm"] = out["ev_fgm_rim"] + out["ev_fgm_jump2"] + out["ev_fgm_3"]

    # cbbd game_id -> hoopR game_id
    gmap = uni_s.set_index("cbbd_game_id")["game_id"]
    out["game_id"] = out["cbbd_game_id"].map(gmap)
    out = out[out["game_id"].notna()].copy()
    out["game_id"] = out["game_id"].astype("int64")
    out["season"] = int(season)
    return out.drop(columns=["cbbd_game_id"])


def build_player_game(universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cw = load_crosswalk()
    frames = []
    crosswalk_report: dict = {}

    for season in SEASONS:
        uni_s = universe[(universe["season"] == season) & universe["is_d1_game"] & ~universe["pbp_truncated"]]
        game_ids = set(uni_s["game_id"].tolist())

        box = build_player_game_box(season)
        box = box[box["game_id"].astype("int64").isin(game_ids)].copy()

        has_crosswalk = season in CROSSWALK_SEASONS and (cw["season"] == season).any()
        if has_crosswalk:
            e2c: dict[int, int] = {}
            season_cw = cw[(cw["season"] == season) & cw["espn_athlete_id"].notna()]
            # keep the first cbbd id per espn id (collisions are rare and reported below)
            season_cw = season_cw.drop_duplicates(subset=["espn_athlete_id"], keep="first")
            e2c = dict(zip(season_cw["espn_athlete_id"].astype("int64"), season_cw["cbbd_player_id"].astype("int64")))
            box["cbbd_player_id"] = box["athlete_id"].astype("int64").map(e2c)
        else:
            box["cbbd_player_id"] = pd.array([pd.NA] * len(box), dtype="Int64")

        n_rows = int(len(box))
        n_matched = int(box["cbbd_player_id"].notna().sum())
        played = box[box["minutes"].fillna(0) > 0]
        n_played = int(len(played))
        n_played_matched = int(played["cbbd_player_id"].notna().sum())
        crosswalk_report[str(season)] = {
            "cbbd_rosters_available": bool(has_crosswalk),
            "player_game_rows": n_rows,
            "matched_rows": n_matched,
            "row_match_rate": (n_matched / n_rows) if n_rows else float("nan"),
            "played_rows": n_played,
            "played_matched": n_played_matched,
            "played_match_rate": (n_played_matched / n_played) if n_played else float("nan"),
        }

        if has_crosswalk:
            ev = build_player_shots_event(season, universe)
            ev = ev.rename(columns={"shot_shooter_id": "cbbd_player_id"})
            box = box.merge(ev.drop(columns=["season"]), on=["game_id", "cbbd_player_id"], how="left")
            ev_cols = ["ev_fga_rim", "ev_fgm_rim", "ev_fga_jump2", "ev_fgm_jump2",
                       "ev_fga_3", "ev_fgm_3", "ev_fta", "ev_ftm", "ev_fga", "ev_fgm"]
            # a matched player with no event-layer shooter row genuinely took 0 --
            # fill 0 ONLY where cbbd_player_id resolved; leave NaN where it didn't
            resolved = box["cbbd_player_id"].notna()
            for c in ev_cols:
                box.loc[resolved, c] = box.loc[resolved, c].fillna(0)
        else:
            for c in ("ev_fga_rim", "ev_fgm_rim", "ev_fga_jump2", "ev_fgm_jump2",
                      "ev_fga_3", "ev_fgm_3", "ev_fta", "ev_ftm", "ev_fga", "ev_fgm"):
                box[c] = np.nan

        for stat, ev_col, box_col in (
            ("fga", "ev_fga", "fga"), ("fgm", "ev_fgm", "fgm"),
            ("fga3", "ev_fga_3", "fg3a"), ("fgm3", "ev_fgm_3", "fg3m"),
            ("fta", "ev_fta", "fta"), ("ftm", "ev_ftm", "ftm"),
        ):
            box[f"diff_{stat}"] = box[ev_col] - box[box_col]

        frames.append(box)

    out = pd.concat(frames, ignore_index=True)

    recon: dict = {}
    for season in SEASONS:
        s = out[out["season"] == season]
        has_ev = s["ev_fga"].notna()
        recon[str(season)] = {
            "n_player_games": int(len(s)),
            "n_with_event_layer_match": int(has_ev.sum()),
            "reconciliation": {
                stat: _diff_buckets(s.loc[has_ev, f"diff_{stat}"])
                for stat in ("fga", "fgm", "fga3", "fgm3", "fta", "ftm")
            } if has_ev.any() else "no event-layer match this season (no CBBD roster crosswalk)",
        }
    return out, {"crosswalk_coverage": crosswalk_report, "shot_reconciliation": recon}


# ---------------------------------------------------------------------------
# 3. game_finals_v1
# ---------------------------------------------------------------------------
#: hoopR serialises `home_linescores` as numpy's `repr` of an array of dicts,
#: which has NO comma between dict elements ("[{'value': 36.0} {'value':
#: 39.0}]") -- not valid Python list syntax, so `ast.literal_eval` silently
#: fails on it. `universe.py`'s own `_linescore_periods` sidesteps this by
#: counting `'value'` occurrences with a regex rather than parsing the
#: structure; this does the same thing but captures the number instead of
#: just counting it.
_LINESCORE_VALUE_RE = re.compile(r"'value':\s*([\-0-9.]+)")


def _parse_hoopr_linescores(s) -> list[float] | None:
    if not isinstance(s, str) or not s.strip():
        return None
    hits = _LINESCORE_VALUE_RE.findall(s)
    if not hits:
        return None
    try:
        return [float(h) for h in hits]
    except ValueError:
        return None


def build_game_finals(universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames = []
    recon: dict = {}
    for season in SEASONS:
        uni_s = universe[(universe["season"] == season) & universe["is_d1_game"] & ~universe["pbp_truncated"]].copy()
        uni_s = uni_s[uni_s["home_score"].notna() & uni_s["away_score"].notna()]

        sched_path = HOOPR_DIR / "schedules" / f"mbb_schedule_{season}.parquet"
        has_linescores = season != 2022
        sched_cols = ["game_id", "home_linescores", "away_linescores"] if has_linescores else ["game_id"]
        sched = pd.read_parquet(sched_path, columns=sched_cols)
        sched["game_id"] = pd.to_numeric(sched["game_id"], errors="coerce").astype("int64")
        if has_linescores:
            sched["hoopr_periods_home"] = sched["home_linescores"].map(_parse_hoopr_linescores)
            sched["hoopr_periods_away"] = sched["away_linescores"].map(_parse_hoopr_linescores)
        else:
            sched["hoopr_periods_home"] = None
            sched["hoopr_periods_away"] = None

        cbbd_path = CBBD_DIR / f"games_{season}.parquet"
        cbbd = pd.read_parquet(cbbd_path, columns=["id", "sourceId", "homePoints", "awayPoints",
                                                     "homePeriodPoints", "awayPeriodPoints"])
        cbbd["sourceId"] = pd.to_numeric(cbbd["sourceId"], errors="coerce").astype("Int64")
        cbbd = cbbd.rename(columns={"sourceId": "game_id"})

        m = uni_s[["game_id", "cbbd_game_id", "home_team_id", "away_team_id",
                   "home_score", "away_score", "n_periods", "neutral_site"]].merge(
            sched[["game_id", "hoopr_periods_home", "hoopr_periods_away"]], on="game_id", how="left"
        ).merge(
            cbbd[["game_id", "homePoints", "awayPoints", "homePeriodPoints", "awayPeriodPoints"]],
            on="game_id", how="left",
        )
        m["season"] = season
        m["cbbd_n_periods"] = m["homePeriodPoints"].map(lambda x: len(x) if x is not None else np.nan)
        m["hoopr_n_ot"] = m["n_periods"] - 2
        m["cbbd_n_ot"] = m["cbbd_n_periods"] - 2
        m["diff_home_score"] = m["home_score"] - m["homePoints"]
        m["diff_away_score"] = m["away_score"] - m["awayPoints"]
        m["diff_n_periods"] = m["n_periods"] - m["cbbd_n_periods"]

        def _period_scores_disagree(row) -> bool | None:
            h1, h2 = row["hoopr_periods_home"], row["homePeriodPoints"]
            a1, a2 = row["hoopr_periods_away"], row["awayPeriodPoints"]
            if h1 is None or h2 is None or a1 is None or a2 is None:
                return None
            n = min(len(h1), len(h2))
            m2 = min(len(a1), len(a2))
            if n == 0 or m2 == 0:
                return None
            return bool(any(abs(h1[i] - h2[i]) > 0 for i in range(n))
                        or any(abs(a1[i] - a2[i]) > 0 for i in range(m2))
                        or len(h1) != len(h2) or len(a1) != len(a2))

        m["period_scores_disagree"] = m.apply(_period_scores_disagree, axis=1)
        frames.append(m)

        both = m[m["homePoints"].notna()]
        both_n_periods = both[both["n_periods"].notna() & both["cbbd_n_periods"].notna()]
        n_period_checked = int(m["period_scores_disagree"].notna().sum())
        recon[str(season)] = {
            "n_games": int(len(m)),
            "n_with_cbbd_match": int(len(both)),
            "final_score_disagree": int(((both["diff_home_score"] != 0) | (both["diff_away_score"] != 0)).sum()),
            "n_periods_both_present": int(len(both_n_periods)),
            "n_periods_disagree": int((both_n_periods["diff_n_periods"] != 0).sum()),
            "n_periods_hoopr_missing": int(both["n_periods"].isna().sum()),
            "hoopr_per_period_available": bool(has_linescores),
            "n_period_scores_checked": n_period_checked,
            "n_period_scores_disagree": int((m["period_scores_disagree"] == True).sum()),  # noqa: E712
        }

    out = pd.concat(frames, ignore_index=True)
    out = out.drop(columns=["hoopr_periods_home", "hoopr_periods_away", "homePeriodPoints", "awayPeriodPoints"])
    return out, recon


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe()

    print("Building team_game_shots_v1 ...")
    tgs, tgs_recon = build_team_game_shots(universe)
    tgs_path = out_dir / "team_game_shots_v1.parquet"
    tgs.to_parquet(tgs_path, index=False)
    print(f"  wrote {tgs_path} ({len(tgs)} rows, {tgs_path.stat().st_size / 1e6:.2f} MB)")

    print("Building player_game_v1 ...")
    pg, pg_recon = build_player_game(universe)
    pg_path = out_dir / "player_game_v1.parquet"
    pg.to_parquet(pg_path, index=False)
    print(f"  wrote {pg_path} ({len(pg)} rows, {pg_path.stat().st_size / 1e6:.2f} MB)")

    print("Building game_finals_v1 ...")
    gf, gf_recon = build_game_finals(universe)
    gf_path = out_dir / "game_finals_v1.parquet"
    gf.to_parquet(gf_path, index=False)
    print(f"  wrote {gf_path} ({len(gf)} rows, {gf_path.stat().st_size / 1e6:.2f} MB)")

    report = {
        "seasons": list(SEASONS),
        "team_game_shots_v1": tgs_recon,
        "player_game_v1": pg_recon,
        "game_finals_v1": gf_recon,
    }
    report_path = out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
