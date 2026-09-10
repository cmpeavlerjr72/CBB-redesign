"""
reference.py -- season-level truth tables for the eval harness.

Every function here reads DATA, never an engine's own output; nothing in this
module knows what a "Control" or any other engine is. This is the shared
ground truth that `gates.py` (G1-G9) and `market.py` (G10) both grade sim
output against.

Sources (all pre-existing, built by other scripts -- read only, never
recomputed here):
    data/processed/games_universe.parquet   verified schedule + finals + D-I /
                                             truncation / seal flags
    data/reference/gate_targets_{season}.parquet   season reference targets
    data/raw/hoopr/team_box_{season}.parquet (via `cbb_sim.ratings.own_ratings
                                             .load_team_games`, the same
                                             possession/box builder the
                                             Control feature table uses)
    data/raw/cbbd/lines_{season}.parquet    market lines
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.data.seal import assert_not_sealed
from cbb_sim.ratings import own_ratings as orat

DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_REFERENCE_DIR = Path("data/reference")
DEFAULT_LINES_DIR = Path("data/raw/cbbd")
DEFAULT_HOOPR_DIR = Path("data/raw/hoopr")

# Provider preference, highest first. Names match `data/raw/cbbd/
# lines_providers.json` verbatim ("Draft Kings" has the space CBBD uses).
# `teamrankings` / `numberfire` are power-rating feeds, not sportsbooks, and
# are deliberately excluded from this list.
PROVIDER_PREFERENCE: tuple[str, ...] = ("Draft Kings", "ESPN BET", "Bovada", "consensus")


# ---------------------------------------------------------------------------
# schedule / finals truth
# ---------------------------------------------------------------------------
def load_actual_games(season: int, universe_path: Path | str = DEFAULT_UNIVERSE) -> pd.DataFrame:
    """One row per D-I, non-truncated, completed game in `season`.

    Columns: game_id, cbbd_game_id, season, game_date, month, tipoff_utc,
    home_team_id, away_team_id, neutral (0.0/1.0), home_score, away_score,
    margin, total, n_periods, went_ot.
    """
    assert_not_sealed(int(season), context="eval_gates/grade_market_games truth season")
    uni = pd.read_parquet(universe_path)
    keep = uni[
        uni["is_d1_game"]
        & ~uni["pbp_truncated"]
        & (uni["season"] == int(season))
        & uni["home_score"].notna()
        & uni["away_score"].notna()
    ].copy()
    keep["neutral"] = keep["neutral_site"].astype(float)
    keep["home_score"] = keep["home_score"].astype("int64")
    keep["away_score"] = keep["away_score"].astype("int64")
    keep["margin"] = keep["home_score"] - keep["away_score"]
    keep["total"] = keep["home_score"] + keep["away_score"]
    keep["month"] = pd.to_datetime(keep["game_date"]).dt.month.astype("int64")
    keep["went_ot"] = keep["n_periods"] > 2
    return keep[[
        "game_id", "cbbd_game_id", "season", "game_date", "month", "tipoff_utc",
        "home_team_id", "away_team_id", "neutral", "home_score", "away_score",
        "margin", "total", "n_periods", "went_ot",
    ]].reset_index(drop=True)


def load_actual_possessions(season: int, universe: pd.DataFrame | None = None,
                             hoopr_dir: Path | str = DEFAULT_HOOPR_DIR) -> pd.DataFrame:
    """One row per game_id: `game_poss`, the box-derived possession estimate
    (FGA - OREB + TOV + 0.44*FTA, averaged over both teams) -- the same
    definition `scripts/build_gate_reference.py` and the Control feature
    table use. Games missing either side's box row are absent (cannot produce
    an estimate), same as `own_ratings.load_team_games`.
    """
    if universe is None:
        universe = orat.load_universe()
    tg = orat.load_team_games(universe, [int(season)], hoopr_dir)
    return tg[["game_id", "game_poss"]].drop_duplicates("game_id").reset_index(drop=True)


def load_actual_team_box(season: int, universe: pd.DataFrame | None = None,
                          hoopr_dir: Path | str = DEFAULT_HOOPR_DIR) -> pd.DataFrame:
    """One row per team-game: raw box counts + the derived four-factor /
    shot-mix columns G2-G4 compare sim output against. Formulas verified
    against `data/reference/gate_targets_{season}.parquet` season-level means
    to the 6th decimal (ppp, efg, tov%, oreb%, ft rate, three-point share all
    use each team's OWN `poss_team` as denominator, not the shared game_poss):

        ppp        = team_score / poss_team
        efg_pct    = (fgm + 0.5*tpm) / fga
        tov_pct    = tov / poss_team
        oreb_pct   = oreb / (oreb + opp_dreb)
        ft_rate    = fta / fga
        three_share= tpa / fga
    """
    if universe is None:
        universe = orat.load_universe()
    tg = orat.load_team_games(universe, [int(season)], hoopr_dir)
    opp = tg[["game_id", "team_id", "dreb"]].rename(
        columns={"team_id": "opp_team_id", "dreb": "opp_dreb"}
    )
    m = tg.merge(opp, on=["game_id", "opp_team_id"], how="left")
    m["fg2a"] = m["fga"] - m["tpa"]
    m["ppp"] = m["team_score"] / m["poss_team"]
    m["efg_pct"] = (m["fgm"] + 0.5 * m["tpm"]) / m["fga"]
    m["tov_pct"] = m["tov"] / m["poss_team"]
    m["oreb_pct"] = m["oreb"] / (m["oreb"] + m["opp_dreb"])
    m["ft_rate"] = m["fta"] / m["fga"]
    m["three_share"] = m["tpa"] / m["fga"]
    return m


# ---------------------------------------------------------------------------
# gate_targets_{season}.parquet
# ---------------------------------------------------------------------------
def load_gate_targets(season: int, reference_dir: Path | str = DEFAULT_REFERENCE_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(reference_dir) / f"gate_targets_{int(season)}.parquet")


def gate_target_value(ref: pd.DataFrame, breakdown: str, group, metric: str,
                       side: str = "overall") -> float:
    m = ref[
        (ref["breakdown"] == breakdown)
        & (ref["group"].astype(str) == str(group))
        & (ref["metric"] == metric)
        & (ref["side"] == side)
    ]
    return float(m["value"].iloc[0]) if len(m) else float("nan")


# ---------------------------------------------------------------------------
# market lines
# ---------------------------------------------------------------------------
def load_lines(season: int, lines_dir: Path | str = DEFAULT_LINES_DIR,
               provider_preference: tuple[str, ...] = PROVIDER_PREFERENCE) -> pd.DataFrame:
    """One row per cbbd_game_id: the best-available provider's line, per the
    preference order (`PROVIDER_PREFERENCE`, DraftKings > ESPN BET > Bovada >
    consensus). `provider_used` records which one won so every downstream
    number is traceable. Only `season == 2025` (this project's current
    non-sealed graded season) has just one provider (ESPN BET) end to end, so
    the harness's numbers on that season are identical to
    `scripts/grade_control.py`'s hardcoded ESPN-BET-only join.
    """
    path = Path(lines_dir) / f"lines_{int(season)}.parquet"
    df = pd.read_parquet(path)
    df = df[df["provider"].isin(provider_preference)].copy()
    rank = {p: i for i, p in enumerate(provider_preference)}
    df["_rank"] = df["provider"].map(rank)
    df = df.sort_values(["gameId", "_rank"], kind="mergesort").drop_duplicates("gameId", keep="first")
    df["cbbd_game_id"] = pd.to_numeric(df["gameId"], errors="coerce").astype("Int64")
    for c in ("spread", "spreadOpen", "overUnder", "overUnderOpen", "homeMoneyline", "awayMoneyline"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"provider": "provider_used"})
    return df[[
        "cbbd_game_id", "provider_used", "spread", "spreadOpen",
        "overUnder", "overUnderOpen", "homeMoneyline", "awayMoneyline",
    ]].reset_index(drop=True)


def team_quality_terciles(actual_games: pd.DataFrame) -> dict[str, pd.Series]:
    """GRADING ONLY -- never a model feature (same caveat as
    `scripts/grade_control.py`'s `team_tiers`). Three leak-tolerant-for-
    grading tercile splits of each team's OWN full-season actuals, built from
    the same `actual_games` frame every gate grades against:

        margin   -- net tercile (as `grade_control.team_tiers`)
        offense  -- tercile of points scored
        defense  -- tercile of points allowed (ascending: bottom = worst D)

    Returns {"margin": Series, "offense": Series, "defense": Series}, each
    indexed by team_id -> tercile label.
    """
    home = actual_games[["home_team_id", "home_score", "away_score"]].rename(
        columns={"home_team_id": "team_id", "home_score": "pts_for", "away_score": "pts_against"}
    )
    away = actual_games[["away_team_id", "away_score", "home_score"]].rename(
        columns={"away_team_id": "team_id", "away_score": "pts_for", "home_score": "pts_against"}
    )
    tg = pd.concat([home, away], ignore_index=True)
    tg["margin"] = tg["pts_for"] - tg["pts_against"]
    g = tg.groupby("team_id")
    labels = ["bottom_tercile", "middle_tercile", "top_tercile"]
    out = {}
    for name, col in (("margin", "margin"), ("offense", "pts_for"), ("defense", "pts_against")):
        mean = g[col].mean()
        try:
            out[name] = pd.qcut(mean, 3, labels=labels)
        except ValueError:
            out[name] = pd.Series(np.nan, index=mean.index)
    return out
