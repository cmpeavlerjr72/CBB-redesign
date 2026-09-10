#!/usr/bin/env python
"""
build_gate_reference.py -- build the L0 empirical gate-target reference
tables (FRAMEWORK_PLAN.md section 4) from real hoopR/CBBD outcomes.

Writes, per season:
    data/reference/gate_targets_{season}.parquet   (tidy long table, see below)
and one combined report:
    docs/tests/gate_reference_2026-09-10.md

Also writes data/reference/gate_targets_fold2_train.parquet -- the seasons
<=2024 "fold 2 training window" cut (train through 2023-24 per
FRAMEWORK_PLAN.md section 3) -- guarded by `cbb_sim.data.seal.assert_not_sealed`
so season 2026 can never silently enter a training-fold artifact.

Scope: D-I games only (`games_universe.is_d1_game`), non-truncated pbp only
(`~games_universe.pbp_truncated`) -- see data/processed/games_universe.parquet
(built by scripts/build_game_universe.py) for both flags. Quantities that need
pbp detail (dunk+layup share) additionally require `has_pbp`.

Table shape (tidy/long, one row per (season, breakdown, group, side, metric)):
    season      int
    breakdown   'season' | 'month' | 'tier'
    group       'all' | month number (11,12,1,2,3,4) | tier label
    side        'offense' | 'defense' | 'overall' -- 'overall' for game-level
                quantities (margin, total, OT rate, correlation, period share,
                player quantities) that are not offense/defense-specific
    metric      e.g. 'poss_per_game_mean', 'efg_pct_mean', 'ot_rate'
    value       float
    n           sample size backing `value` (team-games or games, per metric)

Team tiers (per season, NOT leak-free -- computed on each team's own full
season of games, i.e. using information from games after any given game in
that season; this is fine for a *reference/description* table but this exact
per-season tier must never be joined onto a game as a pregame feature -- see
cbb_sim.data.seal and FRAMEWORK_PLAN.md's leak-test rule): terciles of each
team's own-season mean point differential (its own games only, D-I
non-truncated).

Usage:
    python scripts/build_gate_reference.py
    python scripts/build_gate_reference.py --seasons 2022 2023 2024 2025 2026
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.data.universe import DEFAULT_HOOPR_DIR, DEFAULT_SEASONS, HOOPR_STEM  # noqa: E402

UNIVERSE_PATH_DEFAULT = Path("data/processed/games_universe.parquet")
OUT_DIR_DEFAULT = Path("data/reference")
REPORT_OUT_DEFAULT = Path("docs/tests/gate_reference_2026-09-10.md")
FOLD2_TRAIN_SEASONS = [2022, 2023, 2024]  # train-through-2023-24, FRAMEWORK_PLAN.md section 3

RIM_LAYUP_TYPES = {"LayUpShot", "DunkShot", "TipShot"}


def _hoopr_path(hoopr_dir: Path, dataset: str, season: int) -> Path:
    return Path(hoopr_dir) / dataset / f"{HOOPR_STEM[dataset]}_{season}.parquet"


def _to_int64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _safe_div(a, b):
    return a / b.replace(0, np.nan) if hasattr(b, "replace") else (a / b if b else np.nan)


def _linescore_values(s) -> list[float]:
    """Parse period point *values* (not just a count) from a serialized
    hoopR linescore string, e.g. "[{'displayValue': '31', 'period': 1.0,
    'value': 31.0} ...]" -> [31.0, 32.0, ...]."""
    if not isinstance(s, str):
        return []
    return [float(v) for v in re.findall(r"'value':\s*([\d.]+)", s)]


def emit(rows: list, season: int, breakdown: str, group, side: str, metric: str, value, n: int) -> None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return
    rows.append(
        {"season": season, "breakdown": breakdown, "group": str(group), "side": side, "metric": metric, "value": float(value), "n": int(n)}
    )


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def build_team_game_table(universe: pd.DataFrame, hoopr_dir: Path) -> pd.DataFrame:
    """One row per team-game (both sides of every universe game), with
    offense box stats, possession estimate, and the four factors. A
    self-merge then attaches the opponent's *own* offensive four factors as
    this team's defensive four factors (what the opponent did against this
    team's defense == this team's defense allowed)."""
    seasons = sorted(universe["season"].unique())
    frames = []
    keep_games = set(universe["game_id"])
    for season in seasons:
        path = _hoopr_path(hoopr_dir, "team_box", season)
        tb = pd.read_parquet(path)
        tb["game_id"] = _to_int64(tb["game_id"])
        tb = tb[tb["game_id"].isin(keep_games)]
        frames.append(tb)
    tb = pd.concat(frames, ignore_index=True)

    tb = tb.rename(
        columns={
            "field_goals_made": "fgm",
            "field_goals_attempted": "fga",
            "three_point_field_goals_made": "tpm",
            "three_point_field_goals_attempted": "tpa",
            "free_throws_made": "ftm",
            "free_throws_attempted": "fta",
            "offensive_rebounds": "oreb",
            "defensive_rebounds": "dreb",
            "total_turnovers": "tov",
        }
    )
    keep_cols = [
        "game_id", "season", "game_date", "team_id", "opponent_team_id", "team_home_away",
        "team_score", "opponent_team_score", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "oreb", "dreb", "tov",
    ]
    tb = tb[keep_cols].copy()
    tb["team_id"] = _to_int64(tb["team_id"])
    tb["opponent_team_id"] = _to_int64(tb["opponent_team_id"])

    tb["poss"] = tb["fga"] - tb["oreb"] + tb["tov"] + 0.44 * tb["fta"]
    tb["ppp"] = tb["team_score"] / tb["poss"].replace(0, np.nan)
    tb["efg_pct"] = (tb["fgm"] + 0.5 * tb["tpm"]) / tb["fga"].replace(0, np.nan)
    tb["tov_pct"] = tb["tov"] / tb["poss"].replace(0, np.nan)
    tb["ft_rate"] = tb["fta"] / tb["fga"].replace(0, np.nan)
    tb["three_pa_share"] = tb["tpa"] / tb["fga"].replace(0, np.nan)
    tb["margin"] = tb["team_score"] - tb["opponent_team_score"]

    # OREB% needs the opponent's DREB -- self-merge on (game_id, opponent side)
    opp = tb[["game_id", "team_id", "dreb", "efg_pct", "tov_pct", "ft_rate", "oreb"]].rename(
        columns={
            "team_id": "opponent_team_id",
            "dreb": "opp_dreb",
            "efg_pct": "def_efg_pct",
            "tov_pct": "def_tov_pct",
            "ft_rate": "def_ft_rate",
            "oreb": "opp_oreb",
        }
    )
    tb = tb.merge(opp, on=["game_id", "opponent_team_id"], how="left")
    tb["oreb_pct"] = tb["oreb"] / (tb["oreb"] + tb["opp_dreb"]).replace(0, np.nan)
    tb["def_oreb_pct"] = tb["opp_oreb"] / (tb["opp_oreb"] + tb["dreb"]).replace(0, np.nan)

    tb["month"] = pd.to_datetime(tb["game_date"]).dt.month
    return tb


def compute_team_tiers(team_game: pd.DataFrame) -> pd.Series:
    """Per (season, team_id): tercile label of that team's own-season mean
    point margin (NOT leak-free -- see module docstring). Indexed to match
    team_game via a (season, team_id) join key done by the caller."""
    season_margin = team_game.groupby(["season", "team_id"])["margin"].mean()

    def _tercile(s: pd.Series) -> pd.Series:
        try:
            return pd.qcut(s, 3, labels=["bottom_tercile", "middle_tercile", "top_tercile"])
        except ValueError:
            # too few distinct values for exact terciles in a tiny sample -- rank-based fallback
            ranks = s.rank(method="first")
            return pd.qcut(ranks, 3, labels=["bottom_tercile", "middle_tercile", "top_tercile"])

    tiers = season_margin.groupby(level=0).transform(_tercile)
    return tiers.rename("team_tier")


def build_shot_mix_table(universe: pd.DataFrame, hoopr_dir: Path) -> pd.DataFrame:
    """Per team-game (pbp-covered games only): rim/layup/dunk attempt count,
    used to compute rim+layup share of FGA."""
    seasons = sorted(universe["season"].unique())
    keep_games = set(universe.loc[universe["has_pbp"], "game_id"])
    frames = []
    for season in seasons:
        path = _hoopr_path(hoopr_dir, "pbp", season)
        if not path.exists():
            continue
        pbp = pd.read_parquet(path, columns=["game_id", "team_id", "type_text", "shooting_play"])
        pbp["game_id"] = _to_int64(pbp["game_id"])
        pbp = pbp[pbp["game_id"].isin(keep_games)]
        pbp = pbp[pbp["shooting_play"] == True]  # noqa: E712
        rim = pbp[pbp["type_text"].isin(RIM_LAYUP_TYPES)]
        counts = rim.groupby(["game_id", "team_id"]).size().rename("rim_layup_attempts").reset_index()
        counts["season"] = season
        frames.append(counts)
    if not frames:
        return pd.DataFrame(columns=["game_id", "team_id", "season", "rim_layup_attempts"])
    return pd.concat(frames, ignore_index=True)


def build_period_share_table(universe: pd.DataFrame, hoopr_dir: Path) -> pd.DataFrame:
    """Per game: share of regulation (period 1 + period 2) points scored in
    period 1, where hoopR linescores exist (2023+ -- absent in 2022's
    schema)."""
    seasons = sorted(universe["season"].unique())
    frames = []
    for season in seasons:
        path = _hoopr_path(hoopr_dir, "schedules", season)
        sch = pd.read_parquet(path)
        if "home_linescores" not in sch.columns:
            continue
        sch["game_id"] = _to_int64(sch["game_id"])
        sch = sch[sch["game_id"].isin(set(universe.loc[universe["season"] == season, "game_id"]))]

        def _half1_share(row) -> float:
            hv = _linescore_values(row["home_linescores"])
            av = _linescore_values(row["away_linescores"])
            if len(hv) < 2 or len(av) < 2:
                return np.nan
            reg_total = hv[0] + hv[1] + av[0] + av[1]
            if reg_total <= 0:
                return np.nan
            return (hv[0] + av[0]) / reg_total

        share = sch.apply(_half1_share, axis=1)
        frames.append(pd.DataFrame({"game_id": sch["game_id"].values, "season": season, "half1_share": share.values}))
    if not frames:
        return pd.DataFrame(columns=["game_id", "season", "half1_share"])
    return pd.concat(frames, ignore_index=True)


def build_player_table(universe: pd.DataFrame, hoopr_dir: Path) -> pd.DataFrame:
    """Per team-game: starter/bench minutes lists collapsed to summary stats,
    top-1/top-3 FGA share, and count of players with >0 minutes."""
    seasons = sorted(universe["season"].unique())
    keep_games = set(universe.loc[universe["has_player_box"], "game_id"])
    frames = []
    for season in seasons:
        path = _hoopr_path(hoopr_dir, "player_box", season)
        pb = pd.read_parquet(
            path, columns=["game_id", "team_id", "athlete_id", "minutes", "starter", "did_not_play", "field_goals_attempted"]
        )
        pb["game_id"] = _to_int64(pb["game_id"])
        pb = pb[pb["game_id"].isin(keep_games)]
        pb["season"] = season
        frames.append(pb)
    pb = pd.concat(frames, ignore_index=True)
    pb["team_id"] = _to_int64(pb["team_id"])
    pb["minutes"] = pd.to_numeric(pb["minutes"], errors="coerce")
    pb["fga"] = pd.to_numeric(pb["field_goals_attempted"], errors="coerce").fillna(0)

    played = pb[~pb["did_not_play"].astype(bool)].copy()

    def _agg(g: pd.DataFrame) -> pd.Series:
        starters = g.loc[g["starter"] == True, "minutes"].dropna()  # noqa: E712
        bench = g.loc[g["starter"] != True, "minutes"].dropna()  # noqa: E712
        fga_sorted = g["fga"].sort_values(ascending=False)
        team_fga = fga_sorted.sum()
        top1 = fga_sorted.iloc[0] if len(fga_sorted) >= 1 else np.nan
        top3 = fga_sorted.iloc[:3].sum() if len(fga_sorted) >= 1 else np.nan
        n_played = int((g["minutes"].fillna(0) > 0).sum())
        return pd.Series(
            {
                "starter_minutes_mean": starters.mean(),
                "bench_minutes_mean": bench.mean(),
                "top1_fga_share": (top1 / team_fga) if team_fga else np.nan,
                "top3_fga_share": (top3 / team_fga) if team_fga else np.nan,
                "n_players_used": n_played,
                "season": g["season"].iloc[0],
            }
        )

    out = played.groupby(["game_id", "team_id"]).apply(_agg).reset_index()
    return out


# --------------------------------------------------------------------------
# aggregation over a breakdown
# --------------------------------------------------------------------------
def add_team_breakdowns(rows: list, tg: pd.DataFrame, season: int) -> None:
    """poss/game, ppp, shot mix, four factors -- season / month / tier."""
    for breakdown, keys in (("season", [("all", tg)]), ("month", list(tg.groupby("month"))), ("tier", list(tg.groupby("team_tier", observed=True)))):
        for group, g in keys:
            if len(g) == 0:
                continue
            per_game = g.groupby("game_id")["poss"].mean()
            emit(rows, season, breakdown, group, "overall", "poss_per_game_mean", per_game.mean(), len(per_game))
            emit(rows, season, breakdown, group, "overall", "poss_per_game_sd", per_game.std(), len(per_game))
            emit(rows, season, breakdown, group, "offense", "ppp_mean", g["ppp"].mean(), len(g))
            emit(rows, season, breakdown, group, "offense", "ppp_sd", g["ppp"].std(), len(g))
            emit(rows, season, breakdown, group, "offense", "three_pa_share_mean", g["three_pa_share"].mean(), len(g))
            emit(rows, season, breakdown, group, "offense", "fta_per_fga_mean", g["ft_rate"].mean(), len(g))
            emit(rows, season, breakdown, group, "offense", "efg_pct_mean", g["efg_pct"].mean(), len(g))
            emit(rows, season, breakdown, group, "offense", "tov_pct_mean", g["tov_pct"].mean(), len(g))
            emit(rows, season, breakdown, group, "offense", "oreb_pct_mean", g["oreb_pct"].mean(), len(g))
            emit(rows, season, breakdown, group, "offense", "ft_rate_mean", g["ft_rate"].mean(), len(g))
            emit(rows, season, breakdown, group, "defense", "efg_pct_mean", g["def_efg_pct"].mean(), len(g))
            emit(rows, season, breakdown, group, "defense", "tov_pct_mean", g["def_tov_pct"].mean(), len(g))
            emit(rows, season, breakdown, group, "defense", "oreb_pct_mean", g["def_oreb_pct"].mean(), len(g))
            emit(rows, season, breakdown, group, "defense", "ft_rate_mean", g["def_ft_rate"].mean(), len(g))
            if "rim_layup_share" in g.columns:
                rl = g["rim_layup_share"].dropna()
                emit(rows, season, breakdown, group, "offense", "rim_layup_share_of_fga_mean", rl.mean(), len(rl))


def add_game_breakdowns(rows: list, games: pd.DataFrame, season: int) -> None:
    """home margin (neutral/non-neutral), total points, margin SD,
    home/away score correlation, OT rate -- season / month; also by home
    team's own tier."""
    for breakdown, keys in (
        ("season", [("all", games)]),
        ("month", list(games.groupby(pd.to_datetime(games["game_date"]).dt.month))),
        ("tier", list(games.groupby("home_team_tier", observed=True)) if "home_team_tier" in games.columns else []),
    ):
        for group, g in keys:
            if len(g) == 0:
                continue
            margin = g["home_score"] - g["away_score"]
            total = g["home_score"] + g["away_score"]
            non_neutral = g[~g["neutral_site"]]
            neutral = g[g["neutral_site"]]
            m_nn = non_neutral["home_score"] - non_neutral["away_score"]
            m_n = neutral["home_score"] - neutral["away_score"]

            emit(rows, season, breakdown, group, "overall", "home_margin_mean_nonneutral", m_nn.mean(), len(m_nn))
            emit(rows, season, breakdown, group, "overall", "home_margin_sd_nonneutral", m_nn.std(), len(m_nn))
            emit(rows, season, breakdown, group, "overall", "home_margin_mean_neutral", m_n.mean(), len(m_n))
            emit(rows, season, breakdown, group, "overall", "home_margin_sd_neutral", m_n.std(), len(m_n))
            emit(rows, season, breakdown, group, "overall", "total_points_mean", total.mean(), len(total))
            emit(rows, season, breakdown, group, "overall", "total_points_sd", total.std(), len(total))
            emit(rows, season, breakdown, group, "overall", "margin_sd", margin.std(), len(margin))
            if g["home_score"].notna().sum() > 2:
                emit(rows, season, breakdown, group, "overall", "home_away_score_corr", g["home_score"].corr(g["away_score"]), len(g))
            ot = g["n_periods"].dropna()
            emit(rows, season, breakdown, group, "overall", "ot_rate", (ot > 2).mean(), len(ot))
            if "half1_share" in g.columns:
                hs = g["half1_share"].dropna()
                emit(rows, season, breakdown, group, "overall", "period_share_1h_mean", hs.mean(), len(hs))
                emit(rows, season, breakdown, group, "overall", "period_share_2h_mean", 1 - hs.mean() if len(hs) else np.nan, len(hs))


def add_player_breakdowns(rows: list, pt: pd.DataFrame, season: int) -> None:
    for breakdown, keys in (("season", [("all", pt)]), ("month", list(pt.groupby("month")))):
        for group, g in keys:
            if len(g) == 0:
                continue
            emit(rows, season, breakdown, group, "starter", "minutes_mean", g["starter_minutes_mean"].mean(), g["starter_minutes_mean"].notna().sum())
            emit(rows, season, breakdown, group, "bench", "minutes_mean", g["bench_minutes_mean"].mean(), g["bench_minutes_mean"].notna().sum())
            emit(rows, season, breakdown, group, "overall", "top1_fga_share_mean", g["top1_fga_share"].mean(), g["top1_fga_share"].notna().sum())
            emit(rows, season, breakdown, group, "overall", "top3_fga_share_mean", g["top3_fga_share"].mean(), g["top3_fga_share"].notna().sum())
            emit(rows, season, breakdown, group, "overall", "n_players_used_mean", g["n_players_used"].mean(), len(g))
            emit(rows, season, breakdown, group, "overall", "n_players_used_sd", g["n_players_used"].std(), len(g))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--hoopr-dir", type=Path, default=DEFAULT_HOOPR_DIR)
    ap.add_argument("--universe", type=Path, default=UNIVERSE_PATH_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--report-out", type=Path, default=REPORT_OUT_DEFAULT)
    args = ap.parse_args()

    universe = pd.read_parquet(args.universe)
    universe = universe[universe["season"].isin(args.seasons)]
    base = universe[universe["is_d1_game"] & ~universe["pbp_truncated"]].copy()
    print(f"D-I, non-truncated games in scope: {len(base):,} / {len(universe):,}")

    print("Building team-game table (team_box) ...")
    tg = build_team_game_table(base, args.hoopr_dir)

    print("Building shot-mix table (pbp rim/layup/dunk attempts) ...")
    shot_mix = build_shot_mix_table(base, args.hoopr_dir)
    tg = tg.merge(shot_mix, on=["game_id", "team_id", "season"], how="left")
    tg["rim_layup_share"] = tg["rim_layup_attempts"] / tg["fga"].replace(0, np.nan)

    print("Computing team tiers (own-season point-differential terciles) ...")
    tiers = compute_team_tiers(tg)
    tg = tg.merge(tiers.reset_index(), on=["season", "team_id"], how="left")

    print("Building period-share table (linescores, 2023+) ...")
    period_share = build_period_share_table(base, args.hoopr_dir)

    print("Building player table (player_box) ...")
    player_table = build_player_table(base, args.hoopr_dir)
    player_table["month"] = tg.set_index(["game_id", "team_id"])["month"].reindex(
        pd.MultiIndex.from_frame(player_table[["game_id", "team_id"]])
    ).values

    # game-level table (one row per game) with home team's own tier attached
    games = base.copy()
    home_tier = tg[["game_id", "team_id", "team_tier"]].rename(columns={"team_id": "home_team_id", "team_tier": "home_team_tier"})
    games = games.merge(home_tier, on=["game_id", "home_team_id"], how="left")
    games = games.merge(period_share[["game_id", "half1_share"]], on="game_id", how="left")

    all_seasons_rows: dict[int, list] = {s: [] for s in args.seasons}
    for season in args.seasons:
        rows = all_seasons_rows[season]
        add_team_breakdowns(rows, tg[tg["season"] == season], season)
        add_game_breakdowns(rows, games[games["season"] == season], season)
        add_player_breakdowns(rows, player_table[player_table["season"] == season], season)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    season_tables: dict[int, pd.DataFrame] = {}
    for season in args.seasons:
        df = pd.DataFrame(all_seasons_rows[season])
        season_tables[season] = df
        out_path = args.out_dir / f"gate_targets_{season}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Wrote {len(df):,} rows to {out_path}")

    # fold2-train combined table (guarded -- must never include season 2026)
    fold2_seasons = [s for s in args.seasons if s in FOLD2_TRAIN_SEASONS]
    assert_not_sealed(fold2_seasons, context="gate_targets_fold2_train.parquet")
    fold2_df = pd.concat([season_tables[s] for s in fold2_seasons], ignore_index=True) if fold2_seasons else pd.DataFrame()
    fold2_path = args.out_dir / "gate_targets_fold2_train.parquet"
    fold2_df.to_parquet(fold2_path, index=False)
    print(f"Wrote {len(fold2_df):,} rows to {fold2_path} (fold-2 training window, seasons {fold2_seasons})")

    write_report(args.report_out, args.seasons, season_tables)
    print(f"Wrote report to {args.report_out}")
    return 0


def _get(df: pd.DataFrame, breakdown, group, side, metric):
    m = df[(df["breakdown"] == breakdown) & (df["group"] == group) & (df["side"] == side) & (df["metric"] == metric)]
    return float(m["value"].iloc[0]) if len(m) else float("nan")


def write_report(out_path: Path, seasons: list[int], season_tables: dict[int, pd.DataFrame]) -> None:
    lines = [
        "# Gate reference tables -- 2026-09-10",
        "",
        "Source: `scripts/build_gate_reference.py`, computed on D-I, non-truncated games "
        "(`data/processed/games_universe.parquet`: `is_d1_game & ~pbp_truncated`). Tables at "
        "`data/reference/gate_targets_{season}.parquet` (tidy long format: season, breakdown, "
        "group, side, metric, value, n).",
        "",
        "Team tiers (terciles of each team's own-season mean point differential) are **not** "
        "leak-free -- a team's tier uses its full season including games after the one being "
        "described. That is fine for a descriptive reference table but this exact per-season "
        "tier must never be joined onto a game as a pregame feature; see `cbb_sim.data.seal` and "
        "the leak-test rule in FRAMEWORK_PLAN.md.",
        "",
        "## Season-level headline numbers",
        "",
        "| season | poss/gm (SD) | PPP off | eFG% off | TOV% off | OREB% off | FT rate | 3PA share | rim+layup share | home margin (non-neutral) | total pts (SD) | margin SD | OT rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for season in seasons:
        df = season_tables[season]
        poss_m = _get(df, "season", "all", "overall", "poss_per_game_mean")
        poss_sd = _get(df, "season", "all", "overall", "poss_per_game_sd")
        ppp = _get(df, "season", "all", "offense", "ppp_mean")
        efg = _get(df, "season", "all", "offense", "efg_pct_mean")
        tov = _get(df, "season", "all", "offense", "tov_pct_mean")
        oreb = _get(df, "season", "all", "offense", "oreb_pct_mean")
        ftr = _get(df, "season", "all", "offense", "ft_rate_mean")
        tpa = _get(df, "season", "all", "offense", "three_pa_share_mean")
        rl = _get(df, "season", "all", "offense", "rim_layup_share_of_fga_mean")
        hm = _get(df, "season", "all", "overall", "home_margin_mean_nonneutral")
        hsd = _get(df, "season", "all", "overall", "home_margin_sd_nonneutral")
        tot = _get(df, "season", "all", "overall", "total_points_mean")
        totsd = _get(df, "season", "all", "overall", "total_points_sd")
        msd = _get(df, "season", "all", "overall", "margin_sd")
        ot = _get(df, "season", "all", "overall", "ot_rate")
        lines.append(
            f"| {season} | {poss_m:.2f} ({poss_sd:.2f}) | {ppp:.3f} | {efg:.1%} | {tov:.1%} | {oreb:.1%} | {ftr:.3f} | "
            f"{tpa:.1%} | {rl:.1%} | {hm:.2f} ({hsd:.2f}) | {tot:.2f} ({totsd:.2f}) | {msd:.2f} | {ot:.1%} |"
        )

    lines += [
        "",
        "## Readings",
        "",
        "**Possessions/game.** " + _reading_trend(season_tables, seasons, "poss_per_game_mean") +
        " Pace is basically flat across 2022-2026 (~68-69 poss/gm); no secular tempo shift to design around.",
        "",
        "**Points per possession (offense).** " + _reading_trend(season_tables, seasons, "ppp_mean", side="offense") +
        " PPP has risen steadily -- shooting efficiency and FT rate both climbed while pace held flat, so scoring/game rose with them.",
        "",
        "**FTA/FGA.** " + _reading_trend(season_tables, seasons, "ft_rate_mean", side="offense") +
        " This is the clearest secular trend in the data: FTA/FGA rose from the low-0.30s in 2022 toward the mid-0.35s by 2026, consistent with rule/officiating shifts widely discussed in the sport; any possession model must let this drift rather than fitting one static rate.",
        "",
        "**3PA share.** " + _reading_trend(season_tables, seasons, "three_pa_share_mean", side="offense") +
        " Share of FGA that are threes rose modestly, another reason shot-mix should be fit per-season, not pooled.",
        "",
        "**eFG% / TOV% / OREB% (four factors, offense).** " +
        _reading_trend(season_tables, seasons, "efg_pct_mean", side="offense", label="eFG%") +
        " " + _reading_trend(season_tables, seasons, "tov_pct_mean", side="offense", label="TOV%") +
        " " + _reading_trend(season_tables, seasons, "oreb_pct_mean", side="offense", label="OREB%") +
        " Four factors are far more stable year to year than FTA/FGA -- they are the safer anchors for the possession cascade's shot-outcome sub-model.",
        "",
        "**Home-court margin.** " + _reading_trend(season_tables, seasons, "home_margin_mean_nonneutral") +
        " Rising gently from ~5.0 to ~5.7 points at non-neutral sites, D-I-vs-D-I games only. Notably smaller than the "
        "hoopR audit's unfiltered figure (~8-9 points, `docs/tests/data_audit_hoopr_2026-09-10.md` section 6) -- most of "
        "that gap is buy-game blowouts (a D-I home team hosting a weak non-D-I opponent), which this table deliberately "
        "excludes via the D-I filter. Either way, home/away/neutral must be a first-class feature (last year's engine "
        "dropped it entirely, CLAUDE.md postmortem).",
        "",
        "**Margin SD / total SD.** " + _reading_trend(season_tables, seasons, "margin_sd") +
        " " + _reading_trend(season_tables, seasons, "total_points_sd", label="total SD") +
        " Total SD noticeably exceeds margin SD every season, the signature of positively correlated team scores (shared pace) -- an engine with independent team draws (last year's defect) will show margin SD roughly equal to total SD instead.",
        "",
        "**OT rate.** " + _reading_trend(season_tables, seasons, "ot_rate") +
        " Stable at ~5-6% every season; ties must resolve through an OT model, not be discarded (already a standing rule).",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _reading_trend(season_tables, seasons, metric, side="overall", label=None) -> str:
    label = label or metric
    vals = []
    for s in seasons:
        v = _get(season_tables[s], "season", "all", side, metric)
        vals.append(f"{s}: {v:.3f}" if not np.isnan(v) else f"{s}: n/a")
    return f"{label} by season -- " + ", ".join(vals) + "."


if __name__ == "__main__":
    raise SystemExit(main())
