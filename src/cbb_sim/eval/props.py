"""
props.py -- player-prop market grading (SIM_GUARDRAILS.md G10 extended to
player markets; `docs/postmortem/05_cfb_methodology_extract.md` section 6,
`grade_market_props.py`).

NO PROPS-LINES DATA EXISTS YET for this project (CLAUDE.md data rules ban
scraping PrizePicks/Underdog/DraftKings directly, and no CBBD props endpoint
has been pulled). This module ships anyway, against a DOCUMENTED schema
(`PROPS_LINES_SCHEMA` below), so grading is a one-line CLI call the day a real
props-lines parquet lands -- proven correct now with a synthetic-lines test
(`tests/test_eval.py::test_props_synthetic_grading`).

PROPS-LINES PARQUET SCHEMA (one row per (cbbd_game_id, athlete_id, stat,
provider, snapshot)):

    cbbd_game_id   int      CBBD game id -- same join key `reference.load_lines`
                            uses for game-level lines.
    athlete_id     int      player id, matched to `players.parquet`'s own
                            `athlete_id`. Cross-engine player identity is the
                            existing `data/processed/player_crosswalk.parquet`
                            (hoopR <-> CBBD); this module does not itself
                            resolve identity, it assumes the join key already
                            agrees (same assumption `players.parquet`'s
                            contract makes).
    stat           str      one of `PROP_STATS` ("pts", "reb", "ast", "fga",
                            "fg3a", "fta") -- matches `players.parquet`'s own
                            stat columns 1:1, so no stat-name translation
                            table is needed.
    provider       str      sportsbook name, `reference.PROVIDER_PREFERENCE`
                            convention.
    snapshot       str      "open" or "close".
    line           float    the book's posted number for this prop (e.g. 15.5).
    over_odds      float    American odds for the OVER at `line`.
    under_odds     float    American odds for the UNDER at `line`.

SETTLEMENT VS DE-VIG (methodology doc section 6, quoted): "de-vig is for the
probability comparison only, never for settlement." `--settle book` (default)
settles every prop at the row's own REAL posted `line` and prices the market
probability by de-vigging `(over_odds, under_odds)` AT THAT SAME LINE.
`--settle fair` is kept only as a separately tagged audit mode -- it prices
off a de-vigged "fair" line synthesised across providers/snapshots rather than
one real book's number, and its rows are always labelled `settle=fair` in the
output so they can never be mistaken for a real card.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cbb_sim.eval import market as M
from cbb_sim.eval import reference as ref_mod

PROP_STATS: tuple[str, ...] = ("pts", "reb", "ast", "fga", "fg3a", "fta")
REQUIRED_PROPS_LINES_COLUMNS: tuple[str, ...] = (
    "cbbd_game_id", "athlete_id", "stat", "provider", "snapshot", "line", "over_odds", "under_odds",
)


class PropsContractError(RuntimeError):
    pass


def validate_props_lines(df: pd.DataFrame, strict: bool = True) -> list[str]:
    problems = [f"props lines missing required column '{c}'" for c in REQUIRED_PROPS_LINES_COLUMNS
                if c not in df.columns]
    bad_stats = set(df["stat"].unique()) - set(PROP_STATS) if "stat" in df.columns else set()
    if bad_stats:
        problems.append(f"props lines has stat value(s) outside PROP_STATS: {sorted(bad_stats)}")
    if problems and strict:
        raise PropsContractError("; ".join(problems))
    return problems


def load_props_lines(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_props_lines(df, strict=True)
    return df


def best_provider_snapshot(lines: pd.DataFrame, snapshot: str = "close",
                            provider_preference: tuple[str, ...] = ref_mod.PROVIDER_PREFERENCE) -> pd.DataFrame:
    """One row per (cbbd_game_id, athlete_id, stat): the best-available
    provider at `snapshot`, same preference-order convention as
    `reference.load_lines`."""
    d = lines[lines["snapshot"] == snapshot].copy()
    d = d[d["provider"].isin(provider_preference)]
    rank = {p: i for i, p in enumerate(provider_preference)}
    d["_rank"] = d["provider"].map(rank)
    d = d.sort_values(["cbbd_game_id", "athlete_id", "stat", "_rank"], kind="mergesort")
    return d.drop_duplicates(["cbbd_game_id", "athlete_id", "stat"], keep="first").drop(columns=["_rank"])


# ---------------------------------------------------------------------------
# sim side: empirical P(over) from the seed distribution
# ---------------------------------------------------------------------------
def sim_prop_distribution(players: pd.DataFrame, game_id_to_cbbd: pd.Series) -> pd.DataFrame:
    """Long frame: one row per (cbbd_game_id, athlete_id, stat, seed), value.
    `game_id_to_cbbd` maps the contract's own `game_id` to `cbbd_game_id`
    (from `reference.load_actual_games`)."""
    long_rows = []
    for stat in PROP_STATS:
        if stat not in players.columns:
            continue
        d = players[["game_id", "seed", "athlete_id", stat]].rename(columns={stat: "value"})
        d["stat"] = stat
        long_rows.append(d)
    if not long_rows:
        return pd.DataFrame(columns=["game_id", "seed", "athlete_id", "stat", "value", "cbbd_game_id"])
    long = pd.concat(long_rows, ignore_index=True)
    long["cbbd_game_id"] = long["game_id"].map(game_id_to_cbbd)
    return long.dropna(subset=["cbbd_game_id"])


def empirical_p_over(sim_long: pd.DataFrame) -> pd.DataFrame:
    """P(over) = mean(value > line) + 0.5*mean(value == line) (push mass split),
    per (cbbd_game_id, athlete_id, stat), given `sim_long` already merged with
    `line`. NOTE: this is a plain empirical proportion over seeds; the CFB
    prototype (methodology doc section 6) additionally shrinks toward a
    normal approximation with a pooled cross-game residual SD (SHRINK_K=20
    seeds of pseudo-weight) for low-seed-count stability. That shrinkage is
    NOT implemented here -- a documented simplification, not a silent one --
    and is the natural next refinement once a real seed-count study exists."""
    g = sim_long.groupby(["cbbd_game_id", "athlete_id", "stat"])
    out = g.apply(lambda d: pd.Series({
        "p_over": float((d["value"] > d["line"]).mean() + 0.5 * (d["value"] == d["line"]).mean()),
        "sim_mean": float(d["value"].mean()),
        "n_seeds": int(len(d)),
    }), include_groups=False)
    return out.reset_index()


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
@dataclass
class PropsGradeResult:
    settle_mode: str
    n: int
    push_rate: float
    whole_number_line_share: float
    zero_usage_share: float
    calibration: pd.DataFrame
    model_brier: float
    market_brier: float
    roi_table: pd.DataFrame
    bootstrap: dict
    by_stat: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def grade_props(players: pd.DataFrame, props_lines: pd.DataFrame, actual_games: pd.DataFrame,
                 actual_player_stats: pd.DataFrame, tol: dict, settle: str = "book",
                 n_boot: int = 1000) -> PropsGradeResult:
    """`actual_player_stats`: one row per (cbbd_game_id, athlete_id, stat,
    value) of the REAL outcome, already identity-verified upstream (this
    function does not itself resolve player identity -- see module
    docstring)."""
    game_id_to_cbbd = actual_games.set_index("game_id")["cbbd_game_id"]
    sim_long = sim_prop_distribution(players, game_id_to_cbbd)

    lines = best_provider_snapshot(props_lines, snapshot="close")
    if settle == "fair":
        # audit mode only: synthesise a cross-provider fair line rather than one book's real number.
        lines = props_lines.groupby(["cbbd_game_id", "athlete_id", "stat"], as_index=False).agg(
            line=("line", "mean"), over_odds=("over_odds", "mean"), under_odds=("under_odds", "mean"),
            provider=("provider", "first"),
        )

    merged = sim_long.merge(lines[["cbbd_game_id", "athlete_id", "stat", "line", "over_odds", "under_odds"]],
                             on=["cbbd_game_id", "athlete_id", "stat"], how="inner")
    p_over = empirical_p_over(merged)
    p_over = p_over.merge(lines[["cbbd_game_id", "athlete_id", "stat", "line", "over_odds", "under_odds"]],
                           on=["cbbd_game_id", "athlete_id", "stat"], how="left")
    p_over = p_over.merge(actual_player_stats, on=["cbbd_game_id", "athlete_id", "stat"], how="inner",
                           suffixes=("", "_actual"))

    p_hi = M.ml_to_prob(p_over["over_odds"])
    p_lo = M.ml_to_prob(p_over["under_odds"])
    p_over["market_p_over"] = p_hi / (p_hi + p_lo)  # de-vig: probability comparison ONLY

    p_over["went_over"] = (p_over["value"] > p_over["line"]).astype(float)
    p_over.loc[p_over["value"] == p_over["line"], "went_over"] = np.nan  # push, excluded from Brier/ROI

    n = int(len(p_over))
    push_rate = float((p_over["value"] == p_over["line"]).mean()) if n else float("nan")
    whole_share = float((p_over["line"] % 1 == 0).mean()) if n else float("nan")
    zero_usage_share = float((p_over["sim_mean"] == 0).mean()) if n else float("nan")

    decided = p_over.dropna(subset=["went_over"]).copy()
    model_brier = float(((decided["p_over"] - decided["went_over"]) ** 2).mean()) if len(decided) else float("nan")
    market_brier = float(((decided["market_p_over"] - decided["went_over"]) ** 2).mean()) if len(decided) else float("nan")
    calib = M.calibration_deciles(decided["p_over"], decided["market_p_over"], decided["went_over"]) if len(decided) else pd.DataFrame()

    edge = decided["p_over"] - decided["market_p_over"]
    bet_over = edge > 0
    chosen_odds = np.where(bet_over, decided["over_odds"], decided["under_odds"])
    won = np.where(bet_over, decided["went_over"] == 1, decided["went_over"] == 0)
    profit = M.ml_profit_if_win(chosen_odds)
    pnl = np.where(won, profit, -1.0)
    roi_table, _ = M.ml_edge_table(edge, decided["went_over"].map({1.0: 1, 0.0: 0}), decided["over_odds"], decided["under_odds"])
    boot = M.bootstrap_roi(pnl, n_boot)

    by_stat_rows = []
    for stat, sub in decided.groupby("stat"):
        by_stat_rows.append({
            "stat": stat, "n": int(len(sub)),
            "brier_model": float(((sub["p_over"] - sub["went_over"]) ** 2).mean()),
            "brier_market": float(((sub["market_p_over"] - sub["went_over"]) ** 2).mean()),
        })
    by_stat = pd.DataFrame(by_stat_rows)

    notes = []
    if settle == "fair":
        notes.append("settle=fair: AUDIT MODE ONLY. Bets are priced/settled off a cross-provider "
                     "average line, not a real book's posted number -- never treat this run's ROI as "
                     "a real card (methodology doc section 6: 'de-vig is for the probability "
                     "comparison only, never for settlement').")
    if whole_share > 0.05:
        notes.append(f"whole-number-line share {whole_share:.4f} is elevated (>5%); this was the "
                     "tell that caught the settle-at-fair-line bug in the CFB prototype -- check the "
                     "lines source is real book lines, not a rounded/synthetic consensus.")

    return PropsGradeResult(
        settle_mode=settle, n=n, push_rate=push_rate, whole_number_line_share=whole_share,
        zero_usage_share=zero_usage_share, calibration=calib, model_brier=model_brier,
        market_brier=market_brier, roi_table=roi_table, bootstrap=boot, by_stat=by_stat, notes=notes,
    )
