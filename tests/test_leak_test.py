"""Synthetic tests for `cbb_sim.analysis.leak_test` -- the ported INV-45 statistic.

The point of the harness (CLAUDE.md: "Standing rule: backtests must be
honest") is that it must actually be ABLE to catch a leak, not just look
plausible. These tests build synthetic (team, season, game) panels with a
known-leaked feature and a known-legitimate one and check the statistic
recovers the right verdict on each -- the same sanity check
`docs/tests/leak_test_kenpom_2026-09-10.md` relies on before trusting the
real-data reading.

Math behind the fixed thresholds: for iid margins X_t (mean 0, variance V),
d_t = X_t - X_{t-1}. Cov(d_t, X_t) = V (X_{t-1} independent of X_t), and
Var(d_t) = 2V, so corr(d_t, X_t) = 1/sqrt(2) ~= 0.707 -- exactly the
magnitude the CFB postmortem measured on its leaked rating columns (0.707
FEI, 0.839 Elo). Using `feature = margin` itself as the synthetic leak
reproduces that number by construction, which is why the assertions below
use a loose band around 0.707 rather than merely ">gate".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cbb_sim.analysis import leak_test as lt

SEED = 20260910
N_TEAMS = 40
N_GAMES = 20


def make_synthetic_panel(n_teams: int = N_TEAMS, n_games: int = N_GAMES, season: int = 2099, seed: int = SEED) -> pd.DataFrame:
    """One row per (team, game): iid margins, no real team-quality signal at
    all -- any correlation the test finds has to come from how a feature is
    CONSTRUCTED from the margins, not from genuine team strength, which is
    the point (isolates the leak channel from ordinary predictive skill)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_teams):
        team = f"Team{i:03d}"
        margins = rng.normal(0.0, 10.0, size=n_games)
        dates = pd.date_range("2099-11-01", periods=n_games, freq="3D")
        for g, (m, d) in enumerate(zip(margins, dates)):
            rows.append({"team": team, "season": season, "game_id": i * 1000 + g, "game_date": d, "margin": m})
    return pd.DataFrame(rows)


def test_same_game_margin_feature_is_flagged_leak():
    """`feature = this game's own margin` is the most literal possible leak
    (a column that IS the outcome it precedes). Expect as-joined ~= 0.707
    and a LEAK verdict, comfortably over the 0.15 gate."""
    panel = make_synthetic_panel()
    panel["feature_leak"] = panel["margin"]

    result = lt.leak_test_column(
        panel, "feature_leak", team_col="team", season_col="season",
        date_col="game_date", margin_col="margin",
    )

    assert result["n"] >= lt.MIN_N
    assert not result["static"]
    assert 0.6 < result["corr_asjoined"] < 0.8, result["corr_asjoined"]
    assert result["verdict"] == "LEAK"


def test_previous_game_margin_feature_passes():
    """`feature[t] = margin[t-1]` (yesterday's own result -- a completely
    legitimate pregame value): as-joined should be near zero (margin[t] is
    independent of margin[t-1] and margin[t-2] in this iid synthetic panel)
    and the verdict should be `pass`. `legit-update` should show the healthy
    signature at ~0.707 -- numerically the same value the leaked feature's
    as-joined showed, which is the module's built-in positive control."""
    panel = make_synthetic_panel().sort_values(["team", "game_date"]).reset_index(drop=True)
    panel["feature_lag"] = panel.groupby("team")["margin"].shift(1)

    result = lt.leak_test_column(
        panel, "feature_lag", team_col="team", season_col="season",
        date_col="game_date", margin_col="margin",
    )

    assert result["n"] >= lt.MIN_N
    assert not result["static"]
    assert abs(result["corr_asjoined"]) < lt.GATE, result["corr_asjoined"]
    assert result["verdict"] == "pass"
    assert 0.6 < result["corr_update"] < 0.8, result["corr_update"]


def test_same_season_aggregate_is_static_with_level_form_leak():
    """A same-season aggregate (here: the team's own mean margin over ALL of
    its games that season, including future ones relative to any given row)
    has zero within-season variance -- STATIC, change-form undefined -- but
    its LEVEL correlation with any one game's margin is inflated purely by
    construction (the aggregate is partly computed FROM that game). This is
    the CBBD `/ratings/adjusted`-joined-by-season pattern in
    docs/tests/leak_test_kenpom_2026-09-10.md Arm C."""
    panel = make_synthetic_panel(n_games=10)  # smaller n per team -> a larger, unambiguous 1/sqrt(n) effect
    panel["feature_static"] = panel.groupby("team")["margin"].transform("mean")

    result = lt.leak_test_column(
        panel, "feature_static", team_col="team", season_col="season",
        date_col="game_date", margin_col="margin",
    )

    assert result["static"] is True
    assert result["verdict"] == "STATIC"
    assert result["level_verdict"] == "LEAK (level-form)"
    assert abs(result["corr_level"]) > lt.GATE


def test_run_leak_test_produces_one_row_per_column_and_season():
    panel = make_synthetic_panel()
    panel["feature_leak"] = panel["margin"]
    panel["feature_lag"] = panel.groupby("team")["margin"].shift(1)

    res = lt.run_leak_test(panel, ["feature_leak", "feature_lag"], team_col="team",
                            season_col="season", date_col="game_date", margin_col="margin")

    # one season (2099) + the pooled "ALL" row, per column
    assert set(res["season"]) == {2099, "ALL"}
    assert len(res) == 2 * 2
    leak_row = res[(res["column"] == "feature_leak") & (res["season"] == "ALL")].iloc[0]
    pass_row = res[(res["column"] == "feature_lag") & (res["season"] == "ALL")].iloc[0]
    assert leak_row["verdict"] == "LEAK"
    assert pass_row["verdict"] == "pass"


def test_render_markdown_table_marks_leak_and_static():
    panel = make_synthetic_panel(n_games=10)
    panel["feature_leak"] = panel["margin"]
    panel["feature_static"] = panel.groupby("team")["margin"].transform("mean")

    res = lt.run_leak_test(panel, ["feature_leak", "feature_static"], team_col="team",
                            season_col="season", date_col="game_date", margin_col="margin")
    lines = lt.render_markdown_table(res)
    body = "\n".join(lines)
    assert "**LEAK**" in body
    assert "STATIC" in body
