"""Synthetic tests for `cbb_sim.analysis.variance` -- the ported sequential
nested R^2 decomposition (`docs/postmortem/05_cfb_methodology_extract.md`
section 5, `scripts/exp_variance_decomp_v1.py`).

The point of the harness is the same discipline as `tests/test_leak_test.py`:
build a panel with KNOWN variance shares by construction and check the
statistic recovers them within tolerance, rather than trusting that a
plausible-looking implementation is actually correct. Two shapes are tested
because both are load-bearing for the real study:

1. A genuinely nested nuisance structure (coach effect + team-within-coach
   effect + noise, known variances) -- checks the one-way AND sequential R^2
   numbers land near their analytic targets.
2. The CFB "team-beyond-coach ~= 0%" pattern itself, reproduced by
   construction: when team is a strict refinement of coach with NO
   additional team-level variance (one team per coach, or many teams per
   coach but no team-specific effect), team's SEQUENTIAL R^2 after coach must
   be ~0 even though team's ONE-WAY R^2 can look identical to coach's -- this
   is exactly the distinction the postmortem says CFB's decomposition made
   and CBB's re-run must not assume away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cbb_sim.analysis import variance as v

SEED = 20260910


def make_nested_panel(
    n_coaches: int = 30,
    teams_per_coach: int = 3,
    games_per_team: int = 200,
    sigma_coach: float = 3.0,
    sigma_team: float = 1.0,
    sigma_noise: float = 6.0,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """One row per (team, game): y = coach_effect + team_effect + noise, with
    team strictly nested within coach (each team has exactly one coach).
    Returns (panel, expected_shares) where expected_shares gives the
    analytic fraction of total variance each term contributes."""
    rng = np.random.default_rng(seed)
    coach_effects = {c: rng.normal(0.0, sigma_coach) for c in range(n_coaches)}

    team_of_coach: dict[int, int] = {}
    team_effects: dict[int, float] = {}
    tid = 0
    for c in range(n_coaches):
        for _ in range(teams_per_coach):
            team_of_coach[tid] = c
            team_effects[tid] = rng.normal(0.0, sigma_team)
            tid += 1

    rows = []
    for team_id, coach_id in team_of_coach.items():
        for _ in range(games_per_team):
            y = coach_effects[coach_id] + team_effects[team_id] + rng.normal(0.0, sigma_noise)
            rows.append({"y": y, "coach": coach_id, "team": team_id, "w": 1.0})
    panel = pd.DataFrame(rows)

    total_var = sigma_coach**2 + sigma_team**2 + sigma_noise**2
    expected = {
        "coach": sigma_coach**2 / total_var,
        "team": sigma_team**2 / total_var,
        "residual": sigma_noise**2 / total_var,
    }
    return panel, expected


def test_sequential_r2_recovers_known_nested_variance_shares():
    panel, expected = make_nested_panel()
    res = v.decompose(panel, "y", ["coach", "team"], weight_col="w")

    tol = 0.05
    assert abs(res["coach_sequential_R2"] - expected["coach"]) < tol, res
    assert abs(res["team_sequential_R2"] - expected["team"]) < tol, res
    assert abs(res["residual_R2"] - expected["residual"]) < tol, res

    # sequential shares must sum (with residual) to ~1.0
    total = res["coach_sequential_R2"] + res["team_sequential_R2"] + res["residual_R2"]
    assert abs(total - 1.0) < 1e-9, total

    assert res["coach_n_groups"] == 30
    assert res["team_n_groups"] == 90


def test_one_way_r2_close_to_sequential_when_groupings_are_uncorrelated():
    """Coach and team are drawn independently of each other's effect here
    (team nested in coach for labeling, but team's own effect is iid of
    coach's), so coach's one-way R^2 should already be close to its
    sequential R^2 -- there's little to reallocate."""
    panel, expected = make_nested_panel()
    res = v.decompose(panel, "y", ["coach", "team"], weight_col="w")
    assert abs(res["coach_R2"] - res["coach_sequential_R2"]) < 0.03, res


def test_team_beyond_coach_is_zero_when_team_is_coach_relabeled():
    """The CFB finding this decomposition exists to test for CBB: when a
    grouping adds NO information beyond an earlier one (here: team_id ==
    coach_id, i.e. one team per coach, the degenerate 'team is fully
    determined by coach' case), that grouping's SEQUENTIAL R^2 must be ~0
    even though its ONE-WAY R^2 looks exactly like the earlier grouping's."""
    rng = np.random.default_rng(SEED + 1)
    n_coaches = 25
    games_per_coach = 400
    sigma_coach, sigma_noise = 4.0, 7.0
    coach_effects = {c: rng.normal(0.0, sigma_coach) for c in range(n_coaches)}

    rows = []
    for c in range(n_coaches):
        for _ in range(games_per_coach):
            y = coach_effects[c] + rng.normal(0.0, sigma_noise)
            rows.append({"y": y, "coach": c, "team": c, "w": 1.0})
    panel = pd.DataFrame(rows)

    res = v.decompose(panel, "y", ["coach", "team"], weight_col="w")

    assert res["team_sequential_R2"] < 1e-6, res
    # one-way R^2 for team (a perfect relabeling of coach) equals coach's
    assert abs(res["team_R2"] - res["coach_R2"]) < 1e-9, res


def test_decompose_is_invariant_to_a_constant_weight_rescale():
    """Every R^2 is a ratio of weighted sums-of-squares, so multiplying every
    weight by the same positive constant must leave every reported number
    unchanged -- a basic sanity property any correct weighted implementation
    must have."""
    panel, _ = make_nested_panel(n_coaches=10, teams_per_coach=2, games_per_team=50)
    panel["w2"] = panel["w"] * 37.5

    res1 = v.decompose(panel, "y", ["coach", "team"], weight_col="w")
    res2 = v.decompose(panel, "y", ["coach", "team"], weight_col="w2")

    for key in res1:
        if isinstance(res1[key], float):
            assert abs(res1[key] - res2[key]) < 1e-9, (key, res1[key], res2[key])
        else:
            assert res1[key] == res2[key]


def test_missing_group_values_fill_to_unknown_without_crashing():
    """A row with a missing coach (e.g. a team-season not matched in
    `data/reference/coaches.parquet`) must become its own 'UNKNOWN' group,
    not silently drop out of the total-SS accounting or raise."""
    panel, _ = make_nested_panel(n_coaches=5, teams_per_coach=2, games_per_team=20)
    panel = panel.astype({"coach": "object"})
    panel.loc[panel.sample(frac=0.1, random_state=0).index, "coach"] = np.nan

    res = v.decompose(panel, "y", ["coach", "team"], weight_col="w")
    assert res["coach_n_groups"] == 6  # 5 real coaches + 'UNKNOWN'
    assert np.isfinite(res["residual_R2"])


def test_format_decomp_table_renders_expected_shape():
    panel, _ = make_nested_panel(n_coaches=5, teams_per_coach=2, games_per_team=20)
    res = v.decompose(panel, "y", ["coach", "team"], weight_col="w")
    table = v.format_decomp_table(res, ["coach", "team"], title="Synthetic check")

    assert "### Synthetic check" in table
    assert "| Grouping | # groups | One-way R^2 | Sequential R^2 |" in table
    assert "| coach | 5 |" in table
    assert "| team | 10 |" in table
    assert "| residual |" in table
