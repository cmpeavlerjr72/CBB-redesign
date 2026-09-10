"""Quick data-sanity assertions for the L0 game universe and team crosswalk.

These are cheap smoke checks against real seasons 2022-2026, not full
correctness proofs -- see docs/tests/data_audit_hoopr_2026-09-10.md and
docs/tests/data_audit_cbbd_2026-09-10.md for the underlying audit evidence
these ranges are drawn from.
"""

from __future__ import annotations

import pytest

from cbb_sim.data.ids import build_team_crosswalk
from cbb_sim.data.universe import build_universe, compute_d1_team_seasons, load_schedules

SEASONS = [2022, 2023, 2024, 2025, 2026]

# per-season hoopR schedule game counts are known from the audit (5976-6318);
# give a little headroom in case the raw pull is refreshed later.
EXPECTED_GAME_RANGE = (5800, 6500)


@pytest.fixture(scope="module")
def universe():
    out, report = build_universe(seasons=SEASONS)
    return out, report


def test_row_counts_per_season_in_expected_range(universe):
    out, _ = universe
    counts = out.groupby("season").size()
    assert set(counts.index) == set(SEASONS)
    for season, n in counts.items():
        lo, hi = EXPECTED_GAME_RANGE
        assert lo <= n <= hi, f"season {season} has {n} games, expected {lo}-{hi}"


def test_no_duplicate_game_id(universe):
    out, _ = universe
    assert out["game_id"].duplicated().sum() == 0


def test_d1_flag_share_is_reasonable(universe):
    out, _ = universe
    # hoopR audit: both-teams-D-I games run ~91-92% of the schedule every
    # season (e.g. 2025: 5769/6299). Leave generous headroom either side.
    share = out["is_d1_game"].mean()
    assert 0.85 <= share <= 0.97


def test_sealed_flag_matches_season_2026_only(universe):
    out, _ = universe
    assert set(out.loc[out["sealed"], "season"].unique()) == {2026}
    assert not out.loc[~out["sealed"], "season"].eq(2026).any()


def test_pbp_truncated_rate_is_low(universe):
    out, _ = universe
    # audit found truncated-pbp rates of ~0.1%-1.6% per season among games
    # that have pbp at all.
    with_pbp = out[out["has_pbp"]]
    rate = with_pbp["pbp_truncated"].mean()
    assert rate < 0.03


def test_crosswalk_espn_id_is_unique_and_well_matched():
    sch = load_schedules("data/raw/hoopr", SEASONS)
    d1_team_ids = {tid for (_, tid) in compute_d1_team_seasons(sch)}
    crosswalk, _unmatched, report = build_team_crosswalk(sch, d1_team_ids, "data/raw/cbbd")

    assert crosswalk["espn_team_id"].duplicated().sum() == 0
    assert report["n_cbbd_matched"] / report["n_teams"] >= 0.95
    assert report["n_kenpom_matched"] / report["n_teams"] >= 0.90
