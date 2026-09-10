"""Data-sanity assertions for data/reference/coaches.parquet.

See docs/tests/coaches_table_2026-09-10.md for the source-compliance
verdict, build method, coverage report, and cross-validation result these
checks assume.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COACHES_PATH = REPO_ROOT / "data" / "reference" / "coaches.parquet"
CROSSWALK_PATH = REPO_ROOT / "data" / "reference" / "team_crosswalk.parquet"

SEASONS = [2022, 2023, 2024, 2025, 2026, 2027]

pytestmark = pytest.mark.skipif(not COACHES_PATH.exists(), reason="coaches.parquet not built yet")


@pytest.fixture(scope="module")
def coaches() -> pd.DataFrame:
    return pd.read_parquet(COACHES_PATH)


@pytest.fixture(scope="module")
def crosswalk() -> pd.DataFrame:
    return pd.read_parquet(CROSSWALK_PATH)


def test_expected_columns_present(coaches):
    expected = {
        "season",
        "espn_team_id",
        "team",
        "head_coach",
        "coach_id",
        "interim",
        "midseason_change",
        "notes",
        "source",
        "source_url",
        "fetched_at",
    }
    assert expected.issubset(set(coaches.columns))


def test_one_row_per_season_team(coaches):
    """No duplicate (season, espn_team_id) rows."""
    dupes = coaches.duplicated(subset=["season", "espn_team_id"]).sum()
    assert dupes == 0


def test_seasons_are_2022_to_2027(coaches):
    assert set(coaches["season"].unique()).issubset(set(SEASONS))
    # at least most of the run window is represented
    assert len(set(coaches["season"].unique())) >= 5


def test_espn_team_ids_are_from_the_367_team_crosswalk(coaches, crosswalk):
    valid_ids = set(crosswalk["espn_team_id"].astype(int))
    ids_in_table = set(coaches["espn_team_id"].astype(int))
    assert ids_in_table.issubset(valid_ids)


def test_no_null_head_coach(coaches):
    """Every row we wrote came from a resolved source page -- a team-season
    the source didn't have is simply absent, never a null placeholder."""
    assert coaches["head_coach"].notna().all()
    assert (coaches["head_coach"].str.strip() != "").all()


def test_no_null_coach_id(coaches):
    assert coaches["coach_id"].notna().all()
    assert (coaches["coach_id"].str.strip() != "").all()


def test_coverage_is_substantial(coaches, crosswalk):
    """Sanity floor, not a tight bound -- the real per-season counts are in
    the coverage report in docs/tests/coaches_table_2026-09-10.md."""
    n_teams = len(crosswalk)
    n_season_team_pairs = len(coaches)
    max_possible = n_teams * len(SEASONS)
    assert n_season_team_pairs > 0.7 * max_possible


def test_coach_id_is_stable_across_seasons_for_the_same_name(coaches):
    """A given head_coach display name should resolve to exactly one
    coach_id across every season it appears in (modulo the rare, expected
    case of two *different* people sharing a plain-text name with no
    Wikipedia article to disambiguate them -- flagged, not asserted away)."""
    by_name = coaches.groupby("head_coach")["coach_id"].nunique()
    offenders = by_name[by_name > 1]
    # allow a small residual for undisambiguated plain-text same-name cases
    assert len(offenders) <= max(3, int(0.01 * len(by_name))), (
        f"coach_id is not stable for these head_coach names: {offenders.index.tolist()}"
    )


def test_coach_id_is_a_slug(coaches):
    import re

    pattern = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
    bad = coaches.loc[~coaches["coach_id"].apply(lambda s: bool(pattern.match(str(s)))), "coach_id"]
    assert bad.empty, f"non-slug coach_id values: {bad.unique().tolist()[:10]}"


def test_interim_and_midseason_change_are_bool(coaches):
    assert coaches["interim"].dtype == bool
    assert coaches["midseason_change"].dtype == bool


def test_midseason_change_rows_have_notes(coaches):
    flagged = coaches[coaches["midseason_change"]]
    assert flagged["notes"].notna().all()


def test_source_and_source_url_present(coaches):
    assert coaches["source"].notna().all()
    assert coaches["source_url"].str.startswith("https://en.wikipedia.org/wiki/").all()


def test_fetched_at_is_parseable_timestamp(coaches):
    parsed = pd.to_datetime(coaches["fetched_at"], utc=True, errors="coerce")
    assert parsed.notna().all()
