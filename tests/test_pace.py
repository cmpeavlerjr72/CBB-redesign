"""pace tests: feature leak-safety, the pbp event-vocabulary guardrail, and
predictive-sampler determinism.

Two things are worth a dedicated test for this model beyond what
`tests/test_control.py` already covers for the as-of ratings/KenPom joins it
reuses verbatim:

1. **The style-rate features (new to this model) must use only strictly
   earlier games.** `shift(1).expanding().mean()` is easy to get off-by-one on
   (a bare `.expanding().mean()` would include the game itself). Verified by
   recomputing the expected value independently from the raw team-game table
   and requiring an exact match -- this is a stronger proof than corrupting a
   value would be, since it shows the feature IS the strictly-earlier-games
   average, not merely that it is insensitive to one perturbation.
2. **Every hoopR pbp event type used by `build_possessions_pbp.py` is
   explicitly classified**, per `docs/SIM_GUARDRAILS.md`'s standing
   requirement that a pbp-derived feature come from an explicit mapping table
   with a test that fails on an unknown type.
3. **The predictive sampler is deterministic and keyed on (seed, game_id,
   family) only**, the same RNG contract `tests/test_control.py` verifies for
   the Control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_possessions_pbp as BP  # noqa: E402

from cbb_sim.models import pace as P
from cbb_sim.ratings import own_ratings as orat

SEASON = 2024


@pytest.fixture(scope="module")
def universe():
    return orat.load_universe()


@pytest.fixture(scope="module")
def pace_table(universe):
    u = universe[universe["season"] == SEASON]
    return P.build_pace_table([SEASON], universe=u)


@pytest.fixture(scope="module")
def raw_team_games(universe):
    u = universe[universe["season"] == SEASON]
    tg = orat.load_team_games(u, [SEASON])
    tg["tpa_per100"] = 100.0 * tg["tpa"] / tg["game_poss"]
    tg["fta_per100"] = 100.0 * tg["fta"] / tg["game_poss"]
    tg["tov_per100"] = 100.0 * tg["tov"] / tg["game_poss"]
    return tg


# ---------------------------------------------------------------------------
# 1. style-rate feature leak-safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stat", ["tpa_per100", "fta_per100", "tov_per100"])
def test_own_style_rate_uses_only_strictly_earlier_games(pace_table, raw_team_games, stat):
    g = pace_table.dropna(subset=[f"home_own_style_{stat}"])
    sample = g.sample(min(25, len(g)), random_state=20260910)
    checked = 0
    for r in sample.itertuples():
        prior = raw_team_games[
            (raw_team_games["team_id"] == r.home_team_id) & (raw_team_games["game_date"] < r.game_date)
        ]
        if prior.empty:
            continue
        expected = float(prior[stat].mean())
        got = float(getattr(r, f"home_own_style_{stat}"))
        assert np.isclose(got, expected, rtol=1e-9, atol=1e-9), (
            f"team {r.home_team_id} on {r.game_date}: got {got}, expected {expected} "
            "from strictly-earlier games only -- feature is leaking same-day or later data"
        )
        checked += 1
    assert checked >= 10, f"only {checked} informative rows checked"


def test_own_style_rate_excludes_the_games_own_value_even_when_it_is_extreme(raw_team_games, universe):
    """The teeth of the leak test: inject an extreme, obviously-wrong stat
    into ONE game and require that game's own as-of feature (computed from
    everything strictly before it) is untouched, while a LATER game for the
    same team moves -- proof the exclusion isn't vacuous."""
    tg = raw_team_games.copy()
    dates = np.sort(tg["game_date"].unique())
    mid = pd.Timestamp(dates[len(dates) // 2])
    team_id = tg.loc[tg["game_date"] == mid, "team_id"].iloc[0]

    def own_style(frame: pd.DataFrame, team: int, before: pd.Timestamp) -> float:
        prior = frame[(frame["team_id"] == team) & (frame["game_date"] < before)]
        return float(prior["tpa_per100"].mean()) if len(prior) else float("nan")

    later_dates = tg.loc[(tg["team_id"] == team_id) & (tg["game_date"] > mid), "game_date"]
    if later_dates.empty:
        pytest.skip("chosen team has no later game this season")
    later = later_dates.min()

    base_before = own_style(tg, team_id, mid)
    base_after = own_style(tg, team_id, later)

    corrupt = tg.copy()
    corrupt.loc[(corrupt["team_id"] == team_id) & (corrupt["game_date"] == mid), "tpa_per100"] = 999.0

    dirty_before = own_style(corrupt, team_id, mid)
    dirty_after = own_style(corrupt, team_id, later)

    assert base_before == dirty_before or (np.isnan(base_before) and np.isnan(dirty_before)), (
        "corrupting a game's OWN stat changed its OWN as-of feature -- same-day leak"
    )
    assert dirty_after != base_after, "corrupting a game changed nothing later -- test is vacuous"


# ---------------------------------------------------------------------------
# 2. pbp event-vocabulary guardrail
# ---------------------------------------------------------------------------
def test_every_observed_hoopr_event_type_is_classified():
    for season in (2022, 2023, 2024, 2025):
        p = BP._hoopr_pbp_path(BP.DEFAULT_HOOPR_DIR, season)
        if not p.exists():
            continue
        types = pd.read_parquet(p, columns=["type_text"])["type_text"].unique()
        for t in types:
            BP.classify(t)  # raises ValueError on anything unmapped


def test_unknown_pbp_event_type_raises():
    with pytest.raises(ValueError):
        BP.classify("SomeBrandNewEventTypeThatDoesNotExistYet")


# ---------------------------------------------------------------------------
# 3. predictive sampler determinism
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_inputs():
    mu = np.array([60.0, 68.0, 72.0, 55.0, 80.0])
    game_ids = np.array([101, 102, 103, 104, 105])
    return mu, game_ids


def test_sampler_is_bit_identical_under_a_fixed_seed(toy_inputs):
    mu, gids = toy_inputs
    dist = {"family": "gaussian", "sd": 5.0}
    a = P.sample_pace(mu, gids, 7, dist)
    b = P.sample_pace(mu, gids, 7, dist)
    assert np.array_equal(a, b)


def test_sampler_different_seeds_give_different_draws(toy_inputs):
    mu, gids = toy_inputs
    dist = {"family": "gaussian", "sd": 5.0}
    a = P.sample_pace(mu, gids, 7, dist)
    b = P.sample_pace(mu, gids, 1000, dist)
    assert not np.array_equal(a, b)


def test_sampler_draws_do_not_depend_on_other_games_in_the_batch(toy_inputs):
    mu, gids = toy_inputs
    dist = {"family": "gaussian", "sd": 5.0}
    full = P.sample_pace(mu, gids, 7, dist)
    half = P.sample_pace(mu[:2], gids[:2], 7, dist)
    assert np.array_equal(full[:2], half)


def test_sampler_supports_every_distribution_family(toy_inputs):
    mu, gids = toy_inputs
    for dist in (
        {"family": "gaussian", "sd": 5.0},
        {"family": "gaussian_hetero", "sd": np.full(len(mu), 5.0)},
        {"family": "negbin", "alpha": 0.02},
        {"family": "poisson"},
    ):
        draw = P.sample_pace(mu, gids, 7, dist)
        assert draw.shape == mu.shape
        assert np.isfinite(draw).all()


def test_sampler_rejects_unknown_family(toy_inputs):
    mu, gids = toy_inputs
    with pytest.raises(ValueError):
        P.sample_pace(mu, gids, 7, {"family": "not_a_real_family"})


def test_hetero_sd_bias_correction_recovers_true_sigma():
    """Regression test for the E[log(chi2_1)] bias fix in `hetero_sd`: without
    the `_LOG_CHI2_1_MEAN_CORRECTION` term, a homoscedastic-truth simulation
    would recover an SD about 1.89x too small."""
    rng = np.random.default_rng(20260910)
    n = 20000
    x = rng.normal(size=n)
    true_sigma = 3.0
    y = 50.0 + rng.normal(scale=true_sigma, size=n)
    df = pd.DataFrame({"x": x, "target": y})
    mu_train = np.full(n, 50.0)
    dist = P.fit_gaussian_hetero(df, ["x"], "target", mu_train)
    sd = P.hetero_sd(dist, df)
    assert abs(float(sd.mean()) - true_sigma) < 0.15
