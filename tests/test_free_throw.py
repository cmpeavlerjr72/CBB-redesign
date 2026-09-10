"""free-throw tests: the FT-1 rule table, the era derivation, leak safety of
every as-of feature, the shrinkage algebra, determinism and the seal.

What is worth a dedicated test here, and why:

1. **The FT-1 rule table is complete and the foul class is derived from
   CONTEXT.** If the class were read off the attempt count, checking the count
   against the class would verify nothing. The tests below therefore build
   trips whose class is unambiguous from context and assert both the class and
   the rule's verdict.
2. **The bonus thresholds are DERIVED, not assumed.** The era question the
   pre-registration asks can only be answered by a detector that would find a
   moved threshold, so the detector is fed a synthetic season whose thresholds
   are deliberately NOT the NCAA defaults and is required to find them there.
3. **A shooter's own game cannot enter their own as-of rate.** L15 makes the
   shooter feature the whole model; an off-by-one in the expanding mean would
   make it a near-perfect predictor of its own target and the bake-off would
   pick it for the wrong reason.
4. **The shrinkage algebra is what the doc says it is**, including the two ends
   of the grid, because the reported "how many attempts before a shooter's own
   rate dominates" number is read straight off it.
5. **Determinism and the seal.**
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402

SEASON = 2025
HOME, AWAY = 100, 200


# ===========================================================================
# 1. FT-1: the rule table
# ===========================================================================
def test_every_foul_class_has_a_rule():
    assert set(FT.TRIP_RULES) == set(FT.FOUL_CLASSES)
    for cls, rule in FT.TRIP_RULES.items():
        assert "expected" in rule and "identified" in rule and rule["why"]


def test_foul_class_comes_from_context_not_from_the_attempt_count():
    cause = np.array(["technical", "foul", "foul", "foul", "foul", "none"], dtype=object)
    andone = np.array([False, True, False, False, False, False])
    prior = np.array([-1, 3, 3, 7, 11, -1])
    got = FT.classify_foul(cause, andone, prior)
    assert list(got) == ["technical", "and_one", "shooting", "bonus_one_and_one",
                         "double_bonus", "unknown"]


def test_the_thresholds_are_arguments_so_an_era_change_is_representable():
    cause = np.array(["foul", "foul"], dtype=object)
    andone = np.array([False, False])
    prior = np.array([4, 6])
    default = FT.classify_foul(cause, andone, prior)
    moved = FT.classify_foul(cause, andone, prior, bonus_prior=4, double_prior=6)
    assert list(default) == ["shooting", "bonus_one_and_one"]
    assert list(moved) == ["bonus_one_and_one", "double_bonus"]


def _synthetic_trips(bonus_at: int, double_at: int, n: int = 400) -> pd.DataFrame:
    """A season whose one-and-one window is deliberately somewhere other than
    the NCAA default, built so a detector that ASSUMES 6/9 cannot pass."""
    rows = []
    for p in range(0, 14):
        in_window = bonus_at <= p < double_at
        # one-attempt trips: inside the window they are missed front ends,
        # outside they are and-ones (made about two thirds of the time)
        for i in range(n):
            made = (i % 3 != 0) if not in_window else (i % 20 == 0)
            rows.append({"trip_cause": "foul", "trip_len": 1, "trip_prior_fouls": p,
                         "trip_first_made": made, "trip_andone_ctx": not in_window})
    return pd.DataFrame(rows)


def test_bonus_thresholds_are_derived_from_the_data_not_assumed():
    trips = _synthetic_trips(bonus_at=4, double_at=8)
    got = FT.derive_bonus_thresholds(trips)
    assert got["bonus_prior_fouls"] == 4
    assert got["double_bonus_prior_fouls"] == 8
    assert FT.BONUS_THRESHOLDS_DEFAULT == (6, 9)      # and the default was NOT used


def test_rule_verification_flags_violations_and_not_ambiguity():
    trips = pd.DataFrame([
        # a legal two-shot shooting foul below the bonus
        {"foul_class": "shooting", "trip_len": 2, "trip_first_made": True},
        # ILLEGAL: a foul below the bonus that produced a single attempt
        {"foul_class": "shooting", "trip_len": 1, "trip_first_made": True},
        # a legal and-one
        {"foul_class": "and_one", "trip_len": 1, "trip_first_made": True},
        # ILLEGAL: an and-one with two attempts
        {"foul_class": "and_one", "trip_len": 2, "trip_first_made": True},
        # legal one-and-one: single attempt, MISSED
        {"foul_class": "bonus_one_and_one", "trip_len": 1, "trip_first_made": False},
        # ILLEGAL: single attempt in the one-and-one that was MADE
        {"foul_class": "bonus_one_and_one", "trip_len": 1, "trip_first_made": True},
        # AMBIGUOUS, not a violation: two attempts with the bonus in force
        {"foul_class": "bonus_one_and_one", "trip_len": 2, "trip_first_made": True},
        {"foul_class": "double_bonus", "trip_len": 2, "trip_first_made": False},
        # ILLEGAL: a single attempt in the double bonus
        {"foul_class": "double_bonus", "trip_len": 1, "trip_first_made": False},
    ])
    out = FT.verify_trip_rules(trips)
    v = out["violations"]
    assert v["shooting_with_one_attempt"] == 1
    assert v["and_one_with_more_than_one_attempt"] == 1
    assert v["one_and_one_single_attempt_that_was_MADE"] == 1
    assert v["double_bonus_with_one_attempt"] == 1
    assert v["any_class_with_four_or_more_attempts"] == 0
    assert out["ambiguous"]["one_and_one_two_attempts"] == 1
    assert out["ambiguous"]["double_bonus_two_attempts"] == 1


# ===========================================================================
# 2. FT-1 end to end, on a synthetic event stream
# ===========================================================================
def _play(idx: int, play_type: str, is_home: bool, sec: int, made: bool | None = None,
          shot_range=None, player_id: float = 1.0, game: int = 1, period: int = 1) -> dict:
    shooting = play_type in ("JumpShot", "LayUpShot", "DunkShot", "TipShot", "MadeFreeThrow")
    return {
        "gameId": game, "season": SEASON, "id": idx, "playType": play_type,
        "isHomeTeam": is_home, "teamId": 1 if is_home else 2,
        "opponentId": 2 if is_home else 1, "homeScore": 0, "awayScore": 0,
        "period": period, "secondsRemaining": sec,
        "scoringPlay": bool(made) if made is not None else False,
        "shootingPlay": shooting, "scoreValue": 0, "shot_made": made,
        "shot_range": shot_range, "playText": "", "shot_shooter_id": player_id,
        "shot_location_x": np.nan, "shot_location_y": np.nan,
        "participant_1_id": player_id,
        **{f"home_on_{k}": float(k) for k in range(1, 6)},
        **{f"away_on_{k}": float(10 + k) for k in range(1, 6)},
    }


def _trips(tmp_path: Path, rows: list[dict]):
    df = pd.DataFrame(rows)[list(ES.STREAM_COLUMNS)]
    d = tmp_path / "pbp"
    d.mkdir(exist_ok=True)
    df.to_parquet(d / f"plays_{SEASON}.parquet", index=False)
    universe = pd.DataFrame({
        "game_id": [1000 + g for g in sorted(df["gameId"].unique())],
        "cbbd_game_id": sorted(df["gameId"].unique()), "season": SEASON,
        "game_date": pd.to_datetime("2025-01-01"), "neutral_site": False,
        "home_team_id": HOME, "away_team_id": AWAY,
        "is_d1_game": True, "pbp_truncated": False})
    return FT._season_trips(SEASON, universe, 0.0, d)


def test_a_two_shot_trip_is_one_trip_with_two_attempts(tmp_path):
    trips, att = _trips(tmp_path, [
        _play(1, "PersonalFoul", False, 1190),
        _play(2, "MadeFreeThrow", True, 1190, made=False, shot_range="free_throw"),
        _play(3, "Offensive Rebound", True, 1190),      # administrative reset
        _play(4, "MadeFreeThrow", True, 1190, made=True, shot_range="free_throw"),
        _play(5, "End Period", True, 0),
    ])
    assert len(trips) == 1
    assert len(att) == 2
    assert list(att["trip_pos"]) == [1, 2]
    assert (att["trip_len"] == 2).all()
    assert trips.iloc[0]["foul_class"] == "shooting"     # no bonus in force
    assert bool(trips.iloc[0]["trip_first_made"]) is False


def test_the_and_one_signature_is_recognised(tmp_path):
    trips, att = _trips(tmp_path, [
        _play(1, "LayUpShot", True, 1100, made=True, shot_range="rim"),
        _play(2, "PersonalFoul", False, 1100),
        _play(3, "MadeFreeThrow", True, 1100, made=True, shot_range="free_throw"),
        _play(4, "End Period", True, 0),
    ])
    assert trips.iloc[0]["foul_class"] == "and_one"
    assert int(trips.iloc[0]["trip_len"]) == 1


def test_the_bonus_class_follows_the_running_team_foul_count(tmp_path):
    rows = [_play(i + 1, "PersonalFoul", False, 1190 - i) for i in range(6)]
    rows += [
        _play(7, "PersonalFoul", False, 1180),          # the 7th foul: one-and-one
        _play(8, "MadeFreeThrow", True, 1180, made=False, shot_range="free_throw"),
        _play(9, "Defensive Rebound", False, 1178),
        _play(10, "End Period", True, 0),
    ]
    trips, _ = _trips(tmp_path, rows)
    assert trips.iloc[0]["foul_class"] == "bonus_one_and_one"
    assert int(trips.iloc[0]["trip_prior_fouls"]) == 6


# ===========================================================================
# 3. FT-2 leak safety
# ===========================================================================
def _toy_attempts() -> pd.DataFrame:
    """One shooter (7), three games on three dates, known make counts."""
    rows = []
    plan = {1: (10, 5), 2: (10, 9), 3: (10, 1)}      # game -> (attempts, makes)
    dates = {1: "2024-11-01", 2: "2024-11-08", 3: "2024-11-15"}
    for g, (n, k) in plan.items():
        for i in range(n):
            rows.append({
                "season": 2024, "game_id": g, "cbbd_game_id": g,
                "game_date": dates[g], "neutral_site": False, "shooter_is_home": True,
                "team_id": 10, "opp_id": 20, "shooter_id": 7, "period": 1,
                "seconds_remaining": 600, "score_diff": 0, "made": i < k,
                "trip_id": g * 100 + i, "trip_pos": 1, "trip_len": 1,
                "trip_cause": "foul", "trip_prior_fouls": 2,
                "trip_andone_ctx": False, "trip_first_made": i < k,
                "foul_class": "shooting",
            })
    # a second shooter so the league rate is not the shooter's own rate
    for g, dt in dates.items():
        for i in range(20):
            rows.append({
                "season": 2024, "game_id": g, "cbbd_game_id": g, "game_date": dt,
                "neutral_site": False, "shooter_is_home": False, "team_id": 20,
                "opp_id": 10, "shooter_id": 8, "period": 1, "seconds_remaining": 600,
                "score_diff": 0, "made": i < 14, "trip_id": g * 1000 + i,
                "trip_pos": 1, "trip_len": 1, "trip_cause": "foul",
                "trip_prior_fouls": 2, "trip_andone_ctx": False,
                "trip_first_made": i < 14, "foul_class": "shooting",
            })
    return pd.DataFrame(rows)


def _design(att: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    empty = tmp_path / "rosters"
    empty.mkdir(exist_ok=True)
    return FT.build_ft_design(att, roster_dir=empty, crosswalk_path=tmp_path / "nope.parquet")


def test_shooter_as_of_rate_is_the_strictly_earlier_average(tmp_path):
    d = _design(_toy_attempts(), tmp_path)
    s = d[d["shooter_id"] == 7]

    g1 = s[s["game_id"] == 1].iloc[0]
    assert g1["shooter_fta_asof"] == 0            # first game: nothing prior
    assert g1["shooter_ft_asof"] == pytest.approx(0.0, abs=1e-6)   # exactly the league mean

    g3 = s[s["game_id"] == 3].iloc[0]
    assert g3["shooter_fta_asof"] == 20
    own = (5 + 9) / 20
    league = (5 + 9 + 14 + 14) / (10 + 10 + 20 + 20)
    assert g3["shooter_ft_raw"] == pytest.approx(own, abs=1e-6)
    assert g3["shooter_ft_asof"] == pytest.approx(own - league, abs=1e-6)


def test_a_shooters_own_game_cannot_change_their_own_features(tmp_path):
    att = _toy_attempts()
    before = _design(att, tmp_path)
    corrupted = att.copy()
    mask = (corrupted["game_id"] == 3) & (corrupted["shooter_id"] == 7)
    corrupted.loc[mask, "made"] = True             # every attempt in game 3 now made
    after = _design(corrupted, tmp_path)

    cols = ["shooter_ft_asof", "shooter_fta_asof", "team_ft_asof"]
    b = before[(before["game_id"] == 3) & (before["shooter_id"] == 7)][cols].to_numpy()
    a = after[(after["game_id"] == 3) & (after["shooter_id"] == 7)][cols].to_numpy()
    np.testing.assert_array_equal(a, b)

    # ... and a LATER game must move, or the check above would pass for a
    # builder that ignores the data entirely.
    extra = att[att["game_id"] == 3].assign(game_id=4, game_date="2024-11-22",
                                            trip_id=lambda x: x["trip_id"] + 50000)
    extra_c = corrupted[corrupted["game_id"] == 3].assign(
        game_id=4, game_date="2024-11-22", trip_id=lambda x: x["trip_id"] + 50000)
    b4 = _design(pd.concat([att, extra], ignore_index=True), tmp_path)
    a4 = _design(pd.concat([corrupted, extra_c], ignore_index=True), tmp_path)
    b4v = b4[(b4["game_id"] == 4) & (b4["shooter_id"] == 7)]["shooter_ft_asof"].to_numpy()
    a4v = a4[(a4["game_id"] == 4) & (a4["shooter_id"] == 7)]["shooter_ft_asof"].to_numpy()
    assert not np.allclose(a4v, b4v)


def test_prior_season_rate_is_the_completed_previous_season(tmp_path):
    att = _toy_attempts()
    nxt = att.assign(season=2025, game_id=att["game_id"] + 10,
                     game_date=att["game_date"].str.replace("2024", "2025", regex=False),
                     trip_id=att["trip_id"] + 900000)
    d = _design(pd.concat([att, nxt], ignore_index=True), tmp_path)
    s25 = d[(d["season"] == 2025) & (d["shooter_id"] == 7)].iloc[0]
    assert s25["has_prior_season"] == 1
    assert s25["prior_season_ft_raw"] == pytest.approx((5 + 9 + 1) / 30, abs=1e-6)
    s24 = d[(d["season"] == 2024) & (d["shooter_id"] == 7)].iloc[0]
    assert s24["has_prior_season"] == 0


def test_technical_attempts_are_excluded_from_the_modelled_universe(tmp_path):
    att = _toy_attempts()
    tech = att.head(5).assign(foul_class="technical", trip_id=lambda x: x["trip_id"] + 7777)
    d = _design(pd.concat([att, tech], ignore_index=True), tmp_path)
    assert len(d) == len(att)
    assert not d["is_technical"].any()


# ===========================================================================
# 4. The shrinkage algebra
# ===========================================================================
def test_eb_shrinkage_is_the_stated_formula():
    d = pd.DataFrame({
        "shooter_ftm_prior": [15.0, 0.0],
        "shooter_fta_prior": [20.0, 0.0],
        "lg_ft_asof": [0.70, 0.70],
        "position_ft_asof": [0.68, 0.68],
        "prior_season_ft_raw": [0.80, 0.60],
    })
    p = FT.eb_predict(d, "league", 30.0)[:, FT.CLASS_INDEX["MAKE"]]
    np.testing.assert_allclose(p, [(30 * 0.70 + 15) / 50, (30 * 0.70 + 0) / 30], atol=1e-9)

    # m = 0 is the raw rate; a huge m is the prior
    p0 = FT.eb_predict(d.iloc[[0]], "league", 1e-9)[:, 1]
    assert p0[0] == pytest.approx(0.75, abs=1e-6)
    pbig = FT.eb_predict(d.iloc[[0]], "prior_season", 1e9)[:, 1]
    assert pbig[0] == pytest.approx(0.80, abs=1e-6)


def test_the_shrinkage_grid_covers_all_three_pre_registered_priors():
    assert set(FT.PRIOR_KINDS) == {"league", "position", "prior_season"}
    d = pd.DataFrame({
        "shooter_ftm_prior": [15.0], "shooter_fta_prior": [20.0],
        "lg_ft_asof": [0.70], "position_ft_asof": [0.68],
        "prior_season_ft_raw": [0.80], "y": [1],
    })
    fit = FT.fit_eb(d, grid=(10.0, 20.0))
    assert len(fit["grid"]) == 3 * 2
    assert fit["best"]["prior"] in FT.PRIOR_KINDS


# ===========================================================================
# 5. Seal, determinism, the simplex
# ===========================================================================
def _fit_frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(size=n).astype("float32") for f in FT.FT_FEATURES})
    d["y"] = rng.integers(0, 2, n).astype("int8")
    d["game_id"] = rng.integers(0, 200, n)
    d["in_bonus"] = rng.integers(0, 2, n).astype("float32")
    d["shooter_ft_asof"] = rng.normal(size=n).astype("float32")
    return d


def test_fold_slices_refuse_the_sealed_season():
    from cbb_sim.data.seal import SealedSeasonError

    design = pd.DataFrame({"season": [2022, 2026], "y": [0, 1], "game_id": [1, 2]})
    FT.FOLDS["FSEAL"] = {"train": [2022], "test": [2026]}
    try:
        with pytest.raises(SealedSeasonError):
            FT.fold_slices(design, "FSEAL")
    finally:
        FT.FOLDS.pop("FSEAL")


@pytest.mark.parametrize("arm", ["ridge", "lgbm"])
def test_arm_fits_are_deterministic(arm):
    pytest.importorskip("sklearn")
    if arm == "lgbm":
        pytest.importorskip("lightgbm")
    tr, te = _fit_frame(), _fit_frame(n=500, seed=7)
    cls = FT.RidgeArm if arm == "ridge" else FT.LgbmArm
    p1 = cls(seed=0).fit(FT.design_matrix(tr), tr["y"].to_numpy()).predict_proba(FT.design_matrix(te))
    p2 = cls(seed=0).fit(FT.design_matrix(tr), tr["y"].to_numpy()).predict_proba(FT.design_matrix(te))
    np.testing.assert_array_equal(p1, p2)


def test_probabilities_are_a_valid_simplex_in_the_declared_class_order():
    pytest.importorskip("sklearn")
    tr, te = _fit_frame(), _fit_frame(n=300, seed=3)
    m = FT.RidgeArm().fit(FT.design_matrix(tr), tr["y"].to_numpy())
    p = m.predict_proba(FT.design_matrix(te))
    assert p.shape == (len(te), 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert FT.CLASSES == ("MISS", "MAKE")
    assert FT.CLASS_INDEX["MAKE"] == 1


def test_bonus_era_table_round_trips(tmp_path):
    import json

    p = tmp_path / "bonus_era.json"
    p.write_text(json.dumps({"by_season": {
        "2024": {"bonus_prior_fouls": 6, "double_bonus_prior_fouls": 9},
        "2025": {"bonus_prior_fouls": 6, "double_bonus_prior_fouls": 9}}}))
    era = FT.load_bonus_era(p)
    assert era == {2024: (6, 9), 2025: (6, 9)}
    with pytest.raises(FileNotFoundError):
        FT.load_bonus_era(tmp_path / "missing.json")
