"""field-goal-make tests: leak safety, class isolation, the chance-state
derivation, the EB algebra, determinism and the seal.

What is worth a dedicated test here, and why:

1. **No post-outcome field can reach a feature.** This model is the one place
   in the cascade where a perfect leak is one column away: `blocked` exists only
   because the shot missed, the and-one foul is logged only after it went in,
   and an assist in this feed is a property of a MADE basket. The tests below
   prove that none of the three is in any bundle, that `design_matrix` refuses
   them by name, and that the design frame does not even carry an assist
   column.
2. **An attempt cannot see its own game.** Every as-of rate is the whole model;
   an off-by-one in the expanding mean would make the shooter feature a
   near-perfect predictor of its own target and the bake-off would pick it for
   the wrong reason. Proved two ways, the way the rebound and free-throw suites
   prove it: an independent recomputation of the strictly-earlier average, and
   invariance of every feature to corrupting the attempt's own game while a
   LATER game is required to move.
3. **The chance-state block is at-release, not post-outcome.** The L5 ledger
   bans `is_transition` because it is a function of the chance's own duration,
   i.e. of when the chance ENDS. Here it is a function of when the chance
   STARTED plus the attempt's own clock. The test flips an attempt's outcome and
   requires every one of its own features -- chance number, elapsed, transition
   included -- to be bit-identical.
4. **The three classes never share a fit.** Corrupting one class's rows must
   leave another class's predictions bit-identical, which is a property of
   `fit_by_class` and not of the trainer's discipline.
5. **The EB arm is the stated formula**, including the property that makes it a
   matchup model: a league-average defence must contribute exactly nothing.
6. **Determinism and the seal.**
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402

SEASON = 2024
HOME, AWAY = 100, 200


# ===========================================================================
# Synthetic fixtures
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


def _events_from_plays(tmp_path: Path, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)[list(ES.STREAM_COLUMNS)]
    d = tmp_path / "pbp"
    d.mkdir(exist_ok=True)
    df.to_parquet(d / f"plays_{SEASON}.parquet", index=False)
    universe = pd.DataFrame({
        "game_id": [1000 + g for g in sorted(df["gameId"].unique())],
        "cbbd_game_id": sorted(df["gameId"].unique()), "season": SEASON,
        "game_date": pd.to_datetime("2024-01-01"), "neutral_site": False,
        "home_team_id": HOME, "away_team_id": AWAY,
        "is_d1_game": True, "pbp_truncated": False})
    return FG._season_events(SEASON, universe, 0.0, d)


ON_FLOOR_FILL = {**{f"home_on_{k}": float(k) for k in range(1, 6)},
                 **{f"away_on_{k}": float(10 + k) for k in range(1, 6)}}


def _toy_events(seasons: tuple[int, ...] = (2024,)) -> pd.DataFrame:
    """Two shooters, three games on three dates, known make counts per class.

    Shooter 7 takes 10 rim attempts a game, making 5 / 9 / 1; shooter 8 takes
    20 threes a game making 14 each time, so the league rate is never the same
    number as either shooter's own rate."""
    plan = {1: 5, 2: 9, 3: 1}
    dates = {1: "2024-11-01", 2: "2024-11-08", 3: "2024-11-15"}
    rows = []
    for season in seasons:
        for g, k in plan.items():
            gid = season * 100 + g
            for i in range(10):
                rows.append({"season": season, "game_id": gid, "cbbd_game_id": gid,
                             "game_date": dates[g].replace("2024", str(season)),
                             "off_team_id": 10, "def_team_id": 20, "offense_is_home": True,
                             "shooter_id": 7.0, "shot_class": "FGA_rim", "made": i < k})
            for i in range(20):
                rows.append({"season": season, "game_id": gid, "cbbd_game_id": gid,
                             "game_date": dates[g].replace("2024", str(season)),
                             "off_team_id": 20, "def_team_id": 10, "offense_is_home": False,
                             "shooter_id": 8.0, "shot_class": "FGA_3", "made": i < 14})
    d = pd.DataFrame(rows)
    d["neutral_site"] = False
    d["period"] = 1
    d["seconds_remaining"] = 600
    d["score_diff"] = 0
    d["blocked"] = False
    d["and_one"] = False
    d["off_in_bonus"] = False
    d["off_in_double_bonus"] = False
    d["chance_number"] = 1
    d["chance_start_reason"] = "DREB"
    d["chance_elapsed_s"] = 12.0
    d["chance_elapsed_clipped"] = False
    d["chance_side_ok"] = True
    d["chance_side_known"] = True
    d["is_transition"] = False
    d["class_key"] = d["shot_class"].map(FG.CLASS_KEY)
    for c, v in ON_FLOOR_FILL.items():
        d[c] = v
    return d


def _design(events: pd.DataFrame, tmp_path: Path, **kw) -> pd.DataFrame:
    rosters = tmp_path / "rosters"
    rosters.mkdir(exist_ok=True)
    seasons = sorted(events["season"].unique().tolist())
    universe = events[["game_id", "game_date"]].drop_duplicates().reset_index(drop=True)
    return FG.build_design(seasons, universe=universe, events=events,
                           roster_dir=rosters, crosswalk_path=tmp_path / "nope.parquet",
                           with_ratings=False, **kw)


# ===========================================================================
# 1. Banned, post-outcome fields
# ===========================================================================
def test_no_feature_bundle_contains_a_post_outcome_field():
    for name in FG.FEATURE_SETS:
        feats = FG.feature_set(name)
        assert not set(feats) & set(FG.BANNED_FEATURES), name
        # nor anything that merely looks like the outcome
        assert not [f for f in feats if f in ("made", "y", "blocked", "and_one", "assisted")]


def test_design_matrix_refuses_a_banned_column_by_name():
    d = pd.DataFrame({"off_make_c": [0.0], "blocked": [1.0]})
    with pytest.raises(ValueError, match="post-outcome"):
        FG.design_matrix(d, ["off_make_c", "blocked"])


def test_the_assist_is_not_built_at_all(tmp_path):
    d = _design(_toy_events(), tmp_path)
    assert not [c for c in d.columns if "assist" in c.lower()]


def test_blocked_attempts_are_misses_and_the_flag_is_not_a_feature(tmp_path):
    ev = _events_from_plays(tmp_path, [
        _play(1, "JumpShot", True, 1190, made=False, shot_range="jumper"),
        _play(2, "Block Shot", False, 1190),
        _play(3, "Defensive Rebound", False, 1188),
        _play(4, "End Period", True, 0),
    ])
    assert len(ev) == 1
    assert bool(ev.iloc[0]["blocked"]) is True
    assert bool(ev.iloc[0]["made"]) is False
    assert "blocked" not in FG.feature_set("D_plus_lineup")


def test_an_and_one_is_kept_as_a_make(tmp_path):
    ev = _events_from_plays(tmp_path, [
        _play(1, "LayUpShot", True, 1100, made=True, shot_range="rim"),
        _play(2, "PersonalFoul", False, 1100),
        _play(3, "MadeFreeThrow", True, 1100, made=True, shot_range="free_throw"),
        _play(4, "End Period", True, 0),
    ])
    assert len(ev) == 1
    assert bool(ev.iloc[0]["made"]) is True
    assert bool(ev.iloc[0]["and_one"]) is True
    assert "and_one" not in FG.feature_set("D_plus_lineup")


# ===========================================================================
# 2. As-of leak safety
# ===========================================================================
def test_shooter_as_of_rate_is_the_strictly_earlier_average(tmp_path):
    d = _design(_toy_events(), tmp_path)
    s = d[d["shooter_id"] == 7]

    g1 = s[s["game_id"] == 202401].iloc[0]
    assert g1["shooter_att_c"] == 0
    assert g1["shooter_make_c"] == pytest.approx(0.0, abs=1e-6)   # exactly the league mean

    g3 = s[s["game_id"] == 202403].iloc[0]
    assert g3["shooter_att_c"] == 20
    own = (5 + 9) / 20
    league_rim = (5 + 9) / 20           # league rim rate is shooter 7's own here
    assert g3["shooter_make_raw"] == pytest.approx(own, abs=1e-6)
    assert g3["shooter_make_c"] == pytest.approx(own - league_rim, abs=1e-6)

    t = d[(d["shooter_id"] == 8) & (d["game_id"] == 202403)].iloc[0]
    assert t["shooter_att_c"] == 40
    assert t["shooter_make_raw"] == pytest.approx(28 / 40, abs=1e-6)


def test_team_form_is_the_strictly_earlier_average_both_sides(tmp_path):
    d = _design(_toy_events(), tmp_path)
    # team 10 shoots rim; its as-of rim rate before game 3 is (5+9)/20
    r = d[(d["off_team_id"] == 10) & (d["game_id"] == 202403)].iloc[0]
    assert r["off_att_prior"] == 20
    assert r["off_make_raw"] == pytest.approx((5 + 9) / 20, abs=1e-6)
    # team 10 is the DEFENCE on the threes; its allowed three rate is (14+14)/40
    t = d[(d["def_team_id"] == 10) & (d["game_id"] == 202403)].iloc[0]
    assert t["def_att_prior"] == 40
    assert t["def_allow_raw"] == pytest.approx(28 / 40, abs=1e-6)


def test_an_attempts_own_game_cannot_change_its_own_features(tmp_path):
    ev = _toy_events()
    before = _design(ev, tmp_path)
    corrupted = ev.copy()
    mask = (corrupted["game_id"] == 202403) & (corrupted["shooter_id"] == 7)
    corrupted.loc[mask, "made"] = True
    after = _design(corrupted, tmp_path)

    cols = ["shooter_make_c", "shooter_att_c", "off_make_c", "def_allow_c",
            "shooter_games_asof", "shooter_fga_asof", "chance_number",
            "chance_elapsed_s", "is_transition_f"]
    sel_b = (before["game_id"] == 202403) & (before["shooter_id"] == 7)
    sel_a = (after["game_id"] == 202403) & (after["shooter_id"] == 7)
    np.testing.assert_array_equal(after.loc[sel_a, cols].to_numpy(),
                                  before.loc[sel_b, cols].to_numpy())

    # ... and a LATER game must move, or the check above would also pass for a
    # builder that ignores the data entirely.
    extra = ev[ev["game_id"] == 202403].assign(game_id=202404, cbbd_game_id=202404,
                                               game_date="2024-11-22")
    extra_c = corrupted[corrupted["game_id"] == 202403].assign(
        game_id=202404, cbbd_game_id=202404, game_date="2024-11-22")
    b4 = _design(pd.concat([ev, extra], ignore_index=True), tmp_path)
    a4 = _design(pd.concat([corrupted, extra_c], ignore_index=True), tmp_path)
    # `shooter_make_raw` is the probe rather than the centred column: shooter 7
    # is the only rim shooter in this fixture, so the league rim rate IS his own
    # rate and the centred feature is exactly 0.0 either way -- which is the
    # convention, not a bug.
    bv = b4.loc[(b4["game_id"] == 202404) & (b4["shooter_id"] == 7), "shooter_make_raw"].to_numpy()
    av = a4.loc[(a4["game_id"] == 202404) & (a4["shooter_id"] == 7), "shooter_make_raw"].to_numpy()
    assert not np.allclose(av, bv)
    assert bv[0] == pytest.approx((5 + 9 + 1) / 30, abs=1e-6)
    assert av[0] == pytest.approx((5 + 9 + 10) / 30, abs=1e-6)


def test_prior_season_rate_is_the_completed_previous_season(tmp_path):
    d = _design(_toy_events(seasons=(2023, 2024)), tmp_path)
    s24 = d[(d["season"] == 2024) & (d["shooter_id"] == 7)].iloc[0]
    assert s24["has_prior_season"] == 1
    assert s24["prior_season_make_raw"] == pytest.approx((5 + 9 + 1) / 30, abs=1e-6)
    s23 = d[(d["season"] == 2023) & (d["shooter_id"] == 7)].iloc[0]
    assert s23["has_prior_season"] == 0


# ===========================================================================
# 3. The chance-state block
# ===========================================================================
def test_an_offensive_rebound_chain_increments_the_chance_number(tmp_path):
    ev = _events_from_plays(tmp_path, [
        _play(1, "JumpShot", True, 1190, made=False, shot_range="jumper"),
        _play(2, "Offensive Rebound", True, 1188),
        _play(3, "LayUpShot", True, 1186, made=False, shot_range="rim"),
        _play(4, "Offensive Rebound", True, 1185),
        _play(5, "JumpShot", True, 1180, made=True, shot_range="three_pointer"),
        _play(6, "Defensive Rebound", False, 1178),
        _play(7, "LayUpShot", False, 1174, made=False, shot_range="rim"),
        _play(8, "End Period", True, 0),
    ])
    assert list(ev["chance_number"]) == [1, 2, 3, 1]
    assert list(ev["chance_start_reason"]) == ["period_start", "OREB", "OREB", "DREB"]
    # elapsed is measured from the chance's own start, never from the period's
    assert list(ev["chance_elapsed_s"]) == [10.0, 2.0, 5.0, 4.0]
    assert ev["chance_side_ok"].all()


def test_a_transition_chance_is_a_fast_one_off_a_live_turnover(tmp_path):
    ev = _events_from_plays(tmp_path, [
        _play(1, "Lost Ball Turnover", True, 1190),
        _play(2, "LayUpShot", False, 1186, made=True, shot_range="rim"),
        _play(3, "JumpShot", True, 1150, made=False, shot_range="jumper"),
        _play(4, "End Period", True, 0),
    ])
    assert list(ev["chance_start_reason"]) == ["TOV", "FGA_rim"]
    assert list(ev["is_transition"]) == [True, False]
    # the second shot starts on a made basket and took 36 s: not transition
    assert ev.iloc[1]["chance_elapsed_s"] == pytest.approx(36.0)


def test_flipping_an_attempts_outcome_cannot_move_its_own_chance_state(tmp_path):
    rows = [
        _play(1, "JumpShot", True, 1190, made=False, shot_range="jumper"),
        _play(2, "Offensive Rebound", True, 1188),
        _play(3, "LayUpShot", True, 1186, made=False, shot_range="rim"),
        _play(4, "Defensive Rebound", False, 1184),
        _play(5, "End Period", True, 0),
    ]
    a = _events_from_plays(tmp_path, rows)
    flipped = [dict(r) for r in rows]
    flipped[2]["shot_made"] = True
    flipped[2]["scoringPlay"] = True
    b = _events_from_plays(tmp_path, flipped)
    cols = ["chance_number", "chance_elapsed_s", "chance_start_reason", "is_transition"]
    # plays index 2 is the SECOND field-goal attempt, i.e. row 1 of the events
    np.testing.assert_array_equal(a.iloc[1][cols].to_numpy(), b.iloc[1][cols].to_numpy())
    assert bool(a.iloc[1]["made"]) is not bool(b.iloc[1]["made"])


def test_period_start_is_the_period_length_not_the_previous_periods_clock(tmp_path):
    ev = _events_from_plays(tmp_path, [
        _play(1, "JumpShot", True, 5, made=False, shot_range="jumper", period=1),
        _play(2, "Defensive Rebound", False, 3, period=1),
        _play(3, "End Period", True, 0, period=1),
        _play(4, "LayUpShot", True, 1195, made=False, shot_range="rim", period=2),
        _play(5, "End Period", True, 0, period=2),
    ])
    second = ev[ev["period"] == 2].iloc[0]
    assert second["chance_start_reason"] == "period_start"
    assert second["chance_elapsed_s"] == pytest.approx(5.0)
    assert int(second["chance_number"]) == 1


# ===========================================================================
# 4. The three classes never share a fit
# ===========================================================================
def _fit_frame(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feats = FG.feature_set("C_plus_state")
    d = pd.DataFrame({f: rng.normal(size=n).astype("float32") for f in feats})
    d["shot_class"] = rng.choice(list(FG.SHOT_CLASSES), size=n)
    d["class_key"] = pd.Series(d["shot_class"]).map(FG.CLASS_KEY).to_numpy()
    d["y"] = rng.integers(0, 2, n).astype("int8")
    d["game_id"] = rng.integers(0, 300, n)
    d["season"] = 2024
    d["shooter_att_c"] = rng.integers(0, 50, n).astype("float32")
    d["def_att_prior"] = rng.integers(1, 900, n).astype("float32")
    return d


def test_class_slices_are_disjoint_and_exhaustive():
    d = _fit_frame()
    ns = [len(FG.class_slice(d, c)) for c in FG.SHOT_CLASSES]
    assert sum(ns) == len(d)
    idx = pd.concat([FG.class_slice(d, c) for c in FG.SHOT_CLASSES]).index
    assert idx.is_unique


@pytest.mark.parametrize("arm", ["ridge", "lgbm"])
def test_corrupting_one_class_cannot_move_another_classes_predictions(arm):
    pytest.importorskip("sklearn")
    if arm == "lgbm":
        pytest.importorskip("lightgbm")
    tr, te = _fit_frame(), _fit_frame(n=900, seed=7)
    models = FG.fit_by_class(arm, tr)
    target = FG.SHOT_CLASSES[0]
    p1 = FG.predict_arm(arm, models[target], FG.class_slice(te, target))

    bad = tr.copy()
    other = bad["shot_class"] != target
    bad.loc[other, "y"] = 1 - bad.loc[other, "y"].to_numpy()
    for f in FG.feature_set("C_plus_state"):
        bad.loc[other, f] = 99.0
    models2 = FG.fit_by_class(arm, bad)
    p2 = FG.predict_arm(arm, models2[target], FG.class_slice(te, target))
    np.testing.assert_array_equal(p1, p2)
    assert models[target] is not models2[target]


def test_fit_by_class_returns_one_estimator_per_class():
    pytest.importorskip("sklearn")
    models = FG.fit_by_class("ridge", _fit_frame())
    assert set(models) == set(FG.SHOT_CLASSES)
    assert len({id(m) for m in models.values()}) == len(FG.SHOT_CLASSES)


# ===========================================================================
# 5. The EB arm's algebra
# ===========================================================================
def _eb_frame(def_rate: float, lg: float = 0.50, att: float = 20.0, mk: float = 15.0,
              def_att: float = 1000.0) -> pd.DataFrame:
    return pd.DataFrame({
        "shooter_mk_c": [mk], "shooter_att_c": [att],
        "lg_make_asof": [lg], "position_make_raw": [0.48],
        "prior_season_make_raw": [0.62],
        "def_allow_raw": [def_rate], "def_att_prior": [def_att],
    })


def test_eb_is_the_stated_logit_combination():
    d = _eb_frame(def_rate=0.55)
    p = FG.eb_predict(d, "league", 30.0, 0.0)[0, FG.CLASS_INDEX["MAKE"]]
    p_sh = (30 * 0.50 + 15) / (30 + 20)
    z = np.log(p_sh / (1 - p_sh)) + np.log(0.55 / 0.45) - np.log(0.50 / 0.50)
    assert p == pytest.approx(1 / (1 + np.exp(-z)), abs=1e-9)


def test_a_league_average_defence_contributes_exactly_nothing():
    d = _eb_frame(def_rate=0.50)
    p = FG.eb_predict(d, "league", 30.0, 0.0)[0, FG.CLASS_INDEX["MAKE"]]
    p_sh = (30 * 0.50 + 15) / (30 + 20)
    assert p == pytest.approx(p_sh, abs=1e-9)


def test_the_two_ends_of_the_shrinkage_grid():
    d = _eb_frame(def_rate=0.50)
    raw = FG.eb_predict(d, "league", 1e-9, 0.0)[0, 1]
    assert raw == pytest.approx(0.75, abs=1e-6)
    prior = FG.eb_predict(d, "prior_season", 1e9, 0.0)[0, 1]
    assert prior == pytest.approx(0.62, abs=1e-5)


def test_the_defence_strength_shrinks_the_defence_toward_the_league():
    d = _eb_frame(def_rate=0.60, def_att=10.0)
    strong = FG.eb_predict(d, "league", 30.0, 0.0)[0, 1]
    shrunk = FG.eb_predict(d, "league", 30.0, 1e6)[0, 1]
    neutral = FG.eb_predict(_eb_frame(def_rate=0.50), "league", 30.0, 0.0)[0, 1]
    assert strong > shrunk > neutral - 1e-9
    assert shrunk == pytest.approx(neutral, abs=1e-4)


def test_the_shrinkage_grid_covers_all_three_pre_registered_priors():
    assert set(FG.PRIOR_KINDS) == {"league", "position", "prior_season"}
    d = _eb_frame(def_rate=0.52)
    d["y"] = [1]
    fit = FG.fit_eb(d, grid=(10.0, 20.0), def_grid=(0.0, 100.0))
    assert len(fit["grid"]) == 3 * 2 * 2
    assert fit["best"]["prior"] in FG.PRIOR_KINDS


# ===========================================================================
# 6. Metrics
# ===========================================================================
def test_a_tie_mass_in_the_driver_cannot_collapse_a_quintile():
    n = 5000
    rng = np.random.default_rng(0)
    te = pd.DataFrame({
        # half the rows sit at exactly the league mean -- the no-prior-attempt
        # tie mass that would give a repeated quantile edge on the raw driver
        "shooter_make_c": np.where(rng.random(n) < 0.5, 0.0, rng.normal(size=n)),
        "def_allow_c": rng.normal(size=n),
        "shooter_att_c": rng.integers(1, 40, n),
        "def_att_prior": rng.integers(1, 900, n),
    })
    te["y"] = (rng.random(n) < 0.4).astype("int8")
    p = np.column_stack([1 - np.full(n, 0.4), np.full(n, 0.4)])
    out = FG.responsiveness(te, p)
    for k, v in out.items():
        assert min(v["n"]) > 0, k
        assert len(v["n"]) == 5


def test_rows_with_no_as_of_rate_are_reported_separately_not_bucketed():
    n = 100
    te = pd.DataFrame({
        "shooter_make_c": np.zeros(n), "def_allow_c": np.linspace(-1, 1, n),
        "shooter_att_c": np.concatenate([np.zeros(40), np.ones(60)]),
        "def_att_prior": np.ones(n),
        "y": np.zeros(n, dtype="int8"),
    })
    p = np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])
    out = FG.responsiveness(te, p)
    r = out["shooter_make_c->MAKE"]
    assert r["n_undefined"] == 40
    assert r["n_defined"] == 60
    assert sum(r["n"]) == 60


# ---------------------------------------------------------------------------
# Decision 8's amended responsiveness gate
# ---------------------------------------------------------------------------
def _driver(steps: int, slope: float | None, span_pp: float) -> dict:
    return {"pred_monotone_steps": steps, "slope_ratio": slope,
            "span_actual": span_pp / 100.0}


def _resp(shooter: dict, defence: dict) -> dict:
    return {"shooter_make_c->MAKE": shooter, "def_allow_c->MAKE": defence}


def test_decision8_rejects_a_model_that_is_flat_at_the_mean():
    # the FGA_3 team baseline: monotone in 4 of 4 steps, and 116x too flat
    resp = _resp(_driver(4, 0.0086, 35.9), _driver(4, 1.012, 1.37))
    for reading in FG.DECISION8_READINGS:
        v = FG.decision8_verdict(resp, reading=reading)
        assert not v["pass"], reading
        assert v["failed_drivers"] == ["shooter_make_c->MAKE"]


def test_decision8_rejects_an_over_steep_driver_too():
    # the FGA_jump2 EB arm: 1.388 is outside the band on the OTHER side
    resp = _resp(_driver(4, 0.839, 8.14), _driver(4, 1.3876, 3.33))
    v = FG.decision8_verdict(resp, reading="strict")
    assert not v["pass"]
    assert v["failed_drivers"] == ["def_allow_c->MAKE"]


def test_decision8_the_two_readings_differ_only_on_a_low_span_driver():
    # the FGA_3 tree: 3/4 steps and slope 0.474 on a 1.37 pp driver
    resp = _resp(_driver(4, 0.988, 35.9), _driver(3, 0.4736, 1.37))
    strict = FG.decision8_verdict(resp, reading="strict")
    lenient = FG.decision8_verdict(resp, reading="low_span_exempt")
    assert not strict["pass"]
    assert strict["by_driver"]["def_allow_c->MAKE"]["steps_ok"]        # 3 of 4 is allowed
    assert not strict["by_driver"]["def_allow_c->MAKE"]["slope_ok"]    # the band is not
    assert lenient["pass"]
    assert not lenient["by_driver"]["def_allow_c->MAKE"]["slope_clause_applies"]


def test_decision8_keeps_the_four_of_four_rule_on_a_high_span_driver():
    resp = _resp(_driver(3, 1.0, 35.9), _driver(4, 1.0, 4.98))
    for reading in FG.DECISION8_READINGS:
        v = FG.decision8_verdict(resp, reading=reading)
        assert not v["pass"], reading
        assert not v["by_driver"]["shooter_make_c->MAKE"]["steps_ok"]


@pytest.mark.parametrize("slope,ok", [(0.8, True), (1.2, True), (0.79, False), (1.21, False)])
def test_decision8_band_edges_are_inclusive(slope, ok):
    resp = _resp(_driver(4, slope, 35.9), _driver(4, 1.0, 4.98))
    assert FG.decision8_verdict(resp, reading="strict")["pass"] is ok


def test_decision8_refuses_an_unknown_reading():
    resp = _resp(_driver(4, 1.0, 35.9), _driver(4, 1.0, 4.98))
    with pytest.raises(KeyError):
        FG.decision8_verdict(resp, reading="whatever_makes_my_arm_win")


def test_score_reports_both_the_superseded_and_the_live_responsiveness_verdict():
    n = 4000
    rng = np.random.default_rng(0)
    te = pd.DataFrame({
        "shooter_make_c": rng.normal(size=n), "def_allow_c": rng.normal(size=n),
        "shooter_att_c": rng.integers(1, 40, n), "def_att_prior": rng.integers(1, 900, n),
        "chance_number": np.ones(n), "class_key": "three",
    })
    te["y"] = (rng.random(n) < 0.34).astype("int8")
    p = np.column_stack([np.full(n, 0.66), np.full(n, 0.34)])
    s = FG.score(te, p)
    assert {"resp_pass", "resp_min_steps", "resp_pass_decision8", "resp_decision8",
            "resp_failed_drivers_decision8"} <= set(s)
    # a constant prediction has no slope at all, so the LIVE gate must reject it
    assert s["resp_pass_decision8"] is False
    assert FG.DECISION8_ADOPTED_READING in FG.DECISION8_READINGS


def test_decision8_constants_match_the_recorded_decision():
    assert FG.SLOPE_BAND == (0.8, 1.2)
    assert FG.LOW_SPAN_PP == 2.0
    assert (FG.MIN_STEPS_DEFAULT, FG.MIN_STEPS_LOW_SPAN) == (4, 3)


def test_efg_is_the_standard_formula_on_the_given_shot_mix():
    # one team, 10 threes (3 made) and 10 rim (6 made):
    # eFG = (9 + 0.5*3) / 20 = 0.525
    te = pd.DataFrame({
        "off_team_id": [1] * 400, "def_team_id": [2] * 400,
        "class_key": ["three"] * 200 + ["rim"] * 200,
        "y": [1] * 60 + [0] * 140 + [1] * 120 + [0] * 80,
        "off_make_c": 0.0, "def_allow_c": 0.0,
    })
    p = te["y"].to_numpy().astype("float64")       # a perfect model
    out = FG.efg_table(te, p, side="offense")
    expected = (180 + 0.5 * 60) / 400 * 100
    assert out["overall_actual_efg_pct"] == pytest.approx(expected, abs=1e-6)
    assert out["overall_implied_efg_pct"] == pytest.approx(expected, abs=1e-6)
    assert out["overall_gap_pp"] == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# 7. Seal, determinism, the simplex
# ===========================================================================
def test_fold_slices_refuse_the_sealed_season():
    from cbb_sim.data.seal import SealedSeasonError

    design = pd.DataFrame({"season": [2022, 2026], "y": [0, 1], "game_id": [1, 2]})
    FG.FOLDS["FSEAL"] = {"train": [2022], "test": [2026]}
    try:
        with pytest.raises(SealedSeasonError):
            FG.fold_slices(design, "FSEAL")
    finally:
        FG.FOLDS.pop("FSEAL")


def test_the_pre_registered_folds_are_what_the_spec_says():
    assert FG.FOLDS["F1"] == {"train": [2022, 2023], "test": [2024]}
    assert FG.FOLDS["F2"] == {"train": [2022, 2023, 2024], "test": [2025]}
    assert FG.SELECTION_FOLD == "F2"
    assert FG.LINEUP_FOLDS["L2"] == {"train": [2024], "test": [2025]}
    assert FG.DEFAULT_VERSION == "v2"


@pytest.mark.parametrize("arm", ["ridge", "lgbm"])
def test_arm_fits_are_deterministic(arm):
    pytest.importorskip("sklearn")
    if arm == "lgbm":
        pytest.importorskip("lightgbm")
    tr, te = _fit_frame(), _fit_frame(n=500, seed=7)
    feats = FG.feature_set("C_plus_state")
    X, y, Xte = FG.design_matrix(tr, feats), tr["y"].to_numpy(), FG.design_matrix(te, feats)
    cls = FG.RidgeArm if arm == "ridge" else FG.LgbmArm
    p1 = cls(seed=0).fit(X, y).predict_proba(Xte)
    p2 = cls(seed=0).fit(X, y).predict_proba(Xte)
    np.testing.assert_array_equal(p1, p2)


def test_probabilities_are_a_valid_simplex_in_the_declared_class_order():
    pytest.importorskip("sklearn")
    tr, te = _fit_frame(), _fit_frame(n=300, seed=3)
    feats = FG.feature_set("A_team")
    m = FG.RidgeArm().fit(FG.design_matrix(tr, feats), tr["y"].to_numpy())
    p = m.predict_proba(FG.design_matrix(te, feats))
    assert p.shape == (len(te), 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert FG.CLASSES == ("MISS", "MAKE")
    assert FG.CLASS_INDEX["MAKE"] == 1


def test_feature_sets_are_nested_in_the_pre_registered_order():
    a, b, c, d = (FG.feature_set(n) for n in FG.FEATURE_SETS)
    assert a == c[:len(a)] == b[:len(a)]
    assert b == c[:len(b)]
    assert c == d[:len(c)]
    assert set(d) - set(c) == set(FG.LINEUP_FEATURES)


def test_the_defender_rate_shrinkage_is_toward_the_league(tmp_path):
    ev = _toy_events()
    ev = ev.assign(class_key=np.where(ev["shot_class"] == "FGA_rim", "rim", "three"))
    raw = FG.defender_rates(ev, prior_att=0)
    shrunk = FG.defender_rates(ev, prior_att=10_000)
    # with a huge prior every defender collapses onto the league's own as-of rate
    m = np.isfinite(shrunk["lg_rim_allow"].to_numpy())
    np.testing.assert_allclose(shrunk.loc[m, "rim_allow"].to_numpy(),
                               shrunk.loc[m, "lg_rim_allow"].to_numpy(), atol=1e-3)
    assert not np.allclose(raw["rim_allow"].to_numpy(), shrunk["rim_allow"].to_numpy())
