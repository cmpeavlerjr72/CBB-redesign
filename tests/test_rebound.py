"""rebound tests: the opportunity derivation, leak safety, determinism, and the
version flag.

What is worth a dedicated test here, and why:

1. **The three outcome classes come off the event that carries them.** The
   possession tables cannot express a dead-ball rebound at all
   (`possessions.py` treats `DeadBallReb` as a no-op), so the whole target of
   this model depends on reading the NEXT event correctly, including the cases
   that are not a rebound at all. Each branch is checked on a hand-built event
   stream where the right answer is known by inspection.
2. **A trip's non-final missed free throw is not a rebound opportunity.** The
   ball is dead and the same shooter shoots again. Getting this wrong would add
   ~25,000 phantom opportunities per season with a wildly wrong OREB rate, and
   it depends on the administrative-rebound drop in
   `cbb_sim.models.event_stream`, which is exactly the kind of thing that
   silently regresses.
3. **A game's own events cannot enter its own features.** Same off-by-one
   hazard the L3 team-form builder has, proved the same two ways: an
   independent recomputation of the strictly-earlier average, and invariance to
   corrupting the game's own rows.
4. **The seal holds and fits are deterministic**, so a bake-off difference is
   never a re-run artefact and season 2026 can never enter a fold.
5. **The version flag is read from the build, not assumed.**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

HOME, AWAY = 100, 200
SEASON = 2025


# ===========================================================================
# Synthetic CBBD play frames
# ===========================================================================
def _play(idx: int, play_type: str, is_home: bool, sec: int, made: bool | None = None,
          shot_range=None, player_id: float = 1.0, period: int = 1,
          hs: int = 0, as_: int = 0, game: int = 1) -> dict:
    shooting = play_type in ("JumpShot", "LayUpShot", "DunkShot", "TipShot", "MadeFreeThrow")
    return {
        "gameId": game, "season": SEASON, "id": idx, "playType": play_type,
        "isHomeTeam": is_home, "teamId": 1 if is_home else 2,
        "opponentId": 2 if is_home else 1,
        "homeScore": hs, "awayScore": as_, "period": period, "secondsRemaining": sec,
        "scoringPlay": bool(made) if made is not None else False,
        "shootingPlay": shooting,
        "scoreValue": 0, "shot_made": made, "shot_range": shot_range,
        "playText": "", "shot_shooter_id": player_id,
        "shot_location_x": np.nan, "shot_location_y": np.nan,
        "participant_1_id": player_id,
        **{f"home_on_{k}": float(k) for k in range(1, 6)},
        **{f"away_on_{k}": float(10 + k) for k in range(1, 6)},
    }


def _stream(tmp_path: Path, rows: list[dict]) -> pd.DataFrame:
    """Write a synthetic season of plays and run the real loader over it."""
    df = pd.DataFrame(rows)[list(ES.STREAM_COLUMNS)]
    d = tmp_path / "pbp"
    d.mkdir(exist_ok=True)
    df.to_parquet(d / f"plays_{SEASON}.parquet", index=False)
    universe = pd.DataFrame({
        "game_id": [1000 + g for g in sorted(df["gameId"].unique())],
        "cbbd_game_id": sorted(df["gameId"].unique()),
        "season": SEASON, "game_date": pd.to_datetime("2025-01-01"),
        "neutral_site": False, "home_team_id": HOME, "away_team_id": AWAY,
        "is_d1_game": True, "pbp_truncated": False,
    })
    return ES.build_stream(SEASON, universe, rim_override_max_ft=0.0, pbp_dir=d)


def _opportunities(tmp_path: Path, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)[list(ES.STREAM_COLUMNS)]
    d = tmp_path / "pbp"
    d.mkdir(exist_ok=True)
    df.to_parquet(d / f"plays_{SEASON}.parquet", index=False)
    universe = pd.DataFrame({
        "game_id": [1000 + g for g in sorted(df["gameId"].unique())],
        "cbbd_game_id": sorted(df["gameId"].unique()),
        "season": SEASON, "game_date": pd.to_datetime("2025-01-01"),
        "neutral_site": False, "home_team_id": HOME, "away_team_id": AWAY,
        "is_d1_game": True, "pbp_truncated": False,
    })
    return RB._season_opportunities(SEASON, universe, 0.0, d)


# ===========================================================================
# 1. Outcome classes
# ===========================================================================
def test_offensive_rebound_is_an_oreb_and_starts_a_continuation(tmp_path):
    opp = _opportunities(tmp_path, [
        _play(1, "JumpShot", True, 1190, made=False, shot_range="jumper"),
        _play(2, "Offensive Rebound", True, 1188),
        _play(3, "LayUpShot", True, 1185, made=False, shot_range="rim"),
        _play(4, "Defensive Rebound", False, 1183),
        _play(5, "End Period", True, 0),
    ])
    assert list(opp["outcome"]) == ["OREB", "DREB"]
    assert list(opp["miss_type"]) == ["jump2", "rim"]
    # the second opportunity is a CONTINUATION of the same possession
    assert list(opp["chance_index"]) == [0, 1]
    assert (opp["off_team_id"] == HOME).all()


def test_defensive_rebound_is_a_dreb_charged_to_the_shooting_team(tmp_path):
    opp = _opportunities(tmp_path, [
        _play(1, "JumpShot", False, 1190, made=False, shot_range="three_pointer"),
        _play(2, "Defensive Rebound", True, 1188),
        _play(3, "End Period", True, 0),
    ])
    assert opp.iloc[0]["outcome"] == "DREB"
    assert opp.iloc[0]["miss_type"] == "three"
    assert opp.iloc[0]["off_team_id"] == AWAY      # the team that MISSED
    assert opp.iloc[0]["def_team_id"] == HOME


def test_dead_ball_rebound_is_its_own_class(tmp_path):
    opp = _opportunities(tmp_path, [
        _play(1, "LayUpShot", True, 1190, made=False, shot_range="rim"),
        _play(2, "Dead Ball Rebound", False, 1188),
        _play(3, "End Period", True, 0),
    ])
    assert opp.iloc[0]["outcome"] == "DEAD"


def test_a_rebound_the_feed_never_logs_is_unresolved_not_guessed(tmp_path):
    opp = _opportunities(tmp_path, [
        _play(1, "LayUpShot", True, 1190, made=False, shot_range="rim"),
        _play(2, "PersonalFoul", False, 1188),      # loose-ball foul, no rebound row
        _play(3, "End Period", True, 0),
    ])
    assert opp.iloc[0]["outcome"] == "unresolved"


def test_a_made_shot_is_not_a_rebound_opportunity(tmp_path):
    opp = _opportunities(tmp_path, [
        _play(1, "DunkShot", True, 1190, made=True, shot_range="rim"),
        _play(2, "JumpShot", False, 1180, made=False, shot_range="jumper"),
        _play(3, "Defensive Rebound", True, 1178),
        _play(4, "End Period", True, 0),
    ])
    assert len(opp) == 1
    assert opp.iloc[0]["off_team_id"] == AWAY


# ===========================================================================
# 2. Free-throw trips
# ===========================================================================
def test_only_the_last_free_throw_of_a_trip_is_a_rebound_opportunity(tmp_path):
    """A missed first of two is not an opportunity: the ball is dead and the
    same shooter shoots again. The administrative rebound ESPN logs between the
    two must not be read as a live rebound either."""
    opp = _opportunities(tmp_path, [
        _play(1, "PersonalFoul", False, 1190),
        _play(2, "MadeFreeThrow", True, 1190, made=False, shot_range="free_throw"),
        _play(3, "Offensive Rebound", True, 1190),      # ADMINISTRATIVE reset
        _play(4, "MadeFreeThrow", True, 1190, made=False, shot_range="free_throw"),
        _play(5, "Defensive Rebound", False, 1188),
        _play(6, "End Period", True, 0),
    ])
    assert len(opp) == 1
    assert opp.iloc[0]["miss_type"] == "ft"
    assert opp.iloc[0]["outcome"] == "DREB"


def test_a_made_last_free_throw_is_not_an_opportunity(tmp_path):
    opp = _opportunities(tmp_path, [
        _play(1, "PersonalFoul", False, 1190),
        _play(2, "MadeFreeThrow", True, 1190, made=False, shot_range="free_throw"),
        _play(3, "Offensive Rebound", True, 1190),
        _play(4, "MadeFreeThrow", True, 1190, made=True, shot_range="free_throw"),
        _play(5, "End Period", True, 0),
    ])
    assert len(opp) == 0


def test_a_missed_technical_free_throw_is_not_a_live_rebound(tmp_path):
    opp = _opportunities(tmp_path, [
        _play(1, "Technical Foul", False, 1190),
        _play(2, "MadeFreeThrow", True, 1190, made=False, shot_range="free_throw"),
        _play(3, "Defensive Rebound", False, 1188),
        _play(4, "End Period", True, 0),
    ])
    assert len(opp) == 0


# ===========================================================================
# 3. The blocked flag
# ===========================================================================
def test_a_block_row_becomes_a_flag_and_stops_being_an_event(tmp_path):
    st = _stream(tmp_path, [
        _play(1, "LayUpShot", True, 1190, made=False, shot_range="rim"),
        _play(2, "Block Shot", False, 1190),
        _play(3, "Defensive Rebound", False, 1188),
        _play(4, "End Period", True, 0),
    ])
    assert "block" not in set(st["cls"])
    shot = st[st["cls"] == "FGA_rim"].iloc[0]
    assert bool(shot["blocked"]) is True
    opp = _opportunities(tmp_path, [
        _play(1, "LayUpShot", True, 1190, made=False, shot_range="rim"),
        _play(2, "Block Shot", False, 1190),
        _play(3, "Defensive Rebound", False, 1188),
        _play(4, "End Period", True, 0),
    ])
    assert bool(opp.iloc[0]["blocked"]) is True
    assert opp.iloc[0]["outcome"] == "DREB"      # the block did not hide the rebound


# ===========================================================================
# 4. Leak safety of the team form
# ===========================================================================
def _toy_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three games on three dates. Team 10 plays all three; the OREB counts are
    chosen so the strictly-earlier average is obvious by hand."""
    rows = []

    def add(game, off, deff, n_oreb, n_dreb, chance_index=0):
        for _ in range(n_oreb):
            rows.append({"season": 2024, "game_id": game, "off_team_id": off,
                         "def_team_id": deff, "outcome": "OREB", "chance_index": chance_index})
        for _ in range(n_dreb):
            rows.append({"season": 2024, "game_id": game, "off_team_id": off,
                         "def_team_id": deff, "outcome": "DREB", "chance_index": chance_index})

    add(1, 10, 100, 4, 6)
    add(1, 100, 10, 2, 8)
    add(2, 10, 200, 6, 4)
    add(2, 200, 10, 1, 9)
    add(3, 10, 300, 1, 9)
    add(3, 300, 10, 5, 5)
    ev = pd.DataFrame(rows)
    universe = pd.DataFrame({"game_id": [1, 2, 3],
                             "game_date": pd.to_datetime(["2024-11-01", "2024-11-08", "2024-11-15"])})
    return ev, universe


def test_team_form_uses_only_strictly_earlier_games():
    ev, universe = _toy_events()
    form = RB.team_rebound_form(ev, universe)
    t10 = form[form["team_id"] == 10].sort_values("game_date")

    g1 = t10[t10["game_id"] == 1].iloc[0]
    assert g1["n_prior_off"] == 0
    assert g1["off_oreb_c"] == pytest.approx(0.0)

    # going into game 3, team 10's own OREB rate is (4 + 6) / (10 + 10) = 0.5
    # and the league's is (4+2+6+1) / 40 = 0.325, both over games 1-2 only.
    g3 = t10[t10["game_id"] == 3].iloc[0]
    assert g3["n_prior_off"] == 2
    assert g3["off_oreb_c"] == pytest.approx(0.5 - 13 / 40, abs=1e-5)


def test_a_games_own_events_cannot_change_its_own_features():
    ev, universe = _toy_events()
    before = RB.team_rebound_form(ev, universe)
    corrupted = ev.copy()
    mask = (corrupted["game_id"] == 3) & (corrupted["off_team_id"] == 10)
    corrupted.loc[mask, "outcome"] = "OREB"          # every miss now rebounded by the offence
    after = RB.team_rebound_form(corrupted, universe)

    cols = ["off_oreb_c", "def_dreb_c"]
    b = before[(before["game_id"] == 3) & (before["team_id"] == 10)][cols].to_numpy()
    a = after[(after["game_id"] == 3) & (after["team_id"] == 10)][cols].to_numpy()
    np.testing.assert_array_equal(a, b)

    # ... and a LATER game must move, or the check above would pass for a
    # builder that ignores the data entirely.
    extra = ev[ev["game_id"] == 3].assign(game_id=4)
    extra_c = corrupted[corrupted["game_id"] == 3].assign(game_id=4)
    u4 = pd.concat([universe, pd.DataFrame({"game_id": [4],
                                            "game_date": pd.to_datetime(["2024-11-22"])})],
                   ignore_index=True)
    b4 = RB.team_rebound_form(pd.concat([ev, extra], ignore_index=True), u4)
    a4 = RB.team_rebound_form(pd.concat([corrupted, extra_c], ignore_index=True), u4)
    b4v = b4[(b4["game_id"] == 4) & (b4["team_id"] == 10)][cols].to_numpy()
    a4v = a4[(a4["game_id"] == 4) & (a4["team_id"] == 10)][cols].to_numpy()
    assert not np.allclose(a4v, b4v)


def test_defence_form_is_what_opponents_did_to_this_team():
    ev, universe = _toy_events()
    form = RB.team_rebound_form(ev, universe)
    # team 10's defensive form going into game 3 comes from what teams 100 and
    # 200 did against it in games 1-2: 3 OREB in 20, so DREB rate 17/20.
    g3 = form[(form["game_id"] == 3) & (form["team_id"] == 10)].iloc[0]
    lg_dreb = 1.0 - 13 / 40
    assert g3["def_dreb_c"] == pytest.approx(17 / 20 - lg_dreb, abs=1e-5)


def test_first_chance_only_is_the_default_and_actually_restricts():
    ev, universe = _toy_events()
    cont = ev[ev["game_id"] == 1].assign(chance_index=1, outcome="OREB")
    both = pd.concat([ev, cont], ignore_index=True)
    first = RB.team_rebound_form(both, universe, first_chance_only=True)
    pooled = RB.team_rebound_form(both, universe, first_chance_only=False)
    a = first[(first["game_id"] == 2) & (first["team_id"] == 10)]["off_oreb_c"].iloc[0]
    b = pooled[(pooled["game_id"] == 2) & (pooled["team_id"] == 10)]["off_oreb_c"].iloc[0]
    assert a != b


# ===========================================================================
# 5. Seal, determinism, and the simplex
# ===========================================================================
def _fit_frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(size=n).astype("float32")
                      for f in RB.feature_set("C_plus_state")})
    d["y"] = rng.integers(0, len(RB.CLASSES), n).astype("int8")
    d["season"] = 2024
    d["game_id"] = rng.integers(0, 200, n)
    d["miss_type"] = rng.choice(list(RB.MISS_TYPES), n)
    return d


def test_fold_slices_refuse_the_sealed_season():
    from cbb_sim.data.seal import SealedSeasonError

    design = pd.DataFrame({"season": [2022, 2026], "y": [0, 1], "game_id": [1, 2]})
    RB.FOLDS["FSEAL"] = {"train": [2022], "test": [2026]}
    try:
        with pytest.raises(SealedSeasonError):
            RB.fold_slices(design, "FSEAL")
    finally:
        RB.FOLDS.pop("FSEAL")


@pytest.mark.parametrize("arm", ["ridge_logit", "lgbm"])
def test_arm_fits_are_deterministic(arm):
    pytest.importorskip("sklearn")
    if arm == "lgbm":
        pytest.importorskip("lightgbm")
    tr = _fit_frame()
    te = _fit_frame(n=500, seed=7)
    feats = RB.feature_set("C_plus_state")
    p1 = RB.predict_arm(arm, RB.fit_arm(arm, tr, feats, seed=0), te, feats)
    p2 = RB.predict_arm(arm, RB.fit_arm(arm, tr, feats, seed=0), te, feats)
    np.testing.assert_array_equal(p1, p2)


def test_probabilities_are_a_valid_simplex_in_the_declared_class_order():
    pytest.importorskip("sklearn")
    tr = _fit_frame()
    te = _fit_frame(n=300, seed=3)
    feats = RB.feature_set("C_plus_state")
    p = RB.predict_arm("ridge_logit", RB.fit_arm("ridge_logit", tr, feats), te, feats)
    assert p.shape == (len(te), len(RB.CLASSES))
    assert (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert RB.CLASSES == ("OREB", "DREB", "DEAD")
    assert RB.CLASS_INDEX["OREB"] == 0


def test_baseline_is_a_share_table_by_miss_type():
    tr = _fit_frame()
    b = RB.BaselineArm().fit(tr["y"].to_numpy(), tr["miss_type"].to_numpy(),
                             tr["season"].to_numpy())
    for m in RB.MISS_TYPES:
        sel = tr["miss_type"].to_numpy() == m
        expected = np.bincount(tr["y"].to_numpy()[sel], minlength=len(RB.CLASSES)) / sel.sum()
        np.testing.assert_allclose(b.table_[m], expected)


def test_dead_ball_composition_is_a_valid_simplex_with_the_fixed_share():
    p_oreb = np.array([0.1, 0.5, 0.9])
    mt = np.array(["rim", "three", "ft"])
    share = {"rim": 0.004, "three": 0.008, "ft": 0.007}
    p = RB.compose_binary_plus_fixed_dead(p_oreb, mt, share)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(p[:, RB.CLASS_INDEX["DEAD"]], [0.004, 0.008, 0.007])


# ===========================================================================
# 6. The version flag
# ===========================================================================
def test_rim_override_is_read_from_the_build_report_not_assumed(tmp_path):
    d = tmp_path / "poss"
    d.mkdir()
    assert ES.rim_override_for_version(poss_dir=d) == 0.0      # no report -> no override
    (d / "build_report.json").write_text(json.dumps({"rim_override": {"max_ft": 2.27}}))
    assert ES.rim_override_for_version(poss_dir=d) == pytest.approx(2.27)


def test_the_override_moves_only_the_miss_type_never_the_outcome(tmp_path):
    """The rim override can only turn an `FGA_jump2` into an `FGA_rim`. So for
    THIS model it moves a feature and nothing else: same opportunities, same
    outcomes, same order."""
    rows = [
        _play(1, "JumpShot", True, 1190, made=False, shot_range="jumper"),
        _play(2, "Defensive Rebound", False, 1188),
        _play(3, "End Period", True, 0),
    ]
    rows[0]["shot_location_x"] = 52.5     # on the basket: the override fires
    rows[0]["shot_location_y"] = 250.0
    df = pd.DataFrame(rows)[list(ES.STREAM_COLUMNS)]
    d = tmp_path / "pbp"
    d.mkdir()
    df.to_parquet(d / f"plays_{SEASON}.parquet", index=False)
    universe = pd.DataFrame({
        "game_id": [1001], "cbbd_game_id": [1], "season": [SEASON],
        "game_date": pd.to_datetime(["2025-01-01"]), "neutral_site": [False],
        "home_team_id": [HOME], "away_team_id": [AWAY],
        "is_d1_game": [True], "pbp_truncated": [False]})
    off = RB._season_opportunities(SEASON, universe, 0.0, d)
    on = RB._season_opportunities(SEASON, universe, 2.27, d)
    assert off.iloc[0]["miss_type"] == "jump2"
    assert on.iloc[0]["miss_type"] == "rim"
    assert list(off["outcome"]) == list(on["outcome"])
    assert len(off) == len(on) == 1
