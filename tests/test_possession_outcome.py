"""possession-outcome tests: the event-vocabulary guardrail, the segmentation
invariants, feature leak-safety, and determinism.

What is worth a dedicated test here, and why:

1. **Every CBBD `playType` is explicitly mapped and an unknown one raises.**
   `docs/SIM_GUARDRAILS.md` section 4 makes this a standing requirement:
   hoopR and CBBD both added event types mid-history (Substitution in 2025,
   Coach's Challenge and a bare "Shot" in 2026), and a silent default would
   have mis-segmented every possession containing one.
2. **The segmentation invariants**, checked on hand-built event streams where
   the right answer is known by inspection: an offensive rebound continues a
   possession instead of starting one, a made basket ends it, an and-one folds
   the free throw into the field-goal attempt, technical free throws change
   nothing, and the bonus rules classify a free-throw trip the way the NCAA
   rulebook says they should.
3. **A game's own events cannot enter its own features.** The team-form
   builder is a `shift(1)` expanding mean, which is the classic off-by-one:
   a bare `.expanding()` would include the game itself. This is verified two
   ways -- by recomputing the expected value independently from the raw
   team-game table (which proves the feature IS the strictly-earlier average,
   stronger than showing it is merely insensitive to one perturbation), and by
   rebuilding the features with the game's own row perturbed and requiring the
   value to be unchanged.
4. **Determinism.** Two identical fits of an arm must produce bit-identical
   predictions, so a bake-off difference is never a re-run artefact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.pbp import events as EV  # noqa: E402
from cbb_sim.pbp import possessions as PS  # noqa: E402

#: Every `playType` observed in data/raw/cbbd/pbp/plays_{2022..2026}.parquet.
#: Source: docs/tests/data_audit_cbbd_pbp_2026-09-10.md, "Union playType
#: vocabulary size (event dictionary): 24".
CBBD_PLAY_TYPES_2022_2026 = [
    "JumpShot", "Defensive Rebound", "LayUpShot", "MadeFreeThrow", "PersonalFoul",
    "Substitution", "Lost Ball Turnover", "Offensive Rebound", "Steal", "OfficialTVTimeOut",
    "Block Shot", "DunkShot", "ShortTimeOut", "Dead Ball Rebound", "End Period", "TipShot",
    "End Game", "Jumpball", "RegularTimeOut", "Technical Foul",
    "Coach's Challenge (Stands)", "Coach's Challenge (Overturned)", "Not Available", "Shot",
]


# ===========================================================================
# 1. Event vocabulary guardrail
# ===========================================================================
def test_every_observed_play_type_is_mapped():
    assert len(CBBD_PLAY_TYPES_2022_2026) == 24
    for pt in CBBD_PLAY_TYPES_2022_2026:
        assert pt in EV.PLAY_TYPE_TO_EVENT, f"{pt!r} is not in PLAY_TYPE_TO_EVENT"
        assert EV.map_play_type(pt)


def test_mapping_table_has_no_extra_entries():
    """The table must be the observed vocabulary exactly -- an entry for a type
    that does not exist is a guess, and guesses are what the guardrail bans."""
    assert set(EV.PLAY_TYPE_TO_EVENT) == set(CBBD_PLAY_TYPES_2022_2026)


def test_unknown_play_type_raises():
    with pytest.raises(EV.UnknownPlayTypeError):
        EV.map_play_type("Quadruple Technical Foul (Overturned)")
    with pytest.raises(EV.UnknownPlayTypeError):
        EV.assert_known_play_types(["JumpShot", "A Brand New 2027 Event Type"])


# ===========================================================================
# 2. Row classification: 2 vs 3, and free throws
# ===========================================================================
def _plays(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal plays frame with the columns classify_frame needs."""
    defaults = dict(playType="JumpShot", shot_range=None, playText=None, scoreValue=0,
                    shot_made=None, scoringPlay=False, shootingPlay=False,
                    shot_location_x=None, shot_location_y=None)
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_three_point_rule_prefers_shot_range():
    df = _plays([
        {"playType": "JumpShot", "shot_range": "three_pointer", "playText": "X missed Three Point Jumper.",
         "scoreValue": 3, "shootingPlay": True},
        {"playType": "JumpShot", "shot_range": "jumper", "playText": "X missed Jumper.",
         "scoreValue": 2, "shootingPlay": True},
        # the documented 152-row failure mode: scoreValue says 3, range and text say two
        {"playType": "JumpShot", "shot_range": "jumper", "playText": "X misses 25-foot turnaround jump shot",
         "scoreValue": 3, "shootingPlay": True},
        # playText null (2,845 rows): shot_range still resolves it
        {"playType": "JumpShot", "shot_range": "three_pointer", "playText": None,
         "scoreValue": 0, "shootingPlay": True},
    ])
    assert list(EV.classify_frame(df)) == ["FGA_3", "FGA_jump2", "FGA_jump2", "FGA_3"]


def test_free_throw_type_covers_makes_and_misses():
    df = _plays([
        {"playType": "MadeFreeThrow", "shot_range": "free_throw", "playText": "X made Free Throw.",
         "shot_made": True, "scoringPlay": True, "scoreValue": 1, "shootingPlay": True},
        {"playType": "MadeFreeThrow", "shot_range": "free_throw", "playText": "X missed Free Throw.",
         "shot_made": False, "scoringPlay": False, "scoreValue": 1, "shootingPlay": True},
        # 2026 text format change: "makes"/"misses", and sometimes null text
        {"playType": "MadeFreeThrow", "shot_range": "free_throw", "playText": None,
         "shot_made": True, "scoringPlay": True, "scoreValue": 1, "shootingPlay": True},
    ])
    assert list(EV.classify_frame(df)) == ["FT_made", "FT_missed", "FT_made"]


def test_rim_types_and_inert_types():
    df = _plays([
        {"playType": "DunkShot", "shot_range": "rim", "shootingPlay": True, "shot_made": True},
        {"playType": "LayUpShot", "shot_range": "rim", "shootingPlay": True, "shot_made": False},
        {"playType": "TipShot", "shot_range": "rim", "shootingPlay": True, "shot_made": False},
        {"playType": "Substitution"}, {"playType": "Block Shot"}, {"playType": "Jumpball"},
        {"playType": "Coach's Challenge (Stands)"}, {"playType": "Not Available"},
    ])
    got = list(EV.classify_frame(df))
    assert got[:3] == ["FGA_rim", "FGA_rim", "FGA_rim"]
    assert set(got[3:7]) <= EV.INERT_CLASSES
    assert got[7] == "unknown"


# ===========================================================================
# 3. Free-throw trip classification (the NCAA bonus rules)
# ===========================================================================
@pytest.mark.parametrize(
    "n_ft,first_made,prior_fouls,expected",
    [
        # three free throws are always a shooting foul on a three-point attempt
        (3, True, 0, "FT_trip_shooting"),
        (3, False, 12, "FT_trip_shooting"),
        # no bonus in force: a non-shooting foul produces no free throws at all
        (2, True, 0, "FT_trip_shooting"),
        (2, False, 5, "FT_trip_shooting"),
        (1, False, 3, "FT_trip_shooting"),
        # one-and-one window (7th..9th team foul): a MISSED single is the front end
        (1, False, 6, "FT_trip_bonus"),
        (1, False, 8, "FT_trip_bonus"),
        # ... but two shots after a MISSED first cannot be a one-and-one
        (2, False, 7, "FT_trip_shooting"),
        (2, True, 7, "FT_trip_bonus"),
        # double bonus (10th team foul on): two shots either way
        (2, True, 9, "FT_trip_bonus"),
        (2, False, 11, "FT_trip_bonus"),
    ],
)
def test_classify_ft_trip(n_ft, first_made, prior_fouls, expected):
    got, _ = PS.classify_ft_trip(n_ft, first_made, prior_fouls)
    assert got == expected


def test_ambiguity_is_flagged_exactly_where_the_feed_cannot_decide():
    """Once the bonus is in force a two-shot trip is consistent with BOTH a
    shooting foul and a bonus trip. Those rows -- and only those -- must carry
    the ambiguity flag."""
    _, amb_no_bonus = PS.classify_ft_trip(2, True, 3)
    _, amb_one_and_one = PS.classify_ft_trip(2, True, 7)
    _, amb_double = PS.classify_ft_trip(2, True, 10)
    _, amb_three = PS.classify_ft_trip(3, True, 12)
    assert amb_no_bonus is False
    assert amb_one_and_one is True
    assert amb_double is True
    assert amb_three is False


# ===========================================================================
# 4. Segmentation invariants on hand-built streams
# ===========================================================================
HOME, AWAY = 100, 200


def _segment(events: list[tuple]) -> pd.DataFrame:
    """Run the state machine over an explicit (class, side, seconds, made) list.

    side is 0 for the home team and 1 for the away team, matching the machine's
    internal convention; `_emit` maps them to the ESPN ids below."""
    n = len(events)
    ev = {
        "cls": np.array([e[0] for e in events], dtype=object),
        "team": np.array([e[1] for e in events], dtype="int64"),
        "sec": np.array([e[2] for e in events], dtype="int64"),
        "period": np.ones(n, dtype="int64"),
        "hs": np.zeros(n, dtype="int64"),
        "as_": np.zeros(n, dtype="int64"),
        "made": np.array([bool(e[3]) for e in events], dtype=bool),
        "stolen": np.zeros(n, dtype=bool),
        "on_floor": None,
    }
    meta = {"game_id": 1, "cbbd_game_id": 1, "season": 2025,
            "home_team_id": HOME, "away_team_id": AWAY}
    m = PS._GameMachine(meta, ev)
    m.run()
    poss_rows, chance_rows = [], []
    PS._emit(m, poss_rows, chance_rows)
    return pd.DataFrame(poss_rows)


def test_made_basket_ends_the_possession_and_oreb_does_not():
    poss = _segment([
        ("FGA_3", 0, 1190, False),      # home misses
        ("OREB", 0, 1188, False),       # home rebounds: SAME possession, new chance
        ("FGA_rim", 0, 1185, True),     # home scores: possession ends
        ("FGA_jump2", 1, 1170, False),  # away misses
        ("DREB", 0, 1168, False),       # home rebounds: away's possession ends
        ("end_period", -1, 0, False),
    ])
    assert len(poss) == 2
    first = poss.iloc[0]
    assert first["offense_team_id"] == HOME
    assert first["n_chances"] == 2 and first["oreb_count"] == 1
    assert first["terminal_event"] == "FGA_rim"
    assert first["points"] == 2
    second = poss.iloc[1]
    assert second["offense_team_id"] == AWAY
    assert second["terminal_event"] == "FGA_jump2"
    assert second["n_chances"] == 1


def test_turnover_ends_the_possession_and_charges_the_losing_team():
    poss = _segment([("TOV", 1, 1150, False), ("FGA_3", 0, 1140, True), ("end_period", -1, 0, False)])
    assert poss.iloc[0]["offense_team_id"] == AWAY
    assert poss.iloc[0]["terminal_event"] == "TOV"


def test_and_one_is_a_field_goal_attempt_not_a_free_throw_trip():
    poss = _segment([
        ("FGA_rim", 0, 1100, True),
        ("foul", 1, 1100, False),
        ("FT_made", 0, 1100, True),
        ("end_period", -1, 0, False),
    ])
    assert len(poss) == 1
    r = poss.iloc[0]
    assert r["terminal_event"] == "FGA_rim"
    assert bool(r["and_one"]) is True
    assert r["fta"] == 1 and r["ftm"] == 1
    assert r["points"] == 3


def test_shooting_foul_on_a_miss_is_its_own_terminal_class():
    """ESPN does not log the field-goal attempt when a shooting foul occurs on
    a MISS, so the trip itself is the terminal event -- verified in the audit:
    of 1,982 fouls both preceded by a shot at the same clock and followed by
    free throws, 1,942 follow a MADE shot."""
    poss = _segment([
        ("foul", 1, 1050, False),
        ("FT_missed", 0, 1050, False),
        ("FT_made", 0, 1050, True),
        ("end_period", -1, 0, False),
    ])
    r = poss.iloc[0]
    assert r["terminal_event"] == "FT_trip_shooting"
    assert r["offense_team_id"] == HOME
    assert r["fta"] == 2 and r["ftm"] == 1 and r["points"] == 1
    assert bool(r["and_one"]) is False


def test_bonus_trip_is_labelled_from_the_running_team_foul_count():
    stream = [("foul", 1, 1100 - 5 * k, False) for k in range(6)]   # away's 1st..6th fouls
    stream += [("foul", 1, 1060, False), ("FT_missed", 0, 1060, False)]  # the 7th: one-and-one
    stream += [("DREB", 1, 1058, False), ("end_period", -1, 0, False)]
    poss = _segment(stream)
    trip = poss[poss["terminal_event"] == "FT_trip_bonus"]
    assert len(trip) == 1
    # The trip is classified from the count BEFORE the foul (6 -> the 7th foul
    # is a one-and-one). The possession row's own `def_team_fouls` is stamped
    # when the possession OPENS, and in this synthetic stream the possession
    # opens at the free throw itself -- after the foul was counted -- so it
    # reads 7. In real play the possession is already open from the offence's
    # earlier action and the stamp is the pre-foul count.
    assert trip.iloc[0]["def_team_fouls"] == 7
    assert bool(trip.iloc[0]["off_in_bonus"]) is True
    # ... and the 6th foul (prior count 5) must NOT produce a bonus trip
    early = [("foul", 1, 1100 - 5 * k, False) for k in range(5)]
    early += [("foul", 1, 1070, False), ("FT_missed", 0, 1070, False),
              ("DREB", 1, 1068, False), ("end_period", -1, 0, False)]
    assert _segment(early).iloc[0]["terminal_event"] == "FT_trip_shooting"


def test_technical_free_throws_change_no_possession_but_keep_their_points():
    poss = _segment([
        ("FGA_3", 0, 1000, False),
        ("technical", 1, 998, False),
        ("FT_made", 0, 998, True),
        ("FT_made", 0, 998, True),
        ("DREB", 1, 995, False),
        ("end_period", -1, 0, False),
    ])
    assert len(poss) == 1
    r = poss.iloc[0]
    assert r["terminal_event"] == "FGA_3"
    assert r["fta"] == 0 and r["ftm"] == 0 and r["points"] == 0
    assert r["tech_points_off"] == 2


def test_administrative_rebound_between_free_throws_is_not_a_chance():
    poss = _segment([
        ("foul", 1, 900, False),
        ("FT_missed", 0, 900, False),
        ("OREB", 0, 900, False),      # the dead-ball reset ESPN logs mid-trip
        ("FT_made", 0, 900, True),
        ("end_period", -1, 0, False),
    ])
    r = poss.iloc[0]
    assert r["n_chances"] == 1 and r["oreb_count"] == 0
    assert r["fta"] == 2 and r["ftm"] == 1


def test_mismatch_guard_does_not_delete_a_possession():
    """An unlogged change of possession must close the open possession as
    `unknown`, not silently merge it into the next team's."""
    poss = _segment([("FGA_3", 0, 800, False), ("FGA_rim", 1, 790, True), ("end_period", -1, 0, False)])
    assert len(poss) == 2
    assert poss.iloc[0]["offense_team_id"] == HOME
    assert poss.iloc[0]["terminal_event"] == "FGA_3"
    assert poss.iloc[1]["offense_team_id"] == AWAY


def test_period_end_closes_a_live_possession():
    poss = _segment([("FGA_3", 0, 3, False), ("end_period", -1, 0, False)])
    assert poss.iloc[0]["terminal_event"] == "end_period"


def test_flipped_is_home_flag_is_repaired_from_the_running_score():
    """CBBD inverts `isHomeTeam` on a handful of games per season; the running
    score columns settle it. Left unrepaired, an entire team's scoring lands on
    the opponent."""
    plays = pd.DataFrame({
        "gameId": [7] * 4,
        "isHomeTeam": [True, True, False, False],   # deliberately inverted
        "teamId": [1.0, 1.0, 2.0, 2.0],
        "homeScore": [0, 0, 2, 4],                  # the AWAY-flagged team drives homeScore
        "awayScore": [2, 4, 4, 4],
    })
    side = np.where(plays["isHomeTeam"].to_numpy(), 0, 1)
    fixed = PS._fix_flipped_sides(plays, side, np.ones(4, dtype=bool))
    assert list(fixed) == [1, 1, 0, 0]


# ===========================================================================
# 5. Feature leak safety
# ===========================================================================
def _toy_possessions() -> tuple[dict, pd.DataFrame]:
    """Three games for one team, on three dates, with deliberately different
    shot mixes so that a leaked value would be obvious."""
    rows = []
    spec = [
        # (game_id, date, team A rows, team B rows) -- (rim, jump2, three, fta, tov)
        (1, "2024-11-05", 10, 100, (10, 5, 5, 4, 3), (5, 5, 10, 2, 6)),
        (2, "2024-11-10", 10, 200, (2, 2, 16, 0, 1), (8, 8, 4, 6, 5)),
        (3, "2024-11-20", 10, 300, (16, 2, 2, 10, 8), (6, 6, 6, 4, 4)),
    ]
    for gid, _date, a, b, ra, rb in spec:
        for team, opp, (rim, j2, th, fta, tov) in ((a, b, ra), (b, a, rb)):
            for k in range(rim + j2 + th + tov):
                is_tov = k >= rim + j2 + th
                rows.append({
                    "season": 2024, "game_id": gid, "offense_team_id": team,
                    "defense_team_id": opp, "poss_index": k + 1,
                    "terminal_event": "TOV" if is_tov else "FGA_rim",
                    "fga_rim": 1 if k < rim else 0,
                    "fga_jump2": 1 if rim <= k < rim + j2 else 0,
                    "fga_3": 1 if rim + j2 <= k < rim + j2 + th else 0,
                    "fta": fta if k == 0 else 0, "points": 0,
                })
    poss = pd.DataFrame(rows)
    universe = pd.DataFrame({
        "game_id": [1, 2, 3],
        "game_date": pd.to_datetime(["2024-11-05", "2024-11-10", "2024-11-20"]),
    })
    return {2024: poss}, universe


def test_team_form_uses_only_strictly_earlier_games():
    poss_by_season, universe = _toy_possessions()
    form = PO.build_team_form(poss_by_season, universe, style_source="all_chances")
    team = form[form["team_id"] == 10].sort_values("game_date")

    # game 1 is the team's first: no prior games, so every centred rate is
    # exactly 0.0 -- the league mean, not a fabricated value
    g1 = team[team["game_id"] == 1].iloc[0]
    assert g1["n_prior_off"] == 0
    for r in PO.RATE_DEFS:
        assert g1[f"off_{r}_c"] == pytest.approx(0.0)

    # game 3's offence rate must be the ratio of cumulative sums over games 1-2
    # ONLY, recomputed here independently of the builder
    box = PO.team_game_box(poss_by_season[2024])
    box = box[box["team_id"] == 10]
    prior = box[box["game_id"].isin([1, 2])]
    expected_3pa = 100.0 * prior["fga_3"].sum() / prior["poss"].sum()
    g3 = team[team["game_id"] == 3].iloc[0]
    lg_prior = PO.team_game_box(poss_by_season[2024])
    lg_prior = lg_prior[lg_prior["game_id"].isin([1, 2])]
    expected_lg = 100.0 * lg_prior["fga_3"].sum() / lg_prior["poss"].sum()
    assert g3["off_3pa_c"] == pytest.approx(expected_3pa - expected_lg, abs=1e-4)
    assert g3["n_prior_off"] == 2


def test_a_games_own_events_cannot_change_its_own_features():
    """The direct statement of the leak rule: perturb a game's OWN rows beyond
    recognition and its own feature row must not move."""
    poss_by_season, universe = _toy_possessions()
    before = PO.build_team_form(poss_by_season, universe, style_source="all_chances")

    corrupted = {2024: poss_by_season[2024].copy()}
    mask = (corrupted[2024]["game_id"] == 3) & (corrupted[2024]["offense_team_id"] == 10)
    corrupted[2024].loc[mask, "fga_3"] = 1
    corrupted[2024].loc[mask, "fga_rim"] = 0
    corrupted[2024].loc[mask, "fta"] = 99
    after = PO.build_team_form(corrupted, universe, style_source="all_chances")

    cols = [f"off_{r}_c" for r in PO.RATE_DEFS]
    b = before[(before["game_id"] == 3) & (before["team_id"] == 10)][cols].to_numpy()
    a = after[(after["game_id"] == 3) & (after["team_id"] == 10)][cols].to_numpy()
    np.testing.assert_array_equal(a, b)

    # ... while a LATER game of the same team must move, or the test above
    # would pass for a builder that ignores the data entirely.
    poss_by_season[2024] = pd.concat([poss_by_season[2024], poss_by_season[2024].assign(
        game_id=4, poss_index=lambda d: d["poss_index"] + 1000)], ignore_index=True)
    corrupted[2024] = pd.concat([corrupted[2024], corrupted[2024].assign(
        game_id=4, poss_index=lambda d: d["poss_index"] + 1000)], ignore_index=True)
    universe4 = pd.concat([universe, pd.DataFrame(
        {"game_id": [4], "game_date": pd.to_datetime(["2024-12-01"])})], ignore_index=True)
    b4 = PO.build_team_form(poss_by_season, universe4, style_source="all_chances")
    a4 = PO.build_team_form(corrupted, universe4, style_source="all_chances")
    b4v = b4[(b4["game_id"] == 4) & (b4["team_id"] == 10)][cols].to_numpy()
    a4v = a4[(a4["game_id"] == 4) & (a4["team_id"] == 10)][cols].to_numpy()
    assert not np.allclose(a4v, b4v)


def test_defence_allowed_form_is_what_opponents_did_to_this_team():
    poss_by_season, universe = _toy_possessions()
    form = PO.build_team_form(poss_by_season, universe, style_source="all_chances")
    box = PO.team_game_box(poss_by_season[2024])
    # team 300's only game is game 3, so team 10's defence-allowed form going
    # into game 3 comes from games 1-2 only, i.e. what teams 100 and 200 did.
    opp_prior = box[(box["opp_id"] == 10) & (box["game_id"].isin([1, 2]))]
    expected = 100.0 * opp_prior["fga_3"].sum() / opp_prior["poss"].sum()
    lg = box[box["game_id"].isin([1, 2])]
    expected -= 100.0 * lg["fga_3"].sum() / lg["poss"].sum()
    got = form[(form["game_id"] == 3) & (form["team_id"] == 10)]["def_3pa_c"].iloc[0]
    assert got == pytest.approx(expected, abs=1e-4)


# ===========================================================================
# 6. Seal
# ===========================================================================
def test_fold_slices_refuse_the_sealed_season():
    from cbb_sim.data.seal import SealedSeasonError

    design = pd.DataFrame({
        "season": [2022, 2026], "population": ["first", "first"], "y": [0, 1],
        "game_id": [1, 2],
    })
    folds = dict(PO.FOLDS)
    try:
        PO.FOLDS["FSEAL"] = {"train": [2022], "test": [2026]}
        with pytest.raises(SealedSeasonError):
            PO.fold_slices(design, "FSEAL", "first")
    finally:
        PO.FOLDS.clear()
        PO.FOLDS.update(folds)


# ===========================================================================
# 7. Determinism
# ===========================================================================
def _toy_design(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feats = PO.feature_set("A_team", "first")
    X = rng.normal(size=(n, len(feats))).astype("float32")
    logits = X[:, :6] @ rng.normal(size=(6, len(PO.CLASSES)))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(len(PO.CLASSES), p=row) for row in p], dtype="int8")
    d = pd.DataFrame(X, columns=feats)
    d["y"] = y
    d["season"] = 2024
    d["game_id"] = rng.integers(0, 200, n)
    d["population"] = "first"
    return d


@pytest.mark.parametrize("arm", ["ridge_logit", "cascade", "lgbm"])
def test_arm_fits_are_deterministic(arm):
    d = _toy_design()
    feats = PO.feature_set("A_team", "first")
    p1 = PO.predict_arm(arm, PO.fit_arm(arm, d, feats, seed=0), d, feats)
    p2 = PO.predict_arm(arm, PO.fit_arm(arm, d, feats, seed=0), d, feats)
    np.testing.assert_array_equal(p1, p2)


def test_probabilities_are_a_valid_simplex_in_the_declared_class_order():
    d = _toy_design()
    feats = PO.feature_set("A_team", "first")
    for arm in ("baseline", "ridge_logit", "cascade", "lgbm"):
        p = PO.predict_arm(arm, PO.fit_arm(arm, d, feats, seed=0), d, feats)
        assert p.shape == (len(d), len(PO.CLASSES))
        assert (p >= 0).all()
        # 1e-6 rather than machine epsilon: the cascade arm reaches a class
        # probability through up to three float multiplications, so its rows
        # sum to 1 only to within a few units in the seventh decimal. That is
        # far below any quantity the sim reads off it.
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_block_bootstrap_se_is_seeded_and_reproducible():
    d = _toy_design()
    feats = PO.feature_set("A_team", "first")
    p = PO.predict_arm("ridge_logit", PO.fit_arm("ridge_logit", d, feats, seed=0), d, feats)
    a = PO.block_bootstrap_se(d, p, n_rep=50, seed=99)
    b = PO.block_bootstrap_se(d, p, n_rep=50, seed=99)
    assert a == b and a > 0


# ===========================================================================
# 8. The rim-location override (added 2026-09-10)
# ===========================================================================
# `docs/tests/shot_classification_diag_2026-09-10.md` proved that ESPN's 2025
# feed mistags a batch of true tip-ins as `JumpShot`, and that shot LOCATION is
# the one signal the vendor got right. The override routes around it. What has
# to be tested is not that it fires -- that is one line -- but that its SCOPE
# holds: it must be able to move a two-point jumper toward the rim and must be
# unable to do anything else, because "anything else" would silently change
# points totals and the 2-vs-3 split that reconciles against hoopR's box score.


def _at_basket(dx_ft: float = 0.0, basket: int = 0) -> tuple[float, float]:
    """Shot-chart coordinates `dx_ft` feet from one of the two baskets, in the
    raw 0-940 x 0-500 tenths-of-a-foot grid the feed uses."""
    bx, by = EV.BASKET_XY[basket]
    return (bx + dx_ft * 10.0, by)


def test_a_near_rim_jumpshot_is_reclassified_as_a_rim_attempt():
    """THE synthetic case: a row the feed calls a two-point `JumpShot`, taken
    from under the basket. This is the 2025 tip-in mistag in miniature."""
    x, y = _at_basket(0.39)  # the canned tip-in placeholder pixel
    df = _plays([
        {"playType": "JumpShot", "shot_range": "jumper", "playText": "X made Jumper.",
         "shootingPlay": True, "shot_made": True, "scoreValue": 2,
         "shot_location_x": x, "shot_location_y": y},
    ])
    assert list(EV.classify_frame(df)) == ["FGA_rim"]
    # ... and with the override switched off it is what the feed said it was,
    # so the test is measuring the override rather than some other rule.
    assert list(EV.classify_frame(df, rim_override_max_ft=0.0)) == ["FGA_jump2"]


def test_the_override_reaches_both_baskets():
    rows = [{"playType": "JumpShot", "shot_range": "jumper", "shootingPlay": True,
             "shot_location_x": _at_basket(0.39, b)[0], "shot_location_y": _at_basket(0.39, b)[1]}
            for b in (0, 1)]
    assert list(EV.classify_frame(_plays(rows))) == ["FGA_rim", "FGA_rim"]


def test_the_override_stops_exactly_at_the_threshold():
    inside, outside = EV.RIM_OVERRIDE_MAX_FT - 0.01, EV.RIM_OVERRIDE_MAX_FT + 0.01
    df = _plays([
        {"playType": "JumpShot", "shot_range": "jumper", "shootingPlay": True,
         "shot_location_x": _at_basket(d)[0], "shot_location_y": _at_basket(d)[1]}
        for d in (inside, outside)
    ])
    assert list(EV.classify_frame(df)) == ["FGA_rim", "FGA_jump2"]


def test_the_override_cannot_touch_a_three_or_a_rim_tagged_row():
    """Scope, stated as a test. A three-point row at the rim is a contradiction
    in the feed, not an invitation to reclassify: turning it into `FGA_rim`
    would silently move 3 points to 2 and break the 3PA reconciliation against
    hoopR's box score. And a Dunk/LayUp/Tip row is already `FGA_rim`, so the
    override has nothing to do to it."""
    x, y = _at_basket(0.39)
    df = _plays([
        {"playType": "JumpShot", "shot_range": "three_pointer", "shootingPlay": True,
         "playText": "X missed Three Point Jumper.",
         "shot_location_x": x, "shot_location_y": y},
        {"playType": "LayUpShot", "shot_range": "rim", "shootingPlay": True,
         "shot_location_x": x, "shot_location_y": y},
        {"playType": "DunkShot", "shot_range": "rim", "shootingPlay": True,
         "shot_location_x": _at_basket(30.0)[0], "shot_location_y": _at_basket(30.0)[1]},
    ])
    assert list(EV.classify_frame(df)) == ["FGA_3", "FGA_rim", "FGA_rim"]


def test_an_unlocated_jumpshot_keeps_its_feed_label():
    """78-88% location coverage in 2022-2024 means most of the miss is here.
    A row with no coordinates must keep the feed's own label -- that is a
    missed repair, which is the safe direction, and never a guess."""
    df = _plays([
        {"playType": "JumpShot", "shot_range": "jumper", "shootingPlay": True},
        {"playType": "JumpShot", "shot_range": "jumper", "shootingPlay": True,
         "shot_location_x": float("nan"), "shot_location_y": float("nan")},
    ])
    assert list(EV.classify_frame(df)) == ["FGA_jump2", "FGA_jump2"]


def test_shot_distance_is_measured_to_the_nearer_basket_in_feet():
    df = _plays([
        {"shot_location_x": EV.BASKET_XY[0][0], "shot_location_y": EV.BASKET_XY[0][1]},
        {"shot_location_x": EV.BASKET_XY[1][0], "shot_location_y": EV.BASKET_XY[1][1] + 100.0},
        {"shot_location_x": 470.0, "shot_location_y": 250.0},  # mid-court
    ])
    d = EV.shot_distance_ft(df)
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(10.0)
    # mid-court is equidistant from both baskets: ~41.75 ft either way
    assert d[2] == pytest.approx((470.0 - EV.BASKET_XY[0][0]) / 10.0)


def test_the_loader_asks_for_the_columns_the_override_needs():
    """The override is only as good as the columns reaching it, and
    `classify_frame` cannot repair a row whose coordinates were never loaded.
    This is the one place that contract can be checked cheaply."""
    assert "shot_location_x" in EV.PLAY_COLUMNS
    assert "shot_location_y" in EV.PLAY_COLUMNS


def test_the_threshold_is_a_quantile_of_the_feeds_own_rim_rows():
    """The derivation helper must actually describe the rows the docstring says
    it does, and must survive a frame where a type is absent entirely."""
    df = _plays([
        {"playType": "DunkShot", "shot_range": "rim", "shootingPlay": True,
         "shot_location_x": _at_basket(d)[0], "shot_location_y": _at_basket(d)[1]}
        for d in (1.0, 2.0, 3.0, 4.0, 5.0)
    ] + [
        {"playType": "JumpShot", "shot_range": "jumper", "shootingPlay": True,
         "shot_location_x": _at_basket(12.0)[0], "shot_location_y": _at_basket(12.0)[1]},
    ])
    q = EV.rim_family_distance_quantiles(df)
    assert q["DunkShot"]["n_rows"] == 5
    assert q["DunkShot"]["quantiles_ft"]["p50"] == pytest.approx(3.0, abs=1e-6)
    assert q["TipShot"]["n_rows"] == 0 and q["TipShot"]["quantiles_ft"]["p50"] is None
    assert q["JumpShot(2)"]["quantiles_ft"]["p50"] == pytest.approx(12.0, abs=1e-6)


# ===========================================================================
# 9. First-chance-only style rates close the contamination channel
# ===========================================================================
def _toy_chances() -> tuple[dict, pd.DataFrame]:
    """Two teams, three games, with first chances and continuation chances
    given deliberately OPPOSITE shot mixes: every first chance is a three,
    every continuation chance is a rim attempt. A rate built over all chances
    must see the rim attempts; a first-chance-only rate must not."""
    rows = []
    for gid in (1, 2, 3):
        for team, opp in ((10, 20), (20, 10)):
            for k in range(20):
                rows.append({"season": 2024, "game_id": gid, "offense_team_id": team,
                             "defense_team_id": opp, "poss_index": k + 1, "chance_number": 1,
                             "terminal_event": "FGA_3", "fga_rim": 0, "fga_jump2": 0,
                             "fga_3": 1, "fta": 0, "points": 0})
            for k in range(10):
                rows.append({"season": 2024, "game_id": gid, "offense_team_id": team,
                             "defense_team_id": opp, "poss_index": k + 1, "chance_number": 2,
                             "terminal_event": "FGA_rim", "fga_rim": 1, "fga_jump2": 0,
                             "fga_3": 0, "fta": 0, "points": 2})
    universe = pd.DataFrame({"game_id": [1, 2, 3],
                             "game_date": pd.to_datetime(["2024-11-05", "2024-11-10",
                                                          "2024-11-20"])})
    ch = pd.DataFrame(rows)
    return {2024: ch}, universe


def test_first_chance_style_rates_ignore_continuation_chances():
    ch_by_season, universe = _toy_chances()
    box = PO.team_game_box_first_chance(ch_by_season[2024])
    one = box[(box["team_id"] == 10) & (box["game_id"] == 1)].iloc[0]
    # 20 first chances, all threes: 20 possessions, 20 FGA, 0 rim attempts.
    assert one["poss"] == 20
    assert one["fga"] == 20 and one["fga_3"] == 20 and one["fga_rim"] == 0


def test_a_continuation_chance_label_cannot_move_a_first_chance_feature():
    """THE contamination test, and the reason the source changed. Season 2025's
    continuation chances carried an upstream mislabel; under the round-1
    `all_chances` source that defect moved `off_rim_c`, a PREDICTOR of the
    `first` population whose targets the defect cannot reach
    (`docs/tests/shot_classification_diag_2026-09-10.md` section 6). Relabel
    every continuation chance and the first-chance-only feature must not
    move -- while the all-chances feature demonstrably does, or this test would
    pass for a builder that reads nothing."""
    ch_by_season, universe = _toy_chances()
    before = PO.build_team_form(ch_by_season, universe, style_source="first_chance")

    mangled = {2024: ch_by_season[2024].copy()}
    cont = mangled[2024]["chance_number"] > 1
    mangled[2024].loc[cont, "fga_rim"] = 0
    mangled[2024].loc[cont, "fga_jump2"] = 1
    mangled[2024].loc[cont, "terminal_event"] = "FGA_jump2"
    after = PO.build_team_form(mangled, universe, style_source="first_chance")

    cols = [f"off_{r}_c" for r in PO.RATE_DEFS]
    np.testing.assert_array_equal(after[cols].to_numpy(), before[cols].to_numpy())

    # the same relabel DOES move the all-chances rate, which is the channel
    poss_before = PO.team_game_box(ch_by_season[2024].assign(poss_index=range(len(ch_by_season[2024]))))
    poss_after = PO.team_game_box(mangled[2024].assign(poss_index=range(len(mangled[2024]))))
    assert poss_before["fga_rim"].sum() != poss_after["fga_rim"].sum()


def test_first_chance_box_refuses_a_table_without_per_chance_counts():
    """possessions v1 has no per-chance attempt columns. The builder must say
    so and name the fix, not silently fall back to the contaminated source."""
    ch_by_season, _ = _toy_chances()
    v1_like = ch_by_season[2024].drop(columns=["fga_rim", "fga_jump2", "fga_3"])
    with pytest.raises(KeyError, match="version v2"):
        PO.team_game_box_first_chance(v1_like)


def test_style_source_must_be_named():
    ch_by_season, universe = _toy_chances()
    with pytest.raises(KeyError):
        PO.build_team_form(ch_by_season, universe, style_source="whatever_is_convenient")


# ===========================================================================
# 10. Possessions table versioning
# ===========================================================================
def test_possessions_version_resolves_and_refuses_an_unknown_label():
    assert PS.possessions_dir("v1").name == "possessions"
    assert PS.possessions_dir("v2").name == "possessions_v2"
    assert PS.possessions_dir(None) == PS.POSSESSION_VERSIONS[PS.DEFAULT_POSSESSION_VERSION]
    assert PS.possessions_dir("v1", "some/other/dir").as_posix() == "some/other/dir"
    with pytest.raises(KeyError):
        PS.possessions_dir("v3")


def test_the_default_version_is_still_v1():
    """The PM switches this in one place when the round-2 tables are adopted.
    Until then a caller that names no version must keep reading v1, because
    other workers hold long-running reads on it."""
    assert PS.DEFAULT_POSSESSION_VERSION == "v1"


# ===========================================================================
# 11. Training schemes (round 2)
# ===========================================================================
def test_recency_weights_halve_at_the_half_life():
    dates = pd.Series(pd.to_datetime(["2024-11-01", "2024-08-03", "2024-05-05", "2025-01-01"]))
    ref = pd.Timestamp("2024-11-01")
    w = PO.recency_weights(dates, ref, half_life_days=90)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5)          # 90 days earlier
    assert w[2] == pytest.approx(0.25)         # 180 days earlier
    # a row that post-dates the reference is clipped to weight 1, never boosted
    assert w[3] == pytest.approx(1.0)


def test_month_boundaries_are_the_month_starts_that_contain_games():
    d = pd.Series(pd.to_datetime(["2024-11-08", "2024-11-30", "2025-01-02", "2025-03-19"]))
    assert PO.month_boundaries(d) == [pd.Timestamp("2024-11-01"), pd.Timestamp("2025-01-01"),
                                      pd.Timestamp("2025-03-01")]
    assert PO.month_boundaries(pd.Series([], dtype="datetime64[ns]")) == []


def _toy_scheme_design(n: int = 3000, seed: int = 11) -> pd.DataFrame:
    d = _toy_design(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    d["season"] = np.where(rng.random(n) < 0.6, 2024, 2025)
    start = np.where(d["season"] == 2024, np.datetime64("2023-11-06"), np.datetime64("2024-11-04"))
    d["game_date"] = pd.to_datetime(start) + pd.to_timedelta(rng.integers(0, 150, n), unit="D")
    return d


def test_s1_never_lets_a_game_into_its_own_fit():
    """The leak rule for the walk-forward scheme, checked on the schedule the
    scheme actually ran rather than on the idea of it."""
    d = _toy_scheme_design()
    feats = PO.feature_set("A_team", "first")
    tr = d[d["season"] == 2024]
    te = d[d["season"] == 2025]
    p, meta = PO.fit_predict_scheme("ridge_logit", "S1", tr, te, feats)
    assert p.shape == (len(te), len(PO.CLASSES))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert meta["n_fits"] == len(PO.month_boundaries(te["game_date"]))
    for seg in meta["segments"]:
        assert pd.Timestamp(seg["max_train_date"]) < pd.Timestamp(seg["refit_date"])
    # every test chance is scored, so S1 and S0 are compared on one test set
    assert sum(s["n_scored"] for s in meta["segments"]) == len(te)


def test_s1_first_segment_has_no_test_season_rows_and_later_ones_do():
    d = _toy_scheme_design()
    feats = PO.feature_set("A_team", "first")
    tr, te = d[d["season"] == 2024], d[d["season"] == 2025]
    _p, meta = PO.fit_predict_scheme("ridge_logit", "S1", tr, te, feats)
    assert meta["segments"][0]["n_train_from_test_season"] == 0
    assert meta["segments"][-1]["n_train_from_test_season"] > 0


def test_s2_weights_the_fit_and_s0_does_not():
    d = _toy_scheme_design()
    feats = PO.feature_set("A_team", "first")
    tr, te = d[d["season"] == 2024], d[d["season"] == 2025]
    p0, m0 = PO.fit_predict_scheme("ridge_logit", "S0", tr, te, feats)
    p2, m2 = PO.fit_predict_scheme("ridge_logit", "S2", tr, te, feats, half_life_days=90)
    assert m0["n_fits"] == 1 and m2["half_life_days"] == 90
    # a 90-day half-life over a 150-day training span must throw away real
    # sample, or the scheme is not doing anything
    assert m2["effective_sample_fraction"] < 0.95
    assert not np.allclose(p0, p2)


def test_s2_needs_a_half_life_and_unknown_schemes_raise():
    d = _toy_scheme_design()
    feats = PO.feature_set("A_team", "first")
    tr, te = d[d["season"] == 2024], d[d["season"] == 2025]
    with pytest.raises(ValueError):
        PO.fit_predict_scheme("ridge_logit", "S2", tr, te, feats)
    with pytest.raises(KeyError):
        PO.fit_predict_scheme("ridge_logit", "S9", tr, te, feats)


def test_the_baseline_refuses_sample_weights():
    d = _toy_scheme_design()
    with pytest.raises(ValueError):
        PO.fit_arm("baseline", d, [], sample_weight=np.ones(len(d)))


def test_the_default_style_source_still_reproduces_round_one():
    """The version default and the style-source default have to move together:
    `first_chance` needs per-chance attempt counts that only possessions v2
    writes. A caller that names neither must get the round-1 behaviour, because
    `scripts/train_possession_outcome_v1.py` is such a caller and the scripts
    convention forbids editing it."""
    assert PO.DEFAULT_STYLE_SOURCE == "all_chances"
    assert PS.DEFAULT_POSSESSION_VERSION == "v1"
