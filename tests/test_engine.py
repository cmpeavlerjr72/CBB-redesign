"""
test_engine.py -- the six invariants deliverable 6 asks for.

Every test runs the REAL engine over a handful of real games, because the thing
being tested is the composition of the sub-models, not a mock of it. The games
are taken from `data/processed/models/engine/*_F2_2025.*`; if that prep has not
been run the module skips rather than inventing inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INPUT_DIR = ROOT / "data" / "processed" / "models" / "engine"
TAG = "F2_2025"

pytestmark = pytest.mark.skipif(
    not (INPUT_DIR / f"arrays_{TAG}.npz").exists(),
    reason="engine inputs not built; run scripts/build_engine_inputs.py --fold F2 --season 2025",
)


@pytest.fixture(scope="module")
def bundle():
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(INPUT_DIR, TAG)
    return inp, Adapters.load(inp, "F2")


def _run(bundle, n_games=6, n_seeds=2, seed0=0):
    from cbb_sim.engine import loop as L
    inp, ad = bundle
    gi = np.repeat(np.arange(n_games), n_seeds)
    sd = np.tile(np.arange(seed0, seed0 + n_seeds), n_games)
    return L.simulate_chunk(inp, ad, gi, sd)


# ---------------------------------------------------------------------------
# 1. determinism under a fixed seed
# ---------------------------------------------------------------------------
def test_determinism_under_a_fixed_seed(bundle):
    a = _run(bundle)
    b = _run(bundle)
    import pandas.testing as pdt
    pdt.assert_frame_equal(a.games, b.games)
    pdt.assert_frame_equal(a.players, b.players)


def test_a_games_draws_do_not_depend_on_the_batch_it_is_in(bundle):
    """The RNG rule's whole point: dropping games from the batch must not move
    any remaining game's result."""
    from cbb_sim.engine import loop as L
    inp, ad = bundle
    big = L.simulate_chunk(inp, ad, np.repeat(np.arange(6), 2), np.tile([0, 1], 6))
    small = L.simulate_chunk(inp, ad, np.repeat(np.array([1, 4]), 2), np.tile([0, 1], 2))
    key = ["game_id", "seed"]
    m = big.games.merge(small.games, on=key, suffixes=("_big", "_small"))
    assert len(m) == 4
    for col in ("home_pts", "away_pts", "n_periods"):
        assert (m[f"{col}_big"] == m[f"{col}_small"]).all(), col


# ---------------------------------------------------------------------------
# 2. five on the floor, always
# ---------------------------------------------------------------------------
def test_five_on_floor_invariant(bundle):
    from cbb_sim.engine import loop as L
    from cbb_sim.engine import rotation_adapter as RA
    from cbb_sim.engine import state as S
    from cbb_sim.engine.rng import StreamBook
    inp, ad = bundle
    n = 8
    gi = np.arange(n)
    sd = np.zeros(n, dtype=np.int64)
    gids = inp.games["game_id"].to_numpy()[gi]
    two = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    gg = np.concatenate([gi, gi])
    book = StreamBook(np.concatenate([sd, sd]), np.concatenate([gids, gids]) * 2 + two,
                      families=("rotation", "rotation_foul"))
    rows = np.arange(2 * n)
    rot = RA.init_batch(ad.rot_fit, inp.rot_share[gg, two], inp.rot_srank[gg, two],
                        inp.rot_fpm[gg, two], inp.rot_pavail[gg, two], book, rows)
    fouls = np.zeros((2 * n, inp.n_slots), dtype=np.int8)
    assert (rot.onmask.sum(axis=1) == 5).all()
    live = np.ones(2 * n, dtype=bool)
    for k in range(40):
        per = np.full(2 * n, 1 if k < 20 else 2, dtype=np.int16)
        sr = np.full(2 * n, 1200 - 30 * (k % 20), dtype=np.int16)
        five = RA.next_lineup(rot, per, sr, np.full(2 * n, k - 10), np.full(2 * n, 15.0),
                              fouls, book, rows, live)
        assert (rot.onmask.sum(axis=1) == 5).all(), f"step {k}"
        assert five.shape == (2 * n, 5)
        for r in range(2 * n):
            assert len(set(five[r].tolist())) == 5
    # and the same invariant holds through a real game
    res = L.simulate_chunk(inp, ad, np.arange(4), np.zeros(4, dtype=np.int64))
    assert res.games["n_periods"].min() >= 2
    assert S.FOUL_OUT == 5


# ---------------------------------------------------------------------------
# 3. score bookkeeping equals points from events
# ---------------------------------------------------------------------------
def test_score_equals_points_from_events(bundle):
    """Every point the engine puts on the board is credited to exactly one
    player, so the team score must equal the sum of its players' points."""
    from cbb_sim.engine import loop as L
    from cbb_sim.engine import state as S
    from cbb_sim.engine.adapters import Adapters  # noqa: F401
    inp, ad = bundle

    captured = {}
    real_final = L._finalise

    def spy(inp_, st, gids, seeds, keep):
        captured["st"] = st
        return real_final(inp_, st, gids, seeds, keep)

    L._finalise = spy
    try:
        L.simulate_chunk(inp, ad, np.repeat(np.arange(6), 2), np.tile([0, 1], 6))
    finally:
        L._finalise = real_final
    st: S.GameState = captured["st"]
    player_pts = st.player_box["pts"].sum(axis=2)
    assert (player_pts == st.pts).all(), "team score does not equal the sum of player points"
    # every attempt is also credited: team FGA == sum of player FGA
    team_fga = st.box["fga3"] + st.box["fga2_rim"] + st.box["fga2_jump"]
    assert (st.player_box["fga"].sum(axis=2) == team_fga).all()
    assert (st.player_box["fg3a"].sum(axis=2) == st.box["fga3"]).all()
    assert (st.player_box["fta"].sum(axis=2) == st.box["fta"]).all()
    # eFG% inputs (added 2026-09-10, G4): points == 2*(rim+jump2 makes) +
    # 3*(three makes) + FTM, and per-class FGM is also credited to a player.
    identity = (2 * (st.box["fgm2_rim"] + st.box["fgm2_jump"])
                + 3 * st.box["fgm3"] + st.box["ftm"])
    assert (st.pts == identity).all(), "points do not equal 2*(FGM2) + 3*(FGM3) + FTM"
    assert (st.player_box["fgm2_rim"].sum(axis=2) == st.box["fgm2_rim"]).all()
    assert (st.player_box["fgm2_jump"].sum(axis=2) == st.box["fgm2_jump"]).all()
    assert (st.player_box["fgm3"].sum(axis=2) == st.box["fgm3"]).all()
    # minutes: five players on the floor for every second of every period
    total_s = 2400.0 + 300.0 * st.n_ot.astype(float)
    mins = st.player_seconds.sum(axis=2)
    for side in (0, 1):
        assert np.allclose(mins[:, side], 5.0 * total_s, atol=1.0), side


# ---------------------------------------------------------------------------
# 4. period / overtime rule transitions
# ---------------------------------------------------------------------------
def test_period_and_overtime_rule_transitions(bundle):
    res = _run(bundle, n_games=30, n_seeds=6)
    g = res.games
    # no game ends tied, and no game ends before two periods
    assert (g["home_pts"] != g["away_pts"]).all(), "a game ended tied"
    assert (g["n_periods"] >= 2).all()
    # n_periods is the contract's convention: 2 + the number of overtimes
    assert g["n_periods"].max() <= 2 + 12


def test_overtime_is_five_minutes_and_team_fouls_carry_over(bundle):
    from cbb_sim.engine import state as S
    assert S.OT_SECONDS == 300
    assert S.HALF_SECONDS == 1200
    src = (Path(__file__).resolve().parents[1] / "src" / "cbb_sim" / "engine"
           / "loop.py").read_text(encoding="utf-8")
    # the halftime branch resets team fouls; the overtime branch must not
    assert "st.team_fouls[half] = 0" in src
    ot_block = src.split("if len(ot):", 1)[1].split("if progress", 1)[0]
    assert "team_fouls" not in ot_block, (
        "team fouls must CARRY OVER into overtime: an NCAA extra period is an "
        "extension of the second half")


# ---------------------------------------------------------------------------
# 5. draw independence across ordinals (the L19 shared-uniform bug)
# ---------------------------------------------------------------------------
def test_two_consecutive_draws_in_one_game_differ(bundle):
    """L19: the allocation uniform was once keyed on (seed, game_id) alone, so
    every event of a game drew the SAME uniform and the whole game's usage
    collapsed onto one player (top-1 share 60.9% vs 32.3% real). The engine's
    per-family ordinal counter is what prevents it; this pins it."""
    from cbb_sim.engine.rng import FAMILIES, StreamBook
    gids = np.array([401707981, 401714532, 401715435], dtype=np.int64)
    book = StreamBook(np.array([7, 7, 7]), gids)
    for fam in FAMILIES:
        u1 = book.draw(fam)
        u2 = book.draw(fam)
        u3 = book.draw(fam)
        assert (u1 != u2).all(), fam
        assert (u2 != u3).all(), fam
        assert (u1 != u3).all(), fam
    # and across families at the same ordinal
    b2 = StreamBook(np.array([7, 7, 7]), gids)
    a = b2.draw("clock")
    b = b2.draw("event")
    assert (a != b).all()


def test_uniforms_at_matches_control_rng(bundle):
    """The engine's only RNG addition is a per-row index; with a constant index
    it must reproduce `cbb_sim.control.rng.uniforms` bit for bit."""
    from cbb_sim.control import rng as crng
    from cbb_sim.engine.rng import uniforms_at
    gids = np.array([401707981, 401714532, 401715435, 99], dtype=np.int64)
    keys = crng.stream_keys(11, gids, "clock")
    for j in (0, 1, 7, 4096):
        assert np.array_equal(crng.uniforms(keys, j), uniforms_at(keys, np.full(4, j)))


def test_allocation_does_not_collapse_onto_one_player(bundle):
    """The downstream form of the same bug: a game's shots must be spread over
    the lineup, not all credited to one slot."""
    res = _run(bundle, n_games=12, n_seeds=4)
    p = res.players
    shooters = (p[p["fga"] > 0].groupby(["game_id", "seed", "team_id"]).size())
    assert shooters.mean() > 4.0, f"only {shooters.mean():.2f} distinct shooters per team-game"


# ---------------------------------------------------------------------------
# 6. the contract
# ---------------------------------------------------------------------------
def test_output_satisfies_the_results_contract(bundle):
    from cbb_sim.eval import contract as C
    res = _run(bundle, n_games=8, n_seeds=2)
    assert C.validate_games_frame(res.games, strict=False) == []
    assert C.validate_players_frame(res.players, strict=False) == []
    avail = C.box_columns_available(res.games)
    assert all(avail.values()), avail


# ---------------------------------------------------------------------------
# Dated artifacts (training scheme S1), which is the standing default for every
# sub-model -- `docs/models/possession_outcome/experiments.md` section 4, L21.
# ---------------------------------------------------------------------------
def _manifest_dir():
    return INPUT_DIR / "event_round2_s1_F2_2025"


@pytest.mark.skipif(not _manifest_dir().exists(),
                    reason="round-2 S1 artifacts not built; run "
                           "scripts/build_engine_event_round2.py --fold F2 --season 2025")
def test_a_game_never_gets_an_artifact_refit_at_or_after_its_own_month():
    """The property S1 exists to give, asserted per GAME rather than per month.

    A game played in month M must be scored by an artifact whose refit date is
    strictly BEFORE M's own games -- and, the check that actually binds, whose
    last TRAINING date is strictly before that game's date. A schedule that is
    off by one month is still a schedule; only the training window makes it
    honest (`CLAUDE.md`: created_at < tipoff, enforced in code)."""
    import json

    import pandas as pd

    from cbb_sim.engine.inputs import EngineInputs
    from cbb_sim.engine.manifest import ArtifactManifest

    inp = EngineInputs.load(INPUT_DIR, TAG)
    d = _manifest_dir()
    idx = json.loads((d / "index.json").read_text(encoding="utf-8"))
    gdate = pd.to_datetime(inp.games["game_date"])

    for pop in ("first", "cont"):
        segs = idx["populations"][pop]["segments"]
        man = ArtifactManifest.from_obj(
            {"model": "possession_outcome", "scheme": "S1", "key": pop,
             "artifacts": [{"refit_date": s["refit_date"], "path": s["file"],
                            "max_train_date": s["max_train_date"]} for s in segs]},
            d, inp.games)
        assert not man.is_static
        assert man.flag_name == f"scheme_static_possession_outcome_{pop}"

        chosen = man.seg_of_game
        refit = pd.to_datetime([e.refit_date for e in man.entries])
        maxtr = pd.to_datetime([e.max_train_date for e in man.entries])

        # 1. the refit that scores a game never post-dates its tipoff
        assert (refit[chosen].to_numpy() <= gdate.to_numpy()).all(), pop
        # 2. and never saw a game on or after that game's own date
        assert (maxtr[chosen].to_numpy() < gdate.to_numpy()).all(), pop
        # 3. month-level form of the same claim: no game in month M is scored
        #    by an artifact refit in M+1 or later
        gm = gdate.dt.to_period("M").to_numpy()
        rm = pd.Series(refit[chosen]).dt.to_period("M").to_numpy()
        assert (rm <= gm).all(), pop
        # 4. and the schedule is actually used -- a manifest that collapsed to
        #    one artifact would pass 1-3 while quietly being S0
        assert len(np.unique(chosen)) == len(man.entries), pop


@pytest.mark.skipif(not _manifest_dir().exists(), reason="round-2 S1 artifacts not built")
def test_a_static_manifest_is_a_manifest_of_length_one_and_says_so():
    from cbb_sim.engine.inputs import EngineInputs
    from cbb_sim.engine.manifest import ArtifactManifest

    inp = EngineInputs.load(INPUT_DIR, TAG)
    man = ArtifactManifest.static("clock", INPUT_DIR / f"rebound_{TAG.split('_')[0]}.joblib",
                                  inp.n_games, why="bake-off adopted one undated artifact")
    assert man.is_static
    assert man.flag_name == "scheme_static_clock"
    assert man.seg_of_game.shape == (inp.n_games,)
    assert (man.seg_of_game == 0).all()
    assert man.segments(np.array([0, 5, 9])).tolist() == [0, 0, 0]


@pytest.mark.skipif(not _manifest_dir().exists(), reason="round-2 S1 artifacts not built")
def test_a_manifest_whose_training_window_reaches_the_game_is_rejected():
    """The guard must FAIL on a leaky schedule, not merely pass on a clean one."""
    import json

    from cbb_sim.engine.inputs import EngineInputs
    from cbb_sim.engine.manifest import ArtifactManifest

    inp = EngineInputs.load(INPUT_DIR, TAG)
    d = _manifest_dir()
    idx = json.loads((d / "index.json").read_text(encoding="utf-8"))
    segs = idx["populations"]["first"]["segments"]
    leaky = [{"refit_date": s["refit_date"], "path": s["file"],
              "max_train_date": "2025-12-31"} for s in segs]      # trained past the season
    with pytest.raises(AssertionError, match="leak"):
        ArtifactManifest.from_obj(
            {"model": "possession_outcome", "artifacts": leaky}, d, inp.games)


# ---------------------------------------------------------------------------
# fg_make round 2 (docs/models/fg_make/experiments.md section 13)
# ---------------------------------------------------------------------------
def test_round2_state_definitions_agree_between_training_and_the_engine():
    """The engine's `gt_flag`/`eg_trail`/`eg_lead` must be the SAME function of
    (period, seconds_remaining, margin) that `fg_make.add_round2_state` applies
    to the training rows. They are computed in two different files, so a drift
    between them would be a silent train/serve skew of exactly the kind L23
    exists to catch."""
    import pandas as pd

    from cbb_sim.engine import loop as L
    from cbb_sim.engine import state as S
    from cbb_sim.engine.adapters import STATE_COLS, STATE_INDEX
    from cbb_sim.models import fg_make as FG

    rng = np.random.default_rng(0)
    n = 400
    period = rng.integers(1, 4, n).astype(np.int16)
    sec = rng.integers(0, 1200, n).astype(np.int32)
    sec = np.where(period >= 3, np.minimum(sec, 300), sec).astype(np.int32)
    margin = rng.integers(-30, 31, n).astype(np.int32)

    # engine side: the real state block, over a hand-built GameState
    st = S.new_state(np.zeros(n, np.int32), np.zeros(n, np.int32),
                     np.full(n, 2025), 15, {2025: (7, 10)}, np.zeros(n, np.int8))
    st.period[:] = period
    st.seconds_remaining[:] = sec
    x = L._state_block(st, np.arange(n), len(STATE_COLS),
                       margin.astype(np.float64), np.zeros(n))

    # training side: the model's own builder, on a frame with the SAME numbers.
    # Every row is a MISS, so `score_diff` == `score_diff_pre` and the two
    # sides are comparable without re-deriving the post-outcome correction.
    d = pd.DataFrame({"shot_class": "FGA_3", "made": False, "score_diff": margin,
                      "period": period, "seconds_remaining": sec})
    d = FG.add_round2_state(d)

    assert (d["score_diff_pre"].to_numpy() == margin).all()
    for col in ("score_diff_pre", "gt_flag", "eg_trail", "eg_lead"):
        assert np.array_equal(x[:, STATE_INDEX[col]], d[col].to_numpy(np.float64)), col
    # the indicators must actually fire on this sample, or the test proves nothing
    for col in ("gt_flag", "eg_trail", "eg_lead"):
        assert 0 < d[col].sum() < n, col


def test_engine_fg_make_flag_defaults_to_the_round1_winner_and_rejects_junk():
    """`ENGINE_FG_MAKE` is backward compatible: absent or `winner` is the old
    path, an unknown value raises rather than silently falling back."""
    import os

    from cbb_sim.engine.adapters import FgMakeAdapter
    from cbb_sim.engine.inputs import EngineInputs
    from cbb_sim.models import fg_make as FG

    inp = EngineInputs.load(INPUT_DIR, TAG)
    a = FgMakeAdapter.load(inp, "F2")
    assert a.mode == "winner"
    assert not a.provisional
    with pytest.raises(NotImplementedError, match="ENGINE_FG_MAKE"):
        FgMakeAdapter.load(inp, "F2", "decision8", "round2_NOT_AN_ARM")
    assert set(FG.R2_ARMS) == {"S_A", "S_B", "S_C", "S_D", "S_E"}
    assert os.environ.get("ENGINE_FG_MAKE") in (None, "winner")
