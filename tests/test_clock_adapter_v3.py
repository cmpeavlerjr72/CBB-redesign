"""
test_clock_adapter_v3.py -- the four properties the round-3c/4 clock adapter
must have, asserted against the REAL engine and the REAL fitted artifacts.

Pre-registration: `docs/models/clock/experiments.md` sections 12 and 14.

  1. A sampled duration is the duration the offence INTENDED, and the horn
     truncation happens in the engine (L20). The adapter never clips, and its
     own period-ending counter agrees exactly with the engine's.
  2. A frozen run NEVER reads the margin: two state blocks that differ only in
     `score_diff` produce a bit-identical predictive law under
     `ENGINE_CLOCK_FREEZE=1`, and a different one without it.
  3. Manifest selection never serves a game an artifact refit at or after its
     own month, and never one whose training window reaches the game.
  4. A mixed-month batch is routed per GAME to that game's own refit,
     bit-identically to pinning the run to that segment, and the engine runs a
     dated schedule without a pinned segment at all (round 4, 2026-09-11).

Each test skips rather than inventing inputs if the artifact it needs has not
been built.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INPUT_DIR = ROOT / "data" / "processed" / "models" / "engine"
CK_DIR = ROOT / "data" / "processed" / "models" / "clock"
TAG = "F2_2025"

pytestmark = pytest.mark.skipif(
    not (INPUT_DIR / f"arrays_{TAG}.npz").exists(),
    reason="engine inputs not built; run scripts/build_engine_inputs.py --fold F2 --season 2025",
)


def _modes_available() -> list[str]:
    from cbb_sim.engine.clock_adapter_v3 import V3C_MODES
    return [m for m, s in V3C_MODES.items() if (CK_DIR / s["manifest"]).exists()]


@pytest.fixture(scope="module")
def inp():
    from cbb_sim.engine.inputs import EngineInputs
    return EngineInputs.load(INPUT_DIR, TAG)


def _load(inp, mode: str, freeze: bool = False, segment: int | None = 0):
    """`segment=None` leaves `ENGINE_CLOCK_SEGMENT` UNSET, which is the
    per-game routing path an engine default runs on."""
    from cbb_sim.engine.clock_adapter_v3 import ClockAdapterV3
    old = {k: os.environ.get(k) for k in ("ENGINE_CLOCK_FREEZE", "ENGINE_CLOCK_SEGMENT")}
    os.environ["ENGINE_CLOCK_FREEZE"] = "1" if freeze else "0"
    if segment is None:
        os.environ.pop("ENGINE_CLOCK_SEGMENT", None)
    else:
        os.environ["ENGINE_CLOCK_SEGMENT"] = str(segment)
    try:
        return ClockAdapterV3.load(inp, mode, 2025)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _state(ad, n: int, seconds_remaining, score_diff, period=2.0, rng_seed=7):
    """A state block in the engine's own layout."""
    from cbb_sim.engine.adapters import STATE_COLS, STATE_INDEX
    rng = np.random.default_rng(rng_seed)
    x = np.zeros((n, len(STATE_COLS)), dtype=np.float64)
    sr = np.broadcast_to(np.asarray(seconds_remaining, dtype=np.float64), (n,)).copy()
    sd = np.broadcast_to(np.asarray(score_diff, dtype=np.float64), (n,)).copy()
    x[:, STATE_INDEX["period"]] = period
    x[:, STATE_INDEX["seconds_remaining"]] = sr
    x[:, STATE_INDEX["score_diff"]] = sd
    x[:, STATE_INDEX["score_diff_pre"]] = sd
    x[:, STATE_INDEX["is_ot"]] = float(period >= 3.0)
    x[:, STATE_INDEX["x_score_diff__seconds_remaining"]] = sd * sr / 1200.0
    x[:, STATE_INDEX["chance_number_at_start"]] = 1.0
    # one previous-end dummy per row, chosen at random, exactly as the engine
    # sets them (all-zero means `period_start`)
    from cbb_sim.models import clock as CK
    pick = rng.integers(0, len(CK.PREV_END_DUMMIES) + 1, n)
    for k, name in enumerate(CK.PREV_END_DUMMIES, start=1):
        x[:, STATE_INDEX[name]] = (pick == k).astype(np.float64)
    return x


# ---------------------------------------------------------------------------
# 1. the horn
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _modes_available(), reason="no round-3c clock artifacts built")
def test_a_draw_is_the_intended_duration_and_stays_on_the_grid(inp):
    """L20: the adapter returns what the offence INTENDED; `loop.py` truncates.

    With three seconds left a correctly-specified intended-duration model must
    still produce draws LONGER than three seconds -- that is the whole content
    of the censoring fix. An adapter that clipped would make this test fail."""
    from cbb_sim.models import clock as CK
    mode = _modes_available()[0]
    ad = _load(inp, mode)
    n = 4000
    team = inp.team_static[np.arange(n) % inp.n_games, 0].astype(np.float64)
    x = _state(ad, n, seconds_remaining=3.0, score_diff=0.0)
    u = np.random.default_rng(11).uniform(1e-9, 1 - 1e-9, n)
    dur = ad.draw(team, x, u)

    assert dur.min() >= 0 and dur.max() <= CK.DURATION_CAP
    assert np.array_equal(dur, dur.astype(np.int64))
    assert (dur > 3).mean() > 0.10, "the adapter is truncating; it must not (L20)"
    # what the engine then applies
    left = x[:, 1].astype(np.int64) * 0 + 3
    assert (np.minimum(dur, left) <= left).all()


@pytest.mark.skipif(not _modes_available(), reason="no round-3c clock artifacts built")
def test_the_adapters_period_ending_count_equals_the_engines_truncation_count(inp):
    """The adapter's end-of-half accumulator and `loop.py`'s own censoring
    counter count the SAME possessions. If they ever disagreed, the reported
    end-of-half statistic would be describing a different event than the one the
    engine simulates."""
    from cbb_sim.engine import loop as L
    from cbb_sim.engine.adapters import Adapters
    mode = _modes_available()[0]
    old = {k: os.environ.get(k) for k in
           ("ENGINE_CLOCK", "ENGINE_CLOCK_SEGMENT", "ENGINE_CLOCK_FREEZE")}
    os.environ.update({"ENGINE_CLOCK": mode, "ENGINE_CLOCK_SEGMENT": "0",
                       "ENGINE_CLOCK_FREEZE": "0"})
    try:
        ad = Adapters.load(inp, "F2", 2025)
        gi = np.repeat(np.arange(4), 2)
        sd = np.tile(np.arange(2), 4)
        res = L.simulate_chunk(inp, ad, gi, sd, keep_players=False)
        snap = ad.clock.eoh_snapshot()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert sum(snap["n_horn"]) == res.diag["possessions_censored_by_period_end"]
    assert sum(snap["n_poss"]) == res.n_possessions
    assert (res.games["possessions"] > 40).all() and (res.games["possessions"] < 120).all()


# ---------------------------------------------------------------------------
# 2. the freeze
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _modes_available(), reason="no round-3c clock artifacts built")
@pytest.mark.parametrize("mode", _modes_available())
def test_a_frozen_clock_never_reads_the_margin(inp, mode):
    """Two state blocks identical except for `score_diff`. Frozen, the
    predictive law must be BIT-identical; that is what "frozen at its pregame
    value" means for a model whose margin enters through several derived
    columns (the raw column, its clock interaction, the score-state leg of the
    round-2 cross, and P3's end-game indicators)."""
    from cbb_sim.engine.clock_adapter_v3 import V3C_MODES
    n = 3000
    team = inp.team_static[np.arange(n) % inp.n_games, 0].astype(np.float64)
    # inside the P3 end-game window, where the margin matters most
    x_lead = _state(None, n, seconds_remaining=40.0, score_diff=+12.0)
    x_trail = _state(None, n, seconds_remaining=40.0, score_diff=-12.0)

    frozen = _load(inp, mode, freeze=True)
    assert np.array_equal(frozen.pmf(team, x_lead), frozen.pmf(team, x_trail))

    live = _load(inp, mode, freeze=False)
    same = np.array_equal(live.pmf(team, x_lead), live.pmf(team, x_trail))
    if V3C_MODES[mode]["parametrisation"] == "P2":
        # P2 deletes every score column, so the freeze is a NO-OP by
        # construction -- the implementation's own consistency check.
        assert same
        assert np.array_equal(live.pmf(team, x_lead), frozen.pmf(team, x_lead))
    else:
        assert not same, f"{mode} claims to use the margin but ignores it"


# ---------------------------------------------------------------------------
# 3. the manifest
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _modes_available(), reason="no round-3c clock artifacts built")
@pytest.mark.parametrize("mode", _modes_available())
def test_a_game_never_gets_a_clock_artifact_refit_at_or_after_its_own_month(inp, mode):
    import pandas as pd
    from cbb_sim.engine.clock_adapter_v3 import manifest_for

    man = manifest_for(inp, mode)
    gdate = pd.to_datetime(inp.games["game_date"])
    chosen = man.seg_of_game
    refit = pd.to_datetime([e.refit_date for e in man.entries])
    maxtr = pd.to_datetime([e.max_train_date for e in man.entries])

    assert (refit[chosen].to_numpy() <= gdate.to_numpy()).all()
    assert (maxtr[chosen].to_numpy() < gdate.to_numpy()).all()
    gm = gdate.dt.to_period("M").to_numpy()
    rm = pd.Series(refit[chosen]).dt.to_period("M").to_numpy()
    assert (rm <= gm).all()
    # and the schedule is actually used -- a collapse to one artifact would
    # pass the three checks above while quietly being S0
    assert len(np.unique(chosen)) > 1
    assert not man.is_static


@pytest.mark.skipif(not _modes_available(), reason="no round-3c clock artifacts built")
def test_a_leaky_clock_manifest_is_rejected(inp):
    """The guard must FAIL on a leaky schedule, not merely pass on a clean one."""
    from cbb_sim.engine.clock_adapter_v3 import CK_DIR as _CKD
    from cbb_sim.engine.clock_adapter_v3 import V3C_MODES, _manifest_obj
    from cbb_sim.engine.manifest import ArtifactManifest

    mode = _modes_available()[0]
    obj = _manifest_obj(_CKD / V3C_MODES[mode]["manifest"])
    leaky = {**obj, "artifacts": [{**a, "max_train_date": "2025-12-31"}
                                  for a in obj["artifacts"]]}
    with pytest.raises(AssertionError, match="leak"):
        ArtifactManifest.from_obj(leaky, _CKD, inp.games)


@pytest.mark.skipif(not _modes_available(), reason="no round-3c clock artifacts built")
def test_a_schedule_refuses_to_predict_without_a_game_index(inp):
    """Round 4 replaced the old refusal-at-LOAD with per-game routing, so the
    refusal moved to the one place it is still needed: a multi-artifact
    schedule asked for a predictive law with no game index cannot know which
    month's fit any row belongs to, and must raise rather than silently serve
    one month's fit to a whole season. (Before round 4 `loop.py` passed no game
    index at all and this raised at load; the call site is now indexed.)"""
    mode = _modes_available()[0]
    ad = _load(inp, mode, segment=None)
    assert len(ad.arms) > 1
    team = inp.team_static[np.arange(64) % inp.n_games, 0].astype(np.float64)
    x = _state(None, 64, seconds_remaining=600.0, score_diff=0.0)
    with pytest.raises(ValueError, match="game index"):
        ad.pmf(team, x)


# ---------------------------------------------------------------------------
# 4. the hook is backward compatible
# ---------------------------------------------------------------------------
def test_every_pre_existing_engine_clock_value_still_loads_the_old_adapter(inp):
    from cbb_sim.engine.adapters import ClockAdapter, _load_clock
    for mode in ("reference", "reference_empirical"):
        assert isinstance(_load_clock(inp, mode, 2025), ClockAdapter)
    with pytest.raises(NotImplementedError):
        _load_clock(inp, "not_a_clock_arm", 2025)


def test_the_subset_rule_is_the_pre_registered_one(inp):
    """Section 12.3: sorted by game_id ascending, every 11th row, first 500."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_clk3c", ROOT / "scripts" / "run_clk3c_closed_loop.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rows = m.subset_rows(inp.games)
    assert len(rows) == 500
    gid = inp.games["game_id"].to_numpy()
    assert (np.diff(gid[rows]) > 0).all()
    assert len(np.unique(rows)) == 500
    # a stride, not a prefix: the subset must span the season, not November
    import pandas as pd
    d = pd.to_datetime(inp.games["game_date"].to_numpy()[rows])
    assert d.to_period("M").nunique() >= 4


# ---------------------------------------------------------------------------
# 4. per-game routing of the S1 schedule (round 4, 2026-09-11)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _modes_available(), reason="no round-3c clock artifacts built")
@pytest.mark.parametrize("mode", _modes_available())
def test_a_mixed_month_batch_gets_each_game_its_own_months_artifact(inp, mode):
    """One batch, games from several months, each row served by ITS game's
    refit.

    The reference for each row is the SAME adapter pinned to that game's
    segment with `ENGINE_CLOCK_SEGMENT` -- the path every round-3c result was
    produced on -- so per-game routing is asserted to be bit-identical to the
    run-partitioning it replaces. The last assertion keeps the test from being
    vacuous: two months' artifacts must actually disagree somewhere, or
    routing could not be distinguished from serving one fit to everything."""
    routed = _load(inp, mode, segment=None)
    man = routed.manifests["clock"]
    assert routed.segment is None
    assert len(routed.arms) == len(man.entries)

    segs = man.seg_of_game
    uniq = np.unique(segs)
    assert len(uniq) > 1, "slate covers one refit month; routing would be untested"
    gsel = np.array([int(np.flatnonzero(segs == k)[0]) for k in uniq], dtype=np.int64)

    gidx = np.tile(gsel, 50)                      # interleaved, not blocked
    team = inp.team_static[gidx, 0].astype(np.float64)
    x = _state(None, len(gidx), seconds_remaining=600.0, score_diff=0.0, period=2.0)
    got = routed.pmf(team, x, gidx)

    pmfs = {}
    for k, g in zip(uniq, gsel, strict=True):
        pinned = _load(inp, mode, segment=int(k))
        rows = np.flatnonzero(gidx == g)
        want = pinned.pmf(team[rows], x[rows])
        assert np.array_equal(got[rows], want), f"segment {k} routed to the wrong artifact"
        pmfs[int(k)] = pinned.pmf(team[:len(gsel)], x[:len(gsel)])
    assert any(not np.array_equal(pmfs[int(uniq[0])], pmfs[int(k)]) for k in uniq[1:]), \
        "every month's artifact gives the same law; this test cannot see routing"

    # and the draw path carries the index through
    u = np.random.default_rng(3).uniform(1e-9, 1 - 1e-9, len(gidx))
    dur = routed.draw(team, x, u, gidx)
    assert len(dur) == len(gidx) and dur.min() >= 0


@pytest.mark.skipif(not _modes_available(), reason="no round-3c clock artifacts built")
def test_the_engine_runs_a_dated_clock_schedule_without_a_pinned_segment(inp):
    """`loop.py`'s clock call site hands the adapter the active rows' game
    index, so a schedule serves a mixed-month batch inside ONE run. Before
    round 4 this raised."""
    from cbb_sim.engine import loop as L
    from cbb_sim.engine.adapters import Adapters
    mode = _modes_available()[0]
    old = {k: os.environ.get(k) for k in
           ("ENGINE_CLOCK", "ENGINE_CLOCK_SEGMENT", "ENGINE_CLOCK_FREEZE")}
    os.environ.update({"ENGINE_CLOCK": mode, "ENGINE_CLOCK_FREEZE": "0"})
    os.environ.pop("ENGINE_CLOCK_SEGMENT", None)
    try:
        ad = Adapters.load(inp, "F2", 2025)
        assert ad.clock.segment is None
        assert ad.clock.wants_game_index
        # games spread across the slate, so several months are in one chunk
        gi = np.linspace(0, inp.n_games - 1, 8).astype(np.int64)
        segs = ad.clock.manifests["clock"].seg_of_game[gi]
        assert len(np.unique(segs)) > 1, "chunk does not span two refit months"
        res = L.simulate_chunk(inp, ad, gi, np.zeros(len(gi), dtype=np.int64),
                               keep_players=False)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert (res.games["possessions"] > 40).all() and (res.games["possessions"] < 120).all()
