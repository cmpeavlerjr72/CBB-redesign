"""
test_clock_adapter_v3.py -- the three properties round 3c's clock adapter must
have, asserted against the REAL engine and the REAL fitted artifacts.

Pre-registration: `docs/models/clock/experiments.md` section 12.

  1. A sampled duration is the duration the offence INTENDED, and the horn
     truncation happens in the engine (L20). The adapter never clips, and its
     own period-ending counter agrees exactly with the engine's.
  2. A frozen run NEVER reads the margin: two state blocks that differ only in
     `score_diff` produce a bit-identical predictive law under
     `ENGINE_CLOCK_FREEZE=1`, and a different one without it.
  3. Manifest selection never serves a game an artifact refit at or after its
     own month, and never one whose training window reaches the game.

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


def _load(inp, mode: str, freeze: bool = False, segment: int = 0):
    from cbb_sim.engine.clock_adapter_v3 import ClockAdapterV3
    old = {k: os.environ.get(k) for k in ("ENGINE_CLOCK_FREEZE", "ENGINE_CLOCK_SEGMENT")}
    os.environ["ENGINE_CLOCK_FREEZE"] = "1" if freeze else "0"
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
def test_a_schedule_refuses_to_load_without_an_explicit_segment(inp):
    """`loop.py` gives the clock no game index, so a dated schedule must be
    served one segment per run. Loading one without naming the segment would
    silently serve one month's fit to a whole season."""
    from cbb_sim.engine.clock_adapter_v3 import ClockAdapterV3
    mode = _modes_available()[0]
    old = os.environ.get("ENGINE_CLOCK_SEGMENT")
    os.environ.pop("ENGINE_CLOCK_SEGMENT", None)
    try:
        with pytest.raises(ValueError, match="segment"):
            ClockAdapterV3.load(inp, mode, 2025)
    finally:
        if old is not None:
            os.environ["ENGINE_CLOCK_SEGMENT"] = old


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
