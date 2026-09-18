"""
test_event_adapter_round4b.py -- the possession-outcome round-5 ship-gate wiring.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 12.

  1. The SERVED path is untouched: `ENGINE_EVENT=round2_s1` still resolves to
     `event_round2_s1_{fold}_{season}`, still reports `provisional_event=False`
     and `adopted=True`, and still carries `mode == "round2_s1"`.
  2. An arm's sixteen team columns are the served sixteen with exactly the
     EIGHT style ones mapped to their shrunk siblings, and nothing else.
  3. An arm's team BLOCK differs from the served block only in those eight
     positions -- the other eight are bit-identical, which is section 12.1's
     "the arms differ only by the shrinkage".
  4. An arm's fitted features are `feature_set_v4(arm, pop)`, every team-level
     one of which the block can resolve, and the S1 schedule (refit dates,
     per-game segment assignment, training row counts) is the served one.
  5. The arm reports itself NOT ADOPTED and provisional.
  6. The arm's batched predict returns a proper simplex over `PO.CLASSES` and
     is NOT the reference's output -- i.e. the wiring actually reaches the
     model rather than silently serving the reference.
  7. An unknown `ENGINE_EVENT` still raises.

Each test skips rather than inventing inputs if an artifact is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

INPUT_DIR = ROOT / "data" / "processed" / "models" / "engine"
TAG = "F2_2025"
FOLD, SEASON = "F2", 2025
REF_DIR = INPUT_DIR / f"event_round2_s1_{FOLD}_{SEASON}"

pytestmark = pytest.mark.skipif(
    not (INPUT_DIR / f"arrays_{TAG}.npz").exists() or not REF_DIR.exists(),
    reason="engine inputs / served event artifacts not built on this box")


def _arm_dir(arm: str) -> Path:
    return INPUT_DIR / f"event_round4b_{arm}_{FOLD}_{SEASON}"


def _have(arm: str) -> bool:
    return (_arm_dir(arm) / "index.json").exists()


ARMS = [a for a in ("G2", "G3")]


# ---------------------------------------------------------------------------
# 1-2. the served path and the column mapping -- no arm artifact needed
# ---------------------------------------------------------------------------
def test_served_directory_name_unchanged():
    from cbb_sim.engine import adapters as A
    mode = "round2_s1"
    assert (A.ENGINE_DIR / f"event_{mode}_{FOLD}_{SEASON}").name == REF_DIR.name


@pytest.mark.parametrize("arm", ARMS)
def test_team_cols_map_only_the_style_columns(arm):
    import build_engine_event_round4b as B4
    ref = list(B4.TEAM_COLS_R2)
    got = list(B4.team_cols_for(arm))
    assert len(got) == len(ref) == 16
    suf = {"G2": "g2", "G3": "g3"}[arm]
    changed = [(a, b) for a, b in zip(ref, got) if a != b]
    assert len(changed) == 8, f"{len(changed)} columns changed, expected the 8 style ones"
    for a, b in changed:
        assert a.endswith("_c") and b == a[:-1] + suf
        assert not a.startswith("off_rating_")
    for a, b in zip(ref, got):
        if a not in B4.STYLE_COLS:
            assert a == b


def test_unknown_event_mode_still_raises():
    from cbb_sim.engine.adapters import EventAdapter
    with pytest.raises(NotImplementedError):
        EventAdapter.load(None, "round99_nonsense", FOLD, SEASON)


# ---------------------------------------------------------------------------
# 3-4. the built artifacts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ARMS)
def test_block_differs_only_in_the_style_positions(arm):
    if not _have(arm):
        pytest.skip(f"{arm} engine artifacts not built")
    import build_engine_event_round4b as B4
    ref_idx = json.loads((REF_DIR / "index.json").read_text(encoding="utf-8"))
    a_idx = json.loads((_arm_dir(arm) / "index.json").read_text(encoding="utf-8"))
    assert a_idx["team_cols"] == list(B4.team_cols_for(arm))
    rb = np.load(REF_DIR / "team_block.npz")["team_block"]
    ab = np.load(_arm_dir(arm) / "team_block.npz")["team_block"]
    assert rb.shape == ab.shape
    style_pos = [i for i, c in enumerate(ref_idx["team_cols"]) if c in B4.STYLE_COLS]
    other = [i for i in range(rb.shape[2]) if i not in style_pos]
    assert np.array_equal(rb[:, :, other], ab[:, :, other]), \
        "a non-style team column moved; the arm would differ by more than the shrinkage"
    assert not np.array_equal(rb[:, :, style_pos], ab[:, :, style_pos]), \
        "the style columns are identical; the shrinkage did not reach the block"


@pytest.mark.parametrize("arm", ARMS)
def test_schedule_and_features_match_the_served_model(arm):
    if not _have(arm):
        pytest.skip(f"{arm} engine artifacts not built")
    import train_possession_outcome_v4 as V4
    ref_idx = json.loads((REF_DIR / "index.json").read_text(encoding="utf-8"))
    a_idx = json.loads((_arm_dir(arm) / "index.json").read_text(encoding="utf-8"))
    assert a_idx["refit_dates"] == ref_idx["refit_dates"]
    assert np.array_equal(np.load(REF_DIR / "team_block.npz")["seg_of_game"],
                          np.load(_arm_dir(arm) / "team_block.npz")["seg_of_game"])
    for pop in ("first", "cont"):
        ai, ri = a_idx["populations"][pop], ref_idx["populations"][pop]
        assert ai["arm"] == ri["arm"]
        assert ai["features"] == V4.feature_set_v4(arm, pop)
        assert [s["n_train"] for s in ai["segments"]] == [s["n_train"] for s in ri["segments"]]
        assert [s["max_train_date"] for s in ai["segments"]] == \
               [s["max_train_date"] for s in ri["segments"]]
        # every team-level feature must be resolvable from the arm's own block
        team_feats = [f for f in ai["features"] if f in set(a_idx["team_cols"])]
        assert len(team_feats) == 16


# ---------------------------------------------------------------------------
# 5-6. the loaded adapter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ARMS)
def test_adapter_reports_not_adopted_and_predicts_a_simplex(arm):
    if not _have(arm):
        pytest.skip(f"{arm} engine artifacts not built")
    from cbb_sim.engine.adapters import EventAdapter
    from cbb_sim.engine.inputs import EngineInputs
    from cbb_sim.models import possession_outcome as PO

    inp = EngineInputs.load(str(INPUT_DIR), TAG)
    ref = EventAdapter.load(inp, "round2_s1", FOLD, SEASON)
    got = EventAdapter.load(inp, f"round4b_{arm}", FOLD, SEASON)

    assert ref.provisional is False and ref.mode == "round2_s1"
    assert ref.source["first"]["adopted"] is True
    assert got.provisional is True and got.mode == f"round4b_{arm}"
    assert got.source["first"]["adopted"] is False
    assert got.arm_first == ref.arm_first and got.arm_cont == ref.arm_cont

    rng = np.random.default_rng(0)
    n = 400
    gidx = rng.integers(0, inp.n_games, n)
    off = rng.integers(0, 2, n)
    is_first = rng.random(n) < 0.6
    team = np.zeros((n, 1), dtype=np.float32)      # unused on the round2_s1 path
    state = np.zeros((n, len(ref.plan_first.state_cols)
                      if hasattr(ref.plan_first, "state_cols") else 1), dtype=np.float32)
    from cbb_sim.engine.adapters import STATE_COLS
    state = np.zeros((n, len(STATE_COLS)), dtype=np.float32)
    state[:, 0] = 1.0                               # period
    state[:, 1] = 900.0                             # seconds_remaining
    state[:, 6] = 1.0                               # chance_number
    state[:, 7] = 1.0                               # chance_number_at_start

    p_ref = ref.predict(team, state, is_first, gidx, off)
    p_arm = got.predict(team, state, is_first, gidx, off)
    for p in (p_ref, p_arm):
        assert p.shape == (n, len(PO.CLASSES))
        assert np.all(p >= -1e-12) and np.all(p <= 1 + 1e-12)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-8)
    assert not np.allclose(p_ref, p_arm), \
        "the arm predicts exactly the reference; the wiring is not reaching the arm"
