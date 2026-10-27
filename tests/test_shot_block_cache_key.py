"""shot_block.load cache is keyed on content, not id(inp) (stale-table collision, 2026-10-27)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.engine import shot_block as SBK  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402

TAG_DIR = ROOT / "data/processed/models/engine"
pytestmark = pytest.mark.skipif(not (TAG_DIR / "shot_block_K2_Ocell_F2_2025.npz").exists(),
                                reason="backtest shot_block table not on disk")


def test_different_inputs_get_their_own_table_even_under_the_old_id_key(tmp_path, monkeypatch):
    import run_engine_live as RL
    import build_shot_block_lut_live_v1 as SBL
    monkeypatch.delenv("ENGINE_SHOT_BLOCK", raising=False)
    inp = EngineInputs.load(TAG_DIR, "F2_2025")
    SBK._CACHE.clear()
    full = SBK.load("K2_Ocell", inp)
    gids = inp.games["game_id"].to_numpy()[[0, 7, 1500, 3000, 4500, 5600]]
    s = RL.slice_inputs(inp, gids)
    s.meta["inputs_tag_loaded"] = ""            # live-style slate: table attached from tmp_path
    s.meta.pop("shot_block_lut", None)
    SBL.attach(s, tmp_path)
    # reproduce the old collision: the full table sits under the old-style key (arm, id(s))
    SBK._CACHE[("K2_Ocell", id(s))] = full
    SBK._CACHE[("K2_Ocell", id(inp))] = full
    got = SBK.load("K2_Ocell", s)
    assert got is not full
    assert got.team.shape[0] == len(s.games) == 6
    assert SBK.load("K2_Ocell", inp) is not got and SBK.load("K2_Ocell", inp).team.shape[0] == len(inp.games)
    assert SBK.load("K2_Ocell", s) is got           # same content -> cached
    SBK._CACHE.clear()
