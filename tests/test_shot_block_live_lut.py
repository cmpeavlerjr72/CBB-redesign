"""Live per-slate shot_block table (served K2_Ocell, adopted 2026-10-01).

Reaches the lookup the daily-chain fixtures never did:
  1. `build_shot_block_lut_live_v1.build_table` on a slice of the backtest inputs reproduces
     the backtest table's rows for those games EXACTLY (train/serve identical).
  2. `attach` points `inp.meta` at the table and `shot_block.load` serves it for a slate whose
     tag has no backtest file (the live case that raised FileNotFoundError).
  3. A live-tagged slate with no table attached fails loudly, naming the missing build.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.engine import shot_block as SBK  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402

REF = ROOT / "data/processed/models/engine/shot_block_K2_Ocell_F2_2025.npz"
pytestmark = pytest.mark.skipif(not REF.exists(), reason="backtest shot_block table not on disk")


@pytest.fixture(scope="module")
def sliced():
    import run_engine_live as RL
    inp = EngineInputs.load(ROOT / "data/processed/models/engine", "F2_2025")
    gids = inp.games["game_id"].to_numpy()[[0, 7, 1500, 3000, 4500, 5600]]
    s = RL.slice_inputs(inp, gids)
    s.meta["inputs_tag_loaded"] = ""          # what a live build carries
    return s, gids


def test_live_table_equals_backtest_rows(sliced):
    import build_shot_block_lut_live_v1 as SBL
    s, gids = sliced
    tab = SBL.build_table(s, "K2_Ocell")
    ref = np.load(REF)
    rows = [int(np.flatnonzero(ref["game_id"] == g)[0]) for g in gids]
    for k in ("team", "shooter", "known", "anchor"):
        assert np.array_equal(tab[k], ref[k][rows]), k
    for k in ("coef", "mu", "sd"):
        assert np.array_equal(tab[k], ref[k]), k


def test_attach_serves_live_table_and_missing_fails_loudly(sliced, tmp_path, monkeypatch):
    import build_shot_block_lut_live_v1 as SBL
    s, _ = sliced
    monkeypatch.delenv("ENGINE_SHOT_BLOCK", raising=False)
    bare = type(s)(**{**s.__dict__, "meta": {k: v for k, v in s.meta.items() if k != "shot_block_lut"}}) \
        if hasattr(s, "__dict__") else s
    with pytest.raises(FileNotFoundError, match="build_shot_block_lut_live_v1"):
        SBK.ShotBlock("K2_Ocell", bare)
    path = SBL.attach(s, tmp_path)
    assert path is not None and path.exists()
    sb = SBK.load(SBK.DEFAULT, s)
    assert sb is not None and sb.team.shape == (len(s.games), 2, 2)
    monkeypatch.setenv("ENGINE_SHOT_BLOCK", "reference")
    assert SBL.attach(s, tmp_path / "off") is None
