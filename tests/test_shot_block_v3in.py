"""ENGINE_SHOT_BLOCK=K2_Ocell_v3in (lane F, 2026-10-01): default unchanged; the sibling table equals the served one on unaffected games."""
from pathlib import Path

import numpy as np

from cbb_sim.engine import shot_block as SBK

M = Path(__file__).resolve().parents[1] / "data/processed/models"


def test_default_unchanged_and_arm_registered():
    assert SBK.DEFAULT == "K2_Ocell"
    assert SBK.ARMS["K2_Ocell_v3in"] == "shot_block_K2_Ocell_v3in"


def test_sibling_equals_served_table_on_unaffected_games_and_differs_on_affected():
    a = np.load(M / "engine/shot_block_K2_Ocell_F2_2025.npz")
    b = np.load(M / "engine/shot_block_K2_Ocell_v3in_F2_2025.npz")
    v2 = np.load(M / "engine/arrays_F2_2025.npz")["roster_cbbd"]
    v3 = np.load(M / "engine_v3/arrays_F2_2025.npz")["roster_cbbd"]
    aff = (v2 != v3).any(axis=(1, 2))
    assert np.array_equal(a["game_id"], b["game_id"])
    for k in ("team", "anchor", "shooter", "known", "coef", "mu", "sd"):
        assert np.array_equal(a[k][~aff], b[k][~aff]) if a[k].ndim and a[k].shape[0] == len(aff) else np.array_equal(a[k], b[k])
    assert (a["known"][aff] != b["known"][aff]).any()
