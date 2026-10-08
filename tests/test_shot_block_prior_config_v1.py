import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def test_prior_2025_loads_and_2027_stops_naming_file():
    import build_shot_block_lut_live_v1 as S
    p = S.load_prior(2025)
    assert set(p) == {"rim", "jump2", "three"}
    with pytest.raises(RuntimeError, match="shot_block_prior_2099_v1.parquet"):
        S.load_prior(2099)   # 2027 prior exists since the seal-week build; a missing season still hard-stops


def test_preseason_config_same_dir():
    from cbb_sim.live.preseason import preseason_dir
    assert preseason_dir().name == "2027_v2_20260930"
