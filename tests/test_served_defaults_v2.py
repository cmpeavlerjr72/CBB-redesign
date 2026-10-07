"""Served stack v2 (adopted 2026-10-01, docs/tests/adoption_served_v2_2026-10-01.md).

Pins the five adopted defaults and checks that the SERVED_V1 values switch each
loop-level sub-model fully off (load -> None, so no stream is drawn). The
bit-identity of both stacks is proven by the parity references (v8 for the
default, v6/v7 for SERVED_V1), not here. Served stack v3 (2026-10-07) moves only the
clock default to K2; SERVED_V2 reproduces v9, the default is parity v10.
"""
from __future__ import annotations

import inspect

from cbb_sim.engine import adapters as AD
from cbb_sim.engine import chance_time as CT
from cbb_sim.engine import clock_adapter_v3 as CK3
from cbb_sim.engine import foul_joint as FJ
from cbb_sim.engine import shared_shooting as SSL
from cbb_sim.engine import shot_block as SBK


def test_adopted_defaults():
    assert SBK.DEFAULT == "K2_Ocell"
    assert FJ.DEFAULT == "R9ao3"
    assert SSL.DEFAULT == "G3"
    assert CT.DEFAULT == "KD"
    # served stack v3 (2026-10-07, docs/tests/adoption_clock_K2_2026-10-07.md): clock round 8 K2
    assert AD.CLOCK_DEFAULT == "v5b_r8K2_glat_pmean"
    assert '"ENGINE_CLOCK", CLOCK_DEFAULT' in inspect.getsource(AD.Adapters.load)
    assert {"v5b_glat_pmean", "v5b_r6L2_glat_pmean", "v5b_r8K2_glat_pmean"} <= CK3.ADOPTED_MODES


def test_served_v1_switches_everything_off(monkeypatch):
    assert AD.SERVED_V1 == {
        "ENGINE_CLOCK": "v5b_glat_pmean", "ENGINE_SHOT_BLOCK": "reference",
        "ENGINE_FOUL_JOINT": "reference", "ENGINE_SHARED_SHOOTING": "reference",
        "ENGINE_CHANCE_TIME": "reference", "ENGINE_EVENT_TEAM_BLOCK": "v1"}
    for k, v in AD.SERVED_V1.items():
        monkeypatch.setenv(k, v)
    assert SBK.load(AD.SERVED_V1["ENGINE_SHOT_BLOCK"], None) is None
    assert FJ.load(AD.SERVED_V1["ENGINE_FOUL_JOINT"]) is None
    assert SSL.load() is None
    assert CT.load([0], [1]) is None


def test_served_v2_is_the_old_default_set():
    """SERVED_V2 = served stack v2 exactly: the v3 adoption moved only ENGINE_CLOCK."""
    assert AD.SERVED_V2 == {
        "ENGINE_CLOCK": "v5b_r6L2_glat_pmean", "ENGINE_SHOT_BLOCK": "K2_Ocell",
        "ENGINE_FOUL_JOINT": "R9ao3", "ENGINE_SHARED_SHOOTING": "G3",
        "ENGINE_CHANCE_TIME": "KD", "ENGINE_EVENT_TEAM_BLOCK": "v3"}
    assert AD.SERVED_V2["ENGINE_SHOT_BLOCK"] == SBK.DEFAULT and AD.SERVED_V2["ENGINE_FOUL_JOINT"] == FJ.DEFAULT
    assert AD.SERVED_V2["ENGINE_SHARED_SHOOTING"] == SSL.DEFAULT and AD.SERVED_V2["ENGINE_CHANCE_TIME"] == CT.DEFAULT


def test_parity_references_v9_v10_flags():
    """v10 is the served-v3 reference (K2 clock), v9 the served-v2 one (SERVED_V2); both kept."""
    import json
    from pathlib import Path
    ops = Path(__file__).resolve().parents[1] / "docs" / "ops"
    v10 = json.loads((ops / "parity_reference_windows_v10.json").read_text(encoding="utf-8"))["digest"]["flags"]
    v9 = json.loads((ops / "parity_reference_windows_v9.json").read_text(encoding="utf-8"))["digest"]["flags"]
    assert v10["ENGINE_CLOCK"] == "v5b_r8K2_glat_pmean"
    assert v9["ENGINE_CLOCK"] == AD.SERVED_V2["ENGINE_CLOCK"]
    assert {k: v for k, v in v10.items() if k != "ENGINE_CLOCK"} == {k: v for k, v in v9.items() if k != "ENGINE_CLOCK"}
    for k, v in AD.SERVED_V2.items():
        assert v9[k] == v
