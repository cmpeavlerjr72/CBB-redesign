"""Served stack v2 (adopted 2026-10-01, docs/tests/adoption_served_v2_2026-10-01.md).

Pins the five adopted defaults and checks that the SERVED_V1 values switch each
loop-level sub-model fully off (load -> None, so no stream is drawn). The
bit-identity of both stacks is proven by the parity references (v8 for the
default, v6/v7 for SERVED_V1), not here.
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
    assert '"ENGINE_CLOCK", "v5b_r6L2_glat_pmean"' in inspect.getsource(AD.Adapters.load)
    assert "v5b_r6L2_glat_pmean" in CK3.ADOPTED_MODES and "v5b_glat_pmean" in CK3.ADOPTED_MODES


def test_served_v1_switches_everything_off(monkeypatch):
    assert AD.SERVED_V1 == {
        "ENGINE_CLOCK": "v5b_glat_pmean", "ENGINE_SHOT_BLOCK": "reference",
        "ENGINE_FOUL_JOINT": "reference", "ENGINE_SHARED_SHOOTING": "reference",
        "ENGINE_CHANCE_TIME": "reference"}
    for k, v in AD.SERVED_V1.items():
        monkeypatch.setenv(k, v)
    assert SBK.load(AD.SERVED_V1["ENGINE_SHOT_BLOCK"], None) is None
    assert FJ.load(AD.SERVED_V1["ENGINE_FOUL_JOINT"]) is None
    assert SSL.load() is None
    assert CT.load([0], [1]) is None
