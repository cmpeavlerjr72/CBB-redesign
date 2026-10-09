"""Served path aligned to the live-path harness (PM ruling 2026-10-09): shot-block shooter / known zero on seeded sides only."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import build_shot_block_lut_live_v1 as SBL  # noqa: E402
import build_engine_inputs_day1prior_v1 as DP  # noqa: E402


def _tab():
    return {"shooter": np.full((3, 2, 4), 0.3, dtype="float32"), "known": np.ones((3, 2, 4), dtype="uint8")}


def test_zero_only_seeded_sides():
    t = SBL.zero_seeded_sides(_tab(), {(0, 0), (2, 1)})
    for i, s in ((0, 0), (2, 1)):
        assert (t["known"][i, s] == 0).all() and (t["shooter"][i, s] == 0).all()
    assert t["known"].sum() == 6 * 4 - 2 * 4 + 0 and (t["known"][0, 1] == 1).all() and (t["shooter"][1] == np.float32(0.3)).all()


def test_no_seeded_sides_is_a_no_op():
    a, b = _tab(), SBL.zero_seeded_sides(_tab(), set())
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_attach_passes_seeded_sides(monkeypatch, tmp_path):
    inp = SimpleNamespace(roster_cbbd=np.zeros((3, 2, 4), dtype=np.int64), meta={})
    monkeypatch.setattr(SBL, "build_table", lambda *a, **k: {**_tab(), "x": np.zeros(1)})
    p = SBL.attach(inp, tmp_path, arm="K2_Ocell", seeded_sides={(1, 0)})
    z = np.load(p)
    assert (z["known"][1, 0] == 0).all() and (z["known"][0, 0] == 1).all()
    z0 = np.load(SBL.attach(inp, tmp_path / "b", arm="K2_Ocell"))
    assert (z0["known"] == 1).all()                      # default (replay / experiments) unchanged


def test_seed_fn_records_seeded_sides():
    fn = DP.make_seed_fn("A3", fallback=None)
    assert fn.seeded == set()
