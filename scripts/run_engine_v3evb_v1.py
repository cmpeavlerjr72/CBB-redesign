#!/usr/bin/env python
"""
run_engine_v3evb_v1.py -- run `scripts/run_engine.py` LOCALLY on v3 inputs with the
v3 round-2 event team block served, i.e. the box's `engine_v3_S0` overlay
without bind mounts (lane B, 2026-09-30).

    .venv/Scripts/python.exe scripts/run_engine_v3evb_v1.py <run_engine.py args...>
        --input-dir data/processed/models/engine_v3

The served event adapter reads its team block from the FIXED path
`data/processed/models/engine/event_round2_s1_F2_2025/team_block.npz` (the v2
block). On the box the S0 tag's overlay mounts `engine_v3/event_block_F2_2025.npz`
over it (`docs/ops/engine_inputs_v3_tag_path_2026-09-30.md` section 1; proof (a):
for S0 every other overlay file is identical to the served one). Here the same
substitution is done in-process: `Adapters.load` is wrapped so that, after the
served load, `ad.event.team_block` is replaced by `<input-dir>/event_block_F2_2025.npz`.
The wrap runs at module import, so the spawn workers (which re-import this main
module) carry it too. No served file is touched. Proved against the box rows of
`v3full_S0_s200_o0_off0_n25` (results doc, parity section).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.engine import adapters as _A  # noqa: E402

_orig = _A.Adapters.load.__func__ if hasattr(_A.Adapters.load, "__func__") else _A.Adapters.load


def _load(cls, inp, fold, season, *a, **k):
    ad = _orig(cls, inp, fold, season, *a, **k)
    src = getattr(inp, "input_dir", None) or _input_dir()
    p = Path(src) / f"event_block_{fold}_{season}.npz"
    blk = np.load(p)["team_block"]
    if blk.shape != ad.event.team_block.shape:
        raise ValueError(f"v3 event block {blk.shape} != served {ad.event.team_block.shape}")
    ad.event.team_block = blk
    return ad


def _input_dir() -> str:
    argv = sys.argv
    if "--input-dir" in argv:
        return argv[argv.index("--input-dir") + 1]
    raise SystemExit("run_engine_v3evb_v1.py needs --input-dir (the v3 inputs directory)")


_A.Adapters.load = classmethod(_load)

if __name__ == "__main__":
    import run_engine
    raise SystemExit(run_engine.main())
