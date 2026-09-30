#!/usr/bin/env python
"""
run_shot_block_closed_loop_v1.py -- one drawn-block-flag arm through the engine
(`docs/models/shot_block/experiments.md` section 5).

A thin wrapper over `run_po4b_closed_loop.py` (unchanged): the served stack is
pinned exactly as there (`ENGINE_EVENT=round2_s1`, the served default), the
same 500-game subset and seed range, and the only varying switch is
`ENGINE_SHOT_BLOCK`, set in this process's environment before the worker pool
is spawned (workers inherit it). The arm is recorded into `run_meta.json`.

    .venv/Scripts/python.exe scripts/run_shot_block_closed_loop_v1.py --shot-block K2_Ocell \
        --tag sb_K2O_s25 --seeds 25 --workers 4
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    argv = sys.argv[1:]
    i = argv.index("--shot-block")
    arm = argv[i + 1]
    del argv[i:i + 2]
    os.environ["ENGINE_SHOT_BLOCK"] = arm
    import run_po4b_closed_loop as RUN
    sys.argv = [sys.argv[0], "--arm", "round2_s1", *argv]
    rc = RUN.main()
    tag = argv[argv.index("--tag") + 1]
    rd = Path(argv[argv.index("--results-dir") + 1]) if "--results-dir" in argv else RUN.DEFAULT_RESULTS
    mp = rd / tag / "run_meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta["ENGINE_SHOT_BLOCK"] = arm
    meta["round"] = "shot_block section 5: drawn block flag closed loop"
    mp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
