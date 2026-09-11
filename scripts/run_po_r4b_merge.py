"""Round 4b merge/render: fold the two round-4 cells that were NOT RUN (`G3` on the tree
and the second-seed noise-floor cell) into round 4's ladder, WITHOUT touching a single
round-4 artifact.

This is round 4's own read-only `--render-only --merge` path (experiments.md 9.10) with
the output directory redirected to the versioned sibling
`data/processed/models/possession_outcome/round4b/`, per the worker-discipline rule that
a worker never overwrites a file another worker may be reading. The trainer and the
report renderer are imported UNMODIFIED and their module-level directory constants are
rebound before `main()` runs; no grading, decision, floor or feature code is changed.

The round-4 design cache is hard-linked into round4b, so the same bytes are read.

Usage:  .venv/Scripts/python.exe scripts/run_po_r4b_merge.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CBB_THREADS", "3")
_ROOT = Path(__file__).resolve().parents[1]
OUT4B = _ROOT / "data/processed/models/possession_outcome/round4b"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ckpts = [p.name for p in sorted(OUT4B.glob("ckpt_*.json"))
             if p.name != "ckpt_render_r4b.json"]
    print("merging checkpoints:", ckpts, flush=True)

    v4 = _load("po_v4", _ROOT / "scripts/train_possession_outcome_v4.py")
    v4.OUT_DIR = OUT4B
    sys.argv = ["train_possession_outcome_v4.py", "--render-only", "--stages", "0",
                "--ckpt", "ckpt_render_r4b.json", "--merge", ",".join(ckpts)]
    v4.main()

    rep = _load("po_r4_report", _ROOT / "scripts/diag_po_r4_report.py")
    rep.D = OUT4B
    rep.main()


if __name__ == "__main__":
    main()
