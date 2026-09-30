"""Trace which repo data files a script actually opens (lane J 2026-09-30, box-inputs inventory).

    .venv/Scripts/python.exe scripts/ops_trace_reads_v1.py --out trace.json -- \
        scripts/train_rebound_v3_par_v1.py --mode identity ...

Runs the target script in-process under `sys.addaudithook`, recording every `open` of a path
inside the repo's `data/` or `results/` trees (read modes only), then writes the sorted list with
sizes. Used to build the per-job input list in docs/ops/aws_launch_chain.md section 17
empirically instead of by reading code. Worker subprocesses are not traced (they receive their
data by pickle, not by file).
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEN: dict[str, int] = {}


def _hook(event: str, args) -> None:
    if event != "open":
        return
    path, mode = args[0], args[1]
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    if mode and any(c in str(mode) for c in "wax+"):
        return
    try:
        p = Path(os.fsdecode(path)).resolve()
        rel = p.relative_to(ROOT).as_posix()
    except Exception:                                                     # noqa: BLE001
        return
    if rel.startswith(("data/", "results/")) and p.is_file():
        SEEN[rel] = p.stat().st_size


def main() -> int:
    argv = sys.argv[1:]
    out = "trace.json"
    if argv[:1] == ["--out"]:
        out, argv = argv[1], argv[2:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    sys.addaudithook(_hook)
    sys.argv = argv
    sys.path.insert(0, str(Path(argv[0]).resolve().parent))
    rc = 0
    try:
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0) if isinstance(e.code, int) or e.code is None else 1
    Path(out).write_text(json.dumps(dict(sorted(SEEN.items())), indent=1), encoding="utf-8")
    print(f"traced {len(SEEN)} data/results files -> {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
