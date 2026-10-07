#!/usr/bin/env python
"""run_parity_smoke_v1.py -- the served-default parity smoke as ONE command (seal-week runbook stage 1; the 2026-10-07
rehearsal found it was a manual step).

    run_engine.py --fold F2 --season 2025 --seeds 5 --max-games 60 --input-dir data/processed/models/engine_v3 (no ENGINE_*)
    digest_engine_run.py --compare <ref> --results results/engine_v0/<tag>

Exit 0 only on a bit-identical digest. Default reference: parity v10 (served stack v3, clock K2,
docs/tests/adoption_clock_K2_2026-10-07.md). Every ENGINE_* variable of the caller's environment is DROPPED, so the
smoke always reads the engine defaults (pass --keep-env to test an explicit stack, e.g. SERVED_V2 against v9).
Reads fold-2 2025 backtest inputs only (no sealed data). Writes results/engine_v0/<tag>/ (gitignored).

    .venv/Scripts/python.exe scripts/run_parity_smoke_v1.py
    .venv/Scripts/python.exe scripts/run_parity_smoke_v1.py --ref docs/ops/parity_reference_windows_v9.json --keep-env   # with SERVED_V2 set
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv/Scripts/python.exe") if (REPO / ".venv/Scripts/python.exe").exists() else sys.executable
REF = "docs/ops/parity_reference_windows_v10.json"
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--input-dir", default="data/processed/models/engine_v3")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default=None, help="default parity_smoke_<ref stem>_<yyyymmdd_HHMMSS>")
    ap.add_argument("--keep-env", action="store_true", help="keep the caller's ENGINE_* variables (default: drop them)")
    a = ap.parse_args(argv)
    if not (REPO / a.ref).exists():
        print(f"FAIL: reference {a.ref} missing")
        return 2
    tag = a.tag or f"parity_smoke_{Path(a.ref).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if not a.keep_env:
        for k in [k for k in env if k.startswith("ENGINE_")]:
            env.pop(k)
    for v in THREAD_VARS:
        env[v] = "1"
    run = [PY, "scripts/run_engine.py", "--fold", "F2", "--season", "2025", "--seeds", str(a.seeds),
           "--max-games", str(a.games), "--workers", str(a.workers), "--input-dir", a.input_dir, "--tag", tag]
    t = time.time()
    p = subprocess.run(run, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-1500:])
        print(f"FAIL: run_engine exit {p.returncode}")
        return p.returncode
    cmp_ = [PY, "scripts/digest_engine_run.py", "--compare", a.ref, "--results", f"results/engine_v0/{tag}"]
    q = subprocess.run(cmp_, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (q.stdout + q.stderr).strip()
    print("\n".join(out.splitlines()[-3:]))
    print(f"parity smoke {Path(a.ref).name}: {'PASS' if q.returncode == 0 else 'FAIL'} ({time.time() - t:.0f} s, "
          f"{a.games} games x {a.seeds} seeds, results/engine_v0/{tag})")
    return q.returncode


if __name__ == "__main__":
    raise SystemExit(main())
