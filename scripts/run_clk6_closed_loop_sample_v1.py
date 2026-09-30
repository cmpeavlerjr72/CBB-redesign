"""run_clk6_closed_loop_sample_v1.py -- clock round 6 closed loop (experiments.md section 28).

Lane B, 2026-09-30.  Versioned wrapper over lane G's
`run_po4b_closed_loop_sample_v1.py` (itself a wrapper over
`run_po4b_closed_loop.py`; neither is edited).  That runner varies
ENGINE_EVENT and pins ENGINE_CLOCK to the served `v5b_glat_pmean`; this wrapper
sets the CLOCK pin from `--clock` (the pinned flags are handed to every worker
through the pool initialiser) and keeps ENGINE_EVENT at the served
`round2_s1`.  A non-served clock needs `--allow-drift`, which the runner
records in run_meta; it is added automatically and stamped here.

    .venv/Scripts/python.exe scripts/run_clk6_closed_loop_sample_v1.py --clock v5b_r6L2_glat_pmean \
        --sample-file data/processed/truth/stride500_verified_v1_F2_2025.parquet \
        --input-dir data/processed/models/engine_v3 --seeds 25 --seed-offset 0 --workers 6 --tag clk6_L2_s25
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_po4b_closed_loop_sample_v1 as W  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    i = argv.index("--clock"); clock = argv[i + 1]; del argv[i:i + 2]
    W.R.PINNED_SUBMODELS["ENGINE_CLOCK"] = clock
    if clock != "v5b_glat_pmean" and "--allow-drift" not in argv:
        argv.append("--allow-drift")
    if "--arm" not in argv:
        argv += ["--arm", "round2_s1"]
    sys.argv = [sys.argv[0], *argv]
    rc = W.main()
    tag = argv[argv.index("--tag") + 1]
    rd = Path(argv[argv.index("--results-dir") + 1]) if "--results-dir" in argv else W.R.DEFAULT_RESULTS
    mp = rd / tag / "run_meta.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        m["clock_round6_clock"] = clock
        m["env_CBB_TRUTH"] = os.environ.get("CBB_TRUTH")
        mp.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
