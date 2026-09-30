"""run_po4b_closed_loop_sample_overlay_v1.py -- Lane N (2026-09-30): the sample-file closed-loop runner WITH a tag's serving
overlay served in-process (no Docker). Versioned sibling; `run_po4b_closed_loop.py`, `run_po4b_closed_loop_sample_v1.py`
and `run_engine_live.py` are imported, not edited.

    .venv/Scripts/python.exe scripts/run_po4b_closed_loop_sample_overlay_v1.py \
        --overlay-dir data/processed/models/engine_v3_N_C/overlay \
        --sample-file data/processed/truth/stride500_verified_v1_F2_2025.parquet \
        --arm round2_s1 --input-dir data/processed/models/engine_v3_N_C --seeds 25 --seed-offset 0 --workers 4 --tag <tag>

WHY. The served event adapter reads its round-2 team block from the FIXED path
`data/processed/models/engine/event_round2_s1_F2_2025/team_block.npz`, not from `--input-dir` (docs/ops/engine_inputs_v3_tag_path
section 1, the TRAP). On the box the tag's overlay is bind-mounted over that path; locally this wrapper instead builds a
scratch adapter dir from the overlay (`run_engine_live.prepare_from_overlay`) and rebinds `adapters.ENGINE_DIR` (and FG_DIR /
RB_S1_MANIFEST when the overlay has them) in the parent AND in every worker (initializer wrapper; overrides passed by env).
It also asserts that the scratch event block equals the input dir's `event_block_F2_2025.npz`.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_po4b_closed_loop as R  # noqa: E402

_ORIG_INIT = R._init_worker
_ENV = "LANE_N_ADAPTER_OVERRIDES"


def _apply_overrides() -> None:
    ov = json.loads(os.environ.get(_ENV, "{}"))
    if ov:
        from cbb_sim.engine import adapters as AD
        for k, v in ov.items():
            setattr(AD, k, Path(v))


def _init_worker_overlay(*args):
    _apply_overrides()
    from cbb_sim.engine import adapters as AD
    print(f"[worker {os.getpid()}] ENGINE_DIR={AD.ENGINE_DIR}", flush=True)
    return _ORIG_INIT(*args)


def main() -> int:
    argv = sys.argv[1:]
    if "--overlay-dir" not in argv:
        raise SystemExit("--overlay-dir is required")
    i = argv.index("--overlay-dir"); overlay = Path(argv[i + 1]); del argv[i:i + 2]
    tag = argv[argv.index("--tag") + 1]
    input_dir = Path(argv[argv.index("--input-dir") + 1])
    from run_engine_live import prepare_from_overlay
    scratch = ROOT / "results/engine_v0/_adapter_dirs" / f"{tag}_overlay"
    over = prepare_from_overlay(overlay, "F2", 2025, scratch)
    eb_ov = np.load(scratch / "event_round2_s1_F2_2025/team_block.npz")["team_block"]
    eb_in = np.load(input_dir / "event_block_F2_2025.npz")["team_block"]
    if not np.array_equal(eb_ov, eb_in):
        raise SystemExit("overlay event block != input dir event block")
    os.environ[_ENV] = json.dumps({k: str(Path(v).resolve()) for k, v in over.items()})
    _apply_overrides()
    R._init_worker = _init_worker_overlay
    sys.argv = [sys.argv[0], *argv]
    import run_po4b_closed_loop_sample_v1 as S
    rc = S.main()
    rd = Path(argv[argv.index("--results-dir") + 1]) if "--results-dir" in argv else R.DEFAULT_RESULTS
    mp = rd / tag / "run_meta.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        m["overlay_dir"] = str(overlay); m["adapter_overrides"] = json.loads(os.environ[_ENV])
        m["input_dir"] = str(input_dir)
        mp.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
