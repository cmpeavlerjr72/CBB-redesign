"""run_laneI_overlay_served_v1.py -- run_engine_overlay_v1.py (unedited) with the sample runner's stale clock pin corrected
to the SERVED v2 clock (lane I, 2026-10-01).

`run_po4b_closed_loop.PINNED_SUBMODELS["ENGINE_CLOCK"]` still pins the pre-adoption `v5b_glat_pmean`, so its own
served-stack check refuses to run (correctly). This wrapper sets that one pin to what `adapters.py` now defaults to,
`v5b_r6L2_glat_pmean`, in the PARENT (the pins travel to the workers as the initializer's `flags` argument), and then
hands every argument to `run_engine_overlay_v1.main`. The served-stack check then passes with no drift. Proof that the
result is the served stack: the reference run's games rows equal the box adopted run's on every shared (game, seed).

    python scripts/run_laneI_overlay_served_v1.py --overrides F --runner sample -- <run_po4b_closed_loop_sample_v1 args>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_po4b_closed_loop as R  # noqa: E402

R.PINNED_SUBMODELS["ENGINE_CLOCK"] = "v5b_r6L2_glat_pmean"
import run_engine_overlay_v1 as O  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(O.main())
