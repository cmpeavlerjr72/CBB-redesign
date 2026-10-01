"""train_fg_make_v4_par_v2.py -- `train_fg_make_v4_par_v1.py` with `cbb_sim.team_rate_adapter_v2` (lane D, 2026-10-01).

Versioned sibling (v1 and the adapter are not edited). Under `--team-rate-table`, v1's adapter left
`off_make_raw` / `def_allow_raw` at the served values, so `fit_m` and `shooter_shrunk_dev_c` were trained on a stale
rate while the engine serves the E3-derived one (train/serve skew, corr 0.94-0.97). v2 installs the v2 adapter, which
re-derives the raw columns (the same fix as lane A's `train_fg_make_v4_par_rawfix_v1.py`). Every other line of
execution is v1's. Without a table nothing differs from v1 (the adapter is never called).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cbb_sim.team_rate_adapter as TRA  # noqa: E402
import cbb_sim.team_rate_adapter_v2 as TRA2  # noqa: E402

TRA.apply = TRA2.apply          # v1's trainer imports `apply` from this module at call time

import train_fg_make_v4_par_v1 as P  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(P.main())
