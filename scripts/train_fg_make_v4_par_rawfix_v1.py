"""train_fg_make_v4_par_rawfix_v1.py -- fg_make arm `Tfix` (docs/models/aggregation/experiments.md section 2.5,
lane A 2026-09-30): the Stage B `T` spec with the train/serve skew removed.

Defect found by addendum F/F2: with `--team-rate-table`, `cbb_sim.team_rate_adapter.apply` replaces
`off_make_c` / `def_allow_c` by the E3 values but leaves `off_make_raw` / `def_allow_raw` at the SERVED
expanding-mean values, and `R4M.fit_m` / `R4M.build_extra` (-> `shooter_shrunk_dev_c`) are computed from that
stale raw rate. The engine builder derives the shooter dev from the E3 rate (raw = c + league as-of), so the
served `shooter_shrunk_dev_c` differs from the trained one (corr 0.94-0.97). This wrapper re-derives the raw
columns as c + lg_make_asof right after the adapter (exactly what the `--feature-table` path already does);
execution is `train_fg_make_v4_par_v1.py` unchanged. Neither wrapped module is edited.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cbb_sim.team_rate_adapter as TRA  # noqa: E402

_orig_apply = TRA.apply


def _apply_rawfix(frame, table_path, submodel, *a, **k):
    out = _orig_apply(frame, table_path, submodel, *a, **k)
    if submodel == "fg_make" and "lg_make_asof" in out.columns:
        for c, raw in (("off_make_c", "off_make_raw"), ("def_allow_c", "def_allow_raw")):
            if c in out.columns and raw in out.columns:
                out[raw] = out[c].to_numpy(dtype="float64") + out["lg_make_asof"].to_numpy(dtype="float64")
        print("rawfix: off_make_raw / def_allow_raw re-derived as c + lg_make_asof", flush=True)
    return out


TRA.apply = _apply_rawfix

import train_fg_make_v4_par_v1 as P  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(P.main())
