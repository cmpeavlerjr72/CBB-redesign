"""train_fg_make_v4_par_g2_v1.py -- fg_make fix-round arm G2 (docs/models/aggregation/experiments.md
section 2 / 2.1, lane A 2026-09-30): the Stage B `T` spec (B1 shooter block, E3 v4 team-rate table, S1
schedule) WITHOUT the offence make-rate feature `off_make_c`. Execution is `train_fg_make_v4_par_v1.py`
unchanged; this wrapper only removes `off_make_c` from the trainer's COMMON feature list before any
task is built (tasks are built in the parent, so workers receive the reduced design). Neither wrapped
trainer is edited.

    python scripts/train_fg_make_v4_par_g2_v1.py --mode run --arms B1 --seed 0 --no-floor --no-leak \
        --n-jobs 4 --team-rate-table data/processed/team_rate_features_E3_v4.parquet \
        --team-rate-missing raise --extra-cache <T design_v4_extra_E3.parquet> \
        --out-dir data/processed/models/fg_make/round_aggfix/G2_seed0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_fg_make_v4_shooter_block as R4M  # noqa: E402

R4M.COMMON[:] = [c for c in R4M.COMMON if c != "off_make_c"]

import train_fg_make_v4_par_v1 as P  # noqa: E402

if __name__ == "__main__":
    print("G2 wrapper: COMMON =", R4M.COMMON, flush=True)
    raise SystemExit(P.main())
