#!/usr/bin/env bash
# ops_full_retrain_identity_v1.sh -- lane D 2026-09-30: every full-retrain sibling run with the OLD (served) inputs, compared
# to the served artifact. Core cap: never more than 3 concurrent single-thread processes.
#   scripts/ops_full_retrain_identity_v1.sh [fg] [rb] [small]
# Outputs: data/processed/models/full_retrain_v1/identity/<family>_served/ (gitignored), logs/full_retrain/identity_*.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=python
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
I=data/processed/models/full_retrain_v1/identity
L=logs/full_retrain
mkdir -p "$I" "$L"
ts() { date "+%Y-%m-%d %H:%M:%S %z"; }
for step in "$@"; do
  echo "[$(ts)] start $step"
  case "$step" in
    fg)   # all 18 B1 fits at 3 workers (the served spec; the seed-1 floor is an offline grade, skipped)
      $PY scripts/train_fg_make_v4_par_v1.py --mode run --arms B1 --no-floor --no-leak --n-jobs 3 \
          --out-dir "$I/fg_served" > "$L/identity_fg.log" 2>&1
      $PY scripts/diag_full_retrain_identity_v1.py fg --new "$I/fg_served/B1" >> "$L/identity_fg.log" 2>&1 ;;
    rb)   # first 3 weekly cuts of the served S1_weekly spec (A0B0C0 = served lgbm / C_plus_state) at 3 workers
      $PY scripts/train_rebound_v3_par_artifacts_v1.py --stage 2 --folds F2 --arms A0B0C0 --n-jobs 3 --max-cuts 3 \
          --out-dir "$I/rb_served" > "$L/identity_rb.log" 2>&1
      $PY scripts/diag_full_retrain_identity_v1.py rb --new "$I/rb_served/artifacts/s2_F2_A0B0C0_seed0" >> "$L/identity_rb.log" 2>&1 ;;
    small)  # three single-core jobs at once: clock (whole chain), rotation (2 windows), the two ratings swaps
      $PY scripts/train_clock_chain_v1.py --root "$I/clock_served" --poss-version v1 --ratings-dir data/processed/ratings \
          --identity > "$L/identity_clock.log" 2>&1 &
      P1=$!
      $PY scripts/train_rotation_v3b_s1_poss_v1.py --poss-version v1 --out-dir "$I/rotation_served" --max-windows 2 --identity \
          -- --test-games 60 --seeds 1 > "$L/identity_rotation.log" 2>&1 &
      P2=$!
      ( $PY scripts/build_design_ratings_swap_v1.py --design data/processed/models/fg_make/design_v2_shotshooter.parquet \
            --ratings-dir data/processed/ratings --out "$I/swap/fg_identity.parquet" --identity
        $PY scripts/build_design_ratings_swap_v1.py --design data/processed/models/rebound/round3/design_round3.parquet \
            --ratings-dir data/processed/ratings --out "$I/swap/rb_identity.parquet" --identity ) > "$L/identity_swap.log" 2>&1 &
      P3=$!
      wait $P1 $P2 $P3 ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  echo "[$(ts)] end $step"
done
