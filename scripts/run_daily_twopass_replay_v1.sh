#!/usr/bin/env bash
# Two-pass replay (lane F 2026-10-01): sim + publish for evening and morning passes on four dates, 4 seeds, one core.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for d in 2024-11-04 2025-02-11 2025-02-25 2025-03-04; do
  for p in evening morning; do
    echo "=== $d $p $(date '+%F %T %z')"
    .venv/Scripts/python.exe scripts/chain_daily_v3.py --replay-season 2025 --slate-date $d --replay-pass $p --seeds 4 --root results/daily_replay_twopass 2>&1 | grep -vE "^\s*$" | tail -12
  done
done
echo "=== done $(date '+%F %T %z')"
