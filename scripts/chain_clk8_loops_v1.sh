#!/bin/sh
# clock round 8 (experiments.md section 37): local closed-loop tap of M2D on the 500-game verified sample,
# 2 workers (lane H cap 3; one core kept for offline work). Control = v3full_COMB9GCTKD_s200_o0 (no run needed).
# usage: chain_clk8_loops_v1.sh <clock_mode> <tagstem> <offset>...
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MODE=$1; STEM=$2; shift 2
for o in "$@"; do
  date
  .venv/Scripts/python.exe scripts/run_clk6_closed_loop_sample_v1.py --clock $MODE \
    --sample-file data/processed/truth/stride500_verified_v1_F2_2025.parquet \
    --input-dir data/processed/models/engine_v3 --seeds 25 --seed-offset $o --workers ${LOOP_WORKERS:-2} --no-players \
    --tag ${STEM}_o${o}_s25 > results/clock_r8/loop_${STEM}_o${o}.log 2>&1
  echo "${STEM}_o${o} rc=$?"
done
date
