#!/bin/sh
# clock round 7 local closed loops (experiments.md section 33), lane H: A2 seeds 0-24 then A2 floor draw seeds 1000-1024, 3 workers (core cap 3)
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CBB_TRUTH=verified_v1
for spec in "0 laneH_clk7_A2_s25" "1000 laneH_clk7_A2f1_s25"; do
  set -- $spec
  date
  [ -f results/engine_v0/$2/games.parquet ] || .venv/Scripts/python.exe scripts/run_clk6_closed_loop_sample_v1.py --clock v5b_r7A2_glat_pmean \
    --sample-file data/processed/truth/stride500_verified_v1_F2_2025.parquet --input-dir data/processed/models/engine_v3 \
    --seeds 25 --seed-offset $1 --workers 3 --tag $2 --no-players > results/clock_r7/loop_$2.log 2>&1
  echo "$2 rc=$?"; date
done
