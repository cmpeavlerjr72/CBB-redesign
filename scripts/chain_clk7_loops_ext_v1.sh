#!/bin/sh
# clock round 7 local loop extension (lane H): A2 and L2 seeds 25-99 on the 500-game sample, 3 workers, sequential
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CBB_TRUTH=verified_v1
until grep -q "laneH_clk7_A2f1_s25 rc=" results/clock_r7/chain_loops.log; do sleep 20; done
for off in 25 50 75; do
  for spec in "v5b_r7A2_glat_pmean laneH_clk7_A2_o${off}_s25" "v5b_r6L2_glat_pmean laneH_clk7_L2_o${off}_s25"; do
    set -- $spec
    date
    [ -f results/engine_v0/$2/games.parquet ] || .venv/Scripts/python.exe scripts/run_clk6_closed_loop_sample_v1.py --clock $1 \
      --sample-file data/processed/truth/stride500_verified_v1_F2_2025.parquet --input-dir data/processed/models/engine_v3 \
      --seeds 25 --seed-offset $off --workers 3 --tag $2 --no-players > results/clock_r7/loop_$2.log 2>&1
    echo "$2 rc=$?"; date
  done
done
