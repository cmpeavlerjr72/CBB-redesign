#!/bin/bash
# clock round 10 (experiments.md section 43): FULL-SIZE paired read for the candidate E1, fold 2 / 2025, 5,710 games x 200 seeds,
# local, sequential (one run at a time). Order: served-v3 control o0, E1 o0 (paired seeds 0-199), then the four control floor
# draws at offsets 1000-4000. Trajectories for the first 50 seeds of each run (the 5,710 x 50 definition of the D2 lines and of
# diag_margin_dispersion's baselines); games.parquet for all 200 seeds. No players file.
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1
W=${FULL_WORKERS:-16}
for spec in "ctrl 0" "E1 0" "ctrl 1000" "ctrl 2000" "ctrl 3000" "ctrl 4000"; do
  set -- $spec; LAB=$1; o=$2
  TAG=clk10full_${LAB}_o${o}_s200
  date
  if [ -f results/clock_env10/full/$TAG/games.parquet ]; then echo "$TAG exists, skipped"; continue; fi
  if [ "$LAB" = "E1" ]; then export ENGINE_CLOCK=v5b_r8K2_envE1_glat_pmean; else unset ENGINE_CLOCK; fi
  .venv/Scripts/python.exe scripts/run_engine.py --fold F2 --season 2025 --seeds 200 --seed-offset $o --workers $W \
    --input-dir data/processed/models/engine_v3 --tag $TAG --no-players --trajectory-seeds $((o+50)) \
    --results-dir results/clock_env10/full || exit 1
  unset ENGINE_CLOCK
  echo "$TAG done"
done
date
echo ALL_DONE
