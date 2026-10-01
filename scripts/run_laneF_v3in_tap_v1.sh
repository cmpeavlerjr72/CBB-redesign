#!/usr/bin/env bash
# lane F local tap: default (K2_Ocell) vs ENGINE_SHOT_BLOCK=K2_Ocell_v3in, 400 games x 5 seeds, 2 workers, v3 inputs.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for arm in default v3in; do
  if [ "$arm" = v3in ]; then export ENGINE_SHOT_BLOCK=K2_Ocell_v3in; else unset ENGINE_SHOT_BLOCK; fi
  echo "=== $arm $(date '+%F %T')"
  .venv/Scripts/python.exe -u scripts/run_engine.py --fold F2 --season 2025 --seeds 5 --max-games 400 --workers 2 \
     --games-per-block 100 --seeds-per-block 5 --tag laneF_v3in_tap_${arm}_400x5 --results-dir results/engine_v0 \
     --input-dir data/processed/models/engine_v3 --no-players 2>&1 | tail -3
done
echo "=== done $(date '+%F %T')"
