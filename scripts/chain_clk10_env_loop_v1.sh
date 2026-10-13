#!/bin/bash
# clock round 10 (experiments.md section 43): closed-loop runs on the 500-game verified sample, sequential, 12 workers.
# usage: chain_clk10_env_loop_v1.sh <S seeds> <label: ctrl|E1> <offset>...
#   ctrl -> no ENGINE_* flag (served v3, clock K2); E1 -> ENGINE_CLOCK=v5b_r8K2_envE1_glat_pmean only.
# After each run: diag_c4 extract (trajectory -> results/diag_c4/<tag>_*) and diag_margin_dispersion_v2 on the run's own extract.
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1
S=$1; LAB=$2; shift 2
for o in "$@"; do
  TAG=clk10_${LAB}_o${o}_s${S}
  date
  if [ "$LAB" = "E1" ]; then export ENGINE_CLOCK=v5b_r8K2_envE1_glat_pmean; else unset ENGINE_CLOCK; fi
  .venv/Scripts/python.exe scripts/run_engine_window_v2.py --ids-file data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet \
    --fold F2 --season 2025 --input-dir data/processed/models/engine_v3 --seeds $S --seed-offset $o --workers 12 \
    --games-per-block 25 --seeds-per-block 25 --tag $TAG --trajectory-seeds $((o+S)) --results-dir results/clock_env10/loop || exit 1
  unset ENGINE_CLOCK
  .venv/Scripts/python.exe scripts/diag_c4_extract_v1.py --tag $TAG --traj-file results/trajectories/$TAG/trajectory.parquet || exit 2
  .venv/Scripts/python.exe scripts/diag_margin_dispersion_v2.py --prefix results/diag_c4/${TAG}_ --seeds $S \
    --out results/clock_env10/dispersion/$TAG > results/clock_env10/logs/disp_$TAG.log 2>&1 || exit 3
  echo "$TAG done"
done
date
echo ALL_DONE
