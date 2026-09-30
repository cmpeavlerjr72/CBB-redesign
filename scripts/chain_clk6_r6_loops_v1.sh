#!/usr/bin/env bash
# chain_clk6_r6_loops_v1.sh -- clock round 6 closed loops (experiments.md section 28), lane B 2026-09-30.
# Usage: bash scripts/chain_clk6_r6_loops_v1.sh <CLOCK> <SEED_OFFSET> <TAG> <WORKERS>
# One paired 500-game x 25-seed run on the honest inputs (engine_v3) and the
# verified same-rule sample. Skips if results/engine_v0/<TAG>/games.parquet exists.
set -euo pipefail
CLOCK="$1"; OFF="$2"; TAG="$3"; WK="$4"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CBB_TRUTH=verified_v1
if [ -f "results/engine_v0/${TAG}/games.parquet" ]; then echo "$TAG exists"; exit 0; fi
.venv/Scripts/python.exe scripts/run_clk6_closed_loop_sample_v1.py --clock "$CLOCK" \
  --sample-file data/processed/truth/stride500_verified_v1_F2_2025.parquet \
  --input-dir data/processed/models/engine_v3 --seeds 25 --seed-offset "$OFF" \
  --workers "$WK" --tag "$TAG" --no-players
