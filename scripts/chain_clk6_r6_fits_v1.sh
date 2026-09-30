#!/usr/bin/env bash
# chain_clk6_r6_fits_v1.sh -- clock round 6 (experiments.md section 28) fits, lane B 2026-09-30.
# Usage: bash scripts/chain_clk6_r6_fits_v1.sh <ARM>     (ARM = L2 or L2a)
# Runs the steps of scripts/exp_clk6_r6_arms_v1.py that are not done yet for ARM,
# in order, stopping at the first failure. Logs to results/clock_r6/<ARM>_<step>.log.
set -euo pipefail
ARM="$1"
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p results/clock_r6
ROOT_DIR="data/processed/models/clock/r6_${ARM}"
for STEP in censor design s1 latent; do
  case "$STEP" in
    censor) DONE="$ROOT_DIR/clock_censoring/censoring_v1_2025.parquet" ;;
    design) DONE="$ROOT_DIR/design_v2.parquet" ;;
    s1)     DONE="$ROOT_DIR/v3c_s1/manifest_srfloor_P3.json" ;;
    latent) DONE="$ROOT_DIR/v5b_bakeoff/v5b_bakeoff_report.json" ;;
  esac
  if [ -f "$DONE" ]; then
    echo "[$ARM] $STEP already done"
    continue
  fi
  echo "[$ARM] $STEP start $(date +%H:%M:%S)"
  "$PY" scripts/exp_clk6_r6_arms_v1.py --arm "$ARM" --step "$STEP" > "results/clock_r6/${ARM}_${STEP}.log" 2>&1
  echo "[$ARM] $STEP done $(date +%H:%M:%S)"
done
