#!/usr/bin/env bash
# chain_clk6_r6_taps_v1.sh -- clock round 6 halftime-score taps (G7 first-half share veto), lane B 2026-09-30.
# Usage: bash scripts/chain_clk6_r6_taps_v1.sh <CLOCK> <SEED0> <NAME> [NSEEDS]
set -euo pipefail
CLOCK="$1"; S0="$2"; NAME="$3"; NS="${4:-10}"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="results/clock_r6/tap_${NAME}"
mkdir -p "$OUT"
.venv/Scripts/python.exe scripts/diag_g1g5_tap_v3.py "$CLOCK" data/processed/models/engine_v3 \
  data/processed/truth/stride500_verified_v1_F2_2025.parquet "$NS" "$OUT" "$S0"
