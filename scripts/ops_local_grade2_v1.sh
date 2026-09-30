#!/usr/bin/env bash
# Floors for the 50-seed partial full-size reads (operator 2026-09-30): concat each floor's first two chunks, grade.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe; OUT=results/engine_v0/v3box_grade
for i in 1 2 3 4; do
  $PY scripts/concat_engine_runs.py --tag v3full_S0f${i}_s200_o${i}000 --results-dir results/engine_v0 --out-tag v3part50_S0f$i --allow-partial-players > $OUT/concat_f$i.log 2>&1
  CBB_TRUTH=verified_v1 $PY scripts/eval_gates.py --results results/engine_v0/v3part50_S0f$i --season 2025 --out $OUT/v3part50_S0f${i}__verified.md >> $OUT/concat_f$i.log 2>&1 && echo "graded f$i"
done
