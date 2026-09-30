#!/usr/bin/env bash
# Local grading of the optional sample runs pulled from HF after the spot reclaim (operator 2026-09-30). One process at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
OUT=results/engine_v0/v3box_grade; mkdir -p $OUT
for t in v3box_S0_s200_o0 v3box_S0f1_s200_o1000 v3box_S0f2_s200_o2000 v3box_S1_s200_o0 v3box_S2_s200_o0 v3box_S3_s200_o0 v3box_K2O_s200_o0 v3box_R8b_s200_o0 v3box_R8bS_s200_o0; do
  CBB_TRUTH=verified_v1 $PY scripts/eval_gates.py --results results/engine_v0/$t --season 2025 --out $OUT/${t}__verified.md > $OUT/$t.log 2>&1 && echo "graded $t $(date +%T)"
done
for a in S0 S1 S2 S3; do
  $PY scripts/concat_engine_runs.py --tag v3full_${a}_s200_o0 --results-dir results/engine_v0 --out-tag v3part50_$a --allow-partial-players > $OUT/concat_$a.log 2>&1
  CBB_TRUTH=verified_v1 $PY scripts/eval_gates.py --results results/engine_v0/v3part50_$a --season 2025 --out $OUT/v3part50_${a}__verified.md >> $OUT/concat_$a.log 2>&1 && echo "graded part50 $a $(date +%T)"
done
