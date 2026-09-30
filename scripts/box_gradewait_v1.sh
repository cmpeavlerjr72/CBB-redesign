#!/usr/bin/env bash
# Waits for each full-read result dir in turn, then grades it (verified + current). Operator 2026-09-30. ON the box, detached.
set -uo pipefail
cd "$(dirname "$0")/.."
for t in "$@"; do
  until [ -f "results/engine_v0/$t/games.parquet" ]; do sleep 15; done
  bash scripts/box_fullread_v1.sh grade "$t" > "logs/grade_$t.log" 2>&1
  echo "[graded] $t $(date -u +%H:%M:%SZ)"
done
