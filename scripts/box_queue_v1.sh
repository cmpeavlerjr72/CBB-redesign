#!/usr/bin/env bash
# Sequential queue of full reads (operator 2026-09-30). Runs ON the box, detached.
#   scripts/box_queue_v1.sh <WAIT_FOR_RESULT_DIR|-> <ARM:OFFSET:WORKERS[:SEEDS]>...
# Waits until results/engine_v0/<WAIT_FOR_RESULT_DIR>/games.parquet exists (or no wait with "-"), then runs each read in order.
set -uo pipefail
cd "$(dirname "$0")/.."
W="$1"; shift
if [ "$W" != "-" ]; then until [ -f "results/engine_v0/$W/games.parquet" ]; do sleep 20; done; fi
for spec in "$@"; do
  IFS=: read -r ARM OFF WK SEEDS <<< "$spec"
  echo "[queue] $ARM off=$OFF workers=$WK $(date -u +%H:%M:%SZ)"
  bash scripts/box_fullread_v1.sh run "$ARM" "$WK" "$OFF" "${SEEDS:-200}" 25 > "logs/fullread_${ARM}_o${OFF}.log" 2>&1
done
echo "[queue] done $(date -u +%H:%M:%SZ)"
