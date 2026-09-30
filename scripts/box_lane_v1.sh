#!/usr/bin/env bash
# Sequential lane of 500-game sample sims (operator 2026-09-30). Runs ON the box, detached.
#   scripts/box_lane_v1.sh <WAIT_FOR_FILE|-> <WORKERS> ARM...     (ARMs of box_v3_sims_v2.sh; waits for a file to exist first)
set -uo pipefail
cd "$(dirname "$0")/.."
W="$1"; WK="$2"; shift 2
if [ "$W" != "-" ]; then until [ -f "$W" ]; do sleep 20; done; fi
for arm in "$@"; do
  echo "[lane] $arm $(date -u +%H:%M:%SZ)"
  bash scripts/box_v3_sims_v2.sh sim "$arm" "$WK" > "logs/sample_${arm}.log" 2>&1 || echo "[lane] $arm FAILED rc=$?"
  echo "[lane] $arm end $(date -u +%H:%M:%SZ)"
done
