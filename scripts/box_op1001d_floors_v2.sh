#!/usr/bin/env bash
# operator: floor draws of the PLAIN DEFAULT served stack v2, fold 2, 5,710 games x 200 seeds, inputs engine_v3 (no overlay;
# the default reads the v3 event team block itself), NO ENGINE_* variable set. Chunks of 25 seeds; rerun = resume.
#   floors.sh <DIR=~/cbb> <WORKERS> <K...>   draw k -> tag v3full_S2f<k>_s200_o<k>000, seed offset k*1000
set -uo pipefail
cd "${1:?dir}"; W="${2:?workers}"; shift 2
export CBB_TRUTH=verified_v1
for v in $(env | grep -o '^ENGINE_[A-Za-z0-9_]*' || true); do echo "REFUSE: $v is set"; exit 2; done
ts() { date -u +%H:%M:%SZ; }
IN=data/processed/models/engine_v3
for k in "$@"; do
  OFF=$((k*1000)); RT="d1001D_S2f${k}_s200_o${OFF}"; done_s=0
  while [ $done_s -lt 200 ]; do
    o=$((OFF+done_s)); sub="${RT}_off${o}_n25"
    if [ -f results/engine_v0/$sub/run_meta.json ]; then echo "[skip] $sub"; done_s=$((done_s+25)); continue; fi
    echo "[chunk] $sub $(ts)"
    scripts/box_run_v3.sh scripts/run_engine.py --fold F2 --season 2025 --seeds 25 --seed-offset $o --workers $W \
      --games-per-block 60 --seeds-per-block 25 --tag $sub --results-dir results/engine_v0 --input-dir $IN || { echo "CHUNK FAILED $sub"; exit 4; }
    done_s=$((done_s+25))
  done
  scripts/box_run_v3.sh scripts/concat_engine_runs.py --tag $RT --results-dir results/engine_v0 --overwrite && echo "[done] $RT $(ts)"
  mkdir -p results/engine_v0/v3full_grade
  scripts/box_run_v3.sh scripts/eval_gates.py --results results/engine_v0/$RT --season 2025 --out results/engine_v0/v3full_grade/${RT}__verified.md && echo "[graded] $RT $(ts)"
done
