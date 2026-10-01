#!/usr/bin/env bash
# Lane I 2026-10-01 (second task): local 500 x 25 taps, direction only (Decision 12), three in parallel (one worker each).
#   laneI_UL2_s25   served v2 + ENGINE_USAGE_FT_LATE=UL2
#   laneI_SRA_s25   RBTO + A1 (inputs engine_v3_I_RBTOA1, rebound override, anchor)
#   laneI_SRAU_s25  RBTO + A1 + UL2
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
S=data/processed/truth/stride500_verified_v1_F2_2025.parquet
O=results/laneI_1001
AN=$PWD/data/processed/models/engine_v3_I_RBTOA1/anchor_offsets_F2_2025.npz
run() {  # tag overrides input_dir [env...]
  local tag=$1 ov=$2 in=$3; shift 3
  if [ -f results/engine_v0/$tag/games.parquet ]; then echo "SKIP $tag"; return; fi
  echo "$(date '+%F %T') START $tag"
  env "$@" $PY scripts/run_laneI_overlay_served_v1.py --overrides $ov --runner sample -- --sample-file $S --arm round2_s1 \
    --input-dir $in --seeds 25 --seed-offset 0 --workers 1 --tag $tag --results-dir results/engine_v0 > $O/$tag.log 2>&1
  echo "$(date '+%F %T') END $tag $(tail -1 $O/$tag.log)"
}
run laneI_UL2_s25 $O/ov_ref.json data/processed/models/engine_v3 ENGINE_USAGE_FT_LATE=UL2 &
run laneI_SRA_s25 $O/ov_rbtoa1.json data/processed/models/engine_v3_I_RBTOA1 ENGINE_SEASON_ANCHOR=$AN &
run laneI_SRAU_s25 $O/ov_rbtoa1.json data/processed/models/engine_v3_I_RBTOA1 ENGINE_SEASON_ANCHOR=$AN ENGINE_USAGE_FT_LATE=UL2 &
wait
echo "$(date '+%F %T') CHAIN DONE"
