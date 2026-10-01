#!/usr/bin/env bash
# Lane I 2026-10-01: local 500 x 25 closed-loop taps (direction and parity only; Decision 12) on the verified stride sample.
# rebound experiments.md 12.3 (RBTO), free_throw experiments.md 13.2 (N1) and 13.3 (A1). Reference = plain default.
# Skips a tag whose games.parquet exists (resume = rerun). One worker per run (lane core cap).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
S=data/processed/truth/stride500_verified_v1_F2_2025.parquet
O=results/laneI_1001
run() {  # tag overrides input_dir [env...]
  local tag=$1 ov=$2 in=$3; shift 3
  if [ -f results/engine_v0/$tag/games.parquet ]; then echo "SKIP $tag"; return; fi
  echo "$(date '+%F %T') START $tag"
  env "$@" $PY scripts/run_laneI_overlay_served_v1.py --overrides $ov --runner sample -- --sample-file $S --arm round2_s1 \
    --input-dir $in --seeds 25 --seed-offset 0 --workers 1 --tag $tag --results-dir results/engine_v0 > $O/$tag.log 2>&1
  echo "$(date '+%F %T') END $tag rc=$? $(tail -1 $O/$tag.log)"
}
run laneI_ref_s25 $O/ov_ref.json data/processed/models/engine_v3
run laneI_RBTO_s25 $O/ov_rbto.json data/processed/models/engine_v3_I_RBTO \
  ENGINE_SEASON_ANCHOR=$PWD/data/processed/models/engine_v3_I_RBTO/anchor_offsets_F2_2025.npz
run laneI_N1_s25 $O/ov_n1.json data/processed/models/engine_v3_I_N1
if [ ! -d data/processed/models/engine_v3_I_A1 ]; then $PY scripts/build_engine_inputs_v3_A1_v1.py > $O/build_A1.log 2>&1; fi
run laneI_A1_s25 $O/ov_ref.json data/processed/models/engine_v3_I_A1
echo "$(date '+%F %T') CHAIN DONE"
