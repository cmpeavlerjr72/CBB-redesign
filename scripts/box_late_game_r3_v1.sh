#!/usr/bin/env bash
# box_late_game_r3_v1.sh -- late-game ROUND 3 full-size reads (lane L, 2026-10-01; experiments.md sections 6-7).
#   tierA <DIR> <WORKERS> <ARM...>   500 verified games x 200 seeds, TAPPED (window lines + first-half identity):
#                                    R9 (offsets 0,1000,2000,3000,4000) and each ARM (clk_Dt | clk_Dtt | clk_D) at offset 0.
#   tierB <DIR> <WORKERS> <ARM>      5,710 x 200 untapped, served v2 + ENGINE_LATE_GAME=<ARM>, chunks of 25 seeds,
#                                    concat + eval_gates; pair with ops_pair_bootstrap_v1.py vs v3full_COMB9GCTKD_s200_o0
#                                    and the operator's served-v2 floors v3full_D2f{1..4}_s200_o{k}000.
# NO ENGINE_* variable may be set by the caller (the base is the plain served default); the arm is an argument.
set -uo pipefail
MODE="${1:?tierA|tierB}"; cd "${2:?dir}"; W="${3:?workers}"; shift 3
export CBB_TRUTH=verified_v1
for v in $(env | grep -o '^ENGINE_[A-Za-z0-9_]*' || true); do echo "REFUSE: $v is set"; exit 2; done
ts() { date -u +%H:%M:%SZ; }
IN=data/processed/models/engine_v3
test -f data/processed/models/late_game/round2/clk_D.pkl || { echo "MISSING clk_D.pkl (HF model_artifacts late_game/round2/)"; exit 3; }
if [ "$MODE" = tierA ]; then
  for off in 0 1000 2000 3000 4000; do
    t="lg3box_R9_s200_o${off}"
    [ -f results/engine_v0/$t/run_meta.json ] && { echo "[skip] $t"; continue; }
    echo "[run] $t $(ts)"
    scripts/box_run_v3.sh scripts/run_late_game_r3_closed_loop.py --seeds 200 --seed-offset $off --workers $W --tag $t || exit 4
  done
  for arm in "$@"; do
    t="lg3box_${arm}_s200_o0"
    [ -f results/engine_v0/$t/run_meta.json ] && { echo "[skip] $t"; continue; }
    echo "[run] $t $(ts)"
    scripts/box_run_v3.sh scripts/run_late_game_r3_closed_loop.py --late-game $arm --seeds 200 --workers $W --tag $t || exit 4
  done
  echo "[tierA done] $(ts)"
elif [ "$MODE" = tierB ]; then
  ARM="${1:?arm}"; RT="v3full_LG3${ARM}_s200_o0"; done_s=0
  while [ $done_s -lt 200 ]; do
    sub="${RT}_off${done_s}_n25"
    if [ -f results/engine_v0/$sub/run_meta.json ]; then echo "[skip] $sub"; done_s=$((done_s+25)); continue; fi
    echo "[chunk] $sub $(ts)"
    ENGINE_LATE_GAME=$ARM scripts/box_run_v3.sh scripts/run_engine.py --fold F2 --season 2025 --seeds 25 --seed-offset $done_s \
      --workers $W --games-per-block 60 --seeds-per-block 25 --tag $sub --results-dir results/engine_v0 --input-dir $IN \
      || { echo "CHUNK FAILED $sub"; exit 4; }
    done_s=$((done_s+25))
  done
  scripts/box_run_v3.sh scripts/concat_engine_runs.py --tag $RT --results-dir results/engine_v0 --overwrite && echo "[done] $RT $(ts)"
  mkdir -p results/engine_v0/v3full_grade
  scripts/box_run_v3.sh scripts/eval_gates.py --results results/engine_v0/$RT --season 2025 \
    --out results/engine_v0/v3full_grade/${RT}__verified.md && echo "[graded] $RT $(ts)"
else
  echo "mode must be tierA or tierB"; exit 1
fi
