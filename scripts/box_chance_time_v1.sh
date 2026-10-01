#!/usr/bin/env bash
# chance_time round 1 (lane I, chance_time/experiments.md s1.4): full-size box read of a Decision-11 stack PLUS
# ENGINE_CHANCE_TIME=<arm>. Versioned sibling of scripts/box_r9_v1.sh `full` mode (lane C, not edited); the only
# difference is that ENGINE_CHANCE_TIME is passed into the container (box_run_v2.sh does not pass it).
#   scripts/box_chance_time_v1.sh <BASE> <CT_ARM> <WORKERS> [OFFSET=0] [SEEDS=200]
# BASE: COMB9 (L2 + K2_Ocell + R9ao3) or COMB (L2 + K2_Ocell + R8b). CT_ARM: C12 or C2.
set -uo pipefail
cd "$(dirname "$0")/.."
M=data/processed/models
B=scripts/box_run_v2.sh
IN="$M/engine_v3_S0"
export ENGINE_EVENT=round2_s1 ENGINE_ROTATION=reference ENGINE_FG3=decision8
export CBB_TRUTH=verified_v1
BASE="${1:?base}"; CT="${2:?ct arm}"; W="${3:?workers}"; OFF="${4:-0}"; SEEDS="${5:-200}"
case "$BASE" in
  COMB)  export ENGINE_CLOCK=v5b_r6L2_glat_pmean ENGINE_SHOT_BLOCK=K2_Ocell ENGINE_FOUL_JOINT=R8b ;;
  COMB9) export ENGINE_CLOCK=v5b_r6L2_glat_pmean ENGINE_SHOT_BLOCK=K2_Ocell ENGINE_FOUL_JOINT=R9ao3 ;;
  *) echo "unknown base $BASE"; exit 2 ;;
esac
case "$CT" in C12|C2) export ENGINE_CHANCE_TIME="$CT" ;; *) echo "unknown CT arm $CT"; exit 2 ;; esac
echo "[env] $BASE+$CT ENGINE_CLOCK=$ENGINE_CLOCK ENGINE_SHOT_BLOCK=$ENGINE_SHOT_BLOCK ENGINE_FOUL_JOINT=$ENGINE_FOUL_JOINT ENGINE_CHANCE_TIME=$ENGINE_CHANCE_TIME"
export BOX_DOCKER_ARGS; BOX_DOCKER_ARGS="$(sed "s#\$PWD#$PWD#g" "$IN/docker_mounts.txt" | tr '\n' ' ') -e ENGINE_CHANCE_TIME"
$B scripts/ops_overlay_check_v1.py --input-dir "$IN" --root /app || { echo "OVERLAY CHECK FAILED"; exit 3; }
RT="v3full_${BASE}CT${CT#C}_s${SEEDS}_o${OFF}"; CH=25; done_s=0
while [ "$done_s" -lt "$SEEDS" ]; do
  this=$(( SEEDS - done_s < CH ? SEEDS - done_s : CH )); o=$(( OFF + done_s )); sub="${RT}_off${o}_n${this}"
  if [ -f "results/engine_v0/$sub/run_meta.json" ]; then echo "[skip] $sub"; done_s=$((done_s+this)); continue; fi
  echo "[chunk] $sub $(date -u +%H:%M:%SZ)"
  $B scripts/run_engine.py --fold F2 --season 2025 --seeds "$this" --seed-offset "$o" --workers "$W" \
      --games-per-block 60 --seeds-per-block 25 --tag "$sub" --results-dir results/engine_v0 --input-dir "$IN" || { echo "CHUNK FAILED $sub"; exit 4; }
  done_s=$((done_s+this))
done
$B scripts/concat_engine_runs.py --tag "$RT" --results-dir results/engine_v0 --overwrite && echo "[done] $RT"
mkdir -p results/engine_v0/v3full_grade
$B scripts/eval_gates.py --results "results/engine_v0/$RT" --season 2025 --out "results/engine_v0/v3full_grade/${RT}__verified.md"
