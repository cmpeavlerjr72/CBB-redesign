#!/usr/bin/env bash
# Versioned sibling of box_fullread_v1.sh (AWS operator 2026-10-01): adds the Decision 11 arms L2 (clock round 6) and
# COMB (L2 + K2_Ocell + R8b), grades under verified and legacy truth, and pushes results to HF after each read.
#   scripts/box_fullread_v2.sh run <ARM> <WORKERS> <SEED_OFFSET> [SEEDS=200] [CHUNK=25]
#   ARM: S0 S0f* K2O R8b L2 COMB (all on the S0 tag of the v3 inputs)
#   scripts/box_fullread_v2.sh grade <RESULT_TAG...>   -> results/engine_v0/v3full_grade/<tag>__verified.md and __current.md
#     (__current = CBB_TRUTH=legacy_v0, i.e. the pre-flip truth that 2026-09-30's "current" column used)
# Resume: rerun the same command; finished chunks (run_meta.json present) are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
M=data/processed/models
B=scripts/box_run_v2.sh
export ENGINE_EVENT=round2_s1 ENGINE_CLOCK=v5b_glat_pmean ENGINE_ROTATION=reference ENGINE_FG3=decision8
case "${1:-}" in
run)
  ARM="${2:?arm}"; W="${3:?workers}"; OFF="${4:?offset}"; SEEDS="${5:-200}"; CH="${6:-25}"
  TAG=S0
  case "$ARM" in
    S0|S0f*) ;;
    K2O) export ENGINE_SHOT_BLOCK=K2_Ocell ;;
    R8b) export ENGINE_FOUL_JOINT=R8b ;;
    L2) export ENGINE_CLOCK=v5b_r6L2_glat_pmean ;;
    COMB) export ENGINE_CLOCK=v5b_r6L2_glat_pmean ENGINE_SHOT_BLOCK=K2_Ocell ENGINE_FOUL_JOINT=R8b ;;
    *) echo "unknown arm $ARM"; exit 2 ;;
  esac
  echo "[env] $ARM ENGINE_CLOCK=$ENGINE_CLOCK ENGINE_SHOT_BLOCK=${ENGINE_SHOT_BLOCK:-} ENGINE_FOUL_JOINT=${ENGINE_FOUL_JOINT:-}"
  RT="v3full_${ARM}_s${SEEDS}_o${OFF}"
  IN="$M/engine_v3_$TAG"
  export BOX_DOCKER_ARGS; BOX_DOCKER_ARGS="$(sed "s#\$PWD#$PWD#g" "$IN/docker_mounts.txt" | tr '\n' ' ')"
  $B scripts/ops_overlay_check_v1.py --input-dir "$IN" --root /app || { echo "OVERLAY CHECK FAILED for $ARM"; exit 3; }
  done_s=0
  while [ "$done_s" -lt "$SEEDS" ]; do
    this=$(( SEEDS - done_s < CH ? SEEDS - done_s : CH )); o=$(( OFF + done_s )); sub="${RT}_off${o}_n${this}"
    if [ -f "results/engine_v0/$sub/run_meta.json" ]; then echo "[skip] $sub"; done_s=$((done_s+this)); continue; fi
    echo "[chunk] $sub $(date -u +%H:%M:%SZ)"
    $B scripts/run_engine.py --fold F2 --season 2025 --seeds "$this" --seed-offset "$o" --workers "$W" \
        --games-per-block 60 --seeds-per-block 25 --tag "$sub" --results-dir results/engine_v0 --input-dir "$IN" || { echo "CHUNK FAILED $sub"; exit 4; }
    done_s=$((done_s+this))
  done
  $B scripts/concat_engine_runs.py --tag "$RT" --results-dir results/engine_v0 --overwrite && echo "[done] $RT $(date -u +%H:%M:%SZ)" ;;
grade)
  shift; mkdir -p results/engine_v0/v3full_grade
  for t in "$@"; do
    CBB_TRUTH=verified_v1 $B scripts/eval_gates.py --results "results/engine_v0/$t" --season 2025 --out "results/engine_v0/v3full_grade/${t}__verified.md" &
    CBB_TRUTH=legacy_v0 $B scripts/eval_gates.py --results "results/engine_v0/$t" --season 2025 --out "results/engine_v0/v3full_grade/${t}__current.md" &
    wait
  done ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
