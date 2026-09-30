#!/usr/bin/env bash
# Full-size closed-loop read on a tagged v3 input dir (operator 2026-09-30). All 5,710 fold-2 games.
#   scripts/box_fullread_v1.sh run <ARM> <WORKERS> <SEED_OFFSET> [SEEDS=200] [CHUNK=25]
#   ARM -> input tag: S0, S0f1..S0f4 -> engine_v3_S0 ; S1 -> engine_v3_S1 ; S1tfs -> engine_v3_S1tfs ; S1opp -> engine_v3_S1opp
# Overlay check first (hard stop), chunks of CHUNK seeds (resume: rerun the same command, finished chunks are skipped),
# then concat into results/engine_v0/v3full_<ARM>_s<SEEDS>_o<OFFSET>. Same ENGINE_* env as run_aws_sweep.sh's defaults.
# Grade: scripts/box_fullread_v1.sh grade <RESULT_TAG...>   (eval_gates under CBB_TRUTH=verified_v1 and under current truth)
set -uo pipefail
cd "$(dirname "$0")/.."
M=data/processed/models
B=scripts/box_run_v2.sh
export ENGINE_EVENT=round2_s1 ENGINE_CLOCK=v5b_glat_pmean ENGINE_ROTATION=reference ENGINE_FG3=decision8
case "${1:-}" in
run)
  ARM="${2:?arm}"; W="${3:?workers}"; OFF="${4:?offset}"; SEEDS="${5:-200}"; CH="${6:-25}"
  TRD=/app/data/processed/models/engine_v3_trdraw
  case "$ARM" in
    S0|S0f*) TAG=S0 ;; S1) TAG=S1 ;; S1tfs) TAG=S1tfs ;; S1opp) TAG=S1opp ;;
    S2) TAG=S1; export ENGINE_TEAM_RATE_DRAW=$TRD/S2_e3_K64.npz ;;
    S3) TAG=S1; export ENGINE_TEAM_RATE_DRAW=$TRD/S3_o1a_K64.npz ;;
    K2O) TAG=S0; export ENGINE_SHOT_BLOCK=K2_Ocell ;;
    S1K2O) TAG=S1; export ENGINE_SHOT_BLOCK=K2_Ocell ;;
    R8b) TAG=S0; export ENGINE_FOUL_JOINT=R8b ;;
    R8bS) TAG=S0; export ENGINE_FOUL_JOINT=R8bS ;;
    *) TAG="$ARM" ;;
  esac
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
    CBB_TRUTH=verified_v1 $B scripts/eval_gates.py --results "results/engine_v0/$t" --season 2025 --out "results/engine_v0/v3full_grade/${t}__verified.md"
    $B scripts/eval_gates.py --results "results/engine_v0/$t" --season 2025 --out "results/engine_v0/v3full_grade/${t}__current.md"
  done ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
