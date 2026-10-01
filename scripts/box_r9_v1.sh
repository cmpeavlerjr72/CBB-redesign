#!/usr/bin/env bash
# Foul round 9 (lane C, PO experiments.md s26) box jobs. Runs ON the box from the repo root, inside the operator's
# docker wrapper (scripts/box_run_v2.sh), on the S0 v3 tag (engine_v3_S0 + its docker mounts), verified truth.
#   scripts/box_r9_v1.sh full   <ARM> <WORKERS> [OFFSET=0] [SEEDS=200]   # 5,710 games, chunks of 25 seeds (resumable)
#   scripts/box_r9_v1.sh sample <ARM> <WORKERS> [OFFSET=0]               # 500 verified games x 200 seeds, untapped
#   scripts/box_r9_v1.sh tap    <ARM> <WORKERS> [OFFSET=0]               # same 500 x 200, tapped, half_agg.parquet only
# ARM: S0 (no flag) or any ENGINE_FOUL_JOINT value (R8b, R9ao1, R9ao3).
set -uo pipefail
cd "$(dirname "$0")/.."
M=data/processed/models
B=scripts/box_run_v2.sh
IN="$M/engine_v3_S0"
SAMPLE=data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet
export ENGINE_EVENT=round2_s1 ENGINE_CLOCK=v5b_glat_pmean ENGINE_ROTATION=reference ENGINE_FG3=decision8
export CBB_TRUTH=verified_v1
MODE="${1:?mode}"; ARM="${2:?arm}"; W="${3:?workers}"; OFF="${4:-0}"; SEEDS="${5:-200}"
if [ "$ARM" != "S0" ]; then export ENGINE_FOUL_JOINT="$ARM"; else unset ENGINE_FOUL_JOINT; fi
export BOX_DOCKER_ARGS; BOX_DOCKER_ARGS="$(sed "s#\$PWD#$PWD#g" "$IN/docker_mounts.txt" | tr '\n' ' ')"
$B scripts/ops_overlay_check_v1.py --input-dir "$IN" --root /app || { echo "OVERLAY CHECK FAILED"; exit 3; }
case "$MODE" in
full)
  RT="v3full_${ARM}_s${SEEDS}_o${OFF}"; CH=25; done_s=0
  while [ "$done_s" -lt "$SEEDS" ]; do
    this=$(( SEEDS - done_s < CH ? SEEDS - done_s : CH )); o=$(( OFF + done_s )); sub="${RT}_off${o}_n${this}"
    if [ -f "results/engine_v0/$sub/run_meta.json" ]; then echo "[skip] $sub"; done_s=$((done_s+this)); continue; fi
    echo "[chunk] $sub $(date -u +%H:%M:%SZ)"
    $B scripts/run_engine.py --fold F2 --season 2025 --seeds "$this" --seed-offset "$o" --workers "$W" \
        --games-per-block 60 --seeds-per-block 25 --tag "$sub" --results-dir results/engine_v0 --input-dir "$IN" || { echo "CHUNK FAILED $sub"; exit 4; }
    done_s=$((done_s+this))
  done
  $B scripts/concat_engine_runs.py --tag "$RT" --results-dir results/engine_v0 --overwrite && echo "[done] $RT"
  $B scripts/eval_gates.py --results "results/engine_v0/$RT" --season 2025 --out "results/engine_v0/v3full_grade/${RT}__verified.md" ;;
sample)
  $B scripts/run_po4b_closed_loop_sample_v1.py --sample-file "$SAMPLE" --arm round2_s1 --input-dir "$IN" \
      --seeds 200 --seed-offset "$OFF" --workers "$W" --tag "v3box_${ARM}_s200_o${OFF}" --results-dir results/engine_v0
  $B scripts/eval_gates.py --results "results/engine_v0/v3box_${ARM}_s200_o${OFF}" --season 2025 \
      --out "results/engine_v0/v3box_grade/v3box_${ARM}_s200_o${OFF}.md" ;;
tap)
  E=(); [ "$ARM" != "S0" ] && E=(--env "ENGINE_FOUL_JOINT=$ARM")
  $B scripts/run_foul_joint_tap_v2.py --tag "r9taph_${ARM}_s200_o${OFF}" --sample-file "$SAMPLE" --input-dir "$IN" \
      --seeds 200 --seed-offset "$OFF" --workers "$W" --agg-halves --no-players ${E[@]+"${E[@]}"} ;;
*) sed -n 2,8p "$0"; exit 2 ;;
esac
