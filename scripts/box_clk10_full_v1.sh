#!/usr/bin/env bash
# Clock round 10 full-size read ON THE BOX (docs/models/clock/experiments.md sections 43-45; PM ruling 2026-10-14: four control
# draws at the full 200 seeds). Runs as ec2-user in ~/cbb after box_bootstrap_v1.sh install/clone/pull/build.
#   scripts/box_clk10_full_v1.sh parity     # served-default parity smoke vs parity_reference_windows_v10.json (hard stop on FAIL)
#   scripts/box_clk10_full_v1.sh runs       # control o0, E1 o0, draws o1000..o4000; 5,710 x 200 each, fold 2 / 2025, two at a time
#   scripts/box_clk10_full_v1.sh post       # chunked extracts (all 200 seeds, and the first 50 as <tag>_f50) + dispersion v2
#   scripts/box_clk10_full_v1.sh grade      # grade_clock_env_full_v2 deciding (200 vs 200) and 50-seed-draw reads + tables
# Every python call runs in the cbb-sweep image with the host clone bind-mounted, CBB_TRUTH=verified_v1, thread vars pinned to 1.
# Only ENGINE_CLOCK differs between arm and control; seeds 0-199 are shared (paired), draws at disjoint offsets 1000-4000.
set -uo pipefail
cd "$HOME/cbb"
mkdir -p logs results/clock_env10/logs
IMG="${CBB_IMAGE:-cbb-sweep}"
W="${W:-94}"
dr() {
  docker run --rm -e HF_TOKEN -e ENGINE_CLOCK -e CBB_TRUTH=verified_v1 -e PYTHONIOENCODING=utf-8 \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -v "$PWD:/app" -w /app --entrypoint python "$IMG" -u "$@"
}
run1() {  # run1 <ctrl|E1> <offset>
  local LAB=$1 o=$2 TAG="clk10full_$1_o$2_s200"
  if [ -f "results/clock_env10/full/$TAG/run_meta.json" ]; then echo "$TAG exists, skipped"; return 0; fi
  if [ "$LAB" = E1 ]; then export ENGINE_CLOCK=v5b_r8K2_envE1_glat_pmean; else unset ENGINE_CLOCK; fi
  echo "$(date -u +%FT%TZ) start $TAG"
  dr scripts/run_engine.py --fold F2 --season 2025 --seeds 200 --seed-offset "$o" --workers "$W" \
    --input-dir data/processed/models/engine_v3 --tag "$TAG" --no-players --trajectory-seeds $((o + 200)) \
    --results-dir results/clock_env10/full > "results/clock_env10/logs/run_$TAG.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $TAG rc=$rc"
  return $rc
}
post1() {  # post1 <tag> <offset>
  local TAG=$1 o=$2
  dr scripts/diag_c4_extract_chunked_v1.py --tag "$TAG" --traj-file "results/trajectories/$TAG/trajectory.parquet" \
    > "results/clock_env10/logs/extract_$TAG.log" 2>&1 || return 2
  dr scripts/diag_margin_dispersion_v2.py --prefix "results/diag_c4/${TAG}_" --seeds 200 \
    --out "results/clock_env10/full_dispersion_s200/$TAG" > "results/clock_env10/logs/disp200_$TAG.log" 2>&1 || return 3
  dr scripts/diag_c4_extract_chunked_v1.py --tag "${TAG}_f50" --traj-file "results/trajectories/$TAG/trajectory.parquet" \
    --seed-max $((o + 50)) > "results/clock_env10/logs/extract_${TAG}_f50.log" 2>&1 || return 4
  dr scripts/diag_margin_dispersion_v2.py --prefix "results/diag_c4/${TAG}_f50_" --seeds 50 \
    --out "results/clock_env10/full_dispersion_s50/$TAG" > "results/clock_env10/logs/disp50_$TAG.log" 2>&1 || return 5
  echo "$(date -u +%FT%TZ) post $TAG done"
}
case "${1:-}" in
parity)
  dr scripts/run_parity_smoke_v1.py --workers 32 --tag box1014_parity_v10 ;;
runs)
  rcs=0
  for pair in "ctrl:0 E1:0" "ctrl:1000 ctrl:2000" "ctrl:3000 ctrl:4000"; do
    pids=()
    for j in $pair; do ( run1 "${j%%:*}" "${j##*:}" ) & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || rcs=$((rcs + 1)); done
  done
  echo "RUNS_DONE failures=$rcs" ;;
post)
  rcs=0; pids=()
  for j in E1:0 ctrl:0 ctrl:1000 ctrl:2000 ctrl:3000 ctrl:4000; do
    ( post1 "clk10full_${j%%:*}_o${j##*:}_s200" "${j##*:}" ) & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rcs=$((rcs + 1)); done
  echo "POST_DONE failures=$rcs" ;;
grade)
  dr scripts/grade_clock_env_full_v2.py --traj-seeds 200 --draw-seeds 200 --draw-run-seeds 200 \
    --disp-dir results/clock_env10/full_dispersion_s200 --out results/clock_env10/grade_full_s200_d200.json \
    > results/clock_env10/logs/grade_full_d200.log 2>&1 & p1=$!
  dr scripts/grade_clock_env_full_v2.py --traj-seeds 50 --draw-seeds 50 --draw-run-seeds 200 \
    --disp-dir results/clock_env10/full_dispersion_s50 --out results/clock_env10/grade_full_s200_d50.json \
    > results/clock_env10/logs/grade_full_d50.log 2>&1 & p2=$!
  wait $p1; r1=$?; wait $p2; r2=$?
  echo "grade rc d200=$r1 d50=$r2"
  [ $r1 -eq 0 ] && [ $r2 -eq 0 ] || exit 5
  dr scripts/grade_clock_env_full_tables_v1.py --deciding results/clock_env10/grade_full_s200_d200.json \
    --d50 results/clock_env10/grade_full_s200_d50.json --out results/clock_env10/tables_full.md
  echo GRADE_DONE ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
