#!/usr/bin/env bash
# Lane G (2026-09-30) sibling of box_fullread_v2.sh (not edited): the fg_make G4 site-offset arm
# (ENGINE_FG_MAKE=round4site_G4, default-off, NOT ADOPTED) on the S0 tag of the v3 inputs, plus its grading.
#   scripts/box_fullread_laneG_v1.sh preflight
#   scripts/box_fullread_laneG_v1.sh run <WORKERS> [SEED_OFFSET=0] [SEEDS=200] [CHUNK=25]
#   scripts/box_fullread_laneG_v1.sh grade
# Resume: rerun the same command; finished chunks (run_meta.json present) are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
M=data/processed/models
B=scripts/box_run_v2.sh
G4DIR=$M/fg_make/round4_site/G4
export ENGINE_EVENT=round2_s1 ENGINE_CLOCK=v5b_glat_pmean ENGINE_ROTATION=reference ENGINE_FG3=decision8 CBB_TRUTH=verified_v1
REF=v3full_S0_s200_o0
FLOORS=v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000
case "${1:-}" in
preflight)
  ok=1
  for c in FGA_rim FGA_jump2 FGA_3; do
    [ -f "$G4DIR/manifest_$c.json" ] || { echo "MISSING $G4DIR/manifest_$c.json (HF: hf_sync_data.py pull --dirs model_artifacts --only 'fg_make/round4_site/**')"; ok=0; }
  done
  n=$(ls "$G4DIR"/*.joblib 2>/dev/null | wc -l); [ "$n" -eq 18 ] || { echo "expected 18 G4 joblibs, found $n"; ok=0; }
  for t in $REF ${FLOORS//,/ }; do [ -f "results/engine_v0/$t/games.parquet" ] || { echo "MISSING results/engine_v0/$t (HF results key)"; ok=0; }; done
  grep -q '"round4site_"' src/cbb_sim/engine/adapters.py || { echo "adapter registry line missing: checkout commit c5667bc or later"; ok=0; }
  [ "$ok" = 1 ] && echo "preflight OK" || exit 1 ;;
run)
  W="${2:?workers}"; OFF="${3:-0}"; SEEDS="${4:-200}"; CH="${5:-25}"
  export ENGINE_FG_MAKE=round4site_G4
  RT="v3full_G4_s${SEEDS}_o${OFF}"
  IN="$M/engine_v3_S0"
  export BOX_DOCKER_ARGS; BOX_DOCKER_ARGS="$(sed "s#\$PWD#$PWD#g" "$IN/docker_mounts.txt" | tr '\n' ' ')"
  $B scripts/ops_overlay_check_v1.py --input-dir "$IN" --root /app || { echo "OVERLAY CHECK FAILED"; exit 3; }
  # the overlay mounts must not hide the G4 directory inside the container
  $B -c "import os,sys; sys.exit(0 if len([f for f in os.listdir('$G4DIR') if f.endswith('.joblib')])==18 else 1)" \
    || { echo "G4 artifacts not visible inside the container at $G4DIR"; exit 3; }
  done_s=0
  while [ "$done_s" -lt "$SEEDS" ]; do
    this=$(( SEEDS - done_s < CH ? SEEDS - done_s : CH )); o=$(( OFF + done_s )); sub="${RT}_off${o}_n${this}"
    if [ -f "results/engine_v0/$sub/run_meta.json" ]; then echo "[skip] $sub"; done_s=$((done_s+this)); continue; fi
    echo "[chunk] $sub $(date -u +%H:%M:%SZ)"
    $B scripts/run_engine.py --fold F2 --season 2025 --seeds "$this" --seed-offset "$o" --workers "$W" \
        --games-per-block 60 --seeds-per-block 25 --tag "$sub" --results-dir results/engine_v0 --input-dir "$IN" || { echo "CHUNK FAILED $sub"; exit 4; }
    done_s=$((done_s+this))
  done
  $B scripts/concat_engine_runs.py --tag "$RT" --results-dir results/engine_v0 --overwrite && echo "[done] $RT $(date -u +%H:%M:%SZ)"
  grep -q round4site_G4 "results/engine_v0/$RT/run_meta.json" || { echo "run_meta does not record ENGINE_FG_MAKE=round4site_G4"; exit 5; } ;;
grade)
  O=results/engine_v0/laneG_grade; mkdir -p $O
  $B scripts/eval_gates.py --results results/engine_v0/v3full_G4_s200_o0 --season 2025 --out $O/v3full_G4_s200_o0__verified.md
  $B scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref $REF --floors $FLOORS \
      --arms v3full_G4_s200_o0 --labels G4 --out-json $O/pair_vetoes.json --out-md $O/pair_vetoes.md
  $B scripts/grade_laneG_site_loop_v1.py --results-dir results/engine_v0 --ref $REF --floors $FLOORS \
      --arms v3full_G4_s200_o0 --n-boot 500 --out-json $O/pair_site.json --out-md $O/pair_site.md ;;
*) sed -n 2,7p "$0"; exit 2 ;;
esac
