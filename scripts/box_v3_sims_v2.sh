#!/usr/bin/env bash
# Versioned sibling of lane G's box_v3_sims_v1.sh (operator 2026-09-30): adds arms S1tfs and S1K2O (S1 tag + ENGINE_SHOT_BLOCK=K2_Ocell), sample sims only.
#   scripts/box_v3_sims_v1.sh builds               # S0, S1 (+ S1opp), draw files
#   scripts/box_v3_sims_v1.sh sim <ARM> [WORKERS]  # ARM: S0 S0f1..S0f4 S1 S1opp S2 S3 K2O R8b R8bS
#   scripts/box_v3_sims_v1.sh grade                # eval_gates under CBB_TRUTH=verified_v1 + pairs
# Env: TABLE_E3 TABLE_E3OPP (default the v4 tables), TR_MISSING (raise|keep_served), SEEDS (200), SAMPLE, *_ROOT.
set -uo pipefail
cd "$(dirname "$0")/.."
TABLE_E3="${TABLE_E3:-data/processed/team_rate_features_E3_v4.parquet}"
TABLE_E3OPP="${TABLE_E3OPP:-data/processed/team_rate_features_E3opp_v4.parquet}"
VAR_O1A="${VAR_O1A:-data/processed/team_rate_variance_O1a_v3.parquet}"
TR_MISSING="${TR_MISSING:-raise}"
SEEDS="${SEEDS:-200}"
SAMPLE="${SAMPLE:-data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet}"
PO_ROOT=data/processed/models/possession_outcome/round_stageb
FG_ROOT=data/processed/models/fg_make/round_stageb
RB_ROOT=data/processed/models/rebound/round_stageb
M=data/processed/models
E3=$(basename "$TABLE_E3" .parquet); E3O=$(basename "$TABLE_E3OPP" .parquet)
PO_T="$PO_ROOT/T/$E3"; PO_TOPP="$PO_ROOT/Topp/$E3O"; FG_T="$FG_ROOT/T/$E3"; RB_T="$RB_ROOT/T/$E3/artifacts/s2_F2_A0B0C0_seed0"
B=scripts/box_run_v2.sh
mounts() { sed "s#\$PWD#$PWD#g" "$M/engine_v3_$1/docker_mounts.txt" | tr '\n' ' '; }
TRD="$M/engine_v3_trdraw"

case "${1:-}" in
builds)
  $B scripts/build_engine_inputs_v3_tag_v1.py --tag S0
  # S1: Stage B winners. Mixed stack (a sub-model whose T lost): add --no-table-for po|fg|rb and point that artifact flag at the SERVED dir (omit it).
  $B scripts/build_engine_inputs_v3_tag_v1.py --tag S1 --team-rate-table "$TABLE_E3" --team-rate-missing "$TR_MISSING" \
      --variance-table "$VAR_O1A" --po-artifacts "$PO_T" --fg-artifacts "$FG_T/B1" --rb-artifacts "$RB_T"
  [ -d "$PO_TOPP" ] && $B scripts/build_engine_inputs_v3_tag_v1.py --tag S1opp --table-po "$TABLE_E3OPP" --table-fg "$TABLE_E3" \
      --table-rb "$TABLE_E3" --team-rate-missing "$TR_MISSING" --po-artifacts "$PO_TOPP" --fg-artifacts "$FG_T/B1" --rb-artifacts "$RB_T"
  # Stage C draw files (lane F) on the S1 tag as base
  mkdir -p "$TRD"
  $B scripts/build_engine_inputs_trdraw_v1.py --inputs-dir "$M/engine_v3_S1" --variance none --K 1 --table "$TABLE_E3" --out "$TRD/S1_K1"
  $B scripts/build_engine_inputs_trdraw_v1.py --inputs-dir "$M/engine_v3_S1" --variance e3 --K 64 --seed 20260930 --table "$TABLE_E3" --out "$TRD/S2_e3_K64"
  $B scripts/build_engine_inputs_trdraw_v1.py --inputs-dir "$M/engine_v3_S1" --variance o1a --K 64 --seed 20260930 --table "$TABLE_E3" \
      --variance-table "$VAR_O1A" --out "$TRD/S3_o1a_K64"
  ;;
sim)
  ARM="${2:?arm}"; W="${3:-64}"
  TAG=S0; OFF=0; ENVV=()
  case "$ARM" in
    S0) ;; S0f1) OFF=1000 ;; S0f2) OFF=2000 ;; S0f3) OFF=3000 ;; S0f4) OFF=4000 ;;
    S1) TAG=S1 ;; S1opp) TAG=S1opp ;; S1tfs) TAG=S1tfs ;; S1K2O) TAG=S1; ENVV=(ENGINE_SHOT_BLOCK=K2_Ocell) ;;
    S2) TAG=S1; ENVV=(ENGINE_TEAM_RATE_DRAW=/app/$TRD/S2_e3_K64.npz) ;;
    S3) TAG=S1; ENVV=(ENGINE_TEAM_RATE_DRAW=/app/$TRD/S3_o1a_K64.npz) ;;
    K2O) ENVV=(ENGINE_SHOT_BLOCK=K2_Ocell) ;;           # served artifacts, v3 inputs (S0 tag)
    R8b) ENVV=(ENGINE_FOUL_JOINT=R8b) ;;
    R8bS) ENVV=(ENGINE_FOUL_JOINT=R8bS) ;;
    *) echo "unknown arm $ARM"; exit 2 ;;
  esac
  IN="$M/engine_v3_$TAG"
  export BOX_DOCKER_ARGS; BOX_DOCKER_ARGS="$(mounts "$TAG")"
  $B scripts/ops_overlay_check_v1.py --input-dir "$IN" --root /app || { echo "OVERLAY CHECK FAILED for $ARM"; exit 3; }
  env ${ENVV[@]+"${ENVV[@]}"} CBB_TRUTH=verified_v1 $B scripts/run_po4b_closed_loop_sample_v1.py --sample-file "$SAMPLE" --arm round2_s1 \
      --input-dir "$IN" --seeds "$SEEDS" --seed-offset "$OFF" --workers "$W" --tag "v3box_${ARM}_s${SEEDS}_o${OFF}" --results-dir results/engine_v0
  ;;
grade)
  export CBB_TRUTH=verified_v1
  mkdir -p results/engine_v0/v3box_grade
  for d in results/engine_v0/v3box_*_s*_o*; do t=$(basename "$d"); [ -f "$d/run_meta.json" ] || continue
    $B scripts/eval_gates.py --results "$d" --season 2025 --out "results/engine_v0/v3box_grade/$t.md"; done
  echo "pair with: scripts/diag_pair_gate_reports.py --a <S0>.md --b <arm>.md --noise <S0f1>.md  (floors: S0f1..S0f4)"
  ;;
*) sed -n 2,7p "$0"; exit 2 ;;
esac
