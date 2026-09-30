#!/usr/bin/env bash
# TO arm: tagged inputs, offsets, 500-game verified sample x 200 seeds (operator 2026-09-30; commands from experiments.md 7d). ON the box.
set -uo pipefail
cd "$(dirname "$0")/.."
B=scripts/box_run_v2.sh; M=data/processed/models; E3=data/processed/team_rate_features_E3_v4.parquet; S=team_rate_features_E3_v4
$B scripts/build_engine_inputs_v3_tag_v1.py --tag S1TO --team-rate-table $E3 --team-rate-missing raise \
  --po-artifacts $M/possession_outcome/round_stageb/TO/$S --fg-artifacts $M/fg_make/round_stageb/T/$S/B1 \
  --rb-artifacts $M/rebound/round_stageb/TO/$S/artifacts/s2_F2_O_seed0 2>&1 | tail -1
$B scripts/build_engine_anchor_offsets_v1.py --input-dir $M/engine_v3_S1TO --families po,rb 2>&1 | tail -2
IN=$M/engine_v3_S1TO
export BOX_DOCKER_ARGS="$(sed "s#\$PWD#$PWD#g" $IN/docker_mounts.txt | tr '\n' ' ') -e ENGINE_SEASON_ANCHOR=/app/$IN/anchor_offsets_F2_2025.npz"
$B scripts/ops_overlay_check_v1.py --input-dir $IN --root /app || { echo OVERLAY FAILED; exit 3; }
CBB_TRUTH=verified_v1 $B scripts/run_po4b_closed_loop_sample_v1.py --sample-file data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet --arm round2_s1 \
  --input-dir $IN --seeds 200 --seed-offset 0 --workers 48 --tag v3box_S1TO_s200_o0 --results-dir results/engine_v0
