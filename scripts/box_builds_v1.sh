#!/usr/bin/env bash
# Tagged input builds for tonight's arms (operator 2026-09-30), from the Stage B artifacts. Runs ON the box.
#   scripts/box_builds_v1.sh S1 | S1tfs | S1opp
# S1 = E3 v4 table + T artifacts of all three sub-models; S1tfs = S1 with PO artifacts from Tfs; S1opp = PO on E3opp + Topp artifacts.
set -uo pipefail
cd "$(dirname "$0")/.."
E3=data/processed/team_rate_features_E3_v4.parquet; E3O=data/processed/team_rate_features_E3opp_v4.parquet
VAR=data/processed/team_rate_variance_O1a_v3.parquet
S=team_rate_features_E3_v4; SO=team_rate_features_E3opp_v4
PO=data/processed/models/possession_outcome/round_stageb; FG=data/processed/models/fg_make/round_stageb; RB=data/processed/models/rebound/round_stageb
RBT=$RB/T/$S/artifacts/s2_F2_A0B0C0_seed0
B=scripts/box_run_v2.sh
case "${1:-}" in
S1)    $B scripts/build_engine_inputs_v3_tag_v1.py --tag S1 --team-rate-table $E3 --team-rate-missing raise --variance-table $VAR --po-artifacts $PO/T/$S --fg-artifacts $FG/T/$S/B1 --rb-artifacts $RBT ;;
S1tfs) $B scripts/build_engine_inputs_v3_tag_v1.py --tag S1tfs --team-rate-table $E3 --team-rate-missing raise --variance-table $VAR --po-artifacts $PO/Tfs --fg-artifacts $FG/T/$S/B1 --rb-artifacts $RBT ;;
S1opp) $B scripts/build_engine_inputs_v3_tag_v1.py --tag S1opp --table-po $E3O --table-fg $E3 --table-rb $E3 --team-rate-missing raise --po-artifacts $PO/Topp/$SO --fg-artifacts $FG/T/$S/B1 --rb-artifacts $RBT ;;
*) sed -n 2,5p "$0"; exit 2 ;;
esac
