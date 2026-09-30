#!/usr/bin/env bash
# Arm Tfs launcher (operator 2026-09-30): builds the combined table, then trains PO T+corrected in_bonus.
# The trainer refuses --team-rate-table with --feature-table, so the adapter is applied in ops_build_tfs_table_v1.py.
set -uo pipefail
cd "$(dirname "$0")/.."
PO_ROOT=data/processed/models/possession_outcome/round_stageb
TAB=$PO_ROOT/tfs_feature_table_v1.parquet
mkdir -p logs/stageb
if [ ! -f "$TAB" ]; then
  bash scripts/box_run.sh scripts/ops_build_tfs_table_v1.py --table data/processed/team_rate_features_E3_v4.parquet \
    --overlay data/processed/models/possession_outcome/round8/in_bonus_overlay_v2state_v1.parquet --out "$TAB" > logs/stageb/tfs_table.log 2>&1 || { echo TABLE FAILED; tail -5 logs/stageb/tfs_table.log; exit 1; }
fi
tail -2 logs/stageb/tfs_table.log
nohup bash -c 'date -u +START_%s; bash scripts/box_run.sh scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F2 --season 2025 --seed 0 --n-jobs 12 --feature-table '"$TAB"' --overlay-keys game_id,poss_index,chance_number --out-root '"$PO_ROOT"'/Tfs; echo EXIT_$?; date -u +END_%s' > logs/stageb/po_Tfs.log 2>&1 < /dev/null &
echo started
