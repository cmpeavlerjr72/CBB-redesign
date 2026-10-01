#!/usr/bin/env bash
# run_g9_harness_arms_v1.sh -- lane A day 2026-10-01: fold-2 harness of every fg_make arm (fg_make/experiments.md s23),
# one process at a time. Resumable: an arm whose harness parquet exists is skipped.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=data/processed/models/fg_make/round_g9
LOG=results/g9_team_response_v1
export PYTHONIOENCODING=utf-8
for arm in aR aG3R aG1R aTfix aG3 aG1; do
  for s in 0 1; do
    name=${arm}_s${s}
    [ -f "$LOG/harness_${name}.parquet" ] && { echo "skip $name"; continue; }
    case "$arm" in
      aTfix|aG3|aG1) dir=$R/${arm}_s${s}_F2/team_rate_features_E3_v4/B1; base=X_Tfix ;;
      *) dir=$R/${arm}_s${s}_F2/B1; base=S0 ;;
    esac
    echo "$(date '+%F %T %z') harness $name" >> $LOG/session_log.txt
    $PY scripts/diag_g9_harness_arm_v1.py --arm-dir "$dir" --base $base --name "$name" > "$LOG/log_harness_${name}.txt" 2>&1
    echo "$(date '+%F %T %z') end harness $name rc=$?" >> $LOG/session_log.txt
  done
done
