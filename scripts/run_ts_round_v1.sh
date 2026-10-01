#!/usr/bin/env bash
# run_ts_round_v1.sh -- lane A day 2026-10-01 round 2 (fg_make/experiments.md s25): two-stage trainings
# (F2 then F1, seeds 0 and 1; each writes arms TS1-TS3), then the fold-2 harness of every arm x seed.
# One step at a time, 4 single-thread processes at most. Resumable (finished outputs are skipped).
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=data/processed/models/fg_make/round_ts
LOG=results/g9_team_response_v1
export PYTHONIOENCODING=utf-8
for fold in F2 F1; do
  for s in 0 1; do
    if [ -f "$R/TS3_s${s}_${fold}/stageB_${fold}_s${s}.json" ]; then echo "skip train $fold s$s"; continue; fi
    echo "$(date '+%F %T %z') launch TS train ${fold} s${s}" >> $LOG/session_log.txt
    $PY scripts/train_fg_make_two_stage_v1.py --fold $fold --seed $s --n-jobs 4 --out-root $R > $LOG/log_TS_s${s}_${fold}.txt 2>&1
    rc=$?
    echo "$(date '+%F %T %z') end TS train ${fold} s${s} rc=$rc" >> $LOG/session_log.txt
  done
  if [ "$fold" = "F2" ]; then
    for arm in TS1 TS2 TS3; do for s in 0 1; do
      name=${arm}_s${s}
      [ -f "$LOG/harness_${name}.parquet" ] && continue
      echo "$(date '+%F %T %z') harness $name" >> $LOG/session_log.txt
      $PY scripts/diag_g9_harness_arm_v1.py --arm-dir $R/${arm}_s${s}_F2/B1 --base S0 --name $name > $LOG/log_harness_${name}.txt 2>&1
      rc=$?
      echo "$(date '+%F %T %z') end harness $name rc=$rc" >> $LOG/session_log.txt
    done; done
  fi
done
