#!/usr/bin/env bash
# run_tsr_round_v1.sh -- lane A day 2026-10-01 round 3 (fg_make/experiments.md s27): TSR and aL, seeds 0/1,
# F2 then F1, then the fold-2 harness of each. One step at a time, 4 single-thread processes at most.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=data/processed/models/fg_make/round_tsr
LOG=results/g9_team_response_v1
export PYTHONIOENCODING=utf-8
for fold in F2 F1; do
  for arm in aL TSR; do
    for s in 0 1; do
      [ -f "$R/${arm}_s${s}_${fold}/stageB_${fold}_s${s}.json" ] && continue
      echo "$(date '+%F %T %z') launch ${arm} ${fold} s${s}" >> $LOG/session_log.txt
      $PY scripts/train_fg_make_two_stage_v2.py --fold $fold --seed $s --arm $arm --n-jobs 4 --out-root $R > $LOG/log_${arm}_s${s}_${fold}.txt 2>&1
      rc=$?
      echo "$(date '+%F %T %z') end ${arm} ${fold} s${s} rc=$rc" >> $LOG/session_log.txt
      if [ "$fold" = "F2" ]; then
        echo "$(date '+%F %T %z') harness ${arm}_s${s}" >> $LOG/session_log.txt
        $PY scripts/diag_g9_harness_arm_v1.py --arm-dir $R/${arm}_s${s}_F2/B1 --base S0 --name ${arm}_s${s} > $LOG/log_harness_${arm}_s${s}.txt 2>&1
        rc=$?
        echo "$(date '+%F %T %z') end harness ${arm}_s${s} rc=$rc" >> $LOG/session_log.txt
      fi
    done
  done
done
