#!/usr/bin/env bash
# run_lj_round_v1.sh -- lane A day round 4 (fg_make/experiments.md s29): J1/J2 on aL and TSR, seeds 0/1, F2 then F1,
# fold-2 harness after each F2 run. One step at a time; 4 single-thread processes at most. Resumable.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=data/processed/models/fg_make/round_lj
LOG=results/g9_team_response_v1
export PYTHONIOENCODING=utf-8
for fold in F2 F1; do
  for lg in roll28 seasonal; do
    J=$([ $lg = roll28 ] && echo J1 || echo J2)
    for arm in aL TSR; do
      for s in 0 1; do
        out=$R/$lg/${arm}_s${s}_${fold}
        [ -f "$out/league_level.json" ] && continue
        echo "$(date '+%F %T %z') launch ${arm}${J} ${fold} s${s}" >> $LOG/session_log.txt
        $PY scripts/train_fg_make_two_stage_v3.py --league $lg --arm $arm --fold $fold --seed $s --n-jobs 4 > $LOG/log_${arm}${J}_s${s}_${fold}.txt 2>&1
        rc=$?
        echo "$(date '+%F %T %z') end ${arm}${J} ${fold} s${s} rc=$rc" >> $LOG/session_log.txt
        if [ "$fold" = "F2" ]; then
          $PY scripts/diag_g9_harness_arm_v1.py --arm-dir $out/B1 --base S0 --name ${arm}${J}_s${s} > $LOG/log_harness_${arm}${J}_s${s}.txt 2>&1
          rc=$?
          echo "$(date '+%F %T %z') end harness ${arm}${J}_s${s} rc=$rc" >> $LOG/session_log.txt
        fi
      done
    done
  done
done
