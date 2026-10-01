#!/usr/bin/env bash
# run_g9_fg_arms_v1.sh -- lane A day 2026-10-01: every fg_make arm of fg_make/experiments.md section 23,
# sequential (one trainer at a time, 4 single-thread processes = the lane's 4-core cap). Resumable: a run whose
# preds file exists is skipped.  Usage: bash scripts/run_g9_fg_arms_v1.sh [F2|F1|all]
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=data/processed/models/fg_make/round_g9
LOG=results/g9_team_response_v1
E3=data/processed/team_rate_features_E3_v4.parquet
export PYTHONIOENCODING=utf-8
WHICH=${1:-all}

arm_args() {
  case "$1" in
    aR)    echo "" ;;
    aG3R)  echo "--monotone --min-child cv" ;;
    aG1R)  echo "--extra-features off_att_prior,def_att_prior" ;;
    aTfix) echo "--team-rate-table $E3 --rawfix" ;;
    aG3)   echo "--team-rate-table $E3 --rawfix --monotone --min-child cv" ;;
    aG1)   echo "--team-rate-table $E3 --rawfix --e3-variance --extra-features off_make_v,def_allow_v" ;;
  esac
}

run() {  # arm fold seed
  local arm=$1 fold=$2 seed=$3
  local out=$R/${arm}_s${seed}_${fold}
  local pdir=$out
  case "$arm" in aTfix|aG3|aG1) pdir=$out/team_rate_features_E3_v4 ;; esac
  if [ -f "$pdir/preds_${fold}_s${seed}.parquet" ]; then echo "skip $arm $fold s$seed"; return; fi
  local extra=""
  case "$arm" in aTfix|aG3|aG1) extra="--extra-cache $R/extra_${arm}_s${seed}_${fold}.parquet" ;; esac
  echo "$(date '+%F %T %z') launch ${arm}_s${seed}_${fold}" >> $LOG/session_log.txt
  $PY scripts/train_fg_make_v4_par_g9_v1.py --fold "$fold" $(arm_args "$arm") --arms B1 --no-floor --no-leak \
      --seed "$seed" --n-jobs 4 --out-dir "$out" $extra > "$LOG/log_${arm}_s${seed}_${fold}.txt" 2>&1
  echo "$(date '+%F %T %z') end ${arm}_s${seed}_${fold} rc=$?" >> $LOG/session_log.txt
}

ARMS="aR aG3R aG1R aTfix aG3 aG1"
if [ "$WHICH" = "F2" ] || [ "$WHICH" = "all" ]; then
  for a in $ARMS; do for s in 0 1; do run $a F2 $s; done; done
fi
if [ "$WHICH" = "F1" ] || [ "$WHICH" = "all" ]; then
  for a in $ARMS; do for s in 0 1; do run $a F1 $s; done; done
fi
