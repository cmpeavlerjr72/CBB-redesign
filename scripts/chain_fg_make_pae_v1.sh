#!/usr/bin/env bash
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8
chain() {
  fold=$1
  for spec in "R 0" "P1 0" "R 1" "P2 0"; do
    set -- $spec
    arm=$1; seed=$2
    out=data/processed/models/fg_make/round_pae/${arm}_s${seed}_${fold}
    .venv/Scripts/python.exe scripts/train_fg_make_v5_pae_v1.py --arm $arm --fold $fold --seed $seed --arms B1 \
      --no-floor --no-leak --n-jobs 6 --out-dir $out > results/fg_make_pae_v1/log_${arm}_s${seed}_${fold}.txt 2>&1
    echo "$fold $arm s$seed rc=$?" >> results/fg_make_pae_v1/chain_status.txt
  done
}
chain F2 &
chain F1 &
wait
echo ALLDONE >> results/fg_make_pae_v1/chain_status.txt
