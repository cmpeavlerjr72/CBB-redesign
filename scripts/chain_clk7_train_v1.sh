#!/bin/sh
# clock round 7 (experiments.md section 32): all arm x fold S1 schedules, max 3 concurrent (lane H core cap 3)
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8
run() { .venv/Scripts/python.exe scripts/train_clock_r7_pace_v1.py --arm $1 --fold $2 > results/clock_r7/train_$1_$2.log 2>&1; }
run C0 F2 & run A1 F2 & run A3 F2 & wait
run C0 F1 & run A1 F1 & run A2 F1 & wait
run A3 F1
date
