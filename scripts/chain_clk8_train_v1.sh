#!/bin/sh
# clock round 8 (experiments.md section 36): all arm x fold S1 schedules, max 3 concurrent (lane H core cap 3)
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8
mkdir -p results/clock_r8
run() { .venv/Scripts/python.exe scripts/train_clock_r8_tempo_v1.py --arm $1 --fold $2 > results/clock_r8/train_$1_$2.log 2>&1; }
date
run M1 F2 & run M2 F2 & run M2D F2 & wait
date
run G2D F2 & run M1 F1 & run M2 F1 & wait
date
run M2D F1 & run G2D F1 & wait
date
