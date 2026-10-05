#!/bin/sh
# clock round 8: horn tap (possessions started by time-left bucket), 10 paired seeds per clock mode, max 3 concurrent
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1
run() { ENGINE_CLOCK=$1 .venv/Scripts/python.exe scripts/diag_clock_r8_horn_tap_v1.py --label $2 --seeds 10 > results/clock_r8/horn_$2.log 2>&1; }
date
run v5b_r6L2_glat_pmean served & run v5b_r8K2_glat_pmean K2 & run v5b_r8K2M_glat_pmean K2M & wait
date
run v5b_r8M2D_glat_pmean M2D & run v5b_r8K1_glat_pmean K1 & wait
date
