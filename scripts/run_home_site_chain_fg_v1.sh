#!/usr/bin/env bash
# Lane G (overnight 2026-09-30): fg_make home-site repair arms, one thread
# (core cap 2 shared with the FT/clock chain). Pre-registered in
# fg_make/experiments.md section 21 (commit ed640e4).
set -u
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 CBB_NJOBS=1
PY=.venv/Scripts/python.exe
L=results/home_site
echo "start $(powershell -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'")"
$PY scripts/train_fg_make_v4_site.py --arms G1,G2,G4 --folds F2,F1 --seeds 0 > $L/fg_arms_run.log 2>&1
echo "fg arms done $(powershell -Command "Get-Date -Format 'HH:mm:ss'")"
$PY scripts/train_fg_make_v4_site.py --arms S0 --folds F2,F1 --seeds 1 > $L/fg_S0_seed1_run.log 2>&1
echo "fg S0 seed1 done $(powershell -Command "Get-Date -Format 'HH:mm:ss'")"
