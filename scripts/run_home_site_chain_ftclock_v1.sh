#!/usr/bin/env bash
# Lane G (overnight 2026-09-30): free_throw then clock site bake-offs, one
# thread each (core cap 2 shared with the fg_make chain). Pre-registered in
# free_throw/experiments.md s11 and clock/experiments.md s30 (commit e5dd38c).
set -u
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 CBB_NJOBS=1
PY=.venv/Scripts/python.exe
L=results/home_site
echo "start $(powershell -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'")"
$PY scripts/train_free_throw_v2_site.py --arms FT0,FT1,FT2 --folds F2,F1 --seeds 0 > $L/ft_run.log 2>&1
echo "ft seed0 done $(powershell -Command "Get-Date -Format 'HH:mm:ss'")"
$PY scripts/train_free_throw_v2_site.py --arms FT0 --folds F2,F1 --seeds 1 > $L/ft_run_seed1.log 2>&1
echo "ft seed1 done $(powershell -Command "Get-Date -Format 'HH:mm:ss'")"
$PY scripts/train_clock_v3c_site.py --arms C0,C1,C2 --folds F2,F1 > $L/clock_run.log 2>&1
echo "clock done $(powershell -Command "Get-Date -Format 'HH:mm:ss'")"
