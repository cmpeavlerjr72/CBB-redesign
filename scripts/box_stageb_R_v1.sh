#!/usr/bin/env bash
# Arm R retrain (served spec, served features, seed 0) on the same par trainers, so T/Topp/Tfs/R2 are graded against an
# R produced by the identical code path and grader (operator 2026-09-30; not in the pre-registered job list, cheap: 1-6 min).
set -uo pipefail
cd "$(dirname "$0")/.."
PO=data/processed/models/possession_outcome/round_stageb; FG=data/processed/models/fg_make/round_stageb; RB=data/processed/models/rebound/round_stageb
job() { local n="$1"; shift; nohup bash -c 'date -u +START_%s; bash scripts/box_run.sh "$@"; echo EXIT_$?; date -u +END_%s' _ "$@" > "logs/stageb/$n.log" 2>&1 < /dev/null & }
job po_R scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F2 --season 2025 --seed 0 --n-jobs 12 --out-root $PO/R_seed0
job fg_R scripts/train_fg_make_v4_par_v1.py --mode run --arms B1 --seed 0 --no-floor --no-leak --n-jobs 18 --out-dir $FG/R_seed0
job rb_R scripts/train_rebound_v3_par_artifacts_v1.py --stage 2 --folds F2 --arms A0B0C0 --seed 0 --feeds --n-jobs 23 --out-dir $RB/R_seed0
echo started R arms
