#!/usr/bin/env bash
# TO arm retrains (operator 2026-09-30), commands from team_rate_estimator/experiments.md section 7d. Runs ON the box.
set -uo pipefail
cd "$(dirname "$0")/.."
M=data/processed/models; E3=data/processed/team_rate_features_E3_v4.parquet
mkdir -p logs/stageb
job() { local n="$1"; shift; nohup bash -c 'date -u +START_%s; bash scripts/box_run.sh "$@"; echo EXIT_$?; date -u +END_%s' _ "$@" > "logs/stageb/$n.log" 2>&1 < /dev/null & }
job po_TO scripts/train_possession_outcome_s1_par_anchor_artifacts_v1.py --anchor O --fold F2 --season 2025 --team-rate-table $E3 --team-rate-missing raise --n-jobs 24 --out-root $M/possession_outcome/round_stageb/TO
job rb_TO scripts/train_rebound_v3_par_anchor_artifacts_v1.py --anchor O --stage 2 --folds F2 --arms O --team-rate-table $E3 --team-rate-missing raise --n-jobs 24 --out-dir $M/rebound/round_stageb/TO
echo started TO
