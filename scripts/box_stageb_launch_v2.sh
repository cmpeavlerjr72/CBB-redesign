#!/usr/bin/env bash
# Stage B launcher, operator version 2 (2026-09-30). Versioned sibling of lane J's box_stageb_launch_v1.sh (not edited).
# Differences: v4 tables; rebound through train_rebound_v3_par_artifacts_v1.py (writes the fitted models, which J's
# rebound trainer does not); the extra PO arm Tfs (T + corrected in_bonus overlay); a rebound R2 floor arm;
# anchored TO arms and the stage-2 drift cells are NOT run.
#   scripts/box_stageb_launch_v2.sh waveA    # PO R2/T/Topp/Tfs (48 procs) + fg R2/T (36 procs)
#   scripts/box_stageb_launch_v2.sh waveB    # rebound R2/T (46 procs); start when `free -g` < ~40% used
#   scripts/box_stageb_launch_v2.sh status
# Resumable: rerun the same command after a reclaim; done markers skip finished jobs, finished fits load from cuts/.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/stageb
TABLE_E3="${TABLE_E3:-data/processed/team_rate_features_E3_v4.parquet}"
TABLE_E3OPP="${TABLE_E3OPP:-data/processed/team_rate_features_E3opp_v4.parquet}"
TR_MISSING="${TR_MISSING:-raise}"
OVL=data/processed/models/possession_outcome/round8/in_bonus_overlay_v2state_v1.parquet
PO_ROOT=data/processed/models/possession_outcome/round_stageb
FG_ROOT=data/processed/models/fg_make/round_stageb
RB_ROOT=data/processed/models/rebound/round_stageb
E3=$(basename "$TABLE_E3" .parquet); E3O=$(basename "$TABLE_E3OPP" .parquet)
job() {
  local name="$1" marker="$2"; shift 2
  if [ -e "$marker" ]; then echo "[skip] $name (done: $marker)"; return; fi
  echo "[start] $name $(date -u +%H:%M:%SZ) -> logs/stageb/$name.log"
  nohup bash -c 'date -u +START_%s; scripts/box_run.sh "$@"; echo EXIT_$?; date -u +END_%s' _ "$@" > "logs/stageb/$name.log" 2>&1 &
}
case "${1:-}" in
waveA)
  job po_R2   "$PO_ROOT/R2_seed1/par_v1_report.json" scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F2 --season 2025 --seed 1 --n-jobs 12 --out-root "$PO_ROOT/R2_seed1"
  job po_T    "$PO_ROOT/T/$E3/par_v1_report.json" scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F2 --season 2025 --seed 0 --n-jobs 12 --team-rate-table "$TABLE_E3" --team-rate-missing "$TR_MISSING" --out-root "$PO_ROOT/T"
  job po_Topp "$PO_ROOT/Topp/$E3O/par_v1_report.json" scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F2 --season 2025 --seed 0 --n-jobs 12 --team-rate-table "$TABLE_E3OPP" --team-rate-missing "$TR_MISSING" --out-root "$PO_ROOT/Topp"
  job po_Tfs  "$PO_ROOT/Tfs/$E3/par_v1_report.json" scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F2 --season 2025 --seed 0 --n-jobs 12 --team-rate-table "$TABLE_E3" --team-rate-missing "$TR_MISSING" --feature-table "$OVL" --overlay-keys game_id,poss_index,chance_number --overlay-cols in_bonus --out-root "$PO_ROOT/Tfs"
  job fg_R2 "$FG_ROOT/R2_seed1/run_report.json" scripts/train_fg_make_v4_par_v1.py --mode run --arms B1 --seed 1 --no-floor --no-leak --n-jobs 18 --out-dir "$FG_ROOT/R2_seed1"
  [ -e "$FG_ROOT/T/$E3/run_report.json" ] || rm -f "$FG_ROOT/T/design_v4_extra_E3.parquet"
  job fg_T  "$FG_ROOT/T/$E3/run_report.json" scripts/train_fg_make_v4_par_v1.py --mode run --arms B1 --seed 0 --no-floor --no-leak --n-jobs 18 --team-rate-table "$TABLE_E3" --team-rate-missing "$TR_MISSING" --extra-cache "$FG_ROOT/T/design_v4_extra_E3.parquet" --out-dir "$FG_ROOT/T"
  ;;
waveB)
  job rb_R2 "$RB_ROOT/R2_seed1/artifacts/s2_F2_A0B0C0_seed1/manifest.json" scripts/train_rebound_v3_par_artifacts_v1.py --stage 2 --folds F2 --arms A0B0C0 --seed 1 --feeds --n-jobs 23 --out-dir "$RB_ROOT/R2_seed1"
  job rb_T  "$RB_ROOT/T/$E3/artifacts/s2_F2_A0B0C0_seed0/manifest.json" scripts/train_rebound_v3_par_artifacts_v1.py --stage 2 --folds F2 --arms A0B0C0 --seed 0 --feeds --n-jobs 23 --team-rate-table "$TABLE_E3" --team-rate-missing "$TR_MISSING" --out-dir "$RB_ROOT/T"
  ;;
status)
  echo "--- containers: $(docker ps -q | wc -l)"; free -g | head -2; uptime
  for f in logs/stageb/*.log; do [ -e "$f" ] && { echo "--- $f"; tail -n 2 "$f" | cut -c1-200; }; done ;;
*) sed -n 2,11p "$0"; exit 2 ;;
esac
