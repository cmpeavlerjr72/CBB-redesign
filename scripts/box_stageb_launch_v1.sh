#!/usr/bin/env bash
# Stage B retrain launcher for the box (lane J 2026-09-30). NOTHING here runs locally by itself.
#
#   scripts/box_stageb_launch_v1.sh wave1      # PO R2/T/Topp (F2), fg_make R2/T, rebound T
#   scripts/box_stageb_launch_v1.sh wave2      # rebound stage-2 drift cells (4), start when
#                                              # `free -g` shows < ~40% used after wave1 ramps
#   scripts/box_stageb_launch_v1.sh status     # what is running / done, memory, load
#
# Every job is one detached background `box_run.sh` with its own log in logs/stageb/. Every job is
# RESUMABLE: after a spot reclaim, rerun the same wave command; finished refit dates are loaded from
# their per-task checkpoints and only missing fits run. Jobs already finished (their done marker
# exists) are skipped. n_jobs are sized so wave1 = 95 worker processes, wave2 = 92 (total 187 of 192).
# Tables: data/processed/team_rate_features_{E3,E3opp}_v2.parquet (synced via the `team_rate_tables`
# HF key). Change TABLE_E3 / TABLE_E3OPP below if the "v2 with one added rate" tables arrive.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/stageb
TABLE_E3="${TABLE_E3:-data/processed/team_rate_features_E3_v2.parquet}"
TR_MISSING="${TR_MISSING:-raise}"   # PM decision: raise | keep_served (773 PO rows had no table key in the local smoke)
TABLE_E3OPP="${TABLE_E3OPP:-data/processed/team_rate_features_E3opp_v2.parquet}"
PO_ROOT=data/processed/models/possession_outcome/round_stageb
FG_ROOT=data/processed/models/fg_make/round_stageb
RB_ROOT=data/processed/models/rebound/round_stageb

job() {   # job NAME DONE_MARKER cmd...
  local name="$1" marker="$2"; shift 2
  if [ -e "$marker" ]; then echo "[skip] $name (done: $marker)"; return; fi
  echo "[start] $name -> logs/stageb/$name.log"
  nohup scripts/box_run.sh "$@" > "logs/stageb/$name.log" 2>&1 &
}

case "${1:-}" in
wave1)
  # possession_outcome: 12 fits per arm on fold 2 (6 monthly refits x first/cont)
  job po_R2   "$PO_ROOT/R2_seed1/par_v1_report.json" scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F2 --season 2025 --seed 1 --n-jobs 12 --out-root "$PO_ROOT/R2_seed1"
  job po_T    "$PO_ROOT/T/$(basename "$TABLE_E3" .parquet)/par_v1_report.json" scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F2 --season 2025 --seed 0 --n-jobs 12 --team-rate-table "$TABLE_E3" --team-rate-missing "$TR_MISSING" --out-root "$PO_ROOT/T"
  job po_Topp "$PO_ROOT/Topp/$(basename "$TABLE_E3OPP" .parquet)/par_v1_report.json" scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F2 --season 2025 --seed 0 --n-jobs 12 --team-rate-table "$TABLE_E3OPP" --team-rate-missing "$TR_MISSING" --out-root "$PO_ROOT/Topp"
  # fg_make round-4 B1: 3 classes x ~6 refits = 18 fits per arm
  job fg_R2 "$FG_ROOT/R2_seed1/run_report.json" scripts/train_fg_make_v4_par_v1.py --mode run --arms B1 --seed 1 --no-floor --no-leak --n-jobs 18 --out-dir "$FG_ROOT/R2_seed1"
  [ -e "$FG_ROOT/T/$(basename "$TABLE_E3" .parquet)/run_report.json" ] || rm -f "$FG_ROOT/T/design_v4_extra_E3.parquet"   # resume: cache must be rebuilt (wrapper refuses a stale one)
  job fg_T  "$FG_ROOT/T/$(basename "$TABLE_E3" .parquet)/run_report.json" scripts/train_fg_make_v4_par_v1.py --mode run --arms B1 --seed 0 --no-floor --no-leak --n-jobs 18 --team-rate-table "$TABLE_E3" --team-rate-missing "$TR_MISSING" --extra-cache "$FG_ROOT/T/design_v4_extra_E3.parquet" --out-dir "$FG_ROOT/T"
  # rebound T: same spec/seed as served (A0B0C0 stage 2), E3 table, 23 weekly refits
  job rb_T "$RB_ROOT/T/$(basename "$TABLE_E3" .parquet)/cells/s2_F2_A0B0C0_seed0.json" scripts/train_rebound_v3_par_v1.py --stage 2 --folds F2 --arms A0B0C0 --seed 0 --feeds --n-jobs 23 --team-rate-table "$TABLE_E3" --team-rate-missing "$TR_MISSING" --out-dir "$RB_ROOT/T"
  ;;
wave2)
  # rebound stage 2, the four drift-block-carry cells (docs/tests/rebound_round_drift_block_carry_2026-09-18.md s7)
  job rb_A0s0 "$RB_ROOT/S2/cells/s2_F2_A0B0C0_seed0.json"  scripts/train_rebound_v3_par_v1.py --stage 2 --folds F2 --arms A0B0C0 --seed 0 --feeds --n-jobs 23 --out-dir "$RB_ROOT/S2"
  job rb_A0s1 "$RB_ROOT/S2s1/cells/s2_F2_A0B0C0_seed1.json" scripts/train_rebound_v3_par_v1.py --stage 2 --folds F2 --arms A0B0C0 --seed 1 --feeds --n-jobs 23 --out-dir "$RB_ROOT/S2s1"
  job rb_A5   "$RB_ROOT/S2a5/cells/s2_F2_A5_seed0.json"   scripts/train_rebound_v3_par_v1.py --stage 2 --folds F2 --arms A5 --seed 0 --feeds --n-jobs 23 --out-dir "$RB_ROOT/S2a5"
  job rb_A5C1 "$RB_ROOT/S2a5c1/cells/s2_F2_A5+C1_seed0.json" scripts/train_rebound_v3_par_v1.py --stage 2 --folds F2 --arms A5+C1 --seed 0 --feeds --n-jobs 23 --out-dir "$RB_ROOT/S2a5c1"
  ;;
status)
  echo "--- running containers"; docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | head
  echo "--- python worker processes on host: $(pgrep -fc 'loky|train_.*par_v1' || true)"
  free -g | head -2; uptime
  for f in logs/stageb/*.log; do [ -e "$f" ] && { echo "--- $f"; tail -n 2 "$f" | cut -c1-160; }; done
  ;;
*) sed -n 2,14p "$0"; exit 2 ;;
esac
