#!/usr/bin/env bash
# chain_lanee_v3_v2.sh -- lane E 2026-09-30: paired-round re-reads on v3 inputs + verified same-rule sample + default (verified) truth.
# 2 workers (core cap 2), sequential, priority order. Skips a tag whose games.parquet exists. All tonight's new ENGINE_* flags unset.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/processed/models/engine_v3
STOP_AFTER="${STOP_AFTER:-2026-10-02 02:00:00}"   # no new run launched after this wall-clock time (local EDT)
while [ ! -f results/engine_v0/e3_lg2_W_D_s25/games.parquet ]; do sleep 15; done   # the orphaned in-flight run
stamp() { echo "$1 $2 $(date '+%F %T %z') commit $(git rev-parse --short HEAD)"; }
late() { [ "$(date '+%H%M')" -ge 0200 ] && [ "$(date '+%H%M')" -lt 1200 ]; }
lg2() { # tag latearg offset
  if [ -f "results/engine_v0/$1/games.parquet" ]; then echo "$1 exists"; return; fi
  if late; then echo "SKIP $1 (past 02:00)"; return; fi
  stamp START "$1"
  .venv/Scripts/python.exe scripts/run_late_game_r2_closed_loop.py --late-game "$2" --seeds 25 --seed-offset "$3" \
     --workers 2 --input-dir $IN --tag "$1" > "results/laneE_logs/$1.log" 2>&1
  stamp END "$1 rc=$?"
}
fj() { # tag [ENV=VAL]
  if [ -f "results/engine_v0/$1/games.parquet" ]; then echo "$1 exists"; return; fi
  if late; then echo "SKIP $1 (past 02:00)"; return; fi
  stamp START "$1"
  if [ -n "$2" ]; then E="--env $2"; else E=""; fi
  .venv/Scripts/python.exe scripts/run_foul_joint_tap_v2.py --tag "$1" $E --workers 2 --input-dir $IN > "results/laneE_logs/$1.log" 2>&1
  stamp END "$1 rc=$?"
}
sb() { # tag arm
  if [ -f "results/engine_v0/$1/games.parquet" ]; then echo "$1 exists"; return; fi
  if late; then echo "SKIP $1 (past 02:00)"; return; fi
  stamp START "$1"
  .venv/Scripts/python.exe scripts/run_shot_block_closed_loop_v1.py --shot-block "$2" --tag "$1" --seeds 25 --workers 2 --input-dir $IN --no-players > "results/laneE_logs/$1.log" 2>&1
  stamp END "$1 rc=$?"
}
fj e3_fj_R_s25 ""
fj e3_fj_R8a_s25 ENGINE_FOUL_JOINT=R8a
fj e3_fj_R8aS_s25 ENGINE_FOUL_JOINT=R8aS
fj e3_fj_R8bU_s25 ENGINE_FOUL_JOINT=R8bU
fj e3_fj_R8bS_s25 ENGINE_FOUL_JOINT=R8bS
fj e3_fj_CL1_s25 ENGINE_FOUL_JOINT=CL1
lg2 e3_lg2_W_C2_s25 clk_C2 0
lg2 e3_lg2_E_BL3_s25 ev_BL3 0
sb e3_sb_K2_s25 K2
sb e3_sb_K2Onoteam_s25 K2_Ocell_noteam
lg2 e3_lg2_Rfloor_s25 off 1000
fj e3_fj_CL3_s25 ENGINE_FOUL_JOINT=CL3
fj e3_fj_CL4_s25 ENGINE_FOUL_JOINT=CL4
fj e3_fj_F5e_s25 ENGINE_FOUL_ACCRUAL=round6_F5e
lg2 e3_lg2_E_L0S0_s25 ev_L0S0 0
lg2 e3_lg2_W_C2_E_BL3_s25 clk_C2+ev_BL3 0
lg2 e3_lg2_W_D_E_BL3_s25 clk_D+ev_BL3 0
fj e3_fj_F5_s25 ENGINE_FOUL_ACCRUAL=round6_F5
echo CHAIN_DONE $(date '+%F %T %z')
