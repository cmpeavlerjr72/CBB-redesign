#!/usr/bin/env bash
# chain_lanee_rest_v3_v1.sh -- lane E 2026-09-30: remaining paired-round re-reads on v3 inputs + verified sample + default (verified) truth.
# 2 workers (core cap 2), sequential, after the lg2 chain. Skips a tag whose games.parquet exists.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/processed/models/engine_v3
while [ -z "$(ls results/engine_v0/e3_lg2_Rfloor_s25/games.parquet 2>/dev/null)" ] && kill -0 "$1" 2>/dev/null; do sleep 20; done
stamp() { echo "$1 $2 $(date '+%F %T %z') commit $(git rev-parse --short HEAD)"; }
fj() { # tag [ENV=VAL]
  if [ -f "results/engine_v0/$1/games.parquet" ]; then echo "$1 exists"; return; fi
  stamp START "$1"
  if [ -n "$2" ]; then E="--env $2"; else E=""; fi
  .venv/Scripts/python.exe scripts/run_foul_joint_tap_v2.py --tag "$1" $E --workers 2 --input-dir $IN > "results/laneE_logs/$1.log" 2>&1
  stamp END "$1 rc=$?"
}
sb() { # tag arm
  if [ -f "results/engine_v0/$1/games.parquet" ]; then echo "$1 exists"; return; fi
  stamp START "$1"
  .venv/Scripts/python.exe scripts/run_shot_block_closed_loop_v1.py --shot-block "$2" --tag "$1" --seeds 25 --workers 2 --input-dir $IN --no-players > "results/laneE_logs/$1.log" 2>&1
  stamp END "$1 rc=$?"
}
fj e3_fj_R_s25 ""
fj e3_fj_R8a_s25 ENGINE_FOUL_JOINT=R8a
fj e3_fj_R8aS_s25 ENGINE_FOUL_JOINT=R8aS
fj e3_fj_R8bU_s25 ENGINE_FOUL_JOINT=R8bU
fj e3_fj_R8bS_s25 ENGINE_FOUL_JOINT=R8bS
sb e3_sb_K2_s25 K2
sb e3_sb_K2Onoteam_s25 K2_Ocell_noteam
fj e3_fj_CL1_s25 ENGINE_FOUL_JOINT=CL1
fj e3_fj_CL3_s25 ENGINE_FOUL_JOINT=CL3
fj e3_fj_CL4_s25 ENGINE_FOUL_JOINT=CL4
fj e3_fj_F5e_s25 ENGINE_FOUL_ACCRUAL=round6_F5e
fj e3_fj_F5_s25 ENGINE_FOUL_ACCRUAL=round6_F5
