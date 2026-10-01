#!/usr/bin/env bash
# operator 2026-10-01: one stream of full reads ON the box: op_stream.sh <WORKERS> ARM... ; after each arm: grade, bootstrap, push.
set -uo pipefail
cd ~/cbb; . ~/.hf_env; export HF_TOKEN
W="$1"; shift
ts() { date -u +%H:%M:%SZ; }
FL=v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000
for ARM in "$@"; do
  RT="v3full_${ARM}_s200_o0"
  if [ ! -f "results/engine_v0/$RT/games.parquet" ]; then
    echo "[stream] $ARM run start $(ts)"
    bash scripts/box_fullread_v2.sh run "$ARM" "$W" 0 200 25 > "logs/fullread_${ARM}.log" 2>&1 || { echo "[stream] $ARM run FAILED rc=$? $(ts)"; continue; }
    echo "[stream] $ARM run end $(ts)"
  fi
  if [ ! -f "results/engine_v0/v3full_grade/${RT}__verified.md" ]; then
    bash scripts/box_fullread_v2.sh grade "$RT" > "logs/grade_${ARM}.log" 2>&1; echo "[stream] $ARM graded rc=$? $(ts)"
  fi
  CBB_TRUTH=verified_v1 scripts/box_run_v2.sh scripts/ops_pair_bootstrap_v1.py --ref v3full_S0_s200_o0 --floors $FL --arms "$RT" --labels "$ARM" \
     --out-json "results/engine_v0/v3full_grade/${RT}__boot.json" --out-md "results/engine_v0/v3full_grade/${RT}__boot.md" > "logs/boot_${ARM}.log" 2>&1
  echo "[stream] $ARM boot rc=$? $(ts)"
  touch "logs/ARM_DONE_${ARM}"
  scripts/box_run_v2.sh scripts/hf_sync_data.py push --dirs results --max-attempts 4 > "logs/push_${ARM}.log" 2>&1; echo "[stream] $ARM pushed rc=$? $(ts)"
done
echo "[stream] all done $(ts)"
