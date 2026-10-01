#!/usr/bin/env bash
# chain_lanee_lg2_v3_v1.sh -- lane E 2026-09-30: late-game round 2 re-read on v3 inputs, verified sample, default (verified) truth.
# 2 workers (core cap 2). Skips a tag whose games.parquet exists. All ENGINE_* new flags unset.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/processed/models/engine_v3
run() { # tag latearg offset
  if [ -f "results/engine_v0/$1/games.parquet" ]; then echo "$1 exists"; return; fi
  echo "START $1 $(date '+%F %T %z') commit $(git rev-parse --short HEAD)"
  .venv/Scripts/python.exe scripts/run_late_game_r2_closed_loop.py --late-game "$2" --seeds 25 --seed-offset "$3" \
     --workers 2 --input-dir $IN --tag "$1" > "results/laneE_logs/$1.log" 2>&1
  echo "END   $1 $(date '+%F %T %z') rc=$?"
}
run e3_lg2_R_s25 off 0
run e3_lg2_W_D_s25 clk_D 0
run e3_lg2_W_C2_s25 clk_C2 0
run e3_lg2_E_BL3_s25 ev_BL3 0
run e3_lg2_E_L0S0_s25 ev_L0S0 0
run e3_lg2_W_C2_E_BL3_s25 clk_C2+ev_BL3 0
run e3_lg2_W_D_E_BL3_s25 clk_D+ev_BL3 0
run e3_lg2_Rfloor_s25 off 1000
