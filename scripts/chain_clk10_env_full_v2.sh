#!/bin/bash
# clock round 10 full-size read, chain v2 (v1 stopped by its owner after it had started the control run; that run, PID given
# as $1, keeps going and is waited for here). Sequential: wait for the control clk10full_ctrl_o0_s200 -> post-process it ->
# E1 5,710 x 200 (seeds 0-199, paired) -> post -> four control floor draws 5,710 x 50 at offsets 1000-4000 (grade_clock_env_full_v1
# amendment) -> post each -> full-size grader. usage: chain_clk10_env_full_v2.sh <control engine pid>
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1
W=${FULL_WORKERS:-16}
CPID=$1
while kill -0 $CPID 2>/dev/null; do sleep 30; done
[ -f results/clock_env10/full/clk10full_ctrl_o0_s200/run_meta.json ] || { echo "control run did not finish"; exit 1; }
echo "control finished"; date
bash scripts/chain_clk10_env_full_post_v1.sh clk10full_ctrl_o0_s200 || exit 2
for spec in "E1 0 200" "ctrl 1000 50" "ctrl 2000 50" "ctrl 3000 50" "ctrl 4000 50"; do
  set -- $spec; LAB=$1; o=$2; S=$3
  TAG=clk10full_${LAB}_o${o}_s${S}
  date
  if [ -f results/clock_env10/full/$TAG/games.parquet ]; then echo "$TAG exists, skipped"; continue; fi
  if [ "$LAB" = "E1" ]; then export ENGINE_CLOCK=v5b_r8K2_envE1_glat_pmean; else unset ENGINE_CLOCK; fi
  .venv/Scripts/python.exe scripts/run_engine.py --fold F2 --season 2025 --seeds $S --seed-offset $o --workers $W \
    --input-dir data/processed/models/engine_v3 --tag $TAG --no-players --trajectory-seeds $((o+50)) \
    --results-dir results/clock_env10/full || exit 3
  unset ENGINE_CLOCK
  echo "$TAG done"
  bash scripts/chain_clk10_env_full_post_v1.sh $TAG || exit 4
done
.venv/Scripts/python.exe scripts/grade_clock_env_full_v1.py > results/clock_env10/logs/grade_full.log 2>&1 || exit 5
date
echo ALL_DONE
