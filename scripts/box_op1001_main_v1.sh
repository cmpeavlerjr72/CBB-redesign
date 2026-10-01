#!/usr/bin/env bash
# operator 2026-10-01 main pipeline ON the box. Resumable: rerun; finished stages leave markers in logs/.
set -uo pipefail
cd ~/cbb; mkdir -p logs; . ~/.hf_env; export HF_TOKEN
ts() { date -u +%H:%M:%SZ; }
SHA="${1:?sha}"
git fetch -q origin main && git checkout -q "$SHA" && chmod +x scripts/*.sh && echo "[main] at $(git rev-parse --short HEAD) $(ts)"
if [ ! -f logs/FLOORS_PULLED ]; then
  docker run --rm -e HF_TOKEN -v "$PWD:/w" -w /w python:3.12-slim sh -c \
   "pip install -q huggingface_hub==1.31.0 && python scripts/hf_sync_data.py pull --dirs results --only 'engine_v0/v3full_S0f*_s200_o*/games.parquet' 'engine_v0/v3full_S0f*_s200_o*/run_meta.json' --max-attempts 6" > logs/pull_floors.log 2>&1 && touch logs/FLOORS_PULLED
  echo "[main] floors pulled rc=$? $(ts)"; ls -d results/engine_v0/v3full_S0*_s200_o* 
fi
if [ ! -f logs/PARITY_PASS ]; then
  echo "[main] parity start $(ts)"
  docker run --rm -e HF_TOKEN -v $PWD/out:/out cbb-sweep --tag box1001_parity --parity only --parity-ref docs/ops/parity_reference_windows_v6.json --workers 96 --push off > logs/parity.log 2>&1
  rc=$?; echo "[main] parity rc=$rc $(ts)"; grep -i "digest\|PASS\|FAIL" logs/parity.log | tail -6
  if grep -q "parity PASS" logs/parity.log; then touch logs/PARITY_PASS; else echo "[main] PARITY FAILED: hard stop"; exit 9; fi
fi
if [ ! -f data/processed/models/engine_v3_S0/docker_mounts.txt ]; then
  echo "[main] build S0 tag $(ts)"
  scripts/box_run_v2.sh scripts/build_engine_inputs_v3_tag_v1.py --tag S0 > logs/build_S0.log 2>&1; echo "[main] S0 tag rc=$? $(ts)"
fi
touch logs/READY_FOR_ARMS; echo "[main] ready $(ts)"
