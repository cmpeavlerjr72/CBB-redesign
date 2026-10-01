#!/usr/bin/env bash
# operator 2026-10-01: pull + build + results pull (incl. resume chunks), on the box
set -uo pipefail
cd ~/cbb; mkdir -p logs; . ~/.hf_env
echo "[setup] pull start $(date -u +%H:%M:%SZ)"
docker run --rm -e HF_TOKEN -v "$PWD:/w" -w /w python:3.12-slim sh -c \
  "pip install -q huggingface_hub==1.31.0 && python scripts/hf_sync_data.py pull --dirs engine_inputs model_artifacts team_rate_tables raw engine_inputs_v3 --max-attempts 6 && python scripts/hf_sync_data.py pull --dirs results --only 'engine_v0/v3full_S0_s200_o0/**' 'engine_v0/v3full_grade/**' 'engine_v0/v3full_S0f*_s200_o*/games.parquet' 'engine_v0/v3full_S0f*_s200_o*/run_meta.json' 'engine_v0/v3full_K2O_*/**' 'engine_v0/v3full_L2_*/**' --max-attempts 6" > logs/pull.log 2>&1
echo "[setup] pull rc=$? $(date -u +%H:%M:%SZ)"
touch logs/FLOORS_PULLED
docker build -f Dockerfile.cbb -t cbb-sweep . > logs/build.log 2>&1
echo "[setup] build rc=$? $(date -u +%H:%M:%SZ)"
touch logs/SETUP_DONE
