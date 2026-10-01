#!/usr/bin/env bash
# operator 2026-10-01 day session, ON the box: pull + build + parity v9. Usage: setup.sh <SHA>
set -uo pipefail
SHA="${1:?sha}"; ts() { date -u +%H:%M:%SZ; }
cd ~/cbb; mkdir -p logs; . ~/.hf_env; export HF_TOKEN
git fetch -q origin main && git checkout -q "$SHA" && chmod +x scripts/*.sh && echo "[setup] at $(git rev-parse --short HEAD) $(ts)"
echo "[setup] pull start $(ts)"
docker run --rm -e HF_TOKEN -v "$PWD:/w" -w /w python:3.12-slim sh -c \
  "pip install -q huggingface_hub==1.31.0 && python scripts/hf_sync_data.py pull --dirs engine_inputs model_artifacts team_rate_tables raw engine_inputs_v3 --max-attempts 6 && python scripts/hf_sync_data.py pull --dirs results --only 'engine_v0/v3full_S0_s200_o0/**' 'engine_v0/v3full_S0f*_s200_o*/games.parquet' 'engine_v0/v3full_S0f*_s200_o*/run_meta.json' 'engine_v0/v3full_COMB9GCTKD_s200_o0/**' 'engine_v0/d1001D_*/**' 'engine_v0/d1001B_*/**' 'engine_v0/f1c_*/**' 'engine_v0/fr1_F*/**' 'engine_v0/v3full_grade/**' --max-attempts 6" > logs/pull.log 2>&1
echo "[setup] pull rc=$? $(ts)"
docker build -f Dockerfile.cbb -t cbb-sweep . > logs/build.log 2>&1
echo "[setup] build rc=$? $(ts)"
docker run --rm -e HF_TOKEN -v $PWD/out:/out cbb-sweep --tag box1001d_parity --parity only --parity-ref docs/ops/parity_reference_windows_v9.json --parity-input-dir data/processed/models/engine_v3 --workers 96 --push off > logs/parity.log 2>&1
echo "[setup] parity rc=$? $(ts)"; grep -i "digest\|PASS\|FAIL" logs/parity.log | tail -6
if grep -q "parity PASS" logs/parity.log; then touch logs/PARITY_PASS; echo "[setup] PARITY PASS $(ts)"; else echo "[setup] PARITY FAILED: hard stop"; exit 9; fi
touch logs/SETUP_DONE
