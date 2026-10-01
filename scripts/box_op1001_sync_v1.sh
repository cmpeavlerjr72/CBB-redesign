#!/usr/bin/env bash
# operator: push ~/cbb results to HF every 5 min (chunk-level resume after a reclaim)
cd ~/cbb; . ~/.hf_env; export HF_TOKEN
while true; do sleep 300; scripts/box_run_v2.sh scripts/hf_sync_data.py push --dirs results --max-attempts 2 > logs/syncloop.log 2>&1; echo "[sync] rc=$? $(date -u +%H:%M:%SZ)" >> ~/sync.out; done
