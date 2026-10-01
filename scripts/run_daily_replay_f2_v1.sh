#!/usr/bin/env bash
# Replay of chain v3 stages 9-12 on three fold-2 dates (clock faked to 14:00Z of the slate date). One core.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1
for d in 2025-02-11 2025-02-25 2025-03-04; do
  date "+%F %T %z start $d"
  .venv/Scripts/python.exe scripts/chain_daily_v3.py --replay-season 2025 --slate-date $d --seeds 16 --root results/daily_replay_f2
  date "+%F %T %z end $d"
done
