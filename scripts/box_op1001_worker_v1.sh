#!/usr/bin/env bash
# operator: queue worker. Pops the next line of ~/jobs.txt under a lock and runs it; ~/STOP_NEW stops new pops.
W=$1; ts() { date -u +%H:%M:%SZ; }
until [ -f ~/cbb4/logs/PARITY_PASS ]; do sleep 15; done
until [ -f ~/cbb/logs/ARM_DONE_K2O -o -f ~/cbb/logs/ARM_DONE_L2 ]; do sleep 15; done
[ "$W" = w2 ] && until [ -f ~/cbb/logs/ARM_DONE_K2O -a -f ~/cbb/logs/ARM_DONE_L2 ]; do sleep 15; done
while true; do
  [ -f ~/STOP_NEW ] && { echo "[$W] STOP_NEW seen $(ts)"; exit 0; }
  J=$(flock ~/jobs.lock sh -c 'J=$(head -1 ~/jobs.txt); [ -n "$J" ] && sed -i 1d ~/jobs.txt; echo $J')
  [ -z "$J" ] && { echo "[$W] queue empty $(ts)"; exit 0; }
  echo "[$W] START $J $(ts)"
  bash ~/jobs.sh "$J" > ~/cbb4/logs/job_$J.out 2>&1; rc=$?
  if [ $rc -eq 0 ]; then touch ~/cbb4/logs/JOB_$J.done; echo "[$W] DONE $J $(ts)"; else touch ~/cbb4/logs/JOB_$J.failed; echo "[$W] FAILED $J rc=$rc $(ts)"; fi
done
