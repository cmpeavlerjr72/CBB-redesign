#!/usr/bin/env bash
cd "$(dirname "$0")/.."
timeout 90 bash scripts/ops_op1001d_bx_v1.sh 'cat ~/streamA.out ~/streamB.out ~/streamC.out 2>/dev/null | cut -c1-120; for d in ~/cbb ~/cbb2; do cd $d/logs 2>/dev/null || continue; for f in job_*.out; do [ -s $f ] && echo "$(basename $d) $f: $(grep -aE "^\[chunk\]|^\[done\]|^\[graded\]|FAILED|Traceback|PASS|FAIL" $f | tail -n1 | cut -c1-110)"; done; done; date -u +%H:%M:%SZ'
