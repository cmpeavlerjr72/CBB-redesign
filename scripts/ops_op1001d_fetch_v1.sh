#!/usr/bin/env bash
# local: fetch box paths (relative to the box clone, default ~/cbb) into the repo root, never overwriting an existing file (tar -k). fetch.sh <path...>
cd "$(dirname "$0")/.."
D='$HOME/cbb'; [ -n "${BOXDIR:-}" ] && D="$BOXDIR"
bash scripts/ops_op1001d_bx_v1.sh "cd $D && tar cf - $* 2>/dev/null" | tar xkf - 2>&1 | grep -v 'Not replacing\|Cannot open: File exists' | head
