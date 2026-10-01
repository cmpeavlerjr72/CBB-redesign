#!/usr/bin/env bash
# local: fetch box paths (relative to ~/cbb) into the repo root, never overwriting an existing file (tar -k). fetch.sh <path...>
cd "$(dirname "$0")/.."; OPDIR=${OPDIR:-$HOME/op1001d}
bash scripts/ops_op1001d_bx_v1.sh "cd ~/cbb && tar cf - $* 2>/dev/null" | tar xkf - 2>&1 | grep -v 'Not replacing\|Cannot open: File exists' | head
