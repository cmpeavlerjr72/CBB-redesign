#!/usr/bin/env bash
# Cost clock for the box session (lane J 2026-09-30). Reads the REAL wall clock.
#   scripts/ops_cost_clock_v1.sh <launch-time-ISO-UTC> <usd-per-hour> [cap-hours=2] [cap-usd=20]
# e.g. scripts/ops_cost_clock_v1.sh 2026-09-30T17:05:00Z 3.50
# Prints elapsed, cost so far, and time/money left to the cap. Run it at every checkpoint.
set -euo pipefail
L="$1"; R="$2"; CH="${3:-2}"; CU="${4:-20}"
python3 - "$L" "$R" "$CH" "$CU" <<'PY'
import sys, datetime as d
l = d.datetime.strptime(sys.argv[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=d.timezone.utc)
r, ch, cu = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
now = d.datetime.now(d.timezone.utc)
h = (now - l).total_seconds() / 3600
cost = h * r
print(f"now {now:%H:%M:%SZ}  elapsed {int(h)}h{int(h % 1 * 60):02d}m  cost so far ${cost:.2f} @ ${r}/h")
print(f"to the {ch:g} h cap: {max(ch - h, 0) * 60:.0f} min left;  to the ${cu:g} cap: "
      f"{max(cu - cost, 0) / r * 60:.0f} min left")
if h > ch or cost > cu:
    print("OVER CAP -- push results and terminate NOW")
PY
