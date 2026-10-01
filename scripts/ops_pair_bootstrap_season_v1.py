"""ops_pair_bootstrap_season_v1.py -- the operator's Decision-12 tool (`ops_pair_bootstrap_v1.py`, NOT edited) for a
season other than 2025 (lane D, 2026-10-01; fold-1 read = season 2024). The tool grades on
`gates.build_grading_frame(g, 2025)` with the season as a literal; this wrapper re-points that one call to --season
and passes every other argument through unchanged.

    CBB_TRUTH=verified_v1 python scripts/ops_pair_bootstrap_season_v1.py --season 2024 -- --ref <tag> --floors <tags> \
        --arms <tags> --out-json <p> --out-md <p>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
argv = sys.argv[1:]
if "--season" not in argv or "--" not in argv:
    raise SystemExit("usage: ops_pair_bootstrap_season_v1.py --season YYYY -- <ops_pair_bootstrap_v1 args>")
SEASON = int(argv[argv.index("--season") + 1])
rest = argv[argv.index("--") + 1:]

import ops_pair_bootstrap_v1 as T  # noqa: E402

_orig = T.G.build_grading_frame
T.G.build_grading_frame = lambda g, season: _orig(g, SEASON)
sys.argv = [sys.argv[0], *rest]
if __name__ == "__main__":
    raise SystemExit(T.main())
