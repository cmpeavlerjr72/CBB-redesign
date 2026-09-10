#!/usr/bin/env python
"""
build_team_crosswalk.py -- build data/reference/team_crosswalk.parquet.

D-I-ever teams (2022-2026), one row per ESPN team_id, with:
  espn_team_id, espn_name, espn_names_by_season (JSON: season -> display name,
  captures the 213 team_id rebrands the hoopR audit found), first_season,
  last_season, cbbd_team_id, cbbd_name, kenpom_name, match_method,
  match_confidence (the last two describe the KenPom match only -- the CBBD
  match is a game-join vote, always either present or absent).

See `src/cbb_sim/data/ids.py` module docstring for the full matching
methodology (score-verified game join for CBBD, massy-crosswalk + normalized
name + substring fallback for KenPom).

Anything left unmatched on either side is written to
data/reference/team_crosswalk_unmatched.csv for manual follow-up.

Usage:
    python scripts/build_team_crosswalk.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.data.ids import (  # noqa: E402
    DEFAULT_ESPN_TO_KP_MATCHES,
    build_team_crosswalk,
)
from cbb_sim.data.universe import (  # noqa: E402
    DEFAULT_CBBD_DIR,
    DEFAULT_HOOPR_DIR,
    DEFAULT_SEASONS,
    compute_d1_team_seasons,
    load_schedules,
)

OUT_DEFAULT = Path("data/reference/team_crosswalk.parquet")
UNMATCHED_OUT_DEFAULT = Path("data/reference/team_crosswalk_unmatched.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--hoopr-dir", type=Path, default=DEFAULT_HOOPR_DIR)
    ap.add_argument("--cbbd-dir", type=Path, default=DEFAULT_CBBD_DIR)
    ap.add_argument("--massy-matches", type=Path, default=DEFAULT_ESPN_TO_KP_MATCHES)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--unmatched-out", type=Path, default=UNMATCHED_OUT_DEFAULT)
    args = ap.parse_args()

    print(f"Loading hoopR schedules for seasons {args.seasons} ...")
    sch = load_schedules(args.hoopr_dir, args.seasons)

    d1_team_seasons = compute_d1_team_seasons(sch)
    d1_team_ids = {tid for (_, tid) in d1_team_seasons}
    print(f"D-I-ever team_ids in scope: {len(d1_team_ids)}")

    crosswalk, unmatched, report = build_team_crosswalk(
        sch, d1_team_ids, args.cbbd_dir, massy_matches_path=args.massy_matches
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_parquet(args.out, index=False)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(crosswalk):,} rows to {args.out} ({size_mb:.2f} MB)")

    dup = crosswalk["espn_team_id"].duplicated().sum()
    print(f"Duplicate espn_team_id rows: {dup}")

    args.unmatched_out.parent.mkdir(parents=True, exist_ok=True)
    unmatched.to_csv(args.unmatched_out, index=False)
    print(f"Wrote {len(unmatched):,} unmatched rows to {args.unmatched_out}")

    print()
    print("=== Match rates ===")
    print(f"teams in scope: {report['n_teams']}")
    print(f"CBBD matched: {report['n_cbbd_matched']} / {report['n_teams']} ({report['n_cbbd_matched'] / report['n_teams']:.1%})")
    print(
        f"KenPom matched: {report['n_kenpom_matched']} / {report['n_teams']} "
        f"({report['n_kenpom_matched'] / report['n_teams']:.1%})"
    )
    print("KenPom match method breakdown:", report["kenpom_match_method_counts"])
    print(f"Unmatched on >=1 side: {report['n_unmatched_any']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
