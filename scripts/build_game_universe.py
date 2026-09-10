#!/usr/bin/env python
"""
build_game_universe.py -- build data/processed/games_universe.parquet.

One row per game, seasons 2022-2026 (hoopR/CBBD season = ending year), joining
hoopR schedules (canonical ESPN/hoopR game_id) with hoopR pbp/player_box
coverage flags and the CBBD game/line join. See
`src/cbb_sim/data/universe.py` module docstring for every design decision
(D-I rule, cbbd_game_id join key, n_periods/OT fallback, pbp_truncated
definition) -- this script is a thin CLI wrapper plus the printed report.

Usage:
    python scripts/build_game_universe.py
    python scripts/build_game_universe.py --seasons 2022 2023 2024 2025 2026 \
        --hoopr-dir data/raw/hoopr --cbbd-dir data/raw/cbbd \
        --out data/processed/games_universe.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as `python scripts/build_game_universe.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.data.universe import DEFAULT_CBBD_DIR, DEFAULT_HOOPR_DIR, DEFAULT_SEASONS, build_universe  # noqa: E402

OUT_DEFAULT = Path("data/processed/games_universe.parquet")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--hoopr-dir", type=Path, default=DEFAULT_HOOPR_DIR)
    ap.add_argument("--cbbd-dir", type=Path, default=DEFAULT_CBBD_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    print(f"Building game universe for seasons {args.seasons} ...")
    out, report = build_universe(seasons=args.seasons, hoopr_dir=args.hoopr_dir, cbbd_dir=args.cbbd_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(out):,} rows to {args.out} ({size_mb:.1f} MB)")

    dup = out["game_id"].duplicated().sum()
    print(f"Duplicate game_id rows: {dup}")

    print()
    print("=== D-I rule ===")
    print(
        f"A (season, team_id) pair is D-I if that team_id appears in that season's hoopR "
        f"schedules with a non-null conference_id in >= 5 games. is_d1_game requires both "
        f"teams D-I. {report['d1_team_seasons_n']:,} distinct (season, team_id) pairs qualify "
        f"as D-I across {args.seasons}; this rule excludes {report['non_d1_games']:,} / "
        f"{report['total_games']:,} games overall from is_d1_game."
    )

    print()
    print("=== Per-season flag counts ===")
    counts = report["per_season_counts"]
    print(counts.to_string())

    print()
    print("=== CBBD<->hoopR join quality (sourceId == game_id) ===")
    print(report["join_quality"].to_string(index=False))

    print()
    print(f"n_periods missing (no linescores and no pbp): {report['n_periods_missing']:,} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
