#!/usr/bin/env python
"""
build_player_crosswalk_v2.py -- extend the CBBD <-> ESPN player crosswalk to
seasons 2022 and 2023 (Task B, 2026-09-10).

Does NOT overwrite `data/processed/player_crosswalk.parquet` (the 2024-2026
crosswalk built by `scripts/build_player_crosswalk.py`) or its report --
writes a versioned sibling instead:
    data/processed/player_crosswalk_v2.parquet         (seasons 2022-2026)
    data/processed/player_crosswalk_v2_report.json

Roster pull: one `/teams/roster?season=S` call per missing season (2022,
2023 -- 2024/2025/2026 roster dumps already exist in
`data/raw/cbbd/rosters/` and are reused, 0 calls). Written to the same
`data/raw/cbbd/rosters/roster_{season}.parquet` layout the 2024-2026 pulls
use, so `roster_2024/2025/2026.parquet` are untouched.

Usage:
    .venv/Scripts/python.exe scripts/build_player_crosswalk_v2.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.player_ids import build_crosswalk  # noqa: E402
from build_player_crosswalk import pull_rosters, ROSTER_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_player_crosswalk_v2")

SEASONS = [2022, 2023, 2024, 2025, 2026]
OUT_PATH = ROOT / "data" / "processed" / "player_crosswalk_v2.parquet"
REPORT_PATH = ROOT / "data" / "processed" / "player_crosswalk_v2_report.json"


def main() -> None:
    missing = [s for s in SEASONS if not (ROSTER_DIR / f"roster_{s}.parquet").exists()]
    calls_used = 0
    if missing:
        calls_used = pull_rosters(missing)
        log.info("pulled rosters for %s (%d API calls)", missing, calls_used)
    else:
        log.info("all roster dumps already present -- 0 API calls")

    cw, report = build_crosswalk(
        SEASONS,
        roster_dir=ROOT / "data" / "raw" / "cbbd" / "rosters",
        player_box_dir=ROOT / "data" / "raw" / "hoopr" / "player_box",
        poss_dir=ROOT / "data" / "processed" / "possessions",
        team_crosswalk_path=ROOT / "data" / "reference" / "team_crosswalk.parquet",
        pbp_dir=ROOT / "data" / "raw" / "cbbd" / "pbp",
    )
    cw.to_parquet(OUT_PATH, index=False)
    report["api_calls_used_this_run"] = calls_used
    report["seasons_added_this_run"] = missing
    report["note"] = (
        "Versioned sibling of data/processed/player_crosswalk.parquet (which stays "
        "2024-2026, untouched). This file covers 2022-2026."
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    log.info("crosswalk -> %s (%d rows)", OUT_PATH, len(cw))
    print()
    print("=" * 78)
    print("CBBD <-> ESPN PLAYER CROSSWALK v2 (2022-2026) -- MATCH RATE")
    print("=" * 78)
    hdr = f"{'scope':>10} {'rows':>8} {'row match':>10} {'on-floor players':>18} {'poss-weighted':>15}"
    print(hdr)
    for scope in ["overall"] + [str(s) for s in SEASONS]:
        b = report[scope]
        print(f"{scope:>10} {b['rows']:>8} {b['row_match_rate']*100:>9.2f}% "
              f"{b['on_floor_matched']}/{b['on_floor_players']} "
              f"({b['on_floor_player_match_rate']*100:.2f}%) "
              f"{b['possession_weighted_match_rate']*100:>13.4f}%")
    print()
    print("by match method (overall):", report["overall"]["by_method"])
    print(f"API calls used this run: {calls_used}")


if __name__ == "__main__":
    main()
