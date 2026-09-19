#!/usr/bin/env python
"""
build_possessions_techfix_v3.py -- build the event-layer tables that the
BOUNDED SAME-CLOCK TECHNICAL LOOKAHEAD produces, as VERSIONED SIBLINGS.

    .venv/Scripts/python.exe scripts/build_possessions_techfix_v3.py
    .venv/Scripts/python.exe scripts/build_possessions_techfix_v3.py --seasons 2025
    .venv/Scripts/python.exe scripts/build_possessions_techfix_v3.py --skip-trips

NOTHING EXISTING IS OVERWRITTEN AND NO DEFAULT IS SWITCHED. Other workers hold
long-running reads on `data/processed/possessions{,_v2}` and on
`data/processed/models/free_throw/{trips,attempts}_v1_era.parquet` right now.
This script writes only:

    data/processed/possessions_v3/possessions_{season}.parquet
    data/processed/possessions_v3/chances_{season}.parquet
    data/processed/possessions_v3/build_report.json
    data/processed/models/free_throw/trips_v1_era_techfix.parquet
    data/processed/models/free_throw/attempts_v1_era_techfix.parquet
    results/event_layer_technical/blast_radius.json   (gitignored)

v3 is v2's event layer (the rim-location override is on) PLUS the technical
lookahead; the possession tables' OTHER differences from v1 are therefore v2's,
not this fix's, and the blast radius below is measured v2 -> v3 so that the
lookahead is the only thing that moved.

The defect, the box evidence that the inserted `Lost Ball Turnover` rows are
REAL turnovers (so they are kept, not deleted), and the fix are documented in
`cbb_sim.pbp.possessions` ("TECHNICAL FREE THROWS") and in
`docs/tests/event_layer_technical_lookahead_2026-09-18.md`.

SEAL. Seasons 2022-2025 only by default. 2026 is not built here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402
from cbb_sim.pbp import possessions as poss_mod  # noqa: E402

# fold 2's TEST season first: the PM reads fold 2 before the rest
DEFAULT_SEASONS = [2025, 2024, 2023, 2022]
UNIVERSE = Path("data/processed/games_universe.parquet")
PBP_DIR = Path("data/raw/cbbd/pbp")
FT_DIR = Path("data/processed/models/free_throw")
RESULTS = Path("results/event_layer_technical")


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)


def load_universe() -> pd.DataFrame:
    u = pd.read_parquet(UNIVERSE)
    u = u[u["is_d1_game"] & ~u["pbp_truncated"] & u["cbbd_game_id"].notna()].copy()
    u["cbbd_game_id"] = u["cbbd_game_id"].astype("int64")
    return u


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=DEFAULT_SEASONS)
    ap.add_argument("--skip-poss", action="store_true")
    ap.add_argument("--skip-trips", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_dir = poss_mod.possessions_dir("v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = load_universe()
    report: dict = {"built_at": time.strftime("%Y-%m-%d %H:%M"), "seasons": {}}

    # ------------------------------------------------------------------ v3
    if not args.skip_poss:
        for s in args.seasons:
            ts = time.time()
            poss, chances, diag = poss_mod.segment_season(
                s, universe, pbp_dir=PBP_DIR, tech_lookahead=True)
            poss.to_parquet(out_dir / f"possessions_{s}.parquet", index=False)
            chances.to_parquet(out_dir / f"chances_{s}.parquet", index=False)
            diag["wall_clock_s"] = round(time.time() - ts, 1)
            diag["n_possessions"] = int(len(poss))
            diag["n_chances"] = int(len(chances))
            report["seasons"][str(s)] = diag
            log(f"v3 season {s}: {len(poss):,} possessions, {len(chances):,} chances, "
                f"{diag['n_tech_trips']:,} technical trips "
                f"({diag['n_tech_lookahead_hits']:,} recovered by the lookahead)", t0)
            (out_dir / "build_report.json").write_text(json.dumps(report, indent=1, default=str))

    # ------------------------------------------- free-throw trip siblings
    if not args.skip_trips:
        ts = time.time()
        u_ft = ES.load_universe(UNIVERSE)
        trips, att = FT.build_trips_and_attempts(
            sorted(args.seasons), universe=u_ft, version="v1", pbp_dir=PBP_DIR,
            tech_lookahead=True)
        trips.to_parquet(FT_DIR / "trips_v1_era_techfix.parquet", index=False)
        att.to_parquet(FT_DIR / "attempts_v1_era_techfix.parquet", index=False)
        log(f"techfix trips: {len(trips):,} trips / {len(att):,} attempts "
            f"in {time.time() - ts:.1f}s", t0)
        report["trips_techfix"] = {
            "n_trips": int(len(trips)), "n_attempts": int(len(att)),
            "n_technical_trips": int((trips["foul_class"] == "technical").sum()),
        }
        (out_dir / "build_report.json").write_text(json.dumps(report, indent=1, default=str))

    (RESULTS / "build_report_v3.json").write_text(json.dumps(report, indent=1, default=str))
    log("done", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
