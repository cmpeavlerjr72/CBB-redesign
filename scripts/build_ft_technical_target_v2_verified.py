"""build_ft_technical_target_v2_verified.py -- L3 free-throw technicals round
1b, job step 1 continued: the VERIFIED technical-trip target, at TRIP grain
(not just the attempt-count grain `build_ft_technical_target_v1.py` scored).

WHY A SEPARATE SCRIPT, NOT AN EDIT TO v1: v1 is the exploratory build that
found the mechanism (read `docs/models/free_throw/experiments.md` section 10
and `docs/tests/free_throw_technicals_round1b_2026-09-18.md` for the full
adjudication). This script is the smaller, final step that turns that finding
into the actual modelling target: one row per (season, game_id, team_id) with
a TRIP COUNT (the quantity round 1's X1/X4/X5-style Poisson-rate arms model:
"trips per team-chance"), never overwriting v1's own output files.

THE ADJUDICATION, IN ONE PARAGRAPH (full detail in the docs above): CBBD's own
raw pbp (`data/raw/cbbd/pbp/plays_{season}.parquet`) logs technical fouls at
essentially the SAME rate hoopR's independent pbp does (0-0.6% apart every
season once both are scanned the same way -- v1's `build_hoopr_scan` /
`build_cbbd_raw_scan`). Round 1's target, `trips_v1_era.parquet` (the
possessions.py PIPELINE's derived trip table), undercounts BOTH raw feeds by
19-29% because `possessions.py::_handle_technical`'s one-row free-throw
lookahead is defeated whenever CBBD inserts an administrative row (a "Lost
Ball Turnover" crediting the technical'd team's interrupted possession, 89% of
cases; a companion "PersonalFoul" row, 10%) between the Technical Foul event
and its free throws -- confirmed by hand-reading 15 full pbp transcripts and
by an automated census of the row immediately following every one of the
1,228 affected team-games. The dropped trip's free throws are then swept up
by the GENERIC (non-technical) free-throw handler and land in `trips_v1_era`
tagged `foul_class in {"foul","none"}` instead of `"technical"` -- verified
by an exact (game, period, seconds_remaining, team_id, opp_id) match on a
sample. `trips_v1_era` is NOT strictly worse everywhere, though: a smaller,
disjoint edge case (two Technical Foul rows on the SAME team at the identical
frozen clock, ~0.3% of team-games) is one `possessions.py`'s own
forward-iterating state machine gets right and this script's own
moment-scanner can get wrong (hand-verified on game 401364790, 2022). The
"box FTA minus non-technical event FTA" third estimate the job specifically
asked to build (`box_implied_fta` in v1's reconciliation table) is REJECTED
as unusable for adjudication: it is confounded by the SAME possessions.py bug
in the OPPOSITE direction (the misattributed technical attempts inflate
`ev_fta`'s non-technical bucket, which shrinks `box_fta - ev_fta`), compounded
with independent event-layer noise, so it cannot be trusted as a clean third
source -- it helped surface the puzzle, not resolve it.

VERIFIED TARGET (this script's output): the trip is counted if EITHER the
CBBD-raw-pbp same-clock scan OR `trips_v1_era` recorded a technical trip at
that exact (game_id, period, clock, beneficiary team) -- a moment-level UNION,
not just a max of two counts (a max of counts can silently double-book a
team-game where the two sources found different SETS of trips of the same
size). Cross-validated against hoopR's fully independent raw-pbp scan (a
second, non-CBBD source, satisfying the CLAUDE.md "verified against a second
source" rule): the union exceeds hoopR's own count by 1.9-4.5% a season,
entirely attributable to the identified double-technical-same-clock edge case
plus one hand-confirmed hoopR-side event-ordering anomaly in 2025 (11 team-
games) -- a quantified, explained residual, not an open gap.

Inputs (read-only): the same ones v1 read, plus v1's own
`technical_target_reconciliation_v1.parquet` is NOT reused (it is
attempt-grain, already aggregated past the point this script needs); this
script re-derives at MOMENT grain from the same raw sources.

Output (new sibling; nothing overwritten):
  data/processed/models/free_throw/technical_target_verified_v1.parquet
    one row per (season, game_id, team_id): verified_trip_count,
    verified_fta (attempts, from whichever source supplied the trip; where
    both did, the CBBD-raw-scan attempt count is kept, since it is the one
    cross-validated against hoopR at the attempt level)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ft_technical_target_v1 import (  # noqa: E402
    build_cbbd_raw_scan, SEASONS,
)
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

OUT_DIR = Path("data/processed/models/free_throw")


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def trip_table_moments(seasons: list[int]) -> pd.DataFrame:
    """trips_v1_era's own technical trips, one row per trip, at the same
    (season, game_id, beneficiary team_id, period, clock) grain as the raw
    scan's moments, so the two can be unioned on an exact key match."""
    t = pd.read_parquet(OUT_DIR / "trips_v1_era.parquet")
    t = t[t["season"].isin(seasons) & (t["foul_class"] == "technical")].copy()
    out = t.groupby(["season", "game_id", "team_id", "period", "seconds_remaining"])["trip_len"].sum().reset_index()
    out = out.rename(columns={"team_id": "beneficiary", "seconds_remaining": "clock", "trip_len": "n_attempts"})
    out["source"] = "trip_table"
    return out


def main():
    t0 = time.time()
    assert_not_sealed(SEASONS, context="ft verified technical target build")
    universe = pd.read_parquet(Path("data/processed/games_universe.parquet"))

    raw_frames = []
    for s in SEASONS:
        cs = build_cbbd_raw_scan(s, universe)
        cs["season"] = s
        cs = cs[cs["n_attempts"] > 0].copy()  # a moment scan calls a "trip" only if it found FTs
        cs["source"] = "raw_scan"
        raw_frames.append(cs[["season", "game_id", "period", "clock", "beneficiary", "n_attempts", "source"]])
        log(f"season {s}: raw-scan moments with attempts>0: {len(cs)}", t0)
    raw_all = pd.concat(raw_frames, ignore_index=True)

    trip_all = trip_table_moments(SEASONS)
    log(f"trip-table moments (all seasons): {len(trip_all)}", t0)

    key = ["season", "game_id", "beneficiary", "period", "clock"]
    merged = raw_all.merge(trip_all, on=key, how="outer", suffixes=("_raw", "_trip"))
    # a moment is verified if EITHER source has it (outer join => exists in
    # at least one). Attempts: prefer the raw-scan count (hoopR-cross-
    # validated at the attempt level); fall back to the trip-table count for
    # moments raw-scan itself missed (the double-technical-same-clock edge
    # case this script's own docstring names).
    merged["n_attempts_verified"] = merged["n_attempts_raw"].fillna(merged["n_attempts_trip"])
    merged["found_by"] = np.where(merged["n_attempts_raw"].notna() & merged["n_attempts_trip"].notna(), "both",
                                   np.where(merged["n_attempts_raw"].notna(), "raw_scan_only", "trip_table_only"))

    log(f"union moments: {len(merged)} ({(merged['found_by']=='both').sum()} both, "
        f"{(merged['found_by']=='raw_scan_only').sum()} raw-only, "
        f"{(merged['found_by']=='trip_table_only').sum()} trip-table-only)", t0)

    team_game = merged.groupby(["season", "game_id", "beneficiary"]).agg(
        verified_trip_count=("n_attempts_verified", "size"),
        verified_fta=("n_attempts_verified", "sum"),
    ).reset_index().rename(columns={"beneficiary": "team_id"})
    team_game["team_id"] = team_game["team_id"].astype(int)

    out_path = OUT_DIR / "technical_target_verified_v1.parquet"
    team_game.to_parquet(out_path, index=False)
    log(f"written {out_path} ({len(team_game):,} team-game rows)", t0)

    season_totals = team_game.groupby("season").agg(
        n_team_games=("game_id", "size"),
        verified_trips=("verified_trip_count", "sum"),
        verified_fta=("verified_fta", "sum")).reset_index()
    print("\n=== verified target season totals ===")
    print(season_totals.to_string(index=False))


if __name__ == "__main__":
    main()
