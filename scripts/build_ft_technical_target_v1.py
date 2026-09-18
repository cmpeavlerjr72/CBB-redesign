"""build_ft_technical_target_v1.py -- L3 free-throw technicals round 1b,
job step 1 (target adjudication).

Builds a THIRD, independent estimate of technical free-throw incidence per
team-game -- box FTA minus pbp-accounted non-technical FTA -- and assembles
it alongside the two round-1 estimates (CBBD's own technical-trip table,
`trips_v1_era.parquet`; hoopR's independent pbp technical-foul events) at
season / team / game grain, so the three can be adjudicated against each
other rather than trusting either pbp vendor's technical-trip count alone.

WHY A NEW HOOPR ATTEMPT COUNT, NOT JUST THE ROUND-1 MOMENT COUNT: round 1
(`docs/models/free_throw/experiments.md` 9.9.1) only counted hoopR
"should-produce-a-trip" MOMENTS (one team technical'd, no offsetting pair) at
the season level -- it never counted how many FREE THROW ATTEMPTS each
moment actually produced, so it cannot be compared to box_fta or to
trips_v1_era's attempt counts at team-game grain. This script fixes that by
reading the free throws hoopR's OWN feed logs immediately after each
single-team technical moment.

METHOD (verified by hand on several games before being trusted at scale --
see the results doc): a technical free-throw trip is a dead ball. Every
`MadeFreeThrow` row belonging to that trip carries the IDENTICAL
`start_period_seconds_remaining` (hoopR) / `secondsRemaining` (CBBD raw) as
the technical-foul row itself; the clock only starts moving again on the
first live-ball event after the trip (confirmed on 401700182 and 401700212,
both vendors -- see the results doc section 1). So: group technical-foul
rows into (game, period, clock) "moments"; a moment with exactly one
distinct offending team_id is a single-team moment that MUST produce a trip
(an offsetting pair, one team each, produces none -- the real NCAA out,
already established in round 1 and re-confirmed here); scan forward from the
last technical row of that moment while the clock is unchanged, counting
`MadeFreeThrow` rows credited to the BENEFICIARY (the team that was NOT
called), and stop the instant the clock moves. This is done independently
for hoopR's raw pbp and for CBBD's OWN raw pbp (the latter as a check on
whether `trips_v1_era` -- the possessions.py pipeline's OUTPUT -- already
captures everything CBBD's raw feed itself contains, or drops some of it
itself; see the results doc for why this matters for the mechanism call).

Inputs (all read-only): data/raw/hoopr/pbp/play_by_play_{season}.parquet,
data/raw/cbbd/pbp/plays_{season}.parquet, data/processed/models/free_throw/
trips_v1_era.parquet, data/processed/truth/team_game_shots_v1.parquet,
data/processed/games_universe.parquet. Seasons 2022-2025 only; 2026 is never
opened (assert_not_sealed).

Outputs (versioned siblings; nothing existing is overwritten):
  data/processed/models/free_throw/technical_target_hoopr_v1.parquet
    one row per (season, game_id, benefiting team_id): hoopr_tech_fta
  data/processed/models/free_throw/technical_target_cbbd_raw_v1.parquet
    one row per (season, game_id, benefiting team_id): cbbd_raw_tech_fta
    (independent same-clock scan of CBBD's OWN raw pbp -- NOT trips_v1_era)
  data/processed/models/free_throw/technical_target_reconciliation_v1.parquet
    one row per (season, game_id, team_id) with all four columns:
    cbbd_trip_table_fta (trips_v1_era, foul_class=='technical', sum trip_len),
    cbbd_raw_scan_fta, hoopr_scan_fta, box_implied_fta (box_fta - ev_fta from
    team_game_shots_v1)

Wall clock and season/team/game-level summary tables are printed and also
written to results/free_throw_technicals/target_adjudication_v1.json
(gitignored, per CLAUDE.md).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]  # 2026 never touched

HOOPR_DIR = Path("data/raw/hoopr/pbp")
CBBD_DIR = Path("data/raw/cbbd/pbp")
OUT_DIR = Path("data/processed/models/free_throw")
RESULTS_DIR = Path("results/free_throw_technicals")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# generic same-clock scan (shared by hoopR and CBBD-raw)
# ---------------------------------------------------------------------------
def scan_technical_attempts(game_id: np.ndarray, order: np.ndarray, period: np.ndarray,
                             clock: np.ndarray, is_tech: np.ndarray, is_ft: np.ndarray,
                             offender: np.ndarray, beneficiary: np.ndarray) -> pd.DataFrame:
    """One row per single-team technical moment: game_id, period, clock,
    offender, beneficiary, n_attempts (FT rows credited to the beneficiary
    while the clock stays frozen at the moment's own clock value).

    All arrays must already be sorted by (game_id, order) ascending. `clock`
    is a single numeric key (period-scoped seconds remaining); moments are
    grouped on (game_id, period, clock)."""
    df = pd.DataFrame({
        "game_id": game_id, "order": order, "period": period, "clock": clock,
        "is_tech": is_tech, "is_ft": is_ft, "offender": offender, "beneficiary": beneficiary,
    })
    df = df.sort_values(["game_id", "order"]).reset_index(drop=True)

    rows = []
    for gid, g in df.groupby("game_id", sort=False):
        g = g.reset_index(drop=True)
        tech_mask = g["is_tech"].to_numpy()
        if not tech_mask.any():
            continue
        clock_arr = g["clock"].to_numpy()
        period_arr = g["period"].to_numpy()
        is_ft_arr = g["is_ft"].to_numpy()
        team_arr = g["beneficiary"].to_numpy()  # beneficiary is only meaningful on tech rows
        off_arr = g["offender"].to_numpy()

        tech_idx = np.nonzero(tech_mask)[0]
        # group consecutive-by-key technical rows into moments on (period, clock)
        moment_key = list(zip(period_arr[tech_idx], clock_arr[tech_idx]))
        seen = {}
        for pos, key in zip(tech_idx, moment_key):
            seen.setdefault(key, []).append(pos)

        n_rows = len(g)
        for (per, clk), positions in seen.items():
            offenders_here = set(off_arr[positions].tolist())
            if len(offenders_here) != 1:
                continue  # offsetting technicals -- no trip, by rule
            beneficiary_id = team_arr[positions[0]]
            # scan from the FIRST technical row of the moment, not the last:
            # a second technical-foul row on the SAME team at the identical
            # clock (a companion/administrative disciplinary tech, e.g. a
            # second T logged right after the first one's FTs) sits AFTER the
            # trip's own free throws often enough that starting from the last
            # row silently walks past them (found by hand-reading game
            # 401364790, 2022: PersonalFoul -> Lost Ball Turnover -> Technical
            # -> FT -> FT -> a SECOND Technical, all at one frozen clock).
            # Starting from the first row's position is safe either way: the
            # inner while-loop below does not break on a technical-foul row,
            # so a genuine second, later trip at a DIFFERENT clock is still
            # its own separate moment and is unaffected.
            first_pos = min(positions)
            j = first_pos + 1
            n_att = 0
            while j < n_rows and period_arr[j] == per and clock_arr[j] == clk:
                if is_ft_arr[j] and team_arr[j] == beneficiary_id:
                    n_att += 1
                j += 1
            rows.append({"game_id": gid, "period": per, "clock": clk,
                         "offender": list(offenders_here)[0], "beneficiary": beneficiary_id,
                         "n_attempts": n_att})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# hoopR
# ---------------------------------------------------------------------------
def build_hoopr_scan(season: int) -> pd.DataFrame:
    # SCHEMA DRIFT (found running this script): the 2022 file uses
    # `start_half_seconds_remaining` (period-scoped, same shape CBBD's own
    # `secondsRemaining` is) where 2023-2025 use `start_period_seconds_remaining`
    # -- both are the seconds remaining IN THE CURRENT PERIOD/HALF, so either
    # is a valid clock key when paired with `period_number`, and this is
    # exactly the same (period, clock) composite key CBBD's own raw scan
    # already uses below.
    avail = pd.read_parquet(HOOPR_DIR / f"play_by_play_{season}.parquet").columns
    clock_col = "start_period_seconds_remaining" if "start_period_seconds_remaining" in avail \
        else "start_half_seconds_remaining"
    cols = ["game_id", "sequence_number", "type_text", "team_id",
            "home_team_id", "away_team_id", "period_number", clock_col]
    h = pd.read_parquet(HOOPR_DIR / f"play_by_play_{season}.parquet", columns=cols)
    h = h.rename(columns={clock_col: "start_period_seconds_remaining"})
    h = h.sort_values(["game_id", "sequence_number"]).reset_index(drop=True)

    is_tech = (h["type_text"] == "Technical Foul").to_numpy()
    is_ft = (h["type_text"] == "MadeFreeThrow").to_numpy()
    team = h["team_id"].to_numpy()
    home = h["home_team_id"].to_numpy()
    away = h["away_team_id"].to_numpy()
    # beneficiary is only well-defined / used on technical rows and FT rows
    # (FT rows: the shooting team IS the beneficiary for a technical trip, so
    # "beneficiary" for an FT row is just its own team_id, matched against
    # the moment's stored beneficiary in the scan)
    beneficiary_col = np.where(team == home, away, np.where(team == away, home, np.nan))
    beneficiary_col = np.where(is_ft, team, beneficiary_col)  # FT rows: own team_id
    clock = h["start_period_seconds_remaining"].to_numpy()

    scanned = scan_technical_attempts(
        h["game_id"].to_numpy(), np.arange(len(h)), h["period_number"].to_numpy(),
        clock, is_tech, is_ft, team, beneficiary_col)
    scanned["season"] = season
    return scanned


# ---------------------------------------------------------------------------
# CBBD raw (independent of trips_v1_era / possessions.py)
# ---------------------------------------------------------------------------
def build_cbbd_raw_scan(season: int, universe: pd.DataFrame) -> pd.DataFrame:
    """Independent same-clock scan of CBBD's OWN raw pbp -- NOT trips_v1_era.

    IMPORTANT (found running this script): `gameId`/`teamId` in
    `data/raw/cbbd/pbp/plays_{season}.parquet` are CBBD's OWN internal ids,
    NOT this project's canonical ESPN `game_id`/`team_id` (trips_v1_era,
    team_game_shots, hoopR and games_universe all key on ESPN ids). Per
    `possessions.py`'s own documented convention ("the machine works on SIDE,
    not on team ids ... CBBD's teamId is [used only via isHomeTeam] ... _emit
    maps side -> the universe's ESPN ids"), team identity is resolved from
    `isHomeTeam` joined against `games_universe`'s own
    `(cbbd_game_id -> game_id, home_team_id, away_team_id)`, never from a raw
    CBBD teamId lookup (no reliable teamId crosswalk exists on disk -- checked
    `data/reference/team_crosswalk.parquet`, built for a different purpose,
    and it does not cover the raw pbp `teamId` values at all)."""
    cols = ["gameId", "id", "playType", "isHomeTeam", "period", "secondsRemaining"]
    p = pd.read_parquet(CBBD_DIR / f"plays_{season}.parquet", columns=cols)
    p = p.drop_duplicates(subset=["gameId", "id"], keep="first")
    p = p.sort_values(["gameId", "id"], kind="stable").reset_index(drop=True)

    u = universe[universe["season"] == season][
        ["cbbd_game_id", "game_id", "home_team_id", "away_team_id"]].drop_duplicates("cbbd_game_id")
    p = p.merge(u, left_on="gameId", right_on="cbbd_game_id", how="inner")

    is_home = p["isHomeTeam"].astype("boolean")
    team = np.where(is_home.to_numpy(dtype=object, na_value=None) == True, p["home_team_id"].to_numpy(),  # noqa: E712
                     np.where(is_home.to_numpy(dtype=object, na_value=None) == False, p["away_team_id"].to_numpy(),  # noqa: E712
                               np.nan))
    opp = np.where(team == p["home_team_id"].to_numpy(), p["away_team_id"].to_numpy(),
                    np.where(team == p["away_team_id"].to_numpy(), p["home_team_id"].to_numpy(), np.nan))

    is_tech = (p["playType"] == "Technical Foul").to_numpy()
    is_ft = (p["playType"] == "MadeFreeThrow").to_numpy()
    beneficiary_col = np.where(is_ft, team, opp)  # tech rows: the opposing side is the beneficiary
    clock = p["secondsRemaining"].to_numpy()

    scanned = scan_technical_attempts(
        p["game_id"].to_numpy(), np.arange(len(p)), p["period"].to_numpy(),
        clock, is_tech, is_ft, team, beneficiary_col)
    scanned["season"] = season
    return scanned


# ---------------------------------------------------------------------------
# CBBD trip table (existing pipeline output, read-only)
# ---------------------------------------------------------------------------
def load_trip_table(seasons: list[int]) -> pd.DataFrame:
    t = pd.read_parquet(OUT_DIR / "trips_v1_era.parquet")
    t = t[t["season"].isin(seasons) & (t["foul_class"] == "technical")]
    g = t.groupby(["season", "game_id", "team_id"])["trip_len"].sum().rename(
        "cbbd_trip_table_fta").reset_index()
    return g


# ---------------------------------------------------------------------------
# box-implied
# ---------------------------------------------------------------------------
def load_box_implied(seasons: list[int]) -> pd.DataFrame:
    s = pd.read_parquet(Path("data/processed/truth/team_game_shots_v1.parquet"))
    s = s[s["season"].isin(seasons) & (s["match_source"] == "event_and_box")].copy()
    s["box_implied_fta"] = s["box_fta"] - s["ev_fta"]
    return s[["season", "game_id", "team_id", "box_fta", "ev_fta", "box_implied_fta"]]


def main():
    t0 = time.time()
    assert_not_sealed(SEASONS, context="ft technical target build")

    universe = pd.read_parquet(Path("data/processed/games_universe.parquet"))

    hoopr_frames, cbbd_raw_frames = [], []
    for s in SEASONS:
        hs = build_hoopr_scan(s)
        hoopr_frames.append(hs)
        log(f"season {s}: hoopR scan done, {len(hs)} single-team moments, "
            f"{hs['n_attempts'].sum()} implied attempts", t0)
        cs = build_cbbd_raw_scan(s, universe)
        cbbd_raw_frames.append(cs)
        log(f"season {s}: CBBD-raw scan done, {len(cs)} single-team moments, "
            f"{cs['n_attempts'].sum()} implied attempts", t0)

    hoopr_all = pd.concat(hoopr_frames, ignore_index=True)
    cbbd_raw_all = pd.concat(cbbd_raw_frames, ignore_index=True)

    hoopr_team = hoopr_all.groupby(["season", "game_id", "beneficiary"])["n_attempts"].sum().rename(
        "hoopr_scan_fta").reset_index().rename(columns={"beneficiary": "team_id"})
    cbbd_raw_team = cbbd_raw_all.groupby(["season", "game_id", "beneficiary"])["n_attempts"].sum().rename(
        "cbbd_raw_scan_fta").reset_index().rename(columns={"beneficiary": "team_id"})

    trip_table = load_trip_table(SEASONS)
    box_implied = load_box_implied(SEASONS)

    hoopr_team["team_id"] = hoopr_team["team_id"].astype("Int64")
    cbbd_raw_team["team_id"] = cbbd_raw_team["team_id"].astype("Int64")
    trip_table["team_id"] = trip_table["team_id"].astype("Int64")
    box_implied["team_id"] = box_implied["team_id"].astype("Int64")

    # universe of team-games = every row in box_implied (the box-vs-event
    # matched population, the same universe round 1's ft_trip_reconciliation
    # used) LEFT-merged with the pbp-derived counts, filled with 0 where a
    # source has nothing to report for that team-game (a true zero, not a
    # missing value, given every source covers essentially the same game set)
    recon = box_implied.merge(trip_table, on=["season", "game_id", "team_id"], how="left")
    recon = recon.merge(hoopr_team, on=["season", "game_id", "team_id"], how="left")
    recon = recon.merge(cbbd_raw_team, on=["season", "game_id", "team_id"], how="left")
    for c in ["cbbd_trip_table_fta", "hoopr_scan_fta", "cbbd_raw_scan_fta"]:
        recon[c] = recon[c].fillna(0).astype(int)

    recon_path = OUT_DIR / "technical_target_reconciliation_v1.parquet"
    recon.to_parquet(recon_path, index=False)
    log(f"written {recon_path} ({len(recon):,} team-game rows)", t0)

    hoopr_team.to_parquet(OUT_DIR / "technical_target_hoopr_v1.parquet", index=False)
    cbbd_raw_team.to_parquet(OUT_DIR / "technical_target_cbbd_raw_v1.parquet", index=False)
    log("written hoopr and cbbd-raw sibling target files", t0)

    # ---- season-level summary ----
    season_summary = recon.groupby("season").agg(
        n_team_games=("game_id", "size"),
        cbbd_trip_table_fta=("cbbd_trip_table_fta", "sum"),
        cbbd_raw_scan_fta=("cbbd_raw_scan_fta", "sum"),
        hoopr_scan_fta=("hoopr_scan_fta", "sum"),
        box_implied_fta=("box_implied_fta", "sum"),
    ).reset_index()
    print("\n=== season-level totals (technical FTA) ===")
    print(season_summary.to_string(index=False))

    # game-level disagreement flags for the sampling step
    recon["disagree_cbbd_vs_hoopr"] = recon["cbbd_trip_table_fta"] - recon["hoopr_scan_fta"]
    recon["disagree_cbbdtrip_vs_cbbdraw"] = recon["cbbd_trip_table_fta"] - recon["cbbd_raw_scan_fta"]
    recon["disagree_box_vs_hoopr"] = recon["box_implied_fta"] - recon["hoopr_scan_fta"]
    recon["disagree_box_vs_cbbdtrip"] = recon["box_implied_fta"] - recon["cbbd_trip_table_fta"]

    out = {
        "season_summary": season_summary.to_dict("records"),
        "n_team_games_any_source_nonzero": int((recon[["cbbd_trip_table_fta", "hoopr_scan_fta",
                                                         "cbbd_raw_scan_fta", "box_implied_fta"]].sum(axis=1) > 0).sum()),
        "cbbdtrip_vs_cbbdraw_exact_match_pct": float(
            (recon["cbbd_trip_table_fta"] == recon["cbbd_raw_scan_fta"]).mean() * 100),
        "wall_clock_s": time.time() - t0,
    }
    with open(RESULTS_DIR / "target_adjudication_v1.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log("written target_adjudication_v1.json", t0)
    print("\ncbbd_trip_table vs cbbd_raw_scan exact match on team-games: "
          f"{out['cbbdtrip_vs_cbbdraw_exact_match_pct']:.2f}%")


if __name__ == "__main__":
    main()
