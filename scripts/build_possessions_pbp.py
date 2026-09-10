#!/usr/bin/env python
"""
build_possessions_pbp.py -- T_pbp: possessions per game counted directly from
hoopR play-by-play, as the L2 pace bake-off's alternative target to T_box
(`docs/models/pace/experiments.md` pre-registration, "Two target definitions
compared as a first dimension").

    .venv/Scripts/python.exe scripts/build_possessions_pbp.py
    .venv/Scripts/python.exe scripts/build_possessions_pbp.py --seasons 2022 2023 2024 2025

Writes `data/processed/possessions_pbp.parquet` (one row per game: game_id,
season, home_team_id, away_team_id, poss_home_pbp, poss_away_pbp, poss_pbp,
n_relevant_events, n_technical_trips_suppressed) and
`data/processed/possessions_pbp_compare.json` (T_box vs T_pbp mean/SD/corr per
season and pooled, consumed by `scripts/train_pace_v1.py` for the
experiments.md write-up).

Only seasons 2022-2025 are processed by default -- season 2026 is the sealed
blind test (`cbb_sim.data.seal`) and this script never reads it for anything
that could become a trainable/selection artifact. It CAN be read for a pure
descriptive check the way `scripts/build_gate_reference.py` reads it; that is
deliberately not done here since this file's whole purpose is to feed the L2
bake-off's fold tables.

===========================================================================
THE EXACT POSSESSION-ENDING RULE (T_pbp)
===========================================================================

hoopR's `type_text` vocabulary is not stable across seasons (L6,
`docs/LEARNINGS.md`; `docs/SIM_GUARDRAILS.md` section 4 "Event vocabulary
drift"): 24 distinct values appear across 2022-2026, 19-23 in any one season.
Every one of the 24 is mapped below to an explicit role; `classify()` raises
`ValueError` on any type_text not in the table, per the guardrail's standing
requirement that "every pbp-derived feature must be built from an explicit
mapping table with a test that fails on an unknown type"
(`tests/test_pace.py::test_unknown_pbp_event_type_raises`).

    SHOT    JumpShot, LayUpShot, DunkShot, TipShot, Shot (bare "Shot" is a
            2026-only, 2-row-total type; `shooting_play` confirms it is a
            shot attempt). `scoring_play` is the made/missed flag -- NOT the
            type_text, which does not distinguish makes from misses for any
            shot type.
    FT      MadeFreeThrow. Despite the name this single type covers BOTH
            makes and misses (verified against `text`: "X made Free Throw."
            vs "X missed Free Throw."); `scoring_play` is again the made flag.
    OREB    Offensive Rebound -- the shooting team recovers; not terminal.
    DREB    Defensive Rebound -- the other team recovers; terminal.
    TOV     Lost Ball Turnover -- hoopR's single turnover bucket regardless
            of cause (steal, bad pass, travel, offensive foul, shot-clock
            violation all land here per its own `text`); verified against
            team_box `total_turnovers` (98.6% row-count agreement pooled
            2022-2025).
    PERIOD_END   End Period, End Game -- forces closure of any possession
                 still open when the clock runs out.
    TECH    Technical Foul -- does not itself change or end the flowing
            possession; see the suppression rule below.
    IGNORE  Substitution, PersonalFoul, Steal, OfficialTVTimeOut, Block Shot,
            ShortTimeOut, Dead Ball Rebound, Jumpball, RegularTimeOut,
            Coach's Challenge (Stands/Overturned), Not Available. These carry
            no possession-terminal information by themselves (see "Dead Ball
            Rebound" note below) and are dropped from the event stream before
            the state machine runs, so the machine's "next event" is always
            the next SHOT/FT/OREB/DREB/TOV/PERIOD_END/TECH action.

`Dead Ball Rebound` deserves its own note because it looks like a live
rebound but is not one: spot-checking hoopR `text` shows the event's own
`team_id` is the team that just shot or shot a free throw when the ball goes
out of bounds off the OTHER team (offense retains, e.g. mid-free-throw-trip
administrative resets, or a blocked/airballed shot that stays with the
shooter's team) -- these are non-terminal and dropping the event changes
nothing, the next real event just continues the same team's possession. The
rarer case -- the ball goes out of bounds off the shooting team and is
awarded to the defense -- IS a genuine possession change, but dropping the
event does not lose it: the state machine's mismatch guard (below) detects
that the very next SHOT/FT/TOV event belongs to a different team than the one
still "pending" and closes out the pending team's possession at that point.
So `Dead Ball Rebound` is handled correctly by NOT being specially parsed,
because both of its possible meanings are already covered by other rules.

STATE MACHINE (one forward pass per game, sorted by `game_play_number`, the
one column verified 100% monotonic within `game_id`; hoopR's own
`sequence_number` is NOT reliable for this -- only 58-75% of games are
monotonic on it, `game_play_number` is exact):

  pending = None      # team_id whose possession is open, awaiting resolution
  tech_active = False # suppresses the next same-team FT trip's effect

  for each relevant event (IGNORE types already dropped):
    TECH        -> tech_active = True; continue (no other state change)
    SHOT, made  -> if the very next relevant event is FT by the SAME team,
                   this is an and-one: do nothing now, the FT trip's own
                   last-of-trip event decides. Otherwise TERMINAL for this
                   team now; pending cleared.
    SHOT, miss  -> pending = this team (awaiting rebound)
    FT          -> "last of trip" = the next relevant event is NOT an FT by
                   the same team. Mid-trip FTs (not last) do nothing.
                   If tech_active: this FT (and the rest of its trip) is a
                   technical free throw -- it does not belong to either
                   team's flowing possession, so it changes nothing; clear
                   tech_active once the trip's last FT is reached.
                   Otherwise, at the last FT of a real trip: made -> TERMINAL
                   for the shooter's team, pending cleared. Missed -> pending
                   = the shooter's team (awaiting rebound, exactly like a
                   missed field goal).
    OREB        -> not terminal; pending = this team (still their ball).
    DREB        -> TERMINAL, attributed to `pending` (the team that just
                   missed), NOT to the rebounding team's own team_id --
                   pending is who is actually giving up the possession. Falls
                   back to "whichever of home/away is not this rebound's own
                   team_id" only if pending was never set (a data gap).
    TOV         -> TERMINAL for the team_id on the event (the team that
                   turned it over).
    PERIOD_END  -> if pending is still open (a shot/FT was live when the
                   clock ran out with no recorded rebound), TERMINAL for
                   pending. Otherwise a no-op: the last real possession of
                   the period was already closed by its own terminal event.

  MISMATCH GUARD, applied before processing any SHOT/FT(non-technical)/TOV
  event whose team differs from a currently-open `pending`: credit `pending`
  with a TERMINAL possession end first (an implicit, unlogged change of
  possession -- most commonly the "ball goes out of bounds off the shooting
  team" reading of Dead Ball Rebound above), then process the new event fresh
  from a cleared `pending`. Without this guard a silently-dropped rebound
  event would erase one of the two teams' possessions rather than just
  omitting a stray administrative row.

Each TERMINAL event increments the possession count of the team whose
possession just ended. `poss_pbp` for a game is the mean of the two teams'
counts (mirrors T_box's own "mean over both teams" definition, and the two
counts differ by at most the ordinary game-parity of 1).

KNOWN, DELIBERATELY UNCORRECTED GAPS (all small; see the validation section
of `docs/models/pace/experiments.md` for the measured size):
  - A team that gains the ball on a rebound/turnover but the period ends
    before they attempt a shot, free throw, or commit a turnover is not
    credited with that (zero-duration) trailing possession.
  - `Not Available` (95 rows total, seasons 2022-2023 only, an unclassified
    ESPN placeholder per L6) is dropped as IGNORE; it is too small a share of
    any one game to matter and there is no way to classify it further.
  - The turnover bucket is ~98.6% complete against team_box `total_turnovers`
    pooled 2022-2025; the residual is not separately traced.

EXCLUSIONS. A game enters `possessions_pbp.parquet` only if
`games_universe.is_d1_game & ~pbp_truncated & has_pbp` all hold. `has_pbp`
catches games entirely absent from the season's pbp file (about 1.3-2.0% of
the D-I non-truncated universe per season -- these are NOT the same games
`pbp_truncated` catches, which have a partial feed that stops short of the
final score; these have none at all).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.ratings import own_ratings as orat  # noqa: E402

DEFAULT_HOOPR_DIR = Path("data/raw/hoopr")
DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_OUT = Path("data/processed/possessions_pbp.parquet")
DEFAULT_COMPARE_OUT = Path("data/processed/possessions_pbp_compare.json")
DEFAULT_SEASONS = [2022, 2023, 2024, 2025]

EVENT_ROLE: dict[str, str] = {
    "JumpShot": "SHOT", "LayUpShot": "SHOT", "DunkShot": "SHOT", "TipShot": "SHOT", "Shot": "SHOT",
    "MadeFreeThrow": "FT",
    "Offensive Rebound": "OREB",
    "Defensive Rebound": "DREB",
    "Lost Ball Turnover": "TOV",
    "End Period": "PERIOD_END", "End Game": "PERIOD_END",
    "Technical Foul": "TECH",
    "Substitution": "IGNORE", "PersonalFoul": "IGNORE", "Steal": "IGNORE",
    "OfficialTVTimeOut": "IGNORE", "Block Shot": "IGNORE", "ShortTimeOut": "IGNORE",
    "Dead Ball Rebound": "IGNORE", "Jumpball": "IGNORE", "RegularTimeOut": "IGNORE",
    "Coach's Challenge (Stands)": "IGNORE", "Coach's Challenge (Overturned)": "IGNORE",
    "Not Available": "IGNORE",
}


def classify(type_text: str) -> str:
    try:
        return EVENT_ROLE[type_text]
    except KeyError as e:
        raise ValueError(
            f"unknown hoopR pbp event type_text: {type_text!r}. Add it to "
            f"EVENT_ROLE in {__file__} with an explicit role before proceeding "
            "(SIM_GUARDRAILS.md: every pbp-derived feature needs an explicit "
            "mapping table with a test that fails on an unknown type)."
        ) from e


def _hoopr_pbp_path(hoopr_dir: Path, season: int) -> Path:
    return Path(hoopr_dir) / "pbp" / f"play_by_play_{season}.parquet"


def compute_season_possessions(season: int, hoopr_dir: Path | str = DEFAULT_HOOPR_DIR) -> pd.DataFrame:
    """Run the state machine described in the module docstring over one
    season's pbp file. Returns one row per game_id present in the pbp file
    (callers filter to the D-I/non-truncated/has_pbp universe afterward)."""
    path = _hoopr_pbp_path(Path(hoopr_dir), season)
    pbp = pd.read_parquet(
        path,
        columns=["game_id", "game_play_number", "type_text", "team_id",
                 "home_team_id", "away_team_id", "scoring_play"],
    )
    pbp["game_id"] = pd.to_numeric(pbp["game_id"], errors="coerce").astype("int64")

    unknown = set(pbp["type_text"].unique()) - set(EVENT_ROLE)
    if unknown:
        raise ValueError(f"season {season}: unmapped pbp event types {sorted(unknown)}")
    pbp["role"] = pbp["type_text"].map(EVENT_ROLE)

    rel = pbp[pbp["role"] != "IGNORE"].sort_values(["game_id", "game_play_number"], kind="mergesort")
    n_dropped = int((pbp["role"] == "IGNORE").sum())

    gid = rel["game_id"].to_numpy()
    role = rel["role"].to_numpy()
    team = rel["team_id"].to_numpy(dtype="float64")
    home = rel["home_team_id"].to_numpy(dtype="float64")
    away = rel["away_team_id"].to_numpy(dtype="float64")
    made = rel["scoring_play"].to_numpy(dtype=bool)

    n = len(rel)
    boundaries = np.flatnonzero(np.diff(gid, prepend=gid[0] - 1 if n else 0))
    boundaries = np.append(boundaries, n)

    out_game: list[int] = []
    out_home_id: list[float] = []
    out_away_id: list[float] = []
    out_home_poss: list[int] = []
    out_away_poss: list[int] = []
    out_n_events: list[int] = []
    out_n_mismatch: list[int] = []

    for bi in range(len(boundaries) - 1):
        s, e = int(boundaries[bi]), int(boundaries[bi + 1])
        g = int(gid[s])
        h_id, a_id = home[s], away[s]
        counts = {h_id: 0, a_id: 0}
        pending: float | None = None
        tech_active = False
        n_mismatch = 0

        for i in range(s, e):
            r = role[i]
            tm = team[i]

            if r == "TECH":
                tech_active = True
                continue

            if r == "SHOT":
                if pending is not None and tm != pending:
                    counts[pending] = counts.get(pending, 0) + 1
                    n_mismatch += 1
                    pending = None
                if made[i]:
                    if i + 1 < e and role[i + 1] == "FT" and team[i + 1] == tm:
                        pending = tm  # and-one: the FT trip decides
                    else:
                        counts[tm] = counts.get(tm, 0) + 1
                        pending = None
                else:
                    pending = tm

            elif r == "FT":
                if pending is not None and tm != pending and not tech_active:
                    counts[pending] = counts.get(pending, 0) + 1
                    n_mismatch += 1
                    pending = None
                is_last = not (i + 1 < e and role[i + 1] == "FT" and team[i + 1] == tm)
                if tech_active:
                    if is_last:
                        tech_active = False
                    continue
                if is_last:
                    if made[i]:
                        counts[tm] = counts.get(tm, 0) + 1
                        pending = None
                    else:
                        pending = tm
                else:
                    pending = tm

            elif r == "OREB":
                pending = tm

            elif r == "DREB":
                ender = pending if pending is not None else (a_id if tm == h_id else h_id)
                counts[ender] = counts.get(ender, 0) + 1
                pending = None

            elif r == "TOV":
                if pending is not None and tm != pending:
                    counts[pending] = counts.get(pending, 0) + 1
                    n_mismatch += 1
                counts[tm] = counts.get(tm, 0) + 1
                pending = None

            elif r == "PERIOD_END":
                if pending is not None:
                    counts[pending] = counts.get(pending, 0) + 1
                    pending = None

        out_game.append(g)
        out_home_id.append(h_id)
        out_away_id.append(a_id)
        out_home_poss.append(counts.get(h_id, 0))
        out_away_poss.append(counts.get(a_id, 0))
        out_n_events.append(e - s)
        out_n_mismatch.append(n_mismatch)

    res = pd.DataFrame({
        "game_id": out_game,
        "season": season,
        "home_team_id": pd.array(out_home_id, dtype="Int64"),
        "away_team_id": pd.array(out_away_id, dtype="Int64"),
        "poss_home_pbp": out_home_poss,
        "poss_away_pbp": out_away_poss,
        "n_relevant_events": out_n_events,
        "n_mismatch_events": out_n_mismatch,
    })
    res["poss_pbp"] = (res["poss_home_pbp"] + res["poss_away_pbp"]) / 2.0
    res.attrs["n_ignored_rows"] = n_dropped
    return res


def compare_to_box(pbp_poss: pd.DataFrame, universe: pd.DataFrame, hoopr_dir: Path | str) -> dict:
    """T_box vs T_pbp: mean, SD, corr, per season and pooled, on the games
    both definitions cover."""
    rows = []
    per_season = {}
    for season in sorted(pbp_poss["season"].unique()):
        u = universe[(universe["season"] == season) & universe["is_d1_game"]
                     & ~universe["pbp_truncated"]]
        tg = orat.load_team_games(u, [season], hoopr_dir)
        box = tg.groupby("game_id")["game_poss"].mean().rename("poss_box").reset_index()
        p = pbp_poss[pbp_poss["season"] == season][["game_id", "poss_pbp"]]
        m = box.merge(p, on="game_id", how="inner")
        d = m["poss_pbp"] - m["poss_box"]
        entry = {
            "season": int(season),
            "n_box_universe": int(len(box)),
            "n_matched": int(len(m)),
            "n_missing_from_pbp": int(len(box) - len(m)),
            "box_mean": float(m["poss_box"].mean()), "box_sd": float(m["poss_box"].std()),
            "pbp_mean": float(m["poss_pbp"].mean()), "pbp_sd": float(m["poss_pbp"].std()),
            "corr": float(m["poss_box"].corr(m["poss_pbp"])),
            "mean_diff_pbp_minus_box": float(d.mean()), "sd_diff": float(d.std()),
            "mean_abs_diff": float(d.abs().mean()),
        }
        rows.append(entry)
        per_season[season] = m
    pooled = pd.concat(per_season.values(), ignore_index=True)
    d = pooled["poss_pbp"] - pooled["poss_box"]
    pooled_entry = {
        "season": "ALL", "n_matched": int(len(pooled)),
        "box_mean": float(pooled["poss_box"].mean()), "box_sd": float(pooled["poss_box"].std()),
        "pbp_mean": float(pooled["poss_pbp"].mean()), "pbp_sd": float(pooled["poss_pbp"].std()),
        "corr": float(pooled["poss_box"].corr(pooled["poss_pbp"])),
        "mean_diff_pbp_minus_box": float(d.mean()), "sd_diff": float(d.std()),
        "mean_abs_diff": float(d.abs().mean()),
    }
    return {"by_season": rows, "pooled": pooled_entry}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--hoopr-dir", type=Path, default=DEFAULT_HOOPR_DIR)
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--compare-out", type=Path, default=DEFAULT_COMPARE_OUT)
    args = ap.parse_args()

    assert_not_sealed(args.seasons, context="build_possessions_pbp.py seasons")

    universe = pd.read_parquet(args.universe)
    universe["game_date"] = pd.to_datetime(universe["game_date"])

    frames = []
    for season in args.seasons:
        t0 = time.time()
        r = compute_season_possessions(season, args.hoopr_dir)
        n_ignored = r.attrs.get("n_ignored_rows", 0)
        print(f"season {season}: {len(r):,} games in pbp file, {n_ignored:,} ignored-role rows "
              f"({time.time() - t0:.1f}s)")
        frames.append(r)
    poss = pd.concat(frames, ignore_index=True)

    keep = universe[universe["is_d1_game"] & ~universe["pbp_truncated"] & universe["has_pbp"]
                    & universe["season"].isin(args.seasons)][["game_id", "season"]]
    n_before = len(poss)
    poss = poss.merge(keep, on=["game_id", "season"], how="inner")
    print(f"restricted to D-I, non-truncated, has_pbp: {len(poss):,} / {n_before:,} games")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    poss.to_parquet(args.out, index=False)
    print(f"wrote {len(poss):,} rows to {args.out}")

    print("\nComparing T_box vs T_pbp on the same games ...")
    cmp = compare_to_box(poss, universe, args.hoopr_dir)
    for r in cmp["by_season"]:
        print(f"  {r['season']}: n={r['n_matched']:,} box={r['box_mean']:.3f}({r['box_sd']:.3f}) "
              f"pbp={r['pbp_mean']:.3f}({r['pbp_sd']:.3f}) corr={r['corr']:.4f} "
              f"mean_diff={r['mean_diff_pbp_minus_box']:+.3f} sd_diff={r['sd_diff']:.3f}")
    p = cmp["pooled"]
    print(f"  ALL: n={p['n_matched']:,} box={p['box_mean']:.3f}({p['box_sd']:.3f}) "
          f"pbp={p['pbp_mean']:.3f}({p['pbp_sd']:.3f}) corr={p['corr']:.4f} "
          f"mean_diff={p['mean_diff_pbp_minus_box']:+.3f} sd_diff={p['sd_diff']:.3f}")

    args.compare_out.write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    print(f"wrote {args.compare_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
