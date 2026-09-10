#!/usr/bin/env python
"""
build_roster_continuity.py -- data/processed/roster_continuity_2027.parquet.

The CBB analogue of the CFB continuity-prior inputs described in
`docs/postmortem/05_cfb_methodology_extract.md` section 2 (Decision 7):
`week_1_prior = prior_season_rating * continuity_weight + (1-continuity_weight)
* league_mean`, with `continuity_weight` built from returning-starter flags.
CBB's transfer-portal era turns over personnel far faster than CFB's coach-
led scheme continuity (section 5's translation note), so the CBB analogue is
built at the roster level, not a single HC/QB/RB flag: per-team returning
minutes share, returning usage share, incoming-transfer minutes, freshman
count, and a head-coach-change flag -- the raw ingredients a future
continuity-weighted prior would blend, not the blend itself (that is a
modeling decision for `docs/models/`, out of scope here).

WHY THIS IS HARDER THAN IT LOOKS FOR SEASON 2027. CBBD's `/teams/roster`
endpoint returns zero players for every 2026-27 team as of 2026-09-10 (see
`scripts/pull_preseason.py` and docs/tests/preseason_2027_2026-09-10.md) --
there is no season-2027 roster anywhere yet. "Returning" is therefore
inferred, not observed: a player who suited up for a team in the 2025-26
season (hoopR `player_box_2026`) and does NOT appear in the transfer portal
(`year=2026`) as having LEFT that team is treated as returning. This is a
one-sided signal -- it will not catch graduated seniors, early draft
entrants, or medical retirements, none of which show up in the transfer
portal -- so returning shares here are an UPPER BOUND on true returning
shares, tightest for teams with few outgoing seniors and loosest for teams
that just lost a senior-heavy core to graduation/the draft rather than the
portal. This is stated plainly rather than silently baked in, per CLAUDE.md's
"multi-level evidence" / "underpowered cells are labelled underpowered" rule.

MATCHING METHOD (transfer portal players <-> hoopR player_box_2026 athlete
ids), three tiers, in order, exactly mirroring `src/cbb_sim/data/player_ids.py`'s
own fallback ladder for the analogous CBBD-roster-id <-> ESPN-athlete-id
problem:
  1. `data/processed/player_crosswalk.parquet` (season 2026): the existing
     CBBD<->ESPN player crosswalk, keyed on (cbbd_team_id, name_norm) ->
     espn_athlete_id. This is `src/cbb_sim/data/player_ids.py`'s own output,
     already >=99% source-id-verified for season 2026, and the primary path
     the task calls for ("via the CBBD<->ESPN player crosswalk ... if it
     exists").
  2. `data/raw/cbbd/rosters/roster_2026.parquet` (raw CBBD roster dump):
     same key, using the roster row's own `source_id` directly, for any
     (team, name) pair the crosswalk dropped.
  3. Direct normalized-name match against hoopR `player_box_2026` for that
     ESPN team (name + team only -- the portal payload carries no jersey
     number, so this is "else by normalised name + team" rather than
     "+ jersey"). Ambiguous same-team name collisions on either side of any
     tier are dropped rather than guessed at (same policy as
     `player_ids._unique_key_map`).
The realized match rate (by tier) is reported for both the departure side
and the incoming-transfer side -- see `report()` below and the printed
summary / docs/tests/preseason_2027_2026-09-10.md.

Freshman count: `data/raw/preseason/2027/recruiting_players_2026.parquet`
(CBBD's ranked recruiting board), joined on `committedTo.id` == cbbd_team_id.
This is CBBD's ranked/scouted board only and undercounts unranked or
late-committing signees -- a lower bound, the mirror-image caveat of the
upper-bound returning share above.

Usage:
    .venv/Scripts/python.exe scripts/build_roster_continuity.py

Reads (all local, no network):
    data/raw/hoopr/player_box/player_box_2026.parquet
    data/processed/player_crosswalk.parquet
    data/raw/cbbd/rosters/roster_2026.parquet
    data/raw/preseason/2027/portal_2026.parquet
    data/raw/preseason/2027/recruiting_players_2026.parquet
    data/raw/preseason/2027/teams_2027.parquet
    data/reference/team_crosswalk.parquet
    data/reference/coaches.parquet

Writes:
    data/processed/roster_continuity_2027.parquet
    data/processed/roster_continuity_2027_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.data.player_ids import normalize_name

REPO_ROOT = Path(__file__).resolve().parent.parent
PLAYER_BOX_2026 = REPO_ROOT / "data" / "raw" / "hoopr" / "player_box" / "player_box_2026.parquet"
PLAYER_CROSSWALK = REPO_ROOT / "data" / "processed" / "player_crosswalk.parquet"
RAW_ROSTER_2026 = REPO_ROOT / "data" / "raw" / "cbbd" / "rosters" / "roster_2026.parquet"
PORTAL_PATH = REPO_ROOT / "data" / "raw" / "preseason" / "2027" / "portal_2026.parquet"
RECRUITING_PATH = REPO_ROOT / "data" / "raw" / "preseason" / "2027" / "recruiting_players_2026.parquet"
TEAMS_2027_PATH = REPO_ROOT / "data" / "raw" / "preseason" / "2027" / "teams_2027.parquet"
TEAM_CROSSWALK_PATH = REPO_ROOT / "data" / "reference" / "team_crosswalk.parquet"
COACHES_PATH = REPO_ROOT / "data" / "reference" / "coaches.parquet"
OUT_PATH = REPO_ROOT / "data" / "processed" / "roster_continuity_2027.parquet"
REPORT_PATH = REPO_ROOT / "data" / "processed" / "roster_continuity_2027_report.json"

PRIOR_SEASON = 2026
SEASON = 2027


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_key_map(df: pd.DataFrame, keys: list[str], value: str) -> dict:
    """(key tuple) -> value, dropping any key that is not unique -- same
    ambiguity policy as `cbb_sim.data.player_ids._unique_key_map`."""
    sub = df.dropna(subset=keys + [value]).drop_duplicates(subset=keys, keep=False)
    return dict(zip(zip(*[sub[k] for k in keys]), sub[value]))


# --------------------------------------------------------------------------
# Loaders / prep
# --------------------------------------------------------------------------
def load_box_agg(path: Path = PLAYER_BOX_2026) -> pd.DataFrame:
    """One row per (espn_team_id, athlete_id) for season 2026: summed
    minutes and a simple per-possession usage numerator
    (FGA + 0.44*FTA + TOV), the standard usage-rate numerator."""
    cols = ["athlete_id", "athlete_display_name", "team_id", "minutes",
            "field_goals_attempted", "free_throws_attempted", "turnovers"]
    box = pd.read_parquet(path, columns=cols)
    box = box.dropna(subset=["athlete_id", "team_id"]).copy()
    box["athlete_id"] = box["athlete_id"].astype("int64")
    box["team_id"] = box["team_id"].astype("int64")
    box["minutes"] = pd.to_numeric(box["minutes"], errors="coerce").fillna(0.0)
    fga = pd.to_numeric(box["field_goals_attempted"], errors="coerce").fillna(0.0)
    fta = pd.to_numeric(box["free_throws_attempted"], errors="coerce").fillna(0.0)
    tov = pd.to_numeric(box["turnovers"], errors="coerce").fillna(0.0)
    box["usage_num"] = fga + 0.44 * fta + tov

    agg = box.groupby(["team_id", "athlete_id"], as_index=False).agg(
        name=("athlete_display_name", "first"),
        minutes=("minutes", "sum"),
        usage_num=("usage_num", "sum"),
    )
    agg["name_norm"] = agg["name"].map(normalize_name)
    agg = agg.rename(columns={"team_id": "espn_team_id", "athlete_id": "espn_athlete_id"})
    return agg


def build_match_tiers(box_agg: pd.DataFrame, team_crosswalk: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Returns (tierA, tierB, tierC) lookups, each keyed appropriately, all
    mapping to an espn_athlete_id."""
    espn_to_cbbd = dict(zip(team_crosswalk["espn_team_id"], team_crosswalk["cbbd_team_id"]))

    # Tier A: existing CBBD<->ESPN player crosswalk (season 2026).
    tierA: dict = {}
    if PLAYER_CROSSWALK.exists():
        cw = pd.read_parquet(PLAYER_CROSSWALK)
        cw = cw[(cw["season"] == PRIOR_SEASON) & cw["espn_athlete_id"].notna() & cw["cbbd_team_id"].notna()].copy()
        cw["cbbd_team_id"] = cw["cbbd_team_id"].astype("int64")
        cw["espn_athlete_id"] = cw["espn_athlete_id"].astype("int64")
        tierA = _unique_key_map(cw, ["cbbd_team_id", "name_norm"], "espn_athlete_id")

    # Tier B: raw CBBD roster dump, source_id direct (season 2026).
    tierB: dict = {}
    if RAW_ROSTER_2026.exists():
        raw = pd.read_parquet(RAW_ROSTER_2026)
        raw = raw[raw["season"] == PRIOR_SEASON].copy()
        raw["name_norm"] = raw["name"].map(normalize_name)
        raw["source_id_num"] = pd.to_numeric(raw["source_id"], errors="coerce")
        raw = raw.dropna(subset=["source_id_num", "cbbd_team_id"])
        raw["cbbd_team_id"] = raw["cbbd_team_id"].astype("int64")
        raw["source_id_num"] = raw["source_id_num"].astype("int64")
        tierB = _unique_key_map(raw, ["cbbd_team_id", "name_norm"], "source_id_num")

    # Tier C: direct name+team match against hoopR player_box_2026 itself,
    # keyed by (cbbd_team_id, name_norm) so it composes with tiers A/B.
    box2 = box_agg.copy()
    box2["cbbd_team_id"] = box2["espn_team_id"].map(espn_to_cbbd)
    box2 = box2.dropna(subset=["cbbd_team_id"])
    box2["cbbd_team_id"] = box2["cbbd_team_id"].astype("int64")
    tierC = _unique_key_map(box2, ["cbbd_team_id", "name_norm"], "espn_athlete_id")

    return tierA, tierB, tierC


def match_player(cbbd_team_id: int, name_norm: str, tiers: tuple[dict, dict, dict]) -> tuple[object, str]:
    tierA, tierB, tierC = tiers
    key = (cbbd_team_id, name_norm)
    if key in tierA:
        return tierA[key], "crosswalk_source_id"
    if key in tierB:
        return tierB[key], "roster_source_id_raw"
    if key in tierC:
        return tierC[key], "name_team_direct"
    return None, "unmatched"


# --------------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------------
def build(seasons_note: dict | None = None) -> tuple[pd.DataFrame, dict]:
    box_agg = load_box_agg()
    team_crosswalk = pd.read_parquet(TEAM_CROSSWALK_PATH)
    team_crosswalk = team_crosswalk[team_crosswalk["cbbd_team_id"].notna()].copy()
    team_crosswalk["cbbd_team_id"] = team_crosswalk["cbbd_team_id"].astype("int64")
    team_crosswalk["espn_team_id"] = team_crosswalk["espn_team_id"].astype("int64")
    espn_to_cbbd = dict(zip(team_crosswalk["espn_team_id"], team_crosswalk["cbbd_team_id"]))
    cbbd_to_espn = dict(zip(team_crosswalk["cbbd_team_id"], team_crosswalk["espn_team_id"]))

    tiers = build_match_tiers(box_agg, team_crosswalk)

    team_totals = box_agg.groupby("espn_team_id", as_index=False).agg(
        team_minutes_2026=("minutes", "sum"), team_usage_2026=("usage_num", "sum")
    ).set_index("espn_team_id")

    portal = pd.read_parquet(PORTAL_PATH)
    portal["player_name"] = (portal["firstName"].fillna("") + " " + portal["lastName"].fillna("")).str.strip()
    portal["name_norm"] = portal["player_name"].map(normalize_name)
    portal["origin_id"] = pd.to_numeric(portal.get("origin.id"), errors="coerce")
    portal["destination_id"] = pd.to_numeric(portal.get("destination.id"), errors="coerce")

    recruiting = pd.read_parquet(RECRUITING_PATH)
    recruiting["committed_id"] = pd.to_numeric(recruiting.get("committedTo.id"), errors="coerce")
    freshmen_counts = recruiting.dropna(subset=["committed_id"]).groupby("committed_id").size()

    coaches = pd.read_parquet(COACHES_PATH)
    coaches_26 = coaches[coaches["season"] == PRIOR_SEASON].set_index("espn_team_id")
    coaches_27 = coaches[coaches["season"] == SEASON].set_index("espn_team_id")

    teams_2027 = pd.DataFrame()
    if TEAMS_2027_PATH.exists():
        teams_2027 = pd.read_parquet(TEAMS_2027_PATH).set_index("id")

    box_by_team = {tid: g for tid, g in box_agg.groupby("espn_team_id")}

    departure_match_counts: dict[str, int] = {}
    transfer_match_counts: dict[str, int] = {}

    rows = []
    for _, tc_row in team_crosswalk.iterrows():
        espn_team_id = int(tc_row["espn_team_id"])
        cbbd_team_id = int(tc_row["cbbd_team_id"])
        team_box = box_by_team.get(espn_team_id, pd.DataFrame(columns=["espn_athlete_id", "minutes", "usage_num"]))
        team_minutes = float(team_totals["team_minutes_2026"].get(espn_team_id, 0.0))
        team_usage = float(team_totals["team_usage_2026"].get(espn_team_id, 0.0))

        # --- departures via the portal (origin == this team) ---
        departures = portal[portal["origin_id"] == cbbd_team_id]
        departed_ids: set = set()
        for _, prow in departures.iterrows():
            aid, method = match_player(cbbd_team_id, prow["name_norm"], tiers)
            departure_match_counts[method] = departure_match_counts.get(method, 0) + 1
            if aid is not None:
                departed_ids.add(int(aid))

        returning_box = team_box[~team_box["espn_athlete_id"].isin(departed_ids)]
        returning_minutes = float(returning_box["minutes"].sum())
        returning_usage = float(returning_box["usage_num"].sum())

        # --- incoming transfers (destination == this team) ---
        incoming = portal[portal["destination_id"] == cbbd_team_id]
        n_transfers_in = int(len(incoming))
        transfer_minutes_total = 0.0
        n_transfers_matched = 0
        for _, prow in incoming.iterrows():
            origin_cbbd = prow["origin_id"]
            if pd.isna(origin_cbbd):
                transfer_match_counts["no_origin"] = transfer_match_counts.get("no_origin", 0) + 1
                continue
            origin_cbbd = int(origin_cbbd)
            aid, method = match_player(origin_cbbd, prow["name_norm"], tiers)
            transfer_match_counts[method] = transfer_match_counts.get(method, 0) + 1
            if aid is not None:
                origin_espn = cbbd_to_espn.get(origin_cbbd)
                origin_box = box_by_team.get(origin_espn, pd.DataFrame(columns=["espn_athlete_id", "minutes"]))
                m = origin_box.loc[origin_box["espn_athlete_id"] == int(aid), "minutes"]
                if len(m):
                    transfer_minutes_total += float(m.iloc[0])
                    n_transfers_matched += 1

        freshmen_count = int(freshmen_counts.get(cbbd_team_id, 0))

        coach26 = coaches_26.loc[espn_team_id] if espn_team_id in coaches_26.index else None
        coach27 = coaches_27.loc[espn_team_id] if espn_team_id in coaches_27.index else None
        head_coach_2026 = coach26["head_coach"] if coach26 is not None else None
        head_coach_2027 = coach27["head_coach"] if coach27 is not None else None
        coach_id_2026 = coach26["coach_id"] if coach26 is not None else None
        coach_id_2027 = coach27["coach_id"] if coach27 is not None else None
        if coach_id_2026 is None or coach_id_2027 is None:
            coach_change = None
        else:
            coach_change = bool(coach_id_2026 != coach_id_2027)

        conference_2027 = None
        if len(teams_2027) and cbbd_team_id in teams_2027.index:
            conference_2027 = teams_2027.loc[cbbd_team_id, "conference"]

        rows.append({
            "season": SEASON,
            "espn_team_id": espn_team_id,
            "cbbd_team_id": cbbd_team_id,
            "team": tc_row["espn_name"],
            "conference_2027": conference_2027,
            "team_minutes_2026": team_minutes,
            "team_usage_2026": team_usage,
            "n_players_2026": int(len(team_box)),
            "n_departed_via_portal": int(len(departures)),
            "n_departed_matched": len(departed_ids),
            "returning_minutes_share": (returning_minutes / team_minutes) if team_minutes > 0 else np.nan,
            "returning_usage_share": (returning_usage / team_usage) if team_usage > 0 else np.nan,
            "n_transfers_in": n_transfers_in,
            "n_transfers_in_matched": n_transfers_matched,
            "transfers_in_prior_minutes_total": transfer_minutes_total,
            "freshmen_count": freshmen_count,
            "head_coach_2026": head_coach_2026,
            "head_coach_2027": head_coach_2027,
            "coach_change": coach_change,
        })

    out = pd.DataFrame(rows)
    out["built_at"] = now_iso()

    n_dep_total = sum(departure_match_counts.values())
    n_dep_matched = sum(v for k, v in departure_match_counts.items() if k != "unmatched")
    n_tr_total = sum(transfer_match_counts.values())
    n_tr_matched = sum(v for k, v in transfer_match_counts.items() if k not in ("unmatched", "no_origin"))

    report = {
        "generated_at": now_iso(),
        "n_teams": int(len(out)),
        "departure_match_counts": departure_match_counts,
        "departure_match_rate": (n_dep_matched / n_dep_total) if n_dep_total else None,
        "transfer_match_counts": transfer_match_counts,
        "transfer_match_rate": (n_tr_matched / n_tr_total) if n_tr_total else None,
        "returning_minutes_share_describe": out["returning_minutes_share"].describe().to_dict(),
        "returning_usage_share_describe": out["returning_usage_share"].describe().to_dict(),
        "n_coach_changes": int((out["coach_change"] == True).sum()),  # noqa: E712
        "n_coach_change_unknown": int(out["coach_change"].isna().sum()),
    }
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    out, report = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print(f"Wrote {OUT_PATH} ({len(out)} teams)")
    print(f"Departure match rate: {report['departure_match_rate']:.1%} ({report['departure_match_counts']})")
    print(f"Transfer match rate: {report['transfer_match_rate']:.1%} ({report['transfer_match_counts']})")
    print("returning_minutes_share describe:", report["returning_minutes_share_describe"])
    print(f"Coach changes 2026->2027: {report['n_coach_changes']} (unknown: {report['n_coach_change_unknown']})")
    print("\nLowest 20 teams by returning_minutes_share:")
    lowest = out.sort_values("returning_minutes_share", na_position="last").head(20)
    print(lowest[["team", "conference_2027", "returning_minutes_share", "returning_usage_share",
                   "n_transfers_in", "freshmen_count", "coach_change"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
