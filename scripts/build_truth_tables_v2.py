#!/usr/bin/env python
"""
build_truth_tables_v2.py -- v2 of the G3/G4/G8 truth tables, folding
technical free throws back into the FTA/FTM/points reconciliation (L24,
`docs/tests/ft_trip_reconciliation_2026-09-10.md`) and applying the four 2025
`game_finals` third-source resolutions
(`data/processed/truth/diag_finals_resolution_2025.json`).

    .venv/Scripts/python.exe scripts/build_truth_tables_v2.py

Writes, for seasons 2022-2025 (2026 is sealed and is never read):

    data/processed/truth/team_game_shots_v2.parquet
    data/processed/truth/player_game_v2.parquet
    data/processed/truth/game_finals_v2.parquet
    data/processed/truth/build_report_v2.json

v1 (`scripts/build_truth_tables.py`) is untouched and still builds/owns the
`*_v1.parquet` files; this script imports its team/box/player/finals builders
directly (no re-derivation) and adds:

  1. `team_game_shots_v2`: explicit event-layer `fta_tech`/`ftm_tech` columns
     (identified the same way `_handle_technical` + `_collect_trip` in
     `cbb_sim.pbp.possessions` identify a technical free-throw trip -- a
     `technical` row immediately followed, allowing administrative
     OREB/DeadBallReb skips, by a run of FT rows on one team), plus
     tech-adjusted reconciliation columns (`ev_fta_plus_tech`, `diff_fta_tech`,
     `diff_ftm_tech`) and a naive points identity with and without technicals
     (`ev_points_naive`, `ev_points_tech`, `diff_points_notech`,
     `diff_points_tech`) against hoopR `team_box`'s own `team_score`.
  2. `player_game_v2`: the same technical-FT walk keyed on `shot_shooter_id`,
     joined on `cbbd_player_id` for the two seasons with a roster crosswalk
     (2024-2025); null (not zero) for 2022-2023, same convention v1 already
     uses for `ev_*`.
  3. `game_finals_v2`: the four 2025 resolutions from the diagnostic JSON
     applied (hoopR right on three, CBBD on 401722537 -- hoopR carries the
     side-flip there), a `finals_source` column recording which source is
     authoritative per game, and every other row byte-identical to v1.

Nothing here is a bake-off decision (`CLAUDE.md`'s bake-off rule is about
model/feature choices); this is a second-source reconciliation refinement of
existing truth data, same footing as v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = str(Path(__file__).resolve().parents[1] / "src")
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import build_truth_tables as v1  # noqa: E402
from cbb_sim.pbp.events import INERT_CLASSES, classify_frame, load_plays  # noqa: E402
from cbb_sim.pbp.possessions import _fix_flipped_sides  # noqa: E402

SEASONS = v1.SEASONS
SEALED_SEASON = v1.SEALED_SEASON
OUT_DIR = v1.OUT_DIR
HOOPR_DIR = v1.HOOPR_DIR
CROSSWALK_SEASONS = v1.CROSSWALK_SEASONS
RESOLUTIONS_PATH = Path("data/processed/truth/diag_finals_resolution_2025.json")


def _diff_buckets(diff: pd.Series) -> dict:
    return v1._diff_buckets(diff)


# ---------------------------------------------------------------------------
# Technical free-throw event walk (mirrors possessions.py `_handle_technical`
# + `_collect_trip` exactly, adding `shot_shooter_id` which the possession
# state machine itself never keeps -- FT-2 does not model technical-FT
# shooters, `docs/models/free_throw/model.md` section 9).
# ---------------------------------------------------------------------------
def _prepare_events_tech(plays: pd.DataFrame) -> dict:
    cls = classify_frame(plays)
    keep = ~cls.isin(list(INERT_CLASSES))
    p = plays.loc[keep.to_numpy()].reset_index(drop=True)
    c = cls.loc[keep.to_numpy()].reset_index(drop=True)

    is_home = p["isHomeTeam"]
    if is_home.dtype == object:
        is_home = is_home.map({True: True, False: False})
    is_home = is_home.astype("boolean")
    has_team = pd.to_numeric(p["teamId"], errors="coerce").notna() & is_home.notna()
    side = np.where(is_home.fillna(False).to_numpy(), 0, 1)
    side = _fix_flipped_sides(p, side, has_team.to_numpy())
    team = np.where(has_team.to_numpy(), side, -1).astype("int64")

    shooter = pd.to_numeric(p["shot_shooter_id"], errors="coerce").to_numpy()

    return {
        "cls": c.to_numpy(dtype=object),
        "team": team,
        "shooter": shooter,
        "game": p["gameId"].to_numpy(),
    }


def _walk_technical_trips(ev: dict) -> pd.DataFrame:
    """One row per technical free-throw ATTEMPT: (cbbd_game_id, team [0=home
    side after flip-fix, 1=away], shooter_id, made). A `technical` row
    immediately followed (allowing an administrative OREB/DeadBallReb skip, the
    same rule `_collect_trip` uses for real trips) by a run of FT_made/
    FT_missed rows on one team -- every FT in that run is a technical
    attempt, exactly as `_handle_technical` -> `_handle_ft_trip(technical=True)`
    treats it."""
    cls, team, shooter, game = ev["cls"], ev["team"], ev["shooter"], ev["game"]
    n = len(cls)
    rows: list[tuple] = []
    i = 0
    while i < n:
        if cls[i] == "technical":
            j = i + 1
            if j < n and cls[j] in ("FT_made", "FT_missed"):
                t = team[j]
                k = j
                while k < n:
                    c = cls[k]
                    if c in ("FT_made", "FT_missed") and team[k] == t:
                        rows.append((game[k], t, shooter[k], c == "FT_made"))
                        k += 1
                        continue
                    if c in ("OREB", "DeadBallReb"):
                        nxt = k + 1
                        if nxt < n and cls[nxt] in ("FT_made", "FT_missed") and team[nxt] == t:
                            k += 1
                            continue
                    break
                i = k
                continue
        i += 1
    return pd.DataFrame(rows, columns=["cbbd_game_id", "team_side", "shooter_id", "made"])


def _technical_fts_event(season: int, universe: pd.DataFrame) -> pd.DataFrame:
    uni_s = universe[(universe["season"] == season) & universe["is_d1_game"] & ~universe["pbp_truncated"]]
    wanted_cbbd = set(pd.to_numeric(uni_s["cbbd_game_id"], errors="coerce").dropna().astype("int64").tolist())
    plays = load_plays(season, game_ids=wanted_cbbd)
    ev = _prepare_events_tech(plays)
    return _walk_technical_trips(ev)


def build_team_tech_fts(season: int, universe: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team_id): fta_tech / ftm_tech."""
    tech = _technical_fts_event(season, universe)
    uni_s = universe[(universe["season"] == season) & universe["is_d1_game"] & ~universe["pbp_truncated"]]
    gmap = uni_s.set_index("cbbd_game_id")[["game_id", "home_team_id", "away_team_id"]]
    if tech.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "fta_tech", "ftm_tech"])
    tech = tech.merge(gmap, left_on="cbbd_game_id", right_index=True, how="left")
    tech = tech[tech["game_id"].notna()].copy()
    tech["game_id"] = tech["game_id"].astype("int64")
    tech["team_id"] = np.where(tech["team_side"] == 0, tech["home_team_id"], tech["away_team_id"]).astype("int64")
    agg = tech.groupby(["game_id", "team_id"]).agg(
        fta_tech=("made", "size"), ftm_tech=("made", "sum")
    ).reset_index()
    agg["fta_tech"] = agg["fta_tech"].astype("int64")
    agg["ftm_tech"] = agg["ftm_tech"].astype("int64")
    return agg


def build_player_tech_fts(season: int, universe: pd.DataFrame) -> pd.DataFrame | None:
    """Per (game_id, cbbd_player_id): fta_tech / ftm_tech. None when the
    season has no CBBD roster crosswalk (2022-2023) -- `shot_shooter_id` IS
    already a `cbbd_player_id`, same convention v1's `build_player_shots_event`
    uses, so no crosswalk lookup is needed here beyond the season gate."""
    if season not in CROSSWALK_SEASONS:
        return None
    tech = _technical_fts_event(season, universe)
    if tech.empty:
        return pd.DataFrame(columns=["game_id", "cbbd_player_id", "fta_tech", "ftm_tech"])
    uni_s = universe[(universe["season"] == season) & universe["is_d1_game"] & ~universe["pbp_truncated"]]
    gmap = uni_s.set_index("cbbd_game_id")["game_id"]
    tech = tech[tech["shooter_id"].notna()].copy()
    tech["game_id"] = tech["cbbd_game_id"].map(gmap)
    tech = tech[tech["game_id"].notna()].copy()
    tech["game_id"] = tech["game_id"].astype("int64")
    tech["cbbd_player_id"] = tech["shooter_id"].astype("int64")
    agg = tech.groupby(["game_id", "cbbd_player_id"]).agg(
        fta_tech=("made", "size"), ftm_tech=("made", "sum")
    ).reset_index()
    agg["fta_tech"] = agg["fta_tech"].astype("int64")
    agg["ftm_tech"] = agg["ftm_tech"].astype("int64")
    return agg


# ---------------------------------------------------------------------------
# 1. team_game_shots_v2
# ---------------------------------------------------------------------------
def _load_box_points(season: int) -> pd.DataFrame:
    path = HOOPR_DIR / "team_box" / f"team_box_{season}.parquet"
    tb = pd.read_parquet(path, columns=["game_id", "team_id", "team_score"])
    tb["game_id"] = pd.to_numeric(tb["game_id"], errors="coerce").astype("int64")
    tb["team_id"] = pd.to_numeric(tb["team_id"], errors="coerce").astype("int64")
    tb["box_points"] = pd.to_numeric(tb["team_score"], errors="coerce")
    return tb[["game_id", "team_id", "box_points"]]


def build_team_game_shots_v2(universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames = []
    for season in SEASONS:
        uni_s = universe[(universe["season"] == season) & universe["is_d1_game"] & ~universe["pbp_truncated"]]
        game_ids = set(uni_s["game_id"].tolist())

        ev = v1.build_team_game_event(universe, season)
        box = v1.build_team_game_box(season)
        box = box[box["game_id"].isin(game_ids)].copy()
        m = ev.merge(box, on=["game_id", "team_id"], how="outer", indicator=True)
        m["season"] = season
        m["match_source"] = m["_merge"].map({"both": "event_and_box", "left_only": "event_only", "right_only": "box_only"})
        m = m.drop(columns=["_merge"])

        home_of = m["game_id"].map(uni_s.set_index("game_id")["home_team_id"])
        away_of = m["game_id"].map(uni_s.set_index("game_id")["away_team_id"])
        m["opp_team_id"] = np.where(m["team_id"] == home_of, away_of, home_of)

        for stat in ("fga", "fgm", "fga3", "fgm3", "fta", "ftm"):
            ev_col = "ev_fga_3" if stat == "fga3" else ("ev_fgm_3" if stat == "fgm3" else f"ev_{stat}")
            box_col = f"box_{stat}"
            m[f"diff_{stat}"] = m[ev_col] - m[box_col]

        m["rim_share_ev"] = m["ev_fga_rim"] / m["ev_fga"].replace(0, np.nan)
        m["jump2_share_ev"] = m["ev_fga_jump2"] / m["ev_fga"].replace(0, np.nan)
        m["three_share_ev"] = m["ev_fga_3"] / m["ev_fga"].replace(0, np.nan)
        m["ft_rate_ev"] = m["ev_fta"] / m["ev_fga"].replace(0, np.nan)
        m["three_share_box"] = m["box_fga3"] / m["box_fga"].replace(0, np.nan)
        m["ft_rate_box"] = m["box_fta"] / m["box_fga"].replace(0, np.nan)

        disagree_cols = [f"diff_{s}" for s in ("fga", "fgm", "fga3", "fgm3", "fta", "ftm")]
        m["any_disagreement"] = (m[disagree_cols].abs() > 0).any(axis=1)

        # --- NEW: technical FTA/FTM from the event layer -----------------
        tech = build_team_tech_fts(season, universe)
        m = m.merge(tech, on=["game_id", "team_id"], how="left")
        m["fta_tech"] = m["fta_tech"].fillna(0).astype("int64")
        m["ftm_tech"] = m["ftm_tech"].fillna(0).astype("int64")
        m["ev_fta_plus_tech"] = m["ev_fta"] + m["fta_tech"]
        m["ev_ftm_plus_tech"] = m["ev_ftm"] + m["ftm_tech"]
        m["diff_fta_tech"] = m["ev_fta_plus_tech"] - m["box_fta"]
        m["diff_ftm_tech"] = m["ev_ftm_plus_tech"] - m["box_ftm"]

        # --- NEW: naive points identity, with/without technicals ---------
        bp = _load_box_points(season)
        m = m.merge(bp, on=["game_id", "team_id"], how="left")
        m["ev_fgm2"] = m["ev_fgm"] - m["ev_fgm_3"]
        m["ev_points_naive"] = 2 * m["ev_fgm2"] + 3 * m["ev_fgm_3"] + m["ev_ftm"]
        m["ev_points_tech"] = m["ev_points_naive"] + m["ftm_tech"]
        m["diff_points_notech"] = m["ev_points_naive"] - m["box_points"]
        m["diff_points_tech"] = m["ev_points_tech"] - m["box_points"]

        frames.append(m)

    out = pd.concat(frames, ignore_index=True)

    recon: dict = {}
    for season in SEASONS:
        s = out[out["season"] == season]
        both = s[s["match_source"] == "event_and_box"]
        both_pts = both[both["box_points"].notna()]
        n_matched = int(len(both))
        n_disagree_raw = int((both["diff_fta"] != 0).sum())
        n_disagree_tech = int((both["diff_fta_tech"] != 0).sum())
        resid = both[both["diff_fta_tech"] != 0]
        tech_present_not_exact = int((resid["fta_tech"] > 0).sum())
        tech_absent_unexplained = int((resid["fta_tech"] == 0).sum())
        recon[str(season)] = {
            "n_team_games": int(len(s)),
            "n_event_and_box": n_matched,
            "n_event_only": int((s["match_source"] == "event_only").sum()),
            "n_box_only": int((s["match_source"] == "box_only").sum()),
            "reconciliation": {stat: _diff_buckets(both[f"diff_{stat}"]) for stat in
                                ("fga", "fgm", "fga3", "fgm3", "fta", "ftm")},
            "reconciliation_fta_tech_adjusted": _diff_buckets(both["diff_fta_tech"]),
            "reconciliation_ftm_tech_adjusted": _diff_buckets(both["diff_ftm_tech"]),
            "fta_exact_match_rate_raw": (1.0 - n_disagree_raw / n_matched) if n_matched else float("nan"),
            "fta_exact_match_rate_tech_adjusted": (1.0 - n_disagree_tech / n_matched) if n_matched else float("nan"),
            "residual_disagreement_classes": {
                "n_residual_after_tech_net": n_disagree_tech,
                "tech_present_not_exact": tech_present_not_exact,
                "tech_absent_unexplained": tech_absent_unexplained,
                "note": ("Coarse 2-way split of what remains after netting fta_tech: "
                         "'tech_present_not_exact' = a technical FT trip was found for this "
                         "team-game but the count still disagrees (multi-cause trips, feed "
                         "gaps landing in the same team-game); 'tech_absent_unexplained' = no "
                         "technical FT trip found at all, so the disagreement has some other "
                         "cause (front-end one-and-one orphans, end-of-period trips, blank-"
                         "shooter FT rows, or genuine feed gaps -- see "
                         "docs/tests/ft_trip_reconciliation_2026-09-10.md section 2 for the "
                         "finer 5-way waterfall this is a coarser cut of)."),
            },
            "points_identity": {
                "n_checked": int(len(both_pts)),
                "n_mismatch_notech": int((both_pts["diff_points_notech"] != 0).sum()),
                "n_mismatch_tech": int((both_pts["diff_points_tech"] != 0).sum()),
                "pct_mismatch_notech": float((both_pts["diff_points_notech"] != 0).mean()) if len(both_pts) else float("nan"),
                "pct_mismatch_tech": float((both_pts["diff_points_tech"] != 0).mean()) if len(both_pts) else float("nan"),
            },
        }
    return out, recon


# ---------------------------------------------------------------------------
# 2. player_game_v2
# ---------------------------------------------------------------------------
def build_player_game_v2(universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pg, pg_extra = v1.build_player_game(universe)
    frames = []
    tech_report: dict = {}
    for season in SEASONS:
        s = pg[pg["season"] == season].copy()
        if season in CROSSWALK_SEASONS:
            tech = build_player_tech_fts(season, universe)
            if tech is not None and len(tech):
                tech = tech.rename(columns={"fta_tech": "_fta_tech", "ftm_tech": "_ftm_tech"})
                s = s.merge(tech, on=["game_id", "cbbd_player_id"], how="left")
                resolved = s["cbbd_player_id"].notna()
                s["fta_tech"] = pd.array([pd.NA] * len(s), dtype="Int64")
                s["ftm_tech"] = pd.array([pd.NA] * len(s), dtype="Int64")
                s.loc[resolved, "fta_tech"] = s.loc[resolved, "_fta_tech"].fillna(0).astype("int64")
                s.loc[resolved, "ftm_tech"] = s.loc[resolved, "_ftm_tech"].fillna(0).astype("int64")
                s = s.drop(columns=["_fta_tech", "_ftm_tech"])
                n_with_tech = int((s["fta_tech"].fillna(0) > 0).sum())
            else:
                s["fta_tech"] = pd.array([pd.NA] * len(s), dtype="Int64")
                s["ftm_tech"] = pd.array([pd.NA] * len(s), dtype="Int64")
                n_with_tech = 0
            tech_report[str(season)] = {
                "crosswalk_available": True,
                "player_games_with_tech_fta_gt_0": n_with_tech,
                "note": "null only where cbbd_player_id failed to resolve; 0 is a real count otherwise",
            }
        else:
            s["fta_tech"] = pd.array([pd.NA] * len(s), dtype="Int64")
            s["ftm_tech"] = pd.array([pd.NA] * len(s), dtype="Int64")
            tech_report[str(season)] = {
                "crosswalk_available": False,
                "note": "no CBBD roster crosswalk this season (2022-2023) -> fta_tech/ftm_tech "
                        "carried as null by construction, same treatment v1 gives ev_fga etc.",
            }
        frames.append(s)
    out = pd.concat(frames, ignore_index=True)
    pg_extra = dict(pg_extra)
    pg_extra["technical_ft_coverage"] = tech_report
    return out, pg_extra


# ---------------------------------------------------------------------------
# 3. game_finals_v2
# ---------------------------------------------------------------------------
def apply_finals_resolutions(gf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    gf = gf.copy()
    home_dtype = gf["home_score"].dtype
    away_dtype = gf["away_score"].dtype
    gf["finals_source"] = "hoopr"
    gf["finals_third_source_checked"] = False
    gf["finals_resolution_note"] = pd.array([None] * len(gf), dtype="object")

    data = json.loads(RESOLUTIONS_PATH.read_text(encoding="utf-8"))
    applied = []
    for g in data["games"]:
        gid = int(g["game_id"])
        winner = str(g["winner"]).strip().lower()
        mask = gf["game_id"] == gid
        n = int(mask.sum())
        if n == 0:
            applied.append({"game_id": gid, "found": False})
            continue
        gf.loc[mask, "finals_third_source_checked"] = True
        gf.loc[mask, "finals_resolution_note"] = g["resolution"]
        if winner == "cbbd":
            gf.loc[mask, "finals_source"] = "cbbd"
            before = (gf.loc[mask, "home_score"].iloc[0], gf.loc[mask, "away_score"].iloc[0])
            gf.loc[mask, "home_score"] = gf.loc[mask, "homePoints"]
            gf.loc[mask, "away_score"] = gf.loc[mask, "awayPoints"]
            applied.append({"game_id": gid, "found": True, "source": "cbbd",
                             "home_away_before": before,
                             "home_away_after": (int(gf.loc[mask, "home_score"].iloc[0]), int(gf.loc[mask, "away_score"].iloc[0]))})
        else:
            gf.loc[mask, "finals_source"] = "hoopr"
            applied.append({"game_id": gid, "found": True, "source": "hoopr", "note": "already correct, no numeric change"})

    # `.loc` setitem against float `homePoints`/`awayPoints` upcasts the whole
    # column to float64; restore the original int dtype so every other row's
    # column dtype (not just value) stays byte-identical to v1.
    gf["home_score"] = gf["home_score"].astype(home_dtype)
    gf["away_score"] = gf["away_score"].astype(away_dtype)

    report = {"resolutions_file": str(RESOLUTIONS_PATH), "applied": applied}
    return gf, report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = v1.load_universe()

    print("Building team_game_shots_v2 ...")
    tgs, tgs_recon = build_team_game_shots_v2(universe)
    tgs_path = out_dir / "team_game_shots_v2.parquet"
    tgs.to_parquet(tgs_path, index=False)
    print(f"  wrote {tgs_path} ({len(tgs)} rows, {tgs_path.stat().st_size / 1e6:.2f} MB)")

    print("Building player_game_v2 ...")
    pg, pg_recon = build_player_game_v2(universe)
    pg_path = out_dir / "player_game_v2.parquet"
    pg.to_parquet(pg_path, index=False)
    print(f"  wrote {pg_path} ({len(pg)} rows, {pg_path.stat().st_size / 1e6:.2f} MB)")

    print("Building game_finals_v2 ...")
    gf_v1, gf_recon = v1.build_game_finals(universe)
    gf, finals_report = apply_finals_resolutions(gf_v1)
    gf_path = out_dir / "game_finals_v2.parquet"
    gf.to_parquet(gf_path, index=False)
    print(f"  wrote {gf_path} ({len(gf)} rows, {gf_path.stat().st_size / 1e6:.2f} MB)")

    report = {
        "seasons": list(SEASONS),
        "team_game_shots_v2": tgs_recon,
        "player_game_v2": pg_recon,
        "game_finals_v2": gf_recon,
        "game_finals_v2_resolutions_applied": finals_report,
    }
    report_path = out_dir / "build_report_v2.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
