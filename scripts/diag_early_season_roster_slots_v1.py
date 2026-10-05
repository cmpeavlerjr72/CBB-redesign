#!/usr/bin/env python
"""
diag_early_season_roster_slots_v1.py -- why does a FT shooter have no named roster slot early in a season? (lane F, 2026-10-01)

For fold 2 (season 2025, inputs `engine_v3`) and fold 1 (season 2024, inputs `engine_v3_f1`), every free-throw attempt of the
design's population (`free_throw/attempts_v1_era.parquet`) is classified by (a) whether the shooter holds a NAMED slot of the
game's roster arrays, and, if not, (b) the CAUSE, from the on-floor tables only (no box score of the game itself):

  named                         shooter id is in the game's roster slots
  team_game_all_anonymous       the team-game has NO named slot at all (no earlier appearance for the team this season: the opener, or a
                                team-game the priors never saw); sub-split by what the shooter's history is (below)
  not_in_pool_*                 the team-game has a pool but the shooter is not in it (no earlier appearance FOR THIS TEAM THIS SEASON), split:
      returner                  appeared for THIS team in the previous season
      transfer_in               no appearance for this team last season, but appeared for ANOTHER team last season
      returning_after_gap       no appearance in the previous season, but earlier ones (two or more seasons back)
      newcomer                  no appearance in any earlier season (freshman, JUCO, international, or a non-D-I history)
  beyond_slot_cap               shooter is in the as-of pool but ranked below the 15 slots
  id_unmapped                   (counted separately) shooter id never appears in any on-floor table

Month = calendar month of the game. "Trips" = distinct (game, trip_id) with at least one such shooter.

    .venv/Scripts/python.exe scripts/diag_early_season_roster_slots_v1.py   -> results/early_season_roster_slots.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from cbb_sim.models import rotation as ROT  # noqa: E402

FOLDS = {2: {"season": 2025, "dir": "data/processed/models/engine_v3", "tag": "F2_2025"},
         1: {"season": 2024, "dir": "data/processed/models/engine_v3_f1", "tag": "F1_2024"}}
MONTH_ORDER = [11, 12, 1, 2, 3, 4]


def pgm_of(season: int) -> pd.DataFrame:
    tp = ROT.load_team_possessions(season)
    p = ROT.player_game_minutes(tp)
    p["game_date"] = pd.to_datetime(p["game_date"])
    return p[p["minutes"] > 0]


def presence_events(season: int) -> pd.DataFrame:
    """(team_id, pid) that took at least one field-goal or free-throw attempt in `season` (cbbd ids)."""
    fg = pd.read_parquet(ROOT / "data/processed/models/fg_make/events_v2_shotshooter.parquet", columns=["season", "off_team_id", "shooter_id"])
    fg = fg[(fg["season"] == season) & (fg["shooter_id"] > 0)].rename(columns={"off_team_id": "team_id", "shooter_id": "pid"})[["team_id", "pid"]]
    ft = pd.read_parquet(ROOT / "data/processed/models/free_throw/attempts_v1_era.parquet", columns=["season", "team_id", "shooter_id"])
    ft = ft[(ft["season"] == season) & ft["shooter_id"].notna()].rename(columns={"shooter_id": "pid"})[["team_id", "pid"]]
    out = pd.concat([fg, ft]).drop_duplicates()
    out["pid"] = out["pid"].astype("int64")
    return out


def run(fold: int) -> dict:
    cfg = FOLDS[fold]
    S = cfg["season"]
    d = ROOT / cfg["dir"]
    games = pd.read_parquet(d / f"games_{cfg['tag']}.parquet")
    ros = np.load(d / f"arrays_{cfg['tag']}.npz")["roster_cbbd"]
    gpos = {int(g): i for i, g in enumerate(games["game_id"])}
    att = pd.read_parquet(ROOT / "data/processed/models/free_throw/attempts_v1_era.parquet")
    att = att[(att["season"] == S) & att["shooter_id"].notna()].copy()
    att["shooter_id"] = att["shooter_id"].astype("int64")
    att = att[att["game_id"].isin(gpos)].copy()
    att["row"] = att["game_id"].map(gpos).to_numpy()
    att["side"] = np.where(att["shooter_is_home"].astype(bool), 0, 1)
    r = ros[att["row"].to_numpy(), att["side"].to_numpy(), :]                    # (n, S)
    att["named"] = (r == att["shooter_id"].to_numpy()[:, None]).any(axis=1)
    tg_named = (ros > 0).any(axis=2)                                             # (G, 2): the team-game has any named slot
    att["team_game_named"] = tg_named[att["row"].to_numpy(), att["side"].to_numpy()]
    att["month"] = pd.to_datetime(att["game_date"]).dt.month

    cur = pgm_of(S)
    prev = presence_events(S - 1)          # on-floor tables do not exist before 2024; shot / FT events do for 2022-2025
    prev2 = presence_events(S - 2) if S - 2 >= 2022 else pd.DataFrame({"team_id": [], "pid": []})
    first_this = cur.groupby(["team_id", "pid"])["game_date"].min().rename("first_this").reset_index()
    cur_games = cur[["team_id", "pid", "game_date", "game_id"]].drop_duplicates()
    prev_team = set(zip(prev["team_id"], prev["pid"]))
    prev_any = set(prev["pid"])
    older_any = set(prev2["pid"]) | (set(presence_events(S - 3)["pid"]) if S - 3 >= 2022 else set())
    seen_any_onfloor = set(cur["pid"]) | prev_any | older_any | set(presence_events(S)["pid"])

    # earlier appearance for THIS team THIS season, strictly before the game's date (the as-of pool rule)
    att = att.merge(first_this, left_on=["team_id", "shooter_id"], right_on=["team_id", "pid"], how="left").drop(columns="pid")
    att["in_pool"] = att["first_this"].notna() & (att["first_this"] < pd.to_datetime(att["game_date"]))

    def cause(row_named, tgn, in_pool, tid, sid):
        if row_named:
            return "named"
        if in_pool:
            return "beyond_slot_cap"
        hist = ("returner" if (tid, sid) in prev_team else "transfer_in" if sid in prev_any
                else "returning_after_gap" if sid in older_any else "newcomer" if sid in seen_any_onfloor else "id_unmapped")
        return (hist if tgn else "team_game_all_anonymous:" + hist)

    # split the all-anonymous team-games: the team's first game of the season, or earlier games exist but none is usable
    # (no row in the on-floor table: pbp-incomplete games, or a team whose on-floor columns are incomplete)
    uni = pd.read_parquet(ROOT / "data/processed/games_universe.parquet", columns=["game_id", "season", "game_date", "home_team_id", "away_team_id", "is_d1_game", "pbp_truncated"])
    uni = uni[(uni["season"] == S) & uni["is_d1_game"] & ~uni["pbp_truncated"]]
    ug = pd.concat([uni[["game_id", "game_date", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
                    uni[["game_id", "game_date", "away_team_id"]].rename(columns={"away_team_id": "team_id"})])
    ug["game_date"] = pd.to_datetime(ug["game_date"])
    ug = ug.sort_values(["team_id", "game_date", "game_id"])
    ug["prior_universe"] = ug.groupby("team_id").cumcount()
    cur_tg = set(zip(cur["team_id"], cur["game_id"]))
    ug["usable"] = [(t, g) in cur_tg for t, g in zip(ug["team_id"], ug["game_id"])]
    ug["prior_usable"] = ug.groupby("team_id")["usable"].cumsum() - ug["usable"].astype(int)
    att = att.merge(ug[["team_id", "game_id", "prior_universe", "prior_usable"]], on=["team_id", "game_id"], how="left")
    att["anon_kind"] = np.where(att["prior_universe"] == 0, "first_game", "earlier_games_none_usable")

    att["cause"] = [cause(a, b, c, t, s) for a, b, c, t, s in
                    zip(att["named"].to_numpy(), att["team_game_named"].to_numpy(), att["in_pool"].to_numpy(),
                        att["team_id"].to_numpy(), att["shooter_id"].to_numpy())]
    out = {"season": S, "inputs": cfg["dir"], "attempts": int(len(att)), "named_share": float(att["named"].mean())}
    t = att.groupby(["month", "cause"]).size().unstack(fill_value=0)
    t = t.reindex([m for m in MONTH_ORDER if m in t.index])
    out["attempts_by_month_cause"] = {int(m): {c: int(v) for c, v in row.items() if v} for m, row in t.iterrows()}
    out["attempts_by_month_total"] = {int(m): int(att[att["month"] == m].shape[0]) for m in t.index}
    un = att[~att["named"]]
    out["unnamed_share_by_month"] = {int(m): float(1 - att[att["month"] == m]["named"].mean()) for m in t.index}
    out["all_anonymous_team_games_split"] = {k: int(v) for k, v in un.loc[un["cause"].str.startswith("team_game_all_anonymous"), "anon_kind"].value_counts().items()}
    out["all_anonymous_by_month_kind"] = {int(m): {k: int(v) for k, v in g["anon_kind"].value_counts().items()} for m, g in att[(~att["named"]) & att["cause"].str.startswith("team_game_all_anonymous")].groupby("month")}
    out["unnamed_by_cause_total"] = {c: int(v) for c, v in un["cause"].value_counts().items()}
    trips = att.groupby(["game_id", "trip_id"]).agg(named=("named", "all"), month=("month", "first"), cause=("cause", lambda s: s[s != "named"].iloc[0] if (s != "named").any() else "named")).reset_index()
    out["trips"] = {"total": int(len(trips)), "unnamed": int((~trips["named"]).sum()),
                    "unnamed_by_month": {int(m): int((~g["named"]).sum()) for m, g in trips.groupby("month")},
                    "unnamed_by_cause": {c: int(v) for c, v in trips.loc[~trips["named"], "cause"].value_counts().items()}}
    # share of unnamed attempts that have ANY prior history (returner + transfer_in + returning_after_gap)
    hist = un["cause"].str.replace("team_game_all_anonymous:", "", regex=False)
    out["unnamed_with_prior_history_share"] = float(hist.isin(["returner", "transfer_in", "returning_after_gap"]).mean())
    out["unnamed_returner_share"] = float((hist == "returner").mean())
    # when does the shooter first appear in the pool (team games played before the first named game)? gap in team games
    # for unnamed returners: how many team games into the season is the game
    tgi = cur[["team_id", "game_id", "game_date"]].drop_duplicates().sort_values(["team_id", "game_date", "game_id"])
    tgi["team_game_no"] = tgi.groupby("team_id").cumcount()
    att = att.merge(tgi[["team_id", "game_id", "team_game_no"]], on=["team_id", "game_id"], how="left")
    un = att[~att["named"]]
    out["unnamed_by_team_game_no"] = {str(k): int(v) for k, v in pd.cut(un["team_game_no"], [-1, 0, 2, 5, 9, 14, 20, 40, 100]).value_counts().sort_index().items()}
    out["unnamed_share_by_team_game_no"] = {str(k): float(1 - g["named"].mean()) for k, g in att.groupby(pd.cut(att["team_game_no"], [-1, 0, 2, 5, 9, 14, 20, 40, 100]), observed=True)}
    return out


if __name__ == "__main__":
    # `--fold2-dir <inputs dir>`: classify the same attempts against another fold-2 inputs directory (the seeded sibling);
    # output goes to results/early_season_roster_slots_<name>.json instead
    args = sys.argv[1:]
    if args and args[0] == "--fold2-dir":
        FOLDS[2]["dir"] = args[1]
        name = Path(args[1]).name
        res = {"fold2": run(2)}
        Path(ROOT / f"results/early_season_roster_slots_{name}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    else:
        res = {f"fold{k}": run(k) for k in (2, 1)}
        Path(ROOT / "results/early_season_roster_slots.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
