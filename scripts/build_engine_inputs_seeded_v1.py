#!/usr/bin/env python
"""
build_engine_inputs_seeded_v1.py -- engine inputs v3 with an AS-OF-SAFE SEEDED CANDIDATE ROSTER, as a VERSIONED SIBLING
(lane F, 2026-10-01; docs/tests/early_season_roster_slots_2026-10-01.md). Fold 2 (season 2025) first. NOT wired into the engine, the
chain or the daily builder default; `build_live(..., seed_fn=...)` is the only touch point and is default off.

PROBLEM. The candidate roster of a team-game is the set of players with an EARLIER appearance FOR THAT TEAM THIS SEASON (on-floor table,
`rotation.build_asof_player_features`). A team's first game of a season has no such player, and a team whose earlier games are missing from
the on-floor table (pbp-incomplete) has none either: every slot is anonymous, so a returning player who shoots free throws is served an
all-zero shooter block.

REPAIR (this builder). While a team has fewer than W (default 10) USABLE earlier games this season (games with a row in the on-floor
table before the slate date: an as-of quantity), the ANONYMOUS slots of the team-game are filled with named players, in this order:
  1. RETURNERS: players who appeared for THIS team in the previous season (previous-season on-floor table, complete before this season),
     by previous-season minutes at this team, descending;
  2. TRANSFERS IN: players on the hoopR roster of this team and season (`data/raw/hoopr/rosters/rosters_{season}.parquet`) who appeared
     for ANOTHER team in the previous season, by previous-season minutes, descending.
Nothing from the game's own box score is read; `game_rosters` (which lists the game's own roster) is NOT used. Only slot IDENTITY
changes: the rotation arrays (`rot_*`, `roster_valid`) of the filled slots are left as the anonymous / fallback profile, so the rotation
model sees the same slot structure; the slot-keyed player features (fg / FT shooter blocks, usage, rebound rates, ESPN ids) are joined by
the live builder's own code for the now-named ids. Players already in the as-of pool keep their slots.

AS-OF CAVEAT (stated, not hidden): the hoopR roster file is a season snapshot, not time-stamped; a player added to a roster in mid-season
would appear as a candidate earlier than he was knowable. Step 2 can only add players who already appeared for another team last season;
the count of step-2 seeds is reported by month so the exposure is visible.

    python scripts/build_engine_inputs_seeded_v1.py shard --shard 0 --n-shards 2     # per-date builds into the stage dir
    python scripts/build_engine_inputs_seeded_v1.py assemble                         # -> data/processed/models/engine_v3_seed/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_engine_inputs_live as BL  # noqa: E402
from cbb_sim.data import player_ids as PID  # noqa: E402
from cbb_sim.models import rotation as ROT  # noqa: E402

SEASON = 2025
W_DEFAULT = 10
ED = ROOT / "data/processed/models/engine"
V3 = ROOT / "data/processed/models/engine_v3"
STAGE = ROOT / "data/processed/models/engine_seed_stage"
OUT = ROOT / "data/processed/models/engine_v3_seed"
RES = ROOT / "results/engine_v3_seed/dates"
ROSTERS = ROOT / "data/raw/hoopr/rosters/rosters_2025.parquet"
CROSSWALK = ROOT / "data/processed/player_crosswalk.parquet"
CHANGED = ("roster_cbbd", "roster_espn", "slot_static", "usage_rate", "reb_rate")

_C: dict = {}


def tables(season: int) -> dict:
    """Process-level cache: previous-season minutes by (team, pid), hoopR roster in cbbd ids, usable in-season games per team-date."""
    if season in _C:
        return _C[season]
    prev = ROT.player_game_minutes(ROT.load_team_possessions(season - 1))
    tot = prev.groupby(["team_id", "pid"])["minutes"].sum().reset_index()
    tot = tot[tot["minutes"] > 0]
    by_team = {t: g.sort_values("minutes", ascending=False)["pid"].astype("int64").tolist() for t, g in tot.groupby("team_id")}
    any_min = tot.groupby("pid")["minutes"].sum().to_dict()
    ros = pd.read_parquet(ROSTERS, columns=["team_id", "athlete_id"]).dropna()
    cw = pd.read_parquet(CROSSWALK)
    e2c = {v: k for k, v in PID.cbbd_to_espn_map(cw, season).items()}
    ros["pid"] = ros["athlete_id"].astype("int64").map(e2c)
    ros = ros.dropna(subset=["pid"])
    roster_by_team = {t: set(g["pid"].astype("int64")) for t, g in ros.groupby("team_id")}
    cur = ROT.load_team_possessions(season)[["team_id", "game_id", "game_date"]].drop_duplicates()
    cur["game_date"] = pd.to_datetime(cur["game_date"])
    _C[season] = {"by_team": by_team, "any_min": any_min, "roster_by_team": roster_by_team, "cur": cur,
                  "prev_team_sets": {t: set(v) for t, v in by_team.items()}}
    return _C[season]


def make_seed_fn(W: int = W_DEFAULT):
    def seed_fn(ctx, games, tg, roster_cbbd, roster_valid, S, diag) -> dict:
        T = tables(ctx.season)
        cur = T["cur"]
        before = cur[cur["game_date"] < pd.Timestamp(ctx.slate_date)]
        usable = before.groupby("team_id")["game_id"].nunique().to_dict()
        gpos = {int(g): i for i, g in enumerate(games["game_id"])}
        n_filled = n_ret = n_tr = n_tg = 0
        for g, t, is_home in zip(tg["game_id"], tg["team_id"], tg["is_home"]):
            i, side = gpos[int(g)], (0 if bool(is_home) else 1)
            if usable.get(int(t), 0) >= W:
                continue
            row = roster_cbbd[i, side]
            have = {int(x) for x in row if x > 0}
            free = [j for j in range(S) if row[j] <= 0]
            if not free:
                continue
            returners = [p for p in T["by_team"].get(int(t), []) if p not in have]
            rset = set(returners)
            trans = [p for p in T["roster_by_team"].get(int(t), set()) if p not in have and p not in rset
                     and p in T["any_min"] and p not in T["prev_team_sets"].get(int(t), set())]
            trans.sort(key=lambda p: -T["any_min"][p])
            seeds = (returners + trans)[: len(free)]
            for j, p in zip(free, seeds):
                row[j] = p
            n_filled += len(seeds); n_ret += min(len(returners), len(free)); n_tr += max(len(seeds) - len(returners), 0); n_tg += int(len(seeds) > 0)
        diag.update({"seed_slots_filled": n_filled, "seed_returner_slots": n_ret, "seed_transfer_slots": n_tr, "seed_team_games": n_tg, "seed_W": W})
        return {(int(g), int(t)): [int(x) for x in roster_cbbd[gpos[int(g)], 0 if bool(h) else 1] if x > 0]
                for g, t, h in zip(tg["game_id"], tg["team_id"], tg["is_home"])}
    return seed_fn


def dates_of(gb: pd.DataFrame) -> list[str]:
    return sorted(pd.to_datetime(gb["game_date"]).dt.strftime("%Y-%m-%d").unique())


def shard(k: int, n: int, W: int) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    gb = pd.read_parquet(ED / "games_F2_2025_v2.parquet")
    seed_fn = make_seed_fn(W)
    t0 = time.time()
    for i, D in enumerate(dates_of(gb)):
        if i % n != k or (RES / f"{D}.json").exists():
            continue
        try:
            ids = gb[pd.to_datetime(gb["game_date"]) == pd.Timestamp(D)]["game_id"].tolist()
            slate = BL.load_slate_from_universe(D, SEASON, only_ids=ids)
            as_of = pd.to_datetime(slate["tipoff_utc"], utc=True).min() - pd.Timedelta(minutes=30)
            inp, diag = BL.build_live(slate, as_of, SEASON, "F2", created_at=as_of, season_start="2024-11-04", t0=t0,
                                      strict_finish=False, seed_fn=seed_fn)
            inp.save(STAGE, f"SEED_F2_2025_{D}")
            (RES / f"{D}.json").write_text(json.dumps({k2: v for k2, v in diag.items() if not isinstance(v, (dict, list))}, default=str), encoding="utf-8")
            print(f"[{time.time()-t0:7.0f}s] {D} games {len(slate)} seeded slots {diag.get('seed_slots_filled')}", flush=True)
        except Exception:
            (RES / f"{D}.err").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{time.time()-t0:7.0f}s] {D} FAILED", flush=True)


def assemble() -> None:
    gb = pd.read_parquet(ED / "games_F2_2025_v2.parquet")
    g3 = pd.read_parquet(V3 / "games_F2_2025.parquet")
    z3 = dict(np.load(V3 / "arrays_F2_2025.npz"))
    names = json.loads((V3 / "names_F2_2025.json").read_text(encoding="utf-8"))
    pos = {int(g): i for i, g in enumerate(g3["game_id"])}
    arrs = {k: v.copy() for k, v in z3.items()}
    filled = np.zeros(len(g3), bool)
    unchanged_ok = {}
    for D in dates_of(gb):
        tag = f"SEED_F2_2025_{D}"
        g = pd.read_parquet(STAGE / f"games_{tag}.parquet")
        z = np.load(STAGE / f"arrays_{tag}.npz")
        idx = np.array([pos[int(x)] for x in g["game_id"]])
        for k in z3:
            if k in CHANGED:
                arrs[k][idx] = z[k]
            else:                                   # everything else must equal the v3 inputs for these games
                same = np.array_equal(z[k], z3[k][idx], equal_nan=True) if z[k].dtype.kind == "f" else np.array_equal(z[k], z3[k][idx])
                unchanged_ok[k] = unchanged_ok.get(k, True) and bool(same)
        filled[idx] = True
    assert filled.all(), f"{int((~filled).sum())} games unfilled"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "arrays_F2_2025.npz", **arrs)
    g3.to_parquet(OUT / "games_F2_2025.parquet", index=False)
    shutil.copy2(V3 / "event_block_F2_2025.npz", OUT / "event_block_F2_2025.npz")
    names.setdefault("meta", {})["seeded_roster"] = {"builder": "scripts/build_engine_inputs_seeded_v1.py", "W": W_DEFAULT,
                                                     "changed_arrays": list(CHANGED), "base": "engine_v3"}
    (OUT / "names_F2_2025.json").write_text(json.dumps(names, indent=1, default=str), encoding="utf-8")
    print("assembled ->", OUT, "| non-changed arrays equal v3 on every date:", unchanged_ok, flush=True)
    (ROOT / "results/engine_v3_seed").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/engine_v3_seed/assemble_report.json").write_text(json.dumps({"unchanged_arrays_equal_v3": unchanged_ok}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shard", "assemble"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    a = ap.parse_args()
    shard(a.shard, a.n_shards, a.W) if a.cmd == "shard" else assemble()
