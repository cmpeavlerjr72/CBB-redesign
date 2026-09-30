"""
diag_live_parity_v1.py -- TRAIN/SERVE PARITY of the live inputs path against the F2 backtest builder.

    .venv/Scripts/python.exe scripts/diag_live_parity_v1.py [--dates 2024-11-12,2025-01-15,2025-03-04]

For each date: pretend the slate is unplayed (schedule row only, result columns dropped),
run the live builder with the as-of cutoff 30 min before the first tip, and compare EVERY
array and column with `arrays_F2_2025_v2.npz` for the same game ids. Every differing cell
is attributed to exactly one cause or reported as UNEXPLAINED:

  BT_ROT_FALLBACK   backtest gave the (game, team) the anonymous league-mean rotation profile
                    because the game's OWN on-floor data was incomplete (selection on the
                    game's own data quality; live always builds a real prior)
  BT_CLOCK_FALLBACK backtest clock design did not cover the game (fills 1.0 / a season-wide median)
  BT_STALE_ROW      the player has no design/as-of row for THIS game, so the backtest builder's
                    merge_asof(backward) carried his most recent EARLIER row (possibly from the
                    previous season, at an older league-centring date). Live joins exactly.
  UNEXPLAINED       none expected; each one is a finding.

Writes results/live_parity_2026-09-30/{parity.json, parity.md}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_engine_inputs_live as BL  # noqa: E402
from cbb_sim.live import guards as G  # noqa: E402

ED = ROOT / "data/processed/models/engine"
OUT = ROOT / "results/live_parity_2026-09-30"


def exact_row_sets(game_ids: set) -> dict:
    """Which (game, player[, class]) keys have an exact-date row in the backtest builder's sources."""
    fgd = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet",
                          columns=["game_id", "shooter_id", "shot_class"])
    fgd = fgd[fgd["game_id"].isin(game_ids)]
    fgd["shooter_id"] = fgd["shooter_id"].astype("int64")
    fg_cls = {(int(g), int(p), c) for g, p, c in fgd.itertuples(index=False)}
    fg_any = {(g, p) for g, p, _ in fg_cls}
    att = pd.read_parquet(ROOT / "data/processed/models/free_throw/attempts_v1_era.parquet",
                          columns=["game_id", "shooter_id", "foul_class"])
    att = att[att["game_id"].isin(game_ids) & (att["foul_class"] != "technical") & att["shooter_id"].notna()]
    ft = {(int(g), int(p)) for g, p in zip(att["game_id"], att["shooter_id"].astype("int64"))}
    ua = pd.read_parquet(ROOT / "data/processed/models/usage_v2/asof_v2_shotshooter.parquet",
                         columns=["game_id", "player_id"])
    ua = ua[ua["game_id"].isin(game_ids)]
    us = {(int(g), int(p)) for g, p in zip(ua["game_id"], ua["player_id"])}
    rb = pd.read_parquet(ROOT / "data/processed/models/rebound/events_v1.parquet")
    rb = rb[rb["game_id"].isin(game_ids) & rb["outcome"].isin(["OREB", "DREB"])]
    on = rb[[f"{s}_on_{i}" for s in ("home", "away") for i in range(1, 6)]].to_numpy()
    ok = np.isfinite(on).all(axis=1)
    rbs = set()
    for g, row in zip(rb["game_id"].to_numpy()[ok], on[ok].astype("int64")):
        for p in row:
            rbs.add((int(g), int(p)))
    return {"fg_cls": fg_cls, "fg_any": fg_any, "ft": ft, "usage": us, "reb": rbs}


def run_date(D: str, zb, gb, nb, log) -> dict:
    t0 = time.time()
    sel = gb[pd.to_datetime(gb["game_date"]) == pd.Timestamp(D)]
    ids = sel["game_id"].tolist()
    slate = BL.load_slate_from_universe(D, 2025, only_ids=ids)
    first_tip = pd.to_datetime(slate["tipoff_utc"], utc=True).min()
    as_of = first_tip - pd.Timedelta(minutes=30)
    inp, diag = BL.build_live(slate, as_of, 2025, "F2", created_at=as_of, season_start="2024-11-04", t0=t0)
    inp.save(ROOT / "data/processed/models/engine_live", f"LIVE_F2_2025_{D}")
    np.savez_compressed(ROOT / f"data/processed/models/engine_live/event_block_LIVE_F2_2025_{D}.npz",
                        team_block=inp.event_block, cols=np.array(list(BL.B.TEAM_COLS[:16])))
    pos = {int(g): i for i, g in enumerate(gb["game_id"])}
    bi = np.array([pos[int(g)] for g in inp.games["game_id"]])
    gids = inp.games["game_id"].to_numpy()
    tn, sn = nb["team_names"], nb["slot_names"]
    rows: list[dict] = []
    unexplained: list[str] = []

    def add(name, n_cells, n_diff, maxd, causes):
        rows.append({"date": D, "array": name, "cells": int(n_cells), "n_diff": int(n_diff),
                     "max_abs": float(maxd), "causes": causes})

    bt_roster = zb["roster_cbbd"][bi]
    bt_fb = bt_roster[:, :, 0] < 0          # (G,2) backtest fallback profile
    lv_fb = inp.roster_cbbd[:, :, 0] < 0
    same_roster = np.all(inp.roster_cbbd == bt_roster, axis=2)   # (G,2)
    bt_ts = zb["team_static"][bi]

    # ---- team_static, per column ------------------------------------------
    clock_cols = ("off_tempo_rel", "def_tempo_rel", "tempo_prior_game")
    # backtest team-games whose whole possession-outcome block is 0.0 (season_idx 0, site 0, days 0):
    # the game has no row in the cached PO design (no possession table), so the builder left zeros
    no_po = (bt_ts[:, :, :16] == 0).all(axis=2)
    for c, j in tn.items():
        d = np.abs(inp.team_static[:, :, j].astype("f8") - bt_ts[:, :, j].astype("f8"))
        nd = int((d > 0).sum())
        causes = {}
        if nd:
            d = np.where(no_po & (d > 0), -1.0, d)       # mark, then count separately
            if (d == -1.0).any():
                causes["BT_NO_PO_DESIGN_ROW(all-zero block incl. season_idx=0)"] = int((d == -1.0).sum())
            d = np.where(d == -1.0, 0.0, d)
            if c in clock_cols:
                # backtest clock fallback = 1.0 / 1.0 for both tempo ratios on the same team-game pair
                fbm = (bt_ts[:, :, tn["off_tempo_rel"]] == 1.0) | (bt_ts[:, :, tn["def_tempo_rel"]] == 1.0)
                expl = (d > 0) & fbm
                causes["BT_CLOCK_FALLBACK"] = int(expl.sum())
                rest = int(((d > 0) & ~fbm).sum())
            else:
                rest = int((d > 0).sum())
            if rest:
                causes["UNEXPLAINED"] = rest
                unexplained.append(f"team_static.{c}: {rest}")
        add(f"team_static.{c}", d.size, nd, np.abs(inp.team_static[:, :, j].astype("f8") - bt_ts[:, :, j].astype("f8")).max(), causes)

    # ---- round-2 event team block (the block the SERVED event adapter reads) ---------
    bt_ev = np.load(ED / "event_round2_s1_F2_2025/team_block.npz")["team_block"][bi]
    r2 = pd.read_parquet(ROOT / "data/processed/models/possession_outcome/round2/design.parquet",
                         columns=["game_id", "offense_team_id"]).drop_duplicates()
    covered = {(int(g), int(t)) for g, t in zip(r2["game_id"], r2["offense_team_id"]) if int(g) in set(map(int, gids))}
    sides = np.array([[(int(g), int(h)) in covered, (int(g), int(a)) in covered]
                      for g, h, a in zip(gids, inp.games["home_team_id"], inp.games["away_team_id"])])
    for j, c in enumerate(BL.B.TEAM_COLS[:16]):
        d = np.abs(inp.event_block[:, :, j].astype("f8") - bt_ev[:, :, j].astype("f8"))
        nd = int((d > 0).sum())
        causes = {}
        if nd:
            stale = (d > 0) & ~sides
            if stale.any():
                causes["BT_TEAM_GAME_NOT_IN_R2_DESIGN(backward-filled or 0.0)"] = int(stale.sum())
            rest = int(((d > 0) & sides).sum())
            if rest:
                causes["UNEXPLAINED"] = rest
                unexplained.append(f"event_block_r2.{c}: {rest}")
        add(f"event_block_r2.{c}", d.size, nd, d.max(), causes)

    # ---- roster / rotation arrays -------------------------------------------
    for k in ("roster_cbbd", "roster_espn", "roster_valid", "rot_share", "rot_srank", "rot_start",
              "rot_fpm", "rot_pavail"):
        a, b = getattr(inp, k), zb[k][bi]
        diff = (a != b) if a.dtype.kind not in "fc" else (np.abs(a.astype("f8") - b.astype("f8")) > 0)
        rowdiff = diff.reshape(diff.shape[0], 2, -1).any(axis=2)      # (G,2)
        expl = rowdiff & bt_fb & ~lv_fb
        rest = int((rowdiff & ~expl).sum())
        causes = {}
        if expl.any():
            causes["BT_ROT_FALLBACK(team-games)"] = int(expl.sum())
        if rest:
            causes["UNEXPLAINED(team-games)"] = rest
            unexplained.append(f"{k}: {rest} team-games")
        mx = float(np.abs(a.astype("f8") - b.astype("f8")).max()) if a.dtype.kind in "fc" else float("nan")
        add(k, a.size, int(diff.sum()), mx, causes)

    # ---- slot / usage / reb columns, on identical-roster real slots -----------
    sets = exact_row_sets(set(int(g) for g in gids))
    real = (inp.roster_cbbd > 0) & same_roster[:, :, None]
    ii, ss, kk = np.nonzero(real)
    pids = inp.roster_cbbd[ii, ss, kk]
    gid_cell = gids[ii]

    def attribute(name, d_cells, has_row):
        nd = int(d_cells.sum())
        if not nd:
            add(name, int(real.sum()), 0, 0.0, {})
            return
        stale = d_cells & ~has_row
        unexp = d_cells & has_row
        causes = {"BT_STALE_ROW": int(stale.sum())}
        if unexp.any() and name.endswith("has_prior_season_fg"):
            causes["BT_FIRST_ROW_CLASS_DEPENDENCE"] = int(unexp.sum())     # game-day dependence, LEAK FINDING
        elif unexp.any():
            causes["UNEXPLAINED"] = int(unexp.sum())
            unexplained.append(f"{name}: {int(unexp.sum())}")
        return causes, nd

    cls_of = {"rim": "FGA_rim", "jump2": "FGA_jump2", "three": "FGA_3"}

    def has_row_for(col):
        base = col.split("__")[0]
        key = col.split("__")[1] if "__" in col else None
        if col.startswith(("shooter_ft", "prior_season_ft", "has_prior_season_ft")):
            return np.array([(int(g), int(p)) in sets["ft"] for g, p in zip(gid_cell, pids)])
        if col in ("sh_share_rim", "sh_share_jump2", "sh_share_three", "sh_assisted_share"):
            # backtest wide table keeps these shared columns only on the RIM frame's rows
            return np.array([(int(g), int(p), "FGA_rim") in sets["fg_cls"] for g, p in zip(gid_cell, pids)])
        if key in cls_of:
            return np.array([(int(g), int(p), cls_of[key]) in sets["fg_cls"] for g, p in zip(gid_cell, pids)])
        return np.array([(int(g), int(p)) in sets["fg_any"] for g, p in zip(gid_cell, pids)])

    for c, j in sn.items():
        a = inp.slot_static[ii, ss, kk, j].astype("f8")
        b = zb["slot_static"][bi][ii, ss, kk, j].astype("f8")
        dcell = np.abs(a - b) > 0
        res = attribute(f"slot_static.{c}", dcell, has_row_for(c))
        if res:
            causes, nd = res
            add(f"slot_static.{c}", len(pids), nd, np.abs(a - b).max(), causes)
    for k, cls in enumerate(BL.B.USAGE_CLASSES):
        a = inp.usage_rate[ii, ss, kk, k].astype("f8")
        b = zb["usage_rate"][bi][ii, ss, kk, k].astype("f8")
        hasr = np.array([(int(g), int(p)) in sets["usage"] for g, p in zip(gid_cell, pids)])
        res = attribute(f"usage_rate.{cls}", np.abs(a - b) > 0, hasr)
        if res:
            add(f"usage_rate.{cls}", len(pids), res[1], np.abs(a - b).max(), res[0])
    for k, c in enumerate(("oreb_rate", "dreb_rate")):
        a = inp.reb_rate[ii, ss, kk, k].astype("f8")
        b = zb["reb_rate"][bi][ii, ss, kk, k].astype("f8")
        hasr = np.array([(int(g), int(p)) in sets["reb"] for g, p in zip(gid_cell, pids)])
        res = attribute(f"reb_rate.{c}", np.abs(a - b) > 0, hasr)
        if res:
            add(f"reb_rate.{c}", len(pids), res[1], np.abs(a - b).max(), res[0])

    # ---- leak-finding quantification: fill constants the backtest takes from the full season
    leak = {}
    names_b = nb["rules"]
    for cls in BL.B.USAGE_CLASSES:
        bt_c = names_b[f"usage_prior_{cls}"]["no_history_rate"]
        lv_c = inp.rules[f"usage_prior_{cls}"]["no_history_rate"]
        leak[f"usage_no_history_rate.{cls}"] = {"backtest_full_table_median": bt_c, "live_cut_median": lv_c,
                                                "abs_diff": None if lv_c is None else abs(bt_c - lv_c)}
    leak["clock_tempo_fallback.tempo_prior_game"] = {
        "backtest_full_season_median": names_b["clock_tempo_fallback"]["tempo_prior_game"],
        "live_asof_league_mean": float(np.nanmedian(inp.team_static[:, :, tn["tempo_prior_game"]]))}
    # ---- proof of the BT_STALE_ROW explanation for the fg per-class shooter columns: the
    # backtest value must equal the player's most recent EARLIER design row (merge_asof backward)
    fgd = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet",
                          columns=["shooter_id", "game_date", "shot_class", "shooter_make_c", "shooter_att_c",
                                   "prior_season_make_c"])
    fgd["game_date"] = pd.to_datetime(fgd["game_date"])
    fgd = fgd[fgd["game_date"] <= pd.Timestamp(D)].drop_duplicates(["shooter_id", "game_date", "shot_class"])
    fgd = fgd.sort_values("game_date")
    proof = {}
    for key, cls in cls_of.items():
        sub = fgd[fgd["shot_class"] == cls]
        last = sub.groupby("shooter_id").tail(1).set_index("shooter_id")
        for c in ("shooter_make_c", "shooter_att_c", "prior_season_make_c"):
            j = sn[f"{c}__{key}"]
            a = inp.slot_static[ii, ss, kk, j].astype("f8")
            b = zb["slot_static"][bi][ii, ss, kk, j].astype("f8")
            stale_cells = (np.abs(a - b) > 0) & ~np.array(
                [(int(g), int(p), cls) in sets["fg_cls"] for g, p in zip(gid_cell, pids)])
            exp = last[c].reindex(pids[stale_cells]).to_numpy(dtype="f8")
            exp = np.where(np.isfinite(exp), exp, 0.0)
            proof[f"{c}__{key}"] = {"stale_cells": int(stale_cells.sum()),
                                    "bt_equals_last_earlier_row": int((np.abs(exp - b[stale_cells]) < 1e-6).sum())}
    G.assert_created_before_tipoff(inp.games.assign(created_at=as_of.isoformat()))
    return {"date": D, "n_games": int(len(gids)), "first_tip_utc": str(first_tip), "as_of": str(as_of),
            "rows": rows, "unexplained": unexplained, "stale_proof": proof, "leak_constants": leak, "diag": diag,
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="2024-11-12,2025-01-15,2025-03-04")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)  # noqa: E731
    gb = pd.read_parquet(ED / "games_F2_2025_v2.parquet")
    zb = dict(np.load(ED / "arrays_F2_2025_v2.npz"))
    nb = json.loads((ED / "names_F2_2025_v2.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for D in args.dates.split(","):
        r = run_date(D, zb, gb, nb, log)
        res.append(r)
        n_cols = len(r["rows"])
        n_diff_cols = sum(1 for x in r["rows"] if x["n_diff"])
        log(f"{D}: {r['n_games']} games, {n_cols} column-groups compared, {n_diff_cols} differ, "
            f"UNEXPLAINED: {r['unexplained'] or 'none'}")
        (OUT / "parity.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
