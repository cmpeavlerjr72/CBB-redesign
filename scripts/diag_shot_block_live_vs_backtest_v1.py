#!/usr/bin/env python
"""Fold-2 check of the shot-block live builder (lane F, 2026-10-01): (1) live-built K2_Ocell table vs the backtest table
`shot_block_K2_Ocell_F2_2025.npz` (the one the adopted full-size read used) for the same games, per array, by game id and,
for the shooter array, by roster id; (2) before / after bit-identity of the live builder on a later date, with the pre-fix
`league_before` reimplemented here verbatim. Season 2025 only; nothing sealed is read."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import run_daily_sim_v1 as SIM
import build_engine_inputs_live as BL
import build_shot_block_lut_live_v1 as SBL
import build_engine_shot_block_lut_v1 as B1
from cbb_sim.live import tips as TP

def old_league_before(dates, events, num, den):          # the pre-fix function, verbatim
    day = events.groupby("game_date")[[num, den]].sum().sort_index()
    cn, cd = day[num].cumsum().to_numpy(), day[den].cumsum().to_numpy()
    j = np.searchsorted(day.index.to_numpy(), dates, side="left") - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(j >= 0, cn[np.maximum(j, 0)] / np.maximum(cd[np.maximum(j, 0)], 1e-9), np.nan)

def build(d):
    now = TP.default_clock(d, "evening")
    sl = SIM.load_slate(d, 2025, "universe", None, None, None)
    ok, _ = TP.select_for_pass(sl, now, "evening")
    cols = ["game_id", "cbbd_game_id", "season", "game_date", "tipoff_utc", "home_team_id", "away_team_id", "neutral"]
    inp, _ = BL.build_live(ok[cols], now, 2025, "F2", created_at=now, season_start=SIM.season_start_of(2025, "universe", None),
                           t0=__import__("time").time(), strict_finish=False)
    return inp, now

out = {}
bt = np.load(REPO / "data/processed/models/engine/shot_block_K2_Ocell_F2_2025.npz")
bt_gid = {int(g): i for i, g in enumerate(bt["game_id"])}
for d in (sys.argv[1:] or ("2024-11-04", "2025-02-11")):
    inp, now = build(d)
    tab = SBL.build_table(inp, "K2_Ocell", as_of=now)
    gi = np.array([bt_gid[int(g)] for g in tab["game_id"]])
    r = {"n_games": len(gi)}
    for k in ("team", "anchor", "known"):
        r[f"{k}_max_abs"] = float(np.abs(tab[k].astype("float64") - bt[k][gi].astype("float64")).max())
    # shooter: compare by roster id (slot order may differ between live and backtest inputs)
    # backtest roster ids come from the backtest inputs
    from cbb_sim.engine.inputs import EngineInputs
    bti = EngineInputs.load(REPO / "data/processed/models/engine", "F2_2025")
    mx, n = 0.0, 0
    for a, gidx in enumerate(gi):
        for side in (0, 1):
            live = {int(r_): float(v) for r_, v in zip(inp.roster_cbbd[a, side], tab["shooter"][a, side]) if r_ >= 0}
            back = {int(r_): float(v) for r_, v in zip(bti.roster_cbbd[gidx, side], bt["shooter"][gidx, side]) if r_ >= 0}
            for key in set(live) & set(back):
                mx = max(mx, abs(live[key] - back[key])); n += 1
            r["shooters_only_live"] = r.get("shooters_only_live", 0) + len(set(live) - set(back))
            r["shooters_only_backtest"] = r.get("shooters_only_backtest", 0) + len(set(back) - set(live))
    r["shooter_max_abs_common_ids"], r["shooter_n_compared"] = mx, n
    # before / after bit identity (only meaningful when events are non-empty)
    if d not in ("2024-11-04",):
        new = tab
        B1.league_before = old_league_before; SBL.B1.league_before = old_league_before
        old = SBL.build_table(inp, "K2_Ocell", as_of=now)
        r["old_vs_new_bit_identical"] = all(np.array_equal(old[k], new[k], equal_nan=True) for k in ("team", "shooter", "known", "anchor"))
        import importlib; importlib.reload(B1); SBL.B1 = B1
    out[d] = r
    print(d, json.dumps(r), flush=True)
(REPO / "results/shot_block_live_vs_backtest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
