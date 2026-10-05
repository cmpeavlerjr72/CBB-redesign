#!/usr/bin/env python
"""Live-built K2_Ocell table vs the v3in sibling over many fold-2 dates (lane F, 2026-10-01). Season 2025 only.
    python scripts/diag_shot_block_live_vs_v3in_dates_v1.py DATE [DATE ...]   or   --auto N   (affected dates by affected-game count + controls)"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
sys.argv_in = sys.argv[1:]
sys.path[:0] = [str(R / "src"), str(R / "scripts")]
import run_daily_sim_v1 as SIM, build_engine_inputs_live as BL, build_shot_block_lut_live_v1 as SBL
from cbb_sim.live import tips as TP
args = sys.argv_in
g3 = pd.read_parquet(R / "data/processed/models/engine_v3/games_F2_2025.parquet"); g3["d"] = pd.to_datetime(g3.game_date).dt.strftime("%Y-%m-%d")
v2 = np.load(R / "data/processed/models/engine/arrays_F2_2025.npz")["roster_cbbd"]; v3 = np.load(R / "data/processed/models/engine_v3/arrays_F2_2025.npz")["roster_cbbd"]
aff = (v2 != v3).any(axis=(1, 2))
if args and args[0] == "--auto":
    n = int(args[1]); cnt = g3[aff].groupby("d").size().sort_values(ascending=False)
    top = list(cnt.index[:8]); rng = np.random.default_rng(7)
    rest = [d for d in cnt.index[8:]]; pick = list(rng.choice(rest, size=max(n - 12, 0), replace=False))
    ctrl = [d for d in sorted(set(g3.d) - set(cnt.index))][:4]
    ctrl = ctrl if len(ctrl) == 4 else list(rng.choice(sorted(set(g3.d) - set(top) - set(pick)), size=4, replace=False))
    dates = sorted(set(str(x) for x in top + pick + ctrl))
else:
    dates = args
sib = np.load(R / "data/processed/models/engine/shot_block_K2_Ocell_v3in_F2_2025.npz"); pos = {int(x): i for i, x in enumerate(sib["game_id"])}
out = {}; t0 = time.time()
for d in dates:
    now = TP.default_clock(d, "evening")
    sl = SIM.load_slate(d, 2025, "universe", None, None, None)
    ok, _ = TP.select_for_pass(sl, now, "evening")
    cols = ["game_id", "cbbd_game_id", "season", "game_date", "tipoff_utc", "home_team_id", "away_team_id", "neutral"]
    inp, _ = BL.build_live(ok[cols], now, 2025, "F2", created_at=now, season_start=SIM.season_start_of(2025, "universe", None), t0=time.time(), strict_finish=False)
    tab = SBL.build_table(inp, "K2_Ocell", as_of=now); ix = np.array([pos[int(x)] for x in tab["game_id"]])
    r = {"games": int(len(ix)), "affected_games": int(aff[[int(np.flatnonzero(g3.game_id.values == x)[0]) for x in tab["game_id"]]].sum())}
    for k in ("team", "anchor", "shooter", "known"):
        r[f"{k}_max_abs"] = float(np.abs(tab[k].astype(float) - sib[k][ix].astype(float)).max())
    r["roster_equals_v3"] = bool(np.array_equal(inp.roster_cbbd, v3[[int(np.flatnonzero(g3.game_id.values == x)[0]) for x in tab["game_id"]]]))
    out[d] = r; print(d, json.dumps(r), f"{time.time()-t0:.0f}s", flush=True)
(R / "results/shot_block_live_vs_v3in_dates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
tot = sum(v["games"] for v in out.values()); bad = [d for d, v in out.items() if max(v[f"{k}_max_abs"] for k in ("team", "anchor", "shooter", "known")) > 0 or not v["roster_equals_v3"]]
print("DATES", len(out), "GAMES", tot, "AFFECTED GAMES", sum(v["affected_games"] for v in out.values()), "DATES WITH ANY DIFFERENCE", bad)
