"""diag_g1_lfl_v4_v1.py -- the accepted like-for-like G1 read against the BUILT
event layer v4 (engine 200-seed count vs possessions_v4 count, pbp-complete
graded 2025 games), next to the in-memory correction of
diag_g1_possessions_v2.py.  Lane B, 2026-09-30.  Diagnostic only; prints and
writes results/g1g5_diag/g1_lfl_v4.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402

g200 = pd.read_parquet(ROOT / "results/engine_v0/F2_2025_s200_v5b_A_full/games.parquet")
summary, raw = G.build_grading_frame(g200, 2025)
sim = raw.groupby("game_id")["possessions"].mean()
u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet").set_index("game_id")
tb = R.load_actual_team_box(2025)
two = tb.groupby("game_id").size()
out = {}
for lab, d in (("v2", "possessions_v2"), ("v4", "possessions_v4")):
    p = pd.read_parquet(ROOT / f"data/processed/{d}/possessions_2025.parquet")
    nt = p.groupby("game_id")["offense_team_id"].nunique()
    cnt = p.groupby(["game_id", "offense_team_id"]).size().groupby("game_id").mean()
    pc = [g for g in summary["game_id"] if g in nt.index and nt[g] == 2 and bool(u.at[g, "pbp_complete"])
          and two.get(g, 0) == 2]
    out[lab] = {"n_games": len(pc), "sim": float(sim.loc[pc].mean()), "count": float(cnt.loc[pc].mean()),
                "gap": float(sim.loc[pc].mean() - cnt.loc[pc].mean())}
(ROOT / "results/g1g5_diag/g1_lfl_v4.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
