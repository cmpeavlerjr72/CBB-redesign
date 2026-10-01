"""diag_usage_ft_late_parity_v1.py -- train/serve parity of z (usage experiments.md s13): the trainer's z for each of the
five on-floor players of every 2025 FT trip vs the z the engine computes from its FT slot columns (lane I, 2026-10-01).

Usage: diag_usage_ft_late_parity_v1.py <out_json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import train_usage_ft_late_v1 as T  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402

f = T.load_events().reset_index(drop=True)
f = f[f["season"] == 2025].reset_index(drop=True)
pg, prev, lg = T.ft_skill_table()
Z = T.attach_z(f, pg, prev, lg)
inp = EngineInputs.load(str(ROOT / "data/processed/models/engine_v3"), "F2_2025")
pos = {int(g): i for i, g in enumerate(inp.games["game_id"].to_numpy())}
sn = inp.slot_names
ok, tr, sv = [], [], []
for k in range(5):
    g = f["game_id"].map(pos)
    m = g.notna().to_numpy()
    gi = g[m].astype(int).to_numpy()
    side = np.where(f.loc[m, "team_id"].to_numpy() == inp.games["home_team_id"].to_numpy()[gi], 0, 1)
    pid = f.loc[m, f"alt_{k + 1}"].to_numpy()
    hit = inp.roster_cbbd[gi, side] == pid[:, None]
    h = hit.any(axis=1)
    s = hit.argmax(axis=1)
    blk = inp.slot_static[gi[h], side[h], s[h]]
    own, fta, pri = blk[:, sn["shooter_ft_asof"]], blk[:, sn["shooter_fta_asof"]], blk[:, sn["prior_season_ft"]]
    sv.append((fta * own + 30 * pri) / (fta + 30))
    tr.append(Z[np.flatnonzero(m)[h], k])
sv, tr = np.concatenate(sv), np.concatenate(tr)
res = {"n": int(len(sv)), "share_abs_diff_le_0p002": float(np.mean(np.abs(sv - tr) <= 0.002)),
       "corr": float(np.corrcoef(sv, tr)[0, 1]), "mean_served": float(sv.mean()), "mean_train": float(tr.mean()),
       "mean_abs_diff": float(np.mean(np.abs(sv - tr)))}
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print(res)
