"""diag_rbto_parity_v1.py -- train/serve parity of the RBTO tagged inputs (lane I, 2026-10-01; rebound experiments.md 12.3).

Served `off_oreb_c` / `opp_def_dreb_c` of `engine_v3_I_RBTO` (team_static, per game x offence side) must equal the TRAINED
values: `team_rate_adapter.apply(rebound design, E3_v4, 'rebound', fold='F2')` on the matching (game, offence) rows of the
2025 rebound design, to 1e-6 on >= 99.9% of matched rows. Also: every other team_static / slot_static column equals the
base `engine_v3` (only the two rebound columns may differ), and the anchor offsets file covers every game.

Usage: diag_rbto_parity_v1.py <tag_input_dir> <out_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim import team_rate_adapter as TRA  # noqa: E402

TAGD, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BASE = ROOT / "data/processed/models/engine_v3"
E3 = ROOT / "data/processed/team_rate_features_E3_v4.parquet"
z = np.load(TAGD / "arrays_F2_2025.npz")
z0 = np.load(BASE / "arrays_F2_2025.npz")
names = json.loads((TAGD / "names_F2_2025.json").read_text(encoding="utf-8"))
tn = names["team_names"]
games = pd.read_parquet(TAGD / "games_F2_2025.parquet").reset_index(drop=True)
res = {}
diff_cols = [c for c, j in tn.items() if not np.array_equal(z["team_static"][:, :, j], z0["team_static"][:, :, j])]
res["team_static_changed_columns"] = diff_cols
res["slot_static_identical"] = bool(np.array_equal(z["slot_static"], z0["slot_static"]))
other = [k for k in z.files if k not in ("team_static", "slot_static") and not np.array_equal(z[k], z0[k])]
res["other_arrays_changed"] = other
d = pd.read_parquet(ROOT / "data/processed/models/rebound/round3/design_round3.parquet",
                    columns=["season", "game_id", "off_team_id", "def_team_id", "off_oreb_c", "opp_def_dreb_c"])
d = d[d["season"] == 2025].drop_duplicates(["game_id", "off_team_id"]).reset_index(drop=True)
tr = TRA.apply(d, str(E3), "rebound", fold="F2", missing="raise")
pos = {int(g): i for i, g in enumerate(games["game_id"].to_numpy())}
m = tr["game_id"].map(pos)
tr = tr[m.notna()].reset_index(drop=True)
g = tr["game_id"].map(pos).astype(int).to_numpy()
side = np.where(tr["off_team_id"].to_numpy() == games["home_team_id"].to_numpy()[g], 0,
                np.where(tr["off_team_id"].to_numpy() == games["away_team_id"].to_numpy()[g], 1, -1))
ok = side >= 0
res["design_rows_2025"] = int(len(d))
res["matched_rows"] = int(ok.sum())
for c in ("off_oreb_c", "opp_def_dreb_c"):
    served = z["team_static"][g[ok], side[ok], tn[c]].astype(np.float64)
    trained = tr.loc[ok, c].to_numpy(np.float64)
    base = z0["team_static"][g[ok], side[ok], tn[c]].astype(np.float64)
    res[c] = {"share_equal_1e6": float(np.mean(np.abs(served - trained) <= 1e-6)),
              "max_abs_diff": float(np.max(np.abs(served - trained))),
              "served_vs_base_share_changed": float(np.mean(np.abs(served - base) > 1e-7)),
              "corr_served_base": float(np.corrcoef(served, base)[0, 1])}
off = np.load(TAGD / "anchor_offsets_F2_2025.npz")
res["anchor"] = {"files": off.files, "games_equal": bool(np.array_equal(off["game_ids"], games["game_id"].to_numpy())),
                 "rb_oreb_quantiles": [float(x) for x in np.quantile(off["rb_oreb"], [0, .1, .5, .9, 1])]}
res["PASS"] = bool(res["off_oreb_c"]["share_equal_1e6"] >= 0.999 and res["opp_def_dreb_c"]["share_equal_1e6"] >= 0.999
                   and set(diff_cols) <= {"off_oreb_c", "opp_def_dreb_c"} and res["slot_static_identical"]
                   and not other and res["anchor"]["games_equal"])
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
