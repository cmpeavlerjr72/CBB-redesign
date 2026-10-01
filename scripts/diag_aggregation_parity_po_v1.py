"""diag_aggregation_parity_po_v1.py -- lane A 2026-09-30, addendum H: possession_outcome T train/serve parity
(training design_overlay vs the S1 engine's round-2 team block, fold-2 rows)."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
HF = Path("C:/Users/devuser/AppData/Local/Temp/claude/C--Users-devuser-CBB-clean-sheet/f2e4a65e-7674-468c-ac10-92b191e2329c/scratchpad/hf/model_artifacts/possession_outcome/round_stageb/T/team_rate_features_E3_v4")
d = pd.read_parquet(HF / "design_overlay.parquet")
d = d[d["season"] == 2025]
idx = json.loads((HF / "event_round2_s1_F2_2025" / "index.json").read_text(encoding="utf-8"))
print("T index features (first):", idx["populations"]["first"]["features"])
sdir = ROOT / "data/processed/models/engine_v3_S1_laneA"
tb = np.load(sdir / "overlay/data/processed/models/engine/event_round2_s1_F2_2025/team_block.npz")["team_block"]
cols = idx["team_cols"]
g = pd.read_parquet(sdir / "games_F2_2025.parquet")
rows = []
for side, col in ((0, "home_team_id"), (1, "away_team_id")):
    e = pd.DataFrame(tb[:, side, :], columns=cols); e["game_id"] = g["game_id"].to_numpy(); e["offense_team_id"] = g[col].to_numpy()
    rows.append(e)
eng = pd.concat(rows, ignore_index=True)
dd = d.groupby(["game_id", "offense_team_id"], as_index=False)[[c for c in cols if c in d.columns]].first()
m = dd.merge(eng, on=["game_id", "offense_team_id"], suffixes=("_design", "_engine"))
out = {"n_matched": int(len(m))}
for c in cols:
    if f"{c}_design" in m:
        a, b = m[f"{c}_design"].to_numpy(float), m[f"{c}_engine"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        out[c] = {"corr": float(np.corrcoef(a[ok], b[ok])[0, 1]) if a[ok].std() > 0 else None, "max_abs_diff": float(np.max(np.abs(a[ok] - b[ok]))),
                  "share_gt_0.01": float(np.mean(np.abs(a[ok] - b[ok]) > 0.01))}
print(json.dumps(out, indent=0))
(ROOT / "results/aggregation_v1/analysis_parity_po_v1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
