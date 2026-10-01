"""diag_full_retrain_v4_style_skew_v1.py -- how far do PO's as-of style columns move from possessions v2 to v4?

The engine inputs (v3 live replay) build the PO team block from possessions v2 (`live/features.py::po_team_block_r2`);
the full-retrain PO is trained on v4. This measures that residual train/serve skew on the 2025 test season, per
(game, offence team), for the 8 style columns: mean |v4 - v2|, the 99th percentile, and the SD of each column for scale.

    python scripts/diag_full_retrain_v4_style_skew_v1.py --v4-design <root>/po_design/design.parquet
"""
import argparse
import json
from pathlib import Path

import pandas as pd

COLS = ["off_3pa_c", "off_rim_c", "off_tov_c", "off_ftr_c", "opp_def_3pa_c", "opp_def_rim_c", "opp_def_tov_c",
        "opp_def_ftr_c"]
ap = argparse.ArgumentParser()
ap.add_argument("--v4-design", required=True)
ap.add_argument("--v2-design", default="data/processed/models/possession_outcome/round2/design.parquet")
a = ap.parse_args()
k = ["game_id", "offense_team_id"]
rd = lambda p: (pd.read_parquet(p, columns=[*k, "season", *COLS])  # noqa: E731
                .query("season == 2025").drop_duplicates(k).set_index(k)[COLS])
v2, v4 = rd(a.v2_design), rd(a.v4_design)
j = v2.join(v4, lsuffix="_v2", rsuffix="_v4", how="inner")
out = {"team_games": int(len(j))}
for c in COLS:
    d = (j[c + "_v4"] - j[c + "_v2"]).abs()
    out[c] = {"mean_abs": round(float(d.mean()), 4), "p99_abs": round(float(d.quantile(0.99)), 4),
              "share_changed": round(float((d > 0).mean()), 4), "sd_v2": round(float(j[c + "_v2"].std()), 4)}
print(json.dumps(out, indent=1))
Path(a.v4_design).with_name("v4_style_skew.json").write_text(json.dumps(out, indent=1))
