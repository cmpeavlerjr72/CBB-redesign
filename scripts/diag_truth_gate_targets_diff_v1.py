#!/usr/bin/env python
"""Lane H: old vs new gate_targets (data/reference vs data/reference/verified_v1), all rows."""
import numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "data/reference"
K = ["breakdown", "group", "side", "metric"]
rows = []; tot = {}
for s in (2022, 2023, 2024, 2025):
    o = pd.read_parquet(R / f"gate_targets_{s}.parquet"); n = pd.read_parquet(R / f"verified_v1/gate_targets_{s}.parquet")
    o["group"] = o["group"].astype(str); n["group"] = n["group"].astype(str)
    m = o.merge(n, on=K, how="outer", suffixes=("_old", "_new"), indicator=True)
    ch = m[(m._merge != "both") | ~np.isclose(m.value_old, m.value_new, rtol=1e-9, atol=1e-12, equal_nan=True)]
    tot[s] = (len(m), len(ch), sorted(ch.metric.unique()))
    a = m[(m.breakdown == "season")]
    for met in sorted(ch.metric.unique()):
        x = a[(a.metric == met)]
        if len(x): rows.append({"season": s, "metric": met, "side": x.side.iloc[0], "old": x.value_old.iloc[0], "new": x.value_new.iloc[0], "n_old": x.n_old.iloc[0] if "n_old" in x else np.nan, "n_new": x.n_new.iloc[0] if "n_new" in x else np.nan})
    # any season-all metric unchanged but in team rows? count changed by breakdown
    print(s, "rows", len(m), "changed", len(ch), ch.groupby("breakdown").size().to_dict(), "metrics", sorted(ch.metric.unique()))
d = pd.DataFrame(rows); pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
d.to_csv(R.parent / "processed/truth/truth_gate_targets_old_vs_new_v1.csv", index=False)
