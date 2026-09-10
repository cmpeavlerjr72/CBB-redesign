#!/usr/bin/env python3
"""
Compare boosted_sims / improved / pace-experiment variants against the base
model on the exact same games (joined via date + sorted team slugs), using
base's final scores + closing lines as ground truth (variants don't carry
their own final.json / market odds in a directly usable form).

IMPORTANT: as established via file timestamps, boosted_sims and improved are
NOT live pregame predictions -- they were generated in large retroactive
batches (see report). This script still computes their accuracy for
completeness/reference, but the numbers are not a fair "would have performed"
estimate.
"""
import json
import numpy as np
import pandas as pd

SCRATCH = r"C:\Users\devuser\AppData\Local\Temp\claude\C--Users-devuser-CBB-clean-sheet\f90c786b-0ad5-48b7-b88a-f8ef6f7594ed\scratchpad"


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def bias(a, b):
    return float(np.mean(a - b))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


base = pd.read_parquet(f"{SCRATCH}/master_games.parquet")
base_final = base[base["has_final"]].copy()
base_final["actual_margin"] = base_final["finalA"] - base_final["finalB"]
base_final["actual_total"] = base_final["finalA"] + base_final["finalB"]
base_final["y_A"] = (base_final["finalA"] > base_final["finalB"]).astype(int)
base_final["market_margin_close"] = -base_final["close_spreadA"]

base_keep = base_final[[
    "join_key", "A_slug", "B_slug", "date", "actual_margin", "actual_total", "y_A",
    "market_margin_close", "close_total", "margin_p50", "total_p50", "A_win_prob",
]].rename(columns={
    "A_slug": "base_A_slug", "B_slug": "base_B_slug",
    "margin_p50": "base_margin_p50", "total_p50": "base_total_p50", "A_win_prob": "base_A_win_prob",
})


def score_variant(name, path):
    var = pd.read_parquet(path)
    if len(var) == 0:
        return {"name": name, "n_rows_in_variant_file": 0}
    m = var.merge(base_keep, on="join_key", how="inner", suffixes=("", "_base"))
    n_joined = len(m)
    if n_joined == 0:
        return {"name": name, "n_rows_in_variant_file": len(var), "n_joined": 0}

    same_orientation = m["A_slug"] == m["base_A_slug"]
    flipped = m["A_slug"] == m["base_B_slug"]
    ok = same_orientation | flipped
    m = m[ok].copy()
    same_orientation = same_orientation[ok]

    sign = np.where(same_orientation, 1.0, -1.0)
    m["var_margin_aligned"] = m["margin_p50"] * sign
    m["var_winprob_aligned"] = np.where(same_orientation, m["A_win_prob"], 1 - m["A_win_prob"])
    m["var_total"] = m["total_p50"]

    out = {
        "name": name,
        "n_rows_in_variant_file": int(len(var)),
        "n_joined_to_base_with_final": int(len(m)),
        "date_min": str(m["date"].min()) if len(m) else None,
        "date_max": str(m["date"].max()) if len(m) else None,
        "spread_mae_variant": mae(m["var_margin_aligned"], m["actual_margin"]),
        "spread_mae_base_same_games": mae(m["base_margin_p50"], m["actual_margin"]),
        "spread_bias_variant": bias(m["var_margin_aligned"], m["actual_margin"]),
        "spread_bias_base_same_games": bias(m["base_margin_p50"], m["actual_margin"]),
        "total_mae_variant": mae(m["var_total"], m["actual_total"]),
        "total_mae_base_same_games": mae(m["base_total_p50"], m["actual_total"]),
        "total_bias_variant": bias(m["var_total"], m["actual_total"]),
        "total_bias_base_same_games": bias(m["base_total_p50"], m["actual_total"]),
        "brier_variant": brier(m["var_winprob_aligned"].values, m["y_A"].values),
        "brier_base_same_games": brier(m["base_A_win_prob"].values, m["y_A"].values),
        "market_mae_spread_same_games": mae(m["market_margin_close"], m["actual_margin"]),
        "market_mae_total_same_games": mae(m["close_total_base"], m["actual_total"]),
    }
    return out


results = {}
results["boosted_sims"] = score_variant("boosted_sims", f"{SCRATCH}/boosted_games.parquet")
results["improved"] = score_variant("improved", f"{SCRATCH}/improved_games.parquet")
results["pace_experiment"] = score_variant("pace_experiment", f"{SCRATCH}/pace_games.parquet")

with open(f"{SCRATCH}/results_variants_out.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results, indent=2, default=str))
