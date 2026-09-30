#!/usr/bin/env python
"""diag_own_ratings_asof_parity_v1.py -- does build_own_ratings_asof_v1.py reproduce the stored own_ratings rows?

    .venv/Scripts/python.exe scripts/diag_own_ratings_asof_parity_v1.py --season 2025 --dates 2024-11-12,2025-01-16,2025-03-04
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import build_own_ratings_asof_v1 as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=2025)
ap.add_argument("--dates", default="2024-11-12,2025-01-16,2025-03-04")
ap.add_argument("--root", default=str(REPO))
ap.add_argument("--teams-source", default="tg")
ap.add_argument("--out", default=None)
a = ap.parse_args()
root = Path(a.root)
stored = pd.read_parquet(root / f"data/processed/ratings/own_ratings_{a.season}.parquet")
stored["as_of_date"] = pd.to_datetime(stored["as_of_date"])
cache: dict = {}
res = []
for d in a.dates.split(","):
    new, prov = E.asof_ratings(a.season, d, root, a.teams_source, 2022, cache)
    old = stored[stored["as_of_date"] == pd.Timestamp(d)].sort_values("team_id").reset_index(drop=True)
    new = new.sort_values("team_id").reset_index(drop=True)
    r = {"date": d, "teams_new": len(new), "teams_stored": len(old),
         "same_team_set": bool(np.array_equal(new["team_id"].to_numpy(), old["team_id"].to_numpy())),
         "n_source_games": prov["n_source_games_in_season"], "latest_source_game": prov["latest_source_game_date"], "max_abs_diff": {}}
    if r["same_team_set"]:
        for c in [c for c in old.columns if c not in ("season", "as_of_date", "team_id")]:
            x, y = new[c].to_numpy(dtype=float), old[c].to_numpy(dtype=float)
            r["max_abs_diff"][c] = float(np.nanmax(np.abs(x - y))) if len(x) else 0.0
        r["max_over_all_columns"] = max(r["max_abs_diff"].values())
        r["bitwise_equal_columns"] = int(sum(np.array_equal(new[c].to_numpy(dtype=float), old[c].to_numpy(dtype=float), equal_nan=True)
                                             for c in r["max_abs_diff"]))
        r["n_columns"] = len(r["max_abs_diff"])
    res.append(r)
    print(json.dumps({k: v for k, v in r.items() if k != "max_abs_diff"}))
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
