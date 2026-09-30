"""diag_g1_unknown_box_v1.py -- game-by-game check of each `unknown` class
against the box possession estimator.

Lane B follow-up, 2026-09-30.  DIAGNOSTIC ONLY.  For every season 2022-2025,
per game: pbp possession count per team-game (possessions_v2) minus the box
estimator (mean of FGA - OREB + TOV + 0.44 FTA over the two sides, the G1
truth), regressed on the per-team-game count of each `unknown` class
(results/g1g5_diag/unknown_poss_classified.parquet) and of the and-one
missed-FT same-team restarts.  A spurious possession the box cannot see moves
(count - estimator) by exactly +1 per team-game unit; a possession the box
also counts moves it by 0.  Writes results/g1g5_diag/unknown_box.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.ratings import own_ratings as orat  # noqa: E402

OUT = ROOT / "results/g1g5_diag"
R = pd.read_parquet(OUT / "unknown_poss_classified.parquet")
univ = orat.load_universe()
res = {}
allrows = []
for s in (2022, 2023, 2024, 2025):
    p = pd.read_parquet(ROOT / f"data/processed/possessions_v2/possessions_{s}.parquet")
    cnt = p.groupby(["game_id", "offense_team_id"]).size().groupby("game_id").mean()
    nt = p.groupby("game_id")["offense_team_id"].nunique()
    tg = orat.load_team_games(univ, [s])
    est = tg.drop_duplicates("game_id").set_index("game_id")["game_poss"]
    A = pd.read_parquet(OUT / f"andone_{s}.parquet")
    A["restart"] = (~A["ft_made"]) & (A["next_cls"] == "OREB") & (A["next_team"] == A["shooter_side"])
    rs = A.groupby("game_id")["restart"].sum() / 2.0
    X = R[R["season"] == s].pivot_table(index="game_id", columns="cls", values="period", aggfunc="size", fill_value=0) / 2.0
    ids = cnt.index[(nt.reindex(cnt.index) == 2)].intersection(est.dropna().index)
    X = X.reindex(ids, fill_value=0)
    X["E_andone_missFT_OREB_restart"] = rs.reindex(ids, fill_value=0)
    y = (cnt.loc[ids] - est.loc[ids]).to_numpy()
    M = np.column_stack([np.ones(len(ids)), X.to_numpy()])
    b, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = y - M @ b
    se = np.sqrt(np.diag(np.linalg.inv(M.T @ M)) * resid.var())
    res[s] = {"n_games": int(len(ids)), "mean_gap": float(y.mean()), "intercept": float(b[0]),
              "slopes": {c: [float(b[j + 1]), float(se[j + 1])] for j, c in enumerate(X.columns)},
              "class_means_per_team_game": {c: float(X[c].mean()) for c in X.columns}}
    d = X.copy(); d["y"] = y; d["season"] = s; allrows.append(d)
D = pd.concat(allrows).fillna(0)
cols = [c for c in D.columns if c not in ("y", "season")]
M = np.column_stack([np.ones(len(D))] + [D[c].to_numpy() for c in cols] + [(D["season"] == s).to_numpy(float) for s in (2023, 2024, 2025)])
b, *_ = np.linalg.lstsq(M, D["y"].to_numpy(), rcond=None)
resid = D["y"].to_numpy() - M @ b
se = np.sqrt(np.diag(np.linalg.inv(M.T @ M)) * resid.var())
res["pooled"] = {"n_games": int(len(D)), "slopes": {c: [float(b[j + 1]), float(se[j + 1])] for j, c in enumerate(cols)}}
(OUT / "unknown_box.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
for k, v in res.items():
    print(k, v.get("n_games"), round(v.get("mean_gap", float("nan")), 3) if "mean_gap" in v else "")
    for c, (sl, e) in v["slopes"].items():
        print(f"   {c:40s} slope {sl:+.3f} (se {e:.3f})  mean/tg {v.get('class_means_per_team_game', {}).get(c, float('nan')):.3f}")
