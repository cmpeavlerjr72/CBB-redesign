"""Closed-loop read of free_throw section 18 winner on the F2 opening window (50 seeds, paired with served).

arm = results/ft_exposure/runs/F2_ftX1_o0; ref = results/player_day1/runs/F2_srv_o0 (same seeds, same engine src);
floor draw = F2_srv_o1000. Verified truth. Lines: pooled FT%, total bias, total MAE, margin MAE, margin bias, with
95% game-bootstrap intervals of (arm - ref) and the reseed move (o1000 - o0).
    .venv/Scripts/python.exe scripts/grade_ft_exposure_window_v1.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cbb_sim.eval import reference as R

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"ref": "results/player_day1/runs/F2_srv_o0", "floor": "results/player_day1/runs/F2_srv_o1000",
        "X1": "results/ft_exposure/runs/F2_ftX1_o0"}
act = R.load_actual_games(2025).set_index("game_id")
tb = R.load_actual_team_box(2025)
pg = {}
for k, p in RUNS.items():
    g = pd.read_parquet(ROOT / p / "games.parquet")
    g["total"] = g.home_pts + g.away_pts
    g["margin"] = g.home_pts - g.away_pts
    g["fta"] = g.home_fta + g.away_fta
    g["ftm"] = g.home_ftm + g.away_ftm
    pg[k] = g.groupby("game_id")[["total", "margin", "fta", "ftm"]].mean()
ids = sorted(set.intersection(*[set(v.index) for v in pg.values()]) & set(act.index))
A = act.loc[ids]
ta = tb[tb.game_id.isin(ids)]
act_ft = ta.ftm.sum() / ta.fta.sum()
rng = np.random.default_rng(0)
B = rng.integers(0, len(ids), size=(2000, len(ids)))


def lines(s):
    s = s.loc[ids]
    return {"FT%": (s.ftm.to_numpy(), s.fta.to_numpy()),
            "total_bias": s.total.to_numpy() - A.total.to_numpy(),
            "total_abs": np.abs(s.total.to_numpy() - A.total.to_numpy()),
            "margin_abs": np.abs(s.margin.to_numpy() - A.margin.to_numpy()),
            "margin_bias": s.margin.to_numpy() - A.margin.to_numpy()}


def stat(v, idx=None):
    if isinstance(v, tuple):
        m, a = v
        return m[idx].sum(1) / a[idx].sum(1) if idx is not None else m.sum() / a.sum()
    return v[idx].mean(1) if idx is not None else v.mean()


L = {k: lines(v) for k, v in pg.items()}
out = {"n_games": len(ids), "actual_FT%": act_ft}
for line in L["ref"]:
    r, x, f = (stat(L[k][line]) for k in ("ref", "X1", "floor"))
    d = stat(L["X1"][line], B) - stat(L["ref"][line], B)
    out[line] = {"ref": r, "X1": x, "move": x - r, "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                 "reseed_move": f - r}
print(json.dumps(out, indent=1, default=float))
(ROOT / "results/ft_exposure/window_closed_loop_v1.json").write_text(json.dumps(out, indent=1, default=float))
