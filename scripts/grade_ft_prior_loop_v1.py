"""grade_ft_prior_loop_v1.py -- paired closed-loop read for free_throw section 20 (descriptive P1 loop).

Generalises grade_ft_exposure_window_v1.py (not edited): arm vs ref on the same game ids / seeds, floor = ref reseeded.
Verified truth. Lines: pooled FT%, total bias, total MAE, margin MAE, margin bias; 95% game-bootstrap CI of arm - ref.

    .venv/Scripts/python.exe scripts/grade_ft_prior_loop_v1.py --season 2025 --ref <dir> --floor <dir> --arm <dir> --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import reference as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("ref", "floor", "arm", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--season", type=int, required=True)
    a = ap.parse_args()
    act = R.load_actual_games(a.season).set_index("game_id")
    tb = R.load_actual_team_box(a.season)
    pg = {}
    for k in ("ref", "floor", "arm"):
        g = pd.read_parquet(ROOT / getattr(a, k) / "games.parquet")
        g["total"] = g.home_pts + g.away_pts
        g["margin"] = g.home_pts - g.away_pts
        g["fta"] = g.home_fta + g.away_fta
        g["ftm"] = g.home_ftm + g.away_ftm
        pg[k] = g.groupby("game_id")[["total", "margin", "fta", "ftm"]].mean()
    ids = sorted(set.intersection(*[set(v.index) for v in pg.values()]) & set(act.index))
    A = act.loc[ids]
    ta = tb[tb.game_id.isin(ids)]
    B = np.random.default_rng(0).integers(0, len(ids), size=(2000, len(ids)))

    def lines(s):
        s = s.loc[ids]
        return {"FT%": (s.ftm.to_numpy(), s.fta.to_numpy()),
                "total_bias": s.total.to_numpy() - A.total.to_numpy(),
                "total_abs": np.abs(s.total.to_numpy() - A.total.to_numpy()),
                "margin_abs": np.abs(s.margin.to_numpy() - A.margin.to_numpy()),
                "margin_bias": s.margin.to_numpy() - A.margin.to_numpy()}

    def stat(v, idx=None):
        if isinstance(v, tuple):
            m, n = v
            return m[idx].sum(1) / n[idx].sum(1) if idx is not None else m.sum() / n.sum()
        return v[idx].mean(1) if idx is not None else v.mean()

    L = {k: lines(v) for k, v in pg.items()}
    out = {"n_games": len(ids), "actual_FT%": float(ta.ftm.sum() / ta.fta.sum()), "runs": vars(a)}
    for line in L["ref"]:
        r, x, f = (float(stat(L[k][line])) for k in ("ref", "arm", "floor"))
        d = stat(L["arm"][line], B) - stat(L["ref"][line], B)
        out[line] = {"ref": r, "arm": x, "move": x - r, "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                     "reseed_move": f - r}
    print(json.dumps(out, indent=1, default=float))
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
