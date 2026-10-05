"""grade_po_tovlevel_window_v1.py -- closed-loop read of PO section 30 arms (paired, 50 seeds). PO worker, 2026-10-05.

ref = served run on the same games and seeds (same engine src); floor = served reseed (offset 1000). Verified truth.
Lines: TOV per box possession (P = FGA - OREB + TOV + 0.44 FTA, same in sim and truth), total bias, total MAE,
margin MAE, margin bias; days-since-start buckets d0-14 / d15+ when the game list spans them. 95% game-bootstrap
interval of (arm - ref) and the reseed move (floor - ref).

    .venv/Scripts/python.exe scripts/grade_po_tovlevel_window_v1.py --season 2024 --ref <dir> --floor <dir> \
        --arm T1=<dir> [--arm T2=<dir>] --out results/po_tovlevel/window_F1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402


def sim_game(p: Path) -> pd.DataFrame:
    g = pd.read_parquet(p / "games.parquet")
    out = pd.DataFrame({"game_id": g["game_id"]})
    out["total"] = g.home_pts + g.away_pts
    out["margin"] = g.home_pts - g.away_pts
    tov = g.home_tov + g.away_tov
    fga = sum(g[f"{s}_{k}"] for s in ("home", "away") for k in ("fga3", "fga2_rim", "fga2_jump"))
    poss = fga - (g.home_oreb + g.away_oreb) + tov + 0.44 * (g.home_fta + g.away_fta)
    out["tov"], out["poss"] = tov, poss
    return out.groupby("game_id").mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert_not_sealed(a.season)
    runs = {"ref": Path(a.ref), "floor": Path(a.floor)}
    runs.update({k: Path(v) for k, v in (x.split("=", 1) for x in a.arm)})
    pg = {k: sim_game(ROOT / v) for k, v in runs.items()}
    act = R.load_actual_games(a.season).set_index("game_id")
    tb = R.load_actual_team_box(a.season)
    ids = sorted(set.intersection(*[set(v.index) for v in pg.values()]) & set(act.index) & set(tb["game_id"]))
    A = act.loc[ids]
    tbi = tb[tb.game_id.isin(ids)].copy()
    tbi["P"] = tbi["fga"] - tbi["oreb"] + tbi["tov"] + 0.44 * tbi["fta"]
    act_tov = float(tbi["tov"].sum() / tbi["P"].sum())
    # days-since-start buckets (truth calendar)
    d = pd.to_datetime(A["game_date"] if "game_date" in A else A["date"])
    dss = (d - d.min()).dt.days.to_numpy()
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(ids), size=(2000, len(ids)))

    def lines(s):
        s = s.loc[ids]
        return {"TOV_per_poss": (s.tov.to_numpy(), s.poss.to_numpy()),
                "total_bias": s.total.to_numpy() - A.total.to_numpy(),
                "total_mae": np.abs(s.total.to_numpy() - A.total.to_numpy()),
                "margin_mae": np.abs(s.margin.to_numpy() - A.margin.to_numpy()),
                "margin_bias": s.margin.to_numpy() - A.margin.to_numpy()}

    def stat(v, idx=None):
        if isinstance(v, tuple):
            m, q = v
            return m[idx].sum(-1) / q[idx].sum(-1) if idx is not None else m.sum() / q.sum()
        return v[idx].mean(-1) if idx is not None else v.mean()

    L = {k: lines(v) for k, v in pg.items()}
    out = {"season": a.season, "n_games": len(ids), "actual_TOV_per_poss": act_tov, "runs": {k: str(v) for k, v in runs.items()},
           "arms": {}}
    for arm in [k for k in runs if k not in ("ref", "floor")]:
        res = {}
        for line in L["ref"]:
            r, x, f = (stat(L[k][line]) for k in ("ref", arm, "floor"))
            dd = stat(L[arm][line], B) - stat(L["ref"][line], B)
            res[line] = {"ref": float(r), "arm": float(x), "move": float(x - r),
                         "ci95": [float(np.quantile(dd, .025)), float(np.quantile(dd, .975))],
                         "reseed_move": float(f - r)}
        for lab, m in (("d0-14", dss <= 14), ("d15+", dss > 14)):
            if m.sum() >= 30:
                idx = np.where(m)[0]
                res[f"total_bias_{lab}"] = {"n": int(m.sum()), "ref": float(stat(L["ref"]["total_bias"], idx)),
                                            "arm": float(stat(L[arm]["total_bias"], idx)),
                                            "reseed_move": float(stat(L["floor"]["total_bias"], idx) - stat(L["ref"]["total_bias"], idx))}
        out["arms"][arm] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
