#!/usr/bin/env python
"""grade_foul_team_loop_resp_v1.py -- reported loop line of possession_outcome experiments.md s34.4: per-team
responsiveness of the closed loop. Teams in quintiles of their PRIOR-season league-centred foul committing rate
(defence view: FTA/FGA conceded) and drawn rate (offence view: FTA/FGA earned), from `build_foul_team_feats_v1.team_games`.
Sim (mean over seeds, every run given) vs verified team box; slope = OLS of sim quintile means on actual quintile means.
Windows: all season and d0-14 (engine inputs' days_since_start).

    .venv/Scripts/python.exe scripts/grade_foul_team_loop_resp_v1.py --season 2025 --input-dir data/processed/models/engine_v3 \
        --run ref=results/engine_v0/fcal_F2_ctrl_s50 --run A2t=results/engine_v0/fteam_F2_A2t_s50 --out results/foul_r2/resp_F2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_foul_team_feats_v1 as BF  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402


def team_long(g: pd.DataFrame) -> pd.DataFrame:
    """per (game, team): own FTA/FGA (offence view) and opponent's (defence view)."""
    rows = []
    for s, o in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({"game_id": g["game_id"], "team_id": g[f"{s}_team_id"],
                                  "fta": g[f"{s}_fta"], "fga": g[f"{s}_fga"],
                                  "fta_c": g[f"{o}_fta"], "fga_c": g[f"{o}_fga"]}))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert_not_sealed(a.season)
    fold = "F2" if a.season == 2025 else "F1"
    inp = EngineInputs.load(a.input_dir, f"{fold}_{a.season}")
    gm = inp.games[["game_id", "home_team_id", "away_team_id"]].copy()
    gm["dss"] = inp.team_static[:, 0, inp.team_names["days_since_start"]]
    tb = R.load_actual_team_box(a.season)
    act = tb.pivot_table(index="game_id", columns="team_home_away", values=["fta", "fga"], aggfunc="sum")
    act.columns = [f"{s}_{k}" for k, s in act.columns]
    act = gm.merge(act.reset_index(), on="game_id")
    sims = {}
    for kv in a.run:
        k, p = kv.split("=", 1)
        g = pd.read_parquet(ROOT / p / "games.parquet")
        for s in ("home", "away"):
            g[f"{s}_fga"] = g[f"{s}_fga3"] + g[f"{s}_fga2_rim"] + g[f"{s}_fga2_jump"]
        g = g.groupby("game_id")[["home_fta", "away_fta", "home_fga", "away_fga"]].mean().reset_index()
        sims[k] = gm.merge(g, on="game_id")
    ids = set(act.game_id)
    for v in sims.values():
        ids &= set(v.game_id)
    # prior-season centred rates (x100 per possession)
    tg = BF.team_games()
    fin = tg.groupby(["season", "team_id"])[["fc", "pd_", "fd", "po"]].sum().reset_index()
    lf = tg.groupby("season")[["fc", "pd_", "fd", "po"]].sum()
    fin["prior_def"] = 100 * (fin.fc / fin.pd_ - (lf.fc / lf.pd_).reindex(fin.season).to_numpy())
    fin["prior_off"] = 100 * (fin.fd / fin.po - (lf.fd / lf.po).reindex(fin.season).to_numpy())
    pri = fin[fin.season == a.season - 1].set_index("team_id")[["prior_def", "prior_off"]]
    out = {"season": a.season, "fold": fold, "n_games": len(ids), "windows": {}}
    for wn, lo, hi in (("all", 0, 1e9), ("d0-14", 0, 14)):
        sel = lambda df: df[df.game_id.isin(ids) & (df.dss >= lo) & (df.dss <= hi)]
        A = team_long(sel(act))
        S = {k: team_long(sel(v)) for k, v in sims.items()}
        res = {}
        for view, num, den, pcol in (("offence_drawn", "fta", "fga", "prior_off"), ("defence_committing", "fta_c", "fga_c", "prior_def")):
            ta = A.groupby("team_id")[[num, den]].sum()
            ta = ta[ta[den] >= (60 if wn == "d0-14" else 300)]
            t = pd.DataFrame({"actual": ta[num] / ta[den]})
            for k, s in S.items():
                ts = s.groupby("team_id")[[num, den]].sum()
                t[k] = (ts[num] / ts[den]).reindex(t.index)
            t = t.join(pri[pcol], how="inner").dropna()
            t["q"] = pd.qcut(t[pcol], 5, labels=False)
            q = t.groupby("q")[["actual"] + list(S)].mean()
            r = {"n_teams": int(len(t)), "actual_q": q["actual"].round(4).tolist()}
            for k in S:
                r[k] = {"q": q[k].round(4).tolist(), "slope_q": float(np.polyfit(q["actual"], q[k], 1)[0]),
                        "team_slope": float(np.polyfit(t["actual"], t[k], 1)[0])}
            res[view] = r
        out["windows"][wn] = res
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for wn, res in out["windows"].items():
        for view, r in res.items():
            print(f"{fold} {wn:6s} {view:19s} n={r['n_teams']} actual {r['actual_q']} | " +
                  " | ".join(f"{k} {r[k]['q']} slope_q {r[k]['slope_q']:.3f}" for k in sims))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
