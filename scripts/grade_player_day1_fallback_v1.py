#!/usr/bin/env python
"""
grade_player_day1_fallback_v1.py -- blind grader for the roster-less-team fallback bake-off (docs/models/player_day1/experiments.md
section 4). One function scores every arm. Games = window games with at least one treated team; player level = treated teams only.

  python scripts/grade_player_day1_fallback_v1.py --fold F2 --season 2025 --ref results/player_day1/fb/F2_R0_o0 \
      --reseed results/player_day1/fb/F2_R0_o1000 --arms results/player_day1/fb/F2_R1_o0,results/player_day1/fb/F2_R2_o0 --out <json>

Per run: total MAE / bias, margin MAE, brier on the games; paired delta vs the reference with a 95% game-bootstrap interval; the
reseed floor = |ref o0 - ref o1000| (min 0.06). Player level on treated teams' player-games: coverage = share of actual box
minutes played by sim-named players; minutes MAE on named >=10 min player-games; precision = share of sim-named player-games with
>=10 actual minutes; minutes MAE over ALL actual >=10 min player-games with an unnamed player predicted 0 ("all_rot_mae").
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
import grade_player_day1_v1 as G1  # noqa: E402

TR = ROOT / "data/processed/truth"


def player_level(run: str, season: int, ids, treated: set) -> dict:
    p = pd.read_parquet(Path(run) / "players.parquet", columns=["game_id", "seed", "athlete_id", "team_id", "minutes"])
    nseed = p["seed"].nunique()
    p = p[(p["athlete_id"] > 0) & p["game_id"].isin(ids) & p["team_id"].isin(treated)]
    sm = p.groupby(["game_id", "athlete_id"])["minutes"].sum() / nseed
    pg = pd.read_parquet(TR / "player_game_v2.parquet", columns=["game_id", "athlete_id", "team_id", "minutes", "season"])
    pg = pg[(pg["season"] == season) & pg["game_id"].isin(ids) & pg["team_id"].isin(treated) & (pg["minutes"] > 0)]
    pg = pg.set_index(["game_id", "athlete_id"])
    named = pg.index.isin(sm.index)
    rot = pg["minutes"] >= 10
    pred = sm.reindex(pg.index).fillna(0.0)
    out = {"n_player_games": int(len(pg)), "n_rotation_games": int(rot.sum()),
           "coverage_minutes_share_named": float(pg.loc[named, "minutes"].sum() / pg["minutes"].sum()),
           "all_rot_mae": float((pred[rot] - pg.loc[rot, "minutes"]).abs().mean())}
    nr = named & rot.to_numpy()
    out["named_rot_mae"] = float((pred[nr] - pg.loc[nr, "minutes"]).abs().mean()) if nr.sum() else None
    out["n_named_rot"] = int(nr.sum())
    if len(sm):
        act = pg["minutes"].reindex(sm.index).fillna(0.0)
        out["precision_named_ge10"] = float((act >= 10).mean())
        out["n_named"] = int(len(sm))
    else:
        out["precision_named_ge10"] = None
        out["n_named"] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--reseed", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--treated-json", required=True, help="assemble_report.json of the ref arm (treated_teams)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    treated = set(json.loads(Path(a.treated_json).read_text())["treated_teams"])
    runs = {"ref": a.ref, "reseed": a.reseed, **{Path(x).name: x for x in a.arms.split(",") if x}}
    fr = {k: G1.frame(v, a.season) for k, v in runs.items()}
    gi = pd.read_parquet(Path(a.treated_json).parent / f"games_{a.fold}_{a.season}.parquet",
                         columns=["game_id", "home_team_id", "away_team_id"]).drop_duplicates("game_id")
    gi["game_id"] = gi["game_id"].astype("int64")
    gi = gi.set_index("game_id")
    gi = gi.loc[gi.index.isin(fr["ref"].index)]       # simulated window games only (the input file holds the full season)
    nt = gi["home_team_id"].isin(treated).astype(int) + gi["away_team_id"].isin(treated).astype(int)
    res = {"treated_n": len(treated), "runs": runs, "metrics": {}, "pairs": {}, "player": {}}
    for seg, sel in (("all", nt >= 1), ("both_treated", nt == 2), ("one_treated", nt == 1)):
        ids = nt.index[sel.to_numpy()]
        for k, s in fr.items():
            ss = s.loc[s.index.intersection(ids)]
            res["metrics"][f"{k}|{seg}"] = G1.metrics(ss) if len(ss) else None
        r = fr["ref"].loc[fr["ref"].index.intersection(ids)]
        for k in [x for x in fr if x != "ref"]:
            res["pairs"][f"{k}_vs_ref|{seg}"] = G1.pair(fr[k], r) if len(r) else None
    ids = set(nt.index[(nt >= 1).to_numpy()])
    for k, v in runs.items():
        res["player"][k] = player_level(v, a.season, ids, treated)
    a_ = res["metrics"]["ref|all"]["total_mae"]
    b_ = res["metrics"]["reseed|all"]["total_mae"]
    res["floor_total_mae"] = max(abs(a_ - b_), 0.06)
    res["floor_raw"] = abs(a_ - b_)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"floor": res["floor_total_mae"], "metrics_all": {k: v for k, v in res["metrics"].items() if k.endswith("|all")},
                      "pairs_all": {k: {x: y for x, y in v.items() if x.startswith("d_total") or x.startswith("d_margin_mae")} for k, v in res["pairs"].items() if k.endswith("|all")},
                      "player": res["player"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
