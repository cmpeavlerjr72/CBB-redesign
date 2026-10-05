#!/usr/bin/env python
"""
grade_player_day1_v2.py -- segment / per-type / player tables for the player-layer day-1 bake-off (docs/models/player_day1/
experiments.md section 1). Companion of grade_player_day1_v1.py (headline + paired deltas); same blind scoring for every run.

Per run, window games with verified finals:
  * total MAE / bias by site (home-court vs neutral) and by first-game status (no in-season history in the SERVED inputs);
  * responsiveness: OLS slope of actual on predicted total, and actual vs predicted by predicted-total quintile;
  * pooled make rates (rim, jump2, three, FT) and per-team-game volumes vs truth (`truth/team_game_shots_v2`, ev_* columns);
  * player level (props readiness): share of actual box minutes played by players the sim names, and minutes / points MAE on
    actual player-games (>= 10 min) whose player the sim names (`truth/player_game_v2`).

    python scripts/grade_player_day1_v2.py --season 2025 --served-inputs data/processed/models/engine_v3 --runs <dir>,<dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402

TR = ROOT / "data/processed/truth"


def one(run: str, season: int, first: pd.Series, neutral: pd.Series) -> dict:
    games = pd.read_parquet(Path(run) / "games.parquet")
    s, _ = G.build_grading_frame(games, season)
    s = s.set_index("game_id")
    et = s["sim_total_mean"] - s["total"]
    out = {"n": int(len(s))}
    for name, m in (("neutral", neutral.reindex(s.index).fillna(False).to_numpy()),
                    ("first_game", first.reindex(s.index).fillna(False).to_numpy())):
        for lab, mm in ((name, m), (f"not_{name}", ~m)):
            out[f"total_mae|{lab}"] = float(et[mm].abs().mean()); out[f"total_bias|{lab}"] = float(et[mm].mean())
            out[f"n|{lab}"] = int(mm.sum())
    x, y = s["sim_total_mean"].to_numpy(), s["total"].to_numpy()
    out["resp_slope_actual_on_pred_total"] = float(np.polyfit(x, y, 1)[0])
    q = pd.qcut(x, 5, labels=False)
    out["resp_quintiles_pred_vs_actual"] = [[float(x[q == k].mean()), float(y[q == k].mean())] for k in range(5)]
    # per-type pooled
    tr = pd.read_parquet(TR / "team_game_shots_v2.parquet")
    tr = tr[tr["game_id"].isin(s.index)]
    g = games[games["game_id"].isin(s.index)]
    nseed = g["seed"].nunique()
    for c_sim, c_tr in (("fga2_rim", "ev_fga_rim"), ("fga2_jump", "ev_fga_jump2"), ("fga3", "ev_fga_3"), ("fta", "ev_fta")):
        mk_sim = {"fga2_rim": "fgm2_rim", "fga2_jump": "fgm2_jump", "fga3": "fgm3", "fta": "ftm"}[c_sim]
        mk_tr = c_tr.replace("fga", "fgm").replace("fta", "ftm")
        a_s = g[f"home_{c_sim}"].sum() + g[f"away_{c_sim}"].sum()
        m_s = g[f"home_{mk_sim}"].sum() + g[f"away_{mk_sim}"].sum()
        out[f"make|{c_sim}"] = [float(m_s / a_s), float(tr[mk_tr].sum() / tr[c_tr].sum())]
        out[f"per_team_game|{c_sim}"] = [float(a_s / (2 * len(g))), float(tr[c_tr].sum() / len(tr))]
    # player level
    pp = Path(run) / "players.parquet"
    if pp.exists():
        p = pd.read_parquet(pp, columns=["game_id", "seed", "athlete_id", "minutes", "pts"])
        p = p[(p["athlete_id"] > 0) & p["game_id"].isin(s.index)]
        sm = p.groupby(["game_id", "athlete_id"])[["minutes", "pts"]].sum() / nseed
        pg = pd.read_parquet(TR / "player_game_v2.parquet", columns=["game_id", "athlete_id", "minutes", "pts", "season"])
        pg = pg[(pg["season"] == season) & pg["game_id"].isin(s.index) & (pg["minutes"] > 0)]
        pg = pg.set_index(["game_id", "athlete_id"])
        named = pg.index.isin(sm.index)
        out["player_minutes_share_named"] = float(pg.loc[named, "minutes"].sum() / pg["minutes"].sum())
        rot = pg[named & (pg["minutes"] >= 10)]
        j = sm.reindex(rot.index)
        out["player_n_named_rotation_games"] = int(len(rot))
        out["player_n_rotation_games"] = int((pg["minutes"] >= 10).sum())
        out["player_minutes_mae_named"] = float((j["minutes"] - rot["minutes"]).abs().mean()) if len(rot) else None
        out["player_pts_mae_named"] = float((j["pts"] - rot["pts"]).abs().mean()) if len(rot) else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--served-inputs", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fold = "F2" if a.season == 2025 else "F1"
    gi = pd.read_parquet(Path(a.served_inputs) / f"games_{fold}_{a.season}.parquet")
    z = np.load(Path(a.served_inputs) / f"arrays_{fold}_{a.season}.npz")
    idx = gi["game_id"].astype("int64")
    first = pd.Series(~(z["roster_cbbd"] > 0).any(axis=(1, 2)), index=idx)
    neutral = pd.Series(gi["neutral"].astype(float).to_numpy() > 0.5, index=idx)
    res = {Path(r).name: one(r, a.season, first, neutral) for r in a.runs.split(",") if r}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
