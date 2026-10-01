"""grade_chance_time_loop_v1.py -- chance_time POST-HOC closed-loop diagnostic grader (lane I, 2026-09-30).

One grader for every arm's tap dir (diag_ppp_tap_v1.py output; same 500 verified stride games):
box lines on the sample (make rate per class, eFG, points per team-game, possessions, rim / three share,
TOV per possession, FTA/FGA, OREB%), the sim's mean fg_make p per class, chance-1 transition share
among FGA, chance-2+ elapsed quantiles, and total bias vs verified finals on the same games.
Paired move = arm - R (same seeds 0-31); floor = |R(seeds 100-131) - R(seeds 0-31)| per line.

Usage: grade_chance_time_loop_v1.py <out_json> R=<dir> C12=<dir> C2=<dir> Rfloor=<dir>
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CBB_TRUTH", "verified_v1")
from cbb_sim.eval import reference as R  # noqa: E402

OUT = Path(sys.argv[1])
DIRS = dict(a.split("=", 1) for a in sys.argv[2:])
games = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")
gid_of = games["game_id"].to_numpy()
act = R.load_actual_games(2025).set_index("game_id")
box = R.load_actual_team_box(2025)


def lines(d: Path) -> dict:
    g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(d / "games_*.parquet")))], ignore_index=True)
    if "game_id" not in g:
        g["game_id"] = gid_of[g["gidx"].to_numpy()]
    S = lambda c: float(g[f"home_{c}"].sum() + g[f"away_{c}"].sum())
    fga = S("fga2_rim") + S("fga2_jump") + S("fga3")
    n = len(g)
    out = {
        "n_sims": n, "seeds": [int(g["seed"].min()), int(g["seed"].max())],
        "make_rim": S("fgm2_rim") / S("fga2_rim"), "make_jump2": S("fgm2_jump") / S("fga2_jump"),
        "make_three": S("fgm3") / S("fga3"),
        "efg": (S("fgm2_rim") + S("fgm2_jump") + 1.5 * S("fgm3")) / fga,
        "pts_per_game": float((g["home_pts"] + g["away_pts"]).mean()),
        "possessions": float(g["possessions"].mean()),
        "rim_share": S("fga2_rim") / fga, "three_share": S("fga3") / fga,
        "tov_per_poss": S("tov") / (2 * g["possessions"].sum()),
        "fta_per_fga": S("fta") / fga, "ft_pct": S("ftm") / S("fta"),
        "oreb_pct": S("oreb") / (S("oreb") + S("dreb")),
    }
    m = g.groupby("game_id")[["home_pts", "away_pts"]].mean()
    m = m[m.index.isin(act.index)]
    out["total_bias_sample"] = float((m["home_pts"] + m["away_pts"] - act.loc[m.index, "total"]).mean())
    out["n_games_graded"] = int(len(m))
    for cls in ("FGA_rim", "FGA_jump2", "FGA_3"):
        f = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(str(d / f"fg_{cls}_*.parquet")))],
                      ignore_index=True)
        c1 = f["chance_number"] == 1
        out[f"mean_p_{cls}"] = float(f["p"].mean())
        out[f"trans_share_c1_{cls}"] = float(f.loc[c1, "is_transition_f"].mean())
        out[f"elapsed_q_c1_{cls}"] = np.quantile(f.loc[c1, "chance_elapsed_s"], [.1, .25, .5, .75, .9]).tolist()
        out[f"elapsed_q_c2p_{cls}"] = np.quantile(f.loc[~c1, "chance_elapsed_s"], [.1, .25, .5, .75, .9]).tolist()
        out[f"mean_p_c2p_{cls}"] = float(f.loc[~c1, "p"].mean())
    return out


def actual_lines(sample_ids) -> dict:
    b = box[box["game_id"].isin(sample_ids)]
    t = R.load_team_shot_truth(2025)
    t = t[t["game_id"].isin(sample_ids)]
    a = act.loc[act.index.isin(sample_ids)]
    return {"pts_per_game": float(a["total"].mean()), "possessions_box_est": float(b["poss_team"].sum() / len(a) / 2),
            "make_rim": float(t["ev_fgm_rim"].sum() / t["ev_fga_rim"].sum()),
            "make_jump2": float(t["ev_fgm_jump2"].sum() / t["ev_fga_jump2"].sum()),
            "make_three": float(b["tpm"].sum() / b["tpa"].sum()),
            "efg": float((b["fgm"].sum() + 0.5 * b["tpm"].sum()) / b["fga"].sum()),
            "rim_share": float(t["ev_fga_rim"].sum() / t["ev_fga"].sum()),
            "three_share": float(b["tpa"].sum() / b["fga"].sum()),
            "fta_per_fga": float(b["fta"].sum() / b["fga"].sum()), "ft_pct": float(b["ftm"].sum() / b["fta"].sum()),
            "oreb_pct": float(b["oreb"].sum() / (b["oreb"].sum() + b["opp_dreb"].sum()))}


res = {k: lines(Path(v)) for k, v in DIRS.items()}
ids = pd.concat([pd.read_parquet(f, columns=["gidx"]) for f in glob.glob(str(Path(DIRS["R"]) / "games_*.parquet"))])
res["ACTUAL"] = actual_lines(set(gid_of[ids["gidx"].unique()]))
num = [k for k, v in res["R"].items() if isinstance(v, float)]
res["paired"] = {}
for arm in [a for a in DIRS if a not in ("R", "Rfloor")]:
    res["paired"][arm] = {}
    for k in num:
        mv = res[arm][k] - res["R"][k]
        fl = abs(res["Rfloor"][k] - res["R"][k]) if "Rfloor" in res else float("nan")
        res["paired"][arm][k] = {"move": mv, "floor": fl, "move_over_floor": mv / fl if fl > 0 else float("inf")}
OUT.write_text(json.dumps(res, indent=1, default=float))
for k in num:
    row = [f"{res[a][k]:.4f}" for a in DIRS] + [f"{res['ACTUAL'].get(k, float('nan')):.4f}"]
    print(f"{k:28s}", " ".join(row))
