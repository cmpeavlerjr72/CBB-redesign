"""diag_late_game_r2_levels_v1.py -- late-game ROUND 2: the multi-level evidence tables.

Pre-registration: `docs/models/late_game/experiments.md` section 4.6. Reported
lines only (no decision line lives here); one code path for every run.

  1. KERNEL: P(regulation tie | |home margin at the first H2 possession starting
     at <= 120 s| = k), k = 0..6, per run, against the actual (pbp chances for
     the anchor, verified finals for OT).
  2. R5 closed loop: window possessions per simulation by the same anchor.
  3. PER GAME: paired delta of each game's 25-seed tie count vs R.
  4. PER TEAM: OT rate per team (home or away), cells with >= 200 simulations.

    .venv/Scripts/python.exe scripts/diag_late_game_r2_levels_v1.py --ref lg2_R_s25 \
        --runs lg2_Rfloor_s25 lg2_W_C2_s25 ... --out results/late_game/round2/levels.json
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

from cbb_sim.eval import reference as REF                             # noqa: E402

RES = ROOT / "results/engine_v0"
CHANCES = ROOT / "data/processed/possessions_v2/chances_2025.parquet"
MIN_TEAM_SIMS = 200


def load(tag):
    g = pd.read_parquet(RES / tag / "games.parquet")
    t = pd.read_parquet(RES / tag / "tap_sims.parquet")
    return g.merge(t, on=["game_id", "seed"], validate="one_to_one")


def actual_anchor() -> pd.DataFrame:
    c = pd.read_parquet(CHANCES, columns=["game_id", "period", "poss_index", "chance_number",
                                          "start_clock", "start_score_diff"])
    c = c[(c["period"] == 2) & (c["chance_number"] == 1)]
    w = c[c["start_clock"] <= 120].sort_values(["game_id", "poss_index"])
    first = w.groupby("game_id").first()
    win = w[w["start_score_diff"].abs() <= 6].groupby("game_id").size()
    act = REF.load_actual_games(2025)[["game_id", "n_periods"]].set_index("game_id")
    out = pd.DataFrame({"m120": first["start_score_diff"].abs()}).join(act, how="inner")
    out["window_poss"] = win.reindex(out.index).fillna(0)
    out["ot"] = out["n_periods"] > 2
    return out


def kernel(g: pd.DataFrame) -> dict:
    m = g["home_margin_120"].abs()
    ot = g["n_periods"] > 2
    out = {}
    for k in range(7):
        s = m == k
        out[str(k)] = {"n": int(s.sum()), "p_tie": float(ot[s].mean()) if s.any() else None,
                       "window_poss": float(g.loc[s, "window_poss"].mean()) if s.any() else None}
    return out


def per_game(ref: pd.DataFrame, g: pd.DataFrame) -> dict:
    a = (ref["n_periods"] > 2).groupby(ref["game_id"]).sum()
    b = (g["n_periods"] > 2).groupby(g["game_id"]).sum().reindex(a.index)
    d = b - a
    return {"n_games": int(len(d)), "mean_delta_ties_per_game": float(d.mean()),
            "share_games_up": float((d > 0).mean()), "share_games_down": float((d < 0).mean()),
            "share_games_same": float((d == 0).mean())}


def per_team(ref: pd.DataFrame, g: pd.DataFrame) -> dict:
    eg = pd.read_parquet(ROOT / "data/processed/models/engine/games_F2_2025.parquet",
                         columns=["game_id", "home_team_id", "away_team_id"])

    def team_rates(x):
        x = x.merge(eg, on="game_id")
        x["ot"] = (x["n_periods"] > 2).astype(float)
        long = pd.concat([x[["home_team_id", "ot"]].rename(columns={"home_team_id": "team"}),
                          x[["away_team_id", "ot"]].rename(columns={"away_team_id": "team"})])
        return long.groupby("team")["ot"].agg(["size", "mean"])
    r, a = team_rates(ref), team_rates(g)
    j = r.join(a, lsuffix="_r", rsuffix="_a")
    pw = j[j["size_r"] >= MIN_TEAM_SIMS]
    return {"n_teams": int(len(j)), "n_teams_powered": int(len(pw)),
            "powered_mean_delta_ot": float((pw["mean_a"] - pw["mean_r"]).mean()) if len(pw) else None,
            "powered_share_teams_up": float((pw["mean_a"] > pw["mean_r"]).mean()) if len(pw) else None,
            "label": ("UNDERPOWERED per team: a team's OT count over its ~50-125 simulations "
                      "is a handful of events; cells reported only in aggregate")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = load(a.ref)
    act = actual_anchor()
    ids = set(ref["game_id"].unique())
    out = {"actual_season": {}, "actual_sample": {}}
    for lab, sub in (("actual_season", act), ("actual_sample", act[act.index.isin(ids)])):
        for k in range(7):
            s = sub[sub["m120"] == k]
            out[lab][str(k)] = {"n": int(len(s)), "p_tie": float(s["ot"].mean()) if len(s) else None,
                                "window_poss": float(s["window_poss"].mean()) if len(s) else None}
    out["runs"] = {a.ref: {"kernel": kernel(ref)}}
    for tag in a.runs:
        g = load(tag)
        out["runs"][tag] = {"kernel": kernel(g), "per_game": per_game(ref, g),
                            "per_team": per_team(ref, g)}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("P(tie | |m@2:00| = k), k = 0..6   [n for the actual]")
    row = lambda d: " ".join(f"{(d[str(k)]['p_tie'] or 0):.3f}" for k in range(7))  # noqa: E731
    print(f"{'actual_season':24s} {row(out['actual_season'])}  n={[out['actual_season'][str(k)]['n'] for k in range(7)]}")
    for tag, v in out["runs"].items():
        print(f"{tag:24s} {row(v['kernel'])}  {v.get('per_game', '')}")
    print("window poss by k:")
    rw = lambda d: " ".join(f"{(d[str(k)]['window_poss'] or 0):.2f}" for k in range(7))  # noqa: E731
    print(f"{'actual_season':24s} {rw(out['actual_season'])}")
    for tag, v in out["runs"].items():
        print(f"{tag:24s} {rw(v['kernel'])}  {v.get('per_team', {}).get('n_teams_powered', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
