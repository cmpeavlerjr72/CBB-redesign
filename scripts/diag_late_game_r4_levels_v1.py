"""diag_late_game_r4_levels_v1.py -- late-game ROUND 4 multi-level reported lines (experiments.md 9.4).

Per run: OT rate by site (neutral / non-neutral game), end-of-period PPP (possessions starting <= 10 s) by half
and by offence site (home / away / neutral), against the 2024-25 actual on the same definitions. Per game:
paired tie-count deltas vs the base (via round 2's levels tool). Cells under 200 are UNDERPOWERED.

    .venv/Scripts/python.exe scripts/diag_late_game_r4_levels_v1.py --base lg4_R9_s25 --runs lg4_DtBZ_s25 ... --out results/late_game/round4/levels.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("CBB_TRUTH", "verified_v1")
from cbb_sim.eval import reference as REF                             # noqa: E402
import diag_late_game_r2_levels_v1 as LV                              # noqa: E402

RES = ROOT / "results/engine_v0"
GAMES = ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet"


def lab(n):
    return "UNDERPOWERED" if n < 200 else ""


def site_tables(p: pd.DataFrame, ot: pd.DataFrame, neutral: pd.Series) -> dict:
    out = {}
    o = ot.join(neutral, on="game_id")
    for nm, m in (("neutral", o["neutral"] > 0), ("non_neutral", o["neutral"] == 0)):
        out[f"ot|{nm}"] = {"n": int(m.sum()), "rate": float(o.loc[m, "ot"].mean())}
    p = p.join(neutral, on="game_id")
    site = np.where(p["neutral"] > 0, "neutral", np.where(p["off_home"], "home", "away"))
    for per in (1, 2):
        for s in ("home", "away", "neutral", "all"):
            m = (p["per"] == per) & (p["sec"] <= 10) & ((site == s) if s != "all" else True)
            n = int(m.sum())
            out[f"ppp_le10|P{per}|{s}"] = {"n": n, "label": lab(n), "ppp": float(p.loc[m, "pts"].mean()) if n else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    neutral = pd.read_parquet(GAMES, columns=["game_id", "neutral"]).set_index("game_id")["neutral"]
    d = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2025.parquet",
                        columns=["game_id", "period", "start_clock", "points", "offense_is_home"])
    d = d[(d["period"] <= 2) & (d["start_clock"] <= 35)]
    pa = pd.DataFrame({"game_id": d["game_id"], "per": d["period"], "sec": d["start_clock"], "pts": d["points"],
                       "off_home": d["offense_is_home"]})
    ag = REF.load_actual_games(2025)
    oa = pd.DataFrame({"game_id": ag["game_id"], "ot": ag["n_periods"] > 2})
    out = {"actual": site_tables(pa[pa["game_id"].isin(oa["game_id"])], oa, neutral), "runs": {}}
    base = LV.load(a.base)
    for t in [a.base] + a.runs:
        tp = pd.read_parquet(RES / t / "tap_poss.parquet")
        tp = tp[tp["sec"] <= 35]
        ps = pd.DataFrame({"game_id": tp["game_id"], "per": tp["per"], "sec": tp["sec"], "pts": tp["d_pts_off"],
                           "off_home": tp["off"] == 0})
        g = pd.read_parquet(RES / t / "games.parquet", columns=["game_id", "seed", "n_periods"])
        r = {"site": site_tables(ps, pd.DataFrame({"game_id": g["game_id"], "ot": g["n_periods"] > 2}), neutral)}
        if t != a.base:
            gg = LV.load(t)
            r["per_game"] = LV.per_game(base, gg)
            r["per_team"] = LV.per_team(base, gg)
        out["runs"][t] = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    keys = list(out["actual"].keys())
    print(f"{'line':24s} {'actual':>14s} " + " ".join(f"{t[:12]:>14s}" for t in out["runs"]))
    for k in keys:
        def f(x):
            v = x.get("rate", x.get("ppp"))
            return f"{v:.4f} ({x['n']})" if v is not None else "-"
        print(f"{k:24s} {f(out['actual'][k]):>14s} " + " ".join(f"{f(out['runs'][t]['site'][k]):>14s}" for t in out["runs"]))
    for t, r in out["runs"].items():
        if "per_game" in r:
            print(t, r["per_game"], "teams powered:", r["per_team"]["n_teams_powered"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
