#!/usr/bin/env python
"""
grade_foul6_closed_loop_v1.py -- the round-6 paired closed loop, blind.

    .venv/Scripts/python.exe scripts/grade_foul6_closed_loop_v1.py

1. PARITY. The default path re-run under the edited engine
   (`foul6_PARITY_s25`, `ENGINE_FOUL_ACCRUAL` unset) must equal the stored
   section-12 reference `po4b_R_s25` on EVERY simulated value.
2. FLOORS. `floor = |po4b_R_s25_floor - po4b_R_s25|` (the section-12 seed-offset
   run, reused as pre-registered).
3. The arm `foul6_F5_s25` against the reference, in floors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/engine_v0")
REF = "po4b_R_s25"
FLOOR = "po4b_R_s25_floor"
PARITY = "foul6_PARITY_s25"
ARM = "foul6_F5_s25"
ARM_E = "foul6_F5e_s25"
TRUTH = Path("data/reference")


def load(tag: str) -> pd.DataFrame:
    return pd.read_parquet(R / tag / "games.parquet")


def lines(g: pd.DataFrame) -> dict:
    fta = g["home_fta"] + g["away_fta"]
    fga = (g["home_fga2_rim"] + g["home_fga2_jump"] + g["home_fga3"]
           + g["away_fga2_rim"] + g["away_fga2_jump"] + g["away_fga3"])
    tot = g["home_pts"] + g["away_pts"]
    mar = g["home_pts"] - g["away_pts"]
    per_game = g.groupby("game_id").agg(tot=("home_pts", "size"))
    gm = g.assign(tot=tot, mar=mar).groupby("game_id")[["tot", "mar"]].mean()
    return {
        "fta_fga_pooled": float(fta.sum() / fga.sum()),
        "fta_per_game": float(fta.mean()),
        "fga_per_game": float(fga.mean()),
        "possessions": float(g["possessions"].mean()),
        "total_mean": float(tot.mean()),
        "total_sd": float(gm["tot"].std()),
        "margin_sd": float(gm["mar"].std()),
        "oreb_pct": float((g["home_oreb"] + g["away_oreb"]).sum()
                          / (g["home_oreb"] + g["away_oreb"] + g["home_dreb"] + g["away_dreb"]).sum()),
        "n_game_seeds": int(len(g)), "n_games": int(len(per_game)),
    }


def team_ft_slope(g: pd.DataFrame) -> dict:
    """Team FT-rate spread: per-team sim FT rate against the team's prior-season
    (2024) box FT rate, five quintiles, powered cells only."""
    tb = []
    for side, opp in (("home", "away"), ("away", "home")):
        tb.append(pd.DataFrame({
            "team_id": g[f"{side}_team_id"] if f"{side}_team_id" in g.columns else np.nan,
            "fta": g[f"{side}_fta"],
            "fga": g[f"{side}_fga2_rim"] + g[f"{side}_fga2_jump"] + g[f"{side}_fga3"]}))
    t = pd.concat(tb, ignore_index=True)
    if t["team_id"].isna().all():
        return {"status": "NOT AVAILABLE -- games.parquet carries no team ids"}
    a = t.groupby("team_id").agg(fta=("fta", "sum"), fga=("fga", "sum"), n=("fta", "size"))
    a = a[a["n"] >= 75]
    a["rate"] = a["fta"] / a["fga"]
    return {"n_teams_powered": int(len(a)), "sd_team_ft_rate": float(a["rate"].std())}


def main() -> None:
    out: dict = {}
    ref, par = load(REF), load(PARITY)
    key = ["game_id", "seed"]
    a = ref.sort_values(key).reset_index(drop=True)
    b = par.sort_values(key).reset_index(drop=True)
    same_cols = sorted(set(a.columns) & set(b.columns))
    eq = {c: bool(a[c].equals(b[c])) for c in same_cols}
    out["parity"] = {"shape_ref": list(a.shape), "shape_rerun": list(b.shape),
                     "columns_compared": len(same_cols),
                     "columns_equal": int(sum(eq.values())),
                     "columns_unequal": [c for c, v in eq.items() if not v],
                     "BIT_IDENTICAL": all(eq.values()) and a.shape == b.shape}

    tags = [t for t in (REF, FLOOR, PARITY, ARM, ARM_E) if (R / t / "games.parquet").exists()]
    L = {t: lines(load(t)) for t in tags}
    out["lines"] = L
    floors = {k: abs(L[FLOOR][k] - L[REF][k]) for k in L[REF] if isinstance(L[REF][k], float)}
    out["floors"] = floors
    for arm_tag, label in ((ARM, "arm_vs_ref"), (ARM_E, "arm_e_vs_ref")):
        if arm_tag not in L:
            continue
        out[label] = {
            k: {"ref": L[REF][k], "arm": L[arm_tag][k], "delta": L[arm_tag][k] - L[REF][k],
                "floor": floors.get(k),
                "floors": (abs(L[arm_tag][k] - L[REF][k]) / floors[k]) if floors.get(k) else None}
            for k in L[REF] if isinstance(L[REF][k], float)}
    out["team_spread"] = {t: team_ft_slope(load(t)) for t in tags}
    out["actual_fta_fga_target"] = 0.32955
    p = Path("results/engine_v0/foul6_closed_loop_grade.json")
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"parity": out["parity"],
                      "arm_vs_ref": out.get("arm_vs_ref"),
                      "arm_e_vs_ref": out.get("arm_e_vs_ref")}, indent=1, default=float))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
