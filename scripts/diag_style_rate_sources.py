#!/usr/bin/env python
"""
diag_style_rate_sources.py -- where should each L3 team-style rate come from?

    .venv/Scripts/python.exe scripts/diag_style_rate_sources.py
    .venv/Scripts/python.exe scripts/diag_style_rate_sources.py --version v2

The round-2 pre-registration replaces the L3 style rates' source: they were
built over the POSSESSION table (attempts summed across every chance of the
possession, continuation chances included) and are now built over FIRST chances
only, so that a continuation-chance labelling defect cannot reach a
first-chance model's predictors
(`docs/tests/shot_classification_diag_2026-09-10.md` section 6).

The pre-registration's other option was "or from hoopR `team_box` where the
stat exists". This script answers, per rate, whether that option exists at all
and how closely the chosen source tracks the independent box score. Its output
is the source-manifest evidence quoted in
`docs/models/possession_outcome/features.md` section 1.1.

WHAT IS COMPARED. Per team-game, over `is_d1_game & ~pbp_truncated` games that
both sources cover, joined on ESPN `game_id` + `team_id`:

  * `3pa`  three-point attempts per 100 possessions
  * `tov`  turnovers per 100 possessions
  * `ftr`  free-throw attempts per field-goal attempt
  * `rim`  rim share of field-goal attempts  -- team_box CANNOT express this;
           it carries no rim/jumper split at all (its nearest column,
           `points_in_paint`, is points and not attempts, and paint is not the
           rim). Reported as unavailable rather than approximated.

hoopR possessions use the same box formula the possession build validates
against: `FGA - OREB + TOV + 0.44*FTA`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.pbp import possessions as PS  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025, 2026]
HOOPR_TEAM_BOX = Path("data/raw/hoopr/team_box")


def box_rates(season: int) -> pd.DataFrame:
    p = HOOPR_TEAM_BOX / f"team_box_{season}.parquet"
    cols = ["game_id", "team_id", "field_goals_attempted",
            "three_point_field_goals_attempted", "free_throws_attempted",
            "offensive_rebounds", "turnovers"]
    b = pd.read_parquet(p, columns=cols)
    b["game_id"] = pd.to_numeric(b["game_id"], errors="coerce").astype("Int64")
    b["team_id"] = pd.to_numeric(b["team_id"], errors="coerce").astype("Int64")
    for c in cols[2:]:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["poss"] = (b["field_goals_attempted"] - b["offensive_rebounds"]
                 + b["turnovers"] + 0.44 * b["free_throws_attempted"])
    out = pd.DataFrame({
        "game_id": b["game_id"], "team_id": b["team_id"],
        "box_3pa": 100 * b["three_point_field_goals_attempted"] / b["poss"],
        "box_tov": 100 * b["turnovers"] / b["poss"],
        "box_ftr": 100 * b["free_throws_attempted"] / b["field_goals_attempted"],
    })
    return out.replace([np.inf, -np.inf], np.nan)


def event_rates(box: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": box["game_id"], "team_id": box["team_id"],
        "ev_3pa": 100 * box["fga_3"] / box["poss"],
        "ev_tov": 100 * box["tov"] / box["poss"],
        "ev_ftr": 100 * box["fta"] / box["fga"],
        "ev_rim": 100 * box["fga_rim"] / box["fga"],
    }).replace([np.inf, -np.inf], np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", choices=sorted(PS.POSSESSION_VERSIONS))
    ap.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/models/possession_outcome/round2/"
                                 "style_rate_sources.json"))
    args = ap.parse_args()

    u = pd.read_parquet("data/processed/games_universe.parquet")
    u = u[u["is_d1_game"] & ~u["pbp_truncated"]]
    keep = set(u["game_id"].tolist())

    rows = []
    for season in args.seasons:
        d = PS.possessions_dir(args.version)
        ch = pd.read_parquet(d / f"chances_{season}.parquet")
        ch = ch[ch["game_id"].isin(keep)]
        first = event_rates(PO.team_game_box_first_chance(ch))
        poss = pd.read_parquet(d / f"possessions_{season}.parquet",
                               columns=["season", "game_id", "offense_team_id", "defense_team_id",
                                        "poss_index", "terminal_event", "fga_rim", "fga_jump2",
                                        "fga_3", "fta", "points"])
        poss = poss[poss["game_id"].isin(keep)]
        allch = event_rates(PO.team_game_box(poss))
        bx = box_rates(season)

        m = first.merge(allch, on=["game_id", "team_id"], suffixes=("", "_all"))
        m = m.merge(bx, on=["game_id", "team_id"], how="inner").dropna()
        for rate in ("3pa", "tov", "ftr"):
            rows.append({
                "season": season, "rate": rate, "n_team_games": int(len(m)),
                "corr_first_chance_vs_box": round(float(m[f"ev_{rate}"].corr(m[f"box_{rate}"])), 4),
                "corr_all_chances_vs_box": round(
                    float(m[f"ev_{rate}_all"].corr(m[f"box_{rate}"])), 4),
                "mean_first_chance": round(float(m[f"ev_{rate}"].mean()), 3),
                "mean_all_chances": round(float(m[f"ev_{rate}_all"].mean()), 3),
                "mean_box": round(float(m[f"box_{rate}"].mean()), 3),
            })
        rows.append({
            "season": season, "rate": "rim", "n_team_games": int(len(m)),
            "corr_first_chance_vs_box": None, "corr_all_chances_vs_box": None,
            "mean_first_chance": round(float(m["ev_rim"].mean()), 3),
            "mean_all_chances": round(float(m["ev_rim_all"].mean()), 3),
            "mean_box": None,
        })
        print(f"[{season}] {len(m):,} team-games compared", flush=True)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"version": args.version, "rows": rows}, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
