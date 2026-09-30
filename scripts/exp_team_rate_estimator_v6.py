"""exp_team_rate_estimator_v6.py -- key coverage for the ENGINE universe as well as the designs.

The F2 engine inputs (`games_F2_2025_v2.parquet`, whose game order the v3 replay builder keeps) hold 7 D-I
games that have neither pbp nor a team box: 401714278, 401722532, 401725735, 401706691, 401700283,
401716154 and 401720999. Some of them show a 0-0 final. They are in no design, so table v3 has no key for
them, but the engine simulates them. They enter the estimator panel exactly as v5's six box-less games do:
as schedule-only rows with no observation, one index step, and the state entering the game.

This script is v5 with the engine games added to the key set. Outputs are new versions; nothing is
overwritten:
  data/processed/team_rate_features_E3_v4.parquet, ..._E3opp_v4.parquet, team_rate_variance_O1a_v3.parquet
Coverage is asserted for the three design frames on both folds AND for the F2 engine games (home and away).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exp_team_rate_estimator_v5 as X5  # noqa: E402

ENGINE_GAMES = Path("data/processed/models/engine/games_F2_2025_v2.parquet")
_design_games_v5 = X5.design_games


def design_and_engine_games():
    d = _design_games_v5()
    g = pd.read_parquet(ENGINE_GAMES, columns=["game_id", "season", "home_team_id", "away_team_id"])
    e = pd.concat([g.rename(columns={"home_team_id": "team_id", "away_team_id": "opp_id"}),
                   g.rename(columns={"away_team_id": "team_id", "home_team_id": "opp_id"})])
    return pd.concat([d, e[["season", "game_id", "team_id", "opp_id"]]]).drop_duplicates(["season", "game_id", "team_id"])


X5.design_games = design_and_engine_games
_cov_v5 = X5.coverage_check


def coverage_check(table):
    _cov_v5(table)
    g = pd.read_parquet(ENGINE_GAMES, columns=["game_id", "home_team_id", "away_team_id"])
    t = table[table["fold"] == "F2"]
    k = set(zip(t["game_id"], t["team_id"]))
    miss = sum((a, b) not in k for a, b in zip(g["game_id"], g["home_team_id"])) + \
        sum((a, b) not in k for a, b in zip(g["game_id"], g["away_team_id"]))
    print("engine F2 games:", len(g), "missing team keys:", miss)
    assert miss == 0


X5.coverage_check = coverage_check
X5.OUT_COVERAGE_NOTE = "v6 coverage printed to the log; coverage_v5.json is v5's and is not rewritten"


def rename_outputs():
    """v5 writes *_v3 / O1a_v2; this version writes *_v4 / O1a_v3 (the v5 function bodies with new paths)."""
    import inspect
    src = inspect.getsource(X5.emit_features).replace("_v3.parquet", "_v4.parquet").replace(
        "team_rate_features_{label}_v2.parquet", "team_rate_features_{label}_v3.parquet")
    src_o = inspect.getsource(X5.emit_o1a).replace("team_rate_variance_O1a_v2.parquet", "team_rate_variance_O1a_v3.parquet")
    src_t = inspect.getsource(X5.team_games_with_schedule).replace("schedule_only_v5.json", "schedule_only_v6.json")
    src = src.replace("schedule_only_v5.json", "schedule_only_v6.json")
    ns = X5.__dict__
    exec(src, ns); exec(src_o, ns); exec(src_t, ns)
    X5.X2.team_games = ns["team_games_with_schedule"]
    global _cov_v5
    exec(inspect.getsource(_cov_v5).replace("coverage_v5.json", "coverage_v6.json"), ns)
    _cov_v5 = ns["coverage_check"]
    X5.coverage_check = coverage_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["features", "opp", "o1a"])
    a = ap.parse_args()
    rename_outputs()
    Ps, _, _ = X5.X3.build_all()
    if a.part == "features":
        X5.emit_features(Ps, False)
    elif a.part == "opp":
        X5.emit_features(Ps, True)
    else:
        X5.emit_o1a(Ps)


if __name__ == "__main__":
    main()
