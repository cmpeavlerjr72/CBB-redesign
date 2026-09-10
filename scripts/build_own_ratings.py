"""
build_own_ratings.py -- build the as-of-date own ridge ratings table.

Writes `data/processed/ratings/own_ratings_{season}.parquet` (one row per
season x as_of_date x team_id) plus `own_ratings_manifest.json` with the
FITTED ridge lambdas and prior-season shrinkage weights and the full selection
grid, and runs the INV-45 leak test on the resulting feature columns.

Hyperparameters are selected on the fold-1 TRAINING seasons only (2022, 2023);
`cbb_sim.data.seal.assert_not_sealed` guards every trainable slice, so season
2026 can never enter selection. Ratings themselves ARE produced for 2026 (they
are a pure as-of transform of that season's own games, used only at sim time
once the seal is lifted); pass --seasons to restrict.

Usage:
    .venv/Scripts/python.exe scripts/build_own_ratings.py
    .venv/Scripts/python.exe scripts/build_own_ratings.py --seasons 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.ratings import own_ratings as orat  # noqa: E402

DEFAULT_SEASONS = [2022, 2023, 2024, 2025, 2026]


def run_leak_test(tg: pd.DataFrame, ratings: pd.DataFrame, seasons: list[int]) -> pd.DataFrame | None:
    """Join the as-of own ratings back onto the team-game panel and run the
    change-form (INV-45) leak test. Returns None if the harness is not
    importable (another agent may still be finishing it), in which case the
    caller falls back to computing the change-form correlation directly."""
    try:
        from cbb_sim.analysis.leak_test import run_leak_test as _rlt
    except Exception as exc:  # pragma: no cover - harness may be in flux
        print(f"  leak-test harness not importable ({exc}); falling back to inline change-form corr")
        return None

    panel = tg[["game_id", "season", "game_date", "team_id", "opp_team_id", "margin"]].copy()
    panel = orat.join_as_of(panel, ratings, team_col="team_id", cols=("off_c", "def_c", "tempo_rel", "n_games"))
    panel = panel.rename(columns={"team_id": "team"})
    panel = panel[panel["season"].isin(seasons)]
    return _rlt(panel, ["off_c", "def_c", "tempo_rel"], team_col="team",
                season_col="season", date_col="game_date", margin_col="margin")


def inline_change_form_corr(tg: pd.DataFrame, ratings: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """docs/postmortem/05_cfb_methodology_extract.md section 4, computed here:
    corr(f[t] - f[t-1], margin_t) between consecutive games of the same team
    within a season (the leak channel) alongside corr(delta, margin_{t-1})
    (the honest update signature) and corr(f_t, margin_t) (level)."""
    panel = tg[["game_id", "season", "game_date", "team_id", "margin"]].copy()
    panel = orat.join_as_of(panel, ratings, team_col="team_id", cols=("off_c", "def_c", "tempo_rel"))
    panel = panel[panel["season"].isin(seasons)]
    panel = panel.sort_values(["team_id", "season", "game_date", "game_id"], kind="mergesort")
    g = panel.groupby(["team_id", "season"])
    rows = []
    for col in ["off_c", "def_c", "tempo_rel"]:
        d = g[col].diff()
        prev = g["margin"].shift(1)
        for season in [*seasons, "ALL"]:
            m = np.ones(len(panel), dtype=bool) if season == "ALL" else (panel["season"] == season).to_numpy()
            sub = pd.DataFrame({"d": d[m], "y": panel["margin"].to_numpy()[m], "yprev": prev[m],
                                "f": panel[col].to_numpy()[m]}).dropna(subset=["d"])
            rows.append({
                "column": col, "season": season, "n": len(sub),
                "corr_asjoined": float(sub["d"].corr(sub["y"])),
                "corr_update": float(sub.dropna(subset=["yprev"])["d"].corr(sub.dropna(subset=["yprev"])["yprev"])),
                "corr_level": float(panel.loc[m, col].corr(panel.loc[m, "margin"])),
                "verdict": "pass",
            })
    out = pd.DataFrame(rows)
    out["verdict"] = np.where(out["corr_asjoined"].abs() > 0.15, "LEAK", "pass")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--universe", default=str(orat.DEFAULT_UNIVERSE))
    ap.add_argument("--hoopr-dir", default=str(orat.DEFAULT_HOOPR_DIR))
    ap.add_argument("--out-dir", default=str(orat.DEFAULT_OUT_DIR))
    ap.add_argument("--skip-selection", action="store_true",
                    help="reuse the lambdas/weights already in the manifest")
    args = ap.parse_args()

    t0 = time.time()
    universe = orat.load_universe(args.universe)
    seasons = sorted(args.seasons)
    print(f"Loading team-games for seasons {seasons} ...")
    tg = orat.load_team_games(universe, seasons, args.hoopr_dir)
    print(f"  {len(tg):,} team-game rows, {tg['game_id'].nunique():,} games")

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "own_ratings_manifest.json"

    if args.skip_selection and manifest_path.exists():
        hp = json.loads(manifest_path.read_text(encoding="utf-8"))["hyperparameters"]
        print("  reusing hyperparameters from manifest")
    else:
        sel_seasons = orat.SELECTION_SEASONS
        # SEAL GUARD: selection may only see fold-1 training seasons.
        assert_not_sealed(list(sel_seasons), context="own_ratings hyperparameter selection")
        tg_sel = tg[tg["season"].isin(sel_seasons)]
        assert_not_sealed(tg_sel, context="own_ratings selection slice")
        print(f"Selecting ridge lambda / prior weight on fold-1 training seasons {list(sel_seasons)} ...")
        hp = orat.select_hyperparameters(tg_sel, seasons=sel_seasons)
        print(f"  eff:   lambda={hp['lambda_eff']}  prior_weight={hp['prior_weight_eff']}  "
              f"walk-forward RMSE={hp['rmse_eff']:.4f}")
        print(f"  tempo: lambda={hp['lambda_tempo']}  prior_weight={hp['prior_weight_tempo']}  "
              f"walk-forward RMSE={hp['rmse_tempo']:.4f}")

    print("Building walk-forward ratings for every season ...")
    frames, finals = orat.build_all_seasons(
        tg, seasons,
        lam_eff=hp["lambda_eff"], lam_tempo=hp["lambda_tempo"],
        w_eff=hp["prior_weight_eff"], w_tempo=hp["prior_weight_tempo"],
    )
    paths = orat.write_ratings(frames, out_dir)
    for p in paths:
        print(f"  wrote {p}  ({len(pd.read_parquet(p)):,} rows)")

    ratings = pd.concat(frames.values(), ignore_index=True)

    # ---- leak test -------------------------------------------------------
    leak_seasons = [s for s in seasons if s != 2026]
    print("Running the change-form leak test on the own-rating features ...")
    leak = run_leak_test(tg, ratings, leak_seasons)
    method = "cbb_sim.analysis.leak_test.run_leak_test"
    if leak is None:
        leak = inline_change_form_corr(tg, ratings, leak_seasons)
        method = "inline change-form correlation (postmortem 05 section 4)"
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 100):
        print(leak.to_string(index=False))

    leak_path = out_dir / "own_ratings_leak_test.csv"
    leak.to_csv(leak_path, index=False)
    print(f"  wrote {leak_path}  (method: {method})")

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seasons": seasons,
        "hyperparameters": {k: v for k, v in hp.items() if not k.startswith("grid_")},
        "selection_grid_eff": hp.get("grid_eff", []),
        "selection_grid_tempo": hp.get("grid_tempo", []),
        "n_team_games": int(len(tg)),
        "n_games": int(tg["game_id"].nunique()),
        "leak_test_method": method,
        "leak_test": leak[leak["season"] == "ALL"].to_dict("records") if "season" in leak.columns else [],
        "final_league_means": {
            str(s): {
                "eff_intercept": finals[s]["eff"]["intercept"],
                "eff_home": finals[s]["eff"]["home"],
                "eff_away": finals[s]["eff"]["away"],
                "tempo_intercept": finals[s]["tempo"]["intercept"],
                "tempo_neutral": finals[s]["tempo"]["neutral"],
            }
            for s in seasons
        },
    }
    orat.write_manifest(manifest, out_dir)
    print(f"  wrote {manifest_path}")
    print(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
