"""
run_control.py -- simulate a fold's test season with a trained Control bundle.

    .venv/Scripts/python.exe scripts/run_control.py --fold F2 --anchor A_own --seeds 200
    .venv/Scripts/python.exe scripts/run_control.py --fold F2 --anchor A_own --seeds 200 --seed-offset 1000

Writes `results/control/<fold>_<anchor>[_seedoff{N}]/`:
    games.parquet    one row per (game_id, seed): home_pts, away_pts,
                     possessions, n_ot, went_ot, created_at, tipoff, backtest
    summary.parquet  one row per game: sim mean/SD margin and total, p_home,
                     sim possessions, sim OT rate, the actual result, the
                     rating differentials the responsiveness check buckets on,
                     created_at, tipoff, backtest

ARTIFACT HONESTY (CLAUDE.md "backtests must be honest"): every row carries
`created_at` and `tipoff`. A live run asserts created_at < tipoff; a backtest
(the only mode possible for a completed season) is stamped `backtest=True`
instead of pretending otherwise.

SEAL. The fold's test season is asserted unsealed before anything is read.
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

from cbb_sim.control import features as F  # noqa: E402
from cbb_sim.control import models as M  # noqa: E402
from cbb_sim.control import simulate as S  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

DEFAULT_RESULTS = Path("results/control")


def run_dir(fold: str, anchor: str, seed_offset: int, results_dir: Path) -> Path:
    name = f"{fold}_{anchor}" + (f"_seedoff{seed_offset}" if seed_offset else "")
    return Path(results_dir) / name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=sorted(F.FOLDS))
    ap.add_argument("--anchor", required=True, choices=list(F.ANCHORS))
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--chunk-seeds", type=int, default=20)
    ap.add_argument("--model-dir", default=str(M.DEFAULT_MODEL_DIR))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--live", action="store_true",
                    help="assert created_at < tipoff for every row instead of stamping backtest=True")
    args = ap.parse_args()

    t0 = time.time()
    _, test_seasons = F.fold_seasons(args.fold)
    assert_not_sealed(test_seasons, context=f"control sim {args.fold}/{args.anchor}")

    bundle = M.load(args.fold, args.anchor, args.model_dir)
    print(f"loaded models {args.fold}/{args.anchor} trained on {bundle.train_seasons}")

    tg, g = F.build_team_game_features(test_seasons)
    print(f"  test: {len(g):,} games, {len(tg):,} team-games")

    inp = S.prepare(g, tg, bundle)
    print(f"  preflight OK: {len(inp.features_checked)} model features present in the sim rows")

    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    sims = S.simulate(inp, seeds, family="control", chunk_seeds=args.chunk_seeds)
    print(f"  simulated {len(sims):,} game-seeds in {time.time() - t0:.1f}s "
          f"(possession floor bound on {sims.attrs.get('n_poss_floor', 0)} draws)")

    created_at = pd.Timestamp(datetime.now(UTC))
    tipoff = g.set_index("game_id")["tipoff_utc"]

    sims["went_ot"] = sims["n_ot"] > 0
    sims["tipoff"] = sims["game_id"].map(tipoff).to_numpy()
    sims["created_at"] = created_at
    sims["backtest"] = not args.live
    if args.live:
        bad = int((sims["created_at"] >= sims["tipoff"]).sum())
        if bad:
            raise RuntimeError(f"live run: {bad} rows violate created_at < tipoff")

    summary = S.summarise(sims.drop(columns=["tipoff", "created_at", "backtest", "went_ot"]), g)
    summary["rating_diff_own"] = (
        (summary["h_own_off_c"] - summary["h_own_def_c"])
        - (summary["a_own_off_c"] - summary["a_own_def_c"])
    )
    summary["rating_diff_kp"] = (
        (summary["h_own_kp_adj_o_c"] - summary["h_own_kp_adj_d_c"])
        - (summary["a_own_kp_adj_o_c"] - summary["a_own_kp_adj_d_c"])
    )
    summary["rating_diff"] = (
        summary["rating_diff_kp"] if args.anchor == "B_kp" else summary["rating_diff_own"]
    )
    summary["fold"] = args.fold
    summary["anchor"] = args.anchor
    summary["seed_offset"] = args.seed_offset
    summary["n_seeds_requested"] = args.seeds
    summary["created_at"] = created_at
    summary["tipoff"] = summary["tipoff_utc"]
    summary["backtest"] = not args.live

    out = run_dir(args.fold, args.anchor, args.seed_offset, Path(args.results_dir))
    out.mkdir(parents=True, exist_ok=True)
    sims.to_parquet(out / "games.parquet", index=False)
    summary.to_parquet(out / "summary.parquet", index=False)

    meta = {
        "created_at": created_at.isoformat(),
        "fold": args.fold, "anchor": args.anchor,
        "train_seasons": bundle.train_seasons, "test_seasons": test_seasons,
        "seeds": args.seeds, "seed_offset": args.seed_offset,
        "n_games": int(len(g)), "n_rows": int(len(sims)),
        "backtest": not args.live,
        "features_checked": inp.features_checked,
        "pace_resid_sd": bundle.pace.resid_sd,
        "rate_families": {t: m.family for t, m in bundle.rates.items()},
        "rate_alpha": {t: (None if m.family == "poisson" else m.alpha) for t, m in bundle.rates.items()},
        "pct_rho": {t: m.rho for t, m in bundle.pcts.items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  wrote {out / 'games.parquet'} and summary.parquet")
    print(f"  sim margin mean {summary['sim_margin_mean'].mean():+.3f} "
          f"(actual {summary['margin'].mean():+.3f}); "
          f"sim total mean {summary['sim_total_mean'].mean():.2f} "
          f"(actual {summary['total'].mean():.2f})")
    print(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
