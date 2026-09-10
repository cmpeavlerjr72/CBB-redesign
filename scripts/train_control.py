"""
train_control.py -- fit the Control engine's component models for one fold and
one anchor feature set.

    .venv/Scripts/python.exe scripts/train_control.py --fold F1 --anchor A_own
    .venv/Scripts/python.exe scripts/train_control.py --fold F2 --anchor C_both

Folds (pre-registered, `docs/models/control_engine/experiments.md`):
    F1  train {2022, 2023}  test 2024
    F2  train {2022, 2023, 2024}  test 2025   <- selection metric

Anchors: A_own (own ridge ratings), B_kp (centred KenPom), C_both.

SEAL. `cbb_sim.data.seal.assert_not_sealed` is called on the fold's season
list AND on the materialised training slice, so season 2026 can never enter a
fitted artifact.

Writes `data/processed/models/control_engine/{pace,rates,pcts}_{fold}_{anchor}.pkl`
(each carrying its own feature list, the train-time feature medians used to
fill a missing pregame value, and its fitted dispersion parameter) plus a
`train_{fold}_{anchor}.json` report with every fitted dispersion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.control import features as F  # noqa: E402
from cbb_sim.control import models as M  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=sorted(F.FOLDS))
    ap.add_argument("--anchor", required=True, choices=list(F.ANCHORS))
    ap.add_argument("--model-dir", default=str(M.DEFAULT_MODEL_DIR))
    args = ap.parse_args()

    t0 = time.time()
    train_seasons, test_seasons = F.fold_seasons(args.fold)

    # ---- SEAL GUARD ------------------------------------------------------
    assert_not_sealed(train_seasons, context=f"control train {args.fold}/{args.anchor}")
    assert_not_sealed(test_seasons, context=f"control test {args.fold}/{args.anchor}")

    print(f"fold {args.fold}: train {train_seasons} -> test {test_seasons}; anchor {args.anchor}")
    tg, g = F.build_team_game_features(train_seasons)
    assert_not_sealed(tg, context="control training team-game slice")
    assert_not_sealed(g, context="control training game slice")
    print(f"  train: {len(g):,} games, {len(tg):,} team-games")

    team_feats = list(F.ANCHOR_TEAM_FEATURES[args.anchor])
    pace_feats = list(F.ANCHOR_PACE_FEATURES[args.anchor])
    created_at = datetime.now(UTC).isoformat()

    bundle = M.fit_all(tg, g, team_feats, pace_feats, args.fold, args.anchor,
                       train_seasons, created_at)
    paths = M.save(bundle, args.model_dir)

    summary = M.summary_table(bundle)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))

    report = {
        "created_at": created_at,
        "fold": args.fold,
        "anchor": args.anchor,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "n_games_train": int(len(g)),
        "n_team_games_train": int(len(tg)),
        "team_features": team_feats,
        "pace_features": pace_feats,
        "negbin_rule": f"NegBin iff Poisson training deviance/df > {M.NEGBIN_DEVIANCE_DF_THRESHOLD}",
        "pace": {
            "family": bundle.pace.family,
            "fitted_resid_sd": bundle.pace.resid_sd,
            "r2": bundle.pace.extra.get("r2"),
            "coef": dict(zip(["intercept", *pace_feats], bundle.pace.coef.tolist(), strict=True)),
        },
        "rates": {
            t: {
                "family": m.family,
                "poisson_deviance_df": m.deviance_df,
                "fitted_negbin_alpha": (None if m.family == "poisson" else m.alpha),
                "mean_per_100": m.extra.get("mean_per_100"),
                "coef": dict(zip(["intercept", *team_feats], m.coef.tolist(), strict=True)),
            }
            for t, m in bundle.rates.items()
        },
        "pcts": {
            t: {
                "family": m.family,
                "fitted_betabinom_rho": m.rho,
                "pooled_rate": m.extra.get("pooled_rate"),
                "mean_trials": m.extra.get("mean_trials"),
                "binomial_pearson_dispersion": m.extra.get("pearson_dispersion"),
                "coef": dict(zip(["intercept", *team_feats], m.coef.tolist(), strict=True)),
            }
            for t, m in bundle.pcts.items()
        },
        "artifacts": {k: str(v) for k, v in paths.items()},
    }
    rp = Path(args.model_dir) / f"train_{args.fold}_{args.anchor}.json"
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {rp}")
    print(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
