"""
grade_market_games.py -- G10, the game-level market scorecard.

    .venv/Scripts/python.exe scripts/grade_market_games.py --results results/<tag> --season 2025

Generalises `scripts/grade_control.py`'s `gate_g10` (the prototype, read but
never imported) via `src/cbb_sim/eval/market.py`. Grades ANY contract-
compliant engine (`src/cbb_sim/eval/contract.py`) against
`data/raw/cbbd/lines_{season}.parquet`:

  - provider preference DraftKings > ESPN BET > Bovada > consensus, recorded
    per row (`provider_used`);
  - MAE model vs close, signed bias, on margin and total;
  - ATS/OU record and ROI by disagreement bucket at -110, PLUS a moneyline
    edge-bucket ROI table at REAL posted odds (methodology doc section 6's
    fixed edge buckets);
  - Brier vs de-vigged moneyline, calibration deciles;
  - the surprise correlation and the CLV sign-agreement where opens exist;
  - the LEAK-SUSPECT flag (surprise corr > 0.15 AND CLV agreement < 0.53);
  - a game-clustered bootstrap CI on ROI (1000 reps) for the ATS, OU, and ML
    edge tables.

SETTLEMENT VS DE-VIG (methodology doc section 6, quoted verbatim): "de-vig is
for the probability comparison only, never for settlement." Every bet here
settles at the REAL posted line/odds; de-vigging only builds the probability
used for Brier/calibration. See `market.py`'s module docstring.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval import contract as C  # noqa: E402
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import market as M  # noqa: E402
from cbb_sim.eval import report as R  # noqa: E402
from cbb_sim.eval.cli import load_tolerances, tag_from_results_dir  # noqa: E402

OUT_DIR = Path("docs/tests")


def bootstrap_line(name: str, boot: dict) -> str:
    if boot["n_games"] == 0:
        return f"{name} bootstrap: n/a (no settled bets)"
    return (f"{name} bootstrap ({boot['n_boot']} reps, {boot['n_games']} games): "
            f"mean ROI {boot['mean']:+.4f}, 95% CI [{boot['lo95']:+.4f}, {boot['hi95']:+.4f}], "
            f"P(ROI<=0) = {boot['p_roi_le_0']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--gates-config", default="docs/gates.yaml")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-sealed", action="store_true")
    args = ap.parse_args()

    tol = load_tolerances(Path(args.gates_config))
    engine = C.load_engine_results(args.results, allow_sealed=args.allow_sealed)
    tag = tag_from_results_dir(Path(args.results))
    out_path = Path(args.out) if args.out else OUT_DIR / f"market_games_{tag}_{datetime.now(UTC).date().isoformat()}.md"

    summary, _raw = G.build_grading_frame(engine.games, args.season)
    res = M.gate_g10(summary, args.season, tol, n_boot=args.n_boot)

    now = datetime.now(UTC)
    L: list[str] = [
        f"# Market scorecard (G10) -- {engine.engine_tag} (season {args.season})",
        "",
        f"Generated {now.date().isoformat()} by `scripts/grade_market_games.py`. "
        f"Results: `{args.results}`. Lines: `data/raw/cbbd/lines_{args.season}.parquet`, "
        "provider preference DraftKings > ESPN BET > Bovada > consensus "
        f"(rows used, by provider: {res.provider_counts}). "
        "`spread` is home-perspective, so the market's expected home margin is `-spread`. "
        "G10 is report-only except the leak screen "
        f"(LEAK-SUSPECT if surprise corr > {tol['g10_surprise_corr']} AND CLV agreement < "
        f"{tol['g10_clv_agreement']}). SETTLEMENT is always at the real posted line/odds; "
        "de-vig is used only to build the probability for Brier/calibration.",
        "",
        f"n games with a usable spread line: {res.n_with_line}",
        "",
    ]

    L += ["## Margin and total: model vs close", ""]
    import pandas as pd  # local import to keep the top of the file lean
    core = pd.DataFrame([{
        "n": res.n_with_line,
        "model_margin_MAE": res.model_margin_mae, "close_margin_MAE": res.close_margin_mae,
        "model_margin_bias": res.model_margin_bias, "close_margin_bias": res.close_margin_bias,
        "model_total_MAE": res.model_total_mae, "close_total_MAE": res.close_total_mae,
        "model_total_bias": res.model_total_bias, "close_total_bias": res.close_total_bias,
    }])
    L += R.md_table(core) + [""]
    beat = res.model_margin_mae < res.close_margin_mae
    L += [f"Margin: model MAE {res.model_margin_mae:.4f} vs close {res.close_margin_mae:.4f} "
          f"(model {'beats' if beat else 'trails'} the close by "
          f"{abs(res.model_margin_mae - res.close_margin_mae):.4f}) -- report-only, PASS", ""]

    L += ["## ATS by disagreement bucket (settled at the real spread, -110)", ""]
    L += R.md_table(res.ats_table) + [""]
    L += [bootstrap_line(f"ATS >= {M.ATS_THRESHOLDS[0]:.0f} pt", res.ats_bootstrap), ""]

    L += ["## OU by disagreement bucket (settled at the real total, -110)", ""]
    L += R.md_table(res.ou_table) + [""]
    L += [bootstrap_line(f"OU >= {M.ATS_THRESHOLDS[0]:.0f} pt", res.ou_bootstrap), ""]

    L += ["## Moneyline edge buckets (settled at REAL posted odds; edge = model P(home) - de-vigged market P(home))", ""]
    L += R.md_table(res.ml_edge_table) + [""]
    L += [bootstrap_line("ML edge (all buckets pooled)", res.ml_edge_bootstrap), ""]

    L += [f"## Brier vs de-vigged moneyline (n = {res.n_ml}, mean vig {res.mean_vig:.4f})", ""]
    L += [f"model {res.model_brier_ml_subset:.5f} vs market {res.market_brier:.5f} -- report-only, PASS", ""]
    L += ["### Calibration deciles (model P(home) vs de-vigged market P(home) vs actual)", ""]
    L += R.md_table(res.calibration) + [""]

    clv = f"{res.clv_agreement:.4f}" if res.clv_agreement == res.clv_agreement else "n/a (no opening lines)"
    L += ["## Leak screen", "",
          f"surprise corr {res.surprise_corr:.4f} (gate {tol['g10_surprise_corr']}), "
          f"CLV sign agreement {clv} on {res.n_clv} moved lines -- **{res.leak_status}**", ""]
    for note in res.notes:
        L += [note, ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(L)} lines)")
    print(f"G10 leak status: {res.leak_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
