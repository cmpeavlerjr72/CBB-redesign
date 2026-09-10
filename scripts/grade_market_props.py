"""
grade_market_props.py -- player-prop market scorecard.

    .venv/Scripts/python.exe scripts/grade_market_props.py --results results/<tag> --season 2025 \
        [--props-lines data/raw/cbbd/props_lines_2025.parquet] [--actual-stats <path>] [--settle book|fair]

NO PROPS-LINES DATA EXISTS YET (CLAUDE.md bans scraping PrizePicks/Underdog/
DraftKings directly, and no CBBD props endpoint has been pulled). This script
still ships, complete, against the documented schema in
`src/cbb_sim/eval/props.py`'s module docstring, so grading real props lines
the day they arrive is this one command -- proven correct today by
`tests/test_eval.py::test_props_synthetic_grading`, which drives the same
`props.grade_props()` this CLI calls, with a synthetic-lines fixture.

Without `--props-lines` (or the default path, if present), this reports
NEEDS-INSTRUMENTATION and exits cleanly rather than fabricating a scorecard.
Without `--actual-stats` (a verified player-box actual-outcome parquet keyed
on (cbbd_game_id, athlete_id, stat, value) -- the real-player-identity
crosswalk this needs is the same open gap `scripts/eval_gates.py`'s G8
reports NEEDS-INSTRUMENTATION for), grading also reports
NEEDS-INSTRUMENTATION even if props lines exist, rather than grading against
a guessed truth table.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval import contract as C  # noqa: E402
from cbb_sim.eval import props as P  # noqa: E402
from cbb_sim.eval import reference as ref_mod  # noqa: E402
from cbb_sim.eval import report as R  # noqa: E402
from cbb_sim.eval.cli import load_tolerances, tag_from_results_dir  # noqa: E402

OUT_DIR = Path("docs/tests")
DEFAULT_PROPS_LINES = "data/raw/cbbd/props_lines_{season}.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--props-lines", default=None)
    ap.add_argument("--actual-stats", default=None,
                     help="verified player-box actuals parquet: cbbd_game_id, athlete_id, stat, value")
    ap.add_argument("--settle", choices=["book", "fair"], default="book")
    ap.add_argument("--gates-config", default="docs/gates.yaml")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-sealed", action="store_true")
    args = ap.parse_args()

    tol = load_tolerances(Path(args.gates_config))
    engine = C.load_engine_results(args.results, allow_sealed=args.allow_sealed)
    tag = tag_from_results_dir(Path(args.results))
    out_path = Path(args.out) if args.out else OUT_DIR / f"market_props_{tag}_{datetime.now(UTC).date().isoformat()}.md"

    now = datetime.now(UTC)
    L: list[str] = [
        f"# Player-prop market scorecard -- {engine.engine_tag} (season {args.season}, settle={args.settle})",
        "",
        f"Generated {now.date().isoformat()} by `scripts/grade_market_props.py`. Results: `{args.results}`.",
        "",
    ]

    if not engine.has_players:
        L += ["## Status: NEEDS-INSTRUMENTATION", "",
              "players.parquet not found for this engine; there is nothing to grade props against."]
        _write(out_path, L)
        return 0

    props_lines_path = Path(args.props_lines) if args.props_lines else Path(DEFAULT_PROPS_LINES.format(season=args.season))
    if not props_lines_path.exists():
        L += ["## Status: NEEDS-INSTRUMENTATION", "",
              f"No props-lines file at `{props_lines_path}` (none exist yet for this project -- "
              "CLAUDE.md bans scraping the retail props books directly). Schema this script expects "
              "the day one does: see `src/cbb_sim/eval/props.py` module docstring "
              f"(`{', '.join(P.REQUIRED_PROPS_LINES_COLUMNS)}`). "
              "`tests/test_eval.py::test_props_synthetic_grading` proves the grading logic below "
              "against a synthetic lines fixture in the meantime.", ""]
        _write(out_path, L)
        return 0

    props_lines = P.load_props_lines(props_lines_path)
    L += [f"Props lines: `{props_lines_path}` ({len(props_lines)} rows).", ""]

    if not args.actual_stats or not Path(args.actual_stats).exists():
        L += ["## Status: NEEDS-INSTRUMENTATION", "",
              "Props lines are present but no `--actual-stats` verified player-box truth table was "
              "given (schema: cbbd_game_id, athlete_id, stat, value). The player-identity crosswalk "
              "this needs is the same open gap `scripts/eval_gates.py` G8 reports "
              "NEEDS-INSTRUMENTATION for; wiring it in is out of this task's scope. Grading logic is "
              "proven correct against synthetic data by "
              "`tests/test_eval.py::test_props_synthetic_grading`.", ""]
        _write(out_path, L)
        return 0

    import pandas as pd
    actual_stats = pd.read_parquet(args.actual_stats)
    actual_games = ref_mod.load_actual_games(args.season)
    res = P.grade_props(engine.players, props_lines, actual_games, actual_stats, tol,
                         settle=args.settle, n_boot=args.n_boot)

    L += [f"n props graded: {res.n}", "",
          f"push rate {res.push_rate:.4f}, whole-number-line share {res.whole_number_line_share:.4f}, "
          f"zero-sim-usage share {res.zero_usage_share:.4f}", "",
          f"Brier: model {res.model_brier:.5f} vs market {res.market_brier:.5f}", "",
          "## Calibration deciles", ""]
    L += R.md_table(res.calibration) + [""]
    L += ["## ROI by edge bucket (settled at REAL posted odds)", ""]
    L += R.md_table(res.roi_table) + [""]
    if res.bootstrap["n_games"]:
        L += [f"Game-clustered bootstrap ({res.bootstrap['n_boot']} reps, {res.bootstrap['n_games']} bets): "
              f"mean ROI {res.bootstrap['mean']:+.4f}, 95% CI [{res.bootstrap['lo95']:+.4f}, "
              f"{res.bootstrap['hi95']:+.4f}], P(ROI<=0) = {res.bootstrap['p_roi_le_0']:.3f}", ""]
    L += ["## By stat", ""]
    L += R.md_table(res.by_stat) + [""]
    for note in res.notes:
        L += [note, ""]

    _write(out_path, L)
    return 0


def _write(out_path: Path, lines: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    raise SystemExit(main())
