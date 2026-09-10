"""
eval_gates.py -- engine-agnostic G1-G9 gate report.

    .venv/Scripts/python.exe scripts/eval_gates.py --results results/<tag> --season 2025

Reads `results/<tag>/games.parquet` (+ optional `players.parquet`) through the
results contract (`src/cbb_sim/eval/contract.py`), grades it against the
season truth tables (`src/cbb_sim/eval/reference.py`) with G1-G9
(`src/cbb_sim/eval/gates.py`, SIM_GUARDRAILS.md section 3), and writes
`docs/tests/gates_<tag>_<date>.md`: one section per gate ending in a literal
PASS / FAIL / NEEDS-INSTRUMENTATION, plus a summary tally.

This generalises `scripts/grade_control.py` (the prototype, read but never
imported here): any engine that satisfies the contract gets the same report,
not just the Control. A gate whose required input the results file lacks
(an optional box column pair, players.parquet) reports NEEDS-INSTRUMENTATION,
never a fake PASS -- see `gates.GateResult.status`.

Nothing here adjusts an engine. CLAUDE.md "no hand tuning on engine output".
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval import contract as C  # noqa: E402
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import report as R  # noqa: E402
from cbb_sim.eval.cli import load_tolerances, tag_from_results_dir  # noqa: E402

DEFAULT_GATES_YAML = Path("docs/gates.yaml")
OUT_DIR = Path("docs/tests")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results/<engine_tag> directory")
    ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--gates-config", default=str(DEFAULT_GATES_YAML))
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-sealed", action="store_true",
                     help="grade a run whose run_meta.json has sealed_touched=True (deliberate override)")
    args = ap.parse_args()

    tol = load_tolerances(Path(args.gates_config))
    min_cell_n = int(tol.get("min_cell_n", G.DEFAULT_MIN_CELL_N))

    engine = C.load_engine_results(args.results, allow_sealed=args.allow_sealed)
    tag = tag_from_results_dir(Path(args.results))
    out_path = Path(args.out) if args.out else OUT_DIR / f"gates_{tag}_{datetime.now(UTC).date().isoformat()}.md"

    summary, raw = G.build_grading_frame(engine.games, args.season)

    results: list[G.GateResult] = []
    results.append(G.gate_g1(summary, raw, args.season, tol, min_cell_n))
    results.append(G.gate_g2(summary, raw, args.season, tol, min_cell_n))
    results.append(G.gate_g3(summary, raw, args.season, tol, engine.box_available, min_cell_n))
    results.append(G.gate_g4(summary, raw, args.season, tol, engine.box_available, min_cell_n))
    results.append(G.gate_g5(summary, raw, args.season, tol))
    results.append(G.gate_g6(summary, tol, min_cell_n))
    results.append(G.gate_g7(summary, args.season, tol))
    results.append(G.gate_g8(engine.players, tol, min_cell_n))
    results.append(G.gate_g9(summary, tol, min_cell_n))

    now = datetime.now(UTC)
    L: list[str] = [
        f"# Gate report -- {engine.engine_tag} (season {args.season})",
        "",
        f"Generated {now.date().isoformat()} by `scripts/eval_gates.py`. "
        f"Results: `{args.results}`. Tolerances: `{args.gates_config}` "
        "(provisional until the seed-noise study, `docs/SIM_GUARDRAILS.md` section 3). "
        "Every gate line ends in a literal PASS / FAIL / NEEDS-INSTRUMENTATION.",
        "",
        f"Games graded: {len(summary)} (of {engine.run_meta.get('n_games', 'n/a')} in the run; "
        "the difference is games the truth tables exclude as non-D-I or pbp-truncated, or that "
        "the run itself does not cover). Seeds per game: "
        f"{int(summary['n_seeds'].iloc[0]) if len(summary) else 'n/a'}. "
        f"run_meta: fold={engine.run_meta.get('fold')}, backtest={engine.run_meta.get('backtest')}, "
        f"sealed_touched={engine.run_meta.get('sealed_touched')}.",
        "",
    ]
    if engine.warnings:
        L += ["## Contract notices", ""]
        for w in engine.warnings:
            L += [f"- {w}"]
        L += [""]

    for gr in results:
        L += R.gate_section(gr)

    L += ["## Summary", ""]
    tally: dict[str, int] = {}
    for gr in results:
        tally[gr.status] = tally.get(gr.status, 0) + 1
    L += ["| status | count |", "|---|---|"]
    for st in ("PASS", "FAIL", "NEEDS-INSTRUMENTATION"):
        L.append(f"| {st} | {tally.get(st, 0)} |")
    L += [""]
    L += ["| gate | status |", "|---|---|"]
    for gr in results:
        L.append(f"| {gr.gate} | {gr.status} |")
    L += [""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(L)} lines)")
    print("gate summary: " + ", ".join(f"{gr.gate}={gr.status}" for gr in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
