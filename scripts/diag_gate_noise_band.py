"""diag_gate_noise_band.py -- paired seed-offset noise band between two engine runs.

Given two results dirs that share EVERY config field except the seed range
(e.g. run A seeds 0..N-1 and run B, the seed-offset noise floor, seeds
1000..1000+N-1 -- CLAUDE.md "bake-off before any choice": "A winner must beat
a spec-identical retrain under another seed (the noise floor)"), this runs
the same G1-G9 checks (`cbb_sim.eval.gates`, the exact functions
`scripts/eval_gates.py` calls) on each results dir and prints ONE merged
table: A's own value/target/tolerance/status (the headline read) next to B's
value and the numeric |A-B| gap on every line where both check values carry a
leading float -- the seed-noise band for that gate line, so a FAIL that is
smaller than its own noise band reads differently than one that is not.

This does not decide anything and does not touch either run's output; it is
a report, like `eval_gates.py` and `diag_engine_multilevel.py`.

Usage:
    .venv/Scripts/python.exe scripts/diag_gate_noise_band.py \
        --results-a results/engine_v0/<A> --results-b results/engine_v0/<B> \
        --season 2025 --out docs/tests/gate_noise_band_<tag>_<date>.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval import contract as C  # noqa: E402
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval.cli import load_tolerances  # noqa: E402

DEFAULT_TRUTH_DIR = Path("data/processed/truth")
FLOAT_RE = re.compile(r"[-+]?\d+\.\d+")


def first_float(s: str) -> float | None:
    m = FLOAT_RE.search(str(s))
    return float(m.group()) if m else None


def run_all_gates(results_dir: str, season: int, tol: dict, min_cell_n: int,
                   truth_dir: str, allow_sealed: bool) -> tuple[list[G.GateResult], int]:
    engine = C.load_engine_results(results_dir, allow_sealed=allow_sealed)
    summary, raw = G.build_grading_frame(engine.games, season)
    results = [
        G.gate_g1(summary, raw, season, tol, min_cell_n),
        G.gate_g2(summary, raw, season, tol, min_cell_n),
        G.gate_g3(summary, raw, season, tol, engine.box_available, min_cell_n, truth_dir=truth_dir),
        G.gate_g4(summary, raw, season, tol, engine.box_available, min_cell_n, truth_dir=truth_dir),
        G.gate_g5(summary, raw, season, tol),
        G.gate_g6(summary, tol, min_cell_n),
        G.gate_g7(summary, season, tol),
        G.gate_g8(engine.players, tol, min_cell_n, season=season, truth_dir=truth_dir),
        G.gate_g9(summary, tol, min_cell_n),
    ]
    n_seeds = int(summary["n_seeds"].iloc[0]) if len(summary) else 0
    return results, n_seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-a", required=True, help="headline run (e.g. seeds 0..N-1)")
    ap.add_argument("--results-b", required=True, help="seed-offset noise floor (e.g. seeds 1000..1000+N-1)")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--gates-config", default="docs/gates.yaml")
    ap.add_argument("--truth-dir", default=str(DEFAULT_TRUTH_DIR))
    ap.add_argument("--allow-sealed", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tol = load_tolerances(Path(a.gates_config))
    min_cell_n = int(tol.get("min_cell_n", G.DEFAULT_MIN_CELL_N))

    res_a, n_a = run_all_gates(a.results_a, a.season, tol, min_cell_n, a.truth_dir, a.allow_sealed)
    res_b, n_b = run_all_gates(a.results_b, a.season, tol, min_cell_n, a.truth_dir, a.allow_sealed)

    lines = [
        f"# Gate noise band -- A=`{a.results_a}` ({n_a} seeds) vs B=`{a.results_b}` ({n_b} seeds)",
        "",
        "B is a spec-identical retrain under a different seed offset (the noise floor), "
        "never a second candidate. |A-B| on a line is the seed-noise band for that metric "
        "at this seed count -- a FAIL smaller than its own band is read differently than one "
        "that is not (CLAUDE.md \"backtests must be honest\" / \"bake-off before any choice\").",
        "",
        "| gate | quantity | A (headline) | B (noise floor) | \\|A-B\\| | A status |",
        "|---|---|---|---|---|---|",
    ]
    for ga, gb in zip(res_a, res_b):
        for ca, cb in zip(ga.checks, gb.checks):
            fa, fb = first_float(ca.value), first_float(cb.value)
            gap = f"{abs(fa - fb):.4f}" if fa is not None and fb is not None else "n/a"
            lines.append(f"| {ga.gate} | {ca.quantity} | {ca.value} | {cb.value} | {gap} | {ca.status} |")

    out_path = (Path(a.out) if a.out else
                Path("docs/tests") / f"gate_noise_band_{Path(a.results_a).name}_vs_{Path(a.results_b).name}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    for ln in lines:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
