#!/usr/bin/env python
"""
grade_truth_regrade_v1.py -- regrade a stored run under the current truth and the
verified-finals truth (Lane H, 2026-09-30). No sim run; reads results/<tag>/games.parquet.

    python scripts/grade_truth_regrade_v1.py --results results/engine_v0/F2_2025_s200_v5b_A_full --season 2025

Same gate code path as scripts/eval_gates.py; the corrected pass patches
`eval.reference.load_actual_games` to `verified_finals=True` and nothing else.
Gate reference tables under data/reference are NOT rebuilt (G1 / G7 targets are read from them).
G8 reads only the player truth table (independent of finals) and is run once.
Writes results/truth_regrade/<tag>/{current,corrected}.md and side_by_side.md.
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.eval import contract as C  # noqa: E402
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as ref_mod  # noqa: E402
from cbb_sim.eval import report as R  # noqa: E402
from cbb_sim.eval.cli import load_tolerances, tag_from_results_dir  # noqa: E402


def run_all(engine, season, tol, min_cell_n, g8=None):
    summary, raw = G.build_grading_frame(engine.games, season)
    truth_dir = "data/processed/truth"
    res = [
        G.gate_g1(summary, raw, season, tol, min_cell_n),
        G.gate_g2(summary, raw, season, tol, min_cell_n),
        G.gate_g3(summary, raw, season, tol, engine.box_available, min_cell_n, truth_dir=truth_dir),
        G.gate_g4(summary, raw, season, tol, engine.box_available, min_cell_n, truth_dir=truth_dir),
        G.gate_g5(summary, raw, season, tol),
        G.gate_g6(summary, tol, min_cell_n),
        G.gate_g7(summary, season, tol),
        g8 if g8 is not None else G.gate_g8(engine.players, tol, min_cell_n, season=season, truth_dir=truth_dir),
        G.gate_g9(summary, tol, min_cell_n),
    ]
    return summary, res


def write_report(path, title, summary, res):
    L = [f"# {title}", "", f"Games graded: {len(summary)}", ""]
    for gr in res:
        L += R.gate_section(gr)
    L += ["## Summary", "", "| gate | status |", "|---|---|"] + [f"| {g.gate} | {g.status} |" for g in res] + [""]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def tab_diff(a: pd.DataFrame, b: pd.DataFrame):
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return None
    rows = []
    for i in range(len(a)):
        ra, rb = a.iloc[i], b.iloc[i]
        same = True
        for c in a.columns:
            x, y = ra[c], rb[c]
            if isinstance(x, (float, np.floating)) and isinstance(y, (float, np.floating)):
                if not (np.isclose(x, y, rtol=1e-9, atol=1e-12) or (np.isnan(x) and np.isnan(y))):
                    same = False
            elif x != y:
                same = False
        if not same:
            rows.append(i)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--gates-config", default="docs/gates.yaml")
    a = ap.parse_args()
    tol = load_tolerances(Path(a.gates_config))
    mcn = int(tol.get("min_cell_n", G.DEFAULT_MIN_CELL_N))
    engine = C.load_engine_results(a.results)
    tag = tag_from_results_dir(Path(a.results))
    out = ROOT / "results" / "truth_regrade" / tag
    out.mkdir(parents=True, exist_ok=True)

    s0, r0 = run_all(engine, a.season, tol, mcn)
    orig = ref_mod.load_actual_games
    ref_mod.load_actual_games = functools.partial(orig, verified_finals=True)
    try:
        s1, r1 = run_all(engine, a.season, tol, mcn, g8=r0[7])
    finally:
        ref_mod.load_actual_games = orig
    write_report(out / "current.md", f"{tag} season {a.season}: CURRENT truth (default loader)", s0, r0)
    write_report(out / "corrected.md", f"{tag} season {a.season}: CORRECTED truth (verified_finals=True)", s1, r1)

    L = [f"games graded: current {len(s0)}, corrected {len(s1)}", ""]
    for g0, g1 in zip(r0, r1):
        L += [f"### {g0.gate} -- {g0.title}", "",
              "| quantity | current value | corrected value | target (cur / corr) | tolerance | status cur | status corr | changed |",
              "|---|---|---|---|---|---|---|---|"]
        for c0, c1 in zip(g0.checks, g1.checks):
            ch = "STATUS" if c0.status != c1.status else ("value" if (c0.value != c1.value or c0.target != c1.target) else "")
            L.append(f"| {c0.quantity} | {c0.value} | {c1.value} | {c0.target} / {c1.target} | {c0.tolerance} | {c0.status} | {c1.status} | {ch} |")
        L += [f"| **gate overall** | | | | | **{g0.status}** | **{g1.status}** | {'STATUS' if g0.status != g1.status else ''} |", ""]
        for (h0, t0), (h1, t1) in zip(g0.tables, g1.tables):
            d = tab_diff(t0, t1)
            if d is None:
                L += [f"table `{h0}`: shape differs (current {t0.shape}, corrected {t1.shape}); see full reports", ""]
            elif d:
                L += [f"table `{h0}`: {len(d)}/{len(t0)} rows change (current then corrected)", ""]
                both = pd.concat([t0.iloc[d].assign(truth="current"), t1.iloc[d].assign(truth="corrected")]).sort_index(kind="stable")
                L += R.md_table(both) + [""]
            else:
                L += [f"table `{h0}`: identical ({len(t0)} rows)", ""]
    (out / "side_by_side.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
