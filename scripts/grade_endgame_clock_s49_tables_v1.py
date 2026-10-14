"""grade_endgame_clock_s49_tables_v1.py -- markdown tables from the s49 offline grade
(`results/endgame_clock_v2/grade/s49_offline.json`, written by `grade_endgame_clock_v2.py s49-offline`). Formatting
only: no line is computed here. Writes `results/endgame_clock_v2/grade/s49_offline_tables.md`.

    .venv/Scripts/python.exe scripts/grade_endgame_clock_s49_tables_v1.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "results/endgame_clock_v2/grade"
ARMS = ("CA1", "CA1E")
SRC = ("act", "CA0", "CA1", "CA1E")


def f(x, nd=3, sign=True):
    if x is None or x != x:
        return "nan"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def lab(c: str) -> str:
    return c.replace(" | ", " x ")


def main() -> int:
    r = json.loads((G / "s49_offline.json").read_text(encoding="utf-8"))
    md = ["# s49 offline tables (CA0 control, CA0s1 noise floor, CA1, CA1E)", ""]
    for fold in ("F2", "F1"):
        F = r["folds"][fold]
        p = F["periods"]["2"]
        md += [f"## {fold}{' (selects)' if fold == 'F2' else ' (confirms)'}: period-2 window rows {p['n_rows']:,}, "
               f"games {p['n_games']:,}, control zero-mass rows {p['excluded_zero_mass_rows']}", "",
               "### Primaries and reported deviance lines (delta vs CA0; floor)", "",
               "| arm | dev_pos (P1) | cal (P2) | dev (all rows) | dev_end | cal45 | per-game abs err |",
               "|---|---|---|---|---|---|---|"]
        for a in ("CA0s1", *ARMS):
            c = p["compare"][a]
            md.append(f"| {a} | " + " | ".join(f"{c[k]['arm']:.4f} ({c[k]['delta']:+.4f}; {c[k]['floor']:.4f})"
                                                for k in ("dev_pos", "cal", "dev", "dev_end", "cal45",
                                                          "pergame_abs_err")) + " |")
        md += ["", f"Control CA0: dev_pos {p['compare']['CA1']['dev_pos']['ctrl']:.4f}, cal "
                   f"{p['compare']['CA1']['cal']['ctrl']:.4f}.", "",
               "### Actual-referenced gates (excess = |arm - act| - |ctrl - act|; floor = sqrt(paired^2 + SE_act^2))", "",
               "| line | actual (SE) | control | arm | arm | excess | floor | verdict |", "|---|---|---|---|---|---|---|---|"]
        for a in ARMS:
            for k, g in p["s49_gates"][a].items():
                md.append(f"| {k} | {f(g['actual'], 4)} ({g['se_actual']:.4f}) | {f(g['ctrl'], 4)} | {a} | "
                          f"{f(g['arm'], 4)} | {f(g['excess'], 4)} | {g['floor']:.4f} | {'PASS' if g['pass'] else 'FAIL'} |")
        g1 = F["g1_sd"]
        md += ["", f"G1 possession SD (loop translation; control 5.629, actual 5.474; offline actual SD "
                   f"{g1['act_offline_sd']:.3f}, offline control SD {g1['ctrl_offline_sd']:.3f}, {g1['n_games']:,} games):", ""]
        for a in ARMS:
            x = g1[a]
            md.append(f"- {a}: offline SD {x['offline_sd']:.3f}, translation {x['loop_translation']:.3f}, excess "
                      f"{x['excess']:+.3f}, floor {x['floor']:.3f} -> {'PASS' if x['pass'] else 'FAIL'}")
        if "ca1e_vs_ca1" in p:
            c = p["ca1e_vs_ca1"]
            md += ["", f"CA1E vs CA1 (preference rule, reported): dev_pos {c['dev_pos_delta']:+.4f} (floor "
                       f"{c['dev_pos_floor']:.4f}); within-cell lines closer beyond floor: {c['wc_closer_beyond_floor']}"]
        md += ["", "### Per-cell strength response (row OLS slope of seconds on offence prior strength, s/pt; within-cell "
                   "strength Q5 - Q1, s). `*` = explicit cell (two early trailer-13+ cells, the (30,60] bucket)", "",
               "| cell | rows (Q1/Q5) | power | slope act / CA0 / CA1 / CA1E | Q5-Q1 act / CA0 / CA1 / CA1E |",
               "|---|---|---|---|---|"]
        for c in p["cells_s49"]:
            md.append(f"| {'*' if c['explicit'] else ''}{lab(c['cell'])} | {c['n']:,} ({c['n_q1']}/{c['n_q5']}) | "
                      f"{c['slope_power']}{'' if c['cell_power'] == 'ok' else ', cell UNDERPOWERED'} | "
                      + " / ".join(f(c[f'slope_{s}'], 4) for s in SRC) + " | "
                      + " / ".join(f(c[f'q5_q1_{s}'], 2) for s in SRC) + " |")
        md += ["", "### Cell calibration (mean seconds; cells < 200 rows UNDERPOWERED)", "",
               "| cell | rows | actual | CA0 | CA1 | CA1E |", "|---|---:|---:|---:|---:|---:|"]
        for c in p["cells"]:
            md.append(f"| {lab(c['cell'])}{' (UNDERPOWERED)' if c['underpowered'] else ''} | {c['n']:,} | {c['actual_s']:.2f} | "
                      f"{c['CA0']:.2f} | {c['CA1']:.2f} | {c['CA1E']:.2f} |")
        md += ["", "### Start type (mean seconds)", "", "| start | rows | actual | CA0 | CA1 | CA1E |",
               "|---|---:|---:|---:|---:|---:|"]
        for c in p["start_types"]:
            md.append(f"| {c['start']} | {c['n']:,} | {c['actual_s']:.2f} | {c['CA0']:.2f} | {c['CA1']:.2f} | {c['CA1E']:.2f} |")
        ts = p["team_sign"]
        md += ["", f"Team level (offence teams with >= 200 rows, n = {ts['n_teams']}): " + "; ".join(
            f"{a} right-sign share {ts[a]['right_sign_share']:.3f}, |err| down share {ts[a]['abs_err_down_share']:.3f}"
            for a in ARMS if a in ts), ""]
        p1 = F["periods"].get("1")
        if p1:
            md += ["### Period 1 (reported)", "", "| arm | dev_pos (d; floor) | cal (d; floor) |", "|---|---|---|"]
            for a in ARMS:
                c = p1["compare"][a]
                md.append(f"| {a} | {c['dev_pos']['arm']:.4f} ({c['dev_pos']['delta']:+.4f}; {c['dev_pos']['floor']:.4f}) | "
                          f"{c['cal']['arm']:.4f} ({c['cal']['delta']:+.4f}; {c['cal']['floor']:.4f}) |")
            md.append("")
    d = r["decision"]
    md += ["## Decision (literal s49)", "", f"Rule: {d['rule']}", ""]
    for a, x in d["per_arm"].items():
        md.append(f"- {a}: beats CA0 by floor {x['beats_ctrl_by_floor']} ({', '.join(f'{k} {v:.1f} floors' for k, v in x['primary_floors'].items())}); "
                  f"F1 keeps sign {x['f1_keeps_sign']}; gates failing {x['gates_failing']}; wins {x['wins']}")
    md += ["", f"Winners: {d['winners']}; verdict: **{d['verdict']}**", ""]
    (G / "s49_offline_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
