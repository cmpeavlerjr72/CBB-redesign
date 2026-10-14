"""grade_endgame_clock_v2_tables_v1.py -- clock experiments.md s46 blocks A and B: READS the grade files written by
`grade_endgame_clock_v2.py` (offline_A.json, offline_B.json, m_selection.json, reseed_proof.json) and prints markdown
tables. Computes nothing from predictions or sim output; adopts nothing.

    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2_tables_v1.py > results/endgame_clock_v2/tables_offline.md
"""
from __future__ import annotations

import json
from pathlib import Path

G = Path(__file__).resolve().parents[1] / "results/endgame_clock_v2/grade"
LINES_A = ("dev_pos", "cal", "dev", "cal45", "slope_tempo", "slope_zoff", "slope_defz", "slope_team_lead",
           "slope_team_trail", "slope_lead", "slope_trail", "dev_end", "pergame_abs_err")
ROLE_A = {"dev_pos": "PRIMARY", "cal": "PRIMARY", "dev": "reported (i)", "cal45": "reported",
          "slope_tempo": "HARD GATE (not below)", "slope_zoff": "gate (not below)", "slope_defz": "gate (not below)",
          "slope_team_lead": "gate (not below)", "slope_team_trail": "gate (not below)",
          "slope_lead": "gate (|s-1| not worse)", "slope_trail": "gate (|s-1| not worse)", "dev_end": "reported",
          "pergame_abs_err": "reported (per game)"}
LINES_B = ("mll", "make_dev", "share_3pa", "share_2pa", "share_tov", "share_ft", "make_rate", "shot_pts_per_shot",
           "slope_team_3pa", "slope_team_make")
ROLE_B = {"mll": "PRIMARY", "make_dev": "PRIMARY", "slope_team_3pa": "gate (not below; UNDERPOWERED)",
          "slope_team_make": "gate (not below; UNDERPOWERED)"}


def f(x, d=4):
    return "nan" if x is None or x != x else f"{x:.{d}f}"


def verdict(k, d, role):
    if role.startswith("PRIMARY"):
        return "PASS (beats by > floor)" if d["delta"] < -d["floor"] else "FAIL (inside floor or worse)"
    if "|s-1|" in role:
        return "hold" if d["d_abs_err"] <= d["floor_abs_err"] else "FAIL"
    if role.startswith("gate") or role.startswith("HARD"):
        return "hold" if d["delta"] >= -d["floor"] else "FAIL"
    return "REPORTED"


def block_a(r):
    L = []
    for fold in ("F2", "F1"):
        for per in ("2", "1"):
            P = r["folds"][fold]["periods"][per]
            tag = "selection" if (fold == "F2" and per == "2") else ("confirm" if per == "2" else "period 1, REPORTED only")
            L += [f"#### Block A, {fold} period {per} ({tag}): {P['n_rows']:,} rows, {P['n_games']:,} games; control "
                  f"zero-mass rows excluded from dev_pos: {P['excluded_zero_mass_rows']} ({100 * P['excluded_share']:.2f}%, "
                  f"carrying {P['excluded_dev_contribution']:.4f} of the control's `dev`); powered cells "
                  f"{P['n_cells_powered']} of {P['n_cells_populated']}", "",
                  "| line | role | actual / target | CA0 (control) | CA0s1 (floor arm) | CA1 | CA2 | CA3 |", "|---|---|---|---:|---:|---:|---:|---:|"]
            c = P["compare"]
            for k in LINES_A:
                row = [k, ROLE_A[k], "1" if k.startswith("slope_") else "lower", f(c["CA1"][k]["ctrl"])]
                for a in ("CA0s1", "CA1", "CA2", "CA3"):
                    d = c[a][k]
                    v = verdict(k, d, ROLE_A[k]) if (fold, per) == ("F2", "2") and a != "CA0s1" else ""
                    row.append(f"{f(d['arm'])} ({d['delta']:+.4f}; fl {d['floor']:.4f}) {v}".strip())
                L.append("| " + " | ".join(row) + " |")
            L.append("")
    return L


def cells_a(r):
    L = ["#### Block A per-cell calibration, F2 period 2 (mean first-chance seconds, E[min(T, R)]; U = UNDERPOWERED, < 200 rows)", "",
         "| cell | n | actual | CA0 | CA1 | CA2 | CA3 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for x in r["folds"]["F2"]["periods"]["2"]["cells"]:
        L.append(f"| {x['cell'].replace(' | ', ' x ')}{' U' if x['underpowered'] else ''} | {x['n']} | {f(x['actual_s'], 2)} | {f(x['CA0'], 2)} | "
                 f"{f(x['CA1'], 2)} | {f(x['CA2'], 2)} | {f(x['CA3'], 2)} |")
    L += ["", "(30,60] bucket, the s46 finding (1) cells:", "", "| cell | n | actual | CA0 | CA1 | CA2 | CA3 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for x in r["folds"]["F2"]["periods"]["2"]["cells"]:
        if "(30,60]" in x["cell"]:
            L.append(f"| {x['cell'].replace(' | ', ' x ')}{' U' if x['underpowered'] else ''} | {x['n']} | {f(x['actual_s'], 2)} | {f(x['CA0'], 2)} | "
                     f"{f(x['CA1'], 2)} | {f(x['CA2'], 2)} | {f(x['CA3'], 2)} |")
    L += ["", "#### Block A per possession type (start type), F2 period 2", "", "| start | n | actual | CA0 | CA1 | CA2 | CA3 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for x in r["folds"]["F2"]["periods"]["2"]["start_types"]:
        L.append(f"| {x['start']}{' U' if x['underpowered'] else ''} | {x['n']} | {f(x['actual_s'], 2)} | {f(x['CA0'], 2)} | "
                 f"{f(x['CA1'], 2)} | {f(x['CA2'], 2)} | {f(x['CA3'], 2)} |")
    ts = r["folds"]["F2"]["periods"]["2"]["team_sign"]
    L += ["", f"#### Block A per team, F2 period 2 ({ts['n_teams']} offence teams with >= 200 window rows)", "",
          "| arm | share of teams moved in the actual's direction | share with |error| reduced |", "|---|---:|---:|"]
    for a in ("CA0s1", "CA1", "CA2", "CA3"):
        L.append(f"| {a} | {f(ts[a]['right_sign_share'], 3)} | {f(ts[a]['abs_err_down_share'], 3)} |")
    return L + [""]


def block_b(r):
    L = [f"Rows: F2 {r['folds']['F2']['n_rows']:,} FPS first chances ({r['folds']['F2']['make_rows_matched']} of "
         f"{r['folds']['F2']['fga_rows']} shots matched to an fg_make design row); F1 {r['folds']['F1']['n_rows']:,} "
         f"({r['folds']['F1']['make_rows_matched']} of {r['folds']['F1']['fga_rows']}).", ""]
    for fold in ("F2", "F1"):
        for per in ("2", "1"):
            for cell, P in r["folds"][fold]["periods"][per].items():
                prim = fold == "F2" and per == "2" and cell == "trail 1-3"
                tag = "PRIMARY cell, selection" if prim else ("confirm" if (per == "2" and cell == "trail 1-3") else "REPORTED")
                L += [f"#### Block B, {fold} period {per}, {cell} ({tag}): {P['n_rows']} rows, {P['n_games']} games, "
                      f"{P['n_make_rows']} shot rows; slopes UNDERPOWERED (< 3,000 rows)"
                      + ("; cell UNDERPOWERED" if P["underpowered_cell"] else ""), "",
                      "| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
                c = P["compare"]
                for k in LINES_B:
                    act = P["ctrl"].get(f"act_{k}")
                    role = ROLE_B.get(k, "reported")
                    row = [k, role, f(act) if act is not None else "", f(P["ctrl"][k])]
                    for a in ("FP0s1", "FP1", "FP2", "FP3"):
                        d = c[a][k]
                        v = verdict(k, d, role) if prim and a != "FP0s1" else ""
                        row.append(f"{f(d['arm'])} ({d['delta']:+.4f}; fl {d['floor']:.4f}) {v}".strip())
                    L.append("| " + " | ".join(row) + " |")
                L.append("")
    return L


def main() -> int:
    a = json.loads((G / "offline_A.json").read_text(encoding="utf-8"))
    b = json.loads((G / "offline_B.json").read_text(encoding="utf-8"))
    m = json.loads((G / "m_selection.json").read_text(encoding="utf-8"))
    rp = json.loads((G / "reseed_proof.json").read_text(encoding="utf-8"))
    L = ["## Shrink grid (ruling 7, fold 1 only)", "", f"Rule: {m['rule']}.", "",
         "| arm | m = 20 dev_pos | m = 50 | m = 100 | within one F1 floor | chosen m |", "|---|---:|---:|---:|---|---:|"]
    for arm, x in m["arms"].items():
        v = [f(x["f1_lines"][f"{arm}_m{k}"]["dev_pos"]["arm"], 5) for k in (20, 50, 100)]
        L.append(f"| {arm} | {' | '.join(v)} | {', '.join(x['within_floor'])} | {x['m']} |")
    L += ["", "## Reseed proof (max abs difference)", "", "| pair | max abs diff |", "|---|---:|"]
    L += [f"| {k} | {v:.3g} |" for k, v in rp.items()]
    L += ["", "## Block A (offline)", ""] + block_a(a) + cells_a(a)
    L += ["## Block A decision (literal s46 4.1)", "", "```", json.dumps(a["decision"], indent=1), "```", ""]
    L += ["## Block B (offline)", ""] + block_b(b)
    L += ["## Block B decision (literal s46 4.1)", "", "```", json.dumps(b["decision"], indent=1), "```"]
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
