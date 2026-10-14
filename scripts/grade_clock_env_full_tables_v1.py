"""grade_clock_env_full_tables_v1.py -- clock round 10 full-size read (docs/models/clock/experiments.md sections 43-45): READS the
two grade files written by grade_clock_env_full_v2.py and prints the result tables with the PM rulings of 2026-10-13 / 2026-10-14
applied ON TOP of the grader's literal section-43 verdict. Computes nothing from sim output itself; adopts nothing.
Written and committed BEFORE any full-size arm output existed.

  deciding read  (--deciding): every line at 200 vs 200, floor from four 200-seed control draws (PM ruling 2026-10-14)
  50-seed-draw read (--d50):   v1's definition (trajectory lines on the first 50 seeds, draws read on their first 50 seeds,
                               200-seed lines with the sqrt(50/200)-scaled draw SD), reported beside it

Rulings applied (PROJECT_STATUS 2026-10-13 "PM RULINGS for the full-size read", items 2-4):
  R2 first-half magnitude line: H1 possession slope on env must be closer to the actual than the control's by more than its
     floor; an overshoot beyond the actual by more than the control's shortfall is a FAIL.
  R3 dispersion by |spread| quintile: the 0.009 band read at the full-size floor; both the literal 0.009 reading and the
     max(0.009, floor) band reading are printed (Q3 and Q5 were the 500 x 25 flags).
  R4 env-quintile possession gap (real - sim): FAIL if the arm's gap is further from 0 than the control's by more than the
     cell floor in more than two of the five quintiles.

    .venv/Scripts/python.exe scripts/grade_clock_env_full_tables_v1.py --deciding results/clock_env10/grade_full_s200_d200.json \
        --d50 results/clock_env10/grade_full_s200_d50.json --out results/clock_env10/tables_full.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(x, d=4):
    return "nan" if x is None else f"{x:.{d}f}"


def r2(res):
    h = res["verdict"]["vetoes"]["H1_poss_on_env"]
    tv, c, a, fl = h["target"], h["ctrl"], h["arm"], h["floor"]
    shortfall = abs(tv - c)
    overshoot = (a - tv) if (tv - c) > 0 else (tv - a)  # signed distance beyond the actual, on the far side from the control
    closer_by_floor = bool(abs(c - tv) - abs(a - tv) > fl)
    over_fail = bool(overshoot > shortfall)
    return {"actual": tv, "ctrl": c, "arm": a, "floor": fl, "improve": abs(c - tv) - abs(a - tv),
            "overshoot_beyond_actual": overshoot, "ctrl_shortfall": shortfall, "closer_by_floor": closer_by_floor,
            "overshoot_fail": over_fail, "pass": bool(closer_by_floor and not over_fail)}


def r4(res):
    out, n_bad = [], 0
    for q in res["quintiles"]["envq"]:
        c, a, fl = q["poss_gap_ctrl"], q["poss_gap_arm"], q["poss_gap_floor"]
        worse = bool(abs(a) - abs(c) > fl)
        n_bad += worse
        out.append({"q": q["q"], "n": q["n_games_poss"], "ctrl": c, "arm": a, "floor": fl, "further_beyond_floor": worse,
                    "underpowered": q["underpowered"]})
    return {"cells": out, "n_further": n_bad, "pass": bool(n_bad <= 2)}


def table(res, title):
    v, rows = res["verdict"], res["rows"]
    L = [f"### {title}", "", f"games {res['n_games']} (pbp ok {res['n_pbp_ok']}, round-8 frame {res['n_r8_games']}); "
         f"trajectory seeds {res['seeds_traj']}; draws {', '.join(res['draws'])}", "",
         "| line | actual / target | control | E1 | delta | floor (draw SD; draw SD raw; boot SE) | improve/floor | result |",
         "|---|---:|---:|---:|---:|---:|---:|---|"]

    def fl(k):
        r = rows[k]
        return f"{f(r['floor'])} ({f(r['draw_sd'])}; {f(r['draw_sd_raw'])}; {f(r['boot_se'])})"

    for k, x in {**v["primary"], **v["primary_200_crosscheck"]}.items():
        prim = k in v["primary"]
        tag = "PRIMARY" if prim else "cross-check (200 seeds, games.parquet)"
        res_ = ("PASS" if x["toward_by_floor"] else "FAIL") if prim else ("toward by floor" if x["toward_by_floor"] else "not toward by floor")
        L.append(f"| {tag} {k} | {f(x['target'])} | {f(x['ctrl'])} | {f(x['arm'])} | {f(x['arm'] - x['ctrl'])} | {fl(k)} | "
                 f"{x['floors']:+.1f} | {res_} |")
    for k, x in v["vetoes"].items():
        if k == "REG_regmargin_sd_rise":
            L.append(f"| veto {k} | | {f(x['ctrl'])} | {f(x['arm'])} | {f(x['delta'])} | {fl('disp_regmargin')} | | "
                     f"{'pass' if x['pass'] else 'FAIL'} |")
            continue
        if "arm_in_0974_1014" in x:
            L.append(f"| veto {k} | 0.994 baseline | {f(x['ctrl'])} | {f(x['arm'])} | {f(x['delta'])} | {fl(k)} | | "
                     f"{'pass' if x['pass'] else 'FAIL'}; band {f(x['band'])}; arm in [0.974,1.014]: {x['arm_in_0974_1014']}; "
                     f"ctrl in: {x['ctrl_in_0974_1014']} |")
            continue
        extra = ""
        if "pass_literal_0009" in x:
            extra = f"; band {f(x['band'])}; literal 0.009: {'pass' if x['pass_literal_0009'] else 'FAIL'}"
        if "in_095_105" in x:
            extra = f"; in [0.95,1.05]: {x['in_095_105']}"
        if k == "H1_poss_on_env":
            extra = " (direction-only registered reading; see R2)"
        fl_ = f"{x['improve'] / x['floor']:+.1f}" if x["floor"] else ""
        L.append(f"| veto {k} | {f(x['target'])} | {f(x['ctrl'])} | {f(x['arm'])} | {f(x['arm'] - x['ctrl'])} | {fl(k)} | {fl_} | "
                 f"{'pass' if x['pass'] else 'FAIL'}{extra} |")
    for k, x in v["reported"].items():
        fl_ = f"{x['improve'] / x['floor']:+.1f}" if x["floor"] else ""
        L.append(f"| reported {k} | {f(x['target'])} | {f(x['ctrl'])} | {f(x['arm'])} | {f(x['arm'] - x['ctrl'])} | {fl(k)} | {fl_} | reported |")
    ps = rows["disp_pace_share"]
    L.append(f"| reported disp_pace_share | | {f(ps['ctrl'])} | {f(ps['arm'])} | {f(ps['delta'])} | | | reported |")
    L += ["", f"Literal section-43 verdict: {'CANDIDATE' if v['candidate'] else 'NOT A CANDIDATE'}; primary_moves "
          f"{v['primary_moves']}; vetoes failing {v['vetoes_failing'] or 'none'}", ""]
    a2, a4 = r2(res), r4(res)
    L += [f"R2 first-half magnitude: actual {f(a2['actual'])}, control {f(a2['ctrl'])}, E1 {f(a2['arm'])}, floor {f(a2['floor'])}; "
          f"closer by {f(a2['improve'])} ({'> floor' if a2['closer_by_floor'] else 'NOT > floor'}); overshoot beyond actual "
          f"{f(a2['overshoot_beyond_actual'])} vs control shortfall {f(a2['ctrl_shortfall'])} -> {'PASS' if a2['pass'] else 'FAIL'}", "",
          "R4 env-quintile possession gap real - sim (arm further from 0 than control beyond floor in more than two quintiles = FAIL):", "",
          "| env q | games | control gap | E1 gap | floor | further beyond floor |", "|---|---:|---:|---:|---:|---|"]
    for c in a4["cells"]:
        L.append(f"| Q{c['q']} | {c['n']}{' UNDERPOWERED' if c['underpowered'] else ''} | {f(c['ctrl'], 3)} | {f(c['arm'], 3)} | "
                 f"{f(c['floor'], 3)} | {c['further_beyond_floor']} |")
    L += ["", f"R4: {a4['n_further']} quintile(s) further beyond floor -> {'PASS' if a4['pass'] else 'FAIL'}", ""]
    L += ["Quintile tables (gap = real - sim; floor per cell):", "",
          "| table | q | games poss / total | real poss | ctrl poss | E1 poss | poss gap ctrl / E1 (floor) | real total | ctrl total | E1 total | total gap ctrl / E1 (floor) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for nm, tab in res["quintiles"].items():
        for q in tab:
            L.append(f"| {nm} | Q{q['q']} | {q['n_games_poss']} / {q['n_games_total']}{' UNDERPOWERED' if q['underpowered'] else ''} | "
                     f"{f(q['real_poss'], 2)} | {f(q['poss_sim_ctrl'], 2)} | {f(q['poss_sim_arm'], 2)} | "
                     f"{f(q['poss_gap_ctrl'], 2)} / {f(q['poss_gap_arm'], 2)} ({f(q['poss_gap_floor'], 2)}) | {f(q['real_total'], 2)} | "
                     f"{f(q['total_sim_ctrl'], 2)} | {f(q['total_sim_arm'], 2)} | {f(q['total_gap_ctrl'], 2)} / {f(q['total_gap_arm'], 2)} "
                     f"({f(q['total_gap_floor'], 2)}) |")
    L += ["", "Per possession type (start type), regulation seconds per possession:", "",
          "| start type | games | real rows | real s | ctrl s | E1 s | floor |", "|---|---:|---:|---:|---:|---:|---:|"]
    for lv, s in res["start_type"].items():
        L.append(f"| {lv} | {s['games']}{' UNDERPOWERED' if s['games'] < 150 else ''} | {s['real_rows']} | {f(s['real_sec'], 2)} | "
                 f"{f(s['ctrl_sec'], 2)} | {f(s['arm_sec'], 2)} | {f(s['floor'], 3)} |")
    L += ["", "Per team (offence-team quintile of x_off), regulation possessions per team-game:", "",
          "| quintile | teams | team-games | x_off | real | control | E1 | right-sign share |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in res["team_table"]:
        if t["quintile"] == "ALL":
            L.append(f"| ALL | {t['n_teams']} | {t['n_team_games']} | | | | | {f(t['right_sign_share'], 3)} "
                     f"(abs err down share {f(t['abs_err_down_share'], 3)}) |")
        else:
            L.append(f"| Q{t['quintile']} | {t['n_teams']} | {t['n_team_games']}{' UNDERPOWERED' if t['underpowered'] else ''} | "
                     f"{f(t['x_off_mean'], 3)} | {f(t['real'], 2)} | {f(t['ctrl'], 2)} | {f(t['arm'], 2)} | {f(t['right_sign_share'], 3)} |")
    sc = res["seed_check"]
    L += ["", "Seed check (2 x draw SD vs control gap): " + "; ".join(
        f"{k} floor {f(x['floor_draws'])} gap {f(x['gap'])} {'resolved' if x['resolved'] else 'UNRESOLVED'}"
        for k, x in sc.items() if isinstance(x, dict)), ""]
    return L, {"R2": a2, "R4": a4, "literal_candidate": v["candidate"], "vetoes_failing": v["vetoes_failing"],
               "primary_moves": v["primary_moves"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deciding", required=True)
    ap.add_argument("--d50", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, summ = ["# Clock round 10 full-size read: tables", ""], {}
    for nm, p, title in (("deciding_200", a.deciding, "DECIDING read: 5,710 x 200, four 200-seed control draws"),
                         ("draws_50", a.d50, "50-seed-draw read (v1 definition): trajectory lines on the first 50 seeds, draws read on their first 50 seeds")):
        res = json.loads(Path(p).read_text(encoding="utf-8"))
        L, s = table(res, title)
        out += L
        summ[nm] = s
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    for nm, s in summ.items():
        print(nm, "literal:", "CANDIDATE" if s["literal_candidate"] else "NOT A CANDIDATE", "| vetoes failing", s["vetoes_failing"],
              "| R2", "PASS" if s["R2"]["pass"] else "FAIL", "| R4", "PASS" if s["R4"]["pass"] else "FAIL", s["R4"]["n_further"])


if __name__ == "__main__":
    main()
