"""
diag_clk3c_tables.py -- markdown tables for the round-3c closed-loop report.

Reads the JSON `scripts/grade_clk3c_closed_loop.py` writes (one file per seed
count) and emits the tables `docs/tests/clock_round3c_closed_loop_2026-09-10.md`
carries. It reformats; it computes nothing the grader did not already compute,
except the arm-vs-floor and arm-vs-incumbent differences, which are subtractions
of numbers already in the file.

    .venv/Scripts/python.exe scripts/diag_clk3c_tables.py \
        --s5 data/processed/models/clock/v3c_closed_loop_s5.json \
        --s25 data/processed/models/clock/v3c_closed_loop_s25.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Display order and label, by `ENGINE_CLOCK` value plus the freeze flag.
ORDER = [
    ("reference", False, "I", "incumbent: round-1 lgbm_quantile, static"),
    ("v3c_srfloor_P3_s1", False, "B3", "srfloor P3 + S1 (cell-based, engine-safe)"),
    ("v3c_srfloor_P1_s1", False, "B1", "srfloor P1 + S1 (cell-based, margin live)"),
    ("v3c_gamma_P3_s1", False, "A3", "gamma_aft P3 + S1 (engine-safe end-game)"),
    ("v3c_gamma_P2_s1", False, "A2", "gamma_aft P2 + S1 (margin deleted)"),
    ("v3c_gamma_P1_s1", False, "A1", "gamma_aft P1 + S1 (margin live)"),
    ("v3c_gamma_P1_s1", True, "F", "gamma_aft P1 + S1, margin FROZEN (ablation)"),
    ("v3c_gamma_P3_s1", True, "F3", "gamma_aft P3 + S1, margin FROZEN (added diagnostic)"),
]


def pick(arms: dict, arm: str, freeze: bool, suffix: str = "") -> dict | None:
    for r in arms.values():
        if r["arm"] != arm or bool(r["freeze"]) != freeze:
            continue
        if r["seed_offset"] != 0:
            continue
        if suffix and not r["tag"].endswith(suffix):
            continue
        return r
    return None


def floor_of(arms: dict, arm: str, freeze: bool = False) -> dict | None:
    for r in arms.values():
        if r["arm"] == arm and bool(r["freeze"]) == freeze and r["seed_offset"] != 0:
            return r
    return None


def f(x, d=3, sign=True) -> str:
    if x is None:
        return "--"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if v != v:
        return "--"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5", required=True)
    ap.add_argument("--s25", default="")
    a = ap.parse_args()

    d5 = json.loads(Path(a.s5).read_text(encoding="utf-8"))
    d25 = json.loads(Path(a.s25).read_text(encoding="utf-8")) if a.s25 else None
    act = d5["actual"]

    def block(doc, label):
        arms = doc["arms"]
        print(f"\n### {label}\n")
        print("| id | arm | G1 cc mean | G1 cc SD | G1 all mean | G1 all SD | "
              "margin SD | corr(h,a) | total bias | PPP | OT% |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm, fr, aid, name in ORDER:
            r = pick(arms, arm, fr)
            if r is None:
                continue
            print(f"| {aid} | {name} | {f(r['G1_cc'].get('mean_delta'))} | "
                  f"{f(r['G1_cc'].get('sd_delta'))} | {f(r['G1_all'].get('mean_delta'))} | "
                  f"{f(r['G1_all'].get('sd_delta'))} | {f(r['margin_sd'], 3, False)} | "
                  f"{f(r['corr_home_away'], 3, False)} | {f(r['total_bias'])} | "
                  f"{f(r['ppp'], 4, False)} | {f(100 * r['ot_rate'], 2, False)} |")
        print(f"| -- | **ACTUAL, same games (levels)** | {act['poss_mean_cc']:.3f} | "
              f"{act['poss_sd_cc']:.3f} | {act['poss_mean_all']:.3f} | "
              f"{act['poss_sd_all']:.3f} | {act['margin_sd']:.3f} | "
              f"{act['corr_home_away']:.3f} | 0.000 | {act['ppp']:.4f} | "
              f"{100 * act['ot_rate']:.2f} |")
        # floors
        print("\n| floor (same arm, seeds +1000) | G1 cc mean | G1 all mean | "
              "margin SD | corr(h,a) | total bias |")
        print("|---|---:|---:|---:|---:|---:|")
        for arm, fr, aid, name in ORDER:
            r, fl = pick(arms, arm, fr), floor_of(arms, arm, fr)
            if r is None or fl is None:
                continue
            print(f"| {aid} {name} | "
                  f"{f(fl['G1_cc'].get('mean_delta', 0) - r['G1_cc'].get('mean_delta', 0))} | "
                  f"{f(fl['G1_all'].get('mean_delta', 0) - r['G1_all'].get('mean_delta', 0))} | "
                  f"{f(fl['margin_sd'] - r['margin_sd'])} | "
                  f"{f(fl['corr_home_away'] - r['corr_home_away'])} | "
                  f"{f(fl['total_bias'] - r['total_bias'])} |")

        print("\n| id | end-of-half share < 35 s | gap | last-poss duration s | gap | "
              "mean poss duration s |")
        print("|---|---:|---:|---:|---:|---:|")
        for arm, fr, aid, name in ORDER:
            r = pick(arms, arm, fr)
            if r is None:
                continue
            e = r["end_of_half"]
            print(f"| {aid} {name} | {f(e.get('share_last_starts_under_35s'), 4, False)} | "
                  f"{f(r.get('eoh_share_gap'), 4)} | "
                  f"{f(e.get('mean_last_possession_duration_s'), 3, False)} | "
                  f"{f(r.get('eoh_duration_gap'), 3)} | "
                  f"{f(e.get('mean_possession_duration_s'), 3, False)} |")
        t = act.get("end_of_half")
        if t:
            print(f"| -- | **ACTUAL (clock-complete halves)** | "
                  f"{t['actual_share_last_poss_under_35s']:.4f} | -- | "
                  f"{t['actual_mean_last_poss_duration_s']:.3f} | -- | -- |")

        print("\n| id | tempo-prior quintile slope ratio | sim span | actual span | "
              "monotone steps | margin MAE | total MAE |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for arm, fr, aid, name in ORDER:
            r = pick(arms, arm, fr)
            if r is None:
                continue
            print(f"| {aid} {name} | {f(r['slope_ratio'], 3, False)} | "
                  f"{f(r['slope_span_sim'], 2)} | {f(r['slope_span_act'], 2)} | "
                  f"{r['monotone_steps']}/4 | {f(r['margin_mae'], 3, False)} | "
                  f"{f(r['total_mae'], 3, False)} |")

        months = sorted({m["month"] for r in arms.values() for m in r["months"]})
        print("\n| id | " + " | ".join(months) + " |")
        print("|---|" + "---:|" * len(months))
        for arm, fr, aid, name in ORDER:
            r = pick(arms, arm, fr)
            if r is None:
                continue
            by = {m["month"]: m for m in r["months"]}
            cells = [f"{f(by[m]['mean_delta'], 2)}{'' if by[m]['powered'] else '*'}"
                     if m in by else "--" for m in months]
            print(f"| {aid} {name} | " + " | ".join(cells) + " |")
        ng = {m["month"]: m["n_games"] for r in arms.values() for m in r["months"]}
        print("| n games | " + " | ".join(str(ng[m]) for m in months) + " |")
        print("\n`*` = fewer than 100 games, UNDERPOWERED, excluded from the decision.")

    print(f"Season {d5['season']}, {act['n_games']} games "
          f"({act['n_cc']} clock-complete). Actual possessions per team-game: "
          f"{act['poss_mean_all']:.3f} all / {act['poss_mean_cc']:.3f} clock-complete.")
    block(d5, "Screening read, 5 seeds")
    if d25:
        block(d25, "Deciding read, 25 seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
