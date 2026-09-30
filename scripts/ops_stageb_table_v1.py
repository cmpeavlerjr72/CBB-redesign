"""ops_stageb_table_v1.py -- operator 2026-09-30. Stage B offline table from the trainers' own reports. Decides NOTHING.

    python scripts/ops_stageb_table_v1.py --root <dir holding possession_outcome/ fg_make/ rebound/ round_stageb trees> --out <md>

Primary (quoted from each sub-model's experiments.md via team_rate_estimator/experiments.md section 7.1):
  possession_outcome: multiclass log loss on F2, `first` (floor 0.000804) and `cont` (floor 0.001982)
  fg_make: attempt-level log loss per class (floors 4.386e-5 / 1.1216e-4 / 1.813e-5 for rim / jump2 / three)
  rebound: three-class log loss on F2 (floor 6.7e-5)
Per arm: value, difference vs R (same code path, seed 0, served features), in floors (registered floor, and the
larger of it and tonight's |R2 - R| seed spread). Negative difference = lower log loss than R = better.
Responsiveness lines are the trainers' own (PO: responsiveness_pass; rebound: team_quintile slope_ratio, slope of realised on
predicted OREB in the off_oreb_c / opp_def_dreb_c terms, L1 level). Grading only, no selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
root = Path(a.root)
STEM, STEMO = "team_rate_features_E3_v4", "team_rate_features_E3opp_v4"
J = lambda p: json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None

po = root / "possession_outcome/round_stageb"
PO = {"R": J(po / "R_seed0/par_v1_report.json"), "R2": J(po / "R2_seed1/par_v1_report.json"),
      "T": J(po / f"T/{STEM}/par_v1_report.json"), "Topp": J(po / f"Topp/{STEMO}/par_v1_report.json"),
      "Tfs": J(po / "Tfs/par_v1_report.json")}
fg = root / "fg_make/round_stageb"
FG = {"R": J(fg / "R_seed0/run_report.json"), "R2": J(fg / "R2_seed1/run_report.json"), "T": J(fg / f"T/{STEM}/run_report.json")}
rb = root / "rebound/round_stageb"
RB = {"R": J(rb / "R_seed0/cells/s2_F2_A0B0C0_seed0.json"), "R2": J(rb / "R2_seed1/cells/s2_F2_A0B0C0_seed1.json"),
      "T": J(rb / f"T/{STEM}/cells/s2_F2_A0B0C0_seed0.json")}

L = ["# Stage B offline table, fold 2 (box, 2026-09-30). Grading only; the PM decides.\n",
     "Difference = arm minus R (negative = lower log loss = better). `floors` = difference / registered floor; `spread` = |R2 - R| "
     "(tonight's second-seed spread, same code path). Fold 1 confirmation was not run (fold-1 retrains were not in tonight's list). "
     "Arms TO (anchored) not run (engine could not serve the offset when the job list was written).\n"]


def row(name, v, ref, r2, floor):
    if v is None:
        return f"| {name} | not run | | | |"
    d = v - ref
    sp = abs(r2 - ref) if r2 is not None else float("nan")
    return f"| {name} | {v:.6f} | {d:+.6f} | {d / floor:+.2f} | spread {sp:.6f} ({sp / floor:.2f} floors) |"


for pop, floor in (("first", 0.000804), ("cont", 0.001982)):
    L += [f"\n## possession_outcome `{pop}`: multiclass log loss (floor {floor})\n", "| arm | log loss | diff vs R | floors | R2 spread |", "|---|---:|---:|---:|---|"]
    ref = PO["R"]["scores"][pop]["log_loss"]; r2 = PO["R2"]["scores"][pop]["log_loss"]
    for k in ("R", "R2", "T", "Topp", "Tfs"):
        v = PO[k]["scores"][pop]["log_loss"] if PO[k] else None
        L.append(row(k, v, ref, r2, floor) if k != "R" else f"| R | {ref:.6f} | (reference) | | |")
    L += ["", "| arm | calibration_pass | worst gated gap pp | responsiveness_pass |", "|---|---|---:|---|"]
    for k in ("R", "R2", "T", "Topp", "Tfs"):
        s = PO[k]["scores"][pop] if PO[k] else None
        L.append(f"| {k} | {s['calibration_pass']} | {s['worst_gated_gap_pp']} | {s['responsiveness_pass']} |" if s else f"| {k} | not run | | |")

for cls, floor in (("FGA_rim", 4.386e-5), ("FGA_jump2", 1.1216e-4), ("FGA_3", 1.813e-5)):
    L += [f"\n## fg_make B1 `{cls}`: attempt-level log loss (floor {floor})\n", "| arm | log loss | diff vs R | floors | R2 spread |", "|---|---:|---:|---:|---|"]
    g = lambda k: FG[k]["arms"]["B1"]["by_class"][cls] if FG[k] else None
    ref = g("R")["log_loss"]; r2 = g("R2")["log_loss"]
    for k in ("R", "R2", "T"):
        s = g(k)
        L.append(f"| R | {ref:.6f} | (reference) | | |" if k == "R" else row(k, s["log_loss"] if s else None, ref, r2, floor))
    L.append("")
    L += ["| arm | calib_pass | calib worst gap |", "|---|---|---:|"] + [f"| {k} | {g(k)['calib_pass']} | {g(k).get('calib_worst_gap_pp', g(k).get('calib_worst_gap'))} |" for k in ("R", "R2", "T") if g(k)]

L += ["\n## rebound S1_weekly (A0B0C0): three-class log loss (floor 6.7e-05)\n", "| arm | log loss | diff vs R | floors | R2 spread |", "|---|---:|---:|---:|---|"]
ref = RB["R"]["log_loss"]; r2 = RB["R2"]["log_loss"]
for k in ("R", "R2", "T"):
    L.append(f"| R | {ref:.6f} | (reference) | | |" if k == "R" else row(k, RB[k]["log_loss"] if RB[k] else None, ref, r2, 6.7e-5))
L += ["", "| arm | calib worst gap pp | calib worst class | resp_pass | team-quintile slope ratio | slope off_oreb_c | slope opp_def_dreb_c | L1 level pp | per-game MAE pp |", "|---|---:|---|---|---:|---:|---:|---:|---:|"]
for k in ("R", "R2", "T"):
    s = RB[k]
    L.append(f"| {k} | {s['calib_worst_gap_pp']} | {s['calib_worst_class']} | {s['resp_pass']} | {s['team_quintile']['slope_ratio']} | {s['slope_off_oreb_c']} | {s['slope_opp_def_dreb_c']} | {s['L1_level_pp']} | {s['per_game']['mae_pp']} |")
Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
