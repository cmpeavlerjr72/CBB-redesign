"""Round 3: render the result JSONs as markdown tables and apply the R3.7 pass rule in code. Usage: python scripts/exp_shot_quality_rating_tables_v1.py [--located-only]"""
import sys, json, pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "data/processed/models/shot_location/round3"
tag = "_locOnly" if "--located-only" in sys.argv else ""
DIRS = [("2024", "2025"), ("2025", "2024")]
R = {d: json.load(open(OUT / f"rating_{d[0]}_to_{d[1]}{tag}.json")) for d in DIRS}
lab = lambda d: f"fit {d[0]} / test {d[1]}"
f1 = lambda x: f"{x:+.2f}"

print("### Chosen dials (FIT) and TEST MSE\n")
print("| direction | arm | alpha | weights | K | FIT MSE | TEST MSE |\n|---|---|---|---|---|---|---|")
for d, r in R.items():
    for a in ("RAW", "RAW_SHRUNK", "QUALITY", "CLASS"):
        c = r["chosen"][a]
        print(f"| {lab(d)} | {a} | {c['alpha']} | {'on' if c['weights'] else 'off'} | {c['K'] if a != 'RAW' else 'n/a'} | {c['fit_mse']:.6f} | {r['test_mse'][a]:.6f} |")
print("\n### Primary: TEST MSE gain vs RAW_SHRUNK (floors; positive = arm better)\n")
print("| arm | " + " | ".join(lab(d) for d in DIRS) + " | seed-1 shift (floors) | pass rule |\n|---|---|---|---|---|")
for a in ("RAW", "QUALITY", "CLASS"):
    cells = []
    for d in DIRS:
        g = R[d]["gain_vs_RAW_SHRUNK"][a]
        cells.append(f"{g['gain']:+.2e} (floor {g['floor']:.2e}) = {f1(g['floors'])}")
    verdict = "n/a (reference arm)"
    shift = "n/a"
    if a != "RAW":
        both = all(R[d]["gain_vs_RAW_SHRUNK"][a]["floors"] > 2 for d in DIRS)
        unstable = any(R[d]["seed_shift_floors"][a] > 1 for d in DIRS)
        resp = all(R[d]["responsiveness"][a]["clause"] for d in DIRS)
        verdict = "CANDIDATE" if (both and not unstable and resp) else "NOT CANDIDATE"
        verdict += f" (>2 floors both dirs: {both}; seed-stable: {not unstable}; responsiveness: {resp})"
        shift = " / ".join(f"{R[d]['seed_shift_floors'][a]:.2f}" for d in DIRS)
    print(f"| {a} | " + " | ".join(cells) + f" | {shift} | {verdict} |")
print("\n### Key secondary: QUALITY vs CLASS (positive = QUALITY better)\n")
print("| direction | gain | floor | floors |\n|---|---|---|---|")
for d in DIRS:
    g = R[d]["QUALITY_vs_CLASS"]
    print(f"| {lab(d)} | {g['gain']:+.2e} | {g['floor']:.2e} | {f1(g['floors'])} |")
print("\n### Seed-1 retrain of the grade (dials frozen): gain vs RAW_SHRUNK in floors\n")
print("| direction | arm | seed 0 | seed 1 |\n|---|---|---|---|")
for d in DIRS:
    for a in ("QUALITY", "CLASS"):
        print(f"| {lab(d)} | {a} | {f1(R[d]['gain_vs_RAW_SHRUNK'][a]['floors'])} | {f1(R[d]['seed1_gain_vs_RAW_SHRUNK'][a]['floors'])} |")
print("\n### Gain vs RAW (unshrunk), floors\n")
print("| arm | " + " | ".join(lab(d) for d in DIRS) + " |\n|---|---|---|")
for a in ("RAW_SHRUNK", "QUALITY", "CLASS"):
    print(f"| {a} | " + " | ".join(f1(R[d]["gain_vs_RAW"][a]["floors"]) for d in DIRS) + " |")
print("\n### Segments: gain vs RAW_SHRUNK in floors (n team-games)\n")
print("| segment | " + " | ".join(f"{lab(d)} {a}" for d in DIRS for a in ("QUALITY", "CLASS", "Q-vs-C")) + " |\n|---|" + "---|" * 6)
for s in R[DIRS[0]]["segments"]:
    cells = []
    for d in DIRS:
        sg = R[d]["segments"][s]
        u = " UNDERPOWERED" if sg["underpowered"] else ""
        cells += [f"{f1(sg['QUALITY']['floors'])} (n={sg['n']}){u}", f1(sg["CLASS"]["floors"]), f1(sg["QUALITY_vs_CLASS"]["floors"])]
    print(f"| {s} | " + " | ".join(cells) + " |")
for side in ("offence_only", "defence_only"):
    cells = []
    for d in DIRS:
        so = R[d]["side_only"][side]
        cells += [f1(so["QUALITY"]["floors"]), f1(so["CLASS"]["floors"]), "-"]
    print(f"| {side} | " + " | ".join(cells) + " |")
print("\n### Responsiveness (TEST predicted-value quintiles; weighted slope of actual on predicted)\n")
print("| direction | arm | slope [95%] | monotone | clause | q1..q5 actual pts/FGA | q1..q5 predicted |\n|---|---|---|---|---|---|---|")
for d in DIRS:
    for a in ("RAW", "RAW_SHRUNK", "QUALITY", "CLASS"):
        rr = R[d]["responsiveness"][a]
        print(f"| {lab(d)} | {a} | {rr['slope']:.3f} [{rr['slope_lo']:.3f}, {rr['slope_hi']:.3f}] | {rr['monotone']} | {rr['clause']} | "
              + " ".join(f"{q['actual']:.3f}" for q in rr["quintiles"]) + " | " + " ".join(f"{q['pred']:.3f}" for q in rr["quintiles"]) + " |")
print("\n### RAW alpha curve on TEST (RAW chosen weights)\n")
for d in DIRS:
    print(f"- {lab(d)}: " + ", ".join(f"alpha {a}: {v:.6f}" for a, v in R[d]["raw_alpha_curve_test"].items()))
print("\n### Stability (split-half, odd vs even games), per season\n")
print("| season | side | measure | n teams | half r | full-length rel | FGA to rel 0.5 |\n|---|---|---|---|---|---|---|")
for d in DIRS:
    for key, season in (("stability_fit_season", d[0]), ("stability_test_season", d[1])):
        if d == DIRS[1]:
            continue  # both seasons already covered by direction 1
        for row in R[d][key]:
            print(f"| {season} | {row['side']} | {row['measure']} | {row['n_teams']} | {row['half_r']:.3f} | {row['full_rel']:.3f} | {round(row['fga_to_rel_0.5']) if row['fga_to_rel_0.5'] else 'n/a'} |")
print("\nn team-games scored (TEST): " + ", ".join(f"{lab(d)}: {R[d]['n_test_rows']}" for d in DIRS))
