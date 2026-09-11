"""Render the round-4 possession-outcome results and test-doc tables from the artifacts
`scripts/train_possession_outcome_v4.py` wrote. Reads only; writes two markdown blocks
into the round-4 artifact directory for the worker to append to the docs.

Multi-level evidence, per the standing rule: overall, per week of season, per team
quintile, per possession-outcome class. Underpowered cells are labelled underpowered.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
D = _ROOT / "data/processed/models/possession_outcome/round4"
CLASSES = ("TOV", "FGA_rim", "FGA_jump2", "FGA_3", "FT_trip_shooting", "FT_trip_bonus")

GRID_COLS = ["stage", "role", "population", "fold", "arm", "feature_arm", "scheme", "seed",
             "n_fits", "log_loss", "worst_gated_gap_pp", "wk03_gap_pp", "nonconf_gap_pp",
             "conf4_gap_pp", "quintile_slope_worst", "calibration_pass", "responsiveness_pass",
             "ncss_slope_pass", "fit_seconds"]


def md(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    d = df[cols] if cols else df
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main() -> None:
    grid = pd.read_csv(D / "grid_results.csv")
    verdict = json.loads((D / "verdict.json").read_text())
    floors = json.loads((D / "noise_floor.json").read_text())
    extras = json.loads((D / "extra_metrics.json").read_text())
    detail = json.loads((D / "metrics_detail.json").read_text())
    not_run = json.loads((D / "not_run.json").read_text())
    kmeta = json.loads((D / "shrinkage_k.json").read_text())
    leak = pd.read_csv(D / "leak_test_round4.csv")

    out = []
    out.append("### 9.1 The grid\n")
    for pop in ("first", "cont"):
        for fold in ("F2", "F1"):
            sub = grid[(grid["population"] == pop) & (grid["fold"] == fold)]
            if not len(sub):
                continue
            sel = " (SELECTION)" if fold == "F2" else ""
            sub = sub.sort_values(["arm", "scheme", "feature_arm", "seed"])
            out.append(f"\n**`{pop}` / {fold}**{sel}\n")
            out.append(md(sub, [c for c in GRID_COLS if c in sub.columns]))
    out.append("\n`wk03_gap_pp` is the worst gated decile calibration gap over chances in the "
               "first four calendar weeks of the test season -- the segment round 4 targets. "
               "`quintile_slope_worst` is the own-driver Decision 8 slope ratio furthest from "
               "1.0 among the drivers whose realised quintile span clears 2 pp.\n")

    out.append("\n### 9.2 Noise floor\n")
    for pop, f in floors.items():
        out.append(f"* **`{pop}`**: reference cell `G0 x S1_monthly`, seed 0 log loss "
                   f"{f['seed0_log_loss']}, seed 1 {f['seed1_log_loss']}, spread "
                   f"{f['seed_spread']}; weeks-0-3 gap {f['seed0_wk03_gap_pp']} vs "
                   f"{f['seed1_wk03_gap_pp']} pp; non-conference gap "
                   f"{f['seed0_nonconf_gap_pp']} vs {f['seed1_nonconf_gap_pp']} pp; "
                   f"200-replicate game-block bootstrap SE {f['block_bootstrap_se']}. "
                   f"Applied floor **{f['applied']}**. PARTIAL: {f['partial_reason']}.")

    out.append("\n### 9.3 Decision\n")
    for pop, v in verdict.items():
        if v.get("status"):
            out.append(f"\n**`{pop}`**: {v['status']}.")
            continue
        out.append(f"\n**`{pop}`** (fold F2, noise floor {v['noise_floor']:.6f})\n")
        for lad, key in (("feature_ladder", "feature_arm"), ("scheme_ladder", "scheme")):
            L = v.get(lad) or {}
            if not L.get("rows"):
                continue
            rows = pd.DataFrame(L["rows"])
            out.append(f"\n*{L['ladder']} ladder*\n")
            out.append(md(rows))
            out.append(f"\nWinner: `{L['winner']}` -- {L['reason']}\n")

    out.append("\n### 9.4 Responsiveness by own-rating quintile (Decision 8)\n")
    rr = []
    for key, e in extras.items():
        for r in e.get("responsiveness_own", []):
            rr.append({"cell": key, "driver": r.get("driver"), "class": r.get("class"),
                       "span_pred_pp": r.get("span_pred_pp"), "span_act_pp": r.get("span_act_pp"),
                       "slope_ratio": r.get("slope_ratio"),
                       "steps": f"{r.get('steps_with_actual')}/{r.get('n_steps')}",
                       "exempt_narrow_span": r.get("exempt_narrow_span"), "pass": r.get("pass")})
    out.append(md(pd.DataFrame(rr)))

    out.append("\n### 9.5 Leak test on every column round 4 adds\n")
    out.append(md(leak))
    out.append("\nGate: |as-joined change-form corr| <= 0.15. Round 2's raw-centred columns are "
               "shown alongside so the shrunk numbers are read against columns already accepted.\n")

    out.append("\n### 9.6 The fitted shrinkage weight\n")
    kb = kmeta["k_by_season"]
    krows = []
    for s, per in kb.items():
        for name, v in per.items():
            side, rate = name.split("|")
            krows.append({"season": s, "side": side, "rate": rate, "k (denominator mass)": v["k"],
                          "s2": v.get("s2"), "tau2_raw": v.get("tau2_raw"),
                          "tau2_net": v.get("tau2_net"),
                          "fitted_from_seasons": v.get("seasons_pool"),
                          "note": v.get("fallback", "")})
    out.append(md(pd.DataFrame(krows)))
    out.append("\nEvery k is fitted from COMPLETED PRIOR SEASONS ONLY by method of moments; none "
               "is chosen by looking at a score. `w = D/(D+k)` with `D` the team's as-of "
               "denominator mass, so a team with `D = k` sits at half weight.\n")
    out.append("\n**Reproduction check**: the round-4 rebuild of round 2's raw centred columns "
               "matches the cached round-3 design to "
               f"{max(kmeta['reproduction_max_abs_diff'].values())} absolute, so the arms differ "
               "by the shrinkage and by nothing else.\n")

    out.append("\n### 9.7 Cells the budget did not reach (NOT RUN, not a result)\n")
    if not_run:
        out.append(md(pd.DataFrame(not_run)))
    else:
        out.append("None -- every pre-registered cell ran.\n")
    (D / "results_round4.md").write_text("\n".join(out), encoding="utf-8")

    # ----------------------------------------------------------------- test doc
    t = []
    t.append("### Per week of season\n")
    wk_rows = []
    for key, e in extras.items():
        if "|first|" in key or key.startswith("first|F2|lgbm") or key.startswith("first|F2|"):
            pass
        for w in e.get("per_week", []):
            wk_rows.append({"cell": key, "week": w["week"], "n": w["n"],
                            "underpowered": w["underpowered"],
                            "worst_gated_gap_pp": w["worst_gated_gap_pp"],
                            "mean_shrink_w_off_3pa": w.get("mean_shrink_w_off_3pa"),
                            **{f"resid_{c}_pp": w["signed_residual_pp"].get(c) for c in CLASSES}})
    wk = pd.DataFrame(wk_rows)
    wk.to_csv(D / "per_week_table.csv", index=False)
    t.append(md(wk[wk["cell"].str.startswith("first|F2|lgbm")]) if len(wk) else "no cells")

    t.append("\n### Per possession-outcome class (overall, fold 2)\n")
    crows = []
    for key, s in detail.items():
        if not key.startswith("first|F2|lgbm") and not key.startswith("cont|F2|cascade"):
            continue
        for c in CLASSES:
            cal = (s.get("calibration") or {}).get(c, {})
            crows.append({"cell": key, "class": c, "share_pct": cal.get("share_pct"),
                          "brier": (s.get("brier") or {}).get(c),
                          "max_abs_gap_pp": cal.get("max_abs_gap_pp"),
                          "level_shift_pp": cal.get("level_shift_pp"),
                          "gap_after_level_shift_pp": cal.get("max_abs_gap_pp_after_level_shift")})
    t.append(md(pd.DataFrame(crows)))
    (D / "testdoc_round4.md").write_text("\n".join(t), encoding="utf-8")
    print(f"wrote {D/'results_round4.md'} and {D/'testdoc_round4.md'}")


if __name__ == "__main__":
    sys.exit(main())
