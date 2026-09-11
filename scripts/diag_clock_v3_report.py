"""diag_clock_v3_report.py -- turn round 3's artifacts into the experiments.md
results section.

Every number in the section it prints is read from
`data/processed/models/clock/v3_*` (written by `scripts/train_clock_v3.py`);
none is typed by hand, which is the same discipline rounds 1 and 2 used. The
output goes to stdout and to `data/processed/models/clock/v3_results_section.md`
so the append is a file concatenation, not a transcription.

Usage:
    .venv/Scripts/python.exe scripts/diag_clock_v3_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/processed/models/clock")
ARM_ORDER = ["empirical_km3", "empirical_km3_srfloor", "gamma_aft", "lognormal_aft",
             "hazard3", "xgb_aft", "lgbm_quantile_r2", "empirical_r2", "gamma_r2"]
ARM_ID = {"empirical_km3": "A1", "empirical_km3_srfloor": "A2", "gamma_aft": "A3",
          "lognormal_aft": "A4", "hazard3": "A5", "xgb_aft": "A6",
          "lgbm_quantile_r2": "B1", "empirical_r2": "B2", "gamma_r2": "B3"}
#: Round-2 F2 numbers, read from round 2's own artifact -- never retyped.
R2_GRID = OUT / "v2_grid_results.csv"


def md(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
        elif pd.api.types.is_bool_dtype(d[c]):
            d[c] = d[c].map(lambda v: "PASS" if v else "FAIL")
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def order(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["_o"] = d["arm"].map({a: i for i, a in enumerate(ARM_ORDER)})
    d["id"] = d["arm"].map(ARM_ID)
    return d.sort_values("_o").drop(columns="_o")


def main() -> None:
    g2 = order(pd.read_csv(OUT / "v3_grid_F2.csv"))
    verdict = json.loads((OUT / "v3_verdict.json").read_text(encoding="utf-8"))
    build = json.loads((OUT / "v3_build_diagnostics.json").read_text(encoding="utf-8"))
    lut = json.loads((OUT / "v3_lookup_report.json").read_text(encoding="utf-8"))
    f1_stage = json.loads((OUT / "v3_f1_stage.json").read_text(encoding="utf-8")) \
        if (OUT / "v3_f1_stage.json").exists() else {}
    g1 = order(pd.read_csv(OUT / "v3_grid_F1.csv")) if (OUT / "v3_grid_F1.csv").exists() else None

    parts: list[str] = []
    A = parts.append

    A("\n---\n")
    A("## 9. Run R4 -- the round-3 grid (2026-09-10)\n")
    A("`scripts/train_clock_v3.py`, seed 20260910, module "
      "`src/cbb_sim/models/clock_v3.py`. Artifacts are `v3_*` in "
      "`data/processed/models/clock/`; rounds 1 and 2 files are untouched. Every "
      "number below is written by that script or by "
      "`scripts/diag_clock_v3_report.py` reading its output; none is typed by "
      "hand.\n")

    A("### 9.1 The censoring flag, as it lands on the design\n")
    A(md(pd.DataFrame([{
        "design rows": build["rows"],
        "censored, rounds 1-2 flag (%)": build["censored_old_pct"],
        "censored, horn flag (%)": build["censored_horn_pct"],
        "horn and not old": build["horn_and_not_old"],
        "old and not horn": build["old_and_not_horn"],
        "censored rows not consuming their clock": build["censored_rows_not_consuming_clock"],
    }]), "{:.4f}"))
    A("\n`old and not horn` are the handful of `end_period` rows whose feed "
      "stops a second or two before 0:00; they are reported, not reclassified.\n")

    if f1_stage:
        A("### 9.2 Choices made on F1 only\n")
        A(md(pd.DataFrame(f1_stage["xgb_dist_search"]), "{:.5f}"))
        A(f"\n**Chosen AFT error distribution: `{f1_stage['xgb_dist_chosen']}`** "
          "(highest F1 censored log-likelihood).\n")
        fl = f1_stage["end_of_half_floor"]
        A("End-of-half noise floor, F1, on clock-complete halves: sim 5-seed "
          f"re-chain SD share {fl['sim_seed_sd_share']:.5f} / duration "
          f"{fl['sim_seed_sd_duration']:.4f} s; actual game-block bootstrap SE "
          f"share {fl['actual_block_bootstrap_se_share']:.5f} / duration "
          f"{fl['actual_block_bootstrap_se_duration']:.4f} s; k = {fl['k']}. "
          f"**Floor: share +/- {fl['share']:.5f}, duration +/- {fl['duration']:.4f} s** "
          f"on {fl['n_cc_halves']:,} clock-complete halves.\n")

    A("### 9.3 F2 (selection fold) -- primary metric and the three gates\n")
    cols = ["id", "arm", "flag", "crps_trunc", "censored_loglik", "crps_r2def",
            "pred_mean_duration", "pit_leak_failures", "pit_worst_D",
            "cc_mean_delta", "cc_sd_delta", "cc_months_pass", "cc_n_powered_months",
            "g1_pass", "cc_eoh_share_gap", "cc_eoh_duration_gap", "eoh_pass",
            "pit_pass", "all_gates_pass"]
    A(md(g2[[c for c in cols if c in g2.columns]].sort_values("crps_trunc"), "{:.4f}"))
    A("\n`crps_trunc` is the primary metric (pre-registration 8.4): CRPS of the "
      "predictive law RENORMALISED onto {0..R-1}, on uncensored test rows only. "
      "`crps_r2def` is round 2's definition (all rows, censored rows scored as "
      "complete) and is a labelled bridge, not a decision metric. `cc_*` are read "
      "on CLOCK-COMPLETE games/halves, which is the round-3 gate universe.\n")

    A("### 9.4 The mechanism: predicted INTENDED duration by clock band\n")
    bands = []
    for a in ARM_ORDER:
        p = OUT / f"v3_band_means_F2_{a}.csv"
        if not p.exists():
            continue
        b = pd.read_csv(p)
        b["arm"] = a
        bands.append(b)
    if bands:
        bb = pd.concat(bands)
        piv = bb.pivot_table(index="band", columns="arm", values="pred_mean_intended",
                             sort=False)
        base = bb[bb["arm"] == ARM_ORDER[0]].set_index("band")
        tab = pd.DataFrame({"band": piv.index,
                            "n": base.loc[piv.index, "n"].to_numpy(),
                            "horn_rate": base.loc[piv.index, "horn_rate"].to_numpy(),
                            "actual_mean_observed": base.loc[piv.index, "actual_mean_observed"].to_numpy()})
        for a in ARM_ORDER:
            if a in piv.columns:
                tab[ARM_ID[a]] = piv[a].to_numpy()
        A(md(tab, "{:.3f}"))
        A("\nThis is where the fix has to show up and the one table round 2 could "
          "not produce. `actual_mean_observed` is the TRUNCATED mean the feed "
          "records; the arms trained on the corrected flag predict the INTENDED "
          "duration, which is longer wherever `horn_rate` is non-trivial, and the "
          "chain truncates it back at the horn. An arm whose column tracks "
          "`actual_mean_observed` inside the last 10 seconds has not changed.\n")

    A("### 9.5 Segment breakdowns (pre-registration 8.6)\n")
    segs = []
    for a in ARM_ORDER:
        p = OUT / f"v3_segments_F2_{a}.csv"
        if p.exists():
            s = pd.read_csv(p)
            s["id"] = ARM_ID[a]
            segs.append(s)
    if segs:
        ss = pd.concat(segs)
        for name in ("half", "score_bucket"):
            sub = ss[ss["segment"] == name]
            piv = sub.pivot_table(index="level", columns="id", values="crps_trunc")
            piv = piv[[c for c in [ARM_ID[a] for a in ARM_ORDER] if c in piv.columns]]
            n = sub.groupby("level")["n"].first()
            t = piv.reset_index()
            t.insert(1, "n", n.reindex(piv.index).to_numpy())
            A(f"\n**CRPS_trunc by {name}**\n")
            A(md(t, "{:.4f}"))
        A("\nThe shot-clock era is 30 s in every NCAA men's season 2022-2025, so "
          "that pre-registered segment is degenerate over this fold window and "
          "season stands in its place (`v3_segments_F2_*.csv`, `segment = "
          "season`), reported rather than silently dropped.\n")

    A("\n### 9.6 Responsiveness (CLAUDE.md standing rule)\n")
    rows = []
    for a in ARM_ORDER:
        p = OUT / f"v3_responsiveness_F2_{a}.csv"
        if p.exists():
            r = pd.read_csv(p)
            rows.append({"id": ARM_ID[a], "arm": a,
                         "span_sim": float(r["span_sim"].iloc[0]),
                         "span_actual": float(r["span_actual"].iloc[0]),
                         "slope_ratio": float(r["slope_ratio"].iloc[0]),
                         "steps_agreeing": int(r["steps_agreeing"].iloc[0]),
                         "q1_delta": float(r["delta"].iloc[0]),
                         "q5_delta": float(r["delta"].iloc[-1])})
    if rows:
        A(md(pd.DataFrame(rows), "{:.4f}"))
        A("\nPer-quintile tables: `v3_responsiveness_F2_{arm}.csv`.\n")

    A("\n### 9.7 By-month G1 on clock-complete games, for the arms that pass overall\n")
    mrows = []
    for a in ARM_ORDER:
        p = OUT / f"v3_cc_months_F2_{a}.csv"
        if p.exists():
            m = pd.read_csv(p)
            m["id"] = ARM_ID[a]
            mrows.append(m)
    if mrows:
        mm = pd.concat(mrows)
        piv = mm[mm["powered"]].pivot_table(index="month", columns="id", values="mean_delta")
        piv = piv[[c for c in [ARM_ID[a] for a in ARM_ORDER] if c in piv.columns]]
        ng = mm[mm["powered"]].groupby("month")["n_games"].first()
        t = piv.reset_index()
        t.insert(1, "n_cc_games", ng.reindex(piv.index).to_numpy())
        A(md(t, "{:.3f}"))
        A("\nPowered months only (>= 100 clock-complete games). Tolerance +/- 1.0.\n")

    if g1 is not None:
        A("\n### 9.8 F1 (robustness only)\n")
        c1 = ["id", "arm", "flag", "crps_trunc", "censored_loglik", "crps_r2def",
              "pit_leak_failures", "pit_worst_D"]
        A(md(g1[[c for c in c1 if c in g1.columns]].sort_values("crps_trunc"), "{:.4f}"))

    A("\n### 9.9 Noise floor\n")
    fl = pd.DataFrame([{"arm": k, "id": ARM_ID.get(k, k), "kind": v["kind"], "value": v["value"]}
                       for k, v in verdict["floors_by_arm"].items()])
    A(md(order(fl)[["id", "arm", "kind", "value"]], "{:.6f}"))
    A(f"\nFloor used by the decision rule: **{verdict['floor']:.6f}** (the maximum).\n")

    A("\n### 9.10 Lookup-table export (deliverable, pre-registration 8.9)\n")
    lrow = pd.DataFrame([{
        "source arm": lut["source_arm"], "adopted": lut["adopted"],
        "cells": lut["n_cells"], "empty cells": lut["n_empty_cells"],
        "live CRPS_trunc": lut["live_crps_trunc"], "binned CRPS_trunc": lut["lookup_crps_trunc"],
        "dCRPS": lut["d_crps_trunc"], "TV mean": lut["tv_mean"], "TV max": lut["tv_max"],
        "live G1-CC dmean": lut["live_cc_mean_delta"], "binned G1-CC dmean": lut["lookup_cc_mean_delta"],
        "dG1": lut["d_cc_mean"], "live rows/s": lut["live_rows_per_s"],
        "binned rows/s": lut["lookup_rows_per_s"],
    }])
    A(md(lrow, "{:.6f}"))
    A(f"\nGrid: {' x '.join(f'{d} {s}' for d, s in zip(lut['cell_dims'], lut['cell_sizes'], strict=False))}"
      f" = {lut['n_cells']} cells x 91 durations. Table at `v3_lookup_table.npz`.\n")

    A("\n### 9.11 Verdict\n")
    A("```json\n" + json.dumps({k: v for k, v in verdict.items() if k != "floors_by_arm"},
                               indent=2, default=str) + "\n```\n")

    text = "\n".join(parts)
    (OUT / "v3_results_section.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
