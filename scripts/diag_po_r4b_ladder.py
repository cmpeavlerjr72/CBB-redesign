"""Round 4b: render the completed `first` tree ladder against the MEASURED noise floor.

Reads only, from the versioned sibling `round4b/` that `scripts/run_po_r4b_merge.py`
wrote. One row per feature arm: fold-2 log loss, the pre-registered weeks-0-3 and
non-conference decision cells, the per-week buckets weeks 0-3 / 4-7 / 8+ from round 4's
`per_week_table`, the worst Decision-8 quintile slope, the gates, and whether the arm
beats the reference beyond the floor now that the floor is measured from a second seed
rather than the block bootstrap alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
D = _ROOT / "data/processed/models/possession_outcome/round4b"
RANK = {"G0": 0, "G1": 1, "G4": 1, "G2": 2, "G3": 3}


def main() -> None:
    grid = pd.read_csv(D / "grid_results.csv")
    wk = pd.read_csv(D / "per_week_table.csv")
    floors = json.loads((D / "noise_floor.json").read_text())
    verdict = json.loads((D / "verdict.json").read_text())

    f = floors["first"]
    print("=== MEASURED FLOOR (first) ===")
    print(json.dumps(f, indent=1))

    sub = grid[(grid.population == "first") & (grid.fold == "F2") & (grid.arm == "lgbm")
               & (grid.scheme == "S1_monthly")].copy()
    ref = sub[(sub.feature_arm == "G0") & (sub.seed == 0)].iloc[0]
    nf = f["applied"]

    def wkb(cell, bucket):
        r = wk[(wk.cell == cell) & (wk.week == bucket)]
        return None if not len(r) else float(r.iloc[0]["worst_gated_gap_pp"])

    def wk03_buckets(cell):
        r = wk[(wk.cell == cell) & (wk.week.isin(["wk0", "wk1", "wk2", "wk3"]))]
        return None if not len(r) else round(float(r["worst_gated_gap_pp"].max()), 3)

    rows = []
    for _, r in sub.sort_values(["feature_arm", "seed"]).iterrows():
        cell = f"first|F2|lgbm|{r.feature_arm}|S1_monthly|s{int(r.seed)}"
        ll_gain = float(ref.log_loss) - float(r.log_loss)
        wk_gain = float(ref.wk03_gap_pp) - float(r.wk03_gap_pp)
        nc_gain = float(ref.nonconf_gap_pp) - float(r.nonconf_gap_pp)
        gates = bool(r.calibration_pass) and bool(r.responsiveness_pass)
        beats = gates and (ll_gain > nf or wk_gain > 0.25 or nc_gain > 0.25)
        shuffles = wk_gain < -0.25 or nc_gain < -0.25
        rows.append({
            "arm": r.feature_arm, "seed": int(r.seed), "rank": RANK.get(r.feature_arm),
            "log_loss": round(float(r.log_loss), 6), "ll_gain": round(ll_gain, 6),
            "overall_gap": float(r.worst_gated_gap_pp),
            "wk0_3_cell": float(r.wk03_gap_pp), "wk0_3_gain": round(wk_gain, 3),
            "wk0_3_worst_bucket": wk03_buckets(cell),
            "wk4_7": wkb(cell, "wk4_7"), "wk8plus": wkb(cell, "wk8plus"),
            "nonconf": float(r.nonconf_gap_pp), "nonconf_gain": round(nc_gain, 3),
            "slope_worst": float(r.quintile_slope_worst),
            "cal": bool(r.calibration_pass), "resp": bool(r.responsiveness_pass),
            "beats_ref_beyond_MEASURED_floor": bool(beats and not shuffles),
            "moves_error": bool(shuffles),
            "fit_s": round(float(r.fit_seconds)),
        })
    out = pd.DataFrame(rows).sort_values(["rank", "arm"])
    print("\n=== `first` tree ladder, fold 2, S1_monthly, vs MEASURED floor "
          f"{nf} ===")
    print(out.to_string(index=False))

    print("\n=== per-week, every tree arm ===")
    print(wk[wk.cell.str.startswith("first|F2|lgbm")].to_string(index=False))

    print("\n=== verdict.json (first) ===")
    v = verdict["first"]
    print("noise_floor:", v["noise_floor"])
    for lad in ("feature_ladder", "scheme_ladder"):
        L = v.get(lad) or {}
        if L.get("rows"):
            print(f"\n{L['ladder']}: winner {L['winner']}\n  {L['reason']}")
            print(pd.DataFrame(L["rows"]).to_string(index=False))
    print("\n=== cont verdict winner ===")
    vc = verdict.get("cont", {})
    for lad in ("feature_ladder",):
        L = vc.get(lad) or {}
        if L.get("rows"):
            print(f"{L['ladder']}: winner {L['winner']} -- {L['reason']}")


if __name__ == "__main__":
    main()
