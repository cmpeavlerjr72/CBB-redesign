"""grade_lanee_lg2_v3_v1.py -- lane E 2026-09-30: grade the late-game round-2 re-read (v3 inputs, verified sample, verified truth)
with `grade_late_game_r2_v1`'s own lines / vetoes (imported, unedited) and a Decision-12 floor: the MAX over four seed-offset draws
(offsets 1000 [clk6_Rf1], 2000 [clk6_Rf2], 3000, 4000 [e3_Rf3, e3_Rf4: untapped shared R draws, bit-identical sim, no tap columns]) of |L(R_k) - L(R_0)|.
Tap-derived lines (half share) take the single tapped floor draw only. The paired game bootstrap SE of the primary is the grader's own.

    .venv/Scripts/python.exe scripts/grade_lanee_lg2_v3_v1.py --out results/late_game/round2/grade_e3_v3.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade_late_game_r2_v1 as G  # noqa: E402

REF = "e3_lg2_R_s25"
TAPPED_FLOOR = "e3_lg2_Rfloor_s25"
UNTAPPED_FLOORS = ["clk6_Rf1_s25", "clk6_Rf2_s25", "e3_Rf3_s25", "e3_Rf4_s25"]
ARMS = ["e3_lg2_W_C2_s25", "e3_lg2_W_D_s25", "e3_lg2_E_BL3_s25", "e3_lg2_E_L0S0_s25",
        "e3_lg2_W_C2_E_BL3_s25", "e3_lg2_W_D_E_BL3_s25"]


def load_untapped(tag):
    d = G.RES / tag
    g = pd.read_parquet(d / "games.parquet")
    for c in ("h1_home_pts", "h1_away_pts", "home_margin_120", "fta_at_120", "fta_reg_end_if_ot", "window_poss"):
        g[c] = np.nan
    return {"tag": tag, "g": g, "clk": np.zeros((3, 4, 3)), "ev": np.zeros((3, 4, 4)),
            "meta": json.loads((d / "run_meta.json").read_text(encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    R = G.load(REF); LR = G.lines(R)
    keys = ["P0", "P1", "P0_over_P1"] + list(G.VETO_LINES)
    tapped = (G.RES / TAPPED_FLOOR / "games.parquet").exists()
    Ls = ([G.lines(G.load(TAPPED_FLOOR))] if tapped else []) + [G.lines(load_untapped(t)) for t in UNTAPPED_FLOORS]
    floors, floor_single = {}, {k: abs(Ls[0][k] - LR[k]) for k in keys}
    for k in keys:
        v = [abs(L[k] - LR[k]) for L in Ls if np.isfinite(L[k])]
        floors[k] = max(v) if v else float("nan")
    rows, verdicts = [LR], []
    for tag in ARMS:
        if not (G.RES / tag / "games.parquet").exists():
            verdicts.append({"tag": tag, "status": "NOT RUN"}); continue
        A = G.load(tag); LA = G.lines(A); rows.append(LA)
        v = {"tag": tag, "arm": LA["arm"]}
        for k in ("P0", "P1", "P0_over_P1"):
            v[f"d_{k}"] = LA[k] - LR[k]
            v[f"floors_{k}"] = (LA[k] - LR[k]) / floors[k] if floors[k] else None
        v["ot_in_band"] = G.OT_BAND[0] <= LA["ot_rate"] <= G.OT_BAND[1]
        v.update(G.paired_boot(R["g"], A["g"]))
        v["first_half"] = G.first_half_identical(R["g"], A["g"])
        vet = {}
        for k, (tgt, _) in G.VETO_LINES.items():
            t_r = LR[tgt] if isinstance(tgt, str) else tgt
            t_a = LA[tgt] if isinstance(tgt, str) else tgt
            er, ea = abs(LR[k] - t_r), abs(LA[k] - t_a)
            vet[k] = {"ref": LR[k], "arm": LA[k], "abs_err_ref": er, "abs_err_arm": ea, "floor": floors[k],
                      "floor_single_draw": floor_single[k],
                      "worse_in_floors": (ea - er) / floors[k] if floors[k] else None,
                      "pass": bool(ea - er <= floors[k])}
        vet["first_half_bit_identical"] = {"pass": v["first_half"]["identical"]}
        v["vetoes"] = vet; v["all_vetoes_pass"] = all(x["pass"] for x in vet.values())
        verdicts.append(v)
    out = {"ref": REF, "floor_draws": ([TAPPED_FLOOR] if tapped else []) + UNTAPPED_FLOORS, "tapped_floor_present": tapped, "floors": floors,
           "floors_single_tapped_draw": floor_single, "lines": rows, "verdicts": verdicts,
           "power": {"floor_ratio": floors["P0_over_P1"], "ref_reg_ties": LR["n_reg_ties"],
                     "adequate": bool(floors["P0_over_P1"] <= 0.10 and LR["n_reg_ties"] >= 300)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("floors(max of 4 draws):", {k: round(x, 5) for k, x in floors.items()})
    print("R:", {k: round(LR[k], 4) for k in ("P0", "P1", "P0_over_P1", "G1_poss_mean", "G1_poss_sd", "G5_total_sd_ratio", "G9_total_bias", "half1_share", "n_reg_ties")})
    for v in verdicts:
        if "status" in v: print(v["tag"], v["status"]); continue
        bad = [k for k, x in v["vetoes"].items() if not x["pass"]]
        print(v["tag"], "dRatio %+.3f (%+.2f fl, SE %.3f)" % (v["d_P0_over_P1"], v["floors_P0_over_P1"], v["se_dratio"]),
              "OT %.4f" % [r for r in rows if r["tag"] == v["tag"]][0]["ot_rate"], "vetoes", "PASS" if not bad else "FAIL " + ",".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
