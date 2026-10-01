"""grade_lanee_fj_v3_v1.py -- lane E 2026-09-30: grade the foul-joint (round 7/8) re-read on v3 inputs.
Runs `grade_foul_joint_closed_loop_v2` UNEDITED against a view directory (junctions) in which `po4b_R_s25` = this re-read's reference
`e3_fj_R_s25` and `po4b_R_s25_floor` = a shared untapped R draw at another seed offset (bit-identical sim, same inputs/sample).
Decision 12 floor = MAX over four seed-offset draws (1000..4000): floors for draws 2-4 are computed with the grader's own functions; the
V3 'floors toward' values of draw 1 are rescaled (value is X/floor, X is draw-independent) and V3 is re-evaluated. V1/V2/V4 use
bootstrap SEs, not seed floors. The old grade json files are backed up and restored (nothing old is overwritten).

    .venv/Scripts/python.exe scripts/grade_lanee_fj_v3_v1.py e3_fj_R_s25 e3_fj_R8a_s25 ... --out results/foul_joint/e3_grade_v3.json
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import grade_foul_joint_closed_loop_v1 as G1  # noqa: E402

RES = Path("results/engine_v0")
VIEW = Path("results/e3_view_fj")
DRAWS = ["clk6_Rf1_s25", "clk6_Rf2_s25", "e3_Rf3_s25", "e3_Rf4_s25"]
REF0 = "e3_fj_R_s25"


def junction(link: Path, target: Path):
    if link.exists():
        subprocess.run(["cmd", "/c", "rmdir", str(link)], check=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target.resolve())], check=True, capture_output=True)


def main():
    args = sys.argv[1:]
    i = args.index("--out"); out = Path(args[i + 1]); tags = args[:i] + args[i + 2:]
    VIEW.mkdir(parents=True, exist_ok=True)
    for t in tags:
        junction(VIEW / t, RES / t)
    junction(VIEW / "po4b_R_s25", RES / REF0)
    junction(VIEW / "po4b_R_s25_floor", RES / DRAWS[0])
    G1.R = VIEW
    import grade_foul_joint_closed_loop_v2 as G2  # noqa: E402  (R = G1.R at import)
    G2.R = VIEW
    fdir = Path("results/foul_joint")
    keep = {p: p.read_bytes() for p in (fdir / "closed_loop_grade_v1.json", fdir / "closed_loop_grade_v2.json") if p.exists()}
    try:
        sys.argv = ["x"] + tags
        G2.main()
        v1 = json.loads((fdir / "closed_loop_grade_v1.json").read_text()); v2 = json.loads((fdir / "closed_loop_grade_v2.json").read_text())
    finally:
        for p, b in keep.items():
            p.write_bytes(b)
    # floors for the other draws
    box = pd.read_parquet("data/raw/hoopr/team_box/team_box_2025.parquet")
    act_tot = box.groupby("game_id")["team_score"].sum()
    g0 = pd.read_parquet(RES / REF0 / "games.parquet")
    L0 = G1.lines(g0, act_tot)
    floors = {k: [abs(G1.lines(pd.read_parquet(RES / d / "games.parquet"), act_tot)[k] - L0[k]) for d in DRAWS] for k in L0}
    sc_f = [abs(G2.score_corr(pd.read_parquet(RES / d / "games.parquet")) - G2.score_corr(g0)) for d in DRAWS]
    ft_f = [abs(G2.ft_within_corr(pd.read_parquet(RES / d / "games.parquet")) - G2.ft_within_corr(g0)) for d in DRAWS]
    fmax = {k: max(v) for k, v in floors.items()}
    f1 = {k: v[0] for k, v in floors.items()}
    res = {"floor_draws": DRAWS, "floors_max": fmax, "floors_draw1": f1, "score_corr_floors": sc_f, "ft_corr_floors": ft_f, "arms": {}}
    sc0 = v2["arms"][REF0]["score_corr"]; act = v2["actual_score_corr"]
    for t in tags:
        a = v2["arms"][t]; fv = v1["arms"][t]["floors_vs_ref"]
        def resc(key):
            x = fv[key].get("floors_toward", fv[key].get("floors")); return x * f1[key] / fmax[key] if fmax[key] else None
        r = {"fta_fga": a["fta_fga"], "fta_toward": resc("fta_fga_pooled"), "ft_corr": a["ft_corr"], "ft_corr_se": a["ft_corr_se"],
             "h1": a["h1"], "h2": a["h2"], "team_slope": a["team_slope"], "team_sd_ratio": a["team_sd_ratio"],
             "g5_total": a["g5_total"], "g5_total_toward": resc("g5_ratio_total"), "g5_margin_toward": resc("g5_ratio_margin"),
             "g1_mean_toward": resc("possessions"), "g1_sd_toward": resc("g1_sd_gap"), "g9_toward": resc("total_bias"),
             "score_corr": a["score_corr"], "score_corr_toward": (abs(sc0 - act) - abs(a["score_corr"] - act)) / max(sc_f),
             "occ_max_gap_pp": 100 * max(abs(v) for v in a["occ_gap"].values()), "vetoes_draw1_floor": a["vetoes"]}
        v3 = (r["g5_total_toward"] >= -1 and r["score_corr_toward"] >= -1 and min(r["g1_mean_toward"], r["g1_sd_toward"], r["g5_margin_toward"], r["g9_toward"]) >= -2)
        r["vetoes_max_floor"] = {**a["vetoes"], "V3": bool(v3)}
        r["eligible_max_floor"] = bool(t != REF0 and all(r["vetoes_max_floor"].values()))
        res["arms"][t] = r
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1, default=float))
    print("floors_max:", {k: round(v, 5) for k, v in fmax.items() if k in ("fta_fga_pooled", "possessions", "total_sd", "g5_ratio_total", "g5_ratio_margin", "total_bias", "g1_sd_gap")}, "score_corr", round(max(sc_f), 5), "ft_corr", round(max(ft_f), 5))
    for t, r in res["arms"].items():
        print(f"{t:22s} FTA/FGA {r['fta_fga']:.5f} ({r['fta_toward']:+.2f}) FTcorr {r['ft_corr']:+.4f} H1 {r['h1']:.4f} H2 {r['h2']:.4f} slope {r['team_slope']:.3f} sdR {r['team_sd_ratio']:.3f} "
              f"G5tot {r['g5_total']:.4f} ({r['g5_total_toward']:+.1f}) scorr {r['score_corr']:.4f} ({r['score_corr_toward']:+.1f}) occmax {r['occ_max_gap_pp']:.1f} V {r['vetoes_max_floor']} elig={r['eligible_max_floor']}")


if __name__ == "__main__":
    main()
