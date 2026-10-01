"""diag_laneI_full_read_v1.py -- full-size (5,710 x 200) read of lane I arms vs served stack v2 (lane I, 2026-10-01).

Reference `v3full_COMB9GCTKD_s200_o0`. Floors (Decision 12): max over the four served-v2 seed-offset draws
`d1001D_S2f{1..4}_s200_o{k}000` of |draw - reference| per line; if those are not on disk, the four `v3full_S0f*` draws
against `v3full_S0_s200_o0` (stated in the output). Lines: the gate report's own values (`grade_shot_block_closed_loop_v1.
gate_lines`), pooled FT% and OREB share, and the team OREB% slopes (offence / defence by 2024-prior quintile,
`diag_team_oreb_resp_v1.lines`). Veto rule (rebound 12.3 / free_throw 13.2-13.3): a veto line moving AWAY from its target
by more than 2 floors, or any gate verdict PASS -> FAIL.

Usage: diag_laneI_full_read_v1.py <out_json> <arm_tag> [<arm_tag> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
sys.argv, _argv = [sys.argv[0], "x"], sys.argv            # diag_team_oreb_resp_v1 reads argv at import
import grade_shot_block_closed_loop_v1 as SB  # noqa: E402
sys.argv = [sys.argv[0], str(ROOT / "results/laneI_1001/_tmp_oreb.json")]
import diag_team_oreb_resp_v1 as TO  # noqa: E402
sys.argv = _argv

R = ROOT / "results/engine_v0"
OUT, ARMS = Path(sys.argv[1]), sys.argv[2:]
REF = "v3full_COMB9GCTKD_s200_o0"
S2 = [f"d1001D_S2f{k}_s200_o{k}000" for k in range(1, 5)]
S0 = ("v3full_S0_s200_o0", ["v3full_S0f1_s200_o1000", "v3full_S0f2_s200_o2000", "v3full_S0f3_s200_o3000", "v3full_S0f4_s200_o4000"])
TARGET = {"G4_oreb": 0.2984}
VETO = ["G1_poss_mean", "G1_poss_sd", "G5_margin_ratio", "G5_total_ratio", "G5_corr", "G9_margin_bias", "G9_total_bias",
        "G9_slope", "G4_efg", "G4_tov", "G4_ft_rate"]


def pooled(tag):
    g = pd.read_parquet(R / tag / "games.parquet")
    fta, ftm = g["home_fta"].sum() + g["away_fta"].sum(), g["home_ftm"].sum() + g["away_ftm"].sum()
    return {"ft_pct": float(ftm / fta)}


def all_lines(tag):
    L = SB.gate_lines(tag)
    out = {k: v["value"] for k, v in L.items()}
    tgt = {k: v["target"] for k, v in L.items()}
    st = {k: v["status"] for k, v in L.items()}
    out.update(pooled(tag))
    t = TO.lines(tag)
    out.update({"team_off_slope": t["off_slope"], "team_def_slope": t["def_slope"]})
    return out, tgt, st


if all((R / d / "games.parquet").exists() for d in S2):
    fref, draws, src = REF, S2, "served-v2 draws d1001D_S2f1..4"
else:
    fref, draws, src = S0[0], S0[1], "FALLBACK: v3full_S0f1..4 vs v3full_S0 (served-v2 draws not on disk)"
F0, _, _ = all_lines(fref)
FD = [all_lines(d)[0] for d in draws]
floors = {k: max(abs(f[k] - F0[k]) for f in FD) for k in F0}
L0, T0, S0st = all_lines(REF)
ft_actual = 0.7213   # verified box FT% (ppp decomposition, 5,700 graded games)
res = {"ref": REF, "floor_source": src, "floors": floors, "ref_lines": L0, "arms": {}}
for a in ARMS:
    L, T, St = all_lines(a)
    rows = {}
    for k in L:
        tgt = TARGET.get(k, T.get(k))
        if k == "ft_pct":
            tgt = ft_actual
        if k in ("team_off_slope", "team_def_slope"):
            tgt = 1.0
        if tgt is None:
            tgt = 1.0 if ("ratio" in k or "slope" in k) else 0.0
        fl = floors.get(k) or float("nan")
        d0, d1 = abs(L0[k] - tgt), abs(L[k] - tgt)
        rows[k] = {"ref": L0[k], "arm": L[k], "delta": L[k] - L0[k], "target": tgt, "floor": fl,
                   "floors_toward": (d0 - d1) / fl if fl and fl == fl and fl > 0 else None,
                   "flip": f"{S0st.get(k)}->{St.get(k)}" if k in St and St.get(k) != S0st.get(k) else None}
    veto = [k for k in VETO if k in rows and rows[k]["floors_toward"] is not None and rows[k]["floors_toward"] < -2]
    flips_bad = [k for k, r in rows.items() if r["flip"] == "PASS->FAIL"]
    slope_fall = [k for k in ("team_off_slope", "team_def_slope") if rows[k]["delta"] < -floors[k]]
    res["arms"][a] = {"lines": rows, "vetoes_fired": veto, "pass_to_fail": flips_bad, "team_slope_falls": slope_fall}
    print(a, "| OREB", round(rows["G4_oreb"]["arm"], 4), f"{rows['G4_oreb']['floors_toward']:+.1f}fl | FT%", round(rows["ft_pct"]["arm"], 4),
          f"{rows['ft_pct']['floors_toward']:+.1f}fl | G9 total {rows['G9_total_bias']['arm']:+.3f} | slope offdef",
          round(rows["team_off_slope"]["arm"], 3), round(rows["team_def_slope"]["arm"], 3), "| veto", veto, flips_bad, slope_fall)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
