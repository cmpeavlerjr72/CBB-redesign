"""grade_late_game_r3_v1.py -- late-game ROUND 3: the ONE closed-loop grader (experiments.md 6.4-6.5, 7.4).

Every arm goes through the same code path; the arm's identity only labels the row. Lines come from round
2's grader (`grade_late_game_r2_v1.lines`, unchanged). Floors are Decision 12's:

    floor(line) = max(SD of the line over the base reference + four seed-offset draws,
                      2 x paired game-bootstrap SE of (arm - base))

The bootstrap term is computed for the lines that are means over games (P0, P1, P0/P1, G1 mean, G9
biases, half share, window possessions by k); G1 SD and the G5 ratios carry the draw SD only (stated).
B0 arms are read against R0 with the B9 draw SD (B0 has no draws of its own; stated).

    .venv/Scripts/python.exe scripts/grade_late_game_r3_v1.py --base lg3_R9_s25 \
        --draws lg3_R9_f1_s25 lg3_R9_f2_s25 lg3_R9_f3_s25 lg3_R9_f4_s25 \
        --arms lg3_Dt9_s25 lg3_Dtt9_s25 lg3_D9_s25 --b0 lg3_R0_s25 --b0-arms lg3_Dt0_s25 lg3_D0_s25 \
        --out results/late_game/round3/grade.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("CBB_TRUTH", "verified_v1")

import grade_late_game_r2_v1 as G2                                   # noqa: E402
from cbb_sim.eval import reference as REF                             # noqa: E402

POSS = ROOT / "data/processed/possessions_v4/possessions_2025.parquet"
SAMPLE = ROOT / "data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet"
TARGET_RATIO, OT_BAND = 1.546, (0.046, 0.055)
N_BOOT = 1000
KS = range(7)
VETO = dict(G2.VETO_LINES)
BOOT_LINES = ("P0", "P1", "P0_over_P1", "G1_poss_mean", "G9_margin_bias", "G9_total_bias", "half1_share")


def per_game(g: pd.DataFrame) -> pd.DataFrame:
    """Per-game sums the bootstrap lines are ratios of."""
    ot = (g["n_periods"] > 2).to_numpy()
    fm = (g["home_pts"] - g["away_pts"]).abs().to_numpy()
    tot = (g["home_pts"] + g["away_pts"]).to_numpy().astype(float)
    h1 = (g["h1_home_pts"] + g["h1_away_pts"]).to_numpy().astype(float)
    poss = g["possessions"].to_numpy()
    d = pd.DataFrame({"game_id": g["game_id"].to_numpy(), "n": 1.0, "p0": ot.astype(float),
                      "p1": ((~ot) & (fm == 1)).astype(float), "poss": poss.astype(float),
                      "margin": (g["home_pts"] - g["away_pts"]).to_numpy().astype(float),
                      "total": tot, "h1share": h1 / tot})
    k = g["home_margin_120"].abs().to_numpy()
    for j in KS:
        s = k == j
        d[f"wk{j}_n"] = s.astype(float)
        d[f"wk{j}_w"] = np.where(s, g["window_poss"].to_numpy(), 0.0)
    return d.groupby("game_id").sum()


def boot_lines(A: pd.DataFrame, act: pd.DataFrame) -> dict:
    """Line values from summed per-game frames (A may be a bootstrap resample)."""
    s = A.sum()
    out = {"P0": s["p0"] / s["n"], "P1": s["p1"] / s["n"]}
    out["P0_over_P1"] = out["P0"] / max(out["P1"], 1e-12)
    out["G1_poss_mean"] = s["poss"] / s["n"]
    out["half1_share"] = s["h1share"] / s["n"]
    am = act.reindex(A.index)
    out["G9_margin_bias"] = (A["margin"] / A["n"] - am["margin"]).mean()
    out["G9_total_bias"] = (A["total"] / A["n"] - am["total"]).mean()
    for j in KS:
        out[f"win_k{j}"] = s[f"wk{j}_w"] / max(s[f"wk{j}_n"], 1e-12)
    return out


def paired_boot_se(base: pd.DataFrame, arm: pd.DataFrame, act: pd.DataFrame, seed: int = 11) -> dict:
    arm = arm.loc[base.index]
    rng = np.random.default_rng(seed)
    idx = base.index.to_numpy()
    deltas = []
    for _ in range(N_BOOT):
        ix = idx[rng.integers(0, len(idx), len(idx))]
        b, a = boot_lines(base.loc[ix], act), boot_lines(arm.loc[ix], act)
        deltas.append({k: a[k] - b[k] for k in b})
    return pd.DataFrame(deltas).std().to_dict()


def actual_window(ids: set | None) -> dict:
    c = pd.read_parquet(POSS, columns=["game_id", "period", "poss_index", "start_clock", "start_score_diff",
                                       "offense_is_home"])
    c = c[(c["period"] == 2) & (c["start_clock"] <= 120)].sort_values(["game_id", "poss_index"])
    if ids is not None:
        c = c[c["game_id"].isin(ids)]
    first = c.groupby("game_id").first()
    k = first["start_score_diff"].abs()
    win = c[c["start_score_diff"].abs() <= 6].groupby("game_id").size().reindex(k.index).fillna(0)
    return {str(j): {"n_games": int((k == j).sum()),
                     "window_poss": float(win[k == j].mean()) if (k == j).any() else None,
                     "se": float(win[k == j].std() / np.sqrt(max((k == j).sum(), 1))) if (k == j).sum() > 1 else None}
            for j in KS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--draws", nargs=4, required=True)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--b0", default=None)
    ap.add_argument("--b0-arms", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = set(pd.read_parquet(SAMPLE)["game_id"].astype("int64"))
    actg = REF.load_actual_games(2025).set_index("game_id")
    act = pd.DataFrame({"margin": actg["home_score"] - actg["away_score"],
                        "total": actg["home_score"] + actg["away_score"]})
    aw_sample, aw_season = actual_window(ids), actual_window(None)

    runs = {t: G2.load(t) for t in [a.base, *a.draws, *a.arms] + ([a.b0] if a.b0 else []) + a.b0_arms}
    L = {t: G2.lines(r) for t, r in runs.items()}
    PG = {t: per_game(r["g"]) for t, r in runs.items()}
    for t in runs:
        bl = boot_lines(PG[t], act)
        for j in KS:
            L[t][f"win_k{j}"] = bl[f"win_k{j}"]
    ref_set = [a.base, *a.draws]
    line_names = ["P0", "P1", "P0_over_P1", *VETO, *[f"win_k{j}" for j in KS]]
    draw_sd = {k: float(np.std([L[t][k] for t in ref_set], ddof=1)) for k in line_names}

    def judge(base_tag: str, tag: str) -> dict:
        LB, LA = L[base_tag], L[tag]
        se = paired_boot_se(PG[base_tag], PG[tag], act)
        fl = {k: max(draw_sd[k], 2 * se.get(k, 0.0)) for k in line_names}
        v = {"tag": tag, "base": base_tag, "arm": LA["arm"], "ot_rate": LA["ot_rate"],
             "P0_over_P1": LA["P0_over_P1"], "d_ratio": LA["P0_over_P1"] - LB["P0_over_P1"],
             "floor_ratio": fl["P0_over_P1"], "draw_sd_ratio": draw_sd["P0_over_P1"],
             "boot_se_ratio": se["P0_over_P1"], "d_P0": LA["P0"] - LB["P0"], "se_dP0": se["P0"],
             "floor_P0": fl["P0"]}
        v["floors_ratio"] = v["d_ratio"] / fl["P0_over_P1"]
        v["ratio_ge_1"] = LA["P0_over_P1"] >= 1.0
        v["ot_in_band"] = OT_BAND[0] <= LA["ot_rate"] <= OT_BAND[1]
        vet = {}
        for k, (tgt, _) in VETO.items():
            t_b = LB[tgt] if isinstance(tgt, str) else tgt
            t_a = LA[tgt] if isinstance(tgt, str) else tgt
            eb, ea = abs(LB[k] - t_b), abs(LA[k] - t_a)
            vet[k] = {"base": LB[k], "arm": LA[k], "target": t_a, "floor": fl[k],
                      "floor_source": "draw SD only" if k not in BOOT_LINES else "max(draw SD, 2 x boot SE)",
                      "worse_in_floors": (ea - eb) / fl[k] if fl[k] else None, "pass": bool(ea - eb <= fl[k])}
        fh = G2.first_half_identical(runs[base_tag]["g"], runs[tag]["g"])
        vet["first_half_bit_identical"] = {"pass": fh["identical"], **fh}
        wk = {}
        for j in KS:
            key = f"win_k{j}"
            actv = aw_sample[str(j)]["window_poss"]
            base_excess = LB[key] - actv > fl[key]
            ref_val = LB[key] if base_excess else actv
            wk[str(j)] = {"arm": LA[key], "base": LB[key], "actual_sample": actv,
                          "actual_season": aw_season[str(j)]["window_poss"],
                          "n_actual_games": aw_sample[str(j)]["n_games"], "floor": fl[key],
                          "read_against": "base (base exceeds actual by > 1 floor)" if base_excess else "actual",
                          "excess_floors": (LA[key] - ref_val) / fl[key] if fl[key] else None,
                          "pass": bool(LA[key] - ref_val <= fl[key])}
        vet["window_poss_by_k"] = {"pass": all(x["pass"] for x in wk.values()), "by_k": wk}
        v["vetoes"] = vet
        v["all_vetoes_pass"] = all(x["pass"] for x in vet.values())
        v["candidate"] = bool(v["floors_ratio"] > 1 and v["ratio_ge_1"] and v["all_vetoes_pass"])
        return v

    verdicts = [judge(a.base, t) for t in a.arms]
    if a.b0:
        verdicts.append(judge(a.base, a.b0) | {"note": "B0 reference vs B9 reference: foul-state attribution"})
        verdicts += [judge(a.b0, t) | {"note": "attribution only (B0), cannot be a candidate"}
                     for t in a.b0_arms]
    # the floor draws' own spread, as a sanity line
    out = {"spec": "docs/models/late_game/experiments.md sections 6.4-6.5 and 7.4",
           "base": a.base, "draws": a.draws, "draw_sd": draw_sd,
           "lines": {t: {k: v for k, v in L[t].items() if k not in ("window_duration", "window_event_by_role")}
                     | {"window_duration": L[t]["window_duration"],
                        "window_event_by_role": L[t]["window_event_by_role"]} for t in L},
           "actual_window_sample": aw_sample, "actual_window_season": aw_season,
           "verdicts": verdicts, "target": {"P0_over_P1": TARGET_RATIO, "ot_band": OT_BAND}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    print(f"{'tag':18s} {'OT':>7s} {'P1':>7s} {'P0/P1':>7s} {'G1m':>7s} {'G1sd':>6s} {'G9m':>7s} {'G9t':>7s} "
          f"{'h1sh':>7s} winposs k0..6")
    for t in L:
        x = L[t]
        print(f"{t:18s} {x['P0']:.4f} {x['P1']:.4f} {x['P0_over_P1']:7.3f} {x['G1_poss_mean']:7.3f} "
              f"{x['G1_poss_sd']:6.3f} {x['G9_margin_bias']:+7.3f} {x['G9_total_bias']:+7.3f} "
              f"{x['half1_share']:7.4f} " + " ".join(f"{x[f'win_k{j}']:.2f}" for j in KS))
    print("actual sample winposs:", " ".join(f"{aw_sample[str(j)]['window_poss']:.2f}" for j in KS),
          " season:", " ".join(f"{aw_season[str(j)]['window_poss']:.2f}" for j in KS))
    print("draw SD:", {k: round(v, 5) for k, v in draw_sd.items()})
    for v in verdicts:
        bad = [k for k, x in v["vetoes"].items() if not x["pass"]]
        print(f"{v['tag']:18s} vs {v['base']:12s} ratio {v['P0_over_P1']:.3f} d {v['d_ratio']:+.3f} "
              f"floor {v['floor_ratio']:.3f} (draw {v['draw_sd_ratio']:.3f}, boot {v['boot_se_ratio']:.3f}) "
              f"=> {v['floors_ratio']:+.2f} fl; OT {v['ot_rate']:.4f} band {v['ot_in_band']}; "
              f"vetoes {'PASS' if not bad else 'FAIL ' + ','.join(bad)}; candidate {v['candidate']}")
        wk = v["vetoes"]["window_poss_by_k"]["by_k"]
        print("    window k excess (floors): " + " ".join(
            f"{j}:{(wk[j]['excess_floors'] or 0):+.1f}{'*' if wk[j]['read_against'] != 'actual' else ''}" for j in wk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
