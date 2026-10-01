"""grade_late_game_r4_v1.py -- late-game ROUND 4: the ONE closed-loop grader (experiments.md 9.4).

Round 3's grader (`grade_late_game_r3_v1.py`) with 9.4's one change: the first-half bit-identity veto is
replaced by the FIRST-HALF BUZZER TEST (H1 points per possession for possessions starting at <= 10 s must move
toward the 2024-25 actual and not overshoot it by more than one floor). Every other line, floor and the
window-possession reading are round 3's code, imported.

    .venv/Scripts/python.exe scripts/grade_late_game_r4_v1.py --base lg4_R9_s25 \
        --draws lg4_R9_f1_s25 lg4_R9_f2_s25 lg4_R9_f3_s25 lg4_R9_f4_s25 \
        --arms lg3_Dt9_s25 lg4_DtBZ_s25 lg4_DtA_s25 lg4_DtL_s25 lg4_DtLA_s25 --out results/late_game/round4/grade.json
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
import grade_late_game_r3_v1 as G3                                   # noqa: E402
from cbb_sim.eval import reference as REF                             # noqa: E402

RES = ROOT / "results/engine_v0"
H1_SEC = 10
ORDER = ["lg4_DtBZ_s25", "lg4_DtA_s25", "lg4_DtL_s25", "lg4_DtLA_s25"]   # 9.4 simplicity order


def h1_buzzer_pg(tag: str) -> pd.DataFrame | None:
    """Per game: H1 possessions starting at <= 10 s, their count and points (needs the round-4 tap)."""
    p = RES / tag / "tap_poss.parquet"
    if not p.exists():
        return None
    t = pd.read_parquet(p)
    if "per" not in t:
        return None
    t = t[(t["per"] == 1) & (t["sec"] <= H1_SEC)]
    return t.groupby("game_id").agg(n=("d_pts_off", "size"), pts=("d_pts_off", "sum"))


def _fh_same(b, x) -> bool:
    fh = G2.first_half_identical(b, x)
    return bool(fh["identical"] and fh["n_sims"] == len(b))


def act_h1() -> float:
    d = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2025.parquet",
                        columns=["period", "start_clock", "points"])
    d = d[(d["period"] == 1) & (d["start_clock"] <= H1_SEC)]
    return float(d["points"].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--draws", nargs=4, required=True)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = set(pd.read_parquet(G3.SAMPLE)["game_id"].astype("int64"))
    actg = REF.load_actual_games(2025).set_index("game_id")
    act = pd.DataFrame({"margin": actg["home_score"] - actg["away_score"],
                        "total": actg["home_score"] + actg["away_score"]})
    aw_sample, aw_season = G3.actual_window(ids), G3.actual_window(None)
    a_h1 = act_h1()
    tags = [a.base, *a.draws, *a.arms]
    runs = {t: G2.load(t) for t in tags}
    L = {t: G2.lines(r) for t, r in runs.items()}
    PG = {t: G3.per_game(r["g"]) for t, r in runs.items()}
    H1 = {t: h1_buzzer_pg(t) for t in tags}
    for t in tags:   # a run without the round-4 tap whose first half is bit-identical to the base: same H1
        if H1[t] is None and H1[a.base] is not None and \
                _fh_same(runs[a.base]["g"], runs[t]["g"]):
            H1[t] = H1[a.base]
    for t in tags:
        bl = G3.boot_lines(PG[t], act)
        for j in G3.KS:
            L[t][f"win_k{j}"] = bl[f"win_k{j}"]
        L[t]["h1_buzzer_ppp"] = (float(H1[t]["pts"].sum() / H1[t]["n"].sum()) if H1[t] is not None else None)
    ref_set = [a.base, *a.draws]
    names = ["P0", "P1", "P0_over_P1", *G3.VETO, *[f"win_k{j}" for j in G3.KS]]
    draw_sd = {k: float(np.std([L[t][k] for t in ref_set], ddof=1)) for k in names}
    hv = [L[t]["h1_buzzer_ppp"] for t in ref_set if L[t]["h1_buzzer_ppp"] is not None]
    draw_sd["h1_buzzer_ppp"] = float(np.std(hv, ddof=1)) if len(hv) >= 2 else None

    def h1_boot(bt, at):
        b, x = H1[bt], H1[at]
        if b is None or x is None:
            return None
        j = b.join(x, lsuffix="_b", rsuffix="_a", how="outer").fillna(0)
        rng = np.random.default_rng(13)
        idx = np.arange(len(j))
        A = j.to_numpy()
        out = []
        for _ in range(G3.N_BOOT):
            s = A[rng.integers(0, len(idx), len(idx))].sum(0)
            out.append(s[3] / s[2] - s[1] / s[0])
        return float(np.std(out))

    verdicts = []
    for t in a.arms:
        LB, LA = L[a.base], L[t]
        se = G3.paired_boot_se(PG[a.base], PG[t], act)
        fl = {k: max(draw_sd[k], 2 * se.get(k, 0.0)) for k in names}
        v = {"tag": t, "flags": runs[t]["meta"].get("flags", {"ENGINE_LATE_GAME": runs[t]["meta"].get("arm")}),
             "ot_rate": LA["ot_rate"], "P0_over_P1": LA["P0_over_P1"],
             "d_ratio": LA["P0_over_P1"] - LB["P0_over_P1"], "floor_ratio": fl["P0_over_P1"],
             "d_P0": LA["P0"] - LB["P0"], "se_dP0": se["P0"]}
        v["floors_ratio"] = v["d_ratio"] / fl["P0_over_P1"]
        v["ratio_ge_1"] = LA["P0_over_P1"] >= 1.0
        v["ot_in_band"] = 0.046 <= LA["ot_rate"] <= 0.055
        vet = {}
        for k, (tgt, _) in G3.VETO.items():
            t_b = LB[tgt] if isinstance(tgt, str) else tgt
            t_a = LA[tgt] if isinstance(tgt, str) else tgt
            eb, ea = abs(LB[k] - t_b), abs(LA[k] - t_a)
            vet[k] = {"base": LB[k], "arm": LA[k], "target": t_a, "floor": fl[k],
                      "worse_in_floors": (ea - eb) / fl[k] if fl[k] else None, "pass": bool(ea - eb <= fl[k])}
        wk = {}
        for j in G3.KS:
            key = f"win_k{j}"
            actv = aw_sample[str(j)]["window_poss"]
            base_ex = LB[key] - actv > fl[key]
            ref_val = LB[key] if base_ex else actv
            wk[str(j)] = {"arm": LA[key], "base": LB[key], "actual_sample": actv,
                          "actual_season": aw_season[str(j)]["window_poss"], "floor": fl[key],
                          "read_against": "base" if base_ex else "actual",
                          "excess_floors": (LA[key] - ref_val) / fl[key], "pass": bool(LA[key] - ref_val <= fl[key])}
        vet["window_poss_by_k"] = {"pass": all(x["pass"] for x in wk.values()), "by_k": wk}
        hb, ha = LB["h1_buzzer_ppp"], LA["h1_buzzer_ppp"]
        hse = h1_boot(a.base, t)
        if hb is not None and ha is not None:
            hfl = max(draw_sd["h1_buzzer_ppp"] or 0.0, 2 * (hse or 0.0))
            toward = abs(ha - a_h1) <= abs(hb - a_h1) or abs(ha - hb) <= hfl
            over = (hb - a_h1) * (ha - a_h1) < 0 and abs(ha - a_h1) > hfl
            vet["h1_buzzer_test"] = {"base": hb, "arm": ha, "actual": a_h1, "floor": hfl,
                                     "toward": bool(toward), "overshoot": bool(over),
                                     "pass": bool(toward and not over)}
        else:
            vet["h1_buzzer_test"] = {"pass": False, "note": "no round-4 tap for base or arm"}
        fh = G2.first_half_identical(runs[a.base]["g"], runs[t]["g"])
        v["first_half_identical_reported"] = fh
        v["vetoes"] = vet
        v["all_vetoes_pass"] = all(x["pass"] for x in vet.values())
        v["candidate"] = bool(v["floors_ratio"] > 1 and v["ratio_ge_1"] and v["all_vetoes_pass"])
        verdicts.append(v)
    # clear best (9.4): passes every veto and beats every other arm's primary by > 1 floor
    for v in verdicts:
        others = [w for w in verdicts if w is not v]
        v["clear_best"] = bool(v["all_vetoes_pass"] and all(
            v["d_ratio"] - w["d_ratio"] > max(v["floor_ratio"], w["floor_ratio"]) for w in others))
    out = {"spec": "docs/models/late_game/experiments.md section 9.4", "base": a.base, "draws": a.draws,
           "draw_sd": draw_sd, "actual_h1_buzzer_ppp": a_h1,
           "lines": {t: {k: v for k, v in L[t].items() if k not in ("window_duration", "window_event_by_role")} for t in L},
           "verdicts": verdicts}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"{'tag':16s} {'OT':>7s} {'P1':>7s} {'P0/P1':>6s} {'G1m':>7s} {'G1sd':>6s} {'G9t':>7s} {'h1sh':>7s} {'H1buz':>6s}")
    for t in L:
        x = L[t]
        print(f"{t:16s} {x['P0']:.4f} {x['P1']:.4f} {x['P0_over_P1']:6.3f} {x['G1_poss_mean']:7.3f} {x['G1_poss_sd']:6.3f} "
              f"{x['G9_total_bias']:+7.3f} {x['half1_share']:7.4f} {x['h1_buzzer_ppp'] if x['h1_buzzer_ppp'] is not None else float('nan'):6.3f}")
    print(f"actual H1 buzzer PPP {a_h1:.3f}; draw SD {draw_sd}")
    for v in verdicts:
        bad = [k for k, x in v["vetoes"].items() if not x["pass"]]
        print(f"{v['tag']:16s} ratio {v['P0_over_P1']:.3f} d {v['d_ratio']:+.3f} floor {v['floor_ratio']:.3f} => "
              f"{v['floors_ratio']:+.2f} fl; OT {v['ot_rate']:.4f} (dP0 {v['d_P0']:+.4f} se {v['se_dP0']:.4f}); "
              f"vetoes {'PASS' if not bad else 'FAIL ' + ','.join(bad)}; candidate {v['candidate']}; clear_best {v['clear_best']}")
        for k in bad:
            x = v["vetoes"][k]
            if "worse_in_floors" in x:
                print(f"      {k}: base {x['base']:.4f} arm {x['arm']:.4f} target {x['target']:.4f} worse {x['worse_in_floors']:+.2f} fl")
        h = v["vetoes"]["h1_buzzer_test"]
        if "base" in h:
            print(f"      H1 buzzer PPP base {h['base']:.3f} arm {h['arm']:.3f} actual {h['actual']:.3f} floor {h['floor']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
