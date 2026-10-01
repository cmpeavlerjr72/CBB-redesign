"""grade_late_game_r5_v1.py -- late-game ROUND 5 decision (experiments.md section 11.3).

Reads the round-4 grader's output (same lines, same floors, one code path for every arm) and applies 11.3:
  * HARD vetoes: G1 mean, G1 SD, half share, window possessions, first-half buzzer test, G5 margin and total SD
    ratios, G9 margin bias;
  * G9 TOTAL bias is reported as a priced exposure (points per game vs R9 by half, paired game-bootstrap SE),
    not as a disqualifier;
  * candidate = section 1.3 + every hard veto; clear best = every hard veto and primary > every other
    hard-veto-passing arm's by more than one floor;
  * the remaining gap reported separately: arrival at a tie (share of sims tied at 1:00 / 0:30 / 0:10) and
    the one-point-finish rate, from the tie-loss diagnostic.

    .venv/Scripts/python.exe scripts/grade_late_game_r5_v1.py --grade results/late_game/round5/grade_r4tool.json \
        --tieloss results/late_game/round5/tieloss.json --base lg4_R9_s25 --set lg5_DtLLa1_s25 --out results/late_game/round5/decision.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/engine_v0"
HARD = ("G1_poss_mean", "G1_poss_sd", "half1_share", "window_poss_by_k", "h1_buzzer_test",
        "G5_margin_sd_ratio", "G5_total_sd_ratio", "G9_margin_bias")


def pts_by_half(tag):
    g = pd.read_parquet(RES / tag / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts"])
    t = pd.read_parquet(RES / tag / "tap_sims.parquet", columns=["game_id", "seed", "h1_home_pts", "h1_away_pts"])
    g = g.merge(t, on=["game_id", "seed"])
    g["h1"] = g["h1_home_pts"] + g["h1_away_pts"]
    g["tot"] = g["home_pts"] + g["away_pts"]
    g["rest"] = g["tot"] - g["h1"]
    return g.groupby("game_id")[["h1", "rest", "tot"]].mean()


def paired(base, arm, n=1000, seed=5):
    a, b = base, arm.loc[base.index]
    d = (b - a).to_numpy()
    r = np.random.default_rng(seed)
    bs = np.array([d[r.integers(0, len(d), len(d))].mean(0) for _ in range(n)])
    return {c: {"delta": float(d[:, i].mean()), "se": float(bs[:, i].std())} for i, c in enumerate(base.columns)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade", required=True)
    ap.add_argument("--tieloss", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = json.loads(Path(a.grade).read_text(encoding="utf-8"))
    T = json.loads(Path(a.tieloss).read_text(encoding="utf-8"))
    pb = pts_by_half(a.base)
    rows = []
    for v in G["verdicts"]:
        hard = {k: bool(v["vetoes"][k]["pass"]) for k in HARD if k in v["vetoes"]}
        r = {"tag": v["tag"], "P0_over_P1": v["P0_over_P1"], "floors": v["floors_ratio"], "d_ratio": v["d_ratio"],
             "floor_ratio": v["floor_ratio"], "ot": v["ot_rate"], "hard_vetoes": hard,
             "hard_pass": all(hard.values()), "hard_failed": [k for k, x in hard.items() if not x],
             "g9_total_exposure": v["vetoes"]["G9_total_bias"]}
        try:
            r["points_per_game_vs_base"] = paired(pb, pts_by_half(v["tag"]))
        except Exception as e:                                         # noqa: BLE001
            r["points_per_game_vs_base"] = {"error": str(e)}
        r["candidate"] = bool(v["floors_ratio"] > 1 and v["P0_over_P1"] >= 1.0 and r["hard_pass"])
        tl = T["runs"].get(v["tag"])
        if tl:
            ss = {s["T"]: s for s in tl["shift_share_vs_season"]}
            r["arrival_tied_share"] = {f"T{t}": ss[t]["sim_dist"][0] for t in (60, 30, 10)}
            r["P1_end"] = tl["p_abs1_end"]
        rows.append(r)
    for r in rows:
        others = [w for w in rows if w is not r and w["hard_pass"]]
        r["clear_best"] = bool(r["hard_pass"] and all(r["d_ratio"] - w["d_ratio"] > max(r["floor_ratio"], w["floor_ratio"])
                                                       for w in others))
    base_tl = T["runs"].get(a.base, {})
    act = T["actual"]["actual_season"]
    out = {"spec": "docs/models/late_game/experiments.md section 11.3", "set": a.set, "rows": rows,
           "base_arrival": {f"T{s['T']}": s["sim_dist"][0] for s in base_tl.get("shift_share_vs_season", [])},
           "base_P1_end": base_tl.get("p_abs1_end"),
           "actual_arrival": {f"T{s['T']}": s["act_dist"][0] for s in base_tl.get("shift_share_vs_season", [])},
           "actual_P1_end": act.get("p_abs1_end")}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("base arrival", out["base_arrival"], "P1", out["base_P1_end"], "| actual", out["actual_arrival"], "P1", out["actual_P1_end"])
    for r in rows:
        pp = r["points_per_game_vs_base"]
        ps = (f"pts/g {pp['tot']['delta']:+.3f} (H1 {pp['h1']['delta']:+.3f}, H2+OT {pp['rest']['delta']:+.3f}; se {pp['tot']['se']:.3f})"
              if "tot" in pp else str(pp))
        print(f"{r['tag']:16s} ratio {r['P0_over_P1']:.3f} {r['floors']:+.2f} fl OT {r['ot']:.4f} hard {'PASS' if r['hard_pass'] else 'FAIL ' + ','.join(r['hard_failed'])}"
              f" | cand {r['candidate']} clear_best {r['clear_best']} | {ps} | arrival {r.get('arrival_tied_share')} P1 {r.get('P1_end')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
