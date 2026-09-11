#!/usr/bin/env python
"""
diag_fg_make_r4_closedloop.py -- the Decision-10 closed-loop gate for arbitrary
fg_make engine runs (the round-3 interim arm's missing gate, and every round-4
arm).

    .venv/Scripts/python.exe scripts/diag_fg_make_r4_closedloop.py \
        --reference fgm4_cl_r2b_ref \
        --arm fgm4_cl_r3_stockinputs --arm fgm4_cl_r3_fixedinputs \
        --out data/processed/models/fg_make/round4/closed_loop_interim.json

Same quantities, same tolerances and the same reference-relative form as
`scripts/diag_fg_make_round2_closedloop.py` (`docs/models/fg_make/experiments.md`
section 13.5) -- `read_arm` is IMPORTED from that script rather than
reimplemented, so the two tables are comparable line for line. The only
difference is that the arm list and the reference arm are given by tag on the
command line instead of being a fixed tuple, because rounds 3 and 4 compare
runs that differ in the ENGINE INPUT DIRECTORY as well as in
`ENGINE_FG_MAKE`.

Nothing here adjusts anything; it only reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diag_fg_make_round2_closedloop import TOL, TRUTH, read_arm  # noqa: E402

from cbb_sim.eval import reference as REF  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--results-dir", default="results/engine_v0")
    ap.add_argument("--reference", required=True, help="reference run TAG")
    ap.add_argument("--arm", action="append", default=[], help="run TAG (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    act = REF.load_actual_games(a.season)[
        ["game_id", "margin", "total", "home_score", "away_score"]].rename(
        columns={"margin": "act_margin", "total": "act_total"})
    truth = pd.read_parquet(TRUTH) if TRUTH.exists() else None
    if truth is not None:
        truth = truth[truth["season"] == a.season]

    tags = [a.reference] + [t for t in a.arm if t != a.reference]
    res = {}
    for tag in tags:
        p = Path(a.results_dir) / tag
        if not (p / "games.parquet").exists():
            raise FileNotFoundError(f"{p}/games.parquet missing")
        res[tag] = read_arm(p, a.season, act, truth)
        res[tag]["input_dir"] = json.loads(
            (p / "run_meta.json").read_text(encoding="utf-8")).get("input_dir", "(not recorded)")

    ref = res[a.reference]
    games = set()
    for tag, r in res.items():
        r["d_margin_sd"] = r["margin_sd"] - ref["margin_sd"]
        r["d_corr"] = r["corr_home_away"] - ref["corr_home_away"]
        r["d_poss"] = r["poss_per_game"] - ref["poss_per_game"]
        r["d_total_bias"] = r["total_bias"] - ref["total_bias"]
        r["CL1"] = "PASS" if abs(r["d_margin_sd"]) <= TOL["CL1_margin_sd"] else "FAIL"
        r["CL2"] = "PASS" if abs(r["d_corr"]) <= TOL["CL2_corr"] else "FAIL"
        r["CL3"] = "PASS" if abs(r["d_poss"]) <= TOL["CL3_poss"] else "FAIL"
        r["CL4"] = "PASS" if abs(r["d_total_bias"]) <= TOL["CL4_total_bias"] else "FAIL"
        r["closed_loop"] = "PASS" if all(r[f"CL{i}"] == "PASS" for i in (1, 2, 3, 4)) else "FAIL"
        games.add(r["n_games"])
    assert len(games) == 1, f"arms were scored on different game sets: {games}"

    sub = act[act["game_id"].isin(
        pd.read_parquet(Path(a.results_dir) / a.reference / "games.parquet")
        ["game_id"].unique())]
    actual = {
        "margin_sd": float(sub["act_margin"].std()),
        "total_sd": float(sub["act_total"].std()),
        "corr_home_away": float(np.corrcoef(sub["home_score"].astype(float),
                                            sub["away_score"].astype(float))[0, 1]),
        "margin_mean": float(sub["act_margin"].mean()),
        "total_mean": float(sub["act_total"].mean()),
        "n_games": int(len(sub)),
    }
    ap_ = REF.load_actual_possessions(a.season)
    ap_ = ap_[ap_["game_id"].isin(sub["game_id"])]
    actual["poss_per_game"] = float(ap_["game_poss"].mean())
    actual["ppp"] = float(actual["total_mean"] / (2 * actual["poss_per_game"]))

    w = max(len(t) for t in tags) + 2
    print("=" * (w + 76))
    print(f"fg_make Decision-10 CLOSED-LOOP  (season {a.season}, {actual['n_games']} games "
          f"x {ref['n_seeds']} seeds, paired streams); reference = {a.reference}")
    print("=" * (w + 76))
    print(f"{'run':<{w}}{'marginSD':>10}{'dSD':>8}{'corr':>8}{'dcorr':>8}"
          f"{'poss':>9}{'dposs':>8}{'totbias':>9}{'dtb':>8}{'PPP':>8}  gate")
    for tag in tags:
        r = res[tag]
        print(f"{tag:<{w}}{r['margin_sd']:>10.3f}{r['d_margin_sd']:>+8.3f}"
              f"{r['corr_home_away']:>8.3f}{r['d_corr']:>+8.3f}"
              f"{r['poss_per_game']:>9.3f}{r['d_poss']:>+8.3f}"
              f"{r['total_bias']:>+9.3f}{r['d_total_bias']:>+8.3f}"
              f"{r['ppp']:>8.4f}  {r['closed_loop']}"
              f" (CL1 {r['CL1']}, CL2 {r['CL2']}, CL3 {r['CL3']}, CL4 {r['CL4']})")
    print(f"{'ACTUAL':<{w}}{actual['margin_sd']:>10.3f}{'--':>8}"
          f"{actual['corr_home_away']:>8.3f}{'--':>8}"
          f"{actual['poss_per_game']:>9.3f}{'--':>8}{0.0:>+9.3f}{'--':>8}"
          f"{actual['ppp']:>8.4f}")

    print("\neFG% AND MAKE RATE BY SHOT CLASS (engine vs event-layer truth, same games)")
    print(f"{'run':<{w}}{'eFG%':>8}{'rim%':>8}{'jump2%':>9}{'three%':>9}{'FT%':>8}"
          f"{'rimShr':>9}{'jmpShr':>9}{'3Shr':>8}")
    for tag in tags:
        r = res[tag]
        print(f"{tag:<{w}}{r['efg_pct']:>8.3f}{r['make_pct_rim']:>8.3f}"
              f"{r['make_pct_jump2']:>9.3f}{r['make_pct_three']:>9.3f}{r['ft_pct']:>8.3f}"
              f"{r['share_rim']:>9.3f}{r['share_jump2']:>9.3f}{r['share_three']:>8.3f}")
    if "actual" in ref:
        av = ref["actual"]
        print(f"{'ACTUAL':<{w}}{av['efg_pct']:>8.3f}{av['make_pct_rim']:>8.3f}"
              f"{av['make_pct_jump2']:>9.3f}{av['make_pct_three']:>9.3f}{av['ft_pct']:>8.3f}"
              f"{av['share_rim']:>9.3f}{av['share_jump2']:>9.3f}{av['share_three']:>8.3f}")
        print("  (eFG% actual is box_fgm/box_fga/box_fgm3; per-class make rates and shares "
              "are the EVENT layer's, the only per-class truth there is, and PROVISIONAL.)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"season": a.season, "tolerances": TOL, "reference_tag": a.reference,
         "actual": actual, "runs": res}, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
