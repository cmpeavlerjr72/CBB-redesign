#!/usr/bin/env python
"""
diag_fg_make_round2_closedloop.py -- the Decision-10 closed-loop gate for the
fg_make round-2 arms (`docs/models/fg_make/experiments.md` section 13.5).

    .venv/Scripts/python.exe scripts/diag_fg_make_round2_closedloop.py \
        --season 2025 --tag-prefix fgmake_r2_

Reads the paired engine runs (same 500 games, same five seeds, same RNG streams
by construction) that `run_engine.py` wrote under `ENGINE_FG_MAKE=round2_<arm>`
and reports, per arm:

  CL1 margin SD          across all (game, seed) rows -- the quantity L23's
                         ablation table reports, so the two are comparable
  CL2 home/away score correlation, over the same raw rows (gate_g5's definition)
  CL3 possessions per game
  CL4 total bias vs the verified finals

each as an ABSOLUTE number against the season truth AND as a delta against the
S-B reference arm (the arm with no margin anywhere, hence no loop by
construction), which is the form the pre-registration gates on.

It also reports eFG% overall and BY SHOT CLASS against
`data/processed/truth/team_game_shots_v1.parquet`, because a state
re-parametrisation has to be judged on the make rate it produces and not only
on the variance it stops producing.

Nothing here adjusts anything; it only reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval import reference as REF  # noqa: E402

ARMS = ("S_A", "S_B", "S_C", "S_D", "S_E", "S_C_s1")
REFERENCE_ARM = "S_B"
TRUTH = Path("data/processed/truth/team_game_shots_v1.parquet")
# pre-registered tolerances, section 13.5 (docs/gates.yaml g5_corr, g1_mean, g9_total_bias)
TOL = {"CL1_margin_sd": 1.00, "CL2_corr": 0.05, "CL3_poss": 1.00, "CL4_total_bias": 1.00}


def read_arm(results: Path, season: int, act: pd.DataFrame,
             truth: pd.DataFrame | None) -> dict:
    raw = pd.read_parquet(results / "games.parquet")
    meta = json.loads((results / "run_meta.json").read_text(encoding="utf-8"))
    raw = raw[raw["game_id"].isin(act["game_id"])]
    raw["margin"] = raw["home_pts"].astype("int64") - raw["away_pts"].astype("int64")
    raw["total"] = raw["home_pts"].astype("int64") + raw["away_pts"].astype("int64")

    g = raw.groupby("game_id")
    pg = pd.DataFrame({
        "margin": g["margin"].mean(), "total": g["total"].mean(),
        "possessions": g["possessions"].mean(),
        "within_sd": g["margin"].std(),
    }).reset_index().merge(act, on="game_id", how="inner")

    out = {
        "n_games": int(len(pg)), "n_rows": int(len(raw)),
        "n_seeds": int(meta.get("n_seeds", 0)),
        "flags": {k: v for k, v in meta.get("adapter_flags", {}).items()
                  if k.startswith("ENGINE_")},
        # CL1 -- SD across every simulation, the L23 ablation's own quantity
        "margin_sd": float(raw["margin"].std()),
        "within_game_margin_sd": float(pg["within_sd"].mean()),
        "total_sd": float(raw["total"].std()),
        "team_pts_sd": float(pd.concat([raw["home_pts"], raw["away_pts"]]).std()),
        # CL2
        "corr_home_away": float(np.corrcoef(raw["home_pts"].astype(float),
                                            raw["away_pts"].astype(float))[0, 1]),
        # CL3 / CL4 and the rest
        "poss_per_game": float(pg["possessions"].mean()),
        "total_mean": float(pg["total"].mean()),
        "total_bias": float(pg["total"].mean() - pg["act_total"].mean()),
        "margin_mean": float(pg["margin"].mean()),
        "margin_bias": float(pg["margin"].mean() - pg["act_margin"].mean()),
        "ppp": float((pg["total"] / (2 * pg["possessions"])).mean()),
        "ot_rate": float((raw["n_periods"] > 2).mean()),
    }

    # --- eFG% overall and by class -----------------------------------------
    fga = {k: float(raw[f"home_fga{k}"].sum() + raw[f"away_fga{k}"].sum())
           for k in ("3", "2_rim", "2_jump")}
    fgm = {k: float(raw[f"home_fgm{k}"].sum() + raw[f"away_fgm{k}"].sum())
           for k in ("3", "2_rim", "2_jump")}
    tot_fga = sum(fga.values())
    tot_fgm = sum(fgm.values())
    out["efg_pct"] = 100.0 * (tot_fgm + 0.5 * fgm["3"]) / tot_fga
    for k, lab in (("2_rim", "rim"), ("2_jump", "jump2"), ("3", "three")):
        out[f"make_pct_{lab}"] = 100.0 * fgm[k] / fga[k] if fga[k] else float("nan")
        out[f"share_{lab}"] = 100.0 * fga[k] / tot_fga
    out["ft_pct"] = 100.0 * float(raw["home_ftm"].sum() + raw["away_ftm"].sum()) \
        / max(float(raw["home_fta"].sum() + raw["away_fta"].sum()), 1.0)

    if truth is not None:
        t = truth[truth["game_id"].isin(pg["game_id"])]
        a_fga = {"rim": t["ev_fga_rim"].sum(), "jump2": t["ev_fga_jump2"].sum(),
                 "three": t["ev_fga_3"].sum()}
        a_fgm = {"rim": t["ev_fgm_rim"].sum(), "jump2": t["ev_fgm_jump2"].sum(),
                 "three": t["ev_fgm_3"].sum()}
        out["actual"] = {
            "efg_pct": 100.0 * float((t["box_fgm"].sum() + 0.5 * t["box_fgm3"].sum())
                                     / t["box_fga"].sum()),
            "n_teams_games": int(len(t)),
            **{f"make_pct_{k}": 100.0 * float(a_fgm[k] / a_fga[k]) for k in a_fgm},
            **{f"share_{k}": 100.0 * float(a_fga[k] / sum(a_fga.values())) for k in a_fga},
            "ft_pct": 100.0 * float(t["box_ftm"].sum() / t["box_fta"].sum()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--results-dir", default="results/engine_v0")
    ap.add_argument("--tag-prefix", default="fgmake_r2_")
    ap.add_argument("--out", default="data/processed/models/fg_make/round2/closed_loop.json")
    a = ap.parse_args()

    act = REF.load_actual_games(a.season)[
        ["game_id", "margin", "total", "home_score", "away_score"]].rename(
        columns={"margin": "act_margin", "total": "act_total"})
    truth = pd.read_parquet(TRUTH) if TRUTH.exists() else None
    if truth is not None:
        truth = truth[truth["season"] == a.season]

    res = {}
    for arm in ARMS:
        p = Path(a.results_dir) / f"{a.tag_prefix}{arm}"
        if not (p / "games.parquet").exists():
            print(f"  {arm}: MISSING {p}", flush=True)
            continue
        res[arm] = read_arm(p, a.season, act, truth)

    ref = res[REFERENCE_ARM]
    games = set()
    for arm, r in res.items():
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

    # actual reference row
    sub = act[act["game_id"].isin(
        pd.read_parquet(Path(a.results_dir) / f"{a.tag_prefix}{REFERENCE_ARM}"
                        / "games.parquet")["game_id"].unique())]
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

    print("=" * 100)
    print(f"fg_make ROUND 2 -- Decision-10 CLOSED-LOOP GATE  (season {a.season}, "
          f"{actual['n_games']} games x {ref['n_seeds']} seeds, paired streams)")
    print(f"reference arm = {REFERENCE_ARM}; flags {ref['flags']}")
    print("=" * 100)
    hdr = (f"{'arm':<6}{'marginSD':>10}{'dSD':>8}{'corr':>8}{'dcorr':>8}"
           f"{'poss':>9}{'dposs':>8}{'totbias':>9}{'dtb':>8}{'PPP':>8}  gate")
    print(hdr)
    for arm in ARMS:
        if arm not in res:
            continue
        r = res[arm]
        print(f"{arm:<6}{r['margin_sd']:>10.3f}{r['d_margin_sd']:>+8.3f}"
              f"{r['corr_home_away']:>8.3f}{r['d_corr']:>+8.3f}"
              f"{r['poss_per_game']:>9.3f}{r['d_poss']:>+8.3f}"
              f"{r['total_bias']:>+9.3f}{r['d_total_bias']:>+8.3f}"
              f"{r['ppp']:>8.4f}  {r['closed_loop']}"
              f" (CL1 {r['CL1']}, CL2 {r['CL2']}, CL3 {r['CL3']}, CL4 {r['CL4']})")
    print(f"{'ACTUAL':<6}{actual['margin_sd']:>10.3f}{'--':>8}"
          f"{actual['corr_home_away']:>8.3f}{'--':>8}"
          f"{actual['poss_per_game']:>9.3f}{'--':>8}{0.0:>+9.3f}{'--':>8}"
          f"{actual['ppp']:>8.4f}")

    print("\neFG% AND MAKE RATE BY SHOT CLASS (engine vs event-layer truth on the same games)")
    print(f"{'arm':<6}{'eFG%':>8}{'rim%':>8}{'jump2%':>9}{'three%':>9}{'FT%':>8}"
          f"{'rimShr':>9}{'jmpShr':>9}{'3Shr':>8}")
    for arm in ARMS:
        if arm not in res:
            continue
        r = res[arm]
        print(f"{arm:<6}{r['efg_pct']:>8.3f}{r['make_pct_rim']:>8.3f}"
              f"{r['make_pct_jump2']:>9.3f}{r['make_pct_three']:>9.3f}{r['ft_pct']:>8.3f}"
              f"{r['share_rim']:>9.3f}{r['share_jump2']:>9.3f}{r['share_three']:>8.3f}")
    if "actual" in ref:
        av = ref["actual"]
        print(f"{'ACTUAL':<6}{av['efg_pct']:>8.3f}{av['make_pct_rim']:>8.3f}"
              f"{av['make_pct_jump2']:>9.3f}{av['make_pct_three']:>9.3f}{av['ft_pct']:>8.3f}"
              f"{av['share_rim']:>9.3f}{av['share_jump2']:>9.3f}{av['share_three']:>8.3f}")
        print("  (eFG% actual is box_fgm/box_fga/box_fgm3; the per-class make rates and "
              "shares are the EVENT layer's, which is the only per-class truth there is "
              "and is PROVISIONAL.)")

    payload = {"season": a.season, "tolerances": TOL, "reference_arm": REFERENCE_ARM,
               "actual": actual, "arms": res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
