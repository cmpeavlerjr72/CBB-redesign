"""grade_clock_r6_loop_v1.py -- one grader for the clock round-6 closed loops (experiments.md s28).

Lane B, 2026-09-30.  Run with CBB_TRUTH=verified_v1 (the grader refuses otherwise).
Runs: results/engine_v0/{clk6_R_s25, clk6_L2_s25, clk6_L2a_s25, clk6_Rf1_s25, clk6_Rf2_s25}
(same 500-game verified sample, engine_v3 inputs, 25 seeds; R/L2/L2a share seeds 0-24).

PRIMARY: engine possessions per team-game minus the like-for-like truth, the
event-layer v4 pbp count (possessions_v4, SIM_GUARDRAILS revision 2026-09-30),
on the sample's pbp-complete games with a two-team layer.
Floors (Decision 12): two seed-offset reseeds of R (Rf1 seeds 1000-1024, Rf2
2000-2024); the floor per line is reported as each |R - Rf|, and max of the three pairwise gaps.
Paired game bootstrap (1,000 draws, games resampled, same draw for every arm)
on arm - R for every line.
Vetoes: possession SD (pooled), G5 margin SD ratio and total SD ratio (both
components; PROVISIONAL at 25 seeds), OT rate vs verified finals.  The
first-half share needs per-half points, which this runner does not write: NOT MEASURED.
Writes results/clock_r6/loop_grade.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as REF  # noqa: E402

if os.environ.get("CBB_TRUTH") != "verified_v1":
    raise SystemExit("set CBB_TRUTH=verified_v1")
RUNS = {"R": "clk6_R_s25", "L2": "clk6_L2_s25", "L2a": "clk6_L2a_s25", "D1": "clk6_D1_s25",
        "Rf1": "clk6_Rf1_s25", "Rf2": "clk6_Rf2_s25"}
RNG = np.random.default_rng(20260930)
NB = 1000


def per_game(g: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["total"] = g["home_pts"] + g["away_pts"]
    g["ot"] = (g["n_periods"] > 2).astype(float)
    a = g.groupby("game_id").agg(poss=("possessions", "mean"), poss_sd=("possessions", "std"),
                                 m_mean=("margin", "mean"), m_sd=("margin", "std"),
                                 t_mean=("total", "mean"), t_sd=("total", "std"), ot=("ot", "mean"),
                                 poss_sq=("possessions", lambda x: float((x ** 2).mean())))
    return a.join(truth, how="inner")


def lines(a: pd.DataFrame, pc: np.ndarray) -> dict:
    ap = a[a.index.isin(pc)]          # keeps bootstrap duplicates
    pooled_var = float(a["poss_sq"].mean() - a["poss"].mean() ** 2)
    rm = float((a["margin"] - a["m_mean"]).std())
    rt = float((a["total"] - a["t_mean"]).std())
    return {
        "primary_count_minus_v4": float((ap["poss"] - ap["cnt_v4"]).mean()),
        "count_minus_box_estimator": float((a["poss"] - a["game_poss"]).mean()),
        "poss_sd_pooled": float(np.sqrt(max(pooled_var, 0.0))),
        "g5_margin_sd_ratio": float(a["m_sd"].mean()) / rm, "g5_margin_sim_sd": float(a["m_sd"].mean()),
        "g5_margin_resid_sd": rm,
        "g5_total_sd_ratio": float(a["t_sd"].mean()) / rt, "g5_total_sim_sd": float(a["t_sd"].mean()),
        "g5_total_resid_sd": rt,
        "ot_rate": float(a["ot"].mean()),
    }


def main():
    truth_games = REF.load_actual_games(2025).set_index("game_id")
    poss_box = REF.load_actual_possessions(2025).set_index("game_id")
    p4 = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2025.parquet")
    nt = p4.groupby("game_id")["offense_team_id"].nunique()
    cnt4 = p4.groupby(["game_id", "offense_team_id"]).size().groupby("game_id").mean()
    u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet").set_index("game_id")
    truth = truth_games[["margin", "total", "n_periods"]].join(poss_box, how="left")
    truth["cnt_v4"] = cnt4
    truth["actual_ot"] = (truth["n_periods"] > 2).astype(float)
    ag = {k: per_game(pd.read_parquet(ROOT / f"results/engine_v0/{v}/games.parquet"), truth)
          for k, v in RUNS.items() if (ROOT / f"results/engine_v0/{v}/games.parquet").exists()}
    common = sorted(set.intersection(*[set(a.index) for a in ag.values()]))
    ag = {k: a.loc[common] for k, a in ag.items()}
    pc = np.array([g for g in common if g in nt.index and nt[g] == 2 and bool(u.at[g, "pbp_complete"])
                   and pd.notna(truth.at[g, "cnt_v4"])])
    rep = {"n_games": len(common), "n_pc": int(len(pc)), "runs": RUNS,
           "truth": {"cnt_v4_pc": float(truth.loc[pc, "cnt_v4"].mean()),
                     "cnt_v4_sd_pc": float(truth.loc[pc, "cnt_v4"].std()),
                     "box_estimator_sd": float(truth.loc[common, "game_poss"].std()),
                     "ot_rate": float(truth.loc[common, "actual_ot"].mean())},
           "lines": {k: lines(a, pc) for k, a in ag.items()}}
    L = rep["lines"]
    keys = list(L["R"].keys())
    floors = {}
    for k in keys:
        fr = [abs(L["R"][k] - L[f][k]) for f in ("Rf1", "Rf2") if f in L]
        pw = fr + ([abs(L["Rf1"][k] - L["Rf2"][k])] if "Rf1" in L and "Rf2" in L else [])
        floors[k] = {"R_vs_Rf": fr, "max_pairwise": max(pw) if pw else None}
    rep["floors"] = floors
    # paired game bootstrap of arm - R
    idx = np.arange(len(common))
    boots = {a: {k: [] for k in keys} for a in ag if a != "R"}
    for _ in range(NB):
        b = RNG.choice(idx, size=len(idx), replace=True)
        gb = [common[i] for i in b]
        pcb = np.array([g for g in gb if g in set(pc)])
        base = lines(ag["R"].loc[gb], pcb)
        for a in boots:
            la = lines(ag[a].loc[gb], pcb)
            for k in keys:
                boots[a][k].append(la[k] - base[k])
    rep["paired_bootstrap"] = {a: {k: {"diff": L[a][k] - L["R"][k], "se": float(np.std(v)),
                                       "lo95": float(np.percentile(v, 2.5)), "hi95": float(np.percentile(v, 97.5))}
                                   for k, v in d.items()} for a, d in boots.items()}
    # ---- first-half share veto from the halftime taps (diag_g1g5_tap_v3.py, 10 seeds per arm)
    taps = {"R": ("R", 0), "Rf1": ("Rf1", 1000), "L2": ("L2", 0), "L2a": ("L2a", 0), "D1": ("D1", 0)}
    half = {}
    tg = {}
    for k, (name, s0) in taps.items():
        f = ROOT / f"results/clock_r6/tap_{name}/games_{s0}.parquet"
        if not f.exists():
            continue
        t = pd.read_parquet(f)
        t = t[t["game_id"].isin(common)]
        run = pd.read_parquet(ROOT / f"results/engine_v0/{RUNS[k]}/games.parquet")
        mm = t.merge(run, on=["game_id", "seed"], suffixes=("_t", "_r"))
        cols = [c for c in run.columns if c not in ("game_id", "seed") and c + "_t" in mm.columns]
        mism = int(sum((mm[c + "_t"] != mm[c + "_r"]).sum() for c in cols))
        reg = t[t["n_periods"] == 2]
        tg[k] = reg
        half[k] = {"tap_rows": int(len(t)), "rows_matched_to_runner": int(len(mm)),
                   "mismatching_cells_vs_runner": mism,
                   "h1_share_regulation": float((reg["h1_home_pts"] + reg["h1_away_pts"]).sum()
                                                / (reg["home_pts"] + reg["away_pts"]).sum())}
    p4pc = p4[p4["game_id"].isin(pc) & (p4["period"] <= 2)]
    half["actual_h1_share_pbp_v4"] = float(p4pc.loc[p4pc["period"] == 1, "points"].sum() / p4pc["points"].sum())
    if "R" in tg:
        def h1(df):
            return float((df["h1_home_pts"] + df["h1_away_pts"]).sum() / (df["home_pts"] + df["away_pts"]).sum())
        gids = np.array(sorted(tg["R"]["game_id"].unique()))
        for a in ("L2", "L2a", "D1", "Rf1"):
            if a not in tg:
                continue
            bs = []
            ga = {g: d for g, d in tg[a].groupby("game_id")}
            gr = {g: d for g, d in tg["R"].groupby("game_id")}
            for _ in range(300):
                b = RNG.choice(gids, size=len(gids), replace=True)
                da = pd.concat([ga[g] for g in b if g in ga]); dr = pd.concat([gr[g] for g in b if g in gr])
                bs.append(h1(da) - h1(dr))
            half[f"{a}_minus_R"] = {"diff": half[a]["h1_share_regulation"] - half["R"]["h1_share_regulation"],
                                    "boot_se": float(np.std(bs))}
    rep["first_half_share"] = half
    out = ROOT / "results/clock_r6/loop_grade.json"
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"lines": L, "floors": floors}, indent=1, default=float))


if __name__ == "__main__":
    main()
