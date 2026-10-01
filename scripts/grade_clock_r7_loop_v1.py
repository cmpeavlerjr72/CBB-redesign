"""grade_clock_r7_loop_v1.py -- clock round 7 closed-loop grader (experiments.md section 33).

Lane H, 2026-09-30. Run with CBB_TRUTH=verified_v1 (refuses otherwise).

--scope sample (default): the 500-game verified sample, 25 seeds.
    A2 = laneH_clk7_A2_s25 (seeds 0-24), control L2 = clk6_L2_s25 (lane B, same seeds;
    the L2 path was re-run bit-identical on seeds 0-2 by lane H). Seed draws for the
    floor: lane B's R, Rf1, Rf2 (served clock; 3 draws) and A2f1 (seeds 1000-1024).
--scope full: 5,710 x 200. A2 = laneH_v3full_A2_s200_o0, control v3full_L2_s200_o0,
    seed draws v3full_S0_s200_o0 and v3full_S0f{1..4} (chunks concatenated here).

Lines (arm - L2, each with the paired game bootstrap, 1,000 draws):
  G1 count - v4 count (like-for-like; pbp-complete two-team games), pooled possession SD
  sim team pace responsiveness: team-game sim possessions vs the team's actual v4 count,
      by the team's season-mean as-of off_tempo_rel quintile (slope of sim on actual), and
      by game-prior quintile
  G5 total and margin SD ratio with components
  G9 total and margin bias, MAE and calibration slope (actual on sim mean), MC-corrected slope
  OT rate
Floor per line = max(max pairwise gap among the seed draws, 2 x bootstrap SE).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import reference as REF  # noqa: E402

if os.environ.get("CBB_TRUTH") != "verified_v1":
    raise SystemExit("set CBB_TRUTH=verified_v1")
RNG = np.random.default_rng(20260930)
EV = ROOT / "results/engine_v0"


def load_run(tag) -> pd.DataFrame | None:
    if isinstance(tag, list):
        parts = [load_run(t) for t in tag]
        parts = [p for p in parts if p is not None]
        return pd.concat(parts, ignore_index=True) if parts else None
    f = EV / tag / "games.parquet"
    if f.exists():
        return pd.read_parquet(f)
    chunks = sorted(glob.glob(str(EV / f"{tag}_off*_n25" / "games.parquet")))
    if chunks:
        return pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)
    return None


def per_game(g: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["total"] = g["home_pts"] + g["away_pts"]
    g["ot"] = (g["n_periods"] > 2).astype(float)
    a = g.groupby("game_id").agg(poss=("possessions", "mean"), m_mean=("margin", "mean"), m_sd=("margin", "std"),
                                 t_mean=("total", "mean"), t_sd=("total", "std"), ot=("ot", "mean"),
                                 n_seeds=("seed", "nunique"),
                                 poss_sq=("possessions", lambda x: float((x ** 2).mean())))
    return a.join(truth, how="inner")


def qslope(sim, act, q):
    t = pd.DataFrame({"s": sim, "a": act, "q": q}).groupby("q").mean()
    return float(np.polyfit(t["a"], t["s"], 1)[0]), t


def lines(a: pd.DataFrame, pc: np.ndarray, tg: pd.DataFrame) -> dict:
    ap = a.loc[pc]
    pooled_var = float(a["poss_sq"].mean() - a["poss"].mean() ** 2)
    rm = float((a["margin"] - a["m_mean"]).std())
    rt = float((a["total"] - a["t_mean"]).std())
    sm, im = np.polyfit(a["m_mean"], a["margin"], 1)
    st, it = np.polyfit(a["t_mean"], a["total"], 1)
    var_mc = float((a["m_sd"] ** 2).mean() / a["n_seeds"].mean())
    var_pred = float(a["m_mean"].var())
    var_mct = float((a["t_sd"] ** 2).mean() / a["n_seeds"].mean())
    var_predt = float(a["t_mean"].var())
    t = pd.DataFrame({"game_id": ap.index.to_numpy()}).merge(tg, on="game_id", how="inner")
    t["sim"] = t["game_id"].map(a["poss"].groupby(level=0).first())
    so, tq = qslope(t["sim"], t["cnt_team"], t["tq"])
    gg = ap.dropna(subset=["gq"])
    sg, gqt = qslope(gg["poss"], gg["cnt_v4"], gg["gq"])
    return {
        "g1_count_minus_v4": float((ap["poss"] - ap["cnt_v4"]).mean()),
        "poss_sd_pooled": float(np.sqrt(max(pooled_var, 0.0))),
        "sim_team_pace_slope_teamq": so, "sim_team_pace_slope_gameq": sg,
        "teamq_sim": tq["s"].round(3).tolist(), "teamq_actual": tq["a"].round(3).tolist(),
        "gameq_sim": gqt["s"].round(3).tolist(), "gameq_actual": gqt["a"].round(3).tolist(),
        "g5_total_sd_ratio": float(a["t_sd"].mean()) / rt, "g5_total_sim_sd": float(a["t_sd"].mean()),
        "g5_total_resid_sd": rt,
        "g5_margin_sd_ratio": float(a["m_sd"].mean()) / rm, "g5_margin_sim_sd": float(a["m_sd"].mean()),
        "g5_margin_resid_sd": rm,
        "g9_total_bias": float((a["t_mean"] - a["total"]).mean()),
        "g9_total_mae": float((a["t_mean"] - a["total"]).abs().mean()),
        "g9_total_slope": float(st), "g9_total_slope_mc": float(st * var_predt / max(var_predt - var_mct, 1e-9)),
        "g9_margin_bias": float((a["m_mean"] - a["margin"]).mean()),
        "g9_margin_mae": float((a["m_mean"] - a["margin"]).abs().mean()),
        "g9_margin_slope": float(sm), "g9_margin_slope_mc": float(sm * var_pred / max(var_pred - var_mc, 1e-9)),
        "ot_rate": float(a["ot"].mean()),
    }


SCALAR = None


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--scope", choices=["sample", "full"], default="sample")
    ap_.add_argument("--nboot", type=int, default=1000)
    args = ap_.parse_args()
    if args.scope == "sample":
        runs = {"L2": ["clk6_L2_s25", *[f"laneH_clk7_L2_o{o}_s25" for o in (25, 50, 75)]],
                "A2": ["laneH_clk7_A2_s25", *[f"laneH_clk7_A2_o{o}_s25" for o in (25, 50, 75)]],
                "A2f1": "laneH_clk7_A2f1_s25",
                "R": "clk6_R_s25", "Rf1": "clk6_Rf1_s25", "Rf2": "clk6_Rf2_s25"}
        draws = [("R", "Rf1"), ("R", "Rf2"), ("Rf1", "Rf2"), ("A2", "A2f1")]
    else:
        runs = {"L2": "v3full_L2_s200_o0", "A2": "laneH_v3full_A2_s200_o0", "A2f1": "laneH_v3full_A2f1_s200_o1000",
                "S0": "v3full_S0_s200_o0", **{f"S0f{i}": f"v3full_S0f{i}_s200_o{i}000" for i in range(1, 5)}}
        draws = [(a, b) for i, a in enumerate(["S0", "S0f1", "S0f2", "S0f3", "S0f4"])
                 for b in ["S0", "S0f1", "S0f2", "S0f3", "S0f4"][i + 1:]] + [("A2", "A2f1")]
    truth_games = REF.load_actual_games(2025).set_index("game_id")
    p4 = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2025.parquet",
                         columns=["game_id", "offense_team_id", "period"])
    nt = p4.groupby("game_id")["offense_team_id"].nunique()
    cnt_team = p4.groupby(["game_id", "offense_team_id"]).size().rename("cnt_team").reset_index()
    cnt4 = cnt_team.groupby("game_id")["cnt_team"].mean()
    u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet").set_index("game_id")
    truth = truth_games[["margin", "total", "n_periods"]].copy()
    truth["cnt_v4"] = cnt4
    # team tempo quintiles from the as-of features of the v4 L2 design (2025)
    des = pd.read_parquet(ROOT / "data/processed/models/clock/r6_L2/design_v2.parquet",
                          columns=["season", "game_id", "offense_team_id", "off_tempo_rel", "tempo_prior_game"])
    des = des[des["season"] == 2025]
    tm = des.groupby("offense_team_id")["off_tempo_rel"].mean()
    tq = pd.qcut(tm, 5, labels=False)
    gp = des.groupby("game_id")["tempo_prior_game"].first()
    truth["gq"] = pd.qcut(gp, 5, labels=False).reindex(truth.index)
    tg = cnt_team.copy()
    tg["tq"] = tg["offense_team_id"].map(tq)
    tg = tg.dropna(subset=["tq"])
    raw = {k: load_run(v) for k, v in runs.items()}
    raw = {k: g for k, g in raw.items() if g is not None}
    if "A2" in raw and "L2" in raw:   # pair the arm and control on common seeds
        cs = set(raw["A2"]["seed"]) & set(raw["L2"]["seed"])
        raw["A2"] = raw["A2"][raw["A2"]["seed"].isin(cs)]
        raw["L2"] = raw["L2"][raw["L2"]["seed"].isin(cs)]
    ag = {k: per_game(g, truth) for k, g in raw.items()}
    seeds_used = {k: int(g["seed"].nunique()) for k, g in raw.items()}
    if "A2" not in ag or "L2" not in ag:
        raise SystemExit(f"missing runs: have {sorted(ag)}")
    common = sorted(set.intersection(*[set(a.index) for a in ag.values()]))
    ag = {k: a.loc[common] for k, a in ag.items()}
    pc = np.array([g for g in common if g in nt.index and nt[g] == 2 and bool(u.at[g, "pbp_complete"])
                   and pd.notna(truth.at[g, "cnt_v4"])])
    L = {k: lines(a, pc, tg) for k, a in ag.items()}
    keys = [k for k, v in L["L2"].items() if isinstance(v, float)]
    seedfloor = {k: max([abs(L[a][k] - L[b][k]) for a, b in draws if a in L and b in L] or [np.nan]) for k in keys}
    idx = np.arange(len(common))
    pcs = set(pc)
    boots = {k: [] for k in keys}
    for _ in range(args.nboot):
        b = RNG.choice(idx, size=len(idx), replace=True)
        gb = [common[i] for i in b]
        pcb = np.array([g for g in gb if g in pcs])
        # resampled frames keep duplicate game_ids; .loc[pc] on duplicated index returns all copies
        A = ag["A2"].loc[gb]
        B = ag["L2"].loc[gb]
        la = lines(A, np.unique(pcb), tg)
        lb = lines(B, np.unique(pcb), tg)
        for k in keys:
            boots[k].append(la[k] - lb[k])
    rep = {"scope": args.scope, "runs": {k: runs[k] for k in ag}, "seeds_used": seeds_used,
           "n_games": len(common), "n_pc": int(len(pc)),
           "n_seed_draw_pairs": [f"{a}-{b}" for a, b in draws if a in L and b in L],
           "truth": {"cnt_v4_pc": float(truth.loc[pc, "cnt_v4"].mean()),
                     "cnt_v4_sd_pc": float(truth.loc[pc, "cnt_v4"].std()),
                     "ot_rate": float((truth.loc[common, "n_periods"] > 2).mean())},
           "lines": L, "A2_minus_L2": {}}
    for k in keys:
        se = float(np.std(boots[k]))
        d = L["A2"][k] - L["L2"][k]
        fl = float(np.nanmax([seedfloor[k], 2 * se]))
        rep["A2_minus_L2"][k] = {"diff": d, "boot_se": se, "seed_floor": seedfloor[k], "floor": fl,
                                 "floors_moved": d / fl if fl else None}
    out = ROOT / f"results/clock_r7/loop_grade_{args.scope}.json"
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    for k in keys:
        r = rep["A2_minus_L2"][k]
        print(f"{k:32s} L2 {L['L2'][k]:9.4f}  A2 {L['A2'][k]:9.4f}  diff {r['diff']:+8.4f}  floor {r['floor']:.4f}"
              f"  ({r['floors_moved']:+.1f} floors)")
    for k in ("teamq_sim", "teamq_actual", "gameq_sim", "gameq_actual"):
        print(k, "L2", L["L2"][k], "A2", L["A2"][k])


if __name__ == "__main__":
    main()
