"""grade_clock_r6_v1.py -- grade the clock round-6 closed loop (experiments.md section 28).

Lane B, 2026-09-30.  Reads the three tap_v2 runs (same 500-game subset):
  results/g1g5_diag/tap_r6_R       R,  seeds 0-24
  results/g1g5_diag/tap_r6_L1      L1 (ENGINE_ANDONE_LABEL=training), seeds 0-24 (paired with R)
  results/g1g5_diag/tap_r6_Rfloor  R,  seeds 100-124 (the seed-offset floor)
One grading function scores every arm the same way.

Primary: engine possessions per team-game minus the like-for-like truth (the
phantom-corrected pbp count of `diag_g1_possessions_v2.corrected_table`,
headline classes), on the subset's pbp-complete graded games; with the closed
OT / composition / law / interaction split on its clock-complete games.
Vetoes (arm vs R, judged against |R - Rfloor|): possession SD, G5 margin and
total SD ratios (gate formula, 0-0 finals excluded), first-half points share
(vs the pbp layer's actual), OT rate (vs finals).
Also checks R seeds 0-24 bit-identical to the served 200-seed run.
Writes results/g1g5_diag/clock_r6_grade.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402

spec = importlib.util.spec_from_file_location("g1v2", ROOT / "scripts/diag_g1_possessions_v2.py")
g1v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1v2)

OUT = ROOT / "results/g1g5_diag"
ARMS = {"R": OUT / "tap_r6_R", "L1": OUT / "tap_r6_L1", "Rfloor": OUT / "tap_r6_Rfloor"}
SEED0 = {"R": 0, "L1": 0, "Rfloor": 100}
PREV = g1v2.PREV
TYPES = g1v2.TYPES


def load(arm):
    d = ARMS[arm]
    g = pd.read_parquet(d / f"games_{SEED0[arm]}.parquet")
    t = pd.read_parquet(d / f"poss_{SEED0[arm]}.parquet")
    gid = g.drop_duplicates("gidx").set_index("gidx")["game_id"]
    t["game_id"] = gid.loc[t["gidx"].astype(int)].to_numpy()
    t["type"] = t["prev_end"].astype(int).map(PREV)
    return g, t


def main():
    U = pd.read_parquet(OUT / "unknown_poss_classified.parquet")
    U = g1v2.split_b2(U[U["season"] == 2025])
    p0 = pd.read_parquet(ROOT / "data/processed/possessions_v2/possessions_2025.parquet")
    # PRIMARY truth = the set section 28 pre-registered: classes A, B1, ALL of B2,
    # C1 (+ E merged).  The later B2 duration split is a reported sensitivity.
    PRE_REG = g1v2.PHANTOM + ["B2_dreb_no_open_other"]
    pcor, _ = g1v2.corrected_table(p0, U, PRE_REG)
    pcor2, _ = g1v2.corrected_table(p0, U, g1v2.PHANTOM)
    cnt_split = pcor2.groupby(["game_id", "offense_team_id"]).size().groupby("game_id").mean()
    cntc = pcor.groupby(["game_id", "offense_team_id"]).size().groupby("game_id").mean()
    u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet").set_index("game_id")
    g200 = pd.read_parquet(ROOT / "results/engine_v0/F2_2025_s200_v5b_A_full/games.parquet")
    summary, _ = G.build_grading_frame(g200, 2025)
    summ = summary.set_index("game_id")
    tb = R.load_actual_team_box(2025)
    two = tb.groupby("game_id").size()
    nt = p0.groupby("game_id")["offense_team_id"].nunique()

    arms = {a: load(a) for a in ARMS}
    sub = arms["R"][0]["game_id"].unique()
    pc = np.array([x for x in sub if x in summ.index and x in nt.index and nt[x] == 2
                   and bool(u.at[x, "pbp_complete"]) and two.get(x, 0) == 2])
    cc = np.array([x for x in pc if bool(u.at[x, "clock_complete_reg"])])
    ok00 = summ.index[~((summ["home_score"] == 0) & (summ["away_score"] == 0))]
    gr = np.array([x for x in sub if x in ok00])

    # actual first-half share of points (pbp layer, pc games), and OT rate (finals)
    pa = p0[p0["game_id"].isin(pc)]
    act_h1 = float(pa.loc[pa["period"] == 1, "points"].sum() / pa.loc[pa["period"] <= 2, "points"].sum())
    act_ot = float(summ.loc[gr, "n_periods"].gt(2).mean())
    pr = pcor[pcor["game_id"].isin(cc) & (pcor["period"] <= 2)]
    d_a = float(pr["dur"].mean())
    S_a = float(pr["dur"].sum() / 2.0 / len(cc))
    sh_a = pr["start_reason"].value_counts(normalize=True).reindex(TYPES, fill_value=0)
    du_a = pr.groupby("start_reason")["dur"].mean().reindex(TYPES)

    rep = {"subset": {"games": int(len(sub)), "pc": int(len(pc)), "cc": int(len(cc)), "graded_no00": int(len(gr))},
           "truth": {"cnt_corrected_pc": float(cntc.loc[pc].mean()), "act_h1_share": act_h1, "act_ot_rate": act_ot},
           "arms": {}}
    # bit identity of R vs the served run
    mm = arms["R"][0].merge(g200, on=["game_id", "seed"], suffixes=("_t", "_a"))
    cols = [c for c in g200.columns if c not in ("game_id", "seed")]
    rep["bit_identity_R_vs_served"] = {"rows": int(len(mm)),
                                       "mismatching_cells": int(sum((mm[c + "_t"] != mm[c + "_a"]).sum() for c in cols))}
    for a, (g, t) in arms.items():
        n_s = g["seed"].nunique()
        gp = g[g["game_id"].isin(pc)]
        prim = float(gp.groupby("game_id")["possessions"].mean().loc[pc].mean() - cntc.loc[pc].mean())
        tp = t[t["game_id"].isin(pc)]
        ot_term = float((tp["period"] >= 3).sum() / 2.0 / (len(pc) * n_s)
                        - (pcor[pcor["game_id"].isin(pc)]["period"] >= 3).sum() / 2.0 / len(pc))
        ts = t[t["game_id"].isin(cc) & (t["period"] <= 2)]
        d_s = float(ts["used"].mean())
        K = -S_a / (d_s * d_a)
        sh_s = ts["type"].value_counts(normalize=True).reindex(TYPES, fill_value=0)
        du_s = ts.groupby("type")["used"].mean().reindex(TYPES)
        comp = float(((sh_s - sh_a) * du_a).fillna(0).sum() * K)
        law = float((sh_a * (du_s - du_a)).fillna(0).sum() * K)
        inter = float(((sh_s - sh_a) * (du_s - du_a)).fillna(0).sum() * K)
        slack = (1200.0 - S_a) / d_s
        N_s = len(ts) / 2.0 / (len(cc) * n_s)
        N_a = len(pr) / 2.0 / len(cc)
        reg_pc = float((tp["period"] <= 2).sum() / 2.0 / (len(pc) * n_s)
                       - (pcor[pcor["game_id"].isin(pc)]["period"] <= 2).sum() / 2.0 / len(pc))
        # vetoes
        gg = g[g["game_id"].isin(gr)].copy()
        gg["margin"] = gg["home_pts"] - gg["away_pts"]
        gg["total"] = gg["home_pts"] + gg["away_pts"]
        agg = gg.groupby("game_id").agg(m_mean=("margin", "mean"), m_sd=("margin", "std"),
                                        t_mean=("total", "mean"), t_sd=("total", "std"))
        fin = summ.loc[agg.index]
        ratio_m = float(agg["m_sd"].mean() / (fin["margin"] - agg["m_mean"]).std())
        ratio_t = float(agg["t_sd"].mean() / (fin["total"] - agg["t_mean"]).std())
        h1 = float((g["h1_home_pts"] + g["h1_away_pts"]).sum() / (g["home_pts"] + g["away_pts"]).sum())
        # regulation-only first-half share: second-half = final - h1 - OT points is not separable, so
        # restrict to regulation sims for the share
        reg = g[g["n_periods"] == 2]
        h1_reg = float((reg["h1_home_pts"] + reg["h1_away_pts"]).sum() / (reg["home_pts"] + reg["away_pts"]).sum())
        rep["arms"][a] = {
            "seeds": int(n_s), "primary_count_minus_lfl_truth": prim,
            "split": {"ot": ot_term, "regulation_pc": reg_pc, "subset_pc_to_cc": reg_pc - (N_s - N_a),
                      "tiling_slack": slack, "composition": comp, "law": law, "interaction": inter},
            "closure_primary_eq_ot_plus_regulation": prim - ot_term - reg_pc,
            "by_type": pd.DataFrame({"sim_share": sh_s, "sim_dur": du_s}).reset_index(names="type").to_dict("records"),
            "veto": {"poss_sd": float(g["possessions"].std()), "g5_margin_sd_ratio": ratio_m,
                     "g5_total_sd_ratio": ratio_t, "h1_share_regulation": h1_reg, "h1_share_all": h1,
                     "ot_rate": float(g["n_periods"].gt(2).mean())},
        }
        # exact closure of the regulation split on cc
        rep["arms"][a]["regulation_cc_closure"] = (N_s - N_a) - slack - comp - law - inter
    Rr, L1, Rf = rep["arms"]["R"], rep["arms"]["L1"], rep["arms"]["Rfloor"]
    floor = abs(Rr["primary_count_minus_lfl_truth"] - Rf["primary_count_minus_lfl_truth"])
    move = L1["primary_count_minus_lfl_truth"] - Rr["primary_count_minus_lfl_truth"]
    vf = {k: abs(Rr["veto"][k] - Rf["veto"][k]) for k in Rr["veto"]}
    tgt = {"g5_margin_sd_ratio": 1.0, "g5_total_sd_ratio": 1.0, "h1_share_regulation": act_h1, "ot_rate": act_ot}
    vetoes = {}
    for k in ("poss_sd", "g5_margin_sd_ratio", "g5_total_sd_ratio", "h1_share_regulation", "ot_rate"):
        if k == "poss_sd":
            vetoes[k] = {"R": Rr["veto"][k], "L1": L1["veto"][k], "floor": vf[k],
                         "note": "reported; target is the actual possession SD, read against the G1 SD line"}
            continue
        dist_R = abs(Rr["veto"][k] - tgt[k])
        dist_L = abs(L1["veto"][k] - tgt[k])
        worse = (dist_L - dist_R) > vf[k]
        vetoes[k] = {"R": Rr["veto"][k], "L1": L1["veto"][k], "target": tgt[k], "floor": vf[k],
                     "L1_worse_beyond_floor": bool(worse)}
    rep["sensitivity_truth_B2_split"] = {
        a: float(arms[a][0][arms[a][0]["game_id"].isin(pc)].groupby("game_id")["possessions"].mean().loc[pc].mean()
                 - cnt_split.loc[pc].mean()) for a in ARMS}
    rep["decision_inputs"] = {"floor_primary": floor, "L1_minus_R_primary": move,
                              "L1_moves_toward_0_beyond_floor": bool(
                                  abs(L1["primary_count_minus_lfl_truth"]) < abs(Rr["primary_count_minus_lfl_truth"]) - floor),
                              "vetoes": vetoes}
    (OUT / "clock_r6_grade.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("subset", "truth", "bit_identity_R_vs_served", "sensitivity_truth_B2_split", "decision_inputs")},
                     indent=1, default=float))
    for a in ARMS:
        print(a, json.dumps({k: rep["arms"][a][k] for k in ("primary_count_minus_lfl_truth", "split",
                                                            "regulation_cc_closure", "closure_primary_eq_ot_plus_regulation", "veto")}, default=float))


if __name__ == "__main__":
    main()
