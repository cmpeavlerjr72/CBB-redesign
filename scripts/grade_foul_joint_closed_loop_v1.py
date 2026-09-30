#!/usr/bin/env python
"""
grade_foul_joint_closed_loop_v1.py -- the round-7 paired closed loop, graded
(possession_outcome experiments.md section 20.3). Every arm through the same
function; the arm list is the CLI.

    .venv/Scripts/python.exe scripts/grade_foul_joint_closed_loop_v1.py \
        fj_R_s25 fj_CL1_s25 fj_CL2a_s25 fj_CL2_s25 fj_CL3_s25 fj_CL4_s25 fj_F5e_s25

The FIRST tag is the reference (CL0). Floors: |po4b_R_s25_floor - po4b_R_s25|
per line, the section-12 seed-offset run, reused as in section 17.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/engine_v0")
FTA_TARGET = 0.32955
G1_TARGET = 68.0
BLAB = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-37", "38-40"]
BUCKETS = [0, 5, 10, 15, 20, 25, 30, 35, 38, 40.001]


def fga(g, side):
    return g[f"{side}_fga2_rim"] + g[f"{side}_fga2_jump"] + g[f"{side}_fga3"]


ACT = {}


def g5_g1(g: pd.DataFrame) -> dict:
    """`cbb_sim.eval.gates` G5 SD ratios (mean within-game sim SD / SD(actual - sim
    mean)) and the G1 possessions SD against the box-estimated actual, same games."""
    a = ACT["games"]
    x = g.assign(tot=g["home_pts"] + g["away_pts"], mar=g["home_pts"] - g["away_pts"])
    s = x.groupby("game_id").agg(tm=("tot", "mean"), ts=("tot", "std"), mm=("mar", "mean"),
                                 ms=("mar", "std")).join(a, how="inner")
    act_sd = float(a.loc[a.index.isin(s.index), "act_poss"].std())
    return {"g5_ratio_margin": float(s["ms"].mean() / (s["act_mar"] - s["mm"]).std()),
            "g5_ratio_total": float(s["ts"].mean() / (s["act_tot"] - s["tm"]).std()),
            "g1_sd_gap": float(g["possessions"].std()) - act_sd, "g1_sd_act_box": act_sd,
            "g1_mean_act_box": float(a.loc[a.index.isin(s.index), "act_poss"].mean())}


def lines(g: pd.DataFrame, act_tot: pd.Series) -> dict:
    fta = g["home_fta"] + g["away_fta"]
    fg = fga(g, "home") + fga(g, "away")
    tot = g["home_pts"] + g["away_pts"]
    mar = g["home_pts"] - g["away_pts"]
    gm = g.assign(tot=tot, mar=mar).groupby("game_id")[["tot", "mar"]].mean()
    return {**g5_g1(g),
        "fta_fga_pooled": float(fta.sum() / fg.sum()),
        "fta_per_game": float(fta.mean()), "fga_per_game": float(fg.mean()),
        "possessions": float(g["possessions"].mean()),
        "possessions_sd": float(g["possessions"].std()),
        "total_mean": float(tot.mean()),
        "total_bias": float(gm["tot"].mean() - act_tot.reindex(gm.index).mean()),
        "total_sd": float(gm["tot"].std()), "margin_sd": float(gm["mar"].std()),
        # row-level (game x seed) SDs, reported beside the per-game-mean SDs above
        "total_sd_rows": float(tot.std()), "margin_sd_rows": float(mar.std()),
        "oreb_pct": float((g["home_oreb"] + g["away_oreb"]).sum()
                          / (g["home_oreb"] + g["away_oreb"] + g["home_dreb"] + g["away_dreb"]).sum()),
    }


def team_ft(g2: pd.DataFrame, box: pd.DataFrame, prior: pd.Series, season_rate: pd.Series) -> dict:
    t = pd.concat([pd.DataFrame({"game_id": g2["game_id"], "team_id": g2[f"{s}_team_id"],
                                 "fta": g2[f"{s}_fta"], "fga": fga(g2, s)}) for s in ("home", "away")])
    sim = t.groupby("team_id")[["fta", "fga"]].sum()
    sim["rate"] = sim["fta"] / sim["fga"]
    games = set(g2["game_id"].unique())
    b = box[box["game_id"].isin(games)].groupby("team_id")[["free_throws_attempted", "field_goals_attempted"]].sum()
    b["rate"] = b["free_throws_attempted"] / b["field_goals_attempted"]
    df = sim[["rate"]].rename(columns={"rate": "sim"}).join(b["rate"].rename("act"), how="inner")
    q = df.join(prior.rename("prior"), how="inner")
    q["q"] = pd.qcut(q["prior"], 5, labels=False)
    qq = q.groupby("q")[["sim", "act"]].mean()
    span_s, span_a = qq["sim"].iloc[-1] - qq["sim"].iloc[0], qq["act"].iloc[-1] - qq["act"].iloc[0]
    sd = df.join(season_rate.rename("season"), how="inner")
    return {"n_teams": int(len(q)), "slope_ratio": float(span_s / span_a),
            "gap_q1_pp": float(100 * (qq["sim"].iloc[0] - qq["act"].iloc[0])),
            "gap_q5_pp": float(100 * (qq["sim"].iloc[-1] - qq["act"].iloc[-1])),
            "sd_ratio_vs_season": float(sd["sim"].std() / sd["season"].std()),
            "sd_sim_pp": float(100 * sd["sim"].std()), "sd_season_pp": float(100 * sd["season"].std())}


def actual_true_occupancy(game_ids=None) -> pd.Series:
    d = pd.read_parquet("data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet",
                        columns=["game_id", "season", "period", "start_clock", "def_team_fouls_true"])
    d = d[(d["season"] == 2025) & (d["period"] <= 2)]
    if game_ids is not None:
        d = d[d["game_id"].isin(game_ids)]
    gm = (d["period"] - 1) * 20 + (1200 - d["start_clock"]) / 60.0
    b = pd.cut(gm, BUCKETS, right=False, labels=BLAB)
    return (d["def_team_fouls_true"] >= 6).groupby(b, observed=True).mean()


def tap_lines(tag: str) -> dict:
    t = pd.read_parquet(R / tag / "poss_tap.parquet")
    reg = t["period"] <= 2
    gm = (t["period"] - 1) * 20 + (1200 - t["sec"]) / 60.0
    t["b"] = pd.cut(gm.where(reg), BUCKETS, right=False, labels=BLAB)
    occ = (t["def_fouls"] >= t["bonus_thr"]).groupby(t["b"], observed=True).mean()
    sdc = t.groupby("b", observed=True)["def_fouls"].std()
    half = np.where(t["period"] >= 3, "OT", np.where(t["period"] == 1, "H1", "H2"))
    ff = t.groupby(half)[["fta", "fga"]].sum()
    f2 = (t["period"] == 2) & (t["sec"] <= 120)
    t["tb"] = t["def_fouls"].clip(0, 10)
    trips = (t["n_shoot_trip"] + t["n_bonus_trip"]).groupby([half, t["tb"]]).mean()
    of = t["off_foul"] if "off_foul" in t else 0
    tot = t["trip_fouls"] + t["silent"] + of
    return {"occupancy": occ.to_dict(), "count_sd": sdc.to_dict(),
            "fta_fga_by_half": (ff["fta"] / ff["fga"]).to_dict(),
            "fta_fga_final2": float(t.loc[f2, "fta"].sum() / t.loc[f2, "fga"].sum()),
            "fta_fga_rest": float(t.loc[~f2, "fta"].sum() / t.loc[~f2, "fga"].sum()),
            "fouls_per_poss": float(tot.mean()), "nontrip_per_poss": float((tot - t["trip_fouls"]).mean()),
            "bonus_trips_per_poss": float(t["n_bonus_trip"].mean()),
            "shoot_trips_per_poss": float(t["n_shoot_trip"].mean()),
            "trips_by_half_count": {f"{k[0]}|{k[1]}": float(v) for k, v in trips.items()}}


def main() -> None:
    tags = sys.argv[1:]
    ref = tags[0]
    box = pd.read_parquet("data/raw/hoopr/team_box/team_box_2025.parquet")
    act_tot = box.groupby("game_id")["team_score"].sum()
    h = box[box["team_home_away"] == "home"].drop_duplicates("game_id").set_index("game_id")
    w = box[box["team_home_away"] == "away"].drop_duplicates("game_id").set_index("game_id")

    def est(t):
        return (t["field_goals_attempted"] - t["offensive_rebounds"] + t["turnovers"]
                + 0.475 * t["free_throws_attempted"])
    ACT["games"] = pd.DataFrame({"act_tot": h["team_score"] + w["team_score"],
                                 "act_mar": h["team_score"] - w["team_score"],
                                 "act_poss": 0.5 * (est(h) + est(w))}).dropna()
    b24 = pd.read_parquet("data/raw/hoopr/team_box/team_box_2024.parquet")
    p = b24.groupby("team_id").agg(n=("game_id", "size"), fta=("free_throws_attempted", "sum"),
                                   fga=("field_goals_attempted", "sum"))
    prior = (p["fta"] / p["fga"])[p["n"] >= 10]
    s25 = box.groupby("team_id")[["free_throws_attempted", "field_goals_attempted"]].sum()
    season_rate = s25["free_throws_attempted"] / s25["field_goals_attempted"]

    fl_a = lines(pd.read_parquet(R / "po4b_R_s25_floor/games.parquet"), act_tot)
    fl_b = lines(pd.read_parquet(R / "po4b_R_s25/games.parquet"), act_tot)
    floors = {k: abs(fl_a[k] - fl_b[k]) for k in fl_a}
    sub_ids = pd.read_parquet(R / ref / "games.parquet", columns=["game_id"])["game_id"].unique()
    occ_all = actual_true_occupancy()
    occ_sub = actual_true_occupancy(sub_ids)
    out = {"floors": floors, "actual_true_occupancy_all": occ_all.to_dict(),
           "actual_true_occupancy_sub": occ_sub.to_dict(), "arms": {}}
    targets = {"fta_fga_pooled": FTA_TARGET, "possessions": G1_TARGET, "total_bias": 0.0,
               "g5_ratio_margin": 1.0, "g5_ratio_total": 1.0, "g1_sd_gap": 0.0}
    for tag in tags:
        g = pd.read_parquet(R / tag / "games.parquet")
        L = lines(g, act_tot)
        g2p = R / tag / "games_v2.parquet"
        tf = team_ft(pd.read_parquet(g2p), box, prior, season_rate) if g2p.exists() else {"status": "no games_v2"}
        tl = tap_lines(tag) if (R / tag / "poss_tap.parquet").exists() else {}
        occ_gap = {}
        if tl:
            occ_gap = {k: float(tl["occupancy"].get(k, np.nan) - occ_all.get(k, np.nan)) for k in BLAB}
        out["arms"][tag] = {"lines": L, "team_ft": tf, "tap": tl, "occupancy_gap_vs_true_all": occ_gap}
    R0 = out["arms"][ref]["lines"]
    for tag, a in out["arms"].items():
        L = a["lines"]
        a["floors_vs_ref"] = {}
        for k in L:
            fl = floors.get(k) or np.nan
            if k in targets:
                d0, d1 = abs(R0[k] - targets[k]), abs(L[k] - targets[k])
                a["floors_vs_ref"][k] = {"value": L[k], "dist_ref": d0, "dist_arm": d1,
                                         "floors_toward": (d0 - d1) / fl}
            else:
                a["floors_vs_ref"][k] = {"value": L[k], "delta": L[k] - R0[k], "floors": (L[k] - R0[k]) / fl}
    (Path("results/foul_joint") / "closed_loop_grade_v1.json").write_text(json.dumps(out, indent=1, default=float))
    # printed table
    print(f"floors: fta_fga {floors['fta_fga_pooled']:.6f}  poss {floors['possessions']:.4f}  "
          f"total_sd {floors['total_sd']:.4f}  margin_sd {floors['margin_sd']:.4f}  bias {floors['total_bias']:.4f}")
    print("actual TRUE occupancy (all 2025):", {k: round(v, 3) for k, v in occ_all.items()})
    for tag, a in out["arms"].items():
        L, F = a["lines"], a["floors_vs_ref"]
        print(f"\n{tag}: FTA/FGA {L['fta_fga_pooled']:.5f} ({F['fta_fga_pooled']['floors_toward']:+.2f} floors toward)  "
              f"poss {L['possessions']:.3f} ({F['possessions']['floors_toward']:+.2f})  poss_sd {L['possessions_sd']:.3f} "
              f"({F['possessions_sd']['floors']:+.2f})  total_bias {L['total_bias']:+.3f} ({F['total_bias']['floors_toward']:+.2f})  "
              f"total_sd {L['total_sd']:.3f} ({F['total_sd']['floors']:+.2f})  margin_sd {L['margin_sd']:.3f} "
              f"({F['margin_sd']['floors']:+.2f})  oreb {L['oreb_pct']:.4f}  rowSD tot {L['total_sd_rows']:.3f} mar {L['margin_sd_rows']:.3f}")
        print(f"   G5 ratio margin {L['g5_ratio_margin']:.4f} ({F['g5_ratio_margin']['floors_toward']:+.2f} floors toward)  "
              f"total {L['g5_ratio_total']:.4f} ({F['g5_ratio_total']['floors_toward']:+.2f})  G1 SD gap {L['g1_sd_gap']:+.3f} "
              f"(box SD {L['g1_sd_act_box']:.3f}; {F['g1_sd_gap']['floors_toward']:+.2f} toward)  G1 box mean {L['g1_mean_act_box']:.2f}")
        tf = a["team_ft"]
        if "slope_ratio" in tf:
            print(f"   team FT: slope {tf['slope_ratio']:.3f} gapQ1 {tf['gap_q1_pp']:+.2f} gapQ5 {tf['gap_q5_pp']:+.2f} "
                  f"sd_ratio {tf['sd_ratio_vs_season']:.3f}")
        if a["tap"]:
            og = a["occupancy_gap_vs_true_all"]
            print("   occ gap pp:", " ".join(f"{k}:{100 * v:+.1f}" for k, v in og.items()),
                  f" max|gap| {100 * max(abs(v) for v in og.values()):.2f}")
            t = a["tap"]
            print(f"   FTA/FGA H1 {t['fta_fga_by_half'].get('H1', np.nan):.4f} H2 {t['fta_fga_by_half'].get('H2', np.nan):.4f} "
                  f"final2 {t['fta_fga_final2']:.4f} rest {t['fta_fga_rest']:.4f}  fouls/poss {t['fouls_per_poss']:.4f} "
                  f"nontrip {t['nontrip_per_poss']:.4f} bonus trips {t['bonus_trips_per_poss']:.4f} shoot {t['shoot_trips_per_poss']:.4f}")
            print("   count SD by minute:", " ".join(f"{k}:{v:.2f}" for k, v in t["count_sd"].items()))


if __name__ == "__main__":
    main()
