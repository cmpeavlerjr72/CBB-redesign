#!/usr/bin/env python
"""
grade_foul_joint_closed_loop_v2.py -- round-8 closed loop (possession_outcome experiments.md
s22.3 as amended by s23). v1's lines, plus:
  * primary 2: the two-team FT-rate correlation -- pooled WITHIN-GAME correlation across seeds of
    home vs away FTA/FGA (the G1/G5 diagnostic's "sim within corr"), target +0.207;
  * V3 restated: G5 total SD ratio and the home/away SCORE correlation (gates.py: row-level sim
    corr vs actual across the same games), each vs its paired floor;
  * H1 / H2 FTA/FGA (tap) as mandatory lines;
  * noise bands: paired game-bootstrap SE (200) of arm - CL0 for the team FT slope, the team SD
    ratio and the FT-rate correlation.
The FIRST tag is CL0.

    .venv/Scripts/python.exe scripts/grade_foul_joint_closed_loop_v2.py fj_R_s25 fj_R8a_s25 ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade_foul_joint_closed_loop_v1 as G1  # noqa: E402

R = G1.R
FT_CORR_TARGET = 0.207
NBOOT = 200


def fga(g, s):
    return g[f"{s}_fga2_rim"] + g[f"{s}_fga2_jump"] + g[f"{s}_fga3"]


def ft_within_corr(g: pd.DataFrame, games=None) -> float:
    x = pd.DataFrame({"game_id": g["game_id"], "h": g["home_fta"] / fga(g, "home").clip(lower=1),
                      "a": g["away_fta"] / fga(g, "away").clip(lower=1)})
    if games is not None:
        x = x.set_index("game_id").loc[games].reset_index()
        x["gb"] = np.repeat(np.arange(len(games)), x.groupby(level=0).size().to_numpy()) \
            if False else x["game_id"]
    dh = x["h"] - x.groupby("game_id")["h"].transform("mean")
    da = x["a"] - x.groupby("game_id")["a"].transform("mean")
    return float((dh * da).sum() / np.sqrt((dh ** 2).sum() * (da ** 2).sum()))


def score_corr(g: pd.DataFrame) -> float:
    return float(np.corrcoef(g["home_pts"].astype(float), g["away_pts"].astype(float))[0, 1])


def per_game_ft_parts(g):
    """Per game: within-seed sums for the correlation, so a game bootstrap is a reweighting."""
    x = pd.DataFrame({"game_id": g["game_id"], "h": g["home_fta"] / fga(g, "home").clip(lower=1),
                      "a": g["away_fta"] / fga(g, "away").clip(lower=1)})
    dh = x["h"] - x.groupby("game_id")["h"].transform("mean")
    da = x["a"] - x.groupby("game_id")["a"].transform("mean")
    return pd.DataFrame({"game_id": x["game_id"], "hh": dh * dh, "aa": da * da, "ha": dh * da}
                        ).groupby("game_id").sum()


def team_parts(g2):
    t = pd.concat([pd.DataFrame({"game_id": g2["game_id"], "team_id": g2[f"{s}_team_id"],
                                 "fta": g2[f"{s}_fta"], "fga": fga(g2, s)}) for s in ("home", "away")])
    return t.groupby(["game_id", "team_id"])[["fta", "fga"]].sum().reset_index()


def team_stats(tp, box_g, prior, season_rate, games):
    """Team slope and SD ratio on a (possibly resampled, with multiplicity) game list."""
    w = pd.Series(games).value_counts()
    t = tp[tp["game_id"].isin(w.index)].copy()
    t["w"] = t["game_id"].map(w)
    sim = t.assign(fta=t["fta"] * t["w"], fga=t["fga"] * t["w"]).groupby("team_id")[["fta", "fga"]].sum()
    b = box_g[box_g["game_id"].isin(w.index)].copy()
    b["w"] = b["game_id"].map(w)
    act = b.assign(fta=b["fta"] * b["w"], fga=b["fga"] * b["w"]).groupby("team_id")[["fta", "fga"]].sum()
    df = pd.DataFrame({"sim": sim["fta"] / sim["fga"], "act": act["fta"] / act["fga"]}).dropna()
    q = df.join(prior.rename("prior"), how="inner")
    q["q"] = pd.qcut(q["prior"], 5, labels=False)
    qq = q.groupby("q")[["sim", "act"]].mean()
    slope = (qq["sim"].iloc[-1] - qq["sim"].iloc[0]) / (qq["act"].iloc[-1] - qq["act"].iloc[0])
    sd = df.join(season_rate.rename("season"), how="inner")
    return float(slope), float(sd["sim"].std() / sd["season"].std())


def main() -> None:
    tags = sys.argv[1:]
    ref = tags[0]
    box = pd.read_parquet("data/raw/hoopr/team_box/team_box_2025.parquet")
    box_g = box.rename(columns={"free_throws_attempted": "fta", "field_goals_attempted": "fga"})[
        ["game_id", "team_id", "fta", "fga"]]
    b24 = pd.read_parquet("data/raw/hoopr/team_box/team_box_2024.parquet")
    p = b24.groupby("team_id").agg(n=("game_id", "size"), fta=("free_throws_attempted", "sum"),
                                   fga=("field_goals_attempted", "sum"))
    prior = (p["fta"] / p["fga"])[p["n"] >= 10]
    s25 = box.groupby("team_id")[["free_throws_attempted", "field_goals_attempted"]].sum()
    season_rate = s25["free_throws_attempted"] / s25["field_goals_attempted"]
    h = box[box["team_home_away"] == "home"].drop_duplicates("game_id").set_index("game_id")
    a = box[box["team_home_away"] == "away"].drop_duplicates("game_id").set_index("game_id")

    base = json.loads(Path("results/foul_joint/closed_loop_grade_v1.json").read_text()) \
        if Path("results/foul_joint/closed_loop_grade_v1.json").exists() else None
    # v1 lines (FTA/FGA, G1, G5 ratios, G9, occupancy, team FT) through v1's own main
    sys.argv = ["x"] + tags
    G1.main()
    v1 = json.loads(Path("results/foul_joint/closed_loop_grade_v1.json").read_text())

    fl_a = pd.read_parquet(R / "po4b_R_s25_floor/games.parquet")
    fl_b = pd.read_parquet(R / "po4b_R_s25/games.parquet")
    games = np.sort(fl_b["game_id"].unique())
    act_corr = float(np.corrcoef(h.loc[h.index.isin(games), "team_score"],
                                 a.loc[a.index.isin(games) & a.index.isin(h.index), "team_score"]
                                 .reindex(h.index[h.index.isin(games)]))[0, 1])
    floors = {"score_corr": abs(score_corr(fl_a) - score_corr(fl_b)),
              "ft_corr": abs(ft_within_corr(fl_a) - ft_within_corr(fl_b))}
    rng = np.random.default_rng(20260930)
    boots = [rng.choice(games, len(games), replace=True) for _ in range(NBOOT)]
    G = {t: pd.read_parquet(R / t / "games.parquet") for t in tags}
    PG = {t: per_game_ft_parts(G[t]) for t in tags}
    TP = {t: team_parts(pd.read_parquet(R / t / "games_v2.parquet")) for t in tags}

    def corr_from(pg, gl):
        s = pg.loc[gl].sum()
        return float(s["ha"] / np.sqrt(s["hh"] * s["aa"]))
    ref_b = [(corr_from(PG[ref], b), *team_stats(TP[ref], box_g, prior, season_rate, b)) for b in boots]
    out = {"floors_extra": floors, "actual_score_corr": act_corr, "ft_corr_target": FT_CORR_TARGET,
           "arms": {}}
    for t in tags:
        L = v1["arms"][t]["lines"]
        fc = corr_from(PG[t], games)
        sl, sdr = team_stats(TP[t], box_g, prior, season_rate, games)
        arm_b = ref_b if t == ref else [(corr_from(PG[t], b), *team_stats(TP[t], box_g, prior, season_rate, b))
                                        for b in boots]
        d = np.array(arm_b) - np.array(ref_b)
        # robust bootstrap SE (1.4826 x MAD): the quintile-span ratio has heavy tails when a
        # resample shrinks the actual span, and a plain SD would inflate the noise band
        def rse(m):
            m = np.asarray(m)
            return 1.4826 * np.median(np.abs(m - np.median(m, axis=0)), axis=0)
        se = rse(d) if t != ref else np.array([np.nan] * 3)
        se_abs = rse(arm_b)
        sc = score_corr(G[t])
        tap = v1["arms"][t]["tap"]
        out["arms"][t] = {
            "fta_fga": L["fta_fga_pooled"],
            "fta_floors_toward": v1["arms"][t]["floors_vs_ref"]["fta_fga_pooled"]["floors_toward"],
            "ft_corr": fc, "ft_corr_se": float(se_abs[0]), "ft_corr_delta_se": float(se[0]),
            "ft_corr_dist_in_se": float((FT_CORR_TARGET - fc) / se_abs[0]),
            "h1": tap.get("fta_fga_by_half", {}).get("H1"), "h2": tap.get("fta_fga_by_half", {}).get("H2"),
            "final2": tap.get("fta_fga_final2"),
            "team_slope": sl, "team_slope_delta_se": float(se[1]),
            "team_sd_ratio": sdr, "team_sd_delta_se": float(se[2]),
            "g5_total": L["g5_ratio_total"],
            "g5_total_floors_toward": v1["arms"][t]["floors_vs_ref"]["g5_ratio_total"]["floors_toward"],
            "g5_margin_floors_toward": v1["arms"][t]["floors_vs_ref"]["g5_ratio_margin"]["floors_toward"],
            "score_corr": sc,
            "score_corr_floors_toward": None,
            "g1_mean_floors_toward": v1["arms"][t]["floors_vs_ref"]["possessions"]["floors_toward"],
            "g1_sd_floors_toward": v1["arms"][t]["floors_vs_ref"]["g1_sd_gap"]["floors_toward"],
            "g9_floors_toward": v1["arms"][t]["floors_vs_ref"]["total_bias"]["floors_toward"],
            "occ_gap": v1["arms"][t]["occupancy_gap_vs_true_all"],
        }
    sc0 = out["arms"][ref]["score_corr"]
    occ0 = out["arms"][ref]["occ_gap"]
    for t, r in out["arms"].items():
        r["score_corr_floors_toward"] = (abs(sc0 - act_corr) - abs(r["score_corr"] - act_corr)) / floors["score_corr"]
        og = r["occ_gap"]
        v1_ok = (max(abs(v) for v in og.values()) < max(abs(v) for v in occ0.values())
                 and all(abs(og[k]) - abs(occ0[k]) <= 0.03 for k in og))
        v2_ok = r["team_slope"] >= out["arms"][ref]["team_slope"] - max(0.05, 2 * (r["team_slope_delta_se"] or 0))
        v3_ok = (r["g5_total_floors_toward"] >= -1 and r["score_corr_floors_toward"] >= -1
                 and min(r["g1_mean_floors_toward"], r["g1_sd_floors_toward"], r["g5_margin_floors_toward"],
                         r["g9_floors_toward"]) >= -2)
        v4_ok = r["team_sd_ratio"] >= out["arms"][ref]["team_sd_ratio"] - 2 * (r["team_sd_delta_se"] or 0)
        r["vetoes"] = {"V1": bool(v1_ok), "V2": bool(v2_ok), "V3": bool(v3_ok), "V4": bool(v4_ok)}
        r["eligible"] = bool(t != ref and all(r["vetoes"].values()))
    Path("results/foul_joint/closed_loop_grade_v2.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nactual score corr {act_corr:.4f}; floors score_corr {floors['score_corr']:.5f} ft_corr {floors['ft_corr']:.5f}")
    for t, r in out["arms"].items():
        print(f"{t:14s} FTA/FGA {r['fta_fga']:.5f} ({r['fta_floors_toward']:+.2f} fl) | FTcorr {r['ft_corr']:+.4f} "
              f"(SE {r['ft_corr_se']:.4f}, dSE {r['ft_corr_delta_se']:.4f}) | H1 {r['h1']:.4f} H2 {r['h2']:.4f} f2 {r['final2']:.4f} | "
              f"slope {r['team_slope']:.3f} (dSE {r['team_slope_delta_se']:.3f}) sdR {r['team_sd_ratio']:.3f} (dSE {r['team_sd_delta_se']:.3f}) | "
              f"G5tot {r['g5_total']:.4f} ({r['g5_total_floors_toward']:+.1f}) scorr {r['score_corr']:.4f} ({r['score_corr_floors_toward']:+.1f}) | "
              f"occmax {100 * max(abs(v) for v in r['occ_gap'].values()):.1f} | {r['vetoes']} eligible={r['eligible']}")


if __name__ == "__main__":
    main()
