"""diag_ppp_decomp_v1.py -- closed box-channel decomposition of sim-minus-actual points (lane I, 2026-09-30).

DIAGNOSTIC ONLY. Reads a finished engine run (games.parquet, any seed count) and the verified truth;
nothing in src/ is touched, nothing is fitted.

Per team side, pooled over a cell, points = sum over components c in {rim, jump2, three, ft}:
    FG c : N * f * s_c * m_c * pv_c        N = box-estimator possessions (FGA - OREB + TOV + 0.44 FTA),
    FT   : N * f * r * q                   f = FGA/N, s_c = FGA_c/FGA, m_c = make rate, r = FTA/FGA, q = FT%
LMDI-I (log-mean Divisia) splits sim-minus-actual exactly into factor channels (zero residual).
The f channel is split exactly once more with the estimator identity f = (1 + o - t) / (1 + 0.44 r):
    FT-trip denominator part, and the numerator part allocated to TOV (t = TOV/N) and OREB (o = OREB/N),
    o itself split (LMDI) into OREB% (rho) and live boards per possession.
Then box points vs the verified final (actual) and the sim's own box-vs-score identity are separate
closure terms. Everything is per game (both teams), so the sum equals the G9 total bias on these games.

Actual rim/jumper split: hoopR box 2PA/2PM split by the event layer's rim share per team-game
(team_game_shots_v2), pooled event share where a team-game has no event row.

Usage: diag_ppp_decomp_v1.py <run_dir> <out_json> [<ref_run_dir>]
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
os.environ.setdefault("CBB_TRUTH", "verified_v1")
from cbb_sim.eval import reference as R  # noqa: E402

COLS = ["fga3", "fga2_rim", "fga2_jump", "fta", "tov", "oreb", "dreb", "fgm2_rim", "fgm2_jump", "fgm3", "ftm", "pts"]


def sim_team_games(run_dir: Path) -> pd.DataFrame:
    g = pd.read_parquet(run_dir / "games.parquet")
    nseeds = g["seed"].nunique()
    agg = g.drop(columns=["seed"]).groupby("game_id").mean()
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({"game_id": agg.index})
        for c in COLS:
            d[c] = agg[f"{side}_{c}"].to_numpy()
        d["opp_dreb"] = agg[f"{opp}_dreb"].to_numpy()
        d["side"] = side
        d["count_poss"] = agg["possessions"].to_numpy()
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out.attrs["nseeds"] = nseeds
    return out


def actual_team_games() -> pd.DataFrame:
    b = R.load_actual_team_box(2025)
    t = R.load_team_shot_truth(2025)
    t = t[["game_id", "team_id", "ev_fga_rim", "ev_fgm_rim", "ev_fga_jump2", "ev_fgm_jump2"]]
    b = b.merge(t, on=["game_id", "team_id"], how="left")
    pool_a = t["ev_fga_rim"].sum() / (t["ev_fga_rim"].sum() + t["ev_fga_jump2"].sum())
    pool_m = t["ev_fgm_rim"].sum() / (t["ev_fgm_rim"].sum() + t["ev_fgm_jump2"].sum())
    den_a = b["ev_fga_rim"] + b["ev_fga_jump2"]
    den_m = b["ev_fgm_rim"] + b["ev_fgm_jump2"]
    sa = np.where(den_a > 0, b["ev_fga_rim"] / den_a.replace(0, np.nan), pool_a)
    sm = np.where(den_m > 0, b["ev_fgm_rim"] / den_m.replace(0, np.nan), pool_m)
    sa = np.where(np.isnan(sa), pool_a, sa)
    sm = np.where(np.isnan(sm), pool_m, sm)
    fg2a = b["fga"] - b["tpa"]
    fg2m = b["fgm"] - b["tpm"]
    d = pd.DataFrame({
        "game_id": b["game_id"], "team_id": b["team_id"], "side": b["team_home_away"],
        "fga3": b["tpa"], "fga2_rim": fg2a * sa, "fga2_jump": fg2a * (1 - sa),
        "fta": b["fta"], "tov": b["tov"], "oreb": b["oreb"], "dreb": b["dreb"], "opp_dreb": b["opp_dreb"],
        "fgm2_rim": fg2m * sm, "fgm2_jump": fg2m * (1 - sm), "fgm3": b["tpm"], "ftm": b["ftm"],
        "box_pts": b["team_score"],
    })
    d["side"] = d["side"].astype(str)
    return d


def L(a, b):
    a, b = float(a), float(b)
    if abs(a - b) < 1e-12:
        return a
    return (a - b) / (np.log(a) - np.log(b))


def factors(s: pd.Series) -> dict:
    fga = s["fga3"] + s["fga2_rim"] + s["fga2_jump"]
    N = fga - s["oreb"] + s["tov"] + 0.44 * s["fta"]
    return {
        "N": N, "f": fga / N, "fga": fga,
        "s_rim": s["fga2_rim"] / fga, "s_jump": s["fga2_jump"] / fga, "s_three": s["fga3"] / fga,
        "m_rim": s["fgm2_rim"] / s["fga2_rim"], "m_jump": s["fgm2_jump"] / s["fga2_jump"],
        "m_three": s["fgm3"] / s["fga3"], "r": s["fta"] / fga, "q": s["ftm"] / s["fta"],
        "t": s["tov"] / N, "o": s["oreb"] / N, "rho": s["oreb"] / (s["oreb"] + s["opp_dreb"]),
        "b": (s["oreb"] + s["opp_dreb"]) / N,
    }


def decompose(sim: pd.DataFrame, act: pd.DataFrame, n_games: int) -> dict:
    """sim/act: team-side rows of ONE cell. Returns channels in points per game (both teams)."""
    S = sim[COLS[:-1] + ["opp_dreb"]].sum()
    A = act[COLS[:-1] + ["opp_dreb"]].sum()
    fs, fa = factors(S), factors(A)
    comps = {
        "rim": (["N", "f", "s_rim", "m_rim"], 2.0),
        "jump": (["N", "f", "s_jump", "m_jump"], 2.0),
        "three": (["N", "f", "s_three", "m_three"], 3.0),
        "ft": (["N", "f", "r", "q"], 1.0),
    }
    ch = {}
    tot_s = tot_a = 0.0
    for c, (fl, pv) in comps.items():
        Cs = pv * np.prod([fs[k] for k in fl])
        Ca = pv * np.prod([fa[k] for k in fl])
        tot_s += Cs
        tot_a += Ca
        w = L(Cs, Ca)
        for k in fl:
            key = k if k in ("N", "f") else (f"{k}" if k != "r" else "r_ft_pts")
            ch[key] = ch.get(key, 0.0) + w * np.log(fs[k] / fa[k])
    # split f exactly: ln f = ln(1+o-t) - ln(1+0.44 r)
    Wf = ch.pop("f") / np.log(fs["f"] / fa["f"]) if abs(np.log(fs["f"] / fa["f"])) > 1e-15 else 0.0
    num_s, num_a = 1 + fs["o"] - fs["t"], 1 + fa["o"] - fa["t"]
    ch["f_fttrip_denominator"] = -Wf * np.log((1 + 0.44 * fs["r"]) / (1 + 0.44 * fa["r"]))
    lnnum = np.log(num_s / num_a)
    dnum = num_s - num_a
    do, dt = fs["o"] - fa["o"], -(fs["t"] - fa["t"])
    ch["f_tov"] = Wf * lnnum * (dt / dnum) if abs(dnum) > 1e-15 else 0.0
    f_oreb = Wf * lnnum * (do / dnum) if abs(dnum) > 1e-15 else 0.0
    lo = np.log(fs["o"] / fa["o"])
    ch["f_oreb_pct"] = f_oreb * np.log(fs["rho"] / fa["rho"]) / lo if abs(lo) > 1e-15 else 0.0
    ch["f_oreb_boards_per_poss"] = f_oreb * np.log(fs["b"] / fa["b"]) / lo if abs(lo) > 1e-15 else 0.0
    ch = {k: v / n_games for k, v in ch.items()}
    sim_pts_box = tot_s / n_games
    act_pts_box = tot_a / n_games
    sim_pts_score = sim["pts"].sum() / n_games
    act_final = act["final_pts"].sum() / n_games
    out = {
        "n_games": int(n_games), "n_team_sides": int(len(sim)),
        "sim_pts": sim_pts_score, "act_final_pts": act_final, "total_bias": sim_pts_score - act_final,
        "box_gap": sim_pts_box - act_pts_box,
        "channels": ch, "channels_sum": float(sum(ch.values())),
        "closure_sim_score_minus_box": sim_pts_score - sim_pts_box,
        "closure_act_box_minus_final": -(act_final - act_pts_box),
        "levels_sim": {k: float(v) for k, v in fs.items()},
        "levels_act": {k: float(v) for k, v in fa.items()},
        "sim_efg": float((S["fgm2_rim"] + S["fgm2_jump"] + 1.5 * S["fgm3"]) / fs["fga"]),
        "act_efg": float((A["fgm2_rim"] + A["fgm2_jump"] + 1.5 * A["fgm3"]) / fa["fga"]),
    }
    out["residual"] = out["total_bias"] - out["channels_sum"] - out["closure_sim_score_minus_box"] \
        + (act_pts_box - act_final) * 0 - (-(act_final - act_pts_box))
    # total_bias = (sim_score - sim_box) + (sim_box - act_box) + (act_box - act_final)
    out["residual"] = out["total_bias"] - (out["closure_sim_score_minus_box"] + out["channels_sum"]
                                           + (act_pts_box - act_final))
    out["closure_act_box_minus_final"] = act_pts_box - act_final
    return out


def build(run_dir: Path):
    sim = sim_team_games(run_dir)
    act = actual_team_games()
    games = R.load_actual_games(2025)
    tiers = R.team_quality_terciles(games)
    games = games.set_index("game_id")
    act = act[act["game_id"].isin(games.index) & act["game_id"].isin(sim["game_id"])]
    # keep games with both box sides
    ok = act.groupby("game_id")["side"].nunique()
    ok = ok[ok == 2].index
    act = act[act["game_id"].isin(ok)].copy()
    sim = sim[sim["game_id"].isin(ok)].copy()
    act["final_pts"] = np.where(act["side"] == "home", games.loc[act["game_id"], "home_score"].to_numpy(),
                                games.loc[act["game_id"], "away_score"].to_numpy())
    for d in (sim, act):
        gg = games.loc[d["game_id"]]
        d["month"] = gg["month"].to_numpy()
        d["neutral"] = gg["neutral"].to_numpy()
        d["site"] = np.where(d["neutral"] > 0, "neutral", d["side"])
        d["home_team_id"] = gg["home_team_id"].to_numpy()
        d["away_team_id"] = gg["away_team_id"].to_numpy()
        d["team_id"] = np.where(d["side"] == "home", d["home_team_id"], d["away_team_id"])
        d["off_tier"] = d["team_id"].map(tiers["margin"]).astype(str)
        d["home_tier"] = d["home_team_id"].map(tiers["margin"]).astype(str)
    simt = sim.groupby("game_id")["pts"].sum()
    q = pd.qcut(simt, 3, labels=["low", "mid", "high"]).astype(str)
    for d in (sim, act):
        d["pred_total_tercile"] = d["game_id"].map(q).to_numpy()
    return sim, act


def cell_table(sim, act, key):
    res = {}
    for v in sorted(act[key].dropna().unique(), key=str):
        s, a = sim[sim[key] == v], act[act[key] == v]
        n = a["game_id"].nunique()
        if n == 0:
            continue
        r = decompose(s, a, n)
        r["underpowered"] = bool(n < 300)
        res[str(v)] = r
    return res


def main():
    run_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    sim, act = build(run_dir)
    n = act["game_id"].nunique()
    res = {"run": str(run_dir), "nseeds": sim.attrs.get("nseeds"), "overall": decompose(sim, act, n)}
    for key in ("month", "site", "off_tier", "home_tier", "pred_total_tercile"):
        if key == "site":
            # site is a team-side attribute; per game both sides of a neutral game are neutral
            res[key] = {}
            for v in ("home", "away", "neutral"):
                s, a = sim[sim["site"] == v], act[act["site"] == v]
                ng = a["game_id"].nunique()
                r = decompose(s, a, ng)
                res[key][v] = r
            continue
        if key == "home_tier":
            res[key] = cell_table(sim, act, key)
            continue
        res[key] = cell_table(sim, act, key)
    # per game and per team evidence
    sg = sim.groupby("game_id")["pts"].sum()
    ag = act.groupby("game_id")["final_pts"].sum()
    d = (sg - ag).dropna()
    res["per_game"] = {"n": int(len(d)), "mean": float(d.mean()), "median": float(d.median()),
                       "sd": float(d.std()), "p_sim_below": float((d < 0).mean())}
    # per team (offense): points per team-game bias and eFG gap
    st = sim.groupby("team_id").agg(pts=("pts", "mean"), n=("pts", "size"),
                                     fgm=("fgm2_rim", "sum"), fgmj=("fgm2_jump", "sum"), fgm3=("fgm3", "sum"),
                                     a1=("fga2_rim", "sum"), a2=("fga2_jump", "sum"), a3=("fga3", "sum"))
    at = act.groupby("team_id").agg(pts=("final_pts", "mean"),
                                     fgm=("fgm2_rim", "sum"), fgmj=("fgm2_jump", "sum"), fgm3=("fgm3", "sum"),
                                     a1=("fga2_rim", "sum"), a2=("fga2_jump", "sum"), a3=("fga3", "sum"))
    efg = lambda t: (t["fgm"] + t["fgmj"] + 1.5 * t["fgm3"]) / (t["a1"] + t["a2"] + t["a3"])
    tt = pd.DataFrame({"n": st["n"], "dpts": st["pts"] - at["pts"], "defg": efg(st) - efg(at),
                       "act_efg": efg(at), "sim_efg": efg(st)}).dropna()
    tt = tt[tt["n"] >= 10]
    res["per_team_offense"] = {"n_teams": int(len(tt)), "median_dpts": float(tt["dpts"].median()),
                               "share_dpts_neg": float((tt["dpts"] < 0).mean()),
                               "median_defg_pp": float(100 * tt["defg"].median()),
                               "share_defg_neg": float((tt["defg"] < 0).mean()),
                               "note": "per-team cells UNDERPOWERED individually (~30 games)"}
    # responsiveness: teams by actual eFG quintile
    tt["q"] = pd.qcut(tt["act_efg"], 5, labels=False)
    res["efg_by_actual_team_quintile"] = tt.groupby("q")[["act_efg", "sim_efg"]].mean().reset_index().to_dict("records")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    o = res["overall"]
    print(json.dumps({k: o[k] for k in ("n_games", "sim_pts", "act_final_pts", "total_bias", "box_gap",
                                        "channels_sum", "closure_sim_score_minus_box",
                                        "closure_act_box_minus_final", "residual", "sim_efg", "act_efg")}, indent=1))
    print(json.dumps(o["channels"], indent=1))


if __name__ == "__main__":
    main()
