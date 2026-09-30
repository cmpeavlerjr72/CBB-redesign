"""diag_g9_g6_margin_v1.py -- Lane F (2026-09-30): trace the G9 calibration-slope
miss (0.910) and the G6 neutral-site home-margin miss (+2.105 vs +3.288) of the
served v5b stack on fold 2 (2024-25) to sub-model channels.

DIAGNOSTIC ONLY. Reads existing sim output and truth; writes nothing under
`src/cbb_sim/`, fits no model that is served, changes no default.

Method (exact additive margin identity, applied identically to sim and actual):

  per team-side:   P   = FGA - OREB + TOV + 0.44*FTA        (box possession formula)
                   t   = TOV/P,  rp = FTA/P,  f = FTM/FTA
                   rho = OREB/(OREB+oppDREB),  m = (OREB+oppDREB)/P,  o = rho*m
                   s_k = FGA_k/FGA, p_k = FGM_k/FGA_k, k in {rim, jump2, 3}, v=(2,2,3)
                   A   = 1 + o - t - 0.44*rp  (= FGA/P), Q = sum v_k s_k p_k (pts/FGA)
                   PPP = A*Q + rp*f  == PTS/P exactly
  per game (Bennet midpoint decomposition, exact for these bilinear forms):
     margin = Pbar*dPPP + PPPbar*dP
     dPPP   = -Qbar dt + Qbar mbar drho + Qbar rhobar dm + (fbar-0.44Qbar) drp
              + rpbar df + Abar sum v_k pbar_k ds_k + Abar sum v_k sbar_k dp_k
     channel_c (points) = P_ref*dPPP_c ; pace = (Pbar-P_ref)*dPPP ; possdiff = PPPbar*dP
  Sim: computed per (game, seed) then averaged over seeds -> X_c (sums to sim_margin_mean).
  Actual: hoopR team box, 2PT split rim/jump by the event-layer (CBBD) shares
  -> Y_c; plus `final_vs_box` = verified final margin - box identity margin.

  Slope identity: 1 - slope(Y on X) = sum_c cov(X_c - Y_c, X) / var(X)   (closes exactly).

Usage:
    .venv/Scripts/python.exe scripts/diag_g9_g6_margin_v1.py --part build --part analyse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402

SEASON = 2025
RUN_A = Path("results/engine_v0/F2_2025_s200_v5b_A_full")
RUN_B = Path("results/engine_v0/F2_2025_s200_v5b_B_full")
OUT = Path("results/g9g6_diag")
LINES = Path("data/processed/lines/lines_close_v1.parquet")
POWER = {"ACC", "Big Ten", "Big 12", "SEC", "Big East"}
KS = ("rim", "jump", "3")
V = {"rim": 2.0, "jump": 2.0, "3": 3.0}
CHANNELS = ["tov", "oreb", "reb_chances", "ft_trips", "ft_pct",
            "mix_rim", "mix_jump", "mix_3", "make_rim", "make_jump", "make_3",
            "pace", "possdiff"]
CH_MODEL = {
    "tov": "possession_outcome (event round2_s1)",
    "oreb": "rebound (s1_weekly)",
    "reb_chances": "derived: miss volume (fg_make + PO), not a model",
    "ft_trips": "possession_outcome foul-trip class + foul accrual",
    "ft_pct": "free_throw (s1_conf_aligned)",
    "mix_rim": "possession_outcome shot class", "mix_jump": "possession_outcome shot class",
    "mix_3": "possession_outcome shot class (+ fg3 decision8 routing)",
    "make_rim": "fg_make round4_B1", "make_jump": "fg_make round4_B1",
    "make_3": "fg_make (FGA_3 decision8)",
    "pace": "clock v5b_glat_pmean", "possdiff": "clock / end-of-period possession parity",
}


def sdiv(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=b > 0)
    return out


def side_rates(c: dict) -> dict:
    """c: dict of count arrays for one side (fga_rim, fga_jump, fga_3, fgm_*, fta, ftm, tov, oreb, oppdreb)."""
    fga = c["fga_rim"] + c["fga_jump"] + c["fga_3"]
    P = fga - c["oreb"] + c["tov"] + 0.44 * c["fta"]
    r = {"P": P, "t": sdiv(c["tov"], P), "rp": sdiv(c["fta"], P), "f": sdiv(c["ftm"], c["fta"]),
         "rho": sdiv(c["oreb"], c["oreb"] + c["oppdreb"]), "m": sdiv(c["oreb"] + c["oppdreb"], P)}
    for k in KS:
        r[f"s_{k}"] = sdiv(c[f"fga_{k}"], fga)
        r[f"p_{k}"] = sdiv(c[f"fgm_{k}"], c[f"fga_{k}"])
    r["A"] = 1 + r["rho"] * r["m"] - r["t"] - 0.44 * r["rp"]
    r["Q"] = sum(V[k] * r[f"s_{k}"] * r[f"p_{k}"] for k in KS)
    r["PPP"] = r["A"] * r["Q"] + r["rp"] * r["f"]
    r["pts_identity"] = r["P"] * r["PPP"]
    return r


def decompose(h: dict, a: dict, P_ref: float) -> dict:
    bar = lambda k: 0.5 * (h[k] + a[k])  # noqa: E731
    d = lambda k: h[k] - a[k]  # noqa: E731
    Qb, Ab, mb, rhob, fb, rpb = bar("Q"), bar("A"), bar("m"), bar("rho"), bar("f"), bar("rp")
    ppp = {
        "tov": -Qb * d("t"),
        "oreb": Qb * mb * d("rho"),
        "reb_chances": Qb * rhob * d("m"),
        "ft_trips": (fb - 0.44 * Qb) * d("rp"),
        "ft_pct": rpb * d("f"),
    }
    for k in KS:
        ppp[f"mix_{k}"] = Ab * V[k] * bar(f"p_{k}") * d(f"s_{k}")
        ppp[f"make_{k}"] = Ab * V[k] * bar(f"s_{k}") * d(f"p_{k}")
    dPPP = d("PPP")
    Pb = bar("P")
    out = {c: P_ref * v for c, v in ppp.items()}
    out["pace"] = (Pb - P_ref) * dPPP
    out["possdiff"] = bar("PPP") * d("P")
    out["_margin_identity"] = h["pts_identity"] - a["pts_identity"]
    out["_bennet_gap"] = dPPP - sum(ppp.values())
    return out, {f"{s}_{k}": side[k] for s, side in (("h", h), ("a", a))
                 for k in ("P", "t", "rp", "f", "rho", "m", "s_rim", "s_jump", "s_3",
                           "p_rim", "p_jump", "p_3", "PPP")}


# --------------------------------------------------------------------------- build
def sim_counts(g: pd.DataFrame, side: str) -> dict:
    o = "away" if side == "home" else "home"
    return {"fga_rim": g[f"{side}_fga2_rim"].to_numpy(float), "fga_jump": g[f"{side}_fga2_jump"].to_numpy(float),
            "fga_3": g[f"{side}_fga3"].to_numpy(float), "fgm_rim": g[f"{side}_fgm2_rim"].to_numpy(float),
            "fgm_jump": g[f"{side}_fgm2_jump"].to_numpy(float), "fgm_3": g[f"{side}_fgm3"].to_numpy(float),
            "fta": g[f"{side}_fta"].to_numpy(float), "ftm": g[f"{side}_ftm"].to_numpy(float),
            "tov": g[f"{side}_tov"].to_numpy(float), "oreb": g[f"{side}_oreb"].to_numpy(float),
            "oppdreb": g[f"{o}_dreb"].to_numpy(float)}


def build_sim(run: Path, P_ref: float, tag: str) -> pd.DataFrame:
    g = pd.read_parquet(run / "games.parquet")
    h, a = side_rates(sim_counts(g, "home")), side_rates(sim_counts(g, "away"))
    ch, _ = decompose(h, a, P_ref)
    df = pd.DataFrame({"game_id": g["game_id"].to_numpy()})
    for c in CHANNELS + ["_margin_identity", "_bennet_gap"]:
        df[c] = ch[c]
    df["margin"] = (g["home_pts"].astype(int) - g["away_pts"].astype(int)).to_numpy()
    chk = float(np.abs(df["margin"] - df[CHANNELS].sum(axis=1)).max())
    print(f"[{tag}] max |margin - sum channels| per seed-row = {chk:.3e}; "
          f"max |pts identity gap| = {float(np.abs(df['margin'] - df['_margin_identity']).max()):.3e}")
    X = df.groupby("game_id").mean()
    # expected-rate predictions per side: seed-summed counts -> rates (for team-level work)
    cnt = {}
    for side in ("home", "away"):
        c = sim_counts(g, side)
        cdf = pd.DataFrame(c); cdf["game_id"] = g["game_id"].to_numpy()
        s = cdf.groupby("game_id").sum()
        r = side_rates({k: s[k].to_numpy() for k in c})
        for k in ("t", "rp", "f", "rho", "m", "s_rim", "s_jump", "s_3", "p_rim", "p_jump", "p_3", "PPP"):
            X[f"{side[0]}_{k}"] = r[k]
        X[f"{side[0]}_P"] = r["P"] / g.groupby("game_id").size().reindex(s.index).to_numpy()
        for k in ("fta", "fga_rim", "fga_jump", "fga_3", "oreb", "oppdreb"):
            X[f"{side[0]}_n_{k}"] = s[k].to_numpy() / g.groupby("game_id").size().reindex(s.index).to_numpy()
        cnt[side] = s
    X["n_seeds"] = g.groupby("game_id").size()
    return X.reset_index()


def build_actual(P_ref: float | None = None):
    box = R.load_actual_team_box(SEASON)
    shots = R.load_team_shot_truth(SEASON)
    b = box.merge(shots[["game_id", "team_id", "ev_fga_rim", "ev_fgm_rim", "ev_fga_jump2", "ev_fgm_jump2"]],
                  on=["game_id", "team_id"], how="left")
    fga2 = (b["fga"] - b["tpa"]).astype(float); fgm2 = (b["fgm"] - b["tpm"]).astype(float)
    ev_a = b["ev_fga_rim"] + b["ev_fga_jump2"]; ev_m = b["ev_fgm_rim"] + b["ev_fgm_jump2"]
    lg_sa = float(b["ev_fga_rim"].sum() / ev_a.sum()); lg_sm = float(b["ev_fgm_rim"].sum() / ev_m.sum())
    share_a = np.where(ev_a > 0, b["ev_fga_rim"] / ev_a.where(ev_a > 0), lg_sa)
    share_m = np.where(ev_m > 0, b["ev_fgm_rim"] / ev_m.where(ev_m > 0), lg_sm)
    b["fga_rim"] = fga2 * share_a; b["fga_jump"] = fga2 - b["fga_rim"]
    b["fgm_rim"] = np.minimum(fgm2 * share_m, b["fga_rim"]); b["fgm_jump"] = fgm2 - b["fgm_rim"]
    n_missing_ev = int(b["ev_fga_rim"].isna().sum())
    home = b[b["team_home_away"] == "home"].set_index("game_id")
    away = b[b["team_home_away"] == "away"].set_index("game_id")
    ids = home.index.intersection(away.index)
    home, away = home.loc[ids], away.loc[ids]

    def cnts(x, opp):
        return {"fga_rim": x["fga_rim"].to_numpy(float), "fga_jump": x["fga_jump"].to_numpy(float),
                "fga_3": x["tpa"].to_numpy(float), "fgm_rim": x["fgm_rim"].to_numpy(float),
                "fgm_jump": x["fgm_jump"].to_numpy(float), "fgm_3": x["tpm"].to_numpy(float),
                "fta": x["fta"].to_numpy(float), "ftm": x["ftm"].to_numpy(float),
                "tov": x["tov"].to_numpy(float), "oreb": x["oreb"].to_numpy(float),
                "oppdreb": opp["dreb"].to_numpy(float)}
    h, a = side_rates(cnts(home, away)), side_rates(cnts(away, home))
    if P_ref is None:
        P_ref = float(np.mean(np.concatenate([h["P"], a["P"]])))
    ch, rates = decompose(h, a, P_ref)
    Y = pd.DataFrame({"game_id": ids.to_numpy()})
    for c in CHANNELS + ["_margin_identity", "_bennet_gap"]:
        Y[c] = ch[c]
    for k, v in rates.items():
        Y[k] = v
    for side, x, opp in (("h", home, away), ("a", away, home)):
        Y[f"{side}_n_fta"] = x["fta"].to_numpy(float)
        Y[f"{side}_n_fga_rim"] = x["fga_rim"].to_numpy(float)
        Y[f"{side}_n_fga_jump"] = x["fga_jump"].to_numpy(float)
        Y[f"{side}_n_fga_3"] = x["tpa"].to_numpy(float)
        Y[f"{side}_n_oreb"] = x["oreb"].to_numpy(float)
        Y[f"{side}_n_oppdreb"] = opp["dreb"].to_numpy(float)
    Y["box_margin"] = (home["team_score"].to_numpy(float) - away["team_score"].to_numpy(float))
    return Y, P_ref, n_missing_ev


def part_build():
    OUT.mkdir(parents=True, exist_ok=True)
    Y, P_ref, n_miss = build_actual()
    print(f"P_ref (actual league mean team possessions) = {P_ref:.4f}; team-games missing event-layer rim split: {n_miss}")
    print(f"actual: max |box_margin - identity| = {float(np.abs(Y['box_margin'] - Y['_margin_identity']).max()):.3e}")
    Y.to_parquet(OUT / "actual_channels_v1.parquet", index=False)
    for run, tag in ((RUN_A, "A"), (RUN_B, "B")):
        X = build_sim(run, P_ref, tag)
        X.to_parquet(OUT / f"sim_channels_{tag}_v1.parquet", index=False)
    json.dump({"P_ref": P_ref, "n_missing_ev": n_miss}, open(OUT / "build_meta_v1.json", "w"))


# --------------------------------------------------------------------------- analyse
def slope(y, x):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))


def load_frame(tag="A"):
    meta = json.load(open(OUT / "build_meta_v1.json"))
    X = pd.read_parquet(OUT / f"sim_channels_{tag}_v1.parquet")
    Y = pd.read_parquet(OUT / "actual_channels_v1.parquet")
    games = pd.read_parquet(RUN_A / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts", "possessions", "n_periods"])
    summ, _ = G.build_grading_frame(games if tag == "A" else pd.read_parquet(RUN_B / "games.parquet", columns=games.columns.tolist()), SEASON)
    X = X.add_prefix("X_").rename(columns={"X_game_id": "game_id"})
    Y = Y.add_prefix("Y_").rename(columns={"Y_game_id": "game_id"})
    df = summ.merge(X, on="game_id", how="left").merge(Y, on="game_id", how="left")
    # conference tier
    cg = pd.read_parquet("data/raw/cbbd/games_2025.parquet", columns=["id", "homeConference", "awayConference", "conferenceGame"])
    cg = cg.rename(columns={"id": "cbbd_game_id"})
    df = df.merge(cg, on="cbbd_game_id", how="left")
    npow = df["homeConference"].isin(POWER).astype(int) + df["awayConference"].isin(POWER).astype(int)
    df["conf_tier"] = np.select([npow == 2, npow == 1], ["both power", "one power"], "neither power")
    lines = pd.read_parquet(LINES)
    lines = lines[(lines["season"] == SEASON) & (lines["provider"] == "ESPN BET")].drop_duplicates("game_id")
    df = df.merge(lines[["game_id", "close_spread_home"]], on="game_id", how="left")
    df["close_margin"] = -df["close_spread_home"]
    df["site"] = np.where(df["neutral"] > 0, "neutral", "home/away")
    return df, meta


def slope_decomp(d: pd.DataFrame, chans) -> dict:
    x = d["sim_margin_mean"].to_numpy(float); y = d["margin"].to_numpy(float)
    vx = np.var(x, ddof=1)
    out = {"n": len(d), "slope": slope(y, x), "sd_X": float(np.sqrt(vx))}
    tot = 0.0
    for c in chans:
        k = float(np.cov(d[f"X_{c}"] - d[f"Y_{c}"], x)[0, 1] / vx)
        out[c] = k; tot += k
    # residual terms: sim_margin_mean vs sum of sim channels (should be 0), actual final vs box identity
    k_sim = float(np.cov(d["sim_margin_mean"] - d[[f"X_{c}" for c in chans]].sum(axis=1), x)[0, 1] / vx)
    k_act = float(np.cov(-(d["margin"] - d[[f"Y_{c}" for c in chans]].sum(axis=1)), x)[0, 1] / vx)
    out["sim_identity_resid"] = k_sim; out["final_vs_box"] = k_act
    out["sum_all"] = tot + k_sim + k_act
    out["one_minus_slope"] = 1 - out["slope"]
    return out


def team_fe(df: pd.DataFrame, key: str, src: str, weight: str | None, rows_mask=None):
    """Fixed-effect fit: rate(offence team i vs defence j, game) = mu + off_i + def_j + b_home*site_home + b_away*site_away.
    Returns (off Series, def Series, b_home, b_away). src in {'X','Y'}; key a rate like 't'."""
    recs = []
    for side, o in (("h", "a"), ("a", "h")):
        sub = pd.DataFrame({
            "off": df["home_team_id"] if side == "h" else df["away_team_id"],
            "dfn": df["away_team_id"] if side == "h" else df["home_team_id"],
            "y": df[f"{src}_{side}_{key}"],
            "w": 1.0 if weight is None else df[f"{src}_{side}_n_{weight}"] if weight != "P" else df[f"{src}_{side}_P"],
            "sh": ((df["neutral"] == 0) & (side == "h")).astype(float),
            "sa": ((df["neutral"] == 0) & (side == "a")).astype(float),
        })
        if rows_mask is not None:
            sub = sub[rows_mask.to_numpy()]
        recs.append(sub)
    s = pd.concat(recs, ignore_index=True).dropna()
    s = s[s["w"] > 0]
    teams = np.unique(np.concatenate([s["off"], s["dfn"]]))
    ti = {t: i for i, t in enumerate(teams)}
    n, T = len(s), len(teams)
    Xd = np.zeros((n, 3 + 2 * T))
    Xd[:, 0] = 1; Xd[:, 1] = s["sh"]; Xd[:, 2] = s["sa"]
    oi = s["off"].map(ti).to_numpy(); di = s["dfn"].map(ti).to_numpy()
    Xd[np.arange(n), 3 + oi] = 1; Xd[np.arange(n), 3 + T + di] = 1
    # sum-to-zero soft constraints via tiny ridge on team effects (identifies the split; differences unaffected)
    sw = np.sqrt(s["w"].to_numpy(float))
    A = Xd * sw[:, None]; b = s["y"].to_numpy(float) * sw
    lam = 1e-6 * float((sw ** 2).mean())
    reg = np.zeros(3 + 2 * T); reg[3:] = lam
    beta = np.linalg.solve(A.T @ A + np.diag(reg), A.T @ b)
    off = pd.Series(beta[3:3 + T], index=teams); dfn = pd.Series(beta[3 + T:], index=teams)
    off -= off.mean(); dfn -= dfn.mean()
    ngames = pd.Series(np.concatenate([s["off"], s["dfn"]])).value_counts() / 2
    return off, dfn, float(beta[1]), float(beta[2]), ngames


RATE_KEYS = {  # rate: (weight key, PPP derivative name)
    "t": (None, "tov"), "rp": (None, "ft_trips"), "f": ("fta", "ft_pct"), "rho": ("oreb_chances", "oreb"),
    "s_3": (None, "mix_3"), "s_rim": (None, "mix_rim"), "p_rim": ("fga_rim", "make_rim"),
    "p_jump": ("fga_jump", "make_jump"), "p_3": ("fga_3", "make_3"), "PPP": (None, "PPP"), "P": (None, "pace"),
}


def part_analyse():
    df, meta = load_frame("A")
    dfB, _ = load_frame("B")
    P_ref = meta["P_ref"]
    for d in (df, dfB):
        for s in ("X", "Y"):
            for sd in ("h", "a"):
                d[f"{s}_{sd}_n_oreb_chances"] = d[f"{s}_{sd}_n_oreb"] + d[f"{s}_{sd}_n_oppdreb"]
    res = {"P_ref": P_ref}
    ok = df.dropna(subset=["Y_tov"])
    okB = dfB.dropna(subset=["Y_tov"])
    res["n_games_all"] = len(df); res["n_games_box"] = len(ok)
    h = G.headline(df)
    res["headline"] = h
    res["headline_boxsubset_slope"] = slope(ok["margin"], ok["sim_margin_mean"])
    res["slope_B"] = G.headline(dfB)["slope"]
    # reverse regression / spread
    res["sd_sim_mean"] = float(df["sim_margin_mean"].std()); res["sd_actual"] = float(df["margin"].std())
    res["corr"] = float(df["sim_margin_mean"].corr(df["margin"]))
    # 1) channel decomposition, A and B
    dA = slope_decomp(ok, CHANNELS); dB = slope_decomp(okB, CHANNELS)
    res["decomp_A"] = dA; res["decomp_B"] = dB
    # channel-level stats
    chs = []
    x = ok["sim_margin_mean"]
    for c in CHANNELS:
        chs.append({"channel": c, "model": CH_MODEL[c], "mean_X": ok[f"X_{c}"].mean(), "mean_Y": ok[f"Y_{c}"].mean(),
                    "sd_X": ok[f"X_{c}"].std(), "sd_Y": ok[f"Y_{c}"].std(),
                    "slope_Yc_on_Xc": slope(ok[f"Y_{c}"], ok[f"X_{c}"]),
                    "corr_Xc_X": float(ok[f"X_{c}"].corr(x)),
                    "k_A": dA[c], "k_B": dB[c], "pts_at_1sd_A": dA[c] * dA["sd_X"]})
    res["channels"] = chs
    # 2) segments
    ok = ok.copy()
    ok["absX"] = ok["sim_margin_mean"].abs()
    ok["xband"] = pd.cut(ok["absX"], [0, 3, 6, 10, 15, 100], right=False).astype(str)
    ok["cband"] = pd.cut(ok["close_margin"].abs(), [0, 3, 6, 10, 15, 100], right=False).astype(str)
    seg = {}
    for key in ("month", "conf_tier", "xband", "cband", "site", "conferenceGame"):
        rows = []
        for gk, d in ok.groupby(key):
            if gk == "nan" or len(d) < 30:
                continue
            dd = slope_decomp(d, CHANNELS)
            dl = d.dropna(subset=["close_margin"])
            row = {"seg": str(gk), "n": len(d), "slope": dd["slope"], "sd_X": dd["sd_X"],
                   "bias": float((d["sim_margin_mean"] - d["margin"]).mean()),
                   "n_lined": len(dl)}
            if len(dl) >= 30:
                row.update({"slope_Y_on_close": slope(dl["margin"], dl["close_margin"]),
                            "slope_close_on_X": slope(dl["close_margin"], dl["sim_margin_mean"]),
                            "slope_Y_on_X_lined": slope(dl["margin"], dl["sim_margin_mean"]),
                            "sd_close": float(dl["close_margin"].std()), "sd_X_lined": float(dl["sim_margin_mean"].std()),
                            "bias_vs_close": float((dl["sim_margin_mean"] - dl["close_margin"]).mean()),
                            "close_bias": float((dl["close_margin"] - dl["margin"]).mean())})
            for grp, cs in (("shoot", ["make_rim", "make_jump", "make_3"]), ("mix", ["mix_rim", "mix_jump", "mix_3"]),
                            ("tov", ["tov"]), ("ft", ["ft_trips", "ft_pct"]), ("reb", ["oreb", "reb_chances"]),
                            ("clock", ["pace", "possdiff"]), ("resid", ["final_vs_box", "sim_identity_resid"])):
                row[f"k_{grp}"] = sum(dd[c] for c in cs)
            rows.append(row)
        seg[key] = rows
    # slope segment noise floor: B stream slope by month
    okB2 = okB.copy()
    seg["month_B_slope"] = {str(m): slope(d["margin"], d["sim_margin_mean"]) for m, d in okB2.groupby("month")}
    lined = ok.dropna(subset=["close_margin"])
    res["lined"] = {"n": len(lined), "slope_Y_on_X": slope(lined["margin"], lined["sim_margin_mean"]),
                    "slope_Y_on_close": slope(lined["margin"], lined["close_margin"]),
                    "slope_close_on_X": slope(lined["close_margin"], lined["sim_margin_mean"]),
                    "sd_X": float(lined["sim_margin_mean"].std()), "sd_close": float(lined["close_margin"].std()),
                    "corr_X_close": float(lined["sim_margin_mean"].corr(lined["close_margin"])),
                    "mae_X": float((lined["sim_margin_mean"] - lined["margin"]).abs().mean()),
                    "mae_close": float((lined["close_margin"] - lined["margin"]).abs().mean())}
    # multiple regression Y on X and close
    Z = np.column_stack([np.ones(len(lined)), lined["sim_margin_mean"], lined["close_margin"]])
    res["lined"]["Y_on_X_and_close"] = np.linalg.lstsq(Z, lined["margin"].to_numpy(float), rcond=None)[0].tolist()
    res["segments"] = seg

    # 3) team-level off/def by rate, with split-half reliability for actual
    ref = {}
    for k in ("t", "rp", "f", "rho", "m", "s_rim", "s_jump", "s_3", "p_rim", "p_jump", "p_3"):
        ref[k] = float(pd.concat([ok[f"Y_h_{k}"], ok[f"Y_a_{k}"]]).mean())
    Qr = sum(V[k] * ref[f"s_{k}"] * ref[f"p_{k}"] for k in KS)
    Ar = 1 + ref["rho"] * ref["m"] - ref["t"] - 0.44 * ref["rp"]
    fr = ref["f"]
    deriv = {"t": -Qr, "rp": fr - 0.44 * Qr, "f": ref["rp"], "rho": Qr * ref["m"],
             "s_3": Ar * (3 * ref["p_3"] - 2 * ref["p_jump"]),  # 3PA replacing a jumper
             "s_rim": Ar * (2 * ref["p_rim"] - 2 * ref["p_jump"]),  # rim replacing a jumper
             "p_rim": Ar * 2 * ref["s_rim"], "p_jump": Ar * 2 * ref["s_jump"], "p_3": Ar * 3 * ref["s_3"],
             "PPP": 1.0}
    res["ref_rates"] = ref; res["deriv_ppp"] = deriv; res["Q_ref"] = Qr; res["A_ref"] = Ar
    ok = ok.sort_values("game_date").reset_index(drop=True)
    # split halves per game parity (by date order index)
    half = (np.arange(len(ok)) % 2 == 0)
    half = pd.Series(half, index=ok.index)
    wmap = {"t": "P", "rp": "P", "f": "fta", "rho": "oreb_chances", "s_3": None, "s_rim": None,
            "p_rim": "fga_rim", "p_jump": "fga_jump", "p_3": "fga_3", "PPP": "P"}
    # P weight: need X_h_P / Y_h_P columns
    team_rows = []; site_rows = []; offdef_k = {}
    x_all = ok["sim_margin_mean"].to_numpy(float); vx = np.var(x_all, ddof=1)
    for k, w in wmap.items():
        wk = w if w != "P" else "P"
        offX, defX, bhX, baX, ng = team_fe(ok, k, "X", wk)
        offY, defY, bhY, baY, _ = team_fe(ok, k, "Y", wk)
        o1, d1, *_ = team_fe(ok, k, "Y", wk, rows_mask=half)
        o2, d2, *_ = team_fe(ok, k, "Y", wk, rows_mask=~half)
        site_rows.append({"rate": k, "sim_home": bhX, "sim_away": baX, "act_home": bhY, "act_away": baY,
                          "sim_hca_rate": bhX - baX, "act_hca_rate": bhY - baY,
                          "sim_hca_pts": P_ref * deriv[k] * (bhX - baX) if k != "PPP" else P_ref * (bhX - baX),
                          "act_hca_pts": P_ref * deriv[k] * (bhY - baY) if k != "PPP" else P_ref * (bhY - baY)})
        for nm, eX, eY, e1, e2 in (("off", offX, offY, o1, o2), ("def", defX, defY, d1, d2)):
            idx = eX.index.intersection(eY.index).intersection(e1.index).intersection(e2.index)
            idx = idx[ng.reindex(idx).fillna(0).to_numpy() >= 10]
            ex, ey = eX[idx], eY[idx]
            var_true = float(np.cov(e1[idx], e2[idx])[0, 1])
            sd_true = np.sqrt(var_true) if var_true > 0 else np.nan
            b = slope(ey, ex)
            dv = deriv[k] if k != "PPP" else 1.0
            team_rows.append({"rate": k, "side": nm, "n_teams": len(idx), "sd_pred": float(ex.std()),
                              "sd_realised_raw": float(ey.std()), "sd_true_splithalf": sd_true,
                              "sd_ratio_pred_true": float(ex.std() / sd_true) if sd_true == sd_true else np.nan,
                              "team_slope": b, "corr": float(np.corrcoef(ex, ey)[0, 1]),
                              "pts_overstated_1sd_team": P_ref * dv * (1 - b) * float(ex.std())})
            # game-level additive offence/defence components & slope-miss share
            if k in ("PPP",):
                continue
            ch = RATE_KEYS[k][1]
            if nm == "off":
                gX = ok["home_team_id"].map(eX).to_numpy() - ok["away_team_id"].map(eX).to_numpy()
                gY = ok["home_team_id"].map(eY).to_numpy() - ok["away_team_id"].map(eY).to_numpy()
            else:
                gX = ok["away_team_id"].map(eX).to_numpy() - ok["home_team_id"].map(eX).to_numpy()
                gY = ok["away_team_id"].map(eY).to_numpy() - ok["home_team_id"].map(eY).to_numpy()
            m = np.isfinite(gX) & np.isfinite(gY)
            kk = float(np.cov((P_ref * dv * (gX - gY))[m], x_all[m])[0, 1] / vx)
            offdef_k[f"{k}_{nm}"] = kk
    res["team_level"] = team_rows; res["site_fe"] = site_rows; res["offdef_k"] = offdef_k

    # 4) G6: site means of channels
    g6 = []
    for site, d in ok.groupby("site"):
        row = {"site": site, "n": len(d), "sim_margin": d["sim_margin_mean"].mean(), "act_margin": d["margin"].mean(),
               "close_margin": d["close_margin"].mean(), "n_lined": int(d["close_margin"].notna().sum())}
        dl = d.dropna(subset=["close_margin"])
        row["sim_margin_lined"] = dl["sim_margin_mean"].mean(); row["act_margin_lined"] = dl["margin"].mean()
        row["se_act"] = float(d["margin"].std() / np.sqrt(len(d)))
        for c in CHANNELS:
            row[f"X_{c}"] = d[f"X_{c}"].mean(); row[f"Y_{c}"] = d[f"Y_{c}"].mean()
        row["Y_final_vs_box"] = (d["margin"] - d[[f"Y_{c}" for c in CHANNELS]].sum(axis=1)).mean()
        g6.append(row)
    res["g6_site_means"] = g6
    # neutral: team-strength-adjusted: regress actual margin on sim_margin_mean + neutral dummy etc.
    neu = ok[ok["neutral"] > 0]
    res["neutral_detail"] = {
        "n": len(neu),
        "share_listed_home_fav_sim": float((neu["sim_margin_mean"] > 0).mean()),
        "resid_mean": float((neu["margin"] - neu["sim_margin_mean"]).mean()),
        "resid_se": float((neu["margin"] - neu["sim_margin_mean"]).std() / np.sqrt(len(neu))),
        "close_minus_sim_mean": float((neu["close_margin"] - neu["sim_margin_mean"]).mean()),
        "n_lined": int(neu["close_margin"].notna().sum()),
        "by_month": {str(m): {"n": len(d), "sim": d["sim_margin_mean"].mean(), "act": d["margin"].mean(),
                              "close": d["close_margin"].mean()} for m, d in neu.groupby("month")},
    }
    # neutral games: does the listed home team play in its own state / is venue near? use CBBD venue state vs team? skip.
    # FE margin model on actual vs sim: margin = theta_h - theta_a + b_home*(1-neutral) + b_neu*neutral
    for src, col in (("sim", "sim_margin_mean"), ("act", "margin"), ("close", "close_margin")):
        d = ok.dropna(subset=[col])
        teams = np.unique(np.concatenate([d["home_team_id"], d["away_team_id"]]))
        ti = {t: i for i, t in enumerate(teams)}
        n, T = len(d), len(teams)
        Xd = np.zeros((n, 2 + T))
        Xd[:, 0] = (d["neutral"] == 0).astype(float); Xd[:, 1] = (d["neutral"] > 0).astype(float)
        Xd[np.arange(n), 2 + d["home_team_id"].map(ti).to_numpy()] += 1
        Xd[np.arange(n), 2 + d["away_team_id"].map(ti).to_numpy()] -= 1
        reg = np.zeros(2 + T); reg[2:] = 1e-6
        beta = np.linalg.solve(Xd.T @ Xd + np.diag(reg), Xd.T @ d[col].to_numpy(float))
        res.setdefault("fe_margin", {})[src] = {"hca_nonneutral": float(beta[0]), "listed_home_neutral": float(beta[1]), "n": n}
    json.dump(res, open(OUT / "analysis_v1.json", "w"), indent=1, default=float)
    print_summary(res)


def print_summary(res):
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(json.dumps({k: res[k] for k in ("P_ref", "n_games_all", "n_games_box", "headline_boxsubset_slope", "slope_B",
                                            "sd_sim_mean", "sd_actual", "corr", "lined", "fe_margin", "neutral_detail")},
                     indent=1, default=float))
    print(json.dumps(res["headline"], indent=1, default=float))
    print({k: round(v, 4) for k, v in res["decomp_A"].items()})
    print({k: round(v, 4) for k, v in res["decomp_B"].items()})
    print(pd.DataFrame(res["channels"]).round(4).to_string())
    for k, rows in res["segments"].items():
        print("==", k); print(pd.DataFrame(rows).round(3).to_string() if isinstance(rows, list) else rows)
    print(pd.DataFrame(res["team_level"]).round(4).to_string())
    print({k: round(v, 4) for k, v in res["offdef_k"].items()})
    print(pd.DataFrame(res["site_fe"]).round(4).to_string())
    print(pd.DataFrame(res["g6_site_means"]).T.to_string())


def fe_design(d: pd.DataFrame):
    teams = np.unique(np.concatenate([d["home_team_id"], d["away_team_id"]]))
    ti = {t: i for i, t in enumerate(teams)}
    n, T = len(d), len(teams)
    Xd = np.zeros((n, 2 + T))
    Xd[:, 0] = (d["neutral"] == 0).astype(float); Xd[:, 1] = (d["neutral"] > 0).astype(float)
    Xd[np.arange(n), 2 + d["home_team_id"].map(ti).to_numpy()] += 1
    Xd[np.arange(n), 2 + d["away_team_id"].map(ti).to_numpy()] -= 1
    return Xd


def fe_fit(Xd, y):
    """margin-level FE: y = b_home*nonneutral + b_neu*neutral + theta_h - theta_a + e.
    Returns components (site_nonneutral, site_neutral, team, resid) and coef SEs of the two site terms."""
    reg = np.zeros(Xd.shape[1]); reg[2:] = 1e-6
    XtX = Xd.T @ Xd + np.diag(reg)
    beta = np.linalg.solve(XtX, Xd.T @ y)
    site_nn = Xd[:, 0] * beta[0]; site_neu = Xd[:, 1] * beta[1]
    team = Xd[:, 2:] @ beta[2:]
    resid = y - site_nn - site_neu - team
    dof = len(y) - np.linalg.matrix_rank(Xd)
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.inv(XtX)
    return {"site_nn": site_nn, "site_neu": site_neu, "team": team, "resid": resid,
            "b_home": float(beta[0]), "b_neu": float(beta[1]),
            "se_home": float(np.sqrt(cov[0, 0])), "se_neu": float(np.sqrt(cov[1, 1]))}


def part_extra():
    df, meta = load_frame("A")
    ok = df.dropna(subset=["Y_tov"]).reset_index(drop=True)
    x = ok["sim_margin_mean"].to_numpy(float); y = ok["margin"].to_numpy(float)
    vx = np.var(x, ddof=1)
    Xd = fe_design(ok)
    res = {}
    # 1) margin-level FE split of sim prediction and actual
    fx = fe_fit(Xd, x); fy = fe_fit(Xd, y)
    res["fe_sim"] = {k: fx[k] for k in ("b_home", "b_neu", "se_home", "se_neu")}
    res["fe_act"] = {k: fy[k] for k in ("b_home", "b_neu", "se_home", "se_neu")}
    comp = {}
    for part in ("team", "site_nn", "site_neu", "resid"):
        comp[part] = {"sd_X": float(np.std(fx[part], ddof=1)), "sd_Y": float(np.std(fy[part], ddof=1)),
                      "k": float(np.cov(fx[part] - fy[part], x)[0, 1] / vx)}
    res["margin_components"] = comp
    # regression of Y on X_team(+site) and X_within
    Xfe = fx["team"] + fx["site_nn"] + fx["site_neu"]; Xw = fx["resid"]
    Z = np.column_stack([np.ones(len(y)), Xfe, Xw])
    res["Y_on_Xfe_Xw"] = np.linalg.lstsq(Z, y, rcond=None)[0].tolist()
    res["slope_Y_on_Xfe"] = slope(y, Xfe); res["slope_Y_on_Xw"] = slope(y, Xw)
    res["sd_Xfe"] = float(np.std(Xfe, ddof=1)); res["sd_Xw"] = float(np.std(Xw, ddof=1))
    # same for close (lined subset)
    lm = ok["close_margin"].notna().to_numpy()
    dl = ok[lm].reset_index(drop=True); Xdl = fe_design(dl)
    fc = fe_fit(Xdl, dl["close_margin"].to_numpy(float)); fxl = fe_fit(Xdl, dl["sim_margin_mean"].to_numpy(float))
    yl = dl["margin"].to_numpy(float)
    Cfe = fc["team"] + fc["site_nn"] + fc["site_neu"]; Cw = fc["resid"]
    Xfel = fxl["team"] + fxl["site_nn"] + fxl["site_neu"]; Xwl = fxl["resid"]
    res["close_split"] = {"b_home": fc["b_home"], "b_neu": fc["b_neu"], "se_neu": fc["se_neu"],
                          "sd_Cfe": float(np.std(Cfe, ddof=1)), "sd_Cw": float(np.std(Cw, ddof=1)),
                          "sd_Xfe_lined": float(np.std(Xfel, ddof=1)), "sd_Xw_lined": float(np.std(Xwl, ddof=1)),
                          "slope_Y_on_Cfe": slope(yl, Cfe), "slope_Y_on_Cw": slope(yl, Cw),
                          "slope_Y_on_Xfe": slope(yl, Xfel), "slope_Y_on_Xw": slope(yl, Xwl),
                          "slope_Cfe_on_Xfe": slope(Cfe, Xfel), "slope_Cw_on_Xw": slope(Cw, Xwl),
                          "corr_Xw_Cw": float(np.corrcoef(Xwl, Cw)[0, 1]),
                          "corr_Xfe_Cfe": float(np.corrcoef(Xfel, Cfe)[0, 1])}
    # by month: within-part SD and slope
    mrows = []
    for mth in (11, 12, 1, 2, 3):
        mm = (ok["month"] == mth).to_numpy()
        mrows.append({"month": mth, "n": int(mm.sum()), "slope": slope(y[mm], x[mm]),
                      "slope_Y_on_Xfe": slope(y[mm], Xfe[mm]), "slope_Y_on_Xw": slope(y[mm], Xw[mm]),
                      "sd_Xfe": float(np.std(Xfe[mm], ddof=1)), "sd_Xw": float(np.std(Xw[mm], ddof=1)),
                      "k_team": float(np.cov((fx["team"] - fy["team"])[mm], x[mm])[0, 1] / np.var(x[mm], ddof=1)),
                      "k_within": float(np.cov((fx["resid"] - fy["resid"])[mm], x[mm])[0, 1] / np.var(x[mm], ddof=1))})
    res["month_split"] = mrows
    # 2) per-channel FE split: team / site / within contributions to 1 - slope, and site coefficients
    crow = []
    for c in CHANNELS:
        gx = fe_fit(Xd, ok[f"X_{c}"].to_numpy(float)); gy = fe_fit(Xd, ok[f"Y_{c}"].to_numpy(float))
        r = {"channel": c}
        for part in ("team", "site_nn", "site_neu", "resid"):
            r[f"k_{part}"] = float(np.cov(gx[part] - gy[part], x)[0, 1] / vx)
        r["sd_team_X"] = float(np.std(gx["team"], ddof=1)); r["sd_team_Y"] = float(np.std(gy["team"], ddof=1))
        r["team_slope"] = slope(gy["team"], gx["team"])
        r["sd_within_X"] = float(np.std(gx["resid"], ddof=1))
        r["within_slope"] = slope(gy["resid"], gx["resid"])
        r["hca_sim"] = gx["b_home"]; r["hca_act"] = gy["b_home"]; r["hca_act_se"] = gy["se_home"]
        r["neu_sim"] = gx["b_neu"]; r["neu_act"] = gy["b_neu"]; r["neu_act_se"] = gy["se_neu"]
        neu = ok["neutral"].to_numpy() > 0
        r["neu_teampart_sim"] = float(gx["team"][neu].mean()); r["neu_teampart_act"] = float(gy["team"][neu].mean())
        crow.append(r)
    res["channel_fe"] = crow
    # final-vs-box residual site coefficients
    fr = fe_fit(Xd, y - ok[[f"Y_{c}" for c in CHANNELS]].sum(axis=1).to_numpy(float))
    res["final_vs_box_fe"] = {"b_home": fr["b_home"], "b_neu": fr["b_neu"]}
    neu = ok["neutral"].to_numpy() > 0
    res["neutral_team_part"] = {"sim": float(fx["team"][neu].mean()), "act": float(fy["team"][neu].mean())}
    # 3) neutral games at the listed home team's own venue / state
    cg = pd.read_parquet("data/raw/cbbd/games_2025.parquet",
                         columns=["id", "homeTeamId", "neutralSite", "venueId", "state", "seasonType", "tournament"])
    homev = cg[~cg["neutralSite"].astype(bool)].groupby("homeTeamId")
    home_venue = homev["venueId"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    home_state = homev["state"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
    cg["own_venue"] = cg["venueId"] == cg["homeTeamId"].map(home_venue)
    cg["own_state"] = cg["state"] == cg["homeTeamId"].map(home_state)
    n2 = ok.merge(cg.rename(columns={"id": "cbbd_game_id"})[["cbbd_game_id", "own_venue", "own_state", "seasonType", "tournament"]],
                  on="cbbd_game_id", how="left")
    n2["resid"] = n2["margin"] - n2["sim_margin_mean"]
    n2["resid_close"] = n2["close_margin"] - n2["sim_margin_mean"]
    n2 = n2[n2["neutral"] > 0].copy()
    n2["venue_cls"] = np.select([n2["own_venue"].fillna(False).astype(bool), n2["own_state"].fillna(False).astype(bool)],
                                ["listed-home own arena", "listed-home own state"], "elsewhere")
    vrows = []
    for key in ("venue_cls", "seasonType", "tournament"):
        for gk, d in n2.groupby(key, dropna=False):
            vrows.append({"key": key, "grp": str(gk), "n": len(d), "sim": d["sim_margin_mean"].mean(),
                          "act": d["margin"].mean(), "close": d["close_margin"].mean(),
                          "resid": d["resid"].mean(), "resid_se": float(d["resid"].std() / np.sqrt(len(d))) if len(d) > 1 else np.nan,
                          "close_minus_sim": d["resid_close"].mean()})
    res["neutral_venue"] = vrows
    json.dump(res, open(OUT / "extra_v1.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(json.dumps({k: v for k, v in res.items() if k not in ("channel_fe", "neutral_venue", "month_split")}, indent=1, default=float))
    print(pd.DataFrame(res["month_split"]).round(4).to_string())
    print(pd.DataFrame(res["channel_fe"]).round(4).to_string())
    print(pd.DataFrame(res["neutral_venue"]).round(3).to_string())


def part_close(tag="A"):
    """Close-referenced attribution on the lined subset: 1 - slope(close on X) split into the sim's
    season-average team component and its within-season component, then each split by channel via a
    multiple regression of the close's matching component on the sim's channel components (closes exactly)."""
    df, meta = load_frame(tag)
    d = df.dropna(subset=["Y_tov", "close_margin"]).reset_index(drop=True)
    Xd = fe_design(d)
    x = d["sim_margin_mean"].to_numpy(float); c = d["close_margin"].to_numpy(float); y = d["margin"].to_numpy(float)
    vx = np.var(x, ddof=1)
    fx = fe_fit(Xd, x); fc = fe_fit(Xd, c); fy = fe_fit(Xd, y)
    Xfe = x - fx["resid"]; Xw = fx["resid"]; Cfe = c - fc["resid"]; Cw = fc["resid"]
    res = {"tag": tag, "n": len(d), "slope_close_on_X": slope(c, x),
           "k_team": float(np.cov(Xfe - Cfe, x)[0, 1] / vx), "k_within": float(np.cov(Xw - Cw, x)[0, 1] / vx)}
    # same split for the close predicting the actual (artifact benchmark) and for the sim predicting actual
    vc = np.var(c, ddof=1)
    res["close_vs_act"] = {"slope": slope(y, c), "k_team": float(np.cov(Cfe - (y - fy["resid"]), c)[0, 1] / vc),
                           "k_within": float(np.cov(Cw - fy["resid"], c)[0, 1] / vc)}
    res["sim_vs_act"] = {"slope": slope(y, x), "k_team": float(np.cov(Xfe - (y - fy["resid"]), x)[0, 1] / vx),
                         "k_within": float(np.cov(Xw - fy["resid"], x)[0, 1] / vx)}
    # channel components of the sim
    fe_c = {ch: fe_fit(Xd, d[f"X_{ch}"].to_numpy(float)) for ch in CHANNELS}
    rows = []
    for part, target in (("team", Cfe), ("within", Cw)):
        cols = [(d[f"X_{ch}"].to_numpy(float) - fe_c[ch]["resid"]) if part == "team" else fe_c[ch]["resid"]
                for ch in CHANNELS]
        Z = np.column_stack([np.ones(len(d))] + cols)
        beta, *_ = np.linalg.lstsq(Z, target, rcond=None)
        resid = target - Z @ beta
        s2 = float(resid @ resid / (len(d) - Z.shape[1]))
        se = np.sqrt(np.diag(s2 * np.linalg.inv(Z.T @ Z)))
        for i, ch in enumerate(CHANNELS):
            rows.append({"part": part, "channel": ch, "beta_close_on_sim": float(beta[i + 1]), "se": float(se[i + 1]),
                         "sd_sim_comp": float(np.std(cols[i], ddof=1)),
                         "k": float((1 - beta[i + 1]) * np.cov(cols[i], x)[0, 1] / vx)})
        rows.append({"part": part, "channel": "_unexplained_close_resid", "beta_close_on_sim": np.nan, "se": np.nan,
                     "sd_sim_comp": np.nan, "k": float(-np.cov(resid, x)[0, 1] / vx)})
    res["channels"] = rows
    # G6 by site: team-part means on lined subset
    neu = d["neutral"].to_numpy() > 0
    res["site_means"] = {s: {"sim": float(x[m].mean()), "close": float(c[m].mean()), "act": float(y[m].mean()),
                             "sim_team": float(fx["team"][m].mean()), "close_team": float(fc["team"][m].mean()),
                             "act_team": float(fy["team"][m].mean())}
                         for s, m in (("home/away", ~neu), ("neutral", neu))}
    res["site_coef"] = {"sim": [fx["b_home"], fx["b_neu"]], "close": [fc["b_home"], fc["b_neu"]],
                        "act": [fy["b_home"], fy["b_neu"], fy["se_home"], fy["se_neu"]]}
    # within = drift (team x month effects on the FE residual) + jitter (game-specific remainder)
    teams = np.unique(np.concatenate([d["home_team_id"], d["away_team_id"]]))
    months = sorted(d["month"].unique())
    tm = {(t, m): i for i, (t, m) in enumerate([(t, m) for t in teams for m in months])}
    D = np.zeros((len(d), len(tm)))
    hi = [tm[(t, m)] for t, m in zip(d["home_team_id"], d["month"])]
    ai = [tm[(t, m)] for t, m in zip(d["away_team_id"], d["month"])]
    D[np.arange(len(d)), hi] += 1; D[np.arange(len(d)), ai] -= 1
    DtD = D.T @ D + 1e-3 * np.eye(D.shape[1])
    def drift(v):
        return D @ np.linalg.solve(DtD, D.T @ v)
    Xdr, Cdr = drift(Xw), drift(Cw)
    Xj, Cj = Xw - Xdr, Cw - Cdr
    res["within_split"] = {"sd_X_drift": float(np.std(Xdr, ddof=1)), "sd_C_drift": float(np.std(Cdr, ddof=1)),
                           "sd_X_jitter": float(np.std(Xj, ddof=1)), "sd_C_jitter": float(np.std(Cj, ddof=1)),
                           "slope_Cdrift_on_Xdrift": slope(Cdr, Xdr), "slope_Cjit_on_Xjit": slope(Cj, Xj),
                           "k_drift": float(np.cov(Xdr - Cdr, x)[0, 1] / vx),
                           "k_jitter": float(np.cov(Xj - Cj, x)[0, 1] / vx)}
    # by month, close-referenced
    res["month"] = []
    for mth in (11, 12, 1, 2, 3):
        m = (d["month"] == mth).to_numpy(); v = np.var(x[m], ddof=1)
        res["month"].append({"month": mth, "n": int(m.sum()), "slope_close_on_X": slope(c[m], x[m]),
                             "k_team": float(np.cov((Xfe - Cfe)[m], x[m])[0, 1] / v),
                             "k_within": float(np.cov((Xw - Cw)[m], x[m])[0, 1] / v),
                             "sd_Xw": float(np.std(Xw[m], ddof=1)), "sd_Cw": float(np.std(Cw[m], ddof=1)),
                             "slope_Cw_on_Xw": slope(Cw[m], Xw[m])})
    json.dump(res, open(OUT / f"close_ref_{tag}_v1.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(json.dumps({k: v for k, v in res.items() if k not in ("channels", "month")}, indent=1, default=float))
    print(pd.DataFrame(res["channels"]).round(4).to_string())
    print(pd.DataFrame(res["month"]).round(4).to_string())


def part_boot(n_boot=300, seed=20260930):
    """Game-bootstrap SEs for (a) the vs-actual channel contributions k_c and sub-model group sums, and
    (b) the close-referenced team/within totals and within-channel contributions. FE components are
    computed once on the full sample and held fixed (SEs cover the covariance/regression step only)."""
    df, meta = load_frame("A")
    rng = np.random.default_rng(seed)
    groups = {"possession_outcome": ["tov", "ft_trips", "mix_rim", "mix_jump", "mix_3"],
              "fg_make": ["make_rim", "make_jump", "make_3"], "rebound": ["oreb"], "reb_chances(derived)": ["reb_chances"],
              "free_throw": ["ft_pct"], "clock": ["pace", "possdiff"]}
    ok = df.dropna(subset=["Y_tov"]).reset_index(drop=True)
    x = ok["sim_margin_mean"].to_numpy(float)
    D = np.column_stack([ok[f"X_{c}"] - ok[f"Y_{c}"] for c in CHANNELS])
    out_a = []
    for _ in range(n_boot):
        i = rng.integers(0, len(ok), len(ok))
        xi = x[i]; vx = np.var(xi, ddof=1)
        kc = np.array([np.cov(D[i, j], xi)[0, 1] / vx for j in range(len(CHANNELS))])
        out_a.append(kc)
    A = np.array(out_a)
    res = {"vs_actual_channel_se": dict(zip(CHANNELS, A.std(axis=0).tolist()))}
    res["vs_actual_group"] = {g: {"se": float(A[:, [CHANNELS.index(c) for c in cs]].sum(axis=1).std())} for g, cs in groups.items()}
    res["vs_actual_total_se"] = float(A.sum(axis=1).std())
    # close-referenced
    d = df.dropna(subset=["Y_tov", "close_margin"]).reset_index(drop=True)
    Xd = fe_design(d)
    x = d["sim_margin_mean"].to_numpy(float); c = d["close_margin"].to_numpy(float)
    fx = fe_fit(Xd, x); fc = fe_fit(Xd, c)
    Xw, Cw = fx["resid"], fc["resid"]; Xfe, Cfe = x - Xw, c - Cw
    W = np.column_stack([fe_fit(Xd, d[f"X_{ch}"].to_numpy(float))["resid"] for ch in CHANNELS])
    out_c = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        xi = x[i]; vx = np.var(xi, ddof=1)
        Z = np.column_stack([np.ones(len(i)), W[i]])
        beta = np.linalg.lstsq(Z, Cw[i], rcond=None)[0][1:]
        kw = [(1 - beta[j]) * np.cov(W[i, j], xi)[0, 1] / vx for j in range(len(CHANNELS))]
        out_c.append([np.cov((Xfe - Cfe)[i], xi)[0, 1] / vx, np.cov((Xw - Cw)[i], xi)[0, 1] / vx,
                      1 - np.cov(c[i], xi)[0, 1] / vx] + kw)
    Cb = np.array(out_c)
    res["close_ref_se"] = {"k_team": float(Cb[:, 0].std()), "k_within": float(Cb[:, 1].std()),
                           "one_minus_slope": float(Cb[:, 2].std())}
    res["close_ref_within_channel_se"] = dict(zip(CHANNELS, Cb[:, 3:].std(axis=0).tolist()))
    res["close_ref_within_group_se"] = {g: float(Cb[:, [3 + CHANNELS.index(ch) for ch in cs]].sum(axis=1).std())
                                        for g, cs in groups.items()}
    json.dump(res, open(OUT / "boot_v1.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", choices=["build", "analyse", "extra", "close", "boot"])
    a = ap.parse_args()
    parts = a.part or ["build", "analyse", "extra"]
    if "build" in parts:
        part_build()
    if "analyse" in parts:
        part_analyse()
    if "extra" in parts:
        part_extra()
    if "close" in parts:
        part_close("A")
        part_close("B")
    if "boot" in parts:
        part_boot()
