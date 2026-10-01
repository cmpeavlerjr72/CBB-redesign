#!/usr/bin/env python
"""
exp_shared_shooting_v1.py -- shared (game-level) shooting latent: measurement
(stage `measure`) and the pre-registered offline bake-off (stage `bakeoff`).

    .venv/Scripts/python.exe scripts/exp_shared_shooting_v1.py measure
    .venv/Scripts/python.exe scripts/exp_shared_shooting_v1.py bakeoff

Spec: docs/models/shared_shooting/experiments.md (section 1, committed before
`bakeoff` ran). Inputs: results/shared_shooting/preds_v1.parquet
(scripts/exp_shared_shooting_preds_v1.py: per-FGA held-out make probabilities
of the served fg_make spec), verified finals and the hoopR team box
(`eval.reference`, verified_finals=True), the S0 v3 full-slate engine rows
(results/engine_v0/v3full_S0_s200_o0_off{0,25}_n25) for the arithmetic only.

Notation, per game g, side i in {home, away}, shot type k in {rim, jump2, three}:
  R_ik = sum(y - p)        residual makes
  W_ik = sum(p (1 - p))    binomial variance = d E[makes] / d(logit shift)
A shared logit effect u_k (both teams, one per game) gives
  Cov(R_hk, R_al) = E[W_hk W_al] Sigma_kl     (between-team, exact to first order)
so Sigma is identified by method of moments from BETWEEN-team cross products only
(a team's own hot night cannot enter them). Everything is fitted on TRAINING
seasons; the test season is only scored.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
os.environ["CBB_TRUTH"] = "verified_v1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from cbb_sim.eval import reference as REF  # noqa: E402

OUT = Path("results/shared_shooting")
KEYS = ("rim", "jump2", "three")
PTS = np.array([2.0, 2.0, 3.0])
SEASONS = (2023, 2024, 2025)
FOLDS = {"F1": {"train": [2023], "test": 2024}, "F2": {"train": [2023, 2024], "test": 2025}}
N_BOOT = 200
FT_M = 100.0          # shrinkage pseudo-attempts for the as-of FT% expectation
POSS_MIN_PRIOR = 1


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_games() -> pd.DataFrame:
    """One row per verified game with both sides' shot residual sums, FT
    residuals, the box and a pace residual. Column suffix _h / _a."""
    pr = pd.read_parquet(OUT / "preds_v1.parquet")
    pr["w"] = pr["p"] * (1.0 - pr["p"])
    pr["r"] = pr["y"] - pr["p"]
    pr["side"] = np.where(pr["offense_is_home"].astype(bool), "h", "a")
    agg = pr.groupby(["game_id", "side", "class_key"]).agg(
        n=("y", "size"), mk=("y", "sum"), ep=("p", "sum"), W=("w", "sum"), R=("r", "sum"),
        ce=("chance_elapsed_s", "mean")).reset_index()
    wide = agg.pivot_table(index="game_id", columns=["side", "class_key"],
                           values=["n", "mk", "ep", "W", "R"], fill_value=0.0)
    wide.columns = [f"{v}_{k}_{s}" for v, s, k in wide.columns]
    wide = wide.reset_index()
    rows = []
    for s in SEASONS:
        g = REF.load_actual_games(s, verified_finals=True)
        box = REF.load_actual_team_box(s)
        cols = ["game_id", "team_id", "fga", "fgm", "tpa", "tpm", "fta", "ftm", "team_score", "poss_team"]
        bh = box[box["team_home_away"] == "home"][cols + ["game_poss"]].rename(
            columns={"team_id": "home_team_id"})
        ba = box[box["team_home_away"] == "away"][cols].rename(columns={"team_id": "away_team_id"})
        g = g.merge(bh, on=["game_id", "home_team_id"], how="inner")
        g = g.merge(ba, on=["game_id", "away_team_id"], how="inner", suffixes=("_h", "_a"))
        rows.append(g)
    games = pd.concat(rows, ignore_index=True)
    games = games.merge(wide, on="game_id", how="inner")
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    games = add_ft(games)
    games = add_pace(games)
    games["week"] = pd.to_datetime(games["game_date"]).dt.isocalendar().week.astype(int)
    return games


def add_ft(g: pd.DataFrame) -> pd.DataFrame:
    """As-of FT% expectation per side: the team's prior games this season,
    shrunk to the league's season-to-date FT% (prior dates only). No same-game
    information; FT make residuals are an APPROXIMATION (not the served
    free_throw model) and are labelled so."""
    out = []
    for s, gs in g.groupby("season", sort=False):
        long = pd.concat([
            pd.DataFrame({"game_id": gs["game_id"], "date": gs["game_date"], "team": gs["home_team_id"],
                          "fta": gs["fta_h"], "ftm": gs["ftm_h"], "side": "h"}),
            pd.DataFrame({"game_id": gs["game_id"], "date": gs["game_date"], "team": gs["away_team_id"],
                          "fta": gs["fta_a"], "ftm": gs["ftm_a"], "side": "a"})]).sort_values(["date", "game_id"])
        day = long.groupby("date")[["fta", "ftm"]].sum().cumsum().shift(1)
        lg = (day["ftm"] / day["fta"]).fillna(0.70)
        long["lg"] = long["date"].map(lg)
        long["pa"] = long.groupby("team")["fta"].transform(lambda x: x.shift(1).fillna(0).cumsum())
        long["pm"] = long.groupby("team")["ftm"].transform(lambda x: x.shift(1).fillna(0).cumsum())
        long["e"] = (long["pm"] + FT_M * long["lg"]) / (long["pa"] + FT_M)
        piv = long.pivot_table(index="game_id", columns="side", values="e")
        out.append(piv.rename(columns={"h": "fte_h", "a": "fte_a"}).reset_index())
    e = pd.concat(out, ignore_index=True)
    g = g.merge(e, on="game_id", how="left")
    for s in ("h", "a"):
        g[f"R_ft_{s}"] = g[f"ftm_{s}"] - g[f"fta_{s}"] * g[f"fte_{s}"]
        g[f"W_ft_{s}"] = g[f"fta_{s}"] * g[f"fte_{s}"] * (1.0 - g[f"fte_{s}"])
    return g


def add_pace(g: pd.DataFrame) -> pd.DataFrame:
    """As-of pace expectation: mean of the two teams' prior-game ln(game_poss)
    this season (league season-to-date mean when a team has no prior game),
    then one OLS line ln N ~ a + b * that mean, fitted per fold on TRAIN only
    (in `fit_pace`). Here only the as-of predictor is built."""
    g = g.copy()
    g["lnN"] = np.log(g["game_poss"].astype(float))
    pieces = []
    for s, gs in g.groupby("season", sort=False):
        long = pd.concat([
            pd.DataFrame({"game_id": gs["game_id"], "date": gs["game_date"], "team": gs["home_team_id"],
                          "lnN": gs["lnN"], "side": "h"}),
            pd.DataFrame({"game_id": gs["game_id"], "date": gs["game_date"], "team": gs["away_team_id"],
                          "lnN": gs["lnN"], "side": "a"})]).sort_values(["date", "game_id"])
        day = long.groupby("date")["lnN"].agg(["sum", "count"]).cumsum().shift(1)
        lgm = (day["sum"] / day["count"])
        long["lg"] = long["date"].map(lgm)
        cs = long.groupby("team")["lnN"].transform(lambda x: x.shift(1).fillna(0).cumsum())
        cn = long.groupby("team").cumcount()
        long["t"] = np.where(cn >= POSS_MIN_PRIOR, cs / np.maximum(cn, 1), long["lg"])
        piv = long.pivot_table(index="game_id", columns="side", values="t")
        piv["tempo_asof"] = 0.5 * (piv["h"] + piv["a"])
        pieces.append(piv[["tempo_asof"]].reset_index())
    t = pd.concat(pieces, ignore_index=True)
    return g.merge(t, on="game_id", how="left")


def fit_pace(tr: pd.DataFrame) -> tuple[float, float, float]:
    ok = tr["tempo_asof"].notna() & np.isfinite(tr["lnN"])
    x, y = tr.loc[ok, "tempo_asof"].to_numpy(), tr.loc[ok, "lnN"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    return float(a), float(b), float(res.var())


def arrays(g: pd.DataFrame, centre: bool = False):
    """R, W as (G, 2, 3) arrays (side 0 home, 1 away). `centre` removes the
    (season, ISO week, type) league residual (drift a mean model would own)."""
    R = np.stack([np.stack([g[f"R_{k}_{s}"].to_numpy(float) for k in KEYS], axis=1)
                  for s in ("h", "a")], axis=1)
    W = np.stack([np.stack([g[f"W_{k}_{s}"].to_numpy(float) for k in KEYS], axis=1)
                  for s in ("h", "a")], axis=1)
    if centre:
        key = g["season"].astype(str) + "_" + g["week"].astype(str)
        R = R.copy()
        for kk in key.unique():
            m = (key == kk).to_numpy()
            c = R[m].sum(axis=(0, 1)) / np.maximum(W[m].sum(axis=(0, 1)), 1e-9)
            R[m] = R[m] - W[m] * c[None, None, :]
    return R, W


# ---------------------------------------------------------------------------
# moments
# ---------------------------------------------------------------------------
def sigma_between(R, W, w=None):
    """3x3 shared Sigma by MoM from between-team cross products (symmetrised)."""
    if w is None:
        w = np.ones(len(R))
    num = np.einsum("g,gk,gl->kl", w, R[:, 0], R[:, 1])
    den = np.einsum("g,gk,gl->kl", w, W[:, 0], W[:, 1])
    num, den = 0.5 * (num + num.T), 0.5 * (den + den.T)
    return num / den


def sigma_pooled(R, W, w=None):
    """G1: one shared logit effect on every shot type."""
    if w is None:
        w = np.ones(len(R))
    rh, ra = R[:, 0].sum(1), R[:, 1].sum(1)
    wh, wa = W[:, 0].sum(1), W[:, 1].sum(1)
    return float((w * rh * ra).sum() / (w * wh * wa).sum())


def within_excess(R, W, w=None):
    """Per-type within-team excess over binomial, in logit units^2:
    (sum R^2 - sum W) / sum W^2, both sides pooled (shared + unshared)."""
    if w is None:
        w = np.ones(len(R))
    num = np.einsum("g,gik->k", w, R ** 2 - W)
    den = np.einsum("g,gik->k", w, W ** 2)
    return num / den


def within_cross(R, W, w=None):
    """Within-team 3x3 second moment over binomial: (sum R_k R_l - delta W) / sum W_k W_l."""
    if w is None:
        w = np.ones(len(R))
    num = np.einsum("g,gik,gil->kl", w, R, R) - np.diag(np.einsum("g,gik->k", w, W))
    den = np.einsum("g,gik,gil->kl", w, W, W)
    return num / den


def boot_w(G: int, n: int = N_BOOT, seed: int = 0):
    return np.random.default_rng(seed).poisson(1.0, size=(n, G)).astype(float)


def psd(S):
    v, Q = np.linalg.eigh(0.5 * (S + S.T))
    return (Q * np.clip(v, 0, None)) @ Q.T


# ---------------------------------------------------------------------------
# stage 1: measure
# ---------------------------------------------------------------------------
def s0_within() -> pd.DataFrame:
    rows = []
    for off in (0, 25):
        p = Path(f"results/engine_v0/v3full_S0_s200_o0_off{off}_n25/games.parquet")
        if p.exists():
            rows.append(pd.read_parquet(p, columns=["game_id", "seed", "home_pts", "away_pts"]))
    r = pd.concat(rows, ignore_index=True)
    r["t"] = r["home_pts"] + r["away_pts"]
    r["m"] = r["home_pts"] - r["away_pts"]
    gb = r.groupby("game_id")
    s = gb.agg(h_mean=("home_pts", "mean"), a_mean=("away_pts", "mean"),
               h_var=("home_pts", "var"), a_var=("away_pts", "var"),
               t_sd=("t", "std"), m_sd=("m", "std"), t_mean=("t", "mean"), m_mean=("m", "mean"),
               n=("seed", "size"))
    s["ha_cov"] = gb.apply(lambda x: np.cov(x["home_pts"], x["away_pts"])[0, 1])
    return s.reset_index(), r


def measure(games: pd.DataFrame) -> dict:
    rep: dict = {"created_at": pd.Timestamp.now("UTC").isoformat()}
    # coverage of the design against the box
    cov = {}
    for s, g in games.groupby("season"):
        n_des = sum(g[f"n_{k}_h"] + g[f"n_{k}_a"] for k in KEYS).sum()
        n_box = (g["fga_h"] + g["fga_a"]).sum()
        cov[int(s)] = {"games": int(len(g)), "design_fga_over_box_fga": float(n_des / n_box)}
    rep["coverage"] = cov
    out = {}
    for s in SEASONS + ("2023-24", "all"):
        if s == "2023-24":
            g = games[games["season"].isin([2023, 2024])]
        elif s == "all":
            g = games
        else:
            g = games[games["season"] == s]
        row = {}
        for cen in (False, True):
            R, W = arrays(g, centre=cen)
            S = sigma_between(R, W)
            s1 = sigma_pooled(R, W)
            BW = boot_w(len(R))
            Sb = np.array([sigma_between(R, W, w) for w in BW])
            s1b = np.array([sigma_pooled(R, W, w) for w in BW])
            ex = within_excess(R, W)
            # between-team correlation of per-team-game residual eFG-like units
            vh = (R[:, 0] * PTS).sum(1)
            va = (R[:, 1] * PTS).sum(1)
            # FT
            rf_h, rf_a = g["R_ft_h"].to_numpy(), g["R_ft_a"].to_numpy()
            wf_h, wf_a = g["W_ft_h"].to_numpy(), g["W_ft_a"].to_numpy()
            ok = np.isfinite(rf_h) & np.isfinite(rf_a)
            sft = float((rf_h[ok] * rf_a[ok]).sum() / (wf_h[ok] * wf_a[ok]).sum())
            sftb = np.array([(w[ok] * rf_h[ok] * rf_a[ok]).sum() / (w[ok] * wf_h[ok] * wf_a[ok]).sum()
                             for w in BW])
            row["centred" if cen else "raw"] = {
                "n_games": int(len(R)),
                "Sigma_between": S.tolist(), "Sigma_between_SE": Sb.std(0).tolist(),
                "G1_sigma2": s1, "G1_sigma2_SE": float(s1b.std()),
                "within_excess_logit2": ex.tolist(),
                "pts_resid_corr_h_a": float(np.corrcoef(vh, va)[0, 1]),
                "ft_sigma2": sft, "ft_sigma2_SE": float(sftb.std()),
            }
        out[str(s)] = row
    rep["moments"] = out

    # segments, 2025 and 2023-24 pooled: site, month, pace tercile
    seg = {}
    for lab, gg in (("2025", games[games["season"] == 2025]),
                    ("2023-24", games[games["season"].isin([2023, 2024])])):
        d = {}
        cuts = {"site_home_away": gg["neutral"] == 0, "site_neutral": gg["neutral"] == 1}
        for mth in (11, 12, 1, 2, 3):
            cuts[f"month_{mth}"] = gg["month"] == mth
        q = gg["tempo_asof"].rank(pct=True)
        for i, (lo, hi) in enumerate(((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01))):
            cuts[f"tempo_tercile_{i+1}"] = (q > lo) & (q <= hi)
        for name, m in cuts.items():
            sub = gg[m.to_numpy()]
            if len(sub) < 50:
                d[name] = {"n_games": int(len(sub)), "note": "UNDERPOWERED"}
                continue
            R, W = arrays(sub, centre=True)
            BW = boot_w(len(R), 100)
            s1 = sigma_pooled(R, W)
            d[name] = {"n_games": int(len(sub)), "G1_sigma2": s1,
                       "G1_sigma2_SE": float(np.std([sigma_pooled(R, W, w) for w in BW])),
                       "Sigma_diag": np.diag(sigma_between(R, W)).tolist()}
        seg[lab] = d
    rep["segments"] = seg

    # per team (2025): mean between-team product per game involving the team
    g25 = games[games["season"] == 2025]
    R, W = arrays(g25, centre=True)
    prod = R[:, 0].sum(1) * R[:, 1].sum(1)
    den = W[:, 0].sum(1) * W[:, 1].sum(1)
    long = pd.DataFrame({"team": np.concatenate([g25["home_team_id"], g25["away_team_id"]]),
                         "prod": np.concatenate([prod, prod]), "den": np.concatenate([den, den])})
    pt = long.groupby("team").agg(n=("prod", "size"), num=("prod", "sum"), den=("den", "sum"))
    pt = pt[pt["n"] >= 10]
    pt["s2"] = pt["num"] / pt["den"]
    rep["per_team_2025"] = {"teams": int(len(pt)), "median_games": float(pt["n"].median()),
                            "median_s2": float(pt["s2"].median()),
                            "share_positive": float((pt["s2"] > 0).mean()),
                            "iqr": [float(pt["s2"].quantile(.25)), float(pt["s2"].quantile(.75))],
                            "note": "UNDERPOWERED per team (about 30 games each); distribution only"}
    # per game: the share of games whose between-team product is positive
    rep["per_game_2025"] = {"share_prod_positive": float((prod > 0).mean()),
                            "mean_prod_over_den": float(prod.sum() / den.sum())}

    # pace x efficiency on real data: residual makes vs as-of pace residual
    pc = {}
    for lab, tr_seasons, te in (("F1", [2023], 2024), ("F2", [2023, 2024], 2025)):
        a, b, vq = fit_pace(games[games["season"].isin(tr_seasons)])
        for part, gg in (("train", games[games["season"].isin(tr_seasons)]),
                         ("test", games[games["season"] == te])):
            q = (gg["lnN"] - (a + b * gg["tempo_asof"])).to_numpy()
            R, W = arrays(gg, centre=True)
            rs = R.sum(axis=(1, 2))
            ws = W.sum(axis=(1, 2))
            ok = np.isfinite(q)
            c = float((rs[ok] * q[ok]).sum() / ws[ok].sum())
            BW = boot_w(int(ok.sum()), 100)
            cb = [float((w * rs[ok] * q[ok]).sum() / (w * ws[ok]).sum()) for w in BW]
            pc[f"{lab}_{part}"] = {"pace_line": [a, b], "q_var": float(np.var(q[ok])),
                                   "cov_u_q": c, "cov_u_q_SE": float(np.std(cb)),
                                   "corr_resid_makes_vs_q": float(np.corrcoef(rs[ok], q[ok])[0, 1])}
    rep["pace_x_shooting"] = pc

    # ARITHMETIC: what each fitted-perfectly shared component would add (2025 test)
    s0, raw = s0_within()
    g25 = g25.merge(s0, on="game_id", how="inner")
    act = g25
    R, W = arrays(g25, centre=True)
    S_true = psd(np.array(rep["moments"]["2025"]["centred"]["Sigma_between"]))
    ah, aa = W[:, 0] * PTS, W[:, 1] * PTS
    add_h = np.einsum("gk,kl,gl->g", ah, S_true, ah)
    add_a = np.einsum("gk,kl,gl->g", aa, S_true, aa)
    add_c = np.einsum("gk,kl,gl->g", ah, S_true, aa)
    sft = max(rep["moments"]["2025"]["centred"]["ft_sigma2"], 0.0)
    fh, fa = act["W_ft_h"].fillna(0).to_numpy(), act["W_ft_a"].fillna(0).to_numpy()
    arith = {}
    resid_t = float((act["total"] - act["t_mean"]).std())
    resid_m = float((act["margin"] - act["m_mean"]).std())
    corr_act = float(np.corrcoef(act["home_score"], act["away_score"])[0, 1])
    # pooled sim moments (rows) for the correlation line
    rr = raw[raw["game_id"].isin(act["game_id"])]
    vh0, va0 = rr["home_pts"].var(), rr["away_pts"].var()
    c0 = np.cov(rr["home_pts"], rr["away_pts"])[0, 1]
    for name, (dh, da, dc) in {
        "shooting_Sigma3": (add_h, add_a, add_c),
        "ft_make": (sft * fh * fh, sft * fa * fa, sft * fh * fa),
        "shooting_plus_ft": (add_h + sft * fh * fh, add_a + sft * fa * fa, add_c + sft * fh * fa),
    }.items():
        tsd = np.sqrt(act["t_sd"] ** 2 + dh + da + 2 * dc)
        msd = np.sqrt(np.maximum(act["m_sd"] ** 2 + dh + da - 2 * dc, 0))
        corr_new = (c0 + dc.mean()) / np.sqrt((vh0 + dh.mean()) * (va0 + da.mean()))
        arith[name] = {"mean_added_cov_pts2": float(dc.mean()),
                       "mean_added_var_home": float(dh.mean()), "mean_added_var_away": float(da.mean()),
                       "total_sd_ratio_new": float(tsd.mean() / resid_t),
                       "margin_sd_ratio_new": float(msd.mean() / resid_m),
                       "corr_new": float(corr_new)}
    arith["S0_reference"] = {"games": int(len(act)), "seeds": int(act["n"].median()),
                             "total_sd_ratio": float(act["t_sd"].mean() / resid_t),
                             "margin_sd_ratio": float(act["m_sd"].mean() / resid_m),
                             "corr_sim_rows": float(c0 / np.sqrt(vh0 * va0)), "corr_act": corr_act,
                             "resid_total_var": resid_t ** 2,
                             "mean_sim_total_var": float((act["t_sd"] ** 2).mean())}
    rep["arithmetic_2025"] = arith
    return rep


# ---------------------------------------------------------------------------
# stage 2: bakeoff (spec: docs/models/shared_shooting/experiments.md section 1)
# ---------------------------------------------------------------------------
ARMS = ("N", "G1", "GP", "G3")
SIMPLICITY = {"N": 0, "G1": 1, "GP": 2, "G3": 3}


def fit_arm(arm: str, R, W, q, w=None):
    """Return (Sigma 3x3 shared, c coupling to q per type, T 3x3 within-team
    unshared second moment, v_q). All MoM on the training rows given."""
    if w is None:
        w = np.ones(len(R))
    if arm == "N":
        S = np.zeros((3, 3))
    elif arm in ("G1", "GP"):
        S = np.full((3, 3), max(sigma_pooled(R, W, w), 0.0))
    elif arm == "G3":
        S = psd(sigma_between(R, W, w))
    else:
        raise KeyError(arm)
    T = psd(within_cross(R, W, w) - S)
    ok = np.isfinite(q)
    vq = float((w[ok] * q[ok] ** 2).sum() / w[ok].sum())
    c = np.zeros(3)
    if arm == "GP":
        rs = R.sum(axis=(1, 2))
        ws = W.sum(axis=(1, 2))
        cc = float((w[ok] * rs[ok] * q[ok]).sum() / (w[ok] * ws[ok]).sum())
        # a latent correlation cannot exceed 1: |c| <= sigma * sqrt(vq)
        lim = np.sqrt(S[0, 0] * vq)
        c[:] = np.clip(cc, -lim, lim)
    return S, c, T, vq


def game_ll(R, W, q, S, c, T, vq):
    """Per-game Gaussian log density of x = [R_h(3), R_a(3), q]."""
    G = len(R)
    V = np.zeros((G, 7, 7))
    for i in range(2):
        Wi = W[:, i]
        V[:, 3 * i:3 * i + 3, 3 * i:3 * i + 3] = (np.einsum("gk,kl,gl->gkl", Wi, S + T, Wi)
                                                  + np.einsum("gk,kl->gkl", Wi, np.eye(3)))
    V[:, 0:3, 3:6] = np.einsum("gk,kl,gl->gkl", W[:, 0], S, W[:, 1])
    V[:, 3:6, 0:3] = np.transpose(V[:, 0:3, 3:6], (0, 2, 1))
    for i in range(2):
        V[:, 3 * i:3 * i + 3, 6] = W[:, i] * c[None, :]
        V[:, 6, 3 * i:3 * i + 3] = V[:, 3 * i:3 * i + 3, 6]
    V[:, 6, 6] = vq
    V += np.eye(7)[None] * 1e-6
    x = np.concatenate([R[:, 0], R[:, 1], q[:, None]], axis=1)
    ok = np.isfinite(x).all(1)
    sign, logdet = np.linalg.slogdet(V[ok])
    sol = np.linalg.solve(V[ok], x[ok][..., None])[..., 0]
    ll = np.full(G, np.nan)
    ll[ok] = -0.5 * (logdet + (x[ok] * sol).sum(1) + 7 * np.log(2 * np.pi))
    return ll


def bakeoff(games: pd.DataFrame) -> dict:
    rep: dict = {"created_at": pd.Timestamp.now("UTC").isoformat(), "folds": {}}
    for fold, spec in FOLDS.items():
        tr = games[games["season"].isin(spec["train"])]
        te = games[games["season"] == spec["test"]]
        a, b, _ = fit_pace(tr)
        qtr = (tr["lnN"] - (a + b * tr["tempo_asof"])).to_numpy()
        qte = (te["lnN"] - (a + b * te["tempo_asof"])).to_numpy()
        Rtr, Wtr = arrays(tr, centre=True)
        Rte, Wte = arrays(te, centre=True)
        fits, lls, fits_b1 = {}, {}, {}
        BWtr = boot_w(len(Rtr), 1, seed=1)[0]
        for arm in ARMS:
            S, c, T, vq = fit_arm(arm, Rtr, Wtr, qtr)
            fits[arm] = {"Sigma": S.tolist(), "c": c.tolist(), "T": T.tolist(), "vq": vq}
            lls[arm] = game_ll(Rte, Wte, qte, S, c, T, vq)
            # spec-identical refit under another seed: a training-game bootstrap draw (seed 1)
            S1, c1, T1, vq1 = fit_arm(arm, Rtr, Wtr, qtr, BWtr)
            fits_b1[arm] = game_ll(Rte, Wte, qte, S1, c1, T1, vq1)
        ok = np.isfinite(np.column_stack([lls[a_] for a_ in ARMS])).all(1)
        base = lls["N"][ok]
        BW = boot_w(int(ok.sum()), N_BOOT, seed=2)
        res = {}
        for arm in ARMS:
            d = lls[arm][ok] - base
            dboot = (BW @ d) / BW.sum(1)
            refit_gap = float(abs(np.mean(lls[arm][ok]) - np.mean(fits_b1[arm][ok])))
            floor = max(float(dboot.std()), refit_gap)
            # held-out covariance calibration: observed / predicted between-team cross moments
            S = np.array(fits[arm]["Sigma"])
            obs = np.einsum("gk,gl->kl", Rte[:, 0], Rte[:, 1])
            obs = 0.5 * (obs + obs.T)
            pred = np.einsum("gk,kl,gl->kl", Wte[:, 0], S, Wte[:, 1])
            pred = 0.5 * (pred + pred.T)
            # dispersion: per-team-game variance of residual makes, observed / predicted, per type
            Tm = np.array(fits[arm]["T"])
            obs_v = (Rte ** 2).sum(axis=(0, 1))
            pred_v = np.einsum("gik,kk->k", Wte ** 2, np.diag(np.diag(S + Tm))) + Wte.sum(axis=(0, 1))
            # eFG dispersion in points units per team-game
            ap = Wte * PTS
            obs_pts = ((Rte * PTS).sum(2) ** 2).sum()
            pred_pts = (np.einsum("gik,kl,gil->", ap, S + Tm, ap)
                        + (Wte * PTS ** 2).sum())
            res[arm] = {"mean_ll_minus_N": float(d.mean()), "boot_sd": float(dboot.std()),
                        "refit_seed1_gap": refit_gap, "floor": floor,
                        "floors_vs_N": float(d.mean() / floor) if floor > 0 else 0.0,
                        "between_obs_total": float(obs.sum()), "between_pred_total": float(pred.sum()),
                        "between_obs_by_pair": obs.tolist(), "between_pred_by_pair": pred.tolist(),
                        "within_var_obs_over_pred_by_type": (obs_v / pred_v).tolist(),
                        "pts_var_obs_over_pred": float(obs_pts / pred_pts),
                        "fit": fits[arm]}
        # segments of the LL gain (test): site and month
        segs = {}
        for name, m in (("home_away", te["neutral"].to_numpy() == 0), ("neutral", te["neutral"].to_numpy() == 1)):
            mm = m[ok]
            segs[name] = {arm: float((lls[arm][ok][mm] - base[mm]).mean()) for arm in ARMS}
            segs[name]["n"] = int(mm.sum())
        for mth in (11, 12, 1, 2, 3):
            mm = (te["month"].to_numpy() == mth)[ok]
            segs[f"month_{mth}"] = {arm: float((lls[arm][ok][mm] - base[mm]).mean()) for arm in ARMS}
            segs[f"month_{mth}"]["n"] = int(mm.sum())
        rep["folds"][fold] = {"n_test_games": int(ok.sum()), "arms": res, "segments": segs,
                              "pace_line": [a, b]}
    # decision rule (section 1.6)
    f2 = rep["folds"]["F2"]["arms"]
    f1 = rep["folds"]["F1"]["arms"]
    elig = [a_ for a_ in ARMS if a_ != "N" and f2[a_]["mean_ll_minus_N"] > 2 * f2[a_]["floor"]
            and f1[a_]["mean_ll_minus_N"] > 0]
    if not elig:
        winner = "N"
    else:
        best = max(elig, key=lambda a_: f2[a_]["mean_ll_minus_N"])
        fl = max(f2[best]["floor"], 1e-12)
        tied = [a_ for a_ in elig if f2[best]["mean_ll_minus_N"] - f2[a_]["mean_ll_minus_N"] <= fl]
        winner = min(tied, key=lambda a_: SIMPLICITY[a_])
    rep["eligible"] = elig
    rep["winner"] = winner
    return rep


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "measure"
    t0 = time.time()
    games = load_games()
    print(f"[{time.time()-t0:6.1f}s] games {len(games)} by season "
          f"{games.groupby('season').size().to_dict()}", flush=True)
    rep = measure(games) if stage == "measure" else bakeoff(games)
    (OUT / f"{stage}_v1.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float)[:20000])
    print(f"[{time.time()-t0:6.1f}s] wrote {OUT / (stage + '_v1.json')}")


if __name__ == "__main__":
    main()
