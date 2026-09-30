#!/usr/bin/env python
"""
exp_season_drift_anchor_v1.py -- season-drift anchor round 1: ONE anchor design
applied identically to rebound, shot_block and the free-throw technical rate
(ft_tech), with possession_outcome (po, `first`) as the control.

Pre-registration: `docs/models/season_drift/experiments.md` section 1, committed
(5003ce7) BEFORE this file was run. This script FITS and PREDICTS only; every
cell's predictions are scored by `scripts/grade_season_drift_anchor_v1.py`, one
grader for every arm and sub-model.

Arms (identical in every sub-model):
  R   pooled reference, unchanged
  C0  prior-season level carry, no in-season update           (offset)
  O   target relative to the as-of league level               (offset)
  P   prior-season carry + in-season update, n0 FITTED        (offset)
  W   recency weights, half-life 365 d fixed a priori         (weights)
  F   as-of league level as a feature                         (feature)
  T   CAUTIONARY trend extrapolation                          (offset)

Nothing here is adopted; nothing served is written. Outputs go to
`results/season_drift/round1/` (gitignored, HF-synced bulk).

    .venv/Scripts/python.exe scripts/exp_season_drift_anchor_v1.py --light
    .venv/Scripts/python.exe scripts/exp_season_drift_anchor_v1.py --heavy --jobs 3
    .venv/Scripts/python.exe scripts/exp_season_drift_anchor_v1.py --cells rebound:F2:R:0
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

SMOKE = os.environ.get("SDA_SMOKE") == "1"   # 5-tree code-path check, separate dir
OUT = _ROOT / ("results/season_drift/smoke" if SMOKE else "results/season_drift/round1")
PREDS = OUT / "preds"
FRAMES = OUT / "frames"
FOLDS = {"F1": {"train": [2022, 2023], "test": [2024]},
         "F2": {"train": [2022, 2023, 2024], "test": [2025]}}
ARMS = ["R", "C0", "O", "P", "W", "F", "T"]
OFFSET_ARMS = {"C0", "O", "P", "T"}
HALF_LIFE_DAYS = 365.0
N0_GRID = [0.0] + [float(10 ** (k / 2)) for k in range(2, 17)] + [float("inf")]
FT_PSEUDO_TRIPS = 5.0
SB_SHOOTER_K = 50.0          # shot_block round 1's fitted shooter shrinkage
EPS = 1e-9

_CACHE: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# 1. The anchor object (section 1.2): one implementation for every sub-model
# ===========================================================================
def _link(L: np.ndarray, kind: str) -> np.ndarray:
    L = np.asarray(L, dtype="float64")
    if kind == "binary":
        L = np.clip(L, 1e-6, 1 - 1e-6)
        return np.log(L / (1 - L))
    return np.log(np.clip(L, 1e-12, None))


def _anchor_ll(N: np.ndarray, D: np.ndarray, L: np.ndarray, kind: str) -> float:
    """Anchor-only log likelihood from daily aggregates. N (d,K), D (d,), L (d,K)."""
    if kind == "binary":
        Lc = np.clip(L[:, 0], 1e-9, 1 - 1e-9)
        return float((N[:, 0] * np.log(Lc) + (D - N[:, 0]) * np.log(1 - Lc)).sum())
    if kind == "multi":
        return float((N * np.log(np.clip(L, 1e-12, None))).sum())
    Lc = np.clip(L[:, 0], 1e-12, None)
    return float((N[:, 0] * np.log(Lc) - Lc * D).sum())


def build_anchor(season: np.ndarray, date: np.ndarray, num: np.ndarray, den: np.ndarray,
                 train_seasons: list[int], kind: str) -> tuple[dict, dict]:
    """Per-row anchor levels for every arm, from rows of the fold (train + test).

    num (n,K) target counts, den (n,) exposure. Returns (levels, meta) where
    levels[arm] is (n,K) for C0/O/P/T and `Lbar` (K,)."""
    num = np.asarray(num, dtype="float64")
    if num.ndim == 1:
        num = num[:, None]
    K = num.shape[1]
    den = np.asarray(den, dtype="float64")
    key = pd.DataFrame({"season": season.astype("int64"),
                        "date": pd.to_datetime(date).values.astype("datetime64[D]")})
    gid = key.groupby(["season", "date"], sort=True).ngroup().to_numpy()
    ng = int(gid.max()) + 1
    N = np.zeros((ng, K))
    for k in range(K):
        N[:, k] = np.bincount(gid, weights=num[:, k], minlength=ng)
    D = np.bincount(gid, weights=den, minlength=ng)
    gkey = key.assign(g=gid).drop_duplicates("g").sort_values("g")
    gs = gkey["season"].to_numpy()
    # cumulative strictly before the date, within season
    Nb = np.zeros_like(N)
    Db = np.zeros_like(D)
    for s in np.unique(gs):
        m = np.where(gs == s)[0]
        cn = np.cumsum(N[m], axis=0)
        cd = np.cumsum(D[m])
        Nb[m[1:]] = cn[:-1]
        Db[m[1:]] = cd[:-1]
    tr_mask = np.isin(gs, train_seasons)
    Lbar = N[tr_mask].sum(axis=0) / max(D[tr_mask].sum(), EPS)
    Lend = {int(s): N[gs == s].sum(axis=0) / max(D[gs == s].sum(), EPS) for s in np.unique(gs)}
    prior_g = np.vstack([Lend[int(s) - 1] if (int(s) - 1) in train_seasons else Lbar for s in gs])

    def blend(n0: float) -> np.ndarray:
        if np.isinf(n0):
            return prior_g.copy()
        with np.errstate(invalid="ignore", divide="ignore"):
            L = (Nb + n0 * prior_g) / (Db + n0)[:, None]
        empty = (Db + n0) <= 0
        L[empty] = prior_g[empty]
        return L

    # n0 fitted on TRAIN seasons that have a previous train season
    fit_g = np.array([(int(s) in train_seasons) and ((int(s) - 1) in train_seasons) for s in gs])
    ll = {}
    for n0 in N0_GRID:
        ll[n0] = _anchor_ll(N[fit_g], D[fit_g], blend(n0)[fit_g], kind)
    n0_star = max(ll, key=ll.get)

    # OLS trend over the train seasons' levels, per class
    xs = np.array(sorted(train_seasons), dtype="float64")
    ys = np.vstack([Lend[int(s)] for s in xs])
    coef = [np.polyfit(xs, ys[:, k], 1) for k in range(K)]
    T_g = np.column_stack([np.polyval(coef[k], gs.astype("float64")) for k in range(K)])
    if kind == "binary":
        T_g = np.clip(T_g, 1e-4, 1 - 1e-4)
    elif kind == "multi":
        T_g = np.clip(T_g, 1e-4, None)
        T_g = T_g / T_g.sum(axis=1, keepdims=True)
    else:
        T_g = np.clip(T_g, 1e-9, None)

    levels_g = {"C0": prior_g, "O": blend(0.0), "P": blend(n0_star), "T": T_g}
    levels = {a: v[gid] for a, v in levels_g.items()}
    test_seasons = [int(s) for s in np.unique(gs) if int(s) not in train_seasons]
    meta = {"kind": kind, "K": K, "Lbar": Lbar.round(6).tolist(),
            "Lend_train": {int(s): Lend[int(s)].round(6).tolist() for s in train_seasons},
            "prior_for_test": {s: (Lend[s - 1] if (s - 1) in train_seasons else Lbar).round(6).tolist()
                               for s in test_seasons},
            "n0_fitted": n0_star, "n0_fit_seasons": sorted({int(s) for s in gs[fit_g]}),
            "n0_ll": {str(k): round(v, 3) for k, v in ll.items()},
            "trend_slope_per_season": [round(float(c[0]), 7) for c in coef],
            "trend_test": {s: [round(float(np.polyval(coef[k], s)), 6) for k in range(K)]
                           for s in test_seasons}}
    return {"levels": levels, "Lbar": Lbar}, meta


def offsets_for(arm: str, anc: dict, kind: str) -> np.ndarray | None:
    if arm not in OFFSET_ARMS:
        return None
    return _link(anc["levels"][arm], kind) - _link(anc["Lbar"][None, :], kind)


def feature_cols_for(anc: dict, kind: str) -> np.ndarray:
    """Arm F: the as-of league level (day 0 = prior-season end) as input column(s)."""
    return (_link(anc["levels"]["O"], kind) - _link(anc["Lbar"][None, :], kind)).astype("float32")


def recency_weights(dates: pd.Series, ref: pd.Timestamp) -> np.ndarray:
    age = (pd.Timestamp(ref) - pd.to_datetime(dates)).dt.days.to_numpy().astype("float64")
    return np.exp2(-np.clip(age, 0, None) / HALF_LIFE_DAYS)


def prior_team_rate(season: np.ndarray, team: np.ndarray, num: np.ndarray, den: np.ndarray,
                    test_season: int) -> dict:
    """Each team's realised rate in the season BEFORE the test season (grading key)."""
    num = num if num.ndim == 2 else num[:, None]
    m = season == test_season - 1
    df = pd.DataFrame(num[m], columns=[f"n{k}" for k in range(num.shape[1])])
    df["d"] = den[m]
    df["team"] = team[m]
    g = df.groupby("team").sum()
    return {k: (g[f"n{k}"] / g["d"].clip(lower=EPS)) for k in range(num.shape[1])}


def season_levels(season, sub, num, den) -> dict:
    num = num if num.ndim == 2 else num[:, None]
    out = {}
    df = pd.DataFrame(num, columns=[f"n{k}" for k in range(num.shape[1])])
    df["d"] = den
    df["season"] = season
    df["sub"] = sub
    for (s, u), g in df.groupby(["season", "sub"]):
        out.setdefault(str(u), {})[int(s)] = [float(g[f"n{k}"].sum() / max(g["d"].sum(), EPS))
                                              for k in range(num.shape[1])]
    return out


# ===========================================================================
# 2. Linear models with offsets (shot_block K2, ft_tech Poisson GLM)
# ===========================================================================
def fit_glm(X, y, off, w, family: str, C: float = 1.0):
    """sum_i w_i * loss_i + 0.5/C ||b||^2, intercept unpenalised; features
    standardised on the (unweighted) train slice, as sklearn K2 did."""
    from scipy.optimize import minimize

    X = np.asarray(X, dtype="float64")
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd
    y = np.asarray(y, dtype="float64")
    off = np.zeros(len(y)) if off is None else np.asarray(off, dtype="float64")
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype="float64")
    p = Z.shape[1]

    def f(theta):
        b0, b = theta[0], theta[1:]
        eta = Z @ b + b0 + off
        if family == "binomial":
            loss = np.logaddexp(0.0, eta) - y * eta
            r = 1.0 / (1.0 + np.exp(-eta)) - y
        else:
            mu_ = np.exp(np.clip(eta, -50, 50))
            loss = mu_ - y * eta
            r = mu_ - y
        wr = w * r
        val = float((w * loss).sum() + 0.5 / C * (b @ b))
        g = np.empty(p + 1)
        g[0] = wr.sum()
        g[1:] = Z.T @ wr + b / C
        return val, g

    b0 = 0.0
    if family == "binomial":
        pr = np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6)
        b0 = float(np.log(pr / (1 - pr)) - np.average(off, weights=w))
    else:
        b0 = float(np.log(max((w * y).sum(), EPS) / max((w * np.exp(off)).sum(), EPS)))
    res = minimize(f, np.r_[b0, np.zeros(p)], jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-7, "ftol": 1e-13})
    return {"mu": mu, "sd": sd, "theta": res.x, "family": family,
            "converged": bool(res.success), "nit": int(res.nit), "msg": str(res.message)}


def predict_glm(m, X, off):
    Z = (np.asarray(X, dtype="float64") - m["mu"]) / m["sd"]
    eta = Z @ m["theta"][1:] + m["theta"][0] + (0.0 if off is None else off)
    if m["family"] == "binomial":
        return 1.0 / (1.0 + np.exp(-eta))
    return np.exp(eta)


# ===========================================================================
# 3. Sub-model adapters: data, anchor inputs, grading frame
# ===========================================================================
def _site(te_home, te_away):
    return np.where(te_home > 0, "home", np.where(te_away > 0, "away", "neutral"))


def load_rebound() -> dict:
    if "rebound" in _CACHE:
        return _CACHE["rebound"]
    from cbb_sim.models import rebound as RB
    feats = RB.feature_set("C_plus_state")
    cols = list(dict.fromkeys(feats + ["season", "game_date", "game_id", "off_team_id", "y",
                                       "miss_type", "site_home", "site_away"]))
    d = pd.read_parquet(_ROOT / "data/processed/models/rebound/round3/design_round3.parquet",
                        columns=cols)
    assert_not_sealed(d, context="season_drift rebound design")
    live = (d["y"].to_numpy() != RB.CLASS_INDEX["DEAD"]).astype("float64")
    num = ((d["y"].to_numpy() == RB.CLASS_INDEX["OREB"]).astype("float64") * live)
    out = {"d": d, "feats": feats, "num": num[:, None], "den": live, "kind": "binary",
           "model": "lgbm", "K_out": 3, "offset_cols": [RB.CLASS_INDEX["OREB"]],
           "team": d["off_team_id"].to_numpy(), "sub": d["miss_type"].astype(str).to_numpy()}
    _CACHE["rebound"] = out
    return out


def load_po() -> dict:
    if "po" in _CACHE:
        return _CACHE["po"]
    from cbb_sim.models import possession_outcome as PO
    feats = PO.feature_set("C_plus_state", "first")
    cols = list(dict.fromkeys(feats + ["season", "game_date", "game_id", "offense_team_id", "y",
                                       "population", "site_home", "site_away"]))
    d = pd.read_parquet(_ROOT / "data/processed/models/possession_outcome/round4/design_v4.parquet",
                        columns=cols)
    d = d[d["population"] == "first"].reset_index(drop=True)
    assert_not_sealed(d, context="season_drift po design")
    K = len(PO.CLASSES)
    num = np.zeros((len(d), K))
    num[np.arange(len(d)), d["y"].to_numpy().astype(int)] = 1.0
    out = {"d": d, "feats": feats, "num": num, "den": np.ones(len(d)), "kind": "multi",
           "model": "lgbm", "K_out": K, "offset_cols": list(range(K)),
           "team": d["offense_team_id"].to_numpy(), "sub": np.full(len(d), "all")}
    _CACHE["po"] = out
    return out


def load_shot_block() -> dict:
    if "shot_block" in _CACHE:
        return _CACHE["shot_block"]
    cache = OUT / "shot_block_design.parquet"
    import train_shot_block_v1 as SB
    if cache.exists():
        d = pd.read_parquet(cache)
    else:
        d, _ = SB.build_design()
        d["shooter_blocked_c"] = SB.shooter_block_rates(d, SB_SHOOTER_K)
        keep = list(dict.fromkeys(SB.KC + ["season", "game_date", "game_id", "def_team_id", "y",
                                          "miss_type", "month"]))
        d = d[keep].copy()
        tmp = cache.with_suffix(".tmp.parquet")
        d.to_parquet(tmp, index=False)
        os.replace(tmp, cache)
    assert_not_sealed(d, context="season_drift shot_block design")
    out = {"d": d, "feats": list(SB.KC), "num": d["y"].to_numpy().astype("float64")[:, None],
           "den": np.ones(len(d)), "kind": "binary", "model": "logit", "K_out": 1,
           "team": d["def_team_id"].to_numpy(), "sub": d["miss_type"].astype(str).to_numpy()}
    _CACHE["shot_block"] = out
    return out


def load_ft_tech() -> dict:
    """Team-game rows: verified technical trips COMMITTED by the team, exposure =
    team-chances (each chance exposes both teams), site from the offender's view."""
    if "ft_tech" in _CACHE:
        return _CACHE["ft_tech"]
    seasons = [2022, 2023, 2024, 2025]
    assert_not_sealed(seasons, context="season_drift ft_tech")
    u = pd.read_parquet(_ROOT / "data/processed/games_universe.parquet",
                        columns=["game_id", "season", "game_date", "home_team_id", "away_team_id",
                                 "neutral_site"])
    u = u[u["season"].isin(seasons)].drop_duplicates("game_id")
    ch = pd.concat([pd.read_parquet(_ROOT / f"data/processed/possessions_v2/chances_{s}.parquet",
                                    columns=["game_id", "season"]) for s in seasons])
    n_ch = ch.groupby(["season", "game_id"]).size().rename("exposure").reset_index()
    g = n_ch.merge(u, on=["game_id", "season"], how="inner")
    home = g.assign(team_id=g["home_team_id"], is_home=True)
    away = g.assign(team_id=g["away_team_id"], is_home=False)
    tg = pd.concat([home, away], ignore_index=True)
    tg["neutral_site"] = tg["neutral_site"].fillna(False).astype(bool)
    tg["site_home"] = ((~tg["neutral_site"]) & tg["is_home"]).astype("float32")
    tg["site_away"] = ((~tg["neutral_site"]) & (~tg["is_home"])).astype("float32")
    t = pd.read_parquet(_ROOT / "data/processed/models/free_throw/technical_target_verified_trips_v1.parquet",
                        columns=["season", "game_id", "beneficiary"])
    t = t[t["season"].isin(seasons)].merge(u[["game_id", "home_team_id", "away_team_id"]],
                                           on="game_id", how="inner")
    t["team_id"] = np.where(t["beneficiary"] == t["home_team_id"], t["away_team_id"],
                            t["home_team_id"])
    tc = t.groupby(["season", "game_id", "team_id"]).size().rename("y").reset_index()
    tg = tg.merge(tc, on=["season", "game_id", "team_id"], how="left")
    tg["y"] = tg["y"].fillna(0).astype("float64")
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    tg = tg.sort_values(["season", "game_date", "game_id", "team_id"], kind="stable").reset_index(drop=True)

    # league in-season as-of rate (NaN on day 0) and team in-season as-of counts
    day = tg.groupby(["season", "game_date"])[["y", "exposure"]].sum()
    cum = day.groupby(level=0).cumsum() - day
    lg = (cum["y"] / cum["exposure"].where(cum["exposure"] > 0)).rename("lg_in")
    tg = tg.merge(lg.reset_index(), on=["season", "game_date"], how="left")
    tg = tg.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    grp = tg.groupby(["season", "team_id"])
    tg["c_before"] = grp["y"].cumsum() - tg["y"]
    tg["e_before"] = grp["exposure"].cumsum() - tg["exposure"]
    # expected trips at the league in-season level of each prior game date
    tg["_exp_g"] = tg["exposure"] * tg["lg_in"].fillna(0.0)
    tg["E_before"] = tg.groupby(["season", "team_id"])["_exp_g"].cumsum() - tg["_exp_g"]
    ok = (tg["e_before"] > 0) & tg["lg_in"].notna()
    tg["x_team_in"] = np.where(ok, np.log((tg["c_before"] + FT_PSEUDO_TRIPS)
                                          / (tg["E_before"] + FT_PSEUDO_TRIPS)), 0.0)
    # prior season: team observed / expected at that season's league level
    ts = tg.groupby(["season", "team_id"])[["y", "exposure"]].sum().reset_index()
    ls = tg.groupby("season")[["y", "exposure"]].sum()
    ts["lg"] = ts["season"].map(ls["y"] / ls["exposure"])
    ts["x_team_prev"] = np.log((ts["y"] + FT_PSEUDO_TRIPS)
                               / (ts["exposure"] * ts["lg"] + FT_PSEUDO_TRIPS))
    ts["season"] = ts["season"] + 1
    tg = tg.merge(ts[["season", "team_id", "x_team_prev"]], on=["season", "team_id"], how="left")
    tg["x_team_prev"] = tg["x_team_prev"].fillna(0.0)
    tg = tg.sort_values(["season", "game_date", "game_id", "team_id"], kind="stable").reset_index(drop=True)
    feats = ["x_team_in", "x_team_prev", "site_home", "site_away"]
    out = {"d": tg, "feats": feats, "num": tg["y"].to_numpy()[:, None],
           "den": tg["exposure"].to_numpy().astype("float64"), "kind": "poisson",
           "model": "poisson", "K_out": 1, "team": tg["team_id"].to_numpy(),
           "sub": np.full(len(tg), "all")}
    _CACHE["ft_tech"] = out
    return out


LOADERS = {"rebound": load_rebound, "shot_block": load_shot_block, "ft_tech": load_ft_tech,
           "po": load_po}


def fold_data(model: str, fold: str) -> dict:
    key = (model, fold)
    if key in _CACHE:
        return _CACHE[key]
    A = LOADERS[model]()
    d = A["d"]
    spec = FOLDS[fold]
    assert_not_sealed(spec["train"], context=f"{fold} train")
    assert_not_sealed(spec["test"], context=f"{fold} test")
    in_fold = d["season"].isin(spec["train"] + spec["test"]).to_numpy()
    idx = np.where(in_fold)[0]
    df = d.iloc[idx].reset_index(drop=True)
    num, den = A["num"][idx], A["den"][idx]
    anc, meta = build_anchor(df["season"].to_numpy(), df["game_date"].to_numpy(), num, den,
                             spec["train"], A["kind"])
    tr = df["season"].isin(spec["train"]).to_numpy()
    te = df["season"].isin(spec["test"]).to_numpy()
    out = {"df": df, "num": num, "den": den, "anc": anc, "meta": meta, "tr": tr, "te": te,
           "team": A["team"][idx], "sub": A["sub"][idx]}
    _CACHE[key] = out
    write_frame(model, fold, out, A)
    return out


def write_frame(model: str, fold: str, F: dict, A: dict) -> None:
    """The grading frame for (model, fold): test rows only, arm-independent."""
    p = FRAMES / f"{model}_{fold}.parquet"
    if p.exists():
        return
    FRAMES.mkdir(parents=True, exist_ok=True)
    df, te = F["df"], F["te"]
    test_season = FOLDS[fold]["test"][0]
    t = pd.DataFrame({"season": df.loc[te, "season"].to_numpy(),
                      "game_id": df.loc[te, "game_id"].to_numpy(),
                      "game_date": pd.to_datetime(df.loc[te, "game_date"]).to_numpy(),
                      "team": F["team"][te], "sub": F["sub"][te], "den": F["den"][te]})
    t["site"] = _site(df.loc[te, "site_home"].to_numpy(), df.loc[te, "site_away"].to_numpy())
    num = F["num"][te]
    for k in range(num.shape[1]):
        t[f"y{k}"] = num[:, k]
    if model in ("rebound", "po"):
        t["cls"] = df.loc[te, "y"].to_numpy().astype("int16")
    pr = prior_team_rate(df["season"].to_numpy(), F["team"], F["num"], F["den"], test_season)
    for k, s in pr.items():
        t[f"prior{k}"] = t["team"].map(s).astype("float64")
    tmp = p.with_suffix(f".tmp{os.getpid()}.parquet")
    t.to_parquet(tmp, index=False)
    try:
        os.replace(tmp, p)
    except PermissionError:          # a sibling worker wrote the identical frame first
        if not p.exists():
            raise
        os.remove(tmp)
        return
    lv = season_levels(df["season"].to_numpy(), F["sub"], F["num"], F["den"])
    lv["__all__"] = season_levels(df["season"].to_numpy(), np.full(len(df), "all"),
                                  F["num"], F["den"])["all"]
    meta = {"anchor": F["meta"], "season_levels": lv, "kind": A["kind"],
            "created_at": pd.Timestamp.now("UTC").isoformat()}
    (FRAMES / f"{model}_{fold}.meta.json").write_text(json.dumps(meta, indent=1, default=str),
                                                      encoding="utf-8")


# ===========================================================================
# 4. One cell: fit on train, predict test
# ===========================================================================
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_cell(model: str, fold: str, arm: str, seed: int) -> str:
    tag = f"{model}_{fold}_{arm}_s{seed}"
    out_p = PREDS / f"{tag}.npy"
    if out_p.exists():
        return f"SKIP {tag}"
    PREDS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    A = LOADERS[model]()
    F = fold_data(model, fold)
    df, tr, te = F["df"], F["tr"], F["te"]
    kind = A["kind"]
    feats = list(A["feats"])
    X = df[feats].to_numpy(dtype="float32")
    if arm == "F":
        X = np.hstack([X, feature_cols_for(F["anc"], kind)])
    off = offsets_for(arm, F["anc"], kind)
    w = None
    if arm == "W":
        first_test = pd.to_datetime(df.loc[te, "game_date"]).min()
        w = recency_weights(df.loc[tr, "game_date"], first_test)
    info = {"model": model, "fold": fold, "arm": arm, "seed": seed,
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "n_features": int(X.shape[1])}

    if A["model"] == "lgbm":
        import lightgbm as lgb
        from cbb_sim.models import possession_outcome as PO
        from cbb_sim.models import rebound as RB
        params = dict((RB.LgbmArm.PARAMS if model == "rebound" else PO.LgbmArm.PARAMS))
        params["n_jobs"] = 1
        if SMOKE:
            params["n_estimators"] = 5
        K = A["K_out"]
        y = df.loc[tr, "y"].to_numpy().astype(int)
        init_tr = init_te = None
        if off is not None:
            full = np.zeros((len(df), K))
            for j, c in enumerate(A["offset_cols"]):
                full[:, c] = off[:, j]
            init_tr, init_te = full[tr], full[te]
        clf = lgb.LGBMClassifier(random_state=seed, **params)
        clf.fit(np.ascontiguousarray(X[tr]), y, sample_weight=w, init_score=init_tr)
        assert list(clf.classes_) == list(range(K)), clf.classes_
        if init_te is not None:
            raw = np.asarray(clf.predict(np.ascontiguousarray(X[te]), raw_score=True), dtype="float64")
            p = _softmax(raw + init_te)
        else:
            p = clf.predict_proba(np.ascontiguousarray(X[te]))
    else:
        family = "binomial" if A["model"] == "logit" else "poisson"
        y = F["num"][tr, 0]
        o_tr = None if off is None else off[tr, 0]
        o_te = None if off is None else off[te, 0]
        if family == "poisson":
            le = np.log(F["den"])
            o_tr = le[tr] + (0.0 if o_tr is None else o_tr)
            o_te = le[te] + (0.0 if o_te is None else o_te)
        m = fit_glm(X[tr], y, o_tr, w, family)
        info.update({"converged": m["converged"], "nit": m["nit"], "msg": m["msg"],
                     "coef_std": [round(float(v), 6) for v in m["theta"]]})
        p = predict_glm(m, X[te], o_te)
        if family == "poisson":
            p = p / F["den"][te]            # stored as a RATE per exposure
        p = p[:, None]
    info["fit_seconds"] = round(time.time() - t0, 1)
    info["created_at"] = pd.Timestamp.now("UTC").isoformat()
    info["anchor_meta"] = F["meta"] if arm in OFFSET_ARMS | {"F"} else None
    tmp = PREDS / f"{tag}.tmp.npy"
    np.save(tmp, np.asarray(p, dtype="float64"))
    os.replace(tmp, out_p)
    (PREDS / f"{tag}.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return f"DONE {tag} {info['fit_seconds']}s"


def _worker(cell):
    try:
        r = run_cell(*cell)
    except Exception as e:  # report, never hide
        import traceback
        r = f"FAIL {cell}: {e!r}\n{traceback.format_exc()}"
    print(f"[{time.strftime('%H:%M:%S')}] {r}", flush=True)
    return r


def heavy_cells() -> list[tuple]:
    c = [("rebound", "F2", a, 0) for a in ARMS]
    c += [("po", "F2", "R", 0), ("rebound", "F2", "R", 1), ("po", "F2", "R", 1)]
    c += [("rebound", "F1", a, 0) for a in ARMS]
    c += [("po", "F2", a, 0) for a in ARMS if a != "R"]
    c += [("po", "F1", a, 0) for a in ARMS]
    return c


def light_cells() -> list[tuple]:
    c = []
    for fold in ("F2", "F1"):
        for m in ("ft_tech", "shot_block"):
            c += [(m, fold, a, 0) for a in ARMS]
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--cells", default="", help="model:fold:arm:seed[,..]")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cells = []
    if args.cells:
        for s in args.cells.split(","):
            m, f, a, sd = s.split(":")
            cells.append((m, f, a, int(sd)))
    if args.light:
        cells += light_cells()
    if args.heavy:
        cells += heavy_cells()
    log(f"{len(cells)} cells, jobs={args.jobs}")
    if args.jobs <= 1:
        for c in cells:
            _worker(c)
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.jobs, backend="loky", batch_size=1, pre_dispatch="n_jobs")(
            delayed(_worker)(c) for c in cells)
    log("all cells finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
