#!/usr/bin/env python
"""
train_rebound_v3_round3.py -- L3 REBOUND round 3: the season-drift level, the
`blocked_f` engine feed, the prior-season carry, and Decision 9's mandatory arms.

Pre-registration: `docs/models/rebound/experiments.md` sections 9 (PROPOSED) and
10 (the PM's amendment), both COMMITTED before this file was written to run
anything. This script executes them and writes one JSON per cell; the renderer
`scripts/grade_rebound_round3_v1.py` turns those into the results tables. One
grading function scores every arm; no arm has a scoring path of its own.

STAGE 1 screens every arm on the `S0` static calendar on BOTH folds.
STAGE 2 re-runs the block leaders, the combined arm and the reference on the
SERVED `S1_weekly` calendar on fold 2. The decision is taken on stage-2 numbers.

Nothing here is adopted; no served default changes; `data/processed/models/
rebound/` is not overwritten -- every artifact goes to its `round3/` sibling.

    .venv/Scripts/python.exe scripts/train_rebound_v3_round3.py --stage 1 --folds F2
    .venv/Scripts/python.exe scripts/train_rebound_v3_round3.py --stage 2 \
        --arms A0B0C0,A4 --folds F2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_THREADS = os.environ.get("CBB_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from cbb_sim.features import conference as CF  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

DESIGN = _ROOT / "data/processed/models/rebound/round3/design_round3.parquet"
OUT_DIR = _ROOT / "data/processed/models/rebound/round3"
CELLS = OUT_DIR / "cells"
SEASONS = [2022, 2023, 2024, 2025]
FS = "C_plus_state"
N_JOBS = int(_THREADS)
MIN_CELL_N = 300
OREB, DREB, DEAD = RB.CLASS_INDEX["OREB"], RB.CLASS_INDEX["DREB"], RB.CLASS_INDEX["DEAD"]

#: the served arm's LightGBM hyperparameters, FIXED across every arm of this
#: round so the ladder compares arms and not tuning effort.
PARAMS = {**RB.LgbmArm.PARAMS, "n_jobs": N_JOBS}

# ---------------------------------------------------------------------------
# The arm table. `add`/`drop`/`sub` act on the served `C_plus_state` list.
# ---------------------------------------------------------------------------
ARMS: dict[str, dict] = {
    "A0B0C0": {"why": "reference: the served lgbm / C_plus_state, unchanged"},
    "A1": {"add": ["season_idx"], "why": "season index as a feature"},
    "A2_h60": {"weight": 60.0, "why": "exponential recency weights, 60-day half-life"},
    "A2_h120": {"weight": 120.0, "why": "exponential recency weights, 120-day half-life"},
    "A2_h240": {"weight": 240.0, "why": "exponential recency weights, 240-day half-life"},
    "A3_1s": {"window": 1, "why": "rolling window: most recent 1 season of training rows"},
    "A3_2s": {"window": 2, "why": "rolling window: most recent 2 seasons of training rows"},
    "A4": {"add": ["lg_oreb_asof_c"],
           "why": "within-season as-of league OREB% anchor (PM condition a-ii)"},
    "A5": {"offset": True,
           "why": "TREND EXTRAPOLATION: an OLS season-level offset fitted on the fold's "
                  "completed TRAIN seasons and extrapolated to the test season, carried as a "
                  "model offset (init_score) because a tree cannot extrapolate a feature"},
    "B1": {"drop": ["blocked_f"], "why": "drop blocked_f and refit; marginalise over blocks"},
    "C1": {"sub": {"off_oreb_c": "off_oreb_g2", "opp_def_dreb_c": "opp_def_dreb_g2"},
           "why": "prior-season carry, PO round 4's G2"},
    "C2": {"sub": {"off_oreb_c": "off_oreb_g3", "opp_def_dreb_c": "opp_def_dreb_g3"},
           "why": "prior-season carry with the prior shrunk by its own reliability, G3"},
    "C3": {"sub": {"off_oreb_c": "off_oreb_g1", "opp_def_dreb_c": "opp_def_dreb_g1"},
           "why": "shrink toward the league mean only, G1 (the control)"},
    "D1": {"sub": {"off_oreb_c": "off_oreb_oa1_c", "opp_def_dreb_c": "opp_def_dreb_oa1_c"},
           "why": "Decision 9a: opponent adjustment, one_pass"},
    "D2": {"sub": {"off_oreb_c": "off_oreb_oa2_c", "opp_def_dreb_c": "opp_def_dreb_oa2_c"},
           "why": "Decision 9a: opponent adjustment, iterative"},
    "D3": {"add": ["is_conf_game"], "why": "Decision 9b: conference-game flag"},
}
#: combined arms are assembled in stage 2 from the block leaders; the recipe is
#: `<A arm>+<C arm>`, and the parser below builds it from the ARMS entries so no
#: combined arm can contain anything the blocks did not.
COMBO_SEP = "+"


def arm_spec(name: str) -> dict:
    if name in ARMS:
        return ARMS[name]
    parts = name.split(COMBO_SEP)
    spec: dict = {"add": [], "drop": [], "sub": {}, "why": "combined: " + " + ".join(parts)}
    for p in parts:
        s = ARMS[p]
        spec["add"] += list(s.get("add", []))
        spec["drop"] += list(s.get("drop", []))
        spec["sub"].update(s.get("sub", {}))
        for k in ("weight", "window", "offset"):
            if k in s:
                spec[k] = s[k]
    return spec


def features_for(spec: dict) -> list[str]:
    base = RB.feature_set(FS)
    sub = spec.get("sub", {})
    out = [sub.get(c, c) for c in base if c not in set(spec.get("drop", []))]
    return out + list(spec.get("add", []))


# ---------------------------------------------------------------------------
# Fold-local columns (never built across folds)
# ---------------------------------------------------------------------------
def league_level_by_season(df: pd.DataFrame) -> dict[int, float]:
    live = df[df["y"] != DEAD]
    g = live.groupby("season")["y"].apply(lambda s: float((s == OREB).mean()))
    return {int(k): float(v) for k, v in g.items()}


def add_fold_columns(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """`lg_oreb_asof_c` (A4) and the A5 trend offset, both centred / fitted on the
    TRAIN slice only so a fold can never see another fold's fit."""
    mu = float(tr["lg_oreb_asof"].mean())
    tr = tr.assign(lg_oreb_asof_c=(tr["lg_oreb_asof"] - mu).astype("float32"))
    te = te.assign(lg_oreb_asof_c=(te["lg_oreb_asof"] - mu).astype("float32"))
    lv = league_level_by_season(tr)
    xs = np.array(sorted(lv), dtype="float64")
    ys = np.array([lv[int(s)] for s in xs], dtype="float64")
    b, a = np.polyfit(xs, ys, 1) if len(xs) > 1 else (0.0, float(ys[0]))
    base = float(np.clip((tr["y"] != DEAD).pipe(lambda m: (tr.loc[m, "y"] == OREB).mean()),
                         1e-6, 1 - 1e-6))

    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _off(seasons):
        pred = np.clip(a + b * np.asarray(seasons, dtype="float64"), 1e-4, 1 - 1e-4)
        return _logit(pred) - _logit(base)

    meta = {"trend_slope_per_season": round(float(b), 6),
            "trend_intercept": round(float(a), 6),
            "train_league_level_by_season": {int(k): round(v, 5) for k, v in lv.items()},
            "train_pooled_live_oreb": round(base, 6),
            "extrapolated_level_test": {int(s): round(float(a + b * s), 5)
                                        for s in sorted(te["season"].unique())},
            "lg_oreb_asof_train_mean": round(mu, 6),
            "lg_oreb_asof_test_in_train_support": bool(
                te["lg_oreb_asof"].min() >= tr["lg_oreb_asof"].min()
                and te["lg_oreb_asof"].max() <= tr["lg_oreb_asof"].max())}
    tr = tr.assign(_a5_off=_off(tr["season"].to_numpy()))
    te = te.assign(_a5_off=_off(te["season"].to_numpy()))
    return tr, te, meta


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_arm(spec: dict, pool: pd.DataFrame, feats: list[str], seed: int):
    import lightgbm as lgb

    p = pool
    if spec.get("window"):
        smax = int(p["season"].max())
        p = p[p["season"] > smax - int(spec["window"])]
    X = np.ascontiguousarray(p[feats].to_numpy(dtype="float32"))
    y = p["y"].to_numpy()
    w = None
    if spec.get("weight"):
        dmax = pd.to_datetime(p["game_date"]).max()
        age = (dmax - pd.to_datetime(p["game_date"])).dt.days.to_numpy().astype("float64")
        w = np.exp(-np.log(2.0) * age / float(spec["weight"]))
    init = None
    if spec.get("offset"):
        o = p["_a5_off"].to_numpy(dtype="float64")
        init = np.column_stack([o, np.zeros_like(o), np.zeros_like(o)])
    clf = lgb.LGBMClassifier(random_state=seed, **PARAMS)
    clf.fit(X, y, sample_weight=w, init_score=init)
    return clf, {"n_train": int(len(p)), "weighted": w is not None,
                 "offset": init is not None}


def predict_arm(spec: dict, clf, te: pd.DataFrame, feats: list[str]) -> np.ndarray:
    Xt = np.ascontiguousarray(te[feats].to_numpy(dtype="float32"))
    if spec.get("offset"):
        raw = clf.predict(Xt, raw_score=True)
        o = te["_a5_off"].to_numpy(dtype="float64")
        raw = np.asarray(raw, dtype="float64").copy()
        raw[:, OREB] += o
        return _softmax(raw)
    p = clf.predict_proba(Xt)
    out = np.zeros((len(te), 3), dtype="float64")
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, j]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def predict_with_feed(spec: dict, clf, te: pd.DataFrame, feats: list[str],
                      feed: np.ndarray | None) -> np.ndarray:
    if feed is None or "blocked_f" not in feats:
        return predict_arm(spec, clf, te, feats)
    t2 = te.copy()
    t2["blocked_f"] = np.asarray(feed, dtype="float32")
    return predict_arm(spec, clf, t2, feats)


# ---------------------------------------------------------------------------
# The S1 refit calendar (the SERVED one), reused verbatim from the S1 trainer
# ---------------------------------------------------------------------------
def s1_weekly_cuts(te_dates: pd.Series) -> list[pd.Timestamp]:
    monthly = PO.month_boundaries(te_dates)
    weekly = CF.weekly_boundaries(te_dates)
    return CF.union_boundaries(weekly, monthly[:1])


def fit_predict_scheme(spec, tr, te, feats, cuts, seed, feeds: dict):
    """Returns (p_true, {feed_name: p_feed}, segments)."""
    n = len(te)
    p_true = np.zeros((n, 3), dtype="float64")
    p_feed = {k: np.zeros((n, 3), dtype="float64") for k in feeds}
    segs = []
    if cuts is None:
        clf, m = fit_arm(spec, tr, feats, seed)
        p_true = predict_arm(spec, clf, te, feats)
        for k, v in feeds.items():
            p_feed[k] = predict_with_feed(spec, clf, te, feats, v)
        segs.append({"refit": str(pd.to_datetime(tr["game_date"]).max().date()), **m})
        return p_true, p_feed, segs
    te_dates = pd.to_datetime(te["game_date"])
    cols = list(dict.fromkeys(feats + ["y", "season", "game_date", "_a5_off"]))
    cols = [c for c in cols if c in te.columns]
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        sel = ((te_dates >= cut) if nxt is None
               else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not sel.any():
            continue
        before = (te_dates < cut).to_numpy()
        prior = te.loc[before, cols]
        pool = tr[cols] if not len(prior) else pd.concat([tr[cols], prior], ignore_index=True)
        clf, m = fit_arm(spec, pool, feats, seed)
        sub = te.loc[sel]
        p_true[sel] = predict_arm(spec, clf, sub, feats)
        for kk, v in feeds.items():
            p_feed[kk][sel] = predict_with_feed(spec, clf, sub, feats,
                                                None if v is None else np.asarray(v)[sel])
        segs.append({"refit": str(pd.Timestamp(cut).date()), "n_scored": int(sel.sum()), **m})
    if (p_true.sum(axis=1) == 0).any():
        raise AssertionError("refit calendar is not a cover of the test slice")
    return p_true, p_feed, segs


# ---------------------------------------------------------------------------
# ONE blind grader
# ---------------------------------------------------------------------------
def _binary_oreb(p: np.ndarray) -> np.ndarray:
    return p[:, OREB] / np.maximum(p[:, OREB] + p[:, DREB], 1e-12)


def _level_pp(te: pd.DataFrame, p: np.ndarray, mask: np.ndarray | None = None) -> float:
    live = (te["y"].to_numpy() != DEAD)
    if mask is not None:
        live = live & mask
    if live.sum() == 0:
        return float("nan")
    pb = _binary_oreb(p)[live]
    act = (te["y"].to_numpy()[live] == OREB).mean()
    return float(pb.mean() - act) * 100.0


def _seg_level(te, p, key: pd.Series, name: str) -> dict:
    out = {}
    live = (te["y"].to_numpy() != DEAD)
    pb = _binary_oreb(p)
    y = te["y"].to_numpy()
    kv = key.to_numpy()
    for k in pd.unique(kv):
        m = (kv == k) & live
        n = int(m.sum())
        cell = {"n": n}
        if n < MIN_CELL_N:
            cell["UNDERPOWERED"] = True
        else:
            cell["level_pp"] = round(float(pb[m].mean() - (y[m] == OREB).mean()) * 100, 4)
            cell["actual"] = round(float((y[m] == OREB).mean()), 5)
        out[str(k)] = cell
    return {"cut": name, "cells": out}


def team_quintile_slope(te: pd.DataFrame, p: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """The offline analogue of the G4 diagnostic's engine slope: teams bucketed by
    their PRIOR season's centred OREB%, predicted vs actual team OREB% span."""
    live = (te["y"].to_numpy() != DEAD)
    if mask is not None:
        live = live & mask
    d = pd.DataFrame({"team": te["off_team_id"].to_numpy()[live],
                      "prior": te["off_priorc"].to_numpy()[live],
                      "p": _binary_oreb(p)[live],
                      "a": (te["y"].to_numpy()[live] == OREB).astype("float64")})
    d = d[d["prior"] != 0.0]               # teams with no completed prior season
    g = d.groupby("team").agg(p=("p", "mean"), a=("a", "mean"), n=("a", "size"),
                              prior=("prior", "first")).reset_index()
    g = g[g["n"] >= 50]
    if len(g) < 50:
        return {"UNDERPOWERED": True, "n_teams": int(len(g))}
    g["q"] = pd.qcut(g["prior"].rank(method="first"), 5, labels=False)
    q = g.groupby("q").agg(p=("p", "mean"), a=("a", "mean"), n=("team", "size"))
    span_a = float(q["a"].iloc[-1] - q["a"].iloc[0])
    span_p = float(q["p"].iloc[-1] - q["p"].iloc[0])
    dp = np.diff(q["p"].to_numpy())
    return {"UNDERPOWERED": False, "n_teams": int(len(g)),
            "span_actual": round(span_a, 5), "span_pred": round(span_p, 5),
            "slope_ratio": round(span_p / span_a, 4) if abs(span_a) > 1e-9 else None,
            "monotone_steps": int((dp > 0).sum()),
            "gap_pp_by_q": [round(float(x) * 100, 3) for x in (q["p"] - q["a"]).tolist()],
            "actual_by_q": [round(float(x), 5) for x in q["a"].tolist()],
            "pred_by_q": [round(float(x), 5) for x in q["p"].tolist()],
            "n_teams_by_q": [int(x) for x in q["n"].tolist()]}


def grade(te: pd.DataFrame, p_true: np.ndarray, p_feed: dict, firsts: pd.DataFrame) -> dict:
    s = RB.score(te, p_true)
    month = pd.to_datetime(te["game_date"]).dt.month
    live = (te["y"].to_numpy() != DEAD)
    out = {
        "n": int(len(te)),
        "log_loss": round(s["log_loss"], 6),
        "brier": round(s["brier"], 6),
        "calib_pass": bool(s["calib_pass"]),
        "calib_worst_gap_pp": s["calib_worst_gap_pp"],
        "calib_worst_class": s["calib_worst_class"],
        "resp_pass": bool(s["resp_pass"]),
        "resp_min_steps": s["resp_min_steps"],
        "slope_off_oreb_c": s["responsiveness"]["off_oreb_c->OREB"]["slope_ratio"],
        "slope_opp_def_dreb_c": s["responsiveness"]["opp_def_dreb_c->OREB"]["slope_ratio"],
        "L1_level_pp": round(_level_pp(te, p_true), 4),
        "actual_live_oreb": round(float((te["y"].to_numpy()[live] == OREB).mean()), 6),
        "L2_by_feed_pp": {k: round(_level_pp(te, v), 4) for k, v in p_feed.items()},
        "log_loss_by_feed": {k: round(PM.log_loss(te["y"].to_numpy(), v), 6)
                             for k, v in p_feed.items()},
    }
    out["by_miss_type"] = _seg_level(te, p_true, te["miss_type"], "miss_type")
    bt = te["miss_type"].astype(str) + "|" + np.where(te["blocked"].to_numpy(), "blk", "unblk")
    out["by_miss_type_x_blocked"] = _seg_level(te, p_true, pd.Series(bt, index=te.index),
                                               "miss_type x blocked(true)")
    out["by_month"] = _seg_level(te, p_true, month.astype(str), "month")
    site = pd.Series(np.where(te["site_home"].to_numpy() > 0, "home",
                              np.where(te["site_away"].to_numpy() > 0, "away", "neutral")),
                     index=te.index)
    out["by_site"] = _seg_level(te, p_true, site, "site")
    conf = pd.Series(np.where(te["is_conf_game"].to_numpy() > 0, "conf", "nonconf"),
                     index=te.index)
    out["by_conf"] = _seg_level(te, p_true, conf, "conference game")
    out["by_period"] = _seg_level(te, p_true, te["period"].astype(int).clip(upper=5).astype(str),
                                  "period")
    gm = ((te["period"].to_numpy().clip(max=2) - 1) * 1200
          + (1200 - te["seconds_remaining"].to_numpy())) / 300.0
    out["by_game_minute"] = _seg_level(te, p_true,
                                       pd.Series(np.clip(gm, 0, 7).astype(int).astype(str),
                                                 index=te.index), "5-min game-minute bucket")

    # first four weeks of conference play (section 8's open item)
    t = te[["season", "off_team_id"]].reset_index(drop=True).copy()
    t["game_date"] = pd.to_datetime(te["game_date"]).to_numpy()
    t = t.merge(firsts.rename(columns={"team_id": "off_team_id"}),
                on=["season", "off_team_id"], how="left")
    wk = (t["game_date"] - t["first_conf_date"]).dt.days / 7.0
    c4 = ((wk >= 0) & (wk < 4)).fillna(False).to_numpy()
    out["conf4"] = ({"n": int(c4.sum()), "UNDERPOWERED": True} if c4.sum() < 1000 else
                    {"n": int(c4.sum()), "level_pp": round(_level_pp(te, p_true, c4), 4)})

    out["team_quintile"] = team_quintile_slope(te, p_true)
    mm = month.to_numpy()
    for label, m in (("Nov-Dec", np.isin(mm, [11, 12])), ("Jan", mm == 1),
                     ("Feb-Apr", np.isin(mm, [2, 3, 4]))):
        out[f"team_quintile_{label}"] = team_quintile_slope(te, p_true, m)

    # per-game level error
    gdf = pd.DataFrame({"g": te["game_id"].to_numpy()[live],
                        "p": _binary_oreb(p_true)[live],
                        "a": (te["y"].to_numpy()[live] == OREB).astype("float64")})
    gg = gdf.groupby("g").agg(p=("p", "mean"), a=("a", "mean"), n=("a", "size"))
    gg = gg[gg["n"] >= 10]
    dd = (gg["p"] - gg["a"]).to_numpy() * 100
    out["per_game"] = {"n_games": int(len(gg)), "mean_pp": round(float(dd.mean()), 4),
                       "median_pp": round(float(np.median(dd)), 4),
                       "sd_pp": round(float(dd.std()), 4),
                       "mae_pp": round(float(np.abs(dd).mean()), 4),
                       "p_sim_below_actual": round(float((dd < 0).mean()), 4)}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def block_feed(d: pd.DataFrame, fold: str, spec) -> np.ndarray | None:
    """`B3e`: the shot-block winner's predicted P(blocked) on the fold's TEST
    rows, fitted on the fold's TRAIN seasons only. Returns a full-length vector
    aligned to `d` (0.0 on free throws, which are never blocked)."""
    import train_shot_block_v1 as SB

    bd, _ = SB.build_design()
    bd["shooter_blocked_c"] = SB.shooter_block_rates(bd, 50.0)
    tr = bd[bd["season"].isin(SB.FOLDS[fold]["train"])]
    te = bd[bd["season"].isin(SB.FOLDS[fold]["test"])]
    p, _, _ = SB.fit_predict("K2", tr, te)
    key = ["game_id", "period", "seconds_remaining", "off_team_id", "score_diff"]
    k2 = te[key].copy()
    for c in ("period", "seconds_remaining", "score_diff"):
        k2[c] = k2[c].astype("int64")
    k2["_pblk"] = p
    dd = d[key].copy()
    for c in ("period", "seconds_remaining", "score_diff"):
        dd[c] = dd[c].astype("int64")
    k2 = k2.drop_duplicates(key, keep="first")
    m = dd.merge(k2, on=key, how="left")
    v = m["_pblk"].to_numpy()
    cov = float(np.isfinite(v).mean())
    v = np.where(np.isfinite(v), v, 0.0)
    v = np.where(d["miss_type"].to_numpy() == "ft", 0.0, v)
    print(f"  block feed coverage on fold {fold} test rows: {cov:.4f}", flush=True)
    return v.astype("float64")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--arms", default="")
    ap.add_argument("--folds", default="F2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feeds", action="store_true",
                    help="also score the B2 / B3e engine feeds for this arm")
    args = ap.parse_args()

    t0 = time.time()
    CELLS.mkdir(parents=True, exist_ok=True)
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="rebound round 3")

    d = pd.read_parquet(DESIGN)
    conf_all = CF.build_conference_flags(SEASONS)
    firsts = CF.first_conference_game_dates(conf_all)
    arms = [a for a in (args.arms.split(",") if args.arms else list(ARMS)) if a]
    folds = args.folds.split(",")
    print(f"[{time.time()-t0:.0f}s] design {d.shape}; stage {args.stage}; "
          f"arms {arms}; folds {folds}", flush=True)

    for fold in folds:
        tr0, te0 = RB.fold_slices(d, fold)
        tr, te, fmeta = add_fold_columns(tr0, te0)
        feeds: dict = {"B0_zero": np.zeros(len(te)),
                       "B2_asof_cell": te["blk_asof_cell"].to_numpy(dtype="float64")}
        if args.feeds:
            feeds["B3e_model"] = block_feed(te, fold, None)
            rng = np.random.default_rng(20260918)
            feeds["B3_draw"] = (rng.random(len(te)) < feeds["B3e_model"]).astype("float64")
        feeds["B_true"] = te["blocked_f"].to_numpy(dtype="float64")
        cuts = None if args.stage == 1 else s1_weekly_cuts(pd.to_datetime(te["game_date"]))
        for arm in arms:
            tag = f"s{args.stage}_{fold}_{arm}_seed{args.seed}"
            out_p = CELLS / f"{tag}.json"
            if out_p.exists():
                print(f"[{time.time()-t0:.0f}s] SKIP {tag} (exists)", flush=True)
                continue
            spec = arm_spec(arm)
            feats = features_for(spec)
            t1 = time.time()
            p_true, p_feed, segs = fit_predict_scheme(spec, tr, te, feats, cuts,
                                                      args.seed, feeds)
            g = grade(te, p_true, p_feed, firsts)
            g.update({"arm": arm, "fold": fold, "stage": args.stage, "seed": args.seed,
                      "scheme": "S0" if args.stage == 1 else "S1_weekly",
                      "features": feats, "n_features": len(feats),
                      "why": spec.get("why", ""), "n_fits": len(segs),
                      "fit_seconds": round(time.time() - t1, 1),
                      "fold_meta": fmeta,
                      "created_at": pd.Timestamp.now("UTC").isoformat()})
            out_p.write_text(json.dumps(g, indent=1, default=str), encoding="utf-8")
            print(f"[{time.time()-t0:.0f}s] {tag}: ll={g['log_loss']} L1={g['L1_level_pp']}pp "
                  f"L2={g['L2_by_feed_pp']} calib={'P' if g['calib_pass'] else 'F'}"
                  f"({g['calib_worst_gap_pp']}) resp={'P' if g['resp_pass'] else 'F'} "
                  f"slope={g['team_quintile'].get('slope_ratio')} "
                  f"{g['fit_seconds']}s", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
