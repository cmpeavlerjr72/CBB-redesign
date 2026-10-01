#!/usr/bin/env python
"""
grade_shared_shooting_loop_v1.py -- one grader for the shared-shooting closed loop
(docs/models/shared_shooting/experiments.md section 1.7), arm vs reference, paired.

    .venv/Scripts/python.exe scripts/grade_shared_shooting_loop_v1.py \
        --arm results/engine_v0/laneB_v3full_G3_off0_n25 \
        --ref results/engine_v0/v3full_S0_s200_o0_off0_n25 \
        --draws results/engine_v0/v3full_S0f1_s200_o1000_off1000_n25 ... \
        --out results/shared_shooting/loop_G3_s25.json

G5 lines are recomputed exactly as `gates.gate_g5` (same `build_grading_frame`,
verified truth), plus the G5 components the guardrails revision asks for (within-game
Var home, Var away, Cov; actual residual counterparts), the mechanism line
(within-game between-team covariance of made FG by type), eFG / make rates by type,
segments (site, month), per team (UNDERPOWERED), and both Decision-12 floors:
(a) SD of each line across the reference draws (ref + --draws, same seed count),
(b) paired game bootstrap SD of the arm-minus-ref delta.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CBB_TRUTH", "verified_v1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from cbb_sim.eval import gates as G  # noqa: E402

TYPES = (("rim", "fga2_rim", "fgm2_rim"), ("jump2", "fga2_jump", "fgm2_jump"), ("three", "fga3", "fgm3"))


def per_game(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-game sufficient statistics over seeds."""
    r = raw.copy()
    r["h"] = r["home_pts"].astype(float)
    r["a"] = r["away_pts"].astype(float)
    r["hh"], r["aa"], r["ha"] = r["h"] ** 2, r["a"] ** 2, r["h"] * r["a"]
    cols = {"n": ("h", "size"), "sh": ("h", "sum"), "sa": ("a", "sum"), "shh": ("hh", "sum"),
            "saa": ("aa", "sum"), "sha": ("ha", "sum")}
    for k, fa, fm in TYPES:
        for s in ("home", "away"):
            r[f"{s}_{fm}_"] = r[f"{s}_{fm}"].astype(float)
            r[f"{s}_{fa}_"] = r[f"{s}_{fa}"].astype(float)
            cols[f"{s}_{fm}"] = (f"{s}_{fm}_", "sum")
            cols[f"{s}_{fa}"] = (f"{s}_{fa}_", "sum")
    for k, fa, fm in TYPES:
        for l_, fa2, fm2 in TYPES:
            r[f"x_{k}_{l_}"] = r[f"home_{fm}"].astype(float) * r[f"away_{fm2}"].astype(float)
            cols[f"x_{k}_{l_}"] = (f"x_{k}_{l_}", "sum")
    return r.groupby("game_id").agg(**cols)


def lines(summary: pd.DataFrame, pg: pd.DataFrame, w: np.ndarray | None = None) -> dict:
    """G5 lines and components. `w` = per-game bootstrap weights aligned to summary."""
    s = summary.set_index("game_id")
    p = pg.loc[s.index]
    if w is None:
        w = np.ones(len(s))
    n = p["n"].to_numpy()
    mh, ma = p["sh"].to_numpy() / n, p["sa"].to_numpy() / n
    vh = (p["shh"].to_numpy() - n * mh ** 2) / (n - 1)
    va = (p["saa"].to_numpy() - n * ma ** 2) / (n - 1)
    cha = (p["sha"].to_numpy() - n * mh * ma) / (n - 1)
    vt, vm = vh + va + 2 * cha, vh + va - 2 * cha
    W = w / w.sum()

    def wsd(x):
        mu = (W * x).sum()
        return float(np.sqrt((W * (x - mu) ** 2).sum() * len(x) / (len(x) - 1)))
    rt = s["total"].to_numpy() - s["sim_total_mean"].to_numpy()
    rm = s["margin"].to_numpy() - s["sim_margin_mean"].to_numpy()
    rh = s["home_score"].to_numpy() - mh
    ra = s["away_score"].to_numpy() - ma
    out = {
        "total_sd_ratio": float((W * np.sqrt(vt)).sum() / wsd(rt)),
        "margin_sd_ratio": float((W * np.sqrt(vm)).sum() / wsd(rm)),
        "sim_var_home": float((W * vh).sum()), "sim_var_away": float((W * va).sum()),
        "sim_cov": float((W * cha).sum()),
        "act_resid_var_home": wsd(rh) ** 2, "act_resid_var_away": wsd(ra) ** 2,
        "act_resid_cov": float((W * (rh - (W * rh).sum()) * (ra - (W * ra).sum())).sum()),
        "mean_sim_total_sd": float((W * np.sqrt(vt)).sum()), "sd_resid_total": wsd(rt),
        "mean_sim_margin_sd": float((W * np.sqrt(vm)).sum()), "sd_resid_margin": wsd(rm),
    }
    # pooled row correlation (as gate_g5 computes it, over all (game, seed) rows), weighted by game
    wn = w * n
    Eh = (w * p["sh"]).sum() / wn.sum()
    Ea = (w * p["sa"]).sum() / wn.sum()
    Vh = (w * p["shh"]).sum() / wn.sum() - Eh ** 2
    Va = (w * p["saa"]).sum() / wn.sum() - Ea ** 2
    C = (w * p["sha"]).sum() / wn.sum() - Eh * Ea
    out["corr_sim"] = float(C / np.sqrt(Vh * Va))
    hs, as_ = s["home_score"].to_numpy(float), s["away_score"].to_numpy(float)
    mu_h, mu_a = (W * hs).sum(), (W * as_).sum()
    out["corr_act"] = float((W * (hs - mu_h) * (as_ - mu_a)).sum()
                            / np.sqrt((W * (hs - mu_h) ** 2).sum() * (W * (as_ - mu_a) ** 2).sum()))
    # mechanism: within-game between-team covariance of made FG by type (pooled over pairs)
    tot = 0.0
    for k, fa, fm in TYPES:
        for l_, fa2, fm2 in TYPES:
            c = (p[f"x_{k}_{l_}"].to_numpy() - p[f"home_{fm}"].to_numpy() * p[f"away_{fm2}"].to_numpy() / n) / (n - 1)
            out[f"mech_cov_{k}_{l_}"] = float((W * c).sum())
            tot += out[f"mech_cov_{k}_{l_}"]
    out["mech_cov_total"] = tot
    for k, fa, fm in TYPES:
        made = sum((w * p[f"{s_}_{fm}"]).sum() for s_ in ("home", "away"))
        att = sum((w * p[f"{s_}_{fa}"]).sum() for s_ in ("home", "away"))
        out[f"make_rate_{k}"] = float(made / att)
    fgm = sum((w * p[f"{s_}_{fm}"]).sum() for s_ in ("home", "away") for k, fa, fm in TYPES)
    fga = sum((w * p[f"{s_}_{fa}"]).sum() for s_ in ("home", "away") for k, fa, fm in TYPES)
    fg3 = sum((w * p[f"{s_}_fgm3"]).sum() for s_ in ("home", "away"))
    out["efg"] = float((fgm + 0.5 * fg3) / fga)
    out["mean_total"] = float((W * (mh + ma)).sum())
    out["n_games"] = int(len(s))
    return out


def load(d: str):
    games = pd.read_parquet(Path(d) / "games.parquet")
    summary, raw = G.build_grading_frame(games, 2025)
    return summary, per_game(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--draws", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    sa, pa = load(a.arm)
    sr, pr = load(a.ref)
    common = sorted(set(sa["game_id"]) & set(sr["game_id"]))
    sa = sa[sa["game_id"].isin(common)].sort_values("game_id").reset_index(drop=True)
    sr = sr[sr["game_id"].isin(common)].sort_values("game_id").reset_index(drop=True)
    L_a, L_r = lines(sa, pa), lines(sr, pr)
    rep = {"arm": a.arm, "ref": a.ref, "n_games": len(common), "arm_lines": L_a, "ref_lines": L_r,
           "delta": {k: L_a[k] - L_r[k] for k in L_a if isinstance(L_a[k], float)}}
    # (b) paired game bootstrap
    rng = np.random.default_rng(20260930)
    keys = list(rep["delta"])
    boots = {k: [] for k in keys}
    for _ in range(a.boot):
        w = rng.poisson(1.0, len(common)).astype(float)
        la, lr = lines(sa, pa, w), lines(sr, pr, w)
        for k in keys:
            boots[k].append(la[k] - lr[k])
    rep["boot_sd"] = {k: float(np.std(v)) for k, v in boots.items()}
    # (a) reference draws
    draw_lines = [L_r]
    for d in a.draws:
        sd_, pd_ = load(d)
        sd_ = sd_[sd_["game_id"].isin(common)].sort_values("game_id").reset_index(drop=True)
        draw_lines.append(lines(sd_, pd_))
    rep["draws"] = [a.ref] + list(a.draws)
    rep["draw_sd"] = {k: float(np.std([dl[k] for dl in draw_lines], ddof=1)) if len(draw_lines) > 1 else None
                      for k in keys}
    rep["floor"] = {k: max(rep["boot_sd"][k], rep["draw_sd"][k] or 0.0) for k in keys}
    rep["floors_moved"] = {k: (rep["delta"][k] / rep["floor"][k] if rep["floor"][k] > 0 else None) for k in keys}
    # segments: site and month (G5 lines only)
    seg = {}
    for name, m_a in (("home_away", sa["neutral"] == 0), ("neutral", sa["neutral"] == 1)):
        ids = set(sa.loc[m_a, "game_id"])
        la = lines(sa[sa["game_id"].isin(ids)], pa)
        lr = lines(sr[sr["game_id"].isin(ids)], pr)
        seg[name] = {k: {"arm": la[k], "ref": lr[k]} for k in
                     ("total_sd_ratio", "margin_sd_ratio", "corr_sim", "corr_act", "sim_cov", "mech_cov_total")}
        seg[name]["n_games"] = len(ids)
    for mth in (11, 12, 1, 2, 3):
        ids = set(sa.loc[sa["month"] == mth, "game_id"])
        if len(ids) < 50:
            continue
        la = lines(sa[sa["game_id"].isin(ids)], pa)
        lr = lines(sr[sr["game_id"].isin(ids)], pr)
        seg[f"month_{mth}"] = {k: {"arm": la[k], "ref": lr[k]} for k in
                               ("total_sd_ratio", "margin_sd_ratio", "corr_sim", "corr_act")}
        seg[f"month_{mth}"]["n_games"] = len(ids)
    rep["segments"] = seg
    # per game: change of within-game total SD and of the h/a covariance
    def pg_stats(s, p):
        p = p.loc[s["game_id"]]
        n = p["n"].to_numpy()
        mh, ma = p["sh"] / n, p["sa"] / n
        cha = (p["sha"] - n * mh * ma) / (n - 1)
        vh = (p["shh"] - n * mh ** 2) / (n - 1)
        va = (p["saa"] - n * ma ** 2) / (n - 1)
        return pd.DataFrame({"game_id": s["game_id"].to_numpy(), "cov": cha.to_numpy(),
                             "tsd": np.sqrt(vh + va + 2 * cha).to_numpy(),
                             "msd": np.sqrt(vh + va - 2 * cha).to_numpy(),
                             "home": s["home_team_id"].to_numpy(), "away": s["away_team_id"].to_numpy()})
    ga, gr = pg_stats(sa, pa), pg_stats(sr, pr)
    d = ga[["game_id", "home", "away"]].copy()
    for c in ("cov", "tsd", "msd"):
        d[c] = ga[c].to_numpy() - gr[c].to_numpy()
    rep["per_game"] = {c: {"mean": float(d[c].mean()), "median": float(d[c].median()),
                           "share_positive": float((d[c] > 0).mean())} for c in ("cov", "tsd", "msd")}
    long = pd.concat([d.rename(columns={"home": "team"}).drop(columns="away"),
                      d.rename(columns={"away": "team"}).drop(columns="home")])
    pt = long.groupby("team").agg(n=("tsd", "size"), tsd=("tsd", "mean"), msd=("msd", "mean"))
    pt = pt[pt["n"] >= 10]
    rep["per_team"] = {"teams": int(len(pt)), "share_tsd_up": float((pt["tsd"] > 0).mean()),
                       "share_msd_down": float((pt["msd"] < 0).mean()),
                       "tsd_quartiles": pt["tsd"].quantile([.25, .5, .75]).tolist(),
                       "note": "UNDERPOWERED per team as a G5 line; distribution of paired per-game changes only"}
    Path(a.out).write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_games", "delta", "floor", "floors_moved")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
