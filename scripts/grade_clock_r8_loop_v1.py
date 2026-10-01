"""grade_clock_r8_loop_v1.py -- clock round 8 closed-loop grader (experiments.md sections 37-39).

Lane H, 2026-10-01. Run with CBB_TRUTH=verified_v1 (refuses otherwise). ONE grader for every arm.

    --arm <tag or tag,tag,...>   arm run(s) (concatenated; seeds must not overlap)
    --control <tag>              default v3full_COMB9GCTKD_s200_o0 (served default, full size)
    --draws <tag,...>            optional extra served-stack seed-offset draws (full size: v3full_D2f<k>...)
    --label <name>

Pairing: the control is restricted to the arm's games and seeds. Seed-draw floor: the control's own
seed blocks of the arm's size (e.g. 25-seed blocks 0-24, 25-49, ...) plus any --draws, max pairwise
gap; the binding floor = max(seed-draw floor, 2 x paired game-bootstrap SE (200 draws)).

Lines (arm - control): G1 count - v4 count (pbp-complete two-team games); pooled possession SD;
possession SD ratio = sqrt(mean within-game var) / SD(actual - sim mean); between-game var vs the
calibrated corr^2 Var(actual); sim team pace slope (team quintile, as round 7) and game-prior slope;
game elasticity ratio (log sim mean on X = log rel_h + log rel_a with month FE, over actual's);
within-game possessions on (OREB, FGM, TOV, FTA) vs the actual residual regression (lane B's line);
within-game corr(N, PPP) vs the actual residual corr; G5 total / margin SD ratio; G9 total and
margin bias, MAE, MC-corrected slope; OT rate. Writes results/clock_r8/loop_grade_<label>.json
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
EV = ROOT / "results/engine_v0"
OUT = ROOT / "results/clock_r8"
COLS = ["game_id", "seed", "home_pts", "away_pts", "possessions", "n_periods", "home_oreb", "away_oreb",
        "home_fgm2_rim", "away_fgm2_rim", "home_fgm2_jump", "away_fgm2_jump", "home_fgm3", "away_fgm3",
        "home_tov", "away_tov", "home_fta", "away_fta"]


def load_run(tag: str) -> pd.DataFrame:
    parts = []
    for t in tag.split(","):
        f = EV / t / "games.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=COLS))
            continue
        chunks = sorted(glob.glob(str(EV / f"{t}_off*_n25" / "games.parquet")))
        if not chunks:
            raise SystemExit(f"missing run {t}")
        parts.append(pd.concat([pd.read_parquet(c, columns=COLS) for c in chunks], ignore_index=True))
    g = pd.concat(parts, ignore_index=True)
    if g.duplicated(["game_id", "seed"]).any():
        raise SystemExit(f"duplicate (game, seed) rows in {tag}")
    g["total"] = g["home_pts"] + g["away_pts"]
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["ot"] = (g["n_periods"] > 2).astype(float)
    g["oreb"] = g["home_oreb"] + g["away_oreb"]
    g["fgm"] = g[["home_fgm2_rim", "away_fgm2_rim", "home_fgm2_jump", "away_fgm2_jump", "home_fgm3", "away_fgm3"]].sum(axis=1)
    g["tov"] = g["home_tov"] + g["away_tov"]
    g["fta"] = g["home_fta"] + g["away_fta"]
    g["ppp"] = g["total"] / (2 * g["possessions"])
    return g


def truth_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = REF.load_actual_games(2025).set_index("game_id")
    p4 = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2025.parquet",
                         columns=["game_id", "offense_team_id", "oreb_count", "fgm_rim", "fgm_jump2", "fgm_3",
                                  "fta", "terminal_event"])
    nt = p4.groupby("game_id")["offense_team_id"].nunique()
    cnt_team = p4.groupby(["game_id", "offense_team_id"]).size().rename("cnt_team").reset_index()
    p4["fgm"] = p4["fgm_rim"] + p4["fgm_jump2"] + p4["fgm_3"]
    p4["tov"] = (p4["terminal_event"] == "TOV").astype(float)
    agg = p4.groupby("game_id").agg(a_oreb=("oreb_count", "sum"), a_fgm=("fgm", "sum"), a_tov=("tov", "sum"),
                                    a_fta=("fta", "sum"))
    t["cnt_v4"] = cnt_team.groupby("game_id")["cnt_team"].mean()
    t = t.join(agg)
    t["nt"] = nt
    u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet").set_index("game_id")
    t["pbp_complete"] = u["pbp_complete"].reindex(t.index)
    t["month"] = pd.to_datetime(t["game_date"]).dt.month
    a = np.load(ROOT / "data/processed/models/engine_v3/arrays_F2_2025.npz")
    gi = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")
    names = json.loads((ROOT / "data/processed/models/engine_v3/names_F2_2025.json").read_text())
    tn = names["team_names"] if isinstance(names["team_names"], dict) else json.loads(names["team_names"])
    ts = a["team_static"]
    X = pd.Series(np.log(ts[:, 0, tn["off_tempo_rel"]]) + np.log(ts[:, 1, tn["off_tempo_rel"]]), index=gi["game_id"].to_numpy())
    pr = pd.Series(ts[:, 0, tn["tempo_prior_game"]], index=gi["game_id"].to_numpy())
    t["X"] = X.reindex(t.index)
    t["gq"] = pd.qcut(pr.reindex(t.index), 5, labels=False)
    des = pd.read_parquet(ROOT / "data/processed/models/clock/r6_L2/design_v2.parquet",
                          columns=["season", "offense_team_id", "off_tempo_rel"])
    tm = des[des["season"] == 2025].groupby("offense_team_id")["off_tempo_rel"].mean()
    tq = pd.qcut(tm, 5, labels=False)
    cnt_team["tq"] = cnt_team["offense_team_id"].map(tq)
    return t, cnt_team.dropna(subset=["tq"])


def per_game(g: pd.DataFrame) -> pd.DataFrame:
    a = g.groupby("game_id").agg(poss=("possessions", "mean"), poss_var=("possessions", "var"),
                                 t_mean=("total", "mean"), t_sd=("total", "std"), m_mean=("margin", "mean"),
                                 m_sd=("margin", "std"), ot=("ot", "mean"), n_seeds=("seed", "nunique"),
                                 ppp=("ppp", "mean"), oreb=("oreb", "mean"), fgm=("fgm", "mean"),
                                 tov=("tov", "mean"), fta=("fta", "mean"))
    # within-game regressions need the seed rows: sufficient statistics per game
    dv = g[["possessions", "oreb", "fgm", "tov", "fta", "ppp"]] - g.groupby("game_id")[
        ["possessions", "oreb", "fgm", "tov", "fta", "ppp"]].transform("mean")
    cols = ["oreb", "fgm", "tov", "fta"]
    for i, c in enumerate(cols):
        a[f"xy_{c}"] = (dv[c] * dv["possessions"]).groupby(g["game_id"]).sum()
        for c2 in cols[i:]:
            a[f"xx_{c}_{c2}"] = (dv[c] * dv[c2]).groupby(g["game_id"]).sum()
    a["np_cov"] = (dv["possessions"] * dv["ppp"]).groupby(g["game_id"]).sum()
    a["nn"] = (dv["possessions"] ** 2).groupby(g["game_id"]).sum()
    a["pp"] = (dv["ppp"] ** 2).groupby(g["game_id"]).sum()
    return a


def ols_w(y, X, w):
    A = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return b


def lines(a: pd.DataFrame, T: pd.DataFrame, tg: pd.DataFrame, w: np.ndarray) -> dict:
    """a: per-game sim frame aligned with T (truth) rows; w: game weights (bootstrap)."""
    k = w > 0
    a, T, w = a[k], T[k], w[k]
    act = T["cnt_v4"].to_numpy(float)
    sim = a["poss"].to_numpy(float)
    out = {}
    out["g1_count_minus_v4"] = float(np.average(sim - act, weights=w))
    mv = float(np.average(a["poss_var"], weights=w))
    out["poss_sd_pooled"] = float(np.sqrt(np.average(a["poss_var"] + (sim - np.average(sim, weights=w)) ** 2, weights=w)))
    r = act - sim
    sd_r = float(np.sqrt(np.average((r - np.average(r, weights=w)) ** 2, weights=w)))
    out["poss_sd_ratio_within_over_resid"] = float(np.sqrt(mv) / sd_r)
    vs = float(np.average((sim - np.average(sim, weights=w)) ** 2, weights=w))
    va = float(np.average((act - np.average(act, weights=w)) ** 2, weights=w))
    cv = float(np.average((sim - np.average(sim, weights=w)) * (act - np.average(act, weights=w)), weights=w))
    corr = cv / np.sqrt(vs * va)
    out["between_var"] = vs
    out["between_var_calibrated"] = corr ** 2 * va
    out["count_cal_slope"] = cv / vs
    # game elasticity ratio
    M = pd.get_dummies(T["month"].astype(int), drop_first=True).to_numpy(float)
    Xg = np.column_stack([T["X"].to_numpy(float), M])
    out["elast_sim"] = float(ols_w(np.log(sim), Xg, w)[1])
    out["elast_act"] = float(ols_w(np.log(act), Xg, w)[1])
    out["elast_ratio"] = out["elast_sim"] / out["elast_act"]
    # game-prior quintile slope
    q = pd.DataFrame({"s": sim, "a": act, "w": w, "q": T["gq"].to_numpy()}).dropna()
    gq = q.groupby("q").apply(lambda d: pd.Series({"s": np.average(d["s"], weights=d["w"]),
                                                   "a": np.average(d["a"], weights=d["w"])}), include_groups=False)
    out["slope_gameq"] = float(np.polyfit(gq["a"], gq["s"], 1)[0])
    # team quintile slope (team-game sim possessions = game mean; actual team count)
    gw = pd.Series(w, index=T.index)
    t2 = tg[tg["game_id"].isin(T.index)].copy()
    t2["sim"] = t2["game_id"].map(a["poss"])
    t2["w"] = t2["game_id"].map(gw)
    tq = t2.groupby("tq").apply(lambda d: pd.Series({"s": np.average(d["sim"], weights=d["w"]),
                                                     "a": np.average(d["cnt_team"], weights=d["w"])}), include_groups=False)
    out["slope_teamq"] = float(np.polyfit(tq["a"], tq["s"], 1)[0])
    # within-game N on counts (pooled within-game normal equations)
    cols = ["oreb", "fgm", "tov", "fta"]
    XX = np.zeros((4, 4)); XY = np.zeros(4)
    for i, c in enumerate(cols):
        XY[i] = float(w @ a[f"xy_{c}"].to_numpy())
        for j, c2 in enumerate(cols):
            key = f"xx_{c}_{c2}" if j >= i else f"xx_{c2}_{c}"
            XX[i, j] = float(w @ a[key].to_numpy())
    b = np.linalg.solve(XX, XY)
    for i, c in enumerate(cols):
        out[f"sim_N_on_{c}"] = float(b[i])
    # actual: residual regression (actual minus the sim mean), N and counts
    Ra = np.column_stack([T[f"a_{c}"].to_numpy(float) - a[c].to_numpy(float) for c in cols])
    ba = ols_w(r, Ra, w)
    for i, c in enumerate(cols):
        out[f"act_N_on_{c}"] = float(ba[i + 1])
    out["sim_within_corr_N_ppp"] = float((w @ a["np_cov"].to_numpy()) / np.sqrt((w @ a["nn"].to_numpy()) * (w @ a["pp"].to_numpy())))
    e_act = (T["total"] / (2 * T["cnt_v4"])).to_numpy(float)
    re = e_act - a["ppp"].to_numpy(float)
    cre = np.average((r - np.average(r, weights=w)) * (re - np.average(re, weights=w)), weights=w)
    out["act_resid_corr_N_ppp"] = float(cre / np.sqrt(np.average((r - np.average(r, weights=w)) ** 2, weights=w)
                                                      * np.average((re - np.average(re, weights=w)) ** 2, weights=w)))
    # G5 / G9 / OT on all common games
    tot, mar = T["total"].to_numpy(float), T["margin"].to_numpy(float)
    rt = tot - a["t_mean"].to_numpy(); rm = mar - a["m_mean"].to_numpy()
    sdw = lambda v: float(np.sqrt(np.average((v - np.average(v, weights=w)) ** 2, weights=w)))  # noqa: E731
    out["g5_total_sd_ratio"] = float(np.average(a["t_sd"], weights=w)) / sdw(rt)
    out["g5_margin_sd_ratio"] = float(np.average(a["m_sd"], weights=w)) / sdw(rm)
    out["g9_total_bias"] = float(np.average(-rt, weights=w))
    out["g9_margin_bias"] = float(np.average(-rm, weights=w))
    out["g9_total_mae"] = float(np.average(np.abs(rt), weights=w))
    nse = float(np.average(a["n_seeds"], weights=w))
    for nm, mean, sdc, y in (("total", "t_mean", "t_sd", tot), ("margin", "m_mean", "m_sd", mar)):
        xm = a[mean].to_numpy(float)
        bb = ols_w(y, xm, w)[1]
        vp = sdw(xm) ** 2
        vmc = float(np.average(a[sdc] ** 2, weights=w)) / nse
        out[f"g9_{nm}_slope_mc"] = float(bb * vp / max(vp - vmc, 1e-9))
    out["ot_rate"] = float(np.average(a["ot"], weights=w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--control", default="v3full_COMB9GCTKD_s200_o0")
    ap.add_argument("--draws", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--nboot", type=int, default=200)
    args = ap.parse_args()
    T0, tg = truth_frame()
    arm = load_run(args.arm)
    ctl = load_run(args.control)
    seeds = sorted(arm["seed"].unique())
    games = sorted(set(arm["game_id"]) & set(ctl["game_id"]) & set(T0.index))
    ctl = ctl[ctl["game_id"].isin(games)]
    arm = arm[arm["game_id"].isin(games)]
    n_s = len(seeds)
    blocks = {}
    allc = sorted(ctl["seed"].unique())
    for i in range(0, len(allc) - n_s + 1, n_s):
        blk = allc[i:i + n_s]
        blocks[f"ctl_s{blk[0]}"] = ctl[ctl["seed"].isin(blk)]
    for t in [x for x in args.draws.split(",") if x]:
        d = load_run(t)
        blocks[t] = d[d["game_id"].isin(games)]
    ctl_p = ctl[ctl["seed"].isin(seeds)]
    if ctl_p["seed"].nunique() != n_s:
        raise SystemExit("control lacks the arm's seeds")
    T = T0.loc[games]
    pc = (T["nt"] == 2) & T["pbp_complete"].fillna(False).astype(bool) & T["cnt_v4"].notna() & T["X"].notna()
    T = T[pc]
    ag = {"arm": per_game(arm).loc[T.index], "control": per_game(ctl_p).loc[T.index]}
    for k, d in blocks.items():
        ag[k] = per_game(d).reindex(T.index)
    w1 = np.ones(len(T))
    L = {k: lines(v, T, tg, w1) for k, v in ag.items() if v.notna().all().all()}
    keys = list(L["arm"])
    bl = [k for k in L if k not in ("arm", "control")]
    seedfloor = {k: (max(abs(L[a][k] - L[b][k]) for i, a in enumerate(bl) for b in bl[i + 1:]) if len(bl) > 1 else np.nan)
                 for k in keys}
    rng = np.random.default_rng(20261001)
    boots = {k: [] for k in keys}
    for _ in range(args.nboot):
        w = np.bincount(rng.integers(0, len(T), len(T)), minlength=len(T)).astype(float)
        la, lc = lines(ag["arm"], T, tg, w), lines(ag["control"], T, tg, w)
        for k in keys:
            boots[k].append(la[k] - lc[k])
    rep = {"label": args.label, "arm": args.arm, "control": args.control, "n_games": int(len(T)), "seeds": [int(seeds[0]), int(seeds[-1])],
           "n_seeds": n_s, "seed_draw_blocks": bl, "lines": {k: L[k] for k in ("arm", "control")},
           "draw_lines": {k: L[k] for k in bl}, "diff": {}}
    for k in keys:
        se = float(np.std(boots[k]))
        fl = float(np.nanmax([seedfloor[k], 2 * se]))
        dd = L["arm"][k] - L["control"][k]
        rep["diff"][k] = {"diff": dd, "boot_se": se, "seed_floor": seedfloor[k], "floor": fl, "floors": dd / fl if fl else None}
        print(f"{k:34s} ctl {L['control'][k]:9.4f}  arm {L['arm'][k]:9.4f}  diff {dd:+8.4f}  floor {fl:.4f} ({dd / fl if fl else 0:+.1f})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"loop_grade_{args.label}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
