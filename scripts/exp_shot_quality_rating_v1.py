"""Round 3 Part C: stability and as-of next-game prediction (RAW, RAW_SHRUNK, QUALITY, CLASS).

Usage: python scripts/exp_shot_quality_rating_v1.py FIT_SEASON TEST_SEASON [--located-only]  (2024/2025 only)
--located-only is a post-hoc SENSITIVITY (not pre-registered): drops unlocated FGA from every arm and the target.
Spec: docs/models/shot_location/experiments.md R3.5-R3.7 (committed before any fit).
Inputs: round3/ep_{FIT}.parquet and ep_{TEST}.parquet from train_shot_grade_v1.py (each season's EP comes from the model
trained on the OTHER season). Writes round3/rating_{FIT}_to_{TEST}.json.
Vectorised numpy re-implementation of the partner c_ratings.py solver (same iteration order, 12 iterations).
"""
import sys, json, pathlib, itertools, time
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/models/shot_location/round3"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
KS = [0.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, np.inf]
MIN_GAMES = 5
NBOOT = 500
POWER = {"ACC", "Big 12", "Big East", "Big Ten", "SEC", "Pac-12"}
CLASSES = ["dunk", "layup", "tip", "jumper2", "three"]
ARMS = ["RAW", "RAW_SHRUNK", "QUALITY", "CLASS"]
CURVE_X, CURVE_Y = [0.0, 6.0, 12.0], [1.0, 0.56, 0.25]


# ---------------------------------------------------------------- data
def team_games(ep, ver="", located_only=False):
    d = ep[ep.located].copy() if located_only else ep.copy()
    assert set(d.shot_class.unique()) <= set(CLASSES), d.shot_class.unique()
    d["w"] = np.interp(d.abs_script_index.fillna(0), CURVE_X, CURVE_Y)
    d["ep2"], d["ep0"] = d["ep_L2" + ver], d["ep_L0" + ver]
    d["fga"] = 1.0
    for c in ("pts", "ep2", "ep0", "fga"):
        d["w" + c] = d.w * d[c]
    d["ex"] = d.pts - d.ep0
    d["wex"] = d.w * d.ex
    cols = ["fga", "wfga", "pts", "wpts", "ep2", "wep2", "ep0", "wep0"]
    for c in CLASSES:
        m = (d.shot_class == c).astype(float)
        d["fga_" + c], d["wfga_" + c] = m, m * d.w
        d["ex_" + c], d["wex_" + c] = m * d.ex, m * d.wex
        cols += ["fga_" + c, "wfga_" + c, "ex_" + c, "wex_" + c]
    keys = ["gameId", "team"]
    g = d.groupby(keys)[cols].sum()
    f = d.groupby(keys).agg(date=("date", "first"), opp=("opp_team", "first"), home=("home", "first"),
                            neutral=("neutral", "first"), conf=("team_conf", "first")).reset_index()
    g = f.merge(g.reset_index(), on=keys)
    g = g.sort_values(["date", "gameId", "team"], kind="mergesort").reset_index(drop=True)
    return g


class Season:
    def __init__(self, tg):
        self.tg = tg
        teams = pd.unique(pd.concat([tg.team, tg.opp]))
        code = {t: i for i, t in enumerate(teams)}
        self.nt = len(teams)
        self.T = tg.team.map(code).values.astype(int)
        self.O = tg.opp.map(code).values.astype(int)
        self.date = tg.date.values
        self.udates = np.unique(self.date)
        self.home = tg.home.values.astype(bool)
        self.neutral = tg.neutral.values.astype(bool)
        self.fga = tg.fga.values
        self.pts = tg.pts.values
        self.game = tg.gameId.values
        self.conf = tg.conf.values


def measures(S, weighted):
    p = "w" if weighted else ""
    tg = S.tg
    V = [tg[p + "pts"].values, tg[p + "ep2"].values, tg[p + "ep0"].values] + [tg[p + "ex_" + c].values for c in CLASSES]
    N = [tg[p + "fga"].values] * 3 + [tg[p + "fga_" + c].values for c in CLASSES]
    return np.column_stack(V), np.column_stack(N)


def solve(T, O, V, N, alpha, nt, iters=12):
    """Partner iterative offence/defence solve for every measure column at once."""
    m = V.shape[1]
    ns = N.sum(0)
    lg = np.divide(V.sum(0), ns, out=np.zeros(V.shape[1]), where=ns > 0)  # a class with no prior shots at all: league 0, zero weight
    y = np.divide(V, N, out=np.zeros_like(V), where=N > 0)
    off = np.tile(lg, (nt, 1))
    dfn = off.copy()
    den_o = np.column_stack([np.bincount(T, N[:, j], nt) for j in range(m)])
    den_d = np.column_stack([np.bincount(O, N[:, j], nt) for j in range(m)])
    for _ in range(iters):
        adj_o = y - alpha * (dfn[O] - lg)
        num = np.column_stack([np.bincount(T, adj_o[:, j] * N[:, j], nt) for j in range(m)])
        off = np.where(den_o > 0, num / np.where(den_o > 0, den_o, 1), lg)
        adj_d = y - alpha * (off[T] - lg)
        num = np.column_stack([np.bincount(O, adj_d[:, j] * N[:, j], nt) for j in range(m)])
        dfn = np.where(den_d > 0, num / np.where(den_d > 0, den_d, 1), lg)
    return off, dfn, den_o, den_d, lg


def build_config(S, alpha, weighted):
    """As-of component arrays for every predicted team-game (rows identical across configs)."""
    V, N = measures(S, weighted)
    rows, comp = [], []
    for day in S.udates:
        k = np.searchsorted(S.date, day, side="left")
        if k == 0:
            continue
        a, b = k, np.searchsorted(S.date, day, side="right")
        cnt = np.bincount(S.T[:k], minlength=S.nt)
        idx = np.arange(a, b)
        ok = (cnt[S.T[idx]] >= MIN_GAMES) & (cnt[S.O[idx]] >= MIN_GAMES)
        if not ok.any():
            continue
        idx = idx[ok]
        t, o = S.T[idx], S.O[idx]
        off, dfn, den_o, den_d, lg = solve(S.T[:k], S.O[:k], V[:k], N[:k], alpha, S.nt)
        nn = ~S.neutral[:k]
        hm = nn & S.home[:k]
        edge = S.pts[:k][hm].sum() / S.fga[:k][hm].sum() - S.pts[:k][nn].sum() / S.fga[:k][nn].sum()
        h = np.where(S.neutral[idx], 0.0, np.where(S.home[idx], edge, -edge))
        # class shares for the game mix (same prior games, same weights)
        Lc = N[:k, 3:].sum(0) / N[:k, 3:].sum()
        so = den_o[:, 3:] / np.maximum(den_o[:, 3:].sum(1, keepdims=True), 1e-12)
        sd = den_d[:, 3:] / np.maximum(den_d[:, 3:].sum(1, keepdims=True), 1e-12)
        mix = np.maximum(so[t] + sd[o] - Lc, 1e-6)
        mix = mix / mix.sum(1, keepdims=True)
        c = {"idx": idx, "lg": np.full(len(idx), lg[0]), "h": h,
             "raw_o": off[t, 0] - lg[0], "raw_d": dfn[o, 0] - lg[0], "n_o": den_o[t, 0], "n_d": den_d[o, 0],
             "q_o": off[t, 1] - lg[1], "q_d": dfn[o, 1] - lg[1],
             "mk_o": (off[t, 0] - off[t, 1]) - (lg[0] - lg[1]), "mk_d": (dfn[o, 0] - dfn[o, 1]) - (lg[0] - lg[1]),
             "c_o": off[t, 2] - lg[2], "c_d": dfn[o, 2] - lg[2],
             "cm_o": off[t, 3:] - lg[3:], "cm_d": dfn[o, 3:] - lg[3:],
             "cn_o": den_o[t, 3:], "cn_d": den_d[o, 3:], "mix": mix}
        comp.append(c)
    out = {k: np.concatenate([c[k] for c in comp]) for k in comp[0]}
    idx = out["idx"]
    out["y"] = S.pts[idx] / S.fga[idx]
    out["w"] = S.fga[idx]
    out["game"] = S.game[idx]
    return out


def shrink(n, K):
    if np.isinf(K):
        return np.zeros_like(n)
    return np.divide(n, n + K, out=np.zeros_like(n), where=(n + K) > 0)  # n_c = 0 with K = 0: no data, zero weight


def predict(C, arm, K):
    """Return (pred, offence term, defence term); pred = lg + O + D + h."""
    if arm == "RAW":
        O, D = C["raw_o"], C["raw_d"]
    elif arm == "RAW_SHRUNK":
        O, D = shrink(C["n_o"], K) * C["raw_o"], shrink(C["n_d"], K) * C["raw_d"]
    elif arm == "QUALITY":
        O = C["q_o"] + shrink(C["n_o"], K) * C["mk_o"]
        D = C["q_d"] + shrink(C["n_d"], K) * C["mk_d"]
    else:
        so, sd = shrink(C["cn_o"], K), shrink(C["cn_d"], K)
        O = C["c_o"] + (C["mix"] * so * C["cm_o"]).sum(1)
        D = C["c_d"] + (C["mix"] * sd * C["cm_d"]).sum(1)
    return C["lg"] + O + D + C["h"], O, D


def wmse(C, pred):
    return float(np.average((C["y"] - pred) ** 2, weights=C["w"]))


# ---------------------------------------------------------------- stats helpers
class Boot:
    def __init__(self, game, nboot=NBOOT):
        self.ug, self.gi = np.unique(game, return_inverse=True)
        rng = np.random.default_rng(0)
        self.W = np.array([np.bincount(rng.integers(0, len(self.ug), len(self.ug)), minlength=len(self.ug)) for _ in range(nboot)], dtype=float)

    def gain(self, C, err_ref, err_arm, mask=None):
        m = np.ones(len(err_ref), bool) if mask is None else mask
        d_g = np.bincount(self.gi[m], C["w"][m] * (err_ref[m] - err_arm[m]), len(self.ug))
        w_g = np.bincount(self.gi[m], C["w"][m], len(self.ug))
        bs = (self.W @ d_g) / np.maximum(self.W @ w_g, 1e-12)
        gain = d_g.sum() / w_g.sum()
        sd = float(np.std(bs))
        return float(gain), sd, float(gain / sd), int(m.sum())

    def slope(self, C, x):
        y, w, gi, n = C["y"], C["w"], self.gi, len(self.ug)
        S = [np.bincount(gi, v, n) for v in (w, w * x, w * y, w * x * x, w * x * y)]
        def sl(s):
            sw, sx, sy, sxx, sxy = s
            return (sxy / sw - sx / sw * sy / sw) / (sxx / sw - (sx / sw) ** 2)
        full = sl([s.sum() for s in S])
        bs = sl([self.W @ s for s in S])
        return float(full), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def err(C, pred):
    return (C["y"] - pred) ** 2


# ---------------------------------------------------------------- stability
def split_half(tg):
    g = tg.sort_values(["date", "gameId"]).copy()
    g["ex2"] = g.pts - g.ep2
    g["ex0"] = g.pts - g.ep0
    rows = []
    for side, key in (("offence", "team"), ("defence", "opp")):
        g["half"] = g.groupby(key).cumcount() % 2
        h = g.groupby([key, "half"])[["fga", "pts", "ep2", "ex2", "ep0", "ex0"]].sum().unstack().dropna()
        h = h[(h.fga[0] >= 100) & (h.fga[1] >= 100)]
        for name, col in (("raw", "pts"), ("quality_L2", "ep2"), ("making_L2", "ex2"), ("quality_L0", "ep0"), ("making_L0", "ex0")):
            a, b = h[col][0] / h.fga[0], h[col][1] / h.fga[1]
            r = float(np.corrcoef(a, b)[0, 1])
            n_half = float(h.fga.mean().mean())
            rows.append({"side": side, "measure": name, "n_teams": int(len(h)), "half_r": r, "full_rel": 2 * r / (1 + r),
                         "fga_to_rel_0.5": (n_half * (1 / r - 1)) if r > 0 else None})
    return rows


# ---------------------------------------------------------------- main
def main(fit_s, test_s, located_only=False):
    tag = "_locOnly" if located_only else ""
    assert fit_s in (2024, 2025) and test_s in (2024, 2025) and fit_s != test_s
    t0 = time.time()
    ep_fit, ep_test = pd.read_parquet(OUT / f"ep_{fit_s}.parquet"), pd.read_parquet(OUT / f"ep_{test_s}.parquet")
    SF, ST = Season(team_games(ep_fit, "", located_only)), Season(team_games(ep_test, "", located_only))
    ST1 = Season(team_games(ep_test, "_s1", located_only))
    res = {"fit": fit_s, "test": test_s, "located_only": located_only, "n_tg_fit": len(SF.tg), "n_tg_test": len(ST.tg)}
    res["stability_fit_season"] = split_half(SF.tg)
    res["stability_test_season"] = split_half(ST.tg)
    print("stability done", round(time.time() - t0), flush=True)

    # ---- FIT: full grid
    cfgF = {(a, w): build_config(SF, a, w) for a, w in itertools.product(ALPHAS, (False, True))}
    print("fit configs built", round(time.time() - t0), flush=True)
    grid = {arm: {} for arm in ARMS}
    for (a, w), C in cfgF.items():
        r0 = wmse(C, predict(C, "RAW", 0)[0])
        for K in KS:
            grid["RAW"][(a, w, K)] = r0                       # padded: identical across K (R3.5)
            for arm in ARMS[1:]:
                grid[arm][(a, w, K)] = wmse(C, predict(C, arm, K)[0])
        # identities
        assert abs(wmse(C, predict(C, "RAW_SHRUNK", 0)[0]) - r0) < 1e-9
        assert abs(wmse(C, predict(C, "QUALITY", 0)[0]) - r0) < 1e-9, (wmse(C, predict(C, "QUALITY", 0)[0]), r0)
    for arm in ARMS:
        assert len(grid[arm]) == 70
    assert all(np.isfinite(v) for g in grid.values() for v in g.values()), 'NaN in FIT grid'
    chosen = {arm: min(sorted(grid[arm]), key=lambda k: grid[arm][k]) for arm in ARMS}
    res["fit_grid"] = {arm: {f"a{a}_w{int(w)}_K{K}": v for (a, w, K), v in g.items()} for arm, g in grid.items()}
    res["chosen"] = {arm: {"alpha": k[0], "weights": k[1], "K": "inf" if np.isinf(k[2]) else k[2], "fit_mse": grid[arm][k]} for arm, k in chosen.items()}
    print("chosen", json.dumps(res["chosen"]), flush=True)

    # ---- TEST
    cache, cacheS1 = {}, {}
    def cfg(S, store, a, w):
        if (a, w) not in store:
            store[(a, w)] = build_config(S, a, w)
        return store[(a, w)]
    preds, terms, Cs = {}, {}, {}
    for arm in ARMS:
        a, w, K = chosen[arm]
        C = cfg(ST, cache, a, w)
        Cs[arm] = C
        preds[arm], O, D = predict(C, arm, K)
        terms[arm] = (O, D)
    base = Cs["RAW_SHRUNK"]
    for arm in ARMS:
        assert np.array_equal(Cs[arm]["game"], base["game"]) and np.array_equal(Cs[arm]["idx"], base["idx"])
    B = Boot(base["game"])
    E = {arm: err(base, preds[arm]) for arm in ARMS}
    res["test_mse"] = {arm: wmse(base, preds[arm]) for arm in ARMS}
    res["n_test_rows"] = int(len(base["y"]))
    res["gain_vs_RAW_SHRUNK"] = {arm: dict(zip(("gain", "floor", "floors", "n"), B.gain(base, E["RAW_SHRUNK"], E[arm]))) for arm in ARMS if arm != "RAW_SHRUNK"}
    res["gain_vs_RAW"] = {arm: dict(zip(("gain", "floor", "floors", "n"), B.gain(base, E["RAW"], E[arm]))) for arm in ("RAW_SHRUNK", "QUALITY", "CLASS")}
    res["QUALITY_vs_CLASS"] = dict(zip(("gain", "floor", "floors", "n"), B.gain(base, E["CLASS"], E["QUALITY"])))  # positive = QUALITY better

    # ---- seed-1 check
    seed1 = {}
    for arm in ("QUALITY", "CLASS"):
        a, w, K = chosen[arm]
        C1 = cfg(ST1, cacheS1, a, w)
        assert np.array_equal(C1["idx"], base["idx"])
        p1, _, _ = predict(C1, arm, K)
        seed1[arm] = dict(zip(("gain", "floor", "floors", "n"), B.gain(base, E["RAW_SHRUNK"], err(base, p1))))
        seed1[arm]["mse_seed1"] = wmse(base, p1)
    seed1["QUALITY_vs_CLASS"] = None
    res["seed1_gain_vs_RAW_SHRUNK"] = seed1
    res["seed_shift_floors"] = {arm: abs(seed1[arm]["floors"] - res["gain_vs_RAW_SHRUNK"][arm]["floors"]) for arm in ("QUALITY", "CLASS")}

    # ---- segments
    idx = base["idx"]
    month = pd.to_datetime(ST.tg.date.values[idx]).month
    early = np.isin(month, [11, 12])
    power = np.array([c in POWER for c in ST.conf[idx]])
    segs = {"month_NovDec": early, "month_JanApr": ~early, "tier_power": power, "tier_other": ~power}
    res["segments"] = {}
    for sname, m in segs.items():
        res["segments"][sname] = {"n": int(m.sum()), "underpowered": bool(m.sum() < 2000)}
        for arm in ("QUALITY", "CLASS"):
            res["segments"][sname][arm] = dict(zip(("gain", "floor", "floors", "n"), B.gain(base, E["RAW_SHRUNK"], E[arm], m)))
        res["segments"][sname]["QUALITY_vs_CLASS"] = dict(zip(("gain", "floor", "floors", "n"), B.gain(base, E["CLASS"], E["QUALITY"], m)))
    # offence-only / defence-only (other term set to league)
    res["side_only"] = {}
    for side, j in (("offence_only", 0), ("defence_only", 1)):
        e = {arm: err(base, base["lg"] + terms[arm][j] + base["h"]) for arm in ARMS}
        res["side_only"][side] = {"mse": {arm: float(np.average(e[arm], weights=base["w"])) for arm in ARMS}}
        for arm in ("QUALITY", "CLASS"):
            res["side_only"][side][arm] = dict(zip(("gain", "floor", "floors", "n"), B.gain(base, e["RAW_SHRUNK"], e[arm])))
    # ---- responsiveness
    res["responsiveness"] = {}
    for arm in ARMS:
        p = preds[arm]
        q = pd.qcut(p, 5, labels=False, duplicates="drop")
        tab = [{"q": int(i), "n": int((q == i).sum()), "pred": float(np.average(p[q == i], weights=base["w"][q == i])),
                "actual": float(np.average(base["y"][q == i], weights=base["w"][q == i]))} for i in range(5)]
        sl, lo, hi = B.slope(base, p)
        act = [t["actual"] for t in tab]
        res["responsiveness"][arm] = {"quintiles": tab, "slope": sl, "slope_lo": lo, "slope_hi": hi,
                                      "monotone": bool(all(np.diff(act) > 0)), "clause": bool(all(np.diff(act) > 0) and 0.8 <= sl <= 1.2)}
    # ---- RAW alpha curve on TEST at RAW's chosen weights
    wsel = chosen["RAW"][1]
    res["raw_alpha_curve_test"] = {str(a): wmse(cfg(ST, cache, a, wsel), predict(cfg(ST, cache, a, wsel), "RAW", 0)[0]) for a in ALPHAS}
    res["raw_weights_for_curve"] = bool(wsel)

    # ---- pass rule (computed in code; both directions combined by the report aggregator)
    res["runtime_s"] = round(time.time() - t0)
    (OUT / f"rating_{fit_s}_to_{test_s}{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("chosen", "test_mse", "gain_vs_RAW_SHRUNK", "QUALITY_vs_CLASS", "seed_shift_floors")}, indent=1, default=float))
    print("runtime", res["runtime_s"], "s")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), "--located-only" in sys.argv)
