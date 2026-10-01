#!/usr/bin/env python
"""exp_shared_poss_v1.py -- shared_possession round 1: shared per-game OREB / TOV latents (lane B, 2026-10-01).

Spec: docs/models/shared_possession/experiments.md section 1 (committed 5a62b8e BEFORE either stage ran).

    .venv/Scripts/python.exe scripts/exp_shared_poss_v1.py measure
    .venv/Scripts/python.exe scripts/exp_shared_poss_v1.py bakeoff

Per game, side and channel (oreb, tov): R = y - n p, W = n p (1 - p), p an as-of
expectation (league season-to-date + shrunk team offence / opponent defence log-odds
deviations + a train-season site term). Sigma by MoM on between-team cross products,
(season, ISO week, channel)-centred. Writes results/shared_possession/{measure,bakeoff}_v1.json
and, for the bake-off, data/processed/models/shared_possession/params_v1.json (fold-2 TRAIN fit).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from cbb_sim.eval import reference as REF  # noqa: E402

OUT = Path("results/shared_possession")
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = Path("data/processed/models/shared_possession/params_v1.json")
CHS = ("oreb", "tov")
SEASONS = (2023, 2024, 2025)
FOLDS = {"F1": {"train": [2023], "test": 2024}, "F2": {"train": [2023, 2024], "test": 2025}}
K_SHRINK = 200.0
N_BOOT = 200
ARMS = ("N", "O1", "T1", "OT")
SIMPLICITY = {"N": 0, "O1": 1, "T1": 1, "OT": 2}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------ data
def side_counts(season: int) -> pd.DataFrame:
    c = pd.read_parquet(f"data/processed/possessions_v2/chances_{season}.parquet",
                        columns=["game_id", "offense_team_id", "defense_team_id", "offense_is_home",
                                 "terminal_event", "start_reason", "chance_number"])
    c["tov"] = (c["terminal_event"] == "TOV").astype(float)
    c["oreb"] = (c["start_reason"] == "OREB").astype(float)
    c["dreb"] = (c["start_reason"] == "DREB").astype(float)
    g = c.groupby(["game_id", "offense_team_id", "defense_team_id"]).agg(
        n_ch=("tov", "size"), tov=("tov", "sum"), oreb=("oreb", "sum"), dreb_start=("dreb", "sum")).reset_index()
    # live rebounds of side i's misses = i's OREB starts + opponent's possessions starting DREB
    opp = g[["game_id", "offense_team_id", "dreb_start"]].rename(
        columns={"offense_team_id": "defense_team_id", "dreb_start": "opp_dreb_start"})
    g = g.merge(opp, on=["game_id", "defense_team_id"], how="left").fillna({"opp_dreb_start": 0.0})
    g["n_reb"] = g["oreb"] + g["opp_dreb_start"]
    return g


def load_games() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        fin = REF.load_actual_games(s, verified_finals=True)
        sc = side_counts(s)
        fin = fin[["game_id", "season", "game_date", "home_team_id", "away_team_id", "neutral"]].copy()
        h = sc.rename(columns={"offense_team_id": "home_team_id", "defense_team_id": "away_team_id"})
        a = sc.rename(columns={"offense_team_id": "away_team_id", "defense_team_id": "home_team_id"})
        cols = ["n_ch", "tov", "oreb", "n_reb"]
        g = fin.merge(h[["game_id", "home_team_id", "away_team_id"] + cols], on=["game_id", "home_team_id", "away_team_id"])
        g = g.merge(a[["game_id", "home_team_id", "away_team_id"] + cols], on=["game_id", "home_team_id", "away_team_id"],
                    suffixes=("_h", "_a"))
        rows.append(g)
    g = pd.concat(rows, ignore_index=True)
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g[(g["n_reb_h"] > 0) & (g["n_reb_a"] > 0) & (g["n_ch_h"] > 0) & (g["n_ch_a"] > 0)]
    g = g.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    iso = g["game_date"].dt.isocalendar()
    g["week"] = iso["week"].astype(int)
    g["iso_key"] = iso["year"].astype(int) * 100 + g["week"]
    g["month"] = g["game_date"].dt.month
    g["neutral"] = g["neutral"].astype(bool).astype(int)
    return g


def add_asof(g: pd.DataFrame) -> pd.DataFrame:
    """As-of team offence / opponent defence log-odds deviations per channel (prior dates only)."""
    g = g.copy()
    spec = {"oreb": ("oreb", "n_reb"), "tov": ("tov", "n_ch")}
    for ch, (y, n) in spec.items():
        parts = []
        for s, gs in g.groupby("season", sort=False):
            long = pd.concat([
                pd.DataFrame({"game_id": gs["game_id"], "date": gs["game_date"], "off": gs["home_team_id"],
                              "dfn": gs["away_team_id"], "y": gs[f"{y}_h"], "n": gs[f"{n}_h"], "side": "h"}),
                pd.DataFrame({"game_id": gs["game_id"], "date": gs["game_date"], "off": gs["away_team_id"],
                              "dfn": gs["home_team_id"], "y": gs[f"{y}_a"], "n": gs[f"{n}_a"], "side": "a"}),
            ]).sort_values(["date", "game_id"]).reset_index(drop=True)
            day = long.groupby("date")[["y", "n"]].sum().cumsum().shift(1)
            lg = (day["y"] / day["n"])
            lg = lg.fillna(lg.dropna().iloc[0] if lg.notna().any() else 0.3)
            long["L"] = long["date"].map(lg).to_numpy()
            # strictly earlier DATES for team cumulatives (a team plays at most once a day)
            for role in ("off", "dfn"):
                dd = long.groupby([role, "date"])[["y", "n"]].sum().groupby(level=0).cumsum()
                dd = dd.groupby(level=0).shift(1).fillna(0.0)
                idx = pd.MultiIndex.from_arrays([long[role], long["date"]])
                long[f"{role}_y"] = dd["y"].reindex(idx).to_numpy()
                long[f"{role}_n"] = dd["n"].reindex(idx).to_numpy()
            L = long["L"].to_numpy()
            d_off = logit((long["off_y"] + K_SHRINK * L) / (long["off_n"] + K_SHRINK)) - logit(L)
            d_def = logit((long["dfn_y"] + K_SHRINK * L) / (long["dfn_n"] + K_SHRINK)) - logit(L)
            long["base"] = logit(L) + d_off + d_def
            piv = long.pivot_table(index="game_id", columns="side", values="base")
            piv.columns = [f"base_{ch}_{c}" for c in piv.columns]
            parts.append(piv.reset_index())
        g = g.merge(pd.concat(parts, ignore_index=True), on="game_id", how="left")
    return g


def site_terms(tr: pd.DataFrame) -> dict:
    """Train-season home/away log-odds half-difference per channel (non-neutral games)."""
    t = tr[tr["neutral"] == 0]
    out = {}
    for ch, (y, n) in {"oreb": ("oreb", "n_reb"), "tov": ("tov", "n_ch")}.items():
        ph = t[f"{y}_h"].sum() / t[f"{n}_h"].sum()
        pa = t[f"{y}_a"].sum() / t[f"{n}_a"].sum()
        out[ch] = float(0.5 * (logit(ph) - logit(pa)))
    return out


def arrays(g: pd.DataFrame, site: dict, centre: bool = True):
    """R, W as (G, 2 sides, 2 channels)."""
    spec = {"oreb": ("oreb", "n_reb"), "tov": ("tov", "n_ch")}
    R = np.zeros((len(g), 2, 2))
    W = np.zeros((len(g), 2, 2))
    neu = g["neutral"].to_numpy() == 1
    for j, ch in enumerate(CHS):
        y, n = spec[ch]
        for i, s in enumerate(("h", "a")):
            sgn = (1.0 if s == "h" else -1.0) * np.where(neu, 0.0, 1.0)
            p = expit(g[f"base_{ch}_{s}"].to_numpy() + site[ch] * sgn)
            nn = g[f"{n}_{s}"].to_numpy(float)
            R[:, i, j] = g[f"{y}_{s}"].to_numpy(float) - nn * p
            W[:, i, j] = nn * p * (1 - p)
    if centre:
        key = (g["season"].astype(str) + "_" + g["week"].astype(str)).to_numpy()
        for kk in np.unique(key):
            m = key == kk
            c = R[m].sum(axis=(0, 1)) / np.maximum(W[m].sum(axis=(0, 1)), 1e-9)
            R[m] = R[m] - W[m] * c[None, None, :]
    return R, W


def sigma_between(R, W, w=None):
    if w is None:
        w = np.ones(len(R))
    num = np.einsum("g,gk,gl->kl", w, R[:, 0], R[:, 1])
    den = np.einsum("g,gk,gl->kl", w, W[:, 0], W[:, 1])
    return 0.5 * (num + num.T) / (0.5 * (den + den.T))


def within_cross(R, W, w=None):
    if w is None:
        w = np.ones(len(R))
    k = R.shape[2]
    num = np.einsum("g,gik,gil->kl", w, R, R) - np.diag(np.einsum("g,gik->k", w, W))
    den = np.einsum("g,gik,gil->kl", w, W, W)
    return num / den


def psd(S):
    v, Q = np.linalg.eigh(0.5 * (S + S.T))
    return (Q * np.clip(v, 0, None)) @ Q.T


def boot_w(G, n=N_BOOT, seed=0):
    return np.random.default_rng(seed).poisson(1.0, size=(n, G)).astype(float)


def fit_arm(arm, R, W, w=None):
    Sb = sigma_between(R, W, w)
    if arm == "N":
        S = np.zeros((2, 2))
    elif arm == "O1":
        S = np.diag([max(Sb[0, 0], 0.0), 0.0])
    elif arm == "T1":
        S = np.diag([0.0, max(Sb[1, 1], 0.0)])
    elif arm == "OT":
        S = psd(Sb)
    else:
        raise KeyError(arm)
    T = psd(within_cross(R, W, w) - S)
    return S, T


def ll4(R, W, S, T):
    G = len(R)
    V = np.zeros((G, 4, 4))
    for i in range(2):
        Wi = W[:, i]
        V[:, 2 * i:2 * i + 2, 2 * i:2 * i + 2] = (np.einsum("gk,kl,gl->gkl", Wi, S + T, Wi)
                                                  + np.einsum("gk,kl->gkl", Wi, np.eye(2)))
    V[:, 0:2, 2:4] = np.einsum("gk,kl,gl->gkl", W[:, 0], S, W[:, 1])
    V[:, 2:4, 0:2] = np.transpose(V[:, 0:2, 2:4], (0, 2, 1))
    V += np.eye(4)[None] * 1e-6
    x = np.concatenate([R[:, 0], R[:, 1]], axis=1)
    _, logdet = np.linalg.slogdet(V)
    sol = np.linalg.solve(V, x[..., None])[..., 0]
    return -0.5 * (logdet + (x * sol).sum(1) + 4 * np.log(2 * np.pi))


# ------------------------------------------------------------------ stages
def conf_flags(g):
    try:
        from cbb_sim.features import conference as CF
        cf = CF.build_conference_flags(list(SEASONS))[["game_id", "is_conf_game"]]
        return g.merge(cf, on="game_id", how="left").fillna({"is_conf_game": False})
    except Exception as e:  # noqa: BLE001
        print("conference flags unavailable:", e)
        g = g.copy()
        g["is_conf_game"] = np.nan
        return g


def measure(g):
    rep = {"created_at": pd.Timestamp.now("UTC").isoformat(), "seasons": {}, "segments": {}}
    for lab, seas in (("2023", [2023]), ("2024", [2024]), ("2025", [2025]), ("2023-24", [2023, 2024])):
        gg = g[g["season"].isin(seas)]
        site = site_terms(gg)
        R, W = arrays(gg, site)
        S = sigma_between(R, W)
        BW = boot_w(len(R), N_BOOT, seed=0)
        Sb = np.array([sigma_between(R, W, w) for w in BW])
        ex = within_cross(R, W)
        rep["seasons"][lab] = {"n_games": int(len(R)), "site_terms": site, "Sigma_between": S.tolist(),
                               "Sigma_between_SE": Sb.std(0).tolist(), "within_excess": ex.tolist(),
                               "mean_rate": {"oreb": float((gg["oreb_h"].sum() + gg["oreb_a"].sum()) /
                                                           (gg["n_reb_h"].sum() + gg["n_reb_a"].sum())),
                                             "tov": float((gg["tov_h"].sum() + gg["tov_a"].sum()) /
                                                          (gg["n_ch_h"].sum() + gg["n_ch_a"].sum()))},
                               "resid_corr_h_a": [float(np.corrcoef(R[:, 0, j], R[:, 1, j])[0, 1]) for j in range(2)],
                               "mean_W": W.mean(axis=(0, 1)).tolist()}
    for lab, seas in (("2025", [2025]), ("2023-24", [2023, 2024])):
        gg = g[g["season"].isin(seas)].reset_index(drop=True)
        site = site_terms(gg)
        R, W = arrays(gg, site)
        cuts = {"home_away": gg["neutral"] == 0, "neutral": gg["neutral"] == 1,
                "conference": gg["is_conf_game"] == True, "non_conference": gg["is_conf_game"] == False}  # noqa: E712
        for mth in (11, 12, 1, 2, 3):
            cuts[f"month_{mth}"] = gg["month"] == mth
        d = {}
        for name, m in cuts.items():
            m = m.to_numpy()
            if m.sum() < 150:
                d[name] = {"n": int(m.sum()), "note": "UNDERPOWERED"}
                continue
            BW = boot_w(int(m.sum()), 100, seed=3)
            S = sigma_between(R[m], W[m])
            Sb = np.array([sigma_between(R[m], W[m], w) for w in BW])
            d[name] = {"n": int(m.sum()), "diag": np.diag(S).tolist(), "diag_SE": np.diag(Sb.std(0)).tolist(),
                       "cross": float(S[0, 1])}
        rep["segments"][lab] = d
    return rep


def bakeoff(g):
    rep = {"created_at": pd.Timestamp.now("UTC").isoformat(), "folds": {}}
    for fold, spec in FOLDS.items():
        tr = g[g["season"].isin(spec["train"])].reset_index(drop=True)
        te = g[g["season"] == spec["test"]].reset_index(drop=True)
        site = site_terms(tr)
        Rtr, Wtr = arrays(tr, site)
        Rte, Wte = arrays(te, site)
        w1 = boot_w(len(Rtr), 1, seed=1)[0]
        lls, lls_b1, fits = {}, {}, {}
        for arm in ARMS:
            S, T = fit_arm(arm, Rtr, Wtr)
            fits[arm] = {"Sigma": S.tolist(), "T": T.tolist()}
            lls[arm] = ll4(Rte, Wte, S, T)
            S1, T1 = fit_arm(arm, Rtr, Wtr, w1)
            lls_b1[arm] = ll4(Rte, Wte, S1, T1)
        base = lls["N"]
        BW = boot_w(len(base), N_BOOT, seed=2)
        obs = 0.5 * (np.einsum("gk,gl->kl", Rte[:, 0], Rte[:, 1]) + np.einsum("gk,gl->kl", Rte[:, 1], Rte[:, 0]))
        res = {}
        for arm in ARMS:
            d = lls[arm] - base
            dboot = (BW @ d) / BW.sum(1)
            gap = float(abs(lls[arm].mean() - lls_b1[arm].mean()))
            floor = max(float(dboot.std()), gap)
            S = np.array(fits[arm]["Sigma"])
            T = np.array(fits[arm]["T"])
            pred = np.einsum("gk,kl,gl->kl", Wte[:, 0], S, Wte[:, 1])
            pred = 0.5 * (pred + pred.T)
            obs_v = (Rte ** 2).sum(axis=(0, 1))
            pred_v = np.einsum("gik,kk->k", Wte ** 2, np.diag(np.diag(S + T))) + Wte.sum(axis=(0, 1))
            res[arm] = {"gain_vs_N": float(d.mean()), "boot_sd": float(dboot.std()), "refit_seed1_gap": gap,
                        "floor": floor, "floors": float(d.mean() / floor) if floor > 0 else 0.0,
                        "between_obs": np.diag(obs).tolist(), "between_pred": np.diag(pred).tolist(),
                        "within_var_obs_over_pred": (obs_v / pred_v).tolist(), "fit": fits[arm]}
        segs = {}
        te = conf_flags(te) if "is_conf_game" not in te.columns else te
        for name, m in (("home_away", te["neutral"] == 0), ("neutral", te["neutral"] == 1),
                        ("conference", te["is_conf_game"] == True),  # noqa: E712
                        ("non_conference", te["is_conf_game"] == False)):  # noqa: E712
            m = m.to_numpy()
            segs[name] = {a: float((lls[a][m] - base[m]).mean()) for a in ARMS}
            segs[name]["n"] = int(m.sum())
        for mth in (11, 12, 1, 2, 3):
            m = (te["month"] == mth).to_numpy()
            segs[f"month_{mth}"] = {a: float((lls[a][m] - base[m]).mean()) for a in ARMS}
            segs[f"month_{mth}"]["n"] = int(m.sum())
        rep["folds"][fold] = {"n_test": int(len(te)), "site_terms": site, "arms": res, "segments": segs}
    f1, f2 = rep["folds"]["F1"]["arms"], rep["folds"]["F2"]["arms"]
    elig = [a for a in ARMS if a != "N" and f2[a]["gain_vs_N"] > 2 * f2[a]["floor"] and f1[a]["gain_vs_N"] > 0]
    if not elig:
        winner = "N"
    else:
        best = max(elig, key=lambda a: f2[a]["gain_vs_N"])
        tied = [a for a in elig if f2[best]["gain_vs_N"] - f2[a]["gain_vs_N"] <= f2[best]["floor"]]
        # O1 / T1 tie -> the arm passing the pre-condition (1.4): O1
        winner = min(tied, key=lambda a: (SIMPLICITY[a], 0 if a == "O1" else 1))
    rep["eligible"], rep["winner"] = elig, winner
    # wiring candidate per 1.4: only O1, and only if O1 itself is eligible
    rep["wire"] = "O1" if "O1" in elig and winner in ("O1", "OT") else None
    if rep["wire"]:
        PARAMS.parent.mkdir(parents=True, exist_ok=True)
        PARAMS.write_text(json.dumps({
            "arm": "O1", "O1_s2": float(f2["O1"]["fit"]["Sigma"][0][0]),
            "source": "scripts/exp_shared_poss_v1.py bakeoff, fold-2 TRAIN fit (2023-24), never tuned on sim output",
            "spec": "docs/models/shared_possession/experiments.md section 1",
            "created_at": rep["created_at"]}, indent=1), encoding="utf-8")
    return rep


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "measure"
    t0 = time.time()
    g = add_asof(load_games())
    g = conf_flags(g)
    print(f"[{time.time()-t0:6.1f}s] games {len(g)} by season {g.groupby('season').size().to_dict()}", flush=True)
    rep = measure(g) if stage == "measure" else bakeoff(g)
    (OUT / f"{stage}_v1.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float)[:12000])
    print(f"[{time.time()-t0:6.1f}s] wrote {OUT / (stage + '_v1.json')}")


if __name__ == "__main__":
    main()
