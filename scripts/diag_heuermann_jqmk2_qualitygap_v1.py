"""diag_heuermann_jqmk2_qualitygap_v1.py -- DIAGNOSTIC, report only (2026-11-11).

Question (Heuermann sim_v2 jqmk2 finding, PM hypothesis of over-shrinkage toward the LEAGUE mean): GIVEN our served
sub-models' honest walk-forward per-event predictions, does the pre-game OVERALL quality gap still predict the real
outcome? No simulation; real events only. Nothing here is a proposal to add a term.

Inputs (read-only; every prediction is an S1 walk-forward test-season prediction, fold 1 test 2023-24, fold 2 test
2024-25; 2025-26 sealed and never read):
  fg_make  data/processed/models/fg_make/round_pae/R_s0_F{1,2}/preds_F{1,2}_s0.parquet   (served B1 replica, PAE round
           control arm `R`, fg_make experiments.md s31.3; per-shot y, p)
  PO first data/processed/models/fold1_v1/po/pred_first_F1.npy, full_retrain_v1/identity/po_served/pred_first_F2.npy
           (served lgbm/C_plus_state S1; rows = round2 design `first` rows of the test season in design order, the
           same convention as scripts/grade_po_tovlevel_v1.py)
  rebound  data/processed/models/fold1_v1/rb/cuts/s2_F1_A0B0C0_seed0/*.pkl (served spec S1_weekly, fold 1 only; rows =
           design_round3 test-season rows in design order, segment = [refit_k, refit_k+1))
  quality  data/processed/ratings/own_ratings_{2024,2025}.parquet: q = (off_c - def_c)/100, points per possession
           relative to the snapshot's league mean, as_of_date == game date means fitted strictly before it.
  truth    data/processed/truth/game_finals_v2.parquet (verified finals), possessions counted from the PO round2 design.
  closes   data/processed/lines/lines_close_v2_verified.parquet (lines_open_close_v1 holds 2026-27 only); yardstick only.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/diag_heuermann_jqmk2_qualitygap_v1.py
Writes results/heuermann_jqmk2_qualitygap/{tables.json,tables.md}.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

OUT = ROOT / "results/heuermann_jqmk2_qualitygap"
FOLDS = {"F1": 2024, "F2": 2025}
FG_PRED = {f: ROOT / f"data/processed/models/fg_make/round_pae/R_s0_{f}/preds_{f}_s0.parquet" for f in FOLDS}
PO_PRED = {"F1": ROOT / "data/processed/models/fold1_v1/po/pred_first_F1.npy",
           "F2": ROOT / "data/processed/models/full_retrain_v1/identity/po_served/pred_first_F2.npy"}
PO_DESIGN = ROOT / "data/processed/models/possession_outcome/round2/design.parquet"
RB_DESIGN = ROOT / "data/processed/models/rebound/round3/design_round3.parquet"
RB_CUTS = {"F1": ROOT / "data/processed/models/fold1_v1/rb/cuts/s2_F1_A0B0C0_seed0"}
LINES = ROOT / "data/processed/lines/lines_close_v2_verified.parquet"
FINALS = ROOT / "data/processed/truth/game_finals_v2.parquet"
UNIVERSE = ROOT / "data/processed/games_universe.parquet"
BOOT = 1000
RNG = np.random.default_rng(20261111)
UNDERPOWERED = 300
LOG = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------------------------------------- fitting
def pois_fit(y, off, X, cluster):
    """Poisson IRLS, log mu = off + X b; cluster-robust (by game) sandwich SE."""
    y, off, X = np.asarray(y, float), np.asarray(off, float), np.asarray(X, float)
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(50):
        mu = np.exp(off + X @ b)
        step = np.linalg.solve((X * mu[:, None]).T @ X, X.T @ (y - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = np.exp(off + X @ b)
    bread = np.linalg.inv((X * mu[:, None]).T @ X)
    sc = X * (y - mu)[:, None]
    _, inv = np.unique(np.asarray(cluster), return_inverse=True)
    G = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(G, inv, sc)
    ng = G.shape[0]
    meat = G.T @ G * ng / max(ng - 1, 1)
    V = bread @ meat @ bread
    return b, np.sqrt(np.diag(V)), ng


def gfit(d, cols, ycol="made", pcol="p"):
    d = d[d[pcol] > 0]
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])
    b, se, ng = pois_fit(d[ycol], np.log(d[pcol]), X, d["game_id"])
    return {"n_team_games": int(len(d)), "n_games": int(ng), "n_events": float(d["n"].sum()),
            "level": float(b[0]), **{c: float(b[i + 1]) for i, c in enumerate(cols)},
            **{f"se_{c}": float(se[i + 1]) for i, c in enumerate(cols)},
            "underpowered": bool(len(d) < UNDERPOWERED)}


def zfmt(r, c):
    if r is None or c not in r:
        return "n/a"
    g, se = r[c], r[f"se_{c}"]
    tag = " UNDERPOWERED" if r.get("underpowered") else ""
    return f"{g:+.3f} ({se:.3f}, z {g / se:+.1f}){tag}"


# ----------------------------------------------------------------------------------------------- quality
def load_quality():
    rs = []
    for s in FOLDS.values():
        assert_not_sealed(s)
        r = pd.read_parquet(ROOT / f"data/processed/ratings/own_ratings_{s}.parquet",
                            columns=["season", "as_of_date", "team_id", "off_c", "def_c", "n_games"])
        rs.append(r)
    r = pd.concat(rs, ignore_index=True)
    r["q"] = (r["off_c"] - r["def_c"]) / 100.0
    r["q_off"] = r["off_c"] / 100.0
    r["q_def"] = -r["def_c"] / 100.0           # + = good defence
    r["as_of_date"] = pd.to_datetime(r["as_of_date"]).astype("datetime64[ns]")
    return r[["season", "as_of_date", "team_id", "q", "q_off", "q_def", "n_games"]]


def prior_games_all():
    """Games in our data (games_universe, seasons 2022-2025) strictly before each date, per team."""
    u = pd.read_parquet(UNIVERSE, columns=["game_id", "season", "game_date", "home_team_id", "away_team_id"])
    u = u[u["season"].between(2022, 2025)]
    u["game_date"] = pd.to_datetime(u["game_date"]).astype("datetime64[ns]")
    tg = pd.concat([u[["game_date", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
                    u[["game_date", "away_team_id"]].rename(columns={"away_team_id": "team_id"})])
    c = tg.groupby(["team_id", "game_date"]).size().rename("k").reset_index().sort_values(["team_id", "game_date"])
    c["h_all"] = c.groupby("team_id")["k"].cumsum() - c["k"]       # strictly before the date
    return c[["team_id", "game_date", "h_all"]]


def attach_q(df, R, H, team="off_team_id", opp="def_team_id"):
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).astype("datetime64[ns]")
    for side, col in (("o", team), ("d", opp)):
        r = R.rename(columns={"team_id": col, "as_of_date": "game_date",
                              **{c: f"{c}_{side}" for c in ("q", "q_off", "q_def", "n_games")}})
        df = df.merge(r, on=["season", col, "game_date"], how="left")
        h = H.rename(columns={"team_id": col, "h_all": f"h_{side}"})
        df = df.merge(h, on=[col, "game_date"], how="left")
    df["gap"] = (df["q_o"] - df["q_d"]) / 2
    df["q_own_h"] = df["q_o"] / 2
    df["q_opp_h"] = df["q_d"] / 2
    df["mature"] = (df[["h_o", "h_d"]].min(axis=1) >= 20)
    df["ng_min"] = df[["n_games_o", "n_games_d"]].min(axis=1)
    return df


# ----------------------------------------------------------------------------------------------- tests
def quality_validation(R):
    """OOS slope of real per-game PPP margin on the q difference (q_home - q_away), per season."""
    fin = pd.read_parquet(FINALS, columns=["game_id", "season", "home_team_id", "away_team_id", "home_score",
                                           "away_score"])
    po = pd.read_parquet(PO_DESIGN, columns=["game_id", "season", "poss_index", "offense_team_id", "game_date"])
    out = {}
    for f, s in FOLDS.items():
        p = po[po["season"] == s].drop_duplicates(["game_id", "poss_index", "offense_team_id"])
        poss = p.groupby(["game_id", "offense_team_id"]).size().rename("poss").reset_index()
        gd = p.groupby("game_id")["game_date"].first()
        g = fin[fin["season"] == s].copy()
        g["game_date"] = g["game_id"].map(gd)
        g = g.dropna(subset=["game_date"])
        g = g.merge(poss.rename(columns={"offense_team_id": "home_team_id", "poss": "ph"}), on=["game_id", "home_team_id"])
        g = g.merge(poss.rename(columns={"offense_team_id": "away_team_id", "poss": "pa"}), on=["game_id", "away_team_id"])
        g["game_date"] = pd.to_datetime(g["game_date"]).astype("datetime64[ns]")
        for side, col in (("h", "home_team_id"), ("a", "away_team_id")):
            g = g.merge(R[["season", "as_of_date", "team_id", "q"]].rename(
                columns={"team_id": col, "as_of_date": "game_date", "q": f"q_{side}"}), on=["season", col, "game_date"],
                how="left")
        g = g.dropna(subset=["q_h", "q_a"])
        y = g["home_score"] / g["ph"] - g["away_score"] / g["pa"]
        x = g["q_h"] - g["q_a"]
        X = np.column_stack([np.ones(len(g)), x])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ b
        V = np.linalg.inv(X.T @ X) * res.var()
        out[f] = {"n_games": int(len(g)), "slope": float(b[1]), "se": float(np.sqrt(V[1, 1])),
                  "intercept_home_ppp": float(b[0]), "corr": float(np.corrcoef(x, y)[0, 1])}
    return out


def fg_team_games(R, H):
    frames = []
    for f, s in FOLDS.items():
        assert_not_sealed(s)
        d = pd.read_parquet(FG_PRED[f])
        assert (d["season"] == s).all()
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["fam"] = d["shot_class"].map({"FGA_3": "3P", "FGA_rim": "rim", "FGA_jump2": "jump2"})
    d["val"] = np.where(d["fam"] == "3P", 3.0, 2.0)
    d["game_date"] = pd.to_datetime(d["game_date"]).astype("datetime64[ns]")
    # prior same-season attempts of the shooting team, strictly before the game date (test 6)
    d["is3"] = (d["fam"] == "3P").astype(int)
    daily = d.groupby(["season", "off_team_id", "game_date", "is3"]).size().rename("k").reset_index()
    daily = daily.sort_values(["season", "off_team_id", "is3", "game_date"])
    daily["prior_att"] = daily.groupby(["season", "off_team_id", "is3"])["k"].cumsum() - daily["k"]
    rows = []
    for fam_name, mask in (("3P", d["fam"] == "3P"), ("2P", d["fam"] != "3P"), ("rim", d["fam"] == "rim"),
                           ("jump2", d["fam"] == "jump2")):
        x = d[mask].groupby(["season", "game_id", "game_date", "off_team_id", "def_team_id"]).agg(
            n=("y", "size"), made=("y", "sum"), p=("p", "sum")).reset_index()
        x["fam"] = fam_name
        rows.append(x)
    tg = pd.concat(rows, ignore_index=True)
    tg = attach_q(tg, R, H)
    tg["is3"] = (tg["fam"] == "3P").astype(int)
    tg = tg.merge(daily[["season", "off_team_id", "game_date", "is3", "prior_att"]],
                  on=["season", "off_team_id", "game_date", "is3"], how="left")
    tg["prior_att"] = tg["prior_att"].fillna(0)
    return d, tg


def test1(tg):
    out = {}
    for fam in ("3P", "2P", "rim", "jump2"):
        for lab, m in (("F1_2024", tg["season"] == 2024), ("F2_2025", tg["season"] == 2025),
                       ("pooled", tg["season"].isin([2024, 2025]))):
            x = tg[m & (tg["fam"] == fam)].dropna(subset=["gap"])
            out[f"{fam}|{lab}|gap"] = gfit(x, ["gap"])
            out[f"{fam}|{lab}|split"] = gfit(x, ["q_own_h", "q_opp_h"])
            x = x.assign(qo_off=x["q_off_o"] / 2, qo_def=x["q_def_o"] / 2, qd_off=x["q_off_d"] / 2,
                         qd_def=x["q_def_d"] / 2)
            out[f"{fam}|{lab}|split4"] = gfit(x, ["qo_off", "qo_def", "qd_off", "qd_def"])
    return out


def test2(tg):
    out = {}
    for fam in ("3P", "2P"):
        for lab, s in (("F1_2024", [2024]), ("F2_2025", [2025]), ("pooled", [2024, 2025])):
            x = tg[tg["season"].isin(s) & (tg["fam"] == fam)].dropna(subset=["gap"])
            for mlab, mm in (("mature", x["mature"]), ("low_data", ~x["mature"])):
                out[f"{fam}|{lab}|{mlab}"] = gfit(x[mm], ["gap"])
            # terciles of the min (over the two teams) of prior games THIS season; cut per season
            ter = x.groupby("season")["ng_min"].transform(lambda v: pd.qcut(v.rank(method="first"), 3,
                                                                            labels=["early", "mid", "late"]))
            for t in ("early", "mid", "late"):
                xx = x[ter == t]
                r = gfit(xx, ["gap"])
                r["ng_min_range"] = [int(xx["ng_min"].min()), int(xx["ng_min"].max())]
                out[f"{fam}|{lab}|tercile_{t}"] = r
    return out


def test3(tg):
    out = {}
    for lab, s in (("F1_2024", [2024]), ("F2_2025", [2025])):
        x = tg[tg["season"].isin(s)].dropna(subset=["gap"])
        base = x[x["fam"] == "3P"][["game_id", "off_team_id", "gap"]]
        qcut = pd.qcut(base["gap"], 5, labels=False)
        edges = base.groupby(qcut)["gap"].agg(["min", "max"])
        key = base.assign(qq=qcut)[["game_id", "off_team_id", "qq"]]
        rows = []
        for fam in ("3P", "2P", "rim", "jump2"):
            y = x[x["fam"] == fam].merge(key, on=["game_id", "off_team_id"], how="inner")
            g = y.groupby("qq").agg(tg_n=("n", "size"), att=("n", "sum"), made=("made", "sum"), p=("p", "sum"))
            for q, r in g.iterrows():
                rows.append({"fam": fam, "quintile": int(q) + 1, "gap_lo": float(edges.loc[q, "min"]),
                             "gap_hi": float(edges.loc[q, "max"]), "team_games": int(r["tg_n"]),
                             "att": int(r["att"]), "actual": r["made"] / r["att"], "pred": r["p"] / r["att"],
                             "ratio": r["made"] / r["p"]})
        out[lab] = rows
    return out


def boot_mean(v, reps=BOOT):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = RNG.integers(0, len(v), (reps, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def test4(d, R):
    """Per game: expected-minus-real make-layer points, favourite minus underdog."""
    d = d.copy()
    d["e_m_r"] = (d["p"] - d["y"]) * d["val"]
    pt = d.pivot_table(index=["season", "game_id", "game_date", "off_team_id"], columns="fam", values="e_m_r",
                       aggfunc="sum", fill_value=0.0).reset_index()
    pt["fam2"] = pt.get("rim", 0) + pt.get("jump2", 0)
    pt = pt.rename(columns={"3P": "fam3"})
    opp = d.groupby(["game_id", "off_team_id"])["def_team_id"].first().rename("def_team_id").reset_index()
    pt = pt.merge(opp, on=["game_id", "off_team_id"])
    pt = attach_q(pt, R, prior_games_all())
    two = pt.groupby("game_id")["off_team_id"].transform("size") == 2
    pt = pt[two]
    # games: one row, home/away from finals
    fin = pd.read_parquet(FINALS, columns=["game_id", "home_team_id", "away_team_id"])
    L = pd.read_parquet(LINES, columns=["game_id", "season", "provider", "close_spread_home"]).dropna(
        subset=["close_spread_home"])
    L = L.sort_values(["game_id", "provider"]).groupby("game_id")["close_spread_home"].median()
    g = pt.merge(fin, on="game_id")
    h = g[g["off_team_id"] == g["home_team_id"]].set_index("game_id")
    a = g[g["off_team_id"] == g["away_team_id"]].set_index("game_id")
    idx = h.index.intersection(a.index)
    h, a = h.loc[idx], a.loc[idx]
    G = pd.DataFrame({"season": h["season"], "q_h": h["q_o"], "q_a": a["q_o"]})
    for c in ("fam3", "fam2"):
        G[f"{c}_h"], G[f"{c}_a"] = h[c], a[c]
    G["spread"] = L.reindex(idx)
    out = {}
    for frame in ("close", "qgap", "qspread"):
        if frame == "qspread":
            # our own q expressed as an implied neutral spread: |q_h - q_a| x 70 possessions (no market data)
            X = G.dropna(subset=["q_h", "q_a"]).copy()
            fav_home = (X["q_h"] >= X["q_a"]).to_numpy()
            X["size"] = (X["q_h"] - X["q_a"]).abs() * 70.0
            bins = [-0.01, 3, 6, 10, 15, 20, 100]
            labels = ["0-3", "3-6", "6-10", "10-15", "15-20", "20+"]
        elif frame == "close":
            X = G.dropna(subset=["spread"]).copy()
            fav_home = X["spread"] < 0
            fav_home = np.where(X["spread"] == 0, X["q_h"] >= X["q_a"], fav_home)
            X["size"] = X["spread"].abs()
            bins = [-0.01, 3, 6, 10, 15, 20, 100]
            labels = ["0-3", "3-6", "6-10", "10-15", "15-20", "20+"]
        else:
            X = G.dropna(subset=["q_h", "q_a"]).copy()
            fav_home = (X["q_h"] >= X["q_a"]).to_numpy()
            X["size"] = (X["q_h"] - X["q_a"]).abs()
            bins, labels = None, None
        for c in ("fam3", "fam2"):
            X[f"{c}_fm"] = np.where(fav_home, X[f"{c}_h"] - X[f"{c}_a"], X[f"{c}_a"] - X[f"{c}_h"])
        X["tot_fm"] = X["fam3_fm"] + X["fam2_fm"]
        X["q_fav_minus_dog"] = np.where(fav_home, X["q_h"] - X["q_a"], X["q_a"] - X["q_h"])
        for slab, s in (("F1_2024", [2024]), ("F2_2025", [2025]), ("pooled", [2024, 2025])):
            Y = X[X["season"].isin(s)]
            if frame in ("close", "qspread"):
                bk = pd.cut(Y["size"], bins, labels=labels)
            else:
                bk = pd.qcut(Y["size"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
            rows = []
            for b in bk.cat.categories:
                Z = Y[bk == b]
                r = {"bucket": str(b), "n_games": int(len(Z)), "size_mean": float(Z["size"].mean()),
                     "underpowered": bool(len(Z) < UNDERPOWERED / 2),
                     # our q gap in the favourite's direction, as a 70-possession implied spread
                     "q_implied_fav_spread": float((Z["q_fav_minus_dog"] * 70.0).mean())}
                for c, nm in (("fam3_fm", "3P"), ("fam2_fm", "2P"), ("tot_fm", "total")):
                    lo, hi = boot_mean(Z[c])
                    r[nm] = float(Z[c].mean())
                    r[f"{nm}_ci"] = [lo, hi]
                rows.append(r)
            out[f"{frame}|{slab}"] = rows
    out["_n_close_games"] = {s: int(G[G["season"] == s]["spread"].notna().sum()) for s in (2024, 2025)}
    return out


def po_team_games(R, H):
    frames = []
    for f, s in FOLDS.items():
        assert_not_sealed(s)
        d = pd.read_parquet(PO_DESIGN, columns=["season", "population", "y", "game_id", "game_date",
                                                "offense_team_id", "defense_team_id"])
        te = d[(d["season"] == s) & (d["population"] == "first")].reset_index(drop=True)
        Pm = np.load(PO_PRED[f]).astype("float64")
        assert Pm.shape == (len(te), 6), (f, Pm.shape, len(te))
        y = te["y"].to_numpy().astype(int)
        P(f"  PO first {f}: mean pred {np.round(Pm.mean(0), 4)} realised {np.round(np.bincount(y, minlength=6) / len(y), 4)}")
        te = te.rename(columns={"offense_team_id": "off_team_id", "defense_team_id": "def_team_id"})
        fga = Pm[:, 1:4].sum(1)
        ch = {
            "TOV": (y == 0, Pm[:, 0], np.ones(len(te), bool)),
            "FT_trip_shooting": (y == 4, Pm[:, 4], np.ones(len(te), bool)),
            "FT_trip_bonus (non-shooting foul)": (y == 5, Pm[:, 5], np.ones(len(te), bool)),
            "3PA share of FGA": (y == 3, Pm[:, 3] / fga, np.isin(y, [1, 2, 3])),
        }
        for name, (ev, p, m) in ch.items():
            x = te.loc[m, ["season", "game_id", "game_date", "off_team_id", "def_team_id"]].assign(
                made=ev[m].astype(float), p=p[m], n=1.0)
            x = x.groupby(["season", "game_id", "game_date", "off_team_id", "def_team_id"], as_index=False)[
                ["n", "made", "p"]].sum()
            x["chan"] = name
            frames.append(x)
    return attach_q(pd.concat(frames, ignore_index=True), R, H)


def rb_team_games(R, H):
    frames = []
    for f, cdir in RB_CUTS.items():
        s = FOLDS[f]
        d = pd.read_parquet(RB_DESIGN, columns=["season", "game_id", "game_date", "off_team_id", "def_team_id",
                                                "miss_type", "y"])
        te = d[d["season"] == s].reset_index(drop=True)
        dates = pd.to_datetime(te["game_date"])
        files = sorted(cdir.glob("*.pkl"))
        refits = [pd.Timestamp(p.stem.split("_", 1)[1]) for p in files]
        Pm = np.zeros((len(te), 3))
        for k, (pth, cut) in enumerate(zip(files, refits)):
            nxt = refits[k + 1] if k + 1 < len(refits) else None
            sel = ((dates >= cut) if nxt is None else ((dates >= cut) & (dates < nxt))).to_numpy()
            o = pickle.load(open(pth, "rb"))
            assert o["p"].shape[0] == sel.sum() == o["seg"]["n_scored"], (pth, o["p"].shape, sel.sum())
            Pm[sel] = o["p"]
        assert (Pm.sum(1) > 0).all()
        y = te["y"].to_numpy().astype(int)
        P(f"  rebound {f}: mean pred {np.round(Pm.mean(0), 4)} realised {np.round(np.bincount(y, minlength=3) / len(y), 4)}")
        m = (te["miss_type"] != "ft").to_numpy()
        x = te.loc[m, ["season", "game_id", "game_date", "off_team_id", "def_team_id"]].assign(
            made=(y[m] == 0).astype(float), p=Pm[m, 0], n=1.0)
        x = x.groupby(["season", "game_id", "game_date", "off_team_id", "def_team_id"], as_index=False)[
            ["n", "made", "p"]].sum()
        x["chan"] = "OREB rate (FGA misses)"
        frames.append(x)
    return attach_q(pd.concat(frames, ignore_index=True), R, H)


def test5(po, rb):
    out = {}
    for tbl in (po, rb):
        for chan in tbl["chan"].unique():
            for lab, s in (("F1_2024", [2024]), ("F2_2025", [2025]), ("pooled", [2024, 2025])):
                x = tbl[(tbl["chan"] == chan) & tbl["season"].isin(s)].dropna(subset=["gap"])
                if len(x) == 0:
                    out[f"{chan}|{lab}"] = None
                    continue
                out[f"{chan}|{lab}"] = gfit(x, ["gap"])
                out[f"{chan}|{lab}|split"] = gfit(x, ["q_own_h", "q_opp_h"])
    return out


def test6(tg):
    out = {}
    for fam in ("3P", "2P"):
        for lab, s in (("F1_2024", [2024]), ("F2_2025", [2025]), ("pooled", [2024, 2025])):
            x = tg[tg["season"].isin(s) & (tg["fam"] == fam)].dropna(subset=["gap"]).copy()
            L = np.log1p(x["prior_att"])
            x["L_c"] = L - L.mean()
            x["gapxL"] = x["gap"] * x["L_c"]
            r = gfit(x, ["gap", "L_c", "gapxL"])
            r["L_mean"] = float(L.mean())
            r["L_sd"] = float(L.std())
            out[f"{fam}|{lab}"] = r
    return out


def present(rows, key_fmt, coef="gap"):
    f2, f1 = rows.get(key_fmt.format("F2_2025")), rows.get(key_fmt.format("F1_2024"))
    if not f2:
        return "not run"
    z2 = f2[coef] / f2[f"se_{coef}"]
    if abs(z2) < 3:
        return "not present (F2 |z| < 3)"
    if f1 and np.sign(f1[coef]) == np.sign(f2[coef]) and abs(f1[coef] / f1[f"se_{coef}"]) >= 2:
        return "LOADING PRESENT"
    return "single-season, unconfirmed"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = load_quality()
    H = prior_games_all()
    res = {}
    P("== quality state validation (OOS slope of real PPP margin on q_home - q_away)")
    res["quality_validation"] = quality_validation(R)
    for k, v in res["quality_validation"].items():
        P(f"  {k}: n {v['n_games']} slope {v['slope']:.3f} ({v['se']:.3f}) corr {v['corr']:.3f} home int {v['intercept_home_ppp']:+.4f}")
    d, tg = fg_team_games(R, H)
    P(f"fg shots {len(d):,}; team-game-family rows {len(tg):,}; missing q {int(tg['gap'].isna().sum())}")
    P(f"  maturity: share of 3P team-games with both teams >= 20 prior games in our data: "
      f"{tg[tg.fam == '3P'].groupby('season')['mature'].mean().round(4).to_dict()}")
    res["test1"] = test1(tg)
    P("== test 1: make residual loading g on (q_off - q_def)/2")
    for fam in ("3P", "2P", "rim", "jump2"):
        for lab in ("F1_2024", "F2_2025", "pooled"):
            r = res["test1"][f"{fam}|{lab}|gap"]
            s = res["test1"][f"{fam}|{lab}|split"]
            s4 = res["test1"][f"{fam}|{lab}|split4"]
            P(f"  {fam:5s} {lab:8s} n {r['n_team_games']:6d} g {zfmt(r, 'gap')} | own {zfmt(s, 'q_own_h')} opp "
              f"{zfmt(s, 'q_opp_h')} | own_off {zfmt(s4, 'qo_off')} own_def {zfmt(s4, 'qo_def')} opp_off "
              f"{zfmt(s4, 'qd_off')} opp_def {zfmt(s4, 'qd_def')}")
        P(f"    reading {fam}: {present(res['test1'], fam + '|{}|gap')}")
    res["test2"] = test2(tg)
    P("== test 2: maturity / season-stage")
    for k, r in res["test2"].items():
        P(f"  {k:34s} n {r['n_team_games']:6d} g {zfmt(r, 'gap')} {r.get('ng_min_range', '')}")
    res["test3"] = test3(tg)
    P("== test 3: responsiveness by quality-gap quintile")
    for lab, rows in res["test3"].items():
        for r in rows:
            if r["fam"] in ("3P", "2P"):
                P(f"  {lab} {r['fam']} Q{r['quintile']} gap [{r['gap_lo']:+.3f},{r['gap_hi']:+.3f}] tg {r['team_games']} "
                  f"act {r['actual']:.4f} pred {r['pred']:.4f} ratio {r['ratio']:.4f}")
    res["test4"] = test4(d, R)
    P("== test 4: make-layer margin, expected minus real, favourite minus underdog (points per game)")
    for k, rows in res["test4"].items():
        if k.startswith("_"):
            P(f"  {k} {rows}")
            continue
        for r in rows:
            P(f"  {k:14s} {r['bucket']:6s} n {r['n_games']:5d} qspr {r['q_implied_fav_spread']:+5.1f} 3P {r['3P']:+.2f} [{r['3P_ci'][0]:+.2f},{r['3P_ci'][1]:+.2f}] "
              f"2P {r['2P']:+.2f} [{r['2P_ci'][0]:+.2f},{r['2P_ci'][1]:+.2f}] tot {r['total']:+.2f} "
              f"[{r['total_ci'][0]:+.2f},{r['total_ci'][1]:+.2f}]{' UNDERPOWERED' if r['underpowered'] else ''}")
    P("== test 5: other channels")
    po = po_team_games(R, H)
    rb = rb_team_games(R, H)
    res["test5"] = test5(po, rb)
    for k, r in res["test5"].items():
        if r is None:
            P(f"  {k}: no rows")
            continue
        if k.endswith("|split"):
            P(f"  {k:52s} own {zfmt(r, 'q_own_h')} opp {zfmt(r, 'q_opp_h')}")
        else:
            P(f"  {k:52s} n {r['n_team_games']:6d} g {zfmt(r, 'gap')}")
    for chan in list(po["chan"].unique()) + list(rb["chan"].unique()):
        P(f"    reading {chan}: {present(res['test5'], chan + '|{}')}")
    res["test6"] = test6(tg)
    P("== test 6: gap x log(1 + prior same-season team attempts of that family)")
    for k, r in res["test6"].items():
        P(f"  {k:12s} gap {zfmt(r, 'gap')} L {zfmt(r, 'L_c')} gap x L {zfmt(r, 'gapxL')} (L mean {r['L_mean']:.2f} sd {r['L_sd']:.2f})")
    (OUT / "tables.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (OUT / "tables.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    P(f"wrote {OUT}")


if __name__ == "__main__":
    main()
