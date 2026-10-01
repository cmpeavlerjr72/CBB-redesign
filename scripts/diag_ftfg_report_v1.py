"""diag_ftfg_report_v1.py -- who owns the engine's negative FT x opponent-FG make covariance (lane B, 2026-10-01).

DIAGNOSTIC ONLY. Inputs:
  results/g5_channels/ftfg_tap/{trips,games}_*.parquet   per-trip tap of served v2 (diag_ftfg_tap_v1.py)
  results/engine_v0/v3full_COMB9GCTKD_s200_o0             served v2, 200 seeds: the per-game base rates p0, f0
  data/processed/possessions_v2/chances_<season>.parquet  real per-chance FTA / FTM with period, clock, score
  hoopR team box + pbp design rows (via diag_g5_channels_v1.actual_counts)

The FT make channel of the ownership table, R_ft = FTM - FTA f0, is split EXACTLY by trip context c
(half x late window x lead state x trip kind) and, in the sim, into
  binomial luck  sum(ftm - p)          and
  composition    sum(p) - f0 fta       (who shoots, in which state, how many attempts at which p).
Cov(R_X,h, R_ft,a) + Cov(R_ft,h, R_X,a) for X in rim / jump / three closes to the table's FT x FG rows.
Sim: pooled within-game covariance over the tap's seeds (ddof per game). Actual 2025: residuals against
the 200-seed per-game means. Actual 2023 / 2024 / 2025 (descriptive): residuals against as-of league
season-to-date rates (FG by type) and as-of team FT% (shrunk), the same contexts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import diag_g5_channels_v1 as GC  # noqa: E402
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402

TAP = ROOT / "results/g5_channels/ftfg_tap"
SERVED = ROOT / "results/engine_v0/v3full_COMB9GCTKD_s200_o0/games.parquet"
KT = {"rim": ("fga2_rim", "fgm2_rim"), "jump": ("fga2_jump", "fgm2_jump"), "three": ("fga3", "fgm3")}
PTS = {"rim": 2.0, "jump": 2.0, "three": 3.0}
LATE = 240           # last 4:00 of the second half (and all overtime) = "late"
NB = 200


def ctx_half(period, sec):
    return np.where(period == 1, "H1", np.where((period == 2) & (sec > LATE), "H2early", "H2late+OT"))


def ctx_lead(d):
    return np.where(d >= 4, "lead4+", np.where(d <= -4, "trail4+", "close"))


def base_rates():
    g = pd.read_parquet(SERVED)
    out = {}
    for s, o in (("h", "home"), ("a", "away")):
        for k, (a, m) in KT.items():
            out[f"p0_{k}_{s}"] = g.groupby("game_id")[f"{o}_{m}"].sum() / g.groupby("game_id")[f"{o}_{a}"].sum().clip(lower=1e-9)
        out[f"f0_{s}"] = g.groupby("game_id")[f"{o}_ftm"].sum() / g.groupby("game_id")[f"{o}_fta"].sum().clip(lower=1e-9)
    b = pd.DataFrame(out)
    pm = g.groupby("game_id")["home_pts"].mean() - g.groupby("game_id")["away_pts"].mean()
    b["pred_abs_margin"] = pm.abs()
    return b


def fg_resid(df, b, prefix_home="home", prefix_away="away"):
    """R_X per side, points-weighted, vs base p0 (b aligned on df.game_id)."""
    out = {}
    for s, o in (("h", prefix_home), ("a", prefix_away)):
        for k, (a, m) in KT.items():
            p0 = b.loc[df["game_id"], f"p0_{k}_{s}"].to_numpy()
            out[f"{k}_{s}"] = PTS[k] * (df[f"{o}_{m}"].to_numpy(float) - df[f"{o}_{a}"].to_numpy(float) * p0)
    return pd.DataFrame(out, index=df.index)


# ------------------------------------------------------------------ sim
def sim_side():
    trips = pd.concat([pd.read_parquet(p) for p in sorted(TAP.glob("trips_*.parquet"))], ignore_index=True)
    games = pd.concat([pd.read_parquet(p) for p in sorted(TAP.glob("games_*.parquet"))], ignore_index=True)
    # bit-identity against the served 200-seed rows
    ref = pd.read_parquet(SERVED)
    m = games.merge(ref, on=["game_id", "seed"], suffixes=("", "_r"))
    cols = [c for c in games.columns if c not in ("game_id", "seed") and c + "_r" in m.columns]
    mism = int(sum((m[c] != m[c + "_r"]).sum() for c in cols))
    b = base_rates()
    games = games[games["game_id"].isin(b.index)].reset_index(drop=True)
    X = fg_resid(games, b)
    # trip contexts
    t = trips.copy()
    t["half"] = ctx_half(t["period"].to_numpy(), t["sec"].to_numpy())
    t["lead"] = ctx_lead((t["pts_off"] - t["pts_def"]).to_numpy())
    t["kind"] = np.where(t["oao"], "1and1", np.where(t["n_att"] == 1, "and1", "2or3shot"))
    sname = np.where(t["side"] == 0, "h", "a")
    f0 = np.where(t["side"] == 0, b.loc[t["game_id"], "f0_h"].to_numpy(), b.loc[t["game_id"], "f0_a"].to_numpy())
    t["luck"] = t["ftm"] - t["psum"]
    t["comp"] = t["psum"] - f0 * t["fta"]
    t["s"] = sname
    return games, X, t, mism, b


def wcov_pooled(x, y, gid):
    """pooled within-game covariance, per-game ddof=1."""
    df = pd.DataFrame({"g": gid, "x": x, "y": y})
    gm = df.groupby("g")[["x", "y"]].transform("mean")
    n = df.groupby("g")["x"].transform("size").to_numpy()
    w = n / np.maximum(n - 1, 1)
    return float(np.mean((df["x"] - gm["x"]) * (df["y"] - gm["y"]) * w))


def sim_tables(games, X, t, b, by=("half", "lead", "kind")):
    key = games[["game_id", "seed"]]
    rows = []
    gid = games["game_id"].to_numpy()
    for dims in [(), ("half",), ("lead",), ("kind",), ("half", "lead")]:
        grp = t.groupby(["game_id", "seed", "s"] + list(dims))[["luck", "comp", "fta"]].sum().reset_index()
        cells = grp[list(dims)].drop_duplicates().itertuples(index=False) if dims else [()]
        for cell in cells:
            sub = grp
            for d_, v in zip(dims, cell):
                sub = sub[sub[d_] == v]
            for part in ("luck", "comp", "fta"):
                tot = 0.0
                per = {}
                for k in KT:
                    c = 0.0
                    for sx, sf in (("h", "a"), ("a", "h")):
                        f = sub[sub["s"] == sf].set_index(["game_id", "seed"])[part]
                        y = f.reindex(pd.MultiIndex.from_frame(key)).fillna(0.0).to_numpy()
                        c += wcov_pooled(X[f"{k}_{sx}"].to_numpy(), y, gid)
                    per[k] = c
                    tot += c
                rows.append({"dims": "|".join(dims) or "all", "cell": "|".join(map(str, cell)) or "all",
                             "part": part, **per, "total": tot})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ actual
def actual_ft_ctx(season, f0_by_side=None):
    c = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{season}.parquet",
                        columns=["game_id", "period", "start_clock", "start_score_diff", "offense_is_home",
                                 "offense_team_id", "fta", "ftm", "and_one", "terminal_event", "off_in_bonus",
                                 "off_in_double_bonus"])
    c = c[c["fta"] > 0].copy()
    c["half"] = ctx_half(c["period"].to_numpy(), c["start_clock"].to_numpy())
    c["lead"] = ctx_lead(c["start_score_diff"].to_numpy())
    c["kind"] = np.where(c["and_one"], "and1",
                         np.where((c["terminal_event"] == "FT_trip_bonus") & c["off_in_bonus"] & ~c["off_in_double_bonus"],
                                  "1and1", "2or3shot"))
    c["s"] = np.where(c["offense_is_home"], "h", "a")
    return c


def asof_rates(season):
    """As-of expectations for the descriptive seasons: league season-to-date make rate by FG type, and
    team FT% shrunk (100 attempts) to the league season-to-date FT% (prior dates only)."""
    fin = R.load_actual_games(season).set_index("game_id")
    common, ac = GC.actual_counts(season, fin.index.to_numpy(), fin)
    fr = fin.loc[common].copy()
    fr["date"] = pd.to_datetime(fr["game_date"])
    cnt = pd.DataFrame(index=common)
    for s in ("h", "a"):
        for k in KT:
            cnt[f"A_{k}_{s}"] = ac[s][f"A_{k}"]
            cnt[f"M_{k}_{s}"] = ac[s][f"M_{k}"]
        cnt[f"fta_{s}"] = ac[s]["fta"]
        cnt[f"ftm_{s}"] = ac[s]["ftm"]
    cnt["date"] = fr["date"]
    day = cnt.groupby("date").sum(numeric_only=True).cumsum().shift(1)
    b = pd.DataFrame(index=common)
    for k in KT:
        lg = (day[f"M_{k}_h"] + day[f"M_{k}_a"]) / (day[f"A_{k}_h"] + day[f"A_{k}_a"])
        lg = lg.bfill()
        b[f"p0_{k}_h"] = cnt["date"].map(lg).to_numpy()
        b[f"p0_{k}_a"] = b[f"p0_{k}_h"]
    lgf = ((day["ftm_h"] + day["ftm_a"]) / (day["fta_h"] + day["fta_a"])).bfill()
    long = pd.concat([pd.DataFrame({"game_id": common, "date": fr["date"].to_numpy(), "team": fr["home_team_id"].to_numpy(),
                                    "fta": cnt["fta_h"].to_numpy(), "ftm": cnt["ftm_h"].to_numpy(), "s": "h"}),
                      pd.DataFrame({"game_id": common, "date": fr["date"].to_numpy(), "team": fr["away_team_id"].to_numpy(),
                                    "fta": cnt["fta_a"].to_numpy(), "ftm": cnt["ftm_a"].to_numpy(), "s": "a"})])
    long = long.sort_values(["date", "game_id"])
    long["pa"] = long.groupby("team")["fta"].transform(lambda x: x.shift(1).fillna(0).cumsum())
    long["pm"] = long.groupby("team")["ftm"].transform(lambda x: x.shift(1).fillna(0).cumsum())
    long["lg"] = long["date"].map(lgf).to_numpy()
    long["e"] = (long["pm"] + 100 * long["lg"]) / (long["pa"] + 100)
    piv = long.pivot_table(index="game_id", columns="s", values="e")
    b["f0_h"], b["f0_a"] = piv.loc[common, "h"].to_numpy(), piv.loc[common, "a"].to_numpy()
    return common, ac, b


def actual_tables(season, base=None, boot=NB, seg=None):
    fin = R.load_actual_games(season).set_index("game_id")
    if base is None:
        common, ac, b = asof_rates(season)
    else:
        common, ac = GC.actual_counts(season, fin.index.intersection(base.index).to_numpy(), fin)
        b = base.loc[common]
    df = pd.DataFrame(index=common)
    for s, o in (("h", "home"), ("a", "away")):
        for k, (a, m) in KT.items():
            df[f"{o}_{a}"] = ac[s][f"A_{k}"]
            df[f"{o}_{m}"] = ac[s][f"M_{k}"]
    df = df.reset_index().rename(columns={"index": "game_id"})
    X = fg_resid(df, b.reindex(common))
    X.index = common
    X = X - X.mean()
    c = actual_ft_ctx(season)
    c = c[c["game_id"].isin(common)]
    f0 = np.where(c["s"] == "h", b.loc[c["game_id"], "f0_h"].to_numpy(), b.loc[c["game_id"], "f0_a"].to_numpy())
    c["r"] = c["ftm"] - f0 * c["fta"]
    c["fta_n"] = c["fta"].astype(float)
    if seg is not None:
        keep = seg.reindex(common).fillna(False).to_numpy()
    else:
        keep = np.ones(len(common), bool)
    rows = []
    rng = np.random.default_rng(31)
    W = rng.poisson(1.0, size=(boot, int(keep.sum()))).astype(float)
    for dims in [(), ("half",), ("lead",), ("kind",), ("half", "lead")]:
      for val in ("r", "fta_n"):
        grp = c.groupby(["game_id", "s"] + list(dims))[val].sum().reset_index().rename(columns={val: "r"})
        cells = grp[list(dims)].drop_duplicates().itertuples(index=False) if dims else [()]
        for cell in cells:
            sub = grp
            for d_, v in zip(dims, cell):
                sub = sub[sub[d_] == v]
            per, bs = {}, np.zeros(boot)
            tot = 0.0
            for k in KT:
                cc = 0.0
                for sx, sf in (("h", "a"), ("a", "h")):
                    y = sub[sub["s"] == sf].set_index("game_id")["r"].reindex(common).fillna(0.0).to_numpy()[keep]
                    x = X[f"{k}_{sx}"].to_numpy()[keep]
                    yc = y - y.mean()
                    cc += float(np.mean(x * yc))
                    xw = (W * x[None]).sum(1) / W.sum(1)
                    yw = (W * y[None]).sum(1) / W.sum(1)
                    bs += (W * x[None] * y[None]).sum(1) / W.sum(1) - xw * yw
                per[k] = cc
                tot += cc
            rows.append({"dims": "|".join(dims) or "all", "cell": "|".join(map(str, cell)) or "all", "part": "make" if val == "r" else "fta",
                         **per, "total": tot, "se": float(bs.std())})
    return pd.DataFrame(rows), int(keep.sum())


def main():
    rep = {}
    games, X, t, mism, b = sim_side()
    rep["bit_identity_mismatched_cells"] = mism
    rep["tap_rows"] = int(len(games))
    st = sim_tables(games, X, t, b)
    rep["sim"] = st.to_dict("records")
    act, n = actual_tables(2025, base=b)
    rep["actual_2025_vs_sim_mean"] = {"n": n, "rows": act.to_dict("records")}
    # margin tiers (predicted)
    q = pd.qcut(b["pred_abs_margin"], 3, labels=["close", "mid", "wide"])
    rep["actual_2025_by_tier"] = {}
    rep["sim_by_tier"] = {}
    for tier in ("close", "mid", "wide"):
        seg = q == tier
        a_t, n_t = actual_tables(2025, base=b, seg=seg, boot=100)
        rep["actual_2025_by_tier"][tier] = {"n": n_t, "rows": a_t[a_t["dims"].isin(["all", "half"])].to_dict("records")}
        keep = games["game_id"].map(seg).fillna(False).to_numpy()
        gs, Xs = games[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
        ts = t[t["game_id"].isin(set(gs["game_id"]))]
        s_t = sim_tables(gs, Xs, ts, b)
        rep["sim_by_tier"][tier] = s_t[s_t["dims"].isin(["all", "half"])].to_dict("records")
    rep["actual_by_season_asof"] = {}
    for s in (2023, 2024, 2025):
        a_s, n_s = actual_tables(s)
        rep["actual_by_season_asof"][str(s)] = {"n": n_s, "rows": a_s.to_dict("records")}
    out = ROOT / "results/g5_channels/ftfg_report_v1.json"
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 220)
    print("mismatch", mism, "rows", len(games))
    print(st.round(3).to_string())
    print(act.round(3).to_string())


if __name__ == "__main__":
    main()
