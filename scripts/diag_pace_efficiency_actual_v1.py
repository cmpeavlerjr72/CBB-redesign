"""diag_pace_efficiency_actual_v1.py -- the ACTUAL side of the pace/efficiency
sign question (lane: engine diagnosis, 2026-09-11).

Reads data/processed/possessions_v2/possessions_<season>.parquet and answers,
like for like with the engine's own within-game read:

  1. eFG% by start_reason (prev_end) and by is_transition, with durations.
  2. The between-game corr(P, eFG), decomposed into a MIX term (the game's
     shot-class composition times season-mean make rates) and a WITHIN term
     (make rates inside comparable cells).
  3. corr(P, eFG) computed inside each start_reason cell.
  4. The composition-implied pace P_comp (live seconds / mix-weighted mean
     duration) and the residual pace P_resid = P - P_comp, each against eFG.
     P_comp is the efficiency->pace arrow; P_resid is the tempo latent.

Read-only.  Writes nothing but stdout.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
PATH = f"data/processed/possessions_v2/possessions_{SEASON}.parquet"

COLS = ["game_id", "season", "period", "offense_team_id", "offense_is_home",
        "duration_s", "start_reason", "terminal_event", "n_chances",
        "oreb_count", "is_transition", "start_score_diff",
        "fga_rim", "fgm_rim", "fga_jump2", "fgm_jump2", "fga_3", "fgm_3",
        "fta", "ftm", "points"]

p = pd.read_parquet(PATH, columns=COLS)
p = p[p["period"] <= 2].copy()            # regulation only
p["fga"] = p["fga_rim"] + p["fga_jump2"] + p["fga_3"]
p["fgm"] = p["fgm_rim"] + p["fgm_jump2"] + p["fgm_3"]
p["efg_num"] = p["fgm"] + 0.5 * p["fgm_3"]
p["dur"] = p["duration_s"].astype(float).clip(lower=0)
p["is_tov"] = (p["terminal_event"] == "TOV").astype(float)

print("=== ACTUAL %d, regulation possessions: %d in %d games ===\n"
      % (SEASON, len(p), p.game_id.nunique()))


def cell_table(df, key, label):
    g = df.groupby(key, dropna=False, observed=True).agg(
        n=("fga", "size"), dur=("dur", "mean"),
        fga=("fga", "sum"), num=("efg_num", "sum"),
        pts=("points", "sum"), nch=("n_chances", "mean"))
    g["efg"] = g["num"] / g["fga"].replace(0, np.nan)
    g["fga_per_poss"] = g["fga"] / g["n"]
    g["ppp"] = g["pts"] / g["n"]
    g["share"] = g["n"] / g["n"].sum()
    print("-- %s --" % label)
    print(g[["n", "share", "dur", "efg", "fga_per_poss", "ppp", "nch"]]
          .sort_values("n", ascending=False).round(4).to_string())
    print()
    return g


t_reason = cell_table(p, "start_reason", "eFG% and duration by start_reason (prev_end)")
cell_table(p, "is_transition", "eFG% and duration by is_transition")
p["reason_trans"] = p["start_reason"].astype(str) + "|" + p["is_transition"].astype(str)
cell_table(p[p["start_reason"].isin(["DREB", "TOV"])], "reason_trans",
           "start_reason x is_transition, DREB/TOV only")

p["dur_b"] = pd.cut(p["dur"], [-0.1, 4, 8, 12, 16, 20, 25, 30, 1e9],
                    labels=["0-4", "5-8", "9-12", "13-16", "17-20", "21-25",
                            "26-30", "31+"])
cell_table(p, "dur_b", "eFG% by the possession's OWN duration (all)")
cell_table(p[p["start_reason"].isin(["DREB", "TOV"])], "dur_b",
           "eFG% by own duration, prev_end in (DREB,TOV) "
           "-- the engine's is_transition population")

# ------------------------------------------------- per-game aggregation
g = p.groupby("game_id").agg(
    P=("dur", "size"), live=("dur", "sum"),
    fga=("fga", "sum"), num=("efg_num", "sum"),
    pts=("points", "sum"), tov=("is_tov", "sum"),
    fta=("fta", "sum"), oreb=("oreb_count", "sum"),
    trans=("is_transition", "mean"))
g["efg"] = g["num"] / g["fga"]
g["ppp"] = g["pts"] / g["P"]
g = g[(g["P"] >= 80) & (g["fga"] >= 60)]
print("games after filter: %d\n" % len(g))


def rep(x, y, label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    b = float(np.polyfit(x[m], y[m], 1)[0])
    print("  %-58s corr %+0.4f   slope %+0.6f   n %d" % (label, r, b, m.sum()))
    return r


print("-- between-game correlations, ACTUAL --")
rep(g["P"].values, g["efg"].values, "corr(possessions, eFG%)   [raw]")
rep(g["P"].values, g["ppp"].values, "corr(possessions, PPP)    [raw]")
rep(g["P"].values, g["trans"].values, "corr(possessions, transition share)")
print()

# ------------------------------- MIX vs WITHIN decomposition of eFG
cls = {"rim": ("fga_rim", "fgm_rim", 1.0),
       "jump2": ("fga_jump2", "fgm_jump2", 1.0),
       "three": ("fga_3", "fgm_3", 1.5)}
agg = {}
for k, (a, m, w) in cls.items():
    agg["%s_a" % k] = (a, "sum")
    agg["%s_m" % k] = (m, "sum")
gc = p.groupby("game_id").agg(**agg).loc[g.index]
ebar = {k: float(p[m].sum() * w / p[a].sum()) for k, (a, m, w) in cls.items()}
print("-- season-mean eFG per shot class (ACTUAL) --")
for k, v in ebar.items():
    print("  %-8s %.4f   FGA share %.4f" % (k, v, p[cls[k][0]].sum() / p["fga"].sum()))
print()
tot_a = sum(gc["%s_a" % k] for k in cls)
mix = sum((gc["%s_a" % k] / tot_a) * ebar[k] for k in cls).values
within = g["efg"].values - mix

print("-- eFG decomposed: shot-class MIX vs WITHIN-class make rates, ACTUAL --")
rep(g["P"].values, mix, "corr(P, eFG shot-class MIX term)")
rep(g["P"].values, within, "corr(P, eFG WITHIN-class term)")
cov_t = float(np.cov(g["P"].values, g["efg"].values)[0, 1])
cov_m = float(np.cov(g["P"].values, mix)[0, 1])
cov_w = float(np.cov(g["P"].values, within)[0, 1])
print("  Cov(P,eFG) = %+.6f  =  MIX %+.6f (%+.1f%%) + WITHIN %+.6f (%+.1f%%)"
      % (cov_t, cov_m, 100 * cov_m / cov_t, cov_w, 100 * cov_w / cov_t))
print()

# --------------------------- composition-implied pace vs residual pace
dbar = t_reason["dur"].to_dict()
cnt = p.groupby(["game_id", "start_reason"]).size().unstack(fill_value=0)
cnt = cnt.reindex(g.index).fillna(0.0)
share = cnt.div(cnt.sum(axis=1), axis=0)
mdur_hat = sum(share[c] * dbar[c] for c in share.columns if c in dbar)
P_comp = (g["live"].values / mdur_hat.values)
P_resid = g["P"].values - P_comp

print("-- pace split: composition-implied vs residual, ACTUAL --")
print("  SD(P) %.3f   SD(P_comp) %.3f   SD(P_resid) %.3f   corr(P_comp,P_resid) %+.4f"
      % (g["P"].std(ddof=0), P_comp.std(ddof=0), P_resid.std(ddof=0),
         np.corrcoef(P_comp, P_resid)[0, 1]))
rep(P_comp, g["efg"].values, "corr(P_comp  [efficiency->pace arrow], eFG%)")
rep(P_resid, g["efg"].values, "corr(P_resid [tempo latent],          eFG%)")
cov_c = float(np.cov(P_comp, g["efg"].values)[0, 1])
cov_r = float(np.cov(P_resid, g["efg"].values)[0, 1])
print("  Cov(P,eFG) = %+.6f  =  COMP %+.6f (%+.1f%%) + RESID %+.6f (%+.1f%%)"
      % (cov_t, cov_c, 100 * cov_c / cov_t, cov_r, 100 * cov_r / cov_t))
print()

# --------------------------------- conditioned on start_reason cell
print("-- corr(game P, eFG%) computed WITHIN each start_reason cell, ACTUAL --")
for c in ["DREB", "made_FG", "TOV", "made_FT", "OREB", "period_start", "other",
          "dead_ball", "steal"]:
    sub = p[p["start_reason"] == c]
    if len(sub) < 5000:
        continue
    gg = sub.groupby("game_id").agg(fga=("fga", "sum"), num=("efg_num", "sum"))
    gg = gg[gg["fga"] >= 8].join(g[["P"]], how="inner")
    if len(gg) < 500:
        print("  %-14s UNDERPOWERED n=%d" % (c, len(gg)))
        continue
    rep(gg["P"].values, (gg["num"] / gg["fga"]).values,
        "%-14s corr(game P, eFG%% in this cell)" % c)
print()

# --------------------------------- team-tempo quintile responsiveness
tm = p.groupby(["game_id", "offense_team_id"]).agg(
    n=("dur", "size"), fga=("fga", "sum"), num=("efg_num", "sum"))
tmean = tm.groupby("offense_team_id")["n"].mean()
q = pd.qcut(tmean, 5, labels=[1, 2, 3, 4, 5])
tm = tm.join(q.rename("q"), on="offense_team_id").join(g[["P"]], on="game_id")
print("-- corr(game P, team eFG%) by team TEMPO quintile, ACTUAL --")
for k in [1, 2, 3, 4, 5]:
    s = tm[(tm["q"] == k) & (tm["fga"] >= 25)].dropna(subset=["P"])
    e = (s["num"] / s["fga"]).values
    print("  tempo q%d  n %6d  corr %+0.4f  mean poss/team %.2f  mean eFG %.4f"
          % (k, len(s), np.corrcoef(s["P"].values, e)[0, 1], s["n"].mean(), e.mean()))
