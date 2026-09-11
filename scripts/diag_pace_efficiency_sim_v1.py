"""diag_pace_efficiency_sim_v1.py -- the SIM side of the pace/efficiency sign
question (lane: engine diagnosis, 2026-09-11).

Reads an engine run's games.parquet and reproduces, WITHIN game across seeds,
the same decompositions `diag_pace_efficiency_actual_v1.py` runs BETWEEN games
on the season, so the two are like for like:

  1. within-game corr(P, eFG%)  -- the -0.198 the variance diagnostic names.
  2. eFG split into a shot-class MIX term (the seed's own rim/jump2/three mix
     priced at the run's own season-mean class make rates) and a WITHIN-class
     term.  Cov(P, eFG) splits the same way.
  3. pace split into a COMPOSITION-implied part (the seed's own prev_end mix
     priced at fixed per-state mean durations) and a RESIDUAL part (the clock's
     own draw noise plus any game-level tempo latent).
  4. the same at team-scoring and team-tempo quintile level.

Read-only.  Writes nothing but stdout.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/engine_v0/F2_2025_s200_rewire1"

# per-state mean possession durations measured on the ACTUAL 2025 possessions
# (diag_pace_efficiency_actual_v1.py, table 1).  Fixed weights, identical on
# both sides, so the MIX term prices composition and nothing else.
DUR_BY_REASON = {"made_FG": 21.6609, "DREB": 14.3869, "TOV": 14.7378,
                 "other": 19.4}          # made_FT / dead ball / period start

g = pd.read_parquet("%s/games.parquet" % RUN)
print("=== SIM %s : %d rows, %d games, %d seeds ==="
      % (RUN, len(g), g.game_id.nunique(), g.seed.nunique()))

for s in ["fga3", "fga2_rim", "fga2_jump", "fgm3", "fgm2_rim", "fgm2_jump",
          "tov", "oreb", "dreb", "fta", "ftm"]:
    g[s] = g["home_%s" % s] + g["away_%s" % s]
g["fga"] = g["fga3"] + g["fga2_rim"] + g["fga2_jump"]
g["fgm"] = g["fgm3"] + g["fgm2_rim"] + g["fgm2_jump"]
g["efg"] = (g["fgm"] + 0.5 * g["fgm3"]) / g["fga"]
g["P"] = g["possessions"] * 2.0            # both teams, matches the actual read
g["pts"] = g["home_pts"] + g["away_pts"]
g["ppp"] = g["pts"] / g["P"]
g["live"] = 2400.0 + 300.0 * (g["n_periods"] - 2).clip(lower=0)


def within(df, cols):
    """Demean every column by game: the within-game (across-seed) component."""
    return df[cols] - df.groupby("game_id")[cols].transform("mean")


def rep(x, y, label):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    b = float(np.polyfit(x[m], y[m], 1)[0])
    print("  %-58s corr %+0.4f   slope %+0.6f   n %d" % (label, r, b, m.sum()))
    return r


# ------------------------------------------------- 1. the headline
print("\n-- within-game (across-seed) correlations, SIM --")
w = within(g, ["P", "efg", "ppp"])
rep(w["P"], w["efg"], "corr(P, eFG%)  WITHIN game   [the -0.198 line]")
rep(w["P"], w["ppp"], "corr(P, PPP)   WITHIN game")
rep(g["P"], g["efg"], "corr(P, eFG%)  POOLED")

# ------------------------------- 2. shot-class MIX vs WITHIN
cls = {"rim": ("fga2_rim", "fgm2_rim", 1.0),
       "jump2": ("fga2_jump", "fgm2_jump", 1.0),
       "three": ("fga3", "fgm3", 1.5)}
ebar = {k: float(g[m].sum() * wt / g[a].sum()) for k, (a, m, wt) in cls.items()}
print("\n-- run-mean eFG per shot class (SIM) --")
for k, v in ebar.items():
    print("  %-8s %.4f   FGA share %.4f" % (k, v, g[cls[k][0]].sum() / g["fga"].sum()))

g["mix"] = sum((g[cls[k][0]] / g["fga"]) * ebar[k] for k in cls)
g["wthn"] = g["efg"] - g["mix"]
w = within(g, ["P", "efg", "mix", "wthn"])
print("\n-- eFG decomposed: shot-class MIX vs WITHIN-class make rates, SIM --")
rep(w["P"], w["mix"], "corr(P, eFG shot-class MIX term)   WITHIN game")
rep(w["P"], w["wthn"], "corr(P, eFG WITHIN-class term)     WITHIN game")
cov_t = float(np.cov(w["P"], w["efg"])[0, 1])
cov_m = float(np.cov(w["P"], w["mix"])[0, 1])
cov_w = float(np.cov(w["P"], w["wthn"])[0, 1])
print("  Cov(P,eFG) = %+.6f  =  MIX %+.6f (%+.1f%%) + WITHIN %+.6f (%+.1f%%)"
      % (cov_t, cov_m, 100 * cov_m / cov_t, cov_w, 100 * cov_w / cov_t))

# per class: does the class's own make rate move with P?
print("\n-- within-game corr(P, per-class make rate) and corr(P, per-class FGA share), SIM --")
for k, (a, m, wt) in cls.items():
    g["_r"] = g[m] / g[a].replace(0, np.nan)
    g["_s"] = g[a] / g["fga"]
    ww = within(g.dropna(subset=["_r"]), ["P", "_r", "_s"])
    rep(ww["P"], ww["_r"], "%-8s make rate" % k)
    rep(ww["P"], ww["_s"], "%-8s FGA share" % k)

# ------------------------------- 3. pace: composition vs residual
#  prev_end mix from the box: made_FG starts = FGM, TOV starts = TOV,
#  DREB starts = DREB, everything else (made last FT, dead ball, period start)
#  is the residual count.
n_mfg = g["fgm"].astype(float)
n_tov = g["tov"].astype(float)
n_dreb = g["dreb"].astype(float)
n_oth = (g["P"] - n_mfg - n_tov - n_dreb).clip(lower=0)
tot = n_mfg + n_tov + n_dreb + n_oth
mdur = ((n_mfg * DUR_BY_REASON["made_FG"] + n_tov * DUR_BY_REASON["TOV"]
         + n_dreb * DUR_BY_REASON["DREB"] + n_oth * DUR_BY_REASON["other"]) / tot)
g["P_comp"] = g["live"] / mdur
g["P_resid"] = g["P"] - g["P_comp"]
w = within(g, ["P", "efg", "P_comp", "P_resid"])
print("\n-- pace split: composition-implied vs residual, SIM (within game) --")
print("  SD(P) %.3f   SD(P_comp) %.3f   SD(P_resid) %.3f   corr(P_comp,P_resid) %+.4f"
      % (w["P"].std(ddof=0), w["P_comp"].std(ddof=0), w["P_resid"].std(ddof=0),
         np.corrcoef(w["P_comp"], w["P_resid"])[0, 1]))
rep(w["P_comp"], w["efg"], "corr(P_comp  [efficiency->pace arrow], eFG%)")
rep(w["P_resid"], w["efg"], "corr(P_resid [clock draw noise],      eFG%)")
cov_c = float(np.cov(w["P_comp"], w["efg"])[0, 1])
cov_r = float(np.cov(w["P_resid"], w["efg"])[0, 1])
print("  Cov(P,eFG) = %+.6f  =  COMP %+.6f (%+.1f%%) + RESID %+.6f (%+.1f%%)"
      % (cov_t, cov_c, 100 * cov_c / cov_t, cov_r, 100 * cov_r / cov_t))

# ------------------------------- 4. between vs within, and the prev_end mix
print("\n-- prev_end mix per possession, SIM (share of possession starts) --")
for nm, v in (("made_FG", n_mfg), ("DREB", n_dreb), ("TOV", n_tov), ("other", n_oth)):
    print("  %-10s %.4f" % (nm, float((v / tot).mean())))
print("  implied mean duration %.3f s   mean P %.2f   actual mean P %.2f"
      % (mdur.mean(), g["P"].mean(), 136.2))

# ------------------------------- 5. quintiles
print("\n-- within-game corr(P, eFG) by team-scoring and team-tempo quintile, SIM --")
gm = g.groupby("game_id").agg(mP=("P", "mean"), mpts=("pts", "mean"))
for nm, col in (("tempo", "mP"), ("scoring", "mpts")):
    q = pd.qcut(gm[col], 5, labels=[1, 2, 3, 4, 5])
    gq = g.join(q.rename("q"), on="game_id")
    for k in [1, 2, 3, 4, 5]:
        s = gq[gq["q"] == k]
        ws = within(s, ["P", "efg"])
        print("  %s q%d  n %7d  corr %+0.4f  mean P %.2f  mean eFG %.4f"
              % (nm, k, len(s), np.corrcoef(ws["P"], ws["efg"])[0, 1],
                 s["P"].mean(), s["efg"].mean()))

# ------------------------------- 6. how much pace variance is composition?
print("\n-- within-game pace variance attribution, SIM --")
vP = float(w["P"].var(ddof=0))
vC = float(w["P_comp"].var(ddof=0))
vR = float(w["P_resid"].var(ddof=0))
cCR = float(np.cov(w["P_comp"], w["P_resid"])[0, 1])
print("  Var(P) %.4f = Var(comp) %.4f + Var(resid) %.4f + 2Cov %.4f"
      % (vP, vC, vR, 2 * cCR))
print("  per-team within-game SD(P) %.4f  (gate line: 3.743 produced / 4.972 needed)"
      % (w["P"].std(ddof=0) / 2.0))
