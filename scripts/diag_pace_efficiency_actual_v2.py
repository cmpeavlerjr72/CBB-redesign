"""diag_pace_efficiency_actual_v2.py -- the LIKE-FOR-LIKE actual read
(lane: engine diagnosis, 2026-09-11).

v1 reads the actual BETWEEN games; the engine's -0.198 is a WITHIN-game,
across-seed read.  A season has one realisation per game, so the within-matchup
object is not directly observable.  This script builds the closest honest
substitute: residualise BOTH possessions and eFG on the matchup (each side's
own season means plus the site), then correlate the residuals.  What is left is
the part of a game that a re-draw of the same matchup could have produced,
which is what the engine's across-seed read measures.

WARNING: section 4 reports the BANNED post-outcome transition band (see
the comment there); it is kept as the cautionary comparison only.
It also reports the cell tables the sim side needs for comparison:
per-shot-class make rates and FGA shares inside vs outside the transition band,
and the per-class FGA-share response to pace.

Read-only.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
POSS = "data/processed/possessions_v2/possessions_%d.parquet" % SEASON
CHAN = "data/processed/possessions_v2/chances_%d.parquet" % SEASON

p = pd.read_parquet(POSS, columns=[
    "game_id", "period", "offense_team_id", "defense_team_id", "offense_is_home",
    "duration_s", "start_reason", "terminal_event", "n_chances", "is_transition",
    "fga_rim", "fgm_rim", "fga_jump2", "fgm_jump2", "fga_3", "fgm_3",
    "fta", "ftm", "points"])
p = p[p["period"] <= 2].copy()
p["fga"] = p["fga_rim"] + p["fga_jump2"] + p["fga_3"]
p["efg_num"] = p["fgm_rim"] + p["fgm_jump2"] + 1.5 * p["fgm_3"]
p["dur"] = p["duration_s"].astype(float).clip(lower=0)


def rep(x, y, label):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    print("  %-56s corr %+0.4f  slope %+0.6f  n %d"
          % (label, np.corrcoef(x[m], y[m])[0, 1], np.polyfit(x[m], y[m], 1)[0], m.sum()))


# ------------------------------------------------------------ 1. per game
g = p.groupby("game_id").agg(P=("dur", "size"), fga=("fga", "sum"),
                             num=("efg_num", "sum"), pts=("points", "sum"),
                             trans=("is_transition", "mean"))
g["efg"] = g["num"] / g["fga"]
g = g[(g["P"] >= 80) & (g["fga"] >= 60)]
teams = (p.groupby("game_id")["offense_team_id"].unique()
         .apply(lambda a: tuple(sorted(a)[:2])))
g = g.join(teams.rename("pair"))
g = g[g["pair"].apply(len) == 2]
g["t1"] = g["pair"].apply(lambda t: t[0])
g["t2"] = g["pair"].apply(lambda t: t[1])

print("=== ACTUAL %d, like-for-like read, %d games ===\n" % (SEASON, len(g)))
print("-- raw between-game (v1 numbers, for scale) --")
rep(g["P"], g["efg"], "corr(P, eFG%)")

# ------------------------------- 2. residualise on the matchup
# additive team effects, fitted by two passes of centring (a one-way ANOVA
# projection on the two team factors).  This removes the part of P and eFG that
# is predictable from WHICH teams are playing -- the between-matchup structure
# the engine's within-game read conditions away.
def matchup_residual(col: str) -> np.ndarray:
    """observed - (t1 season mean + t2 season mean - league mean).

    One pass, no iteration: an additive two-team expectation built from each
    side's own season mean of the quantity.  The game itself is inside its two
    team means, which biases the residual SD DOWN by about 1/n_games_per_team
    (~1/29); stated, not corrected.
    """
    lg = pd.concat([pd.DataFrame({"team": g["t1"].values, "v": g[col].values}),
                    pd.DataFrame({"team": g["t2"].values, "v": g[col].values})])
    eff = lg.groupby("team")["v"].mean()
    mu = float(g[col].mean())
    exp = eff.reindex(g["t1"]).values + eff.reindex(g["t2"]).values - mu
    r = g[col].values - exp
    return r - np.nanmean(r)


for col in ("P", "efg"):
    g[col + "_res"] = matchup_residual(col)

print("\n-- residualised on the two teams (the like-for-like object) --")
print("  SD(P) raw %.3f -> residual %.3f      SD(eFG) raw %.5f -> residual %.5f"
      % (g["P"].std(ddof=0), g["P_res"].std(ddof=0),
         g["efg"].std(ddof=0), g["efg_res"].std(ddof=0)))
rep(g["P_res"], g["efg_res"], "corr(P, eFG%)  matchup-residualised  [SIM -0.1976]")
pass  # transition share residual is reported after `mix` below

# ---------------------------- 3. shot-class MIX vs WITHIN, residualised
cls = {"rim": ("fga_rim", "fgm_rim", 1.0), "jump2": ("fga_jump2", "fgm_jump2", 1.0),
       "three": ("fga_3", "fgm_3", 1.5)}
agg = {}
for k, (a, m, w) in cls.items():
    agg["%s_a" % k] = (a, "sum")
    agg["%s_m" % k] = (m, "sum")
gc = p.groupby("game_id").agg(**agg).reindex(g.index)
ebar = {k: float(p[m].sum() * w / p[a].sum()) for k, (a, m, w) in cls.items()}
tot = sum(gc["%s_a" % k] for k in cls)
g["mix"] = sum((gc["%s_a" % k] / tot) * ebar[k] for k in cls)
g["wthn"] = g["efg"] - g["mix"]
for col in ("mix", "wthn", "trans"):
    g[col + "_res"] = matchup_residual(col)

print("\n-- eFG: shot-class MIX vs WITHIN-class, matchup-residualised, ACTUAL --")
rep(g["P_res"], g["mix_res"], "corr(P, MIX term)     [SIM +0.0322]")
rep(g["P_res"], g["wthn_res"], "corr(P, WITHIN term)  [SIM -0.2013]")
ct = float(np.cov(g["P_res"], g["efg_res"])[0, 1])
cm = float(np.cov(g["P_res"], g["mix_res"])[0, 1])
cw = float(np.cov(g["P_res"], g["wthn_res"])[0, 1])
print("  Cov(P,eFG) %+.6f = MIX %+.6f + WITHIN %+.6f" % (ct, cm, cw))
print("  slope split: total %+.6f = MIX %+.6f + WITHIN %+.6f"
      % (ct / g["P_res"].var(ddof=0), cm / g["P_res"].var(ddof=0),
         cw / g["P_res"].var(ddof=0)))

print("\n-- per-class make rate and FGA share vs pace, matchup-residualised, ACTUAL --")
for k, (a, m, w) in cls.items():
    g["_r"] = (gc["%s_m" % k] / gc["%s_a" % k]).values
    g["_s"] = (gc["%s_a" % k] / tot).values
    rep(g["P_res"].values, matchup_residual("_r"), "%-6s make rate" % k)
    rep(g["P_res"].values, matchup_residual("_s"), "%-6s FGA share" % k)
rep(g["P_res"].values, g["trans_res"].values, "transition share  [SIM: see poss log]")

# ------------------------------- 4. the transition cell, per class
# WARNING, READ BEFORE QUOTING THIS TABLE.  `chances.duration_s` is the CHANCE's
# own duration and is POST-OUTCOME: it runs to the REBOUND on a miss and to the
# MAKE on a make, so misses are pushed into longer buckets by construction.  It
# is the quantity the change ledger bans at L5 and `fg_make/features.md`
# section 4 documents.  The lift it reports (+32.5 pp) is FIVE TIMES the honest
# one (+6.5 pp) measured on fg_make's own design column `chance_elapsed_s`.
# The table is kept because the gap between the two IS the finding; it must
# never be quoted as the transition lift.
# Honest numbers: docs/tests/pace_efficiency_sign_2026-09-11.md section 3.2.
print("\n-- ACTUAL per-class rates inside vs outside the transition band "
      "(chance 1, prev_end DREB/TOV) --")
c = pd.read_parquet(CHAN)
c = c[(c["period"] <= 2) & (c["chance_number"] == 1)
      & (c["start_reason"].isin(["DREB", "TOV"]))].copy()
c["fga"] = c["fga_rim"] + c["fga_jump2"] + c["fga_3"]
c["efg_num"] = c["fgm_rim"] + c["fgm_jump2"] + 1.5 * c["fgm_3"]
c["band"] = np.where(c["duration_s"] <= 8, "elapsed<=8", "elapsed>8")
t = c.groupby("band").agg(n=("fga", "size"), fga=("fga", "sum"),
                          num=("efg_num", "sum"),
                          rim_a=("fga_rim", "sum"), rim_m=("fgm_rim", "sum"),
                          j_a=("fga_jump2", "sum"), j_m=("fgm_jump2", "sum"),
                          t_a=("fga_3", "sum"), t_m=("fgm_3", "sum"),
                          tov=("terminal_event", lambda s: (s == "TOV").mean()))
t["efg"] = t["num"] / t["fga"]
t["FGA_per_chance"] = t["fga"] / t["n"]
t["rim_share"] = t["rim_a"] / t["fga"]
t["rim_make"] = t["rim_m"] / t["rim_a"]
t["jump_make"] = t["j_m"] / t["j_a"]
t["three_make"] = t["t_m"] / t["t_a"]
t["share_of_chances"] = t["n"] / t["n"].sum()
print(t[["n", "share_of_chances", "efg", "FGA_per_chance", "rim_share", "rim_make",
         "jump_make", "three_make", "tov"]].round(4).to_string())
lo, hi = t.loc["elapsed>8"], t.loc["elapsed<=8"]
print("  LIFT transition - half court, pp:  eFG %+.2f   rim share %+.2f   "
      "rim make %+.2f   jump make %+.2f   three make %+.2f"
      % (100 * (hi.efg - lo.efg), 100 * (hi.rim_share - lo.rim_share),
         100 * (hi.rim_make - lo.rim_make), 100 * (hi.jump_make - lo.jump_make),
         100 * (hi.three_make - lo.three_make)))
print("  SERVED ENGINE (diag_pace_efficiency_probe_v1 A6/A7): eFG +9.39, "
      "rim share +21.6, rim make +11.3, jump make +5.3, three make +0.8")
