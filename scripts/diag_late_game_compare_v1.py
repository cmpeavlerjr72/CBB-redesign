"""diag_late_game_compare_v1.py -- SIM vs ACTUAL end-game, and the tie attribution.

Lane: late-game regime (2026-09-11).  MEASUREMENT ONLY.  Reads the tap written
by diag_late_game_tap_v1.py and the actual chances table.  Fits nothing.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAP = Path(sys.argv[1] if len(sys.argv) > 1 else "results/late_game/tap_s5")
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/late_game/compare_2025.txt"
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


clk = pd.read_parquet(TAP / "clock_tap.parquet")
ev = pd.read_parquet(TAP / "event_tap.parquet")
G = pd.read_parquet(TAP / "games.parquet")
CLS = [c for c in ev.columns if c not in
       ("seed", "gidx", "is_first", "period", "sec", "sd", "bonus", "chance")]
P("tap: %s   possessions %d  chances %d  sims %d  classes %s"
  % (TAP, len(clk), len(ev), len(G), CLS))

G["reg"] = np.where(G["n_periods"] > 2, 0, G["home_pts"] - G["away_pts"])
P("SIM regulation tie rate %.5f  abs_reg mean %.3f  SD %.4f  n %d"
  % ((G["reg"] == 0).mean(), G["reg"].abs().mean(), G["reg"].std(ddof=0), len(G)))
P("SIM reg-margin excess kurtosis %.4f" % G["reg"].kurtosis())

# ---- margin at the 2:00 mark, sim: first possession of period 2 with sec <= 120
w = clk[(clk["period"] == 2) & (clk["sec"] <= 120)].copy()
w = w.sort_values(["seed", "gidx", "sec"], ascending=[True, True, False])
f = w.groupby(["seed", "gidx"]).first().reset_index()
P("SIM margin-at-2:00 anchored at mean sec %.2f (median %.1f), sims %d"
  % (f["sec"].mean(), f["sec"].median(), len(f)))
f["am"] = f["sd"].abs()
key = G.set_index(["seed", "gidx"])["reg"]
f = f.join(key, on=["seed", "gidx"])
f = f[f["reg"].notna()]
P("SIM sims with abs margin at 2:00 <= 6: %d (%.4f)" % ((f["am"] <= 6).sum(), (f["am"] <= 6).mean()))

A = json.load(open("results/late_game/actual_kernel.json"))
kl = {int(k): v for k, v in A["kl"].items()}
k0 = {int(k): v for k, v in A["k0"].items()}
dist_act = {int(float(k)): v for k, v in A["dist_act"].items()}
act_tie = A["act_tie"]

ksim, dsim = {}, {}
for k in range(0, 25):
    b = f[f["am"] == k]
    dsim[k] = len(b) / len(f)
    if len(b) >= 30:
        ksim[k] = (b["reg"] == 0).mean()

P("")
P("=== A. The margin-at-2:00 DISTRIBUTION, sim vs actual ===")
P("  %5s %10s %10s %8s" % ("absm", "SIM P", "ACT P", "ratio"))
for k in range(0, 16):
    pa = dist_act.get(k, 0.0)
    P("  %5d %10.5f %10.5f %8.3f" % (k, dsim.get(k, 0.0), pa,
                                     dsim.get(k, 0.0) / max(pa, 1e-9)))
sa = sum(v for k, v in dist_act.items() if k <= 6)
ss = sum(v for k, v in dsim.items() if k <= 6)
P("  P(abs m120 <= 6): SIM %.5f  ACT %.5f  ratio %.4f" % (ss, sa, ss / sa))
P("  SD of margin at 2:00: SIM %.4f  (ACT from the same definition, see actual_2025.txt)"
  % f["sd"].std(ddof=0))

P("")
P("=== B. The END-GAME KERNEL P(land on 0 | abs margin at 2:00) ===")
P("  %5s %10s %10s %10s %9s %9s" % ("absm", "SIM", "ACT", "MID K0", "SIM/ACT", "SIM/K0"))
for k in sorted(set(ksim) & set(kl)):
    if k > 12:
        break
    P("  %5d %10.5f %10.5f %10.5f %9.3f %9.3f"
      % (k, ksim[k], kl[k], k0.get(k, float("nan")),
         ksim[k] / max(kl[k], 1e-9), ksim[k] / max(k0.get(k, 1e-9), 1e-9)))


def mix(dist, kk):
    s = 0.0
    cov = 0.0
    for k, p in dist.items():
        if k in kk:
            s += p * kk[k]
            cov += p
    return s, cov


common = sorted(set(ksim) & set(kl))
ks = {k: ksim[k] for k in common}
ka = {k: kl[k] for k in common}
km = {k: k0[k] for k in common if k in k0}
ss_, cs = mix(dsim, ks)
aa_, ca = mix(dist_act, ka)
ax, _ = mix(dist_act, ks)
sx, _ = mix(dsim, ka)
P("")
P("=== C. (a)/(b) ATTRIBUTION of the tie shortfall ===")
P("  All four cells use the SAME kernel support (abs m120 in %s)." % common)
P("  sim dist x sim kernel  = %.5f   [sim, restricted support]" % ss_)
P("  act dist x sim kernel  = %.5f   [(a) alone: fix the 2:00 distribution]" % ax)
P("  sim dist x act kernel  = %.5f   [(b) alone: fix the end-game kernel]" % sx)
P("  act dist x act kernel  = %.5f   [actual, restricted support]" % aa_)
gap = aa_ - ss_
P("  total restricted gap   = %.5f" % gap)
if abs(gap) > 1e-9:
    P("  (a) upstream 2:00-margin distribution : %+.5f  = %+.1f%% of the gap"
      % (ax - ss_, 100 * (ax - ss_) / gap))
    P("  (b) end-game kernel                  : %+.5f  = %+.1f%% of the gap"
      % (sx - ss_, 100 * (sx - ss_) / gap))
    P("  interaction                          : %+.5f  = %+.1f%%"
      % (gap - (ax - ss_) - (sx - ss_), 100 * (gap - (ax - ss_) - (sx - ss_)) / gap))
P("  unrestricted reference: SIM tie %.5f (this tap) vs ACT tie %.5f"
  % ((G["reg"] == 0).mean(), act_tie))
P("  regime-free reference kernel K0 on the ACTUAL 2:00 distribution: %.5f"
  % mix(dist_act, km)[0])

# ----------------------------------------------------- D. window behaviour
P("")
P("=== D. SIM end-game behaviour in the final 2:00 (abs m120 <= 6) ===")
m = f.set_index(["seed", "gidx"])["am"]
clk = clk.join(m.rename("am"), on=["seed", "gidx"])
ev = ev.join(m.rename("am"), on=["seed", "gidx"])
mm = f.set_index(["seed", "gidx"])["sd"].rename("m120sd")
ev = ev.join(mm, on=["seed", "gidx"])
cw = clk[(clk["am"] <= 6) & (clk["period"] == 2) & (clk["sec"] <= 120)]
cr = clk[(clk["am"] <= 6) & (clk["period"] <= 2) & ~((clk["period"] == 2) & (clk["sec"] <= 120))]
P("  duration: window mean %.3f med %.1f Pgt25 %.4f Ple8 %.4f  n %d"
  % (cw["dur"].mean(), cw["dur"].median(), (cw["dur"] > 25).mean(),
     (cw["dur"] <= 8).mean(), len(cw)))
P("  duration: ref    mean %.3f med %.1f Pgt25 %.4f Ple8 %.4f  n %d"
  % (cr["dur"].mean(), cr["dur"].median(), (cr["dur"] > 25).mean(),
     (cr["dur"] <= 8).mean(), len(cr)))
P("  ACTUAL for the same cells: window 10.56 (med 8, Pgt25 .0987, Ple8 .5217); ref 15.78 (med 16, Pgt25 .1619, Ple8 .2634)")
n_w_games = cw.groupby(["seed", "gidx"]).size()
P("  possessions per game in the final 2:00: SIM %.3f   ACTUAL 9.24-11.34 by abs m120 bucket"
  % n_w_games.mean())

ew = ev[(ev["am"] <= 6) & (ev["period"] == 2) & (ev["sec"] <= 120) & (ev["is_first"] > 0)]
er = ev[(ev["am"] <= 6) & (ev["period"] <= 2) & (ev["is_first"] > 0)
        & ~((ev["period"] == 2) & (ev["sec"] <= 120))]


def emix(d, label):
    if len(d) < 200:
        P("  %-28s n=%d UNDERPOWERED" % (label, len(d)))
        return
    mu = d[CLS].mean()
    fga = sum(mu[c] for c in CLS if c.startswith("FGA"))
    P("  %-28s n%7d FGA/p %.4f 3PAsh %.4f TOV %.4f bonusFT %.4f shootFT %.4f"
      % (label, len(d), fga, mu.get("FGA_3", np.nan) / max(fga, 1e-9),
         mu.get("TOV", np.nan), mu.get("FT_trip_bonus", np.nan),
         mu.get("FT_trip_shooting", np.nan)))


P("  -- event-model served class probabilities (first chance) --")
emix(er, "SIM ref (rest of reg)")
emix(ew, "SIM final 2:00 (all)")
emix(ew[ew["m120sd"] < 0], "SIM final 2:00 TRAILING")
emix(ew[ew["m120sd"] > 0], "SIM final 2:00 LEADING")
P("  ACTUAL ref        : FGA/p 0.7398 3PAsh 0.3830 TOV 0.1492 bonusFT 0.0364 shootFT 0.0689")
P("  ACTUAL final 2:00 : FGA/p 0.5626 3PAsh 0.4219 TOV 0.1068 bonusFT 0.2775 shootFT 0.0274")
P("  ACTUAL trailing   : FGA/p 0.6914 3PAsh 0.4788 TOV 0.0914 bonusFT 0.1564")
P("  ACTUAL leading    : FGA/p 0.4126 3PAsh 0.3213 TOV 0.1239 bonusFT 0.4216")

P("  -- by seconds-remaining bucket, SIM (all roles) --")
for lo, hi in [(90, 120), (60, 90), (30, 60), (10, 30), (0, 10)]:
    b = ew[(ew["sec"] > lo) & (ew["sec"] <= hi)]
    c = cw[(cw["sec"] > lo) & (cw["sec"] <= hi)]
    if len(b) < 200:
        P("  sec (%d,%d] UNDERPOWERED n=%d" % (lo, hi, len(b)))
        continue
    mu = b[CLS].mean()
    fga = sum(mu[cc] for cc in CLS if cc.startswith("FGA"))
    P("  sec (%3d,%3d] n%6d FGA/p %.4f 3PAsh %.4f TOV %.4f bonusFT %.4f dur %5.2f"
      % (lo, hi, len(b), fga, mu.get("FGA_3", np.nan) / max(fga, 1e-9),
         mu.get("TOV", np.nan), mu.get("FT_trip_bonus", np.nan), c["dur"].mean()))
P("  ACTUAL same buckets 3PAsh .3562 .3562 .3845 .4803 .6324 ; bonusFT .1623 .1957 .2768 .3874 .3749 ; dur 16.20 15.88 11.39 6.56 2.89")

# ------------------------------------ E. which PLAY creates the tie (actual)
P("")
P("=== E. ACTUAL: which play type lands the game exactly on 0 ===")
ch = pd.read_parquet("data/processed/possessions_v2/chances_2025.parquet")
ch = ch.sort_values(["game_id", "period", "poss_index", "chance_number"]).reset_index(drop=True)
sg = np.where(ch["offense_is_home"].values, 1.0, -1.0)
ch["home_after"] = ch.groupby("game_id")[["points"]].transform(
    lambda s: s).mul(sg, axis=0).groupby(ch["game_id"]).cumsum()
ch["home_before"] = ch["home_after"] - ch["points"].values * sg
regm = ch[ch["period"] <= 2].groupby("game_id")["home_after"].last()
tie_games = set(regm[regm == 0].index)
win = ch[(ch["period"] == 2) & (ch["start_clock"] <= 120) & ch["game_id"].isin(tie_games)]
# the last margin-changing possession of regulation in a tie game
lastc = win[win["points"] > 0].groupby("game_id").last()
lastc = lastc[lastc["home_after"] == 0]
tot = len(lastc)
if tot:
    is3 = (lastc["fgm_3"] > 0)
    isft = (lastc["ftm"] > 0) & (lastc["fgm_3"] == 0) & (lastc["fgm_rim"] + lastc["fgm_jump2"] == 0)
    is2 = (lastc["fgm_rim"] + lastc["fgm_jump2"] > 0) & (lastc["fgm_3"] == 0)
    P("  tie games whose LAST scoring possession in the final 2:00 landed on 0: %d" % tot)
    P("    pure free throws : %.4f (n=%d)" % (isft.mean(), int(isft.sum())))
    P("    a made three     : %.4f (n=%d)" % (is3.mean(), int(is3.sum())))
    P("    a made two       : %.4f (n=%d)" % (is2.mean(), int(is2.sum())))
    P("    points on that possession: mean %.3f  P(1)=%.4f P(2)=%.4f P(3)=%.4f"
      % (lastc["points"].mean(), (lastc["points"] == 1).mean(),
         (lastc["points"] == 2).mean(), (lastc["points"] == 3).mean()))
P("  reference: share of all final-2:00 possessions by terminal event (abs m120 <= 6)")
w2 = ch[(ch["period"] == 2) & (ch["start_clock"] <= 120)]
P("    " + w2["terminal_event"].value_counts(normalize=True).round(4).to_string().replace("\n", " | "))

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("")
print("wrote " + OUT)
