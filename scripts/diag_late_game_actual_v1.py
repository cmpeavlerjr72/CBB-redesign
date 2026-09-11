"""diag_late_game_actual_v1.py -- the ACTUAL 2024-25 late-game regime.

Lane: late-game regime (2026-09-11).  MEASUREMENT ONLY.  Nothing is fitted,
nothing in src/ is touched, no served default is changed.

Window: the final 2:00 of REGULATION (period == 2, start_clock <= 120),
conditioned on |home margin at the 2:00 mark| <= 6.  Reference window: the
rest of regulation (period <= 2, start_clock > 120).

Truth for possessions: data/processed/possessions_v2/chances_2025.parquet.
`duration_s` is the L5-flagged POST-OUTCOME duration; it is used only for the
hold/duration lines and is labelled as such wherever it appears.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/late_game/actual_2025.txt"
CH = "data/processed/possessions_v2/chances_2025.parquet"
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


df = pd.read_parquet(CH)
df = df.sort_values(["game_id", "period", "poss_index", "chance_number"]).reset_index(drop=True)
sgn = np.where(df["offense_is_home"].values, 1.0, -1.0)
df["sgn"] = sgn
df["home_delta"] = df["points"].values * sgn
df["home_after"] = df.groupby("game_id")["home_delta"].cumsum()
df["home_before"] = df["home_after"] - df["home_delta"]
chk = (df["home_before"].values * sgn) - df["start_score_diff"].values
P("cross-check home_before vs start_score_diff: mean_abs_diff %.4f  P_exact %.4f  n %d"
  % (np.abs(chk).mean(), (chk == 0).mean(), len(df)))

reg = df[df["period"] <= 2]
regm = reg.groupby("game_id")["home_after"].last()
P("games %d   regulation tie rate %.5f   abs_reg_margin mean %.3f  SD %.4f"
  % (len(regm), (regm == 0).mean(), regm.abs().mean(), regm.std(ddof=0)))

win = df[(df["period"] == 2) & (df["start_clock"] <= 120)].copy()
first = win.groupby("game_id").first()
m120 = first["home_before"]
P("margin-at-2:00 anchored at mean start_clock %.2f s (median %.1f), games %d"
  % (first["start_clock"].mean(), first["start_clock"].median(), len(first)))

ok = m120.index.intersection(regm.index)
m120 = m120.loc[ok]
regm2 = regm.loc[ok]
P("games with abs margin at 2:00 <= 6: %d (%.4f of %d)"
  % ((m120.abs() <= 6).sum(), (m120.abs() <= 6).mean(), len(ok)))

gm = m120.to_dict()
df["m120"] = df["game_id"].map(gm)
df = df[df["m120"].notna()].copy()
df["win"] = (df["period"] == 2) & (df["start_clock"] <= 120)
df["off_m120"] = df["m120"].values * df["sgn"].values
df["fga"] = df["fga_rim"].values + df["fga_jump2"].values + df["fga_3"].values


def block(d, label):
    n = len(d)
    if n == 0:
        P("  %-24s n=0 EMPTY" % label)
        return None
    if n < 200:
        label = label + " UNDERPOWERED"
    fg = d["fga"].values
    te = d["terminal_event"].values
    P("  %-34s n%7d dur %6.2f med %4.1f Pgt25 %.4f Ple8 %.4f | FGA/p %.4f 3PAsh %.4f | FTA/p %.4f FTA/FGA %.4f | TOV %.4f bonusFT %.4f shootFT %.4f | PPP %.4f"
      % (label, n, d["duration_s"].mean(), d["duration_s"].median(),
         (d["duration_s"] > 25).mean(), (d["duration_s"] <= 8).mean(),
         fg.mean(), d["fga_3"].sum() / max(fg.sum(), 1), d["fta"].mean(),
         d["fta"].sum() / max(fg.sum(), 1e-9), (te == "TOV").mean(),
         (te == "FT_trip_bonus").mean(), (te == "FT_trip_shooting").mean(),
         d["points"].mean()))


P("")
P("=== 1. ACTUAL: end-game window vs rest of regulation (abs margin at 2:00 <= 6) ===")
P("  duration_s is POST-OUTCOME (L5); read the duration lines as a bound.")
C = df[df["m120"].abs() <= 6]
W = C[C["win"]]
R = C[~C["win"]]
block(R, "rest of reg (reference)")
block(W, "final 2:00 (all)")
P("  -- by offense role in the final 2:00 (role set at the 2:00 mark) --")
block(W[W["off_m120"] < 0], "final 2:00 TRAILING")
block(W[W["off_m120"] > 0], "final 2:00 LEADING")
block(W[W["off_m120"] == 0], "final 2:00 TIED")
P("  -- same split in the reference window (separates regime from selection) --")
block(R[R["off_m120"] < 0], "ref TRAILING")
block(R[R["off_m120"] > 0], "ref LEADING")

P("")
P("=== 1b. ACTUAL: the window by seconds-remaining bucket (abs m120 <= 6) ===")
for lo, hi in [(90, 120), (60, 90), (30, 60), (10, 30), (0, 10)]:
    block(W[(W["start_clock"] > lo) & (W["start_clock"] <= hi)], "sec in (%d,%d]" % (lo, hi))

P("")
P("=== 1c. ACTUAL: intentional-foul channel, by clock and role ===")
P("  bonus-FT rate on a TRAILING offense = the LEADING defence fouling.")
for lo, hi in [(60, 120), (30, 60), (10, 30), (0, 10)]:
    b = W[(W["start_clock"] > lo) & (W["start_clock"] <= hi) & (W["off_m120"] < 0)]
    b2 = W[(W["start_clock"] > lo) & (W["start_clock"] <= hi) & (W["off_m120"] > 0)]
    tag = "" if (len(b) >= 200 and len(b2) >= 200) else "  UNDERPOWERED"
    P("  sec (%3d,%3d] trail-off: bonusFT %.4f FTA/p %.4f 3PAsh %.4f dur %5.2f n%6d | lead-off: bonusFT %.4f FTA/p %.4f 3PAsh %.4f dur %5.2f n%6d%s"
      % (lo, hi, (b["terminal_event"] == "FT_trip_bonus").mean() if len(b) else float("nan"),
         b["fta"].mean() if len(b) else float("nan"),
         b["fga_3"].sum() / max(b["fga"].sum(), 1),
         b["duration_s"].mean() if len(b) else float("nan"), len(b),
         (b2["terminal_event"] == "FT_trip_bonus").mean() if len(b2) else float("nan"),
         b2["fta"].mean() if len(b2) else float("nan"),
         b2["fga_3"].sum() / max(b2["fga"].sum(), 1),
         b2["duration_s"].mean() if len(b2) else float("nan"), len(b2), tag))

P("")
P("=== 1d. ACTUAL: possessions in the final 2:00, by abs m120 bucket ===")
for k in range(0, 7):
    b = W[W["m120"].abs() == k]
    ng = b["game_id"].nunique()
    if ng == 0:
        continue
    P("  abs_m120=%d  games %4d  poss/game %.3f  dur %5.2f  3PAsh %.4f  FTA/p %.4f  bonusFT %.4f  TOV %.4f"
      % (k, ng, len(b) / ng, b["duration_s"].mean(),
         b["fga_3"].sum() / max(b["fga"].sum(), 1), b["fta"].mean(),
         (b["terminal_event"] == "FT_trip_bonus").mean(),
         (b["terminal_event"] == "TOV").mean()))

P("")
P("=== 2. ACTUAL: the end-game kernel P(reg margin | margin at 2:00) ===")
K = pd.DataFrame({"m120": m120.values, "reg": regm2.values}, index=ok)
K["am"] = K["m120"].abs()
P("  %5s %7s %9s %10s %10s %10s" % ("absm", "games", "P(m120)", "P(tie|m)", "P(reg1|m)", "P(reg<=1)"))
for k in range(0, 21):
    b = K[K["am"] == k]
    if len(b) == 0:
        continue
    pt = (b["reg"] == 0).mean()
    p1 = (b["reg"].abs() == 1).mean()
    flag = "  UNDERPOWERED" if len(b) < 50 else ""
    P("  %5d %7d %9.5f %10.5f %10.5f %10.5f%s"
      % (k, len(b), len(b) / len(K), pt, p1, pt + p1, flag))
b = K[K["am"] > 20]
P("  %5s %7d %9.5f %10.5f %10.5f %10.5f" % ("gt20", len(b), len(b) / len(K),
  (b["reg"] == 0).mean(), (b["reg"].abs() == 1).mean(),
  (b["reg"] == 0).mean() + (b["reg"].abs() == 1).mean()))

P("")
P("  where the ties COME FROM (share of all regulation ties by abs m120):")
allp = (K["reg"] == 0).mean()
for k in range(0, 13):
    b = K[K["am"] == k]
    if len(b) == 0:
        continue
    c = len(b) / len(K) * (b["reg"] == 0).mean()
    P("    abs_m120=%2d  share of all ties %.4f  (%d tie games of %d)"
      % (k, c / allp, int((b["reg"] == 0).sum()), len(b)))

P("")
P("  reg-margin distribution conditioned on abs m120 <= 6 (the close set):")
b = K[K["am"] <= 6]
for j in range(0, 11):
    P("    abs_reg=%2d  P=%.5f  n=%d" % (j, (b["reg"].abs() == j).mean(),
                                         int((b["reg"].abs() == j).sum())))
P("  close-set excess kurtosis %.4f  SD %.4f  n %d"
  % (b["reg"].kurtosis(), b["reg"].std(ddof=0), len(b)))
P("  ALL-games reg-margin excess kurtosis %.4f  SD %.4f"
  % (K["reg"].kurtosis(), K["reg"].std(ddof=0)))

P("")
P("=== 3. A REGIME-FREE reference kernel K0 (mid-game 2:00 windows, same data) ===")
P("  K0 = the 2-minute margin CHANGE over FIRST-HALF windows at the same margin.")
P("  It is what a model with no late-game regime produces, measured on real basketball.")
mid = df[df["period"] == 1]
recs = []
for anchor in (240, 360, 480, 600, 720, 840, 960):
    f = mid[mid["start_clock"] <= anchor].groupby("game_id").first()
    f2 = mid[mid["start_clock"] <= anchor - 120].groupby("game_id").first()
    j = f[["home_before"]].join(f2[["home_before"]], rsuffix="_end", how="inner").dropna()
    recs.append(pd.DataFrame({"m0": j["home_before"].values,
                              "d": (j["home_before_end"] - j["home_before"]).values}))
K0 = pd.concat(recs, ignore_index=True)
P("  K0 rows %d   mean delta %.4f   SD %.4f   excess kurt %.4f"
  % (len(K0), K0["d"].mean(), K0["d"].std(ddof=0), K0["d"].kurtosis()))
Kl = pd.DataFrame({"m0": m120.values, "d": (regm2.values - m120.values)})
P("  late 2:00 delta (ACTUAL): rows %d  mean %.4f  SD %.4f  excess kurt %.4f"
  % (len(Kl), Kl["d"].mean(), Kl["d"].std(ddof=0), Kl["d"].kurtosis()))


def kern(KK):
    a = KK.copy()
    a["am"] = a["m0"].abs()
    a["sd"] = np.where(a["m0"] < 0, -a["d"], a["d"])
    out = {}
    for k in range(0, 25):
        b = a[a["am"] == k]
        if len(b) < 30:
            continue
        out[k] = ((b["am"] + b["sd"]) == 0).mean()
    return out


kl = kern(Kl)
k0 = kern(K0)
P("")
P("  P(land exactly on 0 after a 2:00 window), by abs margin at window start:")
P("  %5s %11s %11s %8s" % ("absm", "LATE(act)", "MID(K0)", "ratio"))
for k in sorted(set(kl) & set(k0)):
    if k > 12:
        break
    P("  %5d %11.5f %11.5f %8.3f" % (k, kl[k], k0[k], kl[k] / max(k0[k], 1e-9)))

P("")
P("=== 4. Attribution scaffold: P(tie) = sum_k P(absm120=k) * P(land on 0 | k) ===")
dist_act = K["am"].value_counts(normalize=True).sort_index()


def mix(dist, kk):
    s = 0.0
    cov = 0.0
    for k, p in dist.items():
        if k in kk:
            s += p * kk[k]
            cov += p
    return s, cov


a, ca = mix(dist_act, kl)
bb, cb = mix(dist_act, k0)
P("  ACT dist x LATE kernel : %.5f  (coverage %.4f)  [= observed tie rate]" % (a, ca))
P("  ACT dist x MID  kernel : %.5f  (coverage %.4f)  [regime-free counterfactual]" % (bb, cb))
P("  kernel factor (late/mid): %.4f" % (a / max(bb, 1e-9)))
import json
json.dump({"kl": {str(k): v for k, v in kl.items()},
           "k0": {str(k): v for k, v in k0.items()},
           "dist_act": {str(k): float(v) for k, v in dist_act.items()},
           "act_tie": float((K["reg"] == 0).mean())},
          open("results/late_game/actual_kernel.json", "w"), indent=1)

P("")
P("=== 5. By team quintile (offense team season PPG), final-2:00 window ===")
tp = pd.concat([df.groupby("offense_team_id")["points"].sum().rename("pts"),
                df.groupby("offense_team_id")["game_id"].nunique().rename("g")], axis=1)
tp = tp[tp["g"] >= 10]
tp["ppg"] = tp["pts"] / tp["g"]
tp["q"] = pd.qcut(tp["ppg"], 5, labels=[1, 2, 3, 4, 5])
qm = tp["q"].to_dict()
W2 = W.copy()
W2["q"] = W2["offense_team_id"].map(qm)
R2 = R.copy()
R2["q"] = R2["offense_team_id"].map(qm)
P("  %3s %8s %9s %9s %9s %8s %10s" % ("q", "n_win", "3PAsh", "FTA/p", "bonusFT", "dur", "ref_3PAsh"))
for q in [1, 2, 3, 4, 5]:
    b = W2[W2["q"] == q]
    r = R2[R2["q"] == q]
    if len(b) < 200:
        P("  %3s UNDERPOWERED n=%d" % (q, len(b)))
        continue
    P("  %3d %8d %9.4f %9.4f %9.4f %8.2f %10.4f"
      % (q, len(b), b["fga_3"].sum() / max(b["fga"].sum(), 1), b["fta"].mean(),
         (b["terminal_event"] == "FT_trip_bonus").mean(), b["duration_s"].mean(),
         r["fga_3"].sum() / max(r["fga"].sum(), 1)))

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("")
print("wrote " + OUT)
