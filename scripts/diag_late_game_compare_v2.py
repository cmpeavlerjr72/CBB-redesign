"""diag_late_game_compare_v2.py -- the SERVED-STACK re-read of the late-game gaps.

Lane: late-game regime, round 1 (2026-09-18).  MEASUREMENT ONLY; fits nothing,
changes no default.

`docs/models/late_game/experiments.md` section 1.0 makes two things blocking
preconditions on the round: a re-read of the duration lines against the SERVED
default (the 2026-09-11 evidence is on run A's `ENGINE_CLOCK=v3c_srfloor_P3_s1`,
the served default became `v5b_glat_pmean`), and a larger re-read of the sim
kernel than the 1,000 simulations the original tap carried.  This script does
both:

  * the WINDOW BEHAVIOUR and the KERNEL come from the sharded v5b tap
    (`diag_late_game_tap_v2.py`), 12,000 simulations against the original 1,000;
  * the MARGIN LEVELS -- tie rate, `P(0)`, `P(1)`, `P(0)/P(1)`, excess kurtosis
    -- come from the full 75-seed served-stack run
    `results/engine_v0/F2_2025_s200_v5b_A` (428,250 simulations over all 5,710
    games), because a 250-game tap cannot carry a level.

The ACTUAL side is unchanged and is read from `results/late_game/actual_kernel.json`
and the tables in `docs/tests/late_game_regime_2026-09-11.md`, so the comparison
moves only the sim side.

Usage: diag_late_game_compare_v2.py [tap_dir] [out_txt]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAP = Path(sys.argv[1] if len(sys.argv) > 1 else "results/late_game/tap_v5b")
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/late_game/compare_v5b_2025.txt"
V5B_RUN = ROOT / "results/engine_v0/F2_2025_s200_v5b_A/games.parquet"
V3C_RUN = ROOT / "results/engine_v0/F2_2025_s200_v1_clockv3c_A/games.parquet"

lines: list[str] = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def shards(name: str) -> pd.DataFrame:
    fs = sorted(TAP.glob(f"{name}_s*.parquet"))
    if not fs:
        raise FileNotFoundError(f"no {name} shards in {TAP}")
    out = []
    for i, f in enumerate(fs):
        x = pd.read_parquet(f)
        # shards share seeds but hold DISJOINT games, so (seed, gidx) is already
        # unique across shards; the shard id is carried for provenance only.
        x["shard"] = i
        out.append(x)
    return pd.concat(out, ignore_index=True)


clk = shards("clock_tap")
ev = shards("event_tap")
G = shards("games")
flags = json.loads(sorted(TAP.glob("flags_s*.json"))[0].read_text(encoding="utf-8"))["flags"]
CLS = [c for c in ev.columns if c not in
       ("seed", "gidx", "is_first", "period", "sec", "sd", "bonus", "chance",
        "shard", "off_is_home")]

# --- the offence's SIDE on the clock rows ----------------------------------
# `site_home` in the team-static block is 0 for BOTH teams on a neutral court,
# so it cannot name the side on the 14.5% of games played at one.  The engine
# draws the duration and then runs the first-chance event predict over the SAME
# active rows in the SAME order (`loop.py` steps (a) and (b)), so the k-th clock
# row IS the k-th first-chance event row.  That is asserted here on three
# independent columns rather than assumed, and the side is then taken from the
# event tap, which is handed the true `off` index.
_fc = ev[ev["is_first"] > 0].reset_index(drop=True)
assert len(_fc) == len(clk), "clock and first-chance event taps differ in length"
for _c in ("gidx", "sec", "sd", "seed"):
    assert (clk[_c].to_numpy() == _fc[_c].to_numpy()).all(), \
        f"clock/event row alignment broken on {_c}"
clk["off_is_home"] = _fc["off_is_home"].to_numpy()

P("=" * 78)
P("LATE-GAME RE-READ AGAINST THE SERVED STACK (v5b), 2026-09-18")
P("=" * 78)
P("tap: %s   possessions %d  chances %d  sims %d" % (TAP, len(clk), len(ev), len(G)))
P("served flags: " + json.dumps({k: v for k, v in flags.items() if k.startswith("ENGINE_")}))
P("")

# ---------------------------------------------------------------- 0. levels
P("=== 0. MARGIN LEVELS, from the FULL served-stack run (not the tap) ===")
for tag, path in (("v5b served (75 seeds)", V5B_RUN), ("v3c run A (the 2026-09-11 evidence)", V3C_RUN)):
    if not path.exists():
        P("  %-38s MISSING %s" % (tag, path))
        continue
    g = pd.read_parquet(path, columns=["home_pts", "away_pts", "n_periods", "seed"])
    reg = np.where(g["n_periods"].to_numpy() > 2, 0, (g["home_pts"] - g["away_pts"]).to_numpy())
    a = np.abs(reg)
    p0, p1 = float((a == 0).mean()), float((a == 1).mean())
    P("  %-38s n %7d  P(0) %.5f  P(1) %.5f  P0/P1 %.3f  SD %.4f  exkurt %+.4f"
      % (tag, len(g), p0, p1, p0 / max(p1, 1e-12), reg.std(ddof=0),
         pd.Series(reg).kurtosis()))
P("  %-38s n    5710  P(0) 0.05660  P(1) 0.03660  P0/P1 1.546  SD 13.8   exkurt +0.7400"
  % "ACTUAL 2024-25 (verified finals)")
P("  tap's own tie rate (for provenance only, underpowered as a level): %.5f on %d sims"
  % (float((np.where(G["n_periods"] > 2, 0, G["home_pts"] - G["away_pts"]) == 0).mean()), len(G)))
P("")

# ------------------------------------------------- 1. margin at the 2:00 mark
w = clk[(clk["period"] == 2) & (clk["sec"] <= 120)].copy()
w = w.sort_values(["seed", "gidx", "sec"], ascending=[True, True, False])
f = w.groupby(["seed", "gidx"]).first().reset_index()
P("SIM margin-at-2:00 anchored at mean sec %.2f (median %.1f), sims %d"
  % (f["sec"].mean(), f["sec"].median(), len(f)))
f["am"] = f["sd"].abs()
# The anchor in HOME terms, so it can be re-signed for whichever team is on
# offence.  `diag_late_game_actual_v1.py` line 62 does exactly this on the
# actual side; `diag_late_game_compare_v1.py` did NOT do it on the sim side and
# applied one offence's signed anchor to both teams' possessions, which mixed
# the two roles inside each cell.  That is the defect this file corrects.
f["m120_home"] = f["sd"] * np.where(f["off_is_home"].to_numpy() > 0, 1.0, -1.0)
key = G.set_index(["seed", "gidx"])
reg_g = np.where(key["n_periods"] > 2, 0, key["home_pts"] - key["away_pts"])
f = f.join(pd.Series(reg_g, index=key.index, name="reg"), on=["seed", "gidx"])
f = f[f["reg"].notna()]

A = json.load(open(ROOT / "results/late_game/actual_kernel.json"))
kl = {int(k): v for k, v in A["kl"].items()}
k0 = {int(k): v for k, v in A["k0"].items()}
dist_act = {int(float(k)): v for k, v in A["dist_act"].items()}
act_tie = A["act_tie"]

ksim, dsim, nsim = {}, {}, {}
for k in range(0, 25):
    b = f[f["am"] == k]
    dsim[k] = len(b) / len(f)
    nsim[k] = len(b)
    if len(b) >= 100:
        ksim[k] = (b["reg"] == 0).mean()

P("")
P("=== A. The margin-at-2:00 DISTRIBUTION, sim (v5b) vs actual ===")
P("  %5s %8s %10s %10s %8s" % ("absm", "n", "SIM P", "ACT P", "ratio"))
for k in range(0, 16):
    pa = dist_act.get(k, 0.0)
    P("  %5d %8d %10.5f %10.5f %8.3f" % (k, nsim.get(k, 0), dsim.get(k, 0.0), pa,
                                         dsim.get(k, 0.0) / max(pa, 1e-9)))
ss = sum(v for k, v in dsim.items() if k <= 6)
sa = sum(v for k, v in dist_act.items() if k <= 6)
P("  P(abs m120 <= 6): SIM %.5f  ACT %.5f  ratio %.4f" % (ss, sa, ss / sa))
P("  SD of the margin at 2:00: SIM %.4f" % f["sd"].std(ddof=0))
P("  [2026-09-11 v3c read: ratio 0.900, SD 15.71]")

P("")
P("=== B. The END-GAME KERNEL P(land on 0 | abs margin at 2:00) ===")
P("  cells below 100 simulations are omitted; those shown carry n as stated")
P("  %5s %7s %10s %10s %10s %9s %9s" % ("absm", "n", "SIM", "ACT", "MID K0", "SIM/ACT", "SIM/K0"))
for k in sorted(set(ksim) & set(kl)):
    if k > 12:
        break
    lab = " UNDERPOWERED" if nsim[k] < 200 else ""
    P("  %5d %7d %10.5f %10.5f %10.5f %9.3f %9.3f%s"
      % (k, nsim[k], ksim[k], kl[k], k0.get(k, float("nan")),
         ksim[k] / max(kl[k], 1e-9), ksim[k] / max(k0.get(k, 1e-9), 1e-9), lab))


def mix(dist, kk):
    s, cov = 0.0, 0.0
    for k, p in dist.items():
        if k in kk:
            s += p * kk[k]
            cov += p
    return s, cov


common = sorted(set(ksim) & set(kl))
ks = {k: ksim[k] for k in common}
ka = {k: kl[k] for k in common}
km = {k: k0[k] for k in common if k in k0}
ss_, _ = mix(dsim, ks)
aa_, _ = mix(dist_act, ka)
ax, _ = mix(dist_act, ks)
sx, _ = mix(dsim, ka)
P("")
P("=== C. (a)/(b) ATTRIBUTION of the tie shortfall, on the v5b kernel ===")
P("  All four cells use the SAME kernel support (abs m120 in %s)." % common)
P("  sim dist x sim kernel  = %.5f" % ss_)
P("  act dist x sim kernel  = %.5f   [(a) alone]" % ax)
P("  sim dist x act kernel  = %.5f   [(b) alone]" % sx)
P("  act dist x act kernel  = %.5f" % aa_)
gap = aa_ - ss_
P("  total restricted gap   = %.5f" % gap)
if abs(gap) > 1e-9:
    P("  (a) upstream margin-at-2:00 distribution : %+.5f  = %+.1f%% of the gap"
      % (ax - ss_, 100 * (ax - ss_) / gap))
    P("  (b) end-game kernel                      : %+.5f  = %+.1f%% of the gap"
      % (sx - ss_, 100 * (sx - ss_) / gap))
    P("  interaction                              : %+.5f  = %+.1f%%"
      % (gap - (ax - ss_) - (sx - ss_), 100 * (gap - (ax - ss_) - (sx - ss_)) / gap))
P("  [2026-09-11 v3c read: (a) +26.9%, (b) +70.3%, interaction +2.8%]")
P("  regime-free kernel K0 on the ACTUAL 2:00 distribution: %.5f  (ACT tie %.5f)"
  % (mix(dist_act, km)[0], act_tie))

# ----------------------------------------------------- D. window behaviour
P("")
P("=== D. SIM end-game behaviour in the final 2:00 (abs m120 <= 6), SERVED v5b ===")
m = f.set_index(["seed", "gidx"])["am"]
clk = clk.join(m.rename("am"), on=["seed", "gidx"])
ev = ev.join(m.rename("am"), on=["seed", "gidx"])
mh = f.set_index(["seed", "gidx"])["m120_home"].rename("m120h")
ev = ev.join(mh, on=["seed", "gidx"])
clk = clk.join(mh, on=["seed", "gidx"])
# PER POSSESSION: the anchored margin from THIS offence's point of view.
for _d in (ev, clk):
    _d["m120sd"] = _d["m120h"] * np.where(_d["off_is_home"].to_numpy() > 0, 1.0, -1.0)
cw = clk[(clk["am"] <= 6) & (clk["period"] == 2) & (clk["sec"] <= 120)]
cr = clk[(clk["am"] <= 6) & (clk["period"] <= 2)
         & ~((clk["period"] == 2) & (clk["sec"] <= 120))]
P("  duration: window mean %.3f med %.1f Pgt25 %.4f Ple8 %.4f  n %d"
  % (cw["dur"].mean(), cw["dur"].median(), (cw["dur"] > 25).mean(),
     (cw["dur"] <= 8).mean(), len(cw)))
P("  duration: ref    mean %.3f med %.1f Pgt25 %.4f Ple8 %.4f  n %d"
  % (cr["dur"].mean(), cr["dur"].median(), (cr["dur"] > 25).mean(),
     (cr["dur"] <= 8).mean(), len(cr)))
P("  ACTUAL: window 10.56 (Pgt25 .0987 Ple8 .5217); ref 15.78 (.1619 / .2634)")
P("  [2026-09-11 v3c read: window 13.35 Pgt25 .1557 Ple8 .4013; ref 17.74 .2022 .1845]")
n_w_games = cw.groupby(["seed", "gidx"]).size()
P("  possessions per game in the final 2:00: SIM %.3f   ACTUAL 9.24-11.34 by bucket"
  % n_w_games.mean())
P("  [2026-09-11 v3c read: 8.855]")

P("")
P("  -- duration by ROLE at the 2:00 mark (the split v1's compare never printed) --")
for lab, sel in (("trailing offence", cw["m120sd"] < 0), ("tied", cw["m120sd"] == 0),
                 ("leading offence", cw["m120sd"] > 0)):
    dd = cw[sel]
    tag = " UNDERPOWERED" if len(dd) < 200 else ""
    P("   SIM %-18s n%7d dur %6.3f Ple8 %.4f%s"
      % (lab, len(dd), dd["dur"].mean(), (dd["dur"] <= 8).mean(), tag))
P("   ACT trailing 9.60 / leading 11.45 / tied 11.70")

P("")
P("  -- duration by LIVE offence margin sign and the served clock's eg_regime --")
P("     (the served clock's cell grid IS role-aware: EMPIRICAL_DIMS['P3_dummy'] =")
P("      prev_end x r2_bucket x period_type x eg_regime x tempo_tercile, and")
P("      eg_regime = {0 outside, 1 trailing by >=4, 2 leading by >=4, 3 close} )")
for lab, sel in (("eg_trailing_big sd<=-4", cw["sd"] <= -4),
                 ("eg_close |sd|<4", cw["sd"].abs() < 4),
                 ("eg_leading_big sd>=4", cw["sd"] >= 4)):
    dd = cw[sel]
    tag = " UNDERPOWERED" if len(dd) < 200 else ""
    P("   SIM %-24s n%7d dur %6.3f%s" % (lab, len(dd), dd["dur"].mean(), tag))

ew = ev[(ev["am"] <= 6) & (ev["period"] == 2) & (ev["sec"] <= 120) & (ev["is_first"] > 0)]
er = ev[(ev["am"] <= 6) & (ev["period"] <= 2) & (ev["is_first"] > 0)
        & ~((ev["period"] == 2) & (ev["sec"] <= 120))]


def emix(d, label):
    if len(d) < 200:
        P("  %-28s n=%d UNDERPOWERED" % (label, len(d)))
        return None
    mu = d[CLS].mean()
    fga = sum(mu[c] for c in CLS if c.startswith("FGA"))
    P("  %-28s n%7d FGA/p %.4f 3PAsh %.4f TOV %.4f bonusFT %.4f shootFT %.4f"
      % (label, len(d), fga, mu.get("FGA_3", np.nan) / max(fga, 1e-9),
         mu.get("TOV", np.nan), mu.get("FT_trip_bonus", np.nan),
         mu.get("FT_trip_shooting", np.nan)))
    return (mu.get("FGA_3", np.nan) / max(fga, 1e-9), mu.get("FT_trip_bonus", np.nan))


P("")
P("  -- event-model served class probabilities (first chance) --")
emix(er, "SIM ref (rest of reg)")
emix(ew, "SIM final 2:00 (all)")
t = emix(ew[ew["m120sd"] < 0], "SIM final 2:00 TRAILING")
l = emix(ew[ew["m120sd"] > 0], "SIM final 2:00 LEADING")
P("  ACTUAL ref        : FGA/p 0.7398 3PAsh 0.3830 TOV 0.1492 bonusFT 0.0364")
P("  ACTUAL final 2:00 : FGA/p 0.5626 3PAsh 0.4219 TOV 0.1068 bonusFT 0.2775")
P("  ACTUAL trailing   : FGA/p 0.6914 3PAsh 0.4788 TOV 0.0914 bonusFT 0.1564")
P("  ACTUAL leading    : FGA/p 0.4126 3PAsh 0.3213 TOV 0.1239 bonusFT 0.4216")
if t and l:
    P("")
    P("  *** THE HEADLINE SPLITS, v5b, ANCHOR SIGNED PER POSSESSION ***")
    P("  NOTE ON SIGN.  `late_game_regime_2026-09-11.md` section 1.1 states the")
    P("  three-point gap as '-0.158', but its own table has trailing 0.4788 and")
    P("  leading 0.3213, i.e. trailing MINUS leading = +0.1575.  The minus sign")
    P("  in that sentence is a transcription error; the substantive claim (the")
    P("  trailing team shoots MORE threes) is what the table says and is what is")
    P("  compared here.")
    P("  trailing-minus-leading 3PA share : SIM %+.4f   ACT +0.1575   reproduced %.1f%%"
      % (t[0] - l[0], 100 * (t[0] - l[0]) / 0.1575))
    P("  leading-minus-trailing bonus-FT  : SIM %+.4f   ACT +0.2652   reproduced %.1f%%"
      % (l[1] - t[1], 100 * (l[1] - t[1]) / 0.2652))
    P("  [2026-09-11 v3c read, UNSIGNED anchor: +0.007 and +0.024 -- both are")
    P("   artefacts of applying one offence's anchored margin to both teams]")

P("")
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
P("  ACTUAL 3PAsh .3562 .3562 .3845 .4803 .6324 ; bonusFT .1623 .1957 .2768 .3874 .3749")
P("  ACTUAL dur   16.20 15.88 11.39  6.56  2.89")

P("")
P("  -- by seconds bucket x ROLE, duration (gate R4's own cut) --")
for lo, hi in [(60, 120), (30, 60), (10, 30), (0, 10)]:
    b = cw[(cw["sec"] > lo) & (cw["sec"] <= hi)]
    tt, ll = b[b["m120sd"] < 0], b[b["m120sd"] > 0]
    P("   (%3d,%3d] trail n%6d dur %6.2f | lead n%6d dur %6.2f"
      % (lo, hi, len(tt), tt["dur"].mean(), len(ll), ll["dur"].mean()))
P("   ACTUAL     trail 13.52 10.38  7.77  3.53 | lead 18.38 12.14  4.88  2.15")

P("")
P("=== E. The role split on the LIVE margin, not the 2:00 anchor ===")
P("  Roles SWAP inside the window, so an anchored role dilutes any response the")
P("  served model does have.  This cut uses `score_diff` at the possession's own")
P("  start -- the column `possession_outcome`'s served arm actually consumes --")
P("  and is computed the same way on both sides, so it separates 'the model has")
P("  no role response' from 'the anchored definition hides one'.")


def live_split(d, lab):
    t, l = d[d["sd"] < 0], d[d["sd"] > 0]
    if min(len(t), len(l)) < 200:
        P("  %-22s UNDERPOWERED (n %d / %d)" % (lab, len(t), len(l)))
        return
    def sh(x):
        mu = x[CLS].mean()
        fga = sum(mu[c] for c in CLS if c.startswith("FGA"))
        return mu["FGA_3"] / max(fga, 1e-9), mu["FT_trip_bonus"]
    t3, tb = sh(t)
    l3, lb = sh(l)
    P("  %-22s n %6d/%6d | 3PAsh trail %.4f lead %.4f split %+.4f | bonusFT trail "
      "%.4f lead %.4f split %+.4f" % (lab, len(t), len(l), t3, l3, t3 - l3, tb, lb, lb - tb))


live_split(ew, "SIM window, live sd")
for lo, hi in [(60, 120), (30, 60), (10, 30), (0, 10)]:
    live_split(ew[(ew["sec"] > lo) & (ew["sec"] <= hi)], "SIM sec (%d,%d]" % (lo, hi))

_ach = pd.read_parquet(ROOT / "data/processed/possessions_v2/chances_2025.parquet")
_ach = _ach[(_ach["period"] == 2) & (_ach["start_clock"] <= 120)
            & (_ach["start_score_diff"].abs() <= 6)
            & (_ach["chance_number"] == 1)].copy()
_ach["is3"] = (_ach["terminal_event"] == "FGA_3").astype(float)
_ach["isfga"] = _ach["terminal_event"].isin(["FGA_3", "FGA_rim", "FGA_jump2"]).astype(float)
_ach["isb"] = (_ach["terminal_event"] == "FT_trip_bonus").astype(float)
P("")
for lo, hi in [(0, 120), (60, 120), (30, 60), (10, 30), (0, 10)]:
    b = _ach[(_ach["start_clock"] > lo) & (_ach["start_clock"] <= hi)] if lo else \
        _ach[_ach["start_clock"] <= hi]
    t, l = b[b["start_score_diff"] < 0], b[b["start_score_diff"] > 0]
    if min(len(t), len(l)) < 200:
        P("  ACT sec (%d,%d]        UNDERPOWERED" % (lo, hi))
        continue
    t3 = t["is3"].sum() / max(t["isfga"].sum(), 1)
    l3 = l["is3"].sum() / max(l["isfga"].sum(), 1)
    P("  ACT sec (%3d,%3d]      n %6d/%6d | 3PAsh trail %.4f lead %.4f split %+.4f | "
      "bonusFT trail %.4f lead %.4f split %+.4f"
      % (lo, hi, len(t), len(l), t3, l3, t3 - l3,
         t["isb"].mean(), l["isb"].mean(), l["isb"].mean() - t["isb"].mean()))

P("")
P("  -- possessions per game in the window, by abs margin at 2:00 (gate R5) --")
pg = cw.groupby(["seed", "gidx"]).size().rename("n")
pg = pd.DataFrame(pg).join(f.set_index(["seed", "gidx"])["am"])
for k in range(0, 7):
    b = pg[pg["am"] == k]
    tag = " UNDERPOWERED" if len(b) < 200 else ""
    P("   abs_m120=%d  sims %5d  poss/game %6.3f%s" % (k, len(b), b["n"].mean(), tag))
P("   ACTUAL      9.24 9.34 9.54 10.12 10.60 10.80 11.34")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("")
print("wrote " + OUT)
