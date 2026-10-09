"""diag_margin_dispersion_v1.py -- is the sim's per-game margin distribution too wide? DIAGNOSTIC ONLY (2026-10-09).

Fold 2 / season 2025 (2024-25), served stack v3, 5,710 games x 50 seeds. Reuses the cached extracts of
diag_c4_extract_v1.py (results/diag_c4/f2all50_*.parquet); nothing is re-simulated. 2025-26 is never read.
Writes results/diag_margin_dispersion/{tables.md,summary.json}; the doc docs/tests/margin_dispersion_2026-10-09.md
wraps those tables with the verdict.

Conventions
- margin = home minus away, OT included (sim ties resolve through the OT model; zero ties in the sim).
- Market centre m = -close_spread_home (ESPN BET close, lines_close_v2_verified).
- Sim mean mu_i is a 50-seed mean, so it carries MC noise with variance s2_i/50. Every "realised residual variance"
  that uses mu is corrected by subtracting mean(s2_i)/50. Sim own SD = sqrt(mean_i s2_i), s2_i the unbiased
  across-seed variance.
- All CIs are game-cluster bootstraps (1000 reps; 400 for slopes). Seed-noise band: seeds bootstrapped, and
  odd/even seed halves. Underpowered = n < 60 games.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
D = str(ROOT / "results" / "diag_c4" / "f2all50_")
OUT = ROOT / "results" / "diag_margin_dispersion"
OUT.mkdir(parents=True, exist_ok=True)
NS = 50
RNG = np.random.default_rng(20261009)
NB = 1000
MINN = 60
L: list[str] = []
SUM: dict = {}


def w(s=""):
    L.append(s)
    print(s)


# ---------------------------------------------------------------- load
pg = pd.read_parquet(D + "pregame.parquet").sort_values("game_id").reset_index(drop=True)
gids = pg.game_id.to_numpy()
sgs = pd.read_parquet(D + "sim_gs.parquet")
rgs = pd.read_parquet(D + "real_gs.parquet")
stg = pd.read_parquet(D + "sim_tg.parquet")
rtg = pd.read_parquet(D + "real_tg.parquet")
ln = pd.read_parquet(ROOT / "data/processed/lines/lines_close_v2_verified.parquet")
ln = ln[(ln.season == 2025) & (ln.provider == "ESPN BET")][["game_id", "close_spread_home", "p_home_close_prop",
                                                            "p_home_close_power"]]
pg = pg.merge(ln, on="game_id", how="left", suffixes=("", "_l"))
assert (pg.close_spread_home.fillna(0) == pg.close_spread_home_l.fillna(0)).all()
G = len(pg)
y = (pg.fin_h - pg.fin_a).to_numpy().astype(float)  # verified finals
m = -pg.close_spread_home.to_numpy()
has_line = ~np.isnan(m)
has_ml = pg.p_home_close_power.notna().to_numpy()
SM = np.vstack([v for _, v in sgs.sort_values(["game_id", "seed"]).groupby("game_id").final_margin])
assert SM.shape == (G, NS)
SM = SM.astype(float)
mu = SM.mean(1)
s2 = SM.var(1, ddof=1)
r = y - mu
month = pg.game_date.dt.month.to_numpy()
assert (SM == 0).sum() == 0
SUM["n_games"] = int(G)
SUM["n_line"] = int(has_line.sum())
SUM["n_ml"] = int(has_ml.sum())


# ---------------------------------------------------------------- helpers
def boot_idx(n, reps=NB):
    return RNG.integers(0, n, size=(reps, n))


def ratio_stats(r2_, s2_, k=NS):
    """sim SD, corrected realised residual SD, ratio from per-game vectors (second moments)."""
    sim_v = s2_.mean()
    real_v = r2_.mean() - sim_v / k
    return np.sqrt(sim_v), np.sqrt(max(real_v, 1e-12)), np.sqrt(sim_v / max(real_v, 1e-12))


def cell_ratio(mask, ycol=None, mucol=None, s2col=None, reps=NB):
    yy = y if ycol is None else ycol
    mm = mu if mucol is None else mucol
    ss = s2 if s2col is None else s2col
    msk = np.asarray(mask) & ~np.isnan(yy) & ~np.isnan(mm)
    n = int(msk.sum())
    if n < 5:
        return dict(n=n)
    rr = (yy[msk] - mm[msk])
    r2_ = rr ** 2
    sd_s, sd_r, ratio = ratio_stats(r2_, ss[msk])
    bi = boot_idx(n, reps)
    rb = np.array([ratio_stats(r2_[i], ss[msk][i])[2] for i in bi])
    sb = np.array([ratio_stats(r2_[i], ss[msk][i])[0] for i in bi])
    rsb = np.array([ratio_stats(r2_[i], ss[msk][i])[1] for i in bi])
    return dict(n=n, sim_sd=sd_s, real_sd=sd_r, ratio=ratio, ratio_lo=np.percentile(rb, 2.5),
                ratio_hi=np.percentile(rb, 97.5), real_lo=np.percentile(rsb, 2.5), real_hi=np.percentile(rsb, 97.5),
                bias=float(rr.mean()))


def fmt_cell(label, c):
    if c.get("n", 0) < 5 or "ratio" not in c:
        return f"| {label} | {c.get('n', 0)} | - | - | - | - | - |"
    flag = " UNDERPOWERED" if c["n"] < MINN else ""
    return (f"| {label}{flag} | {c['n']} | {c['sim_sd']:.2f} | {c['real_sd']:.2f} [{c['real_lo']:.2f}, {c['real_hi']:.2f}] "
            f"| {c['ratio']:.3f} [{c['ratio_lo']:.3f}, {c['ratio_hi']:.3f}] | {c['bias']:+.2f} | "
            f"{'wide' if c['ratio_lo'] > 1 else ('narrow' if c['ratio_hi'] < 1 else 'n.s.')} |")


HDR1 = ("| cell | n games | sim own SD | realised residual SD (real - sim mean, MC-corrected) [95%] "
        "| ratio sim/real [95% game bootstrap] | mean(real - sim mean) | verdict (CI vs 1) |\n|---|---:|---:|---|---|---:|---|")


def cuts_table(title, ycol=None, mucol=None, mask0=None):
    out = {}
    base = np.ones(G, bool) if mask0 is None else mask0
    w(f"### {title}\n")
    w(HDR1)
    c = cell_ratio(base, ycol, mucol)
    w(fmt_cell("ALL", c))
    out["ALL"] = c
    # |spread| quintile
    sp = np.abs(m)
    q = pd.qcut(pd.Series(sp).where(has_line), 5, labels=False).to_numpy()
    for k in range(5):
        mk = base & (q == k)
        cc = cell_ratio(mk, ycol, mucol)
        lab = f"|spread| Q{k + 1} (mean {np.nanmean(sp[mk]):.1f})" if mk.any() else f"|spread| Q{k + 1}"
        w(fmt_cell(lab, cc))
        out[f"spread_Q{k + 1}"] = cc
    # month
    for mo, nm in [(11, "Nov"), (12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar"), (4, "Apr")]:
        mk = base & (month == mo)
        cc = cell_ratio(mk, ycol, mucol)
        w(fmt_cell(f"month {nm}", cc))
        out[f"month_{nm}"] = cc
    # conference flag
    cf = pg.conference_competition.fillna(False).to_numpy().astype(bool)
    for nm, mk0 in [("conference game", cf), ("non-conference game", ~cf)]:
        cc = cell_ratio(base & mk0, ycol, mucol)
        w(fmt_cell(nm, cc))
        out[nm] = cc
    # site
    neu = pg.neutral_site.to_numpy().astype(bool)
    for nm, mk0 in [("neutral site", neu), ("home/away site", ~neu)]:
        cc = cell_ratio(base & mk0, ycol, mucol)
        w(fmt_cell(nm, cc))
        out[nm] = cc
    # home rating quintile
    hn = pg.home_net.to_numpy()
    qh = pd.qcut(pd.Series(hn), 5, labels=False).to_numpy()
    for k in range(5):
        mk = base & (qh == k)
        cc = cell_ratio(mk, ycol, mucol)
        lab = f"home rating Q{k + 1} (mean net {np.nanmean(hn[mk]):+.1f})" if mk.any() else f"home rating Q{k + 1}"
        w(fmt_cell(lab, cc))
        out[f"homenet_Q{k + 1}"] = cc
    cc = cell_ratio(base & np.isnan(hn), ycol, mucol)
    w(fmt_cell("home rating missing", cc))
    w()
    return out


# ================================================================ PART 1
w("## 1. Sim own margin SD vs realised residual SD (real - sim mean)\n")
w(f"Games: {G} simulated; {int(has_line.sum())} with an ESPN BET close; {int(has_ml.sum())} with a de-vigged moneyline. "
  f"All {G} have verified finals. Each sim game is {NS} seeds; MC correction subtracts mean(s2)/{NS}.\n")
p1 = cuts_table("1a. All games with verified final")
SUM["part1_all"] = p1
p1l = cuts_table("1b. Same cuts, restricted to games with a market close (the sample parts 2-3 use)", mask0=has_line)
SUM["part1_line"] = p1l

# seed-noise band
w("### 1c. Seed-noise band on the overall ratio\n")
def ratio_from_seeds(cols):
    sub = SM[:, cols]
    mu_ = sub.mean(1)
    s2_ = sub.var(1, ddof=1)
    return ratio_stats((y - mu_) ** 2, s2_, k=len(cols))
rows = []
ev, od = np.arange(0, NS, 2), np.arange(1, NS, 2)
for nm, cols in [("all 50 seeds", np.arange(NS)), ("even seeds (25)", ev), ("odd seeds (25)", od)]:
    sd_s, sd_r, ra = ratio_from_seeds(cols)
    rows.append((nm, sd_s, sd_r, ra))
sb = []
for _ in range(300):
    cols = RNG.integers(0, NS, NS)
    sb.append(ratio_from_seeds(cols)[2])
w("| seeds used | sim own SD | realised residual SD | ratio |\n|---|---:|---:|---:|")
for nm, a, b, c in rows:
    w(f"| {nm} | {a:.3f} | {b:.3f} | {c:.4f} |")
w(f"\nSeed bootstrap (300 reps of resampling the 50 seeds, games fixed): ratio 95% band "
  f"[{np.percentile(sb, 2.5):.4f}, {np.percentile(sb, 97.5):.4f}]. This is the pure sim-noise band; it is small next to the game-sampling CI above.\n")
SUM["seed_band"] = [float(np.percentile(sb, 2.5)), float(np.percentile(sb, 97.5))]
# z-score view
z = r / np.sqrt(s2)
w(f"Equivalent view, standardised residual z = (real - sim mean)/sim own game SD: SD(z) = {z.std():.3f} "
  f"(calibrated width = 1.0; below 1 means the sim is too wide); mean z = {z.mean():+.3f}.\n")
SUM["sd_z"] = float(z.std())

# ================================================================ PART 2
w("## 2. Versus the market: wrong centre or wrong width?\n")
w(f"Sample: {int(has_line.sum())} games with a close. e_sim = real - sim mean; e_mkt = real - m (m = -close spread); "
  "d = m - sim mean, so e_sim = e_mkt + d exactly. Var(e_sim) and Var(d) are MC-corrected (minus mean(s2)/50); "
  "Cov(e_mkt, d) is not affected by MC noise (independent of the realised margin). Variances are about each cell's mean; the means are shown separately.\n")


def decomp(mask, reps=NB):
    msk = mask & has_line
    n = int(msk.sum())
    em = (y - m)[msk]
    dd = (m - mu)[msk]
    ss = s2[msk]

    def f(i):
        e1, d1, s1 = em[i], dd[i], ss[i]
        corr = s1.mean() / NS
        v_em = e1.var()
        v_d = d1.var() - corr
        cov = np.cov(e1, d1, bias=True)[0, 1]
        v_es = v_em + v_d + 2 * cov
        sim_v = s1.mean()
        return v_em, v_d, cov, v_es, sim_v
    base = f(np.arange(n))
    bs = np.array([f(i) for i in boot_idx(n, reps)])
    return n, base, bs, em.mean(), dd.mean()


HD = ("| cell | n | SD(e_mkt) market resid [95%] | SD(e_sim) sim resid [95%] | sim own SD | Var(e_mkt) | Var(d) centre gap | 2Cov | "
      "Var(e_sim) | sim own SD / market resid SD [95%] | sim own SD / sim resid SD [95%] | mean e_mkt | mean e_sim |\n"
      "|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|")


def dline(label, mask):
    n, b, bs, me, md = decomp(mask)
    if n < 5:
        return f"| {label} | {n} |" + " - |" * 11, None
    flag = " UNDERPOWERED" if n < MINN else ""
    v_em, v_d, cov, v_es, sim_v = b
    sd_em = np.sqrt(bs[:, 0])
    sd_es = np.sqrt(np.maximum(bs[:, 3], 1e-9))
    r_m = np.sqrt(bs[:, 4] / bs[:, 0])
    r_s = np.sqrt(bs[:, 4] / np.maximum(bs[:, 3], 1e-9))
    ci = lambda a: f"[{np.percentile(a, 2.5):.2f}, {np.percentile(a, 97.5):.2f}]"
    cj = lambda a: f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"
    me_sim = me + md  # mean(real-m) + mean(m - mu) = mean(real - mu)
    s = (f"| {label}{flag} | {n} | {np.sqrt(v_em):.2f} {ci(sd_em)} | {np.sqrt(v_es):.2f} {ci(sd_es)} | {np.sqrt(sim_v):.2f} | "
         f"{v_em:.1f} | {v_d:.1f} | {2 * cov:+.1f} | {v_es:.1f} | {np.sqrt(sim_v / v_em):.3f} {cj(r_m)} | "
         f"{np.sqrt(sim_v / v_es):.3f} {cj(r_s)} | {me:+.2f} | {me_sim:+.2f} |")
    return s, dict(n=n, sd_mkt=float(np.sqrt(v_em)), sd_sim_resid=float(np.sqrt(v_es)), sim_sd=float(np.sqrt(sim_v)),
                   var_mkt=float(v_em), var_d=float(v_d), cov2=float(2 * cov), var_sim_resid=float(v_es),
                   r_mkt=float(np.sqrt(sim_v / v_em)), r_mkt_ci=[float(np.percentile(r_m, 2.5)), float(np.percentile(r_m, 97.5))],
                   r_sim=float(np.sqrt(sim_v / v_es)), r_sim_ci=[float(np.percentile(r_s, 2.5)), float(np.percentile(r_s, 97.5))])


w(HD)
p2 = {}
def add(label, mask):
    s, d = dline(label, mask)
    w(s)
    p2[label] = d
add("ALL (with close)", np.ones(G, bool))
sp = np.abs(m)
q = pd.qcut(pd.Series(sp).where(has_line), 5, labels=False).to_numpy()
for k in range(5):
    add(f"|spread| Q{k + 1}", q == k)
for mo, nm in [(11, "Nov"), (12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar"), (4, "Apr")]:
    add(f"month {nm}", month == mo)
cf = pg.conference_competition.fillna(False).to_numpy().astype(bool)
add("conference game", cf)
add("non-conference game", ~cf)
neu = pg.neutral_site.to_numpy().astype(bool)
add("neutral site", neu)
add("home/away site", ~neu)
hn = pg.home_net.to_numpy()
qh = pd.qcut(pd.Series(hn), 5, labels=False).to_numpy()
for k in range(5):
    add(f"home rating Q{k + 1}", qh == k)
SUM["part2"] = p2
w("\nReading: if sim own SD is larger than SD(e_mkt) the sim claims more per-game uncertainty than the market's own realised error. "
  "Var(d) is the extra variance the sim's centre adds over the market's (a wrong-centre cost); a ratio of sim own SD to sim resid SD "
  "above 1 means the sim is too wide *given its own centre*.\n")

# centre responsiveness: real ~ a + b*mu, real ~ a + b*m, and incremental
msk = has_line
def ols(xv, yv):
    X = np.column_stack([np.ones(len(xv)), xv])
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    return beta
def boot_slope(xv, yv, reps=400):
    n = len(xv)
    bb = np.array([ols(xv[i], yv[i])[1] for i in boot_idx(n, reps)])
    return np.percentile(bb, 2.5), np.percentile(bb, 97.5)
w("### 2b. Centre responsiveness: realised margin regressed on each centre (slope 1 = right spread of means)\n")
w("| regressor | n | slope [95%] | intercept | R^2 |\n|---|---:|---|---:|---:|")
for nm, xv in [("sim mean margin", mu), ("market m = -close", m)]:
    b0, b1 = ols(xv[msk], y[msk])
    lo, hi = boot_slope(xv[msk], y[msk])
    r2v = np.corrcoef(xv[msk], y[msk])[0, 1] ** 2
    w(f"| {nm} | {int(msk.sum())} | {b1:.3f} [{lo:.3f}, {hi:.3f}] | {b0:+.2f} | {r2v:.3f} |")
    SUM[f"slope_{nm}"] = [float(b1), float(lo), float(hi)]
X = np.column_stack([np.ones(msk.sum()), m[msk], mu[msk]])
bt = np.linalg.lstsq(X, y[msk], rcond=None)[0]
bb = np.array([np.linalg.lstsq(X[i], y[msk][i], rcond=None)[0] for i in boot_idx(msk.sum(), 400)])
w(f"\nJoint fit real = a + b1*m + b2*mu: b1 (market) = {bt[1]:.3f} [{np.percentile(bb[:, 1], 2.5):.3f}, {np.percentile(bb[:, 1], 97.5):.3f}], "
  f"b2 (sim) = {bt[2]:.3f} [{np.percentile(bb[:, 2], 2.5):.3f}, {np.percentile(bb[:, 2], 97.5):.3f}]. "
  f"SD of the centres across games: sim mean {mu[msk].std():.2f}, market {m[msk].std():.2f} (mean-noise-corrected sim: "
  f"{np.sqrt(max(mu[msk].var() - s2[msk].mean() / NS, 0)):.2f}).\n")
SUM["joint"] = [float(bt[1]), float(bt[2])]
SUM["centre_sd"] = dict(sim=float(mu[msk].std()), mkt=float(m[msk].std()))

# ================================================================ PART 3
w("## 3. Win-probability calibration and PIT\n")
ph = (SM > 0).mean(1)
pA = (SM[:, ev] > 0).mean(1)
pB = (SM[:, od] > 0).mean(1)
yw = (y > 0).astype(float)
pm = pg.p_home_close_power.to_numpy()
pmp = pg.p_home_close_prop.to_numpy()
sel = has_ml
n3 = int(sel.sum())
w(f"Sample: {n3} games with a de-vigged moneyline (power de-vig primary; proportional shown for slopes). Sim P(home win) = share of 50 seeds "
  f"with margin > 0 (MC SD of that estimate is sqrt(p(1-p)/50) <= 0.071; this attenuates naive slopes, so an odd/even-seed IV slope is reported). "
  f"Realised home win rate {yw[sel].mean():.4f}; sim mean {ph[sel].mean():.4f}; market mean {pm[sel].mean():.4f}.\n")

# bucket tables
def bucket_table(keyp, title):
    w(f"### {title}\n")
    w("| bucket | n | mean sim P | mean market P | realised rate [Wilson 95%] | realised - sim | realised - market |\n|---|---:|---:|---:|---|---:|---:|")
    idx = np.where(sel)[0]
    qb = pd.qcut(pd.Series(keyp[idx]).rank(method="first"), 20, labels=False).to_numpy()
    out = []
    for k in range(20):
        ii = idx[qb == k]
        n = len(ii)
        wins = yw[ii].sum()
        lo, hi = stats.binomtest(int(wins), n).proportion_ci(0.95, method="wilson")
        w(f"| {k + 1} | {n} | {ph[ii].mean():.3f} | {pm[ii].mean():.3f} | {wins / n:.3f} [{lo:.3f}, {hi:.3f}] | "
          f"{wins / n - ph[ii].mean():+.3f} | {wins / n - pm[ii].mean():+.3f} |")
        out.append((n, ph[ii].mean(), pm[ii].mean(), wins / n))
    w()
    return out
b_sim = bucket_table(ph, "3a. 20 equal-count buckets on SIM P(home win)")
b_mkt = bucket_table(pm, "3b. 20 equal-count buckets on MARKET de-vigged P(home win)")
for nm, bt_ in [("sim", b_sim), ("mkt", b_mkt)]:
    arr = np.array(bt_)
    SUM[f"bucket_{nm}_mae_sim"] = float(np.average(np.abs(arr[:, 3] - arr[:, 1]), weights=arr[:, 0]))
    SUM[f"bucket_{nm}_mae_mkt"] = float(np.average(np.abs(arr[:, 3] - arr[:, 2]), weights=arr[:, 0]))

# slopes
def logit_fit(X, yv, iters=30):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(X @ beta)
        Wt = p * (1 - p) + 1e-9
        Hh = X.T @ (X * Wt[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(Hh, X.T @ (yv - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta
def lg(p):
    return logit(np.clip(p, 1 / (NS * 2 + 2), 1 - 1 / (NS * 2 + 2)))
pnorm = stats.norm.cdf(mu / np.sqrt(s2))  # smooth normal-approx sim win prob (diagnostic only)
def slope_set(ix):
    yy = yw[ix]
    o = {}
    # OLS of realised on predicted (probability scale)
    o["ols_sim"] = ols(ph[ix], yy)[1]
    o["ols_mkt"] = ols(pm[ix], yy)[1]
    # IV (even-seed on odd-seed instrument)
    o["iv_sim"] = np.cov(yy, pB[ix])[0, 1] / np.cov(pA[ix], pB[ix])[0, 1]
    # logistic calibration slope on logit
    o["lg_sim"] = logit_fit(np.column_stack([np.ones(len(ix)), lg(ph[ix])]), yy)[1]
    o["lg_simnorm"] = logit_fit(np.column_stack([np.ones(len(ix)), lg(pnorm[ix])]), yy)[1]
    o["lg_mkt"] = logit_fit(np.column_stack([np.ones(len(ix)), lg(pm[ix])]), yy)[1]
    o["lg_mktprop"] = logit_fit(np.column_stack([np.ones(len(ix)), lg(pmp[ix])]), yy)[1]
    # joint
    bj = logit_fit(np.column_stack([np.ones(len(ix)), lg(pnorm[ix]), lg(pm[ix])]), yy)
    o["joint_sim"], o["joint_mkt"] = bj[1], bj[2]
    # probit slope of outcome on mu/s (width)
    zz = mu[ix] / np.sqrt(s2[ix])
    o["probit_z"] = probit_slope(zz, yy)
    return o
def probit_slope(zz, yy):
    from scipy.optimize import minimize
    def nll(b):
        p = np.clip(stats.norm.cdf(b[0] * zz), 1e-9, 1 - 1e-9)
        return -(yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum()
    return minimize(nll, [1.0], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6)).x[0]
idx3 = np.where(sel)[0]
base_s = slope_set(idx3)
bsl = [slope_set(idx3[i]) for i in boot_idx(n3, 400)]
names = {"ols_sim": "OLS realised ~ sim P (naive; attenuated by 50-seed noise)",
         "iv_sim": "IV realised ~ sim P (even-seed P instrumented by odd-seed P; noise-free)",
         "ols_mkt": "OLS realised ~ market P",
         "lg_sim": "logistic slope, logit(sim P) (50-seed empirical, clipped)",
         "lg_simnorm": "logistic slope, logit(Phi(mu/s)) (smooth sim P)",
         "lg_mkt": "logistic slope, logit(market P, power de-vig)",
         "lg_mktprop": "logistic slope, logit(market P, proportional de-vig)",
         "probit_z": "probit slope of win on z = sim mean / sim own SD (width-implied; slope b => calibrated width = sim SD / b)"}
w("### 3c. Reliability slopes (1.0 = calibrated; >1 = under-confident; <1 = over-confident)\n")
w(f"n = {n3}. 95% = game bootstrap, 400 reps.\n")
w("| fit | slope [95%] |\n|---|---|")
for k, nm in names.items():
    arr = np.array([b[k] for b in bsl])
    w(f"| {nm} | {base_s[k]:.3f} [{np.percentile(arr, 2.5):.3f}, {np.percentile(arr, 97.5):.3f}] |")
    SUM[f"slope3_{k}"] = [float(base_s[k]), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
aj = np.array([[b["joint_sim"], b["joint_mkt"]] for b in bsl])
w(f"\nJoint logistic y ~ logit(sim Phi) + logit(market): sim coefficient {base_s['joint_sim']:.3f} "
  f"[{np.percentile(aj[:, 0], 2.5):.3f}, {np.percentile(aj[:, 0], 97.5):.3f}], market coefficient {base_s['joint_mkt']:.3f} "
  f"[{np.percentile(aj[:, 1], 2.5):.3f}, {np.percentile(aj[:, 1], 97.5):.3f}]. The market carries the weight if the sim coefficient is near zero.\n")
SUM["joint3"] = [float(base_s["joint_sim"]), float(base_s["joint_mkt"])]
w(f"Implied width multiplier from the probit slope: sim SD would need to be x{1 / base_s['probit_z']:.3f} to make win probabilities calibrated *with the sim's own centre* "
  "(diagnostic only; nothing is applied to any output). Because the sim centre is noisier than the market's, a slope below 1 can reflect centre error as well as width.\n")

# sim P regressed on market P (dependent-variable noise does not bias this slope)
sl_ = ols(pm[idx3], ph[idx3])[1]; sb_ = np.array([ols(pm[idx3][i], ph[idx3][i])[1] for i in boot_idx(n3, 400)])
lsl = ols(lg(pm[idx3]), lg(pnorm[idx3]))[1]; lsb = np.array([ols(lg(pm[idx3])[i], lg(pnorm[idx3])[i])[1] for i in boot_idx(n3, 400)])
w(f"Sim-vs-market dispersion of probabilities: OLS slope of sim P on market P = {sl_:.3f} [{np.percentile(sb_, 2.5):.3f}, {np.percentile(sb_, 97.5):.3f}]; slope of logit(Phi(mu/s)) on logit(market P) = {lsl:.3f} [{np.percentile(lsb, 2.5):.3f}, {np.percentile(lsb, 97.5):.3f}]. SD of sim P {ph[idx3].std():.4f} vs market P {pm[idx3].std():.4f}; corr {np.corrcoef(ph[idx3], pm[idx3])[0, 1]:.3f}. Slope > 1 means the sim is MORE confident than the market, < 1 less.\n")
SUM["sim_on_mkt"] = [float(sl_), float(lsl)]

# scores
def ll(p, yv):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(yv * np.log(p) + (1 - yv) * np.log(1 - p)).mean()
def br(p, yv):
    return ((p - yv) ** 2).mean()
w("### 3d. Scores on the same games\n")
w("| predictor | log loss | Brier | 95% (paired vs market, log loss diff) |\n|---|---:|---:|---|")
yy = yw[idx3]
for nm, pp in [("sim empirical P (50 seeds)", ph), ("sim Phi(mu/s)", pnorm), ("market power de-vig", pm), ("market proportional", pmp)]:
    d_ = [ll(pp[idx3[i]], yy[i]) - ll(pm[idx3[i]], yy[i]) for i in boot_idx(n3, 300)]
    w(f"| {nm} | {ll(pp[idx3], yy):.4f} | {br(pp[idx3], yy):.4f} | diff {ll(pp[idx3], yy) - ll(pm[idx3], yy):+.4f} [{np.percentile(d_, 2.5):+.4f}, {np.percentile(d_, 97.5):+.4f}] |")
SUM["ll"] = dict(sim=float(ll(ph[idx3], yy)), simn=float(ll(pnorm[idx3], yy)), mkt=float(ll(pm[idx3], yy)))
w()
# confidence-by-edge: sim vs market disagreement
dp = ph - pm
w("### 3e. Sim minus market probability: who is right when they disagree\n")
w("| bucket of (sim P - market P) | n | mean sim - mkt | realised - market | realised - sim |\n|---|---:|---:|---:|---:|")
qd = pd.qcut(pd.Series(dp[idx3]).rank(method="first"), 5, labels=False).to_numpy()
for k in range(5):
    ii = idx3[qd == k]
    w(f"| Q{k + 1} | {len(ii)} | {dp[ii].mean():+.3f} | {yw[ii].mean() - pm[ii].mean():+.3f} | {yw[ii].mean() - ph[ii].mean():+.3f} |")
w("\nIf the sim were merely under-confident (same centre, too wide) its disagreements with the market would be centred on the market: realised - market ~ 0. "
  "A realised - market gap that tracks the sign of sim - market would mean the sim has real information; a gap of the opposite sign would mean the disagreement is noise.\n")

# PIT
w("### 3f. PIT of the realised margin under each game's 50-seed sim margin distribution\n")
rs = np.random.default_rng(11)
lt = (SM < y[:, None]).sum(1)
eq = (SM == y[:, None]).sum(1)
pit = (lt + rs.random(G) * (eq + 1)) / (NS + 1)  # randomised rank PIT, uniform under exchangeability
def pit_report(mask, label):
    pv = pit[mask]
    n = len(pv)
    h = np.histogram(pv, bins=10, range=(0, 1))[0]
    ex = n / 10
    band = 1.96 * np.sqrt(n * 0.1 * 0.9)
    cm = ((pv > 0.25) & (pv < 0.75)).mean()
    tails = ((pv < 0.1) | (pv > 0.9)).mean()
    ks = stats.kstest(pv, "uniform")
    se50 = np.sqrt(0.25 / n)
    return (f"| {label} | {n} | " + " | ".join(f"{x}" for x in h) + f" | {cm:.3f} ({(cm - 0.5) / se50:+.1f} se) | "
            f"{tails:.3f} ({(tails - 0.2) / np.sqrt(0.16 / n):+.1f} se) | {ks.pvalue:.3g} |"), dict(h=h.tolist(), cm=float(cm), tails=float(tails), ks=float(ks.pvalue), n=n)
w(f"PIT is rank-based on the 50 seeds with randomised tie-breaking (margins are integers), so it is exactly uniform if the real margin is exchangeable with the sim draws. "
  f"Expected per decile = n/10; the 95% band on a decile count is +/- 1.96 sqrt(n*0.09) (n={G}: expected {G / 10:.0f} +/- {1.96 * np.sqrt(G * .09):.0f}). "
  "Central mass = P(0.25 < PIT < 0.75), expected 0.50; a hump (too wide) gives > 0.50. Tail mass = P(PIT < 0.1 or > 0.9), expected 0.20; too wide gives < 0.20.\n")
w("| cell | n | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | central mass (z vs .5) | tail mass (z vs .2) | KS p |\n|---|---:|" + "---:|" * 10 + "---|---|---|")
pits = {}
for label, mk in [("ALL games", np.ones(G, bool)), ("with close", has_line)] + \
        [(f"|spread| Q{k + 1}", q == k) for k in range(5)] + \
        [(f"month {nm}", month == mo) for mo, nm in [(11, "Nov"), (12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")]] + \
        [("conference", cf), ("non-conference", ~cf), ("neutral", neu), ("home/away", ~neu)]:
    s_, d_ = pit_report(mk, label + (" UNDERPOWERED" if mk.sum() < MINN else ""))
    w(s_)
    pits[label] = d_
SUM["pit"] = pits
# PIT of market-centred normal for comparison
sd_m = np.sqrt((y - m)[has_line].var())
pit_m = stats.norm.cdf((y - m)[has_line] / sd_m)
cmm = ((pit_m > .25) & (pit_m < .75)).mean()
w(f"\nReference: PIT of the realised margin under N(market, SD={sd_m:.2f}) (a constant-width market model) has central mass {cmm:.3f} on the same {int(has_line.sum())} games "
  f"(Normal-shape baseline: margins are not exactly normal, so this is the honest ceiling for 'shape' error, not 0.50 exactly).\n")
SUM["pit_mkt_central"] = float(cmm)

# ================================================================ PART 4
w("## 4. Where the width comes from\n")
tg = stg.sort_values(["game_id", "seed", "off_side"], kind="stable")
def tgmat(df, col):
    out = {}
    for side in (0, 1):
        s = df[df.off_side == side].sort_values(["game_id", "seed"])
        out[side] = s[col].to_numpy().reshape(-1, NS).astype(float)
    return out
sp_ = tgmat(stg, "poss")
spts = tgmat(stg, "pts")
ph_, pa_ = sp_[0], sp_[1]
th_, ta_ = spts[0], spts[1]
P = (ph_ + pa_) / 2
PPPh, PPPa = th_ / ph_, ta_ / pa_
Dd = PPPh - PPPa
Tot = th_ + ta_
# sanity: margin from tg equals gs margin
RM = np.vstack([v for _, v in sgs.sort_values(['game_id', 'seed']).groupby('game_id').reg_margin]).astype(float)
assert np.allclose(th_ - ta_, RM)  # sim_tg is REGULATION only; OT enters through SM - RM
# increments from sim_late
def late_incr(df, nseeds_hint):
    d = df[["game_id", "seed", "k", "period", "cs", "post"]].copy()
    d = d.sort_values(["game_id", "seed", "k"], kind="stable")
    last = d.groupby(["game_id", "seed"], sort=True).post.last().rename("final")
    r_ = d[d.period <= 2].groupby(["game_id", "seed"], sort=True).post.last().rename("reg")
    e = d[(d.period == 2) & (d.cs > 240)].groupby(["game_id", "seed"], sort=True).post.last().rename("m36")
    d1 = d[d.period == 1].groupby(["game_id", "seed"], sort=True).post.last().rename("m20")
    return pd.concat([d1, e, r_, last], axis=1)
sl = pd.read_parquet(D + "sim_late.parquet", columns=["game_id", "seed", "k", "period", "cs", "post"])
inc = late_incr(sl, NS)
del sl
inc = inc.reset_index()
def imat(col):
    s = inc.sort_values(["game_id", "seed"])
    return s.pivot(index="game_id", columns="seed", values=col).reindex(gids).to_numpy().astype(float)
M20, M36, MREG, MFIN = imat("m20"), imat("m36"), imat("reg"), imat("final")
M36 = np.where(np.isnan(M36), MREG, M36)
assert np.allclose(MFIN, RM)  # sim_late is regulation-only
rl = pd.read_parquet(D + "real_late.parquet", columns=["game_id", "seed", "k", "period", "cs", "post"])
rinc = late_incr(rl, 1).reset_index()
rinc = rinc.set_index("game_id").reindex(gids)
rinc["m36"] = rinc.m36.fillna(rinc.reg)
rmatch = np.zeros(G, bool)
# verified match: pbp final margin equals verified final, pbp present
rg1 = rgs.set_index("game_id").reindex(gids)
rmatch = (rg1.final_margin.to_numpy() == y) & ~np.isnan(rinc.final.to_numpy()) & (rinc.final.to_numpy() == rg1.reg_margin.to_numpy())
w(f"Real-side counterpart games: {int(rmatch.sum())} of {G} (pbp present and its final equals the verified final).\n")
SUM["n_real_match"] = int(rmatch.sum())

# 4a within-game variance decomposition (additive, exact)
def wg_cov(a, b):  # sum over games of across-seed covariance (ddof=1)
    return ((a - a.mean(1, keepdims=True)) * (b - b.mean(1, keepdims=True))).sum(1) / (NS - 1)
Pm, Dm = P.mean(1, keepdims=True), Dd.mean(1, keepdims=True)
dP, dD = P - Pm, Dd - Dm
comp_pace = Dm * dP
comp_eff = Pm * dD
comp_int = dP * dD
comp_res = RM - P * Dd
comp_ot = SM - RM
V = wg_cov(SM, SM)
tot_v = V.mean()
w("### 4a. Within-game variance of the sim margin (across 50 seeds), attributed by covariance share (shares sum to 1)\n")
w(f"Mean within-game margin variance = {tot_v:.2f} pts^2 (SD {np.sqrt(tot_v):.2f}). regulation margin = P*D + small (sim_tg is regulation-only; OT is its own term), P = mean team possessions, D = PPP_home - PPP_away; "
  "pace term = mean(D)*dP, efficiency term = mean(P)*dD, interaction = dP*dD, remainder = home/away possession imbalance. "
  f"Mean |D| across games {np.abs(Dm).mean():.3f} PPP, so pace only moves the margin through a game's expected point gap.\n")
w("| term | variance of term (pts^2) | share of margin variance (Cov(term, margin)/Var(margin)) |\n|---|---:|---:|")
shares = {}
for nm, c in [("pace (possession-count latent) x expected PPP gap", comp_pace), ("per-possession efficiency (PPP gap) x mean possessions", comp_eff),
              ("pace x efficiency interaction", comp_int), ("possession imbalance remainder (regulation)", comp_res), ("overtime increment", comp_ot)]:
    vv = wg_cov(c, c).mean()
    sh = wg_cov(c, SM).mean() / tot_v
    shares[nm] = float(sh)
    w(f"| {nm} | {vv:.3f} | {sh:.4f} |")
w(f"| total | {tot_v:.2f} | {sum(shares.values()):.4f} |\n")
SUM["share_pace_eff"] = shares
# sequence decomposition
I1 = M36                      # start -> 4:00 left in H2
I2 = MREG - M36               # last 4:00 of regulation
I3 = SM - MREG                # overtime
w("Time-sequence attribution of the same variance (increments of home margin): first 36 min of regulation (to the last possession starting with > 4:00 left in H2), last 4:00 of regulation, overtime.\n")
w("| segment | mean within-game variance | share of margin variance (Cov(incr, margin)/Var) | share of games with any activity |\n|---|---:|---:|---:|")
sh2 = {}
for nm, c, act in [("minutes 0-36", I1, 1.0), ("last 4:00 of regulation", I2, float((I2 != 0).mean())), ("overtime", I3, float((I3 != 0).mean()))]:
    vv = wg_cov(c, c).mean()
    sh = wg_cov(c, SM).mean() / tot_v
    sh2[nm] = float(sh)
    w(f"| {nm} | {vv:.2f} | {sh:.4f} | {act:.3f} |")
SUM["share_time"] = sh2
w(f"\nSim OT rate {(I3[:, :] != 0).mean():.4f} per game-seed (OT with exact tie after OT periods would show as 0 increment, so this is a floor) / sim `ot` flag mean {sgs.ot.mean():.4f}; real OT rate {rgs.ot.mean():.4f} (pbp games), verified-final OT rate {(pg.fin_np > 2).mean():.4f}.\n")

# 4b realised counterparts
w("### 4b. Component SD: sim within-game SD vs realised residual SD (real - sim mean, MC-corrected), games with matching pbp\n")
rt_ = rtg.sort_values(["game_id", "off_side"])
rt_ = rt_.set_index("game_id")
def rside(col, side):
    s = rt_[rt_.off_side == side][col].reindex(gids)
    return s.to_numpy().astype(float)
rph, rpa, rth, rta = rside("poss", 0), rside("poss", 1), rside("pts", 0), rside("pts", 1)
rP = (rph + rpa) / 2
rD = rth / rph - rta / rpa
comps = [
    ("final margin (OT incl.)", SM, y),
    ("regulation margin", MREG, rinc.reg.to_numpy().astype(float)),
    ("margin through minute 36", M36, rinc.m36.to_numpy().astype(float)),
    ("last 4:00 regulation increment", I2, (rinc.reg - rinc.m36).to_numpy().astype(float)),
    ("OT increment", I3, (y - rinc.final.to_numpy().astype(float))),
    ("regulation possessions per team (pace)", P, rP),
    ("regulation PPP home", PPPh, rth / rph),
    ("regulation PPP away", PPPa, rta / rpa),
    ("regulation PPP gap D = home - away", Dd, rD),
    ("regulation total points", Tot, rth + rta),
]
w("| component | n | sim within-game SD | realised residual SD [95%] | ratio sim/real [95%] | mean(real - sim mean) | note |\n|---|---:|---:|---|---|---:|---|")
comp_out = {}
for nm, simM, realv in comps:
    mu_c = simM.mean(1)
    s2_c = simM.var(1, ddof=1)
    ok = rmatch & ~np.isnan(realv)
    c = cell_ratio(ok, ycol=realv, mucol=mu_c, s2col=s2_c)
    note = ""
    if "OT" in nm or "last 4" in nm:
        note = "heavy-tailed / mostly zero: SD ratio is a weak summary"
    flag = " UNDERPOWERED" if c["n"] < MINN else ""
    w(f"| {nm}{flag} | {c['n']} | {c['sim_sd']:.4f} | {c['real_sd']:.4f} [{c['real_lo']:.4f}, {c['real_hi']:.4f}] | "
      f"{c['ratio']:.3f} [{c['ratio_lo']:.3f}, {c['ratio_hi']:.3f}] | {c['bias']:+.4f} | {note} |")
    comp_out[nm] = c
SUM["components"] = comp_out
w("\nCaveat: the realised residual of a component is measured against the sim's own mean for that component, so it includes centre error; a ratio below 1 means the sim's within-game spread is smaller than real-vs-sim error for that component, above 1 larger. "
  "Components are not independent and their ratios do not combine linearly; use 4a for shares and 4b for per-component width checks.\n")

# per |spread| quintile: component ratios for margin & pace & D
w("### 4c. Margin, pace and PPP-gap ratios by |spread| quintile (games with close and matching pbp)\n")
w("| cell | n | margin ratio [95%] | pace P ratio [95%] | PPP gap D ratio [95%] | reg margin ratio [95%] |\n|---|---:|---|---|---|---|")
def rr(simM, realv, mk):
    mu_c, s2_c = simM.mean(1), simM.var(1, ddof=1)
    return cell_ratio(mk & rmatch, ycol=realv, mucol=mu_c, s2col=s2_c)
for k in range(5):
    mk = q == k
    cs = [rr(SM, y, mk), rr(P, rP, mk), rr(Dd, rD, mk), rr(MREG, rinc.reg.to_numpy().astype(float), mk)]
    f = lambda c: f"{c['ratio']:.3f} [{c['ratio_lo']:.3f}, {c['ratio_hi']:.3f}]"
    w(f"| |spread| Q{k + 1} | {cs[0]['n']} | " + " | ".join(f(c) for c in cs) + " |")
w()


# ---- 4e: where inside regulation does the excess land: Var(reg) = Var(m36) + Var(late) + 2 Cov(m36, late)
w("### 4e. Regulation margin variance split into minutes 0-36, last 4:00, and their covariance (sim within-game vs realised residual, MC-corrected)\n")
w("Sim column = mean across games of the across-seed (co)variance. Realised column = mean of (real - sim mean) products, minus the MC-noise share (mean sim (co)variance / 50). Games with matching pbp only.\n")
ok = rmatch & ~np.isnan(rinc.m36.to_numpy()) & ~np.isnan(rinc.final.to_numpy())
a_sim, l_sim = M36, I2
a_mu, l_mu = a_sim.mean(1), l_sim.mean(1)
ra = rinc.m36.to_numpy().astype(float) - a_mu
rlv = (rinc.final - rinc.m36).to_numpy().astype(float) - l_mu
def parts(i):
    v_a = wg_cov(a_sim, a_sim)[ok][i].mean(); v_l = wg_cov(l_sim, l_sim)[ok][i].mean(); c_al = wg_cov(a_sim, l_sim)[ok][i].mean()
    r_a = (ra[ok][i] ** 2).mean() - v_a / NS; r_l = (rlv[ok][i] ** 2).mean() - v_l / NS; r_c = (ra[ok][i] * rlv[ok][i]).mean() - c_al / NS
    return np.array([v_a, v_l, 2 * c_al, v_a + v_l + 2 * c_al, r_a, r_l, 2 * r_c, r_a + r_l + 2 * r_c])
n_ok = int(ok.sum())
bp = parts(np.arange(n_ok))
bb4 = np.array([parts(i) for i in boot_idx(n_ok, 500)])
w("| term | sim (pts^2) | realised (pts^2) [95%] | sim - real [95%] |\n|---|---:|---|---|")
for j, nm in enumerate(["Var(minutes 0-36)", "Var(last 4:00)", "2 Cov(0-36, last 4:00)", "Var(regulation margin)"]):
    dif = bb4[:, j] - bb4[:, j + 4]
    w(f"| {nm} | {bp[j]:.2f} | {bp[j + 4]:.2f} [{np.percentile(bb4[:, j + 4], 2.5):.2f}, {np.percentile(bb4[:, j + 4], 97.5):.2f}] | "
      f"{bp[j] - bp[j + 4]:+.2f} [{np.percentile(dif, 2.5):+.2f}, {np.percentile(dif, 97.5):+.2f}] |")
SUM["split_4e"] = dict(sim=bp[:4].tolist(), real=bp[4:].tolist(), n=n_ok)
w()
# ---- 4f: possession level scaling of the margin SD
okp = rmatch & ~np.isnan(rP)
simP_mean, realP_mean = P.mean(1)[okp].mean(), rP[okp].mean()
w("### 4f. How much of the margin SD ratio is the known possession-count level bias\n")
w(f"Mean regulation possessions per team: sim {simP_mean:.2f}, real {realP_mean:.2f} (ratio {simP_mean / realP_mean:.4f}). Margin = possessions x PPP gap, so a sim that runs {100 * (simP_mean / realP_mean - 1):.1f}% more possessions "
  f"scales margin SD by about the same factor at fixed PPP-gap SD. Regulation margin SD ratio {comp_out['regulation margin']['ratio']:.3f}; PPP-gap SD ratio {comp_out['regulation PPP gap D = home - away']['ratio']:.3f}; "
  f"possession-level factor {simP_mean / realP_mean:.3f}; product {comp_out['regulation PPP gap D = home - away']['ratio'] * simP_mean / realP_mean:.3f}. "
  "(Pace *variance* is not the issue - pace SD ratio 1.01 - but the pace *level* bias, owned by the clock round, also widens the margin in points.)\n")
SUM["poss_level"] = [float(simP_mean), float(realP_mean)]

# Excess-variance accounting: how much of margin excess variance (if any) maps to which source
base = cell_ratio(np.ones(G, bool))
excess = base["sim_sd"] ** 2 - base["real_sd"] ** 2
w(f"### 4d. Excess variance accounting\n\nOverall sim own variance {base['sim_sd'] ** 2:.2f} vs MC-corrected realised residual variance {base['real_sd'] ** 2:.2f}; "
  f"excess = {excess:+.2f} pts^2 ({100 * excess / base['real_sd'] ** 2:+.1f}% of the realised residual variance).\n")
SUM["excess_var"] = float(excess)
json.dump(SUM, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
(OUT / "tables.md").write_text("\n".join(L), encoding="utf-8")
print("done")
