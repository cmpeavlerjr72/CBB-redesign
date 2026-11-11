"""diag_heuermann_jqmk2_closegrade_v1.py -- close-bucket grade of the served engine, and the yardstick test. DIAGNOSTIC ONLY.

Fold 2 (2024-25) only. Served v3 stack, 200 seeds: results/clock_env10/full/clk10full_ctrl_o0_s200/games.parquet
(5,710 games x 200 seeds; the control of clock round 10 = served v3). Verified finals and the close spread come from
results/diag_c4/f2all50_pregame.parquet (fin_h/fin_a verified finals; close_spread_home = ESPN BET close from
lines_close_v2_verified, as in diag_margin_dispersion_v1). Nothing is simulated; 2025-26 and 2026-27 are never read.

Orientation: favourite = side the close favours (home if close_spread_home < 0; spread 0 -> home by convention).
s = mean sim fav margin, r = real fav margin, c = |close spread|.  s_odd / s_even = means of odd / even seeds (IV).
Bootstrap = game-level, NB resamples.  Writes results/heuermann_jqmk2_closegrade/tables.md (gitignored).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = 2000
RNG = np.random.default_rng(20261111)
EDGES = [0, 3, 6, 10, 15, 20, 1e9]
LAB = ["0-3", "3-6", "6-10", "10-15", "15-20", "20+"]
MINN = 150
OUT = ROOT / "results/heuermann_jqmk2_closegrade"
OUT.mkdir(parents=True, exist_ok=True)
L: list[str] = []


def w(s=""):
    L.append(s)
    print(s)


# ------------------------------------------------------------------ load
pg = pd.read_parquet(ROOT / "results/diag_c4/f2all50_pregame.parquet").sort_values("game_id").reset_index(drop=True)
sim = pd.read_parquet(ROOT / "results/clock_env10/full/clk10full_ctrl_o0_s200/games.parquet",
                      columns=["game_id", "seed", "home_pts", "away_pts"])
sim = sim.sort_values(["game_id", "seed"])
NS = sim.seed.nunique()
G = len(pg)
assert len(sim) == G * NS and (sim.game_id.unique() == pg.game_id.to_numpy()).all()
H = sim.home_pts.to_numpy().reshape(G, NS).astype(float)
A = sim.away_pts.to_numpy().reshape(G, NS).astype(float)
M = H - A
T = H + A
seeds = sim.seed.to_numpy().reshape(G, NS)[0]
odd = seeds % 2 == 1
fin_h = pg.fin_h.to_numpy().astype(float)
fin_a = pg.fin_a.to_numpy().astype(float)
rh = fin_h - fin_a  # real home margin
sh = M.mean(1)  # sim home margin mean
sh_o, sh_e = M[:, odd].mean(1), M[:, ~odd].mean(1)
s2 = M.var(1, ddof=1)
spr = pg.close_spread_home.to_numpy()
lined = ~np.isnan(spr)
sgn = np.where(np.nan_to_num(spr) <= 0, 1.0, -1.0)  # +1 home favourite
c = np.abs(spr)
bk = np.digitize(np.nan_to_num(c), EDGES[1:-1])  # 0..5

# prior games (in-season, from games_universe) for both teams
gu = pd.read_parquet(ROOT / "data/processed/games_universe.parquet")
gu = gu[gu.season == 2025][["game_id", "game_date", "home_team_id", "away_team_id"]]
long = pd.concat([gu[["game_id", "game_date", "home_team_id"]].rename(columns={"home_team_id": "t"}),
                  gu[["game_id", "game_date", "away_team_id"]].rename(columns={"away_team_id": "t"})])
long = long.sort_values(["t", "game_date", "game_id"])
long["prior"] = long.groupby("t").cumcount()
pr = long.set_index(["game_id", "t"]).prior
ph = np.array([pr.get((g, t), np.nan) for g, t in zip(pg.game_id, pg.home_team_id)])
pa = np.array([pr.get((g, t), np.nan) for g, t in zip(pg.game_id, pg.away_team_id)])
minprior = np.fmin(ph, pa)
exp20 = minprior >= 20

# fav-oriented
r = sgn * rh
s = sgn * sh
so, se = sgn * sh_o, sgn * sh_e
fav_sim = np.where(sgn[:, None] > 0, H, A).mean(1)
dog_sim = np.where(sgn[:, None] > 0, A, H).mean(1)
fav_real = np.where(sgn > 0, fin_h, fin_a)
dog_real = np.where(sgn > 0, fin_a, fin_h)
tot_sim = T.mean(1)
tot_real = fin_h + fin_a


# ------------------------------------------------------------------ helpers
def cov(x, y):
    return np.mean((x - x.mean()) * (y - y.mean()))


def ols(y, x):
    return cov(x, y) / cov(x, x)


def iv(y, xa, xb):
    """slope of y on true x, instrumenting noisy xa with independent-noise xb."""
    return cov(y, xb) / cov(xa, xb)


def boot(f, n, reps=NB):
    """f(ix)->1d array. returns point, lo, hi."""
    p = np.asarray(f(np.arange(n)), float)
    bs = np.array([f(RNG.integers(0, n, n)) for _ in range(reps)])
    return p, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0), bs.std(0)


def fmt(p, lo, hi, d=2):
    return f"{p:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def fmtu(p, lo, hi, d=3):
    return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def comp_flag(p, lo, hi):
    return "COMPRESSION" if (lo > 0 or hi < 0) and abs(p) >= 1.0 else "no"


# ------------------------------------------------------------------ test 1 / 2 core
def test1(mask, label):
    idx0 = np.where(mask & lined)[0]
    n_all = len(idx0)
    ca, ra, sa, soa, sea, ba = c[idx0], r[idx0], s[idx0], so[idx0], se[idx0], bk[idx0]
    fs, ds, fr, dr = fav_sim[idx0], dog_sim[idx0], fav_real[idx0], dog_real[idx0]
    ts, tr = tot_sim[idx0], tot_real[idx0]

    def f(ix):
        out = []
        for k in range(6):
            m = ba[ix] == k
            ii = ix[m]
            nn = len(ii)
            if nn < 2:
                out += [np.nan] * 7
                continue
            out += [nn, ca[ii].mean(), ra[ii].mean(), (sa[ii] - ra[ii]).mean(), (ts[ii] - tr[ii]).mean(),
                    (fs[ii] - fr[ii]).mean(), (ds[ii] - dr[ii]).mean()]
        out += [ols(sa[ix], ca[ix]), ols(ra[ix], ca[ix])]
        return np.array(out)

    p, lo, hi, sd = boot(f, n_all)
    w(f"### {label}: n lined games = {n_all}")
    w("")
    w("| close bucket | n | mean close | real fav margin | sim - real fav margin [95% CI] (SE) | compression? | total sim - real [95%] (SE) | fav pts sim - real [95%] | dog pts sim - real [95%] |")
    w("|---|---:|---:|---:|---|---|---|---|---|")
    for k in range(6):
        o = k * 7
        nn = int(p[o])
        up = " UNDERPOWERED" if nn < MINN else ""
        w(f"| {LAB[k]}{up} | {nn} | {p[o+1]:.2f} | {p[o+2]:+.2f} | {fmt(p[o+3], lo[o+3], hi[o+3])} ({sd[o+3]:.2f}) | "
          f"{comp_flag(p[o+3], lo[o+3], hi[o+3]) if nn >= MINN else 'n/a (underpowered)'} | "
          f"{fmt(p[o+4], lo[o+4], hi[o+4])} ({sd[o+4]:.2f}) | {fmt(p[o+5], lo[o+5], hi[o+5])} | {fmt(p[o+6], lo[o+6], hi[o+6])} |")
    w("")
    w(f"Slope of sim fav margin on close: {fmtu(p[42], lo[42], hi[42])}; slope of real fav margin on close: {fmtu(p[43], lo[43], hi[43])}. "
      f"(Regressor is the close, no seed-noise issue.)")
    w("")
    # test 2: bucket on s
    bs_ = np.digitize(np.abs(sh[idx0]), EDGES[1:-1])
    sgn2 = np.where(sh[idx0] >= 0, 1.0, -1.0)  # favourite by the sim
    rr = sgn2 * rh[idx0]
    ss = sgn2 * sh[idx0]
    sso = sgn2 * sh_o[idx0]
    sse = sgn2 * sh_e[idx0]
    # home-orientation arrays for IV slope
    rho, so_h, se_h = rh[idx0], sh_o[idx0], sh_e[idx0]

    def f2(ix):
        out = []
        for k in range(6):
            ii = ix[bs_[ix] == k]
            if len(ii) < 2:
                out += [np.nan] * 4
                continue
            out += [len(ii), ss[ii].mean(), rr[ii].mean(), (rr[ii] - ss[ii]).mean()]
        out += [iv(rho[ix], se_h[ix], so_h[ix]), iv(rr[ix], sse[ix], sso[ix])]
        out += [iv(rho[ix], so_h[ix], se_h[ix])]
        return np.array(out)

    p2, lo2, hi2, sd2 = boot(f2, n_all)
    w(f"Test 2 ({label}): buckets on |s| (favourite by the SIM), real - sim:")
    w("")
    w("| |s| bucket | n | mean sim fav margin | mean real | real - sim [95% CI] (SE) |")
    w("|---|---:|---:|---:|---|")
    for k in range(6):
        o = k * 4
        nn = int(p2[o])
        up = " UNDERPOWERED" if nn < MINN else ""
        w(f"| {LAB[k]}{up} | {nn} | {p2[o+1]:.2f} | {p2[o+2]:+.2f} | {fmt(p2[o+3], lo2[o+3], hi2[o+3])} ({sd2[o+3]:.2f}) |")
    w("")
    w(f"IV slope of real home margin on sim home margin (odd/even instrument): {fmtu(p2[24], lo2[24], hi2[24])} "
      f"(other direction {p2[26]:.3f}); sim-favourite orientation: {fmtu(p2[25], lo2[25], hi2[25])}.")
    w("")
    return dict(slope_sc=p[42], slope_rc=p[43], iv=p2[24])


w("# Tables (script diag_heuermann_jqmk2_closegrade_v1.py)")
w("")
w(f"Sim: served v3, {NS} seeds, {G} games; lined games {int(lined.sum())}. Truth: verified finals (OT included). "
  f"Margins are favourite-minus-underdog; favourite from the close. Bootstrap {NB} game-level resamples.")
w("")
w(f"Experienced (both teams >= 20 prior in-season games in games_universe): {int((exp20 & lined).sum())} lined games; rest {int((~exp20 & lined).sum())} "
  f"(games with an unknown prior count are in the rest: {int(np.isnan(minprior).sum())}).")
w("")
res_all = test1(np.ones(G, bool), "ALL lined games")

# ------------------------------------------------------------------ test 3
L_idx = np.where(lined)[0]
n = len(L_idx)
cL, rL, sL, soL, seL = c[L_idx], r[L_idx], s[L_idx], so[L_idx], se[L_idx]
sgL = sgn[L_idx]
rhL, shL, soh, seh = rh[L_idx], sh[L_idx], sh_o[L_idx], sh_e[L_idx]
s2L = s2[L_idx]
bkL = bk[L_idx]
ts_, tr_ = tot_sim[L_idx], tot_real[L_idx]


def three(ix, rng=None):
    """all of test 3 on resample ix; returns dict of arrays."""
    d = {}
    ci, ri, si, soi, sei = cL[ix], rL[ix], sL[ix], soL[ix], seL[ix]
    # 3a
    var_s_obs = cov(si, si)
    var_s_true = cov(soi, sei)  # noise-free variance of the sim mean (even/odd covariance)
    var_noise = var_s_obs - var_s_true
    d["var_s_obs"] = var_s_obs
    d["var_s_true"] = var_s_true
    d["cov_sr"] = cov(si, ri)
    d["cov_rc"] = cov(ri, ci)
    d["var_c"] = cov(ci, ci)
    d["cov_sc"] = cov(si, ci)
    d["slope_sc"] = cov(si, ci) / cov(ci, ci)
    d["slope_rc"] = cov(ri, ci) / cov(ci, ci)
    d["implied_A"] = var_s_true / cov(ci, ci)  # nested-information, c calibrated
    d["implied_B"] = var_s_true / cov(ri, ci)  # nested-information, c's own slope on r is not assumed 1
    d["slope_rs_iv"] = iv(ri, sei, soi)
    d["var_noise"] = var_noise
    # same algebra in home orientation with the signed close m = -spread (market centre for the home margin)
    mh = -spr[L_idx][ix]
    rh_, so_h_, se_h_ = rhL[ix], soh[ix], seh[ix]
    sm_ = shL[ix]
    d["h_var_s_true"] = cov(so_h_, se_h_)
    d["h_var_m"] = cov(mh, mh)
    d["h_cov_sr"] = cov(sm_, rh_)
    d["h_cov_rm"] = cov(rh_, mh)
    d["h_slope_sm"] = cov(sm_, mh) / cov(mh, mh)
    d["h_slope_rm"] = cov(rh_, mh) / cov(mh, mh)
    d["h_implied_A"] = d["h_var_s_true"] / cov(mh, mh)
    d["h_slope_rs_iv"] = iv(rh_, se_h_, so_h_)
    # 3b cross-fit in home orientation
    n_ = len(ix)
    fold = (RNG.integers(0, 2, n_) if rng is None else rng.integers(0, 2, n_))
    rho_, sh_, so_, se_ = rhL[ix], shL[ix], soh[ix], seh[ix]
    sstar = np.empty(n_)
    sstar_o = np.empty(n_)
    sstar_e = np.empty(n_)
    for f_ in (0, 1):
        tr = fold != f_
        te = ~tr
        b = iv(rho_[tr], se_[tr], so_[tr])
        a = rho_[tr].mean() - b * sh_[tr].mean()
        sstar[te] = a + b * sh_[te]
        sstar_o[te] = a + b * so_[te]
        sstar_e[te] = a + b * se_[te]
    sg = sgL[ix]
    d["star_slope_rs"] = iv(rho_, sstar_e, sstar_o)
    sstar_f = sg * sstar
    d["sstar_f"] = sstar_f
    # 3c stretch s so that slope on c equals real slope
    k = (cov(ri, ci) / cov(ci, ci)) / (cov(si, ci) / cov(ci, ci))
    d["k"] = k
    d["str_slope_rs"] = iv(ri, k * sei, k * soi)
    sstr = k * si
    # 3d MAEs (fav orientation)
    d["mae_s"] = np.abs(ri - si)
    d["mae_star"] = np.abs(ri - sstar_f)
    d["mae_str"] = np.abs(ri - sstr)
    d["mae_c"] = np.abs(ri - ci)
    d["sstr"] = sstr
    d["dmae_star"] = d["mae_star"] - d["mae_s"]
    d["dmae_str"] = d["mae_str"] - d["mae_s"]
    d["dmae_c"] = d["mae_c"] - d["mae_s"]
    return d


K1 = ["var_s_obs", "var_s_true", "var_noise", "cov_sr", "cov_rc", "var_c", "cov_sc", "slope_sc", "slope_rc",
      "implied_A", "implied_B", "slope_rs_iv", "star_slope_rs", "k", "str_slope_rs",
      "h_var_s_true", "h_var_m", "h_cov_sr", "h_cov_rm", "h_slope_sm", "h_slope_rm", "h_implied_A", "h_slope_rs_iv"]
K2 = ["mae_s", "mae_star", "mae_str", "mae_c", "dmae_star", "dmae_str", "dmae_c"]


def f_three(ix):
    d = three(ix)
    ri, ci, ba = rL[ix], cL[ix], bkL[ix]
    out = [d[k] for k in K1]
    out += [d[m].mean() for m in K2]
    # s* close-bucket gaps (sim - real, fav orientation)
    ss_ = d["sstar_f"]
    for kb in range(6):
        m = ba == kb
        out += [m.sum(), (ss_[m] - ri[m]).mean() if m.sum() > 1 else np.nan]
    for kb in range(6):
        m = ba == kb
        out += [(d["mae_s"][m].mean(), d["mae_star"][m].mean(), d["mae_str"][m].mean(), d["mae_c"][m].mean())[j] if m.sum() > 1 else np.nan
                for j in range(4)]
    # s* gaps on the stretched predictor too (check): sstr - r
    for kb in range(6):
        m = ba == kb
        out += [(d["sstr"][m] - ri[m]).mean() if m.sum() > 1 else np.nan]
    return np.array(out)


p3, lo3, hi3, sd3 = boot(f_three, n)
names = K1 + K2
V = {nm: (p3[i], lo3[i], hi3[i]) for i, nm in enumerate(names)}
o = len(names)
w("## Test 3a: algebra and numbers")
w("")
w("Let e = r - s. Calibration means E[r | s] = s, so E[e | s] = 0 and cov(s, r) = var(s) (var(s) = noise-free variance of the sim mean).")
w("cov(s, c) = cov(r - e, c) = cov(r, c) - cov(e, c). Calibration alone does not fix cov(e, c); an assumption is needed.")
w("ASSUMPTION N (nested information): s = E[r | F_s], c = E[r | F_c], F_s a subset of F_c. Then E[c | F_s] = s, so cov(s, c) = var(s), and")
w("the slope of s on c is var(s) / var(c) < 1 whenever the market has more variance than the sim (it knows more). Form B uses cov(r, c) in place of var(c) (does not assume c calibrated).")
w("Seed noise: var(s) is measured as cov(s_odd, s_even), which is free of the independent seed noise.")
w("")
w("| quantity | value [95%] |")
w("|---|---|")
for nm in ["var_s_obs", "var_noise", "var_s_true", "var_c", "cov_sr", "cov_rc", "cov_sc"]:
    w(f"| {nm} | {fmtu(*V[nm], d=2)} |")
w(f"| check calibration: cov(s,r) vs var(s_true) (diff) | {V['cov_sr'][0]-V['var_s_true'][0]:+.2f} |")
w(f"| IV slope of r on s (fav-by-close orientation) | {fmtu(*V['slope_rs_iv'])} |")
w(f"| real slope of r on c (= cov(r,c)/var(c)) | {fmtu(*V['slope_rc'])} |")
w(f"| MEASURED slope of s on c | {fmtu(*V['slope_sc'])} |")
w(f"| IMPLIED slope of s on c, form A: var(s_true)/var(c) | {fmtu(*V['implied_A'])} |")
w(f"| IMPLIED slope of s on c, form B: var(s_true)/cov(r,c) | {fmtu(*V['implied_B'])} |")
w("")
w("Same algebra in HOME orientation (regressor m = -close spread, signed; this is the orientation the cross-fit below uses):")
w("")
w("| quantity | value [95%] |")
w("|---|---|")
for nm in ["h_var_s_true", "h_var_m", "h_cov_sr", "h_cov_rm"]:
    w(f"| {nm} | {fmtu(*V[nm], d=2)} |")
w(f"| IV slope of r on s | {fmtu(*V['h_slope_rs_iv'])} |")
w(f"| real slope of r on m | {fmtu(*V['h_slope_rm'])} |")
w(f"| MEASURED slope of s on m | {fmtu(*V['h_slope_sm'])} |")
w(f"| IMPLIED slope of s on m (nested + m calibrated): var(s_true)/var(m) | {fmtu(*V['h_implied_A'])} |")
w("")
w("## Test 3b: constructed calibrated predictor s* (2-fold cross-fit, fitted on home margin, applied to fav orientation)")
w("")
w(f"IV slope of r on s* = {fmtu(*V['star_slope_rs'])}. Gaps below are s* - real fav margin (bootstrap redoes the cross-fit each resample).")
w("")
w("| close bucket | n | s* - real [95% CI] | compression-rule flag | s - real (test 1, for comparison) |")
w("|---|---:|---|---|---|")
# gaps s for comparison from test1: recompute directly
for kb in range(6):
    i_n = o + kb * 2
    i_g = i_n + 1
    nn = int(p3[i_n])
    m = bkL == kb
    gs = (sL[m] - rL[m]).mean()
    up = " UNDERPOWERED" if nn < MINN else ""
    w(f"| {LAB[kb]}{up} | {nn} | {fmt(p3[i_g], lo3[i_g], hi3[i_g])} | "
      f"{'excludes 0' if (lo3[i_g] > 0 or hi3[i_g] < 0) else 'includes 0'}{'' if nn >= MINN else ' (underpowered, not read)'} | {gs:+.2f} |")
w("")
sh_hi = [(kb, p3[o + kb * 2 + 1], lo3[o + kb * 2 + 1], hi3[o + kb * 2 + 1]) for kb in (4, 5)]
shown = any((lo > 0 or hi < 0) for _, _, lo, hi in sh_hi)
w(f"Claim (B) by the pre-registered rule (s* gap CI excludes 0 at 15-20 or 20+): **{'SHOWN' if shown else 'NOT SHOWN'}**.")
w("")
w("## Test 3c: stretch s linearly (about 0, fav orientation) so its slope on c equals the real slope of r on c")
w("")
w(f"Stretch factor k = real slope / sim slope = {fmtu(*V['k'])}. IV slope of r on the stretched predictor = {fmtu(*V['str_slope_rs'])} "
  f"(= {1/V['k'][0]:.3f} x IV slope of r on s by construction; 1.0 would mean 'fixing' the close slope kept calibration).")
w("")
w("## Test 3d: margin MAE vs real (fav orientation)")
w("")
w("| predictor | MAE overall [95%] |")
w("|---|---|")
for nm, lab in [("mae_s", "s (sim mean)"), ("mae_star", "s* (calibrated)"), ("mae_str", "s stretched (3c)"), ("mae_c", "c (close, zero HA offset)")]:
    w(f"| {lab} | {fmtu(*V[nm])} |")
w("")
w("Paired differences (bootstrap), MAE(predictor) - MAE(s): " + "; ".join(f"{lab} {fmt(*V[nm], d=3)}" for nm, lab in [("dmae_star", "s*"), ("dmae_str", "stretched"), ("dmae_c", "c")]))
w("")
w("| close bucket | n | MAE s | MAE s* | MAE stretched | MAE c |")
w("|---|---:|---:|---:|---:|---:|")
o2 = o + 12
for kb in range(6):
    nn = int(p3[o + kb * 2])
    row = [p3[o2 + kb * 4 + j] for j in range(4)]
    w(f"| {LAB[kb]}{' UNDERPOWERED' if nn < MINN else ''} | {nn} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
w("")
# stretched gaps
o3 = o2 + 24
w("Stretched-predictor gap (k*s - real) per bucket, for reference: " +
  "; ".join(f"{LAB[kb]} {p3[o3+kb]:+.2f} [{lo3[o3+kb]:+.2f}, {hi3[o3+kb]:+.2f}]" for kb in range(6)))
w("")

# ------------------------------------------------------------------ test 4
w("## Test 4: experienced vs rest")
w("")
test1(exp20, "EXPERIENCED (both teams >= 20 prior in-season games)")
test1(~exp20, "REST (at least one team < 20 prior games, or unknown)")

# all-games test 2 slope
iv_all = iv(rh, sh_e, sh_o)
w(f"All {G} games (no line needed): IV slope of real home margin on sim home margin = {iv_all:.3f}.")
w("")
(OUT / "tables.md").write_text("\n".join(L), encoding="utf-8")
json.dump({k: [float(x) for x in v] for k, v in V.items()}, open(OUT / "summary.json", "w"), indent=1)
