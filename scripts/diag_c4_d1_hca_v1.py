"""diag_c4_d1_hca_v1.py -- D1: home margin level and its channel decomposition, sim vs real, fold 2 (2024-25). DIAGNOSTIC ONLY.
usage: diag_c4_d1_hca_v1.py --tag f2all50 --out docs/tests/hca_channel_decomp_2026-10-09.md [--nboot 400]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_c4_lib_v1 as L  # noqa: E402

RNG = np.random.default_rng(20261009)


def f(x, nd=2):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def ci(a):
    return f"[{f(np.percentile(a, 2.5))}, {f(np.percentile(a, 97.5))}]"


def quint(x):
    x = pd.Series(np.asarray(x, float))
    return pd.qcut(x.rank(method="first"), 5, labels=False).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=400)
    ap.add_argument("--min-n", type=int, default=60)
    a = ap.parse_args()
    D = lambda n: pd.read_parquet(L.OUT / f"{a.tag}_{n}.parquet")  # noqa: E731
    pg, sgs, rgs, stg, rtg = D("pregame"), D("sim_gs"), D("real_gs"), D("sim_tg"), D("real_tg")
    S = sgs.seed.nunique()
    W = []
    w = W.append

    # ---------- game table -------------------------------------------------------------------------------
    pg["fin_margin"] = pg.fin_h - pg.fin_a
    sm = sgs.groupby("game_id").final_margin.mean().rename("sim_margin")
    pg = pg.merge(sm, on="game_id", how="inner")
    pg["neutral"] = pg.neutral_site.fillna(0).astype(int)
    pg["month"] = pd.to_datetime(pg.game_date).dt.strftime("%Y-%m")
    pg["d"] = pg.fin_margin - pg.sim_margin
    seed_means = sgs.merge(pg[["game_id", "neutral"]], on="game_id").groupby(["neutral", "seed"]).final_margin.mean()
    n_all = len(pg)

    w("# D1 home margin: level, channel decomposition, slope (fold 2 / 2024-25), 2026-10-09\n")
    w("DIAGNOSTIC ONLY. Nothing changed, no retrain. Script `scripts/diag_c4_d1_hca_v1.py` (library `diag_c4_lib_v1.py`, extraction `diag_c4_extract_v1.py`).\n")
    w(f"Sim: served stack v3, `engine_v3` inputs, fold 2 / season 2025, **{n_all} games x {S} seeds** per-possession trajectories (`results/trajectories/_runs/f2all50_p*`). "
      "Real: verified finals (`game_finals_v2`) for margins; hoopR/CBBD possession table `possessions_v4otc` for channels. 2025-26 not read. Fold 1 not run (the F1 engine inputs/overlay are not in the served `engine_v3` set; see report).\n")
    w("Definitions. margin = home - away at the final whistle including OT. Sim cell = mean over seeds of the per-game margin, then mean over games; "
      "`seed SE` = SD over seeds of the game-set mean / sqrt(S) (Monte Carlo error of the sim mean only). real - sim: paired by game, SE = SD of the per-game difference / sqrt(n) "
      "(includes real-game sampling noise and sim seed noise). Underpowered = n < %d games.\n" % a.min_n)

    # ---------- (a) -----------------------------------------------------------------------------------------
    w("## (a) Mean home margin, sim vs real\n")
    w("| cell | n games | real | sim | seed SE | real - sim | SE (paired) | 95% | flag |")
    w("|---|---:|---:|---:|---:|---:|---:|---|---|")

    def row(label, m, neutral_for_seed=None):
        x = pg[m]
        n = len(x)
        if n == 0:
            return
        sdiff = x.d.std(ddof=1) / np.sqrt(n)
        if neutral_for_seed is None:
            sub = sgs[sgs.game_id.isin(x.game_id)].groupby("seed").final_margin.mean()
        else:
            sub = sgs[sgs.game_id.isin(x.game_id)].groupby("seed").final_margin.mean()
        se_seed = sub.std(ddof=1) / np.sqrt(S)
        flag = "UNDERPOWERED" if n < a.min_n else ("excludes 0" if abs(x.d.mean()) > 1.96 * sdiff else "")
        w(f"| {label} | {n} | {f(x.fin_margin.mean())} | {f(x.sim_margin.mean())} | {f(se_seed, 3)} | {f(x.d.mean())} | {f(sdiff)} | "
          f"[{f(x.d.mean() - 1.96 * sdiff)}, {f(x.d.mean() + 1.96 * sdiff)}] | {flag} |")

    nn = pg.neutral == 0
    row("ALL games", pg.index == pg.index)
    row("non-neutral", nn)
    row("neutral (listed home)", ~nn)
    for mth in sorted(pg.month.unique()):
        row(f"non-neutral, {mth}", nn & (pg.month == mth))
    row("non-neutral, conference game", nn & pg.same_conf)
    row("non-neutral, non-conference game", nn & ~pg.same_conf)
    row("non-neutral, regular season", nn & (pg.season_type == 2))
    row("non-neutral, postseason", nn & (pg.season_type != 2))
    # window replicated
    win = pg.game_date.astype(str).str[:10].isin(["2025-02-11", "2025-02-15", "2025-02-25", "2025-03-01", "2025-03-04"])
    row("the 5-date window of the trajectory doc (all games)", win)
    # signed-error scale: the home margin is dominated by strength; HCA-like constant from the rating regression is in (c)
    w("\nReading: a pooled mean-margin gap is partly a strength-mix effect (home teams are the favourite more often); the intercept at equal rating is in (c).\n")

    # ---------- (b) channels ---------------------------------------------------------------------------------
    w("## (b) Channel decomposition of the home-minus-away offence gap (non-neutral games)\n")
    ok_real = rgs.set_index("game_id")
    # games whose real pbp path ends at the verified final and has >= 90 possessions
    rt = rtg.groupby("game_id").poss.sum()
    keep = pg[nn].merge(rgs[["game_id", "final_margin"]], on="game_id", how="inner")
    keep = keep[(keep.final_margin == keep.fin_margin) & keep.game_id.map(rt).ge(90)]
    gid = keep.game_id.to_numpy()
    n = len(gid)
    w(f"Games kept: {n} of {int(nn.sum())} non-neutral (real pbp path final = verified final, >= 90 parsed possessions). Regulation possessions only (period <= 2) so OT does not enter the rates. "
      "Sim rates are computed from seed-summed counts per game side; real from the same counts in the possession table; the channel definitions are identical (one code path).\n")

    def side_arrays(tg, seeds_div):
        t = tg[tg.game_id.isin(gid)].groupby(["game_id", "off_side"])[
            ["poss", "fga2", "fgm2", "fga3", "fgm3", "fta", "ftm", "tov", "oreb", "pts", "dur"]].sum() / seeds_div
        H = t.xs(0, level="off_side").reindex(gid)
        A = t.xs(1, level="off_side").reindex(gid)
        return H, A

    Hs, As = side_arrays(stg, S)
    Hr, Ar = side_arrays(rtg, 1)
    ratios = {   # name: (num, den) as callables on a frame of per-game counts
        "poss / team": (lambda t: t.poss, None),
        "PPP": (lambda t: t.pts, lambda t: t.poss),
        "eFG%": (lambda t: t.fgm2 + 1.5 * t.fgm3, lambda t: t.fga2 + t.fga3),
        "2P%": (lambda t: t.fgm2, lambda t: t.fga2),
        "3P%": (lambda t: t.fgm3, lambda t: t.fga3),
        "3PA share": (lambda t: t.fga3, lambda t: t.fga2 + t.fga3),
        "FTA / poss": (lambda t: t.fta, lambda t: t.poss),
        "FT%": (lambda t: t.ftm, lambda t: t.fta),
        "TOV / poss": (lambda t: t.tov, lambda t: t.poss),
        "OREB / poss": (lambda t: t.oreb, lambda t: t.poss),
        "OREB % of misses": (lambda t: t.oreb, lambda t: (t.fga2 + t.fga3 - t.fgm2 - t.fgm3)),
        "sec / poss": (lambda t: t.dur, lambda t: t.poss),
    }
    B = a.nboot
    wts = RNG.multinomial(n, np.ones(n) / n, size=B).astype(float)     # bootstrap game weights

    def pooled_gap(Hf, Af, num, den, wv):
        nh, na = num(Hf).to_numpy(), num(Af).to_numpy()
        if den is None:
            return (wv @ nh - wv @ na) / wv.sum()
        dh, da = den(Hf).to_numpy(), den(Af).to_numpy()
        return (wv @ nh) / (wv @ dh) - (wv @ na) / (wv @ da)

    w("| channel | real home-off | real away-off | real gap (H-A) | sim gap (H-A) | real - sim gap | 95% (game bootstrap) | flag |")
    w("|---|---:|---:|---:|---:|---:|---|---|")
    one = np.ones(n)
    chan_rows = {}
    for name, (num, den) in ratios.items():
        gr = pooled_gap(Hr, Ar, num, den, one)
        gs_ = pooled_gap(Hs, As, num, den, one)
        bs = np.array([pooled_gap(Hr, Ar, num, den, wts[b]) - pooled_gap(Hs, As, num, den, wts[b]) for b in range(B)])
        d = gr - gs_
        sc = 100 if ("%" in name or "share" in name or name in ("FTA / poss", "TOV / poss", "OREB / poss")) else 1
        nd = 2 if sc == 100 else 3
        unit = " pp" if sc == 100 else ""
        h_r = (num(Hr).sum() / (den(Hr).sum() if den else 1)) if den else num(Hr).mean()
        a_r = (num(Ar).sum() / (den(Ar).sum() if den else 1)) if den else num(Ar).mean()
        flag = "excludes 0" if (np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0) else ""
        chan_rows[name] = (gr, gs_, d, bs)
        w(f"| {name} | {f(h_r * sc, nd)} | {f(a_r * sc, nd)} | {f(gr * sc, nd)}{unit} | {f(gs_ * sc, nd)}{unit} | {f(d * sc, nd)}{unit} | "
          f"[{f(np.percentile(bs, 2.5) * sc, nd)}, {f(np.percentile(bs, 97.5) * sc, nd)}] | {flag} |")
    w("\nGap = pooled home-offence rate minus pooled away-offence rate over the same games. Home teams are stronger on average, so the raw gap mixes site and strength; "
      "the sim and real share the same games, so a strength bias in the sim is common to both only if the sim is unbiased on strength (checked in D2). The FE-adjusted version follows.\n")

    # ---- margin contribution by LMDI ---------------------------------------------------------------------------
    def comps(Hf, Af, wv, beta):
        """pooled factor values for home/away offence, then LMDI contributions to PPP gap (points per possession)."""
        out = {}
        st = {}
        for nm, T in (("h", Hf), ("a", Af)):
            P = wv @ T.poss.to_numpy()
            g = lambda c: wv @ T[c].to_numpy()  # noqa: E731
            fga = g("fga2") + g("fga3")
            st[nm] = dict(P=P, A=fga / P, s3=g("fga3") / fga, m2=g("fgm2") / g("fga2"), m3=g("fgm3") / g("fga3"),
                          f=g("fta") / P, ft=g("ftm") / g("fta"), tov=g("tov") / P, oreb=g("oreb") / P, pts=g("pts") / P,
                          pos=P / wv.sum())
        h, aw = st["h"], st["a"]
        T1 = lambda s: s["A"] * s["s3"] * 3 * s["m3"]  # noqa: E731
        T2 = lambda s: s["A"] * (1 - s["s3"]) * 2 * s["m2"]  # noqa: E731
        T3 = lambda s: s["f"] * s["ft"]  # noqa: E731

        def lm(x1, x0):
            return (x1 - x0) / np.log(x1 / x0) if abs(x1 - x0) > 1e-12 else x1

        c = {k: 0.0 for k in ["A", "s3", "m3", "m2", "f", "ft"]}
        for T, facs in ((T1, [("A", 1), ("s3", 1), ("m3", 1)]), (T2, [("A", 1), ("s3", -1), ("m2", 1)]), (T3, [("f", 1), ("ft", 1)])):
            t1, t0 = T(h), T(aw)
            Lw = lm(t1, t0)
            for k, sgn in facs:
                if k == "s3" and sgn == -1:
                    x1, x0 = 1 - h["s3"], 1 - aw["s3"]
                else:
                    x1, x0 = h[k], aw[k]
                c[k] += Lw * np.log(x1 / x0)
        # split the attempts-per-possession effect into tov / oreb / fta-trip share via the fitted linear relation
        dA = h["A"] - aw["A"]
        parts = {"tov": beta["tov"] * (h["tov"] - aw["tov"]), "oreb": beta["oreb"] * (h["oreb"] - aw["oreb"]),
                 "fta": beta["f"] * (h["f"] - aw["f"])}
        tot = sum(parts.values())
        res = dA - tot
        scale = c["A"] / dA if abs(dA) > 1e-12 else 0.0
        cc = {"2P% make": c["m2"], "3P% make": c["m3"], "3PA share": c["s3"], "FT%": c["ft"], "FTA rate (direct)": c["f"],
              "TOV (via attempts)": parts["tov"] * scale, "OREB (via attempts)": parts["oreb"] * scale,
              "FTA-trip (via attempts)": parts["fta"] * scale, "other attempts/poss": res * scale}
        poss_term = (h["pts"] + aw["pts"]) / 2 * ((h["P"] - aw["P"]) / wv.sum())
        ppp_gap = h["pts"] - aw["pts"]
        return cc, ppp_gap, (h["pos"] + aw["pos"]) / 2, poss_term

    # regression of attempts/poss on tov, oreb, fta/poss across real team-games (beta shared by sim and real)
    allr = pd.concat([Hr, Ar])
    X = np.c_[np.ones(len(allr)), allr.tov / allr.poss, allr.oreb / allr.poss, allr.fta / allr.poss]
    yv = ((allr.fga2 + allr.fga3) / allr.poss).to_numpy()
    cf = np.linalg.lstsq(X, yv, rcond=None)[0]
    beta = {"tov": cf[1], "oreb": cf[2], "f": cf[3]}
    w("### Margin contribution of each channel (points of home margin per game; LMDI exact decomposition of the PPP gap x mean possessions)\n")
    w(f"Attempts per possession regressed on TOV, OREB and FTA rates (real team-games, n={len(allr)}): beta TOV {f(beta['tov'], 3)}, OREB {f(beta['oreb'], 3)}, FTA {f(beta['f'], 3)}; "
      "used only to split the attempts-per-possession term.\n")
    w("| channel | real contribution | sim contribution | real - sim | 95% (game bootstrap) | flag |")
    w("|---|---:|---:|---:|---|---|")
    cr, pr_, posr, tr = comps(Hr, Ar, one, beta)
    cs_, ps_, poss_s, ts = comps(Hs, As, one, beta)
    boot = {k: [] for k in cr}
    bt_tot, bt_poss = [], []
    for b in range(B):
        c1, p1, q1, t1 = comps(Hr, Ar, wts[b], beta)
        c2, p2, q2, t2 = comps(Hs, As, wts[b], beta)
        for k in cr:
            boot[k].append((c1[k] * q1) - (c2[k] * q2))
        bt_tot.append(p1 * q1 - p2 * q2 + t1 - t2)
        bt_poss.append(t1 - t2)
    tot_r = sum(cr[k] for k in cr) * posr
    tot_s = sum(cs_[k] for k in cs_) * poss_s
    for k in cr:
        d = cr[k] * posr - cs_[k] * poss_s
        bs = np.array(boot[k])
        fl = "excludes 0" if (np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0) else ""
        w(f"| {k} | {f(cr[k] * posr, 3)} | {f(cs_[k] * poss_s, 3)} | {f(d, 3)} | {ci(bs)} | {fl} |")
    bs = np.array(bt_poss)
    w(f"| possession parity (home - away off. possessions) | {f(tr, 3)} | {f(ts, 3)} | {f(tr - ts, 3)} | {ci(bs)} | |")
    bs = np.array(bt_tot)
    w(f"| **total (home-off PPP gap x poss + parity)** | {f(tot_r + tr, 3)} | {f(tot_s + ts, 3)} | {f(tot_r + tr - tot_s - ts, 3)} | {ci(bs)} | |")
    w(f"\nSanity: real regulation home-minus-away points per game in these games = {f((Hr.pts - Ar.pts).mean(), 3)}; sim = {f((Hs.pts - As.pts).mean(), 3)}. "
      "(Contributions sum to the PPP-gap identity x mean possessions; the identity closes up to the exactness of LMDI and the pooled-vs-mean weighting.)\n")

    # ---- FE-adjusted HCA ---------------------------------------------------------------------------------------
    w("### Team-FE-adjusted site effect (rate ~ offence team + defence team + home-offence indicator; non-neutral games kept above)\n")
    from scipy.sparse import csr_matrix, hstack
    from scipy.sparse.linalg import lsqr
    gtab = pg.set_index("game_id").loc[gid]
    teams = np.unique(np.r_[gtab.home_team_id, gtab.away_team_id])
    ti = {t: i for i, t in enumerate(teams)}
    hi = gtab.home_team_id.map(ti).to_numpy()
    ai = gtab.away_team_id.map(ti).to_numpy()
    nt = len(teams)

    def design(idx_mask_rows):
        # rows: [home offence games ; away offence games]; offence FE, defence FE, home indicator
        r = np.arange(2 * n)
        off = np.r_[hi, ai]
        dfn = np.r_[ai, hi]
        Mo = csr_matrix((np.ones(2 * n), (r, off)), shape=(2 * n, nt))
        Md = csr_matrix((np.ones(2 * n), (r, dfn)), shape=(2 * n, nt))
        hm = csr_matrix((np.r_[np.ones(n), np.zeros(n)], (r, np.zeros(2 * n, int))), shape=(2 * n, 1))
        return hstack([Mo, Md[:, 1:], hm]).tocsr()

    Xd = design(None)

    def fe_hca(Hf, Af, num, den, wv):
        nh, na = num(Hf).to_numpy(), num(Af).to_numpy()
        if den is None:
            y = np.r_[nh, na]
            wt = np.ones(2 * n)
        else:
            dh, da = den(Hf).to_numpy(), den(Af).to_numpy()
            y = np.r_[nh / np.maximum(dh, 1e-9), na / np.maximum(da, 1e-9)]
            wt = np.r_[dh, da]
        wt = np.r_[wv, wv] * wt
        sw = np.sqrt(wt)
        Xw = csr_matrix(Xd.multiply(sw[:, None]))
        return lsqr(Xw, y * sw, atol=1e-10, btol=1e-10, iter_lim=400)[0][-1]

    fe_names = ["PPP", "eFG%", "2P%", "3P%", "FTA / poss", "FT%", "TOV / poss", "OREB / poss", "poss / team"]
    w(f"| channel | real FE site effect | sim FE site effect | real - sim | 95% (game bootstrap, {min(B, 100)} reps) |")
    w("|---|---:|---:|---:|---|")
    fe_res = {}
    for name in fe_names:
        num, den = ratios[name]
        r_ = fe_hca(Hr, Ar, num, den, one)
        s_ = fe_hca(Hs, As, num, den, one)
        bs = np.array([fe_hca(Hr, Ar, num, den, wts[b]) - fe_hca(Hs, As, num, den, wts[b]) for b in range(min(B, 100))])
        sc = 100 if ("%" in name or "/ poss" in name and name != "poss / team") and name != "PPP" else 1
        fe_res[name] = (r_, s_)
        unit = " pp" if sc == 100 else ""
        w(f"| {name} | {f(r_ * sc, 3)}{unit} | {f(s_ * sc, 3)}{unit} | {f((r_ - s_) * sc, 3)}{unit} | [{f(np.percentile(bs, 2.5) * sc, 3)}, {f(np.percentile(bs, 97.5) * sc, 3)}] |")
    ppp_fe = fe_res["PPP"]
    pos_fe = fe_res["poss / team"]
    w(f"\nFE-site PPP effect x 67.9 possessions = real {f(ppp_fe[0] * 67.9, 2)} pts, sim {f(ppp_fe[1] * 67.9, 2)} pts of home advantage per side (x2 is not applied: the indicator is home-offence minus away-offence, i.e. the full home-minus-away gap of one team's offence); poss/team FE site effect real {f(pos_fe[0], 3)}, sim {f(pos_fe[1], 3)}.\n")

    # ---------- (c) slope check ------------------------------------------------------------------------------------
    w("## (c) Slope check: home margin by home-team rating quintile and by |spread| quintile (non-neutral)\n")
    z = pg[nn & pg.home_net.notna() & pg.away_net.notna()].copy()
    z["rdiff"] = z.home_net - z.away_net
    z["hq"] = quint(z.home_net)
    zs = z[z.close_spread_home.notna()].copy()
    zs["sq"] = quint(zs.close_spread_home.abs())

    def qtab(df, col, label, extra):
        w(f"| {label} | n | mean {extra} | real margin | sim margin | real - sim | 95% | flag |")
        w("|---|---:|---:|---:|---:|---:|---|---|")
        for q in range(5):
            x = df[df[col] == q]
            nq = len(x)
            se = x.d.std(ddof=1) / np.sqrt(nq)
            fl = "UNDERPOWERED" if nq < a.min_n else ("excludes 0" if abs(x.d.mean()) > 1.96 * se else "")
            w(f"| Q{q + 1} | {nq} | {f(x[extra].mean(), 2)} | {f(x.fin_margin.mean())} | {f(x.sim_margin.mean())} | {f(x.d.mean())} | "
              f"[{f(x.d.mean() - 1.96 * se)}, {f(x.d.mean() + 1.96 * se)}] | {fl} |")
        w("")

    z["extra"] = z.home_net
    qtab(z, "hq", "home-rating quintile (Q1 weakest home team)", "extra")
    zs["extra"] = zs.close_spread_home.abs()
    qtab(zs, "sq", "|close spread| quintile (Q1 closest)", "extra")

    def ols(x, y):
        X1 = np.c_[np.ones(len(x)), x]
        b = np.linalg.lstsq(X1, y, rcond=None)[0]
        return b

    def boot_ols(df, xcol, B_=300):
        xs = df[xcol].to_numpy()
        yr, ys = df.fin_margin.to_numpy(), df.sim_margin.to_numpy()
        br, bsm = ols(xs, yr), ols(xs, ys)
        diffs = []
        m = len(df)
        for _ in range(B_):
            ix = RNG.integers(0, m, m)
            diffs.append(ols(xs[ix], yr[ix]) - ols(xs[ix], ys[ix]))
        diffs = np.array(diffs)
        return br, bsm, diffs

    w("Regression of final margin on the pre-game predictor, real and sim fitted on the same games (intercept = home margin at zero predictor; slope = points of margin per unit).\n")
    w("| predictor | n | real intercept | sim intercept | int. diff 95% | real slope | sim slope | slope diff 95% |")
    w("|---|---:|---:|---:|---|---:|---:|---|")
    for lab, df, xc in (("own rating diff (home_net - away_net)", z, "rdiff"),
                        ("-close spread (market)", zs.assign(mk=-zs.close_spread_home), "mk")):
        br, bsm, dd = boot_ols(df, xc)
        w(f"| {lab} | {len(df)} | {f(br[0])} | {f(bsm[0])} | {ci(dd[:, 0])} | {f(br[1], 3)} | {f(bsm[1], 3)} | {ci(dd[:, 1])} |")
    w("")

    # per-team
    w("### Per home team: (real - sim) home margin\n")
    ht = pg[nn].groupby("home_team_id").agg(n=("d", "size"), d=("d", "mean"), sd=("d", "std"))
    ht = ht[ht.n >= 8]
    null_sd = np.sqrt((ht.sd ** 2 / ht.n).mean())
    w(f"{len(ht)} home teams with >= 8 non-neutral home games (mean n {f(ht.n.mean(), 1)}). SD across teams of the mean (real - sim) = {f(ht.d.std(), 2)}; "
      f"expected SD from game-level noise alone = {f(null_sd, 2)} (ratio {f(ht.d.std() / null_sd, 2)}; ratio near 1 means no team-specific component beyond noise). "
      f"Share of teams with (real - sim) > 0: {f((ht.d > 0).mean(), 3)}. Per-team cells are individually underpowered (SE about {f(null_sd, 1)} points).\n")

    Path(a.out).write_text("\n".join(W), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
