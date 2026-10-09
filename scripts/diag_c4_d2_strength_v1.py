"""diag_c4_d2_strength_v1.py -- D2: responsiveness to team strength (totals, possessions vs PPP, rating compression, lead-change slopes).
DIAGNOSTIC ONLY.  usage: diag_c4_d2_strength_v1.py --tag f2all50 --out docs/tests/strength_responsiveness_decomp_2026-10-09.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_c4_lib_v1 as L  # noqa: E402

RNG = np.random.default_rng(20261010)
CNT = ["poss", "fga2", "fgm2", "fga3", "fgm3", "fta", "ftm", "tov", "oreb", "pts", "dur", "poss_h1", "poss_h2", "dur_h1", "dur_h2"]


def f(x, nd=2):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def ci(a, nd=2):
    return f"[{f(np.percentile(a, 2.5), nd)}, {f(np.percentile(a, 97.5), nd)}]"


def quint(x):
    return pd.qcut(pd.Series(np.asarray(x, float)).rank(method="first"), 5, labels=False).to_numpy()


def lm(x1, x0):
    return (x1 - x0) / np.log(x1 / x0) if abs(x1 - x0) > 1e-12 else x1


def state(T, wv=None):
    """pooled channel state of a frame of per-team-game counts (rows = team-games)."""
    wv = np.ones(len(T)) if wv is None else wv
    g = lambda c: float(wv @ T[c].to_numpy())  # noqa: E731
    P = g("poss")
    fga = g("fga2") + g("fga3")
    return dict(P=P, n=float(wv.sum()), A=fga / P, s3=g("fga3") / fga, m2=g("fgm2") / g("fga2"), m3=g("fgm3") / g("fga3"),
                f=g("fta") / P, ft=g("ftm") / g("fta"), tov=g("tov") / P, oreb=g("oreb") / P, pts=g("pts") / P,
                efg=(g("fgm2") + 1.5 * g("fgm3")) / fga, pos=P / wv.sum())


def ppp_contrib(sr, ss, beta):
    """LMDI split of PPP(real) - PPP(sim) into channel parts (points per possession)."""
    def T1(s): return s["A"] * s["s3"] * 3 * s["m3"]
    def T2(s): return s["A"] * (1 - s["s3"]) * 2 * s["m2"]
    def T3(s): return s["f"] * s["ft"]
    c = dict(A=0.0, s3=0.0, m3=0.0, m2=0.0, f=0.0, ft=0.0)
    for T, facs in ((T1, [("A", 0), ("s3", 0), ("m3", 0)]), (T2, [("A", 0), ("s3", 1), ("m2", 0)]), (T3, [("f", 0), ("ft", 0)])):
        Lw = lm(T(sr), T(ss))
        for k, inv in facs:
            x1, x0 = (1 - sr["s3"], 1 - ss["s3"]) if inv else (sr[k], ss[k])
            c[k] += Lw * np.log(x1 / x0)
    dA = sr["A"] - ss["A"]
    pt = {"tov": beta["tov"] * (sr["tov"] - ss["tov"]), "oreb": beta["oreb"] * (sr["oreb"] - ss["oreb"]),
          "ftrip": beta["f"] * (sr["f"] - ss["f"])}
    sc = c["A"] / dA if abs(dA) > 1e-12 else 0.0
    out = {"2P% make": c["m2"], "3P% make": c["m3"], "3PA share": c["s3"], "FT%": c["ft"], "FTA rate (direct)": c["f"],
           "TOV (via attempts)": pt["tov"] * sc, "OREB (via attempts)": pt["oreb"] * sc, "FTA-trip (via attempts)": pt["ftrip"] * sc,
           "other attempts/poss": (dA - sum(pt.values())) * sc}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--min-n", type=int, default=60)
    a = ap.parse_args()
    D = lambda n: pd.read_parquet(L.OUT / f"{a.tag}_{n}.parquet")  # noqa: E731
    pg, sgs, rgs, stg, rtg = D("pregame"), D("sim_gs"), D("real_gs"), D("sim_tg"), D("real_tg")
    S = sgs.seed.nunique()
    B = a.nboot
    W = []
    w = W.append
    sgm = sgs.groupby("game_id")[["final_margin", "lead_changes", "ties", "t_decision", "largest_lead", "total_pts", "ot"]].mean().add_prefix("s_")
    rg = rgs.set_index("game_id")[["final_margin", "lead_changes", "ties", "t_decision", "largest_lead", "total_pts", "ot"]].add_prefix("r_")
    pg["fin_margin"] = pg.fin_h - pg.fin_a
    g = pg.merge(sgm, on="game_id").merge(rg, on="game_id")
    g["neutral"] = g.neutral_site.fillna(0).astype(int)
    g["real_total"] = g.fin_h + g.fin_a
    g = g[(g.r_final_margin == g.fin_margin)]            # real path ends at the verified final
    rt = rtg.groupby("game_id").poss.sum()
    g = g[g.game_id.map(rt).ge(90)]
    s = int(g._rating_sign.iloc[0])
    gr = g[g.home_net.notna() & g.away_net.notna()].copy()
    gr["rdiff"] = gr.home_net - gr.away_net
    gr["hq"] = quint(gr.home_net)
    gr["abs_spread"] = gr.close_spread_home.abs()
    n_all = len(gr)
    w("# D2 responsiveness to strength: totals, possessions vs PPP, rating compression, lead-change slopes (fold 2 / 2024-25), 2026-10-09\n")
    w("DIAGNOSTIC ONLY. Nothing changed. Script `scripts/diag_c4_d2_strength_v1.py` on `diag_c4_extract_v1.py` caches.\n")
    w(f"Sim: served stack v3, fold 2 / season 2025, {int(sgs.game_id.nunique())} games x {S} seeds. Games analysed: {n_all} with own as-of ratings, real pbp path = verified final and >= 90 possessions "
      f"(neutral included unless stated). Rating = own as-of net (off_c {'+' if s > 0 else '-'} def_c), day-before snapshot. Spread = ESPN BET close (n with line = {int(gr.close_spread_home.notna().sum())}). "
      f"Totals include OT unless 'regulation'; channels are regulation (period <= 2). Real game sampling noise is in every CI (game-cluster bootstrap, {B} reps); the sim is seed-averaged "
      f"(seed MC SE of a quintile mean is shown where relevant). Underpowered = n < {a.min_n}.\n")

    # team-game frames, joined to quintiles
    def tgframe(tg, div):
        t = tg.groupby(["game_id", "off_side"])[CNT].sum() / div
        t = t.reset_index().merge(gr[["game_id", "rdiff", "hq", "abs_spread", "home_off_c", "home_def_c", "away_off_c", "away_def_c", "neutral"]], on="game_id")
        t["strong"] = np.where(t.rdiff >= 0, 0, 1) == t.off_side
        return t
    Tr, Ts = tgframe(rtg, 1), tgframe(stg, S)
    gr["sq"] = np.where(gr.abs_spread.notna(), quint(gr.abs_spread.fillna(gr.abs_spread.median())), -1)
    Tr = Tr.merge(gr[["game_id", "sq"]], on="game_id")
    Ts = Ts.merge(gr[["game_id", "sq"]], on="game_id")
    allr = Tr
    X = np.c_[np.ones(len(allr)), allr.tov / allr.poss, allr.oreb / allr.poss, allr.fta / allr.poss]
    cf = np.linalg.lstsq(X, ((allr.fga2 + allr.fga3) / allr.poss).to_numpy(), rcond=None)[0]
    beta = {"tov": cf[1], "oreb": cf[2], "f": cf[3]}

    # ---------------- (a) --------------------------------------------------------------------------------
    def part_a(qcol, qname, qmeanname):
        w(f"## (a) {qname}: total points gap = possessions + points per possession; PPP by strong / opposing offence; PPP by channel\n")
        w("Total here is REGULATION points (both teams) so the channel split closes; the OT-inclusive total gap is in the first table's last column.\n")
        w("| cell | n | mean " + qmeanname + " | real total (incl OT) | sim total (incl OT) | real - sim [95%] | seed SE | reg total real | reg total sim | gap reg | of which possessions | of which PPP |")
        w("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        rows = {}
        for q in range(5):
            ids = gr[gr[qcol] == q].game_id.to_numpy()
            if len(ids) == 0:
                continue
            gq = gr[gr[qcol] == q]
            tr = Tr[Tr.game_id.isin(ids)]
            ts = Ts[Ts.game_id.isin(ids)]
            # per-game totals
            reg_r = tr.groupby("game_id").pts.sum()
            reg_s = ts.groupby("game_id").pts.sum()
            pos_r = tr.groupby("game_id").poss.sum()
            pos_s = ts.groupby("game_id").poss.sum()
            d_inc = gq.real_total.to_numpy() - gq.s_total_pts.to_numpy()
            se = d_inc.std(ddof=1) / np.sqrt(len(gq))
            sub = sgs[sgs.game_id.isin(ids)].groupby("seed").total_pts.mean()
            tot_r, tot_s = reg_r.mean(), reg_s.mean()
            Pr, Ps = pos_r.mean(), pos_s.mean()
            pr, ps = tot_r / Pr, tot_s / Ps
            Lw = lm(tot_r, tot_s)
            c_pos, c_ppp = Lw * np.log(Pr / Ps), Lw * np.log(pr / ps)
            fl = " UNDERPOWERED" if len(gq) < a.min_n else ""
            mean_q = gq[("home_net" if qcol == "hq" else "abs_spread")].mean()
            w(f"| Q{q + 1}{fl} | {len(gq)} | {f(mean_q)} | {f(gq.real_total.mean())} | {f(gq.s_total_pts.mean())} | {f(d_inc.mean())} [{f(d_inc.mean() - 1.96 * se)}, {f(d_inc.mean() + 1.96 * se)}] | {f(sub.std(ddof=1) / np.sqrt(S), 3)} | "
              f"{f(tot_r)} | {f(tot_s)} | {f(tot_r - tot_s)} | {f(c_pos)} | {f(c_ppp)} |")
            rows[q] = (ids, tr, ts)
        w("")
        # PPP strong vs opposing offence + channels
        w(f"| cell | side | n team-games | poss/team real | sim | PPP real | sim | PPP gap [95% game bootstrap] | eFG% real/sim | FTA/poss real/sim | TOV/poss real/sim | OREB/poss real/sim |")
        w("|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|")
        for q, (ids, tr, ts) in rows.items():
            for nm, flag in (("strong-side offence", True), ("opposing offence", False)):
                a1, b1 = tr[tr.strong == flag], ts[ts.strong == flag]
                sr_, ss_ = state(a1), state(b1)
                gid_idx = {gid: i for i, gid in enumerate(ids)}
                ra = a1.game_id.map(gid_idx).to_numpy()
                sa = b1.game_id.map(gid_idx).to_numpy()
                wts = RNG.multinomial(len(ids), np.ones(len(ids)) / len(ids), size=B).astype(float)
                bs = []
                for b in range(B):
                    sr2, ss2 = state(a1, wts[b][ra]), state(b1, wts[b][sa])
                    bs.append(sr2["pts"] - ss2["pts"])
                fl = " (UNDERPOWERED)" if len(ids) < a.min_n else ""
                w(f"| Q{q + 1}{fl} | {nm} | {len(a1)} | {f(sr_['pos'])} | {f(ss_['pos'])} | {f(sr_['pts'], 3)} | {f(ss_['pts'], 3)} | {f(sr_['pts'] - ss_['pts'], 3)} {ci(np.array(bs), 3)} | "
                  f"{f(sr_['efg'] * 100)} / {f(ss_['efg'] * 100)} | {f(sr_['f'] * 100)} / {f(ss_['f'] * 100)} | {f(sr_['tov'] * 100)} / {f(ss_['tov'] * 100)} | {f(sr_['oreb'] * 100)} / {f(ss_['oreb'] * 100)} |")
        w("")
        # channel LMDI for the extreme quintile and for Q1
        w("PPP gap (real - sim) by channel, points per team-game (LMDI x possessions), all offences in the cell:\n")
        w("| cell | " + " | ".join(["2P% make", "3P% make", "3PA share", "FT%", "FTA rate (direct)", "TOV (via attempts)", "OREB (via attempts)", "FTA-trip (via attempts)", "other attempts/poss"]) + " | total |")
        w("|---|" + "---:|" * 10)
        for q, (ids, tr, ts) in rows.items():
            sr_, ss_ = state(tr), state(ts)
            c = ppp_contrib(sr_, ss_, beta)
            pos = (sr_["pos"] + ss_["pos"]) / 2
            vals = [c[k] * pos for k in c]
            w(f"| Q{q + 1} | " + " | ".join(f(v, 2) for v in vals) + f" | {f(sum(vals), 2)} |")
        w("")
        return rows

    rows_h = part_a("hq", "By home-team rating quintile (Q1 weakest home team)", "home_net")
    grs = gr[gr.sq >= 0]
    _ = part_a("sq", "By |close spread| quintile (Q1 closest)", "|spread|")

    # where in the game does the possession response live? possessions and seconds per possession by half
    w("### Where in the game the possession gap lives: possessions per team-game and seconds per possession, by half\n")
    w("| cut | quintile | n games | H1 poss/team real / sim | H2 poss/team real / sim | H1 sec/poss real / sim | H2 sec/poss real / sim |")
    w("|---|---|---:|---|---|---|---|")
    for qcol, lab in (("hq", "home-rating"), ("sq", "|spread|")):
        for q in range(5):
            ids = gr[gr[qcol] == q].game_id
            a1, b1 = Tr[Tr.game_id.isin(ids)], Ts[Ts.game_id.isin(ids)]
            m = lambda T, c: T[c].sum()  # noqa: E731
            w(f"| {lab} | Q{q + 1} | {len(ids)} | {f(m(a1, 'poss_h1') / len(a1))} / {f(m(b1, 'poss_h1') / len(b1))} | {f(m(a1, 'poss_h2') / len(a1))} / {f(m(b1, 'poss_h2') / len(b1))} | "
              f"{f(m(a1, 'dur_h1') / m(a1, 'poss_h1'))} / {f(m(b1, 'dur_h1') / m(b1, 'poss_h1'))} | {f(m(a1, 'dur_h2') / m(a1, 'poss_h2'))} / {f(m(b1, 'dur_h2') / m(b1, 'poss_h2'))} |")
    w("")

    # ---------------- (b) rating compression vs mapping ------------------------------------------------------
    w("## (b) Is the compression in the rating itself or in the rating -> rate mapping?\n")
    nn = gr[gr.neutral == 0].copy()

    def ols(x, y):
        X1 = np.c_[np.ones(len(x)), x]
        return np.linalg.lstsq(X1, y, rcond=None)[0]

    def bslope(df, xc, yr, ys, B_=300):
        xs = df[xc].to_numpy()
        a_, b_ = df[yr].to_numpy(), df[ys].to_numpy()
        br, bsm = ols(xs, a_), ols(xs, b_)
        dd = []
        m = len(df)
        for _ in range(B_):
            ix = RNG.integers(0, m, m)
            dd.append([ols(xs[ix], a_[ix])[1], ols(xs[ix], b_[ix])[1]])
        dd = np.array(dd)
        return br, bsm, dd
    nn["x_tot"] = nn.home_off_c + nn.away_off_c - s * (nn.home_def_c + nn.away_def_c)
    gr["x_tot"] = gr.home_off_c + gr.away_off_c - s * (gr.home_def_c + gr.away_def_c)
    w("Regression of the game outcome on our pre-game rating, real outcome vs sim seed-mean outcome on the SAME games and SAME predictor. If the sim passed the rating's information through faithfully, the two slopes agree; "
      "a sim slope BELOW the real slope means the strength spread is compressed between the rating and the simulated game. G9 (realised on sim) is shown for reference.\n")
    w("| outcome ~ predictor | n | real slope [95%] | sim slope [95%] | sim / real | real int | sim int | slope diff (real - sim) 95% |")
    w("|---|---:|---|---|---:|---:|---:|---|")
    for lab, df, xc, yr, ys in (("home margin ~ rating diff (non-neutral)", nn, "rdiff", "fin_margin", "s_final_margin"),
                                ("total points ~ sum of off ratings - s x def ratings (all)", gr, "x_tot", "real_total", "s_total_pts")):
        br, bsm, dd = bslope(df, xc, yr, ys)
        w(f"| {lab} | {len(df)} | {f(br[1], 3)} {ci(dd[:, 0], 3)} | {f(bsm[1], 3)} {ci(dd[:, 1], 3)} | {f(bsm[1] / br[1], 3)} | {f(br[0])} | {f(bsm[0])} | {ci(dd[:, 0] - dd[:, 1], 3)} |")
    br = ols(nn.s_final_margin.to_numpy(), nn.fin_margin.to_numpy())
    w(f"\nG9-style: realised margin on sim seed-mean margin, non-neutral, n={len(nn)}: slope {f(br[1], 3)} (1 = no over-spread). Realised margin on rating diff alone: see row 1; if the sim slope on the rating "
      "is about equal to the real slope, the rating -> margin mapping is not compressed, and the over-spread seen in G9 is not a strength-compression issue.\n")

    # per-channel slopes on the team-game level
    w("### Per-channel slopes: team-game rate on the offence's own strength and the opponent's defence quality\n")
    w("x_off = off_c(team) - s x def_c(opponent) (expected offensive strength of this team-game, rating units). Slope of each channel on x_off with team-game level OLS, real vs sim (sim = seed-mean per team-game), "
      "game-cluster bootstrap of the DIFFERENCE. Slope ratio sim/real < 1 means the sim channel responds less to strength than reality.\n")

    def add_x(T):
        T = T.copy()
        h = T.off_side == 0
        T["x_off"] = np.where(h, T.home_off_c - s * T.away_def_c, T.away_off_c - s * T.home_def_c)
        return T
    Tr2, Ts2 = add_x(Tr), add_x(Ts)
    chans = {"PPP": lambda t: t.pts / t.poss, "eFG%": lambda t: (t.fgm2 + 1.5 * t.fgm3) / (t.fga2 + t.fga3), "2P%": lambda t: t.fgm2 / t.fga2,
             "3P%": lambda t: t.fgm3 / t.fga3, "3PA share": lambda t: t.fga3 / (t.fga2 + t.fga3), "FTA/poss": lambda t: t.fta / t.poss,
             "FT%": lambda t: t.ftm / t.fta.clip(lower=0.5), "TOV/poss": lambda t: t.tov / t.poss, "OREB/poss": lambda t: t.oreb / t.poss,
             "poss/team": lambda t: t.poss, "sec/poss": lambda t: t.dur / t.poss}
    w("| channel | n team-games | real slope per rating unit | sim slope | sim / real | diff (real - sim) 95% | flag |")
    w("|---|---:|---:|---:|---:|---|---|")
    gids = Tr2.game_id.unique()
    for nm, fn in chans.items():
        yr = fn(Tr2).to_numpy()
        ys = fn(Ts2.set_index(["game_id", "off_side"]).loc[list(zip(Tr2.game_id, Tr2.off_side))].reset_index()).to_numpy()
        xs = Tr2.x_off.to_numpy()
        ok = np.isfinite(yr) & np.isfinite(ys) & np.isfinite(xs)
        xs, yr, ys, gid_ = xs[ok], yr[ok], ys[ok], Tr2.game_id.to_numpy()[ok]
        sl_r, sl_s = ols(xs, yr)[1], ols(xs, ys)[1]
        ug, inv = np.unique(gid_, return_inverse=True)
        diffs = []
        for _ in range(B // 2):
            gi = RNG.integers(0, len(ug), len(ug))
            cnt = np.bincount(gi, minlength=len(ug))
            wv = cnt[inv].astype(float)
            sw = np.sqrt(wv)
            Xw = np.c_[np.ones(len(xs)), xs] * sw[:, None]
            b1 = np.linalg.lstsq(Xw, yr * sw, rcond=None)[0][1]
            b2 = np.linalg.lstsq(Xw, ys * sw, rcond=None)[0][1]
            diffs.append(b1 - b2)
        diffs = np.array(diffs)
        sc = 100 if nm not in ("PPP", "poss/team", "sec/poss") else 1
        fl = "excludes 0" if (np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0) else ""
        w(f"| {nm} | {len(xs)} | {f(sl_r * sc, 4)}{' pp' if sc == 100 else ''} | {f(sl_s * sc, 4)} | {f(sl_s / sl_r, 2) if abs(sl_r) > 1e-9 else 'nan'} | {ci(diffs * sc, 4)} | {fl} |")
    w("")

    # ---------------- (c) lead-change & decision slopes -------------------------------------------------------
    w("## (c) Lead-change and time-of-decision slopes against the pre-game |spread| (continuous, per point of spread)\n")
    zz = gr[gr.abs_spread.notna()].copy()
    w(f"n = {len(zz)} games with a close line. Real = one realised path per game; sim = seed-mean per game (so sim noise is small). CI = game-cluster bootstrap ({B} reps); difference CI is of (real - sim).\n")
    w("| statistic | real slope [95%] | sim slope [95%] | sim / real | diff (real - sim) 95% | real mean | sim mean |")
    w("|---|---|---|---:|---|---:|---:|")
    for lab, yr, ys in (("lead changes", "r_lead_changes", "s_lead_changes"), ("ties", "r_ties", "s_ties"),
                        ("time of decision (s elapsed)", "r_t_decision", "s_t_decision"), ("largest lead", "r_largest_lead", "s_largest_lead"),
                        ("OT share", "r_ot", "s_ot")):
        xs = zz.abs_spread.to_numpy()
        a_, b_ = zz[yr].to_numpy(), zz[ys].to_numpy()
        sl_r, sl_s = ols(xs, a_)[1], ols(xs, b_)[1]
        dd = []
        m = len(zz)
        for _ in range(B):
            ix = RNG.integers(0, m, m)
            dd.append([ols(xs[ix], a_[ix])[1], ols(xs[ix], b_[ix])[1]])
        dd = np.array(dd)
        nd = 4
        w(f"| {lab} | {f(sl_r, nd)} {ci(dd[:, 0], nd)} | {f(sl_s, nd)} {ci(dd[:, 1], nd)} | {f(sl_s / sl_r, 2) if abs(sl_r) > 1e-9 else 'nan'} | {ci(dd[:, 0] - dd[:, 1], nd)} | {f(a_.mean(), 3)} | {f(b_.mean(), 3)} |")
    w("")
    w("By |spread| quintile (means):\n")
    w("| quintile | n | lead changes real | sim | t_decision real | sim | largest lead real | sim | OT share real | sim |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    zz["q"] = quint(zz.abs_spread)
    for q in range(5):
        x = zz[zz.q == q]
        w(f"| Q{q + 1} (|spread| {f(x.abs_spread.mean(), 1)}) | {len(x)} | {f(x.r_lead_changes.mean())} | {f(x.s_lead_changes.mean())} | {f(x.r_t_decision.mean(), 0)} | {f(x.s_t_decision.mean(), 0)} | "
          f"{f(x.r_largest_lead.mean())} | {f(x.s_largest_lead.mean())} | {f(x.r_ot.mean(), 3)} | {f(x.s_ot.mean(), 3)} |")
    w("")
    # variance budget around the market line (non-neutral): real residual SD vs sim between-game + within-game SD
    w("### Variance budget around the closing line by |spread| quintile (non-neutral games)"+chr(10))
    w("Real residual = final margin - (-close spread). Sim between = SD over games of (seed-mean margin - (-close spread)) (includes a little MC noise: SD/sqrt(50) of a seed mean is about 1.7); "
      "sim within = root mean over games of the across-seed variance of the final margin; sim total = sqrt(between^2 + within^2). If the sim's within-game spread is wider than the real residual SD the sim is over-dispersed per game "
      "(consistent with the G9 slope < 1: the spread is the right size but part of it carries no signal)."+chr(10))
    w("| quintile | n | real residual SD | sim between SD | sim within SD | sim total SD | sim total / real |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    zn = zz[zz.neutral == 0].copy()
    vv = sgs.groupby("game_id").final_margin.var().rename("wvar")
    zn = zn.merge(vv, left_on="game_id", right_index=True)
    zn["rr"] = zn.fin_margin - (-zn.close_spread_home)
    zn["rs"] = zn.s_final_margin - (-zn.close_spread_home)
    zn["q2"] = quint(zn.abs_spread)
    for q in range(5):
        x = zn[zn.q2 == q]
        tot = np.sqrt(x.rs.var() + x.wvar.mean())
        w(f"| Q{q + 1} | {len(x)} | {f(x.rr.std())} | {f(x.rs.std())} | {f(np.sqrt(x.wvar.mean()))} | {f(tot)} | {f(tot / x.rr.std(), 3)} |")
    w("")
    Path(a.out).write_text("\n".join(W), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
