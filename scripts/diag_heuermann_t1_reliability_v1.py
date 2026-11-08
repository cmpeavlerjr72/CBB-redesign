"""T1: rating reliability / attenuation. Realised team-game rate (centred) regressed on raw vs shrunk as-of rating.
Test seasons 2024 (fold 1) and 2025 (fold 2). k estimated on PRIOR seasons only. Diagnostics only."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from diag_heuermann_common_v1 import *

t = load_team_games()
rows_slope, rows_k, rows_rel, rows_q = [], [], [], []
ARMS = [("raw_asof", "xraw"), ("shrunk_asof", "xsh"), ("raw_LOO_hindsight", "xloo_raw"), ("shrunk_LOO_hindsight", "xloo_sh")]
frames = {}
for m, (num, den) in METRICS.items():
    # league final rate by season (fallback for day-0 league mean)
    fin = t.groupby("season")[[num, den]].sum()
    mu_final = (fin[num] / fin[den])
    prior_mu = {s: mu_final.get(s - 1, np.nan) for s in SEASONS}
    parts = []
    for test in (2024, 2025):
        ks = estimate_k(t, num, den, [s for s in SEASONS if s < test])
        rows_k.append(dict(metric=m, test_season=test, prior_seasons="+".join(map(str, ks["seasons"])), k_attempts=ks["k"],
                           within_team_s2=ks["s2"], between_team_tau2=ks["tau2"],
                           reliab_at_50_att=50 / (50 + ks["k"]), reliab_at_200_att=200 / (200 + ks["k"]),
                           reliab_at_500_att=500 / (500 + ks["k"])))
        d = asof_rates(t[t.season == test], num, den, k=ks["k"], prior_mu_by_season=prior_mu)
        d["metric"] = m
        parts.append(d)
    d = pd.concat(parts)
    d = d[(d.nprior >= 1) & d.yc.notna() & d.xraw.notna() & d.xsh.notna() & (d.w > 0)].copy()
    d["tsid"] = d.season.astype(str) + "_" + d.team_id.astype(str)
    d["gb"] = pd.cut(d.nprior, [0, 5, 15, 999], labels=["1-5", "6-15", "16+"])
    q = d.cden.quantile([1 / 3, 2 / 3]).values
    d["ab"] = pd.cut(d.cden, [-1, q[0], q[1], 1e9], labels=[f"low(<={q[0]:.0f})", f"mid(<={q[1]:.0f})", f"high(>{q[1]:.0f})"])
    frames[m] = d
    for sname, sub in [("2024", d[d.season == 2024]), ("2025", d[d.season == 2025]), ("pooled", d)]:
        for bname, bcol in [("all", None), ("games", "gb"), ("attempts", "ab")]:
            cells = [("all", sub)] if bcol is None else [(str(l), sub[sub[bcol] == l]) for l in sub[bcol].cat.categories]
            for lab, c in cells:
                for arm, xc in ARMS:
                    cc = c[c[xc].notna()]
                    if len(cc) < 200:
                        rows_slope.append(dict(metric=m, season=sname, bucket_type=bname, bucket=lab, arm=arm, n=len(cc), slope=np.nan, se=np.nan))
                        continue
                    b, se, a = wls_slope(cc.yc, cc[xc], cc.w, cc.tsid)
                    rows_slope.append(dict(metric=m, season=sname, bucket_type=bname, bucket=lab, arm=arm, n=len(cc),
                                           mean_att=cc.w.mean(), sd_x=cc[xc].std(), slope=b, se=se,
                                           underpowered=len(cc) < 1500))
    # quintiles (nprior >= 6)
    dq = d[d.nprior >= 6]
    for sname, sub in [("2024", dq[dq.season == 2024]), ("2025", dq[dq.season == 2025]), ("pooled", dq)]:
        for arm, xc in ARMS[:2]:
            sub2 = sub.copy()
            sub2["Q"] = pd.qcut(sub2[xc], 5, labels=[1, 2, 3, 4, 5])
            for qq, c in sub2.groupby("Q", observed=True):
                wy = np.average(c.yc, weights=c.w)
                # SE of weighted mean via team-season cluster of weighted residuals
                rr = (c.w * (c.yc - wy)).groupby(c.tsid).sum()
                se = np.sqrt((rr ** 2).sum()) / c.w.sum()
                rows_q.append(dict(metric=m, season=sname, quintile_by=arm, quintile=int(qq), n=len(c), mean_att=c.w.mean(),
                                   mean_rating=c[xc].mean(), mean_raw_rating=c.xraw.mean(), mean_shrunk_rating=c.xsh.mean(),
                                   realised_centred=wy, se=se))
    # split-half reliability, per full season 2022-2025
    for s in SEASONS:
        ts = t[(t.season == s) & (t[den] > 0)].copy()
        ts["half"] = ts.gnum % 2
        a = ts.groupby(["team_id", "half"])[[num, den]].sum().unstack()
        a = a.dropna()
        ok = (ts.groupby("team_id").size() >= 20).reindex(a.index).fillna(False)
        a = a[ok.values]
        r0, r1 = a[(num, 0)] / a[(den, 0)], a[(num, 1)] / a[(den, 1)]
        r = np.corrcoef(r0, r1)[0, 1]
        rows_rel.append(dict(metric=m, season=s, n_teams=len(a), mean_att_per_half=float((a[(den, 0)] + a[(den, 1)]).mean() / 2),
                             split_half_r=r, spearman_brown_full_season=2 * r / (1 + r)))
slopes = pd.DataFrame(rows_slope); ks = pd.DataFrame(rows_k); rel = pd.DataFrame(rows_rel); qd = pd.DataFrame(rows_q)
# quintile-mean slope of realised on rating
qs = []
for (m, s, qb), g in qd.groupby(["metric", "season", "quintile_by"]):
    A = np.polyfit(g.mean_rating, g.realised_centred, 1)
    qs.append(dict(metric=m, season=s, quintile_by=qb, quintile_mean_slope=A[0],
                   q5_minus_q1_realised=g.realised_centred.iloc[-1] - g.realised_centred.iloc[0],
                   q5_minus_q1_rating=g.mean_rating.iloc[-1] - g.mean_rating.iloc[0]))
qs = pd.DataFrame(qs)
save_csv(slopes, "data_t1_slopes.csv"); save_csv(ks, "data_t1_prior_strength.csv"); save_csv(rel, "data_t1_reliability.csv")
save_csv(qd, "data_t1_quintiles.csv"); save_csv(qs, "data_t1_quintile_slopes.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(ks.to_string()); print(rel.to_string())
print(slopes[slopes.bucket_type == "all"].to_string())
print(slopes[slopes.bucket_type != "all"][slopes.season == "pooled"].to_string())
print(qs.to_string())
