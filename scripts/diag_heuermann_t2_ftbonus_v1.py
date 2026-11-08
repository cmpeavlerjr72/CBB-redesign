"""T2: FT drawing and bonus timing. Quintiles of the offence's as-of FT-drawing rating (FTA/FGA, shrunk) and of the
opponent defence's as-of foul-conceding rate. Seasons 2024 and 2025 (k from prior seasons only). Diagnostics only."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from diag_heuermann_common_v1 import *

MINPRIOR = 8
t = load_team_games()
fin = t.groupby("season")[["box_fta", "box_fga"]].sum()
mu_final = fin.box_fta / fin.box_fga
prior_mu = {s: mu_final.get(s - 1, np.nan) for s in SEASONS}
offp, defp = [], []
for test in (2024, 2025):
    prior = [s for s in SEASONS if s < test]
    ko = estimate_k(t, "box_fta", "box_fga", prior)["k"]
    kd = estimate_k(t, "opp_fta", "opp_fga", prior)["k"]
    print(test, "k_off", ko, "k_def", kd)
    offp.append(asof_rates(t[t.season == test], "box_fta", "box_fga", k=ko, prior_mu_by_season=prior_mu))
    defp.append(asof_rates(t[t.season == test], "opp_fta", "opp_fga", k=kd, prior_mu_by_season=prior_mu))
off = pd.concat(offp)[["season", "game_id", "team_id", "nprior", "xsh"]].rename(
    columns={"xsh": "off_rt", "nprior": "off_np", "team_id": "offense_team_id"})
dfn = pd.concat(defp)[["season", "game_id", "team_id", "nprior", "xsh"]].rename(
    columns={"xsh": "def_rt", "nprior": "def_np", "team_id": "defense_team_id"})

ps = []
for s in (2024, 2025):
    p = pd.read_parquet(P / f"possessions_v2/possessions_{s}.parquet",
                        columns=["game_id", "season", "period", "offense_team_id", "defense_team_id", "start_clock", "fga_rim",
                                 "fga_jump2", "fga_3", "fta", "def_team_fouls", "off_in_bonus", "off_in_double_bonus"])
    ps.append(p[p.period <= 2])
p = pd.concat(ps)
assert p.season.isin([2024, 2025]).all()
p["fga"] = p.fga_rim + p.fga_jump2 + p.fga_3
p["state"] = np.where(p.off_in_double_bonus, "double", np.where(p.off_in_bonus, "1and1", "none"))
p = p.merge(off, on=["season", "game_id", "offense_team_id"], how="left").merge(dfn, on=["season", "game_id", "defense_team_id"], how="left")
p = p[(p.off_np >= MINPRIOR) & (p.def_np >= MINPRIOR) & p.off_rt.notna() & p.def_rt.notna()].copy()
p["half"] = np.where(p.period == 1, "H1", "H2")
p["tsid_o"] = p.season.astype(str) + "_" + p.offense_team_id.astype(str)
print("poss", len(p), "games", p.game_id.nunique())


def qmap(df, key, col):
    u = df[["season", "game_id", key, col]].drop_duplicates(["season", "game_id", key])
    u["Q"] = pd.qcut(u[col], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return u[["season", "game_id", key, "Q"]]


qo = qmap(p, "offense_team_id", "off_rt").rename(columns={"Q": "Q_off"})
qd = qmap(p, "defense_team_id", "def_rt").rename(columns={"Q": "Q_def"})
p = p.merge(qo, on=["season", "game_id", "offense_team_id"]).merge(qd, on=["season", "game_id", "defense_team_id"])
VIEWS = [("offence_draw_rating", "Q_off", "off_rt"), ("defence_foul_rating", "Q_def", "def_rt")]

R_share, R_fta, R_fga, R_time, R_slope = [], [], [], [], []
for view, Q, rc in VIEWS:
    for half in ("H1", "H2", "both"):
        ph = p if half == "both" else p[p.half == half]
        for q, g in ph.groupby(Q):
            R_share.append(dict(view=view, quintile=q, half=half, n_poss=len(g), mean_rating=g[rc].mean(),
                                share_none=(g.state == "none").mean(), share_1and1=(g.state == "1and1").mean(),
                                share_double=(g.state == "double").mean(), fta_per_poss=g.fta.mean()))
            for st, gs in g.groupby("state"):
                R_fta.append(dict(view=view, quintile=q, half=half, state=st, n_poss=len(gs), fta_per_poss=gs.fta.mean(),
                                  se=gs.fta.std() / np.sqrt(len(gs)), underpowered=len(gs) < 2000))
    tgq = p.groupby(["season", "game_id", "offense_team_id", Q]).agg(fta=("fta", "sum"), fga=("fga", "sum"), rt=(rc, "first")).reset_index()
    for q, g in tgq.groupby(Q):
        R_fga.append(dict(view=view, quintile=q, n_team_games=len(g), mean_rating=g.rt.mean(), fta_per_fga=g.fta.sum() / g.fga.sum()))
    for half in ("H1", "H2", "both"):
        ph = p if half == "both" else p[p.half == half]
        sd = ph.drop_duplicates(["season", "game_id", "offense_team_id"])[rc].std()
        for lab, y in [("share_none", (ph.state == "none")), ("share_1and1", (ph.state == "1and1")),
                       ("share_double", (ph.state == "double")), ("fta_per_poss", ph.fta)]:
            b, se, _ = wls_slope(y.astype(float), ph[rc], np.ones(len(ph)), ph.tsid_o)
            R_slope.append(dict(view=view, half=half, outcome=lab, state="all", n=len(ph), slope_per_unit=b, se=se,
                                sd_rating=sd, effect_per_sd=b * sd))
        for st in ("none", "1and1", "double"):
            gs = ph[ph.state == st]
            b, se, _ = wls_slope(gs.fta.astype(float), gs[rc], np.ones(len(gs)), gs.tsid_o)
            R_slope.append(dict(view=view, half=half, outcome="fta_per_poss", state=st, n=len(gs), slope_per_unit=b, se=se,
                                sd_rating=sd, effect_per_sd=b * sd))
    tg = p.groupby(["season", "game_id", "offense_team_id"]).agg(fta=("fta", "sum"), fga=("fga", "sum"), rt=(rc, "first"),
                                                                 tsid=("tsid_o", "first")).reset_index()
    b, se, _ = wls_slope(tg.fta / tg.fga, tg.rt, tg.fga, tg.tsid)
    R_slope.append(dict(view=view, half="both", outcome="fta_per_fga", state="all", n=len(tg), slope_per_unit=b, se=se,
                        sd_rating=tg.rt.std(), effect_per_sd=b * tg.rt.std()))

# time-to-threshold per (game, half, DEFENDING team)
dh = p.groupby(["season", "game_id", "period", "defense_team_id"]).agg(
    Q_def=("Q_def", "first"), Q_off=("Q_off", "first"), def_rt=("def_rt", "first"), off_rt=("off_rt", "first"),
    tsid=("tsid_o", "first")).reset_index()
for thr in (6, 7, 9, 10):
    r = p[p.def_team_fouls >= thr].groupby(["season", "game_id", "period", "defense_team_id"]).start_clock.max().rename("clk")
    dh = dh.join(r, on=["season", "game_id", "period", "defense_team_id"])
    dh[f"el{thr}"] = (1200 - dh.clk) / 60.0
    dh = dh.drop(columns="clk")
for view, Q, rc in (("offence_draw_rating", "Q_off", "off_rt"), ("defence_foul_rating", "Q_def", "def_rt")):
    for per, hn in ((1, "H1"), (2, "H2")):
        d = dh[dh.period == per]
        for thr in (6, 7, 9, 10):
            col = f"el{thr}"
            for q, g in d.groupby(Q):
                reached = g[col].notna()
                R_time.append(dict(view=view, half=hn, fouls_threshold=thr, quintile=int(q), n_def_halves=len(g),
                                   reach_share=reached.mean(), mean_elapsed_min_if_reached=g.loc[reached, col].mean(),
                                   restricted_mean_min_cap20=g[col].fillna(20.0).mean(),
                                   se_restricted=g[col].fillna(20.0).std() / np.sqrt(len(g))))
            sdr = d[rc].std()
            b, se, _ = wls_slope(d[col].fillna(20.0), d[rc], np.ones(len(d)), d.tsid)
            R_slope.append(dict(view=view, half=hn, outcome=f"restricted_mean_min_to_{thr}_fouls", state="all", n=len(d),
                                slope_per_unit=b, se=se, sd_rating=sdr, effect_per_sd=b * sdr))
            b2, se2, _ = wls_slope(d[col].notna().astype(float), d[rc], np.ones(len(d)), d.tsid)
            R_slope.append(dict(view=view, half=hn, outcome=f"reach_share_{thr}_fouls", state="all", n=len(d),
                                slope_per_unit=b2, se=se2, sd_rating=sdr, effect_per_sd=b2 * sdr))
sh, ft, fg, tm, sl = map(pd.DataFrame, (R_share, R_fta, R_fga, R_time, R_slope))
save_csv(sh, "data_t2_bonus_state_share.csv")
save_csv(ft, "data_t2_fta_per_poss_by_state.csv")
save_csv(fg, "data_t2_fta_per_fga.csv")
save_csv(tm, "data_t2_time_to_team_fouls.csv")
save_csv(sl, "data_t2_slopes.csv")
# shift-share decomposition of the Q5-Q1 difference in FTA per possession: bonus TIMING (state shares) vs PER-STATE rate
dec = []
for view, _, _ in VIEWS:
    for half in ("H1", "H2", "both"):
        s = sh[(sh.view == view) & (sh.half == half)].set_index("quintile")
        f = ft[(ft.view == view) & (ft.half == half)].pivot(index="quintile", columns="state", values="fta_per_poss")
        st = {"none": "share_none", "1and1": "share_1and1", "double": "share_double"}
        timing = sum((s.loc[5, st[k]] - s.loc[1, st[k]]) * (f.loc[5, k] + f.loc[1, k]) / 2 for k in st)
        rate = sum((s.loc[5, st[k]] + s.loc[1, st[k]]) / 2 * (f.loc[5, k] - f.loc[1, k]) for k in st)
        tot = s.loc[5, "fta_per_poss"] - s.loc[1, "fta_per_poss"]
        dec.append(dict(view=view, half=half, q5_minus_q1_fta_per_poss=tot, timing_component=timing, per_state_rate_component=rate,
                        timing_share=timing / tot, check_sum=timing + rate))
save_csv(pd.DataFrame(dec), "data_t2_timing_vs_state_decomposition.csv")
print(pd.DataFrame(dec).to_string())
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(sh[sh.half == "both"].to_string())
print(fg.to_string())
print(tm[(tm.fouls_threshold.isin([7, 10])) & (tm.view == "offence_draw_rating")].to_string())
print(sl[sl.half == "both"].to_string())
print(ft[(ft.half == "both") & (ft.view == "offence_draw_rating")].to_string())
