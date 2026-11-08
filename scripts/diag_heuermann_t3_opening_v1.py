"""T3: opening fortnight. Season-week profile (week 1 = opening day + 6 days) of fouls, FTA, pace, turnovers, 3PA share and
rotation measures, normalised to the same season's Jan-Feb level, for 2021-22..2024-25 (labelled 2022..2025). Box data
(player_game_v2) only; regulation-length team-games (total minutes 195-205). Bootstrap SE over games. Diagnostics only."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from diag_heuermann_common_v1 import *

B = 300
rng = np.random.default_rng(20261108)
# RAW hoopR player box (includes non-D1 opponents; the truth-verified player_game_v2 holds D1-vs-D1 games only)
ren = {"field_goals_attempted": "fga", "free_throws_attempted": "fta", "offensive_rebounds": "orb", "turnovers": "tov", "fouls": "pf",
       "three_point_field_goals_attempted": "fg3a", "opponent_team_id": "opp_team_id"}
parts = []
for s_ in SEASONS:
    r = pd.read_parquet(ROOT / f"data/raw/hoopr/player_box/player_box_{s_}.parquet",
                        columns=["game_id", "season", "athlete_id", "team_id", "opponent_team_id", "minutes", "starter",
                                 "field_goals_attempted", "free_throws_attempted", "offensive_rebounds", "turnovers", "fouls",
                                 "three_point_field_goals_attempted"])
    parts.append(r.rename(columns=ren))
pg = pd.concat(parts).drop_duplicates(["game_id", "athlete_id", "team_id"])
pg = pg[pg.season.isin(SEASONS)].copy()
pg["starter"] = pg.starter.fillna(False).astype(bool)
pg["game_id"] = pg.game_id.astype("int64")
pg["minutes"] = pg.minutes.fillna(0.0)
for c in ("fga", "fta", "orb", "tov", "pf", "fg3a"):
    pg[c] = pg[c].fillna(0.0)
gu = load_universe()
gu = gu[gu.season_type == 2]
pg = pg.merge(gu[["game_id", "date", "neutral_site", "home_team_id", "is_d1_game"]], on="game_id", how="inner")

# D1 team set per season
d1 = gu[gu.is_d1_game]
d1set = {s: set(d1[d1.season == s].home_team_id) | set(d1[d1.season == s].away_team_id) for s in SEASONS}

# new-to-team: (athlete, team) with minutes>0 in an earlier season
played = pg[pg.minutes > 0][["season", "athlete_id", "team_id"]].drop_duplicates()
first_season = played.groupby(["athlete_id", "team_id"]).season.min().rename("first_season")
pg = pg.join(first_season, on=["athlete_id", "team_id"])
pg["is_new"] = (pg.season == pg.first_season) & (pg.season > min(SEASONS))      # 2022 has no baseline -> handled below

# team-game aggregates
def agg_team(g):
    m = g.minutes.values
    order = np.sort(m)[::-1]
    tot = m.sum()
    return pd.Series(dict(tot_min=tot, fga=g.fga.sum(), fta=g.fta.sum(), orb=g.orb.sum(), tov=g.tov.sum(), pf=g.pf.sum(), fg3a=g.fg3a.sum(),
                          n10=(m >= 10).sum(), top5=order[:5].sum() / tot if tot > 0 else np.nan, top7=order[:7].sum() / tot if tot > 0 else np.nan,
                          bench=1 - g.loc[g.starter.astype(bool), "minutes"].sum() / tot if tot > 0 else np.nan,
                          new=(g.minutes * g.is_new).sum() / tot if tot > 0 else np.nan))

tg = pg.groupby(["season", "game_id", "team_id", "opp_team_id", "date", "neutral_site", "home_team_id"]).apply(agg_team, include_groups=False).reset_index()
tg = tg[tg.tot_min.between(195, 205)].copy()
tg["poss"] = tg.fga - tg.orb + tg.tov + 0.44 * tg.fta
tg = tg[(tg.poss > 40) & (tg.poss < 110)]
o = tg[["game_id", "team_id", "poss"]].rename(columns={"team_id": "opp_team_id", "poss": "opp_poss"})
tg = tg.merge(o, on=["game_id", "opp_team_id"], how="inner")          # keeps rows whose opponent row is also valid
tg["team_d1"] = [t in d1set[s] for s, t in zip(tg.season, tg.team_id)]
tg["opp_d1"] = [t in d1set[s] for s, t in zip(tg.season, tg.opp_team_id)]
tg = tg[tg.team_d1].copy()
tg["opp_class"] = np.where(tg.opp_d1, "d1", "nonD1")

# pre-season rating gap (first snapshot of the season, carries prior-season rating; 2022 snapshot is all zero -> unavailable)
gap = {}
for s in (2023, 2024, 2025):
    r = pd.read_parquet(P / f"ratings/own_ratings_{s}.parquet", columns=["as_of_date", "team_id", "off_c", "def_c"])
    r = r[r.as_of_date == r.as_of_date.min()]
    gap[s] = (r.off_c - r.def_c).set_axis(r.team_id)
tg["net"] = [gap[s].get(t, np.nan) if s in gap else np.nan for s, t in zip(tg.season, tg.team_id)]
tg["net_opp"] = [gap[s].get(t, np.nan) if s in gap else np.nan for s, t in zip(tg.season, tg.opp_team_id)]
tg["abs_gap"] = (tg.net - tg.net_opp).abs()
tg["gap_class"] = np.where(tg.opp_class == "nonD1", "nonD1 opponent",
                           np.where(tg.abs_gap.isna(), "D1 (gap unavailable)",
                                    np.where(tg.abs_gap < 5, "D1 close (<5)", np.where(tg.abs_gap < 12, "D1 mid (5-12)", "D1 big gap (>=12)"))))
tg["site"] = np.where(tg.neutral_site, "neutral", "home/away")

# calendar
d0 = {s: tg[tg.season == s].date.min() for s in SEASONS}
print("d0", d0)
tg["day"] = [(dt - d0[s]).days for dt, s in zip(tg.date, tg.season)]
tg["week"] = np.where(tg.day >= 0, tg.day // 7 + 1, 0)
tg["jf"] = tg.date.dt.month.isin([1, 2])

# metrics as (num, den) per team-row
tg["m_fouls_per_poss"], tg["d_fouls_per_poss"] = tg.pf, tg.opp_poss
tg["m_fta_per_poss"], tg["d_fta_per_poss"] = tg.fta, tg.poss
tg["m_poss_per_team_game"], tg["d_poss_per_team_game"] = tg.poss, 1.0
tg["m_fouls_per_team_game"], tg["d_fouls_per_team_game"] = tg.pf, 1.0
tg["m_tov_per_poss"], tg["d_tov_per_poss"] = tg.tov, tg.poss
tg["m_fg3a_share"], tg["d_fg3a_share"] = tg.fg3a, tg.fga
for k in ("n10", "top5", "top7", "bench", "new"):
    tg["m_" + k], tg["d_" + k] = tg[k], 1.0
METRICS3 = ["fouls_per_poss", "fouls_per_team_game", "fta_per_poss", "poss_per_team_game", "tov_per_poss", "fg3a_share", "n10", "top5", "top7", "bench", "new"]
LAB = {"n10": "players_with_10plus_min", "top5": "top5_minutes_share", "top7": "top7_minutes_share", "bench": "bench_minutes_share",
       "new": "new_to_team_minutes_share"}


def cell_stat(df, met):
    """ratio-of-sums level and bootstrap SE (resampling games)."""
    gs = df.groupby("game_id")[["m_" + met, "d_" + met]].sum().values
    n = gs[:, 0].sum() / gs[:, 1].sum()
    if len(gs) < 5:
        return n, np.nan, len(gs)
    idx = rng.integers(0, len(gs), size=(B, len(gs)))
    bs = gs[idx, 0].sum(1) / gs[idx, 1].sum(1)
    return n, bs.std(ddof=1), len(gs)


def ratio(a, b):
    (la, sa, na), (lb, sb, nb) = a, b
    r = la / lb
    return r, r * np.sqrt((sa / la) ** 2 + (sb / lb) ** 2) if la and lb else np.nan


rows = []
CELLS = [("d0", lambda d: d.day == 0), ("wk1 (d0-6)", lambda d: d.week == 1), ("wk2 (d7-13)", lambda d: d.week == 2), ("wk3", lambda d: d.week == 3),
         ("wk4", lambda d: d.week == 4), ("wk5", lambda d: d.week == 5), ("wk6", lambda d: d.week == 6), ("wk8", lambda d: d.week == 8),
         ("wk1-2 (d0-13)", lambda d: d.week.isin([1, 2])), ("d14+ to Dec31", lambda d: (d.week >= 3) & (d.date.dt.month >= 11) & (d.date.dt.year == d.date.dt.year) & (d.date.dt.month != 1) & (d.date.dt.month != 2) & (d.date.dt.month != 3)),
         ("Nov 1-14 (calendar)", lambda d: (d.date.dt.month == 11) & (d.date.dt.day <= 14)),
         ("Nov 5-8 (calendar)", lambda d: (d.date.dt.month == 11) & d.date.dt.day.between(5, 8)),
         ("Jan-Feb", lambda d: d.jf)]
for scope, sdf in [("D1 team rows vs D1 opponents", tg[tg.opp_class == "d1"]), ("D1 team rows, any opponent", tg)]:
    for s in SEASONS:
        ds = sdf[sdf.season == s]
        for met in METRICS3:
            if met == "new" and s == 2022:
                continue
            base = cell_stat(ds[ds.jf], met)
            allb = cell_stat(ds, met)
            for lab, f in CELLS:
                c = ds[f(ds)]
                if len(c) == 0:
                    continue
                st = cell_stat(c, met)
                r, rse = ratio(st, base)
                r2, rse2 = ratio(st, allb)
                rows.append(dict(scope=scope, season=s, metric=LAB.get(met, met), cell=lab, n_games=st[2], level=st[0], se_level=st[1],
                                 jan_feb_level=base[0], ratio_to_jan_feb=r, se_ratio=rse, ratio_to_season_all_games=r2, se_ratio_all=rse2,
                                 underpowered=st[2] < 100))
wk = pd.DataFrame(rows)
# pooled over seasons (game-count-weighted ratios)
pool = []
for (scope, met, cell), g in wk.groupby(["scope", "metric", "cell"]):
    w = g.n_games.values.astype(float)
    pool.append(dict(scope=scope, season="pooled", metric=met, cell=cell, n_games=int(w.sum()), level=np.average(g.level, weights=w),
                     jan_feb_level=np.average(g.jan_feb_level, weights=w), ratio_to_jan_feb=np.average(g.ratio_to_jan_feb, weights=w),
                     se_ratio=np.sqrt(((w / w.sum()) ** 2 * g.se_ratio.values ** 2).sum()), ratio_to_season_all_games=np.average(g.ratio_to_season_all_games, weights=w),
                     se_ratio_all=np.sqrt(((w / w.sum()) ** 2 * g.se_ratio_all.values ** 2).sum()), underpowered=w.sum() < 100))
wk = pd.concat([wk, pd.DataFrame(pool)], ignore_index=True)
save_csv(wk, "data_t3_weekly_profile.csv")

# splits for wk1-2 (d0-13) and for d0 alone
sp = []
for scope_name, sdf in [("D1 team rows, any opponent", tg)]:
    for split_col in ("gap_class", "site"):
        for s in SEASONS + ["pooled"]:
            ds = sdf if s == "pooled" else sdf[sdf.season == s]
            if split_col == "site":
                ds = ds[ds.opp_class == "d1"]            # site split on D1-vs-D1 games only (non-D1 games are their own class above)
            for met in METRICS3:
                if met == "new" and (s == 2022):
                    continue
                dsm = ds[ds.season > 2022] if (met == "new" and s == "pooled") else ds
                # baseline: Jan-Feb overall level of the same season(s)
                jb = np.mean([cell_stat(dsm[(dsm.season == ss) & dsm.jf & (dsm.opp_class == "d1")], met)[0] for ss in dsm.season.unique()])   # D1-vs-D1 Jan-Feb level
                for win_lab, win in (("wk1-2 (d0-13)", dsm.week.isin([1, 2])), ("d0", dsm.day == 0)):
                    for cls, c in dsm[win].groupby(split_col):
                        if len(c) < 20:
                            continue
                        if s == "pooled":
                            per = [cell_stat(c[c.season == ss], met) for ss in c.season.unique() if (c.season == ss).sum() >= 5]
                            w = np.array([p_[2] for p_ in per], float); lv = np.average([p_[0] for p_ in per], weights=w)
                            se = np.sqrt(((w / w.sum()) ** 2 * np.array([p_[1] ** 2 for p_ in per])).sum()); ng = int(w.sum())
                        else:
                            lv, se, ng = cell_stat(c, met)
                        # same-class Jan-Feb level (for context)
                        cj = dsm[dsm.jf & (dsm[split_col] == cls)]
                        jc = np.nan if len(cj) < 50 else np.mean([cell_stat(cj[cj.season == ss], met)[0] for ss in cj.season.unique() if (cj.season == ss).sum() >= 5])
                        sp.append(dict(split=split_col, cls=cls, season=s, window=win_lab, metric=LAB.get(met, met), n_games=ng, level=lv, se_level=se,
                                       jan_feb_overall_level=jb, ratio_to_jan_feb_overall=lv / jb, se_ratio=se / jb,
                                       jan_feb_same_class_level=jc, n_games_jan_feb_same_class=cj.game_id.nunique(), ratio_to_jan_feb_same_class=lv / jc if jc == jc else np.nan,
                                       underpowered=ng < 100))
sp = pd.DataFrame(sp)
save_csv(sp, "data_t3_wk12_splits.csv")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
p1 = wk[(wk.season == "pooled") & (wk.scope == "D1 team rows vs D1 opponents")]
print(p1.pivot(index="cell", columns="metric", values="ratio_to_jan_feb").to_string())
print(wk[(wk.season != "pooled") & (wk.scope == "D1 team rows vs D1 opponents") & (wk.metric == "fouls_per_poss")].pivot(index="cell", columns="season", values="ratio_to_jan_feb").to_string())
print(sp[(sp.season == "pooled") & (sp.window == "wk1-2 (d0-13)")].pivot_table(index=["split", "cls"], columns="metric", values="ratio_to_jan_feb_overall").to_string())
print(sp[(sp.season == "pooled") & (sp.window == "wk1-2 (d0-13)") & (sp.metric == "poss_per_team_game")].to_string())
