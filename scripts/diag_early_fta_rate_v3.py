"""Early-season FTA/P diagnostic part 3 (DIAGNOSTIC ONLY): team responsiveness of FTA/P by days bucket, and the
two-step counterfactual for the first-half bonus-trip gap. 2025-26 never read. Writes results/early_fta_diag/report3.txt.
"""
from __future__ import annotations
import io
import numpy as np, pandas as pd
from pathlib import Path
from cbb_sim.eval import reference as R

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/early_fta_diag"
buf = io.StringIO()


def say(*a):
    print(*a); print(*a, file=buf)


def tab(df, **kw):
    say(df.to_string(**kw)); say("")


SIMS = {2024: ROOT / "results/engine_v0/f1c_V2_full_s200_o0/games.parquet",
        2025: ROOT / "results/engine_v0/v3full_COMB9GCTKD_s200_o0/games.parquet"}
BK = [-1, 14, 45, 10_000]; BL = ["d0-14", "d15-45", "d46+"]


def team_box(season):
    tb = R.load_actual_team_box(season)[["game_id", "team_id", "fga", "oreb", "tov", "fta"]].copy()
    tb["P"] = tb.fga - tb.oreb + tb.tov + 0.44 * tb.fta
    return tb


say("== G. TEAM RESPONSIVENESS of FTA/P to the team's PRIOR-season FTA/P, by days bucket (teams with >= 3 games in the bucket) ==")
say("slope = OLS slope of the team's bucket FTA/P on the prior (1.0 = full pass-through of the prior); actual vs sim; gap-slope = slope of (sim - actual).")
rows = []
for sea in (2025, 2024):
    g = R.load_actual_games(sea).copy()
    g["dss"] = (pd.to_datetime(g.game_date) - pd.to_datetime(g.game_date).min()).dt.days
    g["b"] = pd.cut(g.dss, BK, labels=BL).astype(str)
    tb = team_box(sea).merge(g[["game_id", "b", "home_team_id", "away_team_id"]], on="game_id")
    pr = team_box(sea - 1).groupby("team_id")[["fta", "P"]].sum(); prior = pr.fta / pr.P
    s = pd.read_parquet(SIMS[sea])
    for side in ("home", "away"):
        s[f"{side}_P"] = s[f"{side}_fga3"] + s[f"{side}_fga2_rim"] + s[f"{side}_fga2_jump"] - s[f"{side}_oreb"] + s[f"{side}_tov"] + 0.44 * s[f"{side}_fta"]
    sm = s.groupby("game_id")[["home_fta", "home_P", "away_fta", "away_P"]].mean().reset_index().merge(g[["game_id", "b", "home_team_id", "away_team_id"]], on="game_id")
    tb = tb[tb.game_id.isin(sm.game_id)]
    for b in BL:
        ta = tb[tb.b == b].groupby("team_id").agg(fta=("fta", "sum"), P=("P", "sum"), n=("fta", "size"))
        x = sm[sm.b == b]
        h = x.groupby("home_team_id")[["home_fta", "home_P"]].sum().rename(columns={"home_fta": "fta", "home_P": "P"})
        a = x.groupby("away_team_id")[["away_fta", "away_P"]].sum().rename(columns={"away_fta": "fta", "away_P": "P"})
        ts = h.add(a, fill_value=0)
        t = pd.DataFrame({"act": ta.fta / ta.P, "sim": ts.fta / ts.P, "n": ta.n}).dropna(); t = t[t.n >= 3]
        t["prior"] = prior.reindex(t.index); t = t.dropna()
        t["gap"] = t.sim - t.act
        sl_a = np.polyfit(t.prior, t.act, 1)[0]; sl_s = np.polyfit(t.prior, t.sim, 1)[0]
        rows.append(dict(season=sea, bucket=b, teams=len(t), mean_games=t.n.mean(), slope_actual=sl_a, slope_sim=sl_s, slope_ratio=sl_s / sl_a,
                         corr_actual=np.corrcoef(t.prior, t.act)[0, 1], corr_sim=np.corrcoef(t.prior, t.sim)[0, 1],
                         gap_mean=t.gap.mean(), gap_slope=np.polyfit(t.prior, t.gap, 1)[0], gap_share_neg=(t.gap < 0).mean()))
tab(pd.DataFrame(rows).round(3), index=False)

say("== H. H1 bonus-trip gap, two-factor split (F2 window): bonus trips per possession = in-bonus share x trips per in-bonus possession ==")
PO = ROOT / "data/processed/models/possession_outcome"
cp = pd.read_parquet(PO / "round6/foul_accrual_poss_v2.parquet", columns=["game_id", "season", "period", "poss_index", "off_in_bonus", "days_since_start"])
cp = cp[(cp.season == 2025) & (cp.period <= 2)]
ch = pd.read_parquet(ROOT / "data/processed/possessions_v2/chances_2025.parquet", columns=["game_id", "period", "poss_index", "terminal_event"])
ch = ch[ch.terminal_event == "FT_trip_bonus"].groupby(["game_id", "period", "poss_index"]).size().rename("nb").reset_index()
cp = cp.merge(ch, on=["game_id", "period", "poss_index"], how="left").fillna({"nb": 0})
cp["b"] = pd.cut(cp.days_since_start, BK, labels=BL).astype(str)
shards = sorted((ROOT / "results/engine_v0").glob("r9tapfull_COMB9_s200_o0_off*_n25"))
ha = pd.concat([pd.read_parquet(s / "half_agg.parquet") for s in shards]); ha = ha[ha.half <= 2]
gm = cp.groupby("game_id").days_since_start.first().rename("dss").reset_index()
ha = ha.merge(gm, on="game_id"); ha["b"] = pd.cut(ha.dss, BK, labels=BL).astype(str)
rows = []
for h in (1, 2):
    for b in BL:
        a = cp[(cp.period == h) & (cp.b == b)]; s = ha[(ha.half == h) & (ha.b == b)]
        ia, isim = a.off_in_bonus.mean(), s.in_bonus.sum() / s.poss.sum()
        ra, rs = a.nb.sum() / a.off_in_bonus.sum(), s.n_bonus_trip.sum() / s.in_bonus.sum()
        rows.append(dict(half=h, bucket=b, inbonus_act=ia, inbonus_sim=isim, trips_per_inbonus_act=ra, trips_per_inbonus_sim=rs,
                         bonus_p_act=ia * ra, bonus_p_sim=isim * rs,
                         gap_from_state=(isim - ia) * ra, gap_from_trip_rate=isim * (rs - ra)))
tab(pd.DataFrame(rows).round(4), index=False)
(OUT / "report3.txt").write_text(buf.getvalue(), encoding="utf-8")
