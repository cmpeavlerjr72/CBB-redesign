"""Early-season FTA/P diagnostic (DIAGNOSTIC ONLY, nothing fitted or adopted).
Doc: docs/tests/early_fta_rate_diag_2026-10-07.md.  Run: .venv/Scripts/python.exe scripts/diag_early_fta_rate_v1.py
Writes results/early_fta_diag/report.txt (gitignored) with every table in the doc. 2025-26 never read.

Truth: verified finals + hoopR team box (FTA, FGA, OREB, TOV). P = FGA - OREB + TOV + 0.44 FTA (box possessions, as in
diag_total_bias_decomp_v1). Sims: served-v2 200-seed reads (F1 f1c_V2_full_s200_o0, F2 v3full_COMB9GCTKD_s200_o0).
Source split (F2): per-half tap shards r9tapfull_COMB9_s200_o0_off*_n25 (R9ao3 foul, 8 x 25 seeds = 200) vs possessions_v2 chances.
Offline foul check: round9 preds_trip (T2c arm), preds_ao (AO3), round7 preds_poss (A2) F1/F2 seed 0.
"""
from __future__ import annotations
import io
import numpy as np, pandas as pd
from pathlib import Path
from cbb_sim.eval import reference as R
from cbb_sim.features.conference import build_conference_flags

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/early_fta_diag"; OUT.mkdir(exist_ok=True)
SIMS = {2024: ROOT / "results/engine_v0/f1c_V2_full_s200_o0/games.parquet",
        2025: ROOT / "results/engine_v0/v3full_COMB9GCTKD_s200_o0/games.parquet"}
PO = ROOT / "data/processed/models/possession_outcome"
BK = [-1, 14, 45, 10_000]; BL = ["d0-14", "d15-45", "d46+"]
FMB = [-1, 5, 12, 19, 200]; FML = ["0-5", "6-12", "13-19", "20+"]
buf = io.StringIO()


def say(*a):
    print(*a); print(*a, file=buf)


def tab(df, **kw):
    say(df.to_string(**kw)); say("")


# ---------------------------------------------------------------- data
def ctx(season):
    g = R.load_actual_games(season).copy()
    g["date"] = pd.to_datetime(g["game_date"]); g["dss"] = (g["date"] - g["date"].min()).dt.days
    g = g.merge(build_conference_flags([season])[["game_id", "is_conf_game"]], on="game_id", how="left")
    g["conf"] = np.where(g["is_conf_game"].fillna(False).astype(bool), "conf", "nonconf")
    g["site"] = np.where(g["neutral"] > 0, "neutral", "home-court")
    g["dssb"] = pd.cut(g["dss"], BK, labels=BL).astype(str); g["season"] = season
    return g


def box_actual(season, g):
    tb = R.load_actual_team_box(season)[["game_id", "team_id", "team_home_away", "fga", "oreb", "tov", "fta", "ftm"]]
    tb = tb[tb.game_id.isin(g.game_id)].copy()
    tb["P"] = tb.fga - tb.oreb + tb.tov + 0.44 * tb.fta
    return tb


def sim_games(season):
    s = pd.read_parquet(SIMS[season])
    for side in ("home", "away"):
        s[f"{side}_P"] = (s[f"{side}_fga3"] + s[f"{side}_fga2_rim"] + s[f"{side}_fga2_jump"] - s[f"{side}_oreb"]
                          + s[f"{side}_tov"] + 0.44 * s[f"{side}_fta"])
    s["fta"] = s.home_fta + s.away_fta; s["P"] = s.home_P + s.away_P; s["margin"] = s.home_pts - s.away_pts
    return s


def ratio(df, f="fta", p="P"):
    return df[f].sum() / df[p].sum()


G = {}; A = {}; S = {}
for sea in (2023, 2024, 2025):
    G[sea] = ctx(sea)
    tb = box_actual(sea, G[sea])
    ga = tb.groupby("game_id")[["fta", "P"]].sum().reset_index()
    G[sea] = G[sea].merge(ga, on="game_id", how="inner").rename(columns={"fta": "a_fta", "P": "a_P"})
    A[sea] = tb
for sea in (2024, 2025):
    S[sea] = sim_games(sea)
    sm = S[sea].groupby("game_id").agg(m_margin=("margin", "mean")).reset_index()
    G[sea] = G[sea].merge(sm, on="game_id", how="inner")
    G[sea]["mism"] = pd.cut(G[sea].m_margin.abs(), [-1, 5, 10, 15, 100], labels=["<5", "5-10", "10-15", "15+"]).astype(str)
    G[sea]["fmb"] = pd.cut(G[sea].margin.abs(), FMB, labels=FML).astype(str)
    S[sea] = S[sea][S[sea].game_id.isin(G[sea].game_id)].merge(
        G[sea][["game_id", "dssb", "dss", "site", "conf", "mism"]], on="game_id")
    S[sea]["fmb"] = pd.cut(S[sea].margin.abs(), FMB, labels=FML).astype(str)
    G[sea] = G[sea][G[sea].game_id.isin(S[sea].game_id)]

# ---------------------------------------------------------------- 1 overall + calendar
say("== 1. OVERALL FTA/P (sum over both teams, box possessions), by days bucket ==")
rows = []
for sea in (2023, 2024, 2025):
    for b in BL + ["all"]:
        ga = G[sea] if b == "all" else G[sea][G[sea].dssb == b]
        r = dict(season=sea, bucket=b, n_games=len(ga), actual=ratio(ga, "a_fta", "a_P"))
        if sea in S:
            ss = S[sea] if b == "all" else S[sea][S[sea].dssb == b]
            r["sim"] = ratio(ss); r["gap"] = r["sim"] - r["actual"]
            e = (ga.a_fta - r["actual"] * ga.a_P) / ga.a_P.mean()
            r["se_act"] = float(e.std() / np.sqrt(len(ga)))
        rows.append(r)
tab(pd.DataFrame(rows).round(4), index=False)

say("== 1b. CALENDAR PROFILE: FTA/P by week of season (actual 2023-25, sim 2024-25) ==")
rows = []
for wk in range(0, 22):
    r = dict(week=wk)
    for sea in (2023, 2024, 2025):
        ga = G[sea][(G[sea].dss // 7) == wk]
        r[f"act{sea}"] = ratio(ga, "a_fta", "a_P") if len(ga) >= 30 else np.nan
        r[f"n{sea}"] = len(ga)
        if sea in S:
            ss = S[sea][(S[sea].dss // 7) == wk]; r[f"sim{sea}"] = ratio(ss) if len(ga) >= 30 else np.nan
    rows.append(r)
tab(pd.DataFrame(rows).round(4), index=False)


# ---------------------------------------------------------------- 2 game context
def cell_table(sea, col):
    rows = []
    for lv in sorted(set(G[sea][col].unique()) | set(S[sea][col].unique())):
        for b in BL:
            ga = G[sea][(G[sea].dssb == b) & (G[sea][col] == lv)]
            ss = S[sea][(S[sea].dssb == b) & (S[sea][col] == lv)]
            if len(ga) == 0:
                continue
            wa = ga.a_P.sum() / G[sea][G[sea].dssb == b].a_P.sum()
            ws = ss.P.sum() / S[sea][S[sea].dssb == b].P.sum()
            rows.append(dict(season=sea, var=col, level=lv, bucket=b, n_games=len(ga), w_act=wa, w_sim=ws,
                             actual=ratio(ga, "a_fta", "a_P"), sim=ratio(ss) if len(ss) else np.nan))
    d = pd.DataFrame(rows); d["gap"] = d.sim - d.actual
    d["gap_contrib"] = np.where(d.bucket == "d0-14", d.w_act * d.gap, np.nan)
    d["underpowered"] = np.where(d.n_games < 150, "yes", "")
    return d


say("== 2. GAME CONTEXT cells (FTA/P; w = share of bucket possessions; gap_contrib = w_act*gap in d0-14, sums to ~window gap) ==")
for sea in (2025, 2024):
    for col in ("site", "conf", "mism", "fmb"):
        tab(cell_table(sea, col).round(4), index=False)
say("== 2b. MIX STANDARDISATION: window FTA/P if each cell ran at its own d46+ rate, using the WINDOW possession mix ==")
rows = []
for sea in (2025, 2024):
    for col in ("site", "conf", "mism", "fmb"):
        for src in ("actual", "sim"):
            if src == "actual":
                w = G[sea][G[sea].dssb == "d0-14"].groupby(col).a_P.sum(); w = w / w.sum()
                late = G[sea][G[sea].dssb == "d46+"].groupby(col)[["a_fta", "a_P"]].sum(); lr = late.a_fta / late.a_P
                raw = ratio(G[sea][G[sea].dssb == "d0-14"], "a_fta", "a_P"); lateall = ratio(G[sea][G[sea].dssb == "d46+"], "a_fta", "a_P")
            else:
                w = S[sea][S[sea].dssb == "d0-14"].groupby(col).P.sum(); w = w / w.sum()
                late = S[sea][S[sea].dssb == "d46+"].groupby(col)[["fta", "P"]].sum(); lr = late.fta / late.P
                raw = ratio(S[sea][S[sea].dssb == "d0-14"]); lateall = ratio(S[sea][S[sea].dssb == "d46+"])
            std = float((w * lr.reindex(w.index)).sum())
            rows.append(dict(season=sea, var=col, src=src, window_raw=raw, window_if_late_rates=std, d46_all=lateall,
                             window_minus_d46=raw - lateall, mix_part=std - lateall, within_cell_part=raw - std))
tab(pd.DataFrame(rows).round(4), index=False)

# ---------------------------------------------------------------- 3 per team
say("== 3. PER TEAM, window d0-14 (teams with >=2 window games; quintile of PRIOR-season own FTA/P from box, grading-only) ==")
for sea in (2025, 2024):
    prev = box_actual(sea - 1, ctx(sea - 1)); pr = prev.groupby("team_id")[["fta", "P"]].sum(); prior = (pr.fta / pr.P)
    win = G[sea][G[sea].dssb == "d0-14"].game_id
    ta = A[sea][A[sea].game_id.isin(win)].groupby("team_id")[["fta", "P"]].sum()
    s = S[sea][S[sea].dssb == "d0-14"].merge(G[sea][["game_id", "home_team_id", "away_team_id"]], on="game_id")
    sh = s.groupby("home_team_id")[["home_fta", "home_P"]].sum().rename(columns={"home_fta": "fta", "home_P": "P"})
    sa = s.groupby("away_team_id")[["away_fta", "away_P"]].sum().rename(columns={"away_fta": "fta", "away_P": "P"})
    ts = sh.add(sa, fill_value=0)
    ng = A[sea][A[sea].game_id.isin(win)].groupby("team_id").size()
    t = pd.DataFrame({"a": ta.fta / ta.P, "s": ts.fta / ts.P, "n": ng}).dropna(); t = t[t.n >= 2]
    t["gap"] = t.s - t.a; t["prior"] = prior.reindex(t.index)
    t = t.dropna()
    t["q"] = pd.qcut(t.prior, 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    q = t.groupby("q", observed=True).agg(teams=("gap", "size"), prior=("prior", "mean"), actual=("a", "mean"), sim=("s", "mean"),
                                           gap=("gap", "mean"), gap_se=("gap", lambda x: x.std() / np.sqrt(len(x))))
    sl = np.polyfit(t.prior, t.a, 1)[0]; sls = np.polyfit(t.prior, t.s, 1)[0]
    say(f"season {sea}: teams {len(t)}, share with sim < actual {np.mean(t.gap < 0):.3f}, mean gap {t.gap.mean():.4f}, "
        f"slope of window actual FTA/P on prior {sl:.2f}, of sim {sls:.2f}; corr(actual, prior) {np.corrcoef(t.prior, t.a)[0, 1]:.2f}, "
        f"corr(sim, prior) {np.corrcoef(t.prior, t.s)[0, 1]:.2f}, corr(gap, prior) {np.corrcoef(t.prior, t.gap)[0, 1]:.2f}")
    tab(q.round(4))

# ---------------------------------------------------------------- 4 source split (F2)
say("== 4. SOURCE SPLIT, F2 (2025). actual: possessions_v2 chances; sim: r9tapfull_COMB9 half_agg (R9ao3). Per possession ==")
ch = pd.read_parquet(ROOT / "data/processed/possessions_v2/chances_2025.parquet")
ch = ch[ch.period <= 2].copy()
ch["half"] = ch.period
ch["is_sh"] = (ch.terminal_event == "FT_trip_shooting").astype(int); ch["is_bo"] = (ch.terminal_event == "FT_trip_bonus").astype(int)
ch["ao"] = ch.and_one.astype(int)
ch["fta_sh"] = ch.fta * ch.is_sh; ch["fta_bo"] = ch.fta * ch.is_bo; ch["fta_ao"] = ch.fta * ch.ao * (1 - ch.is_sh) * (1 - ch.is_bo)
ch["fta_other"] = ch.fta - ch.fta_sh - ch.fta_bo - ch.fta_ao
ch["late_int"] = ((ch.is_bo == 1) & (ch.period == 2) & (ch.start_clock <= 120) & (ch.start_score_diff > 0)).astype(int)
ch["fta_late_int"] = ch.fta * ch.late_int
cp = pd.read_parquet(PO / "round6/foul_accrual_poss_v2.parquet", columns=["game_id", "season", "period", "poss_index", "offense_team_id"])
cp = cp[(cp.season == 2025) & (cp.period <= 2)].copy()
cp["half"] = cp.period
ga = G[2025][["game_id", "dssb", "dss", "site", "conf", "mism", "fmb", "m_margin"]]
cpg = cp.groupby(["game_id", "half"]).size().rename("poss").reset_index()
chg = ch.groupby(["game_id", "half"])[["is_sh", "is_bo", "ao", "fta", "fta_sh", "fta_bo", "fta_ao", "fta_other", "late_int", "fta_late_int"]].sum().reset_index()
act = cpg.merge(chg, on=["game_id", "half"], how="left").fillna(0).merge(ga, on="game_id")
shards = sorted((ROOT / "results/engine_v0").glob("r9tapfull_COMB9_s200_o0_off*_n25"))
ha = pd.concat([pd.read_parquet(s / "half_agg.parquet") for s in shards])
say(f"tap shards {len(shards)}, seeds {ha.seed.nunique()}, games {ha.game_id.nunique()}, halves {sorted(ha.half.unique())}")
ha = ha[ha.game_id.isin(ga.game_id)].merge(ga, on="game_id")
for b in BL:
    x = ha[ha.dssb == b]; y = act[act.dssb == b]
    say(f"  check {b}: tap FTA/poss {x.fta.sum() / x.poss.sum():.4f}, actual {y.fta.sum() / y.poss.sum():.4f}; "
        f"possessions per team-half sim {x.poss.mean():.2f} actual {y.poss.sum() / (2 * len(y)):.2f}")


def srcrow(x, kind):
    n = x.poss.sum()
    if kind == "act":
        sh, bo, ao = x.is_sh.sum(), x.is_bo.sum(), x.ao.sum()
    else:
        sh, bo, ao = x.n_shoot_trip.sum(), x.n_bonus_trip.sum(), x.and_one.sum()
    return dict(poss=n, fta_p=x.fta.sum() / n, shoot_trip_p=sh / n, bonus_trip_p=bo / n, andone_p=ao / n,
                trips_p=(sh + bo + ao) / n, fta_per_trip=x.fta.sum() / (sh + bo + ao))


rows = []
for b in BL:
    for h in (1, 2, 0):
        xa = act[(act.dssb == b) & ((act.half == h) | (h == 0))]; xs = ha[(ha.dssb == b) & ((ha.half == h) | (h == 0))]
        a, s = srcrow(xa, "act"), srcrow(xs, "sim")
        for k in a:
            if k == "poss":
                continue
            rows.append(dict(bucket=b, half=("both" if h == 0 else f"H{h}"), metric=k, actual=a[k], sim=s[k], gap=s[k] - a[k]))
src = pd.DataFrame(rows)
tab(src.pivot_table(index=["bucket", "half"], columns="metric", values=["actual", "sim", "gap"], sort=False).round(4))
say("-- window gap attributed (FTA/poss = trips_p x fta_per_trip): per half, contribution to bucket gap --")
rows = []
for b in BL:
    for h in (1, 2):
        xa = act[(act.dssb == b) & (act.half == h)]; xs = ha[(ha.dssb == b) & (ha.half == h)]
        a, s = srcrow(xa, "act"), srcrow(xs, "sim")
        rows.append(dict(bucket=b, half=f"H{h}", poss_share=a["poss"] / act[act.dssb == b].poss.sum(), fta_p_gap=s["fta_p"] - a["fta_p"],
                         d_shoot=(s["shoot_trip_p"] - a["shoot_trip_p"]) * a["fta_per_trip"],
                         d_bonus=(s["bonus_trip_p"] - a["bonus_trip_p"]) * a["fta_per_trip"],
                         d_andone=(s["andone_p"] - a["andone_p"]) * a["fta_per_trip"],
                         d_fta_per_trip=s["trips_p"] * (s["fta_per_trip"] - a["fta_per_trip"]),
                         contrib_to_bucket_gap=(a["poss"] / act[act.dssb == b].poss.sum()) * (s["fta_p"] - a["fta_p"])))
tab(pd.DataFrame(rows).round(4), index=False)
rows = []
for b in BL:
    x = act[act.dssb == b]; n = x.poss.sum()
    rows.append(dict(bucket=b, fta_p=x.fta.sum() / n, shoot=x.fta_sh.sum() / n, bonus=x.fta_bo.sum() / n, andone=x.fta_ao.sum() / n,
                     other=x.fta_other.sum() / n, late_int_proxy=x.fta_late_int.sum() / n,
                     bonus_H1=x[x.half == 1].fta_bo.sum() / n, bonus_H2=x[x.half == 2].fta_bo.sum() / n))
say("-- ACTUAL FTA per possession by source (late_int_proxy = bonus-trip FTA in last 2:00, offence leading; subset of bonus) --")
tab(pd.DataFrame(rows).round(4), index=False)
say("-- window F2 source split by context (H1+H2, per possession) --")
rows = []
for col in ("site", "mism", "conf"):
    for lv in sorted(act[col].unique()):
        xa = act[(act.dssb == "d0-14") & (act[col] == lv)]; xs = ha[(ha.dssb == "d0-14") & (ha[col] == lv)]
        if len(xa) < 20:
            continue
        a, s = srcrow(xa, "act"), srcrow(xs, "sim")
        rows.append(dict(var=col, level=lv, n_team_halves_act=len(xa), fta_p_act=a["fta_p"], fta_p_sim=s["fta_p"], gap=s["fta_p"] - a["fta_p"],
                         shoot_act=a["shoot_trip_p"], shoot_sim=s["shoot_trip_p"], bonus_act=a["bonus_trip_p"], bonus_sim=s["bonus_trip_p"],
                         ao_act=a["andone_p"], ao_sim=s["andone_p"], ftapt_act=a["fta_per_trip"], ftapt_sim=s["fta_per_trip"]))
tab(pd.DataFrame(rows).round(4), index=False)

# ---------------------------------------------------------------- 5 offline foul model
say("== 5. OFFLINE FOUL MODEL given its inputs: mean p - mean y by days bucket (served arms: T2c trips, AO3 and-ones, A2 accrual) ==")


def games_played(sea):
    u = ctx(sea)
    long = pd.concat([u[["game_id", "date", "home_team_id"]].rename(columns={"home_team_id": "t"}),
                      u[["game_id", "date", "away_team_id"]].rename(columns={"away_team_id": "t"})]).sort_values(["t", "date", "game_id"])
    long["k"] = long.groupby("t").cumcount()
    return long[["game_id", "t", "k"]]


KB = [-1, 2, 5, 9, 1000]; KL = ["0-2", "3-5", "6-9", "10+"]
rows = []; rows_n = []; rows_ctx = []
for sea, fold in ((2025, "F2"), (2024, "F1")):
    gp = games_played(sea)
    gctx = G[sea][["game_id", "dssb", "dss", "site", "conf", "mism", "fmb"]]
    pt = pd.read_parquet(PO / f"round9/preds_trip_{fold}_seed0.parquet"); pt = pt[pt.season == sea].merge(gctx, on="game_id", how="inner")
    pt = pt.merge(gp.rename(columns={"t": "offense_team_id", "k": "off_k"}), on=["game_id", "offense_team_id"], how="left")
    pt = pt.merge(gp.rename(columns={"t": "defense_team_id", "k": "def_k"}), on=["game_id", "defense_team_id"], how="left")
    pa = pd.read_parquet(PO / f"round9/preds_ao_{fold}_seed0.parquet"); pa = pa[pa.season == sea].merge(gctx, on="game_id", how="inner")
    pp = pd.read_parquet(PO / f"round7/preds_poss_{fold}_seed0.parquet"); pp = pp[pp.season == sea].merge(gctx, on="game_id", how="inner")
    pp = pp.merge(gp.rename(columns={"t": "offense_team_id", "k": "off_k"}), on=["game_id", "offense_team_id"], how="left")
    pp = pp.merge(gp.rename(columns={"t": "defense_team_id", "k": "def_k"}), on=["game_id", "defense_team_id"], how="left")
    pt["kmin"] = pd.cut(np.minimum(pt.off_k, pt.def_k), KB, labels=KL).astype(str)
    pp["kmin"] = pd.cut(np.minimum(pp.off_k, pp.def_k), KB, labels=KL).astype(str)
    for b in BL:
        for nm, df, yc, pc in (("shooting-foul trip (T2c) per chance", pt, "y_shoot", "y_shoot__T2c"),
                               ("bonus trip (T2c) per chance", pt, "y_bonus", "y_bonus__T2c"),
                               ("and-one (AO3) per FGA chance", pa, "y_ao", "y_ao__AO3"),
                               ("non-trip def foul accrual (A2) per poss", pp, "y_nt", "y_nt__A2")):
            x = df[df.dssb == b]
            rows.append(dict(fold=fold, bucket=b, block=nm, n=len(x), mean_y=x[yc].mean(), mean_p=x[pc].mean(),
                             p_minus_y=x[pc].mean() - x[yc].mean(), rel=(x[pc].mean() - x[yc].mean()) / x[yc].mean(),
                             se_rel=x[yc].std() / np.sqrt(len(x)) / x[yc].mean()))
    for b in ("d0-14", "d15-45"):
        for k in KL:
            for nm, df, yc, pc in (("shooting trip", pt, "y_shoot", "y_shoot__T2c"), ("bonus trip", pt, "y_bonus", "y_bonus__T2c"),
                                   ("non-trip accrual", pp, "y_nt", "y_nt__A2")):
                x = df[(df.dssb == b) & (df.kmin == k)]
                if len(x) < 2000:
                    continue
                rows_n.append(dict(fold=fold, bucket=b, min_games_played=k, block=nm, n=len(x), mean_y=x[yc].mean(),
                                   p_minus_y=x[pc].mean() - x[yc].mean(), rel=(x[pc].mean() - x[yc].mean()) / x[yc].mean()))
    for col in ("site", "conf", "mism", "fmb"):
        for lv in sorted(pt[col].unique()):
            x = pt[(pt.dssb == "d0-14") & (pt[col] == lv)]
            if len(x) < 3000:
                continue
            rows_ctx.append(dict(fold=fold, var=col, level=lv, n_chances=len(x),
                                 shoot_rel=(x.y_shoot__T2c.mean() - x.y_shoot.mean()) / x.y_shoot.mean(),
                                 bonus_rel=(x.y_bonus__T2c.mean() - x.y_bonus.mean()) / x.y_bonus.mean(),
                                 y_shoot=x.y_shoot.mean(), y_bonus=x.y_bonus.mean()))
tab(pd.DataFrame(rows).round(5), index=False)
say("-- same, window and d15-45 by as-of sample size = min(games played by the offence, by the defence), before the game --")
tab(pd.DataFrame(rows_n).round(5), index=False)
say("-- trip model p-y (relative) by game context, window only --")
tab(pd.DataFrame(rows_ctx).round(4), index=False)

(OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")
