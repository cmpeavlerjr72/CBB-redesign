"""Early-season FTA/P diagnostic part 2 (DIAGNOSTIC ONLY): the first-half team-foul STATE.
Part 1 (diag_early_fta_rate_v1.py) located the window FTA/P gap in first-half bonus trips. This script asks whether
the sim reaches the bonus too rarely in H1 because (i) the realised team-foul state differs, (ii) the served accrual
(A2) misses the early calendar given its inputs, or (iii) the trip layer misses it given the true state.
Writes results/early_fta_diag/report2.txt (gitignored). 2025-26 never read.
"""
from __future__ import annotations
import io
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/early_fta_diag"
PO = ROOT / "data/processed/models/possession_outcome"
buf = io.StringIO()


def say(*a):
    print(*a); print(*a, file=buf)


def tab(df, **kw):
    say(df.to_string(**kw)); say("")


BK = [-1, 6, 13, 20, 27, 45, 10_000]; BL = ["d0-6", "d7-13", "d14-20", "d21-27", "d28-45", "d46+"]
BK3 = [-1, 14, 45, 10_000]; BL3 = ["d0-14", "d15-45", "d46+"]
cp = pd.read_parquet(PO / "round6/foul_accrual_poss_v2.parquet",
                     columns=["game_id", "season", "period", "poss_index", "offense_team_id", "defense_team_id", "def_silent", "def_trip",
                              "off_silent", "off_trip", "off_in_bonus", "def_team_fouls_true", "off_team_fouls_true", "days_since_start",
                              "terminal_event", "start_clock", "start_score_diff", "offense_is_home", "neutral_site"])
cp = cp[(cp.season == 2025) & (cp.period <= 2)].copy()
cp["b3"] = pd.cut(cp.days_since_start, BK3, labels=BL3).astype(str)
cp["b6"] = pd.cut(cp.days_since_start, BK, labels=BL).astype(str)
cp["half"] = cp.period

# actual per-possession state, per half and bucket
g = cp.groupby(["b6", "half"]).agg(poss=("poss_index", "size"), silent=("def_silent", "mean"), trip=("def_trip", "mean"),
                                    off_silent=("off_silent", "mean"), off_trip=("off_trip", "mean"),
                                    in_bonus=("off_in_bonus", "mean"), def_f=("def_team_fouls_true", "mean")).reset_index()
g["def_fouls_per_poss"] = g.silent + g.trip
say("== A. ACTUAL 2025 first/second half team-foul state per possession by finer days bucket ==")
tab(g.round(4), index=False)

# sim tap: per (seed, game, half, off_side) rows
shards = sorted((ROOT / "results/engine_v0").glob("r9tapfull_COMB9_s200_o0_off*_n25"))
ha = pd.concat([pd.read_parquet(s / "half_agg.parquet") for s in shards])
ha = ha[ha.half <= 2]
gm = cp.groupby("game_id").agg(dss=("days_since_start", "first")).reset_index()
ha = ha[ha.game_id.isin(gm.game_id)].merge(gm, on="game_id")
ha["b3"] = pd.cut(ha.dss, BK3, labels=BL3).astype(str)
ha["b6"] = pd.cut(ha.dss, BK, labels=BL).astype(str)
sg = ha.groupby(["b6", "half"]).agg(poss=("poss", "sum"), in_bonus=("in_bonus", "sum"), silent=("silent", "sum"), trip_fouls=("trip_fouls", "sum"),
                                    off_foul=("off_foul", "sum"), n_shoot=("n_shoot_trip", "sum"), n_bonus=("n_bonus_trip", "sum")).reset_index()
for c in ("in_bonus", "silent", "trip_fouls", "off_foul", "n_shoot", "n_bonus"):
    sg[c] = sg[c] / sg.poss
say("== B. SIM (R9ao3 tap, 200 seeds) same quantities (in_bonus = share of possessions with the offence in the bonus; trip_fouls = fouls charged on trips) ==")
tab(sg.round(4), index=False)

# in-bonus share and its two ingredients, window vs late, H1 only: compare directly
a = cp[cp.half == 1].groupby("b3").agg(in_bonus=("off_in_bonus", "mean"), silent=("def_silent", "mean"), trip=("def_trip", "mean"),
                                         def_f=("def_team_fouls_true", "mean"), poss=("poss_index", "size"))
s = ha[ha.half == 1].groupby("b3").agg(in_bonus=("in_bonus", "sum"), silent=("silent", "sum"), trip_fouls=("trip_fouls", "sum"), poss=("poss", "sum"))
s["in_bonus"] /= s.poss; s["silent"] /= s.poss; s["trip_fouls"] /= s.poss
say("== C. H1 only, 3 buckets: actual vs sim in-bonus share and foul production per possession ==")
tab(pd.concat({"actual": a, "sim": s}, axis=1).round(4))

# actual H1: team fouls at the half (max def_team_fouls_true per team-half) by bucket
mx = cp[cp.half == 1].groupby(["game_id", "defense_team_id"]).agg(f=("def_team_fouls_true", "max"), b3=("b3", "first"), poss=("poss_index", "size")).reset_index()
say("== D. ACTUAL H1 team fouls (max engine-definition count seen) per team-half by bucket; and per 35 possessions ==")
tab(mx.groupby("b3").agg(n=("f", "size"), fouls_seen=("f", "mean"), p_ge7=("f", lambda x: (x >= 6).mean())).round(4))

# E. offline: A2 accrual and T0/T2c trip layer, H1 only, by bucket
pp = pd.read_parquet(PO / "round7/preds_poss_F2_seed0.parquet")
pp = pp[(pp.season == 2025) & (pp.period <= 2)].merge(gm, on="game_id")
pp["b3"] = pd.cut(pp.dss, BK3, labels=BL3).astype(str); pp["b6"] = pd.cut(pp.dss, BK, labels=BL).astype(str)
pt = pd.read_parquet(PO / "round9/preds_trip_F2_seed0.parquet")
pt = pt[(pt.season == 2025) & (pt.period <= 2)].merge(gm, on="game_id")
pt["b3"] = pd.cut(pt.dss, BK3, labels=BL3).astype(str); pt["b6"] = pd.cut(pt.dss, BK, labels=BL).astype(str)
rows = []
for h in (1, 2):
    for b in BL:
        x = pp[(pp.half == h) & (pp.b6 == b)]; y = pt[(pt.half == h) & (pt.b6 == b)]
        rows.append(dict(half=h, bucket=b, n_poss=len(x), accr_y=x.y_nt.mean(), accr_p_A2=x.y_nt__A2.mean(), accr_rel=x.y_nt__A2.mean() / x.y_nt.mean() - 1,
                         n_chances=len(y), shoot_y=y.y_shoot.mean(), shoot_T0=y.y_shoot__T0.mean(), shoot_T2c=y.y_shoot__T2c.mean(),
                         bonus_y=y.y_bonus.mean(), bonus_T0=y.y_bonus__T0.mean(), bonus_T2c=y.y_bonus__T2c.mean(),
                         bonus_rel_T2c=y.y_bonus__T2c.mean() / y.y_bonus.mean() - 1, shoot_rel_T2c=y.y_shoot__T2c.mean() / y.y_shoot.mean() - 1))
say("== E. OFFLINE given TRUE state (F2 seed 0): accrual A2 per poss; trip layer T0 (served PO) and T2c (after offsets) per chance; by half and finer bucket ==")
tab(pd.DataFrame(rows).round(4), index=False)

# F. is the early bonus excess a state effect? actual bonus-trip rate conditional on the defence's team-foul count, H1
cp["dfc"] = pd.cut(cp.def_team_fouls_true, [-1, 2, 4, 5, 6, 20], labels=["0-2", "3-4", "5", "6", "7+"]).astype(str)
ch = pd.read_parquet(ROOT / "data/processed/possessions_v2/chances_2025.parquet", columns=["game_id", "period", "poss_index", "terminal_event", "fta"])
ch = ch[ch.terminal_event == "FT_trip_bonus"].groupby(["game_id", "period", "poss_index"]).size().rename("nb").reset_index()
c2 = cp.merge(ch, on=["game_id", "period", "poss_index"], how="left").fillna({"nb": 0})
x = c2[c2.half == 1].groupby(["b3", "dfc"]).agg(poss=("nb", "size"), bonus_trip_per_poss=("nb", "mean")).reset_index()
say("== F. ACTUAL H1: possession share and bonus-trip rate by the defence's team-foul count at possession start (engine-definition) ==")
tab(x.pivot(index="dfc", columns="b3", values=["poss", "bonus_trip_per_poss"]).round(4))

(OUT / "report2.txt").write_text(buf.getvalue(), encoding="utf-8")
