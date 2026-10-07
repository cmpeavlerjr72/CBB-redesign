"""Early-season FTA/P diagnostic part 4 (DIAGNOSTIC ONLY): is the early first-half foul surplus a calendar effect or a
team-composition effect, and does it repeat every season? Actual data only (foul_accrual_poss_v2, seasons 2022-25;
2025-26 never read). Writes results/early_fta_diag/report4.txt.
"""
from __future__ import annotations
import io
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/early_fta_diag"
buf = io.StringIO()


def say(*a):
    print(*a); print(*a, file=buf)


def tab(df, **kw):
    say(df.to_string(**kw)); say("")


BK = [-1, 6, 13, 20, 45, 10_000]; BL = ["d0-6", "d7-13", "d14-20", "d21-45", "d46+"]
cp = pd.read_parquet(ROOT / "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet",
                     columns=["game_id", "season", "period", "poss_index", "offense_team_id", "defense_team_id", "def_silent", "def_trip",
                              "off_silent", "off_trip", "off_in_bonus", "days_since_start", "season_type"])
cp = cp[(cp.period <= 2)].copy()
cp["b"] = pd.cut(cp.days_since_start, BK, labels=BL).astype(str)
ch = pd.concat([pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{s}.parquet", columns=["game_id", "period", "poss_index", "terminal_event", "fta", "season"])
                for s in (2022, 2023, 2024, 2025)])
ch = ch[ch.period <= 2]
ch["bo"] = (ch.terminal_event == "FT_trip_bonus").astype(int); ch["sh"] = (ch.terminal_event == "FT_trip_shooting").astype(int)
cg = ch.groupby(["game_id", "period", "poss_index"]).agg(nb=("bo", "sum"), ns=("sh", "sum"), fta=("fta", "sum")).reset_index()
cp = cp.merge(cg, on=["game_id", "period", "poss_index"], how="left").fillna({"nb": 0, "ns": 0, "fta": 0})
cp["nontrip"] = cp.def_silent + cp.off_silent

say("== I. REPEATS EVERY SEASON? first-half (H1) and second-half (H2) rates per possession, by days bucket, seasons 2022-25 (actual) ==")
rows = []
for sea in (2022, 2023, 2024, 2025):
    for h in (1, 2):
        for b in BL:
            x = cp[(cp.season == sea) & (cp.period == h) & (cp.b == b)]
            rows.append(dict(season=sea, half=h, bucket=b, poss=len(x), in_bonus=x.off_in_bonus.mean(), nontrip_fouls=x.nontrip.mean(),
                             shoot_trips=x.ns.mean(), bonus_trips=x.nb.mean(), fta=x.fta.mean()))
d = pd.DataFrame(rows)
tab(d[d.half == 1].round(4), index=False)
say("H2:")
tab(d[d.half == 2].round(4), index=False)

say("== J. TEAM-COMPOSITION TEST: H1 in-bonus share and H1 FTA/poss, early (d0-13) vs the SAME teams late (d46+), team-weighted ==")
say("raw_early = early rate; team_std_early = early weights x each team's own d46+ rate (what the early window would show if every team ran at its late rate)")
rows = []
for sea in (2022, 2023, 2024, 2025):
    x = cp[(cp.season == sea) & (cp.period == 1) & (cp.season_type == 2)]
    for who, col in (("defence", "defense_team_id"), ("offence", "offense_team_id")):
        for metric in ("off_in_bonus", "fta", "nb"):
            e = x[x.days_since_start <= 13].groupby(col)[metric].agg(["sum", "size"])
            l = x[x.days_since_start >= 46].groupby(col)[metric].mean()
            e["late"] = l.reindex(e.index)
            e = e.dropna()
            raw = e["sum"].sum() / e["size"].sum(); std = (e["size"] * e["late"]).sum() / e["size"].sum()
            late_all = x[x.days_since_start >= 46][metric].mean()
            rows.append(dict(season=sea, team_side=who, metric=metric, teams=len(e), raw_early=raw, team_std_early=std, late_all=late_all,
                             early_minus_late=raw - late_all, composition_part=std - late_all, calendar_part=raw - std))
tab(pd.DataFrame(rows).round(4), index=False)
(OUT / "report4.txt").write_text(buf.getvalue(), encoding="utf-8")
