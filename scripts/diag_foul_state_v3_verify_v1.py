"""diag_foul_state_v3_verify_v1.py -- round 8 (experiments.md s22.1): verify the v3 foul state
against the box score (second source) and against the one-and-one rule; bonus-state agreement by
game minute. Read-only on foul_accrual_poss_v3 / rowless_trips_v3 / hoopR team box / chances."""
import json
import numpy as np
import pandas as pd

R6 = "data/processed/models/possession_outcome/round6/"
d = pd.read_parquet(R6 + "foul_accrual_poss_v3.parquet")
rl = pd.read_parquet(R6 + "rowless_trips_v3.parquet")
out = {"reconciliation": {}, "agreement_by_minute": {}, "one_and_one_rule": {}}
for s in (2022, 2023, 2024, 2025):
    x = d[d.season == s]
    a = x.groupby(["game_id", "defense_team_id"])[["def_silent", "def_trip"]].sum().sum(axis=1)
    b = x.groupby(["game_id", "offense_team_id"])[["off_silent", "off_trip"]].sum().sum(axis=1)
    a.index.names = b.index.names = ["game_id", "team_id"]
    logged = a.add(b, fill_value=0)
    # the pbp COUNTER also counts and-one fouls (possessions.py `_handle_fga` increments
    # `team_fouls` directly, never through `_handle_foul`, so the round-6/7 replay log misses them)
    ch = pd.read_parquet(f"data/processed/possessions_v2/chances_{s}.parquet",
                         columns=["game_id", "defense_team_id", "and_one"])
    ao = ch[ch.and_one].groupby(["game_id", "defense_team_id"]).size()
    ao.index.names = ["game_id", "team_id"]
    pbp = logged.add(ao, fill_value=0).rename("pbp")
    logged_only = logged.rename("logged")
    r = rl[rl.season == s].groupby(["game_id", "fouling_team_id"]).size()
    r.index.names = ["game_id", "team_id"]
    t = pbp.to_frame().join(r.rename("rowless")).join(logged_only).fillna(0)
    t["v3"] = t.pbp + t.rowless
    box = pd.read_parquet(f"data/raw/hoopr/team_box/team_box_{s}.parquet")[["game_id", "team_id", "fouls"]]
    m = t.reset_index().merge(box, on=["game_id", "team_id"])
    gp, g3 = m.fouls - m.pbp, m.fouls - m.v3
    out["reconciliation"][s] = {
        "team_games": int(len(m)), "box_mean": float(m.fouls.mean()),
        "gap_replaylog_mean": float((m.fouls - m.logged).mean()),
        "gap_pbp_mean": float(gp.mean()), "gap_pbp_mae": float(gp.abs().mean()),
        "gap_v3_mean": float(g3.mean()), "gap_v3_mae": float(g3.abs().mean()),
        "share_exact_pbp": float((gp == 0).mean()), "share_exact_v3": float((g3 == 0).mean()),
        "share_within1_v3": float((g3.abs() <= 1).mean()),
        "dist_v3": {int(k): int(v) for k, v in g3.clip(-4, 6).value_counts().sort_index().items()},
        "dist_pbp": {int(k): int(v) for k, v in gp.clip(-4, 6).value_counts().sort_index().items()},
        "pass_rule_half": bool(g3.abs().mean() < 0.5 * gp.abs().mean())}
x = d[(d.season == 2025) & (d.period <= 2)].copy()
gm = (x.period - 1) * 20 + (1200 - x.start_clock) / 60
x["b"] = pd.cut(gm, [0, 5, 10, 15, 20, 25, 30, 35, 38, 40.01], right=False).astype(str)
lab, v2, v3 = x.def_team_fouls >= 6, x.def_team_fouls_true >= 6, x.def_team_fouls_v3 >= 6
g = pd.DataFrame({"b": x.b, "v3_eq_label": v3 == lab, "v3_eq_v2": v3 == v2,
                  "p_label": lab, "p_v2": v2, "p_v3": v3}).groupby("b").mean()
out["agreement_by_minute"] = g.round(4).to_dict(orient="index")
# one-and-one rule check: a missed single-FT trip that is not an and-one is a one-and-one front
# end, which exists only in the bonus. First chances only (possession-open count = prior count).
rows = []
for s in (2022, 2023, 2024, 2025):
    c = pd.read_parquet(f"data/processed/possessions_v2/chances_{s}.parquet",
                        columns=["game_id", "period", "poss_index", "chance_number", "fta", "ftm",
                                 "and_one", "terminal_event"])
    c = c[(c.chance_number == 1) & (c.fta == 1) & (c.ftm == 0) & ~c.and_one
          & c.terminal_event.isin(["FT_trip_bonus", "FT_trip_shooting"])]
    rows.append(c.merge(d[d.season == s], on=["game_id", "period", "poss_index"]))
oo = pd.concat(rows)
out["one_and_one_rule"] = {"n": int(len(oo)),
                           "share_ge6_label": float((oo.def_team_fouls >= 6).mean()),
                           "share_ge6_v2": float((oo.def_team_fouls_true >= 6).mean()),
                           "share_ge6_v3": float((oo.def_team_fouls_v3 >= 6).mean())}
json.dump(out, open("results/foul_joint/state_v3_verify.json", "w"), indent=1, default=float)
for s, v in out["reconciliation"].items():
    print(s, {k: (round(val, 3) if isinstance(val, float) else val) for k, val in v.items() if not k.startswith("dist")})
    print("   dist v3", v["dist_v3"], " pbp", v["dist_pbp"])
print(pd.DataFrame(out["agreement_by_minute"]).T.to_string())
print(out["one_and_one_rule"])
