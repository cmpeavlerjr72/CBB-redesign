#!/usr/bin/env python
"""OT team-foul state audit (lane F, 2026-10-01). Diagnose only: reads training tables, writes results/ot_team_foul_state.json.
Compares the foul state the training tables carry on overtime possessions (the pbp builder resets team fouls at every period
boundary) with the carry-over state the men's rule and the engine use. Season = end year (2025 = 2024-25)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
a = pd.read_parquet(R / "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet",
                    columns=["game_id", "season", "period", "poss_index", "offense_team_id", "defense_team_id", "off_team_fouls_true",
                             "def_team_fouls_true", "off_in_bonus", "off_in_double_bonus", "off_in_bonus_true", "is_ot"])
a = a.sort_values(["game_id", "period", "poss_index"]).reset_index(drop=True)
# end-of-regulation count per (game, team): max over period 2 of the count the table shows for that team (lower bound: fouls in the
# last possession are not in it)
p2 = a[a["period"] == 2]
e = pd.concat([p2[["game_id", "offense_team_id", "off_team_fouls_true"]].set_axis(["game_id", "team", "n"], axis=1),
               p2[["game_id", "defense_team_id", "def_team_fouls_true"]].set_axis(["game_id", "team", "n"], axis=1)])
end2h = e.groupby(["game_id", "team"])["n"].max().rename("end2h").reset_index()
ot = a[a["is_ot"].astype(bool)].copy()
ot = ot.merge(end2h.rename(columns={"team": "defense_team_id", "end2h": "def_end2h"}), on=["game_id", "defense_team_id"], how="left")
ot = ot.merge(end2h.rename(columns={"team": "offense_team_id", "end2h": "off_end2h"}), on=["game_id", "offense_team_id"], how="left")
ot[["def_end2h", "off_end2h"]] = ot[["def_end2h", "off_end2h"]].fillna(0)
ot["def_carry"] = ot["def_team_fouls_true"] + ot["def_end2h"]
ot["bonus_table"] = ot["def_team_fouls_true"] >= 6
ot["bonus_carry"] = ot["def_carry"] >= 6
ot["dbl_table"] = ot["def_team_fouls_true"] >= 9
ot["dbl_carry"] = ot["def_carry"] >= 9
first = ot.sort_values(["game_id", "period", "poss_index"]).groupby("game_id").head(1)
out = {"first_ot_possession": {"n_games_with_ot": int(len(first)), "def_count_true_mean": float(first["def_team_fouls_true"].mean()),
                              "share_zero": float((first["def_team_fouls_true"] == 0).mean()),
                              "mean_end2h_count_defence_lower_bound": float(first["def_end2h"].mean())}}
rows = []
for s, g in ot.groupby("season"):
    rows.append({"season": int(s), "n_ot_possessions": int(len(g)), "n_ot_games": int(g["game_id"].nunique()),
                 "n_total_possessions": int((a["season"] == s).sum()),
                 "bonus_state_differs": int((g["bonus_table"] != g["bonus_carry"]).sum()),
                 "double_bonus_state_differs": int((g["dbl_table"] != g["dbl_carry"]).sum()),
                 "any_state_differs": int(((g["bonus_table"] != g["bonus_carry"]) | (g["dbl_table"] != g["dbl_carry"])).sum()),
                 "table_off_in_bonus_rate": float(g["off_in_bonus"].astype(float).mean()),
                 "carry_bonus_rate_lower_bound": float(g["bonus_carry"].mean())})
out["by_season"] = rows
d = pd.DataFrame(rows).set_index("season")
fold = lambda ss: {k: int(d.loc[ss, k].sum()) for k in ("n_ot_possessions", "n_ot_games", "bonus_state_differs", "double_bonus_state_differs", "any_state_differs")}
out["fold1_train_2022_2023"] = fold([2022, 2023]); out["fold1_test_2024"] = fold([2024])
out["fold2_train_2022_2024"] = fold([2022, 2023, 2024]); out["fold2_test_2025"] = fold([2025])
# rule check from the feed: free throws per possession in OT, by carry state, vs the regulation second half by true state
b = pd.read_parquet(R / "data/processed/models/possession_outcome/round2/design.parquet", columns=["game_id", "period", "poss_index", "chance_number", "fta", "season"])
pos = b.groupby(["game_id", "period", "poss_index"])["fta"].sum().rename("fta").reset_index()
o2 = ot.merge(pos, on=["game_id", "period", "poss_index"], how="left")
chk = {}
for name, m in (("carry_def_count_>=7", o2["def_carry"] >= 7), ("carry_def_count_<=3", o2["def_carry"] <= 3),
                ("table_def_count_>=7", o2["def_team_fouls_true"] >= 7)):
    chk[name] = {"n": int(m.sum()), "fta_per_possession": float(o2.loc[m, "fta"].mean())}
h2 = a[(a["period"] == 2)].merge(pos, on=["game_id", "period", "poss_index"], how="left")
for name, m in (("H2_def_count_>=7", h2["def_team_fouls_true"] >= 7), ("H2_def_count_<=3", h2["def_team_fouls_true"] <= 3)):
    chk[name] = {"n": int(m.sum()), "fta_per_possession": float(h2.loc[m, "fta"].mean())}
out["feed_rule_check_fta_per_possession"] = chk
(R / "results/ot_team_foul_state.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
