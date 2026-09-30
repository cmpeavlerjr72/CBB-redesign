"""build_po_in_bonus_overlay_v1.py -- round 8 (possession_outcome experiments.md s22/s23): the
corrected-state `in_bonus` overlay for a possession-outcome retrain arm, WITHOUT touching any
served table or trainer.

Keys (game_id, poss_index, chance_number), unique per design row; column `in_bonus` = the
ENGINE's definition: defence team fouls BEFORE the possession's own fouls (v2: pbp counter minus
the possession's pre-open fouls) plus the defence's trip fouls on EARLIER chances of the same
possession (the engine re-reads the count on every chance), >= 6. Design rows with no v2 match get
NaN, which `train_par_common_v1` treats as "keep the design value".

For `scripts/train_possession_outcome_s1_par_v1.py` (not edited):
    --feature-table data/processed/models/possession_outcome/round8/in_bonus_overlay_v2state_v1.parquet
    --overlay-keys game_id,poss_index,chance_number --overlay-cols in_bonus  (+ its own --out-root)
"""
from pathlib import Path

import pandas as pd

OUT = Path("data/processed/models/possession_outcome/round8")
OUT.mkdir(parents=True, exist_ok=True)
ch = pd.read_parquet("data/processed/models/possession_outcome/round2/design.parquet",
                     columns=["game_id", "season", "poss_index", "chance_number", "terminal_event",
                              "and_one", "in_bonus"])
v2 = pd.read_parquet("data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet",
                     columns=["game_id", "season", "poss_index", "def_team_fouls_true"])
d = ch.merge(v2, on=["game_id", "season", "poss_index"], how="left", validate="many_to_one")
d = d.sort_values(["game_id", "poss_index", "chance_number"])
trip = (d["terminal_event"].isin(["FT_trip_shooting", "FT_trip_bonus"]).astype(int)
        + d["and_one"].astype(int))
prev = trip.groupby([d["game_id"], d["poss_index"]]).cumsum() - trip
live = d["def_team_fouls_true"] + prev
d["in_bonus_new"] = (live >= 6).astype("float32").where(d["def_team_fouls_true"].notna())
o = d[["game_id", "poss_index", "chance_number"]].copy()
o["in_bonus"] = d["in_bonus_new"]
assert not o.duplicated(["game_id", "poss_index", "chance_number"]).any()
o.to_parquet(OUT / "in_bonus_overlay_v2state_v1.parquet", index=False)
chg = (d["in_bonus_new"] != d["in_bonus"]) & d["in_bonus_new"].notna()
print(f"rows {len(o):,}  unmatched (NaN, keep design) {o['in_bonus'].isna().sum():,}  "
      f"changed {chg.sum():,} ({100 * chg.mean():.2f}%)  label=1->0 {((d['in_bonus'] == 1) & (d['in_bonus_new'] == 0)).sum():,}  "
      f"0->1 {((d['in_bonus'] == 0) & (d['in_bonus_new'] == 1)).sum():,}")
