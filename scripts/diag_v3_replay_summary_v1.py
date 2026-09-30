"""Per-date and total counts of cells in v3 (live replay) that differ from v2, by cause. Reads
results/engine_v3_replay/dates/*.json (written by build_engine_inputs_v3_replay.py shard)."""
import json, glob, collections
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted(glob.glob(str(ROOT / "results/engine_v3_replay/dates/*.json"))):
    r = json.loads(Path(f).read_text(encoding="utf-8"))
    c = collections.Counter(); cells = 0; diff = 0
    for x in r["rows"]:
        cells += x["cells"]; diff += x["n_diff"]
        for k, n in x["causes"].items():
            key = k.split("(")[0]
            c[key] += n
    lk = r["leak_constants"]
    fill_usage = sum(v["backtest_real_slots_at_the_constant"] for k, v in lk.items() if k.startswith("usage"))
    fill_clock = lk["clock_tempo_fallback.tempo_prior_game"]["backtest_team_games_at_the_constant"]
    rows.append({"date": r["date"], "games": r["n_games"], "cells": cells, "cells_differing": diff,
                 "stale_slot": c["BT_STALE_ROW"], "has_prior_season_fg": c["BT_FIRST_ROW_CLASS_DEPENDENCE"],
                 "all_zero_po_block": c["BT_NO_PO_DESIGN_ROW"], "anonymous_rotation": c["BT_ROT_FALLBACK"],
                 "r2_block_not_in_design": c["BT_TEAM_GAME_NOT_IN_R2_DESIGN"], "clock_fallback": c["BT_CLOCK_FALLBACK"],
                 "fill_const_usage_slots(sum over 5 classes)": fill_usage, "fill_const_clock_team_games": fill_clock,
                 "unexplained": len(r["unexplained"]), "source_unfinished_at_cutoff": r["diag"].get("n_source_unfinished_at_cutoff", 0)})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/engine_v3_replay/per_date_causes.csv", index=False)
print(df.drop(columns=["date"]).sum(numeric_only=True).to_string()); print(len(df), "dates")
