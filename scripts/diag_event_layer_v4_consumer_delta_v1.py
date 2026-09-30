"""diag_event_layer_v4_consumer_delta_v1.py -- read-only: what each sub-model's
training input would lose, gain or change if built from event layer v4.

Lane B, 2026-09-30.  DIAGNOSTIC ONLY.  Rebuilds and retrains nothing.  Seasons
2022-2025.  Writes results/g1g5_diag/event_layer_v4_consumer_delta.json.

Rows are matched across versions by a version-stable identity -- the
possession's (game_id, period, offense_team_id, end_clock) plus an occurrence
counter -- because `poss_index` renumbers when a phantom is removed.
Chances: the same key plus chance_number.  A v2 row with no v4 match is
REMOVED, a v4 row with no v2 match is ADDED (for example a chance created by
turning a restart into a continuation), and matched rows are compared column
by column on the columns each consumer reads (column lists from the consumer
survey in docs/tests/event_layer_v4_2026-09-30.md section 4).

The event-stream consumers (fg_make, rebound, free_throw, usage) read raw
pbp through `models.event_stream`, which does not run the possession machine;
the phantom switches cannot change their rows (checked structurally, not by
rebuilding: `build_stream` has no reference to `_GameMachine`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/g1g5_diag"

POSS_COLS = {
    "clock (design_v2; reads v1 possessions)": ["poss_index", "start_clock", "duration_s", "start_score_diff",
                                                "start_reason", "terminal_event", "off_in_bonus"],
    "rotation (hazard set; reads v1 possessions)": ["poss_index", "duration_s", "start_clock", "start_score_diff"],
}
CH_COLS = {
    "possession_outcome (round2 design; v2 chances, terminal not end_period/unknown)":
        ["terminal_event", "chance_number", "start_clock", "start_score_diff", "off_in_bonus",
         "is_transition", "start_reason", "and_one", "fga_rim", "fga_jump2", "fga_3", "fta", "points"],
    "late_game (PO design + v2 chances)": ["start_score_diff", "points", "chance_number", "poss_index"],
}


def keyed(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    k = ["game_id", "period", "offense_team_id", "end_clock"] + extra
    df = df.sort_values(["game_id", "period", "poss_index"] + extra).copy()
    df["occ"] = df.groupby(k).cumcount()
    return df.set_index(k + ["occ"])


def delta(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> dict:
    common = a.index.intersection(b.index)
    out = {"rows_v2": int(len(a)), "rows_v4": int(len(b)),
           "removed": int(len(a.index.difference(b.index))), "added": int(len(b.index.difference(a.index))),
           "matched": int(len(common)), "changed_by_column": {}}
    aa, bb = a.loc[common], b.loc[common]
    anych = pd.Series(False, index=common)
    for c in cols:
        if c not in aa.columns:
            continue
        ch = ~((aa[c] == bb[c]) | (aa[c].isna() & bb[c].isna()))
        out["changed_by_column"][c] = int(ch.sum())
        anych |= ch
    out["matched_rows_with_any_change"] = int(anych.sum())
    return out


def main():
    rep = {}
    for s in (2022, 2023, 2024, 2025):
        p2 = pd.read_parquet(ROOT / f"data/processed/possessions_v2/possessions_{s}.parquet")
        p4 = pd.read_parquet(ROOT / f"data/processed/possessions_v4/possessions_{s}.parquet")
        p1n = len(pd.read_parquet(ROOT / f"data/processed/possessions/possessions_{s}.parquet", columns=["game_id"]))
        c2 = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{s}.parquet")
        c4 = pd.read_parquet(ROOT / f"data/processed/possessions_v4/chances_{s}.parquet")
        # carry the possession-level start_reason onto chances for the PO start-type feature
        k2, k4 = keyed(p2, []), keyed(p4, [])
        r = {"v1_possession_rows": int(p1n), "v2_possession_rows": int(len(p2))}
        for name, cols in POSS_COLS.items():
            r[name] = delta(k2, k4, cols)
        mask2 = ~c2["terminal_event"].isin(["end_period", "unknown"])
        mask4 = ~c4["terminal_event"].isin(["end_period", "unknown"])
        for name, cols in CH_COLS.items():
            a, b = (c2[mask2], c4[mask4]) if name.startswith("possession_outcome") else (c2, c4)
            r[name] = delta(keyed(a, ["chance_number"]), keyed(b, ["chance_number"]), cols)
        rep[s] = r
        print(s, json.dumps({k: (v if not isinstance(v, dict) else {x: v[x] for x in ("rows_v2", "rows_v4", "removed", "added", "matched_rows_with_any_change")}) for k, v in r.items()}), flush=True)
    (OUT / "event_layer_v4_consumer_delta.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
