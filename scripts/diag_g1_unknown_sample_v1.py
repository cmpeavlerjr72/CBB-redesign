"""diag_g1_unknown_sample_v1.py -- stratified raw-row read of `unknown` possessions.

Lane B follow-up, 2026-09-30.  DIAGNOSTIC ONLY.  Takes the records written by
`diag_g1_unknown_poss_v1.py`, assigns each to a mechanical class, draws a
stratified sample (season x class x period), and prints the raw rows of BOTH
feeds at the same game moment: CBBD `plays_{season}` (playType, team,
secondsRemaining, playText) and hoopR `play_by_play_{season}` (type_text,
team, clock, text), a window of the moment +/- 20 s of game clock.  Also
writes the per-class, per-season count table.

Usage: diag_g1_unknown_sample_v1.py [n_per_cell]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/g1g5_diag"
NPC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
RNG = np.random.default_rng(930)


def classify(R: pd.DataFrame) -> pd.Series:
    prev = R["ctx"].str.split("|").map(lambda xs: xs[max(0, len(xs) - 4)] if len(xs) >= 4 else xs[0])
    prev_cls = prev.str.split(":").str[0]
    c = np.select(
        [
            (R["trigger"] == "dreb_no_open_possession") & R["prev_andone_missed"],
            (R["trigger"] == "dreb_no_open_possession") & prev_cls.isin(["FT_made", "FT_missed"]),
            (R["trigger"] == "dreb_no_open_possession"),
            (R["trigger"] == "mismatch_guard") & (R["opened_by"] == "OREB"),
            (R["trigger"] == "mismatch_guard"),
        ],
        ["A_dreb_after_missed_andone_FT", "B1_dreb_no_open_after_FT", "B2_dreb_no_open_other",
         "C1_mismatch_open_by_OREB", "C2_mismatch_other"],
        default="D_other_dreb_with_nothing_pending")
    return pd.Series(c, index=R.index)


def main():
    parts = []
    for s in (2022, 2023, 2024, 2025):
        R = pd.read_parquet(OUT / f"unknown_poss_{s}.parquet")
        R["dur"] = R["start_clock"] - R["end_clock"]
        R["cls"] = classify(R)
        parts.append(R)
    R = pd.concat(parts, ignore_index=True)
    R.to_parquet(OUT / "unknown_poss_classified.parquet")
    tab = R.pivot_table(index="cls", columns="season", values="game_id", aggfunc="size", fill_value=0)
    med = R.groupby("cls")["dur"].median()
    print(tab.assign(median_dur=med).to_string())
    (OUT / "unknown_class_table.json").write_text(json.dumps(
        {"counts": tab.to_dict(), "median_dur": med.to_dict()}, default=int, indent=1), encoding="utf-8")

    lines = []
    for s in (2022, 2023, 2024, 2025):
        cb = pd.read_parquet(ROOT / f"data/raw/cbbd/pbp/plays_{s}.parquet",
                             columns=["gameId", "id", "period", "secondsRemaining", "playType", "isHomeTeam",
                                      "team", "playText"])
        cb = cb.drop_duplicates(["gameId", "id"]).sort_values(["gameId", "id"])
        hr = pd.read_parquet(ROOT / f"data/raw/hoopr/pbp/play_by_play_{s}.parquet",
                             columns=["game_id", "sequence_number", "period_number", "clock_minutes",
                                      "clock_seconds", "type_text", "team_id", "text"])
        hr["sec"] = pd.to_numeric(hr["clock_minutes"], errors="coerce") * 60 + pd.to_numeric(hr["clock_seconds"], errors="coerce")
        hr["seq"] = pd.to_numeric(hr["sequence_number"], errors="coerce")
        Rs = R[R["season"] == s]
        for c in sorted(Rs["cls"].unique()):
            cell = Rs[Rs["cls"] == c]
            take = cell.sample(min(NPC, len(cell)), random_state=int(RNG.integers(1e9)))
            for r in take.itertuples():
                lines.append(f"\n##### season {s} class {c} game {r.game_id} (cbbd {r.cbbd_game_id}) "
                             f"period {r.period} poss {r.start_clock}->{r.end_clock}s start={r.start_reason} "
                             f"opened_by={r.opened_by}")
                lo, hi = r.end_clock - 20, r.start_clock + 20
                w = cb[(cb["gameId"] == r.cbbd_game_id) & (cb["period"] == r.period)
                       & cb["secondsRemaining"].between(lo, hi)]
                lines.append("  CBBD:")
                for x in w.itertuples():
                    lines.append(f"    {int(x.secondsRemaining):5d} {'H' if x.isHomeTeam else 'A'} "
                                 f"{x.playType:<22s} {str(x.playText)[:90]}")
                v = hr[(hr["game_id"] == r.game_id) & (hr["period_number"] == r.period)
                       & hr["sec"].between(lo, hi)].sort_values("seq")
                lines.append("  hoopR:")
                for x in v.itertuples():
                    lines.append(f"    {int(x.sec) if x.sec == x.sec else -1:5d} {str(x.team_id):>6s} "
                                 f"{str(x.type_text):<22s} {str(x.text)[:90]}")
    (OUT / "unknown_sample_rows.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(lines)} lines")


if __name__ == "__main__":
    main()
