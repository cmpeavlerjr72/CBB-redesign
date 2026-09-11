#!/usr/bin/env python
"""
build_lines_close_v1.py -- one row per (game_id, provider) close line, 2023-2025.

Reads:
    data/raw/cbbd/lines/lines_{season}.parquet
    data/processed/games_universe_v2.parquet  (game_id <-> cbbd_game_id crosswalk)

`spread`/`overUnder`/moneylines are treated as the close line (see
docs/tests/lines_cbbd_validation_2026-09-10.md section 2: CBBD's `/lines`
payload carries no line-level timestamp at all, only the game-level
`startDate`; `spreadOpen`/`overUnderOpen` are near-empty for 2023-2024 and
sparse in 2025, so "close" here means "the only/last line CBBD has on file",
not a timestamp-verified closing snapshot -- flagged, not hidden).

De-vig: proportional method (p_home = p_home_raw / (p_home_raw + p_away_raw))
is the primary column; the power method (solve k s.t. p_h**k + p_a**k == 1)
is reported alongside per the task spec. No calibration curve, offset, or
clip is applied to either -- this is a de-vig of the input odds, not a
post-hoc adjustment of any sim output (CLAUDE.md "no hand tuning" rule).

Writes:
    data/processed/lines/lines_close_v1.parquet
    data/processed/lines/_build_lines_close_v1_report.json  (join-rate report)

Usage:
    .venv/Scripts/python.exe scripts/build_lines_close_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LINES_DIR = ROOT / "data" / "raw" / "cbbd" / "lines"
OUT_PATH = ROOT / "data" / "processed" / "lines" / "lines_close_v1.parquet"
REPORT_PATH = ROOT / "data" / "processed" / "lines" / "_build_lines_close_v1_report.json"

SEASONS = [2023, 2024, 2025]


def ml_to_prob(ml: pd.Series) -> pd.Series:
    ml = ml.astype("float64")
    pos = ml > 0
    p = pd.Series(np.nan, index=ml.index)
    p[pos] = 100.0 / (ml[pos] + 100.0)
    neg = ~pos & ml.notna()
    p[neg] = -ml[neg] / (-ml[neg] + 100.0)
    return p


def power_devig(p_h: float, p_a: float) -> float:
    if not (np.isfinite(p_h) and np.isfinite(p_a)) or p_h <= 0 or p_a <= 0:
        return np.nan
    lo, hi = 0.05, 20.0

    def f(k):
        return p_h ** k + p_a ** k - 1.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return np.nan
    for _ in range(60):
        mid = (lo + hi) / 2
        fm = f(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    k = (lo + hi) / 2
    return p_h ** k


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gu = pd.read_parquet(
        ROOT / "data" / "processed" / "games_universe_v2.parquet",
        columns=["game_id", "cbbd_game_id", "season", "is_d1_game"],
    )
    gu = gu[gu["season"].isin(SEASONS)].drop(columns=["season"])

    frames = []
    for season in SEASONS:
        lines = pd.read_parquet(LINES_DIR / f"lines_{season}.parquet")
        lines = lines[lines["provider"].notna()].copy()
        frames.append(lines)
    lines_all = pd.concat(frames, ignore_index=True)

    n_line_rows_raw = len(lines_all)
    merged = lines_all.merge(
        gu, left_on="gameId", right_on="cbbd_game_id", how="left", indicator=True
    )
    joined = merged[merged["_merge"] == "both"].copy()
    join_rate = len(joined) / n_line_rows_raw if n_line_rows_raw else float("nan")

    joined["p_home_raw"] = ml_to_prob(joined["homeMoneyline"])
    joined["p_away_raw"] = ml_to_prob(joined["awayMoneyline"])
    valid = joined["p_home_raw"].notna() & joined["p_away_raw"].notna()
    joined["p_home_close_prop"] = np.nan
    joined["p_away_close_prop"] = np.nan
    joined.loc[valid, "p_home_close_prop"] = (
        joined.loc[valid, "p_home_raw"] / (joined.loc[valid, "p_home_raw"] + joined.loc[valid, "p_away_raw"])
    )
    joined.loc[valid, "p_away_close_prop"] = 1.0 - joined.loc[valid, "p_home_close_prop"]

    pow_h = [
        power_devig(ph, pa) if ok else np.nan
        for ph, pa, ok in zip(joined["p_home_raw"], joined["p_away_raw"], valid)
    ]
    joined["p_home_close_power"] = pow_h
    joined["p_away_close_power"] = 1.0 - joined["p_home_close_power"]

    out = joined.rename(columns={
        "spread": "close_spread_home",
        "overUnder": "close_over_under",
        "homeMoneyline": "close_ml_home",
        "awayMoneyline": "close_ml_away",
        "spreadOpen": "open_spread_home",
        "overUnderOpen": "open_over_under",
    })[[
        "game_id", "gameId", "season", "provider",
        "close_spread_home", "close_over_under", "close_ml_home", "close_ml_away",
        "open_spread_home", "open_over_under",
        "p_home_close_prop", "p_away_close_prop",
        "p_home_close_power", "p_away_close_power",
    ]].rename(columns={"gameId": "cbbd_game_id"})

    out = out.drop_duplicates(subset=["game_id", "provider"])
    out.to_parquet(OUT_PATH, index=False)

    report = {
        "seasons": SEASONS,
        "raw_line_rows_2023_2025": int(n_line_rows_raw),
        "joined_line_rows": int(len(joined)),
        "join_rate": join_rate,
        "unmatched_gameIds_sample": (
            merged[merged["_merge"] == "left_only"]["gameId"].drop_duplicates().head(10).tolist()
        ),
        "output_rows": int(len(out)),
        "output_path": str(OUT_PATH.relative_to(ROOT)),
        "rows_by_season_provider": (
            out.groupby(["season", "provider"]).size().rename("rows").reset_index().to_dict("records")
        ),
        "de_vig_methods": ["proportional (primary)", "power (reported alongside)"],
        "close_line_caveat": (
            "CBBD /lines carries no line-level timestamp; 'close' means the only/last "
            "value on file per docs/tests/lines_cbbd_validation_2026-09-10.md section 2, "
            "not a timestamp-verified pre-tipoff snapshot."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print(f"raw line rows (2023-2025, non-null provider): {n_line_rows_raw}")
    print(f"joined to game_id (games_universe_v2): {len(joined)} ({join_rate*100:.2f}%)")
    print(f"output rows: {len(out)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
