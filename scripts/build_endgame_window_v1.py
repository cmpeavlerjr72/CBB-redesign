"""build_endgame_window_v1.py -- window rows for the late-game round 7 endgame bake-off (late_game/experiments.md s15).

DATA ASSEMBLY ONLY: no model is fitted here. Trainers (`train_endgame_*_v1.py`) and the ONE blind grader
(`grade_endgame_v1.py`) both read these frames, so every arm is scored on exactly the same rows.

Window (s15.A.2, frozen): regulation, periods 1-2, possession START `seconds_remaining <= 240`, `score_diff` in the
offence's perspective at the possession start. Block F keeps section 29's own window (period 2, start <= 120 s).

Cells (s15.A.2): role = leader / trailer / tied (sign of the offence's start margin); band of |score_diff| =
0-3, 4-7, 8-12, 13+; time bucket (180,240], (120,180], (60,120], (30,60], (0,30] (a start at 0 s is in (0,30]).
Role-band code: leader band b -> b, trailer band b -> 4 + b, tied -> 8 (nine role-bands).

Sources (seasons 2022-2025 only; 2025-26 is never read):
  C  clock design `models/clock/r6_L2/design_v2.parquet` + chances_v4 first-chance duration d1 and chance-1 end
     class, exactly as the served K2 trainer builds them (`train_clock_r8_chance_v1.build_d1_design`).
  E  possession_outcome round-2 design, FIRST chances (`population == 'first'`), the served event model's rows.
  F  `possession_outcome/round6/foul_accrual_poss_v2.parquet` (section 29's engine-definition counts).

    .venv/Scripts/python.exe scripts/build_endgame_window_v1.py      # builds the three caches
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

CACHE = ROOT / "results/endgame/data"
SEASONS = (2022, 2023, 2024, 2025)
FOLDS = {"F1": ((2022, 2023), 2024), "F2": ((2022, 2023, 2024), 2025)}
WIN_SEC = 240
BUCKET_LABELS = ("(180,240]", "(120,180]", "(60,120]", "(30,60]", "(0,30]")
BAND_LABELS = ("0-3", "4-7", "8-12", "13+")
ROLEBAND_LABELS = tuple([f"leader {b}" for b in BAND_LABELS] + [f"trailer {b}" for b in BAND_LABELS] + ["tied"])
N_RB, N_TB = 9, 5


def band(a) -> np.ndarray:
    a = np.abs(np.asarray(a))
    return np.select([a <= 3, a <= 7, a <= 12], [0, 1, 2], 3).astype(np.int64)


def roleband(m) -> np.ndarray:
    m = np.asarray(m)
    b = band(m)
    return np.where(m > 0, b, np.where(m < 0, 4 + b, 8)).astype(np.int64)


def role(m) -> np.ndarray:
    """1 leader, -1 trailer, 0 tied."""
    return np.sign(np.asarray(m)).astype(np.int64)


def tbucket(sec) -> np.ndarray:
    s = np.asarray(sec)
    return np.select([s > 180, s > 120, s > 60, s > 30], [0, 1, 2, 3], 4).astype(np.int64)


def _seal(df: pd.DataFrame) -> pd.DataFrame:
    if (df["season"] >= 2026).any():
        raise SystemExit("2025-26 is SEALED: a window frame carried a 2026 row")
    return df


def clock_window(rebuild: bool = False) -> pd.DataFrame:
    p = CACHE / "clock_window.parquet"
    if p.exists() and not rebuild:
        return pd.read_parquet(p)
    import train_clock_r8_chance_v1 as C
    import train_clock_r8_tempo_v1 as T
    d = T.load_design()
    first, _ = C.chance_tables(T.ALL_SEASONS)
    d = C.build_d1_design(d, first)
    d = d[(d["period"] <= 2) & (d["seconds_remaining"] <= WIN_SEC) & (d["is_ot"] == 0)
          & d["season"].isin(SEASONS)].reset_index(drop=True)
    d = _seal(d)
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(p, index=False)
    return d


def event_window(rebuild: bool = False) -> pd.DataFrame:
    p = CACHE / "event_window.parquet"
    if p.exists() and not rebuild:
        return pd.read_parquet(p)
    d = pd.read_parquet(ROOT / "data/processed/models/possession_outcome/round2/design.parquet")
    d = d[(d["population"] == "first") & (d["period"] <= 2) & (d["start_clock"] <= WIN_SEC)
          & d["season"].isin(SEASONS)].reset_index(drop=True)
    d = _seal(d)
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(p, index=False)
    return d


def foul_window(rebuild: bool = False) -> pd.DataFrame:
    p = CACHE / "foul_window.parquet"
    if p.exists() and not rebuild:
        return pd.read_parquet(p)
    d = pd.read_parquet(ROOT / "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet")
    d = d[(d["period"] == 2) & (d["start_clock"] <= 120) & (d["is_ot"] == 0) & d["season"].isin(SEASONS)]
    d = _seal(d.reset_index(drop=True))
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(p, index=False)
    return d


def split(d: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) for a fold; asserts every training row precedes every test tipoff (created_at < tipoff)."""
    trs, te = FOLDS[fold]
    tr = d[d["season"].isin(trs)].reset_index(drop=True)
    ts = d[d["season"] == te].reset_index(drop=True)
    if len(tr) and len(ts) and pd.to_datetime(tr["game_date"]).max() >= pd.to_datetime(ts["game_date"]).min():
        raise SystemExit(f"{fold}: a training row is not strictly before the test season")
    return tr, ts


if __name__ == "__main__":
    for nm, fn in (("clock", clock_window), ("event", event_window), ("foul", foul_window)):
        x = fn(rebuild=True)
        print(nm, len(x), x.groupby("season").size().to_dict(), flush=True)
