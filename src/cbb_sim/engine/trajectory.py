"""
trajectory.py -- opt-in per-possession trajectory side-channel for the possession loop.

One row per possession per simulated (game, seed). INERT BY CONSTRUCTION: the recorder only COPIES and SUBTRACTS
arrays `loop.simulate_chunk` has already computed. It makes no RNG draw, calls no adapter, mutates no GameState
field and no engine array, so every game and player output is identical with it on or off
(`tests/test_trajectory.py`, `scripts/diag_trajectory_onoff_v1.py`). Default OFF.

Switch: `simulate_chunk(..., trajectory_writer=W)` (W = `TrajectoryWriter`, a parquet sink) and/or env
`CBB_TRAJECTORY=1` (+ `CBB_TRAJECTORY_SEEDS=N` to record only seeds < N). The frame lands on `ChunkResult.trajectory`.
Design: docs/ops/trajectory_cbb_design_2026-10-09.md.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.engine import state as S

#: box counters diffed over one possession, in this order.
_CNT = ("fga2_rim", "fga2_jump", "fga3", "fgm2_rim", "fgm2_jump", "fgm3", "fta", "ftm", "tov", "oreb")
OUTCOMES = ("TOV", "FG3_MAKE", "FG2_MAKE", "FT_TRIP", "MISS")

COLUMNS = (
    "game_id", "seed", "poss_idx", "period", "clock_start", "clock_end", "game_clock_sec_remaining",
    "off_side", "off_team_id", "duration_s", "end_type", "outcome", "n_chances", "n_oreb",
    "fga2", "fgm2", "fga3", "fgm3", "fta", "ftm", "tov", "points",
    "home_score", "away_score", "margin", "shooter_slot", "shooter_id",
    "off_team_fouls", "def_team_fouls", "off_in_bonus", "off_in_dbl_bonus",
    "bonus_prior_fouls", "dbl_bonus_prior_fouls",
)


def env_enabled() -> bool:
    return os.environ.get("CBB_TRAJECTORY", "0") == "1"


def env_seed_limit() -> int:
    v = os.environ.get("CBB_TRAJECTORY_SEEDS", "")
    return int(v) if v.strip() else 0


class TrajectoryRecorder:
    """Per-chunk collector. `keep` is an (n,) bool mask over simulations (seed subset)."""

    def __init__(self, st: S.GameState, gids: np.ndarray, keep: np.ndarray | None = None):
        n = st.n
        self.gids = np.asarray(gids, dtype=np.int64)
        self.keep = np.ones(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
        self.cnt = np.zeros(n, dtype=np.int32)           # possession ordinal per simulation
        self.steps: list[dict] = []
        self._open: dict | None = None
        self._shooter: np.ndarray | None = None

    # -- hook 1: top of the step, before any draw -------------------------------------------------------------
    def open(self, st: S.GameState, act: np.ndarray, off: np.ndarray) -> None:
        dfn = 1 - off
        self._open = {
            "act": act, "off": off,
            "period": st.period[act].copy(), "sec": st.seconds_remaining[act].copy(),
            "usage_sec": st.usage_sec_remaining()[act].copy(),
            "pts0": st.pts[act].copy(),
            "tf_off": st.team_fouls[act, off].copy(), "tf_def": st.team_fouls[act, dfn].copy(),
            "bonus": st.in_bonus()[act].copy(), "dbonus": st.in_double_bonus()[act].copy(),
            "box0": np.stack([st.box[k][act, off] for k in _CNT], axis=1).astype(np.int16),
        }
        self._shooter = np.full(len(act), -1, dtype=np.int16)

    # -- hook 2: inside the chance loop, after the shooter array exists -----------------------------------------
    def note_shooter(self, rows: np.ndarray, shooter: np.ndarray) -> None:
        self._shooter[rows] = shooter

    # -- hook 3: after the possession bookkeeping, before the period advances ------------------------------------
    def close(self, st: S.GameState, used: np.ndarray, end_code: np.ndarray, chance: np.ndarray) -> None:
        o = self._open
        act, off = o["act"], o["off"]
        k = self.keep[act]
        if not k.any():
            self._open = None
            return
        idx = act[k]
        ar = np.arange(len(act))
        box1 = np.stack([st.box[c][act, off] for c in _CNT], axis=1).astype(np.int16)
        d = (box1 - o["box0"])[k]
        pts1 = st.pts[act]
        rec = {
            "sim": idx.astype(np.int32),
            "poss_idx": self.cnt[idx].astype(np.int16),
            "period": o["period"][k].astype(np.int8),
            "clock_start": o["sec"][k].astype(np.int16),
            "clock_end": st.seconds_remaining[act][k].astype(np.int16),
            "game_clock_sec_remaining": o["usage_sec"][k].astype(np.int16),
            "off_side": off[k].astype(np.int8),
            "duration_s": np.asarray(used)[k].astype(np.int16),
            "end_type": np.asarray(end_code)[k].astype(np.int8),
            "n_chances": np.asarray(chance)[k].astype(np.int8),
            "d": d,
            "home_score": pts1[k, 0].astype(np.int16), "away_score": pts1[k, 1].astype(np.int16),
            "points": (pts1[ar, off] - o["pts0"][ar, off])[k].astype(np.int8),
            "shooter_slot": self._shooter[k].astype(np.int8),
            "off_team_fouls": o["tf_off"][k].astype(np.int8), "def_team_fouls": o["tf_def"][k].astype(np.int8),
            "off_in_bonus": o["bonus"][k].astype(np.int8), "off_in_dbl_bonus": o["dbonus"][k].astype(np.int8),
            "bonus_prior_fouls": st.bonus_prior_fouls[idx], "dbl_bonus_prior_fouls": st.double_bonus_prior_fouls[idx],
        }
        self.cnt[idx] += 1
        self.steps.append(rec)
        self._open = None

    # -- flush -------------------------------------------------------------------------------------------------
    def frame(self, st: S.GameState, inp) -> pd.DataFrame:
        if not self.steps:
            return pd.DataFrame({c: [] for c in COLUMNS})

        def cat(key):
            return np.concatenate([s[key] for s in self.steps])

        sim = cat("sim")
        d = np.concatenate([s["d"] for s in self.steps], axis=0)
        j = {c: i for i, c in enumerate(_CNT)}
        fga2 = d[:, j["fga2_rim"]] + d[:, j["fga2_jump"]]
        fgm2 = d[:, j["fgm2_rim"]] + d[:, j["fgm2_jump"]]
        fga3, fgm3, fta = d[:, j["fga3"]], d[:, j["fgm3"]], d[:, j["fta"]]
        ftm, tov = d[:, j["ftm"]], d[:, j["tov"]]
        outcome = np.select([tov > 0, fta > 0, fgm3 > 0, fgm2 > 0],
                            [OUTCOMES[0], OUTCOMES[3], OUTCOMES[1], OUTCOMES[2]], OUTCOMES[4])
        off_side = cat("off_side")
        gi = st.game_index[sim].astype(np.int64)
        shooter_slot = cat("shooter_slot")
        sid = np.where(shooter_slot >= 0,
                       inp.roster_espn[gi, off_side.astype(np.int64), np.maximum(shooter_slot, 0).astype(np.int64)], -1)
        home_t = inp.games["home_team_id"].to_numpy()[gi]
        away_t = inp.games["away_team_id"].to_numpy()[gi]
        hs, as_ = cat("home_score"), cat("away_score")
        df = pd.DataFrame({
            "game_id": self.gids[sim], "seed": st.seed[sim].astype(np.int32),
            "poss_idx": cat("poss_idx"), "period": cat("period"),
            "clock_start": cat("clock_start"), "clock_end": cat("clock_end"),
            "game_clock_sec_remaining": cat("game_clock_sec_remaining"),
            "off_side": off_side, "off_team_id": np.where(off_side == 0, home_t, away_t).astype(np.int64),
            "duration_s": cat("duration_s"), "end_type": cat("end_type"),
            "outcome": pd.Categorical(outcome, categories=list(OUTCOMES)),
            "n_chances": cat("n_chances"), "n_oreb": d[:, j["oreb"]].astype(np.int8),
            "fga2": fga2.astype(np.int8), "fgm2": fgm2.astype(np.int8), "fga3": fga3.astype(np.int8),
            "fgm3": fgm3.astype(np.int8), "fta": fta.astype(np.int8), "ftm": ftm.astype(np.int8),
            "tov": tov.astype(np.int8), "points": cat("points"),
            "home_score": hs, "away_score": as_, "margin": (hs.astype(np.int16) - as_.astype(np.int16)),
            "shooter_slot": shooter_slot, "shooter_id": sid.astype(np.int64),
            "off_team_fouls": cat("off_team_fouls"), "def_team_fouls": cat("def_team_fouls"),
            "off_in_bonus": cat("off_in_bonus"), "off_in_dbl_bonus": cat("off_in_dbl_bonus"),
            "bonus_prior_fouls": cat("bonus_prior_fouls"), "dbl_bonus_prior_fouls": cat("dbl_bonus_prior_fouls"),
        })
        df["_sim"] = sim
        df = df.sort_values(["_sim", "poss_idx"], kind="stable").drop(columns="_sim").reset_index(drop=True)
        return df[list(COLUMNS)]


def make_recorder(st: S.GameState, gids: np.ndarray, trajectory_writer=None, seed_limit: int = 0):
    """None unless a writer was passed or CBB_TRAJECTORY=1. `seed_limit` (arg, else writer, else env) keeps seeds < N."""
    if trajectory_writer is None and not env_enabled():
        return None
    lim = seed_limit or (trajectory_writer.seed_limit if trajectory_writer is not None else 0) or env_seed_limit()
    keep = (st.seed < lim) if lim else None
    return TrajectoryRecorder(st, gids, keep)


class TrajectoryWriter:
    """One parquet file per run (zstd, one row group per chunk). `seed_limit` > 0 records only seeds < N."""

    def __init__(self, path: str | Path, seed_limit: int = 0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seed_limit = int(seed_limit)
        self._w = None
        self.rows = 0
        self.pairs = 0

    def write(self, df: pd.DataFrame) -> None:
        if df is None or not len(df):
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        t = pa.Table.from_pandas(df, preserve_index=False)
        if self._w is None:
            self._w = pq.ParquetWriter(self.path, t.schema, compression="zstd", compression_level=9)
        self._w.write_table(t.cast(self._w.schema))
        self.rows += len(df)
        self.pairs += int(df[["game_id", "seed"]].drop_duplicates().shape[0])

    def close(self) -> None:
        if self._w is not None:
            self._w.close()
            self._w = None
