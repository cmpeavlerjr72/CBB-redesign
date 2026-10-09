"""Trajectory side-channel: writer round-trip, schema, and (when engine inputs exist) on/off identity + score replay."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbb_sim.engine import trajectory as TJ

ROOT = Path(__file__).resolve().parents[1]


def _fake(game_id=1, seed=0, n=4):
    pts = np.array([2, 0, 3, 1], dtype=np.int8)[:n]
    side = np.array([0, 1, 0, 1], dtype=np.int8)[:n]
    h = np.cumsum(np.where(side == 0, pts, 0)).astype(np.int16)
    a = np.cumsum(np.where(side == 1, pts, 0)).astype(np.int16)
    d = {c: np.zeros(n, dtype=np.int8) for c in TJ.COLUMNS}
    df = pd.DataFrame(d)
    df["game_id"], df["seed"] = np.int64(game_id), np.int32(seed)
    df["poss_idx"] = np.arange(n, dtype=np.int16)
    df["points"], df["off_side"], df["home_score"], df["away_score"] = pts, side, h, a
    df["margin"] = h - a
    df["outcome"] = pd.Categorical(["FG2_MAKE", "MISS", "FG3_MAKE", "FT_TRIP"][:n], categories=list(TJ.OUTCOMES))
    df["shooter_id"] = np.int64(-1)
    df["off_team_id"] = np.int64(7)
    return df[list(TJ.COLUMNS)]


def test_writer_roundtrip_and_chunks(tmp_path):
    w = TJ.TrajectoryWriter(tmp_path / "x" / "t.parquet", seed_limit=5)
    assert w.seed_limit == 5
    w.write(_fake(1, 0))
    w.write(_fake(1, 1))
    w.write(pd.DataFrame())            # empty chunk is a no-op
    w.close()
    out = pd.read_parquet(tmp_path / "x" / "t.parquet")
    assert list(out.columns) == list(TJ.COLUMNS)
    assert len(out) == 8 and w.rows == 8 and w.pairs == 2
    assert out.outcome.dtype.name == "category"


def test_replay_of_points_reproduces_score():
    df = _fake()
    assert df.loc[df.off_side == 0, "points"].sum() == df.home_score.iloc[-1] == 5
    assert df.loc[df.off_side == 1, "points"].sum() == df.away_score.iloc[-1] == 1


def test_recorder_off_by_default(monkeypatch):
    monkeypatch.delenv("CBB_TRAJECTORY", raising=False)
    assert TJ.make_recorder(None, None, None) is None


def test_env_seed_limit(monkeypatch):
    monkeypatch.setenv("CBB_TRAJECTORY_SEEDS", "20")
    assert TJ.env_seed_limit() == 20
    monkeypatch.delenv("CBB_TRAJECTORY_SEEDS")
    assert TJ.env_seed_limit() == 0


@pytest.mark.skipif(not (ROOT / "data/processed/models/engine_v3/arrays_F2_2025.npz").exists(), reason="engine inputs absent")
def test_engine_on_off_identical_and_replay():
    """Run in a subprocess: other tests re-point module globals in cbb_sim.engine.adapters (ENGINE_DIR), which must not
    leak into a served-default engine run. The script asserts games/players max abs diff 0.0 and the per-game-seed score replay."""
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if not k.startswith("ENGINE_")}
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/diag_trajectory_onoff_v1.py"), "--games", "3", "--seeds", "2"],
                       cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    assert "max abs diff=0.0" in p.stdout and "mismatch': 0" in p.stdout
