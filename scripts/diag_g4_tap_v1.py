"""diag_g4_tap_v1.py -- instrumented sim tap for the G4 OREB%/FTA-FGA lane.

MEASUREMENT ONLY. NOTHING IN `src/cbb_sim/` IS CHANGED BY THIS SCRIPT.  It
follows the pattern `diag_late_game_tap_v1.py` / `diag_pace_efficiency_poss_
log_v1.py` established: the adapters and two loop-level helpers are wrapped IN
THIS PROCESS ONLY, one seed per chunk so `game_index` identifies the
simulation row, and every wrapper delegates to the real callable and returns
its value unchanged, so the simulation is bit-identical to a normal run.  That
claim is checked, not asserted: `diag_g4_report_v1.py` compares this tap's
`games.parquet` against the served 75-seed run row for row.

Flags are pinned to the SERVED stack (`ENGINE_CLOCK=v5b_glat_pmean`, adopted
2026-09-11), i.e. the stack the v5b gate read used.

Recorded:
  chances.parquet  one row per chance: gidx, state, drawn possession-outcome
                   class.
  trips.parquet    one row per free-throw trip (and-one trips included):
                   gidx, state, trip kind, n_att, one-and-one flag, attempts
                   actually taken.
  rebs.parquet     one row per rebound opportunity: gidx, state, miss type,
                   the model's OREB probability AS THE ENGINE FEEDS IT
                   (blocked_f = 0), and the drawn outcome.

Usage: diag_g4_tap_v1.py [n_games] [n_seeds] [out_dir] [seed0]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FLAGS = {
    "ENGINE_EVENT": "round2_s1", "ENGINE_CLOCK": "v5b_glat_pmean",
    "ENGINE_ROTATION": "reference", "ENGINE_FG3": "decision8",
    "ENGINE_FG_MAKE": "round4_B1", "ENGINE_REBOUND": "s1_weekly",
    "ENGINE_FREE_THROW": "s1_conf_aligned", "ENGINE_ROTATION_SCHEME": "s1",
    "ENGINE_INPUTS_VERSION": "v2",
}
for k, v in FLAGS.items():
    os.environ[k] = v
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[k] = "1"

NGAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 500
NSEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUTD = Path(sys.argv[3] if len(sys.argv) > 3 else "results/g4_diag/tap")
SEED0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
OUTD.mkdir(parents=True, exist_ok=True)

from cbb_sim.engine import loop as L                                 # noqa: E402
from cbb_sim.engine.adapters import STATE_INDEX, Adapters            # noqa: E402
from cbb_sim.engine.inputs import EngineInputs                       # noqa: E402
from cbb_sim.models import possession_outcome as PO                  # noqa: E402
from cbb_sim.models import rebound as RB                             # noqa: E402

I = STATE_INDEX
N_PO = len(PO.CLASSES)
N_RB = len(RB.CLASSES)
MISS_COLS = ("miss_rim", "miss_jump2", "miss_three")

inp = EngineInputs.load(str(ROOT / "data/processed/models/engine"), "F2_2025")
ad = Adapters.load(inp, "F2", 2025)

REC: dict[str, list] = {"chance": [], "trip": [], "reb": []}
CUR: dict = {"seed": 0, "ev_gidx": None, "ev_state": None,
             "rb_gidx": None, "rb_state": None, "rb_p": None,
             "cls_by_gidx": None}
NG = len(inp.games)


class EventTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def predict(self, team, state, is_first, gidx, off):
        p = self._r.predict(team, state, is_first, gidx, off)
        CUR["ev_gidx"] = np.asarray(gidx, dtype=np.int64)
        CUR["ev_state"] = np.column_stack([
            np.asarray(is_first, float), state[:, I["period"]],
            state[:, I["seconds_remaining"]], state[:, I["score_diff"]],
            state[:, I["in_bonus"]], state[:, I["chance_number"]]])
        return p


class RebTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def predict(self, team, state, gidx):
        p = self._r.predict(team, state, gidx)
        CUR["rb_gidx"] = np.asarray(gidx, dtype=np.int64)
        # miss type as the engine encodes it, plus the state the model sees
        mt = np.zeros(len(state), dtype=np.int64) + 3          # 3 == "ft"
        for j, c in enumerate(MISS_COLS):
            mt = np.where(state[:, I[c]] > 0.5, j, mt)
        CUR["rb_state"] = np.column_stack([
            mt.astype(float), state[:, I["period"]],
            state[:, I["seconds_remaining"]], state[:, I["score_diff"]],
            state[:, I["in_bonus"]], state[:, I["blocked_f"]]])
        pp = np.asarray(p, float)
        CUR["rb_p"] = pp[:, RB.CLASS_INDEX["OREB"]] / np.maximum(
            pp[:, RB.CLASS_INDEX["OREB"]] + pp[:, RB.CLASS_INDEX["DREB"]], 1e-12)
        return p


_real_categorical = L.categorical


def categorical_tap(u, probs):
    out = _real_categorical(u, probs)
    if len(u):
        w = probs.shape[1]
        if w == N_PO and CUR["ev_gidx"] is not None and len(CUR["ev_gidx"]) == len(u):
            g = CUR["ev_gidx"]
            REC["chance"].append(np.column_stack([
                np.full(len(g), CUR["seed"], float), g.astype(float),
                CUR["ev_state"], out.astype(float)]))
            m = np.full(NG, -1, dtype=np.int64)
            m[g] = out
            CUR["cls_by_gidx"] = m
            CUR["ev_gidx"] = None
        elif w == N_RB and CUR["rb_gidx"] is not None and len(CUR["rb_gidx"]) == len(u):
            g = CUR["rb_gidx"]
            REC["reb"].append(np.column_stack([
                np.full(len(g), CUR["seed"], float), g.astype(float),
                CUR["rb_state"], CUR["rb_p"], out.astype(float)]))
            CUR["rb_gidx"] = None
    return out


_real_shoot_trip = L._shoot_trip


def shoot_trip_tap(st, inp_, ad_, book, rows, shooter, n_att, one_and_one, n_state):
    g = st.game_index[rows].astype(np.int64)
    side = st.off[rows].astype(np.int64)
    before = st.box["fta"][rows, side].copy()
    per = st.period[rows].astype(float).copy()
    sec = st.seconds_remaining[rows].astype(float).copy()
    sd = st.off_score_diff()[rows].astype(float).copy()
    bon = st.in_bonus()[rows].astype(float).copy()
    dbon = st.in_double_bonus()[rows].astype(float).copy()
    out = _real_shoot_trip(st, inp_, ad_, book, rows, shooter, n_att,
                           one_and_one, n_state)
    after = st.box["fta"][rows, side]
    cls = (CUR["cls_by_gidx"][g].astype(float)
           if CUR["cls_by_gidx"] is not None else np.full(len(g), -1.0))
    REC["trip"].append(np.column_stack([
        np.full(len(g), CUR["seed"], float), g.astype(float), cls, per, sec, sd,
        bon, dbon, np.asarray(n_att, float), np.asarray(one_and_one, float),
        (after - before).astype(float)]))
    return out


ad.event = EventTap(ad.event)
ad.reb = RebTap(ad.reb)
L.categorical = categorical_tap
L._shoot_trip = shoot_trip_tap

# Systematic every-k-th game over the schedule order, so the month / site /
# conference mix of the subset tracks the full slate rather than a lucky draw.
step = max(1, NG // NGAMES)
rows_sel = np.arange(0, NG, step, dtype=np.int64)[:NGAMES]
print(f"=== g4 tap: {len(rows_sel)} games x {NSEEDS} seeds (seed0={SEED0}), served v5b flags ===",
      flush=True)
t0 = time.time()
games = []
for s in range(SEED0, SEED0 + NSEEDS):
    CUR["seed"] = s
    res = L.simulate_chunk(inp, ad, rows_sel,
                           np.full(len(rows_sel), s, dtype=np.int64),
                           keep_players=False)
    games.append(res.games.assign(gidx=rows_sel))
    print(f"  seed {s} done  {time.time() - t0:.1f}s  "
          f"chances {sum(len(x) for x in REC['chance'])}", flush=True)

G = pd.concat(games, ignore_index=True)
ch = pd.DataFrame(np.vstack(REC["chance"]), columns=[
    "seed", "gidx", "is_first", "period", "sec", "sd", "bonus", "chance", "cls"])
tp = pd.DataFrame(np.vstack(REC["trip"]), columns=[
    "seed", "gidx", "cls", "period", "sec", "sd", "bonus", "dbonus",
    "n_att", "one_and_one", "fta"])
rb = pd.DataFrame(np.vstack(REC["reb"]), columns=[
    "seed", "gidx", "miss_type", "period", "sec", "sd", "bonus", "blocked_f",
    "p_oreb", "outcome"])
G.to_parquet(OUTD / f"games_{SEED0}.parquet")
ch.to_parquet(OUTD / f"chances_{SEED0}.parquet")
tp.to_parquet(OUTD / f"trips_{SEED0}.parquet")
rb.to_parquet(OUTD / f"rebs_{SEED0}.parquet")
print(f"  sims {len(G)}  chances {len(ch)}  trips {len(tp)}  rebs {len(rb)}  "
      f"in {time.time() - t0:.1f}s", flush=True)
