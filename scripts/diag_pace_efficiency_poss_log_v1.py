"""diag_pace_efficiency_poss_log_v1.py -- opt-in per-possession state log for
the pace/efficiency sign diagnosis (lane: engine, 2026-09-11).

The engine writes no per-possession log and NOTHING IN `src/cbb_sim/` IS
CHANGED BY THIS SCRIPT.  Instead the three adapters whose calls carry the
state block are wrapped in this process only, and `simulate_chunk` is run ONE
SEED AT A TIME so that the `game_index` each adapter already receives
identifies the simulation row uniquely.  Nothing is monkeypatched inside
`loop.py`; the wrappers delegate to the real adapter and return its value
unchanged, so the simulation is bit-identical to a normal run.

Recorded per possession: the drawn duration, the `prev_end` state, the
`is_transition` the loop derives from the duration, the `chance_elapsed_s` the
loop serves, the event model's class probabilities, and fg_make's served make
probability per attempt.

Usage:  diag_pace_efficiency_poss_log_v1.py [n_games] [n_seeds] [fold] [season]
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
    "ENGINE_EVENT": "round2_s1", "ENGINE_CLOCK": "v3c_srfloor_P3_s1",
    "ENGINE_ROTATION": "reference", "ENGINE_FG3": "decision8",
    "ENGINE_FG_MAKE": "round4_B1", "ENGINE_REBOUND": "s1_weekly",
    "ENGINE_FREE_THROW": "s1_conf_aligned", "ENGINE_ROTATION_SCHEME": "s1",
    "ENGINE_INPUTS_VERSION": "v2",
}
for k, v in FLAGS.items():
    os.environ[k] = v
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "6"

NGAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NSEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
FOLD = sys.argv[3] if len(sys.argv) > 3 else "F2"
SEASON = int(sys.argv[4]) if len(sys.argv) > 4 else 2025

from cbb_sim.engine import loop as L                                 # noqa: E402
from cbb_sim.engine.adapters import STATE_INDEX, Adapters            # noqa: E402
from cbb_sim.engine.inputs import EngineInputs                       # noqa: E402
from cbb_sim.models import possession_outcome as PO                  # noqa: E402

I = STATE_INDEX
inp = EngineInputs.load(str(ROOT / "data/processed/models/engine"), f"{FOLD}_{SEASON}")
ad = Adapters.load(inp, FOLD, SEASON)

REC: dict[str, list] = {"clock": [], "fg": [], "event": []}
SEED = [0]


class ClockTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def draw(self, team, x, u, gidx=None):
        dur = (self._r.draw(team, x, u, gidx) if gidx is not None
               else self._r.draw(team, x, u))
        g = gidx if gidx is not None else np.full(len(x), -1, dtype=np.int64)
        REC["clock"].append(np.column_stack([
            np.full(len(x), SEED[0]), g, np.asarray(dur, float),
            x[:, I["prev_end_DREB"]], x[:, I["prev_end_TOV"]],
            x[:, I["prev_end_made_FG"]], x[:, I["prev_end_made_FT"]],
            x[:, I["prev_end_other"]], x[:, I["period"]],
            x[:, I["seconds_remaining"]]]))
        return dur


class EventTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def predict(self, team, state, is_first, gidx, off):
        p = self._r.predict(team, state, is_first, gidx, off)
        REC["event"].append(np.column_stack([
            np.full(len(state), SEED[0]), gidx, np.asarray(is_first, float),
            state[:, I["is_transition"]], state[:, I["chance_elapsed_s"]],
            state[:, I["chance_number"]], np.asarray(p, float)]))
        return p


CLS_CODE = {"FGA_rim": 0, "FGA_jump2": 1, "FGA_3": 2}


class FgTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def predict(self, cls_name, team, slot, state, gidx):
        p = self._r.predict(cls_name, team, slot, state, gidx)
        REC["fg"].append(np.column_stack([
            np.full(len(state), SEED[0]), gidx,
            np.full(len(state), CLS_CODE[cls_name], dtype=float),
            state[:, I["is_transition_f"]], state[:, I["chance_elapsed_s"]],
            state[:, I["chance_number"]], np.asarray(p, float)]))
        return p


ad.clock = ClockTap(ad.clock)
ad.event = EventTap(ad.event)
ad.fg = FgTap(ad.fg)

rng = np.random.default_rng(7)
rows = np.sort(rng.choice(len(inp.games), size=NGAMES, replace=False)).astype(np.int64)
print("=== per-possession tap: %d games x %d seeds, fold %s season %d ==="
      % (NGAMES, NSEEDS, FOLD, SEASON))
t0 = time.time()
games = []
for s in range(NSEEDS):
    SEED[0] = s
    res = L.simulate_chunk(inp, ad, rows, np.full(NGAMES, s, dtype=np.int64),
                           keep_players=False)
    games.append(res.games.assign(gidx=rows))
print("  simulated in %.1f s" % (time.time() - t0))

G = pd.concat(games, ignore_index=True)
clk = pd.DataFrame(np.vstack(REC["clock"]),
                   columns=["seed", "gidx", "dur", "p_DREB", "p_TOV", "p_madeFG",
                            "p_madeFT", "p_other", "period", "sec"])
ev = pd.DataFrame(np.vstack(REC["event"]),
                  columns=["seed", "gidx", "is_first", "trans", "elapsed",
                           "chance"] + list(PO.CLASSES))
fg = pd.DataFrame(np.vstack(REC["fg"]),
                  columns=["seed", "gidx", "cls", "trans_f", "elapsed", "chance", "p"])
print("  possessions logged %d   chances %d   attempts %d"
      % (len(clk), len(ev), len(fg)))

# ------------------------------------------------------------- 1. cells
W3 = np.where(fg["cls"].values == 2, 1.5, 1.0)
fg["efg_num"] = fg["p"] * W3
clk["prev"] = np.select(
    [clk.p_DREB > 0, clk.p_TOV > 0, clk.p_madeFG > 0, clk.p_madeFT > 0],
    ["DREB", "TOV", "made_FG", "made_FT"], "other")

print("\n-- SIM: duration and served eFG by prev_end (regulation) --")
c = clk[clk["period"] <= 2]
t = c.groupby("prev").agg(n=("dur", "size"), dur=("dur", "mean"))
t["share"] = t["n"] / t["n"].sum()
print(t.round(4).to_string())
print("  ACTUAL for the same cells: made_FG 21.661 (.3696) DREB 14.387 (.3503) "
      "TOV 14.738 (.1704) made_FT 18.644 (.0900)")

print("\n-- SIM: served eFG by is_transition_f (fg_make's own served rows) --")
t = fg.groupby("trans_f").agg(att=("p", "size"), efg=("efg_num", "mean"),
                              elapsed=("elapsed", "mean"))
t["share"] = t["att"] / t["att"].sum()
print(t.round(4).to_string())
print("  ACTUAL (POST-OUTCOME, L5-banned, cautionary only), chance 1, prev_end "
      "in (DREB,TOV): elapsed<=8 eFG 0.7531, elapsed>8 eFG 0.4280; the HONEST "
      "design-column lift is +6.5 pp")

print("\n-- SIM: served eFG by chance_elapsed_s bucket, first chance --")
f1 = fg[fg["chance"] == 1].copy()
f1["b"] = pd.cut(f1["elapsed"], [-0.1, 4, 8, 12, 16, 20, 25, 30, 1e9])
t = f1.groupby("b", observed=True).agg(att=("p", "size"), efg=("efg_num", "mean"),
                                       dur=("elapsed", "mean"))
t["share"] = t["att"] / t["att"].sum()
t["ACTUAL_efg"] = [0.9351, 0.6648, 0.4679, 0.5017, 0.4485, 0.4012, 0.3083, 0.1670]
t["gap_pp"] = 100 * (t["efg"] - t["ACTUAL_efg"])
print(t.round(4).to_string())
print("  (ACTUAL column = chance 1, prev_end DREB/TOV, from "
      "diag_pace_efficiency_probe_v1.py part B; the sim column pools prev_end)")
print("  WARNING: that ACTUAL column is built on `chances.duration_s`, which is"
      " POST-OUTCOME (it runs to the REBOUND on a miss and the MAKE on a make)"
      " and is the L5-banned quantity. It is kept only as the cautionary"
      " comparison -- the honest transition lift measured on fg_make's own"
      " `chance_elapsed_s` is +6.5 pp, not +32.5 pp. See"
      " docs/tests/pace_efficiency_sign_2026-09-11.md section 3.2.")

print("\n-- SIM: event-model shot mix by is_transition, first chance --")
e1 = ev[ev["is_first"] > 0]
t = e1.groupby("trans")[list(PO.CLASSES)].mean()
fga = t[["FGA_rim", "FGA_jump2", "FGA_3"]].sum(axis=1)
t["FGA_per_poss"] = fga
t["rim_share"] = t["FGA_rim"] / fga
print(t.round(4).to_string())
print("  ACTUAL: transition FGA/poss 0.6248 eFG 0.8195 ; half court 0.8852 eFG 0.4676")

# ------------------------------------------- 2. per (game, seed) aggregates
g1 = clk[clk["period"] <= 2].groupby(["seed", "gidx"]).agg(
    P=("dur", "size"), mdur=("dur", "mean"), short=("dur", lambda s: (s <= 8).mean()))
g2 = ev[ev["is_first"] > 0].groupby(["seed", "gidx"]).agg(trans=("trans", "mean"))
g3 = fg.groupby(["seed", "gidx"]).agg(att=("p", "size"), num=("efg_num", "sum"),
                                      tshare=("trans_f", "mean"))
g3["efg"] = g3["num"] / g3["att"]
d = g1.join(g2).join(g3).reset_index()
real = G.rename(columns={"seed": "seed"})[["seed", "gidx", "possessions", "home_pts",
                                           "away_pts"]]
d = d.merge(real, on=["seed", "gidx"], how="left")


def within(df, cols):
    return df[cols] - df.groupby("gidx")[cols].transform("mean")


def rep(x, y, label):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    print("  %-52s corr %+0.4f  slope %+0.6f  n %d"
          % (label, np.corrcoef(x[m], y[m])[0, 1], np.polyfit(x[m], y[m], 1)[0], m.sum()))


w = within(d, ["P", "trans", "efg", "mdur", "short", "tshare"])
print("\n-- SIM within-game (across-seed) correlations, %d games x %d seeds --"
      % (NGAMES, NSEEDS))
rep(w["P"], w["efg"], "corr(P, served eFG)")
rep(w["P"], w["trans"], "corr(P, transition share)      [ACTUAL +0.5997]")
rep(w["P"], w["short"], "corr(P, share of durations <= 8 s)")
rep(w["P"], w["mdur"], "corr(P, mean possession duration)")
rep(w["trans"], w["efg"], "corr(transition share, served eFG)")
print("  SD within game: P %.3f  trans share %.4f  eFG %.4f"
      % (w["P"].std(ddof=0), w["trans"].std(ddof=0), w["efg"].std(ddof=0)))
print("  mean transition share SIM %.4f   ACTUAL %.4f" % (d["trans"].mean(), 0.1585))

out = ROOT / "results" / "engine_v0" / "poss_log_pace_efficiency_2026-09-11"
out.mkdir(parents=True, exist_ok=True)
d.to_parquet(out / "per_game_seed.parquet", index=False)
(f1.drop(columns=["b"])                      # the pd.cut interval is not a
   .sample(min(200000, len(f1)), random_state=3)   # parquet-writable dtype
   .to_parquet(out / "fg_rows_sample.parquet", index=False))
print("\nwrote %s" % out)
