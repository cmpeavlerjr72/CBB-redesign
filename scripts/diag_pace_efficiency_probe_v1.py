"""diag_pace_efficiency_probe_v1.py -- candidate (a)/(c): is the engine's
served fg_make / possession_outcome response to the DURATION-DERIVED state
features (`is_transition_f`, `chance_elapsed_s`) the one the training data has?

Two halves, both read-only, no simulation:

  A. SERVED partial dependence.  Loads `EngineInputs` and `Adapters` with the
     exact flags of `results/engine_v0/F2_2025_s200_rewire1` and sweeps ONLY
     the duration-derived columns of the state block, everything else held at
     a run-typical value drawn from real games.  Gives dp_make/d(feature) per
     shot class for fg_make, and the shot-class mix response for the event
     model.

  B. TRAINED relationship.  The same two features measured on the 2025 chances
     table, which is where the fg_make design's `chance_elapsed_s` and
     `is_transition_f` come from, plus the SERVE-TIME distribution of what the
     engine actually puts in those columns (the drawn POSSESSION duration).

Usage:  diag_pace_efficiency_probe_v1.py [fold] [season]
"""
from __future__ import annotations

import os
import sys
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

FOLD = sys.argv[1] if len(sys.argv) > 1 else "F2"
SEASON = int(sys.argv[2]) if len(sys.argv) > 2 else 2025

from cbb_sim.engine import state as S                                # noqa: E402
from cbb_sim.engine.adapters import STATE_INDEX, Adapters            # noqa: E402
from cbb_sim.engine.inputs import EngineInputs                       # noqa: E402
from cbb_sim.models import fg_make as FG                             # noqa: E402
from cbb_sim.models import possession_outcome as PO                  # noqa: E402

I = STATE_INDEX
inp = EngineInputs.load(str(ROOT / "data" / "processed" / "models" / "engine"), f"{FOLD}_{SEASON}")
ad = Adapters.load(inp, FOLD, SEASON)
print("=== served flags ===")
print({k: ad.flags.get(k) for k in FLAGS})
print("state cols: %d" % len(I))

# ---------------------------------------------------------------- A. probe
rng = np.random.default_rng(11)
NG = 1500
gidx = rng.choice(len(inp.games), size=NG, replace=False).astype(np.int64)
side = np.zeros(NG, dtype=np.int64)
slot = np.zeros(NG, dtype=np.int64)          # slot 0 = the top-rank starter

PREV = S.PREV_END_CODE


def base_block(prev_code: int, sec: float = 900.0, period: float = 1.0,
               chance: float = 1.0, sd: float = 0.0) -> np.ndarray:
    x = np.zeros((NG, len(I)), dtype=np.float64)
    x[:, I["period"]] = period
    x[:, I["seconds_remaining"]] = sec
    x[:, I["score_diff"]] = sd
    x[:, I["score_diff_pre"]] = sd
    x[:, I["x_score_diff__seconds_remaining"]] = sd * sec / 1200.0
    x[:, I["chance_number"]] = chance
    x[:, I["chance_number_at_start"]] = 1.0
    for k, nm in enumerate(("prev_end_DREB", "prev_end_TOV", "prev_end_made_FG",
                            "prev_end_made_FT", "prev_end_other"), start=1):
        x[:, I[nm]] = 1.0 if k == prev_code else 0.0
    return x


def fg_probs(x):
    out = {}
    for key, cls in (("rim", "FGA_rim"), ("jump2", "FGA_jump2"), ("three", "FGA_3")):
        out[key] = ad.fg.predict(cls, inp.team_static[gidx, side],
                                 inp.slot_static[gidx, side, slot], x, gidx).mean()
    return out


print("\n=== A1. fg_make: served response to is_transition_f "
      "(prev_end = DREB, chance 1) ===")
rows = []
for tr in (0.0, 1.0):
    x = base_block(PREV["DREB"])
    x[:, I["is_transition"]] = tr
    x[:, I["is_transition_f"]] = tr
    x[:, I["chance_elapsed_s"]] = 4.0 if tr else 18.0
    p = fg_probs(x)
    p["is_transition_f"] = tr
    rows.append(p)
d = pd.DataFrame(rows).set_index("is_transition_f")
print(d.round(5).to_string())
print("  lift (transition - half court), pp: "
      + "  ".join("%s %+.2f" % (c, 100 * (d[c].iloc[1] - d[c].iloc[0])) for c in d.columns))

print("\n=== A2. fg_make: served response to is_transition_f ALONE "
      "(chance_elapsed_s held at 12 s) ===")
rows = []
for tr in (0.0, 1.0):
    x = base_block(PREV["DREB"])
    x[:, I["is_transition"]] = tr
    x[:, I["is_transition_f"]] = tr
    x[:, I["chance_elapsed_s"]] = 12.0
    p = fg_probs(x)
    p["is_transition_f"] = tr
    rows.append(p)
d2 = pd.DataFrame(rows).set_index("is_transition_f")
print(d2.round(5).to_string())
print("  lift, pp: " + "  ".join("%s %+.2f" % (c, 100 * (d2[c].iloc[1] - d2[c].iloc[0]))
                                 for c in d2.columns))

print("\n=== A3. fg_make: served response to chance_elapsed_s "
      "(is_transition_f consistent, prev_end = DREB) ===")
rows = []
for e in (2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 40):
    x = base_block(PREV["DREB"])
    tr = 1.0 if e <= FG.TRANSITION_MAX_S else 0.0
    x[:, I["is_transition"]] = tr
    x[:, I["is_transition_f"]] = tr
    x[:, I["chance_elapsed_s"]] = float(e)
    p = fg_probs(x)
    p["elapsed_s"] = e
    p["trans"] = tr
    rows.append(p)
d3 = pd.DataFrame(rows).set_index("elapsed_s")
print(d3.round(5).to_string())

print("\n=== A4. fg_make: chance_elapsed_s with is_transition_f PINNED to 0 "
      "(the pure elapsed slope) ===")
rows = []
for e in (2, 4, 8, 12, 16, 20, 25, 30, 40):
    x = base_block(PREV["DREB"])
    x[:, I["chance_elapsed_s"]] = float(e)
    p = fg_probs(x)
    p["elapsed_s"] = e
    rows.append(p)
d4 = pd.DataFrame(rows).set_index("elapsed_s")
print(d4.round(5).to_string())
for c in ("rim", "jump2", "three"):
    b = np.polyfit(d4.index.values.astype(float), d4[c].values, 1)[0]
    print("  d(p_make)/d(elapsed_s) %-6s = %+0.6f per second (%+.3f pp over 10 s)"
          % (c, b, 1000 * b))

print("\n=== A5. fg_make: chance_number (the OREB putback channel) ===")
rows = []
for ch in (1, 2, 3):
    x = base_block(PREV["DREB"], chance=float(ch))
    x[:, I["chance_elapsed_s"]] = 18.0 if ch == 1 else 3.0
    p = fg_probs(x)
    p["chance_number"] = ch
    rows.append(p)
print(pd.DataFrame(rows).set_index("chance_number").round(5).to_string())

print("\n=== A6. possession_outcome (event model): shot-class MIX response to "
      "is_transition ===")
for tr, el in ((0.0, 18.0), (1.0, 4.0)):
    x = base_block(PREV["DREB"])
    x[:, I["is_transition"]] = tr
    x[:, I["is_transition_f"]] = tr
    x[:, I["chance_elapsed_s"]] = el
    pr = ad.event.predict(inp.team_static[gidx, side], x,
                          np.ones(NG, dtype=bool), gidx, side).mean(axis=0)
    nm = {c: i for i, c in enumerate(PO.CLASSES)}
    fga = sum(pr[nm[c]] for c in ("FGA_rim", "FGA_jump2", "FGA_3"))
    print("  is_transition %.0f : " % tr
          + "  ".join("%s %.4f" % (c, pr[nm[c]]) for c in PO.CLASSES)
          + "   | rim share of FGA %.4f  three share %.4f  FGA/poss %.4f"
          % (pr[nm["FGA_rim"]] / fga, pr[nm["FGA_3"]] / fga, fga))

print("\n=== A7. the SERVED eFG of a transition vs a half-court possession, "
      "event mix x fg_make ===")
for tr, el, lbl in ((0.0, 18.0, "half court"), (1.0, 4.0, "transition")):
    x = base_block(PREV["DREB"])
    x[:, I["is_transition"]] = tr
    x[:, I["is_transition_f"]] = tr
    x[:, I["chance_elapsed_s"]] = el
    pr = ad.event.predict(inp.team_static[gidx, side], x,
                          np.ones(NG, dtype=bool), gidx, side).mean(axis=0)
    nm = {c: i for i, c in enumerate(PO.CLASSES)}
    p = fg_probs(x)
    w = {"rim": pr[nm["FGA_rim"]], "jump2": pr[nm["FGA_jump2"]], "three": pr[nm["FGA_3"]]}
    fga = sum(w.values())
    efg = sum(w[k] * p[k] * (1.5 if k == "three" else 1.0) for k in w) / fga
    print("  %-11s eFG %.4f   FGA/poss %.4f   rim share %.4f"
          % (lbl, efg, fga, w["rim"] / fga))

# ---------------------------------------------------------------- B. trained
print("\n\n=== B. the TRAINED relationship, 2025 chances table ===")
c = pd.read_parquet(str(ROOT) + "/data/processed/possessions_v2/chances_%d.parquet" % SEASON)
c = c[c["period"] <= 2].copy()
c["fga"] = c["fga_rim"] + c["fga_jump2"] + c["fga_3"]
c["efg_num"] = c["fgm_rim"] + c["fgm_jump2"] + 1.5 * c["fgm_3"]
c["dur"] = c["duration_s"].astype(float).clip(lower=0)
sh = c[c["fga"] > 0]

print("\n-- chance duration distribution, chance 1 vs 2+ (TRAINED side) --")
print(c.groupby(c["chance_number"].clip(upper=3))["dur"]
      .agg(["size", "mean", "median", "std"]).round(3).to_string())

p = pd.read_parquet(str(ROOT) + "/data/processed/possessions_v2/possessions_%d.parquet" % SEASON,
                    columns=["period", "duration_s", "n_chances", "start_reason"])
p = p[p["period"] <= 2]
print("\n-- what the ENGINE serves into chance_elapsed_s is the POSSESSION "
      "duration; what the model was TRAINED on is the chance's own elapsed --")
print("  possession duration      mean %.3f  median %.1f  P(<=8) %.4f"
      % (p["duration_s"].mean(), p["duration_s"].median(), (p["duration_s"] <= 8).mean()))
c1 = c[c["chance_number"] == 1]
print("  chance-1 elapsed         mean %.3f  median %.1f  P(<=8) %.4f"
      % (c1["dur"].mean(), c1["dur"].median(), (c1["dur"] <= 8).mean()))
print("  all-chance elapsed       mean %.3f  median %.1f  P(<=8) %.4f"
      % (c["dur"].mean(), c["dur"].median(), (c["dur"] <= 8).mean()))
print("  SHIFT served - trained   %+.3f s on chance 1, %+.3f s over all chances"
      % (p["duration_s"].mean() - c1["dur"].mean(),
         p["duration_s"].mean() - c["dur"].mean()))

print("\n-- eFG by chance elapsed, TRAINED side (chance 1, prev_end DREB/TOV) --")
q = sh[(sh["chance_number"] == 1) & (sh["start_reason"].isin(["DREB", "TOV"]))].copy()
q["b"] = pd.cut(q["dur"], [-0.1, 4, 8, 12, 16, 20, 25, 30, 1e9])
t = q.groupby("b", observed=True).agg(n=("fga", "size"), fga=("fga", "sum"),
                                      num=("efg_num", "sum"), dur=("dur", "mean"))
t["efg"] = t["num"] / t["fga"]
t["fga_per_chance"] = t["fga"] / t["n"]
print(t.round(4).to_string())

print("\n-- eFG by chance_number, TRAINED side --")
t2 = sh.groupby(sh["chance_number"].clip(upper=3)).agg(
    n=("fga", "size"), fga=("fga", "sum"), num=("efg_num", "sum"), dur=("dur", "mean"))
t2["efg"] = t2["num"] / t2["fga"]
print(t2.round(4).to_string())
