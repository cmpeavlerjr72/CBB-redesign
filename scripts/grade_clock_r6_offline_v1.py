"""grade_clock_r6_offline_v1.py -- clock round 6 OFFLINE grade (experiments.md section 28.2).

Lane B, 2026-09-30.  One grader for every arm.  Truth = the event-layer v4
table (the like-for-like definition): the fold-2 TEST season (2025) rows of
`data/processed/models/clock/r6_L2/design_v2.parquet` (built from
possessions_v4 by `scripts/exp_clk6_r6_arms_v1.py`), clock-complete games,
regulation.  Each arm's own S1 schedule is routed by game date exactly as the
engine routes it; the quantity is the one `loop.py` subtracts,
E[min(T, R)] = sum_{t<R} t p(t) + R P(T >= R).

Arms: R = served `v3c_srfloor_P3_s1` (the conditional law under the served
v5b; the latent is mean-count preserving and does not move this quantity),
L2 = `v3c_r6L2_srfloor_P3_s1`, L2a = `v3c_r6L2a_srfloor_P3_s1`.

Reported per arm: law gap (model mean consumed - actual, seconds) and the
implied possessions per team-game (1200 / model mean - 1200 / actual mean);
by start type (with the start-type composition); the made-FG cell; by month
(season drift); by team as-of tempo quintile (responsiveness slope).
Offline reseed floor: the served family (`empirical_km3_srfloor`) is not in
`clock_v3.STOCHASTIC_ARMS`, so a spec-identical refit under another seed is
the identical object; the floor is structurally 0 and is stated, not run.
Fold 1 (test 2024) needs an F1 S1 schedule, which the served trainer does
not fit; not run.  Writes results/clock_r6/offline_grade.json.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.engine.clock_adapter_v3 import CK_DIR as _CK, V3C_MODES  # noqa: E402
from cbb_sim.models import clock as CK  # noqa: E402

CKD = ROOT / _CK
OUT = ROOT / "results/clock_r6/offline_grade.json"
ARMS = {"R": "v3c_srfloor_P3_s1", "L2": "v3c_r6L2_srfloor_P3_s1", "L2a": "v3c_r6L2a_srfloor_P3_s1",
        "D1": "v3c_r6D1_srfloor_P3_s1"}
CHUNK = 20000


def load_schedule(mode):
    doc = json.loads((CKD / V3C_MODES[mode]["manifest"]).read_text(encoding="utf-8"))
    out = []
    for m in doc["months"]:
        with open(CKD / m["model_file"], "rb") as f:
            out.append({"refit_date": pd.Timestamp(m["refit_date"]), "arm": pickle.load(f)})
    return sorted(out, key=lambda r: r["refit_date"])


def e_consumed(arm, df):
    grid = np.arange(CK.DURATION_CAP + 1, dtype=np.float64)
    out = np.empty(len(df))
    for lo in range(0, len(df), CHUNK):
        blk = df.iloc[lo:lo + CHUNK]
        pmf = np.asarray(arm.pmf(blk.reset_index(drop=True)), dtype=np.float64)
        r = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
        out[lo:lo + len(blk)] = (pmf * np.minimum(grid[None, :], r)).sum(axis=1)
    return out


def main():
    univ = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet")
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    d = pd.read_parquet(CKD / "r6_L2/design_v2.parquet")
    d = d[(d["season"] == 2025) & (d["period"] <= 2.0) & d["game_id"].isin(cc)].reset_index(drop=True)
    gd = pd.to_datetime(d["game_date"]).to_numpy()
    d["month"] = pd.to_datetime(d["game_date"]).dt.month
    tq = d.groupby("offense_team_id")["off_tempo_rel"].mean()
    d["team_q"] = d["offense_team_id"].map(pd.qcut(tq, 5, labels=False))
    rep = {"n_rows": int(len(d)), "n_games": int(d["game_id"].nunique()),
           "actual_mean": float(d["duration_s"].mean()), "arms": {},
           "offline_reseed_floor": "0 by construction (deterministic family)",
           "fold1": "not run (no F1 S1 schedule)"}
    comp = d["start_reason"].value_counts(normalize=True)
    rep["start_type_composition_truth"] = comp.to_dict()
    for lab, mode in ARMS.items():
        mf = CKD / V3C_MODES[mode]["manifest"]
        if not mf.exists():
            rep["arms"][lab] = "schedule missing"
            continue
        sched = load_schedule(mode)
        cuts = np.array([np.datetime64(r["refit_date"], "ns") for r in sched])
        seg = np.searchsorted(cuts, gd, side="right") - 1
        e = np.empty(len(d))
        for k in np.unique(seg):
            r = np.flatnonzero(seg == k)
            e[r] = e_consumed(sched[int(k)]["arm"], d.iloc[r])
        d["e"] = e
        mm, am = float(e.mean()), float(d["duration_s"].mean())
        by_type = d.groupby("start_reason").agg(share=("e", "size"), model=("e", "mean"), actual=("duration_s", "mean"))
        by_type["share"] /= len(d)
        by_type["gap_s"] = by_type["model"] - by_type["actual"]
        by_type["poss_contrib"] = -by_type["share"] * by_type["gap_s"] * 1200.0 / (mm * am)
        by_month = d.groupby("month").agg(model=("e", "mean"), actual=("duration_s", "mean"))
        by_month["gap_s"] = by_month["model"] - by_month["actual"]
        tqt = d.groupby("team_q").agg(model=("e", "mean"), actual=("duration_s", "mean"))
        span_m = float(tqt["model"].max() - tqt["model"].min())
        span_a = float(tqt["actual"].max() - tqt["actual"].min())
        slope = float(np.polyfit(tqt["actual"], tqt["model"], 1)[0])
        rep["arms"][lab] = {
            "mode": mode, "model_mean": mm, "gap_s": mm - am,
            "implied_poss_per_team_game": 1200.0 / mm - 1200.0 / am,
            "by_start_type": by_type.reset_index().to_dict("records"),
            "made_FG_cell": by_type.loc["made_FG"].to_dict(),
            "by_month": by_month.reset_index().to_dict("records"),
            "team_tempo_quintiles": tqt.reset_index().to_dict("records"),
            "tempo_span_ratio": span_m / span_a if span_a else None,
            "tempo_slope_model_on_actual": slope,
        }
        print(lab, round(mm - am, 4), round(1200.0 / mm - 1200.0 / am, 4), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
