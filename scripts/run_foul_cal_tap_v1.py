#!/usr/bin/env python
"""run_foul_cal_tap_v1.py -- paired closed loop for the foul-accrual calendar round (possession_outcome
experiments.md section 32.4) with the round-9 in-process accrual tap (`run_foul_joint_tap_v2`'s Tap,
imported unedited) writing per (seed, game, offence side, half) sums: possessions, in-bonus possessions,
FTA, FGA, shooting / bonus trips, and-ones, silent fouls.

Differences from tap v2 (which pins the PRE-v3 served stack, clock v5b_glat_pmean): NO pins; the engine
defaults (served stack v3) are used unless `--env` sets a variable; optional fold overlay (the
`run_engine_overlay_v2` JSON, applied in every worker) and `--cal-dir` (re-points `foul_cal.LUT_DIR`, the
fold's calendar tables). The tap wrappers return the real values, so the simulation is unchanged; the
control's games.parquet is compared against the box reads to prove it.

    .venv/Scripts/python.exe scripts/run_foul_cal_tap_v1.py --tag fcal_F2_ctrl_s50 --fold F2 --season 2025 \
        --input-dir data/processed/models/engine_v3 --seeds 50 --workers 8 [--env ENGINE_FOUL_CAL=A2dec]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_foul_joint_tap_v2 as T2  # noqa: E402

PIN = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
       "LIGHTGBM_NUM_THREADS")


def _init(tag, fold, season, input_dir, flags, overlay, cal_dir):
    for k in PIN:
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = str(v)
    os.environ.pop("ENGINE_CLOCK_SEGMENT", None)
    if overlay:
        os.environ["CBB_OVERLAY_V2"] = overlay
        import run_engine_overlay_v2 as OV
        OV.apply_overrides()
    if cal_dir:                      # "<arm>=<stem>": the fold's calendar table under foul_cal.LUT_DIR
        from cbb_sim.engine import foul_cal as FC
        arm, stem = cal_dir.split("=", 1)
        FC.FOLD_STEM[arm] = stem
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(input_dir, tag)
    T2._W["inp"] = inp
    T2._W["ad"] = Adapters.load(inp, fold, season)
    T2._W["tap"] = T2._install_tap(float(inp.rules["silent_foul_per_possession"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games-per-block", type=int, default=30)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--overrides", default=None, help="run_engine_overlay_v2 JSON (fold-1 reads)")
    ap.add_argument("--cal-dir", default=None, help="<arm>=<table stem>, e.g. A2dbk=lut_acc_A2dbk_F1 (fold-1 reads)")
    ap.add_argument("--results-dir", default="results/engine_v0")
    a = ap.parse_args()
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(a.season)], context="foul calendar tap")
    if int(a.season) >= 2026:
        raise SystemExit("season is SEALED")
    extra = dict(kv.split("=", 1) for kv in a.env)
    for k, v in extra.items():
        os.environ[k] = v
    overlay = Path(a.overrides).read_text(encoding="utf-8") if a.overrides else ""
    from run_engine import engine_provenance
    prov = engine_provenance()
    from cbb_sim.engine.inputs import EngineInputs
    in_tag = f"{a.fold}_{a.season}"
    inp = EngineInputs.load(a.input_dir, in_tag)
    rows = np.arange(len(inp.games), dtype=np.int64)
    seeds = np.arange(a.seed_offset, a.seed_offset + a.seeds, dtype=np.int64)
    jobs = [(rows[g0:g0 + a.games_per_block], seeds, False) for g0 in range(0, len(rows), a.games_per_block)]
    flags = {k: v for k, v in os.environ.items() if k.startswith("ENGINE_")}
    print(f"foul-cal tap {a.tag}: {len(rows)} games x {len(seeds)} seeds, flags {flags}, "
          f"overlay {bool(overlay)}, cal_dir {a.cal_dir}, {a.workers} workers, {len(jobs)} blocks", flush=True)
    t0 = time.time()
    gf, recs, diag, n_poss = [], [], {}, 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(in_tag, a.fold, int(a.season), a.input_dir, flags, overlay,
                                       a.cal_dir)) as ex:
        futs = [ex.submit(T2._run_block, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            g, _p, d, np_, R = fut.result()
            gf.append(g)
            R = pd.DataFrame(R, columns=T2.REC_COLS + ["silent", "p_silent_x1e6", "off_foul"])
            R["half"] = np.where(R["period"] >= 3, 3, R["period"]).astype("int8")
            R["and_one"] = R["trip_fouls"] - R["n_shoot_trip"] - R["n_bonus_trip"]
            R["poss"] = 1
            R["in_bonus"] = (R["def_fouls"] >= R["bonus_thr"]).astype("int32")
            R["p_silent"] = R["p_silent_x1e6"] / 1e6
            recs.append(R.groupby(["seed", "gidx", "off_side", "half"])[
                ["poss", "in_bonus", "fta", "fga", "n_shoot_trip", "n_bonus_trip", "and_one", "trip_fouls",
                 "silent", "off_foul", "p_silent"]].sum().reset_index())
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + v
            n_poss += np_
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} blocks {time.time() - t0:.0f}s", flush=True)
    out = Path(a.results_dir) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    games = pd.concat(gf, ignore_index=True)
    games.to_parquet(out / "games.parquet", index=False)
    A = pd.concat(recs, ignore_index=True)
    A["game_id"] = inp.games["game_id"].to_numpy()[A["gidx"].to_numpy()]
    A.to_parquet(out / "half_agg.parquet", index=False)
    meta = {"tag": a.tag, "created_at": datetime.now(UTC).isoformat(),
            "round": "possession_outcome experiments.md section 32 (foul-accrual calendar)",
            "flags": flags, "extra_env": extra, "seeds": [int(s) for s in seeds], "n_games": int(len(rows)),
            "n_rows": int(len(games)), **prov, "possessions_simulated": int(n_poss),
            "runtime_s": round(time.time() - t0, 1), "workers": a.workers, "diagnostics": diag,
            "fold": a.fold, "season": int(a.season), "backtest": True, "input_dir": a.input_dir,
            "overrides": a.overrides, "cal_dir": a.cal_dir}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}: {len(games)} rows, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
