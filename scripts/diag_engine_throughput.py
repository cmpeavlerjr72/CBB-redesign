"""
diag_engine_throughput.py -- possessions/s/core on this box, and the CLOCK
LOOKUP-TABLE BINNING ERROR.

Two questions the engine deliverable asks and `docs/models/engine/RESUME.md`
section 3 lists as unfinished.

1. THROUGHPUT. `possessions / summed worker CPU seconds`, measured on the same
   batch shape a production run uses, with the machine's contention recorded
   next to it. The build session's 862 poss/s/core was taken with the box
   saturated by 14 engine workers plus other agents' jobs; a number without its
   contention is not a measurement.

2. THE BINNING ERROR. The clock arm is ~85% of the engine's model cost, so the
   lookup-table export is the whole of the available speedup. One lookup table
   is already on disk: the clock bake-off's `reference_not_adopted_empirical`
   IS a binned pmf (`EmpiricalArm.level_pmfs`, (n_cells, 91) over 7 binned
   state dimensions). `ENGINE_CLOCK=reference_empirical` runs it.

   The deliverable asks that the binning error be REPORTED, not assumed small.
   This script runs the two clock arms over the SAME games and the SAME seeds
   -- paired streams, so every difference is the arm and not the noise -- and
   reports both the throughput each buys and what it costs in possessions per
   game and points per possession. That is a like-for-like difference: the
   engine's RNG is keyed on (seed, game_id, family), so arm A's game g seed s
   and arm B's game g seed s draw from the same stream positions.

   NOTE ON WHAT THIS MEASURES. The empirical arm is a DIFFERENT MODEL, not a
   binned export of the quantile arm, so the difference below is (binning
   error + model difference) and is an UPPER BOUND on the binning error of a
   faithful export of the quantile arm itself. It is reported as such. Binning
   the quantile arm over its own 20 features remains unbuilt.

Usage:
    .venv/Scripts/python.exe scripts/diag_engine_throughput.py \
        --fold F2 --season 2025 --games 300 --seeds 10 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_PIN = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS")
_W: dict = {}


def _init(tag: str, fold: str, season: int, input_dir: str, flags: dict) -> None:
    for k in _PIN:
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = str(v)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(input_dir, tag)
    _W["inp"] = inp
    _W["ad"] = Adapters.load(inp, fold, season)


def _block(job: tuple) -> tuple:
    from cbb_sim.engine import loop as L
    rows, seeds = job
    gi = np.repeat(np.asarray(rows, dtype=np.int64), len(seeds))
    sd = np.tile(np.asarray(seeds, dtype=np.int64), len(rows))
    res = L.simulate_chunk(_W["inp"], _W["ad"], gi, sd, keep_players=False)
    return res.games, res.n_possessions, res.seconds


def cpu_busy(samples: int = 3) -> float:
    """Total CPU utilisation, so throughput is quoted with its contention."""
    try:
        ps = ("(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 "
              f"-MaxSamples {samples}).CounterSamples | "
              "ForEach-Object {{ $_.CookedValue }}")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
        v = [float(x) for x in out.split() if x.strip()]
        return round(sum(v) / len(v), 1) if v else float("nan")
    except Exception:                                              # noqa: BLE001
        return float("nan")


def run_arm(clock_mode: str, fold: str, season: int, rows: np.ndarray,
            seeds: np.ndarray, workers: int, gpb: int, spb: int,
            input_dir: str, event_mode: str) -> tuple[pd.DataFrame, dict]:
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    os.environ["ENGINE_CLOCK"] = clock_mode
    os.environ["ENGINE_EVENT"] = event_mode
    tag = f"{fold}_{season}"
    inp = EngineInputs.load(input_dir, tag)
    ad = Adapters.load(inp, fold, season)
    flags = {k: ad.flags[k] for k in ("ENGINE_EVENT", "ENGINE_CLOCK",
                                      "ENGINE_ROTATION", "ENGINE_FG3")}
    jobs = [(rows[g0:g0 + gpb], seeds[s0:s0 + spb])
            for s0 in range(0, len(seeds), spb)
            for g0 in range(0, len(rows), gpb)]
    busy0 = cpu_busy()
    frames, n_poss, core_s = [], 0, 0.0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(tag, fold, season, input_dir, flags)) as ex:
        for f in as_completed([ex.submit(_block, j) for j in jobs]):
            g, np_, s_ = f.result()
            frames.append(g)
            n_poss += int(np_)
            core_s += float(s_)
    wall = time.time() - t0
    busy1 = cpu_busy()
    perf = {
        "ENGINE_CLOCK": clock_mode, "ENGINE_EVENT": event_mode,
        "simulations": int(len(rows) * len(seeds)), "possessions": int(n_poss),
        "wall_s": round(wall, 1), "worker_cpu_s": round(core_s, 1),
        "workers": int(workers),
        "poss_per_s_per_core": round(n_poss / max(core_s, 1e-9), 1),
        "poss_per_s_wall": round(n_poss / max(wall, 1e-9), 1),
        "cpu_busy_pct_before": busy0, "cpu_busy_pct_after": busy1,
        "logical_cores": int(os.cpu_count() or 0),
    }
    return pd.concat(frames, ignore_index=True), perf


def summarise(g: pd.DataFrame) -> dict:
    tot = g["home_pts"] + g["away_pts"]
    return {"possessions_per_game": float(g["possessions"].mean()),
            "total": float(tot.mean()),
            "ppp": float(tot.mean() / (2 * g["possessions"].mean())),
            "margin": float((g["home_pts"] - g["away_pts"]).mean()),
            "ot_rate": float((g["n_periods"] > 2).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games-per-block", type=int, default=50)
    ap.add_argument("--seeds-per-block", type=int, default=10)
    ap.add_argument("--event", default="round2_s1")
    ap.add_argument("--input-dir", default="data/processed/models/engine")
    ap.add_argument("--out", default="results/engine_v0/throughput")
    a = ap.parse_args()

    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(a.season)], context="engine throughput diagnostic")

    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(a.input_dir, f"{a.fold}_{a.season}")
    rng = np.random.default_rng(20260910)          # the seed-count study's subset
    rows = np.sort(rng.choice(inp.n_games, size=min(a.games, inp.n_games), replace=False))
    seeds = np.arange(a.seeds, dtype=np.int64)
    del inp

    out = {}
    frames = {}
    for mode in ("reference", "reference_empirical"):
        print(f"\n--- ENGINE_CLOCK={mode} ---", flush=True)
        g, perf = run_arm(mode, a.fold, a.season, rows, seeds, a.workers,
                          a.games_per_block, a.seeds_per_block, a.input_dir, a.event)
        s = summarise(g)
        out[mode] = {"performance": perf, "summary": s}
        frames[mode] = g
        print(json.dumps(perf, indent=1))
        print(json.dumps(s, indent=1))

    q, e = out["reference"], out["reference_empirical"]
    speedup = e["performance"]["poss_per_s_per_core"] / q["performance"]["poss_per_s_per_core"]
    print("\n" + "=" * 74)
    print("CLOCK LOOKUP-TABLE (binned pmf) vs LIVE QUANTILE MODEL -- paired streams")
    print("=" * 74)
    print(f"{'quantity':<26}{'quantile (live)':>18}{'empirical (LUT)':>18}{'delta':>12}")
    for k in ("possessions_per_game", "total", "ppp", "margin", "ot_rate"):
        print(f"{k:<26}{q['summary'][k]:>18.4f}{e['summary'][k]:>18.4f}"
              f"{e['summary'][k] - q['summary'][k]:>+12.4f}")
    print(f"{'poss/s/core':<26}{q['performance']['poss_per_s_per_core']:>18.1f}"
          f"{e['performance']['poss_per_s_per_core']:>18.1f}{speedup:>+11.2f}x")

    # per-game paired difference: the honest size of the binning error
    a_ = frames["reference"].groupby("game_id")[["possessions"]].mean()
    b_ = frames["reference_empirical"].groupby("game_id")[["possessions"]].mean()
    d = (b_ - a_)["possessions"]
    print(f"\nper-game possessions difference: mean {d.mean():+.4f}, "
          f"MAE {d.abs().mean():.4f}, SD {d.std():.4f}, "
          f"p05 {d.quantile(0.05):+.3f}, p95 {d.quantile(0.95):+.3f}")
    print("\nThis is an UPPER BOUND on the binning error of a faithful lookup export of")
    print("the quantile arm: the empirical arm is a different model, not that arm binned.")

    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "throughput.json").write_text(json.dumps(
        {"arms": out, "speedup_per_core": round(speedup, 3),
         "per_game_possessions_delta": {
             "mean": float(d.mean()), "mae": float(d.abs().mean()),
             "sd": float(d.std()), "p05": float(d.quantile(0.05)),
             "p95": float(d.quantile(0.95))},
         "note": "empirical is a different model, not a binned export of the quantile arm; "
                 "the delta is an upper bound on a faithful export's binning error"},
        indent=1), encoding="utf-8")
    print(f"\nwrote {od}/throughput.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
