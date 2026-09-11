"""
run_clk3c_closed_loop.py -- one round-3c clock arm through the engine.

Pre-registration: `docs/models/clock/experiments.md` section 12.

    .venv/Scripts/python.exe scripts/run_clk3c_closed_loop.py \
        --arm v3c_gamma_P3_s1 --seeds 5 --tag clock3c_gamma_P3

WHY THIS EXISTS AND `run_engine.py` IS NOT EDITED
-------------------------------------------------
Two things round 3c needs that the shared runner does not do, and neither is a
change any other worker should have to absorb:

1. **The pre-registered game subset.** Section 12.3 fixes it as a stride:
   sort the F2 2025 slate by `game_id` ascending, take every 11th row, keep the
   first 500. `run_engine.py --max-games` takes the FIRST N, which on this slate
   is November only.
2. **The S1 segment partition.** `loop.py` calls `clock.draw(team, state, u)`
   with no game index, so a dated clock schedule is honoured by dispatching each
   month's games to a run carrying that month's artifact. The partition comes
   from `ArtifactManifest.seg_of_game` -- the same object, the same rule, the
   same two honest-backtest checks -- so the result is identical to per-row
   routing, and this script ASSERTS that every game it sends to segment k is a
   game the manifest assigns to k.

It also folds the clock adapter's end-of-half counters back from the workers,
which is the only way that statistic can be had: the engine writes no
possession-level file.

Every other sub-model is pinned by an EXPLICIT environment value (section 12.2)
and written into `run_meta.json`, so an arm run now and an arm run in three
hours are the same comparison.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_RESULTS = Path("results/engine_v0")
INPUT_DIR = Path("data/processed/models/engine")
_PIN = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS")

#: Section 12.2. Held fixed and named on every run of every arm.
PINNED_SUBMODELS = {
    "ENGINE_EVENT": "round2_s1",
    "ENGINE_FG_MAKE": "round2b_S_C_s1",
    "ENGINE_FG3": "decision8",
    "ENGINE_ROTATION": "reference",
}

#: Section 12.3, fixed before any run.
SUBSET_STRIDE = 11
SUBSET_N = 500

_W: dict = {}


def _init_worker(tag: str, fold: str, season: int, input_dir: str, flags: dict) -> None:
    for k in _PIN:
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = str(v)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbb_sim.engine.adapters import STATE_INDEX, Adapters
    from cbb_sim.engine.clock_adapter_v3 import ClockAdapterV3, RecordingClock
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(input_dir, tag)
    ad = Adapters.load(inp, fold, season)
    if not isinstance(ad.clock, ClockAdapterV3):
        # The incumbent has no end-of-half counters of its own and has to be
        # reported on the same metric as the candidates or the table is not a
        # comparison. Wrapping watches; it changes no draw.
        ad.clock = RecordingClock(ad.clock, dict(STATE_INDEX))
    _W["inp"] = inp
    _W["ad"] = ad


def _run_block(job: tuple) -> tuple:
    from cbb_sim.engine import loop as L
    game_rows, seeds, keep_players = job
    inp, ad = _W["inp"], _W["ad"]
    gi = np.repeat(np.asarray(game_rows, dtype=np.int64), len(seeds))
    sd = np.tile(np.asarray(seeds, dtype=np.int64), len(game_rows))
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=keep_players)
    return res.games, res.diag, res.n_possessions, ad.clock.eoh_drain()


def subset_rows(games: pd.DataFrame) -> np.ndarray:
    """The pre-registered subset, as ROW INDICES into the engine slate."""
    order = np.argsort(games["game_id"].to_numpy(), kind="stable")
    return order[::SUBSET_STRIDE][:SUBSET_N]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="ENGINE_CLOCK value")
    ap.add_argument("--freeze", action="store_true",
                    help="ENGINE_CLOCK_FREEZE=1, the Decision-10 frozen arm")
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games-per-block", type=int, default=40)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--input-dir", default=str(INPUT_DIR))
    args = ap.parse_args()

    if int(args.season) >= 2026 and os.environ.get("CBB_UNSEAL") != "1":
        raise SystemExit(f"season {args.season} is SEALED")
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(args.season)], context=f"clock round 3c {args.fold}")

    t0 = time.time()
    for k in _PIN:
        os.environ.setdefault(k, "1")
    for k, v in PINNED_SUBMODELS.items():
        os.environ[k] = v
    os.environ["ENGINE_CLOCK"] = args.arm
    os.environ["ENGINE_CLOCK_FREEZE"] = "1" if args.freeze else "0"

    from cbb_sim.engine.clock_adapter_v3 import manifest_for, summarise_eoh
    from cbb_sim.engine.inputs import EngineInputs

    in_tag = f"{args.fold}_{args.season}"
    inp = EngineInputs.load(args.input_dir, in_tag)
    rows = subset_rows(inp.games)
    if len(rows) != SUBSET_N:
        raise SystemExit(f"subset rule produced {len(rows)} games, expected {SUBSET_N}")
    sub = inp.games.iloc[rows]
    print(f"clock3c {args.arm}{' FROZEN' if args.freeze else ''}: "
          f"{len(rows)} games {sub['game_date'].min().date()}..{sub['game_date'].max().date()}, "
          f"{args.seeds} seeds from {args.seed_offset}, {args.workers} workers")

    man = manifest_for(inp, args.arm)
    if man is None:
        partition = [(None, rows)]
        print("  static arm: one run")
    else:
        seg = man.seg_of_game[rows]
        partition = [(int(k), rows[seg == k]) for k in np.unique(seg)]
        for k, r in partition:
            e = man.entries[k]
            # the partition and the adapter's own selection cannot disagree
            assert (man.seg_of_game[r] == k).all()
            print(f"  segment {k}: refit {e.refit_date.date()}, trained through "
                  f"{e.max_train_date.date()}, {len(r)} games")

    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    gframes, eohs = [], []
    diag: dict = {}
    n_poss = 0
    for k, grows in partition:
        flags = dict(PINNED_SUBMODELS)
        flags["ENGINE_CLOCK"] = args.arm
        flags["ENGINE_CLOCK_FREEZE"] = "1" if args.freeze else "0"
        if k is not None:
            flags["ENGINE_CLOCK_SEGMENT"] = str(k)
        jobs = []
        for s0 in range(0, len(seeds), args.seeds_per_block):
            sb = seeds[s0:s0 + args.seeds_per_block]
            for g0 in range(0, len(grows), args.games_per_block):
                jobs.append((grows[g0:g0 + args.games_per_block], sb, False))
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(in_tag, args.fold, int(args.season),
                                           args.input_dir, flags)) as ex:
            futs = [ex.submit(_run_block, j) for j in jobs]
            for fut in as_completed(futs):
                g, d, np_, eoh = fut.result()
                gframes.append(g)
                eohs.append(eoh)
                for kk, v in d.items():
                    diag[kk] = diag.get(kk, 0) + v
                n_poss += np_
        print(f"  segment {k}: done, {n_poss:,} possessions cumulative, "
              f"{time.time() - t0:.0f}s", flush=True)

    games = pd.concat(gframes, ignore_index=True)
    per_seed = games.groupby("seed")["game_id"].nunique()
    full = per_seed[per_seed == SUBSET_N].index.to_numpy()
    dropped = int(len(per_seed) - len(full))
    if dropped:
        games = games[games["seed"].isin(full)].reset_index(drop=True)
    if not len(full):
        raise SystemExit("no seed completed every game in the subset")

    out = Path(args.results_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out / "games.parquet", index=False)

    # the adapter provenance, read from a throwaway load in THIS process
    from cbb_sim.engine.adapters import Adapters
    if man is not None:
        os.environ["ENGINE_CLOCK_SEGMENT"] = str(partition[0][0])
    ad = Adapters.load(inp, args.fold, int(args.season))
    elapsed = time.time() - t0
    meta = {
        "engine_tag": f"engine_v0/{args.tag}",
        "round": "clock 3c (docs/models/clock/experiments.md section 12)",
        "created_at": datetime.now(UTC).isoformat(),
        "arm": args.arm, "freeze_score_diff": bool(args.freeze),
        "seeds": sorted(int(s) for s in full), "n_seeds": int(len(full)),
        "n_seeds_requested": int(args.seeds), "seed_offset": int(args.seed_offset),
        "fold": args.fold, "season": int(args.season), "backtest": True,
        "sealed_touched": False, "partial": bool(dropped),
        "n_games": SUBSET_N, "n_rows": int(len(games)),
        "subset_rule": (f"F2 {args.season} slate sorted by game_id ascending, every "
                        f"{SUBSET_STRIDE}th row, first {SUBSET_N}"),
        "game_ids": [int(x) for x in sub["game_id"].to_numpy()],
        "segments": [] if man is None else [
            {"segment": k, "n_games": int(len(r)),
             "refit_date": str(man.entries[k].refit_date.date()),
             "max_train_date": str(man.entries[k].max_train_date.date()),
             "path": str(man.entries[k].path)} for k, r in partition],
        "possessions_simulated": int(n_poss),
        "runtime_s": round(elapsed, 1),
        "possessions_per_second": round(n_poss / max(elapsed, 1e-9), 1),
        "workers": int(args.workers),
        "adapter_flags": ad.flags,
        "end_of_half": summarise_eoh(eohs),
        "diagnostics": diag,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                       encoding="utf-8")
    print(f"wrote {out} ({len(games):,} rows, {len(full)} seeds, {elapsed:.0f}s, "
          f"{meta['possessions_per_second']:,.0f} poss/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
