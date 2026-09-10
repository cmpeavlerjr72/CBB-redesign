"""
run_engine.py -- run the possession engine over a fold's test season.

    .venv/Scripts/python.exe scripts/run_engine.py --fold F2 --seeds 200 --season 2025

Writes `results/engine_v0/<tag>/`:
    games.parquet    one row per (game_id, seed): home_pts, away_pts, possessions,
                     n_periods, and the seven contract box pairs
                     (fga3, fga2_rim, fga2_jump, fta, tov, oreb, dreb)
    players.parquet  one row per (game_id, seed, athlete_id): minutes, pts, reb,
                     ast (a PLACEHOLDER, always 0 -- there is no assist model),
                     fga, fg3a, fta
    run_meta.json    the eval contract's required keys plus every provisional
                     adapter flag and the measured throughput

SEALED GUARD. Season 2026 is sealed until fold-2 selection is done
(`cbb_sim.data.seal`). This script additionally refuses 2026 outright unless
`CBB_UNSEAL=1` is set, so a stray `--season 2026` cannot start a run.

PARALLELISM. Games are split across worker processes; each worker runs every
seed for its own games, so a game's RNG stream never depends on the split.
Thread-count env vars are pinned to 1 in every worker (CLAUDE.md: "Pin
thread-count env vars in any container") because the parallelism is across
processes, not inside LightGBM.

PARTIAL RUNS. `--seeds` may be cut short by wall-clock; a run that completes
fewer seeds than requested is written with `partial=True` and the seed count it
actually has, never padded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_RESULTS = Path("results/engine_v0")
INPUT_DIR = Path("data/processed/models/engine")
_PIN = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS")

_W: dict = {}


def _init_worker(tag: str, fold: str, input_dir: str, flags: dict) -> None:
    for k in _PIN:
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = str(v)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(input_dir, tag)
    _W["inp"] = inp
    _W["ad"] = Adapters.load(inp, fold)


def _run_block(job: tuple) -> tuple:
    """One (game block, seed block) batch."""
    from cbb_sim.engine import loop as L
    game_rows, seeds, keep_players = job
    inp, ad = _W["inp"], _W["ad"]
    gi = np.repeat(np.asarray(game_rows, dtype=np.int64), len(seeds))
    sd = np.tile(np.asarray(seeds, dtype=np.int64), len(game_rows))
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=keep_players)
    return res.games, res.players, res.diag, res.n_possessions, res.seconds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 4, 1))
    ap.add_argument("--games-per-block", type=int, default=60)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--time-budget-s", type=float, default=0.0,
                    help="stop dispatching new blocks after this many seconds and "
                         "write what finished with partial=True")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--input-dir", default=str(INPUT_DIR))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-players", action="store_true")
    args = ap.parse_args()

    if int(args.season) >= 2026 and os.environ.get("CBB_UNSEAL") != "1":
        raise SystemExit(
            f"season {args.season} is SEALED (CLAUDE.md: 2025-26 is sealed until fold-2 "
            "selection is done). Set CBB_UNSEAL=1 deliberately to override.")
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(args.season)], context=f"engine_v0 {args.fold}")

    t0 = time.time()
    for k in _PIN:
        os.environ.setdefault(k, "1")
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs

    in_tag = f"{args.fold}_{args.season}"
    inp = EngineInputs.load(args.input_dir, in_tag)
    ad = Adapters.load(inp, args.fold)
    flags = {k: ad.flags[k] for k in ("ENGINE_EVENT", "ENGINE_CLOCK", "ENGINE_ROTATION",
                                      "ENGINE_FG3")}
    print(f"engine_v0 {args.fold}/{args.season}: {inp.n_games} games, "
          f"{args.seeds} seeds, {args.workers} workers")
    print("adapters: " + json.dumps({k: v for k, v in ad.flags.items() if k != "sources"}))

    n_games = inp.n_games if not args.max_games else min(args.max_games, inp.n_games)
    game_rows = np.arange(n_games)
    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)

    # Blocks: seed-major so that an interrupted run has COMPLETE seeds for every
    # game rather than complete games for some seeds -- the first is gradeable,
    # the second is not.
    jobs = []
    for s0 in range(0, len(seeds), args.seeds_per_block):
        sb = seeds[s0:s0 + args.seeds_per_block]
        for g0 in range(0, n_games, args.games_per_block):
            jobs.append((game_rows[g0:g0 + args.games_per_block], sb, not args.no_players))
    print(f"  {len(jobs)} blocks of <= {args.games_per_block} games x "
          f"{args.seeds_per_block} seeds")

    tag = args.tag or f"{args.fold}_{args.season}_s{args.seeds}"
    out = Path(args.results_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    gframes, pframes = [], []
    diag: dict = {}
    n_poss = 0
    done = 0
    stopped_early = False
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(in_tag, args.fold, args.input_dir, flags)) as ex:
        futs = {ex.submit(_run_block, j): j for j in jobs}
        for fut in futs:
            pass
        from concurrent.futures import as_completed
        for fut in as_completed(futs):
            g, p, d, np_, _ = fut.result()
            gframes.append(g)
            if len(p):
                pframes.append(p)
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + v
            n_poss += np_
            done += 1
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)} blocks, {n_poss:,} possessions, "
                      f"{el:.0f}s, {n_poss / max(el, 1e-9):,.0f} poss/s", flush=True)
            if args.time_budget_s and (time.time() - t0) > args.time_budget_s:
                stopped_early = True
                for f2 in futs:
                    f2.cancel()
                break

    games = pd.concat(gframes, ignore_index=True)
    players = pd.concat(pframes, ignore_index=True) if pframes else None

    # Only seeds that are COMPLETE over every game in the run are kept; a
    # half-finished seed is dropped rather than written as a thin cell.
    per_seed = games.groupby("seed")["game_id"].nunique()
    full = per_seed[per_seed == n_games].index.to_numpy()
    n_dropped = int(len(per_seed) - len(full))
    if n_dropped:
        games = games[games["seed"].isin(full)].reset_index(drop=True)
        if players is not None:
            players = players[players["seed"].isin(full)].reset_index(drop=True)
    seeds_written = sorted(int(s) for s in full)
    partial = bool(stopped_early or len(seeds_written) < args.seeds)

    games.to_parquet(out / "games.parquet", index=False)
    if players is not None:
        players.to_parquet(out / "players.parquet", index=False)

    elapsed = time.time() - t0
    meta = {
        "engine_tag": f"engine_v0/{tag}",
        "created_at": datetime.now(UTC).isoformat(),
        "seeds": seeds_written,
        "n_seeds": len(seeds_written),
        "n_seeds_requested": int(args.seeds),
        "fold": args.fold,
        "backtest": True,
        "sealed_touched": False,
        "partial": partial,
        "seeds_dropped_incomplete": n_dropped,
        "season": int(args.season),
        "n_games": int(n_games),
        "n_rows": int(len(games)),
        "possessions_simulated": int(n_poss),
        "runtime_s": round(elapsed, 1),
        "possessions_per_second": round(n_poss / max(elapsed, 1e-9), 1),
        "workers": int(args.workers),
        "ast_is_placeholder": True,
        "adapter_flags": ad.flags,
        "engine_rules_from_data": inp.rules,
        "inputs_meta": inp.meta,
        "diagnostics": diag,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}/games.parquet ({len(games):,} rows), "
          f"players.parquet ({0 if players is None else len(players):,} rows), run_meta.json")
    print(f"throughput: {meta['possessions_per_second']:,.0f} possessions/second "
          f"on {args.workers} workers; partial={partial}; seeds={len(seeds_written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
