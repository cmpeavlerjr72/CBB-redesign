"""
run_clk4_closed_loop.py -- one round-4 clock arm through the engine.

Pre-registration: `docs/models/clock/experiments.md` section 14.

    .venv/Scripts/python.exe scripts/run_clk4_closed_loop.py \
        --arm v3c_srfloor_P3_s1 --seeds 5 --tag clock4_REF

WHAT IS DIFFERENT FROM `run_clk3c_closed_loop.py`
-------------------------------------------------
1. **No segment partition.** `loop.py`'s clock call site is game-indexed since
   2026-09-11, so a dated S1 schedule is routed per GAME inside one run and the
   whole subset runs as one pool. `ENGINE_CLOCK_SEGMENT` is deliberately NOT
   set; a test asserts per-game routing is bit-identical to pinning.
2. **The state-composition accumulator.** `ENGINE_CLOCK_DIAG=1` makes the clock
   adapter count possessions and consumed seconds per round-2 state cell. The
   engine writes no possession-level file, so the distribution of states the
   engine VISITS -- round 4's whole question (L31) -- is accumulated live or not
   at all.
3. **`ENGINE_FG_MAKE=round3_shooter_S_C_s1`**, the interim served fg_make model
   since L29, not round 3c's `round2b_S_C_s1`. Round-4 numbers are therefore
   NOT comparable to round 3c's table row for row, which is exactly why the
   round-4 reference arm is the served clock arm re-run under this pinning.

The subset rule, the pairing-by-construction and the pinned-submodel discipline
are round 3c's, unchanged.
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

#: Held fixed and named on EVERY run of EVERY arm (section 14).
PINNED_SUBMODELS = {
    "ENGINE_EVENT": "round2_s1",
    "ENGINE_FG_MAKE": "round3_shooter_S_C_s1",
    "ENGINE_FG3": "decision8",
    "ENGINE_ROTATION": "reference",
}

SUBSET_STRIDE = 11
SUBSET_N = 500

_W: dict = {}


def _init_worker(tag: str, fold: str, season: int, input_dir: str, flags: dict) -> None:
    for k in _PIN:
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = str(v)
    os.environ.pop("ENGINE_CLOCK_SEGMENT", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(input_dir, tag)
    _W["inp"] = inp
    _W["ad"] = Adapters.load(inp, fold, season)


def _run_block(job: tuple) -> tuple:
    from cbb_sim.engine import loop as L
    game_rows, seeds, keep_players = job
    inp, ad = _W["inp"], _W["ad"]
    gi = np.repeat(np.asarray(game_rows, dtype=np.int64), len(seeds))
    sd = np.tile(np.asarray(seeds, dtype=np.int64), len(game_rows))
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=keep_players)
    cells = ad.clock.cell_drain() if hasattr(ad.clock, "cell_drain") else None
    return res.games, res.diag, res.n_possessions, ad.clock.eoh_drain(), cells


def subset_rows(games: pd.DataFrame) -> np.ndarray:
    order = np.argsort(games["game_id"].to_numpy(), kind="stable")
    return order[::SUBSET_STRIDE][:SUBSET_N]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="ENGINE_CLOCK value")
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games-per-block", type=int, default=40)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-diag", action="store_true",
                    help="skip the state-composition accumulator")
    ap.add_argument("--cc-only", action="store_true",
                    help="DIAGNOSIS ONLY: keep just the clock-complete games of the "
                         "pre-registered subset, so the engine's cell table and the "
                         "data's are cut on the same games. Never used for a gate: "
                         "the gate reads the whole subset and re-bases on the "
                         "clock-complete games afterwards, as round 3c did.")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--input-dir", default=str(INPUT_DIR))
    args = ap.parse_args()

    if int(args.season) >= 2026 and os.environ.get("CBB_UNSEAL") != "1":
        raise SystemExit(f"season {args.season} is SEALED")
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(args.season)], context=f"clock round 4 {args.fold}")

    t0 = time.time()
    for k in _PIN:
        os.environ.setdefault(k, "1")
    flags = dict(PINNED_SUBMODELS)
    flags["ENGINE_CLOCK"] = args.arm
    flags["ENGINE_CLOCK_FREEZE"] = "0"
    flags["ENGINE_CLOCK_DIAG"] = "0" if args.no_diag else "1"
    for k, v in flags.items():
        os.environ[k] = v
    os.environ.pop("ENGINE_CLOCK_SEGMENT", None)

    from cbb_sim.engine.clock_adapter_v3 import manifest_for, summarise_cells, summarise_eoh
    from cbb_sim.engine.inputs import EngineInputs

    in_tag = f"{args.fold}_{args.season}"
    inp = EngineInputs.load(args.input_dir, in_tag)
    rows = subset_rows(inp.games)
    if len(rows) != SUBSET_N:
        raise SystemExit(f"subset rule produced {len(rows)} games, expected {SUBSET_N}")
    if args.cc_only:
        univ = pd.read_parquet("data/processed/games_universe_v2.parquet")
        cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
        keep = np.array([int(g) in cc for g in inp.games["game_id"].to_numpy()[rows]])
        rows = rows[keep]
        print(f"  --cc-only: {len(rows)} clock-complete games of the subset")
    sub = inp.games.iloc[rows]
    man = manifest_for(inp, args.arm)
    print(f"clk4 {args.arm}: {len(rows)} games "
          f"{sub['game_date'].min().date()}..{sub['game_date'].max().date()}, "
          f"{args.seeds} seeds from {args.seed_offset}, {args.workers} workers, "
          f"{0 if man is None else len(man.entries)} dated artifacts routed per game",
          flush=True)

    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    jobs = []
    for s0 in range(0, len(seeds), args.seeds_per_block):
        sb = seeds[s0:s0 + args.seeds_per_block]
        for g0 in range(0, len(rows), args.games_per_block):
            jobs.append((rows[g0:g0 + args.games_per_block], sb, False))

    gframes, eohs, cells = [], [], []
    diag: dict = {}
    n_poss = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(in_tag, args.fold, int(args.season),
                                       args.input_dir, flags)) as ex:
        futs = [ex.submit(_run_block, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            g, d, np_, eoh, cl = fut.result()
            gframes.append(g)
            eohs.append(eoh)
            if cl is not None:
                cells.append(cl)
            for kk, v in d.items():
                diag[kk] = diag.get(kk, 0) + v
            n_poss += np_
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} blocks, {n_poss:,} possessions, "
                      f"{time.time() - t0:.0f}s", flush=True)

    games = pd.concat(gframes, ignore_index=True)
    n_expect = len(rows)
    per_seed = games.groupby("seed")["game_id"].nunique()
    full = per_seed[per_seed == n_expect].index.to_numpy()
    dropped = int(len(per_seed) - len(full))
    if dropped:
        games = games[games["seed"].isin(full)].reset_index(drop=True)
    if not len(full):
        raise SystemExit("no seed completed every game in the subset")

    out = Path(args.results_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out / "games.parquet", index=False)
    if cells:
        summarise_cells(cells).to_parquet(out / "clock_cells.parquet", index=False)

    from cbb_sim.engine.adapters import Adapters
    ad = Adapters.load(inp, args.fold, int(args.season))
    elapsed = time.time() - t0
    meta = {
        "engine_tag": f"engine_v0/{args.tag}",
        "round": "clock 4 (docs/models/clock/experiments.md section 14)",
        "created_at": datetime.now(UTC).isoformat(),
        "arm": args.arm, "freeze_score_diff": False,
        "loop_commit": os.environ.get("CLK4_LOOP_COMMIT", ""),
        "seeds": sorted(int(s) for s in full), "n_seeds": int(len(full)),
        "n_seeds_requested": int(args.seeds), "seed_offset": int(args.seed_offset),
        "fold": args.fold, "season": int(args.season), "backtest": True,
        "sealed_touched": False, "partial": bool(dropped),
        "n_games": int(len(rows)), "clock_complete_only": bool(args.cc_only),
        "n_rows": int(len(games)),
        "subset_rule": (f"F2 {args.season} slate sorted by game_id ascending, every "
                        f"{SUBSET_STRIDE}th row, first {SUBSET_N}"),
        "game_ids": [int(x) for x in sub["game_id"].to_numpy()],
        "clock_routing": "per_game_manifest",
        "segments": [] if man is None else [
            {"segment": k, "refit_date": str(e.refit_date.date()),
             "max_train_date": str(e.max_train_date.date()),
             "n_games_in_subset": int((man.seg_of_game[rows] == k).sum()),
             "path": str(e.path)} for k, e in enumerate(man.entries)],
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
