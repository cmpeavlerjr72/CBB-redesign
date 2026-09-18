"""
run_po4b_closed_loop.py -- one possession-outcome arm through the engine.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 12
(the SHIP GATE).

    .venv/Scripts/python.exe scripts/run_po4b_closed_loop.py \
        --arm round4b_G2 --seeds 25 --tag po4b_G2_s25

WHAT IS DIFFERENT FROM `run_clk4_closed_loop.py`
------------------------------------------------
1. **The varying flag is `ENGINE_EVENT`, not `ENGINE_CLOCK`.** The clock is
   PINNED at the served, adopted `v5b_glat_pmean`.
2. **The pinned stack is the SERVED stack, not the clock lane's harness.**
   `run_clk4_closed_loop.PINNED_SUBMODELS` pins
   `ENGINE_FG_MAKE=round3_shooter_S_C_s1`, the superseded INTERIM fg_make; the
   served default is `round4_B1`. A ship gate must price an arm on the stack
   that is actually served, so every pin here is the value `adapters.py`
   itself defaults to, and the run asserts that at startup (`--allow-drift`
   turns the assertion into a warning, and is recorded in `run_meta.json` when
   it is used).
3. **Players are kept.** `eval_gates.py`'s G8 is one of the nine gates this
   round is a no-regression test on; a run with no `players.parquet` scores it
   NEEDS-INSTRUMENTATION and the gate goes unread.

The subset rule (F2 slate sorted by `game_id`, every 11th row, first 500), the
pairing-by-construction and the pinned-submodel discipline are the clock
lane's, unchanged, so this round's table sits beside clock rounds 3c/4/5/5b/5d.
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

#: The SERVED stack (section 12.3). Every value here must equal the default
#: `adapters.py` resolves when the variable is unset; `_assert_served_stack`
#: checks that at startup rather than trusting this literal to stay current.
PINNED_SUBMODELS = {
    "ENGINE_CLOCK": "v5b_glat_pmean",
    "ENGINE_FG_MAKE": "round4_B1",
    "ENGINE_FG3": "decision8",
    "ENGINE_REBOUND": "s1_weekly",
    "ENGINE_FREE_THROW": "s1_conf_aligned",
    "ENGINE_ROTATION": "reference",
    "ENGINE_USAGE": "reference",
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
    return res.games, res.players, res.diag, res.n_possessions


def subset_rows(games: pd.DataFrame) -> np.ndarray:
    """The standing 500-game subset: clock experiments.md section 14.4, reused
    verbatim so this round's table lines up with the clock rounds'."""
    order = np.argsort(games["game_id"].to_numpy(), kind="stable")
    return order[::SUBSET_STRIDE][:SUBSET_N]


def _assert_served_stack(allow_drift: bool) -> list[str]:
    """Every pin except `ENGINE_EVENT` must be what `adapters.py` itself
    defaults to. A silent drift between this literal and the served default is
    exactly how a ship gate ends up pricing an arm on a stack nobody serves."""
    import re
    src = (Path(__file__).resolve().parents[1]
           / "src/cbb_sim/engine/adapters.py").read_text(encoding="utf-8")
    drift = []
    for k, v in PINNED_SUBMODELS.items():
        m = re.search(rf'os\.environ\.get\(\s*"{k}"\s*,\s*"([^"]+)"\s*\)', src)
        if m is None:
            drift.append(f"{k}: no default found in adapters.py")
        elif m.group(1) != v:
            drift.append(f"{k}: adapters.py defaults to {m.group(1)!r}, this run pins {v!r}")
    if drift and not allow_drift:
        raise SystemExit("served-stack check FAILED (section 12.3):\n  " + "\n  ".join(drift)
                         + "\nPass --allow-drift only deliberately; it is recorded in run_meta.")
    return drift


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="ENGINE_EVENT value")
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games-per-block", type=int, default=40)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-players", action="store_true")
    ap.add_argument("--allow-drift", action="store_true")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--input-dir", default=str(INPUT_DIR))
    args = ap.parse_args()

    if int(args.season) >= 2026 and os.environ.get("CBB_UNSEAL") != "1":
        raise SystemExit(f"season {args.season} is SEALED")
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(args.season)], context=f"possession-outcome round 5 {args.fold}")

    drift = _assert_served_stack(args.allow_drift)
    # Several lanes share this checkout and the working tree is routinely dirty
    # with another lane's in-progress edits, so record WHICH before simulating.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_engine import engine_provenance
    prov = engine_provenance()
    t0 = time.time()
    for k in _PIN:
        os.environ.setdefault(k, "1")
    flags = dict(PINNED_SUBMODELS)
    flags["ENGINE_EVENT"] = args.arm
    for k, v in flags.items():
        os.environ[k] = v
    os.environ.pop("ENGINE_CLOCK_SEGMENT", None)

    from cbb_sim.engine.inputs import EngineInputs

    in_tag = f"{args.fold}_{args.season}"
    inp = EngineInputs.load(args.input_dir, in_tag)
    rows = subset_rows(inp.games)
    if len(rows) != SUBSET_N:
        raise SystemExit(f"subset rule produced {len(rows)} games, expected {SUBSET_N}")
    sub = inp.games.iloc[rows]
    print(f"po4b {args.arm}: {len(rows)} games "
          f"{sub['game_date'].min().date()}..{sub['game_date'].max().date()}, "
          f"{args.seeds} seeds from {args.seed_offset}, {args.workers} workers", flush=True)

    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    keep_players = not args.no_players
    jobs = []
    for s0 in range(0, len(seeds), args.seeds_per_block):
        sb = seeds[s0:s0 + args.seeds_per_block]
        for g0 in range(0, len(rows), args.games_per_block):
            jobs.append((rows[g0:g0 + args.games_per_block], sb, keep_players))

    gframes, pframes = [], []
    diag: dict = {}
    n_poss = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(in_tag, args.fold, int(args.season),
                                       args.input_dir, flags)) as ex:
        futs = [ex.submit(_run_block, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            g, p, d, np_ = fut.result()
            gframes.append(g)
            if p is not None and len(p):
                pframes.append(p)
            for kk, v in d.items():
                diag[kk] = diag.get(kk, 0) + v
            n_poss += np_
            if i % 5 == 0:
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
    if pframes:
        players = pd.concat(pframes, ignore_index=True)
        players = players[players["seed"].isin(full)].reset_index(drop=True)
        players.to_parquet(out / "players.parquet", index=False)

    from cbb_sim.engine.adapters import Adapters
    ad = Adapters.load(inp, args.fold, int(args.season))
    elapsed = time.time() - t0
    meta = {
        "engine_tag": f"engine_v0/{args.tag}",
        "round": "possession_outcome 5 ship gate "
                 "(docs/models/possession_outcome/experiments.md section 12)",
        "created_at": datetime.now(UTC).isoformat(),
        "arm": args.arm,
        "seeds": sorted(int(s) for s in full), "n_seeds": int(len(full)),
        "n_seeds_requested": int(args.seeds), "seed_offset": int(args.seed_offset),
        "fold": args.fold, "season": int(args.season), "backtest": True,
        "sealed_touched": False, "partial": bool(dropped),
        "n_games": int(len(rows)), "n_rows": int(len(games)),
        "keep_players": bool(keep_players),
        "subset_rule": (f"F2 {args.season} slate sorted by game_id ascending, every "
                        f"{SUBSET_STRIDE}th row, first {SUBSET_N}"),
        "game_ids": [int(x) for x in sub["game_id"].to_numpy()],
        "served_stack_drift": drift,
        "allow_drift": bool(args.allow_drift),
        **prov,
        "possessions_simulated": int(n_poss),
        "runtime_s": round(elapsed, 1),
        "possessions_per_second": round(n_poss / max(elapsed, 1e-9), 1),
        "workers": int(args.workers),
        "adapter_flags": ad.flags,
        "diagnostics": diag,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                       encoding="utf-8")
    print(f"wrote {out} ({len(games):,} rows, {len(full)} seeds, {elapsed:.0f}s, "
          f"{meta['possessions_per_second']:,.0f} poss/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
