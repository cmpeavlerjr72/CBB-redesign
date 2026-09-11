"""
exp_engine_seed_count.py -- how many seeds the engine needs before a gate or an
ROI number may be read.

`CLAUDE.md`, standing rule "backtests must be honest": *"A seed-count study
fixes the minimum seeds before any ROI number is read."* Until this runs, every
engine gate number is a read at whatever seed count happened to be used, with
no statement of its own Monte-Carlo error.

METHOD
------
One run of `N_MAX` seeds over a fixed subset of games, then the seed-count
ladder is read OUT of that run rather than re-simulated. This is exact, not an
approximation, and it is only legal because of the engine's RNG rule: a
simulation's draws depend on `(seed, game_id, family)` and on nothing else, so
seed s of game g is the same realisation whether it was run in a block of 5 or
a block of 200 (`tests/test_engine.py::
test_a_games_draws_do_not_depend_on_the_batch_it_is_in` pins this). The seeds
are therefore PAIRED across every rung of the ladder by construction, and a
k-seed "run" is exactly a k-seed slice of the N_MAX-seed run.

For each rung k, the N_MAX seeds are cut into `floor(N_MAX / k)` DISJOINT
blocks of k. Each block is one independent k-seed estimate of that game's
margin mean, total mean and win probability. The SD across those blocks, per
game, IS the standard error of a k-seed run -- measured, not assumed -- and it
is reported as the mean over games and as the 90th percentile, because a gate
is read on the whole slate and an edge is bet on one game.

The fitted `c / sqrt(k)` curve is reported alongside the measured points. It is
a DESCRIPTION of the measurements used to extrapolate past the largest rung the
run can support, never a substitute for them: every rung with at least two
blocks is measured directly.

Usage:
    .venv/Scripts/python.exe scripts/exp_engine_seed_count.py \
        --fold F2 --season 2025 --games 300 --seeds 200 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
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

#: The pre-registered ladder. Every rung divides 200 so the blocks are disjoint
#: and equal-sized at every rung; a rung that leaves a remainder would compare
#: unequal estimators.
LADDER = (5, 10, 25, 50, 100)


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


def simulate(fold: str, season: int, n_games: int, n_seeds: int, workers: int,
             games_per_block: int, seeds_per_block: int, input_dir: str) -> tuple:
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    tag = f"{fold}_{season}"
    inp = EngineInputs.load(input_dir, tag)
    ad = Adapters.load(inp, fold, season)
    flags = {k: ad.flags[k] for k in ("ENGINE_EVENT", "ENGINE_CLOCK",
                                      "ENGINE_ROTATION", "ENGINE_FG3")}
    # A FIXED subset, chosen by a fixed seed before anything is simulated, so
    # the subset cannot be reselected after seeing a result.
    rng = np.random.default_rng(20260910)
    rows = np.sort(rng.choice(inp.n_games, size=min(n_games, inp.n_games), replace=False))
    seeds = np.arange(n_seeds, dtype=np.int64)
    jobs = [(rows[g0:g0 + games_per_block], seeds[s0:s0 + seeds_per_block])
            for s0 in range(0, n_seeds, seeds_per_block)
            for g0 in range(0, len(rows), games_per_block)]
    print(f"seed-count study: {len(rows)} games x {n_seeds} seeds "
          f"= {len(rows) * n_seeds:,} simulations in {len(jobs)} blocks", flush=True)
    print("adapters: " + json.dumps(flags), flush=True)

    frames, n_poss, core_s = [], 0, 0.0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(tag, fold, season, input_dir, flags)) as ex:
        futs = [ex.submit(_block, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            g, np_, s_ = f.result()
            frames.append(g)
            n_poss += int(np_)
            core_s += float(s_)
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} blocks, {n_poss:,} poss, "
                      f"{time.time() - t0:.0f}s wall", flush=True)
    wall = time.time() - t0
    games = pd.concat(frames, ignore_index=True)
    perf = {"n_games": int(len(rows)), "n_seeds": int(n_seeds),
            "n_simulations": int(len(games)), "possessions": int(n_poss),
            "wall_s": round(wall, 1), "core_s": round(core_s, 1),
            "workers": int(workers),
            "poss_per_s_per_core": round(n_poss / max(core_s, 1e-9), 1),
            "poss_per_s_wall": round(n_poss / max(wall, 1e-9), 1),
            "adapter_flags": flags}
    return games, perf


def ladder_table(games: pd.DataFrame, n_seeds: int) -> tuple[pd.DataFrame, dict]:
    """The measured SE per rung. Returns (table, per-game detail)."""
    g = games.sort_values(["game_id", "seed"])
    gid = g["game_id"].to_numpy()
    ug, inv = np.unique(gid, return_inverse=True)
    G = len(ug)
    margin = (g["home_pts"] - g["away_pts"]).to_numpy(dtype=np.float64)
    total = (g["home_pts"] + g["away_pts"]).to_numpy(dtype=np.float64)
    win = (margin > 0).astype(np.float64)          # the engine leaves no ties
    seed = g["seed"].to_numpy()

    # (G, n_seeds) panels. Every game must carry every seed or the blocks are
    # not comparable across games.
    M = np.full((G, n_seeds), np.nan)
    T = np.full((G, n_seeds), np.nan)
    W = np.full((G, n_seeds), np.nan)
    M[inv, seed] = margin
    T[inv, seed] = total
    W[inv, seed] = win
    keep = np.isfinite(M).all(axis=1)
    M, T, W = M[keep], T[keep], W[keep]
    print(f"  panel: {M.shape[0]} of {G} games complete over all {n_seeds} seeds", flush=True)

    rows, detail = [], {}
    for k in sorted({*LADDER, n_seeds // 2, n_seeds // 4}):
        nb = n_seeds // k
        if nb < 2:
            continue
        cut = nb * k
        r = {"seeds": int(k), "n_blocks": int(nb)}
        for name, A in (("margin_mean", M), ("total_mean", T), ("win_prob", W)):
            blk = A[:, :cut].reshape(A.shape[0], nb, k).mean(axis=2)   # (G, nb)
            sd = blk.std(axis=1, ddof=1)                               # per game
            r[f"{name}_se_mean"] = float(sd.mean())
            r[f"{name}_se_p90"] = float(np.percentile(sd, 90))
            # the slate-level quantity a gate actually reads
            r[f"{name}_slate_se"] = float(blk.mean(axis=0).std(ddof=1))
        rows.append(r)
        detail[str(k)] = r
    return pd.DataFrame(rows).sort_values("seeds").reset_index(drop=True), detail


def seeds_for(target: float, tab: pd.DataFrame, col: str) -> int:
    """Smallest k with SE <= target, from the fitted c/sqrt(k) through the
    measured rungs. Reported next to the measured points, never instead."""
    k = tab["seeds"].to_numpy(dtype=float)
    se = tab[col].to_numpy(dtype=float)
    c = float(np.exp(np.mean(np.log(se) + 0.5 * np.log(k))))    # least squares in log space
    return int(np.ceil((c / target) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games-per-block", type=int, default=50)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--input-dir", default="data/processed/models/engine")
    ap.add_argument("--out", default="results/engine_v0/seed_count")
    a = ap.parse_args()

    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(a.season)], context="engine seed-count study")

    games, perf = simulate(a.fold, a.season, a.games, a.seeds, a.workers,
                           a.games_per_block, a.seeds_per_block, a.input_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out / "games.parquet", index=False)

    tab, detail = ladder_table(games, a.seeds)
    print()
    print(tab.to_string(index=False))
    print()
    print(json.dumps(perf, indent=1))

    targets = {
        "win_prob_se_mean_1pp": seeds_for(0.01, tab, "win_prob_se_mean"),
        "win_prob_se_p90_1pp": seeds_for(0.01, tab, "win_prob_se_p90"),
        "win_prob_se_mean_2pp": seeds_for(0.02, tab, "win_prob_se_mean"),
        "margin_se_mean_0p25pt": seeds_for(0.25, tab, "margin_mean_se_mean"),
        "margin_se_mean_0p5pt": seeds_for(0.5, tab, "margin_mean_se_mean"),
        "total_se_mean_0p5pt": seeds_for(0.5, tab, "total_mean_se_mean"),
        "margin_slate_se_0p05pt": seeds_for(0.05, tab, "margin_mean_slate_se"),
        "total_slate_se_0p05pt": seeds_for(0.05, tab, "total_mean_slate_se"),
    }
    print()
    print(json.dumps(targets, indent=1))
    (out / "seed_count.json").write_text(json.dumps(
        {"performance": perf, "ladder": detail, "extrapolated_seeds_for": targets,
         "ladder_table": tab.to_dict("records")}, indent=1), encoding="utf-8")
    tab.to_csv(out / "ladder.csv", index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
