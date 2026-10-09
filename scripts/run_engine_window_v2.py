#!/usr/bin/env python
"""
run_engine_window_v2.py -- run_engine_window_v2 PLUS `--trajectory-seeds N` (DEFAULT 0 = off): writes
results/trajectories/<tag>/trajectory.parquet (one row per possession, seeds < N). v1 is untouched; game/player outputs are
identical with the flag on or off. Worker initializer and block function are `run_engine`'s own (or `run_engine_overlay_v2`'s wrapped
initializer when `--overrides` is given, for fold-1 artifacts), so a game's RNG stream is (seed, game_id, family) exactly as in a
full-season run. Writes games.parquet / players.parquet / run_meta.json in the results contract.

    python scripts/run_engine_window_v2.py --ids-file <parquet with game_id> --fold F2 --season 2025 \
        --input-dir data/processed/models/engine_v3 --seeds 50 --workers 6 --tag <tag> [--overrides <json>]
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
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_engine as RE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--games-per-block", type=int, default=60)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trajectory-seeds", type=int, default=0)
    ap.add_argument("--results-dir", default="results/player_day1/runs")
    ap.add_argument("--overrides", default=None)
    a = ap.parse_args()
    if a.season >= 2026:
        raise SystemExit("season 2026 is SEALED")
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([a.season], context="run_engine_window_v2")
    prov = RE.engine_provenance()
    init = RE._init_worker
    ov = None
    if a.overrides:
        import run_engine_overlay_v2 as OV
        ov = json.loads(Path(a.overrides).read_text(encoding="utf-8"))
        os.environ[OV.ENV] = json.dumps(ov)
        OV.apply_overrides()
        init = OV._init_full
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    t0 = time.time()
    in_tag = f"{a.fold}_{a.season}"
    inp = EngineInputs.load(a.input_dir, in_tag)
    ad = Adapters.load(inp, a.fold, a.season)
    flags = {k: ad.flags[k] for k in ("ENGINE_EVENT", "ENGINE_CLOCK", "ENGINE_ROTATION", "ENGINE_FG3", "ENGINE_FG_MAKE",
                                      "ENGINE_REBOUND", "ENGINE_FREE_THROW", "ENGINE_ROTATION_SCHEME",
                                      "ENGINE_INPUTS_VERSION") if k in ad.flags}
    ids = pd.read_parquet(a.ids_file)["game_id"].astype("int64").to_numpy()
    pos = {int(g): i for i, g in enumerate(inp.games["game_id"])}
    game_rows = np.array(sorted(pos[int(g)] for g in ids), dtype=np.int64)
    seeds = np.arange(a.seed_offset, a.seed_offset + a.seeds, dtype=np.int64)
    jobs = [(game_rows[g0:g0 + a.games_per_block], seeds[s0:s0 + a.seeds_per_block], True, int(a.trajectory_seeds))
            for s0 in range(0, len(seeds), a.seeds_per_block) for g0 in range(0, len(game_rows), a.games_per_block)]
    out = Path(a.results_dir) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    gf, pf, tf, diag, n_poss = [], [], [], {}, 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init,
                             initargs=(in_tag, a.fold, a.season, a.input_dir, flags)) as ex:
        for fut in as_completed([ex.submit(RE._run_block, j) for j in jobs]):
            g, p, d, np_, _, *tj = fut.result()
            gf.append(g)
            if tj and tj[0] is not None and len(tj[0]):
                tf.append(tj[0])
            if len(p):
                pf.append(p)
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + v
            n_poss += np_
    games = pd.concat(gf, ignore_index=True)
    traj_meta = None
    if tf:
        from cbb_sim.engine.trajectory import TrajectoryWriter
        tw = TrajectoryWriter(Path("results/trajectories") / a.tag / "trajectory.parquet", seed_limit=int(a.trajectory_seeds))
        for t in sorted(tf, key=lambda f: (int(f.seed.iloc[0]), int(f.game_id.iloc[0]))):
            tw.write(t)
        tw.close()
        nb = tw.path.stat().st_size
        traj_meta = {"path": str(tw.path), "rows": tw.rows, "game_seeds": tw.pairs, "bytes": nb,
                     "bytes_per_game_seed": round(nb / max(tw.pairs, 1), 1), "seed_limit": int(a.trajectory_seeds)}
        print("trajectory:", traj_meta)
    games.to_parquet(out / "games.parquet", index=False)
    if pf:
        pd.concat(pf, ignore_index=True).to_parquet(out / "players.parquet", index=False)
    meta = {"engine_tag": f"player_day1/{a.tag}", "created_at": datetime.now(UTC).isoformat(),
            "seeds": [int(s) for s in seeds], "n_seeds": len(seeds), "n_seeds_requested": a.seeds, "fold": a.fold,
            "backtest": True, "sealed_touched": False, "partial": False, "season": a.season, "n_games": len(game_rows),
            "n_rows": len(games), "possessions_simulated": int(n_poss), "runtime_s": round(time.time() - t0, 1),
            "workers": a.workers, "ast_is_placeholder": True, **prov, "input_dir": a.input_dir, "ids_file": a.ids_file,
            "inputs_tag": str(inp.meta.get("inputs_tag_loaded", in_tag)), "adapter_flags": ad.flags,
            "engine_rules_from_data": inp.rules, "inputs_meta": inp.meta, "diagnostics": diag,
            "overlay_v2_overrides": ov, "trajectory": traj_meta}
    if ov:
        import run_engine_overlay_v2 as OV
        bad = OV.check_sources(meta, ov)
        meta["overlay_v2_source_check"] = bad or "PASS"
        if bad:
            (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            raise SystemExit(f"overlay source check FAILED: {bad}")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out} ({len(games):,} rows, {len(game_rows)} games x {len(seeds)} seeds, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
