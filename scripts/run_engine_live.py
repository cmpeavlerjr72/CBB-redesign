"""
run_engine_live.py -- run the possession engine on LIVE inputs (games not yet played).

    .venv/Scripts/python.exe scripts/run_engine_live.py --tag LIVE_F2_2025_2024-11-12 \
        --fold F2 --season 2025 --seeds 3 --replay --created-at 2024-11-12T12:00:00Z

Differences from `run_engine.py` (which is untouched and imported only for
`engine_provenance`):
  * in-process, ONE core (pool-free), thread pins set before numpy loads;
  * inputs come from `build_engine_inputs_live.py`; there is no final score anywhere;
  * every output row carries `created_at` and `tipoff_utc` and `created_at < tipoff`
    is ASSERTED (cbb_sim.live.guards) before anything is written; a violation raises;
  * `run_meta.json` says `backtest: false`, `live: true`, and `replay: true` when the
    caller is re-running a past slate with a pretend `--created-at`;
  * season 2026 stays sealed (CBB_UNSEAL=1 required); 2027 and later are allowed.

`--slice-from-backtest FOLD_SEASON_TAG` runs the SAME games out of an existing backtest
input set (parity control): it slices `arrays_{tag}` down to the slate's game ids so the
engine sees the backtest builder's arrays for exactly these games and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.live import guards as G  # noqa: E402


def slice_inputs(inp: EngineInputs, game_ids) -> EngineInputs:
    """A new EngineInputs holding only `game_ids`, in that order."""
    pos = {int(g): i for i, g in enumerate(inp.games["game_id"])}
    idx = np.array([pos[int(g)] for g in game_ids])
    kw = {}
    for f in ("team_static", "slot_static", "roster_cbbd", "roster_espn", "roster_valid", "rot_share",
              "rot_srank", "rot_start", "rot_fpm", "rot_pavail", "usage_rate", "reb_rate"):
        kw[f] = getattr(inp, f)[idx]
    return EngineInputs(games=inp.games.iloc[idx].reset_index(drop=True), team_names=inp.team_names,
                        slot_names=inp.slot_names, usage_classes=inp.usage_classes, rules=inp.rules,
                        meta=dict(inp.meta), **kw)


def prepare_adapter_dir(block: np.ndarray, fold: str, season: int, dest: Path) -> Path:
    """The served event adapter (`ENGINE_EVENT=round2_s1`) does not read `team_static`: it reads a
    (G, 2, 16) round-2 team block persisted per game ROW of the backtest universe
    (`event_round2_s1_{fold}_{season}/team_block.npz`). For live rows that block is built by
    `build_engine_inputs_live.py`; this writes a private adapter directory holding it, the
    unchanged model/manifest files (absolute paths) and copies of the three engine joblibs, and
    the runner points `adapters.ENGINE_DIR` at it. Nothing under data/processed/models/engine is written."""
    import shutil
    src = ROOT / "data/processed/models/engine"
    d = dest / f"event_round2_s1_{fold}_{season}"
    d.mkdir(parents=True, exist_ok=True)
    src_d = src / f"event_round2_s1_{fold}_{season}"
    idx = json.loads((src_d / "index.json").read_text(encoding="utf-8"))
    for pop in idx["populations"].values():
        for sg in pop["segments"]:
            sg["file"] = str((src_d / sg["file"]).resolve())
    (d / "index.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")
    np.savez_compressed(d / "team_block.npz", team_block=block.astype(np.float32))
    for name in (f"fg_make_FGA_3_decision8_{fold}.joblib", f"free_throw_{fold}.joblib", f"rebound_{fold}.joblib"):
        if not (dest / name).exists():
            shutil.copy2(src / name, dest / name)
    return dest


def prepare_from_overlay(overlay: Path, fold: str, season: int, dest: Path, n_rows: int = 0) -> dict:
    """Serve a `build_engine_inputs_v3_tag_v1.py` overlay in-process (no docker): ENGINE_DIR points at a scratch
    dir holding the overlay's event dir plus the three engine joblibs; FG_DIR / RB_S1_MANIFEST point at the overlay's
    fg_make / rebound trees when those exist (else the served ones stay)."""
    import shutil
    src = ROOT / "data/processed/models/engine"
    m = overlay / "data/processed/models"
    dest.mkdir(parents=True, exist_ok=True)
    ev = dest / f"event_round2_s1_{fold}_{season}"
    if ev.exists():
        shutil.rmtree(ev)
    shutil.copytree(m / f"engine/event_round2_s1_{fold}_{season}", ev)
    if n_rows:                                   # prefix of the rows: the block is indexed by input row
        tb = np.load(ev / "team_block.npz")["team_block"][:n_rows]
        np.savez_compressed(ev / "team_block.npz", team_block=tb)
    for name in (f"fg_make_FGA_3_decision8_{fold}.joblib", f"free_throw_{fold}.joblib", f"rebound_{fold}.joblib"):
        if not (dest / name).exists():
            shutil.copy2(src / name, dest / name)
    over = {"ENGINE_DIR": dest}
    if (m / "fg_make").exists():
        over["FG_DIR"] = m / "fg_make"
    if (m / "rebound/s1_confirm/S1_weekly/F2/manifest.json").exists():
        over["RB_S1_MANIFEST"] = m / "rebound/s1_confirm/S1_weekly/F2/manifest.json"
    return over


def simulate(inp: EngineInputs, fold: str, season: int, seeds, keep_players=True, games_per_block=20,
             flags: dict | None = None, adapter_dir: Path | None = None, attr_overrides: dict | None = None):
    from cbb_sim.engine import adapters as AD
    from cbb_sim.engine import loop as L
    from cbb_sim.engine.adapters import Adapters
    if adapter_dir is not None:
        AD.ENGINE_DIR = Path(adapter_dir)
    for k, v in (attr_overrides or {}).items():
        setattr(AD, k, Path(v))
    for k, v in (flags or {}).items():
        os.environ[k] = str(v)
    ad = Adapters.load(inp, fold, int(season))
    gframes, pframes = [], []
    n = inp.n_games
    for g0 in range(0, n, games_per_block):
        rows = np.arange(g0, min(g0 + games_per_block, n))
        gi = np.repeat(rows, len(seeds))
        sd = np.tile(np.asarray(seeds, dtype=np.int64), len(rows))
        res = L.simulate_chunk(inp, ad, gi, sd, keep_players=keep_players)
        gframes.append(res.games)
        if len(res.players):
            pframes.append(res.players)
    games = pd.concat(gframes, ignore_index=True)
    players = pd.concat(pframes, ignore_index=True) if pframes else None
    return games, players, ad


def stamp_rows(games: pd.DataFrame, inp: EngineInputs, created_at, per_game: bool = False) -> pd.DataFrame:
    """Attach tipoff_utc + created_at to every row and enforce created_at < tipoff.
    `per_game`: replay of many dates, use each game's own build-time created_at column."""
    cols = ["game_id", "tipoff_utc"] + (["created_at"] if per_game else [])
    info = inp.games[cols].drop_duplicates("game_id")
    out = games.merge(info, on="game_id", how="left")
    if not per_game:
        out["created_at"] = pd.Timestamp(created_at)
    G.assert_created_before_tipoff(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--input-dir", default="data/processed/models/engine_live")
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--inputs-version", default="v1", help="v1 = bare tag (live and v3 dirs); v2 for the served backtest inputs")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--created-at", default=None, help="default: now (UTC). A past slate needs --replay")
    ap.add_argument("--replay", action="store_true", help="re-running a past slate with a pretend created_at")
    ap.add_argument("--slice-from-backtest", default=None,
                    help="e.g. F2_2025 : take the arrays for the same game ids from the backtest inputs "
                         "(parity control) instead of the live arrays")
    ap.add_argument("--hybrid-slots-from-backtest", default=None,
                    help="e.g. F2_2025 : LIVE team/roster/rotation/event arrays, but slot_static, usage_rate and "
                         "reb_rate taken from the backtest arrays (isolates the stale-row differences)")
    ap.add_argument("--backtest-dir", default="data/processed/models/engine")
    ap.add_argument("--out-dir", default="results/engine_live")
    ap.add_argument("--no-players", action="store_true")
    ap.add_argument("--subset-po4b", action="store_true",
                    help="the standing 500-game subset (sorted by game_id, every 11th row, first 500)")
    ap.add_argument("--max-games", type=int, default=0, help="debug: first N rows of the (subset) inputs")
    ap.add_argument("--plain-out", action="store_true",
                    help="write results/engine_v0-style dir <out-dir>/<tag>_s<seeds> without the run tag suffixes")
    ap.add_argument("--overlay-dir", default=None,
                    help="<engine_v3_tag>/overlay from build_engine_inputs_v3_tag_v1.py: serve its artifacts in-process")
    ap.add_argument("--reverse", action="store_true", help="reverse game order (RNG-alignment control)")
    args = ap.parse_args()
    if args.season == 2026 and os.environ.get("CBB_UNSEAL") != "1":
        raise SystemExit("season 2026 is SEALED; set CBB_UNSEAL=1 deliberately to override")
    t0 = time.time()
    import run_engine as RE                     # for engine_provenance only
    prov = RE.engine_provenance()
    live = EngineInputs.load(args.input_dir, args.tag, version=args.inputs_version)
    inp = live
    bp_ = Path(args.input_dir) / f"event_block_{args.tag}.npz"
    block = np.load(bp_ if bp_.exists() else Path(args.input_dir) / f"event_round2_s1_{args.tag}/team_block.npz")["team_block"]
    if args.slice_from_backtest:
        bt = EngineInputs.load(args.backtest_dir, args.slice_from_backtest)
        inp = slice_inputs(bt, live.games["game_id"].tolist())
        btpos = {int(g): i for i, g in enumerate(bt.games["game_id"])}
        block = np.load(Path(args.backtest_dir) / f"event_round2_s1_{args.slice_from_backtest}/team_block.npz"
                        )["team_block"][[btpos[int(g)] for g in inp.games["game_id"]]]
        inp.games = inp.games.merge(live.games[["game_id", "created_at"]], on="game_id", how="left") \
            if "created_at" in live.games.columns else inp.games
    if args.hybrid_slots_from_backtest:
        bt = EngineInputs.load(args.backtest_dir, args.hybrid_slots_from_backtest)
        bp = {int(g): i for i, g in enumerate(bt.games["game_id"])}
        bidx = np.array([bp[int(g)] for g in inp.games["game_id"]])
        inp.slot_static = bt.slot_static[bidx]
        inp.usage_rate = bt.usage_rate[bidx]
        inp.reb_rate = bt.reb_rate[bidx]
    if args.reverse:
        order = np.arange(inp.n_games)[::-1]
        inp = slice_inputs(inp, inp.games["game_id"].to_numpy()[order])
        block = block[order]
    if args.subset_po4b:
        order = np.argsort(inp.games["game_id"].to_numpy(), kind="stable")[::11][:500]
        inp = slice_inputs(inp, inp.games["game_id"].to_numpy()[order])
        block = block[order]
    if args.max_games:
        inp = slice_inputs(inp, inp.games["game_id"].to_numpy()[:args.max_games])
        block = block[:args.max_games]
    per_game = "created_at" in inp.games.columns and args.replay
    created_at = pd.Timestamp(args.created_at) if args.created_at else pd.Timestamp.now("UTC")
    created_at = created_at.tz_localize("UTC") if created_at.tzinfo is None else created_at.tz_convert("UTC")
    if per_game:
        G.assert_created_before_tipoff(inp.games)
    elif not args.replay:
        G.assert_created_before_tipoff(inp.games.assign(created_at=created_at))
    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    over = None
    if args.overlay_dir:
        if args.subset_po4b or args.reverse or args.hybrid_slots_from_backtest:
            raise SystemExit("--overlay-dir serves the overlay's own event block by input row; only --max-games "
                             "(a row prefix) may be combined with it")
        adir = Path(args.out_dir) / "_adapter_dirs" / (args.tag + "_overlay")
        over = prepare_from_overlay(Path(args.overlay_dir), args.fold, args.season, adir, args.max_games)
        adir = None
    else:
      adir = prepare_adapter_dir(block, args.fold, args.season,
                               Path(args.out_dir) / "_adapter_dirs" / (args.tag + ("_bt" if args.slice_from_backtest else "")
                                                                       + ("_hyb" if args.hybrid_slots_from_backtest else "")
                                                                       + ("_rev" if args.reverse else "")))
    games, players, ad = simulate(inp, args.fold, args.season, seeds, keep_players=not args.no_players,
                                  adapter_dir=adir, attr_overrides=over)
    games = stamp_rows(games, inp, created_at, per_game)
    out = Path(args.out_dir) / (args.tag + ("__bt_arrays" if args.slice_from_backtest else "")
                                + ("__hybrid" if args.hybrid_slots_from_backtest else "")
                                + ("__rev" if args.reverse else "") + f"_s{args.seeds}")
    if args.plain_out:
        out = Path(args.out_dir) / f"{args.tag}_s{args.seeds}_o{args.seed_offset}"
    out.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out / "games.parquet", index=False)
    if players is not None:
        players = stamp_rows(players, inp, created_at, per_game)
        players.to_parquet(out / "players.parquet", index=False)
    meta = {"engine_tag": f"engine_live/{out.name}", "created_at": str(created_at), "live": True,
            "backtest": False, "replay": bool(args.replay), "input_tag": args.tag,
            "arrays_from": args.slice_from_backtest or "live", "fold": args.fold, "season": args.season,
            "seeds": seeds.tolist(), "n_games": int(inp.n_games), "n_rows": int(len(games)),
            "runtime_s": round(time.time() - t0, 1), **prov, "adapter_flags": ad.flags,
            "inputs_meta": inp.meta, "created_at_before_tipoff_asserted": True}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out} ({len(games)} rows, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
