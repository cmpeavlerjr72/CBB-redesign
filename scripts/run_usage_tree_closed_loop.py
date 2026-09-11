#!/usr/bin/env python
"""
run_usage_tree_closed_loop.py -- the Decision-10 closed-loop check for the
usage round-3 (corrected `score_diff`) LightGBM tree: a paired-stream engine
run with the tree's engine-produced state LIVE, FROZEN at pregame values, and
REFIT-WITHOUT-state, against the served `reference` (U1 proportional).

Pre-registration: `docs/models/usage/experiments.md` section 11.
Verdict: `docs/tests/usage_decision10_gate_2026-09-11.md`.

    .venv/Scripts/python.exe scripts/run_usage_tree_closed_loop.py --arm reference
    .venv/Scripts/python.exe scripts/run_usage_tree_closed_loop.py --arm tree_v3
    .venv/Scripts/python.exe scripts/run_usage_tree_closed_loop.py --arm tree_v3_freeze
    .venv/Scripts/python.exe scripts/run_usage_tree_closed_loop.py --arm tree_v3_nostate
    .venv/Scripts/python.exe scripts/run_usage_tree_closed_loop.py --grade

Follows `scripts/run_rot5_closed_loop.py`'s pattern exactly (same subset rule,
same pinned sub-models, same `simulate_chunk` batched worker) so the usage
arms are directly comparable to every other Decision-10 closed-loop report on
this machine. `ENGINE_USAGE` is read here (via `os.environ`, set before the
worker imports `adapters`) exactly as `run_rot5_closed_loop.py` does for
`ENGINE_ROTATION`; `adapters.py`'s DEFAULT (`reference`) is untouched.
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

DEFAULT_RESULTS = Path("results/engine_v0")
INPUT_DIR = Path("data/processed/models/engine")

#: Held fixed and named on every run of every arm (the rot5 precedent): an arm
#: run now and an arm run later are the same comparison. These are the CURRENT
#: engine defaults as of `docs/tests/engine_v1_gates_F2_2025_s200_2026-09-11.md`.
PINNED_SUBMODELS = {
    "ENGINE_EVENT": "round2_s1",
    "ENGINE_FG_MAKE": "round4_B1",
    "ENGINE_CLOCK": "v3c_srfloor_P3_s1",
    "ENGINE_REBOUND": "s1_weekly",
    "ENGINE_FREE_THROW": "s1_conf_aligned",
    "ENGINE_ROTATION": "reference",
}

SUBSET_STRIDE = 11
SUBSET_N = 500

_W: dict = {}


def _init_worker(tag: str, fold: str, season: int, input_dir: str, flags: dict) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = str(v)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(input_dir, tag)
    ad = Adapters.load(inp, fold, season)
    _W["inp"] = inp
    _W["ad"] = ad


def _run_block(job: tuple) -> tuple:
    from cbb_sim.engine import loop as L
    game_rows, seeds, keep_players = job
    inp, ad = _W["inp"], _W["ad"]
    gi = np.repeat(np.asarray(game_rows, dtype=np.int64), len(seeds))
    sd = np.tile(np.asarray(seeds, dtype=np.int64), len(game_rows))
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=keep_players)
    return res.games, res.players, res.diag, res.n_possessions


def subset_rows(games: pd.DataFrame) -> np.ndarray:
    order = np.argsort(games["game_id"].to_numpy(), kind="stable")
    return order[::SUBSET_STRIDE][:SUBSET_N]


def run(args) -> int:
    if int(args.season) >= 2026 and os.environ.get("CBB_UNSEAL") != "1":
        raise SystemExit(f"season {args.season} is SEALED")
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(args.season)], context="usage Decision-10 closed loop")

    t0 = time.time()
    flags = dict(PINNED_SUBMODELS)
    flags["ENGINE_USAGE"] = args.arm
    for k, v in flags.items():
        os.environ[k] = str(v)

    from cbb_sim.engine.inputs import EngineInputs
    in_tag = f"{args.fold}_{args.season}"
    inp = EngineInputs.load(args.input_dir, in_tag)
    rows = subset_rows(inp.games)
    if len(rows) != SUBSET_N:
        raise SystemExit(f"subset rule produced {len(rows)} games, expected {SUBSET_N}")
    sub = inp.games.iloc[rows]
    print(f"usage {args.arm}: {len(rows)} games "
          f"{sub['game_date'].min().date()}..{sub['game_date'].max().date()}, "
          f"{args.seeds} seeds, {args.workers} workers", flush=True)

    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    jobs = []
    for s0 in range(0, len(seeds), args.seeds_per_block):
        sb = seeds[s0:s0 + args.seeds_per_block]
        for g0 in range(0, len(rows), args.games_per_block):
            jobs.append((rows[g0:g0 + args.games_per_block], sb, True))

    gframes, pframes, diag, n_poss = [], [], {}, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(in_tag, args.fold, int(args.season),
                                       args.input_dir, flags)) as ex:
        futs = [ex.submit(_run_block, j) for j in jobs]
        for fut in as_completed(futs):
            g, pl, d, np_ = fut.result()
            gframes.append(g)
            if pl is not None and len(pl):
                pframes.append(pl)
            for kk, v in d.items():
                diag[kk] = diag.get(kk, 0) + v
            n_poss += np_

    games = pd.concat(gframes, ignore_index=True)
    per_seed = games.groupby("seed")["game_id"].nunique()
    full = per_seed[per_seed == SUBSET_N].index.to_numpy()
    if not len(full):
        raise SystemExit("no seed completed every game in the subset")
    games = games[games["seed"].isin(full)].reset_index(drop=True)

    out = Path(args.results_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out / "games.parquet", index=False)
    if pframes:
        players = pd.concat(pframes, ignore_index=True)
        players = players[players["seed"].isin(full)].reset_index(drop=True)
        players.to_parquet(out / "players.parquet", index=False)

    elapsed = time.time() - t0
    meta = {
        "engine_tag": f"engine_v0/{args.tag}",
        "round": "usage Decision-10 closed loop (docs/models/usage/experiments.md section 11)",
        "created_at": datetime.now(UTC).isoformat(),
        "usage_arm": args.arm,
        "pinned_submodels": PINNED_SUBMODELS,
        "seeds": sorted(int(s) for s in full), "n_seeds": int(len(full)),
        "fold": args.fold, "season": int(args.season), "backtest": True,
        "sealed_touched": False, "n_games": SUBSET_N, "n_rows": int(len(games)),
        "subset_rule": (f"F2 {args.season} slate sorted by game_id ascending, every "
                        f"{SUBSET_STRIDE}th row, first {SUBSET_N}"),
        "game_ids": [int(x) for x in sub["game_id"].to_numpy()],
        "possessions_simulated": int(n_poss),
        "runtime_s": round(elapsed, 1),
        "possessions_per_second": round(n_poss / max(elapsed, 1e-9), 1),
        "diagnostics": diag,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                       encoding="utf-8")
    print(f"wrote {out} ({len(games):,} rows, {len(full)} seeds, {elapsed:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
def _usage_stats(tag: str, results_dir: Path, asof: pd.DataFrame, season: int) -> dict:
    d = results_dir / tag
    g = pd.read_parquet(d / "games.parquet")
    g["margin"] = g["home_pts"] - g["away_pts"]
    per = g.groupby("game_id").agg(margin=("margin", "mean"),
                                   possessions=("possessions", "mean"))
    out = {
        "tag": tag,
        "n_games": int(g["game_id"].nunique()),
        "n_seeds": int(g["seed"].nunique()),
        "margin_sd_per_sim": float(g["margin"].std(ddof=1)),
        "margin_sd_per_game_mean": float(per["margin"].std(ddof=1)),
        "home_away_corr": float(np.corrcoef(g["home_pts"], g["away_pts"])[0, 1]),
        "possessions": float(g["possessions"].mean()),
        "total": float((g["home_pts"] + g["away_pts"]).mean()),
    }
    p = d / "players.parquet"
    if not p.exists():
        return out
    pl = pd.read_parquet(p)
    pl = pl[pl["cbbd_id"].notna()].copy()
    pl["cbbd_id"] = pl["cbbd_id"].astype("int64")
    pl["usage_events"] = pl[["fga", "fta", "tov"]].sum(axis=1) if "tov" in pl.columns \
        else pl["fga"] + pl["fta"]

    # ---- per-player FGA share distribution + top-1/top-3 usage gap --------
    team_tot = pl.groupby(["seed", "game_id", "team_id"])["usage_events"].transform("sum")
    pl["share"] = np.where(team_tot > 0, pl["usage_events"] / team_tot.replace(0, np.nan), 0.0)
    rank = pl.groupby(["seed", "game_id", "team_id"])["usage_events"].rank(
        ascending=False, method="first")
    top1 = pl.loc[rank == 1].groupby(["seed"])["share"].mean()
    top3 = (pl.assign(top3=(rank <= 3).astype(float) * pl["share"])
           .groupby(["seed", "game_id", "team_id"])["top3"].sum())
    out["top1_share_mean"] = float(top1.mean())
    out["top3_share_mean"] = float(top3.mean())
    out["fga_share_p10_p50_p90"] = [float(x) for x in
                                    pl["share"].quantile([0.1, 0.5, 0.9]).to_list()]

    # ---- per-player quintile responsiveness: pregame usage rate vs sim FGA share
    a = asof[asof["season"] == season][["player_id", "game_id", "rate_total"]].copy()
    a = a.rename(columns={"player_id": "cbbd_id"})
    m = pl.merge(a, on=["cbbd_id", "game_id"], how="inner")
    if len(m) > 500:
        m["q"] = pd.qcut(m["rate_total"], 5, duplicates="drop", labels=False)
        g5 = m.groupby("q").agg(pred=("share", "mean"), driver=("rate_total", "mean"))
        if len(g5) >= 2:
            slope = np.polyfit(g5["driver"], g5["pred"], 1)[0]
            realised_span = float(g5["pred"].max() - g5["pred"].min())
            out["quintile_n"] = int(len(g5))
            out["quintile_pred_span_pp"] = round(realised_span * 100, 4)
            out["quintile_slope"] = float(slope)
    return out


def grade(args) -> int:
    from cbb_sim.models import usage as U
    res = Path(args.results_dir)
    tags = args.tags.split(",")
    asof = pd.read_parquet(ROOT / "data/processed/models/usage_v2/asof_v2_shotshooter.parquet")
    rows = [_usage_stats(t, res, asof, int(args.season)) for t in tags if (res / t).exists()]
    payload = {"generated_at": datetime.now(UTC).isoformat(),
               "season": int(args.season), "rows": rows}
    outp = res / "usage_tree_closed_loop_summary.json"
    outp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    hdr = ["tag", "n_games", "n_seeds", "margin_sd_per_sim", "home_away_corr",
          "possessions", "total", "top1_share_mean", "top3_share_mean", "quintile_slope"]
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(round(r[h], 4)) if isinstance(r.get(h), float)
                         else str(r.get(h, "-")) for h in hdr))
    print("wrote", outp)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="reference",
                    choices=["reference", "tree_v3", "tree_v3_freeze", "tree_v3_nostate"])
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--games-per-block", type=int, default=40)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--tag", default="")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--input-dir", default=str(INPUT_DIR))
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--tags", default="usage_reference,usage_tree_v3_live,"
                                      "usage_tree_v3_frozen,usage_tree_v3_nostate")
    args = ap.parse_args()
    if args.grade:
        return grade(args)
    if not args.tag:
        args.tag = f"usage_{args.arm}"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
