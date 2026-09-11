#!/usr/bin/env python
"""
run_rot5_closed_loop.py -- the Decision-10 closed-loop check for the rotation
round-5 wave family: a paired-stream engine run with the rotation's engine-produced state
LIVE against the same run with it FROZEN.

Pre-registration: `docs/models/rotation/experiments.md` section 12.9.

    .venv/Scripts/python.exe scripts/run_rot5_closed_loop.py --arm round5 --round5-arm W4
    .venv/Scripts/python.exe scripts/run_rot5_closed_loop.py --arm round5 --round5-arm W4 --freeze
    .venv/Scripts/python.exe scripts/run_rot5_closed_loop.py --arm round5 --round5-arm W4 --nostate
    .venv/Scripts/python.exe scripts/run_rot5_closed_loop.py --arm round4 --round4-arm H1
    .venv/Scripts/python.exe scripts/run_rot5_closed_loop.py --grade

L31: a freeze DETECTS a loop and only a refit SIZES one, so `--nostate` runs the
arm whose wave tables and composition hazards were REFITTED without any margin
or foul term (`scripts/train_rotation_v5_nostate.py`), live, against the live
arm. Both are reported before any magnitude is quoted.

WHY IT EXISTS AND `run_engine.py` IS NOT EDITED
-----------------------------------------------
Decision 10: a sub-model feature the engine itself produces is adopted only
after a paired-stream run with the feature live vs frozen keeps margin SD ratio,
home/away score correlation and possessions per game inside the G1/G2
tolerances. The rotation consumes `margin` and `fouls`, both engine-produced.
The incumbent R2 consumes them too, through `TiltTables.state` and
`TiltTables.foul`, and has **never** had this check run (L25), so it is run here
for R2 as well as for the round-4 winner.

`ENGINE_ROTATION_FREEZE=1` (added in `engine/rotation_adapter.py`) holds, for the
rotation model only, the margin at its pregame value (0) and the personal- and
team-foul counts at theirs. Foul accrual, the foul-out eviction rule and the
box-score foul counters stay live.

The game subset is the same stride rule `run_clk3c_closed_loop.py` uses -- the
F2 2025 slate sorted by `game_id` ascending, every 11th row, the first 500 --
so the two Decision-10 checks are run on the same games and `--max-games`'s
November-only bias is avoided.

ONE STATED HACK. `engine/adapters.py` still refuses `ENGINE_ROTATION != reference`
("the rotation bake-off adopted nothing"), and that file belongs to another
deliverable. This runner therefore loads the adapters with `ENGINE_ROTATION=
reference` and sets the real mode immediately afterwards, inside the worker,
before any possession is simulated -- which is exactly what a one-line
relaxation of that guard would do. The relaxation itself is left to that file's
owner, and `run_meta.json` records the mode that actually ran.
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
_PIN = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS")

#: Held fixed and named on every run of every arm, so an arm run now and an arm
#: run in three hours are the same comparison.
PINNED_SUBMODELS = {
    "ENGINE_EVENT": "round2_s1",
    "ENGINE_FG_MAKE": "round3_shooter_S_C_s1",
    "ENGINE_CLOCK": "reference",
}

SUBSET_STRIDE = 11
SUBSET_N = 500

_W: dict = {}


def _init_worker(tag: str, fold: str, season: int, input_dir: str, flags: dict,
                 rot_mode: str) -> None:
    for k in _PIN:
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = str(v)
    os.environ["ENGINE_ROTATION"] = "reference"      # see the module docstring
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(input_dir, tag)
    ad = Adapters.load(inp, fold, season)
    os.environ["ENGINE_ROTATION"] = rot_mode
    ad.flags["ENGINE_ROTATION"] = rot_mode
    ad.flags["ENGINE_ROTATION_FREEZE"] = os.environ.get("ENGINE_ROTATION_FREEZE", "0")
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


# ---------------------------------------------------------------------------
def run(args) -> int:
    if int(args.season) >= 2026 and os.environ.get("CBB_UNSEAL") != "1":
        raise SystemExit(f"season {args.season} is SEALED")
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(args.season)], context="rotation round 4 closed loop")

    t0 = time.time()
    for k in _PIN:
        os.environ.setdefault(k, "1")
    flags = dict(PINNED_SUBMODELS)
    flags["ENGINE_ROTATION_FREEZE"] = "1" if args.freeze else "0"
    if args.arm == "round4":
        flags["ENGINE_ROTATION_ARM"] = args.round4_arm
    if args.arm == "round5":
        flags["ENGINE_ROTATION_ARM"] = args.round5_arm
        if args.nostate:
            flags["ENGINE_ROTATION_MANIFEST"] = "nostate"
    for k, v in flags.items():
        os.environ[k] = str(v)

    from cbb_sim.engine.inputs import EngineInputs
    in_tag = f"{args.fold}_{args.season}"
    inp = EngineInputs.load(args.input_dir, in_tag)
    rows = subset_rows(inp.games)
    if len(rows) != SUBSET_N:
        raise SystemExit(f"subset rule produced {len(rows)} games, expected {SUBSET_N}")
    sub = inp.games.iloc[rows]
    print(f"rotation {args.arm}{' FROZEN' if args.freeze else ''}: {len(rows)} games "
          f"{sub['game_date'].min().date()}..{sub['game_date'].max().date()}, "
          f"{args.seeds} seeds, {args.workers} workers", flush=True)

    prov = None
    if args.arm == "round4":
        from cbb_sim.engine import rotation_adapter as RA
        prov = RA.load_round4(inp.games)["provenance"]
        print("  round-4 manifest:", json.dumps(prov, default=str), flush=True)
    if args.arm == "round5":
        from cbb_sim.engine import rotation_adapter as RA
        r5 = RA.load_round5(inp.games)
        prov = r5["provenance"]
        prov["arm"] = r5["arm"]
        prov["draw_exit"], prov["draw_entry"] = r5["draw_exit"], r5["draw_entry"]
        prov["coupled"] = r5["coupled"]
        prov["hazards"] = sorted(set(r5["hazard_sources"]))
        print("  round-5 manifest:", json.dumps(prov, default=str), flush=True)

    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    jobs = []
    for s0 in range(0, len(seeds), args.seeds_per_block):
        sb = seeds[s0:s0 + args.seeds_per_block]
        for g0 in range(0, len(rows), args.games_per_block):
            jobs.append((rows[g0:g0 + args.games_per_block], sb, True))

    gframes, pframes, diag, n_poss = [], [], {}, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(in_tag, args.fold, int(args.season),
                                       args.input_dir, flags, args.arm)) as ex:
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
        "round": "rotation round 5 (docs/models/rotation/experiments.md section 12.9)",
        "created_at": datetime.now(UTC).isoformat(),
        "rotation_arm": args.arm, "round4_variant": args.round4_arm,
        "round5_variant": args.round5_arm, "refit_without_state": bool(args.nostate),
        "rotation_freeze": bool(args.freeze),
        "pinned_submodels": PINNED_SUBMODELS,
        "seeds": sorted(int(s) for s in full), "n_seeds": int(len(full)),
        "fold": args.fold, "season": int(args.season), "backtest": True,
        "sealed_touched": False, "n_games": SUBSET_N, "n_rows": int(len(games)),
        "subset_rule": (f"F2 {args.season} slate sorted by game_id ascending, every "
                        f"{SUBSET_STRIDE}th row, first {SUBSET_N}"),
        "game_ids": [int(x) for x in sub["game_id"].to_numpy()],
        "rotation_manifest": prov,
        "possessions_simulated": int(n_poss),
        "runtime_s": round(elapsed, 1),
        "possessions_per_second": round(n_poss / max(elapsed, 1e-9), 1),
        "adapters_loaded_as": "reference, then ENGINE_ROTATION set in-worker "
                              "(adapters.py guard is owned elsewhere)",
        "diagnostics": diag,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                       encoding="utf-8")
    print(f"wrote {out} ({len(games):,} rows, {len(full)} seeds, {elapsed:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
def _stats(tag: str, results_dir: Path, actual_min: pd.DataFrame | None) -> dict:
    d = results_dir / tag
    g = pd.read_parquet(d / "games.parquet")
    g["margin"] = g["home_pts"] - g["away_pts"]
    per = g.groupby("game_id").agg(margin=("margin", "mean"),
                                   possessions=("possessions", "mean"),
                                   home=("home_pts", "mean"), away=("away_pts", "mean"))
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
    if actual_min is not None and p.exists():
        pl = pd.read_parquet(p)
        pl = pl[pl["cbbd_id"].notna()].copy()
        pl["cbbd_id"] = pl["cbbd_id"].astype("int64")
        # only games the truth side actually has (`load_team_possessions` keeps
        # games whose on-floor set is complete on every possession); a game with
        # no truth would otherwise score its whole roster as error.
        have = set(actual_min["game_id"].unique().tolist())
        pl = pl[pl["game_id"].isin(have)]
        out["player_minutes_games"] = int(pl["game_id"].nunique())
        # BOTH sides restricted to the run's own games before the outer join --
        # otherwise every truth row of the whole season enters as pure error.
        act = actual_min[actual_min["game_id"].isin(set(pl["game_id"].unique()))]
        per_seed = []
        for seed, gg in pl.groupby("seed"):
            m = gg[["game_id", "team_id", "cbbd_id", "minutes"]].merge(
                act, on=["game_id", "cbbd_id"], how="outer", suffixes=("", "_act"))
            m["minutes"] = m["minutes"].fillna(0.0)
            m["act"] = m["act"].fillna(0.0)
            per_seed.append(float((m["minutes"] - m["act"]).abs().mean()))
        out["player_minutes_mae"] = float(np.mean(per_seed))
        out["player_minutes_mae_seed_sd"] = float(
            np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0)
    return out


def grade(args) -> int:
    from cbb_sim.models import rotation as R
    res = Path(args.results_dir)
    tags = args.tags.split(",")
    actual_min = None
    try:
        tp = R.load_team_possessions(int(args.season))
        pg = R.player_game_minutes(tp)
        actual_min = pg[["game_id", "pid", "minutes"]].rename(
            columns={"pid": "cbbd_id", "minutes": "act"})
        actual_min["game_id"] = actual_min["game_id"].astype("int64")
        actual_min["cbbd_id"] = actual_min["cbbd_id"].astype("int64")
    except Exception as e:                       # noqa: BLE001
        print("actual minutes unavailable:", e)

    rowsd = [_stats(t, res, actual_min) for t in tags if (res / t).exists()]
    truth = None
    try:
        from cbb_sim.eval import reference as REF
        truth = REF.season_truth(int(args.season)) if hasattr(REF, "season_truth") else None
    except Exception:                            # noqa: BLE001
        pass
    payload = {"generated_at": datetime.now(UTC).isoformat(),
               "season": int(args.season), "rows": rowsd}
    if truth is not None:
        payload["truth_note"] = "season truth available; see eval_gates.py for G1/G2"
    outp = res / "rot5_closed_loop_summary.json"
    outp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    hdr = ["tag", "n_games", "n_seeds", "margin_sd_per_sim", "home_away_corr",
           "possessions", "total", "player_minutes_mae"]
    print(" | ".join(hdr))
    for r in rowsd:
        print(" | ".join(str(round(r[h], 4)) if isinstance(r.get(h), float)
                         else str(r.get(h, "-")) for h in hdr))
    print("wrote", outp)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="round5",
                    choices=["reference", "round4", "round5"])
    ap.add_argument("--round4-arm", default="H1", choices=["H1", "H2"],
                    help="H1 = hard second-half reset, H2 = earned reset")
    ap.add_argument("--round5-arm", default="W1",
                    choices=["W1", "W2", "W3", "W4", "W5"])
    ap.add_argument("--nostate", action="store_true",
                    help="the L31 refit-WITHOUT-state arm, run LIVE")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games-per-block", type=int, default=40)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--tag", default="")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--input-dir", default=str(INPUT_DIR))
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--tags", default="rot5_W1_live,rot5_W1_frozen,rot5_W1_nostate,"
                                      "rot5_H1_live,rot5_H1_frozen")
    args = ap.parse_args()
    if args.grade:
        return grade(args)
    if not args.tag:
        nm = {"reference": "R2", "round4": args.round4_arm,
              "round5": args.round5_arm}[args.arm]
        sfx = "nostate" if args.nostate else ("frozen" if args.freeze else "live")
        args.tag = f"rot5_{nm}_{sfx}"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
