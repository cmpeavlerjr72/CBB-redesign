#!/usr/bin/env python
"""
diag_day1_rehearsal_v1.py -- seal-safe rehearsal of the day-1 flow on an OPENING day of a past, unsealed season (lane F2, 2026-09-30).

Opening day of fold-2's test season (2024-11-04, season 2025) is the same situation as 2026-11-02: no game of the season played, the prior season's tables carry.
Steps (one core): (1) DAY1.stage_ratings builds the as-of ratings through the chain's day-1 branch (choices file with the SERVED policy; no 2026 read);
(2) sim A = run_sim_stage with `ratings_dir` = that output; (3) sim B = the same run with ratings_dir=None (stored batch ratings, lane F's replay path);
(4) compare games.parquet row by row. Expected: identical up to the ratings parity tolerance (1e-14), i.e. the new ratings_dir plumbing changes nothing.

    .venv/Scripts/python.exe scripts/diag_day1_rehearsal_v1.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import chain_daily_v2 as V2  # noqa: E402
import chain_day1_2027_v1 as DAY1  # noqa: E402
import run_daily_sim_v1 as SIM  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate-date", default="2024-11-04")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-games", type=int, default=0)
    a = ap.parse_args(argv)
    assert a.season < 2026, "rehearsal is seal-safe only on unsealed seasons"
    root = REPO / "results/daily_dry/rehearsal"
    t0 = time.time()
    CD = V2._load("chain_daily")
    tmp = root / "choices.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    served = json.loads(DAY1.CHOICES_PATH.read_text(encoding="utf-8"))
    served["teams_source"] = "tg"           # opening day of a past season: the schedule-based team set is the game-based one (parity json)
    tmp.write_text(json.dumps(served), encoding="utf-8")
    ctx = SimpleNamespace(slate_date=pd.Timestamp(a.slate_date).date(), dry_run=True, state={})
    r = DAY1.stage_ratings(ctx, CD, None, season=a.season, choices_path=tmp)
    print("ratings:", r, f"{time.time() - t0:.0f}s", flush=True)
    now = pd.Timestamp(a.slate_date, tz="America/New_York") - pd.Timedelta(hours=4)       # 20:00 ET the evening before
    out = {}
    for tag, rd in (("A_entrypoint_ratings", ctx.state["ratings_dir"]), ("B_stored_ratings", None)):
        res = SIM.run_sim_stage(a.slate_date, a.season, "F2", a.seeds, 0, now, root / tag, None, "universe", replay=True,
                                max_games=a.max_games, ratings_dir=rd)
        out[tag] = res
        print(tag, res, flush=True)
    ga = pd.read_parquet(Path(out["A_entrypoint_ratings"]["out"]) / "games.parquet").drop(columns="created_at")
    gb = pd.read_parquet(Path(out["B_stored_ratings"]["out"]) / "games.parquet").drop(columns="created_at")
    same_shape = ga.shape == gb.shape
    num = [c for c in ga.columns if pd.api.types.is_numeric_dtype(ga[c])]
    mx = float(np.nanmax(np.abs(ga[num].to_numpy(float) - gb[num].to_numpy(float)))) if same_shape else None
    res = {"slate_date": a.slate_date, "season": a.season, "seeds": a.seeds, "rows": int(len(ga)), "same_shape": same_shape,
           "max_abs_diff_all_numeric_columns": mx, "n_games": int(ga["game_id"].nunique()), "clock": str(now), "runtime_s": round(time.time() - t0, 1)}
    print(json.dumps(res))
    (REPO / "results/day1_rehearsal_2024-11-04.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
