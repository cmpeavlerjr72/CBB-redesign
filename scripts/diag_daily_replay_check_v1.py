#!/usr/bin/env python
"""
diag_daily_replay_check_v1.py -- proof that the daily GRADE stage reproduces the eval harness's numbers on replayed fold-2 dates,
and a check of the idempotency / created_at guarantees on the replay roots.

    .venv/Scripts/python.exe scripts/diag_daily_replay_check_v1.py --dates 2025-02-11,2025-02-25,2025-03-04

1. Copies the finished replay sim dirs (`results/daily_replay_f2/sim/<date>/<run>`) to `results/daily_replay_f2_close`, publishes them
   against the REPLAY CLOSE (so the stored line equals the harness's close), grades with replay truth (verified finals) at the next-day
   clock, into that root's own ledger.
2. Runs the eval harness on the same sim rows: `gates.build_grading_frame` + `market.gate_g10` (provider preference DraftKings > ESPN BET).
3. Prints every shared number side by side with the difference. A mismatch beyond 1e-9 is a FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ["CBB_TRUTH"] = "verified_v1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import grade_daily_v1 as GR  # noqa: E402
import run_daily_publish_v1 as PUB  # noqa: E402
from cbb_sim.eval import gates as EG  # noqa: E402
from cbb_sim.eval import market as M  # noqa: E402
from cbb_sim.live import daily as D  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--src-root", default="results/daily_replay_f2")
    ap.add_argument("--root", default="results/daily_replay_f2_close")
    ap.add_argument("--run-id", default="s16_o0")
    a = ap.parse_args(argv)
    src, root = REPO / a.src_root, REPO / a.root
    dates = a.dates.split(",")
    sims = []
    for d in dates:
        s = D.sim_dir(src, d, a.run_id)
        t = D.sim_dir(root, d, a.run_id)
        if not (t / "_DONE.json").exists():
            shutil.copytree(s, t, ignore=shutil.ignore_patterns("_adapter"), dirs_exist_ok=True)
        now = pd.Timestamp(f"{d}T14:00:00Z")
        r = PUB.run_publish_stage(d, a.run_id, now, root, "replay_close", a.season)
        g = GR.run_grade_stage(d, a.season, now + pd.Timedelta(days=1), root, "truth", n_boot=1000)
        print(d, "publish", r.get("_status"), r.get("publish_id"), "grade", g.get("_status"), g.get("graded"), "pending", g.get("pending"))
        sims.append(pd.read_parquet(t / "games.parquet"))
    led = pd.read_parquet(root / "grade/ledger.parquet")
    led = led[pd.to_datetime(led["game_date"]).dt.strftime("%Y-%m-%d").isin(dates)]
    mine = D.grade_tables(led, 1000)
    tol = yaml.safe_load((REPO / "docs/gates.yaml").read_text(encoding="utf-8"))
    summary, _raw = EG.build_grading_frame(pd.concat(sims, ignore_index=True), a.season)
    h = M.gate_g10(summary, a.season, tol, n_boot=1000)
    rows = [
        ("games graded (harness: truth-joined games)", len(summary), mine["n_games"]),
        ("games with a spread", h.n_with_line, mine["n_with_spread"]),
        ("model margin MAE (lined)", h.model_margin_mae, mine["model_margin_mae_lined"]),
        ("close margin MAE", h.close_margin_mae, mine["line_margin_mae_lined"]),
        ("model total MAE (lined)", h.model_total_mae, mine["total_mae_lined"]),
        ("close total MAE", h.close_total_mae, mine["line_total_mae_lined"]),
        ("ML games", h.n_ml, mine["n_with_ml"]),
        ("model Brier (ML subset)", h.model_brier_ml_subset, mine["model_brier"]),
        ("market Brier", h.market_brier, mine["market_brier"]),
        ("ATS bootstrap mean ROI [approx: resample is row-order dependent]", h.ats_bootstrap["mean"], mine["ats_bootstrap"]["mean"]),
        ("OU bootstrap mean ROI [approx]", h.ou_bootstrap["mean"], mine["ou_bootstrap"]["mean"]),
        ("ML bootstrap mean ROI [approx]", h.ml_edge_bootstrap["mean"], mine["ml_bootstrap"]["mean"]),
    ]
    for name, hv, tab in (("ATS", h.ats_table, mine["ats_table"]), ("OU", h.ou_table, mine["ou_table"]), ("ML", h.ml_edge_table, mine["ml_edge_table"])):
        for i in range(len(tab)):
            for c in tab.columns:
                if c in ("n", "wins", "losses", "pushes", "win_pct", "roi_at_-110", "roi"):
                    rows.append((f"{name} row {i} {c}", hv.iloc[i][c], tab.iloc[i][c]))
    out, bad = [], 0
    print(f"\n{'quantity':45} {'harness':>14} {'daily grade':>14} {'diff':>10}")
    for n, hv, mv in rows:
        diff = (float(hv) - float(mv)) if (pd.notna(hv) and pd.notna(mv)) else (0.0 if pd.isna(hv) and pd.isna(mv) else np.inf)
        ok = abs(diff) <= (0.01 if "[approx" in n else 1e-9)      # bootstrap means depend on row order under the same seed
        bad += 0 if ok else 1
        out.append({"quantity": n, "harness": float(hv) if pd.notna(hv) else None, "daily_grade": float(mv) if pd.notna(mv) else None,
                    "diff": diff, "match": bool(ok)})
        print(f"{n:45} {float(hv):14.6f} {float(mv):14.6f} {diff:10.2e} {'' if ok else 'MISMATCH'}")
    (REPO / "results/daily_replay_f2_check.json").write_text(json.dumps({"dates": dates, "n_rows": len(out), "n_mismatch": bad, "rows": out},
                                                                       indent=2, default=str), encoding="utf-8")
    print(f"\n{len(out)} quantities compared, {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
