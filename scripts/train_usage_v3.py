#!/usr/bin/env python
"""
train_usage_v3.py -- L4 SHOT ALLOCATION (usage), ROUND 3: the `score_diff`
state-feature data fix.

`scripts/diag_usage_state_confound.py`
(`docs/tests/usage_state_confound_2026-09-11.md`) found that
`usage.build_usage_events`'s `score_diff` is POST-OUTCOME: it is read off the
event's OWN row's `home_score`/`away_score`, so a made field goal's own row
already carries its own points (99.96-99.99% of made-shot rows across every
usage event class move by exactly the shot's own value; misses and turnovers
move by 0). This is the same defect L27 found in `fg_make`, reproduced here.
`sec_remaining`, `period` and `chance_number` are NOT affected (same evidence
doc) and are untouched.

THIS TRAINER CHANGES EXACTLY ONE THING relative to round 2
(`scripts/train_usage_v2.py`): `usage.build_usage_events(..., score_diff_mode=
"pre_outcome")`, the corrected column reconstructed from the END of the
PREVIOUS row of the same game (0 on a game's own first row) -- the same
quantity the live engine's `GameState.off_score_diff()` feeds before an event
resolves. The shooter key (`shot_shooter_id`, round 2's fix), arms, features,
folds, metrics, gates, decision rule, seeds and draw counts are all imported
from `train_usage_v2` / `train_usage_v1` and are otherwise byte-identical, so
the two rounds differ by the score_diff construction and by nothing else.

Artifacts go to a VERSIONED SIBLING directory (`data/processed/models/usage_v3`
by default). Nothing under `usage/` or `usage_v2/` is written or moved.

Usage:
    .venv/Scripts/python.exe scripts/train_usage_v3.py
    .venv/Scripts/python.exe scripts/train_usage_v3.py --classes FGA_3 --quick --skip-wf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import train_usage_v1 as V1  # noqa: E402
import train_usage_v2 as V2  # noqa: E402

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import usage as U  # noqa: E402

SEASONS = V1.SEASONS
SHOOTER_KEY = V2.SHOOTER_KEY
SCORE_DIFF_MODE = "pre_outcome"
ROUND2_RESULTS = ROOT / "data/processed/models/usage_v2/results_v2.json"


def build_tables(version: str, out_dir: Path, rebuild: bool, log):
    """`train_usage_v2.build_tables` with `score_diff_mode='pre_outcome'`
    threaded through. Kept as its own copy (rather than parameterising v2) so
    that v2 stays exactly the script that produced round 2."""
    ev_path = out_dir / f"events_{version}_prestate.parquet"
    asof_path = out_dir / f"asof_{version}_prestate.parquet"
    meta_path = out_dir / f"build_report_{version}_prestate.json"
    if ev_path.exists() and asof_path.exists() and meta_path.exists() and not rebuild:
        return (pd.read_parquet(ev_path), pd.read_parquet(asof_path),
                json.loads(meta_path.read_text()))

    universe = ES.load_universe(require_pbp_complete=True)
    rim = ES.rim_override_for_version(version)
    per_season, coverage, drops, relabel = {}, {}, {}, {}
    for s in SEASONS:
        raw = U.build_usage_events(s, universe, rim_override_max_ft=rim,
                                   shooter_key=SHOOTER_KEY,
                                   score_diff_mode=SCORE_DIFF_MODE)
        coverage[str(s)] = U.coverage_report(raw)
        drops[str(s)] = U.shooter_drop_report(raw)
        relabel[str(s)] = {"n_fga_relabelled": raw.attrs["n_fga_relabelled"],
                           "n_fga_no_shooter": raw.attrs["n_fga_no_shooter"]}
        log(f"  {s}: {raw.attrs['n_fga_relabelled']:,} FGA rows relabelled, "
            f"{raw.attrs['n_fga_no_shooter']:,} with no shooter id "
            f"(score_diff_mode={SCORE_DIFF_MODE})")
        per_season[s] = U.usable_events(raw)
    minutes = U.load_minutes([SEASONS[0] - 1, *SEASONS])
    asof = U.build_player_asof(per_season, minutes=minutes)
    events = pd.concat([per_season[s] for s in SEASONS], ignore_index=True)
    meta = {
        "possessions_version": version,
        "shooter_key": SHOOTER_KEY,
        "score_diff_mode": SCORE_DIFF_MODE,
        "rim_override_max_ft": rim,
        "coverage": coverage,
        "shooter_drops": drops,
        "relabelled": relabel,
        "minutes_join_pct": asof.attrs.get("minutes_join_pct"),
        "n_events": int(len(events)),
        "n_player_games": int(len(asof)),
        "position_known_pct": round(
            float((asof["position_group"] != "UNK").mean() * 100), 4),
        "prior_season_pct": {str(s): round(float(
            asof.loc[asof.season == s, "has_prior_season"].mean() * 100), 4)
            for s in SEASONS},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(ev_path, index=False)
    asof.to_parquet(asof_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return events, asof, meta


def load_round2() -> dict:
    if not ROUND2_RESULTS.exists():
        return {}
    return json.loads(ROUND2_RESULTS.read_text())


def write_results(out_dir: Path, report_path: Path, meta: dict, results: list[dict],
                  args) -> None:
    payload = {"round": "round3_score_diff_prestate", "meta": meta,
               "score_diff_mode": SCORE_DIFF_MODE, "results": results,
               "args": vars(args)}
    (out_dir / "results_v3.json").write_text(json.dumps(payload, indent=2, default=str),
                                             encoding="utf-8")
    params = {}
    for r in results:
        if r["fold"] != "F1":
            continue
        cls = r["event_class"]
        lgbm = r["arms"].get("lgbm", {})
        params[cls] = {"prior_kind": r.get("prior_kind"), "m": r.get("shrink_m"),
                       "lgbm_params": lgbm.get("params")}
    (out_dir / "usage_params_v3.json").write_text(json.dumps(params, indent=2, default=str),
                                                   encoding="utf-8")
    lines = [f"# usage round 3 (score_diff data fix) -- {pd.Timestamp.now()}", ""]
    for r in results:
        lines.append(f"## {r['event_class']} / {r['fold']}")
        for arm, sc in r["arms"].items():
            ll = sc.get("log_loss")
            lines.append(f"- {arm}: log_loss={ll}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2",
                    help="possessions build to read the rim override from")
    ap.add_argument("--out", default="data/processed/models/usage_v3")
    ap.add_argument("--report", default=None)
    ap.add_argument("--classes", nargs="*", default=list(U.EVENT_CLASSES))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--mc-draws", type=int, default=200)
    ap.add_argument("--seq-draws", type=int, default=100)
    ap.add_argument("--sim-draws", type=int, default=40)
    ap.add_argument("--alpha-draws", type=int, default=15)
    ap.add_argument("--alpha-max-tg", type=int, default=3000)
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--inner-split", default=V1.INNER_SPLIT_DATE)
    ap.add_argument("--skip-wf", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.mc_draws, args.seq_draws, args.sim_draws = 40, 30, 10
        args.alpha_draws, args.alpha_max_tg, args.boot_reps = 6, 800, 50

    out_dir = ROOT / args.out
    report = Path(args.report) if args.report else out_dir / "report_v3.md"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    log(f"shooter_key = {SHOOTER_KEY}, score_diff_mode = {SCORE_DIFF_MODE} "
        f"(round 2 used own_row)")
    events, asof, meta = build_tables(args.version, out_dir, args.rebuild, log)
    log(f"events {len(events):,} asof {len(asof):,} ({time.time() - t0:.1f}s)")

    results = []
    for cls in args.classes:
        design = U.build_usage_design(events, asof, cls)
        tr, te = U.fold_slices(design)
        f1 = V2.run_fold_with_segments(cls, tr, te, "F1", args, log)
        results.append(f1)
        if not args.skip_wf:
            wtr, wte = U.walkforward_slices(design)
            results.append(V2.run_fold_with_segments(
                cls, wtr, wte, "WF2025", args, log,
                lgbm_params=f1["arms"]["lgbm"]["params"]))
        del design
        write_results(out_dir, report, meta, results, args)   # checkpoint
        (out_dir / "train_log_v3.txt").write_text("\n".join(log_lines), encoding="utf-8")

    write_results(out_dir, report, meta, results, args)
    (out_dir / "train_log_v3.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nwrote {report} and {out_dir}/usage_params_v3.json "
        f"({time.time() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
