#!/usr/bin/env python
"""
chain_daily_v3.py -- the daily chain, version 3 (lane F, 2026-09-30). `chain_daily_v2.py` is untouched; its stages 1-8 (schedule, lines,
ingest, ratings, kenpom, injuries, overrides, inputs) are REUSED by import. The four stages v2 left as stubs are built here:

  9  sim       run_daily_sim_v1.run_sim_stage      refuse tipped games, live inputs, served engine, created_at < tipoff asserted
  10 publish   run_daily_publish_v1.run_publish_stage   distribution summaries + market probabilities vs the lines at run time
  11 grade     grade_daily_v1.run_grade_stage      yesterday (and pending older slates) vs VERIFIED finals, running ledger
  12 bias_clv  diag_daily_bias_clv_v1.run_monitor_stage   rolling bias with SEs, per-stat bias, CLV of leans; reports and alarms only

Order inside a live morning run: stages 1-8 first (ingestion of yesterday's finals feeds the as-of features), then GRADE and BIAS_CLV
for yesterday's slate (they need only yesterday's finals and the fresh close snapshot), then SIM and PUBLISH for today's slate. The
numbering above is the readiness doc's; the execution order is: ... inputs, grade, bias_clv, sim, publish.

`--dry-run` touches nothing under results/ (except the chain's own json) and runs no engine: sim reports the slate census (games, tipped
games it would refuse, unmapped non-D-I games) and the prerequisites it is blocked on; publish / grade / bias_clv report skipped or blocked.

REPLAY (proof on a past season, clock faked, 2025-26 refused):
    .venv/Scripts/python.exe scripts/chain_daily_v3.py --replay-season 2025 --slate-date 2025-02-11 --seeds 16
runs sim (clock = 14:00Z of the slate date), publish (replay lines: spread / total from the open columns), then grade (clock = 14:00Z next
day, replay truth = the eval harness's verified finals) and the monitor (replay close).

    .venv/Scripts/python.exe scripts/chain_daily_v3.py --dry-run [--date 2026-09-30] [--slate-date 2026-11-02] [--now ISO]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import chain_daily_v2 as V2  # noqa: E402
import grade_daily_v1 as GR  # noqa: E402
import run_daily_publish_v1 as PUB  # noqa: E402
import run_daily_sim_v1 as SIM  # noqa: E402
import diag_daily_bias_clv_v1 as MON  # noqa: E402
from cbb_sim.live import daily as D  # noqa: E402

log = logging.getLogger("chain_daily_v3")
SCHED_2027 = REPO / "data/raw/preseason/2027_v2_20260930/games_2027.parquet"
CROSSWALK = REPO / "data/reference/team_crosswalk_v2.parquet"
ENGINE_DIR = REPO / "data/processed/models/engine"
GRADE_RETRY_DAYS = 14


def sim_prereqs(season: int, fold: str, slate_date, ratings_dir) -> list[str]:
    """What the live sim needs that does not exist yet (day-1 list; docs/ops/daily_chain_v3_2026-09-30.md section 6)."""
    miss = []
    if not (ENGINE_DIR / f"event_round2_s1_{fold}_{season}").exists():
        miss.append(f"adapter artifacts: {ENGINE_DIR.relative_to(REPO)}/event_round2_s1_{fold}_{season} (dated-refit set for {season}; only F2_2025 exists)")
    if not (ENGINE_DIR / f"names_{fold}_{season}_v2.json").exists():
        miss.append(f"names / rule constants template names_{fold}_{season}_v2.json (rules: bonus era, dead-ball share, and-one, foul accrual for {season})")
    if not ratings_dir:
        miss.append("own ratings as of the slate date (ratings stage blocked on data/overrides/ratings_day1_choices.json)")
    ros = REPO / "data/raw/preseason/2027_v2_20260930/roster_players_2027.parquet"
    if season >= 2027 and (not ros.exists() or len(pd.read_parquet(ros)) == 0):
        miss.append("2027 rosters (CBBD /teams/roster empty)")
    return miss


def stage_sim(ctx, a, CD) -> dict:
    season = CD.current_season(ctx.slate_date)
    slate = SIM.load_slate(str(ctx.slate_date), season, "cbbd", str(SCHED_2027), str(CROSSWALK), "hoopr")
    ok, late = D.split_tipped(slate, ctx.now)
    census = {"slate_date": str(ctx.slate_date), "clock": str(ctx.now), "slate_games_mapped": int(len(slate)),
              "slate_games_unmapped_non_d1": len(slate.attrs.get("unmapped", [])), "would_simulate": int(len(ok)),
              "would_refuse_already_tipped": int(len(late)),
              "tip_sources": slate["tip_source"].value_counts().to_dict() if len(slate) else {}}
    miss = sim_prereqs(season, a.fold, ctx.slate_date, ctx.state.get("ratings_dir"))
    if ctx.dry_run or miss:
        return {**census, "_status": "blocked", "blocked_on": miss or ["dry run"], "engine_run": False}
    r = SIM.run_sim_stage(str(ctx.slate_date), season, a.fold, a.seeds, 0, ctx.now, ctx.root, None, "cbbd", str(SCHED_2027),
                          str(CROSSWALK), "hoopr", strict=False)
    ctx.state["sim_run_id"] = D.default_run_id(a.seeds, 0)
    return {**census, **r}


def stage_publish(ctx, a, CD) -> dict:
    run_id = ctx.state.get("sim_run_id") or D.default_run_id(a.seeds, 0)
    if ctx.dry_run:
        return {"_status": "blocked", "why": "dry run; needs the sim output and a lines fetch (CBBD /lines, 1 call)"}
    r = PUB.run_publish_stage(str(ctx.slate_date), run_id, ctx.now, ctx.root, "live", session=ctx.session, tracker=ctx.tracker)
    return r


def stage_grade(ctx, a, CD) -> dict:
    season = CD.current_season(ctx.yesterday)
    pubdir = ctx.root / "publish"
    dates = sorted(p.name for p in pubdir.glob("*") if p.is_dir() and p.name < str(ctx.today)
                   and p.name >= str(ctx.today - timedelta(days=GRADE_RETRY_DAYS))) if pubdir.exists() else []
    if not dates:
        return {"_status": "skipped", "why": "no published slate in the retry window (season not running)"}
    out = {}
    for d in dates:
        r = GR.run_grade_stage(d, season, ctx.now, ctx.root, "ingest")
        out[d] = {k: r.get(k) for k in ("_status", "graded", "pending", "ledger_rows", "margin_mae", "why")}
    return {"slates": out}


def stage_bias(ctx, a, CD) -> dict:
    season = CD.current_season(ctx.yesterday)
    r = MON.run_monitor_stage(season, ctx.now, ctx.root, "live")
    return r


class Ctx:
    def __init__(self, today, slate_date, now, dry_run, root, session=None, tracker=None):
        self.today, self.yesterday, self.slate_date, self.now = today, today - timedelta(days=1), slate_date, now
        self.dry_run, self.root, self.session, self.tracker = dry_run, Path(root), session, tracker
        self.state: dict = {}


def run_replay(a) -> int:
    """Past-season replay of stages 9-12 with the clock faked (2025-26 and later refused)."""
    S = int(a.replay_season)
    if S >= 2026:
        raise SystemExit("replay refuses season >= 2026 (2025-26 is SEALED)")
    root = Path(a.root)
    d = a.slate_date
    now = pd.Timestamp(f"{d}T14:00:00Z") if not a.now else D.utc(a.now)
    gnow = pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1, hours=14)
    res = []
    run_id = D.default_run_id(a.seeds, 0)
    V2.run_stage("sim", lambda: SIM.run_sim_stage(d, S, "F2", a.seeds, 0, now, root, None, "universe", replay=True), res)
    V2.run_stage("publish", lambda: PUB.run_publish_stage(d, run_id, now, root, "replay_open", S), res)
    V2.run_stage("grade", lambda: GR.run_grade_stage(d, S, gnow, root, "truth"), res)
    V2.run_stage("bias_clv", lambda: MON.run_monitor_stage(S, gnow, root, "replay_close"), res)
    for r in res:
        print(f"  [{r.status.upper():9}] {r.name:10} ({r.seconds:5.1f}s) {r.error or json.dumps(r.detail, default=str)[:300]}")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date", default=None)
    ap.add_argument("--slate-date", default=None)
    ap.add_argument("--now", default=None, help="injected clock (ISO UTC) for sim / publish / grade / monitor")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--root", default=str(D.DAILY_ROOT))
    ap.add_argument("--replay-season", type=int, default=None)
    ap.add_argument("--with-kenpom", action="store_true")
    ap.add_argument("--no-probe-hoopr", dest="probe_hoopr", action="store_false")
    ap.add_argument("--max-calls", type=int, default=60)
    a = ap.parse_args(argv)
    if a.replay_season:
        if not a.slate_date:
            raise SystemExit("--slate-date required for a replay")
        if a.seeds == 200:
            a.seeds = 16
        return run_replay(a)
    today = date.fromisoformat(a.date) if a.date else date.today()
    CD = V2._load("chain_daily")
    cctx = CD.build_context(today, a.dry_run, max_calls=a.max_calls)
    if a.slate_date:
        slate_date = date.fromisoformat(a.slate_date)
    else:
        g = pd.read_parquet(SCHED_2027, columns=["startDate"])
        dd = pd.to_datetime(g["startDate"], utc=True).dt.tz_convert("America/New_York").dt.date
        later = [x for x in dd if x >= today]
        slate_date = min(later)
    now = D.utc(a.now) if a.now else pd.Timestamp.now("UTC")
    v2ctx = V2.Ctx(today, today - timedelta(days=1), slate_date, a.dry_run)
    cctx.state = v2ctx.state
    ctx = Ctx(today, slate_date, now, a.dry_run, a.root, cctx.session, cctx.tracker)
    ctx.state = v2ctx.state
    results: list = []
    stages = [
        ("schedule", lambda: CD.step_schedule(cctx)),
        ("lines", lambda: CD.step_lines(cctx)),
        ("ingest", lambda: V2.stage_ingest(v2ctx, CD, a)),
        ("ratings", lambda: V2.stage_ratings(v2ctx, CD, a)),
        ("kenpom", (lambda: CD.step_kenpom(cctx)) if a.with_kenpom else (lambda: {"_status": "skipped", "why": "audit gap 10: PM decision"})),
        ("injuries", lambda: CD.step_injuries(cctx)),
        ("overrides", lambda: CD.step_overrides(cctx)),
        ("inputs", lambda: V2.stage_inputs(v2ctx, CD, a)),
        ("grade", lambda: stage_grade(ctx, a, CD)),
        ("bias_clv", lambda: stage_bias(ctx, a, CD)),
        ("sim", lambda: stage_sim(ctx, a, CD)),
        ("publish", lambda: stage_publish(ctx, a, CD)),
    ]
    for name, fn in stages:
        V2.run_stage(name, fn, results)
    calls = cctx.tracker.count
    print(f"\n=== chain_daily_v3 {'DRY RUN' if a.dry_run else 'LIVE RUN'} date={today} slate={slate_date} clock={now} ===")
    for r in results:
        print(f"  [{r.status.upper():9}] {r.name:10} ({r.seconds:5.1f}s) {r.error or json.dumps(r.detail, default=str)[:300]}")
    print(f"--- CBBD calls (chain steps): {calls} ---")
    out = REPO / "results" / f"chain_daily_v3_{'dry' if a.dry_run else 'live'}_{today}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stages": [r.__dict__ for r in results], "cbbd_calls_chain": calls, "clock": str(now)}, indent=2, default=str),
                   encoding="utf-8")
    if not a.dry_run:
        d = REPO / "data/processed/ingest/chain_v3"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{today}.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
