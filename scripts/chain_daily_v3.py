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
from cbb_sim.live import tips as TP  # noqa: E402

log = logging.getLogger("chain_daily_v3")
from cbb_sim.live.preseason import preseason_dir as _preseason_dir, preseason_rel as _preseason_rel  # noqa: E402,F401
SCHED_2027 = _preseason_dir() / "games_2027.parquet"
CROSSWALK = REPO / "data/reference/team_crosswalk_v2.parquet"
ENGINE_DIR = REPO / "data/processed/models/engine"
GRADE_RETRY_DAYS = 14


def sim_warnings(season: int) -> list[str]:
    """Non-blocking degradations (the game-level sim runs; the player layer is degraded)."""
    ros = _preseason_dir() / "roster_players_2027.parquet"
    espn = REPO / "data/raw/cbbd/rosters/roster_2027.parquet"       # pull_rosters_espn_v1 (CBBD 2027 rosters are empty)
    if season >= 2027 and (not ros.exists() or len(pd.read_parquet(ros)) == 0) and not espn.exists():
        return ["2027 rosters empty (CBBD /teams/roster). build_live reads no roster file: on an opening day it finds NO rotation prior for any team-game "
                "(0 of 222 on 2024-11-04, even though the prior season exists), so every team-game takes the anonymous league-mean rotation and no named "
                "candidates exist: the PLAYER layer is degraded on day 1 with or without rosters (game-level outputs do not depend on it). "
                "Pass --require-rosters to block instead."]
    return []


def sim_prereqs(season: int, fold: str, slate_date, ratings_dir, require_rosters: bool = False) -> list[str]:
    """What the live sim needs that does not exist yet (day-1 list; docs/ops/daily_chain_v3_2026-09-30.md section 6)."""
    miss = []
    if not (ENGINE_DIR / f"event_round2_s1_{fold}_{season}").exists():
        miss.append(f"adapter artifacts: {ENGINE_DIR.relative_to(REPO)}/event_round2_s1_{fold}_{season} (dated-refit set for {season}; only F2_2025 exists)")
    if not (ENGINE_DIR / f"names_{fold}_{season}_v2.json").exists():
        miss.append(f"names / rule constants template names_{fold}_{season}_v2.json (rules: bonus era, dead-ball share, and-one, foul accrual for {season})")
    if not ratings_dir:
        miss.append("own ratings as of the slate date (ratings stage blocked on data/overrides/ratings_day1_choices.json)")
    import os
    from cbb_sim.engine import foul_joint as FJ
    from cbb_sim.engine import foul_r9 as FR9
    arm = os.environ.get("ENGINE_FOUL_JOINT", FJ.DEFAULT)
    if arm in FR9.ARMS:          # served R9ao3: a season missing from its team-prior table is a hard stop, never a zero prior term
        have = FR9.table_seasons(arm)
        if have is not None and int(season) not in have:
            miss.append(FR9.missing_season_message([int(season)], arm, have))
    if require_rosters:
        miss += sim_warnings(season)
    return miss


def stage_sim(ctx, a, CD) -> dict:
    season = CD.current_season(ctx.slate_date)
    pass_name = getattr(a, "pass_name", None)
    tips = "table" if pass_name else "hoopr"
    if pass_name and not (REPO / f"data/processed/ingest/tip_times_{season}.parquet").exists():
        return {"_status": "blocked", "blocked_on": [f"tip-time table tip_times_{season}.parquet missing (run the tips stage / pull_tip_times_v1.py)"]}
    slate = SIM.load_slate(str(ctx.slate_date), season, "cbbd", str(SCHED_2027), str(CROSSWALK), tips)
    if pass_name:
        ok, late = TP.select_for_pass(slate, ctx.now, pass_name)
    else:
        ok, late = D.split_tipped(slate, ctx.now)
    census = {"slate_date": str(ctx.slate_date), "pass": pass_name, "clock": str(ctx.now), "slate_games_mapped": int(len(slate)),
              "slate_games_unmapped_non_d1": len(slate.attrs.get("unmapped", [])), "would_simulate": int(len(ok)),
              "would_refuse_already_tipped": int(len(late)),
              "tip_sources": slate["tip_source"].value_counts().to_dict() if len(slate) else {},
              "placeholder_tips": int(slate["tip_time_is_placeholder"].sum()) if len(slate) else 0}
    miss = sim_prereqs(season, a.fold, ctx.slate_date, ctx.state.get("ratings_dir"), getattr(a, "require_rosters", False))
    warn = sim_warnings(season)
    if warn:
        census["warnings"] = warn
    dry_sim = ctx.dry_run and getattr(a, "dry_run_sim", False)
    if (ctx.dry_run and not dry_sim) or miss:
        return {**census, "_status": "blocked", "blocked_on": miss or ["dry run"], "engine_run": False}
    import contextlib
    root = (REPO / "results/daily_dry") if dry_sim else ctx.root
    cm = V2.unsealed() if season >= 2027 else contextlib.nullcontext()   # the 2026 prior tables; the ratings stage already required seal_lift_approved
    with cm:
        r = SIM.run_sim_stage(str(ctx.slate_date), season, a.fold, a.seeds, 0, ctx.now, root, None, "cbbd", str(SCHED_2027),
                              str(CROSSWALK), tips, strict=False, pass_name=pass_name, ratings_dir=ctx.state.get("ratings_dir"))
    ctx.state["sim_run_id"] = D.default_run_id(a.seeds, 0) + ("_morning" if pass_name == "morning" else "")
    return {**census, **r}


def stage_rosters(ctx, a, CD) -> dict:
    """Season-S roster refresh (ESPN team rosters; CBBD is empty for 2027): weekly until 2026-11-02, daily after. Hard error if the pull fails its checks."""
    import pull_rosters_espn_v1 as PR
    season = CD.current_season(ctx.slate_date)
    if season < 2027:
        return {"_status": "skipped", "why": "past-season replay"}
    return PR.run_roster_stage(ctx.today, season, ctx.dry_run)


def stage_injuries_parse(ctx, a) -> dict:
    """ESPN league-wide injuries -> data/processed/injuries/player_out_{today}.csv (override format). Consumed by chain_daily_v2.load_availability -> build_live(availability=...) on live runs."""
    import pull_injuries_player_out_v1 as PI
    return PI.run_injury_parse_stage(ctx.today, ctx.dry_run, now=str(ctx.now))


def stage_tips(ctx, a, CD, cctx) -> dict:
    """Tip-time refresh (hoopR schedule + CBBD /games, 1 CBBD call); records source and tip_time_is_placeholder per game. Dry run writes nothing."""
    import pull_tip_times_v1 as PT
    season = CD.current_season(ctx.slate_date)
    cg = PT.fetch_cbbd_games(cctx.session, cctx.tracker, season, ctx.slate_date, ctx.slate_date)
    hs = PT.fetch_hoopr_schedule(season)
    _, summ = PT.refresh_tip_times(season, ctx.slate_date, ctx.slate_date, cg, hs, pd.Timestamp.now("UTC"), write=not ctx.dry_run)
    return summ


def fallback_report_line(slate_date) -> str | None:
    """'N teams on fallback roster: ...' from the slate's live build diag (None when no live build exists for it)."""
    import glob
    fs = sorted(glob.glob(str(REPO / f"data/processed/models/engine_live/build_diag_LIVE_F2_*_{slate_date}.json")))
    if not fs:
        return None
    return json.loads(Path(fs[-1]).read_text(encoding="utf-8")).get("d1p_fallback_line")


def _run_script(args: list[str]) -> dict:
    import subprocess
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=600, cwd=str(REPO))
    return {"rc": r.returncode, "out": (r.stdout or "")[-300:]}


def stage_lines_snapshot(ctx, a) -> dict:
    """Own lines snapshot (ESPN current line + CBBD /lines, captured_at stamped) then the open/close builder. Never blocks the chain."""
    snap = str(REPO / "scripts" / "pull_espn_odds_snapshot_v1.py")
    build = str(REPO / "scripts" / "build_lines_open_close_v1.py")
    extra = ["--dry-run"] if a.dry_run else []
    out = {"snapshot": _run_script([snap, "--date", str(ctx.slate_date), *extra])}
    if not a.dry_run:
        out["open_close"] = _run_script([build])
    return out


def stage_lines_probe(ctx, a) -> dict:
    """Daily source probe from Oct 20 (dry-run safe: --dry-run prints the plan, no network)."""
    if ctx.today < date(ctx.today.year, 10, 20) and ctx.today.month >= 5:
        return {"_status": "skipped", "why": "probe starts Oct 20"}
    return _run_script([str(REPO / "scripts" / "probe_lines_sources_v1.py"), *(["--dry-run"] if a.dry_run else [])])


def stage_publish(ctx, a, CD) -> dict:
    run_id = ctx.state.get("sim_run_id") or D.default_run_id(a.seeds, 0)
    if ctx.dry_run:
        return {"_status": "blocked", "why": "dry run; needs the sim output and a lines fetch (CBBD /lines, 1 call)"}
    r = PUB.run_publish_stage(str(ctx.slate_date), run_id, ctx.now, ctx.root, "live", session=ctx.session, tracker=ctx.tracker)
    line = fallback_report_line(ctx.slate_date)
    return {**r, "fallback_roster_line": line} if line and isinstance(r, dict) else r


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
        line = fallback_report_line(d)
        if line:
            out[d]["fallback_roster_line"] = line
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
    rp = getattr(a, "replay_pass", None)            # two-pass replay: evening 20:00 ET D-1 or morning 09:00 ET D (tips.default_clock)
    now = D.utc(a.now) if a.now else (TP.default_clock(d, rp) if rp else pd.Timestamp(f"{d}T14:00:00Z"))
    gnow = pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1, hours=14)
    res = []
    run_id = D.default_run_id(a.seeds, 0) + ("_morning" if rp == "morning" else "")
    V2.run_stage("sim", lambda: SIM.run_sim_stage(d, S, "F2", a.seeds, 0, now, root, None, "universe", replay=True, pass_name=rp), res)
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
    ap.add_argument("--pass", dest="pass_name", choices=("evening", "morning"), default="evening",
                    help="evening (DEFAULT): run the evening before the slate, every game tipping after the clock. morning: same-day re-publish, "
                         "only games with a REAL tip time still in the future. Dry run without --now uses the pass's documented clock "
                         "(evening 20:00 ET the day before, morning 09:00 ET the slate date).")
    ap.add_argument("--replay-pass", dest="replay_pass", choices=("evening", "morning"), default=None,
                    help="with --replay-season: replay the sim / publish stages as the evening (20:00 ET the day before) or morning (09:00 ET) pass")
    ap.add_argument("--require-rosters", action="store_true", help="block the sim on empty 2027 rosters instead of warning")
    ap.add_argument("--dry-run-sim", action="store_true", help="in a dry run, still run the (tiny) sim into results/daily_dry when nothing blocks")
    a = ap.parse_args(argv)
    if a.replay_season:
        if not a.slate_date:
            raise SystemExit("--slate-date required for a replay")
        if a.root == str(D.DAILY_ROOT):
            raise SystemExit("--replay-season requires --root <dir other than the live root> (a replay writes sim / publish / grade ledger rows; "
                             "into the live root it would pollute live grading). e.g. --root results/daily_replay_f2")
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
        later = [x for x in dd if (x > today if a.pass_name == "evening" else x >= today)]   # evening pass = the NEXT game day
        slate_date = min(later)
    now = D.utc(a.now) if a.now else (TP.default_clock(slate_date, a.pass_name) if a.dry_run else pd.Timestamp.now("UTC"))
    v2ctx = V2.Ctx(today, today - timedelta(days=1), slate_date, a.dry_run)
    cctx.state = v2ctx.state
    ctx = Ctx(today, slate_date, now, a.dry_run, a.root, cctx.session, cctx.tracker)
    ctx.state = v2ctx.state
    results: list = []
    day1 = CD.current_season(slate_date) >= 2027                       # the season switch: replay / past seasons never take the 2027 branches
    if day1:
        import chain_day1_2027_v1 as DAY1
    stages = [
        ("schedule", lambda: CD.step_schedule(cctx)),
        ("tips", lambda: stage_tips(ctx, a, CD, cctx)),
        ("lines", lambda: CD.step_lines(cctx)),
        ("lines_snapshot", lambda: stage_lines_snapshot(ctx, a)),
        ("lines_probe", lambda: stage_lines_probe(ctx, a)),
        ("ingest", lambda: V2.stage_ingest(v2ctx, CD, a)),
        ("ratings", (lambda: DAY1.stage_ratings(ctx, CD, a)) if day1 else (lambda: V2.stage_ratings(v2ctx, CD, a))),
        ("kenpom", (lambda: CD.step_kenpom(cctx)) if a.with_kenpom else (lambda: {"_status": "skipped", "why": "audit gap 10: PM decision"})),
        ("rosters", lambda: stage_rosters(ctx, a, CD)),
        ("injuries", lambda: CD.step_injuries(cctx)),
        ("injuries_parse", lambda: stage_injuries_parse(ctx, a)),
        ("overrides", lambda: CD.step_overrides(cctx)),
        ("inputs", (lambda: DAY1.stage_inputs(ctx, CD, a, SIM, pass_name=a.pass_name)) if day1 else (lambda: V2.stage_inputs(v2ctx, CD, a))),
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
