#!/usr/bin/env python
"""
chain_daily_v2.py -- the daily chain, version 2 (Lane K follow-up, 2026-09-30). `chain_daily.py` is untouched and its
schedule / lines / kenpom / injuries / overrides steps are REUSED by import; the new stages are ingest, ratings, inputs and
the four that do not exist yet.

STAGE ORDER (docs/ops/readiness_2026-27_2026-09-30.md section 4a, with post-game ingestion first because every later
as-of feature reads what it writes):

  1 schedule    chain_daily.step_schedule        CBBD + hoopR schedule for today/yesterday
  2 lines       chain_daily.step_lines           open/close snapshots
  3 ingest      pull_daily_ingest_v1             yesterday's finals verified across hoopR + CBBD, raw + derived tables, pending retry
  4 ratings     build_own_ratings_asof_v1        own ratings as of the slate date; BLOCKED until the day-1 choices file exists
  5 kenpom      chain_daily.step_kenpom          OFF by default (audit gap 10: scraping not licensed here; PM decision) -> skipped
  6 injuries    chain_daily.step_injuries
  7 overrides   chain_daily.step_overrides
  8 inputs      build_engine_inputs_live         slate inputs with the created_at < tipoff guard; dry run = day-1 breakage census
  9 sim         NOT BUILT (audit gap 6)
 10 publish     NOT BUILT
 11 grade       NOT BUILT (graders exist for backtests only)
 12 bias_clv    NOT BUILT (bias monitor, CLV ledger)

Every stage is exception-isolated (a failing stage never stops the next) and idempotent: ingest skips complete games and
writes nothing when rows are unchanged, ratings/inputs write to date-keyed output files that are replaced, the reused
steps are chain_daily's own idempotent upserts. Every stage is logged (logging + `<root>/data/processed/ingest/chain_v2/<date>.json`
on a live run). Statuses: ok | blocked (a needed decision is missing; lists it) | skipped | not_built | error.

SEAL. CBB_UNSEAL=1 is set only inside `unsealed()`, used only when the ratings or inputs stage has the day-1 choices file
AND actually needs the 2026 prior chain. The dry run never reaches it and prints nothing from 2026 outcomes.

    .venv/Scripts/python.exe scripts/chain_daily_v2.py --dry-run [--date 2026-09-30] [--slate-date 2026-11-02]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

log = logging.getLogger("chain_daily_v2")
CHOICES_PATH = REPO / "data/overrides/ratings_day1_choices.json"
from cbb_sim.live.preseason import preseason_dir as _preseason_dir, preseason_rel as _preseason_rel  # noqa: E402,F401
PRESEASON = _preseason_dir()

#: decisions the day-1 ratings / inputs build needs; value = the implemented options
DAY1_CHOICES = {
    "seal_lift_approved": [True],
    "teams_source": ["cbbd", "schedule", "tg"],
    "prior_weight_policy": ["manifest_uniform"],          # w = 0.8 eff and tempo for every team (the stored default)
    "new_team_prior": ["league_mean"],
    "fixed_term_prior": ["previous_final"],
    "early_season_d1_rule": ["provisional_union"],
}

#: lane G's day-1 breakage list (docs/ops/live_slate_path_2026-09-30.md 5.5): (family, feature prefix, statuses that count as reproduced)
LANE_G_DAY1 = [
    ("ratings", "own_ratings_2027", {"MISSING", "BREAKS"}),
    ("schedule", "tipoff_utc", {"partial", "DEGENERATE"}),
    ("possession_outcome", "off/opp_def", {"DEGENERATE"}),
    ("rebound", "off_oreb_c", {"DEGENERATE"}),
    ("fg_make", "off_make_c", {"DEGENERATE"}),
    ("free_throw", "shooter_ft_asof", {"DEGENERATE"}),
    ("usage", "usage_rate", {"DEGENERATE"}),
    ("rotation", "roster / rot_share", {"DEGENERATE"}),
    ("roster", "CBBD /teams/roster", {"MISSING"}),
    ("ids", "player_crosswalk", {"MISSING"}),
    ("adapters", "dated-refit", {"MISSING"}),
    ("rules", "bonus era", {"BREAKS"}),
    ("rules", "rule constants", {"STALE"}),
    ("lines", "2027 line rows", {"MISSING"}),
]


@contextlib.contextmanager
def unsealed():
    old = os.environ.get("CBB_UNSEAL")
    os.environ["CBB_UNSEAL"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CBB_UNSEAL", None)
        else:
            os.environ["CBB_UNSEAL"] = old


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class StageResult:
    name: str
    status: str
    detail: dict = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0


def run_stage(name, fn, results):
    t0 = time.monotonic()
    try:
        out = fn() or {}
        st = out.pop("_status", "ok")
        res = StageResult(name, st, out, None, time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001 -- isolation is the point
        res = StageResult(name, "error", {"traceback": traceback.format_exc(limit=5)}, f"{type(exc).__name__}: {exc}",
                          time.monotonic() - t0)
    results.append(res)
    log.info("stage %-10s %-9s %.1fs %s", name, res.status, res.seconds, res.error or json.dumps(res.detail, default=str)[:200])
    return res


# ------------------------------------------------------------------------------------------------ stages
def stage_ingest(ctx, CD, a) -> dict:
    import pull_daily_ingest_v1 as PI
    import requests
    season = CD.current_season(ctx.today)
    argv = ["--dates", str(ctx.yesterday), "--season", str(season)] + (["--dry-run"] if ctx.dry_run else [])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = PI.main(argv)
    last = [ln for ln in buf.getvalue().splitlines() if ln.startswith("{")]
    summ = json.loads(last[-1]) if last else {}
    probes = {}
    if a.probe_hoopr:                       # which season files hoopR publishes today (HEAD, live)
        for ds, (dirname, stem) in PI.HOOPR_DATASETS.items():
            try:
                r = requests.head(PI.HOOPR_URL.format(dirname=dirname, stem=stem, season=season), timeout=30, allow_redirects=True)
                probes[ds] = r.status_code
            except Exception as exc:  # noqa: BLE001
                probes[ds] = f"error {type(exc).__name__}"
    pend = REPO / f"data/processed/ingest/pending_{season}.parquet"
    n_pending = int(len(pd.read_parquet(pend))) if pend.exists() else 0
    sched = summ.get("hoopr", {}).get(f"schedules_{season}", {})
    return {"rc": rc, "season": season, "date": str(ctx.yesterday), "verified_finals": summ.get("complete", 0),
            "pending_list": n_pending, "schedule_rows_for_date": summ.get("n_schedule_rows"),
            "hoopr_schedule_download": {k: sched.get(k) for k in ("source", "bytes", "etag", "rows", "fetched_at")},
            "hoopr_dataset_http_status": probes, "ingest_cbbd_calls": summ.get("cbbd_calls", 0),
            "_status": "ok" if rc == 0 else "error"}


def _missing_choices() -> tuple[dict, list]:
    have = json.loads(CHOICES_PATH.read_text(encoding="utf-8")) if CHOICES_PATH.exists() else {}
    miss = [k for k in DAY1_CHOICES if have.get(k) not in DAY1_CHOICES[k]]
    return have, miss


def stage_ratings(ctx, CD, a) -> dict:
    have, miss = _missing_choices()
    if miss:
        return {"_status": "blocked", "needs_choices_file": str(CHOICES_PATH.relative_to(REPO)),
                "missing_or_invalid": {k: {"implemented_options": DAY1_CHOICES[k], "given": have.get(k)} for k in miss},
                "note": "2026 prior chain NOT read (no CBB_UNSEAL); see docs/ops/own_ratings_daily_2026-09-30.md section 4"}
    import build_own_ratings_asof_v1 as E
    season = CD.current_season(ctx.today)
    src = {"cbbd": f"cbbd:{PRESEASON / 'games_2027.parquet'}|{REPO / 'data/reference/team_crosswalk_v2.parquet'}",
           "schedule": "schedule", "tg": "tg"}[have["teams_source"]]
    with unsealed():                                   # the ONLY place the 2026 chain is read
        out, prov = E.asof_ratings(season, str(ctx.slate_date), REPO, src, 2022, {})
    if ctx.dry_run:
        return {"rows": int(len(out)), "dry_run": True, "provenance": prov}
    d = REPO / f"data/processed/ratings_asof/{ctx.slate_date}"
    d.mkdir(parents=True, exist_ok=True)
    out["created_at"] = pd.Timestamp.now("UTC")
    out.to_parquet(d / f"own_ratings_{season}.parquet", index=False)
    (d / "provenance.json").write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    ctx.state["ratings_dir"] = str(d)
    return {"rows": int(len(out)), "out": str(d)}


def stage_inputs(ctx, CD, a) -> dict:
    """Guard first (created_at < tipoff on the slate), then: dry run = the day-1 breakage census; live = build_live."""
    import build_engine_inputs_live as BL
    from cbb_sim.live import guards as G
    slate = BL.load_slate_from_cbbd(str(PRESEASON / "games_2027.parquet"), str(ctx.slate_date),
                                    str(REPO / "data/reference/team_crosswalk_v2.parquet"))
    if not len(slate):
        return {"_status": "blocked", "reason": f"no slate rows for {ctx.slate_date}"}
    created = pd.Timestamp.now("UTC")
    frame = slate[["game_id", "tipoff_utc"]].assign(created_at=created)
    G.assert_created_before_tipoff(frame)                     # raises LeakGuardError -> stage error
    out = {"slate_date": str(ctx.slate_date), "slate_games_mapped": int(len(slate)),
           "slate_games_unmapped": len(slate.attrs.get("unmapped", [])), "guard": "created_at < tipoff: PASS",
           "created_at": str(created), "first_tip": str(pd.to_datetime(slate["tipoff_utc"], utc=True).min())}
    if ctx.dry_run or not ctx.state.get("ratings_dir"):
        r = subprocess.run([sys.executable, str(REPO / "scripts/diag_live_day1_v1.py"), "--date", str(ctx.slate_date)],
                           capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        rows = json.loads((REPO / "results/live_day1_2026-09-30/day1.json").read_text(encoding="utf-8"))["rows"]
        table = [(x.get("family"), x.get("feature"), x.get("status")) for x in rows]
        repro, missing = [], []
        for fam, prefix, stat in LANE_G_DAY1:
            hit = [t for t in table if t[0] == fam and str(t[1]).startswith(prefix)]
            (repro if hit and any(h[2] in stat for h in hit) else missing).append(f"{fam}:{prefix}")
        out.update(day1_rows=len(rows), lane_g_reproduced=len(repro), lane_g_not_reproduced=missing,
                   day1_status_counts=pd.Series([t[2] for t in table]).value_counts().to_dict(),
                   diag_rc=r.returncode, _status="blocked")
        out["blocked_on"] = "own ratings + 2027 adapters + rosters + rules (day-1 list); nothing built"
        return out
    return build_live_inputs(ctx, BL, slate, out)


# Sealed-prior hard stops for the 2027 live build. Each entry: (label, path relative to REPO, what builds it). The 2025-26 season is
# SEALED: nothing here reads it; these tables are produced AFTER the user lifts the seal (runbook docs/ops/readiness_gaps_2026-10-05.md
# section 2, steps 5-6) and the chain refuses to continue without them. No zero prior, no carry-forward of 2025, no silent fallback.
SEALED_PRIORS_2027 = [
    ("shot-block anchor prior (2025-26 end-of-season block rate by miss type, K2_Ocell)",
     "data/processed/models/engine/shot_block_prior_2027_v1.parquet",
     "versioned sibling of build_shot_block_lut_live_v1.py (runbook step 6)"),
    ("R9ao3 team-prior table with season 2027 rows (needs 2025-26 and-one design)",
     None, "ao_team_prior_v2.parquet via train_foul_r9_v1.team_prior_table (runbook step 5)"),
]


def missing_sealed_priors(season: int, repo: Path = REPO, r9_seasons=None) -> list[str]:
    """Names of the missing seal-gated prior tables for `season` (empty = all present). Only season 2027 has the list; the R9ao3 check
    asks the served table which seasons it holds (`r9_seasons` injectable for tests)."""
    if int(season) != 2027:
        return []
    miss = []
    for label, rel, how in SEALED_PRIORS_2027:
        if rel is not None:
            if not (repo / rel).exists():
                miss.append(f"{label}: {rel} missing; {how}")
            continue
        have = r9_seasons
        if have is None:
            from cbb_sim.engine import foul_r9 as FR9
            have = FR9.table_seasons("R9ao3")
        if have is not None and int(season) not in have:
            miss.append(f"{label}: table has seasons {sorted(have)}; {how}")
    return miss


def load_availability(ctx):
    """The injuries_parse output for today plus manual availability.csv rows for the slate date (`player_out_for`), or None when empty (then
    build_live runs exactly as before)."""
    import pull_injuries_player_out_v1 as PI
    days = {getattr(ctx, "today", None) or ctx.slate_date, ctx.slate_date}
    parts = [PI.player_out_for(pd.Timestamp(d).date()) for d in sorted(days, key=str)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df.drop_duplicates("athlete_id") if len(df) else None


def day1_player_prior_missing(season: int, repo: Path = REPO) -> list[str]:
    """Hard-stop list for the A3 day-1 player-prior seed (live 2027 only): its two source tables must exist."""
    miss = []
    ros = repo / f"data/raw/cbbd/rosters/roster_{season}.parquet"
    if not ros.exists():
        miss.append(f"A3 day-1 player priors: roster table {ros.relative_to(repo)} missing (stage rosters, pull_rosters_espn_v1.py)")
    pos = repo / f"data/processed/possessions_v2/possessions_{season - 1}.parquet"
    if not pos.exists():
        miss.append(f"A3 day-1 player priors: season {season - 1} on-floor possessions table {pos.relative_to(repo)} missing")
    return miss


def day1_player_prior_seed(season: int, repo: Path = REPO):
    """(module, seed_fn): the A3 `seed_fn` for `build_live(..., seed_fn=...)` (`build_engine_inputs_day1prior_v1.make_seed_fn('A3')`, called, not edited)."""
    import build_engine_inputs_day1prior_v1 as D1P
    return D1P, D1P.make_seed_fn("A3", roster_path=str(repo / f"data/raw/cbbd/rosters/roster_{season}.parquet"))


def build_live_inputs(ctx, BL, slate, out: dict) -> dict:
    """Live 2027 branch: hard-stop on any missing sealed prior table, else build the slate's EngineInputs (as of the chain clock) and
    save to data/processed/models/engine_live/. Nothing is simulated here."""
    season = int(slate["season"].iloc[0])
    miss = missing_sealed_priors(season) + day1_player_prior_missing(season)
    if not ctx.state.get("ratings_dir"):
        miss.insert(0, "own ratings as of the slate date (ratings stage blocked)")
    if not (REPO / f"data/processed/models/engine/names_F2_{season}_v2.json").exists():
        miss.insert(0, f"rule-constants template names_F2_{season}_v2.json")
    if miss:
        return {**out, "_status": "blocked", "blocked_on": miss, "built": False}
    import numpy as np
    avail = load_availability(ctx)
    D1P, seed_fn = day1_player_prior_seed(season)             # live 2027 only; historical build_live runs keep seed_fn=None
    with unsealed():                                          # prior-season carry and the A3 S-1 minutes read the 2026 tables
        inp, diag = BL.build_live(slate, getattr(ctx, "now", None) or pd.Timestamp.now("UTC"), season, "F2", created_at=pd.Timestamp.now("UTC"),
                                  ratings_dir=ctx.state["ratings_dir"], availability=avail, seed_fn=seed_fn)
    D1P.post(inp, seed_fn)
    tag = f"LIVE_F2_{season}_{ctx.slate_date}"
    out_dir = REPO / "data/processed/models/engine_live"
    inp.save(str(out_dir), tag)
    (out_dir / f"build_diag_{tag}.json").write_text(json.dumps(diag, indent=2, default=str), encoding="utf-8")
    return {**out, "_status": "ok", "built": True, "tag": tag, "n_games": int(len(slate)),
            "availability": diag.get("availability", "none (no player-out rows)")}


def stage_reuse(CD, fn_name):
    return lambda ctx: getattr(CD, fn_name)(ctx)


def not_built(why):
    return lambda: {"_status": "not_built", "why": why}


@dataclass
class Ctx:
    today: date
    yesterday: date
    slate_date: date
    dry_run: bool
    state: dict = field(default_factory=dict)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date", default=None)
    ap.add_argument("--slate-date", default=None, help="slate the ratings / inputs stages target (default: first 2027 game date after the run date)")
    ap.add_argument("--with-kenpom", action="store_true")
    ap.add_argument("--no-probe-hoopr", dest="probe_hoopr", action="store_false")
    ap.add_argument("--max-calls", type=int, default=60)
    a = ap.parse_args(argv)
    today = date.fromisoformat(a.date) if a.date else date.today()
    CD = _load("chain_daily")
    cctx = CD.build_context(today, a.dry_run, max_calls=a.max_calls)
    if a.slate_date:
        slate_date = date.fromisoformat(a.slate_date)
    else:
        g = pd.read_parquet(PRESEASON / "games_2027.parquet", columns=["startDate"])
        d = pd.to_datetime(g["startDate"], utc=True).dt.tz_convert("America/New_York").dt.date
        slate_date = min(x for x in d if x > today)
    ctx = Ctx(today, today - timedelta(days=1), slate_date, a.dry_run)
    cctx.state = ctx.state
    results: list[StageResult] = []
    stages = [
        ("schedule", lambda: CD.step_schedule(cctx)),
        ("lines", lambda: CD.step_lines(cctx)),
        ("ingest", lambda: stage_ingest(ctx, CD, a)),
        ("ratings", lambda: stage_ratings(ctx, CD, a)),
        ("kenpom", (lambda: CD.step_kenpom(cctx)) if a.with_kenpom else (lambda: {"_status": "skipped", "why": "audit gap 10: PM decision"})),
        ("injuries", lambda: CD.step_injuries(cctx)),
        ("overrides", lambda: CD.step_overrides(cctx)),
        ("inputs", lambda: stage_inputs(ctx, CD, a)),
        ("sim", not_built("audit gap 6: run_engine.py is backtest only")),
        ("publish", not_built("audit gap 6")),
        ("grade", not_built("graders take backtest parquet, not daily output")),
        ("bias_clv", not_built("bias monitor and CLV ledger do not exist")),
    ]
    for name, fn in stages:
        run_stage(name, fn, results)
    calls = cctx.tracker.count
    ing = next(r for r in results if r.name == "ingest").detail.get("ingest_cbbd_calls", 0)
    print(f"\n=== chain_daily_v2 {'DRY RUN' if a.dry_run else 'LIVE RUN'} date={today} slate={slate_date} ===")
    for r in results:
        print(f"  [{r.status.upper():9}] {r.name:10} ({r.seconds:5.1f}s) {r.error or json.dumps(r.detail, default=str)[:260]}")
    print(f"--- CBBD calls: chain steps {calls}, ingest {ing} ---")
    if not a.dry_run:
        d = REPO / "data/processed/ingest/chain_v2"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{today}.json").write_text(json.dumps({"date": str(today), "slate": str(slate_date), "cbbd_calls": calls + ing,
                                                      "stages": [r.__dict__ for r in results]}, indent=2, default=str), encoding="utf-8")
    out = REPO / "results" / f"chain_daily_v2_{'dry' if a.dry_run else 'live'}_{today}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stages": [r.__dict__ for r in results], "cbbd_calls_chain": calls, "cbbd_calls_ingest": ing},
                              indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
