#!/usr/bin/env python
"""
ops_seal_week_v1.py -- the Oct-10 seal-week runbook (docs/ops/readiness_gaps_2026-10-05.md section 2) as ordered stages.

    .venv/Scripts/python.exe scripts/ops_seal_week_v1.py                      # DRY RUN (default): per-stage checks, writes nothing
    .venv/Scripts/python.exe scripts/ops_seal_week_v1.py --only ratings,a3_day1_priors
    .venv/Scripts/python.exe scripts/ops_seal_week_v1.py --execute            # runs the stage commands, in order, stop on first failure

Dry run, per stage: (1) the script exists and parses, (2) every documented argument appears in its source (AST/text check, the script is NOT
imported or run), (3) where a cheap probe exists, the named hard stop is triggered on purpose and must fire (so we know the seal blocks the
stage today and with which message), (4) the stage's own safe check command (a read-only dry run) is run when it has one.
Status: OK = checks pass, no seal involved; SEAL_OK = checks pass AND the seal hard stop fires as designed (stage runnable after the lift);
MISSING = a script the runbook needs does not exist yet; MANUAL = human / PM action, never automated; FAIL = a check failed.

Stage 1 is the served-default parity smoke (parity v10, served stack v3, 2026-10-07; the rehearsal of that day found it was a
manual step): its safe check RUNS it (60 games x 5 seeds, fold-2 2025 backtest inputs, ~30 s) and writes only its own gitignored
results/engine_v0/parity_smoke_* dir. Stage clock_serving checks that the served clock (K2) resolves a complete artifact set for
the first slate (carried-forward fold-2 refits, the same mechanism as every dated-refit family; --execute pulls a missing set).

--execute is REFUSED unless data/overrides/ratings_day1_choices.json has seal_lift_approved true. This script never writes that flag
(user / PM only, in either direction); the re-seal stage prints the instruction instead. 2025-26 data is never read in a dry run.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv/Scripts/python.exe") if (REPO / ".venv/Scripts/python.exe").exists() else sys.executable
CHOICES = REPO / "data/overrides/ratings_day1_choices.json"
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from cbb_sim.live.preseason import preseason_dir as _preseason_dir, preseason_rel as _preseason_rel  # noqa: E402,F401
PRESEASON_GAMES = _preseason_rel() + "/games_2027.parquet"


@dataclass
class Stage:
    name: str
    desc: str
    script: str | None                         # relative to repo; None = manual
    args: list[str] = field(default_factory=list)       # the execute command's arguments
    tokens: list[str] = field(default_factory=list)     # strings that must appear in the script source (documented arguments)
    probe: str | None = None                   # python snippet that must RAISE with `probe_expect` in the message (the seal hard stop)
    probe_expect: str = ""
    safe_check: list[str] | None = None        # read-only command run in a dry run
    seal: bool = False                         # stage needs the lifted seal
    note: str = ""
    runner: str = "py"                         # py | ps1 | pytest
    unseal: bool = False                       # --execute sets CBB_UNSEAL=1 for this stage (reads sealed 2025-26; refused anyway unless the flag is true)


def stages(slate: str, as_of: str, tip_to: str, roster_season: int = 2027) -> list[Stage]:
    sd = slate
    return [
        Stage("parity_v10", "served-default parity smoke: run_engine F2 2025 60 games x 5 seeds on engine_v3, digest vs parity v10 (bit-identical)",
              "scripts/run_parity_smoke_v1.py", ["--ref", "docs/ops/parity_reference_windows_v10.json"], tokens=["--ref", "--keep-env", "--workers"],
              safe_check=["scripts/run_parity_smoke_v1.py", "--ref", "docs/ops/parity_reference_windows_v10.json"],
              note="served stack v3 (clock K2, docs/tests/adoption_clock_K2_2026-10-07.md); v9 = SERVED_V2 (--keep-env with the SERVED_V2 values)"),
        Stage("preflight", "pyarrow imports; smoke tests (daily chain v3, hard stops, day-1)", "tests/test_daily_chain_v3.py", runner="pytest",
              args=["tests/test_daily_chain_v3.py", "tests/test_hard_stops_2027.py", "tests/test_day1_2027.py", "-q"],
              tokens=["def test"]),
        Stage("preseason_pull", "re-pull preseason (rosters, portal, schedule, lines probe) into a versioned sibling dir",
              "scripts/pull_preseason_refresh_v2.py", ["--tag", "{today}"], tokens=["--dry-run", "--tag"],
              note="also add West Florida (CBBD 1073) to team_crosswalk as a versioned sibling (manual; see crosswalk_check)"),
        Stage("crosswalk_check", "West Florida (CBBD 1073) present in team_crosswalk_v2.parquet (the file the chain uses)", None,
              safe_check=["-c", "import pandas as pd; c=pd.read_parquet('data/reference/team_crosswalk_v2.parquet'); "
                                "assert (c['cbbd_team_id']==1073).any(), 'West Florida 1073 missing: write team_crosswalk_v3 (versioned sibling)'; print('1073 present')"],
              note="if missing: write a versioned sibling (see build_team_crosswalk_v2_westflorida.py), PM switches the path"),
        Stage("rosters", "ESPN 2027 rosters -> data/raw/cbbd/rosters/roster_2027.parquet (hard checks 250 teams / 3000 players)",
              "scripts/pull_rosters_espn_v1.py", ["--season", str(roster_season)], tokens=["--season", "--out"]),
        Stage("ratings", "own ratings 2027 as-of (reads sealed 2025-26 through the chain's unsealed() context)",
              "scripts/chain_daily_v3.py", ["--slate-date", sd, "--dry-run"], tokens=["--slate-date"],
              probe=("import sys; sys.path[:0]=['scripts','src']; import chain_day1_2027_v1 as M; h,m=M.check_choices(); "
                     "assert 'seal_lift_approved' in m, 'seal flag already true'; raise RuntimeError('BLOCKED seal_lift_approved')"),
              probe_expect="seal_lift_approved", seal=True,
              note="the ratings stage runs inside chain_daily_v3 (--dry-run is the chain's own check; the real run is the live chain with no --dry-run, "
                   "executed by the execute path below as a plain chain run for the slate)"),
        Stage("ratings_parity", "own-ratings entry-point parity on 2025 (no sealed read)", "scripts/diag_own_ratings_asof_parity_v1.py",
              ["--season", "2025"], tokens=["--season", "--dates"]),
        Stage("r9ao3_prior", "R9ao3 and-one team prior with season 2027 (ao_team_prior_v2.parquet = v1 rows + season 2027 from 2026 data)",
              "scripts/build_ao_team_prior_v2.py", ["--season", "2027"], tokens=["--season", "--out", "--validate", "assert_not_sealed"],
              probe="import sys; sys.path[:0]=['scripts','src']; import build_ao_team_prior_v2 as M; M.build_season(2027)",
              probe_expect="sealed", seal=True, unseal=True,
              note="builder validated bit-identical on served seasons 2025 / 2026 (--validate); writes a versioned v2, PM switches the served path"),
        Stage("shot_block_prior", "shot-block 2027 anchor prior engine/shot_block_prior_2027_v1.parquet (PM-approved name)",
              "scripts/build_shot_block_prior_v1.py", ["--season", "2027"], tokens=["--season", "--out", "--validate", "assert_not_sealed"],
              probe="import sys; sys.path[:0]=['scripts','src']; import build_shot_block_prior_v1 as M; M.build_prior(2027)",
              probe_expect="sealed", seal=True, unseal=True,
              note="builder validated bit-identical to the stored 2025 prior_2024_end (--validate). The live LUT builder's own 2025-only guard is a separate step"),
        Stage("a3_day1_priors", "A3 day-1 player-prior build (returners + transfers-in, S-1 on-floor minutes) for the slate",
              "scripts/build_engine_inputs_day1prior_v1.py",
              ["serve", "--arm", "A3", "--season", "2027", "--fold", "F2", "--slate-date", sd, "--as-of", as_of,
               "--schedule-source", "cbbd", "--schedule-path", PRESEASON_GAMES, "--crosswalk", "data/reference/team_crosswalk_v2.parquet",
               "--roster", "data/raw/cbbd/rosters/roster_2027.parquet", "--ratings-dir", "data/processed/ratings_asof/{slate}",
               "--fallback", "R1", "--out-dir", "data/processed/models/engine_live", "--tag", "D1P_A3_F2_2027_{slate}"],
              tokens=["--arm", "--season", "--slate-date", "--as-of", "--schedule-source", "--schedule-path", "--crosswalk", "--roster", "--ratings-dir",
                      "--fallback", "--out-dir", "--tag"],
              probe=("import sys; sys.path[:0]=['scripts','src']; import build_engine_inputs_day1prior_v1 as M; M.tables(2027, True)"),
              probe_expect="sealed", seal=True, unseal=True,
              note="wired into the chain inputs stage for live 2027 via build_live seed_fn (chain_daily_v2.build_live_inputs); needs roster_2027.parquet (stage rosters) and ratings"),
        Stage("inputs_census", "live inputs 2027 census + degenerate day-1 feature list", "scripts/diag_live_day1_v1.py",
              ["--date", sd, "--crosswalk", "data/reference/team_crosswalk_v2.parquet"], tokens=["--date", "--crosswalk"], seal=False,
              note="dry run: ast check only (script builds live inputs); real run follows the ratings / priors stages"),
        Stage("tip_times", "tip-time table refresh for the next window (<= 14 days)", "scripts/pull_tip_times_v1.py",
              ["--season", "2027", "--from", sd, "--to", tip_to], tokens=["--from", "--to", "--dry-run", "--season"],
              safe_check=["scripts/pull_tip_times_v1.py", "--season", "2027", "--from", sd, "--to", tip_to, "--dry-run"]),
        Stage("injuries_parse", "league-wide ESPN injuries -> player_out_{date}.csv", "scripts/pull_injuries_player_out_v1.py", [],
              tokens=["--date", "--dry-run"], safe_check=["scripts/pull_injuries_player_out_v1.py", "--dry-run"]),
        Stage("clock_serving", "served clock (adapters.CLOCK_DEFAULT = K2) artifact set present and selected for the first slate",
              "scripts/diag_clock_serving_2027_v1.py", ["--slate-date", sd, "--pull"], tokens=["--slate-date", "--pull", "CLOCK_DEFAULT"],
              safe_check=["scripts/diag_clock_serving_2027_v1.py", "--slate-date", sd],
              note="clock/r8_K2 is gitignored (HF model_artifacts); a 2026-27 game is served by the last fold-2 refit (carried forward, "
                   "not refit through 2025-26; a 2025-26 refit is a PM retrain-set decision)"),
        Stage("chain_dry_run", "chain_daily_v3 dry run on the first slate until sim_prereqs is empty", "scripts/chain_daily_v3.py",
              ["--dry-run", "--slate-date", sd], tokens=["--dry-run", "--slate-date"],
              safe_check=["scripts/chain_daily_v3.py", "--dry-run", "--slate-date", sd, "--no-probe-hoopr"]),
        Stage("sim_4seed", "4-seed local sim on the slate (then paired parity check by the PM)", "scripts/chain_daily_v3.py",
              ["--dry-run", "--dry-run-sim", "--seeds", "4", "--slate-date", sd], tokens=["--dry-run-sim", "--seeds"], seal=True,
              note="blocked until ratings + sealed priors exist; the dry-run chain names each blocker"),
        Stage("scheduler_preview", "Task Scheduler jobs, PREVIEW only (register after the Oct-17 live build per PM ruling 2)",
              "scripts/ops_register_tasks_v1.ps1", [], runner="ps1", tokens=["-Register", "WakeToRun"],
              safe_check=["scripts/ops_register_tasks_v1.ps1"], note="never registers from this script; -Register is a PM-gated manual step"),
        Stage("hf_sync", "push bulk data to the private HF dataset", "scripts/hf_sync_data.py", ["push"], tokens=["push", "status"],
              safe_check=["scripts/hf_sync_data.py", "status"]),
        Stage("reseal", "RE-SEAL: set seal_lift_approved back to false and record the audit window closing", None,
              note="MANUAL, user / PM only. This script never writes the flag. Edit data/overrides/ratings_day1_choices.json, commit, push"),
    ]


def seal_approved(path: Path = CHOICES) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("seal_lift_approved") is True
    except (OSError, ValueError):
        return False


def env() -> dict:
    e = dict(os.environ, PYTHONIOENCODING="utf-8")
    e.pop("CBB_UNSEAL", None)                    # the dry run must never run unsealed
    return e


def run(cmd: list[str], timeout: int = 600) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=REPO, env=env(), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def source_ok(rel: str, tokens: list[str]) -> str | None:
    p = REPO / rel
    if not p.exists():
        return f"script missing: {rel}"
    src = p.read_text(encoding="utf-8", errors="replace")
    if rel.endswith(".py"):
        try:
            ast.parse(src)
        except SyntaxError as e:
            return f"syntax error {e}"
    miss = [t for t in tokens if t not in src]
    return f"documented args not found in source: {miss}" if miss else None


def check_stage(s: Stage, safe: bool = True) -> tuple[str, str]:
    if s.script is None and s.probe is None and not s.safe_check:
        return "MANUAL", s.note
    if s.script:
        err = source_ok(s.script, s.tokens)
        if err:
            return "FAIL", err
    if s.probe:
        rc, out = run([PY, "-c", s.probe], 300)
        if rc == 0:
            return "FAIL", "hard stop did NOT fire (probe ran clean): " + out[-200:]
        if s.probe_expect.lower() not in out.lower():
            return "FAIL", f"probe raised but not the named stop ({s.probe_expect!r}): " + out[-300:]
        last = out.strip().splitlines()[-1][:220]
        status = "MISSING" if s.script is None else "SEAL_OK"
        return status, f"hard stop fires: {last}" + (f" | {s.note}" if s.script is None else "")
    if s.safe_check and safe:
        if s.runner == "ps1":
            rc, out = run(["powershell", "-NoProfile", "-File", *s.safe_check], 120)
        else:
            rc, out = run([PY, *s.safe_check], 900)
        if rc != 0:
            return "FAIL", f"safe check exit {rc}: " + out[-400:]
        return "OK", "safe check passed: " + out.strip().splitlines()[-1][:200]
    if s.runner == "pytest":
        rc, out = run([PY, "-m", "pytest", *s.args], 900)
        return ("OK", out.strip().splitlines()[-1]) if rc == 0 else ("FAIL", out[-400:])
    return "OK", "script present, documented args present" + ("; needs the seal lift + upstream stages (not exercised here)" if s.seal else "")


def build_cmd(s: Stage, today: str, slate: str) -> list[str]:
    a = [x.replace("{today}", today).replace("{slate}", slate) for x in s.args]
    if s.runner == "pytest":
        return [PY, "-m", "pytest", *a]
    if s.runner == "ps1":
        return ["powershell", "-NoProfile", "-File", s.script, *a]
    return [PY, s.script, *a]


def main(argv=None) -> int:
    from datetime import date
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="run the stage commands (refused unless seal_lift_approved is true); default is a dry run")
    ap.add_argument("--slate-date", default="2026-11-02")
    ap.add_argument("--as-of", default=None, help="as-of date for the day-1 priors (default: the slate date)")
    ap.add_argument("--tip-to", default="2026-11-15")
    ap.add_argument("--only", default=None, help="comma-separated stage names")
    ap.add_argument("--skip", default="", help="comma-separated stage names")
    ap.add_argument("--no-safe-checks", action="store_true", help="dry run: skip the network / chain safe-check commands")
    a = ap.parse_args(argv)
    today = str(date.today())
    allst = stages(a.slate_date, a.as_of or a.slate_date, a.tip_to)
    names = [s.name for s in allst]
    only = [x for x in (a.only or "").split(",") if x]
    bad = [x for x in only + [y for y in a.skip.split(",") if y] if x not in names]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; stages: {names}")
    sel = [s for s in allst if (not only or s.name in only) and s.name not in a.skip.split(",")]
    approved = seal_approved()
    print(f"=== ops_seal_week_v1 {'EXECUTE' if a.execute else 'DRY RUN'}  slate={a.slate_date}  seal_lift_approved={approved} ===")
    if a.execute and not approved:
        print("REFUSED: data/overrides/ratings_day1_choices.json seal_lift_approved is not true (user / PM sets it; this script never does). Nothing was run.")
        return 2
    rc_all = 0
    for i, s in enumerate(sel, 1):
        if a.execute:
            if s.script is None and s.probe is not None:                # a probed stage with no script yet
                print(f"[{i:2}] [MISSING  ] {s.name:18} {s.note}")
                rc_all = 1
                break
            if s.script is None:
                print(f"[{i:2}] [MANUAL   ] {s.name:18} {s.note}")
                continue
            cmd = build_cmd(s, today, a.slate_date)
            if s.name == "ratings":                                    # real run: the live chain, not its dry run
                cmd = [PY, s.script, "--slate-date", a.slate_date]
            if s.name in ("sim_4seed",):
                cmd = [PY, s.script, "--seeds", "4", "--slate-date", a.slate_date]   # REAL under --execute (2026-10-08); the dry run stays the default
            print(f"[{i:2}] [RUNNING  ] {s.name:18} {' '.join(cmd)[:230]}", flush=True)
            e = dict(os.environ, PYTHONIOENCODING="utf-8", **({"CBB_UNSEAL": "1"} if s.unseal else {}))
            rc = subprocess.run(cmd, cwd=REPO, env=e).returncode
            print(f"[{i:2}] [{'DONE' if rc == 0 else 'FAILED':9}] {s.name:18} exit {rc}")
            if rc != 0:
                rc_all = rc
                break
            continue
        st, msg = check_stage(s, safe=not a.no_safe_checks)
        print(f"[{i:2}] [{st:9}] {s.name:18} {s.desc}\n{'':36}{msg[:420]}", flush=True)
        if st in ("FAIL",):
            rc_all = 1
        if st == "MISSING":
            print(f"{'':36}note: {s.note}")
    print("=== end (dry run: nothing written beyond the chain's own dry-run json and the parity smoke's results dir)" if not a.execute else "=== end ===")
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
