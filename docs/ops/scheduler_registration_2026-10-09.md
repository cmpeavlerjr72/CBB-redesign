# Daily chain registered in Windows Task Scheduler (ops worker, 2026-10-09)

## Tasks
Registered under the current user with `scripts/ops_register_tasks_v1.ps1 -Register -SkipLines`. That is the existing registrar plus a new `-SkipLines` switch, which leaves the hourly lines-snapshot task unregistered (not asked for here).

| task | schedule (machine time zone = ET) | pass |
|---|---|---|
| `\CBB\DailyChain_Evening` | daily 20:00 | evening: simulates the NEXT game day; every game whose tip, real or placeholder, is after the clock |
| `\CBB\DailyChain_Morning` | daily 09:00 | morning: same-day re-publish of games with a REAL tip still ahead |

Command (action): `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\devuser\CBB-clean-sheet\scripts\run_daily_chain_pass_v1.ps1" -Pass <evening|morning> -Seeds 200`, working directory = repo.
- The wrapper sets `PYTHONIOENCODING=utf-8`, `CBB_TRUTH=verified_v1` and the thread-count pins (OMP/MKL/OPENBLAS/NUMEXPR = 1).
- It runs `.venv\Scripts\python.exe scripts\chain_daily_v3.py --pass <pass> --seeds 200`, the served daily chain.
- The task exit code = the chain's: 1 if any stage crashed.
- 200 seeds is the chain default. It is not the seed-count minimum for reading probabilities or ROI (pre-registered 2,500; `daily_chain_v3` doc section 7).

Settings:
- Principal `devuser`, interactive logon, RunLevel Limited (no stored password), so the tasks run only while the user is logged on.
- StartWhenAvailable (a missed run starts at next logon or wake), WakeToRun.
- MultipleInstances IgnoreNew; time limit 3 h.
- DisallowStartIfOnBatteries / StopIfGoingOnBatteries are on (Windows default; irrelevant on a desktop).

Log: `results\ops\daily_chain_<pass>_<yyyyMMdd>.log` (gitignored), one file per pass per day, appended with start / end lines carrying the exit code. Before 2026-10-09 the wrapper logged to `results\chain_logs\`. The chain also writes `results\chain_daily_v3_live_<date>.json` and the tracked `data\processed\ingest\chain_v3\<date>.json`.

## Timing
- Measured: the on-demand evening pass took 13.0 min end to end; the sim stage was 752 s for 118 games x 200 seeds, about 0.032 s per game-seed on one core.
- The evening pass at 20:00 ET finishes by about 20:15. The earliest real tip in the current table (mid-November) is 11:00 ET, and one 03:00 ET tip appears on 11-02 (likely an overseas game; the evening pass covers it).
- The morning pass at 09:00 simulates only real-tip games (27 on 11-02 today), so it is a few minutes. That leaves 2 h before an 11:00 ET tip even on a full real-tip day (about 150 games x 200 seeds is roughly 16 min at the measured rate).

## On-demand verification run
`schtasks /run /tn "\CBB\DailyChain_Evening"` at 08:32:22 ET, waited in-turn.
- `LastTaskResult = 0`. The log `results\ops\daily_chain_evening_20261009.log` ends `end exit=0`.
- Stages: schedule, tips, lines, lines_snapshot, ingest, ratings, injuries, injuries_parse, overrides, inputs, sim, publish OK. lines_probe, kenpom, rosters (weekly cadence, not due), grade and bias_clv were skipped (season not running). 6 CBBD calls.
- Output `results\daily\sim\2026-11-02\s200_o0\`: 118 games x 200 seeds. A3+R1 applied: 236 of 236 team-games seeded, anon slot share 0.237. Game-mean total 140.05. `pre_tip_basis`: 91 placeholder games (flagged unverified) and 27 real-tip games.
- Publish `results\daily\publish\2026-11-02\s200_o0\efb6280495\` (no lines posted yet).

First unattended pass: `\CBB\DailyChain_Morning` fired on its own trigger at 09:00:01 ET.
- `LastTaskResult = 0`; log `results\ops\daily_chain_morning_20261009.log` ends `end exit=0`, 3.8 min end to end.
- 27 real-tip games simulated (sim 197.5 s); 91 placeholder games refused by the morning rule.
- Publish `s200_o0_morning/f4d200824a`.
- Passes so far: 2 (1 on-demand, 1 unattended). The evening trigger fires tonight at 20:00.

## Disable / remove
- Pause one task: `schtasks /change /tn "\CBB\DailyChain_Evening" /disable` (`/enable` to resume); same for `DailyChain_Morning`.
- Remove both: `powershell -NoProfile -File scripts\ops_register_tasks_v1.ps1 -Unregister -Force -SkipLines`.
- Without `-SkipLines` it also tries to remove the lines task: harmless, since it is not registered.
- Inspect: `Get-ScheduledTask -TaskPath "\CBB\"`, `Get-ScheduledTaskInfo -TaskPath "\CBB\" -TaskName DailyChain_Evening`.

## Known limits (PM)
1. **No re-sim across days.** Until 11-02 every pass targets slate 11-02 with an identical sim config. The sim config hash has no build date, so passes after the first return the cached sim (`cached: true`); only the stages before the sim (schedule, tips, ratings, rosters) refresh. The 5 unattended passes therefore prove scheduling and ingestion, not five fresh sims. On 11-01 evening the served sim would be the first run's (10-09) unless the cache rule changes or the run dir is cleared. See `docs/ops/a3_seed_wiring_2026-10-09.md` section 4.
2. The tasks run only while `devuser` is logged on (LIMITED, no stored password). Unattended through logoff or reboot needs "run whether user is logged on or not", which needs a stored password. That is a user decision.
3. Two tasks may overlap only if an evening pass ran past 09:00, which is not plausible at 13 min. They write separate logs and run ids; both rewrite the same ratings file and chain json for the day, with deterministic content.
