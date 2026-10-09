# Scheduler: logon / run settings matched to the machine's other tasks (ops worker, 2026-10-09)

Sibling of `scheduler_registration_2026-10-09.md` (which recorded the first registration, Interactive logon). Only `\CBB\DailyChain_Evening` and `\CBB\DailyChain_Morning` were touched; no other task was changed.

## What the other tasks use
Inspected with `Get-ScheduledTask | Get-ScheduledTaskInfo` (all non-Microsoft tasks owned by `devuser`, 78, not counting the two `\CBB\` tasks; counts include disabled ones).

| logon type | tasks | examples (recurring, last result 0) |
|---|---|---|
| **S4U, RunLevel Limited** | 55 (44 enabled) | `SleeperDailySummary`, `SleeperTradeWatch`, `SleeperDMWatch`, `OET League message`, `XSportsbook`, `XScoutFast`, `XBotLive`, `XBotPlan`, `XVoiceLearn`, `CFB-HF-BulkPush`, `CFB_SyncFeed`, `KalshiRFQ_SniperKeeper`, `KalshiRFQ_StartTimesBuilder`, `QuadsWeekly` |
| Interactive, Limited | 21 | `TennisDailyPredictions`, `CollegeBaseballDailyRun`, `CBB_Monte_Daily`, `KalshiRFQ_PruneLogs`, OneDrive and SoftLanding tasks |
| Interactive, Highest | 2 | `MLB Daily Sims`, `NASCAR_Weekly_Predictions` |
| stored password (LogonType Password) | 0 | |
| service account | 1 (`cfb-espn-snapshots-nightly`, SYSTEM) | |

The tasks the user relies on to "run consistently every day" are the S4U family. A stored password (LogonType Password) is used by none of them, so no password has to be supplied.

Settings of the S4U daily tasks (Sleeper, OET, CFB, X*): StartWhenAvailable True for the Sleeper / OET / CFB group (False for most X* tasks), WakeToRun False (only the Kalshi morning tasks and `MLB Daily Sims` wake), MultipleInstances IgnoreNew (all), DisallowStartIfOnBatteries True for the Sleeper / OET group, execution time limit varies by job (PT10M to PT72H, PT0S for the long-lived feeds).

## What was set
`scripts/ops_register_tasks_v1.ps1 -Register -SkipLines` (re-registers with `-Force`; the script now builds the principal with `New-ScheduledTaskPrincipal -LogonType S4U -RunLevel Limited` for the current user):

| setting | before (10-09 first registration) | now | matches |
|---|---|---|---|
| principal / LogonType | `devuser` / Interactive | `devuser` / **S4U** | all S4U daily tasks |
| RunLevel | Limited | Limited | all |
| run whether user is logged on or not | no | **yes** (S4U, no stored password) | S4U tasks |
| StartWhenAvailable | True | True | Sleeper / OET / CFB group (a missed run starts at next availability) |
| WakeToRun | True | **False** | every S4U daily task (none wakes the machine) |
| MultipleInstances | IgnoreNew | IgnoreNew | all |
| DisallowStartIfOnBatteries | True | True | Sleeper / OET group |
| ExecutionTimeLimit | PT3H | PT3H | (job specific; the pass takes 13 min at 200 seeds) |
| triggers | daily 20:00 and 09:00 | unchanged | |

S4U needs no stored password. The chain uses only HTTP APIs with keys on disk (`.env` of the repo and the CFBD key file) and local files, all of which an S4U logon reads. What S4U cannot do is reach network resources that need the user's stored credentials (mapped drives, DPAPI-protected secrets); the chain uses neither. The user does not need to re-save the tasks with a password.

Dropping WakeToRun follows the instruction to match the other tasks; it means a pass scheduled while the machine sleeps starts at wake (StartWhenAvailable) rather than waking it. The other daily tasks have run through the same condition without trouble.

## Verification
- `schtasks /run /tn "\CBB\DailyChain_Evening"` at 09:38:18 ET, waited in-turn (the pass is the 200-seed served run). The log `results\ops\daily_chain_evening_20261009.log` ends `2026-10-09 09:50:46 -04:00 end exit=0`; `LastTaskResult = 0`. `schtasks /query` shows Logon Mode `Interactive/Background` (how Windows labels S4U).
- That run was the first with the new cache key and `players=True`: cache miss on the old (key-less) 10-09 sim, 118 games x 200 seeds re-run in 720 s, 422,860 player rows. A second on-demand pass after the injury-feed wiring is recorded in `injury_feed_2026-10-09.md`.
- The Morning task did not fire on registration (its 09:00 trigger had passed; no catch-up run was started).

## Undo
`schtasks /change /tn "\CBB\DailyChain_Evening" /disable`, or the previous behaviour: edit the script's `$principal` line back to `-LogonType Interactive` and re-register.
