<#
ops_register_tasks_v1.ps1 -- Task Scheduler definitions for the daily chain (2026-27 deployment).
Default is WHAT-IF: prints what would be created and changes nothing. Pass -Register to actually create/replace the tasks.
  powershell -NoProfile -File scripts\ops_register_tasks_v1.ps1                  # preview
  powershell -NoProfile -File scripts\ops_register_tasks_v1.ps1 -Register        # create (PM go-ahead first)
  powershell -NoProfile -File scripts\ops_register_tasks_v1.ps1 -Unregister -Force  # remove both (without -Force: preview)
Jobs (machine time zone is Eastern; times are ET):
  CBB\DailyChain_Evening  20:00 daily, run_daily_chain_pass_v1.ps1 -Pass evening  (simulates the NEXT game day; only pass that can use placeholder tips)
  CBB\DailyChain_Morning  09:00 daily, run_daily_chain_pass_v1.ps1 -Pass morning  (re-publishes TODAY once tips are real)
  CBB\LinesSnapshot_GameDay  hourly 10:00-23:00 ET (14 runs/day, repeat 1 h for 13 h), pull_espn_odds_snapshot_v1.py for today's ET slate
                              (ESPN current line + CBBD /lines, appended with captured_at to data/raw/lines_snapshots; then the open/close builder)
Settings: StartWhenAvailable, WakeToRun, 3 h limit, no parallel instances, run only when logged on (LIMITED, no stored password).
-Seeds default 200 is the chain default, not a recommendation (seed-count study: daily_chain doc section 3).
#>
#   -SkipLines: leave \CBB\LinesSnapshot_GameDay alone (registration of the two chain passes only; 2026-10-09).
param([switch]$Register, [switch]$Unregister, [switch]$Force, [switch]$SkipLines, [int]$Seeds = 200,
      [string]$Repo = "C:\Users\devuser\CBB-clean-sheet")
$ErrorActionPreference = "Stop"
$wrapper = Join-Path $Repo "scripts\run_daily_chain_pass_v1.ps1"
if (-not (Test-Path $wrapper)) { throw "wrapper missing: $wrapper" }
$defs = @(
  @{ Name = "DailyChain_Evening"; At = "20:00"; Pass = "evening" },
  @{ Name = "DailyChain_Morning"; At = "09:00"; Pass = "morning" }
)
foreach ($d in $defs) {
  $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper`" -Pass $($d.Pass) -Seeds $Seeds"
  if ($Unregister) {
    if ($Register) { throw "-Register and -Unregister are exclusive" }
    Write-Host "UNREGISTER \CBB\$($d.Name)"
    if (-not $Force) { Write-Host "  (preview; add -Force to remove)"; continue }
    Unregister-ScheduledTask -TaskName $d.Name -TaskPath "\CBB\" -Confirm:$false -ErrorAction SilentlyContinue
    continue
  }
  Write-Host ("{0} \CBB\{1}: daily {2} ET: powershell.exe {3}" -f $(if ($Register) {"REGISTER"} else {"WHATIF  "}), $d.Name, $d.At, $arg)
  if (-not $Register) { continue }
  $action   = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $Repo
  $trigger  = New-ScheduledTaskTrigger -Daily -At $d.At
  $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -MultipleInstances IgnoreNew `
                -ExecutionTimeLimit (New-TimeSpan -Hours 3)
  Register-ScheduledTask -TaskName $d.Name -TaskPath "\CBB\" -Action $action -Trigger $trigger -Settings $settings `
    -RunLevel Limited -Force | Out-Null
}
# Hourly game-day lines snapshot (own open/close capture). Same WHATIF default; -Unregister handled below.
if ($SkipLines) { Write-Host "SKIP \CBB\LinesSnapshot_GameDay (-SkipLines)"; exit 0 }
$py = Join-Path $Repo ".venv\Scripts\python.exe"
$snapCmd = "& `"$py`" scripts\pull_espn_odds_snapshot_v1.py; if (`$?) { & `"$py`" scripts\build_lines_open_close_v1.py }"
$snapArg = "-NoProfile -ExecutionPolicy Bypass -Command `"`$env:PYTHONIOENCODING='utf-8'; $snapCmd`""
if ($Unregister) {
  Write-Host "UNREGISTER \CBB\LinesSnapshot_GameDay"
  if ($Force) { Unregister-ScheduledTask -TaskName "LinesSnapshot_GameDay" -TaskPath "\CBB\" -Confirm:$false -ErrorAction SilentlyContinue }
} else {
  Write-Host ("{0} \CBB\LinesSnapshot_GameDay: hourly 10:00-23:00 ET: powershell.exe {1}" -f $(if ($Register) {"REGISTER"} else {"WHATIF  "}), $snapArg)
  if ($Register) {
    $action   = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $snapArg -WorkingDirectory $Repo
    $trigger  = New-ScheduledTaskTrigger -Daily -At "10:00" -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Hours 13)
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
    Register-ScheduledTask -TaskName "LinesSnapshot_GameDay" -TaskPath "\CBB\" -Action $action -Trigger $trigger -Settings $settings `
      -RunLevel Limited -Force | Out-Null
  }
}
if (-not $Register -and -not $Unregister) { Write-Host "(preview only; nothing created. Re-run with -Register after PM go-ahead.)" }
