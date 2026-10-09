# Wrapper for the two-pass daily chain (lane F, 2026-10-01). NOT registered anywhere; docs/ops/day1_readiness_2027_2026-10-01.md section 3 has the schtasks lines.
#   -Pass evening  run at 20:00 ET the evening before the slate (chain picks the NEXT game day)
#   -Pass morning  run at 09:00 ET on the slate date (chain picks TODAY if it has games; only real tip times still in the future)
# One core: thread-count env vars pinned. Log: results\ops\daily_chain_<pass>_<yyyyMMdd>.log (gitignored; was results\chain_logs until 2026-10-09).
# Exit code = the chain's. Registered 2026-10-09 as \CBB\DailyChain_Evening / _Morning (docs/ops/scheduler_registration_2026-10-09.md).
param([Parameter(Mandatory = $true)][ValidateSet("evening", "morning")][string]$Pass,
      [int]$Seeds = 200,
      [switch]$DryRun)
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\devuser\CBB-clean-sheet"
$env:PYTHONIOENCODING = "utf-8"; $env:CBB_TRUTH = "verified_v1"
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:NUMEXPR_NUM_THREADS = "1"
New-Item -ItemType Directory -Force "results\ops" | Out-Null
$log = "results\ops\daily_chain_{0}_{1}.log" -f $Pass, (Get-Date -Format "yyyyMMdd")
$argv = @("scripts\chain_daily_v3.py", "--pass", $Pass, "--seeds", $Seeds)
if ($DryRun) { $argv += "--dry-run" }
"{0} start pass={1} seeds={2} dry={3}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss K"), $Pass, $Seeds, $DryRun.IsPresent | Out-File $log -Append -Encoding utf8
# cmd does the redirect: PowerShell 5.1 turns python's stderr (logging) into error records
$cmdline = '".venv\Scripts\python.exe" ' + ($argv -join " ") + " >> `"$log`" 2>&1"
cmd /c $cmdline
$code = $LASTEXITCODE
"{0} end exit={1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss K"), $code | Out-File $log -Append -Encoding utf8
exit $code
