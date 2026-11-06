# Registers the Kalshi collector as a Windows scheduled task that starts at boot and
# restarts if it dies. S4U logon (no interactive session needed; per the kalshi-rfq rule
# that Interactive-logon tasks never fire on this box).
#
#   powershell -ExecutionPolicy Bypass -File register_task.ps1          # install / replace
#   powershell -ExecutionPolicy Bypass -File register_task.ps1 -Remove  # uninstall
#   schtasks /run /tn KalshiLeaderboardCollector                        # start now
#   schtasks /end /tn KalshiLeaderboardCollector                        # stop
param([switch]$Remove)

$TaskName = "KalshiLeaderboardCollector"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = (Get-Command python).Source
$Script = Join-Path $Root "collector\collector.py"
$Log = Join-Path $Root "logs\task_stdout.log"

if ($Remove) {
    schtasks /delete /tn $TaskName /f | Out-Null
    Write-Host "removed $TaskName"
    exit 0
}

# run_collector.cmd holds the restart loop (goto labels do not work inside cmd /c one-liners)
$Cmd = Join-Path $Root "collector\run_collector.cmd"
$Action = New-ScheduledTaskAction -Execute $Cmd -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$Principal = New-ScheduledTaskPrincipal -UserId "devuser" -LogonType S4U -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
Write-Host "registered $TaskName (AtStartup, S4U). Start now with: schtasks /run /tn $TaskName"
