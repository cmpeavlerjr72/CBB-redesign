# run_laneD_f2_tap_v1.ps1 -- lane D 2026-10-01: fold-2 local tap (DIRECTION ONLY, Decision 12: underpowered).
# S2 = plain default (served stack v2) on the stride-500 verified sample at offsets 0 and 1000-4000 (floor draws);
# FRa / FTa = the chain gate under --gate-stack adopted on the same sample at offset 0. 25 seeds each.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"; $env:CBB_TRUTH = "verified_v1"
foreach ($v in 'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','LIGHTGBM_NUM_THREADS') { Set-Item "env:$v" 1 }
Get-ChildItem env: | Where-Object { $_.Name -like "ENGINE_*" } | ForEach-Object { Remove-Item "env:$($_.Name)" }
$W = 3
$S = "data/processed/truth/stride500_verified_v1_F2_2025.parquet"
foreach ($off in 0, 1000, 2000, 3000, 4000) {
  $tag = "laneDd_S2_s25_o$off"
  if (-not (Test-Path "results/engine_v0/$tag/run_meta.json")) {
    Write-Output "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K') S2 offset $off"
    # the sample runner pins the pre-adoption clock; overlay v2 --clock pins the adopted L2 (proved == plain default, 60 x 5)
    & .venv/Scripts/python.exe scripts/run_engine_overlay_v2.py --overrides data/processed/models/full_retrain_v1/laneD_day/overrides_empty.json --runner sample --clock v5b_r6L2_glat_pmean -- --sample-file $S --arm round2_s1 `
      --input-dir data/processed/models/engine_v3 --seeds 25 --seed-offset $off --workers $W --tag $tag --results-dir results/engine_v0
    if ($LASTEXITCODE -ne 0) { throw "S2 $off failed" }
  }
  if (-not (Test-Path "results/engine_v0/v3full_grade/${tag}__verified.md")) {
    & .venv/Scripts/python.exe scripts/eval_gates.py --results "results/engine_v0/$tag" --season 2025 --out "results/engine_v0/v3full_grade/${tag}__verified.md"
  }
}
foreach ($pair in @(@("F_R","FRa_loc"), @("F_T","FTa_loc"))) {
  Write-Output "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K') chain gate $($pair[1])"
  & .venv/Scripts/python.exe scripts/chain_full_retrain_v1.py --variant $pair[0] --tag $pair[1] --stages parity,gate --redo gate `
    --gate-stack adopted --gate-mode sample --gate-sample $S --gate-seeds 25 --gate-offsets 0 --cores $W --gate-workers $W
  if ($LASTEXITCODE -ne 0) { throw "chain $($pair[1]) failed" }
}
Write-Output "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K') done"
