# run_laneD_f2_tap_floors_v1.ps1 -- lane D 2026-10-01: the four S2 seed-offset floor draws of the fold-2 local tap
# (500 x 25 stride sample, served stack v2 through overlay v2 --clock L2; direction only, Decision 12).
$env:PYTHONIOENCODING = "utf-8"; $env:CBB_TRUTH = "verified_v1"
foreach ($v in 'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','LIGHTGBM_NUM_THREADS') { Set-Item "env:$v" 1 }
Get-ChildItem env: | Where-Object { $_.Name -like "ENGINE_*" } | ForEach-Object { Remove-Item "env:$($_.Name)" }
$S = "data/processed/truth/stride500_verified_v1_F2_2025.parquet"
foreach ($off in 1000, 2000, 3000, 4000) {
  $tag = "laneDd_S2_s25_o$off"
  if (-not (Test-Path "results/engine_v0/$tag/run_meta.json")) {
    Write-Output "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K') S2 offset $off"
    & .venv/Scripts/python.exe scripts/run_engine_overlay_v2.py --overrides data/processed/models/full_retrain_v1/laneD_day/overrides_empty.json `
      --runner sample --clock v5b_r6L2_glat_pmean -- --sample-file $S --arm round2_s1 --input-dir data/processed/models/engine_v3 `
      --seeds 25 --seed-offset $off --workers 3 --tag $tag --results-dir results/engine_v0
    if ($LASTEXITCODE -ne 0) { throw "S2 $off failed" }
  }
  if (-not (Test-Path "results/engine_v0/v3full_grade/${tag}__verified.md")) {
    & .venv/Scripts/python.exe scripts/eval_gates.py --results "results/engine_v0/$tag" --season 2025 --out "results/engine_v0/v3full_grade/${tag}__verified.md"
  }
}
Write-Output "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K') done"
