# scoring_1: free_throw section 18 winner `X1` (days_since_start in the FT make model) inside served stack v2 at full size

- **Commit:** this file's commit or later on `main`. The engine `src/` is unchanged, and `ENGINE_FT_SCORE=<arm>` is the existing default-off loader.
- **Pre-registration / results:** `docs/models/free_throw/experiments.md` sections 18-19.
- **Label:** SHIP-DECISION gate read. The registered line is G9 total bias toward 0 with no G1-G9 veto, under the Decision 12 floors.

## 0. Setup
```
python scripts/hf_sync_data.py pull --dirs model_artifacts --only 'free_throw/s1_scorediff/X1/**'
# check: data/processed/models/free_throw/s1_scorediff/X1/S1_conf_aligned/F2/manifest.json + 29 seg_*.joblib
```
Plain-default parity vs v9: skip it if this instance already passed at the current engine src.

## 1. Full-size read (5,710 x 200, offset 0; about 10-15 min at 90 workers)
```
export CBB_TRUTH=verified_v1
ENGINE_FT_SCORE=X1 scripts/box_run_v3.sh scripts/run_engine.py --fold F2 --season 2025 \
  --seeds 200 --seed-offset 0 --workers 90 --games-per-block 60 --seeds-per-block 25 \
  --input-dir data/processed/models/engine_v3 --tag scoring1_X1_s200_o0 --results-dir results/engine_v0
scripts/box_run_v3.sh scripts/eval_gates.py --results results/engine_v0/scoring1_X1_s200_o0 --season 2025 \
  --out results/engine_v0/v3full_grade/scoring1_X1_s200_o0__verified.md
scripts/box_run_v3.sh scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_COMB9GCTKD_s200_o0 \
  --floors d1001D_S2f1_s200_o1000,d1001D_S2f2_s200_o2000,d1001D_S2f3_s200_o3000,d1001D_S2f4_s200_o4000 \
  --arms scoring1_X1_s200_o0 --out-json results/scoring1/boot_X1_vs_S2.json --out-md results/scoring1/boot_X1_vs_S2.md
```
- **Env check:** `run_meta.json` `adapter_flags` must show `ENGINE_FT_SCORE: X1`, and the free_throw source path must contain `s1_scorediff/X1`. If either is missing, the run is the plain default.

## Sync back
`results/engine_v0/scoring1_X1_s200_o0/{games.parquet,run_meta.json}`, its `__verified.md`, `results/scoring1/boot_X1_vs_S2.*`.
The opening-window decomposition is re-graded locally with `scripts/diag_total_bias_decomp_v1.py` (point RUNS at the new dir).
