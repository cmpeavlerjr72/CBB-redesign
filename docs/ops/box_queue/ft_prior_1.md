# ft_prior_1: free_throw section 20 arm `P1` (prior_season_fta in the FT make model), full-size DESCRIPTIVE read

- **Label:** DESCRIPTIVE. NOT a ship-decision read.
  - P1 did not win section 20; the registered vetoes are in `docs/models/free_throw/experiments.md` section 21.
  - Queue this only if the PM wants the full-size G1-G9 picture of the thin-sample channel. The local 50-seed reads are in section 21.1.
- **Commit:** this file's commit or later on `main`. The engine `src/` is unchanged. `ENGINE_FT_SCORE=<arm>` is the existing default-off loader.
- **Floors:** the Decision 12 floors, lane D's served-v2 draws (as in `scoring_1.md`).

## 0. Setup (on the box, after the standard pull + parity v9 setup, e.g. `scripts/box_scoring1_setup_v1.sh <SHA>`)

Both builds are deterministic and read only tracked files plus the pulled `engine_v3` inputs:
```
python scripts/build_ft_prior_serving_v1.py artifacts F2   # -> free_throw/s1_scorediff/P1/S1_conf_aligned/F2 (29 fits, ~3 min); F2 log loss must read 0.574931
python scripts/build_ft_prior_serving_v1.py inputs F2      # -> data/processed/models/engine_v3_FTP_P1; builder_report.json max_abs_diff 0.0, unchanged_columns_identical true
```
Run both through the container (`scripts/box_run_v3.sh scripts/...`) when the host has no venv.

## 1. Full-size read (5,710 x 200, offset 0)

```
export CBB_TRUTH=verified_v1
ENGINE_FT_SCORE=P1 scripts/box_run_v3.sh scripts/run_engine.py --fold F2 --season 2025 \
  --seeds 200 --seed-offset 0 --workers 90 --games-per-block 60 --seeds-per-block 25 \
  --input-dir data/processed/models/engine_v3_FTP_P1 --tag ftprior1_P1_s200_o0 --results-dir results/engine_v0
scripts/box_run_v3.sh scripts/eval_gates.py --results results/engine_v0/ftprior1_P1_s200_o0 --season 2025 \
  --out results/engine_v0/v3full_grade/ftprior1_P1_s200_o0__verified.md
scripts/box_run_v3.sh scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_COMB9GCTKD_s200_o0 \
  --floors d1001D_S2f1_s200_o1000,d1001D_S2f2_s200_o2000,d1001D_S2f3_s200_o3000,d1001D_S2f4_s200_o4000 \
  --arms ftprior1_P1_s200_o0 --out-json results/ftprior1/boot_P1_vs_S2.json --out-md results/ftprior1/boot_P1_vs_S2.md
```

- **Env check.** Both of these must hold, or the run is the plain default:
  - `run_meta.json` `adapter_flags` shows `ENGINE_FT_SCORE: P1`;
  - the free_throw source path contains `s1_scorediff/P1`.
- **Input check.** `input_dir` must be `engine_v3_FTP_P1`. With the flag on and the served `engine_v3` inputs, the adapter cannot find `prior_season_fta` and must fail loudly.

## Sync back

- `results/engine_v0/ftprior1_P1_s200_o0/{games.parquet,run_meta.json}` and its `__verified.md`.
- `results/ftprior1/boot_P1_vs_S2.*`.
- `data/processed/models/engine_v3_FTP_P1/builder_report.json`.
