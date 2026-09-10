> **PARTIAL RUN** -- F1 selection fold only (`--skip-wf --skip-composed`, LightGBM grid truncated to 2 rungs and 2 seeds).

## 2. Run configuration (2026-09-10)

| item | value |
|---|---|
| trainer | `scripts/train_attribution_v1.py` |
| possessions version | `v2` (rim override 2.27 ft) |
| universe | D-I, non-truncated, `pbp_complete` |
| population rows (2024 + 2025) | reb_off 222,775, reb_def 526,310, made_fga 546,514, tov 250,065, miss_fga 689,375 |
| player-games with as-of inputs | 206,169 |
| team-games with as-of inputs | 21,246 |
| roster position known | 99.9791% |
| hoopR minutes joined through the player crosswalk | 99.8167% |
| player-games with a prior season of history | 2024: 0.0%, 2025: 68.4848% |
| game-level / composed draws, bootstrap reps | 20 / 20 / 100 |

### 2.1 Target coverage (reported, not silently filtered)

| season | target | population | binary rate | candidate set resolved | credited id present | modelled |
|---|---|---:|---:|---:|---:|---:|
| 2024 | REB_off | 107,768 | 100.0% | 95.7724% | 82.2517% | 77.8441% |
| 2024 | REB_def | 259,639 | 100.0% | 96.0079% | 92.7819% | 87.9833% |
| 2024 | assist | 267,004 | 50.5764% | 96.1501% | 99.8519% | 94.3595% |
| 2024 | steal | 121,637 | 54.9274% | 95.637% | 100.0% | 94.5175% |
| 2024 | block | 337,450 | 10.0939% | 95.837% | 100.0% | 94.924% |
| 2024 | assisted | 267,004 | 50.5764% | 96.1323% | 99.9251% | 96.1323% |
| 2024 | stolen | 121,637 | 54.9274% | 95.7168% | 100.0% | 95.7168% |
| 2024 | blocked | 337,450 | 10.0939% | 96.0036% | 100.0% | 96.0036% |
| 2025 | REB_off | 115,007 | 100.0% | 99.1157% | 82.5976% | 80.8594% |
| 2025 | REB_def | 266,671 | 100.0% | 99.1581% | 92.7382% | 90.8989% |
| 2025 | assist | 279,510 | 51.4822% | 99.2043% | 99.8436% | 97.7686% |
| 2025 | steal | 128,428 | 56.5383% | 99.0428% | 99.9959% | 97.886% |
| 2025 | block | 351,925 | 9.9084% | 99.1368% | 99.9971% | 98.1675% |
| 2025 | assisted | 279,510 | 51.4822% | 99.1778% | 99.9195% | 99.1778% |
| 2025 | stolen | 128,428 | 56.5383% | 99.0711% | 99.9977% | 99.0711% |
| 2025 | blocked | 351,925 | 9.9084% | 99.1609% | 99.9997% | 99.1609% |

## 3. F1 results (train 2024, test 2025) -- the selection fold

### Choice targets

**REB_off** (K = 5, uniform log loss 1.609438). Train 83,891, test 92,994. Fitted shrinkage: prior `position`, m = 25 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.434713 | 0.7278 | 0.3878 | 0.8114 | 1.773 | 4/4 | 0.935 | 0.002015 | 1.0079 | 4.828 / 4.664 | 38.43% / 39.96% | -1.89 | yes |
| cond_logit | 1.431192 | 0.7257 | 0.3885 | 0.8116 | 0.707 | 4/4 | 0.994 | 0.002128 | 1.0062 | 4.782 / 4.664 | 38.98% / 39.96% | -1.36 | yes |
| lgbm | 1.430888 | 0.7257 | 0.3874 | 0.8113 | 0.568 | 4/4 | 0.999 | 0.002117 | 1.0064 | 4.779 / 4.664 | 39.06% / 39.96% | -1.31 | yes |

**Decision: cond_logit.** the tree leads by 0.000304, inside the 0.002128 floor, so the pre-registration's requirement that a tree beat the best passing non-tree arm by more than the floor is not met; among the non-tree arms cond_logit are inside the floor of each other and the tie-break takes the simplest

P2 ridge penalty 0.001 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `[]`. Coefficients: `{'log_share': 1.08896, 'is_G': 0.01302, 'is_F': -0.03345, 'is_C': 0.01192, 'log_share_x_scorediff': 0.0153, 'log_share_x_sec': 0.02992, 'log_share_x_rim': 0.13292, 'log_share_x_three': -0.1516, 'log_share_x_ft': -0.26221, 'is_C_x_scorediff': 0.01455, 'is_C_x_sec': -0.01394, 'is_C_x_rim': 0.11804, 'is_C_x_three': -0.12822, 'is_C_x_ft': -0.41392}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.430888, 1.430953], seed SD 4.6e-05; feature importance `{'share': 863, 'rate': 785, 'position_code': 107, 'rate_rank': 100, 'score_diff': 752, 'sec_remaining': 1228, 'shot_class_code': 365}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.416133 | 1.436831 | 1.453856 |
| cond_logit | 1.411731 | 1.434530 | 1.449261 |
| lgbm | 1.411069 | 1.434009 | 1.449782 |

(n transfers = 28,612)

**REB_def** (K = 5, uniform log loss 1.609438). Train 228,439, test 242,401. Fitted shrinkage: prior `position`, m = 50 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.565561 | 0.7819 | 0.2898 | 0.7063 | 0.717 | 4/4 | 0.984 | 0.000756 | 0.9855 | 7.641 / 7.464 | 27.53% / 29.05% | -1.89 | yes |
| cond_logit | 1.560274 | 0.7798 | 0.2899 | 0.7065 | 0.426 | 4/4 | 1.004 | 0.000799 | 0.9856 | 7.636 / 7.464 | 27.64% / 29.05% | -1.80 | yes |
| lgbm | 1.559167 | 0.7793 | 0.2914 | 0.7077 | 0.282 | 4/4 | 1.003 | 0.000793 | 0.9855 | 7.638 / 7.464 | 27.70% / 29.05% | -1.77 | yes |

**Decision: lgbm.** eligible: lgbm 1.559167, cond_logit 1.560274, proportional 1.565561; floor 0.000799; clear of the next eligible arm by 0.001107 (1.4 floors)

P2 ridge penalty 1.0 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `[]`. Coefficients: `{'log_share': 1.13889, 'is_G': -0.2941, 'is_F': -0.29702, 'is_C': -0.26453, 'log_share_x_scorediff': 0.00384, 'log_share_x_sec': 0.01389, 'log_share_x_rim': -0.04848, 'log_share_x_three': -0.49661, 'log_share_x_ft': 0.7817, 'is_C_x_scorediff': 0.0054, 'is_C_x_sec': -0.01086, 'is_C_x_rim': 0.12396, 'is_C_x_three': -0.06439, 'is_C_x_ft': -0.05226}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.559167, 1.559195], seed SD 2e-05; feature importance `{'share': 765, 'rate': 740, 'position_code': 232, 'rate_rank': 104, 'score_diff': 791, 'sec_remaining': 1113, 'shot_class_code': 455}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.553335 | 1.562436 | 1.586914 |
| cond_logit | 1.548215 | 1.557233 | 1.581259 |
| lgbm | 1.547855 | 1.555761 | 1.579871 |

(n transfers = 75,197)

**assist** (K = 4, uniform log loss 1.386294). Train 127,424, test 140,687. Fitted shrinkage: prior `position`, m = 10 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.295860 | 0.7020 | 0.4034 | 0.8549 | 2.779 | 4/4 | 0.955 | 0.001420 | 1.0102 | 5.858 / 5.602 | 35.27% / 37.83% | -2.66 | NO (calibration, top1_top3) |
| cond_logit | 1.294558 | 0.7012 | 0.4043 | 0.8556 | 1.927 | 4/4 | 0.994 | 0.001466 | 1.0088 | 5.826 / 5.602 | 35.59% / 37.83% | -2.32 | NO (top1_top3) |
| lgbm | 1.292465 | 0.7003 | 0.4045 | 0.8556 | 0.418 | 4/4 | 1.003 | 0.001484 | 1.0098 | 5.826 / 5.602 | 36.03% / 37.83% | -2.18 | NO (top1_top3) |

**Decision: NO WINNER.** no arm passes every pre-registered gate

P2 ridge penalty 0.001 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `['log_share_x_ft', 'is_C_x_ft']`. Coefficients: `{'log_share': 1.04163, 'is_G': 0.02047, 'is_F': 0.00504, 'is_C': -0.0677, 'log_share_x_scorediff': 0.00411, 'log_share_x_sec': -0.04023, 'log_share_x_rim': 0.06436, 'log_share_x_three': 0.10737, 'is_C_x_scorediff': -0.00471, 'is_C_x_sec': 0.03603, 'is_C_x_rim': 0.26319, 'is_C_x_three': -0.20221}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.292465, 1.292469], seed SD 3e-06; feature importance `{'share': 921, 'rate': 857, 'position_code': 191, 'rate_rank': 80, 'score_diff': 758, 'sec_remaining': 1148, 'shot_class_code': 245}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.283143 | 1.286535 | 1.331516 |
| cond_logit | 1.281088 | 1.285467 | 1.330766 |
| lgbm | 1.278051 | 1.284202 | 1.328308 |

(n transfers = 43,860)

**steal** (K = 5, uniform log loss 1.609438). Train 63,149, test 71,076. Fitted shrinkage: prior `position`, m = 50 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.586249 | 0.7904 | 0.2626 | 0.6791 | 0.814 | 4/4 | 0.997 | 0.000866 | 1.0103 | 4.317 / 4.231 | 39.63% / 40.48% | -1.10 | yes |
| cond_logit | 1.586254 | 0.7904 | 0.2625 | 0.6790 | 0.998 | 4/4 | 1.014 | 0.000855 | 1.0103 | 4.314 / 4.231 | 39.66% / 40.48% | -1.08 | yes |
| lgbm | 1.588532 | 0.7913 | 0.2608 | 0.6760 | 1.367 | 4/4 | 1.021 | 0.000902 | 1.0109 | 4.311 / 4.231 | 39.72% / 40.48% | -1.04 | yes |

**Decision: proportional.** eligible: proportional 1.586249, cond_logit 1.586254, lgbm 1.588532; floor 0.000902; proportional, cond_logit are inside the floor of each other; the pre-registered tie-break takes the simplest

P2 ridge penalty 10.0 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `['log_share_x_rim', 'log_share_x_three', 'log_share_x_ft', 'is_C_x_rim', 'is_C_x_three', 'is_C_x_ft']`. Coefficients: `{'log_share': 0.95072, 'is_G': -0.08653, 'is_F': -0.08605, 'is_C': -0.10983, 'log_share_x_scorediff': 0.01871, 'log_share_x_sec': 0.03284, 'is_C_x_scorediff': -0.04216, 'is_C_x_sec': 0.01773}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.588532, 1.588392], seed SD 9.9e-05; feature importance `{'share': 994, 'rate': 907, 'position_code': 96, 'rate_rank': 131, 'score_diff': 828, 'sec_remaining': 1244, 'shot_class_code': 0}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.582536 | 1.582656 | 1.597363 |
| cond_logit | 1.582101 | 1.582684 | 1.597876 |
| lgbm | 1.584454 | 1.584466 | 1.600953 |

(n transfers = 21,947)

**block** (K = 5, uniform log loss 1.609438). Train 32,333, test 34,231. Fitted shrinkage: prior `position`, m = 10 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.284843 | 0.6535 | 0.4936 | 0.8591 | 4.937 | 4/4 | 0.939 | 0.004781 | 1.0451 | 2.504 / 2.393 | 59.09% / 61.55% | -0.92 | NO (calibration, top1_top3) |
| cond_logit | 1.273338 | 0.6472 | 0.4946 | 0.8592 | 2.422 | 4/4 | 0.983 | 0.004951 | 1.0363 | 2.483 / 2.393 | 59.54% / 61.55% | -0.74 | NO (calibration, top1_top3) |
| lgbm | 1.273376 | 0.6469 | 0.4956 | 0.8588 | 0.360 | 4/4 | 1.000 | 0.004940 | 1.0392 | 2.471 / 2.393 | 59.92% / 61.55% | -0.67 | yes |

**Decision: lgbm.** eligible: lgbm 1.273376; floor 0.004940; the only eligible arm

P2 ridge penalty 10.0 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `['log_share_x_ft', 'is_C_x_ft']`. Coefficients: `{'log_share': 0.99326, 'is_G': -0.07497, 'is_F': -0.06965, 'is_C': 0.15451, 'log_share_x_scorediff': 0.02672, 'log_share_x_sec': -0.01258, 'log_share_x_rim': 0.15517, 'log_share_x_three': -0.48433, 'is_C_x_scorediff': 0.00112, 'is_C_x_sec': -0.02278, 'is_C_x_rim': -0.01539, 'is_C_x_three': -0.42806}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.273376, 1.273084], seed SD 0.000207; feature importance `{'share': 1036, 'rate': 904, 'position_code': 75, 'rate_rank': 82, 'score_diff': 745, 'sec_remaining': 1144, 'shot_class_code': 214}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.299108 | 1.276291 | 1.283285 |
| cond_logit | 1.286514 | 1.264500 | 1.273580 |
| lgbm | 1.289596 | 1.264353 | 1.270351 |

(n transfers = 10,189)
