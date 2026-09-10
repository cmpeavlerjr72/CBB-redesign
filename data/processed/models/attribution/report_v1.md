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
| game-level / composed draws, bootstrap reps | 40 / 20 / 200 |

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
| proportional | 1.434713 | 0.7278 | 0.3878 | 0.8114 | 1.773 | 4/4 | 0.935 | 0.002067 | 1.0078 | 4.828 / 4.664 | 38.45% / 39.96% | -1.89 | yes |
| cond_logit | 1.431192 | 0.7257 | 0.3885 | 0.8116 | 0.707 | 4/4 | 0.994 | 0.002192 | 1.0059 | 4.782 / 4.664 | 38.99% / 39.96% | -1.37 | yes |
| lgbm | 1.430888 | 0.7257 | 0.3874 | 0.8113 | 0.568 | 4/4 | 0.999 | 0.002202 | 1.0061 | 4.778 / 4.664 | 39.07% / 39.96% | -1.31 | yes |

**Decision: cond_logit.** the tree leads by 0.000304, inside the 0.002202 floor, so the pre-registration's requirement that a tree beat the best passing non-tree arm by more than the floor is not met; among the non-tree arms cond_logit are inside the floor of each other and the tie-break takes the simplest

P2 ridge penalty 0.001 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `[]`. Coefficients: `{'log_share': 1.08896, 'is_G': 0.01302, 'is_F': -0.03345, 'is_C': 0.01192, 'log_share_x_scorediff': 0.0153, 'log_share_x_sec': 0.02992, 'log_share_x_rim': 0.13292, 'log_share_x_three': -0.1516, 'log_share_x_ft': -0.26221, 'is_C_x_scorediff': 0.01455, 'is_C_x_sec': -0.01394, 'is_C_x_rim': 0.11804, 'is_C_x_three': -0.12822, 'is_C_x_ft': -0.41392}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.430888, 1.430953, 1.430952], seed SD 3.7e-05; feature importance `{'share': 863, 'rate': 785, 'position_code': 107, 'rate_rank': 100, 'score_diff': 752, 'sec_remaining': 1228, 'shot_class_code': 365}`.

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
| proportional | 1.565561 | 0.7819 | 0.2898 | 0.7063 | 0.717 | 4/4 | 0.984 | 0.000690 | 0.9851 | 7.641 / 7.464 | 27.53% / 29.05% | -1.89 | yes |
| cond_logit | 1.560274 | 0.7798 | 0.2899 | 0.7065 | 0.426 | 4/4 | 1.004 | 0.000759 | 0.9857 | 7.635 / 7.464 | 27.64% / 29.05% | -1.79 | yes |
| lgbm | 1.559167 | 0.7793 | 0.2914 | 0.7077 | 0.282 | 4/4 | 1.003 | 0.000756 | 0.9857 | 7.636 / 7.464 | 27.70% / 29.05% | -1.77 | yes |

**Decision: lgbm.** eligible: lgbm 1.559167, cond_logit 1.560274, proportional 1.565561; floor 0.000759; clear of the next eligible arm by 0.001107 (1.5 floors)

P2 ridge penalty 1.0 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `[]`. Coefficients: `{'log_share': 1.13889, 'is_G': -0.2941, 'is_F': -0.29702, 'is_C': -0.26453, 'log_share_x_scorediff': 0.00384, 'log_share_x_sec': 0.01389, 'log_share_x_rim': -0.04848, 'log_share_x_three': -0.49661, 'log_share_x_ft': 0.7817, 'is_C_x_scorediff': 0.0054, 'is_C_x_sec': -0.01086, 'is_C_x_rim': 0.12396, 'is_C_x_three': -0.06439, 'is_C_x_ft': -0.05226}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.559167, 1.559195, 1.558969], seed SD 0.000123; feature importance `{'share': 765, 'rate': 740, 'position_code': 232, 'rate_rank': 104, 'score_diff': 791, 'sec_remaining': 1113, 'shot_class_code': 455}`.

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
| proportional | 1.295860 | 0.7020 | 0.4034 | 0.8549 | 2.779 | 4/4 | 0.955 | 0.001384 | 1.0101 | 5.858 / 5.602 | 35.26% / 37.83% | -2.66 | NO (calibration, top1_top3) |
| cond_logit | 1.294558 | 0.7012 | 0.4043 | 0.8556 | 1.927 | 4/4 | 0.994 | 0.001419 | 1.0087 | 5.826 / 5.602 | 35.59% / 37.83% | -2.32 | NO (top1_top3) |
| lgbm | 1.292465 | 0.7003 | 0.4045 | 0.8556 | 0.418 | 4/4 | 1.003 | 0.001450 | 1.0093 | 5.828 / 5.602 | 36.02% / 37.83% | -2.20 | NO (top1_top3) |

**Decision: NO WINNER.** no arm passes every pre-registered gate

P2 ridge penalty 0.001 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `['log_share_x_ft', 'is_C_x_ft']`. Coefficients: `{'log_share': 1.04163, 'is_G': 0.02047, 'is_F': 0.00504, 'is_C': -0.0677, 'log_share_x_scorediff': 0.00411, 'log_share_x_sec': -0.04023, 'log_share_x_rim': 0.06436, 'log_share_x_three': 0.10737, 'is_C_x_scorediff': -0.00471, 'is_C_x_sec': 0.03603, 'is_C_x_rim': 0.26319, 'is_C_x_three': -0.20221}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.292465, 1.292469, 1.29255], seed SD 4.8e-05; feature importance `{'share': 921, 'rate': 857, 'position_code': 191, 'rate_rank': 80, 'score_diff': 758, 'sec_remaining': 1148, 'shot_class_code': 245}`.

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
| proportional | 1.586249 | 0.7904 | 0.2626 | 0.6791 | 0.814 | 4/4 | 0.997 | 0.000908 | 1.0104 | 4.317 / 4.231 | 39.63% / 40.48% | -1.09 | yes |
| cond_logit | 1.586254 | 0.7904 | 0.2625 | 0.6790 | 0.998 | 4/4 | 1.014 | 0.000892 | 1.0104 | 4.315 / 4.231 | 39.66% / 40.48% | -1.07 | yes |
| lgbm | 1.588532 | 0.7913 | 0.2608 | 0.6760 | 1.367 | 4/4 | 1.021 | 0.000954 | 1.0110 | 4.312 / 4.231 | 39.72% / 40.48% | -1.03 | yes |

**Decision: proportional.** eligible: proportional 1.586249, cond_logit 1.586254, lgbm 1.588532; floor 0.000954; proportional, cond_logit are inside the floor of each other; the pre-registered tie-break takes the simplest

P2 ridge penalty 10.0 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `['log_share_x_rim', 'log_share_x_three', 'log_share_x_ft', 'is_C_x_rim', 'is_C_x_three', 'is_C_x_ft']`. Coefficients: `{'log_share': 0.95072, 'is_G': -0.08653, 'is_F': -0.08605, 'is_C': -0.10983, 'log_share_x_scorediff': 0.01871, 'log_share_x_sec': 0.03284, 'is_C_x_scorediff': -0.04216, 'is_C_x_sec': 0.01773}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.588532, 1.588392, 1.588602], seed SD 0.000107; feature importance `{'share': 994, 'rate': 907, 'position_code': 96, 'rate_rank': 131, 'score_diff': 828, 'sec_remaining': 1244, 'shot_class_code': 0}`.

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
| proportional | 1.284843 | 0.6535 | 0.4936 | 0.8591 | 4.937 | 4/4 | 0.939 | 0.004514 | 1.0453 | 2.503 / 2.393 | 59.10% / 61.55% | -0.91 | NO (calibration, top1_top3) |
| cond_logit | 1.273338 | 0.6472 | 0.4946 | 0.8592 | 2.422 | 4/4 | 0.983 | 0.004768 | 1.0369 | 2.482 / 2.393 | 59.56% / 61.55% | -0.72 | NO (calibration) |
| lgbm | 1.273376 | 0.6469 | 0.4956 | 0.8588 | 0.360 | 4/4 | 1.000 | 0.004718 | 1.0392 | 2.470 / 2.393 | 59.94% / 61.55% | -0.66 | yes |

**Decision: lgbm.** eligible: lgbm 1.273376; floor 0.004718; the only eligible arm

P2 ridge penalty 10.0 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `['log_share_x_ft', 'is_C_x_ft']`. Coefficients: `{'log_share': 0.99326, 'is_G': -0.07497, 'is_F': -0.06965, 'is_C': 0.15451, 'log_share_x_scorediff': 0.02672, 'log_share_x_sec': -0.01258, 'log_share_x_rim': 0.15517, 'log_share_x_three': -0.48433, 'is_C_x_scorediff': 0.00112, 'is_C_x_sec': -0.02278, 'is_C_x_rim': -0.01539, 'is_C_x_three': -0.42806}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.273376, 1.273084, 1.273295], seed SD 0.000151; feature importance `{'share': 1036, 'rate': 904, 'position_code': 75, 'rate_rank': 82, 'score_diff': 745, 'sec_remaining': 1144, 'shot_class_code': 214}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.299108 | 1.276291 | 1.283285 |
| cond_logit | 1.286514 | 1.264500 | 1.273580 |
| lgbm | 1.289596 | 1.264353 | 1.270351 |

(n transfers = 10,189)

### Binary targets

| target | base rate | arm | log loss | Brier | calib worst (pp) | resp steps | slopes (off / def) | boot SE | count (sim / real) | count SD ratio | eligible |
|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|---|
| assisted | 51.496% | team_ridge | 0.684722 | 0.245813 | 2.101 | 4/4 | 1.039 / 0.974 | 0.000256 | 12.899 / 13.129 | 0.9290 | NO (calibration) |
| assisted | 51.496% | aware_ridge | 0.558057 | 0.189083 | 1.954 | 4/4 | 1.082 / 0.956 | 0.000965 | 13.047 / 13.129 | 0.9356 | yes |
| assisted | 51.496% | lgbm | 0.558261 | 0.189120 | 1.843 | 4/4 | 1.051 / 0.958 | 0.000998 | 13.055 / 13.129 | 0.9399 | yes |
| stolen | 56.5222% | team_ridge | 0.682932 | 0.244926 | 2.525 | 4/4 | 1.067 / 1.019 | 0.000369 | 6.425 / 6.614 | 0.9503 | NO (calibration) |
| stolen | 56.5222% | aware_ridge | 0.681489 | 0.244231 | 4.334 | 4/4 | 1.077 / 1.015 | 0.000420 | 6.423 / 6.614 | 0.9495 | NO (calibration) |
| stolen | 56.5222% | lgbm | 0.642243 | 0.228656 | 6.850 | 3/4 | 1.194 / 0.975 | 0.000890 | 6.398 / 6.614 | 0.9550 | NO (calibration) |
| blocked | 9.906% | team_ridge | 0.320757 | 0.088842 | 0.538 | 4/4 | 1.074 / 1.025 | 0.001217 | 3.238 / 3.179 | 0.9457 | yes |
| blocked | 9.906% | aware_ridge | 0.264337 | 0.078387 | 0.295 | 4/4 | 1.076 / 1.015 | 0.001088 | 3.187 / 3.179 | 0.9740 | yes |
| blocked | 9.906% | lgbm | 0.266217 | 0.078939 | 3.017 | 4/4 | 1.023 / 1.026 | 0.001117 | 3.189 / 3.179 | 0.9809 | NO (calibration) |

**assisted decision: aware_ridge.** eligible: aware_ridge 0.558057, lgbm 0.558261; floor 0.000998; aware_ridge, lgbm are inside the floor of each other; the pre-registered tie-break takes the simplest Fitted shrinkage: m_team 100, own-share prior `position` m_own 10.

**stolen decision: NO WINNER.** no arm passes every pre-registered gate Fitted shrinkage: m_team 200, own-share prior `position` m_own 25.

**blocked decision: aware_ridge.** eligible: aware_ridge 0.264337, team_ridge 0.320757; floor 0.001217; clear of the next eligible arm by 0.056420 (46.4 floors) Fitted shrinkage: m_team 200, own-share prior `position` m_own 50.

## 4. Robustness fold: within-2025 walk-forward (train before 2025-01-15, test after)

### Choice targets

**REB_off** (K = 5, uniform log loss 1.609438). Train 46,579, test 46,415. Fitted shrinkage: prior `position`, m = 25 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.417909 | 0.7213 | 0.3972 | 0.8222 | 1.807 | 4/4 | 0.943 | 0.002857 | 1.0005 | 4.657 / 4.526 | 39.69% / 40.93% | -1.51 | yes |
| cond_logit | 1.414486 | 0.7191 | 0.3977 | 0.8230 | 0.864 | 4/4 | 1.006 | 0.003079 | 0.9954 | 4.602 / 4.526 | 40.34% / 40.93% | -0.90 | yes |
| lgbm | 1.415901 | 0.7199 | 0.3974 | 0.8231 | 0.844 | 4/4 | 0.988 | 0.003147 | 0.9952 | 4.597 / 4.526 | 40.52% / 40.93% | -0.82 | yes |

**Decision: cond_logit.** eligible: cond_logit 1.414486, lgbm 1.415901, proportional 1.417909; floor 0.003147; cond_logit, lgbm are inside the floor of each other; the pre-registered tie-break takes the simplest

P2 ridge penalty 10.0 (searched on `2024-12-06`); dropped as unidentified: `[]`; dropped as constant in this fold: `[]`. Coefficients: `{'log_share': 1.08404, 'log_prior_share': 0.01516, 'is_G': -0.02558, 'is_F': -0.07046, 'is_C': 0.06782, 'log_share_x_scorediff': 0.01972, 'log_share_x_sec': 0.02751, 'log_share_x_rim': 0.11492, 'log_share_x_three': -0.18013, 'log_share_x_ft': -0.28033, 'is_C_x_scorediff': 0.01225, 'is_C_x_sec': -0.00041, 'is_C_x_rim': 0.00929, 'is_C_x_three': -0.15507, 'is_C_x_ft': -0.36006}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.415901], seed SD None; feature importance `{'share': 751, 'rate': 800, 'prior_rate': 869, 'position_code': 61, 'rate_rank': 91, 'score_diff': 610, 'sec_remaining': 765, 'shot_class_code': 253}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.386487 | 1.425330 | 1.443226 |
| cond_logit | 1.384803 | 1.423609 | 1.434613 |
| lgbm | 1.396494 | 1.420073 | 1.432274 |

(n transfers = 14,184)

**REB_def** (K = 5, uniform log loss 1.609438). Train 119,713, test 122,688. Fitted shrinkage: prior `position`, m = 50 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.561747 | 0.7803 | 0.2943 | 0.7123 | 0.528 | 4/4 | 0.998 | 0.001005 | 0.9734 | 7.393 / 7.249 | 28.35% / 29.62% | -1.57 | yes |
| cond_logit | 1.556737 | 0.7784 | 0.2939 | 0.7120 | 0.462 | 4/4 | 1.022 | 0.001104 | 0.9733 | 7.386 / 7.249 | 28.47% / 29.62% | -1.46 | yes |
| lgbm | 1.556970 | 0.7786 | 0.2937 | 0.7109 | 1.350 | 4/4 | 1.031 | 0.001133 | 0.9728 | 7.375 / 7.249 | 28.70% / 29.62% | -1.25 | yes |

**Decision: cond_logit.** eligible: cond_logit 1.556737, lgbm 1.556970, proportional 1.561747; floor 0.001133; cond_logit, lgbm are inside the floor of each other; the pre-registered tie-break takes the simplest

P2 ridge penalty 10.0 (searched on `2024-12-06`); dropped as unidentified: `[]`; dropped as constant in this fold: `[]`. Coefficients: `{'log_share': 1.15011, 'log_prior_share': 0.01999, 'is_G': -0.0126, 'is_F': 0.0057, 'is_C': 0.02399, 'log_share_x_scorediff': 0.00598, 'log_share_x_sec': 0.00056, 'log_share_x_rim': -0.04882, 'log_share_x_three': -0.52324, 'log_share_x_ft': 0.69615, 'is_C_x_scorediff': 0.02238, 'is_C_x_sec': 0.01123, 'is_C_x_rim': 0.11251, 'is_C_x_three': -0.08803, 'is_C_x_ft': -0.10715}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.55697], seed SD None; feature importance `{'share': 688, 'rate': 678, 'prior_rate': 830, 'position_code': 114, 'rate_rank': 89, 'score_diff': 637, 'sec_remaining': 795, 'shot_class_code': 369}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.545667 | 1.558084 | 1.588563 |
| cond_logit | 1.542210 | 1.553494 | 1.580841 |
| lgbm | 1.536930 | 1.553783 | 1.587846 |

(n transfers = 37,401)

**assist** (K = 4, uniform log loss 1.386294). Train 70,329, test 70,358. Fitted shrinkage: prior `position`, m = 10 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.283382 | 0.6956 | 0.4149 | 0.8622 | 3.236 | 4/4 | 0.928 | 0.001792 | 0.9923 | 5.659 / 5.459 | 36.37% / 38.57% | -2.10 | NO (calibration, top1_top3) |
| cond_logit | 1.282563 | 0.6952 | 0.4170 | 0.8632 | 3.178 | 4/4 | 0.926 | 0.001831 | 0.9923 | 5.653 / 5.459 | 36.39% / 38.57% | -2.04 | NO (calibration, top1_top3) |
| lgbm | 1.283376 | 0.6953 | 0.4167 | 0.8596 | 1.403 | 4/4 | 0.951 | 0.001965 | 0.9910 | 5.616 / 5.459 | 37.37% / 38.57% | -1.50 | yes |

**Decision: lgbm.** eligible: lgbm 1.283376; floor 0.001965; the only eligible arm

P2 ridge penalty 0.1 (searched on `2024-12-07`); dropped as unidentified: `[]`; dropped as constant in this fold: `['log_share_x_ft', 'is_C_x_ft']`. Coefficients: `{'log_share': 1.05774, 'log_prior_share': 0.01741, 'is_G': -1.09031, 'is_F': -1.13487, 'is_C': -1.14833, 'log_share_x_scorediff': 0.00979, 'log_share_x_sec': -0.04561, 'log_share_x_rim': -0.04536, 'log_share_x_three': 0.02909, 'is_C_x_scorediff': 0.00402, 'is_C_x_sec': 0.03185, 'is_C_x_rim': 0.23451, 'is_C_x_three': -0.26217}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.283376], seed SD None; feature importance `{'share': 761, 'rate': 741, 'prior_rate': 907, 'position_code': 127, 'rate_rank': 83, 'score_diff': 561, 'sec_remaining': 797, 'shot_class_code': 223}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.265965 | 1.272902 | 1.326435 |
| cond_logit | 1.268096 | 1.273805 | 1.318427 |
| lgbm | 1.255797 | 1.278638 | 1.328470 |

(n transfers = 21,548)

**steal** (K = 5, uniform log loss 1.609438). Train 35,709, test 35,367. Fitted shrinkage: prior `position`, m = 50 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.582525 | 0.7890 | 0.2691 | 0.6819 | 0.978 | 4/4 | 1.016 | 0.001318 | 1.0022 | 4.193 / 4.122 | 40.55% / 41.32% | -1.04 | yes |
| cond_logit | 1.582869 | 0.7891 | 0.2682 | 0.6820 | 1.016 | 4/4 | 0.990 | 0.001307 | 1.0027 | 4.194 / 4.122 | 40.52% / 41.32% | -1.06 | yes |
| lgbm | 1.585498 | 0.7902 | 0.2671 | 0.6794 | 1.949 | 4/4 | 0.990 | 0.001479 | 1.0012 | 4.174 / 4.122 | 40.80% / 41.32% | -0.82 | yes |

**Decision: proportional.** eligible: proportional 1.582525, cond_logit 1.582869, lgbm 1.585498; floor 0.001479; proportional, cond_logit are inside the floor of each other; the pre-registered tie-break takes the simplest

P2 ridge penalty 10.0 (searched on `2024-12-06`); dropped as unidentified: `[]`; dropped as constant in this fold: `['log_share_x_rim', 'log_share_x_three', 'log_share_x_ft', 'is_C_x_rim', 'is_C_x_three', 'is_C_x_ft']`. Coefficients: `{'log_share': 0.93055, 'log_prior_share': 0.0122, 'is_G': -0.01266, 'is_F': -0.02502, 'is_C': -0.09818, 'log_share_x_scorediff': 0.04263, 'log_share_x_sec': 0.00706, 'is_C_x_scorediff': 0.01242, 'is_C_x_sec': 0.03689}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.585498], seed SD None; feature importance `{'share': 806, 'rate': 790, 'prior_rate': 893, 'position_code': 63, 'rate_rank': 102, 'score_diff': 683, 'sec_remaining': 863, 'shot_class_code': 0}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.574934 | 1.578510 | 1.598912 |
| cond_logit | 1.578524 | 1.579579 | 1.594011 |
| lgbm | 1.581979 | 1.578642 | 1.601965 |

(n transfers = 10,797)

**block** (K = 5, uniform log loss 1.609438). Train 16,997, test 17,234. Fitted shrinkage: prior `position`, m = 10 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.255963 | 0.6406 | 0.5046 | 0.8709 | 4.615 | 4/4 | 0.945 | 0.006696 | 1.0157 | 2.436 / 2.345 | 60.30% / 62.34% | -0.67 | NO (calibration, top1_top3) |
| cond_logit | 1.245607 | 0.6344 | 0.5050 | 0.8713 | 2.162 | 4/4 | 0.991 | 0.007327 | 1.0016 | 2.408 / 2.345 | 60.92% / 62.34% | -0.43 | NO (calibration) |
| lgbm | 1.250223 | 0.6367 | 0.5039 | 0.8682 | 2.216 | 4/4 | 1.005 | 0.007571 | 0.9982 | 2.392 / 2.345 | 61.40% / 62.34% | -0.35 | NO (calibration) |

**Decision: NO WINNER.** no arm passes every pre-registered gate

P2 ridge penalty 0.001 (searched on `2024-12-07`); dropped as unidentified: `[]`; dropped as constant in this fold: `['log_share_x_ft', 'is_C_x_ft']`. Coefficients: `{'log_share': 0.91187, 'log_prior_share': 0.03099, 'is_G': -0.06218, 'is_F': -0.07642, 'is_C': 0.13861, 'log_share_x_scorediff': 0.01062, 'log_share_x_sec': -0.00348, 'log_share_x_rim': 0.21739, 'log_share_x_three': -0.50065, 'is_C_x_scorediff': 0.02492, 'is_C_x_sec': -0.05106, 'is_C_x_rim': 0.01094, 'is_C_x_three': -0.73191}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.250223], seed SD None; feature importance `{'share': 760, 'rate': 835, 'prior_rate': 881, 'position_code': 87, 'rate_rank': 99, 'score_diff': 561, 'sec_remaining': 802, 'shot_class_code': 175}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.266553 | 1.251666 | 1.251348 |
| cond_logit | 1.268135 | 1.241602 | 1.226416 |
| lgbm | 1.278244 | 1.247507 | 1.222216 |

(n transfers = 5,066)

### Binary targets

| target | base rate | arm | log loss | Brier | calib worst (pp) | resp steps | slopes (off / def) | boot SE | count (sim / real) | count SD ratio | eligible |
|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|---|
| assisted | 50.8949% | team_ridge | 0.684975 | 0.245943 | 2.538 | 4/4 | 0.779 / 0.918 | 0.000424 | 13.291 / 12.986 | 0.9196 | NO (calibration, responsiveness) |
| assisted | 50.8949% | aware_ridge | 0.562365 | 0.190793 | 6.075 | 4/4 | 0.877 / 0.915 | 0.001361 | 13.330 / 12.986 | 0.9353 | NO (calibration) |
| assisted | 50.8949% | lgbm | 0.564868 | 0.191704 | 4.149 | 4/4 | 0.958 / 0.884 | 0.001488 | 13.322 / 12.986 | 0.9524 | NO (calibration) |
| stolen | 57.3861% | team_ridge | 0.680055 | 0.243502 | 2.535 | 4/4 | 0.915 / 0.717 | 0.000594 | 6.301 / 6.489 | 0.9528 | NO (calibration, responsiveness) |
| stolen | 57.3861% | aware_ridge | 0.677667 | 0.242385 | 6.738 | 4/4 | 1.168 / 0.771 | 0.000727 | 6.327 / 6.489 | 0.9556 | NO (calibration, responsiveness) |
| stolen | 57.3861% | lgbm | 0.653014 | 0.231290 | 9.253 | 3/4 | 0.832 / 0.708 | 0.001213 | 6.423 / 6.489 | 0.9758 | NO (calibration, responsiveness) |
| blocked | 9.8955% | team_ridge | 0.320429 | 0.088745 | 1.084 | 4/4 | 0.696 / 1.111 | 0.001789 | 3.187 / 3.161 | 0.9735 | NO (responsiveness) |
| blocked | 9.8955% | aware_ridge | 0.264498 | 0.078322 | 1.033 | 4/4 | 0.920 / 1.095 | 0.001558 | 3.206 / 3.161 | 1.0050 | yes |
| blocked | 9.8955% | lgbm | 0.268812 | 0.079517 | 5.597 | 4/4 | 0.856 / 1.081 | 0.001647 | 3.200 / 3.161 | 1.0199 | NO (calibration) |

**assisted decision: NO WINNER.** no arm passes every pre-registered gate Fitted shrinkage: m_team 50, own-share prior `prior_season` m_own 10.

**stolen decision: NO WINNER.** no arm passes every pre-registered gate Fitted shrinkage: m_team 200, own-share prior `league` m_own 400.

**blocked decision: aware_ridge.** eligible: aware_ridge 0.264498; floor 0.001558; the only eligible arm Fitted shrinkage: m_team 200, own-share prior `prior_season` m_own 10.
