## 2. Run configuration (2026-09-10)

| item | value |
|---|---|
| trainer | `scripts/train_usage_v1.py` |
| possessions version | `v2` (rim override 2.27 ft) |
| universe | D-I, non-truncated, `pbp_complete` |
| credited events (modelled) | 1,634,792 |
| player-games with as-of inputs | 204,642 |
| roster position known | 99.978% |
| hoopR minutes joined through the player crosswalk | 99.8148% |
| player-games with a prior season of on-floor history | 2024: 0.0%, 2025: 67.7982% |
| Monte-Carlo draws (marginal / Polya / game-level / alpha fit) | 200 / 100 / 40 / 15 |

### 2.1 Event coverage (reported, not silently filtered)

| season | class | events | five resolved | credited id present | modelled |
|---|---|---:|---:|---:|---:|
| 2024 | FGA_rim | 225,021 | 95.8817% | 99.9996% | 95.0382% |
| 2024 | FGA_jump2 | 153,702 | 96.2642% | 99.9707% | 95.3976% |
| 2024 | FGA_3 | 225,731 | 95.9886% | 99.9991% | 95.1637% |
| 2024 | TOV | 121,637 | 95.6239% | 94.9407% | 89.5394% |
| 2024 | FT_trip | 107,004 | 95.617% | 100.0% | 93.8881% |
| 2025 | FGA_rim | 235,458 | 99.1417% | 99.9983% | 98.291% |
| 2025 | FGA_jump2 | 149,087 | 99.0925% | 99.992% | 98.411% |
| 2025 | FGA_3 | 246,890 | 99.1871% | 99.998% | 98.4009% |
| 2025 | TOV | 128,428 | 99.1116% | 94.6632% | 92.764% |
| 2025 | FT_trip | 112,715 | 99.1048% | 99.9982% | 97.4919% |

## 3. F1 results (train 2024, test 2025) -- the selection fold

### 3.1 FGA_rim

Train 213,856 events, test 231,434. Fitted shrinkage: prior `position`, m = 50 pseudo on-floor events. Uniform-over-five log loss = 1.609438.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| proportional | 1.523130 | 0.7655 | 0.3194 | 0.7466 | 0.349 | 4/4 | 0.001049 | 1.0108 | 7.090 / 6.825 | 30.80% / 32.34% | 67.47% / 69.65% | NO (top1_top3) |
| dirichlet | 1.523130 | 0.7655 | 0.3194 | 0.7467 | 0.349 | 4/4 | 0.001049 | 1.0108 | 7.090 / 6.825 | 30.80% / 32.34% | 67.47% / 69.65% | NO (top1_top3) |
| hier_dirichlet | 1.523298 | 0.7656 | 0.3191 | 0.7454 | 0.350 | 4/4 | 0.001044 | 1.0287 | 7.060 / 6.825 | 31.19% / 32.34% | 67.83% / 69.65% | yes |
| cond_logit | 1.521954 | 0.7651 | 0.3210 | 0.7477 | 0.372 | 4/4 | 0.001069 | 1.0049 | 7.042 / 6.825 | 31.08% / 32.34% | 67.88% / 69.65% | yes |
| lgbm | 1.502348 | 0.7564 | 0.3375 | 0.7605 | 0.411 | 4/4 | 0.001157 | 1.0038 | 7.022 / 6.825 | 31.19% / 32.34% | 68.04% / 69.65% | yes |

**Decision: lgbm.** eligible: lgbm 1.502348, cond_logit 1.521954, hier_dirichlet 1.523298; floor 0.001157; clear of the next eligible arm by 0.019606 (16.9 floors)

Fitted concentrations -- U2 `{'a_between': None, 'a_handler': None, 'a_wing': None, 'a_big': None}`; U3 `{'a_between': None, 'a_handler': 64.0, 'a_wing': None, 'a_big': 2.0}` (`null` = no dispersion at that level, the top rung of the grid). Polya-urn (within-game, NOT a pregame quantity, never a decision input) log loss: U2 1.52313, U3 1.522956.

U4 ridge penalty 0.001; dropped as unidentified in this fold: `['log_prior_share', 'prior_rate']`; dropped as constant: `['is_transfer']`. Coefficients: `{'log_share': 0.99474, 'log_exposure': 0.06063, 'log_minutes': 0.01941, 'is_G': -0.27217, 'is_F': -0.2247, 'is_C': -0.3325, 'log_share_x_scorediff': 0.00947, 'log_share_x_sec': 0.00601, 'log_share_x_chance': -0.0649, 'is_C_x_scorediff': 0.01092, 'is_C_x_sec': 0.0599}`.

U5 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200} (searched on 2024 only); seed-varied refits [1.502348, 1.502367, 1.502369], seed SD 1.2e-05.

Transfer subset (2025 credited players whose modal team changed):

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.500334 | 1.515145 | 1.568013 |
| dirichlet | 1.500334 | 1.515145 | 1.568013 |
| hier_dirichlet | 1.501412 | 1.515169 | 1.567256 |
| cond_logit | 1.497221 | 1.509403 | 1.577930 |
| lgbm | 1.477031 | 1.490063 | 1.558594 |

(n transfers = 72,591)

Per-role "too narrow" statistic for the proportional arm:

| role | players | sim SD | real SD | ratio |
|---|---:|---:|---:|---:|
| handler | 1173 | 2.0267 | 2.09107 | 0.96922 |
| wing | 2575 | 1.38544 | 1.32748 | 1.04366 |
| big | 219 | 1.57581 | 1.60487 | 0.9819 |

### 3.2 FGA_jump2

Train 146,628 events, test 146,718. Fitted shrinkage: prior `league`, m = 50 pseudo on-floor events. Uniform-over-five log loss = 1.609438.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| proportional | 1.498369 | 0.7542 | 0.3389 | 0.7645 | 0.514 | 4/4 | 0.001611 | 1.0382 | 5.764 / 5.528 | 35.88% / 37.69% | 74.24% / 76.44% | NO (top1_top3) |
| dirichlet | 1.498369 | 0.7542 | 0.3390 | 0.7645 | 0.514 | 4/4 | 0.001611 | 1.0382 | 5.764 / 5.528 | 35.88% / 37.69% | 74.24% / 76.44% | NO (top1_top3) |
| hier_dirichlet | 1.498540 | 0.7543 | 0.3399 | 0.7634 | 0.547 | 4/4 | 0.001607 | 1.0484 | 5.746 / 5.528 | 36.15% / 37.69% | 74.45% / 76.44% | yes |
| cond_logit | 1.494403 | 0.7527 | 0.3418 | 0.7667 | 0.654 | 4/4 | 0.001590 | 1.0294 | 5.714 / 5.528 | 36.14% / 37.69% | 74.66% / 76.44% | yes |
| lgbm | 1.491034 | 0.7514 | 0.3430 | 0.7686 | 0.373 | 4/4 | 0.001574 | 1.0247 | 5.681 / 5.528 | 36.46% / 37.69% | 75.04% / 76.44% | yes |

**Decision: lgbm.** eligible: lgbm 1.491034, cond_logit 1.494403, hier_dirichlet 1.498540; floor 0.001607; clear of the next eligible arm by 0.003369 (2.1 floors)

Fitted concentrations -- U2 `{'a_between': None, 'a_handler': None, 'a_wing': None, 'a_big': None}`; U3 `{'a_between': None, 'a_handler': 64.0, 'a_wing': None, 'a_big': None}` (`null` = no dispersion at that level, the top rung of the grid). Polya-urn (within-game, NOT a pregame quantity, never a decision input) log loss: U2 1.498369, U3 1.497856.

U4 ridge penalty 0.001; dropped as unidentified in this fold: `['log_prior_share', 'prior_rate']`; dropped as constant: `['is_transfer']`. Coefficients: `{'log_share': 1.05434, 'log_exposure': 0.06756, 'log_minutes': 0.0764, 'is_G': 0.08286, 'is_F': 0.03601, 'is_C': -0.12452, 'log_share_x_scorediff': 0.02265, 'log_share_x_sec': -0.04219, 'log_share_x_chance': -0.19647, 'is_C_x_scorediff': 0.05925, 'is_C_x_sec': 0.07971}`.

U5 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200} (searched on 2024 only); seed-varied refits [1.491034, 1.490728, 1.490911], seed SD 0.000154.

Transfer subset (2025 credited players whose modal team changed):

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.488913 | 1.482348 | 1.541248 |
| dirichlet | 1.488913 | 1.482348 | 1.541248 |
| hier_dirichlet | 1.489059 | 1.482592 | 1.541318 |
| cond_logit | 1.481806 | 1.471998 | 1.553537 |
| lgbm | 1.478860 | 1.462863 | 1.560513 |

(n transfers = 46,219)

Per-role "too narrow" statistic for the proportional arm:

| role | players | sim SD | real SD | ratio |
|---|---:|---:|---:|---:|
| handler | 1172 | 1.63109 | 1.67614 | 0.97312 |
| wing | 2519 | 1.04335 | 0.9589 | 1.08807 |
| big | 213 | 1.05509 | 0.9821 | 1.07432 |

### 3.3 FGA_3

Train 214,814 events, test 242,942. Fitted shrinkage: prior `position`, m = 25 pseudo on-floor events. Uniform-over-five log loss = 1.609438.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| proportional | 1.461517 | 0.7480 | 0.3320 | 0.7926 | 0.523 | 4/4 | 0.001302 | 1.0823 | 6.820 / 6.689 | 32.09% / 32.33% | 69.97% / 70.57% | yes |
| dirichlet | 1.461517 | 0.7480 | 0.3320 | 0.7926 | 0.523 | 4/4 | 0.001302 | 1.0823 | 6.820 / 6.689 | 32.09% / 32.33% | 69.97% / 70.57% | yes |
| hier_dirichlet | 1.461517 | 0.7480 | 0.3320 | 0.7926 | 0.523 | 4/4 | 0.001302 | 1.0823 | 6.820 / 6.689 | 32.09% / 32.33% | 69.97% / 70.57% | yes |
| cond_logit | 1.460492 | 0.7475 | 0.3326 | 0.7932 | 0.361 | 4/4 | 0.001286 | 1.0770 | 6.810 / 6.689 | 32.11% / 32.33% | 70.04% / 70.57% | yes |
| lgbm | 1.458927 | 0.7471 | 0.3334 | 0.7939 | 0.255 | 4/4 | 0.001290 | 1.0728 | 6.783 / 6.689 | 32.10% / 32.33% | 70.12% / 70.57% | yes |

**Decision: lgbm.** eligible: lgbm 1.458927, cond_logit 1.460492, proportional 1.461517, dirichlet 1.461517, hier_dirichlet 1.461517; floor 0.001302; clear of the next eligible arm by 0.001565 (1.2 floors)

Fitted concentrations -- U2 `{'a_between': None, 'a_handler': None, 'a_wing': None, 'a_big': None}`; U3 `{'a_between': None, 'a_handler': None, 'a_wing': None, 'a_big': None}` (`null` = no dispersion at that level, the top rung of the grid). Polya-urn (within-game, NOT a pregame quantity, never a decision input) log loss: U2 1.461517, U3 1.461517.

U4 ridge penalty 0.001; dropped as unidentified in this fold: `['log_prior_share', 'prior_rate']`; dropped as constant: `['is_transfer']`. Coefficients: `{'log_share': 1.02907, 'log_exposure': 0.03924, 'log_minutes': 0.01457, 'is_G': -0.33295, 'is_F': -0.36424, 'is_C': -0.46804, 'log_share_x_scorediff': 0.00853, 'log_share_x_sec': -0.03192, 'log_share_x_chance': -0.02447, 'is_C_x_scorediff': 0.0504, 'is_C_x_sec': 0.02614}`.

U5 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200} (searched on 2024 only); seed-varied refits [1.458927, 1.459033, 1.458985], seed SD 5.3e-05.

Transfer subset (2025 credited players whose modal team changed):

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.455960 | 1.450267 | 1.489556 |
| dirichlet | 1.455960 | 1.450267 | 1.489556 |
| hier_dirichlet | 1.455960 | 1.450267 | 1.489556 |
| cond_logit | 1.453263 | 1.446980 | 1.494869 |
| lgbm | 1.452056 | 1.442796 | 1.497750 |

(n transfers = 74,068)

Per-role "too narrow" statistic for the proportional arm:

| role | players | sim SD | real SD | ratio |
|---|---:|---:|---:|---:|
| handler | 1187 | 1.94612 | 1.88577 | 1.032 |
| wing | 2613 | 1.40722 | 1.2644 | 1.11296 |
| big | 219 | 0.64271 | 0.5469 | 1.17519 |

### 3.4 TOV

Train 108,913 events, test 119,135. Fitted shrinkage: prior `league`, m = 200 pseudo on-floor events. Uniform-over-five log loss = 1.609438.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| proportional | 1.582428 | 0.7890 | 0.2653 | 0.6813 | 0.702 | 4/4 | 0.000777 | 1.0473 | 5.856 / 5.825 | 32.63% / 32.96% | 70.78% / 71.17% | yes |
| dirichlet | 1.582428 | 0.7890 | 0.2652 | 0.6811 | 0.702 | 4/4 | 0.000777 | 1.0473 | 5.856 / 5.825 | 32.63% / 32.96% | 70.78% / 71.17% | yes |
| hier_dirichlet | 1.582482 | 0.7890 | 0.2653 | 0.6808 | 0.698 | 4/4 | 0.000775 | 1.0494 | 5.851 / 5.825 | 32.70% / 32.96% | 70.84% / 71.17% | yes |
| cond_logit | 1.580645 | 0.7882 | 0.2677 | 0.6827 | 0.316 | 4/4 | 0.000716 | 1.0431 | 5.842 / 5.825 | 32.65% / 32.96% | 70.87% / 71.17% | yes |
| lgbm | 1.577664 | 0.7869 | 0.2720 | 0.6863 | 0.274 | 4/4 | 0.000791 | 1.0406 | 5.817 / 5.825 | 32.90% / 32.96% | 71.17% / 71.17% | yes |

**Decision: lgbm.** eligible: lgbm 1.577664, cond_logit 1.580645, proportional 1.582428, dirichlet 1.582428, hier_dirichlet 1.582482; floor 0.000791; clear of the next eligible arm by 0.002981 (3.8 floors)

Fitted concentrations -- U2 `{'a_between': None, 'a_handler': None, 'a_wing': None, 'a_big': None}`; U3 `{'a_between': None, 'a_handler': 256.0, 'a_wing': None, 'a_big': 64.0}` (`null` = no dispersion at that level, the top rung of the grid). Polya-urn (within-game, NOT a pregame quantity, never a decision input) log loss: U2 1.582428, U3 1.582452.

U4 ridge penalty 0.1; dropped as unidentified in this fold: `['log_prior_share', 'prior_rate']`; dropped as constant: `['is_transfer']`. Coefficients: `{'log_share': 0.99461, 'log_exposure': 0.02705, 'log_minutes': 0.04233, 'is_G': 0.05462, 'is_F': 0.03061, 'is_C': -0.0924, 'log_share_x_scorediff': -0.01013, 'log_share_x_sec': -0.03235, 'log_share_x_chance': -0.25313, 'is_C_x_scorediff': 0.01865, 'is_C_x_sec': 0.08882}`.

U5 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200} (searched on 2024 only); seed-varied refits [1.577664, 1.577609, 1.577614], seed SD 3e-05.

Transfer subset (2025 credited players whose modal team changed):

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.561141 | 1.591375 | 1.593073 |
| dirichlet | 1.561141 | 1.591375 | 1.593073 |
| hier_dirichlet | 1.561302 | 1.591396 | 1.593055 |
| cond_logit | 1.557562 | 1.584957 | 1.601074 |
| lgbm | 1.553612 | 1.577331 | 1.606902 |

(n transfers = 36,998)

Per-role "too narrow" statistic for the proportional arm:

| role | players | sim SD | real SD | ratio |
|---|---:|---:|---:|---:|
| handler | 1180 | 1.26668 | 1.26484 | 1.00145 |
| wing | 2512 | 0.98952 | 0.9193 | 1.07638 |
| big | 217 | 0.97241 | 0.9223 | 1.05434 |

### 3.5 FT_trip

Train 100,464 events, test 109,888. Fitted shrinkage: prior `position`, m = 200 pseudo on-floor events. Uniform-over-five log loss = 1.609438.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| proportional | 1.540766 | 0.7718 | 0.3082 | 0.7312 | 1.775 | 4/4 | 0.001169 | 1.0183 | 5.372 / 5.092 | 35.36% / 37.65% | 74.53% / 77.47% | NO (top1_top3) |
| dirichlet | 1.540766 | 0.7718 | 0.3081 | 0.7312 | 1.775 | 4/4 | 0.001169 | 1.0183 | 5.372 / 5.092 | 35.36% / 37.65% | 74.53% / 77.47% | NO (top1_top3) |
| hier_dirichlet | 1.541230 | 0.7720 | 0.3083 | 0.7296 | 1.845 | 4/4 | 0.001162 | 1.0438 | 5.307 / 5.092 | 36.20% / 37.65% | 75.29% / 77.47% | NO (top1_top3) |
| cond_logit | 1.536445 | 0.7702 | 0.3113 | 0.7344 | 0.304 | 4/4 | 0.001290 | 1.0102 | 5.293 / 5.092 | 36.09% / 37.65% | 75.38% / 77.47% | NO (top1_top3) |
| lgbm | 1.522266 | 0.7640 | 0.3243 | 0.7439 | 0.384 | 4/4 | 0.001411 | 1.0063 | 5.277 / 5.092 | 36.19% / 37.65% | 75.54% / 77.47% | yes |

**Decision: lgbm.** eligible: lgbm 1.522266; floor 0.001411; the only eligible arm

Fitted concentrations -- U2 `{'a_between': None, 'a_handler': None, 'a_wing': None, 'a_big': None}`; U3 `{'a_between': None, 'a_handler': 16.0, 'a_wing': None, 'a_big': None}` (`null` = no dispersion at that level, the top rung of the grid). Polya-urn (within-game, NOT a pregame quantity, never a decision input) log loss: U2 1.540766, U3 1.54149.

U4 ridge penalty 10.0; dropped as unidentified in this fold: `['log_prior_share', 'prior_rate']`; dropped as constant: `['is_transfer']`. Coefficients: `{'log_share': 0.91443, 'log_exposure': 0.04198, 'log_minutes': 0.08246, 'is_G': 0.1291, 'is_F': 0.17832, 'is_C': 0.05251, 'log_share_x_scorediff': -0.03616, 'log_share_x_sec': 0.09323, 'log_share_x_chance': 0.10642, 'is_C_x_scorediff': 0.04062, 'is_C_x_sec': 0.08581}`.

U5 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200} (searched on 2024 only); seed-varied refits [1.522266, 1.522452, 1.522406], seed SD 9.7e-05.

Transfer subset (2025 credited players whose modal team changed):

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.528728 | 1.534386 | 1.567731 |
| dirichlet | 1.528728 | 1.534386 | 1.567731 |
| hier_dirichlet | 1.529598 | 1.534817 | 1.567735 |
| cond_logit | 1.519178 | 1.522097 | 1.584521 |
| lgbm | 1.502224 | 1.505483 | 1.578300 |

(n transfers = 34,438)

Per-role "too narrow" statistic for the proportional arm:

| role | players | sim SD | real SD | ratio |
|---|---:|---:|---:|---:|
| handler | 1169 | 1.27159 | 1.34202 | 0.94752 |
| wing | 2482 | 0.9636 | 0.90137 | 1.06903 |
| big | 209 | 1.03028 | 1.02301 | 1.00711 |

## 4. Robustness fold: within-2025 walk-forward (train before 2025-01-15, test after)

| class | arm | log loss | boot SE | calib (pp) | resp | SD ratio | players >=1 delta | eligible |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FGA_rim | proportional | 1.510512 | 0.001562 | 0.650 | 4/4 | 0.9919 | 0.196 | yes |
| FGA_rim | dirichlet | 1.510512 | 0.001562 | 0.650 | 4/4 | 0.9919 | 0.196 | yes |
| FGA_rim | hier_dirichlet | 1.510602 | 0.001561 | 0.727 | 4/4 | 1.0091 | 0.174 | yes |
| FGA_rim | cond_logit | 1.510345 | 0.001541 | 1.124 | 4/4 | 0.9894 | 0.167 | yes |
| FGA_rim | lgbm | 1.495627 | 0.001751 | 0.869 | 4/4 | 0.9850 | 0.043 | yes |
| FGA_jump2 | proportional | 1.477853 | 0.001893 | 1.505 | 4/4 | 1.0239 | 0.260 | no |
| FGA_jump2 | dirichlet | 1.477853 | 0.001893 | 1.505 | 4/4 | 1.0239 | 0.260 | no |
| FGA_jump2 | hier_dirichlet | 1.478457 | 0.001884 | 1.623 | 4/4 | 1.0629 | 0.200 | yes |
| FGA_jump2 | cond_logit | 1.473370 | 0.002078 | 0.509 | 4/4 | 1.0143 | 0.127 | yes |
| FGA_jump2 | lgbm | 1.474558 | 0.002249 | 0.809 | 4/4 | 1.0111 | 0.063 | yes |
| FGA_3 | proportional | 1.445956 | 0.001647 | 0.402 | 4/4 | 1.0557 | 0.090 | yes |
| FGA_3 | dirichlet | 1.445956 | 0.001647 | 0.402 | 4/4 | 1.0557 | 0.090 | yes |
| FGA_3 | hier_dirichlet | 1.445954 | 0.001647 | 0.402 | 4/4 | 1.0557 | 0.089 | yes |
| FGA_3 | cond_logit | 1.446399 | 0.001637 | 0.859 | 4/4 | 1.0536 | 0.080 | yes |
| FGA_3 | lgbm | 1.445341 | 0.001687 | 0.467 | 4/4 | 1.0478 | 0.021 | yes |
| TOV | proportional | 1.574327 | 0.001257 | 0.667 | 4/4 | 1.0374 | 0.010 | yes |
| TOV | dirichlet | 1.574327 | 0.001257 | 0.667 | 4/4 | 1.0374 | 0.010 | yes |
| TOV | hier_dirichlet | 1.574372 | 0.001255 | 0.662 | 4/4 | 1.0396 | 0.005 | yes |
| TOV | cond_logit | 1.572825 | 0.001148 | 0.963 | 4/4 | 1.0375 | 0.002 | yes |
| TOV | lgbm | 1.585052 | 0.001531 | 1.537 | 4/4 | 1.0340 | -0.115 | yes |
| FT_trip | proportional | 1.527189 | 0.001753 | 1.986 | 4/4 | 1.0000 | 0.231 | no |
| FT_trip | dirichlet | 1.527189 | 0.001753 | 1.986 | 4/4 | 1.0000 | 0.231 | no |
| FT_trip | hier_dirichlet | 1.527528 | 0.001751 | 2.071 | 4/4 | 1.0260 | 0.173 | no |
| FT_trip | cond_logit | 1.522809 | 0.001966 | 0.633 | 4/4 | 0.9929 | 0.134 | yes |
| FT_trip | lgbm | 1.514025 | 0.002115 | 1.424 | 4/4 | 0.9914 | 0.126 | yes |

| class | winner | reason |
|---|---|---|
| FGA_rim | lgbm | eligible: lgbm 1.495627, cond_logit 1.510345, proportional 1.510512, dirichlet 1.510512, hier_dirichlet 1.510602; floor 0.001751; clear of the next eligible arm by 0.014718 (8.4 floors) |
| FGA_jump2 | cond_logit | eligible: cond_logit 1.473370, lgbm 1.474558, hier_dirichlet 1.478457; floor 0.002249; cond_logit, lgbm are inside the floor of each other; the pre-registered tie-break takes the simplest |
| FGA_3 | proportional | lgbm leads by 0.000613, inside the 0.001687 floor, so the pre-registration's requirement that the tree beat the best non-tree arm by more than the floor is not met; among the non-tree arms hier_dirichlet, proportional, dirichlet, cond_logit are inside the floor of each other and the tie-break takes the simplest |
| TOV | proportional | eligible: cond_logit 1.572825, proportional 1.574327, dirichlet 1.574327, hier_dirichlet 1.574372, lgbm 1.585052; floor 0.001531; cond_logit, proportional, dirichlet are inside the floor of each other; the pre-registered tie-break takes the simplest |
| FT_trip | lgbm | eligible: lgbm 1.514025, cond_logit 1.522809; floor 0.002115; clear of the next eligible arm by 0.008784 (4.2 floors) |

On this fold the prior-season block IS identified on both sides (2024 is a completed season of on-floor history), so the shrinkage grid can choose it and the logit/tree can use it. The fitted priors are in the per-class rows above.

| class | fitted prior | m | prior-season rung available |
|---|---|---:|---|
| FGA_rim | position | 50 | True |
| FGA_jump2 | position | 100 | True |
| FGA_3 | position | 25 | True |
| TOV | league | 200 | True |
| FT_trip | position | 200 | True |
