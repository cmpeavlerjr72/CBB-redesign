# L4 SHOT ALLOCATION (usage): experiments

Append-only. Status changes go to `docs/models/change_ledger.md` in the same commit
(`CLAUDE.md`, standing rule "bake-off before any choice").

---

## 1. Pre-registration (authored by the PM, 2026-09-10, BEFORE any modelling)

Target: for each chance with a known offensive on-floor five (2024+), the identity of the player credited with the terminal event, modelled per event class {FGA_rim, FGA_jump2, FGA_3, TOV, FT_trip (fouled shooter)} as a choice among the five. Universe: D-I, pbp_complete games with complete on-floor ids; F1 train 2024, test 2025 (selection); robustness: within-2025 walk-forward (train before Jan 15, test after). 2026 sealed (seal.assert_not_sealed). Players are keyed on the CBBD id as in free_throw.
Pregame inputs (as-of, strictly before game date, with shrinkage strength FITTED): each player's season-to-date share of his team's events by class while on the floor (per-possession-on-floor rates, not per game), prior-season rates for returning players, position group prior for new players, minutes-to-date; the five's as-of rates are normalised within the lineup.
Arms: (U1) proportional: probability proportional to the player's as-of per-possession rate for that class, normalised over the five (deterministic shares, no extra dispersion); (U2) Dirichlet-multinomial on the same shares with a fitted concentration (adds game-to-game usage variance; the CFB single-level design); (U3) hierarchical Dirichlet: a between-role draw (primary handlers / wings / bigs, roles from position group and as-of usage rank) then within-role, mean-preserving, per the CFB usage_alloc design, concentration fitted; (U4) conditional logit on the five with features (as-of class rate, prior-season rate, minutes-to-date, position, score diff, seconds remaining, chance number) fitted by ridge; (U5) LightGBM ranker over the five with the same features (parameter search on 2024 only).
Metrics on F1 test: per-class log loss of the credited player; calibration of predicted share vs actual share by player-as-of-rate decile (<= 2 pp); responsiveness by player as-of rate quintile (monotone 4/4); the CFB "too narrow / too short" pair at the game level: SD ratio of simulated per-player per-game event counts vs actual (target 0.9-1.1) and the number of players credited with >= 1 event per team-game (sim vs actual +/- 0.5); top-1 and top-3 usage share per team-game sim vs actual (+/- 2 pp); a transfer subset check; noise floor (seed-varied refits / bootstrap). Simulated counts for the game-level checks are produced by re-allocating the ACTUAL chance sequence with actual on-floor fives (so this test isolates allocation from rotation and from the event model).
Decision rules: per class, winner = lowest log loss among arms passing calibration, responsiveness, and the game-level dispersion checks; U5 must beat the best non-tree passing arm by more than the floor; ties to the simpler (U1 < U2 < U3 < U4 < U5). An arm that passes log loss but fails the SD-ratio check is not eligible (that is the CFB "too narrow" failure). If no arm is eligible for a class, adopt nothing and report.

---

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

---

## 5. Interpretation -- what the data said

### R1. CFB's "too narrow" defect does NOT reproduce in CBB

This is the headline, and it inverts the reason the CFB hierarchical allocator
exists. CFB measured per-player per-game count SD at 0.54-0.67x real for rushers
and 0.82-0.89x for receivers off a FIXED share table, and built the two-level
Dirichlet to fix it. In CBB the plain proportional allocator (U1), re-allocating
the actual event sequence over the actual on-floor fives, lands at:

| class | SD ratio, U1 | gate |
|---|---:|---|
| FGA_rim | 1.0108 | 0.9-1.1 PASS |
| FGA_jump2 | 1.0382 | PASS |
| FGA_3 | 1.0823 | PASS |
| TOV | 1.0473 | PASS |
| FT_trip | 1.0183 | PASS |

Every class passes, and every class errs on the WIDE side. Per role (the
per-class tables in section 3) the widest cell is `FGA_3` bigs at 1.175 and wings
at 1.113, and the narrowest is `FT_trip` handlers at 0.948 -- a band of
0.95-1.18, nowhere near CFB's 0.54.

The mechanism is structural, not luck. CFB allocated from one share vector per
team-game, so the only variance available was multinomial noise around a fixed
mean. Here the allocator is conditioned on the five actually on the floor, and
that five changes between games and within a game. Lineup variation already
supplies the game-to-game usage dispersion CFB had to inject with a Dirichlet.

### R2. CFB's "too short" defect does not reproduce either -- the tail is slightly too LONG

CFB's share table carried 5.24 rushers / 8.08 receivers with share > 0 against
11.07 / 15.71 real. Here, players credited with at least one event per team-game,
U1 sim vs real:

| class | sim | real | delta | gate +/- 0.5 |
|---|---:|---:|---:|---|
| FGA_rim | 7.090 | 6.825 | +0.265 | PASS |
| FGA_jump2 | 5.764 | 5.528 | +0.236 | PASS |
| FGA_3 | 6.820 | 6.689 | +0.131 | PASS |
| TOV | 5.856 | 5.825 | +0.031 | PASS |
| FT_trip | 5.372 | 5.092 | +0.280 | PASS |

Every class passes and every class is sim-HIGH. There is no missing tail to
extend, so CFB's `extend_profile` layer has no defect to fix here and is not
ported. The reason is the same as R1: the allocator cannot drop a low-usage
player, because the rotation model puts him on the floor and the five IS the
candidate set.

### R3. The gate that actually bites is top-1 / top-3 share, and it fails in the OPPOSITE direction

Both CFB diagnostics pass. The third check -- top-1 and top-3 usage share per
team-game, +/- 2 pp -- is what makes U1 ineligible on three of the five classes,
and it fails because the allocator is too FLAT at the top, not too peaked:

| class | arm | top-1 gap (pp) | top-3 gap (pp) | verdict |
|---|---|---:|---:|---|
| FGA_rim | proportional | -1.54 | **-2.18** | FAIL |
| FGA_rim | hier_dirichlet | -1.15 | -1.81 | pass |
| FGA_rim | cond_logit | -1.26 | -1.77 | pass |
| FGA_rim | lgbm | -1.15 | -1.61 | pass |
| FGA_jump2 | proportional | -1.81 | **-2.20** | FAIL |
| FGA_jump2 | lgbm | -1.23 | -1.40 | pass |
| FGA_3 | proportional | -0.25 | -0.60 | pass |
| FGA_3 | lgbm | -0.24 | -0.45 | pass |
| TOV | proportional | -0.32 | -0.40 | pass |
| TOV | lgbm | -0.05 | -0.01 | pass |
| FT_trip | proportional | **-2.29** | **-2.95** | FAIL |
| FT_trip | hier_dirichlet | -1.45 | **-2.18** | FAIL |
| FT_trip | cond_logit | -1.56 | **-2.10** | FAIL |
| FT_trip | lgbm | -1.46 | -1.94 | pass |

Every arm under-concentrates, on every class, on both statistics -- the sign is
negative in all 50 F1 cells (5 classes x 5 arms x 2 statistics). The two
classes where it is harmless (`FGA_3`,
`TOV`) are the two whose real top-3 share is lowest; the three where it bites are
the ones where usage is genuinely concentrated (rim attempts, mid-range attempts,
and who gets fouled). A shrunk as-of rate normalised over five is a smooth
function of history and cannot reproduce the "he is the man we run everything
through in this matchup" concentration. `FT_trip` is the extreme case: U1 is
2.95 pp short on top-3 and only the tree closes it enough to be eligible.

This is a single named defect, not a tuning problem, and it is recorded as such
in `docs/models/change_ledger.md` rather than patched with a sharpening exponent
-- which is exactly the post-hoc shape `docs/SIM_GUARDRAILS.md` section 5 bans.

### R4. Every fitted single-level concentration lands on "no dispersion"

U2's fitted concentration is the grid's top rung (`none` = no dispersion) on all
five classes, so U2 collapses onto U1 EXACTLY -- identical log loss to six
decimals on every class, which is the cleanest possible confirmation that the
Dirichlet is mean-preserving in the implementation as well as on paper. U3's
between-role concentration is also `none` on all five classes; only some
within-role knobs come back finite:

| class | a_between | a_handler | a_wing | a_big |
|---|---|---|---|---|
| FGA_rim | none | 64 | none | 2 |
| FGA_jump2 | none | 64 | none | none |
| FGA_3 | none | none | none | none |
| TOV | none | 256 | none | 64 |
| FT_trip | none | 16 | none | none |

Those finite knobs are fitted against a role SD ratio that is already inside
0.95-1.05 at the no-dispersion rung, so they buy a second-decimal improvement in
one moment at the cost of log loss (U3 is worse than U1 on four of five classes,
by 0.00005 to 0.0005) and, on `FGA_rim` and `FGA_jump2`, a slightly better top-3
share. `a_big = 2` on `FGA_rim` sits near the grid's low edge; the grid was
extended down to 1.0 for exactly that reason and the rung below 2 was not chosen.

### R5. Within-game usage concentration is worth nothing measurable

The Polya-urn predictive -- the same Dirichlet conditioned on the game's OWN
earlier events of that class, and therefore not a pregame quantity and never a
decision input -- improves log loss by 0.0000 to 0.0005 over the marginal on
every class, and is WORSE on `TOV` and `FT_trip`. Since the fitted concentrations
are already "no dispersion", this is the consistent answer from a second
direction: a team's within-game usage is not measurably more concentrated than
its pregame shares predict, once you condition on who is on the floor.

### R6. The task is intrinsically high-entropy, and that frames every gap below

Uniform-over-five log loss is 1.609438. The best arm per class lands at
1.459-1.578, i.e. 2-9% better than a coin flip among five. That is not a weak
model; it is close to what the entropy of a five-way choice with a realistic
share vector permits -- a 0.32 / 0.25 / 0.20 / 0.14 / 0.09 lineup has entropy
1.525 nats, and the classes whose real top-3 share is highest are the ones whose
best log loss is lowest. Top-1
accuracy is 0.27-0.34 against a 0.20 baseline. L15's 60-89% residual on every
player-game rate stat predicted exactly this, and it is why the arm spread is
narrow and the bootstrap floor decides so many cells.

### R7. The tree wins every class on the selection fold, and only two of five on the robustness fold

| class | F1 winner | F1 margin over the best other eligible arm | WF2025 winner | WF2025 tree margin | agree |
|---|---|---|---|---|---|
| FGA_rim | lgbm | +0.019606 (16.9 floors) | lgbm | +0.014718 (8.4 floors) | YES |
| FGA_jump2 | lgbm | +0.003369 (2.1 floors) | cond_logit | tree 0.001188 BEHIND, inside the floor | straddle |
| FGA_3 | lgbm | +0.001565 (1.2 floors) | proportional | tree +0.000615 ahead, inside the floor | straddle |
| TOV | lgbm | +0.002981 (3.8 floors) | proportional | tree 0.010725 BEHIND (7.0 floors) | REVERSES |
| FT_trip | lgbm | only eligible arm | lgbm | +0.008784 (4.2 floors) | YES |

Two classes replicate cleanly. Two straddle the floor. `TOV` reverses outright:
the tree's 3.8-floor win on F1 becomes a 7.0-floor LOSS on the within-2025 fold,
where it is also the only arm whose calibration (1.54 pp) approaches the gate and
the only one whose players-with-an-event count comes in LOW (-0.115). The
pre-registration names F1 as the selection fold and gives the robustness fold no
veto, so the F1 winners stand as the decisions -- but `TOV`'s is recorded as
UNCONFIRMED and `FGA_jump2`/`FGA_3`'s as floor-thin. The two folds are not
independent: the within-2025 fold tests the second half of the same season the F1
fold tests whole.

### R8. Fitted shrinkage: position on three classes, league on turnovers and mid-range

| class | F1 prior | F1 m (pseudo on-floor events) | WF2025 prior | WF2025 m | prior-season rung available on WF |
|---|---|---:|---|---:|---|
| FGA_rim | position | 50 | position | 50 | yes |
| FGA_jump2 | league | 50 | position | 100 | yes |
| FGA_3 | position | 25 | position | 25 | yes |
| TOV | league | 200 | league | 200 | yes |
| FT_trip | position | 200 | position | 200 | yes |

`m` is in pseudo ON-FLOOR EVENTS, the same unit as the denominator, and the
median 2025 event is taken by a player with roughly 400 on-floor events of
history -- so `m = 25` on three-point attempts means a player's own rate carries
~94% of the weight by mid-season, while `m = 200` on turnovers and free-throw
trips means it carries only about two thirds. That ordering is the signal
ordering: who takes the threes is strongly a player property, who commits the
turnover much less so (L15: usage proxy 34% player-beyond-both, FT rate 9.8%).

The `prior_season` rung is **unidentified on F1 by construction** (2024 has no
prior season of on-floor ids, L13) and is reported as such rather than scored. It
IS available on the within-2025 fold, where the grid still chose `position` on
four classes and `league` on one -- the prior-season rate never wins. That is the
same transfer-attenuation shape L15 measured and `free_throw` found independently
(its fitted prior is also `position`).

### R9. Transfer subset: read as composition, not as a transfer penalty

On F1 the credited player is a transfer on 34,438-74,068 events per class. Log
loss on transfers vs continuing players vs players with no prior season at all
(U1):

| class | transfers | continuing | no prior season |
|---|---:|---:|---:|
| FGA_rim | 1.500334 | 1.515145 | 1.568013 |
| FGA_jump2 | 1.488913 | 1.482348 | 1.541248 |
| FGA_3 | 1.455960 | 1.450267 | 1.489556 |
| TOV | 1.561141 | 1.591375 | 1.593073 |
| FT_trip | 1.528728 | 1.534386 | 1.567731 |

Transfers are predicted BETTER than continuing players on three classes and
marginally worse on two. **This is not evidence that transfers are easy.** No arm
on F1 uses the prior-season block at all (R8), so the split cannot be measuring
that prior's quality; what it measures is that transfers in this sample are
higher-usage players, whose within-lineup share is further from uniform and
therefore easier to call. The honest signal is the third column: players with NO
prior season are 0.03-0.07 nats harder on every class, 20-50x the bootstrap
floor. That is the cell a prior-season feature would have to improve, and the
cell to test once more seasons of on-floor data exist.

### R10. Two implementation defects were found by these diagnostics and fixed at source

Both are recorded because the diagnostics that caught them are as much the
deliverable as the arms are.

1. **One RNG uniform per GAME instead of per EVENT.** The first game-level
   measurement read 3.42 players with a three-point attempt per team-game against
   a real 6.69, a top-1 share of 60.88% against 32.33%, and an SD ratio of 2.46.
   Cause: the allocation uniform was keyed on `(seed, game_id)`, so every event of
   a team-game drew the same uniform and the whole game collapsed onto one
   player. Fix: `usage.event_stream_keys` folds the event's ordinal within its
   game into the key, keeping the per-game independence the RNG rule exists to
   give while making a game's own draws independent of each other.
   `tests/test_usage.py::test_c_event_stream_keys_are_distinct_within_a_game` and
   `::test_c_consecutive_draws_are_independent_not_one_uniform_reused` pin it.
2. **The tree arm's objective.** A binary LightGBM objective softmaxed within the
   five optimises a different likelihood than the one being graded, and came out
   systematically over-sharpened: on F1 `TOV`, top as-of-rate decile 30.13%
   predicted against 27.89% real and bottom decile 11.67% against 13.16%, a
   2.24 pp worst gap that FAILED the calibration gate on an arm whose log loss
   looked fine (1.579167). Replaced by the grouped-softmax objective
   (`usage._group_softmax_objective`, `grad = p - y`, `hess = p (1 - p)` within
   each group of five), offset by `log q` so the arm nests U1 at zero trees: the
   same class now reads a 0.27 pp worst gap and 1.577664 log loss, and trains
   three times faster. An objective fix, not a temperature fitted on the model's
   own output -- the latter is the shape `SIM_GUARDRAILS` section 5 bans.

A third, caught pre-emptively: leaving the prior-season block in on F1 -- where it
falls back to the position prior and therefore VARIES, so a constant-column check
does not catch it -- gave the `FGA_3` conditional logit a log loss of 3.74 against
U1's 1.46. `usage.unidentified_features` drops it and the drop is printed in
every per-class block of section 3.

### R11. Coverage

95.0-98.4% of credited events are modelled, per class and season (section 2.1).
The loss is 0.9-4.4% to an unresolved on-floor five (L13: 90% complete in 2024,
98% in 2025) plus, on turnovers only, 5.1-5.3% with no `participant_1_id` at all
-- CBBD leaves the charged player blank on team turnovers (shot-clock, ten-second)
and on steal-promoted turnovers. Roster position resolves on 99.98% of
player-games and hoopR minutes join through the player crosswalk on 99.81%, so
neither the position prior nor `minutes_asof` is a large unmeasured fallback here
(contrast FT-2, where position resolved on only 41.98% of 2022 attempts).

---

## 6. Decisions from data

| class | DECISION (F1, the selection fold) | confidence | why |
|---|---|---|---|
| FGA_rim | **lgbm** | replicated | 1.502348 against the best other eligible arm's 1.521954, 16.9 floors; replicates at 8.4 floors on the robustness fold. U1/U2 are INELIGIBLE (top-3 share -2.17 pp). |
| FGA_jump2 | **lgbm** | floor-thin, straddles | 1.491034 against 1.494403, 2.1 floors; on the robustness fold `cond_logit` leads it by 0.001188, inside that fold's 0.002249 floor. U1/U2 INELIGIBLE (top-3 -2.20 pp). |
| FGA_3 | **lgbm** | floor-thin, straddles | 1.458927 against 1.460492, 1.2 floors -- the narrowest margin in the bake-off; on the robustness fold the tree's 0.000615 lead is inside the floor and the rule falls through to `proportional`. All five arms are eligible on this class. |
| TOV | **lgbm, UNCONFIRMED** | reverses | 1.577664 against 1.582428, 3.8 floors on F1, but 7.0 floors BEHIND `proportional` on the robustness fold. The PM should treat this class as undecided pending a third fold. |
| FT_trip | **lgbm** | replicated | the ONLY eligible arm on F1 -- every other arm fails the top-3 share gate (U1 -2.95 pp, U3 -2.18, U4 -2.10) -- and it replicates at 4.2 floors on the robustness fold. |

Adopted alongside the winners, per class, as fitted parameters rather than
choices: the shrinkage prior and strength of R8, and the tree parameters
`num_leaves=15, learning_rate=0.08, n_estimators=300, min_child_samples=200`
(searched on 2024 only, and the same rung selected on every class).

**NOT adopted, with the reason measured:**

- **The CFB hierarchical Dirichlet (U3) is not adopted for any class.** Its
  motivating defect is absent (R1, R2): the fitted between-role concentration is
  "no dispersion" on all five classes, and the within-role knobs that do come back
  finite cost log loss. Conditioning the allocation on the on-floor five already
  supplies the dispersion a fixed share table lacks. The implementation is kept in
  `usage.py` and tested, because it is the right layer to reach for if a future
  rotation model turns out to under-disperse the lineups themselves -- which is
  the condition under which CFB's defect WOULD appear here.
- **The single-level Dirichlet (U2) is not adopted** for the same reason; at its
  fitted concentration it is identical to U1.
- **`extend_profile` (the CFB "too short" tail) is not ported.** There is no
  missing tail: the allocator is sim-HIGH on players-with-an-event on all five
  classes (R2).
- **No sharpening of the top-k share.** The named defect (R3) is left OPEN in the
  change ledger. Every arm under-concentrates the top of the usage distribution by
  0.5-2.9 pp on top-3; the fix is a model that can represent matchup-specific
  concentration, not an exponent on the share vector.

---

## 7. Decision 8 re-applied: the amended responsiveness gate (2026-09-10, no retraining)

Decision 8 in `ARCHITECTURE_DECISIONS.md` amends the responsiveness gate in every
bake-off and explicitly supersedes the steps-only wording of this model's own
pre-registration (section 1, "responsiveness by player as-of rate quintile
(monotone 4/4)"). The amended gate is:

(a) predicted-vs-actual **slope ratio across the driver's quintiles inside
[0.8, 1.2]**, AND
(b) monotone in at least 3 of 4 quintile steps, with the 4-of-4 requirement
dropped when the driver's **realised** quintile span is below 2 pp.

Every quantity the amendment needs was already computed and stored per arm
(`slope_ratio`, `span_actual`, `pred_monotone_steps` in the `responsiveness`
block of `results_v1.json`), so it is re-applied **mechanically, with no
retraining**, by `scripts/train_usage_v1.py --redecide`
(`reapply_responsiveness_gate`).

**Driver inventory.** This model has exactly ONE responsiveness driver, and it is
the one the pre-registration names: the player's own shrunk as-of rate for the
class being allocated. There is no team driver (the choice is normalised within
one team's lineup, so a team-level term cancels) and no defence driver (the
allocator carries no opponent feature at all -- `model.md` section 9 item 6 and
`features.md` section 5 record that omission). So "applies to every driver" is
satisfied by the single column below; nothing is unreported.

### 7.1 Slope ratio per class and per arm

Span columns are the REALISED quintile span of the credited-share driver, in
percentage points; `slope` is predicted span / realised span.

**F1 (train 2024, test 2025) -- the selection fold**

| class | arm | slope | predicted span (pp) | realised span (pp) | steps | slope in [0.8, 1.2] |
|---|---|---:|---:|---:|---:|---|
| FGA_rim | proportional | 1.0048 | 21.439 | 21.337 | 4/4 | PASS |
| FGA_rim | dirichlet | 1.0048 | 21.439 | 21.337 | 4/4 | PASS |
| FGA_rim | hier_dirichlet | 1.0002 | 21.341 | 21.337 | 4/4 | PASS |
| FGA_rim | cond_logit | 1.0233 | 21.835 | 21.337 | 4/4 | PASS |
| FGA_rim | **lgbm** | 1.0223 | 21.813 | 21.337 | 4/4 | PASS |
| FGA_jump2 | proportional | 0.9977 | 23.004 | 23.057 | 4/4 | PASS |
| FGA_jump2 | dirichlet | 0.9977 | 23.004 | 23.057 | 4/4 | PASS |
| FGA_jump2 | hier_dirichlet | 0.9960 | 22.964 | 23.057 | 4/4 | PASS |
| FGA_jump2 | cond_logit | 0.9856 | 22.725 | 23.057 | 4/4 | PASS |
| FGA_jump2 | **lgbm** | 0.9826 | 22.657 | 23.057 | 4/4 | PASS |
| FGA_3 | proportional | 1.0109 | 26.799 | 26.510 | 4/4 | PASS |
| FGA_3 | dirichlet | 1.0109 | 26.799 | 26.510 | 4/4 | PASS |
| FGA_3 | hier_dirichlet | 1.0109 | 26.799 | 26.510 | 4/4 | PASS |
| FGA_3 | cond_logit | 0.9980 | 26.458 | 26.510 | 4/4 | PASS |
| FGA_3 | **lgbm** | 0.9948 | 26.373 | 26.510 | 4/4 | PASS |
| TOV | proportional | 1.0731 | 12.824 | 11.950 | 4/4 | PASS |
| TOV | dirichlet | 1.0731 | 12.824 | 11.950 | 4/4 | PASS |
| TOV | hier_dirichlet | 1.0723 | 12.814 | 11.950 | 4/4 | PASS |
| TOV | cond_logit | 0.9878 | 11.804 | 11.950 | 4/4 | PASS |
| TOV | **lgbm** | 0.9866 | 11.790 | 11.950 | 4/4 | PASS |
| FT_trip | proportional | 0.9011 | 17.318 | 19.219 | 4/4 | PASS |
| FT_trip | dirichlet | 0.9011 | 17.318 | 19.219 | 4/4 | PASS |
| FT_trip | hier_dirichlet | 0.8955 | 17.210 | 19.219 | 4/4 | PASS (closest to the floor) |
| FT_trip | cond_logit | 0.9945 | 19.114 | 19.219 | 4/4 | PASS |
| FT_trip | **lgbm** | 0.9898 | 19.022 | 19.219 | 4/4 | PASS |

**WF2025 (within-season walk-forward) -- the robustness fold**

| class | arm | slope | predicted span (pp) | realised span (pp) | steps | slope in [0.8, 1.2] |
|---|---|---:|---:|---:|---:|---|
| FGA_rim | proportional | 0.9853 | 22.537 | 22.872 | 4/4 | PASS |
| FGA_rim | dirichlet | 0.9853 | 22.537 | 22.872 | 4/4 | PASS |
| FGA_rim | hier_dirichlet | 0.9819 | 22.458 | 22.872 | 4/4 | PASS |
| FGA_rim | cond_logit | 0.9588 | 21.930 | 22.872 | 4/4 | PASS |
| FGA_rim | **lgbm** | 1.0567 | 24.169 | 22.872 | 4/4 | PASS |
| FGA_jump2 | proportional | 0.9125 | 22.862 | 25.054 | 4/4 | PASS |
| FGA_jump2 | dirichlet | 0.9125 | 22.862 | 25.054 | 4/4 | PASS |
| FGA_jump2 | hier_dirichlet | 0.9058 | 22.694 | 25.054 | 4/4 | PASS |
| FGA_jump2 | **cond_logit** | 0.9829 | 24.626 | 25.054 | 4/4 | PASS |
| FGA_jump2 | lgbm | 1.0187 | 25.522 | 25.054 | 4/4 | PASS |
| FGA_3 | **proportional** | 0.9923 | 27.560 | 27.774 | 4/4 | PASS |
| FGA_3 | dirichlet | 0.9923 | 27.560 | 27.774 | 4/4 | PASS |
| FGA_3 | hier_dirichlet | 0.9923 | 27.560 | 27.774 | 4/4 | PASS |
| FGA_3 | cond_logit | 0.9627 | 26.737 | 27.774 | 4/4 | PASS |
| FGA_3 | lgbm | 0.9998 | 27.768 | 27.774 | 4/4 | PASS |
| TOV | **proportional** | 1.0232 | 14.005 | 13.687 | 4/4 | PASS |
| TOV | dirichlet | 1.0232 | 14.005 | 13.687 | 4/4 | PASS |
| TOV | hier_dirichlet | 1.0227 | 13.997 | 13.687 | 4/4 | PASS |
| TOV | cond_logit | 0.9026 | 12.354 | 13.687 | 4/4 | PASS |
| TOV | lgbm | 1.0209 | 13.973 | 13.687 | 4/4 | PASS |
| FT_trip | proportional | 0.9146 | 19.189 | 20.980 | 4/4 | PASS |
| FT_trip | dirichlet | 0.9146 | 19.189 | 20.980 | 4/4 | PASS |
| FT_trip | hier_dirichlet | 0.9091 | 19.073 | 20.980 | 4/4 | PASS |
| FT_trip | cond_logit | 1.0220 | 21.441 | 20.980 | 4/4 | PASS |
| FT_trip | **lgbm** | 0.9697 | 20.344 | 20.980 | 4/4 | PASS |

(Bold marks the arm adopted for that class-fold in section 6 / section 4.)

### 7.2 Effect on this model: none

- **All 50 arm-fold cells pass the amended gate.** Slope ratios span
  0.8955 to 1.0731 -- the whole range sits inside [0.8, 1.2], with the narrowest
  margin being `FT_trip` `hier_dirichlet` at 0.8955 (0.0955 clear of the floor)
  and `TOV` `proportional` at 1.0731 (0.1269 clear of the ceiling).
- **`reapply_responsiveness_gate` reports 0 verdict flips.** `resp_pass` is
  unchanged for every one of the 50 cells, so no arm's eligibility moves and no
  adopted winner's status changes on either fold. Re-running
  `--redecide` after the amendment left `report_v1.md` and
  `usage_params_v1.json` **byte-identical**.
- **The small-span escape clause never engages.** The smallest realised quintile
  span anywhere in the bake-off is 11.95 pp (`TOV` on F1), against the 2 pp
  threshold, so the stricter 4-of-4 step requirement applies everywhere -- and is
  met by every arm on every class on both folds. For contrast, the fg_make
  failure the amendment was written for had a 1.37 pp defence-driver span.
- **Nothing in this model resembled the failure the amendment targets.** The
  flattest arm here holds 89.6% of the realised spread; the fg_make team baseline
  held 0.86%. The reason is structural: every arm in this bake-off is anchored on
  the player's own as-of rate (U1 is that rate normalised over the five, U4
  carries `log_share` with a fitted coefficient of 0.91-1.05, and U5 is offset by
  `log q` so it nests U1 at zero trees), so none of them can sit at the league
  mean even if the fit adds nothing.
- Consequently `docs/models/change_ledger.md` is **not** amended: no status,
  winner or eligibility in section 6 changes. Decision 8's own entry already
  records "usage 4/4 with slopes near 1"; the table above is the measurement
  behind that claim.

### 7.3 The amendment is enforced in code, not only re-applied once

`usage.SLOPE_BAND`, `usage.RESP_MIN_STEPS_SMALL_SPAN` and `usage.SMALL_SPAN_PP`
now carry the amended gate, and `usage.share_responsiveness` returns
`slope_pass`, `steps_pass`, `steps_required`, `span_actual_pp` and the superseded
`steps_only_pass` alongside its verdict, so a future re-run of this bake-off
cannot silently revert to the steps-only wording and both readings stay on
record. Two tests pin it:
`tests/test_usage.py::test_d_the_responsiveness_gate_rejects_a_flat_arm` builds a
damped arm holding a tenth of the realised spread -- monotone in all four steps,
so it PASSES the superseded gate -- and requires the amended gate to fail it,
which is the fg_make failure reproduced in miniature; and
`::test_d_the_small_span_clause_relaxes_the_step_count` exercises the sub-2 pp
branch, which no real driver in this model reaches.

---

## 8. Round 2 (data fix): shooter label keyed on `shot_shooter_id`

### 8.1 Pre-registration (authored 2026-09-10, BEFORE the re-run; committed in the same commit as this text)

**Why there is a round 2.** Round 1 keyed the credited player of a field-goal
event on CBBD `participant_1_id`, through
`cbb_sim.models.event_stream.build_stream`. That column is not the shooter.
CBBD emits a `participants` array and its ORDER IS NOT STABLE on a
two-participant row: measured over every 2022-2025 row of this model's own
universe, `participant_1_id == shot_shooter_id` on **51.02%** of assisted made
field goals, and on the 48.98% that disagree `participant_1_id` equals
`shot_assisted_by_id` -- the ASSISTER -- on **100.000%** of rows (and
`participant_2_id` equals `shot_shooter_id` on 100.000%). Inside this model's
window that mislabels **11.9% / 5.0% / 14.1%** of `FGA_rim` / `FGA_jump2` /
`FGA_3` rows in 2024 and **12.0% / 4.9% / 14.1%** in 2025. The rate differs by a
factor of ~3 between the classes and by 5.3-22.3% between teams, so it is a
differential bias across exactly the classes and teams the allocator exists to
distinguish, not symmetric noise. Round 1's coverage table did not catch it
because the assister is a teammate on the floor and passes the `in_five` filter
on 95-99% of the mislabelled rows. Full evidence, by season, event type and
team: `docs/tests/shooter_key_audit_2026-09-10.md`.

**THE ONLY CHANGE IS THE LABEL.** Everything else -- target, universe, arms,
features, shrinkage grid, priors, folds, metric battery, gates, decision rule,
tie-break, seeds and Monte-Carlo draw counts -- is unchanged and is IMPORTED
from `scripts/train_usage_v1.py` by `scripts/train_usage_v2.py` rather than
re-typed, so "identical apart from the label" is enforced by construction.
Restated in full so this section stands alone:

- **Target.** For each chance with a known offensive on-floor five (2024+), the
  identity of the player credited with the terminal event, per class
  {`FGA_rim`, `FGA_jump2`, `FGA_3`, `TOV`, `FT_trip`}, as a choice among the
  five. **The three FGA classes now key the shooter on `shot_shooter_id`.**
  `TOV` and `FT_trip` keep `participant_1_id` and are UNCHANGED by construction:
  FTA rows agree with `shot_shooter_id` on 100.000% of rows in all four seasons,
  and TOV rows carry no `shot_shooter_id` at all (0.000% populated), so
  `participant_1_id` is the only and the correct key for both.
- **Missing shooters are DROPPED, never imputed and never fallen back to
  `participant_1_id`** (a fallback would reinstate the assister on exactly the
  rows the fix removes). Expected loss 0.04-0.27% of FGA rows. The drop rate is
  reported by season and by team.
- **Universe.** D-I, non-truncated, `pbp_complete`; possessions `v2` (rim
  override 2.27 ft). Identical to round 1.
- **Folds.** F1 trains 2024 and tests 2025 and is the SELECTION fold. Robustness
  fold: within-2025 walk-forward, train before 2025-01-15, test after. 2026
  sealed (`seal.assert_not_sealed` on both slices). This model has no earlier
  fold because on-floor ids do not exist before 2024 (L13); the project-standard
  fold 1 (train through 2022-23, test 2023-24) and fold 2 (train through
  2023-24, test 2024-25) map onto this model as "no fold 1" and "F1"
  respectively, so **F1 is the fold-2 selection metric** and the within-season
  walk-forward is the extra robustness check, exactly as in round 1.
- **Arms.** U1 proportional, U2 Dirichlet, U3 hierarchical Dirichlet, U4
  conditional logit (ridge), U5 LightGBM grouped-softmax choice arm. Same
  parameter grids; the tree's parameters are searched on the 2024 training slice
  only.
- **Primary metric.** Per-class 5-way log loss of the credited player on the
  fold's test slice.
- **Gates (all pre-registered, all unchanged).** Calibration of predicted vs
  actual share by player as-of-rate decile <= 2.0 pp; responsiveness under
  Decision 8 (slope ratio in [0.8, 1.2] AND monotone in >= 4 of 4 quintile
  steps, relaxing to 3 of 4 when the realised span is under 2 pp); game-level SD
  ratio of per-player per-game counts in [0.9, 1.1]; players with >= 1 event per
  team-game within +/- 0.5; top-1 and top-3 team usage share within +/- 2.0 pp.
- **Segment breakdowns reported.** Per fold, per class, per arm; per-team log
  loss distribution (teams with >= 200 test events of the class; teams below
  that are counted as underpowered and excluded, never presented as signal); the
  per-player-quintile responsiveness table (predicted vs actual share in each
  quintile of the player's own shrunk as-of rate); transfer / continuing /
  no-prior-season subsets; drop rate by season and team.
- **Noise floor.** Game-level block-bootstrap SE of the log loss (200
  replicates, seed 12345) per arm -- the floor the decision rule uses is the
  largest over the eligible arms -- plus spec-identical LightGBM retrains under
  seeds (0, 1, 2) with their SD reported. A winner must beat the next arm by
  more than the floor.
- **Decision rule.** Per class, the winner is the lowest log loss among the arms
  passing every gate; U5 must additionally beat the best non-tree eligible arm
  by more than the floor; ties (differences within the floor) go to the SIMPLER
  arm in the order U1 < U2 < U3 < U4 < U5. If no arm is eligible, adopt nothing
  and report.

### 8.2 What would count as the fix mattering

Stated in advance so the answer cannot be chosen after the fact:

1. **A changed winner on F1 for any FGA class.** Round 1 adopted `lgbm` on all
   five classes (section 6). Any class whose round-2 F1 winner is not `lgbm` is a
   decision the data fix reversed.
2. **A changed ELIGIBILITY verdict.** Round 1's U1/U2 were ineligible on
   `FGA_rim`, `FGA_jump2` and `FT_trip` for under-concentrating top-3 share. If
   cleaning the label moves the top-k gate, that is a substantive change even
   where the winner's name does not move.
3. **A materially different margin over the floor** on the two classes round 1
   flagged as floor-thin and straddling (`FGA_jump2` at 2.1 floors, `FGA_3` at
   1.2 floors) or as reversing (`TOV`).

Log-loss LEVELS are NOT comparable between the rounds: the two rounds score
different labels on nearly the same rows, so a level change is expected and is
not evidence of anything. Only winners, gate verdicts and margins-in-floors are
compared.

### 8.3 Pre-committed handling of the two unaffected classes

`TOV` and `FT_trip` are re-run because the trainer runs all five classes and
because their as-of exposure denominators include the FGA classes' events (a
player's `exposure_asof` counts every credited event of any class while he was
on the floor, so a relabelled FGA row does move his denominator). Their labels
are untouched. If either class's winner changes, the change must be attributed
to the exposure denominator or to Monte-Carlo noise and reported as such, and
the size of the log-loss move must be compared to the floor before anything is
claimed.

Trainer: `scripts/train_usage_v2.py`. Artifacts:
`data/processed/models/usage_v2/` (a versioned sibling; nothing under
`data/processed/models/usage/` is written or moved, because the engine worker
reads it concurrently). Results are appended below as sections 8.4-8.9.
