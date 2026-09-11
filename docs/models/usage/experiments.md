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


### 8.4 Run configuration (run 2026-09-10 by `scripts/train_usage_v2.py`, after the section 8.1-8.3 pre-registration was committed)

| item | value |
|---|---|
| trainer | `scripts/train_usage_v2.py` |
| shooter key | `shot_shooter_id` (round 1: `participant_1_id`) |
| possessions version | `v2` (rim override 2.27 ft) |
| universe | D-I, non-truncated, `pbp_complete` (identical to round 1) |
| credited events (modelled) | 1,633,164 (round 1: 1,634,792) |
| player-games with as-of inputs | 204,630 |
| roster position known | 99.978% |
| hoopR minutes joined through the player crosswalk | 99.8148% |
| player-games with a prior season of on-floor history | 2024: 0.0%, 2025: 67.7998% |
| Monte-Carlo draws (marginal / Polya / game-level / alpha fit) | 200 / 100 / 40 / 15 |
| seed | 20260910 |

### 8.5 What the label fix moved, and what it dropped

| season | FGA rows relabelled | FGA rows with no `shot_shooter_id` |
|---|---:|---:|
| 2024 | 66,120 | 605 |
| 2025 | 70,468 | 531 |

Drop rate by season and class (rows with no credited player id; NEVER imputed), with the per-team distribution over teams with >= 50 rows of the class:

| season | class | rows | dropped | drop % | per-team min / median / p95 / max % | teams > 1% | underpowered teams |
|---|---|---:|---:|---:|---|---:|---:|
| 2024 | FGA_rim | 225,021 | 107 | 0.0476 | 0.0 / 0.0 / 0.1583 / 4.4925 | 5 | 0 |
| 2024 | FGA_jump2 | 153,702 | 415 | 0.27 | 0.0 / 0.1773 / 1.0447 / 3.2967 | 19 | 0 |
| 2024 | FGA_3 | 225,731 | 83 | 0.0368 | 0.0 / 0.0 / 0.0 / 7.2052 | 3 | 0 |
| 2024 | TOV | 121,637 | 6,154 | 5.0593 | 1.2232 / 4.911 / 8.3333 / 11.5385 | 362 | 0 |
| 2024 | FT_trip | 107,004 | 0 | 0.0 | 0.0 / 0.0 / 0.0 / 0.0 | 0 | 0 |
| 2025 | FGA_rim | 235,458 | 110 | 0.0467 | 0.0 / 0.0 / 0.0 / 6.903 | 4 | 0 |
| 2025 | FGA_jump2 | 149,087 | 314 | 0.2106 | 0.0 / 0.0 / 1.0018 / 1.9444 | 19 | 0 |
| 2025 | FGA_3 | 246,890 | 107 | 0.0433 | 0.0 / 0.0 / 0.0 / 6.6574 | 4 | 0 |
| 2025 | TOV | 128,428 | 6,854 | 5.3368 | 1.1628 / 5.1561 / 8.8513 / 14.6707 | 364 | 0 |
| 2025 | FT_trip | 112,715 | 2 | 0.0018 | 0.0 / 0.0 / 0.0 / 0.3509 | 0 | 0 |

### 8.6 Event coverage (reported, not silently filtered)

| season | class | events | five resolved | credited id present | modelled |
|---|---|---:|---:|---:|---:|
| 2024 | FGA_rim | 225,021 | 95.8817% | 99.9524% | 94.9249% |
| 2024 | FGA_jump2 | 153,702 | 96.2642% | 99.73% | 95.1393% |
| 2024 | FGA_3 | 225,731 | 95.9886% | 99.9632% | 95.033% |
| 2024 | TOV | 121,637 | 95.6239% | 94.9407% | 89.5394% |
| 2024 | FT_trip | 107,004 | 95.617% | 100.0% | 93.8881% |
| 2025 | FGA_rim | 235,458 | 99.1417% | 99.9533% | 98.2167% |
| 2025 | FGA_jump2 | 149,087 | 99.0925% | 99.7894% | 98.2044% |
| 2025 | FGA_3 | 246,890 | 99.1871% | 99.9567% | 98.3207% |
| 2025 | TOV | 128,428 | 99.1116% | 94.6632% | 92.764% |
| 2025 | FT_trip | 112,715 | 99.1048% | 99.9982% | 97.4919% |

### 8.7 F1 results (train 2024, test 2025) -- the selection fold

#### FGA_rim

Train 213,601 events, test 231,259. Fitted shrinkage: prior `position`, m = 50 pseudo on-floor events. Uniform-over-five log loss = 1.609438.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope ratio | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| proportional | 1.517114 | 0.7633 | 0.3241 | 0.7528 | 0.358 | 4/4 | 1.0116 | 0.001094 | 1.0105 | 7.108 / 6.885 | 30.65% / 31.81% | 67.26% / 69.05% | yes |
| dirichlet | 1.517114 | 0.7633 | 0.3240 | 0.7528 | 0.358 | 4/4 | 1.0116 | 0.001094 | 1.0105 | 7.108 / 6.885 | 30.65% / 31.81% | 67.26% / 69.05% | yes |
| hier_dirichlet | 1.517255 | 0.7634 | 0.3240 | 0.7522 | 0.341 | 4/4 | 1.0081 | 0.001087 | 1.0263 | 7.083 / 6.885 | 30.98% / 31.81% | 67.59% / 69.05% | yes |
| cond_logit | 1.515365 | 0.7625 | 0.3247 | 0.7537 | 0.406 | 4/4 | 1.0249 | 0.001114 | 1.0061 | 7.077 / 6.885 | 30.86% / 31.81% | 67.54% / 69.05% | yes |
| lgbm | 1.502629 | 0.7567 | 0.3343 | 0.7598 | 0.473 | 4/4 | 1.0246 | 0.001156 | 1.0059 | 7.058 / 6.885 | 30.91% / 31.81% | 67.68% / 69.05% | yes |

**Decision: lgbm.** eligible: lgbm 1.502629, cond_logit 1.515365, proportional 1.517114, dirichlet 1.517114, hier_dirichlet 1.517255; floor 0.001156; clear of the next eligible arm by 0.012736 (11.0 floors)

Round 1 on this fold decided **lgbm**; round 2 decides **lgbm**. Per-arm log loss, round 1 -> round 2 (the LABELS differ between the rounds, so the LEVELS are not a like-for-like comparison and only the ordering and the gate verdicts are):

| arm | round 1 | round 2 | delta |
|---|---:|---:|---:|
| proportional | 1.523130 | 1.517114 | -0.006016 |
| dirichlet | 1.523130 | 1.517114 | -0.006016 |
| hier_dirichlet | 1.523298 | 1.517255 | -0.006043 |
| cond_logit | 1.521954 | 1.515365 | -0.006589 |
| lgbm | 1.502348 | 1.502629 | +0.000281 |

Per-team log loss (teams with >= 200 test events of this class):

| arm | teams | min | p10 | median | p90 | max | SD | underpowered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proportional | 364 | 1.2734 | 1.4420 | 1.5250 | 1.5747 | 1.6293 | 0.0537 | 0 |
| dirichlet | 364 | 1.2734 | 1.4420 | 1.5250 | 1.5747 | 1.6293 | 0.0537 | 0 |
| hier_dirichlet | 364 | 1.2741 | 1.4424 | 1.5249 | 1.5753 | 1.6292 | 0.0536 | 0 |
| cond_logit | 364 | 1.2730 | 1.4412 | 1.5257 | 1.5736 | 1.6275 | 0.0536 | 0 |
| lgbm | 364 | 1.2721 | 1.4287 | 1.5116 | 1.5604 | 1.6215 | 0.0536 | 0 |

Responsiveness by player as-of-rate quintile (the matchup-specific slope check; predicted vs actual credited share, %):

| arm | Q1 pred / act | Q2 | Q3 | Q4 | Q5 | span pred / act (pp) | slope ratio | steps | verdict |
|---|---|---|---|---|---|---|---:|---:|---|
| proportional | 9.48 / 9.67 | 15.29 / 15.35 | 19.21 / 19.27 | 24.06 / 23.82 | 31.95 / 31.89 | 22.47 / 22.21 | 1.0116 | 4/4 | PASS |
| dirichlet | 9.48 / 9.67 | 15.29 / 15.35 | 19.21 / 19.27 | 24.06 / 23.82 | 31.95 / 31.89 | 22.47 / 22.21 | 1.0116 | 4/4 | PASS |
| hier_dirichlet | 9.50 / 9.67 | 15.32 / 15.35 | 19.22 / 19.27 | 24.06 / 23.82 | 31.90 / 31.89 | 22.39 / 22.21 | 1.0081 | 4/4 | PASS |
| cond_logit | 9.45 / 9.67 | 15.16 / 15.35 | 19.09 / 19.27 | 24.08 / 23.82 | 32.22 / 31.89 | 22.77 / 22.21 | 1.0249 | 4/4 | PASS |
| lgbm | 9.42 / 9.67 | 15.03 / 15.35 | 19.12 / 19.27 | 24.24 / 23.82 | 32.18 / 31.89 | 22.76 / 22.21 | 1.0246 | 4/4 | PASS |

Noise floor. Block-bootstrap SE (the floor used by the decision rule) is the per-arm `boot SE` column above; the largest over the eligible arms is 0.001156. Spec-identical LightGBM retrains under seeds [0, 1, 2]: [1.502629, 1.502732, 1.502727], SD 5.8e-05.

Transfer subset (2025 credited players whose modal team changed):

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.500291 | 1.513076 | 1.545612 |
| dirichlet | 1.500291 | 1.513076 | 1.545612 |
| hier_dirichlet | 1.501031 | 1.513188 | 1.545054 |
| cond_logit | 1.496307 | 1.507826 | 1.553004 |
| lgbm | 1.482879 | 1.494769 | 1.541719 |

(n transfers = 71,738)

#### FGA_jump2

Train 146,231 events, test 146,410. Fitted shrinkage: prior `league`, m = 50 pseudo on-floor events. Uniform-over-five log loss = 1.609438.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope ratio | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| proportional | 1.495083 | 0.7530 | 0.3410 | 0.7663 | 0.597 | 4/4 | 0.9953 | 0.001654 | 1.0361 | 5.759 / 5.507 | 35.91% / 37.83% | 74.27% / 76.56% | NO (top1_top3) |
| dirichlet | 1.495083 | 0.7530 | 0.3410 | 0.7663 | 0.597 | 4/4 | 0.9953 | 0.001654 | 1.0361 | 5.759 / 5.507 | 35.91% / 37.83% | 74.27% / 76.56% | NO (top1_top3) |
| hier_dirichlet | 1.495169 | 0.7530 | 0.3410 | 0.7654 | 0.630 | 4/4 | 0.9936 | 0.001652 | 1.0462 | 5.742 / 5.507 | 36.18% / 37.83% | 74.49% / 76.56% | NO (top1_top3) |
| cond_logit | 1.491815 | 0.7517 | 0.3428 | 0.7683 | 0.656 | 4/4 | 0.9867 | 0.001648 | 1.0270 | 5.709 / 5.507 | 36.19% / 37.83% | 74.71% / 76.56% | yes |
| lgbm | 1.488712 | 0.7504 | 0.3450 | 0.7696 | 0.314 | 4/4 | 0.9887 | 0.001656 | 1.0226 | 5.674 / 5.507 | 36.50% / 37.83% | 75.08% / 76.56% | yes |

**Decision: lgbm.** eligible: lgbm 1.488712, cond_logit 1.491815; floor 0.001656; clear of the next eligible arm by 0.003103 (1.9 floors)

Round 1 on this fold decided **lgbm**; round 2 decides **lgbm**. Per-arm log loss, round 1 -> round 2 (the LABELS differ between the rounds, so the LEVELS are not a like-for-like comparison and only the ordering and the gate verdicts are):

| arm | round 1 | round 2 | delta |
|---|---:|---:|---:|
| proportional | 1.498369 | 1.495083 | -0.003286 |
| dirichlet | 1.498369 | 1.495083 | -0.003286 |
| hier_dirichlet | 1.498540 | 1.495169 | -0.003371 |
| cond_logit | 1.494403 | 1.491815 | -0.002588 |
| lgbm | 1.491034 | 1.488712 | -0.002322 |

Per-team log loss (teams with >= 200 test events of this class):

| arm | teams | min | p10 | median | p90 | max | SD | underpowered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proportional | 355 | 1.0105 | 1.4021 | 1.5082 | 1.5724 | 1.6151 | 0.0748 | 9 |
| dirichlet | 355 | 1.0105 | 1.4021 | 1.5082 | 1.5724 | 1.6151 | 0.0748 | 9 |
| hier_dirichlet | 355 | 1.0116 | 1.4033 | 1.5083 | 1.5723 | 1.6155 | 0.0748 | 9 |
| cond_logit | 355 | 1.0066 | 1.3995 | 1.5070 | 1.5679 | 1.6207 | 0.0753 | 9 |
| lgbm | 355 | 1.0095 | 1.3920 | 1.5018 | 1.5673 | 1.6177 | 0.0753 | 9 |

Responsiveness by player as-of-rate quintile (the matchup-specific slope check; predicted vs actual credited share, %):

| arm | Q1 pred / act | Q2 | Q3 | Q4 | Q5 | span pred / act (pp) | slope ratio | steps | verdict |
|---|---|---|---|---|---|---|---:|---:|---|
| proportional | 9.37 / 9.67 | 15.12 / 14.94 | 19.30 / 19.05 | 23.61 / 23.32 | 32.60 / 33.02 | 23.23 / 23.34 | 0.9953 | 4/4 | PASS |
| dirichlet | 9.37 / 9.67 | 15.12 / 14.94 | 19.30 / 19.05 | 23.61 / 23.32 | 32.60 / 33.02 | 23.23 / 23.34 | 0.9953 | 4/4 | PASS |
| hier_dirichlet | 9.39 / 9.67 | 15.13 / 14.94 | 19.30 / 19.05 | 23.60 / 23.32 | 32.58 / 33.02 | 23.19 / 23.34 | 0.9936 | 4/4 | PASS |
| cond_logit | 9.51 / 9.67 | 15.14 / 14.94 | 19.18 / 19.05 | 23.63 / 23.32 | 32.54 / 33.02 | 23.03 / 23.34 | 0.9867 | 4/4 | PASS |
| lgbm | 9.75 / 9.67 | 14.87 / 14.94 | 19.07 / 19.05 | 23.48 / 23.32 | 32.83 / 33.02 | 23.08 / 23.34 | 0.9887 | 4/4 | PASS |

Noise floor. Block-bootstrap SE (the floor used by the decision rule) is the per-arm `boot SE` column above; the largest over the eligible arms is 0.001656. Spec-identical LightGBM retrains under seeds [0, 1, 2]: [1.488712, 1.488765, 1.488702], SD 3.4e-05.

Transfer subset (2025 credited players whose modal team changed):

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.484754 | 1.479408 | 1.538110 |
| dirichlet | 1.484754 | 1.479408 | 1.538110 |
| hier_dirichlet | 1.485007 | 1.479507 | 1.537949 |
| cond_logit | 1.478163 | 1.469191 | 1.552258 |
| lgbm | 1.475134 | 1.460260 | 1.559990 |

(n transfers = 46,033)

#### FGA_3

Train 214,519 events, test 242,744. Fitted shrinkage: prior `position`, m = 25 pseudo on-floor events. Uniform-over-five log loss = 1.609438.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope ratio | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| proportional | 1.420255 | 0.7367 | 0.3448 | 0.8138 | 0.647 | 4/4 | 1.0015 | 0.001356 | 1.0868 | 6.654 / 6.490 | 32.66% / 32.87% | 70.99% / 71.72% | yes |
| dirichlet | 1.420255 | 0.7367 | 0.3448 | 0.8138 | 0.647 | 4/4 | 1.0015 | 0.001356 | 1.0868 | 6.654 / 6.490 | 32.66% / 32.87% | 70.99% / 71.72% | yes |
| hier_dirichlet | 1.420258 | 0.7367 | 0.3448 | 0.8139 | 0.647 | 4/4 | 1.0015 | 0.001356 | 1.0868 | 6.654 / 6.490 | 32.66% / 32.87% | 70.99% / 71.72% | yes |
| cond_logit | 1.419666 | 0.7366 | 0.3454 | 0.8141 | 1.007 | 4/4 | 1.0158 | 0.001370 | 1.0825 | 6.615 / 6.490 | 32.94% / 32.87% | 71.35% / 71.72% | yes |
| lgbm | 1.417636 | 0.7362 | 0.3455 | 0.8145 | 0.336 | 4/4 | 0.9954 | 0.001366 | 1.0766 | 6.582 / 6.490 | 32.67% / 32.87% | 71.21% / 71.72% | yes |

**Decision: lgbm.** eligible: lgbm 1.417636, cond_logit 1.419666, proportional 1.420255, dirichlet 1.420255, hier_dirichlet 1.420258; floor 0.001370; clear of the next eligible arm by 0.002030 (1.5 floors)

Round 1 on this fold decided **lgbm**; round 2 decides **lgbm**. Per-arm log loss, round 1 -> round 2 (the LABELS differ between the rounds, so the LEVELS are not a like-for-like comparison and only the ordering and the gate verdicts are):

| arm | round 1 | round 2 | delta |
|---|---:|---:|---:|
| proportional | 1.461517 | 1.420255 | -0.041262 |
| dirichlet | 1.461517 | 1.420255 | -0.041262 |
| hier_dirichlet | 1.461517 | 1.420258 | -0.041259 |
| cond_logit | 1.460492 | 1.419666 | -0.040826 |
| lgbm | 1.458927 | 1.417636 | -0.041291 |

Per-team log loss (teams with >= 200 test events of this class):

| arm | teams | min | p10 | median | p90 | max | SD | underpowered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proportional | 364 | 0.9797 | 1.3032 | 1.4215 | 1.5190 | 1.5838 | 0.0875 | 0 |
| dirichlet | 364 | 0.9797 | 1.3032 | 1.4215 | 1.5190 | 1.5838 | 0.0875 | 0 |
| hier_dirichlet | 364 | 0.9797 | 1.3032 | 1.4215 | 1.5190 | 1.5838 | 0.0875 | 0 |
| cond_logit | 364 | 0.9803 | 1.3025 | 1.4215 | 1.5183 | 1.5829 | 0.0878 | 0 |
| lgbm | 364 | 0.9762 | 1.2995 | 1.4183 | 1.5174 | 1.5831 | 0.0884 | 0 |

Responsiveness by player as-of-rate quintile (the matchup-specific slope check; predicted vs actual credited share, %):

| arm | Q1 pred / act | Q2 | Q3 | Q4 | Q5 | span pred / act (pp) | slope ratio | steps | verdict |
|---|---|---|---|---|---|---|---:|---:|---|
| proportional | 4.49 / 4.17 | 14.36 / 14.55 | 21.01 / 21.27 | 26.26 / 26.48 | 33.88 / 33.52 | 29.39 / 29.35 | 1.0015 | 4/4 | PASS |
| dirichlet | 4.49 / 4.17 | 14.36 / 14.55 | 21.01 / 21.27 | 26.26 / 26.48 | 33.88 / 33.52 | 29.39 / 29.35 | 1.0015 | 4/4 | PASS |
| hier_dirichlet | 4.49 / 4.17 | 14.36 / 14.55 | 21.01 / 21.27 | 26.26 / 26.48 | 33.88 / 33.52 | 29.39 / 29.35 | 1.0015 | 4/4 | PASS |
| cond_logit | 4.34 / 4.17 | 14.18 / 14.55 | 20.97 / 21.27 | 26.34 / 26.48 | 34.16 / 33.52 | 29.81 / 29.35 | 1.0158 | 4/4 | PASS |
| lgbm | 4.13 / 4.17 | 14.81 / 14.55 | 21.34 / 21.27 | 26.36 / 26.48 | 33.35 / 33.52 | 29.22 / 29.35 | 0.9954 | 4/4 | PASS |

Noise floor. Block-bootstrap SE (the floor used by the decision rule) is the per-arm `boot SE` column above; the largest over the eligible arms is 0.001370. Spec-identical LightGBM retrains under seeds [0, 1, 2]: [1.417636, 1.417666, 1.417662], SD 1.6e-05.

Transfer subset (2025 credited players whose modal team changed):

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.412416 | 1.414946 | 1.439386 |
| dirichlet | 1.412416 | 1.414946 | 1.439386 |
| hier_dirichlet | 1.412433 | 1.414944 | 1.439380 |
| cond_logit | 1.410347 | 1.412689 | 1.443609 |
| lgbm | 1.408200 | 1.409172 | 1.444408 |

(n transfers = 73,440)

#### TOV

Train 108,913 events, test 119,135. Fitted shrinkage: prior `league`, m = 200 pseudo on-floor events. Uniform-over-five log loss = 1.609438.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope ratio | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| proportional | 1.582423 | 0.7890 | 0.2653 | 0.6813 | 0.683 | 4/4 | 1.0720 | 0.000777 | 1.0474 | 5.856 / 5.825 | 32.63% / 32.96% | 70.77% / 71.17% | yes |
| dirichlet | 1.582423 | 0.7890 | 0.2649 | 0.6811 | 0.683 | 4/4 | 1.0720 | 0.000777 | 1.0474 | 5.856 / 5.825 | 32.63% / 32.96% | 70.77% / 71.17% | yes |
| hier_dirichlet | 1.582424 | 0.7890 | 0.2649 | 0.6810 | 0.682 | 4/4 | 1.0719 | 0.000776 | 1.0474 | 5.855 / 5.825 | 32.64% / 32.96% | 70.78% / 71.17% | yes |
| cond_logit | 1.580640 | 0.7882 | 0.2677 | 0.6827 | 0.285 | 4/4 | 0.9872 | 0.000716 | 1.0431 | 5.842 / 5.825 | 32.65% / 32.96% | 70.87% / 71.17% | yes |
| lgbm | 1.577494 | 0.7869 | 0.2726 | 0.6877 | 0.231 | 4/4 | 0.9881 | 0.000799 | 1.0407 | 5.816 / 5.825 | 32.91% / 32.96% | 71.18% / 71.17% | yes |

**Decision: lgbm.** eligible: lgbm 1.577494, cond_logit 1.580640, proportional 1.582423, dirichlet 1.582423, hier_dirichlet 1.582424; floor 0.000799; clear of the next eligible arm by 0.003146 (3.9 floors)

Round 1 on this fold decided **lgbm**; round 2 decides **lgbm**. Per-arm log loss, round 1 -> round 2 (the LABELS differ between the rounds, so the LEVELS are not a like-for-like comparison and only the ordering and the gate verdicts are):

| arm | round 1 | round 2 | delta |
|---|---:|---:|---:|
| proportional | 1.582428 | 1.582423 | -0.000005 |
| dirichlet | 1.582428 | 1.582423 | -0.000005 |
| hier_dirichlet | 1.582482 | 1.582424 | -0.000058 |
| cond_logit | 1.580645 | 1.580640 | -0.000005 |
| lgbm | 1.577664 | 1.577494 | -0.000170 |

Per-team log loss (teams with >= 200 test events of this class):

| arm | teams | min | p10 | median | p90 | max | SD | underpowered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proportional | 363 | 1.4584 | 1.5478 | 1.5864 | 1.6125 | 1.6315 | 0.0261 | 1 |
| dirichlet | 363 | 1.4584 | 1.5478 | 1.5864 | 1.6125 | 1.6315 | 0.0261 | 1 |
| hier_dirichlet | 363 | 1.4584 | 1.5478 | 1.5864 | 1.6125 | 1.6315 | 0.0261 | 1 |
| cond_logit | 363 | 1.4659 | 1.5472 | 1.5846 | 1.6102 | 1.6354 | 0.0255 | 1 |
| lgbm | 363 | 1.4684 | 1.5421 | 1.5817 | 1.6064 | 1.6321 | 0.0257 | 1 |

Responsiveness by player as-of-rate quintile (the matchup-specific slope check; predicted vs actual credited share, %):

| arm | Q1 pred / act | Q2 | Q3 | Q4 | Q5 | span pred / act (pp) | slope ratio | steps | verdict |
|---|---|---|---|---|---|---|---:|---:|---|
| proportional | 13.86 / 14.34 | 17.52 / 17.59 | 19.78 / 19.79 | 22.16 / 21.96 | 26.68 / 26.30 | 12.82 / 11.96 | 1.0720 | 4/4 | PASS |
| dirichlet | 13.86 / 14.34 | 17.52 / 17.59 | 19.78 / 19.79 | 22.16 / 21.96 | 26.68 / 26.30 | 12.82 / 11.96 | 1.0720 | 4/4 | PASS |
| hier_dirichlet | 13.86 / 14.34 | 17.52 / 17.59 | 19.78 / 19.79 | 22.16 / 21.96 | 26.68 / 26.30 | 12.82 / 11.96 | 1.0719 | 4/4 | PASS |
| cond_logit | 14.38 / 14.34 | 17.72 / 17.59 | 19.70 / 19.79 | 22.00 / 21.96 | 26.19 / 26.30 | 11.81 / 11.96 | 0.9872 | 4/4 | PASS |
| lgbm | 14.33 / 14.34 | 17.68 / 17.59 | 19.89 / 19.79 | 21.97 / 21.96 | 26.14 / 26.30 | 11.82 / 11.96 | 0.9881 | 4/4 | PASS |

Noise floor. Block-bootstrap SE (the floor used by the decision rule) is the per-arm `boot SE` column above; the largest over the eligible arms is 0.000799. Spec-identical LightGBM retrains under seeds [0, 1, 2]: [1.577494, 1.577474, 1.577485], SD 1e-05.

Transfer subset (2025 credited players whose modal team changed):

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.561118 | 1.591387 | 1.593062 |
| dirichlet | 1.561118 | 1.591387 | 1.593062 |
| hier_dirichlet | 1.561142 | 1.591382 | 1.593044 |
| cond_logit | 1.557538 | 1.584972 | 1.601059 |
| lgbm | 1.553332 | 1.577234 | 1.606744 |

(n transfers = 36,998)

#### FT_trip

Train 100,464 events, test 109,888. Fitted shrinkage: prior `position`, m = 200 pseudo on-floor events. Uniform-over-five log loss = 1.609438.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope ratio | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 (sim / real) | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| proportional | 1.540768 | 0.7718 | 0.3083 | 0.7313 | 1.801 | 4/4 | 0.9017 | 0.001169 | 1.0183 | 5.372 / 5.092 | 35.36% / 37.65% | 74.53% / 77.47% | NO (top1_top3) |
| dirichlet | 1.540768 | 0.7718 | 0.3082 | 0.7311 | 1.801 | 4/4 | 0.9017 | 0.001169 | 1.0183 | 5.372 / 5.092 | 35.36% / 37.65% | 74.53% / 77.47% | NO (top1_top3) |
| hier_dirichlet | 1.541404 | 0.7721 | 0.3079 | 0.7288 | 1.895 | 4/4 | 0.8949 | 0.001163 | 1.0443 | 5.307 / 5.092 | 36.20% / 37.65% | 75.29% / 77.47% | NO (top1_top3) |
| cond_logit | 1.536446 | 0.7702 | 0.3113 | 0.7344 | 0.322 | 4/4 | 0.9955 | 0.001289 | 1.0102 | 5.293 / 5.092 | 36.09% / 37.65% | 75.38% / 77.47% | NO (top1_top3) |
| lgbm | 1.523321 | 0.7644 | 0.3247 | 0.7433 | 0.382 | 4/4 | 0.9888 | 0.001435 | 1.0071 | 5.274 / 5.092 | 36.20% / 37.65% | 75.56% / 77.47% | yes |

**Decision: lgbm.** eligible: lgbm 1.523321; floor 0.001435; the only eligible arm

Round 1 on this fold decided **lgbm**; round 2 decides **lgbm**. Per-arm log loss, round 1 -> round 2 (the LABELS differ between the rounds, so the LEVELS are not a like-for-like comparison and only the ordering and the gate verdicts are):

| arm | round 1 | round 2 | delta |
|---|---:|---:|---:|
| proportional | 1.540766 | 1.540768 | +0.000002 |
| dirichlet | 1.540766 | 1.540768 | +0.000002 |
| hier_dirichlet | 1.541230 | 1.541404 | +0.000174 |
| cond_logit | 1.536445 | 1.536446 | +0.000001 |
| lgbm | 1.522266 | 1.523321 | +0.001055 |

Per-team log loss (teams with >= 200 test events of this class):

| arm | teams | min | p10 | median | p90 | max | SD | underpowered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proportional | 353 | 1.3422 | 1.4748 | 1.5484 | 1.5937 | 1.6283 | 0.0475 | 11 |
| dirichlet | 353 | 1.3422 | 1.4748 | 1.5484 | 1.5937 | 1.6283 | 0.0475 | 11 |
| hier_dirichlet | 353 | 1.3451 | 1.4749 | 1.5494 | 1.5942 | 1.6293 | 0.0472 | 11 |
| cond_logit | 353 | 1.3534 | 1.4678 | 1.5437 | 1.5937 | 1.6193 | 0.0499 | 11 |
| lgbm | 353 | 1.3433 | 1.4532 | 1.5319 | 1.5806 | 1.6252 | 0.0507 | 11 |

Responsiveness by player as-of-rate quintile (the matchup-specific slope check; predicted vs actual credited share, %):

| arm | Q1 pred / act | Q2 | Q3 | Q4 | Q5 | span pred / act (pp) | slope ratio | steps | verdict |
|---|---|---|---|---|---|---|---:|---:|---|
| proportional | 12.04 / 11.39 | 16.53 / 16.09 | 19.36 / 18.86 | 22.71 / 23.07 | 29.36 / 30.60 | 17.32 / 19.20 | 0.9017 | 4/4 | PASS |
| dirichlet | 12.04 / 11.39 | 16.53 / 16.09 | 19.36 / 18.86 | 22.71 / 23.07 | 29.36 / 30.60 | 17.32 / 19.20 | 0.9017 | 4/4 | PASS |
| hier_dirichlet | 12.09 / 11.39 | 16.56 / 16.09 | 19.36 / 18.86 | 22.71 / 23.07 | 29.28 / 30.60 | 17.18 / 19.20 | 0.8949 | 4/4 | PASS |
| cond_logit | 11.46 / 11.39 | 16.06 / 16.09 | 19.00 / 18.86 | 22.90 / 23.07 | 30.58 / 30.60 | 19.12 / 19.20 | 0.9955 | 4/4 | PASS |
| lgbm | 11.49 / 11.39 | 16.06 / 16.09 | 19.10 / 18.86 | 22.86 / 23.07 | 30.48 / 30.60 | 18.99 / 19.20 | 0.9888 | 4/4 | PASS |

Noise floor. Block-bootstrap SE (the floor used by the decision rule) is the per-arm `boot SE` column above; the largest over the eligible arms is 0.001435. Spec-identical LightGBM retrains under seeds [0, 1, 2]: [1.523321, 1.52344, 1.523438], SD 6.8e-05.

Transfer subset (2025 credited players whose modal team changed):

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.528706 | 1.534400 | 1.567744 |
| dirichlet | 1.528706 | 1.534400 | 1.567744 |
| hier_dirichlet | 1.529653 | 1.535108 | 1.567846 |
| cond_logit | 1.519146 | 1.522108 | 1.584541 |
| lgbm | 1.502847 | 1.506557 | 1.579875 |

(n transfers = 34,438)

### 8.8 Robustness fold: within-2025 walk-forward (train before 2025-01-15, test after)

| class | arm | log loss | boot SE | calib (pp) | resp | slope ratio | SD ratio | players >=1 delta | eligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| FGA_rim | proportional | 1.504910 | 0.001639 | 0.454 | 4/4 | 0.9877 | 0.9934 | 0.164 | yes |
| FGA_rim | dirichlet | 1.504910 | 0.001639 | 0.454 | 4/4 | 0.9877 | 0.9934 | 0.164 | yes |
| FGA_rim | hier_dirichlet | 1.505047 | 0.001632 | 0.587 | 4/4 | 0.9832 | 1.0059 | 0.142 | yes |
| FGA_rim | cond_logit | 1.504517 | 0.001607 | 1.041 | 4/4 | 0.9499 | 0.9929 | 0.167 | yes |
| FGA_rim | lgbm | 1.496060 | 0.001760 | 0.919 | 4/4 | 1.0310 | 0.9902 | 0.054 | yes |
| FGA_jump2 | proportional | 1.472566 | 0.002011 | 0.851 | 4/4 | 0.9602 | 1.0176 | 0.221 | no |
| FGA_jump2 | dirichlet | 1.472566 | 0.002011 | 0.851 | 4/4 | 0.9602 | 1.0176 | 0.221 | no |
| FGA_jump2 | hier_dirichlet | 1.472751 | 0.002003 | 0.914 | 4/4 | 0.9570 | 1.0386 | 0.187 | yes |
| FGA_jump2 | cond_logit | 1.471264 | 0.001961 | 1.753 | 4/4 | 0.9146 | 1.0160 | 0.203 | yes |
| FGA_jump2 | lgbm | 1.471357 | 0.002155 | 0.828 | 4/4 | 0.9585 | 1.0119 | 0.138 | yes |
| FGA_3 | proportional | 1.402224 | 0.001683 | 0.419 | 4/4 | 0.9938 | 1.0572 | 0.104 | yes |
| FGA_3 | dirichlet | 1.402224 | 0.001683 | 0.419 | 4/4 | 0.9938 | 1.0572 | 0.104 | yes |
| FGA_3 | hier_dirichlet | 1.402224 | 0.001683 | 0.419 | 4/4 | 0.9938 | 1.0572 | 0.104 | yes |
| FGA_3 | cond_logit | 1.403343 | 0.001717 | 0.389 | 4/4 | 0.9889 | 1.0527 | 0.072 | yes |
| FGA_3 | lgbm | 1.404042 | 0.001688 | 0.751 | 4/4 | 0.9661 | 1.0514 | 0.057 | yes |
| TOV | proportional | 1.574325 | 0.001257 | 0.659 | 4/4 | 1.0230 | 1.0374 | 0.010 | yes |
| TOV | dirichlet | 1.574325 | 0.001257 | 0.659 | 4/4 | 1.0230 | 1.0374 | 0.010 | yes |
| TOV | hier_dirichlet | 1.574305 | 0.001257 | 0.658 | 4/4 | 1.0228 | 1.0379 | 0.008 | yes |
| TOV | cond_logit | 1.572821 | 0.001148 | 0.994 | 4/4 | 0.9029 | 1.0375 | 0.002 | yes |
| TOV | lgbm | 1.578779 | 0.001440 | 1.506 | 4/4 | 0.9934 | 1.0338 | -0.078 | yes |
| FT_trip | proportional | 1.527195 | 0.001752 | 1.983 | 4/4 | 0.9163 | 1.0000 | 0.231 | no |
| FT_trip | dirichlet | 1.527195 | 0.001752 | 1.983 | 4/4 | 0.9163 | 1.0000 | 0.231 | no |
| FT_trip | hier_dirichlet | 1.528182 | 0.001726 | 2.170 | 4/4 | 0.9035 | 1.0502 | 0.117 | no |
| FT_trip | cond_logit | 1.522811 | 0.001967 | 0.690 | 4/4 | 1.0243 | 0.9929 | 0.134 | yes |
| FT_trip | lgbm | 1.516973 | 0.002210 | 0.856 | 4/4 | 0.9930 | 0.9902 | 0.088 | yes |

| class | winner | reason |
|---|---|---|
| FGA_rim | lgbm | eligible: lgbm 1.496060, cond_logit 1.504517, proportional 1.504910, dirichlet 1.504910, hier_dirichlet 1.505047; floor 0.001760; clear of the next eligible arm by 0.008457 (4.8 floors) |
| FGA_jump2 | hier_dirichlet | eligible: cond_logit 1.471264, lgbm 1.471357, hier_dirichlet 1.472751; floor 0.002155; cond_logit, lgbm, hier_dirichlet are inside the floor of each other; the pre-registered tie-break takes the simplest |
| FGA_3 | proportional | eligible: proportional 1.402224, dirichlet 1.402224, hier_dirichlet 1.402224, cond_logit 1.403343, lgbm 1.404042; floor 0.001717; proportional, dirichlet, hier_dirichlet, cond_logit are inside the floor of each other; the pre-registered tie-break takes the simplest |
| TOV | cond_logit | eligible: cond_logit 1.572821, hier_dirichlet 1.574305, proportional 1.574325, dirichlet 1.574325, lgbm 1.578779; floor 0.001440; clear of the next eligible arm by 0.001484 (1.0 floors) |
| FT_trip | lgbm | eligible: lgbm 1.516973, cond_logit 1.522811; floor 0.002210; clear of the next eligible arm by 0.005838 (2.6 floors) |

### 8.9 Round 2 decision, against round 1

| class | round 1 winner (F1) | round 2 winner (F1) | changed? | round 1 winner (WF) | round 2 winner (WF) | floor (F1) |
|---|---|---|---|---|---|---:|
| FGA_rim | lgbm | lgbm | no | lgbm | lgbm | 0.001156 |
| FGA_jump2 | lgbm | lgbm | no | cond_logit | hier_dirichlet | 0.001656 |
| FGA_3 | lgbm | lgbm | no | proportional | proportional | 0.001370 |
| TOV | lgbm | lgbm | no | proportional | cond_logit | 0.000799 |
| FT_trip | lgbm | lgbm | no | lgbm | lgbm | 0.001435 |


### 8.10 Interpretation -- what the data fix did and did not change

**R12. The winner does not change on the selection fold. All five classes stay
`lgbm`.** That is the headline and it is worth stating plainly: the round-1
adoption survives the label fix. The margins move a little in both directions --
`FGA_rim` 16.9 -> 11.0 floors, `FGA_jump2` 2.1 -> 1.9, `FGA_3` 1.2 -> 1.5,
`TOV` 3.8 -> 3.9, `FT_trip` the only eligible arm in both rounds -- and none of
those moves crosses the decision rule.

| class | R1 winner (F1) | R2 winner (F1) | changed? | R1 floors clear | R2 floors clear |
|---|---|---|---|---:|---:|
| FGA_rim | lgbm | lgbm | no | 16.9 | 11.0 |
| FGA_jump2 | lgbm | lgbm | no | 2.1 | 1.9 |
| FGA_3 | lgbm | lgbm | no | 1.2 | 1.5 |
| TOV | lgbm | lgbm | no | 3.8 | 3.9 |
| FT_trip | lgbm | lgbm | no | only eligible arm | only eligible arm |

On the robustness fold two tie-breaks shuffle inside the floor -- `FGA_jump2`
`cond_logit` -> `hier_dirichlet` (the three leaders are within 0.0015 of each
other against a 0.0022 floor) and `TOV` `proportional` -> `cond_logit` (1.0
floors). Neither is a fact about the label; both are the tie-break landing on a
different member of a set the rule already declared indistinguishable. `FGA_rim`
and `FT_trip` replicate `lgbm` on the robustness fold in both rounds, and
`FGA_3` falls through to `proportional` in both.

**R13. The ELIGIBILITY verdict does change on `FGA_rim`, and it changes in the
direction the defect predicts.** Round 1 ruled U1 and U2 ineligible on `FGA_rim`
for under-concentrating top-3 usage share by 2.17 pp against a 2.0 pp gate.
Cleaning the label moves that gap to **-1.79 pp** and both arms become eligible.

| class | arm | top-3 gap R1 (pp) | top-3 gap R2 (pp) | top-1 gap R1 | top-1 gap R2 |
|---|---|---:|---:|---:|---:|
| FGA_rim | proportional | -2.17 | **-1.79** | -1.54 | -1.15 |
| FGA_rim | lgbm | -1.61 | -1.38 | -1.15 | -0.90 |
| FGA_jump2 | proportional | -2.20 | -2.29 | -1.81 | -1.92 |
| FGA_jump2 | lgbm | -1.40 | -1.48 | -1.23 | -1.33 |
| FGA_3 | proportional | -0.60 | -0.73 | -0.25 | -0.21 |
| FGA_3 | lgbm | -0.45 | -0.52 | -0.24 | -0.20 |
| TOV | proportional | -0.40 | -0.40 | -0.32 | -0.32 |
| TOV | lgbm | -0.01 | +0.00 | -0.05 | -0.05 |
| FT_trip | proportional | -2.95 | -2.95 | -2.29 | -2.29 |
| FT_trip | lgbm | -1.94 | -1.91 | -1.46 | -1.45 |

This is the mechanism, and it is specific rather than generic. **An assist
credited as a shot is credit moved from the finisher to the passer**, and a
lineup's passer is usually not its highest-usage player, so the contaminated
labels flattened the usage distribution. Rim shots are where that mattered:
they are the most heavily assisted class after threes (24.8-25.2% assisted) AND
the class where the finisher and the creator are most reliably different people.
Removing the contamination hands the credit back to the finisher, the top of the
distribution rises, and R3's named defect shrinks by 0.38 pp on `FGA_rim`
without anyone touching a share vector -- which is what `CLAUDE.md`'s no-hand-
tuning rule says a real fix looks like.

It does NOT close the defect. Every one of the 50 F1 arm-class cells is still
negative on top-3 (bar `TOV`/`lgbm` at +0.00), so R3 stays OPEN in the change
ledger: this was a contaminating term inside the defect, not the defect itself.

**R14. `FGA_jump2` moves the other way, and the reason is measurable.** Its
top-3 gap widens from -2.20 to -2.29 pp, which is what puts `hier_dirichlet` out
of the eligible set on F1 (it was eligible in round 1). Mid-range jumpers are the
LEAST assisted class in the model (10.2-10.3% against 24.8-28.2% for rim and
three), so the fix relabels only 4.9% of its rows -- a third of the rate the
other two classes see -- while the shrinkage denominators every class shares move
by the full amount. The class gets the cost of the correction without much of its
benefit. The move is 0.09 pp, well inside the run-to-run variation of a 40-draw
Monte-Carlo game-level statistic, and it should not be read as a finding about
mid-range shooting.

**R15. The two unaffected classes are unaffected, as pre-registered.** Section
8.3 committed in advance to attributing any `TOV` or `FT_trip` movement to the
shared exposure denominator or to noise. `FT_trip`'s top-1 and top-3 gaps are
identical to round 1 to two decimals (-2.29 / -2.95 for U1), `TOV`'s to the same
precision, and both classes' F1 log losses move by less than a quarter of their
own floors (`TOV` 1.577664 -> 1.577494 against a 0.000799 floor; `FT_trip`
1.514025 -> 1.523321 on the WF fold and 2.6 vs 4.2 floors clear). The exposure
denominator carries the FGA relabelling into every class's `exposure_asof`, and
the measured size of that channel is: nothing that moves a decision.

**R16. Noise floor.** Spec-identical LightGBM retrains under seeds (0, 1, 2) on
F1 give SDs of 1.0e-05 to 6.8e-05 -- one to two orders of magnitude below the
block-bootstrap SEs (0.000799 to 0.001656) the decision rule actually uses. The
tree's seed noise is not the binding floor on any class, which is the same
finding round 1 reported, and it means every margin quoted above is a statement
about sampling, not about initialisation.

| class | lgbm seed log losses (F1) | seed SD | block-bootstrap SE (the floor) |
|---|---|---:|---:|
| FGA_rim | 1.502629 / 1.502732 / 1.502727 | 0.000058 | 0.001156 |
| FGA_jump2 | 1.488712 / 1.488765 / 1.488702 | 0.000034 | 0.001656 |
| FGA_3 | 1.417636 / 1.417666 / 1.417662 | 0.000016 | 0.001366 |
| TOV | 1.577494 / 1.577474 / 1.577485 | 0.000010 | 0.000799 |
| FT_trip | 1.523321 / 1.523440 / 1.523438 | 0.000068 | 0.001435 |

**R17. Responsiveness and the per-team segment.** Every winner passes the
amended Decision-8 gate with a slope ratio in [0.988, 1.025] -- comfortably
inside the [0.8, 1.2] band and, on four of five classes, closer to 1.000 than
round 1's. Worst decile calibration gap runs 0.23-0.47 pp against a 2.0 pp gate.
Per-team log loss is tight and has no tail of teams the model fails on: the p10
to p90 spread is 0.10-0.22 nats around a median of 1.42-1.58, with an SD of
0.026-0.088. Teams below 200 test events of a class are excluded and counted
(0-11 per class) rather than reported as signal.

| class (F1 winner `lgbm`) | slope ratio | worst calib gap (pp) | teams | per-team median | p10 | p90 | SD | underpowered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FGA_rim | 1.0246 | 0.474 | 364 | 1.5116 | 1.4287 | 1.5604 | 0.0536 | 0 |
| FGA_jump2 | 0.9887 | 0.314 | 355 | 1.5018 | 1.3920 | 1.5673 | 0.0753 | 9 |
| FGA_3 | 0.9954 | 0.336 | 364 | 1.4183 | 1.2995 | 1.5174 | 0.0884 | 0 |
| TOV | 0.9881 | 0.231 | 363 | 1.5817 | 1.5421 | 1.6064 | 0.0257 | 1 |
| FT_trip | 0.9888 | 0.382 | 353 | 1.5319 | 1.4532 | 1.5806 | 0.0507 | 11 |

**R18. What the drop cost.** 136,588 field-goal rows were relabelled (66,120 in
2024, 70,468 in 2025) and 1,136 dropped for having no `shot_shooter_id` (605 /
531). Per class the drop is 0.04-0.27%; per team the median is 0.00-0.18% with
3-19 teams per season above 1% and a worst case of 5.6%. Total modelled events
1,633,164 against round 1's 1,634,792 -- a 0.10% reduction. Nothing was imputed.
The much larger drop in the coverage table, 5.06-5.34% on `TOV`, is the
PRE-EXISTING team-turnover blank documented in R11, not a cost of this fix.

### 8.11 Round-2 decision

| class | DECISION (F1, the selection fold) | confidence | change vs round 1 |
|---|---|---|---|
| `FGA_rim` | **lgbm** | replicated (11.0 floors F1, 4.8 floors WF) | winner unchanged; U1/U2 become ELIGIBLE (top-3 -2.17 -> -1.79 pp) |
| `FGA_jump2` | **lgbm** | floor-thin, straddles | winner unchanged; `hier_dirichlet` loses eligibility (top-3 -2.29 pp) |
| `FGA_3` | **lgbm** | floor-thin, straddles | winner unchanged; margin 1.2 -> 1.5 floors; WF still falls to `proportional` |
| `TOV` | **lgbm, UNCONFIRMED** | reverses on WF | winner unchanged; the reversal reproduces (lgbm LAST on WF) |
| `FT_trip` | **lgbm** | replicated (only eligible arm on F1) | winner unchanged |

Adopted alongside the winners, as fitted parameters rather than choices: the
per-class shrinkage prior and strength and the tree parameters in
`usage_params_v2.json`. **The engine's usage adapter needs no rewiring of its
arm** -- it runs the U1 proportional path in both rounds -- but
`scripts/build_engine_inputs.py` must be repointed from
`models/usage/asof_v2.parquet` + `usage_params_v1.json` to the `usage_v2`
siblings, and the engine inputs rebuilt, before the engine's usage rates are
clean. That is a PM action, not done here.

---

## 9. Round 2b: S1 training-scheme confirmation

### 9.1 Pre-registration (authored 2026-09-10 AFTER round 2 decided and BEFORE round 2b ran; committed in the same commit as this text, with no results)

**Why.** Round 2 held the TRAINING SCHEME fixed at round 1's -- one static fit on
the training seasons -- because holding it fixed is what isolates the shooter-label
fix. L21 (`docs/LEARNINGS.md`; `docs/models/README.md`, "Standing result") then
makes **S1 the standing default scheme for every sub-model**:

> **S1** in-season walk-forward. Refit at each month boundary of the test season
> on all prior seasons PLUS the test season to date, strictly before the refit
> date. Each test event is scored by the most recent refit at or before its own
> game date, so no event is ever in its own fit and every test event is still
> scored -- which keeps S1's log loss comparable with the static arm's on the
> identical test set.

L21's measured claim is specific: on possession outcome, S1 was a CALIBRATION fix
(worst decile gap 2.78 -> 0.98 pp) for a log-loss gain of only 0.0014. This
section asks whether that carries to the allocator, which is a different kind of
model (a within-lineup choice, normalised over five, with as-of features that
already update inside the season). It is a confirmation, not a re-selection: the
ARM is settled by round 2 and is not reopened here.

**Fold.** The project-standard fold 2 -- train through 2023-24, test 2024-25 --
which for this model is `F1` (train 2024, test 2025), the same selection fold
round 2 decided on, for the reason given in 8.1 (L13 leaves no earlier fold).
2026 stays sealed; `fold_slices` calls `assert_not_sealed` on both slices. The
within-2025 walk-forward fold is NOT re-run: it is itself a within-season split
and would confound the question.

**Arms.** Per class, the round-2 F1 winner, PLUS the runner-up when that
runner-up sits inside the round-2 floor -- the same "inside the floor" test the
round-2 decision rule used to declare a tie, read off the same numbers, not a new
threshold. On round 2's results that selects **`lgbm` alone on all five classes**
(the nearest rival is 1.5-11.0 floors behind on every class). Each selected arm
is run under exactly two schemes:

- **S0** static: one fit on the 2024 training slice. This reproduces round 2's
  own number and is the paired control.
- **S1** monthly in-season walk-forward, as quoted above. Month boundaries come
  from `possession_outcome.month_boundaries`, the same function the
  possession-outcome S1 arms use, so the two models' schedules cannot drift.

**What an S1 refit refits, stated in advance.** Everything the arm's fit chooses:
the shrinkage prior and strength (`usage.fit_shrinkage`, refit on each segment's
own training rows) and the tree itself. The LightGBM HYPER-PARAMETERS are NOT
re-searched monthly -- they are carried unchanged from the round-2 F1 search,
which saw the training season only. Re-searching them each month would let the
test season choose its own capacity, which is the leak this pre-registration
exists to avoid. The responsiveness/calibration driver (the player's own shrunk
as-of rate) is likewise computed piecewise under the shrinkage each segment
chose, so the axis moves with the fit rather than being frozen at S0's choice.

**Primary metric.** Per-class 5-way log loss on the F1 test slice.

**Also reported, per class and per scheme:** predicted-vs-actual share
calibration by player as-of-rate decile (worst gap, pp; gate 2.0 pp); the
Decision-8 quintile responsiveness (slope ratio in [0.8, 1.2] AND monotone in
>= 4 of 4 steps, relaxing to 3 of 4 below a 2 pp realised span); Brier, top-1,
top-3; the per-team log-loss distribution over teams with >= 200 test events of
the class (teams below that counted as underpowered, never presented as signal);
and the S1 refit schedule itself -- refit date, rows trained on, how many of them
come from the test season, last training game date, rows scored -- so the
scheme's cost is a number rather than an idea.

**Noise floor.** As in rounds 1 and 2: the game-level block-bootstrap SE of the
log loss (200 replicates, seed 12345) computed separately for each scheme, plus
spec-identical LightGBM retrains under three seeds. **A seed-varied retrain of an
S1 arm replays the entire monthly schedule**, because a retrain that skipped the
schedule would be measuring a different spec. The floor used by the decision rule
is the LARGER of the two schemes' own bootstrap SEs.

**Decision rule (pre-committed).** Adopt **S1** unless it regresses: (a) log loss
by more than the floor, or (b) the calibration gate -- S1 fails the 2.0 pp decile
gate where S0 passes -- or (c) the responsiveness gate, where S0 passes. A log
loss that is merely FLAT is NOT a reason to reject: L21 predicts a calibration
gain and no log-loss gain, so requiring a log-loss win would be testing a claim
nobody made. If S1 regresses on any of the three, the class stays on S0 and the
reason is reported per class rather than pooled.

### 9.2 Artifact naming (so the engine can select a fit by a game's month)

S1 is a SCHEDULE, not a model, so what is persisted is every monthly fit plus a
manifest. Written to `data/processed/models/usage_s1/` -- a versioned sibling;
nothing under `models/usage/` or `models/usage_v2/` is touched.

```
data/processed/models/usage_s1/
  {event_class}/{arm}_{YYYY-MM-DD}.joblib     one monthly refit (tree arms)
  {event_class}/{arm}_{YYYY-MM-DD}.json       one monthly refit (shrinkage-only arms)
  {event_class}/{arm}_static.json|joblib      the S0 control fit
  s1_manifest.json                            every row above, plus provenance
  results_v2b.json  train_log_v2b.txt
```

`{YYYY-MM-DD}` is the **refit date** -- the first day of the month at which that
fit becomes current. **Selection rule for the engine: for a game, take the
artifact with the LATEST refit date at or before the game's own date**; a game
earlier than the first refit date uses the `static` row. Every manifest row also
carries the `prior_kind` and `shrink_m` that fit chose, which is all the engine's
current U1 proportional path needs (`engine/adapters.py` UsageAdapter) -- the tree
booster is persisted for when that path is upgraded. The manifest records
`shooter_key` and `possessions_version` so an artifact can never be paired with
the wrong event table.

### 9.3 What is NOT being decided here

The arm (settled in round 2), the shooter label (settled in round 2), the folds,
the feature set, and the gates. If S1 wins, the change is to the SCHEME only, and
adopting it in the sim is a PM action that also requires
`scripts/build_engine_inputs.py` to be repointed at the `usage_v2` / `usage_s1`
artifacts. Trainer: `scripts/train_usage_v2b.py`. Results are appended below as
sections 9.4-9.7; this commit contains no results.

### 9.4 Run configuration

| item | value |
|---|---|
| trainer | `scripts/train_usage_v2b.py` (imports `train_usage_v1` and `train_usage_v2`) |
| fold | `F1` = project fold 2 (train 2024, test 2025); 2026 sealed |
| shooter key | `shot_shooter_id` (round 2's fix, carried) |
| arms selected by 9.1 | `lgbm` on all five classes -- no runner-up sits inside the round-2 floor on any class |
| schemes | S0 static (paired control) vs S1 monthly in-season walk-forward |
| S1 refit dates | 2024-11-01, 2024-12-01, 2025-01-01, 2025-02-01, 2025-03-01, 2025-04-01 (6 per class, 30 fits persisted) |
| LightGBM hyper-parameters | carried unchanged from the round-2 F1 search (NOT re-searched monthly) |
| bootstrap replicates / seeds | 200 / 3 (each S1 seed replays all 6 monthly refits) |
| seed | 20260910 |

S0 here is a re-fit at this trainer's own RNG seed, not a copy of round 2's
number, and it reproduces round 2 to 1.1e-04 or better on every class (e.g.
`FGA_rim` 1.502739 vs 1.502629, `TOV` 1.577452 vs 1.577494) -- inside the
seed SD measured below. That agreement is the paired control working.

### 9.5 S1 vs S0 on fold 2

| class | S0 log loss | S1 log loss | delta | delta in floors | floor | S0 calib (pp) | S1 calib (pp) | S0 slope | S1 slope | S0 steps | S1 steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FGA_rim | 1.502739 | **1.502225** | -0.000514 | -0.44 | 0.001175 | 0.541 | **0.513** | 1.0245 | 1.0274 | 4/4 | 4/4 |
| FGA_jump2 | 1.488778 | **1.487860** | -0.000918 | -0.54 | 0.001696 | 0.335 | **0.195** | 0.9883 | **0.9994** | 4/4 | 4/4 |
| FGA_3 | 1.417668 | **1.416718** | -0.000950 | -0.69 | 0.001380 | 0.318 | **0.146** | 0.9958 | **1.0004** | 4/4 | 4/4 |
| TOV | 1.577452 | **1.576761** | -0.000691 | -0.82 | 0.000844 | 0.236 | 0.375 | 0.9862 | 1.0166 | 4/4 | 4/4 |
| FT_trip | 1.523522 | **1.522378** | -0.001144 | -0.80 | 0.001436 | 0.395 | **0.376** | 0.9899 | 1.0128 | 4/4 | 4/4 |

Secondary metrics and the per-team segment (teams with >= 200 test events;
teams below that counted, never presented as signal):

| class | scheme | Brier | top-1 | top-3 | teams | per-team median | p10 | p90 | SD | underpowered |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FGA_rim | S0 | 0.7568 | 0.3342 | 0.7603 | 364 | 1.5112 | 1.4309 | 1.5615 | 0.0536 | 0 |
| FGA_rim | S1 | 0.7566 | 0.3344 | 0.7608 | 364 | 1.5105 | 1.4314 | 1.5603 | 0.0535 | 0 |
| FGA_jump2 | S0 | 0.7504 | 0.3452 | 0.7695 | 355 | 1.5032 | 1.3920 | 1.5680 | 0.0753 | 9 |
| FGA_jump2 | S1 | 0.7502 | 0.3451 | 0.7707 | 355 | 1.5009 | 1.3928 | 1.5687 | 0.0758 | 9 |
| FGA_3 | S0 | 0.7362 | 0.3455 | 0.8146 | 364 | 1.4190 | 1.2987 | 1.5167 | 0.0885 | 0 |
| FGA_3 | S1 | 0.7359 | 0.3460 | 0.8153 | 364 | 1.4189 | 1.2978 | 1.5168 | 0.0889 | 0 |
| TOV | S0 | 0.7869 | 0.2727 | 0.6878 | 363 | 1.5817 | 1.5425 | 1.6064 | 0.0256 | 1 |
| TOV | S1 | 0.7866 | 0.2745 | 0.6888 | 363 | 1.5799 | 1.5400 | 1.6069 | 0.0267 | 1 |
| FT_trip | S0 | 0.7644 | 0.3245 | 0.7427 | 353 | 1.5318 | 1.4490 | 1.5804 | 0.0510 | 11 |
| FT_trip | S1 | 0.7641 | 0.3250 | 0.7449 | 353 | 1.5299 | 1.4498 | 1.5815 | 0.0513 | 11 |

**Noise floor.** Three spec-identical retrains per class per scheme, each S1
retrain replaying all six monthly refits:

| class | S0 seed SD | S1 seed SD | S0 bootstrap SE | S1 bootstrap SE |
|---|---:|---:|---:|---:|
| FGA_rim | 0.000119 | 0.000145 | 0.001159 | 0.001175 |
| FGA_jump2 | 0.000020 | 0.000047 | 0.001647 | 0.001696 |
| FGA_3 | 0.000058 | 0.000065 | 0.001372 | 0.001380 |
| TOV | 0.000092 | 0.000056 | 0.000808 | 0.000844 |
| FT_trip | 0.000099 | 0.000044 | 0.001432 | 0.001436 |

Seed SD is 8-35x below the bootstrap SE on every cell, so the bootstrap SE is
the binding floor, as in rounds 1 and 2. S1's bootstrap SE is 0.4-4.5% LARGER
than S0's on every class -- refitting monthly does not reduce sampling variance,
which is the honest reading of a scheme that changes the fit, not the test set.

### 9.6 What S1 actually did

The refit schedule for `FGA_rim` (identical in shape on all five classes; the
row counts scale with the class):

| refit date | train rows | of which from the test season | last train game | rows scored | fitted prior | m |
|---|---:|---:|---|---:|---|---:|
| 2024-11-01 | 213,601 | 0 | 2024-04-08 | 49,081 | position | 50 |
| 2024-12-01 | 262,682 | 49,081 | 2024-11-30 | 38,770 | position | 50 |
| 2025-01-01 | 301,452 | 87,851 | 2024-12-31 | 58,544 | position | 50 |
| 2025-02-01 | 359,996 | 146,395 | 2025-01-31 | 52,923 | position | 50 |
| 2025-03-01 | 412,919 | 199,318 | 2025-02-28 | 31,407 | position | 50 |
| 2025-04-01 | 444,326 | 230,725 | 2025-03-31 | 534 | position | 50 |

Two things to read off it. First, the leak proof is mechanical: every refit's
`last train game` is strictly before its own refit date, and the six `rows
scored` sum to the full 231,259-row test slice, so no event is in its own fit
and none is scored twice. Second, **the fitted shrinkage never moves** -- prior
`position`, m = 50 -- across all six refits on this class, and the same holds on
the other four. The extra in-season data changes the tree, not the amount the
model is willing to trust a player's own rate.

### 9.7 Round-2b decision

**S1 is adopted on all five classes.** It regresses nothing: log loss improves
on all five (0.44-0.82 floors), the calibration gate is passed by both schemes
everywhere, and the Decision-8 responsiveness gate is passed 4/4 with the slope
ratio inside [0.8, 1.2] under both.

| class | DECISION | why |
|---|---|---|
| FGA_rim | **S1** | ll -0.44 floors, calib 0.541 -> 0.513 pp, slope 1.0245 -> 1.0274, no gate lost |
| FGA_jump2 | **S1** | ll -0.54 floors, calib 0.335 -> **0.195** pp, slope 0.9883 -> **0.9994** |
| FGA_3 | **S1** | ll -0.69 floors, calib 0.318 -> **0.146** pp, slope 0.9958 -> **1.0004** |
| TOV | **S1** | ll -0.82 floors; calib 0.236 -> 0.375 pp is a WORSENING but both pass a 2.0 pp gate by 5x and the pre-registered rule regresses only on a lost gate; slope 0.9862 -> 1.0166 |
| FT_trip | **S1** | ll -0.80 floors, calib 0.395 -> 0.376 pp, slope 0.9899 -> 1.0128 |

**Honest framing, stated because it differs from L21.** On possession outcome S1
was a large calibration fix (worst decile gap 2.78 -> 0.98 pp). **It is not that
here, because there was nothing to fix:** the static allocator was already
calibrated to 0.24-0.54 pp against a 2.0 pp gate, five to eight times inside it.
What S1 delivers on this model is a small, uniformly-signed improvement -- every
class's log loss down, four of five classes' worst calibration gap down, three of
five slope ratios moved closer to 1.000 -- none of which is individually
significant against its own floor. **S1 is therefore adopted here because it is
the standing default and it demonstrably costs nothing, NOT because this model
was shown to need it.** The distinction matters for the next sub-model that runs
this check: a flat result is the expected result when the static fit already
passes, and it should not be reported as a win.

The one thing S1 does cost is operational: six fits per class per season instead
of one, and a deployment that must refit monthly. That is the `SCHEME_SIMPLICITY`
ordering the possession-outcome pre-registration already recorded (S0 < S2 < S1),
and it is the reason a tie goes to S0 rather than to S1 -- but a tie is not what
happened on any class here.

### 9.8 Artifacts

30 monthly fits (5 classes x 6 refit dates) plus 5 static controls, written to
`data/processed/models/usage_s1/` under the naming fixed in 9.2, with
`s1_manifest.json` carrying every row's `refit_date`, `prior_kind`, `shrink_m`,
`artifact` path, `shooter_key` and `possessions_version`. Engine selection rule:
take the artifact with the LATEST `refit_date` at or before a game's own date;
before the first refit date, use the `static` row. Nothing under
`models/usage/` or `models/usage_v2/` was written or moved.

---

## 10. Round 3: the `score_diff` state-feature data fix

Trigger: L28 flags that the adopted U5 tree consumes four engine-produced
state features and that wiring it is blocked on an own-row delta audit of
`score_diff` (the same audit L27 ran for `fg_make`). That audit is
`docs/tests/usage_state_confound_2026-09-11.md`, run and committed BEFORE this
section: **`score_diff` is post-outcome on every usage event class at
99.7-99.99%**, the same construction and a cleaner reproduction of the fg_make
defect. `sec_remaining`, `period` and `chance_number` are clean (pre-outcome by
construction, the last confirmed empirically).

### 10.1 The fix

`usage.build_usage_events` gets a new keyword-only parameter,
`score_diff_mode` (`"own_row"`, the historical default, byte-identical for
every existing caller; `"pre_outcome"`, the correction, reconstructed from the
END of the PREVIOUS row of the same game -- 0 on a game's own first row, the
same quantity `GameState.off_score_diff()` feeds the live engine). Nothing
else in the builder changes; `sec_remaining`, `period`, `chance_number` and
every per-alternative feature are untouched. Verified byte-identical other
columns and a 41% row-level change rate in `score_diff` on the corrected build
(2025: 357,994 of 872,578 rows). Trainer:
`scripts/train_usage_v3_lgbm_refit.py` (paired leaked-vs-corrected, same
train/test split, same fixed shrinkage prior/m and LightGBM hyperparameters
FROM ROUND 2 -- this round does not re-litigate the arm or the shrinkage
choice, only the state feature). Corrected events table:
`data/processed/models/usage_v3/events_v3_prestate.parquet` (a versioned
sibling; nothing under `usage/` or `usage_v2/` touched).

### 10.2 Pre-registration (authored 2026-09-11, BEFORE the refit numbers below
were read)

Question: does correcting `score_diff` change the ADOPTED arm's (lgbm, all
five classes) F1 log loss, calibration, top-1/top-3 share gap, or Decision-8
responsiveness slope beyond the round-2 noise floor (`docs/models/usage/
experiments.md` section 8.10, R16: 1.0e-05 to 6.8e-05 seed SD against a
0.0008-0.0017 block-bootstrap SE)? Decision rule: if the corrected arm's F1
log loss stays within its class's round-2 noise floor of the leaked arm's,
the DATA FIX changes nothing about which arm is adopted offline (round 2's
verdict stands unchanged) and the round-3 corrected tree is simply the
CANDIDATE the Decision-10 closed-loop gate (section 11) tests. If it moves
outside the floor, that is reported and the offline verdict is re-examined
before any closed-loop run.

### 10.3 F1 results, paired leaked (`own_row`) vs corrected (`pre_outcome`),
same split/prior/params/seed

| class | leaked ll | corrected ll | delta | round-2 floor | inside floor? | leaked top3 gap (pp) | corrected top3 gap (pp) | leaked state imp% | corrected state imp% |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `TOV` | 1.577452 | 1.577463 | +0.000011 | 0.000799 | YES | +0.004 | -0.003 | 27.2 | 27.5 |
| `FGA_rim` | 1.502739 | 1.502941 | +0.000202 | 0.001156 | YES | -1.372 | -1.376 | 28.7 | 29.3 |
| `FGA_jump2` | 1.488778 | 1.488748 | -0.000030 | 0.001656 | YES | -1.488 | -1.484 | 29.4 | 28.1 |
| `FGA_3` | 1.417668 | 1.417789 | +0.000121 | 0.001366 | YES | -0.510 | -0.508 | 29.6 | 29.6 |
| `FT_trip` | 1.523522 | 1.523499 | -0.000023 | 0.001435 | YES | -1.913 | -1.909 | 27.8 | 28.1 |

Full per-arm calibration, responsiveness and game-level (top-1/top-3,
per-role SD ratio) blocks for both arms of every class:
`data/processed/models/usage_v3/lgbm_refit_v3.json`.

### 10.4 Round-3 decision

**The corrected `score_diff` moves every class's F1 log loss by less than its
round-2 noise floor** (largest move: `FGA_rim` +0.000202 against a 0.001156
floor, 17% of the floor; smallest: `TOV` +0.000011, 1.4% of the floor). Per the
section 10.2 decision rule, **round 2's verdict is UNCHANGED**: `lgbm` stays
the winner on all five classes (`TOV` stays UNCONFIRMED, reversing on the
robustness fold, exactly as round 2 found -- this fix does not touch that).
Top-3 usage-share gap and split importance also move by less than a rounding
error on every class. **The round-3 corrected tree (`data/processed/models/
usage_v3/lgbm_tree/*_corrected.joblib`) is therefore the SAME arm round 2
adopted offline, with one honestly-built input column**, and it is this
artifact -- not a re-decided arm -- that the Decision-10 closed-loop gate in
section 11 evaluates.

### 10.5 Why the manufactured effect is small here despite score_diff
carrying ~27-29% of split importance

Unlike `fg_make`, where `score_diff`'s own-row leak directly encodes the
row's own target (a made shot's post-outcome margin literally contains
whether the shot the model is predicting went in), usage's target is WHICH
of five teammates took the shot -- an identity question the leak only
touches indirectly, through whatever correlation exists between the
leaked/corrected margin value and rotation patterns already captured by the
alternative-level features (`share`, `usage_rank`, role). A tree can therefore
extract nearly the same amount of USABLE signal from the leaked and the
corrected column for THIS target, even though the corrected column is the
honest one and the leaked one is not. This is why the log-loss deltas above
sit far inside the round-2 noise floor while fg_make's leak was worth 35-38
floors: the same construction defect, a much smaller offline consequence for
this target. It is NOT evidence that the leak is harmless in the engine --
Decision 10 exists precisely because an offline-invisible confound can still
create a closed-loop skew once the ENGINE, not history, generates the state
the tree conditions on (L27's headline: fg_make's leak "passed every offline
gate"). Section 11 is the test that actually answers that question.

---

## 11. Decision-10 closed-loop gate: the corrected tree, live vs frozen vs
refit-without-state (pre-registered 2026-09-11, BEFORE any arm is run)

Authored and committed before `scripts/run_usage_tree_closed_loop.py` is
invoked for any arm other than `reference` (the served baseline, run first
only to confirm the harness reproduces the current engine-v1 gate numbers).

### 11.1 Arms

| arm | `ENGINE_USAGE` | what it is |
|---|---|---|
| served baseline | `reference` | `usage.draw_player`'s U1 proportional rule, unchanged (`adapters.UsageAdapter`) |
| live | `tree_v3` | round-3 corrected LightGBM tree, state features (`score_diff`, `sec_remaining`, `period`, `chance_number`) read LIVE off `GameState` every step |
| frozen | `tree_v3_freeze` | the SAME boosters, state features held at their pregame value for the whole game (`score_diff=0`, `sec_remaining=2400`, `period=1`, `chance_number=1`) -- the Decision-10 freeze ablation |
| refit-without-state | `tree_v3_nostate` | a tree refit with NO state features at all (`LGBM_ALT_FEATURES` only), live otherwise -- L31/L33's refit-without arm, which SIZES a loop a freeze can only DETECT |

All four share the pinned sub-models of the current engine-v1 baseline
(`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round4_B1`,
`ENGINE_CLOCK=v3c_srfloor_P3_s1`, `ENGINE_REBOUND=s1_weekly`,
`ENGINE_FREE_THROW=s1_conf_aligned`, `ENGINE_ROTATION=reference`) and the same
paired RNG streams (`StreamBook`), so the only thing that differs between runs
is `ENGINE_USAGE`.

### 11.2 Population and seeds

F2 2025 slate, the SAME 500-game stride subset every other closed-loop report
on this engine uses (`docs/tests/engine_v1_gates_F2_2025_s200_2026-09-11.md`,
`run_rot5_closed_loop.py`, `run_clk3c_closed_loop.py`): sorted by `game_id`
ascending, every 11th row, the first 500. 5-seed smoke first; scaled to 25
seeds only if it fits the session's wall-clock budget. 6 engine workers (the
lane's thread/worker cap).

### 11.3 Gate lines

- **G1-G5, G9**: no regression beyond the paired noise, read from
 `scripts/eval_gates.py` against `docs/tests/
 engine_v1_gates_F2_2025_s200_2026-09-11.md`'s existing `reference`-arm
 numbers as the noise-floor baseline (same engine, same season, same subset
 rule; the only lever is `ENGINE_USAGE`).
- **Decision-10 core** (`ARCHITECTURE_DECISIONS.md`): margin SD ratio and
 home/away score correlation inside the G1/G2 tolerances, possessions per
 game unmoved, for BOTH the live and the frozen arm; the refit-without arm run
 alongside per L31/L33 ("every closed-loop gate reports both the freeze and
 the refit-without arm").
- **Usage-specific** (this model's own G8-adjacent reads, since G8 in the
 engine-v1 doc is rotation's minutes/players-used gate): per-player FGA share
 distribution (p10/p50/p90), top-1 and top-3 usage-share gap vs the
 `reference` arm's own simulated distribution (not vs a fresh real-world
 grade -- that comparison already exists in the engine-v1 gate doc for
 `reference` and is not re-run here), and the per-player quintile
 responsiveness slope (pregame `rate_total` quintile vs simulated FGA share),
 the same Decision-8 amendment gate (slope ratio in [0.8, 1.2]) applied
 offline in section 7.

### 11.4 Decision rule

Wire the tree (name which arm -- live or frozen) only if: (a) it clears every
gate in 11.3, (b) live and frozen do NOT differ from each other beyond the
paired-seed noise (a difference here is the closed-loop signature L27/L31
describe -- a real feedback effect the offline fold cannot see), and (c) live
beats refit-without on the usage-specific reads (otherwise the state buys
nothing worth the extra model complexity and the simpler nostate arm is
preferred, ties going to the simpler model per `CLAUDE.md`'s bake-off rule).
Any gate not run before the session's deadline is reported as NOT RUN, never
as a pass.

### 11.5 Results

See `docs/tests/usage_decision10_gate_2026-09-11.md`.
