# pace experiments (append-only)

## Pre-registration (2026-09-10, before any code, authored by the PM)

---
Target: possessions per game for a matchup (both teams share it). Two target definitions compared as a first dimension: T_box = mean over both teams of FGA - OREB + TOV + 0.44 FTA (team_box), and T_pbp = possessions counted from hoopR pbp (a possession ends on a made FG not followed by an and-one FT sequence, a defensive rebound, a turnover, the last made FT of a trip, or end of period; document the exact rule; exclude truncated-pbp games). Report how T_box and T_pbp differ (mean, SD, corr) and pick the definition that the possession engine will actually simulate (T_pbp, unless it is unreliable) with reasons.
Universe: D-I, non-truncated games from data/processed/games_universe.parquet. Folds: F1 train {2022, 2023} test 2024; F2 train {2022, 2023, 2024} test 2025 (selection). 2026 sealed (call seal.assert_not_sealed).
Feature sets: A_tempo (own tempo_rel of both teams as-of, KenPom adj_t_rel of both as-of, site); B_plus_season (A + days since season start, month, season index); C_plus_style (B + both teams' as-of per-100 rates of 3PA, FTA, TOV, OREB% and their opponents-allowed versions from the team gamelogs, shift(1) and expanding-mean so strictly pregame); D_plus_state (C + rest days for each team, back-to-back flag, conference-game flag, neutral).
Model classes: multiplicative formula (tempo_A * tempo_B / league_mean, no fitting; the KenPom-style baseline), ridge, Gaussian GLM (incumbent, as in the Control), LightGBM regressor, and for the distribution family: Gaussian with fitted residual SD, Normal with heteroscedastic SD (log-linear in the features), and a NegBin/Poisson on the integer count. Every class x feature set on both folds.
Metrics: RMSE and MAE on F2; calibration of the predictive distribution (PIT K-S p, coverage of 50/80/95% intervals); G1 mean and SD by month; responsiveness (predicted pace by tempo-difference quintile vs actual); noise floor = seed-varied refit of the best tree model and a bootstrap SE for the linear ones.
Decision rules: winner = lowest F2 RMSE among arms whose PIT K-S p > 0.05 and whose by-month G1 passes; if two arms are within the noise floor, the simpler (fewer features, linear before tree) wins; a tree arm must beat the best linear arm by more than the floor to be chosen. The distribution family is chosen by PIT/coverage, not RMSE.
---

## Implementation notes on the pre-registered grid (read before the results below)

Committed here, before any model was fit, so the grid design choices are pinned down as firmly as the prose above:

1. **Target-definition comparison runs first, on both folds' full universe**, and its winner becomes THE target for the entire class x feature-set x fold grid that follows -- the pre-registration frames it as "a first dimension" resolved before the rest, not a second copy of the whole grid.
2. **Feature columns are per-team (`home_`/`away_` pairs), not pre-summed.** The incumbent Control pace GLM sums the two teams' tempo features into one symmetric column (`docs/models/control_engine/features.md` section 1b) because its own spec fixed that form; this is a fresh bake-off and ridge/GLM/LightGBM can all learn a symmetric (or asymmetric) combination on their own from two columns, which is strictly more flexible. `neutral` is the only site term (matching Control; home/away pace edges are not part of this spec).
3. **`multiplicative`** is evaluated with the project's self-contained rating (own tempo_rel x own league_tempo_mean as-of), per L9's finding that own ratings tie centred KenPom within seed noise -- not a second, third-party-dependent arm. It is a zero-fitted-parameter formula (`tempo_rel_home * tempo_rel_away * league_tempo_mean_asof`, algebraically identical to the classic `AdjT_A * AdjT_B / league_AdjT` form, reconstructed from already-centred, as-of pieces so no raw level is ever read). Because it ignores every feature beyond team tempo, its row is identical across feature sets B/C/D by construction -- reported once per fold, not four times.
4. **Distribution family** is fit in two stages, exactly matching the decision rule's own separation of RMSE (point estimate) from PIT/coverage (distribution shape). Stage 1: every (class, feature set, fold) arm in the primary grid carries a Gaussian, fitted-constant-residual-SD wrapper (identical mechanism to the Control's pace model) so every arm has an RMSE, an MAE, a PIT K-S p, and interval coverage -- this is what the winner rule (`lowest F2 RMSE among arms whose PIT p > 0.05 and G1 passes`) is applied to. Stage 2: once the winning (class, feature set) is chosen on Stage 1, that specific point-estimate model is re-wrapped in the two alternative families -- heteroscedastic Gaussian (OLS of log squared residual on the same feature set, i.e. log-linear in the features) and a native NegBin/Poisson GLM fit directly on rounded `game_poss` counts using the same feature set -- and the three are compared by PIT/coverage only, never RMSE, per the pre-registered rule.
5. **Noise floor.** LightGBM: 5 refits at different `random_state` seeds (bagging/feature-subsampling randomness), F2 RMSE spread reported as the floor. Linear arms (ridge, GLM): non-parametric bootstrap (500 resamples of the F2 test games) SE of the RMSE, x1.96 reported as the floor width for a fair comparison against the tree spread.

---

---

---

## Results (appended 2026-09-10, after the run)

Reproduce with:

```
.venv/Scripts/python.exe scripts/build_own_ratings.py
.venv/Scripts/python.exe scripts/build_possessions_pbp.py
.venv/Scripts/python.exe scripts/train_pace_v1.py
```

### R1. T_box vs T_pbp -- the target-definition decision

Pooled over the 2022-2025 D-I, non-truncated universe (the same games both definitions can be computed on; a game missing pbp coverage is excluded from T_pbp only, never imputed):

| season | n_box_universe | n_matched | n_missing_from_pbp | box_mean | box_sd | pbp_mean | pbp_sd | corr | mean_diff_pbp_minus_box | sd_diff | mean_abs_diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 5396.0000 | 5282 | 114.0000 | 68.0526 | 5.5391 | 68.2788 | 5.7816 | 0.9580 | 0.2262 | 1.6582 | 0.9146 |
| 2023 | 5623.0000 | 5542 | 81.0000 | 67.8698 | 5.5342 | 68.0026 | 5.6434 | 0.9781 | 0.1329 | 1.1753 | 0.8542 |
| 2024 | 5632.0000 | 5551 | 81.0000 | 68.4063 | 5.4836 | 68.7204 | 5.5724 | 0.9840 | 0.3141 | 0.9931 | 0.8115 |
| 2025 | 5700.0000 | 5590 | 110.0000 | 67.8528 | 5.4636 | 68.0556 | 5.6525 | 0.9669 | 0.2029 | 1.4421 | 0.8388 |
| ALL | n/a | 21965 | n/a | 68.0450 | 5.5089 | 68.2639 | 5.6683 | 0.9717 | 0.2189 | 1.3388 | 0.8540 |

**Decision: T_pbp.** T_pbp is reliable (pooled corr 0.9717 with T_box, mean abs diff 0.854 poss, SD of the per-game difference 1.339), so the pre-registered default applies: simulate the pbp-derived count, the more direct measurement of an actual possession, rather than the box-score formula's estimate of one.

### R2. Stage-1 grid -- every (model class, feature set, fold), Gaussian fitted-SD wrapper

| fold | feature_set | model_class | n_train | n_test | rmse | mae | resid_sd_fitted | pit_ks_stat | pit_ks_p | coverage_50 | coverage_80 | coverage_95 | g1_month_pass | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | A_tempo | multiplicative | 10824 | 5551 | 4.8083 | 3.7187 | 4.8845 | 0.0740 | 0.0000 | 0.5259 | 0.8301 | 0.9548 | True | 0.0000 |
| F1 | A_tempo | ridge | 10824 | 5551 | 4.8190 | 3.6670 | 4.8385 | 0.0514 | 0.0000 | 0.5440 | 0.8316 | 0.9494 | False | 0.0510 |
| F1 | A_tempo | glm_gaussian | 10824 | 5551 | 4.8199 | 3.6679 | 4.8385 | 0.0514 | 0.0000 | 0.5440 | 0.8312 | 0.9494 | False | 0.0217 |
| F1 | A_tempo | lightgbm | 10824 | 5551 | 4.8538 | 3.6978 | 4.7376 | 0.0467 | 0.0000 | 0.5282 | 0.8200 | 0.9440 | False | 17.5549 |
| F1 | B_plus_season | multiplicative | 10824 | 5551 | 4.8083 | 3.7187 | 4.8845 | 0.0740 | 0.0000 | 0.5259 | 0.8301 | 0.9548 | True | 0.0000 |
| F1 | B_plus_season | ridge | 10824 | 5551 | 4.8366 | 3.6621 | 4.7934 | 0.0815 | 0.0000 | 0.5478 | 0.8308 | 0.9463 | False | 0.0644 |
| F1 | B_plus_season | glm_gaussian | 10824 | 5551 | 4.8383 | 3.6635 | 4.7934 | 0.0818 | 0.0000 | 0.5489 | 0.8301 | 0.9467 | False | 0.0159 |
| F1 | B_plus_season | lightgbm | 10824 | 5551 | 4.7897 | 3.6358 | 4.6673 | 0.0516 | 0.0000 | 0.5327 | 0.8202 | 0.9465 | False | 11.6205 |
| F1 | C_plus_style | multiplicative | 10824 | 5551 | 4.8083 | 3.7187 | 4.8845 | 0.0740 | 0.0000 | 0.5259 | 0.8301 | 0.9548 | True | 0.0000 |
| F1 | C_plus_style | ridge | 10824 | 5551 | 4.7969 | 3.6327 | 4.7807 | 0.0731 | 0.0000 | 0.5489 | 0.8310 | 0.9470 | False | 0.1162 |
| F1 | C_plus_style | glm_gaussian | 10824 | 5551 | 4.7981 | 3.6336 | 4.7807 | 0.0730 | 0.0000 | 0.5493 | 0.8314 | 0.9472 | False | 0.0509 |
| F1 | C_plus_style | lightgbm | 10824 | 5551 | 4.7776 | 3.6245 | 4.3999 | 0.0295 | 0.0001 | 0.5059 | 0.7939 | 0.9351 | True | 19.5915 |
| F1 | D_plus_state | multiplicative | 10824 | 5551 | 4.8083 | 3.7187 | 4.8845 | 0.0740 | 0.0000 | 0.5259 | 0.8301 | 0.9548 | True | 0.0020 |
| F1 | D_plus_state | ridge | 10824 | 5551 | 4.7827 | 3.6245 | 4.7762 | 0.0701 | 0.0000 | 0.5467 | 0.8352 | 0.9487 | False | 0.1433 |
| F1 | D_plus_state | glm_gaussian | 10824 | 5551 | 4.7857 | 3.6272 | 4.7758 | 0.0711 | 0.0000 | 0.5467 | 0.8344 | 0.9490 | False | 0.0615 |
| F1 | D_plus_state | lightgbm | 10824 | 5551 | 4.7602 | 3.6161 | 4.5984 | 0.0394 | 0.0000 | 0.5269 | 0.8130 | 0.9436 | True | 24.8067 |
| F2 | A_tempo | multiplicative | 16375 | 5590 | 4.8805 | 3.7024 | 4.8582 | 0.0930 | 0.0000 | 0.5342 | 0.8349 | 0.9572 | True | 0.0010 |
| F2 | A_tempo | ridge | 16375 | 5590 | 4.8587 | 3.6729 | 4.8279 | 0.0688 | 0.0000 | 0.5358 | 0.8410 | 0.9585 | False | 0.0659 |
| F2 | A_tempo | glm_gaussian | 16375 | 5590 | 4.8585 | 3.6726 | 4.8279 | 0.0690 | 0.0000 | 0.5352 | 0.8408 | 0.9585 | False | 0.0152 |
| F2 | A_tempo | lightgbm | 16375 | 5590 | 4.8654 | 3.6766 | 4.7565 | 0.0644 | 0.0000 | 0.5317 | 0.8315 | 0.9562 | False | 17.6601 |
| F2 | B_plus_season | multiplicative | 16375 | 5590 | 4.8805 | 3.7024 | 4.8582 | 0.0930 | 0.0000 | 0.5342 | 0.8349 | 0.9572 | True | 0.0020 |
| F2 | B_plus_season | ridge | 16375 | 5590 | 4.8433 | 3.6835 | 4.7761 | 0.1010 | 0.0000 | 0.5284 | 0.8326 | 0.9594 | True | 0.0672 |
| F2 | B_plus_season | glm_gaussian | 16375 | 5590 | 4.8439 | 3.6842 | 4.7760 | 0.1019 | 0.0000 | 0.5292 | 0.8318 | 0.9590 | True | 0.0257 |
| F2 | B_plus_season | lightgbm | 16375 | 5590 | 4.8177 | 3.6517 | 4.6728 | 0.0895 | 0.0000 | 0.5254 | 0.8315 | 0.9531 | True | 19.2197 |
| F2 | C_plus_style | multiplicative | 16375 | 5590 | 4.8805 | 3.7024 | 4.8582 | 0.0930 | 0.0000 | 0.5342 | 0.8349 | 0.9572 | True | 0.0020 |
| F2 | C_plus_style | ridge | 16375 | 5590 | 4.8358 | 3.6747 | 4.7603 | 0.0994 | 0.0000 | 0.5281 | 0.8338 | 0.9580 | True | 0.1835 |
| F2 | C_plus_style | glm_gaussian | 16375 | 5590 | 4.8355 | 3.6743 | 4.7601 | 0.0999 | 0.0000 | 0.5279 | 0.8340 | 0.9580 | True | 0.0631 |
| F2 | C_plus_style | lightgbm | 16375 | 5590 | 4.8301 | 3.6760 | 4.4866 | 0.0980 | 0.0000 | 0.5009 | 0.8068 | 0.9460 | True | 20.8365 |
| F2 | D_plus_state | multiplicative | 16375 | 5590 | 4.8805 | 3.7024 | 4.8582 | 0.0930 | 0.0000 | 0.5342 | 0.8349 | 0.9572 | True | 0.0020 |
| F2 | D_plus_state | ridge | 16375 | 5590 | 4.8207 | 3.6568 | 4.7540 | 0.0991 | 0.0000 | 0.5286 | 0.8340 | 0.9571 | True | 0.2038 |
| F2 | D_plus_state | glm_gaussian | 16375 | 5590 | 4.8204 | 3.6564 | 4.7538 | 0.0991 | 0.0000 | 0.5274 | 0.8335 | 0.9576 | True | 0.1292 |
| F2 | D_plus_state | lightgbm | 16375 | 5590 | 4.8185 | 3.6599 | 4.6238 | 0.0932 | 0.0000 | 0.5165 | 0.8231 | 0.9517 | True | 29.0923 |

### R3. Noise floor (F2)

Bootstrap SE of RMSE (500 resamples of the F2 test games) for every linear-class arm; the floor used below is 1.96 x that SE (a 95% CI half-width on the RMSE estimate), maxed against the LightGBM seed-refit spread.

| feature_set | model_class | rmse | bootstrap_se | floor_1_96se |
|---|---|---|---|---|
| A_tempo | multiplicative | 4.88052 | 0.07747 | 0.15184 |
| A_tempo | ridge | 4.85869 | 0.08012 | 0.15704 |
| A_tempo | glm_gaussian | 4.85854 | 0.08011 | 0.15702 |
| B_plus_season | multiplicative | 4.88052 | 0.07747 | 0.15184 |
| B_plus_season | ridge | 4.84327 | 0.07746 | 0.15181 |
| B_plus_season | glm_gaussian | 4.84393 | 0.07738 | 0.15166 |
| C_plus_style | multiplicative | 4.88052 | 0.07747 | 0.15184 |
| C_plus_style | ridge | 4.83577 | 0.07683 | 0.15059 |
| C_plus_style | glm_gaussian | 4.83553 | 0.07678 | 0.15050 |
| D_plus_state | multiplicative | 4.88052 | 0.07747 | 0.15184 |
| D_plus_state | ridge | 4.82068 | 0.07683 | 0.15058 |
| D_plus_state | glm_gaussian | 4.82038 | 0.07678 | 0.15049 |

LightGBM seed-refit spread (B_plus_season, F2, seeds (20260910, 20260911, 20260912, 20260913, 20260914)): 4.8177, 4.8161, 4.8181, 4.8167, 4.8166 -> spread 0.0020.

**Noise floor = 0.1570** (max of the two).

### R4. Decision -- winning (model class, feature set)

- NO ARM passed both pre-registered gates on F2: PIT K-S p > 0.05 failed for EVERY arm (every arm's F2 PIT p is below 1e-19; see the distribution-family diagnosis -- overtime games right-skew and fat-tail the residuals of every point-estimate model alike, a defect in the shared target, not a reason to prefer one arm over another). Falling back to the gate that IS achievable, by-month G1, and selecting only among arms that pass it.
- Lowest-RMSE gate-passing arm: lightgbm/B_plus_season (RMSE 4.8177). 13 arm(s) within the noise floor (0.1570) of it; simplest of those (fewest features, linear before tree): multiplicative/A_tempo (RMSE 4.8805).

**Winner: `multiplicative` / `A_tempo`, target T_pbp.** F2 RMSE 4.8805, MAE 3.7024.

F2 arms that failed a gate (PIT p <= 0.05 or by-month G1 failed):

| feature_set | model_class | rmse | pit_ks_p | g1_month_pass |
|---|---|---|---|---|
| A_tempo | multiplicative | 4.8805 | 0.0000 | True |
| A_tempo | ridge | 4.8587 | 0.0000 | False |
| A_tempo | glm_gaussian | 4.8585 | 0.0000 | False |
| A_tempo | lightgbm | 4.8654 | 0.0000 | False |
| B_plus_season | multiplicative | 4.8805 | 0.0000 | True |
| B_plus_season | ridge | 4.8433 | 0.0000 | True |
| B_plus_season | glm_gaussian | 4.8439 | 0.0000 | True |
| B_plus_season | lightgbm | 4.8177 | 0.0000 | True |
| C_plus_style | multiplicative | 4.8805 | 0.0000 | True |
| C_plus_style | ridge | 4.8358 | 0.0000 | True |
| C_plus_style | glm_gaussian | 4.8355 | 0.0000 | True |
| C_plus_style | lightgbm | 4.8301 | 0.0000 | True |
| D_plus_state | multiplicative | 4.8805 | 0.0000 | True |
| D_plus_state | ridge | 4.8207 | 0.0000 | True |
| D_plus_state | glm_gaussian | 4.8204 | 0.0000 | True |
| D_plus_state | lightgbm | 4.8185 | 0.0000 | True |

### R5. Responsiveness, winning arm (F2)

Literal pre-registered check: predicted pace by tempo-DIFFERENCE quintile. Pace is structurally a function of the two teams' combined tempo, not their difference, so a flat pattern here is the expected, non-defective result; the tempo-SUM quintile check below is the one that should slope.

| quintile | n | x | predicted | actual | delta |
|---|---|---|---|---|---|
| 1.0000 | 1118.0000 | -0.0648 | 68.7136 | 67.9477 | 0.7659 |
| 2.0000 | 1118.0000 | -0.0248 | 68.6937 | 68.2200 | 0.4736 |
| 3.0000 | 1118.0000 | -0.0001 | 68.5019 | 67.8609 | 0.6410 |
| 4.0000 | 1118.0000 | 0.0243 | 68.7808 | 68.1284 | 0.6525 |
| 5.0000 | 1118.0000 | 0.0647 | 68.7550 | 68.1212 | 0.6338 |

Tempo-SUM quintile (both teams' combined tempo -- the quantity pace should actually slope with):

| quintile | n | x | predicted | actual | delta |
|---|---|---|---|---|---|
| 1.0000 | 1118.0000 | 1.9376 | 64.3373 | 63.9459 | 0.3914 |
| 2.0000 | 1118.0000 | 1.9772 | 67.0899 | 66.7706 | 0.3194 |
| 3.0000 | 1118.0000 | 2.0004 | 68.7033 | 67.9723 | 0.7311 |
| 4.0000 | 1118.0000 | 2.0241 | 70.2787 | 69.6731 | 0.6057 |
| 5.0000 | 1118.0000 | 2.0640 | 73.0357 | 71.9164 | 1.1193 |

### R6. G1 by month, winning arm (F2)

Sampled via `pace.sample_pace`, 50 seeds.

| month | n | sim_mean | actual_mean | d_mean | sim_sd | actual_sd | d_sd | status |
|---|---|---|---|---|---|---|---|---|
| 1 | 1399 | 68.414 | 67.617 | 0.797 | 5.808 | 5.737 | 0.071 | PASS |
| 2 | 1351 | 68.017 | 67.266 | 0.750 | 5.816 | 5.434 | 0.383 | PASS |
| 3 | 764 | 68.031 | 67.632 | 0.400 | 5.816 | 5.461 | 0.355 | PASS |
| 4 | 17 | 67.541 | 69.382 | -1.842 | 5.803 | 5.799 | 0.003 | UNDERPOWERED |
| 11 | 1158 | 69.949 | 69.389 | 0.560 | 5.657 | 5.641 | 0.016 | PASS |
| 12 | 901 | 69.069 | 68.542 | 0.528 | 5.719 | 5.663 | 0.056 | PASS |

### R7. Distribution family (winner's feature set, F2) -- chosen by PIT/coverage, not RMSE

| family | rmse | pit_ks_p | coverage_50 | coverage_80 | coverage_95 | param |
|---|---|---|---|---|---|---|
| gaussian_fixed | 4.8805 | 0.0000 | 0.5342 | 0.8349 | 0.9572 | sd=4.8582 |
| gaussian_hetero | 4.8805 | 0.0000 | 0.5054 | 0.8195 | 0.9506 | mean_sd=4.6193 (range 3.1404-6.2496) |
| poisson_count | 4.8653 | 0.0000 | 0.8320 | 0.9764 | 0.9928 | poisson dev/df=0.337 |

**Chosen distribution family: `gaussian_hetero`.**


### R8. Diagnosis -- why does every arm fail the PIT gate?

Traced on the winner (`multiplicative`/`A_tempo`, F2), the same 5,590-game test set as R2-R7. Residual shape, all games vs regulation-only (`n_periods <= 2`):

| subset | n | mean | SD | skew | excess kurtosis |
|---|---|---|---|---|---|
| all F2 test games | 5,590 | -0.633 | 4.839 | +0.665 | +3.909 |
| regulation only | 5,277 | -1.158 | 4.184 | -0.170 | +1.790 |
| overtime only (n_periods > 2) | 313 (5.60% -- matches the reference OT rate) | +8.207 | 6.352 | -- | -- |

Overtime games run about 8.2 possessions hotter than the model's (OT-blind) point estimate, exactly consistent with an extra 5-minute period at the game's own pace (`docs/models/control_engine`'s own OT stub uses the same 5/40 fraction). This single mechanism accounts for essentially all of the pooled right-skew (+0.665 -> -0.170 with OT removed) and about half of the excess kurtosis (+3.909 -> +1.790). None of the four point-estimate model classes carries any OT-awareness (the target, `poss_pbp`, already includes whatever overtime a game played, exactly the L5/Control precedent: "the pace target is the OBSERVED possession count, which already includes any overtime a real game played"), so this is a target-mixture defect every arm inherits identically -- consistent with every arm's F2 PIT p landing below 1e-19 regardless of model class or feature set (R2).

Excess kurtosis of +1.79 remains even after excluding OT games entirely, and a PIT check restricted to regulation-only games (using a SD fitted on regulation residuals alone) still rejects uniformity at p = 1.2e-70 (KS statistic 0.124, n = 5,277; the critical value at p = 0.05 for this n is ~0.019, so this is not a large-n artifact). Pace residuals are genuinely more sharply peaked and fatter-tailed than Gaussian beyond just the OT mixture -- a real, second gap the Gaussian and heteroscedastic-Gaussian families both inherit (R7), and one a Student-t or similar heavy-tailed error family would be the natural next bake-off arm to close.

**Reading for the sim:** this is a target-distribution defect, not a reason to prefer one arm over another (every arm fails identically), so it does not change the R4 winner. It DOES mean the chosen `gaussian_hetero` family (R7) is best-of-three, not well-calibrated in an absolute sense, and the possession engine inherits the same gap the Control's explicit OT stub already carries (`docs/models/control_engine/experiments.md` R11 item 4) until the L5 overtime model exists. Recorded as a follow-up, not patched here (`docs/SIM_GUARDRAILS.md`: fix the sub-model, never the output).
