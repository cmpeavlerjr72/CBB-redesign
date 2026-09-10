# L3 FIELD-GOAL MAKE: experiments

Append-only. Status changes go to `docs/models/change_ledger.md` in the same commit
(`CLAUDE.md`, standing rule "bake-off before any choice").

---

## 1. Pre-registration (authored by the PM, 2026-09-10, BEFORE any modelling)

Target: made vs missed for each field-goal attempt, modelled SEPARATELY by shot class {FGA_rim, FGA_jump2, FGA_3} (three binary models, since the base rates, shooter skill signal and defense signal differ by class), using the v2 event layer (L16 rim override). Blocked shots are misses. And-one attempts are included as makes. Universe: D-I, non-truncated, pbp_complete games; seasons 2022-2025; F1 train {2022, 2023} test 2024; F2 train {2022, 2023, 2024} test 2025 (selection); 2026 sealed (seal.assert_not_sealed).
Shooter identity: CBBD player id (as in free_throw), ESPN id attached where the crosswalk resolves.
Feature sets: A_team (offense as-of class make rate, defense as-of class make rate allowed, own ratings, site, season index), B_plus_shooter (A + shooter as-of class make rate with attempts-to-date, shooter prior-season class rate, shooter position group, minutes-to-date), C_plus_state (B + period, seconds remaining, score diff, bonus, chance number, is_transition/duration bucket, assisted flag NOT included because it is post-outcome in the feed; document), D_plus_lineup (C + for 2024+ only, the defensive five's as-of rim-protection / perimeter rates aggregated; own fold train 2024 test 2025 like rebound's D arm, to decide whether defense is lineup-level or team-level).
Model classes: team-level baseline (A only, logistic ridge); empirical-Bayes shrinkage of the shooter's class rate toward a fitted prior (grid over prior: league, position, prior-season; strength fitted on train), combined multiplicatively with the defense allowed rate on the logit scale; logistic ridge on the full bundle; LightGBM on the full bundle with parameter search on F1 only.
Metrics on F2 per class: log loss and Brier; calibration by decile (<= 2.0 pp worst gap); responsiveness by shooter as-of rate quintile AND by defense as-of allowed quintile (both must be monotone in 4 of 4 steps: the matchup-specific rule applies to defense too); a transfer subset check for the shrinkage arm; the implied team eFG% by team tercile (G4) computed from the three class models on the test season's actual shot mix; noise floor per convention (tree seed refits; game-block bootstrap SE for the others).
Decision rules: per class, winner = lowest log loss among arms passing calibration and both responsiveness checks; a tree arm must beat the best passing non-tree arm by more than the floor; ties to the simpler arm (baseline < EB < ridge < tree). The D_plus_lineup verdict (lineup-level vs team-level defense) is decided on its own fold by the same more-than-the-floor rule and reported separately. If no arm passes for a class, adopt nothing for that class and report.

---

<!-- RESULTS APPENDED BELOW BY scripts/train_fg_make_v1.py -->

## 2. Grid configuration (as executed, run 2026-09-10 15:25, `scripts/train_fg_make_v1.py --version v2`)

| Dimension | Values |
|---|---|
| Target | one field-goal attempt: made vs missed, **three separate binary models** by shot class |
| Shot classes | `FGA_rim`, `FGA_jump2`, `FGA_3` -- no fit is ever shared between them (`fg_make.fit_by_class`) |
| Arms | `team_baseline` (logistic ridge on A_team only), `eb_shrink` (shooter EB x defence allowed, on the logit scale; prior and both strengths FITTED), `ridge` (logistic ridge on the full bundle), `lgbm` (LightGBM on the full bundle, parameters searched on F1 only) |
| Feature sets | `A_team`, `B_plus_shooter`, `C_plus_state` (the full bundle) on F1/F2; `D_plus_lineup` on its own fold (section 8) |
| Folds | F1: train 2022+2023, test 2024. F2: train 2022+2023+2024, test 2025 (selection). L2: train 2024, test 2025 (lineup bundle only) |
| Sealed | 2026 -- `assert_not_sealed` on every train and test slice |
| Primary metric | attempt-level log loss on F2, per class |
| Possessions version | `v2`, rim-location override 2.27 ft (L16). This model's TARGET depends on it |
| Noise floor | per class -- linear: 200-replicate game-level block bootstrap SE on `ridge`; tree: SD over 5 seed-varied refits |

Universe: the pre-registration's **D-I, non-truncated, `pbp_complete`** games -- 19,397 of the 22,414 D-I non-truncated games over [2022, 2023, 2024, 2025] (86.54%). That is STRICTER than the universe the rebound and free-throw bake-offs ran on, so row counts here are not comparable to theirs.

2,241,195 field-goal attempts, of which 2,241,063 are modelled; 132 (0.0059%) carry no shooter id on the row and are dropped rather than imputed.

The two population rules the pre-registration states, as measured rather than asserted: **blocked shots are misses** -- 123,222 attempts carry an adjacent block row and 52 of them (0.0422% of blocks) are logged as MADE, a feed artefact that is left exactly as the feed has it and never patched; **and-one attempts are makes** -- 59,467 attempts (2.653% of all, make rate 1.0) match the and-one signature and are kept as the makes they are. Both `blocked` and `and_one` are POST-OUTCOME and are in `fg_make.BANNED_FEATURES`; the assisted flag the pre-registration excludes is not built at all.

### 2.1 The attempt population by season

| season | n_attempts | n_games | attempts_per_game | FGA_rim_share_pct | FGA_jump2_share_pct | FGA_3_share_pct | FGA_rim_make_pct | FGA_jump2_make_pct | FGA_3_make_pct | efg_pct | espn_id_coverage_pct | position_coverage_pct | on_floor_complete_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 488659 | 4254 | 114.87 | 35.126 | 27.032 | 37.842 | 58.952 | 37.731 | 33.59 | 49.974 | 0.0 | 38.24 | 0.0 |
| 2023 | 516647 | 4520 | 114.3 | 35.882 | 26.664 | 37.454 | 58.85 | 38.326 | 34.001 | 50.438 | 0.0 | 65.98 | 0.0 |
| 2024 | 604454 | 5178 | 116.74 | 37.227 | 25.428 | 37.345 | 58.133 | 38.873 | 33.865 | 50.496 | 100.0 | 99.98 | 92.37 |
| 2025 | 631435 | 5445 | 115.97 | 37.289 | 23.611 | 39.1 | 58.457 | 39.181 | 33.803 | 50.874 | 100.0 | 99.99 | 98.46 |

The ESPN-id column is why shooter identity is keyed on the CBBD player id and why the pre-registered `minutes-to-date` feature could not be built: minutes live in hoopR `player_box` behind the ESPN athlete id, and the crosswalk that reaches them resolves 0% of two of the three training seasons. `shooter_games_asof` and `shooter_fga_asof` carry the exposure instead; `features.md` section 3 records the substitution.

### 2.2 By shot class

| shot_class | n | share_of_attempts_pct | make_pct | blocked_pct | and_one_pct_of_makes | continuation_pct | transition_pct | shooter_no_prior_attempt_pct | n_shooters |
|---|---|---|---|---|---|---|---|---|---|
| FGA_rim | 817509 | 36.476 | 58.561 | 10.523 | 9.824 | 18.93 | 21.001 | 5.374 | 9144 |
| FGA_jump2 | 572643 | 25.551 | 38.558 | 5.073 | 4.741 | 10.097 | 9.048 | 5.983 | 8933 |
| FGA_3 | 851043 | 37.973 | 33.818 | 0.957 | 0.684 | 10.026 | 16.458 | 5.203 | 9092 |

Chance-state derivation (the `C_plus_state` block): mean elapsed time at release 14.721 s, 0.0273% of attempts hit the 60 s clip, 16.222% are transition. The derivation is validated against the feed rather than trusted: the offence implied by the chance's own start event matches the shooter's team on 99.319% of attempts (98.576% have an implied offence at all). Chance-number distribution: {'1': 1943293, '2': 258226, '3': 34200, '4': 4692, '5': 650, '6': 105, '7': 22, '8': 5}.

---

## 3. Full results, per class

`ridge(diagnostic)` is logistic ridge on `B_plus_shooter` -- reported so the shooter block and the state block are separable, NOT a decision arm (the pre-registration's ridge arm is the full bundle).

### 3.1 `FGA_rim`

**F1** -- train [2022, 2023], test [2024]

| arm | feature_set | n | log_loss | brier | pred_make_pct | actual_make_pct | calib | worst_gap_pp | level_pp | shape_pp | respons | steps_shooter_make_c | slope_shooter_make_c | steps_def_allow_c | slope_def_allow_c | chancegap_first | chancegap_continuation | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 225020 | 0.645369 | 0.227516 | 58.53 | 58.133 | PASS | 1.241 | 0.397 | 1.002 | PASS | 4 | 0.9502 | 4 | 1.0423 | 0.271 | 0.944 | 15.6 |
| ridge | C_plus_state | 225020 | 0.662124 | 0.234895 | 58.171 | 58.133 | PASS | 0.967 | 0.038 | 1.005 | PASS | 4 | 0.7882 | 4 | 0.9448 | 0.046 | 0.006 | 1.0 |
| ridge(diagnostic) | B_plus_shooter | 225020 | 0.671805 | 0.239504 | 58.048 | 58.133 | PASS | 1.106 | 0.085 | 1.021 | PASS | 4 | 0.7904 | 4 | 0.9829 | 0.604 | 3.083 | 0.8 |
| eb_shrink | eb | 225020 | 0.671957 | 0.239588 | 58.84 | 58.133 | PASS | 1.365 | 0.71 | 0.844 | PASS | 4 | 1.0029 | 4 | 1.0582 | 1.322 | 1.966 | 0.6 |
| team_baseline | A_team | 225020 | 0.677113 | 0.24206 | 58.778 | 58.133 | PASS | 1.564 | 0.645 | 1.486 | PASS | 4 | 0.1877 | 4 | 0.9674 | 1.214 | 1.829 | 2.4 |

**F2** -- train [2022, 2023, 2024], test [2025]  (SELECTION)

| arm | feature_set | n | log_loss | brier | pred_make_pct | actual_make_pct | calib | worst_gap_pp | level_pp | shape_pp | respons | steps_shooter_make_c | slope_shooter_make_c | steps_def_allow_c | slope_def_allow_c | chancegap_first | chancegap_continuation | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 235454 | 0.641605 | 0.225818 | 58.344 | 58.456 | PASS | 1.186 | 0.112 | 1.298 | PASS | 4 | 1.0075 | 4 | 0.9992 | 0.174 | 0.152 | 25.9 |
| ridge | C_plus_state | 235454 | 0.659766 | 0.23375 | 58.438 | 58.456 | PASS | 1.387 | 0.019 | 1.369 | PASS | 4 | 0.8357 | 4 | 0.9177 | 0.005 | 0.124 | 1.2 |
| ridge(diagnostic) | B_plus_shooter | 235454 | 0.670031 | 0.238648 | 58.483 | 58.456 | PASS | 1.493 | 0.026 | 1.512 | PASS | 4 | 0.8344 | 4 | 0.9719 | 0.727 | 3.001 | 1.1 |
| eb_shrink | eb | 235454 | 0.670433 | 0.238823 | 59.093 | 58.456 | PASS | 1.427 | 0.637 | 1.162 | PASS | 4 | 0.999 | 4 | 0.9988 | 1.262 | 2.07 | 0.6 |
| team_baseline | A_team | 235454 | 0.675996 | 0.241512 | 57.748 | 58.456 | FAIL | 2.232 | 0.708 | 1.524 | PASS | 4 | 0.1795 | 4 | 0.9874 | 0.119 | 3.253 | 0.9 |

Noise floor: game-block bootstrap SE 0.000539 on `ridge`; LightGBM seed-refit SD 0.000211 over seeds [0, 1, 2, 3, 4] ([0.641605, 0.641637, 0.641562, 0.641985, 0.641982]). The floor the decision rule uses is the larger, **0.000539**.

### 3.2 `FGA_jump2`

**F1** -- train [2022, 2023], test [2024]

| arm | feature_set | n | log_loss | brier | pred_make_pct | actual_make_pct | calib | worst_gap_pp | level_pp | shape_pp | respons | steps_shooter_make_c | slope_shooter_make_c | steps_def_allow_c | slope_def_allow_c | chancegap_first | chancegap_continuation | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 153657 | 0.643098 | 0.226654 | 37.986 | 38.883 | PASS | 1.934 | 0.897 | 1.261 | FAIL | 3 | 0.8373 | 4 | 1.0573 | 0.925 | 0.644 | 13.4 |
| ridge | C_plus_state | 153657 | 0.659401 | 0.233445 | 38.373 | 38.883 | FAIL | 2.146 | 0.509 | 1.636 | PASS | 4 | 0.728 | 4 | 1.057 | 0.516 | 0.447 | 0.9 |
| ridge(diagnostic) | B_plus_shooter | 153657 | 0.665282 | 0.236256 | 38.109 | 38.883 | PASS | 1.565 | 0.774 | 1.132 | PASS | 4 | 0.7457 | 4 | 1.0316 | 0.846 | 0.126 | 0.5 |
| eb_shrink | eb | 153657 | 0.665743 | 0.23647 | 38.483 | 38.883 | PASS | 1.206 | 0.4 | 1.097 | PASS | 4 | 0.8215 | 4 | 1.2943 | 0.48 | 0.319 | 0.4 |
| team_baseline | A_team | 153657 | 0.667283 | 0.237195 | 38.879 | 38.883 | PASS | 1.011 | 0.003 | 1.015 | PASS | 4 | 0.1279 | 4 | 0.9893 | 0.12 | 1.045 | 0.5 |

**F2** -- train [2022, 2023, 2024], test [2025]  (SELECTION)

| arm | feature_set | n | log_loss | brier | pred_make_pct | actual_make_pct | calib | worst_gap_pp | level_pp | shape_pp | respons | steps_shooter_make_c | slope_shooter_make_c | steps_def_allow_c | slope_def_allow_c | chancegap_first | chancegap_continuation | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 149075 | 0.642003 | 0.226228 | 38.791 | 39.178 | PASS | 1.58 | 0.386 | 1.652 | PASS | 4 | 0.986 | 4 | 0.8281 | 0.428 | 0.027 | 19.3 |
| ridge | C_plus_state | 149075 | 0.660729 | 0.234095 | 39.993 | 39.178 | PASS | 1.648 | 0.816 | 1.389 | PASS | 4 | 0.8235 | 4 | 0.9657 | 0.765 | 1.252 | 0.9 |
| ridge(diagnostic) | B_plus_shooter | 149075 | 0.666457 | 0.236842 | 40.007 | 39.178 | FAIL | 2.386 | 0.83 | 1.557 | PASS | 4 | 0.8359 | 4 | 1.0508 | 0.733 | 1.656 | 0.7 |
| eb_shrink | eb | 149075 | 0.667461 | 0.2373 | 39.52 | 39.178 | PASS | 1.341 | 0.342 | 0.998 | PASS | 4 | 0.8394 | 4 | 1.3876 | 0.23 | 1.305 | 0.4 |
| team_baseline | A_team | 149075 | 0.668211 | 0.237655 | 39.469 | 39.178 | PASS | 1.503 | 0.291 | 1.211 | PASS | 4 | 0.1417 | 4 | 1.0283 | 0.156 | 1.443 | 0.6 |

Noise floor: game-block bootstrap SE 0.000578 on `ridge`; LightGBM seed-refit SD 0.000124 over seeds [0, 1, 2, 3, 4] ([0.642003, 0.642178, 0.642281, 0.642092, 0.641986]). The floor the decision rule uses is the larger, **0.000578**.

### 3.3 `FGA_3`

**F1** -- train [2022, 2023], test [2024]

| arm | feature_set | n | log_loss | brier | pred_make_pct | actual_make_pct | calib | worst_gap_pp | level_pp | shape_pp | respons | steps_shooter_make_c | slope_shooter_make_c | steps_def_allow_c | slope_def_allow_c | chancegap_first | chancegap_continuation | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 225729 | 0.5624 | 0.191512 | 33.914 | 33.865 | PASS | 1.096 | 0.049 | 1.046 | PASS | 4 | 0.9687 | 4 | 1.9813 | 0.041 | 0.847 | 15.2 |
| ridge | C_plus_state | 225729 | 0.589726 | 0.202178 | 33.845 | 33.865 | PASS | 1.812 | 0.02 | 1.832 | PASS | 4 | 0.9494 | 4 | 2.0729 | 0.078 | 0.491 | 1.0 |
| eb_shrink | eb | 225729 | 0.600482 | 0.207327 | 33.676 | 33.865 | PASS | 1.688 | 0.189 | 1.877 | FAIL | 4 | 1.0515 | 3 | 1.4527 | 0.158 | 0.462 | 0.5 |
| ridge(diagnostic) | B_plus_shooter | 225729 | 0.601354 | 0.207369 | 33.565 | 33.865 | FAIL | 2.727 | 0.3 | 3.026 | PASS | 4 | 0.945 | 4 | 2.3077 | 0.277 | 0.502 | 0.9 |
| team_baseline | A_team | 225729 | 0.639608 | 0.223731 | 34.434 | 33.865 | PASS | 1.14 | 0.569 | 0.571 | PASS | 4 | 0.0098 | 4 | 2.4229 | 0.625 | 0.073 | 0.8 |

**F2** -- train [2022, 2023, 2024], test [2025]  (SELECTION)

| arm | feature_set | n | log_loss | brier | pred_make_pct | actual_make_pct | calib | worst_gap_pp | level_pp | shape_pp | respons | steps_shooter_make_c | slope_shooter_make_c | steps_def_allow_c | slope_def_allow_c | chancegap_first | chancegap_continuation | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 246885 | 0.561085 | 0.190895 | 34.129 | 33.803 | PASS | 1.007 | 0.326 | 1.113 | FAIL | 4 | 0.9878 | 3 | 0.4736 | 0.284 | 0.683 | 18.3 |
| ridge | C_plus_state | 246885 | 0.58837 | 0.201554 | 35.132 | 33.803 | FAIL | 3.177 | 1.329 | 1.848 | FAIL | 4 | 0.9704 | 3 | 0.7097 | 1.261 | 1.905 | 1.3 |
| eb_shrink | eb | 246885 | 0.599327 | 0.206784 | 33.759 | 33.803 | FAIL | 2.043 | 0.044 | 2.087 | PASS | 4 | 1.0663 | 4 | 0.8256 | 0.013 | 0.305 | 0.6 |
| ridge(diagnostic) | B_plus_shooter | 246885 | 0.599614 | 0.206576 | 35.062 | 33.803 | FAIL | 3.751 | 1.259 | 2.492 | PASS | 4 | 0.9733 | 4 | 0.9947 | 1.265 | 1.208 | 1.1 |
| team_baseline | A_team | 246885 | 0.639094 | 0.223488 | 34.034 | 33.803 | PASS | 1.048 | 0.23 | 0.817 | PASS | 4 | 0.0086 | 4 | 1.0117 | 0.273 | 0.136 | 1.0 |

Noise floor: game-block bootstrap SE 0.000849 on `ridge`; LightGBM seed-refit SD 8.9e-05 over seeds [0, 1, 2, 3, 4] ([0.561085, 0.560962, 0.561145, 0.561145, 0.56119]). The floor the decision rule uses is the larger, **0.000849**.

---

## 4. The LightGBM parameter ladder (F1 ONLY)

The pre-registration allows a parameter search on F1 and nowhere else. The winner per class is frozen before F2 is touched and is used for every later LightGBM fit (F2, the seed refits, the lineup fold).

**`FGA_rim`** -- winner rung 1 (`{"num_leaves": 31, "min_child_samples": 200}`), F1 log loss 0.645369

| rung | override | num_leaves | min_child_samples | n_estimators | learning_rate | reg_lambda | F1_log_loss | fit_s |
|---|---|---|---|---|---|---|---|---|
| 0 | (base) | 63 | 400 | 400 | 0.06 | 1.0 | 0.646132 | 7.3 |
| 1 | {"num_leaves": 31, "min_child_samples": 200} | 31 | 200 | 400 | 0.06 | 1.0 | 0.645369 | 3.3 |
| 2 | {"num_leaves": 127, "min_child_samples": 800} | 127 | 800 | 400 | 0.06 | 1.0 | 0.648375 | 8.1 |
| 3 | {"n_estimators": 800, "learning_rate": 0.03} | 63 | 400 | 800 | 0.03 | 1.0 | 0.645448 | 9.6 |
| 4 | {"num_leaves": 15, "min_child_samples": 100, "n_estimators": 800, "learning_rate": 0.03} | 15 | 100 | 800 | 0.03 | 1.0 | 0.646236 | 6.6 |
| 5 | {"reg_lambda": 20.0, "num_leaves": 63, "min_child_samples": 800} | 63 | 800 | 400 | 0.06 | 20.0 | 0.646244 | 7.1 |

**`FGA_jump2`** -- winner rung 1 (`{"num_leaves": 31, "min_child_samples": 200}`), F1 log loss 0.643098

| rung | override | num_leaves | min_child_samples | n_estimators | learning_rate | reg_lambda | F1_log_loss | fit_s |
|---|---|---|---|---|---|---|---|---|
| 0 | (base) | 63 | 400 | 400 | 0.06 | 1.0 | 0.644331 | 4.2 |
| 1 | {"num_leaves": 31, "min_child_samples": 200} | 31 | 200 | 400 | 0.06 | 1.0 | 0.643098 | 4.7 |
| 2 | {"num_leaves": 127, "min_child_samples": 800} | 127 | 800 | 400 | 0.06 | 1.0 | 0.64716 | 17.3 |
| 3 | {"n_estimators": 800, "learning_rate": 0.03} | 63 | 400 | 800 | 0.03 | 1.0 | 0.643807 | 25.0 |
| 4 | {"num_leaves": 15, "min_child_samples": 100, "n_estimators": 800, "learning_rate": 0.03} | 15 | 100 | 800 | 0.03 | 1.0 | 0.643133 | 5.1 |
| 5 | {"reg_lambda": 20.0, "num_leaves": 63, "min_child_samples": 800} | 63 | 800 | 400 | 0.06 | 20.0 | 0.644814 | 5.4 |

**`FGA_3`** -- winner rung 1 (`{"num_leaves": 31, "min_child_samples": 200}`), F1 log loss 0.5624

| rung | override | num_leaves | min_child_samples | n_estimators | learning_rate | reg_lambda | F1_log_loss | fit_s |
|---|---|---|---|---|---|---|---|---|
| 0 | (base) | 63 | 400 | 400 | 0.06 | 1.0 | 0.563037 | 5.9 |
| 1 | {"num_leaves": 31, "min_child_samples": 200} | 31 | 200 | 400 | 0.06 | 1.0 | 0.5624 | 5.0 |
| 2 | {"num_leaves": 127, "min_child_samples": 800} | 127 | 800 | 400 | 0.06 | 1.0 | 0.564971 | 21.3 |
| 3 | {"n_estimators": 800, "learning_rate": 0.03} | 63 | 400 | 800 | 0.03 | 1.0 | 0.562812 | 25.2 |
| 4 | {"num_leaves": 15, "min_child_samples": 100, "n_estimators": 800, "learning_rate": 0.03} | 15 | 100 | 800 | 0.03 | 1.0 | 0.563151 | 10.3 |
| 5 | {"reg_lambda": 20.0, "num_leaves": 63, "min_child_samples": 800} | 63 | 800 | 400 | 0.06 | 20.0 | 0.563496 | 11.9 |

---

## 5. The fitted shrinkage, per class

`eb_shrink` is `logit(p) = logit(shooter_EB) + [logit(defence_allowed_EB) - logit(league_asof)]`: the defence enters as its log-odds deviation from the league on this shot class, so a league-average defence contributes exactly zero. Both strengths are fitted on the training fold (the defence's too -- its as-of rate rests on a handful of attempts in November and thousands in March).

| shot_class | F1 prior | F1 m | F1 m_def | F2 prior | F2 m | F2 m_def | median shooter attempts at prediction time (F2) | % of F2 attempts where the shooter's own rate outweighs the prior |
|---|---|---|---|---|---|---|---|---|
| FGA_rim | position | 30.0 | 300.0 | position | 30.0 | 300.0 | 36.0 | 55.36 |
| FGA_jump2 | position | 75.0 | 300.0 | position | 75.0 | 300.0 | 25.0 | 12.57 |
| FGA_3 | position | 5.0 | 1000.0 | position | 5.0 | 1000.0 | 42.0 | 88.59 |

- `FGA_rim`: at m = 30.0 attempts a shooter's own rate carries half the weight at 30 attempts, three quarters at 90 and nine tenths at 270.
- `FGA_jump2`: at m = 75.0 attempts a shooter's own rate carries half the weight at 75 attempts, three quarters at 225 and nine tenths at 675.
- `FGA_3`: at m = 5.0 attempts a shooter's own rate carries half the weight at 5 attempts, three quarters at 15 and nine tenths at 45.

Full grid (train log loss per prior x shooter strength x defence strength) is in `data/processed/models/fg_make/run_report.json` under `eb_fits`.

### 5.1 Transfer subset

The L15 natural experiment: players whose modal team changed since the prior season, where a prior-season-based prior is least trustworthy.

**`FGA_rim`**

| subset | n | actual_make_rate | eb_pred_make_rate | team_baseline_log_loss | eb_shrink_log_loss | ridge_log_loss | lgbm_log_loss |
|---|---|---|---|---|---|---|---|
| transfer | 73708 | 0.5927 | 0.59303 | 0.673211 | 0.668588 | 0.657543 | 0.639601 |
| non_transfer | 105506 | 0.59166 | 0.59344 | 0.673854 | 0.667922 | 0.656902 | 0.639011 |
| no_prior_season | 56240 | 0.56058 | 0.58345 | 0.683666 | 0.677561 | 0.66805 | 0.649098 |

**`FGA_jump2`**

| subset | n | actual_make_rate | eb_pred_make_rate | team_baseline_log_loss | eb_shrink_log_loss | ridge_log_loss | lgbm_log_loss |
|---|---|---|---|---|---|---|---|
| transfer | 46278 | 0.39332 | 0.39341 | 0.669326 | 0.668343 | 0.660996 | 0.642498 |
| non_transfer | 66332 | 0.40148 | 0.39739 | 0.671961 | 0.67138 | 0.664122 | 0.644626 |
| no_prior_season | 36465 | 0.37217 | 0.39349 | 0.659974 | 0.659212 | 0.654216 | 0.636604 |

**`FGA_3`**

| subset | n | actual_make_rate | eb_pred_make_rate | team_baseline_log_loss | eb_shrink_log_loss | ridge_log_loss | lgbm_log_loss |
|---|---|---|---|---|---|---|---|
| transfer | 74842 | 0.34467 | 0.33878 | 0.6437 | 0.606978 | 0.595629 | 0.568334 |
| non_transfer | 111801 | 0.34841 | 0.34624 | 0.64562 | 0.607276 | 0.59289 | 0.565914 |
| no_prior_season | 60242 | 0.31051 | 0.32007 | 0.621263 | 0.575072 | 0.570965 | 0.543117 |

---

## 6. G4: the implied team eFG%, from the three class models on the test season's own shot mix

eFG% = (FGM + 0.5 x 3PM) / FGA, with every attempt the team actually took weighted by its class model's predicted make probability instead of by the outcome -- so the shot MIX is taken as given and this is a check of the make models alone. Teams with fewer than 200 attempts in the test season are excluded. `asof` terciles bucket teams by their own pregame as-of form (the honest, matchup-specific grouping); `actual` terciles bucket by realised eFG% and are an ORACLE grouping, labelled as such, reported because it is the one that exposes a flat model. Gate: +/- 1.0 pp.

**offense**

| arm | n_teams | actual eFG% | implied eFG% | overall gap pp | team MAE pp | team corr | T1 gap pp (asof) | T2 gap pp (asof) | T3 gap pp (asof) | worst gap pp (asof) | G4 (asof) | T1 gap pp (actual) | T2 gap pp (actual) | T3 gap pp (actual) | worst gap pp (actual) | G4 (actual) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| team_baseline | 364 | 50.874 | 50.814 | -0.06 | 1.433 | 0.8046 | 1.06 | -0.029 | -1.123 | 1.123 | FAIL | 1.413 | -0.017 | -1.464 | 1.464 | FAIL |
| eb_shrink | 364 | 50.874 | 51.166 | 0.292 | 1.31 | 0.8532 | 1.199 | 0.355 | -0.602 | 1.199 | FAIL | 1.436 | 0.391 | -0.859 | 1.436 | FAIL |
| ridge | 364 | 50.874 | 51.839 | 0.965 | 1.932 | 0.7069 | 1.949 | 1.003 | 0.022 | 1.949 | FAIL | 2.022 | 1.167 | -0.204 | 2.022 | FAIL |
| lgbm | 364 | 50.874 | 50.932 | 0.058 | 1.614 | 0.7746 | 0.711 | 0.076 | -0.56 | 0.711 | PASS | 0.74 | 0.238 | -0.743 | 0.743 | PASS |

**defense**

| arm | n_teams | actual eFG% | implied eFG% | overall gap pp | team MAE pp | team corr | T1 gap pp (asof) | T2 gap pp (asof) | T3 gap pp (asof) | worst gap pp (asof) | G4 (asof) | T1 gap pp (actual) | T2 gap pp (actual) | T3 gap pp (actual) | worst gap pp (actual) | G4 (actual) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| team_baseline | 364 | 50.874 | 50.814 | -0.06 | 1.228 | 0.8231 | -0.874 | -0.237 | 0.853 | 0.874 | PASS | 1.163 | -0.199 | -1.271 | 1.271 | FAIL |
| eb_shrink | 364 | 50.874 | 51.166 | 0.292 | 1.443 | 0.7663 | -1.132 | 0.133 | 1.747 | 1.747 | FAIL | 1.954 | 0.217 | -1.47 | 1.954 | FAIL |
| ridge | 364 | 50.874 | 51.839 | 0.965 | 1.531 | 0.7869 | 0.346 | 0.649 | 1.833 | 1.833 | FAIL | 2.019 | 0.754 | 0.019 | 2.019 | FAIL |
| lgbm | 364 | 50.874 | 50.932 | 0.058 | 1.448 | 0.7606 | -0.657 | -0.154 | 0.914 | 0.914 | PASS | 1.03 | -0.11 | -0.843 | 1.03 | FAIL |

---

## 7. Team form from ALL chances vs FIRST chances only

The rebound model restricts its team rates to first-chance opportunities (ledger row B5: pooling continuation chances lets a continuation-chance labelling defect into a first-chance model's predictors). Here the target population is ALL attempts and the L16 defect is repaired at the source by the v2 override, so the default is all chances -- and the cost of that choice is measured rather than argued.

| shot_class | ridge_log_loss_first_chance_form | ridge_log_loss_all_chance_form | delta | delta_in_floors | off_make_c_correlation_between_sources |
|---|---|---|---|---|---|
| FGA_rim | 0.659766 | 0.659766 | -0.0 | -0.0 | 0.94925 |
| FGA_jump2 | 0.66072 | 0.660729 | -9e-06 | -0.02 | 0.95844 |
| FGA_3 | 0.588391 | 0.58837 | 2.1e-05 | 0.02 | 0.95644 |

---

## 8. Is the DEFENCE lineup-level or team-level? (`D_plus_lineup` on its own fold)

Fold L2 (train 2024, test 2025) (CBBD `onFloor` is empty at the source before 2024, L13). 1,202,616 of the fold's attempts (97.31%) carry all ten on-floor ids; C and D are scored on exactly those rows so the comparison is like-for-like. The defender shrinkage strength is FITTED on the train season over [0, 100, 250, 500, 1000] pseudo-attempts (L13) and landed on **100**.

| prior_att | train_log_loss | fit_s |
|---|---|---|
| 0 | 0.633755 | 8.3 |
| 100 | 0.633694 | 8.7 |
| 250 | 0.633694 | 8.9 |
| 500 | 0.633697 | 8.5 |
| 1000 | 0.633702 | 8.9 |

**`FGA_rim`** -- fold block-bootstrap floor 0.000499; ridge gain D over C 0.000149 (0.3x the floor), lgbm gain -8e-05 (-0.16x) -> **TEAM-LEVEL**

| arm | feature_set | n | log_loss | brier | calib | worst_gap_pp | respons | resp_min_steps | fit_s |
|---|---|---|---|---|---|---|---|---|---|
| ridge | C_plus_state | 233217 | 0.659684 | 0.2337 | PASS | 1.609 | PASS | 4 | 0.8 |
| ridge | D_plus_lineup | 233217 | 0.659535 | 0.233632 | PASS | 1.646 | PASS | 4 | 0.9 |
| lgbm | C_plus_state | 233217 | 0.643931 | 0.226791 | PASS | 0.477 | PASS | 4 | 12.9 |
| lgbm | D_plus_lineup | 233217 | 0.644011 | 0.226847 | PASS | 0.678 | PASS | 4 | 10.1 |

**`FGA_jump2`** -- fold block-bootstrap floor 0.000693; ridge gain D over C -4e-06 (-0.01x the floor), lgbm gain -6.8e-05 (-0.1x) -> **TEAM-LEVEL**

| arm | feature_set | n | log_loss | brier | calib | worst_gap_pp | respons | resp_min_steps | fit_s |
|---|---|---|---|---|---|---|---|---|---|
| ridge | C_plus_state | 147682 | 0.660415 | 0.233944 | PASS | 1.056 | PASS | 4 | 0.6 |
| ridge | D_plus_lineup | 147682 | 0.660419 | 0.233946 | PASS | 1.241 | PASS | 4 | 0.6 |
| lgbm | C_plus_state | 147682 | 0.644776 | 0.227412 | FAIL | 2.019 | PASS | 4 | 7.7 |
| lgbm | D_plus_lineup | 147682 | 0.644844 | 0.227406 | FAIL | 2.364 | PASS | 4 | 7.6 |

**`FGA_3`** -- fold block-bootstrap floor 0.000802; ridge gain D over C 2.4e-05 (0.03x the floor), lgbm gain -0.000655 (-0.82x) -> **TEAM-LEVEL**

| arm | feature_set | n | log_loss | brier | calib | worst_gap_pp | respons | resp_min_steps | fit_s |
|---|---|---|---|---|---|---|---|---|---|
| ridge | C_plus_state | 244634 | 0.588068 | 0.201443 | FAIL | 2.573 | FAIL | 3 | 0.9 |
| ridge | D_plus_lineup | 244634 | 0.588044 | 0.201434 | FAIL | 2.556 | FAIL | 3 | 0.9 |
| lgbm | C_plus_state | 244634 | 0.56309 | 0.19172 | PASS | 1.015 | FAIL | 2 | 8.7 |
| lgbm | D_plus_lineup | 244634 | 0.563745 | 0.191954 | PASS | 1.171 | FAIL | 2 | 8.9 |

**Verdict: TEAM-LEVEL.**

---

## 9. Decision, by the pre-registered rule, per class

An arm passes only if its worst gated decile calibration gap is <= 2.00 pp AND BOTH responsiveness drivers -- the shooter's as-of class make rate and the defence's as-of allowed rate -- are monotone in 4 of 4 quintile steps.

### `FGA_rim`

3 of 4 F2 arms pass both gates: `eb_shrink`, `ridge`, `lgbm`.

The tree arm beats the best passing non-tree arm (`ridge`) by 0.018161 log loss = 33.69x the noise floor (0.000539).

**WINNER (FGA_rim): lgbm**, F2 log loss 0.641605.

### `FGA_jump2`

4 of 4 F2 arms pass both gates: `team_baseline`, `eb_shrink`, `ridge`, `lgbm`.

The tree arm beats the best passing non-tree arm (`ridge`) by 0.018726 log loss = 32.4x the noise floor (0.000578).

**WINNER (FGA_jump2): lgbm**, F2 log loss 0.642003.

### `FGA_3`

1 of 4 F2 arms pass both gates: `team_baseline`.

**WINNER (FGA_3): team_baseline**, F2 log loss 0.639094.

Runtime 10.8 min.

## 10. G4 on the adopted trio (appended after the decision, same run)

Section 6 holds one arm fixed across all three classes, which is how arms are compared but is not the engine's model when the classes choose differently. The trio the decision rule actually adopted is `FGA_rim` -> `lgbm`, `FGA_jump2` -> `lgbm`, `FGA_3` -> `team_baseline`. Refit on F2 train, scored on the same F2 test rows and the same actual shot mix:

| side | n_teams | actual eFG% | implied eFG% | overall gap pp | team MAE pp | team corr | T1 gap pp (asof) | T2 gap pp (asof) | T3 gap pp (asof) | worst gap pp (asof) | G4 (asof) | T1 gap pp (actual) | T2 gap pp (actual) | T3 gap pp (actual) | worst gap pp (actual) | G4 (actual) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| offense | 364 | 50.874 | 50.876 | 0.002 | 1.389 | 0.8191 | 0.981 | -0.0 | -0.899 | 0.981 | PASS | 1.207 | 0.058 | -1.165 | 1.207 | FAIL |
| defense | 364 | 50.874 | 50.876 | 0.002 | 1.202 | 0.8338 | -0.755 | -0.177 | 0.865 | 0.865 | PASS | 1.085 | -0.126 | -1.063 | 1.085 | FAIL |

---

## 11. Interpretation -- what the data said (hand-written, same run, 2026-09-10)

Appended by the worker after reading sections 2-10. No arm was re-run, re-tuned
or dropped to produce anything below; the decision rule's own answers stand as
rendered in section 9.

### 11.1 The `FGA_3` gate, in full

The rule's answer for threes is `team_baseline` because it is the only arm that
passes both gates. Here are the two quintile ladders that produced that, from
`run_report.json` under `detail["F2|FGA_3|<arm>"]["responsiveness"]`
(predicted share / actual share per quintile):

**Driver: the shooter's as-of three-point rate** (realised span 35.90 pp)

| arm | Q1 | Q2 | Q3 | Q4 | Q5 | steps | slope |
|---|---|---|---|---|---|---|---|
| actual | 15.10 | 26.59 | 35.16 | 41.42 | 51.00 | 4/4 | — |
| `lgbm` | 15.58 | 27.13 | 35.66 | 41.50 | 51.04 | 4/4 | 0.988 |
| `eb_shrink` | 14.81 | 25.18 | 34.16 | 41.77 | 53.09 | 4/4 | 1.066 |
| `ridge` | 18.62 | 27.67 | 35.03 | 41.14 | 53.46 | 4/4 | 0.970 |
| `team_baseline` | 33.92 | 33.92 | 33.99 | 34.10 | 34.23 | 4/4 | **0.009** |

**Driver: the defence's as-of three-point rate allowed** (realised span 1.37 pp)

| arm | Q1 | Q2 | Q3 | Q4 | Q5 | steps | slope |
|---|---|---|---|---|---|---|---|
| actual | 32.95 | 33.65 | 34.06 | 34.21 | 34.32 | 4/4 | — |
| `lgbm` | 33.77 | 34.33 | 34.08 | 34.24 | 34.43 | **3/4** | 0.474 |
| `ridge` | 34.66 | 35.22 | 35.05 | 35.28 | 35.64 | **3/4** | 0.710 |
| `eb_shrink` | 33.19 | 33.56 | 33.78 | 34.19 | 34.33 | 4/4 | 0.826 |
| `team_baseline` | 33.34 | 33.76 | 34.01 | 34.28 | 34.73 | 4/4 | 1.012 |

Two separate things are visible.

1. **The defence driver on threes is nearly signal-free.** Its realised span is
   1.37 pp, against 4.98 pp for `FGA_rim` and 3.33 pp for `FGA_jump2` (same
   tables for those classes in `run_report.json`). A 4-of-4 monotone requirement
   on a 1.37 pp true span asks an arm to reproduce noise at a single interior
   step; `lgbm` and `ridge` each dip at exactly one. The two arms that clear it
   are the two that are (near-)monotone FUNCTIONS of the driver by construction:
   the linear team baseline, and `eb_shrink`, which puts the defence's allowed
   rate straight into the logit.
2. **The steps-only reading of responsiveness cannot see flatness.**
   `team_baseline` passes the SHOOTER check with a 0.31 pp predicted span
   against a 35.90 pp realised one -- a slope ratio of 0.0086, 116x too flat,
   and a model that cannot separate a 15% shooter from a 51% one. That is the
   "flat at the mean" model `CLAUDE.md`'s matchup rule exists to reject.
   `prob_metrics.quintile_responsiveness` says so in its own docstring; the
   pre-registration gates on the steps and reports the slope.

Under the rule as written: **`FGA_3` -> `team_baseline`, F2 log loss 0.639094.**
Under any reading that also gates the slope ratio (e.g. 0.5-1.5 on both
drivers): **`FGA_3` -> `lgbm`, F2 log loss 0.561085**, which is 0.078 log loss
and 92 noise floors better. The difference is a PM decision about the gate, not
a worker decision about the model, and both answers are recorded here.

### 11.2 Feature blocks, F2, nested logistic ridge

| class | A -> B (shooter block) | in floors | B -> C (state block) | in floors |
|---|---|---|---|---|
| `FGA_rim` | 0.005965 | 11.1 | 0.010265 | 19.0 |
| `FGA_jump2` | 0.001754 | 3.0 | 0.005728 | 9.9 |
| `FGA_3` | 0.039480 | 46.5 | 0.011244 | 13.2 |

(A is the `team_baseline` arm, B the `ridge(diagnostic)` row, C the `ridge` arm;
the three bundles are strictly nested, which `tests/test_fg_make.py` asserts.)

Both blocks clear the floor in every class and are KEPT. **L15 is a per-class
statement here, not a blanket one**: shooter identity is worth 46.5 floors on
threes and 3.0 on two-point jumpers. The state block's value is concentrated
where the mechanism says it should be -- on `FGA_rim` the team baseline misses
the CONTINUATION-chance (putback) make rate by 3.25 pp and the shooter-only
bundle by 3.00 pp, while the two arms carrying `chance_number` /
`chance_elapsed_s` land at 0.12 and 0.15 pp.

### 11.3 What else the run settled

- **The L16 override matters to this target, as predicted.** The rim share of
  attempts is 35.1 / 35.9 / 37.2 / 37.3% across 2022-2025 under v2 with no
  2025 collapse, and the 2025 three-point share (39.1%) matches the independent
  gate-reference table's 39.2%.
- **Dead-level accuracy, narrow spread.** Every arm has the league eFG% level
  essentially right (overall gap 0.002-0.97 pp) and every arm under-spreads the
  team distribution at the tails (section 6). This is the same too-narrow
  signature the rebound and possession-outcome rounds reported, one layer down.
- **Team-level defence, cleanly** (section 8): 0.30 / -0.01 / 0.03 floors on
  ridge and NEGATIVE on the tree in all three classes. Not the rebound model's
  straddle -- a nothing.
- **The position prior wins the EB grid in all three classes on both folds**,
  as it did at the free-throw line (L18), with the shooter strength fitted at
  30 / 75 / 5 attempts for rim / jumper / three. The three's m = 5 is the
  smallest shrinkage anywhere in the cascade so far: a shooter's own
  three-point rate is trusted after five attempts, and 88.6% of F2 three-point
  attempts are past the crossover.
- **The `minutes-to-date` gap is structural, not an oversight** (section 2.1):
  ESPN-id coverage is 0% in 2022 and 2023 and 100% in 2024 and 2025.

## 12. RE-DECISION under ARCHITECTURE_DECISIONS.md Decision 8 (gate amended 2026-09-10; decision step only, nothing retrained)

Decision 8 supersedes the steps-only responsiveness wording of the pre-registration in section 1. The gate is now: slope ratio within [0.8, 1.2] AND monotone in at least 3 of 4 quintile steps, with the 4-of-4 requirement dropped when the driver's realised quintile span is below 2.0 pp, on every driver. Every log loss, calibration table and responsiveness ladder below is read back out of `data/processed/models/fg_make/run_report.json` exactly as the original run wrote it; only the gate verdict and the decision rule are recomputed.

Realised quintile spans of the two drivers (the quantity the 2 pp clause keys on):

| shot_class | shooter span pp | defence span pp | low-span driver |
|---|---|---|---|
| FGA_rim | 16.084 | 4.977 | none |
| FGA_jump2 | 8.138 | 3.329 | none |
| FGA_3 | 35.899 | 1.373 | def_allow_c |

**The one ambiguity in clause (a), stated rather than resolved silently.** Clause (b) exempts a sub-2 pp driver from the step count because its steps are noise; clause (a) does not say whether the SLOPE on such a driver is exempt too. It bites exactly once -- `FGA_3`'s defence driver spans 1.37 pp and LightGBM's slope on it is 0.474 -- so both readings are computed:

### 12.1 `strict`: the slope band applies to EVERY driver

**`FGA_rim`** -- winner **lgbm**, F2 log loss 0.641605 (floor 0.000539)

| arm | log_loss | calib | shooter steps/slope | defence steps/slope | resp | gate |
|---|---|---|---|---|---|---|
| team_baseline | 0.675996 | FAIL (2.232) | 4/4, 0.1795 | 4/4, 0.9874 | FAIL (shooter_make_c) | FAIL |
| eb_shrink | 0.670433 | PASS | 4/4, 0.999 | 4/4, 0.9988 | PASS | PASS |
| ridge | 0.659766 | PASS | 4/4, 0.8357 | 4/4, 0.9177 | PASS | PASS |
| lgbm | 0.641605 | PASS | 4/4, 1.0075 | 4/4, 0.9992 | PASS | PASS |

The tree arm beats the best passing non-tree arm (`ridge`) by 0.018161 = 33.69x the floor.

**`FGA_jump2`** -- winner **lgbm**, F2 log loss 0.642003 (floor 0.000578)

| arm | log_loss | calib | shooter steps/slope | defence steps/slope | resp | gate |
|---|---|---|---|---|---|---|
| team_baseline | 0.668211 | PASS | 4/4, 0.1417 | 4/4, 1.0283 | FAIL (shooter_make_c) | FAIL |
| eb_shrink | 0.667461 | PASS | 4/4, 0.8394 | 4/4, 1.3876 | FAIL (def_allow_c) | FAIL |
| ridge | 0.660729 | PASS | 4/4, 0.8235 | 4/4, 0.9657 | PASS | PASS |
| lgbm | 0.642003 | PASS | 4/4, 0.986 | 4/4, 0.8281 | PASS | PASS |

The tree arm beats the best passing non-tree arm (`ridge`) by 0.018726 = 32.4x the floor.

**`FGA_3`** -- winner **NONE** (floor 0.000849)

| arm | log_loss | calib | shooter steps/slope | defence steps/slope | resp | gate |
|---|---|---|---|---|---|---|
| team_baseline | 0.639094 | PASS | 4/4, 0.0086 | 4/4, 1.0117 | FAIL (shooter_make_c) | FAIL |
| eb_shrink | 0.599327 | FAIL (2.043) | 4/4, 1.0663 | 4/4, 0.8256 | PASS | FAIL |
| ridge | 0.58837 | FAIL (3.177) | 4/4, 0.9704 | 3/4, 0.7097 | FAIL (def_allow_c) | FAIL |
| lgbm | 0.561085 | PASS | 4/4, 0.9878 | 3/4, 0.4736 | FAIL (def_allow_c) | FAIL |

no arm passes both gates under this reading, so nothing is adopted for this class

### 12.2 `low_span_exempt`: a sub-2 pp driver is noise for BOTH clauses

**`FGA_rim`** -- winner **lgbm**, F2 log loss 0.641605 (floor 0.000539)

| arm | log_loss | calib | shooter steps/slope | defence steps/slope | resp | gate |
|---|---|---|---|---|---|---|
| team_baseline | 0.675996 | FAIL (2.232) | 4/4, 0.1795 | 4/4, 0.9874 | FAIL (shooter_make_c) | FAIL |
| eb_shrink | 0.670433 | PASS | 4/4, 0.999 | 4/4, 0.9988 | PASS | PASS |
| ridge | 0.659766 | PASS | 4/4, 0.8357 | 4/4, 0.9177 | PASS | PASS |
| lgbm | 0.641605 | PASS | 4/4, 1.0075 | 4/4, 0.9992 | PASS | PASS |

The tree arm beats the best passing non-tree arm (`ridge`) by 0.018161 = 33.69x the floor.

**`FGA_jump2`** -- winner **lgbm**, F2 log loss 0.642003 (floor 0.000578)

| arm | log_loss | calib | shooter steps/slope | defence steps/slope | resp | gate |
|---|---|---|---|---|---|---|
| team_baseline | 0.668211 | PASS | 4/4, 0.1417 | 4/4, 1.0283 | FAIL (shooter_make_c) | FAIL |
| eb_shrink | 0.667461 | PASS | 4/4, 0.8394 | 4/4, 1.3876 | FAIL (def_allow_c) | FAIL |
| ridge | 0.660729 | PASS | 4/4, 0.8235 | 4/4, 0.9657 | PASS | PASS |
| lgbm | 0.642003 | PASS | 4/4, 0.986 | 4/4, 0.8281 | PASS | PASS |

The tree arm beats the best passing non-tree arm (`ridge`) by 0.018726 = 32.4x the floor.

**`FGA_3`** -- winner **lgbm**, F2 log loss 0.561085 (floor 0.000849)

| arm | log_loss | calib | shooter steps/slope | defence steps/slope | resp | gate |
|---|---|---|---|---|---|---|
| team_baseline | 0.639094 | PASS | 4/4, 0.0086 | 4/4, 1.0117 | FAIL (shooter_make_c) | FAIL |
| eb_shrink | 0.599327 | FAIL (2.043) | 4/4, 1.0663 | 4/4, 0.8256 | PASS | FAIL |
| ridge | 0.58837 | FAIL (3.177) | 4/4, 0.9704 | 3/4, 0.7097 | PASS | FAIL |
| lgbm | 0.561085 | PASS | 4/4, 0.9878 | 3/4, 0.4736 | PASS | PASS |

no non-tree arm passes both gates, so the 'a tree must beat the best PASSING non-tree arm by more than the floor' clause has nothing to bind against; the gap to the best non-tree arm of any gate status (`ridge`, 0.58837) is 0.027285 = 32.1x the floor

### 12.3 What is adopted

Adopted reading: **`low_span_exempt`**. Decision 8 states the corrected gate changes FGA_3 from team_baseline to LightGBM. That follows only if a sub-2 pp driver is exempt from the SLOPE clause as well as from the step count, because LightGBM's slope on the FGA_3 defence driver is 0.474. Under the strict reading no arm passes FGA_3 and nothing would be adopted for that class. Both are computed above; the wording of clause (a) is what needs tightening, not the models.

| shot_class | pre-registered gate | Decision 8 gate | changed |
|---|---|---|---|
| FGA_rim | lgbm | lgbm | no |
| FGA_jump2 | lgbm | lgbm | no |
| FGA_3 | team_baseline | lgbm | yes |

Artifacts re-exported (the only fits in this step; same fold, same frozen parameters, a serialisation of an arm the grid already scored):

- `FGA_rim`: data\processed\models\fg_make\winner_FGA_rim.joblib (unchanged: lgbm)
- `FGA_jump2`: data\processed\models\fg_make\winner_FGA_jump2.joblib (unchanged: lgbm)
- `FGA_3_superseded`: data\processed\models\fg_make\reference_superseded_FGA_3.joblib (the team_baseline arm, kept for reference)
- `FGA_3`: data\processed\models\fg_make\winner_FGA_3.joblib (re-exported: team_baseline -> lgbm)

### 12.4 G4 on the re-decided trio

Same construction as section 10 -- each class's adopted arm refit on F2 train, scored on the same F2 test rows and the same actual shot mix:

| side | n_teams | actual eFG% | implied eFG% | overall gap pp | team MAE pp | team corr | T1 gap pp (asof) | T2 gap pp (asof) | T3 gap pp (asof) | worst gap pp (asof) | G4 (asof) | T1 gap pp (actual) | T2 gap pp (actual) | T3 gap pp (actual) | worst gap pp (actual) | G4 (actual) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| offense | 364 | 50.874 | 50.932 | 0.058 | 1.614 | 0.7746 | 0.711 | 0.076 | -0.56 | 0.711 | PASS | 0.74 | 0.238 | -0.743 | 0.743 | PASS |
| defense | 364 | 50.874 | 50.932 | 0.058 | 1.448 | 0.7606 | -0.657 | -0.154 | 0.914 | 0.914 | PASS | 1.03 | -0.11 | -0.843 | 1.03 | FAIL |

### 12.5 The code now gates on Decision 8 (so a future re-run cannot re-decide under the old rule)

`cbb_sim.models.fg_make.score` reports BOTH verdicts -- `resp_pass` (the
superseded steps-only reading, so the tables in sections 3 and 9 stay readable)
and `resp_pass_decision8` with its per-driver working -- and
`scripts/train_fg_make_v1.py` now decides on the latter, with both columns
(`respons`, `respons_d8`) written to `grid_results.csv`. The reading is
`fg_make.DECISION8_ADOPTED_READING`; `decision8_verdict(..., reading=...)`
computes either on demand. Six tests in `tests/test_fg_make.py` pin the gate:
the flat arm is rejected, the over-steep arm is rejected, the band edges are
inclusive, the 4-of-4 rule survives on a high-span driver, the two readings
differ only on a sub-2 pp driver, and an unknown reading raises.
