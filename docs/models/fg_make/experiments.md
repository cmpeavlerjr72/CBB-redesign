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

---

## 13. ROUND 2 PRE-REGISTRATION: state parametrisation (written 2026-09-10, BEFORE any round-2 modelling)

Owner: Opus cascade worker. Authority: `ARCHITECTURE_DECISIONS.md` **Decision
10** (closed-loop gate for every engine-produced state feature) and
**Decision 8** (responsiveness gate). Trigger: `docs/LEARNINGS.md` **L23**.
Evidence this spec is built on, committed in the same commit and BEFORE the
round runs: **`docs/tests/fg_make_state_confound_2026-09-10.md`**
(`scripts/diag_fg_make_state_confound.py`,
`data/processed/models/fg_make/state_confound.json`).

Nothing in this section may be edited after the round starts. Results append
below it; a status change goes to `docs/models/change_ledger.md` in the same
commit.

### 13.0 What the step-1 evidence established, and what it changes about this round

1. **The round-1 `score_diff` is POST-OUTCOME.** It is read off
   `homeScore`/`awayScore` on the attempt's own row, and that column is the
   score AFTER the play: a made three already carries its own three points.
   Proved three independent ways off the raw feed (evidence doc section 0);
   94.5 / 94.3 / 93.6 / 93.6% of made field goals move their own team's score
   by exactly the shot's value on their own row, against 99.5% of misses moving
   it by zero. It is the same defect class as `blocked` and `and_one`, which
   this model already bans by name.
2. The leak manufactures **62.0% (rim), 83.3% (jumper), 83.5% (three)** of the
   apparent margin effect. Pregame team strength accounts for a further
   19.1 / 4.1 / 12.0 percentage-points of share. The genuine state effect is
   **3.80 / 2.03 / 0.97 pp** of adjusted make probability across the whole
   +/-20 margin range.
3. The genuine effect is **not** monotone in the margin (U-shaped at the rim,
   minimum at tied) and is **concentrated in the last two minutes and
   overtime**: adjusted span 0.98-3.07 pp in every mid-game minute bucket,
   9.33 pp in H2 2:00-0:00, 14.34 pp in OT.
4. It is **not** a lineup effect: adding the defensive five's as-of allowed
   rates moves the span by at most 0.10 pp on 1,202,616 lineup-complete
   attempts.
5. Two indicators carry it, and both are **flat in their thresholds** across a
   3 x 3 (garbage) and 7-cell (end-game) grid, which is what makes fixing the
   thresholds here a reading of the evidence and not a tuning step.

**Consequence for the arm list.** The round-1 winner is not merely
mis-parametrised, it is trained on a post-outcome column. Round 2 therefore
introduces the corrected feature `score_diff_pre` (the margin BEFORE the
attempt: `score_diff` minus the attempt's own points when it went in; a miss is
unchanged by construction, so the correction can only remove outcome
information, never add it) and every state arm is built on it. The round-1 arm
is kept as arm **S-A** and re-scored so the size of the leak's log-loss
advantage is on the record, but it is **declared INELIGIBLE to win here and
now, before the round runs**, on the data-integrity ground above and not on any
number this round will produce.

**The engine already feeds the correct quantity.** `engine/loop._state_block`
writes `st.off_score_diff()`, the live margin before the shot resolves. So
round 1 shipped a train/serve skew as well as a leak.

### 13.1 What is FIXED and not up for selection

- **Model class: LightGBM per shot class.** Round 1 decided it (33.7 / 32.4
  noise floors; Decision 8 for `FGA_3`). This round does not re-open it.
- **Parameters:** the F1-only frozen ladder winner per class from round 1
  (`lgbm_ladder_v2.json`, rung 1: `num_leaves` 31, `min_child_samples` 200, 400
  trees, learning rate 0.06). **No new parameter search.** F1 is not touched
  for tuning.
- **Training scheme: STATIC**, exactly as round 1, so the arms differ ONLY in
  the state block. L21's in-season refit (S1) is the standing default and is a
  SEPARATE round for this model; mixing it in here would confound the two
  changes.
- **Possessions version v2**, L16 rim override, same universe (D-I,
  non-truncated, `pbp_complete`), same shooter key, same banned features.
- **Folds:** F1 = train {2022, 2023} test 2024 (reported, selects nothing);
  **F2 = train {2022, 2023, 2024} test 2025 = SELECTION**. 2026 sealed
  (`assert_not_sealed` on every slice).

### 13.2 The arms (state parametrisations only)

Every arm is `B_plus_shooter` (team block + shooter block, unchanged from round
1) plus the state block named below.

| arm | state block | why it is in |
|---|---|---|
| **S-A** `R2_A_round1_leaked` | round 1's `C_plus_state`: `period`, `seconds_remaining`, **`score_diff` (post-shot, LEAKED)**, `in_bonus`, `chance_number`, `chance_elapsed_s`, `is_transition_f` | Reference. MUST reproduce round 1's F2 log loss per class. **INELIGIBLE** (13.0) |
| **S-B** `R2_B_no_state` | none (identical to `B_plus_shooter`) | The no-engine-produced-state floor, and the closed-loop REFERENCE condition |
| **S-C** `R2_C_safe_state` | `period`, `seconds_remaining`, `in_bonus`, `chance_number`, `chance_elapsed_s`, `is_transition_f` | Engine-safe state only. Every column is pre-shot by construction and none is a function of the score |
| **S-D** `R2_D_safe_plus_indicators` | S-C + `gt_flag`, `eg_trail`, `eg_lead` (13.3) | The margin enters ONLY through saturating indicators the sim cannot run away with |
| **S-E** `R2_E_safe_plus_continuous` | S-C + `score_diff_pre` (continuous, corrected) | The honest counterpart of S-A: is the corrected continuous margin still unsafe in closed loop? |

`D_plus_lineup` is NOT an arm this round: round 1 answered TEAM-LEVEL on its
own fold (gains 0.3x / -0.01x / 0.03x the floor) and the step-1 evidence adds
that the five on the floor explain at most 0.10 pp of the margin span.
Re-opening it would need its own pre-registration and an engine block that does
not exist.

Simplicity order for tie-breaking: **S-B < S-C < S-D < S-E < S-A**.

### 13.3 The S-D indicators, with their thresholds FIXED HERE from the step-1 evidence

`gsr` = seconds left in REGULATION (`seconds_remaining + 1200` in period 1,
`seconds_remaining` in period 2). All three indicators are 0 in overtime.

| indicator | definition | step-1 effect at M3 (rim / jumper / three), pp | attempts behind it |
|---|---|---|---|
| `gt_flag` | `abs(score_diff_pre) >= 15` AND `gsr <= 480` AND `period <= 2` | **+4.79 / +1.74 / +0.53** | 27,907 / 16,297 / 28,942 |
| `eg_trail` | `-9 <= score_diff_pre <= -1` AND `gsr <= 120` AND `period <= 2` | **+0.96 / -0.51 / -7.78** | 14,234 / 6,582 / 19,811 |
| `eg_lead` | `+1 <= score_diff_pre <= +9` AND `gsr <= 120` AND `period <= 2` | **+2.75 / -2.99 / -3.28** | 7,632 / 4,469 / 4,627 |

Thresholds are read off the evidence doc's tables (its section 2 nine-cell
garbage grid, its section 3 seven-cell end-game grid) and are **not** re-tuned
after any round-2 number is seen. The mid-cell of each grid is taken because
the effect is flat across the grid, not because it is the largest.

Why indicators are the engine-safe form: each is bounded in {0, 1} and
**saturates**. Once a simulated lead passes 15 with under 8:00 left the flag is
already 1 and a further point of lead adds nothing, so the feedback path has no
gain. A continuous term (S-A, S-E) has constant gain at every margin, which is
the mechanism L23 measured.

### 13.4 Offline metrics and gates (unchanged from round 1, same `FG.score()`)

Per shot class, on F2:

- **Primary:** attempt-level log loss. Brier reported.
- **Calibration:** worst gated decile gap <= 2.00 pp (the `CALIB_MIN_SHARE`
  gating unchanged), with the level/shape split reported.
- **Responsiveness: `ARCHITECTURE_DECISIONS.md` Decision 8** as implemented in
  `fg_make.decision8_verdict` with the adopted `low_span_exempt` reading, on
  both pre-registered drivers (`shooter_make_c`, `def_allow_c`).
- **By-chance-number calibration gap** reported for first and continuation.
- **G4**, implied team eFG% by as-of tercile on the test season's own shot mix,
  for the adopted trio, +/- 1.0 pp.
- **Noise floor, per class:** the LARGER of (i) round 1's game-block bootstrap
  SE (rim 0.000539, jumper 0.000578, three 0.000849) and (ii) the SD of five
  seed-varied refits of **S-A** on F2, computed this round. "Beyond the floor"
  means a log-loss gain strictly greater than one floor.

### 13.5 The Decision-10 CLOSED-LOOP gate

Each arm is refit on F2 train and exported to
`data/processed/models/fg_make/round2/<arm>/fg_make_<class>_F2.joblib`. Nothing
under `data/processed/models/engine/` and no `winner_FGA_*.joblib` is
overwritten. The engine selects an arm with a new environment flag
**`ENGINE_FG_MAKE`**, whose default value `winner` reproduces today's behaviour
exactly; the round-2 values are `round2_S_A` ... `round2_S_E`.

Run, per arm: `scripts/run_engine.py --fold F2 --season 2025 --seeds 5
--max-games 500`, the same fixed first 500 game rows and the same five seeds
for every arm, so the RNG streams are paired by construction (seeds are on
`(seed, game_id, family)`). Read with `scripts/diag_engine_multilevel.py`.
Reported per arm: margin SD, home/away score correlation, possessions/game,
PPP, total bias, margin bias, per-team-quintile slope ratio.

**S-B is the reference condition** (no margin anywhere in fg_make, so no loop
by construction). The gate is stated RELATIVE to it because Decision 10's own
risk paragraph says a closed-loop number can fail for another sub-model's
reasons -- the clock's own `score_diff` owns the entire +4.24 possession miss
(L23) and is not fixed yet, so an ABSOLUTE margin-SD gate is unattainable this
round for every arm including the correct one. Absolute values against the
verified 2025 finals are reported next to every relative number.

| check | tolerance | source |
|---|---|---|
| **CL1** margin SD | `abs(SD(arm) - SD(S-B)) <= 1.00` point | L23, "must not move margin SD" |
| **CL2** home/away score correlation | `abs(corr(arm) - corr(S-B)) <= 0.05` | `docs/gates.yaml` `g5_corr` |
| **CL3** possessions/game | `abs(poss(arm) - poss(S-B)) <= 1.00` | `docs/gates.yaml` `g1_mean` |
| **CL4** total bias | `abs(bias(arm) - bias(S-B)) <= 1.00` point | `docs/gates.yaml` `g9_total_bias` |

An arm fails the closed-loop gate if ANY of CL1-CL4 fails. S-B passes by
construction and is the fallback if every state arm fails.

### 13.6 Decision rule

Per shot class, in this order:

1. Discard S-A (ineligible, 13.0) and any arm failing calibration or the
   Decision-8 responsiveness gate on F2.
2. Discard any arm failing the closed-loop gate of 13.5.
3. Among the survivors, the winner is the **lowest F2 log loss**, and it must
   beat the next-best surviving arm by **more than one noise floor**.
4. If the gap to a simpler arm is inside the floor, the **simpler** arm wins
   (order S-B < S-C < S-D < S-E).
5. If no arm survives steps 1-2 for a class, adopt **S-B** for that class and
   say so.

Because the closed-loop gate is run once per arm and not per class, an arm that
fails it is unavailable to every class.

### 13.7 What would falsify the whole round

If S-B, S-C, S-D and S-E all land within one noise floor of each other on log
loss, the honest reading is that the state block was never worth anything once
the leak was removed, and S-B is adopted. That outcome is pre-committed here so
it cannot later be presented as a disappointment.

<!-- RESULTS FOR ROUND 2 APPEND BELOW THIS LINE -->

---

## 14. ROUND 2 RESULTS (appended by `scripts/train_fg_make_v2.py` +
## `scripts/diag_fg_make_round2_closedloop.py`, run 2026-09-10)

Executes section 13 exactly as written. No arm was added, dropped, re-tuned or
re-thresholded after any number below existed. Artifacts:
`data/processed/models/fg_make/round2/{grid_results.csv, run_report.json,
closed_loop.json, <arm>/fg_make_<class>_F2.joblib}`.

### 14.0 Harness validity: S-A reproduces round 1 EXACTLY

| class | round 1 `lgbm` / `C_plus_state` F2 log loss | round 2 S-A F2 log loss | delta |
|---|---:|---:|---:|
| `FGA_rim` | 0.641605 | 0.641605 | 0.0 |
| `FGA_jump2` | 0.642003 | 0.642003 | 0.0 |
| `FGA_3` | 0.561085 | 0.561085 | 0.0 |

Bit-identical on all three classes, so every difference below is the state
block and nothing else in the harness.

### 14.1 Noise floors (section 13.4: the larger of the two)

| class | round-1 game-block bootstrap SE | five-seed S-A refit SD (this round) | **floor used** |
|---|---:|---:|---:|
| `FGA_rim` | 0.000539 | 0.000211 | **0.000539** |
| `FGA_jump2` | 0.000578 | 0.000124 | **0.000578** |
| `FGA_3` | 0.000849 | 0.000089 | **0.000849** |

Seed log losses: rim [0.641605, 0.641637, 0.641562, 0.641985, 0.641982];
jumper [0.642003, 0.642178, 0.642281, 0.642092, 0.641986];
three [0.561085, 0.560962, 0.561145, 0.561145, 0.561190].

### 14.2 How often the S-D indicators fire on the F2 test season

| class | `gt_flag` n (%) / make% | `eg_trail` n (%) / make% | `eg_lead` n (%) / make% | base make% |
|---|---|---|---|---:|
| `FGA_rim` | 14,678 (6.23%) / 62.57 | 4,192 (1.78%) / 59.16 | 2,266 (0.96%) / 62.14 | 58.46 |
| `FGA_jump2` | 8,172 (5.48%) / 40.33 | 1,756 (1.18%) / 38.61 | 1,200 (0.81%) / 35.83 | 39.18 |
| `FGA_3` | 15,685 (6.35%) / 33.03 | 5,609 (2.27%) / **26.74** | 1,338 (0.54%) / 29.60 | 33.80 |

Every indicator fires on 0.5-6.4% of the test season, so none of them is a
dead column. The raw make rate inside `eg_trail` on threes is **26.74% against
a 33.80% base** -- the 7-8 pp end-game effect the evidence doc measured on
2022-2025, reproducing in the held-out test fold's own raw data, which is the
strongest single confirmation that the effect is real and not a fit artefact.
`eg_lead` on the jumper and the three is the thinnest cell (1,200 and 1,338
attempts) and is the one number here to read as indicative only. The SD of
`score_diff_pre` on the test fold is 10.7-11.2 points, so the continuous term
S-E carries is not degenerate either.

### 14.3 F2 (SELECTION), per class -- the full grid

Sorted by log loss. `calib` is the worst gated decile gap (<= 2.00 pp);
`respons_d8` is `ARCHITECTURE_DECISIONS.md` Decision 8 under the adopted
`low_span_exempt` reading.

**`FGA_rim`** (n = 235,454, floor 0.000539)

| arm | feature set | log loss | brier | calib | worst gap pp | D8 | slope shooter | slope defence | chance gap first / cont |
|---|---|---:|---:|---|---:|---|---:|---:|---|
| S-A (INELIGIBLE) | `R2_A_round1_leaked` | 0.641605 | 0.225818 | PASS | 1.186 | PASS | 1.0075 | 0.9992 | 0.174 / 0.152 |
| S-E | `R2_E_safe_plus_continuous` | 0.660026 | 0.233927 | PASS | 0.569 | PASS | 0.9983 | 0.9647 | 0.159 / 0.047 |
| S-D | `R2_D_safe_plus_indicators` | 0.660124 | 0.233972 | PASS | 0.644 | PASS | 1.0085 | 0.9556 | 0.118 / 0.050 |
| **S-C** | `R2_C_safe_state` | 0.660471 | 0.234123 | PASS | 0.663 | PASS | 0.9988 | 0.9564 | 0.145 / 0.050 |
| S-B | `R2_B_no_state` | 0.667788 | 0.237594 | PASS | 1.236 | PASS | 1.0116 | 1.0044 | 0.583 / 3.328 |

**`FGA_jump2`** (n = 149,075, floor 0.000578)

| arm | feature set | log loss | brier | calib | worst gap pp | D8 | slope shooter | slope defence | chance gap first / cont |
|---|---|---:|---:|---|---:|---|---:|---:|---|
| S-A (INELIGIBLE) | `R2_A_round1_leaked` | 0.642003 | 0.226228 | PASS | 1.580 | PASS | 0.9860 | 0.8281 | 0.428 / 0.027 |
| **S-C** | `R2_C_safe_state` | 0.664094 | 0.235713 | PASS | 1.212 | PASS | 1.0233 | 0.9597 | 0.368 / 0.059 |
| S-D | `R2_D_safe_plus_indicators` | 0.664116 | 0.235728 | PASS | 1.211 | PASS | 0.9846 | 0.9440 | 0.365 / 0.067 |
| S-E | `R2_E_safe_plus_continuous` | 0.664165 | 0.235748 | PASS | 1.308 | PASS | 1.0035 | 0.9416 | 0.393 / 0.024 |
| S-B | `R2_B_no_state` | 0.665899 | 0.236569 | **FAIL** | 2.014 | PASS | 0.9701 | 0.9459 | 0.386 / 0.521 |

**`FGA_3`** (n = 246,885, floor 0.000849)

| arm | feature set | log loss | brier | calib | worst gap pp | D8 | slope shooter | slope defence | chance gap first / cont |
|---|---|---:|---:|---|---:|---|---:|---:|---|
| S-A (INELIGIBLE) | `R2_A_round1_leaked` | 0.561085 | 0.190895 | PASS | 1.007 | PASS | 0.9878 | 0.4736 | 0.284 / 0.683 |
| S-E | `R2_E_safe_plus_continuous` | 0.591835 | 0.203667 | PASS | 0.770 | **FAIL** | 0.9849 | 0.5064 | 0.251 / 0.659 |
| S-D | `R2_D_safe_plus_indicators` | 0.591880 | 0.203692 | PASS | 0.883 | PASS | 0.9875 | 0.5067 | 0.254 / 0.703 |
| **S-C** | `R2_C_safe_state` | 0.591958 | 0.203708 | PASS | 0.852 | PASS | 0.9880 | 0.4657 | 0.233 / 0.718 |
| S-B | `R2_B_no_state` | 0.594319 | 0.204750 | PASS | 0.863 | **FAIL** | 0.9903 | 0.3895 | 0.286 / 0.298 |

The two Decision-8 failures are both on the `def_allow_c` driver, whose
realised F2 span on threes is **1.373 pp** -- below the 2 pp low-span
threshold. Under the adopted `low_span_exempt` reading a low-span driver is
exempt from the SLOPE band but still needs 3 of 4 monotone steps; S-B and S-E
each manage 2. Section 14.7 records that this is the open scope question
`model.md` section 9 item 1 already flagged, and that it does not change the
winner under either reading.

### 14.4 F1 (reported, selects nothing)

| class | S-A | S-B | S-C | S-D | S-E |
|---|---:|---:|---:|---:|---:|
| `FGA_rim` | 0.645369 | 0.670600 | 0.664051 | 0.663899 | 0.663603 |
| `FGA_jump2` | 0.643098 | 0.666094 | 0.664590 | 0.664514 | 0.664803 |
| `FGA_3` | 0.562400 | 0.595710 | 0.593166 | 0.593309 | 0.593212 |

F1's ordering of S-C / S-D / S-E is not the same as F2's on any class, and
every F1 gap between them is under 0.0004 -- consistent with the three being
indistinguishable, which is what F2's floors also say.

### 14.5 THE DECISION-10 CLOSED-LOOP GATE

500 games x 5 seeds = 2,500 simulations per arm, the same game rows and the
same seeds for every arm, so the RNG streams are paired by construction.
Adapters: `ENGINE_EVENT=round2_s1` (passed explicitly -- the environment
default was changed to `reference` by in-flight work elsewhere and must not be
relied on), `ENGINE_CLOCK=reference`, `ENGINE_ROTATION=reference`,
`ENGINE_FG3=decision8`, `ENGINE_FG_MAKE=round2_<arm>`.

| arm | margin SD | d vs S-B | corr(home,away) | d | poss/gm | d | total bias | d | PPP | **gate** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S-A | **35.389** | **+21.261** | **-0.605** | **-0.850** | 71.259 | +0.447 | +1.494 | -3.057 | 1.0308 | **FAIL** (CL1, CL2, CL4) |
| S-B (reference) | 14.128 | 0.000 | +0.245 | 0.000 | 70.811 | 0.000 | +4.551 | 0.000 | 1.0590 | PASS |
| **S-C** | 14.050 | -0.079 | +0.241 | -0.004 | 70.778 | -0.034 | +3.961 | -0.590 | 1.0554 | **PASS** |
| S-D | 14.109 | -0.020 | +0.217 | -0.028 | 70.776 | -0.035 | +4.046 | -0.505 | 1.0560 | **PASS** |
| S-E | 14.133 | +0.005 | +0.225 | -0.020 | 70.737 | -0.075 | +3.896 | -0.655 | 1.0554 | **PASS** |
| ACTUAL (same 500 games) | 12.127 | -- | +0.423 | -- | 67.503 | -- | 0.000 | -- | 1.0775 | -- |

**S-A reproduces L23's signature on this subset and fails three of the four
checks**: margin SD 35.39 against 14.13 for the same engine with the same
streams, and the home/away correlation inverted to -0.605 against a realised
+0.423. The three honest state arms are indistinguishable from the no-state
reference on every closed-loop quantity: the largest movement any of them makes
is 0.079 points of margin SD, 0.028 of correlation, 0.075 of a possession and
0.66 points of total.

**What remains outside the ABSOLUTE gate is not fg_make's.** Possessions are
+3.3 and PPP -0.022 for every arm including S-B, which carries no state at all
-- that is the clock's `score_diff` (L23) and the clock round-3 censoring fix
(L20), untouched here. Margin SD 14.05 against a realised 12.13 and correlation
+0.24 against +0.42 move by less than 0.08 and 0.03 across the four honest
arms, so they are not this sub-model's to move either.

### 14.6 eFG% AND MAKE RATE BY SHOT CLASS IN THE ENGINE

The same runs, using the FGM-by-class instrumentation added to the results
contract on 2026-09-10. Truth is the event layer
(`data/processed/truth/team_game_shots_v1.parquet`) on the same 500 games;
eFG% actual is that table's `box_fgm` / `box_fga` / `box_fgm3` and is
PROVISIONAL.

| arm | eFG% | rim make% | jump2 make% | three make% | FT% | rim share | jump2 share | 3 share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S-A | **48.633** | 55.777 | 36.967 | 32.659 | 72.300 | 36.509 | 23.578 | 39.914 |
| S-B | 50.329 | 58.019 | 37.814 | 33.838 | 72.577 | 36.798 | 23.953 | 39.249 |
| **S-C** | **50.061** | 57.322 | 38.003 | 33.734 | 72.659 | 36.819 | 23.929 | 39.253 |
| S-D | 50.117 | 57.281 | 38.413 | 33.701 | 72.564 | 36.757 | 23.958 | 39.285 |
| S-E | 50.068 | 57.352 | 38.303 | 33.612 | 72.708 | 36.769 | 23.931 | 39.300 |
| **ACTUAL** | **50.239** | 58.475 | 37.845 | 33.483 | 72.298 | 36.505 | 24.207 | 39.288 |

This is the second first-order finding of the round and it was not the one the
gate was built to catch. **The leak costs 1.61 pp of engine eFG%**: S-A
produces 48.63 against a realised 50.24, and removing `score_diff` takes the
gap to **-0.18 pp (S-C)**. It is a per-class effect concentrated exactly where
the evidence doc said the leak was largest -- rim -2.70 pp under S-A, -1.15 pp
under S-C; three -0.82 pp under S-A, +0.25 pp under S-C.

The mechanism is the one L23 described running the other way. In a simulated
game most attempts happen at a margin the leaked model reads as "this shot did
not go in", because the leaked training column was systematically HIGHER on
makes; fed a pre-shot margin, the model under-predicts makes everywhere except
in the blowouts it drove itself into.

Against the 60-game smoke read taken under the REFERENCE event and clock arms
(pooled eFG% 0.474 vs 0.509, a 3.4 pp deficit): this run is a different
configuration (`ENGINE_EVENT=round2_s1`, 500 games) and is not a like-for-like
comparison, but on its own terms the make-rate deficit is now 0.18 pp rather
than 1.61, so the `score_diff` leak was a large part of whatever that read was
measuring.

### 14.7 G4 -- implied team eFG% on the test season's own shot mix

Offline, not the engine: every attempt the team actually took, weighted by its
class model's predicted make probability. Tolerance +/- 1.0 pp on the worst
as-of tercile gap.

| arm | offence worst gap (asof) | G4 offence | defence worst gap (asof) | G4 defence | team MAE off / def | team corr off / def |
|---|---:|---|---:|---|---|---|
| S-A | 0.711 | PASS | 0.914 | PASS | 1.614 / 1.448 | 0.775 / 0.761 |
| S-B | 0.867 | PASS | 1.021 | **FAIL** | 1.576 / 1.291 | 0.769 / 0.798 |
| **S-C** | 0.896 | PASS | 1.066 | **FAIL** | 1.642 / 1.340 | 0.751 / 0.784 |
| S-D | 0.936 | PASS | 1.078 | **FAIL** | 1.640 / 1.330 | 0.751 / 0.789 |
| S-E | 0.912 | PASS | 1.061 | **FAIL** | 1.620 / 1.331 | 0.754 / 0.787 |

Reported honestly and not hidden: **every honest arm misses the G4 defence
tercile gate by 0.02-0.08 pp, and the leaked arm passes it.** G4 is not a
discard criterion in section 13.6 and does not change the decision, but it is
the round's one regression against round 1 (whose adopted trio read 0.865 on
the same check) and it is an OPEN DEFECT, listed as such in `model.md`. It is a
defence-side tercile miss of under a tenth of a percentage point on a
provisional truth table; it is not a reason to keep a post-outcome feature.

### 14.8 THE DECISION, by the pre-registered rule of section 13.6

S-A is discarded first, as pre-registered before the round ran. Every remaining
arm passed the closed-loop gate, so step 2 discards nothing and the choice is
made on F2 log loss against the floor, with ties going to the simpler arm.

**`FGA_rim`** (floor 0.000539) -- all four honest arms pass calibration and D8.
Lowest is S-E. S-E beats S-D by 0.000098 = **0.18 floors**, inside it, so the
simpler S-D takes it; S-D beats S-C by 0.000347 = **0.64 floors**, inside it,
so the simpler S-C takes it; S-C beats S-B by 0.007317 = **13.57 floors**,
beyond it. **WINNER: S-C.**

**`FGA_jump2`** (floor 0.000578) -- S-B fails calibration (2.014 pp) and is
discarded. Of the three survivors S-C is both the lowest log loss and the
simplest; S-D and S-E are 0.04 and 0.12 floors behind. **WINNER: S-C.**

**`FGA_3`** (floor 0.000849) -- S-B and S-E fail Decision 8 on the `def_allow_c`
driver and are discarded. S-D beats S-C by 0.000078 = **0.09 floors**, inside
it, so the simpler arm takes it. **WINNER: S-C.**

> ### WINNER, all three classes: **S-C `R2_C_safe_state`**
> `B_plus_shooter` + `period`, `seconds_remaining`, `in_bonus`,
> `chance_number`, `chance_elapsed_s`, `is_transition_f`.
> **No margin term of any kind.**

Under the alternative strict reading of Decision 8's low-span clause (a sub-2
pp driver exempt from the steps as well as the band, which is what
`ARCHITECTURE_DECISIONS.md`'s own text says and what
`fg_make.DECISION8_READINGS`'s `low_span_exempt` does NOT implement), S-B and
S-E re-enter on `FGA_3`; S-E would then be lowest at 0.591835, ahead of S-C by
0.09 floors -- inside the floor, so the simpler arm still wins. **The winner is
S-C under both readings.** The scope question is real and still open; it did
not decide anything here.

### 14.9 What the round bought, and what it cost

| quantity | round 1 (S-A, leaked) | round 2 winner (S-C) | change |
|---|---:|---:|---|
| F2 log loss, rim / jumper / three | 0.641605 / 0.642003 / 0.561085 | 0.660471 / 0.664094 / 0.591958 | **+35.0 / +38.2 / +36.4 floors WORSE** |
| worst decile calibration gap, rim / jumper / three | 1.186 / 1.580 / 1.007 pp | 0.663 / 1.212 / 0.852 pp | better on all three |
| engine margin SD (500 games x 5 seeds) | 35.389 | 14.050 | actual 12.127 |
| engine home/away score correlation | -0.605 | +0.241 | actual +0.423 |
| engine eFG% | 48.633% | 50.061% | actual 50.239% |
| engine possessions/game | 71.259 | 70.778 | actual 67.503 (clock's, not this model's) |
| G4 defence worst tercile gap | 0.914 pp PASS | 1.066 pp FAIL | the round's one regression |

**The 35-38 noise floors of log loss that round 1's state block appeared to buy
were the leak.** An offline metric computed on real game states cannot see a
post-outcome feature that is stable in those states, which is precisely the
case Decision 10 was written for -- and the closed-loop gate caught it at
21 points of margin SD and 0.85 of correlation.

### 14.10 Falsification check (section 13.7)

Section 13.7 pre-committed that if S-B, S-C, S-D and S-E were all within one
floor of each other, the state block was worth nothing and S-B would be
adopted. That is NOT what happened: S-C beats S-B by 13.57 floors at the rim
and S-B fails a gate outright on the other two classes (calibration on the
jumper, Decision 8 on the three). The engine-safe state block earns its place;
the margin does not.

### 14.11 Artifacts and how the engine reaches them

| path | what |
|---|---|
| `data/processed/models/fg_make/round2/<arm>/fg_make_<class>_F2.joblib` | one fitted LightGBM per (arm, class), refit on F2 train. S-C's three carry `adopted=True`; every other arm carries `adopted=False` |
| `data/processed/models/fg_make/round2/grid_results.csv` | every (class, fold, arm) row |
| `data/processed/models/fg_make/round2/run_report.json` | full calibration deciles, responsiveness ladders, Decision-8 working, indicator exposure, floors, G4 |
| `data/processed/models/fg_make/round2/closed_loop.json` | section 14.5 and 14.6 |
| `data/processed/models/fg_make/state_confound.json` | the step-1 evidence |
| `results/engine_v0/fgmake_r2_<arm>/` | the five paired engine runs |

`ENGINE_FG_MAKE=round2_S_C` selects the winner. **The engine default is still
`ENGINE_FG_MAKE=winner`** (the round-1 artifacts, byte for byte) and nothing
under `data/processed/models/engine/` or any `winner_FGA_*.joblib` was
overwritten; switching the default is the PM's step, not this round's.

---

## 15. ROUND 2b PRE-REGISTRATION: the round-2 winner under S1 (written 2026-09-10, AFTER round 2 decided and BEFORE 2b ran)

Why this is a separate round and not an arm of round 2: section 13.1 fixed the
training scheme at STATIC so the five arms differed only in the state block, and
section 13 was committed (5e54872) before any round-2 modelling ran. L21 makes
**S1** -- in-season monthly walk-forward refit -- the standing default scheme
for every sub-model, and `src/cbb_sim/engine/manifest.py` exists precisely so an
S1 sub-model arrives as a schedule of dated artifacts. Round 2b asks the one
question round 2 deliberately did not.

**The question.** Does the round-2 winner S-C, refit monthly in-season, beat the
same arm fitted once, on the same fold, by more than the noise floor, without
regressing any gate?

### 15.1 What is FIXED

- **The arm is S-C `R2_C_safe_state`** on all three shot classes, as round 2
  decided. No other state parametrisation is re-opened.
- Model class, parameters, universe, possessions version, folds and floors are
  exactly section 13.1's and section 14.1's. The floors are the SAME numbers
  (rim 0.000539, jumper 0.000578, three 0.000849) because the arm is the same
  arm; no new floor is computed and none may be.
- **S1 is `possession_outcome.month_boundaries` / `fit_predict_scheme`'s
  definition, reused not reimplemented**: one refit per calendar month of the
  test season, fitted on all prior seasons plus the test season strictly before
  that month's first day; every test game scored by the most recent refit at or
  before its own date.

### 15.2 The two arms

| arm | scheme | artifact |
|---|---|---|
| **S-C-static** | one fit on F2 train | `round2/S_C/fg_make_<class>_F2.joblib` (already exists, round 2) |
| **S-C-S1** | monthly walk-forward | `round2b/S_C_s1/<class>_<refit_date>.joblib` + `manifest_<class>.json` in `manifest.py` format (`refit_date`, `path`, `max_train_date`) |

### 15.3 Gates -- identical to round 2, nothing added, nothing relaxed

Offline on F2, per class: log loss (primary), Brier, worst gated decile
calibration gap <= 2.00 pp, Decision 8 responsiveness on both drivers, by-chance
calibration. Closed-loop: the same 500 games x 5 seeds, the same paired streams,
the same S-B reference, the same CL1-CL4 tolerances of section 13.5.

Every artifact is loaded through `ArtifactManifest` with
`require_max_train_date=True`, so the honest-backtest property is enforced by
the same code path the event layer uses and not by this script's good intentions.

### 15.4 Decision rule (as the coordinator specified, 2026-09-10)

**Adopt S1 unless a gate regresses beyond the floor.** Concretely, per class:

1. If S-C-S1 fails calibration or Decision 8 on F2 where S-C-static passed, S1
   is NOT adopted for that class.
2. If S-C-S1 fails the closed-loop gate, S1 is NOT adopted at all.
3. If S-C-S1's log loss is WORSE than S-C-static's by more than one noise
   floor, S1 is not adopted for that class.
4. Otherwise **S1 is adopted**, including when the log-loss difference is
   inside the floor -- L21's finding is that S1 is a calibration fix and not a
   log-loss fix, so "no log-loss gain" is the expected result and is not a
   reason to refuse the standing scheme.

The costs are reported either way: number of refits, fit time, and the artifact
count the engine has to carry.

<!-- ROUND 2b RESULTS APPEND BELOW THIS LINE -->

---

## 16. ROUND 2b RESULTS (`scripts/train_fg_make_v2b_s1.py`, run 2026-09-10)

Executes section 15. One arm (the round-2 winner S-C), two schemes, same fold,
same floors, same gates.

### 16.1 The S1 schedule as it actually ran

Six refits per class, the calendar months of the 2025 test season. Every
artifact declares its own `max_train_date`, and every one of them precedes its
refit date, so the honest-backtest property is a property of the files and not
of this script's intentions.

| refit date | train rows (rim / jumper / three) | of which from the test season | scored | max_train_date |
|---|---|---:|---|---|
| 2024-11-01 | 582,043 / 423,483 / 604,123 | 0 / 0 / 0 | 49,782 / 31,866 / 53,707 | 2024-04-08 |
| 2024-12-01 | 631,825 / 455,349 / 657,830 | 49,782 / 31,866 / 53,707 | 39,246 / 24,378 / 40,658 | 2024-11-30 |
| 2025-01-01 | 671,071 / 479,727 / 698,488 | 89,028 / 56,244 / 94,365 | 59,176 / 37,759 / 62,005 | 2024-12-31 |
| 2025-02-01 | 730,247 / 517,486 / 760,493 | 148,204 / 94,003 / 156,370 | 53,671 / 33,264 / 55,155 | 2025-01-31 |
| 2025-03-01 | 783,918 / 550,750 / 815,648 | 201,875 / 127,267 / 211,525 | 32,905 / 21,264 / 34,514 | 2025-02-28 |
| 2025-04-01 | 816,823 / 572,014 / 850,162 | 234,780 / 148,531 / 246,039 | 674 / 544 / 846 | 2025-03-31 |

The April segment scores 544-846 attempts (the tournament tail) and is
UNDERPOWERED on its own; it is not read separately anywhere.

Cost: 18 artifacts instead of 3, and 117 / 84 / 126 s of fitting instead of
21 / 11 / 12 s -- about 6x, which is the arithmetic of six refits.

### 16.2 F2, static vs S1

| class | scheme | log loss | delta in floors | calib worst gap | D8 | chance gap first / cont |
|---|---|---:|---:|---:|---|---|
| `FGA_rim` | static | 0.660471 | -- | 0.663 PASS | PASS | 0.145 / 0.050 |
| `FGA_rim` | **S1** | **0.660150** | **-0.60** | 0.672 PASS | PASS | 0.008 / 0.158 |
| `FGA_jump2` | static | 0.664094 | -- | 1.212 PASS | PASS | 0.368 / 0.059 |
| `FGA_jump2` | **S1** | **0.663985** | **-0.19** | 1.693 PASS | PASS | 0.049 / 0.306 |
| `FGA_3` | static | 0.591958 | -- | 0.852 PASS | PASS | 0.233 / 0.718 |
| `FGA_3` | **S1** | **0.591644** | **-0.37** | 0.578 PASS | PASS | 0.130 / 0.295 |

S1 is better on log loss on all three classes and on none of them by a full
floor (0.19-0.60), which is **exactly L21's finding reproducing on a second
sub-model**: S1 is not a log-loss fix. Calibration is better on `FGA_3`
(0.852 -> 0.578 pp), materially better on the FIRST-CHANCE segment of every
class (0.145 -> 0.008, 0.368 -> 0.049, 0.233 -> 0.130 pp), and worse on the
jumper's overall worst decile (1.212 -> 1.693 pp) and on the continuation
segment of the rim and the jumper. Nothing crosses a gate in either direction:
every cell is PASS under both schemes.

Unlike possession_outcome, fg_make had no calibration FAILURE for S1 to repair
-- the round-2 winner already passed at 0.66-1.21 pp -- so the honest summary is
that S1 is neutral-to-slightly-positive here rather than decisive.

### 16.3 The closed-loop gate for the S1 arm

Same 500 games, same five seeds, same paired streams, same S-B reference and
the same CL1-CL4 tolerances. Run: `results/engine_v0/fgmake_r2_S_C_s1`,
`ENGINE_FG_MAKE=round2b_S_C_s1`, every artifact selected per game by
`ArtifactManifest` with `require_max_train_date=True`.

| arm | margin SD | d vs S-B | corr | d | poss/gm | d | total bias | d | PPP | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S-C (static) | 14.050 | -0.079 | +0.241 | -0.004 | 70.778 | -0.034 | +3.961 | -0.590 | 1.0554 | PASS |
| **S-C-S1** | 14.002 | -0.126 | +0.229 | -0.016 | 70.758 | -0.053 | +3.912 | -0.640 | 1.0553 | **PASS** |
| ACTUAL | 12.127 | -- | +0.423 | -- | 67.503 | -- | 0.000 | -- | 1.0775 | -- |

eFG% and make rate by class on the same runs:

| arm | eFG% | rim | jump2 | three | FT% |
|---|---:|---:|---:|---:|---:|
| S-C (static) | 50.061 | 57.322 | 38.003 | 33.734 | 72.659 |
| **S-C-S1** | **50.147** | 57.379 | 38.642 | 33.600 | 72.525 |
| ACTUAL | 50.239 | 58.475 | 37.845 | 33.483 | 72.298 |

S-C-S1 is the closest arm to the realised eFG% of any run in either round
(-0.09 pp).

### 16.4 THE DECISION, by section 15.4

No gate regressed: calibration PASS on all three classes under both schemes,
Decision 8 PASS on all three, closed-loop PASS. S1's log loss is better, not
worse, on every class. Rule 4 applies.

> ### **S1 IS ADOPTED for fg_make, all three shot classes.**
> Arm: S-C `R2_C_safe_state`. Scheme: monthly in-season walk-forward.
> Engine flag: `ENGINE_FG_MAKE=round2b_S_C_s1`.
> Artifacts: `data/processed/models/fg_make/round2b/S_C_s1/` -- 18 joblibs and
> three `manifest.py`-format manifests, all marked `adopted=True`.

The static S-C artifacts stay on disk under `round2/S_C/` and stay marked
adopted for the STATE decision they won; `ENGINE_FG_MAKE=round2_S_C` still
serves them, which is what makes "S1 vs static" re-runnable without a refit.

**The engine default remains `ENGINE_FG_MAKE=winner`** -- round 1's artifacts,
byte for byte. Switching the default to `round2b_S_C_s1` is the PM's step.
`run_meta.json` will then report `scheme_static_fg_make=False`, which is the
flag that tells a later reader this model is served by a schedule.

### 16.5 Path taken, for the record

The round-2 pre-registration (section 13) was written, committed (5e54872) and
RUN with the scheme fixed at static, before the standing-scheme instruction
arrived. Per that instruction's own branch, the round-2 spec was not changed
retroactively; instead round 2b was pre-registered (section 15) after round 2
decided and before 2b ran, and is the record of the S1 question. Both rounds'
numbers stand as they were produced.

---

## 17. ROUND 3 PRE-REGISTRATION: the shooter-label data fix (written 2026-09-10, BEFORE round-3 modelling ran)

Owner: Sonnet worker. Trigger: `docs/models/change_ledger.md`'s "CBBD's
`participant_1_id` is the ASSISTER, not the shooter, on half of all assisted
made field goals" row, which names fg_make as "STILL ON THE WRONG COLUMN and
needs a re-run", and `docs/LEARNINGS.md` L27's closing line ("fg_make round 3
re-keys the shooter on `shot_shooter_id`"). Evidence this spec is built on,
committed in the same commit and BEFORE the round runs:
**`docs/tests/fg_make_shooter_key_2026-09-10.md`**
(`scripts/diag_fg_make_shooter_key_v1.py`,
`data/processed/models/fg_make/shooter_key_audit_v1.json`).

Nothing in this section may be edited after the round starts. Results append
below it; a status change goes to `docs/models/change_ledger.md` in the same
commit.

### 17.0 What the step-1 evidence established, and what it changes about this round

1. **`fg_make` reads the wrong participant column.** `_season_events` calls
   `event_stream.build_stream` with no `shooter_key`, so it takes
   `DEFAULT_SHOOTER_KEY = "participant_1_id"`, which is the ASSISTER on
   47.7-50.2% of ASSISTED made field goals in every one of 2022-2025, flat
   across seasons (evidence doc 1.1-1.2). The defect is confined to assisted
   makes (unassisted mismatch 0.000-0.005%) and is 2.4-2.9x larger on rim/three
   than on the mid-range jumper, which tracks the assisted share of each class
   (25.2 / 10.4 / 28.0%).
2. **The shooter as-of feature this drives, `shooter_make_c`, moves a lot.**
   Correlation between the old- and new-keyed feature is 0.694 (rim) / 0.802
   (jumper) / **0.335 (three)**; the mean absolute move is 7.68 / 5.11 / 12.97
   pp across ALL attempts (not just the relabelled ones -- the credited
   player's cumulative tally ripples onto every other attempt he or the
   wrongly-credited assister ever took). Round 2b's own realised
   `shooter_make_c` quintile span on `FGA_rim` is 16.08 pp
   (`round2b/run_report.json`), so this move is roughly half the feature's own
   range -- large enough that the Decision-8 shooter-slope reading is expected
   to be different, not merely re-estimated.
3. **`shooter_att_c` and `shooter_games_asof` barely move** (corr 0.89-1.00):
   attempt COUNTS are conserved in aggregate even when the credited player
   changes; only the MAKE/MISS content of a player's history moves. This
   isolates the fix to exactly the one feature the mislabelling can touch.
4. **The drop cost of keying on `shot_shooter_id` is small and reported, never
   imputed:** 1,385 of 2,241,195 attempts (0.062%) pooled 2022-2025; per-team
   p95 is 0.0-1.0% and the max in any single team-season is 8.94% (`FGA_3`,
   2022), the class with the thinnest per-team volume. `scripts/train_fg_make_v3_shooter.py`
   reports this by season and by team, not just pooled.

**Consequence for the arm list.** This round does NOT re-open the state
parametrisation (round 2, section 13) or the S1-vs-static question (round 2b,
section 15) -- both already decided and orthogonal to the shooter label, since
every round-2/2b arm read the SAME (contaminated) shooter column. It re-opens
exactly one thing: the DATA the round-2b winner is trained on.

### 17.1 What is FIXED and not up for selection

- **Arm: the round-2b winner, S-C `R2_C_safe_state`, under S1.** No state
  parametrisation and no scheme is re-decided here.
- **Model class and parameters:** unchanged (LightGBM per class, the F1-only
  frozen ladder, `lgbm_ladder_v2.json`). No new parameter search.
- **Folds:** F1 reported (selects nothing), **F2 = train {2022, 2023, 2024}
  test 2025 = SELECTION**. 2026 sealed (`assert_not_sealed`).
- **`FG.score()` is unchanged** -- log loss, Brier, calibration, Decision 8
  responsiveness (both drivers), by-chance calibration. Nothing in the metric
  is added or relaxed for this round.
- **Possessions version v2**, L16 rim override, same universe (D-I,
  non-truncated, `pbp_complete`).
- **S1 is reused, not reimplemented**, exactly as round 2b: one refit per
  calendar month of the test season via `possession_outcome.month_boundaries`.

### 17.2 The two arms

| arm | shooter key | scheme | artifact |
|---|---|---|---|
| **REFERENCE** `R2b_S_C_s1` | `participant_1_id` (the defect) | S1 | `round2b/S_C_s1/<class>_<refit_date>.joblib` + `manifest_<class>.json` (already exists, round 2b; read back, NOT refit) |
| **NEW** `R3_S_C_s1_shooterfix` | `shot_shooter_id` (the fix) | S1 | `round3_shooter/S_C_s1/<class>_<refit_date>.joblib` + `manifest_<class>.json` |

The reference's numbers are read back from `round2b/run_report.json` rather
than recomputed, so the comparison is against the exact object round 2b
decided on (the same convention `train_fg_make_v2b_s1.py` used against round
2's own artifacts). Rows with no `shot_shooter_id` are DROPPED from the new
arm's training and test data, never imputed, with the count reported by season
and by team (17.0 item 4); no fallback to `participant_1_id` on those rows,
because a fallback would silently reinstate the assister on exactly the
population this fix exists to remove.

### 17.3 Offline metrics and gates (unchanged from rounds 2/2b, same `FG.score()`)

Per shot class, on F2:

- **Primary:** attempt-level log loss. Brier reported.
- **Calibration:** worst gated decile gap <= 2.00 pp.
- **Responsiveness:** `ARCHITECTURE_DECISIONS.md` Decision 8 on both drivers
  (`shooter_make_c`, `def_allow_c`), `low_span_exempt` reading (the adopted
  one; both readings reported).
- **THE METRIC THIS ROUND EXPECTS TO MOVE: the `shooter_make_c` Decision-8
  slope ratio and quintile step count.** A data fix that removes 48-50% wrong
  labels on assisted makes is expected to make the shooter as-of rate a more
  faithful (not necessarily a stronger) predictor of the true shooter's own
  skill; if the slope does not move outside noise, that is reported as a
  finding, not suppressed.
- **By-chance-number calibration gap** reported for first and continuation.
- **Noise floor: SECOND-SEED REFIT.** Unlike rounds 2/2b (which reused round
  1's five-seed/bootstrap floor because the arm was unchanged), the DATA
  changes this round, so the floor is recomputed on the new arm: the entire
  S1 schedule (all monthly refits, all three classes) is refit a second time
  with `seed=1`, and the floor per class is `abs(log_loss(seed=0) -
  log_loss(seed=1))` on F2. This is a smaller, cheaper floor than the round-2
  five-seed SD by construction (one extra fit instead of four), and is stated
  as such rather than presented as equivalent.

### 17.4 The Decision-10 CLOSED-LOOP gate

Per Decision 10, on the fixed 500-game subset, 5 seeds, via
`scripts/run_engine.py --fold F2 --season 2025 --seeds 5 --max-games 500` +
`scripts/diag_engine_multilevel.py`, `ENGINE_EVENT=round2_s1` passed
EXPLICITLY (the environment default is `reference` and must not be relied on,
per section 14.5's own note), `ENGINE_CLOCK=reference`,
`ENGINE_ROTATION=reference`, `ENGINE_FG3=decision8`.

**Known constraint, stated here before the round runs:** `cbb_sim.engine.adapters.FgMakeAdapter._load_round2b`
resolves `ENGINE_FG_MAKE=round2b_<arm>` to the fixed path
`data/processed/models/fg_make/round2b/<arm>/`; it has no branch for
`data/processed/models/fg_make/round3_shooter/<arm>/`. This worker does not
own `adapters.py` (another worker does) and will not edit it. Consequence,
pre-committed: **the REFERENCE arm's closed-loop number is the one already on
record** (`ENGINE_FG_MAKE=round2b_S_C_s1`, section 16.3: margin SD 14.002,
corr(home,away) +0.229, poss/gm 70.758, total bias +3.912, PPP 1.0553, PASS)
and is cited rather than re-executed, because the artifacts, the 500 games and
the 5 seeds are byte-identical to that run and RNG is seeded on `(seed,
game_id, family)`, so a re-run would reproduce it exactly at the cost of
shared compute five other workers are using. **The NEW (shot_shooter_id)
arm's closed-loop is PENDING an adapter change** -- either a new
`mode.startswith("round3_shooter_")` branch in `FgMakeAdapter.load`, or
generalising `_load_round2b` to take the round directory name as an argument
derived from the mode string -- and is not run this round. The offline
decision (17.5) is therefore made WITHOUT a closed-loop read on the new arm;
this is reported as an open item, not hidden.

| check | tolerance | source |
|---|---|---|
| **CL1** margin SD | `abs(SD(arm) - SD(S-B)) <= 1.00` point | unchanged from round 2 (13.5) |
| **CL2** home/away score correlation | `abs(corr(arm) - corr(S-B)) <= 0.05` | unchanged |
| **CL3** possessions/game | `abs(poss(arm) - poss(S-B)) <= 1.00` | unchanged |
| **CL4** total bias | `abs(bias(arm) - bias(S-B)) <= 1.00` point | unchanged |

### 17.5 Decision rule

**Adopt the re-keyed arm unless a gate regresses beyond the floor.**
Concretely, per class:

1. If the new arm fails calibration or Decision 8 on F2 where the reference
   passed, the new arm is NOT adopted for that class.
2. If the new arm's closed-loop gate is run and fails, the new arm is not
   adopted at all. **If the closed-loop gate could not be run (17.4), this
   step is a no-op and the offline decision stands provisionally, pending the
   adapter change and a closed-loop confirmation before the artifacts replace
   `round2b_S_C_s1` as the engine's served model.**
3. If the new arm's log loss is WORSE than the reference's by more than one
   noise floor (17.3), the new arm is not adopted for that class.
4. Otherwise **the new arm is adopted**, including when the log-loss
   difference is inside the floor or slightly worse than it: a data fix that
   corrects 48-50% wrong labels on assisted makes is not required to buy log
   loss, and a loss inside the floor is explicitly NOT a veto (the
   coordinator's instruction for this round). **The shooter-quintile
   Decision-8 slope moving is treated as the expected, positive result of a
   correct label, not as a surprise to explain away.**

### 17.6 What would falsify this round

If `shooter_make_c`'s Decision-8 slope and quintile steps are UNCHANGED
(within noise) between the reference and the new arm despite 1.1-1.4's
measured feature movement, the honest reading is that the mislabelling washes
out in aggregate and the fix is cosmetic; that outcome is pre-committed here so
it cannot later be presented as a disappointment. If the new arm's log loss is
worse than the reference by more than the floor on any class, that class does
NOT adopt the fix and is reported as an open item rather than forced through.

<!-- RESULTS FOR ROUND 3 APPEND BELOW THIS LINE -->

---
