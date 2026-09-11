# L3 POSSESSION-OUTCOME: experiments

Append-only. Status changes go to `docs/models/change_ledger.md` in the same commit
(`CLAUDE.md`, standing rule "bake-off before any choice").

---

## 1. Pre-registration (authored by the PM, 2026-09-10, BEFORE any modelling)

Target: terminal event of a chance (the first chance of each possession, and separately continuation chances after an OREB): categorical over {TOV, FGA_rim, FGA_jump2, FGA_3, FT_trip_shooting, FT_trip_bonus}; end_period chances excluded from training and handled by the clock model. Make/miss and FT outcomes are separate models (possession_make, to be pre-registered next); this bake-off is the event-type mix only.
Universe: D-I, non-truncated, seasons 2022-2025 (team level; no lineup features in this bake-off). Folds: F1 train {2022, 2023} test 2024; F2 train {2022, 2023, 2024} test 2025 (selection). Sealed 2026.
Feature sets: A_team (offense as-of per-100 rates of 3PA, rim share, TOV%, FTr from the shift(1) expanding team gamelogs; defense-allowed versions for the opponent; own ratings off_c/def_c of both; site), B_plus_season (A + season index, days since season start), C_plus_state (B + period, seconds remaining, score diff from offense view, bonus flag, is_transition, chance number), D_plus_interactions (C + explicit offense-rate x defense-allowed products for the linear arms only).
Model classes: multinomial ridge logit; nested binary cascade of GLMs (TOV? then foul? then shot type) with the same features; LightGBM multiclass; and a matchup-naive baseline (league-average shares by season, the floor).
Metrics: multiclass log loss and per-class Brier on F2; per-class calibration by predicted-probability decile; responsiveness: predicted class share bucketed by the offense's as-of rate quintile must slope with actual (this is the matchup-specific rule); by-state calibration (transition vs not, bonus vs not, late-clock); noise floor = seed-varied refit of the tree arm, bootstrap SE for the linear arms.
Decision rules: winner = lowest F2 log loss among arms passing per-class calibration (max absolute decile miscalibration <= 2 pp on classes with share >= 5%) and responsiveness (monotone in 4 of 5 quintile steps for each of 3PA, rim, TOV); a tree arm must beat the best linear arm by more than the noise floor; ties go to the simpler arm. Feature set is chosen by the same rule within the winning class. State features that do not reduce log loss beyond the floor are recorded as rejected in features.md.

---

## 2. Grid configuration (as executed)

| Dimension | Values |
|---|---|
| Target | terminal event of a chance, 6 classes: TOV, FGA_rim, FGA_jump2, FGA_3, FT_trip_shooting, FT_trip_bonus |
| Chance populations | `first` (chance_number == 1) and `cont` (chance_number > 1), fitted and scored separately |
| Feature sets | A_team, B_plus_season, C_plus_state, D_plus_interactions |
| Model classes | `baseline` (league-average shares by season), `ridge_logit` (multinomial), `cascade` (nested binary GLMs: TOV? -> foul-trip? -> bonus-vs-shooting? -> shot type), `lgbm` (LightGBM multiclass) |
| Folds | F1: train 2022+2023, test 2024. F2: train 2022+2023+2024, test 2025 (selection) |
| Sealed | 2026 -- `cbb_sim.data.seal.assert_not_sealed` is called on every train and test slice |
| Primary metric | multiclass log loss on F2, `first` population |
| Noise floor | tree arm: 5 seed-varied refits, SD of F2 log loss. Linear arms: 200-replicate game-level block bootstrap SE of F2 log loss |

D_plus_interactions is defined for the linear arms only (`ridge_logit`, `cascade`); the tree arm
finds interactions itself, so running it on D would be a duplicate of C, and the pre-registration
says so. It is reported as `n/a` for `lgbm`.

Artifacts: `data/processed/models/possession_outcome/`. Trainer:
`scripts/train_possession_outcome_v1.py`. Feature provenance: `features.md`.

<!-- RESULTS APPENDED BELOW BY scripts/train_possession_outcome_v1.py -->

---

## 3. Full results (run 2026-09-10 11:58, `scripts/train_possession_outcome_v1.py`)

Design: 3,433,227 modelled chances over seasons [2022, 2023, 2024, 2025] (2,996,517 first, 436,710 continuation). `end_period` and `unknown` chances excluded per the pre-registration. Total grid runtime 17.9 min. CSV alongside: `data/processed/models/possession_outcome/grid_results.csv`.

### 3.1 Population `first` (the primary metric)

**F1** -- train [2022, 2023], test [2024]

| arm | feature set | log loss | calib | worst gap (pp) | of which level (pp) | residual shape (pp) | respons. | Brier TOV | Brier rim | Brier jump2 | Brier 3 | Brier FT-shoot | Brier FT-bonus | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 1.525109 | FAIL | 2.391 | 1.205 | 1.186 | PASS | 0.12932 | 0.18443 | 0.154995 | 0.198049 | 0.056125 | 0.037261 | 61.8 |
| cascade | D_plus_interactions | 1.538391 | PASS | 1.829 | 0.92 | 1.847 | PASS | 0.129394 | 0.184898 | 0.15534 | 0.199667 | 0.056576 | 0.040551 | 3.0 |
| cascade | C_plus_state | 1.538421 | PASS | 1.894 | 0.924 | 1.873 | PASS | 0.129396 | 0.184904 | 0.155341 | 0.199671 | 0.056575 | 0.040554 | 2.9 |
| ridge_logit | D_plus_interactions | 1.538805 | PASS | 1.558 | 0.896 | 1.534 | PASS | 0.129351 | 0.184847 | 0.155336 | 0.199704 | 0.056726 | 0.040308 | 5.7 |
| ridge_logit | C_plus_state | 1.538838 | PASS | 1.589 | 0.897 | 1.554 | PASS | 0.129352 | 0.184849 | 0.155338 | 0.19971 | 0.056726 | 0.040312 | 5.8 |
| cascade | B_plus_season | 1.624485 | PASS | 1.068 | 0.884 | 0.76 | PASS | 0.130025 | 0.186501 | 0.158544 | 0.201518 | 0.057213 | 0.048112 | 2.5 |
| ridge_logit | B_plus_season | 1.624524 | PASS | 1.073 | 0.857 | 0.674 | PASS | 0.130023 | 0.186494 | 0.158566 | 0.201533 | 0.057213 | 0.048112 | 2.7 |
| cascade | A_team | 1.625211 | PASS | 1.644 | 1.316 | 0.652 | PASS | 0.130138 | 0.18659 | 0.158566 | 0.201504 | 0.057237 | 0.048109 | 2.4 |
| ridge_logit | A_team | 1.625281 | PASS | 1.688 | 1.309 | 0.746 | PASS | 0.130144 | 0.186583 | 0.158598 | 0.201513 | 0.057238 | 0.04811 | 5.1 |
| lgbm | B_plus_season | 1.626392 | FAIL | 2.238 | 1.237 | 1.105 | PASS | 0.130212 | 0.186603 | 0.158628 | 0.201617 | 0.057271 | 0.048137 | 52.6 |
| lgbm | A_team | 1.626983 | FAIL | 2.472 | 1.393 | 1.091 | PASS | 0.130276 | 0.186652 | 0.158684 | 0.201691 | 0.057271 | 0.048139 | 48.2 |
| baseline | none | 1.637078 | PASS | 1.212 | 1.212 | 0.0 | FAIL | 0.130514 | 0.187542 | 0.160511 | 0.203047 | 0.057244 | 0.048221 | 0.7 |

**F2** -- train [2022, 2023, 2024], test [2025]  (SELECTION)

| arm | feature set | log loss | calib | worst gap (pp) | of which level (pp) | residual shape (pp) | respons. | Brier TOV | Brier rim | Brier jump2 | Brier 3 | Brier FT-shoot | Brier FT-bonus | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 1.518965 | FAIL | 2.954 | 1.802 | 1.152 | PASS | 0.130117 | 0.185684 | 0.144377 | 0.20294 | 0.056785 | 0.037481 | 77.9 |
| cascade | D_plus_interactions | 1.533645 | FAIL | 2.19 | 1.708 | 2.105 | PASS | 0.130333 | 0.186354 | 0.144758 | 0.20461 | 0.057281 | 0.040883 | 3.9 |
| cascade | C_plus_state | 1.533648 | FAIL | 2.229 | 1.697 | 2.148 | PASS | 0.130332 | 0.186356 | 0.144753 | 0.204614 | 0.05728 | 0.040889 | 3.8 |
| ridge_logit | C_plus_state | 1.53427 | FAIL | 2.061 | 1.641 | 1.929 | PASS | 0.130305 | 0.186309 | 0.144725 | 0.204638 | 0.057449 | 0.040662 | 9.2 |
| ridge_logit | D_plus_interactions | 1.53427 | FAIL | 2.09 | 1.652 | 1.901 | PASS | 0.130305 | 0.186312 | 0.144727 | 0.204636 | 0.057448 | 0.040655 | 9.2 |
| cascade | B_plus_season | 1.621769 | FAIL | 2.093 | 1.636 | 0.544 | PASS | 0.130885 | 0.188315 | 0.147678 | 0.20688 | 0.057955 | 0.048592 | 3.0 |
| ridge_logit | B_plus_season | 1.621815 | FAIL | 2.134 | 1.615 | 0.663 | PASS | 0.130887 | 0.188307 | 0.147691 | 0.206882 | 0.057956 | 0.048592 | 3.7 |
| lgbm | B_plus_season | 1.622017 | FAIL | 2.592 | 1.82 | 0.958 | PASS | 0.130907 | 0.188311 | 0.14771 | 0.206847 | 0.057976 | 0.048593 | 64.8 |
| cascade | A_team | 1.622213 | FAIL | 2.449 | 1.892 | 0.556 | PASS | 0.130899 | 0.18842 | 0.147759 | 0.206792 | 0.057969 | 0.048593 | 3.3 |
| ridge_logit | A_team | 1.622292 | FAIL | 2.598 | 1.894 | 0.703 | PASS | 0.130903 | 0.188414 | 0.147784 | 0.206802 | 0.057969 | 0.048594 | 3.7 |
| lgbm | A_team | 1.623143 | FAIL | 2.78 | 1.934 | 1.043 | PASS | 0.130982 | 0.188408 | 0.147789 | 0.206911 | 0.057991 | 0.048614 | 56.9 |
| baseline | none | 1.633501 | PASS | 1.821 | 1.821 | 0.0 | FAIL | 0.131243 | 0.18917 | 0.149613 | 0.208339 | 0.057967 | 0.048694 | 0.6 |

### 3.2 Population `cont` (chances after an offensive rebound)

**F1** -- train [2022, 2023], test [2024]

| arm | feature set | log loss | calib | worst gap (pp) | of which level (pp) | residual shape (pp) | respons. | Brier TOV | Brier rim | Brier jump2 | Brier 3 | Brier FT-shoot | Brier FT-bonus | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cascade | C_plus_state | 1.503393 | FAIL | 2.213 | 1.11 | 1.57 | PASS | 0.104567 | 0.237408 | 0.124476 | 0.166282 | 0.067272 | 0.044421 | 0.5 |
| cascade | D_plus_interactions | 1.503542 | FAIL | 2.054 | 1.092 | 1.47 | PASS | 0.104573 | 0.237409 | 0.124484 | 0.166319 | 0.067269 | 0.044426 | 0.5 |
| ridge_logit | C_plus_state | 1.503995 | PASS | 1.83 | 1.19 | 1.114 | PASS | 0.104566 | 0.237422 | 0.124478 | 0.166286 | 0.067295 | 0.044387 | 0.8 |
| ridge_logit | D_plus_interactions | 1.504159 | PASS | 1.919 | 1.17 | 1.181 | PASS | 0.104571 | 0.237428 | 0.124485 | 0.166325 | 0.067291 | 0.044397 | 0.7 |
| lgbm | C_plus_state | 1.515024 | FAIL | 7.135 | 0.627 | 6.509 | PASS | 0.105207 | 0.239049 | 0.125371 | 0.166843 | 0.067701 | 0.044543 | 18.7 |
| cascade | B_plus_season | 1.5741 | PASS | 1.749 | 1.098 | 0.963 | PASS | 0.10484 | 0.237583 | 0.124868 | 0.166547 | 0.068519 | 0.050986 | 0.4 |
| cascade | A_team | 1.574144 | PASS | 1.161 | 0.516 | 0.944 | PASS | 0.10485 | 0.237463 | 0.124889 | 0.16648 | 0.068521 | 0.05101 | 0.4 |
| ridge_logit | B_plus_season | 1.57415 | FAIL | 2.32 | 1.17 | 1.15 | PASS | 0.104842 | 0.237589 | 0.124875 | 0.166553 | 0.068519 | 0.050989 | 0.5 |
| ridge_logit | A_team | 1.574176 | PASS | 1.514 | 0.534 | 1.283 | PASS | 0.104852 | 0.237458 | 0.124897 | 0.166483 | 0.06852 | 0.051013 | 0.5 |
| baseline | none | 1.583032 | PASS | 0.602 | 0.602 | 0.0 | FAIL | 0.105055 | 0.238413 | 0.125785 | 0.167616 | 0.068527 | 0.051123 | 0.1 |
| lgbm | B_plus_season | 1.588557 | FAIL | 7.293 | 1.082 | 6.211 | PASS | 0.105536 | 0.239205 | 0.12558 | 0.167505 | 0.068873 | 0.051205 | 14.8 |
| lgbm | A_team | 1.589879 | FAIL | 6.739 | 0.794 | 6.567 | PASS | 0.105542 | 0.239403 | 0.125729 | 0.167503 | 0.068881 | 0.051261 | 14.9 |

**F2** -- train [2022, 2023, 2024], test [2025]  (SELECTION)

| arm | feature set | log loss | calib | worst gap (pp) | of which level (pp) | residual shape (pp) | respons. | Brier TOV | Brier rim | Brier jump2 | Brier 3 | Brier FT-shoot | Brier FT-bonus | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cascade | C_plus_state | 1.566837 | FAIL | 7.482 | 6.913 | 1.567 | PASS | 0.103146 | 0.223067 | 0.163806 | 0.173928 | 0.068284 | 0.04346 | 0.7 |
| cascade | D_plus_interactions | 1.56684 | FAIL | 7.359 | 6.911 | 1.568 | PASS | 0.103147 | 0.223053 | 0.163804 | 0.173943 | 0.068285 | 0.043461 | 0.7 |
| ridge_logit | C_plus_state | 1.567678 | FAIL | 7.563 | 6.919 | 1.594 | PASS | 0.103149 | 0.22306 | 0.163835 | 0.173935 | 0.068293 | 0.043469 | 1.0 |
| ridge_logit | D_plus_interactions | 1.567701 | FAIL | 7.482 | 6.922 | 1.838 | PASS | 0.103151 | 0.223052 | 0.163828 | 0.173955 | 0.068294 | 0.04347 | 1.0 |
| lgbm | C_plus_state | 1.572479 | FAIL | 12.208 | 6.933 | 5.276 | PASS | 0.103583 | 0.224121 | 0.164046 | 0.173971 | 0.06854 | 0.043538 | 21.6 |
| cascade | A_team | 1.632362 | FAIL | 7.32 | 6.845 | 1.395 | PASS | 0.103359 | 0.223175 | 0.16329 | 0.174345 | 0.069617 | 0.04967 | 0.5 |
| ridge_logit | A_team | 1.632388 | FAIL | 7.485 | 6.851 | 1.458 | PASS | 0.103363 | 0.223186 | 0.163285 | 0.174353 | 0.069617 | 0.04967 | 0.6 |
| cascade | B_plus_season | 1.634775 | FAIL | 7.717 | 6.9 | 1.321 | PASS | 0.103333 | 0.223178 | 0.164093 | 0.174247 | 0.069611 | 0.049684 | 0.6 |
| ridge_logit | B_plus_season | 1.634868 | FAIL | 7.542 | 6.898 | 1.435 | PASS | 0.103334 | 0.223157 | 0.164117 | 0.174254 | 0.069613 | 0.049688 | 0.7 |
| baseline | none | 1.642373 | FAIL | 6.735 | 6.735 | 0.0 | FAIL | 0.103553 | 0.223869 | 0.164845 | 0.175689 | 0.069627 | 0.049756 | 0.1 |
| lgbm | B_plus_season | 1.643241 | FAIL | 10.427 | 7.166 | 4.552 | PASS | 0.103819 | 0.224318 | 0.164111 | 0.174754 | 0.069838 | 0.049897 | 19.9 |
| lgbm | A_team | 1.643645 | FAIL | 12.421 | 7.19 | 5.231 | PASS | 0.103819 | 0.224665 | 0.163853 | 0.174962 | 0.069874 | 0.049862 | 18.8 |

`worst gap (pp)` is the pre-registered gate quantity: the largest absolute predicted-minus-actual gap over the ten predicted-probability deciles, maximised over the classes with a >= 5% share. It is then split, as a DIAGNOSTIC and never as a correction, into the part that is the same in every decile (`of which level` -- the class's overall rate is wrong) and the part that varies across deciles (`residual shape` -- the model orders chances wrongly). They point at different fixes, which is what `docs/SIM_GUARDRAILS.md` section 5 asks for.

### 3.3 Noise floor

**first**, fold F2:

* Tree arm (`lgbm`, `C_plus_state`), 5 spec-identical refits under different seeds: log loss [1.518965, 1.519147, 1.51906, 1.518994, 1.51909], mean 1.519051, **SD 0.000073**.
* Linear arm (`cascade`, `D_plus_interactions`), 200-replicate GAME-BLOCK bootstrap of the test set: log loss 1.533645, **SE 0.000723**. The resampling unit is the game, not the chance, because chances inside one game share lineups, officials and pace.

**cont**, fold F2:

* Tree arm (`lgbm`, `C_plus_state`), 5 spec-identical refits under different seeds: log loss [1.572479, 1.571743, 1.57175, 1.571707, 1.572638], mean 1.572063, **SD 0.000456**.
* Linear arm (`cascade`, `C_plus_state`), 200-replicate GAME-BLOCK bootstrap of the test set: log loss 1.566837, **SE 0.002267**. The resampling unit is the game, not the chance, because chances inside one game share lineups, officials and pace.

### 3.4 Verdict under the pre-registered decision rule

**first** (fold F2):

* Arms scored: 11; passing both gates: **0**.
* **NO WINNER. no arm passed both pre-registered gates, so the pre-registration yields NO WINNER and nothing is adopted.**
* Noise floor: 0.000723 (tree seed SD 0.000073, linear block-bootstrap SE 0.000723).
* For reference only, the lowest F2 log loss regardless of the gates was `lgbm` + `C_plus_state` at 1.518965, with a worst gated decile gap of 2.95 pp -- of which 1.80 pp is a flat level shift and only 1.15 pp is residual shape. That decomposition is the finding: the arms order chances well and get the LEVEL of the class rates wrong, which is L11 (`docs/LEARNINGS.md`) reappearing one layer down. Nothing is adopted on this run and no post-hoc level correction is applied (`CLAUDE.md`, standing rule 'no hand tuning on engine output').

Every arm and why it failed:

| arm | feature set | F2 log loss | calib | worst gap (pp) | level (pp) | shape (pp) | respons. |
|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 1.518965 | FAIL | 2.954 | 1.802 | 1.152 | PASS |
| cascade | D_plus_interactions | 1.533645 | FAIL | 2.19 | 1.708 | 2.105 | PASS |
| cascade | C_plus_state | 1.533648 | FAIL | 2.229 | 1.697 | 2.148 | PASS |
| ridge_logit | D_plus_interactions | 1.53427 | FAIL | 2.09 | 1.652 | 1.901 | PASS |
| ridge_logit | C_plus_state | 1.53427 | FAIL | 2.061 | 1.641 | 1.929 | PASS |
| cascade | B_plus_season | 1.621769 | FAIL | 2.093 | 1.636 | 0.544 | PASS |
| ridge_logit | B_plus_season | 1.621815 | FAIL | 2.134 | 1.615 | 0.663 | PASS |
| lgbm | B_plus_season | 1.622017 | FAIL | 2.592 | 1.82 | 0.958 | PASS |
| cascade | A_team | 1.622213 | FAIL | 2.449 | 1.892 | 0.556 | PASS |
| ridge_logit | A_team | 1.622292 | FAIL | 2.598 | 1.894 | 0.703 | PASS |
| lgbm | A_team | 1.623143 | FAIL | 2.78 | 1.934 | 1.043 | PASS |

**cont** (fold F2):

* Arms scored: 11; passing both gates: **0**.
* **NO WINNER. no arm passed both pre-registered gates, so the pre-registration yields NO WINNER and nothing is adopted.**
* Noise floor: 0.002267 (tree seed SD 0.000456, linear block-bootstrap SE 0.002267).
* For reference only, the lowest F2 log loss regardless of the gates was `cascade` + `C_plus_state` at 1.566837, with a worst gated decile gap of 7.48 pp -- of which 6.91 pp is a flat level shift and only 1.57 pp is residual shape. That decomposition is the finding: the arms order chances well and get the LEVEL of the class rates wrong, which is L11 (`docs/LEARNINGS.md`) reappearing one layer down. Nothing is adopted on this run and no post-hoc level correction is applied (`CLAUDE.md`, standing rule 'no hand tuning on engine output').

Every arm and why it failed:

| arm | feature set | F2 log loss | calib | worst gap (pp) | level (pp) | shape (pp) | respons. |
|---|---|---|---|---|---|---|---|
| cascade | C_plus_state | 1.566837 | FAIL | 7.482 | 6.913 | 1.567 | PASS |
| cascade | D_plus_interactions | 1.56684 | FAIL | 7.359 | 6.911 | 1.568 | PASS |
| ridge_logit | C_plus_state | 1.567678 | FAIL | 7.563 | 6.919 | 1.594 | PASS |
| ridge_logit | D_plus_interactions | 1.567701 | FAIL | 7.482 | 6.922 | 1.838 | PASS |
| lgbm | C_plus_state | 1.572479 | FAIL | 12.208 | 6.933 | 5.276 | PASS |
| cascade | A_team | 1.632362 | FAIL | 7.32 | 6.845 | 1.395 | PASS |
| ridge_logit | A_team | 1.632388 | FAIL | 7.485 | 6.851 | 1.458 | PASS |
| cascade | B_plus_season | 1.634775 | FAIL | 7.717 | 6.9 | 1.321 | PASS |
| ridge_logit | B_plus_season | 1.634868 | FAIL | 7.542 | 6.898 | 1.435 | PASS |
| lgbm | B_plus_season | 1.643241 | FAIL | 10.427 | 7.166 | 4.552 | PASS |
| lgbm | A_team | 1.643645 | FAIL | 12.421 | 7.19 | 5.231 | PASS |

### 3.5 State and interaction blocks: kept or rejected

**first**:

| block | from | to | log loss gain | noise floor | verdict |
|---|---|---|---|---|---|
| season block | A_team | B_plus_season | 0.001126 | 0.000723 | KEPT |
| state block | B_plus_season | C_plus_state | 0.103052 | 0.000723 | KEPT |

**cont**:

| block | from | to | log loss gain | noise floor | verdict |
|---|---|---|---|---|---|
| season block | A_team | B_plus_season | -0.002413 | 0.002267 | REJECTED |
| state block | B_plus_season | C_plus_state | 0.067938 | 0.002267 | KEPT |
| explicit interaction block | C_plus_state | D_plus_interactions | -3e-06 | 0.002267 | REJECTED |

### 3.6 Per-class decile calibration of the lowest-loss arm `lgbm` + `C_plus_state` (NOT ADOPTED -- it failed a gate) (F2, `first`)

| class | share % | max abs decile gap (pp) | level shift (pp) | residual shape (pp) | gated |
|---|---|---|---|---|---|
| TOV | 15.539 | 1.026 | -0.083 | 1.109 | yes |
| FGA_rim | 25.334 | 1.022 | -0.424 | 0.599 | yes |
| FGA_jump2 | 18.264 | 2.954 | 1.802 | 1.152 | yes |
| FGA_3 | 29.553 | 1.659 | -1.145 | 1.133 | yes |
| FT_trip_shooting | 6.178 | 0.284 | -0.067 | 0.22 | yes |
| FT_trip_bonus | 5.133 | 0.649 | -0.083 | 0.566 | yes |

### 3.7 Responsiveness of the lowest-loss arm `lgbm` + `C_plus_state` (NOT ADOPTED -- it failed a gate) (F2, `first`)

The matchup-specific rule: bucket the test chances by the offence's own as-of rate quintile and check the predicted class share slopes with the actual one instead of sitting flat at the league mean.

`off_3pa_c->FGA_3` -- predicted [0.23942, 0.26267, 0.28286, 0.30425, 0.33118], actual [0.24922, 0.2762, 0.29321, 0.31528, 0.3437]; predicted moves with the actual direction in 4/4 steps (gate: >= 3); span predicted +0.09176 vs actual +0.09448, slope ratio 0.9712.
`off_rim_c->FGA_rim` -- predicted [0.21829, 0.23448, 0.24886, 0.26286, 0.281], actual [0.22285, 0.24088, 0.25157, 0.26705, 0.28434]; predicted moves with the actual direction in 4/4 steps (gate: >= 3); span predicted +0.06271 vs actual +0.06150, slope ratio 1.0197.
`off_tov_c->TOV` -- predicted [0.142, 0.14653, 0.15433, 0.16009, 0.16982], actual [0.14223, 0.14708, 0.156, 0.16056, 0.17105]; predicted moves with the actual direction in 4/4 steps (gate: >= 3); span predicted +0.02782 vs actual +0.02882, slope ratio 0.9654.

### 3.8 By-state calibration of the lowest-loss arm `lgbm` + `C_plus_state` (NOT ADOPTED -- it failed a gate) (F2, `first`)

| segment | n | log loss | max abs gap (pp) | TOV pred/act % | rim pred/act % | 3 pred/act % |
|---|---|---|---|---|---|---|
| transition | 127983 | 1.48376 | 1.547 | 19.283/19.075 | 32.497/34.044 | 21.147/20.889 |
| half_court | 633610 | 1.52607 | 1.9 | 14.682/14.824 | 23.377/23.574 | 29.874/31.303 |
| bonus | 229630 | 1.61378 | 1.537 | 13.88/13.952 | 23.018/23.342 | 25.053/25.878 |
| no_bonus | 531963 | 1.47803 | 1.916 | 16.135/16.223 | 25.726/26.194 | 29.856/31.139 |
| late_clock | 82186 | 1.47177 | 1.291 | 13.821/13.845 | 21.677/22.05 | 27.165/27.755 |
| not_late_clock | 679407 | 1.52467 | 1.863 | 15.653/15.743 | 25.301/25.731 | 28.558/29.77 |


### 3.9 Addendum: the interaction block on the `first` population

Section 3.5 reports the block ladder on the lowest-loss arm of each population. For `first` that arm is `lgbm`, which by pre-registration has no `D_plus_interactions` row (a tree finds interactions itself), so the interaction block was omitted from that table rather than reported. It is answered here on both linear arms, from the same `grid_results.csv`, with no refit:

| arm | C_plus_state | D_plus_interactions | gain | noise floor | verdict |
|---|---|---|---|---|---|
| ridge_logit | 1.534270 | 1.534270 | +0.000000 | 0.000723 | REJECTED |
| cascade | 1.533648 | 1.533645 | +0.000003 | 0.000723 | REJECTED |

**REJECTED on both linear arms.** The explicit offence-rate x defence-allowed products move F2 log loss by at most 3e-06 against a noise floor of 0.000723 -- a factor of 240 below it. The linear arms already carry both sides of each matchup as main effects, and on 2.2M training rows the product terms add nothing that the main effects and the ridge penalty do not already represent. Recorded as rejected in `features.md` section 3.

`scripts/train_possession_outcome_v1.py::rejected_state_features` has since been changed to fall back to the best LINEAR arm for any block the chosen arm cannot measure, so a future run reports this row automatically instead of omitting it. This addendum is appended rather than folded into section 3.5 because `experiments.md` is append-only.


---

## 4. Round 2 pre-registration (PM-authored, 2026-09-10)

Round 2 changes only three things relative to round 1: (i) the event layer has the location-based rim override and first-chance-only feature sources (data fixes, documented above); (ii) the universe is restricted to pbp_complete games; (iii) a TRAINING-SCHEME dimension is added because L4/L11 show real year-over-year shot-mix drift that a static fit with a season index cannot extrapolate. Arms: model class in {ridge_logit, cascade, lgbm} x feature set C_plus_state (the round-1 best for every class; A/B/D are not rerun) x training scheme in {S0 static (as round 1), S1 in-season walk-forward: refit at each month boundary of the test season using all prior seasons plus the test season to date (strictly before the refit date), S2 exponential recency weighting on game date with the half-life fitted on F1 only (grid 90, 180, 365, 730 days)}. Populations: first and cont, separately. Folds, metrics, calibration gate (<= 2.0 pp worst decile gap on classes with share >= 5%), responsiveness gate, noise floor and decision rules are unchanged from round 1. S1 is evaluated only on the test season's games strictly after each refit date (no game contributes to its own fit). If S1 or S2 wins, the same scheme is the default for every later sub-model unless its own bake-off says otherwise.

### 4.1 Execution notes (worker, written before the run, after the text above and changing none of it)

These record how the pre-registered words were turned into code, so the run is reproducible and so
any place the code could have been read two ways is settled in writing rather than after the fact.

* **Trainer:** `scripts/train_possession_outcome_v2.py`. Artifacts:
  `data/processed/models/possession_outcome/round2/`. Round 1's artifacts are not touched.
* **"metrics ... unchanged from round 1"** is enforced by the call graph, not by prose: the round-2
  trainer imports `score()` from `scripts/train_possession_outcome_v1.py` and calls it, so every arm
  in both rounds goes through one scoring function. The decision rule is transcribed as `decide_v2`
  with exactly one addition the round-1 rule could not have had -- a tie-break over the training
  schemes, ordered S0 < S2 < S1 by number of fitted objects and by operational cost.
* **The event layer (i).** Rim override: `cbb_sim.pbp.events`, threshold 2.27 ft = the pooled 50th
  percentile of `DunkShot` release distance, selected from a ladder of eight stated quantiles by a
  rule fixed before the rungs were measured
  (`docs/tests/possessions_build_v2_2026-09-10.md` section 3.1). First-chance-only style rates:
  `cbb_sim.models.possession_outcome.STYLE_SOURCES`, `style_source="first_chance"`. Tables:
  `data/processed/possessions_v2/`.
* **The universe (ii).** `pbp_complete` is defined in `cbb_sim.data.universe` and reconciles the two
  completeness numbers that were in circulation
  (`docs/tests/possessions_build_v2_2026-09-10.md` section 2.1). It is applied by
  `build_design(require_pbp_complete=True)` to the design AND to the games the as-of style rates are
  accumulated over -- a rate built partly from games whose event stream is short of the box score
  would be a rate of a different quantity.
* **S1's partition (iii).** The refit dates are the first day of every calendar month containing a
  test-season game. Each refit uses games STRICTLY BEFORE its own date (all prior seasons, plus the
  test season to date); each test game is scored by the most recent refit at or before its own game
  date. So no game is ever in its own fit -- the pre-registration's requirement -- and every test
  chance is still scored, which is what keeps S1's log loss comparable with S0's on an identical
  test set. The first month's games are therefore scored by a fit on prior seasons only, which is
  S0's fit; that is the honest reading of "walk-forward from the start of the season" and it is
  reported rather than hidden.
* **S2's reference date** is the first game date of the TEST season -- a date fixed before any test
  game is played -- and weights are `0.5 ** (age_days / half_life)` with age clipped at zero. The
  half-life is fitted on F1 per (population, model class) and applied to F2 unchanged; every grid
  point's F1 loss is reported, not just the argmin.
* **The matchup-naive baseline** is run under S0 only. It is not one of the three pre-registered
  model classes, it is excluded from selection exactly as in round 1, and it is carried because a
  grid with no floor cannot say how much of the log loss is matchup information at all.

<!-- ROUND 2 RESULTS APPENDED BELOW BY scripts/train_possession_outcome_v2.py -->

---

## 6. Round 3 pre-registration: refit alignment and opponent adjustment (Decision 9) -- 2026-09-10, written and COMMITTED before any round-3 modelling

Authority: `ARCHITECTURE_DECISIONS.md` Decision 9, **as amended 2026-09-10 (commit `40dbcbd`)**:
opponent adjustment and conference alignment are mandatory bake-off **ARMS, PENDING EVIDENCE**,
not standing rules. The reference arms are round 2's own choices -- raw-centred style rates and
the calendar-monthly S1 -- they stand unless a round-3 arm beats them beyond the noise floor, and
a tie or a loss for the adjusted or aligned arms is a legitimate result to report, not a failure
to fix.

Motivating diagnostic, run first and written up before this section:
`docs/tests/possession_outcome_conference_regime_2026-09-10.md`. It re-scored the round-2 S1
winners on fold 2 through round 2's own code path (log loss reproduced to 1e-6: 1.515428 `first`,
1.499760 `cont`) and bucketed their calibration residuals five ways. Three findings shape the
design below, and none of them decides an arm: calendar week explains MORE residual structure than
either conference-relative alignment; weeks-since-the-monthly-refit explains essentially nothing;
and the round-2 winner's 0.98 pp headline gap is 2.49 pp on non-conference games and 2.95 pp
before each team's conference boundary against a 2.0 pp gate, while conference games sit at
0.98 pp.

Held fixed from round 2 and not reopened: the model class per population (`lgbm` on `first`,
`cascade` on `cont`), the event layer (possessions v2, rim override, first-chance style sources),
the universe (`pbp_complete`), the folds, the seal, and the scoring function -- `score()` is
imported from `train_possession_outcome_v1` and called, as round 2 imported it, so all three
rounds go through one scorer.

### 6.1 The two dimensions

**Scheme (refit alignment).** Every arm keeps round 2's S1 contract exactly: each refit uses games
STRICTLY BEFORE its own date (all prior seasons plus the test season to date), and each test game
is scored by the most recent refit at or before its own date, so no game is ever in its own fit.
Only the CALENDAR changes.

| arm | refit calendar | refits per test season | complexity rank |
|---|---|---|---|
| `S1_monthly` | first day of every calendar month containing a test-season game. Round 2's S1, **the reference**; it must reproduce round 2's recorded numbers as a check | 6 | 0 |
| `S1_conf_aligned` | the monthly dates UNION every distinct date on which at least one team plays its first regular-season conference game | ~29 | 1 |
| `S1_weekly` | every Monday, from the Monday on or before the first test-season game | ~23 | 2 |
| `S1_conf_aligned_weekly` | the weekly dates UNION the conference-boundary dates | ~40 | 3 |

Per-team alignment costs nothing extra: the trainer refits once per distinct boundary date, and
the "latest refit at or before the game date" rule then gives every team a fit that already knows
about its own conference start. The boundary dates come from the PUBLISHED SCHEDULE
(`home_conference_id == away_conference_id`, both non-null, regular season only), which is known
before the season is played and contains no result, so using them to choose a refit calendar is
not a leak. Definitions and standing tests: `cbb_sim.features.conference`,
`tests/test_opponent_adjust.py`.

**Feature.** `own_ratings` (`off_c` / `def_c`) is **ALREADY opponent-adjusted** -- a ridge on
offence dummies AND defence dummies fitted jointly on the same as-of window is a regularised
simultaneous adjustment (`src/cbb_sim/ratings/own_ratings.py`) -- so it is carried unchanged in
every arm and is NOT re-adjusted; doing so would double-count. The STYLE RATES are what these arms
change.

| arm | features | complexity rank |
|---|---|---|
| `F0` | round 2's `C_plus_state`, character for character. **The reference** | 0 |
| `F1` | F0 + `is_conf_game` (first-class, audited like home/away; Decision 9b) | 1 |
| `F2` | F1 with every style rate REPLACED by its one-pass opponent-adjusted sibling | 2 |
| `F3` | F1 with every style rate REPLACED by the alternating-least-squares adjusted sibling | 3 |

Replaced, not appended: the adjusted column is the same quantity measured differently, and
carrying both would let a tree reconstruct the raw one and turn a feature-bundle comparison into a
superset comparison.

**How the as-of window and the league mean are formed, per arm.** Identically to round 2, because
F2 and F3 are built by SUBTRACTING a correction from round 2's own column rather than by
recomputing it -- so the own-rate component is bit-identical across F0-F3 and the arms differ by
the correction and by nothing else. For a rate with numerator n and denominator d:

* own-rate component: `scale * N_i(<t)/D_i(<t) - scale * N_league(<t)/D_league(<t)`, cumulative
  sums over games strictly before the team's own game (round 2's `_expanding_asof`) minus the
  league's cumulative rate on the same window. Unchanged.
* F2's correction: `[rownorm(M(t)) @ def_dev(t)]_i`, where `M(t)` is the denominator mass team i
  produced against team j strictly before date t and `def_dev(t)` is every team's allowed
  deviation on the same strictly-before window. This is the pre-registered "team rate minus the
  mean deviation of its opponents' allowed rates from league".
* F3: the same two equations solved to convergence -- alternating least squares on
  `min sum_g d_g (r_g - o_i - a_j)^2` over the team-games strictly before t, weighted by each
  game's denominator, GAUSS-SEIDEL sweeps (Jacobi diverges on the sparse November schedule graph
  and was rejected by test), re-centred within each connected component of the schedule graph
  after every sweep in a way that preserves the fit, capped at 200 iterations with tolerance 1e-6.
  Dates that hit the cap are counted and reported. Seasons are adjusted independently.
* no shrinkage, no caps, no minimum-games rule on either adjusted arm. A team with no prior games
  gets a correction of exactly 0.0, which on a league-centred scale is the league mean.

Implementation and unit tests: `src/cbb_sim/features/opponent_adjust.py`,
`tests/test_opponent_adjust.py` (including
`test_strictly_as_of_appending_future_games_changes_nothing`).

### 6.2 Folds, populations, metrics

Folds unchanged: F1 trains 2022 and 2023 and tests 2024; F2 trains 2022, 2023 and 2024 and tests
2025 and is the SELECTION fold. 2026 stays sealed. Populations `first` and `cont` fitted and
scored separately.

Metrics are round 2's, through the same imported `score()`: multiclass log loss (primary),
per-class Brier, the worst gated decile calibration gap (classes with share at least 5%, gate
2.0 pp) split into level and shape, the step-monotonicity responsiveness reading, and by-state
calibration. Round 3 adds, for every cell:

1. **Per-decile calibration gap on the regime segments**: the first four weeks of conference play
   by the offence team's own boundary (`conf4_gap_pp`) -- the segment Decision 9 predicted -- AND
   the non-conference segment (`nonconf_gap_pp`) -- the segment Stage A found the damage in. Both
   are DECISION quantities. `first4_season_gap_pp`, `conf_weeks_4plus` and `pre_boundary` are
   reported as evidence.
2. **Responsiveness by own-rating quintile** (round 2's drivers) AND **by non-conference schedule
   strength quintile** -- the mean net as-of quality of the opponents a team met in its
   non-conference games. The second is the direct test of Decision 9's claim. Both are read under
   **Decision 8**: slope ratio in [0.8, 1.2] AND monotone in at least 3 of 4 steps, with a driver
   whose realised quintile span is below 2 pp exempt from BOTH clauses and recorded as exempt. The
   frozen `score()` implements only Decision 8's step clause; the slope clause is computed and
   reported separately rather than by editing a scoring function two rounds of results already
   went through.
3. Per class, per fold, per cell throughout.

### 6.3 The design is STAGED, and the stage order IS the drop order

A full 4 x 4 x 2-fold cross is not affordable on the tree arm. Four workers share this machine, so
this run is capped at four threads, and the alignment arms multiply FITS, not rows. Measured on
2026-09-10 while reproducing the round-2 winner: **264 s per tree fit** on the fold-2 `first`
training slice (1,585 s for 6 refits). So a tree FEATURE cell costs about 26 min, a tree ALIGNMENT
cell about 2 h, and the full tree cross about 30 h. The design is therefore staged, with a hard
wall clock of **6.5 h** checked before each tree cell (a cell that starts, finishes), and
**anything the clock does not reach is written to the results as NOT RUN and never as a result.**

| stage | cells | projected cumulative |
|---|---|---|
| 1 | the FULL 4 x 4 cross for `cascade`: `cont` on BOTH folds (this IS the `cont` selection grid) and `first` on fold 2 as an INTERACTION PROBE, reported in full and explicitly not the selection metric -- the only affordable way to see the whole scheme-by-feature interaction surface on that population | ~0.8 h |
| 2 | tree reference `F0 x S1_monthly`, fold 2 | ~1.2 h |
| 3 | tree reference `F0 x S1_monthly`, fold 1 (per-fold evidence) | ~1.5 h |
| 4 | noise floor: the reference cell under a second seed, both populations | ~1.9 h |
| 5-7 | tree FEATURE ladder at `S1_monthly`, fold 2: `F1`, then `F2`, then `F3` (Decision 9a and 9b) | ~3.2 h |
| 8 | tree `F0 x S1_conf_aligned`, fold 2 (Decision 9c) | ~5.4 h |
| 9 | tree `F0 x S1_weekly`, fold 2 | ~7.1 h |
| 10 | tree interaction cell of the two ladder winners, fold 2, if they moved off the reference | -- |
| 11 | tree `F0 x S1_conf_aligned_weekly`, fold 2 | -- |

The instructed drop order -- `S1_conf_aligned_weekly` first, `F3` second -- is implemented by
construction: the budget drops from the bottom of this order upward. On the measured cost the run
is expected to reach stage 9 and to leave stages 10 and 11 NOT RUN, which overshoots the ~6 h
guidance by about an hour; that is recorded here in advance rather than discovered afterwards. The
alternative, keeping strictly under 6 h, would have cost both tree alignment arms, which are the
whole of Decision 9c.

Stage 1 runs the cheap arm first deliberately. Stage A found `weeks_since_refit` to be the
flattest axis it measured, so the alignment dimension is the LEAST likely of the three to pay; the
order buys the reference, both folds, the noise floor and the entire opponent-adjustment ladder
inside 3.2 h and lets the alignment arms run against the clock rather than the other way round.

### 6.4 Noise floor

The reference cell `F0 x S1_monthly` is refit under a SECOND SEED on fold 2 for each population,
spec-identical including the whole refit calendar (a seed-varied refit of an S1 arm has to redo
every refit or it measures a different spec). The applied floor is the larger of that seed spread
and a 200-replicate GAME-BLOCK bootstrap SE of the reference cell's fold-2 log loss. This is
PARTIAL against round 1's five seeds and is labelled so; the run is wall-clock bound and a third
seed costs another 26 min of tree time that stage 8 needs. A SEGMENT gap improvement must exceed
**0.25 pp** to count; the second-seed refit reports its own `conf4` and `nonconf` gaps so that
number can be checked against a measured spread rather than asserted.

### 6.5 Decision rule

Within each dimension **the simplest arm stands** (`S1_monthly` < `S1_conf_aligned` <
`S1_weekly` < `S1_conf_aligned_weekly` by fitted-object count; `F0` < `F1` < `F2` < `F3`) **unless
a more complex arm, while passing round 1's calibration and responsiveness gates, beats it beyond
the noise floor on the primary metric (fold-2 log loss) OR by more than 0.25 pp on the
first-four-conference-weeks gap OR by more than 0.25 pp on the non-conference gap.** Where several
arms beat the reference, the simplest whose log loss is within the floor of the best beater's
wins, exactly as round 2's scheme tie-break worked. The full cross is reported so interactions are
visible, with the tree cross's unmeasured cells named as unmeasured.

**If nothing beats the reference, the reference stands and opponent adjustment and conference
alignment remain PENDING EVIDENCE for this sub-model. That outcome is a result and is reported as
one.**

An adopted winner is a SCHEDULE, not an object (L21): the trainer writes `round3/manifest.json`
mapping `refit_date` to artifact path per population, which is the format the engine's per-game
selector reads.

### 6.6 Leak test

Every column round 3 adds -- `is_conf_game` and all eight adjusted style columns per method --
goes through the standing INV-45 change-form leak test (`cbb_sim.analysis.leak_test`,
absolute as-joined correlation with own-game margin at most 0.15) BEFORE it is read as a result,
and round 2's raw-centred columns go through the same run so the adjusted numbers are read against
columns already accepted. The numbers are reported in the results section.

### 6.7 Artifacts

Trainer: `scripts/train_possession_outcome_v3.py` (v1 and v2 untouched). Artifacts:
`data/processed/models/possession_outcome/round3/`. Shared feature code:
`src/cbb_sim/features/conference.py` and `src/cbb_sim/features/opponent_adjust.py` with
`tests/test_opponent_adjust.py` -- in the package rather than in the trainer because every other
sub-model's S1 confirmation pass will consume it. Stage-A diagnostic:
`scripts/diag_possession_outcome_conf_regime.py`.

Note for the record: round 2's RESULTS were never appended to this file -- section 5 is missing,
though `model.md` and L21 both cite it. Round 3 takes section 6 for its pre-registration and
section 7 for its results, leaving section 5 free for the PM to fill from
`data/processed/models/possession_outcome/round2/`, which holds the full grid, verdict and noise
floor.

<!-- ROUND 3 RESULTS APPENDED BELOW BY scripts/train_possession_outcome_v3.py -->

---

## 7. Round 3 full results (run 2026-09-11 03:15, `scripts/train_possession_outcome_v3.py`)

Wall clock 0.00 h against a pre-registered budget of 0.0 h, at 4 threads (four workers share the machine). Cells run: 55. Cells the budget did not reach: 3 -- listed in section 7.6 as NOT RUN, never as a result.

Design: conference flag on 0.6234 of chances; 0 D-I games carry no hoopR conference id on one side and are flagged non-conference and counted; 0 chances matched no schedule row.

### 7.1 The cross

**`first` / F2**  (SELECTION)

| stage | role | population | fold | arm | feature_arm | scheme | n_fits | log_loss | worst_gated_gap_pp | worst_gated_level_pp | worst_gated_shape_pp | conf4_gap_pp | nonconf_gap_pp | first4_season_gap_pp | calibration_pass | responsiveness_pass | ncss_slope_pass | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | interaction probe (not selection) | first | F2 | cascade | F0 | S1_conf_aligned | 29 | 1.530448 | 2.075 | 0.698 | 2.027 | 2.164 | 2.145 | 3.247 | False | True | True | 139.7 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F0 | S1_conf_aligned_weekly | 45 | 1.530318 | 2.086 | 0.652 | 2.036 | 2.168 | 1.932 | 3.123 | False | True | True | 201.4 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F0 | S1_monthly | 6 | 1.530507 | 2.092 | 0.734 | 2.043 | 2.235 | 2.177 | 3.429 | False | True | True | 34.5 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F0 | S1_weekly | 23 | 1.530323 | 2.097 | 0.652 | 2.046 | 2.194 | 1.915 | 3.071 | False | True | True | 110.5 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F1 | S1_conf_aligned | 29 | 1.530429 | 2.083 | 0.71 | 2.034 | 2.357 | 2.04 | 3.234 | False | True | True | 135.6 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F1 | S1_conf_aligned_weekly | 45 | 1.5303 | 2.113 | 0.656 | 2.062 | 2.337 | 1.81 | 3.099 | False | True | True | 227.1 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F1 | S1_monthly | 6 | 1.530493 | 2.081 | 0.753 | 2.029 | 2.38 | 2.037 | 3.375 | False | True | True | 29.5 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F1 | S1_weekly | 23 | 1.530306 | 2.103 | 0.658 | 2.051 | 2.347 | 1.811 | 3.08 | False | True | True | 113.3 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F2 | S1_conf_aligned | 29 | 1.529627 | 2.038 | 0.718 | 1.991 | 2.225 | 1.898 | 2.281 | False | True | True | 186.9 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F2 | S1_conf_aligned_weekly | 45 | 1.529511 | 2.036 | 0.646 | 1.987 | 2.218 | 1.829 | 2.127 | False | True | True | 262.3 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F2 | S1_monthly | 6 | 1.529686 | 2.071 | 0.759 | 2.022 | 2.266 | 1.937 | 2.404 | False | True | True | 32.6 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F2 | S1_weekly | 23 | 1.529516 | 2.057 | 0.648 | 2.008 | 2.235 | 1.894 | 2.116 | False | True | True | 161.9 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F3 | S1_conf_aligned | 29 | 1.531221 | 2.205 | 0.709 | 2.154 | 2.534 | 2.453 | 4.283 | False | True | True | 162.2 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F3 | S1_conf_aligned_weekly | 45 | 1.531068 | 2.238 | 0.634 | 2.186 | 2.495 | 2.184 | 4.065 | False | True | True | 235.1 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F3 | S1_monthly | 6 | 1.531289 | 2.205 | 0.751 | 2.15 | 2.56 | 2.555 | 4.36 | False | True | True | 37.0 |
| 1 | interaction probe (not selection) | first | F2 | cascade | F3 | S1_weekly | 23 | 1.531075 | 2.236 | 0.638 | 2.183 | 2.492 | 2.191 | 4.065 | False | True | True | 122.8 |
| 2 | reference | first | F2 | lgbm | F0 | S1_monthly | 6 | 1.515428 | 0.98 | 0.477 | 1.032 | 1.166 | 2.492 | 3.832 | True | True | True | 898.7 |
| 4 | noise floor | first | F2 | lgbm | F0 | S1_monthly | 6 | 1.515541 | 0.977 | 0.465 | 1.059 | 1.242 | 2.625 | 3.704 | True | True | True | 706.1 |
| 5 | selection | first | F2 | lgbm | F1 | S1_monthly | 6 | 1.515482 | 1.01 | 0.51 | 0.978 | 1.16 | 2.392 | 3.647 | True | True | True | 3164.8 |
| 6 | selection | first | F2 | lgbm | F2 | S1_monthly | 6 | 1.515215 | 1.115 | 0.542 | 0.982 | 1.452 | 2.46 | 3.482 | True | True | True | 1628.2 |
| 7 | selection | first | F2 | lgbm | F3 | S1_monthly | 6 | 1.515203 | 1.322 | 0.57 | 0.983 | 1.189 | 2.995 | 3.982 | True | True | True | 538.2 |

**`first` / F1**

| stage | role | population | fold | arm | feature_arm | scheme | n_fits | log_loss | worst_gated_gap_pp | worst_gated_level_pp | worst_gated_shape_pp | conf4_gap_pp | nonconf_gap_pp | first4_season_gap_pp | calibration_pass | responsiveness_pass | ncss_slope_pass | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | reference | first | F1 | lgbm | F0 | S1_monthly | 6 | 1.520961 | 1.549 | 0.533 | 1.015 | nan | 2.396 | 3.242 | True | True | True | 572.1 |

**`cont` / F2**  (SELECTION)

| stage | role | population | fold | arm | feature_arm | scheme | n_fits | log_loss | worst_gated_gap_pp | worst_gated_level_pp | worst_gated_shape_pp | conf4_gap_pp | nonconf_gap_pp | first4_season_gap_pp | calibration_pass | responsiveness_pass | ncss_slope_pass | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | selection | cont | F2 | cascade | F0 | S1_conf_aligned | 29 | 1.49972 | 1.886 | 0.784 | 1.374 | 2.659 | 2.349 | 2.961 | True | True | True | 22.5 |
| 1 | selection | cont | F2 | cascade | F0 | S1_conf_aligned_weekly | 45 | 1.499636 | 1.935 | 0.711 | 1.317 | 2.714 | 2.522 | 2.818 | True | True | True | 30.0 |
| 1 | selection | cont | F2 | cascade | F0 | S1_monthly | 6 | 1.49976 | 1.859 | 0.826 | 1.403 | 2.787 | 2.609 | 2.849 | True | True | True | 5.1 |
| 4 | noise floor | cont | F2 | cascade | F0 | S1_monthly | 6 | 1.49976 | 1.859 | 0.826 | 1.403 | 2.787 | 2.609 | 2.849 | True | True | True | 4.9 |
| 1 | selection | cont | F2 | cascade | F0 | S1_weekly | 23 | 1.499639 | 1.963 | 0.714 | 1.335 | 2.87 | 2.668 | 2.866 | True | True | True | 16.0 |
| 1 | selection | cont | F2 | cascade | F1 | S1_conf_aligned | 29 | 1.499663 | 1.499 | 0.796 | 1.388 | 2.973 | 2.37 | 2.811 | True | True | True | 20.7 |
| 1 | selection | cont | F2 | cascade | F1 | S1_conf_aligned_weekly | 45 | 1.499578 | 1.525 | 0.719 | 1.399 | 2.885 | 2.318 | 2.723 | True | True | True | 33.2 |
| 1 | selection | cont | F2 | cascade | F1 | S1_monthly | 6 | 1.499699 | 1.655 | 0.853 | 1.382 | 2.928 | 2.834 | 2.853 | True | True | True | 4.6 |
| 1 | selection | cont | F2 | cascade | F1 | S1_weekly | 23 | 1.499581 | 1.529 | 0.725 | 1.363 | 2.963 | 2.444 | 2.688 | True | True | True | 18.1 |
| 1 | selection | cont | F2 | cascade | F2 | S1_conf_aligned | 29 | 1.49917 | 1.546 | 0.807 | 1.395 | 2.21 | 2.968 | 4.016 | True | True | True | 26.5 |
| 1 | selection | cont | F2 | cascade | F2 | S1_conf_aligned_weekly | 45 | 1.499107 | 1.639 | 0.726 | 1.433 | 2.93 | 3.008 | 4.017 | True | True | True | 40.7 |
| 1 | selection | cont | F2 | cascade | F2 | S1_monthly | 6 | 1.499206 | 1.621 | 0.861 | 1.416 | 2.569 | 3.286 | 3.94 | True | True | True | 5.0 |
| 1 | selection | cont | F2 | cascade | F2 | S1_weekly | 23 | 1.49911 | 1.652 | 0.731 | 1.415 | 2.856 | 3.162 | 4.135 | True | True | True | 23.6 |
| 1 | selection | cont | F2 | cascade | F3 | S1_conf_aligned | 29 | 1.500195 | 1.812 | 0.792 | 1.452 | 3.703 | 3.477 | 5.072 | True | True | True | 23.2 |
| 1 | selection | cont | F2 | cascade | F3 | S1_conf_aligned_weekly | 45 | 1.5001 | 1.997 | 0.714 | 1.475 | 3.891 | 3.395 | 4.932 | True | True | True | 34.8 |
| 1 | selection | cont | F2 | cascade | F3 | S1_monthly | 6 | 1.500235 | 1.84 | 0.849 | 1.42 | 3.86 | 3.558 | 5.064 | True | True | True | 5.9 |
| 1 | selection | cont | F2 | cascade | F3 | S1_weekly | 23 | 1.500105 | 1.942 | 0.72 | 1.466 | 4.029 | 3.309 | 4.85 | True | True | True | 23.0 |

**`cont` / F1**

| stage | role | population | fold | arm | feature_arm | scheme | n_fits | log_loss | worst_gated_gap_pp | worst_gated_level_pp | worst_gated_shape_pp | conf4_gap_pp | nonconf_gap_pp | first4_season_gap_pp | calibration_pass | responsiveness_pass | ncss_slope_pass | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | selection | cont | F1 | cascade | F0 | S1_conf_aligned | 29 | 1.498981 | 1.157 | 0.63 | 1.345 | nan | 2.435 | 3.308 | True | True | True | 14.5 |
| 1 | selection | cont | F1 | cascade | F0 | S1_conf_aligned_weekly | 46 | 1.498885 | 1.242 | 0.58 | 1.494 | nan | 2.301 | 2.811 | True | True | True | 19.4 |
| 1 | selection | cont | F1 | cascade | F0 | S1_monthly | 6 | 1.499063 | 1.197 | 0.647 | 1.233 | nan | 2.251 | 3.213 | True | True | True | 5.0 |
| 1 | selection | cont | F1 | cascade | F0 | S1_weekly | 23 | 1.498895 | 1.222 | 0.586 | 1.479 | nan | 2.166 | 2.948 | True | True | True | 10.4 |
| 1 | selection | cont | F1 | cascade | F1 | S1_conf_aligned | 29 | 1.498901 | 1.406 | 0.632 | 1.442 | nan | 2.27 | 2.964 | True | True | True | 12.9 |
| 1 | selection | cont | F1 | cascade | F1 | S1_conf_aligned_weekly | 46 | 1.498797 | 1.476 | 0.577 | 1.592 | nan | 2.164 | 2.792 | True | True | True | 21.2 |
| 1 | selection | cont | F1 | cascade | F1 | S1_monthly | 6 | 1.498993 | 1.262 | 0.658 | 1.376 | nan | 2.146 | 3.251 | True | True | True | 3.1 |
| 1 | selection | cont | F1 | cascade | F1 | S1_weekly | 23 | 1.498808 | 1.559 | 0.584 | 1.586 | nan | 1.955 | 2.739 | True | True | True | 11.0 |
| 1 | selection | cont | F1 | cascade | F2 | S1_conf_aligned | 29 | 1.498197 | 1.318 | 0.649 | 1.143 | nan | 2.072 | 2.688 | True | True | True | 16.2 |
| 1 | selection | cont | F1 | cascade | F2 | S1_conf_aligned_weekly | 46 | 1.498117 | 1.293 | 0.587 | 1.105 | nan | 1.838 | 2.39 | True | True | True | 31.6 |
| 1 | selection | cont | F1 | cascade | F2 | S1_monthly | 6 | 1.498283 | 1.515 | 0.677 | 1.336 | nan | 1.849 | 2.652 | True | True | True | 3.3 |
| 1 | selection | cont | F1 | cascade | F2 | S1_weekly | 23 | 1.498127 | 1.314 | 0.594 | 1.124 | nan | 1.978 | 2.318 | True | True | True | 16.5 |
| 1 | selection | cont | F1 | cascade | F3 | S1_conf_aligned | 29 | 1.499195 | 1.479 | 0.626 | 1.726 | nan | 1.721 | 2.804 | True | True | True | 15.4 |
| 1 | selection | cont | F1 | cascade | F3 | S1_conf_aligned_weekly | 46 | 1.499062 | 1.557 | 0.57 | 1.838 | nan | 1.718 | 2.982 | True | True | True | 22.4 |
| 1 | selection | cont | F1 | cascade | F3 | S1_monthly | 6 | 1.499299 | 1.487 | 0.654 | 1.714 | nan | 1.724 | 2.873 | True | True | True | 4.0 |
| 1 | selection | cont | F1 | cascade | F3 | S1_weekly | 23 | 1.499074 | 1.546 | 0.577 | 1.832 | nan | 1.786 | 3.014 | True | True | True | 15.0 |

`conf4_gap_pp` is the worst gated decile calibration gap over the chances in the FIRST FOUR WEEKS OF CONFERENCE PLAY, by the offence team's own boundary -- the segment Decision 9 predicts a mis-aligned refit damages. `ncss_slope_pass` is Decision 8's slope reading against the non-conference-schedule-strength quintile, which is the direct test of Decision 9's claim.

### 7.2 Noise floor

* **`first`**: reference cell `F0 x S1_monthly` refit under a second seed -- log loss 1.515428 (seed 0) vs 1.515541 (seed 1), **spread 0.000113**; first-4-conference-weeks gap 1.166 vs 1.242 pp; 200-replicate game-block bootstrap SE 0.000804. Applied floor **0.000804**. PARTIAL: 2 seeds, against the 5 round 1 pre-registered; the round-3 pre-registration asks for one second seed and the run is wall-clock bound.
* **`cont`**: reference cell `F0 x S1_monthly` refit under a second seed -- log loss 1.499760 (seed 0) vs 1.499760 (seed 1), **spread 0.000000**; first-4-conference-weeks gap 2.787 vs 2.787 pp; 200-replicate game-block bootstrap SE 0.001982. Applied floor **0.001982**. PARTIAL: 2 seeds, against the 5 round 1 pre-registered; the round-3 pre-registration asks for one second seed and the run is wall-clock bound.

### 7.3 Decision

**`first`** (fold F2, arm `lgbm`, noise floor 0.000804)

*scheme ladder*

| scheme | log_loss | gain_vs_reference | conf4_gap_pp | conf4_gain_vs_reference_pp | nonconf_gap_pp | nonconf_gain_vs_reference_pp | gates_pass | beats_reference_beyond_floor |
|---|---|---|---|---|---|---|---|---|
| S1_monthly | 1.515428 | 0.0 | 1.166 | 0.0 | 2.492 | 0.0 | True | False |
| S1_monthly | 1.515541 | -0.000113 | 1.242 | -0.076 | 2.625 | -0.133 | True | False |

Winner: `S1_monthly` -- the simplest arm `S1_monthly` stands -- no more complex arm beat it by more than the noise floor 0.00080 on log loss or by more than 0.25 pp on the first-4-conference-weeks gap. Under the pre-registration that is a RESULT, not a failure.

*feature ladder*

| feature_arm | log_loss | gain_vs_reference | conf4_gap_pp | conf4_gain_vs_reference_pp | nonconf_gap_pp | nonconf_gain_vs_reference_pp | gates_pass | beats_reference_beyond_floor |
|---|---|---|---|---|---|---|---|---|
| F0 | 1.515428 | 0.0 | 1.166 | 0.0 | 2.492 | 0.0 | True | False |
| F0 | 1.515541 | -0.000113 | 1.242 | -0.076 | 2.625 | -0.133 | True | False |
| F1 | 1.515482 | -5.4e-05 | 1.16 | 0.006 | 2.392 | 0.1 | True | False |
| F2 | 1.515215 | 0.000213 | 1.452 | -0.286 | 2.46 | 0.032 | True | False |
| F3 | 1.515203 | 0.000225 | 1.189 | -0.023 | 2.995 | -0.503 | True | False |

Winner: `F0` -- the simplest arm `F0` stands -- no more complex arm beat it by more than the noise floor 0.00080 on log loss or by more than 0.25 pp on the first-4-conference-weeks gap. Under the pre-registration that is a RESULT, not a failure.

**Selected cell: `F0` x `S1_monthly`.** This is the reference cell: the round-3 arms did not beat it beyond the floor, which under the pre-registration is a RESULT and not a failure -- opponent adjustment and conference alignment stay PENDING EVIDENCE.

**`cont`** (fold F2, arm `cascade`, noise floor 0.001982)

*scheme ladder*

| scheme | log_loss | gain_vs_reference | conf4_gap_pp | conf4_gain_vs_reference_pp | nonconf_gap_pp | nonconf_gain_vs_reference_pp | gates_pass | beats_reference_beyond_floor |
|---|---|---|---|---|---|---|---|---|
| S1_monthly | 1.49976 | 0.0 | 2.787 | 0.0 | 2.609 | 0.0 | True | False |
| S1_monthly | 1.49976 | 0.0 | 2.787 | 0.0 | 2.609 | 0.0 | True | False |
| S1_conf_aligned | 1.49972 | 4e-05 | 2.659 | 0.128 | 2.349 | 0.26 | True | True |
| S1_weekly | 1.499639 | 0.000121 | 2.87 | -0.083 | 2.668 | -0.059 | True | False |
| S1_conf_aligned_weekly | 1.499636 | 0.000124 | 2.714 | 0.073 | 2.522 | 0.087 | True | False |

Winner: `S1_conf_aligned` -- `S1_conf_aligned` beats the reference `S1_monthly` beyond the floor (log-loss gain 0.00004 against floor 0.00198; first-4-conference-weeks gap 2.787 -> 2.659 pp) and is the simplest arm within the floor of the best beater (1.499720).

*feature ladder*

| feature_arm | log_loss | gain_vs_reference | conf4_gap_pp | conf4_gain_vs_reference_pp | nonconf_gap_pp | nonconf_gain_vs_reference_pp | gates_pass | beats_reference_beyond_floor |
|---|---|---|---|---|---|---|---|---|
| F0 | 1.49976 | 0.0 | 2.787 | 0.0 | 2.609 | 0.0 | True | False |
| F0 | 1.49976 | 0.0 | 2.787 | 0.0 | 2.609 | 0.0 | True | False |
| F1 | 1.499699 | 6.1e-05 | 2.928 | -0.141 | 2.834 | -0.225 | True | False |
| F2 | 1.499206 | 0.000554 | 2.569 | 0.218 | 3.286 | -0.677 | True | False |
| F3 | 1.500235 | -0.000475 | 3.86 | -1.073 | 3.558 | -0.949 | True | False |

Winner: `F0` -- the simplest arm `F0` stands -- no more complex arm beat it by more than the noise floor 0.00198 on log loss or by more than 0.25 pp on the first-4-conference-weeks gap. Under the pre-registration that is a RESULT, not a failure.

**Selected cell: `F0` x `S1_conf_aligned`.** The round-3 arm beat the reference beyond the floor and is adopted for this sub-model.

### 7.4 Responsiveness, both drivers

| cell | driver | class | span_pred_pp | span_act_pp | slope_ratio | steps | exempt_narrow_span | pass |
|---|---|---|---|---|---|---|---|---|
| cont|F1|cascade|F0|S1_monthly|s0 | off_3pa_c | FGA_3 | 7.818 | 8.352 | 0.9361 | 4/4 | False | True |
| cont|F1|cascade|F0|S1_monthly|s0 | off_rim_c | FGA_rim | 4.268 | 6.082 | 0.7017 | 4/4 | False | False |
| cont|F1|cascade|F0|S1_monthly|s0 | off_tov_c | TOV | 2.197 | 2.402 | 0.9143 | 4/4 | False | True |
| cont|F1|cascade|F0|S1_monthly|s0 | ncss | FGA_3 | 1.128 | 0.932 | 1.2097 | 4/4 | True | True |
| cont|F1|cascade|F0|S1_monthly|s0 | ncss | FGA_rim | -0.719 | 0.252 | -2.8495 | 3/4 | True | True |
| cont|F1|cascade|F0|S1_monthly|s0 | ncss | TOV | -0.878 | -0.693 | 1.2668 | 2/4 | True | True |
| cont|F2|cascade|F0|S1_monthly|s0 | off_3pa_c | FGA_3 | 8.325 | 9.695 | 0.8587 | 4/4 | False | True |
| cont|F2|cascade|F0|S1_monthly|s0 | off_rim_c | FGA_rim | 4.304 | 5.582 | 0.7711 | 4/4 | False | False |
| cont|F2|cascade|F0|S1_monthly|s0 | off_tov_c | TOV | 1.938 | 2.105 | 0.921 | 3/4 | False | True |
| cont|F2|cascade|F0|S1_monthly|s0 | ncss | FGA_3 | 1.228 | 0.144 | 8.5394 | 2/4 | True | True |
| cont|F2|cascade|F0|S1_monthly|s0 | ncss | FGA_rim | -1.097 | -0.496 | 2.2097 | 3/4 | True | True |
| cont|F2|cascade|F0|S1_monthly|s0 | ncss | TOV | -0.763 | -0.583 | 1.3096 | 3/4 | True | True |
| first|F2|cascade|F0|S1_monthly|s0 | off_3pa_c | FGA_3 | 8.604 | 9.36 | 0.9192 | 4/4 | False | True |
| first|F2|cascade|F0|S1_monthly|s0 | off_rim_c | FGA_rim | 5.288 | 6.083 | 0.8693 | 4/4 | False | True |
| first|F2|cascade|F0|S1_monthly|s0 | off_tov_c | TOV | 2.451 | 2.756 | 0.8893 | 4/4 | False | True |
| first|F2|cascade|F0|S1_monthly|s0 | ncss | FGA_3 | -0.214 | -0.319 | 0.6714 | 3/4 | True | True |
| first|F2|cascade|F0|S1_monthly|s0 | ncss | FGA_rim | 0.033 | -0.007 | -5.0495 | 4/4 | True | True |
| first|F2|cascade|F0|S1_monthly|s0 | ncss | TOV | -0.623 | -0.716 | 0.8706 | 2/4 | True | True |
| cont|F1|cascade|F0|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 7.844 | 8.352 | 0.9392 | 4/4 | False | True |
| cont|F1|cascade|F0|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 4.277 | 6.082 | 0.7033 | 4/4 | False | False |
| cont|F1|cascade|F0|S1_conf_aligned|s0 | off_tov_c | TOV | 2.19 | 2.402 | 0.9117 | 4/4 | False | True |
| cont|F1|cascade|F0|S1_conf_aligned|s0 | ncss | FGA_3 | 1.143 | 0.932 | 1.2261 | 4/4 | True | True |
| cont|F1|cascade|F0|S1_conf_aligned|s0 | ncss | FGA_rim | -0.754 | 0.252 | -2.9887 | 3/4 | True | True |
| cont|F1|cascade|F0|S1_conf_aligned|s0 | ncss | TOV | -0.865 | -0.693 | 1.2486 | 2/4 | True | True |
| cont|F2|cascade|F0|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 8.348 | 9.695 | 0.8611 | 4/4 | False | True |
| cont|F2|cascade|F0|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 4.307 | 5.582 | 0.7717 | 4/4 | False | False |
| cont|F2|cascade|F0|S1_conf_aligned|s0 | off_tov_c | TOV | 1.936 | 2.105 | 0.9199 | 3/4 | False | True |
| cont|F2|cascade|F0|S1_conf_aligned|s0 | ncss | FGA_3 | 1.223 | 0.144 | 8.5056 | 2/4 | True | True |
| cont|F2|cascade|F0|S1_conf_aligned|s0 | ncss | FGA_rim | -1.1 | -0.496 | 2.2164 | 3/4 | True | True |
| cont|F2|cascade|F0|S1_conf_aligned|s0 | ncss | TOV | -0.752 | -0.583 | 1.2909 | 3/4 | True | True |
| first|F2|cascade|F0|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 8.601 | 9.36 | 0.9188 | 4/4 | False | True |
| first|F2|cascade|F0|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 5.287 | 6.083 | 0.8692 | 4/4 | False | True |
| first|F2|cascade|F0|S1_conf_aligned|s0 | off_tov_c | TOV | 2.451 | 2.756 | 0.8891 | 4/4 | False | True |
| first|F2|cascade|F0|S1_conf_aligned|s0 | ncss | FGA_3 | -0.21 | -0.319 | 0.6572 | 3/4 | True | True |
| first|F2|cascade|F0|S1_conf_aligned|s0 | ncss | FGA_rim | 0.032 | -0.007 | -4.8852 | 4/4 | True | True |
| first|F2|cascade|F0|S1_conf_aligned|s0 | ncss | TOV | -0.625 | -0.716 | 0.8725 | 2/4 | True | True |
| cont|F1|cascade|F0|S1_weekly|s0 | off_3pa_c | FGA_3 | 7.83 | 8.352 | 0.9375 | 4/4 | False | True |
| cont|F1|cascade|F0|S1_weekly|s0 | off_rim_c | FGA_rim | 4.315 | 6.082 | 0.7094 | 4/4 | False | False |
| cont|F1|cascade|F0|S1_weekly|s0 | off_tov_c | TOV | 2.192 | 2.402 | 0.9123 | 4/4 | False | True |
| cont|F1|cascade|F0|S1_weekly|s0 | ncss | FGA_3 | 1.134 | 0.932 | 1.217 | 4/4 | True | True |
| cont|F1|cascade|F0|S1_weekly|s0 | ncss | FGA_rim | -0.739 | 0.252 | -2.928 | 3/4 | True | True |
| cont|F1|cascade|F0|S1_weekly|s0 | ncss | TOV | -0.861 | -0.693 | 1.243 | 2/4 | True | True |
| cont|F2|cascade|F0|S1_weekly|s0 | off_3pa_c | FGA_3 | 8.342 | 9.695 | 0.8605 | 4/4 | False | True |
| cont|F2|cascade|F0|S1_weekly|s0 | off_rim_c | FGA_rim | 4.299 | 5.582 | 0.7701 | 4/4 | False | False |
| cont|F2|cascade|F0|S1_weekly|s0 | off_tov_c | TOV | 1.919 | 2.105 | 0.9118 | 3/4 | False | True |
| cont|F2|cascade|F0|S1_weekly|s0 | ncss | FGA_3 | 1.222 | 0.144 | 8.4946 | 2/4 | True | True |
| cont|F2|cascade|F0|S1_weekly|s0 | ncss | FGA_rim | -1.098 | -0.496 | 2.2128 | 3/4 | True | True |
| cont|F2|cascade|F0|S1_weekly|s0 | ncss | TOV | -0.757 | -0.583 | 1.2998 | 3/4 | True | True |
| first|F2|cascade|F0|S1_weekly|s0 | off_3pa_c | FGA_3 | 8.596 | 9.36 | 0.9184 | 4/4 | False | True |
| first|F2|cascade|F0|S1_weekly|s0 | off_rim_c | FGA_rim | 5.275 | 6.083 | 0.8672 | 4/4 | False | True |
| first|F2|cascade|F0|S1_weekly|s0 | off_tov_c | TOV | 2.448 | 2.756 | 0.8882 | 4/4 | False | True |
| first|F2|cascade|F0|S1_weekly|s0 | ncss | FGA_3 | -0.205 | -0.319 | 0.6441 | 3/4 | True | True |
| first|F2|cascade|F0|S1_weekly|s0 | ncss | FGA_rim | 0.026 | -0.007 | -3.9341 | 4/4 | True | True |
| first|F2|cascade|F0|S1_weekly|s0 | ncss | TOV | -0.625 | -0.716 | 0.8725 | 2/4 | True | True |
| cont|F1|cascade|F0|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 7.835 | 8.352 | 0.9382 | 4/4 | False | True |
| cont|F1|cascade|F0|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 4.316 | 6.082 | 0.7096 | 4/4 | False | False |
| cont|F1|cascade|F0|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 2.193 | 2.402 | 0.9128 | 4/4 | False | True |
| cont|F1|cascade|F0|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | 1.138 | 0.932 | 1.2207 | 4/4 | True | True |
| cont|F1|cascade|F0|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | -0.744 | 0.252 | -2.9473 | 3/4 | True | True |
| cont|F1|cascade|F0|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.862 | -0.693 | 1.2441 | 2/4 | True | True |
| cont|F2|cascade|F0|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 8.349 | 9.695 | 0.8611 | 4/4 | False | True |
| cont|F2|cascade|F0|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 4.3 | 5.582 | 0.7703 | 4/4 | False | False |
| cont|F2|cascade|F0|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 1.921 | 2.105 | 0.9126 | 3/4 | False | True |
| cont|F2|cascade|F0|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | 1.22 | 0.144 | 8.4859 | 2/4 | True | True |
| cont|F2|cascade|F0|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | -1.099 | -0.496 | 2.2136 | 3/4 | True | True |
| cont|F2|cascade|F0|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.757 | -0.583 | 1.2987 | 3/4 | True | True |
| first|F2|cascade|F0|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 8.597 | 9.36 | 0.9185 | 4/4 | False | True |
| first|F2|cascade|F0|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 5.277 | 6.083 | 0.8675 | 4/4 | False | True |
| first|F2|cascade|F0|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 2.448 | 2.756 | 0.888 | 4/4 | False | True |
| first|F2|cascade|F0|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | -0.206 | -0.319 | 0.6469 | 3/4 | True | True |
| first|F2|cascade|F0|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | 0.025 | -0.007 | -3.8063 | 4/4 | True | True |
| first|F2|cascade|F0|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.624 | -0.716 | 0.872 | 2/4 | True | True |
| cont|F1|cascade|F1|S1_monthly|s0 | off_3pa_c | FGA_3 | 7.819 | 8.352 | 0.9361 | 4/4 | False | True |
| cont|F1|cascade|F1|S1_monthly|s0 | off_rim_c | FGA_rim | 4.259 | 6.082 | 0.7003 | 4/4 | False | False |
| cont|F1|cascade|F1|S1_monthly|s0 | off_tov_c | TOV | 2.174 | 2.402 | 0.9048 | 4/4 | False | True |
| cont|F1|cascade|F1|S1_monthly|s0 | ncss | FGA_3 | 1.123 | 0.932 | 1.2052 | 4/4 | True | True |
| cont|F1|cascade|F1|S1_monthly|s0 | ncss | FGA_rim | -0.733 | 0.252 | -2.9048 | 3/4 | True | True |
| cont|F1|cascade|F1|S1_monthly|s0 | ncss | TOV | -0.878 | -0.693 | 1.2671 | 2/4 | True | True |
| cont|F2|cascade|F1|S1_monthly|s0 | off_3pa_c | FGA_3 | 8.308 | 9.695 | 0.857 | 4/4 | False | True |
| cont|F2|cascade|F1|S1_monthly|s0 | off_rim_c | FGA_rim | 4.288 | 5.582 | 0.7682 | 4/4 | False | False |
| cont|F2|cascade|F1|S1_monthly|s0 | off_tov_c | TOV | 1.926 | 2.105 | 0.9154 | 3/4 | False | True |
| cont|F2|cascade|F1|S1_monthly|s0 | ncss | FGA_3 | 1.22 | 0.144 | 8.4814 | 2/4 | True | True |
| cont|F2|cascade|F1|S1_monthly|s0 | ncss | FGA_rim | -1.103 | -0.496 | 2.2229 | 3/4 | True | True |
| cont|F2|cascade|F1|S1_monthly|s0 | ncss | TOV | -0.768 | -0.583 | 1.3187 | 3/4 | True | True |
| first|F2|cascade|F1|S1_monthly|s0 | off_3pa_c | FGA_3 | 8.594 | 9.36 | 0.9181 | 4/4 | False | True |
| first|F2|cascade|F1|S1_monthly|s0 | off_rim_c | FGA_rim | 5.273 | 6.083 | 0.867 | 4/4 | False | True |
| first|F2|cascade|F1|S1_monthly|s0 | off_tov_c | TOV | 2.449 | 2.756 | 0.8887 | 4/4 | False | True |
| first|F2|cascade|F1|S1_monthly|s0 | ncss | FGA_3 | -0.216 | -0.319 | 0.6767 | 3/4 | True | True |
| first|F2|cascade|F1|S1_monthly|s0 | ncss | FGA_rim | 0.032 | -0.007 | -4.8173 | 4/4 | True | True |
| first|F2|cascade|F1|S1_monthly|s0 | ncss | TOV | -0.63 | -0.716 | 0.8798 | 2/4 | True | True |
| cont|F1|cascade|F1|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 7.845 | 8.352 | 0.9394 | 4/4 | False | True |
| cont|F1|cascade|F1|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 4.269 | 6.082 | 0.7018 | 4/4 | False | False |
| cont|F1|cascade|F1|S1_conf_aligned|s0 | off_tov_c | TOV | 2.168 | 2.402 | 0.9024 | 4/4 | False | True |
| cont|F1|cascade|F1|S1_conf_aligned|s0 | ncss | FGA_3 | 1.138 | 0.932 | 1.2205 | 4/4 | True | True |
| cont|F1|cascade|F1|S1_conf_aligned|s0 | ncss | FGA_rim | -0.765 | 0.252 | -3.0316 | 3/4 | True | True |
| cont|F1|cascade|F1|S1_conf_aligned|s0 | ncss | TOV | -0.865 | -0.693 | 1.2481 | 2/4 | True | True |
| cont|F2|cascade|F1|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 8.332 | 9.695 | 0.8595 | 4/4 | False | True |
| cont|F2|cascade|F1|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 4.292 | 5.582 | 0.7688 | 4/4 | False | False |
| cont|F2|cascade|F1|S1_conf_aligned|s0 | off_tov_c | TOV | 1.924 | 2.105 | 0.9141 | 3/4 | False | True |
| cont|F2|cascade|F1|S1_conf_aligned|s0 | ncss | FGA_3 | 1.214 | 0.144 | 8.4431 | 2/4 | True | True |
| cont|F2|cascade|F1|S1_conf_aligned|s0 | ncss | FGA_rim | -1.106 | -0.496 | 2.229 | 3/4 | True | True |
| cont|F2|cascade|F1|S1_conf_aligned|s0 | ncss | TOV | -0.758 | -0.583 | 1.3008 | 3/4 | True | True |
| first|F2|cascade|F1|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 8.592 | 9.36 | 0.9179 | 4/4 | False | True |
| first|F2|cascade|F1|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 5.272 | 6.083 | 0.8667 | 4/4 | False | True |
| first|F2|cascade|F1|S1_conf_aligned|s0 | off_tov_c | TOV | 2.449 | 2.756 | 0.8887 | 4/4 | False | True |
| first|F2|cascade|F1|S1_conf_aligned|s0 | ncss | FGA_3 | -0.212 | -0.319 | 0.6641 | 3/4 | True | True |
| first|F2|cascade|F1|S1_conf_aligned|s0 | ncss | FGA_rim | 0.031 | -0.007 | -4.7053 | 4/4 | True | True |
| first|F2|cascade|F1|S1_conf_aligned|s0 | ncss | TOV | -0.632 | -0.716 | 0.8822 | 2/4 | True | True |
| cont|F1|cascade|F1|S1_weekly|s0 | off_3pa_c | FGA_3 | 7.833 | 8.352 | 0.9379 | 4/4 | False | True |
| cont|F1|cascade|F1|S1_weekly|s0 | off_rim_c | FGA_rim | 4.307 | 6.082 | 0.7081 | 4/4 | False | False |
| cont|F1|cascade|F1|S1_weekly|s0 | off_tov_c | TOV | 2.173 | 2.402 | 0.9045 | 4/4 | False | True |
| cont|F1|cascade|F1|S1_weekly|s0 | ncss | FGA_3 | 1.131 | 0.932 | 1.2139 | 4/4 | True | True |
| cont|F1|cascade|F1|S1_weekly|s0 | ncss | FGA_rim | -0.748 | 0.252 | -2.9642 | 3/4 | True | True |
| cont|F1|cascade|F1|S1_weekly|s0 | ncss | TOV | -0.861 | -0.693 | 1.2422 | 2/4 | True | True |
| cont|F2|cascade|F1|S1_weekly|s0 | off_3pa_c | FGA_3 | 8.326 | 9.695 | 0.8588 | 4/4 | False | True |
| cont|F2|cascade|F1|S1_weekly|s0 | off_rim_c | FGA_rim | 4.287 | 5.582 | 0.768 | 4/4 | False | False |
| cont|F2|cascade|F1|S1_weekly|s0 | off_tov_c | TOV | 1.907 | 2.105 | 0.9062 | 3/4 | False | True |
| cont|F2|cascade|F1|S1_weekly|s0 | ncss | FGA_3 | 1.213 | 0.144 | 8.4368 | 2/4 | True | True |
| cont|F2|cascade|F1|S1_weekly|s0 | ncss | FGA_rim | -1.105 | -0.496 | 2.2267 | 3/4 | True | True |
| cont|F2|cascade|F1|S1_weekly|s0 | ncss | TOV | -0.761 | -0.583 | 1.3068 | 3/4 | True | True |
| first|F2|cascade|F1|S1_weekly|s0 | off_3pa_c | FGA_3 | 8.59 | 9.36 | 0.9177 | 4/4 | False | True |
| first|F2|cascade|F1|S1_weekly|s0 | off_rim_c | FGA_rim | 5.262 | 6.083 | 0.865 | 4/4 | False | True |
| first|F2|cascade|F1|S1_weekly|s0 | off_tov_c | TOV | 2.446 | 2.756 | 0.8873 | 4/4 | False | True |
| first|F2|cascade|F1|S1_weekly|s0 | ncss | FGA_3 | -0.209 | -0.319 | 0.6567 | 3/4 | True | True |
| first|F2|cascade|F1|S1_weekly|s0 | ncss | FGA_rim | 0.025 | -0.007 | -3.7769 | 4/4 | True | True |
| first|F2|cascade|F1|S1_weekly|s0 | ncss | TOV | -0.629 | -0.716 | 0.8791 | 2/4 | True | True |
| cont|F1|cascade|F1|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 7.839 | 8.352 | 0.9386 | 4/4 | False | True |
| cont|F1|cascade|F1|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 4.308 | 6.082 | 0.7084 | 4/4 | False | False |
| cont|F1|cascade|F1|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 2.174 | 2.402 | 0.9049 | 4/4 | False | True |
| cont|F1|cascade|F1|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | 1.134 | 0.932 | 1.2166 | 4/4 | True | True |
| cont|F1|cascade|F1|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | -0.753 | 0.252 | -2.9817 | 3/4 | True | True |
| cont|F1|cascade|F1|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.861 | -0.693 | 1.2423 | 2/4 | True | True |
| cont|F2|cascade|F1|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 8.334 | 9.695 | 0.8596 | 4/4 | False | True |
| cont|F2|cascade|F1|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 4.288 | 5.582 | 0.7682 | 4/4 | False | False |
| cont|F2|cascade|F1|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 1.909 | 2.105 | 0.907 | 3/4 | False | True |
| cont|F2|cascade|F1|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | 1.212 | 0.144 | 8.4261 | 2/4 | True | True |
| cont|F2|cascade|F1|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | -1.105 | -0.496 | 2.2263 | 3/4 | True | True |
| cont|F2|cascade|F1|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.761 | -0.583 | 1.3063 | 3/4 | True | True |
| first|F2|cascade|F1|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 8.591 | 9.36 | 0.9178 | 4/4 | False | True |
| first|F2|cascade|F1|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 5.264 | 6.083 | 0.8654 | 4/4 | False | True |
| first|F2|cascade|F1|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 2.445 | 2.756 | 0.8872 | 4/4 | False | True |
| first|F2|cascade|F1|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | -0.21 | -0.319 | 0.6577 | 3/4 | True | True |
| first|F2|cascade|F1|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | 0.024 | -0.007 | -3.6426 | 4/4 | True | True |
| first|F2|cascade|F1|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.629 | -0.716 | 0.8791 | 2/4 | True | True |
| cont|F1|cascade|F2|S1_monthly|s0 | off_3pa_c | FGA_3 | 7.736 | 8.352 | 0.9263 | 4/4 | False | True |
| cont|F1|cascade|F2|S1_monthly|s0 | off_rim_c | FGA_rim | 4.515 | 6.082 | 0.7424 | 4/4 | False | False |
| cont|F1|cascade|F2|S1_monthly|s0 | off_tov_c | TOV | 2.08 | 2.402 | 0.8657 | 4/4 | False | True |
| cont|F1|cascade|F2|S1_monthly|s0 | ncss | FGA_3 | 1.389 | 0.932 | 1.4898 | 4/4 | True | True |
| cont|F1|cascade|F2|S1_monthly|s0 | ncss | FGA_rim | -0.508 | 0.252 | -2.0108 | 4/4 | True | True |
| cont|F1|cascade|F2|S1_monthly|s0 | ncss | TOV | -0.87 | -0.693 | 1.2552 | 2/4 | True | True |
| cont|F2|cascade|F2|S1_monthly|s0 | off_3pa_c | FGA_3 | 8.328 | 9.695 | 0.859 | 4/4 | False | True |
| cont|F2|cascade|F2|S1_monthly|s0 | off_rim_c | FGA_rim | 4.536 | 5.582 | 0.8127 | 4/4 | False | True |
| cont|F2|cascade|F2|S1_monthly|s0 | off_tov_c | TOV | 1.848 | 2.105 | 0.8781 | 3/4 | False | True |
| cont|F2|cascade|F2|S1_monthly|s0 | ncss | FGA_3 | 1.308 | 0.144 | 9.0958 | 2/4 | True | True |
| cont|F2|cascade|F2|S1_monthly|s0 | ncss | FGA_rim | -0.942 | -0.496 | 1.897 | 3/4 | True | True |
| cont|F2|cascade|F2|S1_monthly|s0 | ncss | TOV | -0.879 | -0.583 | 1.508 | 2/4 | True | True |
| first|F2|cascade|F2|S1_monthly|s0 | off_3pa_c | FGA_3 | 8.333 | 9.36 | 0.8902 | 4/4 | False | True |
| first|F2|cascade|F2|S1_monthly|s0 | off_rim_c | FGA_rim | 5.201 | 6.083 | 0.8551 | 4/4 | False | True |
| first|F2|cascade|F2|S1_monthly|s0 | off_tov_c | TOV | 2.386 | 2.756 | 0.8655 | 4/4 | False | True |
| first|F2|cascade|F2|S1_monthly|s0 | ncss | FGA_3 | -0.034 | -0.319 | 0.1072 | 3/4 | True | True |
| first|F2|cascade|F2|S1_monthly|s0 | ncss | FGA_rim | 0.225 | -0.007 | -34.0744 | 4/4 | True | True |
| first|F2|cascade|F2|S1_monthly|s0 | ncss | TOV | -0.792 | -0.716 | 1.106 | 2/4 | True | True |
| cont|F1|cascade|F2|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 7.774 | 8.352 | 0.9308 | 4/4 | False | True |
| cont|F1|cascade|F2|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 4.509 | 6.082 | 0.7413 | 4/4 | False | False |
| cont|F1|cascade|F2|S1_conf_aligned|s0 | off_tov_c | TOV | 2.071 | 2.402 | 0.862 | 4/4 | False | True |
| cont|F1|cascade|F2|S1_conf_aligned|s0 | ncss | FGA_3 | 1.406 | 0.932 | 1.5088 | 4/4 | True | True |
| cont|F1|cascade|F2|S1_conf_aligned|s0 | ncss | FGA_rim | -0.539 | 0.252 | -2.1357 | 4/4 | True | True |
| cont|F1|cascade|F2|S1_conf_aligned|s0 | ncss | TOV | -0.855 | -0.693 | 1.2341 | 2/4 | True | True |
| cont|F2|cascade|F2|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 8.355 | 9.695 | 0.8618 | 4/4 | False | True |
| cont|F2|cascade|F2|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 4.536 | 5.582 | 0.8127 | 4/4 | False | True |
| cont|F2|cascade|F2|S1_conf_aligned|s0 | off_tov_c | TOV | 1.846 | 2.105 | 0.8773 | 3/4 | False | True |
| cont|F2|cascade|F2|S1_conf_aligned|s0 | ncss | FGA_3 | 1.304 | 0.144 | 9.0702 | 2/4 | True | True |
| cont|F2|cascade|F2|S1_conf_aligned|s0 | ncss | FGA_rim | -0.947 | -0.496 | 1.907 | 3/4 | True | True |
| cont|F2|cascade|F2|S1_conf_aligned|s0 | ncss | TOV | -0.867 | -0.583 | 1.4887 | 2/4 | True | True |
| first|F2|cascade|F2|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 8.33 | 9.36 | 0.8899 | 4/4 | False | True |
| first|F2|cascade|F2|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 5.204 | 6.083 | 0.8555 | 4/4 | False | True |
| first|F2|cascade|F2|S1_conf_aligned|s0 | off_tov_c | TOV | 2.385 | 2.756 | 0.8653 | 4/4 | False | True |
| first|F2|cascade|F2|S1_conf_aligned|s0 | ncss | FGA_3 | -0.029 | -0.319 | 0.0905 | 3/4 | True | True |
| first|F2|cascade|F2|S1_conf_aligned|s0 | ncss | FGA_rim | 0.223 | -0.007 | -33.7906 | 4/4 | True | True |
| first|F2|cascade|F2|S1_conf_aligned|s0 | ncss | TOV | -0.792 | -0.716 | 1.1058 | 2/4 | True | True |
| cont|F1|cascade|F2|S1_weekly|s0 | off_3pa_c | FGA_3 | 7.791 | 8.352 | 0.9329 | 4/4 | False | True |
| cont|F1|cascade|F2|S1_weekly|s0 | off_rim_c | FGA_rim | 4.55 | 6.082 | 0.748 | 4/4 | False | False |
| cont|F1|cascade|F2|S1_weekly|s0 | off_tov_c | TOV | 2.063 | 2.402 | 0.8586 | 4/4 | False | True |
| cont|F1|cascade|F2|S1_weekly|s0 | ncss | FGA_3 | 1.398 | 0.932 | 1.4999 | 4/4 | True | True |
| cont|F1|cascade|F2|S1_weekly|s0 | ncss | FGA_rim | -0.522 | 0.252 | -2.0665 | 4/4 | True | True |
| cont|F1|cascade|F2|S1_weekly|s0 | ncss | TOV | -0.854 | -0.693 | 1.2322 | 2/4 | True | True |
| cont|F2|cascade|F2|S1_weekly|s0 | off_3pa_c | FGA_3 | 8.373 | 9.695 | 0.8637 | 4/4 | False | True |
| cont|F2|cascade|F2|S1_weekly|s0 | off_rim_c | FGA_rim | 4.548 | 5.582 | 0.8148 | 4/4 | False | True |
| cont|F2|cascade|F2|S1_weekly|s0 | off_tov_c | TOV | 1.843 | 2.105 | 0.8757 | 3/4 | False | True |
| cont|F2|cascade|F2|S1_weekly|s0 | ncss | FGA_3 | 1.303 | 0.144 | 9.0586 | 2/4 | True | True |
| cont|F2|cascade|F2|S1_weekly|s0 | ncss | FGA_rim | -0.946 | -0.496 | 1.9051 | 3/4 | True | True |
| cont|F2|cascade|F2|S1_weekly|s0 | ncss | TOV | -0.872 | -0.583 | 1.496 | 2/4 | True | True |
| first|F2|cascade|F2|S1_weekly|s0 | off_3pa_c | FGA_3 | 8.341 | 9.36 | 0.8911 | 4/4 | False | True |
| first|F2|cascade|F2|S1_weekly|s0 | off_rim_c | FGA_rim | 5.216 | 6.083 | 0.8574 | 4/4 | False | True |
| first|F2|cascade|F2|S1_weekly|s0 | off_tov_c | TOV | 2.387 | 2.756 | 0.866 | 4/4 | False | True |
| first|F2|cascade|F2|S1_weekly|s0 | ncss | FGA_3 | -0.027 | -0.319 | 0.0848 | 3/4 | True | True |
| first|F2|cascade|F2|S1_weekly|s0 | ncss | FGA_rim | 0.217 | -0.007 | -32.8335 | 4/4 | True | True |
| first|F2|cascade|F2|S1_weekly|s0 | ncss | TOV | -0.789 | -0.716 | 1.1023 | 2/4 | True | True |
| cont|F1|cascade|F2|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 7.798 | 8.352 | 0.9337 | 4/4 | False | True |
| cont|F1|cascade|F2|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 4.549 | 6.082 | 0.748 | 4/4 | False | False |
| cont|F1|cascade|F2|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 2.062 | 2.402 | 0.8584 | 4/4 | False | True |
| cont|F1|cascade|F2|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | 1.401 | 0.932 | 1.5034 | 4/4 | True | True |
| cont|F1|cascade|F2|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | -0.526 | 0.252 | -2.0841 | 4/4 | True | True |
| cont|F1|cascade|F2|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.854 | -0.693 | 1.2317 | 2/4 | True | True |
| cont|F2|cascade|F2|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 8.378 | 9.695 | 0.8642 | 4/4 | False | True |
| cont|F2|cascade|F2|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 4.549 | 5.582 | 0.8149 | 4/4 | False | True |
| cont|F2|cascade|F2|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 1.845 | 2.105 | 0.8765 | 3/4 | False | True |
| cont|F2|cascade|F2|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | 1.302 | 0.144 | 9.0504 | 2/4 | True | True |
| cont|F2|cascade|F2|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | -0.945 | -0.496 | 1.9046 | 3/4 | True | True |
| cont|F2|cascade|F2|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.871 | -0.583 | 1.4954 | 2/4 | True | True |
| first|F2|cascade|F2|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 8.34 | 9.36 | 0.891 | 4/4 | False | True |
| first|F2|cascade|F2|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 5.217 | 6.083 | 0.8576 | 4/4 | False | True |
| first|F2|cascade|F2|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 2.387 | 2.756 | 0.8659 | 4/4 | False | True |
| first|F2|cascade|F2|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | -0.028 | -0.319 | 0.0869 | 3/4 | True | True |
| first|F2|cascade|F2|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | 0.216 | -0.007 | -32.7325 | 4/4 | True | True |
| first|F2|cascade|F2|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.789 | -0.716 | 1.1019 | 2/4 | True | True |
| cont|F1|cascade|F3|S1_monthly|s0 | off_3pa_c | FGA_3 | 6.903 | 8.352 | 0.8265 | 4/4 | False | True |
| cont|F1|cascade|F3|S1_monthly|s0 | off_rim_c | FGA_rim | 3.631 | 6.082 | 0.597 | 4/4 | False | False |
| cont|F1|cascade|F3|S1_monthly|s0 | off_tov_c | TOV | 1.883 | 2.402 | 0.7838 | 4/4 | False | False |
| cont|F1|cascade|F3|S1_monthly|s0 | ncss | FGA_3 | 1.367 | 0.932 | 1.4667 | 4/4 | True | True |
| cont|F1|cascade|F3|S1_monthly|s0 | ncss | FGA_rim | -0.454 | 0.252 | -1.7995 | 4/4 | True | True |
| cont|F1|cascade|F3|S1_monthly|s0 | ncss | TOV | -0.913 | -0.693 | 1.3173 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_monthly|s0 | off_3pa_c | FGA_3 | 7.498 | 9.695 | 0.7734 | 4/4 | False | False |
| cont|F2|cascade|F3|S1_monthly|s0 | off_rim_c | FGA_rim | 3.684 | 5.582 | 0.66 | 4/4 | False | False |
| cont|F2|cascade|F3|S1_monthly|s0 | off_tov_c | TOV | 1.653 | 2.105 | 0.7855 | 3/4 | False | False |
| cont|F2|cascade|F3|S1_monthly|s0 | ncss | FGA_3 | 1.306 | 0.144 | 9.0806 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_monthly|s0 | ncss | FGA_rim | -0.857 | -0.496 | 1.7274 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_monthly|s0 | ncss | TOV | -0.9 | -0.583 | 1.5443 | 2/4 | True | True |
| first|F2|cascade|F3|S1_monthly|s0 | off_3pa_c | FGA_3 | 7.546 | 9.36 | 0.8061 | 4/4 | False | True |
| first|F2|cascade|F3|S1_monthly|s0 | off_rim_c | FGA_rim | 4.384 | 6.083 | 0.7208 | 4/4 | False | False |
| first|F2|cascade|F3|S1_monthly|s0 | off_tov_c | TOV | 2.059 | 2.756 | 0.747 | 4/4 | False | False |
| first|F2|cascade|F3|S1_monthly|s0 | ncss | FGA_3 | -0.004 | -0.319 | 0.0127 | 3/4 | True | True |
| first|F2|cascade|F3|S1_monthly|s0 | ncss | FGA_rim | 0.299 | -0.007 | -45.1975 | 4/4 | True | True |
| first|F2|cascade|F3|S1_monthly|s0 | ncss | TOV | -0.782 | -0.716 | 1.0925 | 2/4 | True | True |
| cont|F1|cascade|F3|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 6.933 | 8.352 | 0.8301 | 4/4 | False | True |
| cont|F1|cascade|F3|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 3.638 | 6.082 | 0.5982 | 4/4 | False | False |
| cont|F1|cascade|F3|S1_conf_aligned|s0 | off_tov_c | TOV | 1.876 | 2.402 | 0.781 | 4/4 | False | False |
| cont|F1|cascade|F3|S1_conf_aligned|s0 | ncss | FGA_3 | 1.381 | 0.932 | 1.4818 | 4/4 | True | True |
| cont|F1|cascade|F3|S1_conf_aligned|s0 | ncss | FGA_rim | -0.484 | 0.252 | -1.9189 | 4/4 | True | True |
| cont|F1|cascade|F3|S1_conf_aligned|s0 | ncss | TOV | -0.9 | -0.693 | 1.2988 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 7.52 | 9.695 | 0.7756 | 4/4 | False | False |
| cont|F2|cascade|F3|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 3.679 | 5.582 | 0.6591 | 4/4 | False | False |
| cont|F2|cascade|F3|S1_conf_aligned|s0 | off_tov_c | TOV | 1.647 | 2.105 | 0.7825 | 3/4 | False | False |
| cont|F2|cascade|F3|S1_conf_aligned|s0 | ncss | FGA_3 | 1.301 | 0.144 | 9.0464 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_conf_aligned|s0 | ncss | FGA_rim | -0.861 | -0.496 | 1.7346 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_conf_aligned|s0 | ncss | TOV | -0.888 | -0.583 | 1.5238 | 2/4 | True | True |
| first|F2|cascade|F3|S1_conf_aligned|s0 | off_3pa_c | FGA_3 | 7.542 | 9.36 | 0.8057 | 4/4 | False | True |
| first|F2|cascade|F3|S1_conf_aligned|s0 | off_rim_c | FGA_rim | 4.385 | 6.083 | 0.7209 | 4/4 | False | False |
| first|F2|cascade|F3|S1_conf_aligned|s0 | off_tov_c | TOV | 2.059 | 2.756 | 0.7469 | 4/4 | False | False |
| first|F2|cascade|F3|S1_conf_aligned|s0 | ncss | FGA_3 | -0.0 | -0.319 | 0.0012 | 3/4 | True | True |
| first|F2|cascade|F3|S1_conf_aligned|s0 | ncss | FGA_rim | 0.297 | -0.007 | -44.904 | 4/4 | True | True |
| first|F2|cascade|F3|S1_conf_aligned|s0 | ncss | TOV | -0.781 | -0.716 | 1.0912 | 2/4 | True | True |
| cont|F1|cascade|F3|S1_weekly|s0 | off_3pa_c | FGA_3 | 6.934 | 8.352 | 0.8303 | 4/4 | False | True |
| cont|F1|cascade|F3|S1_weekly|s0 | off_rim_c | FGA_rim | 3.681 | 6.082 | 0.6052 | 4/4 | False | False |
| cont|F1|cascade|F3|S1_weekly|s0 | off_tov_c | TOV | 1.874 | 2.402 | 0.78 | 4/4 | False | False |
| cont|F1|cascade|F3|S1_weekly|s0 | ncss | FGA_3 | 1.374 | 0.932 | 1.4739 | 4/4 | True | True |
| cont|F1|cascade|F3|S1_weekly|s0 | ncss | FGA_rim | -0.464 | 0.252 | -1.837 | 4/4 | True | True |
| cont|F1|cascade|F3|S1_weekly|s0 | ncss | TOV | -0.899 | -0.693 | 1.2974 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_weekly|s0 | off_3pa_c | FGA_3 | 7.516 | 9.695 | 0.7753 | 4/4 | False | False |
| cont|F2|cascade|F3|S1_weekly|s0 | off_rim_c | FGA_rim | 3.657 | 5.582 | 0.6552 | 4/4 | False | False |
| cont|F2|cascade|F3|S1_weekly|s0 | off_tov_c | TOV | 1.628 | 2.105 | 0.7737 | 3/4 | False | False |
| cont|F2|cascade|F3|S1_weekly|s0 | ncss | FGA_3 | 1.295 | 0.144 | 9.0009 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_weekly|s0 | ncss | FGA_rim | -0.862 | -0.496 | 1.7377 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_weekly|s0 | ncss | TOV | -0.89 | -0.583 | 1.5281 | 2/4 | True | True |
| first|F2|cascade|F3|S1_weekly|s0 | off_3pa_c | FGA_3 | 7.53 | 9.36 | 0.8045 | 4/4 | False | True |
| first|F2|cascade|F3|S1_weekly|s0 | off_rim_c | FGA_rim | 4.371 | 6.083 | 0.7185 | 4/4 | False | False |
| first|F2|cascade|F3|S1_weekly|s0 | off_tov_c | TOV | 2.049 | 2.756 | 0.7436 | 4/4 | False | False |
| first|F2|cascade|F3|S1_weekly|s0 | ncss | FGA_3 | 0.003 | -0.319 | -0.0081 | 3/4 | True | True |
| first|F2|cascade|F3|S1_weekly|s0 | ncss | FGA_rim | 0.29 | -0.007 | -43.8554 | 4/4 | True | True |
| first|F2|cascade|F3|S1_weekly|s0 | ncss | TOV | -0.778 | -0.716 | 1.0874 | 2/4 | True | True |
| cont|F1|cascade|F3|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 6.941 | 8.352 | 0.831 | 4/4 | False | True |
| cont|F1|cascade|F3|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 3.682 | 6.082 | 0.6054 | 4/4 | False | False |
| cont|F1|cascade|F3|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 1.875 | 2.402 | 0.7804 | 4/4 | False | False |
| cont|F1|cascade|F3|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | 1.377 | 0.932 | 1.4768 | 4/4 | True | True |
| cont|F1|cascade|F3|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | -0.468 | 0.252 | -1.853 | 4/4 | True | True |
| cont|F1|cascade|F3|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.899 | -0.693 | 1.2971 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 7.523 | 9.695 | 0.776 | 4/4 | False | False |
| cont|F2|cascade|F3|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 3.659 | 5.582 | 0.6556 | 4/4 | False | False |
| cont|F2|cascade|F3|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 1.629 | 2.105 | 0.7741 | 3/4 | False | False |
| cont|F2|cascade|F3|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | 1.293 | 0.144 | 8.9909 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | -0.863 | -0.496 | 1.7383 | 2/4 | True | True |
| cont|F2|cascade|F3|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.89 | -0.583 | 1.5268 | 2/4 | True | True |
| first|F2|cascade|F3|S1_conf_aligned_weekly|s0 | off_3pa_c | FGA_3 | 7.532 | 9.36 | 0.8047 | 4/4 | False | True |
| first|F2|cascade|F3|S1_conf_aligned_weekly|s0 | off_rim_c | FGA_rim | 4.373 | 6.083 | 0.719 | 4/4 | False | False |
| first|F2|cascade|F3|S1_conf_aligned_weekly|s0 | off_tov_c | TOV | 2.05 | 2.756 | 0.7437 | 4/4 | False | False |
| first|F2|cascade|F3|S1_conf_aligned_weekly|s0 | ncss | FGA_3 | 0.001 | -0.319 | -0.0042 | 3/4 | True | True |
| first|F2|cascade|F3|S1_conf_aligned_weekly|s0 | ncss | FGA_rim | 0.289 | -0.007 | -43.7311 | 4/4 | True | True |
| first|F2|cascade|F3|S1_conf_aligned_weekly|s0 | ncss | TOV | -0.778 | -0.716 | 1.0869 | 2/4 | True | True |
| first|F2|lgbm|F0|S1_monthly|s0 | off_3pa_c | FGA_3 | 8.867 | 9.36 | 0.9473 | 4/4 | False | True |
| first|F2|lgbm|F0|S1_monthly|s0 | off_rim_c | FGA_rim | 5.972 | 6.083 | 0.9819 | 4/4 | False | True |
| first|F2|lgbm|F0|S1_monthly|s0 | off_tov_c | TOV | 2.773 | 2.756 | 1.0062 | 4/4 | False | True |
| first|F2|lgbm|F0|S1_monthly|s0 | ncss | FGA_3 | -0.413 | -0.319 | 1.2954 | 3/4 | True | True |
| first|F2|lgbm|F0|S1_monthly|s0 | ncss | FGA_rim | -0.101 | -0.007 | 15.3344 | 4/4 | True | True |
| first|F2|lgbm|F0|S1_monthly|s0 | ncss | TOV | -0.462 | -0.716 | 0.6459 | 2/4 | True | True |
| first|F1|lgbm|F0|S1_monthly|s0 | off_3pa_c | FGA_3 | 9.432 | 9.64 | 0.9784 | 4/4 | False | True |
| first|F1|lgbm|F0|S1_monthly|s0 | off_rim_c | FGA_rim | 6.044 | 6.358 | 0.9507 | 4/4 | False | True |
| first|F1|lgbm|F0|S1_monthly|s0 | off_tov_c | TOV | 3.025 | 3.286 | 0.9205 | 4/4 | False | True |
| first|F1|lgbm|F0|S1_monthly|s0 | ncss | FGA_3 | 0.002 | 0.375 | 0.0042 | 4/4 | True | True |
| first|F1|lgbm|F0|S1_monthly|s0 | ncss | FGA_rim | 0.314 | 0.642 | 0.489 | 4/4 | True | True |
| first|F1|lgbm|F0|S1_monthly|s0 | ncss | TOV | -0.901 | -1.063 | 0.8476 | 3/4 | True | True |
| first|F2|lgbm|F0|S1_monthly|s1 | off_3pa_c | FGA_3 | 8.878 | 9.36 | 0.9485 | 4/4 | False | True |
| first|F2|lgbm|F0|S1_monthly|s1 | off_rim_c | FGA_rim | 5.979 | 6.083 | 0.983 | 4/4 | False | True |
| first|F2|lgbm|F0|S1_monthly|s1 | off_tov_c | TOV | 2.789 | 2.756 | 1.0119 | 4/4 | False | True |
| first|F2|lgbm|F0|S1_monthly|s1 | ncss | FGA_3 | -0.419 | -0.319 | 1.3135 | 4/4 | True | True |
| first|F2|lgbm|F0|S1_monthly|s1 | ncss | FGA_rim | -0.151 | -0.007 | 22.8271 | 4/4 | True | True |
| first|F2|lgbm|F0|S1_monthly|s1 | ncss | TOV | -0.437 | -0.716 | 0.6105 | 1/4 | True | True |
| cont|F2|cascade|F0|S1_monthly|s1 | off_3pa_c | FGA_3 | 8.325 | 9.695 | 0.8587 | 4/4 | False | True |
| cont|F2|cascade|F0|S1_monthly|s1 | off_rim_c | FGA_rim | 4.304 | 5.582 | 0.7711 | 4/4 | False | False |
| cont|F2|cascade|F0|S1_monthly|s1 | off_tov_c | TOV | 1.938 | 2.105 | 0.921 | 3/4 | False | True |
| cont|F2|cascade|F0|S1_monthly|s1 | ncss | FGA_3 | 1.228 | 0.144 | 8.5394 | 2/4 | True | True |
| cont|F2|cascade|F0|S1_monthly|s1 | ncss | FGA_rim | -1.097 | -0.496 | 2.2097 | 3/4 | True | True |
| cont|F2|cascade|F0|S1_monthly|s1 | ncss | TOV | -0.763 | -0.583 | 1.3096 | 3/4 | True | True |
| first|F2|lgbm|F1|S1_monthly|s0 | off_3pa_c | FGA_3 | 8.834 | 9.36 | 0.9438 | 4/4 | False | True |
| first|F2|lgbm|F1|S1_monthly|s0 | off_rim_c | FGA_rim | 6.013 | 6.083 | 0.9886 | 4/4 | False | True |
| first|F2|lgbm|F1|S1_monthly|s0 | off_tov_c | TOV | 2.752 | 2.756 | 0.9984 | 4/4 | False | True |
| first|F2|lgbm|F1|S1_monthly|s0 | ncss | FGA_3 | -0.377 | -0.319 | 1.1817 | 3/4 | True | True |
| first|F2|lgbm|F1|S1_monthly|s0 | ncss | FGA_rim | -0.196 | -0.007 | 29.6271 | 4/4 | True | True |
| first|F2|lgbm|F1|S1_monthly|s0 | ncss | TOV | -0.45 | -0.716 | 0.6289 | 1/4 | True | True |
| first|F2|lgbm|F2|S1_monthly|s0 | off_3pa_c | FGA_3 | 8.389 | 9.36 | 0.8962 | 4/4 | False | True |
| first|F2|lgbm|F2|S1_monthly|s0 | off_rim_c | FGA_rim | 5.736 | 6.083 | 0.943 | 4/4 | False | True |
| first|F2|lgbm|F2|S1_monthly|s0 | off_tov_c | TOV | 2.581 | 2.756 | 0.9366 | 4/4 | False | True |
| first|F2|lgbm|F2|S1_monthly|s0 | ncss | FGA_3 | -0.157 | -0.319 | 0.4914 | 3/4 | True | True |
| first|F2|lgbm|F2|S1_monthly|s0 | ncss | FGA_rim | -0.025 | -0.007 | 3.7135 | 4/4 | True | True |
| first|F2|lgbm|F2|S1_monthly|s0 | ncss | TOV | -0.632 | -0.716 | 0.8823 | 3/4 | True | True |
| first|F2|lgbm|F3|S1_monthly|s0 | off_3pa_c | FGA_3 | 8.405 | 9.36 | 0.898 | 4/4 | False | True |
| first|F2|lgbm|F3|S1_monthly|s0 | off_rim_c | FGA_rim | 5.727 | 6.083 | 0.9416 | 4/4 | False | True |
| first|F2|lgbm|F3|S1_monthly|s0 | off_tov_c | TOV | 2.579 | 2.756 | 0.9356 | 4/4 | False | True |
| first|F2|lgbm|F3|S1_monthly|s0 | ncss | FGA_3 | -0.215 | -0.319 | 0.6737 | 3/4 | True | True |
| first|F2|lgbm|F3|S1_monthly|s0 | ncss | FGA_rim | 0.062 | -0.007 | -9.3858 | 3/4 | True | True |
| first|F2|lgbm|F3|S1_monthly|s0 | ncss | TOV | -0.608 | -0.716 | 0.8499 | 2/4 | True | True |

### 7.5 Leak test on every column round 3 adds

| column | corr_asjoined | corr_update | corr_level | n | static | verdict | level_verdict |
|---|---|---|---|---|---|---|---|
| off_3pa_c | 0.0062 | 0.027 | 0.0601 | 37347 | False | pass | pass (level-form) |
| off_rim_c | 0.017 | 0.0881 | 0.074 | 37347 | False | pass | pass (level-form) |
| off_tov_c | -0.0215 | -0.127 | -0.155 | 37347 | False | pass | LEAK (level-form) |
| off_ftr_c | 0.0078 | 0.0889 | 0.0378 | 37347 | False | pass | pass (level-form) |
| off_3pa_a1 | 0.0043 | 0.009 | 0.053 | 37347 | False | pass | pass (level-form) |
| off_3pa_a3 | -0.003 | 0.0041 | 0.0454 | 37347 | False | pass | pass (level-form) |
| off_rim_a1 | 0.0198 | 0.0371 | 0.0738 | 37347 | False | pass | pass (level-form) |
| off_rim_a3 | 0.0155 | 0.0141 | 0.0698 | 37347 | False | pass | pass (level-form) |
| off_tov_a1 | -0.0228 | -0.0707 | -0.1501 | 37347 | False | pass | LEAK (level-form) |
| off_tov_a3 | -0.0086 | -0.0458 | -0.1269 | 37347 | False | pass | pass (level-form) |
| off_ftr_a1 | 0.0058 | 0.0441 | 0.0412 | 37347 | False | pass | pass (level-form) |
| off_ftr_a3 | 0.0035 | 0.0333 | 0.0328 | 37347 | False | pass | pass (level-form) |
| is_conf_game | -0.0111 | 0.0067 | 0.0 | 37347 | False | pass | pass (level-form) |

Gate: |as-joined change-form corr| <= 0.15 (CLAUDE.md, standing rule 'backtests must be honest'). The raw-centred reference columns are shown alongside the adjusted ones so the adjusted numbers are read against a column already accepted.

### 7.6 Cells the budget did not reach (NOT RUN, not a result)

| stage | population | arm | fold | feature_arm | scheme | role | seed | reason |
|---|---|---|---|---|---|---|---|---|
| 8 | first | lgbm | F2 | F0 | S1_conf_aligned | selection | 0 | wall clock 0.00h over budget 0.0h |
| 9 | first | lgbm | F2 | F0 | S1_weekly | selection | 0 | wall clock 0.00h over budget 0.0h |
| 11 | first | lgbm | F2 | F0 | S1_conf_aligned_weekly | selection | 0 | wall clock 0.00h over budget 0.0h |


### 7.7 Execution note (worker, 2026-09-11)

Written by hand, not by the trainer, because the run's own `run_meta` sees only its last pass.

**Three passes, one checkpoint, no cell computed twice.** Pass 1 (2026-09-10 20:58 - 23:41, 2.73 h)
ran the pre-registered stage order but was launched without `--stages`, and the flag's DEFAULT
still held the earlier stage numbering `1,2,3,4,5,6` -- so stages 7-11 were never scheduled. That
is a launcher defect, not a design change: the pre-registered stage list in section 6.3 is the one
the code builds. Pass 1's partial append to this file was reverted in the working tree before it
was committed, and the pre-registration commit was untouched. Pass 2 (2026-09-11 00:12 - 03:15,
3.05 h) resumed from the checkpoint with the full stage list and added stage 7 (`F3`); it was
stopped at a directed hard stop of 03:15 while stage 8 (`F0 x S1_conf_aligned`, 29 tree refits) was
still fitting, and that cell's partial work was discarded rather than recorded. Pass 3 (03:16,
0.00 h) fitted NOTHING: it re-read the checkpoint, marked every unreached cell NOT RUN, computed
the floors, applied the decision rule and wrote this section.

**What that leaves.** The `cascade` cross is COMPLETE -- all 4 features x 4 schemes, both folds on
`cont` (the selection grid there) and fold 2 on `first` (the interaction probe). The `lgbm` grid
has the reference on both folds, its second-seed noise floor, and the entire feature ladder
(`F1`, `F2`, `F3` at `S1_monthly`). **No tree ALIGNMENT cell finished.** `S1_conf_aligned`,
`S1_weekly` and `S1_conf_aligned_weekly` on the `first` population are NOT RUN, listed in
section 7.6, and nothing about tree alignment is claimed from this run. The drop order the
pre-registration fixed was honoured in the sense that `S1_conf_aligned_weekly` and the interaction
cell were the first casualties; the hard stop then took the rest of the alignment ladder too,
which the pre-registration did not anticipate and which is recorded here rather than smoothed over.

**Cost, for whoever schedules the missing cells.** A tree fit on the fold-2 `first` training slice
ran at 150 s under a quiet machine and 530 s under a busy one, at the four-thread cap; the three
missing alignment cells are 29, 24 and 46 refits. On the 196-core box they are minutes, and they
are the one piece of Decision 9c this sub-model still owes.

**Two readings of the tables above that a reader should not have to derive.**

1. The `cont` scheme ladder's adoption of `S1_conf_aligned` is MARGINAL and rests on one number.
   Its log-loss gain is 0.00004 against a floor of 0.00198 -- a fiftieth of the floor. It clears
   the rule only on the non-conference segment gap, 2.609 -> 2.349 pp, a 0.260 pp gain against a
   0.25 pp threshold. That threshold was fixed before the run, but `cascade` is deterministic, so
   the second-seed refit measured a segment spread of exactly 0.0 pp and cannot corroborate it.
   Treat the `cont` alignment result as a lead, not a finding, until the segment floor has a
   measured spread behind it.
2. The scheme and feature ladder tables each show the reference row TWICE. The second is the
   noise-floor cell -- the same spec under seed 1 -- which the ladder builder does not filter out.
   It is left in deliberately: it puts the seed spread (0.000113 log loss, 0.076 pp conf4,
   0.133 pp non-conference) on the same rows as the gains it has to be compared against.

---

## 8. Round 4 pre-registration: early-season shrinkage of the as-of style rates, plus the two alignment cells round 3 did not run -- 2026-09-11, written and COMMITTED before any round-4 modelling

Authority and scope. Two things are open after round 3 and this round closes what the clock allows
of both.

1. **The residual structure round 3 localised is a SEASON-START effect, not a conference-regime
   effect.** The Stage-A diagnostic
   (`docs/tests/possession_outcome_conference_regime_2026-09-10.md`) found calendar week explains
   more residual structure than conference-relative alignment, and round 3's own grid confirms it:
   under every one of the 21 measured `first` cells the first-four-season-weeks gap
   (`first4_season_gap_pp`, 3.48-4.36 pp on the tree arm) is the worst segment in the table -- worse
   than the first four conference weeks (1.16-1.45 pp) and worse than non-conference as a whole
   (2.39-3.00 pp), all against a 2.0 pp gate. Nothing in rounds 1-3 addressed it. The mechanism is
   named and testable: in season weeks 0-3 a team's as-of style rate is a ratio over a handful of
   possessions, so `off_{rate}_c` is mostly sampling noise around the league mean, and a tree fed an
   unshrunk noisy deviation will act on it.
2. **No tree ALIGNMENT cell finished in round 3** (section 7.6). `S1_conf_aligned` and `S1_weekly`
   on the `first` population are the one piece of Decision 9c this sub-model still owes, and
   Decision 9 cannot be resolved for possession-outcome until they are measured.

Held fixed from rounds 2 and 3, and NOT reopened: the model class per population (`lgbm` on
`first`, `cascade` on `cont`), the event layer (possessions v2, rim override, first-chance style
sources), the universe (`pbp_complete`), the folds, the seal on 2025-26, and the scoring function.
`R1.score()` is imported and called, as rounds 2 and 3 imported it, and round 3's
`conf_window_calibration` and `responsiveness_by` are imported from
`scripts/train_possession_outcome_v3.py` **unmodified** so all four rounds go through one scorer.
The reference cell's own predictions are re-scored through the round-4 grader as a check that the
grader change is additive (see 8.5).

### 8.1 The shrinkage arms

Every arm is a transformation of ROUND 2's OWN centred style column, built from the same
strictly-before accumulations, so the arms differ by the shrinkage and by nothing else. For a rate
`r` with numerator `n` and denominator `d` (`RATE_DEFS`: 3pa = fga_3/poss, rim = fga_rim/fga,
tov = tov/poss, ftr = fta/fga), round 2's column is

```
raw_c(i, t)  =  scale * N_i(<t)/D_i(<t)  -  scale * N_league(<t)/D_league(<t)
```

with `D_i(<t)` the team's own accumulated denominator mass strictly before its own game. All eight
style columns (four rates x offence own-form and opponent defence-allowed) are transformed
together; the two ridge rating columns are NOT touched (they are already regularised, L21) and no
state, site or season column is touched.

| arm | definition | complexity rank |
|---|---|---|
| `G0` | round 2's `C_plus_state`, character for character. **The reference** (identical to round 3's `F0`) | 0 |
| `G1` | `w * raw_c`, `w = D_i(<t) / (D_i(<t) + k_r)` -- empirical-Bayes shrinkage toward the LEAGUE MEAN (a centred column's prior mean is 0 by construction) | 1 |
| `G2` | `w * raw_c + (1 - w) * prior_c`, same `w`, where `prior_c` is the SAME TEAM's full prior-season centred rate (0 -- the league mean -- if the team has no prior season in the panel) | 2 |
| `G3` | `w * raw_c + (1 - w) * w2 * prior_c`, `w2 = D_i(prev season) / (D_i(prev season) + k_r)` -- a two-level empirical Bayes in which the prior-season target is itself shrunk by its own reliability before it is used as a prior | 3 |
| `G4` | `G0` + `off_n_prior` and `def_n_prior`, the number of completed games behind each side's as-of rate, as FEATURES -- the weeks-since-season-start interaction expressed so the model discounts a thin rate itself rather than being told how much to discount it. `days_since_start` is already in `C_plus_state`; what is missing is the RELIABILITY of the rate, not the date | 1 |

`G4` is ranked 1 rather than 4 deliberately: it adds two columns and no arithmetic on an existing
one, and under "ties go to the simpler model" a two-column addition is simpler than a refitted
two-level prior. `G1` and `G4` are tied at rank 1; if both beat the reference and are within the
floor of each other, `G1` wins the tie as the arm that adds no columns at all. That tie-break is
fixed here, before the run.

**Why there is no external-preseason arm.** The lane asked for an arm shrinking toward "the
preseason prior from the existing preseason work". `data/raw/preseason/` holds 2027 only -- roster,
portal, recruiting and conference-change tables pulled for the upcoming season. There is no
preseason artifact for 2024 or 2025, the two test seasons, and building one retroactively is a
separate pull and a separate leak test, not something to improvise inside this round. `G2` and `G3`
are the constructible preseason priors: the prior season's own realised style, used before the new
season has produced any. This substitution is recorded here rather than discovered in the results.

**How the shrinkage weight is fitted, and why it is a model choice and not a post-hoc adjustment.**
`k_r` is a fitted parameter of the FEATURE BUILDER, estimated per rate and per side (offence form,
defence-allowed) from **completed PRIOR SEASONS ONLY** -- for a game in season `s`, from seasons
earlier than `s` in the panel, never from season `s` and never from the game. It is a
method-of-moments empirical-Bayes ratio of within-team sampling dispersion to between-team true
dispersion:

```
s2_r   = weighted mean over prior-season team-games of  D_g * (rate_g - rate_team_season)^2
         (the per-unit-of-denominator sampling variance)
tau2_r = weighted variance over prior-season teams of  rate_team_season - rate_league_season
         MINUS  mean_i( s2_r / D_i_season )       (between-team variance net of sampling noise)
k_r    = s2_r / max(tau2_r, eps)
```

so `w = D/(D + k_r)` is the posterior weight on the team's own observation under a normal-normal
model. `k_r` has units of denominator mass and is reported for every (rate, side, season) in the
results. Season 2022 has no prior season in the panel; its `k_r` falls back to the pooled estimate
over the seasons that do, and 2022 is a TRAIN-ONLY season in both folds, so no test row depends on
that fallback. No `k_r` is chosen by looking at a score, no value is tuned, no cap or clip is
applied to the output of any arm, and nothing downstream of the model is touched -- this is a
feature definition fitted walk-forward, which `docs/SIM_GUARDRAILS.md` allows and which the
no-hand-tuning rule is explicitly about NOT being.

**Verification that the arms differ only by the shrinkage.** The builder recomputes `raw_c` from the
same boxes and asserts it reproduces the cached round-3 design column to 1e-3 on every one of the
eight style columns before any shrunk column is written. A failure of that assertion aborts the run.

### 8.2 The alignment cells round 3 did not run

Unchanged in definition from round 3 section 6.1 -- same `refit_dates`, same S1 contract, same
`CF.conference_boundary_dates` from the published schedule:

| cell | population | arm | fold | features | scheme | n_fits |
|---|---|---|---|---|---|---|
| `A1` | first | lgbm | F2 | `G0` | `S1_conf_aligned` | ~29 |
| `A2` | first | lgbm | F2 | `G0` | `S1_weekly` | ~23 |

`S1_conf_aligned_weekly` on the tree stays out of this round: round 3 dropped it first by design and
nothing since has raised its prior. These two cells are read against round 3's recorded reference
(`G0 x S1_monthly`, log loss 1.515428) and round 4's own noise floor.

### 8.3 Folds, populations, metrics

Folds unchanged: F1 trains 2022 and 2023 and tests 2024; **F2 trains 2022, 2023 and 2024 and tests
2025 and is the SELECTION fold**. 2025-26 stays sealed. Populations `first` and `cont` fitted and
scored separately; `first` is the selection population for the tree arm, `cont` the selection
population for `cascade`.

Primary metric, as in round 3: **multiclass log loss on fold 2**, through the imported `R1.score`.
Reported for every cell, unchanged: per-class Brier, the worst gated decile calibration gap (classes
with share at least 5%, gate 2.0 pp) split into level and shape, the step-monotonicity
responsiveness reading, by-state calibration, and round 3's segment gaps `conf4_gap_pp`,
`nonconf_gap_pp`, `first4_season_gap_pp`.

Round 4's DECISION cells, both pre-registered here:

1. **Weeks 0-3 of the season** -- `first4_season_gap_pp`, the worst gated decile calibration gap
   over chances in the first four calendar weeks of the test season. This already exists in round
   3's grader and is the segment this round targets.
2. **Non-conference** -- `nonconf_gap_pp`, unchanged from round 3. It stays a decision cell because
   round 3 failed it under every arm (2.39-3.00 pp against a 2.0 pp gate) and because the two
   segments overlap without nesting.

`conf4_gap_pp` remains reported EVIDENCE, not a round-4 decision cell: round 3 measured it at
1.16-1.45 pp on the tree, inside the gate, so there is nothing there to fix.

Round 4 ADDS one evidence cell to the grader, and adds it as a new function rather than by editing
anything round 3 scored through: a **per-week-of-season calibration and residual table** (weeks 0,
1, 2, 3, 4-7, 8+ measured from the first game of the test season) giving n, the worst gated decile
gap, and the signed mean residual per class. Every arm is graded through it, **including the
reference**, which is re-scored from round 3's stored `ref_pred_first_F2_seed0.npy` and must
reproduce round 3's recorded log loss 1.515428 to 1e-6 before that re-score is read (8.5).

**Responsiveness** is read under Decision 8 exactly as round 3 read it, on round 2's own drivers and
on `ncss`, and round 4 reports the **team-quintile slope ratio** for each arm as a headline column:
predictions bucketed by the offence team's own prior quintile must slope with actuals. A driver
whose realised quintile span is under 2 pp is exempt from both clauses and recorded as exempt.

### 8.4 Noise floor

The reference cell `G0 x S1_monthly` is refit on fold 2 under a **second seed**, spec-identical
including the whole refit calendar, for the tree population; `cascade` is deterministic and its seed
spread is 0 by construction, which is reported as such and never presented as a small floor. The
applied floor is the larger of that seed spread and a 200-replicate GAME-BLOCK bootstrap SE of the
reference cell's fold-2 log loss. A SEGMENT gap improvement must exceed **0.25 pp** to count, the
same threshold round 3 fixed, and the second-seed cell reports its own weeks-0-3 and non-conference
gaps so that threshold is checked against a measured spread rather than asserted. Two seeds against
the five round 1 pre-registered is PARTIAL and is labelled PARTIAL wherever it is used.

### 8.5 Decision rule

**The simplest arm stands** (`G0` < `G1` = `G4` < `G2` < `G3`, and `S1_monthly` < `S1_conf_aligned` <
`S1_weekly`) **unless a more complex arm, while passing round 1's calibration and responsiveness
gates, beats it beyond the noise floor on fold-2 log loss OR by more than 0.25 pp on the weeks-0-3
gap OR by more than 0.25 pp on the non-conference gap.** Where several arms beat the reference, the
simplest whose log loss is within the floor of the best beater's wins. `G1` beats `G4` on an exact
tie.

An adopted arm must ALSO not regress the other pre-registered segment by more than the same 0.25 pp:
an arm that buys 0.4 pp in weeks 0-3 and gives back 0.4 pp on non-conference has moved error around
and is not adopted. That clause is fixed here because this round's whole premise is that a segment
is where the damage is.

**If nothing beats the reference, the reference stands, and that is a RESULT and is reported as
one.** For the alignment cells specifically: this round REPORTS whether `A1` or `A2` changes the
Decision 9 reading for possession-outcome. It does not amend Decision 9. The PM does that.

The grader is round 3's, unchanged, and every arm is scored blind through it in one pass; the added
per-week cell is applied to every arm including the re-scored reference.

### 8.6 Leak test

`off_n_prior` and `def_n_prior` (the only new RAW columns; the shrunk style columns are functions of
columns round 3 already cleared) go through the standing INV-45 change-form leak test
(`cbb_sim.analysis.leak_test`, absolute as-joined change-form correlation at most 0.15) BEFORE they
are read as a result, alongside round 2's raw-centred columns and the `G1`-`G3` outputs so the
shrunk numbers are read against columns already accepted. The numbers go in the results section.

### 8.7 Staging, budget, and the drop order

Same discipline as round 3: the stage order IS the drop order, the budget is checked before each
tree cell, a cell that starts finishes, and **anything the clock does not reach is written to the
results as NOT RUN and never as a result.** This round runs under a hard external deadline of
**12:15 ET on 2026-09-11** with a six-thread total cap shared across two concurrent processes
(three threads each), on a machine shared with other workers.

| stage | cells | arm | est. cost |
|---|---|---|---|
| 0 | reference re-score from round 3's stored predictions, plus the grader-additivity check | -- | seconds |
| 1 | `cascade` cross: `G0`-`G4` x `S1_monthly` on `cont` folds 1 and 2 (the `cont` selection grid) and on `first` fold 2 as an interaction probe, plus `G0` x {`S1_conf_aligned`, `S1_weekly`} on both populations | cascade | ~15 min |
| 2 | tree `G1` x `S1_monthly`, fold 2 | lgbm | ~15 min |
| 3 | tree `G4` x `S1_monthly`, fold 2 | lgbm | ~15 min |
| 4 | tree `G2` x `S1_monthly`, fold 2 | lgbm | ~15 min |
| 5 | tree `G3` x `S1_monthly`, fold 2 | lgbm | ~15 min |
| 6 | tree noise floor: `G0` x `S1_monthly`, fold 2, seed 1 | lgbm | ~15 min |
| 7 | tree `A1` = `G0` x `S1_conf_aligned`, fold 2 (Decision 9c) | lgbm | ~75 min |
| 8 | tree `A2` = `G0` x `S1_weekly`, fold 2 (Decision 9c) | lgbm | ~60 min |
| 9 | tree fold-1 confirmation of whichever arm stages 2-5 select | lgbm | ~15 min |

Stages 7 and 8 run in a SECOND process concurrently with stages 1-6 in the first, each at three
threads, so the alignment cells are attempted against the clock rather than queued behind the
shrinkage ladder that is this lane's headline. Both processes write separate checkpoints and a
third pass merges them, applies the floor and the decision rule, and writes the results section.
On the measured round-3 cost (150 s per tree refit on a quiet machine at four threads) stage 7 is
expected to be marginal and stages 8 and 9 unlikely; that is recorded in advance.

### 8.8 Artifacts

Trainer: `scripts/train_possession_outcome_v4.py` (v1, v2 and v3 untouched; v3's grading functions
imported, not edited). Artifacts: `data/processed/models/possession_outcome/round4/` -- over 20 MB,
so gitignored and synced to the HF dataset under the `engine_inputs` bulk key per the CLAUDE.md
data rule. Round 3's `design_v3.parquet` and `boxes_first_chance.parquet` are READ, never rewritten
(worker discipline: a versioned sibling `design_v4.parquet` holds the round-4 columns). Test doc:
`docs/tests/possession_outcome_early_season_2026-09-11.md`.

<!-- ROUND 4 RESULTS APPENDED BELOW BY scripts/train_possession_outcome_v4.py -->
