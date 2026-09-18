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

---

## 9. Round 4 full results (run 2026-09-11 10:06-12:15, `scripts/train_possession_outcome_v4.py`)

Wall clock 2.1 h against the pre-registered external stop of 12:15 ET, at a six-thread total cap
split across two concurrent processes (three threads each) on a machine four other workers were
using. Cells run: 22. Cells the budget did not reach: 6 -- listed in 9.7 as NOT RUN, never as a
result.

**Headline.** On the SELECTION population `first`, the first shrinkage arm, `G1` -- every as-of
style rate multiplied by its own empirical-Bayes reliability `w = D/(D+k)` -- **cuts the
weeks-0-3 calibration gap from 3.832 pp to 2.766 pp**, a 1.066 pp gain against a pre-registered
0.25 pp threshold, while giving nothing back on the other decision segment (non-conference 2.492
-> 2.351 pp), passing both gates, improving the Decision 8 quintile slope (0.947 -> 0.974) and
costing 0.000091 of log loss against a 0.000804 floor. That is an adoption under the
pre-registered rule, and it is the first thing in four rounds to move the segment round 3
identified. On `cont`, where the ladder is COMPLETE, the winner is `G2` (shrink toward the team's
prior-season rate), with `G3` close behind; `G1` is disqualified there by the pre-registered
no-shuffling clause (it buys 0.035 pp in weeks 0-3 and gives back 0.314 pp on non-conference).

**The tree ladder is INCOMPLETE and the round is therefore a PARTIAL adoption.** `G4`, `G2` and
`G3` on `first` did not finish inside the wall clock, so `G1` won a ladder of two. The engine
default must not move on this evidence alone: `cont`'s complete ladder prefers `G2` to `G1`, and
`G2`/`G3` have never been measured on the tree. 9.7 gives the measured cost of the missing cells.

### 9.0 Reproduction checks passed before any round-4 number was read

| check | expected | measured |
|---|---|---|
| reference `first` F2 log loss, re-scored from round 3's stored predictions through the round-4 grader | 1.515428 | 1.515428 (diff 0.00e+00) |
| `cont` F2 `G0 x S1_monthly` refit in round 4 | 1.499760 (rounds 2 and 3) | 1.499760 |
| `cont` F2 `G0 x S1_conf_aligned` | 1.499720 (round 3) | 1.499720 |
| `cont` F2 `G0 x S1_weekly` | 1.499639 (round 3) | 1.499639 |
| `first` F2 cascade probe `G0 x S1_monthly` / `conf_aligned` / `weekly` | 1.530507 / 1.530448 / 1.530323 (round 3) | 1.530508 / 1.530448 / 1.530323 |
| round-4 rebuild of round 2's raw centred style columns vs the cached round-3 design | < 1e-3 | **0.0 on every one of the eight columns** -- an exact rebuild |

**One number does not reproduce, and it matters.** The `cont` reference cell's NON-CONFERENCE gap
reads 2.586 pp in round 4 against 2.609 pp in round 3, on the **identical** segment (n = 42,282 in
both) and with log loss matching to 1e-6 and the conference (2.156), first-four-conference-weeks
(2.787) and weeks-0-3 (2.849) gaps matching exactly. The difference is decile-boundary
sensitivity: a perturbation far below the sixth decimal of the fitted probabilities moves which
decile carries the maximum. It is 0.023 pp, and it is enough to move round 3's `cont` alignment
adoption from 0.260 pp -- just over the 0.25 pp threshold -- to 0.237 pp, just under it. Round 3
wrote that this result was "a lead, not a finding" because `cascade` is deterministic and its
second seed could not corroborate the segment gain; round 4 supplies the corroboration it was
missing, and the answer is that the gain sits inside the grader's own discretisation noise. See
9.3 and 9.8.

### 9.1 The grid


**`first` / F2** (SELECTION)

| stage | role | population | fold | arm | feature_arm | scheme | seed | n_fits | log_loss | worst_gated_gap_pp | wk03_gap_pp | nonconf_gap_pp | conf4_gap_pp | quintile_slope_worst | calibration_pass | responsiveness_pass | ncss_slope_pass | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | interaction probe (not selection) | first | F2 | cascade | G0 | S1_conf_aligned | 0 | 29 | 1.530448 | 2.076 | 3.241 | 2.148 | 2.165 | 0.8692 | False | True | True | 204.5 |
| 1 | interaction probe (not selection) | first | F2 | cascade | G0 | S1_monthly | 0 | 6 | 1.530508 | 2.088 | 3.429 | 2.178 | 2.233 | 0.8693 | False | True | True | 35.7 |
| 1 | interaction probe (not selection) | first | F2 | cascade | G1 | S1_monthly | 0 | 6 | 1.529618 | 2.0 | 2.974 | 2.198 | 1.79 | 0.9497 | True | True | True | 43.9 |
| 1 | interaction probe (not selection) | first | F2 | cascade | G2 | S1_monthly | 0 | 6 | 1.529241 | 1.89 | 2.411 | 1.843 | 1.765 | 0.8805 | True | True | True | 44.9 |
| 1 | interaction probe (not selection) | first | F2 | cascade | G3 | S1_monthly | 0 | 6 | 1.529028 | 1.874 | 2.518 | 2.021 | 1.675 | 0.9334 | True | True | True | 48.3 |
| 1 | interaction probe (not selection) | first | F2 | cascade | G4 | S1_monthly | 0 | 6 | 1.5305 | 2.086 | 3.491 | 2.238 | 2.211 | 0.8668 | False | True | True | 51.2 |
| 1 | interaction probe (not selection) | first | F2 | cascade | G0 | S1_weekly | 0 | 23 | 1.530323 | 2.093 | 3.071 | 1.919 | 2.201 | 0.8672 | False | True | True | 198.7 |
| 0 | reference (re-scored from round 3) | first | F2 | lgbm | G0 | S1_monthly | 0 | 6 | 1.515428 | 0.98 | 3.832 | 2.492 | 1.166 | 0.9473 | True | True | True | 0.0 |
| 2 | selection | first | F2 | lgbm | G1 | S1_monthly | 0 | 6 | 1.515519 | 1.456 | 2.766 | 2.351 | 1.166 | 0.9745 | True | True | True | 4883.1 |

**`cont` / F2** (SELECTION)

| stage | role | population | fold | arm | feature_arm | scheme | seed | n_fits | log_loss | worst_gated_gap_pp | wk03_gap_pp | nonconf_gap_pp | conf4_gap_pp | quintile_slope_worst | calibration_pass | responsiveness_pass | ncss_slope_pass | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | selection | cont | F2 | cascade | G0 | S1_conf_aligned | 0 | 29 | 1.49972 | 1.886 | 2.961 | 2.349 | 2.658 | 0.7717 | True | True | True | 27.0 |
| 1 | selection | cont | F2 | cascade | G0 | S1_monthly | 0 | 6 | 1.49976 | 1.859 | 2.849 | 2.586 | 2.787 | 0.7711 | True | True | True | 5.3 |
| 1 | selection | cont | F2 | cascade | G1 | S1_monthly | 0 | 6 | 1.498952 | 1.819 | 2.884 | 2.9 | 2.849 | 0.901 | True | True | True | 5.8 |
| 1 | selection | cont | F2 | cascade | G2 | S1_monthly | 0 | 6 | 1.498802 | 1.692 | 2.76 | 2.249 | 2.737 | 0.7907 | True | True | True | 6.0 |
| 1 | selection | cont | F2 | cascade | G3 | S1_monthly | 0 | 6 | 1.498677 | 1.754 | 2.422 | 2.175 | 2.97 | 0.8425 | True | True | True | 6.8 |
| 1 | selection | cont | F2 | cascade | G4 | S1_monthly | 0 | 6 | 1.499739 | 2.01 | 2.727 | 2.542 | 2.513 | 0.7705 | False | True | True | 8.3 |
| 1 | selection | cont | F2 | cascade | G0 | S1_weekly | 0 | 23 | 1.499639 | 1.974 | 2.866 | 2.691 | 2.87 | 0.7702 | True | True | True | 29.7 |

**`cont` / F1**

| stage | role | population | fold | arm | feature_arm | scheme | seed | n_fits | log_loss | worst_gated_gap_pp | wk03_gap_pp | nonconf_gap_pp | conf4_gap_pp | quintile_slope_worst | calibration_pass | responsiveness_pass | ncss_slope_pass | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | selection | cont | F1 | cascade | G0 | S1_monthly | 0 | 6 | 1.499063 | 1.188 | 3.213 | 2.251 |  | 0.7017 | True | True | True | 3.8 |
| 1 | selection | cont | F1 | cascade | G1 | S1_monthly | 0 | 6 | 1.498338 | 1.549 | 2.888 | 2.486 |  | 0.8216 | True | True | True | 3.5 |
| 1 | selection | cont | F1 | cascade | G2 | S1_monthly | 0 | 6 | 1.497797 | 1.585 | 4.547 | 2.469 |  | 0.7731 | True | True | True | 3.9 |
| 1 | selection | cont | F1 | cascade | G3 | S1_monthly | 0 | 6 | 1.497673 | 1.688 | 3.021 | 1.874 |  | 0.8161 | True | True | True | 4.5 |
| 1 | selection | cont | F1 | cascade | G4 | S1_monthly | 0 | 6 | 1.499069 | 1.2 | 2.972 | 2.164 |  | 0.7036 | True | True | True | 5.1 |

`wk03_gap_pp` is the worst gated decile calibration gap over chances in the first four calendar weeks of the test season -- the segment round 4 targets. `quintile_slope_worst` is the own-driver Decision 8 slope ratio furthest from 1.0 among the drivers whose realised quintile span clears 2 pp.


### 9.2 Noise floor

* **`first`**: reference cell `G0 x S1_monthly`, seed 0 log loss 1.515428, seed 1 None, spread None; weeks-0-3 gap 3.832 vs None pp; non-conference gap 2.492 vs None pp; 200-replicate game-block bootstrap SE 0.000804. Applied floor **0.000804**. PARTIAL: SECOND SEED NOT RUN -- the floor falls back to the block bootstrap SE alone and is labelled PARTIAL.
* **`cont`**: reference cell `G0 x S1_monthly`, seed 0 log loss 1.49976, seed 1 None, spread None; weeks-0-3 gap 2.849 vs None pp; non-conference gap 2.586 vs None pp; 200-replicate game-block bootstrap SE 0.001982. Applied floor **0.001982**. PARTIAL: SECOND SEED NOT RUN -- the floor falls back to the block bootstrap SE alone and is labelled PARTIAL.

### 9.3 Decision


**`first`** (fold F2, noise floor 0.000804)


*feature ladder*

| feature_arm | seed | log_loss | gain_vs_reference | wk03_gap_pp | wk03_gain_pp | nonconf_gap_pp | nonconf_gain_pp | quintile_slope_worst | gates_pass | moves_error_between_segments | beats_reference_beyond_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | 0 | 1.515428 | 0.0 | 3.832 | 0.0 | 2.492 | 0.0 | 0.9473 | True | False | False |
| G1 | 0 | 1.515519 | -9.1e-05 | 2.766 | 1.066 | 2.351 | 0.141 | 0.9745 | True | False | True |

Winner: `G1` -- G1 beats the reference beyond the floor and is the simplest arm within the floor of the best beater (1.515519).


*scheme ladder*

| scheme | seed | log_loss | gain_vs_reference | wk03_gap_pp | wk03_gain_pp | nonconf_gap_pp | nonconf_gain_pp | quintile_slope_worst | gates_pass | moves_error_between_segments | beats_reference_beyond_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_monthly | 0 | 1.515428 | 0.0 | 3.832 | 0.0 | 2.492 | 0.0 | 0.9473 | True | False | False |

Winner: `S1_monthly` -- the simplest arm stands -- no more complex arm beat it by more than the noise floor 0.00080 on log loss or by more than 0.25 pp on the weeks-0-3 or non-conference gap without giving the other segment back. Under the pre-registration that is a RESULT, not a failure.


**`cont`** (fold F2, noise floor 0.001982)


*feature ladder*

| feature_arm | seed | log_loss | gain_vs_reference | wk03_gap_pp | wk03_gain_pp | nonconf_gap_pp | nonconf_gain_pp | quintile_slope_worst | gates_pass | moves_error_between_segments | beats_reference_beyond_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | 0 | 1.49976 | 0.0 | 2.849 | 0.0 | 2.586 | 0.0 | 0.7711 | True | False | False |
| G1 | 0 | 1.498952 | 0.000808 | 2.884 | -0.035 | 2.9 | -0.314 | 0.901 | True | True | False |
| G2 | 0 | 1.498802 | 0.000958 | 2.76 | 0.089 | 2.249 | 0.337 | 0.7907 | True | False | True |
| G3 | 0 | 1.498677 | 0.001083 | 2.422 | 0.427 | 2.175 | 0.411 | 0.8425 | True | False | True |
| G4 | 0 | 1.499739 | 2.1e-05 | 2.727 | 0.122 | 2.542 | 0.044 | 0.7705 | False | False | False |

Winner: `G2` -- G2 beats the reference beyond the floor and is the simplest arm within the floor of the best beater (1.498677).


*scheme ladder*

| scheme | seed | log_loss | gain_vs_reference | wk03_gap_pp | wk03_gain_pp | nonconf_gap_pp | nonconf_gain_pp | quintile_slope_worst | gates_pass | moves_error_between_segments | beats_reference_beyond_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_conf_aligned | 0 | 1.49972 | 4e-05 | 2.961 | -0.112 | 2.349 | 0.237 | 0.7717 | True | False | False |
| S1_monthly | 0 | 1.49976 | 0.0 | 2.849 | 0.0 | 2.586 | 0.0 | 0.7711 | True | False | False |
| S1_weekly | 0 | 1.499639 | 0.000121 | 2.866 | -0.017 | 2.691 | -0.105 | 0.7702 | True | False | False |

Winner: `S1_monthly` -- the simplest arm stands -- no more complex arm beat it by more than the noise floor 0.00198 on log loss or by more than 0.25 pp on the weeks-0-3 or non-conference gap without giving the other segment back. Under the pre-registration that is a RESULT, not a failure.


### 9.4 Responsiveness by own-rating quintile (Decision 8)

| cell | driver | class | span_pred_pp | span_act_pp | slope_ratio | steps | exempt_narrow_span | pass |
|---|---|---|---|---|---|---|---|---|
| first/F2/lgbm/G0/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.867 | 9.36 | 0.9473 | 4/4 | False | True |
| first/F2/lgbm/G0/S1_monthly/s0 | off_rim_c | FGA_rim | 5.972 | 6.083 | 0.9819 | 4/4 | False | True |
| first/F2/lgbm/G0/S1_monthly/s0 | off_tov_c | TOV | 2.773 | 2.756 | 1.0062 | 4/4 | False | True |
| cont/F1/cascade/G0/S1_monthly/s0 | off_3pa_c | FGA_3 | 7.818 | 8.352 | 0.9361 | 4/4 | False | True |
| cont/F1/cascade/G0/S1_monthly/s0 | off_rim_c | FGA_rim | 4.268 | 6.082 | 0.7017 | 4/4 | False | False |
| cont/F1/cascade/G0/S1_monthly/s0 | off_tov_c | TOV | 2.197 | 2.402 | 0.9143 | 4/4 | False | True |
| cont/F2/cascade/G0/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.325 | 9.695 | 0.8587 | 4/4 | False | True |
| cont/F2/cascade/G0/S1_monthly/s0 | off_rim_c | FGA_rim | 4.304 | 5.582 | 0.7711 | 4/4 | False | False |
| cont/F2/cascade/G0/S1_monthly/s0 | off_tov_c | TOV | 1.938 | 2.105 | 0.921 | 3/4 | False | True |
| first/F2/cascade/G0/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.604 | 9.36 | 0.9192 | 4/4 | False | True |
| first/F2/cascade/G0/S1_monthly/s0 | off_rim_c | FGA_rim | 5.288 | 6.083 | 0.8693 | 4/4 | False | True |
| first/F2/cascade/G0/S1_monthly/s0 | off_tov_c | TOV | 2.451 | 2.756 | 0.8893 | 4/4 | False | True |
| cont/F1/cascade/G1/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.407 | 8.352 | 1.0066 | 4/4 | False | True |
| cont/F1/cascade/G1/S1_monthly/s0 | off_rim_c | FGA_rim | 4.997 | 6.082 | 0.8216 | 4/4 | False | True |
| cont/F1/cascade/G1/S1_monthly/s0 | off_tov_c | TOV | 2.164 | 2.402 | 0.9007 | 4/4 | False | True |
| cont/F2/cascade/G1/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.928 | 9.695 | 0.9209 | 4/4 | False | True |
| cont/F2/cascade/G1/S1_monthly/s0 | off_rim_c | FGA_rim | 5.029 | 5.582 | 0.901 | 4/4 | False | True |
| cont/F2/cascade/G1/S1_monthly/s0 | off_tov_c | TOV | 1.94 | 2.105 | 0.9217 | 3/4 | False | True |
| first/F2/cascade/G1/S1_monthly/s0 | off_3pa_c | FGA_3 | 9.143 | 9.36 | 0.9768 | 4/4 | False | True |
| first/F2/cascade/G1/S1_monthly/s0 | off_rim_c | FGA_rim | 5.875 | 6.083 | 0.9658 | 4/4 | False | True |
| first/F2/cascade/G1/S1_monthly/s0 | off_tov_c | TOV | 2.618 | 2.756 | 0.9497 | 4/4 | False | True |
| cont/F1/cascade/G2/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.501 | 8.352 | 1.0179 | 4/4 | False | True |
| cont/F1/cascade/G2/S1_monthly/s0 | off_rim_c | FGA_rim | 4.901 | 6.082 | 0.8058 | 4/4 | False | True |
| cont/F1/cascade/G2/S1_monthly/s0 | off_tov_c | TOV | 1.857 | 2.402 | 0.7731 | 4/4 | False | False |
| cont/F2/cascade/G2/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.741 | 9.695 | 0.9016 | 4/4 | False | True |
| cont/F2/cascade/G2/S1_monthly/s0 | off_rim_c | FGA_rim | 5.006 | 5.582 | 0.8968 | 4/4 | False | True |
| cont/F2/cascade/G2/S1_monthly/s0 | off_tov_c | TOV | 1.664 | 2.105 | 0.7907 | 3/4 | False | False |
| first/F2/cascade/G2/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.989 | 9.36 | 0.9603 | 4/4 | False | True |
| first/F2/cascade/G2/S1_monthly/s0 | off_rim_c | FGA_rim | 5.849 | 6.083 | 0.9616 | 4/4 | False | True |
| first/F2/cascade/G2/S1_monthly/s0 | off_tov_c | TOV | 2.427 | 2.756 | 0.8805 | 4/4 | False | True |
| cont/F1/cascade/G3/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.586 | 8.352 | 1.028 | 4/4 | False | True |
| cont/F1/cascade/G3/S1_monthly/s0 | off_rim_c | FGA_rim | 5.059 | 6.082 | 0.8318 | 4/4 | False | True |
| cont/F1/cascade/G3/S1_monthly/s0 | off_tov_c | TOV | 1.961 | 2.402 | 0.8161 | 4/4 | False | True |
| cont/F2/cascade/G3/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.855 | 9.695 | 0.9133 | 4/4 | False | True |
| cont/F2/cascade/G3/S1_monthly/s0 | off_rim_c | FGA_rim | 5.164 | 5.582 | 0.9251 | 4/4 | False | True |
| cont/F2/cascade/G3/S1_monthly/s0 | off_tov_c | TOV | 1.773 | 2.105 | 0.8425 | 3/4 | False | True |
| first/F2/cascade/G3/S1_monthly/s0 | off_3pa_c | FGA_3 | 9.086 | 9.36 | 0.9707 | 4/4 | False | True |
| first/F2/cascade/G3/S1_monthly/s0 | off_rim_c | FGA_rim | 6.019 | 6.083 | 0.9896 | 4/4 | False | True |
| first/F2/cascade/G3/S1_monthly/s0 | off_tov_c | TOV | 2.573 | 2.756 | 0.9334 | 4/4 | False | True |
| cont/F1/cascade/G4/S1_monthly/s0 | off_3pa_c | FGA_3 | 7.8 | 8.352 | 0.9339 | 4/4 | False | True |
| cont/F1/cascade/G4/S1_monthly/s0 | off_rim_c | FGA_rim | 4.28 | 6.082 | 0.7036 | 4/4 | False | False |
| cont/F1/cascade/G4/S1_monthly/s0 | off_tov_c | TOV | 2.205 | 2.402 | 0.9176 | 4/4 | False | True |
| cont/F2/cascade/G4/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.322 | 9.695 | 0.8584 | 4/4 | False | True |
| cont/F2/cascade/G4/S1_monthly/s0 | off_rim_c | FGA_rim | 4.301 | 5.582 | 0.7705 | 4/4 | False | False |
| cont/F2/cascade/G4/S1_monthly/s0 | off_tov_c | TOV | 1.924 | 2.105 | 0.9141 | 3/4 | False | True |
| first/F2/cascade/G4/S1_monthly/s0 | off_3pa_c | FGA_3 | 8.602 | 9.36 | 0.919 | 4/4 | False | True |
| first/F2/cascade/G4/S1_monthly/s0 | off_rim_c | FGA_rim | 5.273 | 6.083 | 0.8668 | 4/4 | False | True |
| first/F2/cascade/G4/S1_monthly/s0 | off_tov_c | TOV | 2.427 | 2.756 | 0.8805 | 4/4 | False | True |
| cont/F2/cascade/G0/S1_conf_aligned/s0 | off_3pa_c | FGA_3 | 8.348 | 9.695 | 0.8611 | 4/4 | False | True |
| cont/F2/cascade/G0/S1_conf_aligned/s0 | off_rim_c | FGA_rim | 4.307 | 5.582 | 0.7717 | 4/4 | False | False |
| cont/F2/cascade/G0/S1_conf_aligned/s0 | off_tov_c | TOV | 1.936 | 2.105 | 0.9199 | 3/4 | False | True |
| first/F2/cascade/G0/S1_conf_aligned/s0 | off_3pa_c | FGA_3 | 8.601 | 9.36 | 0.9188 | 4/4 | False | True |
| first/F2/cascade/G0/S1_conf_aligned/s0 | off_rim_c | FGA_rim | 5.287 | 6.083 | 0.8692 | 4/4 | False | True |
| first/F2/cascade/G0/S1_conf_aligned/s0 | off_tov_c | TOV | 2.451 | 2.756 | 0.8891 | 4/4 | False | True |
| cont/F2/cascade/G0/S1_weekly/s0 | off_3pa_c | FGA_3 | 8.342 | 9.695 | 0.8604 | 4/4 | False | True |
| cont/F2/cascade/G0/S1_weekly/s0 | off_rim_c | FGA_rim | 4.299 | 5.582 | 0.7702 | 4/4 | False | False |
| cont/F2/cascade/G0/S1_weekly/s0 | off_tov_c | TOV | 1.919 | 2.105 | 0.9118 | 3/4 | False | True |
| first/F2/cascade/G0/S1_weekly/s0 | off_3pa_c | FGA_3 | 8.596 | 9.36 | 0.9184 | 4/4 | False | True |
| first/F2/cascade/G0/S1_weekly/s0 | off_rim_c | FGA_rim | 5.275 | 6.083 | 0.8672 | 4/4 | False | True |
| first/F2/cascade/G0/S1_weekly/s0 | off_tov_c | TOV | 2.448 | 2.756 | 0.8882 | 4/4 | False | True |
| first/F2/lgbm/G1/S1_monthly/s0 | off_3pa_c | FGA_3 | 9.122 | 9.36 | 0.9745 | 4/4 | False | True |
| first/F2/lgbm/G1/S1_monthly/s0 | off_rim_c | FGA_rim | 6.205 | 6.083 | 1.0201 | 4/4 | False | True |
| first/F2/lgbm/G1/S1_monthly/s0 | off_tov_c | TOV | 2.806 | 2.756 | 1.0182 | 4/4 | False | True |

### 9.5 Leak test on every column round 4 adds

| column | n | n_update | n_level | corr_asjoined | corr_update | corr_level | static | n_nonzero_deltas | verdict | level_verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| off_3pa_c | 37347 | 37347 | 38794 | 0.0061628264203572 | 0.0269764877765137 | 0.0600946219636213 | False | 37347 | pass | pass (level-form) |
| off_rim_c | 37347 | 37347 | 38794 | 0.0169587716157511 | 0.0880541406581658 | 0.0739535468275101 | False | 37347 | pass | pass (level-form) |
| off_tov_c | 37347 | 37347 | 38794 | -0.0214933503591643 | -0.1269878664220332 | -0.1549727634400307 | False | 37347 | pass | LEAK (level-form) |
| off_ftr_c | 37347 | 37347 | 38794 | 0.0078106256894635 | 0.0888630993438912 | 0.0377702211154443 | False | 37347 | pass | pass (level-form) |
| off_3pa_g1 | 37347 | 37347 | 38794 | 0.0165674982806949 | 0.0376815702405395 | 0.0572822762416461 | False | 37347 | pass | pass (level-form) |
| off_3pa_g2 | 37347 | 37347 | 38794 | 0.0073662578445344 | 0.0269338499826288 | 0.0623648573465147 | False | 37347 | pass | pass (level-form) |
| off_3pa_g3 | 37347 | 37347 | 38794 | 0.0085125426168135 | 0.0283835277323774 | 0.0620163433890055 | False | 37347 | pass | pass (level-form) |
| off_rim_g1 | 37347 | 37347 | 38794 | 0.0305078241373466 | 0.1352039635350138 | 0.0644765142678936 | False | 37347 | pass | pass (level-form) |
| off_rim_g2 | 37347 | 37347 | 38794 | 0.0220869355098952 | 0.1246052347381212 | 0.0709213089016107 | False | 37347 | pass | pass (level-form) |
| off_rim_g3 | 37347 | 37347 | 38794 | 0.0237929805307076 | 0.1272563690637964 | 0.0706309490745993 | False | 37347 | pass | pass (level-form) |
| off_tov_g1 | 37347 | 37347 | 38794 | -0.0476293505552661 | -0.2083602754028026 | -0.1497966639598137 | False | 37347 | pass | pass (level-form) |
| off_tov_g2 | 37347 | 37347 | 38794 | -0.0217419781123207 | -0.1773111483499024 | -0.181281973616032 | False | 37347 | pass | LEAK (level-form) |
| off_tov_g3 | 37347 | 37347 | 38794 | -0.0287558967510083 | -0.186535366618264 | -0.1788550204224482 | False | 37347 | pass | LEAK (level-form) |
| off_ftr_g1 | 37347 | 37347 | 38794 | 0.0099646531848854 | 0.1359011594735028 | 0.0267759905518781 | False | 37347 | pass | pass (level-form) |
| off_ftr_g2 | 37347 | 37347 | 38794 | 0.0062051585070542 | 0.1307588629797164 | 0.0308775215964918 | False | 37347 | pass | pass (level-form) |
| off_ftr_g3 | 37347 | 37347 | 38794 | 0.0074104081531508 | 0.1331035455129471 | 0.0322219355549812 | False | 37347 | pass | pass (level-form) |
| off_n_prior_g | 37347 | 37347 | 38794 |  |  | 0.0162321982163389 | False | 37347 |  | pass (level-form) |
| def_n_prior_g | 37347 | 37347 | 38794 | -0.0611824636461942 | 0.0483457865066383 | -0.0162321982163389 | False | 31363 | pass | pass (level-form) |

Gate: |as-joined change-form corr| <= 0.15. Round 2's raw-centred columns are shown alongside so the shrunk numbers are read against columns already accepted.


### 9.6 The fitted shrinkage weight

| season | side | rate | k (denominator mass) | s2 | tau2_raw | tau2_net | fitted_from_seasons | note |
|---|---|---|---|---|---|---|---|---|
| 2022 | off | 3pa | 206.896 | 3174.22476 | 17.12519 | 15.34212 | [2022, 2023, 2024] | pooled (no prior season in panel; 2022 is train-only in both folds) |
| 2022 | off | rim | 270.792 | 4354.83627 | 19.44994 | 16.08184 | [2022, 2023, 2024] | pooled (no prior season in panel; 2022 is train-only in both folds) |
| 2022 | off | tov | 616.251 | 1630.01772 | 3.56069 | 2.64505 | [2022, 2023, 2024] | pooled (no prior season in panel; 2022 is train-only in both folds) |
| 2022 | off | ftr | 753.272 | 10131.73428 | 21.28634 | 13.45029 | [2022, 2023, 2024] | pooled (no prior season in panel; 2022 is train-only in both folds) |
| 2022 | def | 3pa | 431.986 | 3650.39947 | 10.50082 | 8.45027 | [2022, 2023, 2024] | pooled (no prior season in panel; 2022 is train-only in both folds) |
| 2022 | def | rim | 362.478 | 4534.14252 | 16.01551 | 12.50873 | [2022, 2023, 2024] | pooled (no prior season in panel; 2022 is train-only in both folds) |
| 2022 | def | tov | 416.417 | 1554.79711 | 4.60713 | 3.73375 | [2022, 2023, 2024] | pooled (no prior season in panel; 2022 is train-only in both folds) |
| 2022 | def | ftr | 434.136 | 9686.97419 | 29.80527 | 22.31321 | [2022, 2023, 2024] | pooled (no prior season in panel; 2022 is train-only in both folds) |
| 2023 | off | 3pa | 207.694 | 3178.22616 | 17.23756 | 15.30242 | [2022] |  |
| 2023 | off | rim | 284.454 | 4326.84464 | 18.85092 | 15.21107 | [2022] |  |
| 2023 | off | tov | 601.431 | 1689.22054 | 3.83719 | 2.80867 | [2022] |  |
| 2023 | off | ftr | 819.949 | 9705.80438 | 20.00185 | 11.83709 | [2022] |  |
| 2023 | def | 3pa | 425.967 | 3644.47476 | 10.77478 | 8.55576 | [2022] |  |
| 2023 | def | rim | 358.08 | 4464.1134 | 16.22212 | 12.46679 | [2022] |  |
| 2023 | def | tov | 396.085 | 1603.54006 | 5.02483 | 4.04847 | [2022] |  |
| 2023 | def | ftr | 437.025 | 9240.29175 | 28.91678 | 21.14362 | [2022] |  |
| 2024 | off | 3pa | 214.997 | 3204.08165 | 16.81191 | 14.90291 | [2022, 2023] |  |
| 2024 | off | rim | 294.163 | 4326.04405 | 18.27013 | 14.70627 | [2022, 2023] |  |
| 2024 | off | tov | 638.33 | 1675.41361 | 3.62289 | 2.62468 | [2022, 2023] |  |
| 2024 | off | ftr | 757.258 | 9884.31322 | 21.19561 | 13.05278 | [2022, 2023] |  |
| 2024 | def | 3pa | 433.799 | 3651.3495 | 10.59262 | 8.41715 | [2022, 2023] |  |
| 2024 | def | rim | 387.945 | 4483.15469 | 15.24944 | 11.55615 | [2022, 2023] |  |
| 2024 | def | tov | 405.609 | 1586.65337 | 4.85711 | 3.91178 | [2022, 2023] |  |
| 2024 | def | ftr | 426.918 | 9433.27503 | 29.8675 | 22.09624 | [2022, 2023] |  |
| 2025 | off | 3pa | 206.896 | 3174.22476 | 17.12519 | 15.34212 | [2022, 2023, 2024] |  |
| 2025 | off | rim | 270.792 | 4354.83627 | 19.44994 | 16.08184 | [2022, 2023, 2024] |  |
| 2025 | off | tov | 616.251 | 1630.01772 | 3.56069 | 2.64505 | [2022, 2023, 2024] |  |
| 2025 | off | ftr | 753.272 | 10131.73428 | 21.28634 | 13.45029 | [2022, 2023, 2024] |  |
| 2025 | def | 3pa | 431.986 | 3650.39947 | 10.50082 | 8.45027 | [2022, 2023, 2024] |  |
| 2025 | def | rim | 362.478 | 4534.14252 | 16.01551 | 12.50873 | [2022, 2023, 2024] |  |
| 2025 | def | tov | 416.417 | 1554.79711 | 4.60713 | 3.73375 | [2022, 2023, 2024] |  |
| 2025 | def | ftr | 434.136 | 9686.97419 | 29.80527 | 22.31321 | [2022, 2023, 2024] |  |

Every k is fitted from COMPLETED PRIOR SEASONS ONLY by method of moments; none is chosen by looking at a score. `w = D/(D+k)` with `D` the team's as-of denominator mass, so a team with `D = k` sits at half weight.


**Reproduction check**: the round-4 rebuild of round 2's raw centred columns matches the cached round-3 design to 0.0 absolute, so the arms differ by the shrinkage and by nothing else.


### 9.7 Cells the budget did not reach (NOT RUN, not a result)

| stage | population | arm | fold | feature_arm | scheme | role | seed | reason |
|---|---|---|---|---|---|---|---|---|
| 3 | first | lgbm | F2 | G4 | S1_monthly | selection | 0 | not reached inside the pre-registered wall clock (round 4 section 8.7 drop order) |
| 4 | first | lgbm | F2 | G2 | S1_monthly | selection | 0 | not reached inside the pre-registered wall clock (round 4 section 8.7 drop order) |
| 5 | first | lgbm | F2 | G3 | S1_monthly | selection | 0 | not reached inside the pre-registered wall clock (round 4 section 8.7 drop order) |
| 6 | first | lgbm | F2 | G0 | S1_monthly | noise floor | 1 | not reached inside the pre-registered wall clock (round 4 section 8.7 drop order) |
| 7 | first | lgbm | F2 | G0 | S1_conf_aligned | selection | 0 | not reached inside the pre-registered wall clock (round 4 section 8.7 drop order) |
| 8 | first | lgbm | F2 | G0 | S1_weekly | selection | 0 | not reached inside the pre-registered wall clock (round 4 section 8.7 drop order) |
### 9.8 What this says about Decision 9 (REPORTED, not decided -- the PM amends Decision 9)

Round 4 does not settle Decision 9c and says so plainly.

1. **The two tree alignment cells round 3 owed are STILL NOT RUN.** `G0 x S1_conf_aligned` on
   `first` was started in its own process at 10:08 ET and was stopped at 10:56 after 48 minutes
   with the cell unfinished; its partial work was discarded rather than recorded. Measured cost at
   the three-thread cap this round could use: **one six-refit tree cell on the fold-2 `first` slice
   took 75 minutes**, so the 29-refit conference-aligned cell is about **6 hours** and the 23-refit
   weekly cell about **4.8 hours** at this thread budget. They are not reachable inside a
   two-hour window on a shared 20-core box under a six-thread cap; they are minutes on the
   196-core box. Nothing about TREE alignment is claimed from this run.
2. **What round 4 DOES add on alignment is negative.** Round 3 adopted `S1_conf_aligned` on `cont`
   on a single number: a 0.260 pp non-conference gain against a 0.25 pp threshold, with a log-loss
   gain of a fiftieth of the floor, and round 3 itself labelled it "a lead, not a finding" because
   `cascade` is deterministic and could not put a measured spread behind the segment. Round 4 refit
   the identical cells and the identical segment and reads the gain at **0.237 pp** -- below the
   threshold -- because the reference's own non-conference gap moved 0.023 pp on decile-boundary
   sensitivity while its log loss reproduced to 1e-6. **Under round 4's grader the `cont` alignment
   adoption does not reproduce**, and the scheme ladder's winner on `cont` is the reference
   `S1_monthly`. A finding that flips on a 0.023 pp discretisation artefact was never a finding.
3. **The `first` cascade interaction probe, complete on all three schemes, points the same way as
   round 3's did.** `S1_weekly` 1.530323 and `S1_conf_aligned` 1.530448 against `S1_monthly`
   1.530508: gains of 0.000185 and 0.000060 on a population whose tree floor is 0.000804, so both
   are inside the floor on the primary metric. On the segments `S1_weekly` does move the weeks-0-3
   gap 3.429 -> 3.071 pp (0.358) and the non-conference gap 2.178 -> 1.919 pp (0.259). That is the
   ONE place alignment has cleared a segment threshold twice; it is a probe on the wrong model
   class and it is reported as a probe.
4. **And the same segments move three times further under shrinkage on the model that actually
   serves.** `G1` on the tree takes weeks 0-3 from 3.832 to 2.766 pp (1.066) at the reference's own
   monthly calendar. If the season-start defect is what the alignment arms were reaching for, the
   feature's reliability is a much larger lever than the refit calendar, which is what L35
   predicted and what round 4 measures.

**Recommendation to the PM, for Decision 9:** leave 9c PENDING EVIDENCE for possession-outcome, and
record that the `cont` adoption round 3 made is withdrawn as inside grader noise unless a segment
floor with a measured spread is put behind it. 9a (opponent adjustment) and 9b (conference flag)
are untouched by round 4 and stand as round 3 left them. This worker does not amend Decision 9.

### 9.9 Execution note (worker, 2026-09-11)

Written by hand, because a round split across three processes has no single `run_meta` that sees
all of it.

**Three processes, three checkpoints, one merged render, no cell computed twice.** Process A
(`--stages 1,2,3,4,5,6 --ckpt ckpt_a.json`, 10:06 ET) ran the cascade cross and then the tree
shrinkage ladder. Process B (`--stages 7 --ckpt ckpt_b.json`, 10:06 ET) ran the tree alignment cell
`G0 x S1_conf_aligned` from the first minute so that Decision 9c competed against the clock in
parallel rather than queueing behind this lane's headline. At 10:56 the measured pace made B's
completion impossible before the external stop -- one six-refit tree cell was already 30 minutes in
without finishing, implying about 6 hours for B's 29 refits -- so **this worker stopped its own
process B** and started process B2 (`--stages 4,5,6 --ckpt ckpt_b2.json`, 10:56 ET) on the
shrinkage cells A would not reach. B's partial work was discarded, not recorded. No process this
worker did not start was signalled at any point. A final pass merged the three checkpoints, fitted
nothing, computed the floor, applied the decision rule and rendered every table.

**Nothing round 3 already measured was refit.** The reference cell was re-scored from round 3's
stored predictions, not refit, and reproduces 1.515428 exactly (9.0). Round 3's `design_v3.parquet`,
`boxes_first_chance.parquet` and `ref_pred_*.npy` were read and never written; the round-4 columns
went to a versioned sibling `round4/design_v4.parquet`.

**Cost, for whoever schedules the missing cells.** At the three-thread cap on a shared 20-core box
with four other workers active (machine load 25-47% throughout), one six-refit `first` tree cell
took 75 minutes -- against round 3's 15 minutes at four threads on a quiet machine and 53 minutes
at four threads under contention. The four missing tree cells (`G4`, `G2`, `G3`, and the
second-seed floor) are about 5 hours at this budget and minutes on the 196-core box; the two
alignment cells are about 11 hours. This is the second consecutive round in which the tree
alignment cells were the casualty of a shared-machine thread cap, which is itself the finding that
should decide where round 5 runs.

**The noise floor is PARTIAL in a way round 3's was not.** The second-seed reference refit (stage 6)
did not finish, so the applied `first` floor is the 200-replicate game-block bootstrap SE alone,
0.000804 -- which is the number round 3 applied as well, because there the seed spread (0.000113)
was an eighth of the SE and never binding. What round 4 does NOT have is a measured seed spread on
the weeks-0-3 SEGMENT gap. Round 3 measured that spread at 0.076 pp on `conf4` and 0.133 pp on
non-conference, both well under the 0.25 pp threshold, and `G1`'s 1.066 pp gain is eight to
fourteen times either. The adoption does not rest on the missing cell; the statement that it does
not is on the record here rather than left to be inferred.

### 9.10 Addendum, 12:03 ET: the tree `G2` cell landed after 9.0-9.9 were written and committed (`db0f8a2`)

Process B2 finished `first | F2 | lgbm | G2 | S1_monthly` at 12:02 ET, after the sections above were
written and committed (`db0f8a2`). It is a pre-registered cell run in its pre-registered spec
through the same grader, and it is reported here rather than by rewriting what stood; the earlier
statement that `G2` on the tree is NOT RUN is superseded by this section. `G4`, `G3` and the
second-seed floor cell remain NOT RUN.

### The `first` feature ladder, now three arms

| arm | log loss | gain vs ref | overall gap | **weeks 0-3 gap** | non-conf gap | worst quintile slope | gates | beats ref beyond floor |
|---|---|---|---|---|---|---|---|---|
| `G0` reference | 1.515428 | 0.0 | 0.980 | 3.832 | 2.492 | 0.9473 | PASS | -- |
| `G1` shrink to league mean | 1.515519 | -0.000091 | 1.456 | **2.766** (+1.066) | 2.351 (+0.141) | 0.9745 | PASS | **YES** |
| `G2` shrink to prior season | **1.514837** | **+0.000591** | 1.196 | **3.359** (+0.473) | 2.500 (-0.008) | 0.9156 | PASS | **YES** |
| `G4`, `G3`, floor seed 1 | NOT RUN | | | | | | | |

Both arms beat the reference on the pre-registered weeks-0-3 cell. `G2` has the best log loss in the
round, but +0.000591 is inside the 0.000804 floor, so the primary metric still separates nothing.
Under the pre-registered rule the winner is unchanged: the best beater by log loss is `G2`
(1.514837), `G1` sits within the floor of it (1.515519 against 1.515641), and among arms inside that
band the simplest wins -- `G1` at complexity rank 1 against `G2` at rank 2. **`G1` remains the
round-4 winner, now over a ladder of three.**

### Per week of season, all three arms

| week | n | `G0` | `G1` | `G2` |
|---|---|---|---|---|
| 0 | 36,499 | 6.418 | **4.867** | 4.950 |
| 1 | 40,718 | 5.216 | **4.054** | 4.166 |
| 2 | 43,658 | 3.313 | **2.942** | 2.956 |
| 3 | 43,562 | **2.751** | 2.398 | 3.247 |
| 4-7 | 111,232 | **1.369** | 2.692 | 1.956 |
| 8+ | 466,356 | 0.939 | 1.141 | **0.902** |

Worst gated decile calibration gap in pp; no bucket is underpowered; the best arm per row in bold.

**The prediction the first write-up made is confirmed in direction and not in full.** Section 4
predicted that `G2` would not carry `G1`'s weeks-4-7 cost, because it holds the feature's dispersion
at 4.0-4.2 in every week instead of collapsing it to 1.15 in week 0. Measured: `G2`'s weeks-4-7
regression is 0.587 pp against `G1`'s 1.323 pp -- **45% of it** -- and `G2` stays UNDER the 2.0 pp
gate there (1.956) where `G1` crosses it (2.692). `G2` also IMPROVES week 8+ (0.902 against the
reference's 0.939) where `G1` degrades it (1.141). So the dispersion mechanism is real.

What the prediction got wrong: `G2` is WORSE than the reference in week 3 (3.247 against 2.751), the
one early bucket where the raw rate has enough mass to be worth something and the prior-season
target is stale. `G1` is better than `G2` in all of weeks 0, 1, 2 and 3 and `G2` is better than `G1`
in weeks 4-7 and 8+. Neither arm dominates, which is the honest shape of the result and is exactly
why the recommendation stays VALIDATED-PENDING-SHIP-ACTION rather than SHIP.

### What this changes in the recommendation

1. The `first` and `cont` populations no longer disagree about whether shrinkage helps -- both now
   have two arms beating their reference. They still disagree about WHICH target: `cont`'s complete
   ladder prefers the prior-season target (`G2` wins there, `G1` is disqualified by the no-shuffling
   clause), while `first`'s three-arm ladder gives `G1` the bigger segment gain and `G2` the better
   log loss and the better late season.
2. **The obvious next cell is no longer `G2`. It is an arm that is `G1` early and `G2` late** -- that
   is what the two per-week columns say when read together -- and the natural construction is `G3`,
   the two-level prior whose target is itself shrunk by its own reliability, which is NOT RUN on the
   tree and is the one cell of the pre-registered ladder that no population has yet rejected.
   `G3` wins `cont`'s log loss outright (1.498677) and `cont`'s weeks-0-3 cell (+0.427 pp). Running
   `G3` and `G4` on the tree closes the ladder.
3. Nothing here touches the alignment cells or Decision 9. Both tree alignment cells are still
   NOT RUN.

### Reproducing this addendum, and picking up the cells still running

Process A is still fitting `G4` and will write it to `ckpt_a.json` when it finishes, then exit on its
own pre-registered stop without starting another cell. The merged render is a read-only pass that
fits nothing:

```
$env:CBB_THREADS="1"
.venv/Scripts/python.exe scripts/train_possession_outcome_v4.py --render-only --stages 0 \
    --ckpt ckpt_render.json --merge "ckpt_a.json,ckpt_b.json,ckpt_b2.json"
.venv/Scripts/python.exe scripts/diag_po_r4_report.py
```

### 9.11 Addendum 2, 12:13 ET: `G4` landed; the tree ladder is four arms and `G4` is the only arm that costs nothing anywhere

Process A finished `first | F2 | lgbm | G4 | S1_monthly` at 12:13 and then exited on its own
pre-registered stop without starting another cell, writing `G2`, `G3` and the second-seed floor to
NOT RUN. This section supersedes every earlier statement that `G4` on the tree is NOT RUN. `G3` and
the second-seed floor cell remain NOT RUN.

### The `first` feature ladder, complete except `G3` and the floor cell

| arm | rank | log loss | gain vs ref | overall gap | **weeks 0-3** | non-conf | worst quintile slope | gates | beats ref beyond floor |
|---|---|---|---|---|---|---|---|---|---|
| `G0` reference | 0 | 1.515428 | 0.0 | **0.980** | 3.832 | 2.492 | 0.9473 | PASS | -- |
| `G1` shrink to league mean | 1 | 1.515519 | -0.000091 | 1.456 | **2.766 (+1.066)** | 2.351 (+0.141) | **0.9745** | PASS | **YES** |
| `G4` reliability counters | 1 | 1.515626 | -0.000198 | 1.108 | 3.467 (+0.365) | 2.430 (+0.062) | 0.9537 | PASS | **YES** |
| `G2` shrink to prior season | 2 | **1.514837** | +0.000591 | 1.318 | 3.359 (+0.473) | 2.500 (-0.008) | 0.9156 | PASS | **YES** |
| `G3`, floor seed 1 | 3 / -- | NOT RUN | | | | | | | |

**Three of the four measured arms beat the reference on the pre-registered weeks-0-3 cell, and none
of them separates on the primary metric**: the whole log-loss spread across the ladder, -0.000198 to
+0.000591, is 0.98 of one noise floor. This is a segment result end to end, which is what the round
was designed to test.

**The winner is unchanged and it is `G1`, by the tie-break this round fixed in advance.** Best beater
by log loss is `G2` (1.514837); the floor band is 1.515641; `G1` (1.515519) and `G4` (1.515626) both
sit inside it; among those the simplest wins, `G1` and `G4` are tied at rank 1, and 8.1 fixed
before the run that "`G1` wins the tie as the arm that adds no columns at all". `G1` it is.

### Per week of season, all four arms -- and why the tie-break and the evidence now point apart

| week | n | `G0` | `G1` | `G4` | `G2` |
|---|---|---|---|---|---|
| 0 | 36,499 | 6.418 | **4.867** | 6.183 | 4.950 |
| 1 | 40,718 | 5.216 | **4.054** | 5.038 | 4.166 |
| 2 | 43,658 | 3.313 | 2.942 | **2.660** | 2.956 |
| 3 | 43,562 | **2.751** | 2.398 | 3.005 | 3.247 |
| 4-7 | 111,232 | **1.369** | 2.692 | 1.410 | 1.956 |
| 8+ | 466,356 | 0.939 | 1.141 | **0.868** | 0.902 |

`G4` is the **only arm in the round that improves the pre-registered segment without paying for it
anywhere else**: weeks 4-7 move 1.369 -> 1.410 (+0.041, inside any floor this round has), week 8+
improves 0.939 -> 0.868, and the overall gated gap rises only 0.980 -> 1.108 against `G1`'s 1.456.
Its weeks-0-3 gain is a third of `G1`'s, but `G1` buys that gain by pushing weeks 4-7 across the
2.0 pp gate and `G4` does not.

That is the shape of the real finding. **Telling the tree how reliable the rate is (`G4`) is weaker
and safer than making the rate reliable (`G1`).** `G1` changes the feature's scale across the season
and the monthly refit sees a different column in November than in February; `G4` leaves every
existing column bit-identical and adds two counters, so nothing the model already knew is disturbed.

### Recommendation, updated

1. `G1` is the winner of the pre-registered bake-off and this worker does not overrule a tie-break
   it fixed before the run. It stays **VALIDATED-PENDING-SHIP-ACTION**, not shipped, for the
   weeks-4-7 reason in section 4.
2. **On the multi-level evidence the arm to ship is `G4`, not `G1`**, and that is a PM call, not a
   worker's: it is the only arm with no measured cost in any week bucket, it passes every gate, its
   quintile slope improves on the reference (0.947 -> 0.954), and it is the cheapest cell in the
   round (1,866 s against `G1`'s 4,883 s) because it adds two columns and rewrites none. The caveat
   on record: `G4` is the ONLY arm in the round to FAIL a gate on the other population --
   `cont` calibration 2.010 pp against a 2.0 gate -- so a `G4` ship would be `first`-only until
   `cont` is re-read.
3. `G3` -- the two-level prior, the one arm no population has rejected, and the winner of `cont`'s
   log loss and weeks-0-3 cell -- is still NOT RUN on the tree and is the next cell either way.
4. Nothing here touches the alignment cells or Decision 9.


## 10. PROPOSED (cross-reference only) -- the late-game regime round, and what it would ask of this model (written 2026-09-11 by the LATE-GAME lane; NOT RUN, NOT ADOPTED, nothing here changes a served arm)

**This section changes nothing in this model and proposes no arm on this
lane's own grid.** It is recorded here so that a future possession-outcome
round cannot collide unknowingly with a pre-registration that touches the same
conditional law. The full spec lives in `docs/models/late_game/experiments.md`
section 1; the evidence is `docs/tests/late_game_regime_2026-09-11.md`.

**The finding that concerns this model.** Inside the final 2:00 of regulation
with the margin within 6 points, the served `round2_s1` / `C_plus_state` arm
reproduces the CLOCK-conditioned averages well -- three-point share 0.4123
against an actual 0.4219, bonus-FT rate 0.2531 against 0.2775, and a
three-point ramp across the five seconds-remaining buckets that tracks the
actual to within 0.02-0.04 -- and reproduces almost none of the ROLE
conditioning. Trailing minus leading: three-point share **+0.007 in the sim
against -0.158 in the data** (wrong sign), bonus-FT rate **+0.024 against
+0.265** (9% of the asymmetry). The engine plays the same last two minutes for
both teams.

**Why this is not filed as a missing feature on this lane.** `C_plus_state`
already carries `score_diff`, `seconds_remaining` and `in_bonus`
(`STATE_FEATURES`, `src/cbb_sim/models/possession_outcome.py`). The arm has the
columns and still averages the regime away, because the region is 2.6% of
regulation possessions with a sign flip inside it. That is why the
pre-registration is written as a regime layer with a hard gate rather than as a
feature addition here -- **and why arm B of that round is exactly the feature
addition, run head to head, so the simpler hypothesis gets a fair test.** If
arm B wins, the regime-layer hypothesis is rejected and the work lands in this
folder as a round of its own.

**What this lane is asked NOT to do in the meantime.** Do not add a
role/late-game column to the served bundle outside that bake-off: it would
consume the comparison before it is run. Nothing here blocks rounds 5+ on any
other axis.

---

## 11. Round 4b: the two round-4 cells that were NOT RUN -- COMPLETION ROUND, no new pre-registration (worker note written and committed 2026-09-11 13:06 ET, BEFORE any fitting)

**This is a completion round, not a new pre-registration.** Section 8 already pre-registers both
cells, character for character, and nothing in their definition, grading, floor or decision rule is
being changed here:

* `G3` on `first`, fold 2, `S1_monthly`, seed 0 -- section 8.1 arm `G3` (two-level empirical Bayes,
  complexity rank 3), section 8.7 stage 5.
* The **second-seed noise-floor cell** -- `G0 x S1_monthly` on `first`, fold 2, **seed 1**,
  spec-identical to the reference including the whole refit calendar -- section 8.4, section 8.7
  stage 6.

Round 4 stopped on its own pre-registered wall clock (12:15 ET) with both of these written to NOT
RUN (sections 9.7, 9.10, 9.11). This round fits exactly those two and nothing else.

Held fixed and NOT reopened: the trainer (`scripts/train_possession_outcome_v4.py`, unedited), the
grader (round 3's `R1.score`, `conf_window_calibration`, `responsiveness_by`, plus round 4's
`per_week_table`), the design (`design_v4.parquet`, read from cache, never rewritten), the folds,
the seal on 2025-26, the 0.25 pp segment threshold, the complexity ranks, and the section 8.5
decision rule including the `G1`-beats-`G4` tie-break. The merge is round 4's own read-only
`--render-only --merge` path (9.10). Outputs go to versioned sibling checkpoints
(`ckpt_c_g3.json`, `ckpt_d_floor.json`, `ckpt_render_r4b.json`); no round-4 file is overwritten.

**Start time: 2026-09-11 13:06 ET.** Both cells start concurrently, three threads each (six total,
the shared cap), under a hard external deadline: a cell that has not finished by **13:35 ET** is
reported as still running and never as a result. What this round can change is bounded and is stated
before the numbers arrive: the measured seed-1 spread **replaces the block-bootstrap-only PARTIAL
floor of 0.000804** in section 9.2 for `first`, and every "beats the reference beyond the floor"
claim in 9.3, 9.10 and 9.11 is re-read against whichever floor is larger. Two seeds is still PARTIAL
against round 1's five and stays labelled PARTIAL.

### 11.1 Round 4b results (run 2026-09-11 13:01:43-13:27:52 ET, both cells concurrent at three threads each)

**Correction to the start time recorded in section 11 before fitting:** the note was written a few
minutes ahead of the launch and said 13:06 ET; the actual launch was **13:01:43 ET** and both cells
finished at **13:27:52 ET**, 1558 s (`G3`) and 1559 s (floor seed 1). Both reproduced round 3's
reference through stage 0 to 0.00e+00 before their own cell was read.

Artifacts: `data/processed/models/possession_outcome/round4b/` (versioned sibling; round 4's own
directory is byte-unchanged, verified by re-running the merge on round 4's three checkpoints alone
and diffing `grid_results.csv` -- IDENTICAL, 23 cells). The merge is round 4's `--render-only
--merge` path with the output directory rebound by `scripts/run_po_r4b_merge.py`; the trainer, the
grader and the decision function are imported unmodified. 25 cells merged. Over 20 MB (the design
cache is hard-linked, not copied), so gitignored and synced under the `engine_inputs` bulk key.

### 11.2 The floor is MEASURED, and it does not move

| quantity | seed 0 | seed 1 | spread | carried floor | applied floor |
|---|---|---|---|---|---|
| fold-2 log loss | 1.515428 | 1.515541 | **0.000113** | 0.000804 (block bootstrap SE) | **0.000804, UNCHANGED** |
| weeks-0-3 gap (decision cell) | 3.832 | 3.704 | 0.128 pp | 0.25 pp threshold | threshold stands |
| non-conference gap (decision cell) | 2.492 | 2.625 | 0.133 pp | 0.25 pp threshold | threshold stands |

**The second seed confirms the carried floor rather than changing it.** The seed spread on the
primary metric is 0.000113, seven times smaller than the 200-replicate game-block bootstrap SE, so
the bootstrap term stays binding and the applied floor is 0.000804 exactly as round 4 carried it
from round 3. **No "beats the reference beyond the floor" claim in 9.3, 9.10 or 9.11 changes because
of the floor.** The 0.25 pp segment threshold, asserted in round 3 and re-asserted in 8.4, is now
measured against a real spec-identical retrain: both decision cells move 0.128-0.133 pp under a seed
change, half the threshold. The floor is now 2 seeds and stays labelled **PARTIAL** against the five
round 1 pre-registered.

**The per-week BUCKETS are a different matter and this is the round's most consequential
measurement.** The seed-1 reference's own bucket gaps are wk0 5.615 (seed 0: 6.418), wk1 5.219
(5.216), wk2 3.301 (3.313), wk3 3.552 (2.751), **wk4-7 2.062 (1.369)**, wk8+ 0.890 (0.939). A
spec-identical retrain moves single-week buckets by up to **0.80 pp** and weeks 4-7 by **0.69 pp**.
Section 9.11's central argument against `G1` -- that it "pushes weeks 4-7 across the 2.0 pp gate" --
is therefore weaker than it was written: **the reference itself crosses that gate under seed 1**
(2.062). `G1`'s 2.692 is still 0.63 pp above the seed-1 reference, about one measured bucket spread,
so the concern is not retired, but it can no longer be stated as a clean gate crossing that only the
arms cause. Every per-week bucket difference under ~0.8 pp in 9.10, 9.11 and the test doc is at or
inside seed noise and must be read that way.

### 11.3 The completed `first` tree ladder, fold 2, `S1_monthly`, against the measured floor 0.000804

| arm | rank | log loss | gain vs ref | floors | overall gap | **weeks 0-3** (cell) | wk4-7 | wk8+ | non-conf | slope | gates | beats ref beyond floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `G0` reference (seed 0) | 0 | 1.515428 | 0.0 | -- | **0.980** | 3.832 | **1.369** | 0.939 | 2.492 | 0.9473 | PASS | -- |
| `G0` floor cell (seed 1) | 0 | 1.515541 | -0.000113 | -0.14 | 0.977 | 3.704 | 2.062 | 0.890 | 2.625 | 0.9485 | PASS | no (it is the floor) |
| `G1` shrink to league mean | 1 | 1.515519 | -0.000091 | -0.11 | 1.456 | **2.766 (+1.066)** | 2.692 | 1.141 | 2.351 (+0.141) | **0.9745** | PASS | YES (segment only) |
| `G4` reliability counters | 1 | 1.515626 | -0.000198 | -0.25 | 1.108 | 3.467 (+0.365) | 1.410 | 0.868 | 2.430 (+0.062) | 0.9537 | PASS | YES (segment only) |
| `G2` shrink to prior season | 2 | 1.514837 | +0.000591 | +0.74 | 1.318 | 3.359 (+0.473) | 1.956 | 0.902 | 2.500 (-0.008) | 0.9156 | PASS | YES (segment only) |
| `G3` two-level EB | 3 | **1.514615** | **+0.000813** | **+1.01** | 1.362 | 3.371 (+0.461) | 2.323 | **0.833** | **2.315 (+0.177)** | 0.9530 | PASS | **YES (log loss AND segment)** |

`G3` on `cont` (fold 2, cascade): log loss **1.498677**, the best of that population's ladder too;
weeks-0-3 2.422, non-conference 2.175, slope 0.8425, both gates PASS.

**`G3` is the first arm in four rounds to beat the reference on the PRIMARY METRIC beyond the
floor** -- and it beats it by 1.01 floors, which is to say by one hundredth of one floor more than
nothing. That is a real crossing under the pre-registered rule and it is a marginal one; it is
reported as both. `G3` also takes the best non-conference gap and the best week-8+ gap in the round,
and its quintile slope (0.9530) improves on the reference (0.9473). Its cost is weeks 4-7: 2.323
against the seed-0 reference's 1.369 (+0.954) but against the seed-1 reference's 2.062 (+0.261,
inside the 0.69 pp bucket spread the floor cell just measured).

### 11.4 The verdict, re-stated with the measured floor -- THE WINNER CHANGES, and it changes because of `G3`, not because of the floor

Run through `decide_v4` unmodified:

* Winners (arms that beat the reference beyond the floor on log loss OR by >0.25 pp on either
  decision cell, pass both gates, and give back no more than 0.25 pp on the other segment):
  **`G1`, `G2`, `G3`, `G4`** -- all four.
* Best beater by log loss: **`G3`, 1.514615**. Eligibility band = best + floor = **1.515419**.
* `G1` (1.515519) and `G4` (1.515626) now fall **outside** that band and lose eligibility. In
  round 4 they were eligible only because no arm had pulled the band down; `G3` pulls it down by
  0.000222.
* Eligible: `G2` (rank 2) and `G3` (rank 3). Simplest wins. **Winner: `G2`.**

**`cont`'s feature ladder independently selects `G2` as well**, by the same mechanism (its best
beater is `G3` at 1.498677). Two populations, one winner, no tie-break invoked.

The scheme ladder is unchanged and still holds only the reference and its floor cell: `A1`
(`S1_conf_aligned`) and `A2` (`S1_weekly`) remain **NOT RUN**, so **Decision 9c is still unresolved
for this sub-model** and nothing here amends Decision 9.

**Does any arm dominate the reference on every cell? No -- not one.** `G3` loses the overall gated
gap (1.362 vs 0.980) and weeks 4-7. `G2` loses non-conference (-0.008), the overall gap, weeks 4-7,
and its quintile slope (0.9156) is the round's worst and is further from 1.0 than the reference's.
`G1` loses the overall gap, weeks 4-7 and week 8+. `G4` loses log loss and the overall gap, and
fails `cont` calibration (2.010 pp). The round-4 finding stands: **this is a segment result, and
every arm buys its segment somewhere.**

### 11.5 What is shippable, and whose call it is

**Under the pre-registered decision rule of 8.5, read against the measured floor, the shippable arm
for `first` is `G2`** -- the same arm `cont` selects, passing both gates on both populations. Section
9.11's recommendation of `G1` is **SUPERSEDED**: `G1` was the winner only while the eligibility band
sat above it, and the band moved when `G3` landed. Section 9.11's separate multi-level argument for
`G4` is also superseded on eligibility, and `G4`'s `cont` calibration failure (2.010 pp) is
unchanged.

Three things the PM should weigh against the rule's answer, stated as evidence and not as an
overrule:

1. `G2` wins by the tie-break, not by dominating. On the primary metric `G3` is better by 0.000222
   (0.28 floors) and on non-conference `G3` is better by 0.185 pp; `G2`'s only cell-level win over
   `G3` is the overall gated gap (1.318 vs 1.362, 0.04 pp).
2. `G2` carries the round's worst responsiveness slope, 0.9156 against the reference's 0.9473 -- a
   matchup-responsiveness regression that passes the gate but moves the wrong way, which the
   standing matchup-specific rule says to look at explicitly.
3. `G3` is the only arm that beats the reference on the primary metric at all, and the only arm no
   population has rejected on any gate.

**The ship is the PM's call, not this worker's.** Nothing here changes a served default, and no arm
is adopted by this section. Status recorded in `docs/models/change_ledger.md` in the same commit.

### 11.6 Cells still NOT RUN after round 4b

| cell | population | arm | fold | features | scheme | status |
|---|---|---|---|---|---|---|
| `A1` | first | lgbm | F2 | `G0` | `S1_conf_aligned` | NOT RUN (round 4 stage 7) |
| `A2` | first | lgbm | F2 | `G0` | `S1_weekly` | NOT RUN (round 4 stage 8) |
| stage 9 | first | lgbm | **F1** | winner | `S1_monthly` | NOT RUN -- no fold-1 confirmation exists for ANY tree shrinkage arm |

Seeds 2-5 of the noise floor are also not run; the floor is 2 seeds and PARTIAL.


---

## 12. Round 5 pre-registration -- the SHIP GATE: `G2` and `G3` WIRED, paired closed loop against the served possession-outcome model (2026-09-18)

Appended VERBATIM **BEFORE** the engine modes `round4b_G2` and `round4b_G3` existed, before any
`po4b_*` results directory existed, before any engine artifact for either arm was fitted, and before
any closed-loop number was read. Sections 1 through 11 are STATIC and are NOT edited.
`experiments.md` is append-only.

**PREMISE.** Section 11.4 ran the pre-registered decision rule of 8.5 against the round-4b measured
floor and selected **`G2`** (shrink the early-season as-of style rate toward the SAME TEAM's
prior-season centred rate) as the OFFLINE winner on both populations -- `first` by the eligibility
band plus the simplicity tie-break, `cont` independently. **`G3`** (two-level empirical Bayes) is the
only arm in four rounds to beat the reference on the PRIMARY metric beyond the floor (1.01 floors)
and holds the round's best non-conference and week-8+ gaps. Section 11.5 recorded the ship as the
PM's call and named three things to weigh, one of which is the standing matchup-specific rule:
**`G2` carries the round's worst responsiveness slope, 0.9156 against the reference's 0.9473.**

The CLAUDE.md standing rule is that **an offline winner ships only after a paired-seed sim run shows
no gate regressed.** That run has never been done for any possession-outcome arm. Round 5 is that
run and nothing else. **It changes no default under any outcome; the PM switches the served arm or
does not.**

### 12.1 The arms, and what "wired" means here

| arm | `ENGINE_EVENT` value | what it is | status |
|---|---|---|---|
| **R** | `round2_s1` (the DEFAULT, untouched) | the served possession-outcome model: round-2 winners `lgbm`+S1 on `first`, `cascade`+S1 on `cont`, feature set `C_plus_state`, raw-centred style columns | REFERENCE |
| **G2** | `round4b_G2` | identical in every respect except that the eight style columns are `w*raw_c + (1-w)*prior_c` (section 8.1 arm `G2`) | ARM UNDER TEST, DEFAULT-OFF |
| **G3** | `round4b_G3` | identical except the eight style columns are `w*raw_c + (1-w)*w2*prior_c` (section 8.1 arm `G3`) | COMPARATOR, DEFAULT-OFF |

`adapters.py` DEFAULTS ARE NOT TOUCHED. `ENGINE_EVENT` stays `round2_s1`; the only change in that
file is widening the `EventAdapter.load` dispatch so a `round4b_*` mode resolves its artifact
directory the same way `round2_s1` already does. Every other sub-model default is untouched.

**The shrunk columns are READ, never re-derived.** `off_{r}_g2` / `opp_def_{r}_g2` (and the `_g3`
pair) come out of `data/processed/models/possession_outcome/round4/design_v4.parquet` exactly as
round 4b's grade scored them, under round 3c's standing rule "the fitted object is READ, never
reimplemented". The fitted `k_r` are not re-estimated in the engine: they are already baked into
those columns and are recorded in `round4/design_v4.meta.json`.

**The arms differ from the reference ONLY by the shrinkage.** Pre-registered precondition, checked in
the builder and aborting it on failure: over the 10,890 season-2025 team-games common to
`round2/design.parquet` and `design_v4.parquet`, the eight NON-style `TEAM_COLS`
(`off_rating_off_c`, `off_rating_def_c`, `def_rating_off_c`, `def_rating_def_c`, `site_home`,
`site_away`, `season_idx`, `days_since_start`) and the eight raw style columns themselves must agree
to **0.0 exactly**. (Confirmed read-only before this pre-registration was written, on all sixteen
columns: max abs diff 0.000e+00. The builder re-asserts it rather than trusting that check.)

**No booster is persisted by `train_possession_outcome_v4.py`**, exactly as none is by the round-2
trainer. So the engine refits each arm's OWN spec -- `feature_set_v4(arm, pop)` through
`cbb_sim.models.possession_outcome.fit_arm`, seed 0, the same `lgbm`/`cascade` choice per population
round 2 adopted -- on the SAME S1 monthly schedule the served model uses (six refits,
2024-11-01 .. 2025-04-01), and re-asserts the per-GAME legality property
(`max_train_date < tipoff`) that `build_engine_event_round2.py::_assert_no_leak` enforces. This is
`build_engine_event_round2.py`'s procedure applied to a different feature set; it is a new script
(`scripts/build_engine_event_round4b.py`), and the round-2 builder and its output directory are not
touched.

**Lookup, not live modelling.** The sim loop makes no per-game or per-possession model call under any
arm: the adapter assembles one contiguous matrix per (population, monthly refit) over the whole
active batch and makes exactly one batched call, which is the `round2_s1` path unchanged and is what
"vectorised, no live model calls" means everywhere in this engine. The team block is a precomputed
`(G, 2, 16)` array. Thread counts are pinned to 1 in every worker.

### 12.2 The precondition: the served path must stay BIT-IDENTICAL

Run BEFORE any arm artifact is fitted and before any G2 or G3 number is read, exactly as clock round
5d ran it (section 26.1 of `docs/models/clock/experiments.md`): a 60-game x 5-seed smoke on the
UNCHANGED engine default, digested by `scripts/digest_engine_run.py --compare` against the digest of
`results/engine_v0/smoke60x5_default_v5b` -- sha256
`492300a7fd1e6dc388a3c47b822f61e04da6501720ef762bde0afe7ba15451a1`, the digest the PM's clock
serving decision (`e3ccce5`) and the two lanes after it were verified on. **A mismatch of any kind
fails round 5 outright and the round reports a wiring defect, not a possession-outcome result.**
`pytest tests/test_engine.py` must pass unchanged.

### 12.3 The closed loop

500 games of the standing subset rule (the F2 2025 slate sorted by `game_id` ascending, every 11th
row, first 500) -- **the same game sample the clock 3c, 4, 5, 5b and 5d closed loops used**, so this
round's table sits beside theirs. `scripts/run_po4b_closed_loop.py`, a copy of
`run_clk4_closed_loop.py`'s harness with the varying flag moved from `ENGINE_CLOCK` to
`ENGINE_EVENT`.

**Pinned sub-models: the SERVED STACK, read from `adapters.py`'s own defaults and written into every
`run_meta.json`** -- `ENGINE_CLOCK=v5b_glat_pmean` (ADOPTED 2026-09-11), `ENGINE_FG_MAKE=round4_B1`,
`ENGINE_FG3=decision8`, `ENGINE_REBOUND=s1_weekly`, `ENGINE_FREE_THROW=s1_conf_aligned`,
`ENGINE_ROTATION=reference`, `ENGINE_ROTATION_SCHEME=s1`, `ENGINE_USAGE=reference`, inputs v2.
`ENGINE_EVENT` is the ONLY flag that differs between arms.

**DEVIATION FROM THE CLOCK-LANE PRECEDENT, DECLARED HERE BEFORE THE RUN.** Clock 5d reused
`results/engine_v0/clk5b_B1_s25` as its reference. That run pins
`ENGINE_FG_MAKE=round3_shooter_S_C_s1`, the *superseded interim* fg_make model; the served default is
`round4_B1`. Reusing it would price a possession-outcome arm on a stack that is not the served one,
which is exactly what a ship gate must not do. **The reference is therefore RE-RUN here on the served
stack rather than reused**, at a cost of one extra run. The clock lane's floors are for the same
reason NOT carried over: every floor in this round is measured on this stack, in this round.

**Pairing.** All three arms draw from the same RNG families at the same ordinals; no arm introduces a
family, an ordinal or a draw the others do not make. `(seed, game_id, family)` seeding is unchanged,
so for a given (game, seed) the three arms consume the SAME uniform stream and differ only in the
probability vector the event draw is compared against. Trajectories diverge downstream of the first
differing terminal event, which is inherent to a closed loop over a changed sub-model and is the same
sense in which rounds 3c, 4, 5, 5b and 5d were paired.

**Seeds.** 25 paired seeds `0..24` on each of the three arms, and a fourth run -- the **noise-floor
run** -- of the REFERENCE arm on seeds `1000..1024`, spec-identical in every other respect. That
fourth run is the floor: `|N - R|` on each line is the seed-offset noise band, and `|arm - R|` is the
movement being judged. Compute cap: **at most 8 engine worker processes at any instant**, other lanes
share this 20-core machine.

### 12.4 The lines, and how a floor is applied

**Primary: gates G1-G9, no regression.** One grading path, `scripts/eval_gates.py` (UNEDITED) at
`docs/gates.yaml`'s tolerances, run blind over all four results directories; the arm's identity is
read from `run_meta.json` and used only to label the row. The four reports are put side by side by
`scripts/diag_pair_gate_reports.py` (UNEDITED) with `--a` the reference, `--b` the arm and `--noise`
the floor run, which is the harness's own A/B/N contract. Every headline check of every gate is a
line; a line whose movement does not exceed its own floor is reported INSIDE FLOOR and is a
non-finding.

"**Regresses**" means: the line moves AWAY from its gate target (or, where the target is a band, away
from the centre of the band) by **more than one measured floor**, OR its gate status changes from
PASS to FAIL. A line whose status is NEEDS-INSTRUMENTATION or UNDERPOWERED in the reference is
reported and never scored.

**The PM's added line -- responsiveness, and it is a one-sided line.** The standing matchup-specific
rule, read CLOSED-LOOP rather than offline, by a new grader
`scripts/grade_po4b_closed_loop.py`. For each of the three Decision-8 drivers in
`PO.RESPONSIVENESS_SPECS` -- (`off_3pa_c`, FGA_3), (`off_rim_c`, FGA_rim), (`off_tov_c`, TOV) -- the
1,000 team-games of the 500-game subset are bucketed into **five quintiles of the OFFENCE team's own
as-of driver value**, taken from the ROUND-2 raw-centred team block, which is **identical in all
three arms**, so every arm is cut on exactly the same rows. Per quintile: the engine's realised
per-possession rate (`fga3/poss`, `fga2_rim/poss`, `tov/poss`, averaged over seeds then over
team-games) and the ACTUAL per-possession rate on the same team-games (`tpa/poss_team` and
`tov/poss_team` from `reference.load_actual_team_box`; `ev_fga_rim/poss_team` from
`reference.load_team_shot_truth`, the event layer, because a box score has no rim class).

    slope_ratio(driver) = (sim Q5 - sim Q1) / (actual Q5 - actual Q1)

which is `train_possession_outcome_v3.responsiveness_by`'s span-ratio definition, transported to the
closed loop. The **headline slope** is the driver whose `slope_ratio` is furthest from 1.0, matching
`train_possession_outcome_v4.quintile_slope_worst`. A driver whose realised ACTUAL quintile span is
under 2 pp is EXEMPT under Decision 8 and is reported as exempt, never scored.

> **The line, as the PM set it: the responsiveness slope must not FALL BELOW the reference's by more
> than the floor** -- per driver and on the headline -- where the floor is `|N - R|` on that same
> quantity. It is one-sided on purpose. Offline every measured slope sits below 1.0 (reference
> 0.9473, `G2` 0.9156, `G3` 0.9530), so "falls" and "flattens toward the league mean" are the same
> movement. **If a closed-loop slope comes out ABOVE 1.0 the two readings stop coinciding**; in that
> case both readings are reported -- the signed fall, and the change in `|slope - 1|` -- and the
> round states which one it scored the line on before it states the verdict, and does not choose
> after seeing the answer: **the scored reading is the signed fall**, fixed here.

**Pre-registered SEGMENT evidence, every cell on every arm, all four runs.** These are EVIDENCE, not
gates; they are where round 4's whole premise says the arms act, and a regression here is reported
even though it cannot by itself fail the round.

| cell | definition |
|---|---|
| weeks 0-3 | games in calendar weeks 0-3 of the test season, the round-4 decision cell |
| weeks 4-7 | the cell round 4 found the shrinkage arms pay for |
| weeks 8+ | the settled season |
| conference | conference games, by the published schedule's own flag |
| non-conference | the round-4 second decision cell |

Per cell, per arm: n games, margin MAE, margin bias, total MAE, total bias, possessions/team-game
mean, PPP, and the three shot-mix rates -- against the actuals on the SAME games, never against a
season-wide figure. **A cell with fewer than `min_cell_n` = 300 team-games is labelled UNDERPOWERED
and is never read as signal or as the absence of signal.** On a 500-game subset the week-0-3 and
week-8+ cells are expected to be thin and the round says so in advance.

**Multi-level evidence, per the standing rule.** Overall (the gate table); per-game (the paired
per-game margin/total/possession deltas, their distribution, and the count of games moving more than
one per-game floor); per-team (the teams appearing in the subset, their own margin and PPP bias under
each arm, and the relation between an arm's per-team change and the size of the shrinkage that arm
applies to that team); per-possession-type (the engine's realised terminal-event mix per possession
against the event layer's own actual mix on the same games). Player-level is NOT read: no arm here
touches the player layer, and the runs carry `keep_players=False`.

### 12.5 Decision rule

**`G2` is the arm this lane puts forward for SERVING if and only if BOTH:**

1. **no gate line G1-G9 regresses beyond its measured floor**, in the sense fixed in 12.4; and
2. **the responsiveness line holds**: neither the headline slope nor any non-exempt driver's slope
   falls below the reference's by more than that quantity's own floor.

**`G3` is read as a COMPARATOR on the identical table and is not put forward**, because the
pre-registered offline rule of 8.5 already selected `G2` over it on simplicity within the eligibility
band (11.4) and this round does not re-open the offline selection. If `G2` fails either condition and
`G3` passes both, that is reported as a stated finding and handed to the PM, who may re-open 8.5; this
worker does not.

If both conditions hold for `G2`, **ties go to the simpler model and `G2` is the recommendation**. If
(1) fails, `G2` is NOT put forward regardless of (2). If (1) holds and (2) does not, `G2` is recorded
as **BLOCKED ON RESPONSIVENESS** and the reference stays the arm put forward -- the standing
matchup-specific rule is not waived because the aggregate gates are clean, and section 11.5's point 2
is exactly the risk this clause exists for.

**A failing line is not waived because the offline table is good.** The offline pass of 11.3 carries
no weight against a closed-loop regression.

**This lane does not change the served default under any outcome**, does not edit
`docs/models/change_ledger.md`, and does not touch `HANDOFF.md`, `PROJECT_STATUS.md`,
`ARCHITECTURE_DECISIONS.md` or `CLAUDE.md`. Results go to section 13 and to
`docs/tests/possession_outcome_closed_loop_2026-09-18.md`. The PM makes the ship decision from the
table.

### 12.6 Budget, drop order, and what is written if the clock runs out

Stage order IS the drop order. Anything not reached is written to the results as **NOT RUN** and
never as a result.

| stage | what | cost |
|---|---|---|
| 0 | bit-identity smoke on the unchanged default + `pytest` | minutes |
| 1 | build the `G2` and `G3` engine artifacts (two processes, 4 threads each) | ~20-40 min |
| 2 | adapter unit tests + a served-path agreement probe | minutes |
| 3 | reference run `po4b_R_s25`, seeds 0..24 | ~10 min |
| 4 | `po4b_G2_s25`, seeds 0..24 | ~10 min |
| 5 | `po4b_G3_s25`, seeds 0..24 | ~10 min |
| 6 | floor run `po4b_R_s25_floor`, seeds 1000..1024 | ~10 min |
| 7 | grade: `eval_gates.py` x4, `diag_pair_gate_reports.py`, `grade_po4b_closed_loop.py` | minutes |
| 8 | docs | -- |

**If stage 6 does not finish, the round has no floor and reports no verdict**, only a table with the
movements unpriced; it does not substitute the clock lane's floors from a different stack. If stage 5
does not finish, `G3` is reported NOT RUN and `G2` is still decidable, because `G3` is a comparator
and not a condition of the rule.

### 12.7 What round 5 cannot say

It cannot re-open the offline grade: nothing is refitted on the bake-off's frame, no round-4 or
round-4b artifact is rewritten, and the shrunk columns are read as committed. It cannot resolve
Decision 9c -- `A1` and `A2` are still NOT RUN (11.6). It cannot give a fold-1 confirmation: no tree
shrinkage arm has one, and running the engine on F1 is not one. It cannot price either arm on any
game set other than the 500-game subset, on the sealed 2025-26 season, or against market lines. And a
PASSING verdict here is a RECOMMENDATION, not an adoption: adoption is the PM's, on the full list.

---

## 13. Round 6 pre-registration: the team-foul accrual law that sets the bonus state, and the `FT_trip_bonus` conditional -- gate G4's FTA/FGA miss (PROPOSED, written 2026-09-18 by the G4 diagnostic lane BEFORE any modelling; NOT RUN, NOT ADOPTED, no served default changed)

Evidence this round is written against: `docs/tests/g4_oreb_fta_diagnostic_2026-09-18.md`.
That document decomposes the engine's -1.229 pp pooled FTA/FGA miss on fold 2 into
channels that close (unexplained residual +0.001 pp). FTA/FGA is a ratio of two
per-possession rates, so the engine's +2.0 possessions/game cancels exactly and is
not a channel:

| channel | pp | share |
|---|---:|---:|
| technical FTs + event/box feed gap (no technical-FT rule in the engine) | -0.282 | +23.0% |
| and-one FTA / FGA | +0.122 | -10.0% |
| shooting-foul trip FTA / FGA | +0.378 | -30.8% |
| **bonus-trip FTA / FGA** | **-1.384** | **+112.6%** |
| other event-layer FTA (168 chances) | -0.044 | +3.6% |
| tap-subset vs full-run offset (measurement) | -0.020 | +1.6% |

**The miss is one thing: the bonus-trip rate is 10.4% low** (5.2586 vs 5.8686
trips per 100 possessions). It is not trip SIZE -- attempts per trip are right on
every kind (and-one 1.000x, shooting 1.022x, bonus 1.004x actual), the 1-and-1 vs
double-bonus split is right (52.9% double-bonus trips vs the actual's 51.4%), and
the bonus-era flag is not implicated. It is also not the FGA denominator: FGA/poss
is slightly LOW in the sim, which pushes the ratio the wrong way for the gate.

Split of the bonus-trip rate gap (Shapley, residual 9e-19):
**bonus-state OCCUPANCY 59.0%, CONDITIONAL trip rate 41.0%.**

| | sim | actual |
|---|---:|---:|
| P(offence in bonus) | 0.28373 | 0.30201 |
| P(`FT_trip_bonus` \| in bonus) | 0.16058 | 0.16763 |
| P(`FT_trip_bonus` \| not in bonus) | 0.00034 | 0.00037 |

and the mechanism behind the occupancy half is visible by game minute. The engine
accrues non-shooting team fouls from **one constant**,
`silent_foul_per_possession = 0.123346`
(`engine_rules_from_data`; `loop.py`, "non-shooting foul that awards no attempt"),
applied identically in both halves and independent of clock, score and role:

| game minute | sim P(in bonus) | actual | delta pp | sim P(trip \| bonus) | actual |
|---|---:|---:|---:|---:|---:|
| 5-9 | 0.0580 | 0.0317 | +2.63 | -- | -- |
| 10-14 | 0.3039 | 0.2343 | **+6.95** | 0.1070 | 0.1212 |
| 15-19 | 0.6102 | 0.5790 | +3.13 | 0.1255 | 0.1301 |
| 25-29 | 0.0975 | 0.1431 | -4.56 | 0.1385 | 0.1391 |
| 30-34 | 0.3984 | 0.5432 | **-14.48** | 0.1576 | 0.1567 |
| 35-37 | 0.6533 | 0.8060 | **-15.27** | 0.1736 | 0.1710 |
| 38-40 | 0.7935 | 0.8037 | -1.02 | 0.2480 | 0.2696 |

Every cell is powered (57k-112k chances). **The engine reaches the bonus too early
in the first half and up to 15 pp too rarely in the second**, and conditional on
actually being in the bonus the served event model is right to within 1% through
minutes 25-37. So most of the "41% conditional" is the engine querying a
correctly-shaped model at the wrong (bonus, clock) states. The FT supply follows
with the same sign flip: sim FTA/FGA is **+1.8 pp too high** in minutes 5-14 and
**-6.1 pp too low** in minutes 30-34. A fix that raises the overall foul rate makes
the first half worse.

**Scope, stated before any arm is run.** The final 2:00 of regulation carries
-0.254 pp = **20.7%** of the FTA/FGA miss, and it is already inside
`docs/models/late_game/experiments.md` section 1 (arm rank 1, the fouling
channel). **This round does not pre-register anything about that window and must
not be run in a way that collides with it**: every arm below is fitted and graded
on `period <= 2 and seconds_remaining > 120` and reports the window as a
held-out segment only. The technical-FT channel (23%) belongs to
`docs/models/free_throw/experiments.md` section 9 and is not an arm here either.
What this round owns is the other ~56%, which sits in minutes 25-37.

### 13.1 Candidates

**Block F -- the foul-accrual law** (the occupancy half, 59%). What the engine
needs is the arrival process for personal fouls that award no trip, i.e. the thing
that moves `team_fouls` and therefore `in_bonus`. Today it is one scalar.

| arm | what it is |
|---|---|
| `F0` | **reference**: the served constant `silent_foul_per_possession`, one number for the whole game |
| `F1` | the same constant, **fitted separately per half** (the minimum arm that can produce the observed sign flip) |
| `F2` | a rate table over `(period, seconds_remaining bucket)` -- clock only, no score, no role |
| `F3` | a fitted Bernoulli model of "this possession produces a non-shooting personal foul on the defence", over the state block already in `STATE_FEATURES` plus `def_team_fouls`, `off_team_fouls`, `is_transition` and the defence's as-of foul rate, league-centred |
| `F4` | `F3` promoted to a **seventh possession-outcome class** (`FOUL_no_FT`), fitted inside the existing multinomial rather than as a side Bernoulli |

`F4` is the structurally cleanest arm and the most invasive: it changes
`PO.CLASSES`, every downstream index, and the meaning of every stored round-2/3/4
artifact. It is on the list because the round must report what the clean form is
worth, not because it is expected to be cheap. `F1` is the cheapest arm that can
reproduce the measured sign flip and is the one to beat.

**Block T -- the `FT_trip_bonus` conditional** (41%, most of which `F` is expected
to absorb by fixing the query states). Run only after the best `F` arm is known,
against it, so `T` is measured on the residual and is not credited with `F`'s work.

| arm | what it is |
|---|---|
| `T0` | **reference**: the served `first`/`cont` arms, unchanged |
| `T1` | `T0` + the defence's own as-of team-foul rate and the offence's as-of drawn-foul rate, both league-centred, as features |
| `T2` | `T0` + `def_team_fouls` and `off_team_fouls` as counts (today the model sees only the binary `in_bonus`, so it cannot tell 7 fouls from 11) |
| `T3` | `T1` + `T2` |

**Block H -- the site asymmetry** (a separate, smaller finding the diagnostic
turned up and that no other lane owns). The actual FT rate is 0.3486 at home
against 0.3068 away, a +4.18 pp home whistle; the sim gives +2.99 pp, reproducing
72%. `site_home`/`site_away` are already in the served bundle, so this is not a
missing column.

| arm | what it is |
|---|---|
| `H0` | **reference** |
| `H1` | `site_home` / `site_away` interacted with the FT-trip classes (explicit interaction terms for the linear arms; for the tree arms, a pre-declared monotone-free split budget so the round reports whether the tree simply is not finding it) |

### 13.2 Features

Base bundle is the served `C_plus_state` (`possession_outcome.feature_set`),
unchanged in name and order. Additions are declared per arm above and nowhere
else. Every rate feature is expressed relative to its own snapshot's league mean;
`site_home`/`site_away` stay in every arm. Per Decision 9 (PENDING EVIDENCE),
opponent adjustment of the as-of rate features, a conference-game flag, and refit
cadence/conference alignment are carried as mandatory bake-off cells for any new
rate feature `T1` introduces, reported even where they lose.

### 13.3 Folds

Fold 1 trains through 2022-23 and tests 2023-24. Fold 2 trains through 2023-24 and
tests 2024-25. **Fold 2 selects.** 2025-26 is SEALED. The refit schedule is held at
the served `S1` for every arm so this is not also a scheme bake-off.

### 13.4 Primary metric

Two primaries, one per block, because `F` and `T` predict different things and
pooling them would hide which one moved:

- **Block F:** log loss of the non-shooting-foul indicator on fold-2 possessions
  (`F4` scored on the same indicator, marginalised out of its seven-class
  prediction, so every `F` arm is compared on one number).
- **Block T (and H):** six-class log loss on fold-2 chances, the model's existing
  primary, so the ladder stays comparable with rounds 1-4b.

One pre-declared **closed-loop** reading, binding through 13.7 rule 7 rather than
replacing the primaries: the engine's **bonus-state occupancy profile by
five-minute game-minute bucket** against the actual's, scored as the maximum
absolute per-bucket gap over minutes 0-37. Served value **15.27 pp**; target
<= 4 pp.

### 13.5 Segment breakdowns (every arm, every fold)

Underpowered cells labelled, never folded into a pass or a fail (min cell n = 300
chances):

1. by **five-minute game-minute bucket** (0-4 ... 35-37), the round's own target
   cut -- occupancy, bonus-trip rate, and FTA/FGA;
2. by half, and by `in_bonus` x half;
3. by `def_team_fouls` count (0-3, 4-6, 7-9, 10+);
4. home / away / neutral (served FTA/FGA gap -0.43 / -1.62 / -2.63 pp);
5. conference vs non-conference (served -0.61 / -2.28 pp) and the first four
   weeks of conference play;
6. by **month** (served -2.99 Nov, -1.21 Dec, -0.95 Jan, -0.20 Feb, -0.67 Mar) --
   this miss is largely an early-season one and any arm must say what it does
   there;
7. by **2024 prior-season FT-rate quintile of the offence** -- slope ratio,
   monotone steps, per-quintile gap (served: **0.523**, 4/4, -0.09 pp in Q1 to
   -2.11 pp in Q5), and the same cut split Nov-Dec / Jan / Feb-Apr (served
   **0.356 / 0.636 / 0.663**);
8. the **final 2:00 of regulation as a HELD-OUT segment**, reported and never
   fitted on, so this round and the late-game lane can be read against each other
   without either claiming the other's ground.

### 13.6 Noise floor

A **spec-identical retrain under a second seed** for the reference and each
block's leading arm on fold 2, including the identical refit calendar; the floor
is the observed spread on that block's primary. Round 4b measured this model's
own floor at 0.000113 on six-class log loss against an applied floor of 0.000804,
and measured that per-week buckets move up to 0.80 pp under nothing but a seed
change -- **that bucket-noise figure applies to breakdown 6 here and no monthly
difference under ~0.8 pp may be read as an arm effect.** Minimum two seeds.

### 13.7 Decision rule

1. An arm is eligible only if it beats its block's reference on that block's
   **primary** by more than the measured floor.
2. Among eligible arms the winner is the lowest primary; ties inside one floor go
   to the **simpler** arm, ordered `F0 < F1 < F2 < F3 < F4`, `T0 < T2 < T1 < T3`,
   `H0 < H1`.
3. A winner must pass this model's existing round-1 gates (calibration and
   responsiveness) unchanged.
4. A winner must **reduce** the maximum per-bucket bonus-occupancy gap of 13.4
   and must **not** convert the sign flip into a uniform shift: the round reports
   the signed per-bucket gaps and an arm that fixes minutes 30-37 by making
   minutes 5-14 worse than +2.63 pp is rejected.
5. A winner must **not reduce** the 2024-prior-quintile FT-rate slope ratio in any
   of the three season segments of breakdown 7.
6. **Fold 1 confirmation is required** before any ship recommendation.
7. **Nothing ships on offline evidence.** An offline winner ships only after a
   paired-seed closed-loop run at >= 25 seeds on the served stack shows no gate
   regressed (Decision 10), the occupancy target of 13.4 is met, and the engine's
   pooled FTA/FGA moves toward 0.32955 without moving G1's possession mean, G2's
   PPP terciles, or G4's TOV% / OREB% / eFG% beyond their measured bands. Because
   free throws stop the clock, **G1's possession count and the clock lane's own
   gates are explicit no-regression lines for this round**, not afterthoughts.
8. Ties at every level go to the simpler model. If no arm clears rule 1, **no arm
   is adopted** and the round says so.

### 13.8 What this round may not do

No post-hoc multiplier, cap, clip, offset, calibration curve or blend on model or
sim output. In particular, **tuning `silent_foul_per_possession` to hit the gate is
banned**: `F1`-`F4` all replace the constant with a law fitted on training folds,
and an arm whose scalar is chosen to make the 2025 FTA/FGA land on 0.32955 is not
on the list and may not be added to it. The 2025-26 season stays sealed. The
final-2:00 window is held out (13.5 breakdown 8) and `docs/models/late_game/` is
not edited by this round. Artifacts go to a versioned sibling
`data/processed/models/possession_outcome/round6/`; rounds 1-4b directories are
not overwritten.

---

## 14. `A1`/`A2` attempted on AWS box three (2026-09-18): NOT RUN -- LightGBM in the
     cloud image does not parallelize, and the two cells cost far more than "minutes"

Section 11.6 leaves `A1` (`first`/lgbm/F2/`G0`/`S1_conf_aligned`, ~29 refits) and `A2`
(`first`/lgbm/F2/`G0`/`S1_weekly`, ~23 refits) NOT RUN, estimated at ~6 h and ~4.8 h on a
shared 20-core box at a 3-thread cap, "minutes on the 196-core box." This section reports an
attempt on a dedicated `c7a.48xlarge` (192 vCPU) AWS box under `scripts/train_possession_outcome_v4.py
--stages 7` / `--stages 8` and why it did **not** land a result.

**Setup.** `data/processed/models/possession_outcome/round4/design_v4.parquet` (94.8 MB) and the
round-4 checkpoints synced via the `model_artifacts` HF bulk key (already fully mirrored, no push
needed). One dependency was missing from that sync: `build_conference_flags` reads
`data/raw/hoopr/schedules/mbb_schedule_{2022..2025}.parquet` directly (not through `model_artifacts`
or `engine_inputs`), and `.dockerignore` excludes `data/raw/`, so the first launch attempt crashed
immediately (`FileNotFoundError: no hoopR schedule for season 2022`). Fixed by pulling the four
schedule parquets directly from the HF dataset (`raw/hoopr/schedules/mbb_schedule_<year>.parquet`)
and bind-mounting them into the container at runtime, rather than a full `--dirs raw` pull (~1.6 GB,
not worth the time cost for four files). **Worth fixing properly**: either add these four files to
the `Dockerfile.cbb` build-time check / `engine_inputs` sync, or note the dependency in this trainer's
own docstring, so the next cloud run does not rediscover it.

**LightGBM does not multi-thread in this container image, confirmed by an isolated benchmark.**
`CBB_THREADS=150` was passed at first (matching the trainer's own env-var contract), but
`Dockerfile.cbb` bakes `OMP_NUM_THREADS=1` as an image-level `ENV`, and the trainer's
`os.environ.setdefault("OMP_NUM_THREADS", _THREADS)` is a no-op when the variable is already set --
so `CBB_THREADS` never reached LightGBM. Fixed by passing `-e OMP_NUM_THREADS=150` (and the sibling
`MKL_/OPENBLAS_/NUMEXPR_/LIGHTGBM_NUM_THREADS`) directly at `docker run`, which does override the
image default. **CPU usage stayed at ~100% (one core) regardless.** An isolated timed benchmark
inside the running container settles the question without ambiguity:

```
X = np.random.rand(300000, 50); y = 6-class random labels
lgb.LGBMClassifier(n_estimators=100, num_leaves=63, n_jobs=1).fit  ->  18.13 s
lgb.LGBMClassifier(n_estimators=100, num_leaves=63, n_jobs=150).fit -> 18.32 s
```

**`n_jobs` has zero measured effect** on this image's `lightgbm==4.7.0` wheel (pinned in
`requirements-cloud.txt`) -- a known class of issue with some manylinux LightGBM wheels not linking
a working OpenMP runtime in a slim base image (`python:3.12-slim`), silently falling back to
single-threaded execution rather than erroring. This is an infrastructure finding independent of
possession-outcome: it means the `c7a.48xlarge`'s 192-vCPU advantage is **not** realised by this
trainer at all -- every refit runs on one core no matter the box size, so "minutes on the 196-core
box" was never true for this trainer, only for the possession-loop-vectorized engine sweep (which
does not call LightGBM inside the timed path).

**What was run and what it cost.** `A1` and `A2` were launched as two independent single-threaded
containers (splitting the two cells across two dedicated cores rather than one 2-stage sequential
process) at 22:41:46Z. **Neither had completed a single cell after 41 minutes** (checkpoint files
unchanged in size from their stage-0-only state at 23:22:34Z, when the session's time budget forced
a stop). Stage 0's reference re-score reproduced round 3's log loss to 0.00e+00 on both, confirming
the trainer, the design cache and the grader all work correctly on this box -- the blocker is purely
wall-clock cost from the broken threading, not a correctness defect.

**Verdict: `A1` and `A2` are STILL NOT RUN.** No log loss, gate, or segment-gap number is reported
for either cell from this session; nothing here amends section 11.6's table or Decision 9. Carried
forward, with a concrete fix for next time: either (a) get a working multi-threaded LightGBM wheel
into the image (a different base image or an explicit OpenMP-linked wheel), or (b) restructure the
trainer to parallelize ACROSS refit dates with `joblib.Parallel` (each refit is an independent fit on
an as-of data cut; nothing in the walk-forward loop requires them to run in fit order) -- the latter
would use the 192 vCPUs regardless of LightGBM's own thread scaling and is the more robust fix.

Artifacts: none promoted (both containers' checkpoints held only the auto-run stage-0 entry).
Session record: `docs/ops/aws_launch_chain.md` section 16.


---

## 13. Run R1 -- round 5, the `G2` / `G3` paired closed loop against the served model, 25 seeds (2026-09-18)

Pre-registration section 12 committed **d57d351** BEFORE the engine modes
`round4b_G2` / `round4b_G3` existed and before any `po4b_*` directory existed;
wiring, builder, runner, grader and tests **6215ac8**. Sections 1 through 12
are STATIC and are NOT edited. `experiments.md` is append-only. Evidence,
multi-level: `docs/tests/possession_outcome_closed_loop_2026-09-18.md`.

**VERDICT, STATED FIRST. `G2` IS NOT PUT FORWARD FOR SERVING.** Both halves of
section 12.5's decision rule fail, and neither failure is marginal: condition 1
fails on the G5 dispersion lines at **1.5 to 22.2 measured floors** with a
PASS -> FAIL flip on the margin SD ratio, and condition 2 -- the PM's
responsiveness line -- fails at **5.17 floors** on the headline slope and
**15.95 floors** on the `off_tov_c` driver, under BOTH readings of the line.
`G3`, the comparator, fails condition 1 on the same dispersion lines. **No
default was changed, `docs/models/change_ledger.md` was not edited, and the
ship decision is the PM's.**

### 13.1 The precondition -- the served path, and the one honest caveat

Run before any arm artifact was fitted. A 60-game x 5-seed smoke on the
unchanged default (`results/engine_v0/smoke60x5_po4b_wiring`) reproduces
`results/engine_v0/smoke60x5_default_v5b` **exactly on every simulated value**:
300 game rows and 4,882 player rows compare EQUAL column for column, and all
nine `ENGINE_*` switches match. `scripts/digest_engine_run.py --compare`
nonetheless reports a sha mismatch, and this round reports it rather than
waiving it: the diff is **exactly one field, `meta.adapter_flags`, on exactly
three of its 248 leaves** -- `provisional_clock` True -> False,
`sources.clock.adopted` False -> True, and `sources.clock.note` -- all three
the CLOCK lane's adoption strings from that lane's own UNCOMMITTED edits
present in this shared checkout, none in the possession-outcome path, none a
computed value. **The property the precondition exists to establish holds, and
holds across a week and two other lanes' commits; the literal sha equality
section 12.2 asked for does not, for a reason outside this lane.** Both are on
the record. `tests/test_event_adapter_round4b.py` 10 passed,
`tests/test_engine.py` 20 passed.

The four builder preconditions of 12.1 all passed for both arms: all 16
`TEAM_COLS` identical to `round2/design.parquet` at **max abs diff 0.0** over
10,890 team-games (the arms differ ONLY by the shrinkage); team-block coverage
530/529/1 identical to the reference's; six S1 refit dates and the per-game
segment array identical; per-refit training row counts and `max_train_date`
identical, with the per-GAME leak assertion passing.

### 13.2 The runs

500 games of the standing subset, 25 paired seeds, players kept, 8 workers, the
SERVED stack pinned on all four and asserted against `adapters.py`'s own
defaults at startup. `po4b_R_s25` 412 s, `po4b_G2_s25` 563 s, `po4b_G3_s25`
635 s, `po4b_R_s25_floor` (seeds 1000-1024) 603 s. The clock lane's
`clk5b_B1_s25` was NOT reused as the reference and 12.3 said so before the run;
every floor below is measured here, on this stack, in this round.

### 13.3 Condition 1 -- the gates, priced in floors measured in this round

`scripts/eval_gates.py` blind over all four directories,
`scripts/diag_pair_gate_reports.py` for the A/B/N pairing, both UNEDITED.
`floor = |N - R|`; a line regresses when it moves AWAY from its own target by
more than one floor, or flips PASS -> FAIL.

| gate | line | target | R | floor | `G2` | floors | `G3` | floors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **G5** | **total SD ratio** | 1.000 | **0.8355** | **0.0009** | **0.8155** | **22.2 AWAY** | **0.8156** | **22.1 AWAY** |
| **G5** | **total SD (points)** | 18.96 | **15.8421** | **0.0119** | **15.6209** | **18.6 AWAY** | **15.5804** | **22.0 AWAY** |
| **G5** | **margin SD (points)** | 12.60 | **12.2264** | **0.0506** | **12.0942** | **2.61 AWAY, PASS->FAIL** | **12.0653** | **3.18 AWAY** |
| **G5** | **margin SD ratio** | 1.000 | **0.9695** | **0.0135** | **0.9495** | **1.48 AWAY, PASS->FAIL** | 0.9560 | 1.00 away |
| G5 | home/away corr | 0.2374 | 0.1063 | 0.0115 | 0.0966 | 0.84 away | **0.0925** | **1.20 AWAY** |
| G3 | three_pa_share pooled | 0.3906 | 0.389413 | 0.0000032 | 0.388942 | 147 AWAY (0.05 pp) | 0.389063 | 109 AWAY (0.03 pp) |
| G3 | rim_share pooled | 0.3733 | 0.372052 | 0.000644 | 0.370572 | **2.30 AWAY** | 0.371367 | 1.06 away |
| G4 | efg_pct pooled | 0.5086 | 0.4990 | 0.0006 | 0.4986 | 0.67 away, **PASS->FAIL** | 0.4988 | 0.33 away |
| G9 | total bias | 0 | -0.9624 | 0.3534 | -1.0857 | 0.35 away, **PASS->FAIL** | -0.9992 | 0.10 away |
| G1 | possessions/game SD | 5.191 | 5.609 | 0.057 | 5.539 | 1.23 **toward** | 5.547 | 1.09 **toward** |
| G4 | oreb_pct pooled | 0.2984 | 0.2836 | 0.0002 | 0.2840 | 2.00 **toward** | 0.2840 | 2.00 **toward** |
| G4 | tov_pct pooled | 0.1739 | 0.1762 | 0.0004 | 0.1759 | 0.75 toward | 0.1754 | 2.00 **toward** |
| G8 | rotation minutes SD ratio | 1.000 | 1.2304 | 0.0003 | 1.2296 | 2.67 **toward** | 1.2287 | 5.67 **toward** |
| G9 | margin bias | 0 | +0.1117 | 0.2209 | +0.0194 | 0.42 toward | +0.0219 | 0.41 toward |

Every other scored line moves under one floor. Lines the reference itself
reports NEEDS-INSTRUMENTATION or UNDERPOWERED are reported and NOT scored, per
12.4: G1 by month (0 powered months on 500 games), G2's nine PPP terciles
(2/9 powered, identical in all four runs), the four per-team G3/G4 breakdowns
(0 powered teams), G7's half-share, G8's top-1 FGA share, G9's three
by-breakdown counts and G6's neutral-site cell. **Gate-level statuses are
identical in all four runs** (G1 F, G2 F, G3 NI, G4 F, G5 F, G6 P, G7 F, G8 F,
G9 F): no arm changes an overall verdict, which is exactly why the round is
decided on floors.

**Condition 1 turns on one thing, and it is mechanical.** Both arms compress
between-game dispersion, and the engine's dispersion is already its worst
defect: the total SD ratio is 0.8355 against a target of 1.0 and both arms push
it to 0.816. Shrinking every team's style profile toward a prior makes teams
less different from one another, and a simulator fed less distinct teams
produces less distinct games. **The arm does exactly what it was designed to
do, and that is the wrong direction for the gate the engine is furthest from.**

### 13.4 Condition 2 -- the PM's responsiveness line

998 team-games, bucketed into five quintiles of the offence team's own as-of
driver value taken from the SERVED round-2 block, so all four runs are cut on
identical rows (quintile n 197-200, all powered). No driver is exempt.

| driver | R | floor | `G2` | signed fall | floors | `G3` | signed fall | floors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `off_3pa_c` | 1.0307 | 0.0148 | 1.0475 | -0.0168 | rises | 1.0717 | -0.0410 | rises |
| `off_rim_c` | **0.8881** | **0.0065** | **0.8545** | **+0.0336** | **5.17 FAIL** | 0.9044 | -0.0163 | rises |
| `off_tov_c` | **1.0835** | **0.0115** | **0.9001** | **+0.1834** | **15.95 FAIL** | **0.9428** | **+0.1407** | **12.23 FAIL** |
| **headline** | **0.8881** | **0.0065** | **0.8545** | **+0.0336** | **5.17 FAIL** | 0.9044 | -0.0163 | rises |

**`G2` fails under BOTH readings**, so the pre-registered choice of the signed
fall as the scored reading does not carry the verdict: on `off_rim_c`
`|slope - 1|` worsens 0.1119 -> 0.1455, and on `off_tov_c` 0.0835 -> 0.0999.
The rim quintile detail shows where it goes: the rim-heaviest quintile drops
from 35.21 pp to 34.88 pp against a realised 35.84 pp -- **`G2` flattens the top
of the distribution**, which is section 11.5's point 2 (offline slope 0.9156 vs
the reference's 0.9473) reproducing closed-loop at 5.17 measured floors.

**The one place the two readings disagree is `G3`'s `off_tov_c`**, and it is
reported before the verdict rather than chosen after it: 1.0835 -> 0.9428 is a
12.23-floor signed fall (scored: FAIL) and a 2.3-floor improvement in
`|slope - 1|`. The reference OVER-slopes that driver, which the one-sided
wording of the line does not contemplate. Nothing turns on it -- `G3` is a
comparator and is not put forward under any outcome -- but the PM should note
that the line as written fits a slope below 1.0 and not one above it.

### 13.5 The segments, and what this round could NOT measure

| cell | n team-games | status | `G2` - R points bias | floor-run - R |
|---|---:|---|---:|---:|
| weeks 0-3 | 220 | **UNDERPOWERED** | +0.009 | -0.533 |
| weeks 4-7 | 156 | **UNDERPOWERED** | -0.122 | +0.076 |
| weeks 8+ | 622 | scored | -0.071 | -0.119 |
| conference | 640 | scored | -0.067 | -0.121 |
| non-conference | 358 | scored | -0.052 | -0.284 |
| ALL | 998 | scored | -0.062 | -0.180 |

**Every arm-vs-reference movement in this table is smaller than the seed-offset
movement in the same cell, so the segment table says nothing either way.** And
the two cells round 4 was actually about are UNDERPOWERED at 220 and 156
team-games, exactly as 12.4 predicted in advance. **This round can neither
confirm nor refute the early-season improvement `G2` was selected for.** The
standing 500-game subset is a stride sample of the whole season, not an
early-season sample; a round that wants to price that cell closed-loop needs a
week-stratified subset, and that is a new pre-registration, not a re-read of
this one.

### 13.6 Multi-level evidence

* **Per game.** Paired on (game, seed): the per-game total moves by SD 2.29
  points under `G2` and 2.32 under `G3`, against **4.80** for the seed-offset
  reference run. An arm perturbs an individual game about half as much as a
  seed change does, so the arms' effect is a systematic compression, not a
  re-ordering of which games are high or low.
* **Per team.** 347 teams, **157 UNDERPOWERED at under 3 subset games and
  excluded**. On the 190 powered teams the mean absolute per-team points bias
  moves 5.988 -> **6.254** (`G2`) and **6.223** (`G3`) against a floor of
  0.037: **7.1 and 6.3 floors WORSE**. Against the quantity the arm
  manipulates, `corr(shrinkage, |per-team bias change|)` is +0.107 / +0.164
  against a floor correlation of +0.013 / +0.026 -- the arm reaches the teams it
  is meant to reach, weakly, and what it does when it gets there is not
  systematically in either direction (the by-quintile means are not monotone).
* **Per possession type.** The engine's standing mix errors (too few threes,
  too many two-point jumpers, too few FT trips, too few offensive rebounds) are
  unchanged by either arm except that `G2` pushes the jump-2 share 0.0018
  further from the actual against a 0.0008 floor -- the same direction as the
  pooled rim-share and three-point-share regressions.
* **Player level NOT read**, as 12.4 fixed: no arm touches the player layer and
  both scored G8 lines move toward their targets.

### 13.7 What the round establishes beyond the verdict

The offline selection and the closed loop disagree about `G2`, and the
disagreement is explainable rather than mysterious. `G2` buys an offline
segment by making every team's style profile more like the league's; the
engine's largest standing defect is that its games are already not different
enough from one another. **An early-season shrinkage arm that does not also
preserve between-team dispersion cannot ship into this engine.** That is a
constraint on the next round's arms, not a property of `G2` alone: the
reliability problem round 4 identified is real and unaddressed, but it needs a
form that shrinks a team's ESTIMATE without shrinking the SPREAD of estimates
-- a variance-preserving or hierarchical-with-rescaling shrinkage rather than a
plain posterior mean. `G3`'s partial rescue of two responsiveness slopes is the
first evidence that the two-level form moves in that direction, and `G3`'s
identical dispersion failure is evidence that two levels alone are not enough.

### 13.8 What round 5 did not do

`A1` and `A2` are still NOT RUN and **Decision 9c is still unresolved for this
sub-model** (11.6 unchanged). No fold-1 confirmation exists for any tree
shrinkage arm. Nothing was refit on the bake-off's frame, no round-4 or
round-4b artifact was rewritten, the sealed 2025-26 season was not touched, and
no market line was read. `ENGINE_EVENT=round4b_G2` and `round4b_G3` stay
selectable, default-off, for reproduction. **The served default stays
`round2_s1`; the PM makes the ship decision from the table above and records it
in `docs/models/change_ledger.md`.**
