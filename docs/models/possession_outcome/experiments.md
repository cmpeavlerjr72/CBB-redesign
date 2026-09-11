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
