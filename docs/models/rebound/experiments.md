# L3 REBOUND: experiments

Append-only. Status changes go to `docs/models/change_ledger.md` in the same commit
(`CLAUDE.md`, standing rule "bake-off before any choice").

---

## 1. Pre-registration (authored by the PM, 2026-09-10, BEFORE any modelling)

Target: after a missed FGA or a missed last free throw, does the offense secure the rebound (OREB) vs the defense (DREB), with dead-ball rebounds treated as a third class (they end the possession without a live rebound; report their share and whether they need their own model or are absorbed as a deterministic share by miss type). Universe: D-I, non-truncated, seasons 2022-2025; F1 train {2022, 2023} test 2024; F2 train {2022, 2023, 2024} test 2025 (selection); 2026 sealed. Feature sets: A_team (offense as-of OREB% and defense as-of DREB% allowed from first-chance-safe sources, own ratings, site), B_plus_miss (A + miss type: rim / jump2 / 3 / FT, blocked flag), C_plus_state (B + period, seconds remaining, score diff, bonus), D_plus_lineup (C + for 2024+ only, the on-floor five's as-of individual OREB/DREB rates aggregated; evaluated on the 2024-2025 subset with its own fold: train 2024, test 2025). Model classes: logistic ridge; LightGBM; and a league-share baseline by miss type. Metrics: log loss and Brier on F2; calibration by decile (<= 2 pp worst gap); responsiveness (predicted OREB rate by offense as-of OREB% quintile must slope with actual, monotone in 4 of 4 steps; same for defense quintile); by-miss-type calibration; noise floor per round-1 conventions. Decision rules: as the L3 round-1 rules (lowest log loss among arms passing calibration and responsiveness; tree must beat linear by more than the floor; ties to the simpler). Report separately whether D_plus_lineup beats C on the 2025 subset by more than the floor, because that decides whether rebounding is a lineup-level sub-model in the engine or a team-level one.

---

<!-- RESULTS APPENDED BELOW BY scripts/train_rebound_v1.py -->

## 2. Grid configuration (as executed, run 2026-09-10 14:00, `scripts/train_rebound_v1.py --version v1`)

| Dimension | Values |
|---|---|
| Target | rebound outcome of an opportunity, 3 classes: OREB, DREB, DEAD |
| Opportunity | a missed FGA, or the missed LAST free throw of a non-technical trip |
| Feature sets | A_team, B_plus_miss, C_plus_state on F1/F2; `D_plus_lineup` on its own fold (section 5) |
| Model classes | `baseline` (league shares BY MISS TYPE, most recent training season), `ridge_logit` (multinomial), `lgbm` (LightGBM multiclass) |
| Folds | F1: train 2022+2023, test 2024. F2: train 2022+2023+2024, test 2025 (selection). L2: train 2024, test 2025 (lineup bundle only) |
| Sealed | 2026 -- `assert_not_sealed` is called on every train and test slice |
| Primary metric | three-class log loss on F2 |
| Possessions version | `v1`, rim-location override 0.0 ft |
| Noise floor | linear: 200-replicate game-level block bootstrap SE; tree: SD over 5 seed-varied refits |

Design: 1,549,406 rebound opportunities over [2022, 2023, 2024, 2025], of which 1,536,570 are modelled and 12,836 (0.83%) are `unresolved` -- the rebound row never appears in the feed, so the row is dropped with its count reported rather than imputed. Runtime 11.9 min. CSV alongside: `data/processed/models/rebound/grid_results.csv`.

### 2.1 Class shares by season

| season | n_opportunities | OREB_pct | DREB_pct | DEAD_pct | unresolved_pct | on_floor_complete_pct |
|---|---|---|---|---|---|---|
| 2022 | 371215 | 28.028 | 71.211 | 0.761 | 0.837 | 0.0 |
| 2023 | 385902 | 28.457 | 70.756 | 0.788 | 0.812 | 0.0 |
| 2024 | 396439 | 28.972 | 70.398 | 0.63 | 0.829 | 92.09 |
| 2025 | 395850 | 29.603 | 69.327 | 1.07 | 0.837 | 98.42 |

### 2.2 Class shares by miss type (pooled 2022-2025)

| miss_type | n | share_of_opportunities_pct | OREB_pct | DREB_pct | DEAD_pct | unresolved_pct |
|---|---|---|---|---|---|---|
| rim | 376248 | 24.283 | 37.704 | 61.781 | 0.515 | 1.035 |
| jump2 | 403705 | 26.055 | 28.043 | 71.228 | 0.729 | 0.782 |
| three | 636280 | 41.066 | 27.354 | 71.612 | 1.033 | 0.747 |
| ft | 133173 | 8.595 | 12.665 | 86.48 | 0.854 | 0.774 |

---

## 3. Full results

**F1** -- train [2022, 2023], test [2024]

| arm | feature_set | n | log_loss | brier | calib | worst_gap_pp | worst_class | level_pp | shape_pp | respons | resp_min_steps | brier_OREB | brier_DREB | brier_DEAD | slope_off_oreb_c | slope_opp_def_dreb_c | missgap_rim | missgap_jump2 | missgap_three | missgap_ft | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 393154 | 0.620446 | 0.407653 | PASS | 1.629 | OREB | 0.673 | 1.932 | PASS | 4 | 0.19945 | 0.202002 | 0.006201 | 0.9138 | 1.087 | 0.609 | 0.795 | 0.717 | 0.294 | 31.8 |
| ridge_logit | C_plus_state | 393154 | 0.621124 | 0.407558 | FAIL | 2.061 | OREB | 0.662 | 1.716 | PASS | 4 | 0.199336 | 0.201968 | 0.006254 | 0.8974 | 1.0103 | 0.546 | 0.663 | 0.7 | 0.82 | 3.2 |
| ridge_logit | B_plus_miss | 393154 | 0.622465 | 0.408376 | FAIL | 2.109 | OREB | 0.651 | 1.569 | PASS | 4 | 0.199718 | 0.202398 | 0.006261 | 0.8928 | 1.0057 | 0.513 | 0.646 | 0.707 | 0.804 | 5.3 |
| lgbm | B_plus_miss | 393154 | 0.623947 | 0.408927 | FAIL | 2.203 | OREB | 0.682 | 1.853 | PASS | 4 | 0.199981 | 0.202672 | 0.006274 | 0.9065 | 1.0906 | 0.59 | 0.859 | 0.694 | 0.417 | 24.7 |
| baseline | A_team | 393154 | 0.626652 | 0.411776 | PASS | 0.768 | OREB | 0.518 | 0.25 | FAIL | 2 | 0.201398 | 0.204117 | 0.006261 | -0.0056 | -0.0183 | 0.466 | 0.39 | 0.449 | 0.768 | 1.4 |
| ridge_logit | A_team | 393154 | 0.635493 | 0.418281 | PASS | 1.749 | OREB | 0.737 | 1.012 | PASS | 4 | 0.204721 | 0.207297 | 0.006263 | 0.89 | 0.9942 | 9.679 | 0.236 | 0.751 | 15.747 | 6.5 |
| lgbm | A_team | 393154 | 0.637455 | 0.41931 | FAIL | 3.059 | DREB | 0.795 | 2.681 | PASS | 4 | 0.205221 | 0.207818 | 0.006271 | 0.9245 | 1.0649 | 9.743 | 0.103 | 0.683 | 15.634 | 21.3 |

**F2** -- train [2022, 2023, 2024], test [2025]  (SELECTION)

| arm | feature_set | n | log_loss | brier | calib | worst_gap_pp | worst_class | level_pp | shape_pp | respons | resp_min_steps | brier_OREB | brier_DREB | brier_DEAD | slope_off_oreb_c | slope_opp_def_dreb_c | missgap_rim | missgap_jump2 | missgap_three | missgap_ft | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | C_plus_state | 392538 | 0.645565 | 0.419188 | PASS | 1.958 | DREB | 1.375 | 1.904 | PASS | 4 | 0.202306 | 0.206503 | 0.010378 | 0.9645 | 1.1008 | 0.65 | 1.544 | 1.744 | 1.143 | 61.2 |
| ridge_logit | C_plus_state | 392538 | 0.647533 | 0.419771 | FAIL | 2.773 | DREB | 1.456 | 2.518 | PASS | 4 | 0.202428 | 0.206772 | 0.010572 | 0.9146 | 1.0647 | 0.618 | 1.616 | 1.844 | 1.467 | 6.9 |
| ridge_logit | B_plus_miss | 392538 | 0.649036 | 0.420674 | FAIL | 2.656 | DREB | 1.436 | 2.45 | PASS | 4 | 0.202841 | 0.20725 | 0.010584 | 0.9126 | 1.0605 | 0.58 | 1.625 | 1.801 | 1.539 | 5.4 |
| lgbm | B_plus_miss | 392538 | 0.650303 | 0.420725 | FAIL | 2.452 | DREB | 1.408 | 1.831 | PASS | 4 | 0.202854 | 0.207274 | 0.010598 | 0.9521 | 1.0819 | 0.681 | 1.594 | 1.74 | 1.311 | 29.0 |
| baseline | A_team | 392538 | 0.653708 | 0.423909 | PASS | 1.405 | DREB | 1.02 | 0.668 | FAIL | 1 | 0.204405 | 0.208905 | 0.010598 | -0.0104 | -0.0008 | 0.352 | 1.302 | 1.467 | 1.104 | 0.5 |
| ridge_logit | A_team | 392538 | 0.660909 | 0.429592 | FAIL | 2.279 | DREB | 1.45 | 0.829 | PASS | 4 | 0.207354 | 0.211646 | 0.010592 | 0.9084 | 1.0384 | 9.515 | 0.83 | 0.604 | 15.11 | 2.8 |
| lgbm | A_team | 392538 | 0.662835 | 0.430095 | FAIL | 2.853 | DREB | 1.475 | 2.13 | PASS | 4 | 0.207585 | 0.211904 | 0.010606 | 0.9597 | 1.0793 | 9.52 | 0.836 | 0.678 | 15.164 | 52.4 |

Noise floor: game-block bootstrap SE 0.001412 on `ridge_logit/C_plus_state`; LightGBM seed-refit SD 6.7e-05 over seeds [0, 1, 2, 3, 4] ([0.645565, 0.645569, 0.645425, 0.645447, 0.645522]). The floor the decision rule uses is the larger, **0.001412**.

Oracle (unattainable) baseline on F2 -- the test season's OWN shares by miss type -- log loss 0.652246. The gap between it and the honest baseline is the season-drift cost L11 measures, reported so it is visible rather than hidden.

---

## 4. Dead-ball rebounds: third class, or a fixed share?

Dead balls are 0.515-1.033% of opportunities by miss type (train shares {'rim': 0.00456, 'jump2': 0.00643, 'three': 0.00923, 'ft': 0.00799}, F2 test shares {'rim': 0.00685, 'jump2': 0.01002, 'three': 0.01343, 'ft': 0.01013}).

Full three-class `ridge_logit`/`C_plus_state`: log loss 0.647533. A LIVE-only binary model of the same shape, composed with a deterministic dead-ball share by miss type taken from the training fold: 0.64822. Difference 0.000686 = 0.49x the noise floor.

---

## 5. Lineup vs team: does `D_plus_lineup` beat `C_plus_state`?

Fold L2 (train 2024, test 2025). 748,398 of the fold's opportunities carry all ten on-floor ids (95.25%); C and D are scored on exactly those rows, so the comparison is like-for-like. Train 362,094, test 386,304.

Individual-rate shrinkage strength FITTED on the train season over [0, 50, 100, 200, 400] pseudo-opportunities (L13: never assumed): **50**. Train log loss per rung: [(0, 0.618419), (50, 0.617945), (100, 0.617955), (200, 0.618028), (400, 0.618132)].

| arm | feature_set | n | log_loss | brier | calib | worst_gap_pp | respons | resp_min_steps | fit_s |
|---|---|---|---|---|---|---|---|---|---|
| ridge_logit | C_plus_state | 386304 | 0.647557 | 0.419437 | FAIL | 2.567 | PASS | 4 | 3.1 |
| ridge_logit | D_plus_lineup | 386304 | 0.646184 | 0.418243 | FAIL | 2.183 | PASS | 4 | 3.4 |
| lgbm | C_plus_state | 386304 | 0.650214 | 0.420464 | FAIL | 3.242 | PASS | 4 | 43.0 |
| lgbm | D_plus_lineup | 386304 | 0.648487 | 0.419172 | FAIL | 3.178 | PASS | 4 | 43.2 |

Block-bootstrap SE on this fold: 0.001557. `ridge_logit` D - C gain 0.001373 (0.88x floor); `lgbm` D - C gain 0.001727 (1.11x floor). **Verdict: TEAM-LEVEL.**

---

## 6. Decision, by the pre-registered rule

1 of 7 F2 arms pass BOTH the calibration gate (worst decile gap <= 2.00 pp on classes with a >= 5% share) and the responsiveness gate (4 of 4 monotone quintile steps on BOTH the offence as-of OREB% and the defence as-of DREB% drivers).

Passing: `lgbm/C_plus_state`.

Best linear arm of any gate status: `ridge_logit/C_plus_state` at 0.647533. Best tree arm: `lgbm/C_plus_state` at 0.645565. Difference 0.001968 = 1.39x the noise floor.

no LINEAR arm passes both gates, so the pre-registered 'a tree arm must beat the best linear arm by more than the floor' clause has no passing linear arm to bind against. The gap to the best linear arm OF ANY gate status is reported above (0.001968 = 1.39x the floor) so the PM can apply the stricter reading if they prefer it.

**WINNER: lgbm/C_plus_state**, F2 log loss 0.645565.
