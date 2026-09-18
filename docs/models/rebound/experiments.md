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

---

## 7. S1 scheme confirmation: refit cadence (2026-09-10, written and COMMITTED before this round's modelling)

Authority: `ARCHITECTURE_DECISIONS.md` Decision 9c ("the in-season refit scheme's cadence and alignment are
bake-off dimensions... the other sub-models' S1 confirmation passes inherit its winner[' ladder, not its
verdict]"); `docs/models/README.md` "Standing result" (S1-monthly is the DEFAULT unless a sub-model's own
bake-off says otherwise); `docs/LEARNINGS.md` L21. The possession-outcome round-3 pre-registration
(`docs/models/possession_outcome/experiments.md` section 6) is in flight on the same question for its own
model and is NOT reused here as a verdict -- only its refit-calendar CODE PATH is reused (below), because
one definition of "conference-aligned" and "weekly" must mean the same thing in every sub-model.

Model class and feature set are HELD FIXED at round 1's winner, **`lgbm` / `C_plus_state`** (section 6).
Nothing about the arm, the event layer, the universe, or the folds is reopened here -- only the CALENDAR
on which it refits.

### 7.1 The four schemes

| scheme | refit calendar | object count (F2, 2025) | complexity rank |
|---|---|---|---|
| `S0` | one static fit on the fold's whole train slice, as round 1 scored it. **The reference; must reproduce round 1's recorded F2 log loss (0.645565) exactly** | 1 | 0 |
| `S1_monthly` | `cbb_sim.models.possession_outcome.month_boundaries` on the test season's game dates -- the L21 default | 6 | 1 |
| `S1_conf_aligned` | `cbb_sim.features.conference.union_boundaries(monthly, conference_boundary_dates(...))` -- the monthly dates UNION every distinct date on which at least one team plays its first regular-season conference game (refit once per distinct date, never once per team) | 29 | 2 |
| `S1_weekly` | `cbb_sim.features.conference.union_boundaries(weekly_boundaries(...), monthly[:1])` -- every Monday on or before the first test-season game through the last, plus the season's first monthly date so the opening partial week is covered | 24 | 3 |

Every S1 scheme keeps the standing S1 contract exactly (L21, `possession_outcome.fit_predict_scheme`): each
refit uses games STRICTLY BEFORE its own date (all prior seasons plus the test season to date), and each
test game is scored by the most recent refit at or before its own date, so no game is ever in its own fit.
Only the calendar changes, and the refit-calendar CONSTRUCTION CODE is imported unchanged from
`scripts/train_possession_outcome_v3.py` (`refit_dates`, which itself calls
`cbb_sim.features.conference.weekly_boundaries` / `conference_boundary_dates` / `union_boundaries` and
`possession_outcome.month_boundaries`) rather than reimplemented, so "weekly" and "conference-aligned" are
the same objects in this model as in the one that is deciding Decision 9. Object counts above were MEASURED
on 2026-09-10 on both fold-2 (2025) and fold-1 (2024) test seasons before any fit ran: monthly is 6 in both
seasons; the conference-aligned union is 29 in both; the weekly union is 24 in both.

### 7.2 Folds

F2 (train 2022+2023+2024, test 2025) is the SELECTION fold, scored for all four schemes. F1 (train
2022+2023, test 2024) is REPORTED for `S0` and `S1_monthly` only: `S0`'s F1 number is READ from section 3's
existing F1 row for `lgbm`/`C_plus_state` (log loss 0.620446) rather than refit, because that row is already
the identical model class, feature set and fold code path -- refitting it would only reproduce a number
already on record, at the cost of one more ~150s tree fit that stage 7.4's budget needs elsewhere.
`S1_monthly` F1 IS refit (it is the standing default and the cheapest S1 arm). `S1_conf_aligned` and
`S1_weekly` are NOT run on F1: Decision 9's own diagnostic (`possession_outcome/experiments.md` section 6,
"Stage A found `weeks_since_refit` to be the flattest axis it measured") makes the alignment dimension the
least likely of the three to pay, so per-fold evidence on it is the first thing this round's budget sheds,
by construction and stated in advance rather than discovered when the clock runs out.

### 7.3 Metrics

Unchanged via `RB.score()` -- log loss, Brier, `calib_pass`/`calib_worst_gap_pp` (2 pp gate, classes with
>= 5% share), `resp_pass`/`resp_min_steps` (this model's own 4-of-4 reading), per-miss-type gap. Two
additions, both computed OUTSIDE `score()` so its call graph stays exactly what round 1 used:

1. **`conf4_gap_pp`**: the worst gated decile-calibration gap (round 1's own gate: classes with >= 5% share,
   `PM.decile_calibration` + `PM.calibration_verdict`) restricted to opportunities in the OFFENCE team's own
   first four weeks of conference play -- `(game_date - first_conf_date).days / 7 in [0, 4)`, where
   `first_conf_date` comes from `cbb_sim.features.conference.first_conference_game_dates`, joined on
   `(season, off_team_id)`. This is Decision 9's predicted damage segment, read the same way
   `train_possession_outcome_v3.conf_window_calibration` reads it for possession outcome, so the two models'
   numbers are comparable. A segment with fewer than **1,000** opportunities is reported `underpowered: true`
   and never as a pass or a gap (round 1's 5% share rule is about CLASSES inside one gate; this is a row-count
   floor on the SEGMENT itself, sized down from possession-outcome's own 20,000 in proportion to this
   model's roughly 5x smaller F2 test population, and it is a judgement call recorded here as one, not a
   fitted quantity).
2. **Decision 8 slope**, reported per scheme from `RB.score()`'s own `responsiveness` block: `slope_ratio`
   for both drivers (`off_oreb_c->OREB`, `opp_def_dreb_c->OREB`), read against the band `[0.8, 1.2]` and this
   model's already-adopted 4-of-4 step count. `ARCHITECTURE_DECISIONS.md` Decision 8 records the round-1
   winner's slopes as 0.96/1.10, both inside the band with a realised quintile span well above the 2 pp
   exemption threshold on both drivers, so no driver here is exempt.

### 7.4 Noise floor and wall-clock budget

**Noise floor.** `S0` refit under a second seed (`seed=1`) on F2, spec-identical. Primary-metric floor =
`|log_loss(seed=0) - log_loss(seed=1)|`. The `conf4_gap_pp` seed spread is ALSO measured and reported for
context, but the decision rule below uses a FIXED **0.25 pp** threshold for that quantity, carried over
unchanged from the possession-outcome round-3 pre-registration's own convention for this exact segment
metric (`experiments.md` section 6.4) rather than invented fresh, so the two models apply one standard to
one segment definition.

**Wall clock.** Measured 2026-09-10 at the 3-thread cap (`OMP_NUM_THREADS`/`n_jobs` pinned to 3; eight other
workers share this machine): one `lgbm`/`C_plus_state` fit on the F2 train slice costs **189.2 s** and
reproduces the adopted F2 log loss to six decimal places (0.645565), which is this section's own
reproduction check, already satisfied before the rest of the grid runs. On that cost the full stage list
(S0 F2 + seed1 + `S1_monthly` F2 + F1 + `S1_conf_aligned` F2 (29 fits) + `S1_weekly` F2 (24 fits), with
later-season fits costing somewhat more than a single-shot fit because the train slice grows by the
accumulated test-season prefix) is projected at **~3.9 h**. Hard wall clock: **4.5 h**, checked before each
cell starts (a cell that starts, finishes). Drop order if the budget is tight, most complex first:
`S1_weekly` F2 before `S1_conf_aligned` F2 (weekly has the higher object count); anything not reached is
written to the results as NOT RUN and never as a result.

### 7.5 Decision rule

`S0` is the reference. A candidate scheme (`S1_monthly`, `S1_conf_aligned`, `S1_weekly`, in the complexity
order of section 7.1) **beats the reference** if, on F2, EITHER its log loss improves on `S0`'s by more than
the primary-metric floor, OR its `conf4_gap_pp` improves on `S0`'s by more than 0.25 pp -- while continuing
to pass `RB.score()`'s calibration and responsiveness gates unchanged. Where more than one scheme beats the
reference, the SIMPLEST whose log loss is within the floor of the best beater's wins (round 1's own
simplicity tie-break, transcribed unchanged). **If nothing beats the reference, `S0` stands and cadence
remains PENDING EVIDENCE for this sub-model** -- an explicitly legitimate outcome (Decision 9: "if it loses
or ties, the raw-centred rate stays and this entry is amended to say so"; the same standard applies to
cadence here).

### 7.6 Artifacts

Trainer: `scripts/train_rebound_v2_s1.py`. Every scheme's dated joblib artifacts and a
`manifest.py`-format manifest (JSON: `refit_date`, `path`, `max_train_date` required per entry) are written
to `data/processed/models/rebound/s1_confirm/<scheme>/<fold>/` (gitignored; HF-synced, never `git add`ed).
`train_rebound_v1.py`'s own artifacts (`events_v1.parquet`, round-1 `run_report.json`) are read back, never
rebuilt or overwritten.

<!-- ROUND-2 (S1 SCHEME CONFIRMATION) RESULTS APPENDED BELOW BY scripts/train_rebound_v2_s1.py -->

## 8. S1 scheme confirmation: results (run 2026-09-11T03:42:00.075924+00:00, `scripts/train_rebound_v2_s1.py`)

S0/F2 reproduction of round 1's adopted number: CONFIRMED (cited 0.645565).

| cell | n_fits | log_loss | calib | worst_gap_pp | respons | slope_off_oreb_c | slope_opp_def_dreb_c | conf4_n | conf4_gap_pp | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|
| F2|S0|s0 | 1 | 0.645565 | PASS | 1.958 | PASS | 0.9645 | 1.1008 | 82748 | 2.462 | 200.4 |
| F2|S0|s1 | 1 | 0.645569 | FAIL | 2.008 | PASS | 0.9663 | 1.1123 | 82748 | 2.917 | 134.6 |
| F2|S1_monthly|s0 | 6 | 0.644949 | PASS | 1.693 | PASS | 0.965 | 1.0843 | 82748 | 2.66 | 833.7 |
| F1|S1_monthly|s0 | 6 | 0.619726 | PASS | 1.326 | PASS | 0.9353 | 1.1075 | 87263 | 1.73 | 524.1 |
| F2|S1_conf_aligned|s0 | 29 | 0.644878 | PASS | 1.826 | PASS | 0.967 | 1.0886 | 82748 | 2.367 | 2271.1 |
| F2|S1_weekly|s0 | 23 | 0.644522 | PASS | 1.674 | PASS | 0.9668 | 1.0988 | 82748 | 2.565 | 1699.5 |

S0 F1 (cited, not refit, section 3): log loss 0.620446, calib gap 1.629 pp (PASS), slopes 0.9138 / 1.087.

Noise floor: primary-metric floor (S0 second-seed refit) = **4e-06**. conf4 seed spread (context only; the decision rule uses the fixed 0.25 pp threshold) = 0.455 pp.

### 8.1 Decision

- `S1_monthly`: log loss 0.644949 (gain +0.000616 vs S0), conf4 gain -0.198 pp, gates PASS, beats reference: True
- `S1_conf_aligned`: log loss 0.644878 (gain +0.000687 vs S0), conf4 gain 0.095 pp, gates PASS, beats reference: True
- `S1_weekly`: log loss 0.644522 (gain +0.001043 vs S0), conf4 gain -0.103 pp, gates PASS, beats reference: True

**WINNER: S1_weekly** -- beats the S0 reference beyond the floor; simplest arm within the floor of the best beater

### 8.2 Floor correction and a conf4 caveat (addendum, written immediately after the run above, before any doc other than this one was touched)

**Floor correction.** Section 7.4 pre-registered the primary-metric floor as `|log_loss(seed=0) -
log_loss(seed=1)|` on a fresh second-seed refit of `S0`. That gap came back at **4e-06** -- far
smaller than round 1's OWN already-published 5-seed noise floor for this identical arm (SD 6.7e-05
over seeds 0-4, `log_loss` values `[0.645565, 0.645569, 0.645425, 0.645447, 0.645522]`, section 3),
because seeds 0 and 1 happen to sit unusually close together in that set. Per the standing
convention this project already uses everywhere a floor is measured two ways (round 1's own rule,
"the floor the decision rule uses is the larger"), the OPERATIVE floor here is revised to
**6.7e-05**, the larger and already-on-record number, rather than the lucky 2-seed gap. This is
reported as a correction, not a re-registration: applying "take the larger of the measured floors"
to numbers already in hand is the standing rule, not a new one chosen after seeing the result.

Re-checked against the corrected floor: `S1_monthly` gain 0.000616 = 9.2x; `S1_conf_aligned` gain
0.000687 = 10.3x; `S1_weekly` gain 0.001043 = 15.6x. All three still clear the corrected floor by a
wide margin, and `S1_weekly` is still the only scheme within the (corrected) floor of itself as the
best beater. **The decision is unchanged: `S1_weekly`.**

**A conf4 caveat, reported rather than smoothed over.** Unlike the free-throw S1 confirmation, no
scheme here -- including the winner -- passes the 2.00 pp conf4 gate on F2: `S0` 2.462 pp,
`S1_monthly` 2.660 pp, `S1_conf_aligned` 2.367 pp (the best of the four), `S1_weekly` 2.565 pp. The
winning scheme's `conf4_gain_pp` vs `S0` is **-0.103 pp (a marginal worsening, not an improvement)**;
it is selected on the primary metric alone (log loss), which the pre-registered OR rule permits.
`S1_conf_aligned` has the best conf4 reading of the four candidates but does not win because its log
loss is outside the corrected floor of `S1_weekly`'s. So for THIS sub-model, Decision 9's predicted
damage segment (first four weeks of conference play) is not resolved by any refit calendar tested;
it is carried forward as an open item rather than reported as fixed.

---

## 9. Round 3 pre-registration: the two level defects and the team-slope compression behind gate G4's OREB% miss (PROPOSED, written 2026-09-18 by the G4 diagnostic lane BEFORE any modelling; NOT RUN, NOT ADOPTED, no served default changed)

Evidence this round is written against: `docs/tests/g4_oreb_fta_diagnostic_2026-09-18.md`.
That document decomposes the engine's -1.561 pp pooled OREB% miss on fold 2 into
four channels that close exactly (residual +0.0000 pp), of which **three belong to
this sub-model or to how the engine feeds it**:

| channel | pp | share |
|---|---:|---:|
| grading source (box -> pbp event layer) | +0.082 | -5.3% |
| **fold-2 calibration of the served `S1_weekly` arm** | **-1.137** | **+72.9%** |
| **engine feeds `blocked_f = 0` (no shot-block model exists)** | **-0.740** | **+47.4%** |
| sim state / mix distribution (mix +0.055, state +0.070, MC +0.074, subset +0.035) | +0.234 | -15.0% |

and, on top of those levels, a **responsiveness** defect: bucketed by 2024
prior-season OREB%, the engine's 2025 team span is 0.0426 against the actual's
0.0631 (**slope ratio 0.676**, monotone 4/4), with the gap running -0.42 pp in Q1
to -2.47 pp in Q5. By season segment the slope ratio is **0.561 (Nov-Dec) / 0.799
(Jan) / 0.738 (Feb-Apr)** -- worst while the as-of form features are still pinned
at their league-mean start value of 0.0.

Ruled out in that document and therefore **not** arms here: the rotation (this is a
team-level model and `loop.py` passes it no slot block), the miss-type mix
(+0.055 pp, wrong sign), dead-ball / team-rebound bookkeeping (DEAD cancels in
`oreb/(oreb+dreb)`; hoopR `total_rebounds - (oreb+dreb) = 0` on all 11,400 2025
team-games), OREB-chain truncation (9 events in 428,250 game-sims), and the
grading source (+0.096 pp, wrong sign).

### 9.1 Candidates

Three blocks, tested as a ladder against the served reference so each block's
marginal value is readable, plus one engine-side arm that this model does not own
but must be evaluated jointly because it changes what the model is asked at
serve time.

**Block A -- season drift / level** (the 73% channel). The served arm pools
2022-to-date with no season term and no recency weight; its training pool sits at
0.28704-0.29015 against a 2025 level of 0.29923, and its own mean prediction
(0.28786) tracks the pool, not the test season.

| arm | what it is |
|---|---|
| `A0` | **reference**: the served `lgbm / C_plus_state / S1_weekly`, unchanged |
| `A1` | `A0` + `season_idx` as a feature (already computed by `build_design`, in no feature set) |
| `A2` | `A0` + exponential recency sample weights over game date, half-life in {60, 120, 240} days, grid pre-declared |
| `A3` | `A0` trained on a rolling window of the most recent N team-games only, N in {1 season, 2 seasons}, pre-declared |
| `A4` | `A0` + a league-level as-of OREB% offset feature (the league's own expanding mean to date, centred), so the level is carried by a covariate rather than by the fit's intercept |

**Block B -- the `blocked_f` feed** (the 47% channel). The feature is in the
served bundle, the model uses it correctly (on blocked rows it predicts 0.4225 /
0.3995 / 0.3997 against actuals 0.4237 / 0.4113 / 0.4274), and `loop.py` hard-sets
it to 0.0 on every opportunity because the cascade has no block model. Blocked
misses are 26.2% of rim, 8.2% of jump2 and 1.4% of three opportunities.

| arm | what it is |
|---|---|
| `B0` | **reference**: `blocked_f` in the bundle, engine feeds 0.0 (the status quo) |
| `B1` | drop `blocked_f` from the bundle entirely and refit, so the model marginalises over the block rate instead of being told a false value |
| `B2` | keep `blocked_f`, and have the engine feed the **as-of measured block rate** for the (shot class, defence) cell rather than 0.0 -- a continuous value in [0,1], which is what the trained column's conditional expectation means |
| `B3` | keep `blocked_f`, and have the engine **draw** a block indicator per missed FGA from a block sub-model, then feed the realised 0/1 |

`B3` requires a block sub-model that does not exist. It is carried as an arm so
the round reports what it would be worth rather than deferring it silently, and
if it wins, the deliverable is a pre-registration for that sub-model, not a ship.
`B2` is the cheapest arm that is not a false statement to the model, and is the
one to beat.

**Block C -- prior-season carry for the as-of form features** (the slope
compression). `off_oreb_c` and `opp_def_dreb_c` are expanding within-season means
that begin at exactly 0.0 for every team. This is the same defect
`possession_outcome` round 4 found for its own style rates and round 4b resolved
in favour of `G2`; it has never been run against this model.

| arm | what it is |
|---|---|
| `C0` | **reference**: as-of expanding mean, league-centred, 0.0 with no prior games |
| `C1` | empirical-Bayes shrinkage of the as-of rate toward the team's **prior-season** value, with the shrinkage weight fitted on the training folds (the direct analogue of `possession_outcome`'s `G2`) |
| `C2` | `C1` with the prior-season target itself shrunk by its own reliability (the analogue of `G3`) |
| `C3` | shrink toward the league mean with a fitted pseudo-count instead of toward the prior season (the simpler control that separates "any shrinkage" from "prior-season information") |

Ladder: the round runs `A0B0C0` (reference), each block alone against it, then the
single best arm of each block combined. Blocks are not crossed exhaustively.

### 9.2 Features

Base bundle is the served `C_plus_state` (`rebound.feature_set`), unchanged in
name and order. Additions are declared per arm above and nowhere else. Every
rating feature stays expressed relative to its own snapshot's league mean;
`site_home` / `site_away` stay in every arm (CLAUDE.md: home/away/neutral is a
first-class feature). No arm may add a feature that is not listed in 9.1, and
`docs/models/rebound/features.md` is updated in the same commit as any arm that
wins.

### 9.3 Folds

Fold 1 trains {2022, 2023} and tests 2024. Fold 2 trains {2022, 2023, 2024} and
tests 2025. **Fold 2 selects.** Season 2026 is SEALED and `fold_slices` calls
`assert_not_sealed` on both slices. The S1 refit schedule is part of each arm's
spec and is held at the served `S1_weekly` for every arm, so this round is not
also a scheme bake-off. Arms `A2`/`A3` change what the schedule *trains on*, never
when it refits.

### 9.4 Primary metric

**Three-class log loss on fold 2**, the same primary this model's rounds 1 and 2
used, so the ladder is comparable with them.

Two pre-declared **level** readings, reported next to the primary and binding
through the decision rule in 9.7 rather than replacing it:

- `L1` **fold-2 level error**: mean predicted P(OREB | live) minus the realised
  rate on the fold-2 live rows, with the TRUE `blocked_f`. Served value **-1.137
  pp**; target |L1| <= 0.25 pp.
- `L2` **engine-feed level error**: the same quantity with `blocked_f` fed as the
  engine feeds it under that arm. Served value **-1.877 pp** (= -1.137 - 0.740);
  target |L2| <= 0.35 pp.

### 9.5 Segment breakdowns (every arm, every fold)

Reported for every arm whether or not it wins, with underpowered cells labelled
and never folded into a pass or a fail (min cell n = 300 opportunities):

1. by miss type (rim / jump2 / three / ft) -- level and log loss;
2. by `blocked` (true value), within each miss type;
3. by **month** of the test season (Nov, Dec, Jan, Feb, Mar; Apr is expected
   underpowered) -- this is where the season-drift arms must show their work;
4. by **2024 prior-season OREB% quintile of the offence** -- slope ratio,
   monotone steps, and the per-quintile level gap (served: 0.676, 4/4, -0.42 to
   -2.47 pp);
5. the same quintile cut **split Nov-Dec / Jan / Feb-Apr** (served: 0.561 / 0.799
   / 0.738) -- block C's own target;
6. home / away / neutral (served level gap -1.51 / -1.56 / -1.73 pp);
7. conference vs non-conference game (served -1.42 / -1.81 pp), and the first
   four weeks of conference play, which section 8 left open for this model;
8. by period (1, 2, OT) and by five-minute game-minute bucket (served: flat
   -1.21 to -2.43 pp, which is the reading any winning arm must flatten further
   rather than tilt).

### 9.6 Noise floor

A **spec-identical retrain under a second seed** for the reference and for each
block's leading arm, on fold 2, including the identical refit calendar. The floor
is the observed log-loss spread; a winner must beat the reference by more than it.
For the level readings `L1`/`L2` the floor is the same retrain's level spread.
Minimum two seeds; more if the first two disagree by more than the smallest
claimed margin. The fold-2 live-row Monte-Carlo floor on any realised sim OREB%
quoted in this round is 0.074 pp (one binomial SE on 368,826 opportunities) and
is not to be confused with the offline floor.

### 9.7 Decision rule

1. An arm is eligible only if it beats `A0B0C0` on the **primary** (fold-2 log
   loss) by more than the measured floor.
2. Among eligible arms, the winner is the one with the **lowest fold-2 log loss**;
   ties inside one floor go to the **simpler** arm, with simplicity ordered
   `A0 < A1 < A4 < A2 < A3`, `B0 < B1 < B2 < B3`, `C0 < C3 < C1 < C2`, and fewer
   blocks beating more.
3. A winner must additionally **pass both existing gates** (calibration: worst
   decile gap <= 2 pp on classes with a >= 5% share; responsiveness: predicted
   OREB share by quintile of the offence's as-of OREB% monotone 4 of 4, and the
   same for the defence's as-of DREB%) -- the round-1 gates, unchanged.
4. A winner must **not worsen** `L1` or `L2`, and the round reports whether it
   meets their targets. An arm that improves log loss while leaving |L1| above
   0.25 pp is reported as a partial result, not a fix for G4.
5. A winner must **not reduce** the 2024-prior-quintile slope ratio in any of the
   three season segments of breakdown 5.
6. **Fold 1 confirmation is required** before any ship recommendation: the fold-2
   winner is refitted on fold 1 and must not reverse sign on the primary.
7. **Nothing ships on offline evidence.** An offline winner ships only after a
   paired-seed closed-loop sim run at >= 25 seeds on the served stack shows no
   gate regressed, per Decision 10, and the closed loop must move the engine's
   pooled OREB% toward 0.29841 without moving G2's PPP terciles, G4's TOV% or
   eFG%, or G1 beyond their own measured bands.
8. Ties at every level go to the simpler model. If no arm clears rule 1, **no arm
   is adopted** and the round says so.

### 9.8 What this round may not do

No post-hoc multiplier, cap, clip, offset, calibration curve or blend on the
model's output or on sim output (`docs/SIM_GUARDRAILS.md`, CLAUDE.md). An arm that
adds an intercept correction fitted on the TEST season is banned outright and is
not on the list above; `A1`-`A4` all change what the model is trained on or told,
never what its output is multiplied by afterwards. The 2025-26 season stays
sealed. `data/processed/models/rebound/` is not overwritten: this round writes to
a versioned sibling `round3/` and the PM switches the manifest.
