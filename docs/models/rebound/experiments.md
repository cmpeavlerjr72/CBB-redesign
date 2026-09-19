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

---

## 10. Round 3 AMENDMENT (PM conditions a-e), written 2026-09-18 by the rebound round-3 lane BEFORE any modelling, committed before the first fit

Section 9 is PROPOSED and the PM approves running it **as written, with the five
conditions below**. This section is the amendment those conditions require. It is
APPEND-ONLY: nothing in section 9 is edited, deleted or reinterpreted. Where a
condition is already satisfied by section 9 that is stated and nothing is added;
where it is not, the arm, the metric or the constraint is added here and the
round runs the union of section 9 and section 10.

Nothing below is adopted. No served default changes. `data/processed/models/rebound/`
is not overwritten; this round writes only to `data/processed/models/rebound/round3/`.

### 10.1 Condition (a): the drift arms must be honest walk-forward

**Constraint (binding, added).** No arm may use the test season's league level,
in any form: not its realised OREB%, not a quantity computed from test rows the
arm's own refit has not yet legitimately absorbed, not a coefficient or offset
fitted on test outcomes. Every level an arm carries is either (i) trained from
completed seasons, or (ii) read from the test season's own **expanding as-of**
history strictly before the row's date, which is the same leak discipline
`team_rebound_form` already enforces on the team features. Each added column is
built by the same `expanding_asof` machinery or from train-only seasons, and the
round reports that provenance per arm rather than asserting it.

**Section 9 already covers.** `A2` (recency-weighted training) is condition
(a)(i). `A0` is condition (a)(iii), the served reference.

**Added: what `A1` actually is, and a genuine trend arm.** A gradient-boosted
tree **does not extrapolate**: with `season_idx` in {0,1,2} in the training pool
and a test row at `season_idx = 3`, every split sends the row to the bin the
2024 rows occupy. `A1` is therefore *not* a trend extrapolation -- it is "use the
most recent completed season's level", which is a legitimate and conservative
arm but is not the thing condition (a) asks to be made visible. So:

| arm | what it is | extrapolating? |
|---|---|---|
| `A5` | `A0` + one feature `trend_level_c`: the value at this row's date of an OLS fit of season league OREB% on season index, **fitted on completed training seasons only** and extrapolated forward, expressed as a deviation from the training pool's own mean level | **YES -- flagged** |

`A5` is the trend/`season_idx` extrapolation arm condition (a) names. It is
reported as an extrapolation everywhere it appears, and rule 10.6 below binds it.

**Added: condition (a)(ii), the within-season as-of league anchor.** Section 9's
`A4` is this arm and is hereby restated in the words of the CLAUDE.md rule it
serves: the league's own **expanding as-of** OREB% to the row's date, within the
current season, centred on the training pool's mean league level, entered as a
feature so the model's level is carried by a covariate that is measured inside
the test season rather than by an intercept fitted outside it. This is the
condition-(a)(ii) arm. It reads only rows strictly before the current date and is
the same object `team_rebound_form` already computes as `lg_orebs / lg_opps`.
Restating it here costs nothing and makes the comparison the condition demands
explicit: **`A5` (extrapolation) vs `A2` (recency) vs `A4` (within-season as-of
anchor) vs `A0` (served reference)**, all four on the same table.

**Added: fold 1 is mandatory for every Block-A arm, not only for the winner.**
Section 9.7 rule 6 required fold-1 confirmation only of the fold-2 winner. Every
`A*` arm now runs on **both folds** and both rows appear in the results table, so
a trend arm that wins on fold 2 only because 2025 continued the 2022-2024 trend
is visible as such: fold 1 tests 2024, whose league level (0.29155) also sits
above its training pool, so a genuine drift mechanism must help on both, while a
lucky extrapolation may help on one. An `A5` that wins fold 2 and loses fold 1 is
reported as **"wins by continuation, not by mechanism"** and cannot be
recommended, per 10.6.

### 10.2 Condition (b): the `blocked_f` channel needs a real shot-block sub-model

Section 9's `B3` names a block sub-model as an arm but proposes no model for it
and would defer it. Condition (b) supersedes that: the sub-model is specified
now, as its own small bake-off, in **`docs/models/shot_block/experiments.md`**
(new, written and committed in the same commit as this amendment). Section 9's
`B2` (feed the as-of measured cell rate) survives unchanged as the cheap arm to
beat, and is precisely the "team-defence as-of rate" arm the block bake-off's own
arm list contains -- the two rounds meet there on purpose.

**Where it sits in the cascade, and why that placement is the one that does not
double count.** `fg_make` lists `blocked` in `BANNED_FEATURES` ("a `Block Shot`
row exists only because the attempt missed; it is a post-outcome field, not a
pre-release one") and does not build it. `fg_make` therefore predicts
`P(make | shot context)` **marginally over block status**: blocked attempts are
already inside its miss population at their natural rate. A block draw placed
*before* the make draw would therefore double count -- it would remove shots from
a make model that has already priced them as misses. The honest placement, and
the one this round specifies, is

    shot selection  ->  make / miss  ->  [if miss] BLOCK draw  ->  rebound

i.e. the sub-model's target is **`P(blocked | the attempt missed, context)`**, not
`P(blocked | attempt)`. That conditional is exactly the quantity `loop.py`'s
rebound block needs, is the quantity the rebound model's `blocked_f` column means
at the rows it is used on, and leaves `fg_make` untouched and un-double-counted.
This is recorded here and in `docs/models/shot_block/experiments.md` section 1 so
the placement is a pre-registered decision and not an implementation accident.

**Consequences for section 9's Block B.** `B3` is now a real arm with a real
candidate behind it. `B2` and `B3` are distinguished as:

| arm | what the engine feeds `blocked_f` |
|---|---|
| `B0` | `0.0` (served status quo) |
| `B1` | nothing -- the column is dropped and the model refit |
| `B2` | the **as-of measured** block rate for the (miss type, defence) cell, a continuous value in [0,1] |
| `B3` | a realised **0/1 draw** from the `shot_block` winner, per missed FGA |
| `B3e` | the `shot_block` winner's **predicted probability**, fed continuously (added: it isolates how much of `B3` is the model and how much is the draw's own Jensen gap through a non-linear `lgbm`) |

`B0`-`B3e` are evaluated against the **same** trained rebound model (`A0`'s) except
`B1`, which must be refit. That is deliberate: Block B is an engine-FEED question,
and holding the rebound model fixed is what makes the four feeds comparable.

### 10.3 Condition (c): prior-season carry arms

Section 9's `C1`/`C2`/`C3` already are the prior-season carry ladder and are the
direct analogue of `possession_outcome` round 4's `G1`/`G2`/`G3`. To remove any
ambiguity about what "the same idea that won PO round 4b as G2" means here, the
construction is pinned to PO round 4's, transcribed rather than re-invented:

    w      = D / (D + k)                 D      = as-of denominator mass (live opportunities)
    w_prev = D_prev / (D_prev + k)       D_prev = the prior season's denominator mass

    C3  ("G1"):  w * raw_c
    C1  ("G2"):  w * raw_c + (1 - w) * prior_c
    C2  ("G3"):  w * raw_c + (1 - w) * w_prev * prior_c

with `k = s2 / tau2` the method-of-moments empirical-Bayes weight computed by the
formula in `scripts/train_possession_outcome_v4.fit_k`, on **completed prior
seasons only**, per test season, never on the test season and never tuned. Both
sides get their own `k`: the offence's OREB rate and the defence's allowed-OREB
rate. `prior_c` is the team's completed prior-season rate minus that season's
league rate, shifted forward one season so it is available to the next season
only; a team with no prior season (a new D-I member, a reclassifying school) gets
`prior_c = 0.0` and `D_prev = 0.0`, which collapses `C1` and `C2` to `C3` for
that team rather than fabricating a level. The count of such team-seasons is
reported.

**Added: `C4`, roster-continuity weighting.** The roster-continuity table
(`scripts/build_roster_continuity.py`) is checked for usability FIRST: coverage of
the 2022-2025 team-seasons, and whether its continuity share is knowable strictly
before the season starts. If it is usable, `C4` is `C1` with the prior-season
weight multiplied by the team's returning share `c` in [0,1]:

    C4:  w * raw_c + (1 - w) * c * prior_c

which says a team that returns nobody carries none of its prior season. If the
table is NOT usable (coverage below 90% of team-seasons, or it can only be
computed after the season starts), `C4` is reported as **NOT RUN, with the
measured reason**, and never silently dropped.

### 10.4 Condition (d): Decision 9's mandatory arms, unbundled

Section 9 held the refit calendar fixed and proposed no opponent-adjustment or
conference-flag arm. Decision 9 makes all three mandatory wherever a model
consumes team rates, which this model does (`off_oreb_c`, `opp_def_dreb_c`).
Added as **Block D**, each a separate arm, never bundled into another block's
winner:

| arm | what it is |
|---|---|
| `D0` | reference (= `A0B0C0`) |
| `D1` | `off_oreb_c` / `opp_def_dreb_c` opponent-adjusted by `cbb_sim.features.opponent_adjust`, method `one_pass` |
| `D2` | the same, method `iterative` |
| `D3` | `A0` + a conference-game flag (`cbb_sim.features.conference.build_conference_flags`) as a feature |
| `D4` | refit cadence / conference alignment |

`D4` is **already on record for this exact arm, feature set and folds**: section 8
ran `S0`, `S1_monthly`, `S1_conf_aligned` and `S1_weekly` on `lgbm/C_plus_state`
and adopted `S1_weekly`. Those numbers are **cited, not refit** (the same
treatment section 7.2 gave `S0`'s F1 number), and section 8's open conf4 caveat is
carried forward verbatim. Re-running 23-fit calendars for a question already
answered on the same cells would buy nothing and would cost the hours this round
needs for the arms that have never been run. This citation is declared here,
before the run, so it is a pre-registered economy and not a retrospective
omission.

### 10.5 Condition (e): folds, grader, floor, evidence -- and the two-stage budget

Folds, the seal, the single blind grader, the "ties to the simpler model" rule and
the multi-level evidence list are sections 9.3, 9.4, 9.5, 9.6 and 9.7 and are
unchanged. Added or made explicit:

- **Segments.** Section 9.5's list already carries per-miss-type, per-month,
  per-quintile (including the Nov-Dec / Jan / Feb-Apr split), site and conference
  cuts. Added to it: **per-game** level error (mean, median, SD, MAE over
  test-season games) so the evidence is multi-level in the CLAUDE.md sense rather
  than pooled-plus-team. Minimum cell n stays 300 and every cell below it is
  printed with the label `UNDERPOWERED` and excluded from every pass/fail.
- **Noise floor.** Section 9.6 stands. Operationally: a spec-identical second-seed
  retrain of the reference **and** of each block's leading arm at the stage each
  is decided at, and -- per this model's own standing convention, "the floor the
  decision rule uses is the larger" (section 8.2) -- the floor is the larger of
  that spread and the already-published 5-seed SD for this arm (6.7e-05,
  section 3). The block-bootstrap SE for this cell (0.001412, section 3) is
  reported alongside as the wider, game-clustered reading.
- **Two-stage budget (added, and binding).** The served calendar is `S1_weekly` =
  23 LightGBM refits per cell per fold at a measured ~190 s each, i.e. ~1.2 h per
  cell. The arm list above is 20+ cells on two folds; run entirely on the served
  calendar it is a 40-hour job on a machine four other workers are using. The
  round therefore runs in two pre-declared stages:

  **Stage 1 -- screen.** Every arm, both folds, on the **`S0` static calendar**
  (one fit per cell), with `A0B0C0 / S0` as the stage-1 reference. `S0` is not a
  contrivance: it is section 8's own reference cell and its F2 log loss
  (0.645565) is the number round 1 adopted, so stage 1's reference is a published,
  reproducible quantity. Stage 1 ranks arms; it decides nothing.

  **Stage 2 -- confirm.** The leading arm of each block, the combined arm, and the
  reference are re-run on the **served `S1_weekly` calendar on fold 2**, and **the
  decision in 9.7 is taken on stage-2 numbers only.** Any arm stage 2 does not
  reach is reported `NOT RUN`, never as a result. Stage 2 is where `A4`'s
  within-season anchor and `A1`'s "latest season" behaviour can actually differ
  from stage 1, because only under a refit calendar does the training pool contain
  test-season rows at all -- and that difference is itself a reported finding, not
  a nuisance.

  A stage-1 winner that stage 2 does not confirm is **not** a winner.

### 10.6 Additions to the decision rule (9.7 otherwise unchanged)

9. An arm that carries an **extrapolated** level (`A5`, and `A1` only if the
   fitted model is shown to extrapolate) must win on **both** folds to be
   recommended. Winning fold 2 while losing fold 1 is reported as "wins by
   continuation, not by mechanism" and is recommended against, whatever its
   fold-2 margin.
10. A Block-D arm (Decision 9's mandatory arms) is reported on its own row and is
    **never** folded into the combined arm unless it independently clears rule 1
    at stage 2. Decision 9 stays PENDING EVIDENCE unless a `D*` arm beats the
    reference beyond the floor here AND on a second sub-model.
11. The `shot_block` sub-model is decided by **its own** pre-registration
    (`docs/models/shot_block/experiments.md`), on its own primary metric. Its
    winner enters this round only as the `B3` / `B3e` feed. A `shot_block` arm
    that wins its own bake-off but whose feed does not clear rule 1 here is
    reported as such and neither is adopted.
12. The round **adopts nothing and changes no default**. It produces a table,
    `docs/tests/rebound_round_drift_block_carry_2026-09-18.md`, and a
    recommendation. The PM decides.

### 10.7 Artifacts, scripts and paths

    scripts/build_rebound_round3_design_v1.py   design cache: the round-3 arm columns
    scripts/train_rebound_v3_round3.py          the rebound ladder, stages 1 and 2
    scripts/train_shot_block_v1.py              the shot-block bake-off
    data/processed/models/rebound/round3/       all rebound round-3 artifacts
    data/processed/models/shot_block/           all shot-block artifacts

Artifact directories over 20 MB are gitignored and HF-synced per CLAUDE.md. The
status change `PROPOSED -> RUN` goes to `docs/models/change_ledger.md` in the same
commit as the results.

---

## 11. Round 3 RESULTS, stage 1 (run 2026-09-18 19:06-20:15 ET, `scripts/train_rebound_v3_round3.py` + `scripts/grade_rebound_round3_v1.py`)

Status: sections 9 and 10 EXECUTED at **stage 1 only**. Stage 2 (the served
`S1_weekly` calendar, 23 refits per cell) and the paired closed loop are
**NOT RUN** -- measured cost 472-631 s per single fit on a machine four other
workers were using, so one stage-2 cell is ~3 h and the session had a hard stop.
Per section 10.5 the round's DECISION is therefore NOT TAKEN: what follows is the
screen. **Nothing is adopted, no served default changed.** Narrative, the Block-B
refutation, the shot-block interaction and the recommendation:
`docs/tests/rebound_round_drift_block_carry_2026-09-18.md`.

Reproduction check: `A0B0C0` on F2 reproduces round 1's adopted S0 log loss
**0.645565** exactly, and reproduces the G4 diagnostic's served level readings
`L1 = -1.137 pp` and `L2 = -1.877 pp` to 4 dp. The design builder's rebuild of
`off_oreb_c` and `opp_def_dreb_c` matches the served columns at max abs diff
**0.0**.

`C4` (roster-continuity weighting): **NOT RUN**, for the measured reason 10.3
required be checked first -- `roster_continuity_2027.parquet` covers season 2027
only, coverage 0/3 of the test seasons this round needs.

`D4` (refit cadence): cited from section 8 as declared in 10.4, not refit.


**Stage 1 / F2 / `S0` (SELECTION)**

| arm | n_feat | n_fits | log_loss | brier | L1_pp | L2_B0_pp | L2_B3_pp | calib | gap_pp | resp | slope_q | mono | fit_s | why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A5+C1 | 16 | 1 | 0.644658 | 0.418606 | -0.4444 | -1.1966 | -0.4422 | PASS | 1.674 | PASS | 1.1219 | 4 | 472.2 | combined: A5 + C1 |
| A5+C3 | 16 | 1 | 0.644767 | 0.41868 | -0.3394 | -1.088 | -0.3381 | PASS | 1.67 | PASS | 0.717 | 4 | 471.3 | combined: A5 + C3 |
| A5 | 16 | 1 | 0.645024 | 0.418919 | -0.337 | -1.0887 | -0.3356 | PASS | 1.921 | PASS | 0.6895 | 4 | 295.6 | TREND EXTRAPOLATION: an OLS season-level offset fitted on the fo |
| C1 | 16 | 1 | 0.645182 | 0.41887 | -1.2439 | -1.9847 | -1.2431 | FAIL | 2.281 | PASS | 1.1072 | 4 | 531.5 | prior-season carry, PO round 4's G2 |
| C3 | 16 | 1 | 0.645196 | 0.418884 | -1.1633 | -1.904 | -1.1622 | PASS | 1.966 | PASS | 0.7138 | 4 | 315.4 | shrink toward the league mean only, G1 (the control) |
| C2 | 16 | 1 | 0.645218 | 0.418878 | -1.2592 | -2.0015 | -1.2584 | FAIL | 2.228 | PASS | 1.0814 | 4 | 504.9 | prior-season carry with the prior shrunk by its own reliability, |
| D3 | 17 | 1 | 0.645315 | 0.419058 | -1.1421 | -1.8841 | -1.1413 | FAIL | 2.147 | PASS | 0.6947 | 4 | 308.7 | Decision 9b: conference-game flag |
| D2 | 16 | 1 | 0.64551 | 0.419295 | -1.1309 | -1.8691 | -1.1289 | PASS | 1.974 | PASS | 0.6298 | 4 | 557.4 | Decision 9a: opponent adjustment, iterative |
| A0B0C0 | 16 | 1 | 0.645565 | 0.419188 | -1.137 | -1.8769 | -1.137 | PASS | 1.958 | PASS | 0.6851 | 4 | 631.4 | reference: the served lgbm / C_plus_state, unchanged |
| D1 | 16 | 1 | 0.645647 | 0.419245 | -1.1272 | -1.8671 | -1.1257 | PASS | 1.944 | PASS | 0.6294 | 4 | 590.8 | Decision 9a: opponent adjustment, one_pass |
| A1 | 17 | 1 | 0.645816 | 0.419003 | -0.8288 | -1.5806 | -0.8279 | PASS | 1.676 | PASS | 0.6728 | 4 | 529.2 | season index as a feature |
| A3_2s | 16 | 1 | 0.64618 | 0.419357 | -1.0167 | -1.7721 | -1.0167 | PASS | 1.984 | PASS | 0.6875 | 4 | 505.4 | rolling window: most recent 2 seasons of training rows |
| A2_h240 | 16 | 1 | 0.646239 | 0.419318 | -0.9962 | -1.7522 | -0.995 | FAIL | 2.181 | PASS | 0.6948 | 4 | 308.1 | exponential recency weights, 240-day half-life |
| B1 | 15 | 1 | 0.647099 | 0.420419 | -1.1573 | -1.1573 | -1.1573 | FAIL | 2.03 | PASS | 0.6867 | 4 | 503.4 | drop blocked_f and refit; marginalise over blocks |
| A2_h120 | 16 | 1 | 0.64768 | 0.419753 | -0.8878 | -1.6525 | -0.8857 | FAIL | 2.34 | PASS | 0.6984 | 4 | 519.4 | exponential recency weights, 120-day half-life |
| A2_h60 | 16 | 1 | 0.649226 | 0.420301 | -0.9445 | -1.7018 | -0.9412 | FAIL | 2.674 | PASS | 0.7179 | 4 | 507.4 | exponential recency weights, 60-day half-life |
| A4 | 17 | 1 | 0.649412 | 0.420191 | 0.3059 | -0.4405 | 0.3072 | FAIL | 2.585 | PASS | 0.397 | 4 | 531.5 | within-season as-of league OREB% anchor (PM condition a-ii) |
| A3_1s | 16 | 1 | 0.649568 | 0.420358 | -0.8344 | -1.591 | -0.8307 | FAIL | 2.851 | PASS | 0.6812 | 4 | 541.7 | rolling window: most recent 1 season of training rows |

**Stage 1 / F1 / `S0`**

| arm | n_feat | n_fits | log_loss | brier | L1_pp | L2_B0_pp | L2_B3_pp | calib | gap_pp | resp | slope_q | mono | fit_s | why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A5+C1 | 16 | 1 | 0.619846 | 0.407335 | -0.138 | -0.9017 | -0.143 | FAIL | 2.212 | PASS | 1.0652 | 4 | 456.4 | combined: A5 + C1 |
| A5+C3 | 16 | 1 | 0.619971 | 0.407472 | -0.0143 | -0.7739 | -0.0197 | FAIL | 2.196 | PASS | 0.664 | 4 | 252.9 | combined: A5 + C3 |
| C2 | 16 | 1 | 0.620077 | 0.40741 | -0.706 | -1.4626 | -0.7106 | PASS | 1.68 | PASS | 1.0114 | 4 | 273.8 | prior-season carry with the prior shrunk by its own reliability, |
| A5 | 16 | 1 | 0.620096 | 0.407541 | -0.0752 | -0.8407 | -0.0804 | PASS | 1.9 | PASS | 0.6124 | 4 | 301.7 | TREND EXTRAPOLATION: an OLS season-level offset fitted on the fo |
| C1 | 16 | 1 | 0.620181 | 0.407457 | -0.7204 | -1.4791 | -0.7246 | PASS | 1.761 | PASS | 1.0588 | 4 | 256.0 | prior-season carry, PO round 4's G2 |
| C3 | 16 | 1 | 0.62021 | 0.407515 | -0.6388 | -1.3934 | -0.6434 | PASS | 1.817 | PASS | 0.6456 | 4 | 289.5 | shrink toward the league mean only, G1 (the control) |
| A1 | 17 | 1 | 0.620267 | 0.407562 | -0.5179 | -1.2707 | -0.5226 | PASS | 1.568 | PASS | 0.6069 | 4 | 210.4 | season index as a feature |
| D3 | 17 | 1 | 0.620313 | 0.4076 | -0.6428 | -1.3975 | -0.6475 | PASS | 1.527 | PASS | 0.6195 | 4 | 287.3 | Decision 9b: conference-game flag |
| A0B0C0 | 16 | 1 | 0.620446 | 0.407653 | -0.6494 | -1.4088 | -0.6544 | PASS | 1.629 | PASS | 0.6093 | 4 | 335.9 | reference: the served lgbm / C_plus_state, unchanged |
| A3_2s | 16 | 1 | 0.620446 | 0.407653 | -0.6494 | -1.4088 | -0.6544 | PASS | 1.629 | PASS | 0.6093 | 4 | 255.0 | rolling window: most recent 2 seasons of training rows |
| D2 | 16 | 1 | 0.620645 | 0.407862 | -0.6355 | -1.391 | -0.6398 | PASS | 1.666 | PASS | 0.5835 | 4 | 286.0 | Decision 9a: opponent adjustment, iterative |
| D1 | 16 | 1 | 0.620661 | 0.407891 | -0.6479 | -1.4024 | -0.6511 | FAIL | 2.008 | PASS | 0.5757 | 4 | 346.3 | Decision 9a: opponent adjustment, one_pass |
| A2_h240 | 16 | 1 | 0.62087 | 0.407876 | -0.6332 | -1.4031 | -0.6395 | PASS | 1.764 | PASS | 0.6103 | 4 | 294.8 | exponential recency weights, 240-day half-life |
| A2_h120 | 16 | 1 | 0.621306 | 0.408194 | -0.6653 | -1.4456 | -0.6721 | PASS | 1.999 | PASS | 0.6131 | 4 | 276.5 | exponential recency weights, 120-day half-life |
| A2_h60 | 16 | 1 | 0.62195 | 0.408576 | -0.6634 | -1.4611 | -0.6706 | FAIL | 2.501 | PASS | 0.6079 | 4 | 260.7 | exponential recency weights, 60-day half-life |
| A4 | 17 | 1 | 0.62198 | 0.408835 | -0.7927 | -1.5705 | -0.7982 | PASS | 1.939 | PASS | 0.3134 | 4 | 274.8 | within-season as-of league OREB% anchor (PM condition a-ii) |
| A3_1s | 16 | 1 | 0.622116 | 0.408602 | -0.6175 | -1.4067 | -0.6258 | FAIL | 2.39 | PASS | 0.6067 | 4 | 232.4 | rolling window: most recent 1 season of training rows |
| B1 | 15 | 1 | 0.622119 | 0.409127 | -0.667 | -0.667 | -0.667 | PASS | 1.901 | PASS | 0.6131 | 4 | 206.8 | drop blocked_f and refit; marginalise over blocks |

### Noise floor

- stage 1, `A0B0C0` second-seed retrain on F2: log-loss spread **0.000004**, L1 spread 0.0393 pp. Operative floor = max(spread, published 5-seed SD 6.7e-05) = **0.000067**.
- game-clustered block-bootstrap SE for this cell (published, section 3): 0.001412.

### Decision rule applied mechanically, stage 1 (floor 0.000067)

| arm | F2 gain | x floor | F1 gain | rule1 (beats ref) | rule3 (gates) | rule4 (L1 not worse) | L1_pp | rule5 (slope not reduced) | slope_q |
|---|---|---|---|---|---|---|---|---|---|
| A5+C1 | 0.000907 | 13.5 | 0.0006 | True | True | True | -0.4444 | True | 1.1219 |
| A5+C3 | 0.000798 | 11.9 | 0.000475 | True | True | True | -0.3394 | False | 0.717 |
| A5 | 0.000541 | 8.1 | 0.00035 | True | True | True | -0.337 | False | 0.6895 |
| C1 | 0.000383 | 5.7 | 0.000265 | True | False | False | -1.2439 | True | 1.1072 |
| C3 | 0.000369 | 5.5 | 0.000236 | True | True | False | -1.1633 | False | 0.7138 |
| C2 | 0.000347 | 5.2 | 0.000369 | True | False | False | -1.2592 | True | 1.0814 |
| D3 | 0.00025 | 3.7 | 0.000133 | True | False | False | -1.1421 | False | 0.6947 |
| D2 | 5.5e-05 | 0.8 | -0.000199 | False | True | True | -1.1309 | False | 0.6298 |
| D1 | -8.2e-05 | -1.2 | -0.000215 | False | True | True | -1.1272 | False | 0.6294 |
| A1 | -0.000251 | -3.7 | 0.000179 | False | True | True | -0.8288 | False | 0.6728 |
| A3_2s | -0.000615 | -9.2 | 0.0 | False | True | True | -1.0167 | False | 0.6875 |
| A2_h240 | -0.000674 | -10.1 | -0.000424 | False | False | True | -0.9962 | True | 0.6948 |
| B1 | -0.001534 | -22.9 | -0.001673 | False | False | False | -1.1573 | False | 0.6867 |
| A2_h120 | -0.002115 | -31.6 | -0.00086 | False | False | True | -0.8878 | True | 0.6984 |
| A2_h60 | -0.003661 | -54.6 | -0.001504 | False | False | True | -0.9445 | True | 0.7179 |
| A4 | -0.003847 | -57.4 | -0.001534 | False | False | True | 0.3059 | False | 0.397 |
| A3_1s | -0.004003 | -59.7 | -0.00167 | False | False | True | -0.8344 | False | 0.6812 |

Eligible: ['A5+C1']. **Stage-1 leader by the rule: `A5+C1`** (ties inside one floor broken by simplicity).

### Multi-level evidence (stage-2 cells if present, else stage 1)


by miss type (level pp)

| arm | rim | jump2 | three | ft |
|---|---|---|---|---|
| A5+C1 | 0.3268 | -0.6047 | -0.7892 | -0.4932 |
| A5+C3 | 0.4305 | -0.5026 | -0.6783 | -0.4057 |
| A5 | 0.4192 | -0.4986 | -0.6667 | -0.4133 |
| C1 | -0.6159 | -1.4019 | -1.58 | -0.9398 |
| C3 | -0.5574 | -1.3279 | -1.4818 | -0.8641 |
| C2 | -0.6271 | -1.4164 | -1.6071 | -0.9115 |
| D3 | -0.5386 | -1.3113 | -1.4555 | -0.8477 |
| D2 | -0.5229 | -1.3212 | -1.4336 | -0.8419 |
| A0B0C0 | -0.5114 | -1.3169 | -1.4487 | -0.8835 |
| D1 | -0.5153 | -1.3095 | -1.428 | -0.8805 |
| A1 | -0.2426 | -0.9855 | -1.1249 | -0.6046 |
| A3_2s | -0.4434 | -1.1727 | -1.3061 | -0.7904 |
| A2_h240 | -0.4456 | -1.1589 | -1.2853 | -0.6896 |
| B1 | -0.6001 | -1.2907 | -1.4594 | -0.8877 |
| A2_h120 | -0.3568 | -1.0617 | -1.1752 | -0.5024 |
| A2_h60 | -0.3873 | -1.1268 | -1.2268 | -0.6342 |
| A4 | 0.1082 | -0.0737 | 0.5383 | 0.7972 |
| A3_1s | -0.3577 | -1.0393 | -1.1002 | -0.3143 |

by month (level pp)

| arm | 11 | 12 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| A5+C1 | -0.8577 | -0.6042 | -0.9641 | 0.1966 | 0.1959 | 1.242 |
| A5+C3 | -0.6152 | -0.532 | -0.8671 | 0.2319 | 0.2626 | 1.2284 |
| A5 | -0.6117 | -0.5381 | -0.8702 | 0.2637 | 0.2227 | 1.6678 |
| C1 | -1.6623 | -1.4162 | -1.7763 | -0.6001 | -0.5639 | 0.5141 |
| C3 | -1.422 | -1.3981 | -1.6903 | -0.5805 | -0.5656 | 0.6847 |
| C2 | -1.6874 | -1.3957 | -1.7802 | -0.6201 | -0.6119 | 0.1781 |
| D3 | -1.1614 | -1.0127 | -1.869 | -0.7456 | -0.6677 | 1.0753 |
| D2 | -1.3912 | -1.3331 | -1.6853 | -0.5344 | -0.5468 | 0.8974 |
| A0B0C0 | -1.427 | -1.3096 | -1.6932 | -0.5451 | -0.5285 | 0.8139 |
| D1 | -1.3836 | -1.3782 | -1.6666 | -0.5129 | -0.5391 | 0.4346 |
| A1 | -1.0448 | -1.0296 | -1.3725 | -0.2623 | -0.2782 | 0.975 |
| A3_2s | -1.3293 | -1.2558 | -1.5168 | -0.413 | -0.421 | 1.094 |
| A2_h240 | -1.2978 | -1.2255 | -1.5274 | -0.3755 | -0.4006 | 1.0478 |
| B1 | -1.4364 | -1.3231 | -1.7071 | -0.581 | -0.5602 | 0.8668 |
| A2_h120 | -1.2024 | -1.1216 | -1.4107 | -0.2694 | -0.2714 | 0.9037 |
| A2_h60 | -1.2115 | -1.2067 | -1.4809 | -0.3327 | -0.3344 | 0.9069 |
| A4 | -0.098 | 0.3876 | 0.2646 | 1.0153 | -0.3359 | 1.1713 |
| A3_1s | -1.1694 | -1.1202 | -1.3006 | -0.216 | -0.2367 | 1.2982 |

by site (level pp)

| arm | home | away | neutral |
|---|---|---|---|
| A5+C1 | -0.5505 | -0.2838 | -0.6434 |
| A5+C3 | -0.4288 | -0.1999 | -0.5216 |
| A5 | -0.4282 | -0.1996 | -0.5059 |
| C1 | -1.37 | -1.0683 | -1.428 |
| C3 | -1.2502 | -1.0274 | -1.3416 |
| C2 | -1.3683 | -1.0971 | -1.4532 |
| D3 | -1.2399 | -1.0176 | -1.2444 |
| D2 | -1.2176 | -1.0051 | -1.2749 |
| A0B0C0 | -1.2422 | -0.9905 | -1.2905 |
| D1 | -1.2269 | -0.9796 | -1.3023 |
| A1 | -1.029 | -0.5993 | -0.9515 |
| A3_2s | -1.1833 | -0.8177 | -1.1462 |
| A2_h240 | -1.1867 | -0.7725 | -1.1314 |
| B1 | -1.2687 | -1.0151 | -1.2755 |
| A2_h120 | -1.1521 | -0.6016 | -0.9914 |
| A2_h60 | -1.2501 | -0.6026 | -1.1024 |
| A4 | 0.4788 | 0.4209 | -0.6674 |
| A3_1s | -1.0785 | -0.547 | -1.0096 |

by conference game (level pp)

| arm | conf | nonconf |
|---|---|---|
| A5+C1 | -0.2938 | -0.7032 |
| A5+C3 | -0.2343 | -0.52 |
| A5 | -0.2241 | -0.5309 |
| C1 | -1.0942 | -1.501 |
| C3 | -1.0548 | -1.3498 |
| C2 | -1.1087 | -1.5177 |
| D3 | -1.2234 | -1.0022 |
| D2 | -1.0295 | -1.3051 |
| A0B0C0 | -1.0287 | -1.3233 |
| D1 | -1.0105 | -1.3277 |
| A1 | -0.7345 | -0.9909 |
| A3_2s | -0.8843 | -1.2442 |
| A2_h240 | -0.8725 | -1.2089 |
| B1 | -1.058 | -1.3279 |
| A2_h120 | -0.7633 | -1.1016 |
| A2_h60 | -0.8222 | -1.1548 |
| A4 | 0.4429 | 0.0705 |
| A3_1s | -0.6915 | -1.08 |

by period (level pp)

| arm | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| A5+C1 | -0.3149 | -0.579 | -0.1256 | -1.474 | UNDERPOWERED |
| A5+C3 | -0.2072 | -0.4765 | 0.0244 | -1.6542 | UNDERPOWERED |
| A5 | -0.218 | -0.4631 | 0.1832 | -1.5625 | UNDERPOWERED |
| C1 | -1.0868 | -1.4061 | -0.9505 | -2.3768 | UNDERPOWERED |
| C3 | -1.0069 | -1.3256 | -0.7792 | -2.3987 | UNDERPOWERED |
| C2 | -1.124 | -1.3988 | -0.9525 | -2.4985 | UNDERPOWERED |
| D3 | -1.0076 | -1.2843 | -0.6481 | -2.2109 | UNDERPOWERED |
| D2 | -0.961 | -1.3066 | -0.8178 | -2.1348 | UNDERPOWERED |
| A0B0C0 | -0.9825 | -1.2996 | -0.6119 | -2.3653 | UNDERPOWERED |
| D1 | -0.9783 | -1.2819 | -0.769 | -2.2117 | UNDERPOWERED |
| A1 | -0.7116 | -0.9553 | -0.2125 | -1.8515 | UNDERPOWERED |
| A3_2s | -0.8747 | -1.1647 | -0.6748 | -1.9737 | UNDERPOWERED |
| A2_h240 | -0.8478 | -1.1446 | -1.0908 | -1.981 | UNDERPOWERED |
| B1 | -1.0105 | -1.3115 | -0.7147 | -2.1638 | UNDERPOWERED |
| A2_h120 | -0.7723 | -0.999 | -1.212 | -1.9355 | UNDERPOWERED |
| A2_h60 | -0.8344 | -1.0462 | -1.3894 | -2.7861 | UNDERPOWERED |
| A4 | 0.6578 | -0.0643 | 1.1554 | -0.4948 | UNDERPOWERED |
| A3_1s | -0.7328 | -0.9318 | -1.1227 | -1.902 | UNDERPOWERED |

per-team prior-quintile slope ratio, by season segment (the G4 responsiveness defect; served engine 0.561 / 0.799 / 0.738)

| arm | all | Nov-Dec | Jan | Feb-Apr | gap_pp_by_q |
|---|---|---|---|---|---|
| A5+C1 | 1.1219 | 1.1973 | 1.1821 | 1.0078 | [-0.744, -0.807, -0.25, -0.336, 0.055] |
| A5+C3 | 0.717 | 0.5186 | 0.8506 | 0.8529 | [0.725, -0.031, -0.253, -0.799, -1.131] |
| A5 | 0.6895 | 0.5905 | 0.797 | 0.7295 | [0.902, -0.028, -0.286, -0.898, -1.134] |
| C1 | 1.1072 | 1.1712 | 1.1736 | 1.0015 | [-1.472, -1.6, -1.089, -1.139, -0.769] |
| C3 | 0.7138 | 0.5176 | 0.8356 | 0.8517 | [-0.105, -0.861, -1.067, -1.599, -1.981] |
| C2 | 1.0814 | 1.1251 | 1.1566 | 0.9949 | [-1.412, -1.522, -1.094, -1.237, -0.879] |
| D3 | 0.6947 | 0.5473 | 0.842 | 0.7648 | [0.044, -0.854, -1.08, -1.654, -1.957] |
| D2 | 0.6298 | 0.5024 | 0.7475 | 0.6942 | [0.295, -0.869, -1.021, -1.671, -2.132] |
| A0B0C0 | 0.6851 | 0.5946 | 0.7792 | 0.7231 | [0.114, -0.823, -1.073, -1.717, -1.951] |
| D1 | 0.6294 | 0.5165 | 0.7635 | 0.669 | [0.278, -0.826, -0.979, -1.691, -2.151] |
| A1 | 0.6728 | 0.5741 | 0.7841 | 0.71 | [0.47, -0.501, -0.787, -1.412, -1.675] |
| A3_2s | 0.6875 | 0.5846 | 0.7962 | 0.7273 | [0.212, -0.694, -0.972, -1.559, -1.837] |
| A2_h240 | 0.6948 | 0.5969 | 0.8015 | 0.7343 | [0.214, -0.622, -1.017, -1.544, -1.787] |
| B1 | 0.6867 | 0.5865 | 0.7806 | 0.7353 | [0.095, -0.858, -1.126, -1.725, -1.959] |
| A2_h120 | 0.6984 | 0.6025 | 0.7958 | 0.7431 | [0.308, -0.533, -0.883, -1.441, -1.669] |
| A2_h60 | 0.7179 | 0.6199 | 0.829 | 0.7543 | [0.202, -0.617, -0.923, -1.51, -1.648] |
| A4 | 0.397 | 0.3966 | 0.4622 | 0.3731 | [2.576, 0.832, 0.506, -0.465, -1.377] |
| A3_1s | 0.6812 | 0.59 | 0.7706 | 0.7223 | [0.42, -0.423, -0.89, -1.377, -1.67] |

per-game level error (pp) and the first four weeks of conference play

| arm | n_games | mean | median | sd | mae | P(pred<act) | conf4_n | conf4_level_pp |
|---|---|---|---|---|---|---|---|---|
| A5+C1 | 5593 | -0.3197 | -0.1063 | 6.0405 | 4.8614 | 0.5074 | 82748 | -1.0447 |
| A5+C3 | 5593 | -0.2092 | -0.0432 | 6.0618 | 4.8755 | 0.5038 | 82748 | -0.9666 |
| A5 | 5593 | -0.2066 | 0.0534 | 6.1228 | 4.9151 | 0.4969 | 82748 | -0.9745 |
| C1 | 5593 | -1.1177 | -0.9722 | 6.0329 | 4.9152 | 0.5594 | 82748 | -1.8496 |
| C3 | 5593 | -1.0324 | -0.8863 | 6.0687 | 4.9365 | 0.5561 | 82748 | -1.7896 |
| C2 | 5593 | -1.1318 | -0.9731 | 6.0299 | 4.9103 | 0.5632 | 82748 | -1.8477 |
| D3 | 5593 | -1.0152 | -0.8798 | 6.1047 | 4.9533 | 0.5518 | 82748 | -1.8908 |
| D2 | 5593 | -0.9934 | -0.7964 | 6.1367 | 4.9824 | 0.5507 | 82748 | -1.8194 |
| A0B0C0 | 5593 | -1.0057 | -0.7879 | 6.127 | 4.9645 | 0.5476 | 82748 | -1.7769 |
| D1 | 5593 | -0.9865 | -0.7954 | 6.1368 | 4.9694 | 0.5541 | 82748 | -1.8075 |
| A1 | 5593 | -0.6969 | -0.4589 | 6.104 | 4.9217 | 0.5308 | 82748 | -1.4568 |
| A3_2s | 5593 | -0.8858 | -0.6599 | 6.1362 | 4.956 | 0.5448 | 82748 | -1.6306 |
| A2_h240 | 5593 | -0.8648 | -0.6535 | 6.125 | 4.9487 | 0.5434 | 82748 | -1.6225 |
| B1 | 5593 | -1.0242 | -0.8247 | 6.1665 | 5.0012 | 0.5502 | 82748 | -1.7829 |
| A2_h120 | 5593 | -0.7574 | -0.4925 | 6.1444 | 4.9551 | 0.5314 | 82748 | -1.5355 |
| A2_h60 | 5593 | -0.8144 | -0.6343 | 6.177 | 4.9826 | 0.5357 | 82748 | -1.6008 |
| A4 | 5593 | 0.4554 | 0.6558 | 6.4259 | 5.1861 | 0.4645 | 82748 | 0.1494 |
| A3_1s | 5593 | -0.7032 | -0.4923 | 6.2195 | 5.0111 | 0.5287 | 82748 | -1.4352 |

### Block B: the `blocked_f` engine feed (all feeds, same trained model)


stage 1, `A0B0C0` on F2 (actual live OREB 0.299232)

| feed | level_pp | log_loss |
|---|---|---|
| B0_zero | -1.8769 | 0.647713 |
| B2_asof_cell | 7.0912 | 0.664252 |
| B3e_model | 8.3292 | 0.667026 |
| B3_draw | -1.137 | 0.648339 |
| B_true | -1.137 | 0.645565 |
