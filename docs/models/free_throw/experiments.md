# L3 FREE THROW: experiments

Append-only. Status changes go to `docs/models/change_ledger.md` in the same commit
(`CLAUDE.md`, standing rule "bake-off before any choice").

---

## 1. Pre-registration (authored by the PM, 2026-09-10, BEFORE any modelling)

Two sub-models. (FT-1) Trip structure: given a foul event class (shooting foul on a 2, on a 3, and-one, bonus one-and-one, double bonus, technical), the number of attempts is deterministic by rule except the one-and-one, whose second attempt depends on the first make; verify the rule-derived counts against the data per season and report violations (the 2024-25 rule change to the bonus structure is a known era boundary: document what the data shows about bonus trips before and after, and put the era flag in GameState, not in the model). (FT-2) Make probability per attempt: keyed on the shooter (L15: player identity dominates). Universe and folds as above, but the shooter-level model is fit on player_box/pbp participants (ESPN athlete id via the crosswalk in src/cbb_sim/data/player_ids.py if present, else CBBD ids) for 2022-2025. Arms: (a) team as-of FT% (the team-level floor), (b) empirical-Bayes shrinkage of the shooter's as-of FT% toward a prior with the prior and the shrinkage strength FITTED (grid over the prior: league mean; position mean; the player's prior-season FT% for returning players), (c) a logistic ridge on shooter as-of FT% + attempts-to-date + prior-season FT% + season index + late-game/pressure state (seconds remaining, score diff, bonus vs shooting), (d) LightGBM with the same features. Metrics: log loss and Brier on F2 at the attempt level; calibration by decile; responsiveness by shooter as-of FT% quintile; a transfer-subset check (players whose team changed since the prior season) for the shrinkage arm; noise floor. Decision rules: as above. Also report the fitted shrinkage strength and how many attempts it takes for a shooter's own rate to dominate the prior.

---

<!-- RESULTS APPENDED BELOW BY scripts/train_free_throw_v1.py -->

## 2. Grid configuration (as executed, run 2026-09-10 14:00, `scripts/train_free_throw_v1.py --version v1`)

| Dimension | Values |
|---|---|
| FT-1 target | number of attempts in a trip, checked against the rule table `cbb_sim.models.free_throw.TRIP_RULES` |
| FT-1 foul class | derived from CONTEXT ONLY (technical row / and-one signature / the fouling team's prior period foul count), never from the attempt count |
| FT-2 target | one free-throw attempt: made vs missed |
| FT-2 arms | `team_asof` (team-level floor), `eb_shrink` (prior and strength FITTED), `ridge` (logistic ridge), `lgbm` |
| FT-2 features | shooter_ft_asof, shooter_fta_asof, prior_season_ft, has_prior_season, season_idx, seconds_remaining, period, score_diff, in_bonus |
| Folds | F1: train 2022+2023, test 2024. F2: train 2022+2023+2024, test 2025 (selection) |
| Sealed | 2026 -- `assert_not_sealed` on every train and test slice. The FT-1 rule check reports 2026 DESCRIPTIVELY (a rule check is data, not a fit) and no fold sees it |
| Primary metric | attempt-level log loss on F2 |
| Possessions version | `v1` (rim override 0.0 ft); nothing in this model depends on it |
| Noise floor | linear: 200-replicate game-level block bootstrap SE; tree: SD over 5 seed-varied refits |

Version invariance check: the attempt table built under `v1` and under `v2` are IDENTICAL over all 809,294 rows. The rim-location override can only move an `FGA_jump2` to an `FGA_rim`, and no free-throw attempt or trip structure reads a field-goal class, so this is the assertion rather than the assumption.

FT-2 universe: 809,294 free-throw attempts over [2022, 2023, 2024, 2025], of which 801,025 are modelled and 8,254 technical attempts are excluded (their make rate is 0.79561 against 0.71804 for the rest -- the shooter on a technical is chosen by the coach, so the two are drawn from different shooter distributions). 8,858 distinct shooters. ESPN athlete id resolves on 52.67% of modelled attempts; roster position resolves on 79.1%. Runtime 2.4 min.

---

## 3. FT-1: trip structure against the rule

### 3.1 Bonus thresholds RE-DERIVED from each season's own data

The signature is the one `cbb_sim.pbp.possessions` validated: a one-attempt trip inside the one-and-one window is a MISSED front end (a made front end earns a second shot), whereas a one-attempt trip outside the window is an and-one, made about two thirds of the time. `r(p)` below is the share of one-attempt trips at prior foul count `p` whose only attempt missed. The onset is the first `p` where it crosses 0.55; the double-bonus onset is the first `p` after that where it falls back.

| season | bonus onset (prior fouls) | double-bonus onset | r(5) | r(6) | r(7) | r(8) | r(9) | r(10) |
|---|---|---|---|---|---|---|---|---|
| 2022 | 6 | 9 | 0.3753 | 0.7169 | 0.7433 | 0.724 | 0.3365 | 0.3053 |
| 2023 | 6 | 9 | 0.3496 | 0.7312 | 0.7359 | 0.7105 | 0.2954 | 0.3228 |
| 2024 | 6 | 9 | 0.3526 | 0.7052 | 0.7182 | 0.694 | 0.3019 | 0.3132 |
| 2025 | 6 | 9 | 0.3438 | 0.7085 | 0.7203 | 0.7166 | 0.3008 | 0.3224 |
| 2026 | 6 | 9 | 0.3608 | 0.6978 | 0.7003 | 0.6907 | 0.3302 | 0.309 |

**Era boundary: NOT DETECTED** (1 distinct threshold pair(s) across [2022, 2023, 2024, 2025, 2026]). Written to `data/processed/models/free_throw/bonus_era.json`, which the ENGINE reads into GameState -- per `CLAUDE.md`, rule-era flags live in GameState and never inside a fitted sub-model.

### 3.2 Free-throw volume by season (the L4 scoring trend, at the trip level)

| season | n_trips | trips_per_game | attempts_per_game | bonus_trips_per_game | double_bonus_trips_per_game | one_and_one_share_of_trips_pct |
|---|---|---|---|---|---|---|
| 2022 | 102346 | 19.376 | 34.801 | 5.489 | 2.923 | 28.327 |
| 2023 | 111149 | 20.067 | 36.023 | 5.651 | 3.112 | 28.162 |
| 2024 | 117662 | 21.189 | 38.218 | 5.816 | 3.236 | 27.451 |
| 2025 | 118075 | 21.111 | 38.212 | 5.874 | 3.152 | 27.824 |
| 2026 | 128808 | 22.515 | 40.834 | 6.189 | 3.603 | 27.491 |

### 3.3 Rule-derived attempt counts vs the data

| season | trips | shooting w/ 1 attempt | and-one w/ >1 | 1-and-1 single attempt that was MADE | double bonus w/ 1 attempt | >= 4 attempts | violation rate % |
|---|---|---|---|---|---|---|---|
| 2022 | 102,346 | 1,877 | 33 | 315 | 325 | 158 | 2.6459 |
| 2023 | 111,149 | 2,115 | 28 | 368 | 382 | 135 | 2.7243 |
| 2024 | 117,662 | 2,049 | 28 | 345 | 385 | 161 | 2.5225 |
| 2025 | 118,075 | 2,107 | 26 | 330 | 377 | 139 | 2.523 |
| 2026 | 128,808 | 2,134 | 25 | 351 | 423 | 140 | 2.3857 |

Ambiguous mass (NOT violations -- the feed cannot separate a bonus trip from a two-shot shooting foul once the bonus is in force, data-defect row D5 of the change ledger):
- 2022: 22,463 two-attempt trips in the one-and-one, 14,789 in the double bonus
- 2023: 24,347 two-attempt trips in the one-and-one, 16,567 in the double bonus
- 2024: 25,478 two-attempt trips in the one-and-one, 17,259 in the double bonus
- 2025: 25,815 two-attempt trips in the one-and-one, 16,960 in the double bonus
- 2026: 28,135 two-attempt trips in the one-and-one, 19,839 in the double bonus

---

## 4. FT-2: full results

**F1** -- train [2022, 2023], test [2024]

| arm | n | log_loss | brier | calib | worst_gap_pp | level_pp | shape_pp | respons | resp_min_steps | slope_ratio | bonus_gap_pp | shooting_gap_pp | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 210174 | 0.578489 | 0.195791 | FAIL | 2.812 | 0.429 | 2.384 | PASS | 4 | 0.9962 | 0.282 | 0.549 | 10.5 |
| eb_shrink | 210174 | 0.581622 | 0.19706 | PASS | 1.263 | 0.498 | 0.853 | PASS | 4 | 0.9897 | 0.617 | 0.401 | 0.2 |
| ridge | 210174 | 0.582647 | 0.197563 | FAIL | 4.32 | 1.642 | 2.678 | PASS | 4 | 0.9619 | 1.357 | 1.876 | 2.1 |
| team_asof | 210174 | 0.598418 | 0.203157 | FAIL | 7.324 | 1.011 | 6.313 | PASS | 4 | 0.2486 | 1.585 | 0.525 | 0.2 |

**F2** -- train [2022, 2023, 2024], test [2025]  (SELECTION)

| arm | n | log_loss | brier | calib | worst_gap_pp | level_pp | shape_pp | respons | resp_min_steps | slope_ratio | bonus_gap_pp | shooting_gap_pp | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 211760 | 0.575281 | 0.194361 | PASS | 1.266 | 0.391 | 1.027 | PASS | 4 | 0.9769 | 0.343 | 0.43 | 14.7 |
| eb_shrink | 211760 | 0.579314 | 0.195973 | PASS | 1.01 | 0.203 | 1.084 | PASS | 4 | 1.0007 | 0.249 | 0.166 | 0.2 |
| ridge | 211760 | 0.580356 | 0.196493 | FAIL | 2.825 | 0.861 | 1.964 | PASS | 4 | 0.9099 | 0.657 | 1.027 | 0.4 |
| team_asof | 211760 | 0.593953 | 0.201674 | FAIL | 6.802 | 0.646 | 6.156 | PASS | 4 | 0.2981 | 1.13 | 0.25 | 0.2 |

Noise floor: game-block bootstrap SE 0.001113 on `ridge`; LightGBM seed-refit SD 7.6e-05 over seeds [0, 1, 2, 3, 4] ([0.575281, 0.575428, 0.575282, 0.575341, 0.575227]). The floor the decision rule uses is the larger, **0.001113**.

---

## 5. The fitted shrinkage

- **F1**: prior = `position`, strength m = **30.0** pseudo-attempts (train log loss 0.585758).
- **F2**: prior = `position`, strength m = **30.0** pseudo-attempts (train log loss 0.584283).

At m = 30.0 pseudo-attempts, a shooter's OWN rate carries half the weight once they have 30 attempts on the season, three quarters at 90 and nine tenths at 270. The median F2 attempt is taken by a shooter with 32 prior attempts this season, and 51.04% of F2 attempts are taken by a shooter whose own rate already outweighs the prior.

Full grid (train log loss per prior x strength) is in `data/processed/models/free_throw/run_report.json` under `eb_fits`.

### 5.1 Transfer subset

The natural experiment L15 used: players whose modal team changed since the prior season, where a prior-season-based prior is least trustworthy.

| subset | n | actual_make_rate | eb_pred_make_rate | team_asof_log_loss | eb_shrink_log_loss | ridge_log_loss | lgbm_log_loss |
|---|---|---|---|---|---|---|---|
| transfer | 64971 | 0.72151 | 0.71852 | 0.593603 | 0.57951 | 0.580669 | 0.576071 |
| non_transfer | 92307 | 0.73118 | 0.72406 | 0.585308 | 0.569852 | 0.568894 | 0.564111 |
| no_prior_season | 54482 | 0.70401 | 0.71175 | 0.609016 | 0.595113 | 0.599404 | 0.593266 |

---

## 6. Decision, by the pre-registered rule

2 of 4 F2 arms pass BOTH the calibration gate (worst decile gap <= 2.00 pp on classes with a >= 5% share) and the responsiveness gate (monotone in at least 3 of the 4 shooter-as-of-FT% quintile steps -- the L3 round-1 reading of "4 of 5 quintile steps"): `eb_shrink`, `lgbm`.

The tree arm beats the best simpler PASSING arm by 0.004033 log loss = 3.62x the noise floor, so the pre-registered "a tree must beat the simpler arm by more than the floor" clause is satisfied and the simplicity tie-break does not fire.

**WINNER: lgbm**, F2 log loss 0.575281.

---

## 7. S1 scheme confirmation: refit cadence (2026-09-10, written and COMMITTED before this round's modelling)

Authority: `ARCHITECTURE_DECISIONS.md` Decision 9c; `docs/models/README.md` "Standing result"; `docs/LEARNINGS.md`
L21. As with the rebound S1 confirmation (`docs/models/rebound/experiments.md` section 7, same date), the
possession-outcome round-3 pre-registration is in flight on the same question for its own model and is not a
verdict this section borrows -- only its refit-calendar CODE PATH is (section 7.1), so "weekly" and
"conference-aligned" mean the same objects in every sub-model that tests them.

Model class and feature bundle are HELD FIXED at round 1's FT-2 winner, **`lgbm` on `FT_FEATURES`** (section
6). FT-1 (trip structure) is a rule, not a fit, and is untouched by this section. Nothing about the arm, the
event layer, the universe, or the folds is reopened -- only the CALENDAR on which FT-2 refits.

### 7.1 The four schemes

| scheme | refit calendar | object count (F2, 2025) | complexity rank |
|---|---|---|---|
| `S0` | one static fit on the fold's whole train slice, as round 1 scored it. **The reference; must reproduce round 1's recorded F2 log loss (0.575281) exactly** | 1 | 0 |
| `S1_monthly` | `cbb_sim.models.possession_outcome.month_boundaries` on the test season's attempt dates -- the L21 default | 6 | 1 |
| `S1_conf_aligned` | `cbb_sim.features.conference.union_boundaries(monthly, conference_boundary_dates(...))` | 29 | 2 |
| `S1_weekly` | `cbb_sim.features.conference.union_boundaries(weekly_boundaries(...), monthly[:1])` | 24 | 3 |

Identical construction and identical measured object counts to the rebound section (same schedule, same
`scripts/train_possession_outcome_v3.refit_dates` code path, imported not reimplemented): monthly 6,
conference-aligned union 29, weekly union 24, in both the 2024 and 2025 test seasons. Every S1 scheme keeps
the standing contract: each refit uses attempts STRICTLY BEFORE its own date (all prior seasons plus the
test season to date), each test attempt is scored by the most recent refit at or before its own date.

### 7.2 Folds

F2 (train 2022+2023+2024, test 2025) selects, scored for all four schemes. F1 (train 2022+2023, test 2024)
is reported for `S0` and `S1_monthly` only. `S0`'s F1 number is READ from section 4's existing F1 row for
`lgbm` (log loss 0.578489), not refit, for the same reason as the rebound section: it is already the
identical model class, feature bundle and fold code path. `S1_conf_aligned` and `S1_weekly` are not run on
F1, for the same budget and prior-evidence reasons as the rebound section (Decision 9's own diagnostic makes
alignment the least likely dimension to pay); this is stated in advance, not discovered at the wall clock.

### 7.3 Metrics

Unchanged via `FT.score()` -- log loss, Brier, `calib_pass`/`calib_worst_gap_pp`, `resp_pass`/`resp_min_steps`
(this model's 3-of-4 reading), the bonus/shooting segment gap. Additions, computed outside `score()`:

1. **`conf4_gap_pp`**: the worst gated decile-calibration gap restricted to attempts by a shooter on the
   shooting TEAM's own first four weeks of conference play -- `(game_date - first_conf_date).days / 7 in
   [0, 4)` joined on `(season, team_id)` via `cbb_sim.features.conference.first_conference_game_dates`,
   read the same way the rebound section and `train_possession_outcome_v3.conf_window_calibration` read it.
   Segments under **1,000** attempts are `underpowered: true` and never a pass/fail, same convention and
   same stated judgement call as the rebound section (this model's F2 test population, 211,760 attempts, is
   the same order of magnitude as rebound's, so one shared floor is used rather than two).
2. **Decision 8 slope**, from `FT.score()`'s own responsiveness block: `slope_ratio` for
   `shooter_ft_asof->MAKE`, band `[0.8, 1.2]`. `ARCHITECTURE_DECISIONS.md` Decision 8 already records this
   winner's slope as 0.977, inside the band, with a realised quintile span well above the 2 pp exemption
   threshold, so the driver is not exempt.
3. **FT-specific segment (L24), a REPORTED number, never a gate**: the share of training-pool attempts
   (2022-2024, the F2 train seasons) that are (a) technical free throws -- already excluded from every FT-2
   arm's universe, section 2 -- or (b) sit in an `ft_trip_ambiguous`-equivalent trip, defined here as a
   two-attempt trip whose context-derived `foul_class` is `bonus_one_and_one` or `double_bonus` (section
   3.3's own "ambiguous mass" definition: the feed cannot separate a genuine two-shot shooting foul from a
   bonus trip once the bonus is in force). For each scheme's F2 test predictions, calibration is ALSO read
   on the complementary "clean trips only" subset (technical and ambiguous rows both excluded) and reported
   next to the overall number -- L24 already showed this ambiguity is label noise uncorrelated with count
   disagreement and not a matchup defect, so it is evidence about how much of the overall calibration gap
   this noise floors, not a criterion the decision rule reads.

### 7.4 Noise floor and wall-clock budget

**Noise floor.** `S0` refit under `seed=1` on F2. Primary-metric floor = `|log_loss(seed=0) -
log_loss(seed=1)|`. `conf4_gap_pp` decision threshold is the same fixed **0.25 pp** carried from the
possession-outcome round-3 convention (rebound section 7.4); the seed spread is measured and reported as
context, not as the threshold.

**Wall clock.** Measured 2026-09-10 at the 3-thread cap: one `lgbm` fit on the F2 train slice (589,265
attempts) costs **51.0 s** and reproduces the adopted F2 log loss to six decimal places (0.575281). Projected
full-stage cost (S0 F2 + seed1 + `S1_monthly` F2 + F1 + `S1_conf_aligned` F2 (29 fits) + `S1_weekly` F2 (24
fits), padded for growing train slices) is **~65 min**. Hard wall clock: **2.0 h**, checked before each cell
starts. Drop order if tight, most complex first: `S1_weekly` F2 before `S1_conf_aligned` F2. Anything not
reached is written to the results as NOT RUN.

### 7.5 Decision rule

Identical in form to the rebound section: `S0` is the reference. A candidate scheme beats it if, on F2,
EITHER its log loss improves on `S0`'s by more than the primary-metric floor, OR its `conf4_gap_pp` improves
on `S0`'s by more than 0.25 pp, while continuing to pass `FT.score()`'s gates unchanged. Where more than one
scheme beats the reference, the simplest whose log loss is within the floor of the best beater's wins. If
nothing beats the reference, `S0` stands and cadence remains PENDING EVIDENCE for this sub-model, an
explicitly legitimate outcome.

### 7.6 Artifacts

Trainer: `scripts/train_free_throw_v2_s1.py`. Every scheme's dated joblib artifacts and a
`manifest.py`-format manifest (JSON: `refit_date`, `path`, `max_train_date` required) are written to
`data/processed/models/free_throw/s1_confirm/<scheme>/<fold>/` (gitignored; HF-synced, never `git add`ed).
`train_free_throw_v1.py`'s own artifacts (`attempts_v1_era.parquet`, `trips_v1_era.parquet`, `bonus_era.json`,
round-1 `run_report.json`) are read back, never rebuilt or overwritten -- in particular `bonus_era.json`,
which the engine already reads into GameState, is not touched by anything in this section.

<!-- ROUND-2 (S1 SCHEME CONFIRMATION) RESULTS APPENDED BELOW BY scripts/train_free_throw_v2_s1.py -->
