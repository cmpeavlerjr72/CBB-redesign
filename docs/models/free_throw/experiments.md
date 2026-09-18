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

## 8. S1 scheme confirmation: results (run 2026-09-11T02:38:13.302562+00:00, `scripts/train_free_throw_v2_s1.py`)

S0/F2 reproduction of round 1's adopted number: CONFIRMED (cited 0.575281).

L24 segment (training pool [2022, 2023, 2024]): technical attempts 1.057% of all attempts (excluded from every FT-2 arm's universe already); `ft_trip_ambiguous`-equivalent attempts (two-attempt trips whose foul class is bonus_one_and_one or double_bonus) 40.601% -- both reported segments, neither a gate.

| cell | n_fits | log_loss | calib | worst_gap_pp | respons | slope_ratio | conf4_n | conf4_gap_pp | clean_gap_pp | ambiguous_pct | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F2|S0|s0 | 1 | 0.575281 | PASS | 1.266 | PASS | 0.9769 | 45398 | 1.271 | 3.27 | 40.4 | 37.5 |
| F2|S0|s1 | 1 | 0.575428 | PASS | 1.427 | PASS | 0.9735 | 45398 | 1.552 | 3.459 | 40.4 | 30.3 |
| F2|S1_monthly|s0 | 6 | 0.575237 | PASS | 0.835 | PASS | 0.9627 | 45398 | 1.08 | 3.553 | 40.4 | 229.3 |
| F1|S1_monthly|s0 | 6 | 0.578044 | PASS | 1.801 | PASS | 0.9862 | 47167 | 2.065 | 4.147 | 40.668 | 190.4 |
| F2|S1_conf_aligned|s0 | 29 | 0.57521 | PASS | 0.697 | PASS | 0.9554 | 45398 | 0.935 | 3.775 | 40.4 | 892.9 |
| F2|S1_weekly|s0 | 23 | 0.57536 | PASS | 0.892 | PASS | 0.9368 | 45398 | 1.057 | 3.782 | 40.4 | 461.1 |

S0 F1 (cited, not refit, section 4): log loss 0.578489, calib gap 2.812 pp (FAIL), slope 0.9962.

Noise floor: primary-metric floor (S0 second-seed refit) = **0.000147**. conf4 seed spread (context only) = 0.281 pp.

### 8.1 Decision

- `S1_monthly`: log loss 0.575237 (gain +0.000044 vs S0), conf4 gain 0.191 pp, gates PASS, beats reference: False
- `S1_conf_aligned`: log loss 0.57521 (gain +0.000071 vs S0), conf4 gain 0.336 pp, gates PASS, beats reference: True
- `S1_weekly`: log loss 0.57536 (gain -0.000079 vs S0), conf4 gain 0.214 pp, gates PASS, beats reference: False

**WINNER: S1_conf_aligned** -- beats the S0 reference beyond the floor; simplest arm within the floor of the best beater

---

## 9. Technical free throws: the engine's missing scoring rule, now sized at 23% of gate G4's FTA/FGA miss (PROPOSED, written 2026-09-18 by the G4 diagnostic lane BEFORE any modelling; NOT RUN, NOT ADOPTED, no served default changed)

This is the open item `model.md` section 9 has carried since 2026-09-10 and that
`docs/tests/ft_trip_reconciliation_2026-09-10.md` identified and priced at
~1,960-2,490 technical attempts a season (~0.14-0.22 points per team-game). It is
pre-registered here because the G4 diagnostic has now shown it is not a rounding
error on a gate: it is **23.0% of the engine's -1.229 pp FTA/FGA miss on fold 2**
(`docs/tests/g4_oreb_fta_diagnostic_2026-09-18.md` section 2).

The measurement, on the 11,179 2025 team-games carrying both sources:

| | box / team-game | pbp event layer / team-game | pbp - box |
|---|---:|---:|---:|
| FTA | 19.117 | 18.934 | **-0.1828** |
| FGA | 58.006 | 57.949 | -0.0578 |

pooled FTA/FGA: box **0.32957**, event layer **0.32674**, difference **-0.282 pp**,
of which -0.318 pp is the FTA count and +0.035 pp the FGA count. Technical
attempts explain 94.3% of disagreeing team-games exactly and 96% of the 2025
aggregate (1,961 of a 2,044-attempt gap). `possessions.py`'s `_handle_ft_trip`
buffers technical free throws into `tech_points_off`/`tech_points_def` and never
into `fta`/`ftm` -- **correctly**, per `model.md` section 9 -- so FT-2 is trained
without them and the engine has no rule that can produce them. The engine is
therefore structurally short by ~0.18 FTA and ~0.13 FTM per team-game against the
box that G4 grades it on, and it is short **one-directionally, every game**.

`engine_rules_from_data` already carries the measured
`technical_trip_rate_per_team_game = 0.098052`, read from data and **used
nowhere** in `loop.py`.

### 9.1 Candidates

| arm | what it is |
|---|---|
| `X0` | **reference**: no technical rule; the engine scores no technical free throws (the status quo) |
| `X1` | a per-team-game Poisson draw of technical TRIPS at the measured league rate, each trip two attempts, shooter drawn from the five on the floor by as-of FT-attempt share, makes drawn by the served FT-2 model |
| `X2` | `X1` with the trip rate conditioned on the pre-game state the feed can actually support (season, conference game, neutral site), fitted on the training folds rather than taken as a league constant |
| `X3` | `X1` with the shooter rule changed to "the team's best as-of FT shooter among the five on the floor", which is what the coach's choice approximates and what `model.md` section 9 says pooling would bias |

`X1` is the simplest arm that closes the scope gap and is the one to beat. Note
that `X2`'s conditioning set is deliberately thin: CBBD's play-by-play has no
flagrant or lane-violation play type at all (`ft_trip_reconciliation` section 2,
0 of 462,118 2025 rows), so anything richer is not observable in this feed and is
not on the list.

### 9.2 Features

`X1` has none beyond the rate constant. `X2` adds only season, conference-game
flag and neutral site, all already available pre-game and all expressed relative
to the league. The FT-2 make model itself is **not** re-fitted by this round: the
served `S1_conf_aligned` shooter-keyed arm is used unchanged, and technical
attempts stay out of its training universe (the section-9 exclusion is correct and
this round does not disturb it).

### 9.3 Folds

Fold 1 trains through 2022-23 and tests 2023-24. Fold 2 trains through 2023-24 and
tests 2024-25. **Fold 2 selects.** 2025-26 is SEALED.

### 9.4 Primary metric

Because `X0` predicts a structural zero, an offline likelihood comparison is
degenerate. The primary is therefore **closed-loop and pre-declared**: the
engine's pooled **FTA/FGA against the verified box** on fold 2, paired-seed, with
`X0` as the reference. Served value **0.31726** against a box actual of
**0.32955**; an arm's primary is the absolute residual `|sim - 0.32955|`.

Two named secondaries: pooled **FTM per team-game** against the box (the engine is
short ~0.13/team-game), and **G9 total bias** (technical free throws add points,
so this arm cannot be read without the points gate next to it).

### 9.5 Segment breakdowns

Underpowered cells labelled, never folded into a pass or a fail:

1. per team-game distribution of technical trips (the arm must not produce a
   different SHAPE from the actual, only a matching mean);
2. by month, home/away/neutral, conference vs non-conference -- the served
   FTA/FGA gaps are -2.99 pp (Nov) to -0.20 pp (Feb), -0.43 / -1.62 / -2.63 pp by
   site, -2.28 / -0.61 pp by conference;
3. by 2024 prior-season FT-rate quintile of the offence (served slope ratio
   **0.523**); a league-constant technical rate must be shown NOT to flatten it
   further;
4. a stated **overlap check** against `docs/models/possession_outcome/`
   experiments.md section 13 and `docs/models/late_game/` section 1: this arm and
   those rounds all move FTA/FGA, so any joint read must be paired-seed and must
   report each arm's marginal contribution, never their sum.

### 9.6 Noise floor

A **seed-offset paired closed-loop run at matched seed count**, the same floor
form the engine gates use. The existing 200-seed paired band on the FTA/FGA line
is **0.0001** (`docs/tests/gate_noise_band_F2_2025_s200_v1_clockv3c_2026-09-11.md`),
and the 20-seed floor already on record is the fallback where 200 seeds are not
affordable. A winner must beat `X0` on the primary by more than the measured band.

### 9.7 Decision rule

1. An arm is eligible only if it beats `X0` on the primary by more than the
   measured paired-seed band.
2. Among eligible arms the winner is the lowest primary; ties inside one band go
   to the simpler arm, ordered `X0 < X1 < X3 < X2`.
3. A winner must **not** regress G9 total bias, G9 calibration slope, G1's
   possession mean, or G2's PPP terciles beyond their own measured bands
   (technical free throws stop no clock in this engine, but they do add points).
4. A winner must not flatten the 2024-prior-quintile FT-rate slope ratio.
5. **This arm may not be graded alone as a fix for G4.** It is 23% of the FTA/FGA
   miss by construction; the other 77% is owned elsewhere (section 13 of
   `possession_outcome`, and the late-game lane). A report that shows FTA/FGA
   improving must state which arms were live.
6. Ties go to the simpler model. If no arm clears rule 1, **no arm is adopted**.

### 9.8 What this round may not do

No post-hoc multiplier, cap, clip, offset or blend on sim output. The technical
rate is READ from `engine_rules_from_data` / fitted on training folds; an arm whose
rate is chosen so that 2025 FTA/FGA lands on 0.32955 is banned and is not on the
list. Technical attempts stay excluded from the FT-2 training universe. The
2025-26 season stays sealed.

---

## 9.9 Amendment (PM conditions, written by the free-throw-technicals worker
before any modelling, 2026-09-18): measurement confirmation, per-possession
re-expression, two added arms, possession-retention rule, extra evidence cuts

The PM approved section 9 as written subject to five conditions. This
amendment satisfies each in turn, append-only; nothing in sections 9.1-9.8 is
edited or retracted.

### 9.9.1 Measurement confirmation (condition a)

**Identification.** Both feeds carry a single, flat event category for a
technical foul and nothing finer: CBBD's `playType` is literally `"Technical
Foul"` (`src/cbb_sim/pbp/events.py`, `FAM_TECHNICAL`); hoopR's `type_text` is
also exactly `"Technical Foul"` (1,888 of 2,190,101 2025 pbp rows), confirmed
by a direct value-count on `data/raw/hoopr/pbp/play_by_play_2025.parquet` —
searching that column for `Flagrant`/`Intentional`/`Administrative`/
`Unsport`/`Bench`/`Coach`/`Delay` returns zero rows, extending
`ft_trip_reconciliation_2026-09-10.md`'s existing CBBD-side finding (0 of
462,118 2025 rows contain "flagrant"/"violation" in `playText`) to hoopR: the
flagrant / intentional / administrative / unsporting distinction the PM asked
for **is not observable in either feed, at all**. One coarser distinction IS
recoverable from hoopR's free text and is reported as the closest available
proxy: `"Technical Foul on <Team Name>."` (a bench/coaching-staff technical,
no athlete id attached) versus `"Technical Foul on <Player Name>."` (an
individual technical). This is not flagrant/intentional/administrative/
unsporting, and no arm below conditions on it (the pre-registered candidates
do not, and this round does not add one to the list post hoc).

**Grading is against the verified box, and the box does include technicals.**
Already established and re-cited rather than re-derived:
`ft_trip_reconciliation_2026-09-10.md` sections 1-2 show netting technical FTA
out of the CBBD event layer's `fta` explains 94.3% of all disagreeing
team-games EXACTLY (to the attempt) and 96-109% of the season aggregate
`box_fta - ev_fta` gap in every season 2022-2025 (1,805/1,654 in 2022,
2,439/2,487 in 2023, 2,053/2,059 in 2024, 1,961/2,044 in 2025). This is the
box-vs-pbp disagreement the PM asked to have quantified, and it is the same
number section 9's opening paragraph already cites as the -0.282 pp channel.

**New cross-source check (not previously run): does CBBD's own technical-trip
count understate the true incidence?** The candidate arms below are trained
on `trips_v1_era.parquet`'s `foul_class == "technical"` count, which comes
from CBBD's pbp through `possessions.py`. hoopR's independently-collected pbp
carries its own technical-foul events and lets that count be cross-checked
without touching a box total. Two things have to be netted out first: (1)
**offsetting simultaneous technicals** (one assessed on each team at the same
game clock) are a real NCAA out — no free throws are shot — and are not a
missed trip; (2) a technical is identified as a `(game_id, period,
clock)` "moment" so that a moment with a technical on exactly one team is
the population that MUST produce a trip.

| season | hoopR technical events | hoopR offsetting moments (both teams, same clock — no FTs) | hoopR single-team moments (should produce a trip) | CBBD-derived technical trips (`trips_v1_era`) | ratio CBBD / hoopR |
|---|---:|---:|---:|---:|---:|
| 2022 | 1,942 | 269 | 1,326 | 902 | 0.680 |
| 2023 | 2,742 | 283 | 2,115 | 1,355 | 0.641 |
| 2024 | 2,106 | 322 | 1,391 | 954 | 0.686 |
| 2025 | 1,888 | 301 | 1,211 | 879 | 0.726 |
| 2026 (descriptive only, sealed) | 2,288 | 385 | 1,452 | 1,045 | 0.720 |

**Finding: CBBD's own technical-trip table — the training target for every
arm below — captures only 64-73% of the technical incidents hoopR's pbp
implies should have produced a free-throw trip, rising monotonically each
season (0.641 -> 0.726) but never closing.** This is a genuine, one-directional
measurement gap, distinct from the already-known exclusion channel: even a
perfectly-calibrated arm fit on CBBD data is fit against an UNDER-COUNTED
target, so it should be expected to close less than the full -0.282 pp
technical channel even at its own internal calibration optimum. This is
**reported as a ceiling on what this round can close, not corrected by
re-scaling the rate** — CLAUDE.md bans exactly that kind of after-the-fact
adjustment (section 9.8), and switching the training source to hoopR pbp
without a pre-registered comparison would be an unregistered arm change, not
this round's job. Flagged for a future free-throw round, not fixed here.

**Trip-length era check** (same method as section 3.1's bonus-threshold
check, applied to `foul_class == "technical"`):

| season | 1 attempt | 2 attempts | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 202 | 588 | 28 | 79 | 4 | 1 |
| 2023 | 522 | 690 | 38 | 103 | 1 | 1 |
| 2024 | 148 | 642 | 44 | 113 | 6 | 1 |
| 2025 | 73 | 652 | 39 | 111 | 3 | 1 |
| 2026 | 84 | 784 | 55 | 118 | 4 | 0 |

2 attempts is the modal outcome every season (54-66% share), consistent with
one stable NCAA rule (2 shots) across the whole window: **no era boundary
detected**, same negative result as section 3.1's bonus-threshold check, so
no new GameState era flag is created (`FT.TRIP_RULES["technical"]` already
carries "1 or 2 by infraction" and needs no change). 2023's outlier
1-attempt share (522, against 73-202 every other season) coincides with
2023's already-documented worse general feed completeness elsewhere in this
project and is flagged as a season-specific DATA-QUALITY wrinkle, not a rule
change — it is not treated as signal by any arm below.

**Possession is retained — confirmed by reading the engine code, not
assumed.** `src/cbb_sim/pbp/possessions.py::_handle_technical` /
`_handle_ft_trip` (read-only, no edit): a technical trip increments
`n_tech_trips` and its points go into `tech_points_off`/`tech_points_def`, but
the routine never opens, closes, or advances a `_Chance` object — the
currently open possession (if any) is untouched and continues exactly as it
was, and if no possession is open the points are buffered in `pending_tech`
until one opens. **Consequence for every arm below and for any future
engine wiring: a simulated technical trip must not consume one of the
engine's simulated possession/chance slots and must not change which team is
next to inbound.** This is the mechanism, stated plainly, behind "possession
is retained": the ball does not change hands and no extra chance is created.

**WHO shoots — already pre-registered, data supports testing it.**
`trips_v1_era.parquet` carries `shooter_id` on technical rows too (verified:
non-null on the same basis as every other trip), so X1's "as-of attempt-share"
shooter rule and X3's "best as-of shooter" rule are both directly testable
against the actual chosen shooter's own as-of FT rate. See 9.9.4.

### 9.9.2 Rates per possession, not counts (condition b, re-expression of X1/X2)

`engine_rules_from_data`'s `technical_trip_rate_per_team_game` (0.098052) is a
PER-GAME constant and is the wrong shape for the standing modelling rule
("Rates per possession, not counts", `CLAUDE.md`) — a fixed count per game
does not scale with a simulated game's own realised pace. X1 and X2 are
therefore RE-EXPRESSED, not replaced: the denominator becomes the number of
team-chances exposed, read from `data/processed/possessions_v2/chances_{season}.parquet`
(one row per offensive chance; each chance exposes BOTH teams once — the
offence to "did we just draw a technical" and the defence to the same
question from their own side — so exposure per game is 2x the chance count).
`chances.start_clock`, `chances.period` and `chances.start_score_diff` are
used for any state conditioning below because they are named and built as
PRE-OUTCOME quantities (the state AT THE START of the chance) — the L27
caution about a possessions-table `score_diff`/`duration_s` being post-outcome
on most rows applies to a different, unprefixed column in a different table;
this table's field is explicitly `start_score_diff`/`start_clock` and no
end-of-chance field is used anywhere in this round.

### 9.9.3 Two arms added (condition b, items iii and iv)

Section 9.1's `X2` conditions on PRE-GAME static covariates only (season,
conference-game flag, neutral site) — it does not cover the PM's item (iii)
("period/minute, score margin bucket"), which is an IN-GAME state, not a
pre-game one. Section 9.1 also has no arm at all matching item (iv) ("team/coach
as-of rate shrunk toward league"). Both are added here, append-only; `X0-X3`
are unchanged and still run exactly as pre-registered.

| arm | what it is |
|---|---|
| `X4` | rate conditioned on IN-GAME state: `game_phase` (`H1_early` 0-9 min, `H1_late` 10-19, `H2_early` 20-29, `H2_late_OT` 30-40+OT, from `chances.period`/`start_clock`), `margin_bucket` (`trailing` <= -5, `close` -4..4, `leading` >= 5, from `start_score_diff`, the ACTOR's own margin — the team being asked "do you commit a technical here"), and `site` (home / away / neutral) — 4 x 3 x 3 = 36 cells, each a trips-per-team-chance rate, shrunk toward `X1`'s pooled rate (Poisson-conjugate, pseudo-exposure `k` chosen the same way `eb_shrink` chooses its strength elsewhere: grid search minimising held-out Poisson deviance on a chronological split of the training fold itself) |
| `X5` | team-level as-of technical-trip rate: each team's own trips-per-team-chance rate on games STRICTLY BEFORE the one being scored, shrunk toward `X1`'s pooled league rate (empirical-Bayes, strength fitted the same way), expressed relative to the SNAPSHOT'S league mean per the standing "every rating feature relative to its own snapshot's league mean" rule, `site` (home/away/neutral) carried alongside as a feature since the standing rule makes it first-class in every scoring-stage model |

`X2`, `X4` and `X5` all carry `site` in their feature list, satisfying
"home/away/neutral in every feature list" for every arm that has one; `X0`,
`X1` and `X3` are rate CONSTANTS with no conditioning features by
construction (the same exemption a pure floor/reference arm gets everywhere
else in this project) and are not retrofitted with a feature they were never
registered to carry.

**Decision-rule ordering extended** (section 9.7 rule 2's tie-break order),
by parameter count, least to most: `X0 < X1 < X3 < X2 < X4 < X5`. `X3` ties
`X1`'s parameter count (it changes the shooter rule, not the rate) and is
kept after `X1` per the original text's own ordering. Everything else in
section 9.7 (eligibility beyond the measured band, no-regression clauses,
the 23%-of-the-gap caveat, ties to the simpler model) applies unchanged to
`X4` and `X5`.

A team/coach-level arm was pre-registered as "team/coach"; this round fits
the TEAM identity only. A coach identity crosswalk (`data/raw/coaches/`) is
on disk but wiring team-to-coach-to-season reliably is out of this round's
scope and is not attempted rather than attempted informally — recorded as a
scope limit, not a silent gap.

### 9.9.4 Extra evidence cuts (conditions c and d)

1. **Period/minute is added as its own cut** to section 9.5's list (it was
   implicit in `X4`'s conditioning set but not listed as a reported
   breakdown): report the technical-trip rate by `game_phase` (as defined in
   9.9.3) on both the actual and every arm's prediction, test fold only.
2. **Power calculation for the team-level quintile cut (section 9.5 item 3),
   stated explicitly rather than left as a label.** At the league technical
   rate (~0.08-0.12 trips/team-game) and ~72 teams per quintile-season (the
   same grouping G4's OREB%/FT-rate quintile cuts used), expected technical
   trips per quintile-season are on the order of 72 games/team x 0.1
   trips/team-game / — computed exactly at run time from each quintile's own
   team-game count and the ACTUAL test-season rate, with the Poisson standard
   error stated alongside every quintile's observed rate. A quintile slope is
   only read as a refutation of responsiveness when the between-quintile gap
   exceeds several such standard errors; otherwise it is labelled
   UNDERPOWERED and reported, never presented as a flat-and-therefore-no-signal
   finding.

Nothing in this amendment changes section 9.3 (folds), 9.4 (primary metric),
9.6 (noise floor form), 9.7 (decision rule, beyond the ordering extension
above) or 9.8 (prohibitions) — all remain exactly as PM-approved and are
carried into the run below unchanged.

---

## 9.10 Round 1 results: OFFLINE ONLY (run 2026-09-18,
`scripts/exp_free_throw_technicals_v1.py`; full detail
`docs/tests/free_throw_technicals_2026-09-18.md`)

**No closed-loop paired-seed run this round** (job permission exercised: "if
time does not allow, stop at the offline table and say so" — reasons in the
results doc section 6: a first-time engine-file wiring change plus a
500x25-seed run on top of four other lanes' closed loops running on the same
tree right now, under a 2-worker compute cap). **NOTHING ADOPTED, no served
default changed.**

`X0` is degenerate on both folds as pre-declared (9.4). `X1`, `X2`, `X4` and
`X5` are statistically indistinguishable from each other on the aggregate
offline primary (pooled Poisson deviance 42-49 on both folds; `X2`'s and
`X4`'s internal shrinkage search picks a pseudo-exposure at or near the grid
maximum, i.e. both collapse toward `X1`) and all four overshoot the test
season's own realised rate by ~22-25% in both folds (the pooled training
window is inflated by 2023's anomalous technical spike). Every arm clears
rule 9.7.1's eligibility (noise floor ~1.2-1.6e-05 on the rate, ~0.008 pp of
FTA/FGA — two orders of magnitude below any positive-rate arm's effect).

**Offline analytic FTA/FGA closure** (additive, closed-form — a technical
trip never touches an existing chance, confirmed by reading
`possessions.py`): 115-134% of the -0.282 pp technical channel, ~26-27% of
the -1.229 pp total FTA/FGA gap, essentially identical across `X1/X2/X4/X5`.
Flagged, not celebrated: this lands near 100% of the channel because a
~25% training-pool overshoot (above) happens to be a similar size to, and
opposite in sign from, a newly-measured CBBD-vs-hoopR technical-count
disagreement (CBBD's own trip table, the training target for every arm,
sits at 64-73% of what hoopR's independent pbp implies the true incidence
is, every season, not explained by clock-precision grouping — checked to a
10-second tolerance). Neither bias is corrected here (9.8); this is reported
as an unvalidated coincidence, not a calibration.

**Section 3.1's team-level quintile check is POWERED (~5.5 sigma end to
end) and finds a real, two-fold-replicated persistence effect** (2025
actual rate rises monotonically 0.000380 → 0.000644 across 2024-prior-rate
quintiles). `X1` fails this by construction (flat, slope ratio 0.0); `X5`
reproduces it on both folds (4/4 monotone, slope ratio 1.336 F2 / 0.893 F1)
despite scoring marginally worst on the aggregate deviance metric above —
the same shape of tension Decision 8 was written to resolve elsewhere in
this cascade, not resolved here.

**Who shoots (X1 vs X3):** the actual technical shooter's as-of FT rate
(0.798, n=4,090) sits closer to the team's best as-of shooter that game
(0.862) than to the attempt-share-weighted average (0.708), favouring `X3`'s
shooter rule; the as-of machinery already explains the elevated observed
technical make rate (0.798 predicted vs 0.796 observed) through shooter
identity alone.

**Mechanically applying the pre-registered tie-break** (`X0 < X1 < X3 < X2
< X4 < X5`) selects `X1` (paired with `X3`'s shooter rule). The PM is handed
this choice alongside the responsiveness tension above rather than having it
resolved by the tie-break's text, which was written before that tension was
visible. Ledger row: `docs/models/change_ledger.md`.

---

## 10. Round 1b: target adjudication and pre-registration (2026-09-18)

### 10.0 PM ruling on round 1 (2026-09-18)

Recorded verbatim-in-substance, as instructed. **NOTHING is selected from
round 1.** (a) `CLAUDE.md` requires grading truth to be verified against a
second source before any grade is trusted; a target two sources disagree on
by ~30% is not verified, so round 1's table cannot select an arm. (b) The
tie-goes-to-simpler rule applies only among ELIGIBLE arms; the standing
matchup-specific rule makes a powered responsiveness failure disqualifying,
so `X1` (flat against a 5.5-sigma team slope, section 9.9.4/3.1) is not
eligible as a final answer regardless of aggregate deviance; it stays in the
table as the baseline. (c) The 115-134% offline FTA/FGA closure is not
evidence of anything.

### 10.1 Target adjudication (job step 1, before any modelling)

**Method.** Built a THIRD estimate from verified box finals per the job spec
(box FTA minus pbp-accounted non-technical FTA), and — because that
comparison alone could not explain round 1's 64-73% CBBD-vs-hoopR gap —
two NEW, independent same-clock reconstructions of technical free-throw
ATTEMPTS (not just moment counts) directly from each vendor's own raw pbp:
`scripts/build_ft_technical_target_v1.py`. A technical free-throw trip is a
dead ball: every free-throw row belonging to it carries the IDENTICAL
period-scoped clock as the technical-foul row itself (confirmed by hand on
game 401700212/401700182, both vendors, before being trusted at scale — see
the results doc). So: group technical-foul rows into `(game, period, clock)`
moments; a moment with exactly one distinct offending team is a
single-team moment that MUST produce a trip (an offsetting pair produces
none, the real NCAA out, re-confirmed here); scan forward from the first
technical row of the moment while the clock is frozen, counting free-throw
rows credited to the team that was NOT called. Run independently on
`data/raw/hoopr/pbp/` and on `data/raw/cbbd/pbp/` (the latter is NOT
`trips_v1_era.parquet` — it is a fresh scan of CBBD's OWN raw feed,
independent of the `possessions.py` pipeline that built `trips_v1_era`).

**Finding 1 — the two VENDORS essentially agree; round 1 was comparing the
wrong pair.** CBBD's raw pbp technical-attempt count matches hoopR's
independent raw-pbp count to within 0-0.6% every season, not 64-73%:

| season | hoopR raw-scan FTA | CBBD raw-scan FTA | ratio | `trips_v1_era` (round-1 target) FTA | ratio to hoopR |
|---|---:|---:|---:|---:|---:|
| 2022 | 2,411 | 2,411 | 1.000 | 1,804 | 0.748 |
| 2023 | 3,285 | 3,285 | 1.000 | 2,439 | 0.743 |
| 2024 | 2,666 | 2,666 | 1.000 | 2,052 | 0.770 |
| 2025 | 2,420 | 2,435 | 1.006 | 1,959 | 0.809 |

**Finding 2 — the mechanism is a `possessions.py` bug, not a CBBD vendor
gap.** `_handle_technical(self, i, t)` (`src/cbb_sim/pbp/possessions.py`,
read-only — not edited, per this lane's scope restriction) looks only ONE
row ahead of the `technical` event for a free throw:
```
def _handle_technical(self, i, t):
    j = i + 1
    if j < self.n and ev["cls"][j] in ("FT_made", "FT_missed"):
        return self._handle_ft_trip(j, ...)
    return i + 1
```
CBBD's raw feed routinely inserts ONE administrative row between the
`Technical Foul` event and its free throws — an automated census of the row
immediately following every one of the 1,228 team-games where
`trips_v1_era` recorded ZERO technical attempts but the raw scan found some
shows it is `Lost Ball Turnover` (crediting the technical'd team's
interrupted possession) 89% of the time and a companion `PersonalFoul` row
10% of the time, pooled across all four seasons (1,141 and 126 of 1,280
matched instances). When that happens, `_handle_technical` gives up
immediately and the trip is never tagged `foul_class == "technical"`; its
free throws are then swept up by the GENERIC (non-technical) free-throw
handler on a later loop iteration and land in `trips_v1_era` tagged
`foul_class in {"foul", "none"}` instead. **Confirmed by hand, not just by
the aggregate pattern**: 15 full pbp transcripts were read line-by-line
(both vendors' text, `data/raw/cbbd/pbp/plays_{season}.parquet` `playText`
and `data/raw/hoopr/pbp/play_by_play_{season}.parquet` `text`), e.g. game
401364434 (2022): `OfficialTVTimeOut -> Technical Foul on A.J. Hoggard ->
Lost Ball Turnover (A.J. Hoggard) -> MadeFreeThrow x2` — and for each, an
EXACT match was verified between the raw-scan's `(game, period, clock,
team_id, opp_id)` and a `trips_v1_era` row at that identical key carrying
`foul_class in {"foul","none"}` instead of `"technical"` (5-for-5 on the
first hand-read batch, then generalized to the full 1,228-team-game
population by the automated next-row census above — a complete census of
the affected population, not a 30-game extrapolation).

**Finding 3 — the job's own suggested third estimate (box-implied) is
confounded by the SAME bug, in the opposite direction, and is REJECTED as
an adjudication source.** `box_implied_fta = box_fta - ev_fta` is smaller
than either raw-pbp scan every season (2022: 1,654 vs 2,411; 2025: 2,044 vs
2,435) and even smaller than `trips_v1_era` in two of four seasons. Reason:
the mis-tagged technical attempts land in `ev_fta`'s NON-technical bucket
(via `_handle_ft_trip(..., technical=False)`), which inflates `ev_fta` and
therefore SHRINKS `box_fta - ev_fta` — the exact opposite direction from
what an under-counted technical target would predict, on top of whatever
independent event-layer noise `ev_fta` already carries for unrelated
reasons. Box-implied is a residual of two entangled biases and cannot
isolate the technical rate; it helped surface the puzzle (round 1 already
used it to size the channel) but cannot adjudicate it.

**Finding 4 — a smaller, disjoint edge case cuts the other way.**
`trips_v1_era`'s own forward-iterating state machine correctly handles a
case this round's moment-scanner does not: two `Technical Foul` rows on the
SAME team at the IDENTICAL frozen clock (a companion/second disciplinary
technical logged after the first one's free throws), e.g. game 401364790
(2022): `PersonalFoul -> Lost Ball Turnover -> Technical Foul -> FT -> FT ->
Technical Foul` (all at one clock value). The scanner's original
"last-technical-row" convention walked past the free throws; found by
hand-reading this exact game, fixed in `build_ft_technical_target_v1.py`
(scan from the FIRST technical row of a moment, not the last — safe because
the inner while-loop does not break on encountering a second technical row)
and confirmed to shrink, not eliminate, this bucket (135 -> 127 team-games
of 43,925, ~0.29%, pooled four seasons).

**Verified target.** A moment-level UNION (not a max of two counts, which
can silently mismatch which trips are being counted) of (a) the CBBD raw
same-clock scan and (b) `trips_v1_era`'s own technical trips, keyed on exact
`(season, game_id, beneficiary team, period, clock)`, built by
`scripts/build_ft_technical_target_v2_verified.py`. Of 5,909 pooled
technical moments recognised by either source: 3,868 (65%) are found by
BOTH, 1,882 (32%) by the raw scan only (finding 2's mechanism), 159 (3%) by
`trips_v1_era` only (finding 4's mechanism). Cross-validated against hoopR's
fully independent raw-pbp scan — the CLAUDE.md "second source" requirement:

| season | verified trip count | verified FTA | hoopR raw-scan FTA (2nd source) | residual vs hoopR |
|---|---:|---:|---:|---:|
| 2022 | 1,281 | 2,606 | 2,411 | +8.1% |
| 2023 | 2,082 | 3,610 | 3,285 | +9.9% |
| 2024 | 1,342 | 2,907 | 2,666 | +9.0% |
| 2025 | 1,204 | 2,644 | 2,420 | +9.3% |

The residual is fully attributed, not a mystery: hoopR's OWN same-clock scan
(before folding in `trips_v1_era`'s finding-4 corrections) ALSO has 56-87
zero-attempt moments a season (the identical double/multi-technical-
same-clock edge case, symmetric across vendors — this is an algorithmic
scanning limitation, not a CBBD- or hoopR-specific defect), and
`trips_v1_era` only resolves 159 of the pooled total correctly, leaving a
small further residual honestly reported as unresolved (order 1-3% of
moments) rather than patched to close it (`CLAUDE.md` 9.8 bans exactly
that). One hand-confirmed hoopR-side event-ordering anomaly (game
401700380, 2025: hoopR's own `sequence_number` order is non-monotonic in
its clock field for one play cluster) accounts for the small
hoopR-scan-vs-CBBD-raw-scan gap (11 of 11,179 2025 team-games, 0 in
2022-2024) on the OTHER side of this comparison.

**Verdict: the CBBD-trip-table target (`trips_v1_era`, round 1's target) is
REJECTED — it is not a vendor-under-logging problem as round 1 characterized
it, it is a root-caused, hand-confirmed pipeline defect that undercounts by
19-29%. The box-implied estimate is REJECTED as confounded by the same
defect in reverse. The VERIFIED target for round 1b is the moment-level
union described above** (`data/processed/models/free_throw/
technical_target_verified_v1.parquet`, one row per `(season, game_id,
team_id)`: `verified_trip_count`, `verified_fta`), residual disagreement
against the second (hoopR) source quantified at 8-10% and fully explained by
a shared, named algorithmic edge case rather than a mechanism gap. No file
this round overwrites `trips_v1_era.parquet`, `attempts_v1_era.parquet`, or
any existing table; three new versioned siblings were written:
`technical_target_hoopr_v1.parquet`, `technical_target_cbbd_raw_v1.parquet`,
`technical_target_reconciliation_v1.parquet` (attempt grain, all four
sources side by side, 43,925 team-game rows), plus
`technical_target_verified_v1.parquet` (trip grain, the round 1b target).
`possessions.py::_handle_technical`'s narrow lookahead is a real,
now-precisely-located defect worth fixing at the source — flagged for the
engine-core owner, NOT fixed here (this lane may not touch `src/cbb_sim/`
and the fix belongs to whoever owns that file next).

### 10.2 Pre-registration: round 1b arms (BEFORE any modelling)

**Folds, unchanged from round 1** (`section 9.3`): F1 trains through 2022-23,
tests 2023-24; F2 trains through 2023-24, tests 2024-25, **F2 selects**;
2025-26 stays sealed (`assert_not_sealed` on every load).

**Target, changed from round 1**: `technical_target_verified_v1.parquet`'s
`verified_trip_count` (a Poisson count of technical trips per team-game),
replacing `trips_v1_era`'s `foul_class == "technical"` count everywhere.
Exposure unchanged from section 9.9.2 (team-chances, both teams exposed
per chance, `data/processed/possessions_v2/chances_{season}.parquet`).

**Arms.**

| arm | what it is | site feature? | team-keyed? |
|---|---|---|---|
| `X0` | reference: no technical rule | n/a (constant zero) | no |
| `X1` | league-constant rate, pooled equally over all training seasons (round 1's `X1`, re-run on the verified target) | no (rate constant, same exemption as round 1) | no |
| `X5` | team as-of rate, EB-shrunk toward `X1`'s pooled rate (round 1's `X5`, re-run on the verified target) | yes (carried forward from 9.9.3) | **yes** |
| `X6` | **recency-weighted training window** (job-mandated addition): pooled rate over training seasons weighted by `exp(-ln(2)*(last_train_season - season)/halflife)`, `halflife` in seasons chosen from grid `{0.5, 1, 2, 4, 999}` (999 ~= uniform = reproduces `X1`) by the SAME chronological internal-validation split `grid_search_shrinkage` already uses (fit on all-but-last train season, validate on the last) | no (rate constant) | no |
| `X7` | **league-level in-season as-of rate** (job-mandated addition): for each test-season exposure row, an EB blend of the test season's OWN cumulative rate strictly before that row's game date (`games_universe.game_date`, a pre-outcome join, never the chance table's own post-hoc fields per the L27 caution already noted in 9.9.2) and `X6`'s recency-weighted prior level, pseudo-exposure `k` fit the same grid-search way; multiplied by a `site` ratio (home/away/neutral rate / overall rate) fit once on the training pool — this is what makes `X7` a genuine multi-feature arm ("league-level ... so a one-season spike does not set the level", features relative to the snapshot's own in-season mean) rather than a second flat constant | **yes** (multiplicative site ratio) | no |
| `X5r` | **added beyond the job's literal four-arm list, in its spirit** ("plus the arms the round-1 overshoot calls for"): `X5`'s team-shrinkage machinery unchanged, but shrunk toward `X6`'s recency-weighted level instead of `X1`'s flat all-season pooled level — the direct, obvious fix for round 1's exact tension (`X1` passes level, fails responsiveness; `X5` passes responsiveness, inherits `X1`'s bad level as its own shrinkage target) | yes (carried from `X5`) | **yes** |

`X0`/`X1`/`X6` are rate constants with no conditioning feature and are
exempt from "home/away/neutral in every feature list" on the same standing
basis round 1's `X0`/`X1`/`X3` were. **`X6` and `X7` are PRE-DECLARED
STRUCTURALLY INELIGIBLE for final adoption, stated before running rather
than discovered after**: neither varies by team, so the team-quintile
responsiveness check (below) is not merely "not attempted" for them the way
it was for round 1's `X2`/`X4` (a genuine exemption for an
orthogonally-conditioned arm) — it is UNPASSABLE for them by construction,
and adopting a structurally team-flat arm as the final answer would
reproduce exactly the tension the PM ruling (10.0.b) just resolved against
`X1`. They are run and reported in full (they are the pieces `X5r`
consumes, and their own primary/level numbers are the direct test of
whether recency-weighting and in-season adaptation fix round 1's overshoot
at all) but cannot be the round's WINNER. `X5i` (an in-season-as-of variant
of the team arm, paralleling `X5r` but drawing on `X7`'s in-season level
instead of `X6`'s recency level) is NOT attempted this round — flagged as a
stated scope limit for a future round, not attempted informally.

**`X3` shooter-rule variants** (who-shoots, orthogonal to the rate arms
above — every rate arm pairs with every shooter rule, since the technical is
additive and never touches the rate model, section 9.9.1): `X3-avg` (team
attempt-share-weighted average shooter, `X1`'s original assumption),
`X3-best` (team's best as-of shooter that game), `X3-blend` (linear
interpolation `avg + f*(best - avg)`, `f` FITTED per fold on that fold's OWN
TRAINING-season who-shoots data only, then scored out-of-sample on the test
season — round 1's section 4 pooled all four seasons including the test
season into one number, which is leakage for a per-fold decision rule; this
round fixes that). Metric: predicted vs actual technical make rate (pp
gap), test season only, per fold.

**Primary metric, unchanged in form from round 1 (9.4/section 2)**:
predicted vs actual trip count on the test fold's real chance-level
exposure, pooled Poisson deviance of that one aggregate comparison.

**Eligibility line 1 (responsiveness, carried from 9.9.4/3.1, POWERED,
stated in advance): team 2024-prior-season-quintile slope.** Attempted for
`X0`, `X1`, `X5`, `X5r` (team-keyed); not attempted for `X6`/`X7` (not
team-keyed, same mechanical exemption as round 1's `X2`/`X4`, but see the
structural-ineligibility note above — the exemption from the CHECK does not
exempt them from the ADOPTION rule). Power is recomputed fresh on the
verified target's own (larger) counts at run time and reported per fold,
not assumed to match round 1's ~5.5 sigma exactly.

**Eligibility line 2 (level calibration, NEW this round, closes the exact
gap round 1 left open).** An arm's primary-metric predicted pooled rate,
summed over the FULL test season, must be within **10% relative** of the
test season's own realised pooled rate. Round 1's `X1/X2/X4/X5` overshot by
22-25% and this round's job exists specifically because nothing flagged
that — 10% is chosen as roughly half of round 1's own overshoot, tight
enough to bind on `X1` (which is expected to fail it again, unless the
verified target's own multi-season trend happens to be flatter than
`trips_v1_era`'s was) and loose enough not to fail on pure sampling noise
(the measured noise floor, below, is two further orders of magnitude
smaller than 10% of the rate in every fold).

**Noise floor, same convention as round 1 (section 2)**: game-level block
bootstrap SE on the pooled rate, 200 replicates, two seeds (0 and 1). Used
(a) to confirm an arm's primary-metric improvement over `X0`/`X1` exceeds
pure resampling noise, and (b) as the "spec-identical rerun under another
seed" the standing bake-off rule requires before any winner is named.

**Decision rule.** Among arms that are BOTH team-keyed (i.e. eligible for
adoption at all — `X0`, `X1`, `X5`, `X5r`) AND pass BOTH eligibility lines,
the winner is the lowest primary; ties inside the noise floor go to the
simpler model, ordered by parameter count `X0 < X1 < X5 < X5r`. `X6`/`X7`
are reported in full (primary, level-calibration gap, segment cuts) but
cannot win by the structural-ineligibility clause above; if `X5r` is the
only arm that clears BOTH eligibility lines, it wins by elimination, not by
beating a field of eligible competitors, and this is stated plainly rather
than dressed up as a clean sweep. If NO team-keyed arm clears both lines,
**no arm is adopted** and that is an explicitly legitimate outcome, exactly
as round 1's own rule 9.7.6 already provided for.

**Multi-level evidence, unchanged in shape from 9.9.4**: game_phase,
margin_bucket, site, month, conference-game, and the team-quintile power
calculation, all on the test fold, all with underpowered cells labelled
rather than smoothed over.

**What this round may not do, unchanged from 9.8**: no post-hoc multiplier,
cap, clip, offset or blend on sim output; the verified target is not
re-scaled to hit any number; technical attempts stay excluded from FT-2's
training universe (a separate, newly-discovered contamination channel —
`possessions.py`'s bug also LEAKS technical attempts INTO `attempts_v1_era`
under `foul_class in {"foul","none"}`, diluting FT-2's training population
with coach-selected-shooter attempts mislabelled as ordinary trips; flagged
for the engine-core owner alongside the `_handle_technical` fix, NOT
corrected by this lane, which may not touch `src/cbb_sim/` or retrain FT-2).
2025-26 stays sealed throughout.

**Run offline only, one grading script, blind, exactly as round 1 (section
6)**: no engine wiring, no closed-loop paired-seed run this round (the same
concurrency conditions apply — other lanes are running closed loops on this
tree right now under the 2-worker compute cap).

---

### 10.3 Round 1b results: OFFLINE ONLY (run 2026-09-18,
`scripts/grade_ft_technical_round1b_v1.py`; full detail
`docs/tests/free_throw_technicals_round1b_2026-09-18.md`)

`X0`/`X1`/`X6` fail eligibility line 1 (responsiveness) BY CONSTRUCTION
(flat — the F2 monotone-step count read 2/4 on a raw floating-point
comparison of a mathematically-constant sequence before a tolerance guard
was added; fixed so both folds now correctly read 0/4 for all three). `X6`
(recency-weighted training window) reproduces `X1` almost exactly on both
folds — its own internal validation found UNIFORM pooling beats every
recency-weighting candidate, because the anomalous season (2023, still the
single highest-rate season on the VERIFIED target: 2,082 trips against
1,204-1,342 elsewhere) sits in the MIDDLE of the training window, so
recency-weighting it MORE heavily makes the fit worse — the opposite of the
mechanism motivating the arm. `X7` (league in-season as-of rate x site)
closes the level gap dramatically on F2 (predicted/actual rate 0.000689 vs
0.000679, **+1.5%**, against `X1`'s +33.6%) but is not team-keyed and is
pre-declared structurally ineligible for adoption; it also FAILS its own
level-calibration line on F1 (+10.9%, just over the 10% bar), so it is not
uniformly good even on its own metric. `X5`/`X5r` reproduce round 1's team
responsiveness on the verified target (POWERED: quintile power **6.93
sigma** F2 / **9.76 sigma** F1, both above round 1's cited ~5.5 sigma;
monotone 4/4 both folds, slope ratio 1.187 F2 / 0.929 F1) but do not close
the level gap at all (+31.4% F2 / +28.4-29.8% F1, essentially unchanged from
`X1`, because `X5r`'s shrinkage target `X6` barely differs from `X1`'s).

**No arm on either fold is both team-keyed and passing both pre-registered
eligibility lines. NOTHING IS ELIGIBLE. NOTHING IS ADOPTED** — the
pre-registered legitimate outcome (section 10.2, decision rule), reached
because fixing the target (section 10.1) did not by itself resolve round 1's
tension, and neither in-season adaptation nor recency-weighting closes it
alone: one is accurate but cannot be team-keyed, the other is team-keyed but
inherits the training pool's own bad level. `X5i` (a team-as-of rate shrunk
toward `X7`'s in-season level instead of a fixed pooled prior — the
combination that could plausibly clear both lines at once) was named and
explicitly NOT attempted this round (section 10.2's stated scope limit),
and is the clear next candidate.

`X3-blend` (who-shoots, orthogonal to the rate arms, fixed this round to fit
its blend fraction on TRAIN-only data per fold rather than round 1's pooled
all-four-seasons number) is the clear best-supported shooter rule on both
folds (gap **-0.29 pp** F1, **-1.58 pp** F2, vs -8 to +7 pp for the pure
average/best rules), fitted fractions f=0.531 (F1) / 0.514 (F2), close to
round 1's pooled 0.59 but leakage-free and fold-specific. This axis has no
rate arm to pair with this round, since none was adopted.

No served default changed; no `src/cbb_sim/` file was read for editing;
2025-26 stayed sealed throughout (`assert_not_sealed` on every load in both
scripts).
