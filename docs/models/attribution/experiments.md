# L4 PLAYER ATTRIBUTION: experiments

Append-only. Status changes go to `docs/models/change_ledger.md` in the same commit
(`CLAUDE.md`, standing rule "bake-off before any choice").

---

## 1. Pre-registration (authored by the PM, 2026-09-10, BEFORE any modelling)

Targets, each a choice among the relevant five (or a binary plus a choice): (R) rebounder: given a live rebound won by team T, which of T's five secured it (offensive and defensive rebounds modelled separately); (A) assist: given a made FGA by shooter S, (A1) was it assisted (binary) and (A2) if so which of the other four; (S) steal: given a turnover by the offense, (S1) was it a steal (binary) and (S2) which defender; (B) block: given a missed FGA, (B1) was it blocked and (B2) which defender. Universe: D-I, pbp_complete games with complete on-floor ids; F1 train 2024 test 2025 (selection); robustness within-2025 walk-forward (train before Jan 15, test after); 2026 sealed (seal.assert_not_sealed). Players keyed on the CBBD id as in usage.
Pregame inputs (as-of, strictly before game date, shrinkage strength fitted): each player's season-to-date per-possession-on-floor rate for the stat, prior-season rate for returners, position group prior for new players; for the binaries, the team and opponent as-of rates (assist rate, steal rate allowed, block rate) plus the shooter's own as-of assisted share and the shot class.
Arms for each choice: (P1) proportional to the as-of rate normalised over the eligible players; (P2) conditional logit with features (as-of rate, prior-season rate, position group, shot class where relevant, score diff, seconds remaining); (P3) LightGBM grouped-softmax ranker with the same features, parameters searched on 2024 only. Arms for each binary: team-level logistic ridge; shooter/miss-class-aware logistic ridge; LightGBM.
Metrics on F1 test: per target log loss; calibration by predicted-probability decile (<= 2 pp); responsiveness per Decision 8 on the player as-of-rate driver (and on the team driver for the binaries); the game-level checks from usage (re-attribute the ACTUAL event sequence with actual fives: per-player per-game count SD ratio 0.9-1.1, players with >= 1 credited per team-game +/- 0.5, top-1 and top-3 share +/- 2 pp); transfer subset; noise floor (tree seed refits / game-block bootstrap).
Decision rules: per target, winner = lowest log loss among arms passing calibration, responsiveness and the game-level checks; a tree must beat the best passing non-tree arm by more than the floor; ties to the simpler (P1 < P2 < P3; team ridge < aware ridge < tree). If no arm is eligible for a target, adopt nothing and report.

---

## 2. Run state -- **PARTIAL, run interrupted 2026-09-10 16:50 ET** (superseded by section 3)

> **SUPERSEDED BY SECTION 3.** This section is left exactly as it was written at
> the 16:50 ET stop -- it is an honest record of the interrupted run, not a
> result. The bake-off is now COMPLETE; see section 3 for the adopted winners.

> **PARTIAL.** The session had a hard stop. Everything except the RUN is
> complete: `src/cbb_sim/models/attribution.py`, 
> `scripts/train_attribution_v1.py`, `tests/test_attribution.py` (27 passing),
> `model.md`, `features.md`, and the registrations in `docs/models/README.md`
> and `docs/models/change_ledger.md`. The data tables below are BUILT and
> cached. **NO WINNER IS ADOPTED FOR ANY TARGET.** The exact command and state
> needed to finish are in [`RESUME.md`](RESUME.md).

### 2.1 What is done and what is not

| stage | state |
|---|---|
| pre-registration (section 1) | **written before any modelling**, verbatim, PM-authored |
| module, trainer, tests, docs | **complete**; `pytest tests/test_attribution.py -q` = 27 passed |
| population tables + as-of tables (2024 + 2025) | **BUILT and cached** (113 s), see section 2.2 |
| F1 selection fold | **NOT COMPLETED for any target.** `REB_off` was in progress when the stop came (shrinkage fitted, P1 and P2 scored, P3 on its second parameter rung); no target has a full arm table, so none is reported |
| F1 targets NOT completed | `REB_off`, `REB_def`, `assist`, `steal`, `block`, `assisted`, `stolen`, `blocked` |
| within-2025 robustness fold | **NOT RUN** (launched with `--skip-wf`) |
| composed per-player diagnostic | **NOT RUN** (launched with `--skip-composed`); reported-only, never a gate |
| noise floors | the launch used `--lgbm-seeds 2 --boot-reps 100 --sim-draws 20` rather than the pre-registered 3 / 200 / 40, so any floor from it is looser than the contract and must not be carried into a decision |

The launch command actually used was:

```
scripts/train_attribution_v1.py --version v2 --skip-wf --skip-composed \
    --lgbm-grid 2 --lgbm-seeds 2 --sim-draws 20 --boot-reps 100
```

### 2.2 Build configuration

| item | value |
|---|---|
| trainer | `scripts/train_attribution_v1.py` |
| possessions version | `v2` (rim override 2.27 ft) |
| universe | D-I, non-truncated, `pbp_complete` |
| population rows (2024 + 2025) | reb_off 222,775, reb_def 526,310, made_fga 546,514, tov 250,065, miss_fga 689,375 |
| player-games with as-of inputs | 206,169 |
| team-games with as-of inputs | 21,246 |
| roster position known | 99.9791% |
| hoopR minutes joined through the player crosswalk | 99.8167% |
| player-games with a prior season of history | 2024: 0.0%, 2025: 68.4848% |

### 2.3 Target coverage (reported, not silently filtered)

| season | target | population | binary rate | candidate set resolved | credited id present | modelled |
|---|---|---:|---:|---:|---:|---:|
| 2024 | REB_off | 107,768 | 100.0% | 95.7724% | 82.2517% | 77.8441% |
| 2024 | REB_def | 259,639 | 100.0% | 96.0079% | 92.7819% | 87.9833% |
| 2024 | assist | 267,004 | 50.5764% | 96.1501% | 99.8519% | 94.3595% |
| 2024 | steal | 121,637 | 54.9274% | 95.637% | 100.0% | 94.5175% |
| 2024 | block | 337,450 | 10.0939% | 95.837% | 100.0% | 94.924% |
| 2024 | assisted | 267,004 | 50.5764% | 96.1323% | 99.9251% | 96.1323% |
| 2024 | stolen | 121,637 | 54.9274% | 95.7168% | 100.0% | 95.7168% |
| 2024 | blocked | 337,450 | 10.0939% | 96.0036% | 100.0% | 96.0036% |
| 2025 | REB_off | 115,007 | 100.0% | 99.1157% | 82.5976% | 80.8594% |
| 2025 | REB_def | 266,671 | 100.0% | 99.1581% | 92.7382% | 90.8989% |
| 2025 | assist | 279,510 | 51.4822% | 99.2043% | 99.8436% | 97.7686% |
| 2025 | steal | 128,428 | 56.5383% | 99.0428% | 99.9959% | 97.886% |
| 2025 | block | 351,925 | 9.9084% | 99.1368% | 99.9971% | 98.1675% |
| 2025 | assisted | 279,510 | 51.4822% | 99.1778% | 99.9195% | 99.1778% |
| 2025 | stolen | 128,428 | 56.5383% | 99.0711% | 99.9977% | 99.0711% |
| 2025 | blocked | 351,925 | 9.9084% | 99.1609% | 99.9997% | 99.1609% |

Three coverage facts are load-bearing and are recorded rather than smoothed:

1. **`REB_off` has the lowest coverage of the eight targets (~81%).** CBBD
   leaves `participant_1_id` blank on 18-25% of `Offensive Rebound` rows and
   ~7% of `Defensive Rebound` rows -- the feed records a TEAM rebound with no
   player. Those rows cannot enter a who-did-it model and are dropped with the
   count reported, never imputed (the treatment `usage` gives team turnovers).
2. **The assist population is only modellable because the shooter is read off
   `shot_shooter_id`.** With `participant_1_id` -- which `usage` uses for its
   own shooter -- only 51% of assisted field goals were modellable, because on
   48.9% of assisted rows `participant_1_id` IS THE ASSISTER and the wrong man
   was being removed from the choice set. After the fix, 98.7%. This is a data
   defect with scope beyond this model and is logged in the change ledger.
3. **The `assist` choice has FOUR alternatives, not five** (uniform log loss
   1.386294, against 1.609438 for the other four choice targets), so its log
   loss is not comparable to theirs without that baseline in view.

### 2.4 Partial numbers from the interrupted `REB_off` fit

Reported for continuity only. These are NOT a result: the arm table is
incomplete, the tree had not finished, and no gate has been evaluated. Nothing
here may be quoted as a comparison.

| quantity | value |
|---|---|
| train / test events | 83,891 / 92,994 |
| fitted shrinkage | prior `position`, m = 25 pseudo opportunities (train log loss 1.426268) |
| `prior_season` rung | **unidentified on F1** (L13: no 2023 on-floor ids), reported not scored |
| P2 dropped as unidentified | `log_prior_share`, `prior_rate` |
| P2 (conditional logit) test log loss | 1.431192 at l2 = 0.001 |
| P3 inner-validation log loss, rung 1 | 1.408099 (`num_leaves=15, lr=0.08, n=300, min_child=200`) |
| uniform-over-five baseline | 1.609438 |

### 2.5 Why the run did not finish, and what to change

P3 is the bottleneck. The grouped-softmax objective is a Python callback
invoked once per boosting iteration over K x n rows, and this model's
populations are the largest in the project (`miss_fga` 689,375 rows,
`made_fga` 546,514, `reb_def` 526,310 across the two seasons). Measured here:
about two minutes per parameter rung on `REB_off`, the SMALLEST choice target.
[`RESUME.md`](RESUME.md) lists the three levers -- cut the grid to its first
rung (which won the inner split here and is the rung `usage` adopted on all
five of its classes), subsample whole games for the parameter search only, or
compile the objective -- none of which changes what is compared. It also warns
against dropping below two LightGBM seeds, because the tree's noise floor IS
the seed-varied refit SD.

---

## 3. Final results -- **BAKE-OFF COMPLETE 2026-09-10 evening**

Run to completion with the exact pre-registered command
(`scripts/train_attribution_v1.py --version v2 --rebuild`, no reduced flags):
both folds, all eight targets, the full three-rung LightGBM grid, three seed
refits on F1 (one on the within-2025 fold, which reuses F1's chosen params
rather than re-searching, per the pre-registration's "parameters searched on
2024 only"), 40 game-level draws, 200 bootstrap reps. **No deviation lever was
needed**: total wall time was 682.4 s (about 11.4 minutes), far under the
~5-hour threshold that would have triggered a lever from `RESUME.md`. The
`--rebuild` flag (not itself a deviation -- it forces the cached tables to
regenerate) was necessary because of the data fix in section 3.1 below.

### 3.1 A second defect, found while resuming: `build_team_asof`'s league-rate columns were zero for all three binaries

Before this run, the cached `team_asof_v2.parquet` had `off_*_lg` / `def_*_lg`
NaN on **100% of rows for `assisted`, `stolen` and `blocked` alike** -- not an
edge case. `build_team_asof` accumulated both seasons' six (population x side)
count groups into one flat list and merged them all sequentially; since 2024's
and 2025's groups share the same column names (`made`, `ast`, ...), pandas
silently suffixed every one of them `_x`/`_y`, the bare names never existed, a
defensive `if c not in tg.columns: tg[c] = 0.0` fallback manufactured an
all-zero column for literally every count (not just the league rate -- the
per-team `num`/`den` were silently zero too, which is the more dangerous half:
had the run not crashed on the `NaN`, the team-level features would have been
uninformative rather than erroring), and `_safe_div(0, 0)` came out `NaN`
everywhere. This is exactly why the interrupted run's three binaries failed
with `ValueError: Input X contains NaN` in `LogisticRegression`.

Fixed by merging each season's six groups column-wise (safe -- unique names
within a season) and concatenating the two seasons row-wise; no other line of
`build_team_asof` changed. Full reproduction, before/after column statistics
and scope: `docs/tests/attribution_team_asof_lg_defect_2026-09-10.md`.
`pytest tests/test_attribution.py -q` is unaffected (27 passed before and
after -- none of the 27 tests exercised the multi-season path, a gap noted but
not closed here to stay in scope). This is a construction-bug fix, not a
modelling choice, per `CLAUDE.md`'s "no hand tuning" rule: it changes nothing
about what is compared, only makes the binaries' team features real instead of
silently zero.

### 3.2 A pre-existing gap surfaced by this run: the composed diagnostic never runs

`--skip-composed` exists as a CLI flag and `--composed-draws` exists as a
parameter, but `args.skip_composed` is **never read** anywhere in
`scripts/train_attribution_v1.py`, and `attribution.composed_check` (the
function itself is implemented and tested) is **never called** from the
trainer's per-target loop. So the composed per-player diagnostic did not run
in the interrupted launch NOR in this completed run -- not because either run
skipped it, but because the trainer never wires it in regardless of the flag.
This is a gap in the trainer script, found while completing the bake-off, not
introduced by this run. It does not affect any adopted winner (the composed
check was always reported-only, never a gate, per the pre-registration), but
is flagged here rather than silently left unmentioned, and is a followup for
whoever next touches the trainer.

### 3.3 Per-target results, both folds

**Choice targets** (winner in **bold**; floor = the decision rule's own
noise-floor SD, i.e. `max(bootstrap SE of eligible arms, LightGBM seed-refit
SD)`; slope = the winning arm's own responsiveness slope ratio on the as-of
rate driver, 4/4 monotone quintile steps on every eligible arm in every row
below unless noted):

| target (K) | fold | winner | log loss | floor SD | slope | decision reason |
|---|---|---|---:|---:|---:|---|
| REB_off (5) | F1 | **cond_logit** | 1.431192 | 0.002202 | 0.994 | tree leads by 0.000304, inside the floor; non-tree arms tie, simplest wins |
| REB_off (5) | WF2025 | **cond_logit** | 1.414486 | 0.003147 | 1.006 | cond_logit/lgbm tie inside floor; simplest wins |
| REB_def (5) | F1 | **lgbm** | 1.559167 | 0.000759 | 1.003 | clear of next eligible arm by 1.5 floors |
| REB_def (5) | WF2025 | **cond_logit** | 1.556737 | 0.001133 | 1.022 | cond_logit/lgbm tie inside floor; simplest wins |
| assist (4) | F1 | **NO WINNER** | lgbm best at 1.292465 | 0.001450 | 1.003 | lgbm calibrates (0.418pp) but fails top1/top3 (-2.18pp); every arm fails top1/top3 |
| assist (4) | WF2025 | **lgbm** | 1.283376 | 0.001965 | 0.951 | only eligible arm (proportional, cond_logit fail calibration + top1/top3) |
| steal (5) | F1 | **proportional** | 1.586249 | 0.000954 | 0.997 | proportional/cond_logit tie inside floor; simplest wins |
| steal (5) | WF2025 | **proportional** | 1.582525 | 0.001479 | 1.016 | proportional/cond_logit tie inside floor; simplest wins |
| block (5) | F1 | **lgbm** | 1.273376 | 0.004718 | 1.000 | only eligible arm (proportional, cond_logit fail calibration + top1/top3) |
| block (5) | WF2025 | **NO WINNER** | lgbm best at 1.250223 | -- | 1.005 | every arm fails calibration (2.16-4.62pp); test set is only 17,234 rows -- **flagged UNDERPOWERED, not a reversal of the F1 result** |

**Binary targets** (slopes reported off / def; floor as above):

| target | fold | winner | log loss | floor SD | slopes (off/def) | decision reason |
|---|---|---|---:|---:|---|---|
| assisted | F1 | **aware_ridge** | 0.558057 | 0.000998 | 1.082 / 0.956 | aware_ridge/lgbm tie inside floor; simplest wins |
| assisted | WF2025 | **NO WINNER** | aware_ridge best at 0.562365 | -- | 0.877 / 0.915 | every arm fails calibration (4.15-6.74pp); test set 141,639 rows, NOT underpowered -- a real within-season calibration miss, reported as such |
| stolen | F1 | **NO WINNER** | lgbm best at 0.642243 | -- | -- | every arm fails calibration (2.53-6.85pp); lgbm additionally fails responsiveness (3/4 steps) |
| stolen | WF2025 | **NO WINNER** | lgbm best at 0.653014 | -- | -- | every arm fails calibration AND responsiveness |
| blocked | F1 | **aware_ridge** | 0.264337 | 0.001217 | 1.076 / 1.015 | clear of next eligible arm (team_ridge) by 46.4 floors |
| blocked | WF2025 | **aware_ridge** | 0.264498 | 0.001558 | 0.920 / 1.095 | only eligible arm |

**Reading the pattern across folds**: five of eight targets carry the SAME
winner (or NO WINNER) on both F1 and the within-2025 robustness fold
(`REB_def`'s winner moves lgbm -> cond_logit but both are inside each other's
floor on F1 already, so this is not a reversal); `assist` and `block` flip
direction between folds in a way that is a genuine finding, not noise -- see
3.3.1. `stolen` fails calibration on every arm, on both folds, which is the
one clean "this credit is not usably modellable with the pre-registered
feature set" result in the whole bake-off.

### 3.3.1 Responsiveness slope check (Decision 8, per target)

Every eligible arm on every target and every fold passes 4/4 monotone
quintile steps on its own as-of-rate driver (choice targets) or on both the
offence and defence as-of-rate drivers (binaries) -- see the `resp steps`
columns in `report_v1.md` sections 3 and 4, none below 4/4. The slope ratios
(realised-vs-predicted quintile span) are close to 1.0 across the board
(0.92-1.03 on the winning choice arms, 0.86-1.11 on the winning binary arms),
so responsiveness is not the binding gate anywhere in this bake-off -- the
same shape `usage` and `fg_make` found: calibration and the top1/top3
game-level check are what fail an arm here, not flatness against the driver.
No target shows the CFB-style "flat at the mean" failure Decision 8 was
written to catch.

### 3.4 The team-rebound blank-rate caveat (repeated here, never imputed)

`REB_off` trains on the lowest-coverage population of the eight targets
because CBBD leaves `participant_1_id` blank on **18-25% of `Offensive
Rebound` rows** and about **7% of `Defensive Rebound` rows** -- the feed
records a TEAM rebound with no player. Per-season, the modelled share is
77.84% (2024) / 80.86% (2025) of `REB_off` and 87.98% (2024) / 90.90% (2025)
of `REB_def` (section 2.3 above). These rows are dropped with the count
reported by `attribution.coverage_report`, exactly the same treatment `usage`
gives team turnovers -- **never imputed**. This is a data-source ceiling, not
a modelling choice, and does not change between arms or between folds.

### 3.5 Full per-arm tables

The complete arm-by-arm tables (Brier, top-1/top-3, calibration worst gap,
transfer/continuing/no-prior-season subsets, LightGBM feature importances and
seed-refit lists, conditional-logit coefficients) are in
`data/processed/models/attribution/report_v1.md`, copied verbatim below.

<!-- BEGIN report_v1.md verbatim -->

## 2. Run configuration (2026-09-10)

| item | value |
|---|---|
| trainer | `scripts/train_attribution_v1.py` |
| possessions version | `v2` (rim override 2.27 ft) |
| universe | D-I, non-truncated, `pbp_complete` |
| population rows (2024 + 2025) | reb_off 222,775, reb_def 526,310, made_fga 546,514, tov 250,065, miss_fga 689,375 |
| player-games with as-of inputs | 206,169 |
| team-games with as-of inputs | 21,246 |
| roster position known | 99.9791% |
| hoopR minutes joined through the player crosswalk | 99.8167% |
| player-games with a prior season of history | 2024: 0.0%, 2025: 68.4848% |
| game-level / composed draws, bootstrap reps | 40 / 20 / 200 |

### 2.1 Target coverage (reported, not silently filtered)

| season | target | population | binary rate | candidate set resolved | credited id present | modelled |
|---|---|---:|---:|---:|---:|---:|
| 2024 | REB_off | 107,768 | 100.0% | 95.7724% | 82.2517% | 77.8441% |
| 2024 | REB_def | 259,639 | 100.0% | 96.0079% | 92.7819% | 87.9833% |
| 2024 | assist | 267,004 | 50.5764% | 96.1501% | 99.8519% | 94.3595% |
| 2024 | steal | 121,637 | 54.9274% | 95.637% | 100.0% | 94.5175% |
| 2024 | block | 337,450 | 10.0939% | 95.837% | 100.0% | 94.924% |
| 2024 | assisted | 267,004 | 50.5764% | 96.1323% | 99.9251% | 96.1323% |
| 2024 | stolen | 121,637 | 54.9274% | 95.7168% | 100.0% | 95.7168% |
| 2024 | blocked | 337,450 | 10.0939% | 96.0036% | 100.0% | 96.0036% |
| 2025 | REB_off | 115,007 | 100.0% | 99.1157% | 82.5976% | 80.8594% |
| 2025 | REB_def | 266,671 | 100.0% | 99.1581% | 92.7382% | 90.8989% |
| 2025 | assist | 279,510 | 51.4822% | 99.2043% | 99.8436% | 97.7686% |
| 2025 | steal | 128,428 | 56.5383% | 99.0428% | 99.9959% | 97.886% |
| 2025 | block | 351,925 | 9.9084% | 99.1368% | 99.9971% | 98.1675% |
| 2025 | assisted | 279,510 | 51.4822% | 99.1778% | 99.9195% | 99.1778% |
| 2025 | stolen | 128,428 | 56.5383% | 99.0711% | 99.9977% | 99.0711% |
| 2025 | blocked | 351,925 | 9.9084% | 99.1609% | 99.9997% | 99.1609% |

## 3'. F1 results (train 2024, test 2025) -- the selection fold

### Choice targets

**REB_off** (K = 5, uniform log loss 1.609438). Train 83,891, test 92,994. Fitted shrinkage: prior `position`, m = 25 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.434713 | 0.7278 | 0.3878 | 0.8114 | 1.773 | 4/4 | 0.935 | 0.002067 | 1.0078 | 4.828 / 4.664 | 38.45% / 39.96% | -1.89 | yes |
| cond_logit | 1.431192 | 0.7257 | 0.3885 | 0.8116 | 0.707 | 4/4 | 0.994 | 0.002192 | 1.0059 | 4.782 / 4.664 | 38.99% / 39.96% | -1.37 | yes |
| lgbm | 1.430888 | 0.7257 | 0.3874 | 0.8113 | 0.568 | 4/4 | 0.999 | 0.002202 | 1.0061 | 4.778 / 4.664 | 39.07% / 39.96% | -1.31 | yes |

**Decision: cond_logit.** the tree leads by 0.000304, inside the 0.002202 floor, so the pre-registration's requirement that a tree beat the best passing non-tree arm by more than the floor is not met; among the non-tree arms cond_logit are inside the floor of each other and the tie-break takes the simplest

P2 ridge penalty 0.001 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `[]`. Coefficients: `{'log_share': 1.08896, 'is_G': 0.01302, 'is_F': -0.03345, 'is_C': 0.01192, 'log_share_x_scorediff': 0.0153, 'log_share_x_sec': 0.02992, 'log_share_x_rim': 0.13292, 'log_share_x_three': -0.1516, 'log_share_x_ft': -0.26221, 'is_C_x_scorediff': 0.01455, 'is_C_x_sec': -0.01394, 'is_C_x_rim': 0.11804, 'is_C_x_three': -0.12822, 'is_C_x_ft': -0.41392}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.430888, 1.430953, 1.430952], seed SD 3.7e-05; feature importance `{'share': 863, 'rate': 785, 'position_code': 107, 'rate_rank': 100, 'score_diff': 752, 'sec_remaining': 1228, 'shot_class_code': 365}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.416133 | 1.436831 | 1.453856 |
| cond_logit | 1.411731 | 1.434530 | 1.449261 |
| lgbm | 1.411069 | 1.434009 | 1.449782 |

(n transfers = 28,612)

**REB_def** (K = 5, uniform log loss 1.609438). Train 228,439, test 242,401. Fitted shrinkage: prior `position`, m = 50 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.565561 | 0.7819 | 0.2898 | 0.7063 | 0.717 | 4/4 | 0.984 | 0.000690 | 0.9851 | 7.641 / 7.464 | 27.53% / 29.05% | -1.89 | yes |
| cond_logit | 1.560274 | 0.7798 | 0.2899 | 0.7065 | 0.426 | 4/4 | 1.004 | 0.000759 | 0.9857 | 7.635 / 7.464 | 27.64% / 29.05% | -1.79 | yes |
| lgbm | 1.559167 | 0.7793 | 0.2914 | 0.7077 | 0.282 | 4/4 | 1.003 | 0.000756 | 0.9857 | 7.636 / 7.464 | 27.70% / 29.05% | -1.77 | yes |

**Decision: lgbm.** eligible: lgbm 1.559167, cond_logit 1.560274, proportional 1.565561; floor 0.000759; clear of the next eligible arm by 0.001107 (1.5 floors)

P2 ridge penalty 1.0 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `[]`. Coefficients: `{'log_share': 1.13889, 'is_G': -0.2941, 'is_F': -0.29702, 'is_C': -0.26453, 'log_share_x_scorediff': 0.00384, 'log_share_x_sec': 0.01389, 'log_share_x_rim': -0.04848, 'log_share_x_three': -0.49661, 'log_share_x_ft': 0.7817, 'is_C_x_scorediff': 0.0054, 'is_C_x_sec': -0.01086, 'is_C_x_rim': 0.12396, 'is_C_x_three': -0.06439, 'is_C_x_ft': -0.05226}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.559167, 1.559195, 1.558969], seed SD 0.000123; feature importance `{'share': 765, 'rate': 740, 'position_code': 232, 'rate_rank': 104, 'score_diff': 791, 'sec_remaining': 1113, 'shot_class_code': 455}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.553335 | 1.562436 | 1.586914 |
| cond_logit | 1.548215 | 1.557233 | 1.581259 |
| lgbm | 1.547855 | 1.555761 | 1.579871 |

(n transfers = 75,197)

**assist** (K = 4, uniform log loss 1.386294). Train 127,424, test 140,687. Fitted shrinkage: prior `position`, m = 10 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.295860 | 0.7020 | 0.4034 | 0.8549 | 2.779 | 4/4 | 0.955 | 0.001384 | 1.0101 | 5.858 / 5.602 | 35.26% / 37.83% | -2.66 | NO (calibration, top1_top3) |
| cond_logit | 1.294558 | 0.7012 | 0.4043 | 0.8556 | 1.927 | 4/4 | 0.994 | 0.001419 | 1.0087 | 5.826 / 5.602 | 35.59% / 37.83% | -2.32 | NO (top1_top3) |
| lgbm | 1.292465 | 0.7003 | 0.4045 | 0.8556 | 0.418 | 4/4 | 1.003 | 0.001450 | 1.0093 | 5.828 / 5.602 | 36.02% / 37.83% | -2.20 | NO (top1_top3) |

**Decision: NO WINNER.** no arm passes every pre-registered gate

P2 ridge penalty 0.001 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `['log_share_x_ft', 'is_C_x_ft']`. Coefficients: `{'log_share': 1.04163, 'is_G': 0.02047, 'is_F': 0.00504, 'is_C': -0.0677, 'log_share_x_scorediff': 0.00411, 'log_share_x_sec': -0.04023, 'log_share_x_rim': 0.06436, 'log_share_x_three': 0.10737, 'is_C_x_scorediff': -0.00471, 'is_C_x_sec': 0.03603, 'is_C_x_rim': 0.26319, 'is_C_x_three': -0.20221}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.292465, 1.292469, 1.29255], seed SD 4.8e-05; feature importance `{'share': 921, 'rate': 857, 'position_code': 191, 'rate_rank': 80, 'score_diff': 758, 'sec_remaining': 1148, 'shot_class_code': 245}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.283143 | 1.286535 | 1.331516 |
| cond_logit | 1.281088 | 1.285467 | 1.330766 |
| lgbm | 1.278051 | 1.284202 | 1.328308 |

(n transfers = 43,860)

**steal** (K = 5, uniform log loss 1.609438). Train 63,149, test 71,076. Fitted shrinkage: prior `position`, m = 50 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.586249 | 0.7904 | 0.2626 | 0.6791 | 0.814 | 4/4 | 0.997 | 0.000908 | 1.0104 | 4.317 / 4.231 | 39.63% / 40.48% | -1.09 | yes |
| cond_logit | 1.586254 | 0.7904 | 0.2625 | 0.6790 | 0.998 | 4/4 | 1.014 | 0.000892 | 1.0104 | 4.315 / 4.231 | 39.66% / 40.48% | -1.07 | yes |
| lgbm | 1.588532 | 0.7913 | 0.2608 | 0.6760 | 1.367 | 4/4 | 1.021 | 0.000954 | 1.0110 | 4.312 / 4.231 | 39.72% / 40.48% | -1.03 | yes |

**Decision: proportional.** eligible: proportional 1.586249, cond_logit 1.586254, lgbm 1.588532; floor 0.000954; proportional, cond_logit are inside the floor of each other; the pre-registered tie-break takes the simplest

P2 ridge penalty 10.0 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `['log_share_x_rim', 'log_share_x_three', 'log_share_x_ft', 'is_C_x_rim', 'is_C_x_three', 'is_C_x_ft']`. Coefficients: `{'log_share': 0.95072, 'is_G': -0.08653, 'is_F': -0.08605, 'is_C': -0.10983, 'log_share_x_scorediff': 0.01871, 'log_share_x_sec': 0.03284, 'is_C_x_scorediff': -0.04216, 'is_C_x_sec': 0.01773}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.588532, 1.588392, 1.588602], seed SD 0.000107; feature importance `{'share': 994, 'rate': 907, 'position_code': 96, 'rate_rank': 131, 'score_diff': 828, 'sec_remaining': 1244, 'shot_class_code': 0}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.582536 | 1.582656 | 1.597363 |
| cond_logit | 1.582101 | 1.582684 | 1.597876 |
| lgbm | 1.584454 | 1.584466 | 1.600953 |

(n transfers = 21,947)

**block** (K = 5, uniform log loss 1.609438). Train 32,333, test 34,231. Fitted shrinkage: prior `position`, m = 10 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.284843 | 0.6535 | 0.4936 | 0.8591 | 4.937 | 4/4 | 0.939 | 0.004514 | 1.0453 | 2.503 / 2.393 | 59.10% / 61.55% | -0.91 | NO (calibration, top1_top3) |
| cond_logit | 1.273338 | 0.6472 | 0.4946 | 0.8592 | 2.422 | 4/4 | 0.983 | 0.004768 | 1.0369 | 2.482 / 2.393 | 59.56% / 61.55% | -0.72 | NO (calibration) |
| lgbm | 1.273376 | 0.6469 | 0.4956 | 0.8588 | 0.360 | 4/4 | 1.000 | 0.004718 | 1.0392 | 2.470 / 2.393 | 59.94% / 61.55% | -0.66 | yes |

**Decision: lgbm.** eligible: lgbm 1.273376; floor 0.004718; the only eligible arm

P2 ridge penalty 10.0 (searched on `2024-01-15`); dropped as unidentified: `['log_prior_share', 'prior_rate']`; dropped as constant in this fold: `['log_share_x_ft', 'is_C_x_ft']`. Coefficients: `{'log_share': 0.99326, 'is_G': -0.07497, 'is_F': -0.06965, 'is_C': 0.15451, 'log_share_x_scorediff': 0.02672, 'log_share_x_sec': -0.01258, 'log_share_x_rim': 0.15517, 'log_share_x_three': -0.48433, 'is_C_x_scorediff': 0.00112, 'is_C_x_sec': -0.02278, 'is_C_x_rim': -0.01539, 'is_C_x_three': -0.42806}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.273376, 1.273084, 1.273295], seed SD 0.000151; feature importance `{'share': 1036, 'rate': 904, 'position_code': 75, 'rate_rank': 82, 'score_diff': 745, 'sec_remaining': 1144, 'shot_class_code': 214}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.299108 | 1.276291 | 1.283285 |
| cond_logit | 1.286514 | 1.264500 | 1.273580 |
| lgbm | 1.289596 | 1.264353 | 1.270351 |

(n transfers = 10,189)

### Binary targets

| target | base rate | arm | log loss | Brier | calib worst (pp) | resp steps | slopes (off / def) | boot SE | count (sim / real) | count SD ratio | eligible |
|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|---|
| assisted | 51.496% | team_ridge | 0.684722 | 0.245813 | 2.101 | 4/4 | 1.039 / 0.974 | 0.000256 | 12.899 / 13.129 | 0.9290 | NO (calibration) |
| assisted | 51.496% | aware_ridge | 0.558057 | 0.189083 | 1.954 | 4/4 | 1.082 / 0.956 | 0.000965 | 13.047 / 13.129 | 0.9356 | yes |
| assisted | 51.496% | lgbm | 0.558261 | 0.189120 | 1.843 | 4/4 | 1.051 / 0.958 | 0.000998 | 13.055 / 13.129 | 0.9399 | yes |
| stolen | 56.5222% | team_ridge | 0.682932 | 0.244926 | 2.525 | 4/4 | 1.067 / 1.019 | 0.000369 | 6.425 / 6.614 | 0.9503 | NO (calibration) |
| stolen | 56.5222% | aware_ridge | 0.681489 | 0.244231 | 4.334 | 4/4 | 1.077 / 1.015 | 0.000420 | 6.423 / 6.614 | 0.9495 | NO (calibration) |
| stolen | 56.5222% | lgbm | 0.642243 | 0.228656 | 6.850 | 3/4 | 1.194 / 0.975 | 0.000890 | 6.398 / 6.614 | 0.9550 | NO (calibration) |
| blocked | 9.906% | team_ridge | 0.320757 | 0.088842 | 0.538 | 4/4 | 1.074 / 1.025 | 0.001217 | 3.238 / 3.179 | 0.9457 | yes |
| blocked | 9.906% | aware_ridge | 0.264337 | 0.078387 | 0.295 | 4/4 | 1.076 / 1.015 | 0.001088 | 3.187 / 3.179 | 0.9740 | yes |
| blocked | 9.906% | lgbm | 0.266217 | 0.078939 | 3.017 | 4/4 | 1.023 / 1.026 | 0.001117 | 3.189 / 3.179 | 0.9809 | NO (calibration) |

**assisted decision: aware_ridge.** eligible: aware_ridge 0.558057, lgbm 0.558261; floor 0.000998; aware_ridge, lgbm are inside the floor of each other; the pre-registered tie-break takes the simplest Fitted shrinkage: m_team 100, own-share prior `position` m_own 10.

**stolen decision: NO WINNER.** no arm passes every pre-registered gate Fitted shrinkage: m_team 200, own-share prior `position` m_own 25.

**blocked decision: aware_ridge.** eligible: aware_ridge 0.264337, team_ridge 0.320757; floor 0.001217; clear of the next eligible arm by 0.056420 (46.4 floors) Fitted shrinkage: m_team 200, own-share prior `position` m_own 50.

## 4'. Robustness fold: within-2025 walk-forward (train before 2025-01-15, test after)

### Choice targets

**REB_off** (K = 5, uniform log loss 1.609438). Train 46,579, test 46,415. Fitted shrinkage: prior `position`, m = 25 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.417909 | 0.7213 | 0.3972 | 0.8222 | 1.807 | 4/4 | 0.943 | 0.002857 | 1.0005 | 4.657 / 4.526 | 39.69% / 40.93% | -1.51 | yes |
| cond_logit | 1.414486 | 0.7191 | 0.3977 | 0.8230 | 0.864 | 4/4 | 1.006 | 0.003079 | 0.9954 | 4.602 / 4.526 | 40.34% / 40.93% | -0.90 | yes |
| lgbm | 1.415901 | 0.7199 | 0.3974 | 0.8231 | 0.844 | 4/4 | 0.988 | 0.003147 | 0.9952 | 4.597 / 4.526 | 40.52% / 40.93% | -0.82 | yes |

**Decision: cond_logit.** eligible: cond_logit 1.414486, lgbm 1.415901, proportional 1.417909; floor 0.003147; cond_logit, lgbm are inside the floor of each other; the pre-registered tie-break takes the simplest

P2 ridge penalty 10.0 (searched on `2024-12-06`); dropped as unidentified: `[]`; dropped as constant in this fold: `[]`. Coefficients: `{'log_share': 1.08404, 'log_prior_share': 0.01516, 'is_G': -0.02558, 'is_F': -0.07046, 'is_C': 0.06782, 'log_share_x_scorediff': 0.01972, 'log_share_x_sec': 0.02751, 'log_share_x_rim': 0.11492, 'log_share_x_three': -0.18013, 'log_share_x_ft': -0.28033, 'is_C_x_scorediff': 0.01225, 'is_C_x_sec': -0.00041, 'is_C_x_rim': 0.00929, 'is_C_x_three': -0.15507, 'is_C_x_ft': -0.36006}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.415901], seed SD None; feature importance `{'share': 751, 'rate': 800, 'prior_rate': 869, 'position_code': 61, 'rate_rank': 91, 'score_diff': 610, 'sec_remaining': 765, 'shot_class_code': 253}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.386487 | 1.425330 | 1.443226 |
| cond_logit | 1.384803 | 1.423609 | 1.434613 |
| lgbm | 1.396494 | 1.420073 | 1.432274 |

(n transfers = 14,184)

**REB_def** (K = 5, uniform log loss 1.609438). Train 119,713, test 122,688. Fitted shrinkage: prior `position`, m = 50 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.561747 | 0.7803 | 0.2943 | 0.7123 | 0.528 | 4/4 | 0.998 | 0.001005 | 0.9734 | 7.393 / 7.249 | 28.35% / 29.62% | -1.57 | yes |
| cond_logit | 1.556737 | 0.7784 | 0.2939 | 0.7120 | 0.462 | 4/4 | 1.022 | 0.001104 | 0.9733 | 7.386 / 7.249 | 28.47% / 29.62% | -1.46 | yes |
| lgbm | 1.556970 | 0.7786 | 0.2937 | 0.7109 | 1.350 | 4/4 | 1.031 | 0.001133 | 0.9728 | 7.375 / 7.249 | 28.70% / 29.62% | -1.25 | yes |

**Decision: cond_logit.** eligible: cond_logit 1.556737, lgbm 1.556970, proportional 1.561747; floor 0.001133; cond_logit, lgbm are inside the floor of each other; the pre-registered tie-break takes the simplest

P2 ridge penalty 10.0 (searched on `2024-12-06`); dropped as unidentified: `[]`; dropped as constant in this fold: `[]`. Coefficients: `{'log_share': 1.15011, 'log_prior_share': 0.01999, 'is_G': -0.0126, 'is_F': 0.0057, 'is_C': 0.02399, 'log_share_x_scorediff': 0.00598, 'log_share_x_sec': 0.00056, 'log_share_x_rim': -0.04882, 'log_share_x_three': -0.52324, 'log_share_x_ft': 0.69615, 'is_C_x_scorediff': 0.02238, 'is_C_x_sec': 0.01123, 'is_C_x_rim': 0.11251, 'is_C_x_three': -0.08803, 'is_C_x_ft': -0.10715}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.55697], seed SD None; feature importance `{'share': 688, 'rate': 678, 'prior_rate': 830, 'position_code': 114, 'rate_rank': 89, 'score_diff': 637, 'sec_remaining': 795, 'shot_class_code': 369}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.545667 | 1.558084 | 1.588563 |
| cond_logit | 1.542210 | 1.553494 | 1.580841 |
| lgbm | 1.536930 | 1.553783 | 1.587846 |

(n transfers = 37,401)

**assist** (K = 4, uniform log loss 1.386294). Train 70,329, test 70,358. Fitted shrinkage: prior `position`, m = 10 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.283382 | 0.6956 | 0.4149 | 0.8622 | 3.236 | 4/4 | 0.928 | 0.001792 | 0.9923 | 5.659 / 5.459 | 36.37% / 38.57% | -2.10 | NO (calibration, top1_top3) |
| cond_logit | 1.282563 | 0.6952 | 0.4170 | 0.8632 | 3.178 | 4/4 | 0.926 | 0.001831 | 0.9923 | 5.653 / 5.459 | 36.39% / 38.57% | -2.04 | NO (calibration, top1_top3) |
| lgbm | 1.283376 | 0.6953 | 0.4167 | 0.8596 | 1.403 | 4/4 | 0.951 | 0.001965 | 0.9910 | 5.616 / 5.459 | 37.37% / 38.57% | -1.50 | yes |

**Decision: lgbm.** eligible: lgbm 1.283376; floor 0.001965; the only eligible arm

P2 ridge penalty 0.1 (searched on `2024-12-07`); dropped as unidentified: `[]`; dropped as constant in this fold: `['log_share_x_ft', 'is_C_x_ft']`. Coefficients: `{'log_share': 1.05774, 'log_prior_share': 0.01741, 'is_G': -1.09031, 'is_F': -1.13487, 'is_C': -1.14833, 'log_share_x_scorediff': 0.00979, 'log_share_x_sec': -0.04561, 'log_share_x_rim': -0.04536, 'log_share_x_three': 0.02909, 'is_C_x_scorediff': 0.00402, 'is_C_x_sec': 0.03185, 'is_C_x_rim': 0.23451, 'is_C_x_three': -0.26217}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.283376], seed SD None; feature importance `{'share': 761, 'rate': 741, 'prior_rate': 907, 'position_code': 127, 'rate_rank': 83, 'score_diff': 561, 'sec_remaining': 797, 'shot_class_code': 223}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.265965 | 1.272902 | 1.326435 |
| cond_logit | 1.268096 | 1.273805 | 1.318427 |
| lgbm | 1.255797 | 1.278638 | 1.328470 |

(n transfers = 21,548)

**steal** (K = 5, uniform log loss 1.609438). Train 35,709, test 35,367. Fitted shrinkage: prior `position`, m = 50 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.582525 | 0.7890 | 0.2691 | 0.6819 | 0.978 | 4/4 | 1.016 | 0.001318 | 1.0022 | 4.193 / 4.122 | 40.55% / 41.32% | -1.04 | yes |
| cond_logit | 1.582869 | 0.7891 | 0.2682 | 0.6820 | 1.016 | 4/4 | 0.990 | 0.001307 | 1.0027 | 4.194 / 4.122 | 40.52% / 41.32% | -1.06 | yes |
| lgbm | 1.585498 | 0.7902 | 0.2671 | 0.6794 | 1.949 | 4/4 | 0.990 | 0.001479 | 1.0012 | 4.174 / 4.122 | 40.80% / 41.32% | -0.82 | yes |

**Decision: proportional.** eligible: proportional 1.582525, cond_logit 1.582869, lgbm 1.585498; floor 0.001479; proportional, cond_logit are inside the floor of each other; the pre-registered tie-break takes the simplest

P2 ridge penalty 10.0 (searched on `2024-12-06`); dropped as unidentified: `[]`; dropped as constant in this fold: `['log_share_x_rim', 'log_share_x_three', 'log_share_x_ft', 'is_C_x_rim', 'is_C_x_three', 'is_C_x_ft']`. Coefficients: `{'log_share': 0.93055, 'log_prior_share': 0.0122, 'is_G': -0.01266, 'is_F': -0.02502, 'is_C': -0.09818, 'log_share_x_scorediff': 0.04263, 'log_share_x_sec': 0.00706, 'is_C_x_scorediff': 0.01242, 'is_C_x_sec': 0.03689}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.585498], seed SD None; feature importance `{'share': 806, 'rate': 790, 'prior_rate': 893, 'position_code': 63, 'rate_rank': 102, 'score_diff': 683, 'sec_remaining': 863, 'shot_class_code': 0}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.574934 | 1.578510 | 1.598912 |
| cond_logit | 1.578524 | 1.579579 | 1.594011 |
| lgbm | 1.581979 | 1.578642 | 1.601965 |

(n transfers = 10,797)

**block** (K = 5, uniform log loss 1.609438). Train 16,997, test 17,234. Fitted shrinkage: prior `position`, m = 10 pseudo opportunities.

| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | slope | boot SE | SD ratio | players >=1 (sim / real) | top-1 (sim / real) | top-3 gap | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| proportional | 1.255963 | 0.6406 | 0.5046 | 0.8709 | 4.615 | 4/4 | 0.945 | 0.006696 | 1.0157 | 2.436 / 2.345 | 60.30% / 62.34% | -0.67 | NO (calibration, top1_top3) |
| cond_logit | 1.245607 | 0.6344 | 0.5050 | 0.8713 | 2.162 | 4/4 | 0.991 | 0.007327 | 1.0016 | 2.408 / 2.345 | 60.92% / 62.34% | -0.43 | NO (calibration) |
| lgbm | 1.250223 | 0.6367 | 0.5039 | 0.8682 | 2.216 | 4/4 | 1.005 | 0.007571 | 0.9982 | 2.392 / 2.345 | 61.40% / 62.34% | -0.35 | NO (calibration) |

**Decision: NO WINNER.** no arm passes every pre-registered gate

P2 ridge penalty 0.001 (searched on `2024-12-07`); dropped as unidentified: `[]`; dropped as constant in this fold: `['log_share_x_ft', 'is_C_x_ft']`. Coefficients: `{'log_share': 0.91187, 'log_prior_share': 0.03099, 'is_G': -0.06218, 'is_F': -0.07642, 'is_C': 0.13861, 'log_share_x_scorediff': 0.01062, 'log_share_x_sec': -0.00348, 'log_share_x_rim': 0.21739, 'log_share_x_three': -0.50065, 'is_C_x_scorediff': 0.02492, 'is_C_x_sec': -0.05106, 'is_C_x_rim': 0.01094, 'is_C_x_three': -0.73191}`.

P3 parameters {'num_leaves': 15, 'learning_rate': 0.08, 'n_estimators': 300, 'min_child_samples': 200}; seed-varied refits [1.250223], seed SD None; feature importance `{'share': 760, 'rate': 835, 'prior_rate': 881, 'position_code': 87, 'rate_rank': 99, 'score_diff': 561, 'sec_remaining': 802, 'shot_class_code': 175}`.

Transfer subset:

| arm | transfers | continuing | no prior season |
|---|---:|---:|---:|
| proportional | 1.266553 | 1.251666 | 1.251348 |
| cond_logit | 1.268135 | 1.241602 | 1.226416 |
| lgbm | 1.278244 | 1.247507 | 1.222216 |

(n transfers = 5,066)

### Binary targets

| target | base rate | arm | log loss | Brier | calib worst (pp) | resp steps | slopes (off / def) | boot SE | count (sim / real) | count SD ratio | eligible |
|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|---|
| assisted | 50.8949% | team_ridge | 0.684975 | 0.245943 | 2.538 | 4/4 | 0.779 / 0.918 | 0.000424 | 13.291 / 12.986 | 0.9196 | NO (calibration, responsiveness) |
| assisted | 50.8949% | aware_ridge | 0.562365 | 0.190793 | 6.075 | 4/4 | 0.877 / 0.915 | 0.001361 | 13.330 / 12.986 | 0.9353 | NO (calibration) |
| assisted | 50.8949% | lgbm | 0.564868 | 0.191704 | 4.149 | 4/4 | 0.958 / 0.884 | 0.001488 | 13.322 / 12.986 | 0.9524 | NO (calibration) |
| stolen | 57.3861% | team_ridge | 0.680055 | 0.243502 | 2.535 | 4/4 | 0.915 / 0.717 | 0.000594 | 6.301 / 6.489 | 0.9528 | NO (calibration, responsiveness) |
| stolen | 57.3861% | aware_ridge | 0.677667 | 0.242385 | 6.738 | 4/4 | 1.168 / 0.771 | 0.000727 | 6.327 / 6.489 | 0.9556 | NO (calibration, responsiveness) |
| stolen | 57.3861% | lgbm | 0.653014 | 0.231290 | 9.253 | 3/4 | 0.832 / 0.708 | 0.001213 | 6.423 / 6.489 | 0.9758 | NO (calibration, responsiveness) |
| blocked | 9.8955% | team_ridge | 0.320429 | 0.088745 | 1.084 | 4/4 | 0.696 / 1.111 | 0.001789 | 3.187 / 3.161 | 0.9735 | NO (responsiveness) |
| blocked | 9.8955% | aware_ridge | 0.264498 | 0.078322 | 1.033 | 4/4 | 0.920 / 1.095 | 0.001558 | 3.206 / 3.161 | 1.0050 | yes |
| blocked | 9.8955% | lgbm | 0.268812 | 0.079517 | 5.597 | 4/4 | 0.856 / 1.081 | 0.001647 | 3.200 / 3.161 | 1.0199 | NO (calibration) |

**assisted decision: NO WINNER.** no arm passes every pre-registered gate Fitted shrinkage: m_team 50, own-share prior `prior_season` m_own 10.

**stolen decision: NO WINNER.** no arm passes every pre-registered gate Fitted shrinkage: m_team 200, own-share prior `league` m_own 400.

**blocked decision: aware_ridge.** eligible: aware_ridge 0.264498; floor 0.001558; the only eligible arm Fitted shrinkage: m_team 200, own-share prior `prior_season` m_own 10.

<!-- END report_v1.md verbatim -->

### 3.6 Wall time, tests, deviations, files

- **Wall time**: 682.4 s (~11.4 minutes) end to end, including the forced
  `--rebuild` of all five population tables plus the player and team as-of
  tables (42.1 s of the total). No lever from `RESUME.md` was invoked -- the
  ~5-hour trigger was never approached.
- **Tests**: `pytest tests/test_attribution.py -q` -> **27 passed**, both
  before and after the `build_team_asof` fix.
- **Deviations from the pre-registration**: none on grid size, seed count,
  draws or bootstrap reps -- all ran at the pre-registered defaults (3-rung
  grid, 3 F1 seeds, 40 game-level draws, 200 bootstrap reps, both folds). The
  only departures from a clean run are the two found-and-fixed/found-and-flagged
  items in 3.1 and 3.2, neither of which changes what is compared.
- **Files written this session**: `src/cbb_sim/models/attribution.py` (the
  `build_team_asof` fix), `data/processed/models/attribution/team_asof_v2.parquet`
  (regenerated), `data/processed/models/attribution/{events_*_v2,asof_v2}.parquet`
  (regenerated, byte-for-byte equivalent populations), `build_report_v2.json`,
  `results_v1.json`, `attribution_params_v1.json`, `report_v1.md`,
  `train_log_v1.txt` (all under `data/processed/models/attribution/`),
  `docs/tests/attribution_team_asof_lg_defect_2026-09-10.md` (new), and this
  file.


