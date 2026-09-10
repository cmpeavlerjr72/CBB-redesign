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

## 2. Run state -- **PARTIAL, run interrupted 2026-09-10 16:50 ET**

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
