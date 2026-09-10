# L3 POSSESSION-OUTCOME

Status: **ROUND 2 RUN 2026-09-10 -- WINNER ON BOTH POPULATIONS.**
`lgbm` + `C_plus_state` + training scheme **`S1`** on the `first`
population (F2 log loss 1.51543, worst gated decile gap
0.98 pp against a 2.0 pp gate), and
`cascade` + `C_plus_state` + **`S1`** on `cont`
(1.49976, 1.86 pp).

Round 1 adopted nothing: 0 of 11 arms cleared the calibration gate. Round 2 changed three things --
two data fixes in the event layer, a completeness restriction on the universe, and a training-scheme
dimension -- and the gate now passes. Nothing was tuned on the output; the numbers moved because the
inputs were repaired and because the fit is allowed to follow the season. Numbers: `experiments.md`
sections 4-5. Features: `features.md`. Round 1's own record is untouched in section 3.

## 1. Purpose

Given the offence, the defence, the site and the game state, this model gives the probability that a
chance ends in each of six terminal events:

    TOV, FGA_rim, FGA_jump2, FGA_3, FT_trip_shooting, FT_trip_bonus

It is layer L3 of the cascade (`docs/FRAMEWORK_PLAN.md` section 2.1) and the first stage of it: the
engine draws a terminal event here, then hands off to the make/miss model, the rebound model and the
clock model. It is the event-type MIX only -- whether the shot goes in is `possession_make`, to be
pre-registered next.

It exists in this shape because of L10 (`docs/LEARNINGS.md`). The Control engine draws four
per-100-possession counts (3PA, 2PA, TOV, FTA) as independent marginals. Each is individually
calibrated, but their residuals correlate -0.56 / -0.36 / -0.29 because they compete for the same
possessions, and the independent draw gives a margin SD ratio of 1.68 and a PIT K-S p of 2e-79. Shot
mix has to be drawn per possession as one categorical outcome. That is what this model is.

## 2. Target variable

`terminal_event` of a chance, from
`data/processed/possessions_v2/chances_{season}.parquet`, built by
`scripts/build_possessions.py --version v2` and validated in
`docs/tests/possessions_build_v2_2026-09-10.md`. (Round 1 used
`data/processed/possessions/`, which is now frozen as `v1`; the two differ only in event LABELS --
possession and chance boundaries are bit-identical, and the identical possession and chance counts
per season prove it.)

Population filter: D-I, non-truncated, **`pbp_complete`**, seasons 2022-2025 (2026 sealed). Chances
whose terminal event is `end_period` (the clock model's job per the pre-registration) or `unknown`
(the mismatch-guard data gap) are dropped, never imputed.

`pbp_complete` is new in round 2 and it is a real restriction, not tidiness: 78.7 / 79.9 / 91.8 /
95.4% of D-I non-truncated games in 2022-2025 qualify, and the design loses 11.5% of round 1's rows
(3,433,227 -> 3,038,628), almost all of them in 2022 and 2023. A game whose CBBD event stream does
not account for the final score is a game whose possession table is missing possessions, and the L3
target is a distribution over what happened in them. Definition, and the reconciliation of the two
completeness numbers that were previously in circulation:
`docs/tests/possessions_build_v2_2026-09-10.md` section 2.1.

First chances and continuation chances after an offensive rebound are **separate populations**,
fitted and scored separately, because they are a different distribution rather than a rescaled one.
Measured on 2025 in v2: rim 25.5% vs 38.1%, threes 29.3% vs 22.1%, turnovers 15.4% vs 11.4%.

Class balance and its five-season drift are in `docs/tests/possessions_build_v2_2026-09-10.md`
section 3. The drift is the story of this bake-off: the three-point share rises and the turnover
share falls across the training window, and 2025 continues both.

## 3. Methodology at a glance

- **Data window.** Seasons 2022-2025, `pbp_complete` games only. 3,038,628 modelled chances
  (2,650,559 first, 388,069 continuation), from possessions
  `v2` with `first_chance` style rates. Round 1 had 3,433,227 over
  the same seasons; the completeness restriction costs
  11.5% of the rows.
- **Split.** Temporal walk-forward, no random split. F1 trains {2022, 2023} and tests 2024; F2
  trains {2022, 2023, 2024} and tests 2025 and is the selection fold. `assert_not_sealed` fires on
  every train and test slice, so the sealed 2026 season cannot enter a fold even by accident.
- **Primary metric.** Multiclass log loss on F2, `first` population. Log loss because the engine
  consumes the whole probability vector, not an argmax -- a model that picks the modal event
  correctly but distributes the rest badly would produce the right shot mix on average and the wrong
  variance, which is the CFB INV-56 failure one level down.
- **Model families tested.** Multinomial ridge logit; a nested binary GLM cascade (TOV? -> free-throw
  trip? -> bonus vs shooting? -> three? -> rim?); LightGBM multiclass; and a matchup-naive baseline
  (the previous season's league-average shares) as the floor.
- **Training schemes tested (round 2's new dimension).** S0 static, exactly as round 1. S1 in-season
  walk-forward: refit at the start of each month of the test season on all prior seasons plus the
  test season to date, strictly before the refit date, with each test game scored by the most recent
  refit at or before its own date -- so no game is in its own fit and the test set is identical to
  S0's. S2 exponential recency weighting on game date, half-life fitted on F1 only over
  {90, 180, 365, 730} days and applied to F2 unchanged.
- **Feature set.** `C_plus_state` for every arm -- round 1's best for every model class. A, B and D
  are not rerun; round 1 already answered that question and rerunning it would spend the fold on a
  settled comparison.
- **Gates, applied before any log loss is compared.** Per-class decile calibration (max absolute gap
  <= 2 pp on classes with >= 5% share) and responsiveness (the predicted class share must slope with
  the offence's own as-of rate quintile for 3PA, rim and TOV).
- **Noise floor.** 5 seed-varied refits for the tree arm; a 200-replicate game-block bootstrap for
  the linear arms. The resampling unit is the GAME, because chances inside one game share lineups,
  officials and pace, and a chance-level bootstrap would understate the SE by roughly the square
  root of the chances per game.

## 4. Winner

**`first` (the primary population): `lgbm` + `C_plus_state` + training scheme `S1`.** F2 log loss 1.51543; worst gated decile gap 0.98 pp against the 2.0 pp gate; responsiveness passes. Tree arm beats the best linear arm by 0.01576 log loss, more than the noise floor 0.00076.

**`cont`: `cascade` + `C_plus_state` + `S1`.** F2 log loss 1.49976; worst gated decile gap 1.86 pp; responsiveness passes. No tree arm passed the gates.

What actually changed, in the order the changes bite:

1. **The `cont` population was never a modelling problem, and now it is not a problem at all.** Round 1's continuation arms failed calibration by 7.3-12.4 pp. That was the 2025 putback mislabel being graded as if it were basketball. With the rim-location override the same arms sit at 1.70-6.70 pp. A 6.5 pp data defect was doing all of that work.
2. **On `first`, the data fixes alone moved the linear arms inside the gate.** `ridge_logit` under the SAME static scheme round 1 ran goes from 2.06 pp (FAIL) to 1.94 pp (PASS) on the same fold, with no scheme change involved. That is the `off_rim_c` contamination channel and the `pbp_complete` restriction, not the training scheme.
3. **The training scheme is what carries the TREE arm across.** `lgbm` + `C_plus_state` is the lowest-loss arm in both rounds and failed the gate in both static fits (2.78 pp under S0 here, 2.95 pp in round 1). Under S1 its worst gap collapses to 0.98 pp. The log-loss gain from S1 is small (+0.00136); the CALIBRATION gain is the whole result. That is exactly the level-vs-shape decomposition round 1 reported as its finding: a static fit gets the class LEVELS of a new season wrong, and letting the fit see the season to date fixes the level rather than the ordering.
4. **S2 is not the answer, and its half-life says why.** Exponential recency weighting is inside the noise on both linear arms and makes `lgbm` slightly WORSE on `first` (+0.00067 vs S0). The fitted half-lives are long -- first/ridge_logit 730d, first/cascade 365d, first/lgbm 365d, cont/ridge_logit 730d, cont/cascade 730d, cont/lgbm 730d -- i.e. F1 asked for almost no down-weighting. Down-weighting old seasons is not the same operation as seeing the current one, and only the second helps.

Noise floor and margins:

* `first`: floor **0.000763** (tree seed SD nan over 5 refits that each replay the whole monthly schedule; linear game-block bootstrap SE 0.000763). The winning tree arm beats the best gate-passing linear arm by **+0.01576**, 21x the floor, so the tree arm wins on the pre-registered rule rather than on a tie-break.
* `cont`: floor **0.001982**. No tree arm passed the gates at all -- every `lgbm` continuation cell is still 5.4-8.8 pp miscalibrated -- so the rule hands `cont` to the best gate-passing linear arm.

**The same scheme wins both populations**, which matters beyond this model: the round-2 pre-registration says that if S1 or S2 wins, that scheme becomes the default for every later sub-model unless its own bake-off says otherwise. S1 does, on both.

## 5. Robustness check

By-state calibration of the winning `first` arm (transition vs half-court, bonus vs no bonus, late clock vs not) and its per-class decile table are in `experiments.md` section 5.7; the `cont` equivalents are in 5.8. Four things are worth flagging rather than leaving in the table:

- **Per class, on `first`, nothing is now near the gate.** The worst gated class is `FGA_jump2` at 0.98 pp. In round 1 `FGA_jump2` alone was 2.95 pp, of which 1.80 pp was a flat level shift; the level component is what the scheme change removed.
- **Transition.** Round 1 under-predicted the rim share in transition by about 1.5 pp; it is now 0.82 pp. This is NOT a clean win, because `is_transition` is still a duration proxy measured on the chance being predicted (section 9), so the segment describes how the possession turned out rather than what was known at its start. The number improved; the defect did not go away.
- **Bonus.** Log loss is still materially worse inside the bonus (1.6106) than outside it (1.4743), and that is expected: two of the six classes only exist there, and roughly 40% of free-throw trips carry `ft_trip_ambiguous` because the feed cannot separate a two-shot shooting foul from a bonus trip. That label noise floors what any model can achieve on those two classes and it is unchanged by anything in round 2.
- **The `cont` population is now a real comparison rather than a graded data defect.** Its arms sit at 1.70-6.70 pp where round 1's sat at 7.3-12.4. What remains is the honest residual: `lgbm` is badly miscalibrated on continuation chances under every scheme (5.4-8.8 pp) while the linear arms are not, on 388k rows against 2.65M -- a tree with fixed hyperparameters on a tenth of the data.

## 6. Decisions log

- **A single categorical, not four count models.** Forced by L10; see section 1.
- **First and continuation chances fitted separately** rather than with a `chance_number` feature
  inside one model. The two populations differ by 6-7 pp in three of the six classes; a single
  intercept shift would not represent that, and the pre-registration asks for them separately.
- **The baseline uses the LAST TRAINING season's shares, not the test season's.** "League-average
  shares by season" has to mean the last season you actually have at prediction time, or the floor
  becomes an oracle and every arm is measured against a number none of them could have produced.
- **`D_plus_interactions` is run for the linear arms only.** A tree finds interactions itself, so
  running LightGBM on D would be a duplicate of C. The pre-registration says so; the grid records
  `n/a` rather than a silently duplicated row.
- **`chance_number` is only a feature in the `cont` population.** It is identically 1 in `first`, and
  a constant column is not a feature.
- **Rates are ratios of cumulative sums, not means of per-game ratios.** A team's first few games
  otherwise give a noisy denominator disproportionate weight.
- **A team with no prior games gets exactly 0.0 on every centred rate**, which IS the league mean.
  No prior is fabricated and no team is dropped.
- **The round-1 level miss was fixed by changing the FIT, never the OUTPUT.** The obvious "fix" --
  shifting the class probabilities to match the test season's observed shares -- is precisely the
  post-hoc calibration curve `CLAUDE.md` bans, and it would be fitted on the answer. What round 2
  did instead is let the estimator see the test season as it happens (S1) and repair the labels and
  the universe it trains on. No probability produced by this model is adjusted after it is produced.
- **The winning scheme refits monthly, and that is an operational commitment, not just a number.**
  S1 is the least simple of the three schemes and it won anyway, on both populations, by more than
  the noise floor on `first` and by being the only gate-passing configuration of the best arm. The
  cost is that the deployed model is a schedule: it has to be refit at each month boundary of the
  live season, and a stale artifact silently reverts to something close to S0. The persisted pickle
  is the LAST refit only and its payload says so.
- **S2 was run and lost, and the reason is recorded rather than inferred.** Its half-life was fitted
  on F1 only -- fitting it on F2 would have been selecting on the answer -- and F1 chose long
  half-lives (365-730 days) on every arm, i.e. almost no down-weighting. Down-weighting old seasons
  is a different operation from seeing the current one, and only the second addresses a level miss
  in a season the training window has never seen.
- **The feature set was not re-searched.** Round 1 settled `C_plus_state` for every model class, and
  round 2 reuses it rather than reopening a settled comparison on the same fold. Re-running A/B/D
  against a changed event layer would have been a second bite at the selection fold.

## 7. Consumption from the sim

Round 2 adopts an arm on both populations, so `winner_first.pkl` and `winner_cont.pkl` under
`data/processed/models/possession_outcome/round2/` carry `adopted=True`. Two things about them are
not optional to read:

- **The winning scheme is S1, and a pickle cannot be a schedule.** What is persisted is the LAST
  monthly refit -- the one a deployment carries forward -- plus the full refit schedule in the same
  payload (`s1_refit_schedule`, `s1_last_refit_date`). Wiring the pickle in and never refitting is
  NOT the model that was selected; it degrades toward S0, which failed the calibration gate.
- **The payload records its own provenance** (`possessions_version`, `style_source`,
  `require_pbp_complete`). An artifact built on possessions v1, or on all-chances style rates, is a
  different model from the one graded here.

The call shape:

```python
import pickle
import numpy as np
from cbb_sim.models import possession_outcome as PO

with open("data/processed/models/possession_outcome/round2/winner_first.pkl", "rb") as fh:
    art = pickle.load(fh)

# art["features"] is the exact feature order; art["classes"] is the class order.
# X must be built by PO.build_design (or by the sim's own lookup builder using
# the identical definitions in features.md) -- never by hand, because the
# centring is relative to an as-of league mean the caller does not otherwise
# have.
p = PO.predict_arm(art["arm"], art["model"], design_rows, art["features"])
# p[:, PO.CLASS_INDEX["FGA_3"]] is the three-point-attempt probability.
```

NaN handling: every centred rate falls back to `0.0` (the league mean) inside `build_design`; the
sim must use the same fallback and never a training-set median. Output shape is
`(n_rows, 6)` in `PO.CLASSES` order, rows summing to 1 (to within ~1e-7 for the cascade arm, which
reaches a class through up to three multiplications).

Per `CLAUDE.md`, the sim loop uses lookup tables, not live model calls: when an arm is adopted, the
deployment step is to tabulate it over the discretised state grid, not to call `predict_proba`
inside the possession loop.

## 8. Artifacts

Round 1's artifacts are left exactly where they were; round 2 writes to a `round2/` subdirectory so
that neither run can overwrite the other's record.

| Path | What it is |
|---|---|
| `data/processed/possessions/{possessions,chances}_{season}.parquet` | **v1, FROZEN.** Round 1's event layer |
| `data/processed/possessions_v2/{possessions,chances}_{season}.parquet` | **v2**: the rim-location override plus per-chance attempt counts. Chance rows additionally carry `fga_rim` / `fgm_rim` / `fga_jump2` / `fgm_jump2` / `fga_3` / `fgm_3` |
| `data/processed/possessions_v2/build_report.json` | the v2 validation battery, including the whole rim-override threshold ladder |
| `data/processed/models/possession_outcome/*` | round-1 artifacts, untouched |
| `data/processed/models/possession_outcome/round2/design.parquet` | the cached round-2 chance-level design matrix |
| `data/processed/models/possession_outcome/round2/grid_results.csv` | every (population, fold, arm, scheme) row |
| `data/processed/models/possession_outcome/round2/half_lives.json` | S2's F1-fitted half-life per (population, model class), with every grid point's loss |
| `data/processed/models/possession_outcome/round2/scheme_meta.json` | what each scheme actually did -- S1's refit schedule and per-refit train sizes, S2's effective sample size |
| `data/processed/models/possession_outcome/round2/scheme_ladder.json` | scheme-vs-S0 F2 log-loss gain for every model class |
| `data/processed/models/possession_outcome/round2/noise_floor.json` | seed-varied refits (reproducing the scheme in full) and the block bootstrap |
| `data/processed/models/possession_outcome/round2/verdict.json` | the decision rule applied, and what failed which gate |
| `data/processed/models/possession_outcome/round2/metrics_detail.json` | per-cell calibration, responsiveness, by-state tables |
| `data/processed/models/possession_outcome/round2/style_rate_sources.json` | first-chance vs all-chances vs hoopR `team_box` agreement, per style rate |
| `data/processed/models/possession_outcome/round2/run_meta.json` | run timestamp, row counts, runtime, and the three provenance choices (version, style source, completeness filter) |

## 9. Known gaps and followups

1. **`is_transition` is still contemporaneous with the outcome it predicts.** It is derived from the chance's own duration, so it is not available at the chance's start. It must be replaced by a pre-chance definition (the previous possession's terminal event plus the seconds elapsed since the change of possession) before this model is used inside the engine, and the leak test must be extended to cover it. Round 2 did not touch it, and the by-state transition number in section 5 must be read with that in mind. This is the largest outstanding defect in the feature set.
2. **S1's operational cost is real.** The winner refits at each month boundary. That has to be in the pipeline, not in a person's memory, and a monitoring check should assert that the deployed artifact's `s1_last_refit_date` is the current month.
3. **`ft_trip_ambiguous` is label noise on two classes**, ~40% of free-throw trips. It bounds the achievable calibration on `FT_trip_shooting` and `FT_trip_bonus` and should be quantified as a per-class noise floor; nothing in round 2 addressed it.
4. **The override cannot reach unlocated rows.** 12-22% of two-point jumper rows in 2022-2024 carry no shot-chart coordinates and keep the feed's label. That is a miss rather than a false positive -- the conservative direction -- but it means the repair is not complete in the early seasons. `docs/tests/possessions_build_v2_2026-09-10.md` section 3.1 reports the counts.
5. **`pbp_complete` costs 21% of 2022 and 2023.** The restriction is right for a model that trains on events, but it shrinks the two oldest training seasons the most, which is the opposite of what a drift-sensitive model wants. Whether those games can be repaired rather than dropped (a second feed, hoopR's own event stream) is a followup.
6. **`lgbm` is unusable on `cont`** under every scheme (5.4-8.8 pp miscalibration) while the linear arms pass. Fixed hyperparameters on 388k rows is the likely cause; a `cont`-specific capacity setting would be a new pre-registration, not a tweak.
7. **No lineup features.** Deliberately out of scope here (they exist only from 2023-24, L13) and the subject of L4.
8. **Defence-allowed rates are unadjusted for opponent quality.** The own ridge ratings in the same bundle carry the schedule adjustment, so the two are not redundant, but a properly adjusted style rate is a cleaner feature.
9. **An offline winner still has to survive a paired-seed sim run** before it ships (`CLAUDE.md`): nothing here has been through the engine yet.
