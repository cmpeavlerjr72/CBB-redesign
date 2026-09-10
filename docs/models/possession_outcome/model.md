# L3 POSSESSION-OUTCOME

Status: **BAKE-OFF RUN 2026-09-10 -- NO ARM ADOPTED.** Every arm failed the pre-registered
per-class calibration gate on the fold-2 test season. The failure is a level error, not a shape
error, and it is L11 reappearing one layer down. Nothing is shipped and no post-hoc correction is
applied. Numbers: `experiments.md` section 3. Features: `features.md`.

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

- **Data window.** Seasons 2022-2025. 3,433,227 modelled chances (2,996,517 first, 436,710
  continuation).
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
- **Gates, applied before any log loss is compared.** Per-class decile calibration (max absolute gap
  <= 2 pp on classes with >= 5% share) and responsiveness (the predicted class share must slope with
  the offence's own as-of rate quintile for 3PA, rim and TOV).
- **Noise floor.** 5 seed-varied refits for the tree arm; a 200-replicate game-block bootstrap for
  the linear arms. The resampling unit is the GAME, because chances inside one game share lineups,
  officials and pace, and a chance-level bootstrap would understate the SE by roughly the square
  root of the chances per game.

## 4. Winner

**There is none.** On F2 every arm's worst per-class decile miscalibration exceeds the 2 pp gate, so
the pre-registered rule eliminates all of them before log loss is consulted, and the standing rule
("no hand tuning on engine output") forbids closing the gap with a correction. The exact table is
`experiments.md` section 3.4.

What the numbers say, in relative terms:

- **The gates are doing their job in the right direction.** Every model arm passes responsiveness
  with slopes essentially equal to reality (predicted-vs-actual span ratios of 0.97, 1.02 and 0.97
  for 3PA, rim and TOV, monotone in all four quintile steps). The matchup-naive baseline fails
  responsiveness, exactly as a flat league-average model should. So the arms are matchup-specific,
  which is the harder property to get.
- **The failure is level, not shape, and it is one class.** Only `FGA_jump2` breaches the gate. On
  the lowest-loss arm its worst decile gap is 2.95 pp, of which 1.80 pp is a flat level shift present
  in every decile and 1.15 pp is residual shape -- and 1.15 pp alone would pass. `FGA_3` is the
  mirror image (-1.15 pp level). Every other class is inside 1.1 pp. The arms order chances
  correctly and put the two-point-jumper rate about a fifth of a shot per game too high.
- **It is a 2025-specific miss, not a general one.** On F1 (test 2024) every ridge and cascade arm
  passes the same calibration gate (worst gap 1.07-1.89 pp). It is the 2024 -> 2025 shot-mix step
  that the model cannot see. LightGBM fails the gate on BOTH folds, so its calibration problem is
  its own rather than the season's.
- **The pre-registered season term helps a little and fixes nothing.** Adding `season_idx` and
  `days_since_start` clears the noise floor by about 1.6x, so the block is recorded KEPT -- but it
  moves the worst gated calibration gap only from 2.60 pp to 2.13 pp, still above the 2.00 pp gate.
  A linear season index has almost nothing to extrapolate from: the two-point-jumper share in the
  three training seasons is 20.1 / 19.8 / 19.9%, essentially flat, and 2025 steps down to 18.1%. The
  trend the model needs to see is not in the training window.
- **Game state is the one block that matters.** Going from `B_plus_season` to `C_plus_state` cuts F2
  log loss by roughly 0.088, two orders of magnitude more than any other block, in every model class.
  Period, clock, score difference, bonus state and the transition proxy are not decoration.
- **The tree arm's edge over the best linear arm is real and, by the noise floor, large:** about
  0.0147 in F2 log loss against a floor of 0.00072, twenty times over. If the calibration gate were
  passed by everything, LightGBM on `C_plus_state` would win outright rather than on a tie-break. It
  also has the worst calibration of any arm on `first`, which is why it does not.
- **The noise floor is small because the test set is large.** Five seed-varied LightGBM refits
  differ by an SD of 0.00007; a 200-replicate game-block bootstrap of the best linear arm gives an
  SE of 0.00072. The larger of the two is applied, which is the conservative choice. At 761,593 test
  chances almost any real difference clears it, so the floor is separating a different fit from a
  different model rather than hiding differences.
- **The explicit interaction block is rejected.** On both linear arms the offence x defence product
  terms move F2 log loss by at most 3e-06, 240 times below the floor (`experiments.md`
  section 3.9).

## 5. Robustness check

By-state calibration of the lowest-loss arm (transition vs half-court, bonus vs no bonus, late clock
vs not) is in `experiments.md` section 3.8. Two segments are worth flagging:

- **Transition.** The rim share is under-predicted in transition by about 1.5 pp. The
  `is_transition` feature is a duration proxy (`duration_s <= 8` after a defensive rebound or
  turnover), not an observed flag, and it is measured on the chance that is being predicted -- it
  describes how the possession turned out, not what was known at its start. That is a design problem
  in the feature, not evidence about the model, and it is called out as a followup rather than
  patched.
- **Bonus.** Log loss is materially worse inside the bonus than outside it. That is expected: two of
  the six classes only exist there, so the conditional distribution is genuinely harder, and roughly
  40% of free-throw trips carry `ft_trip_ambiguous` because the feed cannot separate a two-shot
  shooting foul from a bonus trip (`docs/tests/possessions_build_2026-09-10.md` section 5). The
  label noise floors what any model can achieve on those two classes.

The `cont` population is reported separately in `experiments.md` section 3.2. Every arm fails its
calibration gate there by 6-12 pp -- three to six times worse than on `first` -- and the cause is
not the model. Building the possession layer surfaced a one-season labelling defect in the feed:
the rim share of putbacks (the attempt immediately after an offensive rebound) is 53.9 / 55.0 /
54.5% in 2022-2024, collapses to **44.5% in 2025**, and recovers to 56.3% in 2026. ESPN moved
`LayUpShot` putbacks into `TipShot` (still a rim attempt) and, in 2025 only, into `JumpShot`
labelled a two-point jumper. Full evidence: `docs/tests/possessions_build_2026-09-10.md`
section 3.1.

**2025 is the fold-2 test season, so the `cont` model is trained on clean data and graded against
relabelled data.** Its numbers are blocked on a data defect and must not be read as a comparison of
model classes. The `first` verdict is unaffected: first-chance shares move smoothly and
monotonically across the same boundary and keep moving the same way into 2026, which is the genuine
league-wide drift toward threes rather than this defect in disguise.

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
- **Nothing is adopted, and no level correction is applied.** The obvious "fix" -- shifting the class
  probabilities to match the test season's observed shares -- is precisely the post-hoc calibration
  curve `CLAUDE.md` bans, and it would be fitted on the answer. The correct next step is a model
  change (recency weighting, a season-level random effect, or a preseason refit), pre-registered
  before it is run.

## 7. Consumption from the sim

No artifact is adopted, so nothing should be wired into the engine from this run. The lowest-loss
arm is persisted as `reference_not_adopted_{population}.pkl` with `adopted=False` in its payload,
purely so the next iteration has something to diff against. When an arm does pass the gates, the
call shape is:

```python
import pickle
import numpy as np
from cbb_sim.models import possession_outcome as PO

with open("data/processed/models/possession_outcome/winner_first.pkl", "rb") as fh:
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

1. **The season-drift level miss is the blocking defect.** Candidate fixes, to be pre-registered and
   baked off before any is run: recency-weighted training (exponential decay on game date), a
   season-level random effect fitted on the training seasons and carried forward, a preseason refit
   on the most recent season only, or a target reparametrised as a deviation from the offence's own
   as-of shares (which absorbs the league level by construction). The last of these is the most
   promising and the most invasive.
2. **`is_transition` is contemporaneous with the outcome it predicts.** It is derived from the
   chance's own duration, so it is not available at the chance's start. It must be replaced by a
   pre-chance definition (the previous possession's terminal event plus the seconds elapsed since
   the change of possession) before this model is used inside the engine, and the leak test must be
   extended to cover it.
3. **`ft_trip_ambiguous` is label noise on two classes**, ~40% of free-throw trips. It bounds the
   achievable calibration on `FT_trip_shooting` and `FT_trip_bonus` and should be quantified as a
   noise floor for those classes specifically.
4. **The 2025 putback mislabelling blocks the `cont` model** (section 5). The fix is a
   pre-registered choice between collapsing `FGA_rim` and `FGA_jump2` into one two-point class for
   continuation chances, excluding 2025 from the `cont` folds and re-folding, or re-deriving the
   rim/jumper boundary from a signal stable across the break. It is not a relabelling rule invented
   after seeing the numbers.
5. **No lineup features.** Deliberately out of scope here (they exist only from 2023-24, L13) and
   the subject of L4.
6. **Defence-allowed rates are unadjusted for opponent quality.** The own ridge ratings in the same
   bundle carry the schedule adjustment, so the two are not redundant, but a properly adjusted style
   rate is a cleaner feature.
7. **CBBD feed incompleteness in 2022-2023.** 19% of games in those seasons are missing scoring
   plays from the CBBD stream (`docs/tests/possessions_build_2026-09-10.md` section 2). Those games
   are in the training window. A CBBD-side truncation flag, and a re-run excluding them, is a
   followup that could move the training distribution slightly.
