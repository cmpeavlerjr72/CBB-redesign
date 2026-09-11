# Possession outcome, round 4: is the season-start calibration defect the style rates' own thin-sample noise?

Round-4 bake-off, run 2026-09-11 by the possession-outcome worker. Pre-registration:
`docs/models/possession_outcome/experiments.md` section 8, committed (`4f6b237`) BEFORE the trainer
existed. Results: the same file, section 9. This document is the multi-level evidence behind that
section; it selects nothing on its own.

Grader: round 1's `score()` and round 3's `conf_window_calibration` / `responsiveness_by`, imported
unmodified. Round 4 adds ONE evidence cell -- the per-week-of-season table below -- and every arm
goes through it, the round-3 reference included. The reference is not refit: its stored predictions
(`round3/ref_pred_first_F2_seed0.npy`) are re-scored and reproduce round 3's recorded fold-2 log
loss **1.515428 to 0.00e+00**, which is the check that the added cell is additive and the two rounds
are on one scorer.

---

## 1. The defect, restated at the level it lives on

Round 3 (L35) localised the surviving calibration failure to the first weeks of the SEASON, not to
the conference boundary. The reference model's per-week decile gap on fold 2 (2025 test season,
`first` population, 742,025 chances) is monotone in the week index and clears the 2.0 pp gate only
from week 4:

| week | n | worst gated decile gap (pp) | resid TOV | resid FGA_rim | resid FGA_jump2 | resid FGA_3 | resid FT_shoot | resid FT_bonus |
|---|---|---|---|---|---|---|---|---|
| 0 | 36,499 | **6.418** | 0.166 | 0.769 | 1.169 | -1.852 | -0.160 | -0.092 |
| 1 | 40,718 | **5.216** | 0.412 | 0.307 | 1.482 | -1.652 | -0.300 | -0.249 |
| 2 | 43,658 | **3.313** | 0.138 | -0.029 | 1.715 | -1.547 | -0.014 | -0.264 |
| 3 | 43,562 | **2.751** | 0.099 | 0.916 | 0.651 | -1.466 | -0.066 | -0.135 |
| 4-7 | 111,232 | 1.369 | -0.356 | -0.373 | 0.130 | 0.551 | -0.041 | 0.090 |
| 8+ | 466,356 | 0.939 | -0.073 | -0.270 | 0.285 | 0.007 | 0.082 | -0.032 |

Residuals are predicted minus actual in percentage points; positive means the model over-predicts
that class in that bucket. No bucket is underpowered (the smallest is 36,499 chances over 258
games). The shape is the Stage-A shape reproduced through a different code path: three-point
attempts under-predicted by 1.5-1.9 pp for four weeks, mid-range over-predicted by the same order,
and both effects gone by week 4.

## 2. The mechanism, measured on the feature itself and not on the model

The as-of style rate is a ratio over the possessions a team has already played. In week 0 that is
zero or one game. Measured on the fold-2 test slice (2025, `first`), with `w = D/(D+k)` the
empirical-Bayes weight the round-4 arms apply and `D` the team's own accumulated denominator mass:

| week of season | n chances | mean w, off 3PA rate | mean w, off TOV rate | mean w, opp-def 3PA rate | SD of the RAW centred 3PA column | SD under G1 | SD under G2 | SD under G3 |
|---|---|---|---|---|---|---|---|---|
| 0 | 36,499 | **0.090** | 0.037 | 0.050 | 4.460 | 1.152 | 4.260 | 3.884 |
| 1 | 40,718 | 0.367 | 0.170 | 0.224 | **6.835** | 2.480 | 4.214 | 3.995 |
| 2 | 43,658 | 0.531 | 0.281 | 0.356 | 5.654 | 2.945 | 4.174 | 4.020 |
| 3 | 43,562 | 0.628 | 0.365 | 0.449 | 5.130 | 3.187 | 4.145 | 4.035 |
| 4 | 38,103 | 0.696 | 0.437 | 0.524 | 4.839 | 3.350 | 4.091 | 4.006 |
| 5 | 23,434 | 0.731 | 0.478 | 0.566 | 4.900 | 3.578 | 4.276 | 4.197 |
| 6 | 37,840 | 0.754 | 0.508 | 0.595 | 4.676 | 3.517 | 4.149 | 4.078 |
| 7 | 11,855 | 0.778 | 0.542 | 0.627 | 4.509 | 3.502 | 3.979 | 3.922 |
| 8 | 39,059 | 0.791 | 0.561 | 0.645 | 4.461 | 3.530 | 4.014 | 3.961 |
| 9 | 43,883 | 0.812 | 0.592 | 0.674 | 4.498 | 3.647 | 4.092 | 4.043 |

Two things in that table are the whole round-4 hypothesis.

1. **The reliability of the feature in week 0 is 9%.** The fitted `k` for the offensive three-point
   attempt rate is 207 possessions -- about three games -- so a team with one game behind it gets
   one tenth of its own observed deviation and nine tenths of the league mean. The turnover rate is
   worse: `k` = 616 possessions, about nine games, so week 0 sits at 3.7% reliability.
2. **The raw column's dispersion is largest exactly where its information content is smallest.**
   The raw centred 3PA column has SD 6.84 in week 1 against 4.46 in week 8 -- a 53% inflation that
   is sampling noise, not spread in team style. `G1` removes it by construction (SD 1.15 in week 0).
   `G2` and `G3` hold the dispersion near the late-season level (4.0-4.2 across every week) because
   they replace the missing own-sample information with the team's prior-season style rather than
   with the league mean.

The three shrinkage arms correlate 0.959 / 0.920 / 0.927 with the raw column over the whole panel,
so they are the same feature measured with different reliability, not different features.

---

## 3. What the shrinkage does to the model, on the population that serves

`first`, fold 2, `lgbm`, S1 monthly, seed 0. `G0` is round 2's bundle re-scored from round 3's
stored predictions; `G1` is the same bundle with every style rate multiplied by its own reliability.
The other three arms did not finish inside the wall clock and are NOT RUN (section 6).

| arm | log loss | overall gated gap | **weeks 0-3 gap** | non-conf gap | first-4-conf-weeks gap | worst own-quintile slope | calibration | responsiveness |
|---|---|---|---|---|---|---|---|---|
| `G0` reference | 1.515428 | 0.980 | **3.832** | 2.492 | 1.166 | 0.9473 | PASS | PASS |
| `G1` EB shrink to league mean | 1.515519 | 1.456 | **2.766** | 2.351 | 1.166 | 0.9745 | PASS | PASS |
| `G4`, `G2`, `G3` | NOT RUN | | | | | | | |

Log loss is 0.000091 WORSE under `G1` against a floor of 0.000804, so the primary metric is a tie.
The weeks-0-3 gain is 1.066 pp against a pre-registered 0.25 pp threshold, and the non-conference
segment moves the same way (0.141 pp, inside the threshold, so it neither wins nor disqualifies).

## 4. Per week of season -- and the reason this is a PARTIAL win

| week | n | `G0` gap (pp) | `G1` gap (pp) | change | `G0` resid FGA_3 | `G1` resid FGA_3 | `G0` resid FGA_jump2 | `G1` resid FGA_jump2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 36,499 | 6.418 | **4.867** | -1.551 | -1.852 | -1.917 | 1.169 | 1.199 |
| 1 | 40,718 | 5.216 | **4.054** | -1.162 | -1.652 | -1.646 | 1.482 | 1.483 |
| 2 | 43,658 | 3.313 | **2.942** | -0.371 | -1.547 | -1.643 | 1.715 | 1.762 |
| 3 | 43,562 | 2.751 | **2.398** | -0.353 | -1.466 | -1.460 | 0.651 | 0.566 |
| 4-7 | 111,232 | 1.369 | **2.692** | **+1.323** | 0.551 | 0.552 | 0.130 | 0.170 |
| 8+ | 466,356 | 0.939 | **1.141** | +0.202 | 0.007 | 0.039 | 0.285 | 0.249 |

No bucket is underpowered. Two readings a reader should not have to derive:

1. **`G1` improves every week of the segment it was pre-registered against, and the improvement is
   ordered by the defect**: -1.55 pp in week 0, -1.16 in week 1, -0.37 and -0.35 in weeks 2 and 3.
   It is largest exactly where the reliability weight is smallest (w = 0.090 in week 0).
2. **It pays for that in weeks 4-7, and it pays more than it buys there**: 1.369 -> 2.692 pp, which
   crosses the 2.0 pp gate in a bucket where the reference passes. The overall gated gap rises
   0.980 -> 1.456 pp for the same reason. **`G1` redistributes calibration error across the season
   rather than removing it.** The pre-registered no-shuffling clause names the two DECISION segments
   (weeks 0-3 and non-conference) and `G1` clears it on both; weeks 4-7 is an EVIDENCE cell, so the
   letter of the rule adopts `G1` and this table is why the recommendation below is nevertheless
   VALIDATED-PENDING and not SHIP.

The mechanism is visible in section 2's dispersion table. `G1` collapses the feature's dispersion to
1.15 in week 0 and lets it climb back to 3.65 by week 9, so the column the tree is fitted on changes
scale across the season; the monthly refit sees a different feature in November than in February.
`G2` and `G3` hold dispersion at 4.0-4.2 in every week by substituting prior-season signal rather
than the league mean, which is the arm that should not have this failure mode -- and those are
precisely the cells the clock did not reach. That is a prediction, labelled as one.

## 5. Per team quintile (Decision 8 responsiveness)

Quintiles of the offence team's own as-of driver, fold 2, `first`, `lgbm`. Predicted and actual
class rates in percentage points.

| arm | driver | class | q1 | q2 | q3 | q4 | q5 | slope ratio | steps | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| `G0` | off_3pa_c | FGA_3 | 24.90 / 24.97 | 27.52 / 27.60 | 28.92 / 29.28 | 31.40 / 31.68 | 33.76 / 34.33 | 0.9473 | 4/4 | PASS |
| `G1` | off_3pa_c | FGA_3 | 24.75 / 24.97 | 27.38 / 27.60 | 29.01 / 29.28 | 31.54 / 31.68 | 33.87 / 34.33 | **0.9745** | 4/4 | PASS |
| `G0` | off_rim_c | FGA_rim | 22.56 / 22.64 | 24.38 / 24.53 | 25.51 / 25.62 | 26.91 / 26.98 | 28.53 / 28.72 | 0.9819 | 4/4 | PASS |
| `G1` | off_rim_c | FGA_rim | 22.48 / 22.64 | 24.37 / 24.53 | 25.48 / 25.62 | 26.96 / 26.98 | 28.69 / 28.72 | **1.0201** | 4/4 | PASS |
| `G0` | off_tov_c | TOV | 14.21 / 14.25 | 14.73 / 14.78 | 15.49 / 15.48 | 16.01 / 16.16 | 16.98 / 17.01 | 1.0062 | 4/4 | PASS |
| `G1` | off_tov_c | TOV | 14.17 / 14.25 | 14.66 / 14.78 | 15.47 / 15.48 | 16.04 / 16.16 | 16.98 / 17.01 | 1.0182 | 4/4 | PASS |

Every cell is `pred / act`. Shrinking a feature does NOT flatten the model against its own driver --
the standing worry with any shrinkage -- it moves two of three slopes closer to 1.0 and the third
from 1.006 to 1.018. The same effect is larger on the cascade probe, where `G0`'s slopes 0.919 /
0.869 / 0.889 become `G1`'s 0.977 / 0.966 / 0.950. The model is not less matchup-specific under
shrinkage; it is slightly more so, because the noise it was reacting to in the raw column was not
matchup signal.

## 6. Per possession-outcome class, and the `cont` population

The `cont` ladder is the one that COMPLETED, on both folds and all five arms, so it is the only
place round 4 can read a full ladder. Fold 2, `cascade`:

| arm | log loss | gain vs ref | overall gap | weeks 0-3 gap | non-conf gap | worst quintile slope | gates |
|---|---|---|---|---|---|---|---|
| `G0` reference | 1.499760 | 0.0 | 1.859 | 2.849 | 2.586 | 0.7711 | PASS |
| `G1` | 1.498952 | +0.000808 | 1.819 | 2.884 | **2.900** | 0.9010 | PASS, but DISQUALIFIED |
| `G2` | 1.498802 | +0.000958 | 1.692 | 2.760 | **2.249** | 0.7907 | PASS -- **winner** |
| `G3` | 1.498677 | +0.001083 | **2.422** | 2.422 | 2.175 | 0.8425 | PASS |
| `G4` | 1.499739 | +0.000021 | **2.010** | 2.727 | 2.542 | 0.7705 | **FAIL** (calibration) |

Floor 0.001982, so no arm wins on log loss; `G2` and `G3` win on the non-conference segment
(+0.337 and +0.411 pp against a 0.25 pp threshold) and `G2` is the simpler of the two inside the
floor of the best. `G1` is disqualified by the pre-registered no-shuffling clause: it gives back
0.314 pp on non-conference to buy 0.035 pp in weeks 0-3. `G4` -- the reliability counters as
features rather than a shrinkage -- is the only arm in the round to FAIL a gate.

**The two populations disagree, and the disagreement is informative, not noise.** On `cont` the arms
that shrink toward the team's own prior season (`G2`, `G3`) beat the arm that shrinks toward the
league mean (`G1`), and `G1` there is actively harmful on the non-conference segment. On `first`
only `G1` has been measured. The natural reading -- that `G2` is the arm to run next on the tree --
is a hypothesis until the cell runs, and it is the first item in section 6 of `experiments.md` 9.7.

## 7. Leak test

Every column round 4 adds clears the standing INV-45 change-form gate. Worst as-joined
|corr| = 0.061 (`def_n_prior_g`) against a 0.15 gate; the shrunk style columns sit at 0.006-0.048,
alongside round 2's own raw columns at 0.006-0.021 measured in the same run. `off_n_prior_g` is a
counter that increments by exactly one per game, so its change form is a constant and the
correlation is undefined; it is reported as `n/a` rather than as a pass, and its level-form
correlation with own-game margin is 0.016. The level-form reading flags `off_tov_c`, `off_tov_g2`
and `off_tov_g3` exactly as round 3's run flagged `off_tov_c` -- an inherited property of the
turnover column already accepted as the reference, not something the shrinkage introduces.

## 8. Verdict

* On the SELECTION population `first`, `G1` beats the reference on the pre-registered weeks-0-3
  decision cell by 1.066 pp, ties on log loss inside the floor, and improves Decision 8
  responsiveness. Under the pre-registered rule it is ADOPTED.
* The adoption is over a ladder of TWO. `G4`, `G2`, `G3` and the second-seed floor cell on `first`
  are NOT RUN.
* The multi-level evidence shows `G1` moves calibration error from weeks 0-3 into weeks 4-7 and
  across the 2.0 pp gate there. Recommendation to the PM: record `G1` as
  VALIDATED-PENDING-SHIP-ACTION, do NOT move the engine default, and run `G2` and `G4` on the tree
  first -- `cont`'s complete ladder prefers `G2`, and `G2` is the arm whose construction predicts no
  weeks-4-7 cost.
* The two tree ALIGNMENT cells remain NOT RUN for a second round. Round 4's contribution to
  Decision 9 is that round 3's marginal `cont` alignment adoption does not reproduce under a refit
  of the identical cells (0.237 pp against the 0.25 pp threshold, from a 0.023 pp
  decile-boundary shift in the reference). Reported, not decided.
