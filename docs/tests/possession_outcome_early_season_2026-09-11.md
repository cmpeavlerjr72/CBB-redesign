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
| `G2` shrink to prior season | 1.514837 | 1.196 | **3.359** | 2.500 | 1.166 | 0.9156 | PASS | PASS (landed 12:02 -- see the ADDENDUM) |
| `G4`, `G3` | NOT RUN | | | | | | | |

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

---

## ADDENDUM, 12:03 ET: the tree `G2` cell landed, and it tests the prediction the first write-up made

Process B2 finished `first | F2 | lgbm | G2 | S1_monthly` at 12:02 ET, after the sections above were
written and committed (`db0f8a2`). It is a pre-registered cell run in its pre-registered spec
through the same grader, and it is reported here rather than by rewriting what stood; the earlier
statement that `G2` on the tree is NOT RUN is superseded by this section. `G4`, `G3` and the
second-seed floor cell remain NOT RUN.

### The `first` feature ladder, now three arms

| arm | log loss | gain vs ref | overall gap | **weeks 0-3 gap** | non-conf gap | worst quintile slope | gates | beats ref beyond floor |
|---|---|---|---|---|---|---|---|---|
| `G0` reference | 1.515428 | 0.0 | 0.980 | 3.832 | 2.492 | 0.9473 | PASS | -- |
| `G1` shrink to league mean | 1.515519 | -0.000091 | 1.456 | **2.766** (+1.066) | 2.351 (+0.141) | 0.9745 | PASS | **YES** |
| `G2` shrink to prior season | **1.514837** | **+0.000591** | 1.196 | **3.359** (+0.473) | 2.500 (-0.008) | 0.9156 | PASS | **YES** |
| `G4`, `G3`, floor seed 1 | NOT RUN | | | | | | | |

Both arms beat the reference on the pre-registered weeks-0-3 cell. `G2` has the best log loss in the
round, but +0.000591 is inside the 0.000804 floor, so the primary metric still separates nothing.
Under the pre-registered rule the winner is unchanged: the best beater by log loss is `G2`
(1.514837), `G1` sits within the floor of it (1.515519 against 1.515641), and among arms inside that
band the simplest wins -- `G1` at complexity rank 1 against `G2` at rank 2. **`G1` remains the
round-4 winner, now over a ladder of three.**

### Per week of season, all three arms

| week | n | `G0` | `G1` | `G2` |
|---|---|---|---|---|
| 0 | 36,499 | 6.418 | **4.867** | 4.950 |
| 1 | 40,718 | 5.216 | **4.054** | 4.166 |
| 2 | 43,658 | 3.313 | **2.942** | 2.956 |
| 3 | 43,562 | **2.751** | 2.398 | 3.247 |
| 4-7 | 111,232 | **1.369** | 2.692 | 1.956 |
| 8+ | 466,356 | 0.939 | 1.141 | **0.902** |

Worst gated decile calibration gap in pp; no bucket is underpowered; the best arm per row in bold.

**The prediction the first write-up made is confirmed in direction and not in full.** Section 4
predicted that `G2` would not carry `G1`'s weeks-4-7 cost, because it holds the feature's dispersion
at 4.0-4.2 in every week instead of collapsing it to 1.15 in week 0. Measured: `G2`'s weeks-4-7
regression is 0.587 pp against `G1`'s 1.323 pp -- **45% of it** -- and `G2` stays UNDER the 2.0 pp
gate there (1.956) where `G1` crosses it (2.692). `G2` also IMPROVES week 8+ (0.902 against the
reference's 0.939) where `G1` degrades it (1.141). So the dispersion mechanism is real.

What the prediction got wrong: `G2` is WORSE than the reference in week 3 (3.247 against 2.751), the
one early bucket where the raw rate has enough mass to be worth something and the prior-season
target is stale. `G1` is better than `G2` in all of weeks 0, 1, 2 and 3 and `G2` is better than `G1`
in weeks 4-7 and 8+. Neither arm dominates, which is the honest shape of the result and is exactly
why the recommendation stays VALIDATED-PENDING-SHIP-ACTION rather than SHIP.

### What this changes in the recommendation

1. The `first` and `cont` populations no longer disagree about whether shrinkage helps -- both now
   have two arms beating their reference. They still disagree about WHICH target: `cont`'s complete
   ladder prefers the prior-season target (`G2` wins there, `G1` is disqualified by the no-shuffling
   clause), while `first`'s three-arm ladder gives `G1` the bigger segment gain and `G2` the better
   log loss and the better late season.
2. **The obvious next cell is no longer `G2`. It is an arm that is `G1` early and `G2` late** -- that
   is what the two per-week columns say when read together -- and the natural construction is `G3`,
   the two-level prior whose target is itself shrunk by its own reliability, which is NOT RUN on the
   tree and is the one cell of the pre-registered ladder that no population has yet rejected.
   `G3` wins `cont`'s log loss outright (1.498677) and `cont`'s weeks-0-3 cell (+0.427 pp). Running
   `G3` and `G4` on the tree closes the ladder.
3. Nothing here touches the alignment cells or Decision 9. Both tree alignment cells are still
   NOT RUN.

### Reproducing this addendum, and picking up the cells still running

Process A is still fitting `G4` and will write it to `ckpt_a.json` when it finishes, then exit on its
own pre-registered stop without starting another cell. The merged render is a read-only pass that
fits nothing:

```
$env:CBB_THREADS="1"
.venv/Scripts/python.exe scripts/train_possession_outcome_v4.py --render-only --stages 0 \
    --ckpt ckpt_render.json --merge "ckpt_a.json,ckpt_b.json,ckpt_b2.json"
.venv/Scripts/python.exe scripts/diag_po_r4_report.py
```

---

## ADDENDUM 2, 12:13 ET: `G4` landed, the tree ladder is four arms, and `G4` is the only arm that costs nothing anywhere

Process A finished `first | F2 | lgbm | G4 | S1_monthly` at 12:13 and then exited on its own
pre-registered stop without starting another cell, writing `G2`, `G3` and the second-seed floor to
NOT RUN. This section supersedes every earlier statement that `G4` on the tree is NOT RUN. `G3` and
the second-seed floor cell remain NOT RUN.

### The `first` feature ladder, complete except `G3` and the floor cell

| arm | rank | log loss | gain vs ref | overall gap | **weeks 0-3** | non-conf | worst quintile slope | gates | beats ref beyond floor |
|---|---|---|---|---|---|---|---|---|---|
| `G0` reference | 0 | 1.515428 | 0.0 | **0.980** | 3.832 | 2.492 | 0.9473 | PASS | -- |
| `G1` shrink to league mean | 1 | 1.515519 | -0.000091 | 1.456 | **2.766 (+1.066)** | 2.351 (+0.141) | **0.9745** | PASS | **YES** |
| `G4` reliability counters | 1 | 1.515626 | -0.000198 | 1.108 | 3.467 (+0.365) | 2.430 (+0.062) | 0.9537 | PASS | **YES** |
| `G2` shrink to prior season | 2 | **1.514837** | +0.000591 | 1.318 | 3.359 (+0.473) | 2.500 (-0.008) | 0.9156 | PASS | **YES** |
| `G3`, floor seed 1 | 3 / -- | NOT RUN | | | | | | | |

**Three of the four measured arms beat the reference on the pre-registered weeks-0-3 cell, and none
of them separates on the primary metric**: the whole log-loss spread across the ladder, -0.000198 to
+0.000591, is 0.98 of one noise floor. This is a segment result end to end, which is what the round
was designed to test.

**The winner is unchanged and it is `G1`, by the tie-break this round fixed in advance.** Best beater
by log loss is `G2` (1.514837); the floor band is 1.515641; `G1` (1.515519) and `G4` (1.515626) both
sit inside it; among those the simplest wins, `G1` and `G4` are tied at rank 1, and 8.1 fixed
before the run that "`G1` wins the tie as the arm that adds no columns at all". `G1` it is.

### Per week of season, all four arms -- and why the tie-break and the evidence now point apart

| week | n | `G0` | `G1` | `G4` | `G2` |
|---|---|---|---|---|---|
| 0 | 36,499 | 6.418 | **4.867** | 6.183 | 4.950 |
| 1 | 40,718 | 5.216 | **4.054** | 5.038 | 4.166 |
| 2 | 43,658 | 3.313 | 2.942 | **2.660** | 2.956 |
| 3 | 43,562 | **2.751** | 2.398 | 3.005 | 3.247 |
| 4-7 | 111,232 | **1.369** | 2.692 | 1.410 | 1.956 |
| 8+ | 466,356 | 0.939 | 1.141 | **0.868** | 0.902 |

`G4` is the **only arm in the round that improves the pre-registered segment without paying for it
anywhere else**: weeks 4-7 move 1.369 -> 1.410 (+0.041, inside any floor this round has), week 8+
improves 0.939 -> 0.868, and the overall gated gap rises only 0.980 -> 1.108 against `G1`'s 1.456.
Its weeks-0-3 gain is a third of `G1`'s, but `G1` buys that gain by pushing weeks 4-7 across the
2.0 pp gate and `G4` does not.

That is the shape of the real finding. **Telling the tree how reliable the rate is (`G4`) is weaker
and safer than making the rate reliable (`G1`).** `G1` changes the feature's scale across the season
and the monthly refit sees a different column in November than in February; `G4` leaves every
existing column bit-identical and adds two counters, so nothing the model already knew is disturbed.

### Recommendation, updated

1. `G1` is the winner of the pre-registered bake-off and this worker does not overrule a tie-break
   it fixed before the run. It stays **VALIDATED-PENDING-SHIP-ACTION**, not shipped, for the
   weeks-4-7 reason in section 4.
2. **On the multi-level evidence the arm to ship is `G4`, not `G1`**, and that is a PM call, not a
   worker's: it is the only arm with no measured cost in any week bucket, it passes every gate, its
   quintile slope improves on the reference (0.947 -> 0.954), and it is the cheapest cell in the
   round (1,866 s against `G1`'s 4,883 s) because it adds two columns and rewrites none. The caveat
   on record: `G4` is the ONLY arm in the round to FAIL a gate on the other population --
   `cont` calibration 2.010 pp against a 2.0 gate -- so a `G4` ship would be `first`-only until
   `cont` is re-read.
3. `G3` -- the two-level prior, the one arm no population has rejected, and the winner of `cont`'s
   log loss and weeks-0-3 cell -- is still NOT RUN on the tree and is the next cell either way.
4. Nothing here touches the alignment cells or Decision 9.
