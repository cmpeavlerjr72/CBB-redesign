# Stage B offline table, fold 2 (box, 2026-09-30). Grading only; the PM decides.

Difference = arm minus R (negative = lower log loss = better). `floors` = difference / registered floor; `spread` = |R2 - R| (tonight's second-seed spread, same code path). Fold 1 confirmation was not run (fold-1 retrains were not in tonight's list). Arms TO (anchored) not run (engine could not serve the offset when the job list was written).


## possession_outcome `first`: multiclass log loss (floor 0.000804)

| arm | log loss | diff vs R | floors | R2 spread |
|---|---:|---:|---:|---|
| R | 1.515428 | (reference) | | |
| R2 | 1.515541 | +0.000113 | +0.14 | spread 0.000113 (0.14 floors) |
| T | 1.514592 | -0.000836 | -1.04 | spread 0.000113 (0.14 floors) |
| Topp | 1.513979 | -0.001449 | -1.80 | spread 0.000113 (0.14 floors) |
| Tfs | 1.514009 | -0.001419 | -1.76 | spread 0.000113 (0.14 floors) |

| arm | calibration_pass | worst gated gap pp | responsiveness_pass |
|---|---|---:|---|
| R | True | 0.98 | True |
| R2 | True | 0.977 | True |
| T | True | 1.472 | True |
| Topp | True | 1.322 | True |
| Tfs | True | 1.409 | True |

## possession_outcome `cont`: multiclass log loss (floor 0.001982)

| arm | log loss | diff vs R | floors | R2 spread |
|---|---:|---:|---:|---|
| R | 1.499760 | (reference) | | |
| R2 | 1.499760 | +0.000000 | +0.00 | spread 0.000000 (0.00 floors) |
| T | 1.497960 | -0.001800 | -0.91 | spread 0.000000 (0.00 floors) |
| Topp | 1.497594 | -0.002166 | -1.09 | spread 0.000000 (0.00 floors) |
| Tfs | 1.502163 | +0.002403 | +1.21 | spread 0.000000 (0.00 floors) |

| arm | calibration_pass | worst gated gap pp | responsiveness_pass |
|---|---|---:|---|
| R | True | 1.859 | True |
| R2 | True | 1.859 | True |
| T | True | 1.737 | True |
| Topp | True | 1.749 | True |
| Tfs | True | 1.927 | True |

## fg_make B1 `FGA_rim`: attempt-level log loss (floor 4.386e-05)

| arm | log loss | diff vs R | floors | R2 spread |
|---|---:|---:|---:|---|
| R | 0.667252 | (reference) | | |
| R2 | 0.667208 | -0.000044 | -1.00 | spread 0.000044 (1.00 floors) |
| T | 0.666884 | -0.000368 | -8.39 | spread 0.000044 (1.00 floors) |

| arm | calib_pass | calib worst gap |
|---|---|---:|
| R | True | 1.127 |
| R2 | True | 1.495 |
| T | True | 1.278 |

## fg_make B1 `FGA_jump2`: attempt-level log loss (floor 0.00011216)

| arm | log loss | diff vs R | floors | R2 spread |
|---|---:|---:|---:|---|
| R | 0.666385 | (reference) | | |
| R2 | 0.666273 | -0.000112 | -1.00 | spread 0.000112 (1.00 floors) |
| T | 0.666111 | -0.000274 | -2.44 | spread 0.000112 (1.00 floors) |

| arm | calib_pass | calib worst gap |
|---|---|---:|
| R | True | 1.487 |
| R2 | True | 1.44 |
| T | False | 2.04 |

## fg_make B1 `FGA_3`: attempt-level log loss (floor 1.813e-05)

| arm | log loss | diff vs R | floors | R2 spread |
|---|---:|---:|---:|---|
| R | 0.637738 | (reference) | | |
| R2 | 0.637720 | -0.000018 | -1.00 | spread 0.000018 (1.00 floors) |
| T | 0.637701 | -0.000037 | -2.03 | spread 0.000018 (1.00 floors) |

| arm | calib_pass | calib worst gap |
|---|---|---:|
| R | True | 1.817 |
| R2 | True | 1.641 |
| T | True | 1.539 |

## rebound S1_weekly (A0B0C0): three-class log loss (floor 6.7e-05)

| arm | log loss | diff vs R | floors | R2 spread |
|---|---:|---:|---:|---|
| R | 0.644522 | (reference) | | |
| R2 | 0.644642 | +0.000120 | +1.79 | spread 0.000120 (1.79 floors) |
| T | 0.643946 | -0.000576 | -8.60 | spread 0.000120 (1.79 floors) |

| arm | calib worst gap pp | calib worst class | resp_pass | team-quintile slope ratio | slope off_oreb_c | slope opp_def_dreb_c | L1 level pp | per-game MAE pp |
|---|---:|---|---|---:|---:|---:|---:|---:|
| R | 1.674 | DREB | True | 0.684 | 0.9668 | 1.0988 | -0.9415 | 4.9315 |
| R2 | 1.905 | DREB | True | 0.6843 | 0.9653 | 1.0946 | -0.9491 | 4.9334 |
| T | 1.784 | DREB | True | 1.0447 | 0.9981 | 1.0992 | -0.9029 | 4.8372 |

## Arm TO (anchor O on top of T; lane M wrapper trainers, E3 v4), fold 2, offline (added after the first table; grading only)

| sub-model | TO | diff vs R (floors) | diff vs T (floors) | other lines |
|---|---:|---|---|---|
| possession_outcome `first` log loss (floor 0.000804) | 1.514349 | -0.001079 (-1.34) | -0.000243 (-0.30) | calibration pass, worst gated gap 0.905 pp, responsiveness pass |
| possession_outcome `cont` log loss (floor 0.001982) | 1.497960 | -0.001800 (-0.91) | 0.000000 (cont takes no offset) | calibration pass, worst gated gap 1.737 pp |
| rebound log loss (floor 6.7e-05; R2 spread 1.2e-04) | 0.643672 | -0.000850 (-12.7) | -0.000274 (-4.1) | calib worst gap 1.817 pp (class OREB, pass); team-quintile slope ratio 1.0545 (T 1.0447, R 0.684); L1 held-out level -0.043 pp (T -0.903, R -0.942); resp pass; per-game MAE 4.819 pp (T 4.837, R 4.932) |

Wall time (24 processes each, under concurrent sim load): possession_outcome TO 500 s, rebound TO 179 s of fit. The TO closed-loop sim was NOT run on the box (spot reclaimed 19:45:58Z before the sample sim finished); resume command: `scripts/box_to_sim_v1.sh` on a box with the Stage B artifacts pulled (HF key `model_artifacts`; the TO artifacts were written before the reclaim and may be on HF only if the 10-minute sync loop pushed them after 19:39Z).

## Per-arm retrain wall time (box, 192 vCPU spot, n_jobs as in the launchers; waves overlapped with sims, so not clean timings)

| arm | wall |
|---|---:|
| PO R (seed 0) / R2 / T / Topp / Tfs | 405 s / 359 s / 370 s / 370 s / 386 s |
| fg_make R / R2 / T | 50 s / 55 s / 56 s |
| rebound R / R2 / T (artifacts wrapper) | 138 s / 130 s / 130 s |
| PO TO / rebound TO | 500 s / about 180 s |

Note on R2 spreads: the fg_make R2-minus-R differences equal the registered floors exactly (4.386e-5 / 1.1216e-4 / 1.813e-5), i.e. the second-seed refit reproduces the registered floor; the possession_outcome `cont` cascade is seed-independent (R2 = R).
Note on Tfs: the trainer refuses `--team-rate-table` together with `--feature-table`, so Tfs was run on a combined table built by `scripts/ops_build_tfs_table_v1.py` (adapter applied to the design, then the round-8 corrected `in_bonus` added; verified equal to adapter plus overlay on all 8 style, 4 interaction and the in_bonus columns, 0 unmatched rows).
